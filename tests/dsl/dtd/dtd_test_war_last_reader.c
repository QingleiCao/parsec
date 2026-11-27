/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* A writer inserted right behind the last reader of a tile must wait for it.
 *
 * DTD holds a writer back while the copy it is about to overwrite has readers.
 * When a task completes, the walk over its successors counts each reader it
 * releases. A reader that is the last task inserted on the tile so far is
 * special: the walk also tells the tile that it has no live user any more, and
 * from then on a task inserted on the tile is released by its own insertion.
 * The count of that reader is then the only thing between it and a writer
 * inserted behind it, so it has to be taken before the tile is told.
 *
 * Per tile and round the test inserts a writer, one reader, and a second
 * writer. To make the second writer arrive while the first one completes, the
 * inserting thread waits until the body of the first writer has run, and
 * inserts the second writer at once. A few tiles are kept in flight so that
 * other threads are awake to take the second writer. The bodies check:
 *
 *   - the reader sees the value of the first writer,
 *   - the second writer starts after the reader has finished,
 *
 * and at the end every tile holds its final value and no copy has readers
 * left.
 *
 * A writer overtakes the reader only if it is inserted, taken by another
 * thread and started while the thread that walks the successors is between
 * the two steps, which are a few instructions apart: that thread has to be
 * held up right there, as an interrupt or the scheduler of a busy machine
 * does now and then. On a quiet machine this does not happen often enough to
 * be seen. Where it can (Linux), the test therefore holds its own worker
 * threads up: a timer signals each of them every HOLD_UP_PERIOD_US or so, and
 * the handler keeps the thread busy for HOLD_UP_LENGTH_US, wherever it was.
 * Nothing in the runtime is changed for this.
 *
 * The pattern is repeated many times. The test can pass on a runtime that has
 * the defect, and it does not fail on one that does not.
 *
 * usage: dtd_test_war_last_reader [cores [tiles [rounds [tiles in flight
 *                                 [microseconds between two hold-ups, 0 for none]]]]]
 */

#include "parsec/runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#if defined(__linux__)
#include <signal.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

#include "tests/tests_data.h"
#include "tests/tests_runtime.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#define DEFAULT_NB_TILES        1024
#define DEFAULT_NB_ROUNDS       200
#define DEFAULT_IN_FLIGHT       2
#define MAX_IN_FLIGHT           64
/* How long the inserting thread waits for the first writer of a tile before it
 * inserts the second one anyway, and how many such waits in a row it accepts
 * before it stops waiting at all (a machine too busy for the test to say much). */
#define WAIT_LIMIT_NS           (20 * 1000 * 1000)
#define MAX_WAITS_EXPIRED       8
/* The test takes a second or two on a machine of its own. Where it has to
 * share its cores it starts no new sequence after this long. */
#define TIME_LIMIT_NS           ((int64_t)10 * 1000 * 1000 * 1000)
/* Each worker thread is held up for HOLD_UP_LENGTH_US every HOLD_UP_PERIOD_US,
 * plus HOLD_UP_PERIOD_STEP_US from one thread to the next */
#define HOLD_UP_PERIOD_US       25
#define HOLD_UP_PERIOD_STEP_US  7
#define HOLD_UP_LENGTH_US       20

/* What the bodies have done on a tile, each on its own cache line. */
typedef struct {
    _Atomic int writes;          /* writer bodies that have finished */
    _Atomic int reads;           /* reader bodies that have finished */
    char padding[64 - 2 * sizeof(_Atomic int)];
} tile_progress_t;

/* The tiles are dealt to the slots in turn. A slot takes its tiles one after
 * the other, round after round, and holds one of them at a time: the one whose
 * first writer and reader are inserted and whose second writer is not. A tile
 * is therefore never given its next round before the current one is complete.
 */
typedef struct {
    int     nb_tiles;            /* tiles of this slot */
    int64_t next;                /* next (tile, round) of this slot to start */
    int     tile, round;         /* the one in flight, tile is -1 if none */
    int     polls;
    int64_t since;
} slot_t;

static tile_progress_t *progress;
static _Atomic int readers_saw_later_value;
static _Atomic int readers_saw_earlier_value;
static _Atomic int writers_ran_before_reader;

static int64_t
now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

/* Holding the worker threads up.
 *
 * The first body a worker thread runs asks for a timer that signals this very
 * thread. The handler does nothing but wait: for the runtime it is as if the
 * thread had lost its core for a moment at an arbitrary instruction. The
 * thread that inserts the tasks is left alone.
 */
#if defined(__linux__) && defined(SIGEV_THREAD_ID) && defined(SYS_gettid)
#if !defined(sigev_notify_thread_id)
#define sigev_notify_thread_id _sigev_un._tid
#endif
#define HOLD_UP_SIGNAL          (SIGRTMIN + 4)

static int               hold_up_period_us = HOLD_UP_PERIOD_US;
static timer_t          *hold_up_timers;
static int               hold_up_max_timers;
static _Atomic int       hold_up_nb_timers;
static _Atomic int64_t   hold_up_count;
static __thread int      hold_up_asked;

static void
hold_up_handler(int signal_number)
{
    (void)signal_number;
    int saved_errno = errno;
    int64_t until = now_ns() + (int64_t)HOLD_UP_LENGTH_US * 1000;
    atomic_fetch_add(&hold_up_count, 1);
    while( now_ns() < until ) /* nothing */;
    errno = saved_errno;
}

static void
hold_ups_prepare(int nb_threads)
{
    struct sigaction action;
    if( hold_up_period_us <= 0 ) return;
    memset(&action, 0, sizeof(action));
    action.sa_handler = hold_up_handler;
    action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask);
    if( (HOLD_UP_SIGNAL > SIGRTMAX) || (0 != sigaction(HOLD_UP_SIGNAL, &action, NULL)) ) {
        hold_up_period_us = 0;
        return;
    }
    hold_up_timers = (timer_t *)calloc(nb_threads, sizeof(timer_t));
    hold_up_max_timers = (NULL != hold_up_timers) ? nb_threads : 0;
}

/* Called by the bodies */
static void
hold_ups_for_this_thread(parsec_execution_stream_t *es)
{
    struct sigevent event;
    struct itimerspec period;
    timer_t timer;
    int index;

    if( hold_up_asked ) return;
    hold_up_asked = 1;
    if( (hold_up_period_us <= 0) || (0 == es->th_id) ) return;

    memset(&event, 0, sizeof(event));
    event.sigev_notify = SIGEV_THREAD_ID;
    event.sigev_signo = HOLD_UP_SIGNAL;
    event.sigev_notify_thread_id = (pid_t)syscall(SYS_gettid);
    if( 0 != timer_create(CLOCK_MONOTONIC, &event, &timer) ) return;
    index = atomic_fetch_add(&hold_up_nb_timers, 1);
    if( index >= hold_up_max_timers ) {
        timer_delete(timer);
        return;
    }
    hold_up_timers[index] = timer;
    /* A different period for each thread: with the same one, threads that
     * happen to be held up together stay so for the whole run, and then no
     * thread is there to take the writer while another one is held up. */
    period.it_interval.tv_sec = 0;
    period.it_interval.tv_nsec = (long)(hold_up_period_us + HOLD_UP_PERIOD_STEP_US * index) * 1000;
    period.it_value = period.it_interval;
    timer_settime(timer, 0, &period, NULL);
}

/* Called once the tasks are done, before the worker threads go away */
static int64_t
hold_ups_stop(void)
{
    int nb_timers = atomic_load(&hold_up_nb_timers);
    if( nb_timers > hold_up_max_timers ) nb_timers = hold_up_max_timers;
    for( int i = 0; i < nb_timers; i++ ) {
        timer_delete(hold_up_timers[i]);
    }
    atomic_store(&hold_up_nb_timers, 0);
    hold_up_max_timers = 0;
    free(hold_up_timers);
    hold_up_timers = NULL;
    return atomic_load(&hold_up_count);
}
#else
static int hold_up_period_us = 0;
static void hold_ups_prepare(int nb_threads) { (void)nb_threads; }
static void hold_ups_for_this_thread(parsec_execution_stream_t *es) { (void)es; }
static int64_t hold_ups_stop(void) { return 0; }
#endif

static int
body_first_writer(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int *data, tile;
    hold_ups_for_this_thread(es);
    parsec_dtd_unpack_args(this_task, &data, &tile);
    *data += 1;
    /* The last thing the body does: the completion of this task follows. */
    atomic_fetch_add(&progress[tile].writes, 1);
    return PARSEC_HOOK_RETURN_DONE;
}

static int
body_reader(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int *data, tile, expected_value;
    hold_ups_for_this_thread(es);
    parsec_dtd_unpack_args(this_task, &data, &tile, &expected_value);
    int value = *data;
    if( value > expected_value ) {
        atomic_fetch_add(&readers_saw_later_value, 1);
    } else if( value < expected_value ) {
        atomic_fetch_add(&readers_saw_earlier_value, 1);
    }
    atomic_fetch_add(&progress[tile].reads, 1);
    return PARSEC_HOOK_RETURN_DONE;
}

static int
body_second_writer(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int *data, tile, expected_reads;
    hold_ups_for_this_thread(es);
    parsec_dtd_unpack_args(this_task, &data, &tile, &expected_reads);
    if( atomic_load(&progress[tile].reads) < expected_reads ) {
        atomic_fetch_add(&writers_ran_before_reader, 1);
    }
    *data += 1;
    atomic_fetch_add(&progress[tile].writes, 1);
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int rank, world, cores = -1, rc, arena;
    int nb_tiles = DEFAULT_NB_TILES, nb_rounds = DEFAULT_NB_ROUNDS, in_flight = DEFAULT_IN_FLIGHT;

    if( argc > 1 ) cores = atoi(argv[1]);
    if( argc > 2 ) nb_tiles = atoi(argv[2]);
    if( argc > 3 ) nb_rounds = atoi(argv[3]);
    if( argc > 4 ) in_flight = atoi(argv[4]);
    if( argc > 5 ) hold_up_period_us = atoi(argv[5]);
    if( nb_tiles < 1 ) nb_tiles = 1;
    if( nb_rounds < 1 ) nb_rounds = 1;
    if( in_flight < 1 ) in_flight = 1;
    if( in_flight > MAX_IN_FLIGHT ) in_flight = MAX_IN_FLIGHT;
    if( in_flight > nb_tiles ) in_flight = nb_tiles;

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    if( world > 1 ) {
        parsec_output(0, "DTD last reader is single process only, skipping\n");
        rc = parsec_tests_context_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
        return 0;
    }

    /* With a single thread nothing completes while tasks are inserted: there is
     * no first writer to wait for, and nothing for the test to see. */
    int nb_threads = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_CORES);
    int wait_for_first_writer = (nb_threads > 1);
    hold_ups_prepare(nb_threads);

    parsec_taskpool_t *taskpool = parsec_dtd_taskpool_new();
    parsec_arena_datatype_t *arena_datatype =
        parsec_matrix_adt_new_rect(parsec_datatype_int32_t, 1, 1, 1);
    parsec_dtd_attach_arena_datatype(parsec, arena_datatype, &arena);

    parsec_tiled_matrix_t *matrix = create_and_distribute_data(rank, world, 1, nb_tiles);
    memset(((parsec_matrix_block_cyclic_t *)matrix)->mat, 0,
           (size_t)matrix->nb_local_tiles * (size_t)matrix->bsiz *
           (size_t)parsec_datadist_getsizeoftype(matrix->mtype));
    parsec_data_collection_t *collection = (parsec_data_collection_t *)matrix;
    parsec_data_collection_set_key(collection, "A");
    parsec_dtd_data_collection_init(collection);

    progress = (tile_progress_t *)calloc(nb_tiles, sizeof(tile_progress_t));
    parsec_data_copy_t **host_copy = (parsec_data_copy_t **)calloc(nb_tiles, sizeof(parsec_data_copy_t *));
    for( int tile = 0; tile < nb_tiles; tile++ ) {
        parsec_data_t *data = collection->data_of_key(collection, collection->data_key(collection, tile, 0));
        host_copy[tile] = data->device_copies[0];
    }

    rc = parsec_context_add_taskpool(parsec, taskpool);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

#define TILE_OF(tile_index) \
    parsec_dtd_tile_of(collection, collection->data_key(collection, (tile_index), 0))

    slot_t slot[MAX_IN_FLIGHT];
    int64_t nb_sequences = 0, waits_expired = 0, started_at = now_ns();
    int slots_busy = in_flight, waits_expired_in_a_row = 0, out_of_time = 0;
    int *rounds_of_tile = (int *)calloc(nb_tiles, sizeof(int));
    for( int s = 0; s < in_flight; s++ ) {
        slot[s].nb_tiles = (nb_tiles - s + in_flight - 1) / in_flight;
        slot[s].next = 0;
        slot[s].tile = -1;
    }

    while( slots_busy > 0 ) {
        for( int s = 0; s < in_flight; s++ ) {
            if( -1 != slot[s].tile ) {
                int tile = slot[s].tile;
                if( wait_for_first_writer &&
                    (atomic_load(&progress[tile].writes) < 2 * slot[s].round + 1) ) {
                    /* Look at the clock once in a while only, to stay close to the writer */
                    if( (0 != (++slot[s].polls & 0xff)) ||
                        ((now_ns() - slot[s].since) < WAIT_LIMIT_NS) ) {
                        continue;
                    }
                    waits_expired++;
                    if( ++waits_expired_in_a_row >= MAX_WAITS_EXPIRED ) {
                        wait_for_first_writer = 0;
                    }
                } else {
                    waits_expired_in_a_row = 0;
                }
                int expected_reads = slot[s].round + 1;
                parsec_dtd_insert_task(taskpool, body_second_writer, 0, PARSEC_DEV_CPU, "Second_Writer",
                                       PASSED_BY_REF, TILE_OF(tile), PARSEC_INOUT | arena | PARSEC_AFFINITY,
                                       sizeof(int), &tile, PARSEC_VALUE,
                                       sizeof(int), &expected_reads, PARSEC_VALUE,
                                       PARSEC_DTD_ARG_END);
                slot[s].tile = -1;
            }
            if( slot[s].next < 0 ) {
                continue;  /* this slot is done */
            }
            int64_t now = now_ns();
            if( (now - started_at) > TIME_LIMIT_NS ) {
                out_of_time = 1;
            }
            if( out_of_time || (slot[s].next == (int64_t)slot[s].nb_tiles * nb_rounds) ) {
                slot[s].next = -1;
                slots_busy--;
                continue;
            }
            int tile = s + in_flight * (int)(slot[s].next % slot[s].nb_tiles);
            int round = (int)(slot[s].next / slot[s].nb_tiles);
            int expected_value = 2 * round + 1;
            slot[s].next++;
            nb_sequences++;
            rounds_of_tile[tile]++;
            parsec_dtd_insert_task(taskpool, body_first_writer, 0, PARSEC_DEV_CPU, "First_Writer",
                                   PASSED_BY_REF, TILE_OF(tile), PARSEC_INOUT | arena | PARSEC_AFFINITY,
                                   sizeof(int), &tile, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
            parsec_dtd_insert_task(taskpool, body_reader, 0, PARSEC_DEV_CPU, "Reader",
                                   PASSED_BY_REF, TILE_OF(tile), PARSEC_INPUT | arena | PARSEC_AFFINITY,
                                   sizeof(int), &tile, PARSEC_VALUE,
                                   sizeof(int), &expected_value, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
            slot[s].tile = tile;
            slot[s].round = round;
            slot[s].polls = 0;
            slot[s].since = now;
        }
    }
#undef TILE_OF

    parsec_dtd_data_flush_all(taskpool, collection);

    rc = parsec_taskpool_wait(taskpool);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    int64_t nb_hold_ups = hold_ups_stop();
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    int failures = 0, wrong_tiles = 0, tiles_with_readers = 0;
    for( int tile = 0; tile < nb_tiles; tile++ ) {
        if( *(int *)host_copy[tile]->device_private != 2 * rounds_of_tile[tile] ) {
            wrong_tiles++;
        }
        if( 0 != host_copy[tile]->readers ) {
            tiles_with_readers++;
        }
    }
    if( 0 != atomic_load(&readers_saw_later_value) ) {
        parsec_output(0, "FAILED: %d of %" PRId64 " readers saw the value of the writer inserted after them\n",
                      atomic_load(&readers_saw_later_value), nb_sequences);
        failures++;
    }
    if( 0 != atomic_load(&readers_saw_earlier_value) ) {
        parsec_output(0, "FAILED: %d of %" PRId64 " readers ran before the writer inserted before them\n",
                      atomic_load(&readers_saw_earlier_value), nb_sequences);
        failures++;
    }
    if( 0 != atomic_load(&writers_ran_before_reader) ) {
        parsec_output(0, "FAILED: %d of %" PRId64 " writers started before the reader inserted before them had finished\n",
                      atomic_load(&writers_ran_before_reader), nb_sequences);
        failures++;
    }
    if( 0 != wrong_tiles ) {
        parsec_output(0, "FAILED: %d of %d tiles do not hold their final value\n", wrong_tiles, nb_tiles);
        failures++;
    }
    if( 0 != tiles_with_readers ) {
        parsec_output(0, "FAILED: %d of %d tiles are left with a non-zero reader count\n",
                      tiles_with_readers, nb_tiles);
        failures++;
    }
    if( out_of_time ) {
        parsec_output(0, "Note: stopped after %d s, with %" PRId64 " of %" PRId64 " sequences\n",
                      (int)(TIME_LIMIT_NS / 1000000000), nb_sequences, (int64_t)nb_tiles * nb_rounds);
    }
    if( 0 != waits_expired ) {
        parsec_output(0, "Note: the first writer of a tile had not run after %d ms %" PRId64 " times%s\n",
                      WAIT_LIMIT_NS / 1000000, waits_expired,
                      wait_for_first_writer ? "" : "; the test stopped waiting for it");
    }
    if( 0 != failures ) {
        parsec_output(0, "DTD last reader: %d checks FAILED\n", failures);
    } else {
        parsec_output(0, "DTD last reader: all checks passed (%" PRId64 " sequences, %d threads, %" PRId64 " hold-ups)\n",
                      nb_sequences, nb_threads, nb_hold_ups);
    }

    parsec_taskpool_free(taskpool);
    parsec_dtd_data_collection_fini(collection);
    free(host_copy);
    free(progress);
    free(rounds_of_tile);
    free_data(matrix);
    parsec_dtd_free_arena_datatype(parsec, arena);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return (0 != failures) ? 1 : 0;
}
