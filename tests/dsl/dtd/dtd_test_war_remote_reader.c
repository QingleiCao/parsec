/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* A writer must not change a tile in place while a message still reads it.
 *
 * Per tile and round the test inserts a writer on the rank that owns the tile,
 * readers on every other rank, and a second writer on the owner. DTD holds a
 * writer back for as long as the copy it is about to overwrite has readers, and
 * for the readers on other ranks that reader is the message that carries the
 * tile to them: the second writer may run only once the tile has left. The
 * readers check that they see the value of the first writer, and at the end
 * every tile holds the value of the last writer.
 *
 * The tiles are larger than the short message limit (1 KB unless configured
 * otherwise), so a tile does not travel with the activation: the receiver asks
 * for it once the activation has arrived, and the owner sends it only then, a
 * network round trip later. A second writer that runs before the message counts
 * itself among the readers therefore shows as a reader that received the
 * second writer's value, nearly every time it happens.
 *
 * Two phases:
 *
 *   - ahead, one round: the tasks of a tile are inserted back to back, so the
 *     second writer is inserted well before the first one completes and is
 *     released by that completion. The message has to count itself from its
 *     activation to its end. On a runtime where it does not, nearly every
 *     reader fails.
 *
 *   - behind, several rounds: the owner inserts the second writer while the
 *     completion of the first one walks its successors. That walk tells the
 *     tile that the remote reader is its last user before it activates the
 *     message, and a writer inserted in between is released by its own
 *     insertion and runs at once. Something has to count the message among the
 *     readers across that gap, which lasts a fraction of a microsecond on the
 *     thread completing the first writer. To have second writers inserted in
 *     it, that thread is held up in it on purpose (where the system allows,
 *     Linux): the body of the first writer asks for a signal to its own thread
 *     a little later, from 0 to HOLD_UP_SWEEP_NS from one tile to the next, so
 *     that the signal lands somewhere along the completion, and the handler
 *     keeps the thread for HOLD_UP_LENGTH_NS wherever it was. The owner inserts
 *     the second writer while the thread is held, and another thread runs it.
 *     Nothing in the runtime is changed for this. Where the signal lands in the
 *     gap, a runtime that leaves it open fails; the sweep makes that a few
 *     percent of the tiles on the machines tried. Without the signals the gap
 *     is still met now and then. This takes MIN_THREADS threads per rank or
 *     more. With fewer, as under a launcher that binds each rank to one core
 *     (Open MPI's mpiexec without --bind-to none), the owner does not wait and
 *     the behind phase checks little; the ahead phase needs a single thread.
 *
 * usage: dtd_test_war_remote_reader [cores [tiles [ints per tile [rounds behind [readers per rank]]]]]
 */

#include "parsec/runtime.h"

#include <errno.h>
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

#define DEFAULT_NB_TILES          256
#define DEFAULT_INTS_PER_TILE     16384   /* 64 KB: well above the short message limit */
#define DEFAULT_ROUNDS_BEHIND     8
#define DEFAULT_READERS_PER_RANK  1
/* In the behind phase: how long the owner waits for the body of a first writer
 * before inserting the second one anyway (and how many such waits it accepts
 * before it stops waiting); when the thread of the first writer is held up
 * after its body, from 0 to HOLD_UP_SWEEP_NS in HOLD_UP_STEPS steps, and for
 * how long; and how long after the signal the owner inserts the second writer.
 * Waiting is worth it with MIN_THREADS threads per rank: the inserting one, one
 * to complete the first writer and one to run the second while that is held. */
#define WAIT_LIMIT_NS             (1000 * 1000 * 1000)
#define MAX_WAITS_EXPIRED         8
#define MIN_THREADS               3
#define HOLD_UP_SWEEP_NS          8000
#define HOLD_UP_STEPS             128
#define HOLD_UP_LENGTH_NS         20000
#define INSERT_AFTER_SIGNAL_NS    4000

enum { AHEAD = 0, BEHIND = 1, NB_PHASES = 2 };
static const char *phase_name[NB_PHASES] = { "ahead", "behind" };

static int32_t count_read[NB_PHASES];          /* readers that ran on this rank */
static int32_t count_after_second[NB_PHASES];  /* readers that saw a value of the second writer */
static int32_t count_before_first[NB_PHASES];  /* readers that saw a value from before the first writer */
static volatile int32_t *first_writer_round;   /* per tile, the round of the last first writer whose body ran */
static int wait_for_first_writers = 1;
static int waits_expired = 0;
static int ints_per_tile = DEFAULT_INTS_PER_TILE;
static int TILE_FULL;

static int64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

/* The delay from the body of a first writer to its hold-up, per behind sequence of the owner */
static int64_t hold_up_delay(int sequence)
{
    return (int64_t)(sequence % HOLD_UP_STEPS) * HOLD_UP_SWEEP_NS / HOLD_UP_STEPS;
}

/* Holding the thread of a first writer up.
 *
 * Each worker thread gets a timer of its own the first time it runs a first
 * writer, which signals that very thread. The handler does nothing but wait:
 * for the runtime it is as if the thread had lost its core for a moment at
 * whatever instruction it was. */
#if defined(__linux__) && defined(SIGEV_THREAD_ID) && defined(SYS_gettid)
#if !defined(sigev_notify_thread_id)
#define sigev_notify_thread_id _sigev_un._tid
#endif
#define HOLD_UP_SIGNAL            (SIGRTMIN + 5)

static int               hold_ups_enabled = 0;
static timer_t          *hold_up_timers;
static int               hold_up_max_timers;
static int32_t           hold_up_nb_timers;
static int32_t           hold_up_count;
static __thread int      hold_up_timer_index = -1;

static void hold_up_handler(int signal_number)
{
    int saved_errno = errno;
    int64_t until = now_ns() + HOLD_UP_LENGTH_NS;
    (void)signal_number;
    (void)parsec_atomic_fetch_inc_int32(&hold_up_count);
    while( now_ns() < until ) /* nothing */;
    errno = saved_errno;
}

static void hold_ups_prepare(int nb_threads)
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = hold_up_handler;
    action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask);
    if( (HOLD_UP_SIGNAL > SIGRTMAX) || (0 != sigaction(HOLD_UP_SIGNAL, &action, NULL)) ) return;
    hold_up_timers = (timer_t *)calloc(nb_threads, sizeof(timer_t));
    if( NULL == hold_up_timers ) return;
    hold_up_max_timers = nb_threads;
    hold_ups_enabled = 1;
}

/* Called by the body of a first writer: signal this thread after delay_ns */
static void hold_up_this_thread(int64_t delay_ns)
{
    struct itimerspec when;

    if( !hold_ups_enabled ) return;
    if( hold_up_timer_index < 0 ) {
        struct sigevent event;
        timer_t timer;
        int index;
        memset(&event, 0, sizeof(event));
        event.sigev_notify = SIGEV_THREAD_ID;
        event.sigev_signo = HOLD_UP_SIGNAL;
        event.sigev_notify_thread_id = (pid_t)syscall(SYS_gettid);
        if( 0 != timer_create(CLOCK_MONOTONIC, &event, &timer) ) return;
        index = parsec_atomic_fetch_inc_int32(&hold_up_nb_timers);
        if( index >= hold_up_max_timers ) {
            timer_delete(timer);
            return;
        }
        hold_up_timers[index] = timer;
        hold_up_timer_index = index;
    }
    memset(&when, 0, sizeof(when));
    when.it_value.tv_nsec = (long)(delay_ns > 0 ? delay_ns : 1);  /* 0 would disarm it */
    timer_settime(hold_up_timers[hold_up_timer_index], 0, &when, NULL);
}

/* Called once the tasks are done, before the worker threads go away */
static int hold_ups_stop(void)
{
    int nb_timers = hold_up_nb_timers;
    if( nb_timers > hold_up_max_timers ) nb_timers = hold_up_max_timers;
    for( int i = 0; i < nb_timers; i++ ) timer_delete(hold_up_timers[i]);
    hold_up_max_timers = 0;
    free(hold_up_timers);
    hold_up_timers = NULL;
    return hold_up_count;
}
#else
static int hold_ups_enabled = 0;
static void hold_ups_prepare(int nb_threads) { (void)nb_threads; }
static void hold_up_this_thread(int64_t delay_ns) { (void)delay_ns; }
static int hold_ups_stop(void) { return 0; }
#endif

/* Round 0 is the ahead phase, the others are the behind phase. Each round runs
 * two writers on every tile, so the readers of round r expect 2 r + 1. */
static int read_kernel(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int round, where, *data, after = 0, before = 0;
    (void)es;

    parsec_dtd_unpack_args(this_task, &round, &where, &data);
    for( int i = 0; i < ints_per_tile; i++ ) {
        after  |= (data[i] > 2 * round + 1);
        before |= (data[i] < 2 * round + 1);
    }
    int phase = (0 == round) ? AHEAD : BEHIND;
    (void)parsec_atomic_fetch_inc_int32(&count_read[phase]);
    if( after )  (void)parsec_atomic_fetch_inc_int32(&count_after_second[phase]);
    if( before ) (void)parsec_atomic_fetch_inc_int32(&count_before_first[phase]);
    return PARSEC_HOOK_RETURN_DONE;
}

/* round is the round of a first writer of the behind phase, 0 for any other
 * writer; sequence numbers the behind sequences of the owner. */
static int write_kernel(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int tile, round, sequence, *data;
    (void)es;

    parsec_dtd_unpack_args(this_task, &tile, &round, &sequence, &data);
    for( int i = 0; i < ints_per_tile; i++ ) data[i] += 1;
    if( round > 0 ) {
        /* The last thing the body does: its completion follows */
        hold_up_this_thread(hold_up_delay(sequence));
        parsec_atomic_wmb();
        first_writer_round[tile] = round;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/* The owner inserts the second writer of a tile while the thread of the first
 * one is held up, or would be. */
static void wait_for_first_writer(int tile, int round, int sequence)
{
    int64_t start = now_ns(), delay;
    if( !wait_for_first_writers || (waits_expired >= MAX_WAITS_EXPIRED) ) return;
    while( first_writer_round[tile] < round ) {
        if( now_ns() - start > WAIT_LIMIT_NS ) { waits_expired++; return; }
        parsec_atomic_rmb();
    }
    start = now_ns();
    delay = hold_up_delay(sequence) + INSERT_AFTER_SIGNAL_NS;
    while( now_ns() - start < delay ) ;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int rank, world, cores = -1, rc, failed = 0, no_round = 0, sequence = 0, hold_ups;
    int nb_tiles = DEFAULT_NB_TILES, rounds = DEFAULT_ROUNDS_BEHIND, readers = DEFAULT_READERS_PER_RANK;
    int expected_reads[NB_PHASES] = { 0, 0 }, wrong_tiles = 0;
    parsec_tiled_matrix_t *dcA;
    parsec_arena_datatype_t *adt;

    if( argc > 1 ) cores = atoi(argv[1]);
    if( argc > 2 ) nb_tiles = atoi(argv[2]);
    if( argc > 3 ) ints_per_tile = atoi(argv[3]);
    if( argc > 4 ) rounds = atoi(argv[4]);
    if( argc > 5 ) readers = atoi(argv[5]);
    if( (nb_tiles < 1) || (ints_per_tile < 1) || (rounds < 0) || (readers < 1) ) {
        fprintf(stderr, "usage: %s [cores [tiles [ints per tile [rounds behind [readers per rank]]]]]\n", argv[0]);
        return 1;
    }
    first_writer_round = calloc(nb_tiles, sizeof(int32_t));

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED, &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");
    int nb_threads = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_CORES);
    wait_for_first_writers = (nb_threads >= MIN_THREADS);
    hold_ups_prepare(nb_threads);

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    adt = parsec_matrix_adt_new_rect(parsec_datatype_int32_t, ints_per_tile, 1, ints_per_tile);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);
    dcA = create_and_distribute_data(rank, world, ints_per_tile, nb_tiles);
    memset(((parsec_matrix_block_cyclic_t *)dcA)->mat, 0,
           (size_t)dcA->nb_local_tiles * (size_t)dcA->bsiz * (size_t)parsec_datadist_getsizeoftype(dcA->mtype));
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");
    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    rc = parsec_context_add_taskpool(parsec, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( int round = 0; round <= rounds; round++ ) {
        int phase = (0 == round) ? AHEAD : BEHIND;
        for( int t = 0; t < nb_tiles; t++ ) {
            int key = A->data_key(A, t, 0);
            int owner = (int)A->rank_of_key(A, key);

            parsec_dtd_insert_task(dtd_tp, write_kernel, 0, PARSEC_DEV_CPU, "Write_Task",
                                   sizeof(int), &t, PARSEC_VALUE,
                                   sizeof(int), &round, PARSEC_VALUE,
                                   sizeof(int), &sequence, PARSEC_VALUE,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
            for( int k = 1; k < world; k++ ) {
                int where = (owner + k) % world;
                if( where == rank ) expected_reads[phase] += readers;
                for( int j = 0; j < readers; j++ ) {
                    parsec_dtd_insert_task(dtd_tp, read_kernel, 0, PARSEC_DEV_CPU, "Read_Task",
                                           sizeof(int), &round, PARSEC_VALUE,
                                           sizeof(int), &where, PARSEC_VALUE | PARSEC_AFFINITY,
                                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key), PARSEC_INPUT | TILE_FULL,
                                           PARSEC_DTD_ARG_END);
                }
            }
            if( (BEHIND == phase) && (owner == rank) ) {
                wait_for_first_writer(t, round, sequence);
                sequence++;
            }
            parsec_dtd_insert_task(dtd_tp, write_kernel, 0, PARSEC_DEV_CPU, "Write_Task",
                                   sizeof(int), &t, PARSEC_VALUE,
                                   sizeof(int), &no_round, PARSEC_VALUE,
                                   sizeof(int), &sequence, PARSEC_VALUE,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        }
    }
    parsec_dtd_data_flush_all(dtd_tp, A);
    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    hold_ups = hold_ups_stop();
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    /* Every writer ran on every tile this rank owns */
    for( int t = 0; t < nb_tiles; t++ ) {
        int key = A->data_key(A, t, 0);
        if( (int)A->rank_of_key(A, key) != rank ) continue;
        parsec_data_t *data = A->data_of_key(A, key);
        int *ptr = PARSEC_DATA_COPY_GET_PTR(parsec_data_get_copy(data, 0));
        for( int i = 0; i < ints_per_tile; i++ ) {
            if( 2 * (rounds + 1) != ptr[i] ) { wrong_tiles++; break; }
        }
    }

    for( int phase = AHEAD; phase < NB_PHASES; phase++ ) {
        int phase_failed = (count_after_second[phase] != 0) || (count_before_first[phase] != 0) ||
                           (count_read[phase] != expected_reads[phase]);
        printf("rank %d of %d, %s: %d of %d readers ran here, %d saw the tile after the second writer changed it, "
               "%d before the first writer: %s\n",
               rank, world, phase_name[phase], count_read[phase], expected_reads[phase],
               count_after_second[phase], count_before_first[phase], phase_failed ? "FAILED" : "passed");
        failed |= phase_failed;
    }
    if( wrong_tiles ) {
        printf("rank %d of %d: %d tiles owned here do not hold the final value: FAILED\n", rank, world, wrong_tiles);
        failed = 1;
    }
    printf("rank %d of %d: %d threads, %d hold-ups of %d behind sequences owned here%s%s",
           rank, world, nb_threads, hold_ups, sequence,
           hold_ups_enabled ? "" : " (hold-ups not available here)",
           wait_for_first_writers ? "" : ", too few threads to insert second writers behind the first ones");
    if( waits_expired ) {
        printf(", %d first writers did not run within %d ms%s", waits_expired, WAIT_LIMIT_NS / 1000000,
               (waits_expired >= MAX_WAITS_EXPIRED) ? " (stopped waiting for them)" : "");
    }
    printf("\n");

    /* Only rank 0 reports a failure through its exit code: a launcher seeing
     * several ranks fail at once can get stuck tearing the job down. */
    rc = parsec_tests_allreduce(parsec, NULL, &failed, 1, parsec_datatype_int_t, PARSEC_TESTS_REDUCE_SUM);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_allreduce");

    parsec_taskpool_free(dtd_tp);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);
    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
    free((void*)first_writer_round);
    return ((0 == rank) && failed) ? 1 : 0;
}
