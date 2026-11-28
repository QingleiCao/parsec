/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* A task may name the same tile in several of its flows.
 *
 * A kernel written for distinct tiles is called with one tile twice now and
 * then: C += A * A, B += B, a copy in place. The task is then one user of
 * that tile, a reader if all these flows read it and a writer if one of them
 * writes it: it runs after the tasks inserted before it on the tile, the
 * tasks inserted after it see what it did, and every flow gets the same
 * memory.
 *
 * For each way of naming the tiles (see patterns[]) the test inserts, on many
 * tiles and while the workers execute,
 *
 *     [a writer [, a reader]]   the task under test   [[a reader,] a writer]
 *
 * that is every combination of nothing, a writer, or a writer and a reader
 * before the task on each of its tiles, and nothing, a writer, or a reader and
 * a writer after it. Every write adds one to the tile, so each task knows the
 * value it has to find. The test checks
 *
 *   - in every body, that each flow got the memory of its tile, with the value
 *     that sequential execution gives it, and that no writer of a tile ran
 *     along with another user of that tile,
 *   - that every task ran, once,
 *   - the final value of every tile,
 *   - that no copy is left with readers, nor with other references than the
 *     copy of a tile that a single writer used.
 *
 * With several processes every task runs where its tiles live, and the other
 * processes only track it: no tile travels.
 *
 * usage: dtd_test_same_tile_flows [cores [sequences per combination]]
 */

#include "parsec/runtime.h"

#include <ctype.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "tests/tests_data.h"
#include "tests/tests_runtime.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#define MAX_FLOWS 4
#define NB_KINDS  3   /* nothing, one task, two tasks: before the task under test, and after it */

/* One letter per flow of the task under test. On its first tile, r reads,
 * w reads and writes, o only writes; R and W read, and read and write, its
 * second tile.
 *
 * The first pattern is one that a runtime handling these flows one by one
 * gets wrong without hanging, and the test stops at the first pattern that
 * fails: such a runtime never returns from some of the others.
 */
static const char *patterns[] = {
    "rw", "rr", "wr", "rrr", "rrw", "wrr", "rwr",  /* one tile, one writing flow at most */
    "ro", "or",                                    /* written without being read */
    "ww", "wrw",                                   /* two writing flows */
    "rRr", "rWrR", "rRwW"                          /* two tiles, their flows interleaved */
};
#define NB_PATTERNS ((int)(sizeof(patterns) / sizeof(patterns[0])))

typedef struct {
    const char *pattern;
    int         tile[2];   /* the tiles the task under test names */
    int32_t     value[2];  /* what they hold when it runs */
} sequence_t;

/* What the bodies need, for the pattern being run. */
static sequence_t  *sequences;
static int32_t    **storage;      /* the memory of each tile of this process */
static _Atomic int *readers_in;   /* bodies reading each tile right now */
static _Atomic int *writers_in;   /* bodies writing each tile right now */
static _Atomic int  tasks_run, wrong_pointers, wrong_values, overlaps;

/* The first flow that found another value than expected, to tell what happened. */
static _Atomic int  first_wrong_taken;
static const char  *first_wrong_task;
static int          first_wrong_tile;
static int32_t      first_wrong_expected, first_wrong_found;

static int flow_tile(char flow)   { return isupper((unsigned char)flow) ? 1 : 0; }
static int flow_writes(char flow) { return 'r' != tolower((unsigned char)flow); }

static int
flow_access(char flow)
{
    switch( tolower((unsigned char)flow) ) {
    case 'r': return PARSEC_INPUT;
    case 'w': return PARSEC_INOUT;
    default:  return PARSEC_OUTPUT;
    }
}

static int
pattern_uses(const char *pattern, int which)
{
    for( ; '\0' != *pattern; pattern++ ) {
        if( which == flow_tile(*pattern) ) return 1;
    }
    return 0;
}

static int
pattern_writes(const char *pattern, int which)
{
    for( ; '\0' != *pattern; pattern++ ) {
        if( which == flow_tile(*pattern) && flow_writes(*pattern) ) return 1;
    }
    return 0;
}

static void
enter_tile(int tile, int writes)
{
    if( writes ) {
        if( 0 != atomic_fetch_add(&writers_in[tile], 1) ||
            0 != atomic_load(&readers_in[tile]) ) {
            atomic_fetch_add(&overlaps, 1);
        }
    } else {
        atomic_fetch_add(&readers_in[tile], 1);
        if( 0 != atomic_load(&writers_in[tile]) ) {
            atomic_fetch_add(&overlaps, 1);
        }
    }
}

static void
leave_tile(int tile, int writes)
{
    atomic_fetch_sub(writes ? &writers_in[tile] : &readers_in[tile], 1);
}

/* A flow must get the memory of its tile, holding the value expected.
 * Returns whether the pointer can be used. */
static int
check_flow(const char *task, const int32_t *pointer, int tile, int32_t value)
{
    if( pointer != storage[tile] ) {
        atomic_fetch_add(&wrong_pointers, 1);
        return 0;
    }
    const int32_t found = *pointer;
    if( found != value ) {
        atomic_fetch_add(&wrong_values, 1);
        if( 0 == atomic_exchange(&first_wrong_taken, 1) ) {
            first_wrong_task = task;
            first_wrong_tile = tile;
            first_wrong_expected = value;
            first_wrong_found = found;
        }
    }
    return 1;
}

static int
body_reader(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int32_t *pointer, value;
    int tile;
    parsec_dtd_unpack_args(this_task, &pointer, &tile, &value);

    enter_tile(tile, 0);
    (void)check_flow("a reader", pointer, tile, value);
    leave_tile(tile, 0);
    atomic_fetch_add(&tasks_run, 1);
    return PARSEC_HOOK_RETURN_DONE;
}

static int
body_writer(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int32_t *pointer, value;
    int tile;
    parsec_dtd_unpack_args(this_task, &pointer, &tile, &value);

    enter_tile(tile, 1);
    if( check_flow("a writer", pointer, tile, value) ) {
        *pointer = value + 1;
    }
    leave_tile(tile, 1);
    atomic_fetch_add(&tasks_run, 1);
    return PARSEC_HOOK_RETURN_DONE;
}

/* The task under test: look at every flow first, then write each tile it
 * writes, once, through the first flow that may. */
static void
task_under_test(int sequence_index, int32_t **pointer)
{
    const sequence_t *sequence = &sequences[sequence_index];
    const char *pattern = sequence->pattern;
    int usable[MAX_FLOWS], written[2] = {0, 0};

    for( int which = 0; which < 2; which++ ) {
        if( pattern_uses(pattern, which) ) {
            enter_tile(sequence->tile[which], pattern_writes(pattern, which));
        }
    }
    for( int flow = 0; '\0' != pattern[flow]; flow++ ) {
        int which = flow_tile(pattern[flow]);
        usable[flow] = check_flow("the task under test", pointer[flow],
                                  sequence->tile[which], sequence->value[which]);
    }
    for( int flow = 0; '\0' != pattern[flow]; flow++ ) {
        int which = flow_tile(pattern[flow]);
        if( flow_writes(pattern[flow]) && usable[flow] && !written[which] ) {
            *pointer[flow] = sequence->value[which] + 1;
            written[which] = 1;
        }
    }
    for( int which = 0; which < 2; which++ ) {
        if( pattern_uses(pattern, which) ) {
            leave_tile(sequence->tile[which], pattern_writes(pattern, which));
        }
    }
    atomic_fetch_add(&tasks_run, 1);
}

static int
body_under_test_2(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int32_t *pointer[MAX_FLOWS];
    int sequence_index;
    parsec_dtd_unpack_args(this_task, &pointer[0], &pointer[1], &sequence_index);
    task_under_test(sequence_index, pointer);
    return PARSEC_HOOK_RETURN_DONE;
}

static int
body_under_test_3(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int32_t *pointer[MAX_FLOWS];
    int sequence_index;
    parsec_dtd_unpack_args(this_task, &pointer[0], &pointer[1], &pointer[2], &sequence_index);
    task_under_test(sequence_index, pointer);
    return PARSEC_HOOK_RETURN_DONE;
}

static int
body_under_test_4(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int32_t *pointer[MAX_FLOWS];
    int sequence_index;
    parsec_dtd_unpack_args(this_task, &pointer[0], &pointer[1], &pointer[2], &pointer[3], &sequence_index);
    task_under_test(sequence_index, pointer);
    return PARSEC_HOOK_RETURN_DONE;
}

/* Run one pattern on its own taskpool and tiles.
 * Returns the number of checks that failed in this process. */
static int
run_pattern(parsec_context_t *context, const char *pattern, int sequences_per_combination,
            int rank, int world)
{
    const int nb_flows = (int)strlen(pattern);
    const int nb_sequences = NB_KINDS * NB_KINDS * sequences_per_combination;
    /* Behind the first tiles of the sequences come the control tiles, one in
     * each process, which get a single writer. The two tiles of a sequence
     * are a multiple of the number of processes apart, so that they live in
     * the same one. */
    const int first_control_tile = nb_sequences;
    const int second_tile_offset = ((nb_sequences + world + world - 1) / world) * world;
    const int nb_tiles = 2 * second_tile_offset;
    int arena, failures = 0, rc, tasks_inserted = 0;

    parsec_taskpool_t *taskpool = parsec_dtd_taskpool_new();
    parsec_arena_datatype_t *arena_datatype =
        parsec_matrix_adt_new_rect(parsec_datatype_int32_t, 1, 1, 1);
    parsec_dtd_attach_arena_datatype(context, arena_datatype, &arena);

    parsec_tiled_matrix_t *matrix = create_and_distribute_data(rank, world, 1, nb_tiles);
    memset(((parsec_matrix_block_cyclic_t *)matrix)->mat, 0,
           (size_t)matrix->nb_local_tiles * (size_t)matrix->bsiz *
           (size_t)parsec_datadist_getsizeoftype(matrix->mtype));
    parsec_data_collection_t *collection = (parsec_data_collection_t *)matrix;
    parsec_data_collection_set_key(collection, "A");
    parsec_dtd_data_collection_init(collection);

    sequences  = calloc(nb_sequences, sizeof(*sequences));
    storage    = calloc(nb_tiles, sizeof(*storage));
    readers_in = calloc(nb_tiles, sizeof(*readers_in));
    writers_in = calloc(nb_tiles, sizeof(*writers_in));
    parsec_data_copy_t **host_copy = calloc(nb_tiles, sizeof(*host_copy));
    int32_t *final_value       = calloc(nb_tiles, sizeof(*final_value));
    int32_t *references_before = calloc(nb_tiles, sizeof(*references_before));
    char    *tile_used         = calloc(nb_tiles, sizeof(*tile_used));

    for( int tile = 0; tile < nb_tiles; tile++ ) {
        parsec_data_key_t key = collection->data_key(collection, tile, 0);
        if( (int)collection->rank_of_key(collection, key) != rank ) {
            continue;
        }
        host_copy[tile] = collection->data_of_key(collection, key)->device_copies[0];
        storage[tile] = (int32_t *)host_copy[tile]->device_private;
        references_before[tile] = host_copy[tile]->super.super.obj_reference_count;
    }
    atomic_store(&tasks_run, 0);
    atomic_store(&wrong_pointers, 0);
    atomic_store(&wrong_values, 0);
    atomic_store(&overlaps, 0);
    atomic_store(&first_wrong_taken, 0);

    rc = parsec_context_add_taskpool(context, taskpool);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(context);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

#define TILE_OF(tile) \
    parsec_dtd_tile_of(collection, collection->data_key(collection, (tile), 0))
#define INSERT_READER(tile)                                                                      \
    parsec_dtd_insert_task(taskpool, body_reader, 0, PARSEC_DEV_CPU, "Reader",                   \
                           PASSED_BY_REF, TILE_OF(tile), PARSEC_INPUT | arena | PARSEC_AFFINITY, \
                           sizeof(int), &(tile), PARSEC_VALUE,                                   \
                           sizeof(int32_t), &final_value[tile], PARSEC_VALUE,                    \
                           PARSEC_DTD_ARG_END)
#define INSERT_WRITER(tile)                                                                      \
    do {                                                                                         \
        parsec_dtd_insert_task(taskpool, body_writer, 0, PARSEC_DEV_CPU, "Writer",               \
                               PASSED_BY_REF, TILE_OF(tile), PARSEC_INOUT | arena | PARSEC_AFFINITY, \
                               sizeof(int), &(tile), PARSEC_VALUE,                               \
                               sizeof(int32_t), &final_value[tile], PARSEC_VALUE,                \
                               PARSEC_DTD_ARG_END);                                              \
        final_value[tile]++;                                                                     \
    } while(0)
#define FLOW(flow)                                                                               \
    PASSED_BY_REF, TILE_OF(sequence->tile[flow_tile(pattern[flow])]),                            \
    flow_access(pattern[flow]) | arena | (0 == (flow) ? PARSEC_AFFINITY : 0)

    for( int index = 0; index < nb_sequences; index++ ) {
        /* The combinations take turns, so that all of them meet workers that
         * are busy and workers that are not. */
        const int before = index % NB_KINDS, after = (index / NB_KINDS) % NB_KINDS;
        sequence_t *sequence = &sequences[index];
        int tasks = 1;

        sequence->pattern = pattern;
        sequence->tile[0] = index;
        sequence->tile[1] = index + second_tile_offset;

        for( int which = 0; which < 2; which++ ) {
            int tile = sequence->tile[which];
            if( !pattern_uses(pattern, which) ) continue;
            tile_used[tile] = 1;
            if( before >= 1 ) { INSERT_WRITER(tile); tasks++; }
            if( before == 2 ) { INSERT_READER(tile); tasks++; }
            sequence->value[which] = final_value[tile];
        }

        switch( nb_flows ) {
        case 2:
            parsec_dtd_insert_task(taskpool, body_under_test_2, 0, PARSEC_DEV_CPU, "UnderTest2",
                                   FLOW(0), FLOW(1),
                                   sizeof(int), &index, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
            break;
        case 3:
            parsec_dtd_insert_task(taskpool, body_under_test_3, 0, PARSEC_DEV_CPU, "UnderTest3",
                                   FLOW(0), FLOW(1), FLOW(2),
                                   sizeof(int), &index, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
            break;
        default:
            parsec_dtd_insert_task(taskpool, body_under_test_4, 0, PARSEC_DEV_CPU, "UnderTest4",
                                   FLOW(0), FLOW(1), FLOW(2), FLOW(3),
                                   sizeof(int), &index, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
            break;
        }

        for( int which = 0; which < 2; which++ ) {
            int tile = sequence->tile[which];
            if( !pattern_uses(pattern, which) ) continue;
            if( pattern_writes(pattern, which) ) final_value[tile]++;
            if( after == 2 ) { INSERT_READER(tile); tasks++; }
            if( after >= 1 ) { INSERT_WRITER(tile); tasks++; }
        }

        if( NULL != storage[sequence->tile[0]] ) {
            tasks_inserted += tasks;  /* they run here */
        }
    }
    for( int tile = first_control_tile; tile < first_control_tile + world; tile++ ) {
        tile_used[tile] = 1;
        INSERT_WRITER(tile);
        if( NULL != storage[tile] ) {
            tasks_inserted++;
        }
    }
#undef FLOW
#undef INSERT_WRITER
#undef INSERT_READER
#undef TILE_OF

    parsec_dtd_data_flush_all(taskpool, collection);

    rc = parsec_taskpool_wait(taskpool);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(context);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    int wrong_tiles = 0, tiles_with_readers = 0, tiles_with_references = 0;
    for( int tile = 0; tile < nb_tiles; tile++ ) {
        if( NULL == storage[tile] ) continue;
        if( *storage[tile] != final_value[tile] ) wrong_tiles++;
        if( 0 != host_copy[tile]->readers ) tiles_with_readers++;
    }

    parsec_taskpool_free(taskpool);
    parsec_dtd_data_collection_fini(collection);

    /* What its single writer left on the copy of the control tile of this
     * process is what the tasks may leave on the copy of any tile they used. */
    int32_t references_left = 0;
    for( int tile = first_control_tile; tile < first_control_tile + world; tile++ ) {
        if( NULL == storage[tile] ) continue;
        references_left = host_copy[tile]->super.super.obj_reference_count - references_before[tile];
    }
    for( int tile = 0; tile < nb_tiles; tile++ ) {
        if( NULL == storage[tile] ) continue;
        if( host_copy[tile]->super.super.obj_reference_count - references_before[tile] !=
            (tile_used[tile] ? references_left : 0) ) {
            tiles_with_references++;
        }
    }

#define CHECK(count, ...)                                                        \
    if( 0 != (count) ) {                                                         \
        char what[512];                                                          \
        snprintf(what, sizeof(what), __VA_ARGS__);                               \
        parsec_output(0, "[%d] pattern %s FAILED: %s\n", rank, pattern, what);   \
        failures++;                                                              \
    }
    CHECK(tasks_inserted - atomic_load(&tasks_run),
          "%d of %d tasks ran", atomic_load(&tasks_run), tasks_inserted);
    CHECK(atomic_load(&wrong_pointers),
          "%d flows did not get the memory of their tile", atomic_load(&wrong_pointers));
    /* The tiles of a sequence tell which combination it is, see the insertion. */
    CHECK(atomic_load(&wrong_values),
          "%d flows found another value than sequential execution gives them; the first one, "
          "%s with %d task(s) before the task under test and %d after it, found %d and not %d",
          atomic_load(&wrong_values), first_wrong_task,
          (first_wrong_tile % second_tile_offset) % NB_KINDS,
          ((first_wrong_tile % second_tile_offset) / NB_KINDS) % NB_KINDS,
          (int)first_wrong_found, (int)first_wrong_expected);
    CHECK(atomic_load(&overlaps),
          "%d times a writer ran along with another user of its tile", atomic_load(&overlaps));
    CHECK(wrong_tiles, "%d tiles do not hold their final value", wrong_tiles);
    CHECK(tiles_with_readers, "%d copies are left with a reader count that is not zero",
          tiles_with_readers);
    CHECK(tiles_with_references,
          "%d copies are left with other references than the copy a single writer used",
          tiles_with_references);
#undef CHECK

    free(tile_used);
    free(references_before);
    free(final_value);
    free(host_copy);
    free(writers_in);
    free(readers_in);
    free(storage);
    free(sequences);
    free_data(matrix);
    parsec_dtd_free_arena_datatype(context, arena);
    return failures;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int rank, world, cores = -1, sequences_per_combination = 40, rc;
    int failures = 0, patterns_run = 0;

    if( argc > 1 && isdigit((unsigned char)argv[1][0]) ) {
        cores = atoi(argv[1]);
        if( argc > 2 && atoi(argv[2]) > 0 ) {
            sequences_per_combination = atoi(argv[2]);
        }
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    for( int index = 0; index < NB_PATTERNS && 0 == failures; index++ ) {
        int local_failures = run_pattern(parsec, patterns[index], sequences_per_combination,
                                         rank, world);
        /* Every process inserts the same tasks, so they stop together. */
        rc = parsec_tests_allreduce(parsec, &local_failures, &failures, 1,
                                    parsec_datatype_int_t, PARSEC_TESTS_REDUCE_SUM);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_allreduce");
        patterns_run++;
    }

    /* Before the context is finalized: this closes the output stream. */
    if( 0 != failures ) {
        parsec_output(0, "[%d] DTD same tile in several flows: %d checks FAILED for pattern %s, "
                         "%d patterns not run\n",
                      rank, failures, patterns[patterns_run - 1], NB_PATTERNS - patterns_run);
    } else if( 0 == rank ) {
        parsec_output(0, "DTD same tile in several flows: all checks passed for %d patterns\n",
                      NB_PATTERNS);
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return (0 != failures) ? 1 : 0;
}
