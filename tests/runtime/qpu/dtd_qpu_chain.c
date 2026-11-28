/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * Quantum tasks inserted with DTD.
 *
 * For each chain k, four tasks are inserted in this order:
 *
 *   prep     writes a circuit and its parameters into two new tiles     rank k mod P
 *   qexec    reads them, submits the job, returns ASYNC                 the QPU's rank
 *   scribble overwrites the circuit tile, where that is the QPU's rank  rank k mod P
 *   check    reads the result and verifies it                           rank k mod P
 *
 * scribble is a writer inserted after a reader that is still outstanding.
 * DTD has to hold it until the quantum task has completed, which is when the
 * job is over and not when the body returned. The mock reads the circuit at
 * the end of the job, so a scribble that ran too early changes the result,
 * and check, which rebuilds the original circuit by itself, sees it.
 *
 * scribble is inserted only for the chains whose circuit lives on the QPU's
 * rank (all of them with one rank). Across ranks the quantum task works on
 * its own copy of the circuit, so a scribble there says nothing about the
 * QPU layer; and DTD does not hold a writer back for a reader on another
 * rank until the tile has been sent, so the copy that arrives can be the
 * overwritten one (seen once in a hundred runs of this test under load, and
 * every time with large tiles in a test without any quantum task). That is
 * DTD's to fix, not this test's to trip over.
 *
 * On the QPU's own rank the test can still fail, about once in a hundred runs
 * on a busy machine, with "the circuit flow does not hold a well-formed
 * circuit": scribble ran before the body of the quantum task. That is DTD
 * too, and no quantum task is needed to see it. The walk over the successors
 * of a completed task counts a reader of the copy only after it has marked
 * the tile's last user as gone, and a writer inserted between the two is
 * released by its own insertion and finds no reader counted. This test stays
 * as it is: holding scribble back is what it is for.
 *
 * The chains are inserted in two halves, the second while jobs of the first
 * are outstanding: a taskpool keeps taking quantum tasks while earlier ones
 * have not completed.
 *
 * Options, before "--":
 *   -n NT     chains (default 16)
 *   -s SHOTS  shots per job (default 100)
 *   -q RANK   rank that runs the quantum tasks (default 0)
 *   -v        verbose
 */

#include "qpu_test_common.h"
#include "parsec/parsec_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int verbose = 0;
static int shots = 100;
static int32_t nb_errors = 0;
static int32_t nb_checked = 0;
static int32_t nb_scribbled = 0;

static int prep_body(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int where, k;
    void *circuit, *params;

    (void)es;
    parsec_dtd_unpack_args(this_task, &where, &circuit, &params, &k);
    qpu_test_make_circuit(circuit, k, 0, 0);
    qpu_test_make_params(params, k, 0);
    return PARSEC_HOOK_RETURN_DONE;
}

static int qexec_body(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int where;
    void *circuit, *params, *result;

    parsec_dtd_unpack_args(this_task, &where, &circuit, &params, &result);
    memset(result, 0xff, QPU_TEST_BLOCK);
    /* The three data arguments, in the order they were given at insertion.
     * The chain is the first parameter value. */
    return parsec_qpu_submit_dtd(es, this_task, 0, 1, 2, shots,
                                 qpu_test_key(qpu_test_params_a((const double*)params), 0));
}

static int scribble_body(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    void *circuit;

    (void)es;
    parsec_dtd_unpack_args(this_task, &circuit);
    /* Overwrite the operations. A quantum task that still had the circuit
     * at the backend would be caught: the mock compares what a job reads at
     * its end with what it was at its submission. */
    memset((char*)circuit + sizeof(parsec_qpu_circuit_t), 0x5a, 5 * PARSEC_QPU_OP_SIZE);
    (void)parsec_atomic_fetch_inc_int32(&nb_scribbled);
    return PARSEC_HOOK_RETURN_DONE;
}

static int check_body(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    unsigned char circuit[QPU_TEST_BLOCK], params[QPU_TEST_BLOCK];
    int where, k, errors;
    void *result;

    parsec_dtd_unpack_args(this_task, &where, &result, &k);
    qpu_test_make_circuit(circuit, k, 0, 0);
    qpu_test_make_params(params, k, 0);
    errors = qpu_test_check_result(result, circuit, params, shots, qpu_test_key(k, 0), "check", k, 0);
    if( errors > 0 ) (void)parsec_atomic_fetch_add_int32(&nb_errors, errors);
    (void)parsec_atomic_fetch_inc_int32(&nb_checked);
    if( verbose ) printf("check(%d) on rank %d\n", k, es->virtual_process->parsec_context->my_rank);
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    parsec_taskpool_t *dtd_tp;
    parsec_task_class_t *prep_tc, *qexec_tc, *scribble_tc, *check_tc;
    parsec_arena_datatype_t *adt;
    parsec_dtd_tile_t **tiles;
    int nt = 16, qrank = 0, TILE;
    int rank, world, rc, test_argc, pargc;
    long errors, checked, scribbled;
    char **pargv;

    qpu_test_split_args(argc, argv, &test_argc, &pargc, &pargv);
    for( int i = 1; i < test_argc; i++ ) {
        if( 0 == strcmp(argv[i], "-v") ) { verbose = 1; continue; }
        if( i + 1 >= test_argc ) break;
        if( 0 == strcmp(argv[i], "-n") ) nt = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-s") ) shots = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-q") ) qrank = atoi(argv[++i]);
    }

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_MULTIPLE, &pargc, &pargv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");
    if( qrank >= world ) qrank = world - 1;

    dtd_tp = parsec_dtd_taskpool_new();

    adt = PARSEC_OBJ_NEW(parsec_arena_datatype_t);
    qpu_test_block_type_set(adt);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE);

    rc = parsec_context_add_taskpool(parsec, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    prep_tc = parsec_dtd_create_task_class(dtd_tp, "prep",
                                           sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                           PASSED_BY_REF, PARSEC_OUTPUT | TILE,
                                           PASSED_BY_REF, PARSEC_OUTPUT | TILE,
                                           sizeof(int), PARSEC_VALUE,
                                           PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, prep_tc, PARSEC_DEV_CPU, prep_body);

    /* The quantum task: a task class whose only incarnation is of the QPU type. */
    qexec_tc = parsec_dtd_create_task_class(dtd_tp, "qexec",
                                            sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                            PASSED_BY_REF, PARSEC_INPUT | TILE,
                                            PASSED_BY_REF, PARSEC_INPUT | TILE,
                                            PASSED_BY_REF, PARSEC_OUTPUT | TILE,
                                            PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, qexec_tc, PARSEC_DEV_QPU, qexec_body);

    scribble_tc = parsec_dtd_create_task_class(dtd_tp, "scribble",
                                               PASSED_BY_REF, PARSEC_INOUT | TILE | PARSEC_AFFINITY,
                                               PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, scribble_tc, PARSEC_DEV_CPU, scribble_body);

    check_tc = parsec_dtd_create_task_class(dtd_tp, "check",
                                            sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                            PASSED_BY_REF, PARSEC_INPUT | TILE,
                                            sizeof(int), PARSEC_VALUE,
                                            PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, check_tc, PARSEC_DEV_CPU, check_body);

    tiles = (parsec_dtd_tile_t**)calloc((size_t)3 * nt, sizeof(parsec_dtd_tile_t*));

    for( int half = 0; half < 2; half++ ) {
        int first = half ? nt / 2 : 0, last = half ? nt : nt / 2;

        for( int k = first; k < last; k++ ) {
            int home = k % world;
            parsec_dtd_tile_t *circuit = tiles[3 * k]     = parsec_dtd_tile_new(dtd_tp, home);
            parsec_dtd_tile_t *params  = tiles[3 * k + 1] = parsec_dtd_tile_new(dtd_tp, home);
            parsec_dtd_tile_t *result  = tiles[3 * k + 2] = parsec_dtd_tile_new(dtd_tp, qrank);

            parsec_dtd_insert_task_with_task_class(dtd_tp, prep_tc, 0, PARSEC_DEV_CPU,
                                                   PARSEC_AFFINITY, &home,
                                                   PARSEC_DTD_EMPTY_FLAG, circuit,
                                                   PARSEC_DTD_EMPTY_FLAG, params,
                                                   PARSEC_DTD_EMPTY_FLAG, &k,
                                                   PARSEC_DTD_ARG_END);
            parsec_dtd_insert_task_with_task_class(dtd_tp, qexec_tc, 0, PARSEC_DEV_QPU,
                                                   PARSEC_AFFINITY, &qrank,
                                                   PARSEC_DTD_EMPTY_FLAG, circuit,
                                                   PARSEC_DTD_EMPTY_FLAG, params,
                                                   PARSEC_DTD_EMPTY_FLAG, result,
                                                   PARSEC_DTD_ARG_END);
            if( home == qrank ) {
                parsec_dtd_insert_task_with_task_class(dtd_tp, scribble_tc, 0, PARSEC_DEV_CPU,
                                                       PARSEC_DTD_EMPTY_FLAG, circuit,
                                                       PARSEC_DTD_ARG_END);
            }
            parsec_dtd_insert_task_with_task_class(dtd_tp, check_tc, 0, PARSEC_DEV_CPU,
                                                   PARSEC_AFFINITY, &home,
                                                   PARSEC_DTD_EMPTY_FLAG, result,
                                                   PARSEC_DTD_EMPTY_FLAG, &k,
                                                   PARSEC_DTD_ARG_END);
        }
        if( 0 == half ) {
            /* The second half is to be inserted while tasks of the first are
             * with the QPU service. That needs somebody to run the first
             * half meanwhile: a worker other than this thread. A rank bound
             * to a single core (the default of some MPI launchers) has none,
             * and there the tasks simply run once this thread waits. */
            int workers = 0;
            for( int p = 0; p < parsec->nb_vp; p++ ) workers += parsec->virtual_processes[p]->nb_cores;
            if( (rank == qrank) && (workers > 1) ) {
                parsec_qpu_stats_t stats;
                int tries = 0;
                do {
                    usleep(1000);
                    parsec_qpu_service_stats(&stats);
                } while( (0 == stats.tasks_submitted) && (++tries < 5000) );
                if( stats.tasks_submitted <= stats.tasks_completed ) {
                    fprintf(stderr, "no quantum task was outstanding when the second half was inserted "
                            "(%llu submitted, %llu completed)\n",
                            (unsigned long long)stats.tasks_submitted, (unsigned long long)stats.tasks_completed);
                    nb_errors++;
                }
                if( verbose ) {
                    printf("inserting the second half with %llu quantum task(s) submitted and %llu completed\n",
                           (unsigned long long)stats.tasks_submitted, (unsigned long long)stats.tasks_completed);
                }
            } else {
                usleep(5000);
            }
        }
    }

    for( int i = 0; i < 3 * nt; i++ ) {
        parsec_dtd_data_flush(dtd_tp, tiles[i]);
    }

    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    nb_errors += qpu_test_check_service(rank == qrank, nt, 0, verbose && (rank == qrank));

    errors    = qpu_test_sum(parsec, nb_errors);
    checked   = qpu_test_sum(parsec, nb_checked);
    scribbled = qpu_test_sum(parsec, nb_scribbled);
    if( checked != nt ) {
        if( 0 == rank ) fprintf(stderr, "%ld check tasks ran, expected %d\n", checked, nt);
        errors++;
    }
    {
        /* One per chain that lives on the QPU's rank. */
        int expected = 0;

        for( int k = 0; k < nt; k++ ) expected += ((k % world) == qrank);
        if( scribbled != expected ) {
            if( 0 == rank ) fprintf(stderr, "%ld scribble tasks ran, expected %d\n", scribbled, expected);
            errors++;
        }
    }

    for( int i = 0; i < 3 * nt; i++ ) {
        parsec_dtd_tile_release(tiles[i]);
    }
    free(tiles);

    parsec_dtd_task_class_release(dtd_tp, prep_tc);
    parsec_dtd_task_class_release(dtd_tp, qexec_tc);
    parsec_dtd_task_class_release(dtd_tp, scribble_tc);
    parsec_dtd_task_class_release(dtd_tp, check_tc);

    /* Frees the datatype of the arena as well. */
    parsec_dtd_free_arena_datatype(parsec, TILE);
    parsec_taskpool_free(dtd_tp);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    if( 0 == rank ) {
        printf("dtd_qpu_chain: %d chains on %d rank(s): %s\n", nt, world, errors ? "FAILED" : "passed");
    }
    return errors ? 1 : 0;
}
