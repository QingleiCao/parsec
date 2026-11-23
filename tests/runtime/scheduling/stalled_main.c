/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "parsec/arena.h"
#include "parsec/datatype.h"
#include "parsec/data_distribution.h"
#include "schedmicro_data.h"
#include "stalled_state.h"
#include "stalled.h"
#include "tests/tests_runtime.h"

int32_t stalled_nb_done = 0;
int32_t stalled_nb_parked = 0;
int32_t stalled_released = 0;
int32_t stalled_nb_done_by_master = 0;
char   *stalled_has_parked = NULL;

/* The test fails by not terminating: stop it instead of waiting for the launcher to do so */
static void stalled_watchdog(int sig)
{
    static const char msg[] = "stalled: threads that wait inside a task still hold tasks that nobody else takes\n";
    (void)sig;
    if( write(STDERR_FILENO, msg, sizeof(msg) - 1) < 0 ) { /* nothing more to do */ }
    _exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    parsec_stalled_taskpool_t *tp;
    parsec_data_collection_t *dcA;
    int rank, world, rc, nb_threads, nt, nf, margin;
    int ncores = 0, watchdog = 60;
    int parsec_argc = 0;
    char **parsec_argv = NULL;
    ptrdiff_t lb, extent;

    for(int a = 1; a < argc; a++) {
        if(strcmp(argv[a], "--") == 0) {
            parsec_argc = argc - a;
            parsec_argv = argv + a;
            break;
        }
        if( (strcmp(argv[a], "-c") == 0) && (a + 1 < argc) ) {
            ncores = atoi(argv[++a]);
            continue;
        }
        if( (strcmp(argv[a], "-w") == 0) && (a + 1 < argc) ) {
            watchdog = atoi(argv[++a]);
            continue;
        }
        fprintf(stderr, "Usage: %s [-c NCORES] [-w SECONDS before the test gives up] [-- <parsec parameters>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    rc = parsec_tests_context_init(ncores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &parsec_argc, &parsec_argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");
    if( 1 != world ) {
        fprintf(stderr, "%s counts the tasks of one process: run it on one process\n", argv[0]);
        (void)parsec_tests_context_fini(&parsec);
        return EXIT_FAILURE;
    }

    nb_threads = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_CORES);
    margin = 8 * nb_threads + 8;         /* more than a thread keeps in its own queue */
    nt = 64 * nb_threads + 2 * margin;
    nf = 128 * nb_threads;               /* more than all the threads can hold at once */
    stalled_has_parked = (char*)calloc(nb_threads, sizeof(char));

    dcA = create_and_distribute_data(rank, world, (nt > nf) ? nt : nf, 1);
    parsec_data_collection_set_key(dcA, "A");

    tp = parsec_stalled_new(nt, nf, margin, dcA);
    /* The datatype is irrelevant, the tasks exchange no data */
    rc = parsec_type_extent(parsec_datatype_uint8_t, &lb, &extent);
    PARSEC_CHECK_ERROR(rc, "parsec_type_extent");
    parsec_arena_datatype_set_type( &tp->arenas_datatypes[PARSEC_stalled_DEFAULT_ADT_IDX],
                                    (size_t)extent, PARSEC_ARENA_ALIGNMENT_SSE,
                                    parsec_datatype_uint8_t );

    signal(SIGALRM, stalled_watchdog);
    alarm(watchdog);

    rc = parsec_context_add_taskpool(parsec, (parsec_taskpool_t*)tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    alarm(0);

    printf("%d threads, %d tasks and %d that ask to be executed again: %d threads waited inside a task, thread 0 completed %d tasks\n",
           nb_threads, nt, nf, stalled_nb_parked, stalled_nb_done_by_master);
    rc = (stalled_nb_done == nt) ? EXIT_SUCCESS : EXIT_FAILURE;
    if( EXIT_SUCCESS != rc )
        fprintf(stderr, "%d tasks completed out of %d\n", stalled_nb_done, nt);

    PARSEC_OBJ_DESTRUCT(&tp->arenas_datatypes[PARSEC_stalled_DEFAULT_ADT_IDX]);
    parsec_taskpool_free((parsec_taskpool_t*)tp);
    free_data(dcA);
    free(stalled_has_parked);

    (void)parsec_tests_context_fini(&parsec);
    return rc;
}
