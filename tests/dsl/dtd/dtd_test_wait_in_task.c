/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * One task waits inside its body until all the other tasks of the taskpool
 * have been executed. The main thread inserts every task, so they all become
 * ready in its queues, and it executes the waiting task first, as that one
 * has the highest priority. The other tasks can then only be executed by the
 * other threads, which therefore have to be able to take them from the queues
 * of a thread that stopped selecting tasks, as the thread that manages a GPU
 * does. The waiting task gives up after a while, so a scheduler that does not
 * let them fails instead of hanging.
 */

#include <time.h>

#include "parsec.h"
#include "parsec/sys/atomic.h"
#include "parsec/mca/device/device.h"
#include "parsec/utils/debug.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "tests/tests_runtime.h"

#define NB_OTHER_TASKS 64
#define WAIT_SECONDS   20

static int32_t nb_executed;
static int     gave_up;
static int     nb_executed_while_waiting;

static int
other_task(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;
    parsec_atomic_fetch_inc_int32(&nb_executed);
    return PARSEC_HOOK_RETURN_DONE;
}

static int
waiting_task(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    time_t deadline = time(NULL) + WAIT_SECONDS;

    (void)es; (void)this_task;
    while( NB_OTHER_TASKS != *(volatile int32_t*)&nb_executed ) {
        if( time(NULL) > deadline ) {
            nb_executed_while_waiting = *(volatile int32_t*)&nb_executed;
            gave_up = 1;
            break;
        }
    }
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    parsec_taskpool_t *tp;
    int rc, ret = 0, nb_cores;

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    nb_cores = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_CORES);
    if( nb_cores < 2 ) {
        parsec_warning("This test needs at least 2 cores, it was given %d. "
                       "Raise `runtime_num_cores` or widen the cpuset.", nb_cores);
        rc = parsec_tests_context_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
        return 1;
    }

    tp = parsec_dtd_taskpool_new();
    rc = parsec_context_add_taskpool(parsec, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_dtd_insert_task(tp, waiting_task, 1, PARSEC_DEV_CPU, "WAIT", PARSEC_DTD_ARG_END);
    for( int i = 0; i < NB_OTHER_TASKS; i++ ) {
        parsec_dtd_insert_task(tp, other_task, 0, PARSEC_DEV_CPU, "OTHER", PARSEC_DTD_ARG_END);
    }

    rc = parsec_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if( gave_up ) {
        parsec_warning("The waiting task gave up after %d s: %d of the %d other tasks were executed "
                       "while it waited", WAIT_SECONDS, nb_executed_while_waiting, NB_OTHER_TASKS);
        ret = 1;
    }
    if( NB_OTHER_TASKS != nb_executed ) {
        parsec_warning("%d of the %d other tasks were executed", nb_executed, NB_OTHER_TASKS);
        ret = 1;
    }

    parsec_taskpool_free(tp);
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
    return ret;
}
