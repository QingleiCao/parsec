/*
 * Copyright (c) 2013-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "parsec/parsec_config.h"
#include "parsec/runtime.h"

#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/lfq/sched_lfq.h"
#include "parsec/utils/mca_param.h"
#include "parsec/papi_sde.h"

/*
 * Local function
 */
static int sched_lfq_component_query(mca_base_module_t **module, int *priority);
static int sched_lfq_component_register(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const parsec_sched_base_component_t parsec_sched_lfq_component = {

    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_SCHED_BASE_VERSION_2_0_0,

        /* Component name and version */
        "lfq",
        "", /* options */
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        NULL, /*< No open: sched_lfq is always available, no need to check at runtime */
        NULL, /*< No close: open did not allocate any resource, no need to release them */
        sched_lfq_component_query, 
        /*< specific query to return the module and add it to the list of available modules */
        sched_lfq_component_register,
        "", /*< no reserve */
    },
    {
        /* The component has no metadata */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    }
};

mca_base_component_t *sched_lfq_static_component(void)
{
    return (mca_base_component_t *)&parsec_sched_lfq_component;
}

static int sched_lfq_component_query(mca_base_module_t **module, int *priority)
{
    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&parsec_sched_lfq_module;
    *priority = 20;
    *module = (mca_base_module_t *)ptr;
    return MCA_SUCCESS;
}

int parsec_sched_lfq_overflow_batch = 32;
int parsec_sched_lfq_overflow_spins = 256;
int parsec_sched_lfq_overflow_refill = PARSEC_SCHED_LFQ_REFILL_BEFORE_STEAL;

static int sched_lfq_component_register(void)
{
    parsec_mca_param_reg_int_name("sched_lfq", "overflow_batch",
                                  "Largest number of tasks a thread takes from the overflow queue in one visit: "
                                  "the first is executed, the others wait where sched_lfq_overflow_refill says. "
                                  "1 takes one task per visit.",
                                  false, false, parsec_sched_lfq_overflow_batch, &parsec_sched_lfq_overflow_batch);
    if( parsec_sched_lfq_overflow_batch < 1 )
        parsec_sched_lfq_overflow_batch = 1;
    parsec_mca_param_reg_int_name("sched_lfq", "overflow_spins",
                                  "How many times a thread tries again the lock of the overflow queue when the queue "
                                  "holds tasks but another thread is in it. 0 gives up at the first failure.",
                                  false, false, parsec_sched_lfq_overflow_spins, &parsec_sched_lfq_overflow_spins);
    if( parsec_sched_lfq_overflow_spins < 0 )
        parsec_sched_lfq_overflow_spins = 0;
    parsec_mca_param_reg_int_name("sched_lfq", "overflow_refill",
                                  "Where the tasks a thread took from the overflow queue wait, beyond the one it executes. "
                                  "0: in the local queue of the thread (at most half of its size), where the other threads "
                                  "steal them like any other task. 1: in a list of the thread, that it empties before it "
                                  "steals, and that the other threads take from only once the overflow queue is empty. "
                                  "2: in the same list, that the thread comes to after it found nothing to steal.",
                                  false, false, parsec_sched_lfq_overflow_refill, &parsec_sched_lfq_overflow_refill);
    if( (parsec_sched_lfq_overflow_refill < PARSEC_SCHED_LFQ_REFILL_NONE) ||
        (parsec_sched_lfq_overflow_refill > PARSEC_SCHED_LFQ_REFILL_AFTER_STEAL) )
        parsec_sched_lfq_overflow_refill = PARSEC_SCHED_LFQ_REFILL_BEFORE_STEAL;
    PARSEC_PAPI_SDE_DESCRIBE_COUNTER("SCHEDULER::PENDING_TASKS::SCHED=LFQ",
                              "the number of pending tasks for the LFQ scheduler");
    PARSEC_PAPI_SDE_DESCRIBE_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=<VPID>/<QID>::SCHED=LFQ",
                              "the number of pending tasks that end up in the virtual process <VPID> queue of queue identifier <QID> for the LFQ scheduler");
    return MCA_SUCCESS;
}
