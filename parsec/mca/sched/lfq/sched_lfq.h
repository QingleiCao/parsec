/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

/**
 * @file
 *
 * Local Flat Queue Scheduler
 *
 */


#ifndef MCA_SCHED_LFQ_H
#define MCA_SCHED_LFQ_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/sched/sched.h"


BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_sched_base_component_t parsec_sched_lfq_component;
PARSEC_DECLSPEC extern const parsec_sched_module_t parsec_sched_lfq_module;
/* static accessor */
mca_base_component_t *sched_lfq_static_component(void);

/**
 * The largest number of tasks a thread takes from the overflow queue in one
 * visit (MCA parameter sched_lfq_overflow_batch). The first one is executed,
 * the others wait in a list of the thread. The thread empties that list before
 * it looks into the queues of the other threads; the other threads take from
 * it once the overflow queue is empty, or take it whole when the thread stopped
 * using it. 1 takes one task per visit.
 */
extern int parsec_sched_lfq_overflow_batch;
/**
 * How many times a thread tries again the lock of the overflow queue when the
 * queue holds tasks but another thread is in it (MCA parameter
 * sched_lfq_overflow_spins). 0 gives up at the first failure.
 */
extern int parsec_sched_lfq_overflow_spins;

END_C_DECLS
#endif /* MCA_SCHED_LFQ_H */
