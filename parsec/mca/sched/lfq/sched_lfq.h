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
 * the others wait where sched_lfq_overflow_refill says. 1 takes one task per
 * visit.
 */
extern int parsec_sched_lfq_overflow_batch;
/**
 * How many times a thread tries again the lock of the overflow queue when the
 * queue holds tasks but another thread is in it (MCA parameter
 * sched_lfq_overflow_spins). 0 gives up at the first failure.
 */
extern int parsec_sched_lfq_overflow_spins;
/**
 * Where the tasks of a batch wait, beyond the one that is executed (MCA
 * parameter sched_lfq_overflow_refill).
 */
#define PARSEC_SCHED_LFQ_REFILL_NONE         0  /**< in the bounded buffer of the thread, which takes at most
                                                 *   half of its size that way; the other threads steal them
                                                 *   like any task of that buffer */
#define PARSEC_SCHED_LFQ_REFILL_BEFORE_STEAL 1  /**< in a list of the thread, that the thread empties before it
                                                 *   looks into the buffers of the others; the other threads
                                                 *   take from it only once the overflow queue is empty */
#define PARSEC_SCHED_LFQ_REFILL_AFTER_STEAL  2  /**< in the same list, that the thread comes to after it found
                                                 *   nothing in the buffers of the others */
extern int parsec_sched_lfq_overflow_refill;


END_C_DECLS
#endif /* MCA_SCHED_LFQ_H */
