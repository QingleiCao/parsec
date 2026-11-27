/**
 * Copyright (c) 2013-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024-2026 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/class/dequeue.h"

#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/sched_local_queues_utils.h"
#include "parsec/mca/sched/lhq/sched_lhq.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/papi_sde.h"

/**
 * Module functions
 */
static int sched_lhq_install(parsec_context_t* master);
static int sched_lhq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t*
sched_lhq_select(parsec_execution_stream_t *es,
                 int32_t* distance);
static int flow_lhq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static void sched_lhq_remove(parsec_context_t* master);

const parsec_sched_module_t parsec_sched_lhq_module = {
    &parsec_sched_lhq_component,
    {
        sched_lhq_install,
        flow_lhq_init,
        sched_lhq_schedule,
        sched_lhq_select,
        NULL,
        sched_lhq_remove
    }
};

static int sched_lhq_install( parsec_context_t *master )
{
    (void)master;
    return PARSEC_SUCCESS;
}

static int lhq_core_group(int level, const parsec_execution_stream_t *es)
{
    int core_id = es->core_id >= 0 ? es->core_id : es->th_id;
    int group = parsec_hwloc_master_id(level, core_id);
    if( group < 0 )
        parsec_fatal("lhq scheduler requires a working hwloc");
    return group;
}

/**
 * The scheduler object of a thread: the queues of the hierarchy, the queues
 * of the other threads that this thread does not share with them, and the
 * owner of each queue of the hierarchy.
 *
 * A task reaches a level that several threads share only when the queues
 * below it are full, and the queues at the bottom of the hierarchy belong to
 * one thread (or to the threads of one core). Without the second list, a
 * thread that stops selecting tasks keeps the ones in its own queues from
 * every other thread: for instance the thread that manages a GPU, which stays
 * in the manager as long as the device has work, while the tasks that would
 * feed the device wait in its queues.
 */
typedef struct {
    parsec_mca_sched_local_queues_scheduler_object_t queues;
    int                 nb_foreign_queues;
    int                 next_foreign_queue;  /**< the one the next visit looks at */
    parsec_hbbuffer_t **foreign_queues;      /**< nearest thread first */
    int owner[];  /* VP worker owning each hierarchical queue. */
} lhq_scheduler_object_t;

#define LHQ_SCHEDULER_OBJECT(es) ((lhq_scheduler_object_t*)(es)->scheduler_object)

/**
 * List, for the thread at position t of vp, the queues of the other threads
 * of vp that it does not share with them. The threads it shares the lowest
 * level with come first, starting with the one after t, then those of the
 * next level, and so on. A queue that several of these threads share is
 * listed once.
 */
static void sched_lhq_list_foreign_queues(parsec_vp_t *vp, int t)
{
    lhq_scheduler_object_t *mine = LHQ_SCHEDULER_OBJECT(vp->execution_streams[t]);
    int nb_levels = mine->queues.nb_hierarch_queues;
    int shared, k, idx, i, n = 0;

    mine->foreign_queues = (parsec_hbbuffer_t**)malloc(((size_t)vp->nb_cores * nb_levels + 1) *
                                                       sizeof(parsec_hbbuffer_t*));
    for(shared = 1; shared <= nb_levels; shared++) {
        for(k = 1; k < vp->nb_cores; k++) {
            lhq_scheduler_object_t *other = LHQ_SCHEDULER_OBJECT(vp->execution_streams[(t + k) % vp->nb_cores]);

            /* The levels are nested: once two threads share a queue, they
             * share the queues of all the levels above it too. */
            for(idx = 0; (idx < nb_levels) &&
                         (other->queues.hierarch_queues[idx] != mine->queues.hierarch_queues[idx]); idx++) /* nothing */;
            if( idx != shared ) continue;
            for(idx = 0; idx < shared; idx++) {
                parsec_hbbuffer_t *queue = other->queues.hierarch_queues[idx];
                for(i = 0; (i < n) && (mine->foreign_queues[i] != queue); i++) /* nothing */;
                if( i == n ) mine->foreign_queues[n++] = queue;
            }
        }
    }
    mine->nb_foreign_queues = n;
}

static int flow_lhq_init(parsec_execution_stream_t* ces, struct parsec_barrier_t* barrier)
{
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj = NULL;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    int t;

    vp = ces->virtual_process;

    /* Due to the complexity of the hierarchy, building the LHQ is centralized */
    if( ces->th_id == 0 ) {

        int levels = parsec_hwloc_nb_levels();
        int *group = malloc(vp->nb_cores * sizeof(*group));

        for(t = 0; t < vp->nb_cores; t++) {
            /* First of all, we allocate the scheduling object memory for all threads */
            es = vp->execution_streams[t];

            lhq_scheduler_object_t *lhq = calloc(1, sizeof(*lhq) + levels * sizeof(lhq->owner[0]));
            es->scheduler_object = lhq;
            sched_obj = &lhq->queues;

            if( es->th_id == 0 ) {
                assert(t == 0);
                sched_obj->system_queue = (parsec_dequeue_t*)malloc(sizeof(parsec_dequeue_t));
                PARSEC_OBJ_CONSTRUCT(sched_obj->system_queue, parsec_dequeue_t);
            } else {
                assert(t > 0);
                sched_obj->system_queue = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[0])->system_queue;
            }

            sched_obj->nb_hierarch_queues = levels;
            sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );
        }

        for(int level = 0; level < levels; level++) {
            int idx = levels - 1 - level;
            int max_group = 0;
            for(t = 0; t < vp->nb_cores; t++) {
                group[t] = lhq_core_group(level, vp->execution_streams[t]);
                if( group[t] > max_group ) max_group = group[t];
            }
            /* Group IDs are OS processor IDs and can exceed the VP worker count. */
            struct { int owner, workers; } *groups = calloc((size_t)max_group + 1, sizeof(*groups));
            for(t = 0; t < vp->nb_cores; t++) {
                if( groups[group[t]].workers++ == 0 )
                    groups[group[t]].owner = t;
            }

            for(t = 0; t < vp->nb_cores; t++) {
                es = vp->execution_streams[t];
                lhq_scheduler_object_t *lhq = es->scheduler_object;
                int owner = groups[group[t]].owner;
                lhq->owner[idx] = owner;
                sched_obj = &lhq->queues;
                if( t == owner ) {
                    int nbcores = groups[group[t]].workers;
                    int queue_size = 96 * (level+1) / nbcores;
                    if( queue_size < nbcores ) queue_size = nbcores;

                    /* The master(s) create the shared queues */
                    sched_obj->hierarch_queues[idx] =
                            parsec_hbbuffer_new( queue_size, nbcores,
                                                 level == 0 ? parsec_mca_sched_push_in_system_queue_wrapper : parsec_mca_sched_push_in_buffer_wrapper,
                                                 level == 0 ? (void*)sched_obj : (void*)sched_obj->hierarch_queues[idx+1]);
                    sched_obj->hierarch_queues[idx]->assoc_core_num = ces->virtual_process->vp_id * vp->nb_cores + t; // stored for PINS
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "schedHQ %d: \tcreates hbbuffer of size %d (ideal %d) for level %d stored in %d: %p (parent: %p -- %s)",
                                         es->th_id, queue_size, nbcores,
                                         level, idx, sched_obj->hierarch_queues[idx],
                                         level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1],
                                         level == 0 ? "System queue" : "upper level hhbuffer");
                } else {
                    sched_obj->hierarch_queues[idx] = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[owner])->hierarch_queues[idx];
                }
            }
            free(groups);
        }
        free(group);
        for(t = 0; t < vp->nb_cores; t++) {
            /* Last, the default task queue is the first hierarchical queue of each es */
            es = vp->execution_streams[t];
            sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)es->scheduler_object;
            sched_obj->task_queue = sched_obj->hierarch_queues[0];
        }
        for(t = 0; t < vp->nb_cores; t++) {
            sched_lhq_list_foreign_queues(vp, t);
        }

#if defined(PARSEC_PAPI_SDE)
        {
            char event_name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
            int t;
            parsec_vp_t *vp;
            parsec_execution_stream_t *es;
            parsec_mca_sched_local_queues_scheduler_object_t* sched_obj;
            vp = ces->virtual_process;
        
            snprintf(event_name, 256, "SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LHQ",
                     vp->vp_id);
            parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                         PAPI_SDE_int, (papi_sde_fptr_t)parsec_mca_sched_system_queue_length, vp);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=LHQ", PAPI_SDE_SUM);
            
            for(t = 0; t < vp->nb_cores; t++) {
                es = vp->execution_streams[t];
                sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)es->scheduler_object;
                
                for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                    int idx = sched_obj->nb_hierarch_queues - 1 - level;
                    if( ((lhq_scheduler_object_t *)es->scheduler_object)->owner[idx] == t ) {
                        snprintf(event_name, 256, "SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LHQ",
                                 es->virtual_process->vp_id, idx);
                        parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                                     PAPI_SDE_int, (papi_sde_fptr_t)parsec_hbbuffer_approx_occupency,
                                                     sched_obj->hierarch_queues[idx]);
                        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
                        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=LHQ", PAPI_SDE_SUM);
                    }
                }
            }
        }
#endif
    }

    /* All threads wait here until the main one has completed the build */
    parsec_barrier_wait(barrier);

    return PARSEC_SUCCESS;
}

static parsec_task_t*
sched_lhq_select(parsec_execution_stream_t *es,
                 int32_t* distance)
{
    parsec_task_t *task = NULL;
    int i;

    task = (parsec_task_t*)parsec_hbbuffer_pop_best(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                                                    parsec_execution_context_priority_comparator);
    if( NULL != task ) {
        *distance = 0;
        return task;
    }
    for(i = 0; i <  PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues; i++ ) {
        task = (parsec_task_t*)parsec_hbbuffer_pop_best(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i],
                                                        parsec_execution_context_priority_comparator);
        if( NULL != task ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p",
                                 es->virtual_process->vp_id, es->th_id, task, i, PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i]);
            *distance = i + 1;
            return task;
        }
    }

    task = parsec_mca_sched_pop_from_system_queue_wrapper(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es));
    if( NULL != task ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its system queue %p",
                             es->virtual_process->vp_id, es->th_id, task, PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->system_queue);
        *distance = 1 + PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues;
        return task;
    }

    /* Nothing is left in the queues this thread shares: take a task from the
     * queues of another thread, which may have stopped selecting. A visit
     * looks at one of them, and moves to the next one only when it is empty,
     * so that a thread with nothing to do does not keep reading the queues of
     * all the others. The distance is that of the system queue, so a task
     * that asks to be executed again goes back to a queue that every thread
     * can reach. */
    lhq_scheduler_object_t *obj = LHQ_SCHEDULER_OBJECT(es);
    if( 0 < obj->nb_foreign_queues ) {
        parsec_hbbuffer_t *queue = obj->foreign_queues[obj->next_foreign_queue];
        task = (parsec_task_t*)parsec_hbbuffer_pop_best(queue, parsec_execution_context_priority_comparator);
        if( NULL != task ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in the queue %p of another thread",
                                 es->virtual_process->vp_id, es->th_id, task, queue);
            *distance = 1 + obj->queues.nb_hierarch_queues;
            return task;
        }
        obj->next_foreign_queue = (obj->next_foreign_queue + 1) % obj->nb_foreign_queues;
    }
    return NULL;
}

static int sched_lhq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_hbbuffer_push_all( PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                              (parsec_list_item_t*)new_context,
                              distance );
    return PARSEC_SUCCESS;
}

static void sched_lhq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            sched_obj = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es);

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                if( ((lhq_scheduler_object_t *)es->scheduler_object)->owner[idx] == t ) {
                    parsec_hbbuffer_destruct(sched_obj->hierarch_queues[idx]);
                    sched_obj->hierarch_queues[idx] = NULL;
                    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LHQ", vp->vp_id, idx);
                } else {
                    sched_obj->hierarch_queues[idx] = NULL;
                }
            }

            sched_obj->task_queue = NULL;
            if( es->th_id == 0 ) {
                PARSEC_OBJ_DESTRUCT(sched_obj->system_queue);
                free(sched_obj->system_queue);
            }
            sched_obj->system_queue = NULL;
            free(sched_obj->hierarch_queues);
            sched_obj->hierarch_queues = NULL;
            free(LHQ_SCHEDULER_OBJECT(es)->foreign_queues);
            LHQ_SCHEDULER_OBJECT(es)->foreign_queues = NULL;
            LHQ_SCHEDULER_OBJECT(es)->nb_foreign_queues = 0;

            free(es->scheduler_object);
            es->scheduler_object = NULL;
        }
        PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LHQ", p);
    }
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::SCHED=LHQ");
}
