/**
 * Copyright (c) 2013-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
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
#include "parsec/mca/sched/pbq/sched_pbq.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"

#if defined(PARSEC_PROF_TRACE) && 0
#define TAKE_TIME(ES_PROFILE, KEY, ID)  PARSEC_PROFILING_TRACE((ES_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(ES_PROFILE, KEY, ID) do {} while(0)
#endif

/**
 * Module functions
 */
static int sched_pbq_install(parsec_context_t* master);
static int sched_pbq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t *sched_pbq_select(parsec_execution_stream_t *es,
                                                    int32_t* distance);
static int flow_pbq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static void sched_pbq_remove(parsec_context_t* master);

const parsec_sched_module_t parsec_sched_pbq_module = {
    &parsec_sched_pbq_component,
    {
        sched_pbq_install,
        flow_pbq_init,
        sched_pbq_schedule,
        sched_pbq_select,
        NULL,
        sched_pbq_remove
    }
};

static int sched_pbq_install( parsec_context_t *master )
{
    return 0;
}

static int flow_pbq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    local_queues_scheduler_object_t *sched_obj = NULL;
    int nq = 1, hwloc_levels;
    parsec_vp_t *vp = es->virtual_process;
    uint32_t queue_size = 0;

    sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
    es->scheduler_object = sched_obj;

    if( es->th_id == 0 ) {
        sched_obj->system_queue = (parsec_dequeue_t*)malloc(sizeof(parsec_dequeue_t));
        sched_obj->system_queue = OBJ_NEW(parsec_dequeue_t);
    }

    sched_obj->nb_hierarch_queues = vp->nb_cores;
    sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );
    queue_size = 4 * vp->nb_cores;

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    parsec_barrier_wait(barrier);

    /* Get the flow 0 system queue and store it locally */
    sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_streams[0])->system_queue;

    /* Each thread creates its own "local" queue, connected to the shared dequeue */
    sched_obj->task_queue = parsec_hbbuffer_new( queue_size, 1, push_in_queue_wrapper,
                                                (void*)sched_obj->system_queue);
    sched_obj->hierarch_queues[0] = sched_obj->task_queue;

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    parsec_barrier_wait(barrier);

    nq = 1;
#if defined(PARSEC_HAVE_HWLOC)
    hwloc_levels = parsec_hwloc_nb_levels();
#else
    hwloc_levels = -1;
#endif

    /* Handle the case when HWLOC is present but cannot compute the hierarchy,
     * as well as the casewhen HWLOC is not present
     */
    if( hwloc_levels == -1 ) {
        for( ; nq < sched_obj->nb_hierarch_queues; nq++ ) {
            sched_obj->hierarch_queues[nq] =
                LOCAL_QUEUES_OBJECT(vp->execution_streams[(es->th_id + nq) % vp->nb_cores])->task_queue;
        }
#if defined(PARSEC_HAVE_HWLOC)
    }
    else {
        /* Then, they know about all other queues, from the closest to the farthest */
        for(int level = 0; level <= hwloc_levels; level++) {
            for(int id = (es->th_id + 1) % vp->nb_cores;
                id != es->th_id;
                id = (id + 1) %  vp->nb_cores) {
                int d;
                d = parsec_hwloc_distance(es->th_id, id);
                if( d == 2*level || d == 2*level + 1 ) {
                    sched_obj->hierarch_queues[nq] = LOCAL_QUEUES_OBJECT(vp->execution_streams[id])->task_queue;
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "%d of %d: my %d-preferred queue is the task queue of %d (%p)",
                           es->th_id, es->virtual_process->vp_id, nq, id, sched_obj->hierarch_queues[nq]);
                    nq++;
                    if( nq == sched_obj->nb_hierarch_queues )
                        break;
                }
            }
            if( nq == sched_obj->nb_hierarch_queues )
                break;
        }
        assert( nq == sched_obj->nb_hierarch_queues );
#endif
    }

    return 0;
}

static parsec_task_t*
sched_pbq_select( parsec_execution_stream_t *es,
                  int32_t* distance)
{
    parsec_task_t *task = NULL;
    int i;
    task = (parsec_task_t*)parsec_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(es)->task_queue,
                                                                       parsec_execution_context_priority_comparator);
    if( NULL != task ) {
        *distance = 0;
        return task;
    }
    for(i = 0; i <  LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues; i++ ) {
        task = (parsec_task_t*)parsec_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i],
                                                                           parsec_execution_context_priority_comparator);
        if( NULL != task ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p",
                    es->virtual_process->vp_id, es->th_id, task, i, LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i]);
            *distance = i + 1;
            return task;
        }
    }

    task = (parsec_task_t *)parsec_dequeue_try_pop_front(LOCAL_QUEUES_OBJECT(es)->system_queue);
    if( NULL != task ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its system queue %p",
                es->virtual_process->vp_id, es->th_id, task, LOCAL_QUEUES_OBJECT(es)->system_queue);
        *distance = 1 + LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues;
    }
    return task;}

static int sched_pbq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_hbbuffer_push_all_by_priority( LOCAL_QUEUES_OBJECT(es)->task_queue,
                                          (parsec_list_item_t*)new_context,
                                          distance);
    return 0;
}

static void sched_pbq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            sched_obj = LOCAL_QUEUES_OBJECT(es);

            if( es->th_id == 0 ) {
                OBJ_DESTRUCT( sched_obj->system_queue );
                free( sched_obj->system_queue );
            }
            sched_obj->system_queue = NULL;

            parsec_hbbuffer_destruct( sched_obj->task_queue );
            sched_obj->task_queue = NULL;

            free(sched_obj->hierarch_queues);
            sched_obj->hierarch_queues = NULL;

            free(es->scheduler_object);
            es->scheduler_object = NULL;
        }
    }
}
