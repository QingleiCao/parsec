/**
 * Copyright (c) 2013-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
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
#include "parsec/mca/sched/lfq/sched_lfq.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/papi_sde.h"

/**
 * Module functions
 */
static int sched_lfq_install(parsec_context_t* master);
static int sched_lfq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t*
sched_lfq_select(parsec_execution_stream_t *es,
                 int32_t* distance);
static void sched_lfq_remove(parsec_context_t* master);
static int flow_lfq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);

const parsec_sched_module_t parsec_sched_lfq_module = {
    &parsec_sched_lfq_component,
    {
        sched_lfq_install,
        flow_lfq_init,
        sched_lfq_schedule,
        sched_lfq_select,
        NULL,
        sched_lfq_remove
    }
};

/**
 * What lfq adds to the scheduler object of the local queues: the tasks that a
 * thread brought back from the system queue in one visit and has not executed
 * yet. They are kept apart from the bounded buffer so that a thread that runs
 * out of tasks goes to the system queue for a batch of its own, rather than
 * take, one at a time, what is left of the batches of the others.
 */
typedef struct {
    parsec_mca_sched_local_queues_scheduler_object_t super;
    parsec_list_t  *refill;            /* The tasks of the last batch of this thread that wait */
    parsec_list_t **hierarch_refills;  /* The refill lists of all threads, in the order of super.hierarch_queues */
    parsec_list_item_t **refill_seen;  /* For each of these lists, its first task when this thread last looked at it */
    int             refill_next;       /* The list this thread looks at on its next way to the system queue */
} sched_lfq_object_t;

#define SCHED_LFQ_OBJECT(es) ((sched_lfq_object_t*)(es)->scheduler_object)
#define SCHED_LFQ_CACHE_LINE 64

/* The owner of a refill list writes to it for every task it takes out. The list
 * gets cache lines of its own: what the allocator would put next to it, the
 * bounded buffer of the same thread, is read by every thread that looks for
 * work. */
static parsec_list_t* sched_lfq_refill_new(void)
{
    size_t size = (sizeof(parsec_list_t) + SCHED_LFQ_CACHE_LINE - 1) & ~((size_t)SCHED_LFQ_CACHE_LINE - 1);
    void *list = NULL;
    int rc = posix_memalign(&list, SCHED_LFQ_CACHE_LINE, size);
    assert( (0 == rc) && (NULL != list) ); (void)rc;
    PARSEC_OBJ_CONSTRUCT(list, parsec_list_t);
    return (parsec_list_t*)list;
}

static int sched_lfq_install( parsec_context_t *master )
{
    (void)master;
    return PARSEC_SUCCESS;
}

static int flow_lfq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj = NULL;
    sched_lfq_object_t *lfq = NULL;
    int nq, hwloc_levels;
    uint32_t queue_size;
    parsec_vp_t* vp;

    vp = es->virtual_process;

    /* Every flow creates its own local object */
    lfq = (sched_lfq_object_t*)calloc(1, sizeof(sched_lfq_object_t));
    sched_obj = &lfq->super;
    es->scheduler_object = lfq;
    if( 0 == es->th_id ) {  /* And flow 0 creates the system_queue */
        sched_obj->system_queue = PARSEC_OBJ_NEW(parsec_dequeue_t);
    }

    sched_obj->nb_hierarch_queues = vp->nb_cores;
    sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );
    queue_size = vp->nb_cores * 4;
    lfq->refill = sched_lfq_refill_new();
    lfq->hierarch_refills = (parsec_list_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_list_t*) );
    lfq->refill_seen = (parsec_list_item_t **)calloc(sched_obj->nb_hierarch_queues, sizeof(parsec_list_item_t*) );
    lfq->refill_next = 1;

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    parsec_barrier_wait(barrier);

    /* Get the flow 0 system queue and store it locally */
    sched_obj->system_queue = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[0])->system_queue;

    /* Each thread creates its own "local" queue, connected to the shared dequeue */
    sched_obj->task_queue = parsec_hbbuffer_new( queue_size, 1, parsec_mca_sched_push_in_system_queue_wrapper,
                                                (void*)sched_obj );
    sched_obj->hierarch_queues[0] = sched_obj->task_queue;
    lfq->hierarch_refills[0] = lfq->refill;

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
     * as well as the case when HWLOC is missing.
     */
    if( hwloc_levels == -1 ) {
        for( ; nq < sched_obj->nb_hierarch_queues; nq++ ) {
            sched_obj->hierarch_queues[nq] =
                PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[(es->th_id + nq) % vp->nb_cores])->task_queue;
            lfq->hierarch_refills[nq] = SCHED_LFQ_OBJECT(vp->execution_streams[(es->th_id + nq) % vp->nb_cores])->refill;
        }
#if defined(PARSEC_HAVE_HWLOC)
    } else {
        /* Then, they know about all other queues, from the closest to the farthest */
        for(int level = 0; level <= hwloc_levels; level++) {
            for(int id = (es->th_id + 1) % vp->nb_cores;
                id != es->th_id;
                id = (id + 1) %  vp->nb_cores) {
                int d;
                d = parsec_hwloc_distance(es->th_id, id);
                if( d == 2*level || d == 2*level + 1 ) {
                    sched_obj->hierarch_queues[nq] = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[id])->task_queue;
                    lfq->hierarch_refills[nq] = SCHED_LFQ_OBJECT(vp->execution_streams[id])->refill;
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "%d of %d: my %d preferred queue is the task queue of %d (%p)",
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

#if defined(PARSEC_PAPI_SDE)
    if( 0 == es->th_id ) {
        char event_name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
        int thid;
        parsec_vp_t *vp;
        snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN,
                 "SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LFQ", es->virtual_process->vp_id);
        parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                     PAPI_SDE_int, (papi_sde_fptr_t)parsec_mca_sched_system_queue_length, es->virtual_process);
        parsec_papi_sde_add_counter_to_group(event_name,
                                      "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
        parsec_papi_sde_add_counter_to_group(event_name,
                                      "SCHEDULER::PENDING_TASKS::SCHED=LFQ", PAPI_SDE_SUM);
        vp = es->virtual_process;
        for(thid = 0; thid < vp->nb_cores; thid++) {
            snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN,
                     "SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LFQ", vp->vp_id, thid);
            parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                         PAPI_SDE_int, (papi_sde_fptr_t)parsec_hbbuffer_approx_occupency,
                                         PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[thid])->task_queue);
            parsec_papi_sde_add_counter_to_group(event_name,
                                          "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
            parsec_papi_sde_add_counter_to_group(event_name,
                                          "SCHEDULER::PENDING_TASKS::SCHED=LFQ", PAPI_SDE_SUM);
        }
    }
#endif

    return PARSEC_SUCCESS;
}

/* A task of the refill list of a thread, the calling one or another. These
 * tasks are still counted with the system queue they came from. */
static inline parsec_task_t*
sched_lfq_pop_refill(parsec_mca_sched_local_queues_scheduler_object_t *sched_obj,
                     parsec_list_t *refill, int own)
{
    parsec_task_t *task = (parsec_task_t*)(own ? parsec_list_pop_front(refill)
                                               : parsec_list_try_pop_front(refill));
#if defined(PARSEC_PAPI_SDE)
    if( NULL != task )
        sched_obj->local_system_queue_balance--;
#endif
    (void)sched_obj;
    return task;
}

/* The first task of a ring is for the calling thread to execute, the others
 * wait in its refill list. Only this thread adds to its list, and it does so
 * once the list is empty: the list never holds more than a batch. */
static inline parsec_task_t*
sched_lfq_keep_ring(sched_lfq_object_t *lfq, parsec_list_item_t *ring)
{
    parsec_list_item_t *others = parsec_list_item_ring_chop(ring);
    PARSEC_LIST_ITEM_SINGLETON(ring);
    if( NULL != others )
        parsec_list_chain_back(lfq->refill, others);
    return (parsec_task_t*)ring;
}

/* A thread that stopped selecting tasks (it executes a long one, or it manages
 * a device) leaves its refill list as it is, and the other threads keep away
 * from that list as long as the system queue holds tasks. So, on its way to the
 * system queue, a thread looks at the list of one other thread, a different one
 * each time. A list that is in use changes its first task with every task its
 * owner takes: if the first task is the one that was there on the previous
 * look, the list is taken whole. Taking the list of a thread that was merely
 * slow costs nothing but the move, that thread goes to the system queue next. */
static inline parsec_list_item_t*
sched_lfq_take_stalled_refill(sched_lfq_object_t *lfq)
{
    parsec_list_item_t *head, *ring = NULL;
    parsec_list_t *list;
    int v = lfq->refill_next;

    if( lfq->super.nb_hierarch_queues < 2 )
        return NULL;
    lfq->refill_next = (v + 1 < lfq->super.nb_hierarch_queues) ? v + 1 : 1;
    list = lfq->hierarch_refills[v];
    head = (parsec_list_item_t*)PARSEC_LIST_ITERATOR_FIRST(list);
    if( PARSEC_LIST_ITERATOR_END(list) == head ) {
        lfq->refill_seen[v] = NULL;
        return NULL;
    }
    if( head != lfq->refill_seen[v] ) {
        lfq->refill_seen[v] = head;
        return NULL;
    }
    lfq->refill_seen[v] = NULL;
    if( !parsec_atomic_trylock(&list->atomic_lock) )
        return NULL;
    if( head == (parsec_list_item_t*)PARSEC_LIST_ITERATOR_FIRST(list) )
        ring = parsec_list_nolock_unchain(list);
    parsec_atomic_unlock(&list->atomic_lock);
    return ring;
}

static parsec_task_t*
sched_lfq_select(parsec_execution_stream_t *es,
                 int32_t* distance)
{
    sched_lfq_object_t *lfq = SCHED_LFQ_OBJECT(es);
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj = &lfq->super;
    const int refill = parsec_sched_lfq_overflow_refill;
    parsec_list_item_t *ring;
    parsec_task_t *task = NULL;
    int max, nb;

    task = (parsec_task_t*)parsec_hbbuffer_pop_best(sched_obj->task_queue,
                                                    parsec_execution_context_priority_comparator);
    if( NULL != task ) {
        *distance = 0;
        return task;
    }
    if( PARSEC_SCHED_LFQ_REFILL_BEFORE_STEAL == refill ) {
        task = sched_lfq_pop_refill(sched_obj, lfq->refill, 1);
        if( NULL != task ) {
            *distance = 1 + sched_obj->nb_hierarch_queues;
            return task;
        }
    }
    for(int i = 0; i < sched_obj->nb_hierarch_queues; i++ ) {
        task = (parsec_task_t*)parsec_hbbuffer_pop_best(sched_obj->hierarch_queues[i],
                                                        parsec_execution_context_priority_comparator);
        if( NULL != task ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p",
                    es->virtual_process->vp_id, es->th_id, task, i, sched_obj->hierarch_queues[i]);
            *distance = i + 1;
            return task;
        }
    }
    if( PARSEC_SCHED_LFQ_REFILL_AFTER_STEAL == refill ) {
        task = sched_lfq_pop_refill(sched_obj, lfq->refill, 1);
        if( NULL != task ) {
            *distance = 1 + sched_obj->nb_hierarch_queues;
            return task;
        }
    }

    /* The tasks that a stalled thread holds in its list left the system queue
     * before the ones that are in it now */
    if( (PARSEC_SCHED_LFQ_REFILL_NONE != refill) && parsec_sched_lfq_overflow_takeover ) {
        ring = sched_lfq_take_stalled_refill(lfq);
        if( NULL != ring ) {
            task = sched_lfq_keep_ring(lfq, ring);
#if defined(PARSEC_PAPI_SDE)
            sched_obj->local_system_queue_balance--;
#endif
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in a refill list that stalled",
                    es->virtual_process->vp_id, es->th_id, task);
            *distance = 2 + sched_obj->nb_hierarch_queues;
            return task;
        }
    }

    /* Nothing in the bounded buffers: go to the system queue, and bring back
     * several tasks at once. Every thread that is out of tasks comes to this
     * queue, so the fewer the visits the better. */
    max = parsec_sched_lfq_overflow_batch;
    if( PARSEC_SCHED_LFQ_REFILL_NONE == refill ) {
        /* The tasks beyond the first go to the local buffer, which is empty or
         * close to it at this point; keep half of it for the tasks that the
         * first one will make ready. */
        int room = 1 + (int)(sched_obj->task_queue->size / 2);
        if( max > room )
            max = room;
    }
    ring = parsec_mca_sched_pop_chain_from_system_queue_wrapper(sched_obj, max,
                                                                parsec_sched_lfq_overflow_spins, &nb);
    if( NULL != ring ) {
        if( PARSEC_SCHED_LFQ_REFILL_NONE == refill ) {
            parsec_list_item_t *others = parsec_list_item_ring_chop(ring);
            PARSEC_LIST_ITEM_SINGLETON(ring);
            task = (parsec_task_t*)ring;
            if( NULL != others )
                parsec_hbbuffer_push_all(sched_obj->task_queue, others, 0);
        } else {
            task = sched_lfq_keep_ring(lfq, ring);
#if defined(PARSEC_PAPI_SDE)
            sched_obj->local_system_queue_balance += nb - 1;
#endif
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p and %d others in its system queue %p",
                es->virtual_process->vp_id, es->th_id, task, nb - 1, sched_obj->system_queue);
        *distance = 1 + sched_obj->nb_hierarch_queues;
        return task;
    }

    /* The system queue is empty: what is left waits in the refill lists, take
     * it from there one task at a time. While the system queue holds tasks the
     * lists of the others are left alone (unless they stalled, see above), even
     * if the lock of the queue stayed busy: the thread comes back to the queue. */
    if( (PARSEC_SCHED_LFQ_REFILL_NONE != refill) &&
        parsec_list_nolock_is_empty((parsec_list_t*)sched_obj->system_queue) ) {
        for(int i = 1; i < sched_obj->nb_hierarch_queues; i++ ) {
            task = sched_lfq_pop_refill(sched_obj, lfq->hierarch_refills[i], 0);
            if( NULL != task ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in the %d-preferred refill list %p",
                        es->virtual_process->vp_id, es->th_id, task, i, lfq->hierarch_refills[i]);
                *distance = 1 + sched_obj->nb_hierarch_queues + i;
                return task;
            }
        }
    }
    return NULL;
}

static int sched_lfq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_hbbuffer_push_all(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                             (parsec_list_item_t*)new_context,
                             distance);
    return PARSEC_SUCCESS;
}

static void sched_lfq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            if (es != NULL) {
                sched_obj = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es);

                if( es->th_id == 0 ) {
                    PARSEC_OBJ_DESTRUCT( sched_obj->system_queue );
                    free( sched_obj->system_queue );
                }
                sched_obj->system_queue = NULL;

                parsec_hbbuffer_destruct( sched_obj->task_queue );
                sched_obj->task_queue = NULL;

                free(sched_obj->hierarch_queues);
                sched_obj->hierarch_queues = NULL;

                PARSEC_OBJ_DESTRUCT( SCHED_LFQ_OBJECT(es)->refill );
                free( SCHED_LFQ_OBJECT(es)->refill );
                free( SCHED_LFQ_OBJECT(es)->hierarch_refills );
                free( SCHED_LFQ_OBJECT(es)->refill_seen );

                free(es->scheduler_object);
                es->scheduler_object = NULL;
            }
            // else the scheduler wasn't really initialized anyway
            PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LFQ", vp->vp_id, t);
        }
        PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LFQ", p);
    }
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::SCHED=LFQ");
}
