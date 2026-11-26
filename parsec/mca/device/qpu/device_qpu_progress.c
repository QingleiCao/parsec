/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/constants.h"
#include "parsec/runtime.h"
#include "parsec/scheduling.h"
#include "parsec/execution_stream.h"
#include "parsec/data_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/bindthread.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"
#include "parsec/mca/device/device.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * The QPU service: submission, admission, progress and completion.
 *
 * Who owns a task.
 *   - Until its body calls parsec_qpu_submit(), the worker running it.
 *   - From the moment its record is pushed on the submission stack, this
 *     file: one progress thread at a time, the one of the QPU that holds it,
 *     or whichever holds the lock of the pool while the task waits. The
 *     worker that ran the body returns ASYNC and leaves the task alone.
 *   - From the hand-off on, the scheduler and then whichever worker picks
 *     the task up. No progress thread keeps a pointer to the task, to its
 *     data or to its taskpool past that point: the worker may complete the
 *     task, and the taskpool may terminate and be freed, at any moment.
 *
 * Where it runs. On the ranks that own a QPU. Each QPU of a rank has a
 * service of its own: a backend, opened when the device of that QPU is
 * attached, so that a backend that cannot be used gives no device; and a
 * progress thread, which starts with the first submission of the rank. All
 * of them stop when the context is finalized, before the workers are
 * joined. A submission on a rank that owns no QPU is refused.
 *
 * Which QPU runs a task. The QPUs of a rank share the tasks that wait: the
 * pool. A task is bound to a QPU when that QPU has a place for its job and
 * takes it, not before; a QPU holds no data, so nothing has to be decided
 * earlier. A QPU takes, of the tasks it may run, the first by priority and
 * then by arrival. It may run a task that named it (parsec_qpu_submit_on),
 * and a task that named none when its backend takes the circuit and the
 * shots. So no place stays empty while a task that could use it waits, a
 * faster QPU takes more tasks without anybody estimating anything, and a QPU
 * that stops answering keeps only the tasks it has.
 *   The device layer selected a QPU module for the task before its body ran,
 * always the same one, and added the task's load to it. The body takes that
 * load off again; the QPU that takes the task makes itself the task's device
 * and carries its load until the task completes or goes back to wait. The
 * scheduler takes the load off the task's device when the task completes,
 * as it does for every device.
 *
 * How a task is completed. No progress thread ever calls
 * __parsec_complete_execution(). It marks the task as past its body
 * (PARSEC_TASK_STATUS_COMPLETE) and gives it to the scheduler; the worker that
 * selects it goes straight to the completion in __parsec_task_progress(). So
 * the dependencies are released, the inputs unpinned and the task counted
 * out of its taskpool on a worker's execution stream, exactly as for a task
 * whose body returned DONE. Nothing here touches a termination detector or
 * a task count.
 *
 * What a failure does. Nothing that goes wrong with a job reaches the
 * scheduler as an error: the task completes normally, and its result says
 * what happened.
 *   - A job that fails, cannot give its results, is refused for good, or
 *     takes longer than its QPU's job_timeout for each task it carries (it
 *     is then cancelled) sends its tasks back to wait, each for a new job
 *     with a new tag, on whichever QPU takes it, until the task has had
 *     1 + max_retries jobs.
 *   - A submission the backend cannot take for now is asked again on the
 *     same QPU, with the same tag, after a back-off that doubles; that QPU
 *     submits nothing else in between, because the provider may have the
 *     job all the same. When that has lasted the QPU's unavailable_timeout,
 *     the submission is given up on at the backend (which cancels the job if
 *     it is there), and the QPU is taken for down. A task of that job, and a
 *     task that waits, completes as "unavailable" when no QPU of the rank
 *     that may run it is up; the others wait for a QPU that is.
 *   - A QPU that is down takes the tasks that have nowhere else to go. After
 *     a while it is tried again with one job; if that one is not taken
 *     either, it is given back at once and the QPU is left alone for twice
 *     as long.
 *   - A final state for a job that is not in flight is counted and ignored:
 *     the first final state of a job is the only one acted upon.
 *
 * How tasks share a job. A job carries the circuit of one task, or those of
 * several (max_batch, and what the backend takes per job). No task is held
 * back to fill a job: a task that arrives while a job can be submitted goes
 * at once. Tasks share a job only when they wait together and there are
 * more of them than jobs that can be submitted at that moment, on all the
 * QPUs of the rank; then they are spread evenly over those jobs, in the
 * order they wait in. Each task keeps its own result and completes on its
 * own. Tasks that share a job ask for the same number of shots and are at
 * their first job: a job that fails, with one task or with several, sends
 * each of its tasks back to be retried alone. Sharing a job takes nothing
 * from a task that it had alone: a job of n tasks may stay at the backend n
 * times as long as a job of one before it is given up on.
 *
 * Who wakes whom. A progress thread sleeps in its backend. A QPU shows the
 * places it has (room) only while its thread sleeps or is about to: a thread
 * at work may be inside a call to its backend, which lasts as long as a
 * provider takes to answer, and a task must not wait for that while another
 * QPU sleeps. A body pushes its task, then reads the rooms and wakes a QPU
 * that shows one. A thread that puts tasks into the pool, from the stack or
 * because a job failed, wakes the others that show room, before it calls
 * its backend again. And a thread that is done shows its room first and then
 * looks once more at the stack and at the pool before it sleeps. Whoever
 * comes with a task either sees the room and wakes the thread, or came
 * before the room was shown and is found by that last look: a task is never
 * left beside a free place for longer than it takes a thread to wake up.
 * Nobody is woken for a task it cannot take, so threads do not keep each
 * other awake.
 */

#define QPU_LOAD(PTR)        PARSEC_QPU_LOAD(PTR)
#define QPU_STORE(PTR, VAL)  PARSEC_QPU_STORE(PTR, VAL)

#define QPU_SVC_DOWN      0
#define QPU_SVC_STARTING  1
#define QPU_SVC_UP        2
#define QPU_SVC_FAILED    3

#define QPU_MAX_EVENTS    64
#define QPU_MAX_BACKENDS  8
#define QPU_MAX_QPUS      PARSEC_QPU_MAX_QPUS
/* The longest the service waits before asking an unavailable backend again. */
#define QPU_BACKOFF_MAX_NS  30000000000LL
/* A QPU that is down is left alone for its unavailable_timeout, then for
 * twice that, and so on up to this many times. */
#define QPU_COOLDOWN_MAX  8

/* A task between its body and its completion. */
typedef struct qpu_task_s {
    struct qpu_task_s    *next;        /* submission stack, and lists of tasks to complete */
    parsec_task_t        *task;
    parsec_qpu_request_t  request;
    parsec_qpu_result_t  *result;      /* inside the task's output copy */
    uint64_t              seq;         /* arrival order */
    uint64_t              tag;         /* of the task; a job's tag is that of its first task plus
                                        * the number of the attempt this job is for that task */
    int64_t               t_hook;
    int64_t               t_submit;    /* when its last job was first asked for */
    int32_t               priority;
    int                   vp_id;
    uint32_t              batch;       /* tasks in the last job a backend took for it */
    uint16_t              attempts;    /* jobs the backends took for this task, or refused for good */
    uint16_t              formed;      /* jobs put together for it, taken by a backend or not: the
                                        * low byte of the tag of the next one */
    uint16_t              may_run;     /* the QPUs of this rank that may run it, one bit each */
    int16_t               pin;         /* the QPU it named, among those of this rank; -1 for any */
    int16_t               ran_on;      /* the QPU that had its last job; -1 before the first */
    int16_t               avoid;       /* the QPU whose job for it just failed: its next job goes
                                        * to another one when there is one; -1 for none */
    int16_t               bound;       /* the QPU that carries its load now; -1 while it waits */
    uint8_t               typed;       /* its body is of the QPU type: its device and its load are
                                        * the scheduler's, and ours to move between QPUs */
} qpu_task_t;

/* What a backend is asked to run in one go: the circuit of one task, or
 * those of several that were waiting together. It exists from the moment it
 * is put together until its end is known; its tasks outlive it. */
typedef struct qpu_job_s {
    parsec_qpu_job_id_t   id;          /* at the backend, while in flight */
    uint64_t              tag;
    int64_t               t_submit;    /* when it was first asked for */
    int64_t               deadline;    /* in flight: when it has taken too long */
    uint32_t              nb;          /* tasks it carries, at least one */
    parsec_qpu_request_t *requests;    /* [nb] what the backend is given, in the order they waited in */
    qpu_task_t          **tasks;       /* [nb] */
    parsec_qpu_result_t **results;     /* [nb] where the backend writes */
} qpu_job_t;

/* Tasks that wait: a binary heap, highest task priority first and, among
 * equal priorities, first come first served. */
typedef struct qpu_heap_s {
    qpu_task_t **items;
    uint32_t     n, cap;
} qpu_heap_t;

/* One QPU of this rank. */
typedef struct qpu_service_s {
    int                         index;      /* among the QPUs of this rank */
    int                         number;     /* in the list of the run */
    parsec_device_qpu_module_t *device;
    parsec_qpu_backend_t       *backend;    /* open from the attachment of the device, or from the start */
    parsec_qpu_caps_t           caps;
    pthread_t                   thread;
    parsec_execution_stream_t   es;         /* the progress thread's own stream */
    uint32_t                    max_outstanding;
    uint32_t                    max_batch;              /* tasks a job may carry */
    uint32_t                    max_attempts;           /* jobs a task may have: 1 + retries */
    int64_t                     poll_max_ns;
    int64_t                     submit_interval_ns;
    int64_t                     job_timeout_ns;         /* 0: none */
    int64_t                     unavailable_timeout_ns;
    int64_t                     backoff_min_ns;

    /* Used by its progress thread only. */
    qpu_job_t                 **in_flight;
    uint32_t                    in_flight_n;
    qpu_job_t                  *deferred;    /* its submission was not taken: it is the next one asked */
    int64_t                     t_next_submit; /* nothing is submitted before */
    int64_t                     t_retry;       /* when it has to look again by itself; INT64_MAX: not */
    int64_t                     t_unavailable; /* since when the backend takes nothing; 0 when it does */
    int64_t                     backoff_ns;
    int64_t                     latency_ns;    /* moving average of the time a job takes; 0 before the first */
    int64_t                     t_probe;       /* down: when it is tried again */
    int64_t                     cooldown_ns;   /* down: how long it is left alone the next time */
    int                         probing;       /* it was down, and its next job tells whether it still is */
    uint16_t                    put_back;      /* QPUs that may run what it put into the pool and left there */
    uint16_t                    put_back_pinned; /* ... of which: tasks that named their QPU */

    /* Read by others. */
    int32_t                     room;        /* the places it has, shown while its thread sleeps or is
                                              * about to; 0 while the thread is at work, which may be
                                              * inside a call to its backend */
    int32_t                     places;      /* jobs it could submit now: places left, or 0 while a
                                              * submission is being asked again */
    int32_t                     down;        /* its backend took nothing for too long */
    qpu_heap_t                  pinned;      /* tasks that named it and wait; under the lock of the pool */

    /* Counters, read by other threads. */
    int64_t                     jobs_submitted;
    int64_t                     wait_calls;
    int64_t                     jobs_failed;
    int64_t                     jobs_timed_out;
    int64_t                     retries;
    int64_t                     submit_deferred;
    int64_t                     stray_events;
    int64_t                     c_in_flight, max_in_flight;
    int64_t                     jobs_timed, latency_sum_ns;
    int64_t                     jobs_coalesced, tasks_coalesced, max_job_tasks;
    int64_t                     tasks_moved;
} qpu_service_t;

/* The QPUs of this rank, and what they share. */
typedef struct qpu_pool_s {
    int32_t                    state;      /* QPU_SVC_* */
    int32_t                    enabled;    /* the current context may use the service */
    int32_t                    stop;

    parsec_context_t          *context;
    int                        nb;         /* QPUs of this rank */
    int                        alike;      /* their backends take the same circuits and shots */
    qpu_service_t              svc[QPU_MAX_QPUS];
    int64_t                    t_start;

    qpu_task_t                *submissions; /* stack, pushed by task bodies */
    int64_t                    next_seq;
    int64_t                    wake_next;   /* which QPU the next body tries first */

    /* The tasks that wait for a job: those that named no QPU, and for each
     * QPU those that named it (svc[].pinned). One lock for all of it. */
    pthread_mutex_t            lock;
    qpu_heap_t                 shared;

    /* Counters, read by other threads. */
    int64_t                    tasks_submitted;
    int64_t                    tasks_completed;
    int64_t                    empty_jobs;
    int64_t                    tasks_failed;
    int64_t                    tasks_rejected;
    int64_t                    tasks_unavailable;
    int64_t                    waiting;     /* in the heaps */
    int64_t                    c_pending;   /* that, and the tasks of submissions being asked again */
    int64_t                    max_pending;
} qpu_pool_t;

static qpu_pool_t pool = { .lock = PTHREAD_MUTEX_INITIALIZER };
/* Not reset when a service starts: it is about the process. */
static int64_t qpu_refused_hardware = 0;
/* Why task bodies cannot use the service, when they cannot. */
static const char *qpu_off_reason = "device_qpu_enabled is not set";

static inline void    ctr_add(int64_t *c, int64_t d) { (void)parsec_atomic_fetch_add_int64(c, d); }
static inline int64_t ctr_get(int64_t *c)            { return QPU_LOAD(c); }
/* For high-water marks, which several threads may move. */
static inline void    ctr_max(int64_t *c, int64_t v)
{
    int64_t cur = QPU_LOAD(c);
    while( (v > cur) && !parsec_atomic_cas_int64(c, cur, v) ) cur = QPU_LOAD(c);
}

/* ------------------------------------------------------------------------
 * Backends
 * --------------------------------------------------------------------- */

static const parsec_qpu_backend_ops_t *qpu_backends[QPU_MAX_BACKENDS + 1] = {
    &parsec_qpu_backend_mock_ops,
    NULL
};

const parsec_qpu_backend_ops_t *parsec_qpu_backend_find(const char *name)
{
    if( NULL == name ) return NULL;
    for( int i = 0; NULL != qpu_backends[i]; i++ ) {
        if( 0 == strcmp(name, qpu_backends[i]->name) ) return qpu_backends[i];
    }
    return NULL;
}

int parsec_qpu_backend_register(const parsec_qpu_backend_ops_t *ops)
{
    int i;

    if( (NULL == ops) || (NULL == ops->name) ) return PARSEC_ERR_BAD_PARAM;
    for( i = 0; NULL != qpu_backends[i]; i++ ) {
        if( 0 == strcmp(ops->name, qpu_backends[i]->name) ) return PARSEC_ERR_EXISTS;
    }
    if( QPU_MAX_BACKENDS == i ) return PARSEC_ERR_OUT_OF_RESOURCE;
    qpu_backends[i] = ops;
    return PARSEC_SUCCESS;
}

/* The one rule that keeps a run from reaching hardware by accident. */
static int qpu_backend_refused(const parsec_qpu_backend_ops_t *ops, int says_physical)
{
    if( (!ops->physical && !says_physical) || parsec_qpu_params.allow_hardware ) return 0;
    ctr_add(&qpu_refused_hardware, 1);
    parsec_warning("QPU: backend %s drives a quantum device and device_qpu_allow_hardware is not set; "
                   "it is not used, and quantum tasks cannot run in this context", ops->name);
    return 1;
}

/* ------------------------------------------------------------------------
 * The tasks that wait. Everything here is called with the lock of the pool
 * held, by a progress thread.
 * --------------------------------------------------------------------- */

static inline int task_before(const qpu_task_t *a, const qpu_task_t *b)
{
    if( a->priority != b->priority ) return a->priority > b->priority;
    return a->seq < b->seq;
}

static void heap_insert(qpu_heap_t *heap, qpu_task_t *item)
{
    if( heap->n == heap->cap ) {
        heap->cap   = (0 == heap->cap) ? 64 : 2 * heap->cap;
        heap->items = (qpu_task_t**)realloc(heap->items, heap->cap * sizeof(qpu_task_t*));
    }
    uint32_t i = heap->n++;
    while( i > 0 ) {
        uint32_t parent = (i - 1) / 2;
        if( !task_before(item, heap->items[parent]) ) break;
        heap->items[i] = heap->items[parent];
        i = parent;
    }
    heap->items[i] = item;
}

/* Take out the task at position pos, which is the first one for pos 0. */
static qpu_task_t *heap_remove(qpu_heap_t *heap, uint32_t pos)
{
    qpu_task_t *out = heap->items[pos], *last = heap->items[--heap->n];
    uint32_t i = pos;

    if( pos == heap->n ) return out;       /* it was the last one */
    /* The last task takes the hole, and moves to where it belongs: up when
     * it is before the parent of the hole, which can only be when the hole
     * is not the root, and down otherwise. */
    while( i > 0 ) {
        uint32_t parent = (i - 1) / 2;
        if( !task_before(last, heap->items[parent]) ) break;
        heap->items[i] = heap->items[parent];
        i = parent;
    }
    if( i == pos ) {
        for(;;) {
            uint32_t child = 2 * i + 1;
            if( child >= heap->n ) break;
            if( (child + 1 < heap->n) && task_before(heap->items[child + 1], heap->items[child]) ) child++;
            if( !task_before(heap->items[child], last) ) break;
            heap->items[i] = heap->items[child];
            i = child;
        }
    }
    heap->items[i] = last;
    return out;
}

/* The QPUs of this rank that are not down, one bit each. */
static inline uint16_t qpu_up_mask(void)
{
    uint16_t mask = 0;

    for( int i = 0; i < pool.nb; i++ ) {
        if( !QPU_LOAD(&pool.svc[i].down) ) mask |= (uint16_t)(1u << i);
    }
    return mask;
}

/* May this QPU take this task now? One it may run. But not, when a QPU
 * that is up could take the task instead: a task whose last job failed on
 * this one, which would otherwise come back to the QPU that fails fastest;
 * nor, for a QPU that is down, any task. A task that named its QPU, and one
 * that has nowhere else to go, is taken all the same. */
static inline int qpu_may_take(const qpu_service_t *s, const qpu_task_t *item, uint16_t up)
{
    const uint16_t me = (uint16_t)(1u << s->index);

    if( !(item->may_run & me) ) return 0;
    if( (up & me) && (item->avoid != s->index) ) return 1;
    return (0 == (item->may_run & up & ~me));
}

/* Put a task among those that wait. */
static void waiting_insert(qpu_task_t *item)
{
    heap_insert((item->pin >= 0) ? &pool.svc[item->pin].pinned : &pool.shared, item);
    ctr_add(&pool.waiting, 1);
    ctr_add(&pool.c_pending, 1);
    ctr_max(&pool.max_pending, ctr_get(&pool.c_pending));
}

static qpu_task_t *waiting_take(qpu_heap_t *heap, uint32_t pos)
{
    ctr_add(&pool.waiting, -1);
    ctr_add(&pool.c_pending, -1);
    return heap_remove(heap, pos);
}

/* The task this QPU would take next: the first, by priority and then by
 * arrival, of those that named it and of those that named none and that it
 * may take. NULL when there is none; otherwise where it is. */
static qpu_task_t *waiting_next(const qpu_service_t *s, qpu_heap_t **heap, uint32_t *pos)
{
    const uint16_t up = qpu_up_mask();
    qpu_task_t *mine = (s->pinned.n > 0) ? s->pinned.items[0] : NULL, *any = NULL;
    uint32_t at = 0;

    if( pool.shared.n > 0 ) {
        if( qpu_may_take(s, pool.shared.items[0], up) ) {
            any = pool.shared.items[0];
        } else if( !pool.alike || (pool.shared.items[0]->avoid == s->index) ) {
            /* The first one is not for this QPU, and one further on may be:
             * the backends differ, or the first one just failed here and
             * waits for another QPU. Look at them all. Otherwise every task
             * that named no QPU may run on any of them, and what holds for
             * the first holds for all. */
            for( uint32_t i = 1; i < pool.shared.n; i++ ) {
                qpu_task_t *item = pool.shared.items[i];
                if( qpu_may_take(s, item, up) && ((NULL == any) || task_before(item, any)) ) {
                    any = item;
                    at  = i;
                }
            }
        }
    }
    if( (NULL != mine) && ((NULL == any) || task_before(mine, any)) ) {
        *heap = (qpu_heap_t*)&s->pinned;
        *pos  = 0;
        return mine;
    }
    if( NULL != any ) {
        *heap = &pool.shared;
        *pos  = at;
    }
    return any;
}

/* ------------------------------------------------------------------------
 * The load of a task, which is on the QPU that holds a job for it
 * --------------------------------------------------------------------- */

/* A QPU takes a task: it is the task's device from here on. */
static inline void qpu_bind(qpu_service_t *s, qpu_task_t *item)
{
    item->bound = (int16_t)s->index;
    if( item->typed ) {
        item->task->selected_device = &s->device->super;
        item->task->load            = s->device->super.time_estimate_default;
        (void)parsec_atomic_fetch_add_int64(&s->device->super.device_load, item->task->load);
    }
}

/* The task goes back to wait: no QPU carries it. Its device stays the one it
 * was on, which is as good as any other until a QPU takes it again. */
static inline void qpu_unbind(qpu_task_t *item)
{
    if( item->bound < 0 ) return;
    if( item->typed ) {
        (void)parsec_atomic_fetch_add_int64(&pool.svc[item->bound].device->super.device_load, -item->task->load);
        item->task->load = 0;
    }
    item->bound = -1;
}

/* ------------------------------------------------------------------------
 * Completion
 * --------------------------------------------------------------------- */

/* Give a task whose job is over back to the scheduler. This is the last
 * thing the service does with it. */
static void qpu_hand_off(qpu_task_t *item)
{
    parsec_task_t *task = item->task;
    parsec_execution_stream_t *target =
        pool.context->virtual_processes[item->vp_id]->execution_streams[0];
    /* The device that counts it: the task's own when its body is of the QPU
     * type, which is the QPU that ran its last job or, for a task that never
     * had one, the QPU the device layer selected. A body of another type has
     * the device of that type; then it is the QPU that ran the job, or the
     * first one of the rank. */
    parsec_device_module_t *device = item->typed ? task->selected_device
        : &pool.svc[(item->ran_on >= 0) ? item->ran_on : 0].device->super;

    /* Everything the service keeps about the task goes first, counters
     * included: once the task is scheduled, a worker may complete it and the
     * application may read the counters or tear the taskpool down. */
    ctr_add(&pool.tasks_completed, 1);
    /* The device counts the tasks it executed itself, as an accelerator
     * does: the scheduler counts only tasks whose hook did not return ASYNC. */
    (void)parsec_atomic_fetch_inc_int64((int64_t*)&device->executed_tasks);
    free(item);

    /* Past the body: __parsec_task_progress() will not run the hook again,
     * it completes the task. */
    task->status = PARSEC_TASK_STATUS_COMPLETE;
    PARSEC_LIST_ITEM_SINGLETON(task);
    /* The stream of this thread has no scheduler queue of its own. As the
     * communication thread does, deliver to the first stream of the virtual
     * process the body ran in. */
    __parsec_schedule(target, task, 0);
}

/* A job ended well: it took this long from its submission to its result.
 * The moving average is what the device reports as the time a job takes. It
 * is not what the scheduler adds to the load of the device for a task: that
 * is PARSEC_QPU_TASK_TIME_NS, see there for why. */
static void qpu_note_latency(qpu_service_t *s, int64_t took)
{
    if( took < 1 ) took = 1;
    s->latency_ns = (0 == s->latency_ns) ? took : s->latency_ns + (took - s->latency_ns) / 8;
    if( s->latency_ns < 1 ) s->latency_ns = 1;
    ctr_add(&s->jobs_timed, 1);
    ctr_add(&s->latency_sum_ns, took);
    QPU_STORE(&s->device->latency_ns, s->latency_ns);
}

/* The parts of the result header that are the service's to write. */
static void qpu_finish(qpu_task_t *item)
{
    parsec_qpu_result_t *result = item->result;

    result->version         = PARSEC_QPU_RESULT_VERSION;
    result->shots_requested = item->request.shots;
    result->attempts        = (uint8_t)item->attempts;
    result->qpu             = (item->ran_on >= 0) ? (uint8_t)pool.svc[item->ran_on].number : (uint8_t)PARSEC_QPU_NONE;
    result->batch           = item->batch;
    result->t_hook          = item->t_hook;
    result->t_submit        = item->t_submit;
    result->t_retrieve      = parsec_qpu_now_ns();
    qpu_hand_off(item);
}

/* A task that ends without an answer completes all the same, and its
 * result says why. */
static void qpu_finish_without_result(qpu_task_t *item, parsec_qpu_status_t status)
{
    memset(item->result, 0, sizeof(parsec_qpu_result_t));
    item->result->status    = (uint8_t)status;
    item->result->nb_qubits = (uint16_t)item->request.circuit->nb_qubits;
    if( 0 == item->t_submit ) item->t_submit = parsec_qpu_now_ns();   /* it never left */
    if( PARSEC_QPU_STATUS_OK != status ) {
        ctr_add(&pool.tasks_failed, 1);
        if( PARSEC_QPU_STATUS_REJECTED == status )    ctr_add(&pool.tasks_rejected, 1);
        if( PARSEC_QPU_STATUS_UNAVAILABLE == status ) ctr_add(&pool.tasks_unavailable, 1);
        parsec_debug_verbose(5, parsec_qpu_output_stream,
                             "QPU: a task of class %s completes with status %d after %u job(s)",
                             item->task->task_class->name, (int)status, (unsigned)item->attempts);
    }
    qpu_finish(item);
}

/* Tasks were put into the pool by this thread. Wake the other QPUs that may
 * run them and have a place for them, unless nothing waits any more. This
 * is done before this thread calls its backend again: such a call may take
 * as long as a provider takes to answer, and no QPU may sleep beside a free
 * place for that long while tasks wait. A QPU that is down is woken for a
 * task that named it, and for the others when no QPU that is up may run
 * them. (When only some of them have a QPU that is up to go to, the others
 * wait until the QPU that is down looks again by itself.) */
static void qpu_wake_others(qpu_service_t *s)
{
    const uint16_t me = (uint16_t)(1u << s->index);
    const uint16_t any = (uint16_t)(s->put_back & ~me), named = (uint16_t)(s->put_back_pinned & ~me);
    uint16_t up;

    s->put_back = s->put_back_pinned = 0;
    if( (0 == (any | named)) || (0 == ctr_get(&pool.waiting)) ) return;
    up = qpu_up_mask();
    for( int i = 0; i < pool.nb; i++ ) {
        const uint16_t bit = (uint16_t)(1u << i);
        qpu_service_t *other = &pool.svc[i];

        if( (0 == ((any | named) & bit)) || (QPU_LOAD(&other->room) <= 0) ) continue;
        if( (named & bit) || (up & bit) || (0 == (any & up)) ) {
            (void)other->backend->ops->wakeup(other->backend);
        }
    }
}

/* A task goes back to wait for a job, on whichever QPU takes it. The others
 * are told at once: this QPU is the last one that should take it again. */
static void qpu_wait_again(qpu_service_t *s, qpu_task_t *item)
{
    /* Read now: once the task is in the pool another QPU may take it, run
     * it and free this record. */
    const int named = (item->pin >= 0);
    const uint16_t may_run = item->may_run;

    qpu_unbind(item);
    pthread_mutex_lock(&pool.lock);
    waiting_insert(item);
    pthread_mutex_unlock(&pool.lock);
    if( named ) s->put_back_pinned |= may_run;
    else        s->put_back        |= may_run;
    qpu_wake_others(s);
}

/* One job of a task is over without an answer. The task gets another one,
 * unless it had its share. */
static void qpu_attempt_failed(qpu_service_t *s, qpu_task_t *item, parsec_qpu_status_t status)
{
    if( item->attempts < s->max_attempts ) {
        /* It kept its place in the arrival order, and it has had a job: its
         * next one is for it alone (qpu_can_share), and on another QPU when
         * one can take it. */
        item->avoid = (int16_t)s->index;
        qpu_wait_again(s, item);
        return;
    }
    qpu_finish_without_result(item, status);
}

/* ------------------------------------------------------------------------
 * Jobs
 * --------------------------------------------------------------------- */

/* A job with room for nb tasks. One block: the requests are what the
 * backend is handed, so they are next to each other. */
static qpu_job_t *qpu_job_new(uint32_t nb)
{
    qpu_job_t *job = (qpu_job_t*)malloc(sizeof(qpu_job_t) +
                                        (size_t)nb * (sizeof(parsec_qpu_request_t) + sizeof(qpu_task_t*) +
                                                      sizeof(parsec_qpu_result_t*)));

    job->requests = (parsec_qpu_request_t*)(job + 1);
    job->tasks    = (qpu_task_t**)(job->requests + nb);
    job->results  = (parsec_qpu_result_t**)(job->tasks + nb);
    job->id       = 0;
    job->tag      = 0;
    job->t_submit = 0;
    job->deadline = INT64_MAX;
    job->nb       = 0;
    return job;
}

/* May a task that waits go into the job that `lead` opens? Both are at
 * their first job: a task that had one is retried alone, so that a task
 * whose jobs fail takes nobody with it twice. And they ask for the same
 * number of shots, which is what every provider takes in one job. */
static inline int qpu_can_share(const qpu_task_t *lead, const qpu_task_t *other)
{
    return (0 == lead->attempts) && (0 == other->attempts) &&
           (lead->request.shots == other->request.shots);
}

/* The jobs the other QPUs of the rank could submit right now. */
static inline uint32_t qpu_room_elsewhere(const qpu_service_t *s)
{
    uint32_t room = 0;

    for( int i = 0; i < pool.nb; i++ ) {
        if( (i != s->index) && !QPU_LOAD(&pool.svc[i].down) ) room += (uint32_t)QPU_LOAD(&pool.svc[i].places);
    }
    return room;
}

/* Put the next job of this QPU together, from the tasks it may take; NULL
 * when there is none. The job is submitted at once whatever it holds; the
 * question is only how many of the tasks that wait go with the first one. */
static qpu_job_t *qpu_job_form(qpu_service_t *s)
{
    qpu_task_t *lead, *more;
    int64_t now;
    qpu_heap_t *heap = NULL;
    uint32_t pos = 0, want = 1, waiting;
    qpu_job_t *job;

    pthread_mutex_lock(&pool.lock);
    if( NULL == (lead = waiting_next(s, &heap, &pos)) ) {
        pthread_mutex_unlock(&pool.lock);
        return NULL;
    }
    (void)waiting_take(heap, pos);
    waiting = s->pinned.n + pool.shared.n;
    if( (s->max_batch > 1) && (0 == lead->attempts) && (waiting > 0) ) {
        /* The jobs that can be submitted right now: one per free place at
         * the backend, and at those of the other QPUs when tasks wait that
         * named none; or just this one when submissions have to be spaced.
         * The tasks that wait are spread evenly over them, so that tasks
         * share a job only when there are more of them than jobs, and no
         * place stays empty next to a job that holds everything. */
        uint32_t jobs = s->max_outstanding - s->in_flight_n;
        const uint32_t tasks = waiting + 1;

        if( s->submit_interval_ns > 0 ) jobs = 1;
        else if( pool.shared.n > 0 )    jobs += qpu_room_elsewhere(s);
        want = (tasks + jobs - 1) / jobs;
        if( want > s->max_batch ) want = s->max_batch;
    }

    job = qpu_job_new(want);
    job->tasks[job->nb++] = lead;
    /* In the order they wait in, and no further than the first task that
     * cannot share: nobody overtakes a task that was there before. */
    while( job->nb < want ) {
        if( (NULL == (more = waiting_next(s, &heap, &pos))) || !qpu_can_share(lead, more) ) break;
        job->tasks[job->nb++] = waiting_take(heap, pos);
    }
    pthread_mutex_unlock(&pool.lock);
    /* The time the job is asked for is read once its tasks are taken: another
     * QPU's thread may have put them into the pool a moment ago, and a task
     * is not asked for before its body submitted it. */
    now = parsec_qpu_now_ns();

    /* A new job has a new tag: that of its first task with the number of
     * jobs that were put together for that task, this one included. A job
     * that is asked again is the same job and keeps its tag; a task whose
     * submission was given up on, and that is asked for again here or on
     * another QPU, is in a new job with another tag, also when no backend
     * took the first one, so no backend is ever asked twice under one tag
     * for two jobs. (The count is the low byte of the tag. A task would have
     * to be given up on some 250 times to leave it.) */
    job->t_submit = now;
    for( uint32_t i = 0; i < job->nb; i++ ) {
        qpu_task_t *item = job->tasks[i];

        item->formed++;
        qpu_bind(s, item);
        item->request.attempt = (uint32_t)item->attempts + 1;
        item->t_submit        = now;
        job->requests[i]      = item->request;
        job->results[i]       = item->result;
    }
    job->tag = lead->tag + (uint64_t)lead->formed;
    return job;
}

/* Forget a job and its tasks, which are not completed: for the end of a
 * context that did not wait for them. Returns how many tasks that was. */
static int qpu_job_drop(qpu_job_t *job)
{
    const int nb = (int)job->nb;

    for( uint32_t i = 0; i < job->nb; i++ ) {
        free(job->tasks[i]);
    }
    free(job);
    return nb;
}

/* A job is over without an answer for any of its tasks. */
static void qpu_job_failed(qpu_service_t *s, qpu_job_t *job, parsec_qpu_status_t status)
{
    for( uint32_t i = 0; i < job->nb; i++ ) {
        qpu_attempt_failed(s, job->tasks[i], status);
    }
    free(job);
}

/* A job is over, and each of its tasks completes with this status and no
 * answer. */
static void qpu_job_finish_without_result(qpu_job_t *job, parsec_qpu_status_t status)
{
    for( uint32_t i = 0; i < job->nb; i++ ) {
        qpu_finish_without_result(job->tasks[i], status);
    }
    free(job);
}

/* How many jobs this QPU could submit now, for the others when they put
 * their own jobs together. */
static inline int32_t qpu_note_places(qpu_service_t *s)
{
    const int32_t places = (NULL == s->deferred) ? (int32_t)(s->max_outstanding - s->in_flight_n) : 0;

    QPU_STORE(&s->places, places);
    return places;
}

/* Show the places of this QPU: to be woken for a task. Only by its thread,
 * and only when it is about to look for work once more and then sleep. */
static inline void qpu_publish_room(qpu_service_t *s)
{
    QPU_STORE(&s->room, qpu_note_places(s));
}

static void qpu_job_event(qpu_service_t *s, const parsec_qpu_event_t *event)
{
    qpu_job_t *job = NULL;

    for( uint32_t i = 0; i < s->in_flight_n; i++ ) {
        if( s->in_flight[i]->id == event->id ) {
            job = s->in_flight[i];
            s->in_flight[i] = s->in_flight[--s->in_flight_n];
            break;
        }
    }
    if( NULL == job ) {
        /* A final state for a job that is not in flight: a second report of
         * an end that was handled, or the end of a job that was cancelled.
         * The first final state of a job is the only one acted upon. */
        ctr_add(&s->stray_events, 1);
        parsec_debug_verbose(5, parsec_qpu_output_stream,
                             "QPU %d: ignoring a final state for job %llu, which is not in flight",
                             s->number, (unsigned long long)event->id);
        return;
    }
    ctr_add(&s->c_in_flight, -1);

    if( PARSEC_QPU_JOB_DONE == event->state ) {
        for( uint32_t i = 0; i < job->nb; i++ ) {
            memset(job->results[i], 0, sizeof(parsec_qpu_result_t));
        }
        if( PARSEC_SUCCESS == s->backend->ops->results(s->backend, job->id, job->results, (int)job->nb) ) {
            /* Every result is in its buffer before the first task is given
             * back: from then on the tasks complete in any order. */
            qpu_note_latency(s, parsec_qpu_now_ns() - job->t_submit);
            for( uint32_t i = 0; i < job->nb; i++ ) {
                qpu_finish(job->tasks[i]);
            }
            free(job);
            return;
        }
        /* Done, and nothing to show for it. Whatever the backend wrote before
         * it gave up is not an answer: the next job of each task starts from
         * a clean header, and so does a task that ends here. */
        ctr_add(&s->jobs_failed, 1);
        qpu_job_failed(s, job, PARSEC_QPU_STATUS_FAILED);
        return;
    }
    if( PARSEC_QPU_JOB_CANCELLED == event->state ) {
        /* Not by us: somebody at the provider did. That is not retried. */
        qpu_job_finish_without_result(job, PARSEC_QPU_STATUS_CANCELLED);
        return;
    }
    ctr_add(&s->jobs_failed, 1);
    qpu_job_failed(s, job, PARSEC_QPU_STATUS_FAILED);
}

/* Cancel the jobs that have taken too long. */
static void qpu_check_deadlines(qpu_service_t *s)
{
    const int64_t now = parsec_qpu_now_ns();

    for( uint32_t i = 0; i < s->in_flight_n; ) {
        qpu_job_t *job = s->in_flight[i];
        if( job->deadline > now ) { i++; continue; }
        s->in_flight[i] = s->in_flight[--s->in_flight_n];
        ctr_add(&s->c_in_flight, -1);
        /* From here on the job is not in flight: if the backend reports its
         * end all the same, the report finds nothing and is ignored. */
        (void)s->backend->ops->cancel(s->backend, job->id);
        ctr_add(&s->jobs_timed_out, 1);
        qpu_job_failed(s, job, PARSEC_QPU_STATUS_TIMED_OUT);
    }
}

/* ------------------------------------------------------------------------
 * The progress threads
 * --------------------------------------------------------------------- */

/* Does the backend of this QPU take this circuit and these shots? */
static inline int qpu_fits(const qpu_service_t *s, const qpu_task_t *item)
{
    return (item->request.circuit->nb_qubits <= s->caps.nb_qubits) &&
           ((0 == s->caps.max_shots) || (item->request.shots <= s->caps.max_shots));
}

/* Take everything task bodies submitted since the last time, whichever QPU
 * it is for, and put it among the tasks that wait. */
static void qpu_drain_submissions(qpu_service_t *s)
{
    qpu_task_t *stack, *item, *keep = NULL, *last = NULL;
    uint32_t kept = 0;

    do {
        stack = QPU_LOAD(&pool.submissions);
        if( NULL == stack ) return;
    } while( !parsec_atomic_cas_ptr(&pool.submissions, stack, NULL) );

    while( NULL != (item = stack) ) {
        stack = item->next;
        item->next = NULL;
        if( PARSEC_QPU_CIRCUIT_EMPTY & item->request.circuit->flags ) {
            /* Nothing to run: no backend, no slot, no entry. */
            ctr_add(&pool.empty_jobs, 1);
            item->t_submit = parsec_qpu_now_ns();
            qpu_finish_without_result(item, PARSEC_QPU_STATUS_OK);
            continue;
        }
        if( item->pin >= 0 ) {
            item->may_run = qpu_fits(&pool.svc[item->pin], item) ? (uint16_t)(1u << item->pin) : 0;
        } else {
            item->may_run = 0;
            for( int i = 0; i < pool.nb; i++ ) {
                if( qpu_fits(&pool.svc[i], item) ) item->may_run |= (uint16_t)(1u << i);
            }
        }
        if( 0 == item->may_run ) {
            /* No job of a backend it could go to can do that: it would fail
             * every time. */
            qpu_finish_without_result(item, PARSEC_QPU_STATUS_REJECTED);
            continue;
        }
        /* In the order of the stack, which the heap does not care about. */
        if( NULL == keep ) keep = item; else last->next = item;
        last = item;
    }
    if( NULL == keep ) return;

    pthread_mutex_lock(&pool.lock);
    while( NULL != (item = keep) ) {
        keep = item->next;
        item->next = NULL;
        if( item->pin >= 0 ) s->put_back_pinned |= item->may_run;
        else                 s->put_back        |= item->may_run;
        waiting_insert(item);
        kept++;
    }
    pthread_mutex_unlock(&pool.lock);
    /* More than this QPU has places for: the others are needed now, not
     * after this QPU has submitted its own. With fewer, it most likely takes
     * them all, and tells the others only if it turns out not to. */
    if( (NULL != s->deferred) || (kept > (s->max_outstanding - s->in_flight_n)) ) qpu_wake_others(s);
}

/* This QPU answered: it is up, and the next time it stops answering it is
 * given its whole time again. */
static inline void qpu_is_up(qpu_service_t *s)
{
    if( QPU_LOAD(&s->down) ) {
        parsec_debug_verbose(3, parsec_qpu_output_stream, "QPU %d: its backend takes jobs again", s->number);
        QPU_STORE(&s->down, 0);
    }
    s->probing     = 0;
    s->cooldown_ns = s->unavailable_timeout_ns;
}

/* May every task of this job go to another QPU that is up? */
static int qpu_job_can_move(const qpu_service_t *s, const qpu_job_t *job)
{
    const uint16_t others = (uint16_t)(qpu_up_mask() & ~(1u << s->index));

    for( uint32_t i = 0; i < job->nb; i++ ) {
        if( (job->tasks[i]->pin >= 0) || (0 == (job->tasks[i]->may_run & others)) ) return 0;
    }
    return 1;
}

/* The backend of this QPU has taken nothing for too long, or took nothing
 * when it was tried again. The submission that is being asked for is given
 * up on, which settles it at the backend: the provider may have that job,
 * and nobody may read a task's buffers once the task is completed or runs
 * somewhere else. The QPU is down from here on. Then every task that has
 * no QPU left to go to completes as unavailable: of that job, and of those
 * that wait; the others wait for a QPU that is up. Whoever comes next to
 * this QPU starts asking afresh. */
static void qpu_give_up(qpu_service_t *s, qpu_job_t *asked, int64_t now)
{
    qpu_task_t *gone = NULL, *item;
    const uint32_t asked_nb = asked->nb;
    uint32_t moved = 0, flushed = 0;
    uint16_t up;

    if( NULL != s->backend->ops->abandon ) {
        (void)s->backend->ops->abandon(s->backend, asked->tag);
    }
    QPU_STORE(&s->down, 1);
    s->probing = 0;
    if( s->cooldown_ns <= 0 ) s->cooldown_ns = s->unavailable_timeout_ns;
    s->t_probe = now + s->cooldown_ns;
    if( s->cooldown_ns < QPU_COOLDOWN_MAX * s->unavailable_timeout_ns ) s->cooldown_ns *= 2;
    up = qpu_up_mask();

    for( uint32_t i = 0; i < asked->nb; i++ ) {
        item = asked->tasks[i];
        if( (item->pin < 0) && (0 != (item->may_run & up)) ) {
            /* Another QPU can run it. The backend never took this job, so it
             * was not one of the task's jobs. */
            qpu_wait_again(s, item);
            moved++;
        } else {
            item->next = gone;
            gone = item;
        }
    }
    free(asked);
    ctr_add(&s->tasks_moved, (int64_t)moved);

    /* The tasks that wait and have no QPU that is up to go to: those that
     * named this one, and of the others those that only QPUs that are down
     * may run, in the order they wait in. */
    pthread_mutex_lock(&pool.lock);
    while( s->pinned.n > 0 ) {
        item = waiting_take(&s->pinned, 0);
        item->next = gone; gone = item; flushed++;
    }
    while( (pool.shared.n > 0) && (0 == (pool.shared.items[0]->may_run & up)) ) {
        item = waiting_take(&pool.shared, 0);
        item->next = gone; gone = item; flushed++;
    }
    /* Further on, where the backends differ, some may have nowhere to go
     * while others have. Taking one out of the middle moves others about,
     * so the heap is made again from those that stay. */
    for( uint32_t i = 1; i < pool.shared.n; i++ ) {
        if( 0 != (pool.shared.items[i]->may_run & up) ) continue;
        {
            qpu_heap_t stay = { NULL, 0, 0 };

            for( uint32_t j = 0; j < pool.shared.n; j++ ) {
                item = pool.shared.items[j];
                if( 0 != (item->may_run & up) ) {
                    heap_insert(&stay, item);
                } else {
                    ctr_add(&pool.waiting, -1);
                    ctr_add(&pool.c_pending, -1);
                    item->next = gone; gone = item; flushed++;
                }
            }
            free(pool.shared.items);
            pool.shared = stay;
        }
        break;
    }
    pthread_mutex_unlock(&pool.lock);

    parsec_debug_verbose(3, parsec_qpu_output_stream,
                         "QPU %d: backend %s has taken no job for %.3f s and is left alone for %.3f s; "
                         "%u task(s) wait for another QPU, %u complete as unavailable",
                         s->number, s->backend->ops->name,
                         (0 != s->t_unavailable) ? (now - s->t_unavailable) * 1e-9 : 0.0,
                         (s->t_probe - now) * 1e-9, moved, flushed + (asked_nb - moved));

    /* They were put on the list one before the other; complete them in the
     * order they were found in. */
    {
        qpu_task_t *ordered = NULL;
        while( NULL != (item = gone) ) { gone = item->next; item->next = ordered; ordered = item; }
        while( NULL != (item = ordered) ) {
            ordered = item->next;
            item->next = NULL;
            qpu_unbind(item);
            qpu_finish_without_result(item, PARSEC_QPU_STATUS_UNAVAILABLE);
        }
    }
    s->t_unavailable = 0;
    s->backoff_ns    = s->backoff_min_ns;
}

static inline int qpu_unavailable_for_too_long(const qpu_service_t *s, int64_t now)
{
    return (0 != s->t_unavailable) && (s->unavailable_timeout_ns > 0) &&
           ((now - s->t_unavailable) >= s->unavailable_timeout_ns);
}

/* Does a task wait that this QPU may take? */
static int qpu_has_work(const qpu_service_t *s)
{
    qpu_heap_t *heap;
    uint32_t pos;
    int work;

    pthread_mutex_lock(&pool.lock);
    work = (NULL != waiting_next(s, &heap, &pos));
    pthread_mutex_unlock(&pool.lock);
    return work;
}

/* Give the backend of this QPU as many jobs as its limit, the minimum
 * interval between two submissions and the state of the backend allow. */
static void qpu_admit(qpu_service_t *s)
{
    int64_t now = parsec_qpu_now_ns();

    s->t_retry = INT64_MAX;
    while( s->in_flight_n < s->max_outstanding ) {
        qpu_job_t *job = s->deferred;
        int rc;

        if( (NULL != job) && qpu_unavailable_for_too_long(s, now) ) {
            /* Asked for long enough: it is not asked again. */
            s->deferred = NULL;
            ctr_add(&pool.c_pending, -(int64_t)job->nb);
            qpu_give_up(s, job, now);
            break;
        }
        if( now < s->t_next_submit ) {
            /* Too early. Come back by itself when there is something to do then. */
            if( (NULL != job) || qpu_has_work(s) ) s->t_retry = s->t_next_submit;
            break;
        }
        if( NULL != job ) {
            /* A submission that is asked again is the job it was: the same
             * tasks, the same tag, and the time of its first asking. */
            s->deferred = NULL;
            ctr_add(&pool.c_pending, -(int64_t)job->nb);
        } else {
            /* What bodies submitted while this thread was in its backend. */
            if( NULL != QPU_LOAD(&pool.submissions) ) qpu_drain_submissions(s);
            if( NULL == (job = qpu_job_form(s)) ) break;
        }

        /* The backend may take its time to answer. What this thread put into
         * the pool and did not take is for the others, now. */
        qpu_wake_others(s);
        rc = s->backend->ops->submit(s->backend, job->requests, (int)job->nb, job->tag, &job->id);
        now = parsec_qpu_now_ns();
        /* The minimum interval is between two submissions, whatever the
         * backend made of them. */
        s->t_next_submit = now + s->submit_interval_ns;

        if( PARSEC_ERR_OUT_OF_RESOURCE == rc ) {
            /* Not taken, or not known to be. The provider may have the job
             * all the same, so this QPU submits nothing else until this one
             * is settled: asking again with the same tag cannot run it
             * twice. */
            ctr_add(&s->submit_deferred, 1);
            if( 0 == s->t_unavailable ) s->t_unavailable = now;
            if( qpu_unavailable_for_too_long(s, now) || (s->probing && qpu_job_can_move(s, job)) ) {
                /* For too long; or this was the job that tells whether a QPU
                 * that was down is back, and it is not: its tasks have other
                 * QPUs to go to, and do not wait here. */
                qpu_give_up(s, job, now);
                break;
            }
            s->deferred = job;
            ctr_add(&pool.c_pending, (int64_t)job->nb);
            if( s->t_next_submit < now + s->backoff_ns ) s->t_next_submit = now + s->backoff_ns;
            s->backoff_ns = (2 * s->backoff_ns < QPU_BACKOFF_MAX_NS) ? 2 * s->backoff_ns : QPU_BACKOFF_MAX_NS;
            s->t_retry = s->t_next_submit;
            break;
        }

        /* The backend answered for good, one way or the other: it is there,
         * and that was one of the jobs of each task. */
        s->t_unavailable = 0;
        s->backoff_ns    = s->backoff_min_ns;
        qpu_is_up(s);
        if( job->requests[0].attempt > 1 ) ctr_add(&s->retries, 1);
        for( uint32_t i = 0; i < job->nb; i++ ) {
            job->tasks[i]->attempts++;
            job->tasks[i]->ran_on = (int16_t)s->index;
        }

        if( PARSEC_SUCCESS == rc ) {
            for( uint32_t i = 0; i < job->nb; i++ ) {
                job->tasks[i]->batch = job->nb;
            }
            /* The time a job may take is that of a job of one task, for
             * each task it carries: tasks that would each have been given
             * that long are not cut short for having gone together. */
            job->deadline = (s->job_timeout_ns > 0) ? now + s->job_timeout_ns * (int64_t)job->nb : INT64_MAX;
            s->in_flight[s->in_flight_n++] = job;
            ctr_add(&s->jobs_submitted, 1);
            if( job->nb > 1 ) {
                ctr_add(&s->jobs_coalesced, 1);
                ctr_add(&s->tasks_coalesced, (int64_t)job->nb);
            }
            ctr_max(&s->max_job_tasks, (int64_t)job->nb);
            ctr_add(&s->c_in_flight, 1);
            ctr_max(&s->max_in_flight, (int64_t)s->in_flight_n);
            /* The others count the places that are left when they put their
             * own jobs together. */
            (void)qpu_note_places(s);
            continue;
        }

        /* Refused. */
        ctr_add(&s->jobs_failed, 1);
        qpu_job_failed(s, job, PARSEC_QPU_STATUS_FAILED);
    }
}

/* How long the progress thread may sleep: until the next thing it has to do
 * by itself, and never longer than device_qpu_poll_max. */
static int64_t qpu_wait_timeout(const qpu_service_t *s)
{
    const int64_t now = parsec_qpu_now_ns();
    int64_t until = now + s->poll_max_ns;

    /* The next submission it may make, when it has one to make. */
    if( s->t_retry < until ) until = s->t_retry;
    /* The moment a submission has been asked for long enough. */
    if( (NULL != s->deferred) && (0 != s->t_unavailable) && (s->unavailable_timeout_ns > 0) &&
        (s->t_unavailable + s->unavailable_timeout_ns < until) ) {
        until = s->t_unavailable + s->unavailable_timeout_ns;
    }
    /* The moment a QPU that is down is tried again. */
    if( s->down && (s->t_probe < until) ) until = s->t_probe;
    for( uint32_t i = 0; i < s->in_flight_n; i++ ) {
        if( s->in_flight[i]->deadline < until ) until = s->in_flight[i]->deadline;
    }
    return (until > now) ? until - now : 0;
}

static void *qpu_progress_main(void *arg)
{
    qpu_service_t *s = (qpu_service_t*)arg;
    parsec_context_t *context = pool.context;
    parsec_qpu_event_t events[QPU_MAX_EVENTS];

#if defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    /* The thread that started us is a worker bound to its core. This thread
     * sleeps most of the time and must not be tied to that one core when it
     * wakes: let it run on any core the process may use. */
    (void)parsec_bindthread_mask(context->cpuset_allowed_mask);
#endif

    /* A stream of our own, as the communication thread has: instrumentation
     * looks up the calling thread's stream, and a worker's stream must never
     * be borrowed. It has no scheduler queue and no profiling stream, and
     * nothing is ever allocated or completed through it. */
    memset(&s->es, 0, sizeof(s->es));
    s->es.th_id            = 0;
    s->es.core_id          = -1;
    s->es.socket_id        = -1;
    s->es.virtual_process  = context->virtual_processes[0];
    s->es.scheduler_object = NULL;
    s->es.next_task        = (parsec_task_t*)0xdeadbeef;  /* never used, and never NULL */
    parsec_set_my_execution_stream(&s->es);

    /* At work from here on: see below. */
    QPU_STORE(&s->room, 0);
    for(;;) {
        if( s->down && (parsec_qpu_now_ns() >= s->t_probe) ) {
            /* Left alone for long enough. It takes tasks like the others
             * again, and the first job it submits tells whether it is back. */
            QPU_STORE(&s->down, 0);
            s->probing = 1;
        }
        qpu_drain_submissions(s);
        if( QPU_LOAD(&pool.stop) ) break;
        qpu_check_deadlines(s);
        qpu_admit(s);

        /* While this thread was at work its QPU showed no room, so nobody
         * woke it: a body went to another QPU, or to none. Now show the
         * room, and then look once more at what may have come meanwhile. A
         * body pushes its task and then reads the room, and a thread puts
         * tasks into the pool and then reads it: whoever did so before the
         * room was shown is found by the look that follows, and whoever
         * does so later sees the room and wakes this thread. The look at
         * the pool is left out when this QPU has to wait before it may
         * submit again: it looks by itself then. */
        qpu_publish_room(s);
        qpu_wake_others(s);
        if( (QPU_LOAD(&s->room) > 0) &&
            ((NULL != QPU_LOAD(&pool.submissions)) ||
             ((INT64_MAX == s->t_retry) && (ctr_get(&pool.waiting) > 0) && qpu_has_work(s))) ) {
            QPU_STORE(&s->room, 0);
            continue;
        }

        /* Sleep in the backend. A submission wakes it, and so does the end
         * of a job; the timeout is for what the service has to do by itself. */
        ctr_add(&s->wait_calls, 1);
        int n = s->backend->ops->wait(s->backend, qpu_wait_timeout(s), events, QPU_MAX_EVENTS);
        /* At work: the calls that follow may each take as long as a provider
         * takes to answer, and a task that comes meanwhile is for a QPU
         * whose thread sleeps. */
        QPU_STORE(&s->room, 0);
        if( n < 0 ) {
            struct timespec pause = { 0, 1000000 };
            parsec_warning("QPU %d: backend %s failed while waiting (%d)", s->number, s->backend->ops->name, n);
            nanosleep(&pause, NULL);    /* whatever it is, do not spin on it */
            n = 0;
        }
        for( int i = 0; i < n; i++ ) {
            qpu_job_event(s, &events[i]);
        }
    }

    parsec_set_my_execution_stream(NULL);
    return NULL;
}

/* ------------------------------------------------------------------------
 * Start and stop
 * --------------------------------------------------------------------- */

static void qpu_at_fini(void *unused)
{
    (void)unused;
    (void)parsec_qpu_service_fini();
    pool.enabled = 0;
    qpu_off_reason = "device_qpu_enabled is not set";
    if( (parsec_qpu_output_stream >= 0) && (parsec_qpu_output_stream != parsec_device_output) ) {
        parsec_output_close(parsec_qpu_output_stream);
    }
    parsec_qpu_output_stream = -1;
}

void parsec_qpu_service_refuse(const char *why)
{
    qpu_off_reason = why;
}

int parsec_qpu_service_enable(void)
{
    if( pool.enabled ) return PARSEC_SUCCESS;
    /* Settle now, while the context is being initialized, whether the
     * backends that were asked for may be used at all. One that may not
     * takes the others with it: a run with some of its QPUs is not the run
     * that was asked for. */
    for( int q = 0; q < parsec_qpu_list_size(); q++ ) {
        const char *name = parsec_qpu_list_backend(q);
        const parsec_qpu_backend_ops_t *ops = parsec_qpu_backend_find(name);

        if( NULL == ops ) {
            parsec_warning("QPU: no backend named %s; quantum tasks cannot run in this context",
                           (NULL != name) ? name : "(null)");
            qpu_off_reason = "a backend named by device_qpu_backend or device_qpu_devices does not exist";
            return PARSEC_ERR_NOT_FOUND;
        }
        if( qpu_backend_refused(ops, 0) ) {
            qpu_off_reason = "a backend drives a quantum device and device_qpu_allow_hardware is not set";
            return PARSEC_ERR_NOT_SUPPORTED;
        }
    }
    /* The QPUs of this rank are told one by one, as their devices are
     * attached. What is left of the last context is what its statistics
     * were read from. */
    pool.nb = 0;
    memset(pool.svc, 0, sizeof(pool.svc));
    pool.enabled = 1;
    /* What is said to a body that submits if no QPU device makes it into
     * the context all the same; opening the backend has a better reason
     * when that is what fails. */
    qpu_off_reason = "no QPU device could be attached to this context";
    /* Stop before the workers are joined and the profiling system is torn
     * down, which is long before the device components are closed. */
    parsec_context_at_fini(qpu_at_fini, NULL);
    return PARSEC_SUCCESS;
}

/* Open the backend of a QPU and learn what it is. Called from the main
 * thread when the device of that QPU is attached, and by whoever brings the
 * service up when no backend is open then, which is the case after the
 * service was stopped by hand. Never by two threads at once. */
static int qpu_backend_open(qpu_service_t *s)
{
    const char *name = parsec_qpu_list_backend(s->number);
    const parsec_qpu_backend_ops_t *ops;
    int rc, limit;

    if( NULL != s->backend ) return PARSEC_SUCCESS;
    if( NULL == (ops = parsec_qpu_backend_find(name)) ) {
        parsec_warning("QPU %d: no backend named %s", s->number, (NULL != name) ? name : "(null)");
        return PARSEC_ERR_NOT_FOUND;
    }
    if( qpu_backend_refused(ops, 0) ) return PARSEC_ERR_NOT_SUPPORTED;
    if( (PARSEC_SUCCESS != (rc = ops->open(&s->backend, s->number))) || (NULL == s->backend) ) {
        parsec_warning("QPU %d: backend %s did not open (%d)", s->number, ops->name, rc);
        s->backend = NULL;
        return (PARSEC_SUCCESS != rc) ? rc : PARSEC_ERROR;
    }
    memset(&s->caps, 0, sizeof(s->caps));
    (void)ops->capabilities(s->backend, &s->caps);
    if( qpu_backend_refused(ops, (int)s->caps.is_physical) ) {
        /* It did not say so before it was opened. Nothing was submitted. */
        (void)ops->close(s->backend);
        s->backend = NULL;
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    limit = parsec_qpu_list_option_int(s->number, "max_outstanding", parsec_qpu_params.max_outstanding);
    s->max_outstanding = (uint32_t)(limit > 0 ? limit : 1);
    if( (s->caps.max_jobs > 0) && (s->caps.max_jobs < s->max_outstanding) ) {
        s->max_outstanding = s->caps.max_jobs;
    }
    /* A job carries as many tasks as was asked for and as the backend takes
     * circuits; a backend that does not say takes one. */
    limit = parsec_qpu_list_option_int(s->number, "max_batch", parsec_qpu_params.max_batch);
    s->max_batch = (uint32_t)(limit > 0 ? limit : 1);
    if( s->max_batch > ((s->caps.max_circuits_per_job > 0) ? s->caps.max_circuits_per_job : 1) ) {
        s->max_batch = (s->caps.max_circuits_per_job > 0) ? s->caps.max_circuits_per_job : 1;
    }
    return PARSEC_SUCCESS;
}

int parsec_qpu_service_open(parsec_device_qpu_module_t *module, parsec_qpu_caps_t *caps,
                            uint32_t *max_outstanding, uint32_t *max_batch)
{
    qpu_service_t *s;
    int rc;

    if( QPU_MAX_QPUS == pool.nb ) return PARSEC_ERR_OUT_OF_RESOURCE;
    s = &pool.svc[pool.nb];
    memset(s, 0, sizeof(*s));
    s->index  = pool.nb;
    s->number = module->number;
    s->device = module;
    if( PARSEC_SUCCESS != (rc = qpu_backend_open(s)) ) {
        qpu_off_reason = "the backend of a QPU of this rank could not be used, and no other QPU is on this rank";
        memset(s, 0, sizeof(*s));
        return rc;
    }
    module->service  = pool.nb++;
    *caps            = s->caps;
    *max_outstanding = s->max_outstanding;
    *max_batch       = s->max_batch;
    return PARSEC_SUCCESS;
}

/* The device of the QPU that was opened last could not be added to the
 * context after all: it is not a QPU of this rank. Main thread only. */
void parsec_qpu_service_forget(parsec_device_qpu_module_t *module)
{
    qpu_service_t *s;

    if( (pool.nb < 1) || (pool.svc[pool.nb - 1].device != module) ) return;
    s = &pool.svc[--pool.nb];
    if( NULL != s->backend ) (void)s->backend->ops->close(s->backend);
    memset(s, 0, sizeof(*s));
    module->service = -1;
}

/* What a QPU starts a run with: nothing, and the limits of its entry. */
static void qpu_service_reset(qpu_service_t *s)
{
    const int q = s->number;
    int value;

    s->in_flight_n     = 0;
    s->deferred        = NULL;
    s->t_next_submit   = 0;
    s->t_retry         = INT64_MAX;
    s->t_unavailable   = 0;
    s->latency_ns      = 0;
    s->t_probe         = 0;
    s->probing         = 0;
    s->put_back        = s->put_back_pinned = 0;
    s->pinned.items    = NULL;
    s->pinned.n        = s->pinned.cap = 0;
    /* Somebody may be reading the counters of the previous service. */
    QPU_STORE(&s->down, 0);
    QPU_STORE(&s->jobs_submitted, 0);
    QPU_STORE(&s->wait_calls, 0);
    QPU_STORE(&s->jobs_failed, 0);
    QPU_STORE(&s->jobs_timed_out, 0);
    QPU_STORE(&s->retries, 0);
    QPU_STORE(&s->submit_deferred, 0);
    QPU_STORE(&s->stray_events, 0);
    QPU_STORE(&s->c_in_flight, 0);
    QPU_STORE(&s->max_in_flight, 0);
    QPU_STORE(&s->jobs_timed, 0);
    QPU_STORE(&s->latency_sum_ns, 0);
    QPU_STORE(&s->jobs_coalesced, 0);
    QPU_STORE(&s->tasks_coalesced, 0);
    QPU_STORE(&s->max_job_tasks, 0);
    QPU_STORE(&s->tasks_moved, 0);

    s->poll_max_ns = (int64_t)(parsec_qpu_params.poll_max_ms > 0 ? parsec_qpu_params.poll_max_ms : 1000) * 1000000LL;
    value = parsec_qpu_list_option_int(q, "submit_interval", parsec_qpu_params.submit_interval_ms);
    s->submit_interval_ns     = (int64_t)(value > 0 ? value : 0) * 1000000LL;
    value = parsec_qpu_list_option_int(q, "job_timeout", parsec_qpu_params.job_timeout_ms);
    s->job_timeout_ns         = (int64_t)(value > 0 ? value : 0) * 1000000LL;
    value = parsec_qpu_list_option_int(q, "unavailable_timeout", parsec_qpu_params.unavailable_timeout_ms);
    s->unavailable_timeout_ns = (int64_t)(value > 0 ? value : 0) * 1000000LL;
    s->cooldown_ns            = s->unavailable_timeout_ns;
    /* At least a millisecond: with none, an unavailable backend would be asked without pause. */
    s->backoff_min_ns         = (int64_t)(parsec_qpu_params.retry_backoff_ms > 0 ? parsec_qpu_params.retry_backoff_ms : 1) * 1000000LL;
    s->backoff_ns             = s->backoff_min_ns;
    /* The attempt number is the low byte of a job's tag. */
    value = parsec_qpu_list_option_int(q, "max_retries", parsec_qpu_params.max_retries);
    s->max_attempts           = 1 + (uint32_t)(value > 0 ? (value < 200 ? value : 200) : 0);
}

/* Bring the service up for this context: every QPU of this rank. Called
 * from a task body, by as many workers at once as there are quantum tasks
 * ready. */
static int qpu_service_start(parsec_context_t *context)
{
    int started = 0;

    if( !parsec_atomic_cas_int32(&pool.state, QPU_SVC_DOWN, QPU_SVC_STARTING) ) {
        /* Somebody else is at it, or was. */
        while( QPU_SVC_STARTING == QPU_LOAD(&pool.state) ) sched_yield();
        return (QPU_SVC_UP == QPU_LOAD(&pool.state)) ? PARSEC_SUCCESS : PARSEC_ERROR;
    }

    pool.context      = context;
    pool.shared.items = NULL;
    pool.shared.n     = pool.shared.cap = 0;
    /* Somebody may be reading the counters of the previous service. */
    QPU_STORE(&pool.stop, 0);
    QPU_STORE(&pool.submissions, NULL);
    QPU_STORE(&pool.next_seq, 0);
    QPU_STORE(&pool.wake_next, 0);
    QPU_STORE(&pool.tasks_submitted, 0);
    QPU_STORE(&pool.tasks_completed, 0);
    QPU_STORE(&pool.empty_jobs, 0);
    QPU_STORE(&pool.tasks_failed, 0);
    QPU_STORE(&pool.tasks_rejected, 0);
    QPU_STORE(&pool.tasks_unavailable, 0);
    QPU_STORE(&pool.waiting, 0);
    QPU_STORE(&pool.c_pending, 0);
    QPU_STORE(&pool.max_pending, 0);

    /* The backends are open since the devices were attached, unless the
     * service was stopped in between. */
    pool.alike = 1;
    for( int i = 0; i < pool.nb; i++ ) {
        qpu_service_t *s = &pool.svc[i];

        qpu_service_reset(s);
        if( PARSEC_SUCCESS != qpu_backend_open(s) ) goto failed;
        s->in_flight = (qpu_job_t**)calloc(s->max_outstanding, sizeof(qpu_job_t*));
        /* A body may read them before the thread has run. */
        QPU_STORE(&s->room, (int32_t)s->max_outstanding);
        QPU_STORE(&s->places, (int32_t)s->max_outstanding);
        if( (s->caps.nb_qubits != pool.svc[0].caps.nb_qubits) || (s->caps.max_shots != pool.svc[0].caps.max_shots) ) {
            pool.alike = 0;
        }
    }

    pool.t_start = parsec_qpu_now_ns();
    for( started = 0; started < pool.nb; started++ ) {
        qpu_service_t *s = &pool.svc[started];

        if( 0 != pthread_create(&s->thread, NULL, qpu_progress_main, s) ) {
            parsec_warning("QPU %d: could not create the progress thread", s->number);
            goto failed;
        }
        parsec_debug_verbose(4, parsec_qpu_output_stream,
                             "QPU %d: service up, backend %s, at most %u jobs outstanding of at most %u task(s) each",
                             s->number, s->backend->ops->name, s->max_outstanding, s->max_batch);
    }
    /* A job can be pushed and the backends woken from here on; the progress
     * threads find them whenever they get to run. */
    (void)parsec_atomic_cas_int32(&pool.state, QPU_SVC_STARTING, QPU_SVC_UP);
    return PARSEC_SUCCESS;

  failed:
    /* The threads that did start have nothing to do, and no body has pushed
     * anything: the service is not up. */
    (void)parsec_atomic_fetch_add_int32(&pool.stop, 1);
    for( int i = 0; i < started; i++ ) {
        (void)pool.svc[i].backend->ops->wakeup(pool.svc[i].backend);
        pthread_join(pool.svc[i].thread, NULL);
    }
    for( int i = 0; i < pool.nb; i++ ) {
        qpu_service_t *s = &pool.svc[i];

        if( NULL != s->backend ) {
            (void)s->backend->ops->close(s->backend);
            s->backend = NULL;
        }
        free(s->in_flight); s->in_flight = NULL;
    }
    (void)parsec_atomic_cas_int32(&pool.state, QPU_SVC_STARTING, QPU_SVC_FAILED);
    return PARSEC_ERROR;
}

int parsec_qpu_service_fini(void)
{
    int abandoned = 0;
    int32_t state = QPU_LOAD(&pool.state);

    if( QPU_SVC_UP == state ) {
        qpu_task_t *item;

        (void)parsec_atomic_fetch_add_int32(&pool.stop, 1);
        for( int i = 0; i < pool.nb; i++ ) {
            (void)pool.svc[i].backend->ops->wakeup(pool.svc[i].backend);
        }
        for( int i = 0; i < pool.nb; i++ ) {
            pthread_join(pool.svc[i].thread, NULL);
        }

        /* The progress threads are gone, so this thread now stands in for
         * them towards the backends. Whatever is left belongs to tasks of a
         * taskpool the application did not wait for. None of them is
         * completed: there may be no scheduler to complete them into. */
        for( int k = 0; k < pool.nb; k++ ) {
            qpu_service_t *s = &pool.svc[k];

            for( uint32_t i = 0; i < s->in_flight_n; i++ ) {
                (void)s->backend->ops->cancel(s->backend, s->in_flight[i]->id);
                abandoned += qpu_job_drop(s->in_flight[i]);
            }
            for( uint32_t i = 0; i < s->pinned.n; i++ ) {
                free(s->pinned.items[i]);
                abandoned++;
            }
            if( NULL != s->deferred ) {
                /* Its job may be at the provider. */
                if( NULL != s->backend->ops->abandon ) {
                    (void)s->backend->ops->abandon(s->backend, s->deferred->tag);
                }
                abandoned += qpu_job_drop(s->deferred);
                s->deferred = NULL;
            }
        }
        for( uint32_t i = 0; i < pool.shared.n; i++ ) {
            free(pool.shared.items[i]);
            abandoned++;
        }
        while( NULL != (item = pool.submissions) ) {
            pool.submissions = item->next;
            free(item);
            abandoned++;
        }
        if( abandoned > 0 ) {
            parsec_warning("QPU: %d quantum task(s) were still outstanding when the context was finalized; "
                           "their jobs were cancelled and the tasks never completed. A taskpool was not waited for.",
                           abandoned);
        }
        for( int k = 0; k < pool.nb; k++ ) {
            qpu_service_t *s = &pool.svc[k];

            (void)s->backend->ops->close(s->backend);
            s->backend = NULL;
            free(s->in_flight);    s->in_flight    = NULL; s->in_flight_n = 0;
            free(s->pinned.items); s->pinned.items = NULL; s->pinned.n    = s->pinned.cap = 0;
            QPU_STORE(&s->room, 0);
            QPU_STORE(&s->places, 0);
        }
        free(pool.shared.items); pool.shared.items = NULL; pool.shared.n = pool.shared.cap = 0;
        QPU_STORE(&pool.waiting, 0);
        pool.context = NULL;
        (void)parsec_atomic_cas_int32(&pool.state, QPU_SVC_UP, QPU_SVC_DOWN);
    } else {
        if( QPU_SVC_FAILED == state ) {
            /* Let the next context try again. */
            (void)parsec_atomic_cas_int32(&pool.state, QPU_SVC_FAILED, QPU_SVC_DOWN);
        }
        for( int k = 0; k < pool.nb; k++ ) {
            if( NULL != pool.svc[k].backend ) {
                /* Opened when the device was attached, and no task ever submitted. */
                (void)pool.svc[k].backend->ops->close(pool.svc[k].backend);
                pool.svc[k].backend = NULL;
            }
        }
    }
    return abandoned;
}

/* What one QPU did, added to what is there. */
static void qpu_stats_add(qpu_service_t *s, parsec_qpu_stats_t *stats)
{
    uint32_t value;

    stats->jobs_submitted  += (uint64_t)ctr_get(&s->jobs_submitted);
    stats->wait_calls      += (uint64_t)ctr_get(&s->wait_calls);
    stats->jobs_failed     += (uint64_t)ctr_get(&s->jobs_failed);
    stats->jobs_timed_out  += (uint64_t)ctr_get(&s->jobs_timed_out);
    stats->retries         += (uint64_t)ctr_get(&s->retries);
    stats->submit_deferred += (uint64_t)ctr_get(&s->submit_deferred);
    stats->stray_events    += (uint64_t)ctr_get(&s->stray_events);
    stats->in_flight       += (uint32_t)ctr_get(&s->c_in_flight);
    value = (uint32_t)ctr_get(&s->max_in_flight);
    if( value > stats->max_in_flight ) stats->max_in_flight = value;
    stats->jobs_timed      += (uint64_t)ctr_get(&s->jobs_timed);
    stats->latency_sum_ns  += ctr_get(&s->latency_sum_ns);
    stats->jobs_coalesced  += (uint64_t)ctr_get(&s->jobs_coalesced);
    stats->tasks_coalesced += (uint64_t)ctr_get(&s->tasks_coalesced);
    value = (uint32_t)ctr_get(&s->max_job_tasks);
    if( value > stats->max_job_tasks ) stats->max_job_tasks = value;
    stats->tasks_moved     += (uint64_t)ctr_get(&s->tasks_moved);
    stats->nb_qpus++;
    if( QPU_LOAD(&s->down) ) stats->down++;
    if( NULL != QPU_LOAD(&s->backend) ) stats->backend_open = 1;
    if( QPU_SVC_UP == QPU_LOAD(&pool.state) ) {
        clockid_t cid;
        struct timespec ts;

        stats->running = 1;
        stats->progress_wall_ns = parsec_qpu_now_ns() - pool.t_start;
        if( (0 == pthread_getcpuclockid(s->thread, &cid)) && (0 == clock_gettime(cid, &ts)) ) {
            stats->progress_cpu_ns += (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
        }
    }
}

void parsec_qpu_service_stats(parsec_qpu_stats_t *stats)
{
    memset(stats, 0, sizeof(*stats));
    stats->tasks_submitted   = (uint64_t)ctr_get(&pool.tasks_submitted);
    stats->tasks_completed   = (uint64_t)ctr_get(&pool.tasks_completed);
    stats->empty_jobs        = (uint64_t)ctr_get(&pool.empty_jobs);
    stats->tasks_failed      = (uint64_t)ctr_get(&pool.tasks_failed);
    stats->tasks_rejected    = (uint64_t)ctr_get(&pool.tasks_rejected);
    stats->tasks_unavailable = (uint64_t)ctr_get(&pool.tasks_unavailable);
    stats->refused_hardware  = (uint64_t)ctr_get(&qpu_refused_hardware);
    stats->pending           = (uint32_t)ctr_get(&pool.c_pending);
    stats->max_pending       = (uint32_t)ctr_get(&pool.max_pending);
    for( int i = 0; i < pool.nb; i++ ) {
        qpu_stats_add(&pool.svc[i], stats);
    }
}

int parsec_qpu_service_stats_of(int qpu, parsec_qpu_stats_t *stats)
{
    memset(stats, 0, sizeof(*stats));
    stats->refused_hardware = (uint64_t)ctr_get(&qpu_refused_hardware);
    for( int i = 0; i < pool.nb; i++ ) {
        if( pool.svc[i].number != qpu ) continue;
        qpu_stats_add(&pool.svc[i], stats);
        return PARSEC_SUCCESS;
    }
    return PARSEC_ERR_NOT_FOUND;
}

/* ------------------------------------------------------------------------
 * Submission, from a task body
 * --------------------------------------------------------------------- */

/* A task wrote, or will have written before it completes, the host copy of
 * one of its flows, which it accesses as `access` says. These are the two
 * statements the PTG compiler generates before a CPU body for each flow the
 * body writes. */
static inline void qpu_host_copy_written(parsec_data_copy_t *copy, uint8_t access)
{
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* An accelerator may hold another copy of the same data: it is not the
     * one to read any more. */
    if( NULL != copy->original ) {
        (void)parsec_data_transfer_ownership_to_copy(copy->original, 0, access);
    }
#else
    (void)access;
#endif  /* defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT) */
    copy->version++;
}

/* A task was pushed: wake a QPU that can take it. `pin` is the QPU it named
 * among those of this rank, or -1. The task is not looked at: it may be
 * gone. Nobody is woken when no QPU has room; the first one that has looks
 * for itself. */
static void qpu_wake_one(int pin)
{
    const int nb = pool.nb;
    int start;

    if( pin >= 0 ) {
        if( QPU_LOAD(&pool.svc[pin].room) > 0 ) {
            (void)pool.svc[pin].backend->ops->wakeup(pool.svc[pin].backend);
        }
        return;
    }
    if( 1 == nb ) {
        if( QPU_LOAD(&pool.svc[0].room) > 0 ) {
            (void)pool.svc[0].backend->ops->wakeup(pool.svc[0].backend);
        }
        return;
    }
    /* Each body starts with another QPU, so that the same one is not woken
     * for everything while the others sleep beside free places. A QPU that
     * is down comes after all those that are up. */
    start = (int)((uint64_t)parsec_atomic_fetch_add_int64(&pool.wake_next, 1) % (uint64_t)nb);
    for( int pass = 0; pass < 2; pass++ ) {
        for( int k = 0; k < nb; k++ ) {
            qpu_service_t *s = &pool.svc[(start + k) % nb];

            if( (QPU_LOAD(&s->room) <= 0) || ((0 == pass) && QPU_LOAD(&s->down)) ) continue;
            (void)s->backend->ops->wakeup(s->backend);
            return;
        }
    }
}

parsec_hook_return_t
parsec_qpu_submit_on(parsec_execution_stream_t *es, parsec_task_t *task, int qpu,
                     parsec_data_copy_t *circuit, parsec_data_copy_t *params,
                     parsec_data_copy_t *result, int shots, uint64_t key)
{
    const parsec_qpu_circuit_t *c;
    const char *why = NULL;
    qpu_task_t *item, *top;
    size_t room;
    int pin = -1;

    if( !pool.enabled || (0 == pool.nb) ) {
        const int elsewhere = pool.enabled ? parsec_qpu_module_placeholder() : -2;

        if( -2 != elsewhere ) {
            /* Its affinity put the task on a rank that has no backend. */
            parsec_warning("QPU: a task of class %s submitted a quantum job on rank %d, which owns no QPU "
                           "(QPU 0 is on rank %d; device_qpu_devices, device_qpu_owner_rank): quantum tasks "
                           "have to be placed on a rank that owns one",
                           task->task_class->name, es->virtual_process->parsec_context->my_rank, elsewhere);
        } else {
            parsec_warning("QPU: a task of class %s submitted a quantum job, and quantum tasks cannot run in this "
                           "context: %s", task->task_class->name, qpu_off_reason);
        }
        return PARSEC_HOOK_RETURN_ERROR;
    }
    if( PARSEC_QPU_ANY != qpu ) {
        for( int i = 0; i < pool.nb; i++ ) {
            if( pool.svc[i].number == qpu ) pin = i;
        }
        if( pin < 0 ) {
            const int owner = parsec_qpu_rank_of(qpu);

            if( owner >= 0 ) {
                parsec_warning("QPU: a task of class %s asked for QPU %d on rank %d, and that QPU is on rank %d: "
                               "a task that names its QPU has to be placed on the rank that owns it",
                               task->task_class->name, qpu, es->virtual_process->parsec_context->my_rank, owner);
            } else {
                parsec_warning("QPU: a task of class %s asked for QPU %d, and this run has no such QPU "
                               "(it has %d, numbered from 0)", task->task_class->name, qpu, parsec_qpu_count());
            }
            return PARSEC_HOOK_RETURN_ERROR;
        }
    }
    if( (NULL == circuit) || (NULL == result) || (shots <= 0) ) {
        parsec_warning("QPU: task of class %s: a circuit, a result and at least one shot are needed",
                       task->task_class->name);
        return PARSEC_HOOK_RETURN_ERROR;
    }
    c = (const parsec_qpu_circuit_t*)PARSEC_DATA_COPY_GET_PTR(circuit);
    if( PARSEC_SUCCESS != parsec_qpu_circuit_check(c, (NULL != circuit->original) ? circuit->original->span : 0, &why) ) {
        parsec_warning("QPU: task of class %s: the circuit flow does not hold a well-formed circuit: %s",
                       task->task_class->name, why);
        return PARSEC_HOOK_RETURN_ERROR;
    }
    if( (c->nb_params > 0) && ((NULL == params) || (NULL == PARSEC_DATA_COPY_GET_PTR(params))) ) {
        parsec_warning("QPU: task of class %s: the circuit has %u parameter slots and no parameter flow",
                       task->task_class->name, c->nb_params);
        return PARSEC_HOOK_RETURN_ERROR;
    }
    /* A copy without a known size is trusted to hold them all. */
    room = ((NULL != params) && (NULL != params->original)) ? params->original->span : 0;
    if( (0 != room) && ((room / sizeof(double)) < c->nb_params) ) {
        parsec_warning("QPU: task of class %s: the circuit has %u parameter slots and the parameter flow holds %zu values",
                       task->task_class->name, c->nb_params, room / sizeof(double));
        return PARSEC_HOOK_RETURN_ERROR;
    }
    /* How many entries the result can take. A copy without a known size is
     * trusted to have room for one entry per shot, as documented. */
    room = (NULL != result->original) ? result->original->span : 0;
    if( (NULL == PARSEC_DATA_COPY_GET_PTR(result)) ||
        ((0 != room) && (room < sizeof(parsec_qpu_result_t))) ) {
        parsec_warning("QPU: task of class %s: the result flow is too small for a result header",
                       task->task_class->name);
        return PARSEC_HOOK_RETURN_ERROR;
    }

    if( QPU_SVC_UP != QPU_LOAD(&pool.state) ) {
        if( PARSEC_SUCCESS != qpu_service_start(es->virtual_process->parsec_context) ) {
            return PARSEC_HOOK_RETURN_ERROR;
        }
    }

    item = (qpu_task_t*)malloc(sizeof(qpu_task_t));
    item->task                = task;
    item->request.circuit     = c;
    item->request.params      = (NULL != params) ? (const double*)PARSEC_DATA_COPY_GET_PTR(params) : NULL;
    item->request.shots       = (uint32_t)shots;
    item->request.max_entries = (0 != room)
        ? (uint32_t)((room - sizeof(parsec_qpu_result_t)) / PARSEC_QPU_ENTRY_SIZE)
        : (uint32_t)shots;
    item->result              = (parsec_qpu_result_t*)PARSEC_DATA_COPY_GET_PTR(result);
    item->seq                 = (uint64_t)parsec_atomic_fetch_add_int64(&pool.next_seq, 1);
    /* Unique per task, up to the odds of two 56-bit hashes being equal: the
     * taskpool, the class and the key of the task, mixed in one after the
     * other. The attempt number is added at each submission. */
    item->tag                 = parsec_qpu_mix64(parsec_qpu_mix64(parsec_qpu_mix64((uint64_t)task->taskpool->taskpool_id)
                                                                 ^ (uint64_t)task->task_class->task_class_id)
                                                ^ (uint64_t)task->task_class->make_key(task->taskpool, task->locals)) << 8;
    item->request.key         = (PARSEC_QPU_KEY_AUTO == key) ? (item->tag >> 8) : key;
    item->request.attempt     = 0;
    item->t_hook              = parsec_qpu_now_ns();
    item->t_submit            = 0;
    item->priority            = task->priority;
    item->vp_id               = es->virtual_process->vp_id;
    item->attempts            = 0;
    item->formed              = 0;
    item->batch               = 1;
    item->may_run             = 0;
    item->pin                 = (int16_t)pin;
    item->ran_on              = -1;
    item->avoid               = -1;
    item->bound               = -1;
    item->typed               = (NULL != task->selected_device) && (0 != (PARSEC_DEV_QPU & task->selected_device->type));
    ctr_add(&pool.tasks_submitted, 1);

    /* The device layer selected a QPU of this rank for the task, and the
     * scheduler added the task's load to it. Which QPU runs the job is
     * decided when one has a place for it, so the task is on no QPU while
     * it waits: take the load off again. The QPU that takes the task makes
     * itself the task's device and carries its load from then on. */
    if( item->typed ) {
        (void)parsec_atomic_fetch_add_int64(&task->selected_device->device_load, -task->load);
        task->load = 0;
    }

    /* The code a JDF generates around a body says that the body writes its
     * output flows only when the body is of a CPU type. A body of type QPU
     * works on host copies all the same, and its result is written there
     * later, from another thread. So say here, for every flow the task
     * writes, what that code says for a CPU body: the host copy is the one
     * that is right from now on, and it is a version further. (A DTD task
     * is told so by the wrapper of its class, whatever its type.) */
    if( (PARSEC_TASK_CLASS_TYPE_PTG == task->task_class->task_class_type) && (task->selected_chore >= 0) &&
        (PARSEC_DEV_QPU & task->task_class->incarnations[task->selected_chore].type) ) {
        int result_is_written = 0;

        for( int i = 0; i < task->task_class->nb_flows; i++ ) {
            const parsec_flow_t *flow = task->task_class->out[i];
            parsec_data_copy_t *copy;

            if( (NULL == flow) || !(PARSEC_FLOW_ACCESS_WRITE & flow->flow_flags) ) continue;
            if( NULL == (copy = task->data[flow->flow_index].data_in) ) continue;
            if( copy == result ) result_is_written = 1;
            qpu_host_copy_written(copy, (uint8_t)(PARSEC_FLOW_ACCESS_MASK & flow->flow_flags));
        }
        /* A result given in a flow that is not declared as written. */
        if( !result_is_written ) qpu_host_copy_written(result, PARSEC_FLOW_ACCESS_WRITE);
    }

    /* From the push on, the task is not this thread's any more: it may be
     * completed before this function returns. Whatever this thread still
     * does after the push, here and in the scheduler once the body has
     * returned, it does while the task can be completed by somebody else,
     * so the wake-up that follows must not block, and must not look at the
     * task. */
    do {
        top = QPU_LOAD(&pool.submissions);
        item->next = top;
    } while( !parsec_atomic_cas_ptr(&pool.submissions, top, item) );
    qpu_wake_one(pin);

    return PARSEC_HOOK_RETURN_ASYNC;
}

parsec_hook_return_t
parsec_qpu_submit(parsec_execution_stream_t *es, parsec_task_t *task,
                  parsec_data_copy_t *circuit, parsec_data_copy_t *params,
                  parsec_data_copy_t *result, int shots, uint64_t key)
{
    return parsec_qpu_submit_on(es, task, PARSEC_QPU_ANY, circuit, params, result, shots, key);
}

parsec_hook_return_t
parsec_qpu_submit_dtd_on(parsec_execution_stream_t *es, parsec_task_t *task, int qpu,
                         int circuit_index, int params_index, int result_index,
                         int shots, uint64_t key)
{
    parsec_data_copy_t *copies[3] = { NULL, NULL, NULL };
    int indices[3] = { circuit_index, params_index, result_index };

    for( int i = 0; i < 3; i++ ) {
        if( indices[i] < 0 ) continue;
        if( indices[i] >= task->task_class->nb_flows ) {
            parsec_warning("QPU: task of class %s has no data argument %d", task->task_class->name, indices[i]);
            return PARSEC_HOOK_RETURN_ERROR;
        }
        /* As parsec_dtd_unpack_args() does: data_out is set only when a
         * device elected a copy of its own, otherwise the copy to work on
         * is data_in. */
        copies[i] = task->data[indices[i]].data_out;
        if( NULL == copies[i] ) copies[i] = task->data[indices[i]].data_in;
    }
    return parsec_qpu_submit_on(es, task, qpu, copies[0], copies[1], copies[2], shots, key);
}

parsec_hook_return_t
parsec_qpu_submit_dtd(parsec_execution_stream_t *es, parsec_task_t *task,
                      int circuit_index, int params_index, int result_index,
                      int shots, uint64_t key)
{
    return parsec_qpu_submit_dtd_on(es, task, PARSEC_QPU_ANY, circuit_index, params_index, result_index, shots, key);
}
