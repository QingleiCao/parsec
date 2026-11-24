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
 *     file, and within it the progress thread alone. The worker that ran the
 *     body returns ASYNC and leaves the task alone.
 *   - From the hand-off on, the scheduler and then whichever worker picks
 *     the task up. The progress thread keeps no pointer to the task, to its
 *     data or to its taskpool past that point: the worker may complete the
 *     task, and the taskpool may terminate and be freed, at any moment.
 *
 * Where it runs. On one rank, the one that owns the QPU. Its backend is
 * opened when the QPU device of that rank is attached, so that a backend
 * that cannot be used gives no device; the progress thread starts with the
 * first submission, and both stop when the context is finalized, before the
 * workers are joined. A submission on any other rank is refused.
 *
 * How a task is completed. The progress thread never calls
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
 *     takes longer than device_qpu_job_timeout for each task it carries (it
 *     is then cancelled) is submitted again, as a new job with a new tag,
 *     until the task has had 1 + device_qpu_max_retries jobs.
 *   - A submission the backend cannot take for now is asked again, with the
 *     same tag, after a back-off that doubles; nothing else is submitted in
 *     between, because the provider may have the job all the same. When
 *     that has lasted device_qpu_unavailable_timeout, the submission is
 *     given up on at the backend (which cancels the job if it is there),
 *     and the task and every pending one complete as "unavailable".
 *   - A final state for a job that is not in flight is counted and ignored:
 *     the first final state of a job is the only one acted upon.
 *
 * How tasks share a job. A job carries the circuit of one task, or those of
 * several (device_qpu_max_batch, and what the backend takes per job). The
 * service never holds a task back to fill a job: a task that arrives while
 * a job can be submitted goes at once. Tasks share a job only when they wait
 * together and there are more of them than jobs that can be submitted at
 * that moment; then they are spread evenly over those jobs, in the order of
 * the heap. Each task keeps its own result and completes on its own. Tasks
 * that share a job ask for the same number of shots and are at their first
 * job: a job that fails, with one task or with several, sends each of its
 * tasks back to be retried alone. Sharing a job takes nothing from a task
 * that it had alone: a job of n tasks may stay at the backend n times as
 * long as a job of one before it is given up on.
 */

#define QPU_LOAD(PTR)        PARSEC_QPU_LOAD(PTR)
#define QPU_STORE(PTR, VAL)  PARSEC_QPU_STORE(PTR, VAL)

#define QPU_SVC_DOWN      0
#define QPU_SVC_STARTING  1
#define QPU_SVC_UP        2
#define QPU_SVC_FAILED    3

#define QPU_MAX_EVENTS    64
#define QPU_MAX_BACKENDS  8
/* The longest the service waits before asking an unavailable backend again. */
#define QPU_BACKOFF_MAX_NS  30000000000LL

/* A task between its body and its completion. */
typedef struct qpu_task_s {
    struct qpu_task_s    *next;        /* submission stack */
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
    uint32_t              batch;       /* tasks in the last job the backend took for it */
    uint16_t              attempts;    /* jobs the backend took for this task, or refused for good */
} qpu_task_t;

/* What the backend is asked to run in one go: the circuit of one task, or
 * those of several that were waiting together. It exists from the moment it
 * is put together until its end is known; its tasks outlive it. */
typedef struct qpu_job_s {
    parsec_qpu_job_id_t   id;          /* at the backend, while in flight */
    uint64_t              tag;
    int64_t               t_submit;    /* when it was first asked for */
    int64_t               deadline;    /* in flight: when it has taken too long */
    uint32_t              nb;          /* tasks it carries, at least one */
    parsec_qpu_request_t *requests;    /* [nb] what the backend is given, in the order of the heap */
    qpu_task_t          **tasks;       /* [nb] */
    parsec_qpu_result_t **results;     /* [nb] where the backend writes */
} qpu_job_t;

typedef struct qpu_service_s {
    int32_t                    state;      /* QPU_SVC_* */
    int32_t                    enabled;    /* the current context may use the service */
    int32_t                    stop;

    parsec_context_t          *context;
    pthread_t                  thread;
    parsec_execution_stream_t  es;         /* the progress thread's own stream */
    parsec_qpu_backend_t      *backend;     /* open from the attachment of the device, or from the start */
    parsec_qpu_caps_t          caps;
    parsec_device_qpu_module_t *device;     /* the QPU device of this process, while the thread runs */
    uint32_t                   max_outstanding;
    uint32_t                   max_batch;              /* tasks a job may carry */
    uint32_t                   max_attempts;           /* jobs a task may have: 1 + retries */
    int64_t                    poll_max_ns;
    int64_t                    submit_interval_ns;
    int64_t                    job_timeout_ns;         /* 0: none */
    int64_t                    unavailable_timeout_ns;
    int64_t                    backoff_min_ns;
    int64_t                    t_start;

    qpu_task_t                *submissions; /* stack, pushed by task bodies */
    int64_t                    next_seq;

    /* Used by the progress thread only. */
    qpu_task_t               **pending;     /* heap: priority, then arrival */
    uint32_t                   pending_n, pending_cap;
    qpu_job_t                **in_flight;
    uint32_t                   in_flight_n;
    qpu_job_t                 *deferred;    /* its submission was not taken: it is the next one asked */
    int64_t                    t_next_submit; /* nothing is submitted before */
    int64_t                    t_unavailable; /* since when the backend takes nothing; 0 when it does */
    int64_t                    backoff_ns;
    int64_t                    latency_ns;    /* moving average of the time a job takes; 0 before the first */

    /* Counters, read by other threads. */
    int64_t                    tasks_submitted;
    int64_t                    jobs_submitted;
    int64_t                    tasks_completed;
    int64_t                    empty_jobs;
    int64_t                    wait_calls;
    int64_t                    jobs_failed;
    int64_t                    jobs_timed_out;
    int64_t                    retries;
    int64_t                    submit_deferred;
    int64_t                    stray_events;
    int64_t                    tasks_failed;
    int64_t                    tasks_rejected;
    int64_t                    tasks_unavailable;
    int64_t                    c_pending, c_in_flight;
    int64_t                    max_pending, max_in_flight;
    int64_t                    jobs_timed, latency_sum_ns;
    int64_t                    jobs_coalesced, tasks_coalesced, max_job_tasks;
} qpu_service_t;

static qpu_service_t svc;
/* Not reset when a service starts: it is about the process. */
static int64_t qpu_refused_hardware = 0;
/* Why task bodies cannot use the service, when they cannot. */
static const char *qpu_off_reason = "device_qpu_enabled is not set";

static inline void    ctr_add(int64_t *c, int64_t d) { (void)parsec_atomic_fetch_add_int64(c, d); }
static inline int64_t ctr_get(int64_t *c)            { return QPU_LOAD(c); }
/* For high-water marks, which only the progress thread moves. */
static inline void    ctr_max(int64_t *c, int64_t v)
{
    int64_t cur = QPU_LOAD(c);
    if( v > cur ) ctr_add(c, v - cur);
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
 * Pending tasks: a binary heap, highest task priority first and, among equal
 * priorities, first come first served.
 * --------------------------------------------------------------------- */

static inline int task_before(const qpu_task_t *a, const qpu_task_t *b)
{
    if( a->priority != b->priority ) return a->priority > b->priority;
    return a->seq < b->seq;
}

static void pending_insert(qpu_task_t *item)
{
    if( svc.pending_n == svc.pending_cap ) {
        svc.pending_cap = (0 == svc.pending_cap) ? 64 : 2 * svc.pending_cap;
        svc.pending = (qpu_task_t**)realloc(svc.pending, svc.pending_cap * sizeof(qpu_task_t*));
    }
    uint32_t i = svc.pending_n++;
    while( i > 0 ) {
        uint32_t parent = (i - 1) / 2;
        if( !task_before(item, svc.pending[parent]) ) break;
        svc.pending[i] = svc.pending[parent];
        i = parent;
    }
    svc.pending[i] = item;
}

static qpu_task_t *pending_remove_top(void)
{
    qpu_task_t *top = svc.pending[0], *last = svc.pending[--svc.pending_n];
    uint32_t i = 0;
    for(;;) {
        uint32_t child = 2 * i + 1;
        if( child >= svc.pending_n ) break;
        if( (child + 1 < svc.pending_n) && task_before(svc.pending[child + 1], svc.pending[child]) ) child++;
        if( !task_before(svc.pending[child], last) ) break;
        svc.pending[i] = svc.pending[child];
        i = child;
    }
    if( svc.pending_n > 0 ) svc.pending[i] = last;
    return top;
}

/* The same with the counters that others read. The tasks of a job that
 * waits to be asked for again (svc.deferred) are pending too. */
static void pending_push(qpu_task_t *item)
{
    pending_insert(item);
    ctr_add(&svc.c_pending, 1);
    ctr_max(&svc.max_pending, ctr_get(&svc.c_pending));
}

static qpu_task_t *pending_pop(void)
{
    ctr_add(&svc.c_pending, -1);
    return pending_remove_top();
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
        svc.context->virtual_processes[item->vp_id]->execution_streams[0];

    /* Everything the service keeps about the task goes first, counters
     * included: once the task is scheduled, a worker may complete it and the
     * application may read the counters or tear the taskpool down. */
    ctr_add(&svc.tasks_completed, 1);
    /* The device counts the tasks it executed itself, as an accelerator
     * does: the scheduler counts only tasks whose hook did not return ASYNC. */
    (void)parsec_atomic_fetch_inc_int64((int64_t*)&svc.device->super.executed_tasks);
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
static void qpu_note_latency(int64_t took)
{
    if( took < 1 ) took = 1;
    svc.latency_ns = (0 == svc.latency_ns) ? took : svc.latency_ns + (took - svc.latency_ns) / 8;
    if( svc.latency_ns < 1 ) svc.latency_ns = 1;
    ctr_add(&svc.jobs_timed, 1);
    ctr_add(&svc.latency_sum_ns, took);
    QPU_STORE(&svc.device->latency_ns, svc.latency_ns);
}

/* The parts of the result header that are the service's to write. */
static void qpu_finish(qpu_task_t *item)
{
    parsec_qpu_result_t *result = item->result;

    result->version         = PARSEC_QPU_RESULT_VERSION;
    result->shots_requested = item->request.shots;
    result->attempts        = item->attempts;
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
        ctr_add(&svc.tasks_failed, 1);
        if( PARSEC_QPU_STATUS_REJECTED == status )    ctr_add(&svc.tasks_rejected, 1);
        if( PARSEC_QPU_STATUS_UNAVAILABLE == status ) ctr_add(&svc.tasks_unavailable, 1);
        parsec_debug_verbose(5, parsec_qpu_output_stream,
                             "QPU: a task of class %s completes with status %d after %u job(s)",
                             item->task->task_class->name, (int)status, (unsigned)item->attempts);
    }
    qpu_finish(item);
}

/* One job of a task is over without an answer. The task gets another one,
 * unless it had its share. */
static void qpu_attempt_failed(qpu_task_t *item, parsec_qpu_status_t status)
{
    if( item->attempts < svc.max_attempts ) {
        /* It kept its place in the arrival order, and it has had a job: its
         * next one is for it alone (qpu_can_share). */
        pending_push(item);
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

/* Put the next job together, from the top of the heap, which is not empty.
 * The job is submitted at once whatever it holds; the question is only how
 * many of the tasks that wait go with the first one. */
static qpu_job_t *qpu_job_form(int64_t now)
{
    qpu_task_t *lead = pending_pop();
    uint32_t want = 1;
    qpu_job_t *job;

    if( (svc.max_batch > 1) && (0 == lead->attempts) && (svc.pending_n > 0) ) {
        /* The jobs that can be submitted right now: one per free place at
         * the backend, or just this one when submissions have to be spaced.
         * The tasks that wait are spread evenly over them, so that tasks
         * share a job only when there are more of them than jobs, and no
         * place stays empty next to a job that holds everything. */
        const uint32_t jobs  = (svc.submit_interval_ns > 0) ? 1 : (svc.max_outstanding - svc.in_flight_n);
        const uint32_t tasks = svc.pending_n + 1;

        want = (tasks + jobs - 1) / jobs;
        if( want > svc.max_batch ) want = svc.max_batch;
    }

    job = qpu_job_new(want);
    job->tasks[job->nb++] = lead;
    /* In the order of the heap, and no further than the first task that
     * cannot share: nobody overtakes a task that was there before. */
    while( (job->nb < want) && (svc.pending_n > 0) && qpu_can_share(lead, svc.pending[0]) ) {
        job->tasks[job->nb++] = pending_pop();
    }

    /* A new job of a task has a new attempt number, and the job the tag of
     * its first task with that number: no two jobs have the same tag, since
     * a task has one job per attempt. */
    job->tag      = lead->tag + (uint64_t)lead->attempts + 1;
    job->t_submit = now;
    for( uint32_t i = 0; i < job->nb; i++ ) {
        qpu_task_t *item = job->tasks[i];

        item->request.attempt = (uint32_t)item->attempts + 1;
        item->t_submit        = now;
        job->requests[i]      = item->request;
        job->results[i]       = item->result;
    }
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
static void qpu_job_failed(qpu_job_t *job, parsec_qpu_status_t status)
{
    for( uint32_t i = 0; i < job->nb; i++ ) {
        qpu_attempt_failed(job->tasks[i], status);
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

static void qpu_job_event(const parsec_qpu_event_t *event)
{
    qpu_job_t *job = NULL;

    for( uint32_t i = 0; i < svc.in_flight_n; i++ ) {
        if( svc.in_flight[i]->id == event->id ) {
            job = svc.in_flight[i];
            svc.in_flight[i] = svc.in_flight[--svc.in_flight_n];
            break;
        }
    }
    if( NULL == job ) {
        /* A final state for a job that is not in flight: a second report of
         * an end that was handled, or the end of a job that was cancelled.
         * The first final state of a job is the only one acted upon. */
        ctr_add(&svc.stray_events, 1);
        parsec_debug_verbose(5, parsec_qpu_output_stream,
                             "QPU: ignoring a final state for job %llu, which is not in flight",
                             (unsigned long long)event->id);
        return;
    }
    ctr_add(&svc.c_in_flight, -1);

    if( PARSEC_QPU_JOB_DONE == event->state ) {
        for( uint32_t i = 0; i < job->nb; i++ ) {
            memset(job->results[i], 0, sizeof(parsec_qpu_result_t));
        }
        if( PARSEC_SUCCESS == svc.backend->ops->results(svc.backend, job->id, job->results, (int)job->nb) ) {
            /* Every result is in its buffer before the first task is given
             * back: from then on the tasks complete in any order. */
            qpu_note_latency(parsec_qpu_now_ns() - job->t_submit);
            for( uint32_t i = 0; i < job->nb; i++ ) {
                qpu_finish(job->tasks[i]);
            }
            free(job);
            return;
        }
        /* Done, and nothing to show for it. Whatever the backend wrote before
         * it gave up is not an answer: the next job of each task starts from
         * a clean header, and so does a task that ends here. */
        ctr_add(&svc.jobs_failed, 1);
        qpu_job_failed(job, PARSEC_QPU_STATUS_FAILED);
        return;
    }
    if( PARSEC_QPU_JOB_CANCELLED == event->state ) {
        /* Not by us: somebody at the provider did. That is not retried. */
        qpu_job_finish_without_result(job, PARSEC_QPU_STATUS_CANCELLED);
        return;
    }
    ctr_add(&svc.jobs_failed, 1);
    qpu_job_failed(job, PARSEC_QPU_STATUS_FAILED);
}

/* Cancel the jobs that have taken too long. */
static void qpu_check_deadlines(void)
{
    const int64_t now = parsec_qpu_now_ns();

    for( uint32_t i = 0; i < svc.in_flight_n; ) {
        qpu_job_t *job = svc.in_flight[i];
        if( job->deadline > now ) { i++; continue; }
        svc.in_flight[i] = svc.in_flight[--svc.in_flight_n];
        ctr_add(&svc.c_in_flight, -1);
        /* From here on the job is not in flight: if the backend reports its
         * end all the same, the report finds nothing and is ignored. */
        (void)svc.backend->ops->cancel(svc.backend, job->id);
        ctr_add(&svc.jobs_timed_out, 1);
        qpu_job_failed(job, PARSEC_QPU_STATUS_TIMED_OUT);
    }
}

/* ------------------------------------------------------------------------
 * The progress thread
 * --------------------------------------------------------------------- */

/* Take everything task bodies submitted since the last time. */
static void qpu_drain_submissions(void)
{
    qpu_task_t *stack, *item;

    do {
        stack = QPU_LOAD(&svc.submissions);
        if( NULL == stack ) return;
    } while( !parsec_atomic_cas_ptr(&svc.submissions, stack, NULL) );

    while( NULL != (item = stack) ) {
        stack = item->next;
        item->next = NULL;
        if( PARSEC_QPU_CIRCUIT_EMPTY & item->request.circuit->flags ) {
            /* Nothing to run: no backend, no slot, no entry. */
            ctr_add(&svc.empty_jobs, 1);
            item->t_submit = parsec_qpu_now_ns();
            qpu_finish_without_result(item, PARSEC_QPU_STATUS_OK);
            continue;
        }
        if( (item->request.circuit->nb_qubits > svc.caps.nb_qubits) ||
            ((0 != svc.caps.max_shots) && (item->request.shots > svc.caps.max_shots)) ) {
            /* No job of this backend can do that: it would fail every time. */
            qpu_finish_without_result(item, PARSEC_QPU_STATUS_REJECTED);
            continue;
        }
        pending_push(item);
    }
}

/* The backend has taken nothing for too long. The submission that is being
 * asked for is given up on, which settles it at the backend: the provider
 * may have that job, and nobody may read a task's buffers once the task is
 * completed. Then its tasks and every pending one complete as unavailable.
 * Whoever comes next starts asking afresh. */
static void qpu_flush_unavailable(qpu_job_t *asked, int64_t now)
{
    parsec_debug_verbose(3, parsec_qpu_output_stream,
                         "QPU: backend %s has taken no job for %.3f s; %u pending task(s) complete as unavailable",
                         svc.backend->ops->name, (now - svc.t_unavailable) * 1e-9, svc.pending_n + asked->nb);
    if( NULL != svc.backend->ops->abandon ) {
        (void)svc.backend->ops->abandon(svc.backend, asked->tag);
    }
    qpu_job_finish_without_result(asked, PARSEC_QPU_STATUS_UNAVAILABLE);
    while( svc.pending_n > 0 ) {
        qpu_finish_without_result(pending_pop(), PARSEC_QPU_STATUS_UNAVAILABLE);
    }
    svc.t_unavailable = 0;
    svc.backoff_ns    = svc.backoff_min_ns;
}

static inline int qpu_unavailable_for_too_long(int64_t now)
{
    return (0 != svc.t_unavailable) && (svc.unavailable_timeout_ns > 0) &&
           ((now - svc.t_unavailable) >= svc.unavailable_timeout_ns);
}

/* Give the backend as many jobs as the limit, the minimum interval between
 * two submissions and the state of the backend allow. */
static void qpu_admit(void)
{
    int64_t now = parsec_qpu_now_ns();

    while( svc.in_flight_n < svc.max_outstanding ) {
        qpu_job_t *job = svc.deferred;
        int rc;

        if( (NULL == job) && (0 == svc.pending_n) ) break;
        if( (NULL != job) && qpu_unavailable_for_too_long(now) ) {
            /* Asked for long enough: it is not asked again. */
            svc.deferred = NULL;
            ctr_add(&svc.c_pending, -(int64_t)job->nb);
            qpu_flush_unavailable(job, now);
            break;
        }
        if( now < svc.t_next_submit ) break;
        if( NULL != job ) {
            /* A submission that is asked again is the job it was: the same
             * tasks, the same tag, and the time of its first asking. */
            svc.deferred = NULL;
            ctr_add(&svc.c_pending, -(int64_t)job->nb);
        } else {
            job = qpu_job_form(now);
        }

        rc = svc.backend->ops->submit(svc.backend, job->requests, (int)job->nb, job->tag, &job->id);
        now = parsec_qpu_now_ns();
        /* The minimum interval is between two submissions, whatever the
         * backend made of them. */
        svc.t_next_submit = now + svc.submit_interval_ns;

        if( PARSEC_ERR_OUT_OF_RESOURCE == rc ) {
            /* Not taken, or not known to be. The provider may have the job
             * all the same, so nothing else is submitted until this one is
             * settled: asking again with the same tag cannot run it twice. */
            ctr_add(&svc.submit_deferred, 1);
            if( 0 == svc.t_unavailable ) svc.t_unavailable = now;
            if( qpu_unavailable_for_too_long(now) ) {
                qpu_flush_unavailable(job, now);
                break;
            }
            svc.deferred = job;
            ctr_add(&svc.c_pending, (int64_t)job->nb);
            if( svc.t_next_submit < now + svc.backoff_ns ) svc.t_next_submit = now + svc.backoff_ns;
            svc.backoff_ns = (2 * svc.backoff_ns < QPU_BACKOFF_MAX_NS) ? 2 * svc.backoff_ns : QPU_BACKOFF_MAX_NS;
            break;
        }

        /* The backend answered for good, one way or the other: it is there,
         * and that was one of the jobs of each task. */
        svc.t_unavailable = 0;
        svc.backoff_ns    = svc.backoff_min_ns;
        if( job->requests[0].attempt > 1 ) ctr_add(&svc.retries, 1);
        for( uint32_t i = 0; i < job->nb; i++ ) {
            job->tasks[i]->attempts++;
        }

        if( PARSEC_SUCCESS == rc ) {
            for( uint32_t i = 0; i < job->nb; i++ ) {
                job->tasks[i]->batch = job->nb;
            }
            /* The time a job may take is that of a job of one task, for
             * each task it carries: tasks that would each have been given
             * that long are not cut short for having gone together. */
            job->deadline = (svc.job_timeout_ns > 0) ? now + svc.job_timeout_ns * (int64_t)job->nb : INT64_MAX;
            svc.in_flight[svc.in_flight_n++] = job;
            ctr_add(&svc.jobs_submitted, 1);
            if( job->nb > 1 ) {
                ctr_add(&svc.jobs_coalesced, 1);
                ctr_add(&svc.tasks_coalesced, (int64_t)job->nb);
            }
            ctr_max(&svc.max_job_tasks, (int64_t)job->nb);
            ctr_add(&svc.c_in_flight, 1);
            ctr_max(&svc.max_in_flight, (int64_t)svc.in_flight_n);
            continue;
        }

        /* Refused. */
        ctr_add(&svc.jobs_failed, 1);
        qpu_job_failed(job, PARSEC_QPU_STATUS_FAILED);
    }
}

/* How long the progress thread may sleep: until the next thing it has to do
 * by itself, and never longer than device_qpu_poll_max. */
static int64_t qpu_wait_timeout(void)
{
    const int64_t now = parsec_qpu_now_ns();
    int64_t until = now + svc.poll_max_ns;

    if( (svc.in_flight_n < svc.max_outstanding) && ((NULL != svc.deferred) || (svc.pending_n > 0)) ) {
        if( svc.t_next_submit < until ) until = svc.t_next_submit;
        /* The moment a submission has been asked for long enough. */
        if( (NULL != svc.deferred) && (0 != svc.t_unavailable) && (svc.unavailable_timeout_ns > 0) &&
            (svc.t_unavailable + svc.unavailable_timeout_ns < until) ) {
            until = svc.t_unavailable + svc.unavailable_timeout_ns;
        }
    }
    for( uint32_t i = 0; i < svc.in_flight_n; i++ ) {
        if( svc.in_flight[i]->deadline < until ) until = svc.in_flight[i]->deadline;
    }
    return (until > now) ? until - now : 0;
}

static void *qpu_progress_main(void *arg)
{
    parsec_context_t *context = (parsec_context_t*)arg;
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
    memset(&svc.es, 0, sizeof(svc.es));
    svc.es.th_id            = 0;
    svc.es.core_id          = -1;
    svc.es.socket_id        = -1;
    svc.es.virtual_process  = context->virtual_processes[0];
    svc.es.scheduler_object = NULL;
    svc.es.next_task        = (parsec_task_t*)0xdeadbeef;  /* never used, and never NULL */
    parsec_set_my_execution_stream(&svc.es);

    for(;;) {
        qpu_drain_submissions();
        if( QPU_LOAD(&svc.stop) ) break;
        qpu_check_deadlines();
        qpu_admit();

        /* Sleep in the backend. A submission wakes it, and so does the end
         * of a job; the timeout is for what the service has to do by itself. */
        ctr_add(&svc.wait_calls, 1);
        int n = svc.backend->ops->wait(svc.backend, qpu_wait_timeout(), events, QPU_MAX_EVENTS);
        if( n < 0 ) {
            struct timespec pause = { 0, 1000000 };
            parsec_warning("QPU: backend %s failed while waiting (%d)", svc.backend->ops->name, n);
            nanosleep(&pause, NULL);    /* whatever it is, do not spin on it */
            n = 0;
        }
        for( int i = 0; i < n; i++ ) {
            qpu_job_event(&events[i]);
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
    svc.enabled = 0;
    qpu_off_reason = "device_qpu_enabled is not set";
    if( (parsec_qpu_output_stream >= 0) && (parsec_qpu_output_stream != parsec_device_output) ) {
        parsec_output_close(parsec_qpu_output_stream);
    }
    parsec_qpu_output_stream = -1;
}

int parsec_qpu_service_enable(void)
{
    const parsec_qpu_backend_ops_t *ops;

    if( svc.enabled ) return PARSEC_SUCCESS;
    /* Settle now, while the context is being initialized, whether the
     * backend that was asked for may be used at all. */
    if( NULL == (ops = parsec_qpu_backend_find(parsec_qpu_params.backend)) ) {
        parsec_warning("QPU: no backend named %s; quantum tasks cannot run in this context",
                       (NULL != parsec_qpu_params.backend) ? parsec_qpu_params.backend : "(null)");
        qpu_off_reason = "the backend named by device_qpu_backend does not exist";
        return PARSEC_ERR_NOT_FOUND;
    }
    if( qpu_backend_refused(ops, 0) ) {
        qpu_off_reason = "the backend drives a quantum device and device_qpu_allow_hardware is not set";
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    svc.enabled = 1;
    /* What is said to a body that submits if no QPU device makes it into
     * the context all the same; opening the backend has a better reason
     * when that is what fails. */
    qpu_off_reason = "the QPU device could not be attached to this context";
    /* Stop before the workers are joined and the profiling system is torn
     * down, which is long before the device components are closed. */
    parsec_context_at_fini(qpu_at_fini, NULL);
    return PARSEC_SUCCESS;
}

/* Open the backend that was asked for and learn what it is. Called from the
 * main thread when the QPU device is attached, and by whoever brings the
 * service up when no backend is open then, which is the case after the
 * service was stopped by hand. Never by two threads at once. */
static int qpu_backend_open(void)
{
    const parsec_qpu_backend_ops_t *ops;
    int rc;

    if( NULL != svc.backend ) return PARSEC_SUCCESS;
    if( NULL == (ops = parsec_qpu_backend_find(parsec_qpu_params.backend)) ) {
        parsec_warning("QPU: no backend named %s", parsec_qpu_params.backend);
        return PARSEC_ERR_NOT_FOUND;
    }
    if( qpu_backend_refused(ops, 0) ) return PARSEC_ERR_NOT_SUPPORTED;
    if( (PARSEC_SUCCESS != (rc = ops->open(&svc.backend, 0))) || (NULL == svc.backend) ) {
        parsec_warning("QPU: backend %s did not open (%d)", ops->name, rc);
        svc.backend = NULL;
        return (PARSEC_SUCCESS != rc) ? rc : PARSEC_ERROR;
    }
    memset(&svc.caps, 0, sizeof(svc.caps));
    (void)ops->capabilities(svc.backend, &svc.caps);
    if( qpu_backend_refused(ops, (int)svc.caps.is_physical) ) {
        /* It did not say so before it was opened. Nothing was submitted. */
        (void)ops->close(svc.backend);
        svc.backend = NULL;
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    svc.max_outstanding = (uint32_t)(parsec_qpu_params.max_outstanding > 0 ? parsec_qpu_params.max_outstanding : 1);
    if( (svc.caps.max_jobs > 0) && (svc.caps.max_jobs < svc.max_outstanding) ) {
        svc.max_outstanding = svc.caps.max_jobs;
    }
    /* A job carries as many tasks as was asked for and as the backend takes
     * circuits; a backend that does not say takes one. */
    svc.max_batch = (uint32_t)(parsec_qpu_params.max_batch > 0 ? parsec_qpu_params.max_batch : 1);
    if( svc.max_batch > ((svc.caps.max_circuits_per_job > 0) ? svc.caps.max_circuits_per_job : 1) ) {
        svc.max_batch = (svc.caps.max_circuits_per_job > 0) ? svc.caps.max_circuits_per_job : 1;
    }
    return PARSEC_SUCCESS;
}

int parsec_qpu_service_open(parsec_qpu_caps_t *caps, uint32_t *max_outstanding, uint32_t *max_batch)
{
    int rc = qpu_backend_open();

    if( PARSEC_SUCCESS != rc ) {
        qpu_off_reason = "the backend named by device_qpu_backend could not be used, so this context has no QPU device";
        return rc;
    }
    *caps            = svc.caps;
    *max_outstanding = svc.max_outstanding;
    *max_batch       = svc.max_batch;
    return PARSEC_SUCCESS;
}

/* Bring the service up for this context. Called from a task body, by as
 * many workers at once as there are quantum tasks ready. */
static int qpu_service_start(parsec_context_t *context)
{
    if( !parsec_atomic_cas_int32(&svc.state, QPU_SVC_DOWN, QPU_SVC_STARTING) ) {
        /* Somebody else is at it, or was. */
        while( QPU_SVC_STARTING == QPU_LOAD(&svc.state) ) sched_yield();
        return (QPU_SVC_UP == QPU_LOAD(&svc.state)) ? PARSEC_SUCCESS : PARSEC_ERROR;
    }

    svc.context         = context;
    svc.pending         = NULL;
    svc.pending_n       = svc.pending_cap = 0;
    svc.in_flight_n     = 0;
    svc.deferred        = NULL;
    svc.t_next_submit   = 0;
    svc.t_unavailable   = 0;
    /* Somebody may be reading the counters of the previous service. */
    QPU_STORE(&svc.stop, 0);
    QPU_STORE(&svc.submissions, NULL);
    QPU_STORE(&svc.next_seq, 0);
    QPU_STORE(&svc.tasks_submitted, 0);
    QPU_STORE(&svc.jobs_submitted, 0);
    QPU_STORE(&svc.tasks_completed, 0);
    QPU_STORE(&svc.empty_jobs, 0);
    QPU_STORE(&svc.wait_calls, 0);
    QPU_STORE(&svc.jobs_failed, 0);
    QPU_STORE(&svc.jobs_timed_out, 0);
    QPU_STORE(&svc.retries, 0);
    QPU_STORE(&svc.submit_deferred, 0);
    QPU_STORE(&svc.stray_events, 0);
    QPU_STORE(&svc.tasks_failed, 0);
    QPU_STORE(&svc.tasks_rejected, 0);
    QPU_STORE(&svc.tasks_unavailable, 0);
    QPU_STORE(&svc.c_pending, 0);
    QPU_STORE(&svc.c_in_flight, 0);
    QPU_STORE(&svc.max_pending, 0);
    QPU_STORE(&svc.max_in_flight, 0);
    QPU_STORE(&svc.jobs_timed, 0);
    QPU_STORE(&svc.latency_sum_ns, 0);
    QPU_STORE(&svc.jobs_coalesced, 0);
    QPU_STORE(&svc.tasks_coalesced, 0);
    QPU_STORE(&svc.max_job_tasks, 0);
    svc.latency_ns      = 0;
    svc.device          = parsec_qpu_module;
    svc.poll_max_ns     = (int64_t)(parsec_qpu_params.poll_max_ms > 0 ? parsec_qpu_params.poll_max_ms : 1000) * 1000000LL;
    svc.submit_interval_ns     = (int64_t)(parsec_qpu_params.submit_interval_ms > 0 ? parsec_qpu_params.submit_interval_ms : 0) * 1000000LL;
    svc.job_timeout_ns         = (int64_t)(parsec_qpu_params.job_timeout_ms > 0 ? parsec_qpu_params.job_timeout_ms : 0) * 1000000LL;
    svc.unavailable_timeout_ns = (int64_t)(parsec_qpu_params.unavailable_timeout_ms > 0 ? parsec_qpu_params.unavailable_timeout_ms : 0) * 1000000LL;
    /* At least a millisecond: with none, an unavailable backend would be asked without pause. */
    svc.backoff_min_ns         = (int64_t)(parsec_qpu_params.retry_backoff_ms > 0 ? parsec_qpu_params.retry_backoff_ms : 1) * 1000000LL;
    svc.backoff_ns             = svc.backoff_min_ns;
    /* The attempt number is the low byte of a job's tag. */
    svc.max_attempts    = 1 + (uint32_t)(parsec_qpu_params.max_retries > 0
                                         ? (parsec_qpu_params.max_retries < 200 ? parsec_qpu_params.max_retries : 200) : 0);

    /* The backend is open since the device was attached, unless the service
     * was stopped in between. */
    if( PARSEC_SUCCESS != qpu_backend_open() ) goto failed;
    svc.in_flight = (qpu_job_t**)calloc(svc.max_outstanding, sizeof(qpu_job_t*));

    svc.t_start = parsec_qpu_now_ns();
    if( 0 != pthread_create(&svc.thread, NULL, qpu_progress_main, context) ) {
        parsec_warning("QPU: could not create the progress thread");
        (void)svc.backend->ops->close(svc.backend);
        svc.backend = NULL;
        free(svc.in_flight); svc.in_flight = NULL;
        goto failed;
    }
    parsec_debug_verbose(4, parsec_qpu_output_stream,
                         "QPU: service up, backend %s, at most %u jobs outstanding of at most %u task(s) each",
                         svc.backend->ops->name, svc.max_outstanding, svc.max_batch);
    /* A job can be pushed and the backend woken from here on; the progress
     * thread finds them whenever it gets to run. */
    (void)parsec_atomic_cas_int32(&svc.state, QPU_SVC_STARTING, QPU_SVC_UP);
    return PARSEC_SUCCESS;

  failed:
    (void)parsec_atomic_cas_int32(&svc.state, QPU_SVC_STARTING, QPU_SVC_FAILED);
    return PARSEC_ERROR;
}

int parsec_qpu_service_fini(void)
{
    int abandoned = 0;
    int32_t state = QPU_LOAD(&svc.state);

    if( QPU_SVC_UP == state ) {
        qpu_task_t *item;

        (void)parsec_atomic_fetch_add_int32(&svc.stop, 1);
        (void)svc.backend->ops->wakeup(svc.backend);
        pthread_join(svc.thread, NULL);

        /* The progress thread is gone, so this thread now stands in for it
         * towards the backend. Whatever is left belongs to tasks of a
         * taskpool the application did not wait for. None of them is
         * completed: there may be no scheduler to complete them into. */
        for( uint32_t i = 0; i < svc.in_flight_n; i++ ) {
            (void)svc.backend->ops->cancel(svc.backend, svc.in_flight[i]->id);
            abandoned += qpu_job_drop(svc.in_flight[i]);
        }
        for( uint32_t i = 0; i < svc.pending_n; i++ ) {
            free(svc.pending[i]);
            abandoned++;
        }
        if( NULL != svc.deferred ) {
            /* Its job may be at the provider. */
            if( NULL != svc.backend->ops->abandon ) {
                (void)svc.backend->ops->abandon(svc.backend, svc.deferred->tag);
            }
            abandoned += qpu_job_drop(svc.deferred);
            svc.deferred = NULL;
        }
        while( NULL != (item = svc.submissions) ) {
            svc.submissions = item->next;
            free(item);
            abandoned++;
        }
        if( abandoned > 0 ) {
            parsec_warning("QPU: %d quantum task(s) were still outstanding when the context was finalized; "
                           "their jobs were cancelled and the tasks never completed. A taskpool was not waited for.",
                           abandoned);
        }
        (void)svc.backend->ops->close(svc.backend);
        svc.backend     = NULL;
        free(svc.in_flight); svc.in_flight = NULL; svc.in_flight_n = 0;
        free(svc.pending);   svc.pending   = NULL; svc.pending_n   = svc.pending_cap = 0;
        svc.context     = NULL;
        svc.device      = NULL;
        (void)parsec_atomic_cas_int32(&svc.state, QPU_SVC_UP, QPU_SVC_DOWN);
    } else {
        if( QPU_SVC_FAILED == state ) {
            /* Let the next context try again. */
            (void)parsec_atomic_cas_int32(&svc.state, QPU_SVC_FAILED, QPU_SVC_DOWN);
        }
        if( NULL != svc.backend ) {
            /* Opened when the device was attached, and no task ever submitted. */
            (void)svc.backend->ops->close(svc.backend);
            svc.backend = NULL;
        }
    }
    return abandoned;
}

void parsec_qpu_service_stats(parsec_qpu_stats_t *stats)
{
    memset(stats, 0, sizeof(*stats));
    stats->tasks_submitted = (uint64_t)ctr_get(&svc.tasks_submitted);
    stats->jobs_submitted  = (uint64_t)ctr_get(&svc.jobs_submitted);
    stats->tasks_completed = (uint64_t)ctr_get(&svc.tasks_completed);
    stats->empty_jobs      = (uint64_t)ctr_get(&svc.empty_jobs);
    stats->wait_calls      = (uint64_t)ctr_get(&svc.wait_calls);
    stats->jobs_failed       = (uint64_t)ctr_get(&svc.jobs_failed);
    stats->jobs_timed_out    = (uint64_t)ctr_get(&svc.jobs_timed_out);
    stats->retries           = (uint64_t)ctr_get(&svc.retries);
    stats->submit_deferred   = (uint64_t)ctr_get(&svc.submit_deferred);
    stats->stray_events      = (uint64_t)ctr_get(&svc.stray_events);
    stats->tasks_failed      = (uint64_t)ctr_get(&svc.tasks_failed);
    stats->tasks_rejected    = (uint64_t)ctr_get(&svc.tasks_rejected);
    stats->tasks_unavailable = (uint64_t)ctr_get(&svc.tasks_unavailable);
    stats->refused_hardware  = (uint64_t)ctr_get(&qpu_refused_hardware);
    stats->pending         = (uint32_t)ctr_get(&svc.c_pending);
    stats->in_flight       = (uint32_t)ctr_get(&svc.c_in_flight);
    stats->max_pending     = (uint32_t)ctr_get(&svc.max_pending);
    stats->max_in_flight   = (uint32_t)ctr_get(&svc.max_in_flight);
    stats->jobs_timed      = (uint64_t)ctr_get(&svc.jobs_timed);
    stats->latency_sum_ns  = ctr_get(&svc.latency_sum_ns);
    stats->jobs_coalesced  = (uint64_t)ctr_get(&svc.jobs_coalesced);
    stats->tasks_coalesced = (uint64_t)ctr_get(&svc.tasks_coalesced);
    stats->max_job_tasks   = (uint32_t)ctr_get(&svc.max_job_tasks);
    stats->backend_open    = (NULL != QPU_LOAD(&svc.backend));
    if( QPU_SVC_UP == QPU_LOAD(&svc.state) ) {
        clockid_t cid;
        struct timespec ts;
        stats->running = 1;
        stats->progress_wall_ns = parsec_qpu_now_ns() - svc.t_start;
        if( (0 == pthread_getcpuclockid(svc.thread, &cid)) && (0 == clock_gettime(cid, &ts)) ) {
            stats->progress_cpu_ns = (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
        }
    }
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

parsec_hook_return_t
parsec_qpu_submit(parsec_execution_stream_t *es, parsec_task_t *task,
                  parsec_data_copy_t *circuit, parsec_data_copy_t *params,
                  parsec_data_copy_t *result, int shots, uint64_t key)
{
    const parsec_device_qpu_module_t *qpu = parsec_qpu_module;
    const parsec_qpu_circuit_t *c;
    const char *why = NULL;
    qpu_task_t *item, *top;
    size_t room;

    if( !svc.enabled || (NULL == qpu) || !qpu->attached ) {
        parsec_warning("QPU: a task of class %s submitted a quantum job, and quantum tasks cannot run in this "
                       "context: %s", task->task_class->name, qpu_off_reason);
        return PARSEC_HOOK_RETURN_ERROR;
    }
    if( !qpu->owner ) {
        /* Its affinity put the task on a rank that has no backend. */
        parsec_warning("QPU: a task of class %s submitted a quantum job on rank %d, and the QPU is on rank %d "
                       "(device_qpu_owner_rank): quantum tasks have to be placed on that rank",
                       task->task_class->name, es->virtual_process->parsec_context->my_rank, qpu->owner_rank);
        return PARSEC_HOOK_RETURN_ERROR;
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

    if( QPU_SVC_UP != QPU_LOAD(&svc.state) ) {
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
    item->seq                 = (uint64_t)parsec_atomic_fetch_add_int64(&svc.next_seq, 1);
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
    item->batch               = 1;
    ctr_add(&svc.tasks_submitted, 1);

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
     * so the wake-up that follows must not block. */
    do {
        top = QPU_LOAD(&svc.submissions);
        item->next = top;
    } while( !parsec_atomic_cas_ptr(&svc.submissions, top, item) );
    (void)svc.backend->ops->wakeup(svc.backend);

    return PARSEC_HOOK_RETURN_ASYNC;
}

parsec_hook_return_t
parsec_qpu_submit_dtd(parsec_execution_stream_t *es, parsec_task_t *task,
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
    return parsec_qpu_submit(es, task, copies[0], copies[1], copies[2], shots, key);
}
