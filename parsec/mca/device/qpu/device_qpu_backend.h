/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_QPU_BACKEND_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_QPU_BACKEND_H_HAS_BEEN_INCLUDED

/**
 * @file
 * What a QPU backend implements. This is where scheduling ends and a
 * provider begins: nothing above this interface knows how a job is run.
 *
 * Threading. A backend is opened by the thread that starts the QPU service
 * and from then on every function is called by the progress thread alone, so
 * a backend needs no locking of its own. The one exception is wakeup(),
 * which any thread may call at any time between open() and close(). Once the
 * progress thread is joined, the thread that stops the service stands in for
 * it: it may call cancel() and abandon(), and it calls close().
 */

#include "parsec/mca/device/qpu/device_qpu.h"

BEGIN_C_DECLS

typedef uint64_t parsec_qpu_job_id_t;

typedef struct parsec_qpu_caps_s {
    uint32_t nb_qubits;             /**< widest register a job may use */
    uint32_t max_shots;             /**< most shots one circuit of a job may ask for, 0 for no limit */
    uint32_t max_jobs;              /**< jobs the backend accepts at once, 0 for no limit */
    uint32_t max_circuits_per_job;  /**< 1 when a job carries a single circuit, and 0 is taken for
                                     *   1: the runtime puts the circuits of several tasks in one
                                     *   job only up to this, and a backend that does not say how
                                     *   many it takes is given one */
    uint32_t is_physical;           /**< a quantum device, as opposed to the mock */
} parsec_qpu_caps_t;

/** One circuit instance of a job. The circuit and the parameter values stay
 *  valid until the job has reached a final state and its results were taken,
 *  or it was cancelled. The request itself is the caller's: a backend that
 *  needs it after submit() has returned copies it. */
typedef struct parsec_qpu_request_s {
    const parsec_qpu_circuit_t *circuit;
    const double               *params;       /**< circuit->nb_params values, or NULL */
    uint32_t                    shots;
    uint32_t                    max_entries;  /**< room in the result buffer */
    uint64_t                    key;          /**< names the evaluation (PARSEC_QPU_KEY_AUTO never gets here) */
    uint32_t                    attempt;      /**< 1 for the first job of a task, 2 for the retry of a failed one, ... */
} parsec_qpu_request_t;

typedef enum parsec_qpu_job_state_e {
    PARSEC_QPU_JOB_DONE      = 0,
    PARSEC_QPU_JOB_FAILED    = 1,
    PARSEC_QPU_JOB_CANCELLED = 2
} parsec_qpu_job_state_t;

/** A job that reached a final state. */
typedef struct parsec_qpu_event_s {
    parsec_qpu_job_id_t id;
    int                 state;   /**< parsec_qpu_job_state_t */
} parsec_qpu_event_t;

typedef struct parsec_qpu_backend_s parsec_qpu_backend_t;

typedef struct parsec_qpu_backend_ops_s {
    const char *name;

    /** Not zero when the backend can drive a quantum device. Such a backend
     *  is not even opened unless device_qpu_allow_hardware is set. */
    int physical;

    /** Create the backend of the device_index-th QPU. */
    int (*open)(parsec_qpu_backend_t **backend, int device_index);

    int (*capabilities)(parsec_qpu_backend_t *backend, parsec_qpu_caps_t *caps);

    /** Start one job carrying nb >= 1 circuit instances, never more than
     *  the backend said it takes (max_circuits_per_job). They all ask for
     *  the same number of shots. Must not block for the life of the job.
     *
     *  The job is one thing: it ends well for all its circuits or for none,
     *  and results() then gives an answer for each or fails.
     *
     *  tag identifies the job to the caller and to the provider, and is the
     *  same only when the caller asks again for the same job. The call is
     *  idempotent on it: if the provider already has a job with this tag,
     *  that job is the answer and no second one is created.
     *
     *  Returns PARSEC_SUCCESS when the provider has the job;
     *  PARSEC_ERR_OUT_OF_RESOURCE when it cannot be said that it has (the
     *  provider is unreachable, refuses for now, or did not answer): the
     *  caller will ask again later with the same tag, or call abandon();
     *  any other error when the provider refused this job for good.
     *
     *  After PARSEC_ERR_OUT_OF_RESOURCE the provider may have the job all
     *  the same, and the backend may read the buffers of the requests until
     *  the job is settled by a later submit() or by abandon(). The end of
     *  such a job is not reported before a submit() has returned its
     *  identifier. */
    int (*submit)(parsec_qpu_backend_t *backend, const parsec_qpu_request_t *requests, int nb,
                  uint64_t tag, parsec_qpu_job_id_t *id);

    /** Sleep until a job may have reached a final state, wakeup() was called
     *  since the last wait(), or timeout_ns has passed, whichever comes
     *  first; a negative timeout means no limit. Reports up to max_events
     *  jobs that reached a final state and returns their number, 0 when it
     *  woke for another reason, or a negative PaRSEC error. A wakeup() is
     *  never lost: one that arrives when nobody waits makes the next wait()
     *  return at once.
     *
     *  A job reported as failed or cancelled is forgotten by the backend at
     *  that moment. A job reported as done stays until results() or
     *  cancel(). */
    int (*wait)(parsec_qpu_backend_t *backend, int64_t timeout_ns,
                parsec_qpu_event_t *events, int max_events);

    /** Make a concurrent or the next wait() return. Callable from any thread,
     *  and called from task bodies right after they gave their task away: it
     *  must not block, whatever the thread inside wait() is doing. */
    int (*wakeup)(parsec_qpu_backend_t *backend);

    /** Fill one result per circuit instance of a job that wait() reported
     *  as done, in the order of the requests, and forget the job, also when
     *  this fails. nb is the number of circuit instances the job was
     *  submitted with. The backend sets status, flags, nb_qubits, the shot
     *  counts, the entries, t_start and t_end; the caller owns the rest of
     *  the header. When this fails, whatever it wrote before is ignored. */
    int (*results)(parsec_qpu_backend_t *backend, parsec_qpu_job_id_t id,
                   parsec_qpu_result_t **results, int nb);

    /** Give up on a job, in any state, and forget it. When this returns the
     *  backend no longer reads the buffers of the job's requests, and the
     *  end of the job will not be reported. */
    int (*cancel)(parsec_qpu_backend_t *backend, parsec_qpu_job_id_t id);

    /** Give up on a submission that was answered PARSEC_ERR_OUT_OF_RESOURCE
     *  and will not be asked again. If the provider has a job under this
     *  tag, the job is cancelled and forgotten; if it has none, nothing
     *  happens. When this returns the backend no longer reads the buffers of
     *  that submission's requests. A backend that never answers
     *  PARSEC_ERR_OUT_OF_RESOURCE may leave this NULL. */
    int (*abandon)(parsec_qpu_backend_t *backend, uint64_t tag);

    int (*close)(parsec_qpu_backend_t *backend);
} parsec_qpu_backend_ops_t;

/** Every backend starts with this. */
struct parsec_qpu_backend_s {
    const parsec_qpu_backend_ops_t *ops;
};

/** Backends by name. The mock is always there. */
PARSEC_DECLSPEC const parsec_qpu_backend_ops_t *parsec_qpu_backend_find(const char *name);

/** Add a backend. To be called before parsec_init(), from the thread that
 *  will call it. Returns PARSEC_ERR_EXISTS when the name is taken and
 *  PARSEC_ERR_OUT_OF_RESOURCE when the table is full. */
PARSEC_DECLSPEC int parsec_qpu_backend_register(const parsec_qpu_backend_ops_t *ops);

extern const parsec_qpu_backend_ops_t parsec_qpu_backend_mock_ops;

END_C_DECLS

#endif  /* PARSEC_DEVICE_QPU_BACKEND_H_HAS_BEEN_INCLUDED */
