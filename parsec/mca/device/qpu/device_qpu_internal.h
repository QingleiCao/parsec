/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_QPU_INTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_QPU_INTERNAL_H_HAS_BEEN_INCLUDED

/**
 * @file
 * The QPU layer, as seen from inside the component and from its tests.
 *
 * It has two halves. The device module is what the runtime sees: a device
 * of type PARSEC_DEV_QPU that is selected for the incarnations of that type,
 * whose load the scheduler keeps and whose executed tasks are counted. The
 * service is what runs the jobs: a progress thread that takes the jobs task
 * bodies submit, keeps a bounded number of them at the backend, and
 * completes each task when its job is over by handing it back to a worker.
 *
 * A run has one QPU or several (device_qpu_devices). Each has a module and a
 * service of its own on the rank that owns it, and the QPUs of one rank share
 * the tasks that wait: a task is bound to a QPU when that QPU has a place
 * for its job, not before.
 */

#include "parsec/mca/device/device.h"
#include "parsec/mca/device/qpu/device_qpu.h"
#include "parsec/mca/device/qpu/device_qpu_backend.h"

#include <time.h>

BEGIN_C_DECLS

/** Most QPUs a run may have. A process has at most 16 devices of all kinds
 *  (the device mask of a taskpool is 16 bits), so no rank can own more. */
#define PARSEC_QPU_MAX_QPUS  16

/** Values of the MCA parameters, filled by parsec_qpu_register_params(). */
typedef struct parsec_qpu_params_s {
    int   enabled;                /**< device_qpu_enabled */
    char *devices;                /**< device_qpu_devices */
    char *backend;                /**< device_qpu_backend */
    int   allow_hardware;         /**< device_qpu_allow_hardware */
    int   owner_rank;             /**< device_qpu_owner_rank */
    int   max_outstanding;        /**< device_qpu_max_outstanding */
    int   max_batch;              /**< device_qpu_max_batch */
    int   poll_max_ms;            /**< device_qpu_poll_max */
    int   verbose;                /**< device_qpu_verbose */
    int   submit_interval_ms;     /**< device_qpu_submit_interval */
    int   max_retries;            /**< device_qpu_max_retries */
    int   job_timeout_ms;         /**< device_qpu_job_timeout */
    int   unavailable_timeout_ms; /**< device_qpu_unavailable_timeout */
    int   retry_backoff_ms;       /**< device_qpu_retry_backoff */
    int   mock_seed;              /**< device_qpu_mock_seed */
    int   mock_delay_seed;        /**< device_qpu_mock_delay_seed */
    int   mock_slots;             /**< device_qpu_mock_slots */
    char *mock_delay;             /**< device_qpu_mock_delay */
    char *mock_queue_delay;       /**< device_qpu_mock_queue_delay */
    char *mock_job_delay;         /**< device_qpu_mock_job_delay */
    int   mock_max_circuits;      /**< device_qpu_mock_max_circuits */
    int   mock_submit_delay;      /**< device_qpu_mock_submit_delay */
    char *mock_faults;            /**< device_qpu_mock_faults */
    int   mock_threads;           /**< device_qpu_mock_threads */
    int   mock_max_qubits;        /**< device_qpu_mock_max_qubits */
} parsec_qpu_params_t;

PARSEC_DECLSPEC extern parsec_qpu_params_t parsec_qpu_params;
extern int parsec_qpu_output_stream;

/** Register the parameters above. Safe to call more than once. */
void parsec_qpu_register_params(void);

/* ------------------------------------------------------------------------
 * The list of QPUs (device_qpu_list.c): see there for its syntax.
 * --------------------------------------------------------------------- */

/** Read the list: `spec` when it names a QPU, and `enabled` QPUs of the
 *  backend `backend` otherwise. On a list that cannot be read nothing is
 *  kept, and *why says what is wrong with it. Main thread only. */
PARSEC_DECLSPEC int parsec_qpu_list_set(const char *spec, int enabled, const char *backend, const char **why);
PARSEC_DECLSPEC void parsec_qpu_list_free(void);
/** How many QPUs the list holds: 0 when none was read. */
PARSEC_DECLSPEC int parsec_qpu_list_size(void);
PARSEC_DECLSPEC const char *parsec_qpu_list_backend(int qpu);
/** The rank that owns a QPU, in a run of nb_ranks ranks. It is not checked:
 *  a rank that does not exist means that nobody owns the QPU. */
PARSEC_DECLSPEC int parsec_qpu_list_rank(int qpu, int nb_ranks);
/** What the entry of a QPU says for a key: NULL when it does not have it,
 *  and for a QPU that is not in the list; or `value` then. */
PARSEC_DECLSPEC const char *parsec_qpu_list_option(int qpu, const char *key);
PARSEC_DECLSPEC int parsec_qpu_list_option_int(int qpu, const char *key, int value);
PARSEC_DECLSPEC const char *parsec_qpu_list_option_string(int qpu, const char *key, const char *value);

/* ------------------------------------------------------------------------
 * The service (device_qpu_progress.c)
 * --------------------------------------------------------------------- */

struct parsec_device_qpu_module_s;

/** Let task bodies use the service in the context being initialized, and
 *  arrange for it to stop when that context is finalized. Main thread only.
 *  Fails when a backend the list names does not exist, or drives a quantum
 *  device while device_qpu_allow_hardware is not set: no QPU is used then. */
int parsec_qpu_service_enable(void);

/** Say why quantum tasks cannot run in the context being initialized. */
void parsec_qpu_service_refuse(const char *why);

/** Open the backend of a QPU of this rank and say what it is. Called when
 *  the device of that QPU is attached, so that a backend that cannot be used
 *  gives no device; the progress threads start with the first submission.
 *  max_batch receives the most tasks one job will carry: what was asked for,
 *  and no more circuits than the backend takes in a job. Main thread only. */
int parsec_qpu_service_open(struct parsec_device_qpu_module_s *module, parsec_qpu_caps_t *caps,
                            uint32_t *max_outstanding, uint32_t *max_batch);
/** Take that back: close the backend of the QPU that was opened last, whose
 *  device did not make it into the context. Main thread only. */
void parsec_qpu_service_forget(struct parsec_device_qpu_module_s *module);

/* ------------------------------------------------------------------------
 * The device modules (device_qpu_module.c)
 * --------------------------------------------------------------------- */

/** A QPU as the runtime sees it, on the rank that owns it. A rank that owns
 *  no QPU has one all the same, a placeholder with no backend that takes no
 *  task, so that no rank drops the QPU incarnations of its task classes. */
typedef struct parsec_device_qpu_module_s {
    parsec_device_module_t super;
    int                    number;           /**< of the QPU, in the list */
    int                    owner_rank;       /**< the rank that runs the jobs */
    int                    nb_ranks;         /**< of the run; a rank past them owns nothing */
    int                    owner;            /**< this process is that rank */
    int                    attached;         /**< the runtime has it among its devices */
    int                    service;          /**< which of the QPUs of this rank it is; -1 when not the owner */
    parsec_qpu_caps_t      caps;             /**< of the backend; zero when not the owner */
    uint32_t               max_outstanding;  /**< 0 when not the owner */
    uint32_t               max_batch;        /**< most tasks in one job; 0 when not the owner */
    int64_t                latency_ns;       /**< how long a job takes, as the progress thread measures
                                              *   it; 0 before any job has ended well */
    char                   name_buffer[64];
} parsec_device_qpu_module_t;

/** Create the modules, one per QPU of the list, for the component's query:
 *  a NULL-terminated array that the component owns. Which of them this rank
 *  attaches is decided when they are attached. */
parsec_device_module_t **parsec_qpu_module_new(const parsec_device_base_component_t *component);
/** Take the modules out of the runtime and free them, for the component's close. */
void parsec_qpu_module_free(parsec_device_module_t **modules);
/** The rank that owns the first QPU when this rank owns none and has a
 *  placeholder for them; -2 when this rank has no placeholder. */
int parsec_qpu_module_placeholder(void);

/** What the scheduler is given as the time of a quantum task, for the load
 *  of the device: one second, whatever jobs take. The load is then the
 *  number of tasks the device holds a job for, in seconds. The time jobs do take is
 *  measured (latency_ns above) and is not fed back to the scheduler: the
 *  device layer reads the estimate of a device without synchronisation, so
 *  it is written once, when the device is attached. */
#define PARSEC_QPU_TASK_TIME_NS  1000000000LL

/** Stop the service if it runs: cancel what is left, join the progress
 *  threads, close the backends. Returns the number of tasks abandoned, which
 *  is zero unless the application did not wait for a taskpool. */
PARSEC_DECLSPEC int parsec_qpu_service_fini(void);

/** Counters of the service: of all the QPUs of this rank together
 *  (parsec_qpu_service_stats), or of one of them
 *  (parsec_qpu_service_stats_of). The first group only grows.
 *
 *  Tasks wait for the rank, not for a QPU, and some complete without a job
 *  of any QPU. So tasks_submitted, tasks_completed, empty_jobs, the three
 *  counts of tasks that failed, pending and max_pending are counted for the
 *  rank only, and are zero for one QPU; how many tasks were counted for a
 *  QPU is tasks_executed. For the rank, max_in_flight
 *  and max_job_tasks are the largest of its QPUs, and the times of the
 *  progress threads are added up. */
typedef struct parsec_qpu_stats_s {
    uint64_t tasks_submitted;   /**< bodies that returned ASYNC */
    uint64_t jobs_submitted;    /**< jobs the backend accepted */
    uint64_t tasks_completed;   /**< tasks handed back to the scheduler */
    uint64_t empty_jobs;        /**< tasks completed without a backend job */
    uint64_t wait_calls;        /**< times the progress thread went to wait */

    uint64_t jobs_failed;       /**< jobs that ended in failure, or were refused for good */
    uint64_t jobs_timed_out;    /**< jobs cancelled because they took too long */
    uint64_t retries;           /**< jobs submitted again after a failure or a time-out */
    uint64_t submit_deferred;   /**< submissions the backend could not take at that moment */
    uint64_t stray_events;      /**< final states reported for a job that was not in flight */
    uint64_t tasks_failed;      /**< tasks completed with a status other than OK, in total */
    uint64_t tasks_rejected;    /**< ... of which: the backend cannot run the job */
    uint64_t tasks_unavailable; /**< ... of which: the backend was out of reach for too long */

    uint32_t pending;           /**< accepted, not yet at the backend */
    uint32_t in_flight;         /**< at the backend */
    uint32_t max_pending;       /**< high-water marks */
    uint32_t max_in_flight;

    int64_t  progress_cpu_ns;   /**< CPU time the progress thread used */
    int64_t  progress_wall_ns;  /**< time since it started */
    int      running;           /**< the progress thread exists */

    uint64_t refused_hardware;  /**< times a backend was refused because it drives, or
                                 *   says it drives, a quantum device; since the process
                                 *   started */

    uint64_t jobs_timed;        /**< jobs that ended well, and whose time from submission
                                 *   to result is in the sum below */
    int64_t  latency_sum_ns;
    int      backend_open;      /**< the backend is open, with or without a progress thread */

    /* Of jobs_submitted. A task that is retried is counted once more. */
    uint64_t jobs_coalesced;    /**< jobs that carried more than one task */
    uint64_t tasks_coalesced;   /**< tasks those jobs carried: the tasks the backend took are
                                 *   jobs_submitted - jobs_coalesced + tasks_coalesced */
    uint32_t max_job_tasks;     /**< most tasks in one job */

    uint64_t tasks_moved;       /**< tasks given back to wait for another QPU because this one
                                 *   took no job for too long */
    uint64_t tasks_executed;    /**< tasks the service gave back that its QPU's device
                                 *   counted as executed. A task that ended without a job
                                 *   (empty, rejected, unavailable) counts for the QPU the
                                 *   device layer had picked for it, which never ran it.
                                 *   The device itself (parsec_qpu_device_info()) also
                                 *   counts bodies that did not return ASYNC and earlier
                                 *   runs of the service, until the device layer prints
                                 *   its statistics and starts again */
    uint32_t nb_qpus;           /**< QPUs of this rank; 1 for one QPU */
    uint32_t down;              /**< of them, those that took no job for too long and are
                                 *   left alone for a while */
} parsec_qpu_stats_t;

/** Read the counters of this rank. Always succeeds; everything is zero
 *  before the first submission of a context. */
PARSEC_DECLSPEC void parsec_qpu_service_stats(parsec_qpu_stats_t *stats);
/** The same for one QPU, named by its number in the list.
 *  PARSEC_ERR_NOT_FOUND when this rank does not own it. */
PARSEC_DECLSPEC int parsec_qpu_service_stats_of(int qpu, parsec_qpu_stats_t *stats);

/* ------------------------------------------------------------------------
 * The exact sampler (device_qpu_sampler.c). It needs neither the runtime
 * nor a backend, so a program that does not use PaRSEC can call it to get
 * the counts the mock backend gives for the same seed and key.
 * --------------------------------------------------------------------- */

/** Widest register the sampler takes. Its memory is 2^(n+4) bytes. */
#define PARSEC_QPU_SAMPLER_MAX_QUBITS   30

/** Returned, as a positive value, when the run was given up because the
 *  flag passed as `cancel` was raised. */
#define PARSEC_QPU_SAMPLER_CANCELLED    1

/** Scratch memory, kept from one run to the next. One per thread. */
typedef struct parsec_qpu_sampler_s parsec_qpu_sampler_t;

PARSEC_DECLSPEC parsec_qpu_sampler_t *parsec_qpu_sampler_new(void);
PARSEC_DECLSPEC void parsec_qpu_sampler_free(parsec_qpu_sampler_t *sampler);

/**
 * Compute the state a circuit prepares, measurements left aside: 2^n
 * amplitudes, each a (real, imaginary) pair of doubles, amplitude x being
 * that of the basis state whose bit i is the value of qubit i. *amp points
 * into the sampler's memory and is valid until its next use.
 *
 * cancel, when not NULL, is read between operations; the run is given up
 * when it is not zero.
 */
PARSEC_DECLSPEC int
parsec_qpu_sampler_state(parsec_qpu_sampler_t *sampler, const parsec_qpu_circuit_t *circuit,
                         const double *params, const int32_t *cancel, const double **amp);

/**
 * Measure a circuit `shots` times. The draws depend on (seed, key) and on
 * nothing else that is not in the circuit and its parameter values.
 *
 * Sets, in *result: status (OK), nb_qubits, nb_entries, shots_returned, the
 * entries, and the TRUNCATED flag when max_entries was too few. The rest of
 * the header is left as it was.
 */
PARSEC_DECLSPEC int
parsec_qpu_sampler_run(parsec_qpu_sampler_t *sampler, const parsec_qpu_circuit_t *circuit,
                       const double *params, uint32_t shots, uint64_t seed, uint64_t key,
                       parsec_qpu_result_t *result, uint32_t max_entries, const int32_t *cancel);

/** The same with a sampler of its own, for a caller that does it once. */
PARSEC_DECLSPEC int
parsec_qpu_sampler_sample(const parsec_qpu_circuit_t *circuit, const double *params,
                          uint32_t shots, uint64_t seed, uint64_t key,
                          parsec_qpu_result_t *result, uint32_t max_entries);

/* ------------------------------------------------------------------------
 * Mock backend: what tests need from it.
 * --------------------------------------------------------------------- */

/** A distribution of delays: const:<us>, uniform:<min us>:<max us>,
 *  exp:<mean us>, lognormal:<median us>:<sigma>, or replay:<file>, a text
 *  file of delays in microseconds, one per line. */
typedef struct parsec_qpu_delay_s {
    int      kind;
    double   a, b;          /* nanoseconds, but for the sigma of a lognormal */
    int64_t *samples;       /* replay */
    uint32_t nb_samples;
} parsec_qpu_delay_t;

PARSEC_DECLSPEC int  parsec_qpu_delay_parse(parsec_qpu_delay_t *delay, const char *spec);
PARSEC_DECLSPEC void parsec_qpu_delay_fini(parsec_qpu_delay_t *delay);
/** One delay in nanoseconds, from 64 random bits. The same bits give the
 *  same delay. */
PARSEC_DECLSPEC int64_t parsec_qpu_delay_draw(const parsec_qpu_delay_t *delay, uint64_t bits);

/** The three times the mock draws: for each circuit of a job
 *  (device_qpu_mock_delay), and for the job as a whole from its first
 *  circuit: the time it waits in its slot (device_qpu_mock_queue_delay) and
 *  the part it pays whatever it carries (device_qpu_mock_job_delay). */
typedef enum parsec_qpu_mock_time_e {
    PARSEC_QPU_MOCK_TIME_CIRCUIT = 1,
    PARSEC_QPU_MOCK_TIME_QUEUE   = 2,
    PARSEC_QPU_MOCK_TIME_JOB     = 3
} parsec_qpu_mock_time_t;

/** One of those times for the evaluation `key` at attempt `attempt`, drawn
 *  from `delay` with the seed of the mock's times (device_qpu_mock_delay_seed,
 *  or device_qpu_mock_seed when that is 0), as the mock draws it. For models
 *  that have to see the times a run saw. */
PARSEC_DECLSPEC int64_t parsec_qpu_mock_time(const parsec_qpu_delay_t *delay, uint64_t delay_seed,
                                             uint64_t key, uint32_t attempt, parsec_qpu_mock_time_t what);

/** Time one circuit of a job runs, in nanoseconds, replacing
 *  device_qpu_mock_delay; a job with several circuits is asked once for
 *  each. For tests that need a completion order of their own. Set it before
 *  the first submission; NULL restores the parameter. */
typedef int64_t (*parsec_qpu_mock_delay_fn_t)(uint64_t tag, const parsec_qpu_request_t *request, void *arg);
PARSEC_DECLSPEC void parsec_qpu_mock_set_delay_fn(parsec_qpu_mock_delay_fn_t fn, void *arg);

/** What the mock can be told to do to a job instead of running it. */
typedef enum parsec_qpu_mock_fault_e {
    PARSEC_QPU_MOCK_OK          = 0,
    PARSEC_QPU_MOCK_FAIL        = 1,  /**< the job runs its time and ends in failure */
    PARSEC_QPU_MOCK_HANG        = 2,  /**< the job takes a slot and never ends */
    PARSEC_QPU_MOCK_UNAVAILABLE = 3,  /**< the submission is not taken and nothing is looked
                                       *   up: try later */
    PARSEC_QPU_MOCK_LOST        = 4,  /**< the job is taken (or was, when this is asking
                                       *   again), but the answer is "try later": asking
                                       *   again must find it by its tag */
    PARSEC_QPU_MOCK_REJECT      = 5,  /**< the submission is refused for good */
    PARSEC_QPU_MOCK_NORESULT    = 6,  /**< the job ends well, but its results cannot be had */
    PARSEC_QPU_MOCK_DUPLICATE   = 7,  /**< the job ends well, and its end is reported twice */
    PARSEC_QPU_MOCK_LATE        = 8,  /**< the job never ends, and once it was cancelled its
                                       *   end is reported all the same */
    PARSEC_QPU_MOCK_CANCELLED   = 9,  /**< the job runs its time and is reported as cancelled:
                                       *   somebody at the provider did, not the caller */
    PARSEC_QPU_MOCK_FAULT_MAX
} parsec_qpu_mock_fault_t;

/** Decide the fate of a submission, replacing device_qpu_mock_faults. It is
 *  asked at every submission, also when the same job is asked for again;
 *  then only UNAVAILABLE and LOST change anything, and any other answer lets
 *  the job that is there be found. `call` counts the submissions the mock
 *  received, from 1. For a job with several circuits it is asked for each
 *  of them in turn, with the same tag and call, until one answer is not OK:
 *  that answer is the fate of the job. Set it before the first submission;
 *  NULL restores the parameter. */
typedef parsec_qpu_mock_fault_t (*parsec_qpu_mock_fault_fn_t)(uint64_t tag, const parsec_qpu_request_t *request,
                                                              uint64_t call, void *arg);
PARSEC_DECLSPEC void parsec_qpu_mock_set_fault_fn(parsec_qpu_mock_fault_fn_t fn, void *arg);

/** Counters of the mock: of all the mocks that are open in this process
 *  together (parsec_qpu_mock_stats), or of the one of a QPU
 *  (parsec_qpu_mock_stats_of). Together, the counts are added up, the
 *  high-water marks are the largest of them, and the shortest time between
 *  two submissions is the shortest of them. */
typedef struct parsec_qpu_mock_stats_s {
    uint64_t opens;           /**< backends opened by this process since it started */
    /* The rest is about the backends that are open, if any. */
    uint64_t submit_calls;    /**< submissions received */
    uint64_t submits;         /**< jobs created */
    uint64_t circuits;        /**< circuits those jobs carried */
    uint64_t tag_hits;        /**< submissions answered with a job that was already there */
    uint64_t deferred;        /**< submissions answered "try later" */
    uint64_t rejected;        /**< submissions refused for good */
    uint64_t cancels;
    uint64_t abandoned;       /**< jobs dropped because their submission was given up on */
    uint64_t results;         /**< results given out, one per circuit */
    uint64_t failures;        /**< jobs reported as failed */
    uint64_t provider_cancels;/**< jobs reported as cancelled, which the caller had not asked for */
    uint64_t ghost_events;    /**< ends reported for a job that was gone by then */
    uint64_t computed;        /**< circuits a helper thread sampled, or started to */
    uint64_t late;            /**< jobs that ended after their modelled end, because their
                               *   circuit was still being sampled: for those, and only
                               *   those, the time the mock reports is not the time it drew */
    /* None of these may happen. */
    uint64_t duplicate_tags;  /**< submissions under the tag of a job the caller knew already,
                               *   or had taken the results of */
    uint64_t key_overlaps;    /**< circuits taken while another of the same key was held, in
                               *   another job or in the same one */
    uint64_t inputs_changed;  /**< circuits that changed, or whose parameters did, while their
                               *   job was held */
    uint64_t mixed_shots;     /**< jobs created with circuits that ask for different numbers
                               *   of shots */
    uint64_t changed_jobs;    /**< submissions under the tag of a job that is held, for other
                               *   circuits than the ones the job has: asking again is for the
                               *   same job */

    uint32_t max_running;     /**< most jobs it ran at the same time */
    uint32_t max_held;        /**< most jobs it held, running or queued */
    uint32_t max_circuits;    /**< most circuits in one job */
    uint32_t unknown_jobs;    /**< jobs it holds now that the caller was never told the
                               *   identifier of; zero once every submission is settled */
    int64_t  min_submit_gap;  /**< shortest time between two submissions received, whatever
                               *   became of them, nanoseconds; negative when there were
                               *   fewer than two */
} parsec_qpu_mock_stats_t;
PARSEC_DECLSPEC void parsec_qpu_mock_stats(parsec_qpu_mock_stats_t *stats);
/** The same for the mock that was opened for one QPU, named by its number
 *  in the list. PARSEC_ERR_NOT_FOUND when no mock is open for it here. */
PARSEC_DECLSPEC int parsec_qpu_mock_stats_of(int qpu, parsec_qpu_mock_stats_t *stats);

/* ------------------------------------------------------------------------
 * Small helpers shared by the files of the component.
 * --------------------------------------------------------------------- */

/* A read or a plain write of a word that other threads access. PaRSEC's
 * atomic layer has neither, so use the compiler's where there are some. */
#if defined(__GNUC__) || defined(__clang__)
#define PARSEC_QPU_LOAD(PTR)        __atomic_load_n((PTR), __ATOMIC_SEQ_CST)
#define PARSEC_QPU_STORE(PTR, VAL)  __atomic_store_n((PTR), (VAL), __ATOMIC_SEQ_CST)
#else
#define PARSEC_QPU_LOAD(PTR)        (*(PTR))
#define PARSEC_QPU_STORE(PTR, VAL)  (*(PTR) = (VAL))
#endif

static inline int64_t parsec_qpu_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/* splitmix64 finalizer: a cheap, well-mixed 64-bit hash. */
static inline uint64_t parsec_qpu_mix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/* A number in [0, 1) from 64 random bits. */
static inline double parsec_qpu_unit(uint64_t bits)
{
    return (double)(bits >> 11) * (1.0 / 9007199254740992.0);
}

END_C_DECLS

#endif  /* PARSEC_DEVICE_QPU_INTERNAL_H_HAS_BEEN_INCLUDED */
