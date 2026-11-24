/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/constants.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"
#include "parsec/mca/device/device.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <stdlib.h>

/**
 * The QPU component.
 *
 * It registers the parameters of the QPU layer and, when device_qpu_enabled
 * is set, offers the device layer one module of type PARSEC_DEV_QPU
 * (device_qpu_module.c) and lets task bodies use the QPU service
 * (device_qpu_progress.c) in the context being initialized.
 *
 * The life of the service does not follow open and close here. Its backend
 * is opened when the module is attached, on the rank that owns the QPU; its
 * progress thread starts with the first submission; and both are stopped
 * from a parsec_context_at_fini() callback, which runs before the workers
 * are joined and long before the device layer closes its components. What
 * close has left to do is take the module out of the runtime.
 */

static int device_qpu_component_open(void);
static int device_qpu_component_close(void);
static int device_qpu_component_query(mca_base_module_2_0_0_t **module, int *priority);
static int device_qpu_component_register(void);

parsec_qpu_params_t parsec_qpu_params = {
    .enabled                = 0,
    .backend                = NULL,
    .allow_hardware         = 0,
    .owner_rank             = 0,
    .max_outstanding        = 8,
    .max_batch              = 1,
    .poll_max_ms            = 1000,
    .verbose                = -1,
    .submit_interval_ms     = 0,
    .max_retries            = 2,
    .job_timeout_ms         = 0,
    .unavailable_timeout_ms = 60000,
    .retry_backoff_ms       = 100,
    .mock_seed              = 1,
    .mock_delay_seed        = 0,
    .mock_slots             = 4,
    .mock_delay             = NULL,
    .mock_queue_delay       = NULL,
    .mock_job_delay         = NULL,
    .mock_max_circuits      = 64,
    .mock_faults            = NULL,
    .mock_threads           = 2,
    .mock_max_qubits        = 24
};

int parsec_qpu_output_stream = -1;

parsec_device_base_component_t parsec_device_qpu_component = {
    {
        PARSEC_DEVICE_BASE_VERSION_2_0_0,

        /* Component name and version */
        "qpu",
        "",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        device_qpu_component_open,
        device_qpu_component_close,
        device_qpu_component_query,
        device_qpu_component_register,
        "", /*< no reserve */
    },
    {
        /* The component has no metadata */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    },
    NULL
};

mca_base_component_t *device_qpu_static_component(void)
{
    return (mca_base_component_t*)&parsec_device_qpu_component;
}

void parsec_qpu_register_params(void)
{
    (void)parsec_mca_param_reg_int_name("device_qpu", "enabled",
                                        "Number of QPUs to use in the next PaRSEC context (0 disables QPU support)",
                                        false, false, 0, &parsec_qpu_params.enabled);
    (void)parsec_mca_param_reg_string_name("device_qpu", "backend",
                                           "Backend that runs the quantum jobs. The only one so far is the mock, "
                                           "which runs in the process, samples the circuits exactly and reaches "
                                           "no quantum device",
                                           false, false, "mock", &parsec_qpu_params.backend);
    (void)parsec_mca_param_reg_int_name("device_qpu", "allow_hardware",
                                        "Allow a backend that drives a physical quantum device. Without it such "
                                        "a backend refuses to open, so nothing reaches hardware by accident",
                                        false, false, 0, &parsec_qpu_params.allow_hardware);
    (void)parsec_mca_param_reg_int_name("device_qpu", "owner_rank",
                                        "Rank that owns the QPU: it opens the backend and runs every job, and "
                                        "quantum tasks have to be placed there (-1: the last rank)",
                                        false, false, 0, &parsec_qpu_params.owner_rank);
    (void)parsec_mca_param_reg_int_name("device_qpu", "max_outstanding",
                                        "Most jobs kept at the backend at the same time; further ones wait "
                                        "in the runtime, by task priority and then by arrival",
                                        false, false, 8, &parsec_qpu_params.max_outstanding);
    (void)parsec_mca_param_reg_int_name("device_qpu", "max_batch",
                                        "Most tasks whose circuits go to the backend as one job (1: every task "
                                        "is its own job). Tasks share a job only when more of them wait than "
                                        "jobs can be submitted at that moment, and never beyond what the "
                                        "backend takes per job; no task is held back to fill one",
                                        false, false, 1, &parsec_qpu_params.max_batch);
    (void)parsec_mca_param_reg_int_name("device_qpu", "poll_max",
                                        "Longest time in milliseconds the progress thread sleeps without "
                                        "looking at its jobs again",
                                        false, false, 1000, &parsec_qpu_params.poll_max_ms);
    (void)parsec_mca_param_reg_int_name("device_qpu", "verbose",
                                        "Verbosity of the QPU layer (negative: use the verbosity of the devices)",
                                        false, false, -1, &parsec_qpu_params.verbose);
    (void)parsec_mca_param_reg_int_name("device_qpu", "submit_interval",
                                        "Shortest time in milliseconds between two submissions to the backend, "
                                        "whatever it answers (0: none), for providers that limit their rate",
                                        false, false, 0, &parsec_qpu_params.submit_interval_ms);
    (void)parsec_mca_param_reg_int_name("device_qpu", "max_retries",
                                        "Times the job of a task is submitted again after it failed or timed "
                                        "out, before the task completes with that status",
                                        false, false, 2, &parsec_qpu_params.max_retries);
    (void)parsec_mca_param_reg_int_name("device_qpu", "job_timeout",
                                        "Longest time in milliseconds a job of one task may stay at the backend, "
                                        "its time in the provider's queue included, before it is cancelled (0: no "
                                        "limit); a job that carries several tasks may stay that long for each",
                                        false, false, 0, &parsec_qpu_params.job_timeout_ms);
    (void)parsec_mca_param_reg_int_name("device_qpu", "unavailable_timeout",
                                        "Longest time in milliseconds the backend may take no job before the "
                                        "pending tasks complete as unavailable (0: no limit, they wait)",
                                        false, false, 60000, &parsec_qpu_params.unavailable_timeout_ms);
    (void)parsec_mca_param_reg_int_name("device_qpu", "retry_backoff",
                                        "Time in milliseconds before a submission the backend could not take "
                                        "is asked again; it doubles each time, up to 30 seconds",
                                        false, false, 100, &parsec_qpu_params.retry_backoff_ms);

    /* The mock backend */
    (void)parsec_mca_param_reg_int_name("device_qpu", "mock_seed",
                                        "Seed of everything the mock backend draws",
                                        false, false, 1, &parsec_qpu_params.mock_seed);
    (void)parsec_mca_param_reg_int_name("device_qpu", "mock_delay_seed",
                                        "Seed of the times the mock backend draws, apart from its samples, so "
                                        "that a run can be repeated with other times and the same answers "
                                        "(0: device_qpu_mock_seed)",
                                        false, false, 0, &parsec_qpu_params.mock_delay_seed);
    (void)parsec_mca_param_reg_int_name("device_qpu", "mock_slots",
                                        "Jobs the mock backend runs at the same time; further ones queue "
                                        "inside it, as they would at a provider",
                                        false, false, 4, &parsec_qpu_params.mock_slots);
    (void)parsec_mca_param_reg_string_name("device_qpu", "mock_delay",
                                           "Time one circuit of a job runs on the mock backend: const:<us>, "
                                           "uniform:<min us>:<max us>, exp:<mean us>, "
                                           "lognormal:<median us>:<sigma>, or replay:<file of delays in us, one "
                                           "per line>; drawn per circuit from the seed, the key of the "
                                           "evaluation and the attempt. A job runs for the sum over its "
                                           "circuits, plus device_qpu_mock_job_delay",
                                           false, false, "const:1000", &parsec_qpu_params.mock_delay);
    (void)parsec_mca_param_reg_string_name("device_qpu", "mock_job_delay",
                                           "Time every job runs on the mock backend whatever it carries, on top "
                                           "of the time of its circuits: what a provider spends per job. It "
                                           "takes what device_qpu_mock_delay takes, and is drawn per job",
                                           false, false, "const:0", &parsec_qpu_params.mock_job_delay);
    (void)parsec_mca_param_reg_int_name("device_qpu", "mock_max_circuits",
                                        "Circuits the mock backend takes in one job",
                                        false, false, 64, &parsec_qpu_params.mock_max_circuits);
    (void)parsec_mca_param_reg_string_name("device_qpu", "mock_queue_delay",
                                           "Time a job waits on the mock backend, once it has a slot, before it "
                                           "runs; it models the queue of the provider and takes what "
                                           "device_qpu_mock_delay takes",
                                           false, false, "const:0", &parsec_qpu_params.mock_queue_delay);
    (void)parsec_mca_param_reg_string_name("device_qpu", "mock_faults",
                                           "Faults the mock backend injects: none, or a comma-separated list of "
                                           "<fault>:<rule>=<value>, with fault one of fail, hang, unavailable, "
                                           "lost, reject, noresult, duplicate, late, cancelled and rule one of p "
                                           "(probability), first (attempts), every (submissions), key, for "
                                           "(milliseconds since the backend was opened)",
                                           false, false, "none", &parsec_qpu_params.mock_faults);
    (void)parsec_mca_param_reg_int_name("device_qpu", "mock_threads",
                                        "Helper threads of the mock backend, which compute the samples of "
                                        "the jobs it holds",
                                        false, false, 2, &parsec_qpu_params.mock_threads);
    (void)parsec_mca_param_reg_int_name("device_qpu", "mock_max_qubits",
                                        "Widest circuit the mock backend takes, at most 30; sampling n qubits "
                                        "takes 2^(n+4) bytes per helper thread",
                                        false, false, 24, &parsec_qpu_params.mock_max_qubits);
}

static int device_qpu_component_register(void)
{
    parsec_qpu_register_params();
    /* Not requested: leave the component out, as the other devices do. */
    return (0 == parsec_qpu_params.enabled) ? MCA_ERROR : MCA_SUCCESS;
}

static int device_qpu_component_open(void)
{
    if( 0 == parsec_qpu_params.enabled ) {
        return MCA_ERROR;
    }
    if( parsec_qpu_params.enabled > 1 ) {
        parsec_warning("device_qpu_enabled is %d, but a single QPU is supported so far; using one",
                       parsec_qpu_params.enabled);
    }
    parsec_qpu_output_stream = parsec_device_output;
    if( parsec_qpu_params.verbose >= 0 ) {
        parsec_qpu_output_stream = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_qpu_output_stream, parsec_qpu_params.verbose);
    }
    if( PARSEC_SUCCESS != parsec_qpu_service_enable() ) {
        /* The backend that was asked for may not be used. A task that
         * submits a quantum job in this context is told so. */
        if( parsec_qpu_output_stream != parsec_device_output ) {
            parsec_output_close(parsec_qpu_output_stream);
        }
        parsec_qpu_output_stream = -1;
        return MCA_ERROR;
    }
    return MCA_SUCCESS;
}

/* One module, on every rank: see device_qpu_module.c for what the ranks that
 * do not own the QPU do with theirs. */
static int device_qpu_component_query(mca_base_module_t **module, int *priority)
{
    void *modules;

    *module   = NULL;
    *priority = 0;
    parsec_device_qpu_component.modules = parsec_qpu_module_new(&parsec_device_qpu_component);
    if( NULL == parsec_device_qpu_component.modules ) {
        return MCA_ERROR;
    }
    /* As the other device components do: the module type should be
     * const mca_base_module_t **. */
    modules   = parsec_device_qpu_component.modules;
    *module   = (mca_base_module_t*)modules;
    *priority = 10;
    return MCA_SUCCESS;
}

/* Called when the device layer is finalized, after the workers are gone, or
 * right after the query when it gave no module. The service was stopped
 * before (see the top of this file). */
static int device_qpu_component_close(void)
{
    if( NULL != parsec_device_qpu_component.modules ) {
        parsec_qpu_module_free(parsec_device_qpu_component.modules);
        parsec_device_qpu_component.modules = NULL;
    }
    return MCA_SUCCESS;
}
