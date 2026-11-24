/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/constants.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"
#include "parsec/mca/device/device.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * The QPU device module: the QPU as the runtime sees it.
 *
 * What being a device brings.
 *   - A task class says which of its incarnations is quantum
 *     (BODY [type=QPU] in a JDF, PARSEC_DEV_QPU at a DTD insertion), and
 *     parsec_select_best_device() selects this module for it. Without a QPU
 *     in the context, the QPU incarnations are dropped when a taskpool
 *     starts and another incarnation of the class runs, if it has one.
 *   - The scheduler keeps the load of the device: the estimated time of the
 *     tasks it holds, added when a task is given to it and taken back when
 *     the task completes. The estimate is a nominal second per task, so the
 *     load counts the tasks held; the time jobs really take is measured by
 *     the service and reported next to it (parsec_qpu_device_info).
 *   - The tasks it executed are counted with those of the other devices.
 *
 * What it does not do. A QPU holds no data, so the module has no memory to
 * register and no copy to manage: a quantum task reads and writes host
 * copies, and the scheduler treats its inputs as it treats those of a CPU
 * task. It has no flop rate, and takes no part in the weights the device
 * layer derives from them. And it runs nothing itself: a body of type QPU
 * calls parsec_qpu_submit(), which gives the task to the service.
 *
 * One owner rank. Every rank attaches a module, so that the QPU incarnations
 * of a task class are kept on every rank and a taskpool looks the same
 * everywhere. Only the module on the owner rank has a backend. The others
 * take no work: a quantum task that runs on such a rank was misplaced, and
 * its submission is refused with a message that says so.
 */

parsec_device_qpu_module_t *parsec_qpu_module = NULL;

static int qpu_device_param(const char *name)
{
    int index = parsec_mca_param_find("device", NULL, name), value = 0;

    if( 0 < index ) (void)parsec_mca_param_lookup_int(index, &value);
    return value;
}

/* Every taskpool may use the device, on every rank: see the top of this file
 * for why a rank that does not own the QPU does not decline here. */
static int qpu_module_taskpool_register(parsec_device_module_t *device, parsec_taskpool_t *tp)
{
    (void)device; (void)tp;
    return PARSEC_SUCCESS;
}

static int qpu_module_taskpool_unregister(parsec_device_module_t *device, parsec_taskpool_t *tp)
{
    (void)device; (void)tp;
    return PARSEC_SUCCESS;
}

static int qpu_module_attach(parsec_device_module_t *device, parsec_context_t *context)
{
    parsec_device_qpu_module_t *qpu = (parsec_device_qpu_module_t*)device;
    int owner_rank = parsec_qpu_params.owner_rank, rc;

    if( -1 == owner_rank ) owner_rank = context->nb_nodes - 1;      /* the last rank */
    if( (owner_rank < 0) || (owner_rank >= context->nb_nodes) ) {
        if( 0 == context->my_rank ) {
            parsec_warning("QPU: device_qpu_owner_rank is %d and there are %d rank(s): no rank owns the QPU, "
                           "and quantum tasks cannot run in this context",
                           parsec_qpu_params.owner_rank, context->nb_nodes);
        }
    }
    qpu->owner_rank = owner_rank;
    qpu->owner      = (context->my_rank == owner_rank);
    memset(&qpu->caps, 0, sizeof(qpu->caps));
    qpu->max_outstanding = 0;
    qpu->max_batch       = 0;
    if( qpu->owner ) {
        /* A backend that cannot be used gives no device. What was wrong has
         * been said. */
        rc = parsec_qpu_service_open(&qpu->caps, &qpu->max_outstanding, &qpu->max_batch);
        if( PARSEC_SUCCESS != rc ) return rc;
    }

    snprintf(qpu->name_buffer, sizeof(qpu->name_buffer), "qpu(%s)",
             (NULL != parsec_qpu_params.backend) ? parsec_qpu_params.backend : "?");
    device->name = qpu->name_buffer;
    device->type = PARSEC_DEV_QPU;
    /* No flop rate: the device layer then leaves the time estimate alone
     * (parsec_mca_device_registration_complete). It stays what it is set to
     * here for as long as the device lives. */
    device->gflops_fp16 = device->gflops_fp32 = device->gflops_fp64 = device->gflops_tf32 = 0;
    device->gflops_guess          = 1;
    device->time_estimate_default = PARSEC_QPU_TASK_TIME_NS;
    device->device_load           = 0;
    device->executed_tasks        = 0;
    qpu->latency_ns               = 0;

    rc = parsec_mca_device_add(context, device);
    if( rc < 0 ) {
        /* The backend that was opened above is closed with the service, when
         * the context is finalized. */
        parsec_warning("QPU: the device could not be added to the context (%d)", rc);
        return rc;
    }
    qpu->attached = 1;

    if( qpu_device_param("show_capabilities") ) {
        if( qpu->owner ) {
            parsec_inform("Dev %-2d %11s : backend %s%s\n"
                          "\tQubits             : %u\t[Shots per job: %u%s | Circuits per job: %u]\n"
                          "\tJobs outstanding   : at most %u, on rank %d\t[Tasks per job: at most %u]",
                          device->device_index, device->name, parsec_qpu_params.backend,
                          qpu->caps.is_physical ? " (a quantum device)" : " (no quantum device)",
                          qpu->caps.nb_qubits, qpu->caps.max_shots, (0 == qpu->caps.max_shots) ? " (no limit)" : "",
                          qpu->caps.max_circuits_per_job, qpu->max_outstanding, qpu->owner_rank, qpu->max_batch);
        } else {
            parsec_inform("Dev %-2d %11s : the QPU is on rank %d; this rank takes no quantum task",
                          device->device_index, device->name, qpu->owner_rank);
        }
    }
    return PARSEC_SUCCESS;
}

static int qpu_module_detach(parsec_device_module_t *device, parsec_context_t *context)
{
    parsec_device_qpu_module_t *qpu = (parsec_device_qpu_module_t*)device;
    int rc = PARSEC_SUCCESS;

    (void)context;
    if( qpu->attached ) {
        rc = parsec_mca_device_remove(device);
        qpu->attached = 0;
    }
    return rc;
}

parsec_device_module_t **parsec_qpu_module_new(const parsec_device_base_component_t *component)
{
    parsec_device_module_t **modules;
    parsec_device_qpu_module_t *qpu;

    modules = (parsec_device_module_t**)calloc(2, sizeof(parsec_device_module_t*));
    qpu     = (parsec_device_qpu_module_t*)calloc(1, sizeof(parsec_device_qpu_module_t));
    if( (NULL == modules) || (NULL == qpu) ) {
        free(modules); free(qpu);
        return NULL;
    }
    /* As the device layer does for the CPU device: plain memory, zeroed. */
    qpu->super.component           = component;
    qpu->super.attach              = qpu_module_attach;
    qpu->super.detach              = qpu_module_detach;
    qpu->super.taskpool_register   = qpu_module_taskpool_register;
    qpu->super.taskpool_unregister = qpu_module_taskpool_unregister;
    qpu->super.type                = PARSEC_DEV_QPU;
    qpu->super.name                = "qpu";
    qpu->owner_rank                = -1;

    modules[0] = &qpu->super;
    modules[1] = NULL;
    parsec_qpu_module = qpu;
    return modules;
}

/* One line under the table the device layer prints: what a QPU did has no
 * column there. The service is stopped by now, and its counters are those
 * of its last run. */
static void qpu_module_print_statistics(const parsec_device_qpu_module_t *qpu)
{
    parsec_qpu_stats_t stats;

    if( !qpu->owner ) return;
    parsec_qpu_service_stats(&stats);
    printf("QPU %s on rank %d: %"PRIu64" task(s), %"PRIu64" job(s) at the backend, %"PRIu64" task(s) without a job; "
           "%"PRIu64" job(s) failed, %"PRIu64" timed out, %"PRIu64" retried, %"PRIu64" submission(s) deferred; "
           "%"PRIu64" task(s) ended without a result\n"
           "    at most %u job(s) outstanding (limit %u) and %u task(s) waiting; "
           "%"PRIu64" job(s) carried %"PRIu64" task(s) together, at most %u in one (limit %u); ",
           qpu->name_buffer, qpu->owner_rank,
           stats.tasks_completed, stats.jobs_submitted, stats.empty_jobs,
           stats.jobs_failed, stats.jobs_timed_out, stats.retries, stats.submit_deferred,
           stats.tasks_failed, stats.max_in_flight, qpu->max_outstanding, stats.max_pending,
           stats.jobs_coalesced, stats.tasks_coalesced, stats.max_job_tasks, qpu->max_batch);
    if( stats.jobs_timed > 0 ) {
        printf("a job took %.3f ms on average from its submission to its result\n",
               (double)stats.latency_sum_ns / (double)stats.jobs_timed * 1e-6);
    } else {
        printf("no job was timed\n");
    }
}

void parsec_qpu_module_free(parsec_device_module_t **modules)
{
    parsec_device_qpu_module_t *qpu = parsec_qpu_module;

    if( NULL != qpu ) {
        if( qpu->attached ) {
            if( qpu_device_param("show_statistics") ) qpu_module_print_statistics(qpu);
            (void)qpu_module_detach(&qpu->super, NULL);
        }
        parsec_qpu_module = NULL;
        free(qpu);
    }
    free(modules);
}

int parsec_qpu_device_info(int index, parsec_qpu_device_info_t *info)
{
    parsec_device_qpu_module_t *qpu = parsec_qpu_module;

    if( NULL == info ) return PARSEC_ERR_BAD_PARAM;
    memset(info, 0, sizeof(*info));
    info->device_index = -1;
    info->owner_rank   = -1;
    if( (0 != index) || (NULL == qpu) || !qpu->attached ) return PARSEC_ERR_NOT_FOUND;

    info->device_index         = qpu->super.device_index;
    info->owner_rank           = qpu->owner_rank;
    info->is_owner             = qpu->owner;
    info->backend              = parsec_qpu_params.backend;
    info->nb_qubits            = qpu->caps.nb_qubits;
    info->max_shots            = qpu->caps.max_shots;
    info->max_circuits_per_job = qpu->caps.max_circuits_per_job;
    info->is_physical          = qpu->caps.is_physical;
    info->max_outstanding      = qpu->max_outstanding;
    info->max_batch            = qpu->max_batch;
    info->executed_tasks       = PARSEC_QPU_LOAD(&qpu->super.executed_tasks);
    info->load_ns              = PARSEC_QPU_LOAD(&qpu->super.device_load);
    info->task_time_ns         = qpu->super.time_estimate_default;
    info->latency_estimate_ns  = PARSEC_QPU_LOAD(&qpu->latency_ns);
    return PARSEC_SUCCESS;
}
