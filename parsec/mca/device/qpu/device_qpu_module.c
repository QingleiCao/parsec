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
 * The QPU device modules: the QPUs as the runtime sees them.
 *
 * What being a device brings.
 *   - A task class says which of its incarnations is quantum
 *     (BODY [type=QPU] in a JDF, PARSEC_DEV_QPU at a DTD insertion), and
 *     parsec_select_best_device() selects a module of this type for it.
 *     Without a QPU in the context, the QPU incarnations are dropped when a
 *     taskpool starts and another incarnation of the class runs, if it has
 *     one.
 *   - The scheduler keeps the load of the device: the estimated time of the
 *     tasks it holds, which is a nominal second per task, so the load counts
 *     tasks; the time jobs really take is measured by the service and
 *     reported next to it (parsec_qpu_device_info).
 *   - The tasks it executed are counted with those of the other devices.
 *
 * What it does not do. A QPU holds no data, so the module has no memory to
 * register and no copy to manage: a quantum task reads and writes host
 * copies, and the scheduler treats its inputs as it treats those of a CPU
 * task. It has no flop rate, and takes no part in the weights the device
 * layer derives from them. And it runs nothing itself: a body of type QPU
 * calls parsec_qpu_submit(), which gives the task to the service.
 *
 * Which QPU a task is on. The device layer selects a module for a task
 * before the body runs, and with several QPUs on a rank it always selects
 * the same one. That choice only says "a QPU of this rank". Because a QPU
 * holds no data, nothing has to be decided that early: the service binds
 * the task to the QPU that has a place for its job when it has one, and
 * moves the task's device and load there (device_qpu_progress.c). So the
 * load of a QPU counts the tasks it holds a job for.
 *
 * Which rank has which module. The list of the QPUs is the same on every
 * rank, and every rank is offered a module for each entry. A rank attaches
 * the modules of the QPUs it owns, each with its backend. A rank that owns
 * none attaches the first one as a placeholder with no backend, so that the
 * QPU incarnations of a task class are kept on every rank and a taskpool
 * looks the same everywhere. A placeholder takes no work: a quantum task
 * that runs on such a rank was misplaced, and its submission is refused
 * with a message that says so.
 */

static parsec_device_qpu_module_t *qpu_modules[PARSEC_QPU_MAX_QPUS];
static int qpu_nb_modules = 0;

static int qpu_device_param(const char *name)
{
    int index = parsec_mca_param_find("device", NULL, name), value = 0;

    if( 0 < index ) (void)parsec_mca_param_lookup_int(index, &value);
    return value;
}

/* Every taskpool may use the device, on every rank: see the top of this file
 * for why a rank that owns no QPU does not decline here. */
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

/* Does this rank own any QPU of the list? */
static int qpu_rank_owns_one(const parsec_context_t *context)
{
    for( int q = 0; q < qpu_nb_modules; q++ ) {
        if( context->my_rank == parsec_qpu_list_rank(q, context->nb_nodes) ) return 1;
    }
    return 0;
}

static int qpu_module_attach(parsec_device_module_t *device, parsec_context_t *context)
{
    parsec_device_qpu_module_t *qpu = (parsec_device_qpu_module_t*)device;
    const char *backend = parsec_qpu_list_backend(qpu->number);
    const int owner_rank = parsec_qpu_list_rank(qpu->number, context->nb_nodes);
    int rc;

    if( (owner_rank < 0) || (owner_rank >= context->nb_nodes) ) {
        if( 0 == context->my_rank ) {
            parsec_warning("QPU %d: it is to be on rank %d and there are %d rank(s) (device_qpu_devices, "
                           "device_qpu_owner_rank): no rank owns it, and no quantum task can use it",
                           qpu->number, owner_rank, context->nb_nodes);
        }
    }
    qpu->owner_rank = owner_rank;
    qpu->nb_ranks   = context->nb_nodes;
    qpu->owner      = (context->my_rank == owner_rank);
    qpu->service    = -1;
    memset(&qpu->caps, 0, sizeof(qpu->caps));
    qpu->max_outstanding = 0;
    qpu->max_batch       = 0;
    if( parsec_mca_device_enabled() >= (int)(8 * sizeof(((parsec_taskpool_t*)0)->devices_index_mask)) ) {
        /* A taskpool says which devices it may use with one bit each, and
         * has no bit left for this one: it would be added and never used. */
        parsec_warning("QPU %d: this process has %d devices already, which is as many as a taskpool can use; "
                       "the QPU is left out", qpu->number, parsec_mca_device_enabled());
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    if( qpu->owner ) {
        /* A backend that cannot be used gives no device. What was wrong has
         * been said. */
        rc = parsec_qpu_service_open(qpu, &qpu->caps, &qpu->max_outstanding, &qpu->max_batch);
        if( PARSEC_SUCCESS != rc ) return rc;
    } else if( (0 != qpu->number) || qpu_rank_owns_one(context) ) {
        /* Another rank's, and this rank needs no placeholder or has it in
         * the first module. */
        return PARSEC_ERR_NOT_FOUND;
    }

    if( 1 == qpu_nb_modules ) {
        snprintf(qpu->name_buffer, sizeof(qpu->name_buffer), "qpu(%s)", (NULL != backend) ? backend : "?");
    } else if( qpu->owner ) {
        snprintf(qpu->name_buffer, sizeof(qpu->name_buffer), "qpu%d(%s)", qpu->number, (NULL != backend) ? backend : "?");
    } else {
        snprintf(qpu->name_buffer, sizeof(qpu->name_buffer), "qpu(elsewhere)");
    }
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
        /* The backend that was opened above is closed again: a QPU without
         * a device is no QPU of this rank. */
        parsec_warning("QPU %d: the device could not be added to the context (%d)", qpu->number, rc);
        if( qpu->owner ) parsec_qpu_service_forget(qpu);
        return rc;
    }
    qpu->attached = 1;

    if( qpu_device_param("show_capabilities") ) {
        if( qpu->owner ) {
            parsec_inform("Dev %-2d %11s : QPU %d, backend %s%s\n"
                          "\tQubits             : %u\t[Shots per job: %u%s | Circuits per job: %u]\n"
                          "\tJobs outstanding   : at most %u, on rank %d\t[Tasks per job: at most %u]",
                          device->device_index, device->name, qpu->number, backend,
                          qpu->caps.is_physical ? " (a quantum device)" : " (no quantum device)",
                          qpu->caps.nb_qubits, qpu->caps.max_shots, (0 == qpu->caps.max_shots) ? " (no limit)" : "",
                          qpu->caps.max_circuits_per_job, qpu->max_outstanding, qpu->owner_rank, qpu->max_batch);
        } else {
            parsec_inform("Dev %-2d %11s : this rank owns none of the %d QPU(s) and takes no quantum task; "
                          "QPU 0 is on rank %d",
                          device->device_index, device->name, qpu_nb_modules, qpu->owner_rank);
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
    const int nb = parsec_qpu_list_size();
    parsec_device_module_t **modules;

    if( (nb < 1) || (nb > PARSEC_QPU_MAX_QPUS) ) return NULL;
    modules = (parsec_device_module_t**)calloc((size_t)nb + 1, sizeof(parsec_device_module_t*));
    if( NULL == modules ) return NULL;
    for( int q = 0; q < nb; q++ ) {
        /* As the device layer does for the CPU device: plain memory, zeroed. */
        parsec_device_qpu_module_t *qpu = (parsec_device_qpu_module_t*)calloc(1, sizeof(parsec_device_qpu_module_t));

        if( NULL == qpu ) {
            for( int i = 0; i < q; i++ ) free(qpu_modules[i]);
            memset(qpu_modules, 0, sizeof(qpu_modules));
            free(modules);
            return NULL;
        }
        qpu->super.component           = component;
        qpu->super.attach              = qpu_module_attach;
        qpu->super.detach              = qpu_module_detach;
        qpu->super.taskpool_register   = qpu_module_taskpool_register;
        qpu->super.taskpool_unregister = qpu_module_taskpool_unregister;
        qpu->super.type                = PARSEC_DEV_QPU;
        qpu->super.name                = "qpu";
        qpu->number                    = q;
        qpu->owner_rank                = -1;
        qpu->service                   = -1;
        qpu_modules[q] = qpu;
        modules[q]     = &qpu->super;
    }
    modules[nb]    = NULL;
    qpu_nb_modules = nb;
    return modules;
}

/* One line per QPU under the table the device layer prints: what a QPU did
 * has no column there. The service is stopped by now, and its counters are
 * those of its last run. */
static void qpu_module_print_statistics(const parsec_device_qpu_module_t *qpu)
{
    parsec_qpu_stats_t stats, rank;

    if( !qpu->owner || (PARSEC_SUCCESS != parsec_qpu_service_stats_of(qpu->number, &stats)) ) return;
    parsec_qpu_service_stats(&rank);
    printf("QPU %d, %s on rank %d: %"PRIu64" task(s), %"PRIu64" job(s) at the backend; "
           "%"PRIu64" job(s) failed, %"PRIu64" timed out, %"PRIu64" retried, %"PRIu64" submission(s) deferred\n"
           "    at most %u job(s) outstanding (limit %u); "
           "%"PRIu64" job(s) carried %"PRIu64" task(s) together, at most %u in one (limit %u); ",
           qpu->number, qpu->name_buffer, qpu->owner_rank,
           stats.tasks_executed, stats.jobs_submitted,
           stats.jobs_failed, stats.jobs_timed_out, stats.retries, stats.submit_deferred,
           stats.max_in_flight, qpu->max_outstanding,
           stats.jobs_coalesced, stats.tasks_coalesced, stats.max_job_tasks, qpu->max_batch);
    if( stats.jobs_timed > 0 ) {
        printf("a job took %.3f ms on average from its submission to its result\n",
               (double)stats.latency_sum_ns / (double)stats.jobs_timed * 1e-6);
    } else {
        printf("no job was timed\n");
    }
    if( 0 == qpu->service ) {
        /* What is counted for the rank, once. */
        printf("QPUs of rank %d: %"PRIu64" task(s) in all, %"PRIu64" without a job, %"PRIu64" ended without a result; "
               "at most %u task(s) waiting; %"PRIu64" given to another QPU\n",
               qpu->owner_rank, rank.tasks_completed, rank.empty_jobs, rank.tasks_failed,
               rank.max_pending, rank.tasks_moved);
    }
}

void parsec_qpu_module_free(parsec_device_module_t **modules)
{
    for( int q = 0; q < qpu_nb_modules; q++ ) {
        parsec_device_qpu_module_t *qpu = qpu_modules[q];

        if( NULL == qpu ) continue;
        if( qpu->attached ) {
            if( qpu_device_param("show_statistics") ) qpu_module_print_statistics(qpu);
            (void)qpu_module_detach(&qpu->super, NULL);
        }
        qpu_modules[q] = NULL;
        free(qpu);
    }
    qpu_nb_modules = 0;
    free(modules);
}

int parsec_qpu_module_placeholder(void)
{
    if( (qpu_nb_modules > 0) && (NULL != qpu_modules[0]) && qpu_modules[0]->attached && !qpu_modules[0]->owner ) {
        return qpu_modules[0]->owner_rank;
    }
    return -2;
}

int parsec_qpu_count(void)
{
    return qpu_nb_modules;
}

int parsec_qpu_rank_of(int qpu)
{
    const parsec_device_qpu_module_t *module;

    if( (qpu < 0) || (qpu >= qpu_nb_modules) || (NULL == (module = qpu_modules[qpu])) ) return -1;
    /* Every module was asked to attach, and learnt then where its QPU is. */
    return ((module->owner_rank >= 0) && (module->owner_rank < module->nb_ranks)) ? module->owner_rank : -1;
}

int parsec_qpu_device_info(int index, parsec_qpu_device_info_t *info)
{
    const parsec_device_qpu_module_t *qpu, *here;

    if( NULL == info ) return PARSEC_ERR_BAD_PARAM;
    memset(info, 0, sizeof(*info));
    info->device_index = -1;
    info->owner_rank   = -1;
    if( (index < 0) || (index >= qpu_nb_modules) || (NULL == (qpu = qpu_modules[index])) ) return PARSEC_ERR_NOT_FOUND;
    /* The device that stands for this QPU on this rank: its own on the rank
     * that owns it, the placeholder on a rank that owns none. */
    here = qpu->owner ? qpu : qpu_modules[0];
    if( qpu->owner && !qpu->attached ) return PARSEC_ERR_NOT_FOUND;       /* its backend could not be used */
    if( !qpu->owner && ((NULL == here) || !here->attached || here->owner) ) here = NULL;

    info->owner_rank = qpu->owner_rank;
    info->is_owner   = qpu->owner;
    info->backend    = parsec_qpu_list_backend(index);
    if( NULL == here ) {
        /* A rank that owns other QPUs and not this one: it knows where the
         * QPU is, and has no device for it. */
        info->task_time_ns = PARSEC_QPU_TASK_TIME_NS;
        return PARSEC_SUCCESS;
    }
    info->device_index         = here->super.device_index;
    info->nb_qubits            = qpu->caps.nb_qubits;
    info->max_shots            = qpu->caps.max_shots;
    info->max_circuits_per_job = qpu->caps.max_circuits_per_job;
    info->is_physical          = qpu->caps.is_physical;
    info->max_outstanding      = qpu->max_outstanding;
    info->max_batch            = qpu->max_batch;
    info->executed_tasks       = PARSEC_QPU_LOAD(&here->super.executed_tasks);
    info->load_ns              = PARSEC_QPU_LOAD(&here->super.device_load);
    info->task_time_ns         = here->super.time_estimate_default;
    info->latency_estimate_ns  = PARSEC_QPU_LOAD(&qpu->latency_ns);
    return PARSEC_SUCCESS;
}
