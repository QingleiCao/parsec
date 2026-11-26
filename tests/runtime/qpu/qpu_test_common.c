/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "qpu_test_common.h"
#include "parsec/parsec_internal.h"
#include "parsec/datatype.h"
#include "parsec/data_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * Placement
 * --------------------------------------------------------------------- */

typedef struct {
    parsec_data_collection_t super;
    int fixed_rank;
    int nb_ranks;       /* not zero: element k is on ranks[k mod nb_ranks] */
    int ranks[64];
} qpu_test_placement_t;

static uint32_t placement_rank(parsec_data_collection_t *dc, int k)
{
    qpu_test_placement_t *p = (qpu_test_placement_t*)dc;
    if( k < 0 ) k = -k;
    if( p->nb_ranks > 0 ) return (uint32_t)p->ranks[k % p->nb_ranks];
    if( p->fixed_rank >= 0 ) return (uint32_t)p->fixed_rank;
    return (uint32_t)(k % (int)dc->nodes);
}

static uint32_t placement_rank_of(parsec_data_collection_t *dc, ...)
{
    va_list ap;
    int k;
    va_start(ap, dc);
    k = va_arg(ap, int);
    va_end(ap);
    return placement_rank(dc, k);
}

static uint32_t placement_rank_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    return placement_rank(dc, (int)key);
}

static int32_t placement_vpid_of(parsec_data_collection_t *dc, ...)
{
    (void)dc;
    return 0;
}

static int32_t placement_vpid_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    (void)dc; (void)key;
    return 0;
}

static parsec_data_key_t placement_data_key(parsec_data_collection_t *dc, ...)
{
    va_list ap;
    int k;
    (void)dc;
    va_start(ap, dc);
    k = va_arg(ap, int);
    va_end(ap);
    return (parsec_data_key_t)k;
}

/* Nothing reads data from a placement collection. */
static parsec_data_t *placement_data_of(parsec_data_collection_t *dc, ...)
{
    (void)dc;
    return NULL;
}

static parsec_data_t *placement_data_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    (void)dc; (void)key;
    return NULL;
}

parsec_data_collection_t *qpu_test_placement_new(int myrank, int nodes, int fixed_rank)
{
    qpu_test_placement_t *p = (qpu_test_placement_t*)calloc(1, sizeof(qpu_test_placement_t));
    parsec_data_collection_t *dc = &p->super;

    parsec_data_collection_init(dc, nodes, myrank);
    dc->rank_of     = placement_rank_of;
    dc->rank_of_key = placement_rank_of_key;
    dc->vpid_of     = placement_vpid_of;
    dc->vpid_of_key = placement_vpid_of_key;
    dc->data_key    = placement_data_key;
    dc->data_of     = placement_data_of;
    dc->data_of_key = placement_data_of_key;
    p->fixed_rank   = (fixed_rank >= nodes) ? nodes - 1 : fixed_rank;
    return dc;
}

parsec_data_collection_t *qpu_test_placement_list_new(int myrank, int nodes, const int *ranks, int nb_ranks)
{
    parsec_data_collection_t *dc = qpu_test_placement_new(myrank, nodes, -1);
    qpu_test_placement_t *p = (qpu_test_placement_t*)dc;

    if( nb_ranks > 64 ) nb_ranks = 64;
    for( int i = 0; i < nb_ranks; i++ ) p->ranks[i] = (ranks[i] < nodes) ? ranks[i] : nodes - 1;
    p->nb_ranks = nb_ranks;
    return dc;
}

void qpu_test_placement_free(parsec_data_collection_t *dc)
{
    parsec_data_collection_destroy(dc);
    free(dc);
}

/* ------------------------------------------------------------------------
 * Tiles on one rank
 * --------------------------------------------------------------------- */

typedef struct {
    parsec_data_collection_t super;
    int             owner;
    int             n;
    parsec_data_t **data;
    unsigned char  *memory;
    parsec_data_flag_t flags;
} qpu_test_tiles_t;

static uint32_t tiles_rank_of(parsec_data_collection_t *dc, ...)
{
    return (uint32_t)((qpu_test_tiles_t*)dc)->owner;
}

static uint32_t tiles_rank_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    (void)key;
    return (uint32_t)((qpu_test_tiles_t*)dc)->owner;
}

static parsec_data_t *tiles_data_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    qpu_test_tiles_t *t = (qpu_test_tiles_t*)dc;
    return parsec_data_create(&t->data[key], dc, key, t->memory + (size_t)key * QPU_TEST_BLOCK,
                              QPU_TEST_BLOCK, t->flags);
}

static parsec_data_t *tiles_data_of(parsec_data_collection_t *dc, ...)
{
    va_list ap;
    int k;
    va_start(ap, dc);
    k = va_arg(ap, int);
    va_end(ap);
    return tiles_data_of_key(dc, (parsec_data_key_t)k);
}

parsec_data_collection_t *qpu_test_tiles_new(int myrank, int nodes, int owner, int n)
{
    qpu_test_tiles_t *t = (qpu_test_tiles_t*)calloc(1, sizeof(qpu_test_tiles_t));
    parsec_data_collection_t *dc = &t->super;

    parsec_data_collection_init(dc, nodes, myrank);
    dc->rank_of     = tiles_rank_of;
    dc->rank_of_key = tiles_rank_of_key;
    dc->vpid_of     = placement_vpid_of;
    dc->vpid_of_key = placement_vpid_of_key;
    dc->data_key    = placement_data_key;
    dc->data_of     = tiles_data_of;
    dc->data_of_key = tiles_data_of_key;
    t->owner = (owner >= nodes) ? nodes - 1 : owner;
    t->n     = n;
    t->flags = PARSEC_DATA_FLAG_PARSEC_MANAGED;
    t->data  = (parsec_data_t**)calloc(n > 0 ? n : 1, sizeof(parsec_data_t*));
    if( myrank == t->owner ) {
        t->memory = (unsigned char*)calloc(n > 0 ? n : 1, QPU_TEST_BLOCK);
    }
    return dc;
}

void qpu_test_tiles_user_managed(parsec_data_collection_t *dc)
{
    ((qpu_test_tiles_t*)dc)->flags = 0;
}

void *qpu_test_tiles_ptr(parsec_data_collection_t *dc, int k)
{
    qpu_test_tiles_t *t = (qpu_test_tiles_t*)dc;
    return (NULL != t->memory) ? t->memory + (size_t)k * QPU_TEST_BLOCK : NULL;
}

void qpu_test_tiles_free(parsec_data_collection_t *dc)
{
    qpu_test_tiles_t *t = (qpu_test_tiles_t*)dc;
    for( int k = 0; k < t->n; k++ ) {
        if( NULL != t->data[k] ) parsec_data_destroy(t->data[k]);
    }
    free(t->data);
    free(t->memory);
    parsec_data_collection_destroy(dc);
    free(dc);
}

/* ------------------------------------------------------------------------
 * Data
 * --------------------------------------------------------------------- */

void qpu_test_block_type_set(parsec_arena_datatype_t *adt)
{
    parsec_datatype_t block = PARSEC_DATATYPE_NULL;

    /* Without MPI there is no datatype to create and none is needed. The
     * size of the arena is given in bytes either way. */
    (void)parsec_type_create_contiguous(QPU_TEST_BLOCK, parsec_datatype_uint8_t, &block);
    parsec_arena_datatype_set_type(adt, QPU_TEST_BLOCK, PARSEC_ARENA_ALIGNMENT_SSE, block);
}

void qpu_test_block_type_unset(parsec_arena_datatype_t *adt)
{
    if( PARSEC_DATATYPE_NULL != adt->opaque_dtt ) {
        parsec_type_free(&adt->opaque_dtt);
    }
}

void qpu_test_make_circuit(void *buffer, int a, int b, int empty)
{
    parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)buffer;
    const int n = QPU_TEST_NB_QUBITS;
    int rc = 0;

    memset(buffer, 0, QPU_TEST_BLOCK);
    parsec_qpu_circuit_init(circuit, n, QPU_TEST_NB_PARAMS);
    if( empty ) circuit->flags |= PARSEC_QPU_CIRCUIT_EMPTY;

    /* The shape of a variational circuit: a layer of Hadamards, entangling
     * rotations whose angle is a parameter, local rotations, measurement.
     * Coefficients and immediate angles depend on (a, b). */
    for( int q = 0; q < n; q++ ) {
        rc |= parsec_qpu_circuit_add(circuit, QPU_TEST_BLOCK, PARSEC_QPU_OP_H, q, 0, PARSEC_QPU_NO_PARAM, 0.0);
    }
    for( int q = 0; q + 1 < n; q++ ) {
        rc |= parsec_qpu_circuit_add(circuit, QPU_TEST_BLOCK, PARSEC_QPU_OP_RZZ, q, q + 1, 2, 0.3 * (1 + q));
    }
    for( int q = 0; q < n; q++ ) {
        rc |= parsec_qpu_circuit_add(circuit, QPU_TEST_BLOCK, PARSEC_QPU_OP_RY, q, 0, PARSEC_QPU_NO_PARAM,
                                     0.2 * ((a + 2 * b + q) % 9));
    }
    for( int q = 0; q < n; q++ ) {
        rc |= parsec_qpu_circuit_add(circuit, QPU_TEST_BLOCK, PARSEC_QPU_OP_RX, q, 0, 3, 2.0);
    }
    rc |= parsec_qpu_circuit_measure_all(circuit, QPU_TEST_BLOCK);
    if( 0 != rc ) {
        fprintf(stderr, "qpu_test_make_circuit: the test circuit does not fit in a block\n");
        abort();
    }
}

void qpu_test_make_params(void *buffer, int a, int b)
{
    double *params = (double*)buffer;

    memset(buffer, 0, QPU_TEST_BLOCK);
    params[0] = (double)a;
    params[1] = (double)b;
    params[2] = 0.5 * a - 0.25 * b;
    params[3] = 1.0 / (1.0 + a + b);
}

int qpu_test_params_a(const double *params)
{
    return (int)params[0];
}

#define CHECK(COND, ...)                                                    \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            fprintf(stderr, "%s(%d, %d): ", who, a, b);                     \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            errors++;                                                       \
        }                                                                   \
    } while(0)

int qpu_test_check_counts(const void *vresult, const void *vcircuit, const void *vparams,
                          int shots, uint64_t key, const char *who, int a, int b)
{
    const parsec_qpu_result_t *result = (const parsec_qpu_result_t*)vresult;
    const parsec_qpu_circuit_t *circuit = (const parsec_qpu_circuit_t*)vcircuit;
    const double *params = (const double*)vparams;
    uint64_t bits, previous = 0, total = 0;
    uint32_t count;
    int errors = 0;

    CHECK(result->nb_entries <= QPU_TEST_MAX_ENTRIES, "%u entries do not fit in the block", result->nb_entries);
    if( result->nb_entries > QPU_TEST_MAX_ENTRIES ) return errors;

    for( uint32_t i = 0; i < result->nb_entries; i++ ) {
        parsec_qpu_result_get_entry(result, i, &bits, &count);
        CHECK(0 == (bits >> circuit->nb_qubits), "entry %u is %llx, outside a register of %u qubits",
              i, (unsigned long long)bits, circuit->nb_qubits);
        CHECK((0 == i) || (bits > previous), "entry %u is not after entry %u", i, i - 1);
        CHECK(count > 0, "entry %u counts no shot", i);
        previous = bits;
        total += count;
    }
    CHECK(total == result->shots_returned, "the entries count %llu shots, the header says %u",
          (unsigned long long)total, result->shots_returned);

    if( PARSEC_QPU_KEY_AUTO != key ) {
        /* What the sampler gives for these inputs, outside any runtime. */
        union { parsec_qpu_result_t header; unsigned char bytes[QPU_TEST_BLOCK]; } expected;
        int rc;

        memset(&expected, 0, sizeof(expected));
        rc = parsec_qpu_sampler_sample(circuit, params, (uint32_t)shots, (uint64_t)parsec_qpu_params.mock_seed, key,
                                       &expected.header, QPU_TEST_MAX_ENTRIES);
        CHECK(PARSEC_SUCCESS == rc, "the sampler failed on the inputs (%d)", rc);
        CHECK(expected.header.nb_entries == result->nb_entries, "%u entries, expected %u",
              result->nb_entries, expected.header.nb_entries);
        CHECK(expected.header.shots_returned == result->shots_returned, "%u shots returned, expected %u",
              result->shots_returned, expected.header.shots_returned);
        if( (PARSEC_SUCCESS == rc) && (expected.header.nb_entries == result->nb_entries) ) {
            CHECK(0 == memcmp(result + 1, &expected.header + 1, (size_t)result->nb_entries * PARSEC_QPU_ENTRY_SIZE),
                  "the counts are not the ones of these inputs and this key");
        }
    }
    return errors;
}

int qpu_test_max_batch(void)
{
    int batch = parsec_qpu_params.max_batch, circuits = parsec_qpu_params.mock_max_circuits;

    if( batch < 1 ) batch = 1;
    if( circuits < 1 ) circuits = 1;
    return (batch < circuits) ? batch : circuits;
}

int qpu_test_check_result(const void *vresult, const void *vcircuit, const void *vparams,
                          int shots, uint64_t key, const char *who, int a, int b)
{
    const parsec_qpu_result_t *result = (const parsec_qpu_result_t*)vresult;
    const parsec_qpu_circuit_t *circuit = (const parsec_qpu_circuit_t*)vcircuit;
    int errors = 0;

    CHECK(PARSEC_QPU_RESULT_VERSION == result->version, "result version is %u", (unsigned)result->version);
    CHECK(PARSEC_QPU_STATUS_OK == result->status, "status is %u", (unsigned)result->status);
    CHECK((uint32_t)shots == result->shots_requested, "%u shots requested, expected %d", result->shots_requested, shots);
    CHECK(circuit->nb_qubits == result->nb_qubits, "%u qubits, expected %u", (unsigned)result->nb_qubits, circuit->nb_qubits);
    CHECK((result->batch >= 1) && (result->batch <= (uint32_t)qpu_test_max_batch()),
          "its job carried %u task(s), and a job carries 1 to %d", result->batch, qpu_test_max_batch());
    CHECK(result->t_hook <= result->t_submit, "submitted before its body ran");
    CHECK(result->t_submit <= result->t_retrieve, "retrieved before it was submitted");
    if( PARSEC_QPU_RESULT_VERSION != result->version ) return errors;

    if( PARSEC_QPU_CIRCUIT_EMPTY & circuit->flags ) {
        CHECK(0 == result->nb_entries, "an empty job returned %u entries", result->nb_entries);
        CHECK(0 == result->shots_returned, "an empty job returned %u shots", result->shots_returned);
        CHECK(0 == result->attempts, "an empty job was submitted %u time(s)", (unsigned)result->attempts);
        CHECK(0 == (PARSEC_QPU_RESULT_MOCK & result->flags), "an empty job is flagged as coming from a backend");
        return errors;
    }

    CHECK(PARSEC_QPU_RESULT_MOCK & result->flags, "the result is not flagged as mock");
    CHECK(0 == (PARSEC_QPU_RESULT_TRUNCATED & result->flags), "the result is flagged as truncated");
    CHECK(1 == result->attempts, "%u attempts", (unsigned)result->attempts);
    CHECK((uint32_t)shots == result->shots_returned, "%u shots returned, expected %d", result->shots_returned, shots);
    CHECK(result->t_submit <= result->t_start, "started before it was submitted");
    CHECK(result->t_start <= result->t_end, "ended before it started");
    CHECK(result->t_end <= result->t_retrieve, "retrieved before it ended");
    if( PARSEC_QPU_STATUS_OK == result->status ) {
        errors += qpu_test_check_counts(vresult, vcircuit, vparams, shots, key, who, a, b);
    }
    return errors;
}

#undef CHECK

#define EXPECT(COND, ...)                                                   \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            fprintf(stderr, "QPU service: ");                               \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            errors++;                                                       \
        }                                                                   \
    } while(0)

static long also_executed = 0;

void qpu_test_expect_also_executed(long tasks)
{
    also_executed = tasks;
}

int qpu_test_check_device(int owner, long executed)
{
    parsec_qpu_device_info_t info;
    parsec_qpu_stats_t stats;
    parsec_device_module_t *device;
    int errors = 0;

    if( PARSEC_SUCCESS != parsec_qpu_device_info(0, &info) ) {
        fprintf(stderr, "QPU device: this process has none\n");
        return 1;
    }
    parsec_qpu_service_stats(&stats);
    device = parsec_mca_device_get((uint32_t)info.device_index);
    EXPECT((NULL != device) && (PARSEC_DEV_QPU == device->type), "device %d is not of the QPU type", info.device_index);
    EXPECT(!parsec_mca_device_is_gpu((uint32_t)info.device_index), "device %d is taken for a GPU", info.device_index);
    EXPECT(!!owner == !!info.is_owner, "this rank %s the QPU, and was expected %sto",
           info.is_owner ? "owns" : "does not own", owner ? "" : "not ");
    EXPECT(0 == info.load_ns, "the device still holds a load of %lld ns", (long long)info.load_ns);
    EXPECT((uint64_t)executed == info.executed_tasks, "the device executed %llu tasks, expected %ld",
           (unsigned long long)info.executed_tasks, executed);
    /* The statistics the runtime keeps for every device, as it prints them
     * with device_show_statistics, count the same tasks. */
    {
        uint64_t *saved = NULL;
        parsec_devices_save_statistics(&saved);
        EXPECT((NULL != saved) && ((uint64_t)executed == saved[info.device_index]),
               "the device statistics of the runtime count %llu tasks for the QPU, expected %ld",
               (NULL != saved) ? (unsigned long long)saved[info.device_index] : 0ULL, executed);
        if( NULL != saved ) parsec_devices_free_statistics(&saved);
    }
    /* What the description says is what the runtime itself reads in the module. */
    if( NULL != device ) {
        EXPECT(device->device_load == info.load_ns, "the device is described with a load of %lld ns, and has %lld ns",
               (long long)info.load_ns, (long long)device->device_load);
        EXPECT(device->executed_tasks == info.executed_tasks, "the device is described with %llu executed tasks, and has %llu",
               (unsigned long long)info.executed_tasks, (unsigned long long)device->executed_tasks);
        EXPECT((PARSEC_QPU_TASK_TIME_NS == device->time_estimate_default) && (info.task_time_ns == device->time_estimate_default),
               "the scheduler is given %lld ns per quantum task and the device is described with %lld ns, expected %lld ns",
               (long long)device->time_estimate_default, (long long)info.task_time_ns, (long long)PARSEC_QPU_TASK_TIME_NS);
    }
    if( info.is_owner ) {
        EXPECT(info.nb_qubits == (uint32_t)parsec_qpu_params.mock_max_qubits,
               "the device takes %u qubits, the mock was given %d", info.nb_qubits, parsec_qpu_params.mock_max_qubits);
        EXPECT((info.max_outstanding >= 1) && (info.max_outstanding <= (uint32_t)parsec_qpu_params.max_outstanding),
               "the device keeps at most %u jobs outstanding, the limit is %d",
               info.max_outstanding, parsec_qpu_params.max_outstanding);
        EXPECT(0 == info.is_physical, "the device says it is a quantum device");
        if( stats.jobs_timed > 0 ) {
            /* One job at least ended well: the device says how long a job takes,
             * and it is a time some job of this run can have taken. */
            EXPECT((info.latency_estimate_ns > 0) && (info.latency_estimate_ns <= stats.latency_sum_ns),
                   "the device says a job takes %lld ns; %llu jobs were timed and took %lld ns together",
                   (long long)info.latency_estimate_ns, (unsigned long long)stats.jobs_timed,
                   (long long)stats.latency_sum_ns);
        }
    } else {
        EXPECT(0 == info.max_outstanding, "a rank that does not own the QPU keeps %u jobs outstanding", info.max_outstanding);
        EXPECT(0 == info.nb_qubits, "a rank that does not own the QPU describes a backend");
        EXPECT(0 == stats.backend_open, "a rank that does not own the QPU has a backend open");
    }
    return errors;
}

int qpu_test_check_invariants(long tasks, int verbose)
{
    parsec_qpu_stats_t stats;
    parsec_qpu_mock_stats_t mock;
    int errors = 0;

    parsec_qpu_service_stats(&stats);
    parsec_qpu_mock_stats(&mock);

    EXPECT((uint64_t)tasks == stats.tasks_submitted, "%llu tasks submitted, expected %ld",
           (unsigned long long)stats.tasks_submitted, tasks);
    EXPECT((uint64_t)tasks == stats.tasks_completed, "%llu tasks completed, expected %ld",
           (unsigned long long)stats.tasks_completed, tasks);
    EXPECT(0 == stats.pending, "%u tasks still pending", stats.pending);
    EXPECT(0 == stats.in_flight, "%u jobs still in flight", stats.in_flight);
    EXPECT(stats.max_in_flight <= (uint32_t)parsec_qpu_params.max_outstanding,
           "%u jobs were outstanding at once, the limit is %d", stats.max_in_flight, parsec_qpu_params.max_outstanding);
    /* What the backend saw has to agree with what the service believes. */
    EXPECT(stats.jobs_submitted + mock.abandoned == mock.submits,
           "the backend created %llu jobs; the service counts %llu, and gave up on the submission of %llu",
           (unsigned long long)mock.submits, (unsigned long long)stats.jobs_submitted,
           (unsigned long long)mock.abandoned);
    EXPECT(0 == mock.unknown_jobs, "the backend holds %u job(s) nobody knows of", mock.unknown_jobs);
    EXPECT(mock.max_held <= (uint32_t)parsec_qpu_params.max_outstanding,
           "the backend held %u jobs at once, the limit is %d", mock.max_held, parsec_qpu_params.max_outstanding);
    EXPECT(mock.max_running <= (uint32_t)parsec_qpu_params.mock_slots,
           "the backend ran %u jobs at once on %d slots", mock.max_running, parsec_qpu_params.mock_slots);
    EXPECT(0 == mock.duplicate_tags, "%llu jobs were asked for under a tag that was used",
           (unsigned long long)mock.duplicate_tags);
    EXPECT(0 == mock.key_overlaps, "%llu jobs were created while another job of the same evaluation was held",
           (unsigned long long)mock.key_overlaps);
    EXPECT(0 == mock.inputs_changed, "the inputs of %llu jobs changed while the backend held them",
           (unsigned long long)mock.inputs_changed);
    EXPECT(0 == mock.mixed_shots, "%llu jobs carried circuits that ask for different numbers of shots",
           (unsigned long long)mock.mixed_shots);
    EXPECT(0 == mock.changed_jobs, "%llu jobs were asked for again with other circuits than they had",
           (unsigned long long)mock.changed_jobs);
    /* Tasks that share a job: never more than allowed, and the counters of
     * the service and of the backend tell the same story. */
    EXPECT(stats.max_job_tasks <= (uint32_t)qpu_test_max_batch(), "a job carried %u tasks, the limit is %d",
           stats.max_job_tasks, qpu_test_max_batch());
    EXPECT(mock.max_circuits <= (uint32_t)qpu_test_max_batch(), "the backend was given a job of %u circuits, the limit is %d",
           mock.max_circuits, qpu_test_max_batch());
    EXPECT((stats.jobs_coalesced <= stats.jobs_submitted) && (stats.tasks_coalesced >= 2 * stats.jobs_coalesced) &&
           (stats.tasks_coalesced <= (uint64_t)stats.max_job_tasks * stats.jobs_coalesced),
           "%llu of %llu jobs carried %llu tasks together, at most %u in one: these cannot all be true",
           (unsigned long long)stats.jobs_coalesced, (unsigned long long)stats.jobs_submitted,
           (unsigned long long)stats.tasks_coalesced, stats.max_job_tasks);
    EXPECT(mock.circuits >= mock.submits, "the backend created %llu jobs with %llu circuits in all",
           (unsigned long long)mock.submits, (unsigned long long)mock.circuits);
    if( 0 == mock.abandoned ) {
        EXPECT(mock.circuits == stats.jobs_submitted - stats.jobs_coalesced + stats.tasks_coalesced,
               "the backend was given %llu circuits; the service counts %llu jobs, of which %llu carried %llu tasks",
               (unsigned long long)mock.circuits, (unsigned long long)stats.jobs_submitted,
               (unsigned long long)stats.jobs_coalesced, (unsigned long long)stats.tasks_coalesced);
    }
    if( (parsec_qpu_params.submit_interval_ms > 0) && (mock.min_submit_gap >= 0) ) {
        EXPECT(mock.min_submit_gap >= (int64_t)parsec_qpu_params.submit_interval_ms * 1000000LL,
               "two jobs were submitted %.3f ms apart, the minimum is %d ms",
               mock.min_submit_gap * 1e-6, parsec_qpu_params.submit_interval_ms);
    }

    if( verbose ) {
        printf("QPU service: %llu tasks, %llu jobs, %llu empty; at most %u outstanding, %u pending; "
               "%llu jobs carried %llu tasks together, at most %u in one; "
               "progress thread used %.3f ms of CPU in %.3f ms, %llu waits\n",
               (unsigned long long)stats.tasks_completed, (unsigned long long)stats.jobs_submitted,
               (unsigned long long)stats.empty_jobs, stats.max_in_flight, stats.max_pending,
               (unsigned long long)stats.jobs_coalesced, (unsigned long long)stats.tasks_coalesced, stats.max_job_tasks,
               stats.progress_cpu_ns * 1e-6, stats.progress_wall_ns * 1e-6,
               (unsigned long long)stats.wait_calls);
        printf("QPU service: %llu jobs failed, %llu timed out, %llu retries, %llu submissions deferred, "
               "%llu stray events; %llu tasks without a result (%llu rejected, %llu unavailable)\n",
               (unsigned long long)stats.jobs_failed, (unsigned long long)stats.jobs_timed_out,
               (unsigned long long)stats.retries, (unsigned long long)stats.submit_deferred,
               (unsigned long long)stats.stray_events, (unsigned long long)stats.tasks_failed,
               (unsigned long long)stats.tasks_rejected, (unsigned long long)stats.tasks_unavailable);
        printf("QPU mock: %llu submissions, %llu jobs of %llu circuits, %llu found by tag, %llu deferred, %llu refused, "
               "%llu cancelled, %llu given up on, %llu results, %llu failed, %llu sampled (%llu late); "
               "at most %u running, %u held; shortest time between two submissions %.3f ms\n",
               (unsigned long long)mock.submit_calls, (unsigned long long)mock.submits,
               (unsigned long long)mock.circuits,
               (unsigned long long)mock.tag_hits, (unsigned long long)mock.deferred,
               (unsigned long long)mock.rejected, (unsigned long long)mock.cancels,
               (unsigned long long)mock.abandoned, (unsigned long long)mock.results,
               (unsigned long long)mock.failures, (unsigned long long)mock.computed,
               (unsigned long long)mock.late, mock.max_running, mock.max_held, mock.min_submit_gap * 1e-6);
    }
    /* Every task the service completed is one the device executed, and the
     * load the scheduler added for each of them was taken back. */
    errors += qpu_test_check_device(1, tasks + also_executed);
    return errors;
}

int qpu_test_check_service(int ran_quantum_tasks, long tasks, long empty, int verbose)
{
    parsec_qpu_stats_t stats;
    parsec_qpu_mock_stats_t mock;
    int errors = 0;

    parsec_qpu_service_stats(&stats);
    parsec_qpu_mock_stats(&mock);

    if( !ran_quantum_tasks ) {
        EXPECT(0 == stats.running, "a progress thread runs on a rank that had no quantum task");
        EXPECT(0 == stats.tasks_submitted, "%llu tasks submitted on a rank that had no quantum task",
               (unsigned long long)stats.tasks_submitted);
        EXPECT(0 == mock.opens, "%llu backend(s) opened on a rank that had no quantum task",
               (unsigned long long)mock.opens);
        /* Such a rank does not own the QPU: it has the device, and nothing behind it. */
        errors += qpu_test_check_device(0, 0);
        return errors;
    }

    errors += qpu_test_check_invariants(tasks, verbose);

    EXPECT((uint64_t)empty == stats.empty_jobs, "%llu empty jobs, expected %ld",
           (unsigned long long)stats.empty_jobs, empty);
    /* Every task that needed a job was in exactly one, alone or not. */
    EXPECT((uint64_t)(tasks - empty) == stats.jobs_submitted - stats.jobs_coalesced + stats.tasks_coalesced,
           "%llu jobs submitted, of which %llu carried %llu tasks: that makes %llu tasks, expected %ld",
           (unsigned long long)stats.jobs_submitted, (unsigned long long)stats.jobs_coalesced,
           (unsigned long long)stats.tasks_coalesced,
           (unsigned long long)(stats.jobs_submitted - stats.jobs_coalesced + stats.tasks_coalesced), tasks - empty);
    if( qpu_test_max_batch() <= 1 ) {
        EXPECT((0 == stats.jobs_coalesced) && (stats.max_job_tasks <= 1),
               "%llu jobs carried more than one task, and no job may", (unsigned long long)stats.jobs_coalesced);
    }
    /* Nothing was injected, so nothing may have gone wrong. */
    EXPECT(0 == stats.jobs_failed, "%llu jobs failed", (unsigned long long)stats.jobs_failed);
    EXPECT(0 == stats.jobs_timed_out, "%llu jobs timed out", (unsigned long long)stats.jobs_timed_out);
    EXPECT(0 == stats.retries, "%llu retries", (unsigned long long)stats.retries);
    EXPECT(0 == stats.submit_deferred, "%llu submissions were deferred", (unsigned long long)stats.submit_deferred);
    EXPECT(0 == stats.stray_events, "%llu stray events", (unsigned long long)stats.stray_events);
    EXPECT(0 == stats.tasks_failed, "%llu tasks completed without a result", (unsigned long long)stats.tasks_failed);
    EXPECT(stats.jobs_submitted == stats.jobs_timed, "%llu of the %llu jobs were timed for the estimate of the device",
           (unsigned long long)stats.jobs_timed, (unsigned long long)stats.jobs_submitted);
    EXPECT(mock.submit_calls == mock.submits, "%llu submissions for %llu jobs",
           (unsigned long long)mock.submit_calls, (unsigned long long)mock.submits);
    EXPECT(mock.circuits == mock.results, "the backend gave out %llu results for %llu circuits",
           (unsigned long long)mock.results, (unsigned long long)mock.circuits);
    EXPECT(mock.circuits == mock.computed, "the backend sampled %llu circuits of the %llu it was given",
           (unsigned long long)mock.computed, (unsigned long long)mock.circuits);
    EXPECT(0 == mock.cancels, "%llu jobs were cancelled", (unsigned long long)mock.cancels);
    EXPECT(0 == mock.abandoned, "%llu submissions were given up on", (unsigned long long)mock.abandoned);
    EXPECT(0 == mock.failures, "%llu jobs failed at the backend", (unsigned long long)mock.failures);
    EXPECT(0 == mock.provider_cancels, "%llu jobs were cancelled at the provider", (unsigned long long)mock.provider_cancels);
    EXPECT(0 == mock.ghost_events, "%llu ghost events", (unsigned long long)mock.ghost_events);
    return errors;
}

#undef EXPECT

long qpu_test_sum(parsec_context_t *parsec, long local)
{
    int64_t in = (int64_t)local, out = (int64_t)local;
    if( PARSEC_SUCCESS != parsec_tests_allreduce(parsec, &in, &out, 1,
                                                 parsec_datatype_int64_t, PARSEC_TESTS_REDUCE_SUM) ) {
        out = in;
    }
    return (long)out;
}

void qpu_test_split_args(int argc, char **argv, int *test_argc, int *parsec_argc, char ***parsec_argv)
{
    int i;
    for( i = 1; i < argc; i++ ) {
        if( 0 == strcmp(argv[i], "--") ) break;
    }
    *test_argc   = i;
    *parsec_argc = argc - i;
    *parsec_argv = argv + i;
}
