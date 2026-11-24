/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * The block-decomposed QAOA for weighted MaxCut as one PaRSEC taskpool
 * (qaoa.jdf), and its comparison with the sequential reference.
 *
 * The quantum jobs go to the QPU device of the context, which the test
 * definitions set to the mock backend. Each run is compared with
 * qaoa_sequential_run() on the same instance, settings and seeds:
 *   - the final state of every block is the reference's, byte for byte;
 *   - so the final cut is, and it is the initial cut plus the improvements
 *     the updates reported, after every round and at the end;
 *   - the QPU device ran as many circuits as the reference asked of its
 *     sampler, in as many jobs when every task is its own job and in fewer
 *     when tasks share jobs (device_qpu_max_batch); it completed one task
 *     per evaluation, saw no tag twice and holds nothing at the end;
 *   - only the rank that owns the QPU opened a backend.
 *
 * With no instance on the command line, every fixed instance runs under
 * four settings, one taskpool after the other in the same context.
 *
 * Options, before "--":
 *   -i NAME        one of the fixed instances (see qaoa_graph.h)
 *   -g K:N:C:X     instead: K blocks of N vertices, C chords per block, X
 *                  edges between neighbouring blocks
 *   -p LAYERS  -T STEPS  -R ROUNDS  -s SHOTS  -P PATIENCE  -S SEED
 *   -F MODE        what happens to the jobs:
 *                    none   nothing (default)
 *                    keys   one evaluation in five fails, chosen by its key:
 *                           every job of it fails, the task completes with
 *                           a failure status, and the reference is given
 *                           the same list of failed evaluations
 *                    first  the first job of every evaluation fails and the
 *                           second one works (to be run with
 *                           --mca device_qpu_mock_faults fail:first=1): the
 *                           application sees nothing of it
 *   -C             tasks are to share jobs in this run (to be given with a
 *                  device_qpu_max_batch above one and fewer places at the
 *                  backend than tasks that wait): it is an error if none did
 *   -c 0           do not run the reference, compare nothing
 *   -v             one line per round
 *
 * The seed of the samples is the mock's (--mca device_qpu_mock_seed); -S is
 * the seed of the initial assignment and of the optimizer.
 */

#include "qaoa_core.h"
#include "qaoa_data.h"
#include "parsec/constants.h"
#include "tests/tests_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { FAIL_NONE, FAIL_KEYS, FAIL_FIRST };

static int verbose = 0;
static int fail_mode = FAIL_NONE;
static int compare = 1;
static int expect_shared = 0;
/* Over all the runs of this process, as rank 0 reads them in the reports. */
static long all_evaluations = 0, all_failed = 0;
/* Over all the runs of this process, on the rank that owns the QPU. */
static long all_jobs = 0, all_circuits = 0, all_shared_jobs = 0, all_shared_tasks = 0;
static unsigned most_in_a_job = 0;

/* ------------------------------------------------------------------------
 * Failing evaluations: the same ones in the mock and in the reference.
 * --------------------------------------------------------------------- */

static int job_fails(uint64_t key) { return 0 == parsec_qpu_mix64(key ^ 0xfa11fa11fa11fa11ULL) % 5; }

/* For the mock: every job of such an evaluation fails, whatever the attempt. */
static parsec_qpu_mock_fault_t fault_by_key(uint64_t tag, const parsec_qpu_request_t *request, uint64_t call, void *arg)
{
    (void)tag; (void)call; (void)arg;
    return job_fails(request->key) ? PARSEC_QPU_MOCK_FAIL : PARSEC_QPU_MOCK_OK;
}

/* For the reference: the evaluation gives no counts. */
static int sample_by_key(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                         uint64_t key, parsec_qpu_result_t *result, uint32_t max_entries, void *arg)
{
    if( job_fails(key) ) {
        result->status = PARSEC_QPU_STATUS_FAILED;
        return 0;
    }
    return qaoa_sample_direct(circuit, params, shots, key, result, max_entries, arg);
}

/* ------------------------------------------------------------------------
 * One run
 * --------------------------------------------------------------------- */

#define EXPECT(COND, ...)                                                   \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            fprintf(stderr, "testing_qaoa, rank %d, %s [%s]: ", rank, inst->name, setting); \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            errors++;                                                       \
        }                                                                   \
    } while(0)

/* Say where two states differ. */
static void describe_difference(const qaoa_state_t *got, const qaoa_state_t *want, char *text, size_t size)
{
    if( got->bits != want->bits ) {
        snprintf(text, size, "assignment %#llx, the reference has %#llx",
                 (unsigned long long)got->bits, (unsigned long long)want->bits);
    } else if( (got->version != want->version) || (got->stale != want->stale) ||
               (got->updates != want->updates) || (got->skipped != want->skipped) ) {
        snprintf(text, size, "version %u, stale %u, %u updates of which %u skipped; the reference has %u, %u, %u, %u",
                 got->version, got->stale, got->updates, got->skipped,
                 want->version, want->stale, want->updates, want->skipped);
    } else if( got->improvement != want->improvement ) {
        snprintf(text, size, "improvement %.6f, the reference has %.6f", got->improvement, want->improvement);
    } else if( 0 != memcmp(got->theta, want->theta, sizeof(got->theta)) ) {
        snprintf(text, size, "the same assignment and other angles");
    } else {
        snprintf(text, size, "the same assignment and other versions of its neighbours");
    }
}

static int run_one(parsec_context_t *parsec, int rank, int world, const qaoa_instance_t *inst,
                   const qaoa_config_t *cfg, const char *setting)
{
    const qaoa_partition_t *p = &inst->part;
    const int K = p->nb_blocks, R = cfg->rounds, T = cfg->steps;
    const long evaluations = 2L * T * R * K;
    parsec_data_collection_t *X, *Q, *A, *G, *E;
    parsec_qpu_device_info_t info;
    parsec_qpu_stats_t before, after;
    parsec_qpu_mock_stats_t mock_before, mock_after;
    parsec_taskpool_t *tp;
    qaoa_sizes_t sizes;
    qaoa_run_t ref;
    int64_t t0, t1;
    const int violations = qaoa_taskpool_violations();
    int errors = 0, rc, owner, have_ref = 0;

    /* Every rank has to see a QPU device, and one of them has to own it. A
     * rank that found otherwise must not leave the others to start a run
     * whose quantum tasks will never complete: they decide together. */
    {
        int64_t missing = (PARSEC_SUCCESS != parsec_qpu_device_info(0, &info)), anywhere;

        if( missing ) {
            fprintf(stderr, "testing_qaoa, rank %d: this context has no QPU device (--mca device_qpu_enabled 1, and a backend that opens)\n", rank);
        } else if( (info.owner_rank < 0) || (info.owner_rank >= world) ) {
            if( 0 == rank ) fprintf(stderr, "testing_qaoa: no rank owns the QPU: device_qpu_owner_rank is %d and there are %d rank(s)\n",
                                    info.owner_rank, world);
            missing = 1;
        }
        anywhere = missing;
        if( PARSEC_SUCCESS != parsec_tests_allreduce(parsec, &missing, &anywhere, 1, parsec_datatype_int64_t, PARSEC_TESTS_REDUCE_SUM) ) {
            anywhere = missing;
        }
        if( 0 != anywhere ) return 1;
    }
    owner = info.owner_rank;
    qaoa_sizes(&sizes, inst, cfg);

    /* The reference first, on every rank: it says what to expect of this
     * one, and it is the same computation everywhere. */
    if( compare ) {
        qaoa_direct_t direct = { NULL, (uint64_t)parsec_qpu_params.mock_seed };
        have_ref = (0 == qaoa_sequential_run(inst, cfg, (FAIL_KEYS == fail_mode) ? sample_by_key : qaoa_sample_direct,
                                             &direct, 0, &ref));
        parsec_qpu_sampler_free(direct.sampler);
        EXPECT(have_ref, "the sequential reference could not run");
        if( !have_ref ) return errors;
    }

    X = qaoa_tiles_new(rank, world, -1, K, sizes.state);        /* the state of block b, on rank b mod P */
    Q = qaoa_tiles_new(rank, world, owner, 1, 0);               /* where the quantum tasks run */
    A = qaoa_tiles_new(rank, world, 0, 1, sizes.all);           /* every final state, on rank 0 */
    G = qaoa_tiles_new(rank, world, 0, R, sizes.report);        /* what each round added up to, on rank 0 */
    E = qaoa_tiles_new(rank, world, -1, K, sizes.boundary);     /* an empty boundary per block, next to its state */
    qaoa_empty_boundaries(E, inst);

    tp = qaoa_taskpool_new(inst, cfg, X, Q, A, G, E, &sizes);
    if( NULL == tp ) {
        if( 0 == rank ) fprintf(stderr, "testing_qaoa: %d round(s) of %d step(s), %d layer(s) and %d shot(s) is not a run\n",
                                R, T, cfg->layers, cfg->shots);
        qaoa_tiles_free(X);
        qaoa_tiles_free(Q);
        qaoa_tiles_free(A);
        qaoa_tiles_free(G);
        qaoa_tiles_free(E);
        if( have_ref ) qaoa_run_free(&ref);
        return errors + 1;
    }

    parsec_qpu_service_stats(&before);
    parsec_qpu_mock_stats(&mock_before);
    if( FAIL_KEYS == fail_mode ) parsec_qpu_mock_set_fault_fn(fault_by_key, NULL);

    t0 = parsec_qpu_now_ns();
    rc = parsec_context_add_taskpool(parsec, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    t1 = parsec_qpu_now_ns();

    if( FAIL_KEYS == fail_mode ) parsec_qpu_mock_set_fault_fn(NULL, NULL);
    parsec_qpu_service_stats(&after);
    parsec_qpu_mock_stats(&mock_after);

    /* ---- the order, on every rank: each task that read a state read it
     * after the number of updates its dependencies say ---- */
    EXPECT(violations == qaoa_taskpool_violations(), "%d task(s) read the state of a block from another round than the graph says",
           qaoa_taskpool_violations() - violations);

    /* ---- the answer, on the rank that holds it ---- */
    if( 0 == rank ) {
        const qaoa_state_t *x = (const qaoa_state_t*)qaoa_tiles_ptr(A, 0);
        qaoa_report_t total;
        double cut_initial, cut_final, cut;
        int different = 0;

        memset(&total, 0, sizeof(total));
        for( int r = 0; r < R; r++ ) {
            const qaoa_report_t *round = (const qaoa_report_t*)qaoa_tiles_ptr(G, r);
            total.improvement += round->improvement;
            total.updates     += round->updates;
            total.accepted    += round->accepted;
            total.skipped     += round->skipped;
            total.evaluations += round->evaluations;
            total.failed      += round->failed;
        }
        cut_final = qaoa_cut(inst, x);
        {
            /* The initial cut, from the initial states. */
            qaoa_state_t *first = (qaoa_state_t*)calloc((size_t)K, sizeof(qaoa_state_t));
            for( int b = 0; b < K; b++ ) qaoa_state_init(&first[b], inst, cfg, b);
            cut_initial = qaoa_cut(inst, first);
            free(first);
        }
        /* Exact: weights are multiples of 1/8. */
        EXPECT(cut_final == cut_initial + total.improvement,
               "the final cut %.4f is not the initial cut %.4f plus the improvements the updates reported, %.4f",
               cut_final, cut_initial, total.improvement);
        EXPECT(total.updates == (uint32_t)(R * K), "%u updates reported for %d rounds of %d blocks", total.updates, R, K);
        EXPECT((long)total.evaluations + (long)total.failed + 2L * T * total.skipped == evaluations,
               "%u evaluations gave counts, %u failed and %u updates were skipped, of %ld evaluations",
               total.evaluations, total.failed, total.skipped, evaluations);

        if( have_ref ) {
            for( int b = 0; b < K; b++ ) {
                if( 0 != memcmp(&x[b], &ref.x[b], sizeof(qaoa_state_t)) ) {
                    char text[256];
                    describe_difference(&x[b], &ref.x[b], text, sizeof(text));
                    if( 0 == different ) EXPECT(0, "block %d ends with %s", b, text);
                    different++;
                }
            }
            EXPECT(0 == different, "%d of %d blocks do not end in the state of the sequential reference", different, K);
            EXPECT(cut_final == ref.cut_final, "final cut %.4f, the reference has %.4f", cut_final, ref.cut_final);
            cut = cut_initial;
            for( int r = 0; r < R; r++ ) {
                cut += ((const qaoa_report_t*)qaoa_tiles_ptr(G, r))->improvement;
                EXPECT(cut == ref.cut_after_round[r], "after round %d the updates add up to a cut of %.4f, the reference has %.4f",
                       r, cut, ref.cut_after_round[r]);
                if( verbose ) printf("  %s [%s] round %d: cut %.4f\n", inst->name, setting, r, cut);
            }
            EXPECT((long)total.accepted == ref.accepted, "%u updates accepted, the reference has %ld", total.accepted, ref.accepted);
            EXPECT((long)total.skipped == ref.skipped, "%u updates skipped, the reference has %ld", total.skipped, ref.skipped);
            EXPECT((long)total.evaluations + (long)total.failed == ref.jobs, "%u evaluations asked for a job, the reference asked for %ld",
                   total.evaluations + total.failed, ref.jobs);
            EXPECT((long)total.failed == ref.failed_jobs, "%u evaluations failed, the reference has %ld", total.failed, ref.failed_jobs);
        }
        /* This line is about the states and nothing else: what the QPU did and
         * what the other ranks saw is checked below and summed at the end. */
        printf("%-12s %-22s cut %8.4f -> %8.4f  %4u jobs asked for, %3u/%-3u updates accepted, %2u skipped, %7.1f ms on %d rank(s)%s\n",
               inst->name, setting, cut_initial, cut_final, total.evaluations + total.failed, total.accepted, total.updates,
               total.skipped, (t1 - t0) * 1e-6, world,
               !have_ref ? "" : (different ? ": final states NOT those of the reference"
                                           : ": final states equal to the reference, bit for bit"));
        all_evaluations += (long)total.evaluations + (long)total.failed;
        all_failed      += (long)total.failed;
    }

    /* ---- the QPU, on the rank that owns it and on the others ---- */
    if( rank == owner ) {
        const long tasks   = (long)(after.tasks_completed - before.tasks_completed);
        const long jobs    = (long)(after.jobs_submitted - before.jobs_submitted);
        const long empties = (long)(after.empty_jobs - before.empty_jobs);
        const long failed  = (long)(after.tasks_failed - before.tasks_failed);
        const long retries = (long)(after.retries - before.retries);
        /* Jobs that carried the circuits of several tasks, and those tasks.
         * A job of one task is one circuit. */
        const long shared_jobs  = (long)(after.jobs_coalesced - before.jobs_coalesced);
        const long shared_tasks = (long)(after.tasks_coalesced - before.tasks_coalesced);
        const long circuits     = jobs - shared_jobs + shared_tasks;
        const long limit        = (long)info.max_batch;

        EXPECT(tasks == evaluations, "the QPU completed %ld tasks, one per evaluation makes %ld", tasks, evaluations);
        EXPECT((long)(after.tasks_submitted - before.tasks_submitted) == evaluations, "%ld quantum tasks submitted, expected %ld",
               (long)(after.tasks_submitted - before.tasks_submitted), evaluations);
        EXPECT((0 == after.pending) && (0 == after.in_flight), "the QPU still holds %u pending and %u jobs in flight",
               after.pending, after.in_flight);
        EXPECT(parsec_qpu_device_info(0, &info) == PARSEC_SUCCESS && (0 == info.load_ns),
               "the QPU device still has a load of %lld ns", (long long)info.load_ns);
        EXPECT(mock_after.duplicate_tags == mock_before.duplicate_tags, "a job was asked for twice under one tag");
        EXPECT(mock_after.key_overlaps == mock_before.key_overlaps, "two jobs of one evaluation were held at once");
        EXPECT(mock_after.inputs_changed == mock_before.inputs_changed, "the inputs of a job changed while the backend held it");
        EXPECT((mock_after.mixed_shots == mock_before.mixed_shots) && (mock_after.changed_jobs == mock_before.changed_jobs),
               "a job carried circuits that ask for different numbers of shots, or was asked for again with other circuits");
        EXPECT((long)(mock_after.submits - mock_before.submits) == jobs, "the backend created %ld jobs, the device counts %ld",
               (long)(mock_after.submits - mock_before.submits), jobs);
        EXPECT((long)(mock_after.circuits - mock_before.circuits) == circuits,
               "the backend was given %ld circuits; the device counts %ld jobs, of which %ld carried %ld tasks",
               (long)(mock_after.circuits - mock_before.circuits), jobs, shared_jobs, shared_tasks);
        EXPECT((limit >= 1) && ((long)after.max_job_tasks <= limit) && ((long)mock_after.max_circuits <= limit),
               "a job carried %u tasks and the backend was given one of %u circuits; the limit is %ld",
               after.max_job_tasks, mock_after.max_circuits, limit);
        EXPECT((limit > 1) || (0 == shared_jobs), "%ld jobs carried more than one task, and no job may", shared_jobs);
        EXPECT(0 == (after.stray_events - before.stray_events), "%ld final states came for jobs that were not in flight",
               (long)(after.stray_events - before.stray_events));
        all_jobs         += jobs;
        all_circuits     += circuits;
        all_shared_jobs  += shared_jobs;
        all_shared_tasks += shared_tasks;
        if( after.max_job_tasks > most_in_a_job ) most_in_a_job = after.max_job_tasks;
        if( have_ref ) {
            const long skipped_evaluations = 2L * T * ref.skipped;
            const int  attempts = 1 + ((parsec_qpu_params.max_retries > 0) ? parsec_qpu_params.max_retries : 0);
            /* What the evaluations take when none shares a job that fails
             * because of another: one circuit each, and every job it may
             * have for one that fails. */
            const long alone = (ref.jobs - ref.failed_jobs) + (long)attempts * ref.failed_jobs;

            EXPECT(empties == skipped_evaluations, "%ld tasks needed no job, the %ld skipped updates make %ld",
                   empties, ref.skipped, skipped_evaluations);
            switch( fail_mode ) {
            case FAIL_NONE:
                /* One circuit per evaluation, in however many jobs. */
                EXPECT(circuits == ref.jobs, "the QPU ran %ld circuits in %ld jobs, the reference asked its sampler for %ld",
                       circuits, jobs, ref.jobs);
                EXPECT((0 == failed) && (0 == retries) && (after.jobs_failed == before.jobs_failed),
                       "%ld tasks ended without a result and %ld jobs were retried with nothing to make them", failed, retries);
                break;
            case FAIL_KEYS:
                /* An evaluation that fails takes every job it may have. The
                 * first of them may carry other evaluations, which then
                 * take one more job each, alone: at most as many as a job
                 * has room for besides the one that fails. */
                EXPECT(failed == ref.failed_jobs, "%ld tasks ended without a result, the reference has %ld failed evaluations",
                       failed, ref.failed_jobs);
                EXPECT((circuits >= alone) && (circuits <= alone + (limit - 1) * ref.failed_jobs),
                       "the QPU ran %ld circuits; %ld evaluations that work and %ld that fail %d times make %ld, and at most "
                       "%ld more for those that shared a job with one that fails",
                       circuits, ref.jobs - ref.failed_jobs, ref.failed_jobs, attempts, alone, (limit - 1) * ref.failed_jobs);
                break;
            case FAIL_FIRST:
                /* Every evaluation took two jobs, the second one alone, and
                 * none failed for the application. */
                EXPECT(0 == failed, "%ld tasks ended without a result", failed);
                EXPECT((circuits == 2 * ref.jobs) && (retries == ref.jobs),
                       "the QPU ran %ld circuits of which %ld in second jobs, for %ld evaluations", circuits, retries, ref.jobs);
                break;
            }
        }
    } else {
        /* A rank that does not own the QPU has no backend and took no task. */
        EXPECT(0 == after.backend_open, "a backend is open on a rank that does not own the QPU");
        EXPECT(0 == after.tasks_submitted, "%llu quantum tasks were taken on a rank that does not own the QPU",
               (unsigned long long)after.tasks_submitted);
        EXPECT(0 == mock_after.opens, "a backend was opened on a rank that does not own the QPU");
    }

    qaoa_taskpool_free(tp);
    qaoa_tiles_free(X);
    qaoa_tiles_free(Q);
    qaoa_tiles_free(A);
    qaoa_tiles_free(G);
    qaoa_tiles_free(E);
    if( have_ref ) qaoa_run_free(&ref);
    return errors;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    qaoa_config_t cfg;
    qaoa_instance_t inst;
    const char *name = NULL, *generated = NULL;
    int rank = 0, world = 1, rc, pargc = 0, test_argc = argc, chosen = 0;
    int64_t in, out;
    char **pargv = NULL;
    long errors = 0;

    for( int i = 1; i < argc; i++ ) {
        if( 0 == strcmp(argv[i], "--") ) {
            test_argc = i;
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }
    qaoa_config_default(&cfg);
    for( int i = 1; i < test_argc; i++ ) {
        if( 0 == strcmp(argv[i], "-v") ) { verbose = 1; continue; }
        if( 0 == strcmp(argv[i], "-C") ) { expect_shared = 1; continue; }
        if( i + 1 >= test_argc ) { fprintf(stderr, "testing_qaoa: %s needs a value\n", argv[i]); return 1; }
        if(      0 == strcmp(argv[i], "-i") ) { name = argv[++i]; chosen = 1; }
        else if( 0 == strcmp(argv[i], "-g") ) { generated = argv[++i]; chosen = 1; }
        else if( 0 == strcmp(argv[i], "-p") ) { cfg.layers = atoi(argv[++i]); chosen = 1; }
        else if( 0 == strcmp(argv[i], "-T") ) { cfg.steps = atoi(argv[++i]); chosen = 1; }
        else if( 0 == strcmp(argv[i], "-R") ) { cfg.rounds = atoi(argv[++i]); chosen = 1; }
        else if( 0 == strcmp(argv[i], "-s") ) { cfg.shots = atoi(argv[++i]); chosen = 1; }
        else if( 0 == strcmp(argv[i], "-P") ) { cfg.patience = atoi(argv[++i]); chosen = 1; }
        else if( 0 == strcmp(argv[i], "-S") ) { cfg.seed = strtoull(argv[++i], NULL, 0); chosen = 1; }
        else if( 0 == strcmp(argv[i], "-c") ) compare = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-F") ) {
            const char *mode = argv[++i];
            if(      0 == strcmp(mode, "none") )  fail_mode = FAIL_NONE;
            else if( 0 == strcmp(mode, "keys") )  fail_mode = FAIL_KEYS;
            else if( 0 == strcmp(mode, "first") ) fail_mode = FAIL_FIRST;
            else { fprintf(stderr, "testing_qaoa: -F takes none, keys or first\n"); return 1; }
        }
        else { fprintf(stderr, "testing_qaoa: unknown option %s\n", argv[i]); return 1; }
    }

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED, &pargc, &pargv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    if( chosen ) {
        if( NULL != generated ) {
            int k = 0, n = 0, c = 0, x = 0;
            if( (4 != sscanf(generated, "%d:%d:%d:%d", &k, &n, &c, &x)) || (0 != qaoa_instance_blocks(&inst, k, n, c, x, cfg.seed)) ) {
                fprintf(stderr, "testing_qaoa: -g takes blocks:vertices:chords:cross, and %s is not a graph that exists\n", generated);
                parsec_tests_abort(parsec, 1);
            }
        } else if( 0 != qaoa_instance_named(&inst, (NULL != name) ? name : "ring64") ) {
            fprintf(stderr, "testing_qaoa: no instance named %s\n", name);
            parsec_tests_abort(parsec, 1);
        }
        errors += run_one(parsec, rank, world, &inst, &cfg, "as asked");
        qaoa_instance_free(&inst);
    } else {
        /* Every instance under the four settings of the reference program. */
        static const struct { const char *name; int layers, steps, rounds, shots, patience; uint64_t seed; } settings[] = {
            { "p=1 T=4 R=3",          1, 4, 3, 256, 0, 1 },
            { "p=2 T=3 R=2",          2, 3, 2, 128, 0, 2 },
            { "p=1 T=2 R=6 patient",  1, 2, 6, 100, 2, 3 },
            { "p=1 T=1 R=1 10 shots", 1, 1, 1,  10, 0, 4 },
        };
        for( int i = 0; NULL != qaoa_instance_names[i]; i++ ) {
            if( 0 != qaoa_instance_named(&inst, qaoa_instance_names[i]) ) {
                fprintf(stderr, "testing_qaoa: %s could not be built\n", qaoa_instance_names[i]);
                errors++;
                continue;
            }
            for( size_t s = 0; s < sizeof(settings) / sizeof(settings[0]); s++ ) {
                qaoa_config_default(&cfg);
                cfg.layers   = settings[s].layers;
                cfg.steps    = settings[s].steps;
                cfg.rounds   = settings[s].rounds;
                cfg.shots    = settings[s].shots;
                cfg.patience = settings[s].patience;
                cfg.seed     = settings[s].seed;
                errors += run_one(parsec, rank, world, &inst, &cfg, settings[s].name);
            }
            qaoa_instance_free(&inst);
        }
    }

    /* The runs that were to lose one evaluation in five did lose some. */
    if( (0 == rank) && (FAIL_KEYS == fail_mode) && !chosen ) {
        if( (all_failed * 10 < all_evaluations) || (all_failed * 3 > all_evaluations) ) {
            fprintf(stderr, "testing_qaoa: %ld of %ld evaluations failed in runs that were to lose one in five\n", all_failed, all_evaluations);
            errors++;
        }
        printf("with one evaluation in five failing for good: %ld of %ld failed\n", all_failed, all_evaluations);
    }
    if( (0 == rank) && (FAIL_KEYS != fail_mode) && (0 != all_failed) ) {
        fprintf(stderr, "testing_qaoa: %ld evaluations failed with nothing to make them\n", all_failed);
        errors++;
    }
    /* What the QPU made of the evaluations, on the rank that owns it. */
    if( all_jobs > 0 ) {
        if( expect_shared || verbose || (all_shared_jobs > 0) ) {
            printf("the QPU ran %ld circuits in %ld jobs: %ld jobs carried %ld tasks together, at most %u in one\n",
                   all_circuits, all_jobs, all_shared_jobs, all_shared_tasks, most_in_a_job);
        }
        if( expect_shared && (0 == all_shared_tasks) ) {
            fprintf(stderr, "testing_qaoa: tasks were to share jobs in this run, and every job carried one\n");
            errors++;
        }
    }

    in = out = (int64_t)errors;
    if( PARSEC_SUCCESS != parsec_tests_allreduce(parsec, &in, &out, 1, parsec_datatype_int64_t, PARSEC_TESTS_REDUCE_SUM) ) {
        out = in;
    }
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    if( 0 == rank ) printf("testing_qaoa: %s\n", out ? "FAILED" : "passed");
    return out ? 1 : 0;
}
