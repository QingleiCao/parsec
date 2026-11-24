/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * The sequential reference of the block-decomposed QAOA, without the
 * runtime.
 *
 * With no option it runs every fixed instance under several settings and
 * checks, each time:
 *   - after every update of a block, the cut of the whole graph is the one
 *     before plus the improvement the block reports, and did not go down;
 *   - the final cut, recomputed from the assignments, is the initial one
 *     plus the sum of those improvements, exactly;
 *   - it does not exceed the largest cut of the graph, found by
 *     enumeration (graphs of at most 24 vertices);
 *   - no block ends above the best it could do alone;
 *   - the same settings give the same bits again, and the same bits when
 *     every job goes through the mock backend instead of the sampler;
 *   - the number of jobs is two per step of every update that was not
 *     skipped;
 *   - with some jobs failing, all of the above still holds.
 *
 * Options:
 *   -i NAME        one of the fixed instances (see qaoa_graph.h)
 *   -g K:N:C:X     instead: K blocks of N vertices, C chords per block, X
 *                  edges between neighbouring blocks
 *   -p LAYERS  -T STEPS  -R ROUNDS  -s SHOTS  -P PATIENCE  -S SEED
 *   -b             every job through the mock backend, one at a time, each
 *                  taking the time the mock is set to: a blocking quantum
 *                  call. This is the sequential baseline.
 *   -d SPEC        the mock's delay for -b (default const:0)
 *   -t             print the time spent in each kind of step
 *   -v             print one line per round
 */

#include "qaoa_core.h"
#include "parsec/constants.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nb_errors = 0;
static int verbose = 0;
static long total_failed = 0, total_asked = 0;   /* over all the runs with failing jobs */

#define EXPECT(COND, ...)                                                   \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            fprintf(stderr, "%s [%s]: ", label, setting);                   \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            nb_errors++;                                                    \
        }                                                                   \
    } while(0)

/* A sampler that fails one job in five, chosen by the key. */
typedef struct { qaoa_direct_t direct; long failed; } failing_t;

static int job_fails(uint64_t key) { return 0 == parsec_qpu_mix64(key ^ 0xfa11fa11fa11fa11ULL) % 5; }

static int sample_failing(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                          uint64_t key, parsec_qpu_result_t *result, uint32_t max_entries, void *arg)
{
    failing_t *f = (failing_t*)arg;
    if( job_fails(key) ) {
        result->status = PARSEC_QPU_STATUS_FAILED;
        f->failed++;
        return 0;
    }
    return qaoa_sample_direct(circuit, params, shots, key, result, max_entries, &f->direct);
}

static parsec_qpu_backend_t *mock_open(const char *delay, uint64_t seed)
{
    parsec_qpu_backend_t *backend = NULL;

    /* What parsec_init() would have read from the MCA parameters. */
    parsec_qpu_params.mock_seed        = (int)seed;
    parsec_qpu_params.mock_slots       = 1;
    parsec_qpu_params.mock_delay       = (char*)delay;
    parsec_qpu_params.mock_queue_delay = (char*)"const:0";
    parsec_qpu_params.mock_faults      = (char*)"none";
    parsec_qpu_params.mock_threads     = 1;
    parsec_qpu_params.mock_max_qubits  = 24;
    if( (PARSEC_SUCCESS != parsec_qpu_backend_mock_ops.open(&backend, 0)) || (NULL == backend) ) return NULL;
    return backend;
}

static int same_states(const qaoa_run_t *a, const qaoa_run_t *b)
{
    return (a->nb_blocks == b->nb_blocks) && (0 == memcmp(a->x, b->x, (size_t)a->nb_blocks * sizeof(qaoa_state_t)));
}

/* Run one instance under one setting and check everything there is to check. */
static void check_instance(const qaoa_instance_t *inst, const qaoa_config_t *cfg, const char *setting, int through_mock)
{
    const char *label = inst->name;
    const qaoa_partition_t *p = &inst->part;
    qaoa_direct_t direct = { NULL, cfg->seed };
    qaoa_run_t run, again;
    double optimum = -1.0;
    int at_local_best = 0;

    if( 0 != qaoa_sequential_run(inst, cfg, qaoa_sample_direct, &direct, 1, &run) ) {
        EXPECT(0, "the reference could not run");
        return;
    }
    EXPECT(0 == run.violations, "after %d update(s) the cut was not the one before plus the improvement, or went down", run.violations);
    EXPECT(run.cut_final == run.cut_initial + run.improvement, "final cut %.4f is not the initial %.4f plus the improvements %.4f",
           run.cut_final, run.cut_initial, run.improvement);
    for( int r = 0; r < cfg->rounds; r++ ) {
        const double before = (0 == r) ? run.cut_initial : run.cut_after_round[r - 1];
        EXPECT(run.cut_after_round[r] >= before, "round %d lowered the cut from %.4f to %.4f", r, before, run.cut_after_round[r]);
        if( verbose ) printf("  %s [%s] round %d: cut %.4f\n", label, setting, r, run.cut_after_round[r]);
    }
    EXPECT(run.updates == (long)cfg->rounds * p->nb_blocks, "%ld updates for %d rounds of %d blocks", run.updates, cfg->rounds, p->nb_blocks);
    EXPECT(run.jobs == 2L * cfg->steps * (run.updates - run.skipped), "%ld jobs for %ld updates of which %ld skipped, %d steps each",
           run.jobs, run.updates, run.skipped, cfg->steps);
    EXPECT(0 == run.failed_jobs, "%ld jobs failed with nothing to make them", run.failed_jobs);
    EXPECT((0 == cfg->patience) ? (0 == run.skipped) : 1, "%ld updates skipped with skipping off", run.skipped);

    if( inst->graph.n <= 24 ) {
        optimum = qaoa_exact_maxcut(&inst->graph, NULL);
        EXPECT(run.cut_final <= optimum, "final cut %.4f is above the largest cut %.4f", run.cut_final, optimum);
    }

    /* No block is above the best it could do alone with its neighbours as
     * they ended; count those that are at it. */
    {
        qaoa_boundary_t *h = (qaoa_boundary_t*)malloc(qaoa_boundary_size(p->max_block));
        qaoa_local_t *P = (qaoa_local_t*)malloc(qaoa_local_size(p->max_block, p->max_internal));
        double total = 0.0;

        for( int b = 0; b < p->nb_blocks; b++ ) {
            double best;
            qaoa_boundary_init(h, inst, b);
            for( int k = 0; k < p->deg[b]; k++ ) qaoa_boundary_gather(h, inst, b, k, &run.x[qaoa_block_nbr(p, b, k)]);
            qaoa_local_build(P, inst, cfg, cfg->rounds, b, &run.x[b], h);
            best = qaoa_exact_local(P, NULL);
            EXPECT(P->current <= best, "block %d ends at %.4f, above the best it could do alone, %.4f", b, P->current, best);
            at_local_best += (P->current == best);
            total += run.x[b].improvement;
            EXPECT(run.x[b].updates == (uint32_t)cfg->rounds, "block %d was updated %u times in %d rounds", b, run.x[b].updates, cfg->rounds);
        }
        EXPECT(total == run.improvement, "the blocks' improvements add up to %.4f, the run's to %.4f", total, run.improvement);
        free(h);
        free(P);
    }

    /* The same again. */
    {
        qaoa_direct_t second = { NULL, cfg->seed };
        if( 0 == qaoa_sequential_run(inst, cfg, qaoa_sample_direct, &second, 0, &again) ) {
            EXPECT(same_states(&run, &again) && (run.jobs == again.jobs), "the same settings gave other bits the second time");
            qaoa_run_free(&again);
        } else {
            EXPECT(0, "the reference could not run a second time");
        }
        parsec_qpu_sampler_free(second.sampler);
    }

    /* The same with every job through the mock backend: its counts are the
     * sampler's for the same seed and key. */
    if( through_mock ) {
        parsec_qpu_backend_t *backend = mock_open("const:0", cfg->seed);
        if( NULL == backend ) {
            EXPECT(0, "the mock backend did not open");
        } else {
            if( 0 == qaoa_sequential_run(inst, cfg, qaoa_sample_blocking, backend, 0, &again) ) {
                EXPECT(same_states(&run, &again), "the mock backend gave other bits than the sampler");
                EXPECT((run.jobs == again.jobs) && (0 == again.failed_jobs), "through the mock: %ld jobs, %ld failed; direct: %ld jobs",
                       again.jobs, again.failed_jobs, run.jobs);
                qaoa_run_free(&again);
            } else {
                EXPECT(0, "the reference could not run through the mock backend");
            }
            (void)parsec_qpu_backend_mock_ops.close(backend);
        }
    }

    /* With one job in five failing: a failed job gives no candidate and
     * moves nothing, and everything exact above still holds. */
    {
        failing_t failing = { { NULL, cfg->seed }, 0 };
        if( 0 == qaoa_sequential_run(inst, cfg, sample_failing, &failing, 1, &again) ) {
            EXPECT(0 == again.violations, "with failing jobs: %d update(s) broke the cut's accounting", again.violations);
            EXPECT(again.cut_final == again.cut_initial + again.improvement, "with failing jobs: final cut %.4f is not %.4f plus %.4f",
                   again.cut_final, again.cut_initial, again.improvement);
            EXPECT(again.failed_jobs == failing.failed, "%ld jobs failed, %ld were counted", failing.failed, again.failed_jobs);
            /* As many jobs were asked for, unless blocks may be skipped: which ones are depends on what was measured. */
            EXPECT((cfg->patience > 0) || (again.jobs == run.jobs), "with failing jobs: %ld jobs, %ld without", again.jobs, run.jobs);
            total_failed += again.failed_jobs;
            total_asked  += again.jobs;
            EXPECT((optimum < 0.0) || (again.cut_final <= optimum), "with failing jobs: final cut %.4f is above the largest cut %.4f", again.cut_final, optimum);
            qaoa_run_free(&again);
        } else {
            EXPECT(0, "the reference could not run with failing jobs");
        }
        parsec_qpu_sampler_free(failing.direct.sampler);
    }

    printf("%-12s %-22s cut %8.4f -> %8.4f", label, setting, run.cut_initial, run.cut_final);
    if( optimum > 0.0 ) printf("  (best %8.4f, ratio %.3f)", optimum, run.cut_final / optimum);
    else if( optimum == 0.0 ) printf("  (best %8.4f)", optimum);
    else printf("  (best not enumerated)        ");
    printf("  %4ld jobs, %3ld/%-3ld updates accepted, %2ld skipped, %d/%d blocks at their own best\n",
           run.jobs, run.accepted, run.updates, run.skipped, at_local_best, p->nb_blocks);

    parsec_qpu_sampler_free(direct.sampler);
    qaoa_run_free(&run);
}

/* One run, as asked on the command line. */
static int single_run(const qaoa_instance_t *inst, const qaoa_config_t *cfg, int blocking, const char *delay, int timers)
{
    qaoa_direct_t direct = { NULL, cfg->seed };
    parsec_qpu_backend_t *backend = NULL;
    qaoa_run_t run;
    int64_t t0, t1;
    int rc;

    if( blocking && (NULL == (backend = mock_open(delay, cfg->seed))) ) {
        fprintf(stderr, "qaoa_reference: the mock backend did not open with delay %s\n", delay);
        return 1;
    }
    t0 = parsec_qpu_now_ns();
    rc = blocking ? qaoa_sequential_run(inst, cfg, qaoa_sample_blocking, backend, 0, &run)
                  : qaoa_sequential_run(inst, cfg, qaoa_sample_direct, &direct, 0, &run);
    t1 = parsec_qpu_now_ns();
    if( NULL != backend ) (void)parsec_qpu_backend_mock_ops.close(backend);
    parsec_qpu_sampler_free(direct.sampler);
    if( 0 != rc ) {
        fprintf(stderr, "qaoa_reference: the run failed\n");
        return 1;
    }

    printf("%s: %d vertices, %d edges, %d blocks of at most %d, %d colours; %d layer(s), %d steps, %d rounds, %d shots, seed %llu%s\n",
           inst->name, inst->graph.n, inst->graph.m, inst->part.nb_blocks, inst->part.max_block, inst->part.nb_colours,
           cfg->layers, cfg->steps, cfg->rounds, cfg->shots, (unsigned long long)cfg->seed,
           blocking ? ", every job through the mock backend" : "");
    printf("cut %.4f -> %.4f in %.3f ms; %ld jobs (%ld failed), %ld of %ld updates accepted, %ld skipped\n",
           run.cut_initial, run.cut_final, (t1 - t0) * 1e-6, run.jobs, run.failed_jobs, run.accepted, run.updates, run.skipped);
    if( verbose ) {
        for( int r = 0; r < cfg->rounds; r++ ) printf("  round %d: cut %.4f\n", r, run.cut_after_round[r]);
    }
    if( timers ) {
        const qaoa_partition_t *p = &inst->part;

        for( int k = 0; k < QAOA_NB_TIMERS; k++ ) {
            printf("  %-42s %7ld times, %10.3f us each, %10.3f ms in all\n", qaoa_timer_names[k], run.calls[k],
                   (run.calls[k] > 0) ? (double)run.ns[k] / (double)run.calls[k] * 1e-3 : 0.0, (double)run.ns[k] * 1e-6);
        }
        /* What the enumerations cost, where they can be done. */
        if( p->max_block <= 24 ) {
            qaoa_boundary_t *h = (qaoa_boundary_t*)malloc(qaoa_boundary_size(p->max_block));
            qaoa_local_t *P = (qaoa_local_t*)malloc(qaoa_local_size(p->max_block, p->max_internal));
            int64_t ns = 0;
            int at_best = 0;

            for( int b = 0; b < p->nb_blocks; b++ ) {
                int64_t t;
                qaoa_boundary_init(h, inst, b);
                for( int k = 0; k < p->deg[b]; k++ ) qaoa_boundary_gather(h, inst, b, k, &run.x[qaoa_block_nbr(p, b, k)]);
                qaoa_local_build(P, inst, cfg, cfg->rounds, b, &run.x[b], h);
                t = parsec_qpu_now_ns();
                at_best += (qaoa_exact_local(P, NULL) == P->current);
                ns += parsec_qpu_now_ns() - t;
            }
            printf("  %-42s %7d times, %10.3f us each; %d of %d blocks ended at it\n", "the best a block can do alone, by enumeration",
                   p->nb_blocks, (double)ns / p->nb_blocks * 1e-3, at_best, p->nb_blocks);
            free(h);
            free(P);
        }
        if( inst->graph.n <= 24 ) {
            const int64_t t = parsec_qpu_now_ns();
            const double optimum = qaoa_exact_maxcut(&inst->graph, NULL);
            printf("  %-42s %7d time,  %10.3f ms; it is %.4f, the run ended at %.4f\n", "the largest cut, by enumeration", 1,
                   (parsec_qpu_now_ns() - t) * 1e-6, optimum, run.cut_final);
        }
    }
    if( run.cut_final != run.cut_initial + run.improvement ) {
        fprintf(stderr, "qaoa_reference: final cut %.4f is not the initial %.4f plus the improvements %.4f\n",
                run.cut_final, run.cut_initial, run.improvement);
        rc = 1;
    }
    qaoa_run_free(&run);
    return rc;
}

int main(int argc, char *argv[])
{
    qaoa_config_t cfg;
    qaoa_instance_t inst;
    const char *name = NULL, *generated = NULL, *delay = "const:0";
    int blocking = 0, timers = 0, chosen = 0;

    qaoa_config_default(&cfg);
    for( int i = 1; i < argc; i++ ) {
        if( 0 == strcmp(argv[i], "-b") ) { blocking = 1; chosen = 1; continue; }
        if( 0 == strcmp(argv[i], "-t") ) { timers = 1; chosen = 1; continue; }
        if( 0 == strcmp(argv[i], "-v") ) { verbose = 1; continue; }
        if( i + 1 >= argc ) { fprintf(stderr, "qaoa_reference: %s needs a value\n", argv[i]); return 1; }
        chosen = 1;
        if(      0 == strcmp(argv[i], "-i") ) name = argv[++i];
        else if( 0 == strcmp(argv[i], "-g") ) generated = argv[++i];
        else if( 0 == strcmp(argv[i], "-p") ) cfg.layers = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-T") ) cfg.steps = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-R") ) cfg.rounds = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-s") ) cfg.shots = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-P") ) cfg.patience = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-S") ) cfg.seed = strtoull(argv[++i], NULL, 0);
        else if( 0 == strcmp(argv[i], "-d") ) delay = argv[++i];
        else { fprintf(stderr, "qaoa_reference: unknown option %s\n", argv[i]); return 1; }
    }

    if( chosen ) {
        int rc;
        if( NULL != generated ) {
            int k = 0, n = 0, c = 0, x = 0;
            if( (4 != sscanf(generated, "%d:%d:%d:%d", &k, &n, &c, &x)) || (0 != qaoa_instance_blocks(&inst, k, n, c, x, cfg.seed)) ) {
                fprintf(stderr, "qaoa_reference: -g takes blocks:vertices:chords:cross, and %s is not a graph that exists\n", generated);
                return 1;
            }
        } else if( 0 != qaoa_instance_named(&inst, (NULL != name) ? name : "ring64") ) {
            fprintf(stderr, "qaoa_reference: no instance named %s\n", name);
            return 1;
        }
        rc = single_run(&inst, &cfg, blocking, delay, timers);
        qaoa_instance_free(&inst);
        return rc;
    }

    /* Every instance, under four settings. */
    for( int i = 0; NULL != qaoa_instance_names[i]; i++ ) {
        static const struct { const char *name; int layers, steps, rounds, shots, patience; uint64_t seed; } settings[] = {
            { "p=1 T=4 R=3",          1, 4, 3, 256, 0, 1 },
            { "p=2 T=3 R=2",          2, 3, 2, 128, 0, 2 },
            { "p=1 T=2 R=6 patient",  1, 2, 6, 100, 2, 3 },
            { "p=1 T=1 R=1 10 shots", 1, 1, 1,  10, 0, 4 },
        };
        if( 0 != qaoa_instance_named(&inst, qaoa_instance_names[i]) ) {
            fprintf(stderr, "%s could not be built\n", qaoa_instance_names[i]);
            nb_errors++;
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
            /* Through the mock backend once per instance: it starts and stops threads each time. */
            check_instance(&inst, &cfg, settings[s].name, 0 == s);
        }
        qaoa_instance_free(&inst);
    }

    /* The runs with failing jobs did have failing jobs: about one in five. */
    if( (total_failed * 10 < total_asked) || (total_failed * 3 > total_asked) ) {
        fprintf(stderr, "%ld of %ld jobs failed in the runs that were to lose one in five\n", total_failed, total_asked);
        nb_errors++;
    }
    printf("with one job in five failing: %ld of %ld jobs failed, and every check held\n", total_failed, total_asked);

    printf("qaoa_reference: %s\n", nb_errors ? "FAILED" : "passed");
    return nb_errors ? 1 : 0;
}
