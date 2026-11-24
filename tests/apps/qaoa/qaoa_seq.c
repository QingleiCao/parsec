/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "qaoa_core.h"
#include "parsec/constants.h"

#include <stdlib.h>
#include <string.h>

/**
 * The sequential reference: the algorithm in the order (round, colour,
 * block, step, evaluation), one job at a time.
 *
 * In that order a block that is updated sees its neighbours of a smaller
 * colour as this round left them and the others as the round before did,
 * which is the rule the task graph of the distributed application writes as
 * dependencies. Since every function called here is a pure function of the
 * states it is given, and every job is named by its key, any execution that
 * respects those dependencies ends with the same bits.
 */

const char *const qaoa_timer_names[QAOA_NB_TIMERS] = {
    "gather the boundary",
    "build the local problem and its circuit",
    "run one job",
    "evaluate the counts of one job",
    "one optimizer step",
    "update the block"
};

#define TIMED(RUN, WHICH, CODE)                                 \
    do {                                                        \
        const int64_t t_begin_ = parsec_qpu_now_ns();           \
        CODE;                                                   \
        (RUN)->ns[WHICH] += parsec_qpu_now_ns() - t_begin_;     \
        (RUN)->calls[WHICH]++;                                  \
    } while(0)

int qaoa_sequential_run(const qaoa_instance_t *inst, const qaoa_config_t *cfg,
                        qaoa_sample_fn_t sample, void *arg, int check, qaoa_run_t *run)
{
    const qaoa_partition_t *p = &inst->part;
    const int K = p->nb_blocks;
    const uint32_t max_entries = qaoa_result_entries(cfg, p->max_block);
    const size_t circuit_room = qaoa_circuit_size(p->max_block, p->max_internal, cfg->layers);
    qaoa_boundary_t *h = NULL;
    qaoa_local_t *P = NULL;
    parsec_qpu_circuit_t *circuit = NULL;
    parsec_qpu_result_t *result[2] = { NULL, NULL };
    double theta[2][2 * QAOA_MAX_LAYERS];
    double cut;
    int rc = -1;

    memset(run, 0, sizeof(*run));
    if( (cfg->layers < 1) || (cfg->layers > QAOA_MAX_LAYERS) || (cfg->steps < 0) || (cfg->rounds < 0) || (cfg->shots < 1) ) {
        return -1;
    }
    run->nb_blocks       = K;
    run->x               = (qaoa_state_t*)calloc((size_t)K, sizeof(qaoa_state_t));
    run->cut_after_round = (double*)calloc((size_t)(cfg->rounds > 0 ? cfg->rounds : 1), sizeof(double));
    h         = (qaoa_boundary_t*)malloc(qaoa_boundary_size(p->max_block));
    P         = (qaoa_local_t*)malloc(qaoa_local_size(p->max_block, p->max_internal));
    circuit   = (parsec_qpu_circuit_t*)malloc(circuit_room);
    result[0] = (parsec_qpu_result_t*)malloc(parsec_qpu_result_size(max_entries));
    result[1] = (parsec_qpu_result_t*)malloc(parsec_qpu_result_size(max_entries));
    if( (NULL == run->x) || (NULL == run->cut_after_round) || (NULL == h) || (NULL == P) ||
        (NULL == circuit) || (NULL == result[0]) || (NULL == result[1]) ) goto done;

    for( int b = 0; b < K; b++ ) qaoa_state_init(&run->x[b], inst, cfg, b);
    run->cut_initial = cut = qaoa_cut(inst, run->x);

    for( int r = 0; r < cfg->rounds; r++ ) {
        for( int c = 0; c < p->nb_colours; c++ ) {
            for( int b = 0; b < K; b++ ) {
                qaoa_opt_t S;
                double improvement;

                if( p->colour[b] != c ) continue;

                TIMED(run, QAOA_T_GATHER, {
                    qaoa_boundary_init(h, inst, b);
                    for( int k = 0; k < p->deg[b]; k++ ) {
                        qaoa_boundary_gather(h, inst, b, k, &run->x[qaoa_block_nbr(p, b, k)]);
                    }
                });
                TIMED(run, QAOA_T_BUILD, {
                    qaoa_local_build(P, inst, cfg, r, b, &run->x[b], h);
                    if( 0 != qaoa_circuit_build(circuit, circuit_room, P, cfg) ) goto done;
                    qaoa_opt_init(&S, cfg, &run->x[b], P);
                });

                for( int t = 0; t < cfg->steps; t++ ) {
                    qaoa_eval_t F[2];

                    TIMED(run, QAOA_T_STEP, qaoa_opt_propose(&S, cfg, theta[0], theta[1]));
                    for( int e = 0; e < 2; e++ ) {
                        if( P->skip ) {
                            /* An empty job: it reaches no backend and says nothing. */
                            memset(&F[e], 0, sizeof(F[e]));
                            continue;
                        }
                        TIMED(run, QAOA_T_SAMPLE, {
                            memset(result[e], 0, sizeof(parsec_qpu_result_t));
                            if( 0 != sample(circuit, theta[e], (uint32_t)cfg->shots, qaoa_key(r, b, t, e),
                                            result[e], max_entries, arg) ) goto done;
                        });
                        run->jobs++;
                        TIMED(run, QAOA_T_EVAL, qaoa_eval(&F[e], P, result[e]));
                        if( !F[e].valid ) run->failed_jobs++;
                    }
                    TIMED(run, QAOA_T_STEP, qaoa_opt_absorb(&S, cfg, &F[0], &F[1]));
                }

                TIMED(run, QAOA_T_UPDATE, improvement = qaoa_state_update(&run->x[b], P, &S));
                run->updates++;
                run->improvement += improvement;
                if( P->skip ) run->skipped++;
                if( improvement > 0.0 ) run->accepted++;
                if( check ) {
                    /* The cut of the whole graph moved by exactly what the
                     * block gained, and did not go down. Weights are
                     * multiples of 1/8, so this is an equality. */
                    const double now = qaoa_cut(inst, run->x);
                    if( (now != cut + improvement) || (improvement < 0.0) ) run->violations++;
                    cut = now;
                }
            }
        }
        run->cut_after_round[r] = qaoa_cut(inst, run->x);
    }
    run->cut_final = qaoa_cut(inst, run->x);
    rc = 0;

  done:
    free(h);
    free(P);
    free(circuit);
    free(result[0]);
    free(result[1]);
    if( 0 != rc ) qaoa_run_free(run);
    return rc;
}

void qaoa_run_free(qaoa_run_t *run)
{
    free(run->x);
    free(run->cut_after_round);
    run->x = NULL;
    run->cut_after_round = NULL;
}

int qaoa_sample_direct(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                       uint64_t key, parsec_qpu_result_t *result, uint32_t max_entries, void *arg)
{
    qaoa_direct_t *direct = (qaoa_direct_t*)arg;

    if( NULL == direct->sampler ) {
        if( NULL == (direct->sampler = parsec_qpu_sampler_new()) ) return -1;
    }
    /* What the sampler cannot run is an error of the program, not a job that failed. */
    return (PARSEC_SUCCESS == parsec_qpu_sampler_run(direct->sampler, circuit, params, shots, direct->seed, key,
                                                     result, max_entries, NULL)) ? 0 : -1;
}

int qaoa_sample_blocking(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                         uint64_t key, parsec_qpu_result_t *result, uint32_t max_entries, void *arg)
{
    parsec_qpu_backend_t *backend = (parsec_qpu_backend_t*)arg;
    const parsec_qpu_backend_ops_t *ops = backend->ops;
    parsec_qpu_request_t request;
    parsec_qpu_event_t event;
    parsec_qpu_job_id_t id = 0;

    memset(&request, 0, sizeof(request));
    request.circuit     = circuit;
    request.params      = params;
    request.shots       = shots;
    request.max_entries = max_entries;
    request.key         = key;
    request.attempt     = 1;

    /* One job, no retry: a job that does not end well is an evaluation that failed. */
    result->status = PARSEC_QPU_STATUS_FAILED;
    if( PARSEC_SUCCESS != ops->submit(backend, &request, 1, key + 1, &id) ) return 0;
    for(;;) {
        const int n = ops->wait(backend, 1000000000LL, &event, 1);
        if( n < 0 ) return -1;
        if( (1 == n) && (event.id == id) ) break;
    }
    if( PARSEC_QPU_JOB_DONE != event.state ) return 0;
    if( PARSEC_SUCCESS != ops->results(backend, id, &result, 1) ) {
        memset(result, 0, sizeof(*result));
        result->status = PARSEC_QPU_STATUS_FAILED;
    }
    return 0;
}
