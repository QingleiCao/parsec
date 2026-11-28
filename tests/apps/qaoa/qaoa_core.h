/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef QAOA_CORE_H
#define QAOA_CORE_H

/**
 * @file
 * Block-decomposed QAOA for weighted MaxCut: what each step of the algorithm
 * computes, as functions on flat buffers.
 *
 * Conventions (section 3.1 of docs/design/parsec-qpu-qaoa-plan.md). Every
 * quantity is a cut value, to be maximized. Vertex i of a block is qubit i
 * and bit i of the block's assignment; x = 0 is |0> and spin +1. The local
 * objective of a block S, with everything outside it fixed, is
 *
 *   C_S(x) = c_S + sum over edges (i,j) inside S of w_ij (x_i xor x_j)
 *                - sum over i in S of h_i z_i,            z_i = 1 - 2 x_i
 *
 * with h_i = 1/2 sum of w_ij z_j and c_S = 1/2 sum of w_ij over the edges
 * from i to vertices j outside S. A change of the block that raises C_S by d
 * raises the cut of the whole graph by exactly d, as long as no neighbour of
 * the block changes meanwhile.
 *
 * One update of a block: gather the assignments of its neighbours into the
 * fields h, build the local problem and its circuit, run a fixed number of
 * SPSA steps, each of which measures the circuit at two points, and keep the
 * best bitstring seen if it is strictly better than the current one.
 *
 * Every structure here is a flat buffer with no pointer in it: the
 * distributed application passes them between tasks and between ranks as
 * they are. Every function is a pure function of its arguments, so that a
 * sequential program and a task graph that call them in the same order per
 * block get the same bits.
 */

#include "qaoa_graph.h"
#include "parsec/parsec_config.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

BEGIN_C_DECLS

#define QAOA_MAX_LAYERS  4

typedef struct qaoa_config_s {
    int      layers;      /**< p, at most QAOA_MAX_LAYERS */
    int      steps;       /**< T: SPSA steps per update of a block, two jobs each */
    int      rounds;      /**< R: sweeps over all blocks */
    int      shots;       /**< per job */
    int      patience;    /**< a block that did not improve in this many updates in a row, and
                           *   whose neighbours have not changed since its last one, is not
                           *   evaluated again; 0: every block is evaluated every round */
    uint64_t seed;        /**< of the initial assignment and of the SPSA perturbations */
    /* SPSA gains: a_t = a / (t + 1 + A)^alpha, c_t = c / (t + 1)^gamma */
    double   spsa_a, spsa_c, spsa_A, spsa_alpha, spsa_gamma;
    /* First angles of a block, before it has any of its own. gamma0 is
     * divided by the largest coefficient of the block's local problem. */
    double   gamma0, beta0;
} qaoa_config_t;

/** One layer, 4 steps, 2 rounds, 256 shots, no skipping, seed 1. */
void qaoa_config_default(qaoa_config_t *cfg);

/* ------------------------------------------------------------------------
 * The state of a block: what one update hands to the next, and to the
 * neighbours.
 * --------------------------------------------------------------------- */

typedef struct qaoa_state_s {
    uint64_t bits;                          /**< the assignment */
    uint32_t version;                       /**< number of times it changed */
    uint32_t stale;                         /**< updates in a row that did not change it */
    uint32_t updates;                       /**< updates done */
    uint32_t skipped;                       /**< of which without any evaluation */
    uint32_t have_theta;                    /**< theta holds angles of this block's own */
    uint32_t pad;
    double   improvement;                   /**< sum of the improvements accepted */
    double   theta[2 * QAOA_MAX_LAYERS];    /**< gamma_1, beta_1, gamma_2, ...: where the next update starts */
    uint32_t nbr_version[QAOA_MAX_NBR];     /**< the versions of the neighbours at the last update */
} qaoa_state_t;

/** The first state of block b: an assignment drawn from the seed. */
void qaoa_state_init(qaoa_state_t *x, const qaoa_instance_t *inst, const qaoa_config_t *cfg, int b);

/** The cut of the whole graph for the states of all blocks. */
double qaoa_cut(const qaoa_instance_t *inst, const qaoa_state_t *states);

/* ------------------------------------------------------------------------
 * The boundary of a block: what its neighbours contribute, folded in one
 * neighbour at a time.
 * --------------------------------------------------------------------- */

typedef struct qaoa_boundary_s {
    double   constant;                      /**< c_S so far */
    uint32_t nb;                            /**< neighbours folded in */
    uint32_t n;                             /**< vertices of the block */
    uint32_t nbr_version[QAOA_MAX_NBR];     /**< the version of each neighbour that was read */
    double   h[];                           /**< n fields so far */
} qaoa_boundary_t;

static inline size_t qaoa_boundary_size(int n) { return sizeof(qaoa_boundary_t) + (size_t)n * sizeof(double); }

void qaoa_boundary_init(qaoa_boundary_t *h, const qaoa_instance_t *inst, int b);
/** Fold in the k-th neighbour of block b, whose state is x_nbr. To be called
 *  for k = 0, 1, ... in that order. */
void qaoa_boundary_gather(qaoa_boundary_t *h, const qaoa_instance_t *inst, int b, int k, const qaoa_state_t *x_nbr);

/* ------------------------------------------------------------------------
 * The local problem of one update of a block.
 * --------------------------------------------------------------------- */

typedef struct qaoa_local_edge_s {
    uint16_t i, j;
    uint32_t pad;
    double   w;
} qaoa_local_edge_t;

/** This header is followed by n fields (doubles) and m edges. */
typedef struct qaoa_local_s {
    uint32_t n;                             /**< qubits */
    uint32_t m;                             /**< edges inside the block, those of weight 0 left out */
    uint32_t round, block;
    uint32_t skip;                          /**< this update evaluates nothing */
    uint32_t nb_nbr;
    uint32_t nbr_version[QAOA_MAX_NBR];     /**< the versions of the neighbours it was built from */
    double   constant;                      /**< c_S */
    double   current;                       /**< C_S of the assignment the block enters the update with */
    double   weight;                        /**< sum of |w| and 2|h|: the scale of C_S, at least 1/16 */
    double   largest;                       /**< largest |w| or 2|h|, at least 1/16 */
} qaoa_local_t;

static inline double *qaoa_local_h(const qaoa_local_t *P) { return (double*)(P + 1); }
static inline qaoa_local_edge_t *qaoa_local_edges(const qaoa_local_t *P)
{
    return (qaoa_local_edge_t*)(qaoa_local_h(P) + P->n);
}
static inline size_t qaoa_local_size(int n, int m)
{
    return sizeof(qaoa_local_t) + (size_t)n * sizeof(double) + (size_t)m * sizeof(qaoa_local_edge_t);
}

/** Build the local problem of block b for round `round` from its state and
 *  its boundary, all neighbours folded in. */
void qaoa_local_build(qaoa_local_t *P, const qaoa_instance_t *inst, const qaoa_config_t *cfg,
                      int round, int b, const qaoa_state_t *x, const qaoa_boundary_t *h);

/** C_S of an assignment of the block. */
double qaoa_local_value(const qaoa_local_t *P, uint64_t bits);

/* ------------------------------------------------------------------------
 * The circuit of a local problem (section 3.3): Hadamards; per layer,
 * RZZ(-gamma w_ij) on every edge, RZ(-2 gamma h_i) on every qubit with a
 * field, RX(2 beta) on every qubit; measurement of all. Its parameters are
 * gamma_1, beta_1, gamma_2, beta_2, ...
 * --------------------------------------------------------------------- */

/** Bytes that hold the circuit of any block of n qubits and m edges. */
static inline size_t qaoa_circuit_size(int n, int m, int layers)
{
    return sizeof(parsec_qpu_circuit_t) + (size_t)(2 * n + layers * (m + 2 * n)) * PARSEC_QPU_OP_SIZE;
}

/** Returns 0, or -1 when the buffer is too small. For an update that
 *  evaluates nothing, the circuit is an empty job. */
int qaoa_circuit_build(parsec_qpu_circuit_t *circuit, size_t room, const qaoa_local_t *P, const qaoa_config_t *cfg);

/* ------------------------------------------------------------------------
 * One job's worth of information, and the optimizer.
 * --------------------------------------------------------------------- */

/** What a result says about the local problem. */
typedef struct qaoa_eval_s {
    uint32_t valid;                         /**< the job gave counts */
    uint32_t shots;
    double   mean;                          /**< the estimate of the expected C_S */
    double   best_value;                    /**< the largest C_S among the measured bitstrings */
    uint64_t best_bits;                     /**< the smallest bitstring that has it */
} qaoa_eval_t;

void qaoa_eval(qaoa_eval_t *F, const qaoa_local_t *P, const parsec_qpu_result_t *result);

/** The optimizer of one update of a block. */
typedef struct qaoa_opt_s {
    uint32_t round, block;
    uint32_t t;                             /**< steps taken in */
    uint32_t skip;
    uint32_t evaluations;                   /**< jobs that gave counts */
    uint32_t failed;                        /**< jobs that did not */
    uint32_t have_best, have_mean;
    uint64_t best_bits;                     /**< the best bitstring measured in this update */
    double   best_value;
    double   best_mean;                     /**< the largest estimate, and the angles that gave it */
    double   best_theta[2 * QAOA_MAX_LAYERS];
    double   theta[2 * QAOA_MAX_LAYERS];    /**< the iterate */
    double   delta[2 * QAOA_MAX_LAYERS];    /**< the perturbation of the step in flight, +1 or -1 each */
    double   ck;                            /**< and its size */
    double   weight;                        /**< the scale of the local problem */
} qaoa_opt_t;

void qaoa_opt_init(qaoa_opt_t *S, const qaoa_config_t *cfg, const qaoa_state_t *x, const qaoa_local_t *P);

/** Choose the perturbation of the next step and write the two sets of
 *  angles to measure at: the iterate plus and minus the perturbation. */
void qaoa_opt_propose(qaoa_opt_t *S, const qaoa_config_t *cfg, double *theta_plus, double *theta_minus);

/** Take in the two evaluations of the step that was proposed: remember the
 *  best bitstring and the best angles, and move the iterate. An evaluation
 *  that is not valid gives no candidate, and the iterate does not move
 *  unless both are valid. */
void qaoa_opt_absorb(qaoa_opt_t *S, const qaoa_config_t *cfg, const qaoa_eval_t *plus, const qaoa_eval_t *minus);

/** End the update: the block takes the best bitstring measured if its C_S
 *  is strictly larger than the current one. Returns the improvement, 0 when
 *  the assignment stays. */
double qaoa_state_update(qaoa_state_t *x, const qaoa_local_t *P, const qaoa_opt_t *S);

/* These buffers go from task to task and from rank to rank as bytes: their
 * sizes are part of what the processes of a run have to agree on. */
typedef char qaoa_state_is_232_bytes[(232 == sizeof(qaoa_state_t)) ? 1 : -1];
typedef char qaoa_boundary_header_is_144_bytes[(144 == sizeof(qaoa_boundary_t)) ? 1 : -1];
typedef char qaoa_local_header_is_184_bytes[(184 == sizeof(qaoa_local_t)) ? 1 : -1];
typedef char qaoa_local_edge_is_16_bytes[(16 == sizeof(qaoa_local_edge_t)) ? 1 : -1];
typedef char qaoa_eval_is_32_bytes[(32 == sizeof(qaoa_eval_t)) ? 1 : -1];
typedef char qaoa_opt_is_264_bytes[(264 == sizeof(qaoa_opt_t)) ? 1 : -1];

/** The key of the job of round r, block b, step t, evaluation e (0: plus,
 *  1: minus). The sequential reference and the distributed application
 *  name every evaluation the same way, which is what makes their samples
 *  the same. */
static inline uint64_t qaoa_key(int r, int b, int t, int e)
{
    return (((uint64_t)(uint32_t)r) << 44) | (((uint64_t)(uint32_t)b) << 20) | (((uint64_t)(uint32_t)t) << 1) | (uint64_t)(e & 1);
}

/** Entries a result needs for a block of n qubits: no more distinct
 *  bitstrings than shots, nor than there are. */
static inline uint32_t qaoa_result_entries(const qaoa_config_t *cfg, int n)
{
    const uint64_t all = (n >= 32) ? UINT32_MAX : (((uint64_t)1) << n);
    return (uint32_t)(((uint64_t)cfg->shots < all) ? (uint64_t)cfg->shots : all);
}

/* ------------------------------------------------------------------------
 * Exact answers by enumeration (qaoa_exact.c)
 * --------------------------------------------------------------------- */

/** The largest cut of a graph of at most 30 vertices, and one assignment
 *  that has it (one byte per vertex) when best is not NULL. Returns a
 *  negative value for a graph that is too large. */
double qaoa_exact_maxcut(const qaoa_graph_t *g, unsigned char *best);

/** The largest C_S of a local problem of at most 30 qubits. */
double qaoa_exact_local(const qaoa_local_t *P, uint64_t *best_bits);

/** The exact mean and variance of C_S over the outcomes of the circuit at
 *  the given angles. Returns 0, or a PaRSEC error from the sampler. */
int qaoa_exact_expectation(const qaoa_local_t *P, const parsec_qpu_circuit_t *circuit, const double *theta,
                           double *mean, double *variance);

/* ------------------------------------------------------------------------
 * The sequential reference (qaoa_seq.c): the algorithm in the order
 * (round, colour, block, step, evaluation).
 * --------------------------------------------------------------------- */

/** Run one job and fill its result. Returns 0 also for a job that failed:
 *  the status of the result says so. */
typedef int (*qaoa_sample_fn_t)(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                                uint64_t key, parsec_qpu_result_t *result, uint32_t max_entries, void *arg);

enum { QAOA_T_GATHER, QAOA_T_BUILD, QAOA_T_SAMPLE, QAOA_T_EVAL, QAOA_T_STEP, QAOA_T_UPDATE, QAOA_NB_TIMERS };
extern const char *const qaoa_timer_names[QAOA_NB_TIMERS];

typedef struct qaoa_run_s {
    int           nb_blocks;
    qaoa_state_t *x;                 /**< [nb_blocks] the final states */
    double        cut_initial;       /**< recomputed from the first assignments */
    double        cut_final;         /**< recomputed from the last ones */
    double        improvement;       /**< sum of the improvements the updates accepted */
    double       *cut_after_round;   /**< [rounds] */
    long          updates, accepted, skipped;
    long          jobs;              /**< jobs asked of the sampler */
    long          failed_jobs;       /**< of which gave no counts */
    int           violations;        /**< updates after which the cut of the whole graph was not
                                      *   the one before plus the improvement, or went down */
    int64_t       ns[QAOA_NB_TIMERS];     /**< time spent in each kind of step */
    long          calls[QAOA_NB_TIMERS];
} qaoa_run_t;

/** Run the reference. With check set, the cut of the whole graph is
 *  recomputed after every update and compared with what the update claims.
 *  Returns 0, or -1 when a buffer could not be had or a job could not run. */
int  qaoa_sequential_run(const qaoa_instance_t *inst, const qaoa_config_t *cfg,
                         qaoa_sample_fn_t sample, void *arg, int check, qaoa_run_t *run);
void qaoa_run_free(qaoa_run_t *run);

/** The sampler, called directly. arg points to a qaoa_direct_t. The counts
 *  are those of the mock backend for the same seed and key. */
typedef struct qaoa_direct_s {
    parsec_qpu_sampler_t *sampler;   /**< NULL: one is made, and kept */
    uint64_t              seed;
} qaoa_direct_t;
int qaoa_sample_direct(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                       uint64_t key, parsec_qpu_result_t *result, uint32_t max_entries, void *arg);

/** The mock backend, one job at a time: submit, wait for its end, read its
 *  result. arg is the backend, opened by the caller. This is a blocking
 *  quantum call, with the latency the mock is set to. */
int qaoa_sample_blocking(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                         uint64_t key, parsec_qpu_result_t *result, uint32_t max_entries, void *arg);

END_C_DECLS

#endif  /* QAOA_CORE_H */
