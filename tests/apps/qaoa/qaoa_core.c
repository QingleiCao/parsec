/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "qaoa_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void qaoa_config_default(qaoa_config_t *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->layers     = 1;
    cfg->steps      = 4;
    cfg->rounds     = 2;
    cfg->shots      = 256;
    cfg->patience   = 0;
    cfg->seed       = 1;
    /* The usual SPSA exponents. The objective is divided by the scale of
     * the local problem before it is differenced, so the gains are in
     * radians. */
    cfg->spsa_a     = 0.25;
    cfg->spsa_c     = 0.15;
    cfg->spsa_A     = 1.0;
    cfg->spsa_alpha = 0.602;
    cfg->spsa_gamma = 0.101;
    cfg->gamma0     = 0.5;
    cfg->beta0      = 0.35;
}

/* ------------------------------------------------------------------------
 * State
 * --------------------------------------------------------------------- */

void qaoa_state_init(qaoa_state_t *x, const qaoa_instance_t *inst, const qaoa_config_t *cfg, int b)
{
    const int n = qaoa_block_size(&inst->part, b);

    memset(x, 0, sizeof(*x));
    /* An assignment that depends on the seed and the block, and on nothing else. */
    x->bits = parsec_qpu_mix64(parsec_qpu_mix64(cfg->seed ^ 0x51a0a51a0a51a0a5ULL) ^ (uint64_t)b);
    if( n < 64 ) x->bits &= (((uint64_t)1) << n) - 1;
}

double qaoa_cut(const qaoa_instance_t *inst, const qaoa_state_t *states)
{
    const qaoa_graph_t *g = &inst->graph;
    const qaoa_partition_t *p = &inst->part;
    double cut = 0.0;

    for( int e = 0; e < g->m; e++ ) {
        const int u = g->eu[e], v = g->ev[e];
        const uint64_t xu = (states[p->block_of[u]].bits >> p->local_of[u]) & 1;
        const uint64_t xv = (states[p->block_of[v]].bits >> p->local_of[v]) & 1;
        if( xu != xv ) cut += g->ew[e];
    }
    return cut;
}

/* ------------------------------------------------------------------------
 * Boundary
 * --------------------------------------------------------------------- */

void qaoa_boundary_init(qaoa_boundary_t *h, const qaoa_instance_t *inst, int b)
{
    const int n = qaoa_block_size(&inst->part, b);

    memset(h, 0, qaoa_boundary_size(n));
    h->n = (uint32_t)n;
}

void qaoa_boundary_gather(qaoa_boundary_t *h, const qaoa_instance_t *inst, int b, int k, const qaoa_state_t *x_nbr)
{
    const qaoa_graph_t *g = &inst->graph;
    const qaoa_partition_t *p = &inst->part;
    const int d = qaoa_block_nbr(p, b, k);

    /* Every edge from a vertex of b to a vertex of d: with the outside end
     * fixed at spin z, it adds w/2 to the constant and w z / 2 to the field
     * of the inside end. Weights are multiples of 1/8, so all of it is exact. */
    for( int i = 0; i < (int)h->n; i++ ) {
        const int u = qaoa_block_vertex(p, b, i);
        for( int a = g->adj_off[u]; a < g->adj_off[u + 1]; a++ ) {
            const int v = g->adj_v[a];
            if( p->block_of[v] != d ) continue;
            const double z = ((x_nbr->bits >> p->local_of[v]) & 1) ? -1.0 : 1.0;
            h->h[i]     += 0.5 * g->adj_w[a] * z;
            h->constant += 0.5 * g->adj_w[a];
        }
    }
    h->nbr_version[k] = x_nbr->version;
    h->nb = (uint32_t)(k + 1);
}

/* ------------------------------------------------------------------------
 * Local problem
 * --------------------------------------------------------------------- */

void qaoa_local_build(qaoa_local_t *P, const qaoa_instance_t *inst, const qaoa_config_t *cfg,
                      int round, int b, const qaoa_state_t *x, const qaoa_boundary_t *h)
{
    const qaoa_graph_t *g = &inst->graph;
    const qaoa_partition_t *p = &inst->part;
    const int n = qaoa_block_size(p, b);
    qaoa_local_edge_t *edges;
    double *fields;
    int unchanged = 1;

    memset(P, 0, sizeof(*P));
    P->n        = (uint32_t)n;
    P->round    = (uint32_t)round;
    P->block    = (uint32_t)b;
    P->nb_nbr   = (uint32_t)p->deg[b];
    P->constant = h->constant;
    fields = qaoa_local_h(P);
    memcpy(fields, h->h, (size_t)n * sizeof(double));
    for( int k = 0; k < p->deg[b]; k++ ) {
        P->nbr_version[k] = h->nbr_version[k];
        if( h->nbr_version[k] != x->nbr_version[k] ) unchanged = 0;
    }

    /* The edges inside the block, each once, in the order of their smaller
     * end and then of the adjacency list. An edge of weight 0 is no term. */
    edges = qaoa_local_edges(P);
    for( int i = 0; i < n; i++ ) {
        const int u = qaoa_block_vertex(p, b, i);
        for( int a = g->adj_off[u]; a < g->adj_off[u + 1]; a++ ) {
            const int v = g->adj_v[a];
            if( (p->block_of[v] != b) || (v < u) || (0.0 == g->adj_w[a]) ) continue;
            edges[P->m].i   = (uint16_t)i;
            edges[P->m].j   = (uint16_t)p->local_of[v];
            edges[P->m].pad = 0;
            edges[P->m].w   = g->adj_w[a];
            P->m++;
        }
    }

    for( uint32_t e = 0; e < P->m; e++ ) {
        P->weight += fabs(edges[e].w);
        if( fabs(edges[e].w) > P->largest ) P->largest = fabs(edges[e].w);
    }
    for( int i = 0; i < n; i++ ) {
        P->weight += 2.0 * fabs(fields[i]);
        if( 2.0 * fabs(fields[i]) > P->largest ) P->largest = 2.0 * fabs(fields[i]);
    }
    /* A block with no term at all still has a scale to divide by. */
    if( P->weight  < 0.0625 ) P->weight  = 0.0625;
    if( P->largest < 0.0625 ) P->largest = 0.0625;

    P->current = qaoa_local_value(P, x->bits);

    /* Nothing around the block changed since its last update, and that one
     * and those before it found nothing better: the problem is the same,
     * and it is not asked again. */
    P->skip = (cfg->patience > 0) && (x->updates > 0) && unchanged && (x->stale >= (uint32_t)cfg->patience);
}

double qaoa_local_value(const qaoa_local_t *P, uint64_t bits)
{
    const double *h = qaoa_local_h(P);
    const qaoa_local_edge_t *edges = qaoa_local_edges(P);
    double value = P->constant;

    for( uint32_t i = 0; i < P->n; i++ ) {
        value += ((bits >> i) & 1) ? h[i] : -h[i];
    }
    for( uint32_t e = 0; e < P->m; e++ ) {
        if( ((bits >> edges[e].i) ^ (bits >> edges[e].j)) & 1 ) value += edges[e].w;
    }
    return value;
}

/* ------------------------------------------------------------------------
 * Circuit
 * --------------------------------------------------------------------- */

int qaoa_circuit_build(parsec_qpu_circuit_t *circuit, size_t room, const qaoa_local_t *P, const qaoa_config_t *cfg)
{
    const double *h = qaoa_local_h(P);
    const qaoa_local_edge_t *edges = qaoa_local_edges(P);
    const int n = (int)P->n;
    int rc = 0;

    if( room < sizeof(parsec_qpu_circuit_t) ) return -1;
    parsec_qpu_circuit_init(circuit, P->n, (uint32_t)(2 * cfg->layers));
    if( P->skip ) {
        circuit->flags |= PARSEC_QPU_CIRCUIT_EMPTY;
        return 0;
    }

    for( int q = 0; q < n; q++ ) {
        rc |= parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_H, q, 0, PARSEC_QPU_NO_PARAM, 0.0);
    }
    for( int l = 0; l < cfg->layers; l++ ) {
        /* exp(-i gamma H_S), up to a phase: H_S has w/2 (1 - ZZ) per edge
         * and -h Z per qubit, and a rotation by t is exp(-i t P / 2). */
        for( uint32_t e = 0; e < P->m; e++ ) {
            rc |= parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_RZZ, edges[e].i, edges[e].j, 2 * l, -edges[e].w);
        }
        for( int q = 0; q < n; q++ ) {
            if( 0.0 == h[q] ) continue;
            rc |= parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_RZ, q, 0, 2 * l, -2.0 * h[q]);
        }
        /* exp(-i beta X) on every qubit */
        for( int q = 0; q < n; q++ ) {
            rc |= parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_RX, q, 0, 2 * l + 1, 2.0);
        }
    }
    rc |= parsec_qpu_circuit_measure_all(circuit, room);
    return (0 == rc) ? 0 : -1;
}

/* ------------------------------------------------------------------------
 * Evaluation of a result
 * --------------------------------------------------------------------- */

void qaoa_eval(qaoa_eval_t *F, const qaoa_local_t *P, const parsec_qpu_result_t *result)
{
    double sum = 0.0;

    memset(F, 0, sizeof(*F));
    if( (PARSEC_QPU_STATUS_OK != result->status) || (0 == result->shots_returned) || (0 == result->nb_entries) ) {
        return;     /* a job that failed, or an empty one: it says nothing */
    }
    for( uint32_t i = 0; i < result->nb_entries; i++ ) {
        uint64_t bits;
        uint32_t count;
        double value;

        parsec_qpu_result_get_entry(result, i, &bits, &count);
        value = qaoa_local_value(P, bits);
        sum += (double)count * value;
        /* Entries come in increasing order of their bitstring, so the first
         * one with the largest value is the smallest. */
        if( (0 == i) || (value > F->best_value) ) {
            F->best_value = value;
            F->best_bits  = bits;
        }
    }
    F->valid = 1;
    F->shots = result->shots_returned;
    F->mean  = sum / (double)result->shots_returned;
}

/* ------------------------------------------------------------------------
 * Optimizer
 * --------------------------------------------------------------------- */

void qaoa_opt_init(qaoa_opt_t *S, const qaoa_config_t *cfg, const qaoa_state_t *x, const qaoa_local_t *P)
{
    memset(S, 0, sizeof(*S));
    S->round  = P->round;
    S->block  = P->block;
    S->skip   = P->skip;
    S->weight = P->weight;
    if( x->have_theta ) {
        memcpy(S->theta, x->theta, sizeof(S->theta));
    } else {
        /* gamma w of order one for the largest coefficient, whatever the
         * weights of the instance are. */
        for( int l = 0; l < cfg->layers; l++ ) {
            S->theta[2 * l]     = cfg->gamma0 / P->largest;
            S->theta[2 * l + 1] = cfg->beta0;
        }
    }
}

void qaoa_opt_propose(qaoa_opt_t *S, const qaoa_config_t *cfg, double *theta_plus, double *theta_minus)
{
    /* The perturbation of step t of this update: a hash of the seed, the
     * round, the block and the step, so that it is the same whenever and
     * wherever the step runs. */
    const uint64_t bits = parsec_qpu_mix64(parsec_qpu_mix64(parsec_qpu_mix64(cfg->seed ^ 0x5b5a5b5a5b5a5b5aULL)
                                                            ^ (((uint64_t)S->round << 32) | (uint64_t)S->block))
                                           ^ (uint64_t)S->t);

    S->ck = cfg->spsa_c / pow((double)S->t + 1.0, cfg->spsa_gamma);
    for( int j = 0; j < 2 * cfg->layers; j++ ) {
        S->delta[j]    = ((bits >> j) & 1) ? 1.0 : -1.0;
        theta_plus[j]  = S->theta[j] + S->ck * S->delta[j];
        theta_minus[j] = S->theta[j] - S->ck * S->delta[j];
    }
}

/* One of the two points of a step: its best bitstring is a candidate, and
 * its angles are remembered if no point of this update did better. */
static void opt_take(qaoa_opt_t *S, const qaoa_config_t *cfg, const qaoa_eval_t *F, double sign)
{
    if( !F->valid ) {
        S->failed++;
        return;
    }
    S->evaluations++;
    if( !S->have_best || (F->best_value > S->best_value) ||
        ((F->best_value == S->best_value) && (F->best_bits < S->best_bits)) ) {
        S->have_best  = 1;
        S->best_value = F->best_value;
        S->best_bits  = F->best_bits;
    }
    if( !S->have_mean || (F->mean > S->best_mean) ) {
        S->have_mean = 1;
        S->best_mean = F->mean;
        for( int j = 0; j < 2 * cfg->layers; j++ ) S->best_theta[j] = S->theta[j] + sign * S->ck * S->delta[j];
    }
}

void qaoa_opt_absorb(qaoa_opt_t *S, const qaoa_config_t *cfg, const qaoa_eval_t *plus, const qaoa_eval_t *minus)
{
    if( !S->skip ) {
        opt_take(S, cfg, plus, 1.0);
        opt_take(S, cfg, minus, -1.0);
        if( plus->valid && minus->valid ) {
            /* Ascent: the objective is a cut value. Each component of the
             * perturbation is +1 or -1, so dividing by it is multiplying. */
            const double ak = cfg->spsa_a / pow((double)S->t + 1.0 + cfg->spsa_A, cfg->spsa_alpha);
            const double slope = (plus->mean - minus->mean) / (2.0 * S->ck * S->weight);
            for( int j = 0; j < 2 * cfg->layers; j++ ) S->theta[j] += ak * slope * S->delta[j];
        }
    }
    S->t++;
}

double qaoa_state_update(qaoa_state_t *x, const qaoa_local_t *P, const qaoa_opt_t *S)
{
    double improvement = 0.0;

    x->updates++;
    memcpy(x->nbr_version, P->nbr_version, sizeof(x->nbr_version));
    if( P->skip ) {
        x->skipped++;
        return 0.0;
    }
    if( S->have_best && (S->best_value > P->current) ) {
        improvement     = S->best_value - P->current;
        x->bits         = S->best_bits;
        x->version++;
        x->stale        = 0;
        x->improvement += improvement;
    } else {
        x->stale++;
    }
    if( S->have_mean ) {
        memcpy(x->theta, S->best_theta, sizeof(x->theta));
        x->have_theta = 1;
    }
    return improvement;
}
