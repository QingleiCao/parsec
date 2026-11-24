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
 * Exact answers by enumeration, for graphs and blocks small enough. They are
 * what the sampled answers are measured against; nothing in the algorithm
 * uses them.
 *
 * Both enumerations walk a Gray code, so that going from one assignment to
 * the next flips one vertex and costs its degree.
 */

#define QAOA_EXACT_MAX 30

double qaoa_exact_maxcut(const qaoa_graph_t *g, unsigned char *best)
{
    const int n = g->n;
    unsigned char *x;
    uint64_t best_code = 0, code = 0;
    double cut = 0.0, best_cut = 0.0;

    if( n > QAOA_EXACT_MAX ) return -1.0;
    x = (unsigned char*)calloc((size_t)n, 1);

    /* The cut of an assignment is that of its complement: the last vertex
     * stays at 0, and the others take every value. */
    for( uint64_t k = 1; k < (((uint64_t)1) << (n - 1)); k++ ) {
        const int v = __builtin_ctzll(k);
        for( int a = g->adj_off[v]; a < g->adj_off[v + 1]; a++ ) {
            /* An edge that was not cut becomes cut, and the other way round. */
            cut += (x[v] == x[g->adj_v[a]]) ? g->adj_w[a] : -g->adj_w[a];
        }
        x[v] ^= 1;
        code ^= ((uint64_t)1) << v;
        if( cut > best_cut ) {
            best_cut  = cut;
            best_code = code;
        }
    }
    if( NULL != best ) {
        for( int v = 0; v < n; v++ ) best[v] = (unsigned char)((best_code >> v) & 1);
    }
    free(x);
    return best_cut;
}

double qaoa_exact_local(const qaoa_local_t *P, uint64_t *best_bits)
{
    const int n = (int)P->n;
    const double *h = qaoa_local_h(P);
    const qaoa_local_edge_t *edges = qaoa_local_edges(P);
    uint64_t bits = 0, best = 0;
    double value, best_value;

    if( n > QAOA_EXACT_MAX ) return -1.0;
    value = best_value = qaoa_local_value(P, 0);

    for( uint64_t k = 1; k < (((uint64_t)1) << n); k++ ) {
        const int v = __builtin_ctzll(k);
        const uint64_t was = (bits >> v) & 1;
        /* The field term of v goes from -h to +h, or back. */
        value += was ? -2.0 * h[v] : 2.0 * h[v];
        for( uint32_t e = 0; e < P->m; e++ ) {
            int other;
            if( edges[e].i == v ) other = edges[e].j;
            else if( edges[e].j == v ) other = edges[e].i;
            else continue;
            value += (was == ((bits >> other) & 1)) ? edges[e].w : -edges[e].w;
        }
        bits ^= ((uint64_t)1) << v;
        if( (value > best_value) || ((value == best_value) && (bits < best)) ) {
            best_value = value;
            best       = bits;
        }
    }
    if( NULL != best_bits ) *best_bits = best;
    return best_value;
}

int qaoa_exact_expectation(const qaoa_local_t *P, const parsec_qpu_circuit_t *circuit, const double *theta,
                           double *mean, double *variance)
{
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();
    const double *amp = NULL;
    double m1 = 0.0, m2 = 0.0;
    int rc;

    if( NULL == sampler ) return PARSEC_ERR_OUT_OF_RESOURCE;
    rc = parsec_qpu_sampler_state(sampler, circuit, theta, NULL, &amp);
    if( PARSEC_SUCCESS == rc ) {
        for( uint64_t x = 0; x < (((uint64_t)1) << P->n); x++ ) {
            const double p = amp[2 * x] * amp[2 * x] + amp[2 * x + 1] * amp[2 * x + 1];
            const double v = qaoa_local_value(P, x);
            m1 += p * v;
            m2 += p * v * v;
        }
        if( NULL != mean ) *mean = m1;
        if( NULL != variance ) *variance = m2 - m1 * m1;
    }
    parsec_qpu_sampler_free(sampler);
    return rc;
}
