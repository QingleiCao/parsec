/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * An exact sampler: it computes the state a circuit prepares, amplitude by
 * amplitude, and draws measurement outcomes from it. It is what the mock
 * backend answers with, and what a program that does not use the runtime
 * calls to get the answer the mock would give.
 *
 * The state of n qubits is 2^n complex amplitudes, stored as pairs of
 * doubles (real part, imaginary part). Amplitude x belongs to the basis
 * state whose bit i is the value of qubit i, so a sampled index is the
 * measured bitstring as it is.
 *
 * Complex arithmetic is written out on the doubles. The compiler's complex
 * multiplication has to honour infinities and is several times slower.
 */

struct parsec_qpu_sampler_s {
    double   *amp;        /* 2 doubles per amplitude */
    size_t    amp_len;    /* in doubles */
    uint64_t *draws;      /* one outcome per shot */
    size_t    draws_len;
    uint64_t *runs;       /* (bitstring, count) pairs, when a result must be cut */
    size_t    runs_len;
};

parsec_qpu_sampler_t *parsec_qpu_sampler_new(void)
{
    return (parsec_qpu_sampler_t*)calloc(1, sizeof(parsec_qpu_sampler_t));
}

void parsec_qpu_sampler_free(parsec_qpu_sampler_t *sampler)
{
    if( NULL == sampler ) return;
    free(sampler->amp);
    free(sampler->draws);
    free(sampler->runs);
    free(sampler);
}

static inline int sampler_cancelled(const int32_t *cancel)
{
    return (NULL != cancel) && (0 != PARSEC_QPU_LOAD(cancel));
}

/* ------------------------------------------------------------------------
 * Gates. `dim` is the number of amplitudes; a gate on qubit q pairs the
 * amplitudes whose indices differ in bit q only.
 * --------------------------------------------------------------------- */

/* A one-qubit gate given by its matrix, row by row, each entry as (re, im). */
static void gate_matrix(double *a, uint64_t dim, uint32_t q, const double m[8])
{
    const uint64_t step = ((uint64_t)1) << q;

    for( uint64_t i = 0; i < dim; i += 2 * step ) {
        for( uint64_t j = i; j < i + step; j++ ) {
            double *p0 = a + 2 * j, *p1 = a + 2 * (j + step);
            const double x0r = p0[0], x0i = p0[1], x1r = p1[0], x1i = p1[1];
            p0[0] = m[0] * x0r - m[1] * x0i + m[2] * x1r - m[3] * x1i;
            p0[1] = m[0] * x0i + m[1] * x0r + m[2] * x1i + m[3] * x1r;
            p1[0] = m[4] * x0r - m[5] * x0i + m[6] * x1r - m[7] * x1i;
            p1[1] = m[4] * x0i + m[5] * x0r + m[6] * x1i + m[7] * x1r;
        }
    }
}

/* A diagonal one-qubit gate: the amplitudes with qubit q at 0 are multiplied
 * by (d[0], d[1]), those with qubit q at 1 by (d[2], d[3]). */
static void gate_diagonal(double *a, uint64_t dim, uint32_t q, const double d[4])
{
    const uint64_t step = ((uint64_t)1) << q;

    for( uint64_t i = 0; i < dim; i += 2 * step ) {
        for( uint64_t j = i; j < i + step; j++ ) {
            double *p0 = a + 2 * j, *p1 = a + 2 * (j + step);
            const double x0r = p0[0], x0i = p0[1], x1r = p1[0], x1i = p1[1];
            p0[0] = d[0] * x0r - d[1] * x0i;
            p0[1] = d[0] * x0i + d[1] * x0r;
            p1[0] = d[2] * x1r - d[3] * x1i;
            p1[1] = d[2] * x1i + d[3] * x1r;
        }
    }
}

/* A diagonal two-qubit gate that depends only on whether the two qubits
 * agree: (same[0], same[1]) multiplies the amplitudes where they do,
 * (diff[0], diff[1]) the others. */
static void gate_parity(double *a, uint64_t dim, uint32_t q0, uint32_t q1,
                        const double same[2], const double diff[2])
{
    for( uint64_t x = 0; x < dim; x++ ) {
        const double *d = (((x >> q0) ^ (x >> q1)) & 1) ? diff : same;
        const double xr = a[2 * x], xi = a[2 * x + 1];
        a[2 * x]     = d[0] * xr - d[1] * xi;
        a[2 * x + 1] = d[0] * xi + d[1] * xr;
    }
}

/* Controlled Z: the sign of the amplitudes with both qubits at 1 changes. */
static void gate_cz(double *a, uint64_t dim, uint32_t q0, uint32_t q1)
{
    const uint64_t both = (((uint64_t)1) << q0) | (((uint64_t)1) << q1);

    for( uint64_t x = 0; x < dim; x++ ) {
        if( both != (x & both) ) continue;
        a[2 * x]     = -a[2 * x];
        a[2 * x + 1] = -a[2 * x + 1];
    }
}

/* Controlled X: where the control is 1, the two values of the target swap. */
static void gate_cx(double *a, uint64_t dim, uint32_t control, uint32_t target)
{
    const uint64_t c = ((uint64_t)1) << control, t = ((uint64_t)1) << target;

    for( uint64_t x = 0; x < dim; x++ ) {
        if( (0 == (x & c)) || (0 != (x & t)) ) continue;
        double *p0 = a + 2 * x, *p1 = a + 2 * (x | t);
        const double r = p0[0], i = p0[1];
        p0[0] = p1[0]; p0[1] = p1[1];
        p1[0] = r;     p1[1] = i;
    }
}

/* (u, v) <- (c u - i s v, c v - i s u) */
static inline void mix_pair(double *u, double *v, double c, double s)
{
    const double ur = u[0], ui = u[1], vr = v[0], vi = v[1];
    u[0] = c * ur + s * vi;
    u[1] = c * ui - s * vr;
    v[0] = c * vr + s * ui;
    v[1] = c * vi - s * ur;
}

/* exp(-i t XX / 2) and exp(-i t YY / 2), with c = cos(t/2) and s = sin(t/2).
 * Both pair |00> with |11> and |01> with |10>. XX has the entry 1 in both
 * pairs. YY has 1 between |01> and |10> and -1 between |00> and |11>, since
 * <0|Y|1> = -i. */
static void gate_xx_yy(double *a, uint64_t dim, uint32_t q0, uint32_t q1, double c, double s, int yy)
{
    const uint64_t b0 = ((uint64_t)1) << q0, b1 = ((uint64_t)1) << q1;
    const double s_outer = yy ? -s : s;

    for( uint64_t x = 0; x < dim; x++ ) {
        if( 0 != (x & (b0 | b1)) ) continue;
        mix_pair(a + 2 * x,        a + 2 * (x | b0 | b1), c, s_outer);
        mix_pair(a + 2 * (x | b0), a + 2 * (x | b1),      c, s);
    }
}

static void sampler_apply(double *a, uint64_t dim, const parsec_qpu_op_t *op, const double *params)
{
    static const double inv_sqrt2 = 0.70710678118654752440;
    double t = 0.0, c = 1.0, s = 0.0;

    if( ((op->opcode >= PARSEC_QPU_OP_RX)  && (op->opcode <= PARSEC_QPU_OP_RZ)) ||
        ((op->opcode >= PARSEC_QPU_OP_RXX) && (op->opcode <= PARSEC_QPU_OP_RZZ)) ) {
        t = (PARSEC_QPU_NO_PARAM == op->param) ? op->angle : op->angle * params[op->param];
        c = cos(0.5 * t);
        s = sin(0.5 * t);
    }

    switch( op->opcode ) {
    case PARSEC_QPU_OP_H: {
        const double m[8] = { inv_sqrt2, 0, inv_sqrt2, 0, inv_sqrt2, 0, -inv_sqrt2, 0 };
        gate_matrix(a, dim, op->q0, m);
        break;
    }
    case PARSEC_QPU_OP_X: {
        const double m[8] = { 0, 0, 1, 0, 1, 0, 0, 0 };
        gate_matrix(a, dim, op->q0, m);
        break;
    }
    case PARSEC_QPU_OP_Y: {            /* [[0, -i], [i, 0]] */
        const double m[8] = { 0, 0, 0, -1, 0, 1, 0, 0 };
        gate_matrix(a, dim, op->q0, m);
        break;
    }
    case PARSEC_QPU_OP_Z: {
        const double d[4] = { 1, 0, -1, 0 };
        gate_diagonal(a, dim, op->q0, d);
        break;
    }
    case PARSEC_QPU_OP_RX: {           /* [[c, -i s], [-i s, c]] */
        const double m[8] = { c, 0, 0, -s, 0, -s, c, 0 };
        gate_matrix(a, dim, op->q0, m);
        break;
    }
    case PARSEC_QPU_OP_RY: {           /* [[c, -s], [s, c]] */
        const double m[8] = { c, 0, -s, 0, s, 0, c, 0 };
        gate_matrix(a, dim, op->q0, m);
        break;
    }
    case PARSEC_QPU_OP_RZ: {           /* diag(e^{-it/2}, e^{it/2}) */
        const double d[4] = { c, -s, c, s };
        gate_diagonal(a, dim, op->q0, d);
        break;
    }
    case PARSEC_QPU_OP_CX:
        gate_cx(a, dim, op->q0, op->q1);
        break;
    case PARSEC_QPU_OP_CZ:
        gate_cz(a, dim, op->q0, op->q1);
        break;
    case PARSEC_QPU_OP_RXX:
        gate_xx_yy(a, dim, op->q0, op->q1, c, s, 0);
        break;
    case PARSEC_QPU_OP_RYY:
        gate_xx_yy(a, dim, op->q0, op->q1, c, s, 1);
        break;
    case PARSEC_QPU_OP_RZZ: {          /* e^{-it/2} where the qubits agree, e^{it/2} where they differ */
        const double same[2] = { c, -s }, diff[2] = { c, s };
        gate_parity(a, dim, op->q0, op->q1, same, diff);
        break;
    }
    default:                           /* a measurement: nothing to do to the state */
        break;
    }
}

int parsec_qpu_sampler_state(parsec_qpu_sampler_t *sampler, const parsec_qpu_circuit_t *circuit,
                             const double *params, const int32_t *cancel, const double **amp)
{
    const parsec_qpu_op_t *ops;
    uint64_t dim;
    size_t need;
    int rc;

    if( NULL != amp ) *amp = NULL;
    if( PARSEC_SUCCESS != (rc = parsec_qpu_circuit_check(circuit, 0, NULL)) ) return rc;
    if( PARSEC_QPU_CIRCUIT_EMPTY & circuit->flags ) return PARSEC_ERR_BAD_PARAM;
    if( (circuit->nb_params > 0) && (NULL == params) ) return PARSEC_ERR_BAD_PARAM;
    if( circuit->nb_qubits > PARSEC_QPU_SAMPLER_MAX_QUBITS ) return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;

    dim  = ((uint64_t)1) << circuit->nb_qubits;
    need = (size_t)(2 * dim);
    if( sampler->amp_len < need ) {
        free(sampler->amp);
        sampler->amp     = (double*)malloc(need * sizeof(double));
        sampler->amp_len = (NULL != sampler->amp) ? need : 0;
        if( NULL == sampler->amp ) return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    memset(sampler->amp, 0, need * sizeof(double));
    sampler->amp[0] = 1.0;

    ops = parsec_qpu_circuit_ops(circuit);
    for( uint32_t i = 0; i < circuit->nb_ops; i++ ) {
        if( sampler_cancelled(cancel) ) return PARSEC_QPU_SAMPLER_CANCELLED;
        sampler_apply(sampler->amp, dim, &ops[i], params);
    }
    if( NULL != amp ) *amp = sampler->amp;
    return PARSEC_SUCCESS;
}

/* ------------------------------------------------------------------------
 * Sampling
 * --------------------------------------------------------------------- */

/* xoshiro256**, by Blackman and Vigna (public domain). One generator per
 * job, so nothing is shared between jobs. */
typedef struct { uint64_t s[4]; } sampler_rng_t;

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(sampler_rng_t *rng)
{
    const uint64_t result = rotl64(rng->s[1] * 5, 7) * 9, t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3]  = rotl64(rng->s[3], 45);
    return result;
}

static void rng_seed(sampler_rng_t *rng, uint64_t seed, uint64_t key)
{
    uint64_t z = parsec_qpu_mix64(parsec_qpu_mix64(seed) ^ key);
    for( int i = 0; i < 4; i++ ) rng->s[i] = z = parsec_qpu_mix64(z);
    if( 0 == (rng->s[0] | rng->s[1] | rng->s[2] | rng->s[3]) ) rng->s[0] = 1;
}

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

/* (bitstring, count) pairs: most frequent first, then by bitstring. */
static int cmp_run_by_count(const void *a, const void *b)
{
    const uint64_t *x = (const uint64_t*)a, *y = (const uint64_t*)b;
    if( x[1] != y[1] ) return (x[1] < y[1]) - (x[1] > y[1]);
    return (x[0] > y[0]) - (x[0] < y[0]);
}

int parsec_qpu_sampler_run(parsec_qpu_sampler_t *sampler, const parsec_qpu_circuit_t *circuit,
                           const double *params, uint32_t shots, uint64_t seed, uint64_t key,
                           parsec_qpu_result_t *result, uint32_t max_entries, const int32_t *cancel)
{
    sampler_rng_t rng;
    uint64_t dim, mask, distinct;
    double *cum, total = 0.0;
    int rc;

    if( (NULL == result) || (0 == shots) ) return PARSEC_ERR_BAD_PARAM;
    if( PARSEC_SUCCESS != (rc = parsec_qpu_sampler_state(sampler, circuit, params, cancel, NULL)) ) return rc;
    if( sampler_cancelled(cancel) ) return PARSEC_QPU_SAMPLER_CANCELLED;

    dim  = ((uint64_t)1) << circuit->nb_qubits;
    mask = parsec_qpu_circuit_measured(circuit);

    /* Probabilities, summed up in place: entry x of the first half of the
     * array takes the sum up to x, read from entries 2x and 2x+1. */
    cum = sampler->amp;
    for( uint64_t x = 0; x < dim; x++ ) {
        total += cum[2 * x] * cum[2 * x] + cum[2 * x + 1] * cum[2 * x + 1];
        cum[x] = total;
    }
    if( !(total > 0.0) || !isfinite(total) ) return PARSEC_ERROR;
    if( sampler_cancelled(cancel) ) return PARSEC_QPU_SAMPLER_CANCELLED;

    if( sampler->draws_len < shots ) {
        free(sampler->draws);
        sampler->draws     = (uint64_t*)malloc((size_t)shots * sizeof(uint64_t));
        sampler->draws_len = (NULL != sampler->draws) ? shots : 0;
        if( NULL == sampler->draws ) return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    rng_seed(&rng, seed, key);
    for( uint32_t k = 0; k < shots; k++ ) {
        if( (0 == (k & 0xfff)) && sampler_cancelled(cancel) ) return PARSEC_QPU_SAMPLER_CANCELLED;
        /* 53 random bits make a number in [0, 1). */
        const double u = (double)(rng_next(&rng) >> 11) * (1.0 / 9007199254740992.0) * total;
        uint64_t lo = 0, hi = dim - 1;
        /* The first state whose sum passes u. It has a probability above
         * zero, since the sum did not pass u before it. */
        while( lo < hi ) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if( cum[mid] > u ) hi = mid; else lo = mid + 1;
        }
        if( !(cum[lo] > u) ) {
            /* u rounded up to the total: take the last state that has a
             * probability, which is the first one where the sum is complete. */
            lo = 0; hi = dim - 1;
            while( lo < hi ) {
                const uint64_t mid = lo + (hi - lo) / 2;
                if( cum[mid] >= total ) hi = mid; else lo = mid + 1;
            }
        }
        sampler->draws[k] = lo & mask;
    }
    qsort(sampler->draws, shots, sizeof(uint64_t), cmp_u64);

    distinct = 1;
    for( uint32_t k = 1; k < shots; k++ ) distinct += (sampler->draws[k] != sampler->draws[k - 1]);

    result->status         = PARSEC_QPU_STATUS_OK;
    result->nb_qubits      = (uint16_t)circuit->nb_qubits;
    result->flags         &= (uint8_t)~PARSEC_QPU_RESULT_TRUNCATED;
    result->nb_entries     = 0;
    result->shots_returned = 0;

    if( distinct <= max_entries ) {
        for( uint32_t k = 0; k < shots; ) {
            uint32_t end = k + 1;
            while( (end < shots) && (sampler->draws[end] == sampler->draws[k]) ) end++;
            parsec_qpu_result_set_entry(result, result->nb_entries++, sampler->draws[k], end - k);
            k = end;
        }
        result->shots_returned = shots;
        return PARSEC_SUCCESS;
    }

    /* More outcomes than the buffer holds: keep the most frequent ones. */
    result->flags |= PARSEC_QPU_RESULT_TRUNCATED;
    if( 0 == max_entries ) return PARSEC_SUCCESS;
    if( sampler->runs_len < 2 * distinct ) {
        free(sampler->runs);
        sampler->runs     = (uint64_t*)malloc((size_t)(2 * distinct) * sizeof(uint64_t));
        sampler->runs_len = (NULL != sampler->runs) ? (size_t)(2 * distinct) : 0;
        if( NULL == sampler->runs ) return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    distinct = 0;
    for( uint32_t k = 0; k < shots; ) {
        uint32_t end = k + 1;
        while( (end < shots) && (sampler->draws[end] == sampler->draws[k]) ) end++;
        sampler->runs[2 * distinct]     = sampler->draws[k];
        sampler->runs[2 * distinct + 1] = end - k;
        distinct++;
        k = end;
    }
    qsort(sampler->runs, (size_t)distinct, 2 * sizeof(uint64_t), cmp_run_by_count);
    qsort(sampler->runs, max_entries, 2 * sizeof(uint64_t), cmp_u64);   /* the kept ones, by bitstring */
    for( uint32_t k = 0; k < max_entries; k++ ) {
        parsec_qpu_result_set_entry(result, k, sampler->runs[2 * k], (uint32_t)sampler->runs[2 * k + 1]);
        result->shots_returned += (uint32_t)sampler->runs[2 * k + 1];
    }
    result->nb_entries = max_entries;
    return PARSEC_SUCCESS;
}

int parsec_qpu_sampler_sample(const parsec_qpu_circuit_t *circuit, const double *params,
                              uint32_t shots, uint64_t seed, uint64_t key,
                              parsec_qpu_result_t *result, uint32_t max_entries)
{
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();
    int rc;

    if( NULL == sampler ) return PARSEC_ERR_OUT_OF_RESOURCE;
    rc = parsec_qpu_sampler_run(sampler, circuit, params, shots, seed, key, result, max_entries, NULL);
    parsec_qpu_sampler_free(sampler);
    return rc;
}
