/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * The exact sampler and the circuit format, without the runtime.
 *
 *   - What parsec_qpu_circuit_check() accepts and refuses.
 *   - Every operation against a second implementation that shares no code
 *     with the sampler: dense matrices built from the Pauli matrices and the
 *     conventions of device_qpu.h, applied by a sum over basis states.
 *   - States known in closed form, bit order included, and the expected cut
 *     of the one-edge QAOA circuit.
 *   - Sampling: frequencies against the exact probabilities, the same counts
 *     for the same (seed, key) whatever was sampled before, partial
 *     measurement, and results cut to the room of their buffer.
 *
 * Every random number comes from a fixed seed, so the test gives the same
 * verdict every time.
 *
 * Options:
 *   -t N      also time the sampling of a QAOA circuit on N qubits
 *   -p P      with P layers (default 2)
 *   -s SHOTS  and SHOTS shots (default 1024)
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nb_errors = 0;

#define EXPECT(COND, ...)                                                   \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            fprintf(stderr, "%s:%d: ", __func__, __LINE__);                 \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            nb_errors++;                                                    \
        }                                                                   \
    } while(0)

#define TOLERANCE 1e-12

/* A circuit in a buffer of its own. */
typedef struct {
    parsec_qpu_circuit_t *circuit;
    size_t room;
} test_circuit_t;

static test_circuit_t tc_new(uint32_t nb_qubits, uint32_t nb_params, uint32_t max_ops)
{
    test_circuit_t tc;
    tc.room    = sizeof(parsec_qpu_circuit_t) + (size_t)max_ops * PARSEC_QPU_OP_SIZE;
    tc.circuit = (parsec_qpu_circuit_t*)calloc(1, tc.room);
    parsec_qpu_circuit_init(tc.circuit, nb_qubits, nb_params);
    return tc;
}

static void tc_add(test_circuit_t *tc, int opcode, int q0, int q1, int param, double angle)
{
    if( 0 != parsec_qpu_circuit_add(tc->circuit, tc->room, opcode, q0, q1, param, angle) ) {
        fprintf(stderr, "test circuit full\n");
        exit(2);
    }
}

static void tc_measure_all(test_circuit_t *tc)
{
    if( 0 != parsec_qpu_circuit_measure_all(tc->circuit, tc->room) ) {
        fprintf(stderr, "test circuit full\n");
        exit(2);
    }
}

static uint64_t rng_state = 20261122;
static uint64_t rnd(void) { return rng_state = parsec_qpu_mix64(rng_state); }
static double rnd_angle(void) { return (parsec_qpu_unit(rnd()) - 0.5) * 12.0; }

/* ------------------------------------------------------------------------
 * The reference: dense matrices, applied by the definition.
 * --------------------------------------------------------------------- */

typedef double complex cplx;

static const cplx PAULI[4][2][2] = {
    { { 1, 0 }, { 0,  1 } },            /* I */
    { { 0, 1 }, { 1,  0 } },            /* X */
    { { 0, -I }, { I, 0 } },            /* Y */
    { { 1, 0 }, { 0, -1 } }             /* Z */
};

/* new[x] = sum over b of u[bit q of x][b] * old[x with bit q set to b] */
static void ref_apply1(cplx *state, uint32_t n, uint32_t q, cplx u[2][2])
{
    const uint64_t dim = ((uint64_t)1) << n, bit = ((uint64_t)1) << q;
    cplx *old = (cplx*)malloc(dim * sizeof(cplx));

    memcpy(old, state, dim * sizeof(cplx));
    for( uint64_t x = 0; x < dim; x++ ) {
        const int row = (x & bit) ? 1 : 0;
        state[x] = u[row][0] * old[x & ~bit] + u[row][1] * old[x | bit];
    }
    free(old);
}

/* The same for two qubits; a row or column index is (bit of q0) + 2 (bit of q1). */
static void ref_apply2(cplx *state, uint32_t n, uint32_t q0, uint32_t q1, cplx u[4][4])
{
    const uint64_t dim = ((uint64_t)1) << n, b0 = ((uint64_t)1) << q0, b1 = ((uint64_t)1) << q1;
    cplx *old = (cplx*)malloc(dim * sizeof(cplx));

    memcpy(old, state, dim * sizeof(cplx));
    for( uint64_t x = 0; x < dim; x++ ) {
        const int row = ((x & b0) ? 1 : 0) + ((x & b1) ? 2 : 0);
        const uint64_t base = x & ~(b0 | b1);
        cplx sum = 0;
        for( int col = 0; col < 4; col++ ) {
            sum += u[row][col] * old[base | ((col & 1) ? b0 : 0) | ((col & 2) ? b1 : 0)];
        }
        state[x] = sum;
    }
    free(old);
}

static void ref_apply(cplx *state, uint32_t n, const parsec_qpu_op_t *op, const double *params)
{
    const double t = (PARSEC_QPU_NO_PARAM == op->param) ? op->angle : op->angle * params[op->param];
    cplx u1[2][2], u2[4][4];
    int pauli = 0;

    memset(u1, 0, sizeof(u1));
    memset(u2, 0, sizeof(u2));
    switch( op->opcode ) {
    case PARSEC_QPU_OP_H:
        u1[0][0] = u1[0][1] = u1[1][0] = 1 / sqrt(2.0);
        u1[1][1] = -1 / sqrt(2.0);
        ref_apply1(state, n, op->q0, u1);
        break;
    case PARSEC_QPU_OP_X: case PARSEC_QPU_OP_Y: case PARSEC_QPU_OP_Z:
        pauli = 1 + (op->opcode - PARSEC_QPU_OP_X);
        memcpy(u1, PAULI[pauli], sizeof(u1));
        ref_apply1(state, n, op->q0, u1);
        break;
    case PARSEC_QPU_OP_RX: case PARSEC_QPU_OP_RY: case PARSEC_QPU_OP_RZ:
        /* exp(-i t P / 2) = cos(t/2) I - i sin(t/2) P */
        pauli = 1 + (op->opcode - PARSEC_QPU_OP_RX);
        for( int r = 0; r < 2; r++ ) for( int c = 0; c < 2; c++ ) {
            u1[r][c] = cos(t / 2) * PAULI[0][r][c] - I * sin(t / 2) * PAULI[pauli][r][c];
        }
        ref_apply1(state, n, op->q0, u1);
        break;
    case PARSEC_QPU_OP_CX:
        /* |b0 b1> -> |b0, b1 xor b0>: q0 controls, q1 is flipped. */
        for( int col = 0; col < 4; col++ ) {
            const int b0 = col & 1, b1 = (col >> 1) & 1;
            u2[b0 + 2 * (b1 ^ b0)][col] = 1;
        }
        ref_apply2(state, n, op->q0, op->q1, u2);
        break;
    case PARSEC_QPU_OP_CZ:
        for( int col = 0; col < 4; col++ ) u2[col][col] = (3 == col) ? -1 : 1;
        ref_apply2(state, n, op->q0, op->q1, u2);
        break;
    case PARSEC_QPU_OP_RXX: case PARSEC_QPU_OP_RYY: case PARSEC_QPU_OP_RZZ:
        /* exp(-i t PP / 2) = cos(t/2) II - i sin(t/2) PP, with
         * PP[(r0, r1), (c0, c1)] = P[r0][c0] P[r1][c1]. */
        pauli = 1 + (op->opcode - PARSEC_QPU_OP_RXX);
        for( int row = 0; row < 4; row++ ) for( int col = 0; col < 4; col++ ) {
            const cplx pp = PAULI[pauli][row & 1][col & 1] * PAULI[pauli][row >> 1][col >> 1];
            u2[row][col] = ((row == col) ? cos(t / 2) : 0) - I * sin(t / 2) * pp;
        }
        ref_apply2(state, n, op->q0, op->q1, u2);
        break;
    default:
        break;
    }
}

static cplx *ref_state(const parsec_qpu_circuit_t *circuit, const double *params)
{
    const uint64_t dim = ((uint64_t)1) << circuit->nb_qubits;
    const parsec_qpu_op_t *ops = parsec_qpu_circuit_ops(circuit);
    cplx *state = (cplx*)calloc(dim, sizeof(cplx));

    state[0] = 1;
    for( uint32_t i = 0; i < circuit->nb_ops; i++ ) ref_apply(state, circuit->nb_qubits, &ops[i], params);
    return state;
}

/* Largest distance between an amplitude of the sampler and of the reference. */
static double compare_with_reference(parsec_qpu_sampler_t *sampler, const parsec_qpu_circuit_t *circuit,
                                     const double *params)
{
    const uint64_t dim = ((uint64_t)1) << circuit->nb_qubits;
    const double *amp = NULL;
    double worst = 0.0;
    cplx *ref;
    int rc;

    rc = parsec_qpu_sampler_state(sampler, circuit, params, NULL, &amp);
    if( (PARSEC_SUCCESS != rc) || (NULL == amp) ) return 1e300;
    ref = ref_state(circuit, params);
    for( uint64_t x = 0; x < dim; x++ ) {
        const double d = cabs((amp[2 * x] + I * amp[2 * x + 1]) - ref[x]);
        if( d > worst ) worst = d;
    }
    free(ref);
    return worst;
}

/* ------------------------------------------------------------------------
 * Circuit checks
 * --------------------------------------------------------------------- */

static void test_circuit_check(void)
{
    test_circuit_t tc = tc_new(3, 2, 16);
    parsec_qpu_op_t *ops = parsec_qpu_circuit_ops(tc.circuit);
    const char *why = "untouched";
    parsec_qpu_circuit_t saved;
    parsec_qpu_op_t op;

    tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    tc_add(&tc, PARSEC_QPU_OP_RZZ, 0, 2, 1, -0.5);
    tc_add(&tc, PARSEC_QPU_OP_RX, 1, 0, 0, 2.0);
    tc_add(&tc, PARSEC_QPU_OP_CX, 2, 1, PARSEC_QPU_NO_PARAM, 0.0);
    tc_measure_all(&tc);

    EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(tc.circuit, tc.room, &why), "a good circuit is refused: %s", why);
    EXPECT(NULL == why, "a reason is given for a good circuit");
    EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(tc.circuit, 0, NULL), "a good circuit of unknown size is refused");
    EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(tc.circuit, parsec_qpu_circuit_size(tc.circuit), NULL),
           "a circuit that fills its buffer exactly is refused");
    EXPECT(0x7 == parsec_qpu_circuit_measured(tc.circuit), "the measured qubits are %llx",
           (unsigned long long)parsec_qpu_circuit_measured(tc.circuit));

#define REFUSED(WHAT)                                                                             \
    do {                                                                                          \
        why = NULL;                                                                               \
        EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_circuit_check(tc.circuit, tc.room, &why),       \
               "accepted: %s", WHAT);                                                             \
        EXPECT(NULL != why, "no reason given: %s", WHAT);                                         \
    } while(0)

    saved = *tc.circuit;
    tc.circuit->version = 99;             REFUSED("an unknown version");            *tc.circuit = saved;
    tc.circuit->nb_qubits = 0;            REFUSED("no qubit");                      *tc.circuit = saved;
    tc.circuit->nb_qubits = 65;           REFUSED("65 qubits");                     *tc.circuit = saved;
    tc.circuit->nb_ops = 17;              REFUSED("more operations than the buffer holds"); *tc.circuit = saved;
    tc.circuit->nb_ops = 4;               REFUSED("no measurement");                *tc.circuit = saved;
    tc.circuit->nb_params = 40000;        REFUSED("too many parameter slots");      *tc.circuit = saved;

    op = ops[0]; ops[0].opcode = 0;                    REFUSED("opcode 0");                     ops[0] = op;
    op = ops[0]; ops[0].opcode = PARSEC_QPU_OP_MAX;    REFUSED("an opcode past the last");      ops[0] = op;
    op = ops[0]; ops[0].q0 = 3;                        REFUSED("a qubit that does not exist");  ops[0] = op;
    op = ops[1]; ops[1].q1 = 3;                        REFUSED("a second qubit that does not exist"); ops[1] = op;
    op = ops[1]; ops[1].q1 = ops[1].q0;                REFUSED("twice the same qubit");         ops[1] = op;
    op = ops[1]; ops[1].param = 2;                     REFUSED("a parameter slot that does not exist"); ops[1] = op;
    op = ops[1]; ops[1].param = -2;                    REFUSED("a negative parameter slot");    ops[1] = op;
    op = ops[2]; ops[2].angle = NAN;                   REFUSED("an angle that is not a number"); ops[2] = op;
    op = ops[2]; ops[2].angle = INFINITY;              REFUSED("an infinite angle");            ops[2] = op;
    /* The last operation becomes a gate on a qubit measured just before. */
    op = ops[6]; ops[6].opcode = PARSEC_QPU_OP_X; ops[6].q0 = 0;   REFUSED("a gate after a measurement");       ops[6] = op;
    op = ops[6]; ops[6].q0 = 1;                        REFUSED("a qubit measured twice");       ops[6] = op;
    op = ops[6]; ops[6].opcode = PARSEC_QPU_OP_CZ; ops[6].q0 = 2; ops[6].q1 = 0;
                                                       REFUSED("a two-qubit gate after the measurement of its second qubit"); ops[6] = op;
#undef REFUSED

    EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_circuit_check(NULL, 0, NULL), "no circuit is accepted");
    EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_circuit_check(tc.circuit, 8, NULL), "a buffer shorter than a header is accepted");
    EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(tc.circuit, tc.room, NULL), "the circuit was not restored");

    /* An empty job is a header and nothing else that matters. */
    tc.circuit->flags |= PARSEC_QPU_CIRCUIT_EMPTY;
    tc.circuit->nb_qubits = 0;
    ops[0].opcode = 0;
    EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(tc.circuit, tc.room, NULL), "an empty job is refused");
    EXPECT(0 == parsec_qpu_circuit_measured(tc.circuit), "an empty job measures something");

    /* The buffer cannot take more operations than it has room for. */
    {
        test_circuit_t small = tc_new(2, 0, 2);
        EXPECT(0 == parsec_qpu_circuit_add(small.circuit, small.room, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0), "first add");
        EXPECT(0 == parsec_qpu_circuit_add(small.circuit, small.room, PARSEC_QPU_OP_H, 1, 0, PARSEC_QPU_NO_PARAM, 0), "second add");
        EXPECT(0 != parsec_qpu_circuit_add(small.circuit, small.room, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0), "a third operation fit in room for two");
        EXPECT(2 == small.circuit->nb_ops, "%u operations after a refused add", small.circuit->nb_ops);
        free(small.circuit);
    }
    free(tc.circuit);
}

/* ------------------------------------------------------------------------
 * Gates against the reference
 * --------------------------------------------------------------------- */

static const int ONE_QUBIT[]  = { PARSEC_QPU_OP_H, PARSEC_QPU_OP_X, PARSEC_QPU_OP_Y, PARSEC_QPU_OP_Z,
                                  PARSEC_QPU_OP_RX, PARSEC_QPU_OP_RY, PARSEC_QPU_OP_RZ };
static const int TWO_QUBITS[] = { PARSEC_QPU_OP_CX, PARSEC_QPU_OP_CZ, PARSEC_QPU_OP_RXX, PARSEC_QPU_OP_RYY,
                                  PARSEC_QPU_OP_RZZ };

/* Something that puts weight and phase on every basis state, so that no
 * error of a gate hides behind an amplitude that is zero. */
static void add_scramble(test_circuit_t *tc)
{
    const int n = (int)tc->circuit->nb_qubits;
    for( int q = 0; q < n; q++ ) {
        tc_add(tc, PARSEC_QPU_OP_RY, q, 0, PARSEC_QPU_NO_PARAM, rnd_angle());
        tc_add(tc, PARSEC_QPU_OP_RZ, q, 0, PARSEC_QPU_NO_PARAM, rnd_angle());
    }
    for( int q = 0; q + 1 < n; q++ ) {
        tc_add(tc, PARSEC_QPU_OP_RXX, q, q + 1, PARSEC_QPU_NO_PARAM, rnd_angle());
        tc_add(tc, PARSEC_QPU_OP_RX, q, 0, PARSEC_QPU_NO_PARAM, rnd_angle());
    }
}

static void test_each_gate(parsec_qpu_sampler_t *sampler)
{
    const int n = 4;
    double worst = 0.0, d;

    /* Each one-qubit operation on each qubit, each two-qubit operation on
     * each ordered pair of qubits, after a scramble. */
    for( size_t g = 0; g < sizeof(ONE_QUBIT) / sizeof(int); g++ ) {
        for( int q = 0; q < n; q++ ) {
            test_circuit_t tc = tc_new(n, 0, 64);
            add_scramble(&tc);
            tc_add(&tc, ONE_QUBIT[g], q, 0, PARSEC_QPU_NO_PARAM, rnd_angle());
            tc_measure_all(&tc);
            d = compare_with_reference(sampler, tc.circuit, NULL);
            EXPECT(d < TOLERANCE, "opcode %d on qubit %d is %g away from the reference", ONE_QUBIT[g], q, d);
            if( d > worst ) worst = d;
            free(tc.circuit);
        }
    }
    for( size_t g = 0; g < sizeof(TWO_QUBITS) / sizeof(int); g++ ) {
        for( int q0 = 0; q0 < n; q0++ ) for( int q1 = 0; q1 < n; q1++ ) {
            test_circuit_t tc;
            if( q0 == q1 ) continue;
            tc = tc_new(n, 0, 64);
            add_scramble(&tc);
            tc_add(&tc, TWO_QUBITS[g], q0, q1, PARSEC_QPU_NO_PARAM, rnd_angle());
            tc_measure_all(&tc);
            d = compare_with_reference(sampler, tc.circuit, NULL);
            EXPECT(d < TOLERANCE, "opcode %d on qubits %d and %d is %g away from the reference", TWO_QUBITS[g], q0, q1, d);
            if( d > worst ) worst = d;
            free(tc.circuit);
        }
    }
    printf("each operation against the reference: largest difference %.2e\n", worst);
}

static void test_random_circuits(parsec_qpu_sampler_t *sampler)
{
    double worst = 0.0, worst_norm = 0.0;

    for( int trial = 0; trial < 300; trial++ ) {
        const int n = 1 + (int)(rnd() % 6);
        const int nb_ops = 1 + (int)(rnd() % 40);
        test_circuit_t tc = tc_new(n, 3, 64);
        double params[3] = { rnd_angle(), rnd_angle(), rnd_angle() };
        const double *amp = NULL;
        double d, norm = 0.0;

        for( int i = 0; i < nb_ops; i++ ) {
            const int with_param = (0 == rnd() % 3);
            if( (n >= 2) && (rnd() % 2) ) {
                const int q0 = (int)(rnd() % n);
                int q1 = (int)(rnd() % (n - 1));
                if( q1 >= q0 ) q1++;
                tc_add(&tc, TWO_QUBITS[rnd() % 5], q0, q1, with_param ? (int)(rnd() % 3) : PARSEC_QPU_NO_PARAM, rnd_angle());
            } else {
                tc_add(&tc, ONE_QUBIT[rnd() % 7], (int)(rnd() % n), 0,
                       with_param ? (int)(rnd() % 3) : PARSEC_QPU_NO_PARAM, rnd_angle());
            }
        }
        tc_measure_all(&tc);

        d = compare_with_reference(sampler, tc.circuit, params);
        EXPECT(d < TOLERANCE, "random circuit %d (%d qubits, %d operations) is %g away from the reference", trial, n, nb_ops, d);
        if( d > worst ) worst = d;

        (void)parsec_qpu_sampler_state(sampler, tc.circuit, params, NULL, &amp);
        for( uint64_t x = 0; x < (((uint64_t)1) << n); x++ ) norm += amp[2 * x] * amp[2 * x] + amp[2 * x + 1] * amp[2 * x + 1];
        EXPECT(fabs(norm - 1.0) < TOLERANCE, "random circuit %d has norm %.15f", trial, norm);
        if( fabs(norm - 1.0) > worst_norm ) worst_norm = fabs(norm - 1.0);
        free(tc.circuit);
    }
    printf("300 random circuits against the reference: largest difference %.2e, norm off by at most %.2e\n",
           worst, worst_norm);
}

/* ------------------------------------------------------------------------
 * States known in closed form
 * --------------------------------------------------------------------- */

static void expect_state(parsec_qpu_sampler_t *sampler, test_circuit_t *tc, const double *params,
                         const cplx *expected, const char *what)
{
    const uint64_t dim = ((uint64_t)1) << tc->circuit->nb_qubits;
    const double *amp = NULL;
    int rc = parsec_qpu_sampler_state(sampler, tc->circuit, params, NULL, &amp);

    EXPECT(PARSEC_SUCCESS == rc, "%s: the sampler returned %d", what, rc);
    if( PARSEC_SUCCESS != rc ) return;
    for( uint64_t x = 0; x < dim; x++ ) {
        const double d = cabs((amp[2 * x] + I * amp[2 * x + 1]) - expected[x]);
        EXPECT(d < TOLERANCE, "%s: amplitude %llu is (%.15f, %.15f), expected (%.15f, %.15f)", what,
               (unsigned long long)x, amp[2 * x], amp[2 * x + 1], creal(expected[x]), cimag(expected[x]));
    }
}

static void test_analytic(parsec_qpu_sampler_t *sampler)
{
    const double r = 1 / sqrt(2.0), t = 0.73;
    test_circuit_t tc;
    cplx e[32];

    /* H|0> */
    tc = tc_new(1, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0); tc_measure_all(&tc);
    e[0] = r; e[1] = r;
    expect_state(sampler, &tc, NULL, e, "H|0>");
    free(tc.circuit);

    /* The rotations of |0>, and of |+> about Z. */
    tc = tc_new(1, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_RX, 0, 0, PARSEC_QPU_NO_PARAM, t); tc_measure_all(&tc);
    e[0] = cos(t / 2); e[1] = -I * sin(t / 2);
    expect_state(sampler, &tc, NULL, e, "RX|0>");
    free(tc.circuit);

    tc = tc_new(1, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_RY, 0, 0, PARSEC_QPU_NO_PARAM, t); tc_measure_all(&tc);
    e[0] = cos(t / 2); e[1] = sin(t / 2);
    expect_state(sampler, &tc, NULL, e, "RY|0>");
    free(tc.circuit);

    tc = tc_new(1, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_RZ, 0, 0, PARSEC_QPU_NO_PARAM, t); tc_measure_all(&tc);
    e[0] = r * cexp(-I * t / 2); e[1] = r * cexp(I * t / 2);
    expect_state(sampler, &tc, NULL, e, "RZ|+>");
    free(tc.circuit);

    /* Y|0> = i|1> */
    tc = tc_new(1, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_Y, 0, 0, PARSEC_QPU_NO_PARAM, 0); tc_measure_all(&tc);
    e[0] = 0; e[1] = I;
    expect_state(sampler, &tc, NULL, e, "Y|0>");
    free(tc.circuit);

    /* Bit order: X on qubit 3 of 5 is basis state 8. */
    tc = tc_new(5, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_X, 3, 0, PARSEC_QPU_NO_PARAM, 0); tc_measure_all(&tc);
    memset(e, 0, sizeof(e)); e[8] = 1;
    expect_state(sampler, &tc, NULL, e, "X on qubit 3");
    free(tc.circuit);

    /* Bell state, and the direction of CX: q0 controls. */
    tc = tc_new(2, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0); tc_measure_all(&tc);
    memset(e, 0, sizeof(e)); e[0] = r; e[3] = r;
    expect_state(sampler, &tc, NULL, e, "Bell");
    free(tc.circuit);

    tc = tc_new(2, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_X, 0, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0); tc_measure_all(&tc);
    memset(e, 0, sizeof(e)); e[3] = 1;
    expect_state(sampler, &tc, NULL, e, "CX with its control at 1");
    free(tc.circuit);

    tc = tc_new(2, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_X, 1, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0); tc_measure_all(&tc);
    memset(e, 0, sizeof(e)); e[2] = 1;
    expect_state(sampler, &tc, NULL, e, "CX with its control at 0");
    free(tc.circuit);

    /* GHZ on five qubits. */
    tc = tc_new(5, 0, 16);
    tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0);
    for( int q = 0; q < 4; q++ ) tc_add(&tc, PARSEC_QPU_OP_CX, q, q + 1, PARSEC_QPU_NO_PARAM, 0);
    tc_measure_all(&tc);
    memset(e, 0, sizeof(e)); e[0] = r; e[31] = r;
    expect_state(sampler, &tc, NULL, e, "GHZ");
    free(tc.circuit);

    /* RZZ on |++>: one phase where the qubits agree, the opposite where they differ. */
    tc = tc_new(2, 0, 8);
    tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_H, 1, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_RZZ, 0, 1, PARSEC_QPU_NO_PARAM, t); tc_measure_all(&tc);
    e[0] = e[3] = 0.5 * cexp(-I * t / 2); e[1] = e[2] = 0.5 * cexp(I * t / 2);
    expect_state(sampler, &tc, NULL, e, "RZZ|++>");
    free(tc.circuit);

    /* A parameter slot with a coefficient is the rotation by their product. */
    {
        test_circuit_t a = tc_new(2, 2, 8), b = tc_new(2, 0, 8);
        const double params[2] = { 0.4, -1.3 };
        const double *amp = NULL;
        double first[8];

        tc_add(&a, PARSEC_QPU_OP_RX, 0, 0, 1, 2.5);
        tc_add(&a, PARSEC_QPU_OP_RYY, 0, 1, 0, -0.75); tc_measure_all(&a);
        tc_add(&b, PARSEC_QPU_OP_RX, 0, 0, PARSEC_QPU_NO_PARAM, 2.5 * -1.3);
        tc_add(&b, PARSEC_QPU_OP_RYY, 0, 1, PARSEC_QPU_NO_PARAM, -0.75 * 0.4); tc_measure_all(&b);
        (void)parsec_qpu_sampler_state(sampler, a.circuit, params, NULL, &amp);
        memcpy(first, amp, sizeof(first));
        (void)parsec_qpu_sampler_state(sampler, b.circuit, NULL, NULL, &amp);
        EXPECT(0 == memcmp(first, amp, sizeof(first)), "a parameter with a coefficient is not the rotation by their product");
        EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_sampler_state(sampler, a.circuit, NULL, NULL, &amp),
               "a circuit with parameter slots ran without parameter values");
        free(a.circuit); free(b.circuit);
    }

    /* The QAOA circuit of one edge of weight w, as section 3.3 of the design
     * document writes it: H on both qubits, RZZ(-gamma w), RX(2 beta) on
     * both. The expected cut is (w / 2) (1 + sin(4 beta) sin(gamma w)). */
    for( int trial = 0; trial < 20; trial++ ) {
        const double w = 0.125 * (1 + (double)(rnd() % 24)), gamma = rnd_angle(), beta = rnd_angle();
        const double params[2] = { gamma, beta };
        const double *amp = NULL;
        double cut = 0.0, expected = 0.5 * w * (1 + sin(4 * beta) * sin(gamma * w));

        tc = tc_new(2, 2, 16);
        tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0);
        tc_add(&tc, PARSEC_QPU_OP_H, 1, 0, PARSEC_QPU_NO_PARAM, 0);
        tc_add(&tc, PARSEC_QPU_OP_RZZ, 0, 1, 0, -w);
        tc_add(&tc, PARSEC_QPU_OP_RX, 0, 0, 1, 2.0);
        tc_add(&tc, PARSEC_QPU_OP_RX, 1, 0, 1, 2.0);
        tc_measure_all(&tc);
        (void)parsec_qpu_sampler_state(sampler, tc.circuit, params, NULL, &amp);
        /* The edge is cut in the basis states 01 and 10. */
        for( int x = 1; x <= 2; x++ ) cut += w * (amp[2 * x] * amp[2 * x] + amp[2 * x + 1] * amp[2 * x + 1]);
        EXPECT(fabs(cut - expected) < TOLERANCE, "one-edge QAOA with w=%g gamma=%g beta=%g: expected cut %.15f, closed form %.15f",
               w, gamma, beta, cut, expected);
        free(tc.circuit);
    }
}

/* ------------------------------------------------------------------------
 * Sampling
 * --------------------------------------------------------------------- */

typedef struct {
    parsec_qpu_result_t *result;
    uint32_t max_entries;
} test_result_t;

static test_result_t tr_new(uint32_t max_entries)
{
    test_result_t tr;
    tr.max_entries = max_entries;
    tr.result = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(max_entries));
    return tr;
}

static int same_counts(const parsec_qpu_result_t *a, const parsec_qpu_result_t *b)
{
    return (a->nb_entries == b->nb_entries) && (a->shots_returned == b->shots_returned) &&
           (0 == memcmp(a + 1, b + 1, (size_t)a->nb_entries * PARSEC_QPU_ENTRY_SIZE));
}

static void test_sampling(parsec_qpu_sampler_t *sampler)
{
    const uint32_t shots = 200000;
    test_circuit_t tc = tc_new(3, 0, 64);
    test_result_t a = tr_new(8), b = tr_new(8);
    const double *amp = NULL;
    double p[8], worst = 0.0;
    uint64_t bits, total = 0, previous = 0;
    uint32_t count;
    int rc;

    add_scramble(&tc);
    add_scramble(&tc);
    tc_measure_all(&tc);
    (void)parsec_qpu_sampler_state(sampler, tc.circuit, NULL, NULL, &amp);
    for( int x = 0; x < 8; x++ ) p[x] = amp[2 * x] * amp[2 * x] + amp[2 * x + 1] * amp[2 * x + 1];

    rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, shots, 7, 1234, a.result, a.max_entries, NULL);
    EXPECT(PARSEC_SUCCESS == rc, "sampling returned %d", rc);
    EXPECT(PARSEC_QPU_STATUS_OK == a.result->status, "status %u", (unsigned)a.result->status);
    EXPECT(3 == a.result->nb_qubits, "%u qubits", (unsigned)a.result->nb_qubits);
    EXPECT(shots == a.result->shots_returned, "%u shots returned", a.result->shots_returned);
    EXPECT(0 == (a.result->flags & PARSEC_QPU_RESULT_TRUNCATED), "flagged as truncated");

    /* Frequencies within six standard deviations of the probabilities. */
    for( uint32_t i = 0; i < a.result->nb_entries; i++ ) {
        double sigma, z;
        parsec_qpu_result_get_entry(a.result, i, &bits, &count);
        EXPECT((0 == i) || (bits > previous), "entry %u is not after the one before", i);
        EXPECT(bits < 8, "entry %u is %llx", i, (unsigned long long)bits);
        if( bits >= 8 ) continue;
        previous = bits;
        total += count;
        sigma = sqrt(p[bits] * (1 - p[bits]) / shots);
        z = fabs((double)count / shots - p[bits]) / sigma;
        EXPECT(z < 6.0, "outcome %llu: frequency %.5f, probability %.5f, %.1f standard deviations away",
               (unsigned long long)bits, (double)count / shots, p[bits], z);
        if( z > worst ) worst = z;
        p[bits] = -1.0;     /* seen */
    }
    EXPECT(total == shots, "the counts add up to %llu", (unsigned long long)total);
    for( int x = 0; x < 8; x++ ) {
        /* An outcome that never came out has to be one that hardly could. */
        EXPECT((p[x] < 0) || (p[x] * shots < 36.0), "outcome %d of probability %.5f never came out", x, p[x]);
    }
    printf("sampling: %u shots, every frequency within %.2f standard deviations of its probability\n", shots, worst);

    /* The same (seed, key) gives the same counts, from a sampler that did
     * something else in between and from a new one. */
    {
        test_circuit_t other = tc_new(6, 0, 64);
        test_result_t scratch = tr_new(64);
        add_scramble(&other); tc_measure_all(&other);
        (void)parsec_qpu_sampler_run(sampler, other.circuit, NULL, 5000, 3, 99, scratch.result, scratch.max_entries, NULL);
        free(other.circuit); free(scratch.result);
    }
    (void)parsec_qpu_sampler_run(sampler, tc.circuit, NULL, shots, 7, 1234, b.result, b.max_entries, NULL);
    EXPECT(same_counts(a.result, b.result), "the same seed and key gave other counts after an unrelated run");
    memset(b.result, 0, parsec_qpu_result_size(8));
    rc = parsec_qpu_sampler_sample(tc.circuit, NULL, shots, 7, 1234, b.result, b.max_entries);
    EXPECT((PARSEC_SUCCESS == rc) && same_counts(a.result, b.result), "the same seed and key gave other counts from a new sampler");
    /* Another key, or another seed, is another draw. */
    (void)parsec_qpu_sampler_run(sampler, tc.circuit, NULL, shots, 7, 1235, b.result, b.max_entries, NULL);
    EXPECT(!same_counts(a.result, b.result), "another key gave the same counts");
    (void)parsec_qpu_sampler_run(sampler, tc.circuit, NULL, shots, 8, 1234, b.result, b.max_entries, NULL);
    EXPECT(!same_counts(a.result, b.result), "another seed gave the same counts");

    /* Cut to the room of the buffer: the most frequent outcomes are kept. */
    {
        test_result_t cut = tr_new(3);
        uint32_t kept = 0, smallest_kept = UINT32_MAX, in_full = 0;
        cut.result->flags = PARSEC_QPU_RESULT_MOCK;
        rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, shots, 7, 1234, cut.result, 3, NULL);
        EXPECT(PARSEC_SUCCESS == rc, "sampling into a short buffer returned %d", rc);
        EXPECT(cut.result->flags & PARSEC_QPU_RESULT_TRUNCATED, "a result that was cut is not flagged");
        EXPECT(cut.result->flags & PARSEC_QPU_RESULT_MOCK, "the sampler cleared a flag that is not its own");
        EXPECT(3 == cut.result->nb_entries, "%u entries in room for 3", cut.result->nb_entries);
        previous = 0;
        for( uint32_t i = 0; i < cut.result->nb_entries; i++ ) {
            uint64_t fbits; uint32_t fcount;
            parsec_qpu_result_get_entry(cut.result, i, &bits, &count);
            EXPECT((0 == i) || (bits > previous), "kept entry %u is not after the one before", i);
            previous = bits;
            kept += count;
            if( count < smallest_kept ) smallest_kept = count;
            for( uint32_t j = 0; j < a.result->nb_entries; j++ ) {
                parsec_qpu_result_get_entry(a.result, j, &fbits, &fcount);
                if( fbits == bits ) { EXPECT(fcount == count, "kept entry %u has another count than in the full result", i); in_full++; }
            }
        }
        EXPECT(3 == in_full, "%u of the kept entries are in the full result", in_full);
        EXPECT(kept == cut.result->shots_returned, "shots_returned is %u, the kept entries count %u", cut.result->shots_returned, kept);
        /* No more than two outcomes of the full result beat the weakest kept one. */
        {
            uint32_t larger = 0;
            for( uint32_t j = 0; j < a.result->nb_entries; j++ ) {
                parsec_qpu_result_get_entry(a.result, j, &bits, &count);
                if( count > smallest_kept ) larger++;
            }
            EXPECT(larger <= 2, "%u outcomes are more frequent than the least frequent one that was kept", larger);
        }
        rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, shots, 7, 1234, cut.result, 0, NULL);
        EXPECT((PARSEC_SUCCESS == rc) && (0 == cut.result->nb_entries) && (0 == cut.result->shots_returned) &&
               (cut.result->flags & PARSEC_QPU_RESULT_TRUNCATED), "sampling into no room at all");
        free(cut.result);
    }
    free(tc.circuit);

    /* Measuring some of the qubits: the others read zero. A GHZ state
     * measured on qubits 0 and 2 gives 000 or 101. */
    tc = tc_new(3, 0, 16);
    tc_add(&tc, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_CX, 1, 2, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_MEASURE, 0, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_MEASURE, 2, 0, PARSEC_QPU_NO_PARAM, 0);
    rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, 10000, 1, 1, a.result, a.max_entries, NULL);
    EXPECT((PARSEC_SUCCESS == rc) && (2 == a.result->nb_entries), "GHZ measured on two qubits: rc %d, %u entries", rc, a.result->nb_entries);
    if( 2 == a.result->nb_entries ) {
        uint32_t c0, c1;
        uint64_t b0, b1;
        parsec_qpu_result_get_entry(a.result, 0, &b0, &c0);
        parsec_qpu_result_get_entry(a.result, 1, &b1, &c1);
        EXPECT((0 == b0) && (5 == b1), "GHZ measured on qubits 0 and 2 gave %llx and %llx", (unsigned long long)b0, (unsigned long long)b1);
        EXPECT(10000 == c0 + c1, "the two outcomes count %u shots", c0 + c1);
        EXPECT(fabs((double)c0 - 5000.0) < 6 * 50.0, "outcome 000 came out %u times in 10000", c0);
    }
    free(tc.circuit);

    /* A state that is one basis state: every shot gives it. */
    tc = tc_new(4, 0, 16);
    tc_add(&tc, PARSEC_QPU_OP_X, 1, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_add(&tc, PARSEC_QPU_OP_X, 3, 0, PARSEC_QPU_NO_PARAM, 0);
    tc_measure_all(&tc);
    rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, 777, 1, 1, a.result, a.max_entries, NULL);
    parsec_qpu_result_get_entry(a.result, 0, &bits, &count);
    EXPECT((PARSEC_SUCCESS == rc) && (1 == a.result->nb_entries) && (10 == bits) && (777 == count),
           "X on qubits 1 and 3: rc %d, %u entries, first is %llx with %u shots", rc, a.result->nb_entries,
           (unsigned long long)bits, count);

    /* What the sampler refuses, and giving up when asked to. */
    {
        int32_t cancel = 1;
        rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, 10, 1, 1, a.result, a.max_entries, &cancel);
        EXPECT(PARSEC_QPU_SAMPLER_CANCELLED == rc, "a cancelled run returned %d", rc);
        cancel = 0;
        rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, 10, 1, 1, a.result, a.max_entries, &cancel);
        EXPECT(PARSEC_SUCCESS == rc, "a run that was not cancelled returned %d", rc);
        rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, 0, 1, 1, a.result, a.max_entries, NULL);
        EXPECT(PARSEC_ERR_BAD_PARAM == rc, "no shot: %d", rc);
        tc.circuit->nb_qubits = PARSEC_QPU_SAMPLER_MAX_QUBITS + 1;
        rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, 10, 1, 1, a.result, a.max_entries, NULL);
        EXPECT(PARSEC_ERR_VALUE_OUT_OF_BOUNDS == rc, "a register too wide for the sampler: %d", rc);
        tc.circuit->nb_qubits = 4;
        tc.circuit->flags |= PARSEC_QPU_CIRCUIT_EMPTY;
        rc = parsec_qpu_sampler_run(sampler, tc.circuit, NULL, 10, 1, 1, a.result, a.max_entries, NULL);
        EXPECT(PARSEC_ERR_BAD_PARAM == rc, "an empty job was sampled: %d", rc);
    }
    free(tc.circuit);
    free(a.result);
    free(b.result);
}

/* ------------------------------------------------------------------------
 * Timing: a QAOA circuit on a random graph of n vertices
 * --------------------------------------------------------------------- */

static void time_qaoa(int n, int layers, int shots)
{
    const int m = (7 * n) / 4;
    test_circuit_t tc = tc_new(n, 2 * layers, (uint32_t)(n + layers * (m + 2 * n) + n));
    test_result_t tr = tr_new((uint32_t)shots);
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();
    double *params = (double*)calloc(2 * layers, sizeof(double));
    int64_t t0, t1, t2;
    int rc;

    for( int q = 0; q < n; q++ ) tc_add(&tc, PARSEC_QPU_OP_H, q, 0, PARSEC_QPU_NO_PARAM, 0);
    for( int l = 0; l < layers; l++ ) {
        for( int e = 0; e < m; e++ ) {
            const int i = (int)(rnd() % n);
            int j = (int)(rnd() % (n - 1));
            if( j >= i ) j++;
            tc_add(&tc, PARSEC_QPU_OP_RZZ, i, j, 2 * l, -0.125 * (1 + (double)(rnd() % 16)));
        }
        for( int q = 0; q < n; q++ ) tc_add(&tc, PARSEC_QPU_OP_RZ, q, 0, 2 * l, -0.25 * (1 + (double)(rnd() % 8)));
        for( int q = 0; q < n; q++ ) tc_add(&tc, PARSEC_QPU_OP_RX, q, 0, 2 * l + 1, 2.0);
        params[2 * l] = 0.4 + 0.1 * l;
        params[2 * l + 1] = 0.3 - 0.05 * l;
    }
    tc_measure_all(&tc);

    t0 = parsec_qpu_now_ns();
    rc = parsec_qpu_sampler_run(sampler, tc.circuit, params, (uint32_t)shots, 1, 1, tr.result, tr.max_entries, NULL);
    t1 = parsec_qpu_now_ns();
    rc |= parsec_qpu_sampler_run(sampler, tc.circuit, params, (uint32_t)shots, 1, 2, tr.result, tr.max_entries, NULL);
    t2 = parsec_qpu_now_ns();
    EXPECT(PARSEC_SUCCESS == rc, "timing run failed (%d)", rc);
    printf("timing: QAOA on %d qubits, %d layers, %d edges (%u operations), %d shots: %.3f ms the first time, "
           "%.3f ms with the memory in place; %u distinct outcomes\n",
           n, layers, m, tc.circuit->nb_ops, shots, (t1 - t0) * 1e-6, (t2 - t1) * 1e-6, tr.result->nb_entries);

    parsec_qpu_sampler_free(sampler);
    free(params);
    free(tr.result);
    free(tc.circuit);
}

int main(int argc, char *argv[])
{
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();
    int time_n = 0, layers = 2, shots = 1024;

    for( int i = 1; i + 1 < argc; i++ ) {
        if( 0 == strcmp(argv[i], "-t") ) time_n = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-p") ) layers = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-s") ) shots = atoi(argv[++i]);
    }

    test_circuit_check();
    test_each_gate(sampler);
    test_random_circuits(sampler);
    test_analytic(sampler);
    test_sampling(sampler);
    parsec_qpu_sampler_free(sampler);

    if( (time_n >= 2) && (time_n <= PARSEC_QPU_SAMPLER_MAX_QUBITS) && (layers >= 1) && (shots >= 1) ) {
        time_qaoa(time_n, layers, shots);
    }

    printf("qpu_sampler_test: %s\n", nb_errors ? "FAILED" : "passed");
    return nb_errors ? 1 : 0;
}
