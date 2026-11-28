/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * The adapter backend against an adapter that simulates by its own means
 * (tools/qpu/parsec_qpu_adapter_qiskit.py with Qiskit Aer), without the
 * runtime. The adapter runs already; this program is its client.
 *
 * Such an adapter does not give the counts of PaRSEC's sampler, so nothing
 * here is compared count for count. What is checked is that the circuit
 * PaRSEC wrote means, to the other side, what it means here:
 *
 *   - Bit order, the meaning of a parameter slot, qubits that are not
 *     measured, and a qubit measured before the others are done with, on
 *     circuits with one possible outcome.
 *   - Every operation of the circuit format, the definitions of rxx, ryy and
 *     rzz included: the frequencies of the outcomes against the exact
 *     probabilities PaRSEC computes, by their total variation distance.
 *   - What the adapter promises: the same counts for the same key, also in a
 *     job of several circuits, and with other jobs in flight at the adapter;
 *     other counts for another key.
 *   - A result cut to its buffer, and a circuit too wide for the adapter.
 *   - A parameter that makes an angle infinite: that job is refused or
 *     fails, and the adapter is there for the next one.
 *
 * With -b or -B the adapter is compared with another one, and nothing of the
 * above is done: an evaluation gives the same counts at an adapter that was
 * started with the same seed (-b), since they depend on the seed and the key
 * and on nothing else, and other counts at one started with another seed
 * (-B).
 *
 * The adapter draws from a seed, so the verdict is the same every time.
 *
 * Before any of that, and with no adapter, the test checks its own
 * circuits: a reader that takes one operation for another (25 mistakes: one
 * gate for another, a gate left out, a sign, the qubits of cx the other way,
 * the wrong parameter slot, a coefficient left out) has to move the outcomes
 * of at least one of them by 0.15 in total variation, well above what
 * sampling does. -S runs that check alone.
 *
 * Options:
 *   -S        check the circuits and stop: no adapter is needed
 *   -v        print how far each mistake moves the outcomes
 *   -a PATH   the socket of the adapter (required without -S)
 *   -t TV     largest total variation distance allowed (default 0.08)
 *   -n        the adapter simulates noise: outcomes that are certain without
 *             it only have to come out in three shots of four
 *   -w N      a circuit of N qubits has to be refused as too wide (default 0: not tried)
 *   -b PATH   the socket of an adapter started as the one of -a: the same counts
 *   -B PATH   the socket of an adapter started with another seed: other counts
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

#define MS      1000000LL
#define SHOTS   4000
#define PI      3.14159265358979323846
#define ROOM(NB_OPS)  (sizeof(parsec_qpu_circuit_t) + (size_t)(NB_OPS) * PARSEC_QPU_OP_SIZE)

static const parsec_qpu_backend_ops_t *ops = &parsec_qpu_backend_adapter_ops;
static parsec_qpu_backend_t *backend = NULL;
static double max_tv = 0.08;
static int    noisy = 0;
static uint64_t next_tag = 1;

static parsec_qpu_circuit_t *circuit_new(uint32_t n, uint32_t nb_params, uint32_t max_ops)
{
    parsec_qpu_circuit_t *c = (parsec_qpu_circuit_t*)calloc(1, ROOM(max_ops));
    parsec_qpu_circuit_init(c, n, nb_params);
    return c;
}

/* Every operation of the format on n qubits, with two parameters, placed so
 * that taking any of them for another changes what is measured: the Pauli
 * gates sit in the middle, and a layer of rotations about two axes comes
 * before the measurement, so that a phase becomes a probability.
 * check_circuits() below holds the circuits to that. */
static parsec_qpu_circuit_t *make_circuit(uint32_t n, int variant)
{
    const size_t room = ROOM(12 * n + 16);
    parsec_qpu_circuit_t *c = circuit_new(n, 2, 12 * n + 16);

    for( uint32_t q = 0; q < n; q++ ) parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_H, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.0);
    for( uint32_t q = 0; q + 1 < n; q++ ) {
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RXX + (int)((q + (uint32_t)variant) % 3), (int)q, (int)q + 1, 0, -0.5 - 0.25 * variant - 0.1 * q);
    }
    for( uint32_t q = 0; q < n; q++ ) {
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RX + (int)((q + (uint32_t)variant) % 3), (int)q, 0, 1, 2.0);
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RZ, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.3 * (q + 1) + variant);
    }
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_X, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_Y, 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_Z, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_CX, 0, (int)n - 1, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_CZ, (int)n - 1, 1, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RY, (int)n - 1, 0, 1, -0.7);
    for( uint32_t q = 0; q < n; q++ ) {
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RY, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.4 + 0.2 * q);
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RX, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.9 - 0.15 * q);
    }
    parsec_qpu_circuit_measure_all(c, room);
    return c;
}

static parsec_qpu_request_t request_of(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                                       uint32_t max_entries, uint64_t key)
{
    parsec_qpu_request_t r;
    memset(&r, 0, sizeof(r));
    r.circuit = circuit; r.params = params; r.shots = shots; r.max_entries = max_entries; r.key = key; r.attempt = 1;
    return r;
}

/* Run a job of nb circuits and take its results. Returns 0 when it did. */
static int run(const parsec_qpu_request_t *requests, int nb, parsec_qpu_result_t **results, const char *what)
{
    const int64_t give_up = parsec_qpu_now_ns() + 120000 * MS;
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;
    int rc, n = 0;

    for( int i = 0; i < nb; i++ ) results[i] = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(requests[i].max_entries));
    rc = ops->submit(backend, requests, nb, next_tag++, &id);
    EXPECT(PARSEC_SUCCESS == rc, "%s: submit returned %d", what, rc);
    if( PARSEC_SUCCESS != rc ) return -1;
    while( (0 == n) && (parsec_qpu_now_ns() < give_up) ) n = ops->wait(backend, 1000 * MS, &event, 1);
    EXPECT((1 == n) && (event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "%s: the job did not end well (%d event(s), state %d)",
           what, n, (1 == n) ? event.state : -1);
    if( (1 != n) || (PARSEC_QPU_JOB_DONE != event.state) ) return -1;
    rc = ops->results(backend, id, results, nb);
    EXPECT(PARSEC_SUCCESS == rc, "%s: results returned %d", what, rc);
    for( int i = 0; (PARSEC_SUCCESS == rc) && (i < nb); i++ ) {
        EXPECT(PARSEC_QPU_RESULT_MOCK & results[i]->flags, "%s: a simulated result is not marked as one", what);
        EXPECT(results[i]->nb_qubits == requests[i].circuit->nb_qubits, "%s: a result on %d qubits", what, (int)results[i]->nb_qubits);
    }
    return (PARSEC_SUCCESS == rc) ? 0 : -1;
}

static uint32_t count_of(const parsec_qpu_result_t *result, uint64_t outcome)
{
    for( uint32_t e = 0; e < result->nb_entries; e++ ) {
        uint64_t bits;
        uint32_t count;
        parsec_qpu_result_get_entry(result, e, &bits, &count);
        if( bits == outcome ) return count;
    }
    return 0;
}

/* A circuit that has one outcome when nothing disturbs it. */
static void expect_outcome(const parsec_qpu_circuit_t *circuit, const double *params, uint64_t outcome, const char *what)
{
    parsec_qpu_request_t r = request_of(circuit, params, 400, 400, 0x5000 + next_tag);
    parsec_qpu_result_t *result;
    uint32_t count;

    if( 0 != run(&r, 1, &result, what) ) { free(result); return; }
    count = count_of(result, outcome);
    EXPECT(400 == result->shots_returned, "%s: %u shots came back of 400", what, result->shots_returned);
    if( noisy ) {
        EXPECT(count >= 300, "%s: outcome 0x%llx came out %u times in 400, expected at least three times in four",
               what, (unsigned long long)outcome, count);
    } else {
        EXPECT((400 == count) && (1 == result->nb_entries), "%s: outcome 0x%llx came out %u times in 400, among %u outcomes; expected every time",
               what, (unsigned long long)outcome, count, result->nb_entries);
    }
    free(result);
}

/* The probability of every outcome of a circuit, exactly: an outcome is the
 * state with the qubits that are not measured at 0. NULL when it cannot be had. */
static double *distribution(const parsec_qpu_circuit_t *circuit, const double *params)
{
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();
    const uint64_t dim = ((uint64_t)1) << circuit->nb_qubits, measured = parsec_qpu_circuit_measured(circuit);
    const double *amp = NULL;
    double *p = (double*)calloc(dim, sizeof(double));

    if( PARSEC_SUCCESS != parsec_qpu_sampler_state(sampler, circuit, params, NULL, &amp) ) {
        free(p);
        p = NULL;
    } else {
        for( uint64_t x = 0; x < dim; x++ ) p[x & measured] += amp[2 * x] * amp[2 * x] + amp[2 * x + 1] * amp[2 * x + 1];
    }
    parsec_qpu_sampler_free(sampler);
    return p;
}

/* The frequencies of a result against the probabilities of the circuit. */
static double total_variation(const parsec_qpu_circuit_t *circuit, const double *params, const parsec_qpu_result_t *result)
{
    const uint64_t dim = ((uint64_t)1) << circuit->nb_qubits;
    double *p = distribution(circuit, params), tv = 0.0;

    if( NULL == p ) return 1e300;
    for( uint64_t x = 0; x < dim; x++ ) {
        tv += fabs((double)count_of(result, x) / (double)result->shots_returned - p[x]);
    }
    free(p);
    return 0.5 * tv;
}

/* ---- Do the circuits tell the operations apart? ----
 *
 * A reader on the other side that takes one operation for another, every
 * time it meets it, gives another distribution. How far that is from the
 * right one, exactly, says whether the comparison of frequencies with
 * probabilities can see the mistake: sampling alone puts 4,000 shots about
 * 0.04 from the probabilities, so a mistake has to move them by much more.
 * Each mistake below has to move at least one of the three circuits by
 * MIN_SEEN, which needs no adapter to check. */

#define MIN_SEEN 0.15

static int is_rotation(int opcode) { return ((opcode >= PARSEC_QPU_OP_RX) && (opcode <= PARSEC_QPU_OP_RZ)) || ((opcode >= PARSEC_QPU_OP_RXX) && (opcode <= PARSEC_QPU_OP_RZZ)); }

/* Each takes an operation and makes the mistake on it; 0 when it is not an operation the mistake is about. */
static int mix_one(parsec_qpu_op_t *op, int from, int to) { if( op->opcode != from ) return 0; op->opcode = (uint8_t)to; return 1; }
static int mix_drop(parsec_qpu_op_t *op, int from)
{
    if( op->opcode != from ) return 0;
    op->opcode = PARSEC_QPU_OP_RZ; op->param = PARSEC_QPU_NO_PARAM; op->angle = 0.0;     /* nothing */
    return 1;
}
static int mix_sign(parsec_qpu_op_t *op, int of) { if( op->opcode != of ) return 0; op->angle = -op->angle; return 1; }
static int mix_swap(parsec_qpu_op_t *op, int of)
{
    uint16_t q;
    if( op->opcode != of ) return 0;
    q = op->q0; op->q0 = op->q1; op->q1 = q;
    return 1;
}
static int mix_slot(parsec_qpu_op_t *op) { if( !is_rotation(op->opcode) || (PARSEC_QPU_NO_PARAM == op->param) ) return 0; op->param = (int16_t)(1 - op->param); return 1; }
static int mix_coefficient(parsec_qpu_op_t *op) { if( !is_rotation(op->opcode) || (PARSEC_QPU_NO_PARAM == op->param) ) return 0; op->angle = 1.0; return 1; }

typedef struct mixup_s {
    const char *name;
    int kind, a, b;         /* 0 one for another, 1 left out, 2 sign, 3 qubits swapped, 4 the other slot, 5 the coefficient left out */
} mixup_t;

static const mixup_t mixups[] = {
    { "x read as y", 0, PARSEC_QPU_OP_X, PARSEC_QPU_OP_Y },       { "y read as z", 0, PARSEC_QPU_OP_Y, PARSEC_QPU_OP_Z },
    { "z read as x", 0, PARSEC_QPU_OP_Z, PARSEC_QPU_OP_X },       { "y read as x", 0, PARSEC_QPU_OP_Y, PARSEC_QPU_OP_X },
    { "h left out", 1, PARSEC_QPU_OP_H, 0 },                      { "x left out", 1, PARSEC_QPU_OP_X, 0 },
    { "y left out", 1, PARSEC_QPU_OP_Y, 0 },                      { "z left out", 1, PARSEC_QPU_OP_Z, 0 },
    { "rx read as ry", 0, PARSEC_QPU_OP_RX, PARSEC_QPU_OP_RY },   { "ry read as rz", 0, PARSEC_QPU_OP_RY, PARSEC_QPU_OP_RZ },
    { "rz read as rx", 0, PARSEC_QPU_OP_RZ, PARSEC_QPU_OP_RX },
    { "rxx read as ryy", 0, PARSEC_QPU_OP_RXX, PARSEC_QPU_OP_RYY }, { "ryy read as rzz", 0, PARSEC_QPU_OP_RYY, PARSEC_QPU_OP_RZZ },
    { "rzz read as rxx", 0, PARSEC_QPU_OP_RZZ, PARSEC_QPU_OP_RXX },
    { "cx read as cz", 0, PARSEC_QPU_OP_CX, PARSEC_QPU_OP_CZ },   { "cz read as cx", 0, PARSEC_QPU_OP_CZ, PARSEC_QPU_OP_CX },
    { "cx with its qubits the other way", 3, PARSEC_QPU_OP_CX, 0 },
    { "the sign of rx", 2, PARSEC_QPU_OP_RX, 0 },   { "the sign of ry", 2, PARSEC_QPU_OP_RY, 0 },   { "the sign of rz", 2, PARSEC_QPU_OP_RZ, 0 },
    { "the sign of rxx", 2, PARSEC_QPU_OP_RXX, 0 }, { "the sign of ryy", 2, PARSEC_QPU_OP_RYY, 0 }, { "the sign of rzz", 2, PARSEC_QPU_OP_RZZ, 0 },
    { "the other parameter slot", 4, 0, 0 },        { "the coefficient of a slot left out", 5, 0, 0 }
};

static void check_circuits(int verbose)
{
    static const double params[2] = { 0.37, -1.2 }, other[2] = { 1.5, 0.25 };
    parsec_qpu_circuit_t *circuits[3] = { make_circuit(3, 0), make_circuit(4, 1), make_circuit(5, 2) };
    const double *values[3] = { params, other, params };
    double *right[3];

    for( int c = 0; c < 3; c++ ) {
        EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(circuits[c], 0, NULL), "the test's own circuit %d is not well-formed", c);
        right[c] = distribution(circuits[c], values[c]);
    }
    for( size_t m = 0; m < sizeof(mixups) / sizeof(mixups[0]); m++ ) {
        double best = 0.0;
        int met = 0;
        for( int c = 0; c < 3; c++ ) {
            const size_t size = parsec_qpu_circuit_size(circuits[c]);
            const uint64_t dim = ((uint64_t)1) << circuits[c]->nb_qubits;
            parsec_qpu_circuit_t *wrong = (parsec_qpu_circuit_t*)malloc(size);
            parsec_qpu_op_t *ops_ = parsec_qpu_circuit_ops(wrong);
            double *p, tv = 0.0;
            int changed = 0;

            memcpy(wrong, circuits[c], size);
            for( uint32_t i = 0; i < wrong->nb_ops; i++ ) {
                switch( mixups[m].kind ) {
                case 0: changed += mix_one(&ops_[i], mixups[m].a, mixups[m].b); break;
                case 1: changed += mix_drop(&ops_[i], mixups[m].a); break;
                case 2: changed += mix_sign(&ops_[i], mixups[m].a); break;
                case 3: changed += mix_swap(&ops_[i], mixups[m].a); break;
                case 4: changed += mix_slot(&ops_[i]); break;
                default: changed += mix_coefficient(&ops_[i]); break;
                }
            }
            met += changed;
            if( changed && (NULL != (p = distribution(wrong, values[c]))) && (NULL != right[c]) ) {
                for( uint64_t x = 0; x < dim; x++ ) tv += fabs(p[x] - right[c][x]);
                if( 0.5 * tv > best ) best = 0.5 * tv;
                free(p);
            }
            free(wrong);
        }
        if( verbose ) printf("qpu_adapter_stat: %-36s moves the outcomes by %.3f\n", mixups[m].name, best);
        EXPECT(met > 0, "no circuit of the test has the operation that '%s' is about", mixups[m].name);
        EXPECT(best >= MIN_SEEN, "'%s' moves the outcomes of the test's circuits by %.3f at most: the test could not see it (%.2f needed)",
               mixups[m].name, best, MIN_SEEN);
    }
    for( int c = 0; c < 3; c++ ) { free(right[c]); free(circuits[c]); }
}

static void expect_distribution(const parsec_qpu_circuit_t *circuit, const double *params, uint64_t key, const char *what)
{
    parsec_qpu_request_t r = request_of(circuit, params, SHOTS, SHOTS, key);
    parsec_qpu_result_t *result;
    double tv;

    if( 0 != run(&r, 1, &result, what) ) { free(result); return; }
    EXPECT(SHOTS == result->shots_returned, "%s: %u shots came back of %d", what, result->shots_returned, SHOTS);
    tv = total_variation(circuit, params, result);
    printf("qpu_adapter_stat: %-44s %u outcomes, total variation distance %.4f\n", what, result->nb_entries, tv);
    EXPECT(tv <= max_tv, "%s: the outcomes are %.4f from the probabilities of the circuit in total variation, more than %.4f",
           what, tv, max_tv);
    free(result);
}

static int same_counts(const parsec_qpu_result_t *a, const parsec_qpu_result_t *b)
{
    return (a->nb_entries == b->nb_entries) && (a->shots_returned == b->shots_returned) &&
           (0 == memcmp(a + 1, b + 1, (size_t)a->nb_entries * PARSEC_QPU_ENTRY_SIZE));
}

int main(int argc, char *argv[])
{
    static const double zero[2] = { 0.0, 0.0 }, first[2] = { PI, 0.0 }, second[2] = { 0.0, PI };
    static const double params[2] = { 0.37, -1.2 }, other[2] = { 1.5, 0.25 };
    parsec_qpu_circuit_t *c;
    parsec_qpu_caps_t caps;
    int opt, wide = 0, circuits_only = 0, verbose = 0;
    char *same_seed = NULL, *other_seed = NULL;

    while( -1 != (opt = getopt(argc, argv, "a:b:B:t:nw:Sv")) ) {
        if( 'a' == opt ) parsec_qpu_params.adapter_address = optarg;
        else if( 'b' == opt ) same_seed = optarg;
        else if( 'B' == opt ) other_seed = optarg;
        else if( 't' == opt ) max_tv = atof(optarg);
        else if( 'n' == opt ) noisy = 1;
        else if( 'w' == opt ) wide = atoi(optarg);
        else if( 'S' == opt ) circuits_only = 1;
        else if( 'v' == opt ) verbose = 1;
    }
    /* First, and with no adapter: are the circuits of this test worth running? */
    check_circuits(verbose || circuits_only);
    if( circuits_only ) {
        if( nb_errors ) {
            fprintf(stderr, "qpu_adapter_stat: %d error(s)\n", nb_errors);
            return 1;
        }
        printf("qpu_adapter_stat: the circuits tell the operations apart: passed\n");
        return 0;
    }
    if( NULL == parsec_qpu_params.adapter_address ) {
        fprintf(stderr, "usage: %s -a <socket of the adapter> [-t <distance>] [-n] [-w <qubits>] [-v], or -S, "
                        "or -a <socket> [-b <socket>] [-B <socket>]\n", argv[0]);
        return 2;
    }
    if( (NULL != same_seed) || (NULL != other_seed) ) {
        /* ---- two or three adapters: what the seed they were started with decides ---- */
        char *sockets[3] = { parsec_qpu_params.adapter_address, same_seed, other_seed };
        parsec_qpu_result_t *counts[3][2] = { { NULL, NULL }, { NULL, NULL }, { NULL, NULL } };
        parsec_qpu_circuit_t *c4 = make_circuit(4, 1);

        for( int a = 0; a < 3; a++ ) {
            if( NULL == sockets[a] ) continue;
            parsec_qpu_params.adapter_address = sockets[a];
            backend = NULL;
            if( (PARSEC_SUCCESS != ops->open(&backend, 0)) || (NULL == backend) ) {
                fprintf(stderr, "qpu_adapter_stat: no adapter at %s\n", sockets[a]);
                return 1;
            }
            for( int k = 0; k < 2; k++ ) {
                parsec_qpu_request_t r = request_of(c4, params, 1000, 1000, 0x400 + (uint64_t)k);
                if( 0 != run(&r, 1, &counts[a][k], "an evaluation at one of several adapters") ) { free(counts[a][k]); counts[a][k] = NULL; }
            }
            EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
        }
        for( int k = 0; k < 2; k++ ) {
            EXPECT(NULL != counts[0][k], "no counts from the first adapter");
            if( NULL == counts[0][k] ) continue;
            if( NULL != same_seed ) {
                EXPECT((NULL != counts[1][k]) && same_counts(counts[0][k], counts[1][k]),
                       "evaluation %d gave other counts at another adapter that was started with the same seed", k);
            }
            if( NULL != other_seed ) {
                EXPECT((NULL != counts[2][k]) && !same_counts(counts[0][k], counts[2][k]),
                       "evaluation %d gave the same counts at an adapter that was started with another seed", k);
            }
        }
        for( int a = 0; a < 3; a++ ) { free(counts[a][0]); free(counts[a][1]); }
        free(c4);
        if( nb_errors ) {
            fprintf(stderr, "qpu_adapter_stat: %d error(s)\n", nb_errors);
            return 1;
        }
        printf("qpu_adapter_stat: the counts follow the seed an adapter was started with: passed\n");
        return 0;
    }
    if( (PARSEC_SUCCESS != ops->open(&backend, 0)) || (NULL == backend) ) {
        fprintf(stderr, "qpu_adapter_stat: no adapter at %s\n", parsec_qpu_params.adapter_address);
        return 1;
    }
    EXPECT(PARSEC_SUCCESS == ops->capabilities(backend, &caps), "capabilities");
    EXPECT(0 == caps.is_physical, "the adapter says it drives a quantum device");
    EXPECT(caps.nb_qubits >= 5, "the adapter takes %u qubits, and this test needs 5", caps.nb_qubits);
    printf("qpu_adapter_stat: the adapter takes %u qubits and %u circuits per job\n", caps.nb_qubits, caps.max_circuits_per_job);

    /* ---- one outcome: what a bit, a slot and a measurement mean on the other side ---- */
    c = circuit_new(3, 0, 8);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_X, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_measure_all(c, ROOM(8));
    expect_outcome(c, NULL, 0x1, "x on qubit 0 of 3 is bit 0");
    free(c);
    c = circuit_new(3, 0, 8);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_X, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_X, 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_MEASURE, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    expect_outcome(c, NULL, 0x4, "a qubit that is not measured reads 0");
    free(c);
    /* Three of four qubits measured, not in the order of their numbers, the
     * first of them before the operations on the others; the fourth is
     * flipped and not measured. */
    c = circuit_new(4, 0, 8);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_X, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_MEASURE, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_X, 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_X, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_MEASURE, 3, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_MEASURE, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    expect_outcome(c, NULL, 0x5, "qubits 0, 3 and 2 measured in that order, qubit 1 flipped and not measured");
    free(c);
    c = circuit_new(3, 0, 8);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_X, 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_CX, 1, 2, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_measure_all(c, ROOM(8));
    expect_outcome(c, NULL, 0x6, "cx has its control first");
    free(c);
    c = circuit_new(2, 2, 8);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_RX, 0, 0, 0, 1.0);
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_RX, 1, 0, 1, 1.0);
    parsec_qpu_circuit_measure_all(c, ROOM(8));
    expect_outcome(c, zero,   0x0, "rx by slots at 0");
    expect_outcome(c, first,  0x1, "rx(pi) by slot 0 flips qubit 0");
    expect_outcome(c, second, 0x2, "rx(pi) by slot 1 flips qubit 1");
    free(c);
    c = circuit_new(2, 3, 8);            /* a coefficient, and a slot the circuit does not use */
    parsec_qpu_circuit_add(c, ROOM(8), PARSEC_QPU_OP_RY, 1, 0, 2, -0.5);
    parsec_qpu_circuit_measure_all(c, ROOM(8));
    {
        const double three[3] = { 7.0, 7.0, -2.0 * PI };
        expect_outcome(c, three, 0x2, "ry(-0.5 times slot 2 at -2 pi) flips qubit 1");
    }
    free(c);

    /* ---- every operation: frequencies against probabilities ---- */
    c = make_circuit(3, 0); expect_distribution(c, params, 0x100, "3 qubits, every operation"); free(c);
    c = make_circuit(4, 1); expect_distribution(c, other,  0x101, "4 qubits, every operation"); free(c);
    c = make_circuit(5, 2); expect_distribution(c, params, 0x102, "5 qubits, every operation"); free(c);
    for( int g = 0; g < 3; g++ ) {
        /* Each of rxx, ryy and rzz between two states that tell it from the others. */
        static const char *names[3] = { "rxx alone", "ryy alone", "rzz alone" };
        c = circuit_new(2, 1, 12);
        parsec_qpu_circuit_add(c, ROOM(12), PARSEC_QPU_OP_RY, 0, 0, PARSEC_QPU_NO_PARAM, 0.7);
        parsec_qpu_circuit_add(c, ROOM(12), PARSEC_QPU_OP_RX, 1, 0, PARSEC_QPU_NO_PARAM, 1.1);
        parsec_qpu_circuit_add(c, ROOM(12), PARSEC_QPU_OP_RXX + g, 0, 1, 0, 1.0);
        parsec_qpu_circuit_add(c, ROOM(12), PARSEC_QPU_OP_RX, 0, 0, PARSEC_QPU_NO_PARAM, 0.9);
        parsec_qpu_circuit_add(c, ROOM(12), PARSEC_QPU_OP_RY, 1, 0, PARSEC_QPU_NO_PARAM, -0.4);
        parsec_qpu_circuit_measure_all(c, ROOM(12));
        expect_distribution(c, other, 0x110 + (uint64_t)g, names[g]);
        free(c);
    }

    /* ---- the same key gives the same counts, alone or in company ---- */
    {
        parsec_qpu_circuit_t *c3 = make_circuit(3, 0), *c4 = make_circuit(4, 1);
        parsec_qpu_request_t alone = request_of(c4, params, 1000, 1000, 0x200), batch[3];
        parsec_qpu_result_t *a = NULL, *b = NULL, *d = NULL, *three[3] = { NULL, NULL, NULL };

        if( (0 == run(&alone, 1, &a, "a key")) && (0 == run(&alone, 1, &b, "the same key again")) ) {
            EXPECT(same_counts(a, b), "the same evaluation gave other counts the second time");
        }
        alone.key = 0x201;
        if( 0 == run(&alone, 1, &d, "another key") ) {
            EXPECT((NULL != a) && !same_counts(a, d), "another key gave the same counts");
        }
        batch[0] = request_of(c3, other,  1000, 1000, 0x202);
        batch[1] = request_of(c4, params, 1000, 1000, 0x200);
        batch[2] = request_of(c3, params, 1000, 1000, 0x203);
        if( caps.max_circuits_per_job >= 3 ) {
            if( 0 == run(batch, 3, three, "three circuits in a job") ) {
                EXPECT((NULL != a) && same_counts(a, three[1]), "an evaluation gave other counts in a job of three than alone");
            }
        }
        /* Cut to a buffer of 3 outcomes: the three that came out most. */
        alone = request_of(c4, params, 1000, 3, 0x200);
        free(b); b = NULL;
        if( (0 == run(&alone, 1, &b, "a result that does not fit")) && (NULL != a) ) {
            uint32_t kept = 0;
            EXPECT((3 == b->nb_entries) && (PARSEC_QPU_RESULT_TRUNCATED & b->flags), "%u entries in a buffer of 3, cut or not", b->nb_entries);
            for( uint32_t e = 0; e < b->nb_entries; e++ ) {
                uint64_t bits; uint32_t count;
                parsec_qpu_result_get_entry(b, e, &bits, &count);
                EXPECT(count == count_of(a, bits), "an outcome that was kept has another count than in the whole result");
                kept += count;
            }
            EXPECT(kept == b->shots_returned, "%u shots said to be returned, the entries hold %u", b->shots_returned, kept);
        }
        free(a); free(b); free(d);
        for( int i = 0; i < 3; i++ ) free(three[i]);
        free(c3); free(c4);
    }

    /* ---- a parameter that makes an angle infinite: no job of it ends well, and the adapter goes on ---- */
    {
        static const double huge[1] = { 1e308 };
        const int64_t give_up = parsec_qpu_now_ns() + 60000 * MS;
        parsec_qpu_request_t r;
        parsec_qpu_job_id_t id = 0;
        parsec_qpu_event_t event;
        int rc, n = 0;

        c = circuit_new(1, 1, 4);
        parsec_qpu_circuit_add(c, ROOM(4), PARSEC_QPU_OP_RX, 0, 0, 0, 2.0);
        parsec_qpu_circuit_measure_all(c, ROOM(4));
        r = request_of(c, huge, 100, 100, 0x280);
        rc = ops->submit(backend, &r, 1, next_tag++, &id);
        if( PARSEC_SUCCESS == rc ) {
            while( (0 == n) && (parsec_qpu_now_ns() < give_up) ) n = ops->wait(backend, 1000 * MS, &event, 1);
            EXPECT((1 == n) && (event.id == id) && (PARSEC_QPU_JOB_FAILED == event.state),
                   "a job whose angle is 2 times 1e308: %d event(s), state %d; expected one job, failed", n, (1 == n) ? event.state : -1);
        } else {
            EXPECT(PARSEC_ERR_OUT_OF_RESOURCE != rc, "a job whose angle is 2 times 1e308 was neither taken nor refused for good (%d)", rc);
        }
        free(c);
        /* What follows needs the adapter, and the count of connections at the end says whether it was the same one. */
    }

    /* ---- ... and with other jobs in flight: eight jobs at once, then each of them alone ---- */
    {
        enum { NB = 8 };
        parsec_qpu_circuit_t *c3 = make_circuit(3, 0), *c4 = make_circuit(4, 1);
        const int64_t give_up = parsec_qpu_now_ns() + 120000 * MS;
        parsec_qpu_request_t r[NB];
        parsec_qpu_result_t *together[NB], *alone[NB];
        parsec_qpu_job_id_t ids[NB];
        int submitted = 0, ended = 0, well = 1;

        for( int i = 0; i < NB; i++ ) {
            r[i] = request_of((i & 1) ? c3 : c4, (i & 2) ? other : params, 1000, 1000, 0x210 + (uint64_t)i);
            together[i] = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(1000));
            alone[i] = NULL;
            if( PARSEC_SUCCESS == ops->submit(backend, &r[i], 1, next_tag++, &ids[i]) ) submitted++;
        }
        EXPECT(NB == submitted, "%d of %d jobs submitted at once were taken", submitted, NB);
        while( (NB == submitted) && (ended < NB) && (parsec_qpu_now_ns() < give_up) ) {
            parsec_qpu_event_t events[NB];
            const int n = ops->wait(backend, 1000 * MS, events, NB);
            if( n < 0 ) break;
            for( int e = 0; e < n; e++ ) well = well && (PARSEC_QPU_JOB_DONE == events[e].state);
            ended += n;
        }
        EXPECT((NB == ended) && well, "of %d jobs in flight at once, %d ended, %s", NB, ended, well ? "all well" : "not all well");
        for( int i = 0; (NB == ended) && well && (i < NB); i++ ) {
            int rc = ops->results(backend, ids[i], &together[i], 1);
            EXPECT(PARSEC_SUCCESS == rc, "the results of job %d of %d that were in flight at once: %d", i, NB, rc);
            if( (PARSEC_SUCCESS == rc) && (0 == run(&r[i], 1, &alone[i], "an evaluation alone, after it ran among others")) ) {
                EXPECT(same_counts(together[i], alone[i]), "evaluation %d gave other counts with other jobs in flight than alone", i);
            }
        }
        for( int i = 0; i < NB; i++ ) { free(together[i]); free(alone[i]); }
        free(c3); free(c4);
    }

    /* ---- a circuit the adapter cannot take ---- */
    if( wide > 0 ) {
        parsec_qpu_request_t r;
        parsec_qpu_job_id_t id = 0;
        int rc;
        c = make_circuit((uint32_t)wide, 0);
        r = request_of(c, params, 10, 10, 0x300);
        rc = ops->submit(backend, &r, 1, next_tag++, &id);
        EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_OUT_OF_RESOURCE != rc), "a circuit of %d qubits was not refused for good (%d)", wide, rc);
        free(c);
    }

    {
        parsec_qpu_adapter_stats_t stats;
        parsec_qpu_adapter_stats(&stats);
        EXPECT((1 == stats.connects) && (0 == stats.unanswered) && (0 == stats.bad_results) && (0 == stats.lost_jobs),
               "%llu connections, %llu requests without an answer, %llu answers that were no result, %llu jobs lost",
               (unsigned long long)stats.connects, (unsigned long long)stats.unanswered, (unsigned long long)stats.bad_results,
               (unsigned long long)stats.lost_jobs);
    }
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    if( nb_errors ) {
        fprintf(stderr, "qpu_adapter_stat: %d error(s)\n", nb_errors);
        return 1;
    }
    printf("qpu_adapter_stat: passed\n");
    return 0;
}
