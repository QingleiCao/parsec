/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * Circuits as OpenQASM 3 text, without the runtime.
 *
 *   - What the writer writes for a small circuit, line by line.
 *   - Writing and reading back gives the same bytes: every operation, angles
 *     of every kind (zeros of both signs, the smallest and the largest
 *     doubles), and circuits drawn at random.
 *   - The definitions the writer gives for rxx, ryy and rzz prepare the
 *     states of the native operations: the definitions are read back as
 *     circuits and compared through the exact sampler.
 *   - A text as Qiskit writes it (qiskit.qasm3.dumps of Qiskit 2.6.2, kept
 *     here as a string) gives the circuit it stands for.
 *   - Angles written as expressions.
 *   - What the reader refuses, with the line and the reason.
 *   - Sizes: a text read for its size only, a buffer that is too small.
 *
 * Every random number comes from a fixed seed.
 *
 * With -o DIR the texts are kept as well, for a reader that is not this one
 * (the adapter of tools/qpu for CUDA-Q has one, in Python): cNNN.qasm is a
 * text the writer wrote and cNNN.ops the circuit it stands for; rNNN.qasm is
 * a text the reader here refuses; and oNNN.qasm is a text the reader here
 * takes that the writer does not write for its circuit, which a reader of
 * the writer's texts alone has to refuse. The directory is made if it is not
 * there, and what an earlier run kept in it is removed.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"
#include "parsec/mca/device/qpu/device_qpu_qasm.h"

#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int nb_errors = 0;
static const char *keep_dir = NULL;     /* -o: where the texts are kept */
static int nb_kept = 0, nb_kept_refused = 0, nb_kept_other = 0;

#define EXPECT(COND, ...)                                                   \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            fprintf(stderr, "%s:%d: ", __func__, __LINE__);                 \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            nb_errors++;                                                    \
        }                                                                   \
    } while(0)

#define ROOM(NB_OPS)  (sizeof(parsec_qpu_circuit_t) + (size_t)(NB_OPS) * PARSEC_QPU_OP_SIZE)

static parsec_qpu_circuit_t *circuit_new(uint32_t nb_qubits, uint32_t nb_params, uint32_t max_ops)
{
    parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)calloc(1, ROOM(max_ops));
    parsec_qpu_circuit_init(circuit, nb_qubits, nb_params);
    return circuit;
}

/* Write a circuit into memory of the right size. */
static char *write_text(const parsec_qpu_circuit_t *circuit, size_t *length)
{
    const int64_t n = parsec_qpu_qasm_write(circuit, NULL, 0);
    char *text;

    if( n < 0 ) return NULL;
    text = (char*)malloc((size_t)n + 1);
    if( n != parsec_qpu_qasm_write(circuit, text, (size_t)n + 1) ) { free(text); return NULL; }
    if( NULL != length ) *length = (size_t)n;
    return text;
}

static void keep_text_as(const char *text, const parsec_qpu_circuit_t *circuit, char kind);

/* Read a text into memory of the right size; NULL when it is refused. */
static parsec_qpu_circuit_t *read_text(const char *text, parsec_qpu_qasm_info_t *info)
{
    parsec_qpu_qasm_info_t local;
    parsec_qpu_circuit_t *circuit;

    if( NULL == info ) info = &local;
    if( PARSEC_SUCCESS != parsec_qpu_qasm_read(text, strlen(text), NULL, 0, info) ) return NULL;
    circuit = (parsec_qpu_circuit_t*)calloc(1, info->size);
    if( PARSEC_SUCCESS != parsec_qpu_qasm_read(text, strlen(text), circuit, info->size, info) ) { free(circuit); return NULL; }
    if( NULL != keep_dir ) {
        /* -o: a text that was read, and is not what the writer writes for the circuit it gave. */
        char *written = write_text(circuit, NULL);
        if( (NULL != written) && (0 != strcmp(written, text)) ) keep_text_as(text, NULL, 'o');
        free(written);
    }
    return circuit;
}

static int same_circuit(const parsec_qpu_circuit_t *a, const parsec_qpu_circuit_t *b)
{
    return (parsec_qpu_circuit_size(a) == parsec_qpu_circuit_size(b)) &&
           (0 == memcmp(a, b, parsec_qpu_circuit_size(a)));
}

/* -o: a text, and for one the writer wrote, the circuit it stands for: a line
 * with the qubits, the inputs and the operations, then a line per operation
 * with the opcode, the two qubits, the parameter slot and the angle by its bits. */
static void keep_text_as(const char *text, const parsec_qpu_circuit_t *circuit, char kind)
{
    char path[1024];
    FILE *f;

    if( NULL == keep_dir ) return;
    snprintf(path, sizeof(path), "%s/%c%03d.qasm", keep_dir, kind,
             ('c' == kind) ? nb_kept : ('r' == kind) ? nb_kept_refused : nb_kept_other);
    f = fopen(path, "wb");
    EXPECT(NULL != f, "cannot write %s", path);
    if( NULL == f ) return;
    fwrite(text, 1, strlen(text), f);
    fclose(f);
    if( 'r' == kind ) { nb_kept_refused++; return; }
    if( 'o' == kind ) { nb_kept_other++; return; }
    snprintf(path, sizeof(path), "%s/c%03d.ops", keep_dir, nb_kept++);
    f = fopen(path, "w");
    EXPECT(NULL != f, "cannot write %s", path);
    if( NULL == f ) return;
    fprintf(f, "%u %u %u\n", circuit->nb_qubits, circuit->nb_params, circuit->nb_ops);
    for( uint32_t i = 0; i < circuit->nb_ops; i++ ) {
        const parsec_qpu_op_t *op = &parsec_qpu_circuit_ops(circuit)[i];
        fprintf(f, "%d %d %d %d %a\n", (int)op->opcode, (int)op->q0, (int)op->q1,
                (PARSEC_QPU_NO_PARAM == op->param) ? -1 : (int)op->param, op->angle);
    }
    fclose(f);
}

static void keep_text(const char *text, const parsec_qpu_circuit_t *circuit)
{
    keep_text_as(text, circuit, (NULL != circuit) ? 'c' : 'r');
}

/* -o: the texts of an earlier run, which may be of an earlier version of this test, are not kept. */
static void forget_kept_texts(void)
{
    DIR *dir = opendir(keep_dir);
    struct dirent *entry;
    char path[1024];

    if( NULL == dir ) return;
    while( NULL != (entry = readdir(dir)) ) {
        const size_t n = strlen(entry->d_name);
        const int ours = (NULL != strchr("cro", entry->d_name[0])) &&
                         (((n > 5) && (0 == strcmp(entry->d_name + n - 5, ".qasm"))) || ((n > 4) && (0 == strcmp(entry->d_name + n - 4, ".ops"))));
        if( !ours ) continue;
        snprintf(path, sizeof(path), "%s/%s", keep_dir, entry->d_name);
        (void)unlink(path);
    }
    closedir(dir);
}

/* Written, read back: the same bytes. */
static void round_trip(const parsec_qpu_circuit_t *circuit, const char *what)
{
    parsec_qpu_qasm_info_t info;
    char *text = write_text(circuit, NULL);
    parsec_qpu_circuit_t *back;

    EXPECT(NULL != text, "%s: the circuit could not be written", what);
    if( NULL == text ) return;
    back = read_text(text, &info);
    EXPECT(NULL != back, "%s: what was written is not read back: line %d, %s\n%s", what, info.line, info.why, text);
    if( NULL != back ) {
        keep_text(text, back);
        EXPECT(same_circuit(circuit, back), "%s: what is read back is not what was written\n%s", what, text);
        EXPECT((info.nb_ops == circuit->nb_ops) && (info.nb_qubits == circuit->nb_qubits) &&
               (info.nb_params == circuit->nb_params) && (info.size == parsec_qpu_circuit_size(circuit)),
               "%s: the reader counted %u operations on %u qubits with %u inputs in %zu bytes", what,
               info.nb_ops, info.nb_qubits, info.nb_params, info.size);
    }
    free(back);
    free(text);
}

/* Add an operation with nothing in the fields it does not use: this is what
 * the reader gives back, since the text does not hold those fields. */
static void add_op(parsec_qpu_circuit_t *circuit, size_t room, int opcode, int q0, int q1, int param, double angle)
{
    const int two = (opcode >= PARSEC_QPU_OP_CX) && (opcode <= PARSEC_QPU_OP_RZZ);
    const int rotation = ((opcode >= PARSEC_QPU_OP_RX) && (opcode <= PARSEC_QPU_OP_RZ)) ||
                         ((opcode >= PARSEC_QPU_OP_RXX) && (opcode <= PARSEC_QPU_OP_RZZ));
    if( two && (q1 == q0) ) q1 = (q0 + 1) % (int)circuit->nb_qubits;
    (void)parsec_qpu_circuit_add(circuit, room, opcode, q0, two ? q1 : 0,
                                 rotation ? param : PARSEC_QPU_NO_PARAM, rotation ? angle : 0.0);
}

static void test_written_text(void)
{
    static const char expected[] =
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n"
        "input float[64] p0;\n"
        "input float[64] p1;\n"
        "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\n"
        "qubit[3] q;\n"
        "bit[3] c;\n"
        "h q[0];\n"
        "h q[1];\n"
        "x q[2];\n"
        "rzz(-1.5 * p0) q[0], q[1];\n"
        "rz(0.25) q[0];\n"
        "rx(2.0 * p1) q[1];\n"
        "cx q[0], q[2];\n"
        "c[0] = measure q[0];\n"
        "c[2] = measure q[2];\n";
    parsec_qpu_circuit_t *circuit = circuit_new(3, 2, 16);
    const size_t room = ROOM(16);
    char *text, small[40];
    size_t length = 0;
    int64_t n;

    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_H, 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_X, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_RZZ, 0, 1, 0, -1.5);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_RZ, 0, 0, PARSEC_QPU_NO_PARAM, 0.25);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_RX, 1, 0, 1, 2.0);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_CX, 0, 2, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_MEASURE, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_MEASURE, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);

    text = write_text(circuit, &length);
    EXPECT((NULL != text) && (0 == strcmp(text, expected)), "the writer wrote\n%s\nand not\n%s", text ? text : "(nothing)", expected);
    EXPECT(length == strlen(expected), "the writer says its text has %zu bytes, and it has %zu", length, strlen(expected));

    /* A buffer that is too small: the length of the whole text all the same, and a text that ends. */
    memset(small, 'x', sizeof(small));
    n = parsec_qpu_qasm_write(circuit, small, 20);
    EXPECT(n == (int64_t)strlen(expected), "in a small buffer the writer returns %lld, the text has %zu bytes", (long long)n, strlen(expected));
    EXPECT((19 == strlen(small)) && (0 == strncmp(small, expected, 19)) && ('x' == small[20]),
           "in a buffer of 20 bytes the writer did not write the first 19 bytes of the text and a 0, and nothing more");

    /* What cannot be written. */
    circuit->flags |= PARSEC_QPU_CIRCUIT_EMPTY;
    EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_qasm_write(circuit, NULL, 0), "an empty job was written");
    circuit->flags = 0;
    parsec_qpu_circuit_ops(circuit)[2].q0 = 7;
    EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_qasm_write(circuit, NULL, 0), "a circuit that is not well-formed was written");
    EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_qasm_write(NULL, NULL, 0), "no circuit was written");
    free(text);
    free(circuit);
}

static uint64_t rng_state = 0x243f6a8885a308d3ULL;
static uint64_t rng(void)
{
    rng_state = parsec_qpu_mix64(rng_state);
    return rng_state;
}

static void test_round_trips(void)
{
    static const double angles[] = {
        0.0, -0.0, 1.0, -1.0, 2.0, 0.1, -0.1, 1.0 / 3.0, 3.14159265358979323846, -2.5e-7, 1e22, -1e22,
        DBL_MIN, -DBL_MIN, DBL_MAX, -DBL_MAX, DBL_EPSILON, 4.9406564584124654e-324, 123456789.123456789,
        0.30000000000000004, 5e-324, 1e-310, 6.02214076e23, -7.0
    };
    const int nb_angles = (int)(sizeof(angles) / sizeof(angles[0]));
    parsec_qpu_circuit_t *circuit;
    size_t room;

    /* Every operation, with a number and with an input, and every angle. */
    room = ROOM(8 * nb_angles + 32);
    circuit = circuit_new(5, 3, 8 * nb_angles + 32);
    for( int opcode = PARSEC_QPU_OP_H; opcode <= PARSEC_QPU_OP_RZZ; opcode++ ) {
        add_op(circuit, room, opcode, opcode % 5, (opcode + 2) % 5, PARSEC_QPU_NO_PARAM, 0.5 * opcode);
        add_op(circuit, room, opcode, (opcode + 1) % 5, (opcode + 4) % 5, opcode % 3, -0.25 * opcode);
    }
    for( int i = 0; i < nb_angles; i++ ) {
        add_op(circuit, room, PARSEC_QPU_OP_RX + (i % 3), i % 5, 0, PARSEC_QPU_NO_PARAM, angles[i]);
        add_op(circuit, room, PARSEC_QPU_OP_RXX + (i % 3), i % 5, (i + 1) % 5, i % 3, angles[i]);
    }
    add_op(circuit, room, PARSEC_QPU_OP_MEASURE, 3, 0, PARSEC_QPU_NO_PARAM, 0.0);
    add_op(circuit, room, PARSEC_QPU_OP_MEASURE, 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(circuit, room, NULL), "the test's own circuit is not well-formed");
    round_trip(circuit, "every operation and every angle");
    free(circuit);

    /* What an operation holds in the fields it does not use is not written:
     * it comes back with those fields at rest. */
    {
        parsec_qpu_circuit_t *junk = circuit_new(4, 2, 4), *rest = circuit_new(4, 2, 4), *back;
        char *text;
        parsec_qpu_circuit_add(junk, ROOM(4), PARSEC_QPU_OP_H, 0, 3, 1, 0.7);
        parsec_qpu_circuit_add(junk, ROOM(4), PARSEC_QPU_OP_CX, 0, 1, 1, -2.0);
        parsec_qpu_circuit_add(junk, ROOM(4), PARSEC_QPU_OP_RX, 2, 3, PARSEC_QPU_NO_PARAM, 0.5);
        parsec_qpu_circuit_add(junk, ROOM(4), PARSEC_QPU_OP_MEASURE, 1, 2, 0, 9.0);
        add_op(rest, ROOM(4), PARSEC_QPU_OP_H, 0, 3, 1, 0.7);
        add_op(rest, ROOM(4), PARSEC_QPU_OP_CX, 0, 1, 1, -2.0);
        add_op(rest, ROOM(4), PARSEC_QPU_OP_RX, 2, 3, PARSEC_QPU_NO_PARAM, 0.5);
        add_op(rest, ROOM(4), PARSEC_QPU_OP_MEASURE, 1, 2, 0, 9.0);
        text = write_text(junk, NULL);
        back = (NULL != text) ? read_text(text, NULL) : NULL;
        EXPECT((NULL != back) && same_circuit(rest, back) && !same_circuit(junk, back),
               "fields that an operation does not use did not come back at rest");
        free(back); free(text); free(junk); free(rest);
    }

    /* What a non-rotation holds in the fields it does not use is not
     * written: such an operation comes back with those fields at rest. */
    circuit = circuit_new(2, 0, 4);
    parsec_qpu_circuit_add(circuit, ROOM(4), PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(circuit, ROOM(4), PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_measure_all(circuit, ROOM(4));
    round_trip(circuit, "a Bell pair");
    free(circuit);

    /* One qubit, the widest register, and one operation on the last qubit. */
    circuit = circuit_new(1, 0, 2);
    parsec_qpu_circuit_add(circuit, ROOM(2), PARSEC_QPU_OP_MEASURE, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    round_trip(circuit, "one qubit");
    free(circuit);
    circuit = circuit_new(PARSEC_QPU_MAX_QUBITS, 1, 70);
    parsec_qpu_circuit_add(circuit, ROOM(70), PARSEC_QPU_OP_RYY, 63, 0, 0, 0.5);
    parsec_qpu_circuit_measure_all(circuit, ROOM(70));
    round_trip(circuit, "64 qubits");
    free(circuit);

    /* Circuits drawn at random. */
    for( int run = 0; run < 200; run++ ) {
        const uint32_t nb_qubits = 2 + (uint32_t)(rng() % 20);
        const uint32_t nb_params = (uint32_t)(rng() % 6);
        const uint32_t nb_ops    = (uint32_t)(rng() % 120);
        char what[64];

        room = ROOM(nb_ops + nb_qubits);
        circuit = circuit_new(nb_qubits, nb_params, nb_ops + nb_qubits);
        for( uint32_t i = 0; i < nb_ops; i++ ) {
            const int opcode = PARSEC_QPU_OP_H + (int)(rng() % 12);
            const int q0 = (int)(rng() % nb_qubits);
            const int q1 = (q0 + 1 + (int)(rng() % (nb_qubits - 1))) % (int)nb_qubits;
            const int rotation = ((opcode >= PARSEC_QPU_OP_RX) && (opcode <= PARSEC_QPU_OP_RZ)) || (opcode >= PARSEC_QPU_OP_RXX);
            const int param = (rotation && (nb_params > 0) && (rng() & 1)) ? (int)(rng() % nb_params) : PARSEC_QPU_NO_PARAM;
            union { uint64_t u; double d; } bits;
            double angle = 0.0;
            if( rotation ) {
                /* Any finite double, by its bits, half of the time. */
                bits.u = rng();
                angle = (rng() & 1) ? bits.d : (parsec_qpu_unit(rng()) - 0.5) * 20.0;
                if( !isfinite(angle) ) angle = 1.25;
            }
            parsec_qpu_circuit_add(circuit, room, opcode, q0, (opcode >= PARSEC_QPU_OP_CX) ? q1 : 0,
                                   rotation ? param : PARSEC_QPU_NO_PARAM, angle);
        }
        for( uint32_t q = 0; q < nb_qubits; q++ ) {
            if( (0 == q) || (rng() & 1) ) parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_MEASURE, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.0);
        }
        snprintf(what, sizeof(what), "random circuit %d", run);
        EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(circuit, room, NULL), "%s is not well-formed", what);
        round_trip(circuit, what);
        free(circuit);
    }
}

/* The largest difference between two states of n qubits, a global phase apart. */
static double state_distance(const double *a, const double *b, uint32_t nb_qubits)
{
    const uint64_t dim = ((uint64_t)1) << nb_qubits;
    double re = 0.0, im = 0.0, norm, worst = 0.0;

    /* The phase that takes b to a: that of <b|a>. */
    for( uint64_t x = 0; x < dim; x++ ) {
        re += b[2*x] * a[2*x] + b[2*x+1] * a[2*x+1];
        im += b[2*x] * a[2*x+1] - b[2*x+1] * a[2*x];
    }
    norm = sqrt(re * re + im * im);
    if( norm < 1e-9 ) return 1e300;
    re /= norm; im /= norm;
    for( uint64_t x = 0; x < dim; x++ ) {
        const double dr = a[2*x]   - (re * b[2*x] - im * b[2*x+1]);
        const double di = a[2*x+1] - (re * b[2*x+1] + im * b[2*x]);
        const double d = sqrt(dr * dr + di * di);
        if( d > worst ) worst = d;
    }
    return worst;
}

/* Replace every `from` that stands as a word of its own. */
static void replace_word(char *text, size_t room, const char *from, const char *to)
{
    char *copy = (char*)malloc(room), *out = copy;
    const size_t n = strlen(from);

    for( const char *p = text; '\0' != *p; ) {
        const int before = (p == text) || !(('a' <= p[-1] && p[-1] <= 'z') || ('0' <= p[-1] && p[-1] <= '9') || ('_' == p[-1]));
        const int after  = !(('a' <= p[n] && p[n] <= 'z') || ('0' <= p[n] && p[n] <= '9') || ('_' == p[n]));
        if( before && (0 == strncmp(p, from, n)) && after ) {
            out += snprintf(out, room - (size_t)(out - copy), "%s", to);
            p += n;
        } else {
            *out++ = *p++;
        }
    }
    *out = '\0';
    snprintf(text, room, "%s", copy);
    free(copy);
}

/* The definitions of rxx, ryy and rzz that the writer puts in its text
 * prepare, from a state that tells every two-qubit state apart, the state
 * of the native operation. */
static void test_definitions(void)
{
    static const int opcodes[3] = { PARSEC_QPU_OP_RXX, PARSEC_QPU_OP_RYY, PARSEC_QPU_OP_RZZ };
    static const char *names[3] = { "rxx", "ryy", "rzz" };
    static const double thetas[4] = { 0.37, -1.9, 3.0, 0.0 };
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();
    const size_t room = ROOM(16);

    for( int g = 0; g < 3; g++ ) {
        for( int t = 0; t < 4; t++ ) {
            parsec_qpu_circuit_t *native = circuit_new(2, 0, 16), *defined;
            parsec_qpu_qasm_info_t info;
            char *text, *open, *close, body[512], program[1024], number[40];
            const double *amp = NULL;
            double a[8], distance;

            /* A state with four different amplitudes, then the operation. */
            parsec_qpu_circuit_add(native, room, PARSEC_QPU_OP_RY, 0, 0, PARSEC_QPU_NO_PARAM, 0.7);
            parsec_qpu_circuit_add(native, room, PARSEC_QPU_OP_RX, 1, 0, PARSEC_QPU_NO_PARAM, 1.1);
            parsec_qpu_circuit_add(native, room, PARSEC_QPU_OP_RZ, 0, 0, PARSEC_QPU_NO_PARAM, 0.4);
            parsec_qpu_circuit_add(native, room, PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0.0);
            parsec_qpu_circuit_add(native, room, PARSEC_QPU_OP_RY, 1, 0, PARSEC_QPU_NO_PARAM, -0.3);
            parsec_qpu_circuit_add(native, room, opcodes[g], 0, 1, PARSEC_QPU_NO_PARAM, thetas[t]);
            parsec_qpu_circuit_measure_all(native, room);
            EXPECT(PARSEC_SUCCESS == parsec_qpu_sampler_state(sampler, native, NULL, NULL, &amp), "%s: no state", names[g]);
            memcpy(a, amp, sizeof(a));

            /* The same with the body of the definition in place of the operation. */
            text = write_text(native, NULL);
            snprintf(body, sizeof(body), "gate %s(theta) a, b {", names[g]);
            open  = (NULL != text) ? strstr(text, body) : NULL;
            close = (NULL != open) ? strchr(open, '}') : NULL;
            EXPECT(NULL != close, "%s: the text does not define the gate\n%s", names[g], text ? text : "");
            if( NULL != close ) {
                snprintf(body, sizeof(body), "%.*s", (int)(close - (open + strlen(body))), open + strlen(body));
                snprintf(number, sizeof(number), "(%.17g)", thetas[t]);
                replace_word(body, sizeof(body), "theta", number);
                replace_word(body, sizeof(body), "a", "q[0]");
                replace_word(body, sizeof(body), "b", "q[1]");
                snprintf(program, sizeof(program),
                         "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[2] c;\n"
                         "ry(0.7) q[0]; rx(1.1) q[1]; rz(0.4) q[0]; cx q[0], q[1]; ry(-0.3) q[1];\n"
                         "%s\nc = measure q;\n", body);
                defined = read_text(program, &info);
                EXPECT(NULL != defined, "%s: the body of its definition is not read: line %d, %s\n%s", names[g], info.line, info.why, program);
                if( NULL != defined ) {
                    EXPECT(PARSEC_SUCCESS == parsec_qpu_sampler_state(sampler, defined, NULL, NULL, &amp), "%s: no state", names[g]);
                    distance = state_distance(a, amp, 2);
                    EXPECT(distance < 1e-12, "%s(%g): its definition prepares another state than the operation, %g apart\n%s",
                           names[g], thetas[t], distance, program);
                }
                free(defined);
            }
            free(text);
            free(native);
        }
    }
    parsec_qpu_sampler_free(sampler);
}

/* What qiskit.qasm3.dumps of Qiskit 2.6.2 wrote for a circuit of 3 qubits
 * with the parameters gamma and beta (Qiskit declares them by name). */
static const char qiskit_text[] =
    "OPENQASM 3.0;\n"
    "include \"stdgates.inc\";\n"
    "input float[64] beta;\n"
    "input float[64] gamma;\n"
    "gate rzz(p0) _gate_q_0, _gate_q_1 {\n"
    "  cx _gate_q_0, _gate_q_1;\n"
    "  rz(p0) _gate_q_1;\n"
    "  cx _gate_q_0, _gate_q_1;\n"
    "}\n"
    "gate rxx(p0) _gate_q_0, _gate_q_1 {\n"
    "  h _gate_q_0;\n"
    "  h _gate_q_1;\n"
    "  cx _gate_q_0, _gate_q_1;\n"
    "  rz(p0) _gate_q_1;\n"
    "  cx _gate_q_0, _gate_q_1;\n"
    "  h _gate_q_1;\n"
    "  h _gate_q_0;\n"
    "}\n"
    "gate sxdg _gate_q_0 {\n"
    "  s _gate_q_0;\n"
    "  h _gate_q_0;\n"
    "  s _gate_q_0;\n"
    "}\n"
    "gate ryy(p0) _gate_q_0, _gate_q_1 {\n"
    "  sxdg _gate_q_0;\n"
    "  sxdg _gate_q_1;\n"
    "  cx _gate_q_0, _gate_q_1;\n"
    "  rz(p0) _gate_q_1;\n"
    "  cx _gate_q_0, _gate_q_1;\n"
    "  sx _gate_q_0;\n"
    "  sx _gate_q_1;\n"
    "}\n"
    "bit[3] c;\n"
    "qubit[3] q;\n"
    "h q[0];\n"
    "h q[1];\n"
    "x q[2];\n"
    "y q[2];\n"
    "z q[2];\n"
    "rzz((-1.5)*gamma) q[0], q[1];\n"
    "rxx(0.25) q[1], q[2];\n"
    "ryy(2*beta) q[0], q[2];\n"
    "rz((-0.6)*gamma) q[0];\n"
    "rx(2*beta) q[1];\n"
    "ry(0.7) q[2];\n"
    "cx q[0], q[1];\n"
    "cz q[1], q[2];\n"
    "barrier q[0], q[1], q[2];\n"
    "c[0] = measure q[0];\n"
    "c[2] = measure q[2];\n";

static void test_qiskit_text(void)
{
    parsec_qpu_circuit_t *expected = circuit_new(3, 2, 16), *circuit;
    parsec_qpu_qasm_info_t info;
    const size_t room = ROOM(16);
    const int beta = 0, gamma = 1;           /* in the order of their declarations */

    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_H, 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_X, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_Y, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_Z, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_RZZ, 0, 1, gamma, -1.5);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_RXX, 1, 2, PARSEC_QPU_NO_PARAM, 0.25);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_RYY, 0, 2, beta, 2.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_RZ, 0, 0, gamma, -0.6);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_RX, 1, 0, beta, 2.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_RY, 2, 0, PARSEC_QPU_NO_PARAM, 0.7);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_CZ, 1, 2, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_MEASURE, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(expected, room, PARSEC_QPU_OP_MEASURE, 2, 0, PARSEC_QPU_NO_PARAM, 0.0);

    circuit = read_text(qiskit_text, &info);
    EXPECT(NULL != circuit, "the text Qiskit wrote is not read: line %d, %s", info.line, info.why);
    if( NULL != circuit ) {
        EXPECT(same_circuit(expected, circuit), "the text Qiskit wrote gives another circuit than the one it stands for");
        round_trip(circuit, "the circuit read from Qiskit's text");
    }
    EXPECT(beta == parsec_qpu_qasm_param_slot(qiskit_text, strlen(qiskit_text), "beta"), "beta is not slot %d", beta);
    EXPECT(gamma == parsec_qpu_qasm_param_slot(qiskit_text, strlen(qiskit_text), "gamma"), "gamma is not slot %d", gamma);
    EXPECT(PARSEC_ERR_NOT_FOUND == parsec_qpu_qasm_param_slot(qiskit_text, strlen(qiskit_text), "delta"), "delta was found");
    EXPECT(PARSEC_ERR_NOT_FOUND == parsec_qpu_qasm_param_slot(qiskit_text, strlen(qiskit_text), "p0"),
           "the parameter of a gate definition was taken for an input");

    /* The text need not end with a 0: read from a buffer with something else behind it. */
    {
        const size_t n = strlen(qiskit_text);
        char *copy = (char*)malloc(n + 16);
        parsec_qpu_circuit_t *again = (parsec_qpu_circuit_t*)calloc(1, room);
        memcpy(copy, qiskit_text, n);
        memcpy(copy + n, "h q[9]; garbage", 15);
        EXPECT(PARSEC_SUCCESS == parsec_qpu_qasm_read(copy, n, again, room, &info), "a text without a final 0 is not read: %s", info.why);
        EXPECT(same_circuit(expected, again), "a text without a final 0 gives another circuit");
        free(again);
        free(copy);
    }
    free(circuit);
    free(expected);
}

#define HEAD "OPENQASM 3; include \"stdgates.inc\"; input angle a; input float b; qubit[2] q; bit[2] c;\n"
#define TAIL "\nc = measure q;\n"

/* One rotation with the angle written as `expression`: its parameter slot
 * and its number. */
static void expect_angle(const char *expression, int param, double angle)
{
    parsec_qpu_qasm_info_t info;
    parsec_qpu_circuit_t *circuit;
    char text[512];

    snprintf(text, sizeof(text), HEAD "rz(%s) q[1];" TAIL, expression);
    circuit = read_text(text, &info);
    EXPECT(NULL != circuit, "rz(%s) is not read: line %d, %s", expression, info.line, info.why);
    if( NULL != circuit ) {
        const parsec_qpu_op_t *op = parsec_qpu_circuit_ops(circuit);
        EXPECT((3 == circuit->nb_ops) && (PARSEC_QPU_OP_RZ == op->opcode) && (1 == op->q0), "rz(%s) gave another operation", expression);
        EXPECT((param == op->param) && (fabs(op->angle - angle) <= 1e-15 * fabs(angle)),
               "rz(%s) is read as %.17g times slot %d, expected %.17g times slot %d", expression, op->angle, (int)op->param, angle, param);
    }
    free(circuit);
}

static void test_expressions(void)
{
    const double pi = 3.14159265358979323846;

    expect_angle("0.5", PARSEC_QPU_NO_PARAM, 0.5);
    expect_angle("2", PARSEC_QPU_NO_PARAM, 2.0);
    expect_angle(".5e1", PARSEC_QPU_NO_PARAM, 5.0);
    expect_angle("1_000", PARSEC_QPU_NO_PARAM, 1000.0);
    expect_angle("pi", PARSEC_QPU_NO_PARAM, pi);
    expect_angle("\xcf\x80", PARSEC_QPU_NO_PARAM, pi);
    expect_angle("-pi/2", PARSEC_QPU_NO_PARAM, -pi / 2);
    expect_angle("3*pi/4", PARSEC_QPU_NO_PARAM, 3 * pi / 4);
    expect_angle("tau", PARSEC_QPU_NO_PARAM, 2 * pi);
    expect_angle("(1 + 2) * (3 - 1) / 4", PARSEC_QPU_NO_PARAM, 1.5);
    expect_angle("- - 1", PARSEC_QPU_NO_PARAM, 1.0);
    expect_angle("a", 0, 1.0);
    expect_angle("b", 1, 1.0);
    expect_angle("-a", 0, -1.0);
    expect_angle("+b", 1, 1.0);
    expect_angle("2*a", 0, 2.0);
    expect_angle("a*2", 0, 2.0);
    expect_angle("(-1.5)*b", 1, -1.5);
    expect_angle("b/4", 1, 0.25);
    expect_angle("pi*a/2", 0, pi / 2);
    expect_angle("2*(3*a)", 0, 6.0);
    expect_angle("a + a", 0, 2.0);
    expect_angle("3*b - b", 1, 2.0);
    expect_angle("(1 + 2)*a/4", 0, 0.75);
    expect_angle("0*a + b", 1, 1.0);
    expect_angle("0.0 * a", 0, 0.0);
    expect_angle("/* in the middle */ 2 * // to the end of the line\n a", 0, 2.0);

    /* Signs and parentheses in numbers a text could hold, and more: the
     * reader follows any number of signs, and parentheses down to a limit,
     * after which it refuses. It does not run out of stack on either. */
    {
        const size_t many = 400000;
        char *text = (char*)malloc(2 * many + 256), *at;
        parsec_qpu_qasm_info_t info;
        parsec_qpu_circuit_t *circuit;
        int rc;

        at = text + sprintf(text, HEAD "rz(");
        memset(at, '-', many + 1); at += many + 1;          /* an odd number of them */
        sprintf(at, "a) q[1];" TAIL);
        circuit = read_text(text, &info);
        EXPECT((NULL != circuit) && (0 == parsec_qpu_circuit_ops(circuit)->param) && (-1.0 == parsec_qpu_circuit_ops(circuit)->angle),
               "%zu minus signs before an input are not read as one: %s", many + 1, info.why);
        free(circuit);

        for( int depth = 31; depth <= 33; depth++ ) {
            at = text + sprintf(text, HEAD "rz(");
            memset(at, '(', (size_t)depth); at += depth;
            at += sprintf(at, "0.5");
            memset(at, ')', (size_t)depth); at += depth;
            sprintf(at, ") q[1];" TAIL);
            circuit = read_text(text, &info);
            if( depth <= 32 ) {
                EXPECT((NULL != circuit) && (0.5 == parsec_qpu_circuit_ops(circuit)->angle), "%d parentheses are not read: %s", depth, info.why);
            } else {
                EXPECT((NULL == circuit) && (2 == info.line) && (NULL != strstr(info.why, "parentheses inside each other")),
                       "%d parentheses were not refused for being too many: %s", depth, info.why);
            }
            free(circuit);
        }
        at = text + sprintf(text, HEAD "rz(");
        memset(at, '(', many); at += many;
        sprintf(at, "0.5) q[1];" TAIL);
        rc = parsec_qpu_qasm_read(text, strlen(text), NULL, 0, &info);
        EXPECT((PARSEC_ERR_BAD_PARAM == rc) && (NULL != strstr(info.why, "parentheses inside each other")),
               "%zu parentheses that open were not refused (%d: %s)", many, rc, info.why);
        free(text);
    }
}

/* A text that has to be refused, on a given line, for a reason that holds `word`. */
static void expect_refused(const char *text, int line, const char *word)
{
    parsec_qpu_qasm_info_t info;
    char buffer[ROOM(64)];
    int rc = parsec_qpu_qasm_read(text, strlen(text), (parsec_qpu_circuit_t*)buffer, sizeof(buffer), &info);

    EXPECT(PARSEC_ERR_BAD_PARAM == rc, "this text was not refused (%d):\n%s", rc, text);
    if( PARSEC_ERR_BAD_PARAM != rc ) return;
    keep_text(text, NULL);
    EXPECT(line == info.line, "refused on line %d, expected line %d (%s):\n%s", info.line, line, info.why, text);
    EXPECT(NULL != strstr(info.why, word), "refused for \"%s\", expected a reason with \"%s\":\n%s", info.why, word, text);
    EXPECT(0 == info.size, "a refused text has a size, %zu", info.size);
    /* The same when only the size is asked for. */
    EXPECT(PARSEC_ERR_BAD_PARAM == parsec_qpu_qasm_read(text, strlen(text), NULL, 0, &info), "refused with a buffer, read without:\n%s", text);
}

static void test_refusals(void)
{
    /* Angles the circuit format does not hold. */
    expect_refused(HEAD "rz(a + 1) q[0];" TAIL, 2, "plus a number");
    expect_refused(HEAD "rz(a * b) q[0];" TAIL, 2, "product of inputs");
    expect_refused(HEAD "rz(a + b) q[0];" TAIL, 2, "sum of two inputs");
    expect_refused(HEAD "rz(1 / a) q[0];" TAIL, 2, "divided by an input");
    expect_refused(HEAD "rz(1 / 0) q[0];" TAIL, 2, "not a finite number");
    expect_refused(HEAD "rz(d) q[0];" TAIL, 2, "neither a number nor a declared input");
    expect_refused(HEAD "rz(2 ** a) q[0];" TAIL, 2, "a number or an input");
    expect_refused(HEAD "rz(0x10) q[0];" TAIL, 2, "only decimal numbers");
    expect_refused(HEAD "rz q[0];" TAIL, 2, "the angle of the rotation");
    expect_refused(HEAD "h(0.5) q[0];" TAIL, 2, "takes no angle");
    /* Operations and operands. */
    expect_refused(HEAD "sx q[0];" TAIL, 2, "'sx' is not in the part");
    expect_refused(HEAD "swap q[0], q[1];" TAIL, 2, "'swap' is not in the part");
    expect_refused(HEAD "reset q[0];" TAIL, 2, "'reset' is not in the part");
    expect_refused(HEAD "if (c[0]) { x q[1]; }" TAIL, 2, "'if' is not in the part");
    expect_refused(HEAD "h q[2];" TAIL, 2, "qubit 2 is used, and the register has 2");
    expect_refused(HEAD "h r[0];" TAIL, 2, "'r' is not the register of qubits");
    expect_refused(HEAD "h $0;" TAIL, 2, "physical qubits");
    expect_refused(HEAD "h q;" TAIL, 2, "on a whole register");
    expect_refused(HEAD "cx q[1], q[1];" TAIL, 2, "on qubit 1 twice");
    expect_refused(HEAD "cx q[0];" TAIL, 2, "the second qubit");
    expect_refused(HEAD "h q[0]" TAIL, 3, "';'");
    expect_refused(HEAD "h q[1.5];" TAIL, 2, "a whole number");
    /* Measurements. */
    expect_refused(HEAD "c[0] = measure q[1];", 2, "qubit 1 is measured into bit 0");
    expect_refused(HEAD "measure q[0];", 2, "goes to no bit");
    expect_refused(HEAD "c = measure q[0];", 2, "a register and one element");
    expect_refused(HEAD "c[0] = measure q[0]; h q[0];", 2, "follows the measurement");
    expect_refused(HEAD "c[0] = measure q[0]; cx q[1], q[0];", 2, "follows the measurement");
    expect_refused(HEAD "c[0] = measure q[0]; measure q[0] -> c[0];", 2, "follows the measurement");
    expect_refused(HEAD "h q[0];\n", 3, "measures nothing");
    /* The same in every word as the writer would write it, were such a circuit one. */
    expect_refused("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nx q[0];\nh q[1];\n", 7, "measures nothing");
    expect_refused("OPENQASM 3; qubit[2] q; bit[3] c; c = measure q;", 1, "of another size");
    expect_refused("OPENQASM 3; qubit[2] q; c[0] = measure q[0];", 1, "'c' is not in the part");
    expect_refused("OPENQASM 3; qubit[2] q; measure q[0] -> c[0];", 1, "before the register of bits");
    /* Declarations. */
    expect_refused("OPENQASM 2.0;\nqreg q[2];", 1, "another version");
    expect_refused("OPENQASM 3;\ninclude \"qelib1.inc\";", 2, "stdgates.inc");
    expect_refused("OPENQASM 3;\n\nqubit[2] q;\nqubit[2] r;\n", 4, "a second register of qubits");
    expect_refused("OPENQASM 3;\nqubit[65] q;\n", 2, "between 1 and 64");
    expect_refused("OPENQASM 3;\nqubit[0] q;\n", 2, "between 1 and 64");
    expect_refused("OPENQASM 3;\nbit[2] c;\nh q[0];\n", 3, "before the register of qubits");
    expect_refused("OPENQASM 3;\ninput angle a;\ninput angle a;\n", 3, "declared twice");
    expect_refused("OPENQASM 3;\ninput angle pi;\n", 2, "name of a constant");
    expect_refused("OPENQASM 3;\ninput int[32] n;\n", 2, "of type angle or float");
    expect_refused("OPENQASM 3;\ninput angle a;\nqubit[2] a;\n", 3, "declared twice");
    expect_refused("OPENQASM 3;\nbit[2] c;\n", 3, "no register of qubits");
    expect_refused("", 1, "no register of qubits");
    expect_refused("OPENQASM 3;\ngate g a { h a;\nqubit[2] q;\n", 2, "not closed");
    expect_refused("OPENQASM 3;\ngate g a;\n", 2, "has no body");
    /* The text itself. */
    expect_refused("OPENQASM 3;\nqubit[2] q; /* never\nclosed", 2, "comment is not closed");
    expect_refused("OPENQASM 3;\ninclude \"stdgates.inc;\n", 2, "string is not closed");
    expect_refused("OPENQASM 3;\nqubit[2] q;\nh q[0] # no;\n", 3, "character that is not read");
    expect_refused("OPENQASM 3;\nqubit[2] q; bit[2] c;\n; c = measure q;", 3, "a statement");
}

static void test_other_forms(void)
{
    static const char old_style[] =
        "OPENQASM 3;\ninclude \"stdgates.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n";
    static const char one_by_one[] =
        "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\nc[1] = measure q[1];\n";
    static const char single[] = "qubit q;\nbit c;\nh q;\nc = measure q;\n";
    static const char more_bits[] = "qubit[2] q; bit[5] c; h q[1]; c[1] = measure q[1];";
    parsec_qpu_circuit_t *bell = circuit_new(2, 0, 4), *circuit;
    parsec_qpu_qasm_info_t info;

    parsec_qpu_circuit_add(bell, ROOM(4), PARSEC_QPU_OP_H, 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(bell, ROOM(4), PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_measure_all(bell, ROOM(4));

    circuit = read_text(old_style, &info);
    EXPECT((NULL != circuit) && same_circuit(bell, circuit), "qreg, creg and 'measure q -> c' do not give a Bell pair: %s", info.why);
    free(circuit);
    circuit = read_text(one_by_one, &info);
    EXPECT((NULL != circuit) && same_circuit(bell, circuit), "measurements one by one do not give a Bell pair: %s", info.why);
    free(circuit);
    /* "qubit q;" is a register of one qubit, and a gate on it is a gate on a whole register. */
    EXPECT(NULL == read_text(single, &info), "a gate on a register of one qubit, written without its index, was read");
    circuit = read_text("qubit q;\nbit c;\nh q[0];\nc = measure q;\n", &info);
    EXPECT((NULL != circuit) && (1 == circuit->nb_qubits) && (2 == circuit->nb_ops), "a register of one qubit is not read: %s", info.why);
    free(circuit);
    circuit = read_text(more_bits, &info);
    EXPECT((NULL != circuit) && (2 == circuit->nb_qubits) && (2 == circuit->nb_ops) &&
           (0x2 == parsec_qpu_circuit_measured(circuit)), "more bits than qubits: %s", info.why);
    free(circuit);
    free(bell);

    /* What the writer writes, with one thing spelt another way: a number
     * with a digit more, with an exponent, without its point, with fewer
     * digits than the writer gives it; a definition that is not used. Each
     * is the same circuit to the reader. (And each is kept with -o: a reader
     * of the writer's texts alone, which could take them for what they look
     * like, has to refuse them.) */
    {
        static const char *spelt[][2] = {
            { "(0.5)", "(0.50)" }, { "(0.5)", "(5e-1)" }, { "(2.0 * p0)", "(2 * p0)" }, { "(0.10000000000000001)", "(0.1)" },
            { "qubit[2] q;", "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\nqubit[2] q;" }
        };
        parsec_qpu_circuit_t *c = circuit_new(2, 1, 6), *back;
        char *text, other[1024];

        add_op(c, ROOM(6), PARSEC_QPU_OP_RX, 0, 0, PARSEC_QPU_NO_PARAM, 0.5);
        add_op(c, ROOM(6), PARSEC_QPU_OP_RY, 1, 0, 0, 2.0);
        add_op(c, ROOM(6), PARSEC_QPU_OP_RZ, 0, 0, PARSEC_QPU_NO_PARAM, 0.1);
        add_op(c, ROOM(6), PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0.0);
        parsec_qpu_circuit_measure_all(c, ROOM(6));
        text = write_text(c, NULL);
        EXPECT(NULL != text, "the circuit of the spellings could not be written");
        for( size_t i = 0; (NULL != text) && (i < sizeof(spelt) / sizeof(spelt[0])); i++ ) {
            const char *at = strstr(text, spelt[i][0]);
            EXPECT((NULL != at) && (NULL == strstr(at + 1, spelt[i][0])), "the writer did not write '%s' once:\n%s", spelt[i][0], text);
            if( NULL == at ) continue;
            snprintf(other, sizeof(other), "%.*s%s%s", (int)(at - text), text, spelt[i][1], at + strlen(spelt[i][0]));
            back = read_text(other, &info);
            EXPECT((NULL != back) && same_circuit(c, back), "'%s' written as '%s' is not the same circuit: %s", spelt[i][0], spelt[i][1], info.why);
            free(back);
        }
        free(text);
        free(c);
    }

    /* The writer's text again, with one thing wrong: each is no circuit, and
     * is refused, or is another text for the same circuit, and is taken.
     * (Kept with -o as well. A reader of the writer's texts alone has to
     * refuse every one of them, and nothing but the one thing is in its way:
     * the rest of the text is as the writer writes it.) */
    {
        static const struct { const char *from, *to; int refused; } wrong[] = {
            { "rx(0.5) q[0];", "rx(0.5) q[2];", 1 },                                /* a qubit the circuit does not have */
            { "cx q[0], q[1];", "cx q[1], q[1];", 1 },                              /* one qubit twice */
            { "(2.0 * p0)", "(2.0 * p1)", 1 },                                      /* an input that is not declared */
            { "c[1] = measure q[1];", "c[0] = measure q[1];", 1 },                  /* a qubit measured into another bit */
            { "c[0] = measure q[0];\n", "c[0] = measure q[0];\nx q[0];\n", 1 },    /* an operation after the measurement of its qubit */
            { "rx(0.5)", "rx(1e999)", 1 },                                          /* an angle that is no finite number */
            { "qubit[2] q;\nbit[2] c;", "qubit[65] q;\nbit[65] c;", 1 },           /* more qubits than a circuit has */
            { "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\n", "", 0 }, /* rzz without its definition: the name says it */
            { "rz(theta) b; cx a, b; }", "rz(theta) a; cx a, b; }", 0 }             /* another definition: the reader skips it */
        };
        parsec_qpu_circuit_t *c = circuit_new(2, 1, 6), *back;
        char *text, other[1024];

        add_op(c, ROOM(6), PARSEC_QPU_OP_RX, 0, 0, PARSEC_QPU_NO_PARAM, 0.5);
        add_op(c, ROOM(6), PARSEC_QPU_OP_RZZ, 0, 1, 0, 2.0);
        add_op(c, ROOM(6), PARSEC_QPU_OP_CX, 0, 1, PARSEC_QPU_NO_PARAM, 0.0);
        parsec_qpu_circuit_measure_all(c, ROOM(6));
        text = write_text(c, NULL);
        EXPECT(NULL != text, "the circuit of the texts with one thing wrong could not be written");
        for( size_t i = 0; (NULL != text) && (i < sizeof(wrong) / sizeof(wrong[0])); i++ ) {
            const char *at = strstr(text, wrong[i].from);
            EXPECT((NULL != at) && (NULL == strstr(at + 1, wrong[i].from)), "the writer did not write '%s' once:\n%s", wrong[i].from, text);
            if( NULL == at ) continue;
            snprintf(other, sizeof(other), "%.*s%s%s", (int)(at - text), text, wrong[i].to, at + strlen(wrong[i].from));
            back = read_text(other, &info);
            if( wrong[i].refused ) {
                EXPECT(NULL == back, "'%s' in place of '%s' was read", wrong[i].to, wrong[i].from);
                if( NULL == back ) keep_text(other, NULL);
            } else {
                EXPECT((NULL != back) && same_circuit(c, back), "'%s' in place of '%s' is not the same circuit: %s", wrong[i].to, wrong[i].from, info.why);
            }
            free(back);
        }
        free(text);
        free(c);
    }
}

static void test_sizes(void)
{
    parsec_qpu_qasm_info_t info;
    char buffer[ROOM(15)];
    int rc;

    rc = parsec_qpu_qasm_read(qiskit_text, strlen(qiskit_text), NULL, 0, &info);
    EXPECT((PARSEC_SUCCESS == rc) && (ROOM(15) == info.size) && (15 == info.nb_ops) && (3 == info.nb_qubits) && (2 == info.nb_params),
           "the size of Qiskit's text: %d, %zu bytes, %u operations", rc, info.size, info.nb_ops);
    /* One byte short, and a buffer smaller than a header. */
    memset(buffer, 0x5a, sizeof(buffer));
    rc = parsec_qpu_qasm_read(qiskit_text, strlen(qiskit_text), (parsec_qpu_circuit_t*)buffer, sizeof(buffer) - 1, &info);
    EXPECT((PARSEC_ERR_OUT_OF_RESOURCE == rc) && (ROOM(15) == info.size), "a buffer one byte short: %d, %zu", rc, info.size);
    EXPECT(0x5a == buffer[sizeof(buffer) - 1], "the reader wrote past the room it was given");
    rc = parsec_qpu_qasm_read(qiskit_text, strlen(qiskit_text), (parsec_qpu_circuit_t*)buffer, 8, &info);
    EXPECT((PARSEC_ERR_OUT_OF_RESOURCE == rc) && (ROOM(15) == info.size), "a buffer of 8 bytes: %d, %zu", rc, info.size);
    rc = parsec_qpu_qasm_read(qiskit_text, strlen(qiskit_text), (parsec_qpu_circuit_t*)buffer, sizeof(buffer), &info);
    EXPECT(PARSEC_SUCCESS == rc, "a buffer of the right size: %d", rc);
    rc = parsec_qpu_qasm_read(NULL, 0, NULL, 0, &info);
    EXPECT((PARSEC_ERR_BAD_PARAM == rc) && (1 == info.line), "no text: %d", rc);
    EXPECT(PARSEC_SUCCESS == parsec_qpu_qasm_read(qiskit_text, strlen(qiskit_text), NULL, 0, NULL), "no info asked for");
}

int main(int argc, char *argv[])
{
    if( (3 == argc) && (0 == strcmp(argv[1], "-o")) ) {
        keep_dir = argv[2];
        if( (0 != mkdir(keep_dir, 0777)) && (EEXIST != errno) ) {
            fprintf(stderr, "%s: cannot make the directory %s\n", argv[0], keep_dir);
            return 2;
        }
        forget_kept_texts();
    } else if( 1 != argc ) {
        fprintf(stderr, "usage: %s [-o <directory to keep the texts in>]\n", argv[0]);
        return 2;
    }
    test_written_text();
    test_round_trips();
    test_definitions();
    test_qiskit_text();
    test_expressions();
    test_refusals();
    test_other_forms();
    test_sizes();
    if( nb_errors ) {
        fprintf(stderr, "qpu_qasm_test: %d error(s)\n", nb_errors);
        return 1;
    }
    if( NULL != keep_dir ) printf("qpu_qasm_test: %d texts and their circuits, %d refused texts and %d texts of other forms are in %s\n",
                                  nb_kept, nb_kept_refused, nb_kept_other, keep_dir);
    printf("qpu_qasm_test: passed\n");
    return 0;
}
