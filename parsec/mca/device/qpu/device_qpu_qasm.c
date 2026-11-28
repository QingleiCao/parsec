/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_qasm.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Circuits as OpenQASM 3 text: see device_qpu_qasm.h for what is written and
 * what is read.
 */

#define QASM_MAX_DEPTH   32

#define QASM_ONE_QUBIT   0x1
#define QASM_TWO_QUBITS  0x2
#define QASM_ROTATION    0x4

static const struct qasm_gate_s {
    const char *name;
    int         opcode;
    int         kind;
} qasm_gates[] = {
    { "h",   PARSEC_QPU_OP_H,   QASM_ONE_QUBIT },
    { "x",   PARSEC_QPU_OP_X,   QASM_ONE_QUBIT },
    { "y",   PARSEC_QPU_OP_Y,   QASM_ONE_QUBIT },
    { "z",   PARSEC_QPU_OP_Z,   QASM_ONE_QUBIT },
    { "rx",  PARSEC_QPU_OP_RX,  QASM_ONE_QUBIT  | QASM_ROTATION },
    { "ry",  PARSEC_QPU_OP_RY,  QASM_ONE_QUBIT  | QASM_ROTATION },
    { "rz",  PARSEC_QPU_OP_RZ,  QASM_ONE_QUBIT  | QASM_ROTATION },
    { "cx",  PARSEC_QPU_OP_CX,  QASM_TWO_QUBITS },
    { "cz",  PARSEC_QPU_OP_CZ,  QASM_TWO_QUBITS },
    { "rxx", PARSEC_QPU_OP_RXX, QASM_TWO_QUBITS | QASM_ROTATION },
    { "ryy", PARSEC_QPU_OP_RYY, QASM_TWO_QUBITS | QASM_ROTATION },
    { "rzz", PARSEC_QPU_OP_RZZ, QASM_TWO_QUBITS | QASM_ROTATION },
    { NULL, 0, 0 }
};

static const struct qasm_gate_s *qasm_gate_of_opcode(int opcode)
{
    for( int i = 0; NULL != qasm_gates[i].name; i++ ) {
        if( qasm_gates[i].opcode == opcode ) return &qasm_gates[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------------
 * Writer
 * --------------------------------------------------------------------- */

typedef struct qasm_out_s {
    char  *text;
    size_t room;
    size_t length;      /* of everything written so far, fitted or not */
} qasm_out_t;

static void qasm_out(qasm_out_t *out, const char *format, ...)
{
    const size_t left = (out->length < out->room) ? out->room - out->length : 0;
    va_list ap;
    int n;

    va_start(ap, format);
    n = vsnprintf((0 != left) ? out->text + out->length : NULL, left, format, ap);
    va_end(ap);
    if( n > 0 ) out->length += (size_t)n;
}

/* A double as a literal that reads back as the same double, and that is a
 * floating-point literal whatever its value. */
static const char *qasm_number(char buffer[40], double value)
{
    (void)parsec_qpu_number_format(buffer, value);
    if( NULL == strpbrk(buffer, ".e") ) strcat(buffer, ".0");
    return buffer;
}

/* rxx, ryy and rzz from gates of stdgates.inc. With R(t) = exp(-i t P / 2):
 * cx, rz(t) on the target, cx is exp(-i t ZZ / 2); h on both sides turns Z
 * into X; and rx(-pi/2) Z rx(pi/2) = Y, so rx(pi/2) before and rx(-pi/2)
 * after turn Z into Y. */
static const char *qasm_definitions[] = {
    [PARSEC_QPU_OP_RXX] = "gate rxx(theta) a, b { h a; h b; cx a, b; rz(theta) b; cx a, b; h b; h a; }\n",
    [PARSEC_QPU_OP_RYY] = "gate ryy(theta) a, b { rx(pi/2) a; rx(pi/2) b; cx a, b; rz(theta) b; cx a, b; "
                          "rx(-pi/2) a; rx(-pi/2) b; }\n",
    [PARSEC_QPU_OP_RZZ] = "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\n"
};

int64_t parsec_qpu_qasm_write(const parsec_qpu_circuit_t *circuit, char *text, size_t room)
{
    qasm_out_t out = { .text = text, .room = (NULL != text) ? room : 0, .length = 0 };
    const parsec_qpu_op_t *ops;
    int uses[PARSEC_QPU_OP_MAX] = { 0 };
    char number[40];

    if( PARSEC_SUCCESS != parsec_qpu_circuit_check(circuit, 0, NULL) ) return PARSEC_ERR_BAD_PARAM;
    if( PARSEC_QPU_CIRCUIT_EMPTY & circuit->flags ) return PARSEC_ERR_BAD_PARAM;
    if( (NULL != text) && (room > 0) ) text[0] = '\0';

    ops = parsec_qpu_circuit_ops(circuit);
    for( uint32_t i = 0; i < circuit->nb_ops; i++ ) uses[ops[i].opcode] = 1;

    qasm_out(&out, "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
    for( uint32_t p = 0; p < circuit->nb_params; p++ ) {
        qasm_out(&out, "input float[64] p%u;\n", p);
    }
    for( int opcode = PARSEC_QPU_OP_RXX; opcode <= PARSEC_QPU_OP_RZZ; opcode++ ) {
        if( uses[opcode] ) qasm_out(&out, "%s", qasm_definitions[opcode]);
    }
    qasm_out(&out, "qubit[%u] q;\nbit[%u] c;\n", circuit->nb_qubits, circuit->nb_qubits);

    for( uint32_t i = 0; i < circuit->nb_ops; i++ ) {
        const parsec_qpu_op_t *op = &ops[i];
        const struct qasm_gate_s *gate;

        if( PARSEC_QPU_OP_MEASURE == op->opcode ) {
            qasm_out(&out, "c[%u] = measure q[%u];\n", (unsigned)op->q0, (unsigned)op->q0);
            continue;
        }
        gate = qasm_gate_of_opcode(op->opcode);       /* the check above found the opcode known */
        qasm_out(&out, "%s", gate->name);
        if( QASM_ROTATION & gate->kind ) {
            if( PARSEC_QPU_NO_PARAM == op->param ) {
                qasm_out(&out, "(%s)", qasm_number(number, op->angle));
            } else {
                qasm_out(&out, "(%s * p%d)", qasm_number(number, op->angle), (int)op->param);
            }
        }
        if( QASM_TWO_QUBITS & gate->kind ) {
            qasm_out(&out, " q[%u], q[%u];\n", (unsigned)op->q0, (unsigned)op->q1);
        } else {
            qasm_out(&out, " q[%u];\n", (unsigned)op->q0);
        }
    }
    return (int64_t)out.length;
}

/* ------------------------------------------------------------------------
 * Reader: the tokens
 * --------------------------------------------------------------------- */

enum { QASM_T_END = 0, QASM_T_ID, QASM_T_NUM, QASM_T_STR, QASM_T_SYM, QASM_T_ARROW, QASM_T_BAD };

typedef struct qasm_lex_s {
    const char *p, *end;
    int         line;         /* of the text at p */
    /* The current token. */
    int         kind;
    const char *tok;          /* its text; for a string, what is between the quotes */
    size_t      len;
    int         tok_line;
    double      num;
    int         is_int;       /* the number was written without a point or an exponent */
    const char *bad;          /* QASM_T_BAD: what is wrong */
} qasm_lex_t;

static int qasm_is_id_start(int c) { return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')) || ('_' == c); }
static int qasm_is_digit(int c)    { return (c >= '0') && (c <= '9'); }

static void qasm_lex_next(qasm_lex_t *lx)
{
    /* Blanks and comments. */
    for(;;) {
        while( (lx->p < lx->end) && ((' ' == *lx->p) || ('\t' == *lx->p) || ('\r' == *lx->p) || ('\n' == *lx->p)) ) {
            if( '\n' == *lx->p ) lx->line++;
            lx->p++;
        }
        if( (lx->p + 1 < lx->end) && ('/' == lx->p[0]) && ('/' == lx->p[1]) ) {
            while( (lx->p < lx->end) && ('\n' != *lx->p) ) lx->p++;
            continue;
        }
        if( (lx->p + 1 < lx->end) && ('/' == lx->p[0]) && ('*' == lx->p[1]) ) {
            const int start = lx->line;
            lx->p += 2;
            while( (lx->p + 1 < lx->end) && !(('*' == lx->p[0]) && ('/' == lx->p[1])) ) {
                if( '\n' == *lx->p ) lx->line++;
                lx->p++;
            }
            if( lx->p + 1 >= lx->end ) {
                lx->kind = QASM_T_BAD; lx->bad = "a comment is not closed"; lx->tok_line = start;
                lx->p = lx->end;
                return;
            }
            lx->p += 2;
            continue;
        }
        break;
    }

    lx->tok = lx->p;
    lx->len = 0;
    lx->tok_line = lx->line;
    if( lx->p >= lx->end ) { lx->kind = QASM_T_END; return; }

    if( qasm_is_id_start((unsigned char)*lx->p) || ('$' == *lx->p) ) {
        lx->p++;
        while( (lx->p < lx->end) && (qasm_is_id_start((unsigned char)*lx->p) || qasm_is_digit((unsigned char)*lx->p)) ) lx->p++;
        lx->kind = QASM_T_ID;
        lx->len  = (size_t)(lx->p - lx->tok);
        return;
    }
    /* The Greek letter pi, in UTF-8. */
    if( (lx->p + 1 < lx->end) && (0xcf == (unsigned char)lx->p[0]) && (0x80 == (unsigned char)lx->p[1]) ) {
        lx->p += 2;
        lx->kind = QASM_T_ID; lx->tok = "pi"; lx->len = 2;
        return;
    }
    if( qasm_is_digit((unsigned char)*lx->p) ||
        (('.' == *lx->p) && (lx->p + 1 < lx->end) && qasm_is_digit((unsigned char)lx->p[1])) ) {
        char copy[64];
        size_t n = 0;
        lx->is_int = 1;
        while( (lx->p < lx->end) && (qasm_is_digit((unsigned char)*lx->p) || ('_' == *lx->p)) ) {
            if( ('_' != *lx->p) && (n < sizeof(copy) - 1) ) copy[n++] = *lx->p;
            lx->p++;
        }
        if( (lx->p < lx->end) && ('.' == *lx->p) ) {
            lx->is_int = 0;
            if( n < sizeof(copy) - 1 ) copy[n++] = '.';
            lx->p++;
            while( (lx->p < lx->end) && qasm_is_digit((unsigned char)*lx->p) ) {
                if( n < sizeof(copy) - 1 ) copy[n++] = *lx->p;
                lx->p++;
            }
        }
        if( (lx->p < lx->end) && (('e' == *lx->p) || ('E' == *lx->p)) ) {
            const char *q = lx->p + 1;
            if( (q < lx->end) && (('+' == *q) || ('-' == *q)) ) q++;
            if( (q < lx->end) && qasm_is_digit((unsigned char)*q) ) {
                lx->is_int = 0;
                while( lx->p < q ) { if( n < sizeof(copy) - 1 ) copy[n++] = *lx->p; lx->p++; }
                while( (lx->p < lx->end) && qasm_is_digit((unsigned char)*lx->p) ) {
                    if( n < sizeof(copy) - 1 ) copy[n++] = *lx->p;
                    lx->p++;
                }
            }
        }
        lx->len = (size_t)(lx->p - lx->tok);
        if( n >= sizeof(copy) - 1 ) {
            lx->kind = QASM_T_BAD; lx->bad = "a number is too long";
            return;
        }
        /* A letter right after a number: 0x10, 1im, 10ns and the like. */
        if( (lx->p < lx->end) && qasm_is_id_start((unsigned char)*lx->p) ) {
            lx->kind = QASM_T_BAD; lx->bad = "a number of a kind that is not read (only decimal numbers are)";
            return;
        }
        copy[n] = '\0';
        lx->num  = parsec_qpu_number_parse(copy);
        lx->kind = QASM_T_NUM;
        return;
    }
    if( '"' == *lx->p ) {
        lx->p++;
        lx->tok = lx->p;
        while( (lx->p < lx->end) && ('"' != *lx->p) && ('\n' != *lx->p) ) lx->p++;
        if( (lx->p >= lx->end) || ('"' != *lx->p) ) {
            lx->kind = QASM_T_BAD; lx->bad = "a string is not closed";
            return;
        }
        lx->len = (size_t)(lx->p - lx->tok);
        lx->p++;
        lx->kind = QASM_T_STR;
        return;
    }
    if( (lx->p + 1 < lx->end) && ('-' == lx->p[0]) && ('>' == lx->p[1]) ) {
        lx->p += 2;
        lx->kind = QASM_T_ARROW; lx->len = 2;
        return;
    }
    if( NULL != memchr(";,()[]{}=+-*/", *lx->p, 13) ) {
        lx->p++;
        lx->kind = QASM_T_SYM; lx->len = 1;
        return;
    }
    lx->kind = QASM_T_BAD; lx->bad = "a character that is not read";
    lx->len = 1;
    lx->p++;
}

static void qasm_lex_init(qasm_lex_t *lx, const char *text, size_t length)
{
    memset(lx, 0, sizeof(*lx));
    lx->p    = text;
    lx->end  = text + length;
    lx->line = 1;
    qasm_lex_next(lx);
}

static int qasm_tok_is(const qasm_lex_t *lx, int kind, const char *text)
{
    return (kind == lx->kind) && (strlen(text) == lx->len) && (0 == memcmp(lx->tok, text, lx->len));
}
#define QASM_IS_ID(LX, TEXT)   qasm_tok_is((LX), QASM_T_ID, (TEXT))
#define QASM_IS_SYM(LX, CHAR)  ((QASM_T_SYM == (LX)->kind) && ((CHAR) == (LX)->tok[0]))

/* ------------------------------------------------------------------------
 * Reader: the statements
 * --------------------------------------------------------------------- */

typedef struct qasm_name_s {
    const char *name;
    size_t      len;
} qasm_name_t;

typedef struct qasm_reader_s {
    qasm_lex_t  lx;
    parsec_qpu_op_t *ops;         /* where the operations go, or NULL */
    uint32_t    ops_room;         /* how many fit there */
    uint32_t    nb_ops;
    uint32_t    nb_qubits, nb_bits;
    qasm_name_t qreg, creg;       /* name is NULL until the register is declared */
    qasm_name_t *params;
    uint32_t    nb_params, params_room;
    uint64_t    measured;
    int         depth;            /* parentheses open in the angle being read */
    int         line;             /* of the first thing that is wrong; 0 when nothing is */
    char        why[160];
} qasm_reader_t;

static int qasm_fail(qasm_reader_t *rd, int line, const char *format, ...)
{
    if( 0 == rd->line ) {
        va_list ap;
        rd->line = (line > 0) ? line : 1;
        va_start(ap, format);
        vsnprintf(rd->why, sizeof(rd->why), format, ap);
        va_end(ap);
    }
    return -1;
}

/* The token that is there is not what the statement needs. */
static int qasm_unexpected(qasm_reader_t *rd, const char *wanted)
{
    const qasm_lex_t *lx = &rd->lx;
    if( QASM_T_BAD == lx->kind ) return qasm_fail(rd, lx->tok_line, "%s", lx->bad);
    if( QASM_T_END == lx->kind ) return qasm_fail(rd, lx->tok_line, "the text ends where %s is expected", wanted);
    return qasm_fail(rd, lx->tok_line, "'%.*s' where %s is expected", (int)((lx->len < 40) ? lx->len : 40), lx->tok, wanted);
}

static int qasm_expect_sym(qasm_reader_t *rd, char sym, const char *wanted)
{
    if( !QASM_IS_SYM(&rd->lx, sym) ) return qasm_unexpected(rd, wanted);
    qasm_lex_next(&rd->lx);
    return 0;
}

static int qasm_same_name(const qasm_name_t *name, const qasm_lex_t *lx)
{
    return (NULL != name->name) && (name->len == lx->len) && (0 == memcmp(name->name, lx->tok, lx->len));
}

static int qasm_find_param(const qasm_reader_t *rd, const qasm_lex_t *lx)
{
    for( uint32_t p = 0; p < rd->nb_params; p++ ) {
        if( qasm_same_name(&rd->params[p], lx) ) return (int)p;
    }
    return -1;
}

/* "[ <integer> ]" when it is there: the integer, or -1 when there is no
 * bracket, or -2 after a failure. */
static int64_t qasm_index(qasm_reader_t *rd)
{
    int64_t value;
    if( !QASM_IS_SYM(&rd->lx, '[') ) return -1;
    qasm_lex_next(&rd->lx);
    if( (QASM_T_NUM != rd->lx.kind) || !rd->lx.is_int || (rd->lx.num > 1e9) ) {
        (void)qasm_unexpected(rd, "a whole number");
        return -2;
    }
    value = (int64_t)rd->lx.num;
    qasm_lex_next(&rd->lx);
    if( 0 != qasm_expect_sym(rd, ']', "']'") ) return -2;
    return value;
}

/* ---- Angles: a number, or a number times one parameter ---- */

typedef struct qasm_angle_s {
    double c;     /* the constant part */
    double k;     /* the coefficient of the parameter */
    int    p;     /* the parameter, or -1 */
} qasm_angle_t;

static int qasm_expr(qasm_reader_t *rd, qasm_angle_t *value);

static int qasm_primary(qasm_reader_t *rd, qasm_angle_t *value)
{
    qasm_lex_t *lx = &rd->lx;

    value->c = 0.0; value->k = 0.0; value->p = -1;
    if( QASM_T_NUM == lx->kind ) {
        value->c = lx->num;
        qasm_lex_next(lx);
        return 0;
    }
    if( QASM_IS_SYM(lx, '(') ) {
        /* An angle is read by functions that call each other, one level
         * for each parenthesis: a text may not ask for more than a few. */
        if( rd->depth >= QASM_MAX_DEPTH ) {
            return qasm_fail(rd, lx->tok_line, "an angle has more than %d parentheses inside each other", QASM_MAX_DEPTH);
        }
        rd->depth++;
        qasm_lex_next(lx);
        if( 0 != qasm_expr(rd, value) ) return -1;
        rd->depth--;
        return qasm_expect_sym(rd, ')', "')'");
    }
    if( QASM_T_ID == lx->kind ) {
        int p;
        if( QASM_IS_ID(lx, "pi") )         value->c = 3.14159265358979323846;
        else if( QASM_IS_ID(lx, "tau") )   value->c = 6.28318530717958647692;
        else if( QASM_IS_ID(lx, "euler") ) value->c = 2.71828182845904523536;
        else if( (p = qasm_find_param(rd, lx)) >= 0 ) { value->k = 1.0; value->p = p; }
        else {
            return qasm_fail(rd, lx->tok_line, "'%.*s' in an angle is neither a number nor a declared input",
                             (int)((lx->len < 40) ? lx->len : 40), lx->tok);
        }
        qasm_lex_next(lx);
        return 0;
    }
    return qasm_unexpected(rd, "a number or an input");
}

static int qasm_unary(qasm_reader_t *rd, qasm_angle_t *value)
{
    int negative = 0;

    /* Any number of signs, counted and not followed one call each. */
    while( QASM_IS_SYM(&rd->lx, '-') || QASM_IS_SYM(&rd->lx, '+') ) {
        if( '-' == rd->lx.tok[0] ) negative = !negative;
        qasm_lex_next(&rd->lx);
    }
    if( 0 != qasm_primary(rd, value) ) return -1;
    if( negative ) {
        value->c = -value->c;
        value->k = -value->k;
    }
    return 0;
}

/* Whether a value depends on its parameter. */
#define QASM_VARIES(V)  (((V)->p >= 0) && (0.0 != (V)->k))

static int qasm_term(qasm_reader_t *rd, qasm_angle_t *value)
{
    if( 0 != qasm_unary(rd, value) ) return -1;
    while( QASM_IS_SYM(&rd->lx, '*') || QASM_IS_SYM(&rd->lx, '/') ) {
        const char op = rd->lx.tok[0];
        const int line = rd->lx.tok_line;
        qasm_angle_t right;

        qasm_lex_next(&rd->lx);
        if( 0 != qasm_unary(rd, &right) ) return -1;
        if( '*' == op ) {
            if( QASM_VARIES(value) && QASM_VARIES(&right) ) {
                return qasm_fail(rd, line, "an angle is a product of inputs: the circuit format holds a number times one input");
            }
            /* One product and no sum, so that a coefficient is read back
             * as the double that was written, the sign of a zero included. */
            if( QASM_VARIES(&right) ) {             /* a number times an input */
                value->k = value->c * right.k;
                value->p = right.p;
            } else {                                /* anything times a number */
                value->k = value->k * right.c;
                if( value->p < 0 ) value->p = right.p;      /* 0 times an input keeps its slot */
            }
            value->c = value->c * right.c;
        } else {
            if( QASM_VARIES(&right) ) {
                return qasm_fail(rd, line, "an angle is divided by an input: the circuit format holds a number times one input");
            }
            value->c = value->c / right.c;
            value->k = value->k / right.c;
        }
    }
    return 0;
}

static int qasm_expr(qasm_reader_t *rd, qasm_angle_t *value)
{
    if( 0 != qasm_term(rd, value) ) return -1;
    while( QASM_IS_SYM(&rd->lx, '+') || QASM_IS_SYM(&rd->lx, '-') ) {
        const double sign = ('-' == rd->lx.tok[0]) ? -1.0 : 1.0;
        const int line = rd->lx.tok_line;
        qasm_angle_t right;

        qasm_lex_next(&rd->lx);
        if( 0 != qasm_term(rd, &right) ) return -1;
        if( QASM_VARIES(value) && QASM_VARIES(&right) && (value->p != right.p) ) {
            return qasm_fail(rd, line, "an angle is a sum of two inputs: the circuit format holds a number times one input");
        }
        value->c += sign * right.c;
        if( right.p >= 0 ) {
            if( value->p == right.p ) value->k += sign * right.k;
            else if( !QASM_VARIES(value) ) { value->k = sign * right.k; value->p = right.p; }
            /* else the right side is 0 times another input, and adds nothing */
        }
    }
    return 0;
}

/* The angle of a rotation, between its parentheses. */
static int qasm_angle(qasm_reader_t *rd, int *param, double *angle)
{
    const int line = rd->lx.tok_line;
    qasm_angle_t value;

    if( 0 != qasm_expect_sym(rd, '(', "'(' and the angle of the rotation") ) return -1;
    rd->depth = 0;
    if( 0 != qasm_expr(rd, &value) ) return -1;
    if( 0 != qasm_expect_sym(rd, ')', "')'") ) return -1;
    if( (value.p >= 0) && (0.0 == value.c) ) {
        *param = value.p;
        *angle = value.k;
    } else if( QASM_VARIES(&value) ) {
        return qasm_fail(rd, line, "an angle is an input plus a number: the circuit format holds a number times one input");
    } else {
        *param = PARSEC_QPU_NO_PARAM;
        *angle = value.c;
    }
    if( !isfinite(*angle) ) return qasm_fail(rd, line, "an angle is not a finite number");
    return 0;
}

/* ---- Operands ---- */

/* A qubit of the register, or the whole register (-1). */
static int qasm_qubit(qasm_reader_t *rd, int64_t *qubit)
{
    const qasm_lex_t *lx = &rd->lx;
    const int line = lx->tok_line;

    if( QASM_T_ID != lx->kind ) return qasm_unexpected(rd, "a qubit");
    if( '$' == lx->tok[0] ) {
        return qasm_fail(rd, line, "'%.*s': physical qubits are not read, the circuit has to name the qubits of its register",
                         (int)((lx->len < 40) ? lx->len : 40), lx->tok);
    }
    if( NULL == rd->qreg.name ) return qasm_fail(rd, line, "a qubit is used before the register of qubits is declared");
    if( !qasm_same_name(&rd->qreg, lx) ) {
        return qasm_fail(rd, line, "'%.*s' is not the register of qubits", (int)((lx->len < 40) ? lx->len : 40), lx->tok);
    }
    qasm_lex_next(&rd->lx);
    if( -2 == (*qubit = qasm_index(rd)) ) return -1;
    if( *qubit >= (int64_t)rd->nb_qubits ) {
        return qasm_fail(rd, line, "qubit %lld is used, and the register has %u", (long long)*qubit, rd->nb_qubits);
    }
    return 0;
}

static int qasm_bit(qasm_reader_t *rd, int64_t *bit)
{
    const qasm_lex_t *lx = &rd->lx;
    const int line = lx->tok_line;

    if( QASM_T_ID != lx->kind ) return qasm_unexpected(rd, "a bit");
    if( NULL == rd->creg.name ) return qasm_fail(rd, line, "a measurement comes before the register of bits is declared");
    if( !qasm_same_name(&rd->creg, lx) ) {
        return qasm_fail(rd, line, "'%.*s' is not the register of bits", (int)((lx->len < 40) ? lx->len : 40), lx->tok);
    }
    qasm_lex_next(&rd->lx);
    if( -2 == (*bit = qasm_index(rd)) ) return -1;
    if( *bit >= (int64_t)rd->nb_bits ) {
        return qasm_fail(rd, line, "bit %lld is used, and the register has %u", (long long)*bit, rd->nb_bits);
    }
    return 0;
}

static int qasm_add(qasm_reader_t *rd, int line, int opcode, uint32_t q0, uint32_t q1, int param, double angle)
{
    const uint64_t on = (((uint64_t)1) << q0) | ((PARSEC_QPU_OP_CX <= opcode && opcode <= PARSEC_QPU_OP_RZZ) ? (((uint64_t)1) << q1) : 0);

    if( rd->measured & on ) {
        return qasm_fail(rd, line, "an operation follows the measurement of its qubit");
    }
    if( PARSEC_QPU_OP_MEASURE == opcode ) rd->measured |= on;
    if( UINT32_MAX == rd->nb_ops ) return qasm_fail(rd, line, "too many operations");
    if( (NULL != rd->ops) && (rd->nb_ops < rd->ops_room) ) {
        parsec_qpu_op_t *op = &rd->ops[rd->nb_ops];
        op->opcode   = (uint8_t)opcode;
        op->reserved = 0;
        op->q0       = (uint16_t)q0;
        op->q1       = (uint16_t)q1;
        op->param    = (int16_t)param;
        op->angle    = angle;
    }
    rd->nb_ops++;
    return 0;
}

/* A measurement, after its bit when the bit came first: "measure" <qubit>
 * [ "->" <bit> ]. bit is -3 when no bit was read yet. */
static int qasm_measure(qasm_reader_t *rd, int64_t bit)
{
    const int line = rd->lx.tok_line;
    int64_t qubit;

    qasm_lex_next(&rd->lx);                         /* "measure" */
    if( 0 != qasm_qubit(rd, &qubit) ) return -1;
    if( -3 == bit ) {
        if( QASM_T_ARROW != rd->lx.kind ) {
            return qasm_fail(rd, line, "a measurement whose outcome goes to no bit");
        }
        qasm_lex_next(&rd->lx);
        if( 0 != qasm_bit(rd, &bit) ) return -1;
    }
    if( (qubit < 0) != (bit < 0) ) {
        return qasm_fail(rd, line, "a measurement between a register and one element of the other");
    }
    if( qubit < 0 ) {
        if( rd->nb_bits != rd->nb_qubits ) {
            return qasm_fail(rd, line, "the register of qubits is measured into a register of another size");
        }
        for( uint32_t q = 0; q < rd->nb_qubits; q++ ) {
            if( 0 != qasm_add(rd, line, PARSEC_QPU_OP_MEASURE, q, 0, PARSEC_QPU_NO_PARAM, 0.0) ) return -1;
        }
    } else {
        if( qubit != bit ) {
            return qasm_fail(rd, line, "qubit %lld is measured into bit %lld: the circuit format measures qubit i into bit i",
                             (long long)qubit, (long long)bit);
        }
        if( 0 != qasm_add(rd, line, PARSEC_QPU_OP_MEASURE, (uint32_t)qubit, 0, PARSEC_QPU_NO_PARAM, 0.0) ) return -1;
    }
    return qasm_expect_sym(rd, ';', "';'");
}

/* The declaration of a register: "qubit[n] name;" or "qreg name[n];". */
static int qasm_register(qasm_reader_t *rd, int old_style, int is_qubits)
{
    qasm_name_t *reg  = is_qubits ? &rd->qreg : &rd->creg;
    uint32_t    *size = is_qubits ? &rd->nb_qubits : &rd->nb_bits;
    const char  *what = is_qubits ? "qubits" : "bits";
    const int line = rd->lx.tok_line;
    int64_t n = 1;

    if( NULL != reg->name ) {
        return qasm_fail(rd, line, "a second register of %s: the circuit format has one", what);
    }
    qasm_lex_next(&rd->lx);
    if( !old_style && (-2 == (n = qasm_index(rd))) ) return -1;
    if( QASM_T_ID != rd->lx.kind ) return qasm_unexpected(rd, "the name of the register");
    if( qasm_same_name(&rd->qreg, &rd->lx) || qasm_same_name(&rd->creg, &rd->lx) || (qasm_find_param(rd, &rd->lx) >= 0) ) {
        return qasm_fail(rd, line, "'%.*s' is declared twice", (int)((rd->lx.len < 40) ? rd->lx.len : 40), rd->lx.tok);
    }
    reg->name = rd->lx.tok;
    reg->len  = rd->lx.len;
    qasm_lex_next(&rd->lx);
    if( old_style && (-2 == (n = qasm_index(rd))) ) return -1;
    if( -1 == n ) n = 1;                            /* "qubit q;" is one qubit */
    if( (n < 1) || (n > PARSEC_QPU_MAX_QUBITS) ) {
        reg->name = NULL;
        return qasm_fail(rd, line, "a register of %lld %s: the circuit format has between 1 and %d",
                         (long long)n, what, PARSEC_QPU_MAX_QUBITS);
    }
    *size = (uint32_t)n;
    return qasm_expect_sym(rd, ';', "';'");
}

static int qasm_input(qasm_reader_t *rd)
{
    const int line = rd->lx.tok_line;

    qasm_lex_next(&rd->lx);
    if( !QASM_IS_ID(&rd->lx, "angle") && !QASM_IS_ID(&rd->lx, "float") ) {
        return qasm_unexpected(rd, "an input of type angle or float");
    }
    qasm_lex_next(&rd->lx);
    if( -2 == qasm_index(rd) ) return -1;           /* the width of the type, which changes nothing here */
    if( QASM_T_ID != rd->lx.kind ) return qasm_unexpected(rd, "the name of the input");
    if( qasm_same_name(&rd->qreg, &rd->lx) || qasm_same_name(&rd->creg, &rd->lx) || (qasm_find_param(rd, &rd->lx) >= 0) ||
        QASM_IS_ID(&rd->lx, "pi") || QASM_IS_ID(&rd->lx, "tau") || QASM_IS_ID(&rd->lx, "euler") ) {
        return qasm_fail(rd, line, "'%.*s' is declared twice, or is the name of a constant",
                         (int)((rd->lx.len < 40) ? rd->lx.len : 40), rd->lx.tok);
    }
    if( rd->nb_params >= (uint32_t)INT16_MAX ) return qasm_fail(rd, line, "too many inputs");
    if( rd->nb_params == rd->params_room ) {
        const uint32_t room = (0 == rd->params_room) ? 16 : 2 * rd->params_room;
        qasm_name_t *params = (qasm_name_t*)realloc(rd->params, room * sizeof(qasm_name_t));
        if( NULL == params ) return qasm_fail(rd, line, "out of memory");
        rd->params      = params;
        rd->params_room = room;
    }
    rd->params[rd->nb_params].name = rd->lx.tok;
    rd->params[rd->nb_params].len  = rd->lx.len;
    rd->nb_params++;
    qasm_lex_next(&rd->lx);
    return qasm_expect_sym(rd, ';', "';'");
}

static int qasm_gate_call(qasm_reader_t *rd, const struct qasm_gate_s *gate)
{
    const int line = rd->lx.tok_line;
    int param = PARSEC_QPU_NO_PARAM;
    double angle = 0.0;
    int64_t q0, q1 = 0;

    qasm_lex_next(&rd->lx);
    if( QASM_ROTATION & gate->kind ) {
        if( 0 != qasm_angle(rd, &param, &angle) ) return -1;
    } else if( QASM_IS_SYM(&rd->lx, '(') ) {
        return qasm_fail(rd, line, "%s takes no angle", gate->name);
    }
    if( 0 != qasm_qubit(rd, &q0) ) return -1;
    if( QASM_TWO_QUBITS & gate->kind ) {
        if( 0 != qasm_expect_sym(rd, ',', "',' and the second qubit") ) return -1;
        if( 0 != qasm_qubit(rd, &q1) ) return -1;
    }
    if( (q0 < 0) || (q1 < 0) ) {
        return qasm_fail(rd, line, "%s on a whole register is not read: name each qubit", gate->name);
    }
    if( (QASM_TWO_QUBITS & gate->kind) && (q0 == q1) ) {
        return qasm_fail(rd, line, "%s is on qubit %lld twice", gate->name, (long long)q0);
    }
    if( 0 != qasm_add(rd, line, gate->opcode, (uint32_t)q0, (uint32_t)q1, param, angle) ) return -1;
    return qasm_expect_sym(rd, ';', "';'");
}

static int qasm_statement(qasm_reader_t *rd)
{
    qasm_lex_t *lx = &rd->lx;
    const int line = lx->tok_line;

    if( QASM_T_ID != lx->kind ) return qasm_unexpected(rd, "a statement");

    if( QASM_IS_ID(lx, "OPENQASM") ) {
        qasm_lex_next(lx);
        if( (QASM_T_NUM != lx->kind) || (lx->num < 3.0) || (lx->num >= 4.0) ) {
            return qasm_fail(rd, line, "this reader takes OpenQASM 3, and the text says another version");
        }
        qasm_lex_next(lx);
        return qasm_expect_sym(rd, ';', "';'");
    }
    if( QASM_IS_ID(lx, "include") ) {
        qasm_lex_next(lx);
        if( QASM_T_BAD == lx->kind ) return qasm_unexpected(rd, "the name of a file");
        if( !qasm_tok_is(lx, QASM_T_STR, "stdgates.inc") ) {
            return qasm_fail(rd, line, "the only file that may be included is \"stdgates.inc\"");
        }
        qasm_lex_next(lx);
        return qasm_expect_sym(rd, ';', "';'");
    }
    if( QASM_IS_ID(lx, "input") )  return qasm_input(rd);
    if( QASM_IS_ID(lx, "qubit") )  return qasm_register(rd, 0, 1);
    if( QASM_IS_ID(lx, "qreg") )   return qasm_register(rd, 1, 1);
    if( QASM_IS_ID(lx, "bit") )    return qasm_register(rd, 0, 0);
    if( QASM_IS_ID(lx, "creg") )   return qasm_register(rd, 1, 0);
    if( QASM_IS_ID(lx, "gate") ) {
        /* The definition of a gate is skipped: what rxx, ryy and rzz are is
         * fixed by their names, and no other defined gate can be used. */
        int depth = 0;
        while( !QASM_IS_SYM(lx, '{') ) {
            if( (QASM_T_END == lx->kind) || (QASM_T_BAD == lx->kind) || QASM_IS_SYM(lx, ';') ) {
                return qasm_fail(rd, line, "the definition of a gate has no body");
            }
            qasm_lex_next(lx);
        }
        do {
            if( QASM_IS_SYM(lx, '{') ) depth++;
            else if( QASM_IS_SYM(lx, '}') ) depth--;
            else if( (QASM_T_END == lx->kind) || (QASM_T_BAD == lx->kind) ) {
                return qasm_fail(rd, line, "the definition of a gate is not closed");
            }
            qasm_lex_next(lx);
        } while( depth > 0 );
        return 0;
    }
    if( QASM_IS_ID(lx, "barrier") ) {
        while( !QASM_IS_SYM(lx, ';') ) {
            if( (QASM_T_END == lx->kind) || (QASM_T_BAD == lx->kind) ) return qasm_unexpected(rd, "';'");
            qasm_lex_next(lx);
        }
        qasm_lex_next(lx);
        return 0;
    }
    if( QASM_IS_ID(lx, "measure") ) return qasm_measure(rd, -3);
    if( qasm_same_name(&rd->creg, lx) ) {
        int64_t bit;
        if( 0 != qasm_bit(rd, &bit) ) return -1;
        if( 0 != qasm_expect_sym(rd, '=', "'=' and a measurement") ) return -1;
        if( !QASM_IS_ID(lx, "measure") ) return qasm_unexpected(rd, "a measurement");
        return qasm_measure(rd, bit);
    }
    for( int i = 0; NULL != qasm_gates[i].name; i++ ) {
        if( QASM_IS_ID(lx, qasm_gates[i].name) ) return qasm_gate_call(rd, &qasm_gates[i]);
    }
    return qasm_fail(rd, line, "'%.*s' is not in the part of OpenQASM 3 that this reader takes",
                     (int)((lx->len < 40) ? lx->len : 40), lx->tok);
}

int parsec_qpu_qasm_read(const char *text, size_t length, parsec_qpu_circuit_t *circuit, size_t room,
                         parsec_qpu_qasm_info_t *info)
{
    qasm_reader_t rd;
    size_t size;
    int rc = PARSEC_SUCCESS;

    if( NULL != info ) memset(info, 0, sizeof(*info));
    if( NULL == text ) {
        if( NULL != info ) { info->line = 1; snprintf(info->why, sizeof(info->why), "there is no text"); }
        return PARSEC_ERR_BAD_PARAM;
    }
    memset(&rd, 0, sizeof(rd));
    qasm_lex_init(&rd.lx, text, length);
    if( (NULL != circuit) && (room >= sizeof(parsec_qpu_circuit_t)) ) {
        const size_t fit = (room - sizeof(parsec_qpu_circuit_t)) / PARSEC_QPU_OP_SIZE;
        rd.ops      = (parsec_qpu_op_t*)(circuit + 1);
        rd.ops_room = (fit > UINT32_MAX) ? UINT32_MAX : (uint32_t)fit;
    }

    while( (QASM_T_END != rd.lx.kind) && (0 == rd.line) ) {
        (void)qasm_statement(&rd);
    }
    if( (0 == rd.line) && (NULL == rd.qreg.name) ) {
        (void)qasm_fail(&rd, rd.lx.line, "the text declares no register of qubits");
    }
    if( (0 == rd.line) && (0 == rd.measured) ) {
        (void)qasm_fail(&rd, rd.lx.line, "the circuit measures nothing");
    }
    free(rd.params);

    if( 0 != rd.line ) {
        if( NULL != info ) {
            info->line = rd.line;
            memcpy(info->why, rd.why, sizeof(info->why));
        }
        return PARSEC_ERR_BAD_PARAM;
    }

    size = sizeof(parsec_qpu_circuit_t) + (size_t)rd.nb_ops * PARSEC_QPU_OP_SIZE;
    if( NULL != info ) {
        info->size      = size;
        info->nb_qubits = rd.nb_qubits;
        info->nb_ops    = rd.nb_ops;
        info->nb_params = rd.nb_params;
    }
    if( NULL == circuit ) return PARSEC_SUCCESS;
    if( size > room ) return PARSEC_ERR_OUT_OF_RESOURCE;

    parsec_qpu_circuit_init(circuit, rd.nb_qubits, rd.nb_params);
    circuit->nb_ops = rd.nb_ops;
    {
        /* Everything the check looks at was looked at above, with its line. */
        const char *why = NULL;
        if( PARSEC_SUCCESS != (rc = parsec_qpu_circuit_check(circuit, size, &why)) ) {
            if( NULL != info ) {
                info->size = 0;
                info->line = rd.lx.line;
                snprintf(info->why, sizeof(info->why), "%s", (NULL != why) ? why : "the circuit is not well-formed");
            }
        }
    }
    return rc;
}

int parsec_qpu_qasm_param_slot(const char *text, size_t length, const char *name)
{
    qasm_lex_t lx;
    int slot = 0, start = 1;

    if( (NULL == text) || (NULL == name) ) return PARSEC_ERR_NOT_FOUND;
    qasm_lex_init(&lx, text, length);
    while( (QASM_T_END != lx.kind) && (QASM_T_BAD != lx.kind) ) {
        if( start && QASM_IS_ID(&lx, "input") ) {
            /* The name is the last word of the declaration. */
            qasm_lex_t last = lx;
            last.kind = QASM_T_END;
            qasm_lex_next(&lx);
            while( (QASM_T_END != lx.kind) && (QASM_T_BAD != lx.kind) && !QASM_IS_SYM(&lx, ';') ) {
                if( QASM_T_ID == lx.kind ) last = lx;
                qasm_lex_next(&lx);
            }
            if( qasm_tok_is(&last, QASM_T_ID, name) ) return slot;
            slot++;
            continue;
        }
        start = QASM_IS_SYM(&lx, ';') || QASM_IS_SYM(&lx, '}') || QASM_IS_SYM(&lx, '{');
        qasm_lex_next(&lx);
    }
    return PARSEC_ERR_NOT_FOUND;
}
