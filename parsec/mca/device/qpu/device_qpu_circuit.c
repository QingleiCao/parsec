/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu.h"

#include <math.h>

/**
 * What can be said about a circuit without running it.
 */

#define QPU_OP_ONE_QUBIT   0x1   /* acts on q0 */
#define QPU_OP_TWO_QUBITS  0x2   /* acts on q0 and q1 */
#define QPU_OP_ROTATION    0x4   /* takes an angle */

static const unsigned char qpu_op_kind[PARSEC_QPU_OP_MAX] = {
    [PARSEC_QPU_OP_H]       = QPU_OP_ONE_QUBIT,
    [PARSEC_QPU_OP_X]       = QPU_OP_ONE_QUBIT,
    [PARSEC_QPU_OP_Y]       = QPU_OP_ONE_QUBIT,
    [PARSEC_QPU_OP_Z]       = QPU_OP_ONE_QUBIT,
    [PARSEC_QPU_OP_RX]      = QPU_OP_ONE_QUBIT  | QPU_OP_ROTATION,
    [PARSEC_QPU_OP_RY]      = QPU_OP_ONE_QUBIT  | QPU_OP_ROTATION,
    [PARSEC_QPU_OP_RZ]      = QPU_OP_ONE_QUBIT  | QPU_OP_ROTATION,
    [PARSEC_QPU_OP_CX]      = QPU_OP_TWO_QUBITS,
    [PARSEC_QPU_OP_CZ]      = QPU_OP_TWO_QUBITS,
    [PARSEC_QPU_OP_RXX]     = QPU_OP_TWO_QUBITS | QPU_OP_ROTATION,
    [PARSEC_QPU_OP_RYY]     = QPU_OP_TWO_QUBITS | QPU_OP_ROTATION,
    [PARSEC_QPU_OP_RZZ]     = QPU_OP_TWO_QUBITS | QPU_OP_ROTATION,
    [PARSEC_QPU_OP_MEASURE] = QPU_OP_ONE_QUBIT
};

#define QPU_CIRCUIT_BAD(WHY)                    \
    do {                                        \
        if( NULL != why ) *why = (WHY);         \
        return PARSEC_ERR_BAD_PARAM;            \
    } while(0)

int parsec_qpu_circuit_check(const parsec_qpu_circuit_t *circuit, size_t size, const char **why)
{
    const parsec_qpu_op_t *ops;
    uint64_t measured = 0;

    if( NULL != why ) *why = NULL;
    if( NULL == circuit ) QPU_CIRCUIT_BAD("there is no circuit");
    if( (0 != size) && (size < sizeof(parsec_qpu_circuit_t)) ) QPU_CIRCUIT_BAD("the buffer is smaller than a circuit header");
    if( PARSEC_QPU_CIRCUIT_VERSION != circuit->version ) QPU_CIRCUIT_BAD("this is not a circuit, or not of a known version");
    if( PARSEC_QPU_CIRCUIT_EMPTY & circuit->flags ) return PARSEC_SUCCESS;

    if( (0 == circuit->nb_qubits) || (circuit->nb_qubits > PARSEC_QPU_MAX_QUBITS) ) QPU_CIRCUIT_BAD("the number of qubits is not between 1 and 64");
    if( circuit->nb_params > (uint32_t)INT16_MAX ) QPU_CIRCUIT_BAD("too many parameter slots");
    if( (0 != size) && (((size - sizeof(parsec_qpu_circuit_t)) / PARSEC_QPU_OP_SIZE) < circuit->nb_ops) ) {
        QPU_CIRCUIT_BAD("the operations do not fit in the buffer");
    }

    ops = parsec_qpu_circuit_ops(circuit);
    for( uint32_t i = 0; i < circuit->nb_ops; i++ ) {
        const parsec_qpu_op_t *op = &ops[i];
        unsigned char kind;

        if( (0 == op->opcode) || (op->opcode >= PARSEC_QPU_OP_MAX) ) QPU_CIRCUIT_BAD("an operation is not a known one");
        kind = qpu_op_kind[op->opcode];
        if( op->q0 >= circuit->nb_qubits ) QPU_CIRCUIT_BAD("an operation is on a qubit the circuit does not have");
        if( measured & (((uint64_t)1) << op->q0) ) QPU_CIRCUIT_BAD("an operation follows the measurement of its qubit");
        if( QPU_OP_TWO_QUBITS & kind ) {
            if( op->q1 >= circuit->nb_qubits ) QPU_CIRCUIT_BAD("an operation is on a qubit the circuit does not have");
            if( op->q1 == op->q0 ) QPU_CIRCUIT_BAD("a two-qubit operation is on one qubit twice");
            if( measured & (((uint64_t)1) << op->q1) ) QPU_CIRCUIT_BAD("an operation follows the measurement of its qubit");
        }
        if( QPU_OP_ROTATION & kind ) {
            if( (PARSEC_QPU_NO_PARAM != op->param) &&
                ((op->param < 0) || ((uint32_t)op->param >= circuit->nb_params)) ) {
                QPU_CIRCUIT_BAD("a rotation uses a parameter slot the circuit does not have");
            }
            if( !isfinite(op->angle) ) QPU_CIRCUIT_BAD("a rotation has an angle that is not a finite number");
        }
        if( PARSEC_QPU_OP_MEASURE == op->opcode ) measured |= ((uint64_t)1) << op->q0;
    }
    if( 0 == measured ) QPU_CIRCUIT_BAD("the circuit measures nothing");
    return PARSEC_SUCCESS;
}

uint64_t parsec_qpu_circuit_measured(const parsec_qpu_circuit_t *circuit)
{
    const parsec_qpu_op_t *ops = parsec_qpu_circuit_ops(circuit);
    uint64_t measured = 0;

    if( PARSEC_QPU_CIRCUIT_EMPTY & circuit->flags ) return 0;
    for( uint32_t i = 0; i < circuit->nb_ops; i++ ) {
        if( (PARSEC_QPU_OP_MEASURE == ops[i].opcode) && (ops[i].q0 < PARSEC_QPU_MAX_QUBITS) ) {
            measured |= ((uint64_t)1) << ops[i].q0;
        }
    }
    return measured;
}
