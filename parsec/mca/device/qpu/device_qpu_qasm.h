/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_QPU_QASM_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_QPU_QASM_H_HAS_BEEN_INCLUDED

/**
 * @file
 * Circuits as OpenQASM 3 text, in both directions.
 *
 * The writer gives the text a provider or its SDK takes. The reader takes
 * the text of a circuit that was built elsewhere, in Qiskit for instance,
 * and gives the circuit a quantum task submits. Both hold the part of
 * OpenQASM 3 that a parsec_qpu_circuit_t can say, and nothing more:
 *
 *  - one register of qubits and one of bits, of the same size;
 *  - the gates h, x, y, z, rx, ry, rz, cx, cz, rxx, ryy and rzz, with the
 *    meaning the circuit format gives them (device_qpu.h);
 *  - the measurement of qubit i into bit i;
 *  - parameters, declared as inputs. The k-th "input" declaration of the
 *    text is parameter slot k. The angle of a rotation is a number, or a
 *    number times one parameter.
 *
 * What the writer writes, the reader reads back as the same bytes, but for
 * the fields an operation does not use (the second qubit of a one-qubit
 * operation, the parameter slot and the angle of an operation that is not a
 * rotation): the text does not hold them, and they come back as 0,
 * PARSEC_QPU_NO_PARAM and 0.
 */

#include "parsec/mca/device/qpu/device_qpu.h"

BEGIN_C_DECLS

/**
 * Write a circuit as OpenQASM 3.
 *
 * The parameter slots are the inputs p0, p1, ... of the text, of type
 * float[64], which is how Qiskit declares the parameters of a circuit it
 * writes. rxx, ryy and rzz are not gates of the standard library of
 * OpenQASM 3, so the text defines the ones it uses, from gates of that
 * library.
 *
 * @param[in]  circuit a well-formed circuit that is not an empty job
 * @param[out] text    receives the text and a final 0, as far as it fits;
 *                     may be NULL when room is 0
 * @param[in]  room    bytes at text
 * @return the length of the whole text, without the final 0: it fitted when
 *         this is less than room. PARSEC_ERR_BAD_PARAM for a circuit that
 *         is not well-formed or is an empty job.
 */
PARSEC_DECLSPEC int64_t parsec_qpu_qasm_write(const parsec_qpu_circuit_t *circuit, char *text, size_t room);

/** What the reader found. */
typedef struct parsec_qpu_qasm_info_s {
    size_t   size;        /**< bytes the circuit takes; 0 when the text could not be read */
    uint32_t nb_qubits;
    uint32_t nb_ops;
    uint32_t nb_params;   /**< "input" declarations */
    int      line;        /**< where reading stopped, counted from 1; 0 when nothing is wrong */
    char     why[160];    /**< what is wrong there; empty when nothing is */
} parsec_qpu_qasm_info_t;

/**
 * Read a circuit from OpenQASM 3.
 *
 * Read as well, beyond what the writer writes: inputs of type float; the
 * definition of a gate, which is skipped (rxx, ryy and rzz are taken for
 * what their names say, whatever the text defines them as; any other
 * defined gate cannot be used); barriers, which are ignored; qreg and creg;
 * "measure q -> c" and the measurement of the whole register; angles
 * written as expressions of numbers, pi and one parameter, as long as they
 * come to a number or to a number times that parameter.
 *
 * Anything else is refused: the text is not read as far as it goes, and no
 * part of a circuit is given.
 *
 * @param[in]  text    the text; it need not end with a 0
 * @param[in]  length  its length in bytes
 * @param[out] circuit receives the circuit; NULL to learn its size only
 * @param[in]  room    bytes at circuit
 * @param[out] info    what was found, or where and why reading stopped;
 *                     may be NULL
 * @return PARSEC_SUCCESS; PARSEC_ERR_BAD_PARAM for a text that is not a
 *         circuit of this subset (info says where and why);
 *         PARSEC_ERR_OUT_OF_RESOURCE when the circuit does not fit in room
 *         (info->size says what it takes).
 */
PARSEC_DECLSPEC int parsec_qpu_qasm_read(const char *text, size_t length,
                                         parsec_qpu_circuit_t *circuit, size_t room,
                                         parsec_qpu_qasm_info_t *info);

/**
 * The parameter slot of an input of the text, by its name.
 *
 * @return the slot, from 0; PARSEC_ERR_NOT_FOUND when the text declares no
 *         input of that name.
 */
PARSEC_DECLSPEC int parsec_qpu_qasm_param_slot(const char *text, size_t length, const char *name);

END_C_DECLS

#endif  /* PARSEC_DEVICE_QPU_QASM_H_HAS_BEEN_INCLUDED */
