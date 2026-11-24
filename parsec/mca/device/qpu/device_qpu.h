/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_QPU_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_QPU_H_HAS_BEEN_INCLUDED

/**
 * @file
 * Quantum processing units in PaRSEC: what a task body needs.
 *
 * A quantum task reads a circuit and its parameter values from ordinary
 * flows and writes its measurement result to an ordinary flow, all of them
 * host data. Its body calls parsec_qpu_submit() and returns what that
 * returns. The body does no backend work and does not wait: the task is
 * completed later, when the job has finished, and only then are its
 * successors released.
 *
 * The QPU is a device type, PARSEC_DEV_QPU. A body that submits is declared
 * with that type, "BODY [type=QPU]" in a JDF or PARSEC_DEV_QPU at a DTD
 * insertion, and the runtime then selects the QPU for it as it selects a GPU
 * for a body of a GPU type: in a context that has no QPU device, another
 * incarnation of the task class runs if there is one. A body of another type
 * may submit as well; the job then goes to the QPU of the process all the
 * same, but the runtime does not know the task as a quantum task.
 *
 * One rank owns the QPU (device_qpu_owner_rank) and runs every job. Quantum
 * tasks have to be placed there by their affinity; their inputs and results
 * travel between ranks as any flow does. The owner is the rank the process
 * has when the context is initialized. A quantum task that runs on another
 * rank is refused, also when its class has another incarnation.
 *
 * What a task class with a QPU incarnation has to know:
 *  - A QPU has no flop rate. A time_estimate function of the class is called
 *    with the QPU device as with any other, and finds its gflops_* at zero:
 *    it has to tell the device by its type and return the time it expects a
 *    job to take, in nanoseconds.
 *  - Between incarnations of different types the runtime compares estimated
 *    times. Without a time_estimate function these are defaults that have no
 *    common unit: a quantum task counts for a second, a CPU or GPU task for
 *    a ratio of flop rates. In effect the QPU body is preferred to a CPU
 *    body (unless device_load_balance_allow_cpu is set), and a GPU body is
 *    preferred to the QPU body.
 *  - The body submits or ends: it should not return PARSEC_HOOK_RETURN_AGAIN
 *    (the load the scheduler keeps for the device is then counted once more,
 *    as it is for a GPU body), and the class cannot be an immediate one
 *    (those are completed as soon as their body returns).
 *  - In a JDF, the runtime is told that the task wrote its output flows when
 *    it submits. A QPU body that writes a flow and ends without submitting
 *    leaves that unsaid, which matters when an accelerator holds another
 *    copy of the same data.
 */

#include "parsec/parsec_config.h"
#include "parsec/runtime.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

BEGIN_C_DECLS

/* ------------------------------------------------------------------------
 * Circuit: a flat buffer with no pointers, so it can travel between ranks as
 * bytes. This header is followed by nb_ops operations of PARSEC_QPU_OP_SIZE
 * bytes each.
 *
 * Conventions. Qubit i is bit i of a measured bitstring, bit 0 being the
 * least significant, and every qubit starts in |0>. A rotation about the
 * Pauli operator P by the angle t is exp(-i t P / 2).
 * --------------------------------------------------------------------- */

#define PARSEC_QPU_CIRCUIT_VERSION  1
#define PARSEC_QPU_OP_SIZE          16
#define PARSEC_QPU_MAX_QUBITS       64      /* a measured bitstring is one 64-bit word */

/** The job is empty: it completes at once and never reaches a backend. */
#define PARSEC_QPU_CIRCUIT_EMPTY    0x1u

typedef struct parsec_qpu_circuit_s {
    uint32_t version;     /**< PARSEC_QPU_CIRCUIT_VERSION */
    uint32_t nb_qubits;
    uint32_t nb_ops;      /**< operations that follow this header */
    uint32_t nb_params;   /**< parameter slots, bound from the parameter flow */
    uint32_t flags;       /**< PARSEC_QPU_CIRCUIT_* */
    uint32_t reserved[3];
} parsec_qpu_circuit_t;

typedef enum parsec_qpu_opcode_e {
    PARSEC_QPU_OP_H       =  1,   /**< Hadamard on q0 */
    PARSEC_QPU_OP_X       =  2,   /**< Pauli gates on q0 */
    PARSEC_QPU_OP_Y       =  3,
    PARSEC_QPU_OP_Z       =  4,
    PARSEC_QPU_OP_RX      =  5,   /**< exp(-i angle X / 2) on q0 */
    PARSEC_QPU_OP_RY      =  6,
    PARSEC_QPU_OP_RZ      =  7,
    PARSEC_QPU_OP_CX      =  8,   /**< controlled X: control q0, target q1 */
    PARSEC_QPU_OP_CZ      =  9,   /**< controlled Z on q0 and q1 */
    PARSEC_QPU_OP_RXX     = 10,   /**< exp(-i angle XX / 2) on q0 and q1 */
    PARSEC_QPU_OP_RYY     = 11,
    PARSEC_QPU_OP_RZZ     = 12,
    PARSEC_QPU_OP_MEASURE = 13,   /**< measure q0 into bit q0; nothing may follow on q0 */
    PARSEC_QPU_OP_MAX
} parsec_qpu_opcode_t;

/** The angle of a rotation is given in the operation itself. */
#define PARSEC_QPU_NO_PARAM         (-1)

/**
 * One operation. A rotation takes its angle from `angle` when `param` is
 * PARSEC_QPU_NO_PARAM, and otherwise from a parameter slot, scaled:
 *     angle * params[param].
 * So one circuit serves every evaluation that differs only in the values of
 * the parameters. The other operations ignore both fields.
 */
typedef struct parsec_qpu_op_s {
    uint8_t  opcode;      /**< parsec_qpu_opcode_t */
    uint8_t  reserved;
    uint16_t q0;
    uint16_t q1;          /**< second qubit of a two-qubit operation */
    int16_t  param;       /**< parameter slot, or PARSEC_QPU_NO_PARAM */
    double   angle;       /**< the angle, or the coefficient of the parameter */
} parsec_qpu_op_t;

typedef char parsec_qpu_op_is_16_bytes[(PARSEC_QPU_OP_SIZE == sizeof(parsec_qpu_op_t)) ? 1 : -1];

static inline size_t parsec_qpu_circuit_size(const parsec_qpu_circuit_t *circuit)
{
    return sizeof(parsec_qpu_circuit_t) + (size_t)circuit->nb_ops * PARSEC_QPU_OP_SIZE;
}

/** The operations of a circuit. The buffer has to be aligned for a double,
 *  as the memory of every PaRSEC data copy is. */
static inline parsec_qpu_op_t *parsec_qpu_circuit_ops(const parsec_qpu_circuit_t *circuit)
{
    return (parsec_qpu_op_t*)(circuit + 1);
}

/** Start a circuit with no operation in a buffer. */
static inline void parsec_qpu_circuit_init(parsec_qpu_circuit_t *circuit, uint32_t nb_qubits, uint32_t nb_params)
{
    memset(circuit, 0, sizeof(parsec_qpu_circuit_t));
    circuit->version   = PARSEC_QPU_CIRCUIT_VERSION;
    circuit->nb_qubits = nb_qubits;
    circuit->nb_params = nb_params;
}

/**
 * Append one operation to a circuit that lives in a buffer of `room` bytes.
 * q1 is ignored by one-qubit operations; param and angle are as described
 * for parsec_qpu_op_t. Returns 0, or -1 when the buffer is full. What is
 * appended is not checked here: see parsec_qpu_circuit_check().
 */
static inline int parsec_qpu_circuit_add(parsec_qpu_circuit_t *circuit, size_t room,
                                         int opcode, int q0, int q1, int param, double angle)
{
    parsec_qpu_op_t *op;

    if( parsec_qpu_circuit_size(circuit) + PARSEC_QPU_OP_SIZE > room ) return -1;
    op = parsec_qpu_circuit_ops(circuit) + circuit->nb_ops++;
    op->opcode   = (uint8_t)opcode;
    op->reserved = 0;
    op->q0       = (uint16_t)q0;
    op->q1       = (uint16_t)q1;
    op->param    = (int16_t)param;
    op->angle    = angle;
    return 0;
}

/** Append the measurement of every qubit. Returns 0, or -1 when the buffer is full. */
static inline int parsec_qpu_circuit_measure_all(parsec_qpu_circuit_t *circuit, size_t room)
{
    for( uint32_t q = 0; q < circuit->nb_qubits; q++ ) {
        if( 0 != parsec_qpu_circuit_add(circuit, room, PARSEC_QPU_OP_MEASURE, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.0) ) return -1;
    }
    return 0;
}

/**
 * Check that a buffer of `size` bytes holds a well-formed circuit (size 0:
 * the size is not known and not checked). A well-formed circuit has a known
 * version, between 1 and PARSEC_QPU_MAX_QUBITS qubits, known operations on
 * qubits and parameter slots that exist, finite angles, two different qubits
 * for every two-qubit operation, at least one measurement, and nothing on a
 * qubit after its measurement. An empty job is well-formed whatever follows
 * its header.
 *
 * @param[out] why  when not NULL, receives a constant string saying what is
 *                  wrong, or NULL when nothing is
 * @return PARSEC_SUCCESS or PARSEC_ERR_BAD_PARAM
 */
PARSEC_DECLSPEC int parsec_qpu_circuit_check(const parsec_qpu_circuit_t *circuit, size_t size, const char **why);

/** The qubits a circuit measures, as a mask: bit i is set when qubit i is. */
PARSEC_DECLSPEC uint64_t parsec_qpu_circuit_measured(const parsec_qpu_circuit_t *circuit);

/* ------------------------------------------------------------------------
 * Result: this header, followed by nb_entries entries of
 * PARSEC_QPU_ENTRY_SIZE bytes, each a measured bitstring (64 bits, qubit i
 * in bit i, zero for a qubit that was not measured) and the number of shots
 * that gave it. Entries are in increasing order of their bitstring, and no
 * bitstring appears twice.
 * --------------------------------------------------------------------- */

#define PARSEC_QPU_RESULT_VERSION   1
#define PARSEC_QPU_ENTRY_SIZE       12

typedef enum parsec_qpu_status_e {
    PARSEC_QPU_STATUS_OK          = 0,
    PARSEC_QPU_STATUS_FAILED      = 1,
    PARSEC_QPU_STATUS_TIMED_OUT   = 2,
    PARSEC_QPU_STATUS_UNAVAILABLE = 3,
    PARSEC_QPU_STATUS_CANCELLED   = 4,
    PARSEC_QPU_STATUS_REJECTED    = 5
} parsec_qpu_status_t;

/* What a status other than OK means for the task that reads it:
 *   FAILED       every attempt the runtime made failed at the backend
 *   TIMED_OUT    the last attempt did not end in time and was cancelled
 *   UNAVAILABLE  the backend could not be reached for too long
 *   CANCELLED    the backend cancelled the job
 *   REJECTED     the backend cannot run this job at all (too many qubits or
 *                shots); it was never submitted
 * Such a result has no entry and shots_returned is 0. The task completes
 * normally all the same, and its successors run. */

/** The result came from the mock backend, not from a quantum device. */
#define PARSEC_QPU_RESULT_MOCK      0x1u
/** The buffer had room for fewer entries than the job produced: the most
 *  frequent bitstrings were kept, and shots_returned counts only those. */
#define PARSEC_QPU_RESULT_TRUNCATED 0x2u

typedef struct parsec_qpu_result_s {
    uint16_t version;          /**< PARSEC_QPU_RESULT_VERSION */
    uint8_t  status;           /**< parsec_qpu_status_t */
    uint8_t  flags;            /**< PARSEC_QPU_RESULT_* */
    uint16_t nb_qubits;
    uint16_t attempts;         /**< jobs it took at the backend, 1 when the first one worked */
    uint32_t shots_requested;
    uint32_t shots_returned;   /**< sum of the counts of the entries */
    uint32_t nb_entries;
    uint32_t batch;            /**< tasks whose circuits were in the job that gave this result;
                                *   for a task that got none, in the last job the backend took
                                *   for it. 1 when alone, and for a task that never had a job */
    /* Nanoseconds on the monotonic clock of the process that owns the QPU.
     * Only differences between them mean something on another rank. */
    int64_t  t_hook;           /**< the task's body submitted */
    int64_t  t_submit;         /**< the job was given to the backend */
    int64_t  t_start;          /**< the backend started it */
    int64_t  t_end;            /**< the backend finished it */
    int64_t  t_retrieve;       /**< the result was in this buffer */
} parsec_qpu_result_t;

/* The header is 64 bytes, and code on both sides of a rank boundary counts on it. */
typedef char parsec_qpu_result_header_is_64_bytes[(64 == sizeof(parsec_qpu_result_t)) ? 1 : -1];

/** Bytes a result with room for max_entries entries takes. A job cannot return
 *  more distinct bitstrings than it has shots, so max_entries = shots always fits. */
static inline size_t parsec_qpu_result_size(uint32_t max_entries)
{
    return sizeof(parsec_qpu_result_t) + (size_t)max_entries * PARSEC_QPU_ENTRY_SIZE;
}

/* Entries are 12 bytes, so the 64-bit word is not aligned: go through memcpy. */
static inline void parsec_qpu_result_get_entry(const parsec_qpu_result_t *result, uint32_t i,
                                               uint64_t *bits, uint32_t *count)
{
    const char *entry = (const char*)(result + 1) + (size_t)i * PARSEC_QPU_ENTRY_SIZE;
    memcpy(bits,  entry,                    sizeof(uint64_t));
    memcpy(count, entry + sizeof(uint64_t), sizeof(uint32_t));
}

static inline void parsec_qpu_result_set_entry(parsec_qpu_result_t *result, uint32_t i,
                                               uint64_t bits, uint32_t count)
{
    char *entry = (char*)(result + 1) + (size_t)i * PARSEC_QPU_ENTRY_SIZE;
    memcpy(entry,                    &bits,  sizeof(uint64_t));
    memcpy(entry + sizeof(uint64_t), &count, sizeof(uint32_t));
}

/* ------------------------------------------------------------------------
 * Submission
 * --------------------------------------------------------------------- */

/**
 * The key of an evaluation names it to the application: two jobs with the
 * same key, the same circuit and the same parameter values are the same
 * evaluation. A backend that simulates draws its samples from the key, so
 * that a result depends neither on the order in which jobs run nor on how
 * many attempts a job took, and so that a program that does not use the
 * runtime can reproduce it. Any value but this one is a key; this one asks
 * for a key derived from the identity of the task.
 */
#define PARSEC_QPU_KEY_AUTO         UINT64_MAX

/**
 * Submit the quantum job of a task and give the task up until it is done.
 *
 * To be called once, as the last thing a body does:
 *     return parsec_qpu_submit(es, this_task, circuit, params, result, shots, key);
 * After the call the task belongs to the QPU layer and may already be
 * completed, so the caller must not touch it again.
 *
 * The circuit and the parameters are read where they are, for as long as the
 * job lives; the runtime keeps them in place for a task that returned ASYNC.
 * The result is written before the task is completed.
 *
 * @param[in] es      the execution stream running the body
 * @param[in] task    the task
 * @param[in] circuit host copy holding a parsec_qpu_circuit_t and its operations
 * @param[in] params  host copy holding circuit->nb_params doubles; may be NULL
 *                    when the circuit has no parameter slot
 * @param[in] result  host copy to receive the result; it needs room for at
 *                    least parsec_qpu_result_size(1) bytes, and for
 *                    parsec_qpu_result_size(shots) to never lose an entry
 * @param[in] shots   number of measurements, at least 1
 * @param[in] key     the key of the evaluation, or PARSEC_QPU_KEY_AUTO
 *
 * @return PARSEC_HOOK_RETURN_ASYNC when the job was accepted, and
 *         PARSEC_HOOK_RETURN_ERROR on a programming error (QPU support not
 *         enabled, a rank that does not own the QPU, a missing argument, a
 *         circuit that is not well-formed).
 *         A job that fails later is not an error of this call: the task
 *         completes with a status other than PARSEC_QPU_STATUS_OK in its
 *         result.
 */
PARSEC_DECLSPEC parsec_hook_return_t
parsec_qpu_submit(parsec_execution_stream_t *es, parsec_task_t *task,
                  parsec_data_copy_t *circuit, parsec_data_copy_t *params,
                  parsec_data_copy_t *result, int shots, uint64_t key);

/**
 * The same from a JDF body, naming the three flows:
 *     BODY
 *         return PARSEC_QPU_SUBMIT(C, TH, RES, 1024);
 *     END
 */
#define PARSEC_QPU_SUBMIT_KEYED(CIRCUIT, PARAMS, RESULT, SHOTS, KEY)      \
    parsec_qpu_submit(es, (parsec_task_t*)this_task,                      \
                      _f_##CIRCUIT, _f_##PARAMS, _f_##RESULT, (SHOTS), (KEY))
#define PARSEC_QPU_SUBMIT(CIRCUIT, PARAMS, RESULT, SHOTS)                 \
    PARSEC_QPU_SUBMIT_KEYED(CIRCUIT, PARAMS, RESULT, SHOTS, PARSEC_QPU_KEY_AUTO)

/**
 * The same for a DTD task, naming the three data by their position among the
 * task's data arguments (0 for the first tile passed at insertion, and so
 * on). params_index is negative when there are no parameters.
 */
PARSEC_DECLSPEC parsec_hook_return_t
parsec_qpu_submit_dtd(parsec_execution_stream_t *es, parsec_task_t *task,
                      int circuit_index, int params_index, int result_index,
                      int shots, uint64_t key);

/* ------------------------------------------------------------------------
 * The QPU device
 * --------------------------------------------------------------------- */

/** What the QPU device of this process is and holds. */
typedef struct parsec_qpu_device_info_s {
    int         device_index;          /**< among the devices of the runtime, as parsec_mca_device_get() takes it */
    int         owner_rank;            /**< the rank that runs the jobs */
    int         is_owner;              /**< this process is that rank */
    const char *backend;               /**< name of the backend; valid until parsec_fini() */
    /* What the backend said when it was opened. Zero on a rank that does not own the QPU. */
    uint32_t    nb_qubits;             /**< widest register a job may use */
    uint32_t    max_shots;             /**< most shots a task may ask for, 0 for no limit */
    uint32_t    max_circuits_per_job;
    uint32_t    is_physical;           /**< a quantum device, as opposed to a simulation */
    uint32_t    max_outstanding;       /**< most jobs kept at the backend at once; 0 on a rank that runs none */
    uint32_t    max_batch;             /**< most tasks whose circuits share one job; 1 when every
                                        *   task is its own job, 0 on a rank that runs none */
    /* What it is doing. */
    uint64_t    executed_tasks;        /**< tasks it completed */
    int64_t     load_ns;               /**< what the scheduler keeps as the load of the device:
                                        *   task_time_ns for every task it holds */
    int64_t     task_time_ns;          /**< what one task counts for in the load: a nominal second */
    int64_t     latency_estimate_ns;   /**< how long a job takes from its submission to its
                                        *   result: a moving average over the jobs that ended
                                        *   well, of one task or of several, 0 before any has.
                                        *   The jobs that the tasks held will make, times this,
                                        *   over max_outstanding, estimates the wait; with every
                                        *   task its own job that is the tasks held. */
} parsec_qpu_device_info_t;

/**
 * Describe the QPU device of this process. qpu is 0: there is one QPU per
 * process so far.
 *
 * @return PARSEC_SUCCESS, PARSEC_ERR_NOT_FOUND when this process has no such
 *         device (QPU support is not enabled in this context, or the backend
 *         could not be used), PARSEC_ERR_BAD_PARAM for a NULL info.
 */
PARSEC_DECLSPEC int parsec_qpu_device_info(int qpu, parsec_qpu_device_info_t *info);

END_C_DECLS

#endif  /* PARSEC_DEVICE_QPU_H_HAS_BEEN_INCLUDED */
