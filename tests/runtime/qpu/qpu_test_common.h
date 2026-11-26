/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef QPU_TEST_COMMON_H
#define QPU_TEST_COMMON_H

/* What the QPU runtime tests share: where tasks run, what the data of a
 * quantum task looks like, and the checks every test ends with. */

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/arena.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"
#include "tests/tests_runtime.h"

/* Every flow of these tests is a block of this many bytes: enough for a
 * small circuit, for a few parameters, or for a result with every outcome
 * of that circuit. */
#define QPU_TEST_BLOCK        512
#define QPU_TEST_NB_PARAMS    4
#define QPU_TEST_NB_QUBITS    4
#define QPU_TEST_MAX_ENTRIES  ((QPU_TEST_BLOCK - sizeof(parsec_qpu_result_t)) / PARSEC_QPU_ENTRY_SIZE)

/* A collection that holds no data and only says where a task runs:
 * element k is on rank fixed_rank, or on rank k modulo the number of ranks
 * when fixed_rank is negative. */
parsec_data_collection_t *qpu_test_placement_new(int myrank, int nodes, int fixed_rank);
/* The same with element k on the (k modulo nb_ranks)-th rank of a list. */
parsec_data_collection_t *qpu_test_placement_list_new(int myrank, int nodes, const int *ranks, int nb_ranks);
void qpu_test_placement_free(parsec_data_collection_t *dc);

/* A collection of n blocks of QPU_TEST_BLOCK bytes that all live on one
 * rank. Used where a result has to be read after the taskpool is gone. */
parsec_data_collection_t *qpu_test_tiles_new(int myrank, int nodes, int owner, int n);
void *qpu_test_tiles_ptr(parsec_data_collection_t *dc, int k);   /* on the owner only */
/* Make the tiles copies that the application manages, not PaRSEC, as those
 * of a collection of the user's own are. To be called before the first use. */
void qpu_test_tiles_user_managed(parsec_data_collection_t *dc);
void qpu_test_tiles_free(parsec_data_collection_t *dc);

/* The arena of a QPU_TEST_BLOCK-byte flow. */
void qpu_test_block_type_set(parsec_arena_datatype_t *adt);
void qpu_test_block_type_unset(parsec_arena_datatype_t *adt);

/* The inputs of the quantum task (a, b): a circuit of QPU_TEST_NB_QUBITS
 * qubits and its parameter values. Different (a, b) give different circuits,
 * different values and different outcome probabilities, so a result can be
 * matched to the inputs it was computed from. */
void qpu_test_make_circuit(void *circuit, int a, int b, int empty);
void qpu_test_make_params(void *params, int a, int b);
/* The first parameter is a; tests that steer the mock read it back. */
int  qpu_test_params_a(const double *params);
/* The key the tests give the evaluation of (a, b), and (a, b) back from it. */
static inline uint64_t qpu_test_key(int a, int b) { return (((uint64_t)(uint32_t)a) << 32) | (uint32_t)b; }
static inline int qpu_test_key_a(uint64_t key) { return (int)(key >> 32); }
static inline int qpu_test_key_b(uint64_t key) { return (int)(key & 0xffffffffu); }

/* Check the entries of a result: they are in order, inside the register, and
 * their counts add up to shots_returned; and, when the key is known (not
 * PARSEC_QPU_KEY_AUTO), they are the ones the sampler gives for these inputs,
 * this key and the seed of the mock. Prints what is wrong and returns the
 * number of problems. */
int qpu_test_check_counts(const void *result, const void *circuit, const void *params,
                          int shots, uint64_t key, const char *who, int a, int b);

/* The most tasks one job may carry in this run: device_qpu_max_batch, and
 * no more than the mock takes circuits (device_qpu_mock_max_circuits). The
 * same on every rank. */
int qpu_test_max_batch(void);

/* Check a whole result that had nothing go wrong: one job, of no more tasks
 * than a job may carry, status OK, all the shots, and qpu_test_check_counts(). */
int qpu_test_check_result(const void *result, const void *circuit, const void *params,
                          int shots, uint64_t key, const char *who, int a, int b);

/* Check the QPU service of this process once everything is done. A rank
 * that ran quantum tasks must have completed exactly `tasks` of them, of
 * which `empty` needed no job, and hold nothing; any other rank must not
 * have started a service or opened a backend. Nothing may have gone wrong:
 * no failure, no retry, no stray event. Returns the number of problems. */
int qpu_test_check_service(int ran_quantum_tasks, long tasks, long empty, int verbose);

/* The part of that check that holds whatever was injected: every task that
 * was submitted is completed, nothing is pending or in flight, the limits
 * were respected, and the mock saw no tag twice, no two jobs of one
 * evaluation at once, and no input change under a job. It ends with
 * qpu_test_check_device(1, tasks + what qpu_test_expect_also_executed() was told). */
int qpu_test_check_invariants(long tasks, int verbose);

/* Check the QPU device of this process once everything is done: it is
 * there, with the type of a QPU and not taken for a GPU; it owns the QPU or
 * not, as said; it holds no load; and it executed exactly `executed` tasks.
 * On the owner its backend is described; elsewhere it has none. */
int qpu_test_check_device(int owner, long executed);

/* Tasks the device executed that the service did not complete since it last
 * started: those of bodies of type QPU that ended without giving their task
 * away (a submission that was refused; the scheduler counts them, as it does
 * for any body that returns DONE), and those of a service that was stopped
 * since (the device goes on counting). Zero by default. */
void qpu_test_expect_also_executed(long tasks);

/* Sum a value over all ranks. */
long qpu_test_sum(parsec_context_t *parsec, long local);

/* Split argv at "--": test arguments before, PaRSEC arguments after. */
void qpu_test_split_args(int argc, char **argv, int *test_argc, int *parsec_argc, char ***parsec_argv);

#endif  /* QPU_TEST_COMMON_H */
