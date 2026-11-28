/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef QAOA_DATA_H
#define QAOA_DATA_H

/**
 * @file
 * The data of the distributed QAOA application: where its tasks run, the
 * few tiles that outlive the task graph, and the sizes of what flows between
 * the tasks.
 */

#include "qaoa_core.h"
#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/arena.h"

BEGIN_C_DECLS

/**
 * A collection of n tiles of tile_bytes bytes each.
 *
 * Tile k lives on rank `owner`, or on rank k modulo the number of ranks when
 * owner is negative. A rank allocates the tiles it holds, zeroed, and
 * nothing for the others. With tile_bytes 0 the collection holds no data
 * and only says where a task runs.
 */
parsec_data_collection_t *qaoa_tiles_new(int myrank, int nodes, int owner, int n, size_t tile_bytes);
/** Put tile k on the (k modulo nb_ranks)-th rank of a list instead. Only for
 *  tiles that hold nothing (tile_bytes 0), and before the first use. */
void qaoa_tiles_set_ranks(parsec_data_collection_t *dc, const int *ranks, int nb_ranks);
/** The memory of tile k, or NULL on a rank that does not hold it. */
void *qaoa_tiles_ptr(parsec_data_collection_t *dc, int k);
void  qaoa_tiles_free(parsec_data_collection_t *dc);

/** Fill the tiles of E that this rank holds: tile b is the boundary of
 *  block b before any neighbour is folded in. */
void qaoa_empty_boundaries(parsec_data_collection_t *E, const qaoa_instance_t *inst);

/** Make an arena of blocks of `bytes` bytes, which travel between ranks as
 *  that many bytes. */
void qaoa_arena_set(parsec_arena_datatype_t *adt, size_t bytes);
void qaoa_arena_unset(parsec_arena_datatype_t *adt);

/* ------------------------------------------------------------------------
 * What one update of a block reports, and what a round adds up to. These go
 * to the tasks that keep the accounts; nothing in the algorithm reads them.
 * --------------------------------------------------------------------- */

typedef struct qaoa_report_s {
    double   improvement;       /**< of the cut; 0 when the assignment stayed */
    uint32_t updates;           /**< 1 for one update; their number in a sum */
    uint32_t accepted;          /**< the assignment changed */
    uint32_t skipped;           /**< nothing was evaluated */
    uint32_t evaluations;       /**< jobs that gave counts */
    uint32_t failed;            /**< jobs that did not */
    uint32_t pad;
} qaoa_report_t;

typedef char qaoa_report_is_32_bytes[(32 == sizeof(qaoa_report_t)) ? 1 : -1];

/** Sizes, in bytes, of the kinds of data the tasks exchange, for one
 *  instance and one configuration. Every rank computes the same ones. */
typedef struct qaoa_sizes_s {
    size_t state;       /**< qaoa_state_t */
    size_t boundary;    /**< for the largest block */
    size_t local;       /**< for the largest block with the most edges inside one */
    size_t circuit;     /**< for the same */
    size_t opt;         /**< qaoa_opt_t */
    size_t theta;       /**< the angles of one job */
    size_t result;      /**< with room for every outcome the largest block can give */
    size_t eval;        /**< qaoa_eval_t */
    size_t report;      /**< qaoa_report_t */
    size_t all;         /**< the states of all the blocks, one after the other */
} qaoa_sizes_t;

void qaoa_sizes(qaoa_sizes_t *sizes, const qaoa_instance_t *inst, const qaoa_config_t *cfg);

/* ------------------------------------------------------------------------
 * The taskpool (qaoa.jdf)
 * --------------------------------------------------------------------- */

/**
 * The task graph of cfg->rounds sweeps over the blocks of an instance.
 *
 * @param[in] X  one tile per block, of sizes->state bytes: where the tasks
 *               of a block run, and the memory its state starts in
 * @param[in] Q  one tile with no data, on the rank that owns the QPU: where
 *               the quantum tasks run
 * @param[in] A  one tile of sizes->all bytes: it receives the final state of
 *               every block, in block order
 * @param[in] G  one tile of sizes->report bytes per round, zeroed: it
 *               receives what the updates of that round add up to
 * @param[in] E  one tile of sizes->boundary bytes per block, placed as X is
 *               and filled by qaoa_empty_boundaries(): the boundary of a
 *               block that has no neighbour
 *
 * @param[in] colour_sync  the blocks of a colour start only once every block
 *               of the colour before has had its update: the schedule of a
 *               code with a barrier between colours (C-sync). The results
 *               are the same
 * @param[in] log  keep the timestamps of every job and every update of this
 *               process (qaoa_taskpool_log)
 *
 * Returns NULL for a configuration that is not a run: no round, no step, no
 * shot, or a number of layers outside 1 .. QAOA_MAX_LAYERS.
 *
 * The collections are for this taskpool alone: the tiles of X remember its
 * datatype for a state, which is freed with it. Make new ones for another.
 *
 * The instance, the configuration and the collections have to outlive the
 * taskpool, and to be the same on every rank.
 */
parsec_taskpool_t *qaoa_taskpool_new(const qaoa_instance_t *inst, const qaoa_config_t *cfg,
                                     parsec_data_collection_t *X, parsec_data_collection_t *Q,
                                     parsec_data_collection_t *A, parsec_data_collection_t *G,
                                     parsec_data_collection_t *E, const qaoa_sizes_t *sizes,
                                     int colour_sync, int log, int nb_qpu_ranks, const int *qpus);
/** To be called once the taskpool has been waited for. */
void qaoa_taskpool_free(parsec_taskpool_t *tp);

/** One job of the log: the timestamps of its result (nanoseconds on the
 *  monotonic clock of the rank that owns the QPU), and when its EVAL ran on
 *  the rank of its block. Indexed ((r K + b) T + t) 2 + e; done is 0 for a
 *  job whose EVAL ran on another rank. */
typedef struct qaoa_job_record_s {
    int64_t  t_hook, t_submit, t_start, t_end, t_retrieve, t_eval;
    uint32_t batch, attempts, status, done;
} qaoa_job_record_t;

/** One update of the log: when its BUILD began and its UPDATE ended, on the
 *  rank of its block. Indexed r K + b; 0 for a block of another rank. */
typedef struct qaoa_update_record_s {
    int64_t t_build, t_update;
} qaoa_update_record_t;

/** The log of the last taskpool made with log set, until it is freed; NULL
 *  otherwise. */
void qaoa_taskpool_log(const qaoa_job_record_t **jobs, const qaoa_update_record_t **updates);

/** Tasks of this process, since it started, that read the state of a block
 *  after another number of updates than their place in the graph implies:
 *  zero, unless a dependency of the graph is wrong. */
int qaoa_taskpool_violations(void);

/** JOIN tasks of the colour-synchronous control run in this process since
 *  it started: one per update when colour_sync is set, none otherwise. */
int qaoa_taskpool_joins(void);

END_C_DECLS

#endif  /* QAOA_DATA_H */
