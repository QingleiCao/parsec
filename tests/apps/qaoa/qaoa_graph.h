/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef QAOA_GRAPH_H
#define QAOA_GRAPH_H

/**
 * @file
 * Weighted graphs, their partition into blocks, and the colouring of the
 * blocks, for the block-decomposed QAOA of weighted MaxCut.
 *
 * Everything here is computed before any task exists, from a name or a few
 * integers and a seed, and gives the same tables in every process: the task
 * graph of the distributed application is built from them.
 *
 * Weights are multiples of 1/8, so that every sum of weights the application
 * forms is exact in a double and results can be compared with ==.
 */

#include <stdint.h>
#include <stddef.h>

/** A block is at most one measured bitstring wide. */
#define QAOA_MAX_BLOCK   64
/** Most blocks a block may share an edge with. */
#define QAOA_MAX_NBR     32

typedef struct qaoa_graph_s {
    int     n;          /**< vertices, 0 .. n-1 */
    int     m;          /**< edges, each once, zero weights included */
    int    *eu, *ev;    /**< [m] end points, eu < ev */
    double *ew;         /**< [m] weights */
    int    *adj_off;    /**< [n+1] the edges at vertex v are adj_off[v] .. adj_off[v+1]-1 */
    int    *adj_v;      /**< [2m] the other end */
    double *adj_w;      /**< [2m] the weight */
} qaoa_graph_t;

typedef struct qaoa_partition_s {
    int  nb_blocks;
    int *block_of;      /**< [n] the block of a vertex */
    int *local_of;      /**< [n] its index inside that block: its qubit, and its bit */
    int *blk_off;       /**< [nb_blocks+1] the vertices of block b are blk_v[blk_off[b]] .., in local order */
    int *blk_v;         /**< [n] */
    int  max_block;     /**< size of the largest block */
    int  max_internal;  /**< most edges of non-zero weight inside one block */
    /* The quotient graph: blocks that share an edge of non-zero weight. */
    int *deg;           /**< [nb_blocks] */
    int *nbr_off;       /**< [nb_blocks+1] */
    int *nbr;           /**< the k-th neighbour of b is nbr[nbr_off[b] + k], in increasing order */
    int *rev;           /**< the slot of b in the list of that neighbour */
    int  max_deg;
    int *colour;        /**< [nb_blocks] neighbours have different colours */
    int  nb_colours;
} qaoa_partition_t;

/** A graph with its partition. */
typedef struct qaoa_instance_s {
    char             name[32];
    qaoa_graph_t     graph;
    qaoa_partition_t part;
} qaoa_instance_t;

/* ------------------------------------------------------------------------
 * Graphs
 * --------------------------------------------------------------------- */

/** Build a graph from a list of edges. Each pair of vertices may appear once.
 *  Returns 0, or -1 when an edge is a loop, out of range or repeated. */
int  qaoa_graph_from_edges(qaoa_graph_t *g, int n, int m, const int *eu, const int *ev, const double *ew);
void qaoa_graph_free(qaoa_graph_t *g);

/** A random graph on n vertices with m edges drawn from the seed. Weights
 *  are k/8 for k in 1 .. 16; one edge in zero_one_in has weight 0 (0: none)
 *  and one in negative_one_in is negative (0: none). */
int qaoa_graph_random(qaoa_graph_t *g, int n, int m, uint64_t seed, int zero_one_in, int negative_one_in);

/** nb_blocks groups of block_size vertices. Inside a group: a cycle and
 *  `chords` more edges. Between groups b and b+1 (cyclically): `cross`
 *  edges; none at all when cross is 0. Vertex v is in group v / block_size. */
int qaoa_graph_blocks(qaoa_graph_t *g, int nb_blocks, int block_size, int chords, int cross, uint64_t seed);

/** The same with the groups joined as `topology` says, `cross` edges per
 *  pair of groups that are joined:
 *    ring      b and b+1, cyclically (what qaoa_graph_blocks does)
 *    grid      a square grid of groups, row after row (nb_blocks a square)
 *    complete  every pair
 *    rand      the ring, and pairs drawn from the seed until every group has
 *              four neighbours, or all it can have
 *  Returns -1 for a topology that does not exist or does not fit. */
int qaoa_graph_blocks_shaped(qaoa_graph_t *g, const char *topology, int nb_blocks, int block_size, int chords, int cross,
                             uint64_t seed);

/** The cut of an assignment given as one 0/1 byte per vertex. */
double qaoa_graph_cut(const qaoa_graph_t *g, const unsigned char *x);

/* ------------------------------------------------------------------------
 * Partitions
 * --------------------------------------------------------------------- */

/** Partition a graph from the block of each vertex (0 .. nb_blocks-1, every
 *  block used), then build the quotient graph and colour it greedily in
 *  block order. Returns 0, or -1 when a block is empty, larger than
 *  QAOA_MAX_BLOCK, or has more than QAOA_MAX_NBR neighbours. */
int  qaoa_partition_from_blocks(qaoa_partition_t *p, const qaoa_graph_t *g, int nb_blocks, const int *block_of);
/** Vertex v in block v / block_size. */
int  qaoa_partition_contiguous(qaoa_partition_t *p, const qaoa_graph_t *g, int block_size);
void qaoa_partition_free(qaoa_partition_t *p);

static inline int qaoa_block_size(const qaoa_partition_t *p, int b) { return p->blk_off[b + 1] - p->blk_off[b]; }
static inline int qaoa_block_vertex(const qaoa_partition_t *p, int b, int i) { return p->blk_v[p->blk_off[b] + i]; }
static inline int qaoa_block_nbr(const qaoa_partition_t *p, int b, int k) { return p->nbr[p->nbr_off[b] + k]; }
static inline int qaoa_block_rev(const qaoa_partition_t *p, int b, int k) { return p->rev[p->nbr_off[b] + k]; }

/* ------------------------------------------------------------------------
 * Instances
 * --------------------------------------------------------------------- */

/** The names qaoa_instance_named() knows, NULL-terminated. */
extern const char *const qaoa_instance_names[];

/**
 * One of the fixed test instances:
 *   edge2        two vertices, one edge, one block
 *   edge2-split  the same in two blocks of one vertex
 *   tri          a triangle with weights 1, 2, 3; blocks {0,1} and {2}
 *   c4           a four-cycle; blocks {0,1} and {2,3}
 *   c4-cross     the same with blocks {0,2} and {1,3}: every edge crosses
 *   path6        a path of six vertices with five different weights, in two blocks of three
 *   rand12       12 vertices, 3 blocks, some edges of weight 0
 *   rand12neg    the same graph with some negative weights
 *   rand16       16 vertices, 4 blocks, some edges of weight 0
 *   disc16       4 blocks of 4 with no edge between them
 *   ring64       8 blocks of 8 in a ring, few edges between neighbours
 * Returns 0, or -1 for an unknown name.
 */
int  qaoa_instance_named(qaoa_instance_t *inst, const char *name);

/** nb_blocks blocks of block_size vertices as qaoa_graph_blocks() makes
 *  them, partitioned by group. For runs of a chosen size. */
int  qaoa_instance_blocks(qaoa_instance_t *inst, int nb_blocks, int block_size, int chords, int cross, uint64_t seed);

/** An instance from a specification "[TOPOLOGY:]K:N:C:X": K groups of N
 *  vertices, C chords per group, X edges per pair of joined groups, joined as
 *  TOPOLOGY says (ring by default; see qaoa_graph_blocks_shaped), one block
 *  per group. What -g takes in the programs. Returns 0 or -1. */
int qaoa_instance_spec(qaoa_instance_t *inst, const char *spec, uint64_t seed);

void qaoa_instance_free(qaoa_instance_t *inst);

#endif  /* QAOA_GRAPH_H */
