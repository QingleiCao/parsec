/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "qaoa_graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* splitmix64 finalizer: every "random" choice here is a hash of the seed and
 * of what is being chosen, so a graph depends on nothing else. */
static uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/* ------------------------------------------------------------------------
 * Graphs
 * --------------------------------------------------------------------- */

static int cmp_pair(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

int qaoa_graph_from_edges(qaoa_graph_t *g, int n, int m, const int *eu, const int *ev, const double *ew)
{
    uint64_t *pairs;
    int *fill;

    memset(g, 0, sizeof(*g));
    if( (n <= 0) || (m < 0) ) return -1;

    /* No loop, no vertex out of range, no pair twice. */
    pairs = (uint64_t*)malloc((size_t)(m > 0 ? m : 1) * sizeof(uint64_t));
    for( int e = 0; e < m; e++ ) {
        const int u = (eu[e] < ev[e]) ? eu[e] : ev[e], v = (eu[e] < ev[e]) ? ev[e] : eu[e];
        if( (u < 0) || (v >= n) || (u == v) ) { free(pairs); return -1; }
        pairs[e] = ((uint64_t)u << 32) | (uint64_t)v;
    }
    qsort(pairs, (size_t)m, sizeof(uint64_t), cmp_pair);
    for( int e = 1; e < m; e++ ) {
        if( pairs[e] == pairs[e - 1] ) { free(pairs); return -1; }
    }
    free(pairs);

    g->n  = n;
    g->m  = m;
    g->eu = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    g->ev = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    g->ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    g->adj_off = (int*)calloc((size_t)n + 1, sizeof(int));
    g->adj_v   = (int*)malloc((size_t)(m > 0 ? 2 * m : 1) * sizeof(int));
    g->adj_w   = (double*)malloc((size_t)(m > 0 ? 2 * m : 1) * sizeof(double));
    for( int e = 0; e < m; e++ ) {
        g->eu[e] = (eu[e] < ev[e]) ? eu[e] : ev[e];
        g->ev[e] = (eu[e] < ev[e]) ? ev[e] : eu[e];
        g->ew[e] = ew[e];
        g->adj_off[g->eu[e] + 1]++;
        g->adj_off[g->ev[e] + 1]++;
    }
    for( int v = 0; v < n; v++ ) g->adj_off[v + 1] += g->adj_off[v];
    fill = (int*)malloc((size_t)n * sizeof(int));
    memcpy(fill, g->adj_off, (size_t)n * sizeof(int));
    for( int e = 0; e < m; e++ ) {
        g->adj_v[fill[g->eu[e]]] = g->ev[e]; g->adj_w[fill[g->eu[e]]++] = g->ew[e];
        g->adj_v[fill[g->ev[e]]] = g->eu[e]; g->adj_w[fill[g->ev[e]]++] = g->ew[e];
    }
    free(fill);
    return 0;
}

void qaoa_graph_free(qaoa_graph_t *g)
{
    free(g->eu); free(g->ev); free(g->ew);
    free(g->adj_off); free(g->adj_v); free(g->adj_w);
    memset(g, 0, sizeof(*g));
}

/* A weight k/8, k in 1 .. 16, from 64 random bits. */
static double weight_of(uint64_t bits)
{
    return (double)(1 + bits % 16) / 8.0;
}

/* A growing list of edges that refuses a pair it has. */
typedef struct {
    int *u, *v;
    double *w;
    int m, cap;
} edge_list_t;

static int edge_list_has(const edge_list_t *l, int from, int u, int v)
{
    for( int e = from; e < l->m; e++ ) {
        if( ((l->u[e] == u) && (l->v[e] == v)) || ((l->u[e] == v) && (l->v[e] == u)) ) return 1;
    }
    return 0;
}

static void edge_list_add(edge_list_t *l, int u, int v, double w)
{
    if( l->m == l->cap ) {
        l->cap = (0 == l->cap) ? 64 : 2 * l->cap;
        l->u = (int*)realloc(l->u, (size_t)l->cap * sizeof(int));
        l->v = (int*)realloc(l->v, (size_t)l->cap * sizeof(int));
        l->w = (double*)realloc(l->w, (size_t)l->cap * sizeof(double));
    }
    l->u[l->m] = u; l->v[l->m] = v; l->w[l->m] = w;
    l->m++;
}

static void edge_list_free(edge_list_t *l)
{
    free(l->u); free(l->v); free(l->w);
}

int qaoa_graph_random(qaoa_graph_t *g, int n, int m, uint64_t seed, int zero_one_in, int negative_one_in)
{
    edge_list_t list = { NULL, NULL, NULL, 0, 0 };
    unsigned char *taken;
    uint64_t draw = 0;
    int rc;

    if( (n < 2) || (n > 4096) || (m < 0) || ((long)m > (long)n * (n - 1) / 2) ) return -1;
    taken = (unsigned char*)calloc((size_t)n * n, 1);
    while( list.m < m ) {
        const uint64_t h = mix(mix(seed) ^ (++draw));
        const int u = (int)(h % (uint64_t)n);
        int v = (int)((h >> 32) % (uint64_t)(n - 1));
        double w;
        if( v >= u ) v++;
        if( taken[u * n + v] ) continue;
        taken[u * n + v] = taken[v * n + u] = 1;
        /* The weight, whether it is zero and its sign are three separate
         * draws on the edge's number: the same seed gives the same edges and
         * the same sizes of weights whatever the two options are. */
        w = weight_of(mix(mix(seed + 1) ^ (uint64_t)list.m));
        if( (zero_one_in > 0) && (0 == mix(mix(seed + 2) ^ (uint64_t)list.m) % (uint64_t)zero_one_in) ) w = 0.0;
        if( (negative_one_in > 0) && (0 == mix(mix(seed + 3) ^ (uint64_t)list.m) % (uint64_t)negative_one_in) ) w = -w;
        edge_list_add(&list, u, v, w);
    }
    free(taken);
    rc = qaoa_graph_from_edges(g, n, list.m, list.u, list.v, list.w);
    edge_list_free(&list);
    return rc;
}

/* The pairs of groups that have edges between them, for a graph of groups
 * shaped as `topology`. Returns their number, or -1 for a topology that
 * does not exist or does not fit nb_blocks. */
static int block_pairs(const char *topology, int nb_blocks, uint64_t seed, int **pa, int **pb)
{
    int n = 0, cap = 16, *a = (int*)malloc((size_t)cap * sizeof(int)), *b = (int*)malloc((size_t)cap * sizeof(int));
#define PAIR(X, Y)                                                                          \
    do {                                                                                    \
        if( n == cap ) {                                                                    \
            cap *= 2;                                                                       \
            a = (int*)realloc(a, (size_t)cap * sizeof(int)); b = (int*)realloc(b, (size_t)cap * sizeof(int)); \
        }                                                                                   \
        a[n] = (X); b[n] = (Y); n++;                                                        \
    } while(0)

    if( (0 == strcmp(topology, "ring")) || (0 == strcmp(topology, "rand")) ) {
        /* Group b and b+1, cyclically: one pair for two groups. */
        for( int g = 0; (g < nb_blocks) && (nb_blocks > 1); g++ ) {
            if( (2 == nb_blocks) && (1 == g) ) break;
            PAIR(g, (g + 1) % nb_blocks);
        }
        if( 0 == strcmp(topology, "rand") ) {
            /* Then pairs drawn from the seed until every group has four
             * neighbours at least, or all it can have. */
            int *deg = (int*)calloc((size_t)nb_blocks, sizeof(int));
            const int want = (nb_blocks - 1 < 4) ? nb_blocks - 1 : 4;
            uint64_t draw = 0;
            for( int i = 0; i < n; i++ ) { deg[a[i]]++; deg[b[i]]++; }
            for( int g = 0; g < nb_blocks; g++ ) {
                while( deg[g] < want ) {
                    const int other = (int)(mix(mix(seed ^ 0x7a6e5f) ^ (++draw)) % (uint64_t)nb_blocks);
                    int have = (other == g);
                    for( int i = 0; (i < n) && !have; i++ ) {
                        have = ((a[i] == g) && (b[i] == other)) || ((a[i] == other) && (b[i] == g));
                    }
                    if( have ) continue;
                    PAIR(g, other);
                    deg[g]++; deg[other]++;
                }
            }
            free(deg);
        }
    } else if( 0 == strcmp(topology, "grid") ) {
        /* A square grid, row after row: each group and the ones to its right and below. */
        int side = 1;
        while( side * side < nb_blocks ) side++;
        if( side * side != nb_blocks ) { free(a); free(b); return -1; }
        for( int g = 0; g < nb_blocks; g++ ) {
            if( (g % side) + 1 < side ) PAIR(g, g + 1);
            if( g + side < nb_blocks )  PAIR(g, g + side);
        }
    } else if( 0 == strcmp(topology, "complete") ) {
        for( int g = 0; g < nb_blocks; g++ ) {
            for( int h = g + 1; h < nb_blocks; h++ ) PAIR(g, h);
        }
    } else {
        free(a); free(b);
        return -1;
    }
#undef PAIR
    *pa = a; *pb = b;
    return n;
}

int qaoa_graph_blocks(qaoa_graph_t *g, int nb_blocks, int block_size, int chords, int cross, uint64_t seed)
{
    return qaoa_graph_blocks_shaped(g, "ring", nb_blocks, block_size, chords, cross, seed);
}

int qaoa_graph_blocks_shaped(qaoa_graph_t *g, const char *topology, int nb_blocks, int block_size, int chords, int cross,
                             uint64_t seed)
{
    edge_list_t list = { NULL, NULL, NULL, 0, 0 };
    uint64_t draw = 0;
    int *pa = NULL, *pb = NULL, nb_pairs;
    int rc;

    if( (nb_blocks < 1) || (block_size < 1) || (chords < 0) || (cross < 0) ) return -1;
    /* What is asked for has to exist: chords beyond the cycle, and distinct
     * pairs between two groups. */
    if( chords > block_size * (block_size - 1) / 2 - ((block_size >= 3) ? block_size : block_size - 1) ) return -1;
    if( (nb_blocks > 1) && (cross > block_size * block_size) ) return -1;
    if( 0 > (nb_pairs = block_pairs(topology, nb_blocks, seed, &pa, &pb)) ) return -1;

    for( int b = 0; b < nb_blocks; b++ ) {
        const int base = b * block_size, first = list.m;
        /* The cycle of the group, which is one edge for two vertices. */
        for( int i = 0; (i < block_size) && (block_size >= 2); i++ ) {
            if( (2 == block_size) && (1 == i) ) break;
            edge_list_add(&list, base + i, base + (i + 1) % block_size, weight_of(mix(mix(seed) ^ (++draw))));
        }
        for( int c = 0; c < chords; ) {
            const uint64_t h = mix(mix(seed) ^ (++draw));
            const int i = (int)(h % (uint64_t)block_size);
            int j = (int)((h >> 32) % (uint64_t)(block_size - 1));
            if( j >= i ) j++;
            if( edge_list_has(&list, first, base + i, base + j) ) continue;
            edge_list_add(&list, base + i, base + j, weight_of(mix(h)));
            c++;
        }
    }
    /* The edges between groups, pair after pair. For a ring these are the
     * draws they always were. */
    for( int i = 0; (i < nb_pairs) && (cross > 0); i++ ) {
        const int first = list.m;
        for( int c = 0; c < cross; ) {
            const uint64_t h = mix(mix(seed) ^ (++draw));
            const int u = pa[i] * block_size + (int)(h % (uint64_t)block_size);
            const int v = pb[i] * block_size + (int)((h >> 32) % (uint64_t)block_size);
            if( edge_list_has(&list, first, u, v) ) continue;
            edge_list_add(&list, u, v, weight_of(mix(h)));
            c++;
        }
    }
    free(pa); free(pb);
    rc = qaoa_graph_from_edges(g, nb_blocks * block_size, list.m, list.u, list.v, list.w);
    edge_list_free(&list);
    return rc;
}

double qaoa_graph_cut(const qaoa_graph_t *g, const unsigned char *x)
{
    double cut = 0.0;
    for( int e = 0; e < g->m; e++ ) {
        if( x[g->eu[e]] != x[g->ev[e]] ) cut += g->ew[e];
    }
    return cut;
}

/* ------------------------------------------------------------------------
 * Partitions
 * --------------------------------------------------------------------- */

int qaoa_partition_from_blocks(qaoa_partition_t *p, const qaoa_graph_t *g, int nb_blocks, const int *block_of)
{
    const int n = g->n;
    unsigned char *adjacent = NULL;
    int *fill = NULL, *internal = NULL, rc = -1;

    memset(p, 0, sizeof(*p));
    if( nb_blocks < 1 ) return -1;
    p->nb_blocks = nb_blocks;
    p->block_of  = (int*)malloc((size_t)n * sizeof(int));
    p->local_of  = (int*)malloc((size_t)n * sizeof(int));
    p->blk_off   = (int*)calloc((size_t)nb_blocks + 1, sizeof(int));
    p->blk_v     = (int*)malloc((size_t)n * sizeof(int));
    p->deg       = (int*)calloc((size_t)nb_blocks, sizeof(int));
    p->nbr_off   = (int*)calloc((size_t)nb_blocks + 1, sizeof(int));
    p->colour    = (int*)malloc((size_t)nb_blocks * sizeof(int));
    fill         = (int*)malloc((size_t)nb_blocks * sizeof(int));
    internal     = (int*)calloc((size_t)nb_blocks, sizeof(int));

    for( int v = 0; v < n; v++ ) {
        if( (block_of[v] < 0) || (block_of[v] >= nb_blocks) ) goto done;
        p->block_of[v] = block_of[v];
        p->blk_off[block_of[v] + 1]++;
    }
    for( int b = 0; b < nb_blocks; b++ ) {
        const int size = p->blk_off[b + 1];
        if( (0 == size) || (size > QAOA_MAX_BLOCK) ) goto done;
        if( size > p->max_block ) p->max_block = size;
        p->blk_off[b + 1] += p->blk_off[b];
    }
    /* Vertices in increasing order inside their block: the smallest is qubit 0. */
    memcpy(fill, p->blk_off, (size_t)nb_blocks * sizeof(int));
    for( int v = 0; v < n; v++ ) {
        const int b = block_of[v];
        p->local_of[v] = fill[b] - p->blk_off[b];
        p->blk_v[fill[b]++] = v;
    }

    /* Blocks that share an edge of non-zero weight. An edge of weight zero
     * ties nothing together. */
    adjacent = (unsigned char*)calloc((size_t)nb_blocks * nb_blocks, 1);
    for( int e = 0; e < g->m; e++ ) {
        const int a = block_of[g->eu[e]], b = block_of[g->ev[e]];
        if( 0.0 == g->ew[e] ) continue;
        if( a == b ) { internal[a]++; continue; }
        adjacent[(size_t)a * nb_blocks + b] = adjacent[(size_t)b * nb_blocks + a] = 1;
    }
    for( int b = 0; b < nb_blocks; b++ ) {
        if( internal[b] > p->max_internal ) p->max_internal = internal[b];
        for( int d = 0; d < nb_blocks; d++ ) p->deg[b] += adjacent[(size_t)b * nb_blocks + d];
        if( p->deg[b] > QAOA_MAX_NBR ) goto done;
        if( p->deg[b] > p->max_deg ) p->max_deg = p->deg[b];
        p->nbr_off[b + 1] = p->nbr_off[b] + p->deg[b];
    }
    p->nbr = (int*)malloc((size_t)(p->nbr_off[nb_blocks] > 0 ? p->nbr_off[nb_blocks] : 1) * sizeof(int));
    p->rev = (int*)malloc((size_t)(p->nbr_off[nb_blocks] > 0 ? p->nbr_off[nb_blocks] : 1) * sizeof(int));
    for( int b = 0; b < nb_blocks; b++ ) {
        int k = 0;
        for( int d = 0; d < nb_blocks; d++ ) {
            if( adjacent[(size_t)b * nb_blocks + d] ) p->nbr[p->nbr_off[b] + k++] = d;
        }
    }
    for( int b = 0; b < nb_blocks; b++ ) {
        for( int k = 0; k < p->deg[b]; k++ ) {
            const int d = p->nbr[p->nbr_off[b] + k];
            for( int j = 0; j < p->deg[d]; j++ ) {
                if( p->nbr[p->nbr_off[d] + j] == b ) p->rev[p->nbr_off[b] + k] = j;
            }
        }
    }

    /* Greedy colouring in block order: the smallest colour no earlier
     * neighbour has. */
    for( int b = 0; b < nb_blocks; b++ ) {
        uint64_t used = 0;
        int c = 0;
        for( int k = 0; k < p->deg[b]; k++ ) {
            const int d = p->nbr[p->nbr_off[b] + k];
            if( d < b ) used |= ((uint64_t)1) << p->colour[d];
        }
        while( used & (((uint64_t)1) << c) ) c++;
        p->colour[b] = c;
        if( c + 1 > p->nb_colours ) p->nb_colours = c + 1;
    }
    rc = 0;

  done:
    free(adjacent);
    free(fill);
    free(internal);
    if( 0 != rc ) qaoa_partition_free(p);
    return rc;
}

int qaoa_partition_contiguous(qaoa_partition_t *p, const qaoa_graph_t *g, int block_size)
{
    int *block_of, rc;

    if( block_size < 1 ) return -1;
    block_of = (int*)malloc((size_t)g->n * sizeof(int));
    for( int v = 0; v < g->n; v++ ) block_of[v] = v / block_size;
    rc = qaoa_partition_from_blocks(p, g, (g->n + block_size - 1) / block_size, block_of);
    free(block_of);
    return rc;
}

void qaoa_partition_free(qaoa_partition_t *p)
{
    free(p->block_of); free(p->local_of); free(p->blk_off); free(p->blk_v);
    free(p->deg); free(p->nbr_off); free(p->nbr); free(p->rev); free(p->colour);
    memset(p, 0, sizeof(*p));
}

/* ------------------------------------------------------------------------
 * Instances
 * --------------------------------------------------------------------- */

const char *const qaoa_instance_names[] = {
    "edge2", "edge2-split", "tri", "c4", "c4-cross", "path6",
    "rand12", "rand12neg", "rand16", "disc16", "ring64", NULL
};

int qaoa_instance_named(qaoa_instance_t *inst, const char *name)
{
    int rc = -1;

    memset(inst, 0, sizeof(*inst));
    snprintf(inst->name, sizeof(inst->name), "%s", name);

    if( (0 == strcmp(name, "edge2")) || (0 == strcmp(name, "edge2-split")) ) {
        const int eu[] = { 0 }, ev[] = { 1 }, one[] = { 0, 0 }, two[] = { 0, 1 };
        const double ew[] = { 1.5 };
        const int split = (0 == strcmp(name, "edge2-split"));
        rc = qaoa_graph_from_edges(&inst->graph, 2, 1, eu, ev, ew);
        if( 0 == rc ) rc = qaoa_partition_from_blocks(&inst->part, &inst->graph, split ? 2 : 1, split ? two : one);
    } else if( 0 == strcmp(name, "tri") ) {
        const int eu[] = { 0, 1, 0 }, ev[] = { 1, 2, 2 }, blocks[] = { 0, 0, 1 };
        const double ew[] = { 1.0, 2.0, 3.0 };
        rc = qaoa_graph_from_edges(&inst->graph, 3, 3, eu, ev, ew);
        if( 0 == rc ) rc = qaoa_partition_from_blocks(&inst->part, &inst->graph, 2, blocks);
    } else if( (0 == strcmp(name, "c4")) || (0 == strcmp(name, "c4-cross")) ) {
        const int eu[] = { 0, 1, 2, 3 }, ev[] = { 1, 2, 3, 0 };
        const int side[] = { 0, 0, 1, 1 }, cross[] = { 0, 1, 0, 1 };
        const double ew[] = { 1.0, 0.5, 1.25, 0.75 };
        rc = qaoa_graph_from_edges(&inst->graph, 4, 4, eu, ev, ew);
        if( 0 == rc ) rc = qaoa_partition_from_blocks(&inst->part, &inst->graph, 2,
                                                      (0 == strcmp(name, "c4")) ? side : cross);
    } else if( 0 == strcmp(name, "path6") ) {
        /* Powers of two: no two sets of edges have the same total weight. */
        const int eu[] = { 0, 1, 2, 3, 4 }, ev[] = { 1, 2, 3, 4, 5 };
        const double ew[] = { 0.125, 0.25, 0.5, 1.0, 2.0 };
        rc = qaoa_graph_from_edges(&inst->graph, 6, 5, eu, ev, ew);
        if( 0 == rc ) rc = qaoa_partition_contiguous(&inst->part, &inst->graph, 3);
    } else if( (0 == strcmp(name, "rand12")) || (0 == strcmp(name, "rand12neg")) ) {
        rc = qaoa_graph_random(&inst->graph, 12, 24, 12, 6, (0 == strcmp(name, "rand12")) ? 0 : 3);
        if( 0 == rc ) rc = qaoa_partition_contiguous(&inst->part, &inst->graph, 4);
    } else if( 0 == strcmp(name, "rand16") ) {
        rc = qaoa_graph_random(&inst->graph, 16, 36, 16, 6, 0);
        if( 0 == rc ) rc = qaoa_partition_contiguous(&inst->part, &inst->graph, 4);
    } else if( 0 == strcmp(name, "disc16") ) {
        rc = qaoa_graph_blocks(&inst->graph, 4, 4, 1, 0, 160);
        if( 0 == rc ) rc = qaoa_partition_contiguous(&inst->part, &inst->graph, 4);
    } else if( 0 == strcmp(name, "ring64") ) {
        rc = qaoa_graph_blocks(&inst->graph, 8, 8, 4, 2, 640);
        if( 0 == rc ) rc = qaoa_partition_contiguous(&inst->part, &inst->graph, 8);
    }
    if( 0 != rc ) qaoa_instance_free(inst);
    return rc;
}

int qaoa_instance_spec(qaoa_instance_t *inst, const char *spec, uint64_t seed)
{
    char topology[16] = "ring";
    int k = 0, n = 0, c = 0, x = 0, rc;
    const char *numbers = spec;
    const char *colon = strchr(spec, ':');

    if( (NULL != colon) && ((spec[0] < '0') || (spec[0] > '9')) ) {
        if( (size_t)(colon - spec) >= sizeof(topology) ) return -1;
        memcpy(topology, spec, (size_t)(colon - spec));
        topology[colon - spec] = '\0';
        numbers = colon + 1;
    }
    if( 4 != sscanf(numbers, "%d:%d:%d:%d", &k, &n, &c, &x) ) return -1;
    memset(inst, 0, sizeof(*inst));
    rc = qaoa_graph_blocks_shaped(&inst->graph, topology, k, n, c, x, seed);
    if( 0 == strcmp(topology, "ring") ) snprintf(inst->name, sizeof(inst->name), "blocks%dx%d", k, n);
    else snprintf(inst->name, sizeof(inst->name), "%s%dx%d", topology, k, n);
    if( 0 == rc ) rc = qaoa_partition_contiguous(&inst->part, &inst->graph, n);
    if( 0 != rc ) qaoa_instance_free(inst);
    return rc;
}

int qaoa_instance_blocks(qaoa_instance_t *inst, int nb_blocks, int block_size, int chords, int cross, uint64_t seed)
{
    int rc;

    memset(inst, 0, sizeof(*inst));
    snprintf(inst->name, sizeof(inst->name), "blocks%dx%d", nb_blocks, block_size);
    rc = qaoa_graph_blocks(&inst->graph, nb_blocks, block_size, chords, cross, seed);
    if( 0 == rc ) rc = qaoa_partition_contiguous(&inst->part, &inst->graph, block_size);
    if( 0 != rc ) qaoa_instance_free(inst);
    return rc;
}

void qaoa_instance_free(qaoa_instance_t *inst)
{
    qaoa_partition_free(&inst->part);
    qaoa_graph_free(&inst->graph);
}
