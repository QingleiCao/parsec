/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * The arithmetic of the block-decomposed QAOA, without the runtime.
 *
 * Exact parts, compared with ==, since every weight is a multiple of 1/8:
 *   - graphs, partitions and colourings are what they have to be;
 *   - the cut of the whole graph is the local objective of a block plus the
 *     cut of the edges that do not touch it, on every assignment;
 *   - the fields and the constant are those of the spin form, computed here
 *     a second time from the definition;
 *   - bit i of a block's assignment is its i-th vertex;
 *   - the accept rule is strict;
 *   - the enumerations agree with plain loops.
 *
 * Parts in floating point, to 1e-12:
 *   - the circuit of a local problem prepares the state of the exact layers;
 *   - the expected cut of one edge, and of one vertex in a field, have their
 *     closed forms, and the marginal of each qubit is that of its own field;
 *   - the optimizer does what its formulas say.
 *
 * One statistical part, with fixed seeds: the estimate of a job is within
 * six standard errors of the exact expectation.
 */

#include "qaoa_core.h"
#include "parsec/constants.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nb_errors = 0;

#define EXPECT(COND, ...)                                                   \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            if( nb_errors < 60 ) {                                          \
                fprintf(stderr, "%s:%d: ", __func__, __LINE__);             \
                fprintf(stderr, __VA_ARGS__);                               \
                fprintf(stderr, "\n");                                      \
            }                                                               \
            nb_errors++;                                                    \
        }                                                                   \
    } while(0)

#define TOLERANCE 1e-12

static uint64_t rng_state = 20261123;
static uint64_t rnd(void) { return rng_state = parsec_qpu_mix64(rng_state); }
static double rnd_angle(void) { return (parsec_qpu_unit(rnd()) - 0.5) * 6.0; }

/* The states of all blocks for an assignment of the whole graph. */
static void states_from_vertices(const qaoa_instance_t *inst, const unsigned char *x, qaoa_state_t *states)
{
    memset(states, 0, (size_t)inst->part.nb_blocks * sizeof(qaoa_state_t));
    for( int v = 0; v < inst->graph.n; v++ ) {
        if( x[v] ) states[inst->part.block_of[v]].bits |= ((uint64_t)1) << inst->part.local_of[v];
    }
}

/* The local problem of block b for the states of all blocks. */
static qaoa_local_t *local_new(const qaoa_instance_t *inst, const qaoa_config_t *cfg, int b, const qaoa_state_t *states)
{
    const qaoa_partition_t *p = &inst->part;
    qaoa_boundary_t *h = (qaoa_boundary_t*)malloc(qaoa_boundary_size(p->max_block));
    qaoa_local_t *P = (qaoa_local_t*)malloc(qaoa_local_size(p->max_block, p->max_internal));

    qaoa_boundary_init(h, inst, b);
    for( int k = 0; k < p->deg[b]; k++ ) qaoa_boundary_gather(h, inst, b, k, &states[qaoa_block_nbr(p, b, k)]);
    qaoa_local_build(P, inst, cfg, 0, b, &states[b], h);
    free(h);
    return P;
}

/* ------------------------------------------------------------------------
 * Graphs, partitions, colourings
 * --------------------------------------------------------------------- */

static void test_structure(void)
{
    for( int i = 0; NULL != qaoa_instance_names[i]; i++ ) {
        const char *name = qaoa_instance_names[i];
        qaoa_instance_t inst, again;
        const qaoa_graph_t *g = &inst.graph;
        const qaoa_partition_t *p = &inst.part;
        int total = 0;

        EXPECT(0 == qaoa_instance_named(&inst, name), "%s: could not be built", name);
        EXPECT(0 == qaoa_instance_named(&again, name), "%s: could not be built twice", name);
        if( (0 == g->n) || (0 == again.graph.n) ) continue;

        /* The same name gives the same graph. */
        EXPECT((g->n == again.graph.n) && (g->m == again.graph.m) &&
               (0 == memcmp(g->eu, again.graph.eu, (size_t)g->m * sizeof(int))) &&
               (0 == memcmp(g->ev, again.graph.ev, (size_t)g->m * sizeof(int))) &&
               (0 == memcmp(g->ew, again.graph.ew, (size_t)g->m * sizeof(double))), "%s: built twice, two graphs", name);

        for( int e = 0; e < g->m; e++ ) {
            EXPECT((g->eu[e] >= 0) && (g->eu[e] < g->ev[e]) && (g->ev[e] < g->n), "%s: edge %d is (%d, %d)", name, e, g->eu[e], g->ev[e]);
            EXPECT(g->ew[e] * 8.0 == floor(g->ew[e] * 8.0), "%s: weight %g is not a multiple of 1/8", name, g->ew[e]);
        }
        EXPECT(g->adj_off[g->n] == 2 * g->m, "%s: the adjacency lists hold %d ends for %d edges", name, g->adj_off[g->n], g->m);

        /* Every vertex is in one block, at one place. */
        for( int b = 0; b < p->nb_blocks; b++ ) {
            const int size = qaoa_block_size(p, b);
            EXPECT((size >= 1) && (size <= QAOA_MAX_BLOCK) && (size <= p->max_block), "%s: block %d has %d vertices", name, b, size);
            total += size;
            for( int k = 0; k < size; k++ ) {
                const int v = qaoa_block_vertex(p, b, k);
                EXPECT((p->block_of[v] == b) && (p->local_of[v] == k), "%s: vertex %d is at place %d of block %d, not %d of %d",
                       name, v, p->local_of[v], p->block_of[v], k, b);
                EXPECT((0 == k) || (v > qaoa_block_vertex(p, b, k - 1)), "%s: block %d is not in increasing order", name, b);
            }
        }
        EXPECT(total == g->n, "%s: the blocks hold %d vertices of %d", name, total, g->n);

        /* The neighbours of a block are the blocks it shares an edge of
         * non-zero weight with; the reverse slot is right; neighbours have
         * different colours. */
        for( int e = 0; e < g->m; e++ ) {
            const int a = p->block_of[g->eu[e]], b = p->block_of[g->ev[e]];
            int found = 0;
            if( (a == b) || (0.0 == g->ew[e]) ) continue;
            for( int k = 0; k < p->deg[a]; k++ ) found += (qaoa_block_nbr(p, a, k) == b);
            EXPECT(1 == found, "%s: block %d is %d times in the neighbours of block %d", name, b, found, a);
            EXPECT(p->colour[a] != p->colour[b], "%s: blocks %d and %d share an edge and the colour %d", name, a, b, p->colour[a]);
        }
        /* And no other block: an edge of weight 0 makes no neighbour. Count
         * them again from the edges. */
        for( int b = 0; b < p->nb_blocks; b++ ) {
            int expected = 0, internal = 0;
            for( int d = 0; d < p->nb_blocks; d++ ) {
                int shared = 0;
                for( int e = 0; (e < g->m) && (d != b); e++ ) {
                    const int a1 = p->block_of[g->eu[e]], a2 = p->block_of[g->ev[e]];
                    if( (0.0 != g->ew[e]) && (((a1 == b) && (a2 == d)) || ((a1 == d) && (a2 == b))) ) shared = 1;
                }
                expected += shared;
            }
            for( int e = 0; e < g->m; e++ ) {
                internal += ((0.0 != g->ew[e]) && (p->block_of[g->eu[e]] == b) && (p->block_of[g->ev[e]] == b));
            }
            EXPECT(p->deg[b] == expected, "%s: block %d has %d neighbours, its edges say %d", name, b, p->deg[b], expected);
            EXPECT(internal <= p->max_internal, "%s: block %d has %d edges inside, more than the most (%d)", name, b, internal, p->max_internal);
        }
        for( int b = 0; b < p->nb_blocks; b++ ) {
            EXPECT((p->colour[b] >= 0) && (p->colour[b] < p->nb_colours), "%s: block %d has colour %d of %d", name, b, p->colour[b], p->nb_colours);
            EXPECT(p->deg[b] <= p->max_deg, "%s: block %d has %d neighbours, more than the most", name, b, p->deg[b]);
            for( int k = 0; k < p->deg[b]; k++ ) {
                const int d = qaoa_block_nbr(p, b, k);
                EXPECT(qaoa_block_nbr(p, d, qaoa_block_rev(p, b, k)) == b, "%s: the reverse slot of neighbour %d of block %d is wrong", name, k, b);
                EXPECT((0 == k) || (d > qaoa_block_nbr(p, b, k - 1)), "%s: the neighbours of block %d are not in order", name, b);
            }
        }
        qaoa_instance_free(&inst);
        qaoa_instance_free(&again);
    }

    /* What the instances are for. */
    {
        qaoa_instance_t a, b;
        int zeros = 0, negative = 0, same = 1;

        (void)qaoa_instance_named(&a, "c4-cross");
        EXPECT((2 == a.part.nb_blocks) && (0 == a.part.max_internal), "c4-cross: an edge is inside a block");
        qaoa_instance_free(&a);

        (void)qaoa_instance_named(&a, "disc16");
        EXPECT((4 == a.part.nb_blocks) && (0 == a.part.max_deg) && (1 == a.part.nb_colours), "disc16: %d blocks, %d neighbours at most, %d colours",
               a.part.nb_blocks, a.part.max_deg, a.part.nb_colours);
        qaoa_instance_free(&a);

        (void)qaoa_instance_named(&a, "ring64");
        EXPECT((64 == a.graph.n) && (8 == a.part.nb_blocks) && (2 == a.part.max_deg) && (2 == a.part.nb_colours),
               "ring64: %d vertices, %d blocks, %d neighbours at most, %d colours", a.graph.n, a.part.nb_blocks, a.part.max_deg, a.part.nb_colours);
        qaoa_instance_free(&a);

        (void)qaoa_instance_named(&a, "edge2-split");
        EXPECT((2 == a.part.nb_blocks) && (1 == a.part.max_block) && (0 == a.part.max_internal), "edge2-split is not two blocks of one vertex");
        qaoa_instance_free(&a);

        /* rand12neg is rand12 with some signs changed, and both have edges of weight 0. */
        (void)qaoa_instance_named(&a, "rand12");
        (void)qaoa_instance_named(&b, "rand12neg");
        EXPECT(a.graph.m == b.graph.m, "rand12 and rand12neg have %d and %d edges", a.graph.m, b.graph.m);
        for( int e = 0; (e < a.graph.m) && (a.graph.m == b.graph.m); e++ ) {
            if( (a.graph.eu[e] != b.graph.eu[e]) || (a.graph.ev[e] != b.graph.ev[e]) || (fabs(a.graph.ew[e]) != fabs(b.graph.ew[e])) ) same = 0;
            zeros    += (0.0 == a.graph.ew[e]);
            negative += (b.graph.ew[e] < 0.0);
            EXPECT(a.graph.ew[e] >= 0.0, "rand12 has a negative weight");
        }
        EXPECT(same, "rand12neg is not rand12 with other signs");
        EXPECT((zeros > 0) && (zeros < a.graph.m), "rand12 has %d edges of weight 0 among %d", zeros, a.graph.m);
        EXPECT(negative > 0, "rand12neg has no negative weight");
        qaoa_instance_free(&a);
        qaoa_instance_free(&b);
    }

    /* Two blocks whose only edge has weight 0 are not neighbours, and may have the same colour. */
    {
        qaoa_graph_t g;
        qaoa_partition_t p;
        const int eu[] = { 0, 1, 2 }, ev[] = { 1, 2, 3 }, blocks[] = { 0, 0, 1, 1 };
        const double ew[] = { 1.0, 0.0, 1.0 };

        EXPECT(0 == qaoa_graph_from_edges(&g, 4, 3, eu, ev, ew), "a path of four vertices was refused");
        EXPECT(0 == qaoa_partition_from_blocks(&p, &g, 2, blocks), "its partition in two halves was refused");
        EXPECT((0 == p.max_deg) && (1 == p.nb_colours) && (1 == p.max_internal), "an edge of weight 0 made neighbours: %d at most, %d colours",
               p.max_deg, p.nb_colours);
        qaoa_partition_free(&p);
        qaoa_graph_free(&g);
    }

    /* What a graph or a partition may not be. */
    {
        qaoa_graph_t g;
        qaoa_partition_t p;
        const int eu[] = { 0, 1, 0 }, ev[] = { 1, 2, 1 }, loop_u[] = { 0 }, loop_v[] = { 0 }, far_v[] = { 3 };
        const int two[] = { 0, 0, 2 }, empty_block[] = { 0, 0, 0 };
        const double ew[] = { 1.0, 1.0, 1.0 };

        EXPECT(0 != qaoa_graph_from_edges(&g, 3, 3, eu, ev, ew), "an edge given twice was accepted");
        EXPECT(0 != qaoa_graph_from_edges(&g, 3, 1, loop_u, loop_v, ew), "a loop was accepted");
        EXPECT(0 != qaoa_graph_from_edges(&g, 3, 1, loop_u, far_v, ew), "a vertex out of range was accepted");
        EXPECT(0 == qaoa_graph_from_edges(&g, 3, 2, eu, ev, ew), "a path of three vertices was refused");
        EXPECT(0 != qaoa_partition_from_blocks(&p, &g, 2, two), "a block number out of range was accepted");
        EXPECT(0 != qaoa_partition_from_blocks(&p, &g, 2, empty_block), "an empty block was accepted");
        qaoa_graph_free(&g);
        EXPECT(0 != qaoa_graph_random(&g, 4, 7, 1, 0, 0), "seven edges on four vertices");
        EXPECT(0 != qaoa_graph_blocks(&g, 2, 3, 1, 1, 1), "a chord in a triangle");
    }
}

/* ------------------------------------------------------------------------
 * The decomposition, on every assignment
 * --------------------------------------------------------------------- */

static void test_decomposition(void)
{
    qaoa_config_t cfg;
    long checked = 0;

    qaoa_config_default(&cfg);
    for( int i = 0; NULL != qaoa_instance_names[i]; i++ ) {
        const char *name = qaoa_instance_names[i];
        qaoa_instance_t inst;
        const qaoa_graph_t *g = &inst.graph;
        const qaoa_partition_t *p = &inst.part;
        unsigned char *x;
        qaoa_state_t *states;
        uint64_t nb_assignments;

        if( 0 != qaoa_instance_named(&inst, name) ) continue;
        x      = (unsigned char*)calloc((size_t)g->n, 1);
        states = (qaoa_state_t*)calloc((size_t)p->nb_blocks, sizeof(qaoa_state_t));
        /* All of them up to 16 vertices, 3,000 drawn ones beyond. */
        nb_assignments = (g->n <= 16) ? (((uint64_t)1) << g->n) : 3000;

        for( uint64_t a = 0; a < nb_assignments; a++ ) {
            const uint64_t pattern = (g->n <= 16) ? a : rnd();
            double cut;

            for( int v = 0; v < g->n; v++ ) x[v] = (unsigned char)((g->n <= 16) ? ((pattern >> v) & 1) : (parsec_qpu_mix64(pattern + (uint64_t)v) & 1));
            states_from_vertices(&inst, x, states);
            cut = qaoa_graph_cut(g, x);
            EXPECT(qaoa_cut(&inst, states) == cut, "%s: the cut from the states of the blocks is %g, from the vertices %g",
                   name, qaoa_cut(&inst, states), cut);

            for( int b = 0; b < p->nb_blocks; b++ ) {
                qaoa_local_t *P = local_new(&inst, &cfg, b, states);
                const double *h = qaoa_local_h(P);
                double rest = 0.0, spin, constant = 0.0;

                /* The edges that do not touch the block. */
                for( int e = 0; e < g->m; e++ ) {
                    if( (p->block_of[g->eu[e]] != b) && (p->block_of[g->ev[e]] != b) && (x[g->eu[e]] != x[g->ev[e]]) ) rest += g->ew[e];
                }
                EXPECT(qaoa_local_value(P, states[b].bits) + rest == cut,
                       "%s, block %d: local objective %g plus the rest %g is not the cut %g", name, b,
                       qaoa_local_value(P, states[b].bits), rest, cut);
                EXPECT(P->current == qaoa_local_value(P, states[b].bits), "%s, block %d: the current value is not that of the current assignment", name, b);

                /* The spin form, from the definition: the constant and the
                 * field of each vertex from its edges to the outside, then
                 * c + sum w/2 (1 - z z) - sum h z. */
                spin = 0.0;
                for( int k = 0; k < (int)P->n; k++ ) {
                    const int u = qaoa_block_vertex(p, b, k);
                    const double zu = x[u] ? -1.0 : 1.0;
                    double field = 0.0;
                    for( int e = 0; e < g->m; e++ ) {
                        int other;
                        if( g->eu[e] == u ) other = g->ev[e]; else if( g->ev[e] == u ) other = g->eu[e]; else continue;
                        if( p->block_of[other] == b ) {
                            if( other > u ) spin += 0.5 * g->ew[e] * (1.0 - zu * (x[other] ? -1.0 : 1.0));
                        } else {
                            field    += 0.5 * g->ew[e] * (x[other] ? -1.0 : 1.0);
                            constant += 0.5 * g->ew[e];
                        }
                    }
                    EXPECT(h[k] == field, "%s, block %d: the field of vertex %d is %g, the definition gives %g", name, b, u, h[k], field);
                    spin -= field * zu;
                }
                EXPECT(P->constant == constant, "%s, block %d: the constant is %g, the definition gives %g", name, b, P->constant, constant);
                EXPECT(constant + spin == qaoa_local_value(P, states[b].bits), "%s, block %d: the spin form gives %g, the bit form %g",
                       name, b, constant + spin, qaoa_local_value(P, states[b].bits));

                /* Another assignment of the block alone moves the cut by what it moves the local objective. */
                if( 0 == (a & 15) ) {
                    qaoa_state_t saved = states[b];
                    const uint64_t other = rnd() & ((P->n < 64) ? ((((uint64_t)1) << P->n) - 1) : ~(uint64_t)0);
                    const double before = qaoa_local_value(P, states[b].bits), after = qaoa_local_value(P, other);
                    states[b].bits = other;
                    EXPECT(qaoa_cut(&inst, states) == cut + (after - before), "%s, block %d: the block gained %g and the cut moved by %g",
                           name, b, after - before, qaoa_cut(&inst, states) - cut);
                    states[b] = saved;
                }
                free(P);
                checked++;
            }
        }
        free(x);
        free(states);
        qaoa_instance_free(&inst);
    }
    printf("decomposition: %ld local problems checked against the cut of the whole graph\n", checked);
}

/* ------------------------------------------------------------------------
 * Bit order
 * --------------------------------------------------------------------- */

static void test_bit_order(void)
{
    qaoa_instance_t inst;
    qaoa_state_t states[2];
    double seen[64];
    int distinct = 0;

    /* A path 0-1-2-3-4-5 with weights 1/8, 1/4, 1/2, 1, 2, in blocks
     * {0,1,2} and {3,4,5}. Bit i of block b is vertex 3b + i, and the cut
     * is written out here from that alone. */
    if( 0 != qaoa_instance_named(&inst, "path6") ) { EXPECT(0, "path6 could not be built"); return; }
    for( int a = 0; a < 64; a++ ) {
        const int x[6] = { a & 1, (a >> 1) & 1, (a >> 2) & 1, (a >> 3) & 1, (a >> 4) & 1, (a >> 5) & 1 };
        const double direct = 0.125 * (x[0] ^ x[1]) + 0.25 * (x[1] ^ x[2]) + 0.5 * (x[2] ^ x[3]) + 1.0 * (x[3] ^ x[4]) + 2.0 * (x[4] ^ x[5]);
        int is_new = 1;

        memset(states, 0, sizeof(states));
        states[0].bits = (uint64_t)(a & 7);
        states[1].bits = (uint64_t)(a >> 3);
        EXPECT(qaoa_cut(&inst, states) == direct, "path6, assignment %d: cut %g, expected %g", a, qaoa_cut(&inst, states), direct);
        for( int k = 0; k < distinct; k++ ) is_new &= (seen[k] != direct);
        if( is_new ) seen[distinct++] = direct;
    }
    /* Every assignment has its own cut, shared with its complement only:
     * any other reading of the bits gives another number somewhere. */
    EXPECT(32 == distinct, "path6 has %d different cuts, expected 32", distinct);
    qaoa_instance_free(&inst);
}

/* ------------------------------------------------------------------------
 * The circuit against the exact layers
 * --------------------------------------------------------------------- */

/* The state the layers prepare, from the definition: start uniform; per
 * layer multiply amplitude x by exp(-i gamma C_S(x)), then apply
 * exp(-i beta X) to every qubit. */
static double complex *exact_layers(const qaoa_local_t *P, int layers, const double *theta)
{
    const uint64_t dim = ((uint64_t)1) << P->n;
    double complex *psi = (double complex*)malloc(dim * sizeof(double complex));

    for( uint64_t x = 0; x < dim; x++ ) psi[x] = 1.0 / sqrt((double)dim);
    for( int l = 0; l < layers; l++ ) {
        const double gamma = theta[2 * l], beta = theta[2 * l + 1];
        for( uint64_t x = 0; x < dim; x++ ) psi[x] *= cexp(-I * gamma * qaoa_local_value(P, x));
        for( uint32_t q = 0; q < P->n; q++ ) {
            for( uint64_t x = 0; x < dim; x++ ) {
                if( x & (((uint64_t)1) << q) ) continue;
                const double complex a = psi[x], b = psi[x | (((uint64_t)1) << q)];
                psi[x]                         = cos(beta) * a - I * sin(beta) * b;
                psi[x | (((uint64_t)1) << q)]  = cos(beta) * b - I * sin(beta) * a;
            }
        }
    }
    return psi;
}

static void test_circuit(void)
{
    static const char *names[] = { "tri", "c4", "c4-cross", "rand12", "rand12neg", "rand16", "ring64", NULL };
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();
    qaoa_config_t cfg;
    double worst = 0.0;
    int compared = 0;

    qaoa_config_default(&cfg);
    for( int i = 0; NULL != names[i]; i++ ) {
        qaoa_instance_t inst;
        const qaoa_partition_t *p;
        qaoa_state_t *states;
        unsigned char *x;

        if( 0 != qaoa_instance_named(&inst, names[i]) ) { EXPECT(0, "%s could not be built", names[i]); continue; }
        p = &inst.part;
        x = (unsigned char*)calloc((size_t)inst.graph.n, 1);
        states = (qaoa_state_t*)calloc((size_t)p->nb_blocks, sizeof(qaoa_state_t));
        for( int v = 0; v < inst.graph.n; v++ ) x[v] = (unsigned char)(rnd() & 1);
        states_from_vertices(&inst, x, states);

        for( int b = 0; b < p->nb_blocks; b++ ) {
            for( int layers = 1; layers <= 3; layers++ ) {
                qaoa_local_t *P = local_new(&inst, &cfg, b, states);
                const double *h = qaoa_local_h(P);
                const size_t room = qaoa_circuit_size((int)P->n, (int)P->m, layers);
                parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)malloc(room);
                double theta[2 * QAOA_MAX_LAYERS];
                const double *amp = NULL;
                double complex *psi, overlap = 0.0;
                const char *why = NULL;
                uint32_t fields = 0;

                cfg.layers = layers;
                for( int j = 0; j < 2 * layers; j++ ) theta[j] = rnd_angle();
                EXPECT(0 == qaoa_circuit_build(circuit, room, P, &cfg), "%s, block %d: the circuit does not fit in the size it is said to need", names[i], b);
                EXPECT(PARSEC_SUCCESS == parsec_qpu_circuit_check(circuit, room, &why), "%s, block %d: the circuit is not well-formed: %s", names[i], b, why);
                /* One gate per term: an edge of weight 0 and a vertex with no field have none. */
                for( uint32_t q = 0; q < P->n; q++ ) fields += (0.0 != h[q]);
                EXPECT(circuit->nb_ops == 2 * P->n + (uint32_t)layers * (P->m + fields + P->n),
                       "%s, block %d: %u operations for %u qubits, %u edges, %u fields and %d layers", names[i], b,
                       circuit->nb_ops, P->n, P->m, fields, layers);
                EXPECT(circuit->nb_params == (uint32_t)(2 * layers), "%u parameter slots for %d layers", circuit->nb_params, layers);
                /* One byte less is too little; then build it whole again. */
                EXPECT(0 != qaoa_circuit_build(circuit, parsec_qpu_circuit_size(circuit) - 1, P, &cfg), "the circuit fit in one byte less than it has");
                EXPECT(0 == qaoa_circuit_build(circuit, room, P, &cfg), "the circuit could not be built a second time");

                if( PARSEC_SUCCESS == parsec_qpu_sampler_state(sampler, circuit, theta, NULL, &amp) ) {
                    psi = exact_layers(P, layers, theta);
                    /* Equal up to a phase: the overlap has modulus one, and so the probabilities agree too. */
                    for( uint64_t y = 0; y < (((uint64_t)1) << P->n); y++ ) overlap += conj(psi[y]) * (amp[2 * y] + I * amp[2 * y + 1]);
                    EXPECT(fabs(cabs(overlap) - 1.0) < TOLERANCE, "%s, block %d, %d layers: the circuit's state has overlap %.15f with the exact layers",
                           names[i], b, layers, cabs(overlap));
                    if( fabs(cabs(overlap) - 1.0) > worst ) worst = fabs(cabs(overlap) - 1.0);
                    compared++;
                    free(psi);
                } else {
                    EXPECT(0, "%s, block %d: the sampler refused the circuit", names[i], b);
                }
                free(circuit);
                free(P);
            }
        }
        cfg.layers = 1;
        free(x);
        free(states);
        qaoa_instance_free(&inst);
    }
    printf("circuit against the exact layers: %d states, overlap off one by at most %.2e\n", compared, worst);

    /* An edge of weight 0 is no term and no gate. One block of three
     * vertices with edges 0-1 of weight 1, 1-2 of weight 0 and 0-2 of weight
     * 1/2: two terms, two RZZ per layer, and the values of a graph without
     * the edge 1-2. The fixed instances are counted the same way, from
     * their edges. */
    {
        qaoa_instance_t inst;
        qaoa_state_t state;
        const int eu[] = { 0, 1, 0 }, ev[] = { 1, 2, 2 }, blocks[] = { 0, 0, 0 };
        const double ew[] = { 1.0, 0.0, 0.5 };
        parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)malloc(qaoa_circuit_size(3, 3, 2));
        qaoa_local_t *P;
        uint32_t rzz = 0;

        memset(&inst, 0, sizeof(inst));
        EXPECT(0 == qaoa_graph_from_edges(&inst.graph, 3, 3, eu, ev, ew), "a triangle with an edge of weight 0 was refused");
        EXPECT(0 == qaoa_partition_from_blocks(&inst.part, &inst.graph, 1, blocks), "its partition in one block was refused");
        EXPECT(2 == inst.part.max_internal, "the partition counts %d edges inside, expected the 2 of non-zero weight", inst.part.max_internal);
        memset(&state, 0, sizeof(state));
        P = local_new(&inst, &cfg, 0, &state);
        EXPECT(2 == P->m, "the local problem has %u edges, expected 2", P->m);
        cfg.layers = 2;
        EXPECT(0 == qaoa_circuit_build(circuit, qaoa_circuit_size(3, 2, 2), P, &cfg), "the circuit does not fit in the room of two edges");
        for( uint32_t k = 0; k < circuit->nb_ops; k++ ) rzz += (PARSEC_QPU_OP_RZZ == parsec_qpu_circuit_ops(circuit)[k].opcode);
        EXPECT(4 == rzz, "%u RZZ gates for two edges and two layers", rzz);
        cfg.layers = 1;
        for( uint64_t y = 0; y < 8; y++ ) {
            const double expected = 1.0 * ((y ^ (y >> 1)) & 1) + 0.5 * ((y ^ (y >> 2)) & 1);
            EXPECT(qaoa_local_value(P, y) == expected, "assignment %llu has value %g, expected %g", (unsigned long long)y, qaoa_local_value(P, y), expected);
        }
        free(P);
        free(circuit);
        qaoa_instance_free(&inst);
    }
    for( int i = 0; NULL != qaoa_instance_names[i]; i++ ) {
        qaoa_instance_t inst;
        qaoa_state_t *states;

        if( 0 != qaoa_instance_named(&inst, qaoa_instance_names[i]) ) continue;
        states = (qaoa_state_t*)calloc((size_t)inst.part.nb_blocks, sizeof(qaoa_state_t));
        for( int b = 0; b < inst.part.nb_blocks; b++ ) {
            qaoa_local_t *P = local_new(&inst, &cfg, b, states);
            uint32_t inside = 0;
            for( int e = 0; e < inst.graph.m; e++ ) {
                inside += ((0.0 != inst.graph.ew[e]) && (inst.part.block_of[inst.graph.eu[e]] == b) && (inst.part.block_of[inst.graph.ev[e]] == b));
            }
            EXPECT(P->m == inside, "%s, block %d: the local problem has %u edges, the graph has %u of non-zero weight inside the block",
                   qaoa_instance_names[i], b, P->m, inside);
            free(P);
        }
        free(states);
        qaoa_instance_free(&inst);
    }

    /* An update that evaluates nothing is an empty job. */
    {
        qaoa_instance_t inst;
        qaoa_state_t states[2];
        parsec_qpu_circuit_t circuit;
        qaoa_local_t *P;

        (void)qaoa_instance_named(&inst, "tri");
        memset(states, 0, sizeof(states));
        P = local_new(&inst, &cfg, 0, states);
        P->skip = 1;
        EXPECT(0 == qaoa_circuit_build(&circuit, sizeof(circuit), P, &cfg), "an empty job needs more than a header");
        EXPECT((circuit.flags & PARSEC_QPU_CIRCUIT_EMPTY) && (0 == circuit.nb_ops), "the circuit of a skipped update is not an empty job");
        free(P);
        qaoa_instance_free(&inst);
    }
    parsec_qpu_sampler_free(sampler);
}

/* ------------------------------------------------------------------------
 * Closed forms
 * --------------------------------------------------------------------- */

static void test_closed_forms(void)
{
    qaoa_config_t cfg;
    qaoa_instance_t inst;
    qaoa_state_t states[4];
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();

    qaoa_config_default(&cfg);

    /* One edge of weight w in one block: (w / 2) (1 + sin 4 beta sin gamma w). */
    (void)qaoa_instance_named(&inst, "edge2");
    memset(states, 0, sizeof(states));
    for( int trial = 0; trial < 20; trial++ ) {
        qaoa_local_t *P = local_new(&inst, &cfg, 0, states);
        const size_t room = qaoa_circuit_size(2, 1, 1);
        parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)malloc(room);
        const double theta[2] = { rnd_angle(), rnd_angle() }, w = inst.graph.ew[0];
        double mean = 0.0, variance = 0.0;

        (void)qaoa_circuit_build(circuit, room, P, &cfg);
        EXPECT(0 == qaoa_exact_expectation(P, circuit, theta, &mean, &variance), "the exact expectation could not be had");
        EXPECT(fabs(mean - 0.5 * w * (1 + sin(4 * theta[1]) * sin(theta[0] * w))) < TOLERANCE,
               "one edge at gamma=%g beta=%g: expected cut %.15f, closed form %.15f", theta[0], theta[1], mean,
               0.5 * w * (1 + sin(4 * theta[1]) * sin(theta[0] * w)));
        EXPECT(variance >= -TOLERANCE, "a negative variance");
        free(circuit);
        free(P);
    }
    qaoa_instance_free(&inst);

    /* The same edge with its ends in two blocks: one vertex in the field
     * h = +-w/2 of the other. Its expected local objective is
     * c + h sin(2 gamma h) sin(2 beta). */
    (void)qaoa_instance_named(&inst, "edge2-split");
    for( int other = 0; other < 2; other++ ) {
        for( int trial = 0; trial < 10; trial++ ) {
            const double theta[2] = { rnd_angle(), rnd_angle() }, w = inst.graph.ew[0];
            const double field = other ? -0.5 * w : 0.5 * w;
            const size_t room = qaoa_circuit_size(1, 0, 1);
            parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)malloc(room);
            qaoa_local_t *P;
            double mean = 0.0;

            memset(states, 0, sizeof(states));
            states[1].bits = (uint64_t)other;
            P = local_new(&inst, &cfg, 0, states);
            EXPECT((0 == P->m) && (qaoa_local_h(P)[0] == field) && (P->constant == 0.5 * w), "a block of one vertex: %u edges, field %g, constant %g",
                   P->m, qaoa_local_h(P)[0], P->constant);
            (void)qaoa_circuit_build(circuit, room, P, &cfg);
            (void)qaoa_exact_expectation(P, circuit, theta, &mean, NULL);
            EXPECT(fabs(mean - (0.5 * w + field * sin(2 * theta[0] * field) * sin(2 * theta[1]))) < TOLERANCE,
                   "one vertex in a field %g: expected %.15f, closed form %.15f", field, mean,
                   0.5 * w + field * sin(2 * theta[0] * field) * sin(2 * theta[1]));
            free(circuit);
            free(P);
        }
    }
    qaoa_instance_free(&inst);

    /* Qubit i carries the field of vertex i. In c4-cross every edge leaves
     * the block, so its two qubits are independent, each in its own field,
     * and the two fields differ: the chance of measuring 1 on qubit i is
     * (1 + sin(2 gamma h_i) sin(2 beta)) / 2. */
    (void)qaoa_instance_named(&inst, "c4-cross");
    for( int trial = 0; trial < 10; trial++ ) {
        const double theta[2] = { rnd_angle(), rnd_angle() };
        const size_t room = qaoa_circuit_size(2, 0, 1);
        parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)malloc(room);
        const double *amp = NULL, *h;
        qaoa_local_t *P;

        memset(states, 0, sizeof(states));
        states[1].bits = (uint64_t)(1 + trial % 2);     /* 01 or 10: the two fields are not opposite */
        P = local_new(&inst, &cfg, 0, states);
        h = qaoa_local_h(P);
        EXPECT((0 == P->m) && (fabs(h[0]) != fabs(h[1])), "c4-cross: %u edges inside, fields %g and %g", P->m, h[0], h[1]);
        (void)qaoa_circuit_build(circuit, room, P, &cfg);
        if( PARSEC_SUCCESS == parsec_qpu_sampler_state(sampler, circuit, theta, NULL, &amp) ) {
            for( int q = 0; q < 2; q++ ) {
                double one = 0.0;
                for( int y = 0; y < 4; y++ ) if( (y >> q) & 1 ) one += amp[2 * y] * amp[2 * y] + amp[2 * y + 1] * amp[2 * y + 1];
                EXPECT(fabs(one - 0.5 * (1 + sin(2 * theta[0] * h[q]) * sin(2 * theta[1]))) < TOLERANCE,
                       "c4-cross: qubit %d is 1 with probability %.15f, its field %g gives %.15f", q, one, h[q],
                       0.5 * (1 + sin(2 * theta[0] * h[q]) * sin(2 * theta[1])));
            }
        } else {
            EXPECT(0, "the sampler refused the circuit");
        }
        free(circuit);
        free(P);
    }
    qaoa_instance_free(&inst);
    parsec_qpu_sampler_free(sampler);
}

/* ------------------------------------------------------------------------
 * Evaluation, optimizer, accept rule
 * --------------------------------------------------------------------- */

static void test_eval_and_optimizer(void)
{
    qaoa_config_t cfg;
    qaoa_instance_t inst;
    qaoa_state_t states[2], x;
    qaoa_local_t *P;
    qaoa_opt_t S;
    qaoa_eval_t F, plus, minus;
    union { parsec_qpu_result_t header; unsigned char bytes[256]; } res;
    double tp[2 * QAOA_MAX_LAYERS], tm[2 * QAOA_MAX_LAYERS];

    qaoa_config_default(&cfg);
    (void)qaoa_instance_named(&inst, "tri");
    memset(states, 0, sizeof(states));
    states[1].bits = 1;
    P = local_new(&inst, &cfg, 0, states);       /* vertices 0 and 1; vertex 2 at 1 */

    /* Counts: 00 six times, 01 once, 10 three times. */
    memset(&res, 0, sizeof(res));
    res.header.status = PARSEC_QPU_STATUS_OK;
    res.header.shots_returned = 10;
    res.header.nb_entries = 3;
    parsec_qpu_result_set_entry(&res.header, 0, 0, 6);
    parsec_qpu_result_set_entry(&res.header, 1, 1, 1);
    parsec_qpu_result_set_entry(&res.header, 2, 2, 3);
    qaoa_eval(&F, P, &res.header);
    {
        const double v0 = qaoa_local_value(P, 0), v1 = qaoa_local_value(P, 1), v2 = qaoa_local_value(P, 2);
        const double best = (v0 >= v1 && v0 >= v2) ? v0 : ((v1 >= v2) ? v1 : v2);
        EXPECT(F.valid && (10 == F.shots), "a good result was not taken");
        EXPECT(F.mean == (6 * v0 + 1 * v1 + 3 * v2) / 10.0, "the estimate is %g, expected %g", F.mean, (6 * v0 + v1 + 3 * v2) / 10.0);
        EXPECT(F.best_value == best, "the best measured value is %g, expected %g", F.best_value, best);
        EXPECT(qaoa_local_value(P, F.best_bits) == best, "the best bitstring does not have the best value");
    }
    /* Equal values: the smallest bitstring is kept. Vertex 2 at 0 and no
     * field difference would be needed for a tie; use two entries of the
     * same bitstring value by construction instead. */
    {
        qaoa_local_t *flat = local_new(&inst, &cfg, 1, states);    /* the block of vertex 2 alone */
        double *h = qaoa_local_h(flat);
        h[0] = 0.0;                                               /* both assignments now have the same value */
        memset(&res, 0, sizeof(res));
        res.header.status = PARSEC_QPU_STATUS_OK;
        res.header.shots_returned = 4;
        res.header.nb_entries = 2;
        parsec_qpu_result_set_entry(&res.header, 0, 0, 1);
        parsec_qpu_result_set_entry(&res.header, 1, 1, 3);
        qaoa_eval(&F, flat, &res.header);
        EXPECT(F.valid && (0 == F.best_bits), "of two bitstrings of equal value, %llu was kept", (unsigned long long)F.best_bits);
        free(flat);
    }
    /* A job that failed, and an empty one, say nothing. */
    res.header.status = PARSEC_QPU_STATUS_FAILED;
    qaoa_eval(&F, P, &res.header);
    EXPECT(!F.valid, "a failed job was taken for an evaluation");
    memset(&res, 0, sizeof(res));
    qaoa_eval(&F, P, &res.header);
    EXPECT(!F.valid, "an empty job was taken for an evaluation");

    /* The optimizer. First angles: gamma0 over the largest coefficient, beta0. */
    memset(&x, 0, sizeof(x));
    qaoa_opt_init(&S, &cfg, &x, P);
    EXPECT((S.theta[0] == cfg.gamma0 / P->largest) && (S.theta[1] == cfg.beta0), "first angles %g and %g", S.theta[0], S.theta[1]);
    EXPECT(S.weight == P->weight, "the optimizer's scale is not that of the problem");
    x.have_theta = 1; x.theta[0] = 0.7; x.theta[1] = -0.2;
    qaoa_opt_init(&S, &cfg, &x, P);
    EXPECT((0.7 == S.theta[0]) && (-0.2 == S.theta[1]), "the block's own angles were not taken");

    /* A step: the two points are the iterate plus and minus c_t times a
     * vector of +-1, and the move is a_t (f+ - f-) / (2 c_t W) along it. */
    for( int t = 0; t < 5; t++ ) {
        const double before[2] = { S.theta[0], S.theta[1] };
        const double ck = cfg.spsa_c / pow(t + 1.0, cfg.spsa_gamma), ak = cfg.spsa_a / pow(t + 1.0 + cfg.spsa_A, cfg.spsa_alpha);

        qaoa_opt_propose(&S, &cfg, tp, tm);
        EXPECT(fabs(S.ck - ck) < TOLERANCE, "step %d: c is %g, expected %g", t, S.ck, ck);
        for( int j = 0; j < 2; j++ ) {
            EXPECT((1.0 == S.delta[j]) || (-1.0 == S.delta[j]), "step %d: a perturbation of %g", t, S.delta[j]);
            EXPECT((tp[j] == before[j] + S.ck * S.delta[j]) && (tm[j] == before[j] - S.ck * S.delta[j]), "step %d: the two points are not the iterate plus and minus the perturbation", t);
        }
        memset(&plus, 0, sizeof(plus)); memset(&minus, 0, sizeof(minus));
        plus.valid = minus.valid = 1;
        plus.mean = 1.0 + 0.1 * t; minus.mean = 0.5;
        plus.best_value = 2.0 + t; plus.best_bits = 3;
        minus.best_value = 2.0 + t; minus.best_bits = 1;        /* the same value, a smaller bitstring */
        qaoa_opt_absorb(&S, &cfg, &plus, &minus);
        for( int j = 0; j < 2; j++ ) {
            const double expected = before[j] + ak * (plus.mean - minus.mean) / (2.0 * ck * P->weight) * S.delta[j];
            EXPECT(fabs(S.theta[j] - expected) < TOLERANCE, "step %d: angle %d moved to %.15f, expected %.15f", t, j, S.theta[j], expected);
        }
        EXPECT(S.t == (uint32_t)(t + 1), "the step count is %u after %d steps", S.t, t + 1);
        EXPECT(S.have_best && (S.best_value == 2.0 + t) && (1 == S.best_bits), "step %d: the best candidate is %llu with %g", t,
               (unsigned long long)S.best_bits, S.best_value);
        /* The better of the two points so far is the plus one of this step. */
        EXPECT(S.have_mean && (S.best_mean == plus.mean) && (S.best_theta[0] == tp[0]) && (S.best_theta[1] == tp[1]),
               "step %d: the best angles are not those of the best estimate", t);
    }
    EXPECT((10 == S.evaluations) && (0 == S.failed), "%u evaluations and %u failures after five good steps", S.evaluations, S.failed);

    /* A step with one evaluation missing: no move, and the one that is there still gives its candidate. */
    {
        const double before[2] = { S.theta[0], S.theta[1] };
        qaoa_opt_propose(&S, &cfg, tp, tm);
        memset(&plus, 0, sizeof(plus)); memset(&minus, 0, sizeof(minus));
        minus.valid = 1; minus.mean = 0.1; minus.best_value = 100.0; minus.best_bits = 2;
        qaoa_opt_absorb(&S, &cfg, &plus, &minus);
        EXPECT((S.theta[0] == before[0]) && (S.theta[1] == before[1]), "the iterate moved on one evaluation");
        EXPECT((100.0 == S.best_value) && (2 == S.best_bits), "the candidate of the evaluation that was there was dropped");
        EXPECT((11 == S.evaluations) && (1 == S.failed), "%u evaluations and %u failures", S.evaluations, S.failed);
    }

    /* The perturbation depends on the seed, the round, the block and the step, and on nothing else. */
    {
        qaoa_opt_t A, B;
        unsigned patterns_a = 0, patterns_b = 0;
        int differ = 0;

        cfg.layers = 3;
        for( int t = 0; t < 64; t++ ) {
            memset(&A, 0, sizeof(A)); memset(&B, 0, sizeof(B));
            A.round = B.round = 2; A.block = B.block = 5; A.t = B.t = (uint32_t)t;
            A.theta[0] = 9.0;                 /* the iterate does not enter */
            qaoa_opt_propose(&A, &cfg, tp, tm);
            qaoa_opt_propose(&B, &cfg, tp, tm);
            EXPECT(0 == memcmp(A.delta, B.delta, sizeof(A.delta)), "step %d: the same seed, round, block and step gave two perturbations", t);
            for( int j = 0; j < 6; j++ ) patterns_a = (patterns_a * 3 + (A.delta[j] > 0)) % 1000003;
            B.block = 6;
            qaoa_opt_propose(&B, &cfg, tp, tm);
            differ += (0 != memcmp(A.delta, B.delta, sizeof(A.delta)));
            cfg.seed++;
            B.block = 5;
            qaoa_opt_propose(&B, &cfg, tp, tm);
            for( int j = 0; j < 6; j++ ) patterns_b = (patterns_b * 3 + (B.delta[j] > 0)) % 1000003;
            cfg.seed--;
        }
        EXPECT(differ > 32, "another block had the same perturbation in %d of 64 steps", 64 - differ);
        EXPECT(patterns_a != patterns_b, "another seed gave the same perturbations");
        cfg.layers = 1;
    }

    /* The accept rule: strictly better, or the assignment stays. */
    {
        const double current = P->current;

        memset(&x, 0, sizeof(x));
        x.bits = states[0].bits; x.version = 7; x.stale = 2;
        memset(&S, 0, sizeof(S));
        S.have_best = 1; S.best_bits = 3; S.best_value = current;        /* as good: not enough */
        EXPECT(0.0 == qaoa_state_update(&x, P, &S), "an assignment that is only as good was accepted");
        EXPECT((x.bits == states[0].bits) && (7 == x.version) && (3 == x.stale) && (1 == x.updates), "the state after a refusal: bits %llu, version %u, stale %u",
               (unsigned long long)x.bits, x.version, x.stale);
        S.best_value = current - 0.125;                                  /* worse */
        EXPECT(0.0 == qaoa_state_update(&x, P, &S), "a worse assignment was accepted");
        S.best_value = current + 0.125;                                  /* better by the smallest step */
        EXPECT(0.125 == qaoa_state_update(&x, P, &S), "a better assignment was refused");
        EXPECT((3 == x.bits) && (8 == x.version) && (0 == x.stale) && (0.125 == x.improvement), "the state after an acceptance: bits %llu, version %u, stale %u",
               (unsigned long long)x.bits, x.version, x.stale);
        S.have_best = 0; S.best_value = current + 100.0;                 /* no candidate at all */
        EXPECT(0.0 == qaoa_state_update(&x, P, &S), "an update without a candidate changed the assignment");
        /* A skipped update changes nothing but the counts. */
        P->skip = 1;
        S.have_best = 1;
        EXPECT((0.0 == qaoa_state_update(&x, P, &S)) && (3 == x.bits) && (1 == x.skipped), "a skipped update changed the assignment");
        P->skip = 0;
        EXPECT(5 == x.updates, "%u updates counted, expected 5", x.updates);
    }

    /* When a block is skipped: only with patience, after an update, with
     * enough updates without gain, and with every neighbour unchanged. */
    {
        qaoa_boundary_t *h = (qaoa_boundary_t*)malloc(qaoa_boundary_size(inst.part.max_block));
        qaoa_local_t *Q = (qaoa_local_t*)malloc(qaoa_local_size(inst.part.max_block, inst.part.max_internal));
        qaoa_state_t self, nbr;

        memset(&self, 0, sizeof(self)); memset(&nbr, 0, sizeof(nbr));
        nbr.version = 4;
        qaoa_boundary_init(h, &inst, 0);
        qaoa_boundary_gather(h, &inst, 0, 0, &nbr);
        EXPECT((1 == h->nb) && (4 == h->nbr_version[0]), "the boundary did not note the version it read");

        cfg.patience = 2;
        self.updates = 3; self.stale = 2; self.nbr_version[0] = 4;
        qaoa_local_build(Q, &inst, &cfg, 1, 0, &self, h);
        EXPECT(Q->skip, "a block that is stale, with unchanged neighbours, is evaluated again");
        self.stale = 1;
        qaoa_local_build(Q, &inst, &cfg, 1, 0, &self, h);
        EXPECT(!Q->skip, "a block is skipped before it has been patient enough");
        self.stale = 2; self.nbr_version[0] = 3;
        qaoa_local_build(Q, &inst, &cfg, 1, 0, &self, h);
        EXPECT(!Q->skip, "a block is skipped although a neighbour changed");
        self.nbr_version[0] = 4; self.updates = 0;
        qaoa_local_build(Q, &inst, &cfg, 1, 0, &self, h);
        EXPECT(!Q->skip, "a block is skipped before its first update");
        self.updates = 3; cfg.patience = 0;
        qaoa_local_build(Q, &inst, &cfg, 1, 0, &self, h);
        EXPECT(!Q->skip, "a block is skipped although skipping is off");
        EXPECT((1 == Q->round) && (0 == Q->block) && (4 == Q->nbr_version[0]), "the local problem does not say what it was built from");
        free(h);
        free(Q);
    }

    /* Keys: one per evaluation, none of them the one that means "choose for me". */
    {
        uint64_t last = 0;
        int distinct = 1;
        for( int r = 0; r < 3; r++ ) for( int b = 0; b < 5; b++ ) for( int t = 0; t < 4; t++ ) for( int e = 0; e < 2; e++ ) {
            const uint64_t key = qaoa_key(r, b, t, e);
            EXPECT(PARSEC_QPU_KEY_AUTO != key, "a key is the reserved one");
            if( (r + b + t + e > 0) && (key <= last) ) distinct = 0;
            last = key;
        }
        EXPECT(distinct, "the keys are not increasing in (round, block, step, evaluation)");
    }

    free(P);
    qaoa_instance_free(&inst);
}

/* ------------------------------------------------------------------------
 * Enumerations against plain loops
 * --------------------------------------------------------------------- */

static void test_exact(void)
{
    static const char *names[] = { "edge2", "tri", "c4", "path6", "rand12", "rand12neg", "rand16", "disc16", NULL };
    qaoa_config_t cfg;

    qaoa_config_default(&cfg);
    for( int i = 0; NULL != names[i]; i++ ) {
        qaoa_instance_t inst;
        const qaoa_graph_t *g;
        unsigned char *x, *best;
        qaoa_state_t *states;
        double plain = 0.0, exact;

        if( 0 != qaoa_instance_named(&inst, names[i]) ) { EXPECT(0, "%s could not be built", names[i]); continue; }
        g = &inst.graph;
        x = (unsigned char*)calloc((size_t)g->n, 1);
        best = (unsigned char*)calloc((size_t)g->n, 1);
        states = (qaoa_state_t*)calloc((size_t)inst.part.nb_blocks, sizeof(qaoa_state_t));

        for( uint64_t a = 0; a < (((uint64_t)1) << g->n); a++ ) {
            double cut;
            for( int v = 0; v < g->n; v++ ) x[v] = (unsigned char)((a >> v) & 1);
            cut = qaoa_graph_cut(g, x);
            if( cut > plain ) plain = cut;
        }
        exact = qaoa_exact_maxcut(g, best);
        EXPECT(exact == plain, "%s: the largest cut is %g by enumeration and %g by a plain loop", names[i], exact, plain);
        EXPECT(qaoa_graph_cut(g, best) == exact, "%s: the assignment that is returned has cut %g, not %g", names[i], qaoa_graph_cut(g, best), exact);

        /* The best assignment of each block, the others as in `best`. */
        states_from_vertices(&inst, best, states);
        for( int b = 0; b < inst.part.nb_blocks; b++ ) {
            qaoa_local_t *P = local_new(&inst, &cfg, b, states);
            uint64_t bits = 0, plain_bits = 0;
            double plain_local = qaoa_local_value(P, 0), local;

            for( uint64_t y = 1; y < (((uint64_t)1) << P->n); y++ ) {
                if( qaoa_local_value(P, y) > plain_local ) { plain_local = qaoa_local_value(P, y); plain_bits = y; }
            }
            local = qaoa_exact_local(P, &bits);
            EXPECT((local == plain_local) && (bits == plain_bits), "%s, block %d: best local value %g at %llu, a plain loop gives %g at %llu",
                   names[i], b, local, (unsigned long long)bits, plain_local, (unsigned long long)plain_bits);
            /* No block can do better on its own than it does in a best assignment of the whole graph. */
            EXPECT(local == P->current, "%s, block %d: a block of a best assignment could still gain %g", names[i], b, local - P->current);
            free(P);
        }
        free(x);
        free(best);
        free(states);
        qaoa_instance_free(&inst);
    }

    /* Known answers. */
    {
        qaoa_instance_t inst;
        (void)qaoa_instance_named(&inst, "tri");
        EXPECT(5.0 == qaoa_exact_maxcut(&inst.graph, NULL), "a triangle with weights 1, 2, 3 has largest cut %g, expected 5", qaoa_exact_maxcut(&inst.graph, NULL));
        qaoa_instance_free(&inst);
        (void)qaoa_instance_named(&inst, "c4");
        EXPECT(3.5 == qaoa_exact_maxcut(&inst.graph, NULL), "the four-cycle has largest cut %g, expected all of 3.5", qaoa_exact_maxcut(&inst.graph, NULL));
        qaoa_instance_free(&inst);
        (void)qaoa_instance_named(&inst, "ring64");
        EXPECT(qaoa_exact_maxcut(&inst.graph, NULL) < 0.0, "64 vertices were enumerated");
        qaoa_instance_free(&inst);
    }
}

/* ------------------------------------------------------------------------
 * The estimate of a job against the exact expectation
 * --------------------------------------------------------------------- */

static void test_statistics(void)
{
    static const char *names[] = { "rand12", "rand12neg", "ring64", NULL };
    qaoa_config_t cfg;
    double worst = 0.0;
    int jobs = 0;

    qaoa_config_default(&cfg);
    cfg.shots = 4096;
    for( int i = 0; NULL != names[i]; i++ ) {
        qaoa_instance_t inst;
        qaoa_state_t *states;
        qaoa_direct_t direct = { NULL, 77 };

        if( 0 != qaoa_instance_named(&inst, names[i]) ) { EXPECT(0, "%s could not be built", names[i]); continue; }
        states = (qaoa_state_t*)calloc((size_t)inst.part.nb_blocks, sizeof(qaoa_state_t));
        for( int b = 0; b < inst.part.nb_blocks; b++ ) qaoa_state_init(&states[b], &inst, &cfg, b);

        for( int b = 0; b < inst.part.nb_blocks; b++ ) {
            qaoa_local_t *P = local_new(&inst, &cfg, b, states);
            const size_t room = qaoa_circuit_size((int)P->n, (int)P->m, cfg.layers);
            const uint32_t max_entries = qaoa_result_entries(&cfg, (int)P->n);
            parsec_qpu_circuit_t *circuit = (parsec_qpu_circuit_t*)malloc(room);
            parsec_qpu_result_t *result = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(max_entries));
            const double theta[2] = { 0.37 / P->largest, 0.41 };
            double mean = 0.0, variance = 0.0;

            (void)qaoa_circuit_build(circuit, room, P, &cfg);
            (void)qaoa_exact_expectation(P, circuit, theta, &mean, &variance);
            for( int k = 0; k < 5; k++ ) {
                qaoa_eval_t F;
                double z;
                memset(result, 0, sizeof(*result));
                EXPECT(0 == qaoa_sample_direct(circuit, theta, (uint32_t)cfg.shots, qaoa_key(0, b, k, 0), result, max_entries, &direct), "the sampler failed");
                qaoa_eval(&F, P, result);
                EXPECT(F.valid && ((uint32_t)cfg.shots == F.shots), "a job gave %u shots of %d", F.shots, cfg.shots);
                /* Six standard errors; the small term is for a circuit whose outcomes all have one value. */
                z = fabs(F.mean - mean) / (sqrt(variance / cfg.shots) + 1e-9);
                EXPECT(z < 6.0, "%s, block %d: estimate %.6f, expectation %.6f, %.1f standard errors away", names[i], b, F.mean, mean, z);
                if( (variance > 1e-9) && (z > worst) ) worst = z;
                /* What was measured cannot beat the best there is. */
                EXPECT(F.best_value <= qaoa_exact_local(P, NULL), "a measured bitstring has value %g, above the best of the block", F.best_value);
                jobs++;
            }
            free(circuit);
            free(result);
            free(P);
        }
        parsec_qpu_sampler_free(direct.sampler);
        free(states);
        qaoa_instance_free(&inst);
    }
    printf("statistics: %d jobs of %d shots, every estimate within %.2f standard errors of its expectation\n", jobs, cfg.shots, worst);
}

int main(void)
{
    test_structure();
    test_decomposition();
    test_bit_order();
    test_circuit();
    test_closed_forms();
    test_eval_and_optimizer();
    test_exact();
    test_statistics();

    printf("qaoa_math_test: %s\n", nb_errors ? "FAILED" : "passed");
    return nb_errors ? 1 : 0;
}
