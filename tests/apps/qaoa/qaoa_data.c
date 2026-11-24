/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "qaoa_data.h"
#include "parsec/parsec_internal.h"
#include "parsec/datatype.h"
#include "parsec/data_internal.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * Tiles
 * --------------------------------------------------------------------- */

typedef struct qaoa_tiles_s {
    parsec_data_collection_t super;
    int             owner;          /* negative: tile k on rank k mod nodes */
    int             n;
    size_t          tile_bytes;
    parsec_data_t **data;           /* [n] created on first use */
    unsigned char **memory;         /* [n] NULL for a tile another rank holds */
} qaoa_tiles_t;

static uint32_t tiles_rank(const qaoa_tiles_t *t, int k)
{
    if( t->owner >= 0 ) return (uint32_t)t->owner;
    if( k < 0 ) k = -k;
    return (uint32_t)(k % (int)t->super.nodes);
}

static uint32_t tiles_rank_of(parsec_data_collection_t *dc, ...)
{
    va_list ap;
    int k;

    va_start(ap, dc);
    k = va_arg(ap, int);
    va_end(ap);
    return tiles_rank((qaoa_tiles_t*)dc, k);
}

static uint32_t tiles_rank_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    return tiles_rank((qaoa_tiles_t*)dc, (int)key);
}

static int32_t tiles_vpid_of(parsec_data_collection_t *dc, ...)
{
    (void)dc;
    return 0;
}

static int32_t tiles_vpid_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    (void)dc; (void)key;
    return 0;
}

static parsec_data_key_t tiles_data_key(parsec_data_collection_t *dc, ...)
{
    va_list ap;
    int k;

    (void)dc;
    va_start(ap, dc);
    k = va_arg(ap, int);
    va_end(ap);
    return (parsec_data_key_t)k;
}

static parsec_data_t *tiles_data_of_key(parsec_data_collection_t *dc, parsec_data_key_t key)
{
    qaoa_tiles_t *t = (qaoa_tiles_t*)dc;

    /* A collection that only places tasks has nothing to give, and neither
     * has a rank for a tile it does not hold. */
    if( (0 == t->tile_bytes) || ((int)key >= t->n) || (NULL == t->memory[key]) ) return NULL;
    return parsec_data_create(&t->data[key], dc, key, t->memory[key], t->tile_bytes,
                              PARSEC_DATA_FLAG_PARSEC_MANAGED);
}

static parsec_data_t *tiles_data_of(parsec_data_collection_t *dc, ...)
{
    va_list ap;
    int k;

    va_start(ap, dc);
    k = va_arg(ap, int);
    va_end(ap);
    return tiles_data_of_key(dc, (parsec_data_key_t)k);
}

parsec_data_collection_t *qaoa_tiles_new(int myrank, int nodes, int owner, int n, size_t tile_bytes)
{
    qaoa_tiles_t *t = (qaoa_tiles_t*)calloc(1, sizeof(qaoa_tiles_t));
    parsec_data_collection_t *dc = &t->super;

    parsec_data_collection_init(dc, nodes, myrank);
    dc->rank_of     = tiles_rank_of;
    dc->rank_of_key = tiles_rank_of_key;
    dc->vpid_of     = tiles_vpid_of;
    dc->vpid_of_key = tiles_vpid_of_key;
    dc->data_key    = tiles_data_key;
    dc->data_of     = tiles_data_of;
    dc->data_of_key = tiles_data_of_key;

    t->owner      = (owner >= nodes) ? nodes - 1 : owner;
    t->n          = n;
    t->tile_bytes = tile_bytes;
    t->data       = (parsec_data_t**)calloc((size_t)(n > 0 ? n : 1), sizeof(parsec_data_t*));
    t->memory     = (unsigned char**)calloc((size_t)(n > 0 ? n : 1), sizeof(unsigned char*));
    if( tile_bytes > 0 ) {
        for( int k = 0; k < n; k++ ) {
            if( (int)tiles_rank(t, k) != myrank ) continue;
            /* Zeroed, and in multiples of eight bytes: the tiles hold doubles. */
            t->memory[k] = (unsigned char*)calloc((tile_bytes + 7) / 8, 8);
        }
    }
    return dc;
}

void *qaoa_tiles_ptr(parsec_data_collection_t *dc, int k)
{
    qaoa_tiles_t *t = (qaoa_tiles_t*)dc;

    return ((k >= 0) && (k < t->n)) ? t->memory[k] : NULL;
}

void qaoa_tiles_free(parsec_data_collection_t *dc)
{
    qaoa_tiles_t *t = (qaoa_tiles_t*)dc;

    for( int k = 0; k < t->n; k++ ) {
        if( NULL != t->data[k] ) parsec_data_destroy(t->data[k]);
        free(t->memory[k]);
    }
    free(t->data);
    free(t->memory);
    parsec_data_collection_destroy(dc);
    free(dc);
}

void qaoa_empty_boundaries(parsec_data_collection_t *E, const qaoa_instance_t *inst)
{
    for( int b = 0; b < inst->part.nb_blocks; b++ ) {
        qaoa_boundary_t *h = (qaoa_boundary_t*)qaoa_tiles_ptr(E, b);
        if( NULL != h ) qaoa_boundary_init(h, inst, b);
    }
}

/* ------------------------------------------------------------------------
 * Arenas
 * --------------------------------------------------------------------- */

void qaoa_arena_set(parsec_arena_datatype_t *adt, size_t bytes)
{
    parsec_datatype_t block = PARSEC_DATATYPE_NULL;

    /* Without MPI there is no datatype to create and none is needed; the
     * size of the arena is given in bytes either way. */
    (void)parsec_type_create_contiguous((int)bytes, parsec_datatype_uint8_t, &block);
    parsec_arena_datatype_set_type(adt, bytes, PARSEC_ARENA_ALIGNMENT_SSE, block);
}

void qaoa_arena_unset(parsec_arena_datatype_t *adt)
{
    if( PARSEC_DATATYPE_NULL != adt->opaque_dtt ) {
        (void)parsec_type_free(&adt->opaque_dtt);
    }
}

/* ------------------------------------------------------------------------
 * Sizes
 * --------------------------------------------------------------------- */

void qaoa_sizes(qaoa_sizes_t *sizes, const qaoa_instance_t *inst, const qaoa_config_t *cfg)
{
    const qaoa_partition_t *p = &inst->part;

    sizes->state    = sizeof(qaoa_state_t);
    sizes->boundary = qaoa_boundary_size(p->max_block);
    sizes->local    = qaoa_local_size(p->max_block, p->max_internal);
    sizes->circuit  = qaoa_circuit_size(p->max_block, p->max_internal, cfg->layers);
    sizes->opt      = sizeof(qaoa_opt_t);
    sizes->theta    = (size_t)(2 * cfg->layers) * sizeof(double);
    sizes->result   = parsec_qpu_result_size(qaoa_result_entries(cfg, p->max_block));
    sizes->eval     = sizeof(qaoa_eval_t);
    sizes->report   = sizeof(qaoa_report_t);
    sizes->all      = (size_t)p->nb_blocks * sizeof(qaoa_state_t);
}
