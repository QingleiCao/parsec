/*
 * Copyright (c) 2010-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/************************************************************
 *distributed matrix generation
 ************************************************************/

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_tabular.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif
#include <string.h>

static parsec_data_key_t tiled_matrix_data_key(struct parsec_data_collection_s *desc, ...);

static int      tiled_matrix_key_to_string(struct parsec_data_collection_s * desc, parsec_data_key_t datakey, char * buffer, uint32_t buffer_size);

parsec_data_t*
parsec_tiled_matrix_create_data(parsec_tiled_matrix_t* matrix,
                         void* ptr,
                         size_t pos,
                         parsec_data_key_t key)
{
    assert( pos <= matrix->nb_local_tiles );
    return parsec_data_create( matrix->data_map + pos,
                              &(matrix->super), key, ptr,
                              matrix->bsiz * parsec_datadist_getsizeoftype(matrix->mtype),
                              PARSEC_DATA_FLAG_PARSEC_MANAGED);
}

void
parsec_tiled_matrix_destroy_data( parsec_tiled_matrix_t* matrix )
{
    if ( matrix->data_map != NULL ) {
        parsec_data_t **data = matrix->data_map;

        for(size_t i = 0; i < matrix->nb_local_tiles; i++, data++) {
            if( NULL == *data ) continue;
            parsec_data_destroy( *data );
        }

        free( matrix->data_map );
        matrix->data_map = NULL;
    }
}

/***************************************************************************/
/**
 *  Internal static descriptor initializer
 **/
void parsec_tiled_matrix_init( parsec_tiled_matrix_t *tdesc,
                             parsec_matrix_type_t    mtyp,
                             parsec_matrix_storage_t storage,
                             int dtype, int nodes, int myrank,
                             size_t mb, size_t nb,
                             size_t lm, size_t ln,
                             size_t i,  size_t j,
                             size_t m,  size_t n)
{
    parsec_data_collection_t *o = (parsec_data_collection_t*)tdesc;

    /* Super setup */
    parsec_data_collection_init( o, nodes, myrank );

    /* Change the common data_key */
    o->data_key = tiled_matrix_data_key;

    /**
     * Setup the tiled matrix properties
     */

    /* Matrix address */
    /* tdesc->mat = NULL; */
    /* tdesc->A21 = (lm - lm%mb)*(ln - ln%nb); */
    /* tdesc->A12 = (     lm%mb)*(ln - ln%nb) + tdesc->A21; */
    /* tdesc->A22 = (lm - lm%mb)*(     ln%nb) + tdesc->A12; */

    /* Matrix properties */
    tdesc->data_map = NULL;
    tdesc->mtype    = mtyp;
    tdesc->storage  = storage;
    tdesc->dtype    = parsec_matrix_type | dtype;
    tdesc->tileld   = (storage == PARSEC_MATRIX_TILE) ? mb : lm;
    tdesc->mb       = mb;
    tdesc->nb       = nb;
    tdesc->bsiz     = (size_t)mb * nb;

    /* Large matrix parameters */
    tdesc->lm = lm;
    tdesc->ln = ln;

    /* Large matrix derived parameters */
    /* tdesc->lm1 = (lm/mb); */
    /* tdesc->ln1 = (ln/nb); */
    tdesc->lmt = (lm%mb==0) ? (lm/mb) : (lm/mb+1);
    tdesc->lnt = (ln%nb==0) ? (ln/nb) : (ln/nb+1);

    /* Update lm and ln to include the padding */
    if ( storage != PARSEC_MATRIX_LAPACK ) {
        tdesc->lm = tdesc->lmt * tdesc->mb;
        tdesc->ln = tdesc->lnt * tdesc->nb;
    }

    /* Locally stored matrix dimensions */
    tdesc->llm = tdesc->lm;
    tdesc->lln = tdesc->ln;

    /* Submatrix parameters */
    tdesc->i = i;
    tdesc->j = j;
    tdesc->m = m;
    tdesc->n = n;

    /* Submatrix derived parameters */
    tdesc->mt = (i+m-1)/mb - i/mb + 1;
    tdesc->nt = (j+n-1)/nb - j/nb + 1;

    /* finish to update the main object properties */
    o->key_to_string = tiled_matrix_key_to_string;
    if( asprintf(&(o->key_dim), "(%zu, %zu)", tdesc->lmt, tdesc->lnt) <= 0 ) {
        o->key_dim = NULL;
    }

    /* Define the default datatye of the datacollection */
    parsec_datatype_t elem_dt = PARSEC_DATATYPE_NULL;
    ptrdiff_t extent;
    parsec_translate_matrix_type( tdesc->mtype, &elem_dt );
    if( PARSEC_SUCCESS != parsec_matrix_define_datatype(&o->default_dtt, elem_dt,
                                              PARSEC_MATRIX_FULL, 1 /*diag*/,
                                              tdesc->mb, tdesc->nb, tdesc->mb /*ld*/,
                                              -1/*resized*/, &extent)){
        parsec_fatal("Unable to create a datatype for the data collection.");
    }
}

void
parsec_tiled_matrix_destroy( parsec_tiled_matrix_t *tdesc )
{
    parsec_data_collection_t *dc = (parsec_data_collection_t*)tdesc;
    parsec_type_free(&dc->default_dtt);

    parsec_tiled_matrix_destroy_data( tdesc );
    parsec_data_collection_destroy( dc );
}

parsec_tiled_matrix_t *
parsec_tiled_matrix_submatrix( parsec_tiled_matrix_t *tdesc,
                        size_t i, size_t j, size_t m, size_t n)
{
    size_t mb, nb;
    parsec_tiled_matrix_t *newdesc;

    mb = tdesc->mb;
    nb = tdesc->nb;

    if ( (i == (size_t)-1) || (i%mb) != 0 ) {
        parsec_warning("Invalid value of i");
        return NULL;
    }
    if ( (j == (size_t)-1) || (j%nb) != 0 ) {
        parsec_warning("Invalid value of j");
        return NULL;
    }
    if ( (m == (size_t)-1) || ((m+i) > tdesc->lm) ) {
        parsec_warning("Invalid value of m");
        return NULL;
    }
    if ( (n == (size_t)-1) || ((n+j) > tdesc->ln) ) {
        parsec_warning("Invalid value of n");
        return NULL;
    }

    if( tdesc->dtype & parsec_matrix_block_cyclic_type ) {
        newdesc = (parsec_tiled_matrix_t*) malloc ( sizeof(parsec_matrix_block_cyclic_t) );
        memcpy( newdesc, tdesc, sizeof(parsec_matrix_block_cyclic_t) );
    }
    else if( tdesc->dtype & parsec_matrix_sym_block_cyclic_type ) {
        newdesc = (parsec_tiled_matrix_t*) malloc ( sizeof(parsec_matrix_sym_block_cyclic_t) );
        memcpy( newdesc, tdesc, sizeof(parsec_matrix_sym_block_cyclic_t) );
    }
    else if( tdesc->dtype & parsec_matrix_tabular_type ) {
        newdesc = (parsec_tiled_matrix_t*) malloc ( sizeof(parsec_matrix_tabular_t) );
        memcpy( newdesc, tdesc, sizeof(parsec_matrix_tabular_t) );
    } else {
        parsec_warning("Type not completely defined");
        return NULL;
    }

    // Submatrix parameters
    newdesc->i = i;
    newdesc->j = j;
    newdesc->m = m;
    newdesc->n = n;
    // Submatrix derived parameters
    newdesc->mt = (i+m-1)/mb - i/mb + 1;
    newdesc->nt = (j+n-1)/nb - j/nb + 1;
    return newdesc;
}

/* return a unique key (unique only for the specified parsec_dc) associated to a data */
static parsec_data_key_t tiled_matrix_data_key(struct parsec_data_collection_s *desc, ...)
{
    parsec_tiled_matrix_t * dc;
    unsigned int m, n;
    va_list ap;
    dc = (parsec_tiled_matrix_t*)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->i / dc->mb;
    n += dc->j / dc->nb;

    return ((n * dc->lmt) + m);
}

static int  tiled_matrix_key_to_string(struct parsec_data_collection_s *desc, parsec_data_key_t datakey, char * buffer, uint32_t buffer_size)
/* return a string meaningful for profiling about data */
{
    parsec_tiled_matrix_t * dc;
    unsigned int m, n;
    int res;
    dc = (parsec_tiled_matrix_t*)desc;
    m = datakey % dc->lmt;
    n = datakey / dc->lmt;
    res = snprintf(buffer, buffer_size, "(%u, %u)", m, n);
    if (res < 0)
    {
        parsec_warning("Wrong key_to_string for tile (%u, %u) key: %u", m, n, datakey);
    }
    return res;
}

/*
 * Writes the data into the file filename
 * Sequential function per node
 */
int parsec_tiled_matrix_data_write(parsec_tiled_matrix_t *tdesc, char *filename)
{
    parsec_data_collection_t *dc = &(tdesc->super);
    parsec_data_t* data;
    FILE *tmpf;
    char *buf;
    size_t i, j, k;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  parsec_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        parsec_warning("The file %s cannot be open", filename);
        return PARSEC_ERR_NOT_FOUND;
    }

    if ( tdesc->storage == PARSEC_MATRIX_TILE ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j < tdesc->nt ; j++) {
                unsigned int ui = (unsigned int)i;
                unsigned int uj = (unsigned int)j;
                if ( dc->rank_of( dc, ui, uj ) == myrank ) {
                    data = dc->data_of( dc, ui, uj );
                    buf = parsec_data_get_ptr(data, 0);
                    fwrite(buf, eltsize, tdesc->bsiz, tmpf );
                }
            }
    } else {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j < tdesc->nt ; j++) {
                unsigned int ui = (unsigned int)i;
                unsigned int uj = (unsigned int)j;
                if ( dc->rank_of( dc, ui, uj ) == myrank ) {
                    data = dc->data_of( dc, ui, uj );
                    buf = parsec_data_get_ptr(data, 0);
                    for (k = 0; k < tdesc->nb; k++) {
                        fwrite(buf, eltsize, tdesc->mb, tmpf );
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }


    fclose(tmpf);
    return PARSEC_SUCCESS;
}

/*
 * Read the data from the file filename
 * Sequential function per node
 */
int parsec_tiled_matrix_data_read(parsec_tiled_matrix_t *tdesc, char *filename)
{
    parsec_data_collection_t *dc = &(tdesc->super);
    parsec_data_t* data;
    FILE *tmpf;
    char *buf;
    size_t i, j, k;
    size_t ret;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  parsec_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        parsec_warning("The file %s cannot be open", filename);
        return -1;
    }

    if ( tdesc->storage == PARSEC_MATRIX_TILE ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j < tdesc->nt ; j++) {
                unsigned int ui = (unsigned int)i;
                unsigned int uj = (unsigned int)j;
                if ( dc->rank_of( dc, ui, uj ) == myrank ) {
                    data = dc->data_of( dc, ui, uj );
                    buf = parsec_data_get_ptr(data, 0);
                    ret = fread(buf, eltsize, tdesc->bsiz, tmpf );
                    if ( ret !=  tdesc->bsiz ) {
                        parsec_warning("The read on tile(%d, %d) read %zu elements instead of %zu",
                                i, j, ret, tdesc->bsiz);
                        fclose(tmpf);
                        return PARSEC_ERR_TRUNCATE;
                    }
                }
            }
    } else {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j < tdesc->nt ; j++) {
                unsigned int ui = (unsigned int)i;
                unsigned int uj = (unsigned int)j;
                if ( dc->rank_of( dc, ui, uj ) == myrank ) {
                    data = dc->data_of( dc, ui, uj );
                    buf = parsec_data_get_ptr(data, 0);
                    for (k = 0; k < tdesc->nb; k++) {
                        ret = fread(buf, eltsize, tdesc->mb, tmpf );
                        if ( ret !=  (size_t)tdesc->mb ) {
                            parsec_warning("The read on tile(%d, %d) read %zu elements instead of %zu",
                                    i, j, ret, tdesc->mb);
                            fclose(tmpf);
                            return PARSEC_ERR_TRUNCATE;
                        }
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }

    fclose(tmpf);
    return PARSEC_SUCCESS;
}
