/*
 * Copyright (c) 2026 The University of Tennessee and The University
 *                        of Tennessee Research Foundation. All rights
 *                        reserved.
 */
#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_internal.h"

#if defined(DISTRIBUTED)
#include <mpi.h>
#endif

int parsec_alloc_size_test(parsec_context_t *parsec, parsec_tiled_matrix_t *A);

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_matrix_block_cyclic_t dcA;
    int rank = 0, size = 1;
    int info = 0;

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    parsec = parsec_init(-1, &argc, &argv);
    if(NULL == parsec) return EXIT_FAILURE;

    if(size != 1) {
        if(rank == 0) fprintf(stderr, "alloc_size test supports single-node runtime launch only\n");
        parsec_fini(&parsec);
#if defined(DISTRIBUTED)
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }

    if(parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_CUDA) == 0) {
        if(rank == 0) {
            parsec_warning("This test requires at least one CUDA device");
            printf("TEST SKIPPED\n");
        }
        parsec_fini(&parsec);
#if defined(DISTRIBUTED)
        MPI_Finalize();
#endif
        return -PARSEC_ERR_DEVICE;
    }

    parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                                    rank, 32, 32, 128, 128, 0, 0,
                                    128, 128, 1, 1, 1, 1, 0, 0);
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                   (size_t)dcA.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "dcA_alloc_size");

    {
        int *vals = (int*)dcA.mat;
        size_t count = (size_t)dcA.super.nb_local_tiles * (size_t)dcA.super.bsiz;
        for(size_t i = 0; i < count; i++) vals[i] = 0x11111111;
    }

    {
        parsec_data_collection_t *dc = (parsec_data_collection_t*)&dcA;
        for(int m = 0; m < dcA.super.mt; m++) {
            for(int n = 0; n < dcA.super.nt; n++) {
                parsec_data_t *data = dc->data_of(dc, m, n);
                if(NULL == data) continue;
                if(data->nb_elts_alloc != 0) {
                    fprintf(stderr, "default nb_elts_alloc is expected 0 but got %zu at (%d,%d)\n",
                            data->nb_elts_alloc, m, n);
                    info++;
                }
            }
        }
    }

    if(info == 0) {
        info = parsec_alloc_size_test(parsec, (parsec_tiled_matrix_t *)&dcA);
    }

    if(rank == 0) {
        if(info == 0) printf("TEST PASSED\n");
        else          printf("TEST FAILED (%d errors)\n", info);
    }

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA);
    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif
    return (0 == info) ? EXIT_SUCCESS : EXIT_FAILURE;
}
