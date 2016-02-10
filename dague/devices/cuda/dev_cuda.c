/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/dague_internal.h"
#include "dague/sys/atomic.h"

#include "dague/utils/mca_param.h"
#include "dague/constants.h"

#if defined(HAVE_CUDA)
#include "dague.h"
#include "dague/data_internal.h"
#include "dague/devices/cuda/dev_cuda.h"
#include "dague/profiling.h"
#include "dague/execution_unit.h"
#include "dague/arena.h"
#include "dague/utils/output.h"
#include "dague/utils/argv.h"
#include "dague/scheduling.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <dlfcn.h>
#include <sys/stat.h>

#if defined(DAGUE_PROF_TRACE)
/* Accepted values are: DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT |
 *                      DAGUE_PROFILE_CUDA_TRACK_OWN | DAGUE_PROFILE_CUDA_TRACK_EXEC
 */
int dague_cuda_trackable_events = DAGUE_PROFILE_CUDA_TRACK_EXEC | DAGUE_PROFILE_CUDA_TRACK_DATA_OUT
  | DAGUE_PROFILE_CUDA_TRACK_DATA_IN | DAGUE_PROFILE_CUDA_TRACK_OWN;
int dague_cuda_movein_key_start;
int dague_cuda_movein_key_end;
int dague_cuda_moveout_key_start;
int dague_cuda_moveout_key_end;
int dague_cuda_own_GPU_key_start;
int dague_cuda_own_GPU_key_end;
#endif  /* defined(PROFILING) */

int dague_cuda_output_stream = -1;
static char* cuda_lib_path = NULL;

static int
dague_cuda_memory_reserve( gpu_device_t* gpu_device,
                           int           memory_percentage,
                           int           number_of_elements,
                           size_t        eltsize );
static int
dague_cuda_memory_release( gpu_device_t* gpu_device );

/* the rate represents how many times single is faster than double */
int stod_rate[3] = {8, 2, 3};

/* look up how many cuda cores per SM
 * 1.x    8
 * 2.0    32
 * 2.1    48
 * 3.x    192
 */
static int dague_cuda_lookup_device_cudacores(int *cuda_cores, int major, int minor)
{
    if (major == 1) {
        *cuda_cores = 8;
    } else if (major == 2 && minor == 0) {
        *cuda_cores = 32;
    } else if (major == 2 && minor == 1) {
        *cuda_cores = 48;
    } else if (major == 3) {
        *cuda_cores = 192;
    } else {
        dague_debug_verbose(3, dague_debug_output, "Unsupporttd GPU %d, %d, skipping.", major, minor);
            return DAGUE_ERROR;
    }
    return DAGUE_SUCCESS;
}

static int dague_cuda_device_fini(dague_device_t* device)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    cudaError_t status;
    int j, k;

    status = cudaSetDevice( gpu_device->cuda_index );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_device_fini) cudaSetDevice ", status,
                            {continue;} );

    /* Release the registered memory */
    dague_cuda_memory_release(gpu_device);

    /* Release pending queue */
    OBJ_DESTRUCT(&gpu_device->pending);

    /* Release all streams */
    for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
        dague_gpu_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

        exec_stream->max_events   = DAGUE_MAX_EVENTS_PER_STREAM;
        exec_stream->executed     = 0;
        exec_stream->start        = 0;
        exec_stream->end          = 0;

        for( k = 0; k < exec_stream->max_events; k++ ) {
            assert( NULL == exec_stream->tasks[k] );
            status = cudaEventDestroy(exec_stream->events[k]);
            DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_device_fini) cudaEventDestroy ", status,
                                    {continue;} );
        }
        free(exec_stream->events); exec_stream->events = NULL;
        free(exec_stream->tasks); exec_stream->tasks = NULL;
        free(exec_stream->fifo_pending); exec_stream->fifo_pending = NULL;
        /* Release the stream */
        cudaStreamDestroy( exec_stream->cuda_stream );
    }
    free(gpu_device->exec_stream); gpu_device->exec_stream = NULL;

    gpu_device->cuda_index = -1;

    /* Cleanup the GPU memory. */
    OBJ_DESTRUCT(&gpu_device->gpu_mem_lru);
    OBJ_DESTRUCT(&gpu_device->gpu_mem_owned_lru);

    return DAGUE_SUCCESS;
}

static int dague_cuda_memory_register(dague_device_t* device, dague_ddesc_t* desc, void* ptr, size_t length)
{
    cudaError_t status;
    int rc = DAGUE_ERROR;

    if (desc->memory_registration_status == MEMORY_STATUS_REGISTERED) {
        rc = DAGUE_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the CUDA interface to register the memory
     * as another thread might be submiting tasks at the same time
     * (cuda_scheduling.h), and we do not set a device since we register it for
     * all devices.
     */
    status = cudaHostRegister(ptr, length, cudaHostRegisterPortable );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_register) cudaHostRegister ", status,
                            { goto restore_and_return; } );

    rc = DAGUE_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_REGISTERED;

  restore_and_return:
    (void)device;
    return rc;
}

static int dague_cuda_memory_unregister(dague_device_t* device, dague_ddesc_t* desc, void* ptr)
{
    cudaError_t status;
    int rc = DAGUE_ERROR;

    if (desc->memory_registration_status == MEMORY_STATUS_UNREGISTERED) {
        rc = DAGUE_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the CUDA interface to unregister the memory
     * as another thread might be submiting tasks at the same time (cuda_scheduling.h)
     */
    status = cudaHostUnregister(ptr);
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_ununregister) cudaHostUnregister ", status,
                            {continue;} );

    rc = DAGUE_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_UNREGISTERED;

    (void)device;
    return rc;
}

static int cuda_legal_compute_capabilitites[] = {10, 11, 12, 13, 20, 21, 30, 35};

void* cuda_solve_handle_dependencies(gpu_device_t* gpu_device,
                                     const char* fname)
{
    char library_name[FILENAME_MAX], function_name[FILENAME_MAX], *env;
    int i, index, capability = gpu_device->major * 10 + gpu_device->minor;
    cudaError_t status;
    void *fn = NULL, *dlh = NULL;
    char** argv = NULL, **target;

    status = cudaSetDevice( gpu_device->cuda_index );
    DAGUE_CUDA_CHECK_ERROR( "(cuda_solve_handle_dependencies) cudaSetDevice ", status, {continue;} );

    for( i = 0, index = -1; i < (int)sizeof(cuda_legal_compute_capabilitites); i++ ) {
        if(cuda_legal_compute_capabilitites[i] == capability) {
            index = i;
            break;
        }
    }
    if( -1 == index ) {  /* This shouldn't have happened */
        return NULL;
    }

    /**
     * Prepare the list of PATH or FILE to be searched for a CUDA shared library.
     * In any case this list might be a list of ; separated possible targets,
     * where each target can be either a directory or a specific file.
     */
    env = getenv("DAGUE_CUCORES_LIB");
    if( NULL != env ) {
        argv = dague_argv_split(env, ';');
    } else if( NULL != cuda_lib_path ) {
        argv = dague_argv_split(cuda_lib_path, ';');
    }

  retry_lesser_sm_version:
    if( -1 == index ) {
        capability = 0;
        snprintf(function_name, FILENAME_MAX, "%s", fname);
    }
    else {
        capability = cuda_legal_compute_capabilitites[index];
        snprintf(function_name, FILENAME_MAX, "%s_SM%2d", fname, capability);
    }

    for( target = argv; (NULL != target) && (NULL != *target); target++ ) {
        struct stat status;
        if( 0 != stat(*target, &status) ) {
            dague_debug_verbose(10, dague_cuda_output_stream,
                                 "Could not stat the %s path (%s)", *target, strerror(errno));
            continue;
        }
        if( S_ISDIR(status.st_mode) ) {
            if( capability )
                snprintf(library_name,  FILENAME_MAX, "%s/libdplasma_cucores_sm%d.so", *target, capability);
            else
                snprintf(library_name,  FILENAME_MAX, "%s/libdplasma_cores_cuda.so", *target);
        } else {
            snprintf(library_name,  FILENAME_MAX, "%s", *target);
        }

        dlh = dlopen(library_name, RTLD_NOW | RTLD_NODELETE );
        if(NULL == dlh) {
            dague_debug_verbose(10, dague_cuda_output_stream,
                                 "Could not find %s dynamic library (%s)", library_name, dlerror());
            continue;
        }
        fn = dlsym(dlh, function_name);
        dlclose(dlh);
        if( NULL != fn ) {
            dague_debug_verbose(4, dague_cuda_output_stream,
                                 "Function %s found in shared object %s",
                                 function_name, library_name);
            break;  /* we got one, stop here */
        }
    }
    /* Couldn't load from named dynamic libs, try linked/static */
    if(NULL == fn) {
        dague_debug_verbose(10, dague_cuda_output_stream,
                             "No dynamic function %s found, trying from compile time linked in",
                             function_name);
        dlh = dlopen(NULL, RTLD_NOW | RTLD_NODELETE);
        if(NULL != dlh) {
            fn = dlsym(dlh, function_name);
            if(NULL != fn) {
                dague_debug_verbose(4, dague_cuda_output_stream,
                                     "Function %s found in the application object",
                                     function_name);
            }
            dlclose(dlh);
        }
    }

    /* Still not found?? skip this GPU */
    if(NULL == fn) {
        dague_debug_verbose(10, dague_cuda_output_stream,
                             "No function %s found for CUDA device %s",
                             function_name, gpu_device->super.name);
        index--;
        if(-1 <= index)
            goto retry_lesser_sm_version;
    }

    if( NULL != argv )
        dague_argv_free(argv);

    return fn;
}

static int
dague_cuda_handle_register(dague_device_t* device, dague_handle_t* handle)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    uint32_t i, j, dev_mask = 0x0;
    int32_t rc = DAGUE_ERR_NOT_FOUND;

    /**
     * Let's suppose it is not our job to detect if a particular body can
     * run or not. We will need to add some properties that will allow the
     * user to write the code to assess this.
     */
    assert(DAGUE_DEV_CUDA == device->type);
    for( i = 0; i < handle->nb_functions; i++ ) {
        const dague_function_t* function = handle->functions_array[i];
        __dague_chore_t* chores = (__dague_chore_t*)function->incarnations;
        for( dev_mask = j = 0; NULL != chores[j].hook; j++ ) {
            if( chores[j].type == device->type ) {
                if ( NULL == chores[j].dyld ) {
                    /* No dynamic load for this kernel */
                    chores[gpu_device->cuda_index].dyld_fn = NULL;
                    rc = DAGUE_SUCCESS;
                    dev_mask |= chores[j].type;
                }
                else {
                    void* devf = cuda_solve_handle_dependencies(gpu_device, chores[j].dyld);
                    if( NULL != devf ) {
                        chores[gpu_device->cuda_index].dyld_fn = devf;
                        rc = DAGUE_SUCCESS;
                        dev_mask |= chores[j].type;
                    }
                }
            }
        }
    }
    /* Not a single chore supports this device, there is no reason to check anything further */
    if(DAGUE_SUCCESS != rc) {
        handle->devices_mask &= ~(device->device_index);
    }

    return rc;
}

static int
dague_cuda_handle_unregister(dague_device_t* device, dague_handle_t* handle)
{
    (void)device; (void)handle;
    return DAGUE_SUCCESS;
}

int dague_gpu_init(dague_context_t *dague_context)
{
    int cuda_memory_block_size, cuda_memory_percentage, cuda_memory_number_of_blocks = -1;
    int show_caps_index, show_caps = 0;
    int use_cuda_index, use_cuda;
    int cuda_mask, cuda_verbosity;
    int ndevices, i, j, k;
    cudaError_t cudastatus;

    use_cuda_index = dague_mca_param_reg_int_name("device_cuda", "enabled",
                                                  "The number of CUDA device to enable for the next PaRSEC context",
                                                  false, false, 0, &use_cuda);
    (void)dague_mca_param_reg_int_name("device_cuda", "mask",
                                       "The bitwise mask of CUDA devices to be enabled (default all)",
                                       false, false, 0xffffffff, &cuda_mask);
    (void)dague_mca_param_reg_int_name("device_cuda", "verbose",
                                       "Set the verbosity level of the CUDA device (negative value: use debug verbosity), higher is less verbose)\n",
                                       false, false, -1, &cuda_verbosity);
    (void)dague_mca_param_reg_string_name("device_cuda", "path",
                                          "Path to the shared library files containing the CUDA version of the hooks. It is a ;-separated list of either directories or .so files.\n",
                                          false, false, DAGUE_LIB_CUDA_PREFIX, &cuda_lib_path);
    (void)dague_mca_param_reg_int_name("device_cuda", "memory_block_size",
                                       "The CUDA memory page for PaRSEC internal management.",
                                       false, false, 32*1024, &cuda_memory_block_size);
    (void)dague_mca_param_reg_int_name("device_cuda", "memory_use",
                                       "The percentage of the total GPU memory to be used by this PaRSEC context",
                                       false, false, 95, &cuda_memory_percentage);
    (void)dague_mca_param_reg_int_name("device_cuda", "memory_number_of_blocks",
                                       "Alternative to device_cuda_memory_use: sets exactly the number of blocks to allocate (-1 means to use a percentage of the available memory)",
                                       false, false, -1, &cuda_memory_number_of_blocks);
    if( 0 == use_cuda ) {
        return -1;  /* Nothing to do around here */
    }

    if( cuda_verbosity >= 0 ) {
        dague_cuda_output_stream = dague_output_open(NULL);
        dague_output_set_verbosity(dague_cuda_output_stream, cuda_verbosity);
    } else {
        dague_cuda_output_stream = dague_debug_output;
    }

    cudastatus = cudaGetDeviceCount( &ndevices );
    DAGUE_CUDA_CHECK_ERROR( "cudaGetDeviceCount ", cudastatus,
                            {
                                if( 0 < use_cuda_index )
                                    dague_mca_param_set_int(use_cuda_index, 0);
                                return -1;
                            } );


    if( ndevices > use_cuda ) {
        if( 0 < use_cuda_index ) {
            ndevices = use_cuda;
        }
    } else if (ndevices < use_cuda ) {
        if( 0 < use_cuda_index ) {
            dague_warning("User requested %d GPUs, but only %d are available in this machine. PaRSEC will enable all of them.", use_cuda, ndevices);
            dague_mca_param_set_int(use_cuda_index, ndevices);
        }
    }

    /* Update the number of GPU for the upper layer */
    use_cuda = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }

    show_caps_index = dague_mca_param_find("device", NULL, "show_capabilities");
    if(0 < show_caps_index) {
        dague_mca_param_lookup_int(show_caps_index, &show_caps);
    }
#if defined(DAGUE_PROF_TRACE)
    dague_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                            sizeof(intptr_t), "pointer{int64_t}",
                                            &dague_cuda_movein_key_start, &dague_cuda_movein_key_end);
    dague_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                            sizeof(intptr_t), "pointer{int64_t}",
                                            &dague_cuda_moveout_key_start, &dague_cuda_moveout_key_end);
    dague_profiling_add_dictionary_keyword( "cuda", "fill:#66ff66",
                                            0, NULL,
                                            &dague_cuda_own_GPU_key_start, &dague_cuda_own_GPU_key_end);
#endif  /* defined(PROFILING) */

    for( i = 0; i < ndevices; i++ ) {
        gpu_device_t* gpu_device;
        char *szName;
        int major, minor, concurrency, computemode, streaming_multiprocessor, cuda_cores, clockRate;
        struct cudaDeviceProp prop;

        /* Allow fine grain selection of the GPU's */
        if( !((1 << i) & cuda_mask) ) continue;

        cudastatus = cudaSetDevice( i );
        DAGUE_CUDA_CHECK_ERROR( "cudaSetDevice ", cudastatus, {continue;} );
        cudastatus = cudaGetDeviceProperties( &prop, i );
        DAGUE_CUDA_CHECK_ERROR( "cudaGetDeviceProperties ", cudastatus, {continue;} );

        szName    = prop.name;
        major     = prop.major;
        minor     = prop.minor;
        clockRate = prop.clockRate;
        concurrency = prop.concurrentKernels;
        streaming_multiprocessor = prop.multiProcessorCount;
        computemode = prop.computeMode;

        gpu_device = (gpu_device_t*)calloc(1, sizeof(gpu_device_t));
        OBJ_CONSTRUCT(gpu_device, dague_list_item_t);
        gpu_device->cuda_index = (uint8_t)i;
        gpu_device->major      = (uint8_t)major;
        gpu_device->minor      = (uint8_t)minor;
        gpu_device->super.name = strdup(szName);

        gpu_device->max_exec_streams = DAGUE_MAX_STREAMS;
        gpu_device->exec_stream =
            (dague_gpu_exec_stream_t*)malloc(gpu_device->max_exec_streams
                                             * sizeof(dague_gpu_exec_stream_t));
        for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
            dague_gpu_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

            /* Allocate the stream */
            cudastatus = cudaStreamCreate( &(exec_stream->cuda_stream) );
            DAGUE_CUDA_CHECK_ERROR( "cudaStreamCreate ", cudastatus,
                                    {break;} );
            exec_stream->workspace    = NULL;
            exec_stream->max_events   = DAGUE_MAX_EVENTS_PER_STREAM;
            exec_stream->executed     = 0;
            exec_stream->start        = 0;
            exec_stream->end          = 0;
            exec_stream->fifo_pending = (dague_list_t*)OBJ_NEW(dague_list_t);
            OBJ_CONSTRUCT(exec_stream->fifo_pending, dague_list_t);
            exec_stream->tasks  = (dague_gpu_context_t**)malloc(exec_stream->max_events
                                                                * sizeof(dague_gpu_context_t*));
            exec_stream->events = (cudaEvent_t*)malloc(exec_stream->max_events * sizeof(cudaEvent_t));
            /* and the corresponding events */
            for( k = 0; k < exec_stream->max_events; k++ ) {
                exec_stream->events[k] = NULL;
                exec_stream->tasks[k]  = NULL;
                cudastatus = cudaEventCreate(&(exec_stream->events[k]));
                DAGUE_CUDA_CHECK_ERROR( "(INIT) cudaEventCreate ", (cudaError_t)cudastatus,
                                        {break;} );
            }
#if defined(DAGUE_PROF_TRACE)
            exec_stream->profiling = dague_profiling_thread_init( 2*1024*1024, DAGUE_PROFILE_STREAM_STR, i, j );
            if(j == 0) {
                exec_stream->prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_IN;
                exec_stream->prof_event_key_start    = dague_cuda_movein_key_start;
                exec_stream->prof_event_key_end      = dague_cuda_movein_key_end;
            } else if(j == 1) {
                exec_stream->prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_DATA_OUT;
                exec_stream->prof_event_key_start    = dague_cuda_moveout_key_start;
                exec_stream->prof_event_key_end      = dague_cuda_moveout_key_end;
            } else {
                exec_stream->prof_event_track_enable = dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_EXEC;
                exec_stream->prof_event_key_start    = -1;
                exec_stream->prof_event_key_end      = -1;
            }
#endif  /* defined(DAGUE_PROF_TRACE) */
        }

        gpu_device->super.type                 = DAGUE_DEV_CUDA;
        gpu_device->super.executed_tasks       = 0;
        gpu_device->super.transferred_data_in  = 0;
        gpu_device->super.transferred_data_out = 0;
        gpu_device->super.required_data_in     = 0;
        gpu_device->super.required_data_out    = 0;

        gpu_device->super.device_fini              = dague_cuda_device_fini;
        gpu_device->super.device_memory_register   = dague_cuda_memory_register;
        gpu_device->super.device_memory_unregister = dague_cuda_memory_unregister;
        gpu_device->super.device_handle_register   = dague_cuda_handle_register;
        gpu_device->super.device_handle_unregister = dague_cuda_handle_unregister;

        if (dague_cuda_lookup_device_cudacores(&cuda_cores, major, minor) == DAGUE_ERROR ) {
            return -1;
        }

        gpu_device->super.device_sweight = (float)streaming_multiprocessor * (float)cuda_cores * (float)clockRate * 2.0 / 1000000;
        gpu_device->super.device_dweight = gpu_device->super.device_sweight / stod_rate[major-1];

        if( show_caps ) {
            dague_inform("GPU Device %d (capability %d.%d): %s\n"
                         "\tSM                 : %d\n"
                         "\tclockRate          : %d\n"
                         "\tconcurrency        : %s\n"
                         "\tcomputeMode        : %d\n"
                         "\tFlops capacity     : single %2.4f, double %2.4f",
                         i, major, minor,szName,
                         streaming_multiprocessor,
                         clockRate,
                         (concurrency == 1)? "yes": "no",
                         computemode,
                         gpu_device->super.device_sweight, gpu_device->super.device_dweight);
        }

        if( DAGUE_SUCCESS != dague_cuda_memory_reserve(gpu_device,
                                                       cuda_memory_percentage,
                                                       cuda_memory_number_of_blocks,
                                                       cuda_memory_block_size) ) {
            free(gpu_device);
            continue;
        }

        /* Initialize internal lists */
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_lru,       dague_list_t);
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_owned_lru, dague_list_t);
        OBJ_CONSTRUCT(&gpu_device->pending,           dague_list_t);

        gpu_device->sort_starting_p = NULL;
        dague_devices_add(dague_context, &(gpu_device->super));
    }

#if defined(DAGUE_HAVE_PEER_DEVICE_MEMORY_ACCESS)
    for( i = 0; i < ndevices; i++ ) {
        gpu_device_t *source_gpu, *target_gpu;
        CUdevice source, target;
        int canAccessPeer;

        if( NULL == (source_gpu = (gpu_device_t*)dague_devices_get(i)) ) continue;
        /* Skip all non CUDA devices */
        if( DAGUE_DEV_CUDA != source_gpu->super.type ) continue;

        source_gpu->peer_access_mask = 0;

        for( j = 0; j < ndevices; j++ ) {
            if( (NULL == (target_gpu = (gpu_device_t*)dague_devices_get(j))) || (i == j) ) continue;
            /* Skip all non CUDA devices */
            if( DAGUE_DEV_CUDA != target_gpu->super.type ) continue;

            /* Communication mask */
            cudastatus = cudaDeviceCanAccessPeer( &canAccessPeer, source_gpu->cuda_index, target_gpu->cuda_index );
            DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_init) cudaDeviceCanAccessPeer ", cudastatus,
                                    {continue;} );
            if( 1 == canAccessPeer ) {
                cudastatus = cudaDeviceEnablePeerAccess( target_gpu->cuda_index, 0 );
                DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_init) cuCtxEnablePeerAccess ", cudastatus,
                                        {continue;} );
                source_gpu->peer_access_mask = (int16_t)(source_gpu->peer_access_mask | (int16_t)(1 << target_gpu->cuda_index));
            }
        }
    }
#endif

    return 0;
}

int dague_gpu_fini(void)
{
    gpu_device_t* gpu_device;
    int i;

    for(i = 0; i < dague_devices_enabled(); i++) {
        if( NULL == (gpu_device = (gpu_device_t*)dague_devices_get(i)) ) continue;
        if(DAGUE_DEV_CUDA != gpu_device->super.type) continue;
        dague_cuda_device_fini((dague_device_t*)gpu_device);
        dague_devices_remove((dague_device_t*)gpu_device);
        free(gpu_device);
    }

    if( dague_debug_output != dague_cuda_output_stream ) dague_output_close(dague_cuda_output_stream);
    dague_cuda_output_stream = -1;

    return DAGUE_SUCCESS;
}

/**
 * This function reserve the memory_percentage of the total device memory for PaRSEC.
 * This memory will be managed in chuncks of size eltsize. However, multiple chuncks
 * can be reserved in a single allocation.
 */
static int
dague_cuda_memory_reserve( gpu_device_t* gpu_device,
                           int           memory_percentage,
                           int           number_blocks,
                           size_t        eltsize )
{
    cudaError_t status;
    (void)eltsize;

    size_t how_much_we_allocate;
    size_t total_mem, initial_free_mem;
    uint32_t mem_elem_per_gpu = 0;

    status = cudaSetDevice( gpu_device->cuda_index );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_reserve) cudaSetDevice ", status,
                            {continue;} );

    /* Determine how much memory we can allocate */
    cudaMemGetInfo( &initial_free_mem, &total_mem );
    if( number_blocks != -1 ) {
        if( number_blocks == 0 ) {
            dague_warning("CUDA: Invalid argument: requesting 0 bytes of memory on CUDA device %s", gpu_device->super.name);
            return DAGUE_ERROR;
        } else {
            how_much_we_allocate = number_blocks * eltsize;
        }
    } else {
        /** number_blocks == -1 means memory_percentage is used */
        how_much_we_allocate = (memory_percentage * initial_free_mem) / 100;
    }
    if( how_much_we_allocate > initial_free_mem ) {
        /** Handle the case of jokers who require more than 100% of memory,
         *  and eleventh case of computer scientists who don't know how
         *  to divide a number by another
         */
        dague_warning("CUDA: Requested %d bytes on CUDA device %s, but only %d bytes are available -- reducing allocation to max available",
                 how_much_we_allocate, initial_free_mem);
        how_much_we_allocate = initial_free_mem;
    }
    if( how_much_we_allocate < eltsize ) {
        /** Handle another kind of jokers entirely, and cases of
         *  not enough memory on the device
         */
        dague_warning("CUDA: Cannot allocate at least one element on CUDA device %s", gpu_device->super.name);
        return DAGUE_ERROR;
    }

#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
    size_t free_mem = initial_free_mem;
    /*
     * We allocate a bunch of tiles that will be used
     * during the computations
     */
    while( (free_mem > eltsize )
           && ((total_mem - free_mem) < how_much_we_allocate) ) {
        dague_gpu_data_copy_t* gpu_elem;
        void *device_ptr;

        status = (cudaError_t)cudaMalloc( &device_ptr, eltsize);
        DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_reserve) cudaMemAlloc ", status,
                                ({
                                    size_t _free_mem, _total_mem;
                                    cudaMemGetInfo( &_free_mem, &_total_mem );
                                    dague_inform("Per context: free mem %zu total mem %zu (allocated tiles %u)",
                                           _free_mem, _total_mem, mem_elem_per_gpu);
                                    break;
                                }) );
        gpu_elem = OBJ_NEW(dague_data_copy_t);
        DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                              "Allocate CUDA copy %p [ref_count %d] for data [%p]",
                              gpu_elem, gpu_elem->super.obj_reference_count, NULL);
        gpu_elem->device_private = (void*)(long)device_ptr;
        gpu_elem->device_index = gpu_device->super.device_index;
        mem_elem_per_gpu++;
        OBJ_RETAIN(gpu_elem);
        DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                              "Retain and insert CUDA copy %p [ref_count %d] in LRU",
                              gpu_elem, gpu_elem->super.obj_reference_count);
        dague_ulist_fifo_push( &gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem );
        cudaMemGetInfo( &free_mem, &total_mem );
    }
    if( 0 == mem_elem_per_gpu && dague_ulist_is_empty( &gpu_device->gpu_mem_lru ) ) {
        dague_warning("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!",
                 gpu_device->super.context->my_rank, gpu_device->cuda_index);
    }
    else {
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output,  "GPU:\tAllocate %u tiles on the GPU memory", mem_elem_per_gpu );
    }
    DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                          "GPU:\tAllocate %u tiles on the GPU memory", mem_elem_per_gpu);
#else
    if( NULL == gpu_device->memory ) {
        void* base_ptr;
        /* We allocate all the memory on the GPU and we use our memory management */
        mem_elem_per_gpu = (how_much_we_allocate + eltsize - 1 ) / eltsize;
        size_t total_size = (size_t)mem_elem_per_gpu * eltsize;
        status = (cudaError_t)cudaMalloc(&base_ptr, total_size);
        DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_reserve) cudaMalloc ", status,
                                ({ dague_warning("Allocating memory on the GPU device failed"); }) );

        gpu_device->memory = zone_malloc_init( base_ptr, mem_elem_per_gpu, eltsize );

        if( gpu_device->memory == NULL ) {
            dague_warning("GPU:\tRank %d Cannot allocate memory on GPU %d. Skip it!",
                     gpu_device->super.context->my_rank, gpu_device->cuda_index);
            return DAGUE_ERROR;
        }
        DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                              "GPU:\tAllocate %u segment of size %d on the GPU memory",
                              mem_elem_per_gpu, eltsize );
    }
#endif

    return DAGUE_SUCCESS;
}

static void dague_cuda_memory_release_list(gpu_device_t* gpu_device,
                                           dague_list_t* list)
{
    dague_list_item_t* item;

    while(NULL != (item = dague_ulist_fifo_pop(list)) ) {
        dague_gpu_data_copy_t* gpu_copy = (dague_gpu_data_copy_t*)item;
        dague_data_t* original = gpu_copy->original;

        DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                              "Release CUDA copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p, in map %p",
                              gpu_copy, gpu_copy->device_private, gpu_copy->super.super.obj_reference_count,
                              original, (NULL != original ? original->ddesc : NULL));
        assert( gpu_copy->device_index == gpu_device->super.device_index );
        if( DATA_COHERENCY_OWNED == gpu_copy->coherency_state ) {
            dague_warning("GPU[%d] still OWNS the master memory copy for data %d and it is discarding it!",
                     gpu_device->cuda_index, original->key);
        }
#if defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        cudaFree( gpu_copy->device_private );
#else
        zone_free( gpu_device->memory, (void*)gpu_copy->device_private );
#endif
        gpu_copy->device_private = NULL;

        /* At this point the data copies should have no attachement to a data_t. Thus,
         * before we get here (aka below dague_fini), the destructor of the data
         * collection must have been called, releasing all the copies.
         */
        //OBJ_RELEASE(gpu_copy); assert(NULL == gpu_copy);
    }
}

/**
 * This function release the CUDA memory reserved for this device.
 *
 * One has to notice that all the data available on the GPU is stored in one of
 * the two used to keep track of the allocated data, either the gpu_mem_lru or
 * the gpu_mem_owner_lru. Thus, going over all the elements in these two lists
 * should be enough to enforce a clean release.
 */
static int
dague_cuda_memory_release( gpu_device_t* gpu_device )
{
    cudaError_t status;

#if 0
    dump_GPU_state(gpu_device); // debug only
#endif
    status = cudaSetDevice( gpu_device->cuda_index );
    DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_release) cudaSetDevice ", status,
                            {continue;} );

    /* Free all memory on GPU */
    dague_cuda_memory_release_list(gpu_device, &gpu_device->gpu_mem_lru);
    dague_cuda_memory_release_list(gpu_device, &gpu_device->gpu_mem_owned_lru);

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
    if( gpu_device->memory ) {
        void* ptr = zone_malloc_fini(&gpu_device->memory);
        status = cudaFree(ptr);
        DAGUE_CUDA_CHECK_ERROR( "(dague_cuda_memory_release) cudaFree ", status,
                                { dague_warning("Failed to free the GPU backend memory."); } );
    }
#endif

    return DAGUE_SUCCESS;
}

/**
 * Try to find memory space to move all data on the GPU. We attach a device_elem to
 * a memory_elem as soon as a device_elem is available. If we fail to find enough
 * available elements, we push all the elements handled during this allocation
 * back into the pool of available device_elem, to be picked up by another call
 * (this call will remove them from the current task).
 * Returns:
 *    0: All gpu_mem/mem_elem have been initialized
 *   -2: The task needs to rescheduled
 */
static inline int
dague_gpu_data_reserve_device_space( gpu_device_t* gpu_device,
                                     dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_gpu_data_copy_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    dague_data_t* master, *oldmaster;
    const dague_flow_t *flow;
    int i, j;

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        assert( flow && (flow->flow_index == i) );

        /* Skip CTL flows only */
        if(!(flow->flow_flags)) continue;

        temp_loc[i] = NULL;
        master   = this_task->data[i].data_in->original;
        gpu_elem = DAGUE_DATA_GET_COPY(master, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;

        /* There is already a copy on the device */
        if( NULL != gpu_elem ) continue;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
        gpu_elem = OBJ_NEW(dague_data_copy_t);
        DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                              "Allocate CUDA copy %p [ref_count %d] for data %p",
                              gpu_elem, gpu_elem->super.super.obj_reference_count, master);
    malloc_data:
        gpu_elem->device_private = zone_malloc(gpu_device->memory, master->nb_elts);
        if( NULL == gpu_elem->device_private ) {
#endif

          find_another_data:
            /* Look for a data_copy to free */
            lru_gpu_elem = (dague_gpu_data_copy_t*)dague_ulist_fifo_pop(&gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* We can't find enough room on the GPU. Insert the tiles in the begining of
                 * the LRU (in order to be reused asap) and return without scheduling the task.
                 */
                dague_warning("GPU:\tRequest space on GPU failed for %d out of %d data",
                              this_task->function->nb_flows - i,
                              this_task->function->nb_flows);
                for( j = 0; j < i; j++ ) {
                    if( NULL != temp_loc[j] ) {
                        dague_ulist_lifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)temp_loc[j]);
                    }
                }
#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
                OBJ_RELEASE(gpu_elem);
#endif
                return -2;
            }

            DAGUE_LIST_ITEM_SINGLETON(lru_gpu_elem);
            DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                                  "Release LRU-retrieved CUDA copy %p [ref_count %d]",
                                  lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count);
            assert( NULL != lru_gpu_elem );

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the dague_gpu_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->readers ) {
                goto find_another_data; // TODO: add an assert of some sort to check for leaks here?
            }

            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->original );
            assert( NULL   != lru_gpu_elem->original );

            /* Let's check we're not trying to steal one of our own data */
            oldmaster = lru_gpu_elem->original;
            for( j = 0; j < this_task->function->nb_flows; j++ ) {
                if( NULL == this_task->data[j].data_in ) continue;
                if( this_task->data[j].data_in->original == oldmaster ) {
                    goto find_another_data;
                }
            }

            /* The data is not used, and it's not one of ours: we can free it or reuse it */
            dague_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);
            DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                                "GPU[%d]:\tRepurpose copy %p to mirror block %p (in task %s:i) instead of %p",
                                gpu_device->cuda_index, lru_gpu_elem, master, this_task->function->name, i, oldmaster);

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
            /* Let's free this space, and try again to amlloc some space */
            zone_free( gpu_device->memory, (void*)(lru_gpu_elem->device_private) );
            DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                                "Release LRU-retrieved CUDA copy %p [ref_count %d: must be 0]",
                                lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count);
            OBJ_RELEASE(lru_gpu_elem); assert( NULL == lru_gpu_elem );
            goto malloc_data;
        }
#else
        gpu_elem = lru_gpu_elem;
#endif
        assert( 0 == gpu_elem->readers );
        gpu_elem->coherency_state = DATA_COHERENCY_INVALID;
        gpu_elem->version = 0;
        dague_data_copy_attach(master, gpu_elem, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;
        temp_loc[i] = gpu_elem;
        DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                              "Retain and insert CUDA copy %p [ref_count %d] in LRU",
                              gpu_elem, gpu_elem->super.super.obj_reference_count);
        dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_elem);
    }
    return 0;
}

/**
 * If the most current version of the data is not yet available on the GPU memory
 * schedule a transfer.
 * Returns:
 *    0: The most recent version of the data is already available on the GPU
 *    1: A copy has been scheduled on the corresponding stream
 *   -1: A copy cannot be issued due to CUDA.
 */
static inline int
dague_gpu_data_stage_in( gpu_device_t* gpu_device,
                         int32_t type,
                         dague_data_pair_t* task_data,
                         dague_gpu_context_t *gpu_task,
                         dague_gpu_exec_stream_t *gpu_stream )
{
    dague_data_copy_t* in_elem = task_data->data_in;
    dague_data_t* original = in_elem->original;
    dague_gpu_data_copy_t* gpu_elem = task_data->data_out;
    int transfer_from = -1;

    /**
     * If the data will be accessed in write mode, remove it from any lists
     * until the task is completed.
     */
    if( FLOW_ACCESS_WRITE & type ) {
        if (gpu_elem->readers > 0 ) {
            dague_warning("GPU:\tWrite access to data with existing readers (Possible anti-dependency, or concurrent accesses), please prevent that with CTL dependencies\n");
            return -86;
        }
        dague_list_item_ring_chop((dague_list_item_t*)gpu_elem);
        DAGUE_LIST_ITEM_SINGLETON(gpu_elem);
    }

    /* DtoD copy, if data is read only, then we go back to CPU copy, and fetch data from CPU (HtoD) */
    if( (in_elem != original->device_copies[0]) &&
        (in_elem->version == original->device_copies[0]->version) ) {
        /* We should never enter here, as long as we don't foward the GPU data to the input */
        assert(0);
        dague_data_copy_release(in_elem);  /* release the copy in GPU1 */
        task_data->data_in = original->device_copies[0];
        in_elem = task_data->data_in;
        OBJ_RETAIN(in_elem);  /* retain the corresponding CPU copy */
    }

    transfer_from = dague_data_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);
    gpu_device->super.required_data_in += original->nb_elts;
    if( -1 != transfer_from ) {
        cudaError_t status;

        DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                              "GPU:\tMove H2D data %x (H %p:D %p) %d bytes to GPU %d",
                              original->key, in_elem->device_private, (void*)gpu_elem->device_private, original->nb_elts, gpu_device->cuda_index);

        assert( gpu_elem->version <= in_elem->version );
        assert( gpu_elem->version != in_elem->version || gpu_elem->data_transfer_status == DATA_STATUS_NOT_TRANSFER );

#if defined(DAGUE_PROF_TRACE)
        if( gpu_stream->prof_event_track_enable ) {
            dague_execution_context_t *this_task = gpu_task->ec;

            assert(-1 != gpu_stream->prof_event_key_start);
            DAGUE_PROFILING_TRACE(gpu_stream->profiling,
                                  gpu_stream->prof_event_key_start,
                                  this_task->function->key(this_task->dague_handle, this_task->locals),
                                  this_task->dague_handle->handle_id,
                                  &original);
        }
#endif

        /* Push data into the GPU */
        status = (cudaError_t)cudaMemcpyAsync( gpu_elem->device_private,
                                               in_elem->device_private, original->nb_elts,
                                               cudaMemcpyHostToDevice,
                                               gpu_stream->cuda_stream );
        DAGUE_CUDA_CHECK_ERROR( "cudaMemcpyAsync to device ", status,
                                { dague_warning("<<%p>> -> <<%p>> [%d]", in_elem->device_private, gpu_elem->device_private, original->nb_elts);
                                    return -1; } );
        gpu_device->super.transferred_data_in += original->nb_elts;

        /* update the data version in GPU immediately, and mark the data under transfer */
        gpu_elem->version = in_elem->version;
        gpu_elem->data_transfer_status = DATA_STATUS_UNDER_TRANSFER;
        gpu_elem->push_task = gpu_task->ec;  /* only the task who does the transfer can modify the data status later. */
        /* TODO: take ownership of the data */
        return 1;
    }
    else {
        assert( gpu_elem->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER ||
                gpu_elem->data_transfer_status == DATA_STATUS_UNDER_TRANSFER);

#if defined(DAGUE_DEBUG_NOISIER)
        DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                              "GPU:\tNO PUSH from %p to %p, size %d\n",
                              in_elem->device_private, gpu_elem->device_private, original->nb_elts);
#endif
    }
    /* TODO: data keeps the same coherence flags as before */
    return 0;
}

void* dague_gpu_pop_workspace(gpu_device_t* gpu_device, dague_gpu_exec_stream_t* gpu_stream, size_t size)
{
    (void)gpu_device; (void)gpu_stream; (void)size;
    void *work = NULL;

#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
    if (gpu_stream->workspace == NULL) {
        gpu_stream->workspace = (dague_gpu_workspace_t *)malloc(sizeof(dague_gpu_workspace_t));
        gpu_stream->workspace->total_workspace = DAGUE_GPU_MAX_WORKSPACE;
        gpu_stream->workspace->stack_head = DAGUE_GPU_MAX_WORKSPACE - 1;

        for( int i = 0; i < DAGUE_GPU_MAX_WORKSPACE; i++ ) {
            gpu_stream->workspace->workspace[i] = zone_malloc( gpu_device->memory, size);
        }
    }
    assert (gpu_stream->workspace->stack_head >= 0);
    work = gpu_stream->workspace->workspace[gpu_stream->workspace->stack_head];
    gpu_stream->workspace->stack_head --;
#endif /* !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE) */
    return work;
}

int dague_gpu_push_workspace(gpu_device_t* gpu_device, dague_gpu_exec_stream_t* gpu_stream)
{
    (void)gpu_device; (void)gpu_stream;
#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
    gpu_stream->workspace->stack_head ++;
    assert (gpu_stream->workspace->stack_head < DAGUE_GPU_MAX_WORKSPACE);
#endif /* !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE) */
    return 0;
}

int dague_gpu_free_workspace(gpu_device_t * gpu_device)
{
    (void)gpu_device;
#if !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE)
    int i, j;
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        dague_gpu_exec_stream_t *gpu_stream = &(gpu_device->exec_stream[i]);
        if (gpu_stream->workspace != NULL) {
            for (j = 0; j < gpu_stream->workspace->total_workspace; j++) {
                zone_free( gpu_device->memory, gpu_stream->workspace->workspace[j] );
            }
            free(gpu_stream->workspace);
            gpu_stream->workspace = NULL;
        }
    }
#endif /* !defined(DAGUE_GPU_CUDA_ALLOC_PER_TILE) */
    return 0;
}

static inline int dague_gpu_check_space_needed(gpu_device_t *gpu_device, dague_gpu_context_t *gpu_task)
{
    int i;
    int space_needed = 0;
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_data_t *original;
    dague_data_copy_t *data;
    const dague_flow_t *flow;

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        if(!(flow->flow_flags)) continue;

        data = this_task->data[i].data_in;
        original = data->original;
        if( NULL != DAGUE_DATA_GET_COPY(original, gpu_device->super.device_index) ) {
            continue;
        }
        if(flow->flow_flags & FLOW_ACCESS_READ)
            space_needed++;
    }
    return space_needed;
}

void dump_list(dague_list_t *list)
{
    dague_list_item_t *p = (dague_list_item_t *)list->ghost_element.list_next;
    while (p != &(list->ghost_element)) {
        p = (dague_list_item_t *)p->list_next;
    }
}


int dague_gpu_sort_pending_list(gpu_device_t *gpu_device)
{
    //dague_list_t *sort_list = &(gpu_device->pending);
    dague_list_t *sort_list = gpu_device->exec_stream[0].fifo_pending;
    int lock_required = 0;
    if (lock_required) {
        if ( !dague_atomic_trylock(&(sort_list->atomic_lock)) ) {
            return 0;
        }
    }
    if (dague_list_nolock_is_empty(sort_list) ) { /* list is empty */
        if (lock_required) {
            dague_atomic_unlock(&(sort_list->atomic_lock));
        }
        return 0;
    }

    if (gpu_device->sort_starting_p == NULL || !dague_list_nolock_contains(sort_list, gpu_device->sort_starting_p) ) {
        gpu_device->sort_starting_p = (dague_list_item_t*)sort_list->ghost_element.list_next;
    }

    /* p is head */
    dague_list_item_t *p = gpu_device->sort_starting_p;
    int i, j, NB_SORT = 10, space_q, space_min;

    dague_list_item_t *q, *prev_p, *min_p;
    for (i = 0; i < NB_SORT; i++) {
        if ( p == &(sort_list->ghost_element) ) {
            break;
        }
        min_p = p; /* assume the minimum one is the first one p */
        q = (dague_list_item_t*)min_p->list_next;
        space_min = dague_gpu_check_space_needed(gpu_device, (dague_gpu_context_t*)min_p);
        for (j = i+1; j < NB_SORT; j++) {
            if ( q == &(sort_list->ghost_element) ) {
                break;
            }
            space_q = dague_gpu_check_space_needed(gpu_device, (dague_gpu_context_t*)q);
            if ( space_min > space_q ) {
                min_p = q;
                space_min = space_q;
            }
            q = (dague_list_item_t*)q->list_next;

        }
        if (min_p != p) { /* minimum is not the first one, let's insert min_p before p */
            /* take min_p out */
            dague_list_item_ring_chop(min_p);
            DAGUE_LIST_ITEM_SINGLETON(min_p);
            prev_p = (dague_list_item_t*)p->list_prev;

            /* insert min_p after prev_p */
            dague_list_nolock_add_after( sort_list, prev_p, min_p);
        }
        p = (dague_list_item_t*)min_p->list_next;
    }

    if (lock_required) {
        dague_atomic_unlock(&(sort_list->atomic_lock));
    }
    return 0;
}

/**
 * Transfer at most the DAGUE_GPU_W2R_NB_MOVE_OUT oldest data from the GPU back
 * to main memory. Create a single task to move them all out, then switch the
 * GPU data copy in shared mode.
 */
dague_gpu_context_t* dague_gpu_create_W2R_task(gpu_device_t *gpu_device, dague_execution_unit_t *eu_context)
{
    dague_gpu_context_t *w2r_task = NULL;
    dague_execution_context_t *ec = NULL;
    dague_gpu_data_copy_t *gpu_copy;
    dague_data_t* original;
    dague_list_item_t* item = (dague_list_item_t*)gpu_device->gpu_mem_owned_lru.ghost_element.list_next;
    int nb_cleaned = 0;

    /* Find a data copy that has no pending users on the GPU, and can be
     * safely moved back on the main memory */
    while(nb_cleaned < DAGUE_GPU_W2R_NB_MOVE_OUT) {
        /* Break at the end of the list */
        if( item == &(gpu_device->gpu_mem_owned_lru.ghost_element) ) {
            break;
        }
        gpu_copy = (dague_gpu_data_copy_t*)item;
        original = gpu_copy->original;
        if( (0 != gpu_copy->readers) || (0 != original->device_copies[0]->readers) ) {
            item = (dague_list_item_t*)item->list_next;  /* conversion needed for volatile */
            continue;
        }
        if( NULL == ec ) {  /* allocate on-demand */
            ec = (dague_execution_context_t*)dague_thread_mempool_allocate(eu_context->context_mempool);
            ec->status = DAGUE_TASK_STATUS_NONE;
            if( NULL == ec )  /* we're running out of memory. Bail out. */
                break;
        }
        dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
        DAGUE_LIST_ITEM_SINGLETON(gpu_copy);
        gpu_copy->readers++;
        ec->data[nb_cleaned].data_out = gpu_copy;
        nb_cleaned++;
    }

    if( 0 == nb_cleaned )
        return NULL;

    w2r_task = (dague_gpu_context_t *)malloc(sizeof(dague_gpu_context_t));
    OBJ_CONSTRUCT(w2r_task, dague_list_item_t);
    ec->priority = INT32_MAX;
    ec->function = NULL;
    w2r_task->ec = ec;
    w2r_task->task_type = GPU_TASK_TYPE_D2HTRANSFER;
    return w2r_task;
}

/**
 * Complete a data copy transfer originated from the engine.
 */
int dague_gpu_W2R_task_fini(gpu_device_t *gpu_device,
                            dague_gpu_context_t *w2r_task,
                            dague_execution_unit_t *eu_context)
{
    dague_gpu_data_copy_t *gpu_copy, *cpu_copy;
    dague_execution_context_t *ec = w2r_task->ec;
    dague_data_t* original;
    int i;

    assert(w2r_task->task_type == GPU_TASK_TYPE_D2HTRANSFER);
    for( i = 0; (i < DAGUE_GPU_W2R_NB_MOVE_OUT) && (NULL != (gpu_copy = ec->data[i].data_out)); i++ ) {
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];
        cpu_copy->coherency_state =  DATA_COHERENCY_SHARED;
        cpu_copy->version = gpu_copy->version;
        DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                              "Mirror on CPU and move CUDA copy %p [ref_count %d] in LRU",
                              gpu_copy, gpu_copy->super.super.obj_reference_count);
        dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
        gpu_copy->readers--;
        assert(gpu_copy->readers >= 0);
    }
    dague_thread_mempool_free(eu_context->context_mempool, w2r_task->ec);
    free(w2r_task);
    return 0;
}


/**
 * Try to find the best device to execute the kernel based on the compute
 * capability of the card.
 *
 * Returns:
 *  > 1    - if the kernel should be executed by the a GPU
 *  0 or 1 - if the kernel should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1      - if the kernel is scheduled to be executed on a GPU.
 */

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int dague_gpu_get_best_device( dague_execution_context_t* this_task, double ratio )
{
    int i, dev_index, data_index = 0;
    dague_handle_t* handle = this_task->dague_handle;

    /* Step one: Find the first data in WRITE mode stored on a GPU */
    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        if( (NULL != this_task->function->out[i]) &&
            (this_task->function->out[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            data_index = this_task->function->out[i]->flow_index;
            dev_index  = this_task->data[data_index].data_in->original->owner_device;
            if (dev_index > 1) {
                break;
            }
        }
    }

    /* 0 is CPU, and 1 is recursive device */
    if( dev_index <= 1 ) {  /* This is the first time we see this data for a GPU.
                             * Let's decide which GPU will work on it. */
        int best_index = 0;  /* default value: first CPU device */
        float weight, best_weight = dague_device_load[0] + ratio * dague_device_sweight[0];

        /* Start at 2, to skip the recursive body */
        for( dev_index = 2; dev_index < dague_devices_enabled(); dev_index++ ) {
            /* Skip the device if it is not configured */
            if(!(handle->devices_mask & (1 << dev_index))) continue;
            weight = dague_device_load[dev_index] + ratio * dague_device_sweight[dev_index];
            if( best_weight > weight ) {
                best_index = dev_index;
                best_weight = weight;
            }
        }
        dague_device_load[best_index] += ratio * dague_device_sweight[best_index];
        assert( best_index != 1 );
        dev_index = best_index;
    }

    return dev_index;
}

#if DAGUE_GPU_USE_PRIORITIES

static inline dague_list_item_t* dague_fifo_push_ordered( dague_list_t* fifo,
                                                          dague_list_item_t* elem )
{
    dague_ulist_push_sorted(fifo, elem, dague_execution_context_priority_comparator);
    return elem;
}
#define DAGUE_FIFO_PUSH  dague_fifo_push_ordered
#else
#define DAGUE_FIFO_PUSH  dague_ulist_fifo_push
#endif

static inline int
progress_stream( gpu_device_t* gpu_device,
                 dague_gpu_exec_stream_t* exec_stream,
                 advance_task_function_t progress_fct,
                 dague_gpu_context_t* task,
                 dague_gpu_context_t** out_task )
{
    int saved_rc = 0, rc, i;
    *out_task = NULL;
     dague_execution_context_t *this_task;
     const dague_flow_t *flow;

    if( NULL != task ) {
        DAGUE_FIFO_PUSH(exec_stream->fifo_pending, (dague_list_item_t*)task);
        task = NULL;
    }
 grab_a_task:
    if( NULL == exec_stream->tasks[exec_stream->start] ) {
        /* get the best task */
        task = (dague_gpu_context_t*)dague_ulist_fifo_pop(exec_stream->fifo_pending);
    }
    if( NULL == task ) {
        /* No more room on the event list or no tasks. Keep moving */
        goto check_completion;
    }
    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)task);

    assert( NULL == exec_stream->tasks[exec_stream->start] );
    /**
     * In case the task is succesfully progressed, the corresponding profiling
     * event is triggered.
     */
    if ( NULL == progress_fct ) {
        /* Grab the submit function */
        progress_fct = task->submit;
#if defined(DAGUE_DEBUG_ENABLE)
        for( i = 0; i < task->ec->function->nb_flows; i++ ) {
            flow = task->flow[i];
            assert( flow );
            assert( flow->flow_index == i );
            if(!flow->flow_flags) continue;
            assert(task->ec->data[i].data_out->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER);
        }
#endif /* defined(DAGUE_DEBUG_ENABLE) */
    }
    assert( NULL != progress_fct );
    rc = progress_fct( gpu_device, task, exec_stream );
    if( 0 > rc ) {
        if( -1 == rc ) return -1;  /* Critical issue */
        /* No more room on the GPU. Push the task back on the queue and check the completion queue. */
        DAGUE_FIFO_PUSH(exec_stream->fifo_pending, (dague_list_item_t*)task);
        DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                              "GPU: Reschedule %s(task %p) priority %d: no room available on the GPU for data",
                              task->ec->function->name, (void*)task->ec, task->ec->priority);
        saved_rc = rc;  /* keep the info for the upper layer */
    } else {
        /**
         * Do not skip the cuda event generation. The problem is that some of the inputs
         * might be in the pipe of being transferred to the GPU. If we activate this task
         * too early, it might get executed before the data is available on the GPU.
         * Obviously, this lead to incorrect results.
         */
        rc = cudaEventRecord( exec_stream->events[exec_stream->start], exec_stream->cuda_stream );
        exec_stream->tasks[exec_stream->start] = task;
        exec_stream->start = (exec_stream->start + 1) % exec_stream->max_events;
#if defined(DAGUE_DEBUG_NOISIER)
        if( task->task_type == GPU_TASK_TYPE_D2HTRANSFER ) {
            DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                                  "GPU: Submitted Transfer(task %p) on stream %p",
                                  (void*)task->ec,
                                  (void*)exec_stream->cuda_stream);
        }
        else {
            DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                                  "GPU: Submitted %s(task %p) priority %d on stream %p",
                                  task->ec->function->name, (void*)task->ec, task->ec->priority,
                                  (void*)exec_stream->cuda_stream);
        }
#endif
    }
    task = NULL;

 check_completion:
    if( (NULL == *out_task) && (NULL != exec_stream->tasks[exec_stream->end]) ) {
        rc = cudaEventQuery(exec_stream->events[exec_stream->end]);
        if( CUDA_SUCCESS == rc ) {

            /**
             * Even though cuda event return success, the PUSH may not be
             * completed if no PUSH is required by this task and the PUSH is
             * actually done by another task, so we need to check if the data is
             * actually ready to use
             */
            if (exec_stream == &(gpu_device->exec_stream[0])) {  /* exec_stream[0] is the PUSH stream */
                dague_gpu_context_t *gtask = exec_stream->tasks[exec_stream->end];
                this_task = gtask->ec;
                for( i = 0; i < this_task->function->nb_flows; i++ ) {
                    flow = gtask->flow[i];
                    assert( flow );
                    assert( flow->flow_index == i );
                    if(!flow->flow_flags) continue;
                    if (this_task->data[i].data_out->push_task == this_task) {   /* only the task who did this PUSH can modify the status */
                        this_task->data[i].data_out->data_transfer_status = DATA_STATUS_COMPLETE_TRANSFER;
                        continue;
                    }
                    if (this_task->data[i].data_out->data_transfer_status != DATA_STATUS_COMPLETE_TRANSFER) {  /* data is not ready */
                        assert(0);
                        return saved_rc;
                    }
                }
            }

            /* Save the task for the next step */
            task = *out_task = exec_stream->tasks[exec_stream->end];
#if defined(DAGUE_DEBUG_NOISIER)
            if( task->task_type == GPU_TASK_TYPE_D2HTRANSFER ) {
                DAGUE_DEBUG_VERBOSE(19, dague_cuda_output_stream,
                                      "GPU: Completed Transfer(task %p) on stream %p",
                                      (void*)task->ec,
                                      (void*)exec_stream->cuda_stream);
            }
            else {
                DAGUE_DEBUG_VERBOSE(19, dague_cuda_output_stream,
                                      "GPU: Completed %s(task %p) priority %d on stream %p",
                                      task->ec->function->name, (void*)task->ec, task->ec->priority,
                                      (void*)exec_stream->cuda_stream);
            }
#endif
            exec_stream->tasks[exec_stream->end] = NULL;
            exec_stream->end = (exec_stream->end + 1) % exec_stream->max_events;
#if defined(DAGUE_PROF_TRACE)
            if( exec_stream->prof_event_track_enable ) {
                if( task->task_type == GPU_TASK_TYPE_D2HTRANSFER ) {
                    assert( exec_stream->prof_event_key_end == dague_cuda_moveout_key_end );
                    DAGUE_PROFILING_TRACE(exec_stream->profiling,
                                          exec_stream->prof_event_key_end,
                                          -1, 0, NULL);
                } else {
                    DAGUE_TASK_PROF_TRACE(exec_stream->profiling,
                                          (-1 == exec_stream->prof_event_key_end ?
                                           DAGUE_PROF_FUNC_KEY_END(task->ec->dague_handle,
                                                                   task->ec->function->function_id) :
                                           exec_stream->prof_event_key_end),
                                          task->ec);
                }
            }
#endif /* (DAGUE_PROF_TRACE) */
            task = NULL;  /* Try to schedule another task */
            goto grab_a_task;
        }
        if( cudaErrorNotReady != rc ) {
            DAGUE_CUDA_CHECK_ERROR( "(progress_stream) cudaEventQuery ", rc,
                                    {return -1;} );
        }
    }
    return saved_rc;
}

void dump_exec_stream(dague_gpu_exec_stream_t* exec_stream)
{
    char task_str[128];
    int i;

    dague_debug_verbose(0, dague_cuda_output_stream,
            "Dump GPU exec stream %p [events = %d, start = %d, end = %d, executed = %d]",
            exec_stream, exec_stream->max_events, exec_stream->start, exec_stream->end,
            exec_stream->executed);
    for( i = 0; i < exec_stream->max_events; i++ ) {
        if( NULL == exec_stream->tasks[i] ) continue;
        dague_debug_verbose(0, dague_cuda_output_stream,
            "    %d: %s", i, dague_snprintf_execution_context(task_str, 128, exec_stream->tasks[i]->ec));
    }
    /* Don't yet dump the fifo_pending queue */
}

void dump_GPU_state(gpu_device_t* gpu_device)
{
    int i;

    printf("\n\n");
    printf("Device %d:%d (%p)\n", gpu_device->cuda_index, gpu_device->super.device_index, gpu_device);
    printf("\tpeer mask %x executed tasks %llu max streams %d\n",
           gpu_device->peer_access_mask, (unsigned long long)gpu_device->super.executed_tasks, gpu_device->max_exec_streams);
    printf("\tstats transferred [in %llu out %llu] required [in %llu out %llu]\n",
           (unsigned long long)gpu_device->super.transferred_data_in, (unsigned long long)gpu_device->super.transferred_data_out,
           (unsigned long long)gpu_device->super.required_data_in, (unsigned long long)gpu_device->super.required_data_out);
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        dump_exec_stream(&gpu_device->exec_stream[i]);
    }
    if( !dague_ulist_is_empty(&gpu_device->gpu_mem_lru) ) {
        printf("#\n# LRU list\n#\n");
        i = 0;
        DAGUE_ULIST_ITERATOR(&gpu_device->gpu_mem_lru, item,
                            {
                                dague_gpu_data_copy_t* gpu_copy = (dague_gpu_data_copy_t*)item;
                                printf("  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                dague_dump_data_copy(gpu_copy);
                                i++;
                            });
    };
    if( !dague_ulist_is_empty(&gpu_device->gpu_mem_owned_lru) ) {
        printf("#\n# Owned LRU list\n#\n");
        i = 0;
        DAGUE_ULIST_ITERATOR(&gpu_device->gpu_mem_owned_lru, item,
                            {
                                dague_gpu_data_copy_t* gpu_copy = (dague_gpu_data_copy_t*)item;
                                printf("  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                dague_dump_data_copy(gpu_copy);
                                i++;
                            });
    };
    printf("\n\n");
}

/**
 *  This function schedule the move of all the data required for a
 *  specific task from the main memory into the GPU memory.
 *
 *  Returns:
 *     a positive number: the number of data to be moved.
 *     -1: data cannot be moved into the GPU.
 *     -2: No more room on the GPU to move this data.
 */
int
dague_gpu_kernel_push( gpu_device_t            *gpu_device,
                       dague_gpu_context_t     *gpu_task,
                       dague_gpu_exec_stream_t *gpu_stream)
{
    int i, ret = 0;
    dague_execution_context_t *this_task = gpu_task->ec;
    const dague_flow_t        *flow;

    /**
     * First, let's reserve enough space on the device to transfer the data on the GPU.
     */
    ret = dague_gpu_data_reserve_device_space( gpu_device,
                                               gpu_task );
    if( ret < 0 ) {
        goto release_and_return_error;
    }

    /**
     * Second, We have enough space, let's schedule the required transfer
     */
    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_stream->profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        /* Skip CTL flows */
        if(!(flow->flow_flags)) continue;

        assert( NULL != dague_data_copy_get_ptr(this_task->data[i].data_in) );

        DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                            "GPU[%1d]:\tIN  Data of %s <%x> on GPU\n",
                            gpu_device->cuda_index, flow->name,
                            this_task->data[i].data_out->original->key);
        ret = dague_gpu_data_stage_in( gpu_device, flow->flow_flags,
                                       &(this_task->data[i]), gpu_task, gpu_stream );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

  release_and_return_error:
    return ret;
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
int
dague_gpu_kernel_pop( gpu_device_t            *gpu_device,
                      dague_gpu_context_t     *gpu_task,
                      dague_gpu_exec_stream_t *gpu_stream)
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_gpu_data_copy_t     *gpu_copy;
    dague_data_t              *original;
    const dague_flow_t        *flow;
    int return_code = 0, how_many = 0, i;
    cudaError_t status;

    if (gpu_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        for( i = 0; i < 1; i++ ) {
            gpu_copy = this_task->data[i].data_out;
            original = gpu_copy->original;
            status = cudaMemcpyAsync( original->device_copies[0]->device_private,
                                      gpu_copy->device_private,
                                      original->nb_elts,
                                      cudaMemcpyDeviceToHost,
                                      gpu_stream->cuda_stream );
            DAGUE_CUDA_CHECK_ERROR( "cudaMemcpyAsync from device ", status,
                                    { dague_warning("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
                                               gpu_copy->device_private, original->device_copies[0]->device_private);
                                        return_code = -2;
                                        goto release_and_return_error;} );
        }
        return return_code;
    }

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;

        flow = gpu_task->flow[i];

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        assert(original == this_task->data[i].data_in->original);

        if( !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Do not propagate GPU copies to successors (temporary solution) */
            this_task->data[i].data_out = original->device_copies[0];
        }

        if( flow->flow_flags & FLOW_ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
                DAGUE_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
                continue;  /* done with this element, go for the next one */
            }
        }
        if( flow->flow_flags & FLOW_ACCESS_WRITE ) {
            assert( gpu_copy == dague_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );

            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += original->nb_elts;
            assert( ((dague_list_item_t*)gpu_copy)->list_next == (dague_list_item_t*)gpu_copy );
            assert( ((dague_list_item_t*)gpu_copy)->list_prev == (dague_list_item_t*)gpu_copy );

            DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                                "GPU[%1d]:\tOUT Data of %s\n", gpu_device->cuda_index, flow->name);

            if( gpu_task->pushout[i] ) {
                /* TODO: make sure no readers are working on the CPU version */
                original = gpu_copy->original;
                DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                                    "GPU:\tMove D2H data <%s:%x> from GPU %d %p -> %p requested\n",
                                    flow->name, original->key, gpu_device->cuda_index,
                                    (void*)gpu_copy->device_private, original->device_copies[0]->device_private);
                DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_stream->profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                                    this_task->function->function_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                /* Move the data back into main memory */
                status = cudaMemcpyAsync( original->device_copies[0]->device_private,
                                          gpu_copy->device_private,
                                          original->nb_elts,
                                          cudaMemcpyDeviceToHost,
                                          gpu_stream->cuda_stream );
                DAGUE_CUDA_CHECK_ERROR( "cudaMemcpyAsync from device ", status,
                                        { dague_warning("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
                                                   gpu_copy->device_private, original->device_copies[0]->device_private);
                                            return_code = -2;
                                            goto release_and_return_error;} );
                gpu_device->super.transferred_data_out += original->nb_elts; /* TODO: not hardcoded, use datatype size */
                how_many++;
            }
        }
    }

  release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
int
dague_gpu_kernel_epilog( gpu_device_t        *gpu_device,
                         dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    dague_data_t              *original;
    int i;

    for( i = 0; i < this_task->function->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;

        if( !(gpu_task->flow[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flow has been overwritten in pop */
            continue;
        }

        cpu_copy = original->device_copies[0];

        /**
         * There might be a race condition here. We can't assume the first CPU
         * version is the corresponding CPU copy, as a new CPU-bound data
         * might have been created meanwhile.
         */
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        cpu_copy->coherency_state = DATA_COHERENCY_SHARED;

        /**
         *  The cpu_copy will be updated in the completion, and at that moment
         *  the two versions will be identical.
         */
        cpu_copy->version = gpu_copy->version;
        gpu_copy->version++;  /* on to the next version */

        /**
         * Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;

        if( gpu_task->pushout[i] ) {
            dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
            DAGUE_DEBUG_VERBOSE(20, dague_cuda_output_stream,
                                "CUDA copy %p [ref_count %d] moved to the read LRU in %s\n",
                                gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
        } else {
            dague_ulist_fifo_push(&gpu_device->gpu_mem_owned_lru, (dague_list_item_t*)gpu_copy);
        }
    }
    return 0;
}

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
dague_hook_return_t
dague_gpu_kernel_scheduler( dague_execution_unit_t *eu_context,
                            dague_gpu_context_t    *gpu_task,
                            int which_gpu )
{
    gpu_device_t* gpu_device;
    cudaError_t status;
    int rc, exec_stream = 0;
    dague_gpu_context_t *progress_task, *out_task_push, *out_task_submit, *out_task_pop;
#if defined(DAGUE_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    gpu_device = (gpu_device_t*)dague_devices_get(which_gpu);

#if defined(DAGUE_PROF_TRACE)
    DAGUE_PROFILING_TRACE_FLAGS( eu_context->eu_profile,
                                 DAGUE_PROF_FUNC_KEY_END(gpu_task->ec->dague_handle,
                                                         gpu_task->ec->function->function_id),
                                 gpu_task->ec->function->key( gpu_task->ec->dague_handle, gpu_task->ec->locals),
                                 gpu_task->ec->dague_handle->handle_id, NULL,
                                 DAGUE_PROFILING_EVENT_RESCHEDULED );
#endif /* defined(DAGUE_PROF_TRACE) */

    /* Check the GPU status */
    rc = dague_atomic_inc_32b( &(gpu_device->mutex) );
    if( 1 != rc ) {  /* I'm not the only one messing with this GPU */
        dague_fifo_push( &(gpu_device->pending), (dague_list_item_t*)gpu_task );
        return DAGUE_HOOK_RETURN_ASYNC;
    }

#if defined(DAGUE_PROF_TRACE)
    if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
        DAGUE_PROFILING_TRACE( eu_context->eu_profile, dague_cuda_own_GPU_key_start,
                               (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

    status = cudaSetDevice( gpu_device->cuda_index );
    DAGUE_CUDA_CHECK_ERROR( "(dague_gpu_kernel_scheduler) cudaSetDevice ", status,
                            {return DAGUE_HOOK_RETURN_DISABLE;} );

 check_in_deps:
    if( NULL != gpu_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream,
                            "GPU[%1d]:\tUpload data (if any) for %s priority %d",
                            gpu_device->cuda_index,
                            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                            gpu_task->ec->priority );
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[0]),
                          dague_gpu_kernel_push,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    gpu_task = progress_task;
    out_task_push = progress_task;

    /* Stage-in completed for this task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != gpu_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tExecute %s priority %d", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                 gpu_task->ec->priority );
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[2+exec_stream]),
                          NULL,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    gpu_task = progress_task;
    out_task_submit = progress_task;

    /* This task has completed its execution: we have to check if we schedule DtoN */
    if( NULL != gpu_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tRetrieve data (if any) for %s priority %d", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                 gpu_task->ec->priority );
    }
    if (out_task_submit == NULL && out_task_push == NULL) {
        gpu_task = dague_gpu_create_W2R_task(gpu_device, eu_context);
    }
    /* Task is ready to move the data back to main memory */
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[1]),
                          dague_gpu_kernel_pop,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    if( NULL != progress_task ) {
        /* We have a succesfully completed task. However, it is not gpu_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the progress_task.
         */
        gpu_task = progress_task;
        progress_task = NULL;
        goto complete_task;
    }
    gpu_task = progress_task;
    out_task_pop = progress_task;

 fetch_task_from_shared_queue:
    assert( NULL == gpu_task );
    if (out_task_submit == NULL && out_task_pop == NULL) {
        dague_gpu_sort_pending_list(gpu_device);
    }
    gpu_task = (dague_gpu_context_t*)dague_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != gpu_task ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tGet from shared queue %s priority %d", gpu_device->cuda_index,
                 dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                 gpu_task->ec->priority );
    }
    goto check_in_deps;

 complete_task:
    assert( NULL != gpu_task );
    DAGUE_DEBUG_VERBOSE(10, dague_debug_output,  "GPU[%1d]:\tComplete %s priority %d", gpu_device->cuda_index,
             dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, gpu_task->ec),
             gpu_task->ec->priority );
    /* Everything went fine so far, the result is correct and back in the main memory */
    DAGUE_LIST_ITEM_SINGLETON(gpu_task);
    if (gpu_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        dague_gpu_W2R_task_fini(gpu_device, gpu_task, eu_context);
        gpu_task = progress_task;
        goto fetch_task_from_shared_queue;
    }
    dague_gpu_kernel_epilog( gpu_device, gpu_task );
    __dague_complete_execution( eu_context, gpu_task->ec );
    dague_device_load[gpu_device->super.device_index] -= dague_device_sweight[gpu_device->super.device_index];
    gpu_device->super.executed_tasks++;
    free( gpu_task );
    rc = dague_atomic_dec_32b( &(gpu_device->mutex) );
    if( 0 == rc ) {  /* I was the last one */
#if defined(DAGUE_PROF_TRACE)
        if( dague_cuda_trackable_events & DAGUE_PROFILE_CUDA_TRACK_OWN )
            DAGUE_PROFILING_TRACE( eu_context->eu_profile, dague_cuda_own_GPU_key_end,
                                   (unsigned long)eu_context, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(DAGUE_PROF_TRACE) */

        return DAGUE_HOOK_RETURN_ASYNC;
    }
    gpu_task = progress_task;
    goto fetch_task_from_shared_queue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    printf("Critical issue related to the GPU discovered. Giving up\n");
    return DAGUE_HOOK_RETURN_DISABLE;
}

#endif /* HAVE_CUDA */
