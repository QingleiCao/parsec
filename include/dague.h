/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_H_HAS_BEEN_INCLUDED
#define DAGUE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

typedef struct dague_object              dague_object_t;
typedef struct dague_execution_context_t dague_execution_context_t;
/**< The general context that holds all the threads of dague for this MPI process */
typedef struct dague_context_t           dague_context_t;

/**
 * TO BE REMOVED.
 */
typedef void* (*dague_data_allocate_t)(size_t matrix_size);
typedef void (*dague_data_free_t)(void *data);
extern dague_data_allocate_t dague_data_allocate;
extern dague_data_free_t     dague_data_free;

/**
 * CONTEXT MANIPULATION FUNCTIONS.
 */

/**
 * The completion callback of a dague_handle. Once the handle has been
 * completed, i.e. all the local tasks associated with the handle have
 * been executed, and before the handle is marked as done, this callback
 * will be triggered. Inside the callback the handle should not be
 * modified.
 */
typedef int (*dague_completion_cb_t)(dague_object_t* dague_object, void*);

/**
 * Create a new execution context, using the number of resources passed
 * with the arguments. Every execution happend in the context of such an
 * execution context. Several contextes can cohexist on disjoint resources
 * in same time.
 */
dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[]);

/**
 * Complete all pending operations on the execution context, and release
 * all associated resources. Threads and acclerators attached to this
 * context will be released.
 */
int dague_fini( dague_context_t** pcontext );

/**
 * Compose two dague_objects into a composition dague_object. 
 * The resulting compound dague_object is returned. 
 * If start is already a composition, the same (updated) object may be returned.
 */
dague_object_t* dague_compose( dague_object_t* start, dague_object_t* next);

/**
 * Attach an execution handle on a context, in other words on the set of
 * resources associated to this particular context. This operation will
 * define if accelerators can be used for the execution.
 */
int dague_enqueue( dague_context_t* context, dague_object_t* object);

/**
 * Progress the execution context until no further operations are available.
 * Upon return from this function, all resources (threads and acclerators)
 * associated with the corresponding context are put in a mode where they
 * are not active.
 */
int dague_progress(dague_context_t* context);

/**
 * HANDLE MANIPULATION FUNCTIONS.
 */

/* Accessors to set and get the completion callback and data */
int dague_set_complete_callback( dague_object_t* dague_object,
                                 dague_completion_cb_t complete_cb, void* complete_data );
int dague_get_complete_callback( const dague_object_t* dague_object,
                                 dague_completion_cb_t* complete_cb, void** complete_data );

/**< Retrieve the local object attached to a unique object id */
dague_object_t* dague_object_lookup( uint32_t object_id );
/**< Register the object with the engine. Create the unique identifier for the object */
int dague_object_register( dague_object_t* object );
/**< Unregister the object with the engine. */
void dague_object_unregister( dague_object_t* object );

/**< Print DAGuE usage message */
void dague_usage(void);

/**
 * Allow to change the default priority of an object. It returns the
 * old priority (the default priorityy of an object is 0). This function
 * can be used during the lifetime of an object, however, only tasks
 * generated after this call will be impacted.
 */
int32_t dague_set_priority( dague_object_t* object, int32_t new_priority );

/* Dump functions */
struct dague_function_s;
struct assignment;
char* dague_snprintf_execution_context( char* str, size_t size,
                                        const dague_execution_context_t* task);
char* dague_snprintf_assignments( char* str, size_t size,
                                  const struct dague_function_s* function,
                                  const struct assignment* locals);

#endif  /* DAGUE_H_HAS_BEEN_INCLUDED */
