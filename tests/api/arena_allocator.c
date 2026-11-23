/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * parsec_arena_data_allocate and parsec_arena_data_free, when both are set,
 * are the allocator pair of the arenas constructed afterwards, and each arena
 * keeps the pair it was constructed with.
 */

#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_internal.h"
#include "parsec/datatype.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ELEM_SIZE 4096

static int32_t nb_allocated = 0;
static int32_t nb_freed = 0;

static void *counting_allocate(size_t size)
{
    parsec_atomic_fetch_inc_int32(&nb_allocated);
    return malloc(size);
}

static void counting_free(void *ptr)
{
    parsec_atomic_fetch_inc_int32(&nb_freed);
    free(ptr);
}

static int errors = 0;

#define EXPECT(cond, ...) do {                     \
        if( !(cond) ) {                            \
            fprintf(stderr, "Error: " __VA_ARGS__); \
            fprintf(stderr, "\n");                 \
            errors++;                              \
        }                                          \
    } while(0)

static parsec_data_copy_t *new_copy(parsec_arena_t *arena, size_t count)
{
    parsec_data_copy_t *copy = parsec_arena_get_new_copy(arena, count, 0, parsec_datatype_uint8_t);
    void *ptr = (NULL != copy) ? PARSEC_DATA_COPY_GET_PTR(copy) : NULL;
    if( NULL != ptr ) {
        memset(ptr, 0x5a, count * ELEM_SIZE);  /* the memory must be usable */
    }
    return copy;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    parsec_arena_t *chosen, *plain, *half;
    parsec_data_copy_t *copies[6];
    int rc;

    rc = parsec_tests_context_init(1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    /* An arena constructed while the pair is set uses it ... */
    parsec_arena_data_allocate = counting_allocate;
    parsec_arena_data_free     = counting_free;
    chosen = PARSEC_OBJ_NEW(parsec_arena_t);
    parsec_arena_construct(chosen, ELEM_SIZE, PARSEC_ARENA_ALIGNMENT_SSE);

    /* ... one constructed with only half of it does not ... */
    parsec_arena_data_free = NULL;
    half = PARSEC_OBJ_NEW(parsec_arena_t);
    parsec_arena_construct(half, ELEM_SIZE, PARSEC_ARENA_ALIGNMENT_SSE);

    /* ... nor does one constructed once it is unset. */
    parsec_arena_data_allocate = NULL;
    plain = PARSEC_OBJ_NEW(parsec_arena_t);
    parsec_arena_construct(plain, ELEM_SIZE, PARSEC_ARENA_ALIGNMENT_SSE);

    /* The first arena keeps the pair it was constructed with. Single
     * elements come from its free list, several at once straight from the
     * allocator. */
    copies[0] = new_copy(chosen, 1);
    copies[1] = new_copy(chosen, 1);
    copies[2] = new_copy(chosen, 3);
    EXPECT((NULL != copies[0]) && (NULL != copies[1]) && (NULL != copies[2]),
           "the arena constructed with the pair set failed to allocate");
    EXPECT(3 == nb_allocated, "the arena constructed with the pair set allocated %d time(s) with it, expected 3",
           nb_allocated);

    copies[3] = new_copy(half, 1);
    copies[4] = new_copy(plain, 1);
    copies[5] = new_copy(plain, 2);
    EXPECT((NULL != copies[3]) && (NULL != copies[4]) && (NULL != copies[5]),
           "an arena constructed without the pair failed to allocate");
    EXPECT(3 == nb_allocated, "an arena constructed without the pair allocated with it (%d allocations, expected 3)",
           nb_allocated);

    for( int i = 0; i < 6; i++ ) {
        if( NULL != copies[i] ) PARSEC_OBJ_RELEASE(copies[i]);
    }
    /* The block of three went back to the allocator, the single elements to
     * the free list. */
    EXPECT(1 == nb_freed, "releasing the copies freed %d block(s) with the pair, expected 1", nb_freed);

    PARSEC_OBJ_RELEASE(chosen);
    PARSEC_OBJ_RELEASE(half);
    PARSEC_OBJ_RELEASE(plain);
    EXPECT(nb_allocated == nb_freed, "%d block(s) allocated with the pair, but %d freed with it",
           nb_allocated, nb_freed);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    if( errors ) fprintf(stderr, "arena_allocator: %d error(s)\n", errors);
    return (0 == errors) ? EXIT_SUCCESS : EXIT_FAILURE;
}
