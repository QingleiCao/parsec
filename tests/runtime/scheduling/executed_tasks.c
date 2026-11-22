/*
 * Copyright (c) 2026      Stony Brook University. All rights reserved.
 */

/* The number of executed tasks a device reports is exact once the tasks are
 * complete, whichever streams executed them: the same taskpool gives the same
 * count every time it runs, that count is the number of its tasks, and a
 * reset brings it back to zero.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "parsec/mca/device/device.h"
#include "ep_wrapper.h"
#include "schedmicro_data.h"
#include "tests/tests_runtime.h"

#define NT       512
#define LEVELS     8
#define ROUNDS     5
/* INIT, the NT x LEVELS tasks it releases, and room for the tasks the runtime
 * adds to start a taskpool */
#define MIN_TASKS (1 + NT * LEVELS)
#define MAX_TASKS (MIN_TASKS + 16)

/* The executed tasks are the first of the arrays of the statistics, with one
 * entry per device. */
static uint64_t executed_tasks(uint32_t device_index)
{
    uint64_t *stats = NULL, count;
    parsec_devices_save_statistics(&stats);
    count = stats[device_index];
    parsec_devices_free_statistics(&stats);
    return count;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_data_collection_t *dcA;
    parsec_taskpool_t *ep;
    int rank, world, rc, errors = 0;
    uint32_t cpu = UINT32_MAX;
    uint64_t before, count, first = 0;

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    for(uint32_t i = 0; i < parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( (NULL != device) && (PARSEC_DEV_CPU == device->type) ) {
            cpu = i;
            break;
        }
    }
    if( UINT32_MAX == cpu ) {
        fprintf(stderr, "No CPU device\n");
        return EXIT_FAILURE;
    }

    dcA = create_and_distribute_data(rank, world, NT, 1);
    parsec_data_collection_set_key(dcA, "A");

    for(int round = 0; round < ROUNDS; round++) {
        before = executed_tasks(cpu);

        ep = ep_new(dcA, NT, LEVELS);
        rc = parsec_context_add_taskpool(parsec, ep);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        ep_free(ep);

        count = executed_tasks(cpu) - before;
        if( 0 == round ) {
            first = count;
            if( (1 == world) && ((count < MIN_TASKS) || (count > MAX_TASKS)) ) {
                fprintf(stderr, "Round 0: %"PRIu64" tasks counted on the CPU device, expected between %d and %d\n",
                        count, MIN_TASKS, MAX_TASKS);
                errors++;
            }
        } else if( count != first ) {
            fprintf(stderr, "Round %d: %"PRIu64" tasks counted on the CPU device, %"PRIu64" in round 0\n",
                    round, count, first);
            errors++;
        }
    }

    parsec_mca_device_reset_statistics(parsec);
    for(uint32_t i = 0; i < parsec_nb_devices; i++) {
        if( NULL == parsec_mca_device_get(i) ) continue;
        if( 0 != (count = executed_tasks(i)) ) {
            fprintf(stderr, "Device %u still counts %"PRIu64" tasks after a reset\n", i, count);
            errors++;
        }
    }

    if( 0 == errors ) {
        printf("%"PRIu64" tasks counted on the CPU device in each of %d rounds\n", first, ROUNDS);
    }

    free_data(dcA);
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return (0 == errors) ? EXIT_SUCCESS : EXIT_FAILURE;
}
