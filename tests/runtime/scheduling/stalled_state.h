/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _stalled_state_h
#define _stalled_state_h

#include <stdint.h>

/* What the tasks of stalled.jdf share with the main program */
extern int32_t stalled_nb_done;            /* WORK tasks that completed */
extern int32_t stalled_nb_parked;          /* Worker threads that wait inside a WORK task */
extern int32_t stalled_released;           /* Set once every WORK task is completed or has a thread waiting in it */
extern int32_t stalled_nb_done_by_master;  /* WORK tasks that thread 0 completed */
extern char   *stalled_has_parked;         /* For each thread, whether it waited already */

#endif
