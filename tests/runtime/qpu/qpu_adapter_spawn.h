/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef QPU_ADAPTER_SPAWN_H_HAS_BEEN_INCLUDED
#define QPU_ADAPTER_SPAWN_H_HAS_BEEN_INCLUDED

/**
 * Start and stop an adapter for a test: a program that listens on a
 * Unix-domain socket. Nothing here is specific to one adapter.
 */

#include <sys/types.h>

/** Start `argv` and wait until something accepts connections on
 *  socket_path, for at most wait_seconds. Returns the process, or -1 with a
 *  message on stderr when the program could not be started, ended, or did
 *  not listen in time (it is stopped then). The program ends with its
 *  parent, should the parent be killed. */
pid_t qpu_adapter_spawn(const char *socket_path, char *const argv[], double wait_seconds);

/** Ask a process to end (SIGTERM), wait for it for at most wait_seconds and
 *  kill it after that. Returns its exit code, 128 plus the number of the
 *  signal that ended it, or -1 when it could not be waited for. */
int qpu_adapter_stop(pid_t pid, double wait_seconds);

/** Whether something accepts connections on socket_path now. */
int qpu_adapter_listens(const char *socket_path);

/** Connect to an adapter as a client that is not PaRSEC. Returns the socket or -1. */
int qpu_adapter_dial(const char *socket_path);

#endif  /* QPU_ADAPTER_SPAWN_H_HAS_BEEN_INCLUDED */
