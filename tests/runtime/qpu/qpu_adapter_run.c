/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * Run a command with an adapter beside it, for a test:
 *
 *     qpu_adapter_run [-w SECONDS] -s SOCKET ADAPTER [ARG ...] -- COMMAND [ARG ...]
 *
 * starts ADAPTER, waits until it listens on SOCKET (for at most SECONDS, 60
 * by default), runs COMMAND, and stops the adapter when the command has
 * ended. The exit code is that of the command; when the command ended well,
 * that of the adapter, so that an adapter that saw its client do something
 * wrong fails the test (95, with the adapter's code on stderr).
 *
 * The adapter is given its socket by its own arguments: SOCKET here is only
 * where this program looks for it.
 */

#include "qpu_adapter_spawn.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[])
{
    const char *path = NULL;
    double wait_seconds = 60.0;
    int first = 1, split = -1, status, rc, adapter_rc;
    pid_t adapter, command;

    while( (first + 1 < argc) && ('-' == argv[first][0]) && ('\0' != argv[first][1]) && ('\0' == argv[first][2]) ) {
        if( 's' == argv[first][1] )      path = argv[first + 1];
        else if( 'w' == argv[first][1] ) wait_seconds = atof(argv[first + 1]);
        else break;
        first += 2;
    }
    for( int i = first; i < argc; i++ ) {
        if( 0 == strcmp(argv[i], "--") ) { split = i; break; }
    }
    if( (NULL == path) || (split <= first) || (split + 1 >= argc) ) {
        fprintf(stderr, "usage: %s [-w seconds] -s <socket> <adapter> [arg ...] -- <command> [arg ...]\n", argv[0]);
        return 2;
    }
    argv[split] = NULL;

    if( (adapter = qpu_adapter_spawn(path, &argv[first], wait_seconds)) < 0 ) return 97;

    fflush(NULL);
    if( (command = fork()) < 0 ) {
        fprintf(stderr, "qpu_adapter_run: fork: %s\n", strerror(errno));
        (void)qpu_adapter_stop(adapter, 10.0);
        return 97;
    }
    if( 0 == command ) {
        execvp(argv[split + 1], &argv[split + 1]);
        fprintf(stderr, "qpu_adapter_run: cannot run %s: %s\n", argv[split + 1], strerror(errno));
        _exit(127);
    }
    while( (command != waitpid(command, &status, 0)) ) {
        if( EINTR != errno ) { status = -1; break; }
    }
    rc = (-1 == status) ? 97 : WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);

    adapter_rc = qpu_adapter_stop(adapter, 10.0);
    unlink(path);
    if( (0 == rc) && (0 != adapter_rc) ) {
        fprintf(stderr, "qpu_adapter_run: the command ended well, and the adapter ended with %d\n", adapter_rc);
        return 95;
    }
    return rc;
}
