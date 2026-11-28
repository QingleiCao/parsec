/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "qpu_adapter_spawn.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

static double spawn_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void spawn_nap(void)
{
    const struct timespec ts = { .tv_sec = 0, .tv_nsec = 10000000 };
    nanosleep(&ts, NULL);
}

int qpu_adapter_dial(const char *socket_path)
{
    struct sockaddr_un sun;
    int fd;

    if( strlen(socket_path) >= sizeof(sun.sun_path) ) return -1;
    memset(&sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
    strcpy(sun.sun_path, socket_path);
    if( (fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0 ) return -1;
    if( 0 != connect(fd, (struct sockaddr*)&sun, sizeof(sun)) ) {
        close(fd);
        return -1;
    }
    return fd;
}

int qpu_adapter_listens(const char *socket_path)
{
    const int fd = qpu_adapter_dial(socket_path);
    if( fd < 0 ) return 0;
    close(fd);
    return 1;
}

static int spawn_status(int status)
{
    if( WIFEXITED(status) ) return WEXITSTATUS(status);
    if( WIFSIGNALED(status) ) return 128 + WTERMSIG(status);
    return -1;
}

pid_t qpu_adapter_spawn(const char *socket_path, char *const argv[], double wait_seconds)
{
    const double give_up = spawn_now() + wait_seconds;
    pid_t pid;

    unlink(socket_path);                            /* what an earlier run may have left */
    fflush(NULL);
    if( (pid = fork()) < 0 ) {
        fprintf(stderr, "qpu_adapter_spawn: fork: %s\n", strerror(errno));
        return -1;
    }
    if( 0 == pid ) {
#if defined(__linux__)
        prctl(PR_SET_PDEATHSIG, SIGTERM);           /* not left behind by a test that is killed */
#endif
        execvp(argv[0], argv);
        fprintf(stderr, "qpu_adapter_spawn: cannot run %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    while( !qpu_adapter_listens(socket_path) ) {
        int status;
        if( pid == waitpid(pid, &status, WNOHANG) ) {
            fprintf(stderr, "qpu_adapter_spawn: %s ended (%d) before it listened on %s\n", argv[0], spawn_status(status), socket_path);
            return -1;
        }
        if( spawn_now() > give_up ) {
            fprintf(stderr, "qpu_adapter_spawn: %s did not listen on %s within %g s\n", argv[0], socket_path, wait_seconds);
            (void)qpu_adapter_stop(pid, 2.0);
            return -1;
        }
        spawn_nap();
    }
    return pid;
}

int qpu_adapter_stop(pid_t pid, double wait_seconds)
{
    const double give_up = spawn_now() + wait_seconds;
    int status, killed = 0;

    if( pid <= 0 ) return -1;                       /* kill() of 0 or -1 is not one process */
    kill(pid, SIGTERM);
    for(;;) {
        const pid_t rc = waitpid(pid, &status, WNOHANG);
        if( rc == pid ) return spawn_status(status);
        if( (rc < 0) && (EINTR != errno) ) return -1;
        if( !killed && (spawn_now() > give_up) ) {
            kill(pid, SIGKILL);
            killed = 1;
        }
        spawn_nap();
    }
}
