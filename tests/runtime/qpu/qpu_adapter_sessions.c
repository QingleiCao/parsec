/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * What an adapter keeps for a client whose connection is gone, asked by a
 * client that speaks the protocol itself: neither the adapter backend nor
 * the runtime is used, and no job is submitted. The adapter runs already
 * (tools/qpu/parsec_qpu_adapter_qiskit.py, started with --idle-drop).
 *
 * A session is what one client has at the adapter: its jobs, and the ends
 * it has not heard. It outlives a connection, so that a client that lost
 * its connection finds its jobs again; and the client is told whether the
 * session was there, because an adapter that does not have it is one that
 * was started again, whose jobs are gone.
 *
 *   - A session the adapter never saw is said to be new, and a session it
 *     has is said to be resumed.
 *   - A session two connections use, one of which closes, is kept for as
 *     long as the other uses it, however long that is.
 *   - A session nobody uses is still there for a client that comes back
 *     within the time the adapter was told, is dropped after that time, and
 *     is then a new one to whoever asks for it.
 *
 * Options:
 *   -a PATH      the socket of the adapter (required)
 *   -i SECONDS   the --idle-drop the adapter was started with (required)
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"
#include "qpu_adapter_spawn.h"

#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>

static int nb_errors = 0;

#define EXPECT(COND, ...)                                                   \
    do {                                                                    \
        if( !(COND) ) {                                                     \
            fprintf(stderr, "%s:%d: ", __func__, __LINE__);                 \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            nb_errors++;                                                    \
        }                                                                   \
    } while(0)

static const char *path = NULL;
static int next_seq = 1;

static void pause_for(double seconds)
{
    struct timespec ts;
    ts.tv_sec  = (time_t)seconds;
    ts.tv_nsec = (long)((seconds - (double)ts.tv_sec) * 1e9);
    while( (0 != nanosleep(&ts, &ts)) ) { /* interrupted: the rest */ }
}

/* One request and its answer, which stays in `line`. Returns 0 when the
 * answer came, has the number of the request and says ok. */
static int call(int fd, const char *fields, char *line, size_t room, parsec_qpu_json_t *reply)
{
    char request[256];
    parsec_qpu_json_t member;
    const int number = next_seq++;
    const int length = snprintf(request, sizeof(request), "{\"seq\":%d,%s}\n", number, fields);
    int64_t seq = -1;
    size_t len = 0;
    int ok = 0;

    if( (fd < 0) || (length >= (int)sizeof(request)) ) return -1;
    if( (ssize_t)length != send(fd, request, (size_t)length, MSG_NOSIGNAL) ) return -1;
    while( NULL == memchr(line, '\n', len) ) {
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        ssize_t got;
        if( len >= room - 1 ) return -1;
        if( poll(&pfd, 1, 20000) <= 0 ) return -1;
        if( (got = recv(fd, line + len, room - 1 - len, 0)) <= 0 ) return -1;
        len += (size_t)got;
    }
    if( (PARSEC_SUCCESS != parsec_qpu_json_parse(line, len, reply)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_member(reply, "seq", &member)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_int(&member, &seq)) || (seq != number) ||
        (PARSEC_SUCCESS != parsec_qpu_json_member(reply, "ok", &member)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_bool(&member, &ok)) || !ok ) return -1;
    return 0;
}

/* A new connection that says hello for a session. Returns the connection,
 * and whether the adapter says it had the session (-1: it did not say). */
static int hello(const char *session, int *resumed)
{
    char line[1024], fields[128];
    parsec_qpu_json_t reply, member;
    const int fd = qpu_adapter_dial(path);

    *resumed = -1;
    snprintf(fields, sizeof(fields), "\"op\":\"hello\",\"protocol\":%d,\"session\":\"%s\"", PARSEC_QPU_ADAPTER_PROTOCOL, session);
    if( 0 != call(fd, fields, line, sizeof(line), &reply) ) {
        if( fd >= 0 ) close(fd);
        return -1;
    }
    if( PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "resumed", &member) ) (void)parsec_qpu_json_bool(&member, resumed);
    return fd;
}

/* How many sessions the adapter has, asked on a connection of its own. The
 * adapter looks at what it may drop when a connection comes. */
static long sessions(void)
{
    char line[2048];
    parsec_qpu_json_t reply, member;
    const int fd = qpu_adapter_dial(path);
    int64_t value = -1;

    if( (0 == call(fd, "\"op\":\"stats\"", line, sizeof(line), &reply)) &&
        (PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "sessions", &member)) ) {
        (void)parsec_qpu_json_int(&member, &value);
    }
    if( fd >= 0 ) close(fd);
    return (long)value;
}

int main(int argc, char *argv[])
{
    char name[40], line[1024];
    parsec_qpu_json_t reply;
    double idle = -1.0;
    int64_t t_closed, give_up;
    int a, b, c, d, resumed, opt;
    long n;

    while( -1 != (opt = getopt(argc, argv, "a:i:")) ) {
        if( 'a' == opt ) path = optarg;
        else if( 'i' == opt ) idle = atof(optarg);
    }
    if( (NULL == path) || (idle <= 0.0) ) {
        fprintf(stderr, "usage: %s -a <socket of the adapter> -i <its --idle-drop, in seconds>\n", argv[0]);
        return 2;
    }
    snprintf(name, sizeof(name), "0x5e55%lx", (unsigned long)getpid());

    a = hello(name, &resumed);
    if( a < 0 ) {
        fprintf(stderr, "qpu_adapter_sessions: no answer to hello from the adapter at %s, nothing can be tested\n", path);
        return 1;
    }
    EXPECT(0 == resumed, "a session the adapter never saw: it says resumed %d, expected false", resumed);
    b = hello(name, &resumed);
    EXPECT((b >= 0) && (1 == resumed), "a session a connection uses: a second connection is told resumed %d, expected true", resumed);
    EXPECT(1 == (n = sessions()), "two connections of one session: the adapter counts %ld session(s)", n);

    /* One of the two closes, and the time passes. The other still uses it. */
    close(a);
    pause_for(idle + 0.5);
    EXPECT(1 == (n = sessions()), "a session one connection still uses, %g s after another one left it: the adapter has %ld session(s)",
           idle + 0.5, n);
    EXPECT(0 == call(b, "\"op\":\"wait\",\"timeout_ms\":0,\"max\":1", line, sizeof(line), &reply),
           "the connection that still uses the session got no answer: the session was taken from under it");

    /* Nobody uses it now. A client that comes back in time finds it. */
    close(b);
    c = hello(name, &resumed);
    EXPECT((c >= 0) && (1 == resumed), "a client that comes back at once is told resumed %d, expected true", resumed);
    if( c >= 0 ) close(c);
    t_closed = parsec_qpu_now_ns();

    /* And one that does not come back loses it, once the time is over. */
    give_up = t_closed + (int64_t)((idle + 20.0) * 1e9);
    while( (0 != (n = sessions())) && (parsec_qpu_now_ns() < give_up) ) pause_for(0.1);
    EXPECT(0 == n, "a session nobody uses is still there %g s after its last connection closed (%ld session(s))",
           1e-9 * (double)(parsec_qpu_now_ns() - t_closed), n);
    d = hello(name, &resumed);
    EXPECT((d >= 0) && (0 == resumed), "a session that was dropped: a client that asks for it is told resumed %d, expected false", resumed);
    EXPECT(0 == call(d, "\"op\":\"bye\"", line, sizeof(line), &reply), "no answer to bye");
    if( d >= 0 ) close(d);
    EXPECT(0 == (n = sessions()), "after bye the adapter still has %ld session(s)", n);

    if( nb_errors ) {
        fprintf(stderr, "qpu_adapter_sessions: %d error(s)\n", nb_errors);
        return 1;
    }
    printf("qpu_adapter_sessions: passed\n");
    return 0;
}
