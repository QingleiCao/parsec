/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * What an adapter keeps for a client whose connection is gone, asked by a
 * client that speaks the protocol itself: neither the adapter backend nor
 * the runtime is used. The adapter runs already
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
 * And one job, said in the protocol, for what the adapter backend never
 * asks an adapter that works: a submission whose tag the adapter has is
 * answered with the job it has, and no second one; the results of a job are
 * given once; a key that is not 64 bits as the protocol writes them is
 * refused; and a request that comes while a wait is held gets its answer
 * after the wait got its own.
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

/* One request and its answer, which stays in `line`. Returns 0 when an
 * answer came that has the number of the request; *ok is whether it says ok. */
static int ask(int fd, const char *fields, char *line, size_t room, parsec_qpu_json_t *reply, int *ok)
{
    char request[512];
    parsec_qpu_json_t member;
    const int number = next_seq++;
    const int length = snprintf(request, sizeof(request), "{\"seq\":%d,%s}\n", number, fields);
    int64_t seq = -1;
    size_t len = 0;

    *ok = 0;

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
        (PARSEC_SUCCESS != parsec_qpu_json_bool(&member, ok)) ) return -1;
    return 0;
}

/* The next line a connection sends, within five seconds, parsed: its number, and whether it says ok. 0 when one came. */
static int next_answer(int fd, char *line, size_t room, int64_t *seq, int *ok)
{
    parsec_qpu_json_t reply, member;
    size_t len = 0;

    *seq = -1; *ok = 0;
    for(;;) {
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        if( (len >= room - 1) || (poll(&pfd, 1, 5000) <= 0) || (1 != recv(fd, line + len, 1, 0)) ) return -1;
        if( '\n' == line[len++] ) break;
    }
    if( (PARSEC_SUCCESS != parsec_qpu_json_parse(line, len, &reply)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_member(&reply, "seq", &member)) || (PARSEC_SUCCESS != parsec_qpu_json_int(&member, seq)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_member(&reply, "ok", &member)) || (PARSEC_SUCCESS != parsec_qpu_json_bool(&member, ok)) ) return -1;
    return 0;
}

/* The same, for a request that has to be taken: 0 when the answer says ok. */
static int call(int fd, const char *fields, char *line, size_t room, parsec_qpu_json_t *reply)
{
    int ok = 0;
    return ((0 == ask(fd, fields, line, room, reply, &ok)) && ok) ? 0 : -1;
}

/* The job an answer names, as the adapter wrote it; an empty string when it names none. */
static void job_of(const parsec_qpu_json_t *reply, char name[40])
{
    parsec_qpu_json_t member;
    name[0] = '\0';
    if( (PARSEC_SUCCESS != parsec_qpu_json_member(reply, "job", &member)) ||
        (parsec_qpu_json_string(&member, name, 40) <= 0) ) name[0] = '\0';
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

    /* ---- A job, said in the protocol. One qubit, flipped and measured, as
     *      PaRSEC writes it: ten shots, and every one reads 1. ---- */
    {
        static const char *prepare = "\"op\":\"prepare\",\"circuit\":\"qpu_adapter_sessions\",\"qasm\":\"OPENQASM 3.0;\\n"
                                     "include \\\"stdgates.inc\\\";\\nqubit[1] q;\\nbit[1] c;\\nx q[0];\\nc[0] = measure q[0];\\n\"";
        static const char *submit = "\"op\":\"submit\",\"tag\":\"0x7a61\",\"entries\":[{\"circuit\":\"qpu_adapter_sessions\","
                                    "\"params\":[],\"shots\":10,\"key\":\"0x1\",\"attempt\":1}]";
        char big[4096], first[40], again[40], told[40], fields[128], bits[40];
        parsec_qpu_json_t member, element, entry, pair;
        const int e = hello(name, &resumed);
        int64_t count = -1;
        int ok = 0, ended = 0, others = 0;

        EXPECT(e >= 0, "no answer to hello for the session of the job");
        EXPECT(0 == call(e, prepare, big, sizeof(big), &reply), "a circuit as PaRSEC writes one was not taken");
        EXPECT(0 == call(e, submit, big, sizeof(big), &reply), "a job of that circuit was not taken");
        job_of(&reply, first);
        EXPECT('\0' != first[0], "the answer to a submission names no job");
        /* The same submission again, as a client does whose answer was lost. */
        EXPECT(0 == call(e, submit, big, sizeof(big), &reply), "a submission was not taken when it was asked again");
        job_of(&reply, again);
        EXPECT(0 == strcmp(first, again), "a submission asked again got job %s, and the first time %s", again, first);
        /* One job ends, and no other. */
        give_up = parsec_qpu_now_ns() + (int64_t)60e9;
        while( (0 == ended) && (parsec_qpu_now_ns() < give_up) &&
               (0 == call(e, "\"op\":\"wait\",\"timeout_ms\":2000,\"max\":8", big, sizeof(big), &reply)) &&
               (PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "jobs", &member)) ) {
            element.at = NULL;
            while( 1 == parsec_qpu_json_next(&member, &element) ) {
                parsec_qpu_json_t state;
                job_of(&element, told);
                if( (0 == strcmp(told, first)) && (PARSEC_SUCCESS == parsec_qpu_json_member(&element, "state", &state)) &&
                    parsec_qpu_json_is(&state, "done") ) ended++;
                else others++;
            }
        }
        EXPECT((1 == ended) && (0 == others), "one job was submitted, twice: %d ended well under its name, and %d other(s) were told", ended, others);
        /* Its results, once. */
        snprintf(fields, sizeof(fields), "\"op\":\"results\",\"job\":\"%s\"", first);
        bits[0] = '\0';
        if( (0 == call(e, fields, big, sizeof(big), &reply)) && (PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "entries", &member)) ) {
            element.at = NULL;
            if( (1 == parsec_qpu_json_next(&member, &element)) && (PARSEC_SUCCESS == parsec_qpu_json_member(&element, "counts", &entry)) ) {
                pair.at = NULL;
                if( 1 == parsec_qpu_json_next(&entry, &pair) ) {
                    parsec_qpu_json_t item;
                    item.at = NULL;
                    if( 1 == parsec_qpu_json_next(&pair, &item) ) (void)parsec_qpu_json_string(&item, bits, sizeof(bits));
                    if( 1 == parsec_qpu_json_next(&pair, &item) ) (void)parsec_qpu_json_int(&item, &count);
                }
            }
        }
        EXPECT((0 == strcmp(bits, "0x1")) && (10 == count), "ten shots of a qubit that was flipped: outcome '%s', %lld time(s)", bits, (long long)count);
        EXPECT((0 == ask(e, fields, big, sizeof(big), &reply, &ok)) && !ok, "the results of a job were given a second time");
        /* A key is "0x" and up to sixteen hexadecimal digits: a number written another way is no key. */
        EXPECT((0 == ask(e, "\"op\":\"submit\",\"tag\":\"0x7a62\",\"entries\":[{\"circuit\":\"qpu_adapter_sessions\","
                            "\"params\":[],\"shots\":10,\"key\":\"123\",\"attempt\":1}]", big, sizeof(big), &reply, &ok)) && !ok,
               "a job whose key is written without 0x was taken");
        EXPECT((0 == ask(e, "\"op\":\"submit\",\"tag\":\"0x7a63\",\"entries\":[{\"circuit\":\"qpu_adapter_sessions\","
                            "\"params\":[],\"shots\":10,\"key\":\"0x10000000000000000\",\"attempt\":1}]", big, sizeof(big), &reply, &ok)) && !ok,
               "a job whose key has more than 64 bits was taken");
        /* A wait that would be held for twenty seconds, and a request behind it:
         * both are answered at once, the wait first and well, the other refused. */
        {
            const int held = next_seq++, behind = next_seq++;
            char two[160];
            int64_t seq_a = -1, seq_b = -1;
            int ok_a = 0, ok_b = 0;
            const int length = snprintf(two, sizeof(two), "{\"seq\":%d,\"op\":\"wait\",\"timeout_ms\":20000,\"max\":1}\n{\"seq\":%d,\"op\":\"stats\"}\n",
                                        held, behind);
            EXPECT((ssize_t)length == send(e, two, (size_t)length, MSG_NOSIGNAL), "two requests could not be sent");
            EXPECT((0 == next_answer(e, big, sizeof(big), &seq_a, &ok_a)) && (held == seq_a) && ok_a,
                   "a wait with a request behind it: the first answer is to request %lld and says ok %d; expected the wait's, %d, and ok",
                   (long long)seq_a, ok_a, held);
            EXPECT((0 == next_answer(e, big, sizeof(big), &seq_b, &ok_b)) && (behind == seq_b) && !ok_b,
                   "the request behind a wait: the second answer is to request %lld and says ok %d; expected %d, refused",
                   (long long)seq_b, ok_b, behind);
        }
        EXPECT(0 == call(e, "\"op\":\"bye\"", big, sizeof(big), &reply), "no answer to bye after the job");
        if( e >= 0 ) close(e);
    }

    if( nb_errors ) {
        fprintf(stderr, "qpu_adapter_sessions: %d error(s)\n", nb_errors);
        return 1;
    }
    printf("qpu_adapter_sessions: passed\n");
    return 0;
}
