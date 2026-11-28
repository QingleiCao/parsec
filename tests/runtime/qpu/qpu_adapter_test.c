/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * The adapter backend through the backend interface, without the runtime:
 * this program plays the part of the progress thread, and the adapter is
 * qpu_adapter_loopback, which it starts and stops for every case.
 *
 *   - A job of one circuit and of several: counts equal to the sampler's for
 *     the same seed and key, also when the buffer is too small for them, and
 *     for parameters that need every digit a double has.
 *   - A circuit is given to the adapter once.
 *   - wait() and wakeup(): a time-out, a wake-up that came first, a wake-up
 *     from another thread while the adapter holds the request.
 *   - cancel() and abandon().
 *   - An adapter that takes a job and whose answer is lost, in four ways
 *     (the fourth: an answer under the number of another request): the job
 *     is found again by its tag, and runs once.
 *   - A job that ended before its identifier was known is not reported
 *     before it is.
 *   - An adapter that is busy, that refuses a job, whose job fails or is
 *     cancelled, whose result is not one, that forgot the circuits.
 *   - An adapter that was started again: its jobs are reported as failed.
 *     One that then says it drives a quantum device is not used again. The
 *     new adapter uses the names of the old one's jobs for its own: the
 *     jobs stay apart, no result is taken from the wrong one, and the new
 *     one's job is not dropped for the old one's.
 *   - An adapter that stops answering for too long and then goes on: its
 *     job is given up, and it is told so when it is back. While it takes
 *     no connection the client does not hang.
 *   - An adapter that is gone for good: its jobs fail after the time a QPU
 *     may be unavailable, and closing does not hang.
 *   - No adapter at all.
 *   - A member of a message read into the variable that holds the message.
 *   - Nothing of the environment crosses the socket: the adapter is told a
 *     text that this process has in its environment under the names an SDK
 *     reads its credentials from, and fails the case if it ever sees it.
 *
 * In every case the adapter has to end without having seen its client break
 * the protocol.
 *
 * Options:
 *   -L PATH   the qpu_adapter_loopback program (required)
 *   -d DIR    where the sockets go (default: the current directory)
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"
#include "qpu_adapter_spawn.h"

#include <poll.h>
#include <pthread.h>
#include <signal.h>
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

#define MS      1000000LL
#define SEED    1
#define SECRET  "s3cr3t-credential-that-stays-here"

static const parsec_qpu_backend_ops_t *ops = &parsec_qpu_backend_adapter_ops;
static const char *loopback = NULL, *dir = ".";
static char  path[200];
static pid_t server = -1;
static int   nb_servers = 0;

/* ---- The adapter of a case ---- */

static void server_start(const char *faults, long duration_us, const char *flag)
{
    char duration[32];
    char *argv[16];
    int n = 0;

    snprintf(path, sizeof(path), "%s/qat%d_%d.sock", dir, (int)(getpid() % 100000), nb_servers++);
    snprintf(duration, sizeof(duration), "%ld", duration_us);
    argv[n++] = (char*)loopback;
    argv[n++] = (char*)"-s"; argv[n++] = path;
    argv[n++] = (char*)"-E"; argv[n++] = (char*)SECRET;
    argv[n++] = (char*)"-d"; argv[n++] = duration;
    static char flags[64];
    if( NULL != faults ) { argv[n++] = (char*)"-F"; argv[n++] = (char*)faults; }
    if( NULL != flag ) {
        /* One option, or several separated by blanks. */
        char *save = NULL;
        snprintf(flags, sizeof(flags), "%s", flag);
        for( char *token = strtok_r(flags, " ", &save); (NULL != token) && (n < 15); token = strtok_r(NULL, " ", &save) ) argv[n++] = token;
    }
    argv[n] = NULL;
    server = qpu_adapter_spawn(path, argv, 30.0);
    if( server < 0 ) {
        fprintf(stderr, "qpu_adapter_test: no adapter, nothing can be tested\n");
        exit(1);
    }
    parsec_qpu_params.adapter_address = path;
}

static void server_stop(const char *what)
{
    const int rc = qpu_adapter_stop(server, 10.0);
    EXPECT(0 == rc, "%s: the adapter ended with %d: it saw its client break the protocol, or it crashed", what, rc);
    server = -1;
    unlink(path);
}

/* A counter of the adapter, asked as a client that is not the backend. */
static long server_stat(const char *key)
{
    static const char request[] = "{\"op\":\"stats\",\"seq\":1}\n";
    parsec_qpu_json_t reply, member;
    char line[2048];
    size_t len = 0;
    int64_t value = -1;
    const int fd = qpu_adapter_dial(path);

    if( fd < 0 ) return -1;
    if( (ssize_t)(sizeof(request) - 1) == send(fd, request, sizeof(request) - 1, MSG_NOSIGNAL) ) {
        while( len < sizeof(line) - 1 ) {
            struct pollfd pfd = { .fd = fd, .events = POLLIN };
            ssize_t n;
            if( poll(&pfd, 1, 5000) <= 0 ) break;
            if( (n = recv(fd, line + len, sizeof(line) - 1 - len, 0)) <= 0 ) break;
            len += (size_t)n;
            if( NULL != memchr(line, '\n', len) ) break;
        }
        if( (len > 0) && (PARSEC_SUCCESS == parsec_qpu_json_parse(line, len, &reply)) &&
            (PARSEC_SUCCESS == parsec_qpu_json_member(&reply, key, &member)) ) {
            (void)parsec_qpu_json_int(&member, &value);
        }
    }
    close(fd);
    return (long)value;
}

#define EXPECT_STAT(KEY, VALUE, WHAT)                                                           \
    do {                                                                                        \
        const long _v = server_stat(KEY);                                                       \
        EXPECT((VALUE) == _v, "%s: the adapter counts %ld %s, expected %ld", (WHAT), _v, (KEY), (long)(VALUE)); \
    } while(0)

/* ---- Circuits ---- */

#define ROOM(NB_OPS)  (sizeof(parsec_qpu_circuit_t) + (size_t)(NB_OPS) * PARSEC_QPU_OP_SIZE)

/* A circuit of n qubits that uses every operation and two parameters;
 * `variant` makes different circuits of the same shape. */
static parsec_qpu_circuit_t *make_circuit(uint32_t n, int variant)
{
    const size_t room = ROOM(8 * n + 16);
    parsec_qpu_circuit_t *c = (parsec_qpu_circuit_t*)calloc(1, room);

    parsec_qpu_circuit_init(c, n, 2);
    for( uint32_t q = 0; q < n; q++ ) parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_H, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.0);
    for( uint32_t q = 0; q + 1 < n; q++ ) {
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RXX + (int)((q + (uint32_t)variant) % 3), (int)q, (int)q + 1, 0, -0.5 - 0.25 * variant - 0.1 * q);
    }
    for( uint32_t q = 0; q < n; q++ ) {
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RX + (int)(q % 3), (int)q, 0, 1, 2.0);
        parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RZ, (int)q, 0, PARSEC_QPU_NO_PARAM, 0.3 * (q + 1) + variant);
    }
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_CX, 0, (int)n - 1, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_CZ, (int)n - 1, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_X + (variant % 3), 0, 0, PARSEC_QPU_NO_PARAM, 0.0);
    parsec_qpu_circuit_add(c, room, PARSEC_QPU_OP_RY, (int)n - 1, 0, 1, -0.7);
    parsec_qpu_circuit_measure_all(c, room);
    if( PARSEC_SUCCESS != parsec_qpu_circuit_check(c, room, NULL) ) {
        fprintf(stderr, "qpu_adapter_test: the test's own circuit is not well-formed\n");
        exit(1);
    }
    return c;
}

static parsec_qpu_result_t *result_new(uint32_t max_entries)
{
    return (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(max_entries));
}

static parsec_qpu_request_t request_of(const parsec_qpu_circuit_t *circuit, const double *params, uint32_t shots,
                                       uint32_t max_entries, uint64_t key)
{
    parsec_qpu_request_t r;
    memset(&r, 0, sizeof(r));
    r.circuit = circuit; r.params = params; r.shots = shots; r.max_entries = max_entries; r.key = key; r.attempt = 1;
    return r;
}

/* What the backend gave is what the sampler gives for the same evaluation. */
static void expect_sampler(const parsec_qpu_result_t *result, const parsec_qpu_request_t *r, const char *what)
{
    parsec_qpu_result_t *expected = result_new(r->max_entries);
    int rc = parsec_qpu_sampler_sample(r->circuit, r->params, r->shots, SEED, r->key, expected, r->max_entries);

    EXPECT(PARSEC_SUCCESS == rc, "%s: the sampler failed", what);
    EXPECT(PARSEC_QPU_STATUS_OK == result->status, "%s: status %d", what, (int)result->status);
    EXPECT(result->nb_qubits == r->circuit->nb_qubits, "%s: a result on %d qubits", what, (int)result->nb_qubits);
    EXPECT(PARSEC_QPU_RESULT_MOCK & result->flags, "%s: a result that is not marked as simulated", what);
    EXPECT((PARSEC_QPU_RESULT_TRUNCATED & result->flags) == (PARSEC_QPU_RESULT_TRUNCATED & expected->flags),
           "%s: cut or not, differently from the sampler", what);
    EXPECT((result->nb_entries == expected->nb_entries) && (result->shots_returned == expected->shots_returned),
           "%s: %u entries for %u shots, the sampler gives %u for %u", what, result->nb_entries, result->shots_returned,
           expected->nb_entries, expected->shots_returned);
    if( result->nb_entries == expected->nb_entries ) {
        EXPECT(0 == memcmp(result + 1, expected + 1, (size_t)expected->nb_entries * PARSEC_QPU_ENTRY_SIZE),
               "%s: the counts are not those of the sampler", what);
    }
    EXPECT(result->t_start <= result->t_end, "%s: the job ended before it started", what);
    free(expected);
}

/* The backend does not connect again right after it lost its adapter: give it the time. */
static void nap(void)
{
    const struct timespec ts = { .tv_sec = 0, .tv_nsec = 80 * MS };
    nanosleep(&ts, NULL);
}

/* Wait for the end of one job, for at most `ms`. */
static int wait_event(parsec_qpu_backend_t *backend, int64_t ms, parsec_qpu_event_t *event)
{
    const int64_t give_up = parsec_qpu_now_ns() + ms * MS;
    for(;;) {
        const int64_t now = parsec_qpu_now_ns();
        int n;
        if( now >= give_up ) return 0;
        if( 0 != (n = ops->wait(backend, give_up - now, event, 1)) ) return n;
    }
}

static parsec_qpu_backend_t *backend_open(const char *what)
{
    parsec_qpu_backend_t *backend = NULL;
    const int rc = ops->open(&backend, 0);
    EXPECT((PARSEC_SUCCESS == rc) && (NULL != backend), "%s: the backend did not open (%d)", what, rc);
    if( NULL == backend ) {
        fprintf(stderr, "qpu_adapter_test: %s: nothing can be tested\n", what);
        exit(1);
    }
    return backend;
}

/* Submit, wait, take the result and compare it. */
static void run_one(parsec_qpu_backend_t *backend, const parsec_qpu_request_t *r, uint64_t tag, const char *what)
{
    parsec_qpu_result_t *result = result_new(r->max_entries);
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;

    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, 1, tag, &id), "%s: submit", what);
    EXPECT(1 == wait_event(backend, 20000, &event), "%s: the job did not end", what);
    EXPECT((event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "%s: the end of another job, or not a good end (%d)", what, event.state);
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &result, 1), "%s: results", what);
    expect_sampler(result, r, what);
    free(result);
}

/* ---- The cases ---- */

static void test_jobs(void)
{
    static const double params[2] = { 0.37, -1.2 }, other[2] = { 1.5, 0.25 };
    parsec_qpu_circuit_t *c3 = make_circuit(3, 0), *c5 = make_circuit(5, 1), *c8 = make_circuit(8, 2);
    parsec_qpu_backend_t *backend;
    parsec_qpu_adapter_stats_t stats;
    parsec_qpu_caps_t caps;
    parsec_qpu_request_t r, batch[4];
    parsec_qpu_result_t *results[4];
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;

    server_start(NULL, 2000, NULL);
    backend = backend_open("jobs");
    EXPECT(0 == ops->physical, "the adapter backend says, before it knows its adapter, that it drives a quantum device");
    EXPECT(PARSEC_SUCCESS == ops->capabilities(backend, &caps), "capabilities");
    EXPECT((24 == caps.nb_qubits) && (64 == caps.max_circuits_per_job) && (0 == caps.is_physical) && (0 == caps.max_shots),
           "what the adapter said it is did not arrive: %u qubits, %u circuits per job", caps.nb_qubits, caps.max_circuits_per_job);

    r = request_of(c3, params, 1000, 1000, 0x1111);
    run_one(backend, &r, 1, "one circuit");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, 1, results, 1), "the results of a job were given twice");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->cancel(backend, 987654), "a job that never was was cancelled");

    /* The same circuit with other values, and with a key above 2^63: the circuit is not given again. */
    r = request_of(c3, other, 300, 300, 0xfedcba9876543210ULL);
    run_one(backend, &r, 2, "the same circuit again");
    EXPECT_STAT("prepares", 1, "a circuit used twice");
    /* A buffer too small for the outcomes: cut as the sampler cuts. */
    r = request_of(c8, params, 4000, 7, 0x2222);
    run_one(backend, &r, 3, "a result that does not fit");
    r = request_of(c5, params, 1, 1, 0x3333);
    run_one(backend, &r, 4, "one shot");

    /* Several circuits in one job: a result for each, as when it runs alone. */
    batch[0] = request_of(c3, params, 500, 500, 0x10);
    batch[1] = request_of(c5, other,  500, 500, 0x11);
    batch[2] = request_of(c3, other,  500, 3,   0x12);
    batch[3] = request_of(c8, params, 500, 500, 0x13);
    for( int i = 0; i < 4; i++ ) results[i] = result_new(batch[i].max_entries);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, batch, 4, 5, &id), "submit of four circuits");
    EXPECT(1 == wait_event(backend, 20000, &event), "the job of four circuits did not end");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, results, 4), "results of four circuits");
    for( int i = 0; i < 4; i++ ) {
        expect_sampler(results[i], &batch[i], "one of four circuits");
        free(results[i]);
    }
    EXPECT_STAT("prepares", 3, "three circuits in all");
    EXPECT_STAT("jobs", 5, "five jobs");
    EXPECT_STAT("live_jobs", 0, "every result was taken");

    /* A job asked for with the wrong number of circuits is forgotten all the same. */
    r = request_of(c3, params, 10, 10, 0x4444);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 6, &id), "submit");
    EXPECT(1 == wait_event(backend, 20000, &event), "the job did not end");
    results[0] = result_new(10); results[1] = result_new(10);
    EXPECT(PARSEC_ERR_BAD_PARAM == ops->results(backend, id, results, 2), "results for two circuits of a job of one");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, results, 1), "the job was not forgotten");
    free(results[0]); free(results[1]);
    EXPECT_STAT("live_jobs", 0, "a job taken for another");

    /* Parameters arrive with every digit they have. Each angle here is a
     * parameter times a large number: a parameter cut to a few digits is
     * another angle altogether, and other counts. */
    {
        static const double fine[2] = { 0.123456789012345678, -0.906093942819681683 };
        const size_t room = ROOM(8);
        parsec_qpu_circuit_t *amplified = (parsec_qpu_circuit_t*)calloc(1, room);

        parsec_qpu_circuit_init(amplified, 2, 2);
        parsec_qpu_circuit_add(amplified, room, PARSEC_QPU_OP_H,  0, 0, PARSEC_QPU_NO_PARAM, 0.0);
        parsec_qpu_circuit_add(amplified, room, PARSEC_QPU_OP_RZ, 0, 0, 0, 1e9);
        parsec_qpu_circuit_add(amplified, room, PARSEC_QPU_OP_H,  0, 0, PARSEC_QPU_NO_PARAM, 0.0);
        parsec_qpu_circuit_add(amplified, room, PARSEC_QPU_OP_H,  1, 0, PARSEC_QPU_NO_PARAM, 0.0);
        parsec_qpu_circuit_add(amplified, room, PARSEC_QPU_OP_RZ, 1, 0, 1, -3e8);
        parsec_qpu_circuit_add(amplified, room, PARSEC_QPU_OP_H,  1, 0, PARSEC_QPU_NO_PARAM, 0.0);
        parsec_qpu_circuit_measure_all(amplified, room);
        r = request_of(amplified, fine, 2000, 2000, 0x7777);
        run_one(backend, &r, 9, "parameters with all their digits");
        free(amplified);
    }

    /* A parameter that is not a number is not something a message can hold. */
    {
        const double bad[2] = { 0.5, 0.0 / 0.0 };
        r = request_of(c3, bad, 10, 10, 0x5555);
        EXPECT(PARSEC_ERR_NOT_SUPPORTED == ops->submit(backend, &r, 1, 7, &id), "a job with a parameter that is not a number was taken");
    }
    /* A circuit wider than the adapter takes is refused for good. */
    {
        parsec_qpu_circuit_t *wide = make_circuit(25, 0);
        r = request_of(wide, params, 10, 10, 0x6666);
        EXPECT(PARSEC_ERR_NOT_SUPPORTED == ops->submit(backend, &r, 1, 8, &id), "a circuit wider than the adapter takes was taken");
        free(wide);
    }

    EXPECT(PARSEC_SUCCESS == parsec_qpu_adapter_stats_of(0, &stats), "no counters for the backend that is open");
    EXPECT((1 == stats.connects) && (0 == stats.unanswered) && (4 == stats.prepares) && (7 == stats.submits) &&
           (0 == stats.lost_jobs) && (9 == stats.results) && (0 == stats.bad_results) && (0 == stats.unsettled),
           "the backend counts %llu connections, %llu unanswered, %llu circuits, %llu jobs, %llu results, %llu bad, %llu unsettled",
           (unsigned long long)stats.connects, (unsigned long long)stats.unanswered, (unsigned long long)stats.prepares,
           (unsigned long long)stats.submits, (unsigned long long)stats.results, (unsigned long long)stats.bad_results,
           (unsigned long long)stats.unsettled);
    EXPECT(PARSEC_ERR_NOT_FOUND == parsec_qpu_adapter_stats_of(3, &stats), "counters for a QPU that has no adapter");

    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    EXPECT_STAT("byes", 1, "after close");
    EXPECT_STAT("sessions", 0, "after close");
    server_stop("jobs");
    free(c3); free(c5); free(c8);
}

static void *wake_later(void *arg)
{
    const struct timespec ts = { .tv_sec = 0, .tv_nsec = 300 * MS };
    nanosleep(&ts, NULL);
    ops->wakeup((parsec_qpu_backend_t*)arg);
    return NULL;
}

static void test_wait(void)
{
    parsec_qpu_backend_t *backend;
    parsec_qpu_event_t event;
    pthread_t thread;
    int64_t t0, t1;

    server_start(NULL, 2000, NULL);
    backend = backend_open("wait");

    t0 = parsec_qpu_now_ns();
    EXPECT(0 == ops->wait(backend, 200 * MS, &event, 1), "an event without a job");
    t1 = parsec_qpu_now_ns();
    EXPECT(t1 - t0 >= 199 * MS, "wait(200 ms) returned after %lld ms", (long long)((t1 - t0) / MS));

    /* A wake-up that came first is not lost. */
    ops->wakeup(backend);
    ops->wakeup(backend);
    t0 = parsec_qpu_now_ns();
    EXPECT(0 == ops->wait(backend, 20000 * MS, &event, 1), "an event without a job");
    t1 = parsec_qpu_now_ns();
    EXPECT(t1 - t0 < 10000 * MS, "a wake-up before wait() did not make it return: %lld ms", (long long)((t1 - t0) / MS));
    /* ... and is used up by it. */
    t0 = parsec_qpu_now_ns();
    EXPECT(0 == ops->wait(backend, 200 * MS, &event, 1), "an event without a job");
    EXPECT(parsec_qpu_now_ns() - t0 >= 199 * MS, "one wake-up made two calls return early");

    /* From another thread, while the adapter holds the request. */
    pthread_create(&thread, NULL, wake_later, backend);
    t0 = parsec_qpu_now_ns();
    EXPECT(0 == ops->wait(backend, 30000 * MS, &event, 1), "an event without a job");
    t1 = parsec_qpu_now_ns();
    pthread_join(thread, NULL);
    EXPECT((t1 - t0 >= 250 * MS) && (t1 - t0 < 15000 * MS), "a wake-up after 300 ms made wait() return after %lld ms",
           (long long)((t1 - t0) / MS));
    EXPECT_STAT("wakes", 1, "one wake-up reached the adapter");
    /* No limit: it returns with the event. */
    {
        static const double params[2] = { 0.1, 0.2 };
        parsec_qpu_circuit_t *c = make_circuit(3, 0);
        parsec_qpu_request_t r = request_of(c, params, 20, 20, 0x77);
        parsec_qpu_job_id_t id = 0;
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &id), "submit");
        EXPECT(1 == ops->wait(backend, -1, &event, 1), "a wait without limit did not return the end of the job");
        EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel of a job that ended");
        free(c);
    }
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("wait");
}

static void test_cancel(void)
{
    static const double params[2] = { 0.1, 0.2 };
    parsec_qpu_circuit_t *c = make_circuit(4, 0);
    parsec_qpu_request_t r = request_of(c, params, 100, 100, 0x99);
    parsec_qpu_backend_t *backend;
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;

    server_start(NULL, 400000, NULL);               /* jobs of 0.4 s */
    backend = backend_open("cancel");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &id), "submit");
    EXPECT_STAT("live_jobs", 1, "a job at the adapter");
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel of a job that runs");
    EXPECT_STAT("live_jobs", 0, "the adapter forgot the job that was cancelled");
    EXPECT(0 == wait_event(backend, 700, &event), "the end of a job that was cancelled was reported");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->cancel(backend, id), "a job was cancelled twice");

    /* Giving up on a submission the adapter never had changes nothing there. */
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 4242), "abandon");
    EXPECT_STAT("abandons", 1, "the adapter heard of the tag that was given up");
    EXPECT_STAT("abandon_hits", 0, "there was no job under that tag");
    run_one(backend, &r, 2, "after a cancellation");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("cancel");
    free(c);
}

/* The adapter takes the job and its answer never arrives, in one of four
 * ways: asking again with the same tag finds the job, which ran once. */
static void test_lost_answer(const char *fault, int timeout_ms)
{
    static const double params[2] = { 0.3, 0.4 };
    parsec_qpu_circuit_t *c = make_circuit(4, 1);
    parsec_qpu_request_t r = request_of(c, params, 200, 200, 0xabc);
    parsec_qpu_result_t *result = result_new(200);
    parsec_qpu_backend_t *backend;
    parsec_qpu_adapter_stats_t stats;
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;
    char faults[64];
    int rc;

    snprintf(faults, sizeof(faults), "%s:1", fault);
    server_start(faults, 2000, NULL);
    parsec_qpu_params.adapter_timeout_ms = timeout_ms;
    backend = backend_open(fault);
    rc = ops->submit(backend, &r, 1, 77, &id);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "%s: a submission whose answer was lost returned %d, not \"ask again\"", fault, rc);
    EXPECT_STAT("jobs", 1, fault);
    /* Asked again at once, the backend has not connected again yet: still "ask again". */
    rc = ops->submit(backend, &r, 1, 77, &id);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "%s: asking again at once returned %d", fault, rc);
    nap();
    rc = ops->submit(backend, &r, 1, 77, &id);
    EXPECT(PARSEC_SUCCESS == rc, "%s: asking again returned %d", fault, rc);
    EXPECT_STAT("jobs", 1, fault);
    EXPECT_STAT("tag_hits", 1, fault);
    EXPECT(1 == wait_event(backend, 20000, &event), "%s: the job did not end", fault);
    EXPECT((event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "%s: not the good end of the job", fault);
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &result, 1), "%s: results", fault);
    expect_sampler(result, &r, fault);
    parsec_qpu_adapter_stats_of(0, &stats);
    EXPECT((2 == stats.connects) && (1 == stats.unanswered), "%s: %llu connections and %llu requests without an answer, expected 2 and 1",
           fault, (unsigned long long)stats.connects, (unsigned long long)stats.unanswered);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    parsec_qpu_params.adapter_timeout_ms = 60000;
    server_stop(fault);
    free(result);
    free(c);
}

/* The answer is lost, the job ends, and the caller waits before it asks
 * again: the end of a job whose identifier it does not have is not reported. */
static void test_early_end(void)
{
    static const double params[2] = { 0.3, 0.4 };
    parsec_qpu_circuit_t *c = make_circuit(4, 1);
    parsec_qpu_request_t r = request_of(c, params, 200, 200, 0xabd);
    parsec_qpu_result_t *result = result_new(200);
    parsec_qpu_backend_t *backend;
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;

    server_start("drop_submit:1", 0, NULL);
    backend = backend_open("early end");
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 78, &id), "the answer was not lost");
    EXPECT(0 == wait_event(backend, 300, &event), "the end of a job was reported before submit() returned its identifier");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 78, &id), "asking again");
    EXPECT(1 == wait_event(backend, 20000, &event), "the end that was heard early is not reported once the job is known");
    EXPECT((event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "not the good end of the job");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &result, 1), "results");
    expect_sampler(result, &r, "a job that ended early");
    EXPECT_STAT("jobs", 1, "early end");

    /* And one that the caller gives up on instead: the adapter drops it. */
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("early end");
    server_start("silent:1", 2000, NULL);
    parsec_qpu_params.adapter_timeout_ms = 300;
    backend = backend_open("abandon");
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 79, &id), "the answer was not lost");
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 79), "abandon");
    (void)wait_event(backend, 200, &event);         /* the backend connects again and says so */
    EXPECT_STAT("abandon_hits", 1, "the job that was given up was dropped by the adapter");
    EXPECT_STAT("live_jobs", 0, "abandon");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    parsec_qpu_params.adapter_timeout_ms = 60000;
    server_stop("abandon");
    free(result);
    free(c);
}

static void test_refusals(void)
{
    static const double params[2] = { 0.3, 0.4 };
    parsec_qpu_circuit_t *c = make_circuit(4, 2);
    parsec_qpu_request_t r = request_of(c, params, 200, 200, 0xdef);
    parsec_qpu_result_t *result = result_new(200);
    parsec_qpu_backend_t *backend;
    parsec_qpu_adapter_stats_t stats;
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;
    int rc;

    /* Busy, then a job refused for good, then one whose answer is lost before it is taken. */
    server_start("busy:1,reject:3,drop_before:4", 2000, NULL);
    backend = backend_open("refusals");
    rc = ops->submit(backend, &r, 1, 1, &id);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "a busy adapter: %d, not \"ask again\"", rc);
    run_one(backend, &r, 1, "after busy");
    rc = ops->submit(backend, &r, 1, 2, &id);
    EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_OUT_OF_RESOURCE != rc), "a job the adapter refuses for good: %d", rc);
    rc = ops->submit(backend, &r, 1, 3, &id);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "a connection closed before the job was taken: %d", rc);
    nap();
    run_one(backend, &r, 3, "after a closed connection");
    EXPECT_STAT("jobs", 2, "two jobs were taken");
    EXPECT_STAT("tag_hits", 0, "no job was found by its tag");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("refusals");

    /* Jobs that end badly, and a result that is not one. */
    server_start("fail:1,cancelled:2,badcounts:3", 2000, NULL);
    backend = backend_open("bad ends");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &id), "submit");
    EXPECT(1 == wait_event(backend, 20000, &event), "the job that fails did not end");
    EXPECT((event.id == id) && (PARSEC_QPU_JOB_FAILED == event.state), "a failed job was reported as %d", event.state);
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, &result, 1), "a failed job has results");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 2, &id), "submit");
    EXPECT(1 == wait_event(backend, 20000, &event), "the job that is cancelled did not end");
    EXPECT((event.id == id) && (PARSEC_QPU_JOB_CANCELLED == event.state), "a cancelled job was reported as %d", event.state);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 3, &id), "submit");
    EXPECT(1 == wait_event(backend, 20000, &event), "the job did not end");
    EXPECT(PARSEC_QPU_JOB_DONE == event.state, "the job with a bad result did not end well");
    rc = ops->results(backend, id, &result, 1);
    EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_NOT_FOUND != rc), "counts that miss a shot were taken for a result (%d)", rc);
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, &result, 1), "the job with a bad result was not forgotten");
    (void)wait_event(backend, 50, &event);
    EXPECT_STAT("live_jobs", 0, "the adapter was told to forget the jobs that ended badly");
    parsec_qpu_adapter_stats_of(0, &stats);
    EXPECT((1 == stats.bad_results) && (0 == stats.unsettled), "%llu bad result(s), %llu unsettled",
           (unsigned long long)stats.bad_results, (unsigned long long)stats.unsettled);
    run_one(backend, &r, 4, "after bad ends");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("bad ends");

    /* An adapter that forgot the circuits is given them again. */
    server_start("nocircuit:2", 2000, NULL);
    backend = backend_open("forgotten circuits");
    run_one(backend, &r, 1, "before the circuits are forgotten");
    run_one(backend, &r, 2, "after the circuits were forgotten");
    EXPECT_STAT("prepares", 2, "the circuit was given again");
    EXPECT_STAT("jobs", 2, "forgotten circuits");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("forgotten circuits");
    free(result);
    free(c);
}

/* An adapter that was started again does not have the jobs of the first. */
static void test_restart(int turn)
{
    static const double params[2] = { 0.3, 0.4 };
    const char *what = turn ? "an adapter that turns physical" : "an adapter started again";
    parsec_qpu_circuit_t *c = make_circuit(4, 0);
    parsec_qpu_request_t r = request_of(c, params, 200, 200, 0x123);
    parsec_qpu_backend_t *backend;
    parsec_qpu_adapter_stats_t stats;
    parsec_qpu_job_id_t first = 0, id = 0;
    parsec_qpu_event_t event;
    int rc;

    server_start(turn ? "turn:2" : "forget:2", 300000, NULL);
    backend = backend_open(what);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &first), "%s: submit", what);
    r.key = 0x124;
    rc = ops->submit(backend, &r, 1, 2, &id);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "%s: the submission during which it went away returned %d", what, rc);
    EXPECT(1 == wait_event(backend, 20000, &event), "%s: the job the first adapter had was not reported", what);
    EXPECT((event.id == first) && (PARSEC_QPU_JOB_FAILED == event.state), "%s: the job the first adapter had was reported as %d", what, event.state);
    parsec_qpu_adapter_stats_of(0, &stats);
    EXPECT(1 == stats.lost_jobs, "%s: %llu job(s) counted as lost, expected 1", what, (unsigned long long)stats.lost_jobs);
    EXPECT(0 == wait_event(backend, 500, &event), "%s: the lost job was reported twice, or ran after all", what);
    if( !turn ) {
        run_one(backend, &r, 2, what);
        EXPECT_STAT("prepares", 2, "the new adapter was given the circuit");
        /* ... before the job, and not after the job was refused for it. */
        EXPECT_STAT("submits", 3, "the new adapter was given the job once");
        EXPECT_STAT("hellos", 2, what);
        EXPECT_STAT("resumes", 0, what);
    } else {
        /* The guard looked at the first one. This one is not used. */
        rc = ops->submit(backend, &r, 1, 2, &id);
        EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "%s: a job was given to it (%d)", what, rc);
        (void)wait_event(backend, 1500, &event);
        rc = ops->submit(backend, &r, 1, 2, &id);
        EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "%s: a job was given to it later (%d)", what, rc);
        EXPECT_STAT("submits", 2, "no submission reached the adapter that says it drives a device");
        EXPECT_STAT("hellos", 2, "the backend did not go back to it");
    }
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop(what);
    free(c);
}

/* The adapter is started again, and the caller submits before it waits: the
 * new adapter numbers its jobs from the first again, so its first job has
 * the name the old adapter gave the job that is lost. The two are two jobs. */
static void test_restart_same_names(void)
{
    static const double params[2] = { 0.3, 0.4 };
    parsec_qpu_circuit_t *c = make_circuit(4, 0);
    parsec_qpu_request_t ra = request_of(c, params, 200, 200, 0x501), rb = request_of(c, params, 200, 200, 0x502);
    parsec_qpu_result_t *result = result_new(200);
    parsec_qpu_backend_t *backend;
    parsec_qpu_job_id_t a = 0, b = 0;
    parsec_qpu_event_t event[4];
    int seen_a = 0, seen_b = 0, rc;

    /* 1. The lost job is still in flight. */
    server_start("forget:2", 300000, NULL);
    backend = backend_open("the same names");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &ra, 1, 1, &a), "submit");
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &rb, 1, 2, &b), "the adapter did not go away");
    nap();
    rc = ops->submit(backend, &rb, 1, 2, &b);       /* connects again, to an adapter that knows nothing */
    EXPECT(PARSEC_SUCCESS == rc, "a submission to the adapter that was started again returned %d", rc);
    EXPECT(a != b, "the job of the new adapter has the name of the job that was lost");
    for( int64_t give_up = parsec_qpu_now_ns() + 20000 * MS; (!seen_a || !seen_b) && (parsec_qpu_now_ns() < give_up); ) {
        const int n = ops->wait(backend, 200 * MS, event, 4);
        for( int i = 0; i < n; i++ ) {
            if( event[i].id == a ) { seen_a++; EXPECT(PARSEC_QPU_JOB_FAILED == event[i].state, "the lost job was reported as %d", event[i].state); }
            else if( event[i].id == b ) { seen_b++; EXPECT(PARSEC_QPU_JOB_DONE == event[i].state, "the new job was reported as %d", event[i].state); }
            else EXPECT(0, "the end of a job nobody knows");
        }
    }
    EXPECT((1 == seen_a) && (1 == seen_b), "the lost job was reported %d time(s) and the new one %d, expected once each", seen_a, seen_b);
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, a, &result, 1), "a lost job has results");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, b, &result, 1), "results of the new job");
    expect_sampler(result, &rb, "the job of the adapter that was started again");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("the same names");

    /* 2. The lost job had ended, and its result was not taken yet. The new
     * adapter has another job under its name: that one's result is not it. */
    server_start("forget:2", 2000, NULL);
    backend = backend_open("a result that was not taken");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &ra, 1, 1, &a), "submit");
    EXPECT(1 == wait_event(backend, 20000, event), "the job did not end");
    EXPECT((event[0].id == a) && (PARSEC_QPU_JOB_DONE == event[0].state), "not the good end of the job");
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &rb, 1, 2, &b), "the adapter did not go away");
    nap();
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &rb, 1, 2, &b), "a submission to the adapter that was started again");
    EXPECT(1 == wait_event(backend, 20000, event), "the new job did not end");
    EXPECT((event[0].id == b) && (PARSEC_QPU_JOB_DONE == event[0].state), "not the good end of the new job");
    rc = ops->results(backend, a, &result, 1);
    EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_NOT_FOUND != rc), "the result of a job whose adapter is gone was taken from another job (%d)", rc);
    EXPECT(PARSEC_SUCCESS == ops->results(backend, b, &result, 1), "results of the new job");
    expect_sampler(result, &rb, "the new job, after the result of the old one was asked for");
    EXPECT_STAT("results", 1, "the new adapter gave one result");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("a result that was not taken");

    /* 3. The same, and the caller takes the lost job for a job of two
     * circuits. A job taken for another is dropped at its adapter; this
     * one's adapter is gone, and the job that has its name at the new one
     * is another job, which stays. */
    server_start("forget:2", 2000, NULL);
    backend = backend_open("a lost job taken for another");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &ra, 1, 1, &a), "submit");
    EXPECT(1 == wait_event(backend, 20000, event), "the job did not end");
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &rb, 1, 2, &b), "the adapter did not go away");
    nap();
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &rb, 1, 2, &b), "a submission to the adapter that was started again");
    EXPECT(1 == wait_event(backend, 20000, event), "the new job did not end");
    {
        parsec_qpu_result_t *two[2] = { result_new(200), result_new(200) };
        rc = ops->results(backend, a, two, 2);
        EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_NOT_FOUND != rc), "a lost job, taken for a job of two circuits, gave %d", rc);
        free(two[0]); free(two[1]);
    }
    (void)wait_event(backend, 50, event);           /* what the backend has to tell the adapter, it has told by now */
    EXPECT_STAT("cancels", 0, "the new adapter was told to drop a job, and the only one it has is not the lost one");
    EXPECT_STAT("live_jobs", 1, "the job of the new adapter");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, b, &result, 1), "results of the new job");
    expect_sampler(result, &rb, "the new job, after the lost one was taken for another");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("a lost job taken for another");
    free(result);
    free(c);
}

/* An adapter that stops answering for longer than a QPU may be unavailable,
 * and then goes on: it is the same process, and it still has the job that
 * was given up on. It has to be told to drop it. And while it is stopped,
 * with no room left for a connection to wait in, the client must not hang. */
static void test_comes_back(void)
{
    static const double params[2] = { 0.3, 0.4 };
    parsec_qpu_circuit_t *c = make_circuit(4, 0);
    parsec_qpu_request_t r = request_of(c, params, 200, 200, 0x601);
    parsec_qpu_backend_t *backend;
    parsec_qpu_adapter_stats_t stats;
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;
    int64_t t0, longest = 0;
    int failed = 0, later = 0;

    server_start(NULL, 3000000, "-b 1");            /* jobs of 3 s, and one connection may wait */
    parsec_qpu_params.adapter_timeout_ms     = 60;
    parsec_qpu_params.unavailable_timeout_ms = 400;
    backend = backend_open("comes back");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &id), "submit");
    kill(server, SIGSTOP);
    /* About 2 s of it: more connections are tried than its queue holds. */
    for( int i = 0; i < 20; i++ ) {
        int n;
        t0 = parsec_qpu_now_ns();
        n = ops->wait(backend, 100 * MS, &event, 1);
        if( parsec_qpu_now_ns() - t0 > longest ) longest = parsec_qpu_now_ns() - t0;
        if( 1 == n ) {
            failed++;
            EXPECT((event.id == id) && (PARSEC_QPU_JOB_FAILED == event.state), "the job of an adapter that stopped was reported as %d", event.state);
        }
    }
    EXPECT(longest < 1500 * MS, "a wait of 100 ms took %lld ms while the adapter took no connection", (long long)(longest / MS));
    EXPECT(1 == failed, "the job of an adapter that stopped answering was reported %d time(s), expected once", failed);
    parsec_qpu_adapter_stats_of(0, &stats);
    EXPECT(stats.connects >= 3, "only %llu connection(s) were made while the adapter was stopped: its queue was never full",
           (unsigned long long)stats.connects);
    kill(server, SIGCONT);
    /* It answers again, and hears that the job is given up. */
    for( int i = 0; i < 30; i++ ) later += ops->wait(backend, 100 * MS, &event, 1);
    EXPECT(0 == later, "the job that was given up was reported again once its adapter was back");
    EXPECT_STAT("live_jobs", 0, "the adapter that came back was told to drop the job that was given up on");
    parsec_qpu_adapter_stats_of(0, &stats);
    EXPECT(0 == stats.unsettled, "%llu cancellation(s) the adapter has not heard, and it is back", (unsigned long long)stats.unsettled);
    r.key = 0x602;
    parsec_qpu_params.adapter_timeout_ms = 60000;
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    server_stop("comes back");
    parsec_qpu_params.unavailable_timeout_ms = 60000;
    free(c);
}

static void test_gone(void)
{
    static const double params[2] = { 0.3, 0.4 };
    parsec_qpu_circuit_t *c = make_circuit(4, 0);
    parsec_qpu_request_t r = request_of(c, params, 200, 200, 0x321);
    parsec_qpu_backend_t *backend = NULL;
    parsec_qpu_job_id_t id = 0;
    parsec_qpu_event_t event;
    int64_t t0, t1;
    int rc;

    /* No adapter at all: the backend does not open. */
    parsec_qpu_params.adapter_address = (char*)"qat_nobody_listens_here.sock";
    rc = ops->open(&backend, 0);
    EXPECT((PARSEC_SUCCESS != rc) && (NULL == backend), "a backend opened without an adapter");
    parsec_qpu_params.adapter_address = (char*)"";
    EXPECT(PARSEC_SUCCESS != ops->open(&backend, 0), "a backend opened without the address of an adapter");
    {
        char longer[300];
        memset(longer, 'a', sizeof(longer) - 1);
        longer[sizeof(longer) - 1] = '\0';
        parsec_qpu_params.adapter_address = longer;
        EXPECT(PARSEC_SUCCESS != ops->open(&backend, 0), "a backend opened with an address no socket can have");
    }

    /* An adapter that goes away for good while it has a job. */
    server_start(NULL, 200000, NULL);
    parsec_qpu_params.unavailable_timeout_ms = 400;
    backend = backend_open("gone");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &id), "submit");
    server_stop("gone");
    t0 = parsec_qpu_now_ns();
    EXPECT(1 == wait_event(backend, 20000, &event), "the job of an adapter that is gone was never reported");
    t1 = parsec_qpu_now_ns();
    EXPECT((event.id == id) && (PARSEC_QPU_JOB_FAILED == event.state), "the job of an adapter that is gone was reported as %d", event.state);
    EXPECT(t1 - t0 >= 390 * MS, "the job was given up after %lld ms, before the time a QPU may be unavailable", (long long)((t1 - t0) / MS));
    rc = ops->submit(backend, &r, 1, 2, &id);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == rc, "a submission to nobody returned %d", rc);
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 2), "abandon while the adapter is gone");
    t0 = parsec_qpu_now_ns();
    EXPECT(0 == ops->wait(backend, 100 * MS, &event, 1), "an event from nobody");
    EXPECT(parsec_qpu_now_ns() - t0 >= 99 * MS, "waiting for an adapter that is gone does not sleep");
    t0 = parsec_qpu_now_ns();
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close without an adapter");
    EXPECT(parsec_qpu_now_ns() - t0 < 5000 * MS, "closing without an adapter took %lld ms", (long long)((parsec_qpu_now_ns() - t0) / MS));
    parsec_qpu_params.unavailable_timeout_ms = 60000;
    free(c);
}

/* The answers are read with one variable: a member takes the place of the
 * object it was found in. */
static void test_json(void)
{
    static const char text[] = "{\"x\":1,\"a\":{\"b\":{\"c\":7},\"y\":2}}";
    parsec_qpu_json_t value;
    int64_t number = 0;

    EXPECT(PARSEC_SUCCESS == parsec_qpu_json_parse(text, sizeof(text) - 1, &value), "a message was not read");
    EXPECT((PARSEC_SUCCESS == parsec_qpu_json_member(&value, "a", &value)) &&
           (PARSEC_SUCCESS == parsec_qpu_json_member(&value, "b", &value)) &&
           (PARSEC_SUCCESS == parsec_qpu_json_member(&value, "c", &value)) &&
           (PARSEC_SUCCESS == parsec_qpu_json_int(&value, &number)) && (7 == number),
           "a member read into the variable that holds its object is not found");
}

/* An adapter that says it drives a quantum device opens, and says so: the
 * guard is the service's, which closes it. Here: it is not hidden. */
static void test_physical(void)
{
    parsec_qpu_backend_t *backend;
    parsec_qpu_caps_t caps;

    server_start(NULL, 2000, "-P");
    backend = backend_open("physical");
    EXPECT(PARSEC_SUCCESS == ops->capabilities(backend, &caps), "capabilities");
    EXPECT(1 == caps.is_physical, "an adapter that says it drives a quantum device is not described as one");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    EXPECT_STAT("submits", 0, "physical");
    server_stop("physical");
}

int main(int argc, char *argv[])
{
    int opt;

    while( -1 != (opt = getopt(argc, argv, "L:d:")) ) {
        if( 'L' == opt ) loopback = optarg;
        else if( 'd' == opt ) dir = optarg;
    }
    if( NULL == loopback ) {
        fprintf(stderr, "usage: %s -L <qpu_adapter_loopback> [-d <directory for the sockets>]\n", argv[0]);
        return 2;
    }
    /* What an SDK would read its credentials from: none of it may leave this process. */
    setenv("QISKIT_IBM_TOKEN", SECRET, 1);
    setenv("QISKIT_IBM_INSTANCE", SECRET, 1);
    setenv("IBM_CLOUD_API_KEY", SECRET, 1);
    setenv("IONQ_API_KEY", SECRET, 1);
    setenv("PARSEC_QPU_ADAPTER_TOKEN", SECRET, 1);

    test_json();
    test_jobs();
    test_wait();
    test_cancel();
    test_lost_answer("drop_submit", 60000);
    test_lost_answer("garbage", 60000);
    test_lost_answer("silent", 300);
    test_lost_answer("stale", 60000);
    test_early_end();
    test_refusals();
    test_restart(0);
    test_restart(1);
    test_restart_same_names();
    test_comes_back();
    test_gone();
    test_physical();

    if( nb_errors ) {
        fprintf(stderr, "qpu_adapter_test: %d error(s)\n", nb_errors);
        return 1;
    }
    printf("qpu_adapter_test: passed\n");
    return 0;
}
