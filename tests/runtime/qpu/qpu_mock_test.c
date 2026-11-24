/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * The mock backend through the backend interface, without the runtime: this
 * program plays the part of the progress thread.
 *
 *   - Delay distributions.
 *   - A job: its time, its event, its result against the sampler.
 *   - Slots and the queue in front of them; the queue delay.
 *   - wait() and wakeup(): time-outs, a wake-up that comes first, a wake-up
 *     from another thread.
 *   - Results that do not depend on the order of submission.
 *   - Every fault the mock can inject, as the interface says it looks.
 *   - What the mock watches its caller for: a tag used twice, two jobs of
 *     one evaluation at once, inputs that change under a job.
 *   - cancel() in every state of a job, a job being sampled included, and
 *     abandon() of a submission whose reply was lost.
 *   - Jobs of several circuits: one result per circuit, the same as when
 *     the circuit runs alone; a time that is a part per job plus a part per
 *     circuit; one fate for the whole job.
 *
 * Times are checked from below (nothing happens early) and against the
 * mock's own timestamps, which are exact unless the mock says a job ended
 * late (its sampling outlasted its modelled time); never against an upper
 * bound that a loaded machine could miss, except by seconds.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

#define MS 1000000LL
#define SEED 11

static const parsec_qpu_backend_ops_t *ops = &parsec_qpu_backend_mock_ops;

/* ------------------------------------------------------------------------
 * Inputs
 * --------------------------------------------------------------------- */

typedef struct {
    parsec_qpu_circuit_t *circuit;
    double                params[2];
    parsec_qpu_result_t  *result;
    uint32_t              max_entries;
} job_data_t;

/* A circuit on n qubits whose outcome probabilities depend on v. */
static job_data_t data_new(int n, int v, int depth)
{
    const size_t room = sizeof(parsec_qpu_circuit_t) + (size_t)(n + depth * 3 * n + n) * PARSEC_QPU_OP_SIZE;
    job_data_t d;

    d.circuit = (parsec_qpu_circuit_t*)calloc(1, room);
    parsec_qpu_circuit_init(d.circuit, (uint32_t)n, 2);
    for( int q = 0; q < n; q++ ) (void)parsec_qpu_circuit_add(d.circuit, room, PARSEC_QPU_OP_H, q, 0, PARSEC_QPU_NO_PARAM, 0);
    for( int l = 0; l < depth; l++ ) {
        for( int q = 0; q + 1 < n; q++ ) (void)parsec_qpu_circuit_add(d.circuit, room, PARSEC_QPU_OP_RZZ, q, q + 1, 0, 0.5 + 0.1 * q);
        for( int q = 0; q < n; q++ ) (void)parsec_qpu_circuit_add(d.circuit, room, PARSEC_QPU_OP_RX, q, 0, 1, 1.0 + 0.2 * l);
        for( int q = 0; q < n; q++ ) (void)parsec_qpu_circuit_add(d.circuit, room, PARSEC_QPU_OP_RY, q, 0, PARSEC_QPU_NO_PARAM, 0.05 * (v % 13));
    }
    (void)parsec_qpu_circuit_measure_all(d.circuit, room);
    d.params[0]   = 0.3 + 0.07 * v;
    d.params[1]   = 0.9 - 0.03 * v;
    d.max_entries = 64;
    d.result      = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(d.max_entries));
    return d;
}

static void data_free(job_data_t *d)
{
    free(d->circuit);
    free(d->result);
}

static parsec_qpu_request_t request_of(const job_data_t *d, uint32_t shots, uint64_t key, uint32_t attempt)
{
    parsec_qpu_request_t r;
    memset(&r, 0, sizeof(r));
    r.circuit     = d->circuit;
    r.params      = d->params;
    r.shots       = shots;
    r.max_entries = d->max_entries;
    r.key         = key;
    r.attempt     = attempt;
    return r;
}

static int same_as_sampler(const job_data_t *d, uint32_t shots, uint64_t key)
{
    parsec_qpu_result_t *expected = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(d->max_entries));
    int same;

    (void)parsec_qpu_sampler_sample(d->circuit, d->params, shots, SEED, key, expected, d->max_entries);
    same = (expected->nb_entries == d->result->nb_entries) &&
           (expected->shots_returned == d->result->shots_returned) &&
           (0 == memcmp(expected + 1, d->result + 1, (size_t)expected->nb_entries * PARSEC_QPU_ENTRY_SIZE));
    free(expected);
    return same;
}

/* ------------------------------------------------------------------------
 * The mock, opened with given parameters
 * --------------------------------------------------------------------- */

static parsec_qpu_backend_t *mock_open(int slots, const char *delay, const char *queue_delay, const char *faults)
{
    parsec_qpu_backend_t *backend = NULL;
    int rc;

    parsec_qpu_params.mock_seed        = SEED;
    parsec_qpu_params.mock_slots       = slots;
    parsec_qpu_params.mock_delay       = (char*)delay;
    parsec_qpu_params.mock_queue_delay = (char*)queue_delay;
    parsec_qpu_params.mock_faults      = (char*)faults;
    parsec_qpu_params.mock_threads     = 2;
    parsec_qpu_params.mock_max_qubits  = 20;
    rc = ops->open(&backend, 0);
    if( (PARSEC_SUCCESS != rc) || (NULL == backend) ) {
        fprintf(stderr, "the mock did not open (%d)\n", rc);
        exit(2);
    }
    return backend;
}

/* Wait for one event, for at most `limit_ms`. Returns 1 with the event, or 0. */
static int next_event(parsec_qpu_backend_t *backend, int limit_ms, parsec_qpu_event_t *event)
{
    const int64_t give_up = parsec_qpu_now_ns() + (int64_t)limit_ms * MS;
    for(;;) {
        const int64_t now = parsec_qpu_now_ns();
        int n;
        if( now >= give_up ) return 0;
        n = ops->wait(backend, give_up - now, event, 1);
        if( n < 0 ) { EXPECT(0, "wait() returned %d", n); return 0; }
        if( n > 0 ) return 1;
    }
}

/* ------------------------------------------------------------------------
 * Delays
 * --------------------------------------------------------------------- */

static int cmp_i64(const void *a, const void *b)
{
    const int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return (x > y) - (x < y);
}

static void test_delays(void)
{
    enum { N = 100000 };
    static int64_t draws[N];
    parsec_qpu_delay_t delay;
    double mean, mean_log, var_log;
    char path[128];
    FILE *file;

    EXPECT(PARSEC_SUCCESS == parsec_qpu_delay_parse(&delay, "const:1500"), "const");
    EXPECT(1500000 == parsec_qpu_delay_draw(&delay, 1) && 1500000 == parsec_qpu_delay_draw(&delay, ~0ULL), "const draw");
    parsec_qpu_delay_fini(&delay);

    EXPECT(PARSEC_SUCCESS == parsec_qpu_delay_parse(&delay, "uniform:100:300"), "uniform");
    mean = 0;
    for( int i = 0; i < N; i++ ) {
        draws[i] = parsec_qpu_delay_draw(&delay, parsec_qpu_mix64((uint64_t)i));
        EXPECT((draws[i] >= 100000) && (draws[i] <= 300000), "uniform draw %lld out of bounds", (long long)draws[i]);
        mean += (double)draws[i] / N;
    }
    EXPECT(fabs(mean - 200000.0) < 0.01 * 200000.0, "uniform mean is %.0f ns", mean);
    EXPECT(parsec_qpu_delay_draw(&delay, 12345) == parsec_qpu_delay_draw(&delay, 12345), "the same bits gave two delays");
    parsec_qpu_delay_fini(&delay);

    EXPECT(PARSEC_SUCCESS == parsec_qpu_delay_parse(&delay, "exp:250"), "exp");
    mean = 0;
    for( int i = 0; i < N; i++ ) {
        draws[i] = parsec_qpu_delay_draw(&delay, parsec_qpu_mix64((uint64_t)i));
        mean += (double)draws[i] / N;
    }
    qsort(draws, N, sizeof(int64_t), cmp_i64);
    EXPECT(fabs(mean - 250000.0) < 0.02 * 250000.0, "exponential mean is %.0f ns", mean);
    EXPECT(fabs((double)draws[N / 2] - 250000.0 * log(2.0)) < 0.03 * 250000.0, "exponential median is %lld ns", (long long)draws[N / 2]);
    parsec_qpu_delay_fini(&delay);

    EXPECT(PARSEC_SUCCESS == parsec_qpu_delay_parse(&delay, "lognormal:400:0.5"), "lognormal");
    mean_log = var_log = 0;
    for( int i = 0; i < N; i++ ) {
        draws[i] = parsec_qpu_delay_draw(&delay, parsec_qpu_mix64((uint64_t)i));
        mean_log += log((double)draws[i]) / N;
    }
    for( int i = 0; i < N; i++ ) var_log += pow(log((double)draws[i]) - mean_log, 2) / N;
    qsort(draws, N, sizeof(int64_t), cmp_i64);
    EXPECT(fabs((double)draws[N / 2] - 400000.0) < 0.03 * 400000.0, "lognormal median is %lld ns", (long long)draws[N / 2]);
    EXPECT(fabs(sqrt(var_log) - 0.5) < 0.02, "lognormal sigma is %.4f", sqrt(var_log));
    parsec_qpu_delay_fini(&delay);

    /* Replay: every draw is one of the recorded delays, and each is drawn. */
    snprintf(path, sizeof(path), "qpu_mock_test_replay.%d.txt", (int)getpid());
    file = fopen(path, "w");
    if( NULL != file ) {
        char spec[160];
        int seen[3] = { 0, 0, 0 };
        fprintf(file, "# recorded latencies, microseconds\n120\n\n  4500.5\n70\n");
        fclose(file);
        snprintf(spec, sizeof(spec), "replay:%s", path);
        EXPECT(PARSEC_SUCCESS == parsec_qpu_delay_parse(&delay, spec), "replay of a file that is there");
        EXPECT(3 == delay.nb_samples, "%u delays read, expected 3", delay.nb_samples);
        for( int i = 0; (i < 1000) && (3 == delay.nb_samples); i++ ) {
            const int64_t d = parsec_qpu_delay_draw(&delay, parsec_qpu_mix64((uint64_t)i));
            if( 120000 == d ) seen[0]++; else if( 4500500 == d ) seen[1]++; else if( 70000 == d ) seen[2]++;
            else EXPECT(0, "replay drew %lld ns, which is not in the file", (long long)d);
        }
        EXPECT(seen[0] && seen[1] && seen[2], "replay never drew one of the recorded delays");
        parsec_qpu_delay_fini(&delay);
        /* A file with something that is not a delay in it. */
        file = fopen(path, "w");
        fprintf(file, "120\nsoon\n");
        fclose(file);
        EXPECT(PARSEC_SUCCESS != parsec_qpu_delay_parse(&delay, spec), "replay of a file with a word in it");
        unlink(path);
        EXPECT(PARSEC_SUCCESS != parsec_qpu_delay_parse(&delay, spec), "replay of a file that is not there");
    } else {
        EXPECT(0, "could not write %s", path);
    }

    {
        static const char *bad[] = { "", "const:", "const:-1", "const:12abc", "uniform:5:1", "uniform:5",
                                     "exp:-3", "lognormal:100", "lognormal:100:-1", "gaussian:1:2", "replay:" };
        for( size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++ ) {
            EXPECT(PARSEC_SUCCESS != parsec_qpu_delay_parse(&delay, bad[i]), "'%s' was taken for a delay", bad[i]);
        }
        EXPECT(PARSEC_SUCCESS != parsec_qpu_delay_parse(&delay, NULL), "no specification was taken for a delay");
    }
}

/* ------------------------------------------------------------------------
 * One job, slots, queue delay
 * --------------------------------------------------------------------- */

static void test_one_job(void)
{
    parsec_qpu_backend_t *backend = mock_open(2, "const:20000", "const:0", "none");
    job_data_t d = data_new(4, 1, 1);
    parsec_qpu_request_t r = request_of(&d, 500, 42, 1);
    parsec_qpu_caps_t caps;
    parsec_qpu_mock_stats_t stats;
    parsec_qpu_event_t event;
    parsec_qpu_job_id_t id = 0;
    int64_t t0;

    EXPECT(0 == ops->physical, "the mock says it can drive a quantum device");
    memset(&caps, 0xff, sizeof(caps));
    EXPECT(PARSEC_SUCCESS == ops->capabilities(backend, &caps), "capabilities");
    EXPECT(0 == caps.is_physical, "the mock says it is a quantum device");
    EXPECT(20 == caps.nb_qubits, "the mock takes %u qubits, 20 were asked for", caps.nb_qubits);

    t0 = parsec_qpu_now_ns();
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1001, &id), "submit");
    EXPECT(0 != id, "no job identifier");
    EXPECT(next_event(backend, 10000, &event), "no event for the job");
    EXPECT(parsec_qpu_now_ns() - t0 >= 20 * MS, "the job of 20 ms ended after %.3f ms", (parsec_qpu_now_ns() - t0) * 1e-6);
    EXPECT((event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "the event is for job %llu in state %d",
           (unsigned long long)event.id, event.state);

    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &d.result, 1), "results");
    EXPECT(PARSEC_QPU_STATUS_OK == d.result->status, "status %u", (unsigned)d.result->status);
    EXPECT(PARSEC_QPU_RESULT_MOCK == d.result->flags, "flags are %x: every result of the mock is flagged as such, and only that here",
           (unsigned)d.result->flags);
    EXPECT(4 == d.result->nb_qubits, "%u qubits", (unsigned)d.result->nb_qubits);
    EXPECT(500 == d.result->shots_returned, "%u shots", d.result->shots_returned);
    parsec_qpu_mock_stats(&stats);
    EXPECT((d.result->t_end - d.result->t_start == 20 * MS) ||
           ((stats.late > 0) && (d.result->t_end - d.result->t_start > 20 * MS)),
           "the job ran for %lld ns, not for its 20 ms", (long long)(d.result->t_end - d.result->t_start));
    EXPECT(d.result->t_start >= t0, "the job started before it was submitted");
    EXPECT(same_as_sampler(&d, 500, 42), "the counts are not the sampler's for this seed and key");

    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, &d.result, 1), "the results of a job were given twice");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->cancel(backend, id), "a job that is gone was cancelled");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->cancel(backend, 987654), "a job that never was was cancelled");
    EXPECT(0 == next_event(backend, 20, &event), "an event with no job");

    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.submit_calls) && (1 == stats.submits) && (1 == stats.results) && (1 == stats.computed),
           "counters after one job: %llu calls, %llu jobs, %llu results, %llu sampled",
           (unsigned long long)stats.submit_calls, (unsigned long long)stats.submits,
           (unsigned long long)stats.results, (unsigned long long)stats.computed);
    EXPECT(stats.min_submit_gap < 0, "a gap between submissions after one job");

    /* A result buffer with too little room: flagged, and still a mock result. */
    d.max_entries = 2;
    r = request_of(&d, 500, 43, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1002, &id), "submit");
    EXPECT(next_event(backend, 10000, &event), "no event");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &d.result, 1), "results");
    EXPECT((PARSEC_QPU_RESULT_MOCK | PARSEC_QPU_RESULT_TRUNCATED) == d.result->flags, "flags are %x", (unsigned)d.result->flags);
    EXPECT((2 == d.result->nb_entries) && (d.result->shots_returned < 500), "%u entries, %u shots in room for 2",
           d.result->nb_entries, d.result->shots_returned);

    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    data_free(&d);
}

static void test_slots(void)
{
    enum { NJ = 6 };
    parsec_qpu_backend_t *backend = mock_open(2, "const:30000", "const:0", "none");
    job_data_t d[NJ];
    parsec_qpu_job_id_t ids[NJ];
    parsec_qpu_mock_stats_t stats;
    parsec_qpu_event_t event;
    int64_t t0 = parsec_qpu_now_ns(), starts[NJ], ends[NJ];
    int done = 0;

    for( int i = 0; i < NJ; i++ ) {
        parsec_qpu_request_t r;
        d[i] = data_new(3, i, 1);
        r = request_of(&d[i], 100, 100 + (uint64_t)i, 1);
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 2000 + (uint64_t)i, &ids[i]), "submit %d", i);
    }
    while( (done < NJ) && next_event(backend, 20000, &event) ) {
        int which = -1;
        for( int i = 0; i < NJ; i++ ) if( ids[i] == event.id ) which = i;
        EXPECT(which >= 0, "an event for an unknown job");
        if( which < 0 ) break;
        EXPECT(PARSEC_SUCCESS == ops->results(backend, event.id, &d[which].result, 1), "results %d", which);
        starts[which] = d[which].result->t_start;
        ends[which]   = d[which].result->t_end;
        EXPECT(same_as_sampler(&d[which], 100, 100 + (uint64_t)which), "job %d: not the sampler's counts", which);
        done++;
    }
    EXPECT(NJ == done, "%d of %d jobs ended", done, NJ);
    /* Three rounds of two jobs of 30 ms. */
    EXPECT(parsec_qpu_now_ns() - t0 >= 90 * MS, "6 jobs of 30 ms on 2 slots took %.3f ms", (parsec_qpu_now_ns() - t0) * 1e-6);
    parsec_qpu_mock_stats(&stats);
    if( NJ == done ) {
        for( int i = 0; i < NJ; i++ ) {
            int together = 0, chained = (i < 2);
            EXPECT((ends[i] - starts[i] == 30 * MS) || ((stats.late > 0) && (ends[i] - starts[i] > 30 * MS)),
                   "job %d ran for %lld ns", i, (long long)(ends[i] - starts[i]));
            for( int j = 0; j < NJ; j++ ) {
                /* The jobs running at the instant job i starts, itself included. */
                if( (starts[j] <= starts[i]) && (starts[i] < ends[j]) ) together++;
                /* A job that queued starts at the very instant an earlier one ended. */
                if( (j < i) && (ends[j] == starts[i]) ) chained = 1;
            }
            EXPECT(together <= 2, "%d jobs were running when job %d started, on 2 slots", together, i);
            EXPECT(chained, "job %d did not start when a slot was freed", i);
        }
    }
    EXPECT(2 == stats.max_running, "at most %u jobs ran at once on 2 slots", stats.max_running);
    EXPECT(NJ == stats.max_held, "the mock held at most %u jobs, %d were given at once", stats.max_held, NJ);
    EXPECT(stats.min_submit_gap >= 0, "no gap between submissions was measured");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    for( int i = 0; i < NJ; i++ ) data_free(&d[i]);

    /* A queue delay comes before the job runs, inside its slot. */
    {
        job_data_t q = data_new(3, 0, 1);
        parsec_qpu_request_t r = request_of(&q, 100, 5, 1);
        parsec_qpu_job_id_t id;

        backend = mock_open(1, "const:5000", "const:15000", "none");
        t0 = parsec_qpu_now_ns();
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &id), "submit");
        EXPECT(next_event(backend, 10000, &event), "no event");
        EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &q.result, 1), "results");
        EXPECT(q.result->t_start - t0 >= 15 * MS, "the job started %.3f ms after its submission, before its queue delay of 15 ms",
               (q.result->t_start - t0) * 1e-6);
        parsec_qpu_mock_stats(&stats);
        EXPECT((q.result->t_end - q.result->t_start == 5 * MS) || ((stats.late > 0) && (q.result->t_end - q.result->t_start > 5 * MS)),
               "the job ran for %lld ns", (long long)(q.result->t_end - q.result->t_start));
        EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
        data_free(&q);
    }
}

/* ------------------------------------------------------------------------
 * wait() and wakeup()
 * --------------------------------------------------------------------- */

static void *waker_main(void *arg)
{
    struct timespec ts = { 0, 30 * MS };
    nanosleep(&ts, NULL);
    (void)ops->wakeup((parsec_qpu_backend_t*)arg);
    return NULL;
}

static void test_wait_wakeup(void)
{
    parsec_qpu_backend_t *backend = mock_open(1, "const:1000", "const:0", "none");
    parsec_qpu_event_t event;
    pthread_t waker;
    int64_t t0;
    int n;

    t0 = parsec_qpu_now_ns();
    n = ops->wait(backend, 20 * MS, &event, 1);
    EXPECT(0 == n, "wait() with nothing to wait for returned %d", n);
    EXPECT(parsec_qpu_now_ns() - t0 >= 20 * MS, "wait() for 20 ms returned after %.3f ms", (parsec_qpu_now_ns() - t0) * 1e-6);

    /* A wake-up that comes when nobody waits makes the next wait return at once... */
    (void)ops->wakeup(backend);
    (void)ops->wakeup(backend);
    t0 = parsec_qpu_now_ns();
    n = ops->wait(backend, 5000 * MS, &event, 1);
    EXPECT(0 == n, "wait() after a wake-up returned %d", n);
    EXPECT(parsec_qpu_now_ns() - t0 < 2500 * MS, "wait() after a wake-up slept for %.3f ms", (parsec_qpu_now_ns() - t0) * 1e-6);
    /* ... and only that one. */
    t0 = parsec_qpu_now_ns();
    n = ops->wait(backend, 20 * MS, &event, 1);
    EXPECT((0 == n) && (parsec_qpu_now_ns() - t0 >= 20 * MS), "a wake-up was used twice: wait() returned %d after %.3f ms",
           n, (parsec_qpu_now_ns() - t0) * 1e-6);

    /* A wake-up from another thread ends a wait with no time limit worth the name. */
    pthread_create(&waker, NULL, waker_main, backend);
    t0 = parsec_qpu_now_ns();
    n = ops->wait(backend, 30000 * MS, &event, 1);
    EXPECT(0 == n, "wait() woken by another thread returned %d", n);
    EXPECT(parsec_qpu_now_ns() - t0 < 15000 * MS, "wait() was not woken: it slept %.3f ms", (parsec_qpu_now_ns() - t0) * 1e-6);
    pthread_join(waker, NULL);

    /* A negative time-out is no time-out. */
    pthread_create(&waker, NULL, waker_main, backend);
    n = ops->wait(backend, -1, &event, 1);
    EXPECT(0 == n, "wait() with no time-out, woken by another thread, returned %d", n);
    pthread_join(waker, NULL);

    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
}

/* ------------------------------------------------------------------------
 * Results do not depend on the order of submission
 * --------------------------------------------------------------------- */

static void test_order_independence(void)
{
    enum { NJ = 8 };
    parsec_qpu_result_t *first[NJ];

    for( int pass = 0; pass < 2; pass++ ) {
        parsec_qpu_backend_t *backend = mock_open(3, "uniform:0:4000", "const:0", "none");
        job_data_t d[NJ];
        parsec_qpu_job_id_t ids[NJ];
        parsec_qpu_event_t event;
        int done = 0;

        for( int n = 0; n < NJ; n++ ) {
            const int i = (0 == pass) ? n : NJ - 1 - n;
            parsec_qpu_request_t r;
            d[i] = data_new(5, i, 2);
            r = request_of(&d[i], 2000, 700 + (uint64_t)i, 1);
            /* Other tags the second time: the answer must not depend on them. */
            EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, (uint64_t)(pass * 1000 + i + 1), &ids[i]), "submit");
        }
        while( (done < NJ) && next_event(backend, 20000, &event) ) {
            for( int i = 0; i < NJ; i++ ) {
                if( ids[i] != event.id ) continue;
                EXPECT(PARSEC_SUCCESS == ops->results(backend, event.id, &d[i].result, 1), "results");
                done++;
            }
        }
        EXPECT(NJ == done, "%d of %d jobs ended", done, NJ);
        for( int i = 0; (i < NJ) && (NJ == done); i++ ) {
            if( 0 == pass ) {
                first[i] = d[i].result;
                d[i].result = NULL;
                continue;
            }
            EXPECT((first[i]->nb_entries == d[i].result->nb_entries) &&
                   (0 == memcmp(first[i] + 1, d[i].result + 1, (size_t)first[i]->nb_entries * PARSEC_QPU_ENTRY_SIZE)),
                   "evaluation %d has other counts when the jobs are submitted in the opposite order", i);
            EXPECT(same_as_sampler(&d[i], 2000, 700 + (uint64_t)i), "evaluation %d: not the sampler's counts", i);
            free(first[i]);
        }
        EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
        for( int i = 0; i < NJ; i++ ) data_free(&d[i]);
    }
}

/* ------------------------------------------------------------------------
 * Faults
 * --------------------------------------------------------------------- */

static int lost_left = 0;
static parsec_qpu_mock_fault_t lost_fn(uint64_t tag, const parsec_qpu_request_t *request, uint64_t call, void *arg)
{
    (void)tag; (void)request; (void)call; (void)arg;
    if( lost_left <= 0 ) return PARSEC_QPU_MOCK_OK;
    lost_left--;
    return PARSEC_QPU_MOCK_LOST;
}

static void test_faults(void)
{
    parsec_qpu_backend_t *backend;
    parsec_qpu_mock_stats_t stats;
    parsec_qpu_event_t event;
    parsec_qpu_job_id_t id = 0, id2 = 0;
    parsec_qpu_request_t r;
    job_data_t d = data_new(4, 3, 1);
    int rc;

    /* fail: the job runs its time and ends in failure; the mock forgets it;
     * the next job of the same evaluation succeeds, with the counts the
     * first would have had. */
    backend = mock_open(2, "const:2000", "const:0", "fail:first=1");
    r = request_of(&d, 300, 9, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 11, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id) && (PARSEC_QPU_JOB_FAILED == event.state), "no failure reported");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, &d.result, 1), "a failed job gave results");
    r = request_of(&d, 300, 9, 2);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 12, &id2), "submit of the second attempt");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id2) && (PARSEC_QPU_JOB_DONE == event.state), "the second attempt did not end well");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id2, &d.result, 1), "results");
    EXPECT(same_as_sampler(&d, 300, 9), "the counts of a second attempt are not those of the evaluation");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.failures) && (1 == stats.computed), "%llu failures, %llu sampled", (unsigned long long)stats.failures,
           (unsigned long long)stats.computed);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* hang: the job never ends; cancelling it frees its slot. */
    backend = mock_open(1, "const:1000", "const:0", "hang:key=7");
    r = request_of(&d, 100, 7, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 21, &id), "submit");
    r = request_of(&d, 100, 8, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 22, &id2), "submit");
    EXPECT(0 == next_event(backend, 60, &event), "an event while the only slot is held by a job that hangs");
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel of a job that hangs");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id2), "the queued job did not run once the slot was freed");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id2, &d.result, 1), "results");
    EXPECT(0 == next_event(backend, 20, &event), "an event for a job that was cancelled");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.cancels) && (0 == stats.ghost_events), "%llu cancels, %llu ghost events",
           (unsigned long long)stats.cancels, (unsigned long long)stats.ghost_events);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* unavailable: the submission is not taken, and nothing was created. */
    backend = mock_open(2, "const:1000", "const:0", "unavailable:every=2");
    r = request_of(&d, 100, 1, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 31, &id), "first submission");
    r = request_of(&d, 100, 2, 1);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 32, &id2), "the second submission was to be deferred");
    parsec_qpu_mock_stats(&stats);
    EXPECT((2 == stats.submit_calls) && (1 == stats.submits) && (1 == stats.deferred), "%llu calls, %llu jobs, %llu deferred",
           (unsigned long long)stats.submit_calls, (unsigned long long)stats.submits, (unsigned long long)stats.deferred);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 32, &id2), "asking again");
    parsec_qpu_mock_stats(&stats);
    EXPECT((2 == stats.submits) && (0 == stats.tag_hits) && (0 == stats.duplicate_tags), "%llu jobs, %llu found by tag, %llu duplicate tags",
           (unsigned long long)stats.submits, (unsigned long long)stats.tag_hits, (unsigned long long)stats.duplicate_tags);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* unavailable for a time after the backend was opened. */
    backend = mock_open(2, "const:1000", "const:0", "unavailable:for=50");
    r = request_of(&d, 100, 1, 1);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 41, &id), "a submission in the first 50 ms was taken");
    { struct timespec ts = { 0, 60 * MS }; nanosleep(&ts, NULL); }
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 41, &id), "a submission after 60 ms was not taken");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* lost: the job is created and the caller is told to ask again; asking
     * again finds the job instead of making a second one, unless that reply
     * is lost too. Its end is not reported before the caller knows the job. */
    lost_left = 2;
    parsec_qpu_mock_set_fault_fn(lost_fn, NULL);
    backend = mock_open(2, "const:0", "const:0", "none");
    r = request_of(&d, 100, 5, 1);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 51, &id), "the reply was to be lost");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.submits) && (1 == stats.unknown_jobs), "the job whose reply was lost: %llu created, %u unknown to the caller",
           (unsigned long long)stats.submits, stats.unknown_jobs);
    EXPECT(0 == next_event(backend, 60, &event), "the end of a job was reported to a caller that does not know it");
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 51, &id), "the second reply was to be lost too");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.submits) && (0 == stats.tag_hits) && (2 == stats.deferred), "after two lost replies: %llu jobs, %llu found, %llu deferred",
           (unsigned long long)stats.submits, (unsigned long long)stats.tag_hits, (unsigned long long)stats.deferred);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 51, &id), "asking a third time");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.submits) && (1 == stats.tag_hits) && (0 == stats.key_overlaps) && (0 == stats.unknown_jobs),
           "after the job was found: %llu jobs, %llu found by tag, %llu overlaps, %u unknown", (unsigned long long)stats.submits,
           (unsigned long long)stats.tag_hits, (unsigned long long)stats.key_overlaps, stats.unknown_jobs);
    EXPECT(next_event(backend, 10000, &event) && (event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "the job that was found did not report its end");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &d.result, 1), "results");
    EXPECT(same_as_sampler(&d, 100, 5), "the counts of a job found by its tag");

    /* A submission that is given up on: the job the mock took is gone, it
     * reports nothing, and its tag is not taken again. */
    lost_left = 1;
    r = request_of(&d, 100, 6, 1);
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 52, &id), "the reply was to be lost");
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 52), "abandon");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.abandoned) && (0 == stats.unknown_jobs) && (0 == stats.cancels), "after abandon: %llu given up, %u unknown, %llu cancels",
           (unsigned long long)stats.abandoned, stats.unknown_jobs, (unsigned long long)stats.cancels);
    EXPECT(0 == next_event(backend, 30, &event), "an event for a submission that was given up on");
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 52), "abandon of a tag with no job");
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 9999), "abandon of a tag that never was");
    /* A job the caller knows is not the business of abandon(). */
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 53, &id), "submit");
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 53), "abandon of the tag of a known job");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "abandon() dropped a job the caller knows");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &d.result, 1), "results");
    parsec_qpu_mock_stats(&stats);
    EXPECT(1 == stats.abandoned, "%llu submissions given up on, expected 1", (unsigned long long)stats.abandoned);
    EXPECT(PARSEC_ERR_EXISTS == ops->submit(backend, &r, 1, 52, &id), "the tag of an abandoned submission was taken again");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    parsec_qpu_mock_set_fault_fn(NULL, NULL);

    /* Two jobs that fail at the same instant, read one at a time: the one
     * not read yet can still be cancelled, and is then not reported. */
    backend = mock_open(2, "const:1000", "const:0", "fail:first=1");
    r = request_of(&d, 100, 20, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 55, &id), "submit");
    r = request_of(&d, 100, 21, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 56, &id2), "submit");
    { struct timespec ts = { 0, 20 * MS }; nanosleep(&ts, NULL); }
    EXPECT(next_event(backend, 10000, &event) && (PARSEC_QPU_JOB_FAILED == event.state), "no failure reported");
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, (event.id == id) ? id2 : id), "cancel of a job that failed and was not reported yet");
    EXPECT(0 == next_event(backend, 30, &event), "a failure was reported for a job that was cancelled first");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* reject: refused for good, with an error that is not "ask again". */
    backend = mock_open(2, "const:1000", "const:0", "reject:key=9");
    r = request_of(&d, 100, 9, 1);
    rc = ops->submit(backend, &r, 1, 61, &id);
    EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_OUT_OF_RESOURCE != rc), "a rejected submission returned %d", rc);
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.rejected) && (0 == stats.submits), "%llu rejected, %llu jobs", (unsigned long long)stats.rejected, (unsigned long long)stats.submits);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* noresult: the job ends well and has nothing to give. */
    backend = mock_open(2, "const:1000", "const:0", "noresult:key=3");
    r = request_of(&d, 100, 3, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 71, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (PARSEC_QPU_JOB_DONE == event.state), "no end reported");
    rc = ops->results(backend, id, &d.result, 1);
    EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_NOT_FOUND != rc), "results that cannot be had returned %d", rc);
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, &d.result, 1), "the job was not forgotten after its results failed");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* duplicate: the end is reported twice, the second time for a job that is gone. */
    backend = mock_open(2, "const:1000", "const:0", "duplicate:key=4");
    r = request_of(&d, 100, 4, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 81, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no end reported");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &d.result, 1), "results");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "the end was not reported a second time");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, &d.result, 1), "results were given for the second report");
    parsec_qpu_mock_stats(&stats);
    EXPECT(1 == stats.ghost_events, "%llu ghost events", (unsigned long long)stats.ghost_events);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* late: the job never ends, and its end is reported after it was cancelled. */
    backend = mock_open(2, "const:1000", "const:0", "late:key=6");
    r = request_of(&d, 100, 6, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 91, &id), "submit");
    EXPECT(0 == next_event(backend, 40, &event), "a job that was to hang ended");
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no late report of the cancelled job");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, &d.result, 1), "results were given for a cancelled job");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* By probability: the fate of an attempt depends on the seed, the key
     * and the attempt, not on when it is submitted. */
    {
        enum { NJ = 400 };
        static char fate[2][NJ];
        int failed = 0;

        for( int pass = 0; pass < 2; pass++ ) {
            int done = 0;
            backend = mock_open(64, "const:0", "const:0", "fail:p=0.3");
            for( int n = 0; n < NJ; n++ ) {
                const int i = (0 == pass) ? n : NJ - 1 - n;
                r = request_of(&d, 10, 5000 + (uint64_t)i, 1);
                EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, (uint64_t)(i + 1), &id), "submit");
                /* Identifiers are given in order, from 1. */
                EXPECT(id == (parsec_qpu_job_id_t)(n + 1), "job identifier %llu for the submission %d", (unsigned long long)id, n + 1);
            }
            while( (done < NJ) && next_event(backend, 20000, &event) ) {
                const int n = (int)event.id - 1, i = (0 == pass) ? n : NJ - 1 - n;
                fate[pass][i] = (PARSEC_QPU_JOB_FAILED == event.state);
                if( PARSEC_QPU_JOB_DONE == event.state ) (void)ops->results(backend, event.id, &d.result, 1);
                done++;
            }
            EXPECT(NJ == done, "%d of %d jobs ended", done, NJ);
            EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
        }
        for( int i = 0; i < NJ; i++ ) failed += fate[0][i];
        EXPECT((failed > NJ / 5) && (failed < 2 * NJ / 5), "%d of %d jobs failed with a probability of 0.3", failed, NJ);
        EXPECT(0 == memcmp(fate[0], fate[1], NJ), "which jobs fail depends on the order of submission");
    }

    data_free(&d);
}

static int fault_fn_calls = 0;
static parsec_qpu_mock_fault_t fault_fn(uint64_t tag, const parsec_qpu_request_t *request, uint64_t call, void *arg)
{
    (void)tag; (void)arg;
    fault_fn_calls++;
    return ((1 == call) && (77 == request->key)) ? PARSEC_QPU_MOCK_REJECT : PARSEC_QPU_MOCK_OK;
}

static void test_fault_fn(void)
{
    /* A function decides, and the parameter does not. */
    parsec_qpu_backend_t *backend;
    job_data_t d = data_new(3, 1, 1);
    parsec_qpu_request_t r = request_of(&d, 10, 77, 1);
    parsec_qpu_event_t event;
    parsec_qpu_job_id_t id;
    int rc;

    parsec_qpu_mock_set_fault_fn(fault_fn, NULL);
    backend = mock_open(1, "const:0", "const:0", "fail:first=100");
    rc = ops->submit(backend, &r, 1, 1, &id);
    EXPECT((PARSEC_SUCCESS != rc) && (PARSEC_ERR_OUT_OF_RESOURCE != rc), "the function's verdict was not followed (%d)", rc);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &id), "second call");
    EXPECT(next_event(backend, 10000, &event) && (PARSEC_QPU_JOB_DONE == event.state), "the parameter was followed with a function installed");
    EXPECT(2 == fault_fn_calls, "the function was called %d times for 2 submissions", fault_fn_calls);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    parsec_qpu_mock_set_fault_fn(NULL, NULL);
    data_free(&d);
}

/* ------------------------------------------------------------------------
 * What the mock watches its caller for
 * --------------------------------------------------------------------- */

static void test_watchdogs(void)
{
    parsec_qpu_backend_t *backend = mock_open(4, "const:1000", "const:0", "none");
    parsec_qpu_mock_stats_t stats;
    parsec_qpu_event_t event;
    parsec_qpu_job_id_t id, id2;
    job_data_t d = data_new(4, 2, 1), e = data_new(4, 5, 1);
    parsec_qpu_request_t r = request_of(&d, 50, 1, 1);

    /* A tag asked for twice while its job is known: the same job is the
     * answer, and the mock counts that its caller asked. Once the job is
     * gone, its tag is refused. */
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 500, &id), "submit");
    id2 = 0;
    EXPECT((PARSEC_SUCCESS == ops->submit(backend, &r, 1, 500, &id2)) && (id2 == id), "asking again for a known job did not give that job");
    EXPECT(next_event(backend, 10000, &event), "no event");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &d.result, 1), "results");
    EXPECT(PARSEC_ERR_EXISTS == ops->submit(backend, &r, 1, 500, &id2), "the tag of a job that is gone was taken again");
    parsec_qpu_mock_stats(&stats);
    EXPECT(2 == stats.duplicate_tags, "%llu duplicate tags counted, expected 2", (unsigned long long)stats.duplicate_tags);
    EXPECT(1 == stats.submits, "%llu jobs", (unsigned long long)stats.submits);

    /* Two jobs of one evaluation held at once. */
    r = request_of(&d, 50, 2, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 501, &id), "submit");
    r = request_of(&d, 50, 2, 2);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 502, &id2), "submit");
    parsec_qpu_mock_stats(&stats);
    EXPECT(1 == stats.key_overlaps, "%llu overlaps counted, expected 1", (unsigned long long)stats.key_overlaps);
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel");
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id2), "cancel");
    /* One after the other is what a retry looks like, and is fine. */
    r = request_of(&d, 50, 2, 3);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 503, &id), "submit");
    EXPECT(next_event(backend, 10000, &event), "no event");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &d.result, 1), "results");
    parsec_qpu_mock_stats(&stats);
    EXPECT(1 == stats.key_overlaps, "a job after another of the same evaluation was counted as an overlap");

    /* Inputs that change while the mock holds the job. */
    r = request_of(&e, 50, 3, 1);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 504, &id), "submit");
    EXPECT(next_event(backend, 10000, &event), "no event");
    e.params[1] += 1.0;
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &e.result, 1), "results");
    parsec_qpu_mock_stats(&stats);
    EXPECT(1 == stats.inputs_changed, "%llu changed inputs counted, expected 1", (unsigned long long)stats.inputs_changed);

    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    data_free(&d);
    data_free(&e);
}

/* ------------------------------------------------------------------------
 * cancel() in every state
 * --------------------------------------------------------------------- */

static void test_cancel(void)
{
    parsec_qpu_backend_t *backend = mock_open(1, "const:40000", "const:0", "none");
    parsec_qpu_mock_stats_t stats;
    parsec_qpu_event_t event;
    parsec_qpu_job_id_t id[3];
    job_data_t d[3];
    int64_t t0;

    for( int i = 0; i < 3; i++ ) {
        parsec_qpu_request_t r;
        d[i] = data_new(4, i, 1);
        r = request_of(&d[i], 50, 10 + (uint64_t)i, 1);
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 600 + (uint64_t)i, &id[i]), "submit");
    }
    /* Job 0 runs, jobs 1 and 2 queue. Cancel one that queues, then the one
     * that runs: the one that is left gets the slot. */
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id[1]), "cancel of a queued job");
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id[0]), "cancel of a running job");
    t0 = parsec_qpu_now_ns();
    EXPECT(next_event(backend, 10000, &event) && (event.id == id[2]), "the job that was left did not run");
    EXPECT(parsec_qpu_now_ns() - t0 >= 35 * MS, "the job that was left ended %.3f ms after it got the slot", (parsec_qpu_now_ns() - t0) * 1e-6);
    /* Finished, reported, not taken: cancelled instead. */
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id[2]), "cancel of a finished job");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id[2], &d[2].result, 1), "results of a cancelled job");
    EXPECT(0 == next_event(backend, 20, &event), "an event after every job was cancelled");
    parsec_qpu_mock_stats(&stats);
    EXPECT((3 == stats.cancels) && (0 == stats.results), "%llu cancels, %llu results", (unsigned long long)stats.cancels, (unsigned long long)stats.results);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    for( int i = 0; i < 3; i++ ) data_free(&d[i]);

    /* A job whose circuit a helper is sampling: when cancel() returns the
     * inputs may go away, which they do here at once. Under a memory checker
     * a helper that still read them would show. */
    {
        job_data_t big = data_new(18, 1, 12);
        parsec_qpu_request_t r = request_of(&big, 100, 1, 1);
        struct timespec ts = { 0, 20 * MS };
        parsec_qpu_job_id_t bid;

        backend = mock_open(1, "const:0", "const:0", "none");
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &bid), "submit");
        nanosleep(&ts, NULL);       /* a helper is at it by now */
        t0 = parsec_qpu_now_ns();
        EXPECT(PARSEC_SUCCESS == ops->cancel(backend, bid), "cancel of a job being sampled");
        memset(big.circuit, 0xee, sizeof(parsec_qpu_circuit_t));
        data_free(&big);
        printf("cancel of a job being sampled (18 qubits) returned after %.3f ms\n", (parsec_qpu_now_ns() - t0) * 1e-6);
        EXPECT(0 == next_event(backend, 30, &event), "an event for a job cancelled while it was sampled");
        EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    }

    /* The same for a submission that is given up on while its job is sampled. */
    {
        job_data_t big = data_new(18, 3, 12);
        parsec_qpu_request_t r = request_of(&big, 100, 2, 1);
        struct timespec ts = { 0, 20 * MS };
        parsec_qpu_job_id_t bid;

        lost_left = 1;
        parsec_qpu_mock_set_fault_fn(lost_fn, NULL);
        backend = mock_open(1, "const:0", "const:0", "none");
        EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, &r, 1, 7, &bid), "the reply was to be lost");
        nanosleep(&ts, NULL);
        EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 7), "abandon of a submission whose job is being sampled");
        memset(big.circuit, 0xee, sizeof(parsec_qpu_circuit_t));
        data_free(&big);
        EXPECT(0 == next_event(backend, 30, &event), "an event for a submission given up on while its job was sampled");
        parsec_qpu_mock_stats(&stats);
        EXPECT((1 == stats.abandoned) && (0 == stats.unknown_jobs), "%llu given up on, %u unknown", (unsigned long long)stats.abandoned, stats.unknown_jobs);
        EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
        parsec_qpu_mock_set_fault_fn(NULL, NULL);
    }

    /* A job that outlasts its modelled time because it is still being
     * sampled: it ends when the sampling does, with the right answer. */
    {
        job_data_t big = data_new(16, 2, 6);
        parsec_qpu_request_t r = request_of(&big, 200, 9, 1);
        parsec_qpu_job_id_t bid;

        backend = mock_open(1, "const:0", "const:0", "none");
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 1, &bid), "submit");
        EXPECT(next_event(backend, 120000, &event) && (event.id == bid) && (PARSEC_QPU_JOB_DONE == event.state), "the job did not end");
        EXPECT(PARSEC_SUCCESS == ops->results(backend, bid, &big.result, 1), "results");
        EXPECT(big.result->t_end >= big.result->t_start, "the job ended before it started");
        EXPECT(same_as_sampler(&big, 200, 9), "a job that outlasted its modelled time has other counts than the sampler's");
        parsec_qpu_mock_stats(&stats);
        EXPECT(1 == stats.late, "%llu jobs counted as late, expected the one", (unsigned long long)stats.late);
        printf("a job of no modelled time on 16 qubits ended after %.3f ms, the time its sampling took\n",
               (big.result->t_end - big.result->t_start) * 1e-6);
        /* Closing with a job still being sampled must not hang or crash. */
        r = request_of(&big, 200, 10, 1);
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r, 1, 2, &bid), "submit");
        EXPECT(PARSEC_SUCCESS == ops->close(backend), "close with a job being sampled");
        data_free(&big);
    }
}

/* ------------------------------------------------------------------------
 * Jobs of several circuits
 * --------------------------------------------------------------------- */

/* A circuit runs for 1 to 4 ms, by its key. */
static int64_t circuit_delay_by_key(uint64_t tag, const parsec_qpu_request_t *request, void *arg)
{
    (void)tag;
    (*(int*)arg)++;
    return (int64_t)(1 + request->key % 4) * MS;
}

static uint64_t batch_fault_key = 0;
static parsec_qpu_mock_fault_t batch_fault = PARSEC_QPU_MOCK_OK;
static int batch_fault_asked = 0;

/* The circuit of one evaluation has a fate of its own. */
static parsec_qpu_mock_fault_t fault_by_key_fn(uint64_t tag, const parsec_qpu_request_t *request, uint64_t call, void *arg)
{
    (void)tag; (void)call; (void)arg;
    batch_fault_asked++;
    return (request->key == batch_fault_key) ? batch_fault : PARSEC_QPU_MOCK_OK;
}

static void test_batches(void)
{
    enum { N = 5 };
    parsec_qpu_backend_t *backend;
    parsec_qpu_mock_stats_t stats;
    parsec_qpu_caps_t caps;
    parsec_qpu_event_t event;
    parsec_qpu_job_id_t id = 0, id2 = 0;
    job_data_t d[N];
    parsec_qpu_request_t r[N];
    parsec_qpu_result_t *results[N];
    int64_t ran, fixed;
    int asked = 0;

    for( int i = 0; i < N; i++ ) {
        d[i]       = data_new(3 + i % 3, 20 + i, 2);
        r[i]       = request_of(&d[i], 300, 500 + (uint64_t)i, 1);
        results[i] = d[i].result;
    }

    /* One job of N circuits: one identifier, one event, a result per
     * circuit in the order of the requests, each the sampler's for its own
     * inputs and key; it ran for the part of a job plus N parts of a circuit. */
    parsec_qpu_params.mock_job_delay    = (char*)"const:10000";
    parsec_qpu_params.mock_max_circuits = 8;
    backend = mock_open(1, "const:3000", "const:0", "none");
    memset(&caps, 0, sizeof(caps));
    (void)ops->capabilities(backend, &caps);
    EXPECT(8 == caps.max_circuits_per_job, "the mock says it takes %u circuits per job, 8 were asked for", caps.max_circuits_per_job);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9001, &id), "submit of %d circuits", N);
    EXPECT(next_event(backend, 10000, &event) && (event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "no end for the job");
    EXPECT(0 == next_event(backend, 20, &event), "a second event for one job of %d circuits", N);
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, results, N), "results of %d circuits", N);
    parsec_qpu_mock_stats(&stats);
    for( int i = 0; i < N; i++ ) {
        ran = d[i].result->t_end - d[i].result->t_start;
        EXPECT(PARSEC_QPU_STATUS_OK == d[i].result->status, "circuit %d: status %u", i, (unsigned)d[i].result->status);
        EXPECT(d[i].circuit->nb_qubits == d[i].result->nb_qubits, "circuit %d: %u qubits in its result, it has %u",
               i, (unsigned)d[i].result->nb_qubits, d[i].circuit->nb_qubits);
        EXPECT(same_as_sampler(&d[i], 300, 500 + (uint64_t)i), "circuit %d of the job has other counts than the sampler's", i);
        EXPECT((ran == (10 + 3 * N) * MS) || ((stats.late > 0) && (ran > (10 + 3 * N) * MS)),
               "circuit %d: its job ran for %lld ns, not for 10 ms and %d times 3 ms", i, (long long)ran, N);
        EXPECT((d[i].result->t_start == d[0].result->t_start) && (d[i].result->t_end == d[0].result->t_end),
               "circuit %d has other times than circuit 0 of the same job", i);
    }
    EXPECT((1 == stats.submit_calls) && (1 == stats.submits) && (N == stats.circuits) && (N == stats.results) &&
           (N == stats.computed) && (N == stats.max_circuits) && (0 == stats.key_overlaps) && (1 == stats.max_held),
           "after one job of %d circuits: %llu calls, %llu jobs, %llu circuits, %llu results, %llu sampled, at most %u in a job, "
           "%llu overlaps, %u held", N, (unsigned long long)stats.submit_calls, (unsigned long long)stats.submits,
           (unsigned long long)stats.circuits, (unsigned long long)stats.results, (unsigned long long)stats.computed,
           stats.max_circuits, (unsigned long long)stats.key_overlaps, stats.max_held);

    /* The same evaluation alone: the same counts, in a job of one part and one circuit. */
    r[2] = request_of(&d[2], 300, 502, 2);
    memset(d[2].result, 0, parsec_qpu_result_size(d[2].max_entries));
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, &r[2], 1, 9002, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no end for the job");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, &results[2], 1), "results");
    EXPECT(same_as_sampler(&d[2], 300, 502), "an evaluation has other counts alone than with others");
    parsec_qpu_mock_stats(&stats);
    ran = d[2].result->t_end - d[2].result->t_start;
    EXPECT((ran == 13 * MS) || ((stats.late > 0) && (ran > 13 * MS)), "a job of one circuit ran for %lld ns, not for 10 + 3 ms", (long long)ran);
    r[2] = request_of(&d[2], 300, 502, 1);

    /* results() for another number of circuits than the job has: an error,
     * nothing written, and the job is forgotten all the same. */
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, 3, 9003, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no end for the job");
    memset(d[0].result, 0x5a, sizeof(parsec_qpu_result_t));
    EXPECT(PARSEC_SUCCESS != ops->results(backend, id, results, 2), "the results of a job of 3 circuits were given as 2");
    EXPECT(0x5a == d[0].result->status, "results() wrote a result and failed");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, results, 3), "a job whose results were asked for wrongly is still there");

    /* More circuits than the mock takes, or none: refused for good, no job. */
    parsec_qpu_mock_stats(&stats);
    {
        const uint64_t before = stats.submits;
        parsec_qpu_request_t many[9];
        for( int i = 0; i < 9; i++ ) many[i] = request_of(&d[i % N], 300, 600 + (uint64_t)i, 1);
        EXPECT(PARSEC_ERR_NOT_SUPPORTED == ops->submit(backend, many, 9, 9004, &id), "9 circuits were taken in a job of at most 8");
        EXPECT(PARSEC_ERR_NOT_SUPPORTED == ops->submit(backend, many, 0, 9005, &id), "a job of no circuit was taken");
        parsec_qpu_mock_stats(&stats);
        EXPECT(before == stats.submits, "a refused job was created");
        EXPECT(0 == next_event(backend, 20, &event), "an event for a refused job");
    }

    /* Two circuits of one evaluation in one job, and one while another job
     * holds the same: each is counted. */
    {
        parsec_qpu_request_t twice[2] = { request_of(&d[0], 300, 700, 1), request_of(&d[1], 300, 700, 1) };
        parsec_qpu_request_t again = request_of(&d[3], 300, 700, 1);
        const uint64_t before = stats.key_overlaps;

        EXPECT(PARSEC_SUCCESS == ops->submit(backend, twice, 2, 9006, &id), "submit");
        parsec_qpu_mock_stats(&stats);
        EXPECT(before + 1 == stats.key_overlaps, "two circuits of one evaluation in one job: %llu overlaps counted, expected 1",
               (unsigned long long)(stats.key_overlaps - before));
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, &again, 1, 9007, &id2), "submit");
        parsec_qpu_mock_stats(&stats);
        EXPECT(before + 2 == stats.key_overlaps, "a circuit of an evaluation that a job of two holds was not counted");
        EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel of a running job of two circuits");
        EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id2), "cancel of the job that was queued behind it");
        EXPECT(0 == next_event(backend, 40, &event), "an event for a cancelled job");
    }
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");

    /* The time of a circuit, when a test decides it, is asked for each
     * circuit of the job, and the job runs for their sum and its own part. */
    parsec_qpu_mock_set_delay_fn(circuit_delay_by_key, &asked);
    backend = mock_open(1, "const:0", "const:0", "none");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9010, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no end for the job");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, results, N), "results");
    EXPECT(N == asked, "the time of a circuit was asked %d times for a job of %d", asked, N);
    parsec_qpu_mock_stats(&stats);
    ran = d[0].result->t_end - d[0].result->t_start;
    /* keys 500 .. 504: 1, 2, 3, 4 and 1 ms */
    EXPECT((ran == (10 + 11) * MS) || ((stats.late > 0) && (ran > 21 * MS)), "the job ran for %lld ns, not for 10 + 11 ms", (long long)ran);
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    parsec_qpu_mock_set_delay_fn(NULL, NULL);

    /* What is drawn for the job comes from its first circuit: the same
     * first circuit and attempt give the same part, whatever follows. */
    parsec_qpu_params.mock_job_delay = (char*)"uniform:1000:9000";
    backend = mock_open(1, "const:2000", "const:0", "none");
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, 3, 9020, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no end for the job");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, results, 3), "results");
    parsec_qpu_mock_stats(&stats);
    fixed = (d[0].result->t_end - d[0].result->t_start) - 3 * 2 * MS;
    EXPECT(((fixed >= 1 * MS) && (fixed <= 9 * MS)) || (stats.late > 0), "the part of the job is %lld ns, outside 1 to 9 ms", (long long)fixed);
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, 2, 9021, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no end for the job");
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, results, 2), "results");
    parsec_qpu_mock_stats(&stats);
    EXPECT(((d[0].result->t_end - d[0].result->t_start) - 2 * 2 * MS == fixed) || (stats.late > 0),
           "two jobs with the same first circuit have parts of %lld and %lld ns", (long long)fixed,
           (long long)((d[0].result->t_end - d[0].result->t_start) - 2 * 2 * MS));
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close");
    parsec_qpu_params.mock_job_delay = (char*)"const:0";

    /* One fate for the job: that of the first of its circuits that has one.
     * The circuits after it are not asked. */
    parsec_qpu_mock_set_fault_fn(fault_by_key_fn, NULL);
    backend = mock_open(2, "const:2000", "const:0", "none");

    batch_fault_key = 502; batch_fault = PARSEC_QPU_MOCK_FAIL; batch_fault_asked = 0;
    for( int i = 0; i < N; i++ ) memset(d[i].result, 0x5a, sizeof(parsec_qpu_result_t));
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9030, &id), "submit");
    EXPECT(3 == batch_fault_asked, "the fate of a job whose third circuit fails was asked %d times", batch_fault_asked);
    EXPECT(next_event(backend, 10000, &event) && (event.id == id) && (PARSEC_QPU_JOB_FAILED == event.state),
           "a job with a circuit that fails did not end in failure");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, results, N), "a failed job of %d circuits gave results", N);
    for( int i = 0; i < N; i++ ) {
        EXPECT(0x5a == d[i].result->status, "the result of circuit %d of a failed job was written", i);
    }
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.failures) && (0 == stats.results) && (0 == stats.computed),
           "a failed job of %d circuits: %llu failures, %llu results, %llu sampled", N,
           (unsigned long long)stats.failures, (unsigned long long)stats.results, (unsigned long long)stats.computed);

    batch_fault_key = 504; batch_fault = PARSEC_QPU_MOCK_REJECT;
    EXPECT(PARSEC_ERR_NOT_SUPPORTED == ops->submit(backend, r, N, 9031, &id), "a job whose last circuit is refused was taken");
    batch_fault_key = 500; batch_fault = PARSEC_QPU_MOCK_UNAVAILABLE;
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, r, N, 9032, &id), "a job whose first circuit finds nobody was taken");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.submits) && (N == stats.circuits) && (0 == stats.unknown_jobs),
           "after a refusal and an unavailability: %llu jobs, %llu circuits, %u unknown", (unsigned long long)stats.submits,
           (unsigned long long)stats.circuits, stats.unknown_jobs);

    /* A job that ends well and whose results cannot be had: an error for
     * the whole job, and no result written. */
    batch_fault_key = 501; batch_fault = PARSEC_QPU_MOCK_NORESULT;
    for( int i = 0; i < N; i++ ) memset(d[i].result, 0x5a, sizeof(parsec_qpu_result_t));
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9033, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id) && (PARSEC_QPU_JOB_DONE == event.state), "no end for the job");
    EXPECT(PARSEC_ERROR == ops->results(backend, id, results, N), "the results of a job that has none were given");
    for( int i = 0; i < N; i++ ) {
        EXPECT(0x5a == d[i].result->status, "the result of circuit %d was written by a results() that failed", i);
    }

    /* A job of several circuits that hangs is cancelled as one, and one
     * that is given up on while its answer is lost is dropped as one. */
    batch_fault_key = 503; batch_fault = PARSEC_QPU_MOCK_HANG;
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9034, &id), "submit");
    EXPECT(0 == next_event(backend, 40, &event), "a job that hangs ended");
    EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel of a job of %d circuits that hangs", N);
    batch_fault_key = 500; batch_fault = PARSEC_QPU_MOCK_LOST;
    EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, r, N, 9035, &id), "the reply was to be lost");
    parsec_qpu_mock_stats(&stats);
    EXPECT(1 == stats.unknown_jobs, "%u jobs unknown to the caller after a lost reply", stats.unknown_jobs);
    EXPECT(PARSEC_SUCCESS == ops->abandon(backend, 9035), "abandon");
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.abandoned) && (0 == stats.unknown_jobs), "%llu given up on, %u unknown",
           (unsigned long long)stats.abandoned, stats.unknown_jobs);
    EXPECT(0 == next_event(backend, 40, &event), "an event for a job that was cancelled or given up on");

    /* A job that somebody at the provider cancels: it runs its time, ends
     * as cancelled, and is gone, with no result for any of its circuits. */
    batch_fault_key = 502; batch_fault = PARSEC_QPU_MOCK_CANCELLED;
    for( int i = 0; i < N; i++ ) memset(d[i].result, 0x5a, sizeof(parsec_qpu_result_t));
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9040, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id) && (PARSEC_QPU_JOB_CANCELLED == event.state),
           "a job that the provider cancels did not end as cancelled");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->results(backend, id, results, N), "a job that was cancelled at the provider gave results");
    EXPECT(PARSEC_ERR_NOT_FOUND == ops->cancel(backend, id), "a job that was cancelled at the provider is still there");
    for( int i = 0; i < N; i++ ) {
        EXPECT(0x5a == d[i].result->status, "the result of circuit %d of a cancelled job was written", i);
    }
    parsec_qpu_mock_stats(&stats);
    EXPECT((1 == stats.provider_cancels) && (1 == stats.failures) && (0 == stats.cancels - 1),
           "after a job cancelled at the provider: %llu such, %llu failures, %llu cancelled by the caller",
           (unsigned long long)stats.provider_cancels, (unsigned long long)stats.failures, (unsigned long long)stats.cancels);

    /* Circuits of one job that ask for different numbers of shots, and a
     * job asked for again with other circuits than it has: each is counted.
     * Asking again for the same circuits is not. */
    batch_fault_key = 500; batch_fault = PARSEC_QPU_MOCK_LOST;
    {
        parsec_qpu_request_t mixed[2] = { request_of(&d[0], 300, 800, 1), request_of(&d[1], 301, 801, 1) };
        parsec_qpu_request_t other[3] = { r[0], r[2], r[1] };

        EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, r, 3, 9041, &id), "the reply was to be lost");
        batch_fault_key = 0; batch_fault = PARSEC_QPU_MOCK_OK;
        parsec_qpu_mock_stats(&stats);
        EXPECT((0 == stats.changed_jobs) && (0 == stats.mixed_shots), "%llu changed jobs and %llu jobs of mixed shots before there was one",
               (unsigned long long)stats.changed_jobs, (unsigned long long)stats.mixed_shots);
        batch_fault_key = 500; batch_fault = PARSEC_QPU_MOCK_LOST;
        EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, r, 3, 9041, &id), "the reply was to be lost again");
        parsec_qpu_mock_stats(&stats);
        EXPECT(0 == stats.changed_jobs, "asking again for the same job was counted as asking for another");
        EXPECT(PARSEC_ERR_OUT_OF_RESOURCE == ops->submit(backend, r, 2, 9041, &id), "the reply was to be lost again");
        parsec_qpu_mock_stats(&stats);
        EXPECT(1 == stats.changed_jobs, "a job of three circuits asked for again with two: %llu counted", (unsigned long long)stats.changed_jobs);
        batch_fault_key = 0; batch_fault = PARSEC_QPU_MOCK_OK;
        EXPECT(PARSEC_SUCCESS == ops->submit(backend, other, 3, 9041, &id), "the job was to be found");
        parsec_qpu_mock_stats(&stats);
        EXPECT(2 == stats.changed_jobs, "a job asked for again with its circuits in another order: %llu counted in all, expected 2",
               (unsigned long long)stats.changed_jobs);
        EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel");

        EXPECT(PARSEC_SUCCESS == ops->submit(backend, mixed, 2, 9042, &id), "submit");
        parsec_qpu_mock_stats(&stats);
        EXPECT(1 == stats.mixed_shots, "a job whose circuits ask for 300 and 301 shots: %llu counted", (unsigned long long)stats.mixed_shots);
        EXPECT(PARSEC_SUCCESS == ops->cancel(backend, id), "cancel");
        EXPECT(0 == next_event(backend, 40, &event), "an event for a cancelled job");
    }

    /* Inputs that change under a job of several circuits: counted per circuit. */
    batch_fault_key = 0; batch_fault = PARSEC_QPU_MOCK_OK;
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9036, &id), "submit");
    EXPECT(next_event(backend, 10000, &event) && (event.id == id), "no end for the job");
    d[1].params[0] += 1.0;
    d[4].params[1] += 1.0;
    EXPECT(PARSEC_SUCCESS == ops->results(backend, id, results, N), "results");
    parsec_qpu_mock_stats(&stats);
    EXPECT(2 == stats.inputs_changed, "%llu changed inputs counted in a job where two circuits had theirs changed",
           (unsigned long long)stats.inputs_changed);

    /* Closing with a job of several circuits being sampled must not hang. */
    EXPECT(PARSEC_SUCCESS == ops->submit(backend, r, N, 9037, &id), "submit");
    EXPECT(PARSEC_SUCCESS == ops->close(backend), "close with a job of %d circuits held", N);
    parsec_qpu_mock_set_fault_fn(NULL, NULL);

    parsec_qpu_params.mock_max_circuits = 64;
    for( int i = 0; i < N; i++ ) data_free(&d[i]);
}

int main(void)
{
    test_delays();
    test_one_job();
    test_slots();
    test_wait_wakeup();
    test_order_independence();
    test_faults();
    test_fault_fn();
    test_watchdogs();
    test_cancel();
    test_batches();

    printf("qpu_mock_test: %s\n", nb_errors ? "FAILED" : "passed");
    return nb_errors ? 1 : 0;
}
