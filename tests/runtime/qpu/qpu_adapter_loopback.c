/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * An adapter for tests: it speaks the protocol of the adapter backend on a
 * Unix-domain socket and answers from the exact sampler, as the mock backend
 * does. Neither Python nor a network is involved, and no quantum device can
 * be reached through it.
 *
 * It reads the circuits it is given with the OpenQASM 3 reader of the
 * component, so a job that went through the adapter backend and this
 * program gives the counts of the sampler for the same seed and key only if
 * the circuit that was written is read back as it was.
 *
 * It is one thread around poll(). It runs until it is sent SIGTERM, and
 * then exits with 0, or with 3 when the client broke the protocol (the
 * reasons are printed), so that a test that looks at its exit code sees
 * what the client did wrong.
 *
 *   -s PATH    the socket to listen on (required)
 *   -S SEED    seed of the sampler (default 1, as device_qpu_mock_seed)
 *   -d US      how long a job runs, in microseconds (default 2000)
 *   -q N       widest circuit taken (default 24)
 *   -c N       most circuits in one job (default 64)
 *   -m N       most shots (default 0: no limit)
 *   -P         say that a quantum device is driven (none is)
 *   -N         no job may be submitted: one that is counts as a broken protocol
 *   -b N       connections that may wait to be accepted (default 16)
 *   -T FILE    append every line received to FILE
 *   -E TEXT    a text that must not be in anything received
 *   -F FAULTS  name:N[,name:N...]: what to do at the N-th submission
 *              received, counted from 1:
 *      drop_before  close the connection, and take no job
 *      drop_submit  take the job, and close the connection without answering
 *      silent       take the job, and never answer
 *      garbage      take the job, and answer with something that is no JSON
 *      stale        take the job, and answer under the number of the request
 *                   before, as an answer that comes late would
 *      busy         answer "busy": try later
 *      reject       refuse the job for good
 *      fail         the job ends in failure
 *      cancelled    the job ends cancelled
 *      badcounts    the result of the job misses a shot
 *      nocircuit    forget every circuit first
 *      forget       forget every session, job and circuit and close every
 *                   connection, as an adapter that was started again
 *      turn         the same, and say from then on that a quantum device is driven
 *
 * Beyond the protocol it answers {"op":"stats"} with its counters.
 */

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"
#include "parsec/mca/device/qpu/device_qpu_qasm.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#define LB_MAX_CONNS     64
#define LB_MAX_SESSIONS  256

enum { LB_RUNNING = 0, LB_DONE, LB_FAILED, LB_CANCELLED };
enum { F_DROP_BEFORE = 0, F_DROP_SUBMIT, F_SILENT, F_GARBAGE, F_BUSY, F_REJECT, F_FAIL, F_CANCELLED, F_BADCOUNTS,
       F_NOCIRCUIT, F_FORGET, F_TURN, F_STALE, F_MAX };
static const char *fault_names[F_MAX] = {
    "drop_before", "drop_submit", "silent", "garbage", "busy", "reject", "fail", "cancelled", "badcounts",
    "nocircuit", "forget", "turn", "stale"
};

typedef struct lb_circuit_s {
    struct lb_circuit_s  *next;
    uint64_t              id;
    parsec_qpu_circuit_t *circuit;
} lb_circuit_t;

typedef struct lb_job_s {
    struct lb_job_s      *next;
    uint64_t              id, tag, session;
    int                   state, reported, end_as, bad_counts;
    int64_t               t_start, t_done;
    int                   nb;
    parsec_qpu_result_t **counts;
} lb_job_t;

typedef struct lb_conn_s {
    int       fd;
    char     *in;
    size_t    len, room;
    uint64_t  session;
    int       has_session;
    int       waiting, wait_max, silent;
    int64_t   wait_seq, wait_deadline;
} lb_conn_t;

static lb_conn_t     conns[LB_MAX_CONNS];
static int           nb_conns = 0;
static lb_job_t     *jobs = NULL;
static lb_circuit_t *circuits = NULL;
static uint64_t      sessions[LB_MAX_SESSIONS];
static int           nb_sessions = 0;
static uint64_t      next_job = 1;
static parsec_qpu_sampler_t *sampler;
static parsec_qpu_text_t     out;

static uint64_t opt_seed = 1;
static int64_t  opt_duration_ns = 2000000;
static int      opt_qubits = 24, opt_max_circuits = 64, opt_max_shots = 0, opt_physical = 0, opt_no_job = 0, opt_backlog = 16;
static const char *opt_transcript = NULL, *opt_secret = NULL;
static long     faults[F_MAX];

static struct {
    long hellos, resumes, prepares, submits, jobs, tag_hits, waits, wakes, results, unknown_results,
         cancels, abandons, abandon_hits, byes, violations, lines;
} counters;

static volatile sig_atomic_t stop = 0;
static void on_term(int sig) { (void)sig; stop = 1; }

static void violation(const char *format, ...)
{
    va_list ap;
    counters.violations++;
    fprintf(stderr, "qpu_adapter_loopback: the client broke the protocol: ");
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
    fprintf(stderr, "\n");
}

static void conn_close(lb_conn_t *conn)
{
    if( conn->fd >= 0 ) close(conn->fd);
    free(conn->in);
    *conn = conns[--nb_conns];
    conns[nb_conns].fd = -1;
    conns[nb_conns].in = NULL;
}

static void conn_send(lb_conn_t *conn, const char *text, size_t length)
{
    while( length > 0 ) {
        const ssize_t n = send(conn->fd, text, length, MSG_NOSIGNAL);
        if( n <= 0 ) {
            if( (n < 0) && (EINTR == errno) ) continue;
            return;                                 /* a client that is gone finds out by itself */
        }
        text += n;
        length -= (size_t)n;
    }
}

/* Start an answer: {"seq":<n>,"ok":<bool> and whatever the caller adds. */
static void reply_begin(int64_t seq, int ok)
{
    parsec_qpu_text_reset(&out);
    parsec_qpu_text_add(&out, "{\"seq\":%lld,\"ok\":%s", (long long)seq, ok ? "true" : "false");
}

static void reply_send(lb_conn_t *conn)
{
    parsec_qpu_text_add(&out, "}\n");
    conn_send(conn, out.text, out.length);
}

static void reply_error(lb_conn_t *conn, int64_t seq, const char *error, const char *why)
{
    reply_begin(seq, 0);
    parsec_qpu_text_add(&out, ",\"error\":\"%s\",\"why\":", error);
    parsec_qpu_text_add_string(&out, why, strlen(why));
    reply_send(conn);
}

static void job_free(lb_job_t *job)
{
    for( int i = 0; i < job->nb; i++ ) free(job->counts[i]);
    free(job->counts);
    free(job);
}

static void job_remove(lb_job_t *job)
{
    for( lb_job_t **at = &jobs; NULL != *at; at = &(*at)->next ) {
        if( *at != job ) continue;
        *at = job->next;
        break;
    }
    job_free(job);
}

static void session_drop(uint64_t session)
{
    for( lb_job_t *job = jobs, *next; NULL != job; job = next ) {
        next = job->next;
        if( job->session == session ) job_remove(job);
    }
    for( int i = 0; i < nb_sessions; i++ ) {
        if( sessions[i] == session ) { sessions[i] = sessions[--nb_sessions]; break; }
    }
}

static void circuits_drop(void)
{
    while( NULL != circuits ) {
        lb_circuit_t *next = circuits->next;
        free(circuits->circuit);
        free(circuits);
        circuits = next;
    }
}

/* Everything this adapter knew is gone, as after it was started again. */
static void forget_all(void)
{
    while( NULL != jobs ) job_remove(jobs);
    circuits_drop();
    nb_sessions = 0;
    next_job = 1;                                   /* and it numbers its jobs from the first again */
    while( nb_conns > 0 ) conn_close(&conns[0]);
}

static lb_circuit_t *circuit_find(uint64_t id)
{
    for( lb_circuit_t *c = circuits; NULL != c; c = c->next ) {
        if( c->id == id ) return c;
    }
    return NULL;
}

static int member_u64(const parsec_qpu_json_t *object, const char *key, uint64_t *value)
{
    parsec_qpu_json_t member;
    return (PARSEC_SUCCESS == parsec_qpu_json_member(object, key, &member)) &&
           (PARSEC_SUCCESS == parsec_qpu_json_u64(&member, value));
}

static int member_int(const parsec_qpu_json_t *object, const char *key, int64_t *value)
{
    parsec_qpu_json_t member;
    return (PARSEC_SUCCESS == parsec_qpu_json_member(object, key, &member)) &&
           (PARSEC_SUCCESS == parsec_qpu_json_int(&member, value));
}

static void wait_answer(lb_conn_t *conn)
{
    int n = 0;

    reply_begin(conn->wait_seq, 1);
    parsec_qpu_text_add(&out, ",\"jobs\":[");
    for( lb_job_t *job = jobs; (NULL != job) && (n < conn->wait_max); job = job->next ) {
        if( (job->session != conn->session) || (LB_RUNNING == job->state) || job->reported ) continue;
        parsec_qpu_text_add(&out, "%s{\"job\":\"0x%llx\",\"state\":\"%s\"}", (n > 0) ? "," : "", (unsigned long long)job->id,
                            (LB_DONE == job->state) ? "done" : (LB_FAILED == job->state) ? "failed" : "cancelled");
        job->reported = 1;
        n++;
    }
    parsec_qpu_text_add(&out, "]");
    reply_send(conn);
    conn->waiting = 0;
}

static int has_news(const lb_conn_t *conn)
{
    for( const lb_job_t *job = jobs; NULL != job; job = job->next ) {
        if( (job->session == conn->session) && (LB_RUNNING != job->state) && !job->reported ) return 1;
    }
    return 0;
}

static void do_prepare(lb_conn_t *conn, int64_t seq, const parsec_qpu_json_t *request)
{
    parsec_qpu_json_t member;
    parsec_qpu_qasm_info_t info;
    lb_circuit_t *known;
    parsec_qpu_circuit_t *circuit;
    uint64_t id;
    int64_t length;
    char *text;

    counters.prepares++;
    if( !member_u64(request, "circuit", &id) || (PARSEC_SUCCESS != parsec_qpu_json_member(request, "qasm", &member)) ||
        ((length = parsec_qpu_json_string(&member, NULL, 0)) < 0) ) {
        violation("a prepare without a circuit name or without a text");
        reply_error(conn, seq, "bad_request", "prepare takes circuit and qasm");
        return;
    }
    text = (char*)malloc((size_t)length + 1);
    (void)parsec_qpu_json_string(&member, text, (size_t)length + 1);
    if( PARSEC_SUCCESS != parsec_qpu_qasm_read(text, (size_t)length, NULL, 0, &info) ) {
        char why[256];
        snprintf(why, sizeof(why), "line %d: %s", info.line, info.why);
        violation("a circuit that is not read: %s", why);
        reply_error(conn, seq, "rejected", why);
        free(text);
        return;
    }
    if( info.nb_qubits > (uint32_t)opt_qubits ) {
        reply_error(conn, seq, "rejected", "the circuit is wider than this adapter takes");
        free(text);
        return;
    }
    circuit = (parsec_qpu_circuit_t*)calloc(1, info.size);
    (void)parsec_qpu_qasm_read(text, (size_t)length, circuit, info.size, NULL);
    free(text);
    if( NULL != (known = circuit_find(id)) ) {
        /* Giving a circuit again is allowed; another circuit under the same name is not. */
        if( (parsec_qpu_circuit_size(known->circuit) != info.size) || (0 != memcmp(known->circuit, circuit, info.size)) ) {
            violation("two different circuits under the name 0x%llx", (unsigned long long)id);
        }
        free(circuit);
    } else {
        known = (lb_circuit_t*)calloc(1, sizeof(lb_circuit_t));
        known->id      = id;
        known->circuit = circuit;
        known->next    = circuits;
        circuits       = known;
    }
    reply_begin(seq, 1);
    reply_send(conn);
}

/* Returns 1 when the connection is gone. */
static int do_submit(lb_conn_t *conn, int64_t seq, const parsec_qpu_json_t *request)
{
    parsec_qpu_json_t entries, entry, params, value;
    lb_job_t *job;
    uint64_t tag;
    long nth;
    int nb = 0, fault = -1;

    nth = ++counters.submits;
    if( opt_no_job ) violation("a job was submitted to an adapter that may be given none");
    for( int f = 0; f < F_MAX; f++ ) {
        if( faults[f] == nth ) fault = f;
    }
    if( (F_FORGET == fault) || (F_TURN == fault) ) {
        if( F_TURN == fault ) opt_physical = 1;
        forget_all();
        return 1;
    }
    if( F_DROP_BEFORE == fault ) { conn_close(conn); return 1; }
    if( F_BUSY == fault )   { reply_error(conn, seq, "busy", "try later"); return 0; }
    if( F_REJECT == fault ) { reply_error(conn, seq, "rejected", "this job is refused"); return 0; }
    if( F_NOCIRCUIT == fault ) circuits_drop();

    if( !member_u64(request, "tag", &tag) || (PARSEC_SUCCESS != parsec_qpu_json_member(request, "entries", &entries)) ) {
        violation("a submit without a tag or without entries");
        reply_error(conn, seq, "bad_request", "submit takes tag and entries");
        return 0;
    }
    /* Asked again for a job that is here: it is the answer. */
    for( job = jobs; NULL != job; job = job->next ) {
        if( (job->session == conn->session) && (job->tag == tag) ) {
            counters.tag_hits++;
            goto answer;
        }
    }

    entry.at = NULL;
    while( parsec_qpu_json_next(&entries, &entry) ) nb++;
    if( (nb < 1) || (nb > opt_max_circuits) ) {
        violation("a job of %d circuit(s), when this adapter said it takes 1 to %d", nb, opt_max_circuits);
        reply_error(conn, seq, "rejected", "too many circuits in one job");
        return 0;
    }
    /* Every circuit has to be known before anything is taken. */
    entry.at = NULL;
    while( parsec_qpu_json_next(&entries, &entry) ) {
        uint64_t id;
        if( !member_u64(&entry, "circuit", &id) ) {
            violation("an entry without a circuit");
            reply_error(conn, seq, "bad_request", "an entry takes circuit, shots, key, attempt and params");
            return 0;
        }
        if( NULL == circuit_find(id) ) {
            reply_error(conn, seq, "unknown_circuit", "prepare it first");
            return 0;
        }
    }

    job = (lb_job_t*)calloc(1, sizeof(lb_job_t));
    job->id      = next_job++;
    job->tag     = tag;
    job->session = conn->session;
    job->nb      = nb;
    job->counts  = (parsec_qpu_result_t**)calloc((size_t)nb, sizeof(parsec_qpu_result_t*));
    job->t_start = parsec_qpu_now_ns();
    job->t_done  = job->t_start + opt_duration_ns;
    job->end_as  = (F_FAIL == fault) ? LB_FAILED : (F_CANCELLED == fault) ? LB_CANCELLED : LB_DONE;
    job->bad_counts = (F_BADCOUNTS == fault);
    entry.at = NULL;
    for( int i = 0; i < nb; i++ ) {
        const lb_circuit_t *known;
        uint64_t id = 0, key = 0;
        int64_t shots = 0, attempt = 0;
        double *numbers = NULL;
        uint32_t nb_params = 0;
        int bad = 0;

        (void)parsec_qpu_json_next(&entries, &entry);
        (void)member_u64(&entry, "circuit", &id);
        known = circuit_find(id);
        if( !member_u64(&entry, "key", &key) || !member_int(&entry, "shots", &shots) || !member_int(&entry, "attempt", &attempt) ||
            (PARSEC_SUCCESS != parsec_qpu_json_member(&entry, "params", &params)) || (shots < 1) || (attempt < 1) ||
            ((opt_max_shots > 0) && (shots > opt_max_shots)) ) {
            bad = 1;
        } else {
            value.at = NULL;
            while( parsec_qpu_json_next(&params, &value) ) nb_params++;
            if( nb_params != known->circuit->nb_params ) bad = 1;
        }
        if( !bad ) {
            numbers = (double*)calloc(nb_params + 1, sizeof(double));
            value.at = NULL;
            for( uint32_t p = 0; p < nb_params; p++ ) {
                (void)parsec_qpu_json_next(&params, &value);
                if( PARSEC_SUCCESS != parsec_qpu_json_double(&value, &numbers[p]) ) bad = 1;
            }
        }
        if( !bad ) {
            job->counts[i] = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size((uint32_t)shots));
            job->counts[i]->shots_requested = (uint32_t)shots;
            if( PARSEC_SUCCESS != parsec_qpu_sampler_run(sampler, known->circuit, numbers, (uint32_t)shots, opt_seed, key,
                                                         job->counts[i], (uint32_t)shots, NULL) ) bad = 1;
        }
        free(numbers);
        if( bad ) {
            violation("an entry that cannot be run (shots, key, attempt, or as many params as the circuit has inputs)");
            reply_error(conn, seq, "rejected", "an entry cannot be run");
            job_free(job);
            return 0;
        }
    }
    job->next = jobs;
    jobs = job;
    counters.jobs++;

answer:
    if( F_DROP_SUBMIT == fault ) { conn_close(conn); return 1; }
    if( F_SILENT == fault )      { conn->silent = 1; return 0; }
    if( F_GARBAGE == fault )     { conn_send(conn, "this is not an answer\n", 22); return 0; }
    if( F_STALE == fault )       seq--;
    reply_begin(seq, 1);
    parsec_qpu_text_add(&out, ",\"job\":\"0x%llx\"", (unsigned long long)job->id);
    reply_send(conn);
    return 0;
}

static void do_results(lb_conn_t *conn, int64_t seq, const parsec_qpu_json_t *request)
{
    const int64_t now = parsec_qpu_now_ns();
    lb_job_t *job;
    uint64_t id = 0;

    if( !member_u64(request, "job", &id) ) {
        violation("results without a job");
        reply_error(conn, seq, "bad_request", "results takes job");
        return;
    }
    for( job = jobs; NULL != job; job = job->next ) {
        if( (job->id == id) && (job->session == conn->session) ) break;
    }
    if( (NULL == job) || (LB_DONE != job->state) ) {
        counters.unknown_results++;
        reply_error(conn, seq, "unknown_job", "no job of that name has ended well here");
        return;
    }
    counters.results += job->nb;
    reply_begin(seq, 1);
    parsec_qpu_text_add(&out, ",\"entries\":[");
    for( int i = 0; i < job->nb; i++ ) {
        const parsec_qpu_result_t *counts = job->counts[i];
        parsec_qpu_text_add(&out, "%s{\"since_start_ns\":%lld,\"since_end_ns\":%lld,\"counts\":[", (i > 0) ? "," : "",
                            (long long)(now - job->t_start), (long long)(now - job->t_done));
        for( uint32_t e = 0, written = 0; e < counts->nb_entries; e++ ) {
            uint64_t bits;
            uint32_t count;
            parsec_qpu_result_get_entry(counts, e, &bits, &count);
            if( job->bad_counts && (0 == e) && (0 == i) ) count--;
            if( 0 == count ) continue;
            parsec_qpu_text_add(&out, "%s[\"0x%llx\",%u]", (written++ > 0) ? "," : "", (unsigned long long)bits, count);
        }
        parsec_qpu_text_add(&out, "]}");
    }
    parsec_qpu_text_add(&out, "]");
    reply_send(conn);
    job_remove(job);
}

/* One line from a client. Returns 1 when the connection is gone. */
static int handle(lb_conn_t *conn, const char *line, size_t length)
{
    parsec_qpu_json_t request, op;
    int64_t seq = -1;

    counters.lines++;
    if( NULL != opt_transcript ) {
        FILE *f = fopen(opt_transcript, "a");
        if( NULL != f ) { fprintf(f, "%.*s\n", (int)length, line); fclose(f); }
    }
    if( (NULL != opt_secret) && (NULL != memmem(line, length, opt_secret, strlen(opt_secret))) ) {
        violation("something that had to stay with the client crossed the socket");
    }
    if( conn->silent ) return 0;
    if( (PARSEC_SUCCESS != parsec_qpu_json_parse(line, length, &request)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_member(&request, "op", &op)) ) {
        violation("a line that is not a request: %.60s", line);
        return 0;
    }
    if( parsec_qpu_json_is(&op, "wake") ) {
        counters.wakes++;
        if( conn->waiting ) wait_answer(conn);
        return 0;
    }
    if( !member_int(&request, "seq", &seq) ) {
        violation("a request without a number");
        return 0;
    }
    if( conn->waiting ) {
        violation("a request while another one waits for its answer");
        conn->waiting = 0;
    }
    if( parsec_qpu_json_is(&op, "hello") ) {
        int64_t protocol = -1;
        int resumed = 0;
        counters.hellos++;
        if( !member_u64(&request, "session", &conn->session) || !member_int(&request, "protocol", &protocol) ) {
            violation("a hello without a session or a protocol");
            reply_error(conn, seq, "bad_request", "hello takes protocol and session");
            return 0;
        }
        if( PARSEC_QPU_ADAPTER_PROTOCOL != protocol ) {
            reply_error(conn, seq, "protocol", "this adapter speaks another version");
            return 0;
        }
        conn->has_session = 1;
        for( int i = 0; i < nb_sessions; i++ ) {
            if( sessions[i] == conn->session ) resumed = 1;
        }
        if( resumed ) {
            counters.resumes++;
            /* What it was told and may not have heard is told again. */
            for( lb_job_t *job = jobs; NULL != job; job = job->next ) {
                if( job->session == conn->session ) job->reported = 0;
            }
        } else if( nb_sessions < LB_MAX_SESSIONS ) {
            sessions[nb_sessions++] = conn->session;
        }
        reply_begin(seq, 1);
        parsec_qpu_text_add(&out, ",\"protocol\":%d,\"backend\":\"loopback\",\"qubits\":%d,\"max_shots\":%d,\"max_jobs\":0,"
                            "\"max_circuits\":%d,\"physical\":%s,\"resumed\":%s", PARSEC_QPU_ADAPTER_PROTOCOL, opt_qubits,
                            opt_max_shots, opt_max_circuits, opt_physical ? "true" : "false", resumed ? "true" : "false");
        reply_send(conn);
        return 0;
    }
    if( parsec_qpu_json_is(&op, "stats") ) {
        long live = 0, nb_circuits = 0;
        for( lb_job_t *job = jobs; NULL != job; job = job->next ) live++;
        for( lb_circuit_t *c = circuits; NULL != c; c = c->next ) nb_circuits++;
        reply_begin(seq, 1);
        parsec_qpu_text_add(&out, ",\"hellos\":%ld,\"resumes\":%ld,\"prepares\":%ld,\"submits\":%ld,\"jobs\":%ld,\"tag_hits\":%ld,"
                            "\"waits\":%ld,\"wakes\":%ld,\"results\":%ld,\"unknown_results\":%ld,\"cancels\":%ld,\"abandons\":%ld,"
                            "\"abandon_hits\":%ld,\"byes\":%ld,\"violations\":%ld,\"live_jobs\":%ld,\"circuits\":%ld,\"sessions\":%d",
                            counters.hellos, counters.resumes, counters.prepares, counters.submits, counters.jobs, counters.tag_hits,
                            counters.waits, counters.wakes, counters.results, counters.unknown_results, counters.cancels,
                            counters.abandons, counters.abandon_hits, counters.byes, counters.violations, live, nb_circuits, nb_sessions);
        reply_send(conn);
        return 0;
    }
    if( !conn->has_session ) {
        violation("a request before hello");
        reply_error(conn, seq, "bad_request", "say hello first");
        return 0;
    }
    if( parsec_qpu_json_is(&op, "prepare") ) { do_prepare(conn, seq, &request); return 0; }
    if( parsec_qpu_json_is(&op, "submit") )  return do_submit(conn, seq, &request);
    if( parsec_qpu_json_is(&op, "results") ) { do_results(conn, seq, &request); return 0; }
    if( parsec_qpu_json_is(&op, "wait") ) {
        int64_t timeout_ms = 0, max = 1;
        counters.waits++;
        if( !member_int(&request, "timeout_ms", &timeout_ms) || !member_int(&request, "max", &max) || (timeout_ms < 0) || (max < 1) ) {
            violation("a wait without a timeout or a number of jobs");
            reply_error(conn, seq, "bad_request", "wait takes timeout_ms and max");
            return 0;
        }
        conn->waiting       = 1;
        conn->wait_seq      = seq;
        conn->wait_max      = (int)max;
        conn->wait_deadline = parsec_qpu_now_ns() + timeout_ms * 1000000LL;
        return 0;                                   /* answered from the loop */
    }
    if( parsec_qpu_json_is(&op, "cancel") ) {
        lb_job_t *job;
        uint64_t id = 0;
        counters.cancels++;
        (void)member_u64(&request, "job", &id);
        for( job = jobs; NULL != job; job = job->next ) {
            if( (job->id == id) && (job->session == conn->session) ) break;
        }
        if( NULL == job ) {
            reply_error(conn, seq, "unknown_job", "there is no job of that name here");
        } else {
            job_remove(job);
            reply_begin(seq, 1);
            reply_send(conn);
        }
        return 0;
    }
    if( parsec_qpu_json_is(&op, "abandon") ) {
        lb_job_t *job;
        uint64_t tag = 0;
        counters.abandons++;
        (void)member_u64(&request, "tag", &tag);
        for( job = jobs; NULL != job; job = job->next ) {
            if( (job->tag == tag) && (job->session == conn->session) ) break;
        }
        if( NULL != job ) counters.abandon_hits++;
        reply_begin(seq, 1);
        if( NULL != job ) {
            parsec_qpu_text_add(&out, ",\"found\":true,\"job\":\"0x%llx\"", (unsigned long long)job->id);
        } else {
            parsec_qpu_text_add(&out, ",\"found\":false");
        }
        reply_send(conn);
        if( NULL != job ) job_remove(job);
        return 0;
    }
    if( parsec_qpu_json_is(&op, "bye") ) {
        counters.byes++;
        session_drop(conn->session);
        reply_begin(seq, 1);
        reply_send(conn);
        conn_close(conn);
        return 1;
    }
    violation("a request this adapter does not know: %.40s", line);
    reply_error(conn, seq, "bad_request", "no such request");
    return 0;
}

static void parse_faults(const char *spec)
{
    char *copy = strdup(spec), *save = NULL;
    for( char *token = strtok_r(copy, ",", &save); NULL != token; token = strtok_r(NULL, ",", &save) ) {
        char *colon = strchr(token, ':');
        int found = 0;
        if( NULL == colon ) { fprintf(stderr, "qpu_adapter_loopback: -F takes name:N, not '%s'\n", token); exit(2); }
        *colon = '\0';
        for( int f = 0; f < F_MAX; f++ ) {
            if( 0 == strcmp(token, fault_names[f]) ) { faults[f] = atol(colon + 1); found = 1; }
        }
        if( !found ) { fprintf(stderr, "qpu_adapter_loopback: no fault is called '%s'\n", token); exit(2); }
    }
    free(copy);
}

int main(int argc, char *argv[])
{
    const char *path = NULL;
    struct sockaddr_un sun;
    struct sigaction sa;
    int listener, opt;

    while( -1 != (opt = getopt(argc, argv, "s:S:d:q:c:m:PNb:T:E:F:")) ) {
        switch( opt ) {
        case 's': path = optarg; break;
        case 'S': opt_seed = strtoull(optarg, NULL, 10); break;
        case 'd': opt_duration_ns = atoll(optarg) * 1000LL; break;
        case 'q': opt_qubits = atoi(optarg); break;
        case 'c': opt_max_circuits = atoi(optarg); break;
        case 'm': opt_max_shots = atoi(optarg); break;
        case 'P': opt_physical = 1; break;
        case 'N': opt_no_job = 1; break;
        case 'b': opt_backlog = atoi(optarg); break;
        case 'T': opt_transcript = optarg; break;
        case 'E': opt_secret = optarg; break;
        case 'F': parse_faults(optarg); break;
        default:
            fprintf(stderr, "usage: %s -s <socket> [-S seed] [-d us] [-q qubits] [-c circuits] [-m shots] [-P] [-T file] [-E text] [-F faults]\n", argv[0]);
            return 2;
        }
    }
    if( (NULL == path) || (strlen(path) >= sizeof(sun.sun_path)) ) {
        fprintf(stderr, "qpu_adapter_loopback: -s takes the path of a socket, shorter than %zu bytes\n", sizeof(sun.sun_path));
        return 2;
    }
    if( opt_qubits > PARSEC_QPU_SAMPLER_MAX_QUBITS ) opt_qubits = PARSEC_QPU_SAMPLER_MAX_QUBITS;

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_term;                        /* no SA_RESTART: poll() has to return */
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    memset(&sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
    strcpy(sun.sun_path, path);
    unlink(path);
    listener = socket(AF_UNIX, SOCK_STREAM, 0);
    if( (listener < 0) || (0 != bind(listener, (struct sockaddr*)&sun, sizeof(sun))) || (0 != listen(listener, opt_backlog)) ) {
        fprintf(stderr, "qpu_adapter_loopback: cannot listen on %s: %s\n", path, strerror(errno));
        return 2;
    }
    sampler = parsec_qpu_sampler_new();

    while( !stop ) {
        struct pollfd pfd[LB_MAX_CONNS + 1];
        int64_t now = parsec_qpu_now_ns(), next = now + 1000000000LL;
        int timeout_ms, rc;

        /* Jobs end, and those who wait for them hear it. */
        for( lb_job_t *job = jobs; NULL != job; job = job->next ) {
            if( LB_RUNNING != job->state ) continue;
            if( job->t_done <= now ) job->state = job->end_as;
            else if( job->t_done < next ) next = job->t_done;
        }
        for( int i = 0; i < nb_conns; i++ ) {
            if( !conns[i].waiting ) continue;
            if( has_news(&conns[i]) || (conns[i].wait_deadline <= now) ) wait_answer(&conns[i]);
            else if( conns[i].wait_deadline < next ) next = conns[i].wait_deadline;
        }

        pfd[0].fd = listener; pfd[0].events = POLLIN; pfd[0].revents = 0;
        for( int i = 0; i < nb_conns; i++ ) {
            pfd[i + 1].fd = conns[i].fd; pfd[i + 1].events = POLLIN; pfd[i + 1].revents = 0;
        }
        timeout_ms = (int)((next - now + 999999) / 1000000);
        rc = poll(pfd, (nfds_t)nb_conns + 1, (timeout_ms < 0) ? 0 : timeout_ms);
        if( rc <= 0 ) continue;

        /* From the last to the first: closing one moves the last into its place. */
        for( int i = nb_conns - 1; i >= 0; i-- ) {
            lb_conn_t *conn = &conns[i];
            ssize_t n;
            int gone = 0;

            if( !pfd[i + 1].revents || (i >= nb_conns) ) continue;
            if( conn->len + 1 >= conn->room ) {
                conn->room = (0 == conn->room) ? 4096 : 2 * conn->room;
                conn->in   = (char*)realloc(conn->in, conn->room);
            }
            n = recv(conn->fd, conn->in + conn->len, conn->room - conn->len - 1, 0);
            if( n <= 0 ) {
                if( (n < 0) && (EINTR == errno) ) continue;
                conn_close(conn);                   /* its session stays: it may come back */
                continue;
            }
            conn->len += (size_t)n;
            for(;;) {
                char *newline = (char*)memchr(conn->in, '\n', conn->len);
                size_t used;
                if( NULL == newline ) break;
                *newline = '\0';
                used = (size_t)(newline + 1 - conn->in);
                gone = handle(conn, conn->in, used - 1);
                if( gone ) break;                   /* conn is no longer this connection */
                memmove(conn->in, conn->in + used, conn->len - used);
                conn->len -= used;
            }
            if( gone && (0 == nb_conns) ) break;
        }
        if( pfd[0].revents && (nb_conns < LB_MAX_CONNS) ) {
            const int fd = accept(listener, NULL, NULL);
            if( fd >= 0 ) {
                memset(&conns[nb_conns], 0, sizeof(lb_conn_t));
                conns[nb_conns++].fd = fd;
            }
        }
    }

    close(listener);
    unlink(path);
    parsec_qpu_sampler_free(sampler);
    if( counters.violations > 0 ) {
        fprintf(stderr, "qpu_adapter_loopback: %ld violation(s) of the protocol by the client\n", counters.violations);
        return 3;
    }
    return 0;
}
