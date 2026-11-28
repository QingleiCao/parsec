/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"
#include "parsec/mca/device/qpu/device_qpu_qasm.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

/**
 * The adapter backend: the client of an adapter, a separate program that
 * runs the jobs (a simulator of Qiskit, later a provider's SDK). PaRSEC
 * links to nothing of the adapter and does not start it. The two talk over
 * a Unix-domain socket, one line of JSON per message, one answer per request
 * (tools/qpu/README.md has the protocol).
 *
 * What the client keeps to itself. Nothing but circuits, parameter values,
 * shot counts, keys and tags crosses the socket: this file reads no
 * credential and no variable of the environment, and logs nothing an
 * adapter sent. It talks only to an adapter that runs as the same user.
 *
 * When the adapter is out of reach. A request that gets no answer closes
 * the connection, and the next call connects again, not more often than
 * every 20 ms to 1 s. The adapter keeps the jobs of a client by the session
 * the client names when it connects, so a client that comes back finds its
 * jobs; a job the adapter took and whose identifier never arrived is found
 * again by its tag. When the adapter that answers is not the one that had
 * the jobs (it was started again), they are reported as failed, as are the
 * jobs of an adapter that stays out of reach for device_qpu_unavailable_timeout;
 * those it is told to drop when it comes back, so that none runs on for
 * nobody.
 *
 * The name of a job. An adapter numbers its jobs as it likes, and one that
 * was started again starts again from its first number. The caller is given
 * a number of the client's own, which no two jobs of this backend share; the
 * adapter's is kept beside it, and is not looked at for a job whose adapter
 * is gone.
 *
 * Hardware. Whether the adapter drives a quantum device is what it says
 * when the client connects, and the guard of the service closes a backend
 * that says so unless device_qpu_allow_hardware is set. An adapter that
 * says something else when the client connects again is not used again.
 *
 * Threads. As for every backend, the progress thread makes all the calls
 * and wakeup() comes from anywhere: it writes one byte to a pipe that
 * wait() watches beside the socket.
 */

#define ADAPTER_MAX_OPEN      PARSEC_QPU_MAX_QPUS
#define ADAPTER_MAX_LINE      ((size_t)256 << 20)
#define ADAPTER_WAIT_MAX_MS   60000
#define ADAPTER_RETRY_MIN_NS  20000000LL
#define ADAPTER_RETRY_MAX_NS  1000000000LL

enum { ADAPTER_OK = 0, ADAPTER_REFUSED = 1, ADAPTER_DOWN = -1 };
enum { ADAPTER_FLYING = 0, ADAPTER_REPORTED = 1 };
#define ADAPTER_NO_EVENT  (-1)

typedef struct adapter_entry_s {
    uint32_t shots, max_entries, nb_qubits;
} adapter_entry_t;

typedef struct adapter_job_s {
    struct adapter_job_s *next;
    parsec_qpu_job_id_t   id;           /* the caller's name for the job: the client's own, never used twice */
    uint64_t              remote;       /* the adapter's name for it */
    int                   state;        /* ADAPTER_FLYING, or ADAPTER_REPORTED: done, results not taken yet */
    int                   pending;      /* a final state the caller has not been told yet, or ADAPTER_NO_EVENT */
    int                   lost;         /* the adapter that had the job is gone: its name there means nothing now */
    int                   nb;
    adapter_entry_t       entries[];
} adapter_job_t;

/* Something the adapter has to hear when it can be reached: a job to
 * forget, or a tag to give up on. */
typedef struct adapter_settle_s {
    int      abandon;
    uint64_t value;
} adapter_settle_t;

typedef struct adapter_circuit_s {
    uint64_t id, check;
} adapter_circuit_t;

/* The end of a job that was reported before submit() returned its identifier. */
typedef struct adapter_early_s {
    uint64_t remote;
    int      state;
} adapter_early_t;

typedef struct adapter_backend_s {
    parsec_qpu_backend_t super;
    int                  number;
    char                *address;
    int                  fd;            /* -1 when not connected */
    int                  wake[2];
    uint64_t             session;
    uint64_t             seq;
    int64_t              timeout_ns;    /* for the answer to one request */
    int64_t              unavailable_ns;
    int64_t              t_retry;       /* not before: the next attempt to connect */
    int64_t              retry_ns;
    int64_t              t_down;        /* since when the adapter is out of reach; 0 when it is not */
    int                  greeted;       /* it answered a hello once */
    int                  dead;          /* not to be used again */
    int                  warned;        /* this outage was reported */
    parsec_qpu_caps_t    caps;
    char                 name[64];      /* what the adapter says it runs the jobs on */

    char                *in;            /* what was received and not consumed: [in_start, in_len) */
    size_t               in_start, in_scan, in_len, in_room;
    parsec_qpu_text_t    out;

    adapter_circuit_t   *prepared;
    uint32_t             nb_prepared, prepared_room;
    adapter_job_t       *jobs;
    parsec_qpu_job_id_t  next_id;
    adapter_settle_t    *settle;
    uint32_t             nb_settle, settle_room;
    adapter_early_t     *early;
    uint32_t             nb_early, early_room;

    parsec_qpu_adapter_stats_t stats;
} adapter_backend_t;

#define ADAPTER_COUNT(BE, FIELD)  PARSEC_QPU_STORE(&(BE)->stats.FIELD, (BE)->stats.FIELD + 1)

static pthread_mutex_t    adapter_opened_lock = PTHREAD_MUTEX_INITIALIZER;
static adapter_backend_t *adapter_opened[ADAPTER_MAX_OPEN];
static int                adapter_opened_n = 0;
static uint64_t           adapter_opens = 0;

static void adapter_stats_add(const adapter_backend_t *be, parsec_qpu_adapter_stats_t *stats)
{
    stats->connects    += PARSEC_QPU_LOAD(&be->stats.connects);
    stats->requests    += PARSEC_QPU_LOAD(&be->stats.requests);
    stats->unanswered  += PARSEC_QPU_LOAD(&be->stats.unanswered);
    stats->prepares    += PARSEC_QPU_LOAD(&be->stats.prepares);
    stats->submits     += PARSEC_QPU_LOAD(&be->stats.submits);
    stats->circuits    += PARSEC_QPU_LOAD(&be->stats.circuits);
    stats->lost_jobs   += PARSEC_QPU_LOAD(&be->stats.lost_jobs);
    stats->results     += PARSEC_QPU_LOAD(&be->stats.results);
    stats->bad_results += PARSEC_QPU_LOAD(&be->stats.bad_results);
    stats->unsettled   += PARSEC_QPU_LOAD(&be->stats.unsettled);
}

void parsec_qpu_adapter_stats(parsec_qpu_adapter_stats_t *stats)
{
    memset(stats, 0, sizeof(*stats));
    pthread_mutex_lock(&adapter_opened_lock);
    for( int i = 0; i < adapter_opened_n; i++ ) adapter_stats_add(adapter_opened[i], stats);
    stats->opens = adapter_opens;
    pthread_mutex_unlock(&adapter_opened_lock);
}

int parsec_qpu_adapter_stats_of(int qpu, parsec_qpu_adapter_stats_t *stats)
{
    int rc = PARSEC_ERR_NOT_FOUND;

    memset(stats, 0, sizeof(*stats));
    pthread_mutex_lock(&adapter_opened_lock);
    for( int i = 0; i < adapter_opened_n; i++ ) {
        if( adapter_opened[i]->number != qpu ) continue;
        adapter_stats_add(adapter_opened[i], stats);
        rc = PARSEC_SUCCESS;
    }
    stats->opens = adapter_opens;
    pthread_mutex_unlock(&adapter_opened_lock);
    return rc;
}

uint64_t parsec_qpu_adapter_circuit_id(const parsec_qpu_circuit_t *circuit, uint64_t *check)
{
    const unsigned char *bytes = (const unsigned char*)circuit;
    const size_t size = parsec_qpu_circuit_size(circuit);
    uint64_t id = 0xcbf29ce484222325ULL, second = 0x6a09e667f3bcc908ULL;

    /* Two digests that share nothing: FNV-1a over the bytes, and a chain of
     * splitmix64 over the same bytes with their position. */
    for( size_t i = 0; i < size; i++ ) {
        id = (id ^ bytes[i]) * 0x100000001b3ULL;
        second = parsec_qpu_mix64(second ^ (((uint64_t)bytes[i]) << 8) ^ (uint64_t)i);
    }
    if( NULL != check ) *check = second;
    return parsec_qpu_mix64(id ^ (uint64_t)size);
}

/* ------------------------------------------------------------------------
 * The connection
 * --------------------------------------------------------------------- */

static int adapter_drain(adapter_backend_t *be)
{
    char bytes[64];
    int woken = 0;
    while( read(be->wake[0], bytes, sizeof(bytes)) > 0 ) woken = 1;
    return woken;
}

/* Give up on the jobs the adapter has: the caller will hear that they failed.
 *
 * replaced: the adapter that answers is another one than the one that had
 * the jobs. Nothing of what the first one was told means anything to it:
 * not the names of the jobs, of which it may use the same ones for others,
 * not the ends it has not told, not a result that was not taken yet.
 *
 * Not replaced: the adapter is out of reach for too long. It may still have
 * the jobs, and is told to drop them when it can be reached. */
static void adapter_lose_jobs(adapter_backend_t *be, int replaced)
{
    for( adapter_job_t *job = be->jobs; NULL != job; job = job->next ) {
        if( job->lost ) continue;
        if( ADAPTER_FLYING == job->state ) {
            if( !replaced && (ADAPTER_NO_EVENT != job->pending) ) continue;     /* its end is known */
            if( PARSEC_QPU_JOB_FAILED != job->pending ) ADAPTER_COUNT(be, lost_jobs);
            job->pending = PARSEC_QPU_JOB_FAILED;
        }
        if( replaced ) job->lost = 1;       /* also a job whose result was not taken: it is not there any more */
    }
    if( replaced ) be->nb_early = 0;
}

static void adapter_drop(adapter_backend_t *be, const char *why)
{
    if( be->fd >= 0 ) {
        close(be->fd);
        be->fd = -1;
    }
    be->in_start = be->in_scan = be->in_len = 0;
    if( 0 == be->t_down ) be->t_down = parsec_qpu_now_ns();
    be->t_retry  = parsec_qpu_now_ns() + be->retry_ns;
    be->retry_ns = (2 * be->retry_ns < ADAPTER_RETRY_MAX_NS) ? 2 * be->retry_ns : ADAPTER_RETRY_MAX_NS;
    if( !be->warned ) {
        be->warned = 1;
        parsec_warning("QPU %d: the adapter at %s is out of reach (%s); it is tried again from time to time",
                       be->number, be->address, why);
    }
}

static int adapter_write(adapter_backend_t *be, const char *data, size_t length)
{
    const int64_t deadline = parsec_qpu_now_ns() + be->timeout_ns;

    while( length > 0 ) {
        const ssize_t n = send(be->fd, data, length, MSG_NOSIGNAL | MSG_DONTWAIT);
        if( n > 0 ) {
            data   += n;
            length -= (size_t)n;
            continue;
        }
        if( (n < 0) && (EINTR == errno) ) continue;
        if( (n < 0) && ((EAGAIN == errno) || (EWOULDBLOCK == errno)) ) {
            struct pollfd pfd = { .fd = be->fd, .events = POLLOUT };
            const int64_t left = deadline - parsec_qpu_now_ns();
            if( (left <= 0) || (poll(&pfd, 1, (int)(left / 1000000) + 1) < 0 && (EINTR != errno)) ) return -1;
            continue;
        }
        return -1;
    }
    return 0;
}

/* The next line the adapter sent: 1 with the line, which stays valid until
 * the next call; 0 when `deadline` passed; 2 when `watch` is set and
 * wakeup() was called; -1 when the connection is gone. */
static int adapter_line(adapter_backend_t *be, int64_t deadline, int watch, char **line, size_t *length)
{
    for(;;) {
        struct pollfd pfd[2];
        char *newline;
        int64_t left;
        ssize_t n;
        int rc;

        newline = (be->in_len > be->in_scan) ? (char*)memchr(be->in + be->in_scan, '\n', be->in_len - be->in_scan) : NULL;
        if( NULL != newline ) {
            *line    = be->in + be->in_start;
            *length  = (size_t)(newline - *line);
            *newline = '\0';
            be->in_start = be->in_scan = (size_t)(newline + 1 - be->in);
            return 1;
        }
        be->in_scan = be->in_len;
        if( be->in_start > 0 ) {
            memmove(be->in, be->in + be->in_start, be->in_len - be->in_start);
            be->in_len  -= be->in_start;
            be->in_scan -= be->in_start;
            be->in_start = 0;
        }
        if( be->in_len == be->in_room ) {
            const size_t room = (0 == be->in_room) ? 4096 : 2 * be->in_room;
            char *grown = (room <= ADAPTER_MAX_LINE) ? (char*)realloc(be->in, room) : NULL;
            if( NULL == grown ) return -1;
            be->in      = grown;
            be->in_room = room;
        }

        left = deadline - parsec_qpu_now_ns();
        if( left <= 0 ) return 0;
        pfd[0].fd = be->fd;      pfd[0].events = POLLIN; pfd[0].revents = 0;
        pfd[1].fd = be->wake[0]; pfd[1].events = POLLIN; pfd[1].revents = 0;
        rc = poll(pfd, watch ? 2 : 1, (left / 1000000 >= INT32_MAX) ? INT32_MAX : (int)(left / 1000000) + 1);
        if( rc < 0 ) {
            if( EINTR == errno ) continue;
            return -1;
        }
        if( 0 == rc ) continue;                         /* the deadline is looked at above */
        if( pfd[0].revents ) {
            n = recv(be->fd, be->in + be->in_len, be->in_room - be->in_len, MSG_DONTWAIT);
            if( n > 0 ) { be->in_len += (size_t)n; continue; }
            if( (n < 0) && ((EAGAIN == errno) || (EWOULDBLOCK == errno) || (EINTR == errno)) ) continue;
            return -1;                                  /* closed, or broken */
        }
        if( watch && pfd[1].revents ) return 2;
    }
}

/* Start a request: {"op":"<op>","seq":<n> and whatever the caller adds. */
static void adapter_begin(adapter_backend_t *be, const char *op)
{
    parsec_qpu_text_reset(&be->out);
    be->seq++;
    parsec_qpu_text_add(&be->out, "{\"op\":\"%s\",\"seq\":%llu", op, (unsigned long long)be->seq);
}

static int adapter_send(adapter_backend_t *be)
{
    parsec_qpu_text_add(&be->out, "}\n");
    ADAPTER_COUNT(be, requests);
    if( be->out.failed || (0 != adapter_write(be, be->out.text, be->out.length)) ) {
        ADAPTER_COUNT(be, unanswered);
        adapter_drop(be, "a request could not be sent");
        return ADAPTER_DOWN;
    }
    return ADAPTER_OK;
}

/* Take a line for the answer to the request that was sent last. */
static int adapter_answer(adapter_backend_t *be, const char *line, size_t length, parsec_qpu_json_t *reply)
{
    parsec_qpu_json_t member;
    int64_t seq = -1;
    int ok = 0;

    if( (PARSEC_SUCCESS != parsec_qpu_json_parse(line, length, reply)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_member(reply, "seq", &member)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_int(&member, &seq)) || ((uint64_t)seq != be->seq) ||
        (PARSEC_SUCCESS != parsec_qpu_json_member(reply, "ok", &member)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_bool(&member, &ok)) ) {
        ADAPTER_COUNT(be, unanswered);
        adapter_drop(be, "its answer is not one of the protocol");
        return ADAPTER_DOWN;
    }
    return ok ? ADAPTER_OK : ADAPTER_REFUSED;
}

/* Send the request that was begun and read its answer. The answer is valid
 * until the next request. */
static int adapter_call(adapter_backend_t *be, parsec_qpu_json_t *reply)
{
    char *line = NULL;
    size_t length = 0;

    if( ADAPTER_OK != adapter_send(be) ) return ADAPTER_DOWN;
    if( 1 != adapter_line(be, parsec_qpu_now_ns() + be->timeout_ns, 0, &line, &length) ) {
        ADAPTER_COUNT(be, unanswered);
        adapter_drop(be, "it did not answer");
        return ADAPTER_DOWN;
    }
    return adapter_answer(be, line, length, reply);
}

/* Whether the answer that refuses a request gives this reason. */
static int adapter_error_is(const parsec_qpu_json_t *reply, const char *error)
{
    parsec_qpu_json_t member;
    return (PARSEC_SUCCESS == parsec_qpu_json_member(reply, "error", &member)) && parsec_qpu_json_is(&member, error);
}

static void adapter_settle_add(adapter_backend_t *be, int abandon, uint64_t value)
{
    if( be->nb_settle == be->settle_room ) {
        const uint32_t room = (0 == be->settle_room) ? 16 : 2 * be->settle_room;
        adapter_settle_t *grown = (adapter_settle_t*)realloc(be->settle, room * sizeof(adapter_settle_t));
        if( NULL == grown ) return;                 /* the adapter drops the job with the session, at the latest */
        be->settle      = grown;
        be->settle_room = room;
    }
    be->settle[be->nb_settle].abandon = abandon;
    be->settle[be->nb_settle].value   = value;
    PARSEC_QPU_STORE(&be->stats.unsettled, (uint64_t)++be->nb_settle);
}

/* Tell the adapter what it has not heard yet, as far as it can be reached. */
static void adapter_settle_flush(adapter_backend_t *be)
{
    uint32_t done = 0;

    while( (done < be->nb_settle) && (be->fd >= 0) ) {
        parsec_qpu_json_t reply;
        adapter_begin(be, be->settle[done].abandon ? "abandon" : "cancel");
        parsec_qpu_text_add(&be->out, ",\"%s\":\"0x%llx\"", be->settle[done].abandon ? "tag" : "job",
                            (unsigned long long)be->settle[done].value);
        if( ADAPTER_DOWN == adapter_call(be, &reply) ) break;
        if( be->settle[done].abandon ) {
            /* The job that went with the tag may have ended, and been told, before: that end is nobody's now. */
            parsec_qpu_json_t member;
            uint64_t remote;
            if( (PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "job", &member)) &&
                (PARSEC_SUCCESS == parsec_qpu_json_u64(&member, &remote)) ) {
                for( uint32_t e = 0; e < be->nb_early; e++ ) {
                    if( be->early[e].remote != remote ) continue;
                    be->early[e] = be->early[--be->nb_early];
                    break;
                }
            }
        }
        done++;                                     /* heard, also when the adapter knows no such job */
    }
    if( done > 0 ) {
        memmove(be->settle, be->settle + done, (be->nb_settle - done) * sizeof(adapter_settle_t));
        be->nb_settle -= done;
        PARSEC_QPU_STORE(&be->stats.unsettled, (uint64_t)be->nb_settle);
    }
}

static int64_t adapter_member_int(const parsec_qpu_json_t *object, const char *key, int64_t value)
{
    parsec_qpu_json_t member;
    int64_t given;
    if( (PARSEC_SUCCESS == parsec_qpu_json_member(object, key, &member)) &&
        (PARSEC_SUCCESS == parsec_qpu_json_int(&member, &given)) ) return given;
    return value;
}

static int adapter_member_bool(const parsec_qpu_json_t *object, const char *key, int value)
{
    parsec_qpu_json_t member;
    int given;
    if( (PARSEC_SUCCESS == parsec_qpu_json_member(object, key, &member)) &&
        (PARSEC_SUCCESS == parsec_qpu_json_bool(&member, &given)) ) return given;
    return value;
}

/* Say who this is, and hear what the adapter is. */
static int adapter_hello(adapter_backend_t *be)
{
    parsec_qpu_json_t reply, member;
    int physical, resumed, rc;

    adapter_begin(be, "hello");
    parsec_qpu_text_add(&be->out, ",\"protocol\":%d,\"session\":\"0x%llx\"", PARSEC_QPU_ADAPTER_PROTOCOL,
                        (unsigned long long)be->session);
    if( ADAPTER_OK != (rc = adapter_call(be, &reply)) ) {
        if( ADAPTER_REFUSED == rc ) adapter_drop(be, "it refuses this client");
        return -1;
    }
    if( PARSEC_QPU_ADAPTER_PROTOCOL != adapter_member_int(&reply, "protocol", -1) ) {
        be->dead = 1;
        adapter_drop(be, "it speaks another version of the protocol");
        return -1;
    }
    physical = adapter_member_bool(&reply, "physical", 1);      /* an adapter that does not say is taken for one that drives a device */
    resumed  = adapter_member_bool(&reply, "resumed", 0);
    if( !be->greeted ) {
        be->caps.nb_qubits            = (uint32_t)adapter_member_int(&reply, "qubits", 0);
        be->caps.max_shots            = (uint32_t)adapter_member_int(&reply, "max_shots", 0);
        be->caps.max_jobs             = (uint32_t)adapter_member_int(&reply, "max_jobs", 0);
        be->caps.max_circuits_per_job = (uint32_t)adapter_member_int(&reply, "max_circuits", 1);
        be->caps.is_physical          = (uint32_t)physical;
        if( PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "backend", &member) ) {
            (void)parsec_qpu_json_string(&member, be->name, sizeof(be->name));
            /* It is printed: letters, digits and a little punctuation, whatever was sent. */
            for( char *c = be->name; '\0' != *c; c++ ) {
                if( !(((*c >= 'a') && (*c <= 'z')) || ((*c >= 'A') && (*c <= 'Z')) || ((*c >= '0') && (*c <= '9')) ||
                      (NULL != strchr(" ,.:_-", *c))) ) *c = '?';
            }
        }
        be->greeted = 1;
        return 0;
    }
    if( (uint32_t)physical != be->caps.is_physical ) {
        /* The guard looked at what the first adapter said. Another one, of
         * another kind, is not what the run was allowed to use. */
        be->dead = 1;
        adapter_lose_jobs(be, 1);
        adapter_drop(be, "the adapter that answers now is not of the kind that answered first; it is not used");
        return -1;
    }
    if( !resumed ) {
        /* An adapter that does not know this session: it was started again,
         * and what the other one had is gone. */
        adapter_lose_jobs(be, 1);
        be->nb_prepared = 0;
        /* Nothing is left there to cancel either. */
        be->nb_settle = 0;
        PARSEC_QPU_STORE(&be->stats.unsettled, 0);
    }
    return 0;
}

static int adapter_connect(adapter_backend_t *be)
{
    struct sockaddr_un sun;

    if( be->fd >= 0 ) return 0;
    if( be->dead || (parsec_qpu_now_ns() < be->t_retry) ) return -1;

    memset(&sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
    if( strlen(be->address) >= sizeof(sun.sun_path) ) {
        be->dead = 1;
        adapter_drop(be, "the path of its socket is too long for a Unix-domain socket");
        return -1;
    }
    strcpy(sun.sun_path, be->address);
    /* Not blocking: an adapter that accepts nothing, with its queue full,
     * would hold a blocking connect() for as long as it likes, where no
     * wake-up reaches. Here that is an error at once (EAGAIN). */
    be->fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if( (be->fd < 0) || (0 != connect(be->fd, (struct sockaddr*)&sun, sizeof(sun))) ) {
        adapter_drop(be, (be->fd < 0) ? "no socket could be made" :
                         ((EAGAIN == errno) || (EINPROGRESS == errno)) ? "it takes no connection" : "nothing listens there");
        return -1;
    }
#if defined(SO_PEERCRED)
    {
        /* The path may be in a directory others can write to: whoever
         * listens there has to be a process of the same user. */
        struct ucred peer;
        socklen_t length = sizeof(peer);
        if( (0 != getsockopt(be->fd, SOL_SOCKET, SO_PEERCRED, &peer, &length)) || (peer.uid != geteuid()) ) {
            be->dead = 1;
            adapter_drop(be, "what listens there is not a process of this user; it is not used");
            return -1;
        }
    }
#endif
    ADAPTER_COUNT(be, connects);
    if( 0 != adapter_hello(be) ) return -1;
    if( be->warned ) {
        parsec_debug_verbose(3, parsec_qpu_output_stream, "QPU %d: the adapter at %s answers again", be->number, be->address);
    }
    be->warned   = 0;
    be->t_down   = 0;
    be->retry_ns = ADAPTER_RETRY_MIN_NS;
    adapter_settle_flush(be);
    return (be->fd >= 0) ? 0 : -1;
}

/* ------------------------------------------------------------------------
 * The backend
 * --------------------------------------------------------------------- */

static int adapter_close(parsec_qpu_backend_t *backend);

static int adapter_open(parsec_qpu_backend_t **backend, int device_index)
{
    adapter_backend_t *be = (adapter_backend_t*)calloc(1, sizeof(adapter_backend_t));
    const char *address = parsec_qpu_list_option_string(device_index, "address", parsec_qpu_params.adapter_address);
    const int timeout_ms = parsec_qpu_list_option_int(device_index, "timeout", parsec_qpu_params.adapter_timeout_ms);
    const int unavailable_ms = parsec_qpu_list_option_int(device_index, "unavailable_timeout", parsec_qpu_params.unavailable_timeout_ms);
    char host[64] = "";

    *backend = NULL;
    if( NULL == be ) return PARSEC_ERR_OUT_OF_RESOURCE;
    be->super.ops = &parsec_qpu_backend_adapter_ops;
    be->number    = device_index;
    be->fd        = -1;
    be->wake[0]   = be->wake[1] = -1;
    be->timeout_ns     = (int64_t)((timeout_ms > 0) ? timeout_ms : 60000) * 1000000LL;
    be->unavailable_ns = (int64_t)((unavailable_ms > 0) ? unavailable_ms : 60000) * 1000000LL;
    be->retry_ns       = ADAPTER_RETRY_MIN_NS;

    if( (NULL == address) || ('\0' == address[0]) ) {
        parsec_warning("QPU %d: the adapter backend needs the path of the socket its adapter listens on: "
                       "device_qpu_adapter_address, or address= in device_qpu_devices", device_index);
        free(be);
        return PARSEC_ERR_BAD_PARAM;
    }
    be->next_id = 1;
    be->address = strdup(address);
    if( (NULL == be->address) || (0 != pipe2(be->wake, O_NONBLOCK | O_CLOEXEC)) ) {
        free(be->address);
        free(be);
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    /* The adapter keeps the jobs of a client under this name: it has to be
     * another one for every backend of every process of every run. */
    (void)gethostname(host, sizeof(host) - 1);
    be->session = (uint64_t)parsec_qpu_now_ns() ^ (((uint64_t)getpid()) << 32) ^ (uint64_t)(uintptr_t)be;
    for( const char *c = host; '\0' != *c; c++ ) be->session = parsec_qpu_mix64(be->session ^ (uint64_t)(unsigned char)*c);
    be->session = parsec_qpu_mix64(be->session ^ (uint64_t)device_index);

    if( 0 != adapter_connect(be) ) {
        (void)adapter_close(&be->super);
        return PARSEC_ERR_NOT_FOUND;
    }
    parsec_debug_verbose(3, parsec_qpu_output_stream, "QPU %d: adapter at %s, running jobs on %s (%u qubits%s)",
                         device_index, be->address, be->name, be->caps.nb_qubits,
                         be->caps.is_physical ? ", a quantum device" : "");

    pthread_mutex_lock(&adapter_opened_lock);
    if( adapter_opened_n < ADAPTER_MAX_OPEN ) adapter_opened[adapter_opened_n++] = be;
    adapter_opens++;
    pthread_mutex_unlock(&adapter_opened_lock);
    *backend = &be->super;
    return PARSEC_SUCCESS;
}

static int adapter_capabilities(parsec_qpu_backend_t *backend, parsec_qpu_caps_t *caps)
{
    *caps = ((adapter_backend_t*)backend)->caps;
    return PARSEC_SUCCESS;
}

/* Make sure the adapter has a circuit, and give the name it has there. */
static int adapter_prepare(adapter_backend_t *be, const parsec_qpu_circuit_t *circuit, uint64_t *id)
{
    parsec_qpu_json_t reply;
    uint64_t check;
    int64_t length;
    char *text;
    int rc;

    *id = parsec_qpu_adapter_circuit_id(circuit, &check);
    for( uint32_t i = 0; i < be->nb_prepared; i++ ) {
        if( be->prepared[i].id != *id ) continue;
        if( be->prepared[i].check == check ) return PARSEC_SUCCESS;
        parsec_warning("QPU %d: two different circuits have the same name for the adapter (0x%llx); the job of the second is refused",
                       be->number, (unsigned long long)*id);
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    if( (length = parsec_qpu_qasm_write(circuit, NULL, 0)) < 0 ) return PARSEC_ERR_NOT_SUPPORTED;
    if( NULL == (text = (char*)malloc((size_t)length + 1)) ) return PARSEC_ERR_OUT_OF_RESOURCE;
    (void)parsec_qpu_qasm_write(circuit, text, (size_t)length + 1);
    adapter_begin(be, "prepare");
    parsec_qpu_text_add(&be->out, ",\"circuit\":\"0x%llx\",\"qasm\":", (unsigned long long)*id);
    parsec_qpu_text_add_string(&be->out, text, (size_t)length);
    free(text);
    rc = adapter_call(be, &reply);
    if( ADAPTER_DOWN == rc ) return PARSEC_ERR_OUT_OF_RESOURCE;
    if( ADAPTER_REFUSED == rc ) {
        /* Why is the adapter's to say, in its own log: its words may quote the circuit, and are not copied here. */
        parsec_debug_verbose(3, parsec_qpu_output_stream, "QPU %d: the adapter does not take a circuit of %u qubits and %u operations",
                             be->number, circuit->nb_qubits, circuit->nb_ops);
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    if( be->nb_prepared == be->prepared_room ) {
        const uint32_t room = (0 == be->prepared_room) ? 64 : 2 * be->prepared_room;
        adapter_circuit_t *grown = (adapter_circuit_t*)realloc(be->prepared, room * sizeof(adapter_circuit_t));
        if( NULL == grown ) return PARSEC_SUCCESS;  /* not remembered: it will be given again, which is allowed */
        be->prepared      = grown;
        be->prepared_room = room;
    }
    be->prepared[be->nb_prepared].id    = *id;
    be->prepared[be->nb_prepared].check = check;
    be->nb_prepared++;
    ADAPTER_COUNT(be, prepares);
    return PARSEC_SUCCESS;
}

static adapter_job_t *adapter_find(adapter_backend_t *be, parsec_qpu_job_id_t id)
{
    for( adapter_job_t *job = be->jobs; NULL != job; job = job->next ) {
        if( job->id == id ) return job;
    }
    return NULL;
}

/* A job by the name the adapter that is there now has for it. */
static adapter_job_t *adapter_find_remote(adapter_backend_t *be, uint64_t remote)
{
    for( adapter_job_t *job = be->jobs; NULL != job; job = job->next ) {
        if( !job->lost && (job->remote == remote) ) return job;
    }
    return NULL;
}

static void adapter_forget(adapter_backend_t *be, adapter_job_t *job)
{
    for( adapter_job_t **at = &be->jobs; NULL != *at; at = &(*at)->next ) {
        if( *at != job ) continue;
        *at = job->next;
        break;
    }
    free(job);
}

static int adapter_submit(parsec_qpu_backend_t *backend, const parsec_qpu_request_t *requests, int nb,
                          uint64_t tag, parsec_qpu_job_id_t *id)
{
    adapter_backend_t *be = (adapter_backend_t*)backend;
    parsec_qpu_json_t reply, member;
    adapter_job_t *job;
    uint64_t job_id;
    int rc;

    if( 0 != adapter_connect(be) ) return PARSEC_ERR_OUT_OF_RESOURCE;
    for( int i = 0; i < nb; i++ ) {
        for( uint32_t p = 0; (NULL != requests[i].params) && (p < requests[i].circuit->nb_params); p++ ) {
            if( !isfinite(requests[i].params[p]) ) return PARSEC_ERR_NOT_SUPPORTED;     /* no such number in a message */
        }
    }

    for( int attempt = 0; ; attempt++ ) {
        /* The circuits first: a request at a time, so the names are kept aside. */
        uint64_t *names = (uint64_t*)malloc((size_t)nb * sizeof(uint64_t));
        if( NULL == names ) return PARSEC_ERR_OUT_OF_RESOURCE;
        for( int i = 0; i < nb; i++ ) {
            if( PARSEC_SUCCESS != (rc = adapter_prepare(be, requests[i].circuit, &names[i])) ) {
                free(names);
                return rc;
            }
        }
        adapter_begin(be, "submit");
        parsec_qpu_text_add(&be->out, ",\"tag\":\"0x%llx\",\"entries\":[", (unsigned long long)tag);
        for( int i = 0; i < nb; i++ ) {
            parsec_qpu_text_add(&be->out, "%s{\"circuit\":\"0x%llx\",\"shots\":%u,\"key\":\"0x%llx\",\"attempt\":%u,\"params\":[",
                                (i > 0) ? "," : "", (unsigned long long)names[i], requests[i].shots,
                                (unsigned long long)requests[i].key, requests[i].attempt);
            for( uint32_t p = 0; (NULL != requests[i].params) && (p < requests[i].circuit->nb_params); p++ ) {
                char number[40];
                parsec_qpu_text_add(&be->out, "%s%s", (p > 0) ? "," : "", parsec_qpu_number_format(number, requests[i].params[p]));
            }
            parsec_qpu_text_add(&be->out, "]}");
        }
        parsec_qpu_text_add(&be->out, "]");
        free(names);

        rc = adapter_call(be, &reply);
        if( ADAPTER_DOWN == rc ) return PARSEC_ERR_OUT_OF_RESOURCE;
        if( ADAPTER_OK == rc ) break;
        if( adapter_error_is(&reply, "busy") ) return PARSEC_ERR_OUT_OF_RESOURCE;
        if( adapter_error_is(&reply, "unknown_circuit") && (0 == attempt) ) {
            be->nb_prepared = 0;                    /* it forgot circuits: give them again, once */
            continue;
        }
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    if( (PARSEC_SUCCESS != parsec_qpu_json_member(&reply, "job", &member)) ||
        (PARSEC_SUCCESS != parsec_qpu_json_u64(&member, &job_id)) ) {
        ADAPTER_COUNT(be, unanswered);
        adapter_drop(be, "its answer to a submission names no job");
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    if( NULL == (job = adapter_find_remote(be, job_id)) ) {
        job = (adapter_job_t*)calloc(1, sizeof(adapter_job_t) + (size_t)nb * sizeof(adapter_entry_t));
        if( NULL == job ) {
            adapter_settle_add(be, 0, job_id);
            adapter_settle_flush(be);
            return PARSEC_ERR_NOT_SUPPORTED;
        }
        job->id      = be->next_id++;
        job->remote  = job_id;
        job->state   = ADAPTER_FLYING;
        job->pending = ADAPTER_NO_EVENT;
        job->nb      = nb;
        for( int i = 0; i < nb; i++ ) {
            job->entries[i].shots       = requests[i].shots;
            job->entries[i].max_entries = requests[i].max_entries;
            job->entries[i].nb_qubits   = requests[i].circuit->nb_qubits;
        }
        job->next = be->jobs;
        be->jobs  = job;
        /* Its end may have been heard before its identifier was. */
        for( uint32_t e = 0; e < be->nb_early; e++ ) {
            if( be->early[e].remote != job_id ) continue;
            job->pending = be->early[e].state;
            be->early[e] = be->early[--be->nb_early];
            break;
        }
    }
    ADAPTER_COUNT(be, submits);
    PARSEC_QPU_STORE(&be->stats.circuits, be->stats.circuits + (uint64_t)nb);
    *id = job->id;
    return PARSEC_SUCCESS;
}

/* Tell the caller the ends it has not heard. */
static int adapter_report(adapter_backend_t *be, parsec_qpu_event_t *events, int max_events)
{
    int n = 0;

    for( adapter_job_t *job = be->jobs, *next; (NULL != job) && (n < max_events); job = next ) {
        next = job->next;
        if( ADAPTER_NO_EVENT == job->pending ) continue;
        events[n].id    = job->id;
        events[n].state = job->pending;
        n++;
        if( PARSEC_QPU_JOB_DONE == job->pending ) {
            job->pending = ADAPTER_NO_EVENT;
            job->state   = ADAPTER_REPORTED;
        } else {
            /* The adapter forgets it when it hears so, which for one that
             * is out of reach is when it is back; one that is gone has
             * nothing to forget. */
            if( !job->lost ) adapter_settle_add(be, 0, job->remote);
            adapter_forget(be, job);
        }
    }
    return n;
}

static int adapter_state_of(const parsec_qpu_json_t *value)
{
    if( parsec_qpu_json_is(value, "done") )      return PARSEC_QPU_JOB_DONE;
    if( parsec_qpu_json_is(value, "failed") )    return PARSEC_QPU_JOB_FAILED;
    if( parsec_qpu_json_is(value, "cancelled") ) return PARSEC_QPU_JOB_CANCELLED;
    return ADAPTER_NO_EVENT;
}

static int adapter_wait(parsec_qpu_backend_t *backend, int64_t timeout_ns, parsec_qpu_event_t *events, int max_events)
{
    adapter_backend_t *be = (adapter_backend_t*)backend;
    parsec_qpu_json_t reply, jobs, element;
    char *line = NULL;
    size_t length = 0;
    int64_t timeout_ms, deadline;
    int n, woken, rc;

    if( (n = adapter_report(be, events, max_events)) > 0 ) return n;
    woken = adapter_drain(be);
    deadline = (timeout_ns < 0) ? INT64_MAX : parsec_qpu_now_ns() + timeout_ns;

    while( 0 != adapter_connect(be) ) {
        struct pollfd pfd = { .fd = be->wake[0], .events = POLLIN };
        const int64_t now = parsec_qpu_now_ns();
        int64_t sleep_ns = be->t_retry - now;

        /* Out of reach. Its jobs are not given up at once: an adapter that
         * comes back within the time a QPU may be unavailable still has them. */
        if( (0 != be->t_down) && ((now - be->t_down > be->unavailable_ns) || be->dead) ) adapter_lose_jobs(be, be->dead);
        if( (n = adapter_report(be, events, max_events)) > 0 ) return n;
        if( woken || (now >= deadline) ) return 0;
        /* Sleep until it may be tried again, a wake-up, or the end of this wait. */
        if( be->dead || (sleep_ns > ADAPTER_RETRY_MAX_NS) ) sleep_ns = ADAPTER_RETRY_MAX_NS;
        if( sleep_ns < 1000000 ) sleep_ns = 1000000;
        if( sleep_ns > deadline - now ) sleep_ns = deadline - now;
        (void)poll(&pfd, 1, (int)((sleep_ns + 999999) / 1000000));
        woken = adapter_drain(be);
    }
    /* Connecting again may have shown that the jobs are gone. */
    if( (n = adapter_report(be, events, max_events)) > 0 ) return n;

    timeout_ms = (INT64_MAX == deadline) ? ADAPTER_WAIT_MAX_MS : (deadline - parsec_qpu_now_ns() + 999999) / 1000000;
    if( timeout_ms < 0 ) timeout_ms = 0;
    if( timeout_ms > ADAPTER_WAIT_MAX_MS ) timeout_ms = ADAPTER_WAIT_MAX_MS;
    if( woken ) timeout_ms = 0;                     /* still ask: jobs may have ended */
    adapter_begin(be, "wait");
    parsec_qpu_text_add(&be->out, ",\"timeout_ms\":%lld,\"max\":%d", (long long)timeout_ms, max_events);
    if( ADAPTER_OK != adapter_send(be) ) return 0;
    rc = adapter_line(be, parsec_qpu_now_ns() + timeout_ms * 1000000LL + be->timeout_ns, 1, &line, &length);
    if( 2 == rc ) {
        /* Woken while the adapter holds the request: have it answer now. */
        static const char wake[] = "{\"op\":\"wake\"}\n";
        (void)adapter_drain(be);
        if( 0 != adapter_write(be, wake, sizeof(wake) - 1) ) {
            adapter_drop(be, "a request could not be sent");
            return 0;
        }
        rc = adapter_line(be, parsec_qpu_now_ns() + be->timeout_ns, 0, &line, &length);
    }
    if( 1 != rc ) {
        ADAPTER_COUNT(be, unanswered);
        adapter_drop(be, "it did not answer");
        return 0;
    }
    if( ADAPTER_OK != adapter_answer(be, line, length, &reply) ) {
        if( be->fd >= 0 ) adapter_drop(be, "it refuses to be waited for");
        return 0;
    }

    if( PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "jobs", &jobs) ) {
        element.at = NULL;
        while( parsec_qpu_json_next(&jobs, &element) ) {
            parsec_qpu_json_t member;
            adapter_job_t *job;
            uint64_t id;
            int state;

            if( (PARSEC_SUCCESS != parsec_qpu_json_member(&element, "job", &member)) ||
                (PARSEC_SUCCESS != parsec_qpu_json_u64(&member, &id)) ||
                (PARSEC_SUCCESS != parsec_qpu_json_member(&element, "state", &member)) ||
                (ADAPTER_NO_EVENT == (state = adapter_state_of(&member))) ) continue;
            if( NULL == (job = adapter_find_remote(be, id)) ) {
                /* A job whose identifier submit() has not returned: its end
                 * is kept until it has, and is not reported before. (Told
                 * again after a new connection, it is kept once.) */
                uint32_t e = 0;
                while( (e < be->nb_early) && (be->early[e].remote != id) ) e++;
                if( (e == be->nb_early) && (be->nb_early == be->early_room) ) {
                    const uint32_t room = (0 == be->early_room) ? 16 : 2 * be->early_room;
                    adapter_early_t *grown = (adapter_early_t*)realloc(be->early, room * sizeof(adapter_early_t));
                    if( NULL == grown ) continue;
                    be->early      = grown;
                    be->early_room = room;
                }
                if( e == be->nb_early ) be->nb_early++;
                be->early[e].remote = id;
                be->early[e].state  = state;
                continue;
            }
            if( ADAPTER_FLYING != job->state ) continue;        /* told again after a new connection */
            job->pending = state;
        }
    }
    n = adapter_report(be, events, max_events);
    adapter_settle_flush(be);
    return n;
}

static int adapter_wakeup(parsec_qpu_backend_t *backend)
{
    adapter_backend_t *be = (adapter_backend_t*)backend;
    const char byte = 'w';
    /* A full pipe is a pipe that will wake the thread all the same. */
    if( write(be->wake[1], &byte, 1) < 0 ) { /* nothing to do about it */ }
    return PARSEC_SUCCESS;
}

static int adapter_cmp_bits(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

/* (bitstring, count) pairs: most frequent first, then by bitstring. The
 * order in which the exact sampler cuts a result that does not fit. */
static int adapter_cmp_count(const void *a, const void *b)
{
    const uint64_t *x = (const uint64_t*)a, *y = (const uint64_t*)b;
    if( x[1] != y[1] ) return (x[1] < y[1]) - (x[1] > y[1]);
    return (x[0] > y[0]) - (x[0] < y[0]);
}

/* One result from one entry of the adapter's answer. */
static int adapter_fill(adapter_backend_t *be, const parsec_qpu_json_t *entry, const adapter_entry_t *asked,
                        parsec_qpu_result_t *result, int64_t now)
{
    parsec_qpu_json_t counts, pair, part;
    uint64_t *pairs, total = 0;
    uint32_t distinct = 0, kept, n = 0;

    if( (PARSEC_SUCCESS != parsec_qpu_json_member(entry, "counts", &counts)) || ('a' != parsec_qpu_json_kind(&counts)) ) return PARSEC_ERROR;
    pair.at = NULL;
    while( parsec_qpu_json_next(&counts, &pair) ) n++;
    if( (0 == n) || (n > asked->shots) ) return PARSEC_ERROR;
    if( NULL == (pairs = (uint64_t*)malloc((size_t)n * 2 * sizeof(uint64_t))) ) return PARSEC_ERR_OUT_OF_RESOURCE;

    pair.at = NULL;
    for( uint32_t i = 0; i < n; i++ ) {
        int64_t count = 0;
        uint64_t bits = 0;
        (void)parsec_qpu_json_next(&counts, &pair);
        part.at = NULL;
        if( !parsec_qpu_json_next(&pair, &part) || (PARSEC_SUCCESS != parsec_qpu_json_u64(&part, &bits)) ||
            !parsec_qpu_json_next(&pair, &part) || (PARSEC_SUCCESS != parsec_qpu_json_int(&part, &count)) ||
            (count <= 0) || (count > (int64_t)asked->shots) ||
            ((asked->nb_qubits < 64) && (0 != (bits >> asked->nb_qubits))) ) {
            free(pairs);
            return PARSEC_ERROR;
        }
        pairs[2 * i]     = bits;
        pairs[2 * i + 1] = (uint64_t)count;
        total += (uint64_t)count;
    }
    /* Every shot has to be there, and no bitstring twice. */
    qsort(pairs, n, 2 * sizeof(uint64_t), adapter_cmp_bits);
    for( uint32_t i = 1; i < n; i++ ) {
        if( pairs[2 * i] == pairs[2 * (i - 1)] ) { free(pairs); return PARSEC_ERROR; }
    }
    if( total != asked->shots ) { free(pairs); return PARSEC_ERROR; }
    distinct = n;

    result->status    = PARSEC_QPU_STATUS_OK;
    result->flags     = be->caps.is_physical ? 0 : PARSEC_QPU_RESULT_MOCK;
    result->nb_qubits = (uint16_t)asked->nb_qubits;
    {
        /* How long ago the job started and ended, by the adapter's clock:
         * not before this process could have asked, and not in the future. */
        int64_t since_start = adapter_member_int(entry, "since_start_ns", 0);
        int64_t since_end   = adapter_member_int(entry, "since_end_ns", 0);
        if( (since_end < 0) || (since_end > now) ) since_end = 0;
        if( (since_start < since_end) || (since_start > now) ) since_start = since_end;
        result->t_start = now - since_start;
        result->t_end   = now - since_end;
    }
    kept = distinct;
    if( distinct > asked->max_entries ) {
        /* More outcomes than the buffer holds: keep the most frequent ones. */
        result->flags |= PARSEC_QPU_RESULT_TRUNCATED;
        kept = asked->max_entries;
        qsort(pairs, distinct, 2 * sizeof(uint64_t), adapter_cmp_count);
        qsort(pairs, kept, 2 * sizeof(uint64_t), adapter_cmp_bits);
    }
    result->nb_entries     = kept;
    result->shots_returned = 0;
    for( uint32_t i = 0; i < kept; i++ ) {
        parsec_qpu_result_set_entry(result, i, pairs[2 * i], (uint32_t)pairs[2 * i + 1]);
        result->shots_returned += (uint32_t)pairs[2 * i + 1];
    }
    free(pairs);
    return PARSEC_SUCCESS;
}

static int adapter_results(parsec_qpu_backend_t *backend, parsec_qpu_job_id_t id, parsec_qpu_result_t **results, int nb)
{
    adapter_backend_t *be = (adapter_backend_t*)backend;
    adapter_job_t *job = adapter_find(be, id);
    parsec_qpu_json_t reply, entries, entry;
    int rc = PARSEC_SUCCESS, i = 0;
    int64_t now;

    if( (NULL == job) || (ADAPTER_REPORTED != job->state) ) return PARSEC_ERR_NOT_FOUND;
    if( job->lost ) {
        /* The adapter that had the result is gone. The one that is there
         * may have another job under the same name: it is not asked. */
        adapter_forget(be, job);
        ADAPTER_COUNT(be, bad_results);
        return PARSEC_ERROR;
    }
    if( nb != job->nb ) {
        /* Not the job the caller takes it for. It is forgotten all the same. */
        adapter_settle_add(be, 0, job->remote);
        adapter_settle_flush(be);
        adapter_forget(be, job);
        return PARSEC_ERR_BAD_PARAM;
    }
    if( (0 != adapter_connect(be)) || job->lost ) {
        /* Out of reach; or connecting again showed that it was replaced. */
        if( !job->lost ) adapter_settle_add(be, 0, job->remote);
        adapter_forget(be, job);
        ADAPTER_COUNT(be, bad_results);
        return PARSEC_ERROR;
    }
    adapter_begin(be, "results");
    parsec_qpu_text_add(&be->out, ",\"job\":\"0x%llx\"", (unsigned long long)job->remote);
    rc = adapter_call(be, &reply);
    if( ADAPTER_OK != rc ) {
        /* No answer: the adapter may still hold the job, and forgets it when it hears so. */
        if( ADAPTER_DOWN == rc ) adapter_settle_add(be, 0, job->remote);
        adapter_forget(be, job);
        ADAPTER_COUNT(be, bad_results);
        return PARSEC_ERROR;
    }
    now = parsec_qpu_now_ns();
    rc = PARSEC_ERROR;
    if( PARSEC_SUCCESS == parsec_qpu_json_member(&reply, "entries", &entries) ) {
        rc = PARSEC_SUCCESS;
        entry.at = NULL;
        for( i = 0; (PARSEC_SUCCESS == rc) && (i < nb); i++ ) {
            if( !parsec_qpu_json_next(&entries, &entry) ) { rc = PARSEC_ERROR; break; }
            rc = adapter_fill(be, &entry, &job->entries[i], results[i], now);
        }
        if( (PARSEC_SUCCESS == rc) && parsec_qpu_json_next(&entries, &entry) ) rc = PARSEC_ERROR;   /* more answers than circuits */
    }
    if( PARSEC_SUCCESS == rc ) {
        PARSEC_QPU_STORE(&be->stats.results, be->stats.results + (uint64_t)nb);
    } else {
        ADAPTER_COUNT(be, bad_results);
        parsec_debug_verbose(3, parsec_qpu_output_stream, "QPU %d: what the adapter gave for a job of %d circuit(s) is not its result",
                             be->number, nb);
    }
    adapter_forget(be, job);
    return rc;
}

static int adapter_cancel(parsec_qpu_backend_t *backend, parsec_qpu_job_id_t id)
{
    adapter_backend_t *be = (adapter_backend_t*)backend;
    adapter_job_t *job = adapter_find(be, id);

    if( NULL == job ) return PARSEC_ERR_NOT_FOUND;
    if( !job->lost ) adapter_settle_add(be, 0, job->remote);
    adapter_forget(be, job);
    adapter_settle_flush(be);
    return PARSEC_SUCCESS;
}

static int adapter_abandon(parsec_qpu_backend_t *backend, uint64_t tag)
{
    adapter_backend_t *be = (adapter_backend_t*)backend;
    /* If the adapter has a job under this tag it is cancelled there, now or
     * when the adapter can be reached again. */
    adapter_settle_add(be, 1, tag);
    adapter_settle_flush(be);
    return PARSEC_SUCCESS;
}

static int adapter_close(parsec_qpu_backend_t *backend)
{
    adapter_backend_t *be = (adapter_backend_t*)backend;

    pthread_mutex_lock(&adapter_opened_lock);
    for( int i = 0; i < adapter_opened_n; i++ ) {
        if( adapter_opened[i] != be ) continue;
        adapter_opened[i] = adapter_opened[--adapter_opened_n];
        break;
    }
    pthread_mutex_unlock(&adapter_opened_lock);

    /* Say goodbye, also to an adapter that was out of reach a moment ago:
     * it drops what this client left, and nothing runs on for nobody. */
    if( be->greeted && !be->dead ) {
        parsec_qpu_json_t reply;
        be->t_retry = 0;
        if( be->timeout_ns > 2000000000LL ) be->timeout_ns = 2000000000LL;
        be->warned = 1;                             /* at the end of a run this is not worth a warning */
        if( 0 == adapter_connect(be) ) {
            adapter_settle_flush(be);
        }
        if( be->fd >= 0 ) {
            adapter_begin(be, "bye");
            (void)adapter_call(be, &reply);
        }
    }
    if( be->fd >= 0 ) close(be->fd);
    if( be->wake[0] >= 0 ) close(be->wake[0]);
    if( be->wake[1] >= 0 ) close(be->wake[1]);
    while( NULL != be->jobs ) adapter_forget(be, be->jobs);
    parsec_qpu_text_fini(&be->out);
    free(be->prepared);
    free(be->settle);
    free(be->early);
    free(be->in);
    free(be->address);
    free(be);
    return PARSEC_SUCCESS;
}

const parsec_qpu_backend_ops_t parsec_qpu_backend_adapter_ops = {
    .name         = "adapter",
    /* What it drives is what its adapter says once it is open: the second
     * check of the guard, on the capabilities, is the one that applies. */
    .physical     = 0,
    .open         = adapter_open,
    .capabilities = adapter_capabilities,
    .submit       = adapter_submit,
    .wait         = adapter_wait,
    .wakeup       = adapter_wakeup,
    .results      = adapter_results,
    .cancel       = adapter_cancel,
    .abandon      = adapter_abandon,
    .close        = adapter_close
};
