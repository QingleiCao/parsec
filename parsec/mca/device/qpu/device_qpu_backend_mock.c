/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * The mock QPU backend. It runs in the process, needs no credentials and no
 * network, and reaches no quantum device. Every result it gives is flagged
 * PARSEC_QPU_RESULT_MOCK.
 *
 * Time. The mock has a fixed number of slots and a queue in front of them,
 * as an account at a provider has. A job that gets a slot first waits its
 * queue delay there, then runs for its duration. A queued job starts at the
 * instant a slot was freed, not at the instant somebody noticed, so the
 * schedule of the jobs does not depend on how late the progress thread
 * wakes up.
 *
 * What a job costs. A job carries one circuit or several
 * (device_qpu_mock_max_circuits at most). It runs for a part that every job
 * pays whatever it carries (device_qpu_mock_job_delay) plus a part for each
 * of its circuits (device_qpu_mock_delay): what a provider spends per job,
 * and what it spends running one circuit. A job is one thing: it gets a
 * slot, ends, fails, hangs or is refused as a whole.
 *
 * Answers. The counts of a circuit come from the exact sampler, which
 * helper threads of the mock run while the job waits and runs; one helper
 * samples the circuits of a job one after the other. A job therefore ends
 * at the later of its modelled end and the end of that computation.
 *
 * Reproducibility. Whatever is drawn for a circuit (the time it runs, its
 * fate when faults are injected by probability, its samples) comes from the
 * seed, the key of the evaluation and the attempt number, never from a
 * generator shared between jobs. Times are drawn from a seed of their own
 * when device_qpu_mock_delay_seed is set, so that a run can be repeated with
 * other times and the same samples; what is drawn for a job (its queue delay
 * and its fixed part) comes from those of its first circuit. The samples do
 * not depend on the attempt number, nor on what else the job carries: a
 * circuit that is run again, alone or with others, gives the counts it
 * would have given the first time. What does depend on the company is the
 * fate of the job, which is that of the first of its circuits that has
 * one: how many jobs a task takes is reproducible only as far as the jobs
 * are made of the same tasks.
 *
 * Faults. For tests and experiments the mock can fail jobs, hang them, lose
 * the reply to a submission, be unavailable, and report ends twice or late
 * (parsec_qpu_mock_fault_t). It also watches its caller: a submission under
 * the tag of a job the caller already knows, two jobs of one evaluation held
 * at once, inputs that changed while a job was held, circuits of one job
 * that ask for different numbers of shots, and a job asked for again with
 * other circuits than it has are counted, and none of them may happen.
 */

typedef enum { MOCK_QUEUED, MOCK_RUNNING, MOCK_COMPUTING, MOCK_FINISHED } mock_state_t;
typedef enum { WORK_NONE, WORK_QUEUED, WORK_RUNNING, WORK_DONE } mock_work_t;

typedef struct mock_job_s {
    struct mock_job_s      *next;        /* in the queue or in the list of late jobs */
    struct mock_job_s      *work_next;   /* in the helpers' queue */
    parsec_qpu_job_id_t     id;
    uint64_t                tag;
    uint32_t                nb;          /* circuits it carries */
    parsec_qpu_request_t   *requests;    /* [nb] */
    uint64_t               *inputs;      /* [nb] digest of each circuit and its parameters at submission */
    parsec_qpu_result_t   **counts;      /* [nb] sampled by a helper; NULL for a job that will not end well */
    uint32_t               *counts_cap;  /* [nb] */
    int64_t                 queue_delay;
    int64_t                 duration;    /* negative: it never ends */
    int64_t                 t_arrival;
    int64_t                 t_start;
    int64_t                 t_end;
    mock_state_t            state;
    parsec_qpu_mock_fault_t fault;
    int                     final;       /* PARSEC_QPU_JOB_*, once finished */
    int                     acked;       /* the caller was told its id */
    int                     event_due;   /* it finished before that */
    mock_work_t             work;
    int                     work_rc;
    int32_t                 cancel;      /* tells a helper to give up */
} mock_job_t;

typedef struct mock_event_s {
    struct mock_event_s *next;
    parsec_qpu_event_t   event;
    int64_t              when;           /* not to be reported before */
} mock_event_t;

typedef enum { RULE_P, RULE_FIRST, RULE_EVERY, RULE_KEY, RULE_FOR } mock_rule_kind_t;

typedef struct mock_rule_s {
    parsec_qpu_mock_fault_t fault;
    mock_rule_kind_t        kind;
    double                  p;
    uint64_t                n;
} mock_rule_t;

#define MOCK_MAX_RULES    8
#define MOCK_MAX_HELPERS  64
/* How long after the real one a duplicated or late event is reported. */
#define MOCK_GHOST_DELAY  200000LL

enum { DELAY_CONST, DELAY_UNIFORM, DELAY_EXP, DELAY_LOGNORMAL, DELAY_REPLAY };

typedef struct mock_backend_s {
    parsec_qpu_backend_t  super;

    /* The lock protects everything below it. Apart from wakeup(), the
     * functions of the backend come from one thread, so the lock is
     * contended only by the helpers when they take or give back a job, by
     * readers of the statistics, and, for an instant, by a wakeup() that
     * finds that thread asleep. */
    pthread_mutex_t       lock;
    pthread_cond_t        cond;       /* the thread in wait() sleeps here */
    /* wakeup() must not block on that lock while the progress thread works.
     * Two flags, written and read without the lock: wakeup() raises `wake`
     * and then looks at `sleeping`; wait() raises `sleeping` and then looks
     * at `wake`. One of the two always sees the other, so a wake-up is never
     * lost, and wakeup() takes the lock only to signal a thread that sleeps
     * or is about to, which does not hold it. */
    int32_t               wake;       /* a wakeup() nobody has consumed yet */
    int32_t               sleeping;   /* the progress thread is in, or entering, the condition wait */

    uint64_t              seed;
    uint64_t              delay_seed; /* of the times; the seed unless set apart */
    uint32_t              slots;
    uint32_t              max_qubits;
    uint32_t              max_circuits; /* per job */
    int64_t               submit_delay; /* time it takes to answer a submission */
    parsec_qpu_delay_t    delay;      /* time one circuit of a job runs */
    parsec_qpu_delay_t    job_delay;  /* time a job runs on top of that, whatever it carries */
    parsec_qpu_delay_t    queue_delay;/* time it waits in its slot before that */
    mock_rule_t           rules[MOCK_MAX_RULES];
    int                   nb_rules;
    int64_t               t_open;
    int64_t               t_last_call;

    parsec_qpu_job_id_t   next_id;
    mock_job_t          **held;       /* every job the mock has not forgotten */
    uint32_t              held_n, held_cap;
    mock_job_t          **running;    /* heap, earliest end first */
    uint32_t              running_n;
    mock_job_t           *late;       /* past their modelled end, still being sampled; they keep their slot */
    uint32_t              late_n;
    mock_job_t           *queue_head, *queue_tail;
    mock_event_t         *events, *events_tail;

    uint64_t             *tags;       /* every tag a job was created under */
    size_t                tags_cap, tags_n;

    /* Helpers */
    pthread_t             helpers[MOCK_MAX_HELPERS];
    int                   nb_helpers;
    int                   helpers_stop;
    pthread_cond_t        work_cond;  /* helpers sleep here */
    pthread_cond_t        idle_cond;  /* cancel() waits here for a helper to let go of a job */
    mock_job_t           *work_head, *work_tail;

    parsec_qpu_mock_stats_t stats;
    int                   number;     /* of the QPU it was opened for */
} mock_backend_t;

/* The mocks that are open, one per QPU of this process that uses the mock;
 * tests read their counters. */
#define MOCK_MAX_OPEN  64
static mock_backend_t *mock_opened[MOCK_MAX_OPEN];
static int mock_opened_n = 0;
static pthread_mutex_t mock_current_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t mock_opens = 0;

static parsec_qpu_mock_delay_fn_t mock_delay_fn  = NULL;
static void                      *mock_delay_arg = NULL;
static parsec_qpu_mock_fault_fn_t mock_fault_fn  = NULL;
static void                      *mock_fault_arg = NULL;

void parsec_qpu_mock_set_delay_fn(parsec_qpu_mock_delay_fn_t fn, void *arg)
{
    mock_delay_fn  = fn;
    mock_delay_arg = arg;
}

void parsec_qpu_mock_set_fault_fn(parsec_qpu_mock_fault_fn_t fn, void *arg)
{
    mock_fault_fn  = fn;
    mock_fault_arg = arg;
}

/* What one mock counted, added to what is there: counts are added up, the
 * high-water marks are the largest, and the shortest time between two
 * submissions is the shortest. Called with mock_current_lock held. */
static void mock_stats_add(mock_backend_t *mock, parsec_qpu_mock_stats_t *stats)
{
    const parsec_qpu_mock_stats_t *m = &mock->stats;

    pthread_mutex_lock(&mock->lock);
    stats->submit_calls     += m->submit_calls;
    stats->submits          += m->submits;
    stats->circuits         += m->circuits;
    stats->tag_hits         += m->tag_hits;
    stats->deferred         += m->deferred;
    stats->rejected         += m->rejected;
    stats->cancels          += m->cancels;
    stats->abandoned        += m->abandoned;
    stats->results          += m->results;
    stats->failures         += m->failures;
    stats->provider_cancels += m->provider_cancels;
    stats->ghost_events     += m->ghost_events;
    stats->computed         += m->computed;
    stats->late             += m->late;
    stats->duplicate_tags   += m->duplicate_tags;
    stats->key_overlaps     += m->key_overlaps;
    stats->inputs_changed   += m->inputs_changed;
    stats->mixed_shots      += m->mixed_shots;
    stats->changed_jobs     += m->changed_jobs;
    if( m->max_running  > stats->max_running )  stats->max_running  = m->max_running;
    if( m->max_held     > stats->max_held )     stats->max_held     = m->max_held;
    if( m->max_circuits > stats->max_circuits ) stats->max_circuits = m->max_circuits;
    if( (m->min_submit_gap >= 0) && ((stats->min_submit_gap < 0) || (m->min_submit_gap < stats->min_submit_gap)) ) {
        stats->min_submit_gap = m->min_submit_gap;
    }
    for( uint32_t i = 0; i < mock->held_n; i++ ) {
        if( !mock->held[i]->acked ) stats->unknown_jobs++;
    }
    pthread_mutex_unlock(&mock->lock);
}

void parsec_qpu_mock_stats(parsec_qpu_mock_stats_t *stats)
{
    memset(stats, 0, sizeof(*stats));
    stats->min_submit_gap = -1;
    pthread_mutex_lock(&mock_current_lock);
    for( int i = 0; i < mock_opened_n; i++ ) {
        mock_stats_add(mock_opened[i], stats);
    }
    stats->opens = mock_opens;
    pthread_mutex_unlock(&mock_current_lock);
}

int parsec_qpu_mock_stats_of(int qpu, parsec_qpu_mock_stats_t *stats)
{
    int rc = PARSEC_ERR_NOT_FOUND;

    memset(stats, 0, sizeof(*stats));
    stats->min_submit_gap = -1;
    pthread_mutex_lock(&mock_current_lock);
    for( int i = 0; i < mock_opened_n; i++ ) {
        if( mock_opened[i]->number != qpu ) continue;
        mock_stats_add(mock_opened[i], stats);
        rc = PARSEC_SUCCESS;
        break;
    }
    stats->opens = mock_opens;
    pthread_mutex_unlock(&mock_current_lock);
    return rc;
}

/* ------------------------------------------------------------------------
 * Delays
 * --------------------------------------------------------------------- */

static int delay_read_file(parsec_qpu_delay_t *delay, const char *path)
{
    FILE *file = fopen(path, "r");
    char line[256];
    uint32_t cap = 0;

    if( NULL == file ) return PARSEC_ERR_NOT_FOUND;
    while( NULL != fgets(line, sizeof(line), file) ) {
        char *end, *start = line;
        double us;
        while( (' ' == *start) || ('\t' == *start) ) start++;
        if( ('#' == *start) || ('\n' == *start) || ('\r' == *start) || ('\0' == *start) ) continue;
        errno = 0;
        us = strtod(start, &end);
        if( (end == start) || (0 != errno) || !(us >= 0.0) || !isfinite(us) ) {
            fclose(file);
            return PARSEC_ERR_BAD_PARAM;
        }
        if( delay->nb_samples == cap ) {
            cap = (0 == cap) ? 256 : 2 * cap;
            delay->samples = (int64_t*)realloc(delay->samples, cap * sizeof(int64_t));
        }
        delay->samples[delay->nb_samples++] = (int64_t)(us * 1000.0);
    }
    fclose(file);
    return (delay->nb_samples > 0) ? PARSEC_SUCCESS : PARSEC_ERR_BAD_PARAM;
}

int parsec_qpu_delay_parse(parsec_qpu_delay_t *delay, const char *spec)
{
    double a = 0.0, b = 0.0;
    int n = 0;

    memset(delay, 0, sizeof(*delay));
    if( NULL == spec ) return PARSEC_ERR_BAD_PARAM;

    if( (1 == sscanf(spec, "const:%lf%n", &a, &n)) && ('\0' == spec[n]) && (a >= 0.0) ) {
        delay->kind = DELAY_CONST;
        delay->a    = a * 1000.0;
        return PARSEC_SUCCESS;
    }
    if( (2 == sscanf(spec, "uniform:%lf:%lf%n", &a, &b, &n)) && ('\0' == spec[n]) && (a >= 0.0) && (b >= a) ) {
        delay->kind = DELAY_UNIFORM;
        delay->a    = a * 1000.0;
        delay->b    = b * 1000.0;
        return PARSEC_SUCCESS;
    }
    if( (1 == sscanf(spec, "exp:%lf%n", &a, &n)) && ('\0' == spec[n]) && (a >= 0.0) ) {
        delay->kind = DELAY_EXP;
        delay->a    = a * 1000.0;
        return PARSEC_SUCCESS;
    }
    if( (2 == sscanf(spec, "lognormal:%lf:%lf%n", &a, &b, &n)) && ('\0' == spec[n]) && (a >= 0.0) && (b >= 0.0) ) {
        delay->kind = DELAY_LOGNORMAL;
        delay->a    = a * 1000.0;
        delay->b    = b;
        return PARSEC_SUCCESS;
    }
    if( 0 == strncmp(spec, "replay:", 7) ) {
        int rc;
        delay->kind = DELAY_REPLAY;
        if( PARSEC_SUCCESS != (rc = delay_read_file(delay, spec + 7)) ) {
            parsec_qpu_delay_fini(delay);
        }
        return rc;
    }
    return PARSEC_ERR_BAD_PARAM;
}

void parsec_qpu_delay_fini(parsec_qpu_delay_t *delay)
{
    free(delay->samples);
    memset(delay, 0, sizeof(*delay));
}

int64_t parsec_qpu_delay_draw(const parsec_qpu_delay_t *delay, uint64_t bits)
{
    /* An hour: far beyond anything a mock run is for, and far from overflow. */
    static const double longest = 3.6e12;
    const double u = parsec_qpu_unit(bits);
    double ns = 0.0;

    switch( delay->kind ) {
    case DELAY_CONST:
        ns = delay->a;
        break;
    case DELAY_UNIFORM:
        ns = delay->a + u * (delay->b - delay->a);
        break;
    case DELAY_EXP:
        ns = -delay->a * log(1.0 - u);
        break;
    case DELAY_LOGNORMAL: {
        /* Box-Muller, from two numbers made of the same bits. */
        const double v = parsec_qpu_unit(parsec_qpu_mix64(bits));
        const double z = sqrt(-2.0 * log(1.0 - u)) * cos(6.283185307179586476925 * v);
        ns = delay->a * exp(delay->b * z);
        break;
    }
    case DELAY_REPLAY:
        return delay->samples[bits % delay->nb_samples];
    }
    if( !(ns >= 0.0) ) ns = 0.0;
    if( ns > longest ) ns = longest;
    return (int64_t)ns;
}

/* What is drawn for a job comes from a seed, the key of its evaluation, the
 * attempt, and what the draw is for. */
static inline uint64_t mock_bits_of(uint64_t seed, uint64_t key, uint32_t attempt, uint64_t purpose)
{
    return parsec_qpu_mix64(parsec_qpu_mix64(parsec_qpu_mix64(seed) ^ key)
                            + (uint64_t)attempt * 0x9e3779b97f4a7c15ULL
                            + purpose * 0xbf58476d1ce4e5b9ULL);
}

static inline uint64_t mock_bits(const mock_backend_t *mock, uint64_t key, uint32_t attempt, uint64_t purpose)
{
    return mock_bits_of(mock->seed, key, attempt, purpose);
}

/* The same for a time: from the seed of the times. */
static inline uint64_t mock_time_bits(const mock_backend_t *mock, uint64_t key, uint32_t attempt, uint64_t purpose)
{
    return mock_bits_of(mock->delay_seed, key, attempt, purpose);
}

int64_t parsec_qpu_mock_time(const parsec_qpu_delay_t *delay, uint64_t delay_seed, uint64_t key, uint32_t attempt,
                             parsec_qpu_mock_time_t what)
{
    return parsec_qpu_delay_draw(delay, mock_bits_of(delay_seed, key, attempt, (uint64_t)what));
}

/* ------------------------------------------------------------------------
 * Faults: device_qpu_mock_faults is "none" or a list, separated by commas,
 * of <fault>:<rule>=<value>. The first rule that applies decides.
 *   fault: fail hang unavailable lost reject noresult duplicate late cancelled
 *   rule:  p=<probability>   drawn per job and attempt, or per submission
 *                            for unavailable, lost and reject
 *          first=<n>         the first n attempts of every job
 *          every=<n>         every n-th submission the mock receives
 *          key=<k>           the evaluation with this key, every attempt
 *          for=<ms>          for that long after the backend was opened
 * --------------------------------------------------------------------- */

static const char *mock_fault_names[PARSEC_QPU_MOCK_FAULT_MAX] = {
    [PARSEC_QPU_MOCK_OK]          = "ok",
    [PARSEC_QPU_MOCK_FAIL]        = "fail",
    [PARSEC_QPU_MOCK_HANG]        = "hang",
    [PARSEC_QPU_MOCK_UNAVAILABLE] = "unavailable",
    [PARSEC_QPU_MOCK_LOST]        = "lost",
    [PARSEC_QPU_MOCK_REJECT]      = "reject",
    [PARSEC_QPU_MOCK_NORESULT]    = "noresult",
    [PARSEC_QPU_MOCK_DUPLICATE]   = "duplicate",
    [PARSEC_QPU_MOCK_LATE]        = "late",
    [PARSEC_QPU_MOCK_CANCELLED]   = "cancelled"
};

static int mock_parse_faults(mock_backend_t *mock, const char *spec)
{
    char *copy, *save = NULL;

    mock->nb_rules = 0;
    if( (NULL == spec) || ('\0' == spec[0]) || (0 == strcmp(spec, "none")) ) return PARSEC_SUCCESS;

    copy = strdup(spec);
    for( char *item = strtok_r(copy, ",", &save); NULL != item; item = strtok_r(NULL, ",", &save) ) {
        mock_rule_t *rule = &mock->rules[mock->nb_rules];
        char *colon = strchr(item, ':'), *equal, *end;
        int fault;

        if( (MOCK_MAX_RULES == mock->nb_rules) || (NULL == colon) || (NULL == (equal = strchr(colon, '='))) ) goto bad;
        *colon = '\0';
        *equal = '\0';
        for( fault = 1; fault < PARSEC_QPU_MOCK_FAULT_MAX; fault++ ) {
            if( 0 == strcmp(item, mock_fault_names[fault]) ) break;
        }
        if( PARSEC_QPU_MOCK_FAULT_MAX == fault ) goto bad;
        rule->fault = (parsec_qpu_mock_fault_t)fault;

        errno = 0;
        if( 0 == strcmp(colon + 1, "p") ) {
            rule->kind = RULE_P;
            rule->p    = strtod(equal + 1, &end);
            if( !(rule->p >= 0.0) || !(rule->p <= 1.0) ) goto bad;
        } else {
            if(      0 == strcmp(colon + 1, "first") ) rule->kind = RULE_FIRST;
            else if( 0 == strcmp(colon + 1, "every") ) rule->kind = RULE_EVERY;
            else if( 0 == strcmp(colon + 1, "key") )   rule->kind = RULE_KEY;
            else if( 0 == strcmp(colon + 1, "for") )   rule->kind = RULE_FOR;
            else goto bad;
            rule->n = strtoull(equal + 1, &end, 0);
            if( (RULE_EVERY == rule->kind) && (0 == rule->n) ) goto bad;
        }
        if( (end == equal + 1) || ('\0' != *end) || (0 != errno) ) goto bad;
        mock->nb_rules++;
    }
    free(copy);
    return PARSEC_SUCCESS;

  bad:
    free(copy);
    mock->nb_rules = 0;
    return PARSEC_ERR_BAD_PARAM;
}

static parsec_qpu_mock_fault_t mock_decide(mock_backend_t *mock, uint64_t tag,
                                           const parsec_qpu_request_t *request, uint64_t call, int64_t now)
{
    if( NULL != mock_fault_fn ) {
        return mock_fault_fn(tag, request, call, mock_fault_arg);
    }
    for( int i = 0; i < mock->nb_rules; i++ ) {
        const mock_rule_t *rule = &mock->rules[i];
        /* A submission that is not taken comes back with the same attempt
         * number: what is drawn for it has to change from one call to the
         * next, or it would be refused for ever. */
        const int per_call = (PARSEC_QPU_MOCK_UNAVAILABLE == rule->fault) ||
                             (PARSEC_QPU_MOCK_LOST == rule->fault) ||
                             (PARSEC_QPU_MOCK_REJECT == rule->fault);
        int hit = 0;

        switch( rule->kind ) {
        case RULE_P:
            hit = parsec_qpu_unit(per_call
                                  ? parsec_qpu_mix64(parsec_qpu_mix64(mock->seed) ^ (call * 0x9e3779b97f4a7c15ULL) ^ (uint64_t)(16 + i))
                                  : mock_bits(mock, request->key, request->attempt, (uint64_t)(16 + i))) < rule->p;
            break;
        case RULE_FIRST: hit = (request->attempt <= rule->n);                         break;
        case RULE_EVERY: hit = (0 == (call % rule->n));                               break;
        case RULE_KEY:   hit = (request->key == rule->n);                             break;
        case RULE_FOR:   hit = ((now - mock->t_open) < (int64_t)rule->n * 1000000LL); break;
        }
        if( hit ) return rule->fault;
    }
    return PARSEC_QPU_MOCK_OK;
}

/* ------------------------------------------------------------------------
 * Small containers. The lock is held in all of them.
 * --------------------------------------------------------------------- */

/* The tags jobs were created under: a set, by open addressing. Zero marks a
 * free place, so the tag zero is kept as one. */
static int tags_has(const mock_backend_t *mock, uint64_t tag)
{
    if( 0 == mock->tags_cap ) return 0;
    if( 0 == tag ) tag = 1;
    for( size_t i = parsec_qpu_mix64(tag) & (mock->tags_cap - 1); ; i = (i + 1) & (mock->tags_cap - 1) ) {
        if( 0 == mock->tags[i] ) return 0;
        if( tag == mock->tags[i] ) return 1;
    }
}

static void tags_put(uint64_t *tags, size_t cap, uint64_t tag)
{
    size_t i = parsec_qpu_mix64(tag) & (cap - 1);
    while( 0 != tags[i] ) i = (i + 1) & (cap - 1);
    tags[i] = tag;
}

static void tags_add(mock_backend_t *mock, uint64_t tag)
{
    if( 0 == tag ) tag = 1;
    if( 2 * (mock->tags_n + 1) > mock->tags_cap ) {
        size_t cap = (0 == mock->tags_cap) ? 1024 : 2 * mock->tags_cap;
        uint64_t *tags = (uint64_t*)calloc(cap, sizeof(uint64_t));
        for( size_t i = 0; i < mock->tags_cap; i++ ) {
            if( 0 != mock->tags[i] ) tags_put(tags, cap, mock->tags[i]);
        }
        free(mock->tags);
        mock->tags     = tags;
        mock->tags_cap = cap;
    }
    tags_put(mock->tags, mock->tags_cap, tag);
    mock->tags_n++;
}

static void held_add(mock_backend_t *mock, mock_job_t *job)
{
    if( mock->held_n == mock->held_cap ) {
        mock->held_cap = (0 == mock->held_cap) ? 32 : 2 * mock->held_cap;
        mock->held = (mock_job_t**)realloc(mock->held, mock->held_cap * sizeof(mock_job_t*));
    }
    mock->held[mock->held_n++] = job;
    if( mock->held_n > mock->stats.max_held ) mock->stats.max_held = mock->held_n;
}

static void held_remove(mock_backend_t *mock, mock_job_t *job)
{
    for( uint32_t i = 0; i < mock->held_n; i++ ) {
        if( mock->held[i] != job ) continue;
        mock->held[i] = mock->held[--mock->held_n];
        return;
    }
}

static mock_job_t *held_find(mock_backend_t *mock, parsec_qpu_job_id_t id)
{
    for( uint32_t i = 0; i < mock->held_n; i++ ) {
        if( mock->held[i]->id == id ) return mock->held[i];
    }
    return NULL;
}

/* A job of nb circuits, in one block. */
static mock_job_t *mock_job_new(uint32_t nb)
{
    mock_job_t *job = (mock_job_t*)calloc(1, sizeof(mock_job_t) +
                                             (size_t)nb * (sizeof(parsec_qpu_request_t) + sizeof(uint64_t) +
                                                           sizeof(parsec_qpu_result_t*) + sizeof(uint32_t)));

    job->nb         = nb;
    job->requests   = (parsec_qpu_request_t*)(job + 1);
    job->inputs     = (uint64_t*)(job->requests + nb);
    job->counts     = (parsec_qpu_result_t**)(job->inputs + nb);
    job->counts_cap = (uint32_t*)(job->counts + nb);
    return job;
}

static void mock_job_free(mock_job_t *job)
{
    for( uint32_t i = 0; i < job->nb; i++ ) free(job->counts[i]);
    free(job);
}

/* The running jobs, as a heap on the time they end. */
static void running_push(mock_backend_t *mock, mock_job_t *job)
{
    uint32_t i = mock->running_n++;
    while( i > 0 ) {
        uint32_t parent = (i - 1) / 2;
        if( mock->running[parent]->t_end <= job->t_end ) break;
        mock->running[i] = mock->running[parent];
        i = parent;
    }
    mock->running[i] = job;
    if( mock->running_n + mock->late_n > mock->stats.max_running ) {
        mock->stats.max_running = mock->running_n + mock->late_n;
    }
}

static mock_job_t *running_pop(mock_backend_t *mock)
{
    mock_job_t *top = mock->running[0], *last = mock->running[--mock->running_n];
    uint32_t i = 0;
    for(;;) {
        uint32_t child = 2 * i + 1;
        if( child >= mock->running_n ) break;
        if( (child + 1 < mock->running_n) && (mock->running[child + 1]->t_end < mock->running[child]->t_end) ) child++;
        if( mock->running[child]->t_end >= last->t_end ) break;
        mock->running[i] = mock->running[child];
        i = child;
    }
    if( mock->running_n > 0 ) mock->running[i] = last;
    return top;
}

static void running_remove(mock_backend_t *mock, mock_job_t *job)
{
    uint32_t left = 0, n = mock->running_n;
    for( uint32_t i = 0; i < n; i++ ) {
        if( mock->running[i] != job ) mock->running[left++] = mock->running[i];
    }
    mock->running_n = 0;
    for( uint32_t i = 0; i < left; i++ ) running_push(mock, mock->running[i]);
}

static void late_remove(mock_backend_t *mock, mock_job_t *job)
{
    for( mock_job_t **link = &mock->late; NULL != *link; link = &(*link)->next ) {
        if( *link != job ) continue;
        *link = job->next;
        job->next = NULL;
        mock->late_n--;
        return;
    }
}

static void event_push(mock_backend_t *mock, parsec_qpu_job_id_t id, int state, int64_t when)
{
    mock_event_t *event = (mock_event_t*)calloc(1, sizeof(mock_event_t));
    event->event.id    = id;
    event->event.state = state;
    event->when        = when;
    if( NULL == mock->events_tail ) mock->events = event;
    else mock->events_tail->next = event;
    mock->events_tail = event;
}

/* Forget the events of a job. Returns how many there were. */
static int event_drop(mock_backend_t *mock, parsec_qpu_job_id_t id)
{
    mock_event_t *prev = NULL, *event = mock->events;
    int dropped = 0;

    while( NULL != event ) {
        mock_event_t *next = event->next;
        if( event->event.id == id ) {
            if( NULL == prev ) mock->events = next;
            else prev->next = next;
            if( mock->events_tail == event ) mock->events_tail = prev;
            free(event);
            dropped++;
        } else {
            prev = event;
        }
        event = next;
    }
    return dropped;
}

/* ------------------------------------------------------------------------
 * The life of a job. The lock is held.
 * --------------------------------------------------------------------- */

static void mock_start(mock_backend_t *mock, mock_job_t *job, int64_t when)
{
    job->state   = MOCK_RUNNING;
    job->t_start = when + job->queue_delay;
    job->t_end   = (job->duration < 0) ? INT64_MAX : job->t_start + job->duration;
    running_push(mock, job);
}

/* A slot is free since `when`: it goes to the head of the queue. */
static void mock_slot_freed(mock_backend_t *mock, int64_t when)
{
    mock_job_t *queued = mock->queue_head;

    if( NULL == queued ) return;
    mock->queue_head = queued->next;
    if( NULL == mock->queue_head ) mock->queue_tail = NULL;
    queued->next = NULL;
    /* The slot can have been free before the queued job arrived, when the
     * job that held it had ended but was not collected yet. */
    mock_start(mock, queued, (when > queued->t_arrival) ? when : queued->t_arrival);
}

/* Tell the caller that a job is over, if it knows the job. A job that
 * failed or was cancelled at the provider is forgotten as it is reported. */
static void mock_report(mock_backend_t *mock, mock_job_t *job, int64_t now)
{
    job->event_due = 0;
    event_push(mock, job->id, job->final, 0);
    if( PARSEC_QPU_MOCK_DUPLICATE == job->fault ) {
        event_push(mock, job->id, job->final, now + MOCK_GHOST_DELAY);
        mock->stats.ghost_events++;
    }
    if( PARSEC_QPU_JOB_DONE != job->final ) {
        if( PARSEC_QPU_JOB_FAILED == job->final ) mock->stats.failures++;
        else mock->stats.provider_cancels++;
        held_remove(mock, job);
        mock_job_free(job);
    }
}

/* A job that held a slot is over, at t_end. */
static void mock_finish(mock_backend_t *mock, mock_job_t *job, int64_t now)
{
    const int64_t t_end = job->t_end;

    job->state = MOCK_FINISHED;
    job->final = (PARSEC_QPU_MOCK_FAIL == job->fault) ? PARSEC_QPU_JOB_FAILED
               : (PARSEC_QPU_MOCK_CANCELLED == job->fault) ? PARSEC_QPU_JOB_CANCELLED : PARSEC_QPU_JOB_DONE;
    if( job->acked ) mock_report(mock, job, now);
    else job->event_due = 1;
    mock_slot_freed(mock, t_end);
}

/* Finish every job whose time is over. One that is still being sampled
 * keeps its slot until a helper is done with it. */
static void mock_collect(mock_backend_t *mock, int64_t now)
{
    while( (mock->running_n > 0) && (mock->running[0]->t_end <= now) ) {
        mock_job_t *job = running_pop(mock);
        if( (WORK_QUEUED == job->work) || (WORK_RUNNING == job->work) ) {
            job->state = MOCK_COMPUTING;
            job->next  = mock->late;
            mock->late = job;
            mock->late_n++;
            continue;
        }
        mock_finish(mock, job, now);
    }
}

/* Hand out the events that are due. */
static int mock_deliver(mock_backend_t *mock, int64_t now, parsec_qpu_event_t *events, int max)
{
    mock_event_t *prev = NULL, *event = mock->events;
    int n = 0;

    while( (NULL != event) && (n < max) ) {
        mock_event_t *next = event->next;
        if( event->when <= now ) {
            events[n++] = event->event;
            if( NULL == prev ) mock->events = next;
            else prev->next = next;
            if( mock->events_tail == event ) mock->events_tail = prev;
            free(event);
        } else {
            prev = event;
        }
        event = next;
    }
    return n;
}

/* The earliest instant at which wait() has something to do, or `limit`. */
static int64_t mock_next_instant(const mock_backend_t *mock, int64_t limit)
{
    if( (mock->running_n > 0) && (mock->running[0]->t_end < limit) ) limit = mock->running[0]->t_end;
    for( const mock_event_t *event = mock->events; NULL != event; event = event->next ) {
        if( event->when < limit ) limit = event->when;
    }
    return limit;
}

/* A digest of what a job reads, to notice that it changed. */
static uint64_t mock_inputs_digest(const parsec_qpu_request_t *request)
{
    const size_t words = parsec_qpu_circuit_size(request->circuit) / sizeof(uint64_t);
    uint64_t h = 0x243f6a8885a308d3ULL, word;

    for( size_t i = 0; i < words; i++ ) {
        memcpy(&word, (const char*)request->circuit + i * sizeof(uint64_t), sizeof(uint64_t));
        h = parsec_qpu_mix64(h ^ word);
    }
    if( NULL != request->params ) {
        for( uint32_t i = 0; i < request->circuit->nb_params; i++ ) {
            memcpy(&word, &request->params[i], sizeof(uint64_t));
            h = parsec_qpu_mix64(h ^ word);
        }
    }
    return h;
}

/* ------------------------------------------------------------------------
 * Helpers: they sample the circuits of the jobs the mock holds.
 * --------------------------------------------------------------------- */

static void *mock_helper_main(void *arg)
{
    mock_backend_t *mock = (mock_backend_t*)arg;
    parsec_qpu_sampler_t *sampler = parsec_qpu_sampler_new();

    pthread_mutex_lock(&mock->lock);
    for(;;) {
        mock_job_t *job;
        uint32_t started = 0;
        int rc = PARSEC_SUCCESS;

        while( !mock->helpers_stop && (NULL == mock->work_head) ) {
            pthread_cond_wait(&mock->work_cond, &mock->lock);
        }
        if( mock->helpers_stop ) break;

        job = mock->work_head;
        mock->work_head = job->work_next;
        if( NULL == mock->work_head ) mock->work_tail = NULL;
        job->work_next = NULL;
        job->work      = WORK_RUNNING;
        pthread_mutex_unlock(&mock->lock);

        /* The job cannot go away meanwhile: cancel() waits for this thread.
         * Its circuits one after the other, up to the first that does not
         * go through: the job then has no answer at all. */
        for( uint32_t i = 0; (i < job->nb) && (PARSEC_SUCCESS == rc); i++ ) {
            const parsec_qpu_request_t *request = &job->requests[i];

            started++;
            rc = parsec_qpu_sampler_run(sampler, request->circuit, request->params,
                                        request->shots, mock->seed, request->key,
                                        job->counts[i], job->counts_cap[i], &job->cancel);
        }

        pthread_mutex_lock(&mock->lock);
        job->work_rc = rc;
        job->work    = WORK_DONE;
        mock->stats.computed += started;
        if( job->cancel ) {
            pthread_cond_broadcast(&mock->idle_cond);
        } else if( MOCK_COMPUTING == job->state ) {
            /* Its time was over already: it ends now. */
            const int64_t now = parsec_qpu_now_ns();
            late_remove(mock, job);
            mock->stats.late++;
            job->t_end = now;
            mock_finish(mock, job, now);
            pthread_cond_signal(&mock->cond);
        }
    }
    pthread_mutex_unlock(&mock->lock);

    parsec_qpu_sampler_free(sampler);
    return NULL;
}

/* ------------------------------------------------------------------------
 * The backend interface
 * --------------------------------------------------------------------- */

static void mock_destroy(mock_backend_t *mock)
{
    mock_event_t *event;

    for( uint32_t i = 0; i < mock->held_n; i++ ) mock_job_free(mock->held[i]);
    while( NULL != (event = mock->events) ) { mock->events = event->next; free(event); }
    free(mock->held);
    free(mock->running);
    free(mock->tags);
    parsec_qpu_delay_fini(&mock->delay);
    parsec_qpu_delay_fini(&mock->job_delay);
    parsec_qpu_delay_fini(&mock->queue_delay);
    pthread_cond_destroy(&mock->cond);
    pthread_cond_destroy(&mock->work_cond);
    pthread_cond_destroy(&mock->idle_cond);
    pthread_mutex_destroy(&mock->lock);
    free(mock);
}

static int mock_close(parsec_qpu_backend_t *backend);

static int mock_open(parsec_qpu_backend_t **backend, int device_index)
{
    mock_backend_t *mock = (mock_backend_t*)calloc(1, sizeof(mock_backend_t));
    /* What the entry of this QPU says in device_qpu_devices, and what the
     * parameters of the mock say for what it leaves out. The seeds are those
     * of the run: what an evaluation gives does not depend on which QPU ran
     * it. */
    const int q = device_index;
    const int slots        = parsec_qpu_list_option_int(q, "slots", parsec_qpu_params.mock_slots);
    const int max_qubits   = parsec_qpu_list_option_int(q, "max_qubits", parsec_qpu_params.mock_max_qubits);
    const int max_circuits = parsec_qpu_list_option_int(q, "max_circuits", parsec_qpu_params.mock_max_circuits);
    const int threads      = parsec_qpu_list_option_int(q, "threads", parsec_qpu_params.mock_threads);
    const int submit_delay = parsec_qpu_list_option_int(q, "submit_delay", parsec_qpu_params.mock_submit_delay);
    const char *delay       = parsec_qpu_list_option_string(q, "delay", parsec_qpu_params.mock_delay);
    const char *queue_delay = parsec_qpu_list_option_string(q, "queue_delay", parsec_qpu_params.mock_queue_delay);
    const char *job_delay   = parsec_qpu_list_option_string(q, "job_delay", parsec_qpu_params.mock_job_delay);
    const char *faults      = parsec_qpu_list_option_string(q, "faults", parsec_qpu_params.mock_faults);
    pthread_condattr_t attr;
    int nb_helpers;

    mock->super.ops  = &parsec_qpu_backend_mock_ops;
    mock->number     = q;
    mock->seed       = (uint64_t)parsec_qpu_params.mock_seed;
    mock->delay_seed = (0 != parsec_qpu_params.mock_delay_seed) ? (uint64_t)parsec_qpu_params.mock_delay_seed : mock->seed;
    mock->slots      = (uint32_t)(slots > 0 ? slots : 1);
    mock->max_qubits = (uint32_t)(max_qubits > 0 ? max_qubits : 1);
    if( mock->max_qubits > PARSEC_QPU_SAMPLER_MAX_QUBITS ) mock->max_qubits = PARSEC_QPU_SAMPLER_MAX_QUBITS;
    mock->max_circuits = (uint32_t)(max_circuits > 0 ? max_circuits : 1);
    mock->submit_delay = (int64_t)(submit_delay > 0 ? submit_delay : 0) * 1000LL;
    mock->running    = (mock_job_t**)calloc(mock->slots, sizeof(mock_job_t*));
    mock->next_id    = 1;
    mock->t_open     = parsec_qpu_now_ns();
    mock->t_last_call   = -1;
    mock->stats.min_submit_gap = -1;

    pthread_mutex_init(&mock->lock, NULL);
    /* Deadlines are on the monotonic clock, so the condition has to be too. */
    pthread_condattr_init(&attr);
    pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    pthread_cond_init(&mock->cond, &attr);
    pthread_condattr_destroy(&attr);
    pthread_cond_init(&mock->work_cond, NULL);
    pthread_cond_init(&mock->idle_cond, NULL);

    if( PARSEC_SUCCESS != parsec_qpu_delay_parse(&mock->delay, delay) ) {
        parsec_warning("QPU mock: device_qpu_mock_delay, or delay= for QPU %d, is '%s'; expected const:<us>, "
                       "uniform:<min us>:<max us>, exp:<mean us>, lognormal:<median us>:<sigma> or "
                       "replay:<readable file of delays in us>", q, (NULL != delay) ? delay : "");
        goto bad;
    }
    if( PARSEC_SUCCESS != parsec_qpu_delay_parse(&mock->queue_delay, (NULL != queue_delay) ? queue_delay : "const:0") ) {
        parsec_warning("QPU mock: device_qpu_mock_queue_delay, or queue_delay= for QPU %d, is '%s'; it takes what "
                       "device_qpu_mock_delay takes", q, queue_delay);
        goto bad;
    }
    if( PARSEC_SUCCESS != parsec_qpu_delay_parse(&mock->job_delay, (NULL != job_delay) ? job_delay : "const:0") ) {
        parsec_warning("QPU mock: device_qpu_mock_job_delay, or job_delay= for QPU %d, is '%s'; it takes what "
                       "device_qpu_mock_delay takes", q, job_delay);
        goto bad;
    }
    if( PARSEC_SUCCESS != mock_parse_faults(mock, faults) ) {
        parsec_warning("QPU mock: device_qpu_mock_faults, or faults= for QPU %d, is '%s'; expected none, or a list "
                       "of <fault>:<rule>=<value> with fault one of fail, hang, unavailable, lost, reject, noresult, "
                       "duplicate, late, cancelled and rule one of p, first, every, key, for", q, faults);
        goto bad;
    }

    nb_helpers = (threads > 0) ? threads : 1;
    if( nb_helpers > MOCK_MAX_HELPERS ) nb_helpers = MOCK_MAX_HELPERS;
    for( mock->nb_helpers = 0; mock->nb_helpers < nb_helpers; mock->nb_helpers++ ) {
        if( 0 != pthread_create(&mock->helpers[mock->nb_helpers], NULL, mock_helper_main, mock) ) break;
    }
    if( 0 == mock->nb_helpers ) {
        parsec_warning("QPU mock: could not create a helper thread");
        goto bad;
    }

    pthread_mutex_lock(&mock_current_lock);
    if( MOCK_MAX_OPEN == mock_opened_n ) {
        pthread_mutex_unlock(&mock_current_lock);
        parsec_warning("QPU mock: too many mocks are open in this process");
        /* Its helpers are there by now: go the way of a mock that is closed. */
        return (PARSEC_SUCCESS == mock_close(&mock->super)) ? PARSEC_ERR_OUT_OF_RESOURCE : PARSEC_ERROR;
    }
    mock_opened[mock_opened_n++] = mock;
    mock_opens++;
    pthread_mutex_unlock(&mock_current_lock);

    *backend = &mock->super;
    return PARSEC_SUCCESS;

  bad:
    mock_destroy(mock);
    return PARSEC_ERR_BAD_PARAM;
}

static int mock_capabilities(parsec_qpu_backend_t *backend, parsec_qpu_caps_t *caps)
{
    mock_backend_t *mock = (mock_backend_t*)backend;

    caps->nb_qubits            = mock->max_qubits;
    caps->max_shots            = 1u << 20;
    caps->max_jobs             = 0;    /* it queues what it cannot run */
    caps->max_circuits_per_job = mock->max_circuits;
    caps->is_physical          = 0;
    return PARSEC_SUCCESS;
}

static int mock_submit(parsec_qpu_backend_t *backend, const parsec_qpu_request_t *requests, int nb,
                       uint64_t tag, parsec_qpu_job_id_t *id)
{
    mock_backend_t *mock = (mock_backend_t*)backend;
    parsec_qpu_mock_fault_t fault = PARSEC_QPU_MOCK_OK;
    mock_job_t *job = NULL;
    uint64_t call;
    int64_t now;
    int never_ends, mixed = 0;

    if( (nb < 1) || ((uint32_t)nb > mock->max_circuits) ) return PARSEC_ERR_NOT_SUPPORTED;
    for( int i = 0; i < nb; i++ ) {
        if( (requests[i].circuit->nb_qubits > mock->max_qubits) || (0 == requests[i].shots) ) return PARSEC_ERR_BAD_PARAM;
    }
    if( mock->submit_delay > 0 ) {
        /* The round trip to a provider: the caller is held, and nothing of
         * the mock is. */
        struct timespec trip = { (time_t)(mock->submit_delay / 1000000000LL), (long)(mock->submit_delay % 1000000000LL) };
        nanosleep(&trip, NULL);
    }

    pthread_mutex_lock(&mock->lock);
    now  = parsec_qpu_now_ns();
    call = ++mock->stats.submit_calls;
    if( mock->t_last_call >= 0 ) {
        const int64_t gap = now - mock->t_last_call;
        if( (mock->stats.min_submit_gap < 0) || (gap < mock->stats.min_submit_gap) ) mock->stats.min_submit_gap = gap;
    }
    mock->t_last_call = now;

    /* What becomes of this submission is decided first: a provider that is
     * out of reach cannot say whether it has the job either. A job has one
     * fate: the first that is not "ok" among those of its circuits. */
    for( int i = 0; (i < nb) && (PARSEC_QPU_MOCK_OK == fault); i++ ) {
        fault = mock_decide(mock, tag, &requests[i], call, now);
    }
    if( PARSEC_QPU_MOCK_UNAVAILABLE == fault ) {
        mock->stats.deferred++;
        pthread_mutex_unlock(&mock->lock);
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    /* A job that is here already under this tag is the answer. */
    for( uint32_t i = 0; i < mock->held_n; i++ ) {
        if( mock->held[i]->tag == tag ) { job = mock->held[i]; break; }
    }
    if( NULL != job ) {
        /* Asking again is for the job that was asked for: the same circuits,
         * in the same order. */
        int same = ((uint32_t)nb == job->nb);
        for( int i = 0; same && (i < nb); i++ ) {
            same = (requests[i].key == job->requests[i].key) && (requests[i].circuit == job->requests[i].circuit);
        }
        if( !same ) mock->stats.changed_jobs++;
        if( job->acked ) {
            /* The caller knows this job and asks for it again. It gets the
             * same answer, and this is counted: it has no reason to ask. */
            mock->stats.duplicate_tags++;
            *id = job->id;
            pthread_mutex_unlock(&mock->lock);
            return PARSEC_SUCCESS;
        }
        if( PARSEC_QPU_MOCK_LOST == fault ) {
            /* The answer is lost once more. */
            mock->stats.deferred++;
            pthread_mutex_unlock(&mock->lock);
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        job->acked = 1;
        mock->stats.tag_hits++;
        *id = job->id;
        /* It may be over already; the caller could not have been told.
         * (Reporting a failed job forgets it: nothing reads it after this.) */
        if( job->event_due ) mock_report(mock, job, now);
        pthread_mutex_unlock(&mock->lock);
        return PARSEC_SUCCESS;
    }
    if( tags_has(mock, tag) ) {
        /* A job had this tag, and is gone: its results were taken, or it
         * was cancelled. Running it again would be the duplicate the tags
         * are there to prevent. */
        mock->stats.duplicate_tags++;
        pthread_mutex_unlock(&mock->lock);
        return PARSEC_ERR_EXISTS;
    }
    if( PARSEC_QPU_MOCK_REJECT == fault ) {
        mock->stats.rejected++;
        pthread_mutex_unlock(&mock->lock);
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    never_ends = (PARSEC_QPU_MOCK_HANG == fault) || (PARSEC_QPU_MOCK_LATE == fault);
    job = mock_job_new((uint32_t)nb);
    job->id          = mock->next_id++;
    job->tag         = tag;
    job->fault       = fault;
    job->t_arrival   = now;
    /* What is drawn for the job as a whole comes from its first circuit. */
    job->queue_delay = parsec_qpu_delay_draw(&mock->queue_delay, mock_time_bits(mock, requests[0].key, requests[0].attempt,
                                                                                 PARSEC_QPU_MOCK_TIME_QUEUE));
    job->duration    = parsec_qpu_delay_draw(&mock->job_delay, mock_time_bits(mock, requests[0].key, requests[0].attempt,
                                                                               PARSEC_QPU_MOCK_TIME_JOB));
    for( int i = 0; i < nb; i++ ) {
        const parsec_qpu_request_t *request = &requests[i];
        int64_t runs = 0;
        int seen = 0;

        job->requests[i] = *request;
        job->inputs[i]   = mock_inputs_digest(request);
        if( never_ends ) {
            /* Nothing is drawn for a circuit that will not run its time. */
        } else if( NULL != mock_delay_fn ) {
            runs = mock_delay_fn(tag, request, mock_delay_arg);
            if( runs < 0 ) runs = 0;
        } else {
            runs = parsec_qpu_delay_draw(&mock->delay, mock_time_bits(mock, request->key, request->attempt, PARSEC_QPU_MOCK_TIME_CIRCUIT));
        }
        job->duration += runs;

        /* No two circuits of one evaluation at the same time, in two jobs
         * or in this one. */
        for( int j = 0; (j < i) && !seen; j++ ) {
            seen = (requests[j].key == request->key);
        }
        for( uint32_t h = 0; (h < mock->held_n) && !seen; h++ ) {
            for( uint32_t j = 0; (j < mock->held[h]->nb) && !seen; j++ ) {
                seen = (mock->held[h]->requests[j].key == request->key);
            }
        }
        if( seen ) mock->stats.key_overlaps++;
        if( request->shots != requests[0].shots ) mixed = 1;
    }
    if( mixed ) mock->stats.mixed_shots++;
    if( never_ends ) job->duration = -1;

    tags_add(mock, tag);
    held_add(mock, job);
    mock->stats.submits++;
    mock->stats.circuits += (uint64_t)nb;
    if( (uint32_t)nb > mock->stats.max_circuits ) mock->stats.max_circuits = (uint32_t)nb;

    /* Only a job that will end well needs its counts. */
    if( (PARSEC_QPU_MOCK_OK == fault) || (PARSEC_QPU_MOCK_LOST == fault) || (PARSEC_QPU_MOCK_DUPLICATE == fault) ) {
        for( int i = 0; i < nb; i++ ) {
            job->counts_cap[i] = (requests[i].max_entries < requests[i].shots) ? requests[i].max_entries : requests[i].shots;
            job->counts[i]     = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(job->counts_cap[i]));
        }
        job->work = WORK_QUEUED;
        if( NULL == mock->work_tail ) mock->work_head = job;
        else mock->work_tail->work_next = job;
        mock->work_tail = job;
        pthread_cond_signal(&mock->work_cond);
    }

    if( mock->running_n + mock->late_n < mock->slots ) {
        mock_start(mock, job, now);
    } else {
        job->state = MOCK_QUEUED;
        if( NULL == mock->queue_tail ) mock->queue_head = job;
        else mock->queue_tail->next = job;
        mock->queue_tail = job;
    }

    if( PARSEC_QPU_MOCK_LOST == fault ) {
        /* The job is in, and the caller is told it may not be. */
        mock->stats.deferred++;
        pthread_mutex_unlock(&mock->lock);
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    job->acked = 1;
    *id = job->id;
    pthread_mutex_unlock(&mock->lock);
    return PARSEC_SUCCESS;
}

static int mock_wait(parsec_qpu_backend_t *backend, int64_t timeout_ns,
                     parsec_qpu_event_t *events, int max_events)
{
    mock_backend_t *mock = (mock_backend_t*)backend;
    int64_t now = parsec_qpu_now_ns();
    int64_t give_up = (timeout_ns < 0) ? INT64_MAX : now + timeout_ns;
    int n;

    pthread_mutex_lock(&mock->lock);
    for(;;) {
        mock_collect(mock, now);
        n = mock_deliver(mock, now, events, max_events);
        if( n > 0 ) break;
        if( now >= give_up ) {
            /* Out of time. A wake-up that came meanwhile is the caller's too. */
            PARSEC_QPU_STORE(&mock->wake, 0);
            break;
        }

        /* Sleep until something is due, unless woken or out of time first.
         * Announce the sleep, then look for a wake-up: see the structure. */
        PARSEC_QPU_STORE(&mock->sleeping, 1);
        if( PARSEC_QPU_LOAD(&mock->wake) ) {     /* somebody wants the caller back */
            PARSEC_QPU_STORE(&mock->sleeping, 0);
            PARSEC_QPU_STORE(&mock->wake, 0);
            break;
        }
        int64_t until = mock_next_instant(mock, give_up);
        if( INT64_MAX == until ) {
            pthread_cond_wait(&mock->cond, &mock->lock);
        } else {
            struct timespec ts;
            ts.tv_sec  = (time_t)(until / 1000000000LL);
            ts.tv_nsec = (long)(until % 1000000000LL);
            (void)pthread_cond_timedwait(&mock->cond, &mock->lock, &ts);
        }
        PARSEC_QPU_STORE(&mock->sleeping, 0);
        now = parsec_qpu_now_ns();
    }
    pthread_mutex_unlock(&mock->lock);
    return n;
}

static int mock_wakeup(parsec_qpu_backend_t *backend)
{
    mock_backend_t *mock = (mock_backend_t*)backend;

    PARSEC_QPU_STORE(&mock->wake, 1);
    if( PARSEC_QPU_LOAD(&mock->sleeping) ) {
        /* It sleeps, or has announced it will and is on its way to: either
         * way it is not holding the lock for longer than that. */
        pthread_mutex_lock(&mock->lock);
        pthread_cond_signal(&mock->cond);
        pthread_mutex_unlock(&mock->lock);
    }
    /* Otherwise it is at work, and looks at `wake` before it sleeps again. */
    return PARSEC_SUCCESS;
}

static int mock_results(parsec_qpu_backend_t *backend, parsec_qpu_job_id_t id,
                        parsec_qpu_result_t **results, int nb)
{
    mock_backend_t *mock = (mock_backend_t*)backend;
    mock_job_t *job;
    int rc = PARSEC_SUCCESS;

    pthread_mutex_lock(&mock->lock);
    job = held_find(mock, id);
    if( (NULL == job) || (MOCK_FINISHED != job->state) || (PARSEC_QPU_JOB_DONE != job->final) || !job->acked ) {
        pthread_mutex_unlock(&mock->lock);
        return PARSEC_ERR_NOT_FOUND;
    }
    held_remove(mock, job);
    /* The circuits and their parameters have to have stayed in place and
     * unchanged for as long as the job was here. */
    for( uint32_t i = 0; i < job->nb; i++ ) {
        if( mock_inputs_digest(&job->requests[i]) != job->inputs[i] ) mock->stats.inputs_changed++;
    }
    if( (uint32_t)nb != job->nb ) {
        /* Not the job the caller takes it for. It is forgotten all the same. */
        rc = PARSEC_ERR_BAD_PARAM;
    } else if( (PARSEC_QPU_MOCK_NORESULT == job->fault) || (NULL == job->counts[0]) || (PARSEC_SUCCESS != job->work_rc) ) {
        rc = PARSEC_ERROR;
    } else {
        mock->stats.results += job->nb;
    }
    pthread_mutex_unlock(&mock->lock);

    for( uint32_t i = 0; (PARSEC_SUCCESS == rc) && (i < job->nb); i++ ) {
        const parsec_qpu_result_t *counts = job->counts[i];
        parsec_qpu_result_t *result = results[i];

        result->status         = PARSEC_QPU_STATUS_OK;
        result->flags          = (uint8_t)(PARSEC_QPU_RESULT_MOCK | (counts->flags & PARSEC_QPU_RESULT_TRUNCATED));
        result->nb_qubits      = counts->nb_qubits;
        result->shots_returned = counts->shots_returned;
        result->nb_entries     = counts->nb_entries;
        /* The circuits of a job share its time at the backend. */
        result->t_start        = job->t_start;
        result->t_end          = job->t_end;
        memcpy(result + 1, counts + 1, (size_t)counts->nb_entries * PARSEC_QPU_ENTRY_SIZE);
    }
    mock_job_free(job);
    return rc;
}

/* Take a job out of everything the mock keeps, and wait for a helper that is
 * at it to let go: when this returns nothing reads the job's buffers and the
 * job can be freed. The lock is held, and is given up while waiting. */
static void mock_drop(mock_backend_t *mock, mock_job_t *job, int64_t now)
{
    switch( job->state ) {
    case MOCK_QUEUED:
        for( mock_job_t *prev = NULL, *cur = mock->queue_head; NULL != cur; prev = cur, cur = cur->next ) {
            if( cur != job ) continue;
            if( NULL == prev ) mock->queue_head = cur->next;
            else prev->next = cur->next;
            if( mock->queue_tail == cur ) mock->queue_tail = prev;
            break;
        }
        break;
    case MOCK_RUNNING:
        running_remove(mock, job);
        mock_slot_freed(mock, now);
        break;
    case MOCK_COMPUTING:
        late_remove(mock, job);
        mock_slot_freed(mock, now);
        break;
    case MOCK_FINISHED:
        break;
    }
    job->next = NULL;
    held_remove(mock, job);
    (void)event_drop(mock, job->id);

    if( WORK_QUEUED == job->work ) {
        for( mock_job_t *prev = NULL, *cur = mock->work_head; NULL != cur; prev = cur, cur = cur->work_next ) {
            if( cur != job ) continue;
            if( NULL == prev ) mock->work_head = cur->work_next;
            else prev->work_next = cur->work_next;
            if( mock->work_tail == cur ) mock->work_tail = prev;
            break;
        }
    } else if( WORK_RUNNING == job->work ) {
        PARSEC_QPU_STORE(&job->cancel, 1);
        while( WORK_RUNNING == job->work ) pthread_cond_wait(&mock->idle_cond, &mock->lock);
    }
}

static int mock_cancel(parsec_qpu_backend_t *backend, parsec_qpu_job_id_t id)
{
    mock_backend_t *mock = (mock_backend_t*)backend;
    mock_job_t *job;
    int64_t now;

    pthread_mutex_lock(&mock->lock);
    if( NULL == (job = held_find(mock, id)) ) {
        /* Gone already. If it failed and that was not reported yet, it will
         * not be: the caller gave up on it first. */
        const int dropped = event_drop(mock, id);
        if( dropped > 0 ) mock->stats.cancels++;
        pthread_mutex_unlock(&mock->lock);
        return (dropped > 0) ? PARSEC_SUCCESS : PARSEC_ERR_NOT_FOUND;
    }
    now = parsec_qpu_now_ns();
    mock_drop(mock, job, now);

    if( PARSEC_QPU_MOCK_LATE == job->fault ) {
        /* A provider that reports the end of a job it was told to drop. */
        event_push(mock, id, PARSEC_QPU_JOB_DONE, now + MOCK_GHOST_DELAY);
        mock->stats.ghost_events++;
    }
    mock->stats.cancels++;
    pthread_mutex_unlock(&mock->lock);

    mock_job_free(job);
    return PARSEC_SUCCESS;
}

static int mock_abandon(parsec_qpu_backend_t *backend, uint64_t tag)
{
    mock_backend_t *mock = (mock_backend_t*)backend;
    mock_job_t *job = NULL;

    pthread_mutex_lock(&mock->lock);
    for( uint32_t i = 0; i < mock->held_n; i++ ) {
        /* Only a job the caller was never told of: one it knows, it cancels. */
        if( (mock->held[i]->tag == tag) && !mock->held[i]->acked ) { job = mock->held[i]; break; }
    }
    if( NULL != job ) {
        mock_drop(mock, job, parsec_qpu_now_ns());
        mock->stats.abandoned++;
    }
    pthread_mutex_unlock(&mock->lock);

    if( NULL != job ) mock_job_free(job);
    return PARSEC_SUCCESS;
}

static int mock_close(parsec_qpu_backend_t *backend)
{
    mock_backend_t *mock = (mock_backend_t*)backend;

    pthread_mutex_lock(&mock_current_lock);
    for( int i = 0; i < mock_opened_n; i++ ) {
        if( mock_opened[i] != mock ) continue;
        mock_opened[i] = mock_opened[--mock_opened_n];
        break;
    }
    pthread_mutex_unlock(&mock_current_lock);

    /* Whatever a helper is at, it gives up at the next operation. */
    pthread_mutex_lock(&mock->lock);
    mock->helpers_stop = 1;
    for( uint32_t i = 0; i < mock->held_n; i++ ) PARSEC_QPU_STORE(&mock->held[i]->cancel, 1);
    pthread_cond_broadcast(&mock->work_cond);
    pthread_mutex_unlock(&mock->lock);
    for( int i = 0; i < mock->nb_helpers; i++ ) pthread_join(mock->helpers[i], NULL);

    mock_destroy(mock);
    return PARSEC_SUCCESS;
}

const parsec_qpu_backend_ops_t parsec_qpu_backend_mock_ops = {
    .name         = "mock",
    .physical     = 0,
    .open         = mock_open,
    .capabilities = mock_capabilities,
    .submit       = mock_submit,
    .wait         = mock_wait,
    .wakeup       = mock_wakeup,
    .results      = mock_results,
    .cancel       = mock_cancel,
    .abandon      = mock_abandon,
    .close        = mock_close
};
