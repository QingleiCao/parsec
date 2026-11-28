/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * Baseline B: the block-decomposed QAOA as a hand-written MPI program,
 * without the runtime.
 *
 * Block b lives on rank b mod P; rank 0 owns the QPU backend (the mock, the
 * same one the runtime uses, opened here directly). The algorithm goes
 * colour by colour: in each colour every rank drives its blocks of that
 * colour as independent state machines, with both evaluations of a step in
 * flight together, and at the end of the colour the ranks exchange the
 * states of the blocks they updated. This is the bulk-synchronous structure
 * of a multicolour Gauss-Seidel code, and the schedule C-sync runs on the
 * runtime.
 *
 * A job goes to rank 0 as a message (the circuit, its angles, its key), and
 * comes back as a message (the result). On rank 0 a helper thread owns the
 * backend: it submits under the same limits the QPU device has (at most
 * device_qpu_max_outstanding jobs at the backend, device_qpu_submit_interval
 * between two submissions), first come first served, and sends each result
 * back as its job ends. It does not put several jobs into one: B does not
 * coalesce. A job that does not end well is an evaluation that failed; there
 * is no retry.
 *
 * Every rank calls the kernels of qaoa_core.c in the order the sequential
 * reference calls them per block, and every job is named by the key the
 * reference gives it, so the final states are the reference's byte for byte;
 * with -c 1 (the default) rank 0 checks that they are.
 *
 * Options, before "--":
 *   -i NAME | -g [TOPO:]K:N:C:X  the instance, as testing_qaoa takes it
 *   -p LAYERS  -T STEPS  -R ROUNDS  -s SHOTS  -P PATIENCE  -S SEED
 *   -c 0        do not run the reference, compare nothing
 *   -L PREFIX   write the timestamps of every job (rank 0) and every update
 *               (each rank) to PREFIX.<rank>.jobs and PREFIX.<rank>.updates,
 *               in the format of testing_qaoa
 *   -M          print one line that a script can read
 *   -v          verbose
 * After "--", the parameters of the QPU device and of its mock as the
 * runtime takes them, so that a script gives B and the runtime the same:
 *   --mca device_qpu_<name> <value>   for max_outstanding, submit_interval,
 *   mock_seed, mock_delay_seed, mock_slots, mock_delay, mock_queue_delay,
 *   mock_job_delay, mock_threads, mock_max_qubits; enabled and max_batch are
 *   taken and ignored (B has no device and does not coalesce).
 *
 * Needs MPI_THREAD_MULTIPLE: the helper thread of rank 0 sends results while
 * the main thread waits for messages.
 */

#include "qaoa_core.h"
#include "parsec/constants.h"

#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#define TAG_JOB         7
#define TAG_RESULT(b, e) (1000 + 2 * (b) + (e))
#define MAX_EVENTS      64

static int verbose = 0;

/* ------------------------------------------------------------------------
 * Messages
 * --------------------------------------------------------------------- */

/* A job, as it travels to rank 0: this head, the circuit, its angles. */
typedef struct job_head_s {
    int32_t  src;
    int32_t  block;
    int32_t  e;
    uint32_t shots;
    uint64_t key;
    uint32_t max_entries;
    uint32_t circuit_bytes;
    uint32_t nb_params;
    uint32_t pad;
} job_head_t;

/* ------------------------------------------------------------------------
 * Rank 0: the server of the QPU
 * --------------------------------------------------------------------- */

typedef struct job_s {
    struct job_s         *next;
    job_head_t            head;
    parsec_qpu_circuit_t *circuit;
    double               *params;
    parsec_qpu_result_t  *result;
    parsec_qpu_job_id_t   id;
    int64_t               t_received, t_submit, t_sent;
} job_t;

/* What the log keeps of one job, on rank 0. */
typedef struct job_log_s {
    uint64_t key;
    int32_t  block, e;
    int64_t  t_received, t_submit, t_start, t_end, t_retrieve, t_sent;
    uint32_t status;
} job_log_t;

typedef struct server_s {
    parsec_qpu_backend_t *backend;
    MPI_Comm              comm;
    pthread_t             thread;
    pthread_mutex_t       lock;          /* protects queue and stop */
    job_t                *queue_head, *queue_tail;
    int                   stop;
    uint32_t              max_outstanding;
    int64_t               interval_ns;
    /* The helper thread's own. */
    job_t                *pending_head, *pending_tail;
    job_t               **in_flight;
    uint32_t              in_flight_n;
    int64_t               t_next;
    long                  jobs, refused, failed;   /* taken by the backend; refused; taken and failed */
    uint32_t              max_in_flight;
    job_log_t            *log;
    long                  log_n, log_cap;
} server_t;

static void server_enqueue(server_t *srv, job_t *job)
{
    pthread_mutex_lock(&srv->lock);
    job->next = NULL;
    if( NULL == srv->queue_tail ) srv->queue_head = job;
    else srv->queue_tail->next = job;
    srv->queue_tail = job;
    pthread_mutex_unlock(&srv->lock);
    (void)srv->backend->ops->wakeup(srv->backend);
}

/* Send the result of a job to the rank of its block, and forget the job. */
static void server_reply(server_t *srv, job_t *job, int64_t t_start, int64_t t_end, int64_t t_retrieve)
{
    const size_t bytes = parsec_qpu_result_size(job->result->nb_entries);

    job->t_sent = parsec_qpu_now_ns();
    MPI_Send(job->result, (int)bytes, MPI_BYTE, job->head.src, TAG_RESULT(job->head.block, job->head.e), srv->comm);
    if( NULL != srv->log ) {
        if( srv->log_n == srv->log_cap ) {
            srv->log_cap = (0 == srv->log_cap) ? 1024 : 2 * srv->log_cap;
            srv->log = (job_log_t*)realloc(srv->log, (size_t)srv->log_cap * sizeof(job_log_t));
        }
        srv->log[srv->log_n++] = (job_log_t){ job->head.key, job->head.block, job->head.e, job->t_received, job->t_submit,
                                              t_start, t_end, t_retrieve, job->t_sent, job->result->status };
    }
    free(job->circuit);
    free(job->params);
    free(job->result);
    free(job);
}

static void *server_main(void *arg)
{
    server_t *srv = (server_t*)arg;
    const parsec_qpu_backend_ops_t *ops = srv->backend->ops;
    parsec_qpu_event_t events[MAX_EVENTS];

    for(;;) {
        int64_t now = parsec_qpu_now_ns(), timeout;
        int stop, n;

        /* Take what the main thread received. */
        pthread_mutex_lock(&srv->lock);
        if( NULL != srv->queue_head ) {
            if( NULL == srv->pending_tail ) srv->pending_head = srv->queue_head;
            else srv->pending_tail->next = srv->queue_head;
            srv->pending_tail = srv->queue_tail;
            srv->queue_head = srv->queue_tail = NULL;
        }
        stop = srv->stop;
        pthread_mutex_unlock(&srv->lock);
        if( stop && (NULL == srv->pending_head) && (0 == srv->in_flight_n) ) break;

        /* Submit, first come first served, as the limits allow. */
        while( (NULL != srv->pending_head) && (srv->in_flight_n < srv->max_outstanding) && (now >= srv->t_next) ) {
            job_t *job = srv->pending_head;
            parsec_qpu_request_t request;

            srv->pending_head = job->next;
            if( NULL == srv->pending_head ) srv->pending_tail = NULL;
            memset(&request, 0, sizeof(request));
            request.circuit     = job->circuit;
            request.params      = job->params;
            request.shots       = job->head.shots;
            request.max_entries = job->head.max_entries;
            request.key         = job->head.key;
            request.attempt     = 1;
            job->t_submit = now;
            memset(job->result, 0, sizeof(parsec_qpu_result_t));
            if( PARSEC_SUCCESS != ops->submit(srv->backend, &request, 1, job->head.key + 1, &job->id) ) {
                /* Not taken: an evaluation that failed. */
                job->result->status = PARSEC_QPU_STATUS_FAILED;
                srv->refused++;
                server_reply(srv, job, 0, 0, parsec_qpu_now_ns());
            } else {
                srv->in_flight[srv->in_flight_n++] = job;
                srv->jobs++;
                if( srv->in_flight_n > srv->max_in_flight ) srv->max_in_flight = srv->in_flight_n;
            }
            now = parsec_qpu_now_ns();
            srv->t_next = now + srv->interval_ns;
        }

        /* Wait for the end of a job, a new job, or the moment a submission
         * is allowed again. */
        timeout = 1000000000LL;
        if( (NULL != srv->pending_head) && (srv->in_flight_n < srv->max_outstanding) && (srv->t_next > now) ) {
            timeout = srv->t_next - now;
        }
        n = ops->wait(srv->backend, timeout, events, MAX_EVENTS);
        if( n < 0 ) {
            fprintf(stderr, "qaoa_mpi: the backend failed while waiting (%d)\n", n);
            MPI_Abort(srv->comm, 1);
        }
        for( int i = 0; i < n; i++ ) {
            job_t *job = NULL;
            for( uint32_t j = 0; j < srv->in_flight_n; j++ ) {
                if( srv->in_flight[j]->id != events[i].id ) continue;
                job = srv->in_flight[j];
                srv->in_flight[j] = srv->in_flight[--srv->in_flight_n];
                break;
            }
            if( NULL == job ) continue;
            if( (PARSEC_QPU_JOB_DONE != events[i].state) ||
                (PARSEC_SUCCESS != ops->results(srv->backend, job->id, &job->result, 1)) ) {
                memset(job->result, 0, sizeof(parsec_qpu_result_t));
                job->result->status = PARSEC_QPU_STATUS_FAILED;
                srv->failed++;
            }
            job->result->version         = PARSEC_QPU_RESULT_VERSION;
            job->result->shots_requested = job->head.shots;
            job->result->attempts        = 1;
            job->result->batch           = 1;
            job->result->t_submit        = job->t_submit;
            job->result->t_retrieve      = parsec_qpu_now_ns();
            server_reply(srv, job, job->result->t_start, job->result->t_end, job->result->t_retrieve);
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------------
 * Every rank: its blocks
 * --------------------------------------------------------------------- */

/* When an update began and ended, on the rank of its block. */
typedef struct update_log_s {
    int64_t t_build, t_update;
} update_log_t;

typedef struct block_s {
    int                   b;
    qaoa_local_t         *P;
    parsec_qpu_circuit_t *circuit;
    qaoa_opt_t            S;
    double                theta[2][2 * QAOA_MAX_LAYERS];
    qaoa_eval_t           F[2];
    parsec_qpu_result_t  *result[2];
    int                   t, waiting;     /* step, results still expected */
    int                   done;
} block_t;

typedef struct run_s {
    const qaoa_instance_t *inst;
    const qaoa_config_t   *cfg;
    MPI_Comm               comm;
    int                    rank, world;
    qaoa_state_t          *x;             /* [K] every block: the own ones current, the others as of the last exchange */
    qaoa_boundary_t       *h;
    size_t                 circuit_room, job_bytes;
    uint32_t               max_entries;
    block_t               *blocks;        /* [K], used for the blocks of this rank */
    MPI_Request           *recv;          /* [2 + 2K]: a job (rank 0), the exchange, the results of block b at 2 + 2b + e */
    MPI_Request           *sends;         /* the job messages sent in this colour, and their buffers */
    char                 **send_bufs;
    int                    sends_n, sends_cap;
    char                  *job_buffer;    /* rank 0: the job being received */
    server_t              *server;        /* rank 0 */
    long                   updates, accepted, skipped, jobs, failed_evaluations;
    double                 improvement;
    update_log_t          *update_log;    /* [R K], when asked for */
    int                    round;
} run_t;

static void post_job_receive(run_t *run)
{
    MPI_Irecv(run->job_buffer, (int)run->job_bytes, MPI_BYTE, MPI_ANY_SOURCE, TAG_JOB, run->comm, &run->recv[0]);
}

/* Rank 0: a job arrived. */
static void job_received(run_t *run)
{
    const job_head_t *head = (const job_head_t*)run->job_buffer;
    job_t *job = (job_t*)calloc(1, sizeof(job_t));

    job->t_received = parsec_qpu_now_ns();
    job->head       = *head;
    job->circuit    = (parsec_qpu_circuit_t*)malloc(head->circuit_bytes);
    memcpy(job->circuit, run->job_buffer + sizeof(job_head_t), head->circuit_bytes);
    job->params     = (double*)malloc((size_t)(head->nb_params > 0 ? head->nb_params : 1) * sizeof(double));
    memcpy(job->params, run->job_buffer + sizeof(job_head_t) + head->circuit_bytes, (size_t)head->nb_params * sizeof(double));
    job->result     = (parsec_qpu_result_t*)calloc(1, parsec_qpu_result_size(head->max_entries));
    post_job_receive(run);
    server_enqueue(run->server, job);
}

static void send_job(run_t *run, block_t *blk, int e)
{
    const size_t circuit_bytes = parsec_qpu_circuit_size(blk->circuit);
    const uint32_t nb_params = (uint32_t)(2 * run->cfg->layers);
    const size_t bytes = sizeof(job_head_t) + circuit_bytes + nb_params * sizeof(double);
    job_head_t head;

    /* The result can come back as soon as the job leaves: its receive first. */
    MPI_Irecv(blk->result[e], (int)parsec_qpu_result_size(run->max_entries), MPI_BYTE, 0, TAG_RESULT(blk->b, e),
              run->comm, &run->recv[2 + 2 * blk->b + e]);

    memset(&head, 0, sizeof(head));
    head.src           = run->rank;
    head.block         = blk->b;
    head.e             = e;
    head.shots         = (uint32_t)run->cfg->shots;
    head.key           = qaoa_key(run->round, blk->b, blk->t, e);
    head.max_entries   = run->max_entries;
    head.circuit_bytes = (uint32_t)circuit_bytes;
    head.nb_params     = nb_params;
    if( run->sends_n == run->sends_cap ) {
        run->sends_cap = (0 == run->sends_cap) ? 64 : 2 * run->sends_cap;
        run->sends     = (MPI_Request*)realloc(run->sends, (size_t)run->sends_cap * sizeof(MPI_Request));
        run->send_bufs = (char**)realloc(run->send_bufs, (size_t)run->sends_cap * sizeof(char*));
    }
    {
        /* The buffer lives until its request completes, at the end of the colour. */
        char *buf = (char*)malloc(bytes);
        memcpy(buf, &head, sizeof(head));
        memcpy(buf + sizeof(head), blk->circuit, circuit_bytes);
        memcpy(buf + sizeof(head) + circuit_bytes, blk->theta[e], nb_params * sizeof(double));
        run->send_bufs[run->sends_n] = buf;
        MPI_Isend(buf, (int)bytes, MPI_BYTE, 0, TAG_JOB, run->comm, &run->sends[run->sends_n++]);
    }
    run->jobs++;
}

static void finish_step(run_t *run, block_t *blk);

/* Propose the next step and send its two jobs; an update that evaluates
 * nothing goes through its steps at once, as in the reference. */
static void begin_step(run_t *run, block_t *blk)
{
    qaoa_opt_propose(&blk->S, run->cfg, blk->theta[0], blk->theta[1]);
    if( blk->P->skip ) {
        memset(blk->F, 0, sizeof(blk->F));
        finish_step(run, blk);
        return;
    }
    blk->waiting = 2;
    send_job(run, blk, 0);
    send_job(run, blk, 1);
}

static void finish_step(run_t *run, block_t *blk)
{
    qaoa_opt_absorb(&blk->S, run->cfg, &blk->F[0], &blk->F[1]);
    if( ++blk->t < run->cfg->steps ) {
        begin_step(run, blk);
        return;
    }
    {
        const double improvement = qaoa_state_update(&run->x[blk->b], blk->P, &blk->S);
        run->updates++;
        run->improvement += improvement;
        run->accepted += (improvement > 0.0);
        run->skipped  += (0 != blk->P->skip);
        blk->done = 1;
        if( NULL != run->update_log ) run->update_log[(size_t)run->round * run->inst->part.nb_blocks + blk->b].t_update = parsec_qpu_now_ns();
    }
}

static void start_update(run_t *run, block_t *blk)
{
    const qaoa_instance_t *inst = run->inst;
    const int b = blk->b;

    if( NULL != run->update_log ) run->update_log[(size_t)run->round * inst->part.nb_blocks + b].t_build = parsec_qpu_now_ns();
    qaoa_boundary_init(run->h, inst, b);
    for( int k = 0; k < inst->part.deg[b]; k++ ) {
        qaoa_boundary_gather(run->h, inst, b, k, &run->x[qaoa_block_nbr(&inst->part, b, k)]);
    }
    qaoa_local_build(blk->P, inst, run->cfg, run->round, b, &run->x[b], run->h);
    if( 0 != qaoa_circuit_build(blk->circuit, run->circuit_room, blk->P, run->cfg) ) {
        fprintf(stderr, "qaoa_mpi: the circuit of block %d does not fit\n", b);
        MPI_Abort(run->comm, 1);
    }
    qaoa_opt_init(&blk->S, run->cfg, &run->x[b], blk->P);
    blk->t    = 0;
    blk->done = 0;
    begin_step(run, blk);
}

/* One colour of one round, on this rank: its blocks, then the exchange. */
static void run_colour(run_t *run, int c)
{
    const qaoa_instance_t *inst = run->inst;
    const int K = inst->part.nb_blocks, P = run->world;
    const int nreq = 2 + 2 * K;
    int *counts = (int*)calloc((size_t)P, sizeof(int)), *displs = (int*)calloc((size_t)P, sizeof(int));
    int mine = 0, done = 0, total = 0, posted = 0;
    char *sendbuf, *recvbuf;

    for( int b = 0; b < K; b++ ) {
        if( inst->part.colour[b] != c ) continue;
        counts[b % P] += (int)sizeof(qaoa_state_t);
        total++;
        if( b % P == run->rank ) mine++;
    }
    for( int p = 1; p < P; p++ ) displs[p] = displs[p - 1] + counts[p - 1];
    sendbuf = (char*)malloc((size_t)(mine > 0 ? mine : 1) * sizeof(qaoa_state_t));
    recvbuf = (char*)malloc((size_t)(total > 0 ? total : 1) * sizeof(qaoa_state_t));

    for( int b = run->rank; b < K; b += P ) {
        if( inst->part.colour[b] == c ) start_update(run, &run->blocks[b]);
    }
    for( int b = run->rank; b < K; b += P ) {
        if( (inst->part.colour[b] == c) && run->blocks[b].done ) done++;
    }

    for(;;) {
        MPI_Status status;
        int idx;

        if( !posted && (done == mine) ) {
            /* This rank is through the colour: its states, to everybody. */
            int n = 0;
            for( int b = run->rank; b < K; b += P ) {
                if( inst->part.colour[b] == c ) memcpy(sendbuf + (size_t)(n++) * sizeof(qaoa_state_t), &run->x[b], sizeof(qaoa_state_t));
            }
            MPI_Iallgatherv(sendbuf, counts[run->rank], MPI_BYTE, recvbuf, counts, displs, MPI_BYTE, run->comm, &run->recv[1]);
            posted = 1;
        }
        MPI_Waitany(nreq, run->recv, &idx, &status);
        if( MPI_UNDEFINED == idx ) break;          /* not reached: rank 0 always has a job receive posted */
        if( 1 == idx ) break;                      /* the exchange is over: so is the colour, everywhere */
        if( 0 == idx ) { job_received(run); continue; }
        {
            const int b = (idx - 2) / 2, e = (idx - 2) % 2;
            block_t *blk = &run->blocks[b];
            qaoa_eval(&blk->F[e], blk->P, blk->result[e]);
            if( !blk->F[e].valid ) run->failed_evaluations++;
            if( 0 == --blk->waiting ) {
                finish_step(run, blk);
                if( blk->done ) done++;
            }
        }
    }
    /* The states of the colour, from every rank. */
    for( int p = 0, n = 0; p < P; p++ ) {
        for( int b = p; b < K; b += P ) {
            if( inst->part.colour[b] != c ) continue;
            memcpy(&run->x[b], recvbuf + (size_t)(n++) * sizeof(qaoa_state_t), sizeof(qaoa_state_t));
        }
    }
    MPI_Waitall(run->sends_n, run->sends, MPI_STATUSES_IGNORE);
    for( int i = 0; i < run->sends_n; i++ ) free(run->send_bufs[i]);
    run->sends_n = 0;
    free(counts); free(displs); free(sendbuf); free(recvbuf);
}

/* ------------------------------------------------------------------------
 * Parameters, as the runtime takes them
 * --------------------------------------------------------------------- */

static int take_parameter(const char *name, const char *value)
{
    static const char prefix[] = "device_qpu_";
    if( 0 != strncmp(name, prefix, sizeof(prefix) - 1) ) return -1;
    name += sizeof(prefix) - 1;
    if(      0 == strcmp(name, "max_outstanding") )  parsec_qpu_params.max_outstanding    = atoi(value);
    else if( 0 == strcmp(name, "submit_interval") )  parsec_qpu_params.submit_interval_ms = atoi(value);
    else if( 0 == strcmp(name, "mock_seed") )        parsec_qpu_params.mock_seed          = atoi(value);
    else if( 0 == strcmp(name, "mock_delay_seed") )  parsec_qpu_params.mock_delay_seed    = atoi(value);
    else if( 0 == strcmp(name, "mock_slots") )       parsec_qpu_params.mock_slots         = atoi(value);
    else if( 0 == strcmp(name, "mock_delay") )       parsec_qpu_params.mock_delay         = (char*)value;
    else if( 0 == strcmp(name, "mock_queue_delay") ) parsec_qpu_params.mock_queue_delay   = (char*)value;
    else if( 0 == strcmp(name, "mock_job_delay") )   parsec_qpu_params.mock_job_delay     = (char*)value;
    else if( 0 == strcmp(name, "mock_threads") )     parsec_qpu_params.mock_threads       = atoi(value);
    else if( 0 == strcmp(name, "mock_max_qubits") )  parsec_qpu_params.mock_max_qubits    = atoi(value);
    else if( (0 == strcmp(name, "enabled")) || (0 == strcmp(name, "max_batch")) ) return 0;
    else return -1;
    return 0;
}

int main(int argc, char *argv[])
{
    qaoa_config_t cfg;
    qaoa_instance_t inst;
    const char *name = NULL, *generated = NULL, *log_prefix = NULL;
    int compare = 1, machine = 0, provided, rank, world, errors = 0, test_argc = argc;
    run_t run;
    server_t server;
    int64_t t0, t1, elapsed, longest, cpu_us, cpu_all = 0;
    struct rusage ru0, ru1;
    parsec_qpu_mock_stats_t mock;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if( provided < MPI_THREAD_MULTIPLE ) {
        if( 0 == rank ) fprintf(stderr, "qaoa_mpi: MPI_THREAD_MULTIPLE is needed, and the library gives %d\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* What parsec_init() would have read, then what the command line says. */
    parsec_qpu_params.max_outstanding    = 8;
    parsec_qpu_params.submit_interval_ms = 0;
    parsec_qpu_params.mock_seed          = 1;
    parsec_qpu_params.mock_delay_seed    = 0;
    parsec_qpu_params.mock_slots         = 4;
    parsec_qpu_params.mock_delay         = (char*)"const:1000";
    parsec_qpu_params.mock_queue_delay   = (char*)"const:0";
    parsec_qpu_params.mock_job_delay     = (char*)"const:0";
    parsec_qpu_params.mock_faults        = (char*)"none";
    parsec_qpu_params.mock_threads       = 2;
    parsec_qpu_params.mock_max_qubits    = 24;
    parsec_qpu_params.mock_max_circuits  = 1;
    for( int i = 1; i < argc; i++ ) {
        if( 0 != strcmp(argv[i], "--") ) continue;
        test_argc = i;
        for( int j = i + 1; j < argc; j++ ) {
            if( (0 == strcmp(argv[j], "--mca")) && (j + 2 < argc) ) {
                if( 0 != take_parameter(argv[j + 1], argv[j + 2]) ) {
                    if( 0 == rank ) fprintf(stderr, "qaoa_mpi: %s is not a parameter B takes\n", argv[j + 1]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                j += 2;
            }
        }
        break;
    }
    qaoa_config_default(&cfg);
    for( int i = 1; i < test_argc; i++ ) {
        if( 0 == strcmp(argv[i], "-v") ) { verbose = 1; continue; }
        if( 0 == strcmp(argv[i], "-M") ) { machine = 1; continue; }
        if( i + 1 >= test_argc ) { if( 0 == rank ) fprintf(stderr, "qaoa_mpi: %s needs a value\n", argv[i]); MPI_Abort(MPI_COMM_WORLD, 1); }
        if(      0 == strcmp(argv[i], "-i") ) name = argv[++i];
        else if( 0 == strcmp(argv[i], "-g") ) generated = argv[++i];
        else if( 0 == strcmp(argv[i], "-p") ) cfg.layers = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-T") ) cfg.steps = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-R") ) cfg.rounds = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-s") ) cfg.shots = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-P") ) cfg.patience = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-S") ) cfg.seed = strtoull(argv[++i], NULL, 0);
        else if( 0 == strcmp(argv[i], "-c") ) compare = atoi(argv[++i]);
        else if( 0 == strcmp(argv[i], "-L") ) log_prefix = argv[++i];
        else { if( 0 == rank ) fprintf(stderr, "qaoa_mpi: unknown option %s\n", argv[i]); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if( (cfg.rounds < 1) || (cfg.steps < 1) || (cfg.shots < 1) || (cfg.layers < 1) || (cfg.layers > QAOA_MAX_LAYERS) ) {
        if( 0 == rank ) fprintf(stderr, "qaoa_mpi: %d round(s) of %d step(s), %d layer(s) and %d shot(s) is not a run\n",
                                cfg.rounds, cfg.steps, cfg.layers, cfg.shots);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if( (NULL != generated) ? (0 != qaoa_instance_spec(&inst, generated, cfg.seed))
                            : (0 != qaoa_instance_named(&inst, (NULL != name) ? name : "ring64")) ) {
        if( 0 == rank ) fprintf(stderr, "qaoa_mpi: no such instance\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- the run's memory, on every rank ---- */
    {
        const qaoa_partition_t *p = &inst.part;
        const int K = p->nb_blocks;

        memset(&run, 0, sizeof(run));
        run.inst         = &inst;
        run.cfg          = &cfg;
        run.comm         = MPI_COMM_WORLD;
        run.rank         = rank;
        run.world        = world;
        run.circuit_room = qaoa_circuit_size(p->max_block, p->max_internal, cfg.layers);
        run.max_entries  = qaoa_result_entries(&cfg, p->max_block);
        run.job_bytes    = sizeof(job_head_t) + run.circuit_room + 2 * QAOA_MAX_LAYERS * sizeof(double);
        run.x            = (qaoa_state_t*)calloc((size_t)K, sizeof(qaoa_state_t));
        run.h            = (qaoa_boundary_t*)malloc(qaoa_boundary_size(p->max_block));
        run.blocks       = (block_t*)calloc((size_t)K, sizeof(block_t));
        run.recv         = (MPI_Request*)malloc((size_t)(2 + 2 * K) * sizeof(MPI_Request));
        for( int i = 0; i < 2 + 2 * K; i++ ) run.recv[i] = MPI_REQUEST_NULL;
        for( int b = 0; b < K; b++ ) qaoa_state_init(&run.x[b], &inst, &cfg, b);
        for( int b = rank; b < K; b += world ) {
            block_t *blk = &run.blocks[b];
            blk->b         = b;
            blk->P         = (qaoa_local_t*)malloc(qaoa_local_size(p->max_block, p->max_internal));
            blk->circuit   = (parsec_qpu_circuit_t*)malloc(run.circuit_room);
            blk->result[0] = (parsec_qpu_result_t*)malloc(parsec_qpu_result_size(run.max_entries));
            blk->result[1] = (parsec_qpu_result_t*)malloc(parsec_qpu_result_size(run.max_entries));
        }
        if( NULL != log_prefix ) {
            run.update_log = (update_log_t*)calloc((size_t)cfg.rounds * K, sizeof(update_log_t));
        }
    }

    /* ---- rank 0: the backend and its helper thread ---- */
    memset(&server, 0, sizeof(server));
    if( 0 == rank ) {
        if( (PARSEC_SUCCESS != parsec_qpu_backend_mock_ops.open(&server.backend, 0)) || (NULL == server.backend) ) {
            fprintf(stderr, "qaoa_mpi: the mock backend did not open\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        server.comm            = MPI_COMM_WORLD;
        server.max_outstanding = (uint32_t)((parsec_qpu_params.max_outstanding > 0) ? parsec_qpu_params.max_outstanding : 1);
        server.interval_ns     = (int64_t)((parsec_qpu_params.submit_interval_ms > 0) ? parsec_qpu_params.submit_interval_ms : 0) * 1000000LL;
        server.in_flight       = (job_t**)calloc(server.max_outstanding, sizeof(job_t*));
        if( NULL != log_prefix ) { server.log_cap = 1024; server.log = (job_log_t*)malloc((size_t)server.log_cap * sizeof(job_log_t)); }
        pthread_mutex_init(&server.lock, NULL);
        pthread_create(&server.thread, NULL, server_main, &server);
        run.server     = &server;
        run.job_buffer = (char*)malloc(run.job_bytes);
        post_job_receive(&run);
    }

    /* ---- the run ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    getrusage(RUSAGE_SELF, &ru0);
    t0 = parsec_qpu_now_ns();
    for( int r = 0; r < cfg.rounds; r++ ) {
        run.round = r;
        for( int c = 0; c < inst.part.nb_colours; c++ ) run_colour(&run, c);
    }
    t1 = parsec_qpu_now_ns();
    getrusage(RUSAGE_SELF, &ru1);
    elapsed = t1 - t0;
    MPI_Reduce(&elapsed, &longest, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    /* The processor time of all the ranks, the server's thread included. */
    cpu_us = ((int64_t)(ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) + (int64_t)(ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec)) * 1000000LL
           + (int64_t)(ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) + (int64_t)(ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec);
    MPI_Reduce(&cpu_us, &cpu_all, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    memset(&mock, 0, sizeof(mock));
    if( 0 == rank ) {
        MPI_Cancel(&run.recv[0]);
        MPI_Wait(&run.recv[0], MPI_STATUS_IGNORE);
        pthread_mutex_lock(&server.lock);
        server.stop = 1;
        pthread_mutex_unlock(&server.lock);
        (void)server.backend->ops->wakeup(server.backend);
        pthread_join(server.thread, NULL);
        parsec_qpu_mock_stats(&mock);
    }

    /* ---- what it found, against the reference ---- */
    {
        long sums[6] = { run.updates, run.accepted, run.skipped, run.jobs, run.failed_evaluations, 0 }, all[6];
        double improvement = 0.0;

        MPI_Reduce(sums, all, 6, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&run.improvement, &improvement, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if( 0 == rank ) {
            const int K = inst.part.nb_blocks;
            qaoa_state_t *first = (qaoa_state_t*)calloc((size_t)K, sizeof(qaoa_state_t));
            double cut_initial, cut_final;
            int different = 0;

            for( int b = 0; b < K; b++ ) qaoa_state_init(&first[b], &inst, &cfg, b);
            cut_initial = qaoa_cut(&inst, first);
            cut_final   = qaoa_cut(&inst, run.x);
            free(first);
            if( cut_final != cut_initial + improvement ) {
                fprintf(stderr, "qaoa_mpi: the final cut %.4f is not the initial %.4f plus the improvements %.4f\n",
                        cut_final, cut_initial, improvement);
                errors++;
            }
            if( (long)all[0] != (long)cfg.rounds * K ) {
                fprintf(stderr, "qaoa_mpi: %ld updates for %d rounds of %d blocks\n", all[0], cfg.rounds, K);
                errors++;
            }
            if( (server.jobs + server.refused != all[3]) || (all[4] != server.failed + server.refused) ) {
                fprintf(stderr, "qaoa_mpi: %ld jobs sent, the server ran %ld, %ld were refused and %ld failed; %ld evaluations failed\n",
                        all[3], server.jobs, server.refused, server.failed, all[4]);
                errors++;
            }
            /* The limits of the device, kept. */
            if( (server.max_in_flight > server.max_outstanding) || (mock.max_running > (uint32_t)parsec_qpu_params.mock_slots) ||
                ((server.interval_ns > 0) && (mock.min_submit_gap >= 0) && (mock.min_submit_gap < server.interval_ns)) ) {
                fprintf(stderr, "qaoa_mpi: %u jobs at the backend at once (limit %u), %u running on %d slots, %.3f ms between two "
                        "submissions (at least %.3f)\n", server.max_in_flight, server.max_outstanding, mock.max_running,
                        parsec_qpu_params.mock_slots, mock.min_submit_gap * 1e-6, server.interval_ns * 1e-6);
                errors++;
            }
            if( (0 != mock.duplicate_tags) || (0 != mock.key_overlaps) || (0 != mock.inputs_changed) || (0 != mock.late) ) {
                fprintf(stderr, "qaoa_mpi: the mock saw %llu duplicate tags, %llu overlapping keys, %llu changed inputs; %llu jobs ended late\n",
                        (unsigned long long)mock.duplicate_tags, (unsigned long long)mock.key_overlaps,
                        (unsigned long long)mock.inputs_changed, (unsigned long long)mock.late);
                if( (0 != mock.duplicate_tags) || (0 != mock.key_overlaps) || (0 != mock.inputs_changed) ) errors++;
            }
            if( compare ) {
                qaoa_direct_t direct = { NULL, (uint64_t)parsec_qpu_params.mock_seed };
                qaoa_run_t ref;

                if( 0 != qaoa_sequential_run(&inst, &cfg, qaoa_sample_direct, &direct, 0, &ref) ) {
                    fprintf(stderr, "qaoa_mpi: the reference could not run\n");
                    errors++;
                } else {
                    for( int b = 0; b < K; b++ ) different += (0 != memcmp(&run.x[b], &ref.x[b], sizeof(qaoa_state_t)));
                    if( different ) {
                        fprintf(stderr, "qaoa_mpi: %d of %d blocks do not end in the state of the reference\n", different, K);
                        errors++;
                    }
                    if( (ref.jobs != all[3]) || (ref.accepted != all[1]) || (ref.skipped != all[2]) ) {
                        fprintf(stderr, "qaoa_mpi: %ld jobs, %ld accepted, %ld skipped; the reference has %ld, %ld, %ld\n",
                                all[3], all[1], all[2], ref.jobs, ref.accepted, ref.skipped);
                        errors++;
                    }
                    qaoa_run_free(&ref);
                }
                parsec_qpu_sampler_free(direct.sampler);
            }
            printf("%-12s B (MPI)                cut %8.4f -> %8.4f  %4ld jobs asked for, %3ld/%-3ld updates accepted, %2ld skipped, "
                   "%7.1f ms on %d rank(s)%s\n", inst.name, cut_initial, cut_final, all[3], all[1], all[0], all[2], longest * 1e-6, world,
                   !compare ? "" : (different ? ": final states NOT those of the reference"
                                              : ": final states equal to the reference, bit for bit"));
            if( machine ) {
                printf("qaoa_result system=B instance=%s blocks=%d colours=%d ranks=%d rounds=%d steps=%d layers=%d shots=%d "
                       "time_ms=%.3f cpu_s=%.3f jobs=%ld circuits=%llu late=%llu owner=0\n",
                       inst.name, K, inst.part.nb_colours, world, cfg.rounds, cfg.steps, cfg.layers, cfg.shots, longest * 1e-6, cpu_all * 1e-6,
                       server.jobs, (unsigned long long)mock.circuits, (unsigned long long)mock.late);
            }
            if( verbose ) {
                printf("qaoa_mpi: at most %u jobs at the backend (limit %u); the mock ran at most %u at once\n",
                       server.max_in_flight, server.max_outstanding, mock.max_running);
            }
        }
    }

    /* ---- the log ---- */
    if( NULL != log_prefix ) {
        const int K = inst.part.nb_blocks;
        char path[1024];
        FILE *file;

        if( 0 == rank ) {
            snprintf(path, sizeof(path), "%s.0.jobs", log_prefix);
            if( NULL != (file = fopen(path, "w")) ) {
                fprintf(file, "# r b t e t_hook t_submit t_start t_end t_retrieve t_eval batch attempts status (ns, the run began at %lld; "
                        "B: t_hook when rank 0 received the job, t_eval when it sent the result)\n", (long long)t0);
                for( long i = 0; i < server.log_n; i++ ) {
                    const job_log_t *j = &server.log[i];
                    const int jr = (int)(j->key >> 44), jt = (int)((j->key >> 1) & 0x7ffff);
                    fprintf(file, "%d %d %d %d %lld %lld %lld %lld %lld %lld 1 1 %u\n", jr, j->block, jt, j->e,
                            (long long)j->t_received, (long long)j->t_submit, (long long)j->t_start, (long long)j->t_end,
                            (long long)j->t_retrieve, (long long)j->t_sent, j->status);
                }
                fclose(file);
            } else {
                errors++;
            }
        }
        snprintf(path, sizeof(path), "%s.%d.updates", log_prefix, rank);
        if( NULL != (file = fopen(path, "w")) ) {
            fprintf(file, "# r b colour t_build t_update (ns; the run began at %lld and ended at %lld)\n", (long long)t0, (long long)t1);
            for( int r = 0; r < cfg.rounds; r++ ) for( int b = rank; b < K; b += world ) {
                const update_log_t *u = &run.update_log[(size_t)r * K + b];
                fprintf(file, "%d %d %d %lld %lld\n", r, b, inst.part.colour[b], (long long)u->t_build, (long long)u->t_update);
            }
            fclose(file);
        } else {
            errors++;
        }
    }

    /* ---- the end ---- */
    if( 0 == rank ) {
        (void)server.backend->ops->close(server.backend);
        pthread_mutex_destroy(&server.lock);
        free(server.in_flight);
        free(server.log);
        free(run.job_buffer);
    }
    for( int b = rank; b < inst.part.nb_blocks; b += world ) {
        block_t *blk = &run.blocks[b];
        free(blk->P); free(blk->circuit); free(blk->result[0]); free(blk->result[1]);
    }
    free(run.blocks); free(run.recv); free(run.sends); free(run.send_bufs); free(run.x); free(run.h); free(run.update_log);
    qaoa_instance_free(&inst);
    {
        int all_errors = 0;
        MPI_Allreduce(&errors, &all_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if( 0 == rank ) printf("qaoa_mpi: %s\n", all_errors ? "FAILED" : "passed");
        MPI_Finalize();
        return all_errors ? 1 : 0;
    }
}
