# Quantum processing units in PaRSEC

The `device/qpu` component makes a quantum processing unit a device of the
runtime, next to the CPU and the GPUs. A task whose body is of type QPU
submits a quantum circuit and gives its worker back at once; the task is
completed, and its successors released, when the job has ended. Its inputs
(the circuit, its parameter values) and its output (the measurement counts)
are ordinary flows, so quantum tasks sit in the same task graph as classical
ones, across ranks.

The only backend so far is a **mock**: it runs in the process, samples the
circuits exactly, and models a provider's slots, queue and job times. No test
and no default setting can reach a quantum device.

The design, the implementation log of each milestone and the evaluation are in
[`docs/design/parsec-qpu-qaoa-plan.md`](../../docs/design/parsec-qpu-qaoa-plan.md).
The API is documented in
[`parsec/mca/device/qpu/device_qpu.h`](../../parsec/mca/device/qpu/device_qpu.h).

## Building

```bash
cmake -DPARSEC_DEVICE_WITH_QPU=ON ...
```

Off by default. It needs no provider SDK and no Python. With it off, a task
class that has a QPU body and a CPU body still builds; its QPU body is left
out.

## Writing a quantum task

In a JDF:

```
QEXEC(k)
k = 0 .. N-1
: Q(0)                                   /* the rank that owns the QPU */

READ  C   <- C PREP(k)                   /* a parsec_qpu_circuit_t */
READ  TH  <- TH PREP(k)                  /* its parameter values */
WRITE RES <- NEW  [type = RESULT]        /* a parsec_qpu_result_t and its entries */
          -> RES USE(k)

BODY [type=QPU]
{
    return PARSEC_QPU_SUBMIT_KEYED(C, TH, RES, 1024, my_key(k));
}
END
```

With DTD: a task class with a chore of type `PARSEC_DEV_QPU` whose body
returns `parsec_qpu_submit_dtd(es, this_task, circuit_index, params_index,
result_index, shots, key)`.

What the body must and must not do:

- Submit as the last thing it does, and return what the submission returns.
  It must not touch the task after that: the task may already be completed.
- Place the task on the rank that owns the QPU (`device_qpu_owner_rank`); a
  quantum task on another rank is refused.
- The circuit format is in `device_qpu.h` (`parsec_qpu_circuit_init`,
  `parsec_qpu_circuit_add`, `parsec_qpu_circuit_measure_all`). A circuit
  flagged `PARSEC_QPU_CIRCUIT_EMPTY` completes without a job.
- The key names the evaluation. The mock draws its samples from it, so the
  same key, circuit and parameters give the same counts in any order, at any
  attempt, alone or sharing a job with others; a sequential program that
  calls the sampler with the same key gets them too.
- A job that fails is retried (`device_qpu_max_retries`). A task whose jobs
  all failed still completes; the `status` of its result says what happened,
  and its successors decide what that means.

A task class may also have a CPU body: it runs in a context without a QPU
device.

## Parameters

All are MCA parameters (`--mca device_qpu_<name> <value>`, or
`PARSEC_MCA_device_qpu_<name>` in the environment).

| Parameter | Default | What it does |
|---|---|---|
| `enabled` | 0 | QPUs in the next context (one is supported) |
| `backend` | `mock` | The backend |
| `allow_hardware` | 0 | Let a backend that drives a quantum device open |
| `owner_rank` | 0 | The rank that runs every job (-1: the last) |
| `max_outstanding` | 8 | Jobs at the backend at once; the others wait, by task priority then arrival |
| `max_batch` | 1 | Tasks that may share one job: only when more tasks wait than jobs can be submitted, spread over the free places, never held back to fill one |
| `submit_interval` | 0 | Milliseconds between two submissions |
| `max_retries` | 2 | Jobs a task may have after its first |
| `job_timeout` | 0 | Milliseconds a job of one task may take (n times that for a job of n tasks); 0: no limit |
| `unavailable_timeout` | 60000 | Milliseconds the backend may take nothing before the waiting tasks complete as unavailable |
| `retry_backoff` | 100 | Milliseconds before a submission that was not taken is asked again (doubles, up to 30 s) |
| `poll_max` | 1000 | Longest the progress thread sleeps |
| `verbose` | -1 | Verbosity (negative: that of the devices) |

The mock:

| Parameter | Default | What it does |
|---|---|---|
| `mock_seed` | 1 | Seed of the samples and of everything else it draws |
| `mock_delay_seed` | 0 | Seed of the times alone (0: `mock_seed`): other job times, the same answers |
| `mock_slots` | 4 | Jobs it runs at once; the others queue inside it |
| `mock_delay` | `const:1000` | Time of one circuit: `const:<us>`, `uniform:<min>:<max>`, `exp:<mean>`, `lognormal:<median>:<sigma>`, `replay:<file of us>` |
| `mock_job_delay` | `const:0` | Time every job takes on top of its circuits |
| `mock_queue_delay` | `const:0` | Time a job waits in its slot before it runs |
| `mock_max_circuits` | 64 | Circuits it takes in one job |
| `mock_faults` | `none` | Faults to inject: `<fault>:<rule>=<value>,...` with fault one of `fail hang unavailable lost reject noresult duplicate late cancelled`, rule one of `p first every key for` |
| `mock_threads` | 2 | Helper threads that sample the circuits |
| `mock_max_qubits` | 24 | Widest circuit it takes (at most 30; 2^(n+4) bytes per helper) |

`device_show_statistics 1` prints one line per run: tasks, jobs, failures,
retries, the most jobs outstanding and tasks waiting, the jobs that carried
several tasks, and the mean time of a job.

## The application: block-decomposed QAOA

`tests/apps/qaoa/`. A weighted MaxCut graph is split into blocks; each round
updates every block in colour order, and each update runs a few SPSA steps of
two quantum evaluations each.

| Program | What it is |
|---|---|
| `testing_qaoa` | The application as one PTG taskpool (system C; D with `device_qpu_max_batch` above 1; C-sync with `-y`), compared byte for byte with the sequential reference |
| `qaoa_mpi` | The same algorithm as a hand-written MPI program (baseline B); takes the device parameters after `--` |
| `qaoa_reference` | The sequential reference; with `-b`, every job a blocking call through the mock (system A) |

Instances: `-i NAME` for a fixed one, or `-g [ring|grid|complete|rand:]K:N:C:X`
for K blocks of N vertices. `-M` prints one line a script can read, `-L PREFIX`
the timestamps of every job and update.

```bash
mpiexec -n 4 tests/apps/qaoa/testing_qaoa -g grid:16:10:4:2 -T 8 -R 4 -- \
    --mca device_qpu_enabled 1 --mca device_qpu_mock_delay lognormal:14142:0.8326 \
    --mca device_qpu_max_outstanding 16 --mca device_qpu_mock_slots 16
```

## Benchmarks

`tests/runtime/qpu/qpu_bench` runs chains of classical work and quantum jobs:
with the QPU device (system C), with an empty CPU body instead (what the QPU
path adds), with a CPU body that holds its worker for the time of a job
(C-block), or without the runtime (`-A`, system A).

The experiment scripts, the timing model and the raw results of the evaluation
are kept outside version control.
