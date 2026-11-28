# Quantum processing units in PaRSEC

The `device/qpu` component makes a quantum processing unit a device of the
runtime, next to the CPU and the GPUs. A task whose body is of type QPU
submits a quantum circuit and gives its worker back at once; the task is
completed, and its successors released, when the job has ended. Its inputs
(the circuit, its parameter values) and its output (the measurement counts)
are ordinary flows, so quantum tasks sit in the same task graph as classical
ones, across ranks.

There are two backends. The **mock** runs in the process, samples the circuits
exactly, and models a provider's slots, queue and job times. The **adapter**
backend is the client of a separate program that runs the jobs; the two
adapters that exist, `parsec_qpu_adapter_qiskit.py` and
`parsec_qpu_adapter_cudaq.py` in this directory, run them on simulators,
Qiskit Aer and CUDA-Q's. No test and no default setting can reach a quantum
device, and nothing in this tree has code that could.

The design, the implementation log of each milestone and the evaluation are in
[`docs/design/parsec-qpu-qaoa-plan.md`](../../docs/design/parsec-qpu-qaoa-plan.md).
The API is documented in
[`parsec/mca/device/qpu/device_qpu.h`](../../parsec/mca/device/qpu/device_qpu.h).

## Building

```bash
cmake -DPARSEC_DEVICE_WITH_QPU=ON ...
```

Off by default. It needs no provider SDK and no Python: the adapter backend is
plain C over a socket, and the adapter program is neither built nor installed
with PaRSEC. With the component off, a task class that has a QPU body and a
CPU body still builds; its QPU body is left out.

## Writing a quantum task

In a JDF:

```
QEXEC(k)
k = 0 .. N-1
: Q(0)                                   /* a rank that owns a QPU */

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
- Place the task on a rank that owns a QPU (`device_qpu_owner_rank`, or the
  ranks named in `device_qpu_devices`); a quantum task on a rank that owns
  none is refused.
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

## Several QPUs

A run has one QPU by default. `device_qpu_devices` lists several:

```bash
--mca device_qpu_enabled 1 \
--mca device_qpu_devices "mock@0 slots=4; mock@0 slots=2 delay=exp:20000; mock@3"
```

- **The list.** Entries are separated by `;` or `|`. Each is a backend, an
  optional `@rank` (`-1` is the last rank; without it, `device_qpu_owner_rank`),
  and `key=value` pairs. The position of an entry is the number of its QPU, so
  the list has to be the same on every rank. Without a list there are
  `device_qpu_enabled` QPUs, all alike.
- **Keys.** The limits of that QPU: `max_outstanding`, `max_batch`,
  `submit_interval`, `job_timeout`, `max_retries`, `unavailable_timeout`. They
  default to the parameters of those names. For the mock: `slots`, `delay`,
  `queue_delay`, `job_delay`, `submit_delay`, `max_circuits`, `faults`,
  `threads`, `max_qubits`, which default to `device_qpu_mock_<key>`. For the
  adapter: `address`, `timeout`, which default to `device_qpu_adapter_<key>`.
  A key the backend of the entry does not know is an error, and a list that
  cannot be read gives no QPU at all.
- **Which QPU runs a task.** The QPUs of a rank share the tasks that wait on
  that rank. A task is given to a QPU when that QPU has a place for its job,
  not before: no place stays free while a task that could use it waits, and a
  faster QPU runs more tasks. A QPU takes only circuits its backend can run.
  The result says which QPU ran the job (`qpu`).
- **Naming a QPU.** `PARSEC_QPU_SUBMIT_ON(q, C, TH, RES, shots, key)` in a
  JDF, `parsec_qpu_submit_on()` and `parsec_qpu_submit_dtd_on()` elsewhere:
  the job runs on QPU `q` and on no other. The task has to be on the rank that
  owns it. `PARSEC_QPU_ANY` is what the calls without `_ON` pass.
- **QPUs on several ranks.** A QPU does not take tasks that wait on another
  rank. The application spreads its quantum tasks over the ranks that own a
  QPU, through their affinity: `parsec_qpu_count()` and `parsec_qpu_rank_of(q)`
  say what there is, on every rank.
- **When a QPU fails.** A job that failed is tried again on another QPU when
  one can run it. A QPU whose backend takes no job for `unavailable_timeout`
  is given up on: its tasks go to the others, a task that named it completes
  as unavailable, and it is tried again with one job after a while, then
  after twice as long. A QPU whose backend cannot be opened is left out, with
  a warning, and the others are used.
- **Nothing can reach a quantum device** through a list either: one entry
  whose backend drives hardware, without `device_qpu_allow_hardware`, and no
  QPU of the list is used. (A backend that says so only once it is opened is
  closed again and left out.)

`parsec_qpu_device_info(q, &info)` describes QPU `q` from any rank.

## The adapter backend, and Qiskit

`--mca device_qpu_backend adapter` makes a QPU the client of an **adapter**: a
program that runs beside the application, takes circuits as OpenQASM 3 and
gives counts back. The adapter holds everything that belongs to a provider or
its SDK, so PaRSEC needs none of it. `tools/qpu` has two: one on Qiskit, which
this section starts with, and one on CUDA-Q, further down. What is said of the
socket, of adapters that go away and of hardware holds for both.

```bash
# Once, in a Python environment of your own:
pip install qiskit qiskit-aer qiskit-qasm3-import

# Start the adapter, then the application, on the node of the rank that owns the QPU.
mkdir -m 700 -p /tmp/$USER-qpu            # a directory nobody else can write to
python3 tools/qpu/parsec_qpu_adapter_qiskit.py --socket /tmp/$USER-qpu/a.sock &
mpiexec -n 4 tests/apps/qaoa/testing_qaoa -c 0 -- --mca device_qpu_enabled 1 \
    --mca device_qpu_backend adapter --mca device_qpu_adapter_address /tmp/$USER-qpu/a.sock
kill %1
```

- **What the adapter runs the jobs on.** Qiskit Aer, on the machine it runs
  on, with no account and no network.
  - Without `--device`: an ideal simulator.
  - `--device generic:N[:line|ring|full]`: the circuits are transpiled to the
    gates and the connectivity of a device model of N qubits, and run with
    its noise (`--no-noise`: without). The model is Qiskit's
    `GenericBackendV2`, a made-up device, not an existing one.
- **Results are reproducible.** What an evaluation gives depends on `--seed`
  and on the key of the evaluation, as with the mock: not on the order of the
  jobs, on what shares a job, or on how often a job was tried. They are not
  the counts of the mock, which samples by its own means.
- **Several QPUs.** Each entry of `device_qpu_devices` names its adapter:
  `adapter address=/path/a.sock|adapter address=/path/b.sock`, beside mocks if
  wanted. Two QPUs may share one adapter; each has its own jobs there.
- **The address is a Unix-domain socket**, which only the user who started the
  adapter can open, and the client talks only to an adapter that runs as its
  own user. An adapter does not start on a path where another one listens. Put the socket in a directory nobody else can write to: the
  client connects again by itself to whatever listens at that path. The path
  has to be shorter than 108 bytes. The adapter runs on the node of the rank
  that owns the QPU.
- **When the adapter goes away.** The client connects again by itself. An
  adapter that is the same process still has the client's jobs. One that was
  started again does not: those jobs are reported as failed and the runtime
  submits them again (`device_qpu_max_retries`). Jobs of an adapter that stays
  away for `device_qpu_unavailable_timeout` are given up, and the adapter is
  told to drop them if it comes back.
- **When the application goes away.** A run that closes its QPUs takes its
  jobs with it. One that ends without doing so (it crashed) leaves them at
  the adapter, which drops them `--idle-drop` seconds (600 by default) after
  the last connection of that client closed.
- **A call to the adapter can take as long as `device_qpu_adapter_timeout`**
  when the adapter is there and does not answer; a QPU's thread, and the end
  of the run, wait that long for it at most.
- **Hardware.** An adapter says, when a client connects, whether it drives a
  quantum device. One that does is used only with
  `--mca device_qpu_allow_hardware 1`, and an adapter that changes its answer
  during a run is dropped. This adapter always says no.
- **What crosses the socket**: circuits, parameter values, shot counts, keys
  and tags, and counts on the way back. The client reads no variable of the
  environment and no file.

`ctest -R adapter` runs the client against an adapter written for the tests
(`tests/runtime/qpu/qpu_adapter_loopback`, in C, which answers from the
sampler). The tests with Qiskit Aer are registered only when a Python with the
three packages is found; `-DPARSEC_QPU_ADAPTER_PYTHON=/path/to/python` names
it.

### The adapter for CUDA-Q

`tools/qpu/parsec_qpu_adapter_cudaq.py` is an adapter for the same backend and
the same protocol, which runs the jobs on a simulator of CUDA-Q.

```bash
# Once, in a Python environment of your own:
pip install cudaq

# One adapter is one QPU. Here, two QPUs that are two GPUs of the node:
CUDA_VISIBLE_DEVICES=0 python3 tools/qpu/parsec_qpu_adapter_cudaq.py --socket /tmp/$USER-qpu/g0.sock --target nvidia &
CUDA_VISIBLE_DEVICES=1 python3 tools/qpu/parsec_qpu_adapter_cudaq.py --socket /tmp/$USER-qpu/g1.sock --target nvidia &
tests/apps/qaoa/testing_qaoa -c 0 -- --mca device_qpu_enabled 1 \
    --mca device_qpu_devices "adapter address=/tmp/$USER-qpu/g0.sock|adapter address=/tmp/$USER-qpu/g1.sock"
kill %1 %2
```

- **Simulators only.** `--target qpp-cpu`, the default, is CUDA-Q's simulator
  on the CPU. `--target nvidia` is its simulator on a GPU; CUDA-Q asks for a
  GPU of compute capability 7.5 or more. The program takes no other target,
  and says that it drives no quantum device. With the CPU simulator it hides
  the GPUs of the node from CUDA-Q, which would otherwise start on one.
- **Threads.** The CPU simulator computes with `--threads` threads; without
  the option, with `OMP_NUM_THREADS` of the environment, and with one thread
  when that is not set. CUDA-Q also keeps a thread per core for compiling
  circuits, which do nothing between two new circuits.
- **One adapter is one QPU of CUDA-Q, and runs one job at a time.** Several
  QPUs are several adapters, each with its socket and its entry in
  `device_qpu_devices`, and for the GPU simulator each with a GPU of its own.
  `--target nvidia --option mqpu --qpu K` takes QPU K of CUDA-Q's own
  platform of several QPUs, which has one for each GPU it sees.
- **A GPU for the adapter is a GPU PaRSEC does not use.** PaRSEC's CUDA device
  takes 95% of the memory of every GPU it is given (`device_cuda_memory_use`),
  so the simulator finds none there. Keep the adapters' GPUs out of the
  application's `CUDA_VISIBLE_DEVICES`, or out of its `device_cuda_mask`.
- **Results are reproducible**, by the same rule as with the Qiskit adapter:
  they depend on `--seed` and on the key of an evaluation. They are the counts
  of neither the mock nor Qiskit Aer.
- **The circuits are read by the adapter itself**, and only as the backend
  writes them (`parsec_qpu_qasm_write`): CUDA-Q reads no OpenQASM 3 that has
  parameters. Another text is refused.
- **The first job of a circuit pays for compiling it**: about 0.1 s for a
  circuit of a hundred operations and 2 s for one of two thousand, on top of
  the job. A circuit that is kept takes about a megabyte, and
  `--max-prepared` circuits are kept (256 by default); an older one is asked
  for again, and compiled again, when a job needs it.
- **An angle that is no finite number** (a coefficient times a parameter value
  that is too large) fails its job. CUDA-Q itself would end the process.

The tests with CUDA-Q are registered only when a Python with `cudaq` is found;
`-DPARSEC_QPU_CUDAQ_PYTHON=/path/to/python` names it. They use the CPU
simulator. The adapter's reader is checked with any `python3`, against the
texts the test of the component's writer keeps (`ctest -R cudaq:reader`).

The two programs share `parsec_qpu_adapter_protocol.py`, which is the
protocol's side of an adapter and has to stay beside them.

### Circuits as OpenQASM 3

`parsec/mca/device/qpu/device_qpu_qasm.h` writes a circuit as OpenQASM 3 and
reads one back, without any dependency.

```c
int64_t n = parsec_qpu_qasm_write(circuit, text, room);          /* length of the text */
int rc = parsec_qpu_qasm_read(text, length, circuit, room, &info);
int slot = parsec_qpu_qasm_param_slot(text, length, "gamma");
```

- **The reader takes what `qiskit.qasm3.dumps` writes** for a circuit of the
  twelve gates of the circuit format and measurements: build the circuit in
  Qiskit, write it to a file, and read it into the flow of a quantum task.
  The k-th `input` of the text is parameter slot k (Qiskit writes its
  parameters in the order of their names).
- **What it refuses**, with the line and the reason: any other gate, an angle
  that is not a number times one parameter, a qubit measured into another
  bit, more than one register of qubits or of bits, physical qubits, and
  everything of OpenQASM 3 that is not a gate or a measurement.
- **rxx, ryy and rzz** are not in OpenQASM 3's standard library. The writer
  defines the ones a circuit uses. The reader skips gate definitions and takes
  the three names for what they say.

### The protocol

One line of JSON per message, over a stream socket; every request but `wake`
has a number, `seq`, and gets one answer with the same number and `"ok"`. A
64-bit value is a string, `"0x..."`. This is version 1.

| Request | Fields | Answer |
|---|---|---|
| `hello` | `protocol`, `session`: a name the client makes up, under which the adapter keeps its jobs | `protocol`, `backend`, `qubits`, `max_shots`, `max_jobs`, `max_circuits`, `physical`, `resumed`: the adapter knew the session |
| `prepare` | `circuit`: a name the client derives from the circuit; `qasm` | none. The adapter transpiles once and keeps the result |
| `submit` | `tag`; `entries`: for each circuit of the job `circuit`, `params`, `shots`, `key`, `attempt` | `job`. Asked again with a tag it has, the adapter answers with that job |
| `wait` | `timeout_ms`, `max` | `jobs`: those that ended and were not told yet, each `job` and `state` (`done`, `failed`, `cancelled`). Answered when there is one, when the time is up, or at `wake` |
| `wake` | none, and no `seq` | none of its own: the `wait` that is held is answered |
| `results` | `job` | `entries`: for each circuit `counts`, a list of `[bitstring, count]`, and `since_start_ns`, `since_end_ns`. The adapter forgets the job |
| `cancel` | `job` | none. The adapter forgets the job, in any state |
| `abandon` | `tag` | `found`, and `job` when there was one under that tag, which is cancelled |
| `bye` | none | none. The adapter drops the session and what it holds |

A refusal is `"ok": false` with `error` and `why`. Three errors mean something
to the client: `busy` (ask again later), `unknown_circuit` (prepare it again)
and `unknown_job`; any other refuses the request for good.

## Parameters

All are MCA parameters (`--mca device_qpu_<name> <value>`, or
`PARSEC_MCA_device_qpu_<name>` in the environment).

| Parameter | Default | What it does |
|---|---|---|
| `enabled` | 0 | QPUs in the next context; with `devices`, it only has to be set |
| `devices` | | The QPUs, when they are not all alike: see "Several QPUs" |
| `backend` | `mock` | The backend of the QPUs that `devices` does not describe: `mock` or `adapter` |
| `allow_hardware` | 0 | Let a backend that drives a quantum device open |
| `owner_rank` | 0 | The rank of the QPUs for which `devices` names none (-1: the last) |
| `max_outstanding` | 8 | Jobs at the backend of one QPU at once; the others wait, by task priority then arrival |
| `max_batch` | 1 | Tasks that may share one job: only when more tasks wait than jobs can be submitted, spread over the free places, never held back to fill one |
| `submit_interval` | 0 | Milliseconds between two submissions |
| `max_retries` | 2 | Jobs a task may have after its first |
| `job_timeout` | 0 | Milliseconds a job of one task may take (n times that for a job of n tasks); 0: no limit |
| `unavailable_timeout` | 60000 | Milliseconds a backend may take nothing before its QPU is given up on; the tasks that no other QPU can run complete as unavailable |
| `retry_backoff` | 100 | Milliseconds before a submission that was not taken is asked again (doubles, up to 30 s) |
| `poll_max` | 1000 | Longest a progress thread sleeps (each QPU has one) |
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
| `mock_submit_delay` | 0 | Microseconds it takes to answer a submission, as a provider's round trip does |
| `mock_max_circuits` | 64 | Circuits it takes in one job |
| `mock_faults` | `none` | Faults to inject: `<fault>:<rule>=<value>,...` with fault one of `fail hang unavailable lost reject noresult duplicate late cancelled`, rule one of `p first every key for` |
| `mock_threads` | 2 | Helper threads that sample the circuits |
| `mock_max_qubits` | 24 | Widest circuit it takes (at most 30; 2^(n+4) bytes per helper) |

The adapter backend:

| Parameter | Default | What it does |
|---|---|---|
| `adapter_address` | | Path of the Unix-domain socket the adapter listens on; a QPU of `devices` may name its own with `address=` |
| `adapter_timeout` | 60000 | Milliseconds the client waits for the answer to one request before it takes the adapter for out of reach and connects again (`timeout=` in `devices`) |

`device_show_statistics 1` prints one line per QPU (tasks, jobs, failures,
retries, the most jobs outstanding, the jobs that carried several tasks, and
the mean time of a job) and one for the rank (the tasks in all, the most that
waited, and those given to another QPU).

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
the timestamps of every job and update. With several QPUs the blocks' jobs go
to whichever QPU of their rank has a place; `-Q` makes every block name its
own QPU. With QPUs on several ranks, the blocks' quantum tasks go to those
ranks in turn.

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
