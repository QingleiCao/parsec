# PaRSEC-Q: QPU support and distributed QAOA

Design plan and implementation log. The plan was written on 2026-11-21 in three stages: stage 1 (source audit and integration decision, section 1), stage 2 (architecture and application design, sections 2 and 3), and stage 3 (tests, novelty, evaluation, milestones and recommendation, sections 4 to 8). Stage 3 corrected a few statements of the earlier sections; each correction is marked where it applies. Section 9 records what was built, milestone by milestone, and where it departs from the plan. Section 10 revises the design after M8 (several QPUs per context, QPUs coupled through NVIDIA NVQLink, Qiskit). Of it, several QPUs per context are built (M9, section 9.8); the rest is not.

All file and line references are pinned to commit `09c721868` on branch `qinglei/hybrid_HPC_QC`, except in section 10, whose references are to the branch head `80d6aa45d`. Nothing in PaRSEC was built or run for sections 1 to 8; there, every "verified" statement about the source means the code was read at the cited lines. Section 9 reports builds and runs.

## Project facts

- Repository: a checkout of https://github.com/ICLDisco/parsec, branch `qinglei/hybrid_HPC_QC`.
- Base commit: `09c721868`.
- Build and test machines: methane, leconte and guyot (GPU nodes, CUDA-aware Open MPI).
- QPU access: UNKNOWN (provider, plan, and allocation in QPU seconds or jobs per month).
- Network: UNKNOWN (whether compute nodes can reach the provider's endpoint, or only login nodes). For a QPU coupled to the cluster through NVQLink, the QPU is on the cluster's own network (section 10.2).
- ISC submission deadline: UNKNOWN as a project fact. The ISC 2027 website gives abstract December 14 and full paper December 21, 2026 (section 7.1); to be confirmed. (2026-11-26: set aside. Section 10 is not scoped by it.)
- Destination: UNKNOWN (a pull request to ICLDisco/parsec, or a research branch on the fork).
- This project is separate from the earlier state-vector simulator work and does not reuse or depend on that code.

Goal: add QPUs to PaRSEC as a third kind of execution device next to CPUs and GPUs, and demonstrate it with a distributed QAOA solver for weighted MaxCut. The target is an ISC paper whose contribution is the runtime. The design should later carry SQD, VQE, quantum embedding and quantum Krylov methods.

## What "first-class" has to mean

1. No worker thread blocks while a quantum job is queued or running. A job can be outstanding for seconds, minutes, or longer.
2. A task class declares a QPU incarnation, and the runtime selects it the way it selects CPU and GPU incarnations, in both PTG and DTD.
3. The runtime, not the application, limits how many jobs are outstanding per QPU and tracks QPU load and observed latency.
4. A pending QPU job keeps its taskpool, its data and the device alive. Termination detection stays correct without application-side counters.
5. QPU tasks appear in device statistics and profiling traces like tasks on any other device.
6. A failed, timed-out or cancelled job has a defined path to the application.
7. A build without QPU support behaves exactly as it does today, and so do the CUDA, HIP and Level Zero paths.

---

# 1. Source audit and integration decision

## 1.1 Open questions

Project facts still unknown, and what waits on them:

| Fact | Decisions that depend on it |
|---|---|
| QPU access | Which provider adapter to write, whether a real-hardware run fits the allocation, the block sizes and shot counts of the QAOA demonstration. |
| Network | Where the provider adapter runs. If compute nodes cannot reach the provider, the adapter is a service on a login node and the owning rank talks to it, which changes the "one rank owns the QPU" design in stage 2. |
| ISC deadline | How many of milestones M0 to M8 fit, and whether the scheduling optimization (M7) is in the first submission. |
| Destination | How invasive the changes may be. An upstream PR argues for the smallest change to `scheduling.c`, the DTD task-class layout and `jdf2c.c`; a research branch can change them more freely. |

Technical questions the audit raised and could not settle from the source:

1. Is a second non-worker thread that owns an execution stream acceptable to the maintainers? The communication thread is the only precedent (section 1.5).
2. What is the upper bound on a measurement result's size? Flow data goes through fixed-layout arenas, so a result needs a bound, for example `shots x n` bits or a capped number of distinct bitstrings. Assumed solvable; not checked against the arena code.
3. On ranks that do not own the QPU, should a QPU device module exist at all? Without one, the generated PTG startup drops the QPU incarnation on that rank and warns for every task class left with none (`parsec/interfaces/ptg/ptg-compiler/jdf2c.c:4511-4514`, `parsec/parsec.c:2334-2338`).
4. How should the `user_trigger` termination detector behave with a job pending? It terminates when the user says so, not when the counters reach zero, so nothing in it protects a pending job (not read in full).

## 1.2 Feasibility

| Area | Existing support | Required changes | Risks | Relevant files |
|---|---|---|---|---|
| Device abstraction | MCA component and module framework with attach, detach, taskpool registration and a per-device submit entry (`device.h:157-200`). Per-taskpool device mask (`parsec_internal.h:132`, 16 bits, so at most 16 devices per process). Device statistics (`device.c:521-717`). | A type bit and a name (`device.h:64-73`, `device.c:93-103`). A new component directory. Fix `parsec_mca_device_is_gpu`. Keep a device with no flop rate out of the capacity weights (`device.c:861-890`). | The template component is stale and not built (section 1.3). There is no working example in the tree of a device that is neither a CPU nor a GPU. | `parsec/mca/device/device.{h,c}`, `parsec/mca/device/template/`, `parsec/mca/device/cuda/` |
| Async execution | `PARSEC_HOOK_RETURN_ASYNC` is honoured by the scheduler (`runtime.h:145`, `scheduling.c:218`, `:229`, `:599-602`, `:614-617`). One completion entry point (`scheduling.c:523-567`). Three existing asynchronous users: GPU manager, recursive calls, the PTG-to-DTD PINS module. | A progress thread with its own registered execution stream. No change to the completion routine. | Profiling streams and the private `next_task` slot are not thread-safe (`scheduling.c:330-333`, `:397-399`). | `parsec/scheduling.{c,h}`, `parsec/recursive.h`, `parsec/remote_dep_comm.c` |
| Scheduling | Device selection by estimated completion time in nanoseconds, with a per-class `time_estimate` (`device.c:122-131`, `:283-306`; `parsec_internal.h:285`, `:443`). | Load is tracked for GPU types only (`scheduling.c:151-155`, `:534-537`). A non-GPU device cannot win a load comparison after the first pick (`device.c:299`), so several QPUs would not be balanced. | The model is "nanoseconds of work"; queue latency fits only if the QPU device supplies its own estimate. | `parsec/mca/device/device.c`, `parsec/scheduling.c` |
| Dependencies | For a non-GPU incarnation the scheduler switches every input to its host copy and pins it (`scheduling.c:156-191`), keeps the pins across ASYNC (`:218-220`), and drops them at completion (`:548`). Remote delivery keeps data on the host for a class with no GPU incarnation (`remote_dep_comm.c:902-904`). | None in the core. The PTG hook generated for a non-CPU type does not bump the version of written flows (`jdf2c.c:7081-7108`), so a QPU hook has to. | Result size needs a bound (open question 2). | `parsec/scheduling.c`, `parsec/remote_dep_comm.c`, `jdf2c.c` |
| Termination | A task leaves the task count only in `release_task` (`interfaces/interface.c:30-38`, `interfaces/dtd/insert_function.c:2051`), which `__parsec_complete_execution` calls last (`scheduling.c:562`). The local detector fires only when the pending-action count reaches zero (`termdet_local_module.c:237-259`); four-counter goes idle only when both counts are zero (`termdet_fourcounter_module.c:440`). The detector holds a reference on the taskpool from READY to termination (`termdet_local_module.c:179`, released at `:165`). | None. | `user_trigger` (open question 4). The thread that completes the last task runs the taskpool's completion callback (`scheduling.c:260-267`). Device close runs after the workers are joined (`parsec.c:1313`, `:1345`). | `parsec/mca/termdet/`, `parsec/parsec.c` |
| Error handling | None that is not fatal. A hook returning ERROR, DISABLE or NEXT at the top level ends in `parsec_fatal` (`scheduling.c:603-610`). GPU device failure is "treated as fatal" (`device_gpu.c:6078-6105`). The only recovery is retiring a declined GPU incarnation and rescheduling (`device_gpu.c:5861-5876`). | A QPU failure must not travel through hook return codes. | No precedent for a task that fails without ending the run. | `parsec/scheduling.c`, `parsec/mca/device/device_gpu.c` |
| Build system | Components are discovered by directory, validated by `ValidateModule.CMake`, and skipped by a `.ignore` file (`parsec/mca/CMakeLists.txt:9-37`). `PARSEC_HAVE_DEV_*_SUPPORT` is set in `parsec/mca/device/CMakeLists.txt:12-22` and exported through `parsec/include/parsec/parsec_options.h.in:147-153`. Device tests live in `tests/runtime/cuda/` and are wired through `tests/runtime/{CMakeLists.txt,Testings.cmake}`. | A configure option, a `ValidateModule.CMake`, one `#cmakedefine`, a `tests/runtime/qpu/` directory. | `device_gpu.c` is compiled only when a GPU backend is enabled (`parsec/mca/device/CMakeLists.txt:3-5`), so QPU code cannot call into it. | `parsec/mca/CMakeLists.txt`, `parsec/mca/device/CMakeLists.txt`, `tests/runtime/` |

**Conclusion.** Native QPU support is feasible. The asynchronous path, termination accounting and data handling already do what a QPU needs. The work is a new thin device component, a progress thread, and about a dozen places where "not a GPU" currently means "a CPU".

## 1.3 Device type bit, and the state of the template component

**Take bit 5 (`0x20`).** Verified:

- Bits 0 to 4 and bit 7 are taken; `PARSEC_DEV_ANY_TYPE` and `PARSEC_DEV_ALL` are `0x3f` (`device.h:64-73`).
- Every walk over a class's incarnations stops at the first entry whose type has no bit inside `PARSEC_DEV_ANY_TYPE` (`device.c:167`, `parsec.c:2312`, `parsec.c:2326`). A type on bit 6 would be read as the end of the list, silently cutting off that incarnation and all after it. Bit 5 needs no mask change.
- With bit 5 the name `all` covers QPUs (`device.c:101`). A `qpu` entry belongs in the same table, which is read when a chore specification of the form `task_class:type` is parsed (`parsec.c:2408`).
- `PARSEC_DEV_MAX_NB_TYPE` (7) and the DTD incarnation arrays sized from it (`insert_function.c:2282`) are unaffected. The default chore mask is `0x3f` used as a chore-index mask (`insert_function.c:2857`, `jdf2c.c:3199`), which covers six incarnations: enough for the six types.
- After this, bit 6 is the only free type bit.

**`parsec_mca_device_is_gpu` is wrong for any new type.** It returns `PARSEC_DEV_RECURSIVE < dev->type` (`device.c:1206-1210`), so a QPU would be reported as a GPU. Its three callers test the device that holds a data copy (`scheduling.c:168`, `data.c:763`, `insert_function.c:2451`). A QPU holds no copies, so none of them would fire, but the predicate should become `PARSEC_DEV_IS_GPU(dev->type)`. (Done in M3, with the type bit and the name; section 9.4.)

**The template component cannot be used as is.** It carries a `.ignore` file, so it is not built, and it would not compile: `device_template_module.c:160-161` assigns `transferred_data_in` and `transferred_data_out`, which are not fields of `parsec_device_module_t` (`device.h:157-200`). Its structure (component open, query, register, close; module attach and detach) is still the right skeleton. Note also that it sets every flop rate to zero (`device_template_module.c:173-176`), and `parsec_mca_device_registration_complete` divides by `gflops_fp64` for every non-recursive device (`device.c:887`).

## 1.4 The nine GPU checks outside `parsec/mca/device/`

| # | Site | What it does | For a QPU |
|---|---|---|---|
| 1 | `scheduling.c:151` | Adds the task's load to the device for GPU types. The `else` branch switches inputs to host copies and pins them. | **Needs to change.** A QPU wants both: load accounting and the host-input branch. The `if`/`else` has to be split. |
| 2 | `scheduling.c:233` | The scheduler counts executed tasks for non-GPU devices when the hook did not return ASYNC. | Applies as is. A QPU hook returns ASYNC, so nothing is counted here and the device counts at completion, as GPUs do. |
| 3 | `scheduling.c:534` | Removes the load at completion for GPU types. | **Needs to change**, together with 1. |
| 4 | `remote_dep_comm.c:902` | A successor class with no GPU incarnation receives its data in host memory. | Applies as is, and is what a QPU needs. |
| 5 | `parsec.c:3242` | The peer-access query accepts only GPU types. | Does not apply. |
| 6 | `insert_function.c:1643` | At DTD enqueue, GPU devices are skipped; every other device has `taskpool_register` called, and a failure removes the device from the taskpool (`:1644-1649`). | **Needs to change**, or the QPU's `taskpool_register` must always succeed. The static registration returns "not found" when no class has a matching incarnation (`device.c:1060`, `:1092-1096`), which is the state of a DTD taskpool before tasks are inserted (inferred, not run). |
| 7 | `insert_function.c:2553` | Adding a non-GPU chore overwrites the class's single `cpu_func_ptr` and installs the CPU submit wrapper. | **Needs to change.** A class with a CPU chore and a QPU chore would lose its CPU body. It needs its own function slot and submit wrapper. |
| 8 | `insert_function.c:3480` | Direct insertion with a non-GPU type installs the user function as the hook. | Works as a pass-through, but bypasses the device. **Needs to change** for the device to mediate. |
| 9 | `jdf2c.c:6835` | An assert inside the GPU hook generator. | Does not apply. That generator is used only for `cuda` and `hip` bodies (`jdf2c.c:7014-7016`). |

(As built in M3, section 9.4: checks 1 and 3 changed as described; check 6 is met by a `taskpool_register` that always succeeds; checks 7 and 8 did not have to change for a task class whose only chore is of the QPU type, which is what the DTD test uses, and stay open for a class with several incarnations.)

Three related checks inside the device layer matter as well:

- `device.c:299`: once a first device is picked, a non-GPU device is skipped unless CPU load balancing is allowed. With two QPUs, only the one with the higher index would ever be chosen. Needs to change for more than one QPU. (Section 10.3 avoids the change: the QPU component binds a task to a QPU itself, when that QPU has a free place.)
- `device.c:231` and `:255`: data-ownership affinity, for GPU types only. Does not apply.
- `device.c:363`: batched chores are limited to GPU types. Relevant if stage 2 picks submission batching as the optimization.

## 1.5 Who completes a QPU task

**What exists.** `__parsec_complete_execution(es, task)` is the only correct way to finish a task whose hook returned ASYNC. In order, it removes the device load, runs `prepare_output`, drops the input pins, runs the class's `complete_execution` (which releases the dependencies and schedules the successors), and calls `release_task` (which returns the task to its mempool and decrements the task count) (`scheduling.c:523-567`). It does not read `task->status`.

It has five callers. Two are synchronous: the scheduler after a hook returns DONE (`scheduling.c:588`) and immediate tasks (`parsec.c:1847`). Three complete a task whose hook returned ASYNC:

- The GPU manager (`device_gpu.c:6029`). The first worker to submit to an idle device becomes the manager (`:5739-5755`) and stays in the progress loop until nothing is outstanding (`:6060-6075`). The loop has no sleep (`:5942-6002`).
- Recursive calls (`recursive.h:25-42`): whichever PaRSEC thread detects the inner taskpool's termination completes the generator task, using `parsec_my_execution_stream()`.
- The PTG-to-DTD PINS module, which returns ASYNC at `pins_ptg_to_dtd_module.c:392` and `:399` and completes the original task at `:204`.

**The GPU manager model does not fit a QPU.** A worker would spin for as long as a job is queued, which breaks property 1 and wastes a core per QPU. The stage-in, execute, stage-out pipeline it drives is about device memory. And the file is not compiled without a GPU backend.

**There is no device progress hook in the worker loop.** A waiting worker only selects tasks and, for the master on a single node, progresses communications (`scheduling.c:737-763`). Polling from workers would mean a new hook in the generic scheduler. Polling with a task that returns `PARSEC_HOOK_RETURN_AGAIN` is a busy wait: the task is demoted and rescheduled at once (`scheduling.c:590-598`).

**A non-worker thread with its own execution stream is supported, by one precedent.** The communication thread owns `parsec_comm_es` (`remote_dep_comm.c:252-262`), initialised with `scheduler_object = NULL` and a non-NULL sentinel in `next_task` (`:524-534`), borrowing the mempools of thread 0 of virtual process 0 (`:577-583`), with its own profiling stream (`:548-549`), and registered in thread-local storage (`:2559`). `__parsec_schedule_vp` recognises such a stream by `scheduler_object == NULL` and delivers tasks to stream 0 of each virtual process instead of the caller's queues (`scheduling.c:409-425`).

**Answer.** A dedicated QPU progress thread, owning an execution stream built the way the communication thread's is, completes QPU tasks. Two mechanisms are available:

- **(a) Direct.** The progress thread calls `__parsec_complete_execution(&qpu_es, task)`.
- **(b) Hand-off.** The progress thread writes the result into the task's output copy, sets `task->status = PARSEC_TASK_STATUS_COMPLETE`, and pushes the task to stream 0 with `__parsec_schedule`. A worker picks it up, and `__parsec_task_progress` goes straight to completion because the status is past the hook stage (`scheduling.c:575-589`).

I recommend (b) for the first implementation. All dependency release, mempool use and instrumentation then run on a real worker stream, and the progress thread needs only a minimal stream. The cost is one scheduler round trip plus the idle back-off, which is at most 64 x 5410 ns, about 0.35 ms (`parsec/utils/backoff.h:17-35`), against job latencies of seconds. No existing caller relies on this path, so it needs a dedicated test in M1. (Corrected in stage 3: rescheduling a task that returned ASYNC does have a precedent in `tests/profiling/async.jdf`; see the note in section 4.3 for what the hand-off adds to it.)

**Invariants a QPU completion path must respect** (verified unless marked):

1. The hook returns ASYNC once, and the worker does not touch the task afterwards (`scheduling.c:599-602`). (Stage 3: one exception, the instrumentation that runs right after the hook, `scheduling.c:221-226`. It is hazard H1 of section 4.3.)
2. Inputs stay pinned until `__parsec_complete_execution` unpins them (`scheduling.c:218-220`, `:548`). Completion must go through that function, never straight to the dependency release.
3. `__parsec_complete_execution` runs exactly once per task.
4. A thread uses only its own execution stream. The `next_task` slot is unprotected (`scheduling.c:397-399`), profiling streams are not thread-safe (`:330-333`), and instrumentation dereferences the stream it is given (`mca/pins/pins.c:36`), so the progress thread must register a stream in thread-local storage even for mechanism (b).
5. After completion returns, the caller must not touch the task or its taskpool: dropping the last pending action runs the termination callback and releases the detector's reference (`termdet_local_module.c:145-166`).
6. The device module must outlive every task it holds. At shutdown the device components are closed after the workers are joined (`parsec.c:1313`, `:1345`), so the component's close must stop the progress thread and cancel or drain what is pending.

**Public or internal.** `parsec_my_execution_stream` is public (`runtime.h:610`). `__parsec_complete_execution`, `__parsec_schedule` and `__parsec_reschedule` are declared in `scheduling.h`, which is installed (`parsec/CMakeLists.txt:346`), but they are internal by name and none is documented as callable from a thread PaRSEC did not create. `parsec_set_my_execution_stream` is internal (`parsec_internal.h:784`). The contexts known to be safe are a worker, the GPU manager (which is a worker), and the communication thread with its own stream.

## 1.6 PTG and DTD

**PTG.** `BODY [type=QPU]` already compiles to an incarnation of type `PARSEC_DEV_QPU` guarded by `PARSEC_HAVE_DEV_QPU_SUPPORT` (`jdf2c.c:3968-4009`). Because the type is neither `cuda` nor `hip`, the hook comes from the generic generator (`jdf2c.c:7014-7122`): the body runs inline on the worker and may return ASYNC itself. So a QPU body works today as "user code that submits and returns ASYNC". For the device to mediate, either the body calls a submit helper by convention, or `jdf2c.c:7014` gains a third branch that generates the submit. The generic hook skips the version bump on written flows for types other than CPU and RECURSIVE (`jdf2c.c:7081-7108`).

**DTD.** A task class has one CPU function slot and one GPU function slot (checks 7 and 8 in section 1.4), and the GPU slot's wrapper builds a GPU task and calls `device->kernel_scheduler` (`insert_function.c:2404-2430`). A QPU needs a third slot and wrapper. `parsec_dtd_task_class_add_chore` already validates a type against the devices enabled for the taskpool (`insert_function.c:2478-2504`, `:2524-2528`), which works for a new type without change.

**Dynamic generation.** A taskpool stays alive while it has tasks or pending actions (section 1.7), and DTD inserts tasks at any time, so quantum tasks can be generated while earlier ones are outstanding. Nothing in the source prevents it. The repeated parameter loop of QAOA is the kind of graph DTD is built for; the PTG-or-DTD decision is for stage 2.

**Results as outputs.** A QPU task's result is ordinary host data on an output flow. The progress thread writes it into the output copy before completion. Verified only as far as the pinning and host-copy handling in `scheduling.c:156-191`; the arena side is open question 2.

## 1.7 Termination and shutdown

Verified for the local and four-counter detectors: a task whose hook returned ASYNC stays in the taskpool's task count until `__parsec_complete_execution` reaches `release_task`. While the count is non-zero the taskpool is busy, on every rank that holds such a task. No runtime-action counter needs to be touched by the QPU code, and the detector's own reference keeps the taskpool object alive (references in the table in section 1.2).

Consequences:

- A job outstanding for minutes simply keeps the taskpool busy. `parsec_context_wait` returns only when no taskpool is active (`scheduling.c:255-258`).
- MPI progress is unaffected: it runs on the communication thread, or on the master in single-node runs (`scheduling.c:739-746`).
- If the QPU task is the last one, the thread that completes it runs the taskpool's `on_complete` callback (`scheduling.c:260-267`). With mechanism (b) that is a worker, as it is today.
- `parsec_fini` joins the workers before closing device components (`parsec.c:1313`, `:1345`). A job still pending there means the application did not wait for its taskpool. The component must cancel it and must not complete tasks into a scheduler that is gone.

Not verified: `user_trigger` (open question 4), and whether the four-counter detector may send its termination messages from a thread that is neither a worker nor the communication thread. Mechanism (b) avoids the second question.

## 1.8 Build and tests

The smallest integration that follows existing conventions:

- `parsec/mca/device/qpu/` with `ValidateModule.CMake` modelled on `parsec/mca/device/cuda/ValidateModule.CMake`, turned on by a configure option next to `PARSEC_GPU_WITH_*` (`CMakeLists.txt:195-201`).
- `PARSEC_HAVE_DEV_QPU_SUPPORT` set in `parsec/mca/device/CMakeLists.txt` and exported in `parsec/include/parsec/parsec_options.h.in` beside the other device lines (`:147-153`). The PTG compiler already keys generated code on that macro.
- Backend-specific code (mock, provider adapter) inside the component directory, with no new mandatory dependency.
- Tests in `tests/runtime/qpu/` with their own `CMakeLists.txt` and `Testings.cmake`, added to `tests/runtime/CMakeLists.txt:5-6` and `tests/runtime/Testings.cmake:5-6`, using `parsec_addtest_executable` and `parsec_addtest_cmd` (`tests/CMakeLists.txt:36-98`). `MPI_TEST_CMD_LIST` is the pattern for multi-rank tests (`tests/runtime/Testings.cmake:9`).

## 1.9 Integration options against the seven properties

- **A. GPU-style device**: QPU chores go through `parsec_device_kernel_scheduler`.
- **B. Thin device component**: its own type, module, submit entry, job queue and progress thread.
- **C. No device**: a CPU-typed incarnation that returns ASYNC, plus a completion service in a library.

| Property | A. GPU-style device | B. Thin device | C. CPU chore plus service |
|---|---|---|---|
| 1. No worker blocks | No. The manager is a worker that spins while anything is outstanding. | Yes. | Yes. |
| 2. Selected as an incarnation | Yes. | Yes, with the DTD and PTG changes in 1.4 and 1.6. | No. It is a CPU chore; the runtime cannot tell it from any other, restrict it by type, or offer a CPU fallback in the same class. |
| 3. Runtime-enforced capacity and load | Load yes; capacity would have to be added. | Yes. The device owns the queue; selection across QPUs reuses `parsec_select_best_device`. | No. Both live in the service, outside the runtime. A CPU-only task takes a shortcut that sets its load to zero (`device.c:198-206`). |
| 4. Lifetime and termination | Yes. | Yes. | Yes. The accounting is the same. |
| 5. Statistics and traces | Yes. | Yes, with QPU events for submit, queued, running and result. | No. The task returns ASYNC, so the scheduler does not count it for any device (`scheduling.c:229-237`), and its execute interval closes when the hook returns (`scheduling.c:212-226`). |
| 6. Failure path | No. Device failure is fatal. | To be designed in the device. | To be designed in the service. |
| 7. Unchanged without QPU | No. Depends on `device_gpu.c`, which needs a GPU backend. | Yes. | Yes. |

**Recommendation: B, reached through C's mechanism.** The job manager and completion path of C are the lower half of B. So M1 builds the progress thread, the hand-off completion and the mock backend behind a CPU-typed chore and tests them alone; M3 puts the device component, type bit, selection and DTD and PTG support on top. This keeps the proposed order of M1 before M3.

Properties 2, 3 and 5 are the ones that cannot be had without a real device. They are the argument for one: the runtime chooses the quantum incarnation, bounds and balances jobs across QPUs, and shows them in its own statistics and traces. An asynchronous task alone gives none of these.

**Failure handling, as a starting point for stage 2.** Keep QPU failures out of hook return codes, since those are fatal. The device retries a bounded number of times, then completes the task normally with a status in its result, so the graph continues and the application decides. Falling back to another incarnation by clearing the chore and calling `__parsec_reschedule` (`scheduling.c:499-521`) is possible for classes that have one.

**What would change the recommendation.**

- If a new device type or a second non-worker thread is not acceptable upstream, fall back to C, state plainly that properties 2, 3 and 5 are then met in a library, and reconsider what the paper claims.
- If the hand-off path (b) fails its M1 test, use direct completion (a), which needs the fuller execution stream the communication thread has.
- If the network fact forces the adapter onto a login node, the device on the owning rank becomes a client of that service. The recommendation stands, but shutdown and failure handling grow.

## 1.10 What was not verified

- Nothing was compiled or executed.
- That a DTD taskpool has no task classes when it is enqueued (check 6) is inferred from how DTD creates classes at insertion.
- That another thread may safely share thread 0's mempools rests on the communication thread doing so, not on reading the mempool code.
- The hand-off completion path (b) follows from the control flow of `__parsec_task_progress`; no existing code uses it. (Stage 3: `tests/profiling/async.jdf` reschedules a task that returned ASYNC, from a worker and with the hook running again; the hand-off itself is still untested.)
- `device_gpu.c` was read only around the manager loop; the HIP and Level Zero components and most of `data.c` were not read.
- `user_trigger` was not read in full.
- Whether a taskpool can be cancelled was not searched.

---

# 2. QPU architecture

This section follows the approach approved in stage 1: a thin device component with its own progress thread, completing tasks by hand-off to a worker. Source references are verified at the base commit unless marked as assumed.

## 2.1 Overview

```text
  worker threads                    QPU device component (one per process)
 +----------------+         +---------------------------------------------------+
 | task selected  |         |  module per QPU: capacity, load, statistics       |
 | QPU incarnation|         |                                                   |
 | chosen by      | submit  |  pending heap  --admission, coalescing-->  in     |
 | select_best_   |-------->|  (priority,                                flight |
 | device         | ASYNC   |   then arrival)                              |    |
 +----------------+         |        ^                                     |    |
         ^                  |        | retry                               v    |
         |                  |  progress thread  <----- poll / wait ---- backend |
         | task with status |        |                                  vtable  |
         | COMPLETE, pushed |        | write result into the output copy   |    |
         | to the scheduler |        v                                     |    |
         +------------------+--- hand-off                                  |    |
                            +----------------------------------------------|----+
                                                                           |
                                              +----------------------------+-----+
                                              | mock backend     | adapter client |
                                              | (in process)     | (socket)       |
                                              +------------------+-------+--------+
                                                                         |
                                                              adapter process (per
                                                              provider) --> QPU
```

Only the rank that owns a QPU opens its backend (section 2.7). (Revised in section 10.3: a context may have several QPUs, each with its own owner rank and its own service.)

## 2.2 Device, submission and progress

**Registration.** The component follows the open, register, query, close sequence of the existing components (`parsec/mca/device/template/device_template_component.c:83-189` for the shape, `parsec/mca/device/cuda/` for a live example). It is inactive unless `device_qpu_enabled` is set, so a QPU-enabled build behaves like today's by default. Each module registers with type `PARSEC_DEV_QPU` through `parsec_mca_device_add` (`device.c:1172-1197`) and supplies a `taskpool_register` that always succeeds (section 1.4, check 6). It takes no part in the flop-rate weights: the loop in `device.c:861-890` has to skip QPU types, since it would otherwise divide by a zero rate at `:887`. (As built in M3: the loop skips any device that declares no flop rate, and the backend is opened when the module is attached, so a backend that cannot be used gives no device; section 9.4.)

**Capability discovery.** The backend reports number of qubits, maximum shots, maximum concurrent jobs, whether one job may carry several circuits, and whether it is a physical device. The module stores them, prints them under `device_show_capabilities`, and exposes them through a small query function (`parsec_qpu_device_info`, section 9.4). Counting QPUs works today with `parsec_context_query(ctx, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_QPU)` (`parsec.c:3228-3235`).

**Submission.** A QPU incarnation is a hook that describes the job and returns ASYNC. One helper does it for PTG and DTD:

```c
parsec_hook_return_t
parsec_qpu_submit(parsec_execution_stream_t *es, parsec_task_t *task,
                  int circuit_flow, int params_flow, int result_flow, int shots);
```

(As built in M1, the three flows are passed as data copies and a JDF body uses the macro `PARSEC_QPU_SUBMIT`; see section 9.1. M2 added a last argument, the key of the evaluation; see section 9.2.)

It reads `task->selected_device`, which `parsec_select_best_device` has already set (`scheduling.c:149`), bumps the version of the result flow (section 1.6), pushes a job record onto the device's lock-free submission list, wakes the progress thread, and returns `PARSEC_HOOK_RETURN_ASYNC`. In a JDF, `BODY [type=QPU]` contains only this call, so no change to the PTG compiler is needed. The hook does no network or backend work.

**Progress mechanism.**

| Mechanism | Thread safety | Fit with the runtime | Complexity | Verdict |
|---|---|---|---|---|
| Existing device-manager machinery | Proven for GPUs | A worker spins while anything is outstanding; needs a GPU build (section 1.5) | High: a pipeline about device memory | Rejected |
| Polling from workers with a rescheduled task | Safe | A busy wait (`scheduling.c:590-598`); occupies cores | Low | Rejected |
| Dedicated thread, fixed-interval polling | Needs its own execution stream | No change to the scheduler | Low | Works, but either wastes calls or adds latency |
| Dedicated thread, backend-driven wait | Same | Same | Low to medium | **Chosen** |

The progress thread sleeps in one call, `backend->wait(timeout)`, which returns when a job may have changed state or when a submission arrives. The mock implements it with a timer heap, so wake-ups are exact. A provider that offers no notification implements it as a poll whose interval backs off from a minimum to a maximum (both MCA parameters) and resets when a job changes state. The thread never spins.

One progress thread serves all QPU modules of a process. It owns a minimal execution stream registered in thread-local storage, as the communication thread does (section 1.5). (Revised in section 10.3: one progress thread per QPU, since a thread cannot sleep in the `wait` of two backends and one slow provider would delay the others. The QPUs of a rank share their waiting tasks.)

**Admission.** A module holds at most `max_outstanding` jobs at its backend, the smaller of the backend's limit and an MCA parameter, and spaces submissions by a minimum interval where the provider limits the submission rate (section 2.6). The rest wait in the module's heap, ordered by task priority and then arrival. This is where the runtime, not the application, bounds submissions (property 3).

**Finalization.** The component's close runs after the workers are joined (`parsec.c:1313`, `:1345`). It stops admission, cancels every job still at the backend, reports how many tasks were abandoned (non-zero means the application did not wait for its taskpool), joins the progress thread and closes the backends. Nothing is completed into the scheduler at that point.

## 2.3 Task lifecycle

```text
READY --(worker: select device, hook)--> PENDING --(admission)--> SUBMITTED
                                            ^                         |
                                            |                 QUEUED / RUNNING at the backend
                                 retry, attempts left                 |
                                            |          +--------------+---------------+
                                            +--------- FAILED / TIMED OUT          DONE
                                                             |                        |
                                                   attempts exhausted                 |
                                                             v                        v
                                              result written with a failure   result written
                                              status                                  |
                                                             +-----------+------------+
                                                                         v
                                          HANDED OFF (status COMPLETE, pushed to the scheduler)
                                                                         |
                                             worker: __parsec_complete_execution
                                                                         v
                                                        dependencies released
```

| State | Who owns the task | What keeps things alive |
|---|---|---|
| READY | The scheduler | Task count of the taskpool |
| PENDING, SUBMITTED, QUEUED, RUNNING | The device, exclusively. The worker that ran the hook does not touch it again (`scheduling.c:599-602`), except as described under H1 in section 4.3. | Inputs stay pinned by the task (`scheduling.c:218-220`). The task count keeps the taskpool busy and the detector's reference keeps it allocated (section 1.7). The circuit and parameters are read in place from the pinned input copies. The device load includes the task. |
| HANDED OFF | The scheduler, then one worker | Same, until completion |
| Completed | Nobody | `__parsec_complete_execution` unpins the inputs, releases the dependencies and decrements the task count (`scheduling.c:523-567`) |

Only the progress thread writes the result copy, and only before the hand-off. The job record belongs to the device and is freed at hand-off.

**Timeout.** A job that exceeds `job_timeout` is cancelled at the backend and treated as failed.

**Cancellation.** There is no taskpool cancellation to hook into (not searched exhaustively, section 1.10). Jobs are cancelled only on timeout and at finalization.

## 2.4 Inputs and outputs

A quantum task uses ordinary flows and host copies. No new data-management layer is needed.

| Flow | Access | Content |
|---|---|---|
| Circuit | read | A flat, pointer-free buffer: a header (version, qubits, operations, parameter slots, flags) and an array of operations (opcode, two qubit indices, and either an immediate angle or a parameter slot with a coefficient). The gate set is the standard one- and two-qubit rotations, Hadamard, CNOT, CZ and measurement. A flag marks an empty job. |
| Parameters | read | An array of doubles bound to the slots. |
| Result | write | A header (status, qubits, shots requested and returned, attempts, timestamps for submission, start, end and retrieval, entry count) and an array of (bitstring as a 64-bit word, count). |

Shots and the backend-independent options travel as task parameters.

The circuit structure and its parameters are separate flows so that one circuit copy serves every evaluation that differs only in angles.

The representation is a small native format instead of OpenQASM text because the mock can interpret it directly and each backend can lower it to what its provider accepts. The cost is one translation function per backend. (Section 9.2 has the format as built in M2: the operation layout, the thirteen operations, what is checked.)

**Result size.** Flow data comes from fixed-size arenas whose byte size and datatype are set per taskpool with `parsec_arena_datatype_set_type` (`parsec/data.h:163`; used for example in `tests/dsl/ptg/ptgpp/write_check.jdf:129-132`), and a write-only flow receives a fresh copy from that arena. The result arena is sized for `shots` entries, since a job cannot return more distinct bitstrings than shots: 64 bytes plus 12 per entry, about 12 kB for 1024 shots. The 64-bit bitstring limits a register to 64 qubits, which is enough for QAOA blocks; a wider format is future work.

## 2.5 Backend interface and the mock

```c
typedef struct parsec_qpu_backend_s parsec_qpu_backend_t;

typedef struct parsec_qpu_backend_ops_s {
    const char *name;
    int (*open)(parsec_qpu_backend_t **be, int device_index);
    int (*capabilities)(parsec_qpu_backend_t *be, parsec_qpu_caps_t *caps);
    /* One job carrying nb >= 1 circuit instances. tag is unique per job. */
    int (*submit)(parsec_qpu_backend_t *be, const parsec_qpu_request_t *req, int nb,
                  uint64_t tag, parsec_qpu_job_id_t *id);
    /* Sleep until a job may have changed state, a wake-up is requested, or the
     * timeout expires. Reports the jobs that reached a final state. */
    int (*wait)(parsec_qpu_backend_t *be, int64_t timeout_ns,
                parsec_qpu_event_t *events, int max_events);
    int (*wakeup)(parsec_qpu_backend_t *be);
    /* Fill one result per circuit instance of the job. */
    int (*results)(parsec_qpu_backend_t *be, parsec_qpu_job_id_t id,
                   parsec_qpu_result_t **results, int nb);
    int (*cancel)(parsec_qpu_backend_t *be, parsec_qpu_job_id_t id);
    int (*close)(parsec_qpu_backend_t *be);
} parsec_qpu_backend_ops_t;
```

Every function is called from the progress thread only, so a backend needs no locking of its own. `wakeup` is the one exception: the submit helper calls it from a worker.

**Duplicate submissions.** The `tag` is derived from the taskpool, the task key and the attempt number. A backend whose provider supports job tags passes it along and, after an ambiguous submission failure, looks the job up by tag before submitting again.

**Mock backend.** In process, no network, no credentials.

- Each job's queue and execution delays are drawn from a configurable distribution (constant, uniform, exponential, lognormal) or replayed from a file of recorded latencies.
- A configurable number of slots; jobs beyond it wait in the mock's own queue, which models the provider's queue.
- Failure and timeout injection by probability, by period, or for chosen tags.
- Samples come from an exact state-vector evaluation of the circuit, written for this project, for registers up to about 20 qubits. The mock runs it on a helper thread of its own, so a job's completion time is the later of the modelled delay and the end of the computation.
- The random stream of a job is seeded from the global seed and the job's tag, never from a shared generator, so results do not depend on the order in which jobs are submitted or completed.
- Every result carries a flag saying it came from the mock, and all reporting keeps mock and hardware results apart.

**Guard against accidental hardware use.** A backend whose capabilities say "physical" opens only if `device_qpu_allow_hardware` is set. No test sets it.

(As built in M2, the interface differs in four ways, the mock has a queue delay and a count of jobs whose sampling outlasted their modelled time, and the guard has two layers; see section 9.2.)

## 2.6 Real backend

The facts below come from the providers' documentation, read on 2026-11-21 by a separate research pass. I re-checked a sample of them against saved copies of the pages (the usage formula, the Open Plan allowance, the ISA requirement, the job-creation rate limit, the session restriction, the Sampler deprecation). They are documentation claims, not things observed by running a job. Numbers in brackets refer to the source list at the end of this section.

**IBM Quantum**

- A free Open Plan exists: 10 minutes of QPU time per rolling 28 days [1]. It allows job and batch modes but not sessions [2].
- Jobs can be submitted over plain HTTPS: an IAM token from an API key, then a POST with a program id, a backend name and a list of entries, each a circuit, its parameter values and a shot count [3][4]. Circuits are OpenQASM 3 text or QPY [5].
- Circuits must already match the backend's native gates and connectivity ("ISA") when submitted [6].
- One job may carry many such entries, up to 10 million executions in total [7]. Jobs accept tags and can be listed by tag [5].
- Job creation over REST is limited to 5 per minute per user [5].
- Usage is the time the QPU is held for the job. The documented quick estimate is 2 s plus 0.00035 s per execution [8]. Failed jobs caused by user error and cancelled jobs are charged [8].
- The server-side Sampler program is deprecated as of qiskit-ibm-runtime 0.50.0, in favour of a client-side Sampler that submits to an "executor" program [9]. How long the old program remains, and the executor's payload, were not confirmed.
- Qiskit has a C API that covers circuits and the transpiler without Python, described as experimental; it does not submit jobs [10]. (Section 10.4: since Qiskit 2.2 it has a standalone transpiler, `qk_transpile`, and since 2.6 it reads and writes QPY.)
- QRMI is a vendor-neutral library for HPC systems with a C API to start a task, check it, fetch its result and stop it, for IBM, IQM, Pasqal and others. It takes an already prepared job input, so transpilation stays outside it [11].

**IonQ**

- REST API v0.4 with an API key header; JSON gate lists or an OpenQASM 3 subset; an abstract gate set that IonQ compiles; full connectivity, so no routing on the client; up to 5,000 circuits per job [12][13].
- Access is through prepaid credits or a research grant; no free QPU tier was confirmed [14].

**Others, briefly.** IQM Resonance and Quantinuum document Python clients; a public REST specification was not confirmed for either. AWS Braket has a REST API usable from C, with request signing and results in object storage. OLCF's quantum user program gives merit-reviewed access to IBM, Quantinuum, IonQ and IQM, with a default of 400 IBM minutes per 28 days per project [15].

**Recommendation.** Target IBM Quantum, through a persistent adapter process written in Python on the official SDK. (Revised in section 10.4: the adapter gets a Qiskit Aer mode first, and a backend in C on the Qiskit C API and QRMI is a second route, M10. The C API now transpiles and writes QPY, which removes the first of the two reasons below.)

- IBM is the only provider with confirmed free time on physical hardware, and the HPC-centre programmes sit on the same platform.
- The adapter is recommended over calling the REST API from C for two reasons. Circuits must be transpiled to the device before submission, and the supported transpiler is Qiskit. And the program a job is submitted to is changing, which the SDK absorbs. A C client today would bind PaRSEC to a deprecated program and an experimental library.
- The adapter is a separate program. PaRSEC does not link to Python, does not need it to build, and no test uses it.
- If the access you have turns out to be IonQ, a backend in C over its REST API is the simpler route, since nothing has to be transpiled. The backend interface of section 2.5 allows either.

This departs from the research pass, which suggested REST from C. The facts are the same; the difference is how much weight to give the deprecation and the transpilation requirement.

**Why not adopt QDMI or QRMI as the backend interface.** Both expose submit, status, wait, result and cancel in C, and a backend built on either can sit behind the interface of section 2.5. PaRSEC's own interface stays small and internal because it needs things neither provides (a wake-up for the progress thread, one job carrying several tasks with separate results, tags), and because QDMI states that minor releases may break compatibility [16]. QRMI is the first thing to try for a C-only IBM backend later.

**Adapter protocol.** One line of JSON per message, over a stream socket.

| Request | Reply | Notes |
|---|---|---|
| `hello` | protocol version, backend name, capabilities | |
| `prepare` (circuit id, circuit as OpenQASM 3 with declared input angles) | ok | Transpiles once and caches the result. Keeping this separate from submission is what would let transpilation become its own task (section 3.4). |
| `submit` (tag, list of circuit id, parameter values, shots) | job id | One provider job, one entry per task |
| `wait` (timeout) | list of jobs that reached a final state | |
| `results` (job id) | per entry: status, counts, timings | The adapter turns per-shot output into counts |
| `cancel` (job id) | ok | |
| `bye` | | |

- The adapter starts once and serves the whole run. It listens on a Unix-domain socket by default.
- If compute nodes cannot reach the provider (the unknown network fact), the adapter runs on a login node and listens on TCP. In that mode it accepts only clients that present a token read from a file readable by the user alone, because an open port would otherwise let anyone on the machine spend the allocation.
- Credentials exist only in the adapter's environment or in the SDK's own account file. They never cross the socket and are never logged by either side.

**What the provider's rules mean for the device.**

- A minimum interval between job submissions is needed next to `max_outstanding`, because of the rate limit.
- Coalescing maps directly onto several entries per job, and the split between a circuit and its parameter values matches the provider's own.
- Timeouts must be generous and cancellation rare, since cancelled jobs are charged.

**Hardware budget, from the documented estimate (not measured).**

| Run | Jobs | QPU time by the estimate | Against 10 minutes per 28 days |
|---|---|---|---|
| Validation instance of section 3.4, one circuit per job | 64 | 64 x (2 + 0.00035 x 1024) s, about 2.5 minutes | Fits, with room for one repeat |
| Same, two circuits per job | 32 | 32 x (2 + 0.00035 x 2048) s, about 1.5 minutes | Fits |
| Controlled instance, one circuit per job | 1024 | about 40 minutes | Does not fit; needs an allocation of the OLCF kind |

The second row assumes the 2 s overhead is paid once per job, which the documentation suggests but which has to be measured. Wall-clock time adds queueing, which is unknown, and at 5 submissions per minute the 64-job run takes about 13 minutes just to submit.

**Sources**

1. https://quantum.cloud.ibm.com/docs/en/ibm-quantum-platform/plans-overview
2. https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/run-jobs-session
3. https://quantum.cloud.ibm.com/docs/en/ibm-quantum-platform/cloud-setup-rest-api
4. https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/sampler-rest-api
5. https://quantum.cloud.ibm.com/docs/en/api/qiskit-runtime-rest/tags/jobs
6. https://quantum.cloud.ibm.com/docs/en/qiskit/transpile
7. https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/job-limits
8. https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/estimate-job-run-time
9. https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/release-notes
10. https://quantum.cloud.ibm.com/docs/en/api/qiskit-c
11. https://github.com/qiskit-community/qrmi
12. https://docs.ionq.com/api-reference/v0.4/jobs/create-job
13. https://ionq.com/best-practices
14. https://docs.ionq.com/guides/billing-and-usage/working-with-payg-credits
15. https://docs.olcf.ornl.gov/quantum/quantum_systems/ibm_quantum.html
16. https://munich-quantum-software-stack.github.io/QDMI/v1.4.0/md_CHANGELOG.html

## 2.7 Sharing a QPU across ranks

| Option | Capacity control | Duplicate jobs | New mechanism needed | Verdict |
|---|---|---|---|---|
| One owner rank per QPU. QPU tasks are placed there by affinity; inputs and results travel as ordinary remote dependencies. | In one place, by the device | Impossible by construction: a task executes on one rank | None | **Chosen** |
| Every rank submits directly | Needs a distributed token or credit protocol | No worse | Yes, plus credentials on every rank | Rejected for now |
| An external broker shared by all ranks | In the broker, outside the runtime | Depends on the broker | Yes, and it weakens property 3 | Rejected |

Details of the chosen option:

- `device_qpu_owner_rank` (default 0; `-1` for the last rank, since M3) names the owner. Every rank attaches a QPU module so that QPU incarnations are not dropped at taskpool startup (open question 3 of section 1.1), but only the owner opens a backend. On other ranks the module has zero capacity and refuses work; it is never selected, because no QPU task is placed there.
- Placement uses what exists: in PTG, the task's affinity names a tile of a one-element collection living on the owner rank; in DTD, the `PARSEC_AFFINITY` flag on a tile of that rank (`parsec/interfaces/dtd/insert_function.h:45`).
- Results reach the ranks that need them the way any flow does.
- Several QPUs are several modules on the owner rank. Selection among them is `parsec_select_best_device` with the changes of section 1.4, fed by each module's load and latency estimate. Placing QPUs on different ranks would need cross-rank load balancing, which PaRSEC does not do for any device. (Revised in section 10.3: QPUs may sit on several ranks, and the application places its quantum tasks on those ranks, so nothing is balanced across ranks. Within a rank, a task is bound to a QPU when that QPU has a free place for it, unless the task names its QPU. `parsec_select_best_device` does not change.)
- If the backend becomes unavailable, submissions are retried with back-off while pending tasks wait. After `unavailable_timeout` every pending task is completed with an "unavailable" status.

## 2.8 Scheduling

$$T_{\mathrm{QPU}} = T_{\mathrm{queue}} + T_{\mathrm{submission}} + T_{\mathrm{execution}} + T_{\mathrm{result}}.$$

Each module keeps a moving average of the four terms from its own completed jobs. The progress thread writes their sum, in nanoseconds, to the module's `time_estimate_default` (`device.h:191`). That is the value `parsec_select_best_device` uses for a task class that provides no estimate of its own (`device.c:122-131`), in the unit it already compares (`device.c:283-306`). (As built in M3, the time of a job is measured and reported, but it is not written to `time_estimate_default`, which stays a nominal second per task: section 9.4 says why. With one QPU nothing depends on it.)

**The one optimization: work-conserving job coalescing under the capacity limit.**

When a slot frees and several tasks are pending, the module packs up to `max_batch` compatible tasks into one provider job. Each task still gets its own result and completes on its own. The module never waits to fill a job: with spare capacity every task is its own job, and coalescing appears only under contention.

(As built in M7, the tasks that wait are spread evenly over the places that are free at that moment, which is this rule when one place frees at a time and keeps tasks from sharing when there is a place for each. "Compatible" is: the same number of shots, and no job yet. The time-out of a job is per task it carries. Section 9.6.)

Why this one:

- **It needs the device.** Tasks from different blocks and different ranks meet only in the device's pending heap. Neither the application nor an asynchronous task body can see them together.
- **It fits what PaRSEC already has.** Batched incarnations exist for GPUs (`device.c:359-372`), and the GPU manager groups consecutive tasks of the same incarnation (`device_gpu.c:5968-5987`). This is the same idea applied to provider jobs.
- **It is measurable.** If a job costs a fixed part (queue, submission, retrieval) and a part per circuit, coalescing removes fixed parts in proportion to contention. The mock controls both parts, so the effect can be predicted and then measured.
- **It matches the hardware.** On cloud QPUs the fixed cost per job dominates small circuits, which is why providers accept several circuits per job.

A failed coalesced job sends each of its tasks back to the heap to be retried alone. (Built so in M7; a job cancelled at the provider completes its tasks as cancelled, and is not retried, as for one task.)

What waits, and why:

- Critical-path-aware ordering. The heap already honours task priority, and the JDF can set one. Deriving priorities from the graph is a larger piece of work, and this workload has little slack between blocks to exploit.
- Latency-informed choice among QPUs. It needs at least two devices; the mechanism is in place and can be evaluated on the mock later.

## 2.9 Failure handling

QPU failures never use hook return codes, which are fatal (`scheduling.c:603-610`).

| Event | Device action | What the application sees |
|---|---|---|
| Job fails or times out | Retry up to `max_retries`, alone | Nothing, apart from `attempts` in the result |
| Retries exhausted | Complete the task normally | A result whose status says failed or timed out, with no entries |
| Backend unavailable | Back off; after `unavailable_timeout` complete pending tasks | Status "unavailable" |
| Finalization with jobs pending | Cancel, warn | The taskpool was not waited for; this is an application error |

(Section 9.2 has this table as built in M2, with three more rows: a job the backend cannot run at all, a job somebody else cancelled, and an end reported for a job that is not in flight.)

The task graph always continues, and the successor decides what a failed result means. Falling back to another incarnation of the same class by clearing the chore and calling `__parsec_reschedule` (`scheduling.c:499-521`) is possible later for classes that have one.

## 2.10 Statistics and tracing

- The module counts completed tasks itself, as GPUs do, and adds jobs submitted, tasks coalesced, retries, failures, and the latency averages to the device statistics output. (Built in M3, section 9.4. The trace events of the next item are not built yet.)
- The progress thread owns a profiling stream and records one interval per task from submission to hand-off, with the job identifier, attempts and the four latency terms attached. A second pair of events marks each provider job, so slot occupancy can be read from a trace. The execute interval the scheduler records around the hook (`scheduling.c:212-226`) stays as it is and shows only the submission.

## 2.11 Changes outside the new component

| File | Change |
|---|---|
| `parsec/mca/device/device.h` | `PARSEC_DEV_QPU` on bit 5 |
| `parsec/mca/device/device.c` | Name table entry; `parsec_mca_device_is_gpu` uses the GPU mask; load comparison at `:299` treats QPUs like accelerators; guard the capacity weights at `:861-890` |
| `parsec/scheduling.c` | Load added and removed for QPU types at `:151` and `:534`, keeping the host-input branch |
| `parsec/interfaces/dtd/insert_function.{c,h}` | A QPU function slot and wrapper for classes with several incarnations (`:2553`) |
| `parsec/include/parsec/parsec_options.h.in`, `parsec/mca/device/CMakeLists.txt`, `CMakeLists.txt` | `PARSEC_HAVE_DEV_QPU_SUPPORT` and the configure option |

(As built in M3, section 9.4: `device.h`, `device.c` and `scheduling.c` changed as listed, except that the comparison at `device.c:299` waits for a second QPU; `insert_function.{c,h}` and `parsec/mca/device/CMakeLists.txt` did not change; and `device.c` got one fix that this table did not foresee, in the list of device components.)

New files live in `parsec/mca/device/qpu/`: the component, the module, the progress thread, the backend interface, the mock, and the adapter client. `ValidateModule.CMake` collects the `.c` files of that directory, as the other components do. The provider adapter is a separate program outside the library.

## 2.12 Assumptions in this section

- The hand-off completion path works for PTG and DTD tasks (to be tested in M1).
- A write-only flow's copy can be written by the progress thread between the hook and the hand-off without other synchronization, because nothing else references it until the dependencies are released.
- Reading the circuit and parameters in place from pinned host copies is safe for the whole life of the job.
- The details of the DTD wrapper for multi-incarnation classes were not designed beyond the need for one.

---

# 3. Distributed QAOA for weighted MaxCut

The application has to be correct and has to expose real task dependencies. It does not aim at quantum advantage.

## 3.1 Problem and conventions

For a weighted graph $G=(V,E)$, maximize the cut

$$C(\mathbf{x})=\sum_{(i,j)\in E} w_{ij}\,(x_i\oplus x_j),\qquad \mathbf{x}\in\{0,1\}^{|V|}.$$

One convention is used everywhere:

- **Direction.** Every quantity is a cut value to be maximized. No energy with the opposite sign appears anywhere in the code.
- **Bits and spins.** $z_i = 1-2x_i$, so $x_i=0$ is $\lvert 0\rangle$ and $z_i=+1$. Then $x_i\oplus x_j=(1-z_iz_j)/2$.
- **Cost Hamiltonian.** $H_C=\sum_{(i,j)\in E}\frac{w_{ij}}{2}(I-Z_iZ_j)$, whose eigenvalue on $\lvert\mathbf{x}\rangle$ is $C(\mathbf{x})$.
- **Bit order.** Within a block, local vertex $i$ is qubit $i$ and bit $i$ of a 64-bit word (bit 0 is the least significant). Measuring qubit $i$ gives $x_i$ directly. Blocks are limited to 64 vertices.
- **Gates.** $R_{ZZ}(\theta)=e^{-i\theta ZZ/2}$, $R_Z(\theta)=e^{-i\theta Z/2}$, $R_X(\theta)=e^{-i\theta X/2}$.

## 3.2 Decomposition

Partition $V=V_1\cup\dots\cup V_K$ into blocks. For a block $S$ with every outside variable fixed at $\bar{\mathbf{x}}$, the cut splits exactly:

$$C(\mathbf{x}) = C_S(\mathbf{x}_S;\bar{\mathbf{x}}) + C_{\bar S}(\bar{\mathbf{x}}),$$

$$C_S(\mathbf{x}_S;\bar{\mathbf{x}})=\sum_{(i,j)\in E[S]} w_{ij}\,(x_i\oplus x_j)+\sum_{\substack{i\in S\\ j\notin S}} w_{ij}\,(x_i\oplus \bar x_j),$$

where $C_{\bar S}$ collects the edges with both ends outside $S$ and does not depend on $\mathbf{x}_S$. So a change of $\mathbf{x}_S$ that raises $C_S$ by $\delta$ raises the global cut by exactly $\delta$, as long as no neighbour of $S$ changes in between.

In spin form the boundary terms become local fields. With $h_i=\frac12\sum_{j\notin S} w_{ij}\,\bar z_j$ and $c_S=\frac12\sum_{i\in S,\,j\notin S} w_{ij}$,

$$H_S = c_S\,I+\sum_{(i,j)\in E[S]}\frac{w_{ij}}{2}(I-Z_iZ_j)-\sum_{i\in S} h_i Z_i .$$

**Update policy: conflict-free block colouring, expressed as dataflow.** Build the quotient graph (one node per block, an edge when two blocks share a cut edge) and colour it. Two blocks of the same colour share no edge, so their local objectives share no variable and no term. The schedule is a multicolour Gauss-Seidel sweep: in round $r$, block $b$ reads each neighbour $d$ at round $r$ if $\mathrm{colour}(d)<\mathrm{colour}(b)$, and at round $r-1$ otherwise.

The point of expressing it as dataflow is that the only ordering imposed is between neighbouring blocks. A block starts its update when its own neighbours have finished theirs, not when the whole colour class has. There is no barrier per colour and none per round.

What it guarantees:

- Any two adjacent blocks are totally ordered by the task graph and non-adjacent blocks commute, so every execution is equivalent to a sequential sweep in the order (round, colour, block).
- An update is accepted only if the candidate's $C_S$ is strictly larger than the current one's, both evaluated classically. So the global cut is non-decreasing, and it equals the initial cut plus the sum of the accepted improvements.

The alternatives were synchronous rounds over all blocks (needs a global validation step and a barrier, and can lower the cut), optimistic proposals validated against the global objective, and versioned boundary states. The last two allow more concurrency but need re-validation logic; they are future work.

## 3.3 Circuit and optimizer

For a block of $n$ qubits and $p$ layers,

$$\lvert\psi(\boldsymbol\gamma,\boldsymbol\beta)\rangle=\prod_{\ell=1}^{p} e^{-i\beta_\ell H_M}\,e^{-i\gamma_\ell H_S}\,\lvert +\rangle^{\otimes n},\qquad H_M=\sum_i X_i .$$

Up to a global phase, one layer is

- $R_{ZZ}(-\gamma_\ell w_{ij})$ on every internal edge $(i,j)$,
- $R_Z(-2\gamma_\ell h_i)$ on every qubit with a non-zero field,
- $R_X(2\beta_\ell)$ on every qubit,

preceded once by a Hadamard on every qubit and followed by measurement of all qubits. Zero-weight edges produce no gate. Start with $p=1$, then $p=2$.

The circuit is built once per block update as a structure with $2p$ parameter slots, and reused for every evaluation of that update.

**Objective estimate.** From the counts of one job, $\hat f = \frac{1}{\text{shots}}\sum_{\mathbf{x}} \mathrm{count}(\mathbf{x})\,C_S(\mathbf{x};\bar{\mathbf{x}})$. The same pass records the best sampled bitstring and its value.

**Optimizer.** SPSA with a fixed number $T$ of iterations. Each iteration evaluates $\boldsymbol\theta_t\pm c_t\boldsymbol\Delta_t$, which are two independent QPU jobs, and updates $\boldsymbol\theta$ classically. The perturbation $\boldsymbol\Delta_t$ comes from a generator seeded by (seed, round, block, $t$), so it does not depend on execution order. Initial angles are fixed constants in round 0 and the block's previous best afterwards.

SPSA was chosen over a grid (no iterative structure, so a flat task graph) and over COBYLA or Nelder-Mead (one evaluation per step, so no width). A fixed $T$ keeps the task graph static.

**Candidate.** The best bitstring seen over all $2T$ jobs of the update, compared against the current assignment.

## 3.4 Time budget

The per-task figures in the first table were measured in M5 with the sequential reference (section 9.3), Release build, one thread on methane; the estimate each one replaces is given next to it. The mock's QPU job was measured in M2 (section 9.2). Job counts and the hardware row are still estimates.

Per task, for a block of $n$ vertices and $m$ internal edges, 1024 shots:

| Task | Work | $n=14$, $m\approx 25$ | $n=20$, $m\approx 35$ |
|---|---|---|---|
| Gather boundary, build local problem and circuit | $O(n+m+\text{boundary edges})$ | 2 to 2.5 µs (estimate: microseconds) | 3 to 5 µs (estimate: microseconds) |
| Process counts, evaluate candidates | distinct bitstrings $\times\,(n+m)$ | 130 to 160 µs (estimate: tens of microseconds) | 210 to 270 µs (estimate: tens of microseconds) |
| Optimizer step | $O(p)$ | 0.2 to 0.35 µs (estimate: under a microsecond) | 0.7 to 1.2 µs (estimate: under a microsecond) |
| Update block | $O(n+m)$ | 0.05 µs (estimate: microseconds) | 0.15 µs (estimate: microseconds) |
| Exact conditional optimum (reference only, Gray-code enumeration) | $2^n\times m$ | 0.5 to 0.65 ms (estimate: about 0.1 ms) | about 43 ms (estimate: about 10 ms) |
| QPU job on the mock | exact sampling, $O(p\,n\,2^n)$; the job ends at the later of its modelled end and the end of the sampling | measured: 1.6 ms for $p=1$, 2.7 ms for $p=2$ (the estimate was under 1 ms) | measured: 0.15 s for $p=1$, 0.27 s for $p=2$ (the estimate was about 0.1 s) |
| QPU job on hardware | queue + submission + execution + result | see section 2.6 | see section 2.6 |

Job counts are $R\times K\times 2T$ for $R$ rounds, $K$ blocks and $T$ optimizer iterations:

| Instance | Purpose | $N$ | $K$, block size | $T$ | $R$ | Jobs | Shots in total |
|---|---|---|---|---|---|---|---|
| Controlled | Mock experiments | 256 | 16 blocks of 16 | 8 | 4 | 1024 | about 1.05 million |
| Validation | One real-hardware run | 40 | 4 blocks of 10 | 4 | 2 | 64 | about 66 thousand |

**What this says about overlap.** The classical work attached to one QPU job is about 0.15 to 0.3 milliseconds as measured, nearly all of it the evaluation of the counts (the estimate was 50 to 100 microseconds). A hardware job takes seconds before queueing. So at realistic latencies the CPUs are idle almost all the time, whatever the runtime does, and overlap between classical and quantum work is not measurable in this workload. It would be measurable only if the job latency were within roughly two orders of magnitude of the classical step, about 30 ms or less, which no physical QPU offers.

**What is measurable instead.** The scarce resource is the QPU, and the runtime's effect shows on the quantum side:

- how full the QPU's job slots are kept while quantum work is ready upstream;
- how soon a job is submitted after its inputs exist;
- time to solution against a barrier-synchronous execution when job latency varies. With a barrier per colour, every block waits for the slowest job of the colour class. With dataflow, a block waits only for its own neighbours. The gap grows with the variance of the latency, and cloud queue times are far from constant.

The critical path has $R\times\chi\times T$ jobs in sequence for $\chi$ colours, against $R\times K\times 2T$ jobs in total, so the speedup over sequential execution is bounded by $2K/\chi$ and needs about that many job slots.

**What would bring classical work into range**, without adding work for its own sake:

- Circuit transpilation for a real backend, if the adapter exposes it as its own task. It is a real cost of hybrid workflows; its size for these circuits is to be measured in M4.
- The exact reference solve at $n\ge 26$, which takes seconds on a CPU. The mock cannot sample blocks that large, so this exists only with real hardware.
- A workload with heavy classical linear algebra, such as SQD. That comparison belongs to stage 3.

**GPU.** At the block sizes the mock supports there is no kernel in this application that benefits from a GPU. The only candidate is the exact reference enumeration at $n\ge 26$. I recommend not claiming GPU use in the QAOA application, and showing CPU, GPU and QPU incarnations coexisting in one task graph at the level of a runtime test instead.

## 3.5 PTG or DTD

**Decision: PTG, one taskpool, statically unrolled over $R$ rounds and $T$ optimizer steps, with a skip flag for blocks that have converged.**

| Consideration | PTG | DTD |
|---|---|---|
| Dependencies only between neighbouring blocks, across rounds, with no barrier | Expressed directly, including a variable number of successors through a named range (`tests/dsl/ptg/local-indices/local_indices.jdf:33`). | Same graph, if every round is inserted up front. |
| Data-dependent continuation | Not available. Replaced by a skip flag carried with the block's state, so converged blocks cost microseconds. | Available, but in distributed mode every rank inserts the task sequence and tracks non-local tasks (`insert_function.c:212`, `:2824`), so the decision must be identical on all ranks, which means a wait and a broadcast per round: a global barrier. |
| QPU incarnation | `BODY [type=QPU]` compiles today (section 1.6). | Direct insertion works with the submit helper; a class with both CPU and QPU chores needs the change in section 1.4. |
| Cost on every rank | None beyond the tasks a rank runs. | Every rank walks every task. |

The barrier is what the comparison turns on: the regime where dataflow helps (section 3.4) is the one where rounds overlap. DTD support for QPU tasks is still part of the runtime work and is exercised by runtime tests.

This is static unrolling with dynamic no-ops, not dynamic task generation, and the paper should say so.

A variable number of inputs cannot be one PTG data flow, since a data flow has a single source. Neighbour assignments are therefore folded in by a chain of tasks, one per neighbour.

## 3.6 Task graph

(As built in M6, the graph differs from this section in four places, which section 9.5 lists with the reasons: `BUILD` proposes the first step and `STEP` runs for t = 1 to T; a block with no neighbour reads an empty boundary from a tile; `SUM` runs on the rank of its block and ends in a task `TOTAL`; and the final states are collected by `FINAL` on one rank instead of being written back to `X`.)

The graph, its partition and the colouring are computed before the taskpool is created, identically on every rank, because the task space (number of blocks, neighbours per block, colours) must be known then. They are read-only tables: `deg(b)`, `nbr(b,k)`, `rev(b,k)` (the slot of `b` in the list of its `k`-th neighbour) and `colour(b)`.

Data collections: `X(b)`, one tile per block holding its assignment, a version counter and its skip state, placed on rank `b mod P`; and `Q(0)`, a one-tile collection on the QPU owner rank, used only for affinity.

| Task | Device | Runs on | Role in the list of the stage 2 prompt |
|---|---|---|---|
| `INIT(b)` | CPU | rank of `X(b)` | Initial assignment |
| `GATHER(r,b,k)`, `k = 0 .. deg(b)-1` | CPU | rank of `X(b)` | `BUILD_QUBO`, boundary part |
| `BUILD(r,b)` | CPU | rank of `X(b)` | `BUILD_QUBO` and `BUILD_CIRCUIT` |
| `STEP(r,b,t)`, `t = 0 .. T` | CPU | rank of `X(b)` | `UPDATE_PARAMETERS` |
| `QEXEC(r,b,t,e)`, `e = 0 .. 1` | QPU | rank of `Q(0)` | `QPU_EXECUTE` |
| `EVAL(r,b,t,e)` | CPU | rank of `X(b)` | `PROCESS_COUNTS` and `EVALUATE_CANDIDATES` |
| `UPDATE(r,b)` | CPU | rank of `X(b)` | `UPDATE_GLOBAL` |
| `SUM(r,b)`, `FINAL(b)` | CPU | rank of `X(b)` | `CHECK_CONVERGENCE`, reporting only |

`GRAPH_PARTITION` is the pre-step above and is not a task.

Flows, for round `r` and block `b`:

```text
INIT(b)        --x-->  BUILD(0,b)
               --x-->  GATHER(0,d,rev) for each neighbour d with colour(d) < colour(b)

GATHER(r,b,k)  <--h--  NEW if k = 0, else GATHER(r,b,k-1)
               <--x--  s = nbr(b,k):  UPDATE(r,s)                 if colour(s) < colour(b)
                                      UPDATE(r-1,s) or INIT(s)    otherwise
               --h-->  GATHER(r,b,k+1), or BUILD(r,b) after the last neighbour

BUILD(r,b)     <--x--  UPDATE(r-1,b) or INIT(b)
               <--h--  GATHER(r,b,deg(b)-1)
               --P-->  EVAL(r,b,*,*), UPDATE(r,b)        local problem
               --C-->  QEXEC(r,b,*,*)                    circuit, sent to the owner rank once
               --S-->  STEP(r,b,0)                       optimizer state

STEP(r,b,t)    <--S--  BUILD(r,b) if t = 0, else STEP(r,b,t-1)
               <--F--  EVAL(r,b,t-1,0), EVAL(r,b,t-1,1)  for t > 0
               --TH->  QEXEC(r,b,t,0), QEXEC(r,b,t,1)    for t < T
               --S-->  STEP(r,b,t+1), or UPDATE(r,b) when t = T

QEXEC(r,b,t,e) <--C--  BUILD(r,b)
               <--TH-  STEP(r,b,t)
               --RES-> EVAL(r,b,t,e)

EVAL(r,b,t,e)  <--RES- QEXEC(r,b,t,e)
               <--P--  BUILD(r,b)
               --F-->  STEP(r,b,t+1)

UPDATE(r,b)    <--S--  STEP(r,b,T)
               <--P--  BUILD(r,b)
               <--x--  UPDATE(r-1,b) or INIT(b)
               --x-->  BUILD(r+1,b)
               --x-->  GATHER(r or r+1, d, rev) for each neighbour d, by the colour rule
               --x-->  X(b) after the last round
               --d-->  SUM(r,b)
```

`SUM(r,b)` is a chain over the blocks that accumulates the accepted improvements of round `r`. Nothing depends on it except the final report, so it is a sink and not a barrier. `FINAL(b)` recomputes the cut from the final assignments, and the run checks that it equals the initial cut plus the accumulated improvements.

**Skip.** A block's state records the versions of its neighbours at its last update and how many consecutive updates brought no improvement. If the versions are unchanged and that count has reached a set patience, `BUILD` marks the circuit as empty; the QPU device completes an empty job at once without contacting the backend, and the remaining tasks of the update pass the state through.

**Data moved between ranks per job.** The circuit, about $p(n+m)$ operations of 16 bytes, goes to the owner rank once per block update. The parameters are $2p$ doubles per job. The result comes back as a 64-byte header plus 12 bytes per distinct bitstring, at most about 12 kB for 1024 shots. PaRSEC records the destination ranks of each output dependency as a bitmask and adds a rank once (`parsec/remote_dep.h:136`, `parsec/remote_dep.c:408-409`), so the circuit travels to the owner rank once per block update and serves all $2T$ jobs there.

Per-flow dependency counts stay within the configured limit of 10 (`CMakeLists.txt:153-154`).

## 3.7 Reference implementation

A sequential program runs the same algorithm in the order (round, colour, block, step, evaluation). It shares the code that builds local problems and circuits, evaluates counts and performs the SPSA update, and calls the mock sampler synchronously with the same seeds. The distributed run must reproduce its assignments exactly.

Exhaustive search provides the exact optimum of a whole graph up to about 24 vertices and of each conditional local problem up to about 20, for approximation ratios.

Exact tests cover the deterministic parts: local-problem construction including fields and constants (checked by comparing $C_S + C_{\bar S}$ with $C$ on every assignment of small graphs), bit-order interpretation, the accept rule, and the final identity between the recomputed cut and the accumulated improvements.

Statistical tests cover measurement: sampled frequencies against the exact distribution, and the estimate $\hat f$ against the exact expectation within a tolerance that scales as $1/\sqrt{\text{shots}}$. For a single edge of weight $w$ at $p=1$, the expected cut has the closed form $\frac{w}{2}\,(1+\sin 4\beta\,\sin\gamma w)$ under the conventions of section 3.1, which gives an analytic check.

(As built in M5, the reference is `qaoa_sequential_run` in `tests/apps/qaoa/qaoa_seq.c`, and the tests named here are `apps/qaoa/math` and `apps/qaoa/reference`; section 9.3 has the choices section 3 left open and what was measured.)

The identities this section relies on were checked numerically while writing it, with a throwaway script that is not part of the repository: the split $C=C_S+C_{\bar S}$ and the spin form of $H_S$ on every assignment of a random weighted 7-vertex graph, the three gate angles of section 3.3 against the exact layer on a 4-qubit problem with fields and a zero-weight edge, and the single-edge formula at 500 random points (error below $10^{-14}$). The same checks become unit tests in M5.

## 3.8 Why it stays correct when tasks complete in any order

1. **Inputs are fixed by the graph, not by arrival order.** Every task names the producer of each input. With the mock, each job's samples depend only on a seed derived from (seed, round, block, step, evaluation), so a run gives the same assignments whatever the completion order, and the same as the sequential reference, bit for bit. On hardware the samples are random, but each run is still a valid sequential sweep.
2. **Improvements are exact.** `UPDATE(r,b)` evaluates its candidate against the boundary values it was built from, and the graph orders it against every neighbour's update. Section 3.2 then gives equivalence to a sequential sweep and a non-decreasing cut.
3. **A failed job cannot corrupt the state.** A result with a failure status is ignored by `STEP` (that iteration makes no move) and contributes no candidate. The current assignment is always a candidate, so `UPDATE` remains valid.
4. **No stale parameters.** The optimizer of one block update is a chain, so each step sees the previous one's result.

---

# 4. Test plan

None of the tests below exists yet and nothing in this section was run. Source references are verified at the base commit unless marked as assumed.

## 4.1 Where tests live and how they are registered

| Kind | Directory | Built when |
|---|---|---|
| Runtime tests of the QPU device, the mock and the sampler | `tests/runtime/qpu/` (new) | `PARSEC_HAVE_QPU`, the way `tests/runtime/cuda/CMakeLists.txt:6` is guarded by `PARSEC_HAVE_CUDA` |
| A JDF with a QPU body that must still build and run without QPU support | `tests/dsl/ptg/qpu_fallback.jdf` (new) | Always |
| QAOA application, sequential reference, correctness tests | `tests/apps/qaoa/` (new) | `PARSEC_HAVE_QPU`; the unit tests of the QAOA arithmetic need only the sampler |
| Statistics and trace checks | `tests/profiling/check-qpu.py` (new), entries in `tests/profiling/Testings.cmake` | Python tools and tracing enabled, as for the existing checks (`tests/profiling/Testings.cmake:4`) |

Registration follows the existing pattern:

- Executables: `parsec_addtest_executable(C <target> SOURCES ...)` and `target_ptg_sources(<target> PRIVATE "<file>.jdf")` (`tests/CMakeLists.txt:36-50`; example at `tests/runtime/cuda/CMakeLists.txt:38-41`).
- Commands: `parsec_addtest_cmd(runtime/qpu/<name> ${SHM_TEST_CMD_LIST} runtime/qpu/<exe> ... -- --mca device_qpu_enabled 1`), and `${MPI_TEST_CMD_LIST} <ranks>` with the suffix `:mp` for MPI runs (`tests/runtime/Testings.cmake:9`, `tests/apps/stencil/Testings.cmake:1-7`).
- Directory hooks: `add_subdirectory(qpu)` in `tests/runtime/CMakeLists.txt` after line 6, `include(runtime/qpu/Testings.cmake)` in `tests/runtime/Testings.cmake` after line 6, `add_subdirectory(qaoa)` in `tests/apps/CMakeLists.txt:5-14`, and one `include` line in `tests/apps/Testings.cmake`.
- Every QPU test gets a CTest `TIMEOUT` property. No existing test sets one (searched), but a lost completion shows up as a hang, and a hang has to fail quickly.

Mock delays in registered tests stay at or below 100 ms, so the whole QPU suite adds about two minutes to a `ctest` run. Stress tests take a size argument, small in CTest and large for manual sanitizer runs.

## 4.2 No test can reach a real QPU

Five independent layers:

1. **Default environment.** `parsec_addtest_cmd` already forces `PARSEC_MCA_device_cuda_enabled=0` and its siblings into every test's environment (`tests/CMakeLists.txt:94-95`). Add `PARSEC_MCA_device_qpu_enabled=0` and `PARSEC_MCA_device_qpu_allow_hardware=0` to that list. As the comment at `tests/CMakeLists.txt:74-93` explains, a test opts in with explicit `--mca` arguments; QPU tests pass `--mca device_qpu_enabled 1 --mca device_qpu_backend mock`.
2. **The hardware guard of section 2.5.** A backend that reports itself as physical opens only when `device_qpu_allow_hardware` is set. (Since M3 such a backend gives no QPU device at all: a task class with another incarnation runs that one, and a body that submits all the same is refused.) `runtime/qpu/hardware_guard` registers a test-only backend that claims to be physical and counts its `open` and `submit` calls; the test passes only if both counts stay at zero without the setting.
3. **A check on the test definitions.** `runtime/qpu/no_hardware_in_tests` is a CMake script run as a test. It fails if the string `allow_hardware` followed by a non-zero value appears in any `Testings.cmake`.
4. **The adapter client is tested against a loopback server.** `runtime/qpu/adapter_loopback` starts a small C program that speaks the protocol of section 2.6 on a Unix-domain socket in the build directory and answers from the mock. Neither Python nor the network is involved.
5. **The provider adapter is never started by CTest.** It is a separate program (section 2.6), it needs credentials in its own environment, and it refuses to start without its own explicit flag.

The hardware run of milestone M4 is a manual procedure with its own instructions, not a test.

## 4.3 Hazards: races and lifetimes

These are the ways a pending job could end a taskpool early, complete a task twice, or leave a dangling reference. Each has a design rule and a test.

| # | Hazard | Why it can happen | Rule | Test |
|---|---|---|---|---|
| H1 | Completion overtakes the worker that submitted | After a hook returns ASYNC the worker still stores the return code in the task (profiling builds) and runs the `EXEC_END` instrumentation on it (`scheduling.c:221-226`). A job that finishes at once can be handed off, completed by another worker and returned to its mempool (`interfaces/interface.c:24-38`) before that. | See the note below the table. | `stress` with a zero-delay mock, in a profiling build with the task profiler loaded, under the thread sanitizer |
| H2 | A task completes twice | A backend reports a final state twice; a retry races a late success of the previous attempt; a timeout cancellation races the completion. | A job record has one state, changed only by the progress thread; the first final state wins; the hand-off happens once and clears the record's task pointer. Each attempt has its own tag. | `failure`: the mock injects duplicate and late events; the successor counts executions per task and any count other than one fails |
| H3 | The taskpool terminates while a job is pending | It cannot by construction: the task stays in the task count until `release_task` (section 1.7). Tested because property 4 rests on it. | No counter is touched by the device. | `termination`, all variants |
| H4 | A reference outlives the task | After the hand-off, a worker may complete the task and the taskpool may be freed at any moment (invariant 5 of section 1.5). | The progress thread drops every pointer to the task, its copies and its taskpool before it calls `__parsec_schedule`. | `termination:loop` (many short taskpools, each freed right after its wait) under the address sanitizer |
| H5 | Data changes or disappears while the job is outstanding | The job reads the circuit and parameters in place. | Inputs stay pinned by the task (`scheduling.c:218-220`); in DTD a later writer of a tile the QPU task reads has to wait for it. | `chain` checks result contents against the mock's deterministic function; `chain:dtd` inserts a writer after an outstanding reader |
| H6 | Lost wake-up | A submission lands between the progress thread's last check and its sleep. | The wake-up is level-triggered, and `wait` always has a finite timeout, so a lost wake-up costs latency and never liveness. | `stress` with zero and random delays and many workers; a hang hits the CTest timeout |
| H7 | Shutdown order | `parsec_fini` joins the workers (`parsec.c:1313`), finalizes profiling (`:1326`) and only then closes the device components (`:1345`). At close there is no scheduler and no profiling. | Close cancels, counts and joins. It schedules nothing and traces nothing. | `init_fini`; a white-box test of close with job records still pending |
| H8 | Repeated initialization | Static state (submission list, thread-local stream, thread, backend registry) survives a `parsec_fini`. | All of it is created in component open and destroyed in close. | `init_fini`: twenty init, run, fini cycles in one process, under the leak sanitizer |
| H9 | Scheduling from a foreign thread | `__parsec_schedule` from a stream with no scheduler object delivers to stream 0 of each virtual process (`scheduling.c:409-425`); profiling streams are not thread-safe (`:330-333`). | The progress thread uses its own execution stream and its own profiling stream, nothing else. | `stress` with one worker, with many, and with more than one virtual process |
| H10 | Capacity overshoot, or a slot that never comes back | Admission and completion run concurrently with submissions; failure paths skip the release. | Slot accounting lives in the progress thread only. | `capacity`: the mock records its high-water mark, which must equal the limit; `failure` then reuses the full capacity |
| H11 | A provider job is submitted twice | Ambiguous submission failure; retry logic. | Section 2.5: one tag per task and attempt, lookup by tag before resubmitting. | The mock records every tag; a repeated tag fails the test |
| H12 | Counters drift | Load added at submission and removed at completion on different threads. | | Every runtime test ends by checking, through a test-only query, that load, pending and in-flight are zero and that the executed count equals the number of QPU tasks |

**Note on H1.** This window is not new. The profiling test `tests/profiling/async.jdf` has a task publish itself and return ASYNC (`:88-91`) while another task reschedules it at once with `__parsec_schedule` (`:135`), and the GPU manager completes tasks on behalf of workers that are still returning (as far as I read `device_gpu.c`). Tasks are mempool objects, so the memory stays valid; the visible effect would be a trace event attributed to a recycled task. M1 measures it. If the sanitizer reports it, the clean remedy is a small change in `__parsec_execute` so that nothing touches the task after an ASYNC return, which I would propose to the maintainers instead of working around it in the device. (M1 measured the window directly; the result is in section 9.1.)

H2, H5, H10 and H11 are tested since M2 (section 9.2): `failure` and its variants for H2, H10 and H11, and three counters of the mock that every test requires to be zero for H5 and H11.

That test also changes one statement of section 1.5. Rescheduling a task that returned ASYNC through `__parsec_schedule` does have a precedent in the tree. What the hand-off adds is the status set to `PARSEC_TASK_STATUS_COMPLETE` (`parsec_internal.h:568-573`), so the hook does not run again, and a caller that is not a worker.

## 4.4 Runtime tests

Sources are in `tests/runtime/qpu/` unless a path is given. "Ranks" lists the process counts registered with CTest. (The M1 tests as registered differ in a few names and files; section 9.1 lists them. Section 9.2 does the same for M2, and section 9.4 for M3.)

| CTest name | Source | Ranks | What it checks | From |
|---|---|---|---|---|
| `runtime/qpu/sampler` | `qpu_sampler_test.c` | 1 | No runtime. Each gate against its analytic state, norm, Bell and GHZ states, sampled frequencies against exact probabilities, identical counts for identical (seed, tag). | M2 |
| `runtime/qpu/mock` | `qpu_mock_test.c` | 1 | No runtime. The backend contract: delays, slots, `wait` and `wakeup`, failure injection, tags, results independent of submission order, every result flagged as mock. | M2 |
| `runtime/qpu/discovery` | `qpu_discovery.c` | 1, 2 | Registration and discovery: zero QPUs when disabled, the configured number on the owner rank, capabilities as configured, `parsec_mca_device_is_gpu` false, no backend opened on other ranks. | M3 |
| `runtime/qpu/chain` | `qpu_chain.jdf` | 1, 2, 4 | Submission, asynchronous completion and dependency release: CPU task, QPU task, CPU task, in many independent chains. The successor verifies the result's status, shot total, mock flag and contents. In MPI runs the CPU tasks live on other ranks than the QPU task. | M1 |
| `runtime/qpu/chain:dtd` | `dtd_qpu_chain.c` | 1, 2 | The same in DTD, plus insertion of new quantum tasks while earlier ones are outstanding, and a writer inserted after an outstanding reader (H5). | M1, QPU-typed in M3 |
| `runtime/qpu/order` | `qpu_chain.jdf` | 1 | Completion ordering: delays arranged so jobs finish in reverse and in seeded random order. Dependents are released in completion order, each receives its own predecessor's result, and results are identical across orders. | M1 |
| `runtime/qpu/capacity` | `qpu_chain.jdf` | 1 | Several outstanding tasks and capacity enforcement: far more tasks than slots; high-water mark equal to `max_outstanding`; minimum submission interval respected. | M2 |
| `runtime/qpu/overlap` | `qpu_overlap.jdf` | 1 | Classical and quantum work concurrently: CPU tasks start and finish inside the interval of one long job. Run once with a single worker thread, where a blocking implementation would fail. Also checks that the progress thread's own CPU time stays under 1% of the wall time while it waits. | M1 |
| `runtime/qpu/termination` | `qpu_termination.jdf` | 1, 2, 4 | Termination with tasks outstanding: the last task of the taskpool is quantum, and `parsec_context_wait` returns only after its result is written; a second, classical taskpool finishes while the first still has a job pending; in MPI runs a successor on another rank must have run before any rank's wait returns. Variants: `:dynamic` uses the four-counter detector through the JDF property `termdet = "dynamic"` (`jdf.h:205-210`, `jdf2c.c:8210-8227`, `termdet.c:104-107`); `:loop` is H4. | M1 |
| `runtime/qpu/failure` | `qpu_failure.jdf` | 1, 2 | Failure and timeout handling: fail once then succeed (attempts is 2); always fail (failure status, no entries, successor still runs, run exits normally); timeout (the mock sees the cancel); backend unavailable for less and for more than `unavailable_timeout`; duplicate and late events (H2); ambiguous submission (H11). | M2 |
| `runtime/qpu/stress` | `qpu_stress.jdf` | 1, 4 | H1, H6, H9: many tasks, zero and random delays. | M1 |
| `runtime/qpu/init_fini` | `qpu_init_fini.c` | 1, 4 | Repeated initialization and finalization (H7, H8), following `tests/api/init_fini.c`. | M1, again in M3 |
| `runtime/qpu/hardware_guard`, `runtime/qpu/no_hardware_in_tests` | `qpu_guard.c`, `check_no_hardware.cmake` | 1 | Section 4.2. | M2 for the guard, M1 for the check |
| `runtime/qpu/coalesce` | `qpu_coalesce.jdf` | 1, 2 | Coalescing: no coalescing with spare capacity; under contention several tasks per job, each with its own correct result; a failed coalesced job retried task by task; results identical with coalescing on and off. | M7 |
| `runtime/qpu/adapter_loopback`, `runtime/qpu/qasm` | `qpu_adapter_loopback.c`, `qpu_qasm_test.c` | 1 | The socket client against the loopback server, including a server that dies mid-run; the OpenQASM text emitted for known circuits; a dummy token that must not appear in any output at the highest verbosity. | M4 |
| `runtime/qpu/cpu_gpu_qpu:gpu` | `cpu_gpu_qpu.jdf` | 1 | CUDA builds only: CPU, CUDA and QPU incarnations in one task graph, one rank per GPU. | M3 |
| `dsl/ptg/qpu_fallback` | `tests/dsl/ptg/qpu_fallback.jdf` | 1 | A class with a QPU body and a CPU body. Without QPU support, or with the device disabled, the CPU body runs; with it, the QPU body runs. | M3 |
| `profiling/qpu_*` | `tests/profiling/check-qpu.py` | 1 | QPU tasks appear in the device statistics and in the trace: one interval per task and per provider job, with the latency terms attached. | M3 |

## 4.5 QAOA correctness tests

All in `tests/apps/qaoa/`, all on the seeded mock. Graphs are generated in code; there are no data files. Weights are multiples of 1/8, so sums are exact in floating point and comparisons use equality. (The first two tests of the second table exist since M5, with the instances as named in section 9.3; `apps/qaoa` and `apps/qaoa:fail` since M6, with the variants section 9.5 lists.)

| Instance | Graph and partition | What it is for |
|---|---|---|
| `edge2` | Two vertices, one edge. One block of two, and two blocks of one. | The closed form of section 3.7. With two blocks of one, a block has only a field and no internal edge. |
| `tri` | Triangle with weights 1, 2, 3; blocks {0,1} and {2}. | Smallest case with a frustrated edge and an uneven partition. |
| `c4` | Four-cycle; blocks {0,1},{2,3}, and blocks {0,2},{1,3}. | In the second partition every edge crosses blocks. |
| `rand12`, `rand16` | Seeded random graphs with nonuniform weights, some of them zero; 3 and 4 blocks. | Zero-weight edges emit no gate and change no value. |
| `rand12neg` | As `rand12`, with some negative weights. | Sign errors that positive weights hide. |
| `disc16` | Four blocks with no edge between them. | One colour, empty boundary: no ordering may be imposed. |
| `ring64` | 64 vertices, 8 blocks of 8, sparse cross edges. | Enough width for the distributed tests. |

| CTest name | Source | What it checks |
|---|---|---|
| `apps/qaoa/math` | `qaoa_math_test.c` | No runtime. $C = C_S + C_{\bar S}$ on every assignment; fields and constants; bit order, on a path graph with distinct weights where any reversal changes the value; the gate angles of section 3.3 against the exact layer; the single-edge closed form; the strict accept rule; SPSA perturbations depend only on (seed, round, block, step). Statistical part: $\hat f$ within six standard errors of the exact expectation, with fixed seeds so the test is deterministic. |
| `apps/qaoa/reference` | `qaoa_reference.c` | The sequential reference on every instance: the global cut never decreases; the recomputed final cut equals the initial cut plus the accepted improvements; the final cut does not exceed the exhaustive optimum (whole graph up to 24 vertices). |
| `apps/qaoa` | `testing_qaoa --check` | The PaRSEC run against the reference, in one process: final assignments identical bit for bit, hence identical objective, on every instance; repeated with three delay seeds and with 1 and 4 slots; the number of provider jobs equals the reference's count and every tag appears once. |
| `apps/qaoa:fail` | same | With failures injected at fixed tags, the run still equals the reference given the same list of failed evaluations. |
| `apps/qaoa:coalesce` | same | Results identical with coalescing on (M7). (As built: seven tests, and 24 variants of `runtime/qpu/coalesce`; section 9.6.) |

## 4.6 Distributed tests

`apps/qaoa:mp` and the `:mp` runtime tests run on 1, 2 and 4 ranks in CTest, and on 8 and 16 by hand where the machines allow. Each checks:

- **Same answer.** Final assignments equal the sequential reference, for every rank count.
- **No job twice.** The mock on the owner rank counts submissions and distinct tags; both equal the reference's job count.
- **One owner.** Ranks other than the owner opened no backend. Repeated with the owner on rank 0 and on the last rank.
- **Dependencies.** A block's state carries its round and version. `GATHER` asserts that each neighbour state it receives is the one the colour rule of section 3.2 predicts.
- **Uneven distributions.** More ranks than blocks (`tri` on 4 ranks, so some ranks own nothing and must still terminate), and a block count that is not a multiple of the rank count.

Several ranks sharing one QPU is the normal case in all of these, since there is one owner.

## 4.7 Build configurations and sanitizers

Before M1, record which existing tests pass at the base commit on each machine, so "existing tests keep passing" has a baseline.

| Configuration | What must hold |
|---|---|
| QPU off (the default), with and without CUDA | The existing suite gives the baseline result. `dsl/ptg/qpu_fallback` runs its CPU body. No QPU file is compiled. |
| QPU on, no GPU, MPI on | Existing suite at baseline, plus every QPU test. |
| QPU on, MPI off (`PARSEC_DIST_WITH_MPI=OFF`) | Builds; shared-memory QPU tests pass. |
| QPU on, CUDA on | CUDA tests at baseline, plus `runtime/qpu/cpu_gpu_qpu:gpu`. |
| QPU on, Debug with `PARSEC_DEBUG_PARANOID` | QPU tests pass with assertions enabled. |
| QPU on, `PARSEC_PROF_TRACE` on | Trace tests pass; H1 is exercised here. |
| Sanitizers: `PARSEC_DEBUG_MEM_RACE`, `PARSEC_DEBUG_MEM_ADDR`, `PARSEC_DEBUG_MEM_LEAK` (`CMakeLists.txt:354-394`) | `runtime/qpu/*` and `apps/qaoa*` at large sizes, by hand at each milestone. I have not checked that the existing runtime is clean under the thread sanitizer; if it is noisy, compare against a run of an existing test and read only the reports that involve QPU code. |

If the destination is an upstream pull request, the `cpu` rows of the CI matrix (`.github/workflows/main.yml`, options built in `.github/CI/env_setup.sh:31-43`) gain the QPU option, and the test step's environment gains `PARSEC_MCA_device_qpu_enabled: 0`.

## 4.8 What these tests do not cover

- A real provider. Its failures, queueing and rate limits are seen only in the manual run of M4.
- Performance. That is section 6.
- More than one QPU, beyond `discovery` with two mock modules.
- The `user_trigger` detector with a job pending (open question 4 of section 1.1). It stays a documented limitation until someone needs it.
- What `parsec_fini` does when a taskpool is still active. I did not read that path; the white-box test of H7 covers the device's side only.

---

# 5. Novelty

Adding QPU support is not novel by itself. The sources below show that it is not new for a task-based runtime either: it has been published twice.

**How this section was made.** Two web searches were run on 2026-11-21 by separate research passes, which read the papers, documentation pages and repositories listed at the end of the section and saved copies of them. I checked a sample of about forty quoted statements against those copies and all were there; I did not read the papers in full myself. Many sources are preprints, and several are from the last few months. "Not stated" means the sources read are silent. An absence is reported as "not found in the sources read", never as "does not exist".

## 5.1 Task runtimes and HPC middleware

| System | QPU is exposed as | Does quantum work hold a thread? | One graph across nodes? | QPU capacity and batching | Hardware in the evaluation | Code |
|---|---|---|---|---|---|---|
| OmpSs-2 extension [R1] | A task with `device(qpu)`, body in OpenQASM, "similar to the already supported GPU offloading" | No. Device tasks "do not occupy any CPU core"; the runtime polls the backend and releases dependents on completion | Not stated; the evaluation ran on one machine | No bound stated; one circuit per job | Simulator only | Extension not found in the public Nanos6 tree |
| Q-IRIS [R2][R3][R4] | A "quantum device backend (similar to the GPU backend of IRIS)" that runs QIR programs through QIR-EE | The papers describe asynchronous scheduling. The driver in the repository is synchronous and sleep-polls every millisecond | No: "on a single node"; multi-node is future work | Not stated | IonQ Forte-1 [R3] | Public, BSD-3-Clause |
| DQR [R5] | A backend of a router for circuit fragments, inside one MPI job | No. The coordinator rank polls without blocking | One MPI job, coordinator and workers; fragments are independent, with no dependencies | A slot count read from an environment variable; one fragment per job; retry and failover per fragment | CESGA Qmio, IBM Torino | Public, no licence |
| PyCOMPSs with Qdislib [R6][R7] | A resource type, reached through an agent or from a CPU task | In the repository the task waits on the provider job | One task graph over many nodes through the COMPSs master and workers, not MPI | No runtime bound stated; packing of subcircuits into one job is a user option of the library | MareNostrum Ona, IBM Marrakesh | Public, Apache-2.0 |
| OpenMP target offloading [R8][R9][R10] | An OpenMP target region | Blocking by default, `nowait` proposed [R8]; blocking in [R9][R10] | No | In a separate queue service [R9] | Simulators; one validation on a 54-qubit device [R10] | Not found |
| XACC, QCOR [R11][R12][R13] | An accelerator object behind a compiler | A host thread carries the call; futures | Not stated | Not stated | IBM device for one example [R12] | Public |
| QFw [R14] | A backend behind an RPC service | Results return asynchronously; the application uses threads | No task graph | A worker pool in the service | Simulators only | Not found |
| Pilot-Quantum [R15] | Reached from inside classical tasks | Not stated | No: it "lacks high-level representations like DAGs" | Not stated; batching is future work | IBM Eagle, small runs | Public |
| Tierkreis [R16][R17] | Workers and executors for provider services | No; a workflow server polls | One workflow graph; each task is its own scheduler job, with shared files between them | Not stated | None reported | Public |
| MQSS, QDMI [R18][R19] | A Slurm generic resource plus a second-level dispatcher | The job interface is asynchronous; hybrid jobs can idle waiting for the QPU | No task graph | By Slurm and the site's resource manager | IQM 20-qubit device | Partly public |
| CRS4 workflow layer [R22] | A backend for blocks of a decomposed optimization problem | Not stated | Not stated | One batched submission per round for some solvers; timeouts, retries and failover | IQM devices | Not stated |

Covalent [R20] and QCCP [R21] add asynchronous quantum calls to Python workflow systems and do not change this picture. A 2025 survey of 107 publications places the OpenMP and OmpSs-2 extensions together as pragma-based models, and says of middleware that "the existing work mainly focuses on scheduling and managing resources" [R23]. For StarPU, Legion, HPX, Charm++, PaRSEC and TTG, no QPU device or quantum offloading was found.

The three closest:

- **OmpSs-2** has the core idea already: a QPU device task in a dataflow runtime that holds no core and is completed by the runtime. It was shown on one machine with a simulator, reports one timing result (a parameter scan "reduced by 14%"), and describes no capacity bound, batching or failure handling.
- **Q-IRIS** has a QPU device class and a run on physical hardware. It is single-node, its first paper says it is "not a performance study", and nothing about capacity, batching or failures is described.
- **DQR** is the only system found that bounds QPU slots, polls without blocking and retries failed work inside an MPI program, on two physical devices. It dispatches independent fragments from a coordinator; it has no task graph.

## 5.2 Vendor frameworks

| Framework | How quantum work is submitted | Dependency graph | Who bounds QPU use | Batching |
|---|---|---|---|---|
| IBM Qiskit Runtime modes [R25] | A library call that returns a job handle | None. Batch mode is for jobs that "have no conditional relationship with each other" | The cloud queue | The user decides what goes into a job |
| IBM Qiskit Serverless [R26] | A Python function run remotely, with Ray inside | None described | The service bounds its own classical jobs | Not stated |
| IBM QRMI and Slurm plugins [R27][R28] | The QPU is a Slurm resource; the application uses the SDK inside the job | None: "Slurm does not orchestrate anything inside the unit" | Slurm, per Slurm job | Not stated |
| NVIDIA CUDA-Q [R30] | `sample_async` and similar calls return futures; a multi-QPU platform; MPI helpers | None described; the user holds the futures | Not stated | Not stated |
| Pasqal [R31] | SDK calls routed to a site daemon on a "Quantum Access Node" | Not stated | The daemon: a priority queue and a second level of scheduling | Not stated |
| AWS Braket Hybrid Jobs [R32] | A containerised script next to a device | Not stated | The service: one hybrid job per QPU | User-built sets of up to 100 programs |
| Quantinuum Tierkreis | See section 5.1 | | | |

Every vendor stack read treats the QPU in one of two ways: as a batch-scheduler resource held for a whole job, or as a cloud endpoint called from application code. The layer between them is an application-level runtime that owns the connection and schedules circuit jobs against a dependency graph. IBM's reference architecture names it as a future need: schedulers that consider "the complex dependencies and data flow patterns between quantum and classical components" [R29]. The openQSE survey asks for a "second-level quantum scheduler" [R24]. The published closed-loop run on Fugaku overlapped quantum and classical work by hand, for two populations, and reports the reduced idle time as a result [R33].

## 5.3 Decomposed QAOA

| Work | Decomposition | Coupling between sub-problems | Orchestration | Physical QPU |
|---|---|---|---|---|
| DQAOA [R34] | Random subsets of variables, redrawn every iteration | Concurrent solves from the same global state; afterwards, bit changes that lower the global cost are accepted | mpi4py, every core submitting its own jobs; later QFw with Python threads [R14] | IBM Eagle devices, 100 variables |
| QAOA-in-QAOA [R35][R36] | Graph partition | Ignored while solving, then one merge that flips whole blocks | Not stated [R35]; MPI and Slurm with a coordinator rank, simulated devices only [R36] | None |
| DC-QAOA [R37] | Vertex separators with shared nodes | Agreement on the shared nodes at reconstruction | Not stated | Not located |
| Quantum local search, multilevel [R38][R39] | One subset of vertices per iteration | **Fixed boundary values**, as in section 3.2 | A sequential driver | IBM 16-qubit device, D-Wave |
| CRS4 layer [R22] | Blocks of a routing problem | Aggregation on a quorum of blocks | A workflow layer | IQM devices |

Two consequences:

- The local problem of section 3.2, with boundary values as local fields, is established. The quantum local search paper itself calls it "a commonly used technique" [R38]. It is not a contribution.
- "Distributed QAOA" is already the name of a published algorithm with a different decomposition [R34]. The paper should use another name for the application and cite DQAOA as the closest work by setting.

What was not found in the sources read: concurrent block updates ordered by a colouring of the block graph and expressed as task dependencies, and any decomposed QAOA run on a task runtime.

## 5.4 What is not new

- A QPU as a device or target of a task-based runtime: OmpSs-2, Q-IRIS, OpenMP [R1][R2][R8].
- Quantum tasks that hold no core and are completed by the runtime: OmpSs-2 [R1].
- A physical QPU driven from a task runtime: Q-IRIS [R3].
- A bound on QPU slots, non-blocking polling, and retry with failover in an MPI program: DQR [R5].
- A distributed task graph with QPU tasks on physical hardware, without MPI: PyCOMPSs with Qdislib [R6].
- Several circuits in one provider job when the user asks for it: Qdislib, IBM, AWS [R7][R25][R32].
- Decomposed QAOA with concurrent sub-problems on HPC systems, with and without hardware [R34][R14][R22].

## 5.5 What would be new

| Capability | Section | Closest prior work | What differs |
|---|---|---|---|
| A QPU as a device of a dataflow runtime whose single task graph spans MPI ranks, with one rank owning the connection and the others reaching it through task placement and data dependencies | 2.7 | OmpSs-2 and Q-IRIS (one machine or node); DQR (MPI, no graph); Qdislib (graph, no MPI) | The combination. Q-IRIS lists multi-node support as future work [R3]. |
| Coalescing of pending quantum tasks into one provider job, decided by the runtime under contention | 2.8 | A user option in Qdislib; user-built batches at IBM and AWS | The runtime decides, across tasks of different blocks and ranks, and only when the QPU is contended. |
| Capacity, submission rate, load and observed latency kept by the device and used by the runtime's incarnation selection | 2.2, 2.8 | DQR's static slot count; the site-level schedulers of section 5.2 | Inside the application's runtime, per circuit job, fed by measurements. |
| A failure path inside the task graph: the device retries, then the task completes with a status in its result and the graph continues | 2.9 | DQR and the CRS4 layer retry and re-route; OmpSs-2 and Q-IRIS describe nothing | Minor. It is engineering that the others lack, not a research result. |
| An evaluation of a task-runtime QPU device: overhead, utilization, a control that separates schedule from runtime, null results | 6 | One 14% figure [R1]; "not a performance study" [R2]; wall times of one circuit [R3] | Neither prior runtime paper has one. |
| Block-coordinate QAOA as a barrier-free dataflow graph, reproducible bit for bit under any completion order | 3.2, 3.8 | Quantum local search (sequential); DQAOA (repair after concurrent solves) | Secondary. The algorithmic pieces exist; the formulation as dependencies was not found. |

The progress thread that sleeps in a backend wait call, where the others poll, is an implementation choice and not a claim.

## 5.6 Assessment

- The paper cannot say "first QPU device in a task-based runtime". OmpSs-2 and Q-IRIS are there, and a reviewer in this area will know both.
- What it can defend is narrower: the first such device in a distributed-memory dataflow runtime, with runtime-side policies (capacity and coalescing) that the prior runtime papers do not have, and with the evaluation they do not have.
- Of these, coalescing is the strongest single item. It is the only one for which the sources show nothing comparable at runtime level, and section 6.1 predicts it to be the largest measurable effect.
- The application is a vehicle, as intended. Its parts are known.
- This field moves quickly: eight of the sources above date from 2026. The search was done once, today. It should be repeated before submission, and Q-IRIS's announced multi-node work is the thing to watch.

## 5.7 Can an existing runtime serve as a baseline?

| System | Feasible? | Why |
|---|---|---|
| OmpSs-2 extension | No | The extension's code was not found in the public Nanos6 repository, and the paper gives no link [R1]. |
| Q-IRIS | Later, on one node | IRIS and its quantum driver are public [R4], but the driver takes QIR programs through QIR-EE, is single-node and waits synchronously. A comparison means porting the application to QIR, and can only cover one node. |
| DQR | No | Independent fragments, no dependencies [R5]; the QAOA graph cannot be expressed. |
| PyCOMPSs with Qdislib | Later | Public and distributed [R7], but a Python workflow system with a different execution model; a fair port is a project of its own. |

So the first submission compares against A, B and C-sync, and positions itself against existing runtimes with the sourced tables of this section. A single-node microbenchmark against IRIS, on the questions of E1 and E2, is the realistic next step.

## Sources for section 5

Papers are given by arXiv identifier; all were retrieved on 2026-11-21.

- [R1] Döbler, Álvarez, Menger, Lippert, Beltran, Jattana. Extending the OmpSs-2 Programming Model for Hybrid Quantum-Classical Programming. arXiv:2502.21104.
- [R2] Miniskar, Monil, Wong, Leyton-Ortega, Vetter, Johnson, Humble. Q-IRIS: The Evolution of the IRIS Task-Based Runtime to Enable Classical-Quantum Workflows. SCA/HPCAsia 2026 Workshops. arXiv:2512.13931.
- [R3] Miniskar, Wong, Leyton-Ortega, Vetter, Humble. Classic and Quantum Task-Based Intelligent Runtime for QIRs Running on Multiple QPUs. arXiv:2605.11382.
- [R4] IRIS repository: https://github.com/ORNL/iris (`src/runtime/DeviceQIREE.cpp`, `src/runtime/ProcessQiree.h`).
- [R5] Raigada-García, Jorba, Iserte. Wave-based dispatch for circuit cutting in hybrid HPC-quantum systems. arXiv:2604.15279.
- [R6] Tejedor, Casas, Conejero, Cervera-Lierta, Badia. Distributed Quantum Circuit Cutting for Hybrid Quantum-Classical High-Performance Computing. arXiv:2505.01184.
- [R7] Qdislib repository: https://github.com/bsc-wdc/qdislib
- [R8] Lee, Brown, Bull, Ruefenacht, Doerfert, Klemm, Schulz. Quantum Task Offloading with the OpenMP API. SC23 poster. arXiv:2311.03210.
- [R9] Mahesh, Mittal, Mueller. CONQURE: A Co-Execution Environment for Quantum and Classical Resources. arXiv:2505.02241.
- [R10] Kaya, Şahinkaya, Chung, Schulz. OpenMQPI: An MLIR-based Pipeline to Offload Quantum Circuits via OpenMP. arXiv:2610.18056.
- [R11] McCaskey, Lyakh, Dumitrescu, Powers, Humble. XACC: A System-Level Software Infrastructure for Heterogeneous Quantum-Classical Computing. arXiv:1911.02452.
- [R12] Nguyen, Santana, Kharazi, Claudino, Finkel, McCaskey. Extending C++ for Heterogeneous Quantum-Classical Computing. arXiv:2010.03935.
- [R13] Hayashi, Adams, Young, McCaskey, Dumitrescu, Sarkar, Conte. Enabling Multi-threading in Heterogeneous Quantum-Classical Programming Models. arXiv:2301.11559.
- [R14] Chundury, Shehata, Kim, Gopalakrishnan Meena, Lu, Gottiparthi, Coello Perez, Mueller, Suh. Scaling Hybrid Quantum-HPC Applications with the Quantum Framework. arXiv:2509.14470.
- [R15] Mantha, Kiwit, Saurabh, Jha, Luckow. Pilot-Quantum: A Middleware for Quantum-HPC Resource, Workload and Task Management. arXiv:2412.18519.
- [R16] Sivarajah, Heidemann, Lawrence, Duncan. Tierkreis: a Dataflow Framework for Hybrid Quantum-Classical Computing. arXiv:2211.02350.
- [R17] Tierkreis documentation: https://quantinuum.github.io/tierkreis/
- [R18] Burgholzer et al. The Munich Quantum Software Stack. arXiv:2509.02674.
- [R19] Farooqi, Chung, Mete, Mansfield, Hoffmann, Mattsson, Schulz, Echavarria. Enabling Hybrid HPCQC Workflows with a Heterogeneous Software Stack. arXiv:2608.14827.
- [R20] Covalent documentation: https://covalent.readthedocs.io/en/latest/api/executors/qiskit.html
- [R21] Du et al. QCCP: a taskflow programming model for emerging computing scenario. EPJ Quantum Technology 12, 23 (2025). https://doi.org/10.1140/epjqt/s40507-025-00318-5
- [R22] Siddi Moreau et al. Managing Iterative Hybrid Quantum-Classical Optimization as a First-Class Scientific Workflow. arXiv:2609.30532.
- [R23] Döbler, Jattana. A Survey on Integrating Quantum Computers into High Performance Computing Systems. arXiv:2507.03540.
- [R24] Shehata et al. Quantum-HPC Software Stacks and the openQSE Reference Architecture: A Survey. arXiv:2604.20912.
- [R25] IBM, execution modes: https://quantum.cloud.ibm.com/docs/guides/execution-modes
- [R26] Qiskit Serverless: https://quantum.cloud.ibm.com/docs/guides/serverless and https://github.com/Qiskit/qiskit-serverless
- [R27] QRMI and the Slurm plugins: arXiv:2506.10052 and https://quantum.cloud.ibm.com/docs/guides/slurm-plugin
- [R28] NVIDIA, Slurm 26.11 plugins for QPUs: https://developer.nvidia.com/blog/integrating-qpus-into-supercomputing-centers-using-slurm-hres-plugins/
- [R29] Seelam et al. Reference Architecture of a Quantum-Centric Supercomputer. arXiv:2603.10970.
- [R30] CUDA-Q specification: https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/algorithmic_primitives.html
- [R31] Wennersteen et al. Towards a user-centric HPC-QC environment. arXiv:2509.20525.
- [R32] AWS Braket Hybrid Jobs: https://docs.aws.amazon.com/braket/latest/developerguide/braket-what-is-hybrid-job.html and https://docs.aws.amazon.com/braket/latest/developerguide/braket-batching-tasks.html
- [R33] Shirakawa et al. Closed-loop calculations of electronic structure on a quantum processor and a classical supercomputer at full scale. arXiv:2511.00224.
- [R34] Kim, Pascuzzi, Xu, Luo, Lee, Suh. Distributed Quantum Approximate Optimization Algorithm on a Quantum-Centric Supercomputing Architecture. arXiv:2407.20212.
- [R35] Zhou, Du, Tian, Tao. QAOA-in-QAOA: Solving Large-Scale MaxCut Problems on Small Quantum Machines. Physical Review Applied 19, 024027 (2023). arXiv:2205.11762.
- [R36] Esposito, Danzig. Hybrid Classical-Quantum Simulation of MaxCut using QAOA-in-QAOA. IPDPS Workshops 2024. arXiv:2406.17383.
- [R37] Li, Alam, Ghosh. Large-Scale Quantum Approximate Optimization via Divide-and-Conquer. IEEE TCAD 42 (2023). arXiv:2102.13288.
- [R38] Shaydulin, Ushijima-Mwesigwa, Safro, Mniszewski, Alexeev. Network Community Detection On Small Quantum Computers. Advanced Quantum Technologies 2(9), 2019. arXiv:1810.12484.
- [R39] Ushijima-Mwesigwa, Shaydulin, Negre, Mniszewski, Alexeev, Safro. Multilevel Combinatorial Optimization Across Quantum Architectures. ACM Transactions on Quantum Computing 2(1), 2021. arXiv:1910.09985.
- [R40] Robledo-Moreno et al. Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer. Science Advances 11, eadu9991 (2025). arXiv:2405.05068.
- [R41] SQD software: https://github.com/Qiskit/qiskit-addon-sqd and https://github.com/r-ccs-cms/sbd

Not verified: a published venue for [R1]; whether the QIR-EE branch that the IRIS quantum driver needs is publicly readable; the venues of several preprints beyond what arXiv states.

---

# 6. Evaluation plan

This section is a plan. Every number in it comes from a model or from provider documentation and is labelled so. Nothing was measured.

## 6.1 What a timing model says before anything is built

To find the regime in which each effect should appear, I wrote two small event simulations of the task graph of section 3.6. They are throwaway scripts outside the repository; M8 needs the same model again for the predictions drawn on its plots, so it is rewritten there with the experiment scripts. They assume zero classical time and zero runtime overhead, so they show what the schedule alone can give.

**Model 1: schedule.** Job time is lognormal with mean 1 and coefficient of variation CV. 16 blocks, $T=8$, $R=4$ (the controlled instance, 1024 jobs), except the torus (64 blocks). Each cell is a mean over 40 latency seeds.

| Block graph (colours), job slots | CV 0 | CV 0.5 | CV 1 | CV 2 |
|---|---|---|---|---|
| **Colour-synchronous time over dataflow time** | | | | |
| 4 x 4 grid (2), 4 slots | 1.00 | 1.02 | 1.04 | 1.13 |
| 4 x 4 grid (2), 16 slots | 1.00 | 1.04 | 1.09 | 1.15 |
| Ring of 16 (2), 16 slots | 1.00 | 1.07 | 1.13 | 1.22 |
| 8 x 8 torus (2), 64 slots | 1.00 | 1.10 | 1.19 | 1.32 |
| Random, degree at least 4 (4), 8 slots | 1.00 | 1.09 | 1.17 | 1.24 |
| Complete (16), any | 1.00 | 1.00 | 1.00 | 1.00 |
| **Sequential time over dataflow time** | | | | |
| 4 x 4 grid (2), 16 slots | 16.0 | 10.8 | 8.2 | 5.7 |
| Any graph, 1 slot | 1.00 | 1.00 | 1.00 | 1.00 |

(M7 wrote this model again and measured its regimes on the mock: section 9.6.)

**Model 2: coalescing.** A job with $n$ circuits takes a queue delay plus $T_{\mathrm{fixed}} + n\,T_{\mathrm{circ}}$, with $T_{\mathrm{fixed}}=2$ s and $T_{\mathrm{circ}}=0.358$ s, the provider's documented estimate at 1024 shots (section 2.6). Dataflow schedule, controlled instance on the 4 x 4 grid. Each cell is the time with `max_batch = 1` over the time with the given `max_batch`.

| Regime | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| 1 slot, no rate limit, no queue delay | 1.73 | 2.74 | 3.82 | 3.88 |
| 1 slot, queue delay of mean 10 s | 1.93 | 3.68 | 6.38 | 6.57 |
| One submission per 12 s (5 per minute), no queue delay | 2.00 | 3.99 | 7.97 | 15.8 |
| One submission per 12 s, 3 slots, queue delay of mean 10 s | 2.00 | 3.94 | 6.80 | 7.01 |
| 3 slots, no rate limit, no queue delay | 1.73 | 2.66 | 2.24 | 1.79 |
| Unlimited slots, no rate limit | 1.00 | 1.00 | 1.00 | 1.00 |

What this means for the paper:

1. **The largest effect is concurrency itself**, 6 to 16 times over sequential execution with 16 slots. A hand-written MPI code with nonblocking submission gets it too. It is not a PaRSEC result.
2. **Dataflow over colour barriers is a modest effect**: nothing at constant latency, 4 to 19% at CV 1, 13 to 32% at CV 2, and nothing at all when every block touches every other. It cannot carry the paper alone.
3. **Coalescing is the large runtime-side effect**, but only under contention: about 2 to 7 times with one slot, close to the batch size under a submission-rate limit, and nothing with spare capacity. It is also not monotone: with 3 slots and no rate limit, a batch limit of 4 beats 16, because one job that takes the whole queue leaves the other slots to singletons. The limit is a parameter to study, not a "bigger is better" switch.
4. **What is specific to the runtime** is that capacity, progress and coalescing need no application code and apply to any application, and that this costs nothing against hand-written MPI on the same schedule. A hand-written code can reach each number above. That is the honest shape of the claim.

## 6.2 Systems compared

All systems share the same QAOA kernels (`qaoa_core`), the same backend objects (the mock, linked without starting the runtime in A and B), the same latency model and seeds, the same capacity limit and the same submission interval. With the mock they therefore compute the same sequence of assignments and the same jobs, and only time differs.

| System | What it is |
|---|---|
| **A**, sequential hybrid | The reference of section 3.7 with blocking calls: submit one job, wait, continue, in the order (round, colour, block, step, evaluation). One process. |
| **B**, MPI + QPU | Blocks on rank `b mod P`; rank 0 owns the backend. Each rank drives its blocks of the current colour as independent state machines with nonblocking requests to rank 0, both evaluations of a step in flight together. Rank 0 submits under the same capacity limit, with a helper thread waiting on the backend. Ranks exchange boundary values at the end of each colour. This is the bulk-synchronous structure of a multicolour Gauss-Seidel code, and what I would expect a competent MPI programmer to write first. |
| **C-sync**, control | The PaRSEC application with a control dependency that joins each colour before the next starts (control gathers exist in PTG: `tests/dsl/ptg/controlgather/`). Same schedule as B, on the runtime. (As built in M8, a binary tree of `JOIN` tasks per colour: a control gather over a named range counts its inputs wrongly on this base; section 9.7.) |
| **C**, PaRSEC, basic QPU support | The task graph of section 3.6, `max_batch = 1`. |
| **D**, PaRSEC with coalescing | The same binary with `max_batch > 1`. |
| **C-block**, experiment E2 only | The quantum call made synchronously inside a CPU task body, holding a worker. The integration that property 1 rules out. |
| **B-async**, if time allows | Event-driven MPI in which a block starts when its neighbours' updates arrive: the dataflow schedule written by hand. Expected to match C. Its size in lines is reported next to B's and the JDF's. (Not built in M8; section 9.7.) |

C-sync is what keeps the comparison fair. C-sync against B measures the runtime's overhead on equal schedules. C against C-sync measures the dataflow alone. Without it, a gain of C over B would mix the two.

B does not coalesce. A hand-written rank 0 could; point 4 of section 6.1 applies.

**An existing hybrid runtime as a baseline.** Not in the first submission. Section 5.7 gives the reason per system: the OmpSs-2 extension is not public, Q-IRIS is single-node and would need the application ported to another circuit representation, and DQR cannot express dependencies. The comparison with existing runtimes is the sourced capability tables of section 5.

## 6.3 Metrics and where each number comes from

| Metric | Source |
|---|---|
| End-to-end time to solution | Wall clock from start to the end of the wait, setup reported apart. Since all systems compute the same result, this is time for equal work. |
| QPU queue latency, execution latency | Per job, from the timestamps in the result header (section 2.4): the mock's own, or the provider's on hardware. Always two separate columns. |
| Time pending in the runtime | Per task, from the hook to the backend submission. This is PaRSEC's own queue under the capacity limit, and is reported apart from the provider's queue. |
| Classical computation time | Per task class, from the PaRSEC trace; the same kernels are timed the same way in A and B. |
| Submission overhead | Hook entry to return of the backend's `submit`, minus the time pending. |
| Runtime scheduling overhead | Per job: result available at the backend to start of the successor task. Covers the hand-off, the scheduler, and the message when the successor is on another rank. B's counterpart is its message latency. |
| CPU idle time | Per worker, wall time minus executed intervals, from the trace; process CPU time from `getrusage`, to show that idle is not spinning. |
| GPU idle time | Not applicable to QAOA (section 3.4). Reported only for the coexistence test. |
| Overlap of classical and quantum execution | Fraction of wall time with at least one CPU task executing and at least one job outstanding, from trace intervals. Expected near zero for QAOA; meaningful in E2. |
| Outstanding quantum tasks | Pending and in-flight counts logged by the device at every change: a time series, plus mean and maximum. |
| QPU utilization | Busy slot-seconds over slots times makespan. Also the time a slot was free while a task was pending, which must be zero outside the submission interval. |
| MPI communication time | From the communication thread's trace events (`RDEP_ACTIVATE`, `RDEP_DATA_CTL`, `RDEP_DATA_PLD_SND`, `RDEP_DATA_PLD_RCV`, registered at `parsec/remote_dep_mpi.c:1460-1472`); timers around the MPI calls in B. |
| Optimization iterations | Fixed at $R\times K\times T$ by construction; reported with the number of evaluations actually run (not skipped) and failed. |
| Final cut, approximation ratio | Against the exhaustive optimum up to 24 vertices. For larger graphs the optimum is unknown; report the cut and its ratio to exact block-coordinate ascent (each block solved by enumeration), which is the best this decomposition can do. |
| Provider jobs and QPU seconds | Counted by the device; on hardware, the provider's reported usage. |

## 6.4 Experiments

Mock backend throughout, except E7. "Regime" refers to section 3.4 and section 6.1: classical work per job is 50 to 100 microseconds, so every result below should hold for any job latency well above a millisecond. To check that, E3 and E5 are run at two time scales (mean job time 0.1 s and 1 s) and must give the same ratios.

| # | Hypothesis | Varied | Systems | Regime where an effect is expected |
|---|---|---|---|---|
| E1 | A quantum task costs the runtime under a millisecond and no worker. | Zero-latency mock; number of tasks; 1 to 36 workers | C, and the same graph with CPU tasks | Everywhere. This is an overhead measurement, not a speedup. |
| E2 | Without blocking, classical tasks proceed while jobs are outstanding; with blocking they stall once outstanding jobs reach the number of workers. | Synthetic graph (not QAOA): classical work per job from $10^{-4}$ to 10 times the job latency; outstanding jobs from 1 to twice the workers | C, C-block, A | Classical work within two orders of magnitude of the latency, or outstanding jobs at or above the worker count. QAOA sits at about $10^{-5}$ and is marked on the plot as outside this regime. |
| E3 | Dataflow finishes sooner than colour-synchronous execution when latency varies and slots are available, and not otherwise. | Latency CV 0, 0.5, 1, 2 (lognormal), and a replayed hardware trace if E7 yields one; slots 1 to 32; block graph (grid, ring, random, complete) | A, B, C-sync, C | CV above 0, slots at or above twice the blocks per colour, sparse block graph. Predicted size: the first table of section 6.1. Expected null results: CV 0, one slot, complete block graph. |
| E4 | The device keeps slots full whenever quantum work is ready. | Slots and `max_outstanding` | B, C-sync, C | Few slots. Model: utilization 0.97 to 1.00 for dataflow against 0.86 to 0.98 for barriers at 4 slots. Shown as a time series of outstanding jobs. |
| E5 | Coalescing raises throughput under contention and is neutral without it. | `max_batch` 1 to 16; slots 1, 3, unlimited; submission interval 0 and 12 s (scaled); $T_{\mathrm{fixed}}/T_{\mathrm{circ}}$ of 0, 1, 5.6, 20; queue delay | C, D | Contention or a rate limit. Predicted size: the second table of section 6.1, including the non-monotone row. Expected null result: unlimited slots. |
| E6 | Spreading the graph over ranks adds negligible time. | Ranks 1 to 8 (16 where machines allow); vertices 64 to 1024 at block size 16, so 4 to 64 blocks | B, C | None: the expected result is a flat line, with MPI time per job under a millisecond against jobs of a second. It is an overhead bound, and the text must not present it as scaling. |
| E7 | The hardware path works end to end, and coalescing reduces jobs and billed time on a real device. | See section 6.5 | C, D | Physical QPU. |
| E8 | A failing backend delays the run but does not stop it. Optional. | Injected failure rate 1% and 10% | C | |

Each mock configuration is run at least five times with different latency seeds and the same algorithm seed, and reported as a median with its range. Next to each measured curve the plots show the model's prediction for the same latency samples, drawn differently and labelled as a model.

(As run in M8, section 9.7: the full grids at mean job times of 20 ms (E3) and 10 ms (E5), with the ratios checked at 0.1 s and 1 s; blocks of 8 vertices in E3 and of 12 in E6; E6 on one node; three seeds where the job times do not vary or there is one place. On the grid and the ring the gain of E3 is largest at one place per block of a colour, not at twice that.)

Machines: methane for single-node runs (36 cores, ranks bound to disjoint cores); leconte for the CUDA coexistence test, one rank per GPU; a multi-node run for the MPI-time measurement of E6 if nodes are available.

## 6.5 Hardware run (E7)

Depends on the unknown QPU access. Planned for the validation instance (40 vertices, 4 blocks of 10, 64 evaluations), assuming its four blocks form a ring:

| Run | Provider jobs | QPU time by the documented estimate | Wall time floor from the 5-per-minute limit |
|---|---|---|---|
| C, one circuit per job | 64 | about 2.5 minutes | about 13 minutes |
| D, `max_batch = 4` | 17 (model) | about 1 minute (model) | about 3.3 minutes (model) |

Both together take about 3.5 of the 10 free minutes per 28 days, which leaves room for one repeat of either. Every figure in this table is a projection from documentation and from the model, and rests on the assumption that the 2 s overhead is paid once per job.

What the run is for:

- Showing that submission, polling, results and failure paths work against a real provider.
- Recording per-job timestamps. The measured queue and execution latencies become a trace that the mock can replay in E3, labelled as mock with replayed latencies.
- One C run and one D run: provider jobs, billed seconds and wall time. Two runs are an observation, not a statistic, and are presented as such.

What it is not for: comparing A, B, C and D on hardware. Four runs would use the whole allowance, and queue variation between runs would exceed the differences.

## 6.6 Reporting rules

- Every result file records `backend = mock` or `hardware`, the latency model and its parameters, all seeds, the commit, the configure options, the MCA parameters and the machine. Raw results are kept unmodified outside the source tree.
- Mock and hardware results never share a curve or a table column. A mock replaying hardware latencies is still mock.
- A model prediction is drawn and labelled as a prediction. Speedups stated in the text are measured ones.
- Provider queue delay, time pending in the runtime, and runtime overhead are three separate quantities and are never summed under "PaRSEC".
- Null results stay in: E3 at CV 0 and on the complete block graph, E5 with unlimited slots, E6 throughout.
- If an improvement is small, the run's breakdown says why. The first number to look at is the QPU-bound fraction: total job time over slots times makespan.

## 6.7 Threats to validity

- The mock's latency model is a model. The replayed trace from E7 is the only link to real behaviour, and it is one day on one device.
- The mock samples exactly. Hardware noise changes the optimizer's path and the final cut, not the runtime's behaviour; E7 reports the difference in cut without interpreting it.
- One QPU. Nothing here supports claims about several. (Several QPUs per context are built, section 9.8, and three experiments ran on them on 2026-11-27, with mock QPUs on one and on two nodes: "The checks that were left, and the experiments on several QPUs" there. E1 to E6 as reported in section 9.7 are still on one QPU.)
- Blocks of at most about 20 qubits on the mock. Larger blocks change the classical side (section 3.4) and are not evaluated.
- The model of section 6.1 ignores classical time and runtime overhead. If measurements depart from it, the departure is itself a result to explain.
- The mock computes its answers on the machine under test: 7 to 12 ms per job at 16 qubits (section 9.2). A job whose sampling outlasts its modelled time ends late, and the latency measured is then not the one modelled. The mock counts such jobs; a run is reported only when the count is zero, with as many helper threads as slots and cores for them that the workers do not use. (In M8 the helper threads shared the cores of rank 0, and the 11 runs of E3 that had a late job were run once more; section 9.7.)

---

# 7. Milestones

## 7.1 The deadline

The project fact was unknown, so I looked it up. The ISC 2027 research-paper page (https://isc-hpc.com/submissions/research-paper/, read on 2026-11-21) gives:

- Abstract: **December 14, 2026**. Full paper: **December 21, 2026**, marked "no extension". Notification: March 15, 2027.
- Up to 10 pages, references not counted, IEEE template, double-anonymous review.
- Topic areas include "Runtime Systems for HPC" and "Integration of Quantum Computing and HPC".

Another page of the same site (https://isc-hpc.com/submissions/) says the call for research papers is closed, and carries one table with the year 2027 on the same December date. Both contradict the dates above and look like stale text. Please confirm the deadline before relying on it.

That is 30 days from today, with the abstract in 23.

My effort estimates, in focused working days, made before any code exists:

| | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | Paper |
|---|---|---|---|---|---|---|---|---|---|
| Days | 3 | 3 | 4 | 3, plus access | 3 | 4 | 2 | 7 | 6 or more |

The total is 35 days or more for a 30-day window, before any slip, and each milestone also waits for your approval. **The full scope does not fit.** A reduced scope fits only with no slack:

| Dates | Work | Checkpoint |
|---|---|---|
| Nov 22 to 24 | M1 | Nov 24: hand-off completion works in PTG, DTD and MPI runs. Decide hand-off or direct completion. |
| Nov 25 to 27 | M2 | |
| Nov 28 to 30 | M5 | Nov 30: the reference passes; the estimates of section 3.4 are replaced by measurements. |
| Dec 1 to 3 | M3, must-have part | |
| Dec 4 to 7 | M6 | **Dec 7: the distributed run equals the reference on 1, 2 and 4 ranks. Go or no-go for ISC.** |
| Dec 8 to 9 | M7 | |
| Dec 8 to 13 | Baseline B, C-sync, experiments E1, E3, E4, E5 | Dec 13: the headline numbers exist. |
| Dec 14 | Abstract due | |
| Dec 10 to 20 | Writing, remaining experiments | |
| Dec 21 | Paper due | |
| From Dec 1, cut off Dec 12 | M4 and E7, only if QPU access exists | Not on the critical path. |

**Order.** The sequence M1 to M8 of the stage 3 prompt is kept, with two changes that the audit supports:

- **M5 moves before M3.** The reference needs only the sampler from M2. Doing it early fixes the application's arithmetic before the runtime work that depends on it, and gives the measured time budget.
- **M4 leaves the critical path.** Nothing depends on it, and it is the only milestone with unknowns outside the code (access, network, provider changes).

One more safeguard follows from section 1.9: M6 can run on M1 and M2 alone, because the submit helper is the same and only the body's type changes from CPU to QPU. If M3 slips, the application and the experiments still run, on option C of section 1.9, and the paper's claims shrink to match.

## 7.2 Names fixed by this plan

| Thing | Name |
|---|---|
| Configure option | `PARSEC_DEVICE_WITH_QPU`, off by default, next to `PARSEC_GPU_WITH_*` (`CMakeLists.txt:195-201`) |
| Build macros | `PARSEC_HAVE_QPU` (component built); `PARSEC_HAVE_DEV_QPU_SUPPORT` (type known to the PTG compiler, from M3) |
| Device type | `PARSEC_DEV_QPU`, bit 5; name `qpu` |
| Component | `parsec/mca/device/qpu/` |
| Public header | `parsec/mca/device/qpu/device_qpu.h`: `parsec_qpu_submit`, the circuit and result formats |
| MCA parameters, device | `device_qpu_enabled` (number of QPU modules, default 0), `device_qpu_backend` (default `mock`), `device_qpu_allow_hardware` (default 0), `device_qpu_owner_rank` (default 0), `device_qpu_max_outstanding`, `device_qpu_submit_interval`, `device_qpu_max_batch` (default 1, no coalescing), `device_qpu_max_retries`, `device_qpu_job_timeout`, `device_qpu_unavailable_timeout`, `device_qpu_poll_min`, `device_qpu_poll_max` |
| MCA parameters, mock | `device_qpu_mock_seed`, `device_qpu_mock_slots`, `device_qpu_mock_delay`, `device_qpu_mock_fail` (as built in M2: `device_qpu_mock_faults`, and `device_qpu_mock_queue_delay`, `device_qpu_mock_threads`, `device_qpu_mock_max_qubits`; on the device side `device_qpu_retry_backoff` was added; section 9.2. M7 added `device_qpu_mock_job_delay` and `device_qpu_mock_max_circuits`, and the fault `cancelled`; section 9.6) |
| MCA parameter, adapter | `device_qpu_adapter_address` |
| Tests | `tests/runtime/qpu/`, `tests/apps/qaoa/`; CTest names `runtime/qpu/<name>` and `apps/qaoa[/<name>]`, with `:mp` for MPI |
| Experiment scripts and raw results | A directory outside version control, as with the existing experiment directories of this working tree |

## 7.3 The milestones

Paths are relative to the repository root. "Baseline" means the list of existing tests that pass at the base commit (section 4.7).

### M0: source inspection and design approval

| Item | Description |
|---|---|
| Objective | Stages 1 to 3 of this document, approved. |
| Files | `docs/design/parsec-qpu-qaoa-plan.md` |
| Dependencies | None |
| Difficulty | Done, except approval |
| Risks | The four unknown project facts. The deadline found in section 7.1 needs confirming. |
| Tests | None |
| Acceptance criteria | You approve this stage and state the scope: the reduced plan of section 7.1, or the full plan for a later deadline. |

### M1: minimal asynchronous QPU task integration

Done on 2026-11-22; section 9.1 has the result and the departures from this table.

| Item | Description |
|---|---|
| Objective | A task whose hook returns ASYNC is completed by a progress thread through the hand-off of section 1.5, in PTG and in DTD, on one and on several ranks. The hook is still CPU-typed. The mock only delays and returns fixed counts. |
| Files | New: `parsec/mca/device/qpu/ValidateModule.CMake`, `device_qpu.h`, `device_qpu_internal.h`, `device_qpu_component.c` (parameters; starts and stops the progress thread; registers no device yet), `device_qpu_progress.c` (thread, execution stream, submission list, hand-off), `device_qpu_backend.h`, `device_qpu_backend_mock.c`. New tests in `tests/runtime/qpu/`: `CMakeLists.txt`, `Testings.cmake`, `qpu_test_common.c`, `qpu_test_common.h`, `qpu_chain.jdf`, `dtd_qpu_chain.c`, `qpu_overlap.jdf`, `qpu_termination.jdf`, `qpu_stress.jdf`, `qpu_init_fini.c`, `check_no_hardware.cmake`. Edited: `CMakeLists.txt` (the option), `parsec/include/parsec/parsec_options.h.in` (`PARSEC_HAVE_QPU`), `tests/CMakeLists.txt` (`:94-95`), `tests/runtime/CMakeLists.txt`, `tests/runtime/Testings.cmake`. |
| Dependencies | M0 |
| Difficulty | High. It exercises the least documented part of the runtime: scheduling from a thread that is not a worker. |
| Risks | Hazards H1, H6 and H9 of section 4.3. Sharing thread 0's mempools (section 1.10). Whether DTD tasks complete correctly through the hand-off. Whether the thread sanitizer is usable on the runtime at all. |
| Tests | `runtime/qpu/chain`, `chain:dtd`, `order`, `overlap`, `termination` with `:mp`, `:dynamic` and `:loop`, `stress`, `init_fini`, `no_hardware_in_tests`. Sanitizer runs by hand. |
| Acceptance criteria | The tests pass 100 times in a row on 1, 2 and 4 ranks. Sanitizer reports are read and each is fixed or explained in this document. The existing suite is at baseline with the option off and on. The choice between hand-off and direct completion is recorded here with the evidence. |

### M2: mock backend and correctness tests

Done on 2026-11-22; section 9.2 has the result and the departures from this table.

| Item | Description |
|---|---|
| Objective | The full mock of section 2.5 (delay distributions, latency replay, slots, failure and timeout injection, exact sampler, per-job seeds, mock flag). The circuit and result formats of section 2.4. Admission under `max_outstanding` and the submission interval. Retry, timeout and unavailability handling of section 2.9. The hardware guard. |
| Files | New: `parsec/mca/device/qpu/device_qpu_sampler.c`, `device_qpu_circuit.c`; `tests/runtime/qpu/qpu_sampler_test.c`, `qpu_mock_test.c`, `qpu_failure.jdf`, `qpu_guard.c`. Edited: `device_qpu_backend_mock.c`, `device_qpu_progress.c`, `device_qpu.h`, `tests/runtime/qpu/CMakeLists.txt`, `Testings.cmake`. |
| Dependencies | M1 |
| Difficulty | Medium |
| Risks | A bit-order or sign error in the sampler, which every later result would inherit. Statistical tests that fail by chance; fixed seeds remove this. Delays that are inaccurate on a loaded machine, which matters for the experiments more than for the tests. |
| Tests | `runtime/qpu/sampler`, `mock`, `capacity`, `failure`, `hardware_guard` |
| Acceptance criteria | The tests pass. The sampler matches analytic states to $10^{-12}$. Counts are identical across submission orders. The time to sample a 20-qubit, two-layer QAOA circuit is measured and written into section 3.4. |

### M3: native QPU device integration

Done on 2026-11-23: the must-have part named below the table, and the test with CPU, GPU and QPU incarnations. Section 9.4 has the result and the departures from this table.

| Item | Description |
|---|---|
| Objective | `PARSEC_DEV_QPU` as a device type with its own module: selected as an incarnation, load and capacity in the module, owner rank, latency estimate, statistics and trace events. The bodies of the M1 tests become `[type=QPU]`. |
| Files | Edited: `parsec/mca/device/device.h` (`:64-78`), `parsec/mca/device/device.c` (`:93-103`, `:299`, `:861-890`, `:1206-1210`), `parsec/scheduling.c` (`:151-155`, `:534-537`), `parsec/interfaces/dtd/insert_function.c` (`:1643`, `:2553`, `:3480`) with its headers, `parsec/mca/device/CMakeLists.txt` (after `:22`), `parsec/include/parsec/parsec_options.h.in` (`:147-153`). New: `parsec/mca/device/qpu/device_qpu_module.c`; `tests/runtime/qpu/qpu_discovery.c`, `cpu_gpu_qpu.jdf`; `tests/dsl/ptg/qpu_fallback.jdf` with its lines in `tests/dsl/ptg/CMakeLists.txt` and `Testings.cmake`; `tests/profiling/check-qpu.py` with its lines in `tests/profiling/Testings.cmake`. |
| Dependencies | M1, M2 |
| Difficulty | High. These are the only edits to shared runtime code. |
| Risks | A side effect on the CUDA, HIP or Level Zero paths from a type inside the "any" mask. A task class whose only incarnation is QPU on a rank whose module refuses work. The DTD change for classes with several incarnations, which section 2.12 left undesigned. |
| Tests | `runtime/qpu/discovery`, `dsl/ptg/qpu_fallback`, `runtime/qpu/cpu_gpu_qpu:gpu`, `profiling/qpu_*`, and every M1 and M2 test again with QPU-typed bodies |
| Acceptance criteria | Each of the seven properties has a passing test: 1 `overlap`, 2 `qpu_fallback` and `discovery`, 3 `capacity`, 4 `termination`, 5 `profiling/qpu_*`, 6 `failure`, 7 the build configurations of section 4.7. The GPU suite on leconte is at baseline with the option on. |

For the reduced plan, the must-have part of M3 is the type, the module, selection, load, capacity, the owner rank, statistics and PTG. The DTD slot for classes with several incarnations, the per-job trace events and the change at `device.c:299` for a second QPU can follow.

### M4: real QPU provider adapter

| Item | Description |
|---|---|
| Objective | The adapter client in C, the OpenQASM emitter, and the provider adapter program of section 2.6, for the provider your access gives. |
| Files | New: `parsec/mca/device/qpu/device_qpu_backend_adapter.c`, `device_qpu_qasm.c`; `tools/qpu/parsec_qpu_adapter_ibm.py`, `tools/qpu/README.md`; `tests/runtime/qpu/qpu_adapter_loopback.c`, `qpu_qasm_test.c`. Edited: the two files of `tests/runtime/qpu/` that list tests. |
| Dependencies | M2. Independent of M3, M5, M6 and M7. |
| Difficulty | Medium for the code. Unknown for everything outside it. |
| Risks | No access, or access too late. Compute nodes that cannot reach the provider. The provider replacing the program jobs are submitted to (section 2.6). The submission-rate limit. Queue times. Charges for cancelled jobs. Circuit depth after transpilation of a 10-qubit block. |
| Tests | `runtime/qpu/adapter_loopback`, `runtime/qpu/qasm`. Then, by hand and with your explicit go-ahead for each submission: one two-qubit circuit, then E7. |
| Acceptance criteria | The loopback tests pass. The two-qubit circuit returns counts from a physical device. E7 completes. A search of all logs at full verbosity finds no credential. |

(Revised in section 10.4, on 2026-11-26: M4 is split into M4a, the adapter with Qiskit Aer and the OpenQASM 3 reader, and M4b, IBM hardware. Section 10.5 adds M9, several QPUs per context; M10, a backend in C on QRMI and the Qiskit C API; and M11, a CUDA-Q backend.)

### M5: sequential QAOA reference

Done on 2026-11-22; section 9.3 has the result and the departures from this table.

| Item | Description |
|---|---|
| Objective | The algorithm of section 3 as a sequential program, with exhaustive search, on the mock sampler. It is the correctness reference and, with blocking backend calls, baseline A. |
| Files | New, all in `tests/apps/qaoa/`: `CMakeLists.txt`, `Testings.cmake`, `qaoa_graph.c`, `qaoa_graph.h` (graphs, generators, partition, colouring), `qaoa_core.c`, `qaoa_core.h` (local problem, circuit, evaluation of counts, SPSA, accept rule), `qaoa_exact.c`, `qaoa_reference.c`, `qaoa_math_test.c`. Edited: `tests/apps/CMakeLists.txt`, `tests/apps/Testings.cmake`. |
| Dependencies | M2 |
| Difficulty | Medium |
| Risks | Sign and bit-order errors, which the tests of section 4.5 target. Optimizer settings that give a poor cut, which affects solution quality and no runtime claim. The partition is a simple built-in method; no partitioning library is added. |
| Tests | `apps/qaoa/math`, `apps/qaoa/reference` |
| Acceptance criteria | Both pass on every instance of section 4.5. The per-task times of section 3.4 are replaced by measured ones. |

### M6: distributed QAOA on PaRSEC

Done on 2026-11-23; section 9.5 has the result and the departures from this table.

| Item | Description |
|---|---|
| Objective | The task graph of section 3.6 as one PTG taskpool, with its data collections and driver. |
| Files | New in `tests/apps/qaoa/`: `qaoa.jdf`, `qaoa_data.c`, `qaoa_data.h`, `testing_qaoa.c`. Edited: `tests/apps/qaoa/CMakeLists.txt`, `Testings.cmake`. |
| Dependencies | M5, and M3. M1 and M2 are enough for the fallback of section 7.1. |
| Difficulty | High: neighbour tables in dependency expressions, the gather chain, skip flags, and six kinds of flow data with their arenas. |
| Risks | A dependency the JDF cannot express as written in section 3.6. Arena sizes for circuits of different sizes. A flaw in the colour rule that only some graphs expose, which is why `GATHER` asserts versions. |
| Tests | `apps/qaoa`, `apps/qaoa:mp` on 2 and 4 ranks, `apps/qaoa:fail` |
| Acceptance criteria | Assignments equal the reference bit for bit on every instance, for every rank count, delay seed and slot count tested. Job counts equal the reference's. No rank but the owner opened a backend. |

### M7: QPU-aware scheduling optimization

| Item | Description |
|---|---|
| Objective | The coalescing of section 2.8, and the mock's cost model with a fixed part and a part per circuit. |
| Files | Edited: `parsec/mca/device/qpu/device_qpu_progress.c`, `device_qpu_module.c`, `device_qpu_backend_mock.c`. New: `tests/runtime/qpu/qpu_coalesce.jdf`. |
| Dependencies | M3. M6 for the application-level test. |
| Difficulty | Medium |
| Risks | The policy is not monotone in the batch limit (section 6.1). A job that fails after some of its tasks' results were taken. Which tasks may share a job. |
| Tests | `runtime/qpu/coalesce`, `apps/qaoa:coalesce` |
| Acceptance criteria | The tests pass. Results are identical with coalescing on and off. On the mock, two regimes of the second table of section 6.1 are reproduced within 10%, or the difference is explained. |

Done on 2026-11-23; section 9.6 has the result and the departures from this table.

### M8: performance experiments and documentation

| Item | Description |
|---|---|
| Objective | Baselines B and C-sync, the experiments of section 6.4, and the documentation. |
| Files | New: `tests/apps/qaoa/qaoa_mpi.c` (baseline B); `tests/runtime/qpu/qpu_bench.jdf` (E1 and E2). Edited: `tests/apps/qaoa/qaoa.jdf` (the C-sync option), the test lists. Documentation in `parsec/mca/device/qpu/device_qpu.h` and `tools/qpu/README.md`. Deviations recorded in this document. Experiment scripts and results outside version control. |
| Dependencies | M6 for E3, E4 and E6; M7 for E5; M4 for E7; M3 for E1 and E2 |
| Difficulty | Medium. The risk is in the results, not the code. |
| Risks | Effects smaller than the model predicts. Timer noise on shared machines. A baseline B that is unfair in either direction. |
| Tests | B and C-sync must give the reference's assignments, like every other system. |
| Acceptance criteria | Each experiment has raw data, a plot with the model's prediction on it, and a paragraph that states what was found, null results included. A repeat of a subset reproduces the medians within their ranges. |

## 7.4 What a strong submission must have

| Class | Items |
|---|---|
| **Must have** | M1, M2, M5, M6. The must-have part of M3, so that the paper is about a device and not a library. Baselines A, B and C-sync. Experiments E1, E3 and E4 with their null results. The correctness evidence of sections 4.5 and 4.6. |
| **The optimization that matters** | M7 with E5. By the model it is the only runtime-side effect larger than tens of percent, and it is the one that needs the device. I would treat it as must-have for a research paper and cut something else first. |
| **Important** | One hardware run (M4, E7). A reviewer in the quantum track will look for it; without it the paper has to say plainly that all results come from a mock. E2. The DTD path for QPU chores. The CPU, GPU and QPU coexistence test. Trace-based figures. |
| **Future work** | Several QPUs and latency-informed selection (designed in section 10.3 as M9). Critical-path ordering. Optimistic or versioned block updates. A C-only provider backend. B-async. Direct completion. Registers wider than 64 qubits. SQD, VQE and other applications. |

## 7.5 Rules every milestone has to pass

| Rule | How it is checked |
|---|---|
| C consistent with the component; PaRSEC infrastructure reused; no invasive change elsewhere | Edits outside `parsec/mca/device/qpu/` are limited to the files listed under M1 and M3. Anything more is recorded here first. |
| QPU support optional; CPU, CUDA and HIP unchanged; no mandatory Python or SDK | The build configurations of section 4.7. The provider adapter is not built, installed or needed by any test. |
| Diagnostics; no credentials in code, logs or test output | Credentials live only in the adapter's environment (section 2.6). The dummy-token check of `adapter_loopback`. |
| Bounded submissions; nothing busy-waits | `capacity`; the progress thread's CPU-time check in `overlap`. |
| Real backend opt-in | Section 4.2. |
| Termination detection not bypassed; no counter adjusted by hand | The component calls no termination-detector function and writes no task count. `termination` in all variants. |
| Existing tests keep passing | The baseline of section 4.7, compared after every milestone. |

Implementation proceeds one milestone at a time, on `qinglei/hybrid_HPC_QC`, each built and tested before the next starts.

---

# 8. Recommendation

**1. Is this architecture suitable for PaRSEC?**

Yes. PaRSEC already has the three things a QPU needs: hooks that return ASYNC with one completion entry point, termination accounting that keeps a task counted until it is released, and host-copy handling for incarnations that are not GPUs (section 1). The design adds a thin, optional component and edits about a dozen places in five shared files (section 2.11).

Two reservations. A QPU fits the device layer only in part: memory and flop rates mean nothing for it, and what it gains from being a device is selection, load, capacity and statistics. And the design needs a second thread that is not a worker to schedule tasks, for which the communication thread is the only precedent. If the maintainers do not want that, the fallback is option C of section 1.9, with a smaller claim.

**2. Is the implementation feasible before the deadline?**

Not the full scope. By the ISC website the paper is due on December 21, 2026, 30 days from today (section 7.1, to be confirmed). My estimate for everything is 35 working days or more.

A reduced scope fits only with no slack: PTG only, the mock only, hardware if it happens to work. I would not plan on it without accepting a real chance of either missing the date or submitting with thinner results than the paper needs.

The first two weeks of work are the same under both plans. So start M1 now, and decide on December 7 with evidence: if the distributed run does not yet match the reference then, move the target. The next candidate deadlines I could find are not published yet. Search-result summaries, which I did not confirm on the conference sites, put last cycle's full-paper deadlines at April 8, 2026 for SC26 and April 13, 2026 for IEEE Quantum Week 2026, and the ISC 2027 poster deadline at January 20, 2027.

**3. What are the three biggest technical risks?**

- **Completing a task from a thread that is not a worker.** The hand-off is untested, a window after the hook exists (H1 in section 4.3), and DTD tasks and shared mempools are unverified. Everything else stands on this. It is why M1 comes first and why direct completion is kept as a fallback.
- **The measured effect may be small, or may belong to the algorithm and not the runtime.** The model of section 6.1 gives dataflow 0 to about 30% over barriers, and coalescing a large gain only under contention. A hand-written MPI code can reach each number. The paper stands only if the claim is shaped as in point 4 of section 6.1, with the C-sync control and the null results in it.
- **The real-QPU path.** Access is unknown. The free allowance is 10 minutes per 28 days at 5 job submissions per minute, and the provider is replacing the program that jobs are submitted to (section 2.6). This is why M4 is off the critical path and the adapter is a separate program.

One risk is not technical but belongs here: positioning. OmpSs-2 and Q-IRIS already have a QPU device in a task runtime (section 5), and Q-IRIS has announced multi-node support as future work.

**4. Which implementation approach should be used?**

- The thin device component (option B of section 1.9), built on the completion mechanism of option C, so M1 can be tested alone.
- Completion by hand-off to a worker, with one progress thread per process sleeping in the backend's wait call.
- `parsec_qpu_submit` as the only entry point, so the PTG compiler is not changed.
- The mock first, with its exact sampler. The provider adapter as a separate Python program on the official SDK if the access is IBM's.
- One owner rank per QPU. Coalescing as the one scheduling optimization.
- The application as one PTG taskpool.

**5. Is distributed QAOA the best first example, compared with SQD?**

It is the best first example. It is not the best example for the claim the project started from.

QAOA suits the building of the device. It issues many small jobs, so it exercises capacity, ordering, coalescing and failure handling. The mock can sample its circuits exactly, and a sequential reference can be matched bit for bit. It takes days. But its classical work per job is about $10^{-5}$ of the job time (section 3.4), so it cannot show classical and quantum work overlapping.

SQD is the opposite. The published runs put classical time at one to four times the quantum time per iteration. Open loop: each diagonalization took "about 1.5 hours" on 64 Fugaku nodes, 100 of them in parallel, against a quantum runtime of about 45 minutes [R40]. Closed loop: diagonalizations of about 2 to 20 minutes on 384 to 152,064 nodes, against a quantum step of "several tens of minutes" [R33]. The ratios are the research pass's arithmetic on those figures. That closed-loop run overlapped the two sides by hand, for two populations, and reports the reduced idle time as a result; a task runtime would do it without that code. Open building blocks exist, including a distributed eigensolver [R41].

SQD is still the wrong first step. It needs chemistry inputs and a distributed eigensolver that is itself an MPI program. It issues only a few large jobs per iteration, so capacity and coalescing have nothing to do. And at the sizes where its classical side is heavy, its circuits are beyond what an exact mock can sample.

So: QAOA now. If the target moves past December, SQD at small scale is the second application, and it is the one that would let the paper show CPU, GPU and QPU work overlapping.

One naming point from section 5.3: "Distributed QAOA" is the name of an existing algorithm [R34]. The application needs another name.

**6. What is the smallest publishable contribution?**

A QPU device class for a distributed dataflow runtime, with:

- quantum tasks that hold no worker, in one task graph across MPI ranks (M1, the must-have part of M3);
- capacity enforced and jobs coalesced by the runtime, not the application (M2, M7);
- correctness shown by a distributed run that equals a sequential reference bit for bit under any completion order (M5, M6);
- an evaluation on the mock that is honest about its limits: overhead (E1), parity with hand-written MPI on the same schedule and the dataflow gain under latency variance (E3), and coalescing under contention (E5), with the null results.

Without the device type it is a library on top of a runtime. Without coalescing it has no runtime-side result larger than tens of percent. Either cut makes it a workshop paper or a poster. One hardware run would strengthen it considerably, and its absence has to be stated in the paper.

**What I need from you before M1**

- Confirmation of the deadline, and the scope: the reduced plan with the December 7 checkpoint, or the full plan for a later target.
- The QPU access you have, if any, and whether compute nodes can reach the provider.
- The destination: an upstream pull request or a research branch. It decides how freely `scheduling.c` and the DTD interface may be edited, and whether this document stays out of version control.

---

# 9. Implementation log

One entry per milestone: what was built, where it departs from the plan above, and what was measured. Where an entry contradicts an earlier section, the entry is right and the earlier section carries a pointer here.

## 9.1 M1: minimal asynchronous QPU task integration (2026-11-22)

Built and run on methane. Unless said otherwise, "the build" is Debug with `PARSEC_DEBUG_PARANOID`, MPI on, CUDA, HIP and Level Zero off, and `PARSEC_DEVICE_WITH_QPU=ON`. Nothing is committed.

### What exists now

| File | Content |
|---|---|
| `parsec/mca/device/qpu/device_qpu.h` | Public: the circuit header, the result layout, `parsec_qpu_submit`, `PARSEC_QPU_SUBMIT`, `parsec_qpu_submit_dtd` |
| `parsec/mca/device/qpu/device_qpu_backend.h` | The backend interface of section 2.5 |
| `parsec/mca/device/qpu/device_qpu_internal.h` | Parameters, the service's counters, what tests need from the mock |
| `parsec/mca/device/qpu/device_qpu_component.c` | The component: parameters, enabling |
| `parsec/mca/device/qpu/device_qpu_progress.c` | The service: submission, admission under `device_qpu_max_outstanding`, the progress thread, the hand-off |
| `parsec/mca/device/qpu/device_qpu_backend_mock.c` | The mock: slots, a queue, a duration per job, a stand-in result |
| `parsec/mca/device/qpu/ValidateModule.CMake` | Builds the component only with `PARSEC_DEVICE_WITH_QPU` |
| `tests/runtime/qpu/` | `qpu_chain.jdf`, `qpu_termination.jdf`, `qpu_overlap.jdf`, `dtd_qpu_chain.c`, `qpu_test_common.{c,h}`, `check_no_hardware.cmake`, `CMakeLists.txt`, `Testings.cmake` |

Edits to existing files, 21 lines in all: the option in `CMakeLists.txt`, `PARSEC_HAVE_QPU` in `parsec_options.h.in`, the two QPU defaults in the test environment of `tests/CMakeLists.txt`, and one line each in `tests/runtime/CMakeLists.txt` and `tests/runtime/Testings.cmake`. No file of the runtime proper was touched.

### The decision M1 had to make: hand-off or direct completion

**Hand-off, as recommended in section 1.5.** The progress thread sets `task->status` to `PARSEC_TASK_STATUS_COMPLETE` and calls `__parsec_schedule` on stream 0 of the virtual process the body ran in; a worker then completes the task. It works for PTG and DTD tasks, with one worker and with 72, with two virtual processes, on 1 to 8 ranks, with the local and the four-counter detector, and under ten of the eleven schedulers (the eleventh fails without QPU too, see below). Direct completion was not needed and not built.

Evidence, beyond the test results below:

- `overlap`: one job of 400 ms and a chain of 20 classical tasks of 5 ms. The chain ran from 0.5 ms to 100.6 ms after the start, the job ended at 400.6 ms, the run took 401.1 ms. With a single worker thread: the same, 413.0 ms. The progress thread used 0.06 ms of CPU in those 400 ms.
- `stress` at zero delay: 100,000 quantum tasks in 1.1 s of wall time (Debug build), the pending heap peaking at about 19,000, never more than 16 jobs at the backend.

### Verification

All runs on methane, 2026-11-22, with the code as it stands at the end of M1.

| Check | Result |
|---|---|
| Baseline: the unmodified base commit, full suite | 167 of 167 pass |
| The modified tree with the option off, full suite | 167 of 167 pass, the same test names |
| The modified tree with the option on, full suite | 194 of 194 pass: the 167 and 27 QPU tests |
| Every QPU test 100 times in a row | 2,700 of 2,700 pass |
| QPU tests in a profiling build (`PARSEC_PROF_TRACE`) | 27 of 27 pass |
| QPU tests in a build without MPI, under the address sanitizer | 16 of 16 pass (the 11 others need MPI) |
| MPI by hand, four workers per rank: `chain` on 2, 4 and 8 ranks, `chain:dtd` on 2 and 4, four-counter `termination` on 2 and 4 | 25, 25 and 10 runs each, all pass |
| `stress` under each scheduler, five runs | 10 of 11 schedulers pass; `llp` aborts, as it does without QPU (see below) |

The 27 QPU tests, by what they cover: `no_hardware_in_tests`; `chain`, `chain:dtd`, `chain:mp:2`, `chain:mp:4`, `chain:dtd:mp` (submission, completion, dependency release, empty jobs, a writer held behind an outstanding reader, insertion while tasks are outstanding); `order:reverse`, `order:random`; `capacity`; `overlap`, `overlap:1worker`; `termination`, `termination:two`, `termination:loop`, `termination:dynamic`, and `termination` and `termination:dynamic` on 2 and 4 ranks; `stress`, `stress:jitter`, `stress:1worker`, `stress:vp`, `stress:mp:2`, `stress:mp:4`; `restart`, `restart:mp`.

Not done in M1: a CUDA build (M1 changes no runtime file, and section 4.7 puts the GPU suite at M3), a build without MPI outside the sanitizer runs, and any multi-node run.

### Departures from the plan

1. **`parsec_qpu_submit` takes data copies, not flow numbers** (section 2.2). A JDF body has no name for a flow's number, but the generated hook has the copy of flow `X` in a local variable `_f_X`. So the body writes `return PARSEC_QPU_SUBMIT(C, TH, RES, shots);`, and DTD bodies call `parsec_qpu_submit_dtd` with the positions of the three data arguments.
2. **The service does not live between the component's open and close.** A device component that offers no module is closed during `parsec_init`, right after it was queried (`device.c:453-469`). So the progress thread starts at the first submission and stops from a `parsec_context_at_fini` callback (`parsec.c:1228`), which runs before the workers are joined and before profiling is finalized (`parsec.c:1293-1299` against `:1313` and `:1326`). This is better than the plan for hazard H7, and M3 should keep it even when a module exists.
3. **Repeated `parsec_init` and `parsec_fini` in one process (hazard H8) cannot be tested, because the runtime does not support it at the base commit.** A second `parsec_init` crashes in `parsec_mca_device_attach` (`device.c:1165`), with or without QPU support; a twelve-line program on the unmodified build shows it. The test `restart` covers what is the QPU layer's own: it stops the service between taskpools, 30 times in one process, so thread, backend and counters are created and destroyed each time.
4. **Fewer test files.** Stress and restart are modes of `qpu_chain`; there is no `qpu_stress.jdf` and no `qpu_init_fini.c`. The four-counter variant of `qpu_termination` is generated at configure time by changing one line, and only in MPI builds.
5. **Done earlier than planned:** capacity and ordering tests (planned for M2), empty jobs, and the `device_qpu_allow_hardware` check at backend open. Its test with a backend that claims to be physical stays in M2.
6. **The mock's result is a stand-in until M2.** One bitstring, a digest of the circuit bytes, the parameter bytes and the seed, for all shots. It is read at the end of the job, which is what lets the tests check that inputs stay in place while a job is outstanding. `device_qpu_mock_delay` is `const:<us>` or `uniform:<min>:<max>`; tests that need a completion order install a delay function.
7. **The result header gained two fields**, still 64 bytes: `t_hook` (when the body submitted, so time pending in the runtime can be told from time at the backend) and `batch` (for M7).
8. **`wakeup` must not block**, and the interface now says so. See the measurement of H1 below.

### Hazard H1, measured

Section 4.3 left open whether a job can be completed before the worker that submitted it is done with the task. It can. A measurement mode of `qpu_chain` (`-H`) looks at each quantum task from the `EXEC_END` instrumentation that follows its body:

| Worker threads | Job duration | Tasks looked at | Already handed back to the scheduler | Already another task |
|---|---|---|---|---|
| 1 | 0 | 100,000 | 99,625 | 0 |
| 2 | 0 | 100,000 | 38,008 | 0 |
| 4 | 0 | 100,000 | 13,591 | 0 |
| 8 | 0 | 100,000 | 24 | 0 |
| 72 | 0 | 100,000 | 0 | 0 |
| 2 | 100 to 1000 microseconds | 20,000 | 0 | 0 |
| all, body held 2 ms in the instrumentation | 50 ms | 64 | 0 | 0 |

What it means:

- The window is hit when two things hold together: the job takes no time at all, and the progress thread shares cores with the workers, so that waking it preempts the worker that just submitted. Neither holds for a real backend. With jobs of 100 microseconds it never happened.
- In more than a million tasks looked at, the task was never reused for another task inside the window. The memory is a mempool object and stays valid either way.
- What can go wrong today is limited to instrumentation: an `EXEC_END` callback may look at a task that another worker is completing. No callback is installed by default.
- The thread sanitizer cannot settle this. It reports the post-body lines of `__parsec_execute` in QPU runs, and equally in the existing `tests/profiling/async.jdf`, but in every report I read the instrumentation access came first and the conflicting write was a later reuse of the task's memory, which the sanitizer cannot order because the scheduler's queues and the mempools are lock-free.

**Recommendation.** No workaround in the QPU layer. Close the window where it is, in `__parsec_execute` (`scheduling.c:218-226`), when M3 edits that file anyway: nothing should touch a task after its hook returned ASYNC. That is a change to shared code and needs your decision; the GPU path and `tests/profiling/async.jdf` have the same window.

### Sanitizers

The system compiler has no sanitizer runtimes, and PaRSEC's configure then drops the sanitizer flags with only a warning: my first sanitizer runs were on uninstrumented binaries and showed nothing. The runs below use the Spack GCC 12.4, without MPI.

- **Thread sanitizer.** The runtime is not clean under it: 103 reports for the existing `dsl/ptg/ptgpp/write_check`, 50 for `profiling/async`. QPU runs add reports of the same two kinds with the progress thread on one side: its push into the scheduler's buffer against a worker's pop (`hbbuffer.c:63` against `:243`), and its write of `task->status` against a later reuse of the task's memory (`parsec.c:1820`). No report is between two accesses of the QPU layer's own state, after one fix: the counters were reset by plain writes while a test read them.
- **Address sanitizer.** No memory error in 15,000 quantum tasks, 300 taskpools created and freed one after the other, 100 service restarts, or 400 DTD chains. The leak checker reports one site, 72 bytes per copy of a write-only PTG flow (80 with the QPU device on: the record has a pointer per device; section 9.7) (`arena.c:382`, `arena.c:283`, `data.c:163`); the existing `write_check` reports the same site, and no leak stack has a QPU frame.

### Things found in the runtime, none of them caused by this work

Each was reproduced on the unmodified build or follows from code that M1 does not touch. They are listed because the tests had to go around them.

| What | Where | Effect on M1 |
|---|---|---|
| A second `parsec_init` in a process crashes | `device.c:1165` | Departure 3 |
| `parsec_taskpool_wait` as the first wait of a context crashes on one rank of an MPI build: it enables the communication engine without the reconfiguration that `__parsec_context_wait` does | `scheduling.c:729` against `:824-826`; crash at `remote_dep_comm.c:1339` | `termination:two` starts with a context wait, as `tests/api/taskpool_wait` happens to |
| The `llp` scheduler aborts in a paranoid build when a startup task is rescheduled: `dsl/ptg/startup -i=200 -j=200 -k=20 --mca sched llp` | `sched_llp_module.c:343`, reached from `scheduling.c:596-597` | QPU tests pass under `llp` at sizes below that |
| A taskpool that asks for the four-counter detector crashes at creation in a build without MPI | `termdet.c:85`, `mca_repository.c:210` | The four-counter test is built with MPI only |
| The `rr` virtual-process map is not implemented | `vpmap.c:384` | The two-virtual-process test uses `hwloc` (two on methane) |
| Each copy of a write-only PTG flow leaks 72 bytes | `arena.c:283` | None |
| Sanitizer options are dropped silently when the compiler lacks the runtime | `CMakeLists.txt:354-394` | See above |

One more thing to know when reading MPI results: under CTest's default `mpiexec`, Open MPI binds each rank to one core, so each rank has one worker thread. The MPI tests were therefore also run by hand with four workers per rank.

## 9.2 M2: mock backend and correctness tests (2026-11-22)

Built and run on methane, on top of M1. "The build" is the same as in section 9.1: Debug with `PARSEC_DEBUG_PARANOID`, MPI on, CUDA, HIP and Level Zero off, `PARSEC_DEVICE_WITH_QPU=ON`. Nothing is committed. No file outside `parsec/mca/device/qpu/` and `tests/runtime/qpu/` changed in M2.

### What exists now

| File | Content | In M2 |
|---|---|---|
| `parsec/mca/device/qpu/device_qpu.h` | Public. The circuit header and the 16-byte operation, the result layout, helpers to build a circuit in a buffer, `parsec_qpu_circuit_check`, `parsec_qpu_submit` with the key of the evaluation | Extended |
| `parsec/mca/device/qpu/device_qpu_circuit.c` | What can be said about a circuit without running it | New |
| `parsec/mca/device/qpu/device_qpu_sampler.c` | Exact state-vector evaluation and sampling | New |
| `parsec/mca/device/qpu/device_qpu_backend.h` | The backend interface | Extended |
| `parsec/mca/device/qpu/device_qpu_backend_mock.c` | The mock: slots, queue, delay distributions, helper threads that sample, fault injection, checks on its caller | Rewritten |
| `parsec/mca/device/qpu/device_qpu_progress.c` | The service: retries, time-outs, an unavailable backend, the minimum interval, the hardware guard, the list of backends | Extended |
| `parsec/mca/device/qpu/device_qpu_component.c`, `device_qpu_internal.h` | Nine more parameters; the sampler's and the mock's interfaces for tests | Extended |
| `tests/runtime/qpu/qpu_sampler_test.c`, `qpu_mock_test.c` | The sampler and the mock without the runtime | New |
| `tests/runtime/qpu/qpu_failure.jdf`, `qpu_guard.jdf`, `qpu_misuse.jdf` | Jobs that go wrong; the hardware guard; what a body may not submit | New |
| `tests/runtime/qpu/qpu_chain.jdf`, `qpu_termination.jdf`, `qpu_overlap.jdf`, `dtd_qpu_chain.c`, `qpu_test_common.{c,h}` | The M1 tests, now on real circuits, with their counts checked against the sampler | Changed |

The component is about 3,700 lines and its tests about 4,700.

### The circuit format (section 2.4, as built)

A circuit is a 32-byte header and `nb_ops` operations of 16 bytes: opcode (1 byte), one reserved byte, two qubit indices (2 bytes each), a parameter slot (2 bytes, signed, -1 for none), and a double. There are thirteen operations: H, X, Y, Z, RX, RY, RZ, CX, CZ, RXX, RYY, RZZ and MEASURE.

- **Conventions.** Qubit $i$ is bit $i$ of a measured bitstring and every qubit starts in $\lvert 0\rangle$, as in section 3.1. A rotation about a Pauli operator $P$ by $t$ is $e^{-itP/2}$. CX has its control in the first qubit index.
- **Angles.** A rotation takes its angle from the operation, or from a parameter slot multiplied by the operation's coefficient. The QAOA layer of section 3.3 is then one circuit for every evaluation: `RZZ` with slot $\gamma_\ell$ and coefficient $-w_{ij}$, `RZ` with slot $\gamma_\ell$ and coefficient $-2h_i$, `RX` with slot $\beta_\ell$ and coefficient 2.
- **Measurement.** `MEASURE` names one qubit and has to be the last operation on it. A qubit that is not measured reads 0. A circuit that measures nothing is refused.
- **Checking.** `parsec_qpu_circuit_check` refuses an unknown version or operation, a qubit or a parameter slot that does not exist, a two-qubit operation on one qubit, an angle that is not finite, anything after a measurement, and operations that do not fit in the buffer. `parsec_qpu_submit` calls it and, in addition, refuses a circuit with more parameter slots than its parameter flow holds. A refusal is a programming error and is returned as one; test `misuse` runs seven such cases and one well-formed job as a control.
- **Result.** Entries are in increasing order of their bitstring, with no bitstring twice. When the buffer has room for fewer entries than the job produced, the most frequent ones are kept, `shots_returned` counts only those, and the result is flagged `PARSEC_QPU_RESULT_TRUNCATED`.

### The key of an evaluation (new)

`parsec_qpu_submit` gained a last argument, a 64-bit key that names the evaluation to the application. `PARSEC_QPU_KEY_AUTO` asks for a key derived from the identity of the task; a JDF body uses `PARSEC_QPU_SUBMIT` for that and `PARSEC_QPU_SUBMIT_KEYED` to give its own.

The reason is section 3.7. The mock draws the samples of a job from (seed, key) and from nothing else, so:

- a result does not depend on the order in which jobs are submitted or end, nor on how many jobs the task needed: a job that is retried gives the counts it would have given the first time;
- a program that does not use the runtime gets the same counts from `parsec_qpu_sampler_sample(circuit, params, shots, seed, key, ...)`. This is what the sequential reference of M5 will call, with the same keys as the PaRSEC application (for example round, block, iteration and sign). Every M1 test but one now checks its counts this way. `overlap` leaves the key to the runtime, so that path stays tested, and checks only that its counts are well-formed.

The delays and the fate of a job under fault injection are drawn from (seed, key, attempt). The tag of a job stays what section 2.5 says, an identifier for the provider, and nothing is drawn from it.

### The sampler

`device_qpu_sampler.c` computes the $2^n$ amplitudes of the state a circuit prepares and draws outcomes from them. It was written for this project and uses no other code. It takes up to 30 qubits; the mock takes up to `device_qpu_mock_max_qubits` (default 24), and its memory is $2^{n+4}$ bytes per helper thread.

Accuracy, from `runtime/qpu/sampler`:

| Check | Result |
|---|---|
| Each of the twelve gates, on each qubit or ordered pair of four qubits, against a second implementation that shares no code with the sampler (dense matrices built from the Pauli matrices, applied by the definition) | Largest difference in an amplitude $1.1\times10^{-16}$ |
| 300 random circuits of 1 to 6 qubits and up to 40 operations, some with parameter slots, against the same reference | Largest difference $1.4\times10^{-15}$; norm off by at most $1.3\times10^{-15}$ |
| States known in closed form: $H\lvert0\rangle$, the three rotations, $Y\lvert0\rangle$, Bell, GHZ on five qubits, X on qubit 3 of 5 (bit order), CX in both directions, RZZ on $\lvert{+}{+}\rangle$ | Equal to $10^{-12}$ |
| The one-edge QAOA circuit of section 3.3 against $\frac{w}{2}\,(1+\sin 4\beta\,\sin\gamma w)$, 20 random $(w,\gamma,\beta)$ | Equal to $10^{-12}$ |
| 200,000 shots of a three-qubit circuit against its exact probabilities | Every frequency within 1.9 standard deviations (the test allows 6) |
| Same (seed, key): same counts, from a sampler that did something else in between and from a new one. Another key or another seed: other counts | Holds |

The acceptance criterion asked for $10^{-12}$. A sign or bit-order error would show in the first two rows; three such errors put in on purpose are all caught (see "Does a test notice" below).

Time to sample one QAOA circuit on a random graph with $1.75\,n$ edges, 1024 shots, one thread, Xeon Gold 6140 at 2.3 GHz, three runs each, which agree to within 8%:

| Qubits | Layers | Operations | Release (`-O3`) | Debug (`-g`) |
|---|---|---|---|---|
| 14 | 1 | 80 | 1.6 ms | 6.7 ms |
| 14 | 2 | 132 | 2.7 ms | 10.8 ms |
| 16 | 1 | 92 | 6.7 ms | 28 ms |
| 16 | 2 | 152 | 11.7 ms | 49 ms |
| 20 | 1 | 115 | 151 ms | 0.56 s |
| 20 | 2 | 190 | 266 ms | 0.98 s |
| 24 | 2 | 228 | 6.6 s | 19.4 s |

That is 1.3 ns per amplitude and operation in the Release build. Section 3.4 estimated "under 1 ms" for 14 qubits and "about 0.1 s" for 20; both were low by a factor of two to three, and section 3.4 now carries the measured values. What it means for the experiments is under "For later milestones" below.

### The mock (section 2.5, as built)

- **Time.** A fixed number of slots and a queue in front of them. A job that gets a slot waits its queue delay there (`device_qpu_mock_queue_delay`, default none), then runs for its duration (`device_qpu_mock_delay`). Both take `const:<us>`, `uniform:<min>:<max>`, `exp:<mean>`, `lognormal:<median>:<sigma>` or `replay:<file>`, a text file of recorded delays in microseconds from which each job draws one. A queued job starts at the instant a slot was freed, not when somebody noticed, so the schedule does not depend on how late the progress thread wakes up.
- **Answers.** Helper threads of the mock (`device_qpu_mock_threads`, default 2) run the sampler on the jobs it holds, from the moment a job arrives. A job ends at the later of its modelled end and the end of its sampling. The mock counts the jobs for which that was the sampling (`late`): for those, and only those, the reported time is not the drawn one.
- **Faults.** `device_qpu_mock_faults` is `none` or a list of `<fault>:<rule>=<value>`. Faults: `fail` (the job runs and ends in failure), `hang` (it never ends), `unavailable` (the submission is not taken), `lost` (the job is taken and the answer says "ask again"), `reject` (refused for good), `noresult` (it ends well and has nothing to give), `duplicate` (its end is reported twice), `late` (it never ends, and its end is reported after it was cancelled). Rules: `p` (probability), `first` (attempts), `every` (submissions), `key`, `for` (milliseconds since the backend was opened). A test can install a function instead.
- **What it checks in its caller.** Three counters that have to stay at zero: a submission under the tag of a job the caller already knows or already took (hazard H11), two jobs of one evaluation held at once, and inputs that changed between a job's submission and the reading of its results (hazard H5). M1 detected a changed input by reading the inputs late; the mock now samples early, so it compares a digest instead.
- **Flag.** Every result of the mock carries `PARSEC_QPU_RESULT_MOCK`; a task that ends without a result carries no flag.

### Failure handling (section 2.9, as built)

| Event | What the service does | What the task's result says |
|---|---|---|
| A job fails, ends without results to give, or is refused for good | The task gets another job, with a new attempt number and so a new tag, until it has had 1 + `device_qpu_max_retries` (default 2) | `attempts` |
| A job is still at the backend after `device_qpu_job_timeout` (default 0, no limit) | Cancel it at the backend, then as above | `attempts` |
| The task has had all its jobs | Complete it normally | Status failed or timed out, no entry, no flag |
| The backend answers "ask again" to a submission | Ask again with the same tag after `device_qpu_retry_backoff` (default 100 ms), doubling up to 30 s. Nothing else is submitted meanwhile, since the provider may have that job | Nothing |
| That has lasted `device_qpu_unavailable_timeout` (default 60 s; 0 for no limit) | Give up on the submission at the backend, which cancels the job if the provider has it; complete that task and every pending one | Status unavailable, `attempts` as it was |
| The backend reports an end for a job that is not in flight | Count it, ignore it | Nothing |
| The backend cannot run the job at all (too many qubits or shots for it) | Complete the task without submitting | Status rejected (new), `attempts` 0 |
| The backend reports that somebody cancelled the job | Complete the task; no retry | Status cancelled |

- `device_qpu_submit_interval` (default 0) is the shortest time between two submissions, whatever the backend answered.
- `attempts` counts the jobs the backend took or refused for good. A submission that is asked again is not a new attempt.
- A task is completed exactly once in every case. The job record is in one place at a time (the submission stack, the pending heap, the one deferred slot, or the in-flight array), and an end reported for anything else finds nothing.

### The backend interface (section 2.5, as built)

Four changes to the interface of M1:

1. **A request carries the key and the attempt number.**
2. **`submit` has three answers.** Success; `PARSEC_ERR_OUT_OF_RESOURCE` when it cannot be said that the provider has the job (unreachable, refusing for now, or no answer), after which the caller asks again with the same tag; any other error when the job is refused for good. The call is idempotent on the tag.
3. **`abandon(tag)` is new.** It gives up on a submission that was answered "ask again": if the provider has the job, it is cancelled. Without it there was no way to settle such a job, and the service could complete a task while the backend still read its buffers. An independent review of the code found this; see below.
4. **`physical` is a field of the backend's description**, known before the backend is opened.

Backends are found by name in a small table; `parsec_qpu_backend_register` adds one before `parsec_init`. The mock is always there.

### The hardware guard (section 4.2, layer 2, as built)

Two checks, both on `device_qpu_allow_hardware`:

- A backend whose description says it can drive a quantum device is refused while the context is initialized. It is never opened. Quantum tasks then cannot run in that context, and a body that submits is told why.
- A backend that says so only in its capabilities, once open, is closed at once. It receives no job.

A backend name that does not exist is refused the same way, and no other backend is used in its place. `runtime/qpu/hardware_guard` and its three variants test the two checks, the unknown name, and a control in which the same task runs on the mock. Check 3 of section 4.2 (`no_hardware_in_tests`) is unchanged, and no test sets the parameter.

### Verification

All runs on methane, 2026-11-22, with the code as it stands at the end of M2 (after the review described below).

| Check | Result |
|---|---|
| The modified tree with the option off, full suite | 167 of 167 pass, the same test names as the baseline of section 9.1 |
| The modified tree with the option on, full suite | 215 of 215 pass: the 167 and 48 QPU tests |
| Every QPU test 100 times in a row | 4,700 of 4,700 pass, for the 47 tests there were at that point. The failure test then gained one mode (`down:twice`); its 13 registered runs passed 100 times in a row again, 1,300 of 1,300 |
| QPU tests in a profiling build (`PARSEC_PROF_TRACE`) | 48 of 48 pass |
| QPU tests in a Release build (`-O3 -DNDEBUG`) | 48 of 48 pass |
| QPU tests in a build without MPI, under the address sanitizer | 36 of 36 pass (the 12 others need MPI) |
| The 28 tests that depend on time, 40 times, six at once (each with 72 threads on 72 cores) | 1,120 runs, all pass, after the change to the ordering tests (departure 6) |
| MPI by hand, four workers per rank: `chain` on 2, 4 and 8 ranks; `failure` (mixed) on 2, 4 and 8; `chain:dtd`, four-counter `termination`, and `failure` in the modes `down:long`, `down:twice` and `lost:down` on 2 and 4 | 25, 15, 25, 10 and 10 runs each, all pass |
| `stress` under each scheduler, 10,000 quantum tasks, five runs | 10 of 11 schedulers pass; `llp` passes at 400 tasks and is the scheduler that aborts without QPU too (section 9.1) |
| 100,000 quantum tasks at zero delay; 2,000 tasks of `failure` (mixed) | Three runs each, all pass |
| Compiler warnings in the QPU sources and tests, six builds, GCC 11.5 and 12.4 | None |

The 48 QPU tests are the 27 of section 9.1 and 21 new ones: `sampler`; `mock`; `misuse`; `hardware_guard`, `hardware_guard:late`, `hardware_guard:unknown`, `hardware_guard:control`; `capacity:interval`; `failure` and its variants `:noretry`, `:once`, `:always`, `:timeout`, `:ambiguous`, `:down:short`, `:down:long`, `:down:twice`, `:lost:down`, `:rejected`, `:interval`, and `failure:mp` on two ranks.

What `failure` checks. The mock is told, job by job and attempt by attempt, what to do instead of running the job; ten kinds of task by default (nothing wrong; one failure; failure every time; one job that never ends; no job that ever ends; no results to give; an end reported twice; an end reported after cancellation; a lost reply; a refusal), and every seventh submission deferred. What each task must end with follows from that plan alone. Each successor checks the status, the number of jobs, and, when the task ended well, that its counts are the ones the evaluation has without any fault. The main program then checks sixteen counters of the service and of the mock against the plan, exactly: jobs taken, failed, timed out, retried, cancelled, found by their tag, and so on. Every successor runs exactly once.

**Sanitizers**, on builds without MPI made with GCC 12.4, as in section 9.1.

- **Address sanitizer.** 19 larger runs (3,000 chains at zero delay, 100 service restarts, 300 taskpools, 400 DTD chains, 400 tasks of `failure`, the unavailable and lost-reply modes, both unit tests): no memory error. The leak checker reports 88 records and none has a frame in the QPU component. 82 are the write-only-flow site of section 9.1. The six others come from the DTD test and are the runtime's too: the existing `dsl/dtd/new_tile` reports the same three sites (`parsec_dtd_data_flush.c:261` through the task mempool, `insert_function.c:2121`, `insert_function.c:134`) with no QPU code in the process. The two unit tests leak nothing.
- **Thread sanitizer.** 18 runs, 3,112 reports. None is between two accesses to the QPU layer's own state. In 24, one of the two threads is in QPU code, and they are the two kinds of section 9.1: the progress thread's push into the scheduler's buffer against a worker's pop, and its write of `task->status` against a later reuse of the task's memory. In 4, all from `misuse`, both threads are inside `parsec_warning`, called by `parsec_qpu_submit` as it refuses several bodies at the same time. The accesses are in the runtime's output layer (`output.c:849-939`), behind a spin lock of the runtime that the sanitizer does not see as a lock; any two threads that warn at once produce them. The two unit tests, where helper threads, `cancel`, `abandon`, `close` and a `wakeup` from another thread all meet, produce no report.

One more race in the runtime showed up in these reports, with a QPU frame only in the stack that says where the memory was allocated: `lookup_env` sets a flag in the parameter it looks up (`mca_param.c:1600`), and two workers that initialize two task classes at the same time both do. It writes the same value and is harmless.

### What an independent review found

Before the final runs I had the component read by a reviewer that had not written it, with the instruction to break it. The mock's locking, the sampler's arithmetic and the circuit check held. It found one defect and several smaller things, all fixed before the runs above:

| Finding | Fix | Test that now covers it |
|---|---|---|
| **A task could be completed while the backend still held its job.** A submission answered "ask again" can be a job the provider took. When the unavailable time-out then fired, the task was completed, and a helper of the mock could still be reading its circuit. | `abandon(tag)` in the backend interface; the service calls it before it completes such a task, and when it stops. | `failure:lost:down`; `mock` gives up on a submission while its 18-qubit job is being sampled and frees the inputs at once, which the address sanitizer watches |
| The minimum interval held only after a submission that was taken; a refused or deferred one was followed at once by the next. | The interval is between two submissions, whatever the answer. | `failure:interval` (the mock measures every submission it receives) |
| After one unavailable time-out, a single later "ask again" completed everything at once, however long ago the outage was; and a time-out of 0 meant "at once". | The clock restarts after a flush and when the backend answers for good; 0 means no limit. | `failure:down:twice`: a second taskpool, refused three times after an outage that timed out, has to end well |
| The parameter flow's length was not checked against the circuit's slots. | Checked in `parsec_qpu_submit`. | `misuse`, case 3 |
| Counters: the deferred job was not counted as pending; the high-water mark of pending missed retries; a retry was counted when queued, not when made. | Fixed. | `failure` and its variants check every counter against the plan |
| The time of submission was overwritten each time a submission was asked again, so a job found by its tag could have started before it was "submitted". | Kept from the first asking. | `failure:ambiguous` checks the order of the four times |
| The tag packed the taskpool, class and task key by exclusive or, so a task key of 2^32 or more could collide with another class. | The three are hashed in one after the other. | None; the collision needs a key that large |
| Mock: asking again for a job the caller knows was an error, against the interface. A failed job not yet reported could not be cancelled. | Both follow the interface now. | `mock` |

One remark is not fixed, and is a limit to know: `cancel` and `abandon` wait for a helper that is sampling the job to notice, which takes up to one gate plus part of the sampling. That is under a millisecond at 18 qubits (measured) and tens of milliseconds at 24, during which the progress thread does nothing else.

### Does a test notice

Passing tests say little if they would also pass on wrong code. Fifteen deliberate bugs were put in one at a time, the QPU tests run, and the bug taken out again.

| Bug put in | Tests that fail |
|---|---|
| RYY with the sign of its $\lvert00\rangle$, $\lvert11\rangle$ pair wrong | `sampler` |
| CX with control and target exchanged | `sampler` |
| RZ with the sign of its angle wrong | `sampler` |
| The mock draws samples from the key and the attempt, not the key alone | 14, among them `chain`, `failure:once`, `mock` |
| A submission that is asked again gets a new tag | `failure`, `failure:ambiguous`, `failure:down:short`, `failure:down:long`, `failure:lost:down` and two more |
| A job that timed out is not cancelled at the backend | `failure`, `failure:noretry`, `failure:timeout`, `failure:interval` |
| One retry fewer than allowed | `failure`, `failure:always`, `failure:interval` |
| The submission that is given up on is not settled at the backend (the defect the review found) | `failure:lost:down` |
| The time of submission is taken again each time a submission is asked again | `failure`, `failure:noretry`, `failure:ambiguous`, `failure:interval` |
| The clock of the unavailable time-out is not restarted after it fired | `failure:down:twice` |
| The minimum interval is ignored | `capacity:interval`, `failure:interval` |
| The minimum interval holds only after a submission that was taken | `failure:interval` |
| The parameter flow's length is not checked | `misuse` |
| The hardware guard always lets a backend through | `hardware_guard`, `hardware_guard:late` |
| The mock reports the end of a job its caller was never told of | `mock`, `failure`, `failure:ambiguous`, `failure:lost:down` and two more |

All fifteen are caught. Each run was the single-rank QPU tests that do not repeat an M1 test (23 of them, 24 for the last bug, which was tried after `failure:down:twice` was added); the sources were compared with their saved copies afterwards and the build redone.

### Departures from the plan

1. **A key per evaluation** was added to `parsec_qpu_submit`, to the DTD variant and to the request. Section 3.7 needs it and the plan had not said how.
2. **The backend interface changed** in the four ways listed above. `abandon` and the three-way answer of `submit` are what section 2.5 called "looks the job up by tag before submitting again", made explicit.
3. **Status `PARSEC_QPU_STATUS_REJECTED`** is new, for a job no attempt could run.
4. **Names.** The parameter is `device_qpu_mock_faults`, not `device_qpu_mock_fail`, since it injects more than failures. New and not in section 7.2: `device_qpu_retry_backoff`, `device_qpu_mock_queue_delay`, `device_qpu_mock_threads`, `device_qpu_mock_max_qubits`. Not there yet: `device_qpu_owner_rank` (M3), `device_qpu_max_batch` (M7), `device_qpu_poll_min` and `device_qpu_adapter_address` (M4).
5. **Tests.** The guard test is a JDF (`qpu_guard.jdf`), not `qpu_guard.c`, because it needs a task that submits. `misuse` is new. The failure test is registered thirteen times, twelve modes or settings on one rank and one on two, where the plan had one name.
6. **The ordering tests of M1 were changed.** They required the successors of jobs ending 8 ms apart to run in that order. With several tests running at once, each with 72 threads on the 72 cores, two successors swapped twice in about a hundred runs, because two workers ran them. Run one at a time, as in the 200 runs of M1, it never happened. The tests now read the order of completion from the results, where the progress thread wrote it, and require every job to have run for exactly its duration. That is what the service guarantees; the order in which workers run released tasks is not.
7. **The sampler is slower than estimated**, by a factor of two to three (above).

### For later milestones

- **M4, a job the provider took and the runtime gave up on.** `abandon` settles it in the mock. A real adapter has to cancel the job at the provider when it can reach it again, or it runs and is paid for with nobody to read it. The interface allows that; nothing implements it yet.
- **M4, tags across runs.** A tag is unique within a process. Two runs of the same program produce the same tags, which a provider that keeps jobs by tag would take for the same jobs. The adapter has to add something of the run.
- **M5, keys.** The reference and the application have to agree on the key of each evaluation and on the seed. The sampler is the only thing the reference needs from this component.
- **M8, the mock's own computation.** Sampling takes real time on real cores: 7 to 12 ms per job at 16 qubits. With 16 slots and a modelled latency of 100 ms, two helper threads are not enough, jobs end late, and the modelled latencies are not the ones measured. The experiments need as many helper threads as slots, cores for them that the workers do not use, and the `late` counter at zero in every reported run. This goes into the threats of section 6.7.
- **M3, hazard H1.** Unchanged by M2. The decision of section 9.1 is still open.
- **Not done in M2:** a CUDA build (M3), multi-node runs, a build without MPI outside the sanitizer runs, and coalescing (M7: the mock takes one circuit per job).

## 9.3 M5: sequential QAOA reference (2026-11-22)

Built and run on methane, on top of M2, in the build of section 9.1. Nothing is committed. Outside the new directory, M5 adds one line each to `tests/apps/CMakeLists.txt` and `tests/apps/Testings.cmake`. Nothing in the runtime or in the QPU component changed.

### What exists now

All in `tests/apps/qaoa/`, about 2,900 lines:

| File | Content |
|---|---|
| `qaoa_graph.h`, `qaoa_graph.c` | Weighted graphs, two generators, the partition into blocks, the quotient graph and its greedy colouring, and the eleven fixed instances |
| `qaoa_core.h`, `qaoa_core.c` | What each step of the algorithm computes, as pure functions on flat buffers: block state, boundary, local problem, circuit, evaluation of a result, optimizer, accept rule, keys |
| `qaoa_exact.c` | The largest cut of a graph, and the best a block can do alone, by Gray-code enumeration; the exact expectation of a circuit |
| `qaoa_seq.c` | The sequential reference, in the order (round, colour, block, step, evaluation), and two ways to run a job: the sampler called directly, and the mock backend one job at a time |
| `qaoa_reference.c` | The program: every instance with its invariants, or one chosen run with timers |
| `qaoa_math_test.c` | The arithmetic, without the runtime |

No program of this directory starts the runtime yet. They use the circuit format, the sampler and the mock backend of the QPU component and nothing else.

### The algorithm as built

Section 3 left some choices to the implementation. They are settled as follows, and M6 has to make the same ones, which it does by calling the same functions.

- **Initial assignment.** A hash of the seed and the block.
- **Local problem.** The constant $c_S$, one field $h_i$ per vertex and the edges inside the block, those of weight zero left out. Its value on an assignment is $c_S+\sum_{(i,j)} w_{ij}(x_i\oplus x_j)-\sum_i h_i z_i$. Weights are multiples of 1/8 and fields multiples of 1/16, so every value is exact in a double and the tests compare with `==`.
- **Boundary.** Folded in one neighbour at a time, in the order of the neighbour list, by a function that takes the block, the slot and the neighbour's state. This is the `GATHER` chain of section 3.6.
- **Circuit.** As section 3.3: per layer `RZZ` with slot $\gamma_\ell$ and coefficient $-w_{ij}$, `RZ` with slot $\gamma_\ell$ and coefficient $-2h_i$ where $h_i\neq 0$, `RX` with slot $\beta_\ell$ and coefficient 2. Built once per update; the $2T$ jobs differ only in the $2p$ parameter values.
- **Evaluation of a result.** The mean of $C_S$ over the counts, and the best bitstring measured; of two with the same value, the smaller.
- **Optimizer.** SPSA with $a_t=a/(t+1+A)^{0.602}$ and $c_t=c/(t+1)^{0.101}$, $a=0.25$, $c=0.15$, $A=1$. The objective is divided by the scale of the local problem (the sum of $|w_{ij}|$ and $2|h_i|$) before it is differenced, so the gains are in radians whatever the weights are. The perturbation of step $t$ is a hash of (seed, round, block, $t$).
- **First angles.** $\gamma=0.5$ divided by the largest coefficient of the local problem, $\beta=0.35$, for a block that has none of its own. Afterwards the block starts from the angles that gave the best estimate in its last update. They travel in the block's state.
- **Accept rule.** The best bitstring measured in the $2T$ jobs replaces the current one if its $C_S$ is strictly larger.
- **A failed job** gives no candidate, and the step moves only if both of its jobs gave counts.
- **Skip.** With a patience $q>0$, a block that has not improved in $q$ updates in a row and whose neighbours all have the version they had at its last update is not evaluated: its circuit is an empty job. The default is $q=0$, no skipping.
- **Keys.** The job of round $r$, block $b$, step $t$, evaluation $e$ has the key $r\cdot2^{44}+b\cdot2^{20}+2t+e$. The reference and the distributed application name every evaluation this way, which is what makes their samples the same (section 9.2).

The flat buffers, which M6 passes through flows as they are: block state, 232 bytes; boundary, 144 bytes plus 8 per vertex; local problem, 184 bytes plus 8 per vertex and 16 per edge; one evaluation, 32 bytes; optimizer state, 264 bytes; circuit, 32 bytes plus 16 per operation. For 16 blocks of 16 vertices with 28 edges inside, that is 760 bytes for the local problem and 1.5 kB for a one-layer circuit.

### Verification

All runs on methane, 2026-11-22, with the code as it stands at the end of M5.

| Check | Result |
|---|---|
| The modified tree with the option off, full suite | 167 of 167 pass |
| The modified tree with the option on, full suite | 219 of 219 pass: the 167, the 48 QPU tests and 4 new ones |
| The four new tests 100 times in a row | 400 of 400 pass |
| QPU and QAOA tests in a profiling build and in a Release build | 52 of 52 each |
| The same in a build without MPI, under the address sanitizer | 40 of 40 (the 12 others need MPI) |
| The QAOA programs under the address sanitizer with the leak checker, five runs, the largest with 12 blocks of 8 and with 14-qubit blocks through the mock | No error, no leak |
| The same five under the thread sanitizer (the mock's helper thread runs in two of them) | No report |
| Compiler warnings in the new sources, six builds | None |

The four tests: `apps/qaoa/math`; `apps/qaoa/reference`; `apps/qaoa/reference:generated` (a generated graph, two layers, patience 1, timers); `apps/qaoa/reference:blocking` (`ring64` with every job a blocking call of 1 ms through the mock).

**`apps/qaoa/math`** checks, among other things:

| What | How | Result |
|---|---|---|
| The cut of the whole graph is a block's local objective plus the cut of the edges that do not touch it | Every assignment of every instance up to 16 vertices, 3,000 drawn ones for `ring64`; every block | 573,084 local problems, all equal |
| Fields and constant | Computed a second time in the test, from the definition of the spin form | Equal |
| Bit $i$ of a block is its vertex $i$ | `path6`, whose 64 assignments have 32 different cuts, against a formula written out vertex by vertex; and the marginal of each qubit against the closed form for its own field | Equal |
| The circuit prepares the state of the exact layers | 72 blocks and depths from `tri` to `ring64`, one to three layers, against $e^{-i\gamma C_S(x)}$ on each amplitude followed by $e^{-i\beta X}$ on each qubit, written out in the test | Overlap off one by at most $2\times10^{-15}$ |
| Closed forms | One edge: $\frac{w}{2}(1+\sin4\beta\sin\gamma w)$. One vertex in a field $h$: $c+h\sin(2\gamma h)\sin2\beta$ | Equal to $10^{-12}$ |
| An edge of weight zero | No term, no gate, no neighbour | Holds |
| The estimate of a job | 70 jobs of 4,096 shots against the exact expectation | Within 2.5 standard errors (6 allowed) |
| Accept rule, optimizer, skip rule, enumerations | Against their formulas, and the enumerations against plain loops | Hold |

**`apps/qaoa/reference`** runs every instance under four settings (one and two layers; 10 to 256 shots; with and without skipping) and requires each time:

- after every update of a block, the cut of the whole graph is the one before plus the improvement the block reports, and did not go down;
- the final cut, recomputed from the assignments, is the initial one plus the sum of the improvements, with `==`;
- it does not exceed the largest cut, found by enumeration, for the ten instances of at most 24 vertices;
- no block ends above the best it could do alone;
- the same settings give the same bits a second time, and the same bits when every job goes through the mock backend;
- the number of jobs is $2T$ per update that was not skipped;
- with one job in five failing (313 of 1,638 over all runs), all the exact checks still hold.

### What the reference finds

Not a claim of the paper, but worth knowing before the experiments. One layer, 4 steps, 3 rounds, 256 shots:

| Instance | Initial cut | Final cut | Largest cut | Ratio |
|---|---|---|---|---|
| `tri` | 4 | 5 | 5 | 1.000 |
| `c4`, `c4-cross` | 1.75 | 3.5 | 3.5 | 1.000 |
| `path6` | 2.125 | 3.875 | 3.875 | 1.000 |
| `rand12` | 16 | 17.75 | 20.5 | 0.866 |
| `rand12neg` | 4.75 | 12.375 | 12.375 | 1.000 |
| `rand16` | 20.375 | 30 | 31.25 | 0.960 |
| `disc16` | 11 | 18.875 | 18.875 | 1.000 |
| `ring64` | 63.125 | 103.625 | not enumerated | |

- **With blocks this small the quantum part hardly matters.** A block of 4 to 8 qubits has 16 to 256 assignments, and $2T$ jobs of 256 shots see nearly all of them. In every run above each block ended at the best it could do alone. What is left is block-coordinate ascent, which stops at assignments no single block can improve: `rand12` at 0.87 of the optimum, and `c4-cross` at 0.64 under another seed.
- **At the size of the experiments it is the same.** For 16 blocks of 16 vertices, 8 steps and 1,024 shots, 20 of 32 updates were accepted in two rounds and the cut went from 284.6 to 470.4.
- This does not touch the runtime claims, and the plan already keeps solution quality out of them (risks of M5 in section 7.3; every system of section 6.2 computes the same result). It does mean the paper cannot present the cut values as evidence for the quantum part.

### Measured times

Release build, one thread, three runs each; blocks of 14 vertices with 25 edges inside and of 20 with 35, 1,024 shots, 8 steps:

| Step | 14 vertices | 20 vertices |
|---|---|---|
| Gather the boundary | 0.7 to 0.8 µs | 1.2 to 1.7 µs |
| Build the local problem and its circuit | 1.3 to 1.7 µs | 2 to 3.5 µs |
| Evaluate the counts of one job | 130 to 160 µs | 210 to 270 µs |
| One optimizer step | 0.2 to 0.35 µs | 0.7 to 1.2 µs |
| Update the block | 0.05 µs | 0.15 µs |
| The best a block can do alone, by enumeration | 0.5 to 0.65 ms | about 43 ms |
| One job on the sampler, one layer | 1.9 to 2.1 ms | 155 ms |
| One job on the sampler, two layers | 3.1 to 3.6 ms | 230 to 350 ms |

Section 3.4 now carries these. The classical work per job is 0.15 to 0.3 ms, nearly all of it the evaluation of the counts, against the 50 to 100 µs estimated. The conclusion of section 3.4 stands: next to a job of seconds it is nothing, and overlap of classical and quantum work is not what this application can show.

The sequential reference on the controlled instance (16 blocks of 16, one layer, 8 steps, 2 rounds: 512 jobs) takes 5.3 s with the sampler called directly, 7 to 10 ms per job. With a modelled latency per job it takes the sum of the latencies, as baseline A should: 64 jobs of 1 ms took 69 ms.

### Does a test notice

27 deliberate bugs, one at a time, as in section 9.2:

| Kind | Bugs | Caught by |
|---|---|---|
| Signs and factors of the local problem | The sign of the boundary spin; the constant doubled; the sign of the field in the value; the current value taken at assignment 0; each edge inside a block counted twice | `math`, and `reference` for four of the five |
| The circuit | The sign of `RZZ`; `RZ` with half its angle, and with the wrong sign; `RX` with half its angle; `RZ` on the slot of $\beta$ | `math` |
| Bit order | The cut read with two bits exchanged | `math`, `reference` |
| Evaluation | The larger bitstring kept on a tie; the mean divided by the shots asked for and not those returned | `math`; `reference` for the second |
| Optimizer | Descent in place of ascent; no candidate from the second point; the best angles on the wrong side; a move on one evaluation alone | `math` |
| Accept and skip | A candidate that is only as good is accepted; a block is skipped although a neighbour changed; a skipped block is sampled all the same | `math` for the first two, `reference` for the last |
| Graph | An edge of weight zero makes two blocks neighbours; every block gets colour 0; the reverse slot is wrong | `math` |
| Bookkeeping | The improvements are not summed | `reference` |
| Enumerations | The wrong sign in each of the two | `math`, `reference` |
| An edge of weight zero kept in the local problem | | Not caught at first: the test counted gates against the same wrong number. A check that counts the edges from the graph was added, and catches it |

All 27 are caught, the last one after that addition.

### Departures from the plan

1. **One more file,** `qaoa_seq.c`. The sweep has to be callable from M6's test program, which compares the distributed run with it in one process, so it cannot live in the file that has `main`.
2. **One more instance,** `path6`, for the bit-order test; and the two partitions of `edge2` and `c4` have names of their own (`edge2-split`, `c4-cross`).
3. **Two more tests** than the two planned: the reference run as it is for a measurement, and with blocking calls through the mock.
4. **The times of section 3.4** were three to five times too low for the evaluation of counts and for the enumeration; the others were right.

### For M6

- The task bodies call the functions of `qaoa_core.c` and nothing else of their own: `GATHER` is `qaoa_boundary_gather`; `BUILD` is `qaoa_local_build`, `qaoa_circuit_build` and `qaoa_opt_init`; `STEP(t)` is `qaoa_opt_absorb` for $t>0$ followed by `qaoa_opt_propose` for $t<T$; `EVAL` is `qaoa_eval`; `UPDATE` is `qaoa_state_update`.
- `QEXEC` submits with `PARSEC_QPU_SUBMIT_KEYED` and `qaoa_key(r, b, t, e)`, and the mock's seed has to be the seed the reference gives the sampler.
- The arenas are sized from the partition: the largest block and the most edges inside one.
- A skipped update still has its `QEXEC` tasks; their circuit is an empty job, which the QPU layer completes without a backend (tested since M1).
- The test of M6 is equality of the block states with `qaoa_sequential_run`, byte for byte.

## 9.4 M3: native QPU device integration (2026-11-23)

Built and run on methane, on top of M5, in the build of section 9.1, and in a CUDA build on leconte (8 V100; three of them, one per rank). Nothing is committed. This is the first milestone that edits the runtime proper: three source files, 41 lines added and 16 removed, of which about half fix two defects that were there before (below).

### What exists now

| File | Content |
|---|---|
| `parsec/mca/device/device.h` | `PARSEC_DEV_QPU` on bit 5; `PARSEC_DEV_HAS_LOAD`, the types whose load the scheduler keeps (the GPU types and the QPU) |
| `parsec/mca/device/device.c` | The name `qpu`; a device with no flop rate keeps its own time estimate; `parsec_mca_device_is_gpu` uses the GPU mask; the paranoid check on copies the application manages expects them on the host for a device that holds no data; and a fix to the list of device components (below) |
| `parsec/scheduling.c` | `__parsec_execute` adds the load for the types of `PARSEC_DEV_HAS_LOAD` and switches the inputs to their host copies for every type that is not a GPU, each input under its own flow (the second fix below); `__parsec_complete_execution` takes the load back for the same types |
| `CMakeLists.txt`, `parsec_options.h.in` | `PARSEC_HAVE_DEV_QPU_SUPPORT`, set with `PARSEC_HAVE_QPU` from the one option |
| `parsec/mca/device/qpu/device_qpu_module.c` (new) | The device module: attach, the owner rank, capabilities, the statistics line, `parsec_qpu_device_info` |
| `parsec/mca/device/qpu/device_qpu_component.c` | Offers the module; `device_qpu_owner_rank` |
| `parsec/mca/device/qpu/device_qpu_progress.c` | The backend is opened when the device is attached; submission is refused on a rank that does not own the QPU; the device counts the tasks it executed; the time a job takes is measured; the runtime is told which flows a QPU task wrote |
| `tests/runtime/qpu/` | New: `qpu_discovery.jdf`, `cpu_gpu_qpu.jdf`. `qpu_guard.jdf` reworked. Every quantum body is `[type=QPU]`; the DTD quantum task class has a chore of type `PARSEC_DEV_QPU`. Every test ends with a check of the device |
| `tests/dsl/ptg/qpu_fallback.jdf` (new) | A task class with a QPU body and a CPU body; built with and without the option |

`parsec/interfaces/dtd/insert_function.c` and `parsec/interfaces/ptg/ptg-compiler/jdf2c.c` did not change.

### How the QPU is a device

- **Selection.** `BODY [type=QPU]` compiles to an incarnation of type `PARSEC_DEV_QPU`, as section 1.6 found, and `parsec_select_best_device` selects the QPU module for it with no change to that function. A context that has no QPU device (the default) drops the QPU incarnations when a taskpool starts, as it drops the CUDA ones on a machine without a GPU, and another incarnation of the class runs. `dsl/ptg/qpu_fallback` runs the same executable both ways.
- **Load.** The scheduler adds `task->load` to the device before the hook and takes it back at completion, as for a GPU. For a QPU a task counts for a nominal second, so the load is the number of tasks the device holds, in seconds. `overlap` reads it from a classical task while the job is outstanding; every test requires it to be zero at the end.
- **Executed tasks.** Counted by the service at the hand-off, since the scheduler counts only tasks whose hook did not return ASYNC. A QPU body that ends by itself (a submission that was refused) is counted by the scheduler. Both land in the device statistics of the runtime, which every test reads through `parsec_devices_save_statistics`.
- **Capacity.** `device_qpu_max_outstanding`, capped by the backend, is a property of the device and is in `parsec_qpu_device_info`. The test is `capacity`, unchanged since M2.
- **Capabilities.** The backend is opened when the device is attached, on the owner rank, so the device knows the number of qubits, the shot limit and whether the backend is physical before any task runs, and prints them with `device_show_capabilities`. A backend that may not be used (the hardware guard, either layer) or does not open gives no device at all.
- **Owner rank.** `device_qpu_owner_rank`, default 0, `-1` for the last rank. Every rank attaches a module; only the owner has a backend. A quantum task that runs on another rank was misplaced: `parsec_qpu_submit` refuses it with a message that names both ranks. An owner rank that does not exist leaves the QPU with no owner, with a warning.
- **Statistics.** The QPU has its row in the table of `device_show_statistics`, and one more line under it with what has no column there: jobs, tasks without a job, failures, retries, the peak of jobs outstanding and of tasks waiting, and the mean time of a job.
- **Data.** A QPU holds none. The scheduler switches the inputs of a QPU task to their host copies and pins them, as for a CPU task. The code a JDF generates around a body says which flows the body wrote only for CPU bodies, so `parsec_qpu_submit` says it for a QPU body: for every flow the task writes, the host copy is a version further and is the one to read (in a CUDA build, `parsec_data_transfer_ownership_to_copy`). Without it an accelerator that holds an older copy of the same data keeps reading it; `cpu_gpu_qpu` is built to show that.
- **A body that is not of the QPU type may still submit.** The job goes to the QPU of the process all the same, and the device counts it; the runtime then does not know the task as a quantum task, keeps no load for it and cannot choose another incarnation. This is option C of section 1.9, kept as the fallback of section 7.1, and one test (`chain:cputype`) keeps it working.

### Two defects of the runtime that M3 ran into

Both are in the base commit and in `master` of the upstream repository (read on 2026-11-23), with no issue or pull request that I could find. Both are fixed in this tree because the QPU device needs them fixed, and both are independent of QPUs and should go upstream by themselves.

**The list of device components.** `parsec_mca_device_init` compacts the list of device components that returned a module. Its counter advanced only when a component had to be moved, so with two components that both returned modules the second overwrote the first (`device.c:470-474` at the base commit). The first one's devices were then never attached, and at shutdown its close asserted on a device that was never added. On leconte, with the loop as it was and both CUDA and the QPU enabled: the CUDA component returned a module, the context had no CUDA device, and `parsec_fini` aborted in `device_cuda_component_close`. In a profiling build the same loop dereferences the slot it has just emptied.

No configuration met this before, since it needs two device components in use in one process; CUDA with HIP or with Level Zero would have. The fix is six lines in the same loop: every component that gave a module takes the next place.

**The inputs of a task that does not run on a GPU.** Before the body of such a task, `__parsec_execute` switches each input that is a GPU copy to the host copy and counts the body as a reader of it (`scheduling.c:161-192` at the base commit). The loop took the class's list of input flows and the data of the task by the same index. The list is compact; the data are indexed by flow; and the PTG compiler numbers the flows a task writes before the ones it only reads (`jdf.c:1546-1557`). So for a task class with a flow it only writes, the two are out of step: the last input is skipped, and a written flow is taken in its place. A quantum task that reads a circuit and parameters and writes a result has exactly that shape, and so has any CPU task like it. The independent review found this by reading; on leconte, a quantum task with a write-only flow whose last input comes from a CUDA task then reads device memory on the host and the run ends in a segmentation fault. The fix is three lines of that loop: the data are taken by the flow's own index.

### Verification

All runs with the code as it stands at the end of M3, on 2026-11-23. On methane unless said otherwise.

| Check | Result |
|---|---|
| The modified tree with the option off, full suite | 168 of 168 pass: the 167 and `dsl/ptg/qpu_fallback` |
| The modified tree with the option on, full suite | 231 of 231 pass: the 167, the 48 QPU tests of M1 and M2, the 4 of M5 and 12 new ones |
| Every QPU and QAOA test 100 times in a row | 6,400 of 6,400 pass (64 tests) |
| The same tests in a profiling build and in a Release build | 64 of 64 each |
| The same in a build without MPI, under the address sanitizer | 48 of 48 (the 16 others need MPI) |
| MPI by hand, four workers per rank: `chain`, `discovery` with the QPU on the first and on the last rank, and `failure` on 2, 4 and 8 ranks; `chain:cputype`, `chain:dtd`, four-counter `termination`, `failure:lost:down` and `restart` on 2 and 4 | 35 loops, 310 runs in all, none failed |
| `stress` under each scheduler, five runs; 100,000 tasks at zero delay; two virtual processes | All 11 schedulers pass (`llp` at 400 tasks, as before); both others pass three times |
| Compiler warnings in the new and changed sources, six builds | None |
| **leconte, CUDA build of the base commit**, full suite, one V100 per rank | 257 of 257 pass (63 GPU tests, 194 others) |
| **leconte, CUDA build of the modified tree with the option on**, full suite | 323 of 323 pass: the 257, the 64 QPU and QAOA tests, and the two that need a CUDA device |
| leconte, seven existing GPU tests with a QPU device in the same context (`runtime/cuda/stress`, `get_best_device`, `nvlink`, `stage`, `ptg_pingpong`, `dtd_pingpong`, `dsl/dtd/dtd_test_cuda_task_insert`) | All pass; the device statistics show `cpu-cores`, `cuda(0)`, `cuda(1)`, `qpu(mock)` |
| leconte, `cpu_gpu_qpu` with two CUDA devices and 64 chains | 128 CUDA bodies, 64 QPU bodies, 0 CPU bodies in place of a CUDA one; every result right |

Not done in M3: a HIP or Level Zero build, a build without MPI outside the sanitizer runs, and any multi-node run.

### The tests

Twelve tests are new on methane, and two more exist in a CUDA build:

| CTest name | What it checks |
|---|---|
| `runtime/qpu/discovery`, `:mp:2`, `:mp:4`, `:mp:last` | The type has a name; one QPU device on every rank; a backend, described as configured, on the owner rank only, before any task; no progress thread before the first submission. Then one task per rank whose only body is of the QPU type: the runtime keeps and selects that incarnation everywhere, the job runs on the owner, and on every other rank the submission is refused and nothing writes the result. The copy of every flow the quantum task wrote is one version further. After `parsec_fini` nothing is open |
| `runtime/qpu/discovery:off` | No device, no backend and no thread in a context that did not ask for a QPU |
| `runtime/qpu/discovery:idle` | A backend that was opened with the device and never used is closed |
| `runtime/qpu/discovery:noowner` | An owner rank that does not exist: a device on every rank, a backend on none, every submission refused |
| `runtime/qpu/cpu_gpu_qpu`, and `:gpu` in a CUDA build | A tile goes from a CUDA task to a quantum task, which uses it as its result, and on to a CUDA task: what the accelerator wrote is on the host when the QPU body runs, and what the QPU layer wrote replaces the accelerator's older copy. A second tile from the CUDA task is the last input of the quantum task, which also has a flow it only writes. Without a CUDA device the CPU bodies of the same classes stand in |
| `runtime/qpu/discovery:gpu`, CUDA build | Two device components that both give modules: every device of both is there |
| `runtime/qpu/chain:cputype`, `:mp` | The chain test with a quantum body that is not of the QPU type |
| `dsl/ptg/qpu_fallback`, `:qpu` | One executable, one task class with a QPU body and a CPU body: the CPU body runs in a context without a QPU device, the QPU body in one with it |

The tests of M1 and M2 run with QPU-typed bodies, and each of them now ends with a check of the device: it holds no load, the runtime's device statistics count exactly the quantum tasks, and what `parsec_qpu_device_info` says is what the module holds. `overlap` also reads the load while its one job is outstanding. `hardware_guard` was reworked: a backend that may not be used now gives no device, so its task with a QPU body and a CPU body runs the CPU body, and a body that submits all the same is refused.

The seven properties of section 1.9, by test: 1 `overlap`; 2 `dsl/ptg/qpu_fallback`, `discovery`, `hardware_guard`; 3 `capacity`, and the load in `overlap`; 4 `termination`; 5 the device statistics in every test (the trace events are not built); 6 `failure`; 7 the builds above.

### Does a test notice

One deliberate bug at a time in a private copy of the sources, rebuilt, and the QPU tests run. All 26 are caught.

| Deliberate bug | Tests that fail, of 60 |
|---|---|
| The load of a QPU task is not taken back at completion | 48 |
| It is not added before the hook | 49 |
| The scheduler's loop takes the data of a task by position again | 5: the `discovery` tests that run a task |
| A QPU task's inputs are not switched to their host copies, nor counted as read | 5: the same |
| A copy the application manages is expected on the QPU device | 1: `misuse` |
| The type has no name | 7: every `discovery` test |
| `parsec_mca_device_is_gpu` as it was | 51 |
| The time estimate of the QPU derived from flop rates | 52 |
| The type on bit 6 | 55 |
| Every rank owns the QPU | 16 |
| `-1` names one past the last rank | 4: `chain:mp:2`, `chain:mp:4`, `discovery:mp:last`, `failure:mp` |
| The module refuses every taskpool | 49 |
| `parsec_qpu_device_info` always reports no load | 2: `overlap`, `overlap:1worker` |
| It describes a device that was not attached | 1: `hardware_guard:late` |
| The backend is not opened when the device is attached | 8 |
| A device is attached although its backend was refused | 1: `hardware_guard:late` |
| The guard that follows the opening of a backend is dropped | 1: `hardware_guard:late` |
| The executed tasks are not counted | 49 |
| They are counted twice | 49 |
| The time of a job is not measured | 36 |
| It is measured and does not reach the device | 45 |
| Only the result of a QPU task is marked as written | 4: `discovery` and its three MPI variants |
| The version of a written copy is not advanced | 4: the same |
| A rank that does not own the QPU takes the job | 4: `discovery:noowner` and the three MPI variants |
| A backend that was never used is not closed | 1: `discovery:idle` |
| The component offers no module | 53 |

Six more need an accelerator to show, and were run the same way on leconte against the three tests that use one:

| Deliberate bug | Fails |
|---|---|
| A QPU task's inputs are not switched to their host copies | `cpu_gpu_qpu:gpu`, `discovery:gpu` |
| The scheduler's loop takes the data by position again | `cpu_gpu_qpu:gpu` (the QPU body reads device memory: segmentation fault), `discovery:gpu` |
| The host copy does not take ownership when a QPU task writes it | `cpu_gpu_qpu:gpu` |
| Its version is not advanced | `cpu_gpu_qpu:gpu`, `discovery:gpu` |
| The written flows are not announced at all | `cpu_gpu_qpu:gpu`, `discovery:gpu` |
| The list of device components as it was | `cpu_gpu_qpu:gpu`, `discovery:gpu` |

### Sanitizers

The no-MPI builds of section 9.1, 23 runs under each sanitizer: every new test, the tests of M1 and M2 with QPU bodies at larger sizes, and `qpu_fallback` both ways.

- **Thread sanitizer.** 2,795 reports. None is between two accesses of the QPU layer. 28 have a frame of it on one side, and are of the kinds section 9.1 describes: the progress thread's push into a scheduler queue against a worker's pop, and its write of `task->status` against a later reuse of the task's memory. A first version of M3 added a third kind, ten reports of the progress thread writing the device's time estimate against a worker reading it; that write is gone (departure 7).
- **Address sanitizer.** No memory error. 93 leak records, none with a frame of the QPU layer: 87 are the copy of a write-only PTG flow (section 9.1), the others are in DTD and in the task mempool.

### What an independent review found

An independent reviewer read the M3 changes without building them. Three of its findings I had fixed while it read (the progress thread wrote the device's time estimate, which the device layer reads without synchronisation; only the result was marked as written, not every written flow; the access mode of a flow was not passed on). The others:

| Finding | What was done |
|---|---|
| The loop of `__parsec_execute` that switches the inputs of a task to their host copies and counts the body as their reader walked the class's list of input flows and the data of the task with one index (`scheduling.c:161-192` at the base commit). The list is compact, the data are indexed by flow, and the compiler numbers the flows a task writes first: for a task that reads a circuit and parameters and only writes its result, the last input was skipped and the result was taken in its place. This is in the base commit, for CPU tasks too. In a CUDA build a skipped input that comes from a GPU task stays a device pointer. | Fixed, three lines of that loop: the data are taken by the flow's own index. `discovery` checks, in the QPU body, that the scheduler made it the reader of exactly its inputs; `cpu_gpu_qpu:gpu` gives the quantum task a write-only flow and a last input that comes from a CUDA task. |
| In a paranoid build `parsec_select_best_device` ends the run when a QPU task has a copy that the application manages: it expects such a copy on the selected device (`device.c:327-343`), and a QPU holds none. | Fixed, one line: the host, for a device that holds no data. The results of `misuse` are now copies of that kind. |
| Between incarnations of different types the default estimates have no common unit: a quantum task counts for a second, a CPU or GPU task for a ratio of flop rates. The QPU body wins over a CPU body unless `device_load_balance_allow_cpu` is set, and a GPU body wins over the QPU body. | Written into `device_qpu.h`. A class decides with its own `time_estimate`; selection informed by measured latency is later work (section 7.4). |
| A `time_estimate` function that divides by the flop rate of the device divides by zero on a QPU. | Written into `device_qpu.h`: such a function has to tell the QPU by its type. Only classes with a QPU incarnation are asked about the QPU. |
| A QPU body that returns AGAIN has its load counted once more per return; an immediate task class is completed when its body returns, whatever it returns. | Written into `device_qpu.h`. Both are how the scheduler treats a GPU body too. |
| The owner is the rank the process has at `parsec_init`. An application that changes the communicator afterwards (`parsec_remote_dep_set_ctx`) may move rank 0 to a process without the backend. | Written into `device_qpu.h`; not solved. |
| A QPU body that writes a flow and ends without submitting, and a written flow that no successor reads, are not marked as written. | Written into `device_qpu.h`. Marking them needs the PTG compiler to treat a QPU body as it treats a CPU body (one condition, `jdf2c.c:7081`), which section 8 ruled out; it is your decision. It matters only where an accelerator holds another copy of the same data. |
| Tests: `discovery` dropped one error count by assigning where it should add; its skip without a CUDA device asked for the wrong thing; `qpu_fallback` checked which body ran, not that the incarnation was dropped. | Fixed; `qpu_fallback` now reads the incarnations of its task class after the run. |
| A refusal after the device failed to attach gave a stale reason. | Fixed. |

It looked for, and did not find: a pattern of components the new compaction gets wrong; any change for CPU or GPU types when no QPU is enabled; a loop over the devices that takes the QPU for a GPU (`device_gpu.c`, `data.c`, `arena.c`, DTD, generated code); a use of the module after it is freed; an unsynchronised access in the new code.

### Departures from the plan

1. **`insert_function.c` did not change.** A DTD task class whose only chore is of the QPU type goes through the wrapper DTD has for every chore that is not a GPU one, which is what a QPU chore needs: it marks the written tiles and calls the body. `chain:dtd` runs that way, on one and two ranks. A class with a CPU chore and a QPU chore would still lose its CPU body (check 7 of section 1.4); that slot is not built.
2. **`parsec/mca/device/CMakeLists.txt` did not change.** `PARSEC_HAVE_DEV_QPU_SUPPORT` is set next to `PARSEC_HAVE_QPU`, where the option is, as a plain variable: the device layer caches its `PARSEC_HAVE_DEV_*` variables, and a cached one would survive the option being turned off.
3. **The device layer skips any device with no flop rate,** not QPU types by name, when it derives the time estimates.
4. **`device.c:299` did not change.** It decides between two devices that are not GPUs, which needs a second QPU; there is one per process.
5. **Three changes to shared code that the plan did not foresee:** the two fixes above, in the list of device components and in the loop over a task's inputs, and one line of the paranoid check of `parsec_select_best_device`, which took a host copy the application manages for an error when the selected device is a QPU.
6. **The backend is opened when the device is attached,** not at the first submission, so that capabilities are known at start-up and a backend that may not be used gives no device. The progress thread still starts at the first submission: at attach time the runtime has not yet created the thread-local key a thread needs to register its execution stream on a platform without compiler thread-local storage (`parsec.c:932` against `:913`).
7. **The time of a job is measured but not given to the scheduler.** Section 2.8 has the progress thread write it to the module's `time_estimate_default`. I built that first, and the thread sanitizer reported it ten times in 23 runs: the device layer reads that field in `parsec_select_best_device` without synchronisation, as it may for a value set once. The estimate is now a nominal second per task, set at attach, and the measured time is in `parsec_qpu_device_info`. With one QPU nothing depends on the difference. Latency-informed selection among several QPUs will need the device layer to read that field atomically, which is a change to propose with it.
8. **A quantum task on a rank that does not own the QPU is refused, also when its class has a CPU body.** Running the CPU body there instead would hide a misplaced task. The CPU body runs when the context has no QPU device at all.
9. **Every flow a QPU task writes is marked as written,** not only its result, since a QPU body may fill another flow on the host before it submits.
10. **Not built:** the trace events of section 2.10 with `tests/profiling/check-qpu.py`, the DTD slot of departure 1, and more than one QPU per process. Section 7.3 lists them as able to follow the must-have part. `cpu_gpu_qpu`, which that list puts after it, is built.
11. **Tests.** `qpu_discovery` is a JDF, since it needs a task that submits on every rank. `dsl/ptg/qpu_fallback` adds one test to a build without the option, which now has 168.

### For later milestones

- **Hazard H1.** M3 did not touch the lines of `__parsec_execute` that follow an ASYNC return. The load of a QPU task is added before its hook, so the device type adds nothing to the window. The decision of section 9.1 is still yours.
- **M6.** The quantum task class takes `BODY [type=QPU]` and has to be placed on the owner rank by its affinity. A class that should also run without a QPU needs a CPU body; without one, a context with no QPU device stops at the first quantum task with "ran out of valid incarnations".
- **M7.** Coalescing changes how many tasks one job stands for, not the load: a task still counts for one.
- **M8.** `device_show_statistics` gives the mean time of a job and the peaks of jobs outstanding and tasks waiting per run, which is most of what the experiment scripts need from the device.
- **Applications built on PaRSEC** that take every device after the CPU ones for a GPU would be wrong in a context with a QPU. In this tree only a printed number in `tests/runtime/cuda/testing_get_best_device.c:154` does that.

## 9.5 M6: distributed QAOA on PaRSEC (2026-11-23)

Built and run on methane, on top of M3, in the build of section 9.1. Nothing is committed. M6 changed no file of the runtime and none of the QPU component: everything new is in `tests/apps/qaoa/`, and one test of M3 changed (departure 6).

### What exists now

| File | Content |
|---|---|
| `tests/apps/qaoa/qaoa.jdf` | The task graph: ten task classes, and the function that creates the taskpool with its nine arenas |
| `tests/apps/qaoa/qaoa_data.h`, `qaoa_data.c` | Tile collections (where tasks run, and the few tiles that outlive the graph), the arena of one kind of data, the sizes of all of them |
| `tests/apps/qaoa/testing_qaoa.c` | The driver: runs an instance, runs the sequential reference on the same instance, compares |
| `tests/apps/qaoa/CMakeLists.txt`, `Testings.cmake` | One more executable, fourteen more tests |

The quantum task class, `QEXEC`, has one body, `[type=QPU]`, which is one line: `return PARSEC_QPU_SUBMIT_KEYED(C, TH, RES, cfg->shots, qaoa_key(r, b, t, e));`. Every other body is one or two calls into `qaoa_core.c`, the same calls the reference makes.

### The task graph as built

Section 3.6 stands, with four changes that the runtime imposed.

```text
INIT(b)        --x-->  BUILD(0,b), UPDATE(0,b), GATHER(0,d,rev) for each neighbour d updated before b

GATHER(r,b,k)  <--h--  NEW if k = 0, else GATHER(r,b,k-1)
               <--x--  s = nbr(b,k):  UPDATE(r,s) if s is updated before b,  else UPDATE(r-1,s) or INIT(s)
               --h-->  GATHER(r,b,k+1), or BUILD(r,b) after the last neighbour

BUILD(r,b)     <--x--  UPDATE(r-1,b) or INIT(b)
               <--h--  GATHER(r,b,deg-1), or the tile E(b) for a block with no neighbour
               --P-->  EVAL(r,b,*,*), UPDATE(r,b)
               --C-->  QEXEC(r,b,*,*)
               --S-->  STEP(r,b,1)
               --TH0, TH1-->  QEXEC(r,b,0,0), QEXEC(r,b,0,1)        the angles of step 0

STEP(r,b,t)    t = 1 .. T
               <--S--  BUILD(r,b) if t = 1, else STEP(r,b,t-1)
               <--F0, F1--  EVAL(r,b,t-1,0), EVAL(r,b,t-1,1)
               --TH0, TH1-->  QEXEC(r,b,t,0), QEXEC(r,b,t,1)        for t < T
               --S-->  STEP(r,b,t+1), or UPDATE(r,b) when t = T

QEXEC(r,b,t,e) <--C--  BUILD(r,b);   <--TH--  BUILD(r,b) if t = 0, else STEP(r,b,t);   --RES-->  EVAL(r,b,t,e)
EVAL(r,b,t,e)  <--RES-- QEXEC(r,b,t,e);   <--P--  BUILD(r,b);   --F-->  STEP(r,b,t+1)

UPDATE(r,b)    <--S--  STEP(r,b,T);   <--P--  BUILD(r,b);   <--x--  UPDATE(r-1,b) or INIT(b)
               --x-->  BUILD(r+1,b), UPDATE(r+1,b), the GATHER tasks of the neighbours by the colour rule,
                       FINAL(b) after the last round
               --D-->  SUM(r,b)

FINAL(b)       on the rank of A(0): copies the last state of block b next to the others
SUM(r,b)       on the rank of block b: adds the report of UPDATE(r,b) to what SUM(r,b-1) passed on
TOTAL(r)       on the rank of G(r): keeps the sum of round r
```

- **`BUILD` proposes step 0, and `STEP` runs for t = 1 to T.** In a PTG a data flow whose guards are all false is never satisfied (`parsec_check_IN_dependencies_with_mask`, `parsec.c:1406`: "it is assumed that in the data case, one input will always become true"), so a task cannot have an input that exists for some of its instances only. A `STEP(r,b,0)` with no evaluation to read never became ready, and the first version of the graph ran `INIT` and nothing else.
- **A block with no neighbour reads its boundary from a tile**, `E(b)`, which the driver fills with an empty boundary, for the same reason: `BUILD` needs a source for that flow when there is no `GATHER`.
- **`SUM` runs on the rank of its block, and `TOTAL` ends the chain.** In the first version every `SUM` ran on rank 0, so `UPDATE` sent its state to the ranks of its neighbours and its report to rank 0. On three ranks or more that fails an assertion of the runtime (below).
- **The state is not written back to `X(b)`.** `FINAL` collects the last states into one tile on rank 0, which is where the driver compares them with the reference; `FINAL` does not recompute the cut, the driver does.

**Nine kinds of data, nine arenas.** State (232 bytes), boundary, local problem, circuit, optimizer (264), angles ($2p$ doubles), result, evaluation (32), report (32). A flow that a task writes gets its copy from `<- NEW [type = X]`; a dependency that can cross ranks says `[type_remote = X]` at both ends; no dependency has a local type, so nothing is ever converted (the runtime calls that a reshape). For 16 blocks of 16 vertices, one layer and 1,024 shots, a job moves 16 bytes of angles to the owner rank and 12.4 kB of result back, and an update moves 1.4 kB of circuit once.

**The state of a block is one buffer, rewritten in place.** `UPDATE(r,b)` writes it and hands it to its readers: `BUILD(r+1,b)`, the `GATHER` tasks of the neighbours, `FINAL(b)`. Every one of them is upstream of `UPDATE(r+1,b)`, since that update needs the updates of the neighbours, which need their own gathers. So no task reads the buffer while it is rewritten, and a copy sent to another rank has been received before.

**Each task checks the rule its dependencies encode.** A state carries the number of updates its block has had. `GATHER(r,b,k)` requires its neighbour's state to have had `r+1` updates when the neighbour is updated before `b` in a round and `r` otherwise; `BUILD` and `UPDATE` require `r`; `FINAL` requires `R`. A violation is counted per process and fails the run. This is the check section 4.6 asked of `GATHER`.

### What the driver compares

`testing_qaoa` runs `qaoa_sequential_run` on every rank with the same instance, settings and seeds, then the taskpool, and requires:

| On | What |
|---|---|
| Rank 0 | The final state of every block equal to the reference's, 232 bytes each, by `memcmp`: assignment, version, counters, angles, the versions of the neighbours it last read |
| Rank 0 | The final cut equal to the initial cut plus the improvements the updates reported, exactly; the same after every round, against the reference's cut after that round |
| Rank 0 | As many updates, accepted updates, skipped updates, evaluations and failed evaluations as the reference |
| The owner rank | One completed quantum task per evaluation; as many jobs at the backend as the reference asked of its sampler; as many tasks without a job as skipped evaluations; no tag twice, no two jobs of one evaluation at once, no input changed under a job; nothing pending, no load left on the device |
| Every other rank | No backend opened, no quantum task taken |
| Every rank | No task read a state of another round than the graph says |

With `-F keys` one evaluation in five fails for good, chosen by its key, in the mock and in the reference alike; the device then runs three jobs for each of them. With `-F first` and the mock set to fail the first job of every evaluation, the device runs two jobs per evaluation and the application must end exactly as with none failing.

### Verification

All runs with the code as it stands at the end of M6, on 2026-11-23. On methane unless said otherwise.

| Check | Result |
|---|---|
| The modified tree with the option off, full suite | 168 of 168 pass, as after M3 |
| The modified tree with the option on, full suite | 245 of 245 pass: the 231 of M3 and 14 new ones |
| The application's default run: every fixed instance under four settings, each compared with the reference | 44 of 44 equal to the reference, byte for byte, on 1, 2, 3, 4 and 8 ranks |
| The 18 QAOA tests (these 14 and the 4 of M5) 100 times in a row | 1,800 of 1,800 pass |
| QPU and QAOA tests in a profiling build and in a Release build | 78 of 78 each |
| The same in a build without MPI, under the address sanitizer | 57 of 57 (the 21 others need MPI) |
| MPI by hand, four workers per rank, on 2, 3, 4 and 8 ranks: the default run (five times each); with the QPU on the last rank; with failing evaluations (three times each) | 44 runs, none failed |
| The three broadcast topologies of the runtime on 4 ranks; every payload in a message of its own (`runtime_comm_short_limit 0`) on 4 ranks; 16 blocks of 12 qubits on 4 ranks; 40 blocks of 6 on 8 ranks with skipping; 3 blocks on 5 ranks | All pass, three times each |
| The default run under each of the 11 schedulers (twice each), and with two virtual processes (three times) | All pass |
| 16 blocks of 14 qubits, 1,024 shots, 256 jobs, one rank | Equal to the reference, twice |
| Compiler warnings in the new sources, and warnings of the PTG compiler on them, in six builds from scratch | None |
| **leconte, CUDA build with the option on**, full suite, one V100 per rank | 337 of 337 pass: the 323 of M3 and the 14 new ones. No warning of the PTG compiler |

The fourteen new tests: `apps/qaoa` (the default run); `:seed2`, `:seed3` (other samples and other job times); `:1slot`, `:16slots` (one job at a time at the QPU, and sixteen); `:1worker` (one worker thread); `:fail` and `:fail:retried` (the two failure modes); `:generated` (8 blocks of 10 qubits, two layers); `:mp:2`, `:mp:3`, `:mp:4` (the default run on that many ranks; several instances have fewer blocks than there are ranks); `:mp:last` (the QPU on the last of 4 ranks); `:fail:mp`. Every job takes up to two milliseconds, drawn per job, so jobs do not end in the order they were submitted in.

Not done in M6: a multi-node run (every MPI run here is on one node), and a run on a real device.

### A first look at run times

One run each in the Release build on methane, which was not idle, with the mock taking 20 ms per job. This is a look, not an experiment: M8 measures. `ring64` asks for 128 jobs and has two colours of four blocks, so at most eight of its jobs can be outstanding at once.

| Run | Time |
|---|---|
| `ring64`, the reference with one job at a time (`qaoa_reference -b`) | 2,585 ms |
| `ring64` on PaRSEC, the QPU taking 1 job at a time | 2,569 ms |
| 2 at a time | 1,285 ms |
| 4 at a time (the default of the mock) | 643 ms |
| 8 at a time | 324 ms |
| 16 at a time | 324 ms |
| 16 at a time, one worker thread | 332 ms |
| 16 blocks of 12 qubits, 256 jobs: the reference with one job at a time | 5,180 ms |
| The same on PaRSEC, 16 at a time | 325 ms |

Each time is the number of jobs, divided by how many the device and the graph allow at once, times 20 ms, to within two percent (four with one worker thread). What the runtime adds does not show at this job time; at what job time it does is for M8 to measure.

### Does a test notice

One deliberate bug at a time in a private copy of the sources, rebuilt, and the 18 QAOA tests run. All 20 are caught, and the copy passes again once restored.

| Deliberate bug | Application tests that fail, of 14 |
|---|---|
| A job gets the key of the other evaluation of its step; or a key without the round | 14 each |
| A job asks for half the shots | 14 |
| The two evaluations of a step are swapped where the optimizer takes them in | 14 |
| The two sets of angles of step 0 are swapped | 14 |
| The last step is not taken in | 14 |
| A neighbour is folded into the boundary under another slot | 14 |
| The boundary is started afresh in round 0 only | 14 |
| The local problem is built for round 0 whatever the round | 14 |
| A block starts from the assignment of another block | 14 |
| The colour rule reversed at both ends of every dependency, or replaced by the block number: a valid sweep in another order | 14 each |
| The last state of a block is stored in another slot | 14 |
| The improvement of an update is not reported; a sum counts it twice; the total of a round is not kept | 14 each |
| The failed evaluations of an update are not reported | 2: `apps/qaoa:fail`, `apps/qaoa:fail:mp` |
| The result arena has room for half the outcomes | 14 |
| The reference is run with another seed for its samples | 14 |
| The reference is run without the failing evaluations | 2: the same two |

### Sanitizers

The no-MPI builds of section 9.1, seven runs of the application under each: the default run, the two failure modes, one slot, sixteen slots with skipping, jobs that take no time, one worker.

- **Thread sanitizer.** 4,452 reports, none between two accesses of the QPU layer or of the application's own state; six have a frame of the QPU layer on one side and are of the two kinds of section 9.1 (five are the progress thread's write of `task->status` against a later reuse of the task's memory, one is its push into a scheduler queue against a worker's pop).
- **Address sanitizer.** No memory error. Every leak record (151) is the one of section 9.1, 72 bytes for each copy of a flow that a PTG task only writes (`arena.c:283`; 80 with the QPU device on, section 9.7); the application has many such flows, so it shows more often. For the controlled instance of section 3.4 that is about a quarter of a megabyte per run.

### What an independent review found

An independent reviewer read the application without building it, and wrote its own model of the dependencies, which it ran on 4,000 random instances (up to 9 blocks, 4 rounds, 3 steps, 6 ranks). It found no dependency whose two ends disagree, no reader of a state that is not upstream of the next writer, no difference with the order of calls of the reference, no flow in the wrong arena, and no task class with two outputs that leave for different sets of ranks. It agreed with the reading of the broadcast failure below, and made it more precise: the failure needs a rank that receives two outputs and whose predecessor in the chain of the first one does not receive the second.

What it did find, all in the driver and around the taskpool:

| Finding | What was done |
|---|---|
| A rank that found no QPU device returned by itself and left the others to start a run whose quantum tasks never complete; an owner rank that does not exist was placed on the last rank without a word | The ranks now decide together before anything starts, and say which it is. Both cases end the run at once on 2 and 4 ranks |
| `qaoa_taskpool_new` refused no rounds and no steps, and took any number of layers and shots. With the comparison off, five layers write past the angles of the optimizer state | It refuses what the reference refuses |
| Two comparisons of the driver tested nothing: the initial cut against the reference's (both computed by the driver), and a sum that is an identity of the reference. With failing evaluations nothing required that any had failed | The two are gone. The default run with failing evaluations now requires between one in ten and one in three to have failed (313 of 1,634 did) |
| The line a run prints said "the reference, bit for bit" before the QPU and the other ranks had been checked | The line is about the final states only; the verdict is the last line |
| The tiles of `X` kept the datatype of a state after the taskpool had freed it | Cleared when the taskpool is freed; the header says a collection serves one taskpool |
| The path that refuses a configuration freed nothing | Fixed |
| The check on the number of updates of a state does not tell a neighbour from another of the same round | Not changed: the state does not carry its block. The final comparison catches it, and so does the deliberate bug "another slot" below |

Two more are about the runtime and the machines, not about these files. The taskpool's destructor may run on another thread after the driver freed its collections, which it reads for devices that register memory: a window the runtime documents, harmless without an accelerator in the context. And equality byte for byte assumes that every rank computes `pow` alike, which nodes with different processors need not do: a multi-node run on mixed nodes may differ in a last bit and say so.

### Things found in the runtime

None is caused by this work. The first one changes what a PTG application may do, and nothing in the tree documents it.

| What | Where | Effect on M6 |
|---|---|---|
| With the default (chain) broadcast, a task that sends two different outputs to different sets of ranks can fail `Assertion 'idx < data_sizes[0]'` on three ranks or more. A rank that the first output reaches through a relay is marked as served and is not sent the second output, which the relay does not have; the receiver then finds a consumer for a payload that is not in the message. The star and the binomial topology (`runtime_comm_coll_bcast` 0 and 2) ran the same graph. The cause is read from the code, by me and by the reviewer; it is not reduced to a small test | `remote_dep_comm.c:1153`; the mark is taken per rank and for all outputs at once in `parsec_remote_dep_activate` (`remote_dep.c:566-628`), while a message carries only the outputs its receiver consumes (`remote_dep_comm.c:1536-1540`) | `SUM` moved to the rank of its block, so that `UPDATE` has one output that leaves its rank |
| A data flow whose guards are all false is never satisfied, and the task never runs. For a control flow the same thing means "none expected" | `parsec.c:1406-1470` | `BUILD` proposes step 0; the tile `E(b)` |
| Two named ranges in the dependencies of one flow cannot use the same index name: the generated function declares it twice | `jdf2c.c`, the code generated for `iterate_successors` | The two ranges of `UPDATE` use `k` and `j` |
| Every reconfiguration rewrites `parsec_options.h` and rebuilds everything: the file is configured twice with different content when 128-bit compare-and-swap is used | `CMakeLists.txt:1259-1262` | Build time only |

### Departures from the plan

1. **The four changes to the task graph** listed above: step 0 in `BUILD`, the tile of an empty boundary, `SUM` on the rank of its block with `TOTAL` at the end, and no write-back to `X(b)`.
2. **Nine arenas** where the plan counted six kinds of flow data: the optimizer state, the evaluation and the report are kinds of their own.
3. **Five collections** where section 3.6 had two: `A` (the final states), `G` (the sums per round) and `E` (the empty boundaries) besides `X` and `Q`. All are of one small type, in `qaoa_data.c`.
4. **A round and a step at least.** The graph has no task for an update without evaluations; `qaoa_taskpool_new` refuses such a configuration. The reference accepts it.
5. **The test names.** `apps/qaoa:mp` is `:mp:2`, `:mp:3`, `:mp:4` and `:mp:last`; the delay seeds and the slot counts of section 4.5 are tests of their own; `apps/qaoa:fail:retried` is new. `apps/qaoa:coalesce` waits for M7.
6. **One test of M3 changed.** `cpu_gpu_qpu.jdf` took its second tile from the collection it is placed by, at another index, which the PTG compiler warns about; I had searched the build logs of M3 for lower-case "warning" and missed it. The tile now comes from a second collection declared as aligned with the first. Rerun on leconte: it passes on a CUDA device.

### For later milestones

- **M7.** The device's pending heap orders by task priority, and no task of the application has one. Coalescing will meet jobs of different blocks whose circuits differ in size; the result arena is sized for the largest block.
- **M8, baselines.** C-sync is this graph with a body for `QEXEC` that blocks its worker; the body is one more incarnation, or one more line of `qaoa.jdf`. Baseline B needs the same keys. (A body that blocks its worker is C-block of section 6.2, not C-sync; M8 built C-sync as section 6.2 describes it, with a barrier between colours, and C-block in the benchmark of E2; section 9.7.)
- **M8, measurements.** The driver prints the time from `parsec_context_add_taskpool` to the end of `parsec_context_wait` per run; the comparison with the reference is outside it and can be turned off (`-c 0`). The reference runs on every rank when it is on, which a large run should avoid.
- **M8, memory.** The 72 bytes that every copy of a written flow leaks add up over a long run.
- **A second application** written as a PTG has to respect the first two rows of the table above.

## 9.6 M7: tasks that share a job (2026-11-23)

Built and run on methane, on top of M6, in the build of section 9.1. Nothing is committed. M7 changed no file of the runtime outside the QPU component: the twelve tracked files that M1 to M5 edited are as M6 left them.

### What exists now

| File | Content |
|---|---|
| `parsec/mca/device/qpu/device_qpu_progress.c` | A task and a job are two records. New: how a job is put together, and every path a job of several tasks takes (end, failure, refusal, time-out, cancellation, a backend out of reach, shutdown) |
| `parsec/mca/device/qpu/device_qpu_backend_mock.c` | Jobs of several circuits, a cost of one part per job and one per circuit, a fault that cancels a job at the provider, two more things the mock watches its caller for |
| `device_qpu_backend.h`, `device_qpu.h`, `device_qpu_internal.h` | The contract of a job of several circuits; `max_batch` in the description of the device; three new counters of the service and five of the mock |
| `device_qpu_component.c`, `device_qpu_module.c` | Three parameters, the statistics line |
| `tests/runtime/qpu/qpu_coalesce.jdf` | New: the runtime test of this milestone, 24 variants |
| `tests/runtime/qpu/qpu_mock_test.c` | The mock with several circuits per job |
| `tests/runtime/qpu/qpu_test_common.c`, `tests/apps/qaoa/testing_qaoa.c` | The checks every test ends with, for jobs that carry several tasks |

Parameters: `device_qpu_max_batch` (most tasks in one job; 1 by default, so nothing changes for a run that does not ask), `device_qpu_mock_job_delay` (the time every job takes whatever it carries; `const:0` by default), `device_qpu_mock_max_circuits` (64).

### The policy as built

- **No task waits for company.** A job is submitted as soon as a place at the backend is free and a submission is allowed, with whatever waits at that moment.
- **How many go together.** The tasks that wait are spread evenly over the jobs that can be submitted at that moment: one per free place, or a single one when submissions have to be spaced (`device_qpu_submit_interval`). So with a place for every task that waits no task shares a job, and with one place the job takes all that wait. Never more than `device_qpu_max_batch`, nor than the backend takes circuits in a job.
- **Which ones.** In the order of the heap, which is task priority and then arrival, up to the first task that cannot share: nobody is taken from behind it. Two tasks can share when they ask for the same number of shots and neither has had a job.
- **A job is one thing to the backend.** It ends well for all its tasks or for none. Every result is written before the first task is handed back; the tasks then complete on their own.
- **Failure.** A job that fails, is refused, gives no result or takes too long sends each of its tasks back to wait, and each is tried again alone: a task whose circuit makes jobs fail takes nobody with it twice. That job counts as one of the jobs each of its tasks may have (`device_qpu_max_retries`). A job that somebody cancels at the provider completes its tasks as cancelled, as it did for one task. When the backend has been out of reach for too long, the job that was being asked for is taken back and its tasks complete as unavailable with those that wait.
- **Time-out.** `device_qpu_job_timeout` is the time a job of one task may take; a job of n tasks may take n times that. Tasks that would each have been given that long are not cut short for having gone together.
- **What a task reads.** `batch` in its result is the number of tasks in the job that gave it; `attempts` counts the jobs it took, shared or not.

The tag of a job is that of its first task with that task's attempt number, so asking again for a job that the backend did not confirm finds the same job, and no two jobs have the same tag.

### The cost of a job on the mock

A job waits its queue delay in its slot, then runs for `device_qpu_mock_job_delay` plus `device_qpu_mock_delay` for each of its circuits. What is drawn for the job comes from its first circuit, what is drawn for a circuit from its own key and attempt. The counts of a circuit depend on the seed and its key only: a circuit gives the same counts alone, in a job of sixteen, and at its third attempt. One helper thread samples the circuits of a job one after the other.

A job has one fate: the first that is not "ok" among those of its circuits. So how many jobs a task takes now depends on which tasks it met; its counts do not.

### Verification

All runs with the code as it stands at the end of M7, on 2026-11-23. On methane unless said otherwise.

| Check | Result |
|---|---|
| The modified tree with the option off, full suite | 168 of 168 pass, as after M6 |
| The modified tree with the option on, full suite | 277 of 277 pass: the 245 of M6 and 32 new ones |
| Every QPU and QAOA test 100 times in a row | 11,000 of 11,000 pass (110 tests) |
| QPU and QAOA tests in a profiling build and in a Release build | 110 of 110 each |
| The same in a build without MPI, under the address sanitizer | 85 of 85 (the 25 others need MPI) |
| The application with tasks that share jobs, each run compared with the reference | Equal to the reference, byte for byte, on 1, 2, 3, 4 and 8 ranks, with one place and with three, with evaluations that fail for good and with every first job failing |
| MPI by hand, four workers per rank, on 2, 3, 4 and 8 ranks; the 11 schedulers; two virtual processes; one worker | 90 loops, 273 runs, none failed. (A 91st loop is a control: see the scheduler `llp` under "Things found in the runtime") |
| Compiler warnings in the new and changed sources, and warnings of the PTG compiler on them, in six builds from scratch | None |
| **leconte, CUDA build with the option on**, full suite, one V100 per rank | 369 of 369 pass: the 337 of M6 and the 32 new ones. No warning of the PTG compiler |

Not done in M7: a multi-node run, and a run on a real device, as before. No real provider was asked whether it takes several circuits per job the way the mock does; section 2.6 has what the documentation says.

### The tests

`runtime/qpu/coalesce` lets one task go first and holds the backend inside the submission of its job until all the others are with the service. From there on which tasks go into which job depends on the limits alone, and the test compares it job by job and task by task. The quantum tasks have priorities, so the order in which they wait is known.

| Test | What it pins down |
|---|---|
| `coalesce` | One place, 16 tasks wait, limit 4: four jobs of four, tasks 1 to 4, 5 to 8 and so on |
| `:off`, `:cap` | Limit 1: 16 jobs. Limit 4 and a backend that takes 2 circuits: jobs of 2 |
| `:arrival` | Tasks with no priority: the same jobs, of whichever tasks arrived first |
| `:spare` | 8 places, 5 tasks wait, limit 8: five jobs of one |
| `:spread` | Two free places, 13 tasks wait, limit 8: 7 and 6, not 8 and 5 |
| `:interval` | Places to spare and 5 ms between two submissions: each submission takes 4 |
| `:shots` | Numbers of shots that change every three tasks: jobs of 2, 3, 3, ... that end where the number changes |
| `:fail`, `:fail:always`, `:fail:noretry` | A circuit that fails once, every time, and with no retry allowed: the job fails, its four tasks are tried again one by one; the one that always fails ends as failed after its three jobs and the others end well; with no retry all four end as failed |
| `:fail:overtaken` | Two tasks of a higher priority arrive while the job that fails is at the backend: they go next, together, and the four that failed are not taken along |
| `:cancelled` | A job cancelled at the provider: four tasks read "cancelled", nothing is tried again |
| `:deadline` | Jobs of four circuits of 60 ms under a time-out of 200 ms: none is cancelled |
| `:unavailable` | The backend takes one job and loses every answer after it: the job of four that is being asked for is taken back, 16 tasks complete as unavailable |
| `:abandoned`, `:abandoned:asked` | The service stopped with a job of four at the backend, or being asked for, and twelve tasks waiting: 16 tasks given up, none completed |
| `:down`, `:lost`, `:mixed`, `:timeout` | 300 tasks (120 in the last), two places, jobs of up to 8, through a backend that takes every other submission, loses answers, refuses jobs, fails them, gives no result, cancels, hangs and reports ends twice: each kind happens, every result that is there is right, nothing is left |
| `:stress` | 600 tasks that arrive as they come |
| `:mp`, `:fail:mp` | On 2 ranks, and on 4 with the QPU on the last |
| `termination:loop:coalesce` | Fifty taskpools whose tasks are completed four by four |
| `apps/qaoa:coalesce`, `:fail`, `:fail:retried`, `:3slots`, `:interval`, `:mp`, `:fail:mp` | The application with jobs of up to 8 tasks on one place, of up to 4 on three, and of up to 16 under a submission interval: final states equal to the reference, also with failing evaluations |

The checks every QPU test ends with were written for one task per job. They now say the same thing for any job size: the circuits the backend was given are the tasks that needed a job, each in exactly one job per attempt.

### The regimes of section 6.1 on the mock

Section 6.1 took its numbers from two throwaway scripts, which are gone. I wrote the second one again (`model.py`, with the other scripts of M7, outside the repository): the task graph on a ring of 16 blocks, no classical time, no overhead, and the policy as built. With tasks that reach the device one at a time it gives the table of section 6.1 to within 3% (the plan's block graph was a 4 x 4 grid; at constant job times both advance colour by colour). It also says what the rule as built changes against the plan's. Nothing when tasks reach the device one at a time, since a place that frees then takes whatever waits. When a burst is seen at once, the plan's rule puts all of it in one job with every place free: 0.31 with unlimited places and a limit of 16, three times slower than no sharing, where the rule as built gives 1.00.

Measured in the Release build: `testing_qaoa -g 16:4:1:1 -T 8 -R 4` (16 blocks of 4 qubits in a ring, 1,024 evaluations; the final states equal the reference in every run), a job taking 2 units and 0.358 more per circuit as in section 6.1, with the unit at 1 ms (median of three runs) and at 10 ms (one run). Each cell is the time with every task its own job over the time with the given limit.

| Regime | Limit 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| **One place** — section 6.1 | 1.73 | 2.74 | 3.82 | 3.88 |
| measured, unit 1 ms | 1.74 | 2.76 | 3.86 | 3.92 |
| measured, unit 10 ms | 1.74 | 2.75 | 3.84 | 3.89 |
| **One submission per 12 units, places to spare** — section 6.1 | 2.00 | 3.99 | 7.97 | 15.8 |
| measured, unit 1 ms | 2.00 | 4.01 | 8.02 | 16.0 |
| measured, unit 10 ms | 2.00 | 4.01 | 8.03 | 16.1 |
| **Three places** — section 6.1 | 1.73 | 2.66 | 2.24 | 1.79 |
| measured, unit 1 ms | 1.73 | 2.37 | 2.29 | 2.24 |
| measured, unit 10 ms | 1.74 | 2.54 | 2.32 | 2.44 |
| **64 places** — section 6.1 | 1.00 | 1.00 | 1.00 | 1.00 |
| measured, both units | 1.00 | 1.00 | 1.00 | 1.00 |

- **One place, and a submission interval: reproduced to within 2%.** These are the two regimes the acceptance criterion asks for. With one place the 1,024 circuits went in 512, 256, 132 and 127 jobs; under the interval in 512, 256, 128 and 64.
- **Places to spare: no task shared a job**, 1,024 jobs at every limit.
- **Three places: the shape of the model, not its numbers.** A limit of 4 is the best, as predicted, but the measurements move from run to run (293 to 334 jobs at a limit of 4) and lie 5 to 11% below the model at 4 and 25 to 36% above it at 16. Both come from the same thing: what a job carries there depends on how many tasks the progress thread finds when a place frees. The model lets them arrive one at a time; the runtime finds them a few at a time, and spreads what it finds over the places that are free, where the plan's rule put it all in the first.
- **What the runtime adds.** Against the model, which has no overhead, the runs with every task its own job took 0.2%, 0.05% and 0.7% longer at the 10 ms unit (one place, interval, 64 places) and 2.3%, 0.5% and 6.5% longer at the 1 ms unit: about 55 microseconds per job with one place, 0.15 ms per step of the graph with 64. M8 measures this properly.

### Does a test notice

One deliberate bug at a time in a private copy of the sources, rebuilt, and the 108 QPU and QAOA tests of that copy run. All 37 are caught, and the copy passes before and after.

| Deliberate bug | Tests that fail |
|---|---|
| **Which tasks go together** | |
| A job takes all that wait, whatever places are free | 2: `coalesce:spare`, `:spread` |
| With a minimum time between submissions, the tasks are spread over the free places all the same | 2: `coalesce:interval`, `apps/qaoa:coalesce:interval` |
| The number of tasks per job is rounded down; the first task is not counted among those that wait | 1 (`:spread`); 13 |
| The limit on tasks per job is ignored; the backend's limit on circuits is | 19; 1 (`:cap`) |
| A job takes one task more than it should | 76: every test that submits |
| The number of shots is not compared | 1: `:shots` |
| A task that had a job shares its next one; or is taken into the job of a task that had none | 9; 3 (`:fail:overtaken`, `:mixed`, `:timeout`) |
| A task that cannot share is passed over, and the job filled from behind it | 2: `:fail:overtaken`, `:shots` |
| The priority of the tasks is ignored | 20 |
| **What a job is to the backend** | |
| The tag of a job has no attempt number | 18 |
| The requests, or the result buffers, are in another order than the tasks | 27; 26 |
| A shared job counts as a job of its first task only | 18 |
| The size of the job is not recorded for the results | 19 |
| A job of several tasks has the time-out of a job of one | 1: `:deadline` |
| **When things go wrong** | |
| A job that fails completes all its tasks but the first as failed | 7 |
| A job cancelled at the provider is tried again | 1: `:cancelled` |
| A job that has to be asked for again is put together anew | 6 |
| Its tasks count as one while it is asked for | 4 |
| It is not taken back from a backend that stays out of reach | 2: `failure:lost:down`, `:unavailable` |
| Stopping the service with a job left counts one task | 2: `:abandoned`, `:abandoned:asked` |
| The tasks that shared jobs are not counted | 27 |
| **The mock** | |
| The part of a job is charged for each circuit | 18 |
| The fate of a job is that of its first circuit only | 9 |
| A circuit gets the counts of another; only the first result is written; every circuit is sampled with the key of the first | 27 each |
| A job cancelled at the provider is reported as done | 3 |
| More circuits taken than it says it takes; results given for another number of circuits than the job has; two circuits of one evaluation in one job not seen; inputs that change under a circuit other than the first not seen; mixed shot counts not seen; a job asked for again with other circuits not seen | 1 each: `runtime/qpu/mock` |

Two bugs of a first list taught something. One changed nothing that can be observed (the tasks of a job given up on went back to wait and were completed by the next statement), and was replaced. One was caught by no test: a task that had a job being taken into the job of a task that had none. With equal priorities such a task is always ahead of every task that has had no job, so the case never arose; `coalesce:fail:overtaken` makes it arise, and the reviewer had named the same gap from its model.

### Sanitizers

The no-MPI builds of section 9.1, 29 runs under each: the application with tasks that share jobs (four settings), the runtime test in 17 of its modes, the mock's unit test, and the seven runs of M6.

- **Thread sanitizer.** 10,356 reports. None is between two accesses of the QPU layer or of the tests' own state. 24 have a frame of the QPU layer on one side and are of the two kinds of section 9.1 (13 are the progress thread's push into a scheduler queue against a worker's pop, 11 its write of `task->status` against a later reuse of the task's memory). One is new and belongs to the test that stops the service under a running taskpool: the stopping thread frees the backend that the workers read when they submitted, and nothing but time orders the two. That is the application error the test stages; with a taskpool that is waited for, no thread submits while the service stops. A first version of the test held its own lock while it asked the mock for its counters, which the sanitizer reported as a possible lock inversion; it no longer does.
- **Address sanitizer.** No memory error in 29 runs, the two that stop the service with work left included. Every leak record (311) is the one of section 9.1, 72 bytes for each copy of a flow that a PTG task only writes (80 with the QPU device on, section 9.7).

### What an independent review found

An independent reviewer read the change without building it and wrote its own model of how jobs are put together and of the mock's slots, which it ran against the expectations coded in the test (3,656 configurations of limits, and 13,068 of the failure modes). It found no path that loses a task, completes one twice, reuses a tag, leaves a count or a stale result behind, or touches a task after it was handed back.

| Finding | What was done |
|---|---|
| The time-out of a job was the same whatever the job carried: with circuits of 5 ms and a time-out of 30 ms every job of six tasks or more was cancelled and its tasks tried again alone, slower than no sharing | The time-out is per task carried; `coalesce:deadline` |
| `coalesce:spread` expected sizes that hold only while the first job is at the backend; a process stopped for 50 ms at the wrong moment gives other sizes | The test records when the jobs were made and does not compare them one by one when that happened |
| Nothing tested the order in which waiting tasks are taken, that a job ends at the first task that cannot share, or that a task that had a job is not taken into another one | The quantum tasks have priorities; the test compares the tasks of every job; `:shots` expects exact jobs; `:fail:overtaken` |
| Paths that no test reached: shutdown with work left, a job cancelled at the provider, tasks handed back several at a time under a memory checker | `:abandoned`, `:abandoned:asked`, the mock's `cancelled` fault with `:cancelled`, `termination:loop:coalesce` |
| The test's expectations for a failing circuit were wrong above 200 retries, where the service stops counting | The test clamps as the service does |
| Two places for twelve tasks cannot tell rounding up from rounding down | Thirteen tasks |
| The contract did not say whether `max_shots` is per circuit, what 0 circuits per job means, or that the stopping thread calls `abandon()`; the mock did not notice a job of mixed shot counts or a job asked for again with other circuits | Said; the mock counts both, and every test requires zero |
| The tests with faults drawn by the mock required only that some job carried several tasks | Each kind of fault that is asked for has to happen; the rare kinds are also given one evaluation by its key |
| A field that was written and never read, two comments that were out of date | Removed, corrected |

### Things found in the runtime

None is caused by this work. The first two are defects; the last two are how things are.

| What | Where | Effect on M7 |
|---|---|---|
| A task class with no data input gets no input section in the function the PTG compiler generates to say what datatype a dependency has. A control input that reaches such a class from another rank falls through to the section for outputs, where the bit of its flow can match a dependency that has data. The receiving rank then asks the sender for data; the sender, which sent none, fails `Assertion '0 != deps->pending_ack'` | `jdf2c.c:6429-6437` emits the input section only when it is not empty; the receiver asks at `remote_dep_comm.c:908`; the sender asserts at `remote_dep_comm.c:1856` | The gate of `qpu_coalesce.jdf` is in front of the quantum tasks, which are on its rank and have data inputs, and not in front of the tasks that write the circuits |
| A thread that adds a taskpool keeps the first of its start-up tasks for itself until it enters the wait. A program that starts the context and does not wait leaves that task where no other thread takes it | `scheduling.c:445-456` | The test that stops the service under a running taskpool does it from a thread of its own |
| The scheduler `llp` fails `Assertion '0 == item->refcount'` when tasks have different priorities, in a build that checks its lists (`PARSEC_DEBUG_PARANOID`). The existing `dsl/ptg/startup`, which has no quantum task, fails the same way under `--mca sched llp`, three times out of three | `sched_llp_module.c:343` (`lifo_merge_ring`), from `list_item.h:77` | The runtime test has a mode without priorities (`-p 0`), which is what runs under `llp` in the runs by hand |
| The scheduler `rnd` gives every task a random priority when it schedules it, so tasks reach the QPU device with that one and wait in that order. This is what that scheduler is for | `sched_rnd_module.c:108` | The same mode, under `rnd` |

### Departures from the plan

1. **Tasks are spread over the free places.** Section 2.8 packs up to `max_batch` tasks into the job of the place that frees. With several places free at once that puts everything in one job and leaves the others empty, and with a place for every task it makes tasks share for nothing. The rule as built is the plan's when one place frees at a time.
2. **The time-out is per task carried**, which the plan did not consider.
3. **The mock has a fault for a job cancelled at the provider** (`cancelled`), which M2 had left untested, and counts jobs of mixed shot counts and jobs asked for again with other circuits.
4. **`device_qpu_mock_job_delay` and `device_qpu_mock_max_circuits`** are new parameters; section 7.2 named `device_qpu_max_batch` only.
5. **More tests than the two of section 7.3**: 25 runtime tests and 7 of the application.
6. **Files.** `device_qpu_backend.h`, `device_qpu.h`, `device_qpu_internal.h` and `device_qpu_component.c` changed besides the three files section 7.3 lists.

### For later milestones

- **M8, E5.** The batch limit is not "bigger is better" with several places: the first tasks of a burst take the free places one each, and the first place that frees then takes all that wait. Measured with three places: 2.4 to 2.5 at a limit of 4, 2.3 at 8, 2.2 to 2.4 at 16, and not the same from run to run.
- **M8, measurements.** `device_show_statistics` now says how many jobs carried how many tasks and the largest job. The mean time of a job mixes jobs of all sizes.
- **M8, failures.** With faults drawn per circuit a job of n circuits fails more often than a job of one: 39% of the jobs of 16 at 3% per circuit, each followed by 16 jobs of one. The limit and the retry rule interact; an experiment with faults should say so.
- **M4.** A provider's job of several circuits has to end well for all or for none, with a result per circuit in the order given, and all circuits of a job ask for the same number of shots: that is the contract the service relies on.
- **A real device.** Sharing changes how many jobs a task takes when jobs fail, so a run on hardware is reproducible in its counts only as far as the simulator is.

## 9.7 M8: baselines, experiments and documentation (2026-11-24)

M1 to M7 were committed on 2026-11-24, before M8 began, in five commits by area on `qinglei/hybrid_HPC_QC` (`9d108b816` the device type and three runtime fixes, `e1accb548` the component, `f55aa7fea` its tests, `38b433993` the application, `8cb244735` this document); each of the first three builds and passes its own suite from scratch (167, 167 and 252 tests). The commits are by area and not by milestone: the same files changed in several milestones. M8 was committed after its experiments, in five commits on top of those: `f0d3a53ae` the seed of the mock's times, `58d5fde3a` the benchmark, `e77de8362` the application, `82a5183f2` the README, and the commit that adds this section. The code of the first three is the code the experiments ran, file for file (checked against the record of the last run, `exp_qpu/provenance/methane-8cb244735-299dd9d803ed/`).

### What exists now

| File | Content |
|---|---|
| `tests/apps/qaoa/qaoa_mpi.c` | Baseline B: the algorithm as a hand-written MPI program on the same kernels and the same mock |
| `tests/apps/qaoa/qaoa.jdf` | C-sync (`-y`): a tree of `JOIN` tasks per colour, a barrier between colours; a log of the timestamps of every job and update |
| `tests/apps/qaoa/testing_qaoa.c` | `-y`, `-L` (the log), `-M` (a line for scripts); the clock starts after a barrier, and the time is the slowest rank's |
| `tests/apps/qaoa/qaoa_graph.c` | Block graphs shaped as a ring, a square grid, all pairs, or random pairs (`-g grid:16:10:4:2`) |
| `tests/apps/qaoa/qaoa_reference.c` | `-G` (the block graph, for the model), `-D` (the seed of the mock's times), a line for scripts |
| `tests/runtime/qpu/qpu_bench.jdf` | E1 and E2: chains of classical work and quantum jobs, with the QPU body (C), an empty CPU body, a CPU body that holds its worker (C-block), or without the runtime (A) |
| `parsec/mca/device/qpu/` | `device_qpu_mock_delay_seed`: the mock's times drawn from a seed of their own; `parsec_qpu_mock_time()` |
| `tools/qpu/README.md` | How to build, write a quantum task, the parameters, the application, the benchmarks |
| Tests | 22 new: B on 1, 2 and 4 ranks, with spare ranks, on a random block graph, one job at a time; C-sync on 1, 3 and 4 ranks, with failures, with spare ranks, on a complete block graph; the shapes; the benchmark in six settings; the instances pinned by digest |

The experiment harness is outside version control, in `exp_qpu/` of this working tree: the model (`model.py`), the runner (`runner.py`, one CSV row per run, the whole output of each run kept, a provenance record per code version), one script per experiment, `analyze.py` for tables and plots, `retry_late.py` to run again once each run that had a late job, and `host_load.sh` for the load of a shared machine. The scripts and logs of the checks under Verification, Does a test notice and Sanitizers are in `exp_qpu/checks/` (`histamine2/`, `leconte/`, `serotonin/`).

### The systems as built

| System | What it is |
|---|---|
| A | `qaoa_reference -b`: one job at a time through the mock, in the order (round, colour, block, step, evaluation) |
| B | `qaoa_mpi`: blocks on rank b mod P, rank 0 owns the mock and runs a helper thread that keeps at most `max_outstanding` jobs there, first come first served, with the submission interval; colour by colour, both evaluations of a step in flight; `MPI_Iallgatherv` of the states at the end of each colour. Its ranks wait in `MPI_Waitany`, which spins |
| C-sync | `testing_qaoa -y`: the task graph with a barrier between colours, a binary tree of `JOIN` tasks per colour |
| C | `testing_qaoa` |
| D | `testing_qaoa` with `device_qpu_max_batch` above 1 |
| C-block (E2) | `qpu_bench` with `-b`: the CPU body of the quantum task sleeps for the time of the job, then samples |

All of them run the kernels of `qaoa_core.c` in the reference's order per block and name every job by its key, so all of them end in the reference's final states byte for byte, which every run of every experiment checked.

### How the experiments were run

| | E1, E2 | E3 (and E4 from its logs) | E5 | E6 |
|---|---|---|---|---|
| Machine | serotonin: AMD Ryzen 9 7950X, 16 cores, one socket, an exclusive Slurm allocation | methane: Intel Xeon Gold 6140, cores 0 to 17 (first socket) | methane, cores 18 to 35 | methane, cores 18 to 35 |
| Processes | one, 1 to 14 workers (E1), 8 workers (E2) | 4 ranks of 4 cores (A: one process) | 4 ranks of 4 cores | 1 to 8 ranks of 2 cores |
| Job times | none (E1); 5 ms (E2) | lognormal, mean 20 ms, coefficient of variation 0, 0.5, 1, 2; checks at 0.1 s and 1 s | 10 ms for a job of one circuit; check at 0.1 s | 0.2 s |
| Instance | `qpu_bench` chains | 16 blocks of 8 vertices: grid, ring, random, complete; T 8, R 4 (1024 jobs) | 16 blocks of 10, grid | rings of 4 to 64 blocks of 12; T 8, R 2 |
| Runs per setting | 12 (E1), 3 (E2), in rotating order | 5 seeds of the job times (3 at CV 0 and at one place) | 3 (constant times), 5 (with a queue delay) | 3 |

The build is Release (`-O3 -DNDEBUG`, gcc 11.5, Open MPI 5.0.5, hwloc 2.11.1) of `8cb244735` with the M8 changes before they were committed (their code is that of `e77de8362`). Methane is a login node. E5 and then E6 ran on its second socket from 17:12 to 18:05 UTC, and E3 on its first from 17:59 to 20:56 (and its 11 runs again until 20:59). Methane is shared with the CI runner of another project, which takes cores without warning: `exp_qpu/host_load.sh` logged the load and every process of another user above a tenth of a core every 5 minutes from 18:01 (`exp_qpu/host_load.log`). Such processes were seen between 18:41 and 19:01 (up to 18 cores, at 18:56), 20:06 and 20:21, and 20:26 and 20:51 (under 2 cores), after E5 and E6 had ended. The 364 E3 runs that overlap those windows do not differ from the others: a run's time over the model, divided by the median of the same over its setting, is 0.992 to 1.006 inside the windows and 0.995 to 1.005 outside (5th to 95th percentile), so they are kept. Each run is one process tree in a session of its own, killed with it on a time-out, and writes one CSV row next to its whole output; every row names a provenance record (`exp_qpu/provenance/<id>/`) with the commit, the uncommitted changes, the digests of the programs and of the harness, the configure options and the machine. A run is done again whenever any of those changed.

What every run reports and checks:

- **Time**: from a barrier after setup to the end of the wait, the slowest rank's. **Processor time**: user and system time of all ranks (`getrusage`), the mock's threads included, since the mock runs in rank 0.
- **Correctness**: every run of A, B, C-sync, C and D compares its final states with the sequential reference byte for byte, and C-sync counts its `JOIN` tasks. A run that fails a check does not count.
- **Late jobs**: the mock counts the jobs whose sampling outlasted their drawn time; a run with any does not count (section 6.7).
- **Seeds**: the algorithm's seed is 1 everywhere; the job times come from `device_qpu_mock_delay_seed`, and the answers do not depend on it, so every run of every system computes the same states.

The model (`exp_qpu/model.py`) draws every job time as the mock does, bit for bit (the same mixing function and the same Box-Muller step), so a prediction drawn next to a measurement is for the same job times. It has no classical time and no runtime overhead, as in section 6.1.

### Results

Tables are in `exp_qpu/tables/`, plots in `exp_qpu/plots/`, both made by `analyze.py` from the CSV files and logs alone.

**E1: what a quantum task costs** (240 runs, all passed). `qpu_bench` with jobs of no time, the QPU body (C) against the same chains with an empty CPU body; the difference is what the QPU path adds: the hand-off to the progress thread, the mock's submission and end, and the completion back to the scheduler.

| Workers | 1 | 2 | 4 | 8 | 14 |
|---|---|---|---|---|---|
| One chain of 2000 links: added latency per quantum task, us | 65.2 | 62.4 | 50.9 | 40.8 | 36.9 |
| 64 chains of 200 links: added time per quantum task, us | 7.6 | 8.9 | 7.1 | 6.5 | 5.6 |

The hypothesis holds by three orders of magnitude: a quantum task costs tens of microseconds of latency and a few of throughput, and holds no worker. Why the latency of one chain falls as workers are added was not measured; the likely cause is that with more workers one of them is awake when a job ends, so it is the latency of waking a thread and not work.

**E2: holding a worker** (324 runs, all passed). Chains of 20 links of classical work and a 5 ms job, 8 workers:

| Chains | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| C-block / C, work 1e-4 of a job | 1.00 | 0.99 | 1.02 | 1.03 | 2.05 | 2.09 |
| C-block / C, work 1 job | 1.00 | 1.00 | 0.99 | 1.00 | 1.90 | 1.90 |
| C-block / C, work 10 jobs | 1.00 | 1.00 | 1.00 | 1.00 | 1.45 | 1.10 |

With no more chains than workers, a body that blocks costs nothing; beyond, it doubles the time, as the bound of the text says (`ceil(chains / workers)` rounds of jobs). The gap closes only when classical work saturates the workers anyway (10 job times of work, 16 chains: C 2.01 s against a bound of 2.0 s, C-block 2.22 s against 2.2 s). A, one job at a time without the runtime, takes chains times longer throughout. QAOA on a device, with jobs of seconds and 50 to 100 us of classical work per job, sits near 1e-5, at the left end of the plot (shaded), where the effect is all or nothing: it depends only on outstanding jobs against workers.

**E3: dataflow against colour barriers** (1,080 runs at 20 ms and 15 at other time scales; every check of correctness passed, and the 11 runs with a late job were run again). 16 blocks of 8 vertices on 4 ranks, T 8 and R 4 (1,024 jobs). The grid and the ring have two colours of 8 blocks, the random block graph four colours of 6, 4, 4 and 2, the complete one 16 colours of one block. C-sync time over C time, measured (model), medians over the seeds:

| Shape | Places | CV 0 | CV 0.5 | CV 1 | CV 2 |
|---|---|---|---|---|---|
| grid | 4 | 1.001 (1.000) | 1.013 (1.014) | 1.026 (1.028) | 1.099 (1.093) |
| grid | 8 | 1.002 (1.000) | 1.055 (1.044) | 1.131 (1.161) | 1.229 (1.237) |
| grid | 16 | 1.001 (1.000) | 1.044 (1.046) | 1.107 (1.112) | 1.172 (1.179) |
| ring | 8 | 1.003 (1.000) | 1.065 (1.061) | 1.157 (1.161) | 1.231 (1.250) |
| ring | 16 | 1.002 (1.000) | 1.086 (1.088) | 1.127 (1.132) | 1.179 (1.181) |
| random | 4 | 1.005 (1.000) | 1.057 (1.057) | 1.116 (1.129) | 1.225 (1.228) |
| random | 8 | 1.048 (1.043) | 1.091 (1.086) | 1.155 (1.160) | 1.247 (1.245) |
| random | 16 | 1.006 (1.000) | 1.082 (1.083) | 1.157 (1.156) | 1.245 (1.245) |
| complete | 2 | 1.000 (1.000) | 1.000 (1.000) | 1.003 (1.000) | 1.000 (1.000) |
| complete | 16 | 1.000 (1.000) | 0.999 (1.000) | 1.000 (1.000) | 1.001 (1.000) |

(All 84 settings, with A, B and the times, are in `exp_qpu/tables/e3.md`; plots `e3_slots` and `e3_cv`.)

The hypothesis holds, with its null results. With the barrier the run takes up to 25% longer at CV 2 (C ends 20% sooner) and up to 16% longer at CV 1, and no longer when every job takes the same time, with one place (1.000 everywhere), or on the complete block graph, where the barrier is the dependency itself. The one gain at CV 0, 1.048 on the random graph at 8 places, is in the model too (1.043): its colours differ in size, so a barrier leaves places empty even when all jobs are equal. Over the 84 settings the measured ratio is within 1% of the model in 78 and within 2.6% in all. On the grid and the ring at CV 1 and 2 the gain is largest at 8 places, one per block of a colour, and not at twice that, where section 6.4 expected it; the model says the same.

Where measurement and model differ most (grid at CV 1 and ring at CV 2, both at 8 places), it is the model of the barrier systems that is off, not C. The model starts the blocks of a colour in block order; B submits them in the order of its ranks (0, 8, 2, 5, ... on the grid) and C-sync in the order its tasks become ready, as the job logs show, and with fewer places than tasks that order decides which blocks finish first. Over all settings B is within 2.5% of its model either way, C-sync from 2.2% below to 3.8% above, and C from 0.2% below to 3.6% above, with a median of 1.0% above (the model has no classical work and no overhead).

B against C-sync, the same schedule by hand and by the runtime: B / C-sync is between 0.973 and 1.011 in every setting. B takes more than 1% less time in 27 of the 84 settings, 23 of which have 16 or 32 places or are on the complete graph, whose colours are one block each. With a place for every task, a run lasts as long as the chain of steps of its slowest block, and every step adds its turnaround. The gap is largest at CV 0, 2.4 to 2.7% at 16 and 32 places (1.29 s against 1.33 s on the grid). E4 shows where the time goes.

The ratios do not depend on the time scale. The grid at CV 1 and 16 places was run again with jobs of 0.1 s (seeds 1 to 3) and 1 s (seeds 1 and 2). With the same seed the job times are the same up to the scale, so the ratios should be equal, and they are:

| Mean job time | C-sync / C, seed 1 | seed 2 | C over the model |
|---|---|---|---|
| 20 ms | 1.137 | 1.014 | 1.7% |
| 0.1 s | 1.142 | 1.013 | 0.4% |
| 1 s | 1.143 | 1.013 | 0.04% |
| Model | 1.143 | 1.013 | |

What C adds over the model is about 45 ms per run at every scale (41 to 48 ms) (the classical work and the turnaround of E4), so its share falls as jobs get longer. The medians over all seeds of each scale (1.107, 1.111 and 1.077) differ only because the scales have different seeds (`exp_qpu/tables/e3_scales.md`).

**Late jobs.** 11 of the 1,080 runs had one late job out of 1,024: 10 of the 296 runs at CV 2, one of the 296 at CV 0.5, none at CV 0 or 1. Section 6.7 does not count such a run. Each was run once more with the same command (`exp_qpu/retry_late.py`), all 11 without a late job, and the second run takes the place of the first, so every setting keeps its seeds. The programs were the same: the provenance records of the two differ only in the harness and in one line of the test list (`bench:work`). At CV 2 the shortest drawn jobs are 157 to 235 us, three to four times the 55 us of sampling a block of 8, so a helper thread that waits a tenth of a millisecond for a core ends one late. 6 of the 11 fell in the windows when another user's processes ran, which hold a third of the runs. The mock counts late jobs and not by how much, so the cause of each was not measured.

**E4: how full the places stay** (from the per-job logs of the E3 runs of B, C-sync and C). Busy place-time over places times the run's time, medians over the seeds, with the model; the grid:

| CV | Places | B | C-sync | C | Model, barriers | Model, dataflow |
|---|---|---|---|---|---|---|
| 0 | 16 | 0.993 | 0.966 | 0.967 | 1.000 | 1.000 |
| 1 | 4 | 0.944 | 0.953 | 0.993 | 0.956 | 0.997 |
| 1 | 8 | 0.812 | 0.809 | 0.915 | 0.800 | 0.924 |
| 2 | 4 | 0.861 | 0.865 | 0.973 | 0.869 | 0.975 |
| 2 | 8 | 0.620 | 0.619 | 0.729 | 0.620 | 0.735 |
| 2 | 16 | 0.334 | 0.332 | 0.374 | 0.335 | 0.379 |

(Every setting is in `exp_qpu/tables/e4.md`, with the time a place was free while a task waited; the plot `e4_series` shows the jobs outstanding over time in one run of each system, grid, CV 1, 4 places.)

At 4 places, where section 6.4 put the model at 0.97 to 1.00 for dataflow and 0.86 to 0.98 for barriers, the grid and the ring measure 0.973 to 0.996 for C and 0.861 to 0.996 for B and C-sync. The random block graph is lower for both (at CV 2, C 0.851 and C-sync 0.718), and so is its model (0.851 and 0.721). The gain of E3 is a gain in busy places. Every system is at most 1.2 points above its model, and at most 2 points below it in 243 of 252 cases. Six of the 9 others are C and C-sync at CV 0 with a place for every task: the grid and the ring at 16 places (3.3 to 3.6 points below) and the complete graph at 2 (2.8). The last three are 2.0 to 2.3 points below: C and C-sync on the random graph at CV 0 with 8 places, and C-sync on the ring at CV 1 with 4.

At CV 0 with 16 places a place was hardly ever free while a task waited (7 ms over a run of 1.3 s on the grid): the places waited for the next step. The logs give the turnaround of a step on the critical path (medians over three seeds; grid, ring and complete at CV 0). C takes 480 to 540 us from retrieving a result at rank 0 to the next step's job reaching the device, of which 360 to 390 us pass before `EVAL` begins on the block's rank. B takes 16 to 25 us from sending a result to receiving the next request, with its ranks spinning. On the 64 steps of the grid's critical path that is 31 to 35 ms of the 44 ms C takes over its model. How the 0.37 ms divides between the remote dependency and waking a sleeping worker was not measured. It is the one place where the runtime costs more than hand-written MPI, and it matters only for jobs of tens of milliseconds (1.7% of the time at 20 ms, 0.04% at 1 s).

With one place the order reverses. A place is free while a task waits for 53 to 55 us per job in C and C-sync and 117 to 126 us in B (grid, ring and random): the time from a job's end to the next submission. B's helper thread retrieves a result 65 to 127 us after the job ends, C's progress thread 47 to 61 us, which makes B 0.3% slower than C with one place (20.61 s against 20.54 s on the grid at CV 0).

**E5: tasks that share jobs** (205 runs, all passed). Time with every task its own job over time with the limit, measured (model, as a range over the two ways tasks arrive at the device):

| Regime | Limit 2 | 4 | 8 | 16 | Section 6.1 |
|---|---|---|---|---|---|
| 1 place | 1.74 (1.73 to 1.74) | 2.75 (2.74) | 3.79 (3.82) | 3.88 (3.88) | 1.73, 2.74, 3.82, 3.88 |
| 1 place, queue delay | 1.94 (1.92 to 1.99) | 3.72 (3.62 to 3.68) | 6.50 (6.60 to 6.74) | 6.52 (6.93 to 6.95) | 1.93, 3.68, 6.38, 6.57 |
| Submission interval | 2.00 (2.00) | 4.01 (3.99) | 8.03 (7.97) | 15.80 (15.83) | 2.00, 3.99, 7.97, 15.8 |
| Interval, 3 places, queue delay | 2.00 (2.00) | 3.97 (3.97 to 3.98) | 6.79 (6.52 to 6.84) | 7.58 (7.32 to 7.78) | 2.00, 3.94, 6.80, 7.01 |
| 3 places | 1.73 (1.73 to 1.74) | 2.46 (2.64 to 2.66) | 2.25 (2.30 to 2.37) | 2.16 (1.75 to 2.29) | 1.73, 2.66, 2.24, 1.79 |
| 64 places | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

With one place or a submission interval, the measurement is the model within 2% (the queue delay rows within 6%: five draws of an exponential per setting). The non-monotone row of section 6.1 is there: with three places, a limit of 4 beats 16 in the model, and the device peaks at 4 too, but 7% lower than the model (2.46 against 2.64). The reason is in the logs: at a limit of 4 the device made 288 to 334 jobs where the model makes 272 or 273, more of them of one to three tasks. The released tasks reach the device over tens of microseconds, after their `EVAL`, `STEP` and `QEXEC` ran, and a place that is free takes what waits at once and never holds it back to fill a job; the model takes in the tasks of one event at the same instant. A third variant of the model, where all tasks an event releases arrive together, gives 2.73 and moves away from the measurement, so the gap is the trickle and not the rule. With jobs of 0.1 s the one-place row is 1.74, 2.75, 3.80, 3.88: the ratios do not depend on the time scale. With unlimited places sharing changes nothing, as expected.

Every row of the second table of section 6.1, for the share of a job that does not depend on what it carries:

| T_fixed / T_circ | Places | Limit 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 0 | 1 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| 1 | 1 | 1.34 (1.33) | 1.60 (1.60) | 1.77 (1.77) | 1.78 (1.78) |
| 5.6 | 1 | 1.74 (1.73) | 2.75 (2.74) | 3.79 (3.82) | 3.88 (3.88) |
| 20 | 1 | 1.91 (1.91) | 3.50 (3.49) | 5.80 (5.84) | 5.98 (6.00) |
| 0 | 3 | 1.00 (1.00) | 1.00 (1.00) | 0.99 (0.90 to 0.99) | 0.98 (0.82 to 1.00) |
| 1 | 3 | 1.34 (1.33) | 1.52 (1.57 to 1.58) | 1.39 (1.16 to 1.48) | 1.37 (1.06 to 1.45) |
| 5.6 | 3 | 1.73 (1.73 to 1.74) | 2.46 (2.64 to 2.66) | 2.25 (2.30 to 2.37) | 2.16 (1.75 to 2.29) |
| 20 | 3 | 1.90 (1.90 to 1.91) | 2.87 (3.34 to 3.36) | 2.90 (3.24 to 3.29) | 2.72 (2.90 to 3.16) |

Sharing a job gains in proportion to the share of a job that is fixed, and nothing when there is none: with T_fixed = 0 a job of n circuits takes as long as n jobs of one. With one place the device is the model to 1%. With three places it is below the model at limits of 4 and 8 for the reason above, by up to 14% when the fixed share is large.

**E6: spreading the graph over ranks** (120 runs, all passed). Rings of 4 to 64 blocks of 12 vertices (48 to 768 vertices), jobs of 0.2 s, a place for every job, so the critical path is 32 jobs in a row and the model gives 6.40 s for every size:

| Blocks (vertices) | System | 1 rank, s | 2 ranks | 4 ranks | 8 ranks | Processor time, 1 / 8 ranks, s |
|---|---|---|---|---|---|---|
| 4 (48) | B | 6.407 | 6.407 | 6.407 | 6.408 | 6.5 / 51.3 |
| 4 (48) | C | 6.413 | 6.423 | 6.428 | 6.425 | 0.4 / 1.5 |
| 16 (192) | B | 6.421 | 6.420 | 6.421 | 6.418 | 6.7 / 51.7 |
| 16 (192) | C | 6.416 | 6.430 | 6.437 | 6.439 | 0.7 / 6.8 |
| 64 (768) | B | 6.443 | 6.457 | 6.466 | 6.451 | 7.5 / 52.9 |
| 64 (768) | C | 6.429 | 6.448 | 6.456 | 6.465 | 1.5 / 7.9 |

(8 and 32 blocks lie between; `exp_qpu/tables/e6.md`.) The lines are flat, as expected: from 1 to 8 ranks C adds 12 to 36 ms over 6.4 s, at most 1.1 ms per job on the critical path, and B adds at most 23 ms. Both stay within 1% of the model at every size. Part of what grows with the number of blocks is there on one rank too (from 4 to 64 blocks, C goes from 13 to 29 ms over the model on one rank and B from 7 to 43 ms): most likely the work of taking in up to 128 results that end at the same moment, which was not measured apart. B holds a core per rank for the whole run (51 to 53 s of processor on 8 ranks); C uses 1.5 to 8 s. This bounds what the runtime adds per rank through shared memory; it says nothing about a network (departure 4) and is not a scaling result.

### Verification

All runs with the code as it stands at the end of M8 (`8cb244735` and the M8 changes, committed since as `f0d3a53ae` to `e77de8362`), on 2026-11-24, on histamine2 (AMD EPYC 7302, 16 cores, one socket, an exclusive Slurm allocation) unless said otherwise: methane was running the experiments.

| Check | Result |
|---|---|
| The modified tree with the option off, full suite | 168 of 168 pass, as after M7 |
| The modified tree with the option on, full suite | 298 of 298 pass: the 277 of M7 and 21 new ones |
| Every QPU and QAOA test 100 times in a row | 13,100 of 13,100 pass (131 tests) |
| QPU and QAOA tests in a profiling build and in a Release build | 131 of 131 each |
| The same in a build without MPI, under the address sanitizer | 96 of 96 (the 35 others need MPI) |
| MPI by hand, two or four cores per rank, on 2, 3, 4 and 8 ranks: B, C-sync and C on every instance, with failing evaluations, the QPU on the last rank, more ranks than blocks, the three broadcast trees, the 11 schedulers, two virtual processes, one worker | 123 loops, 361 runs, none failed. (A 124th loop is the control of section 9.6: the existing `dsl/ptg/startup` under the scheduler `llp`, which fails 3 times of 3 as before) |
| Compiler warnings in the new and changed sources, and warnings of the PTG compiler on them, in six builds from scratch | None |
| **leconte, CUDA build with the option on**, full suite, one V100 per GPU test | 390 of 390 pass: the 66 GPU tests one at a time and the 324 others. The application with the CUDA device enabled, on 1 and 3 ranks, passes |
| `runtime/qpu/bench:work`, added after the mutations below (so not in the rows above) | 20 of 20 in the Debug build and 20 of 20 in the Release build; the run takes 21.4 to 22.3 ms against the 20 ms its work cannot take less than |

Not done in M8: a multi-node run, and a run on a real device, as before.

### Does a test notice

One deliberate bug at a time in a private copy of the sources, rebuilt, and the 129 QPU and QAOA tests of that copy run (the two `qpu_fallback` tests are not in that set). 17 of 18 were caught at first; the 18th was caught once a test was added for it.

| Deliberate bug | Tests that fail |
|---|---|
| **B** | |
| A job named by the key of the next step; by a key without the round | 6 each: every test of B |
| The two evaluations of a step swapped when the optimizer takes them in | 6 |
| A step that should be skipped submits its jobs anyway | 1: `apps/qaoa/mpi:rand`, the one instance where steps are skipped |
| Each rank keeps its own states and never takes the others' | 5: every test of B on more than one rank |
| Each block stops one step short | 6 |
| The helper thread keeps twice `max_outstanding` jobs at the mock | 1: `mpi:1slot` |
| **C-sync** | |
| The first task of a colour does not wait for the barrier | 4 |
| The barrier's tree leaves out its left subtrees | 4 |
| An update counted for the next colour | 6: every test of C-sync |
| **The mock's seed of the times** | |
| The job times drawn from the seed of the samples; a circuit's time drawn with the bits of the queue delay; the samples drawn from the seed of the times | 1 each: `runtime/qpu/mock` |
| **The shapes of block graphs** | |
| A ring's pairs reversed; a grid whose rows wrap onto the next; a random graph of three neighbours instead of four | 1 each: `apps/qaoa/math` (the instances pinned by digest) |
| **The benchmark** | |
| The blocking body does not hold its worker | 2: `bench:block`, `bench:sequential` |
| The classical work of a link is not done | None at first; 1 (`bench:work`) once it was added |

The last bug taught something. Every test of the benchmark gave each link 20 us of work, less than what the runtime and the mock add per link, so its check that a run took at least as long as its work could not fail. `bench:work` gives each link 2 ms of work, longer than anything else in the run; with it the copy has 130 tests, all pass before and after, and the bug fails `bench:work` alone.

### Sanitizers

The no-MPI builds of section 9.1 with gcc 12.4 (histamine2's own compiler has no sanitizer libraries), 35 runs under each: the 29 of M7, the benchmark in three modes, and C-sync with failures and on another shape of block graph.

- **Thread sanitizer.** 16,043 reports. Two have the QPU layer on both sides, both in the test that stops the service under a running taskpool: the stopping thread frees the backend while a worker submits (`device_qpu_progress.c:1215`) and while the mock's wake-up runs (`device_qpu_backend_mock.c:1144`). That is the application error the test stages, as in section 9.6; M7 had one report of it, this run two. 20 have the QPU layer on one side, all of the two kinds of section 9.1: 15 are the progress thread's push into a scheduler queue against a worker's pop, 5 an access to a task in the QPU layer (`device_qpu_progress.c:321` and `:1189`) against a later reuse of the task's memory from the pool. None is in code that M8 changed.
- **Address sanitizer.** No memory error in 35 runs. Every leak record (425) is the one of section 9.1, a copy of a flow that a PTG task only writes. Sections 9.1, 9.5 and 9.6 give 72 bytes per copy. The record holds a pointer per device, so its size depends on how many devices a run has: 80 bytes with the QPU device on (413 records here, and the logs of M7 show 80 too), 72 in the two benchmark runs without it (12 records).

### What an independent review found

An independent reviewer read the new code and the harness while the first campaign ran, and modelled the device's admission rule against the model's. It found no correctness problem in B (the order of kernel calls per block, the keys, the messages, the end of a colour and of the run, ranks without blocks) and found the `JOIN` barrier exact at every colour and round boundary. What it found was in what was timed and in the harness; the first campaign was stopped, set aside (`exp_qpu/superseded/`) and run again from scratch after the fixes.

| Finding | What was done |
|---|---|
| C and C-sync started their clock on rank 0 without a barrier, after a reference computed on every rank, and reported rank 0's time; B used a barrier and the slowest rank's time. With 16-qubit blocks (E6 as planned) the skew would have been tenths of a second and grown with ranks, in C only | A barrier before the clock in `testing_qaoa`, and the slowest rank's time |
| The C-sync barrier was a chain of `JOIN` tasks in block order, up to 7 remote hops after the last update of a colour, against one collective in B: about 30 ms over 8 barriers, charged to the runtime | A binary tree (depth 3 for 8 blocks) |
| B's ranks spin in `MPI_Waitany` while PaRSEC's idle threads sleep, so B reacts faster and burns cores | Kept (it is what B is), and the processor time of every run of every system is now reported next to its time |
| E6 with 16-qubit blocks on two cores per rank would have measured the mock's sampling (7 to 12 ms of processor per job) and not MPI | Blocks of 12 (a departure); a run with a job that ended late now counts as failed, as section 6.7 asks |
| Every run of A was recorded as failed (the reference prints no verdict in a single run) | It prints one, and the real count of late jobs |
| The runner skipped a configuration found in its CSV whatever code produced it, hashed the sources but not the programs, and a hang would have stopped a whole campaign and left ranks running | The provenance id covers the sources, the programs run and the harness; a run is done again under another id; a run that hangs is killed with its process group and recorded |
| The model gave a freed place to the tasks its end released before those that were waiting; the device does the opposite. That moved D with three places from 1.75 to 2.29 at a limit of 16 | Fixed; the model also has a variant where the two evaluations of a step arrive together, and D's prediction is drawn as the band between the two |
| Nothing failed if `-y` never reached the taskpool (the states are the same with or without the barrier); B's limits and the mock's queue and job draws were untested; a ring that changed its edges went unnoticed | The driver counts the `JOIN` tasks (one per update with `-y`, none without); B checks its limits and the mock's watchdogs; the mock's test covers all three draws; the instances are pinned by digest, generated once by the committed code and once by the new |
| B freed a send buffer before its request completed; B counted a refused job as failed twice; `qpu_bench` ignored unknown options | Fixed; `qpu_bench` also fails a run that took less time than its work and jobs allow |


### Departures from the plan

1. **Time scales.** Section 6.4 runs E3 and E5 at mean job times of 0.1 s and 1 s. The full grids ran at 20 ms (E3) and 10 ms per job (E5), which keeps classical work under half a percent of a job and makes about a thousand runs fit in hours; the ratios were then checked at 0.1 s and 1 s on one setting of E3 and at 0.1 s on one row of E5.
2. **Smaller blocks.** E3 uses blocks of 8 vertices and E6 blocks of 12, not 10 and 16. The mock samples every circuit on the machine under test, and section 6.7 counts a run only when no job outlasted its drawn time: with blocks of 10, 46 of the first 64 E3 runs at a coefficient of variation of 2 had such jobs, and none at lower ones (the shortest drawn times are 157 to 235 us over the five seeds; sampling takes 125 us at 10 qubits and 55 us at 8), and with blocks of 16 E6 would have measured the mock. The schedule does not depend on the size of a block, and a probe gave the same times within 0.4% for blocks of 10, 8 and 6, with no late job at 8 or 6. The runs on blocks of 10 are kept in `exp_qpu/superseded/`.
3. **Helpers share cores with workers.** Section 6.7 asks for helper threads with cores the workers do not use. They share the cores of rank 0; the count of late jobs, zero in every reported run, is the check instead.
4. **E6 on one node.** The compute nodes were down, so E6 ran 1 to 8 ranks on the second socket of methane: MPI through shared memory, not a network. It bounds what the runtime adds per rank, not what a network adds.
5. **E1 up to 14 workers**, not 36: it ran on serotonin (16 cores, nothing else on it) and left two cores for the progress thread and the mock.
6. **C-sync is a tree of `JOIN` tasks**, not a control gather (section 6.2): a gather over a named range counts its inputs wrongly on this base (fork draft PR #19).
7. **B waits by spinning** in `MPI_Waitany`, which is what a first MPI version does; its processor time is reported next to its time.
8. **E4 has no runs of its own**: it is read from the per-job timestamps of E3, which every run of B, C-sync and C writes.
9. **Not done**: B-async (section 6.2, "if time allows"), a replayed hardware trace in E3 (it needs E7), and E8 (optional).
10. **The component changed in M8**: `device_qpu_mock_delay_seed` and `parsec_qpu_mock_time()`, so that the job times can vary while the answers stay the same, and the model can be checked against the mock's own draws.
11. **The model** has three ways for released tasks to reach the device (one at a time, a step's two evaluations together, all at once); D's prediction is drawn as the range of the first two, which bracket the measurement with one place and not with three (E5).
12. **Runs with a late job were run once more.** Section 6.7 only says such a run is not reported. The 11 of E3 were run again with the same command and replaced by the second run, so that every setting keeps its seeds; the first runs stay in the CSV, marked.
13. **The places at which dataflow gains most.** Section 6.4 expected the gain of E3 at places "at or above twice the blocks per colour"; on the grid and the ring at CV 1 and 2 it is largest at one place per block of a colour, in the model as in the measurement.

## 9.8 M9: several QPUs per context (2026-11-26)

Approved with "Approve, start M9" after the design revision of section 10. Built on `80d6aa45d`, and verified before it was committed.

### What exists now

- **`parsec/mca/device/qpu/device_qpu_list.c`** (new): the list of the QPUs of a run, `device_qpu_devices`.
- **`device_qpu_component.c`, `device_qpu_module.c`**: one module per QPU of the list. A rank attaches the modules of the QPUs it owns, each with its backend; a rank that owns none attaches the first one as a placeholder.
- **`device_qpu_progress.c`** (rewritten around two structures): a service per QPU (backend, progress thread, jobs in flight, limits, counters) and a pool per rank (the submission stack and the tasks that wait, under one lock).
- **`device_qpu.h`**:
  - `parsec_qpu_submit_on()`, `parsec_qpu_submit_dtd_on()` and `PARSEC_QPU_SUBMIT_ON()` name a QPU, and `PARSEC_QPU_ANY` names none;
  - `parsec_qpu_count()` and `parsec_qpu_rank_of()`;
  - the result header says which QPU ran the job (`qpu`), and its version is 2.
- **`device_qpu_backend_mock.c`**: each mock reads its own options from its entry, and its counters can be read per QPU. A new option, `submit_delay` (`device_qpu_mock_submit_delay`), holds the thread that submits for a given time, as a provider's round trip does.
- **Tests**: `tests/runtime/qpu/qpu_multi.jdf` (new), run by 27 of the 28 tests named `runtime/qpu/multi*`; two more tests of the hardware guard; ten of the QAOA application; `repeat_test.cmake`, which runs a test several times; and the guard script reads the lists of the test definitions.
- **`tests/apps/qaoa`**: the quantum tasks of block `b` run on the rank `Q(b mod NQ)`, and `-Q` makes every block name a QPU.
- **Nothing outside the component and its tests changed.** `parsec/mca/device/device.c` and `parsec/scheduling.c` are as M3 left them.

### How a task gets its QPU, as built

Section 10.3 is what was built, with these additions.

- **The load.** `parsec_qpu_submit_on()` takes the task's load off the QPU the device layer selected and sets `task->load` to 0. The QPU that takes the task (`qpu_bind`) makes itself `task->selected_device`, sets the load and adds it. A task that goes back to wait gives it back (`qpu_unbind`). This is done only for a task whose body is of the QPU type. A quantum task with a CPU body (option C of section 1.9) keeps the CPU device and no load, as before.
- **A job that failed is tried again on another QPU**, when one that is up can run the task. Without that rule a QPU whose jobs fail at once is free first every time, and takes the same task again until its attempts are used up. The task remembers the QPU it failed on (`avoid`), and that QPU takes it again only when nobody else can.
- **A QPU that is given up on is tried again.** It is left alone for its `unavailable_timeout`, then takes tasks again, and its first job tells whether it is back. If that job is not taken either, its tasks go back to the pool at once and the QPU is left alone for twice as long, up to eight times the time-out. With one QPU none of this shows: the one QPU takes every task, as before.
- **The tasks of a submission that is given up on** go to another QPU when one that is up can run them. Section 10.3 had them complete as unavailable, which contradicted its own test ("every unpinned task still completes with a result").
- **The tag of a job** counts the jobs that were put together for its first task, whether a backend took them or not. Section 2.5 counted the attempts a backend took. With one QPU the two are the same number. With several, a task whose submission is given up on is asked for again without an attempt having been counted, and would have had the same tag twice (the review found it, below).
- **The time a job is asked for** (`t_submit`) is read after its tasks are taken from the pool. Read before, as with one QPU, it could be earlier than the moment the body submitted the task, when another QPU's thread had put the task into the pool in between. The new test found it under the thread sanitizer.

### The wake-up rule, and what two versions of it got wrong

A thread sleeps in its backend, and somebody has to wake it when there is a task for its QPU. Section 10.3 gave a rule in two sentences. It took three versions to get a rule that holds.

**First version: the others are woken at the end of a thread's pass,** after it has taken tasks from the pool and submitted its own jobs.

- **What happened.** `runtime/qpu/multi:noidle` failed in 3 or 4 runs of 20 to 30. For about 8 ms at the start of a run one QPU held a single job and another none, while 45 tasks waited.
- **Why.** A thread had put 48 tasks into the pool and was then held up for 7 to 14 ms inside its first call to its backend (measured with a probe around the call, in 3 of 25 runs). The other threads had looked at the pool just before the tasks were there, and slept. Nobody woke them until that call returned.
- **What a real backend would make of it.** That call is a round trip to a provider, so the same wait would be the rule, not the exception.

**Second version: whoever puts tasks into the pool wakes the others before it calls its backend again.** 0 of 60 runs of `multi:noidle` then showed a free place. The review found what was still wrong (finding 2 below): a QPU showed its places before it looked for work, and its places could rise later in the same pass, when a submission that was being asked again was taken, or a job was cancelled for taking too long. A body that had read "no place" a moment earlier woke nobody, and the thread went to sleep beside a free place with that body's task still on the stack. With one QPU, M8 had not had this, because there every body woke the thread.

**The rule as built.**

- **A QPU shows its places only while its thread sleeps or is about to.** A thread at work may be inside a call to its backend, so a task that comes then is for a QPU whose thread sleeps.
- **A body** pushes its task, then reads the places the QPUs show, and wakes one that shows a place; none if none does.
- **A thread that puts tasks into the pool** wakes the others that show a place, before it calls its backend again. It does so when it drained the stack and holds more tasks than it has places, when a task goes back to wait, and before each submission.
- **A thread that is done** shows its places first, and then looks once more at the stack and at the pool before it sleeps.
- **Why nothing is left behind.** Whoever comes with a task either sees the places and wakes the thread, or came before they were shown and is found by that last look.
- **Why threads do not keep each other awake.** Nobody is woken for a task it cannot take, and the last look goes around again only when there is a task the QPU can submit at once.

**What tests it.**

- **`multi:noidle`** sees the first defect only when a thread happens to be held up, about one run in eight.
- **`multi:fail:prompt`** is for the wake-up by a thread that puts a task back: a QPU whose jobs all fail, two chains, and threads that sleep for 30 s when nobody wakes them. A task whose job failed has to go to the other QPU at once, and the run has to be done in 5 s; it takes a fraction of one. It is run 10 times, because whether the last task goes to the failing QPU first is a matter of turns.
- **`multi:wake:reask`** is the reviewer's case, steered through the mock: the mock does not take the first job, and while it is asked again it holds the thread until a second task has been submitted. That task found no place shown and woke nobody. Its job is asked for 0.2 ms later; with neither look after the backend call, it waits the two seconds that the first job takes. The test allows half the time of the first job. It had allowed 300 ms, and failed once, in the clean run of a planted-bug pass with other work on the machine; the output of that run was not kept. It then passed 88 runs alone or eight at a time and six runs of the whole set, and was given the wider margin.
- **Two stricter tests were tried and dropped.** With a mock that takes 20 ms to answer a submission, QPUs of one place and not one moment allowed to show a free place, a burst and chains were run 20 and 10 times each. They passed 5 of 5 times alone (150 runs) and the first failed once in the full suite, with another build going on on the machine. Neither caught a planted bug that the tests above do not. The mock's new option and the strict mode of the test (`-S`) stay.


Why the mock's call takes 7 to 14 ms now and then, without the new option, was not found. The mock holds its lock only for short sections, so the likely cause is the thread waiting for a core. It is not M9: a single QPU has the same wait, where it only delays that QPU's own job.

### Verification

Debug build with `PARSEC_DEBUG_PARANOID` and MPI (`/tmp/qcao3_qpu/build_qpu`, gcc 11.5, Open MPI 5.0.5), on methane.

- **Build.** No new warning in the component or in the tests.
- **The whole suite: 339 of 339 pass** (`ctest -j 6`, 80 s), twice. 172 of them are the QPU and QAOA tests: the 132 there were, and 40 new ones.
- **Every test there was passes with the list left out**, so one QPU behaves as before. One of them had to change, `runtime/qpu/overlap` (below).
- **Not run, when M9 was committed:**
  - a Release build, a build with profiling, and a build without the component;
  - more than one node;
  - CUDA.

  All were run the day after: see "The CUDA run, and what it led to" and "The checks that were left, and the experiments on several QPUs" below.

### The tests

| Test | What it checks |
|---|---|
| `runtime/qpu/multi`, `:three`, `:enabled` | Two and three QPUs of one rank, from a list and from `device_qpu_enabled 3`. Every answer is the sampler's for its inputs and key, whichever QPU gave it. No QPU goes over its own limit, nor its mock over its slots. At the end nothing waits, no QPU carries a load, each task was counted by one QPU device, and each QPU's answers are as many as its device counted |
| `:named`, `:named:half`, `:named:one` | Tasks that name their QPU, all of them, every other chain, and all the same one: each answer is from the QPU its task named |
| `:batch` | Jobs that carry several tasks, on two QPUs of one place |
| `:limits` | One QPU of one place and one of three: each has the limit of its entry, and reaches it |
| `:noidle` | No place stays free while a task waits: 48 tasks at once on three QPUs of two places, job times drawn from an exponential law. Read from the times in the results: at the middle of the time a task waited, every QPU held as many jobs as it may |
| `:wake:reask` | A task submitted while its QPU is asked again for a job, steered through the mock: its own job is asked for within half the time of the first job (it takes 0.2 ms), and not when the first job ends |
| `:speeds` | A QPU twice as fast gives more than half of the answers |
| `:width`, `:width:none` | Circuits of four qubits and a QPU of three: the other one runs them all; with two QPUs too narrow, every task completes as rejected, without a job |
| `:fail`, `:fail:named`, `:fail:prompt` | A QPU whose jobs all fail: every task that named none has its answer from the other one, and a task that named it ends without one. And at once: with threads that sleep for 30 s unless woken, ten runs of two chains each take a fraction of a second |
| `:down`, `:down:named`, `:down:away` | A QPU that never takes a job: the same; and in a run of two seconds its backend is asked 11 times, where a QPU that is handed every task in turn would be asked about 200 times (the test allows 40) |
| `:back` | A QPU that takes no job for 300 ms: its tasks go to the other one, and at the end it is up and has run jobs again |
| `:lost`, `:lost:noretry` | A QPU that takes its jobs and loses the answer for 400 ms: the submission is given up on, and the task is asked for again, there or on the other QPU, under another tag. No mock sees a tag twice, and no task loses an attempt to it |
| `:noopen` | A QPU whose backend cannot be opened has no device, and the other one runs the tasks |
| `:badlist` | A list with a misspelt key gives no QPU at all |
| `:mp`, `:mp:named`, `:mp:4`, `:mp:fail` | QPUs on two ranks, one and two of them, on 2 and 4 ranks; each QPU is on the rank the list says; a rank that owns none has the placeholder and nothing behind it |
| `runtime/qpu/hardware_guard:list`, `:list:unknown` | In a list, one backend that drives a quantum device, or that does not exist, takes the others with it: the mock listed first is not opened |
| `runtime/qpu/no_hardware_in_tests` | Also reads every `device_qpu_devices` of the test definitions: no backend but the mock and the guard test's own |
| `apps/qaoa:qpus`, `:named`, `:sync`, `:fail`, `:fail:named`, `:grid` | The application on two and three QPUs of one rank, with and without `-Q`: final states equal byte for byte to the reference |
| `apps/qaoa:qpus:mp:2`, `:mp:4`, `:sync:mp:3`, `:fail:mp` | The same with QPUs on two ranks, on 2 to 4 ranks. The counts of every rank that owns a QPU are added up and compared with the reference |

One existing test changed. `runtime/qpu/overlap` checked that the QPU device carried the load of its one task from the moment the body returned. The load is now on a QPU from the moment that QPU takes the task. The test reads the times in the result and expects no load before the job is asked for, and one task's worth once it runs.

### Does a test notice

One deliberate bug at a time in a private copy of the sources, rebuilt, and the 170 QPU and QAOA tests of that copy run. 22 bugs: 18 fail a test, one more is seen by the thread sanitizer, and three are seen by nothing. The bugs were run in three passes, and tests were added and two dropped between them; each count is that of the pass the bug was last run in.

| Deliberate bug | Tests that fail |
|---|---|
| **The load of a task** | |
| The load stays on the QPU the device layer picked | 139: nearly every test that runs a quantum task |
| The load is not taken back when a task goes back to wait | 31: every test with a job that fails |
| **Which QPU runs a task** | |
| A task forgets the QPU it named | 7: every test with named QPUs |
| Every circuit fits every QPU | 3: `failure:rejected`, `multi:width`, `multi:width:none` |
| A job that failed is tried again on the same QPU | 2: `multi:fail`, `multi:mp:fail` |
| A result names the first QPU of its rank | 22: every test with more than one QPU on a rank |
| **A QPU that takes no job** | |
| The tasks of a submission that is given up on are not moved | 5: `multi:down`, `:down:away`, `:back`, `:lost`, `:lost:noretry` |
| A QPU that is down takes tasks like the others | 1: `multi:down:away` |
| A QPU that is down is never tried again | 3: `multi:back`, `multi:lost`, `multi:lost:noretry` |
| The tag of a job counts attempts, as before | 2: `multi:lost`, `multi:lost:noretry` |
| **The list** | |
| The rank of an entry is ignored | 2: `multi:mp`, `multi:mp:4` |
| The limit of an entry is ignored | 2: `multi:batch`, `multi:limits` |
| The mock ignores the faults of its entry; its width | 9; 2 |
| A misspelt key is accepted | 1: `multi:badlist` |
| **The hardware guard** | |
| Only the first entry of a list is checked | 2: `hardware_guard:list`, `:list:unknown` |
| **Who wakes whom** | |
| A thread that puts tasks into the pool wakes nobody | 1: `multi:fail:prompt` |
| Neither look after a backend call (in the pass, and before sleeping) | 1: `multi:wake:reask` |
| The pool is read without its lock when a job is put together | None of the tests. Under the thread sanitizer, three runs gave 72 reports with the QPU component in both stacks, in the heap of the tasks that wait, and one of the runs aborted |
| The others are woken only after the backend call, not before | None |
| The look before sleeping is left out, the one in the pass kept | None |
| A QPU shows its places while its thread is at work | None |

The three that nothing sees are parts of the wake-up rule that the other parts cover for in every case these tests create.

- **Woken after the call, not before.** The task bodies wake the QPUs that sleep, and a thread looks at the pool again before it sleeps. What is lost is one round trip to the backend, in the one case where a thread holds tasks that only another QPU can run and has jobs of its own to submit. No test has QPUs that differ and tasks for each at once.
- **The look before sleeping.** With the look inside the pass kept, a task is missed only when it comes in the few instructions between the two. No test can place a task there.
- **Places shown while at work.** A body then wakes a thread that is inside its backend where it could have woken one that sleeps. That costs time, and no test measures it with a backend that is slow.

The first pass, on 18 bugs, missed both wake-up bugs it had. `multi:fail:prompt` and `multi:wake:reask` were written for them; and thinking about what a planted bug would need to be caught found three checks the test did not have: where the list puts each QPU, that a QPU has the limit of its entry and reaches it, and how often a QPU that is down is asked.

### Sanitizers

The no-MPI builds of section 9.1 (gcc 12.4), 32 runs under each, on the code as built: 18 of the multi-QPU test, 9 of one QPU through the code that was rewritten, 5 of the application. The two tests written after these runs, `multi:fail:prompt` and `multi:wake:reask`, were not run under the sanitizers.

- **Address sanitizer.** No memory error. 272 leak records, all the one of section 9.1 (a copy of a flow that a PTG task only writes).
- **Thread sanitizer.** 9,795 reports. Three have the QPU component in both access stacks, and 129 have it in one. None is in the code of the pool: the heaps, the tasks that wait, the places a QPU shows.
  - **The three with both** are one report three times: two QPU progress threads give a task back to the scheduler at the same time, into the same queue (`hbbuffer.c:63`). The queue's push reads a slot and then takes it with a compare-and-swap, so two pushers cannot both take it. It is the report of section 9.1 about one progress thread and a worker, now between two progress threads.
  - **Of the 129:**
    - 28 are that same push against a worker's pop;
    - 75 are an access to a task in the QPU layer against a later reuse of the task's memory from its pool. That is the second kind of section 9.1, at more lines: `qpu_bind` now writes the task's device and load;
    - 26 are new here, and are the device layer's own habit: `parsec_select_best_device` reads `device_load` without synchronisation (`device.c:293`), and so does the assertion at `scheduling.c:541`, while the QPU layer moves a load with an atomic add. The scheduler's own atomic adds meet those reads in the same way for GPUs.
- **Two runs of the 32 did not pass under either sanitizer, neither because of it:** the script gave the test faults without telling it which to expect.
- **The first pass**, before the review's fixes, had found one defect through a test that failed only under the thread sanitizer: the time stamp of "How a task gets its QPU".

### What an independent review found

An independent reviewer read the change while the first deliberate-bug pass ran, and ran the test program against the build. It found four defects, one of them by a run, and one weakness of a test. All are fixed, and each fix has a test or a deliberate bug where one can be written.

| Finding | What was done |
|---|---|
| **A job was asked for twice under one tag** (reproduced: "4 tags used twice", and with no retry allowed, 3 tasks ended as failed beside a QPU that worked). A QPU took a job and lost the answer; the submission was given up on and the task moved; the same QPU, tried again, asked for the task under the tag it had used, since no attempt had been counted. The mock refuses a tag it has seen, which cost the task an attempt | The tag counts the jobs put together for the task. `multi:lost` and `multi:lost:noretry` run the reviewer's case |
| **A task could be left on the stack beside a free place** (by reading): the second version of the wake-up rule, above. With one QPU this was a step back from M8 | The rule as built. `multi:wake:reask` is the reviewer's case |
| **A task's record was read after it was put into the pool** (by reading): `qpu_wait_again` read which QPUs may run the task after it had released the lock, when another QPU may have run the task and freed the record | Read before |
| **A task could be skipped when a QPU is given up on** (by reading): the clean-up took tasks out of the middle of the heap while it walked it. A skipped task would have waited for the QPU that is down to look again, then for its time-out. It needs backends that differ | The heap is made again from the tasks that stay. I had found the same on my own by then |
| **`multi:noidle` is weaker than it sounds**: it counts tasks where it should count jobs, looks at one moment per task, skips waits under 10 ms and lets one moment in ten be off | It refuses to run when a job may carry several tasks. `-S` (no moment off) and the two `wake` tests are the strict form |
| **The design said a list that gives a rank more than 16 devices disables the layer**, and nothing did | A QPU that would be the 17th device of its process is left out with a warning; and a QPU whose device cannot be added is taken out of the service again |

The reviewer found nothing in: the load of a task on every path (retry, give-up, rejected, empty, a body that is not of the QPU type, stop); the other reads without the lock; how jobs are put together; start, stop, a second context and the statistics after a stop; the list and the modules; the mock's registry; and one QPU against M8, apart from the second finding. It found no check that cannot fail, and the collective calls of the tests are the same on every rank.

### The CUDA run, and what it led to (2026-11-27)

Run on leconte after M9 was committed, on the committed tree (`690eede03`): a Debug build with `PARSEC_DEBUG_PARANOID`, MPI and CUDA, on three V100 (Slurm job 68014, devices 4, 5 and 6).

- **GPU tests: 66 of 66. The other tests: 365 of 365.**
- **The application with the CUDA devices up in the same context**, one GPU per rank: its final states are the reference's with one QPU and with three, on 1 and on 3 ranks.
- **One more check failed, and it was the test.** The job also ran `runtime/qpu/cpu_gpu_qpu` with three QPUs. That is M3's test of a quantum task between two accelerator tasks, and it was written for one QPU: it asked for exactly one QPU device, and counted the tasks of the first QPU only. It found 5 tasks where it expected 16. Every body had run (32 CUDA, 16 QPU), and no check of the data failed. On methane, without CUDA, the test failed in the same way with three QPUs.
- **What that says about M9's tests.** None had a quantum task that reads what an accelerator task wrote, and is read by one, with several QPUs. That is the path on which M3 found the defect of `__parsec_execute`.
- **Reading the output found a defect of M9.** The line a QPU prints with `device_show_statistics` said "0 task(s)" for every QPU, and with one QPU as well. M8 printed the count of the service. M9 printed the count of the QPU's device, and the device layer has set that back to zero by then (`parsec_mca_device_reset_statistics`, right after it prints its own table). No test read the line.

**What was changed**, in six files, after the M9 commits:

- **The service counts the tasks of each QPU** (`tasks_executed`, for the rank and for one QPU), at the place where it gives a task back and adds it to the count of the device. The line prints that count.
- **The device check of the tests** (`qpu_test_check_device`) goes over every QPU of the rank. Each device is checked on its own (type, load, limits, what it says a job takes), the tasks are those of all of them together, and the count of the service for a QPU may not be above that of its device.
- **`cpu_gpu_qpu -q N`** says how many QPU devices the context must have; without it, one at least.
- **Three tests.** `runtime/qpu/cpu_gpu_qpu:qpus` has three QPUs, and CPU bodies where there is no CUDA device. `runtime/qpu/cpu_gpu_qpu:gpu:qpus` has three QPUs and CUDA. `runtime/qpu/cpu_gpu_qpu:statistics` reads the line of a run of 16 tasks on one QPU. The two tests there were ask for one QPU (`-q 1`), which is what they checked before.

**How the changes were verified.**

- **methane: 341 of 341** (`ctest -j 6`, 83 s): the 339, and the two new tests that need no CUDA.
- **Sanitizers**, 14 runs under each: the mixed test with one QPU and with three, four runs with many tasks through the place where a task is given back, the application on three QPUs, and three runs each of `multi:fail:prompt` and `multi:wake:reask`, which the first pass had not run. All pass.
  - **Address sanitizer:** no memory error, and the leak of section 9.1.
  - **Thread sanitizer:** 3,083 reports. One has the QPU component in both stacks, the two pushes into one queue of "Sanitizers" above. 38 have it in one, all of the three kinds listed there. None is at the new counter.
- **Deliberate bugs**, four, each caught:

  | Deliberate bug | Tests that fail |
  |---|---|
  | The line prints the count of the device | 1: `cpu_gpu_qpu:statistics` |
  | A task is not counted for its QPU | 77: every test that checks the service at its end |
  | Every task is counted for the first QPU | 1: `cpu_gpu_qpu:qpus` |
  | The device check reads the first QPU only | 1: `cpu_gpu_qpu:qpus` |

- **leconte, CUDA again** (Slurm job 68103, devices 3, 4 and 5), on the working tree:
  - `cpu_gpu_qpu:gpu:qpus` passes. So does the same run with the statistics on: 32 CUDA bodies, 16 QPU bodies, and the three QPUs ran 4, 4 and 8 tasks, which is what their lines say.
  - The other tests: 365 of 365. The application: as in the first job.
  - **GPU tests: 67 of 69.** `runtime/cuda/stress:gpu` and `runtime/cuda/ptg_pingpong:gpu` aborted, and `runtime/cuda/nvlink:gpu` was skipped. The first job had passed all three.
- **The two failures are neither about QPUs nor from these changes.** Both tests abort on an assertion of the GPU layer, `PARSEC_DATA_COHERENCY_OWNED == gpu_copy->coherency_state` in `parsec_device_kernel_pop_task` (`device_gpu.c`). A third job (68104, devices 3 to 6) built the committed head and the working tree side by side, and ran the three tests on two sets of devices, twice each:

  | Devices | Committed head | Working tree |
  |---|---|---|
  | 3, 4, 5 | two abort, one skipped, both times | the same, both times |
  | 4, 5, 6 | all three pass, both times | all three pass, both times |

  On leconte, GPU 3 cannot read the memory of GPUs 4 and 5 directly, and GPUs 4, 5 and 6 can all read each other's (`nvidia-smi topo -p2p r`). So the two tests abort when the GPUs of a run cannot all reach each other, with this branch's changes or without. On devices 4, 5 and 6 the GPU tests of the working tree pass, **69 of 69**.
- **Not done.** Why the GPU layer asserts there was not looked into, and it was not reported upstream. The mixed test runs on one GPU (`device_cuda_enabled 1`), so no run here had a quantum task between tasks on two GPUs. A Release build, a build with profiling, a build without the component and more than one node were not run by these jobs; they were run in the night that followed (next subsection).

The logs of the three jobs are with the harness, in `exp_qpu/checks/leconte_m9/`, outside version control.

### The checks that were left, and the experiments on several QPUs (2026-11-27)

Run in the night after the CUDA jobs, with the mock backend only. Nothing of it is committed. The scripts, status files and logs are with the harness, in `exp_qpu/checks/night_m9/`, and `REPORT.md` there is the longer account.

**What was built.** Every job built from a fixed copy of the working tree, so that later edits could not change what a job tests:

| Copy | Taken (UTC) | What it holds | sha256 of its difference to `690eede03` |
|---|---|---|---|
| 1 | 03:23 | The committed head and the follow-up of the CUDA run | `b287b3ed448f6146` |
| 2 | 04:23 | And the fixes of the second review, below | `ca024c5089f750f3` |
| 3 | 05:35 | And the two test changes after the repeated runs, below | `020fb15af7033233` |

The working tree as this record is written is copy 3 and one change more, made at 07:28 after everything below had run on the copies: `multi:noidle` is marked `RUN_SERIAL` (one property and its comment in `tests/runtime/qpu/Testings.cmake`). A comment was added to the header of `dtd_qpu_chain.c` at 08:41, about the second DTD defect below. The other eight changed files are those of copy 3 byte for byte, this document apart. The difference of the ten to `690eede03` has the sha256 `2d8f027d76d31385`. As it stands, methane passes the whole suite with it: 342 of 342 (Debug, 08:58). That working tree was committed later the same day, unchanged, in the three commits that follow `690eede03`: the count of each QPU, the tests, and this record.

**Build configurations, on one node.** gcc 11.5, Open MPI 5.0.5, no CUDA; the whole suite with `ctest -j 4`.

| Build | Copy 1, serotonin (job 68115) | Copy 2, caffeine | Copy 3, caffeine |
|---|---|---|---|
| Release, shared, profiling off, component on | 341 of 341 | 342 of 342 | 342 of 342 |
| Release, shared, profiling on, component on | 341 of 341 | 342 of 342 | 342 of 342 |
| Release, shared, default options (the component is off) | 168 of 168 | 168 of 168 | 168 of 168 |
| Debug, paranoid, component on | 341 of 341 | 342 of 342 | 342 of 342 |
| Debug, paranoid, component off | 168 of 168 | 168 of 168 | 168 of 168 |
| Debug, paranoid, profiling on, component on | 341 of 341 | 342 of 342 | 342 of 342 |
| The two Debug builds of the upstream CI (noisier, paranoid, profiling; shared and static) | compile | compile | compile |

- The first two rows are what the upstream CI tests. Its Debug builds are only compiled there, and here.
- **No warning** from the compiler or from the PTG compiler in the QPU files, in any of the 24 builds.
- **With the component off**, no file of it is compiled, and `libparsec.so` has no `parsec_qpu` symbol (46 with it on). The one test left with "qpu" in its name is `dsl/ptg/qpu_fallback`, which is about the PTG compiler.
- **The 167 other tests** pass with the component on and off, in Debug and in Release, and the lists of passing tests are the same.
- Copy 2 has one test more than copy 1 (`multi:named:statistics`, below): 175 QPU and QAOA tests of 342.

**Every QPU and QAOA test again and again**, three tests at a time, each until it fails or has passed the given number of times:

| Copy | Machine | Build | Times | Tests that never failed | Failed once |
|---|---|---|---|---|---|
| 1 | serotonin | Debug | 100 | 171 of 174 | `overlap:1worker`, `termination:dynamic:mp:4`, `chain:dtd:mp` |
| 1 | serotonin | Release | 100 | 172 of 174 | `overlap:1worker`, `multi:noidle` |
| 2 | caffeine | Debug | 20 | 174 of 175 | `termination:dynamic:mp:4` |
| 2 | caffeine | Release | 20 | 175 of 175 | |
| 3 | caffeine | Debug | 100 | 173 of 175 | `chain:dtd`, `termination:dynamic:mp:4` |
| 3 | caffeine | Release | 50 | 175 of 175 | |

What each of the five is:

| Test | What happened | Whose it is | What was done |
|---|---|---|---|
| `runtime/qpu/chain:dtd:mp` | The quantum task on rank 1 was handed a circuit that a later task on rank 0 had already overwritten. The QPU layer refused it ("an operation is not a known one"), and the test failed cleanly | **PaRSEC's DTD, at the base commit**: the first defect below | The test overwrites the circuit only on the QPU's own rank, which is the only place where that says anything about the QPU layer |
| `runtime/qpu/chain:dtd` | One process. The body of the quantum task found its circuit already overwritten, with the same message: `scribble`, the writer inserted after the quantum task, had run before it | **PaRSEC's DTD again, at the base commit and upstream**: the second defect below. The body was handed the right circuit and ran once | Left as it is, with the defect named in the header of the test. It can fail again, about once in a hundred runs on a busy machine, and so can `chain:dtd:mp`, whose chains on the QPU's rank are the same |
| `runtime/qpu/termination:dynamic:mp:4` | An assertion of the dynamic termination detector, on the communication thread: a message arrived for a taskpool that this rank had not yet marked ready (`termdet_fourcounter_module.c:596`) | **PaRSEC's dynamic termination detector**: below | Left as it is, with a comment at the test that names the race. It can fail again, about once in a hundred runs beside other tests |
| `runtime/qpu/overlap:1worker` | The run took 506.8 ms against a limit of 500. Every other check of the test passed: the classical chain did run during the job | The test. It measures time, its one worker is bound to core 0, and so are threads of the tests that run beside it. M8 saw the same under load | The two `overlap` tests are marked `RUN_SERIAL`: no test runs beside them |
| `runtime/qpu/multi:noidle` | Not known for that run: the job had not kept the output of failing runs. Reproduced afterwards with the output kept: when other processes take the test's cores, the three QPUs ask for their first jobs 8 to 29 ms after the tasks were submitted, and the moments before that are moments with a free place beside a waiting task | The test, on a machine that is busy: below | Marked `RUN_SERIAL`, like the two `overlap` tests |

**`multi:noidle` under load.** The test's command with its full output (`-v`), again and again; "off" is a moment at which a task waited beside a free place, and a run fails with more than one in ten.

| Machine and build | How | Runs | Failed | Runs with a moment off |
|---|---|---|---|---|
| methane, Release of copy 3 | Alone | 600 | 0 | 0 |
| | Three at a time | 1,800 | 7 | 12, all in the half minute in which a build of another tree ran on the machine |
| | Beside the other QPU and QAOA tests, three at a time | 600 | 0 | 2 (1 and 3 moments) |
| serotonin, Release of copy 1, as in the repeat that failed | Alone | 400 | 0 | 0 |
| | Beside the other tests, three at a time | 3,000 | 0 | 2 (3 and 4 moments; 5 fail the test) |
| histamine3, Release of copy 3, the test on 8 cores | Alone, before and after | 600 | 0 | 0 |
| | 32 processes that spin, on the same 8 cores | 300 | 152 | 221 |
| | The same 32 processes on 8 other cores | 300 | 0 | 0 |

- **It is the threads waiting for a core, at the start.** The same load fails half the runs when it takes the test's cores, and leaves every moment in place when it does not.
- **Once started, no place stays free.** In the 27 failing runs whose output was kept (7 on methane, 20 on histamine3), every moment off lies before the first job of the last QPU to start, 12 to 29 ms after the tasks were submitted; not one lies after it.
- **What the run on serotonin shows:** the load of the test suite alone brings the test within one moment of failing, twice in 3,000 runs. That it was this that failed the one run of the repeat is likely and not shown: that output is lost.
- With jobs of seconds a start that is 20 ms late does not matter. With the mock's jobs of 40 ms it fails this test, so the test runs alone. A machine that is busy with something else can still fail it, as it can the two `overlap` tests.

**The first DTD defect, without any QPU code: a reader on another rank.** A reader on another rank, between two writers of a tile, can be sent the tile after the second writer has changed it in place. The reproducer is the upstream `dtd_test_war.c` with the readers moved to the next rank (`dtd_test_war_remote.c`, with the harness), built at `09c721868`. Per tile: a writer, a reader on the next rank, a second writer. 2 ranks, 20 runs for each size:

| Tile | Runs that failed | Readers that saw the tile of the second writer |
|---|---|---|
| 4 bytes | 0 of 20 | 0 of 40,000 |
| 1 KB | 0 of 20 | 0 of 40,000 |
| 64 KB | 20 of 20 | 39,739 of 40,000 |
| 1 MB | 20 of 20 | 3,869 of 4,000 |

A small tile travels with the first message and the window is narrow, which is why the QPU test, whose tiles are small, met it once in a hundred runs under load. A tile that is sent in a second message reaches nearly every remote reader overwritten. This is wrong data without an error, in distributed DTD.

- **It is in the upstream master of this day too** (`f81532dbe`, five commits after the base, one of them in DTD: `bfa2f8bfa`, "dtd: do not release a reader that was never counted"). The same reproducer there: 4 bytes and 1 KB, 0 of 20 runs; 64 KB, 20 of 20 (39,601 of 40,000 readers); 1 MB, 20 of 20 (3,940 of 4,000).
- **Not reported upstream.** A search of the upstream issues and pull requests by keyword found no report of it. Where in DTD the remote reader is let go before the tile has left was not looked for.

**The second DTD defect, without any QPU code: on one rank, a writer inserted behind a reader can run before it.** Found by reading DTD after `chain:dtd` had failed once in one process, and then shown.

- **What DTD does** (`parsec/interfaces/dtd/overlap_strategies.c` and `insert_function.c`). A writer that follows readers of a tile is held back by the count of readers on the data copy: its `prepare_input` returns `AGAIN` while the count is above zero (`data_lookup_of_dtd_task`). When a task completes, the walk over its successors counts each reader. For the reader that is the last task inserted on the tile so far, it first marks the tile's last user as gone (`made_sure_nextinline_is_null`), and counts the reader a few instructions later (`parsec_dtd_data_copy_reader_retain`).
- **The window.** A writer inserted on the tile between the two takes the path of a task whose parent is gone, is released by its own insertion, finds the count at zero, and runs before the reader. It takes the walking thread losing its core within those few instructions, which is why it showed once, on a busy node.
- **Shown with a pause at that place.** Scratch copies of the sources, with `usleep` between the mark and the count when a variable is set. Without QPU code the test is a writer, one reader and a second writer per tile, 2,000 tiles, one process of 8 cores (`dtd_test_war_remote.c`, whose readers are on the writers' rank when there is one rank).

| Sources | Test | Pause | Runs that failed | What was seen |
|---|---|---|---|---|
| This branch (copy 3) | `dtd_qpu_chain`, with counters per chain added to the test | none | 0 of 100 | |
| | | 20 us | 68 of 100 | In every failing run: `scribble` ran before the body of the quantum task; the body then ran once, on the circuit `prep` had written, and found it overwritten |
| | | 200 us | 31 of 100 | The same |
| The base commit `09c721868`, no QPU code | Writer, reader, writer | none | 1 of 10 | 1 reader of 20,000 saw the tile of the second writer |
| | | 20 us | 10 of 10 | 3,316 of 20,000 |
| | | 200 us | 10 of 10 | 2,994 of 20,000 |
| Upstream master `f81532dbe`, no QPU code | Writer, reader, writer | none | 0 of 10 | 0 of 20,000 |
| | | 20 us | 10 of 10 | 3,374 of 20,000 |
| | | 200 us | 10 of 10 | 3,112 of 20,000 |

- **Without the pause** the base commit failed one run of ten on a quiet machine: the window is met in real runs, rarely.
- **Upstream's own test** `dsl/dtd/dtd_test_war` has one tile per rank, so in one process it has one chance per run: 30 of 30 pass with the pause.
- **With five readers per tile** and the pause, no run failed (0 of 100,000 readers, in both trees). This is not a case of its own; the session that made the fix (below) found why. The pause also held the inserting thread, once for each reader that follows, so the second writer always arrived after the walk had gone on. With the pause on the other threads only, it reports that five readers fail in 10 runs of 10 without the fix and in none with it.
- **A candidate fix: count the reader before the mark** (16 lines of `overlap_strategies.c`; `dtd_war_window_candidate.patch`, with the harness). In scratch copies, with the pause still in place:
  - writer, reader, writer: 0 of 10 runs at 20 us and at 200 us, 0 of 20,000 readers, on the base commit and on upstream master;
  - `dtd_qpu_chain`: 0 of 100 at 20 us, at 200 us and without the pause, and the counters report nothing;
  - every test with "dtd" in its name, without the pause: 42 of 42 on upstream master and 41 of 41 on the base commit, with the candidate and without it; ten times each on upstream master, 42 of 42 without it and 42 of 42 with it.
- **The candidate is not in this branch**, which changes nothing in DTD, and it was run on one node only. Not reported upstream.
- **Since then (2026-11-27, later that day): the fix is a branch of its own**, cut from upstream master, in the fork as draft pull request QingleiCao/parsec#28 (`fix/dtd-reader-count-window`: the same idea with one place that counts, and a test `dsl/dtd/war_last_reader` that holds its own threads up to meet the window). By that pull request's account the test fails 280 runs of 280 without the fix and none with it, and no build with an accelerator was run. Until the fix is in PaRSEC, `chain:dtd` can fail as described.
- **The file is the same at both commits.** `overlap_strategies.c` is the same at the base commit and in upstream master, and the DTD commit that upstream made after the base (`bfa2f8bfa`) is about another place: with its two lines applied nothing differed.
- **What did not reproduce it before the reading**, for the record: pauses after every insertion on a quiet methane (0 of 1,200 runs); counters behind a lock in the test, on histamine3, quiet and with 32 spinning processes on the test's cores (0 of 3,600).
- **The two defects are not one.** For a reader on another rank no count is taken at all, and the tile is sent later; here the count is taken, a moment too late.

**The detector's race, without any QPU code.** The test of that detector on the branch of ICLDisco/parsec#869 (the base commit and that fix, no QPU code), `dsl/ptg/dynamic_termdet:mp` on 4 ranks, met the same assertion once in 2,037 runs, three loops at a time on methane. On an idle methane, `termination:dynamic:mp:4` and `chain:dtd:mp` of this branch then passed 400 times out of 400 each. The detector's files are the same in the upstream master of this day, and no open pull request there is about this. Not reported upstream.

**More than one node.** histamine2 and histamine3 (AMD, 16 cores each, Ethernet), a Debug build in the shared home, the ranks of every test on the two nodes in turn (`--map-by node`).

- **With Open MPI's default path (UCX), 13 tests fail, and the base commit fails the same way.** 7 of the 174 QPU and QAOA tests, all of 4 ranks, and 6 of the 167 others (`collections/redistribute:mp`, `collections/redistribute_random:mp`, `collections/reshape:mp`, `collections/reshape:mp:mt`, `apps/merge_sort:mp`, `apps/stencil:mp`). The same tests fail every time. A build of `09c721868`, with no QPU code, fails the same 6 (161 of 167, job 68132). So the failures are not this branch's.
- **Where it fails.** Inside the PMIx client of Open MPI 5.0.5 (`libpmix.so.2`), reached from the UCX layer when a rank asks for the address of a peer it contacts for the first time: `pml_ucx.c:188 Error: Failed to receive UCX worker address`, then a segmentation fault or an error of `MPI_Send`.
- **What does not reproduce it**, all programs that use MPI only, across the two nodes: every rank sending to every other; the same from a second thread on a duplicated communicator, as PaRSEC talks; first contacts made by 1, 2 and 4 threads at the same moment, on 4, 8 and 16 ranks (20 of 20 each).
- **What does not avoid it:** two ranks next to each other on a node (0 of 12 runs pass); `mpi_preconnect_mpi` (4 of 12); ranks started by `srun` with Slurm's PMIx (2 of 12).
- **Over TCP the whole suite passes** (`OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=tcp,self,sm` and the site's `btl_tcp_if_include`; job 68142, copy 2): the QPU and QAOA tests **175 of 175**, the others **167 of 167**; 126 of the 342 tests start through `mpiexec`. The 37 QPU and QAOA tests with ranks in their names, 10 times each: 370 runs, all pass.
- **Not found:** what in PaRSEC's use of MPI brings out the failure on the UCX path of these nodes. It is there at the base commit.

**A second independent review**, of the follow-up of the CUDA run (a reviewer that had the difference and none of the reasoning; it changed nothing). It found no defect in the runtime part: the counter, its threads, how a QPU is looked up, the reset, and the line. It found six gaps in the tests, and each was checked against the code and closed (7 files, 79 lines added, 22 removed; copy 2):

| Finding | What was done | The deliberate bug that is now caught |
|---|---|---|
| The device check of the tests passed on a rank with no QPU device at all, also with one QPU. The follow-up had lost that check | It is an error again | The device that stands for the QPUs of other ranks is missing: 19 tests fail |
| `cpu_gpu_qpu:statistics` ignored the exit code, as `PASS_REGULAR_EXPRESSION` does | It runs through `repeat_test.cmake`, which checks the exit code and, new, what is printed (`-DEXPECT`) | The test returns 3 after "passed": 3 tests fail |
| No test read the count of each QPU with several QPUs | `multi:named:statistics` (two QPUs, 12 tasks each, read from the two lines), and `qpu_multi` compares the count of each QPU's service with that of its device | Every task is counted for the first QPU: 27 tests fail (1 before). A task is not counted: 105 |
| `-q 0`, `-q x` and `-q` without a value dropped the check silently | The options are checked | Each ends with an error |
| The device check ignored whether a QPU it owns has a service | Checked | |
| Comments said a QPU "ran" the tasks counted for it. A task that ends without a job is counted for the QPU the device layer had picked | The comments say so | |

**The test changes after the repeated runs.** In copy 3: `dtd_qpu_chain.c` and the `RUN_SERIAL` mark of the two `overlap` tests, both in the table above. On methane the suite passed with them, 342 of 342, and the four tests concerned passed 50 times each, three at a time. After copy 3: the `RUN_SERIAL` mark of `multi:noidle`. On methane, a Debug build of the working tree with it: the suite 342 of 342, and every QPU and QAOA test 10 times, three at a time, 175 of 175 (ctest runs the three marked tests with nothing beside them). No other build configuration was run with this last mark; it changes when a test runs and nothing that is compiled.

**The experiments.** A Release build of copy 1, the mock backend, the checks of section 9.7 on every run (final states equal to the reference byte for byte, no late job). All 526 runs that count passed, with no late job. The model is `exp_qpu/model_multi.py`: the model of section 9.7 for several QPUs. Its self-test holds two things: with one QPU it gives the times of `model.py` to the nanosecond, and QPUs that are alike, with tasks bound when a place is free, give the times of one QPU with the places of all. Tables are in `exp_qpu/tables/` (`e9`, `e9_jobs_per_qpu`, `e5q`, `e6q`, `e6q_caffeine`, `e6q_2nodes`), plots in `exp_qpu/plots/` (`e9_binding`, `e5q_sharing`, `e6q_ranks`).

| | E9 | E5 on several QPUs | E6 with QPUs on several ranks |
|---|---|---|---|
| Machine | caffeine: Intel Xeon E5-2660 v4, 14 cores, an exclusive Slurm allocation; no process of another user in the load log | the same | histamine3 (16 cores); caffeine; histamine3 and histamine2 over TCP |
| Processes | 4 ranks of 3 cores | 4 ranks of 3 cores | 8 ranks: of 2 cores; of 1 core; of 4 cores, on the two nodes in turn |
| QPUs | 2 and 4 on rank 0, of 1, 2 and 4 places | 1 and 3 on rank 0 | one on each of the first 1, 2, 4 and 8 ranks, with as many places in all as jobs can be in flight |
| Job times | lognormal, mean 20 ms, coefficient of variation 0, 1, 2; one setting at 0.1 s | 10 ms for a job of one circuit, as in E5 | constant: 200 ms and 20 ms |
| Instance | E3's grid: 16 blocks of 8 vertices, T 8, R 4 (1,024 jobs) | E5's: 16 blocks of 10 | rings of 16 and 64 blocks, of 12 vertices (200 ms) and 8 (20 ms); T 8, R 2 |
| Runs | 307; 5 seeds, 3 at constant times | 75; 3 seeds | 48 on each of the three layouts; 3 seeds |

**E9: what binding when a place is free is worth.** "Bound when free" is the rule of section 10.3. "In turn" lets each block name a QPU, in turn (`-Q`), so that its tasks wait for that one. "One" is one QPU with the places of all. Ratios per seed, then the median and the range.

| QPUs x places | CV of the job time | Bound when free / one | In turn / bound when free | The same, model |
|---|---|---|---|---|
| 2 x 1 | 1 | 1.000 [1.000, 1.003] | 1.056 [1.020, 1.077] | 1.045 |
| 2 x 1 | 2 | 1.000 [0.999, 1.006] | 1.111 [1.057, 1.155] | 1.100 |
| 2 x 2 | 1 | 1.001 [0.999, 1.002] | 1.037 [1.020, 1.052] | 1.033 |
| 2 x 2 | 2 | 0.998 [0.993, 1.006] | 1.072 [1.051, 1.122] | 1.048 |
| 2 x 4 | 1 | 0.997 [0.994, 0.999] | 1.033 [1.027, 1.076] | 1.065 |
| 2 x 4 | 2 | 0.999 [0.992, 1.009] | 1.025 [1.013, 1.042] | 1.026 |
| 4 x 1 | 1 | 1.001 [0.998, 1.006] | 1.089 [1.066, 1.173] | 1.083 |
| 4 x 1 | 2 | 1.002 [0.996, 1.021] | 1.163 [1.110, 1.282] | 1.143 |
| 4 x 2 | 1 | 1.002 [0.995, 1.012] | 1.060 [1.056, 1.110] | 1.071 |
| 4 x 2 | 2 | 1.001 [0.997, 1.008] | 1.055 [1.022, 1.113] | 1.039 |
| 4 x 4 | 1 and 2 | 1.000 | 1.000 | 1.000 |
| any | 0 | 1.000 | 1.000 | 1.000 |

- **Several QPUs with one pool cost nothing** against one QPU with the same places: 0.3% at most in the median of any row.
- **Naming a QPU in turn costs 3 to 16%** when job times vary and places are short, as the model says, and nothing at constant times or with a place for every task (4 x 4: 16 blocks, 16 places).
- **Measured against the model:** within 2.5% in every row but one (4 x 4 at constant times, 3.4%: the run lasts 1.3 s, and 43 ms of it are not jobs).
- **With jobs of 0.1 s** (4 x 2, CV 1, 3 seeds): 1.082 [1.068, 1.113], model 1.071.

The first QPU slower than the others, 2 places each:

| QPUs | Slower by | CV | In turn / bound when free | The same, model | Jobs on the slow QPU, of 1,024: bound when free | In turn |
|---|---|---|---|---|---|---|
| 2 | 2 | 0 | 1.497 | 1.497 | 342 | 512 |
| 2 | 2 | 1 | 1.475 [1.433, 1.526] | 1.50 | 347 [331, 378] | 512 |
| 2 | 4 | 0 | 2.485 | 2.485 | 206 | 512 |
| 2 | 4 | 1 | 2.390 [2.337, 2.481] | 2.49 to 2.50 | 210 [185, 225] | 512 |
| 4 | 2 | 0 | 1.727 | 1.73 to 1.74 | 148 | 256 |
| 4 | 2 | 1 | 1.664 [1.346, 1.715] | 1.54 to 1.63 | 158 [142, 161] | 256 |
| 4 | 4 | 0 | 3.177 | 3.18 to 3.20 | 80 | 256 |
| 4 | 4 | 1 | 2.445 [2.217, 2.824] | 2.15 to 2.78 | 100 [97, 101] | 256 |

- **With one slow QPU, naming a QPU in turn takes 1.5 to 3.2 times as long.** Every block that names the slow QPU waits for it, and its neighbours wait for the block. Bound when free, the slow QPU runs fewer jobs, in proportion to its speed.
- The model's two values are for the two extreme orders in which QPUs with a free place are asked. In the runtime the order is that of its threads' wake-ups, which nothing fixes.
- The job counts are read from the line each QPU prints with `device_show_statistics`, the line that the follow-up of the CUDA run fixed. At constant times they are the model's exactly.

**E5 on several QPUs: tasks that share jobs.** The limit is `device_qpu_max_batch`. Gain is the time with every task its own job over the time with the limit, per seed.

| Places | Limit 2 | 4 | 8 | 16 | Time at limit 1, ms |
|---|---|---|---|---|---|
| One QPU of 3 places (E5's "three") | 1.73 | 2.38 | 2.28 | 2.16 | 3,444 |
| 3 QPUs of 1 place, bound when free | 1.73 | 2.63 | 2.32 | 2.25 | 3,442 |
| 3 QPUs of 1 place, named in turn | 1.70 | 2.55 | 2.62 | 2.61 | 4,655 |
| One QPU, one submission per 51 ms | 2.00 | 4.01 | 8.03 | 15.81 | 52,241 |
| 3 QPUs, each one submission per 51 ms | 2.00 | 3.98 | 6.35 | 6.42 | 17,423 |

- **Three QPUs of one place share jobs at least as well as one QPU of three places**: the same time at limits 1 and 2, and 2 to 10% less at 4, 8 and 16. At limit 4 they reach the model's gain (2.63 against 2.64), where one QPU of three places stays 10% below it, as in M8.
- **Named in turn, every time is 20 to 40% longer** than bound when free, with shared jobs or without. The gains look alike because the time without sharing is longer too.
- **With a submission interval per QPU**, three QPUs submit three times as often (17.4 s against 52.2 s), and sharing stops paying beyond 8 tasks a job, where the interval no longer limits the run.
- **Measured against the model:** within 3.5% up to limit 4, except one QPU of three places at limit 4 (12% above). Beyond, the times lie between the model's two assumptions about how released tasks arrive.

**E6 with QPUs on several ranks.** The quantum tasks of block `b` run on the owner rank `b` modulo the number of owners. With jobs of a constant time J the schedule alone gives 32 J.

| Layout | J | Blocks | Time with 1, 2, 4, 8 owner ranks, ms | 8 owners / 1 owner | Per step beyond J: 1 owner, 8 owners |
|---|---|---|---|---|---|
| One node, 2 cores per rank | 200 ms | 16 | 6,433 / 6,432 / 6,429 / 6,413 | 0.997 | 1.0 ms, 0.4 ms |
| | 200 ms | 64 | 6,455 / 6,449 / 6,441 / 6,421 | 0.995 | 1.7 ms, 0.7 ms |
| | 20 ms | 16 | 670.9 / 665.7 / 669.2 / 656.7 | 0.979 | 1.0 ms, 0.5 ms |
| | 20 ms | 64 | 672.6 / 676.4 / 673.0 / 659.9 | 0.981 | 1.0 ms, 0.6 ms |
| Two nodes, TCP | 200 ms | 16 | 6,436 / 6,432 / 6,431 / 6,417 | 0.997 | 1.1 ms, 0.5 ms |
| | 200 ms | 64 | 6,454 / 6,444 / 6,440 / 6,425 | 0.996 | 1.7 ms, 0.8 ms |
| | 20 ms | 16 | 673.8 / 668.8 / 668.5 / 655.7 | 0.973 | 1.1 ms, 0.5 ms |
| | 20 ms | 64 | 685.2 / 677.5 / 678.0 / 664.8 | 0.970 | 1.4 ms, 0.8 ms |

- **The flat line that was expected.** Giving several ranks a QPU of their own changes the time by 3% at most. It gets slightly shorter with more owners: with a QPU on every rank a block's quantum tasks run on the block's own rank, and the message to the rank of the QPU and back is gone.
- **One core per rank** (caffeine): the same, 0.975 to 0.994 with 8 owners.
- Every QPU ran the same number of jobs in every run (16 blocks: 512, 256, 128 and 64 each).
- **This is the run on more than one node that E6 lacked in M8** (section 9.7), on two nodes and over TCP on Ethernet. It says nothing about more nodes, or about a faster network.

**What this night does not show.**

- Real hardware: every QPU is the mock, and QPUs differ only in the job times they are given.
- More than two nodes, and a network faster than Ethernet.
- QPUs that differ in what they can run (width, shots, gates) under load: the tests of that are functional, and no experiment has them.
- CUDA with the changes of copies 2 and 3. They touch tests and comments only, and the CUDA jobs above ran on the tree before them.

### Things found on the way

- **Q-IRIS on several QPUs** (arXiv:2605.11382, read in full through a separate pass on 2026-11-26). The paper names no rule for choosing among QPUs: it says "configurable scheduling policies", and that the driver "can concurrently dispatch multiple QIR programs ... to different accelerators". It does not say when a task is bound, whether tasks wait in one queue or one per device, what happens when a QPU fails, or how many QPUs ran at once; it runs "on a single node" and lists "multi-node support" as future work. So nothing published there conflicts with the rule of section 10.3, and nothing there can be compared with it. Section 5 still needs this added.
- **The PTG body property `device=`** does not work at the base commit (section 10.3). A separate session reproduced it and has a fix on a branch of its own.
- **A progress thread can be held up for milliseconds** inside a call to the mock (above). With one QPU that only delays its own job, which is why it was not seen before.

### Departures from the plan

1. **`@*` is not in the list syntax** (section 10.3 dropped it at the recheck), and entries may be separated by `|` as well as `;`: a `;` ends an argument in a CMake list, which is where the tests write their lists.
2. **`device_qpu_enabled` still has to be set** with a list. The list says which QPUs there are, and the parameter stays the one switch that turns the layer on.
3. **Statistics.** `parsec_qpu_service_stats()` keeps its signature and gives the rank's totals; `parsec_qpu_service_stats_of(q, ...)` and `parsec_qpu_mock_stats_of(q, ...)` are per QPU. Section 10.3 gave the first one a QPU number, which would have changed every existing test.
4. **The hardware guard in a list.** A backend whose name says it drives hardware, or that does not exist, disables every QPU of the list. One that says so only when it is opened is closed again and left out, and the others stay, like any backend that cannot be opened.
5. **The tasks of a submission that is given up on, the retry on another QPU, and the QPU that is tried again**: see "How a task gets its QPU".
6. **The wake-up rule** is stricter than section 10.3 wrote it: see above. And **the tag of a job** counts jobs put together, not attempts (section 2.5).
7. **The timing test** of section 10.3 ("in the time the model gives for one QPU of six places, within 5%") became `multi:noidle`, which reads the same property from the times in the results and does not depend on how fast the machine is.
8. **Jobs that carry several tasks across QPUs** count the other QPUs' free places from what those last published, without a lock. The count can be a moment old. `multi:batch` checks the limits, not how evenly tasks are spread.
9. **The seven deliberate bugs of M7** (section 9.6) were not planted again: their text no longer applies to the rewritten file. The tests that caught them all still pass.
10. **More than 16 devices in a process**: the QPU that does not fit is left out, and the others stay. Section 10.3 disabled the layer.
11. **Not built:** taking tasks across ranks (by design); a test of QPUs with different limits of shots; a run with a real backend.

### For later milestones

- **A backend sees its QPU's number** as the `device_index` of `open`, and reads its options with `parsec_qpu_list_option(number, key)`.
- **A backend's calls may block.** `submit`, `results`, `cancel` and `abandon` are called without the pool's lock, and the other QPUs go on meanwhile.
- **Experiments.** E5 and E6 were run again on several QPUs, and the comparison of binding rules of section 10.6 (E9) with them, on 2026-11-27: above. The model with one queue and several groups of places is `exp_qpu/model_multi.py`.

## 9.9 M4a: the adapter backend, OpenQASM 3, and Qiskit Aer (2026-11-27)

Approved with "sure. go ahead" to the question "Start M4a, the adapter with Qiskit Aer?". Built on `c419f982f` (M9 and the checks that followed it). The dependency it adds is outside the build: a Python environment with `qiskit`, `qiskit-aer` and `qiskit-qasm3-import`, for the adapter program and for the optional tests. No provider was contacted, no account exists, and nothing in the tree has code that could reach a quantum device.

### What exists now

- **`parsec/mca/device/qpu/device_qpu_qasm.c` and `device_qpu_qasm.h`** (new, the header installed): a circuit as OpenQASM 3 text, written and read, with no dependency.
- **`device_qpu_backend_adapter.c`** (new): the backend `adapter`, the client of a separate program over a Unix-domain socket. It is in the table of backends beside the mock.
- **`device_qpu_json.c`** (new): as much JSON as the protocol needs, read in place, and a text that grows.
- **`tools/qpu/parsec_qpu_adapter_qiskit.py`** (new): the adapter. It runs the jobs on Qiskit Aer and has no other backend.
- **Two parameters**, `device_qpu_adapter_address` (the name section 7.2 fixed) and `device_qpu_adapter_timeout`, and two keys of a list entry, `address` and `timeout`.
- **`check_no_hardware.cmake`** knows the adapter, and has a test of itself (`check_no_hardware_selftest.cmake`).
- **Tests** (`tests/runtime/qpu`): `qpu_qasm_test.c`; `qpu_adapter_loopback.c`, an adapter in C for the tests, which answers from the sampler; `qpu_adapter_test.c`, the client alone against it; `qpu_adapter_run.c`, which starts an adapter beside a command; `qpu_adapter_stat.c`, the client against an adapter that simulates by its own means; `qpu_adapter_sessions.c`, which asks that adapter, in the protocol and without the client, what it keeps for a client whose connection is gone. 26 new tests: 21 that need nothing but the build, and 5 that are registered only when a Python with Qiskit Aer is found. The suite has 368 tests with all of them.
- **`tools/qpu/README.md`**: how to use the adapter, the OpenQASM 3 functions, and the protocol.
- **Nothing outside the component, its tests and its tools changed.**

### Circuits as OpenQASM 3, as built

- **The writer** declares the parameter slots as `input float[64] p0;`, `p1`, and so on, writes an angle as a number or as `number * pK`, and measures qubit i into bit i. Numbers are written with 17 significant digits, which reads back as the same double.
- **rxx, ryy and rzz are not in OpenQASM 3's standard library**, so the writer defines the ones a circuit uses: `cx, rz, cx` for rzz; that between `h` on both qubits for rxx; that between `rx(pi/2)` before and `rx(-pi/2)` after for ryy. `qpu_qasm_test` reads the three definitions back as circuits and compares the states they prepare with those of the native operations, through the exact sampler.
- **The reader** takes what the writer writes, and what `qiskit.qasm3.dumps` of Qiskit 2.6.2 writes for a circuit of these gates (a text of it is in the test): inputs of type `float[64]` or `angle`, gate definitions (skipped), `barrier` (ignored), angles such as `(-1.5)*gamma` or `pi/2`. The k-th `input` is slot k; Qiskit declares its parameters in the order of their names.
- **Angles are an expression of numbers, `pi` and one input**, which has to come to a number or to a number times that input, since that is all an operation of the circuit format holds. `a + 1`, `a * b` and `1 / a` are refused, with the line.
- **Written and read back, a circuit is the same bytes**, the fields an operation does not use apart (the text does not hold them). The test does it for every operation, for angles down to the smallest double and for the two zeros, and for 200 circuits drawn at random. This is what lets a job through the adapter of the tests give the counts of the sampler.

### The adapter protocol, as built

Section 2.6's table, with these additions. `tools/qpu/README.md` has the table as built.

- **`hello` names a session.** The client makes the name up when it opens (time, process, host, QPU). The adapter keeps a client's jobs under it, so a client that connects again finds them, and two runs of one program do not share tags (the point section 9.2 left for M4).
- **`submit` carries the key and the attempt of each evaluation.** The adapter of the tests needs the key to answer as the sampler does. The Qiskit adapter seeds its simulator from its `--seed` and the key, so its counts depend on those two and on nothing else, as the mock's do.
- **`wake`.** A `wait` is held by the adapter until a job ends or its time is up. The client's thread sits in `poll()` on the socket and on a pipe; `wakeup()` writes a byte to the pipe, and the thread then sends `wake`, which makes the adapter answer the `wait` at once. So every byte on the socket is written and read by the progress thread, and the backend has no lock.
- **`abandon(tag)`**, for a submission whose answer never came.
- **A job that ended is told again after a new connection** until the client has taken its result or said `cancel`, so an answer lost on the way does not lose the job.
- **64-bit values are strings**, `"0x..."`: a JSON number cannot be trusted with them.

### The client when things go wrong

| What happens | What the client does |
|---|---|
| A request gets no answer, or one that is not of the protocol | Closes the connection. The next call connects again, not more often than every 20 ms to 1 s |
| The answer to `submit` is lost | `submit` returns "ask again". Asked again with the same tag, the adapter answers with the job it has: the job ran once |
| The job of such a submission ends before the caller asked again | The end is kept, and reported only once `submit` has returned the job's identifier, as the interface requires |
| The adapter was started again (it does not know the session) | Its jobs are reported as failed, and the runtime submits them again. The circuits are given again |
| The adapter stays out of reach for `unavailable_timeout` | Its jobs are reported as failed, and it is told to drop them if it comes back. `wait` sleeps between attempts |
| An adapter that takes no connection, with its queue full | An error at once: the socket does not block |
| The adapter that was started again gives its jobs the names the first one used | The caller knows a job by a name of the client's own, which is never used twice |
| The adapter says `busy`, or `unknown_circuit` | "Ask again"; the circuit is given again, once |
| The result is not one: a shot missing, a bitstring twice or too wide | `results` fails, and the job is forgotten on both sides |
| The adapter that answers after a new connection says it drives a device, and the first one did not (or the reverse) | It is not used again |

### The hardware guard and the adapter

- **What an adapter drives is what it says in `hello`.** The backend's description cannot know it before, so it is the second check of section 9.2 that applies: a backend that says so once open is closed at once, with no job. An adapter that does not say is taken for one that drives a device.
- **An adapter that changes its answer** when the client connects again is dropped (table above). Without that, a simulator replaced by a hardware adapter in the middle of a run would get past a guard that looked once.
- **`no_hardware_in_tests`** accepts `adapter` in a list of QPUs, and has a third check: no test definition may start the Qiskit adapter with a backend other than `aer`, or with an option that names a device or an account.
- **The adapter of this milestone has one backend, Aer.** `--backend` takes nothing else. M4b adds the one that reaches a provider, behind a flag of its own (section 4.2, layer 5).
- **Credentials.** The client reads no variable of the environment and no file, and writes no message to a log. The test sets the variables an SDK reads its credentials from to a known text, tells the adapter of the tests that text, and the adapter fails the test if it ever sees it.

### The Qiskit adapter

- **One thread per client**, a pool of threads for the jobs, one lock.
- **`prepare`** reads the text with `qiskit.qasm3.loads` and transpiles it once for the simulator.
- **`--device generic:N[:line|ring|full]`** makes the simulator a device model: `GenericBackendV2` of Qiskit, with the gates `cx, id, rz, sx, x`, the connectivity asked for, and made-up error rates. It is not the model of an existing device. Section 10.4 left open which noise models of IBM devices the SDK ships: they are in `qiskit-ibm-runtime`, which this milestone does not install.
- **Its socket** can be opened by the user who started it and nobody else.

### Verification

Every job built from a fixed copy of the working tree. The numbers are those of snapshot 4: `c419f982f` with the uncommitted work of M4a as it is after the review and after the tests that the deliberate bugs led to (the difference in `parsec`, `tests` and `tools` has the sha256 `fdc607f2be562f71`). No file of the component or of `tools/qpu` differs between snapshots 3 and 4; the tests do. Snapshot 3, the code after the review, had passed the same configurations on serotonin (367 of 367, 168 of 168, each test of M4a 30 times), and snapshot 1, before the review, too (364 of 364, 168 of 168). Scripts and logs: `exp_qpu/checks/m4a`.

| Check | Result (caffeine, an exclusive allocation; gcc 11.5, Open MPI 5.0.5) |
|---|---|
| The whole suite: Release, Release with profiling, Debug, Debug with profiling | 368 of 368 in each: 201 QPU and QAOA tests, 26 of them new, 5 with Aer |
| The component off (Debug), and the default options (Release) | 168 of 168 in each. No symbol of the component in the library, no file of it compiled. The 167 other tests pass with the component on and off, the same list |
| The two Debug configurations of the upstream CI | Compile |
| Compiler warnings in the component and its tests, 8 builds | 0 |
| Configured with a Python that has no Qiskit | 363 tests, none with Aer, and a line that says so. Nothing else differs |
| Headers installed for the component | `device_qpu.h`, `device_qpu_qasm.h` |
| The 33 tests of M4a and the ones it touched, 30 times each, three at a time: Debug, Release | 33 of 33, and 33 of 33 |
| Every QPU and QAOA test, 5 times each (Debug) | 201 of 201 |
| Adapter processes left when everything was over | None |

- **On methane**, the whole suite before the review: 363 of 364. `overlap:1worker` failed, the timing test of M1, which uses the mock only; other sessions had the machine at a load of 16, and the test passed 5 times of 5 alone.
- **Since snapshot 4** only this section and a paragraph of `tools/qpu/README.md` (on `--idle-drop`) were written: nothing that is built, or that a test reads.
- **The commits one by one.** M4a is five commits: the component, the adapter program with the README, the tests of the runtime, those of the application, and this section. The states after the first and after the third were built and run through the QPU and QAOA tests on methane (Debug), and the state after the fourth is snapshot 4 with the paragraph of the README. After the first: 175 of 175, the tests as they were before M4a. After the third: 196 of 197. The one that failed is `termination:dynamic:mp:2`, with the assertion of the dynamic termination detector that section 9.8 found on four ranks (`termdet_fourcounter_module.c:596`). Repeated on that build it failed again at the 79th run, with other sessions on the machine; it is the same program in snapshot 4, where it passed each of the 9 times it ran.
- **Not run, when M4a was committed:** a CUDA build (M4a has no GPU code, and the tests of M4a were not built with CUDA); more than one node; a locale that writes numbers with a comma; any Qiskit but 2.6.2 with Aer 0.18.4 and Python 3.12; and no provider, since the adapter has none. The CUDA build and two nodes were run the night after: "The checks that were left, what a job costs through the adapter, and three installs" below.

### The tests

| Test | What it checks |
|---|---|
| `runtime/qpu/qasm` | Section "Circuits as OpenQASM 3": the text written for a small circuit, line by line; round trips; the three definitions; Qiskit's text; 27 angle expressions; 50 texts that have to be refused, each with its line and reason; sizes |
| `runtime/qpu/adapter` | The client alone, as the progress thread drives it, against 20 adapters started one after the other, each with its own faults: the table "when things go wrong", and jobs, `wait` and `wakeup`, `cancel` and `abandon`. Every result is compared with the sampler's, entry for entry, one of them for parameters that need all 17 digits |
| `adapter:chain`, `:chain:dtd`, `:chain:shared`, `:chain:shared:cap`, `:chain:slow` | PTG and DTD chains of quantum tasks through the adapter; jobs of several circuits; an adapter that takes fewer circuits than the runtime would give; jobs longer than the runtime waits in one go |
| `adapter:multi`, `adapter:multi:two` | An adapter between two mocks; two QPUs on one adapter |
| `adapter:restart`, `adapter:lost_answer` | The adapter is started again with jobs in flight, or loses the answer to a submission: every task still has the counts of the sampler |
| `adapter:guard`, `adapter:absent`, `adapter:badlist` | An adapter that says it drives a device gets no job (the adapter of the tests fails the test if it gets one); no adapter where one is said to be; a misspelt key |
| `adapter:chain:mp`, `adapter:chain:mp:last` | Four ranks, the QPU on rank 0 and on the last |
| `apps/qaoa:adapter`, `:adapter:shared`, `:adapter:mp` | The application: final states equal to the reference bit for bit |
| `adapter:stat:circuits` | The circuits of the tests with Aer tell the operations apart (the review, below). No adapter is needed |
| `no_hardware_in_tests:itself` | The script that keeps test definitions away from devices refuses 12 definitions that each break one of its rules, and accepts one that breaks none |
| `adapter:aer`, `adapter:aer:device`, `adapter:aer:noise`, `apps/qaoa:adapter:aer` (optional) | With Qiskit Aer: below |
| `adapter:aer:sessions` (optional) | What the Qiskit adapter keeps for a client whose connection is gone, asked in the protocol by a client that is not PaRSEC's: a session is new or resumed as the adapter says; it stays while a second connection uses it, however long; it is there for a client that comes back within `--idle-drop`, and gone after |

- **Every test with the adapter of the tests** also fails when that adapter saw its client break the protocol: it ends with a code of its own, and the launcher passes it on.
- **What the tests with Aer check.** A simulator of its own does not give the sampler's counts, so nothing is compared count for count.
  - Circuits with one outcome: bit order, the control of `cx`, a qubit that is not measured, the parameter slots, a coefficient, an input the circuit does not use.
  - Every operation: the frequencies of 4,000 shots against the exact probabilities, by total variation distance.
  - The same key gives the same counts, alone and in a job of three; another key gives others.
  - The application runs on it without the comparison with the reference (`-c 0`).

**Measured with Aer** (Qiskit 2.6.2, Aer 0.18.4, `--seed 1`, on methane; the adapter draws from its seed, so a run gives these numbers again). Total variation distance between the frequencies of 4,000 shots and the exact probabilities:

| Circuit | Ideal simulator | Device model `generic:5:line`, without its noise | With its noise |
|---|---|---|---|
| 3 qubits, every operation | 0.023 | 0.023 | 0.037 |
| 4 qubits, every operation | 0.023 | 0.023 | 0.033 |
| 5 qubits, every operation | 0.028 | 0.028 | 0.041 |
| rxx alone | 0.007 | 0.007 | 0.010 |
| ryy alone | 0.020 | 0.020 | 0.017 |
| rzz alone | 0.006 | 0.006 | 0.010 |

- **The tests allow 0.08**, and 0.2 with noise. For scale: 4,000 shots drawn exactly from 32 equally likely outcomes are about 0.036 from them, so the first two columns are what sampling alone gives. And a reader that took one operation for another would move the outcomes of one of the three circuits by 0.32 at least (`adapter:stat:circuits`).
- **Transpiled to the gates and the line of the device model**, without noise, the distances are the same to four digits as on the ideal simulator.
- **The circuits with one outcome** give it in 400 shots of 400 on the first two, and have to give it in 300 of 400 with noise.
- **The application on Aer**: `ring64` with 192 jobs takes its cut from 63.1 to 103.6, in 5.5 s. The run of 96 jobs with two layers takes 5.6 s on Aer and 0.05 s with the adapter of the tests. Where the time on Aer goes (reading, transpiling, simulating) was not measured, nor how many of the jobs are different circuits. Both were measured the night after the commits: E11, below.

### Does a test notice

For each of 61 deliberate bugs, a private copy of the sources was built with that one bug (Debug) and the tests M4a adds or touches were run. A bug counts as noticed when at least one test fails. The bugs are one-line changes of the kind a slip would make: a sign, an index, a check left out, a wrong field. They are listed in `exp_qpu/checks/m4a/mutate_m4a.py`; the verdicts are in `status_mutations_3` and `status_mutations_4` there, and the logs beside them.

| Where the bug is | Bugs | Noticed at first | Noticed with the tests added after |
|---|---|---|---|
| The OpenQASM writer | 6 | 6 | 6 |
| The OpenQASM reader | 8 | 8 | 8 |
| The JSON reader and writer | 4 | 3 | 4 |
| The client (`device_qpu_backend_adapter.c`) | 29 | 24 | 28 |
| The keys of a list entry | 1 | 1 | 1 |
| `check_no_hardware.cmake` | 6 | 5 | 6 |
| The circuits of the test with Aer | 1 | 1 | 1 |
| The Qiskit adapter | 6 | 4 | 6 |
| All | 61 | 52 | 60 |

- **The first pass** (snapshot 3, 33 tests, histamine3): 52 noticed, 9 not. The 33 tests pass before the first bug and after the last one is taken out.
- **The nine that no test noticed**, and what was done:

| Bug | Why no test failed | The test that fails now |
|---|---|---|
| The client sends a parameter with 6 digits, not 17 | The parameters of the tests were short decimals, and a change in the seventh digit moves no count | `runtime/qpu/adapter`: a circuit whose angles are 1e9 and -3e8 times a parameter, compared with the sampler entry for entry |
| The client does not look at the number of an answer | No adapter of the tests answered under another number | The same test: the adapter of the tests has a fault `stale`, an answer under the number of the request before; it is a fourth way to lose the answer to a submission |
| After the adapter was started again the client takes its circuits for known there | The submission is refused (`unknown_circuit`), the circuit is given, and the job is submitted again: the same result, with one request more | The same test counts the submissions the new adapter received |
| `results` asks the adapter for the result of a job whose adapter is gone | A second check, six lines below, gives the same error. Only a caller that takes the lost job for a job of another size gets past both, and the client then told the new adapter to drop the job that has the lost one's name | The same test, a third case of "the same names": the new adapter's job is still there and gives its result |
| `parsec_qpu_json_member` clears its result before it reads its object | It matters only when both are one variable, which no code of the component does (the fuzzing driver did, section "Sanitizers") | The same test reads three members into the variable that holds their object |
| `check_no_hardware.cmake` lets any other adapter script be named | The definition that names one in the test of the script was already refused by the rule on what a test starts | `no_hardware_in_tests:itself`: a twelfth definition, which starts another script through a shell |
| The Qiskit adapter drops a session that a connection still uses (the review's fourth finding) | No test had two connections of one session | `adapter:aer:sessions`, a new test |
| The Qiskit adapter says `resumed` for a session it never saw | The client does nothing with that on its first connection, and no test started the Qiskit adapter again | The same |
| The client talks to an adapter of another user | It takes a second account to listen | None: not tested |

- **The second pass** (snapshot 4, 34 tests, histamine3): the nine again. Eight now fail one test each, the one named above. The 52 others were not run again; the tests gained checks and lost none, and the 34 pass with no bug.
- **One test holds most of them.** Of the 60 that are noticed, 37 fail one test only: 19 `runtime/qpu/adapter`, 9 `runtime/qpu/qasm`, 5 one of the two tests of the script, 2 `adapter:aer:sessions`, 1 `adapter:badlist` and 1 `adapter:multi:two`. The six bugs in the Qiskit adapter are noticed only by the tests with Aer, so only where Qiskit is installed.
- **What this does not show.** The list was written by the author of the code, after the code, so it holds the mistakes that were thought of. The six defects of the review (below) were in code whose tests all passed.

### Sanitizers

Builds without MPI, as for the earlier milestones (gcc 12.4). Run on snapshots 1 and 3 on serotonin, and on snapshot 4 on histamine3; the numbers are those of snapshot 4.

- **Address**: every QPU and QAOA test of that build, 155 (22 of M4a). No report.
- **Undefined behaviour**: one report, the same in 130 of the 155 tests, and it is the runtime's: `parsec/parsec.c:2057`, a member read through a pointer that is not aligned. No report has a file of the component, of its tests or of the application in its place, and none has a frame of the component in its stack.
  - **What it is.** In a Debug build `parsec_get_copy_reshape_from_dep` (`parsec_reshape.c:753`) prints the task into a string for its messages, and `parsec_task_snprintf` reads `data_in->original->key` of every flow. For the flow whose dependency made the task ready, `data_in` holds what the predecessor's repository entry holds (`parsec.c:1887` and `:1840`), which at that point is a reshape promise and not a data copy (`:1953` casts the same value to one). What is taken for `original` is an address in the code of the library. It is read, for a string, and nothing is written. Both files are as in `09c721868`. Not reported upstream.
  - **What the runs on snapshots 1 and 3 could not show.** Their script counted the reports in ctest's log, which keeps the output of failed tests only, and no test had failed: a report in a test that passes went uncounted, and the count was 0. The outputs of the runs by hand on snapshot 3 hold this report in 11 of 15, so it was there. Snapshot 4 was run again with all output kept (`status_san_4b`), which is where the counts above come from.
- **One test fails in that build on histamine3**, `overlap:1worker`: 154 of 155. It is the timing test of M1, with the mock: the run has to be shorter than the job and the classical chain one after the other, 500 ms, and takes 500 to 512. The time is the sanitizer's: it prints the report above with its stack, on the only worker thread, before the job is submitted. Alone, 40 times each way: 34 failures with the stack printed, none without it (`UBSAN_OPTIONS=print_stacktrace=0`). On serotonin, snapshots 1 and 3, it had passed.
- **Leaks**: 15 runs by hand (the two tests without the runtime, 12 runs of the runtime through the adapter of the tests, the client against Aer). No error. Leak records in one run, the DTD chain: 6, in DTD's tiles and in the task mempool, where section 9.4 found them for the same test with the mock. None has a frame in the component.
- **The two readers, with texts that are wrong.** They read what comes from outside the process, so they were given 2,000,000 OpenQASM texts and as many JSON lines, each a valid one with one to four random edits, in buffers of exactly their size, under the address and undefined-behaviour sanitizers (`exp_qpu/checks/m4a/fuzz`). No report. Of the OpenQASM texts 89,826 were accepted: each is a well-formed circuit, and each is the same bytes after it is written and read again. The 1,898,459 that were refused each have a line and a reason.
- **Threads**: the same 15 runs. No report is located in the component. The reports are the runtime's own, as in every milestone (`parsec.c:1820` and `:1831`, the LIFO, the generated start-up code); none has a frame in a file of M4a.

### What an independent review found

A reviewer that had the sources (snapshot 2) and none of the reasoning. It changed nothing and ran experiments of its own, which are kept in `exp_qpu/checks/m4a/review`. It found six defects and four tests that could not fail or missed their point. All ten are fixed, and the checks of this section were run again on the result (snapshot 3).

| Finding | What was done | The test that now holds it |
|---|---|---|
| A job given up because the adapter was out of reach for too long was never cancelled at an adapter that came back. It ran on while the runtime ran the task again: twice, and on a device paid twice | Such a job is told to the adapter (`cancel`) when it answers again. Only the jobs of an adapter that was replaced are not, since nobody is left to tell | `runtime/qpu/adapter`: the adapter is stopped for longer than the time-out and continued. Afterwards it holds no job, and the end is not reported a second time |
| An adapter that is started again numbers its jobs from the first again. Its first job took the place of a lost job that was not reported yet: one task never completed, and a result that was not taken yet could be taken from another job | The client names the jobs itself and never uses a name twice. The adapter's name is kept beside it, and is not looked at for a job whose adapter is gone; the result of such a job is an error, and the adapter is not asked | Two cases of the same test, with an adapter of the tests that numbers from the first again as a new process does |
| `connect()` waited without limit on an adapter that accepts nothing and whose queue is full. No wake-up reaches there, and the end of the run would wait with it | The socket does not block: a full queue is an error at once | The same case: the stopped adapter has room for one connection to wait, and no wait of 100 ms may take 1.5 s |
| The Qiskit adapter dropped a session in use: the thread of an old connection marked it as left after the client had connected again, and ten minutes later every job of that client was gone | A session counts its connections and is left when it has none | `adapter:aer:sessions`, added when the deliberate bugs showed that no test held it (above) |
| The Qiskit adapter wrote to the pipe of a session after closing it; the number could be another file's by then | Written under the lock, and not after the session is closed | None |
| The OpenQASM reader followed signs and parentheses one call each: 400,000 of them in an angle overran the stack | Signs are counted in a loop, and an angle may have 32 parentheses inside each other | `qasm`: 400,001 signs are read as one; 32 parentheses are read, 33 and 400,000 are refused |
| `qpu_guard` checked that the client had submitted no job from counters that were gone with the client: it could not fail | Taken out, with a comment. The adapter of the tests does that check (`-N`) | |
| The adapter of the tests kept its job numbers when it "forgot", and no test had an adapter come back: the first two defects could not show | It numbers from the first again; the cases above | |
| In the test with a simulator of its own, the Pauli gates stood right before the measurement: "y read as x" and "z left out" changed no probability | The circuits have them in the middle, and a layer of rotations before the measurement. The test now checks its own circuits against 25 such mistakes: each has to move the outcomes of one of them by 0.15, and the smallest moves them by 0.32 | `adapter:stat:circuits`, in every build |
| The third check of `no_hardware_in_tests` accepted three definitions the reviewer made up: another adapter program, `--backend aer2`, and a backend named on the line after `--backend` | It reads a file as one text, and checks which program a test starts, the whole name of the backend, and the device | `no_hardware_in_tests:itself` gives the script definitions that each break one rule (11 then, 12 now) and one that breaks none |

- **Smaller points, changed.**
  - The client logs nothing an adapter sent. It logged the reason of a refusal, which the Qiskit adapter fills with Qiskit's words, and those can quote a circuit.
  - Numbers are written and read with a point whatever the locale of the process. This could not be tested: methane has no locale that writes a comma.
  - The client talks only to an adapter that runs as the same user (`SO_PEERCRED`), and the README's example puts the socket in a directory of the user's own: the client connects again by itself to whatever listens at the path.
  - The times of a result cannot lie in the future. A 64-bit value has to be hexadecimal digits and nothing else. The list of ends heard early grows, and an end is taken out when its job is abandoned.
  - The Qiskit adapter removes only a socket at its path, and answers a request of an unexpected shape where it ended the connection.
  - `qpu_adapter_stop()` refuses the process numbers 0 and -1.
- **Left as it is.** A request to the adapter, the first `hello` and the cancellations after a new connection included, cannot be ended by a wake-up: a QPU's thread can sit in one for `device_qpu_adapter_timeout` when the adapter is there and silent. The README says so. The seed the Qiskit adapter gives Aer has 31 bits.
- **What the fuzzing did not find**: the stack overrun. Its texts are valid ones with a few edits, which never nest that deep.
- **Looked at and found sound**, in the reviewer's words: the line buffer of the client and its growth; that no request is sent while an answer is still read; the hardware guard in the client (hello first on every connection, a missing answer taken for a device, a changed answer ends everything before anything is sent); no access to the environment or to a file in the client; the JSON reader's bounds and depth; the OpenQASM lexer's bounds, the angle arithmetic and the round trip; that no wake-up is lost; that an early end is held back; the Qiskit adapter's fields against the README and the client; that the launcher reaps what it starts.

### The checks that were left, what a job costs through the adapter, and three installs (2026-11-28)

Run in the night after M4a was committed, with simulators only: the mock, the adapter of the tests, and Qiskit Aer. No provider was contacted. Every job built from `git archive` of the committed head, `e61386550`. No source file changed: this subsection, with the pointers to it elsewhere in this document, is the only edit. The scripts, status files and logs are with the harness, in `exp_qpu/checks/m4a_night/`, and `REPORT.md` there is the account of the night.

**With CUDA.** leconte, three V100 that read each other directly (devices 4, 5 and 6; Slurm job 68282, beside another user's job, on its own CPUs and GPUs). Debug with `PARSEC_DEBUG_PARANOID`, MPI and CUDA, the tests with Qiskit Aer registered, one GPU per rank.

- **The suite, 461 tests: GPU tests 69 of 69, every other test 392 of 392**, none skipped. No compiler warning in the component or its tests.
- **The 27 tests of the adapter, of OpenQASM 3 and of the guard script, five times each:** all pass.
- **By hand, with a CUDA device up in the same context** (ctest gives a test that is not a GPU test `device_cuda_enabled=0`):
  - `cpu_gpu_qpu -g`, the quantum task between two accelerator tasks, with its job through the adapter backend: passes, 16 tasks in 16 jobs. With an adapter between two mocks: passes, and the three QPUs ran 7, 1 and 8 tasks.
  - The application through the adapter of the tests, on 1 and on 3 ranks with a GPU each: final states equal to the reference, 256 jobs.
  - The application on Qiskit Aer, on 1 and on 3 ranks: it runs (2,046 tasks in 1,634 jobs; nothing is compared).
- No adapter process was left.

**Two nodes.** histamine3 and caffeine (AMD and Intel, Ethernet), a Debug build in the shared home, the ranks of every test on the two nodes in turn, and Open MPI over TCP as in section 9.8 (Slurm job 68285).

- **The QPU and QAOA tests: 200 of 201.** The one that fails has to fail there: `adapter:chain:mp:last`. A test's adapter is started on the node ctest runs on; this test has its QPU on the last rank, which is on the other node; and a Unix-domain socket is reached from its own node only. The rank says "the adapter at runtime/qpu/a_mplast.sock is out of reach (nothing listens there)", its QPU does not open, and the quantum tasks fail. This is the rule the README gives: the adapter runs on the node of the rank that owns the QPU.
- **The 39 other tests that go through `mpiexec`, 10 times each:** all pass.
- **By hand, the adapter on the node of the rank that owns the QPU** (Slurm job 68290, which kept the exit code of every adapter: 0 says it saw nothing wrong in what its client sent):

  | Run, 4 ranks | Result | The adapters |
  |---|---|---|
  | The chain test, the QPU on the last rank, which is on the other node, and the adapter of the tests there | Passes: 72 tasks in 72 jobs | 0 |
  | The application, a QPU on rank 0 and one on rank 1, each with an adapter of the tests on its own node | Final states equal to the reference; 1,116 tasks in 900 jobs on one QPU, 930 in 734 on the other | 0 and 0 |
  | The application on Qiskit Aer, the Qiskit adapter on the other node, the QPU on rank 1 | Runs: 2,046 tasks in 1,634 jobs | 0 |

- **The sockets were files in the shared home** (NFS). Binding and connecting work there. From the other node such a socket is seen as a file, and cannot be connected to.
- Not run: the 167 other tests (M4a changes nothing outside the component, and section 9.8 has them on two nodes), and Open MPI's UCX path, which fails on these nodes at the base commit.

**The experiments.** serotonin (AMD Ryzen 9 7950X, 16 cores), exclusive allocations (Slurm jobs 68289, 68292 and 68293); a Release build. 822 runs, none failed. Tables are in `exp_qpu/tables/` (`e10`, `e10b`, `e11`, `e12`, `e12b`), from `exp_qpu/analysis/analyze_adapter.py`.

**E10: what a job costs through the adapter backend.** The chains of `qpu_bench` with no classical work, as in E1. C is the mock backend in the process; A is the adapter backend with the adapter of the tests, so every job crosses a Unix-domain socket twice and its circuit is read as OpenQASM 3; both answer from the same sampler. 12 runs per setting in rotating order, 456 runs, none failed. Microseconds per link, medians; the difference is taken per repetition.

| Workers | One chain: C | A | A - C | 64 chains: C | A | A - C |
|---|---|---|---|---|---|---|
| 1 | 57.9 | 60.7 | 2.8 | 2.9 | 27.7 | 24.8 |
| 2 | 39.5 | 63.9 | 24.8 | 4.6 | 29.4 | 25.0 |
| 4 | 30.4 | 63.1 | 32.6 | 3.6 | 28.5 | 24.9 |
| 8 | 24.1 | 61.7 | 37.5 | 2.6 | 27.6 | 25.0 |
| 14 | 21.2 | 62.3 | 41.1 | 2.5 | 28.5 | 25.9 |

- **A job through the adapter takes 61 to 64 us on a chain of its own, whatever the number of workers,** where the mock takes 21 to 58. In throughput it costs 25 us more than a job of the mock (28 against 3), with every number of workers: about 35,000 jobs a second through one adapter.
- **A result of 1,000 shots** (4 workers): 52 us more than the mock on one chain, and 31 with 64 chains.
- **A job that takes 2 ms** (4 workers, one chain): the link takes 97 us more than the job with the mock, and 167 us with the adapter. With 64 chains the adapter costs 13 us more per task.
- **What it says.** The protocol, the JSON and the socket cost tens of microseconds per job. That is 6 to 11 times the mock's cost of a job, and nothing beside a job of milliseconds or seconds.
- Not measured: several QPUs on one adapter, and jobs that carry several circuits.

**E10b: the Qiskit adapter beyond its simulation.** `qpu_bench` through the adapter of `tools/qpu` (the ideal simulator, 100 shots, 4 workers), with the timers of E11 below; 6 runs each. On one chain a link takes 2.94 ms. Of that, 0.55 ms are the job in its thread (0.41 ms of them the simulation), and 2.23 ms are the reading and transpiling of the 9 circuits of the run, spread over its 300 jobs. What is left, 0.16 ms a job, is the protocol, the adapter's own Python, the client and the runtime, against 0.06 to 0.11 ms for a whole link with the adapter of the tests (1 and 1,000 shots). With 64 chains a task takes 1.14 ms; the sums of the two job threads overlap there, and the same split cannot be made.

**E11: where the time goes when the jobs run on Qiskit Aer.** The application on `ring64` (8 blocks of 8 qubits, 256 shots) without the reference, through the adapter of `tools/qpu` with its defaults, started through `exp_qpu/aer_timed_adapter.py`, which loads the adapter's module from the source tree and puts timers around four of its calls. Three runs each, 30 runs, none failed. Milliseconds, medians; the columns after "the run" are sums over the calls inside the adapter.

| Simulator | Layers | Jobs | The run | Reading the text (`qasm3.loads`) | Transpiling | Simulating | Per circuit: reading | transpiling | Per job: simulating |
|---|---|---|---|---|---|---|---|---|---|
| Ideal | 1 | 128 | 1,456 | 888 | 398 | 264 | 74 | 36 | 2.1 |
| Ideal | 1 | 512 | 1,990 | 899 | 399 | 898 | 75 | 36 | 1.8 |
| Ideal | 2 | 128 | 2,107 | 1,501 | 397 | 326 | 131 | 36 | 2.5 |
| Ideal | 2 | 512 | 2,808 | 1,538 | 407 | 1,134 | 133 | 37 | 2.2 |
| Device model `generic:8:line`, without its noise | 1 | 128 | 1,250 | 880 | 159 | 384 | 74 | 14 | 3.0 |
| | 1 | 512 | 1,871 | 889 | 144 | 1,162 | 74 | 13 | 2.3 |
| | 2 | 128 | 1,980 | 1,542 | 148 | 496 | 134 | 13 | 3.9 |
| | 2 | 512 | 2,799 | 1,506 | 146 | 1,758 | 130 | 13 | 3.4 |
| With its noise | 1 | 128 | 11,371 | 1,327 | 35 | 17,646 | 119 | 3 | 138 |
| | 2 | 128 | 19,772 | 2,198 | 39 | 30,530 | 200 | 4 | 239 |

- **The application has 11 different circuits**, with 128 jobs and with 512: its circuits take their angles as parameters, so each is read and transpiled once.
- **On the ideal simulator most of a short run is the reading of those 11 texts.** `qiskit.qasm3.loads` takes 74 ms for a circuit of one layer (8 qubits, 12 two-qubit gates) and 131 ms for one of two layers; the first call takes 140 to 200 ms. Reading and transpiling are 1.3 s of the 1.5 s of a run of 128 jobs. A simulation takes 2 ms.
- **With noise the run is simulation**: 138 ms per job with one layer and 239 ms with two, where the transpiled circuits have 53 and 111 two-qubit gates on the line. The sums of that row are above the time of the run because the two job threads take turns (next paragraph): a call of one waits while the other computes, and both count the time.
- **The 5.5 s** that "Measured with Aer" gives for 192 jobs were measured on methane, a machine in use. Where the time of such a run goes is now measured for 128 and 512 jobs: above.

**Two simulations do not run at once in one adapter.** `exp_qpu/aer_gil_check.py`, on the same node: two simulations one after the other, the same two in two threads, and a thread that only counts while one simulation runs. Aer has one thread per simulation, as the adapter gives it.

| Circuit | One after the other | In two threads | The counting thread, of its rate alone |
|---|---|---|---|
| 8 qubits, ideal, 200,000 shots | 0.322 s | 0.319 s | 8% |
| 16 qubits, ideal, 200,000 shots | 0.528 s | 0.529 s | 4% |
| 8 qubits, noise, 256 shots | 0.199 s | 0.201 s | 19% |
| 10 qubits, noise, 256 shots | 0.829 s | 0.852 s | 4% |
| 10 qubits, noise, density matrix | 1.489 s | 1.482 s | 3% |

- **Two threads take as long as one**, and a thread of Python hardly runs meanwhile. So Aer 0.18.4 keeps the interpreter's lock while it simulates (concluded from these times; its source was not read).
- **What that means for the adapter of `tools/qpu`.** `--workers`, 2 by default, does not make two jobs run at the same time. And the thread that answers a client waits behind a simulation: with jobs of 138 ms, the reading of a circuit took 119 ms where it takes 74. Nothing is wrong in what it answers. An adapter that is to simulate several jobs at once needs a process per job, or Aer's own batch of circuits.
- **A noisy evaluation gets slow fast.** On methane, a machine in use, one of 256 shots took 0.3 s at 8 qubits, 1.3 s at 10 and 7.3 s at 12 with the method Aer picks. With `--method density_matrix` it took 0.1 s at 8 qubits, and a run of 32 jobs at 12 qubits did not end in 500 s. This is why E12 stops at 10 qubits.

**E12: the application with its jobs on a noisy device model.** The same optimisation, with the same instance and the same seed of the initial assignment and of the optimiser, on four samplers:

- `mock`: the exact sampler of the component, in the process;
- `aer`: Qiskit Aer through the adapter, the ideal simulator;
- `aer-map`: Aer as the device model `generic:N:line` without its noise, so that the circuits are transpiled to its gates and to a line;
- `aer-noise`: the same with its noise. The error rates are made up: it is not the model of a device.

`ring64` (8 blocks of 8 vertices) with one and two layers, 5 seeds; 8 blocks of 10 vertices with one layer, 3 seeds. T 8, R 4: 512 jobs of 256 shots. 52 runs.

- **The cut after is the same on the four, in all 13 settings**, and so is the number of updates accepted.
- **Why.** An update of a block takes the best assignment that any of its jobs measured, when it is better than the one the block has. An update has 16 jobs of 256 shots: 4,096 measured assignments of a block that has 256 of them, or 1,024. The best one is among them whatever state they are drawn from. So the cut after cannot tell a noisy sampler from an exact one at these sizes, and it does not tell one layer from two either.
- **E12b: the same with few shots a job** (`ring64` with 8 seeds, the instance of 10 vertices with 5; one and two layers; 272 runs). Settings in which a system ends in the cut that `aer` ends in:

  | Shots a job | Settings | `mock` | `aer-map` | `aer-noise` |
  |---|---|---|---|---|
  | 4 (`ring64` only) | 16 | 10 | 16 | 9 |
  | 16 | 26 | 25 | 26 | 25 |
  | 64 | 26 | 26 | 26 | 26 |

  - **With 4 shots a job the runs differ, and noise changes the outcome about as often as another exact sampler does**: 7 settings of 16 against 6. The cut after is then -2.00 to +1.75 from that of `aer` with noise, and -0.38 to +1.75 with the mock, at cuts of about 104.
  - **`aer-map` ends in the cut of `aer` every time.** That is what the same seed and the same outcome probabilities would give; it was not looked into further.
- **What E12 shows.** The path works end to end with a device model: 512 jobs written by the component, read and transpiled to a line by Qiskit, simulated with noise, and the application ends where it ends without noise.
- **What it does not show: what noise does to QAOA.** The result of this application is decided by its accept rule, not by the quality of the states. A measure that would see the noise is the estimate each evaluation gives the optimiser (`mean` of `qaoa_eval_t`), which the application computes and writes nowhere.
- **The time of a run** of 512 jobs of 256 shots: 0.3 s on the mock; 2 to 3 s on Aer without noise; with noise 43 s and 74 s on 8 qubits (one and two layers), and 183 s on 10 qubits.

**Three installs: can M10 and M11 start?** Section 10.6 has them wait for an installation. A separate pass tried the three pieces of software on methane, in a directory outside the repository (`/home/qcao3/qpu_m10_m11_probe`, 7.8 GB), with no account and no provider. Its report, with what was run and what was only read, is `exp_qpu/checks/m4a_night/probe_m10_m11.md`; the main points were run again afterwards.

| | What was run | What it found |
|---|---|---|
| **CUDA-Q** 0.17.0, from PyPI (`pip install cudaq`, 1.8 GB) | A circuit built gate by gate at run time, with angles that are a number or a number times a parameter, sampled on the CPU and on an A100; the same from C through a C++ wrapper, with no Python in the process; asynchronous sampling on a QPU chosen by number | It works. The builder has no rxx, ryy or rzz: they are written with the three definitions of this section. A result string has qubit 0 on the left. The seed is set for the process (`cudaq.set_random_seed`); `sample` and `sample_async` take none of their own |
| CUDA-Q, several QPUs | `nvidia` with `mqpu` on two A100 of guyot (Slurm job 68291) | Two QPUs, one per GPU; both sample, and give the same counts for the same seed |
| CUDA-Q on leconte | The same on two V100 (Slurm job 68288), with the CUDA 13 and the CUDA 12 packages | It fails: `[custatevec] architecture mismatch`. The install guide asks for compute capability 7.5; a V100 has 7.0 |
| **Qiskit C API** 2.6.2, built from source (`make c`, 6 minutes, Rust 1.89) | A C program: a circuit with a symbolic parameter, a target of `cx, id, rz, sx, x` on a line built in C, `qk_transpile`, `qk_qpy_dump_file`; the file loaded by Python Qiskit 2.6.2 | It works, and the parameter survives the transpilation. No function of the C API gives a parameter its value, and none reads or writes OpenQASM. `libqiskit.so` needs a `libpython` to load (3.12 here), and starts no interpreter |
| **QRMI** 0.28.1, built from source (2.5 minutes) | The header (47 functions); its C examples for IBM compiled, not run; a 2-qubit job through the C API on the kind `MAESTRO_LOCAL`, a simulator daemon on a Unix-domain socket | It works without an account, with two more programs from the daemon's vendor (GPL-3.0). A task's status is read without waiting |

What it means for the milestones, as far as the facts go:

- **M11 does not wait for an installation any more**, on a machine with an A100 or on CPUs. Its table can be written. Three facts will shape it:
  - **The result of `sample_async` has one call, `get()`, which blocks.** There is no test for "ready", no time-out and no wait on several. A backend's `wait` then needs a thread per job in flight.
  - **CUDA-Q is C++.** The wheel has headers and libraries and no C interface, so the backend is a C++ file, or a second adapter program in Python.
  - **A job has no seed of its own**, where the mock and the Qiskit adapter draw from the key of an evaluation. Whether CUDA-Q's counts can be made to depend on the key alone, with jobs on several QPUs at once, was not tried.
- **M10.**
  - **The first of section 10.4's two open points is answered by reading**: QRMI's own Python client puts the circuit into the input of the Sampler as OpenQASM 3 text (`python/qrmi/primitives/base_sampler.py`), and QPY occurs nowhere in its repository. Whether the service takes QPY there was not determined. The component writes OpenQASM 3 since M4a.
  - **The C API cannot give a parameter its value, and may not have to.** In the input QRMI's client writes, an entry holds the circuit, an array of parameter values, and the shots: the values travel beside a circuit that stays symbolic.
  - **There is a way to run a QRMI job end to end without an account** (`MAESTRO_LOCAL`), at the price of a daemon and two GPL-3.0 programs on the test machine.
  - Still open: the second point of section 10.4 (the input of the program that replaces the server-side Sampler), and everything that needs a service.
- **M4b is as it was**: it waits for an account.

**What this night does not show.**

- A quantum device, a provider, or QRMI against a service.
- More than two nodes, and a network faster than Ethernet.
- The cost of the Qiskit adapter's own protocol work under load: E10 is the adapter of the tests.
- The noise of an existing device: the device model's error rates are made up.
- Any Qiskit but 2.6.2 with Aer 0.18.4, and any CUDA-Q but 0.17.0.
- CUDA-Q's GPU simulators on leconte, which they do not support.

### Departures from the plan

1. **Inputs are declared `float[64]`, not `angle`** (sections 2.6 and 10.4 said `angle`). It is what Qiskit's own writer declares, and an angle times a real number, which the circuit format needs, is a product I did not find defined for the `angle` type of OpenQASM 3 (from memory of the specification; not checked against it in this pass). The reader takes both.
2. **The adapter is `parsec_qpu_adapter_qiskit.py`**, not `parsec_qpu_adapter_ibm.py` (section 7.3): it holds what is common to Qiskit, and M4b adds IBM's service to it.
3. **Reading OpenQASM 3 in the adapter needs a third package**, `qiskit-qasm3-import`: `qiskit.qasm3.loads` does not work without it, and Qiskit's other reader (`loads_experimental`) refuses a gate that the text defines.
4. **The protocol** gained the session, the key and attempt of an entry, `wake`, `abandon` (whose answer names the job it dropped), and the repetition of ends after a new connection (above). **A job has two names**: the adapter's, and one of the client's own that the service sees (the review, below). `prepare` names a circuit by a digest of its bytes; the client keeps a second digest and refuses a job whose circuit shares a name with another.
5. **A Unix-domain socket only.** Section 2.6's TCP mode with a token is for an adapter on a login node that reaches a provider, which is M4b's.
6. **The test of the client is `runtime/qpu/adapter`** and the adapter of the tests is a program of its own (section 4.2 named one test, `adapter_loopback`, that would do both).
7. **A result of a simulator behind an adapter carries the flag `PARSEC_QPU_RESULT_MOCK`**, whose comment said "from the mock backend". The flag keeps simulated results apart from a device's, which is what section 2.5 wanted it for.
8. **The shared checks of the tests** know the adapter: a run with it is compared with the client's counters where a run with the mock is compared with the mock's, and a test whose adapter loses jobs on purpose says so (`QPU_TEST_ADAPTER_UNSTEADY`).
9. **Not built:** IBM hardware (M4b); the TCP mode; reading QPY; a reader for circuits with more than one register; an adapter that prepares circuits in the background (`prepare` holds the progress thread for the time of a transpilation).

### For later milestones

- **M4b** adds a backend to the adapter program, and nothing to PaRSEC: the client, the protocol and the guard are there. What it has to settle: the provider's job tags against the session and tag of the protocol; cancelling at the provider what the client gave up on while the adapter could not be reached; the rate limit, for which `device_qpu_submit_interval` exists; and the flag without which the adapter does not start.
- **A circuit is prepared in the call that submits it**, so a job waits for the transpilation of its circuit. With a device model, a QAOA circuit of 16 qubits takes Qiskit some tens of milliseconds. Measured since, for 8 qubits (E11): reading the text takes 74 to 131 ms, and transpiling 13 to 37 ms. Making it a task of its own is what section 3.4 foresaw.
- **M10** can use the OpenQASM 3 writer as its test of what it lowers, and the reader for circuits an application brings.

---

# 10. Design revision: several QPUs, NVQLink, Qiskit (2026-11-26)

Asked for after M8: state the deployment assumption for QPUs coupled to GPU systems through NVIDIA NVQLink, let a context have several QPUs, and say how Qiskit is supported. Nothing in this section was built when it was written. Section 10.3 has been built since, as M9: section 9.8 has what was built, and the places where this text was changed by it are marked.

- **No deadline scopes this section.** Each choice is made on its merits, and the milestones are ordered by what they depend on.
- **Source references** are to the branch head, `80d6aa45d`, not to the base.
- **Other products.** Facts about them come from their documentation and public repositories, read on 2026-11-26 (sources at the end of the section). They are their vendors' claims, or readings of their code. None was run here. What could not be confirmed is marked.

The project facts and sections 1.4, 2.1, 2.2, 2.6, 2.7, 6.7, 7.3 and 7.4 point here where this section replaces what they say.

## 10.1 What changes

| Before | Now |
|---|---|
| One QPU per run, on `device_qpu_owner_rank` (2.7) | Any number of QPUs. Each has an owner rank, a rank may own several, and the ranks that own none keep a placeholder module (10.3) |
| One progress thread serves all QPU modules of a process (2.2) | One progress thread and one backend per QPU. The QPUs of a rank share one set of waiting tasks (10.3) |
| A task is bound to a QPU when the device layer selects it, and `device.c` changes so that two QPUs are balanced (1.4, 2.7, 2.8) | A task is bound to a QPU when that QPU has a free place for it. The device layer does not change (10.3) |
| Where the owner rank runs: wherever the provider can be reached (project facts, 2.6) | For an NVQLink-coupled QPU, on its real-time host; PaRSEC stays out of the real-time loop (10.2) |
| IBM through a Python adapter on the Qiskit SDK (2.6, M4) | Two routes: that adapter, with Qiskit Aer first, and a backend in C on the Qiskit C API and QRMI. Also an OpenQASM 3 reader for circuits written in Qiskit (10.4) |

## 10.2 Deployment assumption: QPUs coupled through NVQLink

**What NVQLink is, as NVIDIA describes it [N1][N2].**

- **Three parts.** The architecture names three parts:
  - a *real-time host*, a GPU server that runs CUDA code;
  - a *quantum system controller*, which does the coherent control and the readout of the QPU;
  - a *real-time network* between the two. The reference network is RoCE (RDMA over Ethernet), and a vendor may use its own if it supports the `cudaq-realtime` API.
- **Speed.** The product page gives a minimum round trip of under 4 us (controller to GPU to controller) and 400 Gb/s. The architecture blog says NVIDIA does not prescribe a latency, because QPUs and the work offloaded to the host differ. Instead it defines a benchmark that measures the round trip of a controller-to-host callback.
- **What it is for.** The stated uses are calibration and decoding for quantum error correction.
- **Who uses it.** Quantinuum announced that its future processors deploy with NVQLink, and its Helios QPU uses a GH200 as its real-time host [N2].
- **Ising** is a set of open AI models for the same two jobs [N3][N4]:
  - a 35-billion-parameter vision-language model for calibration;
  - small 3D convolutional decoders, of 0.9 and 1.8 million parameters, for real-time decoding.

  They work with CUDA-Q and NVQLink.

**Assumptions of this design.**

- **A1. Two time scales, two layers.** The real-time loop never passes through PaRSEC. That loop is decoding each error-correction cycle and feeding calibration back, and it runs on the real-time host under CUDA-Q.
  - PaRSEC's unit is the job: a circuit, physical or logical, with its shots, or several circuits in one job. A job lasts milliseconds or more.
  - A quantum task costs 37 to 65 us of latency in PaRSEC (E1, section 9.7). That is about ten times NVQLink's whole round trip, so putting PaRSEC in that loop would be wrong by design, not a matter of tuning.
  - When circuits are shorter than PaRSEC's cost per task, they belong in one job. The device already forms such jobs under contention (section 2.8). An application that issues many short circuits should also give one task several circuits.
- **A2. A QPU belongs to a node.**
  - The owner rank of an NVQLink-coupled QPU runs on that QPU's real-time host, or on a node of the same system that reaches it. Which of the two a site allows was not found in NVIDIA's pages.
  - With several QPUs, each has its own owner rank. One rank owns several QPUs when its node hosts or reaches several. Section 10.3 makes both possible.
  - On such a system the project fact "can compute nodes reach the provider" has an answer by construction: the QPU is on the cluster's own network.
- **A3. The real-time host's GPUs belong to the decoder.**
  - While its QPU runs logical circuits, the rank that owns an NVQLink-coupled QPU should not run PaRSEC's GPU tasks on the host's GPUs.
  - This is a deployment rule, not code: `device_cuda_enabled=0` on that rank, or a mask of its GPUs.
  - Not verified: how much GPU capacity decoding leaves free. NVIDIA's pages do not say.
- **A4. The backend for such a QPU is CUDA-Q.**
  - The backend builds a kernel at run time from the native circuit (`cudaq::make_kernel`) and samples it with `cudaq::sample_async`. That call takes the index of the QPU, `qpu_id`, and returns a result object [N7].
  - The result object has one call, a blocking `get()`. It has no timed wait and no test for readiness [N7]. So the backend waits for each job on a helper thread of its own, which wakes the progress thread, as the mock's helper threads do.
  - CUDA-Q is C++. It would be an optional dependency of one backend file, built only on request, like any provider SDK (section 7.5).
  - The same backend serves CUDA-Q's multi-QPU platform, where every GPU of a node is a simulated QPU chosen by its index [N5][N6]: one PaRSEC QPU per CUDA-Q QPU index.
  - Not verified: that a kernel built at run time is sampled on a physical QPU the same way as on a simulator.
- **A5. Nothing else changes in the runtime.** To the runtime, an NVQLink-coupled QPU is a QPU device with a backend, as the mock is. NVQLink changes where the owner rank runs and how short a useful job can be.

This design does not assume that the runtime sees qubits, pulses or syndromes.

## 10.3 Several QPUs per context

### The cases

1. **One process reaches several QPUs.** Examples:
   - a node that hosts or reaches more than one QPU;
   - CUDA-Q's multi-QPU platform, one simulated QPU per GPU of the node (`nvidia` target, `mqpu` option [N5]), which is the multi-QPU setting most users meet first;
   - several cloud devices, of one provider or of two.
2. **QPUs on several ranks.** One per NVQLink host, under assumption A2.
3. **A mix of the two.**

All three are served by one mechanism.

### Where the code assumes one QPU

| Where (`80d6aa45d`) | What |
|---|---|
| `device_qpu_component.c:217-219` | `device_qpu_enabled` above 1 is warned about and treated as 1 |
| `device_qpu_module.c:52`, `:154-179` | One module, kept in the global `parsec_qpu_module` |
| `device_qpu_module.c:76-99` | One owner rank, `device_qpu_owner_rank` |
| `device_qpu_progress.c:184` | One service `svc` per process: one backend, one progress thread, one heap of waiting tasks, one set of counters |
| `device_qpu_progress.c:1099`, `:1110` | `parsec_qpu_submit` uses the global module and does not read `task->selected_device` |
| `device_qpu_progress.c:596-600` | A task whose circuit is too wide for the backend, or asks for too many shots, completes as rejected at once. With several QPUs that is right only when no QPU of the rank can run it |
| `device_qpu_progress.c:188` | One reason why quantum tasks cannot run, for the whole process |
| `device_qpu_backend_mock.c:837-843`, `:174`, `:178-191` | The mock's `open` ignores its device index and reads the process-wide parameters. The mock keeps one current backend for its statistics, and its two test hooks are process-wide |
| `parsec_internal.h:132` | A taskpool's device mask is 16 bits, so a process has at most 16 devices of all kinds. The CPU takes one, and so does the recursive device in a build with it (`device.c:1121-1148`), so a process with 8 GPUs can have at most 6 QPUs. Nothing checks this where a device is added (`device.c:1182-1200`) |
| `device.c:300` | Once a first device is picked, a device that is not a GPU is skipped unless `device_load_balance_allow_cpu` is set. So of several QPUs the device layer always picks the one with the highest index. Section 1.4 found this at the base, at `:299`. This design leaves it as it is (see "Binding a task to a QPU") |

Two things are ready already. `parsec_qpu_device_info` takes a QPU index (`device_qpu.h:395`), so far only 0. A backend's `open` takes a device index (`device_qpu_backend.h:75`).

### Configuration

The MCA parameter `device_qpu_devices` lists the QPUs. The list is the same on every rank. Entries are separated by `;` (as built, also by `|`), and each entry is `backend[@rank] [key=value ...]`. For example:

```text
--mca device_qpu_devices "mock@0 slots=4; mock@0 slots=2 delay=exp:20000; qrmi@3 resource=<name>"
```

- **Numbering.** A QPU's number `q` is its position in the list, the same on every rank. That is how an application names a QPU.
- **Rank.** `@rank` defaults to `device_qpu_owner_rank`, and `@-1` is the last rank.
- **Keys.**
  - Limits per QPU: `max_outstanding`, `max_batch`, `submit_interval`, `job_timeout`, `max_retries`, `unavailable_timeout`. Each defaults to the MCA parameter of the same name.
  - The backend's own keys. For the mock: `slots`, `delay`, `queue_delay`, `job_delay`, `max_circuits`, `faults`, `threads`, `max_qubits`. For QRMI: the resource name.
  - No key carries a credential (section 2.6).
- **An empty list** means `device_qpu_enabled` copies of `device_qpu_backend@device_qpu_owner_rank`. Every existing command line and test therefore means what it meant, except that `device_qpu_enabled` above 1 now gives that many QPUs, which is what its description already says.
- **Seeds.** The mock's seed of the samples stays global. The samples of an evaluation then do not depend on which QPU ran it, since they depend on (seed, key) only (section 9.2). Every byte-for-byte check of sections 4.5 and 9.5 therefore stays valid with any number of QPUs, whichever QPU runs each task. The times may differ per QPU, which is what QPUs of different speeds look like.
- **The hardware guard** applies to every entry: an entry whose backend drives a quantum device opens only with `device_qpu_allow_hardware`.
- **Errors.** A list that does not parse disables the QPU layer with a message on rank 0, as an unusable backend does now. So does a list that would give a rank more devices than the 16-bit mask holds. (As built: a list holds at most 16 QPUs, and a QPU that would be the 17th device of its process is left out with a warning; the others stay.)

### Modules and services

- **Modules.**
  - The component's query gets no context (`device_qpu_component.c:240-256`), so it makes one module per entry of the list on every rank.
  - Each module decides at attach, where the rank and the number of ranks are known (`parsec.c:814`, `:909-913`). On the entry's rank it opens the backend and adds a device of type `PARSEC_DEV_QPU`, named `qpu<q>(<backend>)`. On another rank it adds nothing; the device layer goes on to the next module when an attach declines (`device.c:1159`).
  - On a rank the list gives no QPU, the first module attaches as a placeholder with no backend, as every rank but the owner does now. The QPU incarnations then survive taskpool start-up everywhere (open question 3 of section 1.1).
- **One service per QPU.** Each QPU has its own backend, progress thread, jobs in flight, limits and counters. Each progress thread sleeps in its own backend's `wait`.
- **Why a thread per QPU, not one for all, as section 2.2 planned:**
  - The backend interface has one blocking `wait` per backend (section 9.2), and one thread cannot sleep in two.
  - `submit`, `results` and `cancel` are called from the progress thread and return when the provider has answered. With a provider that is a network round trip. On a shared thread, one slow provider would delay the jobs of every other QPU.
  - The extra threads cost little: they sleep, and a process has a handful of QPUs at most.
- **One pool per rank.** The QPUs of a rank share what is now one service's: the stack task bodies push to, and the waiting tasks. A lock protects the waiting tasks, which today belong to one thread.
- **Life cycle, per QPU, as now.**
  - The backend opens when its module is attached. A QPU whose backend cannot open gives no device, and the others stay.
  - The thread starts at the rank's first submission.
  - Everything stops from the context's at-fini callback, before the workers are joined.
- **Statistics are per QPU.** `parsec_qpu_device_info(q, ...)` and `parsec_qpu_service_stats(q, ...)` take the QPU's number in the list; today the second takes no number. The information says which rank owns QPU `q` and whether the calling rank does. What waits is counted per rank. (As built: `parsec_qpu_service_stats()` keeps its signature and gives the rank's totals, and `parsec_qpu_service_stats_of(q, ...)` is per QPU.)

### Binding a task to a QPU

**The rule: a task is bound to a QPU when that QPU has a free place for it, not before.**

PaRSEC binds a GPU task to its GPU when the task is selected, because the task's data must be moved to that GPU's memory. A QPU holds no data (section 9.4), so nothing forces the choice that early, and waiting costs nothing. Binding late gives four things:

1. **No place stays free while a task that could use it waits** on the same rank. With early binding each QPU keeps the tasks it was dealt, and when job times are as uneven as a provider's, some QPUs run dry while others still have tasks waiting.
2. **QPUs of different speeds need no estimate.** A faster QPU frees its places sooner and so takes more tasks. Early binding would need the time a job takes on each QPU. The device layer has a nominal second for that (section 9.4), and giving it measured times needs a change to the device layer.
3. **A QPU sees the circuit before it takes the task.** A QPU too narrow for a circuit never gets it. The device layer selects without seeing the circuit.
4. **A QPU that stops answering keeps only the tasks it has.** The tasks that wait go to the others.

With QPUs that are alike, `n` QPUs of `k` places each then behave as one QPU of `n` times `k` places, which is the model the experiments already use for one QPU.

**How it works, with the device layer unchanged.**

1. **Selection.** `parsec_select_best_device` picks the QPU incarnation and a QPU module of the rank. It is always the same one (`device.c:300`), and here that pick means only "a QPU of this rank". The scheduler adds the task's load to that module (`scheduling.c:151`).
2. **Submission.** `parsec_qpu_submit`, still on the worker, takes that load off again, sets `task->load` to 0 and pushes the task to the rank's pool.
3. **Binding.** A QPU with a free place takes, among the waiting tasks it may run, the first by priority and then arrival. It sets `task->selected_device` to its own module and `task->load` to its nominal time, and adds that to its `device_load`.
4. **Completion.** The scheduler subtracts `task->load` from the load of `task->selected_device` (`scheduling.c:539`). That is right in every case, including a task that never had a job, whose load is 0.
5. **Retry.** A task whose job failed goes back to the pool: its QPU takes the load off again. Its next job may run on another QPU.

- **Why this is safe.** Between the body's return and the hand-off, the task belongs to the QPU layer (section 2.3). After the hook has returned ASYNC, the worker reads neither `selected_device` nor `load` (`scheduling.c:218-240`, `:604`), and no instrumentation module in the tree reads them. So no thread reads those fields while the QPU layer changes them.
- **What the load means then.** A QPU's `device_load` counts the tasks it holds a job for. The test of hazard H12 (section 4.3) stays: every load is zero at the end.
- **Which tasks a QPU may run:** those pinned to it, and the unpinned ones whose circuit and shots its backend takes (`nb_qubits`, `max_shots`).
- **A task no QPU of the rank can run** completes as rejected at once, as now.

**Pinning.** A task may name its QPU: when the circuit was compiled for one device, or when related evaluations must come from the same device, since noise differs from device to device.

- A new call, `parsec_qpu_submit_on`, takes the QPU's number or `PARSEC_QPU_ANY`. `parsec_qpu_submit` becomes that call with `PARSEC_QPU_ANY`. A JDF body uses `PARSEC_QPU_SUBMIT_ON(Q, CIRCUIT, PARAMS, RESULT, SHOTS, KEY)`.
- It is the same call in PTG and in DTD.
- A task pinned to a QPU of another rank is refused with a message, as a quantum task on a rank without a QPU is now.

**Jobs that carry several tasks** (section 2.8, as built in M7) stay per QPU, since a job runs on one device. The rule "tasks share a job only when more of them wait than jobs can be submitted" now counts the free places of all the rank's QPUs that may run them, under the pool's lock. So no task shares a job while another QPU has a place for it.

**Failures, per QPU.**

- **A job that fails or times out** sends its tasks back to the pool, each to be retried alone, as now. An unpinned task may be retried on another QPU. (As built: it is retried on another QPU whenever one that is up can run it. Otherwise a QPU whose jobs fail at once is free first every time and takes the same task again.)
- **A submission the backend could not take** is asked again on the same QPU with the same tag, or abandoned there, as now. The provider may have the job, so it cannot move.
- **A QPU unavailable for longer than its `unavailable_timeout`** stops taking tasks. The waiting tasks pinned to it complete as unavailable. An unpinned task, waiting or of the submission that was given up, completes as unavailable only when every QPU of the rank that may run it is in that state; otherwise it waits for one that is not. (Corrected in M9: this text had the tasks of the given-up submission complete as unavailable, against its own test.) As built, such a QPU is tried again after a while with one job (section 9.8).
- **The result says which QPU ran the job.** The header gets the QPU's number in 8 of the 16 bits of `attempts` (`device_qpu.h`, `parsec_qpu_result_t`), so it stays 64 bytes, and its version goes from 1 to 2.

**Wake-ups.** A task body wakes one QPU that has a free place, and none when all are full: a QPU that frees a place looks at the pool itself. A QPU that leaves a task it cannot take wakes a sibling that has a free place. (As built, the rule is stricter and has more parts: a QPU shows its places only while its thread sleeps; whoever puts tasks into the pool wakes the others before it calls its backend again; and a thread looks once more before it sleeps. Section 9.8 has the two failures that showed why.)

### Alternatives considered

| Alternative | For | Against |
|---|---|---|
| Bind at selection: the device layer picks the QPU with the least load, with `device.c:300` changed for QPU types | No state shared between the QPUs' threads. Which QPU runs a task does not depend on timing | Points 1 to 4 above. It also changes the device layer |
| Pin with the PTG body property `device=` | The syntax exists | It does not work at this commit. For a body that is not CUDA or HIP, the compiler refers to a function it never generates (`jdf2c.c:4003`; the generator is reached for those two types only, `:7016`): a QPU body with `device=0` does not compile (tried). For CUDA and HIP, the generated loop declares a second `dev_index` (`:6806`), so the function always asks for a device one past the last and gets none, and the property has no effect. The loop was run alone, not on GPUs. No JDF in the tree uses the property. This is a defect of the base (`09c721868`, same line), not fixed here |
| Pin with a body's `evaluate=` function that sets `task->selected_device` | By reading, it works for a body of any type (`jdf2c.c:4006-4007`, `device.c:186-193`). Not tried | PTG only: a DTD insertion names a device type, not a device (`insert_function.h:331-340`). An argument of the submission serves both |
| One progress thread for all QPUs | No lock on the waiting tasks | The blocking calls, above. Waiting on several backends at once needs a new backend interface |

### Placement across ranks

The pool is per rank, because a task runs on the rank its affinity gives it.

- **The application chooses the rank**, as now: the task's affinity names an element of a collection that lives on a rank that owns a QPU.
- **Helpers:** `parsec_qpu_count()` and `parsec_qpu_rank_of(q)`.
- **A QPU on one rank does not take tasks that wait on another.** PaRSEC moves no task between ranks for any device. Where QPUs can be reached from one node (cloud devices), giving them all to one rank avoids the question. Where each sits on its own node (assumption A2), the application spreads its quantum tasks over the owner ranks.

### The QAOA application

- **All QPUs on one rank:** nothing changes in the task graph. `Q` keeps its one element (`testing_qaoa.c:195`), and the tasks go to the pool unpinned.
- **QPUs on several ranks:** `Q` gets one element per owner rank, and the quantum tasks of block `b` go to the owner rank number `b` modulo their count.
- **Pinning per block** is an option of the driver (`-Q`), for hardware. It keeps both evaluations of a step on one device, so the comparison the optimizer makes between them carries no difference between devices. On the mock it changes nothing in the answers.
- **No change for correctness elsewhere.** The sequential reference (A) and the MPI baseline (B) need none, since the samples depend on the key only.

### Tests

- **`runtime/qpu/multi`**: two and three mock QPUs on one rank. Checks:
  - every task completes once, and its result names a QPU that exists;
  - no QPU exceeds its own `max_outstanding` (the mock's `max_running`, per instance);
  - a pinned task runs on its QPU only;
  - every load is zero at the end.
- **No free place while a task waits.** From the time stamps of the jobs: with QPUs alike, three QPUs of two places finish a fixed set of tasks in the time the model gives for one QPU of six places, within 5%. This is a timing test, to be run on idle cores (section 9.7). (As built, `multi:noidle` reads the property itself from the time stamps: at the middle of the time a task waited, every QPU held as many jobs as it may.)
- **Speeds.** With one QPU twice as fast as the other, the fast one runs more than half of the unpinned tasks.
- **Width.** With a QPU of 4 qubits and one of 12, every 8-qubit circuit runs on the second, and a 16-qubit circuit completes as rejected.
- **Isolation.**
  - With faults on one QPU (fail, unavailable), every unpinned task still completes with a result, through the other.
  - The tasks pinned to the faulty QPU complete with its status, and no other task does.
  - A QPU whose backend cannot open leaves the others working. A task pinned to a QPU that does not exist is refused with a message.
- **`runtime/qpu/multi:mp`**: QPUs on rank 0 and on the last rank.
- **Hardware guard.** `runtime/qpu/no_hardware_in_tests` also reads the values of `device_qpu_devices` in the test definitions.
- **QAOA.** Final states equal byte for byte to the reference with 1, 2 and 3 QPUs, on one rank and spread over ranks, pinned and not.
- **Unchanged behaviour.** All existing tests pass with an empty list.
- **Sanitizers and deliberate bugs**, as in section 9. Bugs to plant:
  - a load that is not moved with its task;
  - a pinned task taken by another QPU;
  - a QPU that takes a circuit too wide for it;
  - the pool read without its lock.

## 10.4 Qiskit

"Supporting Qiskit" covers four different things, and each gives PaRSEC something else.

1. **IBM hardware through Qiskit Runtime.** This is the plan of section 2.6 (M4): a persistent adapter process on the official SDK, which transpiles with Qiskit and submits with `qiskit-ibm-runtime`.
2. **Qiskit Aer, a local simulator with noise models.** Through the same adapter, it gives a backend that:
   - is noisy, where the mock is exact;
   - is transpiled to a real device's gates and connectivity;
   - needs no account and reaches no network.

   It also tests the adapter, its protocol and the transpilation end to end without a provider. This changes one rule of section 2.6, that no test uses the adapter: an optional test may, registered only when Python and Qiskit Aer are found, and never required. The build still needs neither. Not verified in this pass: which noise models of IBM devices the current SDK ships.
3. **A backend in C only, on the Qiskit C API and QRMI.** Both have moved since section 2.6 called the C API experimental and unable to submit.
   - **The Qiskit C API transpiles.**
     - Qiskit 2.2 added `qk_transpile()`, a standalone transpiler in C. It mirrors the preset pass managers, with some passes skipped [Q1][Q2].
     - Qiskit 2.3 added functions for the individual stages and access to the DAG [Q3].
     - The reference page warns that `qk_transpile()` may not work correctly when Python and C are mixed in one process. That does not arise here: PaRSEC has no Python.
   - **The Qiskit C API reads and writes QPY.** Release 2.6.2, the latest (2026-11-05), has `qk_qpy_dump_file`, `qk_qpy_dump_buffer`, `qk_qpy_load_file` and `qk_qpy_load_buffer` (`crates/cext/src/qpy.rs` [Q6]). The file is not in release 2.5.0. QPY is one of the two forms in which the provider takes a circuit (section 2.6, source 5). No OpenQASM 3 writer was found in the C API: its file names and its circuit functions were searched.
   - **QRMI has a C API**, read in `src/cext.rs` of its repository and in its example for IBM [Q4]:
     - `qrmi_resource_new` and `qrmi_resource_free`;
     - `qrmi_resource_acquire` and `qrmi_resource_release`;
     - `qrmi_resource_target`, the device's description as text;
     - `qrmi_resource_task_start`, `qrmi_resource_task_status`, `qrmi_resource_task_result`, `qrmi_resource_task_logs` and `qrmi_resource_task_stop`;
     - `qrmi_resource_status_pending_job_count`, by its name the length of the device's queue (its code was not read);
     - `qrmi_config_resource_names_get`, the resource names in a QRMI configuration.

     No call waits for an event, so this backend's `wait` is a poll with back-off, as section 2.2 foresaw for such providers.
   - **One payload kind per vendor.** A task's payload is of one of six kinds (`QiskitPrimitive`, `PasqalCloud`, `AliceBobFelis`, `IQMServer`, `Oqtopus`, `MaestroLocal`). The Qiskit kind holds the input of a primitive, as text, and a program id. This design covers that kind only.
   - **QRMI backs IBM's Slurm plugin**, where a job asks for quantum resources with `--qpu=<names>`, a comma-separated list [Q5]. That fits section 10.3: each name becomes an entry `qrmi@<rank> resource=<name>`. The launch script writes the list from the names, so that every rank numbers the QPUs the same way.
   - **Not verified:**
     - how the QPY bytes go into the text input of a primitive (read since, on 2026-11-28: QRMI's own client puts OpenQASM 3 text there, and no QPY; section 9.9, "Three installs");
     - the input of the program that replaces the server-side Sampler (section 2.6, source 9).
4. **Circuits written in Qiskit.** An application builds its circuits in Python and gives PaRSEC an OpenQASM 3 file. This needs a reader for the subset of OpenQASM 3 that the native format holds: the thirteen operations of section 9.2, with parameters as `input angle`. The reader is small and in C, with no dependency. A build with the Qiskit C library (M10) can read QPY files as well.

**The two routes to IBM, compared.**

| | Adapter in Python (M4) | Backend in C (M10) |
|---|---|---|
| Needs at run time, on the owner rank's node | Python, Qiskit and the IBM SDK | Two C libraries |
| When IBM changes the program jobs go to | The SDK absorbs it | The backend writes the program's input itself, so it must follow |
| Simulation with noise | Qiskit Aer | None |
| At a centre that hands out QPUs through Slurm | Works, through the SDK | Fits directly: the job's resources are QRMI resources |
| Maturity of what it builds on | The official SDK | Two young libraries, one of them (QRMI) written in Rust with a generated C header [Q4] |

**Recommendation.** Build both, the adapter first.

- **M4a**: the adapter with Qiskit Aer. It needs no account, and it tests everything up to the network. The OpenQASM 3 reader and writer come with it, since the adapter speaks OpenQASM 3 (section 2.6).
- **M4b**: IBM hardware through the adapter, once there is an allocation.
- **M10**: the backend in C. Both libraries are optional dependencies of one backend file, off by default (`-DPARSEC_QPU_WITH_QRMI=ON`), as section 7.5 requires.
- **Qiskit never goes inside PaRSEC.** The build has no Python, as before.

## 10.5 Milestones

### M9: several QPUs per context

Built on 2026-11-26: section 9.8.

| Item | Description |
|---|---|
| Objective | Section 10.3: a list of QPUs, a service per QPU, one pool per rank, binding when a place is free, pinning, the QAOA application on several QPUs |
| Files | Edited: `parsec/mca/device/qpu/device_qpu_component.c`, `device_qpu_module.c`, `device_qpu_progress.c`, `device_qpu_internal.h`, `device_qpu.h`, `device_qpu_backend_mock.c`; `tests/apps/qaoa/qaoa.jdf`, `testing_qaoa.c`; `tools/qpu/README.md`; the two files of `tests/runtime/qpu/` that list tests. New: `parsec/mca/device/qpu/device_qpu_list.c` (the list), `tests/runtime/qpu/qpu_multi.jdf`. Nothing outside the component and its tests |
| Dependencies | M3, M7 |
| Difficulty | Medium to high. The waiting tasks, which one thread owns today, become shared by several, in the code that M2 and M7 tested hardest |
| Risks | A race in the pool, or a wake-up that is lost so that a task waits beside a free place. A load that does not follow its task. A process-wide assumption hidden in the split: the arrival counter, the reason quantum tasks cannot run, the mock's current backend and its test hooks. Picking the first task a QPU may run when tasks pinned elsewhere, or too wide for it, are ahead of it. The 16-device mask |
| Tests | Section 10.3; all existing tests with an empty list; thread and address sanitizers; the deliberate-bug pass |
| Acceptance criteria | Three mock QPUs on one rank, and QPUs on two ranks, give final QAOA states equal byte for byte to the reference. No QPU exceeds its limits. The timing test of section 10.3 holds. Faults stay on their QPU. Every load is zero at the end. Every existing test passes unchanged |

### M4, split

M4a and M4b as in section 10.4. M4's files, risks and tests stay as section 7.3 gives them, plus the Aer mode of the adapter and the OpenQASM 3 reader (`parsec/mca/device/qpu/device_qpu_qasm.c`, which M4 already lists, now in both directions). M4b still needs an allocation and `device_qpu_allow_hardware`, and no test reaches it.

### M10: a backend in C on QRMI and the Qiskit C API

| Item | Description |
|---|---|
| Objective | `device_qpu_backend_qrmi.c`: lower the native circuit to the Qiskit C API, transpile with `qk_transpile` against the target QRMI reports, write the circuit as QPY and the primitive's input around it, run it through QRMI. One QPU per QRMI resource |
| Files | New: `parsec/mca/device/qpu/device_qpu_backend_qrmi.c`; a CMake option and a find module for the two libraries; `tests/runtime/qpu/qpu_qrmi_lowering.c` |
| Dependencies | M2. M9 for more than one resource |
| Difficulty | Medium to high: two young libraries |
| Risks | The two unverified points of section 10.4. Version drift of both libraries: the QPY calls need Qiskit 2.6 |
| Tests | The lowering, the transpilation and the input, without any QRMI call. A QRMI run only with `device_qpu_allow_hardware`, and never in the test suite |
| Acceptance criteria | The input written for the QAOA circuits is accepted by the provider for one two-qubit circuit, by hand and with an explicit go-ahead, as in M4. The hardware guard covers the backend |

### M11: a CUDA-Q backend

Assumption A4 as a backend, for NVQLink-coupled QPUs and for CUDA-Q's simulated multi-QPU platform. It needs a CUDA-Q installation, and nothing here has one. Its milestone table is to be written when an installation exists. Since 2026-11-28 one exists, outside the repository: section 9.9, "Three installs".

## 10.6 What this means for the plan

- **Order.** M9 comes first: it needs nothing from outside, and the three backends are worth more with it. Then M4a. M10, M4b and M11 each wait for something from outside: access to hardware, or an installation. (The three installations were tried on 2026-11-28, and work: section 9.9, "Three installs". M4b still waits for an account, and M10's acceptance for a provider.)
- **Experiments that M9 makes possible.** All three were run on 2026-11-27 with mock QPUs (section 9.8, "The checks that were left, and the experiments on several QPUs").
  - E5 with several QPUs: tasks that share jobs across QPUs.
  - E6 with QPUs on several ranks.
  - A new one, E9: binding when a place is free against pinning in turn, at coefficients of variation of the job time of 0, 1 and 2. It measures what the rule of section 10.3 is worth, and the model of section 6.1 predicts it.
- **What can be said about several QPUs.** They are built and measured with mock QPUs, on one node and on two. Not measured: real hardware, more than two nodes, a network faster than Ethernet. Any text about this work should say so.
- **Related work** (section 5) gains:
  - NVQLink, `cudaq-realtime` and Ising, below the runtime;
  - QRMI with its Slurm plugin, beside it;
  - Q-IRIS on several QPUs [R3], the closest prior runtime with more than one QPU. Read since (section 9.8): the paper names no rule for choosing among them.
- **Found in passing:** the defect of the PTG `device=` property (section 10.3, alternatives). It is in ICLDisco/parsec at the base commit and is worth reporting there.

## Sources for section 10

- [N1] NVIDIA NVQLink product page: https://www.nvidia.com/en-us/solutions/quantum-computing/nvqlink/
- [N2] NVIDIA NVQLink architecture: https://developer.nvidia.com/blog/nvidia-nvqlink-architecture-integrates-accelerated-computing-with-quantum-processors
- [N3] NVIDIA Ising announcement: https://nvidianews.nvidia.com/news/nvidia-launches-ising-the-worlds-first-open-ai-models-to-accelerate-the-path-to-useful-quantum-computers
- [N4] NVIDIA Ising workflows: https://developer.nvidia.com/blog/nvidia-ising-introduces-ai-powered-workflows-to-build-fault-tolerant-quantum-systems/
- [N5] CUDA-Q, multiple QPUs: https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/mqpusims.html
- [N6] CUDA-Q, multi-processor platforms: https://nvidia.github.io/cuda-quantum/latest/using/backends/platform.html
- [N7] CUDA-Q repository, main branch: https://github.com/NVIDIA/cuda-quantum (`runtime/cudaq/algorithms/sample.h` for `sample_async` and `qpu_id`; `runtime/common/Future.h` for the result object)
- [Q1] Qiskit SDK 2.2 release notes: https://quantum.cloud.ibm.com/docs/api/qiskit/release-notes/2.2
- [Q2] Qiskit C API, transpiler: https://quantum.cloud.ibm.com/docs/api/qiskit-c/qk-transpiler
- [Q3] Qiskit SDK 2.3 release notes: https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.3
- [Q4] QRMI repository, main branch: https://github.com/qiskit-community/qrmi (`src/cext.rs`; `examples/qrmi/c/ibm_quantum_compute_service/src/quantum_compute_service.c`; `cbindgen.toml`)
- [Q5] IBM Quantum, Slurm plugin: https://quantum.cloud.ibm.com/docs/guides/slurm-plugin
- [Q6] Qiskit repository: https://github.com/Qiskit/qiskit (`crates/cext/src/qpy.rs` at tag 2.6.2; absent at tag 2.5.0)
