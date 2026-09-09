# Reasoning for commit `4c3558b4b38ea2035afb863759b18276f08b1d21`

This document explains, in detail, what changed in this commit and why each change was necessary.

## Goal

The main goal is to support `type_remote = AUTO` in PTG/JDF so receive buffers can be sized at runtime based on the actual incoming message size, instead of requiring fixed-size arena-based declarations.

This solves variable-size communication cases where fixed `type_remote` declarations over-allocate or cannot model runtime variability.

---

## 1) Compiler changes (`parsec/interfaces/ptg/ptg-compiler/jdf2c.c`)

### What changed

- Added logic to recognize `type_remote = AUTO` (`jdf_datatype_is_auto`).
- For AUTO receives, generated code sets:
  - `arena = PARSEC_REMOTE_DEP_AUTO_ALLOC`
  - **keeps the expected remote datatype** (layout/opaque datatype), same as non-AUTO
  - uses count from runtime metadata (`data->remote.src_count` / `data.remote.src_count`)
  - displacement `0`

### Why

The existing codegen path assumes a predefined arena datatype and fixed element sizing.  
AUTO needs the receive buffer size to be resolved dynamically at runtime from message metadata.

This preserves receiver type information while still allowing runtime-sized receives.
The key AUTO property is dynamic count/allocation, not forcing type to packed.

---

## 2) Remote dependency metadata changes (`parsec/remote_dep.h`, `parsec/remote_dep.c`)

### What changed

- Added sentinel marker:
  - `PARSEC_REMOTE_DEP_AUTO_ALLOC`
- Extended `parsec_dep_type_description_s` with:
  - `int16_t device_index`
- Initialized/reset `device_index` in remote dep lifecycle (`remote_dep.c`).

### Why

- Sentinel marker distinguishes AUTO allocations from normal arena allocations.
- `device_index` carries placement intent (CPU vs selected GPU), required for GPU-aware AUTO receives.

---

## 3) Runtime AUTO allocation and communication logic (`parsec/remote_dep_mpi.c`)

This file contains the core runtime implementation.

### 3.1 AUTO allocator entry

When `arena == PARSEC_REMOTE_DEP_AUTO_ALLOC`, allocation uses AUTO-specific codepath instead of fixed arena datatype path.

### Why

Needed to decouple receive allocation from static arena element size.

### 3.2 Bucketed allocator for performance

Added power-of-two bucket arenas for AUTO receives with MCA tuning:

- `runtime_comm_auto_bucket_min_shift` (default 3 => 8B)
- `runtime_comm_auto_bucket_max_shift` (default 20 => 1MiB)

If message size fits range, allocate from bucket arena; otherwise fallback to malloc-based allocation.

### Why

Per-message malloc/free is expensive in hot communication paths.  
Bucketed arenas reduce allocator churn and improve reuse.

### 3.3 GPU-aware AUTO allocation

Added:

- `runtime_comm_auto_gpu_enable` MCA parameter
- GPU target selection through task-class affinity (`task_class->data_affinity`) and
  referenced data ownership/preference
- GPU memory allocation via device module (`memory_allocate`)
- matching release callback using device `memory_free`

### Why

If successor task is expected on GPU, receiving directly into GPU memory avoids unnecessary host staging and copy overheads.
Using data affinity here avoids selecting a GPU too early from partially initialized task inputs.

### 3.4 Eager and rendezvous behavior

- Eager activation unpack explicitly forces host allocation (`device_index = 0`) for that path.
- Rendezvous GET path:
  - attempts comm-engine memory registration
  - if AUTO+GPU registration fails, fallback to host AUTO buffer and retry registration.

### Why

Different comm paths and backends have different pointer registration capabilities.  
Fallback preserves correctness across backends that cannot register some GPU pointers.

### 3.5 Stability fix for non-AUTO paths

During integration, a regression was identified: registration failure handling became too strict for non-AUTO receives.

Fix: keep fallback/fatal retry logic scoped to AUTO receives only, preserving previous behavior for non-AUTO paths.

### Why

Without this, unrelated MPI tests failed with forced fatal errors on registration return codes that were previously tolerated.

---

## 4) PTG tests (`tests/dsl/ptg/receive_auto.jdf`, `tests/dsl/ptg/receive_auto_mixed.jdf`)

### What changed

- Added correctness-focused AUTO tests:
  - `receive_auto.jdf` (AUTO-only receive)
  - `receive_auto_mixed.jdf` (AUTO + fixed receive side-by-side)
- Added full byte validation and global MPI reduction of error counters.
- Added explicit PASS output.
- Added CUDA bodies for SEND and AUTO RECV paths with CPU fallbacks.
- Added GPU execution telemetry:
  - `g_gpu_hits`
  - GPU ID bitmask (`g_gpu_mask`)
  - reduced `distinct_gpus` and `visible_cuda` checks

### Why

These tests validate:

1. Functional correctness for variable-size data movement.
2. AUTO and fixed receive compatibility in same graph.
3. Actual GPU execution (not just successful run).
4. Multi-GPU usage when multiple devices are available.

---

## 5) Test build integration (`tests/dsl/ptg/CMakeLists.txt`)

### What changed

- Added build/test targets for `receive_auto` and `receive_auto_mixed`.

### Why

Ensures tests are part of normal CI/ctest workflows.

---

## 6) Test runner script (`tests/dsl/ptg/run-receive-auto-tests.sh`)

### What changed

Added a script to automate:

- optional build
- local and distributed runs
- CPU and GPU-forced runs
- launcher selection (`auto|mpirun|srun`)
- GPU count precheck (`--require-gpus`)
- result parsing (`gpu_hits`) and failure conditions
- Slurm hardening:
  - `srun --gpus-per-task=1`
  - distributed GPU run adds `--mca device_cuda_enabled 1`
  - PMIx psec environment fallback (`OMPI_MCA_psec=native`, `PMIX_MCA_psec=native`)

### Why

Cluster environments vary widely in launcher behavior, GPU exposure, and PMIx security plugin setup.  
The script makes validation reproducible and less fragile.

---

## Design intent summary

This commit intentionally introduces AUTO as a minimal but practical runtime-level abstraction:

- Keep PTG syntax simple (`type_remote = AUTO`).
- Route size resolution to runtime where actual message count is known.
- Preserve performance with bucket pooling.
- Enable GPU-aware receives behind an MCA knob.
- Preserve portability with robust fallback behavior.
- Validate correctness with focused PTG tests and a reproducible runner script.

---

## 7) Review-driven refinements (follow-up to PR comments)

After review feedback, three focused adjustments were applied:

1. **Preserve type for AUTO**
   - Removed forced `PARSEC_DATATYPE_PACKED` assignment in AUTO receive metadata path.
   - AUTO now keeps expected datatype and only relies on runtime count for sizing.
   - This improves heterogeneity safety compared to treating AUTO as byte-void payload.

2. **Device selection moved away from early `select_best_device`**
   - Replaced `parsec_select_best_device` call in datatype-retrieval phase with
     affinity-based lookup:
     - call `task_class->data_affinity`
     - fetch referenced data object
     - prefer `preferred_device`, then `owner_device`, if GPU.
   - This avoids relying on input copies that may not be fully initialized yet.

3. **Clarified GPU release callback intent**
   - Renamed callback to indicate it releases **GPU memory backing the copy**, not the copy object.
   - Uses `copy->device_index` consistently as release target.

These changes were validated with:

- `run-receive-auto-tests.sh` (local + distributed, CPU + GPU-forced)
- full `ctest --output-on-failure -j8` (117/117 passing)

---

## Known tradeoffs / limitations

- Host AUTO buffers are normal host allocations unless comm-engine registration pins them transiently.
- GPU coverage checks rely on runtime-visible device indexing; environments with GPU masking/cgroup restrictions may report fewer visible devices.
- Distributed multi-rank GPU tests can be sensitive to launcher and resource policy (Slurm task/GPU mapping, PMIx plugin availability).

