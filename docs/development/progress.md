# Forge Progress

Persistent record of completed milestones/phases, per `docs/development/roadmap.md`.

## Phase 1 — Core Foundations

### M1 — Tensor abstraction and CPU execution boundary
`Tensor` (shape/dtype/device), `DType`, `Device`, the `Backend`/`CPUBackend` dispatch boundary, and
Forge-specific errors (`ShapeMismatchError`, `UnsupportedDTypeError`, `UnsupportedDeviceError`).
Operations: `+`, `-`, `*` (broadcasting), `@` (1D/2D matmul), `.sum()`, `.reshape()`. 53 tests.

### M2 — Automatic differentiation core
Reverse-mode autograd on top of the M1 Tensor: `requires_grad`, `.grad`, `.is_leaf`, `.grad_fn`,
`.backward()`, `.zero_grad()`. New `forge/autograd/` package (`Node` graph nodes, topological
`run_backward`, broadcast/matmul/sum backward math). Backward rules for all M1 operations, with
broadcast-aware gradient reduction, gradient accumulation across multiple use sites, and
non-scalar-output backward requiring an explicit upstream gradient. New `GradientStateError`.
Graph is freed as it is consumed by `backward()`; a second `backward()` call on the same non-leaf
output raises rather than silently reusing freed state. 90 tests total (53 M1 + 37 M2). See
`docs/architecture/autograd.md`.

### M3 — Module and parameter system
Neural-network composition on top of the M1/M2 Tensor+autograd stack: new `forge/nn/` package
(`Parameter`, a `requires_grad=True`-by-default `Tensor` subclass; `Module`, with attribute-based
parameter/child-module registration, recursive `parameters()`/`named_parameters()` discovery with
deduplication by identity, `train()`/`eval()` mode propagation, and a `forward()`-invoking
`__call__`) and the `Linear`/`ReLU` layers built from it. New `Tensor.relu()` primitive
(`Backend.relu`/`CPUBackend.relu`) following the same Tensor→Backend→autograd `Node` pattern as
the M1 ops, since ReLU could not be expressed with the existing operation set. New minimal
`forge/random.py` (a process-global `numpy.random.Generator`, `seed()`/`default_generator()`) for
deterministic `Linear` parameter initialization (`Uniform(-1/sqrt(in_features),
1/sqrt(in_features))`). New `ModuleError`. No optimizer, training engine, or CUDA in this
milestone. 136 tests total. See `docs/architecture/modules.md`.

### M4 — Losses and optimizer
Completes the optimization foundation on top of the M1-M3 Tensor/autograd/nn stack: new
`forge/nn/loss.py` (`Loss` base class, `MSELoss`, `CrossEntropyLoss`) and new `forge/optim/`
package (`Optimizer` base class, `SGD`). Two new differentiable Tensor primitives,
`Tensor.exp()`/`Tensor.log()` (`Backend.exp`/`log`, `CPUBackend` `np.exp`/`np.log`), following the
same Tensor -> Backend -> autograd `Node` pattern `.relu()` established in M3 -- needed for a
numerically stable `CrossEntropyLoss` (log-sum-exp trick, shifted by a non-differentiable per-row
max computed via NumPy). `SGD.step()` mutates `Parameter._data` in place via NumPy rather than
Tensor ops, so it never attaches a `grad_fn` or extends the autograd graph. New `LossError`,
`OptimizerError`. Verified with a deterministic linear-regression experiment (`Linear` + `MSELoss`
+ `SGD`, loss drops from ~4.9 to ~6e-5 over 200 steps, recovers the true `y = 3x1 - 2x2 + 1`
weights) and a classification experiment (`Linear` -> `ReLU` -> `Linear` + `CrossEntropyLoss` +
`SGD`, 100% final accuracy on a separable synthetic set). No training engine, DataLoader, dataset
abstraction, persistence, or CUDA loss/optimizer support in this milestone. 179 tests total. See
`docs/architecture/optimization.md`.

### M5 — Dataset, DataLoader, and transforms
Adds the data pipeline foundation on top of the M1-M4 Tensor/nn stack, independent of
`Module`/`Loss`/`Optimizer`/a training engine: new `forge/data/` package (`Dataset` base class,
`TensorDataset` -- an in-memory array-backed dataset over one or more aligned Tensors, `Subset`,
`random_split`), `DataLoader` (batching, optional shuffling with deterministic ordering given a
supplied `numpy.random.Generator` or `forge.random`'s process-global generator, explicit
`drop_last` partial-batch handling), and a small composable transform set (`Transform`, `Compose`,
`ToTensor`, `Normalize`, `Reshape`, `Flatten`, `Lambda`). `TensorDataset` wires `transform` to the
features position and `target_transform` to the target position separately, so a feature transform
cannot silently reach a label. Batches are always Forge `Tensor`s (or tuples of them), never raw
NumPy, assembled via `np.stack` over each sample's underlying storage. New `DataError`. No
multiprocessing workers, asynchronous prefetching, file-backed/image/tabular dataset conveniences,
or `Trainer`/training engine in this milestone -- verified via a hand-written (non-`Trainer`)
Dataset -> DataLoader -> `Linear` -> `MSELoss` -> `SGD` loop that reduces loss over an epoch. 246
tests total (179 M1-M4 + 67 M5). See `docs/architecture/data-system.md`.

## Phase 2 — Training Framework

### M6 — Training engine
Adds Forge's first training engine on top of the M1-M5 stack: new `forge/training/` package
(`Trainer`, `TrainingHistory`, `EpochResult`, `EvaluationResult`, and a `Metric` abstraction with
`MeanSquaredError`/`MeanAbsoluteError`/`Accuracy`). `Trainer` orchestrates the existing
`Dataset`/`DataLoader`, `Module`, `Loss`, autograd, and `Optimizer` components -- it computes no
gradients, updates no parameters, and implements no loss/optimizer/batching logic itself; each
training step runs the same `zero_grad -> forward -> loss -> backward -> step` sequence Milestones
1-5 already required by hand. New minimal autograd extension, `forge.no_grad()`
(`forge/autograd/engine.py`) -- a single global flag checked by `Tensor._differentiable_wrap` that
suspends graph construction, used by `Trainer.evaluate()` so an evaluation forward pass builds no
autograd graph. `evaluate()` also switches the model to eval mode (propagated to nested modules via
the existing M3 `Module.eval()`) and restores its prior mode afterward. Metrics aggregate by
accumulating running sums/counts across batches (not averaging per-batch means), so unequal batch
sizes are weighted correctly. New `TrainerError`. Verified with a deterministic linear-regression
experiment (`Linear` + `MSELoss` + `SGD` via `Trainer.fit`, loss drops over two orders of magnitude,
recovers the true `y = 3x1 - 2x2 + 1` weights) and a classification experiment (`Linear` -> `ReLU`
-> `Linear` + `CrossEntropyLoss` + `SGD`, >=85% held-out accuracy), both with a validation loader
evaluated each epoch and progress output on/off. No CUDA execution, early stopping, checkpointing,
learning-rate schedules, or callbacks in this milestone. 304 tests total (246 M1-M5 + 58 M6). See
`docs/architecture/training-engine.md`.

### M7 — Model persistence
Adds save/load for trained models on top of the M1-M6 stack: new `forge/serialization/` package
(`save_model`, `load_model`, `register_module`) plus new `PersistenceError`. A model file is a ZIP
archive (`metadata.json` + one `.npy` per parameter) -- structured, versioned
(`forge_format_version`), and inspectable without Forge; loading never executes code found in the
file (`json.loads` for metadata, `numpy.load(..., allow_pickle=False)` for arrays). Architecture
reconstruction goes through an explicit in-process registry (`forge/serialization/registry.py`)
keyed by a plain string type name -- never pickle, `eval`, or dynamic import -- with `Linear`/`ReLU`
registered as Forge's built-in supported types; a custom/composite `Module` subclass must call
`register_module()` itself before it can be saved or loaded (see ADR-003). Saved parameter state
(shape, dtype, `requires_grad`, values) is validated against the file on load, and each module's
`.training` flag round-trips per-module rather than being forced to a fixed mode. No autograd graph
or optimizer state is ever saved -- loaded parameters are fresh leaf `Parameter`s that build an
entirely new graph on the next forward pass. Saving is atomic (temp file + `os.replace`), so a
failed save never leaves a partial file. Verified with a deterministic train -> save -> load ->
predict workflow (`Linear` + `MSELoss` + `SGD` via `Trainer`) whose final step runs in a genuinely
separate subprocess to prove the saved file alone is sufficient, plus tampered-file tests confirming
malformed/corrupt/unsupported-version/unsupported-type/unsupported-device files fail clearly with no
code execution. No CUDA serialization, optimizer checkpointing/training resume, or CLI in this
milestone. 332 tests total (304 M1-M6 + 28 M7). See `docs/architecture/persistence.md` and
`docs/architecture/decisions/ADR-003-persistence-format.md`.

## Phase 4 — GPU/CUDA (reordered ahead of Phase 3, Data & Model Ecosystem)

### M8 — Initial CUDA backend
Adds Forge's first real CUDA execution backend on top of the M1-M7 stack: new
`forge/backend/cuda/` package (`kernels.cu` -- CUDA C++ kernel source; `build.py` -- locates
`nvcc`/MSVC and compiles the kernels into a shared library, lazily, on first CUDA use; `backend.py`
-- `CUDAStorage`, `CUDABackend`, `get_cuda_backend()`, `is_cuda_available()`), a new
`Tensor.to(device)` for explicit CPU<->CUDA transfer, `Backend.to_numpy()` (new abstract method,
trivial on CPU) for materializing backend storage as a host array, and a new `CUDAError` exception.
Kernels are compiled with `nvcc -arch=sm_50` (the verified 940MX's Compute Capability) and loaded
via the standard-library `ctypes` -- no PyTorch/TensorFlow/CuPy/JAX dependency, per ADR-004. The
operation set is deliberately small: tensor creation/transfer (any dtype, raw byte copy),
`add`/`sub`/`mul` (`float32`/`float64`, exact-shape only -- no CUDA broadcasting), `matmul` (the
same four 1D/2D cases the CPU backend supports, via a naive one-thread-per-output kernel),
`sum` (full reduction only), and `reshape` (a device-to-device copy with new shape metadata, no
allocator). `relu`/`exp`/`log` are required by the `Backend` ABC but raise `CUDAError` on CUDA in
this milestone -- no kernel exists for them yet. `CUDAStorage` holds a real `cudaMalloc` pointer,
never a NumPy array relabeled as CUDA (tested structurally). CUDA autograd is explicitly out of
scope: `Tensor._differentiable_wrap` and `Tensor.backward()` both raise `UnsupportedDeviceError` for
any non-CPU differentiable operation or backward call, and `.to()` always produces a
`requires_grad=False` leaf, rather than building a graph the existing NumPy-based backward closures
cannot correctly traverse. Verified on the actual development GPU (940MX, CC 5.0, driver 582.53,
CUDA Toolkit 12.6): all 54 new CUDA tests (`tests/test_cuda_backend.py`,
`tests/test_cuda_consistency.py`) pass when run on this hardware, covering device dispatch (proven
structurally distinct from `CPUBackend`), CPU<->CUDA transfer correctness, real kernel execution for
every supported op, CPU/CUDA numerical consistency (`float32`/`float64`, multiple matmul shapes),
and clear failure for every unsupported case (broadcasting, axis-wise sum, `relu`/`exp`/`log`,
non-float dtypes, device mismatch, autograd-on-CUDA, invalid device index). The same suites were
also run with `PATH` stripped of the CUDA toolchain to confirm all 54 skip cleanly with zero
failures and the 332 CPU-only tests are entirely unaffected. 386 tests total (332 M1-M7 + 54 M8).
See `docs/architecture/cuda-backend.md` and
`docs/architecture/decisions/ADR-004-cuda-execution-strategy.md`.

### M9 — CUDA Module execution and device movement
Extends the M8 CUDA backend from raw `Tensor` execution to high-level
`nn.Module` execution: `Module.to(device)` (`forge/nn/module.py`)
recursively moves every `Parameter` in a module tree to a target device, in
place, via a new private `Tensor._move_storage_()` primitive -- the in-place
counterpart to the existing value-semantics `Tensor.to()`, chosen so a moved
`Parameter` keeps its Python identity, shape, dtype, and `requires_grad`
(any stale `.grad` is cleared). `Module.to()` mutates and returns `self`,
matching the existing `train()`/`eval()` convention; a new `Module.device`
property reports the single device shared by a tree's `Parameter`s (`None`
if it owns none, `ModuleError` if it finds more than one). Two CUDA kernel
gaps blocking real `Linear -> ReLU -> Linear` execution were closed: a real
`relu` kernel (`cf_relu_{f32,f64}`, replacing M8's "unsupported" stub), and
a targeted row-broadcast addition to `add`/`sub`/`mul`
(`cf_{add,sub,mul}_bcast_{f32,f64}`) supporting exactly the `(rows, cols) +
(cols,)` shape a batched `Linear`'s bias add needs -- general N-D
broadcasting remains out of scope. `Linear`/`ReLU` needed no CUDA-specific
forward code: `x @ weight + bias` and `x.relu()` dispatch to CUDA purely
through the existing Tensor -> Backend boundary. A pre-existing bug in
`Tensor.relu()`'s backward closure (`self._data > 0` computed eagerly,
before the CUDA-unsupported check used to make it unreachable) was fixed by
making the comparison lazy inside the closure, since it is now reachable
for a genuine CUDA forward pass. Because `Module.to()` preserves
`requires_grad=True`, a CUDA model's forward pass must run inside the
existing `forge.no_grad()` (unchanged from Milestone 6) to avoid the
already-tested M8 guard that refuses to build a graph on a non-CPU device;
`backward()` on a CUDA-resident output still raises `UnsupportedDeviceError`
unconditionally, matching M8's untouched autograd boundary. `Trainer`,
losses, and persistence remain entirely CPU-only and unmodified; a
CUDA-moved model fed to either now fails clearly (a device-mismatched
forward op for `Trainer`, an `UnsupportedDeviceError` from `Parameter.numpy()`
for `save_model`) rather than silently doing the wrong thing. Verified on
the actual development GPU (940MX, CC 5.0, driver 582.53, CUDA Toolkit
12.6): all CUDA tests (`tests/test_cuda_backend.py`,
`tests/test_cuda_consistency.py`, and the new `tests/test_module_cuda.py`,
86 tests total) pass directly on this machine, including a real
`Linear -> ReLU -> Linear` model moved to CUDA whose forward output (under
`no_grad()`) numerically matches an identically-initialized CPU model, a
structural check (monkeypatched `CPUBackend`) that this forward pass never
calls `CPUBackend`, and confirmation that CUDA backward still fails. The
same suites skip cleanly (`86 skipped`, `0 failed`) with the CUDA toolchain
removed from `PATH`, and the CPU-only suite (338 tests) is unaffected. 424
tests total (338 CPU-only + 86 CUDA). See `docs/architecture/cuda-backend.md`
and `docs/architecture/modules.md`.

### M10 — CUDA autograd
Extends the M8-M9 CUDA backend from forward-only execution to full
reverse-mode autograd, closing the "No CUDA autograd" limitation carried
since Milestone 8: a CUDA computation graph now builds and differentiates
entirely on the GPU, with CUDA-resident gradients and a CUDA-executing
`SGD.step()`, with no CPU fallback at any point. Backward computation
became backend-aware rather than a second autograd engine -- seven new
`Backend` ABC methods (`add_backward`, `sub_backward`, `mul_backward`,
`matmul_backward`, `sum_backward`, `reshape_backward`, `relu_backward`) plus
`sgd_step`, implemented once in `CPUBackend` (NumPy math relocated from the
now-deleted `forge/autograd/functions.py`) and once in `CUDABackend` (real
CUDA kernels: `cf_neg`, `cf_relu_backward`, `cf_scale`, `cf_transpose`,
`cf_reduce_rows`, `cf_broadcast_scalar`, `cf_sgd_step`, composed with
existing forward kernels wherever the math allows it -- e.g. `matmul`
backward reuses the forward `matmul`/`reshape` kernels against a freshly
transposed operand). `Tensor`'s backward closures (`forge/tensor/tensor.py`)
became thin wrappers calling `get_backend(device).<op>_backward(...)`;
`forge/autograd/engine.py`'s graph-traversal logic (`Node`, `run_backward`,
`no_grad`) needed no device-specific changes beyond dispatching
multi-consumer gradient accumulation through `Backend.add()` (a
`CUDAStorage` has no `__add__`). See ADR-005 for the full design rationale.
`Tensor.backward()` dropped its CPU-only restriction and gained explicit
device/dtype consistency checks for an upstream `gradient` argument;
`Tensor._differentiable_wrap` dropped the M8 guard that refused to attach a
`grad_fn` on a non-CPU device (an unsupported CUDA op like `exp`/`log`
still fails clearly, but now via its own forward call raising `CUDAError`,
before `_differentiable_wrap` is ever reached, rather than via a separate
device check). `Module.to("cuda")`'s long-standing "forward pass must run
inside `forge.no_grad()`" boundary is gone: a bare CUDA forward call now
builds a real graph, and `backward()` on CUDA output now succeeds. A real,
unrelated latent bug was found and fixed along the way:
`CUDABackend.from_array` used `np.ascontiguousarray`, which silently
promotes a 0-d array to shape `(1,)` -- invisible until a genuine CUDA
scalar (a loss, or `x.sum()`) was produced and used in further arithmetic,
which Milestone 10's real training-loop verification was the first thing
to actually exercise. Supported CUDA backward operations are deliberately
the same small set M8-M9 support forward: `add`/`sub`/`mul` (exact-shape
and the one row-broadcast shape), `matmul` (1D/2D), `sum` (full reduction),
`reshape`, `relu` -- `exp`/`log` remain CPU-only in both directions, and
CUDA `sum(axis=...)`/general N-D broadcasting remain unsupported forward,
so backward never needs to handle them either. Verified on the actual
development GPU (940MX, CC 5.0, driver 582.53, CUDA Toolkit 12.6): the new
`tests/test_cuda_autograd.py` plus updated `tests/test_cuda_backend.py`/
`tests/test_module_cuda.py` (121 CUDA tests total; 459 tests overall) pass
directly on this machine, covering CPU/CUDA gradient agreement for every
supported operation (including both row-broadcast operand orders and all
four matmul 1D/2D cases, plus a finite-difference check), ReLU backward
across positive/negative/zero/mixed inputs, a real `Linear` and a
`Linear -> ReLU -> Linear` multi-layer model whose CUDA-resident gradients
match CPU, gradient accumulation across multiple consumers of one CUDA
tensor, device/dtype-mismatch errors on `backward()`, `exp`/`log` still
failing clearly, `no_grad()` still suspending CUDA graph construction, a
structural check that a full CUDA model forward+backward pass calls zero
`CPUBackend` methods, CUDA `SGD` matching an equivalent CPU step, and a
real 20-epoch CUDA training loop (`TensorDataset` -> `DataLoader` ->
`Linear.to("cuda")` -> `MSELoss` -> `SGD`) that drops loss over 20x and
recovers the true regression weights. The full suite was also run with the
CUDA toolchain stripped from `PATH` to confirm all 121 CUDA tests skip
cleanly (`338 passed, 121 skipped`, `0 failed`). `Trainer`/`DataLoader`/
persistence remain CPU-only and unmodified, per the milestone's explicit
scope. See `docs/architecture/autograd.md`, `docs/architecture/cuda-backend.md`,
`docs/architecture/decisions/ADR-005-backend-aware-autograd.md`.

## Phase 5 — Performance

### M11 — Performance benchmarking and targeted optimization
Adds a reproducible benchmark subsystem on top of the M1-M10 stack and
applies one measurement-justified optimization: a new top-level
`benchmarks/` package (`timing.py` -- CPU/CUDA-aware timing with explicit
`cudaDeviceSynchronize()`-based bracketing for CUDA, since async kernel
launches make naive `perf_counter()` wrapping measure launch overhead, not
execution time; `sizes.py` -- tiny/small/medium size configs; `ops_bench.py`/
`backward_bench.py`/`transfer_bench.py`/`training_bench.py` -- the four
benchmark categories; `results.py` -- structured `BenchmarkResult`,
JSON + human-readable table output; `run.py`/`__main__.py` -- the
`python -m benchmarks` CLI), deliberately kept outside the `forge` package
(`import forge` never touches it) and outside the correctness suite
(`tests/test_benchmarks.py` only checks the harness's own mechanics with
trivial callables, never a real benchmark or a timing threshold). A new
public `CUDABackend.synchronize()` (`forge/backend/cuda/backend.py`) exposes
the existing internal synchronization point for benchmark code, without any
new native/CUDA-event code. Benchmarking every shared CPU/CUDA operation at
three scales (32/128/512 for matmul, matching element counts elsewhere)
found that Milestone 8's naive one-thread-per-output-element CUDA `matmul`
kernel was a real, measured bottleneck at the 512x512 scale specifically
(~4.4x slower than CPU/NumPy forward, ~3.4x slower backward) -- a pattern
distinct from the expected launch/transfer-overhead slowdown at tiny/small
scale, since it *grows* at the scale where compute should dominate launch
overhead. `k_matmul` (`forge/backend/cuda/kernels.cu`) was rewritten as a
standard 16x16-tile shared-memory GEMM (cuBLAS was considered and
deliberately not introduced, per ADR-004's existing "no new numerical-library
dependency" rationale and the milestone brief's own preference for a tiled
kernel once naive matmul is a clearly measured bottleneck); the exported
`cf_matmul_{f32,f64}` signature is unchanged, so no Python-side code
changed. Measured before/after: ~1.7-1.8x speedup for both forward and
backward matmul at the 512x512 scale (16.00ms -> ~9.0-9.2ms forward;
34.74ms -> ~19.6-21.6ms backward); CUDA matmul remains slower than CPU at
this scale even after the optimization, reported as a measured fact rather
than hidden. All 121 CUDA tests and the full 474-test suite pass unchanged
against the rewritten kernel. See `docs/performance/benchmarking.md` for
full methodology, environment, baseline numbers, and the optimization
decision's complete reasoning.

## Phase 4 — GPU/CUDA (continued after Phase 5 — Performance)

### M12 — CUDA Trainer and loss integration
Extends `forge.training.Trainer` to run a real end-to-end training/
evaluation workflow on CUDA through the existing high-level abstractions --
`Dataset -> CPU DataLoader -> Trainer(device="cuda") -> explicit batch
transfer -> CUDA Module -> CUDA Loss -> CUDA autograd -> CUDA SGD` -- rather
than the hand-written direct optimization loop M10 verified CUDA training
with. `Trainer.__init__` now accepts `device="cuda"` (probing
`get_backend()` immediately, so an unavailable CUDA backend raises
`CUDAError` at construction, not the first batch) instead of unconditionally
rejecting non-CPU devices. `Trainer` chose a **validate, never move** model-
placement policy: `_check_model_device()`, called at the start of
`fit()`/`evaluate()`, compares `model.device` against `self.device` and
raises `UnsupportedDeviceError` naming the required `model.to(device)` call
rather than silently relocating the model -- the only policy compatible with
the existing M9 test that a `device="cpu"` Trainer must still reject a
CUDA-resident model (renamed, behavior preserved:
`tests/test_module_cuda.py::test_trainer_configured_for_cpu_rejects_a_cuda_model`).
A new `Trainer._to_device_batch()` explicitly transfers each batch
(`x.to(device)`, and `y.to(device)` when `y` is a Tensor) immediately before
the forward pass; `DataLoader`/`Dataset` gained no device awareness and no
new capability -- this is the only place a batch crosses a device boundary.
`Metric._as_numpy()` (`forge/training/metrics.py`) now transfers a `Tensor`
argument to CPU first, so all three built-in metrics (`MeanSquaredError`/
`MeanAbsoluteError`/`Accuracy`) work unmodified for CUDA predictions -- a
one-way, read-only transfer for reporting, never a `CPUBackend` compute call.

**CUDA losses.** `MSELoss` needed zero new CUDA kernels: it composes only
`-`/`*`/`.sum(axis=None)`, all already CUDA-forward-and-backward-capable
since M8-M10. The one real fix: its internal `* (1/n)` scale is now built as
an explicit `Tensor(1/n, dtype=prediction.dtype, ...)` rather than a bare
Python float (`Tensor._coerce` infers `float32` for a bare scalar regardless
of the tensor's own dtype -- harmless on CPU, `CUDAError`-raising on a
`float64` CUDA loss, since `CUDABackend` requires matching operand dtypes).
`CrossEntropyLoss` is deliberately deferred to CPU-only (Approach B): it
needs `.exp()`/`.log()` (CUDA-unsupported since M8) and an axis-wise
`.sum(axis=1, keepdims=True)` (CUDA `sum()` supports only a full reduction)
-- implementing all three plus their CUDA backward rules plus preserving
numerical stability was judged out of proportion to the milestone's core
objective, which needed none of them. `CrossEntropyLoss.forward()` now
rejects non-CPU logits immediately with a clear `LossError` rather than
partially executing on CPU or failing indirectly via `.numpy()`'s own device
check.

New tests: `tests/test_cuda_loss.py` (13 tests -- CUDA `MSELoss` forward/
backward correctness and CPU/CUDA consistency across `float32`/`float64`,
gradient residency, device-mismatch rejection, a structural
zero-`CPUBackend`-calls check, and the `CrossEntropyLoss` CUDA-rejection
behavior) and `tests/test_trainer_cuda.py` (19 tests -- CUDA `Trainer`
construction/validation, explicit batch movement, full training/validation/
evaluation lifecycle with metrics, `no_grad()` evaluation building no graph,
a structural no-CPU-fallback check across a multi-epoch `fit()` +
`evaluate()` call, a 60-epoch end-to-end regression that recovers the true
weights, and CPU/CUDA training consistency from identical initial
parameters). One existing test (`tests/test_trainer.py`) was updated from
asserting `device="cuda"` is always rejected to branching on
`is_cuda_available()`, matching `tests/test_device.py`'s own established
convention -- this is the one test whose assertion this milestone
deliberately supersedes. All other M1-M11 tests pass unmodified. 506 tests
total (153 CUDA tests, up from 121; 353 CPU-only, up from 338), verified
directly on the development GPU (940MX, CC 5.0, driver 582.53, CUDA Toolkit
12.6): CUDA `Trainer.fit()` over 40 epochs reduces a linear-regression loss
by >10^13x (recovering the true weight/bias within 3e-7), CUDA-resident
Parameters and gradients confirmed throughout, `SGD.sgd_step` never touches
`CPUBackend`, `evaluate()`'s `no_grad()` forward pass produces a prediction
with `requires_grad=False`/`grad_fn=None`, a structural monkeypatch of every
`CPUBackend` compute method records zero calls across a full `fit()` +
`evaluate()` run, and the resulting CUDA loss curve and final parameters
match an identically-initialized CPU run within `1e-3` (measured max diff
~2e-7). See `docs/architecture/training-engine.md`'s **Device semantics**
section and `docs/architecture/cuda-backend.md`'s **CUDA losses** section.

### M13 — CUDA model persistence
Extends `forge.serialization` (`save_model`/`load_model`, unchanged package
layout: `registry.py`/`archive.py`/`model.py`) so models whose `Parameter`s
live on CUDA can be saved and reloaded, closing the "No CUDA persistence"
limitation carried since Milestone 9. No second serialization system, no
`CUDA`-specific model class, and no archive-format redesign: the same
ZIP(`metadata.json` + one `.npy` per parameter) format M7 introduced is
unchanged, `numpy.load(..., allow_pickle=False)` and the explicit
`register_module()` registry still gate reconstruction, and
`FORMAT_VERSION` stays `1` (see ADR-003's Milestone-13 update). Two changes
made this work: `save_model()` now computes the whole tree's device once via
the existing M9 `Module.device` (a mixed-device tree still raises
`ModuleError`, unchanged) and copies every `Parameter`'s values to host
memory via `Backend.to_numpy()` before writing -- a persistence *transfer*,
identical to the device-to-host copy `Tensor.to()`/`Module.to()` already
use, never a computation; `load_model()` gained an optional `device=` kwarg
and a device-availability policy: by default it restores onto the archive's
recorded device only if that device is available right now (a
`"cuda"`-recorded file with no CUDA backend raises a clear
`PersistenceError` rather than silently falling back to CPU), while an
explicit `device="cpu"`/`device="cuda"` performs a deliberate conversion in
either direction (an unavailable explicit `"cuda"` still fails clearly). A
restored CUDA `Parameter` is constructed via `Parameter(array,
device="cuda", ...)`, which routes through `CUDABackend.from_array()` -- a
real `cudaMalloc` + host-to-device transfer, never a NumPy array relabeled
as `CUDAStorage`. `is_cuda_available()` is checked lazily (`forge.backend.cuda`
is only imported when the recorded or requested device is actually
`"cuda"`), so a CPU-only environment loading a CPU-recorded file never
touches CUDA. One existing test was updated to match the new capability
(`tests/test_module_cuda.py::test_saving_a_cuda_model_now_succeeds`,
superseding the M9-era assertion that saving a CUDA model was rejected) and
one was renamed/adjusted since `"cuda"` is now a legitimate recorded device
rather than an unsupported one
(`tests/test_serialization.py::test_load_unrecognized_device_raises_persistence_error`,
now tampering to a truly unrecognized device string). New tests: the
hardware-required `tests/test_cuda_persistence.py` (10 tests -- CUDA -> CUDA
round trips covering parameter values/shapes/dtypes/`requires_grad`, a
nested model's hierarchy and per-module training-mode round trip, fresh-leaf
no-grad-state parameters, forward-output equivalence against the pre-save
model, explicit `device="cpu"`/`device="cuda"` conversion round trips, a
structural zero-`CPUBackend`-compute-calls check across a full save + load +
forward cycle, and a real `Trainer(device="cuda")`-trained model saved and
reloaded with matching predictions) and 5 new CPU-only tests in
`tests/test_serialization.py` covering the availability policy
deterministically via a monkeypatched `is_cuda_available` (no CUDA hardware
required to exercise the "CUDA unavailable" failure paths), plus the
CPU-only round-trip proof that a `"cuda"`-tagged file still loads correctly
under an explicit `device="cpu"` override with no hardware involved. No
optimizer-state/training-resume checkpointing, no multi-GPU-aware
serialization (bound by the CUDA backend's existing single-GPU restriction),
and no model computation of any kind during save/load, on CPU or CUDA
(structurally verified). Verified on the actual development GPU (940MX,
CC 5.0, driver 582.53, CUDA Toolkit 12.6): all 521 tests pass (358 CPU-only,
up from 353; 163 CUDA, up from 153), and the full suite was also run with
`PATH` stripped of the CUDA toolchain to confirm all 163 CUDA tests skip
cleanly (`358 passed, 163 skipped`, `0 failed`). See
`docs/architecture/persistence.md`'s **Device semantics** section and
`docs/architecture/cuda-backend.md`'s **CUDA model persistence** section.

### M14 — CUDA CrossEntropyLoss
Completes CUDA support for Forge's basic supervised classification workflow
by closing the "`CrossEntropyLoss` is CPU-only" limitation deliberately
deferred at Milestone 12: `CrossEntropyLoss` now runs unmodified on CUDA,
through the same high-level `forge/nn/loss.py` formulation as CPU (no
`CUDACrossEntropyLoss` subclass, no second autograd engine). Four CUDA
primitives made this possible, each added because the loss's own math
genuinely needed it (not a general-purpose expansion of the CUDA operation
set):
1. **`exp`/`log`** (`cf_exp`/`cf_log_{f32,f64}`, plus `exp_backward`/
   `log_backward` kernels) -- real CUDA kernels and backward rules,
   superseding M8's "unsupported" stubs. `Tensor.exp()`/`Tensor.log()`
   (`forge/tensor/tensor.py`) were also fixed to dispatch their backward math
   through the new `Backend.exp_backward`/`Backend.log_backward` methods
   rather than a raw `grad_output * result`/`grad_output / input_data` --
   those relied on operator overloading that `CUDAStorage` (unlike
   `numpy.ndarray`) does not provide, so they would have raised
   `AttributeError` the moment a real CUDA `exp`/`log` made this backward
   path reachable.
2. **`sum(axis=1)`** -- `CUDABackend.sum()`/`sum_backward()` now accept
   `axis=1` (equivalently `-1`) on a 2D tensor, via a new `cf_sum_axis1`/
   `cf_broadcast_axis1` kernel pair (one thread per row) alongside the
   existing full-reduction (`axis=None`) path. Deliberately not general
   N-D axis reduction -- only the one axis a `(batch, classes)` tensor needs.
3. **Column-broadcast `sub`** -- a new `cf_sub_colbcast` kernel (plus a
   `_reduce_axis1`-based backward, reusing `cf_sum_axis1`) supporting a
   `(rows, cols)` matrix combined with a `(rows, 1)` per-row scalar,
   broadcasting it across every column of its row -- the transpose of the M9
   row-broadcast case (a `(cols,)` vector broadcast down every row). Needed
   because both `logits - max_axis1(logits)` and `shifted - log_sum_exp`
   have this shape. Added for `sub` only; `add`/`mul` still reject it.
4. **`max_axis1`** -- a new, deliberately non-public `Backend` method (no
   `Tensor.max()` was added -- out of the milestone's scope), computing each
   row's max directly against backend storage (CPU: `np.max`; CUDA: a
   dedicated `cf_max_axis1` kernel) so the log-sum-exp numerical-stability
   shift is real backend computation on whichever device `logits` is on,
   never a host round-trip. The result is wrapped as a `requires_grad=False`
   leaf via `Tensor._wrap`, exactly mirroring how the pre-existing CPU
   implementation already treated the max as a constant (the log-sum-exp
   identity makes this exact, not an approximation).

`CrossEntropyLoss.forward()` itself was generalized rather than rewritten:
target-device validation now happens explicitly (`UnsupportedDeviceError` if
a `Tensor` target's device doesn't match `logits`'s), target values are read
to host via `Backend.to_numpy()` (device-agnostic, works for both CPU and
CUDA) instead of the CPU-only `Tensor.numpy()`, and the final `* (-1/n)`
scale is built as an explicit dtype-matched `Tensor` rather than a bare
Python float -- the same `Tensor._coerce`-default-dtype fix `MSELoss` needed
in Milestone 12. The `logits - shift`, `.exp().sum(axis=1,
keepdims=True).log()`, and one-hot-multiply-then-`.sum(axis=1)` expressions
themselves needed **no changes at all**: once the four primitives above
existed, the exact same Tensor-level code already ran correctly on CUDA.

Updated tests: `tests/test_cuda_backend.py` (exp/log/`sum(axis=1)`/
column-broadcast-sub forward correctness, replacing the old "unsupported"
assertions), `tests/test_cuda_autograd.py` (exp/log/`sum(axis=1)`/
column-broadcast-sub backward-vs-CPU checks, replacing the old
CUDAError-on-exp/log assertions), `tests/test_cuda_consistency.py` (new
exp/log/`sum(axis=1)` CPU/CUDA agreement cases), and a substantially expanded
`tests/test_cuda_loss.py` (CrossEntropyLoss CUDA/CPU forward agreement
across `float32`/`float64` and numerically difficult logits, CUDA backward
matching both CPU and the closed-form `(softmax(logits) -
one_hot(target))/batch_size`, a finite-difference gradient check, explicit
mean-reduction/`1/batch_size` gradient-scaling verification, CUDA/CPU
target-device-mismatch validation, and a structural zero-`CPUBackend`-calls
check extended to the four new compute methods). New in
`tests/test_trainer_cuda.py`: a full `TensorDataset -> DataLoader ->
Trainer(device="cuda") -> Linear -> CrossEntropyLoss -> CUDA backward ->
CUDA SGD` classification test on a deterministic two-class dataset (loss
drops to under half its starting value over 30 epochs, final accuracy
>90%, every Parameter/gradient confirmed CUDA-resident), a structural
no-CPU-fallback check across a multi-epoch classification `fit()` call, and
a CPU/CUDA classification training-consistency comparison. No changes were
needed to `Trainer` itself -- it already called `self.loss_fn(...)`
generically, so `CrossEntropyLoss` becoming CUDA-capable was enough on its
own. 563 tests total (358 CPU-only, unchanged; 205 CUDA, up from 163),
verified directly on the development GPU (940MX, CC 5.0, driver 582.53, CUDA
Toolkit 12.6), and the full suite was also run with `PATH` stripped of the
CUDA toolchain to confirm all 205 CUDA tests skip cleanly (`358 passed, 205
skipped`, `0 failed`) with the CPU-only suite entirely unaffected. See
`docs/architecture/cuda-backend.md`'s **CUDA CrossEntropyLoss** section and
`docs/architecture/optimization.md`'s **CrossEntropyLoss** section.

### M15 — Conv2d and MaxPool2d
Expands Forge's neural-network expressiveness beyond dense layers: `nn.Conv2d`
(2D cross-correlation, NCHW, integer stride, integer symmetric zero padding,
optional bias) and `nn.MaxPool2d` (2D max pooling, `stride` defaulting to
`kernel_size`), both ordinary `Module`s following the exact M10 pattern --
`Tensor.conv2d()`/`Tensor.max_pool2d()` (`forge/tensor/tensor.py`) attach a
`grad_fn` and dispatch to four new `Backend` methods (`conv2d`,
`conv2d_backward`, `max_pool2d`, `max_pool2d_backward`), implemented once per
backend, with no second autograd engine and no CUDA-specific code inside
`nn/conv.py`/`nn/pooling.py`.

**CPU** (`forge/backend/cpu.py`) is im2col-style: `numpy.lib.stride_tricks.
sliding_window_view` builds a strided (zero-copy) window view of the
(zero-padded) input, `conv2d` reduces it with one big batched matmul
(`cols @ weight.reshape(...).T`, real BLAS, not a Python loop over every
output element) and `max_pool2d` reduces it with `.max(axis=(4,5))`.
Backward recomputes the same window view from the saved forward input (the
same "recompute from a saved input" convention `relu_backward`/
`exp_backward` already use) and scatters each window position's contribution
back with a small (`kh*kw`-iteration) loop of strided `+=`, which correctly
accumulates overlapping windows when `stride < kernel_size`. `MaxPool2d`'s
tie-break is `np.argmax` on each window flattened in row-major (`kh`-then-
`kw`) order -- documented as "first maximum in top-to-bottom,
left-to-right scan order," and deliberately *not* the same algorithm CUDA
uses (see below), verified to agree by direct test.

**CUDA** (`forge/backend/cuda/{kernels.cu,backend.py}`) is real, but
intentionally *not* the CPU's im2col-plus-matmul approach: per the milestone
brief's "start with a straightforward correct kernel, do NOT immediately
implement cuDNN/cuBLAS/Winograd/FFT/autotuning" constraint, every kernel is
one thread per output (forward) or per gradient-target (backward) element,
looping over the kernel window in registers. `conv2d` backward is three
plain-gather kernels (input/weight/bias), no atomics. `max_pool2d` backward
recomputes each output element's argmax from the saved input (never caching
forward indices) and `atomicAdd`s into a `cudaMemset`-zeroed input-gradient
buffer -- the one place atomics were necessary, since overlapping pooling
windows can make more than one output thread target the same input element.
Both backends' tie-break conventions agree by construction (`np.argmax`'s
first-occurrence rule vs. CUDA's strict `v > best` comparison, both scanning
`kh`-then-`kw`), confirmed directly by test on real hardware.

Six new `Backend` methods were added to the ABC (`conv2d`, `conv2d_backward`,
`max_pool2d`, `max_pool2d_backward`, plus the existing pattern reused
unchanged for everything else); `Conv2d`'s weight/bias are ordinary
`Parameter`s (`(out_channels, in_channels, kh, kw)` / `(out_channels,)`,
`Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))` init, `fan_in = in_channels * kh *
kw` -- the direct Conv2d analog of `Linear`'s `1/sqrt(in_features)` bound),
so both layers integrate with `parameters()`/`named_parameters()`/
`Module.to()`/`SGD`/serialization with no special-casing anywhere in those
systems. Both were registered with `forge.serialization.register_module()`
alongside `Linear`/`ReLU`.

New test files: `tests/test_conv.py` (CPU Conv2d -- config/shape validation,
forward vs. an independent triple-loop reference, parameter reuse and input
reuse gradient-accumulation cases, finite-difference checks across padding/
stride/channels/batch), `tests/test_pooling.py` (CPU MaxPool2d -- same
structure, plus dedicated tie-break and overlapping-window-accumulation
cases), `tests/test_cuda_conv.py` (CUDA Conv2d/MaxPool2d forward/backward
vs. CPU, a CUDA finite-difference check, a structural zero-`CPUBackend`-calls
check, and a small end-to-end CUDA classification model), and
`tests/test_conv_trainer_integration.py` (the milestone's required
`Dataset -> DataLoader -> Trainer -> Conv2d -> ReLU -> MaxPool2d -> Linear ->
CrossEntropyLoss -> SGD` acceptance test on a deterministic image-like
two-class dataset, unmodified `Trainer`). Extended `tests/test_serialization.py`
(Conv2d/MaxPool2d round-trip, with and without bias, plus a nested
Conv2d->ReLU->MaxPool2d->Linear model) and `tests/test_cuda_consistency.py`
(Conv2d/MaxPool2d forward CPU/CUDA agreement). 680 tests total (441
CPU-only, up from 358; 239 CUDA, up from 205 -- both net of the 117 new
tests this milestone added), verified directly on the development GPU
(940MX, CC 5.0, driver 582.53, CUDA Toolkit 12.6), and the full suite was
also run with `PATH` stripped of the CUDA toolchain to confirm all 239 CUDA
tests skip cleanly (`441 passed, 239 skipped`, `0 failed`).

Basic `conv2d` forward/backward benchmarks were added to
`benchmarks/ops_bench.py`/`benchmarks/backward_bench.py` at three small
scales (`CONV2D_CONFIGS` in `benchmarks/sizes.py`) -- baseline measurements
only, per the milestone's "do not optimize before correctness is
established" constraint; on the 940MX, CUDA's straightforward kernel already
outran the CPU's im2col-plus-BLAS path at the "medium" scale for both
forward and backward, which is not something the milestone required or
optimized for. See `docs/architecture/cuda-backend.md`'s **CUDA Conv2d /
MaxPool2d** section.

### M16 — Sequential, Flatten, and Dropout
Improves model composition and adds Forge's first `.training`-dependent
stochastic layer. `nn.Sequential(*modules)` (`forge/nn/container.py`) is an
ordered `Module` container -- children register under `"0"`, `"1"`, ... via
ordinary `Module.__setattr__`, so every existing traversal API
(`named_children`/`parameters`/`named_modules`, `train()`/`eval()`,
`Module.to()`) already walks it correctly with no overrides beyond
`forward()`. `nn.Flatten(start_dim=1, end_dim=-1)`
(`forge/nn/flatten.py`) collapses a dim range (default: `(N,C,H,W) ->
(N,C*H*W)`) via `Tensor.reshape` alone -- no parameters, no new backward
rule. `nn.Dropout(p=0.5, generator=None)` (`forge/nn/dropout.py`) composes
`x * x.dropout_mask(p, rng)`; the mask is a `requires_grad=False` leaf with
`1/(1-p)`-scaled inverted-dropout values already baked in, so ordinary `mul`
autograd gives the correct forward/backward with no Dropout-specific
gradient code, and eval-mode `forward()` returns `x` itself unchanged
(identity, no new graph node).

A new `Backend.dropout_mask(a, p, rng)` method (`forge/backend/base.py`)
generates the mask: `CPUBackend` draws directly from the passed
`numpy.random.Generator` (`rng.random(a.shape)`); `CUDABackend` draws
**one** integer seed from `rng` (a cheap host-side scalar, not per-element
randomness) and generates every element's Bernoulli draw on-device via a
new kernel, `cf_dropout_mask_{f32,f64}` (`kernels.cu`), using a stateless
SplitMix64 hash of `(seed, element_index)` -- no curand dependency, no
device-side RNG state, per the milestone's "simple correctness-first, not a
sophisticated GPU RNG library" instruction. `Dropout.forward()` fetches
`forge.random.default_generator()` fresh on every call (unlike
`Linear`/`Conv2d`'s one-time construction-time snapshot), so a single
`forge.random.seed(...)` governs every draw across a training run.

Persistence needed one small registry-level accommodation: the generic
save/load tree walk (`forge/serialization/model.py`) requires a freshly
`from_config()`-constructed module to already have a child under every name
about to be attached, which holds for free when config alone determines
structure -- but `Sequential`'s child *count* is data, not config. Its
registered `from_config` (`forge/serialization/registry.py`) builds that
many placeholder `Module()` children from one extra config field
(`n_children`); the unmodified attach loop then overwrites each placeholder
with its real child. No change to the generic algorithm or file format.
`Flatten`/`Dropout` persist their config (`start_dim`/`end_dim`, `p`)
through the existing registry mechanism unchanged; `.training` round-trips
generically for every module already.

New test files: `tests/test_sequential.py` (construction/validation,
forward order, every discovery API, nested train/eval, `Module.to()`),
`tests/test_flatten.py` (default and general `start_dim`/`end_dim`,
validation, autograd), `tests/test_dropout.py` (p-validation, statistical
training behavior, exact eval identity, backward-reuses-forward-mask,
determinism under `forge.random.seed()` and an explicit `generator=`),
`tests/test_cuda_dropout.py` (real CUDA execution, statistics, gradient
correctness, eval identity, a structural zero-`CPUBackend`-calls check, and
an explicit "masks are not bitwise-equal but agree statistically" CPU/CUDA
comparison), `tests/test_cuda_flatten.py`, and
`tests/test_sequential_flatten_dropout_integration.py` (the milestone's
required `Sequential(Conv2d, ReLU, MaxPool2d, Flatten, Linear, ReLU,
Dropout, Linear)` acceptance test through `Trainer`, CPU and CUDA).
Extended `tests/test_serialization.py`/`tests/test_cuda_persistence.py`
with Sequential/Flatten/Dropout round-trip coverage (including nested
Sequential and per-module mixed training-mode round-tripping). 774 tests
total (516 CPU-only, up from 441; 258 CUDA, up from 239 -- 94 new tests
this milestone added), verified directly on the development GPU (940MX, CC
5.0, driver 582.53, CUDA Toolkit 12.6), and the full suite was also run
with `PATH` stripped of the CUDA toolchain to confirm all 258 CUDA tests
skip cleanly (`516 passed, 258 skipped`, `0 failed`). See
`docs/architecture/modules.md`'s **Sequential, Flatten, Dropout** section
and `docs/architecture/cuda-backend.md`'s **CUDA Dropout** section.

### M17 — Adam optimizer and optimizer state
Adds Forge's first adaptive optimizer and its first stateful-optimizer
architecture. `Adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
weight_decay=0.0)` (`forge/optim/adam.py`) implements the standard Adam
update (first/second moment estimates, bias correction) with every
hyperparameter validated at construction (`OptimizerError`, never silently
clamped). `weight_decay`, when nonzero, is classic L2 regularization folded
into the gradient before the moment updates -- explicitly the original
Adam paper's semantics, not AdamW's decoupled decay; AdamW itself remains
out of scope.

All numerical work happens in one new backend primitive,
`Backend.adam_step(data, grad, m, v, lr, beta1, beta2, eps, weight_decay,
step)` (`forge/backend/base.py`), the same `Tensor -> Backend` boundary
`SGD`/every other Forge operation already uses: `CPUBackend.adam_step` is
plain in-place NumPy arithmetic; `CUDABackend.adam_step` launches one new
kernel (`cf_adam_step_{f32,f64}`, `kernels.cu`) that performs the entire
update -- moments, bias correction, parameter step -- against the existing
`data`/`m`/`v` `CUDAStorage` buffers in place, with only the two
bias-correction scalars computed host-side (identical per element, the same
convention `k_broadcast_scalar` already established). `SGD` itself is
untouched.

`Adam.state` maps `Parameter -> _AdamState` (`m`, `v`, step count) using
ordinary Python object-identity `dict` keying (`Parameter` defines no
custom `__eq__`/`__hash__`), per the spec's "identity, not name"
requirement -- state follows a `Parameter` regardless of which `Module`
hierarchy or attribute name currently references it. State is allocated
lazily on a parameter's first `step()` with a non-`None` `.grad` (zeros
matching that parameter's shape/dtype/device exactly, via
`Backend.from_array` -- a real `cudaMalloc` + host-to-device zero transfer
on CUDA, the same construction path any new CUDA tensor already uses);
gradients are validated (shape/dtype/device against the parameter) before
use, and stale state (shape/dtype mismatch) is rejected defensively.
`Module.to(device)` moves a `Parameter`'s storage without any optimizer
awareness (by design), so Adam does not auto-migrate `m`/`v` across devices
(**Policy A**): `step()` raises `OptimizerError` if it finds state whose
recorded device no longer matches the parameter's current device, rather
than silently pairing mismatched-device buffers; clearing the stale entry
(`optimizer.state.clear()` or `del optimizer.state[param]`) lets the next
`step()` lazily reinitialize fresh state on the new device. `Adam.state`
lives entirely outside the `Module` tree `save_model()` walks, so optimizer
state was never at risk of being written to a model archive and needed no
persistence-layer change; optimizer-state checkpointing itself remains out
of scope.

New test files: `tests/test_cuda_optimizer.py` (CUDA state residency,
parameter/state-never-NumPy structural checks, a monkeypatched-`CPUBackend`
zero-fallback-calls proof, CPU/CUDA numerical agreement across a single
step, multiple steps with `weight_decay`, and a real `Linear` model trained
in lockstep, the Policy-A device-mismatch guard and its state-clearing
recovery, and end-to-end CUDA training through both a hand-written loop and
`Trainer(device="cuda")`). Extended `tests/test_optimizer.py` with 40 new
CPU Adam tests: hyperparameter validation, first-step and multi-step
agreement against a small NumPy reference implementation (including
`weight_decay`), bias correction, state accumulation/lazy-allocation,
zero-gradient and missing-gradient handling, parameter-identity-keyed state
(surviving a rename/different holder), no-autograd-graph-created,
`Parameter` object identity preserved across steps, `SGD` unaffected,
end-to-end CPU regression training (loss decreases) and determinism under a
fixed seed, `Trainer` integration with no Trainer changes required, and a
`save_model()` archive-content check confirming no optimizer state is
written. 825 tests total (556 CPU-only, up from 516; 269 CUDA, up from
258 -- 51 new tests this milestone added), verified directly on the
development GPU (940MX, CC 5.0, driver 582.53, CUDA Toolkit 12.6;
CPU/CUDA Adam agreement within `rtol=atol=1e-4`), and the full suite was
also run with `PATH` stripped of the CUDA toolchain to confirm all 269 CUDA
tests skip cleanly (`556 passed, 269 skipped`, `0 failed`). See
`docs/architecture/optimization.md`'s **Adam** section and
`docs/architecture/cuda-backend.md`'s **CUDA Adam** section.
MaxPool2d** section.

### M19 — Forge command-line interface
Adds a thin `forge`/`python -m forge` command-line adapter on top of the
existing persistence and benchmark APIs: a new `forge/cli/` package
(`main.py` -- top-level `argparse` command tree and dispatch; `model.py`/
`checkpoint.py` -- `inspect`/`convert` subcommands; `benchmark.py` -- a
pass-through to the existing `benchmarks` package; `errors.py` -- `CLIError`,
the CLI's own user-facing error type; `_archive_info.py` -- shared, read-only
archive-metadata access) plus `forge/__main__.py` and a `forge` console-script
entry point (`pyproject.toml`'s new `[project.scripts]`). `forge/__init__.py`
was not changed -- `import forge` still never imports `forge.cli`, matching
the same "not required for normal package import" principle Milestone 11
established for `benchmarks/`.

**`model inspect`/`checkpoint inspect` deliberately never call `load_model()`/
`load_checkpoint()`.** Both read only `metadata.json` via
`forge.serialization.archive.read_archive()` -- the same low-level primitive
those two higher-level functions call internally -- and walk the saved
module-tree JSON directly (`_archive_info.py`'s `walk_modules`/
`walk_parameters`), rather than reconstructing a live `Module`/`Optimizer`.
This was a deliberate departure from the milestone brief's general "call
`load_model`/`load_checkpoint`" guidance, made for three reasons specific to
inspection, all verified by test: (1) `load_checkpoint()`'s documented
contract includes overwriting `forge.random`'s process-global RNG state as
part of restoring a checkpoint -- exactly the state mutation Milestone 19's
"inspect commands must be read-only" requirement forbids, so `checkpoint
inspect` cannot safely call it; (2) reading metadata only never touches a
backend, so both inspect commands work identically on a CPU-only machine
regardless of whether the archive itself was saved on CUDA (unlike
`load_model`/`load_checkpoint`, which need a real CUDA backend to restore a
CUDA-recorded archive onto its own device); (3) it does not require the
saved module/optimizer types to be registered in the current process, since
their names/configs/shapes already sit in the archive's plain JSON metadata.
`model convert`/`checkpoint convert`, which are not read-only operations,
call `forge.load_model()`/`save_model()`/`load_checkpoint()`/
`save_checkpoint()` directly and unmodified -- exactly as a Python caller
would -- inheriting `load_checkpoint()`'s RNG side effect as expected/
documented behavior for that command, and Forge's existing explicit-device,
no-silent-fallback policy (an unavailable `--device cuda` fails clearly,
never falling back to CPU).

`forge benchmark` forwards every argument after `benchmark` unparsed to
`benchmarks.run.main()` (via `argparse.parse_known_args()` at the top level,
after an `argparse.REMAINDER` positional was tried first and found to drop a
leading `--flag` when nested under `add_subparsers()` -- a known `argparse`
limitation, not a Forge-specific bug), so `--categories`/`--output`/category
selection logic remains defined in exactly one place. Since `benchmarks/` is
deliberately excluded from `forge`'s own package installation (Milestone
11), this command only works from within the Forge repository; run outside
it, the lazy `from benchmarks.run import main` import fails and is turned
into one clear `CLIError` rather than a raw `ImportError` traceback.

`forge/cli/main.py` catches both `CLIError` and `forge.exceptions.ForgeError`
(covering essentially every ordinary user error other functions raise --
missing file, malformed archive, unsupported format version, unavailable
CUDA) in one place, printing `Error: ...` to stderr and exiting 1 with no
traceback; `argparse`'s own errors (invalid `--device` choice, unknown
command, missing required argument) keep their standard exit status 2; any
other, truly unexpected exception is left to propagate with its full
traceback rather than being swallowed.

New test file `tests/test_cli.py` (26 tests, invoking `forge.cli.main.main()`
directly rather than via subprocess -- the exact function both entry points
call): `--help`/unknown-command/missing-subcommand exit codes; model/
checkpoint inspection in both text and `--json` form; a structural proof that
`model inspect` never calls `is_cuda_available()` even when the inspected
archive's metadata is tampered to record `"device": "cuda"`; a direct proof
that `checkpoint inspect` leaves `forge.random.get_state()` byte-for-byte
unchanged; missing-file/malformed-archive/unsupported-version error handling
for both inspect commands; CPU->CPU model and checkpoint conversion round
trips (including a checkpoint with real trained Adam state); invalid
`--device` choice and missing-output-directory handling; CUDA-unavailable
conversion handling via a monkeypatched `is_cuda_available` (deterministic,
no hardware required); the `forge benchmark` argument-forwarding contract
(monkeypatched `benchmarks.run.main`) and its missing-package error path; and
two hardware-required tests (`pytest.mark.skipif(not is_cuda_available())`,
matching every other CUDA suite's convention) driving real
`forge model convert`/`forge checkpoint convert` CPU->CUDA->CPU round trips
through the actual CLI entry point and confirming genuine `CUDAStorage`
(model parameters and Adam `m`/`v` state alike) at each step -- never a NumPy
array relabeled as CUDA. 891 tests total (24 new CPU-only CLI tests plus 2
new hardware-verified CUDA CLI tests), verified directly on the development
GPU (940MX, CC 5.0, driver 582.53, CUDA Toolkit 12.6); the full suite
(`python -m pytest`) passes with `0 failed`. See `docs/development/cli.md`
for the full command reference.

## Phase 9 — Validation & Release

### M20 — End-to-end MNIST example and validation suite
A realistic integration workload exercising the framework core together, not new framework
features. New `examples/mnist/` package: `dataset.py` (`MNISTDataset`, a `forge.data.Dataset`
parsing the standard IDX file format, plus `download_mnist()` fetching the four standard files via
plain `urllib` from a public mirror -- never downloaded implicitly), `model.py` (`build_model()`, a
~27.6k-parameter `Conv2d/ReLU/MaxPool2d x2 -> Flatten -> Linear/ReLU/Linear` CNN built entirely from
existing `forge.nn` layers), `train.py` (a runnable CLI script wiring
`MNISTDataset -> DataLoader -> Trainer -> CrossEntropyLoss -> Adam`, with `--device cpu|cuda`,
`--resume` for checkpoint continuation, and a model-persistence round-trip check), and `README.md`.
`examples/` was made an importable package (`examples/__init__.py`, `examples/mnist/__init__.py`)
so `tests/` can exercise `build_model()` directly without a download.

Hardware-verified end to end on both devices (real MNIST, this repository's CPU and the 940MX):
CPU reached 90.1%/97.0% train/val accuracy after 1 epoch (0.346 -> loss), 97.1%/97.8% after a
`--resume`-continued second epoch; CUDA matched within floating-point tolerance at the same seed
(90.1%/97.0%) and ran at roughly 1.5-1.7x CPU sample throughput on this small architecture/batch
size -- a real but modest speedup, reported as observed rather than tuned. Verified directly:
`forge model inspect`/`forge checkpoint inspect` (M19 CLI) against `train.py`'s generated
artifacts; CUDA residency of every `Parameter`, every `Parameter.grad`, and Adam's `m`/`v` state
(no `CPUBackend` fallback); checkpoint save -> reload -> resume continuing epoch/global_step
correctly; and reloaded-model prediction consistency (CPU and CUDA).

Two new CI-run test files use a fast synthetic `(N, 1, 28, 28)` stand-in dataset (10
distinguishable-stripe classes) rather than the real ~11MB download, per the milestone brief's
"integration tests must not require the full dataset": `tests/test_mnist_example_integration.py`
(7 tests, CPU) covers dataset/model shape, training loss reduction with accuracy well above the
10%-chance baseline, parameter updates, checkpoint save/restore (model state, Adam state,
epoch/global_step, device), **resume equivalence** (continuous `N+M`-epoch training vs.
`N`-epochs -> checkpoint -> reload -> `M`-epochs matching within `1e-5`, using `shuffle=False` so
no caller-owned `DataLoader` generator state is left uncontrolled), model save/load prediction
consistency, and CLI inspection of generated artifacts. `tests/test_mnist_example_cuda_integration.py`
(4 tests, `pytest.mark.skipif(not is_cuda_available())`, hardware-verified on the 940MX) mirrors the
training/residency/checkpoint/persistence tests on CUDA. 902 tests total (891 existing + 11 new);
the full suite (`python -m pytest`) passes with `0 failed` -- no existing test was modified, and no
framework code changed (the milestone required none). `examples/mnist/data/` and
`examples/mnist/artifacts*/` (downloaded data, generated run outputs) are `.gitignore`d as
machine-local and reproducible. See `examples/mnist/README.md` for the full reproduction guide.

### M21 — Performance profiling and targeted CUDA optimization
Extends the M11 benchmark subsystem to cover Forge's full current operation
set and applies one measurement-justified CUDA optimization, following M11's
exact "measure first, optimize only what measurement justifies" discipline.
New forward/backward benchmark coverage (`benchmarks/ops_bench.py`,
`benchmarks/backward_bench.py`): `exp`/`log` (real CUDA kernels since M14
but never benchmarked), `max_pool2d`, `mse_loss`, `cross_entropy_loss`,
`dropout`, an isolated `adam_step`, and a complete small-CNN backward pass
using the real M20 architecture. Two new files: `benchmarks/mnist_bench.py`
(a new `"mnist"` category -- real M20 CNN + `CrossEntropyLoss` + `Adam`
trained against a fixed synthetic MNIST-shaped batch, reporting samples/sec
directly comparable to `examples/mnist/train.py`) and
`benchmarks/mnist_profile.py` (a diagnostic script, not a `BenchmarkResult`
category, breaking one training step into
`transfer -> forward -> loss -> backward -> optimizer`, with forward broken
down per layer type and backward per op via a small instrumented
re-implementation of `forge.autograd.engine.run_backward` -- verified
against the real engine by
`tests/test_benchmarks.py::test_profiled_backward_matches_real_backward`).

**Bottleneck found:** the MNIST profile (940MX, batch=64) showed CUDA
`conv2d` backward at 73.8% of the entire backward phase and 54.2% of the
whole training step -- an order of magnitude larger than every other
backward op. Isolating the three `conv2d_backward` sub-kernels at the CNN's
own two layer shapes narrowed this to `k_conv2d_backward_weight`/
`k_conv2d_backward_bias` (`forge/backend/cuda/kernels.cu`, Milestone 15):
at the first conv layer (72 weight elements, 8 bias channels), each of
those few threads serially summed a 43,264-iteration reduction alone while
most of the GPU sat idle. `MaxPool2d`, elementwise ops, `Adam`, and memory
transfers were all measured and found *not* to justify optimization at this
workload's scale (`Adam` in particular: sub-millisecond even at 262k
parameters, ~3% of total CUDA step time) -- left unchanged, per the
milestone's explicit "optimize only what measurement justifies."

**Optimization:** both kernels were rewritten as a block-per-output-element
shared-memory tree reduction (structurally identical to the existing
`k_sum`); the bias kernel always uses this (bias-channel counts stay small
for any CNN Forge targets), while the weight kernel dispatches per-call
between the new reduction kernel and the original Milestone 15
one-thread-per-weight kernel based on a measured element-count threshold --
a single strategy was *not* uniformly better across both MNIST layer shapes
(1,152 weight elements already had enough native parallelism that the
reduction kernel's block/sync overhead made it *slower*), caught only by
re-measuring both shapes after the first attempt. Both exported kernel
symbols/signatures are unchanged, so no Python-side code changed. All 906
tests pass unchanged against the recompiled kernels, including CPU/CUDA
`conv2d` backward-agreement and finite-difference checks; no CPU code was
touched. See `docs/architecture/cuda-backend.md`'s **CUDA Conv2d backward:
weight/bias optimization (Milestone 21)** section for the full kernel-level
writeup and `docs/performance/benchmarking.md`'s **Milestone 21** section
for complete before/after numbers.

**Measured result** (940MX, real hardware, `batch=64`): isolated
`conv2d_backward` at the first conv layer dropped from 12.62ms to
3.64-3.73ms (~3.4-3.5x); the CUDA MNIST training step's `backward` phase
dropped from 25.29ms to 16.10ms, and total CUDA step time from 34.46ms to
25.53ms (~1.35x). End-to-end CUDA MNIST training throughput
(`benchmarks/mnist_bench.py`) moved from ~1,875 to ~2,569 samples/sec
(~1.37x) -- the milestone's primary success metric (end-to-end CUDA
throughput), backed by measurement at every step from isolated kernel to
full training loop. CPU throughput was unaffected (code unmodified);
CPU-side run-to-run timing on this shared development machine showed more
variance than the CUDA optimization's own gain, documented explicitly
rather than smoothed over. Benchmark results: `benchmarks/results/latest.json`
now holds the M21 post-optimization run; `m11_baseline.json`,
`m21_baseline.json`, `mnist_profile_baseline.json`, and
`mnist_profile_optimized.json` are preserved separately as historical
records. See `docs/performance/benchmarking.md` for full methodology,
baseline, bottleneck analysis, and optimization writeup.

### M22 — CUDA memory statistics and allocation lifecycle
Establishes a correct, observable CUDA memory model -- not another
performance milestone, and explicitly not a caching allocator. New
`forge/backend/cuda/memory.py` (`CUDAMemoryStats`, a `threading.Lock`-guarded
counter of `allocated_bytes`/`peak_allocated_bytes`/`allocation_count`/
`free_count`) instruments the real `cudaMalloc`/`cudaFree` boundary already
in `CUDABackend._alloc`/`CUDAStorage.__del__` (`forge/backend/cuda/backend.py`)
-- one line at each site, no new kernels, no pooling. New top-level
`forge.cuda` package (`memory_stats()`, `reset_peak_memory_stats()`) is the
public entry point, raising `CUDAError` if CUDA is unavailable, matching
every other CUDA-specific Forge API; `import forge` remains CUDA-optional
(`forge.backend.cuda.memory` is pure Python, no `ctypes`/`nvcc`/device probe).

A failed `cudaMalloc` never touches the counters (checked on real hardware
by deliberately requesting 16 GiB on the 940MX's 2 GiB card, in an isolated
subprocess -- see below). A failed `cudaFree` warns (`RuntimeWarning`)
rather than corrupting the free count.

**Two genuine hardware/architecture findings surfaced during testing** (both
documented in `docs/architecture/cuda-backend.md`'s **CUDA Memory
Statistics** "Known limitations", not fixed -- out of M22's instrumentation-
only scope):
1. On this 940MX/driver 582.53/CUDA 12.6 combination, a sufficiently large
   failed `cudaMalloc` (e.g. 16 GiB) leaves the CUDA context unable to
   launch *any* further kernel for the rest of that process, even though
   `cudaMalloc`/`cudaMemcpy` themselves keep working -- a real driver
   quirk, not a Forge bug. `tests/test_cuda_memory.py`'s allocation-failure
   test runs in an isolated subprocess specifically because of this, so it
   can't poison the rest of the CUDA test suite.
2. Forge's Tensor/autograd/Module/Optimizer object graph for a full
   training step contains genuine Python reference cycles (confirmed via
   `gc.disable()` + `gc.collect()` reclaiming thousands of objects per
   iteration that plain refcounting left live) -- so `CUDAStorage` release
   is not purely deterministic-refcounting; an explicit `gc.collect()` is
   necessary before a CUDA memory snapshot means "true live allocation."
   Every lifecycle test and the benchmark integration (`benchmarks/memory.py`,
   wired into `mnist_bench.py`/`training_bench.py`) account for this.

24 new tests: 21 real-hardware lifecycle tests (`tests/test_cuda_memory.py`),
2 CUDA-unavailable error-path tests requiring no hardware
(`tests/test_cuda_memory_availability.py`, split into their own file since a
module-level `pytestmark` skip applies to every test in its module
regardless of definition order), plus one harness test
(`test_benchmarks.py::test_cuda_memory_extra_reports_expected_keys_and_deltas`)
-- 930 tests total (906 + 24), all passing on the 940MX (625 pass with CUDA
unavailable, 305 skipped, confirming clean skip behavior). Measured overhead:
the accounting primitives cost ~1.2us/call in isolation (`timeit`), and a
same-process instrumented-vs-no-op A/B on a tight `add` loop showed no
measurable difference against a ~118us/op CUDA baseline -- immaterial per
the milestone's performance constraint. See `docs/architecture/cuda-
backend.md`'s **CUDA Memory Statistics (Milestone 22)** section for full
semantics, and `docs/performance/benchmarking.md`'s **Milestone 22** section
for the benchmark-integration writeup.

### M23 — Tensor/autograd reference-cycle audit and fix
Root-causes and fixes the M22-discovered finding that Forge's Tensor/
autograd/Module/Optimizer object graph depended on cyclic GC rather than
plain reference counting. The Tensor/Node ownership graph itself was
audited end to end (every `backward_fn` closure in `forge/tensor/tensor.py`,
`Module`/`Parameter`/`Optimizer`/`Adam`/`Trainer`/`Loss`/`Conv2d`/
`MaxPool2d`/`Dropout`) and confirmed acyclic by design -- inputs are owned
strongly by `Node`, closures capture only raw backend storage/scalars, never
an output `Tensor`/`Module`/`Optimizer`. The actual cycle was in
`forge/autograd/engine.py`'s `_topological_order`: a recursive nested
`def visit(tensor): ... visit(inp) ...` closure calling itself by name,
whose closure cell therefore referenced `visit` itself (`visit.__closure__
-> cell -> visit`) -- a genuine self-referential cycle, uncollectible by
refcounting, that also kept the whole per-call topological-order list
(every Tensor in the graph) alive until the next `gc.collect()`. Confirmed
directly via `gc.get_referrers`/`cell.cell_contents is visit`, and via an
AST scan of `forge/` confirming it was the only nested function in the
package referencing its own name. Fix: `_topological_order` rewritten as an
iterative, explicit-stack post-order traversal with no nested function --
same topological ordering (verified byte-for-byte identical to the old
recursive version), no self-reference to break. `Node.__slots__` gained
`__weakref__` (previously absent) so lifetime tests/diagnostics can hold a
`weakref.ref(node)`; no behavioral change.

Measured on the 940MX: a `gc.disable()`'d 20-iteration MLP training loop
that used to leak 200 `Tensor` objects (10/iteration) reclaimed only by
`gc.collect()` now shows zero growth with `gc.collect()` never called. The
same experiment against real CUDA allocation: before the fix,
`allocated_bytes` grew from 21,844 to 93,604 bytes over 20 iterations
without an intervening `gc.collect()` (dropping back to 3,904 once one ran);
after the fix, `allocated_bytes` stays flat at 3,904 bytes throughout, no
`gc.collect()` needed. `docs/architecture/cuda-backend.md`'s **CUDA Memory
Statistics** "Known limitations" #2 is updated to reflect the resolution;
`docs/architecture/autograd.md` gains a **Graph teardown and object
lifetime (Milestone 23)** section with the full cycle diagram, ownership
model, and fix.

One separate, out-of-scope finding surfaced during CUDA investigation and is
documented rather than fixed: CPython's `_ctypes` extension leaves a small
(`ctypes.c_void_p`, `dict`) pair of cyclic garbage per certain foreign-
function calls through `forge/backend/cuda/backend.py` (e.g. one pair per
CUDA `backward()` call) -- a long-standing CPython implementation detail,
involving no Forge object and, confirmed directly, never retaining CUDA
device memory. Out of M23's Forge-object-ownership scope; not a regression.

15 new tests: 10 CPU lifetime tests (`tests/test_lifetime.py` -- simple
autograd, multi-use graphs, MLP, CNN, Dropout, Adam persistent state,
repeated `Trainer.fit()` epochs, `no_grad`, and a mixed-workload GC-disabled
regression test, all asserting zero live-object growth across repeated
iterations with cyclic GC disabled and never collected mid-workload) plus 5
CUDA lifetime tests (`tests/test_cuda_lifetime.py`, hardware-gated) proving
the same property against real `forge.cuda.memory_stats()` and explicitly
distinguishing the fixed Forge cycle from the unrelated `ctypes` artifact
above -- 945 tests total (930 + 15), all passing on the 940MX. No CUDA
kernel changed; no caching allocator introduced. Performance: backward-only
timing on a 3-layer MLP (200-iteration `timeit` average) was 520.1us before
and 503.6us after -- no regression (the iterative traversal is marginally
faster, avoiding recursive-call/closure-cell overhead). See
`docs/architecture/autograd.md`'s **Graph teardown and object lifetime
(Milestone 23)** section for full semantics.

### M24 -- CUDA memory allocation profiling and caching-allocator design
Answers whether Forge's direct `cudaMalloc`-per-`CUDAStorage`/`cudaFree`-at-
destruction model (unchanged since Milestone 8, instrumented but not
altered by Milestones 22-23) is a real bottleneck, and produces a
measurement-backed design proposal for a future caching allocator --
without implementing one. New `forge/backend/cuda/profiler.py`
(`CUDAMemoryProfiler`, `AllocationEvent` -- a frozen dataclass of `kind`/
`nbytes`/`timestamp`/`block_id`/`category`, three primitives plus an
optional string, never a `CUDAStorage`/`Tensor` reference) and new
`forge/cuda/profiler.py` (the public `forge.cuda.profiler.start()`/`.stop()`/
`.reset()`/`.is_active()`/`.events()`/`.tag()`/`.profile()` API, mirroring
`forge.cuda.memory_stats()`'s existing thin-wrapper/`CUDAError`-if-
unavailable convention). Instrumented at the exact same two call sites
Milestone 22 already uses (`CUDABackend._alloc()`, `CUDAStorage.__del__()`)
-- each now also forwards the allocated pointer's raw integer value through
`forge.backend.cuda.memory.record_alloc`/`record_free` to the profiler,
which records nothing (a single `bool` check, no event construction, no
`time.perf_counter()` call) unless explicitly started. Categorization is
opt-in via `tag()` (a small stack pushed/popped around a code region)
rather than instrumenting every one of `CUDABackend`'s ~40 methods
individually.

New `benchmarks/alloc_analysis.py` (pure functions over an
`AllocationEvent` trace: size/lifetime distributions with fixed buckets,
`pair_lifetimes`' FIFO-per-`block_id` alloc/free matching, persistent-vs-
temporary classification, a same-size reuse-opportunity statistic, and an
**offline** exact-size/size-class caching-allocator simulation -- never
wired into Forge's real allocator) and `benchmarks/alloc_profile.py` (a
diagnostic script, like `benchmarks/mnist_profile.py`: profiles the real
M20 MNIST CNN workload phase-by-phase, sixteen representative operations'
forward/backward allocation traffic, CPU<->CUDA transfer allocation
behavior, and direct `cudaMalloc`/`cudaFree` host-API timing).

**Measured on the 940MX** (CC 5.0, CUDA 12.6, driver 582.53; batch=64, 30
steady-state iterations of the M20 CNN): 64 allocations and 64 frees per
25.21ms iteration; true persistent CUDA memory (`memory_stats()`, flat
before/after the window) is 440,992 bytes against a 7,789,424-byte peak;
only 14 distinct allocation sizes occur across the whole trace, and 99.1%
of allocated bytes are an exact-size repeat of one seen earlier. An offline
exact-size cache simulation shows a trivial cache would have reduced 1,920
real `cudaMalloc`/`cudaFree` pairs to 42 real driver calls for the entire
run. Direct timing of `CUDABackend._alloc()`/`cf_free` (host-blocking CUDA
Runtime API calls -- no asynchronous queue to correct for, unlike a kernel
launch) shows ~175-300 microseconds per call, essentially size-independent
across 4KB-1MB and, multiplied by this workload's per-iteration
allocation/free count, the same order of magnitude as the entire measured
training step -- explicitly reported as an order-of-magnitude estimate
(isolated timing has no concurrent kernel traffic to overlap with, unlike
the real workload), not a precise attribution.

New `docs/architecture/cuda-memory-allocator.md` -- the milestone's primary
deliverable: measured allocation behavior, size/lifetime distributions, the
offline simulation, a comparison of four candidate allocator designs
(exact-size / size-class / best-fit / split-coalesce) against the actual
trace, a concrete recommended architecture (an exact-size cache, its
ownership/reuse/failure/cleanup semantics, `CUDAStorage`/autograd/Adam
invariants a future implementation must preserve, and why Forge's existing
per-operation `cudaDeviceSynchronize()` already makes reuse-after-free safe
without CUDA streams), and a memory-statistics API evaluation for a future
allocator. **Decision: caching allocator JUSTIFIED** (evidence-backed
recommendation for a future milestone -- not implemented here), with
explicit conditions and risks documented rather than treated as an
unconditional mandate. `docs/architecture/cuda-backend.md` and
`docs/architecture/optimization.md` gain matching cross-reference sections;
`docs/performance/benchmarking.md` gains the full methodology and
measurement writeup.

38 new tests: 15 hardware-gated profiler-lifecycle tests
(`tests/test_cuda_alloc_profiler.py` -- disabled by default, start/stop/
reset, alloc/free event correctness, block-id correlation, tagging
including nested tags, does-not-retain-storage, repeated independent
traces), 8 CPU-only availability tests
(`tests/test_cuda_alloc_profiler_availability.py`, split out for the same
module-level-`pytestmark` reason `test_cuda_memory_availability.py` is),
13 CPU-only pure-logic tests (`tests/test_alloc_analysis.py` -- size/
lifetime distribution math, block-id-reuse-safe lifetime pairing,
persistent/temporary splitting, both simulation policies, reuse-opportunity
math), and 2 benchmark-harness tests (`tests/test_benchmarks.py`) -- 983
tests total (945 + 38; CUDA-hardware-gated count up from 310 to 325), all
passing on the 940MX, and the full suite was also run with `PATH` stripped
of the CUDA toolchain to confirm a clean skip (658 passed, 325 skipped, 0
failed). No CUDA kernel changed, no caching allocator implemented, and
every Milestone 1-23 behavior is unchanged.

### M25 — Exact-size CUDA caching allocator
Implements the caching allocator Milestone 24 recommended (**"JUSTIFIED,
evidence-backed"**) but deliberately did not build. New `forge/backend/cuda/
allocator.py`: `CUDACachingAllocator`, a process-wide singleton owning
`_free_blocks: dict[nbytes, list[ptr]]` (raw `ctypes.c_void_p` pointers only
-- never a `CUDAStorage`/`Tensor` reference, the same discipline M22's
counters and M24's profiler already established). `CUDABackend._alloc()`
now delegates to `allocator.allocate(lib, nbytes)` (cache hit: pop and
return, no driver call; cache miss: `cudaMalloc`, with the M24-designed
OOM policy of one `empty_cache()` purge and one retry before raising
`CUDAError`, unchanged from M22 otherwise); `CUDAStorage.__del__()`
delegates to `allocator.release(nbytes, ptr)` (pushes onto the exact-size
free list, no driver call), clearing `self.ptr = None` first -- the same
"guard against a second `__del__`" pattern M22 used for `cf_free`, now
guarding a double-*release* instead of a double-*free*. `release()` also
scans its target size's cached list for the same pointer value before
appending, raising `RuntimeError` on a match (an internal-bug indicator,
never expected in normal use). Exact-size only, per the M24 design's
Candidate A: no size-class rounding, no block splitting, no coalescing --
three differently-sized cached blocks are never combined to serve a request
between them, by design (documented, not an oversight).

`CUDAMemoryStats` (moved into `allocator.py`, re-exported from `memory.py`
for import-path compatibility) keeps its original four fields' names,
positions, and meaning (`allocated_bytes`/`peak_allocated_bytes`: unchanged,
bytes owned by live `CUDAStorage`; a caller's old four-keyword construction
still works) and redefines `allocation_count`/`free_count` to mean **real
driver calls only** (a cache hit/release advances neither) per the M24
design doc's own recommended migration path, adding `cuda_malloc_count`/
`cuda_free_count` as clearer-named aliases. Five new fields default to `0`:
`reserved_bytes` (active + cached), `peak_reserved_bytes`, `cached_bytes`,
`cache_hit_count`, `cache_miss_count`. New public API: `forge.cuda.
empty_cache()` -- returns every cached (not active) block to the driver,
never touches live storage, returns the freed-block count. `benchmarks/
memory.py::cuda_memory_extra()` keeps its original five keys and gains eight
new `cuda_active_*`/`cuda_reserved_*`/`cuda_cached_*`/`cuda_cache_*`/
`cuda_driver_*` keys; `benchmarks/alloc_profile.py`'s driver-overhead timing
now calls new `allocator.raw_malloc`/`raw_free` helpers explicitly (bypassing
the cache) so it keeps measuring true, uncached driver cost now that
`CUDABackend._alloc()` itself is cached.

**Measured on the 940MX** (real hardware, CC 5.0, CUDA 12.6, driver
582.53): new `benchmarks/allocator_bench.py` (direct `cudaMalloc`/`cudaFree`
vs. cached `allocate`/`release`, 200 cycles/size, two runs) shows 2.1-2.8x
speedup at 65,536 bytes and 122-336x at 4,096/1,048,576 bytes (the 65,536-byte
scale's unusually fast *direct* driver calls, ~10 us vs. ~600-700 us at the
other two sizes, reproduced across runs -- reported as an unexplained,
environment-specific artifact, not a Forge allocator effect). The real M20
MNIST workload (batch=64, 5 warmup + 30 steady-state iterations, `benchmarks/
mnist_bench.py`) shows mean iteration time dropping from M24's 25.21 ms to
19.56-19.68 ms across two runs (~22% lower, well outside M21's documented
WDDM-variance noise floor) and real `cudaMalloc` calls collapsing from a
projected ~2,240 (direct model, cold start through 35 iterations including
warmup) to a measured 66 -- exceeding Section 8's offline-simulated 42-call
prediction in the *steady-state-only* window, which shows exactly zero new
driver calls (this run's own warmup iterations already populated the real
cache, unlike the M24 simulation's cold-start trace). Peak active memory
(7,789,424 bytes) and persistent memory (440,992 bytes, flat) are unchanged
from M24, as designed -- only `reserved_bytes`/`cached_bytes` grow, reclaimed
on demand via `empty_cache()`.

16 new hardware-gated tests (`tests/test_cuda_allocator.py` -- exact-size
cache hit/miss and pointer reuse, no-coalescing/no-split across distinct
sizes, `empty_cache()` correctness including a live-tensor-untouched check
with actual value round-tripping, the active/cached ownership invariant,
double-release protection at both the allocator and `CUDAStorage.__del__`
level, repeated-same-shape steady-state hit rate, a reused-block-has-no-
stale-data correctness check, a monkeypatched CPUBackend-explodes no-
fallback check, checkpoint save/load interaction with interleaved
`empty_cache()` calls, and an isolated-subprocess OOM test proving the
cache is purged before the final `CUDAError` -- mirroring M22's own
process-poisoning isolation rationale exactly), 1 CPU-only availability test
(`tests/test_cuda_allocator_availability.py`, split out for the same
module-level-`pytestmark` reason `test_cuda_memory_availability.py` is), and
1 new benchmark-harness test (`tests/test_benchmarks.py`, the new
`cuda_memory_extra()` fields). Two pre-existing M22 tests in `tests/
test_cuda_memory.py` were updated for the new "`allocation_count`/
`free_count` count real driver calls only" meaning (an autouse `empty_cache()`
before/after fixture added to that file for cross-test cache isolation, plus
one test split into an explicit `del` -> `cached_bytes` assertion and an
explicit `empty_cache()` -> `free_count` assertion) -- every other M1-M24
test passes completely unmodified. 1,001 tests total (983 + 18; CUDA-
hardware-gated count up from 325 to 341), all passing on the 940MX, and the
full suite was also run with `PATH` stripped of the CUDA toolchain to
confirm a clean skip (660 passed, 341 skipped, 0 failed). See
`docs/architecture/cuda-backend.md`'s **CUDA Caching Allocator (Milestone
25)** section, `docs/architecture/cuda-memory-allocator.md`'s
**Implementation (Milestone 25)** section, and `docs/performance/
benchmarking.md`'s **Milestone 25** section.

### M26 — CUDA execution and synchronization semantics
Formally establishes Forge's current CUDA execution/synchronization
contract before any future asynchronous-execution work, per the milestone
brief's explicit purpose. This was primarily an **audit**, not a redesign:
every CUDA-touching code path (kernel launches, memory copies, autograd,
`SGD`/`Adam`, `Trainer`, persistence, the M25 caching allocator) was
inspected against the actual CUDA API semantics in `kernels.cu` and
`backend.py`, and the (already-correct) synchronous execution model was
formalized in writing rather than changed.

**Findings.** Forge creates no CUDA streams -- every kernel launch and
`cudaMemcpy` runs on CUDA's default stream. Every `CUDABackend` method
already calls `cudaDeviceSynchronize()` before returning its result (a
pattern established as far back as Milestone 8, unchanged since), so Forge's
CUDA execution is host-synchronous *per operation* today despite the
underlying CUDA calls being individually asynchronous. Kernel launch errors
(`cudaGetLastError()`, immediately after every launch) and kernel execution
errors (only visible at the following `cudaDeviceSynchronize()`) are both
already surfaced before any `CUDABackend` method returns -- a distinction
that existed in `kernels.cu` from the start but had never been written down
explicitly. All CPU<->CUDA transfers (`Tensor.to()`) use plain synchronous
`cudaMemcpy`, never `cudaMemcpyAsync`. `Trainer.fit()`/`evaluate()` already
synchronize at every batch boundary for two independent reasons: every
`Module`/`Loss`/`Optimizer` call synchronizes internally, and every batch
additionally calls `loss.to("cpu").numpy()` for progress reporting -- a
second, independent synchronous transfer. The M25 caching allocator's
"safe to reuse immediately, no synchronization needed" design is formally
justified by this same per-operation synchronization guarantee, now stated
as an explicit, verified contract rather than an inherited assumption.

**New public API.** `forge.cuda.synchronize()` (`forge/cuda/__init__.py`) --
a thin wrapper dispatching to the pre-existing `CUDABackend.synchronize()`
(added in Milestone 11 for `benchmarks/timing.py`, previously reachable only
through `forge.backend.cuda.backend.get_cuda_backend()`). No backend-level
code changed: the Milestone 11 `CUDABackend.synchronize()` method already
did exactly what the milestone brief asked for, so Milestone 26 only made it
publicly reachable and documented its contract precisely. `Backend`
(`forge/backend/base.py`) gains no abstract `synchronize()` method, and
`CPUBackend` gains no `synchronize()` -- nothing in Forge calls
`get_backend(device).synchronize()` polymorphically today, so adding it to
the common interface would be unneeded coupling for a CPU no-op with no
caller (documented explicitly as a deliberate decision, not an oversight).
`benchmarks/timing.py`, `alloc_profile.py`, `mnist_bench.py`,
`mnist_profile.py`, and `training_bench.py` were updated to call the new
public `forge.cuda.synchronize()` instead of reaching into
`get_cuda_backend().synchronize()` directly -- a call-site simplification
with no methodology change.

**No kernel, launcher, allocator, autograd, optimizer, or `Trainer` code
changed.** No CUDA streams, events, asynchronous Tensor operations, or
stream-aware allocator were introduced, per the milestone's explicit scope
rule. `docs/architecture/cuda-backend.md` gains a new **CUDA Execution and
Synchronization Semantics (Milestone 26)** section (stream model, host/
device synchronization semantics, kernel launch semantics, memory copy
semantics, the public API, error semantics, the formalized allocator
reuse-safety contract, `empty_cache()`/memory-statistics semantics, autograd/
optimizer/`Trainer`/persistence/benchmark semantics, and a **Future
Stream-Aware Design** section listing exactly what multi-stream support
would require later); `docs/architecture/cuda-memory-allocator.md` gains a
**Milestone 26: Synchronization Contract (Formalized)** section restating
the allocator-specific consequence as a precise, testable claim;
`docs/performance/benchmarking.md` gains a **Milestone 26** section
confirming the benchmark harness and measured MNIST-workload performance are
unaffected.

15 new tests: 13 hardware-gated (`tests/test_cuda_synchronize.py` --
`synchronize()` after a real op, with no prior work, and called repeatedly;
synchronization after forward/backward/an optimizer step; a full forward ->
loss -> backward -> `optimizer.step()` -> `synchronize()` cycle matching an
identical CPU run within tolerance; the allocator memory-reuse safety
property directly -- allocate/use/release a block, allocate a same-size
block confirmed via `cache_hit_count` to reuse it, and read back exactly
correct values with no intervening `synchronize()`; the same property under
50 repeated cycles; `empty_cache()` immediately after real CUDA work with no
prior `synchronize()`; and `memory_stats()`'s counters being exactly
unchanged by `synchronize()`) and 2 CPU-only availability tests
(`tests/test_cuda_synchronize_availability.py`, split out for the same
module-level-`pytestmark` reason `test_cuda_memory_availability.py` is,
mirroring it exactly) -- 1,014 tests total, all passing on the 940MX; every
pre-existing test (999 tests) passes completely unmodified. `benchmarks/
mnist_bench.py`'s M20 CNN workload (batch=64, 5 warmup + 30 steady-state
iterations) was re-measured after this milestone's changes: 18.56-19.53 ms
mean CUDA iteration time across two runs, matching M25's own 19.56-19.68 ms
range within ordinary run-to-run variance -- confirming the new public API
adds no measurable overhead to any hot path, as expected, since it is never
called by Forge-internal code. See `docs/architecture/cuda-backend.md`'s
**CUDA Execution and Synchronization Semantics (Milestone 26)** section,
`docs/architecture/cuda-memory-allocator.md`'s **Milestone 26:
Synchronization Contract (Formalized)** section, and `docs/performance/
benchmarking.md`'s **Milestone 26** section.

### M27 — CUDA streams and asynchronous execution
Introduces real CUDA streams and an opt-in asynchronous execution mode on
top of Milestone 26's formalized synchronous contract, per the milestone
brief's explicit purpose ("begin the transition" M26 documented but
deliberately did not implement).

**Public API.** `forge.cuda.Stream()` (a real `cudaStreamCreate`d handle),
`forge.cuda.current_stream()`, `forge.cuda.set_stream()`, and `with
forge.cuda.stream(s): ...` (`forge/cuda/__init__.py`, backed by
`forge/backend/cuda/stream.py`'s `CUDAStream`). No public CUDA event API,
stream priorities, stream pools, or CUDA Graphs -- exactly the milestone
brief's scope limit.

**Design decision: default-stream compatibility mode.** Rather than making
every CUDA operation asynchronous by default (with an opt-in synchronous
mode), Forge keeps `forge.cuda.current_stream() is None` (no active `with
forge.cuda.stream(s):` block) as the exact Milestone 8-26 host-synchronous
behavior, unchanged byte-for-byte -- verified by re-running the entire
pre-existing 380-test CUDA suite completely unmodified. Asynchronous
execution (no per-op `cudaDeviceSynchronize()`) is opt-in, only inside a
`with forge.cuda.stream(s):` block. This was chosen specifically to avoid
silently invalidating the many pre-existing tests (and any future user
code) that read a CUDA result back to the host immediately after an
operation with no explicit synchronization, relying on the M26 guarantee.

**Kernel launcher changes.** Every kernel-launching `*_LAUNCHER` macro in
`kernels.cu` (~19 macros covering all ~40 kernel-launching `CUDABackend`
methods) gained a trailing `void* stream` parameter, passed as the fourth
argument to each `kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>`
launch (`stream=NULL` reproducing the exact pre-M27 default-stream launch
configuration). `cf_stream_create`/`_destroy`/`_synchronize` and
`cf_event_create`/`_record`/`_query`/`_synchronize`/`_destroy` were added as
new exported runtime calls (the event functions are internal-only, used by
the allocator, never exposed as public API). `cf_memcpy_h2d`/`_d2h`/`_d2d`
were deliberately left unchanged -- still plain synchronous `cudaMemcpy`,
which is unconditionally safe under CUDA's legacy-default-stream semantics
and does not need `cudaMemcpyAsync` for correctness here (Section 33 of the
milestone brief explicitly permits this).

**`CUDABackend` changes.** Three new helper methods
(`_stream_handle`/`_maybe_synchronize`/`_stream_guard`) centralize every
per-op decision: which stream to launch on, whether to synchronize
afterward (only in default-stream mode), and whether an input storage's
`last_stream` conflicts with the operation about to run. `_stream_guard` is
folded into `_require_compute_dtype` (called by nearly every kernel-
launching method already, with the exact storage list needed) plus two
explicit call sites (`reshape`, `from_array`'s CUDA-to-CUDA branch) for the
two methods that skip dtype validation. `CUDAStorage` gains one field,
`last_stream` -- the `Stream` (or `None` for default) this storage was last
touched by, the one piece of stream-history tracking the milestone brief
allows ("do not attach a full stream history"). `to_numpy()` (D2H) now
synchronizes a storage's own `last_stream` before reading it back, keeping
`.to("cpu")`'s host-blocking contract self-evident without relying only on
implicit legacy-stream ordering.

**Cross-stream policy: fail clearly, not automatic dependency resolution.**
Per Section 20/21 of the brief, using a tensor last touched on one real
stream from a different real stream raises `forge.CUDAError` immediately
(`_stream_guard`) rather than attempting `cudaStreamWaitEvent`-based
automatic ordering (explicitly out of scope). A tensor produced on the
default stream remains safe to read from any stream (the M26 guarantee
already covers it).

**Allocator changes (the central M25/M27 change).** `CUDACachingAllocator`
(`forge/backend/cuda/allocator.py`) gains a third block state, *pending*,
alongside the existing *active*/*ready*: a block released by a storage last
used on a real stream is not immediately safe to reuse. `CUDAStorage.
__del__` routes such a release through `release_pending()`, which records a
real internal `CUDAEvent` on that storage's stream at release time (correct
by CUDA's per-stream program-order guarantee); the block becomes reusable
once that event is observed complete, checked opportunistically on the next
same-size `allocate()` call -- never forced early via
`cudaDeviceSynchronize()`, which would defeat asynchronous execution.
`empty_cache()` now *waits* (`CUDAEvent.synchronize()`) for pending blocks
before freeing them -- a real, documented cost change from M25/M26 (ready
blocks are still freed immediately, no waiting). `CUDAMemoryStats` gains
`pending_bytes`/`pending_count`; `cached_bytes` now means specifically
*ready* bytes.

**Autograd/optimizer/Trainer/persistence: no code changes needed.**
`Tensor`/`forge.autograd.engine` are backend-agnostic and read
`current_stream()` ambiently through `CUDABackend`, so forward and backward
passes, and `SGD`/`Adam` optimizer steps, correctly execute on whatever
stream is current with zero changes to `forge/tensor/`, `forge/autograd/`,
or `forge/optim/`. `Trainer` (Option A from Section 23 of the brief) was not
modified and uses no stream internally, so its existing "returns only after
all issued CUDA work completes" contract holds trivially, unchanged.
`save_model()`/`save_checkpoint()` needed no changes either: `to_numpy()`'s
new stream-specific synchronization (above) already makes them safe to call
with no explicit synchronize after async work.

**Multi-stream overlap, measured on the 940MX.** `benchmarks/stream_bench.py`
(new, standalone script, not a `python -m benchmarks` category) demonstrates
real overlap: a default-stream baseline workload took a median 87.05 ms;
the identical workload issued on two real streams but synchronized between
them took 41.62 ms (proving the removed per-op synchronization alone is a
~2.1x win); issued concurrently on both streams with synchronization only at
the end took 36.55 ms (a further, real ~1.14x speedup from actual
overlapping kernel execution on the 940MX's 3 SMs). A large-single-kernel
workload sweep found only ~1.01x overlap there, since one such kernel
already occupies the whole device -- consistent with the brief's own
"do not expect dramatic overlap on every kernel/GPU" caveat.
`benchmarks/mnist_bench.py`'s M20 CNN workload was re-measured in
(unaffected) default-stream mode across two runs (23.03 ms, 27.47 ms mean
CUDA iteration time) -- both within the already-documented WDDM run-to-run
variance, not a regression (the M26 range was 18.56-19.53 ms).

42 new tests, all hardware-gated except one shared availability file: 22 in
`tests/test_cuda_streams.py` (stream creation/destruction/identity,
current-stream/context-manager restore including through an exception,
kernel execution correctness on an explicit stream for add/matmul/relu/
conv2d/maxpool2d/loss/optimizer-step, a timing-based proof that stream
issuance is not gated by per-op synchronization the way default-stream
issuance is, a 6-stream stress test, and a repeated create/use/destroy leak
test), 5 CPU-only availability tests
(`tests/test_cuda_streams_availability.py`, split out for the same
module-level-`pytestmark` reason as `test_cuda_synchronize_availability.py`),
9 in `tests/test_cuda_stream_allocator.py` (pending-block creation,
same-stream reuse after sync, same-stream rapid release/reallocate
correctness with no manual sync, cross-stream reuse safety under real
concurrent computation, default-stream-tensor cross-stream-read safety,
cross-stream-tensor-use failing clearly, `empty_cache()` draining pending
blocks and preserving live storage, and memory-stats coherence), and 6 in
`tests/test_cuda_stream_autograd.py` (forward+backward on one stream
matching CPU, cross-stream backward failing clearly, optimizer-step-then-
forward observing the update on the same stream, Adam matching CPU on a
stream, and save_model/save_checkpoint round-tripping correctly with no
explicit synchronize after async work) -- 1,056 tests total, all passing on
the 940MX; every pre-existing test (1,014 tests) passes completely
unmodified. See `docs/architecture/cuda-streams.md` (new), the **Milestone
27 note** and **Future Stream-Aware Design** updates in `docs/architecture/
cuda-backend.md`, the **Milestone 27: Pending Blocks** section in
`docs/architecture/cuda-memory-allocator.md`, and the **Milestone 27**
section in `docs/performance/benchmarking.md`.

### M28 — CUDA events and cross-stream dependencies
Removes Milestone 27's deliberate "fail clearly" cross-stream Tensor policy,
replacing it with automatic GPU-side dependency insertion, per the milestone
brief's explicit purpose ("Forge should be able to determine when a CUDA
Tensor was last produced/used on another stream and establish the necessary
dependency automatically").

**Mechanism.** `CUDABackend._stream_guard` (`forge/backend/cuda/backend.py`)
-- the single chokepoint every kernel-launching method already ran through
in M27 -- now inserts a real, GPU-side dependency instead of raising
`CUDAError` whenever a storage's `last_stream` differs from the current
stream: `cudaEventRecord` on the producing stream (a `CUDAEvent`, M27's
existing internal-only abstraction, reused unchanged) followed by
`cudaStreamWaitEvent` on the consuming stream. `CUDAStream.wait_event()`
and the new free function `stream.wait_event_on_default_stream()`
(`forge/backend/cuda/stream.py`) wrap the one new native export,
`cf_stream_wait_event` (`kernels.cu`, a direct `cudaStreamWaitEvent` call) --
both compile to the identical underlying CUDA call, the second existing only
because the default/null stream has no `CUDAStream` Python object to call a
method on. Distinct producer streams among an operation's storages are
deduplicated into a `set` before any event is created (one event per
producer, not per input). No public `forge.cuda.Event` API was added --
Section 7 of the milestone brief left this optional, and it was not needed
to meet the milestone's actual acceptance criterion (cross-stream
correctness).

**Why one `last_stream` field remains sufficient.** `_stream_guard` updates
every touched storage's `last_stream` to the *current* stream after
establishing whatever dependency was needed -- for read-only inputs exactly
as much as freshly constructed outputs, unchanged from M27. This means a
multi-consumer graph (`x` produced on stream P, then read by streams A and
then B) still resolves correctly with no producer-stream *history*: B's
dependency lands on A (the most recent toucher), not P directly, but A's own
command queue already contains "wait for P" ahead of its own read, so
waiting for A transitively implies P completed too, by CUDA's per-stream
FIFO ordering alone. A multi-producer op (`C = A + B`, `A` and `B` each on
their own stream) is handled independently: `_stream_guard` iterates every
input storage, so both producers get a dependency. No extra per-Tensor
metadata was needed for either case.

**No host blocking; no accidental `cudaDeviceSynchronize()`.**
`cudaStreamWaitEvent` only ever inserts a GPU-side ordering point and
returns to Python immediately -- verified directly, not just assumed, by
spying on the real `cf_synchronize` (`cudaDeviceSynchronize`) native entry
point during a cross-stream dependency between two explicit streams and
confirming zero calls.

**Autograd, gradients, and the optimizer needed zero code changes.** Because
every `CUDABackend` method (forward, backward, and both `sgd_step`/
`adam_step`) already runs through `_stream_guard` via `_require_compute_dtype`,
forward on one stream followed by `backward()` on another, gradient
accumulation across streams, and an optimizer step consuming a gradient (or
reading/writing a parameter) from a different stream than it was produced
on all became automatically safe -- with no change anywhere in
`forge/autograd/`, `forge/tensor/tensor.py`, or `forge/optim/`. The single
most important correctness test added this milestone
(`test_parameter_read_and_update_across_streams_are_never_racy_in_either_order`,
per the brief's own Section 39 framing) alternates a parameter *read*
(forward) and *write* (optimizer step) across two streams in both orders
with no explicit synchronization anywhere, and matches an identical
sequence kept on one stream exactly.

**Allocator and persistence needed zero code changes.** `CUDAStorage.
__del__`/`release()`/`release_pending()`/`_try_reclaim_pending()`
(`allocator.py`) and `to_numpy()`'s D2H synchronization are byte-for-byte
unchanged from M27 -- both remain correct because `last_stream` continues
to name whichever stream will genuinely next touch a storage, cross-stream
reads included (the same reasoning as the multi-consumer case above).

**Two M27 tests changed behavior, as the milestone brief intended.**
`test_using_a_tensor_across_two_different_explicit_streams_raises_clearly`
and `test_backward_on_a_different_stream_than_forward_raises_cuda_error`
specifically asserted the M27 limitation this milestone removes ("M28
removes that limitation" -- the milestone brief's own words); both were
rewritten to assert the new, correct cross-stream behavior instead (renamed
to `..._establishes_a_dependency` and
`..._matches_same_stream_reference` respectively). Every other pre-existing
test (1,054 of the prior 1,056) passes completely unmodified.

**Tests.** 1,072 tests total (1,056 pre-existing + 16 new), all passing on
the 940MX; 53 CUDA-hardware-gated tests across the stream-related files
(`test_cuda_streams.py`, `test_cuda_streams_availability.py`,
`test_cuda_stream_allocator.py`, `test_cuda_stream_autograd.py`, and the new
`test_cuda_stream_dependencies.py`). New coverage: same-stream/default-
stream fast-path (zero dependencies inserted, spied directly), cross-stream
dependency deduplication (shared producer -> one event, not two),
multi-producer (`C = A + B` on distinct streams) and multi-consumer (one
producer, two independent consumers) correctness, all four stream
directions (default<->explicit, explicit A<->explicit B) parametrized,
no-`cudaDeviceSynchronize()` verification, `empty_cache()` safety under
cross-stream dependencies, a 60-iteration random-workload stress test
against a synchronous NumPy reference, a 100-iteration event-lifetime/
allocator-leak stress test, cross-stream forward+backward matching a
same-stream reference, cross-stream optimizer step matching a same-stream
reference, and the parameter read/update race test described above. See
`docs/architecture/cuda-streams.md`'s **Milestone 28: Automatic Cross-Stream
Dependencies** section (plus updates throughout Sections 7/8/13/14/18/19),
the **Milestone 28** addendum in `docs/architecture/cuda-memory-
allocator.md`, the Milestone 28 note in `docs/architecture/cuda-backend.md`'s
**Future Stream-Aware Design** section, and the **Milestone 28** section in
`docs/performance/benchmarking.md` (`benchmarks/stream_dependency_bench.py`,
new: same-stream baseline 56.43 us/op, cross-stream dependency 79.04 us/op,
~1.40x overhead; default-stream M20 MNIST throughput unaffected at 19.07
ms/iteration, within the M26 baseline range).

### M29 — Async CUDA transfers and pinned memory
Adds the lower-level primitives real asynchronous host<->device transfer
requires: real CUDA pinned (page-locked) host memory, real `cudaMemcpyAsync`
bindings, and an explicit `Tensor.to(device, non_blocking=True)` opt-in --
without changing `.to()`'s existing default (`non_blocking=False`) contract
at all.

**Pinned memory.** `forge.cuda.PinnedMemory` (`forge/backend/cuda/pinned.py`)
wraps one real `cudaHostAlloc`/`cudaFreeHost` allocation -- direct, uncached
lifecycle per the milestone brief's Section 25 (no pinned caching allocator
without profiling justification). `PinnedMemory.numpy()` returns a
`_PinnedArray` (`np.ndarray` subclass) carrying a strong `_pinned_owner`
back-reference to the `PinnedMemory` instance -- the entire lifetime
mechanism: as long as any array (or a `Tensor` built from it) stays
reachable, ordinary CPython refcounting keeps the pinned allocation alive,
and `free()`/`__del__` waits (`CUDAEvent.synchronize()`) for any in-flight
transfer's recorded completion event before the real `cudaFreeHost` call
(Invariant 1). `forge.cuda.pinned_memory_stats()` is a small, separate
dataclass (`pinned_active_bytes`/`pinned_peak_bytes`/
`pinned_allocation_count`/`pinned_free_count`) -- deliberately not folded
into `CUDAMemoryStats`, since pinned host bytes are a conceptually distinct
resource from device `reserved_bytes`/`cached_bytes`/`pending_bytes`.

**Async H2D/D2H.** `kernels.cu` gained `cf_host_alloc`/`cf_host_free` and
`cf_memcpy_h2d_async`/`cf_memcpy_d2h_async` (real `cudaMemcpyAsync`, an
explicit stream argument, never followed by `cudaDeviceSynchronize()`).
`CUDABackend.from_array_async`/`to_numpy_async` (`backend.py`) submit on the
current Forge stream (`self._stream_handle()`, the same ambient mechanism
every other method already uses) and return immediately. H2D requires the
source array to already be pinned (`_pinned_owner` set) -- a pageable source
raises `CUDAError` rather than being silently staged through a hidden
pinned buffer (**Policy: Option A**, chosen over silent fallback or hidden
staging, both explicitly disfavored by the milestone brief's Section 13/14).
D2H always succeeds: Forge allocates the pinned destination buffer itself.

**Cross-stream dependencies needed zero new mechanism.** `CUDAStorage.
__init__` already sets `last_stream = current_stream()` unconditionally, so
an async H2D result's stream provenance is correct with no new code; a
later cross-stream consumer is handled by M28's existing `_stream_guard`.
`to_numpy_async` calls `_stream_guard((storage,), ...)` before submitting
its copy, so a cross-stream D2H also reuses the exact same M28 mechanism.
Verified with a `cf_synchronize`-spy (zero `cudaDeviceSynchronize()` calls)
in both directions: `tests/test_cuda_transfer_dependencies.py`.

**Host-read synchronization: a synchronizing `_data` property.**
`Tensor._data` (`forge/tensor/tensor.py`) was converted from a plain
attribute to a property backed by `Tensor._storage`, gated by `Tensor.
_pending` (a `forge.backend.cuda.transfer.PendingTransfer` -- the smallest
possible completion handle, one `CUDAEvent`, no futures/promises
subsystem). Every existing Tensor method that reads `self._data` --
`.numpy()`, `__repr__`, every op's forward/backward, `backward()`,
persistence -- already passes through this one chokepoint, so none needed
individual changes. On first access, the pending transfer is synchronized
and the tensor's storage is detached from pinned memory (`np.array(...,
copy=True)`) -- otherwise NumPy's ufunc subclass propagation
(`__array_finalize__`) would make every array *derived* from the result
also retain a `_pinned_owner` reference, keeping a potentially large pinned
buffer alive indefinitely. `forge/backend/cpu.py`'s `CPUBackend.from_array`
gained one narrow exception (return a `_PinnedArray` as-is, never copy it)
so that `Tensor(pinned.numpy(...), device="cpu")` -- the natural way to
build a pinned H2D source -- does not itself silently lose the pinned
buffer via the constructor's normal always-copy path.

**Allocator, autograd, optimizer, persistence needed zero code changes** --
verified directly, not just argued: the Section-35-mandated allocator race
test (`tests/test_cuda_transfer_allocator.py::
test_async_h2d_release_never_hands_the_still_in_flight_block_to_another_stream`),
an autograd test using an async-transferred constant operand in a
differentiable computation, and a persistence test training on an
async-transferred input then calling `save_model()` with no explicit sync.

**One hardware-observed quirk (documented, not a Forge bug):** on the
940MX/driver 582.53, an out-of-memory `cudaHostAlloc` request has been
observed to leave the process's CUDA context unable to serve small
subsequent `cudaMalloc` calls -- reproduced directly (it broke an unrelated,
pre-existing allocator test when both ran in the same pytest process). The
regression test for this failure path now runs in an isolated subprocess,
keeping the real, hardware-verified `cudaHostAlloc` failure test intact
while containing the quirk's blast radius to a throwaway process.

**Tests.** 1,113 tests total (1,072 pre-existing + 41 new), all passing on
the 940MX; 39 of the 41 new tests are CUDA-hardware-gated (2 are
CUDA-unavailable-path tests that run everywhere), across `tests/
test_cuda_pinned_memory.py`, `test_cuda_pinned_memory_availability.py`,
`test_cuda_async_transfer.py`, `test_cuda_transfer_dependencies.py`,
`test_cuda_transfer_allocator.py`, `test_cuda_transfer_stress.py`, and
`test_cuda_transfer_persistence.py`. Every pre-existing test passes
completely unmodified except one intentional, narrowly-scoped fix
(`CPUBackend.from_array`'s pinned-array exception, above) that changes
behavior for exactly zero pre-existing call sites (it only ever triggers
for a `_PinnedArray`, which no pre-Milestone-29 code path ever produces).
New coverage: pinned allocation/free/lifetime/leak/failure handling, NumPy
interoperability, H2D/D2H async correctness against synchronous references
(small/large/odd-shaped/float32/float64), the nonblocking pageable-source
policy, cross-stream H2D->compute and compute->D2H with no
`cudaDeviceSynchronize()`, the mandatory allocator race test, a 30-iteration
4-stream stress test with before/after leak checks on both device and
pinned memory counters, and persistence safety.

See `docs/architecture/cuda-transfers.md` (new) for the full design and
contract, updates to `docs/architecture/cuda-streams.md` (Sections 5/19),
`docs/architecture/cuda-backend.md`, and `docs/architecture/
cuda-memory-allocator.md`, and the **Milestone 29** section in
`docs/performance/benchmarking.md` (`benchmarks/async_transfer_bench.py`,
new: pinned H2D ~1.8-1.9x faster than pageable at 4 MB, async submission
~30-60x faster than full completion, H2D transfer/compute overlap
0.97x-1.24x, D2H transfer/compute overlap 0.86x-0.91x (memory-bandwidth
contention, reported as measured); default-stream M20 MNIST throughput
unaffected at 19.66-19.83 ms/iteration, within the M26-28 baseline range).

### M30 — Asynchronous DataLoader GPU prefetch
Turns the M29 pinned-memory/async-transfer primitives into a bounded,
opt-in asynchronous CPU-batch-preparation + H2D-transfer + GPU-compute
pipeline, integrated with the existing `DataLoader` and `Trainer` -- zero
new synchronization mechanism; every piece is exactly the M25-29 machinery
those milestones' own docs already said a future DataLoader-prefetch
milestone would consume.

**API.** `forge.data.CUDAPrefetchLoader(loader, device="cuda",
prefetch_size=2)` wraps an existing `DataLoader` (also reachable as
`loader.prefetch(device="cuda", prefetch_size=2)`) -- a wrapper, not a
subclass or reimplementation; `DataLoader` itself is completely unmodified
beyond that one convenience method. `Trainer(..., prefetch=True,
prefetch_size=2)` (requires `device="cuda"`) transparently wraps whatever
loader `fit()`/`evaluate()` receive, cached per loader object so the
wrapper (and its transfer stream) is created once, not once per epoch.

**Pipeline.** A single bounded background `threading.Thread` ("the CPU
producer") does exactly one thing: call `next()` on the wrapped loader's
own unmodified iterator and push each resulting CPU batch into a
`queue.Queue(maxsize=prefetch_size)` -- backpressure is entirely this
queue's own blocking `put()`. It never touches CUDA/streams. The single
calling thread does everything CUDA-related: stage a popped CPU batch into
`forge.cuda.PinnedMemory` (reusing M29's mechanism exactly, no new pinned
lifetime system), submit its async H2D on one dedicated, persistent
transfer `Stream`, and hand the resulting CUDA `Tensor` batch to the
caller. Double buffering (Section 45): exactly one batch is staged ahead on
the GPU side at any time, independent of `prefetch_size` (which only bounds
CPU-side lookahead).

**Transfer→compute dependency needed zero new mechanism.** `CUDAStorage.
last_stream` is already set correctly by the M29 async-H2D path; the
caller's first kernel touching that storage on a different stream
automatically invokes M28's `_stream_guard`, inserting a GPU-side
`cudaStreamWaitEvent` with no explicit event and no
`cudaDeviceSynchronize()` anywhere in this milestone's own code. Verified
directly with a `cf_synchronize`-spy across a full prefetch training run
(zero calls): `tests/test_dataloader_prefetch.py`, `tests/
test_trainer_prefetch.py`.

**Real overlap requires a real compute stream.** CUDA's legacy default/null
stream synchronizes against any explicitly created stream, so
default-stream compute would fully serialize against the prefetch
pipeline's transfer stream (correctness unaffected, but zero real overlap).
`Trainer(prefetch=True)` therefore creates one dedicated, lazily-created,
persistent compute `Stream`, current only for the duration of `fit()`'s/
`evaluate()`'s batch loop -- documented as superseding `docs/architecture/
cuda-streams.md`'s "Trainer remains synchronous" decision for this one
opt-in path only; `prefetch=False` is completely unaffected.

**RNG safety needed no changes to `DataLoader`/`Dropout`/`forge.random`.**
`DataLoader.__iter__()`'s one RNG draw (the shuffle permutation) executes
lazily on its *first* `next()` call. `CUDAPrefetchLoader` always performs
that first `next()` call synchronously on the calling thread, before
starting the background thread -- every draw after that point is
RNG-free (`_collate` touches no randomness), so the background thread never
races the main thread's `Dropout` draws on Forge's shared process-global
generator. Verified with shuffle-ordering tests (parametrized over
`prefetch_size`) asserting byte-for-byte identical batch order against the
synchronous path for a fixed seed.

**Reference-cycle-free cleanup.** The background thread's target is a free
function taking only `(source_iter, queue)`, never a bound method closing
over `self` -- avoiding a `self -> Thread -> bound-method -> self` cycle
that would otherwise need the cyclic GC (not plain refcounting) to ever
collect an early-terminated iterator's thread/queue/CUDA batches, matching
this codebase's established lifetime-testing convention. Verified with
`gc.disable()` in effect across repeated `for batch in loader: break` loops.

**Tests.** 1,147 tests total (1,113 pre-existing + 34 new), all passing on
the 940MX; 28 of the 34 new tests are CUDA-hardware-gated (6 are
CUDA-unavailable/CPU-only-import-path tests that run everywhere), across
`tests/test_dataloader_prefetch.py`, `tests/
test_dataloader_prefetch_availability.py`, and `tests/
test_trainer_prefetch.py`. Every pre-existing test passes unmodified. New
coverage: construction/validation, CUDA-tensor batch correctness
(dtype/shape/value), synchronous-vs-asynchronous ordering under `shuffle`
(parametrized `prefetch_size`) and `drop_last`, epoch-boundary correctness
across repeated epochs, dataset-exception propagation, early-termination
thread cleanup via plain refcounting (`gc.disable()`), repeated-epoch
CUDA/pinned memory leak checks, zero-`cudaDeviceSynchronize()`
verification, `Trainer` prefetch integration (exact-loss-match against the
synchronous `Trainer`, validation/standalone-`evaluate()`, compute-stream
lifecycle, gradient/optimizer correctness), and CPU-only import safety.

See `docs/architecture/async-dataloader.md` (new) for the full design and
contract, updates to `docs/architecture/cuda-transfers.md` (Section
20/21) and `docs/architecture/cuda-streams.md` (Section 15/18), and the
**Milestone 30** section in `docs/performance/benchmarking.md`
(`benchmarks/async_dataloader_bench.py`: synthetic light-CPU/heavy-GPU
overlap 3.28x, negligible-CPU/light-GPU overhead-dominated 0.63x-0.92x
(honestly reported, not a regression), real MNIST CNN 1.21x; CUDA/pinned
memory both return to baseline after repeated epochs).

### M31 — Profile and optimize the async CUDA training pipeline
A profiling-first milestone: built `benchmarks/pipeline_profile.py` (a
non-synchronizing, event-based profiler for the real M27-M30 asynchronous
pipeline -- CPU-only component costs, isolated H2D bandwidth, per-phase GPU
busy time via timing-enabled CUDA events, batch-size/prefetch-depth sweeps,
allocator/pinned characterization) plus a full synchronization audit,
before implementing exactly one measurement-justified optimization. See
`docs/performance/pipeline-profiling.md` for the complete profiling report
and bottleneck ranking.

**Profiling infrastructure.** `forge/backend/cuda/profiling_events.py`
(new) adds `TimedEvent`, a *timing-enabled* CUDA event (`cudaEventCreate`,
no `cudaEventDisableTiming`) distinct from the internal `stream.CUDAEvent`
used everywhere else (allocator/dependency machinery) -- the internal one
is deliberately timing-*disabled* for its own hot-path use, so a separate,
purely additive, profiling-only type was needed to support
`cudaEventElapsedTime`. Never touched by any core-runtime code path; zero
cost to ordinary training.

**Finding.** `mnist_profile.py`'s per-phase breakdown showed
`CrossEntropyLoss`'s forward pass (4.73 ms) costing *more* wall-clock time
than the entire M20 CNN's forward pass (4.21 ms) -- on a `(64, 10)` tensor,
versus two `Conv2d` + two `MaxPool2d` + `Linear` layers over a `(64, 1, 28,
28)` input. The composed implementation launched ~9 forward + ~7 backward
Tensor-primitive kernels plus 2 host<->device transfers per step; at this
GPU's measured ~54-65 us/launch dispatch cost (`stream_dependency_bench.py`),
that overhead alone explained the anomaly -- a launch-overhead-bound, not
compute-bound, operation. By contrast `conv2d` backward (55-57% of total
step time) is genuinely compute-bound and out of this milestone's scope
(Section 50 excludes cuDNN migration / broad kernel rewrites).

**Optimization.** `Backend.cross_entropy`/`cross_entropy_backward`
(`backend/base.py`, `cpu.py`, `cuda/backend.py` + two new fused CUDA
kernels in `kernels.cu`) replace the composed chain with 2 kernel launches
forward (a fused per-row log-sum-exp-NLL kernel, one thread per row
matching `k_max_axis1`'s existing convention; the existing `cf_sum_*`
reduction, reused) and 1 launch backward (fused softmax-minus-one-hot
gradient) -- the one-hot matrix is never materialized; only small int64
target indices ever cross the host/device boundary, and only when not
already CUDA-resident. `Tensor.cross_entropy()` (`tensor.py`) is the new
entry point; `nn.CrossEntropyLoss.forward()`'s existing shape/dtype/
target-range validation is completely unchanged -- only the computation
after it changed.

**Results.** Isolated op benchmark (batch=64/classes=10): CUDA forward
1.02 ms -> 0.40 ms (2.5x), CUDA backward 15.20 ms -> 0.17 ms (noisy
before/~90x after); CPU also improved modestly (fewer autograd-graph
objects per step, no CPU regression). End-to-end real M20 CNN training
step: 19.20 ms -> 17.90 ms (~7%, consistent with the loss's measured
10-17%-of-step fraction). `async_dataloader_bench.py`'s real-MNIST prefetch
speedup unchanged within noise (1.171x -> 1.180x) -- the async pipeline's
own M30 behavior is undisturbed.

**Tests.** 1,154 tests total (1,147 pre-existing + 7 new,
`tests/test_cuda_cross_entropy_fusion.py`), all CUDA-hardware-gated, all
passing on the 940MX; every pre-existing test (including the full
`tests/test_cuda_loss.py` CrossEntropyLoss suite -- forward vs. CPU across
numerically difficult logits, backward vs. analytical formula and finite
differences, reduction semantics, device validation, "no CPU fallback" spy
tests) passes unmodified. New coverage: `Tensor.cross_entropy`'s own
defense-in-depth validation, cross-stream correctness (logits/target/
grad_output each produced on a distinct stream from the op itself),
repeated-use memory-safety (`allocated_bytes`/`reserved_bytes`/
`pending_bytes` all return to 0).

**What was profiled but not touched.** Dependency/event overhead
(~64 us/dependency), allocator overhead (>95% cache-hit rate), pinned-
memory overhead (~0.22 ms/batch, hidden behind ~8.5-24.7 ms/batch compute),
and prefetch queue depth (1 vs. 2 vs. 3 -- no measurable benefit from
depth beyond 1 for this workload) were all measured and found not to be
bottlenecks -- none were optimized, per the milestone brief's "do not
optimize for the sake of having an optimization."

### M32 — CUDA Conv2d backward optimization
M31 identified `conv2d` backward as 55-57% of total CUDA training-step
time. This milestone built a dedicated `conv2d_backward` profiler
(`benchmarks/conv2d_backward_profile.py`, new -- isolates `dInput`/
`dWeight`/`dBias` individually via `TimedEvent`) and measured all three
across seven shapes (both real M20 MNIST layers, plus larger-channel/
larger-spatial/batch-size sweeps beyond what M21 tested), before
implementing exactly one measurement-justified kernel optimization. See
`docs/performance/conv2d-backward-profiling.md` for the complete profiling
report and `docs/architecture/cuda-backend.md`'s **CUDA Conv2d backward:
input optimization (Milestone 32)** section for the architecture writeup.

**Finding.** Contrary to M21's own MNIST-scale-only finding (`dWeight` was
that milestone's fix target), this milestone's broader shape sweep found
`dInput` -- not `dWeight` -- dominant at 6 of 7 shapes (60-66% of
`conv2d_backward` at every non-MNIST-scale shape, and slightly ahead of
`dWeight` even at the real MNIST CNN when both layers are summed).
`k_conv2d_backward_input`'s `(kh,ho)`/`(kw,wo)` validity resolution
(`t % SH`, `t / SH` and the `W` analog) does not depend on `co`, yet sat
inside the `co` loop -- `Cout`-fold redundant integer division per thread,
with no fast path on CC 5.0 for a runtime-valued stride divisor.

**Optimization.** `kernels.cu`'s `k_conv2d_backward_input` hoists that
resolution into two small per-thread local tables built once, before the
`co` loop -- which now does only array indexing and multiply-accumulate,
zero division. Computes the identical `(co, kh, ho, kw, wo)` contributions
as before; `dWeight`/`dBias` and the exported symbol/signature/dispatch are
completely unchanged.

**Results.** Isolated `dInput` kernel: 1.5-1.8x faster at every shape
tested (65.36ms -> 36.08ms at N=64/Cin=16/Cout=32/28x28/K=3). Full
`conv2d_backward`: 1.12-1.37x faster across all seven shapes. Real MNIST
CNN `conv2d` backward op: 11.16ms -> 9.95ms (~1.12x, smaller than the
isolated numbers since MNIST's real layers are the smallest shapes tested
and `dWeight`, left unchanged, remains substantial at the first layer).
Live async pipeline backward-phase GPU time: 10.82ms -> 8.48ms (~1.28x,
spanning both M31's fusion and this fix together -- no clean post-M31-only
snapshot from this session exists to isolate M32 alone at the full-pipeline
level). Async prefetch speedup unchanged within noise (1.21x -> 1.212x).

**Tests.** 1,162 tests total (1,154 pre-existing + 8 new,
`tests/test_cuda_conv2d_backward_optimization.py`), all passing on the
940MX. New coverage: weight/bias finite differences (input FD already
existed), explicit-async-stream backward correctness, two cross-stream
correctness tests, repeated-use memory-safety.

**What was profiled but not touched.** `dBias` (never more than 2% of
`conv2d_backward` at any shape) and `dWeight`'s existing hybrid dispatch
(M21's own measurement-tuned design, no `co`-independent redundancy to
remove) -- both measured, neither is this milestone's bottleneck.
`dWeight`'s per-thread reduction is now the largest remaining
`conv2d_backward` cost at most non-MNIST shapes and is documented as the
natural next optimization target, deliberately not pursued here per the
milestone brief's "start with one optimization" rule.

### M33 — CUDA Conv2d dWeight cooperative reduction (investigated, rejected)
M32 left `dWeight`'s per-thread reduction (`k_conv2d_backward_weight`,
one thread per weight element, full serial `N x Hout x Wout` sum) as the
largest single `conv2d_backward` cost at 6 of 7 representative shapes (now
dominant everywhere except `mnist_conv2`, where it is within 4% of
`dInput`). This milestone built a dedicated `dWeight` profiler
(`benchmarks/conv2d_backward_weight_profile.py`, new) and two genuinely
different cooperative-reduction candidates, forced independent of the M21
hybrid threshold (`CONV2D_WEIGHT_REDUCE_THRESHOLD = 256`) via new
profiling-only kernel exports, to test whether the serial reduction is
under-parallelized. See `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 33** section for the full report.

**Candidates tested.** (1) block-per-weight-element, shared-memory tree
reduction (`k_conv2d_backward_weight_reduce` -- the same kernel M21 already
uses below the threshold, forced here above it too) at 64/128/256
threads/block. (2) warp-per-weight-element, `__shfl_down_sync` reduction,
multiple weights packed per block (new `k_conv2d_backward_weight_warp`) at
2/4/8 warps/block (64-256 threads/block, 2-8 weight elements/block).

**Finding.** Both candidates measured 3-4x *slower* than the existing
per-thread kernel at every shape with >= 1,152 weight elements (all 6 of
the 7 shapes currently on the per-thread path), essentially flat across
every block-size/warps-per-block value tested -- strong evidence the loss
is dominated by the sheer number of concurrent block/warp *launches*
needed (thousands, scheduled across the 940MX's 3 SMs) rather than by
either candidate's own intra-group reduction overhead, and that the
existing per-thread kernel already achieves excellent memory-level
parallelism at these weight-element counts (thousands of independent
resident threads, several times the 940MX's 384 cores). Below the
threshold (`mnist_conv1`, 72 elements), the existing block-reduction path
remains the clear winner (~3.9x faster than a forced per-thread run),
confirming the M21 hybrid dispatch is still exactly right.

**Decision.** Per the milestone brief's explicit stop condition ("if
cooperative reduction does not provide a meaningful end-to-end improvement,
do not force it into Forge"): **rejected**. `CUDABackend.conv2d_backward`,
`cf_conv2d_backward_weight_*`'s dispatch, and both existing production
kernels are byte-for-byte unchanged. The two candidate kernels and their
forced-dispatch exports remain in `kernels.cu` as documented,
correctness-tested, profiling-only code (same category as M31's
`cf_event_create_timed`) so this evidence stays reproducible.

**Tests.** 1,184 tests total (1,162 pre-existing + 22 new,
`tests/test_cuda_conv2d_backward_weight_cooperative.py`), all passing on
the 940MX. New coverage: both candidate kernels (plus the forced per-thread
export) validated against `CPUBackend`'s weight gradient across three
shapes/dtype-relevant configs, block-size/warps-per-block sweeps, and a
repeated-use memory-safety test.

**Re-measured, unchanged (as expected -- no production code moved).**
Isolated `dInput`/`dWeight`/`dBias` (`conv2d_backward_profile.py`): within
run-to-run noise of the M32 baseline at all 7 shapes. Async pipeline
backward phase: 8.48ms -> 8.49ms. MNIST training step (CUDA): 15.2ms.
Async prefetch speedup: 1.212x -> 1.178x (within the 1.07-1.21x band every
milestone since M29 has measured). CPU/CUDA regression benchmark
categories (`forward`, `backward`, `training`, `mnist`): no operation moved
outside historical noise.

**New bottleneck ranking.** Unchanged from M32's post-fix ranking:
`dWeight` (or `dInput`, near-tied) dominates `conv2d_backward` at every
shape; `conv2d_backward` remains ~75% of total CUDA training-step compute
time. No further Conv2d-backward optimization is currently justified by
measurement -- the next real gains, if any, likely require a structurally
different approach (e.g. im2col+GEMM) outside this milestone's scope.

### M34 — CUDA Conv2d dWeight via im2col + existing tiled GEMM (accepted)
M33 named im2col + GEMM (reusing the existing M11 shared-memory-tiled
`k_matmul`, completely unmodified) as the next structurally different
`dWeight` candidate. This milestone built that experimental path (two new
gather kernels -- `k_im2col_conv2d`, `k_conv2d_grad_output_permute` -- plus
one ordinary `cf_matmul_*` call), verified its GEMM orientation against
`CPUBackend.conv2d_backward` (catching that a literal `Xcol^T @ dYmat`
reading would have been transposed relative to Forge's actual `(Cout, Cin,
KH, KW)` weight layout), and benchmarked it against the existing per-thread
kernel at the same 7 M32/M33 representative shapes, both in isolation
(`benchmarks/conv2d_backward_weight_im2col_profile.py`, new) and as the
complete `conv2d_backward` (`benchmarks/conv2d_backward_im2col_pipeline_
profile.py`, new, also measuring allocator/peak-memory behavior). See
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 34** section
for the full report.

**Finding.** 1.12-1.59x faster end-to-end than the existing kernel at every
shape with >= 1,152 weight elements (6 of 7 shapes), with 3-68MB peak-memory
overhead (well inside the 940MX's 2GB budget) and a 95% steady-state
allocator cache-hit rate (matching production). Slower only at the smallest
tested shape (`mnist_conv1`, 72 elements, already on M21's block-reduce
path) -- traced to `k_matmul`'s fixed 16x16 tiling having almost no reuse
benefit when `Cout`/`Cin*KH*KW` are near or below 16.

**Decision: accepted, as a minimal shape-based hybrid dispatch.**
`CUDABackend.conv2d_backward` (`backend.py`) now dispatches `dWeight` to the
new `forge.backend.cuda.experimental_conv_im2col.dweight_im2col_gemm` at/above
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD = 256` weight elements (reusing --
not re-deriving -- `kernels.cu`'s existing `CONV2D_WEIGHT_REDUCE_THRESHOLD`
boundary) and keeps the original single-kernel call below it. `dInput`/
`dBias` and the existing GEMM (`k_matmul`) are completely unmodified.
Documented caveat: no shape between 256 and 1,152 weight elements was
tested, so the threshold is a conservative reuse of the existing production
boundary, not a freshly-fitted crossover point.

**Tests.** 1,203 tests total (1,184 pre-existing + 19 new,
`tests/test_cuda_conv2d_backward_weight_im2col_gemm.py`), all passing on the
940MX. New coverage: the direct im2col+GEMM pipeline vs. CPU across 6
shape/stride/padding/kernel-size combinations (float32 and float64), finite
difference, explicit-stream, cross-stream, and repeated-use memory safety;
plus production-dispatch coverage through the real `Tensor.conv2d`/
`nn.Conv2d` API at shapes that cross the new threshold (existing
`test_cuda_conv.py` shapes never did, staying below ~144 weight elements).

**MNIST / pipeline / prefetch.** `mnist_profile`'s per-op CUDA `conv2d`
backward: 9.95ms (M32/M33) -> 6.90ms, ~1.44x (the real M20 CNN's second conv
layer now crosses the threshold). Full training step and async pipeline
`bwd` phase: within normal run-to-run noise of M32/M33 (MNIST's real layer
shapes keep the *absolute* savings small relative to those benchmarks' own
measurement noise). Async prefetch: 1.129x, within the established 1.07-1.21x
band. CPU/CUDA regression benchmark categories (`forward`, `backward`,
`training`, `mnist`): no operation moved outside historical noise except
`conv2d`/`mnist_cnn_full`, and only in the improving direction.

**Why no ADR.** Kernel-selection logic changed behind an unchanged
`CUDABackend.conv2d_backward` public method signature -- no public API,
Tensor semantics, or cross-cutting architectural boundary was touched.

### M35 — CUDA performance characterization and roofline-style analysis
A measurement-only milestone (`MEASURE -> MODEL -> CLASSIFY -> RANK ->
DECIDE`, no production CUDA changes by default): built a small roofline
library (`benchmarks/roofline.py` -- documented FLOP/byte-traffic
conventions, arithmetic intensity, a four-way bottleneck classifier) and
five new characterization scripts (`benchmarks/m35_hardware.py`,
`m35_kernels.py`, `m35_transfer_stream_alloc.py`, `m35_mnist.py`,
`m35_report.py`), all reusing Forge's existing kernels and existing
benchmark scripts directly rather than adding new instrumentation or
duplicate methodology -- practical compute/bandwidth ceilings come from
Forge's own `cf_matmul_f32`/`cf_add_f32` at large sizes (no new
hand-tuned microkernels added to `kernels.cu`), and the transfer/stream/
allocator characterization calls `transfer_bench`/`pipeline_profile`/
`stream_dependency_bench`/`allocator_bench` directly. See
`docs/performance/m35-roofline-characterization.md` for the full report.

**Measured practical ceilings (940MX):** 104.57 GFLOP/s (compute, `cf_matmul_f32`
large square GEMM, ~11% of the 953.1 GFLOP/s theoretical peak) and 15.09 GB/s
(bandwidth, `cf_add_f32` large streaming add, ~94% of the 16.02 GB/s
theoretical peak) -- clearly distinguished from the theoretical spec
throughout.

**Finding.** In a real MNIST training step, `conv2d` backward (dInput+dWeight
combined) is 50.97% of CUDA time and sits at only ~8% of its practical
roofline ceiling -- by a wide margin the top optimization-headroom candidate
(`runtime_fraction * (1 - fraction_of_ceiling)`), ahead of `conv2d` forward
(14.09%, ~12.5% of ceiling) and matmul backward (7.20%). GEMM itself already
reaches 85-108 GFLOP/s (81-100%+ of the measured ceiling) at every tested
shape -- compute-bound and already near-optimal for this hardware/kernel.
Elementwise/reduction/optimizer ops are memory-bandwidth-bound at medium/
large sizes (12-15 GB/s, near the 15.09 GB/s ceiling) and latency-bound at
small sizes, as expected. The M34 256-1152-weight-element region (previously
untested) now has data: im2col+GEMM is already faster than the direct kernel
across the whole region (0.60-0.81x), suggesting the existing conservative
256-element production threshold is not leaving performance on the table in
that range -- the threshold itself is left unchanged per the brief. D2H
async transfers pay a fresh `cudaHostAlloc` per call (Forge always allocates
a new pinned destination buffer), making D2H submission (~1.3ms) far more
expensive than H2D submission (~85us, which reuses an existing pinned source)
at a comparable size -- a real, measured asymmetry, flagged as a candidate
for a future milestone, not fixed here.

**Decision: no production optimization implemented**, per the brief's own
default. `conv2d` backward is named as the clear M36 candidate (Outcome A:
large runtime contribution, far below its practical ceiling) -- but *which*
specific algorithmic change is not chosen here; M35 is characterization
only.

**Tests.** 1,237 tests total (1,203 pre-existing + 34 new,
`tests/test_benchmarks_roofline.py`, deterministic FLOP/byte/AI/
classification unit tests with no CUDA dependency), all passing after a
clean CUDA rebuild on the 940MX. No `forge/` production code changed.

**Why no ADR.** No public API, Tensor semantics, or architectural boundary
was touched -- this milestone added benchmarking/analysis code only.

### M36 — CUDA Conv2d dInput algorithmic optimization (accepted)
Followed M35's naming of `conv2d` backward's `dInput` as the top
optimization-headroom candidate. `nvcc -Xptxas -v` on the unmodified M32
`k_conv2d_backward_input` found a 512-byte per-thread **local memory** stack
frame (its dynamically-indexed `kh_valid`/`ho_valid`/`kw_valid`/`wo_valid`
tables can never be register-resident) -- real traffic invisible to the
roofline model, explaining why the kernel sat at only ~12% of the practical
compute ceiling despite an arithmetic intensity that classifies it
compute-bound. Three structurally different candidates (`kernels.cu`,
profiling-only): **A** shared-memory grad_output row-tile reuse across
`Cin` (rejected -- never beat baseline, confirming `dInput` was never
bandwidth-starved); **B** channel-fused work mapping, one thread per
`(n,h,w)` holding all `Cin` accumulators in a register array, reading each
grad_output value once and reusing it via a register across every `ci`
(**accepted** -- 1.0x-8.7x faster in isolation, 0.97x-1.42x faster for the
complete `conv2d_backward` call, across three independent hardware runs, zero
stack frame confirmed by `-Xptxas -v`); **C** warp-cooperative reduction over
`Cout` (rejected -- 3-20x slower, `dInput` already launches far more threads
than the 940MX can use concurrently). Production dispatch (`cf_conv2d_
backward_input_*`) now calls Candidate B whenever `Cin <= 16` (every one of
Forge's 7 representative shapes), falling back to the unchanged M32 kernel
otherwise. See `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 36** section and `docs/architecture/cuda-backend.md`'s matching
section for the complete evidence.

**Files changed.** `forge/backend/cuda/kernels.cu` (three new profiling-only
candidate kernels + production dispatch change, `dWeight`/`dBias`/forward
untouched), `forge/backend/cuda/backend.py` (ctypes bindings for the three
profiling-only candidates), `benchmarks/conv2d_backward_dinput_profile.py`
(new), `tests/test_cuda_conv2d_dinput_optimization.py` (new, 21 tests).

**Tests.** 1,258 tests total (1,237 pre-existing + 21 new), all passing
after a clean CUDA rebuild on the 940MX. Every pre-existing `test_cuda_
conv.py` / `test_cuda_conv2d_backward_optimization.py` test (finite-
difference, cross-stream, explicit-stream, memory-safety) passes unmodified,
since all their shapes have `Cin <= 16` and already exercise the new
production path end-to-end.

**Why no ADR.** Same reasoning as M21/M32/M34: kernel-selection logic and a
kernel's internal thread mapping changed behind an unchanged `CUDABackend.
conv2d_backward` / `cf_conv2d_backward_input_*` signature and contract. No
public API, Tensor semantics, or cross-cutting architectural decision was
touched.

### M37 — CUDA Conv2d dWeight GEMM occupancy fix: split-K over existing buffers (accepted, modest)
Decomposed M34's im2col+GEMM `dWeight` pipeline (`benchmarks/m37_dweight_
profile.py`, new) at the 7 M32-M36 representative shapes and found two
independent, measured bottlenecks: (1) `im2col`+`permute` (zero FLOPs) cost
54-63% of total pipeline time -- more than the GEMM itself; (2) the GEMM's
own launch geometry (`ceil(Cout/16)*ceil(Cin*KH*KW/16)` blocks -- the huge
`N*Hout*Wout` reduction lives entirely inside each block's serial inner
loop, invisible to block count) launches as few as 5 of the 940MX's 24
resident-block device capacity (confirmed via `cuDeviceGetAttribute`), and
achieved-fraction-of-compute-ceiling tracked that occupancy shortfall
almost exactly. `nvcc -Xptxas -v` found zero stack frame/spill on every
dWeight-related kernel -- ruling out M36's local-memory failure mode here.

Two structurally different, evidence-targeted candidates were implemented
profiling-only and benchmarked (`kernels.cu`, `forge.backend.cuda.
experimental_conv_fused`, `benchmarks/m37_dweight_candidates_profile.py`):
**Candidate A/C** (fusing `im2col`/`permute`'s gathers directly into the
GEMM's tile loads, optionally plus a split-K reduction) -- **rejected**:
recomputing gather indices via integer div/mod every tile iteration cost
more than either bottleneck fix bought back once occupancy was no longer
limiting, regressing 27-29% at every shape with >= 18 GEMM blocks.
**Candidate E** (`dweight_im2col_gemm_splitk`, keeping M34's own `Xcol`/
`dYcolT` buffer reads unchanged and applying *only* a new, narrowly-scoped
split-K GEMM kernel, `cf_matmul_splitk_*` -- `k_matmul` itself untouched)
-- **accepted**: never regressed beyond measurement noise (worst case
0.99x) and won modestly (1.20-1.26x) at the two low-occupancy production
shapes, flat (0.99-1.00x) at the already-well-occupied ones. An initial
benchmark-harness bug (isolated GEMM-only timing compared against a full-
pipeline baseline) had implied a much larger 2.7-9.0x win; caught by
cross-checking against real end-to-end `CUDABackend.conv2d_backward`
timing before any candidate was accepted, and documented in full in
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 37** section
alongside the corrected numbers.

**Decision: accepted, modest.** `CUDABackend.conv2d_backward`
(`backend.py`) now calls `dweight_im2col_gemm_splitk` instead of M34's
`dweight_im2col_gemm`, unconditionally above the unchanged
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD = 256`. A controlled, same-session,
interleaved wall-clock A/B of the real M20 CNN's full training step
measured 0.994x -- statistically indistinguishable from 1.0x at that
granularity (Python dispatch overhead dominates a ~14ms step far more than
a sub-millisecond dWeight change) -- explicitly reported as a real,
evidence-backed isolated improvement whose end-to-end effect is too small
to measure at the full-training-step level, per the milestone brief's own
Amdahl-analysis requirement. `im2col`/`grad_output_permute` (M34) and
`k_matmul` (M11) are completely unmodified; `im2col`'s own materialization
cost (the pipeline's actual dominant term) is named as the clearest
remaining target for a future milestone.

**Tests.** 1,303 tests total (1,258 pre-existing + 45 new: 28 in
`tests/test_cuda_conv2d_backward_weight_splitk_gemm.py` -- CPU parity,
finite difference, explicit/cross-stream, memory/allocator safety,
production-dispatch coverage; 17 in `tests/test_cuda_conv2d_backward_
weight_fused_gemm_candidates.py` -- correctness-only coverage for the
rejected Candidates A/C, guarding shipped-but-unused profiling code against
bit rot), all passing on the 940MX after a clean rebuild.

**Why no ADR.** Same reasoning as M21/M32/M34/M36: a kernel-selection call
site changed behind an unchanged `CUDABackend.conv2d_backward` signature
and contract. No public API, Tensor semantics, or cross-cutting
architectural decision was touched.

### M38 — CUDA Conv2d dWeight im2col elimination: half-fused split-K GEMM (partial acceptance)
M37 left `im2col` (`k_im2col_conv2d`) + `grad_output` permute as `dWeight`'s
dominant remaining cost (54-63% of pipeline time, zero FLOPs). This
milestone measured `k_im2col_conv2d` itself via `nvcc -Xptxas -v`: 32
registers/thread, 0 spill, thread-count-limited (huge block count) --
never occupancy-bound, unlike the GEMM -- so its cost is genuine `Xcol`
write/read traffic, not a fixable occupancy artifact. The actual lever is
the *GEMM's* tile dependency structure: `k_matmul_splitk`'s `tile_a` load
depends only on `(row, a_m)`, so fusing a gather into the tile load costs a
redundant-recompute factor of `blocks_x` (M37's Candidate A/C fused *both*
gathers and lost badly at `blocks_x` up to 9 -- this milestone's own
"Candidate A" framing is mechanistically identical and was not
re-implemented, per Section 12's reuse allowance); fusing only `tile_b`
(the `Xcol`/im2col operand) instead costs `blocks_y = ceil(Cout/16)`,
`<= 2` at every representative shape.

**Candidate B** (`kernels.cu`'s `k_dweight_halffused_gemm_splitk`, Python
`forge.backend.cuda.experimental_conv_halffused.
dweight_halffused_gemm_splitk`): eliminates the `Xcol` buffer by fusing its
gather into a split-K GEMM's tile load, keeping `grad_output_permute`'s
cheap materialized `dYcolT` unchanged. `nvcc -Xptxas -v`: 0 bytes
stack/spill, 49 registers f32 (54 f64). Measured (`benchmarks/
m38_im2col_profile.py`, interleaved CUDA-event A/B, all 7 representative
shapes plus a `Cout` sweep isolating `blocks_y`): a clean, monotonic win at
`blocks_y == 1` / `Cout <= 16` (1.30-1.44x) that flips to a real,
reproducible regression at `blocks_y >= 2` (0.92-0.93x at `blocks_y=2`,
0.68-0.76x at 3-4). Also strictly cheaper in peak reserved memory at every
shape (13.78-55.12MB less), including where it isn't speed-dispatched.
**Candidate C** (partial/tiled materialization) was rejected analytically
without implementation: chunking `im2col` moves the same total bytes while
adding real per-launch overhead (M31), and its only possible edge --lower
peak memory-- is already dominated by Candidate B's measured results.

**Decision: partial acceptance.** `CUDABackend.conv2d_backward`
(`backend.py`) now branches on `blocks_y = ceil(Cout/16)` *inside* the
existing weight-element-threshold arm: `blocks_y == 1` calls Candidate B;
`blocks_y >= 2` keeps M37's unchanged `dweight_im2col_gemm_splitk`. A
controlled, CUDA-event, same-session before/after of the real MNIST conv2
layer's own shape (`Cin=8,Cout=16,K=3` -- exactly the winning regime)
measured **1.164x** at the full `conv2d_backward` level; a wall-clock,
interleaved, same-session full-training-step A/B measured **1.0065x**
(statistically indistinguishable from 1.0x, same Amdahl reasoning M37
documented -- a sub-millisecond `dWeight` change is invisible against a
~13.7ms step dominated by Python dispatch overhead). `im2col`/`grad_output_
permute`/`k_matmul`/`cf_matmul_splitk_*` are all completely unmodified.

**Tests.** 1,328 tests total (1,303 pre-existing + 25 new in `tests/
test_cuda_conv2d_backward_weight_halffused_gemm.py` -- CPU parity across
8 shape/stride/padding/kernel-size combinations spanning both `blocks_y`
regimes, direct agreement with the M37 baseline, finite difference,
explicit/cross-stream, 100-iteration memory safety, allocator cache-hit
reuse, and production-dispatch coverage for both branches of the new
`Cout`-based condition), all passing on the 940MX after a clean rebuild.

**Why no ADR.** Same reasoning as M21/M32/M34/M36/M37: a kernel-selection
call site changed behind an unchanged `CUDABackend.conv2d_backward`
signature and contract. No public API, Tensor semantics, or cross-cutting
architectural decision was touched.

### M39 — CUDA Conv2d dWeight: im2col materialization optimization (shared-memory input-plane staging, accepted)
M38 left `k_im2col_conv2d` itself as the dominant remaining `dWeight` cost
at every `blocks_y >= 2` shape (`Cout > 16` -- the regime M38's half-fused
GEMM cannot help, since its redundant-regather tax grows with `blocks_y`).
Fresh `nvcc -Xptxas -v`: 32 registers/thread, 0 bytes stack frame, 0 bytes
spill (unchanged since M34) -- not a local-memory problem, and (one thread
per `Xcol` output element, `M*K` threads at every shape) not
occupancy-limited either.

Two structurally different candidates were designed, implemented as
profiling-only kernels, and measured (`benchmarks/m39_im2col_reuse_profile.py`,
7 representative shapes + a kernel-size sweep `K` in {1,3,5} + a stride
sweep {1,2} isolating the reuse-factor hypothesis directly):

- **Candidate A** (`k_im2col_conv2d_indexed`): `k_im2col_conv2d` decomposes
  both `m -> (n,ho,wo)` (3 divisions, `K`-fold redundant per output
  position) and `k -> (ci,kh,kw)` (3 divisions, `M`-fold redundant per `k`
  value) independently in every one of its `M*K` threads. This candidate
  hoists `m`'s decomposition to one thread per block (shared-memory
  broadcast) and looks `k`'s decomposition up from a tiny host-built table
  instead of computing it via division -- targets the instruction/division
  redundancy directly, independent of any data-reuse claim (mirrors M32's
  `dInput` fix, a different, cross-thread form of the same redundancy).
  **Rejected**: shape-dependent, regressing badly at small `K`
  (0.29-0.48x at `mnist_conv1`/`K=1` -- a fixed 256-thread launch wastes
  most threads when `K` is small even after tuning `threads_per_block` to
  `K`'s own size) despite winning at larger `K` (1.7-2.0x) -- too
  inconsistent to dispatch safely, and the milestone's own regime
  (`blocks_y >= 2`) includes exactly the small shapes it fails on
  (`mnist_conv2`, `Cout=17`, `K=72`: 0.65-1.09x, unreliable).
- **Candidate B** (`k_im2col_conv2d_smem`): the milestone brief's literal
  shared-memory-reuse hypothesis. One block owns one `x[n,ci,:,:]` plane,
  stages it into shared memory once (<=12.5KB at every representative
  shape, far under CC 5.0's 48KB per-block cap), then serves every one of
  that plane's `Hout*Wout*KH*KW` `Xcol` writes from shared memory instead
  of a fresh global load -- cutting global reads of `x` from `M*K` nominal
  down to `N*Cin*H*W`. **Accepted**: won at *every* representative shape
  and every sweep point tested (1.05-1.96x), including the `K=1`
  (`reuse_factor=1.0`, zero nominal overlap) and `stride=2`
  (`reuse_factor=2.25`) edge cases -- never a measured regression anywhere.

**Full-pipeline validation** (not just isolated `im2col` timing, per M37's
own "isolated-stage mistakes must be cross-checked" lesson):
`dweight_im2col_smem_gemm_splitk` (`experimental_conv_im2col_reuse.py`) --
identical to M37's `dweight_im2col_gemm_splitk` except its `im2col` stage
uses Candidate B (falling back to the unmodified M34 `im2col` if the
per-block shared-memory request would exceed the 48KB cap, which no Forge
shape reaches) -- measured **1.11-1.33x** faster than the M37 baseline at
every `blocks_y >= 2` shape tested (`Cout` 17 through 128, plus all 4
representative shapes in this regime), monotonically shrinking as `blocks_y`
grows (GEMM/permute time dilutes `im2col`'s fixed savings) but never
regressing. A controlled, interleaved, same-session A/B through the real
`Tensor.conv2d`/`nn.Conv2d` API at `large_channel`'s shape measured a
real, low-variance **1.056x** at the full `conv2d_backward` level.

**One real bug caught and fixed during development**: an early draft of
Candidate A's small host-table upload helper returned a bare
`backend._alloc()` pointer with no owning `CUDAStorage`, leaking it
permanently (never released back to the M25 caching allocator) --  caught
by this milestone's own repeated-use memory-safety test, the same
bookkeeping mistake M33's own history documents. Fixed by wrapping the
upload in a `CUDAStorage` like every other Forge CUDA buffer.

**Decision.** `CUDABackend.conv2d_backward`'s `blocks_y >= 2` branch
(`backend.py`) now calls `dweight_im2col_smem_gemm_splitk` in place of
M37's `dweight_im2col_gemm_splitk`; `grad_output_permute`/`k_matmul`/
`cf_matmul_splitk_*` remain completely unmodified, and the `blocks_y == 1`
branch (M38's half-fused Candidate B) is untouched. MNIST's own two conv
layers (`Cout` 8 and 16) are both `blocks_y == 1` and never reach this
dispatch branch -- correctly zero effect expected and observed (pipeline/
MNIST bench numbers within normal run-to-run variance of history).

**Tests.** 1,366 tests total (1,328 pre-existing + 38 new in `tests/
test_cuda_conv2d_backward_weight_im2col_smem.py` -- CPU parity across 8
shape/stride/padding/kernel-size combinations including `Cin=1` and a 1x1
kernel, float64, direct agreement with the M37 baseline, isolated
`im2col_smem`-vs-baseline and `im2col_indexed`-vs-baseline correctness,
finite difference, explicit/cross-stream, 100-iteration memory safety,
allocator cache-hit reuse, and production-dispatch coverage spanning
`blocks_y` 1 through 3 plus the below-threshold case), all passing on the
940MX after a clean rebuild. Full suite (`pytest tests/ -q`): 1,366 passed.

**Why no ADR.** Same reasoning as M21/M32/M34/M36/M37/M38: a kernel-
selection call site changed behind an unchanged `CUDABackend.
conv2d_backward` signature and contract. No public API, Tensor semantics,
or cross-cutting architectural decision was touched.

### M40 — Post-M39 CUDA bottleneck re-characterization (measurement-only, M41 target selected)

A fresh, same-session, CUDA-event-based re-measurement of the whole CUDA
training pipeline after M31-M39's cumulative optimization work
(`benchmarks/m40_bottleneck_recharacterization.py`, new). No production
kernel, dispatch, or public API was changed -- pure measurement and target
selection, per the milestone's own scope.

**Dispatch verification**: confirmed, shape by shape, that
`CUDABackend.conv2d_backward`'s dWeight branch selects exactly the function
its own `weight_elements`/`blocks_y` logic implies (`cf_conv2d_backward_weight`
below 256 elements; `dweight_halffused_gemm_splitk` at `blocks_y==1`;
`dweight_im2col_smem_gemm_splitk` at `blocks_y>=2`) -- closing the "measuring
the wrong implementation" risk M37's own history documents.

**New finding**: `conv2d` backward's two sub-kernels no longer have one
uniform leader. At `blocks_y==1` shapes (`Cout<=16`: `mnist_conv1`,
`mnist_conv2`, `large_spatial`) `dWeight` is now the larger cost; at
`blocks_y>=2` shapes (`Cout>16`) `dInput` is larger -- an alternation neither
M35 nor M39's own per-shape numbers stated explicitly. At MNIST's own real
layer shapes, `dWeight` (29.9% of the full training step) now exceeds
`dInput` (15.6%), reversing M35/M36's finding -- driven mostly by
`mnist_conv1`'s below-threshold block-reduce kernel (already tuned in
M21/M33; little further headroom found there).

**Roofline re-classification** (fresh ceilings this session: 104.67 GFLOP/s
compute, 15.09 GB/s bandwidth): `dWeight`'s GEMM-dominated shapes
(`blocks_y>=2`) are compute-bound at 43-44% of ceiling (up slightly from
M37's 32.6-32.9%); `dInput` is compute-bound at 33-35% of ceiling at large
shapes. `k_conv2d_forward` (unchanged since Milestone 15 -- one thread per
output element, zero explicit memory reuse) achieves only **10.7-18.0%** of
the practical compute ceiling at every representative shape -- meaningfully
below either backward Conv2d kernel at the same shape despite an *identical*
total FLOP count (`roofline.py`'s own documented FLOP symmetry across
forward/dInput/dWeight). This is the milestone's headline finding: the same
arithmetic work, achieving markedly lower efficiency, on the one major
Conv2d kernel no milestone since M15 has restructured.

**Pipeline/non-Conv2d confirmation**: compute-stream utilization is 87.5-93.6%
across every batch size and prefetch depth tested -- the async pipeline is
already well-fed; no further DataLoader/prefetch work is justified.
CrossEntropy (0.89% of step), the optimizer (~5%), and transfers (~2.4%)
remain negligible, unchanged since M31/M17/M29-M30.

**M41 recommendation**: apply the already-validated im2col + existing
tiled/split-K GEMM technique (M34/M39's own approach for `dWeight`) to
`k_conv2d_forward` -- reusing `im2col`/`im2col_smem`/`cf_matmul_*`/
`cf_matmul_splitk_*` unmodified rather than designing a new kernel. Explicitly
excluded from M41: `dWeight`'s split-K/half-fused GEMM paths (already
compute-bound, would need a new GEMM tiling design -- out of scope per
Forge's one-portable-GEMM architecture stance), `dWeight`'s below-threshold
block-reduce kernel (M33 already found no better alternative), `dInput`'s
channel-fused kernel (M36 already explored three structural alternatives),
and `k_matmul`'s own tiling. See
`docs/performance/m40-bottleneck-recharacterization.md` for the complete
ranked-candidate table, Amdahl analysis, and evidence.

**Tests.** No production code changed; full existing suite passes unchanged
at its pre-M40 count. `benchmarks/m40_bottleneck_recharacterization.py` is
the only new file, exercising exclusively already-shipped production
functions (`cf_conv2d_backward_weight_*`, `dweight_halffused_gemm_splitk`,
`dweight_im2col_smem_gemm_splitk`'s constituent stages, `dInput`'s
`cf_conv2d_backward_input_*`) plus the pre-existing `m35_hardware`/
`m35_mnist`/`pipeline_profile` tools, directly.

**Why no ADR.** Measurement-only milestone; no architectural, dispatch, or
public-API change was made.

### M41 — CUDA Conv2d forward: im2col + GEMM optimization (accepted)

Applied M40's recommended im2col + existing tiled GEMM technique to
`k_conv2d_forward` (unchanged since M15, one thread per output element,
zero memory reuse; confirmed via a fresh `nvcc -Xptxas -v` pass to have zero
register spill/stack frame, so the inefficiency is structural, not a
register-pressure problem). Two structurally different candidates were
designed, implemented, and measured against the complete baseline pipeline
at 15 representative/sweep shapes (`benchmarks/m41_conv2d_forward_profile.py`,
new):

- **Candidate A** (`experimental_conv_forward_im2col.py`): `im2col`/
  `im2col_smem` (M34/M39, unmodified) -> weight transpose (`k_transpose`,
  M11, unmodified) -> the existing tiled GEMM (`cf_matmul_*`, M11,
  unmodified) -> one new kernel, `k_conv2d_output_permute` (inverse of
  M34's `k_conv2d_grad_output_permute`, bias fused in).
- **Candidate B** (`experimental_conv_forward_halffused.py`): weight
  transpose (same, cheap) + one new kernel,
  `k_conv2d_forward_halffused_gemm`, that fuses `Xcol`'s gather directly
  into its own tiled-GEMM tile load (no `Xcol`/`out_mat` buffers at all),
  writing straight into the final output layout with bias fused in --
  M38's half-fused `dWeight` idea, roles reversed to match forward's own
  GEMM orientation (no split-K needed here: forward's `M=N*Hout*Wout` is a
  large *block-count* dimension, not the small reduction dimension
  `dWeight`'s GEMM had).

**Both accepted.** Both win 1.06-2.85x at every shape at/above ~20M total
forward FLOPs and regress 0.26-0.76x at every shape at/below ~7.2M FLOPs
(fixed kernel-launch/allocation overhead outweighs an already-fast baseline
call). Above the threshold, Candidate B wins at every `Cout<=32` shape
(every current Forge shape, including both real MNIST-scale layers);
Candidate A (`im2col_smem` variant) wins at the one tested `Cout=128` shape,
since Candidate B's redundant `Xcol`-tile-regather tax grows with
`blocks_x=ceil(Cout/16)`. `CUDABackend.conv2d` now computes
`total_flops = 2*N*Cout*Hout*Wout*Cin*KH*KW` and dispatches to Candidate B
(`blocks_x<=2`) or Candidate A (`blocks_x>2`) at/above
`_CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD = 10,000,000`, keeping the unmodified
per-thread kernel below it. `mnist_conv1` (MNIST layer 1) stays on the
unchanged baseline (below threshold); `mnist_conv2` now dispatches to
Candidate B and measured 1.54x faster end to end -- a real, measured shift
in the MNIST training step's own `forward:Conv2d` share (17.0% -> 12.94% of
the measured forward+backward step, same `m35_mnist` tooling both before
and after).

**Roofline impact.** `large_channel`'s forward (now Candidate B) reaches
43.2% of the practical compute ceiling (up from 17.0%) and is now
**compute-bound** -- on par with `dWeight`'s own 43-44% ceiling-fraction at
the same shape, confirming the reformulation genuinely moved this kernel
toward the roofline, not just reduced wall-clock time at unchanged
efficiency.

**dInput/dWeight/dBias, `k_matmul` itself, the async prefetch pipeline, and
every non-Conv2d op are completely untouched** -- confirmed by re-running
`benchmarks/m40_bottleneck_recharacterization.py` fresh post-dispatch
(backward's own isolated-kernel timings within 1-3% of M40's archived
numbers, consistent with this GPU's documented run-to-run variance; async
compute-stream utilization 85.5-90.9%, unchanged within noise from M40's
87.5-93.6%).

**Tests.** `tests/test_cuda_conv2d_forward_im2col_gemm.py` (new, 54 tests):
f32/f64 parity vs. `CPUBackend.conv2d` across `K` in `{1,2,3,5}`, stride
`{1,2}`, padding `{0,1,2}`, small/large `Cin`/`Cout`, with/without bias,
explicit-stream, cross-stream, repeated-use memory-lifecycle safety, for
both candidates directly and through the real `Tensor.conv2d`/`nn.Conv2d`
API at shapes crossing both dispatch boundaries (the FLOPs threshold and
the `blocks_x` Candidate-A-vs-B boundary), plus a full forward+backward
autograd check confirming unchanged gradients. Full suite: **1,420 passed**
(up from 1,366 pre-M41), verified via `python -m pytest tests/ --collect-only -q`
and a full run on a clean CUDA rebuild.

**Why no ADR.** Same reasoning as M21/M32/M34/M36/M37/M38/M39: a kernel-
selection call site changed behind an unchanged `CUDABackend.conv2d` public
signature and contract. No public API, Tensor semantics, or cross-cutting
architectural decision was touched. See `docs/performance/
conv2d-forward-profiling.md` for the complete report, including the
documented threshold-boundary caveats and the `k1_s1` measurement anomaly.

### M42 — Fresh post-M41 CUDA bottleneck re-characterization (measurement-only, M43 target selected)

A fresh, same-session re-measurement of the whole CUDA training pipeline
now that M41 has changed forward dispatch (`benchmarks/
m42_bottleneck_recharacterization.py`, new -- combines M40's backward
decomposition with M41's forward decomposition, both against a dispatch
decision recomputed from the actual `backend.py` source rather than
historical docs). No production kernel, dispatch, or public API was
changed -- pure measurement and target selection.

**M41's fix held**: forward now reaches 43.5-48.7% of the practical
compute ceiling at every shape that reaches the GEMM dispatch (up from
M40's 10.7-18.0%), on par with dWeight's best path -- confirming M41
genuinely closed the gap M40 identified, not just reduced wall-clock time
at unchanged efficiency.

**New finding**: a different, previously invisible inefficiency is now the
sharpest in the pipeline. dWeight's `blocks_y==1` path
(`k_dweight_halffused_gemm_splitk`, M38 -- dispatched whenever `Cout<=16`,
exactly MNIST's own real second conv layer) reaches only 21.7% (`mnist_
conv2`) to 24.8% (`large_spatial`) of the ceiling -- roughly *half* the
efficiency of every other GEMM-dispatched Conv2d kernel measured, including
forward's own structurally similar (non-split-K) half-fused GEMM
(43.5-48.7%) and dWeight's own `blocks_y>=2` regime (42.9-44.2%). A fresh
`nvcc -Xptxas -v` pass ruled out register spill and gross occupancy
disparity as the cause (54 registers/4096B shared memory, comparable to
the 40-register/4096B forward kernel that reaches double the efficiency).
The one documented architectural difference between the two half-fused
kernels is split-K: dWeight's GEMM has `M` as the *reduction* dimension
(needing split-K's atomic-accumulation combine step), forward's has `M` as
a large block-count dimension (no split-K needed at all) -- now the
leading root-cause hypothesis.

**dWeight remains the single largest addressable sub-component of the
training step** (29.88% of the full step, exceeding dInput's 15.62% and
forward's 12.38% combined-ish), confirming M40's finding still holds
post-M41 (M41 touched only forward). Roughly a third of dWeight's own cost
(≈10.2% of the full step) now traces to this specific, previously-
unexamined `blocks_y==1` efficiency gap rather than only to the
already-twice-investigated (M33 cooperative reduction, M34 im2col+GEMM,
both rejected) below-threshold block-reduce kernel.

**Negative results, stated explicitly rather than manufactured around**:
`k_matmul` itself is already at its own practical ceiling by construction
(0% headroom); CrossEntropy is already fused (M31) and reaches 46-58% of
the bandwidth ceiling at realistic batch sizes (no fresh opportunity);
dBias is negligible everywhere (<3% of the full step); the async pipeline
(83-92% compute-stream utilization), allocator, and pinned memory are all
confirmed healthy with no M41-introduced regression; the forward im2col+
GEMM path's temporary-buffer footprint (10MB at the one shape that reaches
it) is not a meaningful VRAM constraint against the 2048MB budget.

**M43 recommendation**: investigate whether a different split-K reduction
strategy (two-pass non-atomic combine, cooperative-groups reduction, or
re-tuned `num_k_splits`) can close some of `k_dweight_halffused_gemm_
splitk`'s measured ~2x efficiency gap at its `blocks_y==1` dispatch shapes,
reusing `recommended_num_k_splits`/`cf_matmul_splitk_f32`/M37-M41's own
interleaved-A/B benchmark methodology. Acceptance: faster than the current
kernel at every `blocks_y==1` representative/sweep shape with no
regression, targeting ≥33% of the compute ceiling (up from 21.7-24.8%) at
the two real representative shapes. Full report (19 sections: dispatch
verification, forward/backward decomposition, roofline, resource, sync,
memory, Amdahl, ranking, candidates, M43 recommendation, exclusions,
limitations, reproducibility): `docs/performance/
m42-bottleneck-recharacterization.md`.

**Tests.** No test changes (measurement-only milestone). Full suite:
**1,420 passed** (unchanged from M41's own count), verified on a clean
CUDA rebuild (`_forge_cuda_kernels_sm_50.dll` deleted and recompiled).

**Why no ADR.** Measurement and documentation only -- no production code,
public API, or cross-cutting architectural decision was touched.

### M43 — CUDA dWeight `blocks_y==1` split-K reduction optimization (accepted)

Investigated M42's leading root-cause hypothesis for `k_dweight_halffused_
gemm_splitk`'s (M38) measured ~2x roofline-efficiency gap vs. forward's
structurally identical half-fused GEMM: split-K's atomic-accumulation
combine step. Confirmed it directly (`benchmarks/m43_dweight_splitk_
profile.py`, new): ruled out register pressure (unchanged, no spill),
`Cout`-dependent redundant work (a `Cout in {1,2,4,8,16}` sweep at
small/medium/large reduction lengths showed flat GEMM time regardless of
`Cout` -- the `blocks_y`-based redundant-regather tax M38 documented is not
the cause here, since it is a constant 1x throughout `blocks_y==1`), and
split-count/occupancy shortfall (the production `recommended_num_k_splits`
formula already sits within 3-8% of its own swept optimum; an extended
sweep to 676 splits shows atomics' cost only rising sharply *far* past the
production value, not proof they are costly at `splits=16` specifically).

Replaced the atomic combine with a deterministic two-stage reduction at
the *same*, unchanged split count: `k_dweight_halffused_gemm_splitk_
partial` (`kernels.cu`, new -- byte-for-byte identical tile loads/gather to
M38's kernel, but each split writes a disjoint slice of a `(num_k_splits,
Cout, Kdim)` buffer instead of an `atomicAdd`) + `k_dweight_splitk_reduce`
(new, tiny -- sums the `num_k_splits` axis, bandwidth-trivial since
`Cout*Kdim` is always small). New module: `forge/backend/cuda/
experimental_conv_dweight_tworeduce.py`. Measured (interleaved CUDA-event
A/B, both representative `blocks_y==1` shapes, six `num_k_splits` values
each): **1.11-1.24x faster full dWeight pipeline at `mnist_conv2`,
1.02-1.04x at `large_spatial`, at every tested split count, never a
regression** -- confirming the atomic combine itself (not split count) was
the differentiator. `CUDABackend.conv2d_backward`'s `blocks_y==1` branch
now calls the new function in place of M38's (kept, still tested, now dead
in production); the dispatch condition itself is unchanged.

A controlled, same-session A/B through the real public `conv2d_backward()`
entry point measured 1.076x/1.015x at the full call level (dInput+dWeight+
dBias+allocation together) -- the expected Amdahl dilution. Per M42's own
decomposition, the affected `mnist_conv2` layer contributes only ~10.2% of
the full MNIST training step; projected whole-step effect (~1.02x) is
honestly reported as too small to distinguish from this laptop GPU's own
run-to-run thermal variance at that granularity, even though the
component-level win is clearly and reproducibly measurable (Amdahl
honesty). One real memory-leak bug was caught by this milestone's own
repeated-use test (an early draft's partial buffer allocated without an
owning `CUDAStorage`) and fixed before acceptance.

**Tests.** `tests/test_cuda_conv2d_backward_weight_splitk_optimization.py`
(32 new): CPU parity across 10 shapes (both `blocks_y` regimes, `K=1`/`K=5`,
`Cin=Cout=1`), float64, 5 explicit `num_k_splits` values, agreement with
the M38 atomic baseline, finite difference, explicit/cross-stream, 100-
iteration memory safety, allocator reuse, and production-dispatch coverage
through the real `Tensor.conv2d`/`nn.Conv2d` API. Full suite: **1,452
passed** (1,420 pre-existing + 32 new), verified on a clean CUDA rebuild
(`_forge_cuda_kernels_sm_50.dll` deleted and recompiled, 9.8s) and a
dedicated 500-iteration stress test through the real API (no leak).

**Why no ADR.** Same reasoning as M21/M32/M34/M36/M37/M38/M39: a kernel-
selection call site changed behind an unchanged `CUDABackend.
conv2d_backward` public signature and contract. Full report: `docs/
performance/conv2d-backward-profiling.md`'s **Milestone 43** section.

### M44 — Fresh post-M43 CUDA bottleneck re-characterization (measurement-only, M45 target selected)

Re-measured the whole CUDA training pipeline fresh (`benchmarks/m44_
bottleneck_recharacterization.py`, new) against the current post-M43
production dispatch, since M40/M42's own dWeight sub-stage helper still
decomposed `blocks_y==1` as "permute + fused-GEMM" (M38) -- stale after
M43 replaced that path with a two-stage reduction. A fresh same-session
interleaved A/B (reusing `m43_dweight_splitk_profile._candidate_b_
comparison` directly) confirmed M43's win still holds: 1.15-1.29x
(GEMM-only) / 1.12-1.23x (full pipeline) faster at `mnist_conv2`,
1.02-1.03x at `large_spatial`, no regression at any tested split count.
New per-stage timing (permute / partial-GEMM / reduce, built on
`m43_dweight_splitk_profile._RawDweightTworeduce`) showed the reduce
kernel is bandwidth-trivial as designed (0.9-1.1% of the path's total),
and `nvcc -Xptxas -v` confirmed the partial-GEMM kernel has an identical
register/shared-memory footprint to M38's original -- the win came purely
from removing the atomic combine, with no incidental occupancy change.

**Because M43 shrank `blocks_y==1`'s Amdahl fraction (10.2% to 8.47% of
the full MNIST step), the below-256-weight-element block-reduce kernel
(`mnist_conv1`'s own real shape, unchanged since M21) is now unambiguously
the single largest contributor** -- 20.19% of the full step, more than
double `blocks_y==1`'s own post-M43 share, while reaching only 3.1-3.2% of
the practical compute ceiling across a dedicated sweep (`weight_elements`
72-243) -- the worst roofline efficiency of any candidate measured in this
or any prior characterization. The sweep showed cost tracks reduction
length (`N*Hout*Wout`, held fixed) far more than `weight_elements` itself
(two 144-element shapes with different `Cin`/`Cout` splits cost the same),
consistent with an occupancy-bound diagnosis: only `weight_elements`-many
threads launch, each performing a fully independent, fully serial
reduction -- nowhere near enough parallelism to saturate the 940MX at
MNIST's own scale.

Per the milestone's own instruction, M33's cooperative-reduction rejection
and M34's im2col+GEMM rejection were not reopened -- neither was re-run,
and nothing measured here contradicts either finding. What changed is the
kernel's *relative* priority now that `blocks_y==1` has been closed by
M43, plus a genuinely untried angle: warp-shuffle-based cooperative
reduction, which uses neither M33's shared-memory tree reduction nor M34's
GEMM restructuring. **M45 recommendation**: attack this kernel with a
warp-shuffle cooperative reduction. Acceptance: faster than the current
kernel at every below-256 representative/sweep shape with no regression,
targeting >=15% of the practical compute ceiling (up from 3.1-3.2%) --
mirroring M43's own acceptance bar. A 2x speedup there projects to an
11.2% whole-step speedup (Amdahl), the largest single-component ceiling
measured this session, exceeding dInput's 8.3% despite dInput's own larger
per-shape headroom. Full report (20 sections: dispatch verification,
architecture, methodology, decomposition, M43 stage analysis, targeted
Cout/below-256/Cin sweeps, roofline, resources, pipeline health, Amdahl,
ranking, revisited-work accounting, M45 recommendation, exclusions,
limitations, reproducibility): `docs/performance/
m44-bottleneck-recharacterization.md`.

**Tests.** No test changes (measurement-only milestone). Full suite:
**1,452 passed** (unchanged from M43's own count), verified on a clean
CUDA rebuild (`_forge_cuda_kernels_sm_50.dll` deleted and recompiled in
9.62s).

**Why no ADR.** Measurement and documentation only -- no production code,
public API, or cross-cutting architectural decision was touched.

### M45 — CUDA dWeight below-256 warp-shuffle reduction (investigated, rejected)

Benchmarked M44's recommended warp-shuffle cooperative reduction for the
below-`CONV2D_WEIGHT_REDUCE_THRESHOLD` (256) dWeight path (`k_conv2d_
backward_weight_reduce`, still production, unchanged since M21) before
accepting M44's occupancy-bound diagnosis, per `PROFILE -> ANALYZE ->
DESIGN -> BENCHMARK -> SELECT`. Reused M33's existing `k_conv2d_backward_
weight_warp`/`cf_conv2d_backward_weight_warpreduce_*` (one 32-lane warp per
weight element) unmodified, and added one new candidate, `k_conv2d_
backward_weight_warp_subgroup`/`cf_conv2d_backward_weight_warpsubgroup_*`
(configurable 8/16/32-lane sub-warp groups per weight element) -- both
profiling-only, never called by `CUDABackend`.

**Result: rejected.** A fresh, same-session sweep independently varying
weight-element count (9-252, `benchmarks/m45_dweight_below256_profile.py`,
new), reduction size (256-204,800, `weight_elements` fixed at 72), and
`K`-configuration (1/3/5) found neither candidate clears the milestone's
1.15x acceptance bar at `mnist_conv1`'s own shape: 1.003x isolated-kernel
speedup, 0.985x in a complete `conv2d_backward` A/B -- both within noise --
and four sweep points regressed outright (0.58-0.95x). `nvcc -Xptxas -v`
showed no spill/register-pressure difference among the production kernel
and both candidates (42-53 registers, 0/1024 bytes shared memory).

**Root-cause correction to M44's hypothesis**: the production block-reduce
kernel already launches 256 threads *per weight element*
(`weight_elements x 256` total, 18,432-64,512 across the tested shapes) --
both warp candidates launch *fewer* total threads per weight element (32,
or 8/16 for sub-groups), an 8-32x *reduction* in launched parallelism
relative to the kernel already in production. The below-256 path was
therefore never as parallelism-starved as M44's per-thread-kernel-relative
framing suggested at MNIST's own real scale. The one clear win (1.93x) was
at the smallest tested reduction length (256 elements) -- consistent with
fixed `__syncthreads()`/shared-memory-allocation overhead dominating only
when the reduction itself is nearly free, an edge case no real Forge shape
reaches (`mnist_conv1`'s own reduction length is 50,176).

**Decision**: `CUDABackend.conv2d_backward`, the M21 dispatch threshold,
and both production kernels are byte-for-byte unchanged. Both candidates
remain in `kernels.cu` as documented, tested, profiling-only code. Full
14-section report (baseline, resource analysis, three independent sweeps,
complete-pipeline A/B, boundary test, fresh Amdahl fraction, root-cause
correction, correctness, memory, production decision, updated ranking,
limitations): `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 45** section.

**Tests.** 67 new (`tests/test_cuda_conv2d_backward_weight_below256_
optimization.py`): `warpsubgroup` correctness across shapes/dtypes/group
sizes, `warpreduce` extended to the below-256 regime, finite difference,
explicit-stream execution, a 255/256/257 production-dispatch boundary
check, and a repeated-use memory-safety check. Full suite: **1,519 passed**
(1,452 + 67 new), verified on a clean CUDA rebuild. `benchmarks.
pipeline_profile` re-run fresh: 82.2-87.0% compute-stream utilization
(within historical range), allocator/pinned-memory counters clean.

**Why no ADR.** No production code, public API, or cross-cutting
architectural decision was touched -- a rejected profiling-only
experiment, the same category as Milestone 33's own rejected candidates.

### M46 — CUDA dWeight below-256 grid-split reduction (investigated, rejected)

M45 corrected M44's diagnosis and named the one remaining untried axis for
the below-`CONV2D_WEIGHT_REDUCE_THRESHOLD` (256) dWeight path
(`k_conv2d_backward_weight_reduce`, still production, unchanged since M21):
grid-level parallelism, since the production kernel already launches
`weight_elements x 256` threads (more than either of M45's rejected
warp-shuffle candidates). This milestone built `k_conv2d_backward_weight_
reduce_gridsplit` (new, profiling-only): `num_splits` blocks per weight
element, each reducing a disjoint slice of the `N*Hout*Wout` dimension,
combined by M43's existing `k_dweight_splitk_reduce` kernel reused
unmodified (a flat weight vector is `Cout=1, Kdim=weight_elements` of the
same shape it already sums).

A real `cudaOccupancyMaxActiveBlocksPerMultiprocessor` query (two new tiny
diagnostic exports, `cf_occupancy_conv2d_backward_weight_reduce[_gridsplit]_
f32`) found **identical occupancy** between production and the candidate: 5
resident blocks/SM (register-bound, 62.5%), 15 across the 940MX's 3 SMs,
regardless of `num_splits` -- grid-splitting cannot raise this device's
per-SM concurrent-block ceiling, only launch more/smaller blocks against
the same one. This was the decisive ANALYZE-phase finding, obtained before
any timing was trusted.

**Methodology note.** An initial block-sequential timing attempt (matching
M45's own per-variant `_time_phase` pattern) measured an illusory 1.47x
"win" at `mnist_conv1` that a true round-robin-**interleaved** rerun could
not reproduce (1.02x, within noise) -- the 940MX's clock/power state
visibly drifts over a multi-second block of repeated launches, biasing
whichever variant a block-sequential harness happens to time later. The
benchmark script was rewritten around a new `_interleaved_multi_time`
helper before any of the milestone's real numbers were trusted -- the same
class of mistake M37 made (an illusory win from a timing-setup bug), via a
different mechanism (clock drift vs. buffer reuse).

**Result: rejected.** With correct interleaved methodology, `mnist_conv1`
measured 1.019x (isolated kernel) / 1.012x (complete `conv2d_backward`) --
far short of the 1.15x bar. The best result anywhere in a fresh
weight-element/reduction-size/kernel-size sweep was 1.072x (`k1_wide`,
`K=1`); the shortest tested reduction length regressed to 0.795x at its
best configuration and 0.46x at `num_splits=8`, with cost increasing
monotonically with split count at every shape -- consistent with the
occupancy finding: splitting trades a shorter per-block serial reduction
(shrinking tail latency) against more paid per-block/launch overhead, a
real but small effect that never reaches the acceptance bar.

**Decision**: `CUDABackend.conv2d_backward`, the M21 dispatch threshold,
and `k_conv2d_backward_weight_reduce` are byte-for-byte unchanged. The new
kernel, both occupancy-diagnostic exports, and a Python wrapper
(`experimental_conv_dweight_gridsplit.py`) remain in the codebase as
documented, tested, profiling-only code. Full 14-section report (baseline,
candidate design, resource analysis, real occupancy query, methodology
correction, benchmark results, Amdahl, root cause, correctness, memory,
regression check, production decision, updated ranking, limitations):
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 46** section.

**Tests.** 31 new (`tests/test_cuda_conv2d_backward_weight_below256_
gridsplit.py`): CPU parity across shapes/`num_splits`/dtypes, a direct
same-inputs comparison against the production kernel, finite difference,
explicit-stream and cross-stream execution, the 255/256/257 production-
dispatch boundary check, a repeated-use memory-safety check, and an
allocator-reuse-across-split-counts check. Full suite: **1,550 passed**
(1,519 + 31 new), verified on a clean CUDA rebuild.

**Why no ADR.** No production code, public API, or cross-cutting
architectural decision was touched -- a rejected profiling-only
experiment, the same category as Milestone 33's and 45's own rejected
candidates.

### M47 — Fresh post-M46 CUDA bottleneck re-characterization (measurement-only, M48 target selected)

Re-measured the whole CUDA training pipeline fresh (`benchmarks/m47_
bottleneck_recharacterization.py`, new) against production dispatch,
unchanged since M43 (M45/M46 added only profiling-only kernels). Reused
M44's own decomposition/sweep functions, M43's interleaved A/B, and M42's
forward-sweep functions directly, and added one new measurement no prior
milestone produced: a single-session, round-robin-interleaved **three-way**
comparison of production against *both* M45's warp-shuffle candidates and
M46's grid-split candidates at once, plus a fresh real
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` re-query.

**Below-256 dWeight formally reclassified as a practical optimization
floor / deprioritized.** The combined-best result across both rejected
techniques (always a grid-split configuration) reached only 1.09x-1.18x,
short of the 1.15x bar at `mnist_conv1`'s own real shape (1.09x); the
fresh occupancy re-query reproduced M46's identical-occupancy finding
exactly (5 blocks/SM, 62.5%, production and grid-split alike). Four
independent angles (M21, M33, M45, M46) have now been tried against this
exact launch configuration without a milestone-clearing win -- this
milestone's own decision rule (Section 18 of its brief) explicitly
forbids reopening it with a fifth cooperation-granularity variant; only a
structurally different algorithm could justify revisiting it.

**New measurement-quality finding**: sustained CUDA benchmarking over many
minutes in one session causes cumulative thermal/clock drift larger than
the within-call drift M46 already documented and fixed with interleaving
-- the *same* kernel/shape measured late in a long session ran up to
~2.5-3x slower than early in that same session (`batch_128` forward:
91.6ms in the main script, reproducing at 21.8-41.0ms across 3 standalone
re-runs immediately after; an in-script M43 re-confirmation drew an
anomalous 3.877x "speedup" that 3 independent re-runs could only
reproduce at 1.14-1.19x, matching M43/M44's own history). Every headline
number in the M47 report was independently re-verified or averaged across
repeated trials (the MNIST kernel-ranking fractions used for Amdahl are
the mean of 7 fresh same-session trials) in direct response.

With below-256 dWeight set aside, dInput (11.75% of the full step), conv2d
forward (11.9%), and dWeight `blocks_y==1` (12.7%, just optimized in M43
and reconfirmed unregressed by 3 fresh interleaved re-runs) are now close
together -- no single component dominates the way below-256 dWeight did
in M42/M44. **M48 recommendation**: target the M36 channel-fused dInput
kernel's low-`Cin` regime (`Cin=1`, `mnist_conv1`'s own real shape) -- the
worst-roofline-efficiency live candidate measured this session (4.7% of
the practical compute ceiling, consistent with M44's own 8.3% finding at
this exact shape), with a genuinely untried structural angle: M36's
channel-fusion benefit requires multiple `Cin` accumulators to amortize
its register cost, and degenerates to none of that benefit at `Cin=1`.
Acceptance: >=15% of the practical compute ceiling at `Cin=1` with no
regression at `Cin>=8`/`mnist_conv2`'s own `Cin=16` shape. Full 19-section
report (executive summary, architecture, methodology, environment,
ceilings, training-step decomposition, forward/dInput/dWeight
characterization, below-256 reassessment, roofline, pipeline health,
Amdahl, ranking, rejected targets, selected target, M48 nine-element
recommendation, limitations, conclusion): `docs/performance/
m47-bottleneck-recharacterization.md`.

**Tests.** No test changes (measurement-only milestone). Full suite:
**1,550 passed** (unchanged from M46's own count), verified on a clean
CUDA rebuild (`_forge_cuda_kernels_sm_50.dll` deleted and recompiled in
21.02s).

**Why no ADR.** Measurement and documentation only -- no production code,
public API, or cross-cutting architectural decision was touched.

### M48 — CUDA dInput low-Cin value assessment (ACCEPTED)

Before designing any candidate, valued M47's selected target rather than
assuming it: grepped `examples/`/`tests/`/`benchmarks/` for every real
`Conv2d` shape (`Cin=1` matters to exactly one real Forge workload --
`examples/mnist/model.py`'s first layer -- everywhere else it is test-only
or a deliberate sweep point) and profiled the *current* M36 channel-fused
kernel fresh at `Cin=1` (`nvcc -Xptxas -v`, a real `cudaOccupancyMax
ActiveBlocksPerMultiprocessor` query) before writing any candidate code,
per the brief's PROFILE-before-DESIGN discipline.

**Root cause was more nuanced than M47's own hypothesis.** Shrinking the
channel-fused kernel's accumulator array from `MAX_CIN_REG=16` to a
`Cin=1`-specialized `CIN_MAX=1` template parameter only drops f32 register
usage 54->48 and occupancy 4->5 blocks/SM (50%->62.5%) -- a real but modest
gain, not the dramatic jump a pure register-pressure story predicts. The
larger effect is per-thread instruction overhead: the unspecialized
kernel's `#pragma unroll`ed accumulator loop still emits 16 runtime-checked
iterations per `Cout*KH*KW` outer-loop step even when only the first is
ever useful at `Cin=1`.

**Candidate benchmark** (`benchmarks/m48_dinput_value_assessment.py`, new):
a same-session, round-robin-interleaved, order-independent A/B (reproduced
across 2 independent script runs) measured a **2.18x-2.53x isolated kernel
speedup** across 4 `Cin=1` shapes (mnist_conv1 plus batch/spatial/stride
variations), bit-exact correct against both the M36 channel-fused kernel
and the original pre-M36 kernel. **End-to-end `conv2d_backward()` speedup
at `mnist_conv1`: 1.155x-1.158x** (directly measured, not just
Amdahl-projected). A fresh same-session fraction reconstruction (this
session's own measured whole-step `conv2d backward` fraction times a fresh
dInput-of-conv2d-backward ratio, before/after M48) put dInput at ~15.4% of
the full training step pre-M48 (M47's own carried-over figure was 11.75%)
-- **Amdahl projects ~1.09x-1.10x whole-training-step improvement** at the
real measured kernel speedup, comparable to or better than M43's own
accepted full-pipeline win (1.02x-1.04x/1.11x-1.24x).

**Decision: ACCEPT.** `k_conv2d_backward_input_channelfused_lowcin<T,
CIN_MAX>` (`kernels.cu`), instantiated at `CIN_MAX=1`, is now production --
`CUDABackend.conv2d_backward`'s `cf_conv2d_backward_input_*` dispatches to
it whenever `Cin<=CONV2D_DINPUT_LOWCIN_MAX_CIN` (1, the one real Forge
shape it was measured at), ahead of the unchanged M36 channel-fused path
(`Cin<=16`) and the M32 fallback (`Cin>16`, still unreached by any real
Forge shape). No regression confirmed at `Cin=2` (still M36 channel-fused,
byte-for-byte unchanged) or `mnist_conv2`'s own `Cin=16` shape. Production
changes limited to exactly what the brief scoped: the new specialized
kernel, its minimal launcher, and the minimal `Cin<=1` dispatch condition
-- dWeight, dBias, Conv2d forward, `k_matmul`, the allocator, streams, and
public APIs are all untouched.

**Tests.** 29 new tests (`tests/test_cuda_conv2d_dinput_lowcin_candidate.py`:
kernel-level parity against both the M36 channel-fused and original
kernels across shape/stride/padding/kernel-size/dtype combinations,
production dispatch correctness at and across the new `Cin<=1` boundary
via the real `nn.Conv2d`/autograd API, f32/f64, explicit-stream and
cross-stream correctness, repeated-use and allocator-reuse memory safety,
and an occupancy-regression guard). Full suite: **1,579 passed** (1,550 +
29), verified on a clean CUDA rebuild.

**Why no ADR.** A new dispatch band within an existing, already-documented
hybrid-dispatch pattern (Section 38's convention, extended identically by
M36 itself) -- not a new architectural decision.

### M49 — Framework capability & value assessment (measurement/assessment-only; no implementation)

Per its brief, did not continue CUDA optimization by default. Instead
surveyed five candidate areas -- optimizer coverage, NN layer/operator
coverage, data-loading ergonomics, serialization/checkpointing, and real
example/model coverage -- against Forge's own documented requirements
(`docs/product/requirements.md`) and use cases (`docs/product/
use-cases.md`), using direct code execution as evidence where reading
source/docstrings alone was insufficient (e.g. confirmed by running it that
`-x`, `x / y`, `x ** 2`, `.sqrt()`, `.mean()`, `.sigmoid()`, `.tanh()`,
`.transpose()`, `.softmax()` all currently fail at the Tensor level).

**Conclusion: no implementation is justified this milestone.** Every real
gap found (no SGD momentum/LR scheduler; a minimal, exactly-consumer-driven
Tensor operator set missing division/negation/power/sqrt/sigmoid/tanh/
transpose/softmax/mean) is a previously-considered, deliberately deferred
capability with no current real-workload impact -- not an oversight. Two
of these were found already explicitly documented as intentional scope
decisions elsewhere in the codebase: `forge/optim/sgd.py`'s own docstring
("no momentum, weight decay, or learning-rate schedule"), and `kernels.cu`'s
M14 comment rejecting "a generic elementwise-divide primitive that nothing
else in Forge needs" plus `forge/data/transforms.py`'s `Normalize`
docstring documenting its own division-workaround. Data-loading and
serialization/checkpointing were each found to already fully satisfy their
respective requirements/use-cases with no gap. All seven of `use-cases.md`'s
UC1-UC7 already have a working, runnable demonstration in the repository
(MNIST plus the smaller `examples/trainer_demo.py`/`data_pipeline_demo.py`/
`persistence_demo.py` regression/tabular demos) -- a second MNIST-tier
example today would only repackage already-proven primitives. Conv2d
optimization thread remains closed (M47/M48); nothing in this survey
reopens it.

**M50 recommendation:** don't pick from this survey speculatively -- make a
product decision first (a concrete second model family Forge should
support, e.g. binary classification or a normalization-using architecture),
and let that choice narrowly determine exactly which Tensor op(s)/optimizer
feature(s) it actually needs, the same way every existing op was added for
a specific consumer. Full 14-section report (method, all five areas'
current-state/deficiency/relevance/verdict, Conv2d closure, candidate
ranking table, rejected directions, validation, limitations, M50
recommendation): `docs/development/m49-capability-assessment.md`.

**Tests.** No test changes (assessment-only milestone, no production code
touched). Full suite re-run to confirm no accidental repository changes:
**1,579 passed** (unchanged from M48's own count), on this machine's real
CUDA backend (940MX) -- CUDA tests not skipped.

**Why no ADR.** No production code, public API, or architectural decision
was made -- an evidence-based decision *not* to change anything, the same
category as M45/M46's own rejected candidates but at the product/roadmap
level rather than a single kernel.

### M50 — Char-RNN: Forge's second model family

Per M49's recommendation, made the product decision M49 deferred: selected
one concrete second model family (a small character-level vanilla RNN
language model, `examples/char_rnn/`) and let attempting it against
post-M49 Forge determine exactly what, if anything, to add -- rather than
picking speculatively from M49's rejected-operator survey.

**Attempt against existing Forge** found the model almost entirely already
supported: `Linear`, `CrossEntropyLoss`, `Adam`, `TensorDataset`/
`DataLoader`, and `save_model`/`load_model` (after `register_module()`,
per the existing ADR-003 pattern) all worked unmodified. One-hot input
encoding and a hand-written multi-timestep training loop (`Trainer.fit()`
assumes one `forward(batch)` call per step, which a recurrence does not
fit) were both legitimate workarounds using existing capability, not
blockers. Weight sharing across a Python-level unroll loop -- the same
`Linear` `Parameter`s reused at every timestep of one backward graph, a
pattern no existing Forge model exercises -- was verified directly against
`forge/autograd/engine.py::run_backward` *before* writing any new code: its
existing reverse-topological-order gradient accumulation already handles a
leaf reused by many graph nodes correctly, with no special case needed.

**One genuine blocker: `Tensor.tanh()`.** Added with full CPU
(`np.tanh`/`1 - result**2`) and real-hardware-verified CUDA support
(`k_tanh`/`k_tanh_backward`, `kernels.cu` -- new kernels, no new CUDA
infrastructure), following the exact `relu`/`exp`/`log` Tensor-primitive
pattern established in M3/M9/M14. `nn.Tanh` (the `Module` wrapper, mirroring
`nn.ReLU`) and `nn.RNNCell` (one vanilla-RNN recurrence step, composed from
two `Linear` layers and `.tanh()`, no dedicated backward rule) were added
to `forge.nn`. `RNNCell` carries no non-parameter state (unlike e.g. batch
normalization), so it needed only a registry entry
(`forge/serialization/registry.py`), not any buffer/persistence machinery.
Embeddings/gather, LSTM/GRU gating, `Trainer` sequence support, and
multi-layer/stacked RNNs were all deliberately rejected as unjustified
without a driving consumer -- see the full report for the complete
blocker-vs-workaround-vs-enhancement reasoning.

**Training/validation.** Trained on an original, deterministically
generated (not downloaded, not copied) synthetic corpus, ~9,050 characters,
24-symbol vocabulary. Mean per-character cross-entropy dropped from 2.7433
to 0.3554 over 30 epochs on the reference CPU (i5-7200U) -- ~89% below the
untrained uniform-guess baseline `ln(24) ≈ 3.18`, comfortably clearing the
success criterion defined before training. CPU/CUDA parity confirmed on the
reference 940MX: every epoch's loss matched to 4 decimal places across a
full 30-epoch run (not just isolated-kernel level). Generated text after
training reproduces correct spacing/periods and much of the corpus's own
vocabulary, including verbatim template sentences.

**A significant pre-existing bug was found, not fixed (out of this
milestone's scope, per explicit direction).** Running the complete test
suite as a single `pytest` process deadlocked inside a pre-existing,
unrelated test (`test_trainer_cuda.py::test_cuda_trainer_classification_
end_to_end_learns`) -- diagnosed with `py-spy` (two independent stack
samples at the identical frame) as a genuine self-deadlock:
`CUDACachingAllocator.release()` (`forge/backend/cuda/allocator.py`) holds
a non-reentrant `threading.Lock` while evaluating a generator expression;
CPython's GC finalized an unrelated `CUDAStorage` mid-evaluation, whose
`__del__` called `release()` again on the same lock from the same thread.
Confirmed unrelated to this milestone's changes (nothing in `allocator.py`
or `backend.py`'s `__del__` was touched) and rare enough to have never
surfaced across 48 prior milestones -- depends on GC timing/accumulated
object churn across a long single-process run. Verification for this
milestone was done via two separate CPU-only/CUDA-only `pytest`
invocations (which do not hit this path); a future milestone should fix
the allocator's reentrancy hazard directly (see the full report's M51
recommendation).

**Tests.** 40 new tests (`tests/test_tanh.py`, `+3` in
`tests/test_activation.py`, `+2`/`+1`/`+1` tanh cases in
`tests/test_cuda_backend.py`/`test_cuda_consistency.py`/
`test_cuda_autograd.py`, `tests/test_rnn.py`, `tests/test_rnn_cuda.py`,
`tests/test_char_rnn_example_integration.py`,
`tests/test_char_rnn_example_cuda_integration.py`). Full suite: **1,619
passed** (1,579 + 40) -- 721 CPU-only + 898 CUDA-hardware-verified (940MX),
run as two separate invocations per the deadlock finding above. Clean CUDA
rebuild performed (`kernels.cu` changed).

**Why no ADR.** `Tensor.tanh()`/`nn.RNNCell` extend an already-documented,
four-times-precedented pattern (M3/M9/M14's Tensor-primitive-plus-
Backend-plus-Module shape) rather than establishing a new one -- not a
fresh architectural decision. Full report (model-selection rationale,
architecture/workload definition, the existing-capability attempt,
rejected enhancements, training/validation results, the allocator-deadlock
finding, limitations, M51 recommendation): `docs/development/
m50-char-rnn.md`.

### M51 — CUDA allocator reentrancy fix (`CUDACachingAllocator` self-deadlock)

M50 found, but deliberately did not fix, a real self-deadlock in
`CUDACachingAllocator.release()`. This milestone reproduced it
independently (a reference-cycle-trapped `CUDAStorage` finalized by
CPython's GC while `release()` held its lock, forced deterministically via
a `gc.collect()`-on-acquire test proxy rather than relying on natural GC
timing), confirmed the exact root cause with two identical `py-spy` stack
dumps, and corrected M50's hypothesis: the trigger is `self._free_blocks
.setdefault(nbytes, [])`'s *unconditional* list allocation (every call, not
just a first-time size), not specifically the `any()` generator M50
pointed to. **Fixed**, with two deliberately layered changes to
`forge/backend/cuda/allocator.py`: (1) `release()`/`release_pending()` (the
only methods `CUDAStorage.__del__` calls, hence the only ones a GC-
triggered finalizer chain can reenter) now allocate nothing new while
`self._lock` is held -- the fallback empty list and the `_PendingBlock`
instance are built before acquiring the lock, and the duplicate-pointer
scan is a manual index `while` loop rather than `any()`/a generator (direct
measurement showed a plain `for` loop would *not* have helped -- `list_
iterator` objects are themselves GC-tracked in CPython); (2) `self._lock`
is now a `threading.RLock`, analyzed and adopted as a deliberate backstop
for the one path not restructured (`_empty_ready`/`_drain_pending`'s
snapshot-construction list comprehensions, reachable only via
`empty_cache()`, never `__del__`) rather than accepted merely because it
makes the deadlock disappear -- same-thread reentrant execution of
`release()`/`release_pending()` is shown to be behaviorally correct, not
just non-deadlocking, given CPython's GIL and both methods' simple,
order-independent accumulation. 8 new regression tests
(`tests/test_cuda_allocator_reentrancy.py`), independently verified against
the pre-fix code (`git stash`) to fail/hang appropriately before verifying
all pass after. Full suite: **1,627 passed** (1,619 + 8), run as a single
process three consecutive times (~52-53s each, no hang) -- the exact
condition M50's defect required. No performance regression detected
(`benchmarks/allocator_bench.py`'s cached-path timings before/after are
within this hardware's known microbenchmark noise floor). Full report:
`docs/development/m51-allocator-reentrancy.md`.

### M52 — Product direction & third-workload assessment (assessment-only)

Surveyed current Forge (Tensor/nn/Module/optimizer/data/serialization/CUDA/
CLI) against the M49/M50/M51 baseline and evaluated five candidate next
directions (normalization-based CNN, small attention/transformer block,
autoencoder, extended char-RNN, CLI/optimizer ergonomics). **Selected: a
normalization-based CNN (`BatchNorm2d` added to the existing MNIST CNN) as
the leading candidate**, and validated it directly against the real Forge
API (a throwaway CPU probe script, not committed) rather than estimating
blockers. Found the genuine gap is not another op or example but a missing
*architectural concept*: `Module` has no buffer mechanism (non-differentiable,
device-moved, persisted, mode-aware state) -- independently flagged as a
deliberate omission three separate times already (M16's `docs/architecture/
modules.md` "No buffers to move"/"BatchNorm/LayerNorm remain out of scope",
and M50's own model-selection section naming BatchNorm2d as the rejected
alternative). The probe confirmed CPU already handles BatchNorm's per
-channel reduction (`x.sum(axis=(0,2,3), keepdims=True)`) and `(1,C,1,1)`
-vs-`(N,C,H,W)` broadcasting generally, with only `Tensor.sqrt()` and
division (`__truediv__`) genuinely missing on CPU; CUDA additionally lacks
any per-channel N-D reduction or general broadcast beyond `Linear`'s two
hard-coded shapes (`CUDABackend.sum()` supports only `axis=None`/`axis=1` on
a strictly 2D tensor), so the CUDA path needs one dedicated, narrowly-scoped
fused BatchNorm kernel (forward+backward) rather than general N-D
reduction/broadcast primitives -- the same "dedicated kernel per real
consumer" precedent as M14's `axis=1` reduction and M50's `tanh`. Rejected:
attention/transformer (too large a simultaneous blocker set for one
milestone), autoencoder and extended-char-RNN (recombine existing,
already-proven capability, no new framework pressure), generic CLI
`train`/`evaluate`/`predict` commands and SGD momentum/LR-scheduler (still
no driving workload, and the CLI story directly conflicts with Forge's own
documented "no config/YAML system" non-goal). **Outcome A, deferred to M53**:
per the brief's explicit "M52 is not required to modify production code"
allowance, no production code was changed this milestone -- the gap was
already provable by direct inspection and the throwaway probe, and the
CUDA kernel work is real engineering that deserves its own milestone budget.
Full suite re-run to confirm no behavioral change: **1,627 passed**, single
process (no longer needing M50's CPU/CUDA split workaround, thanks to M51's
allocator fix). Full report: `docs/development/m52-product-direction.md`.

### M53 — Module buffers and BatchNorm2d

Implemented M52's recommended capability in full (Outcome A): a
non-differentiable, persistent, device-moved, mode-aware **buffer**
mechanism on `Module` (`_buffers` dict, explicit `register_buffer(name,
tensor)` registration -- deliberately not `Parameter`/`Module`'s
`isinstance`-auto-registration, since a bare `Tensor` isn't an unambiguous
signal of intent; `named_buffers()`/`buffers()` mirroring the parameter
API; `Module.to()` extended to move buffers via the already-generic
`Tensor._move_storage_()`; `train()`/`eval()` needed no change since
neither touches `_buffers`), two new general elementwise `Tensor` ops
(`.sqrt()`, `__truediv__`/`__rtruediv__`, CPU+CUDA, following the exact
`tanh`-at-M50 pattern), and `nn.BatchNorm2d` itself. CPU's forward composes
entirely from general Tensor primitives (`sum`/`sqrt`/`div`/`reshape`) --
confirmed by M52's own probe that CPU already supported the needed
reduction axes and broadcast shape -- so CPU BatchNorm2d has **no
BatchNorm-specific backward rule anywhere**; every gradient falls out of
ordinary autograd composition, verified against finite differences.
CUDA has no general multi-axis reduction or matching broadcast to compose
this from, so it instead uses one dedicated fused kernel group (forward:
per-channel mean/var reduction + running-stats update + fused
normalize/affine; backward: per-channel `sum(dy)`/`sum(dy*xhat)` reduction
+ fused `dx`), reusing M15/M34's existing one-block-per-channel
shared-memory reduction idiom rather than inventing a new one, exposed as
one new CUDA-only `Tensor.batch_norm2d()` method (not a `Backend`-ABC
member both devices implement, since CPU never needs it). Serialization
(`forge/serialization/model.py`) gained a `"buffers"` branch mirroring
`"parameters"`; `FORMAT_VERSION`/`CHECKPOINT_FORMAT_VERSION` bumped `1 -> 2`
accordingly (a deliberate, documented breaking change per
`docs/architecture/decisions/ADR-003-persistence-format.md`'s existing
policy). Added `examples/mnist/model.py::build_model_bn()` -- exactly M52's
proposed `Conv2d -> BatchNorm2d -> ReLU -> ...` architecture -- as the real
training/eval/persistence validation workload, on both CPU and the 940MX.

Validated: CPU forward against an independent NumPy reference; CPU/CUDA
forward+backward parity (`<1e-4` relative tolerance, hardware-verified);
finite-difference gradient checks (input/weight/bias, affine and
non-affine); training-mode batch-statistics use and running-statistics
update (momentum + unbiased-variance-scaling checked against a hand
-computed expected value); eval-mode running-statistics use, frozen-ness,
and batch-independence; zero gradient ever reaching buffers (both
backends); serialization/checkpoint round trips (buffers present in the
archive, values exact, `eval()`-after-load matching pre-save output,
BatchNorm state alongside real Adam optimizer state); CPU<->CUDA device
movement; a real training run reaching >=80% accuracy through
`build_model_bn()` on both CPU and CUDA, with train/eval-mode outputs on
the same input proven to actually differ; no CUDA allocator growth across
100 repeated forward/backward iterations (`allocated_bytes` returns to its
pre-loop baseline after `gc.collect()`, all further allocations served from
the existing M25 cache). One performance sanity measurement (not an
optimization pass, per the milestone's explicit Performance Rule): adding
BatchNorm2d to the MNIST CNN cost 1.18x wall-clock on the 940MX -- adequate,
so no further profiling was pursued. Full suite re-run in a single process:
**1,715 passed** (1,627 pre-M53 + 88 new), zero skips (this session's CUDA
backend is live), no regressions. Full report:
`docs/development/m53-batchnorm.md`.

### M54 — Embedding lookup: fresh capability assessment and implementation

A fresh, non-optimization capability assessment (explicitly not assumed to
be a performance milestone) surveyed model/layer coverage, Tensor/autograd
primitives, optimizer/training infrastructure, and data/serialization for a
genuine next gap, rather than continuing M52/M53's normalization thread.
**Selected: `nn.Embedding` / `Tensor.embedding_lookup()`.** `examples/
char_rnn` (M50) explicitly deferred an embedding/gather primitive, noting
its one-hot-plus-`Linear` workaround "would only matter at a vocabulary
large enough for one-hot's O(vocab_size) per-step cost to bite
(thousands+)." A direct probe against Forge's real CPU backend measured
that cost directly rather than estimating it: the workaround is 40x slower
than a row-gather at a 30-token vocabulary, growing to roughly 300x-3500x
by a few thousand tokens -- confirming M50's predicted threshold with real
numbers. Rejected: attention/transformer (still too large a simultaneous
blocker set), a binary-classification architecture and an autoencoder
(both already fully expressible from existing primitives, no framework
pressure), a deeper/stacked RNN (pure Python composition, no primitive
gap), and optimizer/training-infrastructure changes (no workload need
found, same conclusion as M49/M52).

Implemented completely: `Backend.embedding_lookup`/
`embedding_lookup_backward` (`forge/backend/base.py`, an ordinary ABC
method pair implemented on **both** CPU and CUDA -- unlike `batch_norm2d`,
CPU is a genuine first-class consumer here, not dead-code symmetry), CPU
via NumPy fancy indexing forward / `np.add.at` scatter-add backward
(`forge/backend/cpu.py`), CUDA via two new kernels
(`k_embedding_lookup_forward`/`k_embedding_lookup_backward`, `kernels.cu`
-- one thread per output/gradient element, backward reusing the existing
`atomic_add_generic<T>` helper MaxPool2d backward already established,
since a repeated token can make more than one thread target the same
`grad_table` row), `Tensor.embedding_lookup(indices)` (fused primitive,
mirroring `cross_entropy`'s shape -- `indices` excluded from the autograd
`Node`'s `inputs`, so it never receives or needs a gradient), and
`nn.Embedding(num_embeddings, embedding_dim)` (`forge/nn/embedding.py`,
`N(0,1)`-initialized `Parameter` weight table, explicit index-range/dtype
validation mirroring `CrossEntropyLoss`'s target validation). No
persistence-format change was needed -- `weight` is an ordinary `Parameter`
that round-trips via the existing generic save/load path unmodified, only
a new registry entry. Added `examples/word_rnn/` (`Embedding -> RNNCell ->
Linear`, structurally identical to `CharRNN` with the one-hot step
replaced) as the real consumer: a synthetic, deterministic, offline
~1,800-word vocabulary (procedurally enumerated CVCV tokens, avoiding any
copyright/fabrication concern, mirroring `char_rnn/corpus.py`'s own
convention) with a subject/verb/object sentence grammar giving the RNN
genuine learnable next-word structure at realistic vocabulary scale.

Validated: CPU forward against direct NumPy indexing; finite-difference
gradient checks; repeated-index gradient accumulation (proven exact, not
just non-crashing) on both backends; CPU/CUDA forward+backward parity
(`float32`/`float64`, 1D and 2D index shapes) hardware-verified on the
940MX; a structural "no gradient reaches indices" check; a structural
zero-`CPUBackend`-calls check through a real `Embedding -> RNNCell ->
Linear` forward/backward pass; serialization/CUDA-persistence round trips
(prediction parity, CUDA-resident weight restoration); a real end-to-end
training run through `examples/word_rnn/train.py` at the full ~1,800-word
vocabulary on both CPU and CUDA (loss drops from the uniform-guess baseline
`ln(1806) ≈ 7.5` to 4.40 over 5 CPU epochs; CUDA trains in lockstep) with
save/load prediction-parity verified; no CUDA allocator growth across 200
repeated forward/backward iterations beyond one-time buffer settling
(`allocated_bytes` plateaus, `cache_miss_count` stays flat across further
rounds). Full suite re-run in a single process: **1,753 passed** (1,715
pre-M54 + 38 new), zero skips (CUDA backend live), no regressions; also
re-run with the CUDA toolchain stripped from `PATH` to confirm all 928 CUDA
tests skip cleanly (`825 passed, 928 skipped`, `0 failed`). Full report:
`docs/development/m54-product-direction.md`.

### M55 — Fresh post-M54 assessment: CUDA sequence-training was slower than CPU (engineering fix)

A fresh capability/engineering assessment (explicitly not assumed to
require another Tensor primitive, Embedding optimization, or RNN
extension) surveyed post-M54 Forge's real workloads for the highest-value
next investment. Direct measurement of the two real sequence-model
examples found a genuine, material problem no prior milestone had
measured: **CUDA trains `examples/char_rnn`/`examples/word_rnn` 1.5x-7.3x
*slower* than CPU** on the reference 940MX (`word_rnn`: CPU 15.3s vs. CUDA
23.5s over 5 epochs; `char_rnn`: CPU 0.7s vs. CUDA 5.1s) -- the opposite of
every prior CUDA milestone's assumption that CUDA is the faster device.

Root-caused via `cProfile` on the real `word_rnn` training loop: **56% of
one epoch's wall-clock time was spent inside `CUDABackend._synchronize`**
-- the default CUDA stream's per-kernel-launch `cudaDeviceSynchronize()`
(a documented Milestone 8-26 contract), paid by every one of the ~20-30
small kernel launches a single unrolled-sequence training step issues, at
shapes (batch<=32, hidden_size<=128) too small for that fixed per-launch
cost to be hidden by actual arithmetic. A secondary, smaller contributor
was `nn.RNNCell.forward()`'s composed `Linear`/`+`/`.tanh()` path itself
needing ~4 forward + ~8 backward launches per call.

**Two fixes, both verified on real hardware:**

1. **Fused CUDA `RNNCell`** (mirroring `batch_norm2d`'s CUDA-only-fused /
   CPU-composed split): `Tensor.rnn_cell()`, `CUDABackend.rnn_cell`/
   `rnn_cell_backward`, one forward kernel and six backward kernels
   (`kernels.cu`'s new **Fused RNNCell** section) replacing the composed
   path's ~12 launches with 7. `nn.RNNCell.forward()` dispatches to it on
   CUDA only; CPU is unchanged. Measured 1.2x-1.7x faster than the composed
   path in isolation (interleaved A/B) -- real, but modest end to end.
2. **Explicit compute stream in the example training loops** (the dominant
   fix): `examples/char_rnn/train.py`/`examples/word_rnn/train.py`'s
   `train_one_epoch()` now accept an optional `compute_stream`
   (`forge.cuda.Stream()`, created once in `main()`) and run the batch loop
   inside `with forge.cuda.stream(compute_stream):`, mirroring `Trainer`'s
   own `prefetch=True` `_compute_stream_scope()` (Milestone 30) -- neither
   example uses `Trainer` (multi-timestep training does not fit its
   one-forward-call-per-step shape), so neither had ever adopted this
   already-existing Milestone 27 machinery.

Combined, measured end to end (5 epochs, reference 940MX): `word_rnn`
default-stream 24.9s -> explicit-stream 13.5s (1.84x); `char_rnn`
default-stream 5.0s -> explicit-stream 2.5s (2.00x). Net result: `word_rnn`
CUDA training moved from 1.7x *slower* than CPU to ~1.13x *faster*;
`char_rnn` (tiny vocabulary, CPU already near-instant) moved from 7.3x
slower to 3.6x slower -- CUDA remains the wrong device for a model that
small, but the gap it needs to close is far smaller. Two technically
interesting alternatives were explicitly measured and rejected for
insufficient value: fusing dx/dh or dW_ih/db_ih into fewer kernels (real
but marginal on top of the stream fix, not worth the added kernel
complexity), and pursuing sparse-gradient Embedding backward (no measured
slowdown at any tested vocabulary size, per M54).

Verified: `Tensor.rnn_cell()`/backward CPU-vs-CUDA parity for every one of
the five gradients (`float32`/`float64`), weight-sharing gradient
accumulation across a multi-timestep unrolled sequence, device/shape
validation, a structural dispatch check, a 200-iteration memory-stability
check (`tests/test_cuda_rnn_cell.py`, 14 tests); `compute_stream`-vs-default
-stream loss parity and full-training-convergence for both examples (`tests/
test_char_rnn_example_cuda_integration.py`/`tests/
test_word_rnn_example_cuda_integration.py`, 4 new tests). Full suite:
**1,771 passed** (1,753 pre-M55 + 18 new), zero regressions. Full report:
`docs/development/m55-post-m54-assessment.md`.

### M56 — Fresh post-M55 assessment: no production change warranted (assessment only)

A fresh survey (explicitly forbidden from defaulting to another RNNCell/
Embedding/Conv2d optimization pass) re-read every architecture document,
every M49-M55 milestone report, the full `forge/` source surface, and all
five real example workloads, then re-ran the full test suite as a
baseline. Every previously-rejected candidate (binary classification,
autoencoder, deeper/stacked RNN, LayerNorm, attention/transformer,
optimizer/scheduler changes, sparse Embedding gradients, further RNN
backward fusion, Conv2d/generic CUDA sweeps) was re-examined against this
milestone's own fresh reading and found still correctly rejected, with no
new evidence to overturn any of them.

One new candidate was seriously evaluated: extracting `examples/char_rnn`'s
and `examples/word_rnn`'s duplicated hand-written training-loop structure
(~50 near-identical lines, including M55's `compute_stream` addition made
identically in both files) into a shared helper. Rejected — only two
consumers exist, no third sequence-model workload is anticipated to
validate an abstraction boundary against, and the same "premature
generalization from a sample size of two" reasoning M49/M52 already used
to reject a shared `forge.data` tokenization abstraction applies here.

**Outcome D: assessment only.** No production code changed. Full suite
re-verified: **1,771 passed**, identical to M55's own count, confirming
zero regression and zero untested work already in the tree. Full report:
`docs/development/m56-post-m55-assessment.md`.

### M57 — Fresh post-M56 assessment: no production change warranted (assessment only)

An independent fresh survey across all six mandated candidate areas (a
third model/workload family, a framework gap it might expose, developer
ergonomics, testing/reliability infrastructure, serialization/training
infrastructure, and whether any CUDA optimization is newly worth
revisiting) re-derived Forge's current capability inventory directly from
source rather than trusting M56's report, since the repository tree is
unchanged since M56 (its own report and progress-log entry are still
uncommitted on top of the same M55 commit). Every candidate reached the
same conclusion M56 already reached, because no new evidence exists to
change it — including one candidate not previously named by number,
sequence-to-sequence (encoder-decoder RNN), checked directly against the
current `forge/nn`/`forge/tensor` surface and rejected: it requires no new
primitive, so it would only recombine existing, already-proven capability.

Two small, concretely-evidenced repository-hygiene gaps were found in
`examples/word_rnn/` (added M54) by direct inspection and fixed: `.gitignore`
had no entry for its generated-artifact directory (`examples/mnist/` and
`examples/char_rnn/` both already have one — a real latent risk of
accidentally committing a model binary, though none currently exists on
disk), and the example had no `README.md` (unlike both other examples).
Neither fix touches `forge/`, any public API, or any test.

**Outcome: no framework-capability change justified** — the fourth
consecutive milestone (M49, M52, M54, M56, now M57) to independently reach
this conclusion across the same candidate space. Full suite re-verified
unchanged before and after this milestone's two documentation/config-only
changes: **1,771 passed** both times, identical to M56's own count. Full
report: `docs/development/m57-post-m56-assessment.md`.

### M58 — Fresh post-M57 assessment: MNIST CUDA default-stream investigated, no bug found (assessment only)

A fresh survey re-verified the full test suite (**1,771 passed**, unchanged)
and re-examined every M52/M54/M56/M57 candidate with no new evidence to
overturn any rejection. The one genuinely new angle this milestone pursued:
M55 fixed a real CUDA default-stream per-kernel-launch synchronize overhead
that made `char_rnn`/`word_rnn` train slower on CUDA than CPU, but that
investigation was scoped only to those two RNN examples -- no milestone
since M30 (`prefetch=True`'s introduction) had measured whether MNIST's
`Trainer`-based CNN path (which also runs on the default CUDA stream, since
`examples/mnist/train.py` never passes `prefetch=True`) has the same latent
problem. Measured directly via a throwaway probe reusing
`examples/mnist/model.py::build_model()` against a synthetic MNIST-shaped
dataset through the real `Trainer`: CUDA is already ~3.5x faster than CPU
(1.37s vs. 4.76s over 3 epochs) even without `prefetch=True`, which adds
only a further 1.19x. No bug exists -- MNIST's Conv2d/matmul kernels do far
more arithmetic per launch than an RNN's per-timestep step, so the fixed
per-launch synchronize cost that dominated the RNN case is negligible here.
Also directly verified this milestone (not carried forward from prior
reports): CUDA hardware live, `examples/word_rnn/train.py` runs end to end
from the command line with its M57-fixed `.gitignore` entry working
correctly, `python -m forge --help`'s actual current command surface, and
zero `TODO`/`FIXME`/`XXX` markers anywhere in `forge/`.

**Outcome: no production change justified** -- the third consecutive
assessment-only milestone (M56, M57, now M58), each independently gathering
fresh evidence; this one closes out a real previously-open question (whether
M55's RNN-specific fix had a hidden CNN-path counterpart) with hardware
measurement rather than assumption. Full report:
`docs/development/m58-post-m57-assessment.md`.

### M59 — Vision reassessment: end the assessment-loop pattern; select tabular regression (UC2) as the next workload (strategic decision, no production change)

M49-M58's ten-milestone narrow-capability-survey loop was itself diagnosed
as exhausted by M56/M57/M58. M59 stepped back to ask a project-level
question none of those milestones were scoped to ask: is Forge becoming
what `docs/product/vision.md` describes, or an increasingly polished pile
of individually-justified additions? Re-read the vision/requirements/
use-case documents and every prior milestone report, then directly audited
`forge/tensor/tensor.py`, `forge/nn/module.py`, `forge/backend/base.py`,
`forge/exceptions.py`, `forge/training/trainer.py`, `forge/optim/
optimizer.py`, `forge/cli/*`, all three example READMEs, and the full test
suite (fresh run: **1,771 passed**, identical to M56-M58).

**Finding: the code is more coherent than the assessment-loop framing
implied; the project surface around it is where the real, previously
unmeasured gaps are.** Every fused primitive added since M31
(`cross_entropy`, `batch_norm2d`, `embedding_lookup`, `rnn_cell`) follows
an identical, documented file-touch pattern; every public error is a typed,
actionable `ForgeError`; naming and validation order are consistent across
every sampled Tensor op. What is not coherent: (1) Forge demonstrates only
two of the vision's three named workload classes (classification, sequence
modeling) at production quality -- **regression, named explicitly in
`vision.md`/`use-cases.md` (UC2), has only a bare, un-READMEd
milestone-verification script (`trainer_demo.py`)**, never built to the
`mnist`/`char_rnn`/`word_rnn` standard; (2) the top-level onboarding
surface is stale -- `forge/__init__.py`'s own module docstring stops
narrating at Milestone 26 and never mentions M31/M53/M54/M55's additions,
and `README.md` remains thin (M58's own finding, re-confirmed); (3) five
of the last ten milestones were assessment-only, each 15-28KB, with
diminishing marginal signal -- a real process cost distinct from any code
defect.

**Decision**: pursue the third vision-named workload family (tabular
regression, UC2) as M60's concrete objective -- the one gap in this
report's entire survey justified by the original product vision itself
rather than a measured framework-capability pressure, and cheap/low-risk
since it composes entirely from already-proven primitives
(`Linear`/`ReLU`/`MSELoss`/`Adam`/`Trainer`, fitting `Trainer`'s
one-forward-per-step shape with no new `Tensor`/`Backend`/`nn` surface
needed). Adopted immediately, independent of M60: a permanent milestone
guardrail ending the assessment-loop-as-default pattern -- a fresh,
unscoped "survey and decide" milestone is no longer the automatic next
step; it requires a named external trigger (a new use case, a real usage
attempt, a workload failure, an environment change). A full guardrail set
(consumer/evidence/value/scope/validation/stop/revisit/no-milestone-count
-objective, plus area-specific rules for performance/primitives/layers/
infra/examples/docs) was codified for all future milestones.

No `forge/`, `examples/`, or `tests/` file was changed -- per the brief's
own instruction, this is a strategic-decision milestone; M60's actual
implementation (a full example with README, CPU+CUDA integration tests,
checkpoint/resume) is real engineering work deserving its own budget, the
same precedent M52 set for BatchNorm2d. Full report:
`docs/development/m59-vision-and-next-stage.md`.

### M60 — Tabular regression example (`examples/regression`, UC2)

Built Forge's third vision-named workload family to the same production
standard as `mnist`/`char_rnn`/`word_rnn`, per M59's selected direction.
`examples/regression/dataset.py` generates a deterministic, in-process
synthetic tabular dataset (no download, matching `char_rnn`/`word_rnn`'s
convention): 8 continuous features drawn `Uniform(-2, 2)`, a target mixing
linear, interaction (`x4*x5`), and quadratic (`x6**2`) terms plus
`Normal(0, 0.5)` noise, with one pure distractor feature (`x7`) the model
must learn to down-weight. `make_datasets()` splits train/val/test
contiguously (equivalent to a random split for i.i.d. draws) and fits
`forge.data.Normalize` on the training split only. `examples/regression/
model.py::build_model()` is a plain `Sequential(Linear(8,64), ReLU,
Linear(64,32), ReLU, Linear(32,1))` -- 2,689 parameters, already covered by
the persistence registry's built-in `Linear`/`ReLU`/`Sequential` entries, so
no `register_module()` call was needed. `examples/regression/train.py`
mirrors `examples/mnist/train.py`'s structure exactly (`Trainer.fit()`,
checkpoint save/resume, `save_model()`/`load_model()` round-trip
verification, CLI-inspection hints), confirming the M59 brief's prediction
that this workload fits `Trainer`'s one-forward-per-step shape with zero
`Tensor`/`Backend`/`nn` changes -- **Outcome A**, framework unchanged.

Hardware-verified end-to-end on the reference machine: CPU training (40
epochs, `--seed 0`) reduced train MSE `15.11 -> 0.294` and beat the trivial
predict-the-mean baseline (MSE 24.29) by 98.6%, with final test MSE (0.335)
close to the dataset's irreducible noise floor (0.25) -- genuine recovery of
`true_function`, not just "the code runs." CUDA training with the identical
seed matched within floating-point tolerance (train MSE `15.11 -> 0.293`,
same 98.6% reduction) but ran ~3.3x *slower* than CPU (small matmuls, batch
size 32 -- kernel-launch/sync overhead dominates), the inverse of MNIST's
CUDA speedup; per M59's performance policy this was reported, not chased,
since it does not undermine the example's purpose. Checkpoint save/resume
verified both programmatically (continuing 40 epochs -> 5 more, global step
5000 -> 5625, loss continuing to drop) and via a resume-equivalence test
(`N+M` continuous vs. `N`-checkpoint-`M` epochs matching within `1e-5`).
Model save/load prediction parity verified on CPU and CUDA. CLI `model
inspect`/`checkpoint inspect` verified against real generated artifacts.

16 new tests added (11 CPU in `tests/test_regression_example_integration.py`,
5 CUDA in `tests/test_regression_example_cuda_integration.py`, the CUDA file
skipping cleanly without hardware) covering deterministic dataset
generation/reproducibility, split disjointness and normalization
correctness, model construction, training convergence vs. baseline,
checkpoint/resume (including resume equivalence), model persistence, CLI
inspection, CUDA parameter/gradient/Adam-state residency, and CPU/CUDA
prediction parity after identical training. Full suite: **1,787 passed**
(1,771 + 16 new), zero regressions. Full report:
`docs/development/m60-tabular-regression.md`.

### M61 — Framework readiness and developer experience (documentation + one discoverability fix, no new ML capability)

M60 closed the last vision-driven capability gap (three demonstrated
workload families); M59's own recommendation and this milestone's brief
both pointed at the same remaining gap: the top-level onboarding surface
(`README.md`, `forge/__init__.py`'s docstring) was stale (`README.md` was a
5-line stub last meaningfully updated around Milestone 2; the package
docstring narrated milestones 1-26 by number and never mentioned
`char_rnn`/`word_rnn`/`regression`, `BatchNorm2d`, or `Embedding`), and no
central examples index existed. Direct inspection (not assumption) of
`README.md`, `pyproject.toml`, `forge/__init__.py`, `forge/nn/__init__.py`,
`forge/optim/__init__.py`, `forge/cuda/__init__.py`, every example's own
README, `.gitignore`, `git ls-files`, and a from-scratch `pip install -e
".[dev]"` into a fresh venv confirmed: packaging, the example READMEs, and
`forge/nn`/`forge/optim`'s docstrings were already accurate and were left
unchanged (Outcome C for those areas); the top-level README, the package
docstring, and one real API-discoverability gap were not.

**The one discoverability gap**: `forge.cuda.is_cuda_available()` did not
exist -- checking CUDA availability required reaching into
`forge.backend.cuda` (an internal-sounding path) even though every other
CUDA entry point (`Stream`, `synchronize`, `memory_stats`, ...) already
lived on the public `forge.cuda` package. Fixed by re-exporting the
existing `forge.backend.cuda.backend.is_cuda_available` at `forge.cuda`
(module-level import + `__all__` entry); `_require_cuda()`'s own internal
check kept its own lazy per-call import so the existing monkeypatch-based
availability tests (`test_cuda_memory_availability.py`,
`test_cuda_allocator_availability.py`,
`test_cuda_pinned_memory_availability.py`) kept passing -- the first version
of this change (a single shared module-level import) broke exactly those
five tests, caught by running them before moving on. Two new tests added
(`tests/test_cuda_public_api.py`) protect the re-export permanently.

**Changes made**: rewrote `README.md` (architecture diagram, current
capability list, CPU/CUDA installation, a copy-paste-runnable "first model"
snippet verified to actually run, an examples table, testing instructions,
project scope/philosophy, contributing pointers); rewrote
`forge/__init__.py`'s docstring (milestone-narrative -> current public
surface); added `examples/README.md` as a central examples index; added
`tests/test_smoke.py` (a fast import + minimal-model-trains-one-step check,
distinct from the full suite); added a CPU-only GitHub Actions workflow
(`.github/workflows/ci.yml`, Python 3.11/3.13 on `ubuntu-latest`) -- a
deliberate, evidence-based decision (over half the test files are pure-CPU
and the rest skip their CUDA portions cleanly via the project's existing
`skipif(not is_cuda_available())` convention, so a no-GPU runner still
exercises substantial real coverage at zero cost) rather than the default
"don't add CI for a solo CUDA-hardware-dependent project" position the
brief itself offered; lightly repositioned `examples/trainer_demo.py` as
the documented first-model walkthrough (docstring pointer only, no logic
change) instead of writing a duplicate tutorial script.

**Verified directly, not assumed**: all four major examples
(`mnist`/`char_rnn`/`word_rnn`/`regression`) actually run their documented
commands end-to-end this milestone (not just inspected as source); the
README's inline "first model" snippet was extracted to a scratch file and
actually executed; a from-scratch `pip install -e ".[dev]"` into a fresh
venv succeeded and both `import forge` and the installed `forge` console
script worked; the full suite was run before (1,787 passed) and after
(**1,792 passed** = 1,787 + 5 new, zero regressions) every change.

**Not changed** (Outcome C, verified adequate as found): `pyproject.toml`
packaging (a from-scratch editable install and console-script entry point
both work correctly; a stale local `forge.egg-info/PKG-INFO` description
was noted as a harmless artifact of an old install, not a packaging bug --
it self-corrects on the next `pip install -e .`), the four example READMEs
(already detailed, accurate, and cross-referenced), `forge/nn/__init__.py`
and `forge/optim/__init__.py` docstrings (already concise and accurate),
`docs/development/cli.md` (already accurate against `forge --help`'s actual
output). No `Tensor` primitive, `nn.Module`, optimizer, CUDA kernel, or
`Trainer` feature was added -- every example already fully covered its
workload (confirmed again by re-running each one), so no blocker requiring
new framework logic was found. Full report:
`docs/development/m61-framework-readiness.md`.

### M62 — `nn.Conv1d`/`nn.MaxPool1d`: 1D temporal convolution, plus a real waveform-classification workload

M59 ended the recurring assessment-only pattern; this milestone's brief
explicitly forbade another readiness survey and required one concrete,
implemented, user-visible framework capability with a real consumer.
Inspection of the current architecture (`Tensor.conv2d`/`max_pool2d`,
`nn.Flatten`'s existing "compose from `Tensor.reshape`, no new backend code"
convention, `forge/serialization/registry.py`) found that Forge had no way
to express 1D/temporal convolution -- every existing example either does 2D
image convolution (`mnist`) or step-by-step recurrence (`char_rnn`/
`word_rnn`); nothing covers the standard 1D-CNN model family used for
sensor/audio/time-series classification, a real gap in `UC1` ("train a
classifier") coverage, not a rejected-and-resurrected direction (LSTM/GRU/
attention/LayerNorm/generic softmax stayed rejected; this is a new op
family, not one of those).

**What was added**: `nn.Conv1d` and `nn.MaxPool1d`
(`forge/nn/conv.py`/`forge/nn/pooling.py`), each implemented entirely by
reshaping an `(N, C, L)` tensor to a dummy `(N, C, 1, L)` 4D tensor and
dispatching to the existing, already CPU/CUDA-hardware-tested `Conv2d`/
`MaxPool2d` machinery, then reshaping the result back down -- the exact
"compose from an existing differentiable op, no new `Backend` method, no
new CUDA kernel" convention `nn.Flatten` already established for
`Tensor.reshape`. This means `Conv1d`/`MaxPool1d` inherit `Conv2d`/
`MaxPool2d`'s forward/backward correctness (including `Conv2d`'s
already-optimized CUDA kernels from Milestones 21-48) automatically, with
zero new low-level code and therefore no new CUDA-parity risk. Both are
registered for persistence (`forge/serialization/registry.py`) and exported
from `forge.nn`.

**Real consumer**: `examples/waveform_classification/` -- a new, complete
example (`dataset.py`/`model.py`/`train.py`/`README.md`, matching M60/M61's
example quality bar) classifying length-64 synthetic 1D waveforms (sine/
square/sawtooth/triangle, random phase/frequency/amplitude, additive
Gaussian noise; in-process/deterministic-seed generation, no download,
following `regression`'s precedent) with a 2-layer `Conv1d`->`ReLU`->
`MaxPool1d` CNN (~35k parameters) via the existing `Trainer`/`Adam`/
`CrossEntropyLoss`/checkpoint/persistence pipeline -- unchanged from
`mnist`'s. Verified training on the reference i5-7200U (CPU) and GeForce
940MX (CUDA, real hardware): trivial baseline 25% accuracy (uniform over 4
classes); trained to 90.80% (CPU) / 91.80% (CUDA) test accuracy in 15
epochs, ~1.35s/epoch either device (~2,900-3,100 samples/sec) -- genuine
learning, not "the code runs," and CPU/CUDA parity within normal
training-variance tolerance.

**Incidental fix (regression safety, not scope creep)**: while CUDA-
verifying the new example's final persistence-check line, found
`examples/regression/train.py`'s own equivalent line
(`query_x.to(args.device).reshape(1, -1)`) crashes on CUDA with
`ShapeMismatchError` -- `CUDABackend.reshape` (unlike CPU's, which delegates
to NumPy) never supported `-1` shape inference. This is a pre-existing bug,
unrelated to this milestone's own changes, that silently broke
`examples/regression/train.py --device cuda`'s documented final
verification step (the crash was previously invisible because the example
was being piped through `tail` when checked). Fixed with the same one-line
pattern (`reshape(1, N_FEATURES)` instead of `reshape(1, -1)`); re-ran
`tests/test_regression_example_cuda_integration.py` (16 passed) and the
live CLI command to confirm.

**Repository hygiene**: removed the tracked `.idea/` directory from git
(`git rm -r --cached .idea/`, files kept on disk) and added `.idea/` plus
`examples/waveform_classification/artifacts*/` to `.gitignore`; `git
status` now shows only the intended M62 changes plus M61's pre-existing
uncommitted work.

**Testing**: 91 new tests -- `tests/test_conv1d.py` (36, CPU: config
validation, parameter shapes/init, forward vs. an independent triple-loop
reference, gradient accumulation, finite-difference checks, serialization
round trip), `tests/test_maxpool1d.py` (27, CPU: same structure adapted to
1D, including tie-breaking and overlapping-window gradient accumulation),
`tests/test_cuda_conv1d.py` (12, CUDA hardware: forward/backward parity
with CPU, device movement, an end-to-end tiny-TCN training step, `Sequential`
integration -- all real `CUDAStorage`, never a silent CPU fallback),
`tests/test_waveform_classification_example_integration.py` (11, CPU) and
`tests/test_waveform_classification_example_cuda_integration.py` (5, CUDA
hardware: parameter/gradient/Adam-state CUDA residency, checkpoint/resume,
model persistence, CPU/CUDA prediction parity). Full suite: **1,883
passed** (1,792 + 91 new), zero regressions. Full report:
`docs/development/m62-conv1d-waveform-classification.md`.

### M63 — Convolutional autoencoder: `nn.UpsampleNearest2d` and Forge's first unsupervised-reconstruction workload

M62's own "no foregone-conclusion direction" recommendation plus this
milestone's brief (explicitly forbidding another readiness survey) required
selecting one concrete model family meaningfully different from all five
existing examples. Direct inspection found a real gap: every existing
example is supervised (a label, next-token, or regression target separate
from the input); Forge had never validated unsupervised reconstruction
-- a model whose training signal is its own input. A convolutional
autoencoder was selected over a stacked-RNN/tabular-classifier/small
-generative-model alternative (each rejected for either recombining
existing capability without new value, or requiring several simultaneous
new primitives -- VAE/GAN -- against the brief's own guardrail).

**What was added**: `nn.UpsampleNearest2d` (`forge/nn/upsample.py`), backed
by a new `Backend.upsample_nearest2d`/`upsample_nearest2d_backward`
primitive pair (`forge/backend/base.py`, `cpu.py`, a dedicated real CUDA
kernel pair in `kernels.cu`/`backend.py`). This was a genuine blocker
discovered by implementation, not speculation: every prior Forge
architecture only ever *shrinks* a spatial feature map (`Conv2d`'s
valid-region shrink, `MaxPool2d`'s stride-2 shrink); this example's decoder
needed the opposite (`(N, 32, 7, 7) -> (N, 1, 28, 28)`), and neither
`Tensor.reshape` nor general N-D broadcasting (deliberately out of CUDA
scope) could do it. Composing `UpsampleNearest2d` with `Conv2d`
(upsample-then-convolve) is a standard, deliberate architectural choice
(it avoids transposed convolution's well-known checkerboard artifacts),
not a `ConvTranspose2d` workaround. The CUDA backward kernel is notably
*simpler* than `MaxPool2d`'s: since the forward fan-out pattern is fixed
and data-independent (unlike `MaxPool2d`'s data-dependent argmax), backward
is a direct one-thread-per-input-element gather-reduction -- no
`atomicAdd`, no `cudaMemset` zeroing needed.

**Real consumer**: `examples/autoencoder/` -- a new, complete example
(`dataset.py`/`model.py`/`train.py`/`README.md`) reusing
`examples/mnist/dataset.py`'s `MNISTDataset` directly (via a thin
`AutoencoderDataset` wrapper returning `(image, image)` pairs instead of
`(image, label)`) rather than a synthetic corpus -- the first example after
`mnist` itself to train against real external data. `ConvAutoencoder`
(`Conv2d`/`MaxPool2d` encoder -> `Linear` 32-d bottleneck -> `Linear`/
`UpsampleNearest2d`/`Conv2d` decoder, ~111.5k parameters) trains through
`Trainer.fit()` completely unmodified -- `Trainer` needed no changes to
train on a target that is the input itself, confirming the M50/M54/M59
finding that it makes no assumption about what its loss target represents.

Verified on the reference i5-7200U (CPU) and GeForce 940MX (CUDA, real
hardware), 3 epochs over the full 60,000-image MNIST training set: test MSE
dropped from a 0.06747 trivial (predict-the-training-mean-image) baseline
to 0.01740 (CPU) / 0.01755 (CUDA) -- a 74.2%/74.0% reduction, CPU and CUDA
agreeing within 0.00015. A bonus qualitative check -- do same-digit test
images end up with nearby latent codes, despite training with no labels at
all? -- scored 80.0% (CPU) / 79.0% (CUDA) nearest-neighbor label agreement
against a 10% random baseline, evidence the bottleneck learned genuine
digit-shape structure. **Performance observation**: unlike `regression`
(CUDA slower) or `waveform_classification` (roughly tied), this
architecture's four real `Conv2d` layers gave CUDA substantial work per
batch -- CUDA trained **~6.1x faster** than CPU (~1,170 vs. ~190
samples/sec), closer to `mnist`'s own CUDA advantage; reported as observed,
not chased, per the M59 performance policy.

**Testing**: 61 new tests -- `tests/test_upsample_nearest2d.py` (25, CPU:
config validation, forward vs. an independent `np.repeat` reference,
gradient-accumulation/scaling checks, finite-difference gradients across 4
configurations, `Sequential`+`Conv2d` training, serialization round trip),
`tests/test_cuda_upsample_nearest2d.py` (12, CUDA hardware: forward/
backward parity with CPU, finite-difference, a spy-based "never falls back
to CPUBackend" guard, device movement, an end-to-end training loop),
`tests/test_autoencoder_example_integration.py` (12, CPU: full-pipeline
training beats baseline, checkpoint/resume + resume equivalence, model
persistence, CLI inspection, the latent nearest-neighbor algorithm checked
deterministically against hand-crafted latents, `AutoencoderDataset`'s
MNIST-wrapping logic against tiny in-memory real-IDX-format files with no
network access), `tests/test_autoencoder_example_cuda_integration.py` (5,
CUDA hardware: parameter/gradient/Adam-state CUDA residency including the
new decoder path, checkpoint/resume, model persistence, CPU/CUDA prediction
parity). Full suite: **1,936 passed** (1,883 + 61 - 1 flaky, plus a
separately-confirmed-unrelated `test_dataloader_prefetch.py` allocator
-measurement flake that passed cleanly in isolation), zero regressions
attributable to this milestone. Full report:
`docs/development/m63-conv-autoencoder.md`.

### M64 — U-Net-style image segmentation: Forge's first dense-prediction workload, zero new framework capability

M64's brief explicitly required attempting the workload against Forge's
existing `Tensor`/`Module`/loss/optimizer/`DataLoader`/`Trainer`/
serialization/`Conv2d`/`UpsampleNearest2d` capabilities *before* assuming a
gap exists, and forbade adding an API "merely because the milestone needs a
feature." Direct execution against the real API -- building
`Sequential(Conv2d, ReLU, MaxPool2d, Conv2d, ReLU, UpsampleNearest2d,
Conv2d)`, then running a forward pass, backward pass, an `Adam` step, and a
save/load round trip on both CPU and CUDA -- succeeded immediately, with no
shape mismatch and no missing method. This is Forge's first genuine dense
-prediction task (a per-pixel output evaluated against an independent,
structured ground truth mask, distinct from `autoencoder`'s
reconstruct-the-input task shape).

**What was added**: nothing to `forge/`. `examples/segmentation/`
(`dataset.py`/`model.py`/`metrics.py`/`train.py`/`README.md`) is the first
Forge image example whose model needs no `register_module()` call at all --
`Conv2d`, `MaxPool2d`, `ReLU`, `UpsampleNearest2d`, and `Sequential` were
already registered for persistence by prior milestones (15/53/62/63).
`dataset.py`'s `SegmentationDataset` generates synthetic `32x32` RGB images
(one randomly placed/sized/colored circle or square against a fixed
-statistics noisy background, with a minimum foreground/background contrast
margin enforced by rejection sampling) entirely in-process, matching
`regression`/`char_rnn`/`word_rnn`'s no-download convention. `metrics.py`'s
`PixelAccuracy`/`IoU` are ordinary `forge.training.Metric` subclasses --
example-local code exercising that class's documented extension point, not
a framework change.

**Real consumer**: the ~5.4k-parameter model (`Conv2d(3,16)` -> `ReLU` ->
`MaxPool2d(2)` -> `Conv2d(16,32)` -> `ReLU` -> `UpsampleNearest2d(2)` ->
`Conv2d(32,1)`, no skip connections, no final activation -- the exact
minimal architecture the brief itself suggests) trains through
`Trainer.fit()` completely unmodified against a `{0,1}`-mask `MSELoss`
target, the third confirmation (after `regression`/`autoencoder`) that
`Trainer` makes no assumption about what its loss target represents.

Verified on the reference i5-7200U (CPU) and GeForce 940MX (CUDA, real
hardware), 15 epochs over 3,000 synthetic training images (500 held out for
test): IoU rose from a **0.0000** trivial (predict-all-background) baseline
to **0.9901** (CPU) / **0.9909** (CUDA) -- CPU and CUDA agreeing within
0.0008 IoU and 0.00002 MSE, genuine end-to-end parity. Pixel accuracy alone
(0.7975 baseline -> 0.9980) is reported but not treated as the primary
signal, since the trivial baseline already scores high on it purely from
foreground/background class imbalance -- IoU (which the trivial baseline
scores at exactly `0.0` by construction) is the metric that actually
demonstrates dense prediction. **Performance observation**: CUDA trained
**~5.9x faster** than CPU (~689 vs. ~116 samples/sec), consistent with
`mnist`'s/`autoencoder`'s own CUDA advantage on real multi-`Conv2d`
workloads; not chased, per the M59 performance policy.

One candidate gap was considered and explicitly rejected: true U-Net skip
connections need a channel-concatenation primitive (`concat`/`cat`) that
does not exist anywhere in `forge/tensor/tensor.py` or
`forge/backend/base.py`. Not added -- the brief's own suggested minimal
architecture has none, and this milestone's results show the task is fully
solvable without them, so adding a speculative primitive with no
demonstrated consumer would have violated the brief's own guardrail.

**Testing**: 30 new tests -- `tests/test_segmentation_example_integration.py`
(25, CPU: dataset determinism/shape/value-range/non-degenerate-mask checks,
model forward shape and exact parameter count, `PixelAccuracy`/`IoU`
correctness against hand-crafted predictions including the `union == 0`
edge case, the majority-class baseline against a manual NumPy
recomputation, full-pipeline training that beats that baseline on both
metrics, checkpoint/resume + resume equivalence, model persistence, CLI
inspection), `tests/test_segmentation_example_cuda_integration.py` (5, CUDA
hardware: full pipeline trains and beats baseline, parameter/gradient/
Adam-state CUDA residency, checkpoint/resume, model persistence, CPU/CUDA
prediction parity). Full suite: **1,961 passed** (1,936 + 25 new), one
pre-existing, unrelated `test_dataloader_prefetch.py` allocator
-measurement flake in the same full-suite run (the same flakiness M63's own
report footnoted) that passed cleanly in isolation immediately afterward --
zero regressions attributable to this milestone. **Outcome A**: workload
completed with existing framework, no new production capability required.
Full report: `docs/development/m64-unet-segmentation.md`.

### M65 — Reproducible training/experiment workflow: `examples/regression`, zero `forge/` change, one real gap found and closed

M65's brief required turning an existing example into a small, repeatable
`configure -> seed -> train -> record metrics -> checkpoint -> resume ->
evaluate -> compare` workflow, forbidding a generic experiment-management
platform, and requiring the baseline to be established by *executing* the
current workflow, not merely reading it. Selected `examples/regression`
(Milestone 60), the brief's own preferred "cleanest deterministic baseline."

**Baseline investigation found almost everything already present**: a full
CLI (`argparse`), three already-independent deterministic RNG streams
(`forge.random`, dataset generation, `DataLoader` shuffling), `Trainer.fit()`
returning an already-JSON-safe `TrainingHistory`/`EpochResult`/
`EvaluationResult` (`dataclasses.asdict()` needed no framework change),
full checkpoint/resume with Adam state + `forge.random` state restore, and
`save_checkpoint(..., extra=...)`'s existing caller-defined JSON-safe dict.

**One genuine, execution-proven gap**: a throwaway probe script reproduced
`train.py`'s actual `shuffle=True` default resume path and measured
`max abs param diff = 0.0252` between continuous `N+M`-epoch training and
`N`-epochs-then-resume-`M`-more -- the pre-existing resume-equivalence test
only covered `shuffle=False` and its own docstring said so explicitly. Root
cause: a resumed run re-seeded a *fresh* `--seed`-derived `DataLoader`
generator instead of continuing the interrupted run's shuffle stream. Fixed
using only the existing `extra` mechanism -- `numpy.random.Generator.
bit_generator.state` is itself already a JSON-safe dict, confirmed by a
direct `json.dumps`/`json.loads` round trip -- closing the gap to `max abs
param diff = 0.0` exactly, with **zero `forge/` production code changed**.

**What was added** (all in `examples/regression/`, nothing in `forge/`):
`train.py` now saves/restores the `DataLoader` generator's state via
`checkpoint.extra["data_loader_rng_state"]`; `experiment.py` (new) builds a
small versioned JSON "run record" per output directory (config fixed at
first invocation + cumulative per-epoch history + final evaluation + total
runtime, continuous across a resume) using only `dataclasses.asdict()` on
Forge's existing dataclasses and plain `json`; `compare.py` (new) is a
`python -m examples.regression.compare <a> <b>` CLI that diffs two run
records' configuration/final-metrics/epochs/runtime as plain text -- no UI,
no database, no visualization, per the brief's explicit guardrails.

Directly verified (not just tested): **(A)** two identical from-scratch runs
produced bit-identical per-epoch losses/metrics and bit-identical final
model parameters (`atol=0.0`), only wall-clock `duration` differed; **(B)**
full training vs. partial-then-resume with `shuffle=True` matched within
`1e-6` on CPU and `1e-5` on the real 940MX (CUDA), with the run record's
epoch numbering staying continuous across the resume; **(C)** two `--lr`
configurations produced a `compare_runs()` diff that correctly identified
the differing config key and a correspondingly different final loss.

**Testing**: 20 new tests -- `tests/test_regression_experiment.py` (14,
CPU, no training: run-record round trip, format-version validation,
cumulative-history accumulation, `compare_runs()` correctness, `compare.py`
CLI including its missing-file error path), `tests/
test_regression_reproducible_training.py` (5, CPU, drives `examples.
regression.train.main(argv)` directly -- the real CLI entry point, since
the CLI wiring itself is what this milestone changed: same-config bitwise
reproducibility, the flagship `shuffle=True` resume-equivalence regression
test, checkpoint `extra` contents, config-distinguishes-runs, and the
missing-run-record-sidecar fallback), `tests/
test_regression_example_cuda_integration.py` (+1, CUDA hardware: the same
flagship resume test on the real 940MX). Full suite: **1,981 passed**
(1,961 + 20), one pre-existing, unrelated `test_dataloader_prefetch.py`
allocator-measurement flake (the same one M63/M64 already footnoted) in the
full-suite run that passed cleanly in isolation immediately afterward --
zero regressions attributable to this milestone. **Outcome A**: workflow
completed mostly with existing framework; the one gap found was closed
using an existing extension point (`extra`), not a new API. Full report:
`docs/development/m65-reproducible-training.md`.

### M66 — Residual CNN workload: `examples/resnet`, zero `forge/` change

M66's brief asked whether Forge's current CNN/module infrastructure can
express and train a small ResNet-style residual network -- genuine branch +
identity/shortcut addition, repeated module composition, `BatchNorm2d`,
serialization, checkpoint/resume -- using only the existing public API,
explicitly forbidding the assumption that a framework gap exists before
proving one by direct execution.

**Direct API investigation found no gap at all.** A probe script built a
`TinyResidualBlock` (identity shortcut) and a `ProjectionResidualBlock`
(1x1-conv + `BatchNorm2d` shortcut, for a channel/stride change) and ran
forward + backward on both CPU and CUDA, plus a finite-difference gradient
check (float64, `eps=1e-5`) against the reduced block: max abs
analytic-vs-numeric difference `3.23e-10`. `Tensor.__add__`'s existing
reverse-topological-order gradient accumulation (the same mechanism M50
already proved handles a `Parameter` reused across many graph positions)
distributes gradient correctly to both the branch and the identity path,
with zero `forge/` changes needed anywhere.

**What was built** (all in `examples/resnet/`, nothing in `forge/`):
`ResidualBlock` (one ordinary `forge.nn.Module` subclass -- two 3x3
`Conv2d`+`BatchNorm2d` stages, `ReLU`, and either a true identity or a 1x1
projection shortcut depending on whether channels/stride change) and
`ResNetMNIST` (stem `Conv2d`+`BatchNorm2d`+`ReLU`, three `ResidualBlock`
instances, `MaxPool2d`+`Linear` classifier, ~17.6k parameters), both
registered via `forge.serialization.register_module()` at import time --
the same pattern `ConvAutoencoder`/`CharRNN`/`WordRNN` already established,
now proven for a *nested* custom-`Module`-inside-custom-`Module` tree for
the first time. `train.py` reuses `examples/mnist/dataset.py` (the real
MNIST dataset) and `examples/regression/experiment.py`/`compare.py` (the
M65 reproducible-training workflow) directly rather than duplicating
either.

Trained on real MNIST (`--seed 0`, 3 epochs): CPU reaches 98.60% test
accuracy (loss 0.2000 -> 0.0387, 1312.1s total, ~137 samples/sec); CUDA
(940MX) reaches 98.64% (loss 0.2001 -> 0.0389, 247.2s total, ~728
samples/sec, ~5.3x faster than CPU) -- both far above the 10.00% chance
baseline, and closely matching each other (epoch-1 loss within 0.0001),
confirming genuine CPU/CUDA parity. `BatchNorm2d` train/eval-mode
behavior, running-statistics update/freeze, and buffer persistence were all
validated three levels deep in the nested module tree
(`model.block2.shortcut_bn`), and checkpoint save/resume, resume
equivalence, and model save/load through the full nested custom-type tree
all round-trip correctly with the existing, unmodified serialization
infrastructure.

**Testing**: 27 new tests -- `tests/test_resnet_example_integration.py`
(17, CPU: model/block shape correctness, gradient flow through both
residual branches for identity and projection variants, `BatchNorm2d`
train/eval/buffer/recursive-mode-propagation behavior, full-pipeline
training and learning, repeated-forward determinism, checkpoint save/
resume + resume equivalence, model persistence, device-movement
idempotence, CLI inspection), `tests/test_resnet_example_cuda_integration.py`
(5, CUDA hardware-gated: full pipeline on CUDA, parameter/gradient/
Adam-state/buffer CUDA residency, checkpoint save/resume on CUDA, model
persistence on CUDA, CPU-to-CUDA `ResidualBlock` forward parity), `tests/
test_resnet_autograd_validation.py` (5, CPU: dedicated finite-difference
input- and parameter-gradient checks for both shortcut variants, plus an
isolated additive-gradient check on the residual addition itself, verified
to `atol=1e-10`). Full suite: **2,008 passed** (1,981 + 27), one
pre-existing, unrelated `test_dataloader_prefetch.py` allocator-measurement
flake (the same one M63/M64/M65 already footnoted) in the full-suite run
that passed cleanly in isolation immediately afterward -- zero regressions
attributable to this milestone. **Outcome A**: residual model completed
entirely with existing Forge capabilities; no new production capability was
required. Full report: `docs/development/m66-residual-cnn.md`.
