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
