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
