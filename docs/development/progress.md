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
