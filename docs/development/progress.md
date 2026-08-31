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
