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
