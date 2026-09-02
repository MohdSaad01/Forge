# Losses and Optimization (Milestone 4; CUDA-aware `SGD` as of Milestone 10; CUDA `MSELoss` as of Milestone 12)

## Package layout
```
forge/
    nn/
        loss.py        Loss, MSELoss, CrossEntropyLoss
    optim/
        optimizer.py    Optimizer (base)
        sgd.py           SGD
    tensor/tensor.py     new `.exp()` / `.log()` primitives (see below)
```
`forge.optim` is exposed as a submodule of `forge` (`forge.optim.SGD`), alongside `forge.nn`. Loss classes are exposed from `forge.nn` (`forge.nn.MSELoss`, `forge.nn.CrossEntropyLoss`), matching where `Linear`/`ReLU` already live.

## Loss abstraction
`Loss` (`forge/nn/loss.py`) is a small callable base class: `loss_fn(prediction, target)` delegates to `forward()`, mirroring `Module.__call__` -> `forward()`. It is deliberately **not** a `Module` subclass -- a loss owns no parameters and is not part of a model's module tree, matching the domain model's distinction between `Module` (composable, trainable, owns state) and `Loss` (stateless, computed each step). The base `forward()` raises `LossError`, the same "must implement forward()" pattern `Module` already uses for `ModuleError`.

Both built-in losses are implemented entirely from ordinary Tensor operations (`-`, `*`, `.sum()`, and the two primitives added by this milestone, `.exp()`/`.log()`). No loss computes a gradient by hand -- the existing autograd graph (`docs/architecture/autograd.md`) differentiates through the loss exactly as it would through any other Tensor expression.

### MSELoss
```text
MSE = mean((prediction - target)^2)
```
`prediction` and `target` must have **exactly** the same shape (no implicit broadcasting between them) -- a mismatch raises `LossError` before reaching a confusing broadcast/shape error deeper in the op. The mean is taken over every element of that shape: for a `(batch, features)` pair this averages over both batch and feature dimensions, matching the common per-element convention. Implementation: `((prediction - target) * (prediction - target)).sum() * scale`, entirely ordinary Tensor ops (`scale` is a `Tensor(1/n, dtype=prediction.dtype, device=prediction.device)` built explicitly, rather than multiplying by a bare Python float, so a CUDA `prediction` of either supported compute dtype still gets a matching-dtype operand for that last multiply -- see `docs/architecture/cuda-backend.md`'s **CUDA losses** section, Milestone 12). This composition needs only `-`, `*`, `.sum()` -- as of Milestone 12, every one of those already has a CUDA forward *and* backward implementation, so `MSELoss` runs on CUDA with no CUDA-specific code of its own.

### CrossEntropyLoss
```text
logits:  (batch_size, num_classes)   -- unnormalized scores
target:  (batch_size,)                -- integer class indices in [0, num_classes)
loss = -mean(log_softmax(logits)[i, target[i]])
```
Validates, in order: `logits.ndim == 2`; `target.shape == (batch_size,)`; `target` has an integer dtype; every target value is a valid class index. Each failure raises `LossError` with the specific mismatch.

**Numerical stability** uses the standard log-sum-exp trick:
```text
log_softmax(x) = (x - c) - log(sum(exp(x - c)))     where c = max(x, axis=1)
```
`c` is computed as a plain NumPy array from `logits.numpy()`, wrapped in a `Tensor` with `requires_grad=False`, and subtracted before exponentiating -- so `exp` never sees an argument larger than `0` and cannot overflow regardless of the input's scale. Treating `c` as a constant (not differentiating through how it was computed) is exact, not an approximation: the identity above holds for *any* `c`, so `c`'s own dependence on `x` contributes nothing to the correct gradient.

**Target selection** avoids adding a new gather/indexing primitive: targets are expanded into a one-hot NumPy array (also wrapped as a non-differentiable `Tensor`), multiplied elementwise against `log_softmax(logits)`, and summed over the class axis. This is exact (the one-hot row zeroes out every non-target class) and needs only ops Forge already has.

### New Tensor primitives: `.exp()` / `.log()`
The existing operation set (`+`, `-`, `*`, `@`, `.sum()`, `.reshape()`, `.relu()`) could not express a numerically stable cross-entropy, so this milestone adds two elementwise primitives following the exact `Tensor` -> `Backend` -> `autograd.Node` pattern `.relu()` established in Milestone 3:
- `Tensor.exp()`, backed by `Backend.exp`/`CPUBackend.exp` (`np.exp`). Backward: `grad_output * exp(x)` (reuses the forward result, no recomputation).
- `Tensor.log()`, backed by `Backend.log`/`CPUBackend.log` (`np.log`). Backward: `grad_output / x`.

No domain validation (`x > 0` for `log`) is added at the Tensor level -- exactly like the rest of the op set, invalid domains are the caller's responsibility. `CrossEntropyLoss` only ever calls `.log()` on a sum of exponentials, which is always `> 0`, so this is safe in the one place Forge currently uses it.

## Optimizer abstraction
`Optimizer` (`forge/optim/optimizer.py`) owns a flat `list[Parameter]`, built from whatever iterable is passed to its constructor -- typically `model.parameters()`, but never a model instance. This keeps the optimizer decoupled from `Module` internals, per the architecture's design rule that abstractions should enforce only the boundaries they need to.

Responsibility boundary (unchanged from the milestone spec):
```text
Autograd  -> computes gradients (Tensor.backward())
Optimizer -> consumes gradients (step()), clears them (zero_grad())
```
An optimizer never triggers a forward or backward pass, and never computes a gradient itself.

`zero_grad()` is defined once on the base class and delegates to `Parameter.zero_grad()` (inherited from `Tensor.zero_grad()`) for every owned parameter -- no gradient-clearing logic is duplicated between `Tensor` and `Optimizer`. The base `step()` raises `OptimizerError`, the same "must implement" pattern as `Module.forward()`/`Loss.forward()`.

### SGD
```text
parameter = parameter - learning_rate * gradient
```
`SGD(parameters, lr)` validates `lr` is a real, non-NaN, strictly positive number at construction, raising `OptimizerError` otherwise (`lr <= 0`, or a non-numeric/`bool`/`NaN` value). No momentum, weight decay, or learning-rate schedule -- explicitly out of scope for this milestone.

`step()` skips any parameter whose `.grad is None` (e.g. a parameter unused by the current forward pass) rather than treating it as an error -- consistent with autograd only accumulating `.grad` for tensors actually reached by `backward()`.

### Parameter mutation does not extend the autograd graph
`SGD.step()` calls `param._data = backend.sgd_step(param._data, param.grad._data, self.lr)`, where `backend = get_backend(param.device)` (Milestone 10) -- a direct, in-place storage mutation of the parameter's backing data, not a Tensor arithmetic expression (`param - lr * param.grad` would go through `Tensor.__sub__`/`__mul__`, allocate a new `Tensor`, and -- since `lr * param.grad` does not itself require grad but the result would still be freshly wrapped -- reassigning `param` to that new Tensor would replace the actual `Parameter` object model code and the optimizer both hold a reference to, breaking identity). This is the same reasoning that drove the original Milestone 4 design (`param._data -= lr * param.grad._data`, plain NumPy in-place arithmetic); Milestone 10 only changes *how* that in-place update is performed, not *why* it is in-place:
- **CPU**: `CPUBackend.sgd_step` still does `data -= lr * grad; return data` -- identical to the Milestone 4 behavior, `param._data` stays the same `np.ndarray` object.
- **CUDA**: `CUDABackend.sgd_step` launches one kernel (`cf_sgd_step`, `param[i] -= lr * grad[i]`) that mutates the existing `CUDAStorage` buffer in place and returns the same object -- no new `cudaMalloc`, no host round-trip. See `docs/architecture/cuda-backend.md`'s **CUDA `SGD.step()`** section.
- Never attaches a `grad_fn`: `param.grad_fn` stays `None` and `param.is_leaf` stays `True` after `step()`, on either device.
- Preserves the `Parameter` object's identity, so `model.fc1.weight` and the optimizer's stored reference stay the same object across every step.
- Matches the spec's framing directly: an optimizer update is a state change, not a differentiable model operation. `SGD` itself contains no CUDA-specific code -- `Backend.sgd_step` is the one dispatch point, matching every other operation's `Tensor -> Backend` boundary; there is no separate `CUDA_SGD` class.

## Gradient lifecycle
The expected training sequence, unchanged from the spec:
```python
optimizer.zero_grad()   # clear every Parameter's .grad from the previous step
output = model(x)       # forward pass, builds a fresh autograd graph
loss = loss_fn(output, target)
loss.backward()         # accumulates gradients into leaf Parameters
optimizer.step()        # in-place parameter update from .grad; no new graph
```
`zero_grad()` must run before `backward()` in a given step (not merely before `step()`), because gradients *accumulate* (`docs/architecture/autograd.md`) -- skipping it would silently sum the new step's gradient onto the previous step's.

## Known limitations
- SGD only: no momentum, Adam, RMSProp, weight decay, or learning-rate schedules.
- No training engine/`Trainer`, `DataLoader`, or dataset abstraction yet -- the training loop above is written by hand in this milestone.
- `CrossEntropyLoss` remains CPU-only -- it depends on `.exp()`/`.log()` (CUDA has no kernel for either) and an axis-wise `.sum()` (CUDA `sum()` supports only a full reduction). As of Milestone 12 this is a confirmed, deliberate deferral (not just an unexercised gap): `CrossEntropyLoss.forward()` rejects non-CPU logits explicitly with `LossError` rather than silently running part of the computation on CPU -- see `docs/architecture/cuda-backend.md`'s **CUDA losses** section. `MSELoss` (built only from `-`, `*`, `.sum()`) works on CUDA like any other differentiable Tensor expression, and is now exercised end-to-end through `forge.training.Trainer(device="cuda")` -- see `docs/architecture/training-engine.md`. As of Milestone 10, `SGD` is device-aware via `Backend.sgd_step()` -- see **Parameter mutation does not extend the autograd graph** above; no CUDA-specific optimizer class exists.
- `CrossEntropyLoss` supports exactly the `(batch_size, num_classes)` / `(batch_size,)` shape convention; no class weighting, label smoothing, or ignored-index support.
- `Tensor.log()` has no domain validation; calling it directly (outside `CrossEntropyLoss`'s controlled use) on non-positive values produces NumPy's usual `-inf`/`nan` rather than a Forge-level error.
