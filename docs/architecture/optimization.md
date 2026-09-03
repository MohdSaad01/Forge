# Losses and Optimization (Milestone 4; CUDA-aware `SGD` as of Milestone 10; CUDA `MSELoss` as of Milestone 12; CUDA `CrossEntropyLoss` as of Milestone 14; `Adam` as of Milestone 17; CUDA Conv2d-backward performance work in Milestone 21; allocation profiling in Milestone 24; CUDA caching allocator in Milestone 25)

## Package layout
```
forge/
    nn/
        loss.py        Loss, MSELoss, CrossEntropyLoss
    optim/
        optimizer.py    Optimizer (base)
        sgd.py           SGD
        adam.py          Adam (Milestone 17)
    tensor/tensor.py     new `.exp()` / `.log()` primitives (see below)
```
`forge.optim` is exposed as a submodule of `forge` (`forge.optim.SGD`, `forge.optim.Adam`), alongside `forge.nn`. Loss classes are exposed from `forge.nn` (`forge.nn.MSELoss`, `forge.nn.CrossEntropyLoss`), matching where `Linear`/`ReLU` already live.

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
`c` is computed by `Backend.max_axis1(logits._data)` -- CPU: a plain `np.max`; CUDA (Milestone 14): a dedicated kernel, so a CUDA `logits` never has to leave the device to compute its shift. Either way the result is wrapped in a `Tensor` with `requires_grad=False` and subtracted before exponentiating -- so `exp` never sees an argument larger than `0` and cannot overflow regardless of the input's scale. Treating `c` as a constant (not differentiating through how it was computed) is exact, not an approximation: the identity above holds for *any* `c`, so `c`'s own dependence on `x` contributes nothing to the correct gradient.

**Target selection** avoids adding a new gather/indexing primitive: targets are expanded into a one-hot NumPy array (also wrapped as a non-differentiable `Tensor`, built on `logits`'s own device), multiplied elementwise against `log_softmax(logits)`, and summed over the class axis. This is exact (the one-hot row zeroes out every non-target class) and needs only ops Forge already has.

### New Tensor primitives: `.exp()` / `.log()`
The existing operation set (`+`, `-`, `*`, `@`, `.sum()`, `.reshape()`, `.relu()`) could not express a numerically stable cross-entropy, so this milestone adds two elementwise primitives following the exact `Tensor` -> `Backend` -> `autograd.Node` pattern `.relu()` established in Milestone 3:
- `Tensor.exp()`, backed by `Backend.exp` (CPU: `np.exp`; CUDA, Milestone 14: `cf_exp_{f32,f64}`). Backward dispatches to `Backend.exp_backward` (`grad_output * exp(x)`, reusing the forward result -- CPU: NumPy multiply; CUDA: `cf_exp_backward_{f32,f64}`, since a `CUDAStorage` has no `__mul__` of its own for the closure to fall back on).
- `Tensor.log()`, backed by `Backend.log` (CPU: `np.log`; CUDA: `cf_log_{f32,f64}`). Backward dispatches to `Backend.log_backward` (`grad_output / x`; CUDA: `cf_log_backward_{f32,f64}`).

No domain validation (`x > 0` for `log`) is added at the Tensor level -- exactly like the rest of the op set, invalid domains are the caller's responsibility. `CrossEntropyLoss` only ever calls `.log()` on a sum of exponentials, which is always `> 0`, so this is safe in the one place Forge currently uses it. See `docs/architecture/cuda-backend.md`'s **CUDA CrossEntropyLoss** section for the full Milestone 14 CUDA primitive set (`exp`/`log`, axis=1 `sum`, and the column-broadcast `sub` the shift/log-probability steps need).

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

### Adam
```text
m_t   = beta1 * m_(t-1) + (1 - beta1) * g_t
v_t   = beta2 * v_(t-1) + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```
`Adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`
(`forge/optim/adam.py`) validates every hyperparameter at construction,
raising `OptimizerError` for a non-positive `lr`, a `beta1`/`beta2` outside
`[0, 1)`, a non-positive `eps`, or a negative `weight_decay` -- matching
`SGD`'s "validate at construction, never silently clamp" convention.

**`weight_decay` semantics.** When nonzero, it is classic L2 regularization
folded directly into the gradient *before* the moment updates:
`g_t <- g_t + weight_decay * theta`. This is the original Adam paper's
semantics, not AdamW's decoupled decay (which instead subtracts `lr *
weight_decay * theta` from the parameter directly, outside the moment
estimates and independent of `lr`'s interaction with `eps`). Forge
implements only the former under the name `Adam` -- AdamW is explicitly out
of scope for this milestone (see the milestone's Non-Goals).

**Backend boundary.** All numerical work -- both moment updates, bias
correction, and the parameter step -- happens in one call to
`Backend.adam_step(data, grad, m, v, lr, beta1, beta2, eps, weight_decay,
step)` (`forge/backend/base.py`), the same `Tensor -> Backend` boundary
every other Forge operation uses. `Adam` itself contains no backend-specific
code: `CPUBackend.adam_step` is plain in-place NumPy arithmetic;
`CUDABackend.adam_step` launches one kernel (`cf_adam_step_{f32,f64}`,
`kernels.cu`) that performs the entire update against the existing
`CUDAStorage` buffers in place. See `docs/architecture/cuda-backend.md`'s
**CUDA Adam** section for the CUDA-specific writeup.

**Optimizer state.** `Adam.state` maps `Parameter -> _AdamState` (`m`, `v`,
and a per-parameter step count). `Parameter` defines no custom
`__eq__`/`__hash__`, so this is ordinary Python object-identity `dict`
keying -- state follows the same `Parameter` object regardless of which
`Module` hierarchy currently references it or what attribute name it is
assigned to, per the spec's "identity, not name" requirement. State is
allocated lazily: a parameter with `.grad is None` is skipped entirely (no
state is created for it, matching `SGD`), and the first `step()` call that
does see a gradient allocates `m`/`v` as zeros matching that parameter's
shape, dtype, and device exactly (`Backend.from_array(np.zeros(...),
dtype)` -- a real `cudaMalloc` + host-to-device transfer of zeros on CUDA,
the same construction path any new CUDA tensor already uses, never a NumPy
array standing in for CUDA state).

Before using existing state, `step()` validates the incoming gradient's
shape/dtype/device against the parameter (`OptimizerError` on any
mismatch) and the existing state's shape/dtype against the parameter's
current shape/dtype -- defensive checks that can't currently be triggered by
ordinary Forge usage (a `Parameter`'s shape/dtype never change after
construction) but keep the invariant explicit rather than assumed.

**Device transfer after optimizer state exists.** `Module.to(device)` moves
a `Parameter`'s storage in place but has no knowledge of any optimizer --
by design, `Optimizer` stays decoupled from `Module` internals (see above).
Adam therefore does not migrate `m`/`v` automatically (**Policy A** from the
milestone spec): if `step()` finds existing state whose recorded device no
longer matches the parameter's current device, it raises `OptimizerError`
rather than silently pairing, say, a CUDA parameter with CPU-resident
moment buffers. `Adam.state` is itself the explicit migration mechanism --
no separate API was needed: clearing the stale entry (`del
optimizer.state[param]`, or `optimizer.state.clear()`) lets the next
`step()` lazily reinitialize fresh, zeroed state on the parameter's current
device. This deliberately restarts that parameter's moment estimates and
step count rather than guessing how to transfer them across devices. See
`docs/architecture/cuda-backend.md`'s **CUDA Adam** section for the
hardware-verified version of this behavior.

**Optimizer state is never model-persisted.** `save_model()`/`load_model()`
(`forge/serialization/model.py`) walk only the `Module` tree's registered
`Parameter`s; `Adam.state` lives entirely outside that tree (on the
`Optimizer` instance, keyed by `Parameter` identity), so it was never at
risk of being written to a model archive and required no persistence-layer
change. Optimizer-state/training-resume checkpointing itself remains out of
scope (unchanged from Milestone 7/13) -- `Adam.state`'s design (state keyed
cleanly by parameter identity, holding plain backend-storage-shaped values)
is intended to make that a future addition rather than a redesign, per the
milestone spec, but M17 does not implement it.

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

## Performance optimization (Milestone 21)
Milestone 21 profiled the actual M20 CNN's CUDA training step (see
`docs/performance/benchmarking.md`'s **Milestone 21** section for the full
methodology and numbers) and found `Adam.step()` itself was never a
meaningful contributor: an isolated CUDA `Adam.step()` call is well under a
millisecond even at 262,144 parameters (`benchmarks/ops_bench.py`'s
`adam_step` entries), and the MNIST workload profile attributes only
~3-4% of total CUDA training-step time to the `optimizer` phase. **`Adam`
itself was left unchanged** -- no optimization was justified here, matching
the milestone's "if Adam is negligible, leave it alone" instruction. The
dominant CUDA cost was `Conv2d`'s *backward* kernels (`conv2d_backward`),
not the optimizer -- see `docs/architecture/cuda-backend.md`'s **CUDA
Conv2d backward: weight/bias optimization (Milestone 21)** section for that
optimization's full writeup (the change lives entirely in
`forge/backend/cuda/kernels.cu`, not in this document's `nn.optim`
package).

## Allocation behavior (Milestone 24)
Milestone 24's allocation-profiling pass (`docs/architecture/cuda-memory-
allocator.md`) confirms, from a real CUDA allocation trace rather than the
timing-only observation above, that `Adam.step()`/`SGD.step()` allocate
**zero** new CUDA memory once optimizer state exists: profiling 30 steady-
state `adam_step`/`sgd_step` calls each (`benchmarks/alloc_profile.py`)
recorded no "alloc" events in either category. This is the allocation-level
counterpart to `tests/test_cuda_memory.py::
test_adam_first_step_allocates_persistent_state_subsequent_steps_do_not_grow`
(state is allocated once, lazily, on the first step with a real gradient,
and reused in place forever after) -- Milestone 24 adds the confirmation
that this holds for allocation *traffic*, not just net byte counts.

Unchanged by Milestone 25's caching allocator: `Adam`'s `m`/`v` (and every
`Parameter`) stay allocated for the optimizer's/module's entire lifetime, so
their underlying blocks are never released to the allocator's cache at all
(`allocator.release()` only ever runs from `CUDAStorage.__del__`, which only
runs once nothing references that storage) -- `docs/architecture/cuda-
memory-allocator.md`'s **Implementation (Milestone 25)** section confirms
`memory_stats().allocated_bytes` (the persistent footprint) is
byte-for-byte identical before and after caching was introduced.

## Known limitations
- `SGD` and `Adam` only (as of Milestone 17): no RMSProp, Adagrad, AdamW, learning-rate schedules, gradient clipping, parameter groups, mixed precision, or fused/multi-GPU optimizers -- all explicit Milestone 17 non-goals.
- No training engine/`Trainer`, `DataLoader`, or dataset abstraction yet -- the training loop above is written by hand in this milestone.
- `CrossEntropyLoss` works on CUDA as of Milestone 14 -- see `docs/architecture/cuda-backend.md`'s **CUDA CrossEntropyLoss** section for the primitives that made this possible (`exp`/`log` kernels, an axis=1 `sum`, and a column-broadcast `sub`) and the device-validation/no-fallback guarantees that carry over from `MSELoss`. `MSELoss` (built only from `-`, `*`, `.sum()`) has worked on CUDA since Milestone 12 like any other differentiable Tensor expression; both are now exercised end-to-end through `forge.training.Trainer(device="cuda")` -- see `docs/architecture/training-engine.md`. As of Milestone 10, `SGD` is device-aware via `Backend.sgd_step()` -- see **Parameter mutation does not extend the autograd graph** above; as of Milestone 17, `Adam` is device-aware via `Backend.adam_step()` the same way -- see **Adam** above and `docs/architecture/cuda-backend.md`'s **CUDA Adam** section.
- `CrossEntropyLoss` supports exactly the `(batch_size, num_classes)` / `(batch_size,)` shape convention; no class weighting, label smoothing, or ignored-index support.
- `Tensor.log()` has no domain validation; calling it directly (outside `CrossEntropyLoss`'s controlled use) on non-positive values produces NumPy's usual `-inf`/`nan` rather than a Forge-level error.
- `Adam`'s optimizer state (`m`, `v`, step count) is never persisted or checkpointed -- `save_model()` writes model (`Module`/`Parameter`) state only, unchanged from Milestone 7. Resuming training with matching Adam state after a save/load round trip is not supported; a freshly constructed `Adam` after `load_model()` starts with empty state (bias correction restarts from step 1), same as any newly constructed optimizer.
- `Adam` does not migrate optimizer state across a `Module.to(device)` call (Policy A -- see **Adam** above); `step()` raises `OptimizerError` if it detects a parameter that moved device after state was created for it, rather than guessing how to transfer `m`/`v`.
