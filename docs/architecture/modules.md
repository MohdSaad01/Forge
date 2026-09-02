# Modules and Parameters (Milestone 3 + 9; CUDA autograd boundary updated in Milestone 10; Sequential/Flatten/Dropout added in Milestone 16)

## Package layout
```
forge/
    nn/
        parameter.py    Parameter (Tensor subclass)
        module.py        Module (registration, discovery, train/eval, __call__)
        linear.py         Linear
        activation.py     ReLU
    random.py             process-global default RNG for deterministic init
```
`forge.nn` and `forge.random` are exposed as submodules of `forge` (`forge.nn.Linear`, `forge.random.seed(...)`).

## Parameter
`Parameter` is a thin `Tensor` subclass (`forge/nn/parameter.py`) that defaults `requires_grad=True`. It adds no numerical behavior: constructing one runs the same validation as `Tensor.__init__` (dtype must be `float32`/`float64` when `requires_grad=True`), and operations applied to it go through Tensor's ordinary differentiable ops. The result of an operation on a `Parameter` is a plain `Tensor`, not a `Parameter` -- only leaves representing owned model state are `Parameter`s, matching `Tensor._differentiable_wrap`, which always wraps results as `Tensor`.

```text
Parameter
   ↓ (is-a)
Tensor
   ↓
requires_grad=True by default
```

## Module
`Module` (`forge/nn/module.py`) owns two plain dicts, populated automatically by attribute assignment rather than requiring manual registration:
- `_parameters: dict[str, Parameter]`
- `_modules: dict[str, Module]`

`Module.__setattr__` is overridden to route `self.x = value`:
- `value` is a `Parameter` → stored in `_parameters["x"]`.
- `value` is a `Module` → stored in `_modules["x"]`.
- otherwise → a normal instance attribute (and any previous parameter/module registration under that name is removed, e.g. `self.bias = None` when a layer is constructed without a bias).

`Module.__getattr__` (only invoked when normal lookup fails) resolves `self.x` back out of `_parameters`/`_modules`, so registered parameters and child modules read exactly like ordinary attributes:
```python
self.fc1 = Linear(4, 8)   # registered as a child module
self.fc1.weight           # Parameter, resolved via __getattr__
```
A subclass that assigns a `Parameter`/`Module` before calling `super().__init__()` raises `ModuleError` -- `_parameters`/`_modules` don't exist yet, so this is caught explicitly rather than failing with a confusing `AttributeError`.

### Discovery
`named_parameters(prefix="")` walks `_parameters` then recurses into `_modules`, building dotted names (`fc1.weight`) as it descends. Each parameter is tracked by `id()` in a `seen` set threaded through the recursion, so a parameter reachable through more than one attribute path (intentional weight sharing) is yielded exactly once, at the first name it is found under. `parameters()` is `named_parameters()` with the names dropped. `named_modules()`/`modules()`/`children()`/`named_children()` follow the same shape for module (rather than parameter) discovery.

### Training / evaluation mode
`Module.__init__` sets `_training = True`. `train(mode=True)` sets `_training` on `self` and recurses into every child module; `eval()` is `train(False)`. `.training` exposes the current state. No M3 layer reads `.training` yet -- the state exists so a later layer (dropout, batch norm) has something to branch on.

### Invocation
`Module.__call__` invokes `self.forward(*args, **kwargs)`. The base `Module.forward` raises `ModuleError` rather than a bare `NotImplementedError`, keeping the "must implement forward()" mistake identifiable as a Forge-level configuration error.

## Linear
`forge/nn/linear.py`. `y = x @ weight + bias`, with `weight` shape `(in_features, out_features)` and `bias` shape `(out_features,)`. Because `Tensor.__matmul__` already supports both the 1D·2D and 2D·2D cases, `Linear.forward` needs no separate batching logic -- a single sample `(in_features,)` and a batch `(batch, in_features)` both work through the same matmul.

### Initialization
Both `weight` and `bias` are drawn from `Uniform(-1/sqrt(in_features), 1/sqrt(in_features))`. Each output is a sum of `in_features` terms of variance `~1/(3 * in_features)`; scaling the draw by `1/sqrt(in_features)` keeps that sum's variance `O(1)` regardless of layer width, so activations don't blow up or vanish purely as a function of `in_features`. This is the same bound PyTorch's default `nn.Linear` init reduces to. Draws come from `forge.random.default_generator()` unless a `numpy.random.Generator` is passed explicitly via `Linear(..., generator=...)`.

### `forge.random`
A minimal process-global `numpy.random.Generator` (`forge/random.py`), not a general random-number framework. `forge.random.seed(value)` reseeds it, making `Linear` construction (and anything else that draws from `default_generator()`) deterministic for a given seed.

### Validation
`Linear(in_features, out_features)` rejects non-positive dimensions at construction (`ShapeMismatchError`). `forward(x)` rejects `x.ndim not in (1, 2)` and `x.shape[-1] != in_features`, both as `ShapeMismatchError`, before the mismatch would otherwise surface as an opaque NumPy matmul error.

## ReLU
`forge/nn/activation.py`. `ReLU.forward` delegates entirely to a new Tensor-level primitive, `Tensor.relu()` (`forge/tensor/tensor.py`), backed by `Backend.relu`/`CPUBackend.relu` (`np.maximum(a, 0)`) -- following the same Tensor → Backend → autograd `Node` pattern as every other differentiable op (`+`, `-`, `*`, `@`, `.sum()`, `.reshape()`). Its backward rule is `grad_output` where the input was `> 0`, else `0` (strict inequality, so the subgradient at exactly `0` is `0`, matching common ReLU convention). No gradient math lives in the `ReLU` module itself -- it is ordinary autograd through `Tensor.relu()`.

## Device movement: `Module.to(device)` (Milestone 9)
`Module.to(device)` (`forge/nn/module.py`) recursively moves every
`Parameter` owned by this module tree to `device`:
```python
model = MLP()             # a Linear -> ReLU -> Linear model, all CPU
model.to("cuda")           # every Parameter is now CUDA-resident
```

### Return semantics
`to()` **mutates `self` and returns it**, the same convention `train()`/
`eval()` already use in this class -- not a copy-construct-and-return
convention. This was chosen over building a full recursive-copy Module tree
because Forge's `Module` has no general "clone a module tree" mechanism
(persistence's `save`/`load` round-trip is the closest thing, and that goes
through the module registry, not a generic copy), and a mutate-in-place
convention keeps every existing reference to the model (an optimizer's
`model.parameters()` list, a `Trainer`'s stored `model`, a user's own
variable) valid after the call without any of them needing to be
reassigned. This is documented here as the one sanctioned convention --
`to()` never has ambiguous "sometimes copies, sometimes mutates" behavior.

### Parameter movement mechanism
`to()` walks `self.modules()` (this module plus every descendant) and, for
each module's own directly-owned `_parameters`, calls a new private Tensor
primitive, `Tensor._move_storage_(device)`:
```python
def to(self, device):
    target = Device.parse(device)
    for module in self.modules():
        for param in module._parameters.values():
            param._move_storage_(target)
    return self
```
`_move_storage_()` is the in-place counterpart to the existing
`Tensor.to()` (`docs/architecture/cuda-backend.md`): `Tensor.to()` has
*value* semantics -- it always returns a **fresh** leaf tensor with
`requires_grad=False`, leaving the original untouched, which is exactly
wrong for moving a `Parameter` in place (it would silently replace the
`Parameter` object every other part of the program is holding a reference
to, and would silently drop `requires_grad`). `_move_storage_()` instead
mutates `self._data`/`self._device` directly, via the same
`Backend.to_numpy()` → `Backend.from_array()` host round-trip `Tensor.to()`
uses internally, and leaves everything else on the object alone:
- **Identity is preserved**: `model.fc1.weight is weight_before_to` is
  `True` after `model.to("cuda")`. No second CUDA-specific `Parameter`
  class exists, and none is created here -- a `Parameter` remains exactly
  what it was, just backed by different device storage.
- **`requires_grad` is preserved**: unlike `Tensor.to()`, a moved
  `Parameter`'s `requires_grad` flag is untouched (stays `True` for an
  ordinary trainable parameter). See
  `docs/architecture/cuda-backend.md`'s **Interaction with
  `Module.to("cuda")`** section for the consequence this has for running a
  CUDA model's forward pass (it must run inside `forge.no_grad()`).
- **Leaf status is preserved**: `is_leaf` stays `True` and `grad_fn` stays
  `None` -- no autograd graph is built or altered by a device move.
- **`.grad` is cleared**: any previously accumulated gradient was computed
  for the parameter's *previous* device/data and CUDA has no backward
  support to recompute it there, so leaving a stale, now-mismatched
  gradient in place would be misleading. A fresh gradient is expected from
  the next `backward()` call (which, on a CUDA-resident model, must itself
  happen on a CPU-resident copy -- CUDA backward is unsupported, see
  `docs/architecture/cuda-backend.md`).
- **Shape and dtype are unaffected** -- the round-trip preserves both
  exactly, since it is the identical mechanism `Tensor.to()` already uses
  for correctness.

Because `_move_storage_()` is a no-op when the target device already
matches (mirroring `Tensor.to()`'s same-device no-op), calling `model.to(x)`
twice in a row, or calling it when a `Parameter` is shared by two attribute
paths (`docs/architecture/modules.md`'s existing "intentional weight
sharing" case), is always safe and idempotent.

### No buffers to move
Forge has no "buffer" concept (non-trainable module state, e.g. batch-norm
running statistics) as of this milestone -- only `Parameter`s exist as
per-module tensor state, and `to()` moves exactly those. A future milestone
introducing buffers would need to extend `to()`'s walk accordingly; nothing
about the current design assumes buffers don't exist, but nothing handles
them yet either.

### Device consistency and mismatch
`Module.to()` never touches non-`Parameter` inputs -- calling a CUDA-moved
model with a CPU tensor (or vice versa) is not specially detected by
`Module`/`Linear`/`ReLU` at all. It fails exactly the same way any other
Tensor-level device mismatch already fails: `Linear.forward`'s `x @
self.weight` raises `UnsupportedDeviceError` from `Tensor.__matmul__`'s
existing device-equality check, the same one that has always guarded every
binary Tensor operation. No new device-consistency logic was added to
`Linear`/`ReLU` for this milestone -- see
`docs/architecture/cuda-backend.md` for why "no CUDA-specific code inside
`Linear`" was achievable at all (the CUDA backend's new row-broadcast `add`
support is what makes a *batched* `Linear` forward -- `x @ weight + bias`
-- work unmodified).

### Device introspection: `Module.device`
```python
model.device   # -> Device('cpu') or Device('cuda') or None
```
Returns the single `Device` shared by every `Parameter` reachable from this
module (via `parameters()`), or `None` if the module tree owns no
`Parameter`s at all (e.g. a bare `ReLU()`) -- Forge does not assume every
`Module` has one device merely because most do. If `Parameters` are found
on more than one device, `Module.device` raises `ModuleError` rather than
guessing which one to report. Normal usage (constructing a model on one
device, or calling `.to(device)`) always leaves a tree with a single
coherent device; this is a chosen **single-device-per-module-tree
invariant**, enforced at introspection time (not at every `__setattr__`,
which would be a much larger change for comparatively little benefit) --
a tree can still be pushed into an inconsistent state by reassigning one
child's `Parameter` directly (`model.fc2.weight = Parameter(..., device=
"cpu")`), and `.device` is exactly the check that catches it. See
`tests/test_module_cuda.py::test_module_device_raises_for_manually_mixed_parameters`.

## Autograd integration
No module in this milestone computes a gradient directly. `Linear.forward` and `ReLU.forward` are built from ordinary Tensor operations (`@`, `+`, `.relu()`); the existing autograd graph (`docs/architecture/autograd.md`) tracks and backpropagates through them exactly as it would for hand-written Tensor code. `Module`/`Parameter` are a composition and discovery layer over that graph, not a second gradient mechanism.

```text
Module.forward
   ↓
Tensor operations (@, +, .relu())
   ↓
Autograd graph (Node, grad_fn)
   ↓
Tensor.backward()
   ↓
Parameter.grad (Parameter is a leaf Tensor)
```

## Sequential, Flatten, Dropout (Milestone 16)

### `Sequential`
`forge/nn/container.py`. An ordered `Module` container: `Sequential(m0, m1, ...)` registers each argument via ordinary `Module.__setattr__` under the deterministic names `"0"`, `"1"`, `"2"`, ... in construction order, then `forward(x)` applies them in that order (`for _, child in self.named_children(): x = child(x)`). Because `Module._modules` is already an insertion-ordered `dict`, every existing traversal API (`named_children`/`children`/`named_modules`/`modules`/`parameters`/`named_parameters`, `train()`/`eval()`, `Module.to()`, `Module.device`) already walks a `Sequential` tree correctly with **no overrides** beyond `forward()` itself -- this is the whole point of building it directly on `Module` rather than a separate container abstraction.

Every constructor argument must be a `Module`; a non-`Module` argument raises `ModuleError` immediately (before anything is registered). `Sequential()` with zero modules is explicitly supported as the identity function (`forward(x)` returns `x` unchanged), matching the common convention (PyTorch) that an empty container composes to a no-op.

### `Flatten`
`forge/nn/flatten.py`. `Flatten(start_dim=1, end_dim=-1)` collapses dims `[start_dim, end_dim]` (inclusive, negative-indexed like NumPy) into one, resolved against the input's actual `ndim` on every `forward()` call. The default collapses an `(N, C, H, W)` `Conv2d`/`MaxPool2d` output down to `(N, C*H*W)` ahead of a `Linear` layer. Built entirely on the existing `Tensor.reshape` (already differentiable, already real on CPU and CUDA -- `docs/architecture/cuda-backend.md`), so `Flatten` has no parameters and no backward rule of its own, exactly like `ReLU` through `Tensor.relu()`. Out-of-range or `start_dim > end_dim` configurations raise `ShapeMismatchError` at `forward()` time (the specific input shape is needed to resolve negative dims, so this cannot be validated at construction).

### `Dropout`
`forge/nn/dropout.py`. `Dropout(p=0.5, generator=None)` is Forge's first `.training`-dependent stochastic layer: `0 <= p < 1` is validated at construction (`ModuleError` otherwise). During training, `forward(x)` computes `x * mask` where `mask = x.dropout_mask(p, rng)` (`Tensor.dropout_mask`, `forge/tensor/tensor.py`) -- a fresh `requires_grad=False` leaf whose values are already `1/(1-p)` (kept) or `0` (dropped), i.e. **inverted dropout**: the rescaling happens at training time, so `eval()`'s forward pass is a plain, unscaled identity. During evaluation, `forward()` returns `x` itself unchanged -- no new graph node is inserted, so gradients flow through exactly as if `Dropout` were absent.

**No Dropout-specific backward rule exists.** `x * mask` is ordinary `Tensor.__mul__`, whose existing `mul_backward` autograd rule already gives `grad_input = grad_output * mask` -- exactly the required backward formula -- and the *same* `mask` object computed during forward is what that backward closure captures and reuses (see `Tensor._binary_op`'s closure over `b_data`); backward never redraws a mask. This is why Dropout integrates with autograd, CPU, and CUDA with essentially zero Dropout-specific code at the Tensor/autograd layer -- all the real work is in `Backend.dropout_mask` (mask generation), described next.

`Dropout` reads its own `self.training` -- no new state mechanism; **Module state** above is unchanged.

#### Randomness (`Backend.dropout_mask`)
A new `Backend` method, `dropout_mask(a, p, rng) -> mask`, added alongside `conv2d`/`max_pool2d` in the same ABC (`forge/backend/base.py`). `a` is read only for its shape/dtype; `rng` is a live `numpy.random.Generator` -- `forge.random.default_generator()` by default (fetched **fresh on every `forward()` call**, unlike `Linear`/`Conv2d`'s one-time construction-time snapshot, so a `forge.random.seed(...)` call governs every subsequent Dropout draw across an entire training run), or an explicit one passed via `Dropout(..., generator=...)`. No second global RNG is introduced.

- **CPU** (`CPUBackend.dropout_mask`): draws directly from `rng.random(a.shape)`, thresholds against `p`, scales by `1/(1-p)` -- an ordinary NumPy computation, exactly like every other `CPUBackend` method.
- **CUDA** (`CUDABackend.dropout_mask`): draws **exactly one integer seed** from `rng` (a cheap host-side scalar draw -- not per-element randomness), then launches `cf_dropout_mask_{f32,f64}`, a real CUDA kernel that generates every element's Bernoulli draw independently on-device from a stateless hash of `(seed, element_index)` (`kernels.cu`'s **Dropout mask** section, SplitMix64-based -- see `docs/architecture/cuda-backend.md`'s **CUDA Dropout** section for the full mechanism and no-CPU-fallback rationale).

### Known limitations (Sequential/Flatten/Dropout)
- `Sequential` has no `__getitem__`/`__len__`/`__iter__` convenience accessors -- only the `Module` traversal API (`named_children()`, `children()`, etc.) is available, per the milestone's "do not silently expand scope" constraint.
- `Sequential`'s persistence needs one small registry-level accommodation beyond the generic tree walk: `forge/serialization/model.py`'s `_build_load_node` requires a freshly `from_config()`-constructed module to already have a child under every name the file is about to attach (a fixed-shape invariant that holds for free when config alone determines structure, e.g. `Linear`'s `in_features`/`out_features`). `Sequential`'s child *count* is data, not config, so its registered `from_config` builds that many placeholder `Module()` children up front (`n_children` is the one extra config field `get_config` reports); the existing attach loop then overwrites each placeholder with its real child, unmodified. See `forge/serialization/registry.py`'s `"Sequential"` registration comment. No change to the generic save/load algorithm or file format was needed.
- `Flatten`'s `start_dim`/`end_dim` are resolved against the input's `ndim` on every `forward()` call rather than fixed at construction -- correct for the fixed-rank `(N, C, H, W)` inputs every Forge layer that feeds one actually produces, but means a shape error only ever surfaces at `forward()` time, not construction time.
- No layer whose forward behavior differs by `.training` existed before Milestone 16; `Dropout` is now the first. BatchNorm/LayerNorm remain out of scope (see the milestone's explicit non-goals).
- `forge.random` remains a single global generator, not a per-module or thread-local RNG; `Dropout(generator=...)` is the escape hatch for an independent stream.

## Known limitations
- No module serialization gap remains for the module types this milestone covers (see **Sequential/Flatten/Dropout** limitations above for the one Sequential-specific accommodation).
- No buffer concept (see **No buffers to move** above) -- `Module.to()` moves `Parameter`s only.
- `Module.to()` moves `Parameter`s, never a module's plain Python attributes -- an `in_features`-style config int, or any non-Tensor/non-Module attribute, is left exactly as constructed.
- As of Milestone 10, a CUDA-resident model's forward pass no longer needs `forge.no_grad()`: since `Module.to()` preserves `requires_grad=True` and CUDA autograd is now supported for `Linear`/`ReLU`'s operations, a bare forward call succeeds and builds a real graph, and `backward()` on the result runs on CUDA -- see `docs/architecture/cuda-backend.md`'s **CUDA autograd** section. `no_grad()` remains available (and still suspends graph construction on CUDA exactly as on CPU) for inference-only forward passes.

As of Milestone 4, an optimizer (`forge.optim.SGD`) exists and updates `Parameter` data from `.grad` -- see `docs/architecture/optimization.md`. As of Milestone 9, `Module.to(device)` moves a module tree's `Parameter`s between CPU and CUDA -- see **Device movement** above and `docs/architecture/cuda-backend.md`. As of Milestone 16, `Sequential`/`Flatten`/`Dropout` compose all of this with no new subsystem -- see **Sequential, Flatten, Dropout** above.
