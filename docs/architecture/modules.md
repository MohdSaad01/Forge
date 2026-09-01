# Modules and Parameters (Milestone 3 + 9)

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

## Known limitations
- No layer whose forward behavior differs by `.training` (dropout, batch norm) yet; the mode plumbing exists for later milestones to use.
- No module serialization (`state_dict`-equivalent) yet.
- `forge.random` is a single global generator, not a per-module or thread-local RNG.
- No buffer concept (see **No buffers to move** above) -- `Module.to()` moves `Parameter`s only.
- `Module.to()` moves `Parameter`s, never a module's plain Python attributes -- an `in_features`-style config int, or any non-Tensor/non-Module attribute, is left exactly as constructed.
- A CUDA-resident model's forward pass must run inside `forge.no_grad()` (see `docs/architecture/cuda-backend.md`); a bare forward call raises `UnsupportedDeviceError`, since `Module.to()` preserves `requires_grad=True` and CUDA autograd is unsupported.

As of Milestone 4, an optimizer (`forge.optim.SGD`) exists and updates `Parameter` data from `.grad` -- see `docs/architecture/optimization.md`. As of Milestone 9, `Module.to(device)` moves a module tree's `Parameter`s between CPU and CUDA -- see **Device movement** above and `docs/architecture/cuda-backend.md`.
