# Modules and Parameters (Milestone 3)

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

As of Milestone 4, an optimizer (`forge.optim.SGD`) exists and updates `Parameter` data from `.grad` -- see `docs/architecture/optimization.md`.
