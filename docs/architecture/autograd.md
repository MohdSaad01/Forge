# Forge Autograd (Milestone 2; backend-aware as of Milestone 10)

## Package layout
```
forge/
    autograd/
        engine.py     Node (grad_fn), topological traversal, run_backward
    tensor/tensor.py  requires_grad, grad, backward(), zero_grad()
    backend/
        base.py        Backend ABC -- forward ops + *_backward ops + sgd_step
        cpu.py          CPUBackend's backward math (NumPy)
        cuda/backend.py CUDABackend's backward math (real CUDA kernels)
```

`forge.autograd` has no dependency on `forge.tensor`; it operates on
tensor-like objects via duck typing (`is_leaf`, `requires_grad`, `device`,
`_grad_fn`, `_accumulate_grad`). `Tensor` depends on `forge.autograd`, not
the reverse, so there is no import cycle and the graph engine stays testable
on its own.

As of Milestone 10, the per-operation backward *math* no longer lives in
`forge/autograd/` at all -- Milestone 2's `forge/autograd/functions.py` (pure
NumPy broadcast-reduction/matmul/sum helpers) has been removed, and its
content absorbed into `CPUBackend`'s own `*_backward` methods (see
**Backend-aware backward dispatch** below). `forge/autograd/engine.py` is
now purely the *graph* engine: topological ordering, `Node`/`grad_fn`
bookkeeping, `no_grad`, and combining gradient contributions from multiple
consumers -- it performs no numerical work of its own and holds no
op-specific formulas, matching this milestone's "do not blindly duplicate
the entire autograd engine" constraint.

## requires_grad
`Tensor(data, requires_grad=True)` marks a tensor for gradient tracking.
Only floating-point tensors (`float32`/`float64`) may set `requires_grad`;
requesting it on an int/bool tensor raises `GradientStateError`.

A tensor produced by a differentiable operation requires grad if **any**
input to that operation requires grad, matching the usual autodiff rule
that gradient tracking is contagious forward through the graph.

## Computation graph
Each differentiable operation (`+`, `-`, `*`, `@`, `.sum()`, `.reshape()`,
`.relu()`) that has at least one grad-requiring input attaches a `Node` to
its output tensor as `output.grad_fn`. A `Node` holds:
- `inputs`: the operation's input tensors (not copies of their data).
- `backward_fn`: a closure that maps an upstream gradient to a tuple of
  gradients, one per input, given the small amount of forward-pass state
  the operation needs (shapes, or in matmul's/mul's case the input arrays
  themselves — sum and reshape need no saved data beyond shape metadata).
  As of Milestone 10, `backward_fn` is a thin closure over a *backend*
  method (see below) rather than inline NumPy math — see **Backend-aware
  backward dispatch**.

## Backend-aware backward dispatch (Milestone 10)
Every differentiable `Tensor` operation's `backward_fn` closure calls
straight into the same backend the forward call used, via one new method
per operation on the `Backend` ABC (`forge/backend/base.py`):
`add_backward`, `sub_backward`, `mul_backward`, `matmul_backward`,
`sum_backward`, `reshape_backward`, `relu_backward` (`.exp()`/`.log()` are
CPU-only in both forward and backward — see **Unsupported CUDA
operations** below — so they keep their original inline NumPy closures
unchanged; adding backend dispatch for two ops with no second
implementation would be pure overhead).

```text
Tensor.__add__ / __sub__ / __mul__ / __matmul__ / .sum() / .reshape() / .relu()
   |
   v
backend = get_backend(self.device)          # forward call, e.g. backend.add(...)
   |
   v
backward_fn = lambda grad_output: backend.add_backward(grad_output, a, b)
   |
   v
Node(inputs, backward_fn, name)             # attached as the result's grad_fn
```

`a`/`b`/`grad_output` inside a `*_backward` method are always raw backend
storage — a `numpy.ndarray` on `CPUBackend`, a `CUDAStorage` on
`CUDABackend` — the exact same objects the forward call itself received and
produced. Neither `Tensor` nor `forge/autograd/engine.py` ever branches on
which backend it is; the branching lives entirely inside each backend's own
`*_backward` implementation:
- **`CPUBackend`** (`forge/backend/cpu.py`) implements the four elementwise/
  matmul/sum backward rules directly in NumPy — the exact formulas
  Milestone 2's `forge/autograd/functions.py` used, now living as private
  module-level helpers (`_reduce_grad_to_shape`) plus methods on the class
  itself.
- **`CUDABackend`** (`forge/backend/cuda/backend.py`) implements the same
  seven methods by launching real CUDA kernels (`forge/backend/cuda/
  kernels.cu`'s "backward-only kernels" section) and composing existing
  forward kernels (`matmul`, `reshape`, the row-broadcast `mul`) — see
  `docs/architecture/cuda-backend.md`'s **CUDA autograd** section for the
  kernel-level detail.

This is the concrete realization of the milestone's target architecture:
```text
Autograd Engine
      |
      +-- Node
           |
           +-- backward_fn (a closure over one Backend method)
                    |
             backend dispatch (get_backend(tensor.device))
               +----+----+
               v         v
             CPU       CUDA
```
Only seven methods needed backend dispatch — not a second autograd engine,
and not a rewrite of `Node`/`run_backward`'s graph-traversal logic, matching
the "do not blindly duplicate the entire autograd engine" constraint.

### Gradient accumulation is backend-dispatched too
When two consumers contribute a gradient to the same tensor,
`run_backward` (`forge/autograd/engine.py`) must combine them. A bare `+`
does not work generically: a `CUDAStorage` has no `__add__`. Instead,
`run_backward` calls `get_backend(inp.device).add(existing, new)` — reusing
the ordinary forward `add` kernel rather than introducing a second
accumulation mechanism. Because gradient contributions to the same tensor
always share that tensor's own shape, this is always the *exact-shape*
branch of `add` (never the row-broadcast case), so it needs no special
CUDA-kernel support beyond what forward `add` already has.
`Tensor._accumulate_grad` (a leaf's own `.grad` update) follows the same
pattern via `backend.add`.

A tensor is a **leaf** (`is_leaf == True`) when it was not produced by a
tracked operation — either it was constructed directly, or none of an
operation's inputs required grad. Only leaves accumulate `.grad`; the
gradient of a non-leaf is computed and propagated to its inputs but not
retained (matching common autodiff practice — retaining every intermediate
gradient is unnecessary for the optimizer this graph feeds).

## Backward entry point
```python
loss.backward()          # loss must be scalar (shape == ())
y.backward(grad_tensor)  # required for a non-scalar y; grad_tensor.shape == y.shape
```
`backward()` never invents a gradient for a non-scalar output — omitting
`gradient` on a non-scalar tensor raises `GradientStateError`. A supplied
gradient with the wrong shape raises `ShapeMismatchError`. Calling
`backward()` on a tensor that does not require grad raises
`GradientStateError`.

Internally, `run_backward` builds a dependency-ordered (topological) list of
every tensor reachable from the root via `grad_fn.inputs`, then walks it in
reverse so a node's gradient is fully accumulated (from every consumer)
before it is propagated to its own inputs.

## Gradient accumulation
A tensor used by more than one downstream operation accumulates
contributions from every path:
```python
a = x * 2
b = x * 3
c = (a + b).sum()
c.backward()  # x.grad == 5, i.e. d(2x+3x)/dx
```
This also holds across separate forward passes that reuse the same leaf:
calling `backward()` again after building a new graph adds to the existing
`.grad` rather than replacing it.

## Gradient lifecycle
- **Accumulation, not replacement**: repeated `backward()` calls add into
  `.grad`. Call `tensor.zero_grad()` (sets `.grad = None`) before a fresh
  backward pass if accumulation is not wanted — the same pattern a later
  optimizer will use per training step.
- **Graph freed on use**: as `run_backward` consumes a non-leaf tensor's
  `grad_fn`, it clears it (`grad_fn = None`), dropping references to saved
  inputs so the graph can be garbage-collected instead of kept alive
  indefinitely. Calling `backward()` again on that same non-leaf output
  raises `GradientStateError` — build a new forward pass to differentiate
  again. Leaves are unaffected: calling `backward()` directly on a leaf
  multiple times is allowed and simply accumulates each time.

## Broadcasting
Elementwise ops (`+`, `-`, `*`) support NumPy-style broadcasting on CPU
exactly as in Milestone 1: `CPUBackend.{add,sub,mul}_backward` compute the
"raw" gradient at the broadcast output shape, then reduce it down to each
input's original shape by summing over broadcast dimensions
(`_reduce_grad_to_shape`, `forge/backend/cpu.py`) — dimensions introduced by
broadcasting are summed away entirely; dimensions broadcast from size 1 are
summed back to size 1. CUDA's broadcasting is narrower (only the one
`(rows, cols)` + `(cols,)` row-broadcast shape forward already supports —
see `docs/architecture/cuda-backend.md`), so `CUDABackend.{add,sub,
mul}_backward` reduce the vector operand's gradient with a dedicated
`cf_reduce_rows` kernel (summing over rows) instead of a general
NumPy-style reduction.

## Matmul
`CPUBackend.matmul_backward` implements the four supported 1D/2D
combinations (vector·vector, matrix·vector, vector·matrix, matrix·matrix)
directly in NumPy, matching the forward semantics already present in
Milestone 1. `CUDABackend.matmul_backward` implements the same four cases
by composing existing/new CUDA kernels — the matrix·matrix case, for
example, is `grad_output @ b.T` and `a.T @ grad_output`, built from the
existing `matmul` kernel plus a new `cf_transpose` kernel (see
`docs/architecture/cuda-backend.md`). Higher-rank matmul remains
unsupported on both backends, as in Milestone 1.

## Sum
`x.sum(axis=..., keepdims=...)`'s CPU backward rule reinserts any reduced
dimensions (when `keepdims=False`) via `np.expand_dims`, then broadcasts the
upstream gradient back to `x`'s original shape. A full reduction
(`axis=None`) is the same code path with no dimensions to reinsert. CUDA
`sum()` only ever supports a full reduction forward (`axis=None` — see
`docs/architecture/cuda-backend.md`), so `CUDABackend.sum_backward` only
ever needs to broadcast the one upstream scalar back to every element of
the original shape (a dedicated `cf_broadcast_scalar` kernel); a non-`None`
axis can never reach `sum_backward` in the first place, since the forward
call already raised `CUDAError` before a `Node` was ever attached.

## Reshape
`x.reshape(...)`'s backward rule reshapes the upstream gradient back to
`x`'s original shape. On CPU this is a plain `np.ndarray.reshape` call; on
CUDA, `CUDABackend.reshape_backward` reuses the existing forward `reshape`
op (itself a device-to-device copy into a freshly shaped buffer) rather than
introducing a separate code path.

## ReLU
`x.relu()`'s backward rule is `grad_output` where the input was `> 0`, else
`0` (strict inequality, so the subgradient at exactly `0` is `0`). On CPU
this is a NumPy `np.where` over a boolean mask; on CUDA,
`CUDABackend.relu_backward` computes it with one dedicated kernel
(`cf_relu_backward`, `input[i] > 0 ? grad_output[i] : 0`) rather than a
separate mask-then-multiply pair of kernels, and never copies the input to
CPU to determine the mask (see `docs/architecture/cuda-backend.md`).

## Suspending graph construction: `no_grad`
As of Milestone 6, `forge.no_grad()` (`forge/autograd/engine.py`) is a
context manager that suspends `Node`/`grad_fn` attachment entirely:
```python
with forge.no_grad():
    prediction = model(x)   # requires_grad=False, grad_fn=None, regardless of model parameters
```
It is a single global flag (`is_grad_enabled()`), checked by
`Tensor._differentiable_wrap` alongside the existing "any input requires
grad" rule:
```python
requires_grad = is_grad_enabled() and any(t._requires_grad for t in inputs)
```
`__enter__`/`__exit__` save and restore the previous flag value (including
on exception), so nested `no_grad()` blocks are safe and grad tracking
always resumes correctly outside the `with` block. This is the mechanism
`forge.training.Trainer.evaluate()` uses to run a forward pass without
building or retaining a computation graph it will never call `backward()`
on -- see `docs/architecture/training-engine.md`.

## Device consistency in `backward()`
`Tensor.backward()` (`forge/tensor/tensor.py`) validates the same device
rule every other binary Tensor operation already enforces: an explicit
upstream `gradient` argument must already be on `self`'s own device, or
`backward()` raises `UnsupportedDeviceError` — a CUDA tensor's backward
pass never implicitly accepts a CPU-resident gradient (or vice versa), and
`.to()` remains the only sanctioned way to cross devices. A default
gradient of `1` (the omitted-`gradient` case, only valid for a scalar
output) is constructed directly on `self`'s device via
`backend.from_array(...)`, never on CPU-then-transferred. A gradient
dtype mismatch against `self`'s own dtype is silently cast on CPU (matching
pre-Milestone-10 behavior) but raises `UnsupportedDTypeError` on CUDA,
since the CUDA backend has no implicit-cast compute path.

## Unsupported CUDA operations
`.exp()`/`.log()` remain CPU-only in both forward and backward — the CUDA
backend has no `exp`/`log` kernel (`CUDABackend.exp`/`.log` raise
`CUDAError` unconditionally, unchanged since Milestone 8). Because the
forward call itself raises before `Tensor._differentiable_wrap` is ever
reached, a `Node` is never attached for these two operations on a CUDA
tensor — there is no separate "CUDA autograd not supported for exp/log"
check anywhere; the existing forward-unsupported check *is* the backward
failure point, which is the most localized place to report it. Moving the
tensor to CPU with `.to('cpu')` remains the documented way to use them.

## Known limitations
- No CUDA autograd for operations the CUDA backend doesn't implement at
  all: `exp`/`log` (see above), CUDA `sum(axis=...)` (forward already
  raises `CUDAError`; see `docs/architecture/cuda-backend.md`), and general
  N-D broadcasting beyond the one CUDA row-broadcast shape.
- No `retain_graph`-style option to keep a graph alive across multiple
  `backward()` calls on the same non-leaf output.
- Non-leaf tensors do not retain `.grad` (no `retain_grad()` equivalent).
- No graph visualization/debugging tools beyond `grad_fn`/`is_leaf`
  inspection.
- `no_grad()` is a single global flag, not per-tensor/per-thread state --
  safe for Forge's single-threaded synchronous execution model, but not a
  general context-management system.
- `Tensor.to(device)`/`Module.to(device)` remain non-differentiable device
  transfers (`requires_grad=False` on the result, any prior `.grad`
  cleared) -- CUDA autograd covers computation that happens *on* a device,
  not differentiating *through* a transfer between devices.
