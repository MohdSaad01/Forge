# Forge Autograd (Milestone 2)

## Package layout
```
forge/
    autograd/
        engine.py     Node (grad_fn), topological traversal, run_backward
        functions.py  local backward math (broadcast reduction, matmul, sum)
    tensor/tensor.py  requires_grad, grad, backward(), zero_grad()
```

`forge.autograd` has no dependency on `forge.tensor`; it operates on
tensor-like objects via duck typing (`is_leaf`, `requires_grad`, `_grad_fn`,
`_accumulate_grad`). `Tensor` depends on `forge.autograd`, not the reverse,
so there is no import cycle and the graph engine stays testable on its own.

## requires_grad
`Tensor(data, requires_grad=True)` marks a tensor for gradient tracking.
Only floating-point tensors (`float32`/`float64`) may set `requires_grad`;
requesting it on an int/bool tensor raises `GradientStateError`.

A tensor produced by a differentiable operation requires grad if **any**
input to that operation requires grad, matching the usual autodiff rule
that gradient tracking is contagious forward through the graph.

## Computation graph
Each differentiable operation (`+`, `-`, `*`, `@`, `.sum()`, `.reshape()`)
that has at least one grad-requiring input attaches a `Node` to its output
tensor as `output.grad_fn`. A `Node` holds:
- `inputs`: the operation's input tensors (not copies of their data).
- `backward_fn`: a closure that maps an upstream gradient to a tuple of
  gradients, one per input, given the small amount of forward-pass state
  the operation needs (shapes, or in matmul's case the input arrays
  themselves — sum and reshape need no saved data beyond shape metadata).

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
Elementwise ops (`+`, `-`, `*`) support NumPy-style broadcasting exactly as
in Milestone 1. Their backward rules compute the "raw" gradient at the
broadcast output shape, then reduce it down to each input's original shape
by summing over broadcast dimensions (`forge.autograd.functions.
reduce_grad_to_shape`) — dimensions introduced by broadcasting are summed
away entirely; dimensions broadcast from size 1 are summed back to size 1.

## Matmul
`forge.autograd.functions.matmul_backward` implements the four supported
1D/2D combinations (vector·vector, matrix·vector, vector·matrix,
matrix·matrix), matching the forward semantics already present in
Milestone 1. Higher-rank matmul remains unsupported, as in Milestone 1.

## Sum
`x.sum(axis=..., keepdims=...)`'s backward rule reinserts any reduced
dimensions (when `keepdims=False`) via `np.expand_dims`, then broadcasts the
upstream gradient back to `x`'s original shape. A full reduction
(`axis=None`) is the same code path with no dimensions to reinsert.

## Reshape
`x.reshape(...)`'s backward rule reshapes the upstream gradient back to
`x`'s original shape; no numerical computation is needed.

## Known limitations
- No optimizers or losses (later milestones). Neural-network modules and
  parameters exist as of Milestone 3 (`forge.nn`, see
  `docs/architecture/modules.md`) and are built entirely on the autograd
  system described here.
- No CUDA autograd.
- No `retain_graph`-style option to keep a graph alive across multiple
  `backward()` calls on the same non-leaf output.
- Non-leaf tensors do not retain `.grad` (no `retain_grad()` equivalent).
- No graph visualization/debugging tools beyond `grad_fn`/`is_leaf`
  inspection.
