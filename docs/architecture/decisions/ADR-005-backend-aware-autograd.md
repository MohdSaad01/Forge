# ADR-005: Backend-Aware Backward Dispatch for CUDA Autograd

## Status
Accepted

## Context
Milestones 8-9 built a real, forward-only CUDA execution backend
(`docs/architecture/cuda-backend.md`). Forge's autograd engine (Milestone
2, `docs/architecture/autograd.md`) predates that backend entirely: every
differentiable Tensor operation's backward closure (`Tensor._binary_op`,
`.matmul()`, `.sum()`, `.reshape()`, `.relu()`) was written directly against
`numpy.ndarray` -- `grad_output * b`, `a.T @ grad_output`,
`np.where(mask, grad_output, 0)`, and so on. None of that code can run
against a `CUDAStorage` operand: NumPy operators are not defined for it, and
Milestones 8-9 deliberately left CUDA autograd unimplemented rather than
attempt a partial extension mid-forward-execution-milestone (see
`docs/architecture/cuda-backend.md`'s pre-Milestone-10 **Autograd**
section).

Milestone 10's brief is explicit that CUDA autograd must not be implemented
by copying a CUDA graph's tensors to CPU, running the existing NumPy
backward closures there, and copying gradients back -- that would make CUDA
training *appear* to work while silently executing on the CPU. It must also
not duplicate the entire autograd engine (a second `Node`/`run_backward` for
CUDA) -- graph traversal, topological ordering, and gradient-lifecycle
bookkeeping (leaf accumulation, graph freeing, `no_grad`) have nothing
device-specific about them; only the *numerical content* of a backward rule
does.

## Decision
Make the `Backend` ABC (`forge/backend/base.py`) own backward computation,
not just forward computation. Each differentiable Tensor operation gets one
new `Backend` method alongside its existing forward method:
`add_backward`, `sub_backward`, `mul_backward`, `matmul_backward`,
`sum_backward`, `reshape_backward`, `relu_backward` (plus `sgd_step` for the
optimizer boundary -- see below). `Tensor`'s backward closures
(`forge/tensor/tensor.py`) become thin: each one calls
`get_backend(self.device).<op>_backward(...)` instead of containing any
NumPy math itself. `CPUBackend` implements these seven methods with the
same NumPy formulas the pre-Milestone-10 closures used (relocated from the
now-deleted `forge/autograd/functions.py`); `CUDABackend` implements them
with real CUDA kernels (`forge/backend/cuda/kernels.cu`'s "backward-only
kernels" section: `cf_neg`, `cf_relu_backward`, `cf_scale`, `cf_transpose`,
`cf_reduce_rows`, `cf_broadcast_scalar`), composing them with existing
forward kernels (`matmul`, `reshape`, row-broadcast `mul`) wherever
possible rather than writing a fully independent kernel per gradient.

`forge/autograd/engine.py` (`Node`, `run_backward`, `no_grad`) is untouched
in its graph-traversal logic and stays completely backend-*agnostic*: it
never branches on device type, it only calls whatever `backward_fn` closure
a `Node` was built with, and combines multiple gradient contributions to
the same tensor via `get_backend(tensor.device).add(...)` (since a
`CUDAStorage` has no `__add__`, this reuses the existing forward `add`
kernel rather than inventing a second accumulation mechanism).

`SGD.step()` (`forge/optim/sgd.py`) follows the identical pattern:
`Backend.sgd_step(data, grad, lr)` replaces the Milestone 4 implementation's
direct NumPy in-place mutation, so a CUDA `Parameter` updates via a real
kernel (`cf_sgd_step`, in-place, no new allocation) instead of `self.lr *
param.grad._data` (which would attempt NumPy arithmetic on a `CUDAStorage`
and fail).

## Rationale
- **No CPU fallback, by construction.** Because `Tensor` never contains
  backend-specific math -- only `Backend` implementations do -- there is no
  code path where a CUDA tensor's backward computation could silently run
  on NumPy. `tests/test_cuda_autograd.py::test_cuda_linear_backward_never_calls_cpu_backend`
  and `::test_cuda_multilayer_model_backward_never_calls_cpu_backend` assert
  this structurally (a monkeypatched `CPUBackend` records zero calls during
  a full CUDA model forward + backward pass), the same technique
  `tests/test_module_cuda.py` already used to prove forward execution never
  fell back.
- **Minimal duplication.** Only seven methods (plus `sgd_step`) needed a
  second implementation. The graph engine -- the harder-to-get-right half
  of an autograd system (topological ordering, leaf/non-leaf bookkeeping,
  graph freeing, `no_grad` suspension) -- was written once in Milestone 2
  and needed zero changes for Milestone 10 beyond the one
  backend-dispatched `add()` call for multi-consumer accumulation.
- **Matches the existing `Backend` boundary exactly.** `Backend` already
  drew the line between "what Tensor validates and orchestrates" and "what
  a device-specific implementation computes" for forward ops
  (`docs/architecture/backend-architecture.md`). Extending that same
  boundary to backward ops, rather than inventing a parallel
  `GradientBackend` or a `cuda_backward()` public API, keeps `Tensor
  -> Backend` as the *only* place a device branches, and keeps
  `loss.backward()` as the single public entry point the milestone brief
  requires.
- **CUDA kernels reuse forward kernels wherever the math allows it.**
  E.g. `matmul_backward`'s matrix·matrix case is `grad_output @ b.T` and
  `a.T @ grad_output` -- implemented as the existing `matmul` kernel called
  against a freshly transposed operand (`cf_transpose`, one new kernel),
  not a bespoke "matmul-backward" kernel. This keeps the *new* kernel
  surface small (six new kernels total: `neg`, `relu_backward`, `scale`,
  `transpose`, `reduce_rows`, `broadcast_scalar`, plus `sgd_step`) despite
  covering seven backward rules across four operations with several shape
  cases each.
- **No unsupported combinations invented to satisfy a theoretical API.**
  `sum_backward`'s CUDA implementation only ever handles a full reduction,
  because forward `CUDABackend.sum(axis=...)` already raises `CUDAError`
  for anything else -- there is no reachable path to a non-`None`-axis
  backward call, so none was built. `add_backward`/`sub_backward`/
  `mul_backward`'s CUDA implementations only ever handle exact-shape or the
  one row-broadcast shape, for the same reason.

## Consequences
- `forge/autograd/functions.py` (Milestone 2's pure-NumPy backward-math
  module) no longer exists; its content lives inside `CPUBackend`
  (`forge/backend/cpu.py`) as private helpers plus `Backend` method
  implementations. Anything that previously imported from
  `forge.autograd.functions` directly (nothing outside `forge/tensor/
  tensor.py` did) would need to import from `forge.backend.cpu` instead --
  not a public API, so this is not a breaking change to `forge`'s public
  surface.
- Adding a new differentiable operation to Forge now means adding a forward
  method **and** a backward method to the `Backend` ABC, with both a
  `CPUBackend` and (if CUDA support is intended) a `CUDABackend`
  implementation -- one more required method per backend than before, but
  no new *kind* of file or subsystem.
- `Tensor.backward()` and `Tensor._accumulate_grad()` are now
  device-generic: they no longer hard-code `np.ndarray`-specific behavior
  (`np.ones(())`, `.astype()`, `array + array`) inline, instead routing
  through `Backend.from_array`/`Backend.add`/dtype comparison. This makes
  `Tensor`'s autograd-facing code slightly more abstract, in exchange for
  it working unmodified on any future backend that implements the same
  `Backend` contract.
- A device-mismatched explicit `backward(gradient=...)` call, or a
  dtype-mismatched one on CUDA, now raises before `run_backward` is ever
  invoked (`Tensor.backward()`'s own validation), rather than failing
  wherever the mismatch would have first surfaced numerically -- a direct
  benefit of the boundary being explicit rather than implicit.
