# Forge Backend Architecture

## Goal
Allow CPU and CUDA execution behind a stable tensor/framework API.

## Boundary

```text
Tensor / Autograd / Optimizer API
        ↓
Backend dispatch (forward AND backward, as of Milestone 10)
   ┌────┴────┐
 CPU       CUDA
```

The backend boundary should answer:
- Which device owns this tensor?
- Which implementation performs this operation (forward *and*, as of
  Milestone 10, its gradient)?
- What dtype/shape combinations are supported?
- How are device transfers performed?
- What error is raised for unsupported combinations?
- How does an optimizer update a parameter's storage on this device (as of
  Milestone 10, `Backend.sgd_step`)?

## CPU
CPU is the reference backend for correctness and testability.

## CUDA
CUDA is a real execution backend, not a label. As of Milestone 8, a small,
measured operation set (tensor transfer, `add`/`sub`/`mul`, `matmul`,
`sum`, `reshape`) executes as genuine CUDA kernels on the verified 940MX
environment via an `nvcc`-compiled kernel library loaded through `ctypes`
(see `docs/architecture/cuda-backend.md` and
`docs/architecture/decisions/ADR-004-cuda-execution-strategy.md`). Kernels
target Compute Capability 5.0. As of Milestone 9, `relu` is also a real
CUDA kernel, `add`/`sub`/`mul` additionally support one targeted
row-broadcast shape (needed for a batched `Linear`'s bias add), and this
operation set is reachable through `nn.Module.to("cuda")` -- see
`docs/architecture/cuda-backend.md` and `docs/architecture/modules.md`. As
of Milestone 10, every one of these operations also has a real CUDA
*backward* kernel, so `Backend` dispatch now covers gradient computation
too (`add_backward`, `matmul_backward`, `relu_backward`, etc. -- see
`docs/architecture/autograd.md`'s **Backend-aware backward dispatch**
section and `docs/architecture/cuda-backend.md`'s **CUDA autograd**
section) -- CUDA execution is no longer forward-only.

## Consistency
For operations implemented on both backends, tests should compare CPU and CUDA results using appropriate tolerances.

## Performance
Do not optimize every abstraction prematurely. Establish correctness first, benchmark, then optimize measured hot paths.
