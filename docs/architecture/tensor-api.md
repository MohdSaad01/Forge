# Tensor API (Milestone 1 + 2 + 3 + 4)

## Package layout
```
forge/
    tensor/    Tensor, DType
    backend/   Device, Backend, CPUBackend, get_backend
    autograd/  Node, run_backward, backward-math helpers (M2)
    nn/        Module, Parameter, Linear, ReLU (M3), Loss, MSELoss, CrossEntropyLoss (M4)
               (see docs/architecture/modules.md, docs/architecture/optimization.md)
    optim/     Optimizer, SGD (M4, see docs/architecture/optimization.md)
    random.py  process-global default RNG for deterministic init (M3)
    exceptions.py
```

## Tensor
`forge.Tensor(data, dtype=None, device="cpu", requires_grad=False)`

- `data`: array-like, a NumPy array, or another `Tensor`.
- `dtype`: `None` (infer), a `DType`, or a dtype name string (`"float32"`, `"float64"`, `"int32"`, `"int64"`, `"bool"`). A NumPy array/Tensor with an already-supported dtype keeps it when `dtype=None`. Raw Python data defaults to `float32` for floats and `int64` for integers, regardless of platform.
- `device`: `"cpu"` (only executable device in M1/M2) or a device string like `"cuda"`/`"cuda:0"`, which parses but raises `UnsupportedDeviceError` on construction since no CUDA backend exists yet.
- `requires_grad`: `False` by default. `True` enables gradient tracking; only `float32`/`float64` tensors may set it (raises `GradientStateError` otherwise). See `docs/architecture/autograd.md`.

Properties: `.shape`, `.dtype`, `.device`, `.ndim`. `.numpy()` returns the underlying CPU storage (shares memory; raises for non-CPU tensors). Gradient-tracking properties (M2): `.requires_grad`, `.is_leaf`, `.grad_fn`, `.grad`.

Operations: `+`, `-`, `*` (broadcasting elementwise ops), `@` (matmul, 1D/2D operands only), `.sum(axis=None, keepdims=False)`, `.reshape(*shape)`, `.relu()` (M3, elementwise `max(x, 0)`), `.exp()`, `.log()` (M4, elementwise; added for a numerically stable `CrossEntropyLoss`, see `docs/architecture/optimization.md`). All operations return `Tensor`, never a raw NumPy array, and are differentiable when at least one operand requires grad.

Autograd methods (M2): `.backward(gradient=None)` runs reverse-mode differentiation from this tensor; `.zero_grad()` clears `.grad`. See `docs/architecture/autograd.md` for full semantics.

## Errors
`forge.exceptions.ForgeError` is the base class, with `ShapeMismatchError`, `UnsupportedDTypeError`, `UnsupportedDeviceError`, and `GradientStateError` (M2: invalid backward/gradient usage) covering the invalid-input cases above.

## Backend boundary
`forge.backend.get_backend(device)` dispatches to a registered `Backend` implementation. Only `"cpu"` is registered in M1/M2 (`CPUBackend`, a thin NumPy wrapper). `Device.parse` can name a `"cuda"` device without a backend existing for it — naming and executing are deliberately separate steps so CUDA support can be added later without changing the public Tensor API.

## Not yet implemented
No training engine, dataset/DataLoader abstraction, persistence, or CUDA execution. Neural-network modules/parameters exist as of Milestone 3 (`forge.nn`, see `docs/architecture/modules.md`); losses and an SGD optimizer exist as of Milestone 4 (`forge.nn.MSELoss`/`CrossEntropyLoss`, `forge.optim.SGD`, see `docs/architecture/optimization.md`). See `docs/development/roadmap.md`.
