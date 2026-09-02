# Tensor API (Milestone 1 + 2 + 3 + 4 + 5 + 8 + 9)

## Package layout
```
forge/
    tensor/    Tensor, DType
    backend/   Device, Backend, CPUBackend, get_backend
    autograd/  Node, run_backward, backward-math helpers (M2)
    nn/        Module, Parameter, Linear, ReLU (M3), Loss, MSELoss, CrossEntropyLoss (M4)
               (see docs/architecture/modules.md, docs/architecture/optimization.md)
    optim/     Optimizer, SGD (M4, see docs/architecture/optimization.md)
    data/      Dataset, TensorDataset, Subset, random_split, DataLoader, transforms (M5,
               see docs/architecture/data-system.md)
    random.py  process-global default RNG for deterministic init (M3)
    exceptions.py
```

## Tensor
`forge.Tensor(data, dtype=None, device="cpu", requires_grad=False)`

- `data`: array-like, a NumPy array, or another `Tensor`.
- `dtype`: `None` (infer), a `DType`, or a dtype name string (`"float32"`, `"float64"`, `"int32"`, `"int64"`, `"bool"`). A NumPy array/Tensor with an already-supported dtype keeps it when `dtype=None`. Raw Python data defaults to `float32` for floats and `int64` for integers, regardless of platform.
- `device`: `"cpu"` or a device string like `"cuda"`/`"cuda:0"`. As of Milestone 8, `"cuda"` executes on a real CUDA backend when one is available on the machine (see `docs/architecture/cuda-backend.md`); when unavailable, or for an index other than `0`, it raises `forge.CUDAError`. `x.to(device)` (Milestone 8) explicitly moves a tensor's data to another device, copying it.
- `requires_grad`: `False` by default. `True` enables gradient tracking; only `float32`/`float64` tensors may set it (raises `GradientStateError` otherwise). See `docs/architecture/autograd.md`.

Properties: `.shape`, `.dtype`, `.device`, `.ndim`. `.numpy()` returns the underlying CPU storage (shares memory; raises for non-CPU tensors). Gradient-tracking properties (M2): `.requires_grad`, `.is_leaf`, `.grad_fn`, `.grad`.

Operations: `+`, `-`, `*` (broadcasting elementwise ops), `@` (matmul, 1D/2D operands only), `.sum(axis=None, keepdims=False)` (CPU: any axis; CUDA, as of M14: `axis=None` or `axis=1`/`-1` on a 2D tensor only, see `docs/architecture/cuda-backend.md`), `.reshape(*shape)`, `.relu()` (M3, elementwise `max(x, 0)`), `.exp()`, `.log()` (M4, elementwise; added for a numerically stable `CrossEntropyLoss`, see `docs/architecture/optimization.md`; CUDA-capable as of M14), `.to(device)` (M8, explicit device transfer, see `docs/architecture/cuda-backend.md`). All operations return `Tensor`, never a raw NumPy array, and are differentiable when at least one operand requires grad. As of M10, CUDA operations this backend implements build and differentiate a real CUDA graph too (no longer forward-only); an operation a given device's backend doesn't implement (e.g. `sum(axis=0)` on CUDA) raises `CUDAError`/`UnsupportedDeviceError` as appropriate rather than building a graph it cannot correctly back-propagate through.

Autograd methods (M2): `.backward(gradient=None)` runs reverse-mode differentiation from this tensor; `.zero_grad()` clears `.grad`. See `docs/architecture/autograd.md` for full semantics.

## Errors
`forge.exceptions.ForgeError` is the base class, with `ShapeMismatchError`, `UnsupportedDTypeError`, `UnsupportedDeviceError`, `GradientStateError` (M2: invalid backward/gradient usage), and `CUDAError` (M8: CUDA-specific runtime/toolchain/unsupported-operation failures, see `docs/architecture/cuda-backend.md`) covering the invalid-input cases above.

## Backend boundary
`forge.backend.get_backend(device)` dispatches to a registered `Backend` implementation. `"cpu"` (`CPUBackend`, a thin NumPy wrapper) is always available. As of Milestone 8, `"cuda"` dispatches to a real `CUDABackend`, built lazily on first use; `Device.parse` continues to name a `"cuda"` device independently of whether a backend can actually execute on it (naming and executing remain deliberately separate steps).

## Module-level device movement (Milestone 9)
`forge.nn.Module.to(device)` (see `docs/architecture/modules.md`) recursively
moves a module tree's `Parameter`s (`Tensor` subclasses) between devices,
using a new private in-place counterpart to `Tensor.to()`,
`Tensor._move_storage_(device)`: unlike the public `.to()` (value semantics,
always a fresh `requires_grad=False` leaf), this mutates `self._data`/
`self._device` directly so a `Parameter`'s object identity and
`requires_grad` flag survive the move. It is not part of the public Tensor
API -- ordinary code should use `Tensor.to()` (for a plain `Tensor`) or
`Module.to()` (for a `Module`'s `Parameter`s), never call
`_move_storage_()` directly.

## Not yet implemented
No general CUDA elementwise broadcasting beyond the targeted row-broadcast (`add`/`sub`/`mul`) and column-broadcast (`sub` only, M14) shapes, and no general CUDA N-D axis reduction beyond `sum(axis=1)` on a 2D tensor (M14) -- see `docs/architecture/cuda-backend.md`. Neural-network modules/parameters exist as of Milestone 3 (`forge.nn`, see `docs/architecture/modules.md`); losses and an SGD optimizer exist as of Milestone 4 (`forge.nn.MSELoss`/`CrossEntropyLoss`, `forge.optim.SGD`, see `docs/architecture/optimization.md`); a dataset/DataLoader/transform abstraction exists as of Milestone 5 (`forge.data`, see `docs/architecture/data-system.md`); a training engine exists as of Milestone 6 (`forge.training.Trainer`, see `docs/architecture/training-engine.md`); model persistence exists as of Milestone 7 (`forge.save_model`/`load_model`, see `docs/architecture/persistence.md`); a CUDA backend exists as of Milestone 8 (see `docs/architecture/cuda-backend.md`); `Module.to(device)`, CUDA `relu`, and CUDA `Linear`/`ReLU` module execution exist as of Milestone 9; CUDA autograd exists as of Milestone 10; CUDA `Trainer`/`MSELoss` integration exists as of Milestone 12; CUDA model persistence exists as of Milestone 13; CUDA `exp`/`log`/`sum(axis=1)`/`CrossEntropyLoss` exist as of Milestone 14 (see `docs/architecture/cuda-backend.md` and `docs/architecture/modules.md`). See `docs/development/roadmap.md`.
