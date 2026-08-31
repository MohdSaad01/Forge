# Tensor API (Milestone 1)

## Package layout
```
forge/
    tensor/    Tensor, DType
    backend/   Device, Backend, CPUBackend, get_backend
    exceptions.py
```

## Tensor
`forge.Tensor(data, dtype=None, device="cpu")`

- `data`: array-like, a NumPy array, or another `Tensor`.
- `dtype`: `None` (infer), a `DType`, or a dtype name string (`"float32"`, `"float64"`, `"int32"`, `"int64"`, `"bool"`). A NumPy array/Tensor with an already-supported dtype keeps it when `dtype=None`. Raw Python data defaults to `float32` for floats and `int64` for integers, regardless of platform.
- `device`: `"cpu"` (only executable device in M1) or a device string like `"cuda"`/`"cuda:0"`, which parses but raises `UnsupportedDeviceError` on construction since no CUDA backend exists yet.

Properties: `.shape`, `.dtype`, `.device`, `.ndim`. `.numpy()` returns the underlying CPU storage (shares memory; raises for non-CPU tensors).

Operations: `+`, `-`, `*` (broadcasting elementwise ops), `@` (matmul, 1D/2D operands only), `.sum(axis=None, keepdims=False)`, `.reshape(*shape)`. All operations return `Tensor`, never a raw NumPy array.

## Errors
`forge.exceptions.ForgeError` is the base class, with `ShapeMismatchError`, `UnsupportedDTypeError`, and `UnsupportedDeviceError` covering the invalid-input cases above.

## Backend boundary
`forge.backend.get_backend(device)` dispatches to a registered `Backend` implementation. Only `"cpu"` is registered in M1 (`CPUBackend`, a thin NumPy wrapper). `Device.parse` can name a `"cuda"` device without a backend existing for it — naming and executing are deliberately separate steps so CUDA support can be added later without changing the public Tensor API.

## Not yet implemented
No autograd, no gradient tracking, no neural-network modules, no CUDA execution. See `docs/development/roadmap.md`.
