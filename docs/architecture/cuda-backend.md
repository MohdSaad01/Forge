# CUDA Backend (Milestone 8)

## Summary
Forge has a real CUDA execution backend for a small, forward-only operation
set: tensor creation/transfer, `add`/`sub`/`mul` (exact-shape only),
`matmul` (the same 1D/2D cases the CPU backend supports), `sum` (full
reduction only), and `reshape`. Every one of these actually launches a CUDA
kernel (or issues a real `cudaMemcpy`/`cudaMalloc` call) on the GPU --
nothing in this list is a disguised CPU fallback. See
`docs/architecture/backend-architecture.md` for the general backend
boundary this fits into.

## Package layout
```
forge/
    backend/
        base.py            Backend ABC (+ `to_numpy`, new in M8)
        cpu.py               CPUBackend (+ `to_numpy`, trivial identity)
        __init__.py          get_backend() -- lazily imports forge.backend.cuda
        cuda/
            kernels.cu        CUDA C++ kernel source (compiled by build.py)
            build.py           Locates nvcc/MSVC, compiles kernels.cu -> a DLL
            backend.py         CUDAStorage, CUDABackend, get_cuda_backend(), is_cuda_available()
            __init__.py        Re-exports the above
    tensor/tensor.py         Tensor.to(device); CUDA-autograd guards
    exceptions.py             CUDAError (new in M8)
```

## Why `nvcc` + `ctypes`, not CuPy/PyTorch/Numba
The project constraints explicitly forbid delegating CUDA execution to
another deep-learning framework (PyTorch, TensorFlow, CuPy, JAX). The
verified development environment already has a working `nvcc` 12.6 + MSVC
2022 toolchain (confirmed by compiling and running a real `sm_50` kernel
before any Forge code was written for this milestone). Given that, the
simplest reliable strategy compatible with Windows + Python 3.13 + CC 5.0 is:

1. Write kernels in a single `.cu` file with `extern "C"` exported launcher
   functions (no C++ name mangling, so no binding-generator dependency).
2. Compile it with `nvcc` into a standard shared library (`.dll`).
3. Load that library from Python with the standard-library `ctypes` module
   and call the exported functions directly.

This introduces no new binary dependency beyond the CUDA Toolkit the
hardware verification already required, and keeps every line of kernel code
and every line of Python dispatch code inside Forge. See
`docs/architecture/decisions/ADR-004-cuda-execution-strategy.md`.

## Build
`forge/backend/cuda/build.py` compiles `kernels.cu` into
`_forge_cuda_kernels_sm_50.dll` (in the same directory) the first time a
CUDA device is actually requested -- never at `import forge` time, so a
CPU-only environment never needs `nvcc` on PATH. The compiled `.dll` is
cached (skipped if newer than the `.cu` source) and is a per-machine build
artifact, not something committed to the repository.

On Windows, `nvcc` needs MSVC's `cl.exe` as its host compiler. `build.py`
tries compiling first with the ambient `PATH`, and if that fails, locates
the latest MSVC installation via `vswhere.exe` and retries with its
`Hostx64/x64` bin directory prepended -- this is what makes the build work
from an ordinary shell rather than only a "Developer Command Prompt".

Compilation targets `-arch=sm_50` specifically (Compute Capability 5.0, the
verified 940MX), and links the CUDA runtime statically (`--cudart static`)
so the resulting `.dll` does not depend on `cudart64_*.dll` being on `PATH`
at import time -- only the NVIDIA display driver itself is required at run
time.

## Backend boundary
`forge/backend/__init__.py`'s `get_backend()` dispatches `"cpu"` to an
eagerly constructed `CPUBackend` singleton (unchanged from Milestone 1) and
`"cuda"` to a lazily constructed `CUDABackend` singleton, imported only on
first use. Constructing `CUDABackend` compiles the kernel library (if not
already cached) and calls `cudaGetDeviceCount()`; any failure at either step
-- missing `nvcc`, a failed compile, no CUDA-capable device, a bad driver --
raises `forge.CUDAError` with the underlying reason, and that failure is
cached so repeated `device="cuda"` requests on a non-CUDA machine fail fast
instead of retrying an expensive compile every time.

`device="cuda:N"` for `N != 0` raises `CUDAError` immediately: Forge targets
a single GPU (index 0) in this milestone (see **Non-goals** below).

## Tensor storage: `CUDAStorage`
A CPU `Tensor._data` is a `numpy.ndarray`. A CUDA `Tensor._data` is a
`CUDAStorage` (`forge/backend/cuda/backend.py`) -- a handle around a real
`cudaMalloc`-allocated device pointer plus `shape`/`dtype` metadata. It is
never a NumPy array relabeled as CUDA: `isinstance(cuda_tensor._data,
CUDAStorage)` is `True` and `isinstance(cuda_tensor._data, np.ndarray)` is
always `False` for a genuine CUDA tensor (see
`tests/test_cuda_backend.py::test_tensor_moves_cpu_to_cuda`, which asserts
exactly this). Device memory is released via `cudaFree` when a
`CUDAStorage` is garbage collected. No custom GPU allocator is implemented
-- each operation calls `cudaMalloc` directly for its output, per the
milestone's "do not build an allocator" constraint.

Because `CUDAStorage` exposes the same `shape`/`dtype`/`ndim`/`size`
surface a NumPy array does, `Tensor` code that only reads that metadata
(shape validation, dtype resolution, `__repr__`, etc.) works unmodified
across both backends -- the divergence is entirely inside `Backend` method
implementations, matching the documented backend boundary.

## Device transfer: `Tensor.to(device)`
New in this milestone. `x.to("cuda")` / `x.to("cpu")` explicitly moves a
tensor's data to another device, always by copying through a host
`numpy.ndarray` intermediate (`Backend.to_numpy()` on the source, then
`Backend.from_array()` on the target) -- there is no direct GPU peer-copy
mechanism in this milestone, matching the "no elaborate memory management"
constraint. Moving to the same device is a no-op that returns the original
tensor object unchanged (no copy).

Every other Tensor operation (`+`, `-`, `*`, `@`, `.sum()`, `.reshape()`)
raises `UnsupportedDeviceError` if its operands are on different devices --
`.to()` is the only sanctioned way to cross devices; no operation silently
transfers a tensor to make itself work.

## Operation set
| Operation | Kernel(s) | Dtypes | Notes |
|---|---|---|---|
| create/transfer | `cf_malloc`/`cf_free`/`cf_memcpy_{h2d,d2h,d2d}` | float32, float64, int32, int64, bool | Raw byte copy; dtype-generic |
| `add`/`sub`/`mul` | `cf_{add,sub,mul}_{f32,f64}` | float32, float64 | Exact-shape only, no broadcasting |
| `matmul` | `cf_matmul_{f32,f64}` | float32, float64 | Naive one-thread-per-output-element kernel; 1D vectors are reinterpreted as degenerate `M=1`/`N=1` 2D matmuls on the host side, matching the CPU backend's four 1D/2D cases |
| `sum` | `cf_sum_{f32,f64}` | float32, float64 | Full reduction only (`axis=None`); block-level shared-memory tree reduction + atomic accumulation across blocks (a CAS-based emulation for `atomicAdd(double*)`, since CC 5.0 has no native one) |
| `reshape` | `cf_memcpy_d2d` | any transferable dtype | Implemented as a device-to-device copy into a freshly allocated buffer with new shape metadata (no in-place aliasing, to avoid any storage-ownership ambiguity without an allocator) |

`relu`/`exp`/`log` are required by the `Backend` ABC (the CPU backend
implements them) but the CUDA backend raises `CUDAError` for all three --
no kernel exists for them in this milestone. This is a real, tested failure
path (`tests/test_cuda_backend.py::test_relu_is_unsupported_on_cuda` etc.),
not an oversight: moving a tensor to CPU (`.to("cpu")`) is the documented
way to use them today.

Every kernel launch is followed by an explicit `cf_synchronize()`
(`cudaDeviceSynchronize()`) before its result is trusted or returned, so
CUDA's asynchronous execution model can never be mistaken for a completed
operation.

## Numerical consistency
`tests/test_cuda_consistency.py` runs the same inputs through both
backends for every shared operation (`add`/`sub`/`mul`/`matmul`/`sum`/
`reshape`, both `float32` and `float64`, several matmul shape
combinations) and asserts `np.testing.assert_allclose` agreement
(`rtol=atol=1e-5`) -- never bit-for-bit equality, per
`docs/architecture/backend-architecture.md`.

## Autograd
**CUDA autograd is not supported in this milestone**, by design (the
milestone brief explicitly scopes M8 to forward execution). The existing
autograd backward closures (`Tensor._binary_op`, `.matmul`, `.sum`,
`.reshape`) are written directly against NumPy arrays (`grad_output * b`,
`a.T @ grad_output`, etc.) -- extending them to dispatch through
`Backend`-level gradient kernels for every operation is a substantial
redesign of the differentiation machinery, not a boundary Forge can safely
extend "cleanly" within this milestone's forward-execution scope.

Rather than silently building a graph that would crash or produce wrong
gradients on first `backward()`, or silently copying tensors to CPU to fake
CUDA gradients (both explicitly disallowed), Forge fails clearly at the two
points where this would otherwise go wrong:
- `Tensor._differentiable_wrap` raises `UnsupportedDeviceError` if a
  differentiable operation on a non-CPU device would need to attach a
  gradient-tracking node (i.e., grad is enabled and an input requires
  grad).
- `Tensor.backward()` raises `UnsupportedDeviceError` immediately if called
  on a non-CPU tensor.
- `Tensor.to(device)` always returns a fresh leaf tensor with
  `requires_grad=False`, regardless of the source tensor's gradient state
  -- a device transfer never carries a gradient requirement across the
  boundary it is not equipped to differentiate through.

Constructing a CUDA leaf tensor with `requires_grad=True` still succeeds
(it is just a flag on a leaf with no operation performed yet); the failure
surfaces at the first attempted differentiable operation or `backward()`
call, which is the most localized point to report it.

## Errors
All CUDA-specific failures raise `forge.CUDAError` (new in this milestone,
`forge/exceptions.py`): CUDA unavailable (no `nvcc`, no compatible device),
backend initialization/compile failure, an unsupported operation
(`relu`/`exp`/`log`, `sum(axis=...)`, elementwise broadcasting) or dtype
(non-float compute), a memory allocation failure, or an invalid device
index. Device-mismatch (`cpu_tensor + cuda_tensor`) and unrecognized device
strings remain `UnsupportedDeviceError`, matching the existing convention;
CUDA-autograd errors also use `UnsupportedDeviceError` since they describe
a device *capability* boundary rather than a CUDA runtime/toolchain
failure. See `forge/exceptions.py::CUDAError`'s docstring for the exact
split.

## Detecting a fake CUDA backend
`tests/test_cuda_backend.py::test_backend_dispatch_is_structurally_distinct_from_cpu`
asserts `get_backend("cpu")` and `get_backend("cuda")` return instances of
different, unrelated classes (`CPUBackend` is not a base of `CUDABackend`
or vice versa). Every CUDA test additionally asserts
`isinstance(tensor._data, CUDAStorage)` and
`not isinstance(tensor._data, np.ndarray)`, so a regression that silently
routed `"cuda"` back to `CPUBackend`, or that stored a NumPy array under a
CUDA-tagged `Tensor`, would fail these tests immediately.

## Hardware verification
Verified on the actual development GPU (NVIDIA GeForce 940MX, CC 5.0,
driver 582.53, CUDA Toolkit 12.6): `tests/test_cuda_backend.py` and
`tests/test_cuda_consistency.py` (54 tests) all pass when run directly on
this machine, exercising real kernel compilation, device-count probing,
memory allocation, host<->device transfer, kernel launch +
synchronization, and CPU/CUDA numeric agreement. The same suites were also
run with `PATH` stripped of the CUDA toolchain to confirm they skip cleanly
(`54 skipped`, `0 failed`) rather than erroring, and that the CPU-only
suite (332 tests) is entirely unaffected by CUDA's absence.

## Limitations
- **Operation set is intentionally small**: only `add`/`sub`/`mul`
  (exact-shape), `matmul` (1D/2D), `sum` (full reduction), `reshape`, and
  transfer. No `relu`/`exp`/`log`, no elementwise broadcasting, no
  axis-wise reduction on CUDA.
- **No CUDA autograd**: forward execution only (see **Autograd** above).
  No CUDA-trained model is possible in this milestone.
- **No CUDA `Trainer`/DataLoader integration**: `forge.training.Trainer`
  remains CPU-only (unchanged from Milestone 6); nothing in this milestone
  wires a training loop to CUDA tensors.
- **No `Module.to(device)`**: only `Tensor.to(device)` exists. `forge.nn.Module`
  (`Linear`, `ReLU`, any composite model) has no mechanism to move its
  `Parameter`s to CUDA in this milestone, so a module's `forward()` cannot
  currently run on CUDA -- its parameters stay on CPU, and calling it with a
  CUDA input raises `UnsupportedDeviceError` (device mismatch), correctly
  rather than silently. This milestone verifies CUDA execution at the raw
  `Tensor` level only (`+`, `-`, `*`, `@`, `.sum()`, chained -- see
  `tests/test_cuda_consistency.py::test_chained_ops_consistency`), not
  through `nn.Module`. Wiring `Module`/`Trainer` to CUDA is future work.
- **No CUDA persistence**: `save_model`/`load_model` remain CPU-only
  (unchanged from Milestone 7); a CUDA parameter is never saved or loaded.
- **Single GPU, index 0 only**: `device="cuda:N"` for `N != 0` raises
  `CUDAError`. No multi-GPU support.
- **No custom GPU memory allocator**: every operation's output is a fresh
  `cudaMalloc`, freed via `cudaFree` on garbage collection. Reasonable for
  the small models this milestone targets; not tuned for throughput.
- **Naive matmul**: one thread per output element, no tiling/shared-memory
  GEMM optimization. Correctness-first, per the milestone brief;
  performance optimization is out of scope until Milestone 11.
- **No int/bool compute kernels**: transfer works for `int32`/`int64`/
  `bool`, but arithmetic ops on CUDA require `float32`/`float64`.
- **No CLI/benchmarking integration**: this milestone adds the backend and
  its tests only; CLI and benchmark surfaces (`docs/product/requirements.md`)
  are unaffected.
- **No performance claims**: `docs/architecture/cuda-backend.md` reports
  correctness and hardware verification only. The 940MX is old and small
  workloads may well be slower on CUDA than CPU due to transfer and launch
  overhead; benchmarking is Milestone 11's concern.
