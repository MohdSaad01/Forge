# CUDA Backend (Milestones 8-10; CUDA `Trainer`/loss integration in Milestone 12; CUDA model persistence in Milestone 13; CUDA `CrossEntropyLoss` in Milestone 14; CUDA `Conv2d`/`MaxPool2d` in Milestone 15; CUDA `Dropout` in Milestone 16; CUDA Adam in Milestone 17; CUDA Conv2d-backward performance optimization in Milestone 21; CUDA memory statistics and allocation lifecycle in Milestone 22)

## Summary
Forge has a real CUDA execution backend for a small operation set: tensor
creation/transfer, `add`/`mul` (exact-shape, plus one targeted row-broadcast
shape added in Milestone 9), `sub` (those two shapes plus one targeted
column-broadcast shape added in Milestone 14), `matmul` (the same 1D/2D
cases the CPU backend supports), `sum` (full reduction, plus axis=1 on a 2D
tensor as of Milestone 14), `reshape`, `relu` (Milestone 9), `exp`/`log`
(Milestone 14), and `conv2d`/`max_pool2d` (Milestone 15). Every one of these
actually launches a CUDA kernel (or issues a real `cudaMemcpy`/`cudaMalloc`
call) on the GPU -- nothing in this list is a disguised CPU fallback. As of
Milestone 9, this operation set is also reachable through the high-level
`nn.Module` API (`Module.to("cuda")`, then an ordinary `Linear`/`ReLU`
forward pass) -- see **Module and Parameter device movement** below and
`docs/architecture/modules.md`. As of
**Milestone 10, this operation set is no longer forward-only**: every one of
these operations has a real CUDA *backward* kernel too, so a CUDA
computation graph can be built and differentiated end-to-end on the GPU,
including through `nn.Module`/`Linear`/`ReLU`, with CUDA-resident gradients
and a CUDA-executing `SGD.step()` -- see **CUDA autograd** below. As of
**Milestone 14**, that graph extends through `nn.CrossEntropyLoss` too --
see **CUDA CrossEntropyLoss** below. As of **Milestone 15**, it extends
through `nn.Conv2d`/`nn.MaxPool2d` too -- see **CUDA Conv2d / MaxPool2d**
below. See `docs/architecture/backend-architecture.md` for the general
backend boundary this fits into.

## Package layout
```
forge/
    backend/
        base.py            Backend ABC (+ `to_numpy`, M8; `*_backward`/`sgd_step`, M10)
        cpu.py               CPUBackend (+ `to_numpy`, trivial identity; `*_backward` math, M10)
        __init__.py          get_backend() -- lazily imports forge.backend.cuda
        cuda/
            kernels.cu        CUDA C++ kernel source (compiled by build.py)
            build.py           Locates nvcc/MSVC, compiles kernels.cu -> a DLL
            backend.py         CUDAStorage, CUDABackend, get_cuda_backend(), is_cuda_available()
            memory.py            CUDAMemoryStats, allocation/free counters (M22)
            __init__.py        Re-exports the above
    cuda/__init__.py           forge.cuda: memory_stats(), reset_peak_memory_stats() (M22)
    tensor/tensor.py         Tensor.to(device); Tensor._move_storage_() (M9); backward() (device-generic as of M10)
    autograd/engine.py        run_backward -- backend-dispatched gradient accumulation (M10)
    nn/module.py              Module.to(device), Module.device (new in M9)
    nn/loss.py                MSELoss CUDA-compatible unmodified (M12); CrossEntropyLoss CUDA-compatible (M14)
    nn/conv.py, nn/pooling.py Conv2d/MaxPool2d -- backend-agnostic, dispatch via Tensor.conv2d/max_pool2d (M15)
    optim/sgd.py               SGD.step() -- Backend.sgd_step() dispatch (M10)
    training/trainer.py        Trainer device validation + explicit batch transfer (M12)
    training/metrics.py        Metric._as_numpy() transfers CUDA inputs to CPU for reporting (M12)
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
| `add`/`mul` | `cf_{add,mul}_{f32,f64}`, `cf_{add,mul}_bcast_{f32,f64}` | float32, float64 | Exact-shape, plus (Milestone 9) one row-broadcast shape: a `(rows, cols)` matrix combined with a `(cols,)` vector -- see **Row-broadcast** below |
| `sub` | `cf_sub_{f32,f64}`, `cf_sub_bcast_{f32,f64}`, `cf_sub_colbcast_{f32,f64}` | float32, float64 | Exact-shape, the same row-broadcast shape as `add`/`mul`, plus (Milestone 14) one column-broadcast shape: a `(rows, cols)` matrix combined with a `(rows, 1)` per-row scalar -- see **Column-broadcast** below |
| `matmul` | `cf_matmul_{f32,f64}` | float32, float64 | Shared-memory-tiled kernel (16x16 tiles, Milestone 11 -- see below); 1D vectors are reinterpreted as degenerate `M=1`/`N=1` 2D matmuls on the host side, matching the CPU backend's four 1D/2D cases |
| `sum` | `cf_sum_{f32,f64}`, `cf_sum_axis1_{f32,f64}` | float32, float64 | Full reduction (`axis=None`, block-level shared-memory tree reduction + atomic accumulation across blocks -- a CAS-based emulation for `atomicAdd(double*)`, since CC 5.0 has no native one), plus (Milestone 14) `axis=1`/`-1` on a 2D tensor -- one thread per row, see **CUDA CrossEntropyLoss** below |
| `reshape` | `cf_memcpy_d2d` | any transferable dtype | Implemented as a device-to-device copy into a freshly allocated buffer with new shape metadata (no in-place aliasing, to avoid any storage-ownership ambiguity without an allocator) |
| `relu` (Milestone 9) | `cf_relu_{f32,f64}` | float32, float64 | `out[i] = max(a[i], 0)`, one thread per element -- the same launch pattern as `add`/`sub`/`mul` |
| `exp`/`log` (Milestone 14) | `cf_exp_{f32,f64}`, `cf_log_{f32,f64}` | float32, float64 | One thread per element, via `expf`/`logf` (float) or `exp`/`log` (double) -- added for `CrossEntropyLoss`'s log-sum-exp; see **CUDA CrossEntropyLoss** below |
| `conv2d` (Milestone 15) | `cf_conv2d_forward_{f32,f64}` | float32, float64 | One thread per output element (`N*C_out*H_out*W_out`), looping over `C_in*KH*KW` in registers; see **CUDA Conv2d / MaxPool2d** below |
| `max_pool2d` (Milestone 15) | `cf_maxpool2d_forward_{f32,f64}` | float32, float64 | One thread per output element, looping over `KH*KW`; see **CUDA Conv2d / MaxPool2d** below |
| `dropout_mask` (Milestone 16) | `cf_dropout_mask_{f32,f64}` | float32, float64 | One thread per element; a stateless SplitMix64 hash of `(seed, element_index)` decides keep/drop -- no curand, no RNG state on-device; see **CUDA Dropout** below |

`relu` moved out of this list in Milestone 9; `exp`/`log` moved out of the
"required by the ABC but unsupported on CUDA" state they were in through
Milestone 13 -- see **CUDA CrossEntropyLoss** below for the full Milestone
14 story (why they were added, what else came with them, and how the
no-CPU-fallback guarantee still holds).

### Row-broadcast (Milestone 9)
M8's elementwise `add`/`sub`/`mul` required exact-matching shapes -- no CUDA
broadcasting at all. That is too narrow to run `nn.Linear`'s forward
(`x @ weight + bias`) on a *batched* input: the matmul result is `(batch,
out_features)`, the bias is `(out_features,)`. Rather than implementing
general N-dimensional broadcasting (out of scope), special-casing `Linear`
(the milestone brief explicitly disallows CUDA-specific code inside
`Linear`), or copying to CPU to broadcast there (a disguised CPU fallback),
`CUDABackend._elementwise` adds exactly the one broadcast shape `Linear`
needs: a 2D `(rows, cols)` matrix combined with a 1D `(cols,)` vector, computed
entirely on-device by `k_{add,sub,mul}_bcast` (`kernels.cu`) -- one thread per
output element, the vector value re-read for every row (`vec[i % cols]`). A
`vec_is_left` flag preserves operand order for the non-commutative `sub`.
Every other shape mismatch (including ordinary N-D broadcasting the CPU
backend supports) still raises `CUDAError` -- this is a targeted addition
for the shape Forge's own `Linear` produces, not general CUDA broadcasting
support. See `tests/test_cuda_backend.py` (`test_row_broadcast_*`,
`test_elementwise_broadcasting_beyond_the_row_case_is_unsupported_on_cuda`).

### Column-broadcast (Milestone 14)
The transpose problem, for `sub` only: `CrossEntropyLoss`'s log-sum-exp shift
(`logits - max_axis1(logits)`) and its log-probability step (`shifted -
log_sum_exp`) both combine a 2D `(rows, cols)` matrix with a `(rows, 1)`
per-row scalar, broadcasting that scalar across every *column* of its own
row -- the opposite direction from row-broadcast's `(cols,)`-vector-down-
every-row. `CUDABackend.sub` checks for this shape first
(`_col_broadcast_kind`) and, if found, dispatches to `k_sub_colbcast`
(`kernels.cu`) instead of `_elementwise`; every other shape still goes
through the row-broadcast/exact-shape path above unchanged. Only `sub` gained
this: nothing else in Forge ever combines a `(rows, cols)` operand with a
`(rows, 1)` one, so `add`/`mul` still raise `CUDAError` for that shape (see
`tests/test_cuda_backend.py::test_column_broadcast_add_and_mul_remain_unsupported_on_cuda`).
See **CUDA CrossEntropyLoss** below for the full picture this fits into.

Every kernel launch is followed by an explicit `cf_synchronize()`
(`cudaDeviceSynchronize()`) before its result is trusted or returned, so
CUDA's asynchronous execution model can never be mistaken for a completed
operation.

## Numerical consistency
`tests/test_cuda_consistency.py` runs the same inputs through both
backends for every shared operation (`add`/`sub`/`mul`/`matmul`/`sum`
(including `axis=1`, Milestone 14)/`reshape`/`relu`/`exp`/`log` (Milestone
14), both `float32` and `float64`, several matmul shape combinations, plus a
chained `Linear -> ReLU -> Linear`-shaped expression) and asserts
`np.testing.assert_allclose` agreement (`rtol=atol=1e-5`) -- never
bit-for-bit equality, per `docs/architecture/backend-architecture.md`.
`tests/test_module_cuda.py` runs the same comparison through the high-level
`nn.Module`/`Linear`/`ReLU` API (`rtol=atol=1e-4`, a looser tolerance
appropriate for a longer op chain).

## CUDA autograd (Milestone 10)
**CUDA autograd is supported** for every operation this backend implements
forward (see **Operation set** above): `add`/`mul` (exact-shape and the one
row-broadcast shape), `sub` (those two plus the Milestone 14
column-broadcast shape), `matmul` (1D/2D), `sum` (full reduction, plus
Milestone 14's `axis=1`), `reshape`, `relu`, and (Milestone 14) `exp`/`log`.
A CUDA computation graph now builds and
differentiates entirely on the GPU -- see `docs/architecture/autograd.md`'s
**Backend-aware backward dispatch** section for how `Tensor`'s backward
closures reach `CUDABackend`'s methods, and ADR-005 for why this replaced
the Milestone 8/9 "CUDA autograd is not supported" boundary rather than
adding a parallel mechanism.

### Backward kernels
| Backward rule | Kernel(s) | Notes |
|---|---|---|
| `add_backward`/`mul_backward` | `cf_neg_{f32,f64}`, `cf_reduce_rows_{f32,f64}`, plus the existing forward `add`/`mul`/`mul_bcast` kernels | `add`: identity for both operands (row-broadcast case reduces the vector operand's gradient with `cf_reduce_rows`). `mul`: reuses forward `mul` (elementwise or row-broadcast) to build each raw gradient, then `cf_reduce_rows` where a vector operand needs its gradient reduced |
| `sub_backward` | `cf_neg_{f32,f64}`, `cf_reduce_rows_{f32,f64}`, `cf_sum_axis1_{f32,f64}` (Milestone 14, for the column-broadcast case), plus the existing forward `sub`/`sub_bcast`/`sub_colbcast` kernels | Identity/`cf_neg` depending on operand order for the exact-shape and row-broadcast cases; the Milestone 14 column-broadcast case (see below) reduces the vector operand's gradient with `cf_sum_axis1` instead of `cf_reduce_rows` -- the same reduction `sum(axis=1)`'s own forward uses, reused here for the opposite reason |
| `matmul_backward` | `cf_scale_{f32,f64}`, `cf_transpose_{f32,f64}`, plus the existing forward `matmul`/`reshape` kernels | All four 1D/2D cases are built by composing existing kernels -- e.g. matrix·matrix's `grad_output @ b.T` and `a.T @ grad_output` reuse `matmul` on a freshly transposed operand; the 1D·1D (dot product) case uses `cf_scale`, a device-resident scalar-times-vector kernel that reads the scalar via a device pointer so no value ever crosses back to the host |
| `sum_backward` | `cf_broadcast_scalar_{f32,f64}` (`axis=None`), `cf_broadcast_axis1_{f32,f64}` (Milestone 14, `axis=1`) | `axis=None` broadcasts the one upstream scalar to every element of the original shape; `axis=1` broadcasts each row's own upstream scalar to every element of that row. Any other axis raises `CUDAError` -- forward `sum(axis=...)` already rejects it first |
| `reshape_backward` | (reuses forward `reshape`, `cf_memcpy_d2d`) | Reshaping the upstream gradient back to the original shape is exactly the forward `reshape` op run again |
| `relu_backward` | `cf_relu_backward_{f32,f64}` | `out[i] = input[i] > 0 ? grad_output[i] : 0` -- one kernel, no separate mask kernel, input never copied to CPU |
| `exp_backward`/`log_backward` (Milestone 14) | `cf_exp_backward_{f32,f64}`, `cf_log_backward_{f32,f64}` | `exp`: `out[i] = grad_output[i] * result[i]` (reuses exp's own saved forward output). `log`: `out[i] = grad_output[i] / input[i]` (reads the saved forward input). Each a small dedicated two-array kernel, matching `relu_backward`'s shape, rather than a generic elementwise-divide primitive nothing else needs |
| `conv2d_backward` (Milestone 15; weight/bias re-optimized in Milestone 21) | `cf_conv2d_backward_{input,weight,bias}_{f32,f64}` | `input`: one thread per input pixel (unchanged since Milestone 15). `bias`: always a block-per-channel shared-memory reduction (Milestone 21). `weight`: dispatches per-call between a block-per-weight-element reduction (few weight elements) and the original one-thread-per-weight-element kernel (many), based on a measured threshold -- see **CUDA Conv2d backward: weight/bias optimization (Milestone 21)** below. No atomics in any of the three |
| `max_pool2d_backward` (Milestone 15) | `cf_maxpool2d_backward_{f32,f64}` | One thread per *output* element, recomputing that window's argmax from the saved input and `atomicAdd`-ing into the (zeroed) input gradient -- see **CUDA Conv2d / MaxPool2d** below |

Every backward kernel launch is followed by the same explicit
`cf_synchronize()` convention the forward kernels use (see above) -- a
CUDA backward pass never returns a result CUDA hasn't actually finished
computing.

### Row-broadcast gradient reduction
The row-broadcast forward kernels (`add`/`sub`/`mul` combining a `(rows,
cols)` matrix with a `(cols,)` vector -- see **Row-broadcast** above)
produce a `(rows, cols)`-shaped upstream gradient when differentiated. The
matrix operand's gradient keeps that shape unchanged; the vector operand's
gradient must be reduced from `(rows, cols)` down to `(cols,)` by summing
over rows -- exactly the shape `nn.Linear`'s bias gradient needs from a
batched forward pass. `cf_reduce_rows` (`kernels.cu`) is a dedicated kernel
for exactly this one reduction (one thread per output column, looping over
rows) -- not a general CUDA axis-sum primitive, and not reachable for any
shape combination the row-broadcast forward kernels don't already support.

### Column-broadcast gradient reduction (Milestone 14)
The mirror image of the above, for `sub`'s column-broadcast case (see
**Column-broadcast** above): the `(rows, 1)` operand's gradient is reduced
from a `(rows, cols)` upstream gradient by summing over *columns* --
`CUDABackend._reduce_axis1`, which calls the exact same `cf_sum_axis1`
kernel `sum(axis=1)`'s own forward uses (see **CrossEntropyLoss** below),
then negates (`cf_neg`) and reshapes as the operand order requires. No new
kernel was needed for this reduction direction -- only for the forward
broadcast (`cf_sub_colbcast`) and the `sum(axis=1)` backward broadcast
(`cf_broadcast_axis1`) it composes with.

### `matmul_backward`'s helper kernels
Two new CUDA-only helpers back `matmul_backward`, used only through
`CUDABackend`'s own composition (they are not exposed as `Backend` forward
ops, since nothing in the M1/M8 matmul forward semantics needs them
directly):
- **`cf_transpose`**: a naive one-thread-per-element 2D transpose, needed
  for the matrix·vector, vector·matrix, and matrix·matrix backward cases
  (`a.T @ grad_output`, `grad_output @ b.T`).
- **`cf_scale`**: a device-resident scalar times a vector (`out[i] =
  *scalar * vec[i]`), needed for the 1D·1D (dot product) backward case
  (`grad_output * b`, `grad_output * a`, where `grad_output` is itself a
  0-d/scalar `CUDAStorage`). The scalar is read via a device pointer inside
  the kernel -- this is the same convention `sum_backward`'s
  `cf_broadcast_scalar` uses -- so no CUDA scalar value ever crosses back
  to the host as part of computing a gradient.

### CUDA `SGD.step()`
`SGD.step()` (`forge/optim/sgd.py`) dispatches to `Backend.sgd_step(data,
grad, lr)` rather than mutating `Parameter._data` with NumPy arithmetic
directly (the Milestone 4 implementation, which only worked for
`np.ndarray` storage). `CUDABackend.sgd_step` launches one kernel
(`cf_sgd_step_{f32,f64}`, `param[i] -= lr * grad[i]`) that updates the
existing parameter buffer **in place** -- no new `CUDAStorage` is
allocated, so a CUDA `Parameter`'s identity, device, dtype, and shape are
all untouched by an optimizer step, matching the CPU behavior exactly. `lr`
is passed as a plain launch argument (like `rows`/`cols` elsewhere in this
file), not read from device memory -- it is Python-side hyperparameter
state, not tensor data crossing the device boundary. The update is a plain
in-place mutation on both backends: it never attaches a `grad_fn` or
otherwise extends the autograd graph.

### `Tensor.backward()` device/dtype consistency
`Tensor.backward()` no longer restricts itself to `device.type == "cpu"`.
It still enforces device consistency the same way every other binary
Tensor operation does: an explicit upstream `gradient` argument must
already be on `self`'s own device (`UnsupportedDeviceError` otherwise), and
if its dtype doesn't match `self`'s dtype, CPU silently casts (matching
pre-Milestone-10 behavior) while CUDA raises `UnsupportedDTypeError` (no
implicit-cast compute path exists on CUDA). See
`docs/architecture/autograd.md`'s **Device consistency in `backward()`**
section.

### Interaction with `Module.to("cuda")` (Milestone 9; superseded boundary)
`Module.to(device)` (see `docs/architecture/modules.md`) preserves each
moved `Parameter`'s `requires_grad` flag -- it stays `True`. Through
Milestone 9, this meant a bare (non-`no_grad`) forward call through a
CUDA-resident model raised `UnsupportedDeviceError` immediately, and
`forge.no_grad()` was the only sanctioned way to run a CUDA model's forward
pass. **As of Milestone 10, a bare forward call now succeeds and builds a
real graph**, and `backward()` on the result now runs on CUDA and produces
CUDA-resident gradients:
```python
model.to("cuda")
x_cuda = x.to("cuda")
y = model(x_cuda)        # succeeds; builds a real CUDA graph
y.sum().backward()        # runs on CUDA; model.fc1.weight.grad.device.type == "cuda"
```
`forge.no_grad()` still works exactly as before (see
`docs/architecture/autograd.md`) -- it remains the way to run a CUDA
forward pass *without* building a graph (e.g. for inference), it is simply
no longer *required* to run one. See
`tests/test_module_cuda.py::test_cuda_full_model_forward_without_no_grad_now_builds_a_graph`
and `::test_cuda_model_backward_now_produces_cuda_resident_gradients`, and
`tests/test_cuda_autograd.py` for the full CUDA-autograd test suite.

### No CPU fallback
Every backward computation described above executes as a real CUDA kernel
against `CUDAStorage` operands -- never a NumPy computation against a
CPU-copied array. `tests/test_cuda_autograd.py`'s
`test_cuda_linear_backward_never_calls_cpu_backend` and
`test_cuda_multilayer_model_backward_never_calls_cpu_backend` assert this
structurally (via a monkeypatched `CPUBackend`, the same technique
`tests/test_module_cuda.py::test_cuda_model_forward_does_not_call_cpu_backend`
already used for forward execution): a full CUDA model forward + backward
pass calls zero `CPUBackend` methods.

## CUDA losses (Milestone 12)
### `MSELoss`: CUDA-compatible with zero new kernels
`MSELoss` (`forge/nn/loss.py`) computes `mean((prediction - target)^2)` as
```python
diff = prediction - target                                    # exact-shape sub
squared = diff * diff                                          # exact-shape mul
scale = Tensor(1.0 / n, dtype=prediction.dtype, device=prediction.device)
return squared.sum() * scale                                   # full-reduction sum, exact-shape mul
```
-- exclusively `-`, `*`, and `.sum(axis=None)`, every one of which
`CUDABackend` already implements forward *and* backward (Milestones 8-10;
see **Operation set** and **CUDA autograd** above). No new CUDA kernel, no
new `Backend` method, and no `CUDAMSELoss` subclass were needed: this is the
"determine the minimum CUDA primitives required" outcome the milestone
brief asked for turning out to be *zero*, because Milestone 8's own
operation-set choices (driven by what `Linear`'s forward pass needed)
already happened to cover what `MSELoss` needs.

The one real fix this milestone made: the scale is built as
`Tensor(1.0 / n, dtype=prediction.dtype, ...)` rather than the more obvious
`squared.sum() * (1.0 / n)`. `Tensor._coerce()` (`forge/tensor/tensor.py`)
turns a bare Python scalar into a `Tensor` using Forge's global default
dtype (`float32`) regardless of the *other* operand's dtype -- harmless on
CPU (`np.multiply` freely upcasts a dtype mismatch) but fatal on CUDA
(`CUDABackend._require_compute_dtype` requires exact dtype agreement and
raises `CUDAError` otherwise). A `float64` CUDA loss would otherwise fail on
its very last multiply. Building the scale with `prediction`'s own dtype
explicitly sidesteps `_coerce`'s inference and keeps both operands aligned,
for either compute dtype. This is a `loss.py`-local fix, not a change to
`Tensor._coerce()` itself -- the latter's default-dtype inference is
long-standing, exercised-elsewhere CPU behavior (e.g. `int_tensor + 1`
promoting `1` to `int64` rather than the tensor's own dtype) that this
milestone had no reason to touch.

`MSELoss` also already satisfies the milestone's **loss device validation**
requirement with no new code: `prediction - target` is an ordinary
`Tensor.__sub__` call, and `_binary_op`'s existing device-consistency check
(**Device transfer** above) rejects a CUDA/CPU operand mismatch with
`UnsupportedDeviceError` before any op-specific logic runs -- exactly the
same guarantee every other Tensor operation already has.

## CUDA CrossEntropyLoss (Milestone 14)
Milestone 12 deferred `CrossEntropyLoss` to CPU-only: it needed `.exp()`,
`.log()` (neither implemented on CUDA at the time), and an axis-wise
`logits.exp().sum(axis=1, keepdims=True).log()` (CUDA `sum()` supported only
a full reduction). This milestone implements exactly those primitives --
and one more the CPU implementation didn't need a dedicated abstraction for
-- so `CrossEntropyLoss` now runs unmodified on CUDA, using the same
high-level formulation as CPU (`forge/nn/loss.py`, no `CUDACrossEntropyLoss`
subclass, no second autograd engine): see
`docs/architecture/optimization.md`'s **CrossEntropyLoss** section for the
loss-level math this backs.

### The four primitives
1. **`exp`/`log`** -- real CUDA kernels now (see **Operation set** above and
   `kernels.cu`'s exp/log sections), each with a real backward
   (`exp_backward`/`log_backward`). `Tensor.exp()`/`Tensor.log()`
   (`forge/tensor/tensor.py`) were also fixed to dispatch their backward math
   through `Backend.exp_backward`/`Backend.log_backward` rather than a raw
   `grad_output * result` / `grad_output / input_data` -- those relied on
   NumPy operator overloading and would have raised `AttributeError` the
   first time a CUDA `grad_output`/`result` (a `CUDAStorage`, which defines
   no `__mul__`/`__truediv__`) reached them. `CPUBackend` gained the same two
   methods (trivial NumPy wrappers) so both backends implement the same
   `Backend` ABC surface.
2. **`sum(axis=1)`** -- `CUDABackend.sum()` now accepts `axis=1` (or `-1`,
   equivalently, since only 2D tensors reach this branch) on a 2D tensor, in
   addition to the existing `axis=None` full reduction; `cf_sum_axis1` is one
   thread per row, looping over that row's columns. Any other axis (`0`, a
   tuple, a non-2D tensor) still raises `CUDAError` -- this is *not* general
   N-D axis reduction, only the one axis `CrossEntropyLoss` (or anything else
   shaped `(batch, classes)`) needs. `sum_backward` gained the mirror-image
   `cf_broadcast_axis1` kernel for this case.
3. **Column-broadcast `sub`** -- see **Column-broadcast** and **Column-
   broadcast gradient reduction** above. Needed because `logits -
   max_axis1(logits)` and `shifted - log_sum_exp` each combine a `(batch,
   classes)` matrix with a `(batch, 1)` per-row scalar, which is not a shape
   `_elementwise`'s existing row-broadcast case (a `(cols,)` vector broadcast
   down every row) handles.
4. **`max_axis1`** -- *not* a `Tensor`-level operation (no `Tensor.max()` was
   added; the milestone brief explicitly scopes this to what
   `CrossEntropyLoss` needs, not a general reduction API). It is a plain
   `Backend` method (`CPUBackend.max_axis1`: `np.max(a, axis=1,
   keepdims=True)`; `CUDABackend.max_axis1`: a dedicated `cf_max_axis1`
   kernel, one thread per row) that `CrossEntropyLoss.forward()` calls
   directly on `logits._data`, wrapping the result as a `requires_grad=False`
   leaf via `Tensor._wrap`. This mirrors exactly what the CPU implementation
   always did (`np.max(logits.numpy(), axis=1, keepdims=True)`, also wrapped
   as a non-differentiable constant) -- the only change is *how* the max is
   computed, never *that* it's treated as a constant (the log-sum-exp
   identity `logsumexp(x - c) == logsumexp(x) - c` for any per-row `c` makes
   this exact, not an approximation). Critically, this keeps the max
   computation **on-device for CUDA**: `logits.numpy()` is never callable on
   a CUDA tensor in the first place (`UnsupportedDeviceError`), and even if
   it were, reading the max on host and transferring it back would be exactly
   the "silent CPU fallback for a real computation" this milestone forbids.

### Target handling and device validation
`target` (a `Tensor` or array-like of integer class indices) follows the
same explicit-device-consistency rule as everywhere else in Forge: if
`target` is a `Tensor`, its device must equal `logits`'s device or
`CrossEntropyLoss.forward()` raises `UnsupportedDeviceError` immediately
(before any computation) -- CUDA logits with a CPU target, or vice versa, is
never silently reconciled. Once validated, the (small, integer) target
values are read to host via `get_backend(target.device).to_numpy(...)` --
this is metadata/index preparation for the one-hot construction and
range/dtype validation below, exactly the same category of read-only
transfer `Trainer`/`Metric`/persistence already use for non-computational
purposes (see **No CPU fallback** above), never a stand-in for computing the
loss on CPU. The one-hot mask itself is then built as an ordinary
`Tensor(one_hot, dtype=logits.dtype, device=logits.device)`, so the actual
`(log_probs * one_hot).sum(axis=1)` selection runs as a real device
operation on whichever device `logits` is on. A non-`Tensor` target (e.g. a
raw NumPy class-index array, `CrossEntropyLoss`'s longstanding convention)
carries no device of its own and is used as-is regardless of `logits`'s
device, unchanged from before this milestone.

### No CPU fallback
`tests/test_cuda_loss.py::test_cross_entropy_cuda_forward_and_backward_never_call_cpu_backend`
asserts this structurally (the same monkeypatched-`CPUBackend` technique
used throughout this document), covering every compute method this loss can
reach on CUDA: `add`/`sub`/`mul`/`sum`/`exp`/`log`/`exp_backward`/
`log_backward`/`max_axis1`. A full forward + backward pass calls zero of
them. `tests/test_trainer_cuda.py::test_cuda_trainer_classification_never_calls_cpu_backend_compute_ops`
extends the same check through a full `Trainer(device="cuda")` classification
`fit()` call (adding `matmul`/`reshape`/`relu`/`sgd_step` to the spied set,
matching the regression no-fallback test).

### Numerical correctness and gradient verification
`tests/test_cuda_loss.py` compares CUDA and CPU forward output (`float32`
and `float64`, plus a battery of numerically difficult logits: large
positive, large negative, large inter-class differences, repeated/equal
logits, a single-sample batch, and seven-class logits) within tolerance,
verifies CUDA backward matches CPU backward and independently matches the
closed form `(softmax(logits) - one_hot(target)) / batch_size`, includes a
finite-difference gradient check across several batch/class-count
combinations, and verifies the mean-reduction/`1/batch_size` gradient
scaling explicitly (doubling the batch by duplicating rows halves each
duplicated row's gradient contribution). See that file's module docstring
for the full list.

### Trainer integration
`Trainer` needed no changes at all: `trainer.fit()`/`evaluate()` call
`self.loss_fn(prediction, y)` exactly as before, and whether that succeeds on
CUDA has always been the loss's own concern (see
`docs/architecture/training-engine.md`'s **CUDA losses through Trainer**
section). `tests/test_trainer_cuda.py::test_cuda_trainer_classification_end_to_end_learns`
exercises the full milestone-brief workflow -- `TensorDataset -> DataLoader
-> Trainer(device="cuda") -> Linear -> CrossEntropyLoss -> CUDA backward ->
CUDA SGD` -- on a small deterministic two-class dataset, confirming loss
drops substantially, accuracy exceeds 90%, and every parameter/gradient
stays CUDA-resident throughout.

## CUDA Conv2d / MaxPool2d (Milestone 15)
`nn.Conv2d`/`nn.MaxPool2d` (`forge/nn/conv.py`, `forge/nn/pooling.py`) are
backend-agnostic Modules, exactly like `Linear`/`ReLU`: their `forward()`
calls `Tensor.conv2d()`/`Tensor.max_pool2d()` (`forge/tensor/tensor.py`),
which dispatch to `Backend.conv2d`/`Backend.conv2d_backward`/
`Backend.max_pool2d`/`Backend.max_pool2d_backward` -- the same
Tensor -> `grad_fn` -> `Backend` pattern every other differentiable op
uses (`docs/architecture/autograd.md`). No CUDA-specific code exists in
`nn/conv.py`/`nn/pooling.py`.

Unlike `CPUBackend`'s im2col-plus-matmul implementation (a vectorized
`sliding_window_view` reduced with NumPy/BLAS), `CUDABackend`'s kernels are
deliberately the "straightforward correct kernel" the milestone brief asks
for: one thread per output element (forward) or per gradient-target element
(backward), looping over the small kernel window in registers -- no im2col
buffer, no cuBLAS, no cuDNN, no tiling. This is a *different* algorithm from
the CPU path, not a shared numerical core, matching how CUDA's `matmul`
(tiled GEMM) and CPU's `matmul` (`np.matmul`/BLAS) are already unrelated
implementations of the same forward contract.

- **`conv2d` forward** (`k_conv2d_forward`): one thread per `(n, c_out, h_out,
  w_out)`, accumulating over `c_in`/`kh`/`kw` with an explicit in-bounds
  check per tap (the zero-padding boundary condition).
- **`conv2d` backward** is three separate kernels, each a plain gather (no
  atomics): `k_conv2d_backward_input` (one thread per input pixel, summing
  over every output window that read it), `k_conv2d_backward_weight` (one
  thread per weight element, summing over every batch/spatial position it
  touched), and `k_conv2d_backward_bias` (one thread per output channel,
  summing `grad_output` over batch and spatial dims). `bias` is optional at
  every layer (a null device pointer, `has_bias=0`, when `Conv2d(bias=False)`)
  -- `conv2d_backward` skips the bias kernel entirely rather than allocating
  and returning a zero gradient.
- **`max_pool2d` forward** (`k_maxpool2d_forward`): one thread per output
  element, scanning its `KH x KW` window with a strict `v > best` comparison
  -- so the *first* maximum encountered in row-major (`kh` outer, `kw` inner)
  scan order wins ties, never a later equal value.
- **`max_pool2d` backward** (`k_maxpool2d_backward`) recomputes each output
  element's argmax from the saved forward input `x` (the same
  "recompute from a saved input" convention `k_relu_backward`/
  `k_exp_backward` already use) rather than caching indices from the forward
  pass, then `atomicAdd`s the upstream gradient into that one input
  position. The atomic is necessary here (unlike `conv2d_backward`'s input
  kernel): overlapping windows (`stride < kernel_size`) can select the same
  input element as more than one output window's argmax, so more than one
  thread can write to the same `grad_x` element. `grad_x` is
  `cudaMemset`-zeroed by the launcher before any thread writes to it, the
  same convention `cf_sum_*`'s launcher already uses for its output
  accumulator.

**Tie-breaking agreement.** `CPUBackend.max_pool2d_backward` (`forge/backend/cpu.py`)
applies `np.argmax` to each window flattened in the same row-major
(`kh`-then-`kw`) order the CUDA kernel scans in, and `np.argmax` is
documented to return the *first* occurrence of the maximum -- so both
backends select the identical element on a tie. `tests/test_pooling.py`
and `tests/test_cuda_conv.py` both assert the same concrete tie-break
example (`test_maxpool2d_tie_breaks_row_major_not_column_major` /
`test_cuda_maxpool2d_tie_break_matches_cpu_convention`).

**No CPU fallback.** `tests/test_cuda_conv.py::test_cuda_conv2d_never_calls_cpu_backend`
and `::test_cuda_maxpool2d_never_calls_cpu_backend` monkeypatch every
`CPUBackend` method (the same spy pattern `test_cuda_autograd.py` already
uses for `Linear`/multilayer models) and assert the recorded call list is
empty across a full forward + backward pass.

**Explicit non-goals kept out of scope.** No dilation, no groups, no
transposed convolution, no average/adaptive/global pooling, no cuDNN/cuBLAS,
no autotuning, no return-indices API -- see the milestone brief's "Explicit
Non-Goals" list. `docs/architecture/backend-architecture.md`'s
"straightforward kernel first, optimize later" precedent (Milestone 8-11's
matmul) applies here too: these kernels are correctness-first, and
`benchmarks/ops_bench.py`/`benchmarks/backward_bench.py`'s `conv2d`
entries exist to *measure* that, not to claim CUDA already beats the CPU
im2col-plus-BLAS path at every scale on the 940MX.

## CUDA Dropout (Milestone 16)
`nn.Dropout` (`forge/nn/dropout.py`) is backend-agnostic, exactly like
`Linear`/`ReLU`/`Flatten`: `forward()` composes `x * x.dropout_mask(p, rng)`
(`Tensor.dropout_mask`, `forge/tensor/tensor.py`), which dispatches to a new
`Backend.dropout_mask(a, p, rng)` method. No CUDA-specific code exists in
`nn/dropout.py`, and no Dropout-specific backward kernel exists either --
`x * mask` is ordinary `mul`, whose existing `mul_backward` (this document's
**CUDA autograd** section) already gives the correct gradient. The only new
CUDA work this milestone needed was mask *generation*.

### Why not curand
The milestone brief explicitly asks for "a simple correctness-first CUDA
implementation," not "implement a sophisticated GPU RNG library" -- so
Dropout does not link against `curand`. Instead, `CUDABackend.dropout_mask`
draws **exactly one** integer seed from the caller's `rng`
(`numpy.random.Generator`, `forge.random`'s default or an explicit one) --
a cheap, one-time host-side scalar draw, not per-element randomness -- and
passes it to a real CUDA kernel, `cf_dropout_mask_{f32,f64}`
(`kernels.cu`'s **Dropout mask** section). That kernel generates every
element's Bernoulli(1-p) draw independently, entirely on-device, with one
thread per element:

```text
h = splitmix64(seed XOR splitmix64(element_index))
u = (h >> 11) * 2^-53          # uniform double in [0, 1)
mask[i] = (u >= p) ? 1/(1-p) : 0
```

`cf_splitmix64` is the standard SplitMix64 finalizer (Vigna, 2015): a
stateless, statistically well-distributed hash -- not a cryptographic or
general-purpose GPU RNG, just enough to make each thread's draw look
independent of its neighbors given a fixed `seed`. No RNG *state* is
allocated, stored, or freed anywhere on the device: every thread computes
its own draw purely as a function of `(seed, i)`, so there is nothing to
initialize (no `curandState` array, no per-thread setup kernel) and the
whole mechanism stays narrowly scoped to Dropout, per the milestone brief.

### No CPU fallback
Per-element randomness never touches NumPy or the host for a CUDA tensor:
`CPUBackend.dropout_mask` (`rng.random(a.shape)`, an ordinary NumPy draw) is
a completely separate implementation, used only when `a`'s device is CPU.
`tests/test_cuda_dropout.py::test_cuda_dropout_never_calls_cpu_backend`
monkeypatches every `CPUBackend` method (the same spy pattern
`test_cuda_conv.py` uses) and asserts the recorded call list is empty
across a full forward + backward pass.

### CPU/CUDA consistency: statistical, not bitwise
Unlike every other CUDA operation in this document (which is checked against
CPU within floating-point tolerance for the *same* inputs), Dropout's CPU
and CUDA masks are **not** expected to agree element-for-element, even
given the same `forge.random.seed(...)` -- NumPy's `Generator.random()` and
`cf_dropout_mask`'s SplitMix64-based kernel are different algorithms
consuming the seed differently. `tests/test_cuda_dropout.py`'s
`test_cpu_and_cuda_dropout_agree_statistically_not_bitwise` makes this
explicit: it asserts the two masks are *not* bit-for-bit equal, then checks
both realize the same `fraction_zeroed ≈ p` distribution. Every other CUDA
Dropout test compares shape/dtype, statistical behavior (`fraction_zeroed`,
`mean`), gradient correctness (`grad == mask` pattern), and eval-mode exact
identity -- never exact per-element forward values against CPU.

### Randomness lifecycle
`Dropout.forward()` fetches `forge.random.default_generator()` **fresh on
every call** (unless an explicit `generator=` was passed at construction),
so a single `forge.random.seed(...)` governs every Dropout draw across an
entire training run, on either device -- see
`docs/architecture/modules.md`'s **Dropout** section for the full
randomness-lifecycle writeup (this differs from `Linear`/`Conv2d`, which
snapshot a generator once at construction for their one-time weight init).

## CUDA Adam (Milestone 17)
`Adam.step()` (`forge/optim/adam.py`) dispatches to `Backend.adam_step(data,
grad, m, v, lr, beta1, beta2, eps, weight_decay, step)`, mirroring `SGD`'s
`Backend.sgd_step` boundary exactly. `CUDABackend.adam_step` launches **one**
kernel (`cf_adam_step_{f32,f64}`, `kernels.cu`) that computes the entire
update -- both moment estimates, bias correction, and the parameter step --
directly against the existing `data`/`m`/`v` `CUDAStorage` buffers, in
place; no new `CUDAStorage` is allocated by the step itself and no tensor
value crosses the device boundary mid-update. The two bias-correction
scalars (`1 - beta1**step`, `1 - beta2**step`) are the only values computed
on the host, in Python -- identical for every element in a given call, so
this is a cheap scalar `pow()` on hyperparameter state, the same convention
`k_broadcast_scalar` and `CrossEntropyLoss`'s per-call scalars already use,
not a per-element host computation.

### Optimizer state is real CUDA storage
A CUDA `Parameter`'s Adam state (`m`, `v`) is allocated via
`Backend.from_array(np.zeros(...), dtype)` on first `step()` -- for
`CUDABackend` this is a real `cudaMalloc` plus one host-to-device transfer of
zeros (the same mechanism any new CUDA tensor's construction already uses,
e.g. `Parameter(data, device="cuda")` itself), never a NumPy array
relabeled as CUDA state. `tests/test_cuda_optimizer.py::
test_adam_cuda_state_is_cuda_resident` asserts `state.m`/`state.v` are
`CUDAStorage` instances, not `np.ndarray`.

### No CPU fallback
Every numerical step for a CUDA parameter -- moment updates, bias
correction, and the parameter update -- executes as the one
`cf_adam_step_{f32,f64}` kernel launch against `CUDAStorage` operands; no
CUDA gradient is ever copied to the host for computation, and no NumPy
arithmetic runs against it.
`tests/test_cuda_optimizer.py::test_adam_cuda_grad_and_param_never_leave_device_mid_step`
asserts this structurally (a monkeypatched `CPUBackend.adam_step`, the same
spy pattern `test_cuda_autograd.py`/`test_cuda_dropout.py` already use,
records zero calls across a full CUDA `Adam.step()`).

### CPU/CUDA numerical agreement
`tests/test_cuda_optimizer.py` compares CPU and CUDA Adam against matched
initial parameters and matched gradient sequences: a single step, eight
steps with `weight_decay` enabled, and a real `Linear` model trained five
steps through both devices in lockstep -- all agree within `rtol=atol=1e-4`.

### Device transfer after optimizer state exists
Adam's Policy-A device-mismatch guard (see `Adam`'s docstring in
`forge/optim/adam.py` and `docs/architecture/optimization.md`'s **Adam**
section) is exercised on real hardware in
`tests/test_cuda_optimizer.py::test_adam_state_device_mismatch_after_module_to_raises`
(a CPU-created Adam state followed by `model.to("cuda")` raises
`OptimizerError` on the next `step()`, rather than silently pairing a CUDA
parameter with CPU-resident `m`/`v`) and
`::test_adam_state_cleared_after_move_reinitializes_on_new_device` (clearing
the stale entry lets `step()` lazily allocate fresh, CUDA-resident state).

## CUDA Conv2d backward: weight/bias optimization (Milestone 21)
Milestone 21's purpose was measurement-driven optimization, not speculative
kernel rewrites: `benchmarks/mnist_profile.py` (new this milestone -- see
`docs/performance/benchmarking.md`'s **Milestone 21** section) broke the
real M20 CNN's CUDA training step into per-phase and per-op timings and
found `conv2d`'s *backward* pass was the single largest contributor --
73% of total backward time and 54% of the entire training step, on the
940MX, at the fixed `batch=64` MNIST configuration. Isolating the three
`conv2d_backward` kernels individually (`cf_conv2d_backward_{input,weight,bias}_*`)
at the CNN's own two layer shapes narrowed this further:

| Layer | Shape | `input` | `weight` (before) | `bias` (before) | Full `conv2d_backward` (before) |
|---|---|---|---|---|---|
| conv1 | N=64,Cin=1,Cout=8,H=28,W=28,K=3 | 1.08ms | 7.61ms | 3.32ms | 12.62ms |
| conv2 | N=64,Cin=8,Cout=16,H=13,W=13,K=3 | 3.50ms | 1.37ms | 0.56ms | 7.20ms |

**Hypothesis.** `k_conv2d_backward_weight`/`k_conv2d_backward_bias`
(Milestone 15) launched one CUDA *thread* per output element (one thread
per `(co, ci, kh, kw)` weight, one thread per `co` bias channel), each
thread serially summing the *entire* `N * Hout * Wout` reduction alone. At
conv1's shape that is only 72 weight threads and 8 bias threads -- each
doing a 64 x 26 x 26 = 43,264-iteration serial loop -- while the 940MX's
other ~300+ CUDA cores sat idle for the whole kernel. This is a classic
under-parallelized-reduction pattern, not a memory-access or algorithmic
problem, so a standard block-per-output-element, shared-memory
tree-reduction restructuring (the same reduction shape `k_sum`, above,
already uses) was the natural fix -- not im2col, not cuDNN, per the
milestone's explicit non-goals.

**Change.** `forge/backend/cuda/kernels.cu` gained
`k_conv2d_backward_weight_reduce`/`k_conv2d_backward_bias_reduce`: one
256-thread block per output element (a weight, or a bias channel), each
thread striding through a slice of the `N * Hout * Wout` reduction via a
grid-stride loop, combined by a shared-memory tree reduction (structurally
identical to `k_sum`), with the single output value written by thread 0 --
no atomics needed, since exactly one block ever owns a given output
element. `cf_conv2d_backward_bias_{f32,f64}` always dispatches to the new
reduction kernel: every measured/plausible bias-channel count for a Forge
CNN (8, 16, ...) is small enough that the reduction kernel wins.

**Weight gradients needed a second measurement, not just the first.**
Re-measuring after switching *both* kernels unconditionally to the
reduction strategy showed conv1's `weight` kernel improved sharply
(7.61ms -> 1.89ms) but conv2's got *worse* (1.37ms -> 5.48ms): at 1,152
weight elements, the original one-thread-per-weight kernel already had
enough threads to occupy the GPU well, and paying 1,152 blocks' worth of
`__syncthreads()`/shared-memory-reduction overhead for an already-short
(7,744-iteration) per-thread reduction lost to the simpler kernel. So
`cf_conv2d_backward_weight_{f32,f64}` **dispatches between the two
kernels** at launch time, based on the weight-element count (`total =
Cout * Cin * KH * KW`) against a fixed threshold (256, chosen to sit
between the two measured cases: 72 and 1,152) -- below it, the new block
reduction; at or above it, the original Milestone 15 one-thread-per-weight
kernel (kept, unchanged, in `kernels.cu`, not deleted). This hybrid
dispatch is itself a measured decision, not a guess: a single strategy
that looked correct for one MNIST layer was measurably wrong for the
other, and only re-measuring both shapes after the first attempt caught
it. `cf_conv2d_backward_input_*` (`k_conv2d_backward_input`) was left
completely unchanged: it already launches one thread per *input* pixel
(tens of thousands at every shape this milestone measured), so it was
never the measured bottleneck.

Both exported symbol names and signatures
(`cf_conv2d_backward_{weight,bias}_{f32,f64}`) are unchanged -- this is
purely an internal kernel-algorithm change behind the existing
`CUDABackend.conv2d_backward` boundary (`forge/backend/cuda/backend.py`
required no Python-side changes), matching the precedent Milestone 11's
matmul re-tiling and Milestone 15's own Conv2d/MaxPool2d kernels set.

**Correctness.** All 906 tests pass unchanged after the kernel rewrite and
recompile, including `tests/test_cuda_conv.py`'s CPU/CUDA `conv2d`
backward-agreement and finite-difference checks (both MNIST-scale and the
three `CONV2D_CONFIGS` scales, none of which happen to sit exactly at the
72/1,152-element boundary but which exercise both branches of the hybrid
dispatch across `tiny`/`small`/`medium`) and `tests/test_cuda_persistence.py`/
`tests/test_mnist_example_cuda_integration.py`'s real trained-model
round trips. No CPU code changed; `CPUBackend.conv2d_backward` is
byte-for-byte unmodified.

**Measured result (940MX, real hardware).** Full `conv2d_backward` call
(input + weight + bias together, matching what a real backward pass
invokes):

| Layer | Before | After | Speedup |
|---|---|---|---|
| conv1 (N=64,Cin=1,Cout=8,H=28,W=28,K=3) | 12.62ms | 3.64-3.73ms | ~3.4-3.5x |
| conv2 (N=64,Cin=8,Cout=16,H=13,W=13,K=3) | 7.20ms | 6.71-10.34ms (run-to-run noise; consistently <= before) | ~1.0-1.1x (bias-reduction gain only; weight kernel unchanged at this shape) |

End-to-end (`benchmarks/mnist_profile.py`, batch=64, 30 iterations, 5
warmup; see `docs/performance/benchmarking.md` for the full table): the
CUDA training step's `backward` phase dropped from 25.29ms to 16.10ms
mean, and `conv2d`'s share of that dropped from 18.66ms to 9.58ms --
roughly halved. Total CUDA training-step time dropped from 34.46ms to
25.53ms (~1.35x), and the real end-to-end MNIST throughput benchmark
(`benchmarks/mnist_bench.py`) moved from ~1,875 to ~2,569 samples/sec on
CUDA (~1.37x) at this fixed configuration -- the same order of speedup,
confirming the isolated-kernel gain actually reaches end-to-end training
throughput rather than being absorbed elsewhere. CPU throughput was not
touched by this change (`CPUBackend.conv2d_backward` unmodified); its own
run-to-run measurement varied by more than this optimization's CUDA gain
on this shared, non-dedicated development machine, so CPU numbers are
reported as an unrelated baseline, not a comparison point for this specific
optimization -- see `docs/performance/benchmarking.md` for the full
before/after tables and that caveat's full reasoning.

**Why no ADR.** Same reasoning as Milestone 11's matmul re-tiling and
Milestone 15's Conv2d/MaxPool2d kernels: this changes kernel internals
behind an unchanged `Backend`/`CUDABackend` boundary and an unchanged
exported C symbol/signature -- no public API, abstraction boundary, or
cross-cutting architectural decision was touched.

## CUDA Memory Statistics (Milestone 22)
Milestone 22 is observability, not optimization: it instruments the
`cudaMalloc`/`cudaFree` boundary that already existed (`CUDABackend._alloc`,
`CUDAStorage.__del__`, above) so Forge can answer "how much CUDA memory is
live," "what was the peak," and "did this workload leak" -- without
introducing a caching allocator, memory pool, or any change to the
allocate-on-every-op model described at the top of this document.

### Public API
```python
forge.cuda.memory_stats()             # -> CUDAMemoryStats(allocated_bytes, peak_allocated_bytes, allocation_count, free_count)
forge.cuda.reset_peak_memory_stats()  # resets peak only; live allocations untouched
```
`forge.cuda` (`forge/cuda/__init__.py`) is a thin public package fronting
`forge.backend.cuda.memory` (the actual counters), mirroring how
`forge.optim`/`forge.serialization` front `forge.backend`-internal pieces.
Both functions raise `forge.CUDAError` if CUDA is unavailable on this
machine -- the same convention every other CUDA-specific entry point in
Forge already follows (e.g. `Tensor(..., device="cuda")` on a machine with
no GPU) -- rather than returning a misleading all-zero snapshot. Importing
`forge.cuda` itself never requires CUDA: `forge.backend.cuda.memory` is pure
Python (`threading`, `dataclasses`, no `ctypes`/`nvcc`/device probe), so
`import forge` remains CUDA-optional, unchanged from every earlier
milestone.

### Instrumentation point
`CUDAMemoryStats` is a frozen dataclass; `forge/backend/cuda/memory.py`
holds one `threading.Lock`-guarded `_MemoryTracker` (a process-wide
singleton) with `record_alloc(nbytes)`/`record_free(nbytes)`. Exactly two
call sites use it:
- `CUDABackend._alloc()` calls `record_alloc(nbytes)` immediately after
  `cf_malloc` returns success -- never before, and never on the `raise`
  path for a failed allocation (see **Allocation failure semantics** below).
- `CUDAStorage.__del__()` calls `record_free(self.nbytes)` immediately after
  `cf_free` returns success.

No other code path touches these counters. In particular, `CUDAStorage.
__init__` and `Tensor` construction are never involved -- the milestone
brief's "instrument the real boundary, not Tensor construction" requirement
-- so the counters describe actual device allocations, not how many Tensor
objects happen to exist.

The tracker holds only integers, never a `CUDAStorage` reference: keeping
one would keep that allocation alive forever (the exact leak the milestone
brief warns a naive "registry" design could introduce), and would be a
reference cycle Forge's own `__del__`-based free depends on being absent.

### Statistics semantics
- **`allocated_bytes`**: sum of `CUDAStorage.nbytes` (`size * dtype.itemsize`)
  across every currently live `CUDAStorage` -- not the raw `cudaMalloc`
  request size, which `CUDABackend._alloc` clamps to a minimum of 1 byte for
  a zero-element tensor. A zero-element CUDA tensor therefore contributes 0
  to `allocated_bytes` but still advances `allocation_count`/`free_count` by
  one each, since a real (1-byte) `cudaMalloc`/`cudaFree` pair still occurs.
- **`peak_allocated_bytes`**: the historical maximum of `allocated_bytes`
  since process start or the most recent `reset_peak_memory_stats()` --
  updated inside the same locked region as `record_alloc`, so it can never
  observe a stale `allocated_bytes` value.
- **`allocation_count`/`free_count`**: count of successful `cudaMalloc`/
  `cudaFree` calls made through `CUDABackend`/`CUDAStorage`. A failed
  `cudaMalloc` is never counted (see below); a failed `cudaFree` is never
  counted as a free either (see below).
- **`reset_peak_memory_stats()`**: sets `peak_allocated_bytes` to the
  *current* `allocated_bytes` -- it does not free anything, and does not
  touch `allocated_bytes`, `allocation_count`, or `free_count`.

### Allocation failure semantics
`CUDABackend._alloc()` calls `record_alloc()` only after `cf_malloc` returns
success; the existing `raise CUDAError(...)` path for a nonzero return code
returns before that call, so a failed allocation leaves every counter
byte-for-byte unchanged. Verified on real hardware
(`tests/test_cuda_memory.py::test_failed_allocation_does_not_corrupt_statistics`)
by requesting 2**34 bytes (16 GiB) on the 940MX's 2 GiB card and asserting
`memory_stats()` is identical before and after.

`CUDAStorage.__del__` mirrors this for frees: `record_free()` runs only if
`cf_free` returns 0. A nonzero return instead emits a `RuntimeWarning`
naming the CUDA error -- not a silent no-op, per the milestone's "do not
silently swallow allocation/free failures" requirement -- but does not raise,
since raising inside `__del__` is a Python anti-pattern (CPython prints
"Exception ignored in..." and continues regardless; a raised exception here
cannot be caught by any caller). `self.ptr` is always set to `None` after
the `cf_free` call (success or failure) so a second `__del__` invocation
(not expected in normal operation, but structurally guarded against) can
never double-free or double-decrement.

### Known limitations
Two genuine hardware/architecture findings surfaced while testing this
milestone. Neither is a Forge accounting bug, and fixing either is out of
M22's instrumentation-only scope -- both are documented here instead.

**1. A sufficiently large failed `cudaMalloc` poisons kernel launches for the
rest of the process, on this hardware/driver combination.** Empirically, on
the 940MX (driver 582.53, CUDA 12.6), requesting an allocation far beyond
the card's 2 GiB VRAM (e.g. 2\*\*34 bytes) fails cleanly and leaves
`cudaMalloc`/`cudaMemcpy` themselves still working -- but every subsequent
*kernel launch* (`add`, `relu`, `matmul`, anything) in that same process then
fails with the same `cudaErrorMemoryAllocation` (code 2), even for a
trivial, few-byte operation. This is a real CUDA-context-level driver
behavior, not something Forge's accounting causes or could paper over.
Because of this, `test_failed_allocation_does_not_corrupt_statistics` runs
its provoking allocation inside an isolated **subprocess**
(`subprocess.run([sys.executable, "-c", ...])`) rather than in the main
`pytest` process -- provoking it directly in-process would silently corrupt
every CUDA test that runs afterward, in this file and any other, for the
rest of that `pytest` invocation.

**2. (Resolved in Milestone 23.) Forge's Tensor/autograd/Module/Optimizer
object graph contained a genuine Python reference cycle, so CUDA memory
release depended on Python's cyclic garbage collector, not refcounting
alone.** `run_backward` (`forge/autograd/engine.py`) already cleared
`tensor._grad_fn = None` for every node it consumed, and no `backward_fn`
closure captured an output `Tensor` (each captures raw backend storage
instead -- see `docs/architecture/autograd.md`) -- the Tensor/Node ownership
graph itself was, and remains, acyclic. Empirically, though, running
`gc.disable()` then a normal `zero_grad -> forward -> loss -> backward ->
step` loop for 10 iterations, followed by one manual `gc.collect()`,
reclaimed several hundred otherwise-unreachable `Tensor`/`CUDAStorage`/
closure (`function`/`cell`) objects that plain refcounting left live --
measured on this hardware, 50 SGD iterations of a small MLP inflated
`allocated_bytes` from a ~1.2MB steady state to ~3.1MB before any
`gc.collect()`, dropping back to ~96KB (the model's true persistent
footprint) after one.

Milestone 23 root-caused this: not the Tensor/Node graph, but a **recursive
nested closure** in `_topological_order` (`forge/autograd/engine.py`) whose
`def visit(tensor): ... visit(inp) ...` captured its own name in its
closure cell -- `visit.__closure__` held a cell referencing `visit` itself,
a genuine self-referential cycle, which also kept the entire per-call
topological-order list (every `Tensor` in the graph) alive until the next
`gc.collect()`. See `docs/architecture/autograd.md`'s **Graph teardown and
object lifetime (Milestone 23)** section for the full cycle diagram and fix
(an iterative, non-recursive traversal with no nested function). Re-running
the same 50-iteration measurement post-fix shows zero `allocated_bytes`
growth across iterations with `gc.collect()` never called at all --
`tests/test_cuda_lifetime.py` is the permanent regression test for this.

Every lifecycle test in `tests/test_cuda_memory.py`, and the benchmark
integration below, still call `gc.collect()` immediately before each
snapshot -- this is now purely defensive (establishing a clean baseline
against an unrelated, memory-free `ctypes` artifact documented in
`docs/architecture/autograd.md`'s **Known limitations**, not because
`CUDAStorage` release depends on it any more) and is left as-is rather than
removed, since a `gc.collect()` immediately before a snapshot is harmless
and these tests' job is measuring `allocated_bytes`, not re-proving M23's
fix (`test_cuda_lifetime.py` does that instead).

### Thread safety
`_MemoryTracker` guards its four counters with one `threading.Lock`,
acquired once per `record_alloc`/`record_free`/`stats`/`reset_peak` call --
the smallest mechanism that keeps concurrent callers from corrupting the
counters. Forge is single-threaded everywhere else, so this is not a
general concurrency subsystem, just cheap insurance.

### No synchronization added
Recording an allocation/free is pure Python bookkeeping around an already-
completed `cf_malloc`/`cf_free` call -- no new `cudaDeviceSynchronize()` was
added anywhere. Every kernel-launching operation already synchronizes
internally before trusting its own result (see **Operation set** above);
memory accounting adds no additional synchronization point.

### Benchmark integration
`benchmarks/memory.py`'s `cuda_memory_extra(before, after)` turns a
before/after `CUDAMemoryStats` pair into a dict (`cuda_allocated_before_bytes`,
`cuda_peak_allocated_bytes`, `cuda_allocated_after_bytes`,
`cuda_allocation_count_delta`, `cuda_free_count_delta`) merged into a
`BenchmarkResult.extra` (Milestone 11's existing extension point) for
CUDA-device results only -- `training_bench.py` and `mnist_bench.py` call
`forge.cuda.reset_peak_memory_stats()` plus a `gc.collect()` immediately
before and after their timed loop (see **Known limitations** above for why
the `gc.collect()` calls are there). CPU results are unaffected; existing
benchmark JSON consumers that only read the established `BenchmarkResult`
fields see no change. See `docs/performance/benchmarking.md`'s **Milestone
22** section for full methodology and measured numbers.

### No caching allocator
Explicitly not introduced: no memory pool, block cache, best-fit/slab
allocator, or CUDA memory pool. Every `CUDAStorage` is still one
`cudaMalloc` at construction and one `cudaFree` at garbage collection,
exactly as every earlier milestone's diagram at the top of this document
describes -- Milestone 22 only counts those events, it does not change them.

## CUDA model persistence (Milestone 13)
`save_model()`/`load_model()` (`forge/serialization/model.py`) now support
models whose `Parameter`s live on CUDA -- see
`docs/architecture/persistence.md`'s **Device semantics** for the full
loading policy. Summary of the CUDA-specific mechanics:
- **Saving** a CUDA `Parameter` copies its values to host memory via
  `Backend.to_numpy()` -- the exact same device-to-host transfer
  `Tensor.to()`/`Module.to()` already use, and a no-op copy for a CPU
  `Parameter`. This is a *persistence transfer*, not computation: no
  forward/backward pass runs as part of `save_model()`, on CPU or CUDA. The
  archive format itself is unchanged (still ZIP(`metadata.json` + one `.npy`
  per parameter, `numpy.load(..., allow_pickle=False)`) -- only the
  `"device"` metadata field's legal values (`"cpu"` or `"cuda"`, from
  `Module.device`) and what `load_model()` will do with that value changed.
- **Loading** onto CUDA constructs each `Parameter` with `device="cuda"`
  directly, which routes through `CUDABackend.from_array()` -- a real
  `cudaMalloc` + host-to-device `memcpy`, never a NumPy array relabeled as
  `CUDAStorage`.
- **Availability policy**: `load_model()` restores onto the archive's
  recorded device only when that device is actually available;
  `is_cuda_available()` is checked (lazily -- `forge.backend.cuda` is only
  imported when the recorded or requested device is `"cuda"`) before any
  parameter is constructed, so a CUDA-saved file on a CUDA-less machine
  fails with a clear `PersistenceError` rather than silently loading onto
  CPU. An explicit `load_model(path, device="cpu")` or `device="cuda")`
  overrides the recorded device for a deliberate conversion in either
  direction; an unavailable explicit `device="cuda"` still fails clearly.
- **No optimizer state, autograd graph, or `.grad`** is saved for a CUDA
  model, same as CPU -- this milestone is model (inference-time)
  persistence only, unchanged from Milestone 7's scope.
See `tests/test_cuda_persistence.py` for the hardware-verified CUDA<->CUDA
round-trip suite, and `tests/test_serialization.py`'s **CUDA device policy**
section for the metadata-level availability-policy tests (deterministic on
any machine via a monkeypatched `is_cuda_available`).

## Errors
All CUDA-specific failures raise `forge.CUDAError` (`forge/exceptions.py`):
CUDA unavailable (no `nvcc`, no compatible device), backend
initialization/compile failure, an unsupported operation (`sum()` for any
axis other than `None`/`1`/`-1`, elementwise broadcasting beyond the row- and
(for `sub`) column-broadcast shapes) or dtype (non-float compute), a memory
allocation failure, or an invalid device index. This applies equally to
backward computation as of Milestone 10 -- e.g. a shape combination
`add_backward`/`sub_backward`/`mul_backward` don't recognize (unreachable in
practice, since it mirrors the same shapes forward `_elementwise`/`sub`
already restrict to) raises `CUDAError`, not a raw CUDA kernel-launch
failure. Device-mismatch (`cpu_tensor + cuda_tensor`,
or a `backward()` call whose explicit `gradient` argument is on the wrong
device) and unrecognized device strings remain `UnsupportedDeviceError`,
matching the existing convention -- including a `backward()` gradient dtype
mismatch on CUDA (`UnsupportedDTypeError`; see `docs/architecture/
autograd.md`'s **Device consistency in `backward()`**). `Module.to(device)`
(Milestone 9) raises the same `UnsupportedDeviceError` for an unrecognized
device string, and `ModuleError` if `Module.device` finds Parameters on more
than one device (see `docs/architecture/modules.md`). See
`forge/exceptions.py::CUDAError`'s docstring for the exact split.
`forge.cuda.memory_stats()`/`reset_peak_memory_stats()` (Milestone 22) also
raise `CUDAError` if CUDA is unavailable, matching this convention -- see
**CUDA Memory Statistics** above.

## Detecting a fake CUDA backend
`tests/test_cuda_backend.py::test_backend_dispatch_is_structurally_distinct_from_cpu`
asserts `get_backend("cpu")` and `get_backend("cuda")` return instances of
different, unrelated classes (`CPUBackend` is not a base of `CUDABackend`
or vice versa). Every CUDA test additionally asserts
`isinstance(tensor._data, CUDAStorage)` and
`not isinstance(tensor._data, np.ndarray)`, so a regression that silently
routed `"cuda"` back to `CPUBackend`, or that stored a NumPy array under a
CUDA-tagged `Tensor`, would fail these tests immediately.

## Module and Parameter device movement (Milestone 9)
`Module.to(device)` recursively moves every `Parameter` in a module tree to
`device`, in place -- full design and semantics are documented in
`docs/architecture/modules.md` (Parameter identity, `requires_grad`
preservation, the mutate-and-return-`self` convention, and the `Module.device`
introspection property). This section covers only the CUDA-specific
consequences:
- Every moved `Parameter` becomes backed by a real `CUDAStorage`, via the
  same `Backend.to_numpy()` / `Backend.from_array()` round-trip
  `Tensor.to()` uses -- `Module.to()` introduces no second CUDA code path.
- `nn.Linear`'s existing `x @ weight + bias` forward requires no
  CUDA-specific code once its `Parameter`s are CUDA-resident: `@` dispatches
  to `matmul`, `+` dispatches to `add` (using the row-broadcast path above
  for a batched input's bias add).
- `nn.ReLU`'s existing `x.relu()` forward dispatches to the new CUDA `relu`
  kernel the same way.
- See **Interaction with `Module.to("cuda")`** above for the autograd
  consequence: as of Milestone 10, forward *and* backward now both succeed
  on a real CUDA graph, with or without `no_grad()`.

## Hardware verification
Verified on the actual development GPU (NVIDIA GeForce 940MX, CC 5.0,
driver 582.53, CUDA Toolkit 12.6):
- **Milestone 8**: `tests/test_cuda_backend.py` and
  `tests/test_cuda_consistency.py` (54 tests at the time) passed directly on
  this machine, exercising real kernel compilation, device-count probing,
  memory allocation, host<->device transfer, kernel launch + synchronization,
  and CPU/CUDA numeric agreement.
- **Milestone 9**: `tests/test_cuda_backend.py`, `tests/test_cuda_consistency.py`,
  and the new `tests/test_module_cuda.py` (86 CUDA tests total) all pass
  directly on this machine, additionally exercising: the new `relu` kernel
  (values and CPU/CUDA consistency), the new row-broadcast `add`/`sub`/`mul`
  kernels (both operand orders), `Module.to("cuda")` moving every `Parameter`
  of a real `Linear -> ReLU -> Linear` model to genuine `CUDAStorage`, a full
  forward pass through that model under `no_grad()` whose output agrees with
  an identically-initialized CPU model within tolerance, a structural check
  (via monkeypatched `CPUBackend` methods) that this forward pass never calls
  `CPUBackend`, and confirmation that `.backward()` on the CUDA output still
  raises `UnsupportedDeviceError`. The full suites were also run with `PATH`
  stripped of the CUDA toolchain to confirm they skip cleanly (`86 skipped`,
  `0 failed`) rather than erroring, and that the CPU-only suite (338 tests)
  is entirely unaffected by CUDA's absence.
- **Milestone 10**: the new `tests/test_cuda_autograd.py`, plus updated
  tests in `tests/test_cuda_backend.py` and `tests/test_module_cuda.py`
  (121 CUDA tests total; 459 tests overall), pass directly on this machine,
  exercising: CUDA gradient checks for `add`/`sub`/`mul` (exact-shape and
  both row-broadcast operand orders) against CPU, all four 1D/2D `matmul`
  backward cases against CPU plus a finite-difference check, `sum`/`reshape`
  backward, `relu` backward across positive/negative/zero/mixed inputs, a
  real `Linear` backward whose weight/bias gradients are CUDA-resident and
  match CPU, a `Linear -> ReLU -> Linear` multi-layer model backward
  matching CPU, gradient accumulation across multiple consumers of the same
  CUDA tensor, device/dtype-mismatch errors on `backward()`, `exp`/`log`
  still failing clearly with no graph built, `no_grad()` still suspending
  CUDA graph construction, `SGD.step()` updating a CUDA parameter in place
  (matching an equivalent CPU step) without creating a graph, `zero_grad()`
  clearing CUDA gradients, a structural check (monkeypatched `CPUBackend`)
  that a full CUDA model forward + backward pass calls zero `CPUBackend`
  methods, and a real 20-epoch CUDA training loop (`TensorDataset` ->
  `DataLoader` -> `Linear.to("cuda")` -> `MSELoss` -> `SGD`) whose loss drops
  more than 20x and recovers the true regression weights within tolerance.
  The full suite was also run with `PATH` stripped of the CUDA toolchain to
  confirm all 121 CUDA tests skip cleanly (`338 passed, 121 skipped`, `0
  failed`) rather than erroring.
- **Milestone 12**: the new `tests/test_cuda_loss.py` and
  `tests/test_trainer_cuda.py` (153 CUDA tests total; 506 tests overall)
  pass directly on this machine, plus a standalone hardware-verification
  script following the milestone's 12-step checklist exactly: constructing
  a matched CPU/CUDA `Linear(2, 1)` pair, a plain CPU `DataLoader`,
  `Trainer(device="cuda")`, training 40 epochs through `Trainer.fit()`,
  confirming every `Parameter` and gradient is `CUDAStorage`-backed,
  confirming zero `CPUBackend` compute-method calls across both `fit()` and
  a subsequent `evaluate()` (via monkeypatched `CPUBackend`), an explicit
  `CUDABackend.synchronize()` before trusting the results, and a CPU/CUDA
  comparison. Measured: loss dropped from `2.7186` to effectively `0`
  (>10^13x reduction) recovering the true weight `[3, -2]`/bias `[1]` within
  `3e-7`; CUDA and CPU loss curves agreed within `1.8e-7` (max abs diff) at
  every one of the 40 epochs, and final CUDA/CPU parameters agreed within
  `2.4e-7`. See `docs/architecture/training-engine.md`'s **Device
  semantics** section for the `Trainer`-level design this exercises, and
  `docs/development/progress.md`'s M12 entry for the full test/verification
  summary.
- **Milestone 13**: the new `tests/test_cuda_persistence.py` (10 CUDA tests;
  163 CUDA tests total, 521 tests overall) passes directly on this machine,
  exercising: CUDA -> CUDA round trips (parameter values, shapes, dtypes,
  `requires_grad`, a nested model's hierarchy and per-module training mode,
  fresh-leaf/no-grad-state parameters, and forward-output equivalence
  against the pre-save model), explicit `device="cpu"`/`device="cuda"`
  conversion round trips, a structural check (monkeypatched `CPUBackend`)
  that save+load+forward for a CUDA model calls zero `CPUBackend` compute
  methods, and a real `Trainer(device="cuda")`-trained model saved and
  reloaded with matching post-load predictions. The full suite was also run
  with `PATH` stripped of the CUDA toolchain to confirm all 163 CUDA tests
  skip cleanly (`358 passed, 163 skipped`, `0 failed`).
- **Milestone 14**: updated `tests/test_cuda_backend.py`,
  `tests/test_cuda_autograd.py`, `tests/test_cuda_consistency.py`, and
  `tests/test_cuda_loss.py`, plus a new `CrossEntropyLoss`-classification
  section in `tests/test_trainer_cuda.py` (205 CUDA tests total, 563 tests
  overall) pass directly on this machine, exercising: real `exp`/`log`
  forward+backward CUDA kernels agreeing with CPU, `sum(axis=1)` (with and
  without `keepdims`) forward+backward agreeing with CPU, the column-
  broadcast `sub` forward+backward (both operand orders) agreeing with CPU,
  `CrossEntropyLoss` CUDA/CPU forward agreement across `float32`/`float64`
  and numerically difficult logits (large positive/negative values, large
  inter-class differences, repeated/equal logits, a single-sample batch,
  seven-class logits), CUDA backward matching both CPU backward and the
  closed-form `(softmax(logits) - one_hot(target)) / batch_size`, a
  finite-difference gradient check across several batch-size/class-count
  combinations, explicit mean-reduction/`1/batch_size` gradient-scaling
  verification, CUDA/CPU target-device-mismatch validation, and a structural
  check (monkeypatched `CPUBackend`, extended with the four new compute
  methods `exp_backward`/`log_backward`/`max_axis1`) that CUDA
  `CrossEntropyLoss` forward+backward calls zero `CPUBackend` compute
  methods. At the `Trainer` level: a full `TensorDataset -> DataLoader ->
  Trainer(device="cuda") -> Linear -> CrossEntropyLoss -> CUDA backward ->
  CUDA SGD` classification run on a deterministic two-class dataset, over 30
  epochs, with loss dropping to less than half its starting value and final
  accuracy exceeding 90%; every `Parameter` and gradient confirmed
  `CUDAStorage`-backed throughout; a structural no-CPU-fallback check across
  a full multi-epoch classification `fit()` call; and a CPU/CUDA
  classification training-consistency comparison (matched initial
  parameters, unshuffled batches, identical loss curves within tolerance
  across 10 epochs). The full suite was also run with `PATH` stripped of the
  CUDA toolchain to confirm all 205 CUDA tests skip cleanly (`358 passed,
  205 skipped`, `0 failed`) and the CPU-only suite is entirely unaffected.
- **Milestone 16 (`Dropout` CUDA execution, plus `Sequential`/`Flatten`
  composed on top of it)**: real GPU verification on the 940MX confirms
  real `CUDAStorage` output/gradients; `fraction_zeroed`/`mean` statistical
  agreement with the requested `p` (`test_cuda_dropout_zeroes_approximately_
  p_fraction`, `..._preserves_mean_within_tolerance`); exact
  `1/(1-p)`-scaled nonzero elements; `grad`-matches-`mask` backward
  correctness; exact eval-mode identity; a structural
  zero-`CPUBackend`-calls check across a full Dropout forward + backward
  (`test_cuda_dropout_never_calls_cpu_backend`); an explicit CPU-vs-CUDA
  test asserting masks are *not* bitwise-equal while still agreeing
  statistically; a `Sequential(Conv2d, ReLU, MaxPool2d, Flatten, Linear,
  ReLU, Dropout, Linear)` end-to-end `Trainer(device="cuda")` classification
  run (loss dropping to <60% of its start, ≥85% final accuracy); and a
  CUDA `Sequential`+`Dropout` save/load round trip. The full suite was
  again run with `PATH` stripped of the CUDA toolchain to confirm all 258
  CUDA tests (up from 205, net of this milestone's new CUDA tests) skip
  cleanly (`516 passed, 258 skipped`, `0 failed`).
- **Milestone 17 (`Adam` CUDA execution)**: real GPU verification on the
  940MX confirms `Adam`'s optimizer state (`m`, `v`) is genuine
  `CUDAStorage`, never `np.ndarray`; parameter updates mutate the existing
  CUDA buffer in place; CPU and CUDA Adam agree within `1e-4` across a
  single step, multiple steps with `weight_decay`, and a real `Linear`
  model trained in lockstep on both devices; a structural
  zero-`CPUBackend.adam_step`-calls check across a full CUDA `step()`; the
  Policy-A device-mismatch guard raising `OptimizerError` after
  `model.to("cuda")` invalidates CPU-created state, and recovering cleanly
  once that stale state is cleared; a full CUDA `Trainer(device="cuda")` +
  `Adam` training run with loss decreasing; and every `Parameter`/gradient/
  optimizer-state buffer confirmed `CUDAStorage`-backed throughout. The full
  suite (269 CUDA tests total, up from 258; 825 tests overall) was again run
  with `PATH` stripped of the CUDA toolchain to confirm all 269 CUDA tests
  skip cleanly (`556 passed, 269 skipped`, `0 failed`).
- **Milestone 22 (CUDA memory statistics and allocation lifecycle)**: the
  new `tests/test_cuda_memory.py` (21 real-hardware tests, plus 2
  CUDA-unavailable error-path tests in a separate
  `tests/test_cuda_memory_availability.py` that run regardless of hardware
  -- 930 tests overall) passes directly on the 940MX, exercising:
  basic/multiple allocation byte
  accounting, deallocation, peak tracking and its persistence after a
  temporary is freed, `reset_peak_memory_stats()` isolation from live
  allocations, CPU/CUDA statistic isolation, CPU<->CUDA transfer accounting
  (including the same-device `.to()` no-op allocating nothing), repeated
  autograd forward/backward without an optimizer, a full `Linear -> ReLU ->
  Linear` + `Adam` training/eval lifecycle, a `Dropout`-containing model's
  repeated training and eval cycles, Adam's persistent `m`/`v` state
  correctly distinguished from temporaries (including its release once
  `optimizer.state.clear()` drops the last reference), checkpoint
  save/load lifecycle (save adds no persistent CUDA growth; load allocates
  a second model+optimizer's worth, released on deletion), a real
  allocation failure (16 GiB request) leaving statistics untouched --
  provoked in an isolated subprocess for the reason documented in **CUDA
  Memory Statistics**'s "Known limitations" above -- and a 100-iteration
  bounded-growth leak regression. Every lifecycle assertion in this suite
  calls `gc.collect()` plus `CUDABackend.synchronize()` before reading
  `memory_stats()`, per that same "Known limitations" discussion.

## Limitations
- **Operation set is intentionally small**: `add`/`mul` (exact-shape, plus
  the one Milestone 9 row-broadcast shape), `sub` (those two plus the
  Milestone 14 column-broadcast shape), `matmul` (1D/2D), `sum` (full
  reduction, plus Milestone 14's `axis=1`), `reshape`, `relu`, and (Milestone
  14) `exp`/`log` -- forward *and* backward on every one of these. Still no
  general N-D broadcasting (only the two targeted shapes above) and no
  general N-D axis reduction (only `axis=None` and `axis=1` on a 2D tensor).
- **`forge.training.Trainer` now supports CUDA (Milestone 12)**, superseding
  the note this bullet used to make about Milestone 6-10: `Trainer(...,
  device="cuda")` runs a real end-to-end training/evaluation workflow --
  `DataLoader` stays CPU-only, and `Trainer` explicitly transfers each batch
  and validates (never moves) the model's device. See
  `docs/architecture/training-engine.md`'s **Device semantics** section and
  this file's **CUDA losses**/**CUDA CrossEntropyLoss** sections above. Both
  built-in losses now work on CUDA (`MSELoss` since Milestone 12,
  `CrossEntropyLoss` since Milestone 14). What remains unsupported:
  - No GPU `DataLoader`, pinned memory, async prefetch, or multiprocessing
    workers -- explicit non-goals, unchanged.
  - A `device="cpu"` `Trainer` fed a CUDA-resident model still fails clearly
    (now via `Trainer._check_model_device()`'s explicit validation, rather
    than incidentally from the first device-mismatched forward op) -- see
    `tests/test_module_cuda.py::test_trainer_configured_for_cpu_rejects_a_cuda_model`.
- **CUDA persistence (Milestone 13)**: `save_model`/`load_model` support
  CUDA models -- see **CUDA model persistence** above and
  `docs/architecture/persistence.md`. What remains unsupported, unchanged
  from Milestone 7: no optimizer-state/training-resume checkpointing, no
  CUDA-specific archive format (the same portable ZIP(json + .npy) format
  CPU files have always used), and only the device-coherent trees
  `Module.device` already requires (a manually mixed-device tree raises
  `ModuleError` before anything is written, same as `Module.device` itself).
  This still holds as of Milestone 17: `Adam`'s per-parameter state (`m`,
  `v`, step count) lives entirely in `Adam.state`, outside the `Module` tree
  `save_model()` walks, so it is never written to a model archive -- see
  **CUDA Adam** above and `docs/architecture/optimization.md`'s **Adam**
  section.
- **Single GPU, index 0 only**: `device="cuda:N"` for `N != 0` raises
  `CUDAError`. No multi-GPU support.
- **No custom GPU memory allocator**: every operation's output is a fresh
  `cudaMalloc`, freed via `cudaFree` on garbage collection. Reasonable for
  the small models this milestone targets; not tuned for throughput.
  Milestone 22 added observability into this model (`forge.cuda.
  memory_stats()`) but deliberately did not change it -- no pooling/caching,
  per that milestone's explicit non-goal; see **CUDA Memory Statistics**
  above.
- **Matmul is a single fixed 16x16-tile shared-memory GEMM** (re-tiled in
  Milestone 11 from Milestone 8's naive one-thread-per-output-element
  kernel, after benchmarking measured the naive kernel as a real bottleneck
  at the 512x512 scale -- see `docs/performance/benchmarking.md`'s
  "Optimization decision" section). Not a generalized, autotuned GEMM: one
  fixed tile size, no double-buffering/prefetch, no vectorized loads. CUDA
  matmul remains slower than the CPU/NumPy backend at that scale on the
  940MX even after this optimization -- the benchmarking document reports
  the measured before/after numbers rather than claiming otherwise.
- **No int/bool compute kernels**: transfer works for `int32`/`int64`/
  `bool`, but arithmetic ops on CUDA require `float32`/`float64`.
- **CrossEntropyLoss-adjacent primitives stay narrowly scoped (Milestone
  14)**: no `Tensor.max()`/general reduction API was added (`max_axis1` is a
  private `Backend` method, not a public Tensor operation); no general
  `softmax` public API; no arbitrary N-D axis reduction; column-broadcast
  support exists for `sub` only, not `add`/`mul`. `CrossEntropyLoss` itself
  still supports only the `(batch_size, num_classes)`/`(batch_size,)` shape
  convention -- no class weighting, label smoothing, or ignored-index
  support, unchanged from before this milestone (see
  `docs/architecture/optimization.md`).
- **CUDA Dropout uses a stateless hash, not curand (Milestone 16)**: no
  `curandState` allocation, no cuRAND dependency -- see **CUDA Dropout**
  above for the SplitMix64-based mechanism and why the milestone brief
  favors it over a general GPU RNG library. CPU and CUDA masks are not
  expected to agree element-for-element even under the same
  `forge.random.seed(...)` (different algorithms consuming the seed
  differently) -- only statistical/semantic behavior is compared across
  backends, never exact per-element values.
- **No CLI/benchmarking integration**: this milestone adds the backend and
  its tests only; CLI and benchmark surfaces (`docs/product/requirements.md`)
  are unaffected.
- **Performance is documented separately**: this file reports correctness
  and hardware verification; measured performance (including the confirmed
  fact that small workloads, and even the "medium" 512x512 matmul, run
  slower on this CUDA backend than on CPU) is in
  `docs/performance/benchmarking.md` (Milestone 11), not repeated here.
