# CUDA Backend (Milestones 8-10; CUDA `Trainer`/loss integration in Milestone 12; CUDA model persistence in Milestone 13)

## Summary
Forge has a real CUDA execution backend for a small operation set: tensor
creation/transfer, `add`/`sub`/`mul` (exact-shape, plus one targeted
row-broadcast shape added in Milestone 9), `matmul` (the same 1D/2D cases
the CPU backend supports), `sum` (full reduction only), `reshape`, and, as
of Milestone 9, `relu`. Every one of these actually launches a CUDA kernel
(or issues a real `cudaMemcpy`/`cudaMalloc` call) on the GPU -- nothing in
this list is a disguised CPU fallback. As of Milestone 9, this operation set
is also reachable through the high-level `nn.Module` API (`Module.to("cuda")`,
then an ordinary `Linear`/`ReLU` forward pass) -- see **Module and Parameter
device movement** below and `docs/architecture/modules.md`. As of
**Milestone 10, this operation set is no longer forward-only**: every one of
these operations has a real CUDA *backward* kernel too, so a CUDA
computation graph can be built and differentiated end-to-end on the GPU,
including through `nn.Module`/`Linear`/`ReLU`, with CUDA-resident gradients
and a CUDA-executing `SGD.step()` -- see **CUDA autograd** below. See
`docs/architecture/backend-architecture.md` for the general backend
boundary this fits into.

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
            __init__.py        Re-exports the above
    tensor/tensor.py         Tensor.to(device); Tensor._move_storage_() (M9); backward() (device-generic as of M10)
    autograd/engine.py        run_backward -- backend-dispatched gradient accumulation (M10)
    nn/module.py              Module.to(device), Module.device (new in M9)
    nn/loss.py                MSELoss CUDA-compatible unmodified; CrossEntropyLoss CPU-only guard (M12)
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
| `add`/`sub`/`mul` | `cf_{add,sub,mul}_{f32,f64}`, `cf_{add,sub,mul}_bcast_{f32,f64}` | float32, float64 | Exact-shape, plus (Milestone 9) one row-broadcast shape: a `(rows, cols)` matrix combined with a `(cols,)` vector -- see **Row-broadcast** below |
| `matmul` | `cf_matmul_{f32,f64}` | float32, float64 | Shared-memory-tiled kernel (16x16 tiles, Milestone 11 -- see below); 1D vectors are reinterpreted as degenerate `M=1`/`N=1` 2D matmuls on the host side, matching the CPU backend's four 1D/2D cases |
| `sum` | `cf_sum_{f32,f64}` | float32, float64 | Full reduction only (`axis=None`); block-level shared-memory tree reduction + atomic accumulation across blocks (a CAS-based emulation for `atomicAdd(double*)`, since CC 5.0 has no native one) |
| `reshape` | `cf_memcpy_d2d` | any transferable dtype | Implemented as a device-to-device copy into a freshly allocated buffer with new shape metadata (no in-place aliasing, to avoid any storage-ownership ambiguity without an allocator) |
| `relu` (Milestone 9) | `cf_relu_{f32,f64}` | float32, float64 | `out[i] = max(a[i], 0)`, one thread per element -- the same launch pattern as `add`/`sub`/`mul` |

`exp`/`log` remain required by the `Backend` ABC (the CPU backend implements
them) but the CUDA backend still raises `CUDAError` for both -- no kernel
exists for them, unchanged from Milestone 8. This is a real, tested failure
path (`tests/test_cuda_backend.py::test_exp_is_unsupported_on_cuda` etc.),
not an oversight: moving a tensor to CPU (`.to("cpu")`) is the documented
way to use them today. `relu` moved out of this list in Milestone 9 -- see
above.

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

Every kernel launch is followed by an explicit `cf_synchronize()`
(`cudaDeviceSynchronize()`) before its result is trusted or returned, so
CUDA's asynchronous execution model can never be mistaken for a completed
operation.

## Numerical consistency
`tests/test_cuda_consistency.py` runs the same inputs through both
backends for every shared operation (`add`/`sub`/`mul`/`matmul`/`sum`/
`reshape`/`relu`, both `float32` and `float64`, several matmul shape
combinations, plus a chained `Linear -> ReLU -> Linear`-shaped expression)
and asserts `np.testing.assert_allclose` agreement (`rtol=atol=1e-5`) --
never bit-for-bit equality, per `docs/architecture/backend-architecture.md`.
`tests/test_module_cuda.py` runs the same comparison through the high-level
`nn.Module`/`Linear`/`ReLU` API (`rtol=atol=1e-4`, a looser tolerance
appropriate for a longer op chain).

## CUDA autograd (Milestone 10)
**CUDA autograd is supported** for every operation this backend implements
forward (see **Operation set** above): `add`/`sub`/`mul` (exact-shape and
the one row-broadcast shape), `matmul` (1D/2D), `sum` (full reduction),
`reshape`, and `relu`. A CUDA computation graph now builds and
differentiates entirely on the GPU -- see `docs/architecture/autograd.md`'s
**Backend-aware backward dispatch** section for how `Tensor`'s backward
closures reach `CUDABackend`'s methods, and ADR-005 for why this replaced
the Milestone 8/9 "CUDA autograd is not supported" boundary rather than
adding a parallel mechanism.

### Backward kernels
| Backward rule | Kernel(s) | Notes |
|---|---|---|
| `add_backward`/`sub_backward`/`mul_backward` | `cf_neg_{f32,f64}`, `cf_reduce_rows_{f32,f64}`, plus the existing forward `add`/`sub`/`mul`/`mul_bcast` kernels | `add`: identity for both operands (row-broadcast case reduces the vector operand's gradient with `cf_reduce_rows`). `sub`: identity/`cf_neg` depending on operand order. `mul`: reuses forward `mul` (elementwise or row-broadcast) to build each raw gradient, then `cf_reduce_rows` where a vector operand needs its gradient reduced |
| `matmul_backward` | `cf_scale_{f32,f64}`, `cf_transpose_{f32,f64}`, plus the existing forward `matmul`/`reshape` kernels | All four 1D/2D cases are built by composing existing kernels -- e.g. matrix·matrix's `grad_output @ b.T` and `a.T @ grad_output` reuse `matmul` on a freshly transposed operand; the 1D·1D (dot product) case uses `cf_scale`, a device-resident scalar-times-vector kernel that reads the scalar via a device pointer so no value ever crosses back to the host |
| `sum_backward` | `cf_broadcast_scalar_{f32,f64}` | Broadcasts the one upstream scalar to every element of the original shape -- the only case reachable, since forward `sum(axis=...)` already raises `CUDAError` for anything but a full reduction |
| `reshape_backward` | (reuses forward `reshape`, `cf_memcpy_d2d`) | Reshaping the upstream gradient back to the original shape is exactly the forward `reshape` op run again |
| `relu_backward` | `cf_relu_backward_{f32,f64}` | `out[i] = input[i] > 0 ? grad_output[i] : 0` -- one kernel, no separate mask kernel, input never copied to CPU |

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

### `CrossEntropyLoss`: deliberately deferred to CPU
`CrossEntropyLoss` needs `.exp()`, `.log()` (both CPU-only -- see
**Operation set** above), and an axis-wise `logits.exp().sum(axis=1,
keepdims=True).log()` (CUDA `sum()` supports only a full reduction). Making
it CUDA-compatible (the milestone brief's "Approach A") would mean: a new
CUDA `exp` kernel and backward, a new CUDA `log` kernel and backward (with
care to preserve the log-sum-exp numerical-stability trick this loss
depends on -- see `docs/architecture/optimization.md`'s **CrossEntropyLoss**
section), and a new axis-wise-reduction CUDA kernel and backward distinct
from the existing full-reduction `sum`/`sum_backward`. That is real,
multi-kernel surface area unrelated to what the milestone's core objective
(CUDA training through `Trainer`, demonstrated end-to-end by `MSELoss`)
actually required -- exactly the kind of scope the milestone brief's
"Approach B" explicitly sanctions deferring ("do not force CrossEntropy
into the milestone merely to claim completeness").

`CrossEntropyLoss.forward()` therefore rejects non-CPU logits immediately
and explicitly:
```python
if logits.device.type != "cpu":
    raise LossError(
        "CrossEntropyLoss is CPU-only in this milestone ..."
    )
```
rather than silently running part of the computation on CPU (a disguised
fallback) or letting the failure surface indirectly from deep inside
`.numpy()`'s own device check. A `Trainer(device="cuda",
loss_fn=CrossEntropyLoss())` fails this way the first time `fit()`/
`evaluate()` calls the loss -- clearly, immediately, and without executing
any part of the forward pass on CPU. Moving logits to CPU explicitly first
(`logits.to("cpu")`) still works exactly as before Milestone 12; see
`docs/architecture/training-engine.md`'s **CUDA losses through Trainer**
section for the `Trainer`-level framing of this same boundary.

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
initialization/compile failure, an unsupported operation (`exp`/`log`,
`sum(axis=...)`, elementwise broadcasting beyond the one row-broadcast shape)
or dtype (non-float compute), a memory allocation failure, or an invalid
device index. This applies equally to backward computation as of Milestone
10 -- e.g. a shape combination `add_backward`/`sub_backward`/`mul_backward`
don't recognize (unreachable in practice, since it mirrors the same shapes
forward `_elementwise` already restricts to) raises `CUDAError`, not a raw
CUDA kernel-launch failure. Device-mismatch (`cpu_tensor + cuda_tensor`,
or a `backward()` call whose explicit `gradient` argument is on the wrong
device) and unrecognized device strings remain `UnsupportedDeviceError`,
matching the existing convention -- including a `backward()` gradient dtype
mismatch on CUDA (`UnsupportedDTypeError`; see `docs/architecture/
autograd.md`'s **Device consistency in `backward()`**). `Module.to(device)`
(Milestone 9) raises the same `UnsupportedDeviceError` for an unrecognized
device string, and `ModuleError` if `Module.device` finds Parameters on more
than one device (see `docs/architecture/modules.md`). See
`forge/exceptions.py::CUDAError`'s docstring for the exact split.

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

## Limitations
- **Operation set is intentionally small**: `add`/`sub`/`mul` (exact-shape,
  plus the one Milestone 9 row-broadcast shape), `matmul` (1D/2D), `sum`
  (full reduction), `reshape`, `relu`, and transfer -- forward *and*
  backward, as of Milestone 10. No `exp`/`log`, no general N-D
  broadcasting, no axis-wise reduction on CUDA, in either direction.
- **`forge.training.Trainer` now supports CUDA (Milestone 12)**, superseding
  the note this bullet used to make about Milestone 6-10: `Trainer(...,
  device="cuda")` runs a real end-to-end training/evaluation workflow --
  `DataLoader` stays CPU-only, and `Trainer` explicitly transfers each batch
  and validates (never moves) the model's device. See
  `docs/architecture/training-engine.md`'s **Device semantics** section and
  this file's **CUDA losses** section above. What remains unsupported:
  - `CrossEntropyLoss` is CPU-only (deferred; see **CUDA losses** above) --
    a CUDA `Trainer` using it fails clearly via `LossError`.
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
- **Single GPU, index 0 only**: `device="cuda:N"` for `N != 0` raises
  `CUDAError`. No multi-GPU support.
- **No custom GPU memory allocator**: every operation's output is a fresh
  `cudaMalloc`, freed via `cudaFree` on garbage collection. Reasonable for
  the small models this milestone targets; not tuned for throughput.
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
- **No CLI/benchmarking integration**: this milestone adds the backend and
  its tests only; CLI and benchmark surfaces (`docs/product/requirements.md`)
  are unaffected.
- **Performance is documented separately**: this file reports correctness
  and hardware verification; measured performance (including the confirmed
  fact that small workloads, and even the "medium" 512x512 matmul, run
  slower on this CUDA backend than on CPU) is in
  `docs/performance/benchmarking.md` (Milestone 11), not repeated here.
