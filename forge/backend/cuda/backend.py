"""The real CUDA execution backend: `CUDAStorage` and `CUDABackend`.

`CUDAStorage` is a thin handle around a `cudaMalloc`-allocated device
pointer plus the shape/dtype metadata the rest of Forge needs -- it is the
CUDA analog of the plain `numpy.ndarray` the CPU backend stores directly in
`Tensor._data`. It never holds a NumPy array as its backing data; a Tensor
tagged `device="cuda"` is only ever backed by one of these, never a CPU
array pretending to be one (see `docs/architecture/cuda-backend.md`).

`CUDABackend` implements the same `Backend` interface `CPUBackend` does
(`docs/architecture/backend-architecture.md`), dispatching each operation to
a small, explicit CUDA kernel compiled by `build.py` and loaded once per
process via `ctypes`. Every kernel launch is followed by an explicit
`cudaDeviceSynchronize()` (via `cf_synchronize`) before its result is
trusted, so CUDA's asynchronous execution model can never be mistaken for a
completed, verified operation.

Deliberately small operation set (per the milestone brief): tensor
creation/transfer, `add`/`sub`/`mul` (exact-shape, plus one targeted
row-broadcast shape added in Milestone 9 -- see `_elementwise` -- still no
general CUDA broadcasting), `matmul` (the same 1D/2D cases the CPU backend
supports), `sum` (full reduction only, no `axis`), and, as of Milestone 9,
`relu`. `exp`/`log` remain required by the `Backend` ABC but are not
implemented for CUDA; calling them raises `CUDAError` rather than silently
running on CPU.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ...exceptions import CUDAError
from ..base import Backend
from . import build as _build

_COMPUTE_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_SUFFIX = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64"}

_TRANSFERABLE_DTYPES = (
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.bool_),
)

_lib_cache: "ctypes.CDLL | None" = None


def _configure_signatures(lib: "ctypes.CDLL") -> None:
    lib.cf_device_count.argtypes = []
    lib.cf_device_count.restype = ctypes.c_int

    lib.cf_error_string.argtypes = [ctypes.c_int]
    lib.cf_error_string.restype = ctypes.c_char_p

    lib.cf_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cf_malloc.restype = ctypes.c_int

    lib.cf_free.argtypes = [ctypes.c_void_p]
    lib.cf_free.restype = ctypes.c_int

    for name in ("cf_memcpy_h2d", "cf_memcpy_d2h", "cf_memcpy_d2d"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_int

    lib.cf_synchronize.argtypes = []
    lib.cf_synchronize.restype = ctypes.c_int

    for suffix in ("f32", "f64"):
        for op in ("add", "sub", "mul"):
            fn = getattr(lib, f"cf_{op}_{suffix}")
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
            fn.restype = ctypes.c_int

            bcast_fn = getattr(lib, f"cf_{op}_bcast_{suffix}")
            bcast_fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
            ]
            bcast_fn.restype = ctypes.c_int

        matmul_fn = getattr(lib, f"cf_matmul_{suffix}")
        matmul_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        matmul_fn.restype = ctypes.c_int

        sum_fn = getattr(lib, f"cf_sum_{suffix}")
        sum_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
        sum_fn.restype = ctypes.c_int

        relu_fn = getattr(lib, f"cf_relu_{suffix}")
        relu_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
        relu_fn.restype = ctypes.c_int


def _load_library() -> "ctypes.CDLL":
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    library_path = _build.ensure_kernel_library()
    try:
        lib = ctypes.CDLL(str(library_path))
    except OSError as exc:
        raise CUDAError(f"Failed to load the compiled CUDA kernel library '{library_path}': {exc}") from exc
    _configure_signatures(lib)
    _lib_cache = lib
    return lib


class CUDAStorage:
    """A handle to a `cudaMalloc`-allocated device buffer plus its shape/dtype.

    Holds real GPU-resident memory -- never a NumPy array. Freed via
    `cudaFree` when garbage collected.
    """

    __slots__ = ("ptr", "shape", "dtype", "_lib")

    def __init__(self, ptr: "ctypes.c_void_p", shape: "tuple[int, ...]", dtype: Any, lib: "ctypes.CDLL"):
        self.ptr = ptr
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self._lib = lib

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __repr__(self) -> str:
        return f"CUDAStorage(shape={self.shape}, dtype={self.dtype})"

    def __del__(self) -> None:
        try:
            if self.ptr:
                self._lib.cf_free(self.ptr)
        except Exception:
            pass  # interpreter shutdown may have already torn down the library handle


class CUDABackend(Backend):
    device_type = "cuda"

    def __init__(self):
        self._lib = _load_library()
        count = self._lib.cf_device_count()
        if count <= 0:
            raise CUDAError(
                f"No CUDA-capable device detected (cudaGetDeviceCount() returned {count}). "
                "A CUDA-capable NVIDIA GPU and driver are required to use device='cuda'."
            )
        self.device_count = count

    # -- error handling ----------------------------------------------------

    def _check(self, code: int, action: str) -> None:
        if code != 0:
            message = self._lib.cf_error_string(code)
            message = message.decode() if message is not None else "<no message>"
            raise CUDAError(f"CUDA error during {action}: {message} (code {code}).")

    def _synchronize(self, action: str) -> None:
        self._check(self._lib.cf_synchronize(), f"{action} (synchronize)")

    def _alloc(self, nbytes: int) -> "ctypes.c_void_p":
        ptr = ctypes.c_void_p()
        code = self._lib.cf_malloc(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 1)))
        if code != 0:
            message = self._lib.cf_error_string(code)
            message = message.decode() if message is not None else "<no message>"
            raise CUDAError(f"CUDA memory allocation of {nbytes} bytes failed: {message} (code {code}).")
        return ptr

    # -- transfer ------------------------------------------------------------

    def from_array(self, data: Any, dtype: "np.dtype | None") -> CUDAStorage:
        if isinstance(data, CUDAStorage):
            target_dtype = np.dtype(dtype) if dtype is not None else data.dtype
            if target_dtype != data.dtype:
                raise CUDAError(
                    "Converting a CUDA tensor's dtype during construction is not supported "
                    f"in this milestone (source dtype '{data.dtype}', requested '{target_dtype}')."
                )
            new_ptr = self._alloc(data.nbytes)
            if data.nbytes > 0:
                code = self._lib.cf_memcpy_d2d(new_ptr, data.ptr, ctypes.c_size_t(data.nbytes))
                self._check(code, "device-to-device copy")
            return CUDAStorage(new_ptr, data.shape, data.dtype, self._lib)

        host = np.ascontiguousarray(np.array(data, dtype=dtype))
        if host.dtype not in _TRANSFERABLE_DTYPES:
            supported = ", ".join(str(d) for d in _TRANSFERABLE_DTYPES)
            raise CUDAError(f"Unsupported dtype for a CUDA tensor: '{host.dtype}'. Supported: {supported}.")

        ptr = self._alloc(host.nbytes)
        if host.nbytes > 0:
            code = self._lib.cf_memcpy_h2d(ptr, host.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(host.nbytes))
            self._check(code, "host-to-device transfer")
        return CUDAStorage(ptr, host.shape, host.dtype, self._lib)

    def to_numpy(self, storage: Any) -> np.ndarray:
        if not isinstance(storage, CUDAStorage):
            raise CUDAError(f"CUDABackend.to_numpy() expects a CUDAStorage, got {type(storage).__name__}.")
        host = np.empty(storage.shape, dtype=storage.dtype)
        if host.nbytes > 0:
            code = self._lib.cf_memcpy_d2h(host.ctypes.data_as(ctypes.c_void_p), storage.ptr, ctypes.c_size_t(host.nbytes))
            self._check(code, "device-to-host transfer")
        return host

    # -- compute-dtype validation -------------------------------------------

    def _require_compute_dtype(self, *storages: CUDAStorage, op: str) -> np.dtype:
        dtype = storages[0].dtype
        for s in storages[1:]:
            if s.dtype != dtype:
                raise CUDAError(
                    f"CUDA '{op}' requires matching dtypes, got {[str(s.dtype) for s in storages]}."
                )
        if dtype not in _COMPUTE_DTYPES:
            raise CUDAError(f"CUDA '{op}' does not support dtype '{dtype}'. Supported: float32, float64.")
        return dtype

    # -- elementwise ops -------------------------------------------------------

    def _elementwise(self, a: CUDAStorage, b: CUDAStorage, op: str) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, b, op=op)

        if a.shape == b.shape:
            n = a.size
            out_ptr = self._alloc(n * dtype.itemsize)
            fn = getattr(self._lib, f"cf_{op}_{_SUFFIX[dtype]}")
            code = fn(a.ptr, b.ptr, out_ptr, ctypes.c_longlong(n))
            self._check(code, op)
            self._synchronize(op)
            return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

        # Milestone 9: the one broadcast shape Linear's `x @ weight + bias`
        # needs -- a 2D (rows, cols) matrix combined with a 1D (cols,)
        # vector, e.g. a batched matmul result plus a bias. See kernels.cu's
        # comment above `k_add_bcast` for why this is a targeted addition
        # rather than general CUDA broadcasting.
        if a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0]:
            mat, vec, vec_is_left, out_shape = a, b, 0, a.shape
        elif a.ndim == 1 and b.ndim == 2 and b.shape[1] == a.shape[0]:
            mat, vec, vec_is_left, out_shape = b, a, 1, b.shape
        else:
            raise CUDAError(
                f"CUDA '{op}' does not support general broadcasting in this milestone -- only "
                "exact-shape operands, or a (rows, cols) matrix combined with a (cols,) vector "
                f"(e.g. a Linear bias add), are supported; got shapes {a.shape} and {b.shape}. "
                "Broadcast on CPU first, or reshape explicitly, before moving to CUDA."
            )

        rows, cols = mat.shape
        out_ptr = self._alloc(rows * cols * dtype.itemsize)
        fn = getattr(self._lib, f"cf_{op}_bcast_{_SUFFIX[dtype]}")
        code = fn(
            mat.ptr, vec.ptr, out_ptr,
            ctypes.c_longlong(rows), ctypes.c_longlong(cols), ctypes.c_int(vec_is_left),
        )
        self._check(code, f"{op} (row-broadcast)")
        self._synchronize(f"{op} (row-broadcast)")
        return CUDAStorage(out_ptr, out_shape, dtype, self._lib)

    def add(self, a: Any, b: Any) -> CUDAStorage:
        return self._elementwise(a, b, "add")

    def sub(self, a: Any, b: Any) -> CUDAStorage:
        return self._elementwise(a, b, "sub")

    def mul(self, a: Any, b: Any) -> CUDAStorage:
        return self._elementwise(a, b, "mul")

    # -- matmul ----------------------------------------------------------------

    def matmul(self, a: CUDAStorage, b: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, b, op="matmul")

        if a.ndim == 1 and b.ndim == 1:
            if a.shape[0] != b.shape[0]:
                raise ValueError(f"matmul: inner dimensions {a.shape} and {b.shape} do not match.")
            M, K, N, out_shape = 1, a.shape[0], 1, ()
        elif a.ndim == 2 and b.ndim == 1:
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"matmul: inner dimensions {a.shape} and {b.shape} do not match.")
            M, K, N, out_shape = a.shape[0], a.shape[1], 1, (a.shape[0],)
        elif a.ndim == 1 and b.ndim == 2:
            if a.shape[0] != b.shape[0]:
                raise ValueError(f"matmul: inner dimensions {a.shape} and {b.shape} do not match.")
            M, K, N, out_shape = 1, a.shape[0], b.shape[1], (b.shape[1],)
        elif a.ndim == 2 and b.ndim == 2:
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"matmul: inner dimensions {a.shape} and {b.shape} do not match.")
            M, K, N, out_shape = a.shape[0], a.shape[1], b.shape[1], (a.shape[0], b.shape[1])
        else:
            raise CUDAError(
                f"CUDA matmul supports 1D and 2D tensors only, got shapes {a.shape} and {b.shape}."
            )

        out_ptr = self._alloc(M * N * dtype.itemsize)
        fn = getattr(self._lib, f"cf_matmul_{_SUFFIX[dtype]}")
        code = fn(a.ptr, b.ptr, out_ptr, ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N))
        self._check(code, "matmul")
        self._synchronize("matmul")
        return CUDAStorage(out_ptr, out_shape, dtype, self._lib)

    # -- reduction ---------------------------------------------------------------

    def sum(self, a: CUDAStorage, axis: Any, keepdims: bool) -> CUDAStorage:
        if axis is not None:
            raise CUDAError(
                "CUDA sum() supports only a full reduction (axis=None) in this milestone; "
                f"got axis={axis!r}. Move the tensor to CPU with .to('cpu') for axis-wise reduction."
            )
        dtype = self._require_compute_dtype(a, op="sum")
        out_ptr = self._alloc(dtype.itemsize)
        fn = getattr(self._lib, f"cf_sum_{_SUFFIX[dtype]}")
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(a.size))
        self._check(code, "sum")
        self._synchronize("sum")
        shape = (1,) * a.ndim if keepdims else ()
        return CUDAStorage(out_ptr, shape, dtype, self._lib)

    # -- reshape (metadata + device-side copy, no kernel needed) -----------------

    def reshape(self, a: CUDAStorage, shape: "tuple[int, ...]") -> CUDAStorage:
        total = 1
        for s in shape:
            total *= s
        if total != a.size:
            raise ValueError(f"cannot reshape CUDA tensor of size {a.size} into shape {shape}.")
        new_ptr = self._alloc(a.nbytes)
        if a.nbytes > 0:
            code = self._lib.cf_memcpy_d2d(new_ptr, a.ptr, ctypes.c_size_t(a.nbytes))
            self._check(code, "reshape (device-to-device copy)")
        return CUDAStorage(new_ptr, shape, a.dtype, self._lib)

    # -- unary elementwise ops (Milestone 9) --------------------------------------

    def relu(self, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="relu")
        n = a.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_relu_{_SUFFIX[dtype]}")
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(n))
        self._check(code, "relu")
        self._synchronize("relu")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    # -- explicitly unsupported in this milestone --------------------------------

    def exp(self, a: Any) -> Any:
        raise CUDAError(
            "The CUDA backend does not implement exp() in this milestone. "
            "Move the tensor to CPU with .to('cpu') to use it."
        )

    def log(self, a: Any) -> Any:
        raise CUDAError(
            "The CUDA backend does not implement log() in this milestone. "
            "Move the tensor to CPU with .to('cpu') to use it."
        )


_backend_singleton: "CUDABackend | None" = None
_unavailable_reason: "str | None" = None


def get_cuda_backend() -> CUDABackend:
    """Return the process-wide `CUDABackend` singleton, building/initializing it if needed.

    Initialization (compiling the kernel library and probing the device) is
    attempted at most once per process; a failure is cached so repeated
    `device="cuda"` requests on a non-CUDA machine fail fast with the same
    clear reason instead of retrying an expensive compile every time.
    """
    global _backend_singleton, _unavailable_reason
    if _backend_singleton is not None:
        return _backend_singleton
    if _unavailable_reason is not None:
        raise CUDAError(_unavailable_reason)

    try:
        backend = CUDABackend()
    except CUDAError as exc:
        _unavailable_reason = str(exc)
        raise
    except Exception as exc:  # pragma: no cover -- defensive: never let a raw exception leak here
        _unavailable_reason = f"Unexpected error initializing the CUDA backend: {exc}"
        raise CUDAError(_unavailable_reason) from exc

    _backend_singleton = backend
    return backend


def is_cuda_available() -> bool:
    """Whether the CUDA backend can be initialized on this machine right now.

    Actually attempts initialization (compiling the kernel library the first
    time, probing for a device) rather than just checking for `nvcc` on
    PATH, so a "yes" here means CUDA tensors can actually be constructed.
    """
    try:
        get_cuda_backend()
        return True
    except CUDAError:
        return False


__all__ = ["CUDABackend", "CUDAStorage", "get_cuda_backend", "is_cuda_available"]
