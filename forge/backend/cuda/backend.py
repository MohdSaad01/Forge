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
process via `ctypes`, on whichever CUDA stream is currently current
(`forge.backend.cuda.stream.current_stream()`). On the CUDA default stream
(current unless a `with forge.cuda.stream(s):` block is active), every
kernel launch is still followed by an explicit `cudaDeviceSynchronize()`
before its result is trusted, so CUDA's asynchronous execution model can
never be mistaken for a completed, verified operation there -- the exact
Milestone 8-26 contract. On an explicit `CUDAStream`, that per-op
synchronization is skipped instead (Milestone 27's asynchronous execution
mode); see `docs/architecture/cuda-streams.md` for the full contract.

Deliberately small operation set (per the milestone brief): tensor
creation/transfer, `add`/`sub`/`mul` (exact-shape, plus one targeted
row-broadcast shape added in Milestone 9, plus one targeted column-broadcast
shape for `sub` added in Milestone 14 -- see `_elementwise`/
`_col_broadcast_kind` -- still no general CUDA broadcasting), `matmul` (the
same 1D/2D cases the CPU backend supports), `sum` (full reduction, plus
axis=1 on a 2D tensor as of Milestone 14), `relu`, and, as of Milestone 14,
`exp`/`log` (needed by `CrossEntropyLoss`'s log-sum-exp) plus a private
`max_axis1` row-reduction helper. See `docs/architecture/cuda-backend.md`'s
**CUDA CrossEntropyLoss** section for the Milestone 14 additions.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ...exceptions import CUDAError
from ..base import Backend
from . import allocator as _allocator
from . import build as _build
from . import stream as _stream

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

    # -- CUDA streams and events (Milestone 27) -- see stream.py --
    lib.cf_stream_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cf_stream_create.restype = ctypes.c_int
    lib.cf_stream_destroy.argtypes = [ctypes.c_void_p]
    lib.cf_stream_destroy.restype = ctypes.c_int
    lib.cf_stream_synchronize.argtypes = [ctypes.c_void_p]
    lib.cf_stream_synchronize.restype = ctypes.c_int
    lib.cf_event_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cf_event_create.restype = ctypes.c_int
    lib.cf_event_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cf_event_record.restype = ctypes.c_int
    lib.cf_event_query.argtypes = [ctypes.c_void_p]
    lib.cf_event_query.restype = ctypes.c_int
    lib.cf_event_synchronize.argtypes = [ctypes.c_void_p]
    lib.cf_event_synchronize.restype = ctypes.c_int
    lib.cf_event_destroy.argtypes = [ctypes.c_void_p]
    lib.cf_event_destroy.restype = ctypes.c_int

    # Every kernel-launching function below gained a trailing `void* stream`
    # parameter in Milestone 27 (`ctypes.c_void_p` -- `None` means the CUDA
    # default stream); the memcpy/malloc/free/synchronize functions above did
    # not (see `docs/architecture/cuda-streams.md`'s **Memory copy semantics**
    # section for why those stay plain, stream-implicit `cudaMemcpy`).

    for suffix in ("f32", "f64"):
        for op in ("add", "sub", "mul"):
            fn = getattr(lib, f"cf_{op}_{suffix}")
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
            fn.restype = ctypes.c_int

            bcast_fn = getattr(lib, f"cf_{op}_bcast_{suffix}")
            bcast_fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p,
            ]
            bcast_fn.restype = ctypes.c_int

        matmul_fn = getattr(lib, f"cf_matmul_{suffix}")
        matmul_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        matmul_fn.restype = ctypes.c_int

        sum_fn = getattr(lib, f"cf_sum_{suffix}")
        sum_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
        sum_fn.restype = ctypes.c_int

        relu_fn = getattr(lib, f"cf_relu_{suffix}")
        relu_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
        relu_fn.restype = ctypes.c_int

        # -- backward-only kernels (Milestone 10) --
        neg_fn = getattr(lib, f"cf_neg_{suffix}")
        neg_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
        neg_fn.restype = ctypes.c_int

        for name in (f"cf_relu_backward_{suffix}", f"cf_scale_{suffix}"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
            fn.restype = ctypes.c_int

        transpose_fn = getattr(lib, f"cf_transpose_{suffix}")
        transpose_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
        ]
        transpose_fn.restype = ctypes.c_int

        reduce_rows_fn = getattr(lib, f"cf_reduce_rows_{suffix}")
        reduce_rows_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
        ]
        reduce_rows_fn.restype = ctypes.c_int

        sgd_step_fn = getattr(lib, f"cf_sgd_step_{suffix}")
        sgd_step_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_longlong, ctypes.c_void_p,
        ]
        sgd_step_fn.restype = ctypes.c_int

        adam_step_fn = getattr(lib, f"cf_adam_step_{suffix}")
        adam_step_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_longlong, ctypes.c_void_p,
        ]
        adam_step_fn.restype = ctypes.c_int

        broadcast_scalar_fn = getattr(lib, f"cf_broadcast_scalar_{suffix}")
        broadcast_scalar_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
        broadcast_scalar_fn.restype = ctypes.c_int

        # -- Milestone 14: exp/log, axis=1 reduction, column-broadcast sub --
        for name in (f"cf_exp_{suffix}", f"cf_log_{suffix}"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
            fn.restype = ctypes.c_int

        for name in (f"cf_exp_backward_{suffix}", f"cf_log_backward_{suffix}"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
            fn.restype = ctypes.c_int

        for name in (f"cf_max_axis1_{suffix}", f"cf_sum_axis1_{suffix}"):
            fn = getattr(lib, name)
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int

        broadcast_axis1_fn = getattr(lib, f"cf_broadcast_axis1_{suffix}")
        broadcast_axis1_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
        ]
        broadcast_axis1_fn.restype = ctypes.c_int

        sub_colbcast_fn = getattr(lib, f"cf_sub_colbcast_{suffix}")
        sub_colbcast_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p,
        ]
        sub_colbcast_fn.restype = ctypes.c_int

        # -- Milestone 15: Conv2d / MaxPool2d --
        conv_fwd_fn = getattr(lib, f"cf_conv2d_forward_{suffix}")
        conv_fwd_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ] + [ctypes.c_int] * 14 + [ctypes.c_void_p]
        conv_fwd_fn.restype = ctypes.c_int

        conv_bwd_input_fn = getattr(lib, f"cf_conv2d_backward_input_{suffix}")
        conv_bwd_input_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ] + [ctypes.c_int] * 13 + [ctypes.c_void_p]
        conv_bwd_input_fn.restype = ctypes.c_int

        conv_bwd_weight_fn = getattr(lib, f"cf_conv2d_backward_weight_{suffix}")
        conv_bwd_weight_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ] + [ctypes.c_int] * 13 + [ctypes.c_void_p]
        conv_bwd_weight_fn.restype = ctypes.c_int

        conv_bwd_bias_fn = getattr(lib, f"cf_conv2d_backward_bias_{suffix}")
        conv_bwd_bias_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        conv_bwd_bias_fn.restype = ctypes.c_int

        pool_fwd_fn = getattr(lib, f"cf_maxpool2d_forward_{suffix}")
        pool_fwd_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 12 + [ctypes.c_void_p]
        pool_fwd_fn.restype = ctypes.c_int

        pool_bwd_fn = getattr(lib, f"cf_maxpool2d_backward_{suffix}")
        pool_bwd_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ] + [ctypes.c_int] * 12 + [ctypes.c_void_p]
        pool_bwd_fn.restype = ctypes.c_int

        # -- Milestone 16: Dropout --
        dropout_mask_fn = getattr(lib, f"cf_dropout_mask_{suffix}")
        dropout_mask_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong, ctypes.c_double, ctypes.c_uint64, ctypes.c_void_p,
        ]
        dropout_mask_fn.restype = ctypes.c_int


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
    """A handle to a device buffer (allocated via the Milestone 25 caching allocator) plus its shape/dtype.

    Holds real GPU-resident memory -- never a NumPy array. On destruction the
    underlying pointer is returned to `forge.backend.cuda.allocator` (see
    that module) rather than immediately `cudaFree`d -- `self.ptr = None` is
    set *before* that hand-off so a second `__del__` call can never return
    the same pointer to the cache twice (the ownership invariant
    `allocator.py` depends on: a live `CUDAStorage` and a cached free block
    never alias the same pointer).

    `last_stream` (Milestone 27) is the `CUDAStream` this storage was last
    touched by (as an operation's input *or* output -- see
    `CUDABackend._stream_guard`), or `None` if it has only ever been touched
    on the CUDA default stream. This is the one piece of stream-lifetime
    metadata Forge tracks per storage (per the milestone brief's "do not
    attach a full stream history" constraint) -- just enough for `__del__`
    to know whether an immediate cache release is still safe (`None`,
    exactly the M26 contract) or whether the allocator must defer reuse
    until that stream's recorded event completes (`release_pending()`, see
    `docs/architecture/cuda-streams.md`). Holding a strong reference to the
    `CUDAStream` object itself (not just its raw handle) is deliberate: it
    keeps the stream's underlying CUDA resource valid until this storage's
    own `__del__` has recorded an event on it -- see `stream.py`'s
    `CUDAStream` docstring.
    """

    __slots__ = ("ptr", "shape", "dtype", "_lib", "last_stream")

    def __init__(self, ptr: "ctypes.c_void_p", shape: "tuple[int, ...]", dtype: Any, lib: "ctypes.CDLL"):
        self.ptr = ptr
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self._lib = lib
        self.last_stream = _stream.current_stream()

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
        ptr = self.ptr
        if not ptr:
            return
        nbytes = self.nbytes
        last_stream = self.last_stream
        self.ptr = None  # transfer ownership to the allocator's cache before it can be touched again
        self.last_stream = None
        try:
            if last_stream is None:
                _allocator.release(nbytes, ptr)
            else:
                _allocator.release_pending(self._lib, nbytes, ptr, last_stream.handle)
        except Exception:
            return  # interpreter shutdown may have already torn down module globals


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

    # -- stream plumbing (Milestone 27) --------------------------------------
    #
    # Every kernel-launching method below reads these three helpers, never
    # `_stream.current_stream()` directly, so the "which stream, and does
    # this op synchronize" decision lives in one place. See
    # `docs/architecture/cuda-streams.md` for the full contract.

    def _stream_handle(self) -> "ctypes.c_void_p | None":
        """The raw stream handle every kernel launch below passes as its trailing argument."""
        current = _stream.current_stream()
        return current.handle if current is not None else None

    def _maybe_synchronize(self, action: str) -> None:
        """`cudaDeviceSynchronize()` only in default-stream (M26-compatible) mode.

        Skipped for an operation issued on an explicit `CUDAStream`: that is
        the entire point of Milestone 27's asynchronous execution mode (see
        Section 8/9 of the milestone brief) -- the caller must synchronize
        explicitly (`stream.synchronize()` or `forge.cuda.synchronize()`)
        before relying on the result from the host.
        """
        if _stream.current_stream() is None:
            self._synchronize(action)

    def _stream_guard(self, storages: "tuple[CUDAStorage, ...]", op: str) -> None:
        """Validate and refresh each storage's `last_stream` before a kernel launch touches it.

        Raises `CUDAError` if any storage was last touched on a *different*,
        still-live stream than the one this operation is about to launch on
        (Forge does not support this cross-stream dependency -- see
        Section 20/21 of the milestone brief and Invariant 5 in
        `docs/architecture/cuda-streams.md`: "fail clearly rather than
        silently produce incorrect results"). A storage whose `last_stream`
        is `None` (only ever touched on the default stream) is always safe
        to read from any stream, since default-stream work is already fully
        complete by the time Python can see the storage at all (the M26
        contract, still intact for that one stream). Marking every storage
        to the current stream here -- *before* the launch -- is correct
        because Python is single-threaded: nothing else can run between this
        check and the launch it guards.
        """
        current = _stream.current_stream()
        for s in storages:
            previous = s.last_stream
            if previous is not None and previous is not current:
                raise CUDAError(
                    f"CUDA '{op}' would use a tensor last used on stream {previous!r} while "
                    f"the current Forge CUDA stream is {current!r} -- Forge does not support "
                    "this cross-stream dependency in this milestone. Synchronize explicitly "
                    "(that stream's .synchronize(), or forge.cuda.synchronize()) before "
                    "crossing streams, or keep the whole producer/consumer chain on one stream."
                )
        for s in storages:
            s.last_stream = current

    def _alloc(self, nbytes: int) -> "ctypes.c_void_p":
        """Request `nbytes` of device memory via the Milestone 25 caching allocator.

        A cache hit returns an already-owned device pointer with no driver
        call; a cache miss (or the first request of a new size) falls
        through to a real `cudaMalloc`, including the M24-designed
        purge-and-retry-once policy on OOM (see `allocator.py`). Either way,
        `CUDAMemoryStats.allocated_bytes` gains `nbytes` -- the caching
        allocator does not change what "active" means, only whether serving
        it required a driver call.
        """
        return _allocator.allocate(self._lib, nbytes)

    # -- transfer ------------------------------------------------------------

    def from_array(self, data: Any, dtype: "np.dtype | None") -> CUDAStorage:
        if isinstance(data, CUDAStorage):
            target_dtype = np.dtype(dtype) if dtype is not None else data.dtype
            if target_dtype != data.dtype:
                raise CUDAError(
                    "Converting a CUDA tensor's dtype during construction is not supported "
                    f"in this milestone (source dtype '{data.dtype}', requested '{target_dtype}')."
                )
            # `cf_memcpy_d2d` stays a plain synchronous `cudaMemcpy` (never
            # `cudaMemcpyAsync`, no stream argument -- see `docs/architecture/
            # cuda-streams.md`'s **Memory copy semantics**), which under
            # CUDA's legacy-default-stream semantics is already implicitly
            # ordered against every other Forge stream. `_stream_guard` is
            # not required for correctness here, but is kept for a clear
            # Forge-level error (rather than a silent, implicit ordering
            # guarantee) and to keep `data.last_stream` accurate.
            self._stream_guard((data,), "from_array (device-to-device copy)")
            new_ptr = self._alloc(data.nbytes)
            if data.nbytes > 0:
                code = self._lib.cf_memcpy_d2d(new_ptr, data.ptr, ctypes.c_size_t(data.nbytes))
                self._check(code, "device-to-device copy")
            return CUDAStorage(new_ptr, data.shape, data.dtype, self._lib)

        host = np.array(data, dtype=dtype)
        original_shape = host.shape
        host = np.ascontiguousarray(host)
        if host.shape != original_shape:
            # `np.ascontiguousarray` silently promotes a 0-d array to shape
            # (1,) (it guarantees ndim >= 1) -- reshape back so a CUDA
            # scalar tensor (e.g. a loss, or `x.sum()`) keeps shape (),
            # matching the CPU backend and `Tensor`'s own shape contract.
            host = host.reshape(original_shape)
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
        # Explicit stream-specific synchronization (Milestone 27) before the
        # plain, host-blocking `cudaMemcpy` below -- keeps `to_numpy()`'s
        # "always host-blocking, always correct" contract self-evident
        # rather than relying only on `cudaMemcpy`'s implicit legacy-stream
        # ordering. Only waits for `storage`'s own last-use stream, not the
        # whole device.
        if storage.last_stream is not None:
            storage.last_stream.synchronize()
        host = np.empty(storage.shape, dtype=storage.dtype)
        if host.nbytes > 0:
            code = self._lib.cf_memcpy_d2h(host.ctypes.data_as(ctypes.c_void_p), storage.ptr, ctypes.c_size_t(host.nbytes))
            self._check(code, "device-to-host transfer")
        return host

    # -- compute-dtype validation -------------------------------------------

    def _require_compute_dtype(self, *storages: CUDAStorage, op: str) -> np.dtype:
        self._stream_guard(storages, op)
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
            code = fn(a.ptr, b.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
            self._check(code, op)
            self._maybe_synchronize(op)
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
            self._stream_handle(),
        )
        self._check(code, f"{op} (row-broadcast)")
        self._maybe_synchronize(f"{op} (row-broadcast)")
        return CUDAStorage(out_ptr, out_shape, dtype, self._lib)

    def add(self, a: Any, b: Any) -> CUDAStorage:
        return self._elementwise(a, b, "add")

    def sub(self, a: Any, b: Any) -> CUDAStorage:
        col_kind = self._col_broadcast_kind(a, b)
        if col_kind is not None:
            return self._sub_col_broadcast(a, b, col_kind)
        return self._elementwise(a, b, "sub")

    def mul(self, a: Any, b: Any) -> CUDAStorage:
        return self._elementwise(a, b, "mul")

    # -- column-broadcast sub (Milestone 14) --------------------------------
    #
    # `CrossEntropyLoss`'s log-sum-exp shift combines a (rows, cols) matrix
    # with a (rows, 1) per-row scalar (`max_axis1`'s or a preceding
    # `sum(axis=1, keepdims=True)`'s own output shape), broadcasting that
    # scalar across every column of its row -- the transpose of
    # `_elementwise`'s existing row-broadcast case (a (cols,) vector
    # broadcast down every row, for Linear's bias add). Only `sub` needs this
    # broadcast direction in this milestone (see `kernels.cu`'s
    # **column-broadcast subtract kernel** comment), so `add`/`mul` still
    # raise `CUDAError` via `_elementwise` for a (rows, 1) operand.

    def _col_broadcast_kind(self, a: Any, b: Any) -> "str | None":
        a_shape, b_shape = getattr(a, "shape", None), getattr(b, "shape", None)
        if a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2:
            return None
        if a_shape == b_shape:
            return None  # exact-shape case, handled by `_elementwise` as usual
        if a_shape[0] == b_shape[0] and b_shape[1] == 1 and a_shape[1] != 1:
            return "a_mat_b_col"
        if a_shape[0] == b_shape[0] and a_shape[1] == 1 and b_shape[1] != 1:
            return "a_col_b_mat"
        return None

    def _sub_col_broadcast(self, a: CUDAStorage, b: CUDAStorage, kind: str) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, b, op="sub (column-broadcast)")
        if kind == "a_mat_b_col":
            mat, colvec, vec_is_left, out_shape = a, b, 0, a.shape
        else:
            mat, colvec, vec_is_left, out_shape = b, a, 1, b.shape
        rows, cols = mat.shape
        out_ptr = self._alloc(rows * cols * dtype.itemsize)
        fn = getattr(self._lib, f"cf_sub_colbcast_{_SUFFIX[dtype]}")
        code = fn(
            mat.ptr, colvec.ptr, out_ptr,
            ctypes.c_longlong(rows), ctypes.c_longlong(cols), ctypes.c_int(vec_is_left),
            self._stream_handle(),
        )
        self._check(code, "sub (column-broadcast)")
        self._maybe_synchronize("sub (column-broadcast)")
        return CUDAStorage(out_ptr, out_shape, dtype, self._lib)

    def _reduce_axis1(self, mat: CUDAStorage) -> CUDAStorage:
        """Sum a (rows, cols) matrix down to a (rows,) vector -- undoes column-broadcast."""
        dtype = self._require_compute_dtype(mat, op="reduce_axis1")
        rows, cols = mat.shape
        out_ptr = self._alloc(rows * dtype.itemsize)
        fn = getattr(self._lib, f"cf_sum_axis1_{_SUFFIX[dtype]}")
        code = fn(mat.ptr, out_ptr, ctypes.c_longlong(rows), ctypes.c_longlong(cols), self._stream_handle())
        self._check(code, "reduce_axis1 (column-broadcast gradient reduction)")
        self._maybe_synchronize("reduce_axis1")
        return CUDAStorage(out_ptr, (rows,), dtype, self._lib)

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
        code = fn(a.ptr, b.ptr, out_ptr, ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N), self._stream_handle())
        self._check(code, "matmul")
        self._maybe_synchronize("matmul")
        return CUDAStorage(out_ptr, out_shape, dtype, self._lib)

    # -- reduction ---------------------------------------------------------------

    def sum(self, a: CUDAStorage, axis: Any, keepdims: bool) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="sum")
        if axis is None:
            out_ptr = self._alloc(dtype.itemsize)
            fn = getattr(self._lib, f"cf_sum_{_SUFFIX[dtype]}")
            code = fn(a.ptr, out_ptr, ctypes.c_longlong(a.size), self._stream_handle())
            self._check(code, "sum")
            self._maybe_synchronize("sum")
            shape = (1,) * a.ndim if keepdims else ()
            return CUDAStorage(out_ptr, shape, dtype, self._lib)

        # Milestone 14: the one axis-wise reduction CrossEntropyLoss needs --
        # summing a 2D (batch, classes) tensor's class dimension down to one
        # value per row. See kernels.cu's "axis=1 reduction kernels" comment
        # for why this stays a single fixed axis rather than general N-D
        # reduction.
        if a.ndim == 2 and axis in (1, -1):
            rows, cols = a.shape
            out_ptr = self._alloc(rows * dtype.itemsize)
            fn = getattr(self._lib, f"cf_sum_axis1_{_SUFFIX[dtype]}")
            code = fn(a.ptr, out_ptr, ctypes.c_longlong(rows), ctypes.c_longlong(cols), self._stream_handle())
            self._check(code, "sum(axis=1)")
            self._maybe_synchronize("sum(axis=1)")
            shape = (rows, 1) if keepdims else (rows,)
            return CUDAStorage(out_ptr, shape, dtype, self._lib)

        raise CUDAError(
            "CUDA sum() supports only a full reduction (axis=None) or axis=1 (equivalently "
            f"-1) on a 2D tensor in this milestone; got axis={axis!r} for a tensor of shape "
            f"{a.shape}. Move the tensor to CPU with .to('cpu') for other axis-wise reductions."
        )

    # -- reshape (metadata + device-side copy, no kernel needed) -----------------

    def reshape(self, a: CUDAStorage, shape: "tuple[int, ...]") -> CUDAStorage:
        total = 1
        for s in shape:
            total *= s
        if total != a.size:
            raise ValueError(f"cannot reshape CUDA tensor of size {a.size} into shape {shape}.")
        self._stream_guard((a,), "reshape (device-to-device copy)")  # see from_array's identical note
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
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "relu")
        self._maybe_synchronize("relu")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    # -- exp / log (Milestone 14) -------------------------------------------------

    def exp(self, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="exp")
        n = a.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_exp_{_SUFFIX[dtype]}")
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "exp")
        self._maybe_synchronize("exp")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    def log(self, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="log")
        n = a.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_log_{_SUFFIX[dtype]}")
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "log")
        self._maybe_synchronize("log")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    # -- backward helpers (Milestone 10) ---------------------------------------
    #
    # Private, CUDA-only composition helpers used by the `*_backward` methods
    # below. Each wraps one real kernel (`kernels.cu`'s "backward-only
    # kernels" section) and returns a fresh `CUDAStorage` -- gradient
    # computation for a CUDA graph never leaves the device.

    def _neg(self, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="neg")
        n = a.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_neg_{_SUFFIX[dtype]}")
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "neg")
        self._maybe_synchronize("neg")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    def _scale(self, scalar: CUDAStorage, vec: CUDAStorage) -> CUDAStorage:
        """`scalar` (a 0-d/scalar CUDAStorage) times `vec`, elementwise -- read entirely on-device."""
        dtype = self._require_compute_dtype(scalar, vec, op="scale")
        n = vec.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_scale_{_SUFFIX[dtype]}")
        code = fn(scalar.ptr, vec.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "scale")
        self._maybe_synchronize("scale")
        return CUDAStorage(out_ptr, vec.shape, dtype, self._lib)

    def _transpose(self, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="transpose")
        if a.ndim != 2:
            raise CUDAError(f"CUDA transpose (a matmul-backward helper) requires a 2D tensor, got shape {a.shape}.")
        rows, cols = a.shape
        out_ptr = self._alloc(rows * cols * dtype.itemsize)
        fn = getattr(self._lib, f"cf_transpose_{_SUFFIX[dtype]}")
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(rows), ctypes.c_longlong(cols), self._stream_handle())
        self._check(code, "transpose")
        self._maybe_synchronize("transpose")
        return CUDAStorage(out_ptr, (cols, rows), dtype, self._lib)

    def _reduce_rows(self, mat: CUDAStorage) -> CUDAStorage:
        """Sum a (rows, cols) matrix down to a (cols,) vector -- undoes row-broadcast."""
        dtype = self._require_compute_dtype(mat, op="reduce_rows")
        rows, cols = mat.shape
        out_ptr = self._alloc(cols * dtype.itemsize)
        fn = getattr(self._lib, f"cf_reduce_rows_{_SUFFIX[dtype]}")
        code = fn(mat.ptr, out_ptr, ctypes.c_longlong(rows), ctypes.c_longlong(cols), self._stream_handle())
        self._check(code, "reduce_rows (row-broadcast gradient reduction)")
        self._maybe_synchronize("reduce_rows")
        return CUDAStorage(out_ptr, (cols,), dtype, self._lib)

    def _row_broadcast_kind(self, a: CUDAStorage, b: CUDAStorage) -> str:
        """Classify `(a, b)`'s shape relationship, matching `_elementwise`'s forward rule."""
        if a.shape == b.shape:
            return "exact"
        if a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0]:
            return "a_mat_b_vec"
        if a.ndim == 1 and b.ndim == 2 and b.shape[1] == a.shape[0]:
            return "a_vec_b_mat"
        raise CUDAError(
            f"CUDA backward does not support this broadcast shape combination: {a.shape} and {b.shape}."
        )

    # -- backward (Milestone 10) -----------------------------------------------

    def add_backward(self, grad_output: CUDAStorage, a: CUDAStorage, b: CUDAStorage):
        kind = self._row_broadcast_kind(a, b)
        if kind == "exact":
            return grad_output, grad_output
        if kind == "a_mat_b_vec":
            return grad_output, self._reduce_rows(grad_output)
        return self._reduce_rows(grad_output), grad_output

    def sub_backward(self, grad_output: CUDAStorage, a: CUDAStorage, b: CUDAStorage):
        col_kind = self._col_broadcast_kind(a, b)
        if col_kind == "a_mat_b_col":
            # out = a(mat) - b(colvec); d/da = grad_output, d/db = -sum_over_cols(grad_output)
            grad_b = self.reshape(self._neg(self._reduce_axis1(grad_output)), b.shape)
            return grad_output, grad_b
        if col_kind == "a_col_b_mat":
            # out = a(colvec) - b(mat); d/da = sum_over_cols(grad_output), d/db = -grad_output
            grad_a = self.reshape(self._reduce_axis1(grad_output), a.shape)
            return grad_a, self._neg(grad_output)

        kind = self._row_broadcast_kind(a, b)
        if kind == "exact":
            return grad_output, self._neg(grad_output)
        if kind == "a_mat_b_vec":
            # out = a(mat) - b(vec)
            return grad_output, self._neg(self._reduce_rows(grad_output))
        # a_vec_b_mat: out = a(vec) - b(mat)
        return self._reduce_rows(grad_output), self._neg(grad_output)

    def mul_backward(self, grad_output: CUDAStorage, a: CUDAStorage, b: CUDAStorage):
        kind = self._row_broadcast_kind(a, b)
        if kind == "exact":
            return self.mul(grad_output, b), self.mul(grad_output, a)
        if kind == "a_mat_b_vec":
            return self.mul(grad_output, b), self._reduce_rows(self.mul(grad_output, a))
        # a_vec_b_mat
        return self._reduce_rows(self.mul(grad_output, b)), self.mul(grad_output, a)

    def matmul_backward(self, grad_output: CUDAStorage, a: CUDAStorage, b: CUDAStorage):
        if a.ndim == 1 and b.ndim == 1:
            return self._scale(grad_output, b), self._scale(grad_output, a)
        if a.ndim == 2 and b.ndim == 1:
            M, K = a.shape
            grad_col = self.reshape(grad_output, (M, 1))
            b_row = self.reshape(b, (1, K))
            grad_a = self.matmul(grad_col, b_row)  # (M,1) @ (1,K) = (M,K)
            grad_b = self.matmul(self._transpose(a), grad_output)  # (K,M) @ (M,) = (K,)
            return grad_a, grad_b
        if a.ndim == 1 and b.ndim == 2:
            K, N = b.shape
            grad_a = self.matmul(b, grad_output)  # (K,N) @ (N,) = (K,)
            a_col = self.reshape(a, (K, 1))
            grad_row = self.reshape(grad_output, (1, N))
            grad_b = self.matmul(a_col, grad_row)  # (K,1) @ (1,N) = (K,N)
            return grad_a, grad_b
        # a.ndim == 2 and b.ndim == 2
        grad_a = self.matmul(grad_output, self._transpose(b))
        grad_b = self.matmul(self._transpose(a), grad_output)
        return grad_a, grad_b

    def sum_backward(
        self, grad_output: CUDAStorage, original_shape: "tuple[int, ...]", ndim: int, axis, keepdims: bool
    ) -> CUDAStorage:
        dtype = self._require_compute_dtype(grad_output, op="sum_backward")
        if axis is None:
            n = 1
            for s in original_shape:
                n *= s
            out_ptr = self._alloc(n * dtype.itemsize)
            fn = getattr(self._lib, f"cf_broadcast_scalar_{_SUFFIX[dtype]}")
            code = fn(grad_output.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
            self._check(code, "sum backward (broadcast)")
            self._maybe_synchronize("sum backward")
            return CUDAStorage(out_ptr, original_shape, dtype, self._lib)

        if ndim == 2 and axis in (1, -1):
            rows, cols = original_shape
            out_ptr = self._alloc(rows * cols * dtype.itemsize)
            fn = getattr(self._lib, f"cf_broadcast_axis1_{_SUFFIX[dtype]}")
            code = fn(grad_output.ptr, out_ptr, ctypes.c_longlong(rows), ctypes.c_longlong(cols), self._stream_handle())
            self._check(code, "sum(axis=1) backward (broadcast)")
            self._maybe_synchronize("sum(axis=1) backward")
            return CUDAStorage(out_ptr, original_shape, dtype, self._lib)

        # Forward CUDA `sum(axis=...)` already raises CUDAError for any other
        # axis (see `sum` above), so this branch is defensive, not a
        # reachable path from `Tensor.sum()`.
        raise CUDAError(
            "CUDA sum() backward supports only a full reduction (axis=None) or axis=1 "
            f"(equivalently -1) on a 2D tensor in this milestone; got axis={axis!r}."
        )

    def reshape_backward(self, grad_output: CUDAStorage, original_shape: "tuple[int, ...]") -> CUDAStorage:
        return self.reshape(grad_output, original_shape)

    def relu_backward(self, grad_output: CUDAStorage, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(grad_output, a, op="relu_backward")
        n = a.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_relu_backward_{_SUFFIX[dtype]}")
        code = fn(grad_output.ptr, a.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "relu backward")
        self._maybe_synchronize("relu backward")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    def exp_backward(self, grad_output: CUDAStorage, result: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(grad_output, result, op="exp_backward")
        n = result.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_exp_backward_{_SUFFIX[dtype]}")
        code = fn(grad_output.ptr, result.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "exp backward")
        self._maybe_synchronize("exp backward")
        return CUDAStorage(out_ptr, result.shape, dtype, self._lib)

    def log_backward(self, grad_output: CUDAStorage, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(grad_output, a, op="log_backward")
        n = a.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_log_backward_{_SUFFIX[dtype]}")
        code = fn(grad_output.ptr, a.ptr, out_ptr, ctypes.c_longlong(n), self._stream_handle())
        self._check(code, "log backward")
        self._maybe_synchronize("log backward")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    # -- CrossEntropyLoss support (Milestone 14) ---------------------------------

    def max_axis1(self, a: CUDAStorage) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="max_axis1")
        if a.ndim != 2:
            raise CUDAError(f"CUDA max_axis1 requires a 2D tensor, got shape {a.shape}.")
        rows, cols = a.shape
        out_ptr = self._alloc(rows * dtype.itemsize)
        fn = getattr(self._lib, f"cf_max_axis1_{_SUFFIX[dtype]}")
        code = fn(a.ptr, out_ptr, ctypes.c_longlong(rows), ctypes.c_longlong(cols), self._stream_handle())
        self._check(code, "max_axis1")
        self._maybe_synchronize("max_axis1")
        return CUDAStorage(out_ptr, (rows, 1), dtype, self._lib)

    # -- Conv2d / MaxPool2d (Milestone 15) ---------------------------------------
    #
    # Real, straightforward CUDA kernels (`kernels.cu`'s "Conv2d / MaxPool2d"
    # section) -- one thread per output/gradient-target element, looping over
    # the kernel window in registers. No CPU fallback: every array here stays
    # `CUDAStorage` throughout, matching every other method on this class.

    def conv2d(
        self, x: CUDAStorage, weight: CUDAStorage, bias: "CUDAStorage | None",
        stride: "tuple[int, int]", padding: "tuple[int, int]",
    ) -> CUDAStorage:
        storages = (x, weight) if bias is None else (x, weight, bias)
        dtype = self._require_compute_dtype(*storages, op="conv2d")
        N, Cin, H, W = x.shape
        Cout, _, KH, KW = weight.shape
        SH, SW = stride
        PH, PW = padding
        Hout = (H + 2 * PH - KH) // SH + 1
        Wout = (W + 2 * PW - KW) // SW + 1

        out_ptr = self._alloc(N * Cout * Hout * Wout * dtype.itemsize)
        fn = getattr(self._lib, f"cf_conv2d_forward_{_SUFFIX[dtype]}")
        code = fn(
            x.ptr, weight.ptr, bias.ptr if bias is not None else None, out_ptr,
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
            ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
            ctypes.c_int(Hout), ctypes.c_int(Wout), ctypes.c_int(1 if bias is not None else 0),
            self._stream_handle(),
        )
        self._check(code, "conv2d")
        self._maybe_synchronize("conv2d")
        return CUDAStorage(out_ptr, (N, Cout, Hout, Wout), dtype, self._lib)

    def conv2d_backward(
        self, grad_output: CUDAStorage, x: CUDAStorage, weight: CUDAStorage, bias: "CUDAStorage | None",
        stride: "tuple[int, int]", padding: "tuple[int, int]",
    ) -> "tuple[CUDAStorage, CUDAStorage, CUDAStorage | None]":
        storages = (grad_output, x, weight) if bias is None else (grad_output, x, weight, bias)
        dtype = self._require_compute_dtype(*storages, op="conv2d_backward")
        N, Cin, H, W = x.shape
        Cout, _, KH, KW = weight.shape
        SH, SW = stride
        PH, PW = padding
        Hout, Wout = grad_output.shape[2], grad_output.shape[3]
        shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
            ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )

        grad_x_ptr = self._alloc(N * Cin * H * W * dtype.itemsize)
        fn_gx = getattr(self._lib, f"cf_conv2d_backward_input_{_SUFFIX[dtype]}")
        code = fn_gx(grad_output.ptr, weight.ptr, grad_x_ptr, *shape_args, self._stream_handle())
        self._check(code, "conv2d backward (input)")
        self._maybe_synchronize("conv2d backward (input)")
        grad_x = CUDAStorage(grad_x_ptr, x.shape, dtype, self._lib)

        grad_w_ptr = self._alloc(Cout * Cin * KH * KW * dtype.itemsize)
        fn_gw = getattr(self._lib, f"cf_conv2d_backward_weight_{_SUFFIX[dtype]}")
        code = fn_gw(grad_output.ptr, x.ptr, grad_w_ptr, *shape_args, self._stream_handle())
        self._check(code, "conv2d backward (weight)")
        self._maybe_synchronize("conv2d backward (weight)")
        grad_w = CUDAStorage(grad_w_ptr, weight.shape, dtype, self._lib)

        grad_b = None
        if bias is not None:
            grad_b_ptr = self._alloc(Cout * dtype.itemsize)
            fn_gb = getattr(self._lib, f"cf_conv2d_backward_bias_{_SUFFIX[dtype]}")
            code = fn_gb(
                grad_output.ptr, grad_b_ptr,
                ctypes.c_int(N), ctypes.c_int(Cout), ctypes.c_int(Hout), ctypes.c_int(Wout),
                self._stream_handle(),
            )
            self._check(code, "conv2d backward (bias)")
            self._maybe_synchronize("conv2d backward (bias)")
            grad_b = CUDAStorage(grad_b_ptr, (Cout,), dtype, self._lib)

        return grad_x, grad_w, grad_b

    def max_pool2d(
        self, x: CUDAStorage, kernel_size: "tuple[int, int]", stride: "tuple[int, int]", padding: "tuple[int, int]"
    ) -> CUDAStorage:
        dtype = self._require_compute_dtype(x, op="max_pool2d")
        N, C, H, W = x.shape
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding
        Hout = (H + 2 * PH - KH) // SH + 1
        Wout = (W + 2 * PW - KW) // SW + 1

        out_ptr = self._alloc(N * C * Hout * Wout * dtype.itemsize)
        fn = getattr(self._lib, f"cf_maxpool2d_forward_{_SUFFIX[dtype]}")
        code = fn(
            x.ptr, out_ptr,
            ctypes.c_int(N), ctypes.c_int(C), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(KH), ctypes.c_int(KW), ctypes.c_int(SH), ctypes.c_int(SW),
            ctypes.c_int(PH), ctypes.c_int(PW), ctypes.c_int(Hout), ctypes.c_int(Wout),
            self._stream_handle(),
        )
        self._check(code, "max_pool2d")
        self._maybe_synchronize("max_pool2d")
        return CUDAStorage(out_ptr, (N, C, Hout, Wout), dtype, self._lib)

    def max_pool2d_backward(
        self, grad_output: CUDAStorage, x: CUDAStorage, kernel_size: "tuple[int, int]",
        stride: "tuple[int, int]", padding: "tuple[int, int]",
    ) -> CUDAStorage:
        dtype = self._require_compute_dtype(grad_output, x, op="max_pool2d_backward")
        N, C, H, W = x.shape
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding
        Hout, Wout = grad_output.shape[2], grad_output.shape[3]

        grad_x_ptr = self._alloc(N * C * H * W * dtype.itemsize)
        fn = getattr(self._lib, f"cf_maxpool2d_backward_{_SUFFIX[dtype]}")
        code = fn(
            x.ptr, grad_output.ptr, grad_x_ptr,
            ctypes.c_int(N), ctypes.c_int(C), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(KH), ctypes.c_int(KW), ctypes.c_int(SH), ctypes.c_int(SW),
            ctypes.c_int(PH), ctypes.c_int(PW), ctypes.c_int(Hout), ctypes.c_int(Wout),
            self._stream_handle(),
        )
        self._check(code, "max_pool2d backward")
        self._maybe_synchronize("max_pool2d backward")
        return CUDAStorage(grad_x_ptr, x.shape, dtype, self._lib)

    # -- Dropout (Milestone 16) ---------------------------------------------------
    #
    # Real on-device mask generation (`kernels.cu`'s "Dropout mask" section):
    # every element's Bernoulli draw happens inside `k_dropout_mask`, computed
    # from a stateless hash of (seed, element index). `rng` (the live
    # `numpy.random.Generator` -- `forge.random`'s default, or an explicit one
    # passed to `nn.Dropout`) is used for exactly one host-side scalar draw
    # (the seed) per call, never to generate the mask array itself -- no
    # NumPy involvement in the per-element randomness, no host round-trip, no
    # CPUBackend call. See `docs/architecture/cuda-backend.md`'s **CUDA
    # Dropout** section.

    def dropout_mask(self, a: CUDAStorage, p: float, rng: np.random.Generator) -> CUDAStorage:
        dtype = self._require_compute_dtype(a, op="dropout_mask")
        seed = int(rng.integers(0, 2**63 - 1))
        n = a.size
        out_ptr = self._alloc(n * dtype.itemsize)
        fn = getattr(self._lib, f"cf_dropout_mask_{_SUFFIX[dtype]}")
        code = fn(out_ptr, ctypes.c_longlong(n), ctypes.c_double(p), ctypes.c_uint64(seed), self._stream_handle())
        self._check(code, "dropout_mask")
        self._maybe_synchronize("dropout_mask")
        return CUDAStorage(out_ptr, a.shape, dtype, self._lib)

    # -- optimizer (Milestone 10) -----------------------------------------------

    def sgd_step(self, data: CUDAStorage, grad: CUDAStorage, lr: float) -> CUDAStorage:
        dtype = self._require_compute_dtype(data, grad, op="sgd_step")
        fn = getattr(self._lib, f"cf_sgd_step_{_SUFFIX[dtype]}")
        code = fn(data.ptr, grad.ptr, ctypes.c_double(lr), ctypes.c_longlong(data.size), self._stream_handle())
        self._check(code, "sgd_step")
        self._maybe_synchronize("sgd_step")
        return data

    # -- Adam optimizer (Milestone 17) ---------------------------------------
    #
    # One kernel launch does the entire update (moment estimates, bias
    # correction, and the parameter step) directly on the existing `data`/
    # `m`/`v` buffers -- no new `cudaMalloc`, no host round-trip for any
    # tensor value. Only the two bias-correction scalars are computed here,
    # in Python (`1 - beta**step`, via the standard library `**` -- a cheap
    # host-side float op on hyperparameter state, not tensor data) since
    # they are identical for every element and match `k_broadcast_scalar`'s
    # existing convention of precomputing a per-call scalar rather than
    # recomputing it once per thread.

    def adam_step(
        self, data: CUDAStorage, grad: CUDAStorage, m: CUDAStorage, v: CUDAStorage,
        lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, step: int,
    ) -> "tuple[CUDAStorage, CUDAStorage, CUDAStorage]":
        dtype = self._require_compute_dtype(data, grad, m, v, op="adam_step")
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        fn = getattr(self._lib, f"cf_adam_step_{_SUFFIX[dtype]}")
        code = fn(
            data.ptr, grad.ptr, m.ptr, v.ptr,
            ctypes.c_double(lr), ctypes.c_double(beta1), ctypes.c_double(beta2),
            ctypes.c_double(eps), ctypes.c_double(weight_decay),
            ctypes.c_double(bias_correction1), ctypes.c_double(bias_correction2),
            ctypes.c_longlong(data.size), self._stream_handle(),
        )
        self._check(code, "adam_step")
        self._maybe_synchronize("adam_step")
        return data, m, v

    # -- public synchronization (Milestone 11; exposed as forge.cuda.synchronize() in Milestone 26) --
    #
    # Every operation above already synchronizes internally before trusting
    # its own result, so `synchronize()` is never required for correctness.
    # It exists for external callers -- `benchmarks/timing.py`'s
    # synchronize-bracketed timing methodology, and, as of Milestone 26, the
    # public `forge.cuda.synchronize()` -- that need to bracket a *sequence*
    # of CUDA calls (or the boundary around one) with an explicit
    # synchronization point of their own, without reaching into the private
    # `_synchronize`/`_lib` internals. See `docs/architecture/cuda-backend.
    # md`'s **CUDA Execution and Synchronization Semantics (Milestone 26)**
    # section for the full contract this dispatches into.

    def synchronize(self) -> None:
        """Block until all previously issued CUDA work on this device completes."""
        self._synchronize("explicit synchronize")


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
