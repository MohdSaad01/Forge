"""Real CUDA pinned (page-locked) host memory (Milestone 29).

`PinnedMemory` owns one `cudaHostAlloc`d host buffer (`kernels.cu`'s
`cf_host_alloc`/`cf_host_free`, Milestone 29) -- genuine page-locked memory,
never an ordinary NumPy allocation pretending to be pinned. This is the host
side of `cudaMemcpyAsync`: the CUDA driver can DMA directly to/from a pinned
buffer without an internal staging copy, which is what makes a transfer using
it *actually* overlap with host/device execution rather than merely avoiding
one Python-level synchronization call. See `docs/architecture/cuda-transfers.md`.

**Lifetime** (Invariant 1 in the milestone brief): pinned memory must never be
`cudaFreeHost`d while an asynchronous CUDA operation may still reference it.
`PinnedMemory` tracks every in-flight transfer's completion `CUDAEvent` via
`_mark_pending()` (called by `CUDABackend.from_array_async`/`to_numpy_async`,
`backend.py`); `free()` (also `__del__`) waits for each one
(`CUDAEvent.synchronize()`) *before* the real `cudaFreeHost` call, rather than
either freeing prematurely (a use-after-free/data race) or leaking (never
freeing). This is deliberately simpler than `allocator.py`'s ready/pending
free-list model -- no caching, no reuse, direct `cudaHostAlloc`/`cudaFreeHost`
per the milestone brief's Section 25 ("start with direct lifecycle... do not
build a pinned caching allocator unless profiling demonstrates it is
necessary").

**NumPy interoperability**: `numpy()` returns a `_PinnedArray` -- a thin
`np.ndarray` subclass wrapping the pinned buffer with zero copies (a real
view via `np.ctypeslib.as_array`), whose `_pinned_owner` attribute is this
`PinnedMemory` instance. That back-reference is the whole mechanism by which
downstream Tensor/transfer lifetime works: as long as some NumPy array view
(and anything holding it, e.g. a `Tensor`) is reachable, plain CPython
refcounting keeps this `PinnedMemory` alive too, and its `__del__` cannot run
out from under an in-use buffer -- no `gc.collect()`, no manual bookkeeping,
matching this codebase's established lifetime-testing convention (see
`docs/development/progress.md`'s CUDA test notes).
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...exceptions import CUDAError

if TYPE_CHECKING:
    from .stream import CUDAEvent


def _format_cuda_error(lib: "ctypes.CDLL", code: int, action: str) -> str:
    message = lib.cf_error_string(code)
    message = message.decode() if message is not None else "<no message>"
    return f"{action}: {message} (code {code})."


# -- raw driver calls, bypassing all Forge-side bookkeeping ---------------------


def raw_host_alloc(lib: "ctypes.CDLL", nbytes: int) -> "ctypes.c_void_p":
    """A direct, uncached `cudaHostAlloc(max(nbytes, 1))`. Raises `CUDAError` on failure.

    A failed allocation here touches no counter and leaves no partial state --
    `cudaHostAlloc` either fully succeeds (one valid pointer) or fully fails
    (nothing to release), so there is nothing for Forge to clean up either way
    (Section 8 of the milestone brief).
    """
    ptr = ctypes.c_void_p()
    code = lib.cf_host_alloc(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 1)))
    if code != 0:
        raise CUDAError(_format_cuda_error(lib, code, f"CUDA pinned host allocation of {nbytes} bytes failed"))
    return ptr


def raw_host_free(lib: "ctypes.CDLL", ptr: "ctypes.c_void_p") -> None:
    """A direct `cudaFreeHost(ptr)`. Raises `CUDAError` on failure."""
    code = lib.cf_host_free(ptr)
    if code != 0:
        raise CUDAError(_format_cuda_error(lib, code, "cudaFreeHost failed"))


# -- pinned-memory statistics (Section 7) ----------------------------------------


@dataclass(frozen=True)
class PinnedMemoryStats:
    """A snapshot of Forge's pinned-host allocation accounting, separate from device memory.

    Deliberately its own small dataclass rather than new fields on
    `forge.backend.cuda.allocator.CUDAMemoryStats` -- pinned host bytes are a
    conceptually distinct resource from device `allocated_bytes`/
    `reserved_bytes`/`cached_bytes`/`pending_bytes` (Section 7/55 of the
    milestone brief: "do not mix host-pinned bytes into device
    `reserved_bytes`"), and every existing device-memory field/test stays
    byte-for-byte unaffected by this addition.
    """

    pinned_active_bytes: int
    pinned_peak_bytes: int
    pinned_allocation_count: int
    pinned_free_count: int

    def as_dict(self) -> "dict[str, int]":
        return {
            "pinned_active_bytes": self.pinned_active_bytes,
            "pinned_peak_bytes": self.pinned_peak_bytes,
            "pinned_allocation_count": self.pinned_allocation_count,
            "pinned_free_count": self.pinned_free_count,
        }


class _PinnedStatsTracker:
    """Process-wide pinned-allocation counters, guarded by one lock (matches `allocator.py`'s convention)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_bytes = 0
        self._peak_bytes = 0
        self._allocation_count = 0
        self._free_count = 0

    def record_alloc(self, nbytes: int) -> None:
        with self._lock:
            self._active_bytes += nbytes
            self._allocation_count += 1
            if self._active_bytes > self._peak_bytes:
                self._peak_bytes = self._active_bytes

    def record_free(self, nbytes: int) -> None:
        with self._lock:
            self._active_bytes -= nbytes
            self._free_count += 1

    def snapshot(self) -> PinnedMemoryStats:
        with self._lock:
            return PinnedMemoryStats(
                pinned_active_bytes=self._active_bytes,
                pinned_peak_bytes=self._peak_bytes,
                pinned_allocation_count=self._allocation_count,
                pinned_free_count=self._free_count,
            )


_stats = _PinnedStatsTracker()


def pinned_memory_stats() -> PinnedMemoryStats:
    """Internal -- see `forge.cuda.pinned_memory_stats`."""
    return _stats.snapshot()


# -- NumPy interoperability -------------------------------------------------------


class _PinnedArray(np.ndarray):
    """A `np.ndarray` view over a `PinnedMemory` buffer, carrying a strong back-reference.

    `_pinned_owner` is the whole lifetime mechanism (see this module's
    docstring): as long as this array (or anything sliced/viewed from it that
    NumPy keeps a `.base` chain to) is reachable, the owning `PinnedMemory`
    stays reachable too via ordinary refcounting.
    """

    _pinned_owner: "PinnedMemory | None"

    def __array_finalize__(self, obj: Any) -> None:
        self._pinned_owner = getattr(obj, "_pinned_owner", None)


def _view_as_pinned_array(ptr: "ctypes.c_void_p", nbytes: int, owner: "PinnedMemory") -> _PinnedArray:
    """A flat `uint8` `_PinnedArray` of `nbytes` elements over `ptr`, zero-copy."""
    if nbytes == 0:
        # `np.ctypeslib.as_array` rejects a NULL/zero-length buffer; an empty
        # view needs no real backing pointer to be well-defined.
        array = np.empty((0,), dtype=np.uint8)
    else:
        ctypes_array_type = ctypes.c_uint8 * nbytes
        buffer = ctypes.cast(ptr, ctypes.POINTER(ctypes_array_type)).contents
        array = np.ctypeslib.as_array(buffer)
    view = array.view(_PinnedArray)
    view._pinned_owner = owner
    return view


# -- PinnedMemory -----------------------------------------------------------------


class PinnedMemory:
    """A real, page-locked CUDA host allocation. See this module's docstring.

    Not constructed directly by ordinary Tensor code paths -- `Tensor.to(...,
    non_blocking=True)`'s D2H direction allocates one internally per transfer
    (`CUDABackend.to_numpy_async`); the H2D direction instead *requires* the
    caller to already hold one (Section 13/14: no hidden pinned staging for
    H2D). Direct construction (`forge.cuda.PinnedMemory(nbytes)`) is the
    public entry point for building H2D source buffers.
    """

    __slots__ = ("_ptr", "_nbytes", "_lib", "_freed", "_pending_events")

    def __init__(self, nbytes: int, lib: "ctypes.CDLL | None" = None) -> None:
        if lib is None:
            from .backend import get_cuda_backend

            lib = get_cuda_backend()._lib
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError(f"PinnedMemory size must be non-negative, got {nbytes}.")
        self._lib = lib
        self._nbytes = nbytes
        self._ptr = raw_host_alloc(lib, nbytes)
        self._freed = False
        self._pending_events: "list[CUDAEvent]" = []
        _stats.record_alloc(nbytes)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def is_freed(self) -> bool:
        return self._freed

    def numpy(self, shape: "tuple[int, ...] | None" = None, dtype: Any = np.uint8) -> np.ndarray:
        """A zero-copy NumPy view over this buffer, reshaped/retyped as requested.

        Defaults to a flat `uint8` view of the whole buffer. Raises
        `CUDAError` if this allocation has already been freed, or `ValueError`
        if `shape`/`dtype` do not fit within `nbytes`.
        """
        if self._freed:
            raise CUDAError("Cannot view a PinnedMemory allocation that has already been freed.")
        byte_view = _view_as_pinned_array(self._ptr, self._nbytes, self)
        if shape is None:
            return byte_view
        dtype = np.dtype(dtype)
        count = 1
        for dim in shape:
            count *= int(dim)
        needed = count * dtype.itemsize
        if needed > self._nbytes:
            raise ValueError(
                f"Requested shape {shape} of dtype '{dtype}' needs {needed} bytes, "
                f"but this PinnedMemory allocation is only {self._nbytes} bytes."
            )
        typed = byte_view[:needed].view(dtype).reshape(shape)
        typed._pinned_owner = self
        return typed

    # -- lifetime (Invariant 1) ---------------------------------------------

    def _mark_pending(self, event: "CUDAEvent") -> None:
        """Record that `event` must complete before this buffer may actually be freed."""
        if self._freed:
            raise CUDAError("Cannot submit an async transfer referencing an already-freed PinnedMemory allocation.")
        self._pending_events.append(event)

    def free(self) -> None:
        """Release this allocation back to the driver, waiting for any in-flight transfer first.

        Safe to call more than once (a no-op after the first successful
        call). Blocks the calling thread only if a transfer that referenced
        this buffer has not yet completed -- see this module's docstring.
        """
        if self._freed:
            return
        for event in self._pending_events:
            event.synchronize()
        self._pending_events = []
        ptr = self._ptr
        self._ptr = None
        self._freed = True
        raw_host_free(self._lib, ptr)
        _stats.record_free(self._nbytes)

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass  # interpreter shutdown may have already torn down module globals

    def __repr__(self) -> str:
        state = "freed" if self._freed else f"{self._nbytes} bytes"
        return f"PinnedMemory({state})"


__all__ = [
    "PinnedMemory",
    "PinnedMemoryStats",
    "pinned_memory_stats",
    "raw_host_alloc",
    "raw_host_free",
]
