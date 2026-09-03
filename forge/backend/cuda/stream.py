"""Real CUDA streams, current-stream tracking, and internal CUDA events (Milestone 27).

`CUDAStream` wraps one real `cudaStreamCreate`d handle (backing the public
`forge.cuda.Stream`); `CUDAEvent` wraps one real `cudaEventCreate`d handle and
is used *only* internally, by `forge.backend.cuda.allocator`, to know when a
block released on a non-default stream has actually finished being used --
there is no public CUDA event API (per the Milestone 27 brief's explicit
scope limit).

**Current-stream state** is one process-global variable (`_current_stream`),
matching the "Forge is single-threaded elsewhere" convention `allocator.py`'s
own docstring already states -- not a per-thread/contextvar mechanism. `None`
means CUDA's own default (null) stream: the fully host-synchronous stream
Forge has always implicitly used through Milestone 26 (see
`docs/architecture/cuda-streams.md`'s **Default stream compatibility mode**
section for why this stays the default rather than a real created stream).
`CUDABackend` (`backend.py`) reads `current_stream()` at the start of every
kernel-launching method to decide which stream to launch on and whether to
synchronize afterward -- it never receives a stream as an explicit argument,
matching the milestone brief's "do not require users to pass a stream
argument to every Tensor operation" requirement.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from typing import Any

from ...exceptions import CUDAError

# cudaError_t enum value for "the query would have blocked" (`cudaErrorNotReady`,
# `cuda_runtime_api.h`) -- verified directly against the real driver on the
# development machine (CUDA 12.6, driver 582.53: `cudaEventQuery()` on a
# not-yet-complete event returns 600, whose `cudaGetErrorString()` reads
# "device not ready"). A non-zero code that is neither this nor 0 (success)
# is a genuine CUDA error.
_CUDA_ERROR_NOT_READY = 600


def _err(lib: "ctypes.CDLL", code: int, action: str) -> str:
    message = lib.cf_error_string(code)
    message = message.decode() if message is not None else "<no message>"
    return f"CUDA error during {action}: {message} (code {code})."


class CUDAStream:
    """A real CUDA stream: owns one `cudaStreamCreate`d handle.

    Backs the public `forge.cuda.Stream`. Creation and destruction are real
    CUDA runtime calls (`cf_stream_create`/`cf_stream_destroy` in
    `kernels.cu`) -- never simulated, and never the default/null stream
    handle wearing a `CUDAStream` costume (the default stream is represented
    by `current_stream() is None` instead; see this module's docstring).

    A `CUDAStorage` last used on this stream holds a strong reference to it
    (`CUDAStorage.last_stream`, `backend.py`) for as long as that storage is
    alive -- necessary so the stream handle stays valid for the
    `cudaEventRecord` call `CUDAStorage.__del__` makes at release time (see
    `allocator.py`'s pending-block design). Once an event has actually been
    recorded on a stream, the event remains valid to query/synchronize even
    after that stream is later destroyed (a guarantee documented CUDA
    runtime behavior) -- so nothing needs to keep the stream alive beyond
    that point.
    """

    __slots__ = ("_ptr", "_lib", "_destroyed")

    def __init__(self) -> None:
        from .backend import get_cuda_backend

        backend = get_cuda_backend()
        self._lib = backend._lib
        ptr = ctypes.c_void_p()
        code = self._lib.cf_stream_create(ctypes.byref(ptr))
        if code != 0:
            raise CUDAError(_err(self._lib, code, "cudaStreamCreate"))
        self._ptr = ptr
        self._destroyed = False

    @property
    def handle(self) -> "ctypes.c_void_p | None":
        """The raw `cudaStream_t`, as a `ctypes.c_void_p` -- for internal use only."""
        return None if self._destroyed else self._ptr

    def synchronize(self) -> None:
        """Block the host until every previously issued operation on this stream completes.

        Raises `forge.CUDAError` if this stream has already been destroyed,
        or if the underlying `cudaStreamSynchronize()` call itself reports
        an error (including a surfaced asynchronous execution error from a
        kernel issued to this stream -- see Invariant 7 in
        `docs/architecture/cuda-streams.md`).
        """
        if self._destroyed:
            raise CUDAError("Cannot synchronize a Stream that has already been destroyed.")
        code = self._lib.cf_stream_synchronize(self._ptr)
        if code != 0:
            raise CUDAError(_err(self._lib, code, "cudaStreamSynchronize"))

    def destroy(self) -> None:
        """Explicitly destroy the underlying CUDA stream. Safe to call more than once."""
        if getattr(self, "_destroyed", True) or not getattr(self, "_ptr", None):
            self._destroyed = True
            return
        ptr = self._ptr
        self._destroyed = True
        try:
            self._lib.cf_stream_destroy(ptr)
        except Exception:
            pass  # interpreter shutdown may have already torn down module globals

    def __del__(self) -> None:
        # `getattr(..., True)` above means a `Stream` whose `__init__` raised
        # before `CUDAStream.__init__` ran (e.g. a subclass's own
        # `_require_cuda()` check failing first -- see `forge.cuda.Stream`)
        # destroys as a safe no-op here rather than raising `AttributeError`
        # out of `__del__` (which Python can only report, never propagate).
        self.destroy()

    def __repr__(self) -> str:
        if self._destroyed:
            return "Stream(<destroyed>)"
        return f"Stream({hex(self._ptr.value or 0)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CUDAStream):
            return NotImplemented
        return (self._ptr.value or 0) == (other._ptr.value or 0)

    def __hash__(self) -> int:
        return hash(self._ptr.value or 0)


class CUDAEvent:
    """A real CUDA event, used only internally by `allocator.py` for stream-safe block reuse.

    Not exposed as public API anywhere (`forge.cuda` has no `Event`) -- see
    the Milestone 27 brief's explicit scope limit ("public CUDA event APIs"
    are out of scope). Created with `cudaEventDisableTiming`: Forge's
    internal events are only ever queried/waited for completion, never used
    to measure elapsed time.
    """

    __slots__ = ("_ptr", "_lib")

    def __init__(self, lib: "ctypes.CDLL") -> None:
        ptr = ctypes.c_void_p()
        code = lib.cf_event_create(ctypes.byref(ptr))
        if code != 0:
            raise CUDAError(_err(lib, code, "cudaEventCreate"))
        self._ptr = ptr
        self._lib = lib

    def record(self, stream_handle: "ctypes.c_void_p | None") -> None:
        """Record this event on `stream_handle` (`None` for the default stream)."""
        code = self._lib.cf_event_record(self._ptr, stream_handle)
        if code != 0:
            raise CUDAError(_err(self._lib, code, "cudaEventRecord"))

    def query(self) -> bool:
        """True if every operation preceding this event's `record()` has completed."""
        code = self._lib.cf_event_query(self._ptr)
        if code == 0:
            return True
        if code == _CUDA_ERROR_NOT_READY:
            return False
        raise CUDAError(_err(self._lib, code, "cudaEventQuery"))

    def synchronize(self) -> None:
        """Block the host until this event (and everything recorded before it) completes."""
        code = self._lib.cf_event_synchronize(self._ptr)
        if code != 0:
            raise CUDAError(_err(self._lib, code, "cudaEventSynchronize"))

    def __del__(self) -> None:
        ptr = self._ptr
        if not ptr:
            return
        self._ptr = None
        try:
            self._lib.cf_event_destroy(ptr)
        except Exception:
            pass  # interpreter shutdown may have already torn down module globals


# -- current-stream tracking ------------------------------------------------

_current_stream: "CUDAStream | None" = None


def current_stream() -> "CUDAStream | None":
    """The stream Forge CUDA operations issued right now will execute on.

    `None` means CUDA's default (null) stream -- Forge's host-synchronous
    compatibility mode (see this module's docstring and
    `docs/architecture/cuda-streams.md`).
    """
    return _current_stream


def set_stream(stream: "CUDAStream | None") -> "CUDAStream | None":
    """Set the current Forge CUDA stream, returning whatever was current before.

    `stream=None` restores the default (host-synchronous) stream. Raises
    `TypeError` for anything other than a `CUDAStream` or `None`.
    """
    global _current_stream
    if stream is not None and not isinstance(stream, CUDAStream):
        raise TypeError(f"forge.cuda.set_stream() requires a Stream or None, got {type(stream).__name__}.")
    previous = _current_stream
    _current_stream = stream
    return previous


@contextmanager
def stream_context(stream: "CUDAStream"):
    """`with stream_context(s): ...` -- makes `s` current for the block, then restores the prior stream."""
    if not isinstance(stream, CUDAStream):
        raise TypeError(f"forge.cuda.stream() requires a Stream, got {type(stream).__name__}.")
    previous = set_stream(stream)
    try:
        yield stream
    finally:
        set_stream(previous)


__all__ = ["CUDAStream", "CUDAEvent", "current_stream", "set_stream", "stream_context"]
