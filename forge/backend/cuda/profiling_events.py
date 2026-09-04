"""Profiling-only, timing-enabled CUDA events (Milestone 31).

Forge's internal `forge.backend.cuda.stream.CUDAEvent` is created with
`cudaEventDisableTiming` -- correct for its hot allocator/cross-stream-
dependency use, but unable to answer "how long did this GPU work take"
(`cudaEventElapsedTime` requires a timing-*enabled* event). `TimedEvent`
below is the one place Forge creates such an event. It is used only by
`benchmarks/pipeline_profile.py` -- never by any part of the core runtime,
so ordinary training never pays for it (Section 5/34/39 of the M31 brief:
real GPU-side timing via CUDA events rather than bracketing individual async
operations with `time.perf_counter()`, kept optional, outside the core
runtime, and with no Nsight/CUPTI/NVTX dependency -- this is a thin wrapper
around two native calls already compiled into the existing kernel library,
see `kernels.cu`'s matching "profiling-only timed events" section).

Usage (mirrors the module docstring's own "warmup -> synchronize -> record
start -> submit workload -> record end -> synchronize -> measure" recipe):

    start = TimedEvent(); start.record(stream_handle)
    ... submit GPU work on that stream ...
    end = TimedEvent(); end.record(stream_handle)
    end.synchronize()          # waits only for this event, not the whole device
    ms = elapsed_ms(start, end)
"""

from __future__ import annotations

import ctypes

from ...exceptions import CUDAError


def _lib() -> "ctypes.CDLL":
    from .backend import get_cuda_backend

    return get_cuda_backend()._lib


def _err(lib: "ctypes.CDLL", code: int, action: str) -> str:
    message = lib.cf_error_string(code)
    message = message.decode() if message is not None else "<no message>"
    return f"CUDA error during {action}: {message} (code {code})."


class TimedEvent:
    """A timing-enabled CUDA event (`cudaEventCreate`, no `cudaEventDisableTiming`)."""

    __slots__ = ("_ptr", "_lib")

    def __init__(self) -> None:
        lib = _lib()
        ptr = ctypes.c_void_p()
        code = lib.cf_event_create_timed(ctypes.byref(ptr))
        if code != 0:
            raise CUDAError(_err(lib, code, "cudaEventCreate (timed, profiling-only)"))
        self._ptr = ptr
        self._lib = lib

    def record(self, stream_handle: "ctypes.c_void_p | None" = None) -> None:
        code = self._lib.cf_event_record(self._ptr, stream_handle)
        if code != 0:
            raise CUDAError(_err(self._lib, code, "cudaEventRecord (timed, profiling-only)"))

    def synchronize(self) -> None:
        code = self._lib.cf_event_synchronize(self._ptr)
        if code != 0:
            raise CUDAError(_err(self._lib, code, "cudaEventSynchronize (timed, profiling-only)"))

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if not ptr:
            return
        try:
            self._lib.cf_event_destroy(ptr)
        except Exception:
            pass  # interpreter shutdown may have already torn down module globals


def elapsed_ms(start: TimedEvent, end: TimedEvent) -> float:
    """Milliseconds between two completed `TimedEvent`s (`cudaEventElapsedTime`).

    Both events must already have completed (call `.synchronize()` on `end`
    first) -- matches `cudaEventElapsedTime`'s own precondition.
    """
    lib = _lib()
    out = ctypes.c_float()
    code = lib.cf_event_elapsed_ms(start._ptr, end._ptr, ctypes.byref(out))
    if code != 0:
        raise CUDAError(_err(lib, code, "cudaEventElapsedTime"))
    return out.value


__all__ = ["TimedEvent", "elapsed_ms"]
