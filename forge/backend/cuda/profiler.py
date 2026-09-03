"""CUDA allocation profiling (Milestone 24) -- optional, low-overhead diagnostic.

Distinct from `forge/backend/cuda/memory.py`'s `CUDAMemoryStats` (Milestone
22): that module answers "what is the current/peak allocation state?" via a
handful of running counters. This module answers "what allocation *behavior*
produced that state?" -- a chronological trace of individual alloc/free
events, recorded only while a profiler is explicitly started.

Instrumented at exactly the same two real `cudaMalloc`/`cudaFree` call sites
`memory.py` already instruments (`CUDABackend._alloc`, `CUDAStorage.__del__`,
in `backend.py`) -- no new call sites, no per-operation instrumentation
elsewhere. `record()` is called unconditionally from `memory.py`'s
`record_alloc`/`record_free`, so the *disabled* cost is exactly one
`bool` attribute check per allocation/free -- no `AllocationEvent`
construction, no `time.perf_counter()` call, no list append -- matching the
milestone's "low overhead when disabled" requirement.

**Never retains a `CUDAStorage`, `Tensor`, or any other Forge object.**
`record()`'s signature is `(kind: str, nbytes: int, block_id: int)` -- three
primitives. `block_id` is the raw integer value of the `cudaMalloc`-returned
pointer (`ctypes.c_void_p.value`), used only to correlate an allocation with
its eventual free in offline analysis (`benchmarks/alloc_analysis.py`); it is
never dereferenced and never kept the underlying memory alive a moment
longer than it already would be. A profiler left running indefinitely grows
one small dataclass instance per allocation/free -- bounded by how long
profiling runs and how much the caller allocates, entirely under the
caller's control via `reset()`.

Categorization is opt-in via `tag()`, a context manager pushing a name onto a
small stack -- not per-call-site instrumentation of `CUDABackend`'s ~40
methods. A caller wraps the code region it wants labeled (e.g. `with
tag("forward"): ...`), and every allocation/free recorded while that tag is
active is stamped with it. Nested tags use the innermost (top-of-stack) name.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationEvent:
    """One real, successful `cudaMalloc` or `cudaFree` call, primitives only.

    `timestamp` is `time.perf_counter()` at the moment the underlying driver
    call returned -- comparable only to other timestamps from the same
    process, per the standard library's own guarantee for that clock.
    `block_id` correlates an "alloc" event with its later "free" event (the
    same underlying pointer value); it is opaque data, not a live reference,
    and a value may be reused across allocations after a free (a caller
    pairing events by `block_id` should do so in chronological order -- see
    `benchmarks/alloc_analysis.py`'s `pair_lifetimes`).
    """

    kind: str  # "alloc" | "free"
    nbytes: int
    timestamp: float
    block_id: int
    category: "str | None"


class CUDAMemoryProfiler:
    """Process-wide allocation-event collector. Inactive (and free) by default."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._events: "list[AllocationEvent]" = []
        self._tag_stack: "list[str]" = []

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        with self._lock:
            self._active = True

    def stop(self) -> None:
        with self._lock:
            self._active = False

    def reset(self) -> None:
        """Discard all recorded events. Does not change `active`/start-stop state."""
        with self._lock:
            self._events = []

    def events(self) -> "tuple[AllocationEvent, ...]":
        """A snapshot of every event recorded since the last `reset()`."""
        with self._lock:
            return tuple(self._events)

    def record(self, kind: str, nbytes: int, block_id: int) -> None:
        """Internal -- called only by `forge.backend.cuda.memory.record_alloc`/`record_free`."""
        if not self._active:
            return
        event = AllocationEvent(
            kind=kind,
            nbytes=nbytes,
            timestamp=time.perf_counter(),
            block_id=block_id,
            category=self._tag_stack[-1] if self._tag_stack else None,
        )
        with self._lock:
            if self._active:  # re-checked under the lock: a race with stop() must not record
                self._events.append(event)

    @contextmanager
    def tag(self, name: str):
        """Label every allocation/free recorded during this block with `name`.

        Not thread-safe by design -- Forge is single-threaded elsewhere too
        (see `memory.py`'s own convention) -- a plain list used as a stack,
        no lock, since only `start`/`stop`/`reset`/`events` (called
        concurrently with recording, in principle) need the lock.
        """
        self._tag_stack.append(name)
        try:
            yield
        finally:
            self._tag_stack.pop()


_profiler = CUDAMemoryProfiler()


def get_profiler() -> CUDAMemoryProfiler:
    """Return the process-wide `CUDAMemoryProfiler` singleton."""
    return _profiler


__all__ = ["AllocationEvent", "CUDAMemoryProfiler", "get_profiler"]
