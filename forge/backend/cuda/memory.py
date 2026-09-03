"""CUDA allocation/free accounting (Milestone 22).

Instruments the real `cudaMalloc`/`cudaFree` boundary in `backend.py`
(`CUDABackend._alloc`, `CUDAStorage.__del__`) -- never `Tensor`/`CUDAStorage`
construction itself. Every counter here corresponds to one real CUDA driver
call that actually succeeded; a failed `cudaMalloc`/`cudaFree` never touches
these counters (see `CUDABackend._alloc` and `CUDAStorage.__del__`).

No caching allocator: this module holds only small integer counters, never a
`CUDAStorage` reference (which would keep otherwise-unreachable device memory
alive forever -- exactly the reference-cycle/leak hazard the milestone brief
warns against). It has no knowledge of which storage a given allocation
belongs to; it only tracks aggregate bytes/counts.

Thread safety: Forge is single-threaded in every other respect, so this is
deliberately the smallest mechanism that keeps concurrent `record_alloc`/
`record_free` calls from corrupting the counters -- one `threading.Lock`
around each read-modify-write, not a lock-free/atomic scheme.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class CUDAMemoryStats:
    """A snapshot of Forge's CUDA allocation accounting at one point in time.

    `allocated_bytes` is the sum of the logical byte sizes (`CUDAStorage.nbytes`,
    i.e. `size * dtype.itemsize`) of every currently live `CUDAStorage` --
    not the raw `cudaMalloc` request size, which is clamped to a minimum of 1
    byte for a zero-element tensor (see `CUDABackend._alloc`). `allocation_count`/
    `free_count` count real, successful `cudaMalloc`/`cudaFree` calls, so a
    zero-byte tensor still advances both counters by exactly one.
    """

    allocated_bytes: int
    peak_allocated_bytes: int
    allocation_count: int
    free_count: int

    def as_dict(self) -> "dict[str, int]":
        return {
            "allocated_bytes": self.allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "allocation_count": self.allocation_count,
            "free_count": self.free_count,
        }


class _MemoryTracker:
    """Process-wide counters, guarded by a single lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._allocated_bytes = 0
        self._peak_allocated_bytes = 0
        self._allocation_count = 0
        self._free_count = 0

    def record_alloc(self, nbytes: int) -> None:
        with self._lock:
            self._allocated_bytes += nbytes
            self._allocation_count += 1
            if self._allocated_bytes > self._peak_allocated_bytes:
                self._peak_allocated_bytes = self._allocated_bytes

    def record_free(self, nbytes: int) -> None:
        with self._lock:
            self._allocated_bytes -= nbytes
            self._free_count += 1

    def stats(self) -> CUDAMemoryStats:
        with self._lock:
            return CUDAMemoryStats(
                allocated_bytes=self._allocated_bytes,
                peak_allocated_bytes=self._peak_allocated_bytes,
                allocation_count=self._allocation_count,
                free_count=self._free_count,
            )

    def reset_peak(self) -> None:
        with self._lock:
            self._peak_allocated_bytes = self._allocated_bytes


_tracker = _MemoryTracker()


def record_alloc(nbytes: int) -> None:
    """Record one successful `cudaMalloc` of `nbytes` logical bytes. Internal -- called only by `CUDABackend._alloc`."""
    _tracker.record_alloc(nbytes)


def record_free(nbytes: int) -> None:
    """Record one successful `cudaFree` releasing `nbytes` logical bytes. Internal -- called only by `CUDAStorage.__del__`."""
    _tracker.record_free(nbytes)


def memory_stats() -> CUDAMemoryStats:
    """Return a snapshot of the current CUDA allocation counters. See `forge.cuda.memory_stats`."""
    return _tracker.stats()


def reset_peak_memory_stats() -> None:
    """Reset `peak_allocated_bytes` to the current `allocated_bytes`. See `forge.cuda.reset_peak_memory_stats`."""
    _tracker.reset_peak()


__all__ = [
    "CUDAMemoryStats",
    "memory_stats",
    "reset_peak_memory_stats",
    "record_alloc",
    "record_free",
]
