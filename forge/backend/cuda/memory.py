"""CUDA memory-statistics API surface (Milestone 22; caching allocator in Milestone 25).

The real bookkeeping -- the exact-size free-block cache, the active/reserved/
peak counters, cache hit/miss and driver-call counts -- lives in
`forge.backend.cuda.allocator` (Milestone 25) now that `CUDABackend._alloc()`
and `CUDAStorage.__del__()` go through a caching allocator rather than a
direct `cudaMalloc`/`cudaFree` per storage. This module is kept as a thin,
backwards-compatible re-export so `forge.backend.cuda.memory.CUDAMemoryStats`/
`memory_stats`/`reset_peak_memory_stats` -- the Milestone 22 import paths --
keep working unchanged; see `forge/cuda/__init__.py`, the public entry point,
and `allocator.py`'s module docstring for the actual allocator design.
"""

from __future__ import annotations

from .allocator import CUDAMemoryStats, memory_stats, reset_peak_memory_stats

__all__ = [
    "CUDAMemoryStats",
    "memory_stats",
    "reset_peak_memory_stats",
]
