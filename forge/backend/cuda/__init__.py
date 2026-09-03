"""Forge's CUDA backend (Milestone 8; memory accounting in Milestone 22; caching allocator in Milestone 25).

Not imported by `forge.backend` at package-import time -- only pulled in
lazily by `forge.backend.get_backend()` when a `"cuda"` device is actually
requested, so a machine with no CUDA toolchain never pays any cost (or risks
any import-time failure) for a CPU-only session. See
`docs/architecture/cuda-backend.md`.

`.allocator` (Milestone 25) and `.memory` (Milestone 22, now a thin
re-export of `.allocator`) are the one exception to that laziness: pure
Python bookkeeping (no `ctypes` calls made at import time, no `nvcc`, no
device probe) instrumenting `CUDABackend._alloc`/`CUDAStorage.__del__`
below, so they are safe to import unconditionally -- see `forge/cuda/
__init__.py`, the public entry point.
"""

from .allocator import CUDAMemoryStats, memory_stats, reset_peak_memory_stats
from .backend import CUDABackend, CUDAStorage, get_cuda_backend, is_cuda_available

__all__ = [
    "CUDABackend",
    "CUDAStorage",
    "get_cuda_backend",
    "is_cuda_available",
    "CUDAMemoryStats",
    "memory_stats",
    "reset_peak_memory_stats",
]
