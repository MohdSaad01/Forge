"""Forge's CUDA backend (Milestone 8; memory accounting in Milestone 22; caching allocator in Milestone 25; streams in Milestone 27).

Not imported by `forge.backend` at package-import time -- only pulled in
lazily by `forge.backend.get_backend()` when a `"cuda"` device is actually
requested, so a machine with no CUDA toolchain never pays any cost (or risks
any import-time failure) for a CPU-only session. See
`docs/architecture/cuda-backend.md`.

`.allocator` (Milestone 25), `.memory` (Milestone 22, now a thin re-export
of `.allocator`), and `.stream` (Milestone 27) are the exception to that
laziness: pure Python at import time -- no `ctypes` call is made, no `nvcc`
invoked, no device probed, until a `CUDAStream`/`CUDAEvent` is actually
constructed or a `CUDABackend` method actually runs -- so they are safe to
import unconditionally. See `forge/cuda/__init__.py`, the public entry point.
"""

from .allocator import CUDAMemoryStats, memory_stats, reset_peak_memory_stats
from .backend import CUDABackend, CUDAStorage, get_cuda_backend, is_cuda_available
from .stream import CUDAStream, current_stream, set_stream, stream_context

__all__ = [
    "CUDABackend",
    "CUDAStorage",
    "get_cuda_backend",
    "is_cuda_available",
    "CUDAMemoryStats",
    "memory_stats",
    "reset_peak_memory_stats",
    "CUDAStream",
    "current_stream",
    "set_stream",
    "stream_context",
]
