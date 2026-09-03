"""Public CUDA memory-statistics API (Milestone 22; caching allocator in Milestone 25).

```python
forge.cuda.memory_stats()             # -> CUDAMemoryStats (now active/reserved/cached, see below)
forge.cuda.reset_peak_memory_stats()  # resets peak only, live allocations untouched
forge.cuda.empty_cache()              # returns every cached (not active) block to the driver
```

Thin, explicit wrappers around `forge.backend.cuda.allocator` (the actual
caching allocator and its counters, sitting between `CUDABackend._alloc()`/
`CUDAStorage.__del__()` and the real `cudaMalloc`/`cudaFree` boundary -- see
that module's docstring) -- mirroring how `forge.optim`/`forge.serialization`
are public packages fronting `forge.backend`-internal implementation. See
`docs/architecture/cuda-memory-allocator.md` for the full allocator design
and `docs/architecture/cuda-backend.md`'s **CUDA Memory Statistics** section
for the field-by-field semantics.

Importing `forge.cuda` itself never requires a CUDA-capable device or
`nvcc` -- it only imports pure-Python counters (see
`forge/backend/cuda/__init__.py`'s module docstring) -- so `import forge`
remains CUDA-optional. Only *calling* `memory_stats()`/
`reset_peak_memory_stats()`/`empty_cache()` requires a working CUDA backend,
raising `forge.CUDAError` otherwise, matching every other CUDA-specific
entry point in Forge (e.g. `Tensor(..., device="cuda")` on a machine with no
GPU).
"""

from __future__ import annotations

from ..backend.cuda.allocator import CUDAMemoryStats
from ..exceptions import CUDAError
from . import profiler


def _require_cuda() -> None:
    from ..backend.cuda.backend import is_cuda_available

    if not is_cuda_available():
        raise CUDAError(
            "forge.cuda.memory_stats()/reset_peak_memory_stats()/empty_cache() require a working "
            "CUDA backend; CUDA is not available on this machine."
        )


def memory_stats() -> CUDAMemoryStats:
    """Return a snapshot of Forge's CUDA allocation accounting.

    Raises `forge.CUDAError` if CUDA is not available on this machine.
    """
    from ..backend.cuda.allocator import memory_stats as _memory_stats

    _require_cuda()
    return _memory_stats()


def reset_peak_memory_stats() -> None:
    """Reset `peak_allocated_bytes`/`peak_reserved_bytes` to their current values; live allocations are untouched.

    Raises `forge.CUDAError` if CUDA is not available on this machine.
    """
    from ..backend.cuda.allocator import reset_peak_memory_stats as _reset_peak

    _require_cuda()
    _reset_peak()


def empty_cache() -> int:
    """Release every currently *cached* (not active) CUDA allocation back to the driver.

    Live Tensor/Parameter/Adam-state storage is never touched -- only blocks
    a `CUDAStorage` has already released to the allocator's exact-size cache
    (see `forge.backend.cuda.allocator`). Returns the number of blocks
    actually freed. Raises `forge.CUDAError` if CUDA is not available, or if
    a `cudaFree` call itself fails partway through (see `CUDACachingAllocator.
    empty_cache`'s docstring for the partial-failure behavior in that case).
    """
    from ..backend.cuda.allocator import empty_cache as _empty_cache
    from ..backend.cuda.backend import get_cuda_backend

    _require_cuda()
    backend = get_cuda_backend()
    return _empty_cache(backend._lib)


__all__ = ["memory_stats", "reset_peak_memory_stats", "empty_cache", "CUDAMemoryStats", "profiler"]
