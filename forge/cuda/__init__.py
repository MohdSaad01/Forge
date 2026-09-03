"""Public CUDA API: synchronization (Milestone 26) plus memory statistics (Milestone 22; caching allocator in Milestone 25).

```python
forge.cuda.synchronize()              # block until all issued CUDA work on this device completes
forge.cuda.memory_stats()             # -> CUDAMemoryStats (now active/reserved/cached, see below)
forge.cuda.reset_peak_memory_stats()  # resets peak only, live allocations untouched
forge.cuda.empty_cache()              # returns every cached (not active) block to the driver
```

Thin, explicit wrappers around `forge.backend.cuda` (`synchronize()` around
`CUDABackend.synchronize()`; the memory functions around `forge.backend.cuda.
allocator`, the caching allocator and its counters sitting between
`CUDABackend._alloc()`/`CUDAStorage.__del__()` and the real `cudaMalloc`/
`cudaFree` boundary -- see that module's docstring) -- mirroring how
`forge.optim`/`forge.serialization` are public packages fronting
`forge.backend`-internal implementation. See `docs/architecture/cuda-backend.
md`'s **CUDA Execution and Synchronization Semantics (Milestone 26)** section
for the full execution/synchronization contract, `docs/architecture/cuda-
memory-allocator.md` for the full allocator design, and `docs/architecture/
cuda-backend.md`'s **CUDA Memory Statistics** section for the memory-stats
field-by-field semantics.

`synchronize()` exists for callers that need an explicit host-side barrier
of their own (bracketing a benchmark measurement, or simply wanting a
device-idle checkpoint) -- it is never required for correctness anywhere
inside Forge itself: every CUDA-backed operation already synchronizes
internally before trusting its own result (see the **Kernel Launch
Semantics** section of the doc above), so Forge never returns a value to
Python that depended on still-in-flight device work.

Importing `forge.cuda` itself never requires a CUDA-capable device or
`nvcc` -- it only imports pure-Python counters (see
`forge/backend/cuda/__init__.py`'s module docstring) -- so `import forge`
remains CUDA-optional. Only *calling* `synchronize()`/`memory_stats()`/
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
            "forge.cuda.synchronize()/memory_stats()/reset_peak_memory_stats()/empty_cache() "
            "require a working CUDA backend; CUDA is not available on this machine."
        )


def synchronize() -> None:
    """Block the host until all previously issued Forge CUDA work on this device has completed.

    A thin wrapper around `CUDABackend.synchronize()` (`forge/backend/cuda/
    backend.py`), itself a direct `cudaDeviceSynchronize()` call -- no dummy
    kernel, no sleep/poll loop. Forge has exactly one CUDA device and one
    implicit stream (the CUDA default stream; see `docs/architecture/
    cuda-backend.md`'s **Stream Model** section), so there is no device or
    stream argument to pass.

    Calling this is never required for correctness anywhere inside Forge:
    every CUDA-backed `Tensor`/`Module`/`Loss`/`Optimizer` operation already
    calls `cudaDeviceSynchronize()` internally before returning its result
    (see that doc's **Kernel Launch Semantics** section), so by the time any
    Forge call returns, the work it issued has already completed. This
    function exists for callers that want an explicit host-side barrier of
    their own -- e.g. bracketing a benchmark measurement
    (`benchmarks/timing.py`) or a manual device-idle checkpoint. Calling it
    repeatedly, or when no CUDA work is outstanding, is always safe (a
    completed/empty queue synchronizes trivially).

    Raises `forge.CUDAError` if CUDA is not available on this machine, or if
    the underlying `cudaDeviceSynchronize()` call itself reports an error
    (e.g. an asynchronous kernel-execution error from a *previous* launch
    that had not yet been observed).
    """
    from ..backend.cuda.backend import get_cuda_backend

    _require_cuda()
    get_cuda_backend().synchronize()


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

    Safe to call with no prior `forge.cuda.synchronize()` -- a cached block
    was, by construction, released by a `CUDAStorage.__del__()` that already
    ran after the operation owning it had synchronized (see this module's
    **Milestone 26** docstring section above), so nothing this function
    could `cudaFree` is still potentially in use by an outstanding kernel.
    `empty_cache()` itself performs no extra synchronization -- it does not
    need to.
    """
    from ..backend.cuda.allocator import empty_cache as _empty_cache
    from ..backend.cuda.backend import get_cuda_backend

    _require_cuda()
    backend = get_cuda_backend()
    return _empty_cache(backend._lib)


__all__ = ["synchronize", "memory_stats", "reset_peak_memory_stats", "empty_cache", "CUDAMemoryStats", "profiler"]
