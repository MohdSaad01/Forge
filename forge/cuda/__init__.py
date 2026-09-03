"""Public CUDA memory-statistics API (Milestone 22).

```python
forge.cuda.memory_stats()             # -> CUDAMemoryStats
forge.cuda.reset_peak_memory_stats()  # resets peak only, live allocations untouched
```

Thin, explicit wrappers around `forge.backend.cuda.memory` (the actual
counters, instrumented at the real `cudaMalloc`/`cudaFree` boundary in
`forge/backend/cuda/backend.py`) -- mirroring how `forge.optim`/
`forge.serialization` are public packages fronting `forge.backend`-internal
implementation. See `docs/architecture/cuda-backend.md`'s **CUDA Memory
Statistics** section for the full semantics.

Importing `forge.cuda` itself never requires a CUDA-capable device or
`nvcc` -- it only imports pure-Python counters (see
`forge/backend/cuda/__init__.py`'s module docstring) -- so `import forge`
remains CUDA-optional. Only *calling* `memory_stats()`/
`reset_peak_memory_stats()` requires a working CUDA backend, raising
`forge.CUDAError` otherwise, matching every other CUDA-specific entry point
in Forge (e.g. `Tensor(..., device="cuda")` on a machine with no GPU).
"""

from __future__ import annotations

from ..backend.cuda.memory import CUDAMemoryStats
from ..exceptions import CUDAError


def _require_cuda() -> None:
    from ..backend.cuda.backend import is_cuda_available

    if not is_cuda_available():
        raise CUDAError(
            "forge.cuda.memory_stats()/reset_peak_memory_stats() require a working CUDA "
            "backend; CUDA is not available on this machine."
        )


def memory_stats() -> CUDAMemoryStats:
    """Return a snapshot of Forge's CUDA allocation accounting.

    Raises `forge.CUDAError` if CUDA is not available on this machine.
    """
    from ..backend.cuda.memory import memory_stats as _memory_stats

    _require_cuda()
    return _memory_stats()


def reset_peak_memory_stats() -> None:
    """Reset `peak_allocated_bytes` to the current `allocated_bytes`; live allocations are untouched.

    Raises `forge.CUDAError` if CUDA is not available on this machine.
    """
    from ..backend.cuda.memory import reset_peak_memory_stats as _reset_peak

    _require_cuda()
    _reset_peak()


__all__ = ["memory_stats", "reset_peak_memory_stats", "CUDAMemoryStats"]
