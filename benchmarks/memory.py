"""CUDA memory-stats reporting for benchmarks (Milestone 22).

Complements the existing timing methodology (`timing.py`) -- never replaces
it. `cuda_memory_extra()` turns a before/after `forge.cuda.memory_stats()`
pair into a plain dict merged into a `BenchmarkResult.extra` (Milestone 11),
so existing benchmark result files/consumers that only read the established
`BenchmarkResult` fields are unaffected; only CUDA-device results gain these
extra keys. CPU benchmarks never call this -- CUDA memory statistics have no
CPU meaning (`docs/architecture/cuda-backend.md`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.backend.cuda.memory import CUDAMemoryStats


def cuda_memory_extra(before: "CUDAMemoryStats", after: "CUDAMemoryStats") -> "dict[str, int]":
    """Build the `extra`-dict fragment describing a workload's CUDA memory behavior.

    `before`/`after` should bracket the *timed* portion of a benchmark, with
    `forge.cuda.reset_peak_memory_stats()` called once, right before `before`
    is captured, so `after.peak_allocated_bytes` reflects only that workload's
    own peak rather than an earlier, unrelated one.
    """
    return {
        "cuda_allocated_before_bytes": before.allocated_bytes,
        "cuda_peak_allocated_bytes": after.peak_allocated_bytes,
        "cuda_allocated_after_bytes": after.allocated_bytes,
        "cuda_allocation_count_delta": after.allocation_count - before.allocation_count,
        "cuda_free_count_delta": after.free_count - before.free_count,
    }


__all__ = ["cuda_memory_extra"]
