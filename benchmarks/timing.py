"""Benchmark timing methodology (Milestone 11).

CPU timing uses `time.perf_counter()` -- Python's monotonic, highest-
resolution wall-clock timer, unaffected by system clock adjustments.

CUDA execution is asynchronous: a kernel launch returns to Python as soon
as the launch is *queued*, not once it has actually run. Naively wrapping
`start = perf_counter(); cuda_op(); end = perf_counter()` measures launch
overhead, not execution time. Every CUDA measurement here instead follows:

    synchronize()          # drain any prior in-flight work
    start = perf_counter()
    cuda_op()               # launches, returns immediately
    synchronize()           # block until the launched work actually finishes
    end = perf_counter()

`synchronize()` is `CUDABackend.synchronize()` (`forge/backend/cuda/backend.py`),
a thin wrapper around the same `cudaDeviceSynchronize()` every CUDA
operation already calls internally before trusting its own result (see
`docs/architecture/cuda-backend.md`) -- this file adds no new native code,
only reuses that existing, already-verified synchronization point.

Every measured callable is first run `warmup` times, uncounted, before any
timing starts -- this absorbs first-call effects (CUDA context/lazy kernel-
module initialization, the one-time `nvcc` compile-cache check, CPU cache
warming) that would otherwise inflate the first few measured iterations and
misrepresent steady-state performance. See `docs/performance/benchmarking.md`.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Timing:
    """One measured operation's raw per-iteration durations, in seconds."""

    durations: "tuple[float, ...]"
    warmup: int
    iterations: int

    def __post_init__(self):
        if len(self.durations) != self.iterations:
            raise ValueError(
                f"Timing recorded {len(self.durations)} durations but iterations={self.iterations}."
            )

    @property
    def mean(self) -> float:
        return statistics.mean(self.durations)

    @property
    def median(self) -> float:
        return statistics.median(self.durations)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.durations) if len(self.durations) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.durations)

    @property
    def max(self) -> float:
        return max(self.durations)


def _cuda_synchronize() -> None:
    from forge.backend.cuda.backend import get_cuda_backend

    get_cuda_backend().synchronize()


def time_cpu(fn: Callable[[], object], warmup: int, iterations: int) -> Timing:
    """Time a CPU callable, called repeatedly (the same graph/state each time)."""
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        durations.append(end - start)
    return Timing(tuple(durations), warmup, iterations)


def time_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> Timing:
    """Time a CUDA callable with explicit synchronization bracketing each iteration."""
    for _ in range(warmup):
        fn()
    _cuda_synchronize()
    durations = []
    for _ in range(iterations):
        _cuda_synchronize()
        start = time.perf_counter()
        fn()
        _cuda_synchronize()
        end = time.perf_counter()
        durations.append(end - start)
    return Timing(tuple(durations), warmup, iterations)


def time_calls(calls: "Iterable[Callable[[], object]]", warmup: int, iterations: int, device: str) -> Timing:
    """Time `warmup + iterations` *distinct* zero-arg callables, one call each.

    Needed for backward-pass benchmarks: a Tensor's non-leaf `grad_fn` is
    freed the moment `backward()` consumes it (see
    `docs/architecture/autograd.md`'s "Graph freed on use"), so calling
    `backward()` a second time on the same output raises `GradientStateError`
    -- unlike a forward op, a backward measurement cannot simply repeat one
    call. Callers build one fresh forward pass per call up front (outside
    any timing) and pass a `y.backward` (or equivalent) closure per call;
    this function only times the calls themselves.
    """
    calls = list(calls)
    expected = warmup + iterations
    if len(calls) != expected:
        raise ValueError(f"Expected {expected} calls (warmup={warmup} + iterations={iterations}), got {len(calls)}.")

    sync = _cuda_synchronize if device == "cuda" else (lambda: None)

    for fn in calls[:warmup]:
        fn()
    sync()
    durations = []
    for fn in calls[warmup:]:
        sync()
        start = time.perf_counter()
        fn()
        sync()
        end = time.perf_counter()
        durations.append(end - start)
    return Timing(tuple(durations), warmup, iterations)
