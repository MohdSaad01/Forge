"""Cross-stream dependency overhead benchmark (Milestone 28).

    python -m benchmarks.stream_dependency_bench

Standalone script, not part of `python -m benchmarks`'s category list --
matching `allocator_bench.py`/`stream_bench.py`'s own "diagnostic script, not
a stable-schema benchmark category" status (Section 45 of the milestone
brief asks for benchmark *methodology*, not a new `results.json` category).

Measures, each as median-of-`TRIALS` wall-clock time over `REPEATS`
iterations (`forge.cuda.synchronize()`-bracketed, per `timing.py`'s
methodology):

- **same-stream baseline**: a chained add loop entirely on one explicit
  stream -- the fast path (`CUDABackend._stream_guard` establishes no
  dependency at all, Section 12 of the milestone brief).
- **cross-stream dependency**: the identical loop, but every iteration's
  operand is handed from the *other* stream -- forces one `cudaEventRecord`
  + one `cudaStreamWaitEvent` per iteration (Section 45: "record event +
  stream wait").
- **multi-input dependency**: each iteration combines two operands, each
  produced on a distinct stream -- two dependencies established per
  iteration (Section 45: "multiple producer streams").
- **event creation**: `CUDAEvent` construction/destruction in isolation, no
  stream/wait involved -- isolates that cost from the two measurements
  above.
- **cross-stream allocator reuse**: release a block on stream A, immediately
  request the same size back (still on stream A) -- the pending -> ready
  transition `allocator.py`'s `_try_reclaim_pending` performs, timed
  separately from ordinary same-stream cache-hit reuse for comparison.

Run `python -m benchmarks.mnist_bench` separately for the M20 CNN/MNIST
representative-workload comparison (Section 45's last bullet) -- that
benchmark is untouched by this milestone (it never uses an explicit stream,
so it never exercises `_stream_guard`'s new cross-stream path) and is not
duplicated here.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import forge
from forge.backend.cuda import allocator as cuda_allocator
from forge.backend.cuda import stream as cuda_stream
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available

WORKLOAD_ELEMENTS = 4_096
REPEATS = 300
TRIALS = 7
EVENT_REPEATS = 2_000


def _median_seconds(fn, repeats: int, trials: int) -> float:
    samples = []
    for _ in range(trials):
        forge.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        forge.cuda.synchronize()
        end = time.perf_counter()
        samples.append((end - start) / repeats)
    return statistics.median(samples)


def _bench_same_stream_baseline() -> float:
    s = forge.cuda.Stream()
    rng = np.random.default_rng(1)
    with forge.cuda.stream(s):
        x = forge.Tensor(rng.standard_normal(WORKLOAD_ELEMENTS).astype(np.float32), device="cuda")
        b = forge.Tensor(rng.standard_normal(WORKLOAD_ELEMENTS).astype(np.float32), device="cuda")

        def step():
            nonlocal x
            x = x + b

        return _median_seconds(step, REPEATS, TRIALS)


def _bench_cross_stream_dependency() -> float:
    """Every iteration's operand was last produced on the *other* stream -- one dependency each time."""
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    rng = np.random.default_rng(2)
    b = forge.Tensor(rng.standard_normal(WORKLOAD_ELEMENTS).astype(np.float32), device="cuda")
    with forge.cuda.stream(stream_a):
        x = forge.Tensor(rng.standard_normal(WORKLOAD_ELEMENTS).astype(np.float32), device="cuda")

    streams = (stream_a, stream_b)
    state = {"x": x, "i": 0}

    def step():
        current = streams[state["i"] % 2]
        state["i"] += 1
        with forge.cuda.stream(current):
            state["x"] = state["x"] + b  # state["x"]'s last_stream is always the *other* stream

    return _median_seconds(step, REPEATS, TRIALS)


def _bench_multi_input_dependency() -> float:
    """Each iteration combines two operands, each last produced on a distinct stream."""
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    stream_c = forge.cuda.Stream()
    rng = np.random.default_rng(3)

    def step():
        with forge.cuda.stream(stream_a):
            a = forge.Tensor(rng.standard_normal(WORKLOAD_ELEMENTS).astype(np.float32), device="cuda")
        with forge.cuda.stream(stream_b):
            b = forge.Tensor(rng.standard_normal(WORKLOAD_ELEMENTS).astype(np.float32), device="cuda")
        with forge.cuda.stream(stream_c):
            _ = a + b  # two distinct producer streams -> two dependencies

    return _median_seconds(step, REPEATS, TRIALS)


def _bench_event_creation() -> float:
    """`CUDAEvent` construction + destruction alone, no record/wait -- isolates that fixed cost."""
    backend = get_cuda_backend()
    lib = backend._lib

    def step():
        cuda_stream.CUDAEvent(lib)  # unreferenced -> __del__ (cf_event_destroy) runs immediately

    forge.cuda.synchronize()
    samples = []
    for _ in range(TRIALS):
        start = time.perf_counter()
        for _ in range(EVENT_REPEATS):
            step()
        end = time.perf_counter()
        samples.append((end - start) / EVENT_REPEATS)
    return statistics.median(samples)


def _bench_cross_stream_allocator_reuse() -> float:
    """Release on stream A, immediately request the same size back on stream A -- pending -> ready."""
    shape_bytes = WORKLOAD_ELEMENTS * 4
    s = forge.cuda.Stream()
    cuda_allocator.empty_cache(get_cuda_backend()._lib)

    def step():
        with forge.cuda.stream(s):
            t = forge.Tensor(np.zeros(WORKLOAD_ELEMENTS, dtype=np.float32), device="cuda")
        del t  # -> pending block, event recorded on `s`
        s.synchronize()  # force the event complete so the next allocate() can reclaim it
        with forge.cuda.stream(s):
            t2 = forge.Tensor(np.zeros(WORKLOAD_ELEMENTS, dtype=np.float32), device="cuda")
        del t2
        s.synchronize()

    return _median_seconds(step, 40, TRIALS)


def main() -> None:
    if not is_cuda_available():
        print("CUDA is not available on this machine -- stream_dependency_bench requires real CUDA hardware.")
        return

    print(f"Workload: {WORKLOAD_ELEMENTS:,} float32 elements/op, {REPEATS} repeats, {TRIALS} trials (median reported).\n")

    same_stream = _bench_same_stream_baseline()
    cross_stream = _bench_cross_stream_dependency()
    multi_input = _bench_multi_input_dependency()
    event_creation = _bench_event_creation()
    allocator_reuse = _bench_cross_stream_allocator_reuse()

    print(f"same-stream baseline (no dependency):                    {same_stream * 1e6:8.2f} us/op")
    print(f"cross-stream dependency (1 producer):                    {cross_stream * 1e6:8.2f} us/op")
    print(f"multi-input dependency (2 producers, incl. producing both): {multi_input * 1e6:8.2f} us/op")
    print(f"event creation + destruction (isolated):                 {event_creation * 1e6:8.2f} us/event")
    print(f"cross-stream allocator reuse (release+alloc):            {allocator_reuse * 1e6:8.2f} us/cycle")
    print()
    if same_stream > 0:
        print(f"cross-stream / same-stream overhead ratio: {cross_stream / same_stream:.2f}x")
    print(
        "Same-stream should remain close to Milestone 27 performance (no event/wait on that path); "
        "the cross-stream/multi-input numbers above are the real, GPU-side-only cost of "
        "cudaEventRecord + cudaStreamWaitEvent this milestone adds -- see docs/architecture/"
        "cuda-streams.md's Automatic cross-stream dependencies section for the full contract, and "
        "run `python -m benchmarks.mnist_bench` separately for the existing-workload comparison."
    )


if __name__ == "__main__":
    main()
