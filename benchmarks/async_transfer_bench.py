"""Pinned-vs-pageable and async-transfer/compute-overlap benchmarks (Milestone 29).

    python -m benchmarks.async_transfer_bench

Standalone script, not part of `python -m benchmarks`'s category list --
matching `stream_bench.py`/`stream_dependency_bench.py`'s own "diagnostic
script, not a stable-schema benchmark category" status.

Three things are measured, each following `benchmarks/timing.py`'s
synchronize-bracketed methodology (`docs/performance/benchmarking.md`):

1. **Pinned vs. pageable transfer**: a pageable synchronous `.to()` against a
   pinned asynchronous `.to(..., non_blocking=True)` immediately followed by
   `forge.cuda.synchronize()` (so both numbers represent "submit and wait for
   completion," a fair apples-to-apples comparison), at the same sizes
   `transfer_bench.py` (Milestone 11) already uses.
2. **Async submission latency vs. synchronized completion**: the host-side
   cost of *issuing* `cudaMemcpyAsync` (returns to Python almost
   immediately) versus the cost of actually waiting for it to complete --
   reported separately, per Section 39 of the milestone brief ("clearly
   label the two measurements").
3. **Transfer/compute overlap**: an H2D transfer issued on one stream
   concurrently with independent compute issued on another, compared against
   the same two workloads run strictly sequentially -- proving Forge does
   not implicitly serialize an async transfer against unrelated compute
   (Section 36/37). As with `stream_bench.py`: "do not expect dramatic
   overlap on every kernel/GPU" -- whatever ratio is measured is reported
   as-is.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import forge
from forge import Tensor
from forge.backend.cuda.backend import is_cuda_available

from .sizes import DEFAULT_ITERATIONS, DEFAULT_WARMUP, TRANSFER_SIZES
from .timing import time_cuda

TRIALS = 7
OVERLAP_TRANSFER_ELEMENTS = 2_000_000  # ~8 MB float32 -- large enough for a real, measurable PCIe transfer
OVERLAP_COMPUTE_ELEMENTS = 20_000  # matches stream_bench.py's own chosen "does not fill all 3 SMs" size
OVERLAP_COMPUTE_REPEATS = 400


def _pinned_tensor(values: np.ndarray) -> Tensor:
    mem = forge.cuda.PinnedMemory(values.nbytes)
    array = mem.numpy(shape=values.shape, dtype=values.dtype)
    array[:] = values
    return Tensor(array, device="cpu")


# -- 1. Pinned vs. pageable ------------------------------------------------------


def _run_pinned_vs_pageable() -> None:
    print("=== Pinned (async, synchronized) vs. pageable (synchronous) H2D transfer ===\n")
    for scale, n in TRANSFER_SIZES.items():
        rng = np.random.default_rng(0)
        data = rng.standard_normal(n).astype(np.float32)
        nbytes = n * 4

        pageable_cpu = Tensor(data, device="cpu")
        pinned_cpu = _pinned_tensor(data)

        def _pageable_h2d():
            pageable_cpu.to("cuda")

        def _pinned_h2d_sync_completion():
            pinned_cpu.to("cuda", non_blocking=True)
            forge.cuda.synchronize()

        pageable_timing = time_cuda(_pageable_h2d, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)
        pinned_timing = time_cuda(_pinned_h2d_sync_completion, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)

        pageable_gbps = (nbytes / pageable_timing.mean) / 1e9 if pageable_timing.mean > 0 else float("inf")
        pinned_gbps = (nbytes / pinned_timing.mean) / 1e9 if pinned_timing.mean > 0 else float("inf")

        print(
            f"[{scale:>6}] {nbytes:>10,} bytes  "
            f"pageable={pageable_timing.mean * 1000:.4f} ms ({pageable_gbps:.2f} GB/s)  "
            f"pinned={pinned_timing.mean * 1000:.4f} ms ({pinned_gbps:.2f} GB/s)"
        )
    print()


# -- 2. Async submission latency vs. synchronized completion --------------------


def _run_submission_vs_completion() -> None:
    print("=== Async H2D: submission latency vs. synchronized completion ===\n")
    n = TRANSFER_SIZES["medium"]
    rng = np.random.default_rng(1)
    data = rng.standard_normal(n).astype(np.float32)
    pinned_cpu = _pinned_tensor(data)

    # Warmup (first-call effects: lazy kernel-module init, driver page-locking cache).
    for _ in range(DEFAULT_WARMUP):
        pinned_cpu.to("cuda", non_blocking=True)
    forge.cuda.synchronize()

    submission_durations = []
    completion_durations = []
    for _ in range(DEFAULT_ITERATIONS):
        forge.cuda.synchronize()
        start = time.perf_counter()
        cuda_t = pinned_cpu.to("cuda", non_blocking=True)  # returns once the copy is *submitted*, not complete
        submitted = time.perf_counter()
        forge.cuda.synchronize()  # now wait for it to actually finish
        completed = time.perf_counter()
        submission_durations.append(submitted - start)
        completion_durations.append(completed - start)
        del cuda_t

    submission_median = statistics.median(submission_durations)
    completion_median = statistics.median(completion_durations)
    print(f"submission (host returns once cudaMemcpyAsync is queued): median={submission_median * 1000:.4f} ms")
    print(f"full completion (submission + forge.cuda.synchronize()):  median={completion_median * 1000:.4f} ms")
    print(
        "Submission latency well below completion latency confirms `non_blocking=True` returns "
        "to Python before the device transfer necessarily finishes, rather than secretly "
        "synchronizing internally (Section 31's 'async D2H must not lie', applied here to H2D).\n"
    )


# -- 3. Transfer/compute overlap --------------------------------------------------


def _make_transfer_source() -> Tensor:
    rng = np.random.default_rng(2)
    data = rng.standard_normal(OVERLAP_TRANSFER_ELEMENTS).astype(np.float32)
    return _pinned_tensor(data)


def _make_compute_tensors():
    rng = np.random.default_rng(3)
    a = Tensor(rng.standard_normal(OVERLAP_COMPUTE_ELEMENTS).astype(np.float32), device="cuda")
    b = Tensor(rng.standard_normal(OVERLAP_COMPUTE_ELEMENTS).astype(np.float32), device="cuda")
    return a, b


def _issue_compute(a, b, repeats: int) -> None:
    x = a
    for _ in range(repeats):
        x = x + b


def _run_sequential_transfer_then_compute() -> float:
    transfer_stream = forge.cuda.Stream()
    compute_stream = forge.cuda.Stream()
    source = _make_transfer_source()
    a, b = _make_compute_tensors()

    forge.cuda.synchronize()
    start = time.perf_counter()
    with forge.cuda.stream(transfer_stream):
        source.to("cuda", non_blocking=True)
    transfer_stream.synchronize()
    with forge.cuda.stream(compute_stream):
        _issue_compute(a, b, OVERLAP_COMPUTE_REPEATS)
    compute_stream.synchronize()
    end = time.perf_counter()
    return end - start


def _run_concurrent_transfer_and_compute() -> float:
    transfer_stream = forge.cuda.Stream()
    compute_stream = forge.cuda.Stream()
    source = _make_transfer_source()
    a, b = _make_compute_tensors()

    forge.cuda.synchronize()
    start = time.perf_counter()
    with forge.cuda.stream(transfer_stream):
        source.to("cuda", non_blocking=True)
    with forge.cuda.stream(compute_stream):
        _issue_compute(a, b, OVERLAP_COMPUTE_REPEATS)
    transfer_stream.synchronize()
    compute_stream.synchronize()
    end = time.perf_counter()
    return end - start


def _run_overlap_comparison() -> None:
    print("=== H2D transfer / compute overlap (two streams) ===\n")
    print(f"Transfer: {OVERLAP_TRANSFER_ELEMENTS:,} float32 elements (~{OVERLAP_TRANSFER_ELEMENTS * 4 / 1e6:.1f} MB)")
    print(f"Compute:  {OVERLAP_COMPUTE_REPEATS} chained adds of {OVERLAP_COMPUTE_ELEMENTS:,} float32 elements\n")

    sequential = [_run_sequential_transfer_then_compute() for _ in range(TRIALS)]
    concurrent = [_run_concurrent_transfer_and_compute() for _ in range(TRIALS)]

    seq_median = statistics.median(sequential)
    conc_median = statistics.median(concurrent)
    print(f"sequential (transfer, then compute, synchronized between): median={seq_median * 1000:.2f} ms")
    print(f"concurrent (transfer + compute issued together):           median={conc_median * 1000:.2f} ms")
    if conc_median > 0:
        print(f"overlap speedup: {seq_median / conc_median:.3f}x")
    print(
        "A speedup meaningfully above 1.0x demonstrates real transfer/compute overlap; a ratio "
        "close to 1.0x still confirms the transfer was genuinely submitted asynchronously rather "
        "than silently serialized against the compute stream -- see this module's docstring and "
        "docs/architecture/cuda-transfers.md's Transfer/Compute Overlap section.\n"
    )


# -- 4. D2H transfer/compute overlap ----------------------------------------------


def _make_d2h_source() -> Tensor:
    rng = np.random.default_rng(4)
    return Tensor(rng.standard_normal(OVERLAP_TRANSFER_ELEMENTS).astype(np.float32), device="cuda")


def _run_sequential_d2h_then_compute() -> float:
    transfer_stream = forge.cuda.Stream()
    compute_stream = forge.cuda.Stream()
    source = _make_d2h_source()
    a, b = _make_compute_tensors()

    forge.cuda.synchronize()
    start = time.perf_counter()
    with forge.cuda.stream(transfer_stream):
        source.to("cpu", non_blocking=True)
    transfer_stream.synchronize()
    with forge.cuda.stream(compute_stream):
        _issue_compute(a, b, OVERLAP_COMPUTE_REPEATS)
    compute_stream.synchronize()
    end = time.perf_counter()
    return end - start


def _run_concurrent_d2h_and_compute() -> float:
    transfer_stream = forge.cuda.Stream()
    compute_stream = forge.cuda.Stream()
    source = _make_d2h_source()
    a, b = _make_compute_tensors()

    forge.cuda.synchronize()
    start = time.perf_counter()
    with forge.cuda.stream(transfer_stream):
        result = source.to("cpu", non_blocking=True)
    with forge.cuda.stream(compute_stream):
        _issue_compute(a, b, OVERLAP_COMPUTE_REPEATS)
    compute_stream.synchronize()
    result.numpy()  # the host-read boundary -- synchronizes just this transfer (Tensor._data)
    end = time.perf_counter()
    return end - start


def _run_d2h_overlap_comparison() -> None:
    print("=== D2H transfer / compute overlap (two streams) ===\n")
    sequential = [_run_sequential_d2h_then_compute() for _ in range(TRIALS)]
    concurrent = [_run_concurrent_d2h_and_compute() for _ in range(TRIALS)]

    seq_median = statistics.median(sequential)
    conc_median = statistics.median(concurrent)
    print(f"sequential (D2H, then compute, synchronized between): median={seq_median * 1000:.2f} ms")
    print(f"concurrent (D2H + compute issued together):           median={conc_median * 1000:.2f} ms")
    if conc_median > 0:
        print(f"overlap speedup: {seq_median / conc_median:.3f}x")
    print()


def main() -> None:
    if not is_cuda_available():
        print("CUDA is not available on this machine -- async_transfer_bench requires real CUDA hardware.")
        return

    _run_pinned_vs_pageable()
    _run_submission_vs_completion()
    _run_overlap_comparison()
    _run_d2h_overlap_comparison()


if __name__ == "__main__":
    main()
