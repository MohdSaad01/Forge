"""CPU<->CUDA transfer-cost benchmarks (Milestone 11).

Measures `Tensor.to()` in each direction at three representative sizes.
Transfer time is measured as its own category -- never folded into a kernel
timing elsewhere in this suite -- per the milestone brief's "Do not hide
transfer time inside kernel execution time." Throughput is reported in
GB/s (bytes moved / mean seconds) since that is the natural unit for
comparing against PCIe bandwidth expectations.

Skipped entirely (returns no results) when CUDA is unavailable -- there is
nothing to transfer to/from without a CUDA backend.
"""

from __future__ import annotations

import numpy as np

import forge
from forge.backend.cuda.backend import is_cuda_available

from .results import BenchmarkResult
from .sizes import DEFAULT_ITERATIONS, DEFAULT_WARMUP, TRANSFER_SIZES
from .timing import time_cuda


def run_transfer_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    if not is_cuda_available():
        return results

    for scale, n in TRANSFER_SIZES.items():
        rng = np.random.default_rng(0)
        data = rng.standard_normal(n).astype(np.float32)
        cpu_tensor = forge.Tensor(data, device="cpu")
        cuda_tensor = cpu_tensor.to("cuda")
        nbytes = n * 4  # float32

        h2d_timing = time_cuda(lambda: cpu_tensor.to("cuda"), warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)
        d2h_timing = time_cuda(lambda: cuda_tensor.to("cpu"), warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)

        for direction, timing in (("h2d", h2d_timing), ("d2h", d2h_timing)):
            throughput_gbps = (nbytes / timing.mean) / 1e9 if timing.mean > 0 else float("inf")
            results.append(
                BenchmarkResult.from_timing(
                    category="transfer",
                    operation=direction,
                    device="cuda",
                    scale=scale,
                    shape=f"({n},)",
                    dtype="float32",
                    timing=timing,
                    extra={"bytes": nbytes, "throughput_GBps": throughput_gbps},
                )
            )

    return results
