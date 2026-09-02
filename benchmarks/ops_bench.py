"""Forward Tensor operation benchmarks (Milestone 11).

Covers exactly the operation set both backends share (see
`docs/architecture/cuda-backend.md`'s Operation set table): `add`, `sub`,
`mul`, `relu`, `sum`, `reshape`, `matmul`. Each is measured at the Tensor
API level (`a + b`, `a.relu()`, ...) -- the same call site real model code
uses -- rather than reaching into `Backend` methods directly, so the
numbers include whatever argument-checking/dispatch overhead
`Tensor`/`Backend` add on top of the raw kernel.

CUDA-unsupported CPU-only ops (`exp`/`log`) are intentionally not
benchmarked on CUDA, matching the milestone brief's "do not benchmark
unsupported CUDA operations."
"""

from __future__ import annotations

import numpy as np

import forge
from forge.backend.cuda.backend import is_cuda_available

from .results import BenchmarkResult
from .sizes import DEFAULT_ITERATIONS, DEFAULT_WARMUP, ELEMENTWISE_SIZES, MATMUL_DIMS
from .timing import time_cpu, time_cuda


def _make_vector(n: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal(n).astype(np.float32), device=device)


def _make_matrix(dim: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device=device)


def _run_forward_ops(device: str, results: "list[BenchmarkResult]") -> None:
    timer = time_cpu if device == "cpu" else time_cuda

    for scale, n in ELEMENTWISE_SIZES.items():
        a = _make_vector(n, device, seed=1)
        b = _make_vector(n, device, seed=2)

        ops = {
            "add": lambda a=a, b=b: a + b,
            "sub": lambda a=a, b=b: a - b,
            "mul": lambda a=a, b=b: a * b,
            "relu": lambda a=a: a.relu(),
            "sum": lambda a=a: a.sum(),
            "reshape": lambda a=a, n=n: a.reshape(n),
        }
        for op_name, fn in ops.items():
            timing = timer(fn, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)
            results.append(
                BenchmarkResult.from_timing(
                    category="forward",
                    operation=op_name,
                    device=device,
                    scale=scale,
                    shape=f"({n},)",
                    dtype="float32",
                    timing=timing,
                )
            )

    for scale, dim in MATMUL_DIMS.items():
        a = _make_matrix(dim, device, seed=3)
        b = _make_matrix(dim, device, seed=4)
        timing = timer(lambda a=a, b=b: a @ b, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)
        results.append(
            BenchmarkResult.from_timing(
                category="forward",
                operation="matmul",
                device=device,
                scale=scale,
                shape=f"({dim},{dim})@({dim},{dim})",
                dtype="float32",
                timing=timing,
            )
        )


def run_forward_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    _run_forward_ops("cpu", results)
    if is_cuda_available():
        _run_forward_ops("cuda", results)
    return results
