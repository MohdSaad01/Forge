"""Forward Tensor operation benchmarks (Milestone 11).

Covers exactly the operation set both backends share (see
`docs/architecture/cuda-backend.md`'s Operation set table): `add`, `sub`,
`mul`, `relu`, `sum`, `reshape`, `matmul`. Each is measured at the Tensor
API level (`a + b`, `a.relu()`, ...) -- the same call site real model code
uses -- rather than reaching into `Backend` methods directly, so the
numbers include whatever argument-checking/dispatch overhead
`Tensor`/`Backend` add on top of the raw kernel.

`exp`/`log` gained real CUDA kernels in Milestone 14 (for `CrossEntropyLoss`)
but remain outside this suite's fixed operation set -- benchmarking/CLI
integration for them is out of that milestone's scope, matching its "no
performance autotuning" constraint; this file's coverage is unchanged.
"""

from __future__ import annotations

import numpy as np

import forge
from forge.backend.cuda.backend import is_cuda_available

from .results import BenchmarkResult
from .sizes import CONV2D_CONFIGS, DEFAULT_ITERATIONS, DEFAULT_WARMUP, ELEMENTWISE_SIZES, MATMUL_DIMS
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


def _run_conv2d_forward(device: str, results: "list[BenchmarkResult]") -> None:
    """Conv2d(stride=1, padding=0) forward, at the three `CONV2D_CONFIGS` scales.

    Milestone 15: correctness, not performance, is the objective here -- see
    that module's docstring. This is a baseline measurement, not a claim
    that the 940MX's straightforward (unoptimized) Conv2d kernel beats
    NumPy/BLAS at any of these scales.
    """
    timer = time_cpu if device == "cpu" else time_cuda
    rng = np.random.default_rng(5)
    for scale, cfg in CONV2D_CONFIGS.items():
        x = forge.Tensor(
            rng.standard_normal((cfg["N"], cfg["Cin"], cfg["H"], cfg["W"])).astype(np.float32), device=device
        )
        w = forge.Tensor(
            rng.standard_normal((cfg["Cout"], cfg["Cin"], cfg["K"], cfg["K"])).astype(np.float32), device=device
        )
        b = forge.Tensor(rng.standard_normal((cfg["Cout"],)).astype(np.float32), device=device)
        timing = timer(
            lambda x=x, w=w, b=b: x.conv2d(w, b, (1, 1), (0, 0)),
            warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS,
        )
        results.append(
            BenchmarkResult.from_timing(
                category="forward", operation="conv2d", device=device, scale=scale,
                shape=f"N={cfg['N']},Cin={cfg['Cin']},Cout={cfg['Cout']},HxW={cfg['H']}x{cfg['W']},k={cfg['K']}",
                dtype="float32", timing=timing,
            )
        )


def run_forward_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    _run_forward_ops("cpu", results)
    _run_conv2d_forward("cpu", results)
    if is_cuda_available():
        _run_forward_ops("cuda", results)
        _run_conv2d_forward("cuda", results)
    return results
