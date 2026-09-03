"""Forward Tensor operation benchmarks (Milestone 11; expanded in Milestone 21).

Covers the operation set both backends share (see
`docs/architecture/cuda-backend.md`'s Operation set table): `add`, `sub`,
`mul`, `relu`, `sum`, `reshape`, `matmul`, `exp`, `log`. Each is measured at
the Tensor API level (`a + b`, `a.relu()`, ...) -- the same call site real
model code uses -- rather than reaching into `Backend` methods directly, so
the numbers include whatever argument-checking/dispatch overhead
`Tensor`/`Backend` add on top of the raw kernel.

Milestone 21 adds forward coverage for `Conv2d` (already present since
Milestone 15), `MaxPool2d`, `MSELoss`, `CrossEntropyLoss`, `Dropout`, and an
isolated `Adam` optimizer step -- the operations Section 1 of the M21 brief
lists that M11 did not yet measure.
"""

from __future__ import annotations

import numpy as np

import forge
import forge.nn as nn
import forge.optim as optim
from forge.backend.cuda.backend import is_cuda_available

from .results import BenchmarkResult
from .sizes import (
    ADAM_PARAM_SIZES,
    CONV2D_CONFIGS,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP,
    DROPOUT_P,
    ELEMENTWISE_SIZES,
    LOSS_CONFIGS,
    MATMUL_DIMS,
    POOL2D_KERNEL,
)
from .timing import time_cpu, time_cuda


def _make_vector(n: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal(n).astype(np.float32), device=device)


def _make_matrix(dim: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device=device)


def _make_positive_vector(n: int, device: str, seed: int) -> forge.Tensor:
    """Strictly positive values -- `log()` is undefined (NaN/-inf) at <= 0."""
    rng = np.random.default_rng(seed)
    return forge.Tensor((rng.random(n).astype(np.float32) + 0.1), device=device)


def _run_forward_ops(device: str, results: "list[BenchmarkResult]") -> None:
    timer = time_cpu if device == "cpu" else time_cuda

    for scale, n in ELEMENTWISE_SIZES.items():
        a = _make_vector(n, device, seed=1)
        b = _make_vector(n, device, seed=2)
        c = _make_positive_vector(n, device, seed=6)  # for exp/log: strictly positive input

        ops = {
            "add": lambda a=a, b=b: a + b,
            "sub": lambda a=a, b=b: a - b,
            "mul": lambda a=a, b=b: a * b,
            "relu": lambda a=a: a.relu(),
            "sum": lambda a=a: a.sum(),
            "reshape": lambda a=a, n=n: a.reshape(n),
            "exp": lambda c=c: c.exp(),
            "log": lambda c=c: c.log(),
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


def _run_maxpool2d_forward(device: str, results: "list[BenchmarkResult]") -> None:
    """MaxPool2d(kernel=stride=2) forward, on the Conv2d output shapes it typically follows."""
    timer = time_cpu if device == "cpu" else time_cuda
    rng = np.random.default_rng(7)
    for scale, cfg in CONV2D_CONFIGS.items():
        h_out, w_out = cfg["H"] - cfg["K"] + 1, cfg["W"] - cfg["K"] + 1  # matches Conv2d(stride=1,pad=0)'s output
        x = forge.Tensor(
            rng.standard_normal((cfg["N"], cfg["Cout"], h_out, w_out)).astype(np.float32), device=device
        )
        pool = nn.MaxPool2d(POOL2D_KERNEL)
        timing = timer(lambda x=x, pool=pool: pool(x), warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)
        results.append(
            BenchmarkResult.from_timing(
                category="forward", operation="max_pool2d", device=device, scale=scale,
                shape=f"N={cfg['N']},C={cfg['Cout']},HxW={h_out}x{w_out},k={POOL2D_KERNEL}",
                dtype="float32", timing=timing,
            )
        )


def _run_loss_forward(device: str, results: "list[BenchmarkResult]") -> None:
    timer = time_cpu if device == "cpu" else time_cuda
    for scale, cfg in LOSS_CONFIGS.items():
        rng = np.random.default_rng(8)
        batch, classes = cfg["batch"], cfg["classes"]
        logits = forge.Tensor(rng.standard_normal((batch, classes)).astype(np.float32), device=device)
        pred = forge.Tensor(rng.standard_normal((batch, classes)).astype(np.float32), device=device)
        target_reg = forge.Tensor(rng.standard_normal((batch, classes)).astype(np.float32), device=device)
        target_cls = rng.integers(0, classes, size=(batch,)).astype(np.int64)

        mse = nn.MSELoss()
        timing = timer(
            lambda pred=pred, target_reg=target_reg: mse(pred, target_reg),
            warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS,
        )
        results.append(
            BenchmarkResult.from_timing(
                category="forward", operation="mse_loss", device=device, scale=scale,
                shape=f"batch={batch},classes={classes}", dtype="float32", timing=timing,
            )
        )

        ce = nn.CrossEntropyLoss()
        timing = timer(
            lambda logits=logits, target_cls=target_cls: ce(logits, target_cls),
            warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS,
        )
        results.append(
            BenchmarkResult.from_timing(
                category="forward", operation="cross_entropy_loss", device=device, scale=scale,
                shape=f"batch={batch},classes={classes}", dtype="float32", timing=timing,
            )
        )


def _run_dropout_forward(device: str, results: "list[BenchmarkResult]") -> None:
    timer = time_cpu if device == "cpu" else time_cuda
    for scale, n in ELEMENTWISE_SIZES.items():
        a = _make_vector(n, device, seed=9)
        dropout = nn.Dropout(DROPOUT_P)
        dropout.train()
        timing = timer(lambda a=a, dropout=dropout: dropout(a), warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)
        results.append(
            BenchmarkResult.from_timing(
                category="forward", operation="dropout", device=device, scale=scale,
                shape=f"({n},)", dtype="float32", timing=timing,
            )
        )


def _run_adam_step_forward(device: str, results: "list[BenchmarkResult]") -> None:
    """One isolated `Adam.step()` call against a single parameter with a fixed gradient.

    Not a training loop -- just the optimizer-step primitive itself, matching
    Section 1's "Adam step" forward-benchmark requirement independent of
    `training_bench.py`'s end-to-end throughput measurement.
    """
    timer = time_cpu if device == "cpu" else time_cuda
    for scale, n in ADAM_PARAM_SIZES.items():
        rng = np.random.default_rng(10)
        param = nn.Parameter(rng.standard_normal(n).astype(np.float32), device=device)
        param.grad = forge.Tensor(rng.standard_normal(n).astype(np.float32), device=device)
        adam = optim.Adam([param], lr=1e-3)

        timing = timer(lambda adam=adam: adam.step(), warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS)
        results.append(
            BenchmarkResult.from_timing(
                category="forward", operation="adam_step", device=device, scale=scale,
                shape=f"({n},)", dtype="float32", timing=timing,
            )
        )


def run_forward_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    _run_forward_ops("cpu", results)
    _run_conv2d_forward("cpu", results)
    _run_maxpool2d_forward("cpu", results)
    _run_loss_forward("cpu", results)
    _run_dropout_forward("cpu", results)
    _run_adam_step_forward("cpu", results)
    if is_cuda_available():
        _run_forward_ops("cuda", results)
        _run_conv2d_forward("cuda", results)
        _run_maxpool2d_forward("cuda", results)
        _run_loss_forward("cuda", results)
        _run_dropout_forward("cuda", results)
        _run_adam_step_forward("cuda", results)
    return results
