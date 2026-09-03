"""Backward-pass benchmarks (Milestone 11).

Covers: elementwise backward (`add`), `relu` backward, `matmul` backward,
`sum` backward, and a representative multi-layer backward pass
(`Linear -> ReLU -> Linear`).

A Tensor's non-leaf `grad_fn` is freed the moment `backward()` consumes it
(`docs/architecture/autograd.md`'s "Graph freed on use" -- a second
`backward()` call on the same non-leaf output raises `GradientStateError`),
so timing "backward" by simply repeating one `y.backward()` call
`iterations` times, the way the forward benchmarks repeat `a + b`, is not
possible. Instead every measured iteration gets its own freshly built
forward pass (constructed *before* timing starts, so forward-pass cost is
never folded into the backward measurement), and `timing.time_calls` times
each iteration's single `backward()` call individually. See
`benchmarks/timing.py`'s `time_calls` docstring for the same reasoning.
"""

from __future__ import annotations

import numpy as np

import forge
import forge.nn as nn
from forge.backend.cuda.backend import is_cuda_available

from .results import BenchmarkResult
from .sizes import (
    CONV2D_CONFIGS,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP,
    ELEMENTWISE_SIZES,
    LOSS_CONFIGS,
    MATMUL_DIMS,
    POOL2D_KERNEL,
)
from .timing import time_calls


def _ones_grad(shape, device: str) -> forge.Tensor:
    return forge.Tensor(np.ones(shape, dtype=np.float32), device=device)


def _leaf_vector(n: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal(n).astype(np.float32), device=device, requires_grad=True)


def _leaf_matrix(dim: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device=device, requires_grad=True)


def _leaf_positive_vector(n: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor((rng.random(n).astype(np.float32) + 0.1), device=device, requires_grad=True)


def _build_calls(build_one, count: int) -> "list":
    """Pre-build `count` fresh (forward pass, backward closure) pairs, untimed."""
    return [build_one() for _ in range(count)]


def _bench_elementwise_backward(device: str, results: "list[BenchmarkResult]") -> None:
    count = DEFAULT_WARMUP + DEFAULT_ITERATIONS
    for scale, n in ELEMENTWISE_SIZES.items():
        grad = _ones_grad((n,), device)

        def make_add_call():
            a = _leaf_vector(n, device, seed=1)
            b = _leaf_vector(n, device, seed=2)
            y = a + b
            return lambda y=y: y.backward(grad)

        calls = _build_calls(make_add_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="add", device=device, scale=scale,
                shape=f"({n},)", dtype="float32", timing=timing,
            )
        )

        def make_relu_call():
            a = _leaf_vector(n, device, seed=1)
            y = a.relu()
            return lambda y=y: y.backward(grad)

        calls = _build_calls(make_relu_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="relu", device=device, scale=scale,
                shape=f"({n},)", dtype="float32", timing=timing,
            )
        )

        def make_sum_call():
            a = _leaf_vector(n, device, seed=1)
            y = a.sum()
            return lambda y=y: y.backward()

        calls = _build_calls(make_sum_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="sum", device=device, scale=scale,
                shape=f"({n},)", dtype="float32", timing=timing,
            )
        )

        def make_exp_call():
            a = _leaf_positive_vector(n, device, seed=6)
            y = a.exp()
            return lambda y=y: y.backward(grad)

        calls = _build_calls(make_exp_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="exp", device=device, scale=scale,
                shape=f"({n},)", dtype="float32", timing=timing,
            )
        )

        def make_log_call():
            a = _leaf_positive_vector(n, device, seed=6)
            y = a.log()
            return lambda y=y: y.backward(grad)

        calls = _build_calls(make_log_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="log", device=device, scale=scale,
                shape=f"({n},)", dtype="float32", timing=timing,
            )
        )


def _bench_matmul_backward(device: str, results: "list[BenchmarkResult]") -> None:
    count = DEFAULT_WARMUP + DEFAULT_ITERATIONS
    for scale, dim in MATMUL_DIMS.items():
        grad = _ones_grad((dim, dim), device)

        def make_call():
            a = _leaf_matrix(dim, device, seed=3)
            b = _leaf_matrix(dim, device, seed=4)
            y = a @ b
            return lambda y=y: y.backward(grad)

        calls = _build_calls(make_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="matmul", device=device, scale=scale,
                shape=f"({dim},{dim})@({dim},{dim})", dtype="float32", timing=timing,
            )
        )


class _MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _bench_multilayer_backward(device: str, results: "list[BenchmarkResult]") -> None:
    from .sizes import TRAINING_CONFIG

    cfg = TRAINING_CONFIG
    count = DEFAULT_WARMUP + DEFAULT_ITERATIONS

    forge.random.seed(0)
    model = _MLP(cfg["in_features"], cfg["hidden_features"], cfg["out_features"])
    if device == "cuda":
        model.to("cuda")

    rng = np.random.default_rng(0)
    x = forge.Tensor(
        rng.standard_normal((cfg["batch_size"], cfg["in_features"])).astype(np.float32), device=device
    )
    grad = _ones_grad((cfg["batch_size"], cfg["out_features"]), device)

    def make_call():
        y = model(x)
        return lambda y=y: y.backward(grad)

    calls = _build_calls(make_call, count)
    timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
    results.append(
        BenchmarkResult.from_timing(
            category="backward", operation="multilayer_linear_relu_linear", device=device, scale="fixed",
            shape=f"batch={cfg['batch_size']}", dtype="float32", timing=timing,
        )
    )


def _bench_conv2d_backward(device: str, results: "list[BenchmarkResult]") -> None:
    """Conv2d(stride=1, padding=0) backward, at the three `CONV2D_CONFIGS` scales.

    A baseline measurement, not a performance target -- see
    `ops_bench._run_conv2d_forward`'s docstring and Milestone 15's
    "do not optimize before correctness is established" constraint.
    """
    count = DEFAULT_WARMUP + DEFAULT_ITERATIONS
    for scale, cfg in CONV2D_CONFIGS.items():
        rng = np.random.default_rng(5)
        x_data = rng.standard_normal((cfg["N"], cfg["Cin"], cfg["H"], cfg["W"])).astype(np.float32)
        w_data = rng.standard_normal((cfg["Cout"], cfg["Cin"], cfg["K"], cfg["K"])).astype(np.float32)
        b_data = rng.standard_normal((cfg["Cout"],)).astype(np.float32)
        H_out, W_out = cfg["H"] - cfg["K"] + 1, cfg["W"] - cfg["K"] + 1
        grad = _ones_grad((cfg["N"], cfg["Cout"], H_out, W_out), device)

        def make_call(x_data=x_data, w_data=w_data, b_data=b_data):
            x = forge.Tensor(x_data.copy(), device=device, requires_grad=True)
            w = forge.Tensor(w_data.copy(), device=device, requires_grad=True)
            b = forge.Tensor(b_data.copy(), device=device, requires_grad=True)
            y = x.conv2d(w, b, (1, 1), (0, 0))
            return lambda y=y: y.backward(grad)

        calls = _build_calls(make_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="conv2d", device=device, scale=scale,
                shape=f"N={cfg['N']},Cin={cfg['Cin']},Cout={cfg['Cout']},HxW={cfg['H']}x{cfg['W']},k={cfg['K']}",
                dtype="float32", timing=timing,
            )
        )


def _bench_maxpool2d_backward(device: str, results: "list[BenchmarkResult]") -> None:
    count = DEFAULT_WARMUP + DEFAULT_ITERATIONS
    for scale, cfg in CONV2D_CONFIGS.items():
        h_out, w_out = cfg["H"] - cfg["K"] + 1, cfg["W"] - cfg["K"] + 1
        rng = np.random.default_rng(7)
        x_data = rng.standard_normal((cfg["N"], cfg["Cout"], h_out, w_out)).astype(np.float32)
        pool_h, pool_w = h_out // POOL2D_KERNEL, w_out // POOL2D_KERNEL
        grad = _ones_grad((cfg["N"], cfg["Cout"], pool_h, pool_w), device)

        def make_call(x_data=x_data):
            x = forge.Tensor(x_data.copy(), device=device, requires_grad=True)
            pool = nn.MaxPool2d(POOL2D_KERNEL)
            y = pool(x)
            return lambda y=y: y.backward(grad)

        calls = _build_calls(make_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="max_pool2d", device=device, scale=scale,
                shape=f"N={cfg['N']},C={cfg['Cout']},HxW={h_out}x{w_out},k={POOL2D_KERNEL}",
                dtype="float32", timing=timing,
            )
        )


def _bench_cross_entropy_backward(device: str, results: "list[BenchmarkResult]") -> None:
    count = DEFAULT_WARMUP + DEFAULT_ITERATIONS
    for scale, cfg in LOSS_CONFIGS.items():
        batch, classes = cfg["batch"], cfg["classes"]
        rng = np.random.default_rng(8)
        logits_data = rng.standard_normal((batch, classes)).astype(np.float32)
        target = rng.integers(0, classes, size=(batch,)).astype(np.int64)
        ce = nn.CrossEntropyLoss()

        def make_call(logits_data=logits_data, target=target):
            logits = forge.Tensor(logits_data.copy(), device=device, requires_grad=True)
            y = ce(logits, target)
            return lambda y=y: y.backward()

        calls = _build_calls(make_call, count)
        timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
        results.append(
            BenchmarkResult.from_timing(
                category="backward", operation="cross_entropy_loss", device=device, scale=scale,
                shape=f"batch={batch},classes={classes}", dtype="float32", timing=timing,
            )
        )


def _bench_full_cnn_backward(device: str, results: "list[BenchmarkResult]") -> None:
    """A complete small-CNN backward pass -- the real M20 MNIST architecture.

    `examples.mnist.model.build_model()` -> `CrossEntropyLoss` -> `.backward()`,
    one fixed synthetic `(N, 1, 28, 28)` batch, matching Section 1's
    "complete small CNN" backward-workload requirement.
    """
    from examples.mnist.model import build_model

    count = DEFAULT_WARMUP + DEFAULT_ITERATIONS
    batch = 64
    rng = np.random.default_rng(11)
    x_data = rng.standard_normal((batch, 1, 28, 28)).astype(np.float32)
    target = rng.integers(0, 10, size=(batch,)).astype(np.int64)
    loss_fn = nn.CrossEntropyLoss()

    forge.random.seed(0)
    model = build_model()
    if device == "cuda":
        model.to("cuda")

    def make_call():
        x = forge.Tensor(x_data.copy(), device=device)
        y = loss_fn(model(x), target)
        return lambda y=y: y.backward()

    calls = _build_calls(make_call, count)
    timing = time_calls(calls, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS, device=device)
    results.append(
        BenchmarkResult.from_timing(
            category="backward", operation="mnist_cnn_full", device=device, scale="fixed",
            shape=f"batch={batch} (Conv2d/ReLU/MaxPool2d x2 -> Flatten -> Linear/ReLU/Linear)",
            dtype="float32", timing=timing,
        )
    )


def run_backward_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    for device in (["cpu", "cuda"] if is_cuda_available() else ["cpu"]):
        _bench_elementwise_backward(device, results)
        _bench_matmul_backward(device, results)
        _bench_multilayer_backward(device, results)
        _bench_conv2d_backward(device, results)
        _bench_maxpool2d_backward(device, results)
        _bench_cross_entropy_backward(device, results)
        _bench_full_cnn_backward(device, results)
    return results
