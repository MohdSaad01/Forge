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
from .sizes import DEFAULT_ITERATIONS, DEFAULT_WARMUP, ELEMENTWISE_SIZES, MATMUL_DIMS
from .timing import time_calls


def _ones_grad(shape, device: str) -> forge.Tensor:
    return forge.Tensor(np.ones(shape, dtype=np.float32), device=device)


def _leaf_vector(n: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal(n).astype(np.float32), device=device, requires_grad=True)


def _leaf_matrix(dim: int, device: str, seed: int) -> forge.Tensor:
    rng = np.random.default_rng(seed)
    return forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device=device, requires_grad=True)


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


def run_backward_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    for device in (["cpu", "cuda"] if is_cuda_available() else ["cpu"]):
        _bench_elementwise_backward(device, results)
        _bench_matmul_backward(device, results)
        _bench_multilayer_backward(device, results)
    return results
