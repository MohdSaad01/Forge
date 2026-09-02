"""End-to-end training-throughput benchmarks (Milestone 11).

A small `Linear -> ReLU -> Linear` model, trained for a fixed number of
iterations against one fixed synthetic batch. This never asserts or depends
on convergence (per the milestone brief) -- only wall-clock throughput of
the standard `zero_grad -> forward -> loss -> backward -> step` sequence
(`docs/architecture/training-engine.md`, `docs/architecture/optimization.md`).

`forge.training.Trainer` has no CUDA integration
(`docs/architecture/cuda-backend.md`'s "No CUDA Trainer/DataLoader
integration" limitation), so both the CPU and CUDA cases use the same
hand-written loop -- matching how `tests/test_cuda_autograd.py`'s own
training-loop test exercises CUDA. This keeps the CPU and CUDA measurements
comparable (identical code path, only the device differs) rather than
comparing `Trainer` overhead against a hand-written loop.
"""

from __future__ import annotations

import time

import numpy as np

import forge
import forge.nn as nn
import forge.optim as optim
from forge.backend.cuda.backend import is_cuda_available

from .results import BenchmarkResult
from .sizes import TRAINING_CONFIG
from .timing import Timing


class _MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _sync(device: str) -> None:
    if device == "cuda":
        from forge.backend.cuda.backend import get_cuda_backend

        get_cuda_backend().synchronize()


def _run_training(device: str, results: "list[BenchmarkResult]") -> None:
    cfg = TRAINING_CONFIG
    forge.random.seed(0)
    model = _MLP(cfg["in_features"], cfg["hidden_features"], cfg["out_features"])
    if device == "cuda":
        model.to("cuda")
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    rng = np.random.default_rng(0)
    x = forge.Tensor(
        rng.standard_normal((cfg["batch_size"], cfg["in_features"])).astype(np.float32), device=device
    )
    y = forge.Tensor(
        rng.standard_normal((cfg["batch_size"], cfg["out_features"])).astype(np.float32), device=device
    )

    def step() -> None:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    for _ in range(cfg["warmup_iterations"]):
        step()
    _sync(device)

    total_durations, forward_durations, backward_durations, optim_durations = [], [], [], []
    for _ in range(cfg["iterations"]):
        _sync(device)
        t0 = time.perf_counter()

        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        _sync(device)
        t1 = time.perf_counter()

        loss.backward()
        _sync(device)
        t2 = time.perf_counter()

        optimizer.step()
        _sync(device)
        t3 = time.perf_counter()

        forward_durations.append(t1 - t0)
        backward_durations.append(t2 - t1)
        optim_durations.append(t3 - t2)
        total_durations.append(t3 - t0)

    batch_size = cfg["batch_size"]
    total_time = sum(total_durations)
    batches_per_sec = cfg["iterations"] / total_time if total_time > 0 else float("inf")
    samples_per_sec = batches_per_sec * batch_size

    timing = Timing(tuple(total_durations), cfg["warmup_iterations"], cfg["iterations"])
    results.append(
        BenchmarkResult.from_timing(
            category="training",
            operation="zero_grad_forward_loss_backward_step",
            device=device,
            scale="fixed",
            shape=(
                f"batch={batch_size}, in={cfg['in_features']}, "
                f"hidden={cfg['hidden_features']}, out={cfg['out_features']}"
            ),
            dtype="float32",
            timing=timing,
            extra={
                "batches_per_sec": batches_per_sec,
                "samples_per_sec": samples_per_sec,
                "mean_forward_seconds": sum(forward_durations) / len(forward_durations),
                "mean_backward_seconds": sum(backward_durations) / len(backward_durations),
                "mean_optimizer_seconds": sum(optim_durations) / len(optim_durations),
            },
        )
    )


def run_training_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    _run_training("cpu", results)
    if is_cuda_available():
        _run_training("cuda", results)
    return results
