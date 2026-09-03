"""End-to-end MNIST training-throughput benchmark (Milestone 21, Section 14).

The real M20 CNN (`examples.mnist.model.build_model()`) trained against a
fixed synthetic `(N, 1, 28, 28)` batch (the same stand-in shape
`tests/test_mnist_example_integration.py` uses -- no ~11MB dataset download
needed here), through the real `CrossEntropyLoss -> Adam` combination M20
uses. Reports samples/sec and epoch time (extrapolated from measured
per-batch throughput over one epoch's worth of batches) for both CPU and
CUDA on this exact fixed configuration, directly comparable to
`examples/mnist/train.py`'s own reported throughput.

Like `training_bench.py`, this never asserts or depends on convergence --
only wall-clock throughput of the standard
`zero_grad -> forward -> loss -> backward -> step` sequence.
"""

from __future__ import annotations

import gc
import time

import numpy as np

import forge
import forge.nn as nn
import forge.optim as optim
from forge.backend.cuda.backend import is_cuda_available

from .memory import cuda_memory_extra
from .results import BenchmarkResult
from .sizes import MNIST_TRAINING_CONFIG
from .timing import Timing

# A representative single-epoch sample count (the real MNIST training set is
# 60,000) -- used only to extrapolate "epoch time" from a measured per-batch
# rate, not to actually construct that many samples.
_EPOCH_SAMPLE_COUNT = 60_000


def _sync(device: str) -> None:
    if device == "cuda":
        forge.cuda.synchronize()


def _run_mnist_training(device: str, results: "list[BenchmarkResult]") -> None:
    from examples.mnist.model import build_model

    cfg = MNIST_TRAINING_CONFIG
    batch_size = cfg["batch_size"]

    forge.random.seed(0)
    model = build_model()
    if device == "cuda":
        model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    rng = np.random.default_rng(0)
    x_data = rng.standard_normal((batch_size, 1, 28, 28)).astype(np.float32)
    y_data = rng.integers(0, 10, size=(batch_size,)).astype(np.int64)

    def step() -> None:
        # Fresh CPU->device transfer every step, matching how a real
        # DataLoader batch crosses the device boundary in `Trainer`/
        # `examples/mnist/train.py` -- transfer cost is included here, not
        # hidden by reusing one already-resident batch tensor.
        x = forge.Tensor(x_data, device=device)
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y_data)
        loss.backward()
        optimizer.step()

    for _ in range(cfg["warmup_iterations"]):
        step()
    _sync(device)

    if device == "cuda":
        # See `training_bench.py`'s matching comment: Forge's Tensor/autograd/
        # Module object graph for a full training step contains reference
        # cycles (`docs/architecture/cuda-backend.md`'s **CUDA Memory
        # Statistics** "Known limitations"), so an explicit `gc.collect()`
        # before each snapshot keeps the reported delta a property of the
        # workload rather than of the cyclic collector's own schedule.
        gc.collect()
        forge.cuda.reset_peak_memory_stats()
        mem_before = forge.cuda.memory_stats()

    durations = []
    for _ in range(cfg["iterations"]):
        _sync(device)
        t0 = time.perf_counter()
        step()
        _sync(device)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    total_time = sum(durations)
    batches_per_sec = cfg["iterations"] / total_time if total_time > 0 else float("inf")
    samples_per_sec = batches_per_sec * batch_size
    epoch_seconds = _EPOCH_SAMPLE_COUNT / samples_per_sec if samples_per_sec > 0 else float("inf")

    extra = {
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "extrapolated_epoch_seconds": epoch_seconds,
        "epoch_sample_count_used_for_extrapolation": _EPOCH_SAMPLE_COUNT,
    }
    if device == "cuda":
        gc.collect()  # see the matching comment above the "before" snapshot
        extra.update(cuda_memory_extra(mem_before, forge.cuda.memory_stats()))

    timing = Timing(tuple(durations), cfg["warmup_iterations"], cfg["iterations"])
    results.append(
        BenchmarkResult.from_timing(
            category="training",
            operation="mnist_cnn_zero_grad_forward_loss_backward_step",
            device=device,
            scale="fixed",
            shape=f"batch={batch_size} (M20 CNN: Conv2d/ReLU/MaxPool2d x2 -> Flatten -> Linear/ReLU/Linear)",
            dtype="float32",
            timing=timing,
            extra=extra,
        )
    )


def run_mnist_training_benchmarks() -> "list[BenchmarkResult]":
    results: "list[BenchmarkResult]" = []
    _run_mnist_training("cpu", results)
    if is_cuda_available():
        _run_mnist_training("cuda", results)
    return results
