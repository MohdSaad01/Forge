"""Asynchronous DataLoader GPU-prefetch benchmark (Milestone 30).

    python -m benchmarks.async_dataloader_bench

Standalone script, not part of `python -m benchmarks`'s category list --
matching `async_transfer_bench.py`/`stream_bench.py`'s own "diagnostic
script, not a stable-schema benchmark category" status.

Two things are measured, each comparing a synchronous `Trainer` (`prefetch=
False`, unchanged since Milestone 12) against the same `Trainer` configured
with `prefetch=True` (Milestone 30):

1. **Synthetic workload** (Section 58): a `Dataset` whose `__getitem__` does
   a controllable amount of pure-Python/NumPy work (`cpu_work_iters`) feeding
   a model with a controllable amount of GPU compute (`gpu_work_layers`), at
   a controllable batch/feature size -- independently tunable, so the
   benchmark can distinguish "no overlap opportunity" (GPU work dominates,
   or CPU work is negligible) from "broken pipeline" (Section 60): a
   configuration with genuinely overlappable CPU-prep and GPU-compute time
   should show a real speedup; one without real overlap opportunity should
   not, and that is not itself a bug.
2. **Real MNIST workload** (Section 59): the actual M20 CNN
   (`examples.mnist.model.build_model()`) trained against a synthetic
   MNIST-shaped batch (no dataset download needed, matching
   `benchmarks/mnist_bench.py`'s own convention), through the real
   `CrossEntropyLoss -> Adam` combination `examples/mnist/train.py` uses.

Both report: total epoch wall-clock time, samples/sec, and (for the
prefetch run) CUDA/pinned-memory overhead added by the prefetch pipeline
(Section 47/48) -- measured via `forge.cuda.memory_stats()`/`forge.cuda.
pinned_memory_stats()` deltas, following `benchmarks/memory.py`'s
methodology.
"""

from __future__ import annotations

import gc
import statistics
import time

import numpy as np

import forge
from forge import Tensor
from forge.backend.cuda.backend import is_cuda_available
from forge.data import DataLoader, Dataset, TensorDataset
from forge.nn import CrossEntropyLoss, Linear, MSELoss, ReLU, Sequential
from forge.optim import SGD, Adam
from forge.training import Trainer

TRIALS = 5


# -- 1. Synthetic workload: independently controllable CPU/GPU/transfer cost --


class _SyntheticDataset(Dataset):
    """A dataset whose `__getitem__` does `cpu_work_iters` small NumPy matmuls before returning a sample.

    Simulates real per-sample CPU preprocessing cost (a decode, an
    augmentation, a normalization pass) without depending on any actual
    image/audio pipeline -- purely to make "CPU batch-preparation duration"
    an independently controllable benchmark parameter (Section 58).
    """

    def __init__(self, n: int, feature_dim: int, cpu_work_iters: int, seed: int = 0):
        self.n = n
        self.feature_dim = feature_dim
        self.cpu_work_iters = cpu_work_iters
        rng = np.random.default_rng(seed)
        self._x = rng.standard_normal((n, feature_dim)).astype(np.float32)
        self._y = rng.standard_normal((n, 1)).astype(np.float32)
        self._work_a = rng.standard_normal((32, 32)).astype(np.float32)
        self._work_b = rng.standard_normal((32, 32)).astype(np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        acc = self._work_a
        for _ in range(self.cpu_work_iters):
            acc = (acc @ self._work_b) * 1e-3  # rescale to avoid overflow across iterations
        # `acc` is deliberately unused beyond forcing the work above to run --
        # the sample itself is a plain, fixed-cost slice.
        return Tensor(self._x[index]), Tensor(self._y[index])


def _make_synthetic_model(in_features: int, gpu_work_layers: int) -> Sequential:
    layers = [Linear(in_features, 64), ReLU()]
    for _ in range(gpu_work_layers):
        layers += [Linear(64, 64), ReLU()]
    layers.append(Linear(64, 1))
    model = Sequential(*layers)
    model.to("cuda")
    return model


def _time_one_epoch(trainer: Trainer, loader) -> float:
    forge.cuda.synchronize()
    start = time.perf_counter()
    trainer.fit(loader, epochs=1)
    forge.cuda.synchronize()
    return time.perf_counter() - start


def _run_synthetic_case(name: str, n: int, batch_size: int, feature_dim: int, cpu_work_iters: int, gpu_work_layers: int) -> None:
    ds = _SyntheticDataset(n, feature_dim, cpu_work_iters)

    def _sync_durations():
        durations = []
        for trial in range(TRIALS):
            forge.random.seed(trial)
            model = _make_synthetic_model(feature_dim, gpu_work_layers)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=np.random.default_rng(trial))
            trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False)
            durations.append(_time_one_epoch(trainer, loader))
        return durations

    def _prefetch_durations():
        durations = []
        for trial in range(TRIALS):
            forge.random.seed(trial)
            model = _make_synthetic_model(feature_dim, gpu_work_layers)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=np.random.default_rng(trial))
            trainer = Trainer(
                model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False, prefetch=True
            )
            durations.append(_time_one_epoch(trainer, loader))
        return durations

    sync_durations = _sync_durations()
    prefetch_durations = _prefetch_durations()

    sync_median = statistics.median(sync_durations)
    prefetch_median = statistics.median(prefetch_durations)
    samples_per_sec_sync = n / sync_median if sync_median > 0 else float("inf")
    samples_per_sec_prefetch = n / prefetch_median if prefetch_median > 0 else float("inf")
    speedup = sync_median / prefetch_median if prefetch_median > 0 else float("inf")

    print(f"--- {name} (n={n}, batch={batch_size}, cpu_work_iters={cpu_work_iters}, gpu_work_layers={gpu_work_layers}) ---")
    print(f"synchronous : {sync_median * 1000:8.2f} ms/epoch  ({samples_per_sec_sync:9.0f} samples/sec)")
    print(f"prefetch    : {prefetch_median * 1000:8.2f} ms/epoch  ({samples_per_sec_prefetch:9.0f} samples/sec)")
    print(f"speedup     : {speedup:.3f}x\n")


def _run_synthetic_workloads() -> None:
    print("=== Synthetic workload: CPU prep / H2D transfer / GPU compute overlap ===\n")
    gc.collect()
    _run_synthetic_case(
        "negligible CPU work, light GPU work (no overlap opportunity expected)",
        n=512, batch_size=32, feature_dim=64, cpu_work_iters=0, gpu_work_layers=1,
    )
    _run_synthetic_case(
        "heavy CPU work, light GPU work (CPU-prep/transfer should overlap with compute)",
        n=512, batch_size=32, feature_dim=64, cpu_work_iters=40, gpu_work_layers=1,
    )
    _run_synthetic_case(
        "light CPU work, heavy GPU work (transfer should hide behind compute)",
        n=512, batch_size=32, feature_dim=64, cpu_work_iters=2, gpu_work_layers=24,
    )
    print(
        "A ratio meaningfully above 1.0x demonstrates real overlap; a ratio near 1.0x is expected, "
        "not a bug, whenever there is little CPU/transfer cost left to hide behind compute (Section 60).\n"
    )


# -- 2. Real MNIST workload (Section 59) ---------------------------------------


def _make_mnist_dataset(n: int, seed: int = 0) -> TensorDataset:
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal((n, 1, 28, 28)).astype(np.float32))
    y = Tensor(rng.integers(0, 10, size=(n,)).astype(np.int64))
    return TensorDataset(x, y)


def _run_mnist_case(prefetch: bool, batch_size: int, n: int) -> "tuple[float, float]":
    from examples.mnist.model import build_model

    ds = _make_mnist_dataset(n)
    durations = []
    for trial in range(TRIALS):
        forge.random.seed(trial)
        model = build_model()
        model.to("cuda")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=np.random.default_rng(trial))
        trainer = Trainer(
            model, CrossEntropyLoss(), Adam(model.parameters(), lr=1e-3),
            device="cuda", verbose=False, prefetch=prefetch,
        )
        durations.append(_time_one_epoch(trainer, loader))
    median = statistics.median(durations)
    samples_per_sec = n / median if median > 0 else float("inf")
    return median, samples_per_sec


def _run_mnist_workload() -> None:
    print("=== Real MNIST CNN workload (synthetic MNIST-shaped batches) ===\n")
    n, batch_size = 1024, 64

    gc.collect()
    forge.cuda.empty_cache()
    mem_before = forge.cuda.memory_stats()
    pinned_before = forge.cuda.pinned_memory_stats()

    sync_median, sync_sps = _run_mnist_case(prefetch=False, batch_size=batch_size, n=n)
    prefetch_median, prefetch_sps = _run_mnist_case(prefetch=True, batch_size=batch_size, n=n)

    gc.collect()
    forge.cuda.empty_cache()
    mem_after = forge.cuda.memory_stats()
    pinned_after = forge.cuda.pinned_memory_stats()

    speedup = sync_median / prefetch_median if prefetch_median > 0 else float("inf")
    print(f"n={n}, batch_size={batch_size}")
    print(f"synchronous : {sync_median * 1000:8.2f} ms/epoch  ({sync_sps:9.0f} samples/sec)")
    print(f"prefetch    : {prefetch_median * 1000:8.2f} ms/epoch  ({prefetch_sps:9.0f} samples/sec)")
    print(f"speedup     : {speedup:.3f}x\n")
    print(
        f"CUDA active bytes:   before={mem_before.allocated_bytes:,}  after={mem_after.allocated_bytes:,}  "
        f"(after both runs + empty_cache(); should not have grown)"
    )
    print(
        f"pinned active bytes: before={pinned_before.pinned_active_bytes:,}  "
        f"after={pinned_after.pinned_active_bytes:,} (should be 0 -- no leaked pinned buffers)\n"
    )


def main() -> None:
    if not is_cuda_available():
        print("CUDA is not available on this machine -- async_dataloader_bench requires real CUDA hardware.")
        return

    _run_synthetic_workloads()
    _run_mnist_workload()


if __name__ == "__main__":
    main()
