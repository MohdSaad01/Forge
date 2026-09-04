"""End-to-end asynchronous training-pipeline profiler (Milestone 31).

    python -m benchmarks.pipeline_profile

Standalone script, not part of `python -m benchmarks`'s stable-schema
category list -- matching `async_dataloader_bench.py`/`stream_bench.py`'s own
"diagnostic script" status. This is the M31 profiler the milestone brief
asks for: it measures a *real* asynchronous M20 CNN MNIST training epoch
(`forge.data.CUDAPrefetchLoader` + a dedicated compute `Stream`, exactly
`Trainer(prefetch=True)`'s own pipeline -- see `docs/architecture/
async-dataloader.md`) without collapsing it into a synchronous one.

Unlike Milestone 21's `mnist_profile.py` (which synchronizes between every
phase to get clean per-phase numbers, deliberately sacrificing overlap in
the process), this script never synchronizes mid-epoch. Per-phase GPU-side
cost is instead measured with timing-enabled CUDA events (`forge.backend.
cuda.profiling_events.TimedEvent` -- see that module's docstring for why a
*separate* event type from the internal, timing-disabled `CUDAEvent` was
needed): one event pair per phase per batch, recorded on the compute stream
without any intervening host synchronization, all queried together in one
`forge.cuda.synchronize()` at the very end of the timed loop. This is the
"warmup -> synchronize -> record start -> submit workload -> record end ->
synchronize -> measure" recipe the milestone brief's Section 5 describes,
applied per-phase across a whole epoch rather than once around the whole
thing.

H2D transfer bandwidth is profiled separately, in isolation
(`_profile_transfer_sizes`) -- bracketing the live prefetch pipeline's own
transfer-stream submissions would require instrumenting `forge/data/
prefetch.py` itself, which the milestone brief's Section 34 asks to avoid
("keep profiling instrumentation ... outside normal execution where
practical"). Whether that transfer is actually hidden behind compute is
instead inferred honestly, without guessing: by comparing one epoch's real
wall-clock time against the *sum* of that epoch's measured GPU compute-phase
busy time (see `_render_report`'s "compute-stream utilization" line) --
wall time close to compute-busy time means the compute stream stays
continuously fed (transfer/CPU-prep successfully hidden); wall time
meaningfully larger means a bubble exists somewhere upstream.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

import forge
from forge import Tensor
from forge.backend.cuda.backend import is_cuda_available
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms
from forge.data import CUDAPrefetchLoader, DataLoader, TensorDataset
from forge.nn import CrossEntropyLoss
from forge.optim import Adam

from .environment import collect_environment

WARMUP_BATCHES = 5
BATCH_SIZES = (32, 64, 128)
PREFETCH_DEPTHS = (1, 2, 3)
TRANSFER_SIZES_BYTES = {"small": 4_096, "medium": 400_000, "large": 4_000_000}


def _make_mnist_dataset(n: int, seed: int = 0) -> TensorDataset:
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal((n, 1, 28, 28)).astype(np.float32))
    y = Tensor(rng.integers(0, 10, size=(n,)).astype(np.int64))
    return TensorDataset(x, y)


def _mean_elapsed_ms(pairs: "list[tuple[TimedEvent, TimedEvent]]") -> float:
    if not pairs:
        return 0.0
    return statistics.mean(elapsed_ms(s, e) for s, e in pairs)


# -- CPU-only component costs (Section 6) ------------------------------------
#
# These are inherently host-side, serial operations -- timing them with plain
# `time.perf_counter()` does not have the "destroys async overlap" hazard
# Section 5 warns about for *GPU* operations, since nothing here is dispatched
# asynchronously to a stream.


def _cpu_dataloader_component_cost(n: int, batch_size: int, iterations: int) -> float:
    """Mean wall-clock seconds for one `next(iter(loader))` call: dataset access + collation."""
    ds = _make_mnist_dataset(n)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=np.random.default_rng(0))

    def _iter_batches():
        # Re-`iter()`s the loader whenever it runs out, so the caller can
        # pull an arbitrary number of batches regardless of the dataset's
        # own size (matching `_pinned_staging_cost`'s identical helper).
        it = iter(loader)
        while True:
            try:
                yield next(it)
            except StopIteration:
                it = iter(loader)

    batches = _iter_batches()
    for _ in range(WARMUP_BATCHES):
        next(batches)

    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        next(batches)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return statistics.mean(durations)


def _pinned_staging_cost(n: int, batch_size: int, iterations: int) -> float:
    """Mean wall-clock seconds to stage one CPU batch (x, y) into fresh pinned buffers."""
    ds = _make_mnist_dataset(n)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=np.random.default_rng(0))
    it = iter(loader)

    def _next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    for _ in range(WARMUP_BATCHES):
        x, y = _next_batch()
        for component in (x, y):
            host = component.numpy()
            mem = forge.cuda.PinnedMemory(host.nbytes)
            pinned_view = mem.numpy(shape=host.shape, dtype=host.dtype)
            pinned_view[:] = host

    durations = []
    for _ in range(iterations):
        x, y = _next_batch()
        t0 = time.perf_counter()
        for component in (x, y):
            host = component.numpy()
            mem = forge.cuda.PinnedMemory(host.nbytes)
            pinned_view = mem.numpy(shape=host.shape, dtype=host.dtype)
            pinned_view[:] = host
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return statistics.mean(durations)


def _transfer_submission_cost(batch_size: int, iterations: int) -> float:
    """Mean wall-clock seconds to *submit* (not complete) one async H2D copy of an MNIST batch."""
    stream = forge.cuda.Stream()
    host = np.random.default_rng(0).standard_normal((batch_size, 1, 28, 28)).astype(np.float32)
    mem = forge.cuda.PinnedMemory(host.nbytes)
    pinned_view = mem.numpy(shape=host.shape, dtype=host.dtype)
    pinned_view[:] = host
    pinned_tensor = Tensor(pinned_view, device="cpu")

    for _ in range(WARMUP_BATCHES):
        with forge.cuda.stream(stream):
            pinned_tensor.to("cuda", non_blocking=True)
    stream.synchronize()

    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with forge.cuda.stream(stream):
            pinned_tensor.to("cuda", non_blocking=True)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    stream.synchronize()
    return statistics.mean(durations)


# -- isolated H2D transfer bandwidth sweep (Section 27) -----------------------


def _profile_transfer_sizes() -> "dict[str, dict]":
    results = {}
    stream = forge.cuda.Stream()
    for name, nbytes in TRANSFER_SIZES_BYTES.items():
        n_elements = nbytes // 4  # float32 -- Tensor has no uint8 dtype (see dtype.py's DType enum)
        host = np.random.default_rng(0).standard_normal(n_elements).astype(np.float32)
        mem = forge.cuda.PinnedMemory(host.nbytes)
        pinned_view = mem.numpy(shape=(n_elements,), dtype=np.float32)
        pinned_view[:] = host
        nbytes = host.nbytes
        pinned_tensor = Tensor(pinned_view, device="cpu")

        for _ in range(WARMUP_BATCHES):
            with forge.cuda.stream(stream):
                pinned_tensor.to("cuda", non_blocking=True)
        stream.synchronize()

        pairs = []
        for _ in range(30):
            start = TimedEvent()
            start.record(stream.handle)
            with forge.cuda.stream(stream):
                pinned_tensor.to("cuda", non_blocking=True)
            end = TimedEvent()
            end.record(stream.handle)
            pairs.append((start, end))
        stream.synchronize()

        mean_ms = _mean_elapsed_ms(pairs)
        bandwidth_gbps = (nbytes / 1e9) / (mean_ms / 1e3) if mean_ms > 0 else float("inf")
        results[name] = {"bytes": nbytes, "mean_ms": mean_ms, "effective_gbps": bandwidth_gbps}
    return results


# -- live async epoch, per-phase GPU busy time (Sections 4, 5, 14, 15, 16) ----


def _profile_async_epoch(batch_size: int, prefetch_size: int, n_samples: int) -> "dict[str, float]":
    from examples.mnist.model import build_model

    forge.random.seed(0)
    model = build_model()
    model.to("cuda")
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    ds = _make_mnist_dataset(n_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=np.random.default_rng(0))
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=prefetch_size)
    compute_stream = forge.cuda.Stream()

    # Warmup: absorbs lazy kernel-module load / nvcc cache probe / CUDA
    # context first-touch effects, uncounted -- matches `timing.py`'s
    # established convention.
    with forge.cuda.stream(compute_stream):
        count = 0
        for x, y in prefetch_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            count += 1
            if count >= WARMUP_BATCHES:
                break
    forge.cuda.synchronize()

    fwd_events: "list[tuple[TimedEvent, TimedEvent]]" = []
    loss_events: "list[tuple[TimedEvent, TimedEvent]]" = []
    bwd_events: "list[tuple[TimedEvent, TimedEvent]]" = []
    opt_events: "list[tuple[TimedEvent, TimedEvent]]" = []
    cpu_step_wall: "list[float]" = []

    gc.disable()
    try:
        epoch_start = time.perf_counter()
        with forge.cuda.stream(compute_stream):
            for x, y in prefetch_loader:
                t0 = time.perf_counter()

                e0 = TimedEvent()
                e0.record(compute_stream.handle)
                optimizer.zero_grad()
                out = model(x)
                e1 = TimedEvent()
                e1.record(compute_stream.handle)
                fwd_events.append((e0, e1))

                loss = loss_fn(out, y)
                e2 = TimedEvent()
                e2.record(compute_stream.handle)
                loss_events.append((e1, e2))

                loss.backward()
                e3 = TimedEvent()
                e3.record(compute_stream.handle)
                bwd_events.append((e2, e3))

                optimizer.step()
                e4 = TimedEvent()
                e4.record(compute_stream.handle)
                opt_events.append((e3, e4))

                cpu_step_wall.append(time.perf_counter() - t0)
        forge.cuda.synchronize()
        epoch_wall = time.perf_counter() - epoch_start
    finally:
        gc.enable()

    fwd_ms = _mean_elapsed_ms(fwd_events)
    loss_ms = _mean_elapsed_ms(loss_events)
    bwd_ms = _mean_elapsed_ms(bwd_events)
    opt_ms = _mean_elapsed_ms(opt_events)
    gpu_busy_ms_per_step = fwd_ms + loss_ms + bwd_ms + opt_ms
    n_steps = len(fwd_events)
    epoch_ms = epoch_wall * 1000
    compute_busy_total_ms = gpu_busy_ms_per_step * n_steps

    return {
        "batch_size": batch_size,
        "prefetch_size": prefetch_size,
        "n_samples": n_samples,
        "steps": n_steps,
        "epoch_wall_ms": epoch_ms,
        "samples_per_sec": n_samples / epoch_wall if epoch_wall > 0 else float("inf"),
        "mean_cpu_step_wall_ms": statistics.mean(cpu_step_wall) * 1000 if cpu_step_wall else 0.0,
        "gpu_forward_ms": fwd_ms,
        "gpu_loss_ms": loss_ms,
        "gpu_backward_ms": bwd_ms,
        "gpu_optimizer_ms": opt_ms,
        "gpu_busy_ms_per_step": gpu_busy_ms_per_step,
        "compute_stream_busy_total_ms": compute_busy_total_ms,
        "compute_stream_utilization": (compute_busy_total_ms / epoch_ms) if epoch_ms > 0 else 0.0,
    }


# -- allocator / pinned-memory characterization (Sections 11, 12) ------------


def _profile_allocator_and_pinned(batch_size: int, n_samples: int) -> "dict[str, dict]":
    gc.collect()
    forge.cuda.empty_cache()
    mem_before = forge.cuda.memory_stats()
    pinned_before = forge.cuda.pinned_memory_stats()

    result = _profile_async_epoch(batch_size, prefetch_size=2, n_samples=n_samples)

    mem_after = forge.cuda.memory_stats()
    pinned_after = forge.cuda.pinned_memory_stats()
    gc.collect()
    forge.cuda.empty_cache()
    mem_after_gc = forge.cuda.memory_stats()

    return {
        "epoch": result,
        "allocator_before": mem_before.as_dict(),
        "allocator_after": mem_after.as_dict(),
        "allocator_after_gc_and_empty_cache": mem_after_gc.as_dict(),
        "pinned_before": pinned_before.as_dict(),
        "pinned_after": pinned_after.as_dict(),
    }


# -- reporting ----------------------------------------------------------------


def _render_report(profile: dict) -> str:
    lines = ["=== M31 asynchronous pipeline profile ===", ""]

    lines.append("-- CPU-only component costs (isolated, batch_size=64) --")
    cpu = profile["cpu_components"]
    lines.append(f"  dataset+collate (DataLoader.__next__):  {cpu['dataloader_component_ms']:8.4f} ms/batch")
    lines.append(f"  pinned staging (x + y -> PinnedMemory):  {cpu['pinned_staging_ms']:8.4f} ms/batch")
    lines.append(f"  async H2D submission (launch-only):      {cpu['transfer_submission_ms']:8.4f} ms/batch")
    lines.append("")

    lines.append("-- H2D transfer bandwidth (isolated, pinned source, async) --")
    for name, r in profile["transfer_sizes"].items():
        lines.append(f"  {name:<8} {r['bytes']:>9,} bytes: {r['mean_ms']:8.4f} ms  ({r['effective_gbps']:6.3f} GB/s)")
    lines.append("")

    lines.append("-- Live async MNIST epoch: per-phase GPU busy time (batch size sweep) --")
    for r in profile["batch_size_sweep"]:
        lines.append(
            f"  batch={r['batch_size']:>4}  epoch={r['epoch_wall_ms']:9.2f}ms  "
            f"({r['samples_per_sec']:8.0f} samples/sec)  "
            f"fwd={r['gpu_forward_ms']:.4f}  loss={r['gpu_loss_ms']:.4f}  "
            f"bwd={r['gpu_backward_ms']:.4f}  opt={r['gpu_optimizer_ms']:.4f}  "
            f"compute_util={r['compute_stream_utilization'] * 100:5.1f}%"
        )
    lines.append("")

    lines.append("-- Live async MNIST epoch: prefetch depth sweep (batch_size=64) --")
    for r in profile["prefetch_depth_sweep"]:
        lines.append(
            f"  prefetch_size={r['prefetch_size']}  epoch={r['epoch_wall_ms']:9.2f}ms  "
            f"({r['samples_per_sec']:8.0f} samples/sec)  compute_util={r['compute_stream_utilization'] * 100:5.1f}%"
        )
    lines.append("")

    alloc = profile["allocator_and_pinned"]
    lines.append("-- Allocator characterization (one profiled epoch, batch_size=64, prefetch_size=2) --")
    for key in ("allocator_before", "allocator_after", "allocator_after_gc_and_empty_cache"):
        d = alloc[key]
        lines.append(
            f"  {key:<34} active={d['allocated_bytes']:>10,}B  reserved={d['reserved_bytes']:>10,}B  "
            f"cached={d['cached_bytes']:>10,}B  pending={d['pending_bytes']:>8,}B  "
            f"hits={d['cache_hit_count']:>6}  misses={d['cache_miss_count']:>4}"
        )
    lines.append("-- Pinned-memory characterization --")
    for key in ("pinned_before", "pinned_after"):
        d = alloc[key]
        lines.append(
            f"  {key:<15} active={d['pinned_active_bytes']:>10,}B  peak={d['pinned_peak_bytes']:>10,}B  "
            f"allocs={d['pinned_allocation_count']:>6}  frees={d['pinned_free_count']:>6}"
        )

    return "\n".join(lines)


def _run(n_samples: int) -> dict:
    profile = {
        "cpu_components": {
            "dataloader_component_ms": _cpu_dataloader_component_cost(n_samples, 64, 30) * 1000,
            "pinned_staging_ms": _pinned_staging_cost(n_samples, 64, 30) * 1000,
            "transfer_submission_ms": _transfer_submission_cost(64, 30) * 1000,
        },
        "transfer_sizes": _profile_transfer_sizes(),
        "batch_size_sweep": [
            _profile_async_epoch(bs, prefetch_size=2, n_samples=n_samples) for bs in BATCH_SIZES
        ],
        "prefetch_depth_sweep": [
            _profile_async_epoch(64, prefetch_size=p, n_samples=n_samples) for p in PREFETCH_DEPTHS
        ],
        "allocator_and_pinned": _profile_allocator_and_pinned(64, n_samples),
    }
    return profile


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/pipeline_profile.json")
    parser.add_argument("--n-samples", type=int, default=1024)
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- pipeline_profile requires real CUDA hardware.")
        return

    profile = _run(args.n_samples)
    print(_render_report(profile))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"environment": collect_environment(), "profile": profile}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved profile -> {output_path}")


if __name__ == "__main__":
    main()
