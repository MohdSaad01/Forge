"""Transfer / stream-dependency / allocator characterization (Milestone 35).

    python -m benchmarks.m35_transfer_stream_alloc

Covers Sections 21-23 of the M35 brief. Forge already has mature, dedicated
benchmarks for every piece of this -- this script does not reimplement any
of them, it **calls them directly** and reinterprets their output through
`benchmarks.roofline`:

- **Transfers** (Section 21): pageable sync H2D/D2H reuses
  `transfer_bench.run_transfer_benchmarks()` (`Tensor.to()`, `non_blocking=
  False`) directly. Pinned async H2D reuses `pipeline_profile.
  _profile_transfer_sizes`/`_transfer_submission_cost` directly (already
  splits submission latency from measured completion time, exactly Section
  21's "do not confuse submission latency with transfer throughput"). Pinned
  async D2H is the one genuinely new piece -- a same-shape mirror of
  `pipeline_profile`'s H2D pattern, since that module only covers H2D.
- **Streams** (Section 22): calls `stream_dependency_bench`'s five private
  `_bench_*` functions directly (same-stream baseline, cross-stream
  dependency, multi-input dependency, event creation, cross-stream allocator
  reuse) -- Milestone 28's own dedicated benchmark for exactly this.
- **Allocator** (Section 23): calls `allocator_bench.bench_single_size`/
  `bench_multi_size` directly -- Milestone 25's own dedicated benchmark for
  cold-alloc/cache-hit/release/multi-size behavior.

Stream-dependency and allocator numbers are reported as absolute overhead
(seconds), not run through the FLOP/byte roofline model -- there is no
meaningful FLOP/byte count for "wait on an event" or "look up a free-list
entry" (Section 27 allows a classification to be explicitly qualitative when
a numeric roofline model does not apply).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import is_cuda_available

from forge.backend.cuda.profiling_events import TimedEvent

from . import allocator_bench, stream_dependency_bench
from .environment import collect_environment
from .pipeline_profile import TRANSFER_SIZES_BYTES, _mean_elapsed_ms, _profile_transfer_sizes, _transfer_submission_cost
from .sizes import ELEMENTWISE_SIZES
from .transfer_bench import run_transfer_benchmarks

WARMUP = 5
ITERATIONS = 30


# -- pinned async D2H (mirrors pipeline_profile._profile_transfer_sizes, but D2H) --


def _profile_pinned_d2h_sizes() -> "dict[str, dict]":
    """Mirrors `pipeline_profile._profile_transfer_sizes`'s exact loop shape
    (submit -> record end -> *no* per-iteration sync -> one `stream.
    synchronize()` after the whole batch) so H2D and D2H are measured the
    same way and are directly comparable. Each iteration's result Tensor
    (a fresh pinned buffer -- see `Tensor.to`'s docstring) is kept alive in
    `keep_alive` until the final synchronize, matching H2D's single
    long-lived source buffer -- never reading `._data` inside the timed
    loop, which would force a per-iteration host round-trip H2D does not
    pay either.
    """
    results = {}
    stream = forge.cuda.Stream()
    for name, nbytes in TRANSFER_SIZES_BYTES.items():
        n_elements = nbytes // 4
        host = np.random.default_rng(0).standard_normal(n_elements).astype(np.float32)
        cuda_tensor = forge.Tensor(host, device="cuda")

        for _ in range(WARMUP):
            with forge.cuda.stream(stream):
                cuda_tensor.to("cpu", non_blocking=True)
        stream.synchronize()

        pairs = []
        keep_alive = []
        for _ in range(30):
            start = TimedEvent()
            start.record(stream.handle)
            with forge.cuda.stream(stream):
                result = cuda_tensor.to("cpu", non_blocking=True)
            end = TimedEvent()
            end.record(stream.handle)
            pairs.append((start, end))
            keep_alive.append(result)
        stream.synchronize()

        mean_ms = _mean_elapsed_ms(pairs)
        bandwidth_gbps = (nbytes / 1e9) / (mean_ms / 1e3) if mean_ms > 0 else float("inf")
        results[name] = {"bytes": nbytes, "mean_ms": mean_ms, "effective_gbps": bandwidth_gbps}
    return results


def _d2h_submission_cost(batch_bytes: int, iterations: int) -> float:
    """Mean wall-clock seconds to *submit* (not complete) one async D2H copy."""
    import time
    import statistics as _stats

    stream = forge.cuda.Stream()
    n_elements = batch_bytes // 4
    host = np.random.default_rng(0).standard_normal(n_elements).astype(np.float32)
    cuda_tensor = forge.Tensor(host, device="cuda")

    for _ in range(WARMUP):
        with forge.cuda.stream(stream):
            cuda_tensor.to("cpu", non_blocking=True)
    stream.synchronize()

    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with forge.cuda.stream(stream):
            cuda_tensor.to("cpu", non_blocking=True)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    stream.synchronize()
    return _stats.mean(durations)


def _profile_transfers() -> dict:
    pageable_results = run_transfer_benchmarks()
    pageable = {}
    for r in pageable_results:
        pageable.setdefault(r.scale, {})[r.operation] = {
            "mean_s": r.mean_seconds, "bytes": r.extra["bytes"], "gbps": r.extra["throughput_GBps"],
        }

    pinned_h2d = _profile_transfer_sizes()
    pinned_d2h = _profile_pinned_d2h_sizes()

    submission = {
        "h2d_submission_s": _transfer_submission_cost(64, ITERATIONS),  # matches pipeline_profile's own units
        "d2h_submission_s": _d2h_submission_cost(TRANSFER_SIZES_BYTES["medium"], ITERATIONS),
    }

    return {
        "pageable_sync": pageable,
        "pinned_async_h2d": pinned_h2d,
        "pinned_async_d2h": pinned_d2h,
        "submission_vs_completion": submission,
    }


# -- streams (Section 22) ---------------------------------------------------


def _profile_streams() -> dict:
    same_stream_s = stream_dependency_bench._bench_same_stream_baseline()
    cross_stream_s = stream_dependency_bench._bench_cross_stream_dependency()
    multi_input_s = stream_dependency_bench._bench_multi_input_dependency()
    event_creation_s = stream_dependency_bench._bench_event_creation()
    alloc_reuse_s = stream_dependency_bench._bench_cross_stream_allocator_reuse()

    return {
        "same_stream_baseline_s": same_stream_s,
        "cross_stream_dependency_s": cross_stream_s,
        "multi_input_dependency_s": multi_input_s,
        "event_creation_s": event_creation_s,
        "cross_stream_allocator_reuse_s": alloc_reuse_s,
        "cross_stream_overhead_s": cross_stream_s - same_stream_s,
        "multi_input_overhead_s": multi_input_s - same_stream_s,
        "cross_stream_overhead_fraction_of_op": (cross_stream_s - same_stream_s) / same_stream_s if same_stream_s > 0 else float("inf"),
    }


# -- allocator (Section 23) -------------------------------------------------


def _profile_allocator() -> dict:
    per_size = [allocator_bench.bench_single_size(scale, n) for scale, n in ELEMENTWISE_SIZES.items()]
    multi = allocator_bench.bench_multi_size()
    return {"per_size": per_size, "multi_size": multi}


def _run() -> dict:
    return {
        "transfers": _profile_transfers(),
        "streams": _profile_streams(),
        "allocator": _profile_allocator(),
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M35 transfer / stream / allocator characterization (940MX, real CUDA) ===", ""]

    lines.append("-- Transfers: pageable sync (Tensor.to(), non_blocking=False) --")
    for scale, ops in profile["transfers"]["pageable_sync"].items():
        for direction, r in ops.items():
            lines.append(f"  {scale:<8}{direction:<5}{r['mean_s']*1e3:9.4f}ms  {r['gbps']:7.3f} GB/s  ({r['bytes']:,} bytes)")
    lines.append("")

    lines.append("-- Transfers: pinned async H2D (completion time, TimedEvent) --")
    for name, r in profile["transfers"]["pinned_async_h2d"].items():
        lines.append(f"  {name:<8}{r['mean_ms']:9.4f}ms  {r['effective_gbps']:7.3f} GB/s  ({r['bytes']:,} bytes)")
    lines.append("")

    lines.append("-- Transfers: pinned async D2H (completion time, TimedEvent) --")
    for name, r in profile["transfers"]["pinned_async_d2h"].items():
        lines.append(f"  {name:<8}{r['mean_ms']:9.4f}ms  {r['effective_gbps']:7.3f} GB/s  ({r['bytes']:,} bytes)")
    lines.append("")

    sub = profile["transfers"]["submission_vs_completion"]
    lines.append("-- Submission latency vs. completion time --")
    lines.append(f"  H2D submission (launch-only): {sub['h2d_submission_s']*1e6:8.2f}us")
    lines.append(f"  D2H submission (launch-only): {sub['d2h_submission_s']*1e6:8.2f}us")
    lines.append("")

    s = profile["streams"]
    lines.append("-- Streams (Section 22) --")
    lines.append(f"  same-stream baseline:         {s['same_stream_baseline_s']*1e6:8.2f}us")
    lines.append(f"  cross-stream dependency:      {s['cross_stream_dependency_s']*1e6:8.2f}us  (+{s['cross_stream_overhead_s']*1e6:.2f}us, {s['cross_stream_overhead_fraction_of_op']*100:.1f}% overhead)")
    lines.append(f"  multi-input dependency:       {s['multi_input_dependency_s']*1e6:8.2f}us  (+{s['multi_input_overhead_s']*1e6:.2f}us)")
    lines.append(f"  event creation (isolated):    {s['event_creation_s']*1e6:8.2f}us")
    lines.append(f"  cross-stream allocator reuse: {s['cross_stream_allocator_reuse_s']*1e6:8.2f}us")
    lines.append("")

    lines.append("-- Allocator (Section 23) --")
    for r in profile["allocator"]["per_size"]:
        lines.append(
            f"  {r['scale']:<8}({r['nbytes']:>9,}B)  direct={r['direct']['mean_seconds']*1e6:8.2f}us  "
            f"cached={r['cached']['mean_seconds']*1e6:8.2f}us  speedup={r['speedup_x']:.1f}x"
        )
    multi = profile["allocator"]["multi_size"]
    lines.append(
        f"  multi-size: {multi['total_requests']} requests, "
        f"{multi['cache_hit_count_delta']} hits / {multi['cache_miss_count_delta']} misses"
    )
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m35_transfer_stream_alloc.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m35_transfer_stream_alloc requires real CUDA hardware.")
        return

    profile = _run()
    print(_render_report(profile))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"environment": collect_environment(), "profile": profile}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved profile -> {output_path}")


if __name__ == "__main__":
    main()
