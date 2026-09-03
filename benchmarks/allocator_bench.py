"""Allocation-path microbenchmark: direct `cudaMalloc`/`cudaFree` vs. the Milestone 25 caching allocator.

Section 21 of the milestone brief: "Add a focused benchmark that repeatedly
performs allocations of the same size. Compare direct cudaMalloc/cudaFree
against allocator allocate/release ... Also benchmark multiple sizes to
verify exact-size behavior." This is that benchmark -- a standalone script
(`python -m benchmarks.allocator_bench`), not part of `python -m benchmarks`'s
category list (`benchmarks/run.py`), matching `alloc_profile.py`'s own
"diagnostic script, not a stable-schema benchmark category" status.

Two paths are timed per size, each `warmup + iterations` alloc/free cycles:

- **direct**: `forge.backend.cuda.allocator.raw_malloc`/`raw_free` -- a real,
  uncached `cudaMalloc`/`cudaFree` pair every single iteration, bypassing the
  cache entirely (the same host-blocking driver calls `alloc_profile.py`'s
  `_measure_alloc_free_overhead` already measures in isolation).
- **cached**: `forge.backend.cuda.allocator.allocate`/`release` -- after one
  warmup iteration primes the cache with a block of that exact size, every
  further iteration is a cache hit/release: no driver call at all.

`multi_size` repeats the cached path across several *distinct* sizes in the
same run (rather than one size repeated) to directly demonstrate the
exact-size policy: each size gets its own cache entry, a same-size request
always hits, and a different-size request never does (Section 13 of the
milestone brief).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from forge.backend.cuda import allocator as _allocator
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available

from .environment import collect_environment
from .sizes import ELEMENTWISE_SIZES

_WARMUP = 10
_ITERATIONS = 200


def _time(fn, iterations: int) -> dict:
    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return {
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "stdev_seconds": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "iterations": iterations,
    }


def _bench_direct(lib, nbytes: int) -> dict:
    def cycle():
        ptr = _allocator.raw_malloc(lib, nbytes)
        _allocator.raw_free(lib, ptr)

    for _ in range(_WARMUP):
        cycle()
    return _time(cycle, _ITERATIONS)


def _bench_cached(lib, nbytes: int) -> dict:
    def cycle():
        ptr = _allocator.allocate(lib, nbytes)
        _allocator.release(nbytes, ptr)

    for _ in range(_WARMUP):  # primes the cache: the first call is a miss, the rest hits
        cycle()

    stats_before = _allocator.memory_stats()
    timing = _time(cycle, _ITERATIONS)
    stats_after = _allocator.memory_stats()
    timing["cache_hit_count_delta"] = stats_after.cache_hit_count - stats_before.cache_hit_count
    timing["cache_miss_count_delta"] = stats_after.cache_miss_count - stats_before.cache_miss_count
    return timing


def bench_single_size(scale: str, n: int) -> dict:
    lib = get_cuda_backend()._lib
    nbytes = n * 4
    direct = _bench_direct(lib, nbytes)
    cached = _bench_cached(lib, nbytes)
    speedup = direct["mean_seconds"] / cached["mean_seconds"] if cached["mean_seconds"] > 0 else float("inf")
    return {"scale": scale, "nbytes": nbytes, "direct": direct, "cached": cached, "speedup_x": speedup}


def bench_multi_size() -> dict:
    """Interleave several distinct sizes through the cached path once each is
    warm, to demonstrate exact-size reuse never crosses sizes (Section 13)."""
    lib = get_cuda_backend()._lib
    sizes = [n * 4 for n in ELEMENTWISE_SIZES.values()]

    # Warm every size's cache slot once.
    for nbytes in sizes:
        ptr = _allocator.allocate(lib, nbytes)
        _allocator.release(nbytes, ptr)

    stats_before = _allocator.memory_stats()
    for _ in range(_ITERATIONS):
        for nbytes in sizes:
            ptr = _allocator.allocate(lib, nbytes)
            _allocator.release(nbytes, ptr)
    stats_after = _allocator.memory_stats()

    total_requests = _ITERATIONS * len(sizes)
    return {
        "sizes_bytes": sizes,
        "requests_per_size": _ITERATIONS,
        "total_requests": total_requests,
        "cache_hit_count_delta": stats_after.cache_hit_count - stats_before.cache_hit_count,
        "cache_miss_count_delta": stats_after.cache_miss_count - stats_before.cache_miss_count,
        "cached_bytes_after": stats_after.cached_bytes,
    }


def _render_summary(results: dict) -> str:
    lines = ["Direct cudaMalloc/cudaFree vs. caching allocator allocate/release:"]
    for r in results["single_size"]:
        lines.append(
            f"  {r['scale']:<8} ({r['nbytes']:>9} bytes): "
            f"direct {r['direct']['mean_seconds'] * 1e6:8.2f} us, "
            f"cached {r['cached']['mean_seconds'] * 1e6:8.2f} us, "
            f"speedup {r['speedup_x']:6.1f}x "
            f"(hits={r['cached']['cache_hit_count_delta']}, misses={r['cached']['cache_miss_count_delta']})"
        )
    m = results["multi_size"]
    lines.append("")
    lines.append(
        f"Multi-size exact-size check: {m['total_requests']} requests across {len(m['sizes_bytes'])} sizes -> "
        f"{m['cache_hit_count_delta']} hits, {m['cache_miss_count_delta']} misses "
        f"(expect misses == 0 once warm, hits == total_requests)."
    )
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/allocator_bench.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- the allocator microbenchmark requires real CUDA hardware.")
        return

    environment = collect_environment()
    results = {
        "single_size": [bench_single_size(scale, n) for scale, n in ELEMENTWISE_SIZES.items()],
        "multi_size": bench_multi_size(),
    }

    print(_render_summary(results))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"environment": environment, "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved allocator microbenchmark -> {output_path}")


if __name__ == "__main__":
    main()
