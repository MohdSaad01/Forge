"""Offline analysis of a `forge.cuda.profiler` allocation trace (Milestone 24).

Every function here is a pure function over a `tuple[AllocationEvent, ...]`
(`forge.backend.cuda.profiler.AllocationEvent`) -- no CUDA calls, no Forge
Tensor/Module objects, nothing mutated. This is analysis tooling, not
runtime code: `simulate_caching_allocator` in particular is an **offline
simulation** of what a caching allocator's bookkeeping would look like
replayed against a trace Forge's real (direct `cudaMalloc`/`cudaFree`)
allocator already produced -- it never runs during, or changes the behavior
of, any real Forge allocation. See `docs/architecture/cuda-memory-
allocator.md`.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.backend.cuda.profiler import AllocationEvent

# -- size distribution ---------------------------------------------------------

# Buckets per the milestone brief's suggested set. Upper bound is exclusive
# except the last (unbounded) bucket.
SIZE_BUCKETS: "list[tuple[str, int, float]]" = [
    ("< 1 KB", 0, 1_024),
    ("1-4 KB", 1_024, 4_096),
    ("4-16 KB", 4_096, 16_384),
    ("16-64 KB", 16_384, 65_536),
    ("64-256 KB", 65_536, 262_144),
    ("256 KB-1 MB", 262_144, 1_048_576),
    ("1-4 MB", 1_048_576, 4_194_304),
    ("4-16 MB", 4_194_304, 16_777_216),
    ("16-64 MB", 16_777_216, 67_108_864),
    ("64+ MB", 67_108_864, float("inf")),
]


def _bucket_for(nbytes: int) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= nbytes < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def size_distribution(events: "tuple[AllocationEvent, ...]", kind: str = "alloc") -> dict:
    """Summarize the byte-size distribution of every `kind` ("alloc" or "free") event.

    Returns total/min/max/mean/median, a size -> count histogram of the most
    common exact sizes (for detecting repeated-shape allocation patterns),
    and a bucketed histogram (`SIZE_BUCKETS`) for the coarse
    tiny/medium/large-buffer question the milestone brief asks.
    """
    sizes = [e.nbytes for e in events if e.kind == kind]
    if not sizes:
        return {
            "count": 0, "total_bytes": 0, "min_bytes": 0, "max_bytes": 0,
            "mean_bytes": 0.0, "median_bytes": 0.0, "common_sizes": [], "bucket_counts": {},
        }

    exact_counts: "dict[int, int]" = {}
    for s in sizes:
        exact_counts[s] = exact_counts.get(s, 0) + 1
    common = sorted(exact_counts.items(), key=lambda kv: -kv[1])[:10]

    bucket_counts: "dict[str, int]" = {name: 0 for name, _, _ in SIZE_BUCKETS}
    for s in sizes:
        bucket_counts[_bucket_for(s)] += 1

    return {
        "count": len(sizes),
        "total_bytes": sum(sizes),
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "mean_bytes": statistics.mean(sizes),
        "median_bytes": statistics.median(sizes),
        "common_sizes": [{"nbytes": n, "count": c} for n, c in common],
        "bucket_counts": bucket_counts,
    }


# -- lifetime analysis -----------------------------------------------------------

LIFETIME_BUCKETS: "list[tuple[str, float, float]]" = [
    ("< 1 ms", 0.0, 1e-3),
    ("1-10 ms", 1e-3, 1e-2),
    ("10-100 ms", 1e-2, 1e-1),
    ("100 ms-1 s", 1e-1, 1.0),
    ("1-10 s", 1.0, 10.0),
    ("> 10 s", 10.0, float("inf")),
]


def _lifetime_bucket_for(seconds: float) -> str:
    for name, lo, hi in LIFETIME_BUCKETS:
        if lo <= seconds < hi:
            return name
    return LIFETIME_BUCKETS[-1][0]


def pair_lifetimes(events: "tuple[AllocationEvent, ...]") -> "tuple[list[float], int]":
    """Pair each "alloc" with its chronologically-next "free" of the same `block_id`.

    Events are assumed already in chronological order (as `AllocationProfiler.events()`
    returns them -- appended in real time). A `block_id` (a raw pointer value) may be
    reused by the CUDA driver after a free, so pairing is done as a per-`block_id`
    FIFO queue of pending allocation timestamps, not a dict keyed once -- this is
    correct precisely because, single-threaded, a given address cannot be
    reallocated before its previous occupant is freed.

    Returns `(lifetimes_seconds, still_live_count)` -- `still_live_count` is the
    number of "alloc" events with no matching "free" in this trace (persistent
    allocations still alive when the trace was captured).
    """
    pending: "dict[int, list[float]]" = {}
    lifetimes: "list[float]" = []
    for e in events:
        if e.kind == "alloc":
            pending.setdefault(e.block_id, []).append(e.timestamp)
        elif e.kind == "free":
            queue = pending.get(e.block_id)
            if queue:
                start = queue.pop(0)
                lifetimes.append(e.timestamp - start)
    still_live = sum(len(q) for q in pending.values())
    return lifetimes, still_live


def lifetime_distribution(events: "tuple[AllocationEvent, ...]") -> dict:
    """Bucket paired alloc/free lifetimes (`LIFETIME_BUCKETS`) plus the persistent count."""
    lifetimes, still_live = pair_lifetimes(events)
    bucket_counts: "dict[str, int]" = {name: 0 for name, _, _ in LIFETIME_BUCKETS}
    for lt in lifetimes:
        bucket_counts[_lifetime_bucket_for(lt)] += 1
    return {
        "paired_count": len(lifetimes),
        "still_live_count": still_live,
        "mean_seconds": statistics.mean(lifetimes) if lifetimes else 0.0,
        "median_seconds": statistics.median(lifetimes) if lifetimes else 0.0,
        "min_seconds": min(lifetimes) if lifetimes else 0.0,
        "max_seconds": max(lifetimes) if lifetimes else 0.0,
        "bucket_counts": bucket_counts,
    }


# -- persistent vs. temporary classification -------------------------------------


def persistent_vs_temporary(events: "tuple[AllocationEvent, ...]") -> dict:
    """Split total allocated bytes into "persistent" (never freed in this trace) vs. "temporary".

    A block with no matching free by the end of the trace is classified
    persistent -- appropriate for a trace captured over a bounded steady-state
    window (e.g. N training iterations): Parameters/Adam moments allocated
    once before the window, or still live at the window's end, show up this
    way; per-iteration intermediates show up as paired alloc/free pairs
    (temporary), per `pair_lifetimes` above.
    """
    pending: "dict[int, list[int]]" = {}
    for e in events:
        if e.kind == "alloc":
            pending.setdefault(e.block_id, []).append(e.nbytes)
        elif e.kind == "free":
            queue = pending.get(e.block_id)
            if queue:
                queue.pop(0)
    persistent_bytes = sum(sum(q) for q in pending.values())
    persistent_count = sum(len(q) for q in pending.values())

    total_alloc_events = sum(1 for e in events if e.kind == "alloc")
    total_alloc_bytes = sum(e.nbytes for e in events if e.kind == "alloc")
    temporary_bytes = total_alloc_bytes - persistent_bytes
    temporary_count = total_alloc_events - persistent_count

    return {
        "persistent_bytes": persistent_bytes,
        "persistent_count": persistent_count,
        "temporary_bytes": temporary_bytes,
        "temporary_count": temporary_count,
        "total_alloc_bytes": total_alloc_bytes,
        "total_alloc_count": total_alloc_events,
    }


# -- offline caching-allocator simulation ----------------------------------------


def _round_exact(nbytes: int) -> int:
    return nbytes


def _round_size_class(nbytes: int) -> int:
    """Round up to the bucket ceiling a size-class allocator would hand out.

    Uses `SIZE_BUCKETS`' own boundaries as the size classes -- a block
    request is rounded up to the smallest bucket boundary >= its size (the
    top, unbounded bucket rounds up to the next power-of-two-ish step above
    its floor, since it has no finite ceiling to round to).
    """
    if nbytes <= 0:
        return 0
    for _, lo, hi in SIZE_BUCKETS[:-1]:  # all but the unbounded "64+ MB" bucket
        if nbytes <= hi:
            return int(hi)
    # Above the largest finite boundary: round up to the next 64 MB step.
    step = 67_108_864
    return ((nbytes + step - 1) // step) * step


def simulate_caching_allocator(events: "tuple[AllocationEvent, ...]", policy: str = "exact") -> dict:
    """Replay `events` against a minimal offline cache-simulation policy.

    **Simulation only** -- this never runs inside Forge's real allocator (see
    module docstring). Two policies are modeled, corresponding to Candidate
    Designs A and B in `docs/architecture/cuda-memory-allocator.md`:

    - `"exact"`: a freed block is reused only for a request of the identical
      size (Candidate A, exact-size cache).
    - `"size_class"`: requests are rounded up to `SIZE_BUCKETS` boundaries
      before the cache lookup, so same-bucket-but-different-size requests can
      still hit (Candidate B, size-class cache) -- at the cost of internal
      fragmentation (the rounded-up bytes never actually used).

    Simulated policy, on each "alloc" event: reuse a cached block of the
    (rounded) requested size if one is free; otherwise simulate a fresh
    `cudaMalloc` of that (rounded) size. On each "free" event: return the
    block to the cache (never actually released back to the driver in this
    simulation -- see `reserved_bytes`/`peak_reserved_bytes` below).
    """
    round_fn = _round_exact if policy == "exact" else _round_size_class

    free_cache: "dict[int, int]" = {}  # rounded size -> count of cached free blocks
    active_bytes = 0
    reserved_bytes = 0  # bytes ever obtained from a simulated cudaMalloc, never released
    peak_active_bytes = 0
    peak_reserved_bytes = 0
    simulated_malloc_count = 0
    cache_hits = 0
    cache_misses = 0
    internal_fragmentation_bytes = 0

    for e in events:
        if e.kind == "alloc":
            rounded = round_fn(e.nbytes)
            if free_cache.get(rounded, 0) > 0:
                free_cache[rounded] -= 1
                cache_hits += 1
            else:
                simulated_malloc_count += 1
                reserved_bytes += rounded
                peak_reserved_bytes = max(peak_reserved_bytes, reserved_bytes)
                cache_misses += 1
            active_bytes += rounded
            internal_fragmentation_bytes += rounded - e.nbytes
            peak_active_bytes = max(peak_active_bytes, active_bytes)
        elif e.kind == "free":
            rounded = round_fn(e.nbytes)
            active_bytes -= rounded
            free_cache[rounded] = free_cache.get(rounded, 0) + 1

    total_requests = cache_hits + cache_misses
    cached_bytes = sum(size * count for size, count in free_cache.items())

    return {
        "policy": policy,
        "total_alloc_requests": total_requests,
        "simulated_malloc_count": simulated_malloc_count,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": (cache_hits / total_requests) if total_requests else 0.0,
        "reserved_bytes_final": reserved_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_active_bytes": peak_active_bytes,
        "cached_bytes_final": cached_bytes,
        "internal_fragmentation_bytes": internal_fragmentation_bytes,
    }


def reuse_opportunity(events: "tuple[AllocationEvent, ...]") -> dict:
    """What fraction of allocated *bytes* could theoretically be served by exact-size reuse.

    This is the milestone's Section 12 "reuse analysis": distinct from
    `simulate_caching_allocator`, which models the moment-by-moment allocator
    state, this just asks "of all allocation requests after the first
    occurrence of their size, how many bytes do they represent?" -- a purely
    descriptive statistic, not a claim about achievable allocator speedup
    (see the milestone brief's explicit caution against overclaiming this).
    """
    seen_sizes: set = set()
    total_bytes = 0
    reusable_bytes = 0
    total_count = 0
    reusable_count = 0
    for e in events:
        if e.kind != "alloc":
            continue
        total_bytes += e.nbytes
        total_count += 1
        if e.nbytes in seen_sizes:
            reusable_bytes += e.nbytes
            reusable_count += 1
        seen_sizes.add(e.nbytes)
    return {
        "total_bytes": total_bytes,
        "total_count": total_count,
        "reusable_bytes": reusable_bytes,
        "reusable_count": reusable_count,
        "reusable_byte_fraction": (reusable_bytes / total_bytes) if total_bytes else 0.0,
        "reusable_count_fraction": (reusable_count / total_count) if total_count else 0.0,
        "distinct_sizes": len(seen_sizes),
    }


__all__ = [
    "SIZE_BUCKETS",
    "LIFETIME_BUCKETS",
    "size_distribution",
    "pair_lifetimes",
    "lifetime_distribution",
    "persistent_vs_temporary",
    "simulate_caching_allocator",
    "reuse_opportunity",
]
