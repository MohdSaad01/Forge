"""Tests for `benchmarks/alloc_analysis.py` (Milestone 24).

Pure-function tests over hand-built `AllocationEvent` tuples -- no CUDA
hardware required (constructing an `AllocationEvent` needs no CUDA backend,
per `forge/backend/cuda/profiler.py`'s pure-Python design), matching
`docs/development/development-environment.md`'s CPU-testable requirement.
"""

from __future__ import annotations

import pytest

from forge.backend.cuda.profiler import AllocationEvent
from benchmarks import alloc_analysis as aa


def _event(kind, nbytes, t, block_id, category=None):
    return AllocationEvent(kind=kind, nbytes=nbytes, timestamp=t, block_id=block_id, category=category)


# -- size_distribution ------------------------------------------------------------


def test_size_distribution_empty_trace():
    dist = aa.size_distribution(())
    assert dist["count"] == 0
    assert dist["total_bytes"] == 0
    assert dist["common_sizes"] == []


def test_size_distribution_basic_stats():
    events = (
        _event("alloc", 100, 0.0, 1),
        _event("alloc", 200, 0.1, 2),
        _event("alloc", 100, 0.2, 3),
        _event("free", 100, 0.3, 1),
    )
    dist = aa.size_distribution(events, kind="alloc")
    assert dist["count"] == 3
    assert dist["total_bytes"] == 400
    assert dist["min_bytes"] == 100
    assert dist["max_bytes"] == 200
    assert dist["mean_bytes"] == pytest.approx(400 / 3)
    assert dist["common_sizes"][0] == {"nbytes": 100, "count": 2}


def test_size_distribution_buckets_by_size():
    events = (_event("alloc", 500, 0.0, 1), _event("alloc", 2000, 0.0, 2), _event("alloc", 2_000_000, 0.0, 3))
    dist = aa.size_distribution(events)
    assert dist["bucket_counts"]["< 1 KB"] == 1
    assert dist["bucket_counts"]["1-4 KB"] == 1
    assert dist["bucket_counts"]["1-4 MB"] == 1


def test_size_distribution_free_kind_is_independent_of_alloc():
    events = (_event("alloc", 100, 0.0, 1), _event("free", 100, 0.1, 1), _event("free", 200, 0.2, 2))
    alloc_dist = aa.size_distribution(events, kind="alloc")
    free_dist = aa.size_distribution(events, kind="free")
    assert alloc_dist["count"] == 1
    assert free_dist["count"] == 2
    assert free_dist["total_bytes"] == 300


# -- pair_lifetimes / lifetime_distribution -----------------------------------------


def test_pair_lifetimes_matches_alloc_to_next_free_of_same_block_id():
    events = (
        _event("alloc", 100, 0.0, 1),
        _event("alloc", 100, 0.1, 2),
        _event("free", 100, 0.5, 1),
        _event("free", 100, 0.6, 2),
    )
    lifetimes, still_live = aa.pair_lifetimes(events)
    assert sorted(round(x, 6) for x in lifetimes) == [0.5, 0.5]
    assert still_live == 0


def test_pair_lifetimes_handles_block_id_reuse_in_chronological_order():
    """The same address reused by a second allocation after its first free --
    pairing must be FIFO per block_id, not "any alloc with any free"."""
    events = (
        _event("alloc", 100, 0.0, 1),   # first occupant of block 1
        _event("free", 100, 0.1, 1),    # lifetime 0.1
        _event("alloc", 100, 0.2, 1),   # second occupant reuses the same address
        _event("free", 100, 0.5, 1),    # lifetime 0.3
    )
    lifetimes, still_live = aa.pair_lifetimes(events)
    assert [round(x, 6) for x in lifetimes] == [0.1, 0.3]
    assert still_live == 0


def test_pair_lifetimes_reports_unfreed_allocations_as_still_live():
    events = (_event("alloc", 100, 0.0, 1), _event("alloc", 100, 0.1, 2), _event("free", 100, 0.2, 1))
    lifetimes, still_live = aa.pair_lifetimes(events)
    assert len(lifetimes) == 1
    assert still_live == 1


def test_lifetime_distribution_buckets():
    events = (
        _event("alloc", 10, 0.0, 1), _event("free", 10, 0.0005, 1),      # < 1ms
        _event("alloc", 10, 1.0, 2), _event("free", 10, 1.005, 2),        # 1-10ms
        _event("alloc", 10, 2.0, 3), _event("free", 10, 5.0, 3),          # 1-10s
    )
    dist = aa.lifetime_distribution(events)
    assert dist["bucket_counts"]["< 1 ms"] == 1
    assert dist["bucket_counts"]["1-10 ms"] == 1
    assert dist["bucket_counts"]["1-10 s"] == 1
    assert dist["paired_count"] == 3
    assert dist["still_live_count"] == 0


# -- persistent_vs_temporary -------------------------------------------------------


def test_persistent_vs_temporary_splits_by_whether_freed_in_trace():
    events = (
        _event("alloc", 1000, 0.0, 1),   # never freed -> persistent
        _event("alloc", 100, 0.1, 2),
        _event("free", 100, 0.2, 2),      # freed -> temporary
    )
    split = aa.persistent_vs_temporary(events)
    assert split["persistent_bytes"] == 1000
    assert split["persistent_count"] == 1
    assert split["temporary_bytes"] == 100
    assert split["temporary_count"] == 1
    assert split["total_alloc_bytes"] == 1100


# -- simulate_caching_allocator ------------------------------------------------------


def test_exact_policy_reuses_only_identical_sizes():
    events = (
        _event("alloc", 100, 0.0, 1),
        _event("free", 100, 0.1, 1),
        _event("alloc", 100, 0.2, 2),   # exact-size reuse of the freed block
        _event("alloc", 200, 0.3, 3),   # different size -> new simulated malloc
    )
    sim = aa.simulate_caching_allocator(events, policy="exact")
    assert sim["cache_hits"] == 1
    assert sim["cache_misses"] == 2
    assert sim["simulated_malloc_count"] == 2
    assert sim["internal_fragmentation_bytes"] == 0


def test_size_class_policy_rounds_up_and_can_still_produce_fragmentation():
    events = (
        _event("alloc", 500, 0.0, 1),
        _event("free", 500, 0.1, 1),
        _event("alloc", 900, 0.2, 2),  # different exact size, same "< 1 KB" bucket -> hit under size_class
    )
    sim_exact = aa.simulate_caching_allocator(events, policy="exact")
    sim_class = aa.simulate_caching_allocator(events, policy="size_class")
    assert sim_exact["cache_hits"] == 0  # 500 != 900, no exact reuse
    assert sim_class["cache_hits"] == 1  # both round up into the same bucket
    assert sim_class["internal_fragmentation_bytes"] > 0


def test_simulation_never_reports_more_hits_than_total_requests():
    events = tuple(_event("alloc", 64, float(i), i) for i in range(10)) + tuple(
        _event("free", 64, float(i) + 0.5, i) for i in range(10)
    )
    sim = aa.simulate_caching_allocator(events, policy="exact")
    assert sim["cache_hits"] + sim["cache_misses"] == sim["total_alloc_requests"]
    assert sim["total_alloc_requests"] == 10


# -- reuse_opportunity ------------------------------------------------------------


def test_reuse_opportunity_counts_bytes_after_first_occurrence_of_a_size():
    events = (
        _event("alloc", 100, 0.0, 1),
        _event("alloc", 100, 0.1, 2),  # repeat of a seen size -> reusable
        _event("alloc", 200, 0.2, 3),  # first occurrence -> not reusable
    )
    reuse = aa.reuse_opportunity(events)
    assert reuse["total_bytes"] == 400
    assert reuse["reusable_bytes"] == 100
    assert reuse["distinct_sizes"] == 2
    assert reuse["reusable_byte_fraction"] == 0.25
