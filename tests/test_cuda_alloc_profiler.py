"""Milestone 24 tests: CUDA allocation profiler (`forge.cuda.profiler`).

Every test in this module requires an actual working CUDA backend (see the
module-level `pytestmark`), matching the convention in `tests/test_cuda_memory.py`
and every other `test_cuda_*.py` file. The CUDA-unavailable error-path tests
live in `tests/test_cuda_alloc_profiler_availability.py` instead, for the
same reason `test_cuda_memory_availability.py` is split out from
`test_cuda_memory.py`: a module-level `pytestmark` skip applies to every test
in its module regardless of definition order.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_profiler():
    """Every test starts and ends with the profiler stopped and empty -- process-wide state."""
    forge.cuda.profiler.stop()
    forge.cuda.profiler.reset()
    yield
    forge.cuda.profiler.stop()
    forge.cuda.profiler.reset()


# -- 1. Disabled by default / start-stop ------------------------------------------


def test_profiler_is_inactive_by_default():
    assert forge.cuda.profiler.is_active() is False


def test_profiler_records_nothing_while_inactive():
    t = Tensor(np.zeros(100, dtype=np.float32), device="cuda")
    del t
    assert forge.cuda.profiler.events() == ()


def test_start_activates_and_stop_deactivates():
    forge.cuda.profiler.start()
    assert forge.cuda.profiler.is_active() is True
    forge.cuda.profiler.stop()
    assert forge.cuda.profiler.is_active() is False


# -- 2. Allocation / free events ---------------------------------------------------


def test_allocation_produces_an_alloc_event_with_correct_size():
    forge.cuda.profiler.start()
    t = Tensor(np.zeros(1000, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()
    events = [e for e in forge.cuda.profiler.events() if e.kind == "alloc"]
    assert len(events) == 1
    assert events[0].nbytes == 1000 * 4
    del t


def test_deletion_produces_a_free_event_with_correct_size():
    t = Tensor(np.zeros(500, dtype=np.float32), device="cuda")
    forge.cuda.profiler.start()
    del t
    forge.cuda.profiler.stop()
    events = [e for e in forge.cuda.profiler.events() if e.kind == "free"]
    assert len(events) == 1
    assert events[0].nbytes == 500 * 4


def test_alloc_and_free_events_share_the_same_block_id():
    forge.cuda.profiler.start()
    t = Tensor(np.zeros(200, dtype=np.float32), device="cuda")
    del t
    forge.cuda.profiler.stop()
    events = forge.cuda.profiler.events()
    assert len(events) == 2
    alloc_event, free_event = events
    assert alloc_event.kind == "alloc" and free_event.kind == "free"
    assert alloc_event.block_id == free_event.block_id
    assert alloc_event.block_id != 0


def test_multiple_allocations_produce_one_event_each():
    forge.cuda.profiler.start()
    tensors = [Tensor(np.zeros(n, dtype=np.float32), device="cuda") for n in (10, 20, 30)]
    forge.cuda.profiler.stop()
    alloc_events = [e for e in forge.cuda.profiler.events() if e.kind == "alloc"]
    assert sorted(e.nbytes for e in alloc_events) == [40, 80, 120]
    del tensors


# -- 3. Reset ------------------------------------------------------------------------


def test_reset_discards_events_without_changing_active_state():
    forge.cuda.profiler.start()
    t = Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    assert len(forge.cuda.profiler.events()) >= 1
    forge.cuda.profiler.reset()
    assert forge.cuda.profiler.events() == ()
    assert forge.cuda.profiler.is_active() is True
    del t


def test_stop_then_reset_then_start_gives_a_fresh_trace():
    forge.cuda.profiler.start()
    Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()
    assert len(forge.cuda.profiler.events()) >= 1

    forge.cuda.profiler.reset()
    forge.cuda.profiler.start()
    t2 = Tensor(np.zeros(20, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()
    events = forge.cuda.profiler.events()
    assert all(e.nbytes != 10 * 4 for e in events)
    assert any(e.nbytes == 20 * 4 for e in events)
    del t2


# -- 4. Tagging / categorization ---------------------------------------------------


def test_tag_labels_events_recorded_within_the_block():
    forge.cuda.profiler.start()
    with forge.cuda.profiler.tag("forward"):
        t1 = Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    t2 = Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()

    events = forge.cuda.profiler.events()
    categories = {e.category for e in events}
    assert "forward" in categories
    assert None in categories  # t2's allocation was untagged
    del t1, t2


def test_nested_tags_use_the_innermost_name():
    forge.cuda.profiler.start()
    with forge.cuda.profiler.tag("outer"):
        with forge.cuda.profiler.tag("inner"):
            t = Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()
    alloc_events = [e for e in forge.cuda.profiler.events() if e.kind == "alloc"]
    assert alloc_events[-1].category == "inner"
    del t


def test_tag_stack_restores_after_exiting_the_block():
    forge.cuda.profiler.start()
    with forge.cuda.profiler.tag("a"):
        pass
    t = Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()
    alloc_events = [e for e in forge.cuda.profiler.events() if e.kind == "alloc"]
    assert alloc_events[-1].category is None
    del t


# -- 5. Does not retain CUDAStorage / does not affect lifecycle -------------------------


def test_profiler_does_not_keep_storage_alive():
    """`CUDAStorage` has no `__weakref__` slot, so liveness is checked the same way
    `tests/test_cuda_memory.py` does -- via `memory_stats()`'s real allocated-byte
    count returning to baseline once every ordinary Python reference is gone. If the
    profiler secretly retained the `CUDAStorage`, `cudaFree` would never run and
    `allocated_bytes` would stay inflated."""
    before = forge.cuda.memory_stats().allocated_bytes
    forge.cuda.profiler.start()
    t = Tensor(np.zeros(10_000, dtype=np.float32), device="cuda")
    del t
    gc.collect()
    forge.cuda.profiler.stop()
    after = forge.cuda.memory_stats().allocated_bytes
    assert after == before

    events = forge.cuda.profiler.events()
    assert not any(hasattr(e, "storage") or hasattr(e, "tensor") for e in events)


def test_profiler_running_does_not_change_real_memory_stats():
    """Profiling is purely observational -- `forge.cuda.memory_stats()` must agree
    whether or not the profiler happens to be running."""
    before = forge.cuda.memory_stats()
    forge.cuda.profiler.start()
    t = Tensor(np.zeros(5000, dtype=np.float32), device="cuda")
    during = forge.cuda.memory_stats()
    del t
    gc.collect()
    forge.cuda.profiler.stop()
    after = forge.cuda.memory_stats()
    assert during.allocated_bytes - before.allocated_bytes == 5000 * 4
    assert after.allocated_bytes == before.allocated_bytes


# -- 6. Repeated traces can be analyzed ------------------------------------------------


def test_repeated_start_stop_cycles_each_produce_independent_traces():
    from benchmarks.alloc_analysis import size_distribution

    forge.cuda.profiler.reset()
    forge.cuda.profiler.start()
    t1 = Tensor(np.zeros(100, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()
    dist1 = size_distribution(forge.cuda.profiler.events())
    assert dist1["count"] == 1
    assert dist1["total_bytes"] == 400

    forge.cuda.profiler.reset()
    forge.cuda.profiler.start()
    t2 = Tensor(np.zeros(50, dtype=np.float32), device="cuda")
    t3 = Tensor(np.zeros(50, dtype=np.float32), device="cuda")
    forge.cuda.profiler.stop()
    dist2 = size_distribution(forge.cuda.profiler.events())
    assert dist2["count"] == 2
    assert dist2["total_bytes"] == 400
    del t1, t2, t3
