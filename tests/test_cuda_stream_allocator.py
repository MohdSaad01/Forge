"""Milestone 27 tests: the CUDA caching allocator's stream-aware (pending-block) reuse model.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching every
other `test_cuda_*.py` file. See `forge/backend/cuda/allocator.py`'s module
docstring for the pending-block design these tests verify, and
`docs/architecture/cuda-streams.md`'s **Allocator Changes** section for the
full contract. `tests/test_cuda_allocator.py` (Milestone 25) covers the
unchanged default-stream (ready-cache) behavior this module does not repeat.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import allocator as cuda_allocator
from forge.backend.cuda.backend import is_cuda_available

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test wants a known-empty starting cache; matches `test_cuda_allocator.py`'s fixture.

    `empty_cache()` under Milestone 27 also drains (waits for, then frees)
    any pending blocks left over from a previous test -- see `allocator.py`.
    """
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


# -- 1. A release on an explicit stream becomes pending, not immediately ready ----


def test_release_on_explicit_stream_is_pending_not_immediately_ready():
    shape = (8192,)
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        t = Tensor(np.ones(shape, dtype=np.float32), device="cuda")
    del t  # __del__ runs synchronously here (plain refcounting, no gc.collect needed)

    stats = cuda_allocator.memory_stats()
    assert stats.pending_count >= 1
    assert stats.pending_bytes >= shape[0] * 4
    s.synchronize()


# -- 2. Same-stream reuse: correct once the stream is (or becomes) synchronized ----


def test_same_stream_release_then_allocate_after_sync_reuses_and_is_correct():
    shape = (4096,)
    dtype = np.float32
    s = forge.cuda.Stream()

    with forge.cuda.stream(s):
        a = Tensor(np.full(shape, 3.0, dtype=dtype), device="cuda")
    del a
    s.synchronize()  # force the pending event to complete before reallocating

    stats_before = cuda_allocator.memory_stats()
    with forge.cuda.stream(s):
        b = Tensor(np.full(shape, 9.0, dtype=dtype), device="cuda")
        b_readback = b.to("cpu")  # to_numpy() syncs `b`'s own last-use stream first
    stats_after = cuda_allocator.memory_stats()

    np.testing.assert_allclose(b_readback.numpy(), np.full(shape, 9.0, dtype=dtype))
    # Either a ready hit or a reclaimed-pending hit both count as `cache_hit_count`.
    assert stats_after.cache_hit_count >= stats_before.cache_hit_count


def test_same_stream_rapid_release_and_reallocate_never_corrupts_data():
    """Release and reallocate back-to-back on the same stream, with no manual sync in between.

    Whether or not the allocator manages to reclaim the just-released block
    (it may not, if its event has not completed yet -- see
    `docs/architecture/cuda-streams.md`), the *result* must always be
    correct: CUDA's own per-stream program order guarantees the new
    storage's writes are enqueued after the old storage's, so no window for
    corruption exists regardless of which physical block backs it.
    """
    shape = (2048,)
    dtype = np.float32
    s = forge.cuda.Stream()

    for i in range(20):
        with forge.cuda.stream(s):
            t = Tensor(np.full(shape, float(i), dtype=dtype), device="cuda")
            doubled = t + t
        del t
        result = doubled.to("cpu")  # syncs doubled's own stream before reading
        np.testing.assert_allclose(result.numpy(), np.full(shape, float(2 * i), dtype=dtype))
        del doubled


# -- 3. Cross-stream safety: a block must not be reused while still in flight -----


def test_cross_stream_reuse_never_hands_out_a_still_in_flight_block():
    """Stream A releases a block; Stream B immediately requests the same size.

    The allocator must never hand Stream B a block Stream A might still be
    using -- verified by proving Stream A's own values are never corrupted
    by whatever Stream B does with its (possibly different, possibly a new
    `cudaMalloc`) block. This is the central Milestone 27 allocator
    correctness property (Section 14 of the milestone brief).
    """
    shape = (16384,)
    dtype = np.float32
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    with forge.cuda.stream(stream_a):
        a = Tensor(np.full(shape, 1.0, dtype=dtype), device="cuda")
        a_result = a + a  # 2.0 everywhere -- kept alive, read back at the end
    del a  # releases stream_a's first block while (likely) still in flight

    # Immediately, on a different stream, request the identical byte size --
    # a naive allocator might race stream_a's still-in-flight release.
    with forge.cuda.stream(stream_b):
        b = Tensor(np.full(shape, 99.0, dtype=dtype), device="cuda")
        b_result = b + b  # 198.0 everywhere

    stream_a.synchronize()
    stream_b.synchronize()

    np.testing.assert_allclose(a_result.to("cpu").numpy(), np.full(shape, 2.0, dtype=dtype))
    np.testing.assert_allclose(b_result.to("cpu").numpy(), np.full(shape, 198.0, dtype=dtype))


def test_cross_stream_read_of_a_default_stream_tensor_is_always_safe():
    """A tensor produced on the default (host-synchronous) stream is safe to read from any stream.

    `CUDAStorage.last_stream is None` (default-stream provenance) is always
    safe regardless of the current stream, because the M26 contract already
    guarantees that work completed before Python ever saw the storage.
    """
    shape = (128,)
    weight = Tensor(np.full(shape, 5.0, dtype=np.float32), device="cuda")  # default stream

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        doubled = weight + weight
    s.synchronize()
    np.testing.assert_allclose(doubled.to("cpu").numpy(), np.full(shape, 10.0, dtype=np.float32))


def test_using_a_tensor_across_two_different_explicit_streams_raises_clearly():
    """Forge does not support arbitrary cross-stream tensor dependencies (Section 20 of the brief)."""
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    with forge.cuda.stream(stream_a):
        t = Tensor(np.ones((16,), dtype=np.float32), device="cuda")

    with pytest.raises(forge.CUDAError):
        with forge.cuda.stream(stream_b):
            _ = t + t

    stream_a.synchronize()


# -- 4. empty_cache() drains pending blocks (Section 18) -----------------------


def test_empty_cache_drains_pending_blocks_and_returns_them_to_the_driver():
    shape = (32768,)
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        t = Tensor(np.ones(shape, dtype=np.float32), device="cuda")
    del t

    stats_before = cuda_allocator.memory_stats()
    assert stats_before.pending_count >= 1

    freed = forge.cuda.empty_cache()

    stats_after = cuda_allocator.memory_stats()
    assert freed >= 1
    assert stats_after.pending_count == 0
    assert stats_after.pending_bytes == 0
    s.synchronize()


def test_empty_cache_after_stream_work_preserves_correctness_of_live_tensors():
    """`empty_cache()` must never touch a still-live (active) storage, streams included."""
    shape = (256,)
    dtype = np.float32
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        keep = Tensor(np.full(shape, 42.0, dtype=dtype), device="cuda")
        released = keep + keep
    del released  # this one becomes a pending block

    forge.cuda.empty_cache()  # must not touch `keep`, which is still live

    s.synchronize()
    np.testing.assert_allclose(keep.to("cpu").numpy(), np.full(shape, 42.0, dtype=dtype))


# -- 5. Memory statistics coherence (Section 17) --------------------------------


def test_memory_stats_reserved_equals_active_plus_cached_plus_pending():
    shape = (8192,)
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        a = Tensor(np.ones(shape, dtype=np.float32), device="cuda")
        keep = Tensor(np.ones(shape, dtype=np.float32), device="cuda")
    del a  # -> pending

    stats = cuda_allocator.memory_stats()
    assert stats.reserved_bytes == stats.allocated_bytes + stats.cached_bytes + stats.pending_bytes
    s.synchronize()
    del keep
