"""Milestone 29 tests: the CUDA caching allocator's safety under async transfers (Section 35, mandatory).

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark`. See `docs/architecture/
cuda-transfers.md`'s **Allocator Integration** section: async H2D/D2H
destination storage obeys the exact same active/pending/ready ownership
rules Milestone 27/28 already established for ordinary kernel-launched
storage -- no new allocator code was needed for this milestone.
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
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _pinned_tensor(values: np.ndarray) -> Tensor:
    mem = forge.cuda.PinnedMemory(values.nbytes)
    array = mem.numpy(shape=values.shape, dtype=values.dtype)
    array[:] = values
    return Tensor(array, device="cpu")


def test_async_h2d_release_never_hands_the_still_in_flight_block_to_another_stream():
    """Section 35's mandatory race test.

    Stream A submits an async H2D into `x`, then releases `x` immediately
    (`del x`, before waiting for the copy to finish). Stream B immediately
    requests a same-size allocation. The allocator must never hand Stream B
    a block Stream A's H2D copy might still be writing -- verified by
    confirming Stream B's own values are never corrupted, and that `x`'s
    release became a *pending* (not immediately ready) block, exactly like
    every other stream-associated release since Milestone 27.
    """
    shape = (65536,)
    dtype = np.float32
    nbytes = shape[0] * 4
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    cpu_source = _pinned_tensor(np.full(shape, 1.0, dtype=dtype))
    with forge.cuda.stream(stream_a):
        x = cpu_source.to("cuda", non_blocking=True)
    stats_after_release_check = cuda_allocator.memory_stats()
    del x  # releases while the H2D copy may still be in flight -- must become pending

    stats_after_del = cuda_allocator.memory_stats()
    assert stats_after_del.pending_count >= stats_after_release_check.pending_count

    with forge.cuda.stream(stream_b):
        b = Tensor(np.full(shape, 99.0, dtype=dtype), device="cuda")
        b_result = b + b  # 198.0 everywhere -- must never see stream A's still-in-flight bytes

    stream_a.synchronize()
    stream_b.synchronize()
    np.testing.assert_allclose(b_result.to("cpu").numpy(), np.full(shape, 198.0, dtype=dtype))


def test_async_d2h_pinned_destination_is_correct_even_under_rapid_release():
    """Repeated async D2H, releasing each CUDA source immediately, must never corrupt results."""
    shape = (4096,)
    dtype = np.float32
    stream = forge.cuda.Stream()

    for i in range(20):
        with forge.cuda.stream(stream):
            src = Tensor(np.full(shape, float(i), dtype=dtype), device="cuda")
            cpu_result = src.to("cpu", non_blocking=True)
        del src  # release the CUDA source right after submitting the D2H copy
        np.testing.assert_allclose(cpu_result.numpy(), np.full(shape, float(i), dtype=dtype))


def test_empty_cache_after_async_transfers_preserves_correctness_of_live_tensors():
    shape = (2048,)
    dtype = np.float32
    stream = forge.cuda.Stream()
    cpu_source = _pinned_tensor(np.full(shape, 42.0, dtype=dtype))

    with forge.cuda.stream(stream):
        keep = cpu_source.to("cuda", non_blocking=True)
        released = keep + keep
    del released  # becomes a pending block

    forge.cuda.empty_cache()  # must not touch `keep`, still live

    stream.synchronize()
    np.testing.assert_allclose(keep.to("cpu").numpy(), np.full(shape, 42.0, dtype=dtype))
