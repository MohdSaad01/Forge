"""Milestone 29 stress/leak tests: repeated async H2D -> compute -> D2H cycles across multiple streams.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark`. Follows the M23/M28 leak-testing
methodology: measure Forge-owned counters before and after a repeated
workload and require them to return to baseline -- see Section 51/52 of the
milestone brief.
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


def test_repeated_multi_stream_h2d_compute_d2h_cycles_leave_no_residue():
    """2-4 streams, each doing: pinned H2D -> compute -> pinned D2H -> host sync -> release, repeated."""
    shape = (8192,)
    dtype = np.float32
    streams = [forge.cuda.Stream() for _ in range(4)]
    iterations = 30

    active_before = cuda_allocator.memory_stats().allocated_bytes
    pinned_before = forge.cuda.pinned_memory_stats().pinned_active_bytes

    for i in range(iterations):
        stream = streams[i % len(streams)]
        mem = forge.cuda.PinnedMemory(shape[0] * 4)
        host_array = mem.numpy(shape=shape, dtype=dtype)
        host_array[:] = float(i)
        cpu_source = Tensor(host_array, device="cpu")

        with forge.cuda.stream(stream):
            x = cpu_source.to("cuda", non_blocking=True)
            y = x + x
            result = y.to("cpu", non_blocking=True)

        stream.synchronize()
        np.testing.assert_allclose(result.numpy(), np.full(shape, float(2 * i), dtype=dtype))
        del x, y, result, cpu_source, host_array, mem

    for s in streams:
        s.synchronize()
    forge.cuda.empty_cache()

    active_after = cuda_allocator.memory_stats().allocated_bytes
    pinned_after = forge.cuda.pinned_memory_stats().pinned_active_bytes
    assert active_after == active_before
    assert pinned_after == pinned_before


def test_repeated_h2d_only_stress_does_not_grow_pinned_or_device_memory():
    shape = (16384,)
    dtype = np.float32
    stream = forge.cuda.Stream()

    pinned_before = forge.cuda.pinned_memory_stats().pinned_active_bytes
    device_before = cuda_allocator.memory_stats().allocated_bytes

    for i in range(40):
        mem = forge.cuda.PinnedMemory(shape[0] * 4)
        array = mem.numpy(shape=shape, dtype=dtype)
        array[:] = float(i)
        with forge.cuda.stream(stream):
            x = Tensor(array, device="cpu").to("cuda", non_blocking=True)
        del x, array, mem

    stream.synchronize()
    forge.cuda.empty_cache()

    assert forge.cuda.pinned_memory_stats().pinned_active_bytes == pinned_before
    assert cuda_allocator.memory_stats().allocated_bytes == device_before
