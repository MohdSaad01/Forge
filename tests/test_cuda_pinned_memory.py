"""Milestone 29 tests: real CUDA pinned (page-locked) host memory.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark`, matching every other
`test_cuda_*.py` file. See `forge/backend/cuda/pinned.py`'s module docstring
for the design these tests verify.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import forge
from forge.backend.cuda import pinned as cuda_pinned
from forge.backend.cuda.backend import is_cuda_available
from forge.exceptions import CUDAError

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    """Return device memory to the driver between tests -- the 940MX has only 2 GB VRAM.

    Pinned host allocations are always freed directly by each test (never
    cached by Forge -- see `pinned.py`'s module docstring), so this only
    needs to purge the *device*-memory caching allocator, matching `tests/
    test_cuda_stream_allocator.py`'s identical fixture.
    """
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


# -- 1. Allocation / free lifecycle ---------------------------------------------


def test_allocation_reports_requested_size_and_is_not_freed():
    mem = forge.cuda.PinnedMemory(4096)
    assert mem.nbytes == 4096
    assert mem.is_freed is False
    mem.free()
    assert mem.is_freed is True


def test_free_is_idempotent():
    mem = forge.cuda.PinnedMemory(1024)
    mem.free()
    mem.free()  # must not raise or double-free
    assert mem.is_freed is True


def test_del_frees_without_error():
    mem = forge.cuda.PinnedMemory(1024)
    del mem  # __del__ runs synchronously here (plain refcounting, no gc.collect needed)


# -- 2. NumPy interoperability ---------------------------------------------------


def test_numpy_view_default_is_flat_uint8():
    mem = forge.cuda.PinnedMemory(256)
    array = mem.numpy()
    assert array.shape == (256,)
    assert array.dtype == np.uint8


def test_numpy_view_reshaped_and_retyped_round_trips_writes():
    n = 1024
    mem = forge.cuda.PinnedMemory(n * 4)
    array = mem.numpy(shape=(n,), dtype=np.float32)
    assert array.shape == (n,)
    assert array.dtype == np.float32

    values = np.arange(n, dtype=np.float32)
    array[:] = values
    # A second view over the same buffer must see the same writes -- both
    # are zero-copy windows onto identical page-locked memory.
    array2 = mem.numpy(shape=(n,), dtype=np.float32)
    np.testing.assert_array_equal(array2, values)


def test_numpy_view_requesting_more_bytes_than_allocated_raises_value_error():
    mem = forge.cuda.PinnedMemory(16)
    with pytest.raises(ValueError):
        mem.numpy(shape=(1024,), dtype=np.float32)


def test_numpy_view_after_free_raises_cuda_error():
    mem = forge.cuda.PinnedMemory(64)
    mem.free()
    with pytest.raises(CUDAError):
        mem.numpy()


def test_numpy_view_keeps_owning_pinned_memory_alive():
    """Dropping the `PinnedMemory` reference while a view survives must not free the buffer."""
    mem = forge.cuda.PinnedMemory(4096)
    array = mem.numpy(shape=(1024,), dtype=np.float32)
    array[:] = 7.0
    del mem  # only `array` (via `_pinned_owner`) keeps the allocation alive now
    np.testing.assert_allclose(array, np.full((1024,), 7.0, dtype=np.float32))


# -- 3. Statistics (Section 7) ---------------------------------------------------


def test_pinned_memory_stats_reflect_allocation_and_free():
    before = forge.cuda.pinned_memory_stats()
    mem = forge.cuda.PinnedMemory(8192)
    during = forge.cuda.pinned_memory_stats()
    assert during.pinned_active_bytes == before.pinned_active_bytes + 8192
    assert during.pinned_allocation_count == before.pinned_allocation_count + 1
    assert during.pinned_peak_bytes >= during.pinned_active_bytes

    mem.free()
    after = forge.cuda.pinned_memory_stats()
    assert after.pinned_active_bytes == before.pinned_active_bytes
    assert after.pinned_free_count == before.pinned_free_count + 1
    # Peak never decreases on a free -- it is a historical high-water mark.
    assert after.pinned_peak_bytes >= during.pinned_peak_bytes


def test_pinned_memory_stats_as_dict_has_expected_keys():
    stats = forge.cuda.pinned_memory_stats()
    assert set(stats.as_dict()) == {
        "pinned_active_bytes",
        "pinned_peak_bytes",
        "pinned_allocation_count",
        "pinned_free_count",
    }


# -- 4. Failure handling (Section 8) ---------------------------------------------


def test_absurdly_large_pinned_allocation_raises_cuda_error_and_touches_no_counters():
    """Run in a subprocess -- deliberately, not merely for hygiene.

    Verified directly on the 940MX (driver 582.53): an out-of-memory
    `cudaHostAlloc` request (even a "merely" 64 GiB one, not just `1 << 62`)
    leaves this process's CUDA context unable to serve small `cudaMalloc`
    requests afterward -- confirmed by reproducing
    `test_cuda_stream_allocator.py::
    test_same_stream_rapid_release_and_reallocate_never_corrupts_data`
    failing with `code 2` (`out of memory`) on an 8 KB device allocation
    whenever this test ran earlier in the same process. This is a real
    driver/WDDM quirk, not a Forge bug (Forge's own accounting is unaffected
    -- the assertions below still pass) -- isolating the real, hardware-
    verified failure call in a throwaway subprocess keeps that quirk from
    poisoning every other CUDA test sharing this process's context, while
    still exercising the genuine `cudaHostAlloc` failure path on real
    hardware (never simulated).
    """
    script = (
        "import forge\n"
        "from forge.exceptions import CUDAError\n"
        "before = forge.cuda.pinned_memory_stats()\n"
        "try:\n"
        "    forge.cuda.PinnedMemory(1 << 62)\n"
        "except CUDAError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('expected CUDAError, none was raised')\n"
        "after = forge.cuda.pinned_memory_stats()\n"
        "assert after.pinned_active_bytes == before.pinned_active_bytes\n"
        "assert after.pinned_allocation_count == before.pinned_allocation_count\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_negative_size_raises_value_error():
    with pytest.raises(ValueError):
        forge.cuda.PinnedMemory(-1)


# -- 5. Lifetime under an in-flight async transfer (Section 26) -----------------


def test_free_while_transfer_pending_waits_rather_than_corrupting(monkeypatch):
    """Marking a pending event and freeing must wait for it rather than racing it.

    Uses the real internal `_mark_pending`/`free()` machinery directly (the
    same one `CUDABackend.from_array_async`/`to_numpy_async` drive) without
    needing a full end-to-end transfer, to isolate exactly the lifetime
    contract this module promises (Invariant 1).
    """
    from forge.backend.cuda import stream as cuda_stream
    from forge.backend.cuda.backend import get_cuda_backend

    backend = get_cuda_backend()
    mem = cuda_pinned.PinnedMemory(4096, backend._lib)
    event = cuda_stream.CUDAEvent(backend._lib)
    event.record(None)  # default stream -- completes almost immediately
    mem._mark_pending(event)

    mem.free()  # must wait for `event`, not free while (potentially) still pending
    assert mem.is_freed is True


# -- 6. Leak testing (Section 52): repeated alloc/free returns to baseline ------


def test_repeated_alloc_free_does_not_grow_active_bytes():
    baseline = forge.cuda.pinned_memory_stats().pinned_active_bytes
    for _ in range(50):
        mem = forge.cuda.PinnedMemory(65536)
        mem.numpy(shape=(16384,), dtype=np.float32)[:] = 1.0
        mem.free()
    after = forge.cuda.pinned_memory_stats().pinned_active_bytes
    assert after == baseline


def test_repeated_alloc_then_drop_reference_does_not_grow_active_bytes():
    """Same as above but relying on `__del__` (plain refcounting) instead of explicit `.free()`."""
    baseline = forge.cuda.pinned_memory_stats().pinned_active_bytes
    for _ in range(50):
        mem = forge.cuda.PinnedMemory(65536)
        del mem
    after = forge.cuda.pinned_memory_stats().pinned_active_bytes
    assert after == baseline
