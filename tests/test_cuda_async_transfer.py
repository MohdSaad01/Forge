"""Milestone 29 tests: `Tensor.to(..., non_blocking=True)` -- async H2D/D2H transfer semantics.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark`, matching every other
`test_cuda_*.py` file. See `docs/architecture/cuda-transfers.md` for the
full contract these tests verify.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available
from forge.exceptions import CUDAError, UnsupportedDeviceError

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    """Return device memory to the driver between tests -- see `test_cuda_stream_allocator.py`."""
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _pinned_tensor(values: np.ndarray) -> Tensor:
    mem = forge.cuda.PinnedMemory(values.nbytes)
    array = mem.numpy(shape=values.shape, dtype=values.dtype)
    array[:] = values
    return Tensor(array, device="cpu")


# -- 1. H2D: requires pinned memory (Section 13, Option A) ----------------------


def test_h2d_nonblocking_with_pinned_source_transfers_correct_values():
    values = np.arange(4096, dtype=np.float32)
    cpu_t = _pinned_tensor(values)
    cuda_t = cpu_t.to("cuda", non_blocking=True)
    forge.cuda.synchronize()
    np.testing.assert_allclose(cuda_t.to("cpu").numpy(), values)


def test_h2d_nonblocking_with_pageable_source_raises_cuda_error():
    cpu_t = Tensor(np.arange(1024, dtype=np.float32), device="cpu")  # ordinary, pageable memory
    with pytest.raises(CUDAError):
        cpu_t.to("cuda", non_blocking=True)


def test_h2d_nonblocking_does_not_call_cuda_device_synchronize(monkeypatch):
    backend = get_cuda_backend()
    calls = {"n": 0}
    original = backend._lib.cf_synchronize

    def spy():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(backend._lib, "cf_synchronize", spy)

    cpu_t = _pinned_tensor(np.ones(65536, dtype=np.float32))
    cuda_t = cpu_t.to("cuda", non_blocking=True)

    assert calls["n"] == 0
    forge.cuda.synchronize()
    np.testing.assert_allclose(cuda_t.to("cpu").numpy(), np.ones(65536, dtype=np.float32))


def test_h2d_nonblocking_on_explicit_stream_records_it_as_last_stream():
    s = forge.cuda.Stream()
    cpu_t = _pinned_tensor(np.full((256,), 3.0, dtype=np.float32))
    with forge.cuda.stream(s):
        cuda_t = cpu_t.to("cuda", non_blocking=True)
    assert cuda_t._data.last_stream is s
    s.synchronize()


# -- 2. D2H: pending Tensor, host-read synchronization (Section 18/19) ----------


def test_d2h_nonblocking_numpy_read_is_correct():
    values = np.arange(8192, dtype=np.float32).reshape(64, 128)
    cuda_t = Tensor(values, device="cuda")
    cpu_t = cuda_t.to("cpu", non_blocking=True)
    np.testing.assert_allclose(cpu_t.numpy(), values)


def test_d2h_nonblocking_result_has_pending_marker_before_first_read():
    cuda_t = Tensor(np.ones((1024,), dtype=np.float32), device="cuda")
    cpu_t = cuda_t.to("cpu", non_blocking=True)
    assert cpu_t._pending is not None  # not yet synchronized
    _ = cpu_t.numpy()  # first host-boundary read
    assert cpu_t._pending is None  # synchronized and cleared


def test_d2h_nonblocking_repr_does_not_observe_incomplete_data():
    values = np.full((4096,), 5.0, dtype=np.float32)
    cuda_t = Tensor(values, device="cuda")
    cpu_t = cuda_t.to("cpu", non_blocking=True)
    text = repr(cpu_t)  # must synchronize before formatting -- never garbage/incomplete
    assert "5." in text


def test_d2h_nonblocking_used_in_further_cpu_arithmetic_is_correct():
    values = np.arange(2048, dtype=np.float32)
    cuda_t = Tensor(values, device="cuda")
    cpu_t = cuda_t.to("cpu", non_blocking=True)
    result = cpu_t + Tensor(np.ones((2048,), dtype=np.float32), device="cpu")
    np.testing.assert_allclose(result.numpy(), values + 1.0)


def test_d2h_nonblocking_large_tensor_matches_synchronous_reference():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((512, 512)).astype(np.float32)
    cuda_t = Tensor(values, device="cuda")
    async_result = cuda_t.to("cpu", non_blocking=True).numpy()
    sync_result = cuda_t.to("cpu").numpy()
    np.testing.assert_array_equal(async_result, sync_result)


def test_d2h_nonblocking_odd_shapes_and_float64():
    rng = np.random.default_rng(1)
    for shape, dtype in [((), np.float32), ((1,), np.float64), ((3, 1, 7), np.float32), ((17,), np.float64)]:
        values = rng.standard_normal(shape).astype(dtype)
        cuda_t = Tensor(values, device="cuda")
        cpu_t = cuda_t.to("cpu", non_blocking=True)
        np.testing.assert_allclose(cpu_t.numpy(), values)


def test_d2h_nonblocking_detaches_from_pinned_memory_after_sync():
    """After the first host read, the tensor's storage must no longer reference pinned memory.

    Prevents an unbounded-retention hazard: every array derived from a
    `_PinnedArray` (via ordinary NumPy ufuncs) would otherwise also carry a
    `_pinned_owner` back-reference, keeping a potentially large pinned
    buffer alive indefinitely (see `Tensor._data`'s property docstring).
    """
    cuda_t = Tensor(np.ones((256,), dtype=np.float32), device="cuda")
    cpu_t = cuda_t.to("cpu", non_blocking=True)
    array = cpu_t.numpy()
    assert getattr(array, "_pinned_owner", None) is None


# -- 3. Nonblocking direction policy (Section 12) --------------------------------


def test_nonblocking_unsupported_direction_raises_unsupported_device_error():
    cuda_t = Tensor(np.ones((8,), dtype=np.float32), device="cuda")
    with pytest.raises(UnsupportedDeviceError):
        cuda_t.to("cuda:0", non_blocking=True)


# -- 4. non_blocking=False remains exactly the pre-existing synchronous contract --


def test_default_to_remains_synchronous_and_unaffected():
    values = np.arange(1024, dtype=np.float32)
    cpu_t = Tensor(values, device="cpu")
    cuda_t = cpu_t.to("cuda")  # non_blocking defaults to False
    back = cuda_t.to("cpu")
    np.testing.assert_allclose(back.numpy(), values)
    assert back._pending is None
