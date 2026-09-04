"""Milestone 29 tests: cross-stream dependencies for async H2D/D2H transfers.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark`. These tests verify that an
asynchronous transfer participates in the *existing* Milestone 28
`CUDABackend._stream_guard` dependency machinery with no new mechanism --
see `docs/architecture/cuda-transfers.md`'s **Cross-Stream Transfer
Dependencies** section.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available

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


# -- 1. H2D on stream A -> compute on stream B, with no explicit sync in between --


def test_h2d_on_one_stream_then_compute_on_another_needs_no_explicit_sync():
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    values = np.full((16384,), 3.0, dtype=np.float32)
    cpu_t = _pinned_tensor(values)

    with forge.cuda.stream(stream_a):
        x_cuda = cpu_t.to("cuda", non_blocking=True)

    with forge.cuda.stream(stream_b):
        y = x_cuda + x_cuda  # must correctly wait for the H2D transfer, GPU-side only

    stream_a.synchronize()
    stream_b.synchronize()
    np.testing.assert_allclose(y.to("cpu").numpy(), values * 2.0)


def test_h2d_then_compute_never_calls_cuda_device_synchronize(monkeypatch):
    backend = get_cuda_backend()
    calls = {"n": 0}
    original = backend._lib.cf_synchronize

    def spy():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(backend._lib, "cf_synchronize", spy)

    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    cpu_t = _pinned_tensor(np.ones((4096,), dtype=np.float32))

    with forge.cuda.stream(stream_a):
        x_cuda = cpu_t.to("cuda", non_blocking=True)
    with forge.cuda.stream(stream_b):
        y = x_cuda + x_cuda

    assert calls["n"] == 0
    stream_a.synchronize()
    stream_b.synchronize()
    np.testing.assert_allclose(y.to("cpu").numpy(), np.full((4096,), 2.0, dtype=np.float32))


# -- 2. compute on stream A -> D2H on stream B ------------------------------------


def test_compute_on_one_stream_then_d2h_on_another_needs_no_explicit_sync():
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    with forge.cuda.stream(stream_a):
        x = Tensor(np.full((8192,), 5.0, dtype=np.float32), device="cuda")
        y = x + x  # 10.0 everywhere, last_stream = stream_a

    with forge.cuda.stream(stream_b):
        cpu_result = y.to("cpu", non_blocking=True)  # D2H on stream_b, must wait for stream_a's write

    np.testing.assert_allclose(cpu_result.numpy(), np.full((8192,), 10.0, dtype=np.float32))


def test_compute_then_d2h_never_calls_cuda_device_synchronize(monkeypatch):
    backend = get_cuda_backend()
    calls = {"n": 0}
    original = backend._lib.cf_synchronize

    def spy():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(backend._lib, "cf_synchronize", spy)

    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    with forge.cuda.stream(stream_a):
        x = Tensor(np.full((2048,), 4.0, dtype=np.float32), device="cuda")
        y = x + x

    with forge.cuda.stream(stream_b):
        cpu_result = y.to("cpu", non_blocking=True)  # submission only -- must not synchronize

    assert calls["n"] == 0
    np.testing.assert_allclose(cpu_result.numpy(), np.full((2048,), 8.0, dtype=np.float32))  # this read DOES sync


# -- 3. Autograd / optimizer with an asynchronously transferred input (Section 42) --


def test_autograd_works_with_an_asynchronously_transferred_constant_input():
    """An async H2D transfer (always `requires_grad=False`, like every `.to()` result) can
    safely participate as a constant operand in a differentiable CUDA computation."""
    weight = Tensor(np.full((256,), 3.0, dtype=np.float32), device="cuda", requires_grad=True)
    bias_cpu = _pinned_tensor(np.full((256,), 1.0, dtype=np.float32))

    stream = forge.cuda.Stream()
    with forge.cuda.stream(stream):
        bias = bias_cpu.to("cuda", non_blocking=True)  # constant, requires_grad=False
        y = (weight * weight + bias).sum()
        y.backward()

    stream.synchronize()
    np.testing.assert_allclose(weight.grad.to("cpu").numpy(), np.full((256,), 6.0, dtype=np.float32))
