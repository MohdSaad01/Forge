"""Milestone 31 tests: fused CrossEntropyLoss (`Tensor.cross_entropy`, `CUDABackend.cross_entropy*`).

`tests/test_cuda_loss.py` already covers `nn.CrossEntropyLoss`'s CUDA
forward/backward correctness (vs. CPU, vs. finite differences), reduction
semantics, device validation, and "no CPU fallback" end-to-end through the
public `Loss` API -- those tests were written before Milestone 31 and were
not modified by it (they still pass unchanged, since the fusion preserves
`CrossEntropyLoss`'s exact public behavior). This module instead covers what
is new at Milestone 31: `Tensor.cross_entropy`'s own defense-in-depth
validation (reachable only by calling it directly, since `nn.CrossEntropyLoss`
already validates everything before it dispatches here), cross-stream
correctness of the fused primitive (Section 31 of the milestone brief), and
that repeated use leaves no CUDA/allocator resource growing unbounded
(Section 32/33).
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda.backend import is_cuda_available
from forge.exceptions import CUDAError, ShapeMismatchError, UnsupportedDeviceError

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-5, atol=1e-5)


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


# -- Tensor.cross_entropy: own validation (defense-in-depth) ------------------


def test_tensor_cross_entropy_rejects_non_2d_logits():
    logits = Tensor([1.0, 2.0, 3.0], device="cuda")
    target = Tensor([0], dtype="int64", device="cuda")
    with pytest.raises(ShapeMismatchError):
        logits.cross_entropy(target)


def test_tensor_cross_entropy_rejects_mismatched_target_length():
    logits = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    target = Tensor([0, 1, 0], dtype="int64", device="cuda")
    with pytest.raises(ShapeMismatchError):
        logits.cross_entropy(target)


def test_tensor_cross_entropy_rejects_device_mismatch():
    logits = Tensor([[1.0, 2.0, 3.0]], device="cuda")
    target = Tensor([0], dtype="int64")  # cpu
    with pytest.raises(UnsupportedDeviceError):
        logits.cross_entropy(target)


def test_cuda_backend_cross_entropy_rejects_non_int64_target():
    """`CUDABackend.cross_entropy` requires int64 target indices internally."""
    logits = Tensor([[1.0, 2.0, 3.0]], device="cuda")
    target = Tensor([0], dtype="int32", device="cuda")
    with pytest.raises(CUDAError):
        logits.cross_entropy(target)


# -- Cross-stream correctness (Section 31 of the milestone brief) ------------


def test_cross_entropy_forward_correct_when_logits_and_target_from_different_streams():
    """logits produced on stream A, target produced on stream B, cross_entropy on stream C."""
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    stream_c = forge.cuda.Stream()

    logits_data = np.array([[2.0, 1.0, 0.1], [0.0, 1.0, 2.0]], dtype=np.float32)
    target_data = np.array([0, 2], dtype=np.int64)

    with forge.cuda.stream(stream_a):
        logits = Tensor(logits_data.copy(), device="cuda", requires_grad=True)
    with forge.cuda.stream(stream_b):
        target = Tensor(target_data.copy(), device="cuda")

    with forge.cuda.stream(stream_c):
        loss = logits.cross_entropy(target)
        loss.backward()
    stream_c.synchronize()

    cpu_logits = Tensor(logits_data.copy(), requires_grad=True)
    expected = forge.nn.CrossEntropyLoss()(cpu_logits, target_data)
    expected.backward()

    np.testing.assert_allclose(loss.to("cpu").numpy(), expected.numpy(), **TOL)
    assert logits.grad is not None
    np.testing.assert_allclose(logits.grad.to("cpu").numpy(), cpu_logits.grad.numpy(), **TOL)


def test_cross_entropy_backward_correct_when_grad_output_from_different_stream():
    """A non-default upstream gradient (from a producing op on a different stream) still gives the right result."""
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    logits_data = np.array([[1.0, 0.5, -0.5], [0.2, 0.1, -0.3]], dtype=np.float32)
    target_data = np.array([1, 0], dtype=np.int64)

    with forge.cuda.stream(stream_a):
        logits = Tensor(logits_data.copy(), device="cuda", requires_grad=True)
        target = Tensor(target_data.copy(), device="cuda")
        loss = logits.cross_entropy(target)
        # Multiply by a scalar on stream B before backward -- grad_output
        # reaching cross_entropy_backward is then last-produced on stream B,
        # not stream A, exercising `_stream_guard` on the backward path too.
    with forge.cuda.stream(stream_b):
        scaled = loss * Tensor(2.0, device="cuda")
    with forge.cuda.stream(stream_a):
        scaled.backward()
    stream_a.synchronize()

    cpu_logits = Tensor(logits_data.copy(), requires_grad=True)
    (forge.nn.CrossEntropyLoss()(cpu_logits, target_data) * 2.0).backward()

    np.testing.assert_allclose(logits.grad.to("cpu").numpy(), cpu_logits.grad.numpy(), **TOL)


# -- Resource lifetime / memory safety (Section 32/33) ------------------------


def test_cross_entropy_repeated_use_does_not_grow_active_memory():
    logits_data = np.random.default_rng(0).standard_normal((32, 10)).astype(np.float32)
    target_data = np.random.default_rng(1).integers(0, 10, size=32).astype(np.int64)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    for _ in range(50):
        logits = Tensor(logits_data.copy(), device="cuda", requires_grad=True)
        target = Tensor(target_data.copy(), device="cuda")
        loss = logits.cross_entropy(target)
        loss.backward()

    # The last iteration's `logits`/`target`/`loss` (and `logits.grad`) are
    # still bound to these locals -- release them explicitly before
    # snapshotting "after", or the comparison below would fail on live,
    # correctly-retained memory rather than a real leak.
    del logits, target, loss

    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes == 0
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0
