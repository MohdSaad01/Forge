"""Milestone 27 tests: real CUDA streams, current-stream tracking, and asynchronous kernel execution.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching every
other `test_cuda_*.py` file. `tests/test_cuda_streams_availability.py` holds
the complementary "CUDA unavailable" tests. `tests/test_cuda_stream_allocator.py`
holds the allocator-specific (same-stream/cross-stream reuse, pending
blocks) tests, and `tests/test_cuda_stream_autograd.py` holds
autograd/optimizer/persistence-on-a-stream tests -- this file covers stream
lifecycle, the current-stream/context-manager mechanism, per-op kernel
execution correctness under an explicit stream, and the "no hidden
synchronization" contract itself.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.stream import CUDAStream
from forge.exceptions import CUDAError
from forge.nn import Conv2d, Linear, MaxPool2d
from forge.nn.loss import MSELoss
from forge.optim import SGD

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


# -- 1. Stream creation / destruction / identity ------------------------------


def test_stream_is_a_real_cuda_stream_with_a_nonnull_handle():
    s = forge.cuda.Stream()
    assert isinstance(s, CUDAStream)
    assert s.handle is not None
    assert s.handle.value not in (None, 0)


def test_two_streams_have_distinct_handles():
    s1 = forge.cuda.Stream()
    s2 = forge.cuda.Stream()
    assert s1.handle.value != s2.handle.value
    assert s1 != s2


def test_stream_synchronize_is_safe_with_no_work_issued():
    s = forge.cuda.Stream()
    s.synchronize()  # must not raise


def test_stream_destroy_is_explicit_and_idempotent():
    s = forge.cuda.Stream()
    s.destroy()
    assert s.handle is None
    s.destroy()  # second call must not raise
    assert s.handle is None


def test_synchronize_on_a_destroyed_stream_raises():
    s = forge.cuda.Stream()
    s.destroy()
    with pytest.raises(CUDAError):
        s.synchronize()


def test_stream_repr_reflects_identity_and_destroyed_state():
    s = forge.cuda.Stream()
    assert "Stream(" in repr(s)
    assert "destroyed" not in repr(s)
    s.destroy()
    assert "destroyed" in repr(s)


# -- 2. Current stream / context manager --------------------------------------


def test_default_current_stream_is_none():
    assert forge.cuda.current_stream() is None


def test_stream_context_manager_makes_stream_current_then_restores_none():
    s = forge.cuda.Stream()
    assert forge.cuda.current_stream() is None
    with forge.cuda.stream(s) as entered:
        assert entered is s
        assert forge.cuda.current_stream() is s
    assert forge.cuda.current_stream() is None


def test_nested_stream_contexts_restore_previous_stream_correctly():
    s1 = forge.cuda.Stream()
    s2 = forge.cuda.Stream()
    with forge.cuda.stream(s1):
        assert forge.cuda.current_stream() is s1
        with forge.cuda.stream(s2):
            assert forge.cuda.current_stream() is s2
        assert forge.cuda.current_stream() is s1
    assert forge.cuda.current_stream() is None


def test_stream_context_restores_previous_stream_even_on_exception():
    s1 = forge.cuda.Stream()
    with pytest.raises(ValueError):
        with forge.cuda.stream(s1):
            assert forge.cuda.current_stream() is s1
            raise ValueError("boom")
    assert forge.cuda.current_stream() is None


def test_set_stream_returns_previous_and_can_be_restored_manually():
    s1 = forge.cuda.Stream()
    previous = forge.cuda.set_stream(s1)
    assert previous is None
    assert forge.cuda.current_stream() is s1
    restored = forge.cuda.set_stream(previous)
    assert restored is s1
    assert forge.cuda.current_stream() is None


def test_stream_context_manager_rejects_non_stream_argument():
    with pytest.raises(TypeError):
        with forge.cuda.stream(object()):
            pass


# -- 3. Kernel execution on an explicit stream --------------------------------


def test_add_on_explicit_stream_produces_correct_result():
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        a = Tensor(np.full((256,), 3.0, dtype=np.float32), device="cuda")
        b = Tensor(np.full((256,), 4.0, dtype=np.float32), device="cuda")
        c = a + b
    s.synchronize()
    np.testing.assert_allclose(c.to("cpu").numpy(), np.full((256,), 7.0, dtype=np.float32))


def test_matmul_on_explicit_stream_matches_cpu():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((32, 16)).astype(np.float32)
    b_np = rng.standard_normal((16, 8)).astype(np.float32)
    expected = a_np @ b_np

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        a = Tensor(a_np, device="cuda")
        b = Tensor(b_np, device="cuda")
        c = a @ b
    s.synchronize()
    np.testing.assert_allclose(c.to("cpu").numpy(), expected, rtol=1e-4, atol=1e-4)


def test_relu_on_explicit_stream_matches_cpu():
    x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x = Tensor(x_np, device="cuda")
        y = x.relu()
    s.synchronize()
    np.testing.assert_allclose(y.to("cpu").numpy(), np.maximum(x_np, 0.0))


def test_conv2d_on_explicit_stream_matches_cpu():
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 1, 8, 8)).astype(np.float32)

    forge.random.seed(0)
    cpu_model = Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
    cpu_result = cpu_model(Tensor(x_np, device="cpu")).numpy()

    forge.random.seed(0)
    cuda_model = Conv2d(1, 4, kernel_size=3, stride=1, padding=1).to("cuda")
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        cuda_out = cuda_model(Tensor(x_np, device="cuda"))
    s.synchronize()
    np.testing.assert_allclose(cuda_out.to("cpu").numpy(), cpu_result, rtol=1e-3, atol=1e-3)


def test_maxpool2d_on_explicit_stream_matches_default_stream():
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)

    default_out = MaxPool2d(2, stride=2)(Tensor(x_np, device="cuda")).to("cpu").numpy()

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        out = MaxPool2d(2, stride=2)(Tensor(x_np, device="cuda"))
    s.synchronize()
    np.testing.assert_allclose(out.to("cpu").numpy(), default_out)


def test_loss_on_explicit_stream_matches_cpu():
    rng = np.random.default_rng(3)
    pred_np = rng.standard_normal((8, 4)).astype(np.float32)
    target_np = rng.standard_normal((8, 4)).astype(np.float32)
    expected = float(np.mean((pred_np - target_np) ** 2))

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        pred = Tensor(pred_np, device="cuda")
        target = Tensor(target_np, device="cuda")
        loss = MSELoss()(pred, target)
    s.synchronize()
    assert loss.to("cpu").numpy() == pytest.approx(expected, rel=1e-4)


def test_optimizer_step_on_explicit_stream_updates_parameters():
    forge.random.seed(0)
    model = Linear(4, 3).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.5)
    x = Tensor(np.random.default_rng(4).standard_normal((6, 4)).astype(np.float32), device="cuda")

    s = forge.cuda.Stream()
    before = [p.to("cpu").numpy().copy() for p in model.parameters()]
    with forge.cuda.stream(s):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
    s.synchronize()
    after = [p.to("cpu").numpy() for p in model.parameters()]
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


# -- 4. No hidden per-operation synchronization -------------------------------


def _issue_many_matmuls(n_ops: int):
    rng = np.random.default_rng(5)
    a = Tensor(rng.standard_normal((256, 256)).astype(np.float32), device="cuda")
    b = Tensor(rng.standard_normal((256, 256)).astype(np.float32), device="cuda")
    x = a
    for _ in range(n_ops):
        x = x @ b
    return x


def test_stream_issuance_does_not_block_like_default_stream_does():
    """Issuing many chained matmuls under an explicit stream must not synchronize per-op.

    A loose, relative comparison (not a hard latency bound, to avoid
    flakiness across machines/load): issuing the same chain on the default
    stream (M26 behavior -- every op synchronizes before returning) must
    take *at least* as long, host-side, to merely *issue*, as issuing it
    under an explicit stream with no synchronization at all. This is a
    coarse but real proof that Milestone 27 removed the per-op
    `cudaDeviceSynchronize()` for the explicit-stream path -- see Section 30
    of the milestone brief ("verify no accidental synchronization").
    """
    n_ops = 60

    s = forge.cuda.Stream()
    start = time.perf_counter()
    with forge.cuda.stream(s):
        _issue_many_matmuls(n_ops)
    stream_issue_time = time.perf_counter() - start
    s.synchronize()

    start = time.perf_counter()
    _issue_many_matmuls(n_ops)  # default stream: every op synchronizes internally
    default_issue_time = time.perf_counter() - start

    assert stream_issue_time < default_issue_time


# -- 5. Stress / leak testing (Sections 38-39 of the milestone brief) --------


def test_small_number_of_streams_repeated_alloc_use_release_sync_stays_correct():
    """N=6 streams, each doing its own allocate/compute/destroy cycle repeatedly."""
    streams = [forge.cuda.Stream() for _ in range(6)]
    for round_ in range(5):
        for i, s in enumerate(streams):
            with forge.cuda.stream(s):
                t = Tensor(np.full((128,), float(i + round_), dtype=np.float32), device="cuda")
                result = t + t
            s.synchronize()
            np.testing.assert_allclose(
                result.to("cpu").numpy(), np.full((128,), 2.0 * (i + round_), dtype=np.float32)
            )


def test_repeated_stream_create_use_synchronize_destroy_does_not_leak():
    """Repeated create -> submit work -> synchronize -> destroy must not grow unboundedly.

    Verified indirectly: the loop itself must complete without error or
    slowdown-to-failure across many cycles, and CUDA allocation accounting
    (`allocated_bytes`) must return to its pre-loop value once every
    storage created inside the loop has gone out of scope.
    """
    before = forge.cuda.memory_stats().allocated_bytes
    t = result = None
    for _ in range(30):
        s = forge.cuda.Stream()
        with forge.cuda.stream(s):
            t = Tensor(np.ones((64,), dtype=np.float32), device="cuda")
            result = t + t
        s.synchronize()
        s.destroy()
    del t, result  # the last iteration's storages would otherwise stay referenced by these locals
    after = forge.cuda.memory_stats().allocated_bytes
    assert after == before
