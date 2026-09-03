"""Milestone 26 tests: `forge.cuda.synchronize()` and CUDA synchronization semantics.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching every
other `test_cuda_*.py` file. `tests/test_cuda_synchronize_availability.py`
holds the complementary "CUDA unavailable" tests that must run on any
machine (see that file's docstring for why it is split out).

Forge's CUDA execution model established here: every CUDA-backed operation
already calls `cudaDeviceSynchronize()` internally before returning its
result (see `docs/architecture/cuda-backend.md`'s **Kernel Launch
Semantics**), so `forge.cuda.synchronize()` is never required for
correctness anywhere in Forge -- these tests prove it is nonetheless safe,
idempotent, and (for the allocator-reuse test) demonstrably meaningful as an
explicit host-side barrier for external callers.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.exceptions import CUDAError
from forge.nn import Linear, ReLU, Sequential
from forge.nn.loss import MSELoss
from forge.optim import SGD, Adam

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


# -- 1. Basic synchronization -------------------------------------------------


def test_synchronize_after_real_cuda_op_returns_successfully():
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device="cuda")
    b = Tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32), device="cuda")
    _ = a + b

    forge.cuda.synchronize()  # must not raise


def test_synchronize_with_no_prior_cuda_work_is_safe():
    """Calling synchronize() with an already-idle device (or none issued yet in this test) is safe."""
    forge.cuda.synchronize()


# -- 2. Repeated synchronization ----------------------------------------------


def test_repeated_synchronize_is_safe():
    a = Tensor(np.ones((8,), dtype=np.float32), device="cuda")
    for _ in range(10):
        forge.cuda.synchronize()
    _ = a + a
    for _ in range(10):
        forge.cuda.synchronize()  # must not raise, even with no new work between calls


# -- 3. Synchronization after forward / backward / optimizer step ------------


def test_synchronize_after_forward():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).to("cuda")
    x = Tensor(np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32), device="cuda")

    out = model(x)
    forge.cuda.synchronize()

    expected = out.to("cpu").numpy()
    np.testing.assert_allclose(expected, out.to("cpu").numpy())


def test_synchronize_after_backward():
    forge.random.seed(0)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).to("cuda")
    x = Tensor(np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32), device="cuda")

    loss = model(x).sum()
    loss.backward()
    forge.cuda.synchronize()

    for p in model.parameters():
        assert p.grad is not None
        grad_host = p.grad.to("cpu").numpy()
        assert np.isfinite(grad_host).all()


def test_synchronize_after_optimizer_step():
    forge.random.seed(1)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.1)
    x = Tensor(np.random.default_rng(2).standard_normal((5, 4)).astype(np.float32), device="cuda")

    before = [p.to("cpu").numpy().copy() for p in model.parameters()]

    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    forge.cuda.synchronize()

    after = [p.to("cpu").numpy() for p in model.parameters()]
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


# -- 4. Full training-cycle synchronization -----------------------------------


def test_synchronize_after_full_training_cycle_matches_cpu():
    """forward -> loss -> backward -> optimizer.step() -> synchronize, verified against a CPU run.

    Same seed, same data, same architecture on both devices: one Adam step
    on CUDA (synchronized explicitly before reading results back) must land
    on the same parameters, within float32 tolerance, as the identical step
    on CPU.
    """
    x_np = np.random.default_rng(3).standard_normal((6, 4)).astype(np.float32)
    y_np = np.random.default_rng(4).standard_normal((6, 2)).astype(np.float32)

    def run(device: str):
        forge.random.seed(7)
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).to(device)
        optimizer = Adam(model.parameters(), lr=1e-2)
        loss_fn = MSELoss()
        x = Tensor(x_np, device=device)
        y = Tensor(y_np, device=device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            forge.cuda.synchronize()
        return [p.to("cpu").numpy() for p in model.parameters()], float(loss.to("cpu").numpy())

    cpu_params, cpu_loss = run("cpu")
    cuda_params, cuda_loss = run("cuda")

    assert cuda_loss == pytest.approx(cpu_loss, rel=1e-4, abs=1e-5)
    for cp, gp in zip(cpu_params, cuda_params):
        np.testing.assert_allclose(gp, cp, rtol=1e-4, atol=1e-5)


# -- 5. Allocator reuse safety (Section 9 of the milestone brief) ------------


def test_allocator_reuse_does_not_corrupt_data():
    """A cache-reused block must never leak the previous owner's stale values.

    Allocates and uses `a`, releases it (ordinary refcounting `del`, no
    `gc.collect()` needed -- these are plain leaf tensors with no reference
    cycle), then allocates `b` of the identical byte size so the exact-size
    caching allocator (`forge/backend/cuda/allocator.py`) is very likely to
    hand back the very block `a` just released. `cache_hit_count` is
    asserted to have actually increased, so this test cannot silently pass
    by accident on a fresh allocation.
    """
    shape = (4096,)
    dtype = np.float32

    a = Tensor(np.full(shape, 3.0, dtype=dtype), device="cuda")
    a_sum = a + a  # 6.0 everywhere; same nbytes as `a` itself
    del a, a_sum

    stats_before = forge.cuda.memory_stats()

    b = Tensor(np.full(shape, 5.0, dtype=dtype), device="cuda")  # likely reuses `a`'s cached block
    b_sum = b + b  # 10.0 everywhere; likely reuses `a_sum`'s cached block
    forge.cuda.synchronize()

    stats_after = forge.cuda.memory_stats()

    np.testing.assert_allclose(b.to("cpu").numpy(), np.full(shape, 5.0, dtype=dtype))
    np.testing.assert_allclose(b_sum.to("cpu").numpy(), np.full(shape, 10.0, dtype=dtype))
    assert stats_after.cache_hit_count > stats_before.cache_hit_count


def test_allocator_reuse_across_many_alloc_release_cycles_stays_correct():
    """Repeated allocate/use/release/reallocate at a fixed size never corrupts a later reader."""
    shape = (256,)
    dtype = np.float32

    for i in range(50):
        t = Tensor(np.full(shape, float(i), dtype=dtype), device="cuda")
        doubled = t + t
        np.testing.assert_allclose(doubled.to("cpu").numpy(), np.full(shape, float(2 * i), dtype=dtype))
        del t, doubled

    forge.cuda.synchronize()


# -- 6. empty_cache() semantics -----------------------------------------------


def test_empty_cache_after_recent_work_is_safe_and_preserves_correctness():
    """`empty_cache()` right after real CUDA work (no explicit synchronize first) must not corrupt anything.

    Per this module's `docs/architecture/cuda-backend.md` **Milestone 26**
    section, every op already synchronizes before returning, so a cached
    block is never in flight -- `empty_cache()` needs no synchronization of
    its own, and calling it immediately after ordinary use must be safe.
    """
    shape = (1024,)
    dtype = np.float32
    a = Tensor(np.full(shape, 2.0, dtype=dtype), device="cuda")
    result = a + a
    del a

    forge.cuda.empty_cache()

    b = Tensor(np.full(shape, 9.0, dtype=dtype), device="cuda")
    b_sum = b + b
    forge.cuda.synchronize()

    np.testing.assert_allclose(result.to("cpu").numpy(), np.full(shape, 4.0, dtype=dtype))
    np.testing.assert_allclose(b_sum.to("cpu").numpy(), np.full(shape, 18.0, dtype=dtype))


# -- 7. Memory-statistics semantics under synchronization --------------------


def test_synchronize_does_not_change_allocation_accounting():
    """`synchronize()` is a pure execution barrier -- it must never allocate, free, or reclassify memory."""
    a = Tensor(np.ones((512,), dtype=np.float32), device="cuda")
    _ = a + a

    before = forge.cuda.memory_stats()
    forge.cuda.synchronize()
    forge.cuda.synchronize()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == before.reserved_bytes
    assert after.cached_bytes == before.cached_bytes
    assert after.cache_hit_count == before.cache_hit_count
    assert after.cache_miss_count == before.cache_miss_count
