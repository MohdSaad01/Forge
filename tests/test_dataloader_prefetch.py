"""Milestone 30 tests: `CUDAPrefetchLoader` (asynchronous GPU-prefetch DataLoader wrapper).

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark`, matching this codebase's
established convention (see `tests/test_cuda_transfer_dependencies.py`).
CUDA-unavailable / CPU-only-import behavior lives in the separate
`tests/test_dataloader_prefetch_availability.py`, per that same convention.
"""

from __future__ import annotations

import gc
import threading
import time

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda.backend import is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.data.prefetch import CUDAPrefetchLoader
from forge.exceptions import DataError

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    """Return device memory to the driver between tests -- see `test_cuda_stream_allocator.py`."""
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _range_dataset(n: int, feature_dim: int = 2) -> TensorDataset:
    x = Tensor(np.arange(n * feature_dim, dtype=np.float32).reshape(n, feature_dim))
    y = Tensor(np.arange(n, dtype=np.float32))
    return TensorDataset(x, y)


# -- construction / validation -----------------------------------------------


def test_prefetch_loader_rejects_cpu_device():
    loader = DataLoader(_range_dataset(8), batch_size=4)
    with pytest.raises(DataError):
        CUDAPrefetchLoader(loader, device="cpu")


def test_prefetch_loader_rejects_non_positive_prefetch_size():
    loader = DataLoader(_range_dataset(8), batch_size=4)
    with pytest.raises(DataError):
        CUDAPrefetchLoader(loader, device="cuda", prefetch_size=0)
    with pytest.raises(DataError):
        CUDAPrefetchLoader(loader, device="cuda", prefetch_size=-1)


def test_dataloader_prefetch_method_returns_prefetch_loader():
    loader = DataLoader(_range_dataset(8), batch_size=4)
    prefetch_loader = loader.prefetch(device="cuda", prefetch_size=3)
    assert isinstance(prefetch_loader, CUDAPrefetchLoader)
    assert prefetch_loader.prefetch_size == 3


def test_prefetch_loader_len_matches_wrapped_loader():
    loader = DataLoader(_range_dataset(10), batch_size=4)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda")
    assert len(prefetch_loader) == len(loader) == 3


# -- correctness: batches land on CUDA, values/dtype/shape preserved --------


def test_batches_are_cuda_tensors_with_correct_values():
    ds = _range_dataset(12)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    seen_x = []
    for x, y in prefetch_loader:
        assert str(x.device) == "cuda"
        assert str(y.device) == "cuda"
        assert x.dtype == Tensor(np.zeros(1, dtype=np.float32)).dtype
        seen_x.append(x.to("cpu").numpy())

    expected = np.arange(24, dtype=np.float32).reshape(12, 2)
    np.testing.assert_allclose(np.concatenate(seen_x, axis=0), expected)


def test_dtype_shape_preserved_relative_to_synchronous_path():
    ds = _range_dataset(9, feature_dim=3)
    sync_loader = DataLoader(ds, batch_size=4, shuffle=False)
    sync_batches = [(x.shape, x.dtype, y.shape, y.dtype) for x, y in sync_loader]

    prefetch_loader = CUDAPrefetchLoader(DataLoader(ds, batch_size=4, shuffle=False), device="cuda")
    prefetch_batches = [(x.shape, x.dtype, y.shape, y.dtype) for x, y in prefetch_loader]

    assert sync_batches == prefetch_batches


# -- correctness: synchronous vs. asynchronous ordering (Section 61) --------


def test_ordering_and_values_match_synchronous_loader_unshuffled():
    ds = _range_dataset(23)
    sync_loader = DataLoader(ds, batch_size=5, shuffle=False)
    sync_batches = [(x.numpy().copy(), y.numpy().copy()) for x, y in sync_loader]

    prefetch_loader = CUDAPrefetchLoader(DataLoader(ds, batch_size=5, shuffle=False), device="cuda", prefetch_size=2)
    async_batches = [(x.to("cpu").numpy(), y.to("cpu").numpy()) for x, y in prefetch_loader]

    assert len(sync_batches) == len(async_batches)
    for (sx, sy), (ax, ay) in zip(sync_batches, async_batches):
        np.testing.assert_array_equal(sx, ax)
        np.testing.assert_array_equal(sy, ay)


@pytest.mark.parametrize("prefetch_size", [1, 2, 3])
def test_ordering_and_values_match_synchronous_loader_shuffled(prefetch_size):
    ds = _range_dataset(37)
    sync_loader = DataLoader(ds, batch_size=6, shuffle=True, generator=np.random.default_rng(7))
    sync_batches = [(x.numpy().copy(), y.numpy().copy()) for x, y in sync_loader]

    async_loader = DataLoader(ds, batch_size=6, shuffle=True, generator=np.random.default_rng(7))
    prefetch_loader = CUDAPrefetchLoader(async_loader, device="cuda", prefetch_size=prefetch_size)
    async_batches = [(x.to("cpu").numpy(), y.to("cpu").numpy()) for x, y in prefetch_loader]

    assert len(sync_batches) == len(async_batches)
    for (sx, sy), (ax, ay) in zip(sync_batches, async_batches):
        np.testing.assert_array_equal(sx, ax)
        np.testing.assert_array_equal(sy, ay)


def test_drop_last_semantics_preserved():
    ds = _range_dataset(10)
    sync_loader = DataLoader(ds, batch_size=4, drop_last=True)
    sync_sizes = [x.shape[0] for x, _ in sync_loader]

    prefetch_loader = CUDAPrefetchLoader(DataLoader(ds, batch_size=4, drop_last=True), device="cuda")
    async_sizes = [x.shape[0] for x, _ in prefetch_loader]

    assert sync_sizes == async_sizes == [4, 4]


# -- epoch boundaries / repeated iteration (Section 33) ----------------------


def test_repeated_epochs_each_yield_full_dataset():
    ds = _range_dataset(20)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    for _ in range(4):
        total_samples = sum(x.shape[0] for x, _ in prefetch_loader)
        assert total_samples == 20


def test_no_batches_leak_across_epoch_boundary():
    """Every epoch's shuffled index set, reconstructed from consumed batches, covers 0..n-1 exactly once."""
    ds = _range_dataset(24)
    loader = DataLoader(ds, batch_size=5, shuffle=True)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    for _ in range(3):
        seen = []
        for x, _ in prefetch_loader:
            seen.extend((x.to("cpu").numpy()[:, 0] // 2).astype(int).tolist())
        assert sorted(seen) == list(range(24))


# -- exception propagation (Section 31/63) -----------------------------------


class _RaisingDataset:
    def __init__(self, n: int, fail_at: int):
        self.n = n
        self.fail_at = fail_at

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        if index == self.fail_at:
            raise RuntimeError(f"synthetic dataset failure at index {index}")
        return Tensor(np.array([float(index)], dtype=np.float32)), Tensor(np.array([float(index)], dtype=np.float32))


def test_dataset_exception_propagates_to_consumer():
    ds = _RaisingDataset(20, fail_at=7)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    with pytest.raises(RuntimeError, match="synthetic dataset failure"):
        for _ in prefetch_loader:
            pass


def test_exception_does_not_leave_a_live_background_thread():
    ds = _RaisingDataset(20, fail_at=3)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    with pytest.raises(RuntimeError):
        for _ in prefetch_loader:
            pass

    _wait_for_thread_count(threading.active_count())


# -- early termination / thread + resource cleanup (Section 30/64/66) --------


def _wait_for_thread_count(expected_max: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        prefetch_threads = [t for t in threading.enumerate() if t.name == "forge-prefetch-cpu-producer" and t.is_alive()]
        if not prefetch_threads:
            return
        time.sleep(0.02)
    remaining = [t for t in threading.enumerate() if t.name == "forge-prefetch-cpu-producer" and t.is_alive()]
    assert not remaining, f"prefetch producer thread(s) still alive: {remaining}"


def test_early_termination_stops_background_thread_via_refcounting():
    """Plain `del`/scope-exit refcounting must collect the iterator -- no reliance on gc.collect()."""
    gc.disable()
    try:
        ds = _range_dataset(200)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

        for trial in range(5):
            for i, batch in enumerate(prefetch_loader):
                if i == 1:
                    break

        _wait_for_thread_count(0)
    finally:
        gc.enable()


def test_repeated_early_exit_and_full_epochs_interleaved_do_not_accumulate_threads():
    ds = _range_dataset(100)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    for trial in range(6):
        if trial % 2 == 0:
            for i, batch in enumerate(prefetch_loader):
                if i == 2:
                    break
        else:
            for batch in prefetch_loader:
                pass

    _wait_for_thread_count(0)


# -- leak testing (Section 65) ------------------------------------------------


def test_repeated_epochs_do_not_grow_cuda_or_pinned_memory():
    ds = _range_dataset(64, feature_dim=8)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    for x, y in prefetch_loader:  # warmup epoch (first-use allocations)
        pass
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats().allocated_bytes
    before_pinned = forge.cuda.pinned_memory_stats().pinned_active_bytes

    for _ in range(20):
        for x, y in prefetch_loader:
            z = x + x
            del z

    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats().allocated_bytes
    after_pinned = forge.cuda.pinned_memory_stats().pinned_active_bytes

    assert after == before, f"CUDA active bytes grew: {before} -> {after}"
    assert after_pinned == before_pinned == 0, f"pinned bytes not fully released: {after_pinned}"


# -- no global device-wide synchronization in the hot path (Section 51) -----


def test_consuming_prefetched_batches_never_calls_cuda_device_synchronize(monkeypatch):
    from forge.backend.cuda.backend import get_cuda_backend

    backend = get_cuda_backend()
    calls = {"n": 0}
    original = backend._lib.cf_synchronize

    def spy():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(backend._lib, "cf_synchronize", spy)

    ds = _range_dataset(32)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)

    compute_stream = forge.cuda.Stream()
    with forge.cuda.stream(compute_stream):
        for x, y in prefetch_loader:
            _ = x + x  # touch the tensor on the compute stream, triggering _stream_guard

    assert calls["n"] == 0
