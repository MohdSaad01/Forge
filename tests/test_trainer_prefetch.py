"""Milestone 30 tests: `Trainer(..., prefetch=True)` integration.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark` (see `tests/test_cuda_transfer_
dependencies.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.data.prefetch import CUDAPrefetchLoader
from forge.exceptions import TrainerError
from forge.nn import Linear, MSELoss, ReLU, Sequential
from forge.optim import SGD
from forge.training import Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _dataset(n: int = 128, in_features: int = 10, out_features: int = 1) -> TensorDataset:
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((n, in_features)).astype(np.float32))
    y = Tensor(rng.standard_normal((n, out_features)).astype(np.float32))
    return TensorDataset(x, y)


def _model() -> Sequential:
    forge.random.seed(42)
    model = Sequential(Linear(10, 16), ReLU(), Linear(16, 1))
    model.to("cuda")
    return model


# -- construction --------------------------------------------------------


def test_prefetch_requires_cuda_device():
    model = Linear(4, 1)
    with pytest.raises(TrainerError):
        Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cpu", prefetch=True)


# -- correctness: identical training trajectory to synchronous Trainer ----


def test_training_loss_matches_synchronous_trainer_exactly():
    ds = _dataset()

    sync_model = _model()
    loader = DataLoader(ds, batch_size=16, shuffle=True, generator=np.random.default_rng(1))
    sync_trainer = Trainer(sync_model, MSELoss(), SGD(sync_model.parameters(), lr=0.01), device="cuda", verbose=False)
    sync_history = sync_trainer.fit(loader, epochs=4)

    pf_model = _model()
    pf_loader = DataLoader(ds, batch_size=16, shuffle=True, generator=np.random.default_rng(1))
    pf_trainer = Trainer(
        pf_model, MSELoss(), SGD(pf_model.parameters(), lr=0.01), device="cuda", verbose=False, prefetch=True
    )
    pf_history = pf_trainer.fit(pf_loader, epochs=4)

    assert sync_history.train_losses == pytest.approx(pf_history.train_losses, rel=1e-5)


def test_validation_loop_works_with_prefetch():
    ds = _dataset(64)
    vds = _dataset(32)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    vloader = DataLoader(vds, batch_size=16)

    model = _model()
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False, prefetch=True)
    history = trainer.fit(loader, epochs=2, validation_loader=vloader)

    assert all(v is not None for v in history.val_losses)
    assert len(history.val_losses) == 2


def test_standalone_evaluate_works_with_prefetch_and_reuses_wrapper():
    ds = _dataset(32)
    loader = DataLoader(ds, batch_size=8)
    model = _model()
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False, prefetch=True)

    result1 = trainer.evaluate(loader)
    result2 = trainer.evaluate(loader)

    assert result1.loss == pytest.approx(result2.loss)
    assert len(trainer._prefetch_loaders) == 1  # same loader object -> cached, not re-wrapped


def test_loader_already_wrapped_is_used_as_is():
    ds = _dataset(32)
    loader = DataLoader(ds, batch_size=8)
    wrapped = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=3)

    model = _model()
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False, prefetch=True)
    result = trainer.evaluate(wrapped)
    assert result.samples == 32
    assert len(trainer._prefetch_loaders) == 0  # never wrapped -- already a CUDAPrefetchLoader


# -- compute stream lifecycle ----------------------------------------------


def test_compute_stream_created_once_and_reused_across_epochs():
    ds = _dataset(32)
    loader = DataLoader(ds, batch_size=8)
    model = _model()
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False, prefetch=True)

    assert trainer._compute_stream is None
    trainer.fit(loader, epochs=1)
    stream_after_epoch_1 = trainer._compute_stream
    assert stream_after_epoch_1 is not None
    trainer.fit(loader, epochs=1)
    assert trainer._compute_stream is stream_after_epoch_1


def test_non_prefetch_trainer_never_touches_compute_stream():
    ds = _dataset(32)
    loader = DataLoader(ds, batch_size=8)
    model = _model()
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False)
    trainer.fit(loader, epochs=1)
    assert trainer._compute_stream is None


# -- no global synchronization in the training hot path (Section 51) ------


def test_fit_never_calls_cuda_device_synchronize(monkeypatch):
    backend = get_cuda_backend()
    calls = {"n": 0}
    original = backend._lib.cf_synchronize

    def spy():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(backend._lib, "cf_synchronize", spy)

    ds = _dataset(64)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = _model()
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cuda", verbose=False, prefetch=True)
    trainer.fit(loader, epochs=2)

    assert calls["n"] == 0


# -- gradients / optimizer semantics unaffected ----------------------------


def test_parameters_actually_update_and_gradients_are_none_free():
    ds = _dataset(64)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = _model()
    first_param = next(iter(model.parameters()))
    initial_weight = first_param.to("cpu").numpy().copy()

    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.1), device="cuda", verbose=False, prefetch=True)
    trainer.fit(loader, epochs=2)

    updated_weight = first_param.to("cpu").numpy()
    assert not np.allclose(initial_weight, updated_weight)
    for p in model.parameters():
        assert p.grad is not None
