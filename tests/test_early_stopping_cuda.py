"""CUDA `EarlyStopping` tests (Milestone 100).

Proves the best-model snapshot/restore mechanism
(`forge.training.early_stopping._snapshot_model_state`/`_restore_model_state`)
works against real `CUDAStorage`, not just NumPy arrays -- both functions go
through `Backend.to_numpy()`/`Backend.from_array()` (the same transfer
primitives `save_model()`/`save_checkpoint()` already use for CUDA), so a
CUDA `Trainer.fit(..., early_stopping=...)` run must restore the correct
epoch's parameters entirely on-device. Skips cleanly on a CUDA-less machine,
matching every other `tests/test_*_cuda.py` file's `pytestmark`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.exceptions import TrainerError
from forge.nn import Linear, ReLU, Sequential
from forge.nn.loss import MSELoss
from forge.optim import SGD
from forge.training import EarlyStopping, EvaluationResult, Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


def _regression_loader(n=16, batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _cuda_trainer(seed=5):
    forge.random.seed(seed)
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1)).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.05)
    return Trainer(model=model, loss_fn=MSELoss(), optimizer=optimizer, device="cuda", verbose=False)


def _script_validation(trainer, values):
    it = iter(values)

    def fake_evaluate(loader):
        return EvaluationResult(loss=next(it), metrics={}, samples=1, duration=0.0, device="cuda")

    trainer.evaluate = fake_evaluate


def test_cuda_early_stopping_requires_validation_loader():
    trainer = _cuda_trainer()
    loader = _regression_loader()
    with pytest.raises(TrainerError):
        trainer.fit(loader, epochs=5, early_stopping=EarlyStopping())


def test_cuda_early_stopping_stops_and_reports_correctly():
    trainer = _cuda_trainer()
    loader = _regression_loader()
    _script_validation(trainer, [1.0] * 20)
    history = trainer.fit(loader, epochs=20, validation_loader=loader, early_stopping=EarlyStopping(patience=2))
    assert len(history) == 3
    assert history.stopped_early is True
    assert history.best_epoch == 1
    assert str(trainer.model.device) == "cuda"


def test_cuda_restore_best_restores_the_correct_epochs_parameters():
    trainer = _cuda_trainer(seed=7)
    loader = _regression_loader(seed=7)
    values = [1.0, 0.5, 0.6, 0.7]  # best is epoch 2
    _script_validation(trainer, values)
    history = trainer.fit(
        loader, epochs=10, validation_loader=loader,
        early_stopping=EarlyStopping(patience=2, restore_best=True),
    )
    assert history.best_epoch == 2
    assert history.stopped_early is True
    restored_params = [p.to("cpu").numpy().copy() for p in trainer.model.parameters()]
    assert str(trainer.model.device) == "cuda"  # restoration never moved the model off CUDA

    ref_trainer = _cuda_trainer(seed=7)
    ref_loader = _regression_loader(seed=7)
    ref_trainer.fit(ref_loader, epochs=2)
    expected_params = [p.to("cpu").numpy().copy() for p in ref_trainer.model.parameters()]

    assert len(restored_params) == len(expected_params) > 0
    for restored, expected in zip(restored_params, expected_params):
        np.testing.assert_allclose(restored, expected, atol=1e-4)
