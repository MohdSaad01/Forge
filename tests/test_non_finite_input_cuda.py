"""Issue I1 CUDA tests: the non-finite-loss guard and numeric-inference check on real CUDA hardware.

CPU counterparts (and the full contract) live in `tests/test_non_finite_input.py`.
The guard reads the loss on the host (`loss.to("cpu")`), and on a failure inspects
the device batch, so these confirm both work when the model and batch are on CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.exceptions import DataError, TrainerError
from forge.nn import Linear, ReLU, Sequential
from forge.nn.loss import MSELoss
from forge.optim import SGD
from forge.serialization import save_model
from forge.training import Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


def _arrays(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    return X, y


def _cuda_trainer():
    forge.random.seed(0)
    model = Linear(2, 1).to("cuda")
    return Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        device="cuda", verbose=False,
    )


def _loader(X, y):
    return DataLoader(TensorDataset(Tensor(X), Tensor(y)), batch_size=8, shuffle=False)


def test_cuda_fit_rejects_non_finite_features_before_any_update():
    X, y = _arrays()
    X[5, 1] = np.nan
    trainer = _cuda_trainer()
    before = [p.to("cpu").numpy().copy() for p in trainer.model.parameters()]
    with pytest.raises(TrainerError, match="training batch 1.*1 non-finite value\\(s\\) in the features"):
        trainer.fit(_loader(X, y), epochs=2)
    for old, param in zip(before, trainer.model.parameters()):
        np.testing.assert_array_equal(old, param.to("cpu").numpy())


def test_cuda_fit_rejects_non_finite_targets():
    X, y = _arrays()
    y[3, 0] = np.inf
    with pytest.raises(TrainerError, match="in the targets"):
        _cuda_trainer().fit(_loader(X, y), epochs=1)


def test_cuda_evaluate_rejects_non_finite_features():
    X, y = _arrays()
    X[2, 0] = np.nan
    with pytest.raises(TrainerError, match="evaluation batch 1"):
        _cuda_trainer().evaluate(_loader(X, y))


def test_cuda_clean_data_still_trains_with_finite_losses():
    X, y = _arrays()
    history = _cuda_trainer().fit(_loader(X, y), epochs=3)
    assert all(np.isfinite(r.train_loss) for r in history)


def test_cuda_loaded_predictor_rejects_non_finite_input(tmp_path):
    forge.random.seed(0)
    path = tmp_path / "tabular.forge"
    save_model(
        Sequential(Linear(3, 6), ReLU(), Linear(6, 2)), str(path),
        classes=["a", "b"], task="tabular_classification",
    )
    predictor = forge.load_predictor(str(path), device="cuda")
    with pytest.raises(DataError, match="non-finite"):
        predictor.predict([[0.5, np.nan, 0.1]])
    assert np.isfinite(predictor.predict([[0.5, 0.2, 0.1]])[0].confidence)
