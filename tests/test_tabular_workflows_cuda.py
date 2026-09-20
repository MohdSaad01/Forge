"""Milestone 114 CUDA tests: the tabular training workflows on the real CUDA backend.

Device-specific behaviour only; the data/validation/result contract is covered on
CPU in `tests/test_tabular_workflows.py`. Training on CUDA is not bit-identical to CPU
(a documented, pre-existing property), so cross-device checks compare
(a) exactly what is device-independent -- the split, the fitted preprocessing, the
baselines -- and (b) *the same saved weights* run on both devices, within float
tolerance; independently trained CPU and CUDA models are only required to be
comparably good.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.nn import Linear, ReLU, Sequential
from forge.training import Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tabular_workflows import (  # noqa: E402  (shared fixtures data, not a test module dependency of pytest)
    classification_data, regression_data, split_indices,
)


@pytest.fixture(autouse=True)
def _empty_cuda_cache():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


@pytest.fixture
def no_training(monkeypatch):
    def tripwire(self, *args, **kwargs):
        raise AssertionError("training started: a preflight check ran too late")

    monkeypatch.setattr(Trainer, "fit", tripwire)


@pytest.fixture(scope="module")
def cuda_cls(tmp_path_factory):
    X, y = classification_data()
    path = tmp_path_factory.mktemp("cuda_cls") / "model.forge"
    r = forge.train_tabular_classifier(X, y, path=path, classes=["neg", "pos"], device="cuda", seed=3, epochs=60)
    return X, y, path, r


@pytest.fixture(scope="module")
def cuda_reg(tmp_path_factory):
    X, y = regression_data()
    path = tmp_path_factory.mktemp("cuda_reg") / "model.forge"
    r = forge.train_tabular_regressor(X, y, path=path, device="cuda", seed=3, epochs=300)
    return X, y, path, r


def test_classifier_trains_on_cuda_and_the_artifact_predicts_and_evaluates(cuda_cls):
    X, y, path, r = cuda_cls
    assert forge.inspect_model(str(path)).device == "cuda"
    assert r.validation_accuracy > r.baseline_accuracy + 0.2
    predictor = forge.load_predictor(str(path))  # lands on CUDA, the device recorded in the artifact
    assert next(iter(predictor.model.parameters())).device.type == "cuda"
    train_idx, val_idx = split_indices(len(X), 3)
    val = predictor.evaluate(X[val_idx], y[val_idx])  # float64 raw rows, straight from NumPy
    assert (val.accuracy, val.loss) == (r.validation_accuracy, r.validation_loss)
    assert val.baseline_accuracy == np.bincount(y[val_idx]).max() / len(val_idx)
    assert predictor.predict(X[:3])[0].label in ("neg", "pos")


def test_regressor_trains_on_cuda_and_the_artifact_predicts_and_evaluates(cuda_reg):
    X, y, path, r = cuda_reg
    assert forge.inspect_model(str(path)).device == "cuda"
    assert r.validation_mse < 0.1 * r.baseline_mse
    predictor = forge.load_predictor(str(path))
    _, val_idx = split_indices(len(X), 3)
    val = predictor.evaluate(X[val_idx], y[val_idx])
    assert (val.mse, val.mae) == (r.validation_mse, r.validation_mae)
    assert val.baseline_mse == np.mean((y[val_idx] - y[val_idx].mean()) ** 2)
    assert tuple(predictor.predict(X[:4]).shape) == (4, 1)


def test_cpu_and_cuda_training_split_and_preprocess_identically_and_learn_comparably(cuda_cls, cuda_reg, tmp_path):
    Xc, yc, cls_path, cuda_c = cuda_cls
    cpu_c = forge.train_tabular_classifier(Xc, yc, path=tmp_path / "c.forge", classes=["neg", "pos"], seed=3, epochs=60)
    assert cuda_c.baseline_accuracy == cpu_c.baseline_accuracy  # same validation rows
    assert (forge.load_preprocessing(str(cls_path)).transforms[-1].mean
            == forge.load_preprocessing(str(tmp_path / "c.forge")).transforms[-1].mean)  # fitted on the host, identically
    assert abs(cuda_c.validation_accuracy - cpu_c.validation_accuracy) < 0.1

    Xr, yr, _, cuda_r = cuda_reg
    cpu_r = forge.train_tabular_regressor(Xr, yr, path=tmp_path / "r.forge", seed=3, epochs=300)
    assert cuda_r.baseline_mse == cpu_r.baseline_mse
    assert cuda_r.validation_mse < 0.1 * cuda_r.baseline_mse and cpu_r.validation_mse < 0.1 * cpu_r.baseline_mse


def test_the_same_weights_agree_on_cpu_and_cuda(cuda_cls, cuda_reg):
    Xc, yc, cls_path, _ = cuda_cls
    on_cuda, on_cpu = forge.load_predictor(str(cls_path)), forge.load_predictor(str(cls_path), device="cpu")
    a, b = on_cuda.evaluate(Xc, yc), on_cpu.evaluate(Xc, yc)
    assert a.accuracy == b.accuracy and a.confusion_matrix.tolist() == b.confusion_matrix.tolist()
    assert a.loss == pytest.approx(b.loss, abs=1e-4)
    assert [p.index for p in on_cuda.predict(Xc[:30])] == [p.index for p in on_cpu.predict(Xc[:30])]

    Xr, yr, reg_path, _ = cuda_reg
    reg_cuda, reg_cpu = forge.load_predictor(str(reg_path)), forge.load_predictor(str(reg_path), device="cpu")
    np.testing.assert_allclose(reg_cuda.predict(Xr[:30]).numpy(), reg_cpu.predict(Xr[:30]).numpy(), rtol=1e-4, atol=1e-3)
    assert reg_cuda.evaluate(Xr, yr).mse == pytest.approx(reg_cpu.evaluate(Xr, yr).mse, rel=1e-4)


@pytest.mark.parametrize("cast", [np.float64, np.int64, list])
def test_a_cuda_artifact_accepts_the_dtypes_a_csv_reader_gives(cuda_cls, cast):
    """Found by M114's real-data CUDA run: float64/int input raised `CUDA 'matmul' requires matching dtypes`."""
    X, y, path, _ = cuda_cls
    predictor = forge.load_predictor(str(path))
    rows = X[:6].round(0)
    expected = [p.index for p in predictor.predict(rows.astype(np.float32))]
    given = rows.tolist() if cast is list else rows.astype(cast)
    assert [p.index for p in predictor.predict(given)] == expected
    assert predictor.evaluate(given, y[:6]).samples == 6


def test_custom_model_on_cpu_is_moved_to_cuda_and_trained(tmp_path):
    X, y = classification_data(120)
    mine = Sequential(Linear(3, 6), ReLU(), Linear(6, 2))
    r = forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", model=mine, device="cuda", epochs=8, patience=None)
    assert next(iter(mine.parameters())).device.type == "cuda"
    assert forge.load_predictor(str(tmp_path / "m.forge")).evaluate(X, y).samples == 120 and r.epochs_completed == 8


def test_preflight_failures_are_raised_before_epoch_1_with_device_cuda(tmp_path, no_training):
    X, y = classification_data(100)
    with pytest.raises(forge.TrainerError, match="needs"):
        forge.train_tabular_classifier(
            X, y, path=tmp_path / "m.forge", device="cuda", model=Sequential(Linear(3, 4), Linear(4, 5)),
        )
    with pytest.raises(forge.PersistenceError):
        forge.train_tabular_regressor(X, y, path=tmp_path / "nope" / "m.forge", device="cuda")
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(forge.DataError, match="non-finite"):
        forge.train_tabular_classifier(Xn, y, path=tmp_path / "m.forge", device="cuda")
    assert list(tmp_path.iterdir()) == []
