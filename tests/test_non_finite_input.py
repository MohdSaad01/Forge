"""Issue I1 tests: NaN/Inf data is rejected with a Forge error, never silently trained on.

Before I1, a NaN feature built a `TensorDataset` fine, trained to `loss: nan`
every epoch, and `train_and_save()` then failed only after all epochs with a
misleading "reloaded model differs (max abs diff nan)" persistence error; a
NaN input row to a numeric artifact returned an arbitrary
`ClassificationPrediction(label=..., confidence=nan)` with no error.

Contract pinned here:

- `Trainer` (`fit()`/`evaluate()`, and so `forge.train()`/`train_and_save()`)
  raises `TrainerError` on the first batch whose loss is NaN/Inf, *before*
  `backward()`/`optimizer.step()`, with a message that says whether the batch
  data or the computation is at fault.
- Numeric artifact inference raises `DataError` when the input the model would
  receive (after preprocessing) has NaN/Inf.
- `ReplaceValue(sentinel=nan)` is rejected (it could never match).

The CUDA counterparts live in `tests/test_non_finite_input_cuda.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.data import DataLoader, ReplaceValue, TensorDataset
from forge.exceptions import DataError, ForgeError, TrainerError
from forge.nn import Linear, ReLU, Sequential
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD
from forge.serialization import save_model
from forge.training import Trainer


def _regression_arrays(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    return X, y


def _trainer(model=None, lr=0.05):
    forge.random.seed(0)
    model = model if model is not None else Linear(2, 1)
    return Trainer(
        model=model,
        loss_fn=MSELoss(),
        optimizer=SGD(model.parameters(), lr=lr),
        device="cpu",
        verbose=False,
    )


def _loader(X, y, batch_size=8):
    return DataLoader(TensorDataset(Tensor(X), Tensor(y)), batch_size=batch_size, shuffle=False)


def _params(model):
    return [p.numpy().copy() for p in model.parameters()]


# -- training ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_non_finite_features(bad):
    X, y = _regression_arrays()
    X[5, 1] = bad
    trainer = _trainer()
    with pytest.raises(TrainerError) as info:
        trainer.fit(_loader(X, y), epochs=2)
    message = str(info.value)
    assert "Non-finite loss" in message
    assert "training batch 1" in message  # row 5 is in the first batch of 8
    assert "1 non-finite value(s) in the features" in message
    assert "before building the dataset" in message


def test_fit_rejects_non_finite_targets():
    X, y = _regression_arrays()
    y[3, 0] = np.nan
    with pytest.raises(TrainerError, match="1 non-finite value\\(s\\) in the targets"):
        _trainer().fit(_loader(X, y), epochs=1)


def test_fit_raises_before_any_parameter_update_when_the_bad_batch_is_first():
    X, y = _regression_arrays()
    X[0, 0] = np.nan
    trainer = _trainer()
    before = _params(trainer.model)
    with pytest.raises(TrainerError):
        trainer.fit(_loader(X, y), epochs=1)
    for old, new in zip(before, _params(trainer.model)):
        np.testing.assert_array_equal(old, new)
    assert trainer.global_step == 0


def test_fit_reports_the_batch_where_a_late_bad_row_sits():
    X, y = _regression_arrays(n=32)
    X[20, 0] = np.nan  # batch_size 8 -> third batch
    with pytest.raises(TrainerError, match="training batch 3"):
        _trainer().fit(_loader(X, y), epochs=1)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_fit_distinguishes_divergence_from_bad_data():
    X, y = _regression_arrays()
    trainer = _trainer(lr=1e3)  # finite data, diverging optimizer
    with pytest.raises(TrainerError) as info:
        trainer.fit(_loader(X, y), epochs=40)
    message = str(info.value)
    assert "Non-finite loss" in message
    assert "features and targets are finite" in message
    assert "learning rate" in message


def test_non_finite_error_is_a_forge_error():
    assert issubclass(TrainerError, ForgeError)


def test_evaluate_rejects_non_finite_features():
    X, y = _regression_arrays()
    X[2, 0] = np.nan
    trainer = _trainer()
    with pytest.raises(TrainerError, match="evaluation batch 1"):
        trainer.evaluate(_loader(X, y))


def test_evaluate_failure_still_restores_training_mode():
    X, y = _regression_arrays()
    X[2, 0] = np.nan
    trainer = _trainer()
    trainer.model.train()
    with pytest.raises(TrainerError):
        trainer.evaluate(_loader(X, y))
    assert trainer.model.training is True


def test_fit_rejects_non_finite_validation_data():
    X, y = _regression_arrays()
    Xv, yv = _regression_arrays(n=16, seed=1)
    Xv[4, 1] = np.nan
    with pytest.raises(TrainerError, match="evaluation batch"):
        _trainer().fit(_loader(X, y), epochs=2, validation_loader=_loader(Xv, yv))


def test_forge_train_rejects_non_finite_features():
    X, y = _regression_arrays()
    X[5, 1] = np.nan
    model = Linear(2, 1)
    with pytest.raises(TrainerError, match="Non-finite loss"):
        forge.train(
            model,
            TensorDataset(Tensor(X), Tensor(y)),
            loss=MSELoss(),
            optimizer=SGD(model.parameters(), lr=0.05),
            epochs=2,
            batch_size=8,
            verbose=False,
        )


def test_train_and_save_rejects_non_finite_features_before_writing_an_artifact(tmp_path):
    X, y = _regression_arrays()
    X[5, 1] = np.nan
    model = Linear(2, 1)
    path = tmp_path / "nan.forge"
    with pytest.raises(TrainerError, match="Non-finite loss"):
        forge.train_and_save(
            model,
            TensorDataset(Tensor(X), Tensor(y)),
            loss=MSELoss(),
            optimizer=SGD(model.parameters(), lr=0.05),
            epochs=2,
            path=str(path),
            sample=Tensor(X[:2]),
            batch_size=8,
            verbose=False,
        )
    assert not path.exists()


def test_clean_data_still_trains_and_saves(tmp_path):
    X, y = _regression_arrays()
    forge.random.seed(0)
    model = Linear(2, 1)
    path = tmp_path / "clean.forge"
    result = forge.train_and_save(
        model,
        TensorDataset(Tensor(X), Tensor(y)),
        loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05),
        epochs=3,
        path=str(path),
        sample=Tensor(X[:2]),
        batch_size=8,
        verbose=False,
    )
    assert np.isfinite(result.train_loss)
    assert path.exists()


def test_integer_class_targets_are_not_flagged():
    """Class-index targets are integers and can never be NaN; a finite classification run is unaffected."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    forge.random.seed(0)
    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    trainer = Trainer(
        model=model, loss_fn=CrossEntropyLoss(), optimizer=SGD(model.parameters(), lr=0.1),
        device="cpu", verbose=False,
    )
    history = trainer.fit(
        DataLoader(TensorDataset(Tensor(X), Tensor(y)), batch_size=8, shuffle=False), epochs=2
    )
    assert all(np.isfinite(r.train_loss) for r in history)


# -- inference ---------------------------------------------------------------


def _save_tabular(tmp_path, *, preprocessing=None):
    forge.random.seed(0)
    model = Sequential(Linear(3, 6), ReLU(), Linear(6, 2))
    path = tmp_path / "tabular.forge"
    save_model(
        model, str(path), preprocessing=preprocessing, classes=["a", "b"], task="tabular_classification"
    )
    return str(path)


def _save_regression(tmp_path):
    forge.random.seed(0)
    path = tmp_path / "regression.forge"
    save_model(Linear(3, 1), str(path), task="regression")
    return str(path)


_GOOD = [[0.5, 0.2, 0.1]]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_tabular_classification_inference_rejects_non_finite_input(tmp_path, bad):
    path = _save_tabular(tmp_path)
    with pytest.raises(DataError, match="predict_tabular_classification_artifact.*non-finite"):
        forge.predict_tabular_classification_artifact(path, [[0.5, bad, 0.1]])


def test_inference_rejects_the_whole_call_when_one_row_is_bad(tmp_path):
    path = _save_tabular(tmp_path)
    with pytest.raises(DataError, match="1 non-finite value"):
        forge.predict_tabular_classification_artifact(path, [[0.5, 0.2, 0.1], [0.5, np.nan, 0.1]])


def test_regression_inference_rejects_non_finite_input(tmp_path):
    path = _save_regression(tmp_path)
    with pytest.raises(DataError, match="predict_tensor_artifact.*non-finite"):
        forge.predict_tensor_artifact(path, np.array([[0.5, np.nan, 0.1]], dtype=np.float32))


def test_predict_model_rejects_non_finite_input(tmp_path):
    path = _save_tabular(tmp_path)
    with pytest.raises(DataError, match="non-finite"):
        forge.predict_model(path, [[np.nan, 0.2, 0.1]])


def test_load_predictor_rejects_non_finite_input_for_both_numeric_tasks(tmp_path):
    for saver in (_save_tabular, _save_regression):
        predictor = forge.load_predictor(saver(tmp_path))
        with pytest.raises(DataError, match="non-finite"):
            predictor.predict([[0.5, np.nan, 0.1]])


def test_finite_input_still_predicts(tmp_path):
    path = _save_tabular(tmp_path)
    results = forge.predict_tabular_classification_artifact(path, _GOOD)
    assert len(results) == 1 and np.isfinite(results[0].confidence)
    assert forge.load_predictor(path).predict(_GOOD)[0].label in ("a", "b")


def test_inference_check_runs_after_preprocessing(tmp_path):
    """A persisted transform that removes the offending value is honored; only what
    the model would actually receive is checked."""
    path = _save_tabular(
        tmp_path, preprocessing=ReplaceValue(sentinel=np.inf, columns=[1], fill=[0.0])
    )
    result = forge.predict_tabular_classification_artifact(path, [[0.5, np.inf, 0.1]])
    assert np.isfinite(result[0].confidence)
    with pytest.raises(DataError, match="non-finite"):
        forge.predict_tabular_classification_artifact(path, [[0.5, np.nan, 0.1]])


# -- ReplaceValue ------------------------------------------------------------


def test_replace_value_rejects_nan_sentinel():
    with pytest.raises(DataError, match="NaN as a sentinel"):
        ReplaceValue(sentinel=float("nan"), columns=[0], fill=[0.0])


def test_replace_value_still_supports_infinite_sentinel():
    out = ReplaceValue(sentinel=np.inf, columns=[0], fill=[7.0])(Tensor([np.inf, 1.0]))
    np.testing.assert_allclose(out.numpy(), [7.0, 1.0])
