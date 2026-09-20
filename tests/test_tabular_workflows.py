"""Milestone 114 tests: `forge.train_tabular_classifier()` / `forge.train_tabular_regressor()` on CPU.

Behaviour through the public API only. The split is reproduced independently of the
implementation from its documented rule (`random_split`: `default_rng(seed).permutation(n)`,
train block first), so leakage, baselines and result numbers are checked against
NumPy, not against the code under test.

Data is deliberately awkward: raw features sit around 50,000 / -3 / 200 with scales of
1000 / 0.01 / 5, so a model that does not receive the artifact's persisted preprocessing
cannot work -- that is what the mutation tests below rely on.

`Trainer.fit` is replaced by a tripwire wherever a test claims "rejected before epoch 1":
if training starts, the test fails with `AssertionError`, not with the expected error.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.cli.main import main as cli_main
from forge.nn import Linear, Module, ReLU, Sequential
from forge.training import Trainer

N = 300
VAL_FRACTION = 0.2
SCALES = np.array([1000.0, 0.01, 5.0])
OFFSETS = np.array([50000.0, -3.0, 200.0])


def classification_data(n: int = N, seed: int = 0) -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    y = (2.0 * z[:, 0] - 1.5 * z[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return z * SCALES + OFFSETS, y


def regression_data(n: int = N, seed: int = 0) -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    y = 60.0 + 10.0 * z[:, 0] - 6.0 * z[:, 1] + 0.3 * rng.normal(size=n)
    return z * SCALES + OFFSETS, y


def split_indices(n: int, seed: int, val_fraction: float = VAL_FRACTION) -> "tuple[np.ndarray, np.ndarray]":
    n_val = max(1, round(n * val_fraction))
    permutation = np.random.default_rng(seed).permutation(n)
    return permutation[: n - n_val], permutation[n - n_val:]


@pytest.fixture
def no_training(monkeypatch):
    def tripwire(self, *args, **kwargs):
        raise AssertionError("training started: a preflight check ran too late")

    monkeypatch.setattr(Trainer, "fit", tripwire)


@pytest.fixture(scope="module")
def cls_run(tmp_path_factory):
    X, y = classification_data()
    path = tmp_path_factory.mktemp("cls") / "model.forge"
    result = forge.train_tabular_classifier(X, y, path=path, classes=["neg", "pos"], seed=3, epochs=80)
    return X, y, path, result


@pytest.fixture(scope="module")
def reg_run(tmp_path_factory):
    X, y = regression_data()
    path = tmp_path_factory.mktemp("reg") / "model.forge"
    result = forge.train_tabular_regressor(X, y, path=path, seed=3, epochs=300)
    return X, y, path, result


def _persisted_normalize(path):
    return forge.load_preprocessing(str(path)).transforms[-1]


# --------------------------------------------------------------------------- public surface


def test_public_exports():
    assert forge.train_tabular_classifier is forge.training.train_tabular_classifier
    assert forge.train_tabular_regressor is forge.training.train_tabular_regressor
    for name in ("train_tabular_classifier", "train_tabular_regressor",
                 "TabularClassificationResult", "TabularRegressionResult"):
        assert name in forge.__all__ and name in forge.training.__all__


# --------------------------------------------------------------------------- classification: happy path


def test_classifier_trains_saves_and_beats_the_baseline(cls_run):
    X, y, path, r = cls_run
    assert isinstance(r, forge.TabularClassificationResult)
    assert r.task == "tabular_classification"
    assert (r.samples, r.features) == (N, 3)
    assert (r.train_samples, r.validation_samples) == (240, 60)
    assert r.classes == ["neg", "pos"]
    assert r.artifact_path == str(path) and path.is_file()
    assert 1 <= r.epochs_completed <= 80
    for value in (r.train_loss, r.validation_loss, r.train_accuracy, r.validation_accuracy, r.baseline_accuracy):
        assert np.isfinite(value)
    assert r.validation_accuracy > r.baseline_accuracy + 0.2
    assert r.train_accuracy > 0.85
    assert isinstance(r.history, forge.TrainingResult) and isinstance(r.model, forge.nn.Module)


def test_only_the_artifact_is_left_on_disk(cls_run):
    _, _, path, _ = cls_run
    assert [p.name for p in path.parent.iterdir()] == ["model.forge"]  # the preflight temp file is gone


def test_result_is_frozen(cls_run):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cls_run[3].validation_accuracy = 1.0


def test_result_numbers_are_the_evaluate_numbers_of_the_saved_artifact(cls_run):
    X, y, path, r = cls_run
    train_idx, val_idx = split_indices(N, 3)
    predictor = forge.load_predictor(str(path))
    val = predictor.evaluate(X[val_idx], y[val_idx])
    train = predictor.evaluate(X[train_idx], y[train_idx])
    assert r.validation_accuracy == val.accuracy and r.validation_loss == val.loss
    assert r.train_accuracy == train.accuracy and r.train_loss == train.loss
    # M113 baseline: majority share of the *validation* rows, computed here independently.
    assert r.baseline_accuracy == np.bincount(y[val_idx]).max() / len(val_idx)


def test_artifact_predicts_and_evaluates_raw_rows(cls_run):
    X, y, path, r = cls_run
    predictor = forge.load_predictor(str(path))
    assert predictor.task == "tabular_classification" and predictor.classes == ["neg", "pos"]
    assert predictor.input_schema.feature_count == 3
    predictions = predictor.predict(X[:20])
    assert [p.label for p in predictions] == [p.label for p in forge.predict_model(str(path), X[:20])]
    assert {p.label for p in predictions} <= {"neg", "pos"}
    accuracy = np.mean([p.label == ("neg", "pos")[t] for p, t in zip(predictions, y[:20])])
    assert accuracy >= 0.8
    held_out_X, held_out_y = classification_data(200, seed=99)
    evaluation = predictor.evaluate(held_out_X, held_out_y)
    assert evaluation.accuracy > evaluation.baseline_accuracy + 0.2
    assert evaluation.classes == ("neg", "pos")


def test_early_stopping_is_on_by_default_and_saves_the_best_epoch(tmp_path):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(120, 4)), rng.integers(0, 2, size=120)  # no signal: validation loss only gets worse
    r = forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", epochs=200, patience=3)
    assert r.stopped_early and r.epochs_completed < 200
    assert r.best_epoch is not None and r.best_epoch <= r.epochs_completed - 3
    # the saved model is the best epoch's: its validation loss is the monitored best, not the last epoch's
    assert r.validation_loss == pytest.approx(r.best_monitored_value, rel=1e-4)
    assert r.validation_loss <= r.history[-1].val_loss


def test_patience_none_runs_every_epoch(tmp_path):
    X, y = classification_data(80)
    r = forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", epochs=7, patience=None)
    assert r.epochs_completed == 7 and not r.stopped_early
    assert r.best_epoch is None and r.best_monitored_value is None


def test_same_seed_reproduces_and_a_different_seed_splits_differently(tmp_path):
    X, y = classification_data(120)
    a = forge.train_tabular_classifier(X, y, path=tmp_path / "a.forge", seed=5, epochs=10, patience=None)
    b = forge.train_tabular_classifier(X, y, path=tmp_path / "b.forge", seed=5, epochs=10, patience=None)
    c = forge.train_tabular_classifier(X, y, path=tmp_path / "c.forge", seed=6, epochs=10, patience=None)
    assert (a.validation_loss, a.train_loss) == (b.validation_loss, b.train_loss)
    assert _persisted_normalize(tmp_path / "a.forge").mean == _persisted_normalize(tmp_path / "b.forge").mean
    assert _persisted_normalize(tmp_path / "a.forge").mean != _persisted_normalize(tmp_path / "c.forge").mean


# --------------------------------------------------------------------------- classes


def test_string_labels_get_sorted_deterministic_classes_whatever_the_row_order(tmp_path):
    X, y = classification_data(120)
    names = np.where(y == 1, "zebra", "ant")
    order = np.random.default_rng(0).permutation(len(X))
    first = forge.train_tabular_classifier(X, names, path=tmp_path / "a.forge", epochs=5)
    second = forge.train_tabular_classifier(X[order], names[order], path=tmp_path / "b.forge", epochs=5)
    assert first.classes == second.classes == ["ant", "zebra"]
    assert forge.load_predictor(str(tmp_path / "b.forge")).classes == ["ant", "zebra"]


def test_integer_labels_without_classes_are_named_by_index(tmp_path):
    X, y = classification_data(120)
    r = forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", epochs=5)
    assert r.classes == ["0", "1"]


def test_explicit_classes_fix_the_order_for_names_and_for_indices(tmp_path):
    X, y = classification_data(120)
    by_index = forge.train_tabular_classifier(X, y, path=tmp_path / "a.forge", classes=["low", "high"], epochs=5)
    by_name = forge.train_tabular_classifier(
        X, np.where(y == 1, "high", "low"), path=tmp_path / "b.forge", classes=["low", "high"], epochs=5,
    )
    assert by_index.classes == by_name.classes == ["low", "high"]
    # evaluate() takes the same y the training call did
    predictor = forge.load_predictor(str(tmp_path / "b.forge"))
    assert predictor.evaluate(X, np.where(y == 1, "high", "low")).samples == len(X)


@pytest.mark.parametrize("cast", [lambda y: y.astype(np.float64), lambda y: y.astype(bool), lambda y: y.astype(np.int32),
                                  lambda y: y.reshape(-1, 1), lambda y: y.tolist()])
def test_labels_from_a_csv_reader_are_accepted(tmp_path, cast):
    X, y = classification_data(120)
    r = forge.train_tabular_classifier(X, cast(y), path=tmp_path / "m.forge", epochs=5)
    assert r.classes == ["0", "1"]


def test_multiclass(tmp_path):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 2))
    y = np.digitize(z[:, 0], [-0.5, 0.5])  # 3 classes
    r = forge.train_tabular_classifier(z, y, path=tmp_path / "m.forge", epochs=120)
    assert r.classes == ["0", "1", "2"] and r.validation_accuracy > r.baseline_accuracy
    assert forge.load_predictor(str(tmp_path / "m.forge")).evaluate(z, y).confusion_matrix.shape == (3, 3)


# --------------------------------------------------------------------------- preprocessing


def test_preprocessing_is_fitted_on_the_training_rows_only(cls_run):
    X, _, path, _ = cls_run
    train_idx, _ = split_indices(N, 3)
    normalize = _persisted_normalize(path)
    np.testing.assert_allclose(normalize.mean, X[train_idx].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(normalize.std, X[train_idx].std(axis=0), rtol=1e-5)
    assert not np.allclose(normalize.mean, X.mean(axis=0), rtol=1e-9)  # not the whole dataset's


def test_validation_rows_cannot_influence_the_fitted_preprocessing(tmp_path):
    X, y = classification_data(120)
    _, val_idx = split_indices(120, 0)
    poisoned = X.copy()
    poisoned[val_idx] = 1e9  # would move any statistic that looked at these rows
    forge.train_tabular_classifier(X, y, path=tmp_path / "a.forge", epochs=3, patience=None)
    forge.train_tabular_classifier(poisoned, y, path=tmp_path / "b.forge", epochs=3, patience=None)
    assert _persisted_normalize(tmp_path / "a.forge").mean == _persisted_normalize(tmp_path / "b.forge").mean


def test_a_constant_feature_is_centred_not_divided_by_zero(tmp_path):
    X, y = classification_data(120)
    X[:, 2] = 7.5
    r = forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", epochs=10)
    assert np.isfinite(r.train_loss) and _persisted_normalize(tmp_path / "m.forge").std[2] == 1.0
    predictions = forge.load_predictor(str(tmp_path / "m.forge")).predict(X[:3])
    assert all(np.isfinite(p.confidence) for p in predictions)


def test_removing_the_persisted_preprocessing_breaks_the_artifact(cls_run, tmp_path):
    """Mutation check: the same trained weights, saved without the preprocessing, must be worse."""
    X, y, path, _ = cls_run
    _, val_idx = split_indices(N, 3)
    with_preprocessing = forge.load_predictor(str(path)).evaluate(X[val_idx], y[val_idx]).accuracy
    stripped = tmp_path / "stripped.forge"
    forge.save_model(forge.load_model(str(path)), str(stripped), classes=["neg", "pos"], task="tabular_classification")
    without = forge.load_predictor(str(stripped)).evaluate(X[val_idx], y[val_idx]).accuracy
    assert with_preprocessing > 0.85
    assert without < with_preprocessing - 0.25


def test_missing_columns_are_imputed_from_the_training_split_and_persisted(tmp_path):
    X, y = classification_data(200)
    X[np.random.default_rng(4).random(200) < 0.25, 0] = 0.0  # 0 == "not measured" in column 0
    train_idx, _ = split_indices(200, 0)
    r = forge.train_tabular_classifier(
        X, y, path=tmp_path / "m.forge", missing_columns=[0], epochs=30, patience=None,
    )
    replace = forge.load_preprocessing(str(tmp_path / "m.forge")).transforms[0]
    assert type(replace).__name__ == "ReplaceValue" and replace.columns == [0] and replace.sentinel == 0.0
    column = X[train_idx, 0]
    assert replace.fill == pytest.approx([float(np.median(column[column != 0.0]))], rel=1e-6)
    # a raw row carrying the sentinel predicts like the same row carrying the fill value
    predictor = forge.load_predictor(str(tmp_path / "m.forge"))
    with_sentinel, with_fill = X[:1].copy(), X[:1].copy()
    with_sentinel[0, 0], with_fill[0, 0] = 0.0, replace.fill[0]
    assert predictor.predict(with_sentinel)[0].confidence == pytest.approx(predictor.predict(with_fill)[0].confidence)
    assert r.epochs_completed == 30


def test_a_missing_column_with_no_real_values_in_training_is_rejected(tmp_path, no_training):
    X, y = classification_data(100)
    X[:, 1] = 0.0
    with pytest.raises(forge.DataError, match="column 1"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", missing_columns=[1])


# --------------------------------------------------------------------------- custom model


def test_custom_model_is_used_as_given(tmp_path):
    X, y = classification_data(120)
    forge.random.seed(0)
    mine = Sequential(Linear(3, 6), ReLU(), Linear(6, 2))
    r = forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", model=mine, epochs=15, patience=None)
    assert r.history.model is mine and r.epochs_completed == 15
    assert forge.load_predictor(str(tmp_path / "m.forge")).evaluate(X, y).samples == 120


def test_model_with_the_wrong_number_of_outputs_fails_before_epoch_1(tmp_path, no_training):
    X, y = classification_data(100)
    five_outputs = Sequential(Linear(3, 8), ReLU(), Linear(8, 5))
    with pytest.raises(forge.TrainerError, match=r"output of shape \(2, 5\).*needs \(batch, 2\)"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", model=five_outputs)
    assert list(tmp_path.iterdir()) == []


def test_model_with_the_wrong_input_width_fails_before_epoch_1(tmp_path, no_training):
    X, y = classification_data(100)
    with pytest.raises(forge.TrainerError, match="could not process"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", model=Sequential(Linear(7, 4), Linear(4, 2)))


def test_a_non_module_model_is_rejected(tmp_path, no_training):
    X, y = classification_data(100)
    with pytest.raises(forge.TrainerError, match="forge.nn.Module"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", model="not a model")


def test_a_model_that_cannot_be_serialised_fails_before_epoch_1(tmp_path, no_training):
    class Unregistered(Module):
        def __init__(self):
            super().__init__()
            self.fc = Linear(3, 2)

        def forward(self, x):
            return self.fc(x)

    X, y = classification_data(100)
    with pytest.raises(forge.PersistenceError):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", model=Unregistered())
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- classification: invalid input


def _bad(x=None, y=None, **kwargs):
    X, Y = classification_data(100)
    return dict(X=X if x is None else x, y=Y if y is None else y, **kwargs)


@pytest.mark.parametrize("name, kwargs, match", [
    ("X 1-D", lambda: _bad(x=np.zeros(100)), "2-D"),
    ("X 3-D", lambda: _bad(x=np.zeros((100, 3, 1))), "2-D"),
    ("X strings", lambda: _bad(x=np.array([["a", "b"]] * 100)), "numeric"),
    ("X ragged", lambda: _bad(x=[[1.0, 2.0], [3.0]]), "could not interpret X"),
    ("X empty", lambda: _bad(x=np.zeros((0, 3)), y=np.zeros(0, int)), "non-empty"),
    ("X no columns", lambda: _bad(x=np.zeros((100, 0))), "non-empty"),
    ("more rows than labels", lambda: _bad(y=np.zeros(99, int)), "100 sample.*99"),
    ("fewer rows than labels", lambda: _bad(x=np.zeros((90, 3))), "90 sample.*100"),
    ("single class", lambda: _bad(y=np.zeros(100, int)), "at least 2 classes"),
    ("2-D labels", lambda: _bad(y=np.zeros((100, 2), int)), "1-D"),
    ("continuous labels", lambda: _bad(y=np.linspace(0.0, 1.0, 100)), "train_tabular_regressor"),
    ("indices not 0..K-1", lambda: _bad(y=classification_data(100)[1] + 1), "0..K-1"),
    ("negative indices", lambda: _bad(y=classification_data(100)[1] - 1), "0..K-1"),
    ("mixed labels", lambda: _bad(y=np.array(["a", 1] * 50, dtype=object)), "names \\(str\\) or integer"),
    ("NaN label", lambda: _bad(y=np.where(np.arange(100) == 4, np.nan, 1.0)), "non-finite"),
    ("labels outside classes=", lambda: _bad(y=np.where(np.arange(100) % 2, "cat", "dog"), classes=["cat", "cow"]), "not among"),
    ("unused class in classes=", lambda: _bad(classes=["a", "b", "c"]), "no rows in y"),
    ("duplicate classes=", lambda: _bad(classes=["a", "a"]), "duplicates"),
    ("blank class name", lambda: _bad(classes=["a", " "]), "non-empty strings"),
    ("classes= as a string", lambda: _bad(classes="ab"), "list of non-empty strings"),
    ("two rows", lambda: _bad(x=np.zeros((2, 3)), y=np.array([0, 1])), "at least 3 samples"),
])
def test_invalid_data_is_rejected_before_epoch_1(tmp_path, no_training, name, kwargs, match):
    with pytest.raises(forge.DataError, match=match):
        forge.train_tabular_classifier(path=tmp_path / "m.forge", **kwargs())
    assert list(tmp_path.iterdir()) == [], f"{name}: nothing may be written"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_features_never_reach_training(tmp_path, no_training, bad):
    X, y = classification_data(100)
    X[7, 1] = bad
    with pytest.raises(forge.DataError, match=r"non-finite.*first at index \(7, 1\)"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge")


def test_a_float64_overflow_to_float32_is_treated_as_non_finite(tmp_path, no_training):
    X, y = classification_data(100)
    X[0, 0] = 1e300  # finite as float64, inf as float32
    with pytest.raises(forge.DataError, match="non-finite"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge")


def test_a_class_with_no_training_rows_is_rejected_before_epoch_1(tmp_path, no_training):
    X, y = classification_data(N)
    names = np.where(y == 1, "b", "a").astype(object)
    row = next(r for r in range(N) if r in set(split_indices(N, 0)[1]))  # a row that lands in validation
    names[:] = "a"
    names[row] = "rare"  # the only 'rare' row is a validation row
    with pytest.raises(forge.DataError, match=r"'rare'.*no rows in the training split"):
        forge.train_tabular_classifier(X, names.astype(str), path=tmp_path / "m.forge")


# --------------------------------------------------------------------------- shared argument + path validation


@pytest.mark.parametrize("kwargs, match", [
    ({"epochs": 0}, "epochs"), ({"epochs": 2.5}, "epochs"), ({"epochs": True}, "epochs"),
    ({"batch_size": 0}, "batch_size"), ({"learning_rate": 0.0}, "learning_rate"),
    ({"learning_rate": float("nan")}, "learning_rate"), ({"learning_rate": -1e-3}, "learning_rate"),
    ({"val_fraction": 0.0}, "val_fraction"), ({"val_fraction": 1.0}, "val_fraction"),
    ({"patience": 0}, "patience"), ({"patience": 1.5}, "patience"), ({"seed": -1}, "seed"),
    ({"missing_columns": []}, "missing_columns"), ({"missing_columns": [0, 0]}, "unique"),
    ({"missing_columns": [9]}, "out of range"), ({"missing_columns": 3}, "missing_columns"),
    ({"missing_value": float("nan")}, "missing_value"),
])
@pytest.mark.parametrize("workflow", ["classifier", "regressor"])
def test_bad_arguments_are_rejected_before_epoch_1(tmp_path, no_training, workflow, kwargs, match):
    X, y = classification_data(100)
    train = getattr(forge, f"train_tabular_{workflow}")
    with pytest.raises(forge.DataError, match=match):
        train(X, y, path=tmp_path / "m.forge", **kwargs)


@pytest.mark.parametrize("workflow", ["classifier", "regressor"])
class TestOutputPath:
    def _train(self, workflow, path):
        X, y = classification_data(100)
        return getattr(forge, f"train_tabular_{workflow}")(X, y, path=path)

    def test_missing_directory(self, tmp_path, no_training, workflow):
        with pytest.raises(forge.PersistenceError, match="does not exist"):
            self._train(workflow, tmp_path / "nope" / "m.forge")
        assert list(tmp_path.iterdir()) == []

    def test_path_is_a_directory(self, tmp_path, no_training, workflow):
        with pytest.raises(forge.PersistenceError, match="is a directory"):
            self._train(workflow, tmp_path)

    def test_parent_is_a_file(self, tmp_path, no_training, workflow):
        (tmp_path / "file.txt").write_text("x")
        with pytest.raises(forge.PersistenceError):
            self._train(workflow, tmp_path / "file.txt" / "m.forge")

    @pytest.mark.skipif(os.name != "nt", reason="'|' is only an invalid filename character on Windows")
    def test_invalid_filename(self, tmp_path, no_training, workflow):
        with pytest.raises(forge.PersistenceError):
            self._train(workflow, tmp_path / "a|b.forge")
        assert list(tmp_path.iterdir()) == []

    def test_empty_path(self, tmp_path, no_training, workflow):
        with pytest.raises(forge.DataError, match="path"):
            self._train(workflow, "")

    def test_str_and_pathlike_paths_both_work(self, tmp_path, workflow):
        X, y = classification_data(80)
        train = getattr(forge, f"train_tabular_{workflow}")
        assert train(X, y, path=str(tmp_path / "s.forge"), epochs=2, patience=None).artifact_path == str(tmp_path / "s.forge")
        assert train(X, y, path=tmp_path / "p.forge", epochs=2, patience=None).artifact_path == str(tmp_path / "p.forge")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the weights overflow on purpose
def test_diverging_training_fails_loudly_instead_of_reporting_nan(tmp_path):
    """Finite data, absurd learning rate: `Trainer` stops at the first non-finite loss."""
    X, y = classification_data(100)
    with pytest.raises(forge.TrainerError, match="(?i)non-finite"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge", learning_rate=1e30, epochs=20, patience=None)
    assert not (tmp_path / "m.forge").exists()


# --------------------------------------------------------------------------- regression


def test_regressor_trains_saves_and_beats_the_baseline(reg_run):
    X, y, path, r = reg_run
    assert isinstance(r, forge.TabularRegressionResult)
    assert (r.task, r.samples, r.features, r.outputs) == ("regression", N, 3, 1)
    assert (r.train_samples, r.validation_samples) == (240, 60)
    assert r.artifact_path == str(path) and path.is_file() and 1 <= r.epochs_completed <= 300
    for value in (r.train_loss, r.validation_loss, r.train_mse, r.validation_mse, r.train_mae,
                  r.validation_mae, r.baseline_mse):
        assert np.isfinite(value)
    assert r.validation_mse < 0.1 * r.baseline_mse  # the unscaled target (~60 +- 12) is learned
    assert r.validation_loss == pytest.approx(r.validation_mse, rel=1e-4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.validation_mse = 0.0


def test_regressor_result_numbers_are_the_evaluate_numbers_of_the_saved_artifact(reg_run):
    X, y, path, r = reg_run
    train_idx, val_idx = split_indices(N, 3)
    predictor = forge.load_predictor(str(path))
    val = predictor.evaluate(X[val_idx], y[val_idx])
    train = predictor.evaluate(X[train_idx], y[train_idx])
    assert (r.validation_mse, r.validation_mae) == (val.mse, val.mae)
    assert (r.train_mse, r.train_mae) == (train.mse, train.mae)
    assert r.baseline_mse == np.mean((y[val_idx] - y[val_idx].mean()) ** 2)  # M113: predict-the-validation-mean
    # ... and they are what NumPy says about the artifact's own predictions
    predicted = predictor.predict(X[val_idx]).numpy()[:, 0]
    assert val.mse == pytest.approx(np.mean((predicted - y[val_idx]) ** 2), rel=1e-4)
    assert val.mae == pytest.approx(np.mean(np.abs(predicted - y[val_idx])), rel=1e-4)


def test_regression_artifact_predicts_in_the_targets_own_units(reg_run):
    X, y, path, _ = reg_run
    predictor = forge.load_predictor(str(path))
    assert predictor.task == "regression" and predictor.classes is None
    prediction = predictor.predict(X[:5])
    assert tuple(prediction.shape) == (5, 1)
    assert np.abs(prediction.numpy()[:, 0] - y[:5]).max() < 5.0  # targets are ~60, not standardised
    assert prediction.numpy().mean() > 40.0
    assert forge.predict_model(str(path), X[:5]).numpy().tolist() == prediction.numpy().tolist()


def test_targets_are_never_transformed(reg_run):
    _, _, path, _ = reg_run
    steps = forge.load_preprocessing(str(path)).transforms
    assert [type(s).__name__ for s in steps] == ["Normalize"]  # features only; no target scaler exists


def test_regression_preprocessing_is_fitted_on_the_training_rows_only(reg_run):
    X, _, path, _ = reg_run
    train_idx, _ = split_indices(N, 3)
    np.testing.assert_allclose(_persisted_normalize(path).mean, X[train_idx].mean(axis=0), rtol=1e-6)


def test_regression_without_the_persisted_preprocessing_is_broken(reg_run, tmp_path):
    X, y, path, _ = reg_run
    _, val_idx = split_indices(N, 3)
    good = forge.load_predictor(str(path)).evaluate(X[val_idx], y[val_idx]).mse
    stripped = tmp_path / "stripped.forge"
    forge.save_model(forge.load_model(str(path)), str(stripped), task="regression")
    bad = forge.load_predictor(str(stripped)).evaluate(X[val_idx], y[val_idx]).mse
    assert bad > 20 * good


@pytest.mark.parametrize("shape", ["flat", "column"])
def test_1d_and_column_targets_are_equivalent(tmp_path, shape):
    X, y = regression_data(120)
    r = forge.train_tabular_regressor(X, y if shape == "flat" else y.reshape(-1, 1), path=tmp_path / "m.forge",
                                      epochs=5, patience=None)
    assert r.outputs == 1


def test_multi_output_targets(tmp_path):
    X, y = regression_data(150)
    Y = np.stack([y, -y / 2.0], axis=1)
    r = forge.train_tabular_regressor(X, Y, path=tmp_path / "m.forge", epochs=80)
    assert r.outputs == 2
    predictor = forge.load_predictor(str(tmp_path / "m.forge"))
    assert tuple(predictor.predict(X[:4]).shape) == (4, 2)
    assert predictor.evaluate(X, Y).samples == 150


def test_regressor_custom_model(tmp_path):
    X, y = regression_data(120)
    mine = Sequential(Linear(3, 4), ReLU(), Linear(4, 1))
    r = forge.train_tabular_regressor(X, y, path=tmp_path / "m.forge", model=mine, epochs=10, patience=None)
    assert r.history.model is mine


@pytest.mark.parametrize("outputs, targets", [(2, 1), (1, 2), (3, 1)])
def test_regressor_model_output_must_match_the_target_width(tmp_path, no_training, outputs, targets):
    X, y = regression_data(100)
    Y = np.stack([y] * targets, axis=1)
    wrong = Sequential(Linear(3, 8), ReLU(), Linear(8, outputs))
    with pytest.raises(forge.TrainerError, match=rf"needs \(batch, {targets}\)"):
        forge.train_tabular_regressor(X, Y, path=tmp_path / "m.forge", model=wrong)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name, kwargs, error, match", [
    ("X 1-D", lambda X, y: (np.zeros(100), y), forge.DataError, "2-D"),
    ("X strings", lambda X, y: (np.array([["a", "b"]] * 100), y), forge.DataError, "numeric"),
    ("X NaN", lambda X, y: (np.where(np.arange(300)[:, None] == 9, np.nan, X), y), forge.DataError, "non-finite"),
    ("X Inf", lambda X, y: (np.where(np.arange(300)[:, None] == 9, np.inf, X), y), forge.DataError, "non-finite"),
    ("y NaN", lambda X, y: (X, np.where(np.arange(300) == 5, np.nan, y)), forge.DataError, r"y contains 1 non-finite.*\(5,\)"),
    ("y Inf", lambda X, y: (X, np.where(np.arange(300) == 5, -np.inf, y)), forge.DataError, "non-finite"),
    ("y 3-D", lambda X, y: (X, y.reshape(-1, 1, 1)), forge.DataError, r"\(n,\) or \(n, outputs\)"),
    ("y strings", lambda X, y: (X, np.array(["a"] * 300)), forge.DataError, "numeric"),
    ("y bool", lambda X, y: (X, y > 300), forge.DataError, "numeric"),
    ("y too short", lambda X, y: (X, y[:10]), forge.DataError, "300 sample.*10"),
    ("y empty", lambda X, y: (X[:0], y[:0]), forge.DataError, "non-empty|no targets"),
    ("y float64 overflow", lambda X, y: (X, np.where(np.arange(300) == 1, 1e300, y)), forge.DataError, "non-finite"),
])
def test_regressor_invalid_data_is_rejected_before_epoch_1(tmp_path, no_training, name, kwargs, error, match):
    X, y = regression_data()
    X, y = kwargs(X, y)
    with pytest.raises(error, match=match):
        forge.train_tabular_regressor(X, y, path=tmp_path / "m.forge")
    assert list(tmp_path.iterdir()) == [], f"{name}: nothing may be written"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the weights overflow on purpose
def test_regressor_diverging_training_fails_loudly(tmp_path):
    X, y = regression_data(100)
    with pytest.raises(forge.TrainerError, match="(?i)non-finite"):
        forge.train_tabular_regressor(X, y, path=tmp_path / "m.forge", learning_rate=1e30, epochs=20, patience=None)


# --------------------------------------------------------------------------- inputs of other kinds


def test_lists_and_tensors_are_accepted_as_inputs(tmp_path):
    X, y = classification_data(80)
    a = forge.train_tabular_classifier(X.tolist(), y.tolist(), path=tmp_path / "a.forge", epochs=3, patience=None)
    b = forge.train_tabular_classifier(forge.Tensor(X.astype(np.float32)), y, path=tmp_path / "b.forge", epochs=3, patience=None)
    assert a.samples == b.samples == 80


def test_integer_features_are_accepted(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 100, size=(100, 3))
    r = forge.train_tabular_classifier(X, (X[:, 0] > 50).astype(int), path=tmp_path / "m.forge", epochs=5)
    assert r.features == 3


# --------------------------------------------------------------------------- the artifact's input boundary (found by M114's CUDA run)


@pytest.mark.parametrize("cast", [np.float64, np.float32, np.int64, list])
def test_predict_and_evaluate_accept_the_dtypes_a_csv_reader_gives(cls_run, cast):
    X, y, path, _ = cls_run
    predictor = forge.load_predictor(str(path))
    rows = X[:6].round(0)  # exactly representable in every dtype, so all casts agree
    expected = [p.index for p in predictor.predict(rows.astype(np.float32))]
    given = rows.astype(cast) if cast is not list else rows.tolist()
    assert [p.index for p in predictor.predict(given)] == expected
    assert predictor.evaluate(given, y[:6]).samples == 6


def test_regression_predictions_are_float32_whatever_the_input_dtype(reg_run):
    X, _, path, _ = reg_run
    predictor = forge.load_predictor(str(path))
    reference = predictor.predict(X[:4].astype(np.float32)).numpy()
    for given in (X[:4].astype(np.float64), X[:4].round(0).astype(np.int64)):
        out = predictor.predict(given)
        assert out.dtype == forge.DEFAULT_DTYPE
    np.testing.assert_allclose(predictor.predict(X[:4].astype(np.float64)).numpy(), reference, rtol=1e-6)


# --------------------------------------------------------------------------- fresh process + CLI


_FRESH = """
import json, sys
import numpy as np
import forge
d = np.load(sys.argv[2])
p = forge.load_predictor(sys.argv[1])
first = p.predict(d["X"][:5])
ev = p.evaluate(d["X"], d["y"])
if p.task == "regression":
    print(json.dumps({"task": p.task, "pred": first.numpy()[:, 0].tolist(), "mse": ev.mse, "mae": ev.mae, "base": ev.baseline_mse}))
else:
    print(json.dumps({"task": p.task, "labels": [r.label for r in first], "acc": ev.accuracy, "base": ev.baseline_accuracy,
                      "conf": ev.confusion_matrix.tolist()}))
"""


def _fresh_process(path, X, y, tmp_path):
    data = tmp_path / "eval.npz"
    np.savez(data, X=X, y=y)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    out = subprocess.run([sys.executable, "-c", _FRESH, str(path), str(data)], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_classification_artifact_in_a_fresh_process(cls_run, tmp_path):
    X, y, path, _ = cls_run
    held_X, held_y = classification_data(150, seed=42)
    fresh = _fresh_process(path, held_X, held_y, tmp_path)
    predictor = forge.load_predictor(str(path))
    here = predictor.evaluate(held_X, held_y)
    assert fresh["task"] == "tabular_classification"
    assert fresh["labels"] == [p.label for p in predictor.predict(held_X[:5])]
    assert fresh["acc"] == here.accuracy and fresh["base"] == here.baseline_accuracy
    assert fresh["conf"] == here.confusion_matrix.tolist()


def test_regression_artifact_in_a_fresh_process(reg_run, tmp_path):
    X, y, path, _ = reg_run
    held_X, held_y = regression_data(150, seed=42)
    fresh = _fresh_process(path, held_X, held_y, tmp_path)
    predictor = forge.load_predictor(str(path))
    here = predictor.evaluate(held_X, held_y)
    np.testing.assert_allclose(fresh["pred"], predictor.predict(held_X[:5]).numpy()[:, 0], rtol=1e-6)
    assert (fresh["mse"], fresh["mae"], fresh["base"]) == (here.mse, here.mae, here.baseline_mse)


def test_existing_cli_reads_the_new_artifacts(cls_run, reg_run, tmp_path, capsys):
    Xc, _, cls_path, _ = cls_run
    Xr, _, reg_path, _ = reg_run
    row = tmp_path / "row.json"
    row.write_text(json.dumps(Xc[0].tolist()))
    assert cli_main(["model", "inspect", str(cls_path)]) == 0
    out = capsys.readouterr().out
    assert "tabular_classification" in out and "neg, pos" in out and "Normalize" in out
    assert cli_main(["model", "predict", str(cls_path), str(row)]) == 0
    assert "Prediction:" in capsys.readouterr().out
    assert cli_main(["model", "inspect", str(reg_path)]) == 0
    assert "Task: regression" in capsys.readouterr().out
    row.write_text(json.dumps(Xr[0].tolist()))
    assert cli_main(["model", "predict", str(reg_path), str(row)]) == 0
    assert capsys.readouterr().out.startswith("Prediction:")
