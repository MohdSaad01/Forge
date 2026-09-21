"""Milestone 119 tests: `feature_names=` on the tabular training functions.

Names are identity metadata about the *raw* input columns. Training them in must therefore change exactly
one thing about the artifact -- a `"feature_names"` entry -- and nothing about the model, the fitted
preprocessing or any reported number; leaving them out must leave the artifact unnamed (never
`feature_0, feature_1, ...`); and a bad list must be refused before epoch 1 with nothing written.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.exceptions import DataError, PersistenceError
from forge.nn import Flatten, Linear, ReLU, Sequential
from forge.serialization import inspect_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema_support import (  # noqa: E402
    CLASSES, FEATURES, make_data, metadata_of, write_named_csv,
)

X, Y_REG, Y_CLS = make_data()
COMMON = dict(seed=3, epochs=8, patience=None)


def train(kind, path, **kwargs):
    if kind == "regression":
        return forge.train_tabular_regressor(X, Y_REG, path=str(path), **COMMON, **kwargs)
    return forge.train_tabular_classifier(X, Y_CLS, path=str(path), classes=CLASSES, **COMMON, **kwargs)


def state_sha(path) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(forge.load_model(str(path)).named_parameters(), key=lambda kv: kv[0]):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(parameter.numpy()).tobytes())
    return digest.hexdigest()


KINDS = ["regression", "tabular_classification"]


@pytest.mark.parametrize("kind", KINDS)
def test_names_are_recorded_in_the_artifact(tmp_path, kind):
    train(kind, tmp_path / "m.forge", feature_names=FEATURES)
    assert metadata_of(tmp_path / "m.forge")["feature_names"] == FEATURES
    assert forge.load_predictor(str(tmp_path / "m.forge")).input_schema.feature_names == tuple(FEATURES)


@pytest.mark.parametrize("kind", KINDS)
def test_arrays_alone_produce_an_unnamed_artifact_with_nothing_invented(tmp_path, kind):
    train(kind, tmp_path / "m.forge")
    assert "feature_names" not in metadata_of(tmp_path / "m.forge")
    assert inspect_model(str(tmp_path / "m.forge")).input_schema.feature_names is None


@pytest.mark.parametrize("kind", KINDS)
def test_names_change_nothing_but_the_names_entry(tmp_path, kind):
    """Same seed with and without names: identical parameters, identical preprocessing, identical numbers."""
    a = train(kind, tmp_path / "named.forge", feature_names=FEATURES)
    b = train(kind, tmp_path / "plain.forge")
    assert state_sha(tmp_path / "named.forge") == state_sha(tmp_path / "plain.forge")
    md_named, md_plain = metadata_of(tmp_path / "named.forge"), metadata_of(tmp_path / "plain.forge")
    assert {k: v for k, v in md_named.items() if k != "feature_names"} == md_plain
    for field in ("train_loss", "validation_loss", "epochs_completed", "samples", "features"):
        assert getattr(a, field) == getattr(b, field), field


def test_names_and_a_target_transform_train_the_same_model_as_the_target_transform_alone(tmp_path):
    train("regression", tmp_path / "a.forge", feature_names=FEATURES, target_transform="standardize")
    train("regression", tmp_path / "b.forge", target_transform="standardize")
    assert state_sha(tmp_path / "a.forge") == state_sha(tmp_path / "b.forge")
    assert metadata_of(tmp_path / "a.forge")["target_transform"] == metadata_of(tmp_path / "b.forge")["target_transform"]


@pytest.mark.parametrize("kind", KINDS)
def test_names_are_positional_documentation_not_a_sort_key(tmp_path, kind):
    """Given in reverse, they are stored in reverse: order is authoritative and is the caller's."""
    reverse = FEATURES[::-1]
    train(kind, tmp_path / "m.forge", feature_names=reverse)
    assert metadata_of(tmp_path / "m.forge")["feature_names"] == reverse


@pytest.mark.parametrize("kind", KINDS)
def test_a_custom_model_records_names_when_its_input_width_is_readable(tmp_path, kind):
    outputs = 1 if kind == "regression" else 2
    forge.random.seed(0)
    model = Sequential(Linear(4, 6), ReLU(), Linear(6, outputs))
    train(kind, tmp_path / "m.forge", feature_names=FEATURES, model=model)
    assert forge.load_predictor(str(tmp_path / "m.forge")).input_schema.feature_names == tuple(FEATURES)


# ------------------------------------------------------------------------------------------ refused before epoch 1


@pytest.fixture()
def no_training(monkeypatch):
    """Fail the test if a run gets as far as training: every refusal below must happen first."""
    import forge.training.tabular as tabular

    def boom(*args, **kwargs):
        raise AssertionError("training started although feature_names= should have been refused first")

    monkeypatch.setattr(tabular, "train_and_save", boom)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("names, message", [
    (["a", "b", "c"], r"feature_names has 3 name\(s\) but X has 4 feature column\(s\)"),
    (["a", "b", "c", "d", "e"], r"feature_names has 5 name\(s\) but X has 4 feature column\(s\)"),
    (["a", "b", "b", "c"], "must be unique"),
    (["a", "b", "c", ""], "empty or blank name"),
    (["a", "b", "c", "   "], "empty or blank name"),
    (["a", "b", "c", 4], "only strings"),
    ("abcd", "must be a list of strings"),
    ([], "must not be empty"),
    (["Age", "age", "AGE", "Age "], None),                       # four distinct names: valid, NOT an error
])
def test_a_bad_name_list_is_refused_before_training_with_nothing_written(tmp_path, no_training, kind, names, message):
    if message is None:                                          # the case/whitespace-only-different list is fine ...
        with pytest.raises(AssertionError, match="training started"):    # ... it got past validation to training
            train(kind, tmp_path / "m.forge", feature_names=names)
        return
    fn = "train_tabular_regressor" if kind == "regression" else "train_tabular_classifier"
    with pytest.raises(DataError, match=rf"^{fn}\(\) .*{message}"):
        train(kind, tmp_path / "m.forge", feature_names=names)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", KINDS)
def test_names_with_a_model_whose_input_width_is_not_readable_are_refused_before_epoch_1(tmp_path, no_training, kind):
    outputs = 1 if kind == "regression" else 2
    model = Sequential(Flatten(), Linear(4, outputs))
    with pytest.raises(PersistenceError, match="input width Forge can read"):
        train(kind, tmp_path / "m.forge", feature_names=FEATURES, model=model)
    assert list(tmp_path.iterdir()) == []                        # the preflight's temporary file is gone too


def test_the_same_custom_model_still_trains_unnamed(tmp_path):
    forge.random.seed(0)
    train("regression", tmp_path / "m.forge", model=Sequential(Flatten(), Linear(4, 1)))
    assert inspect_model(str(tmp_path / "m.forge")).input_schema is None      # width not derivable; unchanged M114 behaviour


# ------------------------------------------------------------------------------------------ CSV-derived names


def test_csv_header_names_reach_the_artifact_and_the_target_is_not_one_of_them(tmp_path):
    csv_path = write_named_csv(tmp_path / "d.csv", FEATURES, X, target=("price", Y_REG))
    Xc, yc, names = forge.data.load_csv(csv_path, target="price", return_feature_names=True)
    assert names == FEATURES
    forge.train_tabular_regressor(Xc, yc, path=str(tmp_path / "m.forge"), feature_names=names, **COMMON)
    assert metadata_of(tmp_path / "m.forge")["feature_names"] == FEATURES        # 4, not 5: no "price"


def test_a_target_in_the_middle_of_the_header_is_excluded_and_the_others_keep_file_order(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("age,target,income,rooms,distance\n" + "\n".join(
        f"{float(r[1])!r},{float(y)!r},{float(r[0])!r},{float(r[2])!r},{float(r[3])!r}"
        for r, y in zip(X[:20], Y_REG[:20])) + "\n")
    _, _, names = forge.data.load_csv(path, target="target", return_feature_names=True)
    assert names == ["age", "income", "rooms", "distance"]


def test_classifier_from_a_csv_records_the_header_names(tmp_path):
    csv_path = write_named_csv(tmp_path / "d.csv", FEATURES, X, target=("label", Y_CLS))
    Xc, yc, names = forge.data.load_csv(csv_path, target="label", labels=True, return_feature_names=True)
    forge.train_tabular_classifier(Xc, yc, path=str(tmp_path / "m.forge"), classes=CLASSES, feature_names=names, **COMMON)
    assert metadata_of(tmp_path / "m.forge")["feature_names"] == FEATURES


def test_a_named_csv_and_the_same_arrays_train_the_same_model(tmp_path):
    """Named CSV == unnamed arrays in every weight (M118's CSV == NumPy property survives the names)."""
    csv_path = write_named_csv(tmp_path / "d.csv", FEATURES, X, target=("price", Y_REG))
    Xc, yc, names = forge.data.load_csv(csv_path, target="price", return_feature_names=True)
    forge.train_tabular_regressor(Xc, yc, path=str(tmp_path / "csv.forge"), feature_names=names, **COMMON)
    forge.train_tabular_regressor(X, Y_REG, path=str(tmp_path / "np.forge"), **COMMON)
    assert state_sha(tmp_path / "csv.forge") == state_sha(tmp_path / "np.forge")


# ------------------------------------------------------------------------------------------ train_and_save / save_and_verify


def test_save_and_verify_records_and_reads_names_back(tmp_path):
    from forge.data import Normalize  # noqa: F401  (only to show preprocessing is independent)

    model = Sequential(Linear(4, 3), ReLU(), Linear(3, 1))
    sample = forge.Tensor(X[:2].astype(np.float32))
    forge.save_and_verify(model, str(tmp_path / "m.forge"), sample, task="regression", feature_names=FEATURES)
    assert metadata_of(tmp_path / "m.forge")["feature_names"] == FEATURES


def test_save_and_verify_fails_loudly_if_the_names_did_not_survive_the_save(tmp_path, monkeypatch):
    """A save that lost the names must not be discovered later as a reordered CSV being accepted unchecked."""
    import forge.serialization.model as model_module

    real = model_module.save_model

    def lossy(model, path, **kwargs):
        kwargs.pop("feature_names", None)
        return real(model, path, **kwargs)

    monkeypatch.setattr(model_module, "save_model", lossy)
    model = Sequential(Linear(4, 3), ReLU(), Linear(3, 1))
    with pytest.raises(PersistenceError, match=r"feature names read back from .* \(None\) differ"):
        forge.save_and_verify(model, str(tmp_path / "m.forge"), forge.Tensor(X[:2].astype(np.float32)),
                              task="regression", feature_names=FEATURES)


def test_train_and_save_passes_names_through(tmp_path):
    from forge.data import DataLoader, TensorDataset
    from forge.nn import MSELoss
    from forge.optim import Adam

    forge.random.seed(0)
    model = Sequential(Linear(4, 3), ReLU(), Linear(3, 1))
    data = TensorDataset(forge.Tensor(X.astype(np.float32)), forge.Tensor(Y_REG.astype(np.float32).reshape(-1, 1)))
    forge.train_and_save(
        model, DataLoader(data, batch_size=32), loss=MSELoss(), optimizer=Adam(model.parameters(), lr=1e-3),
        epochs=1, path=str(tmp_path / "m.forge"), sample=forge.Tensor(X[:1].astype(np.float32)), verbose=False,
        task="regression", feature_names=FEATURES,
    )
    assert forge.inspect_model(str(tmp_path / "m.forge")).input_schema.feature_names == tuple(FEATURES)
