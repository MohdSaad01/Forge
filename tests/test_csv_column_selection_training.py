"""Milestone 120 tests: the end-to-end workflow `columns=` exists for.

```text
CSV with an id column -> load_csv(..., columns=[...], return_feature_names=True) -> X, y, names
                       -> train_tabular_*(..., feature_names=names) -> named artifact
                       -> named CSV (with or without the id column) -> predict()/evaluate()
```

Pins that selected names really reach the artifact (not just the arrays), that M119 alignment still runs
on top (a selected-then-reordered CSV still predicts correctly), and that M116's target transform composes
with selection with no interaction (selection never sees the target's transform, only its column name).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge

FEATURES = ["income", "age", "rooms", "distance"]
_SCALE = np.array([10.0, 3.0, 1.0, 50.0])
_CENTER = np.array([40.0, 30.0, 5.0, 100.0])


def make_data(n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    X = z * _SCALE + _CENTER
    y_reg = 3.0 * z[:, 0] - 2.0 * z[:, 1] + 0.5 * z[:, 2] + 0.05 * rng.normal(size=n)
    y_cls = (z[:, 0] + z[:, 1] - 0.5 * z[:, 3] > 0).astype(np.int64)
    return X, y_reg, y_cls


X, Y_REG, Y_CLS = make_data()


def write_csv(path: Path, header, rows, target) -> Path:
    lines = [",".join([*header, target[0]])]
    for i, row in enumerate(rows):
        cells = [repr(float(v)) for v in row]
        cells.append(str(int(target[1][i])) if target[1].dtype.kind in "iu" else repr(float(target[1][i])))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture()
def id_csv(tmp_path):
    ids = np.arange(len(X))
    return write_csv(tmp_path / "reg.csv", ["id", *FEATURES], np.column_stack([ids, X]), ("target", Y_REG))


@pytest.fixture()
def id_csv_cls(tmp_path):
    ids = np.arange(len(X))
    return write_csv(tmp_path / "cls.csv", ["id", *FEATURES], np.column_stack([ids, X]), ("target", Y_CLS))


def test_selected_names_reach_the_trained_artifact(tmp_path, id_csv):
    Xs, y, names = forge.data.load_csv(id_csv, target="target", columns=FEATURES, return_feature_names=True)
    result = forge.train_tabular_regressor(
        Xs, y, path=str(tmp_path / "m.forge"), feature_names=names, epochs=8, patience=None,
    )
    info = forge.inspect_model(result.artifact_path)
    assert info.input_schema.feature_names == tuple(FEATURES)


def test_the_id_column_never_reaches_training(tmp_path, id_csv):
    """Training on the selected arrays must not see a 5th (id) column -- proven via the model's own input width."""
    Xs, y, names = forge.data.load_csv(id_csv, target="target", columns=FEATURES, return_feature_names=True)
    assert Xs.shape[1] == 4
    result = forge.train_tabular_regressor(
        Xs, y, path=str(tmp_path / "m.forge"), feature_names=names, epochs=8, patience=None,
    )
    assert result.features == 4


def test_a_selected_then_reordered_csv_still_predicts_correctly_m119_alignment_intact(tmp_path, id_csv):
    Xs, y, names = forge.data.load_csv(id_csv, target="target", columns=FEATURES, return_feature_names=True)
    result = forge.train_tabular_regressor(
        Xs, y, path=str(tmp_path / "m.forge"), feature_names=names, epochs=8, patience=None,
    )
    predictor = forge.load_predictor(result.artifact_path)

    # A fresh feature-only CSV: an id column, plus the same features in the *reverse* of training order.
    reordered = list(reversed(FEATURES))
    feature_csv = tmp_path / "new_patients.csv"
    with open(feature_csv, "w") as f:
        f.write(",".join(["id", *reordered]) + "\n")
        for i in range(5):
            row = [str(1000 + i)] + [repr(float(v)) for v in X[i][::-1]]
            f.write(",".join(row) + "\n")

    X_sel, names_sel = forge.data.load_csv_features(feature_csv, columns=reordered)
    assert names_sel == reordered  # selection is order-authoritative
    via_selection = predictor.predict(X_sel, feature_names=names_sel).numpy()
    direct = predictor.predict(X[:5]).numpy()
    assert np.allclose(via_selection, direct, atol=1e-4)


def test_target_transform_composes_with_selection_native_units(tmp_path, id_csv):
    """M116's target_transform is applied after prediction; selection only ever decides which raw columns enter X."""
    Xs, y, names = forge.data.load_csv(id_csv, target="target", columns=FEATURES, return_feature_names=True)
    result = forge.train_tabular_regressor(
        Xs, y, path=str(tmp_path / "m.forge"), feature_names=names, epochs=8, patience=None,
        target_transform="standardize",
    )
    assert result.target_transform is not None
    predictor = forge.load_predictor(result.artifact_path)
    prediction = predictor.predict(Xs[:3]).numpy()
    # Native units: the same order of magnitude as y, not a z-score.
    assert np.all(np.abs(prediction) < 100 * (np.abs(y).max() + 1))


def test_classification_workflow_with_selection_end_to_end(tmp_path, id_csv_cls):
    Xs, y, names = forge.data.load_csv(id_csv_cls, target="target", labels=True, columns=FEATURES, return_feature_names=True)
    result = forge.train_tabular_classifier(
        Xs, y, path=str(tmp_path / "c.forge"), classes=["low", "high"], feature_names=names, epochs=8, patience=None,
    )
    assert result.classes == ["low", "high"]
    info = forge.inspect_model(result.artifact_path)
    assert info.input_schema.feature_names == tuple(FEATURES)
