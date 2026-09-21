"""Milestone 118 tests: CSV -> `forge.data.load_csv()` -> the *unchanged* `train_tabular_*()` functions.

The claim under test is that CSV is only a way to obtain arrays, so a CSV workflow and the NumPy workflow on
the same values are the same computation: the arrays are bit-identical, and so are the trained parameters, the
persisted preprocessing / target transform, the class list and every number in the result. Nothing in the
training functions changed; these tests prove that nothing about a CSV origin can weaken what they already do:

- the arrays go through the same preflight (before epoch 1, `Trainer.fit` replaced by a tripwire);
- preprocessing and the M116 target transform are still fitted on the training split only -- recomputed here
  from the CSV's own values with the documented split rule, on data whose all-rows statistics differ;
- a value the reader accepts but training must refuse (finite in float64, infinite in float32) is refused.

The data is the awkward M114 recipe (raw features around 50,000 / -3 / 200; a ~2x10^5 target), so a missing
preprocessing step or a target that is not inverted cannot pass by accident. CLI evaluation of these CSVs is
`tests/test_cli_evaluate_csv.py`; CUDA is `tests/test_csv_tabular_workflow_cuda.py`.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.data import StandardizeTarget, load_csv
from forge.training import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tabular_workflows import (  # noqa: E402  (shared data recipe)
    N, classification_data, no_training, regression_data, split_indices,
)
from test_target_transform import big_target_data  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PIMA = REPO_ROOT / "examples" / "tabular_diabetes" / "data" / "diabetes.csv"
SEED = 3


def write_csv(path, columns: "dict[str, np.ndarray]") -> Path:
    """Write named columns as CSV. Floats use `repr()`, which round-trips a double exactly."""
    names = list(columns)
    n = len(next(iter(columns.values())))
    lines = [",".join(names)]
    for i in range(n):
        lines.append(",".join(
            repr(float(c[i])) if isinstance(c[i], (float, np.floating)) else str(c[i]) for c in columns.values()
        ))
    path.write_text("\n".join(lines) + "\n")
    return path


def feature_columns(X: np.ndarray) -> "dict[str, np.ndarray]":
    return {f"f{i}": X[:, i] for i in range(X.shape[1])}


def parameters(model_path) -> "dict[str, np.ndarray]":
    model = forge.load_model(str(model_path))
    return {name: p.numpy().copy() for name, p in model.named_parameters()}


def assert_same_training(a, b, a_path, b_path) -> None:
    """Every result number, and every trained parameter, is *equal* -- not close."""
    for field in dataclasses.fields(a):
        if field.name in ("history", "model", "artifact_path", "target_transform"):
            continue
        assert getattr(a, field.name) == getattr(b, field.name), field.name
    pa, pb = parameters(a_path), parameters(b_path)
    assert pa.keys() == pb.keys() and pa
    for name in pa:
        np.testing.assert_array_equal(pa[name], pb[name], err_msg=name)
    assert a.history.epochs_completed == b.history.epochs_completed


def normalize_of(path):
    return forge.load_preprocessing(str(path)).transforms[-1]


# ------------------------------------------------------------------------------------------ the arrays are the arrays


def test_the_csv_arrays_are_the_numpy_arrays_bit_for_bit(tmp_path):
    X, y = classification_data()
    path = write_csv(tmp_path / "data.csv", {**feature_columns(X), "label": y})
    Xc, yc = load_csv(path, target="label", labels=True)
    np.testing.assert_array_equal(Xc, X)
    np.testing.assert_array_equal(yc, y)
    assert yc.dtype == np.int64


# ------------------------------------------------------------------------------------------ classification


@pytest.fixture(scope="module")
def classification_pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("csv_cls")
    X, y = classification_data()
    csv_path = write_csv(tmp / "data.csv", {**feature_columns(X), "label": y})
    kwargs = dict(classes=["neg", "pos"], seed=SEED, epochs=60)
    reference = forge.train_tabular_classifier(X, y, path=tmp / "numpy.forge", **kwargs)
    Xc, yc = load_csv(csv_path, target="label", labels=True)
    from_csv = forge.train_tabular_classifier(Xc, yc, path=tmp / "csv.forge", **kwargs)
    return X, y, csv_path, reference, from_csv, tmp


def test_csv_classification_is_the_numpy_classification(classification_pair):
    _, _, _, reference, from_csv, tmp = classification_pair
    assert_same_training(reference, from_csv, tmp / "numpy.forge", tmp / "csv.forge")
    assert from_csv.classes == ["neg", "pos"] and from_csv.validation_accuracy > from_csv.baseline_accuracy
    assert normalize_of(tmp / "numpy.forge").mean == normalize_of(tmp / "csv.forge").mean
    assert normalize_of(tmp / "numpy.forge").std == normalize_of(tmp / "csv.forge").std


def test_csv_classification_preprocessing_is_fitted_on_the_training_rows_only(classification_pair):
    X, _, csv_path, _, _, tmp = classification_pair
    Xc, _ = load_csv(csv_path, target="label", labels=True)
    train_idx, _ = split_indices(N, SEED)
    normalize = normalize_of(tmp / "csv.forge")
    np.testing.assert_allclose(normalize.mean, Xc[train_idx].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(normalize.std, Xc[train_idx].std(axis=0), rtol=1e-5)
    assert not np.allclose(normalize.mean, Xc.mean(axis=0), rtol=1e-9)          # not all rows: mutation 4's signature


def test_validation_rows_of_a_csv_cannot_influence_the_fitted_preprocessing(tmp_path):
    X, y = classification_data(120)
    _, val_idx = split_indices(120, 0)
    poisoned = X.copy()
    poisoned[val_idx] = 1e9
    a = write_csv(tmp_path / "a.csv", {**feature_columns(X), "label": y})
    b = write_csv(tmp_path / "b.csv", {**feature_columns(poisoned), "label": y})
    for name, csv_path in (("a", a), ("b", b)):
        Xc, yc = load_csv(csv_path, target="label", labels=True)
        forge.train_tabular_classifier(Xc, yc, path=tmp_path / f"{name}.forge", epochs=3, patience=None)
    assert normalize_of(tmp_path / "a.forge").mean == normalize_of(tmp_path / "b.forge").mean


def test_text_labels_from_a_csv_get_the_apis_sorted_class_order_whatever_the_row_order(tmp_path):
    X, y = classification_data(120)
    names = np.where(y == 1, "zeta", "alpha")
    forward = write_csv(tmp_path / "f.csv", {**feature_columns(X), "kind": names})
    order = np.random.default_rng(0).permutation(120)
    shuffled = write_csv(tmp_path / "s.csv", {**feature_columns(X[order]), "kind": names[order]})
    runs = []
    for name, path in (("f", forward), ("s", shuffled)):
        Xc, yc = load_csv(path, target="kind", labels=True)
        assert yc.dtype.kind == "U"
        runs.append(forge.train_tabular_classifier(Xc, yc, path=tmp_path / f"{name}.forge", epochs=5, patience=None))
    assert runs[0].classes == runs[1].classes == ["alpha", "zeta"]            # sorted, not first-seen
    Xc, yc = load_csv(forward, target="kind", labels=True)
    explicit = forge.train_tabular_classifier(Xc, yc, path=tmp_path / "e.forge", classes=["zeta", "alpha"], epochs=5)
    assert explicit.classes == ["zeta", "alpha"]                              # the caller's order wins, and is persisted
    assert forge.load_predictor(str(tmp_path / "e.forge")).classes == ["zeta", "alpha"]


def test_missing_columns_sentinel_zero_works_from_a_csv(tmp_path):
    """Forge's existing missing-value mechanism (`ReplaceValue`) is reachable from a CSV: a `0` sentinel, not an empty cell."""
    X, y = classification_data(200)
    X = X.copy()
    X[::7, 1] = 0.0
    csv_path = write_csv(tmp_path / "m.csv", {**feature_columns(X), "label": y})
    Xc, yc = load_csv(csv_path, target="label", labels=True)
    kwargs = dict(missing_columns=[1], seed=SEED, epochs=20)
    ref = forge.train_tabular_classifier(X, y, path=tmp_path / "n.forge", **kwargs)
    got = forge.train_tabular_classifier(Xc, yc, path=tmp_path / "c.forge", **kwargs)
    assert_same_training(ref, got, tmp_path / "n.forge", tmp_path / "c.forge")
    assert type(forge.load_preprocessing(str(tmp_path / "c.forge")).transforms[0]).__name__ == "ReplaceValue"


# ------------------------------------------------------------------------------------------ regression + M116


@pytest.fixture(scope="module")
def regression_pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("csv_reg")
    X, y = big_target_data()
    csv_path = write_csv(tmp / "data.csv", {"price": y, **feature_columns(X)})       # the target is the FIRST column
    kwargs = dict(seed=SEED, epochs=150, target_transform="standardize")
    reference = forge.train_tabular_regressor(X, y, path=tmp / "numpy.forge", **kwargs)
    Xc, yc = load_csv(csv_path, target="price")
    from_csv = forge.train_tabular_regressor(Xc, yc, path=tmp / "csv.forge", **kwargs)
    return X, y, csv_path, reference, from_csv, tmp


def test_csv_regression_with_a_target_transform_is_the_numpy_regression(regression_pair):
    X, y, _, reference, from_csv, tmp = regression_pair
    assert_same_training(reference, from_csv, tmp / "numpy.forge", tmp / "csv.forge")
    assert from_csv.target_transform == reference.target_transform
    assert isinstance(from_csv.target_transform, StandardizeTarget)
    assert from_csv.baseline_mse > 1e9 and from_csv.validation_mse < 0.2 * from_csv.baseline_mse   # native units, and it learned


def test_the_target_transform_of_a_csv_is_fitted_on_the_training_targets_only(regression_pair):
    """Mutation 4/5: fitting on all rows, or ignoring the transform, changes these numbers."""
    _, _, csv_path, _, from_csv, tmp = regression_pair
    Xc, yc = load_csv(csv_path, target="price")
    train_idx, _ = split_indices(len(yc), SEED)
    transform = forge.load_predictor(str(tmp / "csv.forge")).target_transform
    np.testing.assert_allclose(transform.mean, [yc[train_idx].mean()], rtol=1e-12)
    np.testing.assert_allclose(transform.std, [yc[train_idx].std()], rtol=1e-12)
    assert abs(yc.mean() - yc[train_idx].mean()) > 1e-6 * abs(yc.mean())          # all rows would give other numbers
    assert transform == from_csv.target_transform


def test_a_csv_regression_artifact_predicts_in_native_units(regression_pair):
    X, y, _, _, _, tmp = regression_pair
    predicted = forge.load_predictor(str(tmp / "csv.forge")).predict(X[:20]).numpy()[:, 0]
    assert np.all(np.abs(predicted) > 1e4)                                        # z-scores would be |.| < ~20
    assert np.corrcoef(predicted, y[:20])[0, 1] > 0.95


def test_csv_regression_without_a_transform_is_the_numpy_regression_too(tmp_path):
    X, y = regression_data()
    csv_path = write_csv(tmp_path / "d.csv", {**feature_columns(X), "y": y})
    Xc, yc = load_csv(csv_path, target="y")
    ref = forge.train_tabular_regressor(X, y, path=tmp_path / "n.forge", seed=SEED, epochs=40)
    got = forge.train_tabular_regressor(Xc, yc, path=tmp_path / "c.forge", seed=SEED, epochs=40)
    assert_same_training(ref, got, tmp_path / "n.forge", tmp_path / "c.forge")
    assert got.target_transform is None


# ------------------------------------------------------------------------------------------ preflight is unchanged


def test_a_csv_too_small_to_split_is_rejected_by_the_existing_preflight(tmp_path, no_training):
    path = write_csv(tmp_path / "two.csv", {"a": np.array([1.0, 2.0]), "y": np.array([0, 1])})
    X, y = load_csv(path, target="y", labels=True)
    with pytest.raises(forge.DataError, match="at least 3 samples"):
        forge.train_tabular_classifier(X, y, path=tmp_path / "m.forge")
    Xr, yr = load_csv(path, target="y")
    with pytest.raises(forge.DataError, match="at least 3 samples"):
        forge.train_tabular_regressor(Xr, yr, path=tmp_path / "r.forge")


def test_a_single_class_csv_is_rejected_by_the_existing_preflight(tmp_path, no_training):
    X = np.arange(30.0).reshape(10, 3)
    path = write_csv(tmp_path / "one.csv", {**feature_columns(X), "y": np.zeros(10, dtype=int)})
    Xc, yc = load_csv(path, target="y", labels=True)
    with pytest.raises(forge.DataError, match="at least 2 classes"):
        forge.train_tabular_classifier(Xc, yc, path=tmp_path / "m.forge")


def test_a_csv_whose_class_has_no_training_rows_is_rejected_before_epoch_1(tmp_path, no_training):
    X, y = classification_data(N)
    names = np.array(["a"] * N, dtype=object)
    row = int(split_indices(N, 0)[1][0])                                          # a row that lands in validation
    names[row] = "rare"
    path = write_csv(tmp_path / "rare.csv", {**feature_columns(X), "kind": names})
    Xc, yc = load_csv(path, target="kind", labels=True)
    with pytest.raises(forge.DataError, match=r"'rare'.*no rows in the training split"):
        forge.train_tabular_classifier(Xc, yc, path=tmp_path / "m.forge")


def test_a_value_finite_in_float64_but_not_in_float32_is_still_refused(tmp_path, no_training):
    """Mutation 3: the reader accepts 1e39 (a valid double); training's own finite check must still stop it."""
    X, y = classification_data(30)
    X = X.copy()
    X[3, 0] = 1e39
    path = write_csv(tmp_path / "big.csv", {**feature_columns(X), "label": y})
    Xc, yc = load_csv(path, target="label", labels=True)
    assert np.isfinite(Xc).all()
    # The *first* finite check (which names the position) must be the one that fires, not a later layer's.
    with pytest.raises(forge.DataError, match=r"X contains 1 non-finite value\(s\).*first at index \(3, 0\)"):
        forge.train_tabular_classifier(Xc, yc, path=tmp_path / "m.forge", classes=["a", "b"])


def test_a_bad_output_path_and_a_bad_model_still_fail_before_epoch_1(tmp_path, no_training):
    X, y = classification_data(60)
    path = write_csv(tmp_path / "d.csv", {**feature_columns(X), "label": y})
    Xc, yc = load_csv(path, target="label", labels=True)
    with pytest.raises(forge.PersistenceError):
        forge.train_tabular_classifier(Xc, yc, path=tmp_path / "missing_dir" / "m.forge")
    wrong = forge.nn.Sequential(forge.nn.Linear(3, 5))                            # 5 outputs for 2 classes
    with pytest.raises(forge.TrainerError, match="output"):
        forge.train_tabular_classifier(Xc, yc, path=tmp_path / "m.forge", model=wrong)


def test_the_feature_count_of_the_csv_is_what_the_artifact_expects(tmp_path):
    """Feature *count* is what an artifact records; a CSV with another number of columns is refused at evaluation."""
    X, y = classification_data(60)
    train_csv = write_csv(tmp_path / "t.csv", {**feature_columns(X), "label": y})
    Xc, yc = load_csv(train_csv, target="label", labels=True)
    forge.train_tabular_classifier(Xc, yc, path=tmp_path / "m.forge", epochs=3, patience=None)
    fewer = write_csv(tmp_path / "fewer.csv", {**feature_columns(X[:, :2]), "label": y})
    Xf, yf = load_csv(fewer, target="label", labels=True)
    with pytest.raises(forge.DataError, match="feature"):
        forge.load_predictor(str(tmp_path / "m.forge")).evaluate(Xf, yf)


# ------------------------------------------------------------------------------------------ the boundary is thin


def test_the_training_functions_are_unchanged_array_apis():
    for fn in (forge.train_tabular_classifier, forge.train_tabular_regressor):
        parameters_ = list(inspect.signature(fn).parameters)
        assert parameters_[:2] == ["X", "y"] and "target" not in parameters_ and "labels" not in parameters_
        assert not any("csv" in name.lower() for name in parameters_)


def test_the_training_module_does_not_read_csv():
    """CSV parsing must not be reachable from training: composition, not a second ingestion path inside it."""
    import ast

    import forge.training.tabular as tabular

    tree = ast.parse(Path(tabular.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):                       # executable code only: its docstrings may (and do) point at load_csv()
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False):
            node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    assert "load_csv" not in code and "csv_reader" not in code and "import csv" not in code
    assert not hasattr(tabular, "load_csv") and not hasattr(tabular, "csv")


# ------------------------------------------------------------------------------------------ real data: Pima


@pytest.fixture(scope="module")
def pima_pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("csv_pima")
    reference = np.genfromtxt(PIMA, delimiter=",", skip_header=1)                   # the M114/tests/real_world way
    kwargs = dict(classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5], seed=0)
    numpy_run = forge.train_tabular_classifier(
        reference[:, :-1], reference[:, -1].astype(int), path=tmp / "numpy.forge", **kwargs,
    )
    X, y = load_csv(PIMA, target="Outcome", labels=True)
    csv_run = forge.train_tabular_classifier(X, y, path=tmp / "csv.forge", **kwargs)
    return reference, numpy_run, csv_run, tmp


def test_pima_csv_rows_features_and_labels_are_the_numpy_reference(pima_pair):
    reference, *_ = pima_pair
    X, y = load_csv(PIMA, target="Outcome", labels=True)
    assert X.shape == (768, 8)
    np.testing.assert_array_equal(X, reference[:, :-1])                           # rows, feature order, values
    np.testing.assert_array_equal(y, reference[:, -1].astype(np.int64))           # labels


def test_pima_csv_workflow_is_the_numpy_workflow(pima_pair):
    _, numpy_run, csv_run, tmp = pima_pair
    assert_same_training(numpy_run, csv_run, tmp / "numpy.forge", tmp / "csv.forge")
    assert csv_run.validation_accuracy > csv_run.baseline_accuracy               # it learned
    assert (csv_run.samples, csv_run.features, csv_run.train_samples, csv_run.validation_samples) == (768, 8, 614, 154)
    assert normalize_of(tmp / "numpy.forge").mean == normalize_of(tmp / "csv.forge").mean
    assert type(forge.load_preprocessing(str(tmp / "csv.forge")).transforms[0]).__name__ == "ReplaceValue"
