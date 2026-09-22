"""Milestone 121 tests: `forge.train_tabular_classifier_csv()` / `forge.train_tabular_regressor_csv()`.

The claim under test is that these two functions are *exactly* `forge.data.load_csv(..., columns=...,
return_feature_names=True)` followed by the unmodified `train_tabular_classifier()`/`train_tabular_regressor()`,
with the CSV's own header names threaded into `feature_names=` automatically -- nothing new is parsed,
validated, trained or persisted here. Every test below either:

- proves the CSV-wrapper result is *identical* (same result fields, same trained parameters, byte-for-byte)
  to the equivalent `load_csv()` + `train_tabular_*(..., feature_names=...)` calls written out by hand, or
- proves a rejection (unknown/duplicate/empty/target-overlapping `columns`, a non-existent file, ...) is
  exactly `load_csv()`'s own `DataError`, under `load_csv()`'s own name -- there is no second validation layer.

CUDA is `tests/test_train_tabular_csv_cuda.py`; the CLI (`forge model train`) is `tests/test_cli_train.py`;
the installed-wheel/fresh-process proof is the Milestone 121 section of `tests/test_packaging_smoke.py`.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.data import load_csv, load_csv_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tabular_workflows import classification_data, regression_data  # noqa: E402  (shared data recipe)

REPO_ROOT = Path(__file__).resolve().parents[1]
PIMA_DIR = REPO_ROOT / "examples" / "tabular_diabetes" / "data"
FEATURES = ["f0", "f1", "f2"]
SEED = 5


def write_csv_with_id(path: Path, X: np.ndarray, y: np.ndarray, *, target: str = "target", label: bool) -> Path:
    """`id, f0, f1, f2, target` -- an id column the caller must exclude via `columns=`."""
    lines = [",".join(["id", *FEATURES, target])]
    for i in range(len(X)):
        y_cell = str(int(y[i])) if label else repr(float(y[i]))
        lines.append(",".join([str(1000 + i), *(repr(float(v)) for v in X[i]), y_cell]))
    path.write_text("\n".join(lines) + "\n")
    return path


def parameters(model_path) -> "dict[str, np.ndarray]":
    model = forge.load_model(str(model_path))
    return {name: p.numpy().copy() for name, p in model.named_parameters()}


def assert_identical_training(a, b, a_path, b_path) -> None:
    for field in dataclasses.fields(a):
        if field.name in ("history", "model", "artifact_path", "target_transform"):
            continue
        assert getattr(a, field.name) == getattr(b, field.name), field.name
    pa, pb = parameters(a_path), parameters(b_path)
    assert pa.keys() == pb.keys() and pa
    for name in pa:
        np.testing.assert_array_equal(pa[name], pb[name], err_msg=name)


# --------------------------------------------------------------------------------------- classification


@pytest.fixture()
def id_csv_cls(tmp_path):
    X, y = classification_data(150)
    return write_csv_with_id(tmp_path / "cls.csv", X, y, label=True), X, y


def test_classifier_csv_explicit_columns_excludes_the_id_and_matches_the_manual_pipeline(tmp_path, id_csv_cls):
    path, X, y = id_csv_cls
    kwargs = dict(classes=["neg", "pos"], seed=SEED, epochs=25, patience=None)

    Xc, yc, names = load_csv(path, target="target", labels=True, columns=FEATURES, return_feature_names=True)
    manual = forge.train_tabular_classifier(Xc, yc, path=tmp_path / "manual.forge", feature_names=names, **kwargs)

    from_csv = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "wrapper.forge", columns=FEATURES, **kwargs,
    )

    assert_identical_training(manual, from_csv, tmp_path / "manual.forge", tmp_path / "wrapper.forge")
    assert from_csv.features == 3  # the id column never reached training


def test_classifier_csv_default_all_non_target_columns_matches_the_manual_pipeline(tmp_path):
    X, y = classification_data(120)
    lines = [",".join([*FEATURES, "target"])]
    for i in range(len(X)):
        lines.append(",".join([*(repr(float(v)) for v in X[i]), str(int(y[i]))]))
    path = tmp_path / "plain.csv"
    path.write_text("\n".join(lines) + "\n")
    kwargs = dict(classes=["neg", "pos"], seed=SEED, epochs=15, patience=None)

    Xc, yc, names = load_csv(path, target="target", labels=True, return_feature_names=True)
    manual = forge.train_tabular_classifier(Xc, yc, path=tmp_path / "manual.forge", feature_names=names, **kwargs)
    from_csv = forge.train_tabular_classifier_csv(path, target="target", path=tmp_path / "wrapper.forge", **kwargs)

    assert_identical_training(manual, from_csv, tmp_path / "manual.forge", tmp_path / "wrapper.forge")
    assert names == FEATURES


def test_feature_names_persist_automatically_with_no_feature_names_argument(tmp_path, id_csv_cls):
    """The wrapper has no `feature_names=` parameter at all; the artifact still records the selected names."""
    path, _, _ = id_csv_cls
    assert "feature_names" not in inspect.signature(forge.train_tabular_classifier_csv).parameters
    result = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "m.forge", columns=FEATURES,
        classes=["neg", "pos"], epochs=5, patience=None,
    )
    info = forge.inspect_model(result.artifact_path)
    assert info.input_schema.feature_names == tuple(FEATURES)


def test_selected_column_order_is_authoritative_not_file_order(tmp_path, id_csv_cls):
    path, _, _ = id_csv_cls
    reversed_columns = list(reversed(FEATURES))
    result = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "m.forge", columns=reversed_columns,
        classes=["neg", "pos"], epochs=5, patience=None,
    )
    info = forge.inspect_model(result.artifact_path)
    assert info.input_schema.feature_names == tuple(reversed_columns)


def test_target_listed_in_columns_is_rejected_by_loads_csv_own_check(tmp_path, id_csv_cls):
    path, _, _ = id_csv_cls
    with pytest.raises(forge.DataError, match="columns includes the target column 'target'"):
        forge.train_tabular_classifier_csv(
            path, target="target", path=tmp_path / "m.forge", columns=[*FEATURES, "target"],
        )


def test_unknown_column_is_rejected_and_named(tmp_path, id_csv_cls):
    path, _, _ = id_csv_cls
    with pytest.raises(forge.DataError, match=r"not in the header.*'nope'"):
        forge.train_tabular_classifier_csv(path, target="target", path=tmp_path / "m.forge", columns=["nope"])


def test_duplicate_column_is_rejected(tmp_path, id_csv_cls):
    path, _, _ = id_csv_cls
    with pytest.raises(forge.DataError, match="unique"):
        forge.train_tabular_classifier_csv(
            path, target="target", path=tmp_path / "m.forge", columns=["f0", "f0", "f1"],
        )


def test_empty_columns_is_rejected(tmp_path, id_csv_cls):
    path, _, _ = id_csv_cls
    with pytest.raises(forge.DataError, match="empty"):
        forge.train_tabular_classifier_csv(path, target="target", path=tmp_path / "m.forge", columns=[])


def test_reordered_feature_only_csv_predicts_identically_after_csv_training(tmp_path, id_csv_cls):
    """A CSV-trained artifact still exercises M119 alignment exactly as an array-trained one does."""
    path, X, _ = id_csv_cls
    result = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "m.forge", columns=FEATURES,
        classes=["neg", "pos"], seed=SEED, epochs=20, patience=None,
    )
    predictor = forge.load_predictor(result.artifact_path)

    reordered = list(reversed(FEATURES))
    feature_csv = tmp_path / "new_rows.csv"
    lines = [",".join(["id", *reordered])]
    for i in range(5):
        lines.append(",".join([str(2000 + i), *(repr(float(v)) for v in X[i][::-1])]))
    feature_csv.write_text("\n".join(lines) + "\n")

    X_sel, names_sel = load_csv_features(feature_csv, columns=reordered)
    via_reordered_csv = predictor.predict(X_sel, feature_names=names_sel)
    direct = predictor.predict(X[:5])
    assert [p.label for p in via_reordered_csv] == [p.label for p in direct]
    assert [p.confidence for p in via_reordered_csv] == pytest.approx([p.confidence for p in direct], abs=1e-5)


def test_missing_columns_sentinel_replacement_works_from_the_csv_wrapper(tmp_path):
    X, y = classification_data(180)
    X = X.copy()
    X[::7, 1] = 0.0
    path = write_csv_with_id(tmp_path / "sentinel.csv", X, y, label=True)
    kwargs = dict(missing_columns=[1], seed=SEED, epochs=15, patience=None)

    Xc, yc, names = load_csv(path, target="target", labels=True, columns=FEATURES, return_feature_names=True)
    manual = forge.train_tabular_classifier(Xc, yc, path=tmp_path / "manual.forge", feature_names=names, **kwargs)
    from_csv = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "wrapper.forge", columns=FEATURES, **kwargs,
    )
    assert_identical_training(manual, from_csv, tmp_path / "manual.forge", tmp_path / "wrapper.forge")
    assert type(forge.load_preprocessing(str(tmp_path / "wrapper.forge")).transforms[0]).__name__ == "ReplaceValue"


def test_classifier_csv_reads_text_class_labels_not_numeric_targets(tmp_path):
    """The classifier wrapper must read the target column as labels (`load_csv(..., labels=True)`), not numbers.

    A binary 0/1-coded target cannot distinguish `labels=True` from `labels=False` (an integer-valued float
    is accepted as a class index either way) -- text class names can: `labels=False` fails to parse them as
    numbers at all, so this is the test that actually pins the classifier wrapper's `labels=True` call.
    """
    X, y = classification_data(120)
    names = np.where(y == 1, "pos", "neg")
    lines = [",".join(["id", *FEATURES, "target"])]
    for i in range(len(X)):
        lines.append(",".join([str(1000 + i), *(repr(float(v)) for v in X[i]), names[i]]))
    path = tmp_path / "text_labels.csv"
    path.write_text("\n".join(lines) + "\n")

    result = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "m.forge", columns=FEATURES, epochs=10, patience=None,
    )
    assert result.classes == ["neg", "pos"]


def test_custom_model_is_forwarded_and_contract_checked(tmp_path, id_csv_cls):
    path, _, _ = id_csv_cls
    good = forge.nn.Sequential(forge.nn.Linear(3, 8), forge.nn.ReLU(), forge.nn.Linear(8, 2))
    result = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "m.forge", columns=FEATURES,
        classes=["neg", "pos"], model=good, epochs=5, patience=None,
    )
    assert result.model is not None

    wrong = forge.nn.Sequential(forge.nn.Linear(3, 5))  # 5 outputs for 2 classes
    with pytest.raises(forge.TrainerError, match="output"):
        forge.train_tabular_classifier_csv(
            path, target="target", path=tmp_path / "bad.forge", columns=FEATURES, model=wrong,
        )


# --------------------------------------------------------------------------------------- regression + M116


@pytest.fixture()
def id_csv_reg(tmp_path):
    X, y = regression_data(150)
    return write_csv_with_id(tmp_path / "reg.csv", X, y, label=False), X, y


def test_regressor_csv_explicit_columns_matches_the_manual_pipeline(tmp_path, id_csv_reg):
    path, X, y = id_csv_reg
    kwargs = dict(seed=SEED, epochs=40, patience=None)

    Xc, yc, names = load_csv(path, target="target", columns=FEATURES, return_feature_names=True)
    manual = forge.train_tabular_regressor(Xc, yc, path=tmp_path / "manual.forge", feature_names=names, **kwargs)
    from_csv = forge.train_tabular_regressor_csv(
        path, target="target", path=tmp_path / "wrapper.forge", columns=FEATURES, **kwargs,
    )
    assert_identical_training(manual, from_csv, tmp_path / "manual.forge", tmp_path / "wrapper.forge")


def test_regressor_csv_target_transform_standardize_persists_and_is_native_units(tmp_path, id_csv_reg):
    path, X, y = id_csv_reg
    result = forge.train_tabular_regressor_csv(
        path, target="target", path=tmp_path / "m.forge", columns=FEATURES,
        target_transform="standardize", seed=SEED, epochs=40, patience=None,
    )
    assert result.target_transform is not None
    predictor = forge.load_predictor(result.artifact_path)
    predicted = predictor.predict(X[:10]).numpy()[:, 0]
    assert np.corrcoef(predicted, y[:10])[0, 1] > 0.8  # native units, correlated with the real target


def test_regressor_csv_target_transform_rejects_non_regression_choice(tmp_path, id_csv_reg):
    path, _, _ = id_csv_reg
    with pytest.raises(forge.DataError):
        forge.train_tabular_regressor_csv(
            path, target="target", path=tmp_path / "m.forge", target_transform="not-a-real-transform",
        )


def test_regressor_csv_target_excluded_when_selected_as_a_column(tmp_path, id_csv_reg):
    path, _, _ = id_csv_reg
    with pytest.raises(forge.DataError, match="columns includes the target column 'target'"):
        forge.train_tabular_regressor_csv(path, target="target", path=tmp_path / "m.forge", columns=[*FEATURES, "target"])


# --------------------------------------------------------------------------------------- API surface / thinness


def test_csv_wrapper_defaults_match_the_array_trainer_defaults():
    """Every shared keyword must have the identical default -- no silently introduced difference (the brief's §6/§7)."""
    shared_classifier = ["epochs", "batch_size", "learning_rate", "val_fraction",
                          "missing_value", "patience", "seed", "verbose"]
    cls_csv = inspect.signature(forge.train_tabular_classifier_csv).parameters
    cls_array = inspect.signature(forge.train_tabular_classifier).parameters
    for name in shared_classifier:
        assert cls_csv[name].default == cls_array[name].default, name

    shared_regressor = shared_classifier + ["target_transform"]
    reg_csv = inspect.signature(forge.train_tabular_regressor_csv).parameters
    reg_array = inspect.signature(forge.train_tabular_regressor).parameters
    for name in shared_regressor:
        assert reg_csv[name].default == reg_array[name].default, name


def test_the_csv_wrappers_have_no_own_csv_parsing(tmp_path):
    """A thin orchestration layer: no second reader, no `csv`/`open()` call inside the wrapper module itself."""
    import ast

    import forge.training.tabular_csv as tabular_csv

    tree = ast.parse(Path(tabular_csv.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False):
            node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    assert "import csv" not in code and "csv.reader" not in code and "open(" not in code
    assert "load_csv(" in code  # it does call the one existing reader


def test_the_wrappers_bypass_no_array_trainer_validation(tmp_path, id_csv_cls):
    """Every failure of the underlying array trainer is still reachable through the CSV door."""
    path, _, _ = id_csv_cls
    with pytest.raises(forge.PersistenceError):
        forge.train_tabular_classifier_csv(
            path, target="target", path=tmp_path / "missing_dir" / "m.forge", columns=FEATURES,
        )


def test_existing_array_apis_are_unaffected_by_this_milestone():
    for fn in (forge.train_tabular_classifier, forge.train_tabular_regressor):
        parameters_ = list(inspect.signature(fn).parameters)
        assert parameters_[:2] == ["X", "y"]
        assert not any("csv" in name.lower() for name in parameters_)


def test_the_csv_wrapper_return_value_is_the_array_trainers_own_result_type(tmp_path):
    X, y = classification_data(60)
    path = write_csv_with_id(tmp_path / "d.csv", X, y, label=True)
    result = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "m.forge", columns=FEATURES, epochs=3, patience=None,
    )
    assert isinstance(result, forge.training.TabularClassificationResult)

    Xr, yr = regression_data(60)
    path_r = write_csv_with_id(tmp_path / "r.csv", Xr, yr, label=False)
    result_r = forge.train_tabular_regressor_csv(
        path_r, target="target", path=tmp_path / "r.forge", columns=FEATURES, epochs=3, patience=None,
    )
    assert isinstance(result_r, forge.training.TabularRegressionResult)


# --------------------------------------------------------------------------------------- real data: Pima


@pytest.fixture(scope="module")
def pima_with_id(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pima_id")
    source = PIMA_DIR / "diabetes.csv"
    with open(source, newline="") as fh:
        table = list(csv.reader(fh))
    header, rows = table[0], table[1:]
    out = tmp / "diabetes_with_id.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["patient_id", *header])
        for i, row in enumerate(rows):
            writer.writerow([10000 + i, *row])
    return out, tmp


PIMA_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def test_real_pima_csv_with_an_id_column_trains_identically_to_the_id_free_manual_pipeline(pima_with_id):
    path, tmp = pima_with_id
    kwargs = dict(classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5], seed=0, epochs=15)

    reference = np.genfromtxt(PIMA_DIR / "diabetes.csv", delimiter=",", skip_header=1)
    manual = forge.train_tabular_classifier(
        reference[:, :-1], reference[:, -1].astype(int), path=tmp / "manual.forge", **kwargs,
    )
    from_csv = forge.train_tabular_classifier_csv(
        path, target="Outcome", path=tmp / "wrapper.forge", columns=PIMA_FEATURES, **kwargs,
    )
    assert_identical_training(manual, from_csv, tmp / "manual.forge", tmp / "wrapper.forge")
    assert from_csv.validation_accuracy > from_csv.baseline_accuracy
    info = forge.inspect_model(from_csv.artifact_path)
    assert info.input_schema.feature_names == tuple(PIMA_FEATURES)
