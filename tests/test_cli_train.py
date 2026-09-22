"""Milestone 121 tests: `forge model train DATA.csv --task ... --target COLUMN --output PATH`.

A thin CLI adapter over `forge.train_tabular_classifier_csv()` / `forge.train_tabular_regressor_csv()`
(`forge/cli/model.py::cmd_train`) -- `--task` alone dispatches to one of the two Python functions, every
other flag is forwarded only when given (so an omitted flag is that function's own default), and the printed
summary / `--json` payload comes straight from the fields of the `TabularClassificationResult`/
`TabularRegressionResult` the call returns. These tests exercise the CLI surface itself (argument handling,
error reporting, `--json`); the training numerics are already proven identical to the manual pipeline by
`tests/test_train_tabular_csv.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.cli.main import main

FEATURES = ["f0", "f1", "f2"]


def write_csv_with_id(path: Path, X: np.ndarray, y: np.ndarray, *, target: str = "target", label: bool) -> Path:
    lines = [",".join(["id", *FEATURES, target])]
    for i in range(len(X)):
        y_cell = str(int(y[i])) if label else repr(float(y[i]))
        lines.append(",".join([str(1000 + i), *(repr(float(v)) for v in X[i]), y_cell]))
    path.write_text("\n".join(lines) + "\n")
    return path


def classification_data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    y = (2.0 * z[:, 0] - 1.5 * z[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return z, y


def regression_data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    y = 5.0 + 3.0 * z[:, 0] - 2.0 * z[:, 1] + 0.2 * rng.normal(size=n)
    return z, y


@pytest.fixture()
def cls_csv(tmp_path):
    X, y = classification_data(150)
    return write_csv_with_id(tmp_path / "cls.csv", X, y, label=True)


@pytest.fixture()
def reg_csv(tmp_path):
    X, y = regression_data(150)
    return write_csv_with_id(tmp_path / "reg.csv", X, y, label=False)


def run(argv, capsys):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------------------- classification


def test_cli_trains_a_classifier_from_csv_with_explicit_columns(tmp_path, cls_csv, capsys):
    output = tmp_path / "cls.forge"
    code, out, err = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "target",
         "--output", output, "--columns", *FEATURES, "--epochs", "10"],
        capsys,
    )
    assert code == 0 and err == ""
    assert output.is_file()
    assert "Trained tabular_classification model" in out
    assert "Validation accuracy:" in out
    info = forge.inspect_model(str(output))
    assert info.task == "tabular_classification"
    assert info.input_schema.feature_names == tuple(FEATURES)


def test_cli_trains_a_classifier_and_prints_json(tmp_path, cls_csv, capsys):
    output = tmp_path / "cls.forge"
    code, out, err = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "target",
         "--output", output, "--columns", *FEATURES, "--epochs", "10", "--json"],
        capsys,
    )
    assert code == 0 and err == ""
    payload = json.loads(out)
    assert payload["task"] == "tabular_classification"
    assert payload["features"] == 3
    assert payload["artifact_path"] == str(output)
    assert "validation_accuracy" in payload and "baseline_accuracy" in payload


def test_cli_classifier_training_without_columns_uses_every_non_target_column(tmp_path, capsys):
    X, y = classification_data(100)
    path = tmp_path / "plain.csv"
    lines = [",".join([*FEATURES, "target"])]
    for i in range(len(X)):
        lines.append(",".join([*(repr(float(v)) for v in X[i]), str(int(y[i]))]))
    path.write_text("\n".join(lines) + "\n")

    output = tmp_path / "m.forge"
    code, out, err = run(
        ["model", "train", str(path), "--task", "classification", "--target", "target",
         "--output", str(output), "--epochs", "5", "--json"],
        capsys,
    )
    assert code == 0
    assert json.loads(out)["features"] == 3


# --------------------------------------------------------------------------------------- regression


def test_cli_trains_a_regressor_with_target_transform(tmp_path, reg_csv, capsys):
    output = tmp_path / "reg.forge"
    code, out, err = run(
        ["model", "train", reg_csv, "--task", "regression", "--target", "target",
         "--output", output, "--columns", *FEATURES, "--epochs", "20",
         "--target-transform", "standardize", "--seed", "3"],
        capsys,
    )
    assert code == 0 and err == ""
    assert "Trained regression model" in out
    assert "Validation MSE:" in out
    info = forge.inspect_model(str(output))
    assert info.target_transform is not None
    assert info.input_schema.feature_names == tuple(FEATURES)


def test_cli_regression_json_payload_has_regression_fields(tmp_path, reg_csv, capsys):
    output = tmp_path / "reg.forge"
    code, out, err = run(
        ["model", "train", reg_csv, "--task", "regression", "--target", "target",
         "--output", output, "--columns", *FEATURES, "--epochs", "20", "--json"],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["task"] == "regression"
    assert {"validation_mse", "validation_mae", "baseline_mse", "outputs"} <= payload.keys()


def test_cli_target_transform_rejected_for_classification(tmp_path, cls_csv, capsys):
    code, out, err = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "target",
         "--output", tmp_path / "m.forge", "--target-transform", "standardize"],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "--target-transform applies only to --task regression" in err


# --------------------------------------------------------------------------------------- argument handling / errors


def test_cli_missing_target_is_an_argparse_error(cls_csv):
    with pytest.raises(SystemExit):
        main(["model", "train", str(cls_csv), "--task", "classification", "--output", "m.forge"])


def test_cli_missing_task_is_an_argparse_error(cls_csv):
    with pytest.raises(SystemExit):
        main(["model", "train", str(cls_csv), "--target", "target", "--output", "m.forge"])


def test_cli_missing_output_is_an_argparse_error(cls_csv):
    with pytest.raises(SystemExit):
        main(["model", "train", str(cls_csv), "--task", "classification", "--target", "target"])


def test_cli_invalid_target_column_is_a_clean_error(tmp_path, cls_csv, capsys):
    code, out, err = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "nope",
         "--output", tmp_path / "m.forge"],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "not in the header" in err


def test_cli_invalid_column_selection_is_a_clean_error(tmp_path, cls_csv, capsys):
    code, out, err = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "target",
         "--output", tmp_path / "m.forge", "--columns", "does_not_exist"],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "not in the header" in err


def test_cli_target_included_in_columns_is_a_clean_error(tmp_path, cls_csv, capsys):
    code, out, err = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "target",
         "--output", tmp_path / "m.forge", "--columns", *FEATURES, "target"],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "columns includes the target column" in err


def test_cli_missing_input_file_is_a_clean_error(tmp_path, capsys):
    code, out, err = run(
        ["model", "train", str(tmp_path / "nope.csv"), "--task", "classification", "--target", "target",
         "--output", tmp_path / "m.forge"],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "input file not found" in err


def test_cli_non_csv_input_is_rejected(tmp_path, capsys):
    (tmp_path / "data.json").write_text("[[1, 2, 3]]")
    code, out, err = run(
        ["model", "train", str(tmp_path / "data.json"), "--task", "classification", "--target", "target",
         "--output", tmp_path / "m.forge"],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "reads a .csv file" in err


def test_cli_output_directory_must_already_exist(tmp_path, cls_csv, capsys):
    code, out, err = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "target",
         "--output", tmp_path / "missing_dir" / "m.forge"],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "directory" in err


def test_cli_invalid_task_choice_is_an_argparse_error(cls_csv):
    with pytest.raises(SystemExit):
        main(["model", "train", str(cls_csv), "--task", "nonsense", "--target", "target", "--output", "m.forge"])


# --------------------------------------------------------------------------------------- fresh-process consumption


def test_a_cli_trained_artifact_is_consumable_from_a_freshly_loaded_predictor(tmp_path, cls_csv, capsys):
    """`forge model train` writes a real, standalone artifact -- loadable with no reference to the CLI call at all."""
    output = tmp_path / "m.forge"
    code, _, _ = run(
        ["model", "train", cls_csv, "--task", "classification", "--target", "target",
         "--output", output, "--columns", *FEATURES, "--epochs", "10"],
        capsys,
    )
    assert code == 0

    predictor = forge.load_predictor(str(output))
    assert predictor.classes == ["0", "1"]
    prediction = predictor.predict([[0.1, -0.2, 0.3]])
    assert len(prediction) == 1 and prediction[0].label in ("0", "1")

    predict_code, predict_out, predict_err = run(
        ["model", "predict", str(output), str(_write_predict_csv(tmp_path)), "--columns", *FEATURES],
        capsys,
    )
    assert predict_code == 0 and predict_err == ""
    assert "Sample" in predict_out or "Prediction" in predict_out


def _write_predict_csv(tmp_path: Path) -> Path:
    path = tmp_path / "predict_rows.csv"
    lines = [",".join(FEATURES)]
    lines.append("0.1,-0.2,0.3")
    lines.append("-0.5,0.4,0.1")
    path.write_text("\n".join(lines) + "\n")
    return path
