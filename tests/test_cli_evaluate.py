"""Milestone 117 tests: `forge model evaluate` -- the CLI door to `ArtifactPredictor.evaluate()`.

```text
forge model evaluate MODEL INPUT [TARGETS]
    -> load_predictor() -> ArtifactPredictor.evaluate() -> printed result
```

The contract under test is that the command is an *interface*, not a second evaluation system:

- **parity** -- for the same artifact and data, the CLI's numbers are the Python API's, field by field
  (`assert_matches_result`), and both are checked against numbers computed by hand / with plain NumPy,
  so "the CLI and the API agree" cannot pass just because both are wrong the same way;
- **thinness** -- a sentinel replaces `ArtifactPredictor.evaluate()`; whatever it returns is exactly what the
  CLI prints, and the arrays it receives are exactly the files' contents (no caller-side preprocessing);
- **M116** -- a `target_transform="standardize"` artifact is reported in native units without the CLI knowing;
- **errors** -- normal user mistakes are one `Error: ...` line and exit status 1, never a traceback, and
  `--json` never prints half a document;
- **read-only** -- artifact, inputs and the directory around them are byte-identical afterwards.

CUDA parity lives in `tests/test_cli_evaluate_cuda.py`; the installed-wheel, outside-the-repo run lives in
`tests/test_packaging_smoke.py`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.cli.main import main as cli_main
from forge.data import StandardizeTarget
from forge.data.transforms import Normalize
from forge.nn import Linear, ReLU, Sequential
from forge.nn.parameter import Parameter
from forge.serialization import save_model
from forge.training import ClassificationEvaluationResult, RegressionEvaluationResult, load_predictor
from forge.training.inference import ArtifactPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_artifact_evaluation import (  # noqa: E402  (shared hand-computed fixtures)
    _CLASSES, _CONFUSION, _TRUE, _X, _image_root, _save_image_artifact, _save_scoring_artifact,
)
from test_target_transform import big_target_data, independent_native_prediction  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
TOL = dict(rel=1e-6, abs=1e-9)  # the CLI runs the same code on the same arrays, so this is tight on purpose


# ------------------------------------------------------------------------------------------------ helpers


def run_cli(argv, capsys):
    """Run `forge <argv>` in-process: `(exit code, stdout, stderr)`."""
    code = cli_main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def evaluate_json(capsys, *argv) -> dict:
    code, out, err = run_cli(["model", "evaluate", *argv, "--json"], capsys)
    assert code == 0, err
    return json.loads(out, parse_constant=_reject_constant)


def _reject_constant(name):  # NaN / Infinity are not JSON
    raise AssertionError(f"stdout contains the non-JSON constant {name}")


def assert_matches_result(payload: dict, result) -> None:
    """Every field of the API's result dataclass equals the CLI's JSON value (and no other key exists)."""
    import dataclasses

    assert set(payload) == {f.name for f in dataclasses.fields(result)}
    for field in dataclasses.fields(result):
        expected, got = getattr(result, field.name), payload[field.name]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(np.array(got), expected)
        elif isinstance(expected, tuple) and expected and isinstance(expected[0], float):
            assert got == pytest.approx(list(expected), **TOL), field.name
        elif isinstance(expected, tuple):
            assert got == list(expected), field.name
        elif isinstance(expected, float):
            assert got == pytest.approx(expected, **TOL), field.name
        else:
            assert got == expected, field.name


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_state(root: Path) -> dict:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def assert_clean_error(code, out, err, *, match=None) -> None:
    assert code != 0
    assert err.startswith("Error: ") and "Traceback" not in err, err
    assert out == "", f"an error must not print to stdout (it may be piped as JSON): {out!r}"
    if match is not None:
        assert match in err, err


# -- artifact builders (plain functions so the CUDA test file can reuse them) ---------------------------------


def save_regression_artifact(tmp_path, *, outputs=1, name="regression.forge") -> str:
    """A regression artifact whose persisted preprocessing changes the answer (`Normalize`, non-trivial)."""
    forge.random.seed(0)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, outputs))
    path = tmp_path / name
    save_model(
        model, str(path),
        preprocessing=Normalize(
            mean=np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
            std=np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32),
        ),
        task="regression",
    )
    return str(path)


def regression_data(n=64, outputs=1, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32) * 3.0 + 1.0
    y = rng.standard_normal(n if outputs == 1 else (n, outputs)).astype(np.float32) * 2.0 + 5.0
    return X, y


def train_standardized_regressor(tmp_path):
    """The real M116 path: `train_tabular_regressor(..., target_transform="standardize")` on a ~2e5-scale target."""
    X, y = big_target_data()
    path = tmp_path / "standardized.forge"
    forge.train_tabular_regressor(X, y, path=path, seed=3, epochs=150, target_transform="standardize")
    return str(path)


@pytest.fixture(scope="module")
def m116_artifact(tmp_path_factory) -> str:
    return train_standardized_regressor(tmp_path_factory.mktemp("cli_m116"))


@pytest.fixture()
def scoring(tmp_path):
    """Hand-computed tabular classification fixture (classes deliberately not alphabetical)."""
    model = _save_scoring_artifact(tmp_path)
    np.save(tmp_path / "X.npy", _X)
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    return model, tmp_path / "X.npy", tmp_path / "y.npy"


# ==================================================================================== tabular classification


def test_tabular_classification_matches_hand_computation_and_the_python_api(scoring, capsys):
    model, X, y = scoring
    payload = evaluate_json(capsys, model, X, y)

    # By hand (see tests/test_artifact_evaluation.py): 6 of 8 right, zebra is 4 of 8, this artifact's own class order.
    assert payload["task"] == "tabular_classification" and payload["samples"] == 8
    assert payload["classes"] == _CLASSES
    assert payload["confusion_matrix"] == _CONFUSION.tolist()
    assert payload["support"] == [4, 3, 1, 0]
    assert payload["accuracy"] == pytest.approx(6 / 8)
    assert payload["baseline_accuracy"] == pytest.approx(4 / 8)
    assert payload["precision"] == pytest.approx([3 / 3, 2 / 3, 1 / 2, 0.0])
    assert payload["recall"] == pytest.approx([3 / 4, 2 / 3, 1 / 1, 0.0])

    assert_matches_result(payload, load_predictor(model).evaluate(_X, np.array(_TRUE)))


def test_human_readable_output_reports_the_result(scoring, capsys):
    model, X, y = scoring
    code, out, err = run_cli(["model", "evaluate", model, X, y], capsys)
    assert code == 0 and err == ""
    lines = out.splitlines()
    assert lines[:5] == [
        "Task: tabular_classification", "Samples: 8", "Accuracy: 75.00%", "Baseline accuracy: 50.00%",
        f"Loss: {load_predictor(model).evaluate(_X, np.array(_TRUE)).loss:.4f}",
    ]
    # Per-class rows and matrix rows follow the artifact's class order, not alphabetical order.
    class_rows = [ln.split() for ln in lines[lines.index("Classes:") + 2 : lines.index("Classes:") + 6]]
    assert [r[0] for r in class_rows] == _CLASSES
    assert class_rows[0] == ["zebra", "1.0000", "0.7500", "4"]
    assert class_rows[3] == ["unused", "0.0000", "0.0000", "0"]
    header = next(i for i, ln in enumerate(lines) if ln.startswith("Confusion matrix"))
    assert lines[header + 1].split() == _CLASSES
    assert [ln.split() for ln in lines[header + 2 : header + 6]] == [
        [name, *(str(v) for v in row)] for name, row in zip(_CLASSES, _CONFUSION.tolist())
    ]


def test_class_indices_work_as_targets_and_agree_with_class_names(scoring, tmp_path, capsys):
    model, X, y = scoring
    np.save(tmp_path / "y_idx.npy", np.array([_CLASSES.index(name) for name in _TRUE]))
    by_index = evaluate_json(capsys, model, X, tmp_path / "y_idx.npy")
    assert by_index == evaluate_json(capsys, model, X, y)


def test_the_artifacts_preprocessing_is_what_runs(scoring, capsys):
    """Every raw row has x0 >= x1, so a model fed *raw* rows answers 'zebra' for all 8; only the persisted
    Normalize gives the hand-computed matrix above. No caller-side preprocessing exists in the command."""
    model, X, y = scoring
    payload = evaluate_json(capsys, model, X, y)
    assert payload["confusion_matrix"] != [[4, 0, 0, 0], [3, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    assert payload["accuracy"] == pytest.approx(0.75)


def test_batch_size_is_passed_through_and_does_not_change_the_answer(scoring, capsys):
    model, X, y = scoring
    default = evaluate_json(capsys, model, X, y)
    small = evaluate_json(capsys, model, X, y, "--batch-size", "3")
    assert small["confusion_matrix"] == default["confusion_matrix"]
    assert small["loss"] == pytest.approx(default["loss"], rel=1e-5)
    code, out, err = run_cli(["model", "evaluate", model, X, y, "--batch-size", "0"], capsys)
    assert_clean_error(code, out, err, match="batch_size")


def test_explicit_cpu_device(scoring, capsys):
    model, X, y = scoring
    assert evaluate_json(capsys, model, X, y, "--device", "cpu") == evaluate_json(capsys, model, X, y)


# ============================================================================================== regression


def test_regression_matches_numpy_and_the_python_api(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)
    payload = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy")

    predictions = independent_native_prediction(model, X)[:, 0]  # persisted Normalize + raw forward, by hand
    assert payload["task"] == "regression" and payload["samples"] == 64
    assert payload["mse"] == pytest.approx(float(np.mean((predictions - y) ** 2)), rel=1e-5)
    assert payload["mae"] == pytest.approx(float(np.mean(np.abs(predictions - y))), rel=1e-5)
    assert payload["baseline_mse"] == pytest.approx(float(np.mean((y - y.mean()) ** 2)), rel=1e-6)
    assert_matches_result(payload, load_predictor(model).evaluate(X, y))


def test_regression_human_readable_output(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)
    code, out, err = run_cli(["model", "evaluate", model, tmp_path / "X.npy", tmp_path / "y.npy"], capsys)
    result = load_predictor(model).evaluate(X, y)
    assert code == 0 and err == ""
    assert out.splitlines() == [
        "Task: regression", "Samples: 64", f"MSE: {result.mse:.6g}", f"MAE: {result.mae:.6g}",
        f"Baseline MSE: {result.baseline_mse:.6g}", f"Loss: {result.loss:.6g}",
    ]


def test_one_dimensional_and_column_targets_are_the_same_evaluation(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y1.npy", y)
    np.save(tmp_path / "y2.npy", y[:, None])
    assert evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y1.npy") == evaluate_json(
        capsys, model, tmp_path / "X.npy", tmp_path / "y2.npy"
    )


def test_multi_output_regression_is_scored_over_samples_and_outputs(tmp_path, capsys):
    model = save_regression_artifact(tmp_path, outputs=2)
    X, y = regression_data(outputs=2)
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)
    payload = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy")

    predictions = independent_native_prediction(model, X)
    assert predictions.shape == (64, 2)
    assert payload["mse"] == pytest.approx(float(np.mean((predictions - y) ** 2)), rel=1e-5)
    assert payload["baseline_mse"] == pytest.approx(float(np.mean((y - y.mean(axis=0)) ** 2)), rel=1e-6)
    assert_matches_result(payload, load_predictor(model).evaluate(X, y))


# ========================================================================= M116: native units, unaware CLI


def test_standardized_regression_artifact_is_reported_in_native_units(m116_artifact, tmp_path, capsys):
    """The most important M117 test. The model itself speaks z-scores (|output| < ~20); the target is ~2e5.
    The CLI has no transform code, yet its metrics are in dollars-squared, equal to what the API reports and to
    an independent NumPy calculation from the artifact's raw metadata."""
    held_X, held_y = big_target_data(120, seed=42)
    np.save(tmp_path / "X.npy", held_X)
    np.save(tmp_path / "y.npy", held_y)
    payload = evaluate_json(capsys, m116_artifact, tmp_path / "X.npy", tmp_path / "y.npy")

    predictions = independent_native_prediction(m116_artifact, held_X)[:, 0]
    oracle_mse = float(np.mean((predictions - held_y) ** 2))
    oracle_mae = float(np.mean(np.abs(predictions - held_y)))
    assert isinstance(load_predictor(m116_artifact).target_transform, StandardizeTarget)   # the fixture is a real M116 one
    assert payload["baseline_mse"] > 1e9 and payload["mse"] > 1e3   # z-space numbers would be < ~10
    assert payload["mse"] == pytest.approx(oracle_mse, rel=1e-5)
    assert payload["mae"] == pytest.approx(oracle_mae, rel=1e-5)
    assert payload["loss"] == pytest.approx(oracle_mse, rel=1e-4)
    assert payload["baseline_mse"] == pytest.approx(float(np.mean((held_y - held_y.mean()) ** 2)), rel=1e-12)
    assert payload["mse"] < 0.2 * payload["baseline_mse"]            # and it really learned the native target

    assert_matches_result(payload, load_predictor(m116_artifact).evaluate(held_X, held_y))


def test_standardized_regression_human_output_is_native_units_too(m116_artifact, tmp_path, capsys):
    held_X, held_y = big_target_data(120, seed=42)
    np.save(tmp_path / "X.npy", held_X)
    np.save(tmp_path / "y.npy", held_y)
    code, out, err = run_cli(["model", "evaluate", m116_artifact, tmp_path / "X.npy", tmp_path / "y.npy"], capsys)
    assert code == 0, err
    printed = {ln.split(": ")[0]: float(ln.split(": ")[1]) for ln in out.splitlines() if ln.startswith(("MSE", "MAE"))}
    predictions = independent_native_prediction(m116_artifact, held_X)[:, 0]
    assert printed["MSE"] == pytest.approx(float(np.mean((predictions - held_y) ** 2)), rel=1e-4)   # ".6g" text
    assert printed["MAE"] == pytest.approx(float(np.mean(np.abs(predictions - held_y))), rel=1e-4)


def test_the_standardized_artifact_is_unchanged_by_evaluating_it(m116_artifact, tmp_path, capsys):
    held_X, held_y = big_target_data(50, seed=7)
    np.save(tmp_path / "X.npy", held_X)
    np.save(tmp_path / "y.npy", held_y)
    before = sha256(m116_artifact)
    first = evaluate_json(capsys, m116_artifact, tmp_path / "X.npy", tmp_path / "y.npy")
    assert sha256(m116_artifact) == before
    assert evaluate_json(capsys, m116_artifact, tmp_path / "X.npy", tmp_path / "y.npy") == first


# ============================================================================================ image folders


def test_image_classification_matches_the_api_and_matches_folders_by_class_name(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)              # artifact order: dog, cat  (ImageFolder order: cat, dog)
    root = _image_root(tmp_path)
    payload = evaluate_json(capsys, model, root)

    predictor = load_predictor(model)
    expected = np.zeros((2, 2), dtype=np.int64)
    for class_dir in root.iterdir():
        for image in class_dir.iterdir():
            expected[predictor.classes.index(class_dir.name), predictor.predict(str(image)).index] += 1
    assert payload["classes"] == ["dog", "cat"]
    assert payload["confusion_matrix"] == expected.tolist()
    assert payload["support"] == expected.sum(axis=1).tolist()
    assert_matches_result(payload, predictor.evaluate(str(root)))


def test_image_classification_may_omit_a_class_the_artifact_has(tmp_path, capsys):
    model = _save_image_artifact(tmp_path, classes=("dog", "cat", "bird"))
    payload = evaluate_json(capsys, model, _image_root(tmp_path))
    assert payload["classes"] == ["dog", "cat", "bird"] and payload["support"][2] == 0


def test_image_human_readable_output_lists_the_artifacts_classes(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    code, out, err = run_cli(["model", "evaluate", model, _image_root(tmp_path)], capsys)
    assert code == 0, err
    assert out.startswith("Task: classification\nSamples: ")
    lines = out.splitlines()
    start = lines.index("Classes:") + 2
    assert [ln.split()[0] for ln in lines[start : start + 2]] == ["dog", "cat"]   # the artifact's order, not ImageFolder's


# ================================================================================================== JSON


def test_json_mode_prints_exactly_one_json_document_of_plain_types(scoring, capsys):
    model, X, y = scoring
    code, out, err = run_cli(["model", "evaluate", model, X, y, "--json"], capsys)
    assert code == 0 and err == ""
    payload = json.loads(out, parse_constant=_reject_constant)          # the whole of stdout, nothing else
    assert list(payload) == [
        "task", "samples", "loss", "accuracy", "baseline_accuracy", "classes", "confusion_matrix",
        "precision", "recall", "support",
    ]
    assert isinstance(payload["samples"], int) and isinstance(payload["loss"], float)
    assert all(isinstance(v, int) for row in payload["confusion_matrix"] for v in row)
    assert all(isinstance(v, str) for v in payload["classes"])


def test_regression_json_keys_are_documented_ones(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)
    assert list(evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy")) == [
        "task", "samples", "loss", "mse", "mae", "baseline_mse",
    ]


def test_numpy_scalars_in_a_result_still_serialise(monkeypatch, scoring, capsys):
    model, X, y = scoring
    result = ClassificationEvaluationResult(
        task="tabular_classification", samples=np.int64(3), loss=np.float32(0.5), accuracy=np.float64(2 / 3),
        baseline_accuracy=np.float32(0.5), classes=("a", "b"), confusion_matrix=np.array([[1, 0], [1, 1]]),
        precision=(np.float32(0.5), np.float64(1.0)), recall=(1.0, np.float32(0.5)), support=(np.int64(1), np.int64(2)),
    )
    monkeypatch.setattr(ArtifactPredictor, "evaluate", lambda self, *a, **k: result)
    payload = evaluate_json(capsys, model, X, y)
    assert payload["samples"] == 3 and payload["confusion_matrix"] == [[1, 0], [1, 1]] and payload["support"] == [1, 2]


def test_a_non_finite_metric_is_an_error_not_invalid_json(monkeypatch, tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)
    bad = RegressionEvaluationResult(task="regression", samples=1, loss=float("nan"), mse=1.0, mae=1.0, baseline_mse=1.0)
    monkeypatch.setattr(ArtifactPredictor, "evaluate", lambda self, *a, **k: bad)
    code, out, err = run_cli(["model", "evaluate", model, tmp_path / "X.npy", tmp_path / "y.npy", "--json"], capsys)
    assert_clean_error(code, out, err, match="non-finite")


# ==================================================================== thinness: a door, not a second engine


def test_the_cli_prints_whatever_the_api_returns_and_hands_it_the_files_untouched(monkeypatch, scoring, capsys):
    """Replace `ArtifactPredictor.evaluate()` with a sentinel. If the CLI computed anything itself, or
    preprocessed/converted the arrays, the printed numbers or the received arrays would differ."""
    model, X, y = scoring
    sentinel = ClassificationEvaluationResult(
        task="tabular_classification", samples=12345, loss=0.123456, accuracy=0.654321, baseline_accuracy=0.111111,
        classes=("p", "q"), confusion_matrix=np.array([[7, 8], [9, 10]], dtype=np.int64),
        precision=(0.25, 0.75), recall=(0.5, 0.125), support=(15, 19),
    )
    seen = []

    def spy(self, features, targets=None, **kwargs):
        seen.append((features, targets, kwargs))
        return sentinel

    monkeypatch.setattr(ArtifactPredictor, "evaluate", spy)
    payload = evaluate_json(capsys, model, X, y)
    assert_matches_result(payload, sentinel)

    assert len(seen) == 1
    features, targets, kwargs = seen[0]
    np.testing.assert_array_equal(features, np.load(X))
    np.testing.assert_array_equal(targets, np.load(y))
    assert features.dtype == np.load(X).dtype and kwargs == {}


def test_the_artifact_is_loaded_exactly_once(monkeypatch, scoring, capsys):
    """One `load_model()` per invocation -- never once per batch or per sample."""
    import forge.serialization.model as serialization_model

    real_load_model, calls = serialization_model.load_model, []

    def counting_load_model(*args, **kwargs):
        calls.append(args)
        return real_load_model(*args, **kwargs)

    monkeypatch.setattr(serialization_model, "load_model", counting_load_model)
    model, X, y = scoring
    assert evaluate_json(capsys, model, X, y, "--batch-size", "1")["samples"] == 8
    assert len(calls) == 1


# ============================================================================================ read-only


def test_evaluation_leaves_every_file_untouched(scoring, capsys):
    model, X, y = scoring
    root = Path(model).parent
    before = tree_state(root)
    for extra in ([], ["--json"], ["--device", "cpu"]):
        assert run_cli(["model", "evaluate", model, X, y, *extra], capsys)[0] == 0
    assert tree_state(root) == before                     # artifact + inputs byte-identical, no new files


def test_image_evaluation_leaves_the_artifact_and_the_image_tree_untouched(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    root = _image_root(tmp_path)
    before = tree_state(tmp_path)
    assert run_cli(["model", "evaluate", model, root, "--json"], capsys)[0] == 0
    assert tree_state(tmp_path) == before


def test_repeated_evaluation_is_deterministic(scoring, capsys):
    model, X, y = scoring
    assert evaluate_json(capsys, model, X, y) == evaluate_json(capsys, model, X, y)


# ===================================================================================================== errors


def test_missing_artifact(tmp_path, capsys):
    code, out, err = run_cli(["model", "evaluate", tmp_path / "nope.forge", tmp_path / "X.npy", tmp_path / "y.npy"], capsys)
    assert_clean_error(code, out, err, match="artifact not found")


def test_malformed_artifact(tmp_path, scoring, capsys):
    _, X, y = scoring
    (tmp_path / "junk.forge").write_bytes(b"this is not a zip archive")
    code, out, err = run_cli(["model", "evaluate", tmp_path / "junk.forge", X, y, "--json"], capsys)
    assert_clean_error(code, out, err)


def test_missing_input_and_missing_targets_files(scoring, tmp_path, capsys):
    model, X, y = scoring
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path / "gone.npy", y], capsys), match="input file not found")
    assert_clean_error(*run_cli(["model", "evaluate", model, X, tmp_path / "gone.npy"], capsys), match="targets file not found")


def test_a_tabular_artifact_without_a_targets_argument(scoring, capsys):
    model, X, _ = scoring
    assert_clean_error(*run_cli(["model", "evaluate", model, X], capsys), match="needs a TARGETS file")


def test_incompatible_feature_count(scoring, tmp_path, capsys):
    model, _, y = scoring
    np.save(tmp_path / "wide.npy", np.zeros((8, 5), dtype=np.float32))
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path / "wide.npy", y, "--json"], capsys))


def test_a_one_dimensional_input_is_refused(scoring, tmp_path, capsys):
    model, _, y = scoring
    np.save(tmp_path / "flat.npy", np.zeros(8, dtype=np.float32))
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path / "flat.npy", y], capsys), match="batched")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_features_and_targets(bad, scoring, tmp_path, capsys):
    model, X, y = scoring
    poisoned = _X.copy()
    poisoned[2, 1] = bad
    np.save(tmp_path / "bad_X.npy", poisoned)
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path / "bad_X.npy", y], capsys), match="non-finite")

    regression = save_regression_artifact(tmp_path)
    Xr, yr = regression_data()
    yr[3] = bad
    np.save(tmp_path / "Xr.npy", Xr)
    np.save(tmp_path / "yr.npy", yr)
    assert_clean_error(*run_cli(["model", "evaluate", regression, tmp_path / "Xr.npy", tmp_path / "yr.npy"], capsys), match="non-finite")


def test_invalid_target_shapes_and_counts(scoring, tmp_path, capsys):
    model, X, y = scoring
    np.save(tmp_path / "y2d.npy", np.zeros((8, 2), dtype=np.int64))
    assert_clean_error(*run_cli(["model", "evaluate", model, X, tmp_path / "y2d.npy"], capsys), match="1-D")
    np.save(tmp_path / "short.npy", np.array(_TRUE[:5]))
    assert_clean_error(*run_cli(["model", "evaluate", model, X, tmp_path / "short.npy"], capsys), match="8 sample")

    regression = save_regression_artifact(tmp_path)
    Xr, yr = regression_data()
    np.save(tmp_path / "Xr.npy", Xr)
    np.save(tmp_path / "wide_y.npy", np.zeros((64, 3), dtype=np.float32))
    assert_clean_error(*run_cli(["model", "evaluate", regression, tmp_path / "Xr.npy", tmp_path / "wide_y.npy"], capsys), match="one target per model output")
    np.save(tmp_path / "cube.npy", np.zeros((64, 1, 1), dtype=np.float32))
    assert_clean_error(*run_cli(["model", "evaluate", regression, tmp_path / "Xr.npy", tmp_path / "cube.npy"], capsys))


def test_unknown_class_labels_and_out_of_range_indices(scoring, tmp_path, capsys):
    model, X, _ = scoring
    np.save(tmp_path / "unknown.npy", np.array(["zebra", "apple", "mango", "unused", "okapi", "zebra", "zebra", "apple"]))
    assert_clean_error(*run_cli(["model", "evaluate", model, X, tmp_path / "unknown.npy"], capsys), match="okapi")
    np.save(tmp_path / "range.npy", np.array([0, 1, 2, 3, 4, 0, 0, 1]))
    assert_clean_error(*run_cli(["model", "evaluate", model, X, tmp_path / "range.npy"], capsys), match="outside")


def test_corrupted_and_unsupported_input_files(scoring, tmp_path, capsys):
    model, X, y = scoring
    (tmp_path / "garbage.npy").write_bytes(b"\x93NUMPY but then nothing sensible")
    (tmp_path / "text.csv").write_text("1,2\n3,4\n")
    np.savez(tmp_path / "bundle.npz", X=_X)
    np.save(tmp_path / "objects.npy", np.array([{"a": 1}, None], dtype=object), allow_pickle=True)
    (tmp_path / "empty.npy").write_bytes(b"")
    for bad in ("garbage.npy", "text.csv", "bundle.npz", "objects.npy", "empty.npy"):
        code, out, err = run_cli(["model", "evaluate", model, tmp_path / bad, y], capsys)
        assert_clean_error(code, out, err, match="readable .npy" if bad != "bundle.npz" else "not a single .npy")
    assert_clean_error(*run_cli(["model", "evaluate", model, X, tmp_path / "text.csv"], capsys), match="targets file")
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path, y], capsys), match="input file not found")


def test_bad_image_directories(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    not_a_dir = tmp_path / "file.png"
    not_a_dir.write_bytes(b"x")
    assert_clean_error(*run_cli(["model", "evaluate", model, not_a_dir], capsys), match="not a directory")
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path / "missing_dir"], capsys), match="not a directory")

    (tmp_path / "empty_root").mkdir()
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path / "empty_root"], capsys), match="no class subdirectories")

    (tmp_path / "no_images" / "cat").mkdir(parents=True)
    assert_clean_error(*run_cli(["model", "evaluate", model, tmp_path / "no_images"], capsys), match="no images")

    root = _image_root(tmp_path, {"cat": [30], "dog": [200]})
    (root / "cat" / "broken.png").write_bytes(b"not an image")
    assert_clean_error(*run_cli(["model", "evaluate", model, root], capsys), match="could not read image")


def test_image_directory_with_an_unexpected_class_folder(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    root = _image_root(tmp_path, {"cat": [30], "dog": [200], "wolf": [128]})
    assert_clean_error(*run_cli(["model", "evaluate", model, root], capsys), match="wolf")


def test_an_image_artifact_does_not_take_a_targets_file(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    np.save(tmp_path / "y.npy", np.array([0, 1]))
    assert_clean_error(*run_cli(["model", "evaluate", model, _image_root(tmp_path), tmp_path / "y.npy"], capsys), match="no TARGETS")


def test_tasks_without_evaluation_semantics_and_task_less_artifacts(tmp_path, scoring, capsys):
    _, X, y = scoring
    forge.random.seed(0)
    from forge.nn import Conv2d

    segmentation = tmp_path / "seg.forge"
    save_model(Sequential(Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), Conv2d(4, 1, kernel_size=3, padding=1)),
               str(segmentation), preprocessing=Normalize(mean=0.0, std=255.0), task="segmentation")
    assert_clean_error(*run_cli(["model", "evaluate", segmentation, X, y], capsys), match="'segmentation'")

    legacy = tmp_path / "legacy.forge"
    save_model(Sequential(Linear(4, 1)), str(legacy))
    assert_clean_error(*run_cli(["model", "evaluate", legacy, X, y], capsys), match="does not declare a task")


def test_a_classification_artifact_saved_without_classes_is_a_persistence_error(tmp_path, capsys):
    path = tmp_path / "no_classes.forge"
    save_model(Sequential(Linear(2, 2)), str(path), task="tabular_classification")
    np.save(tmp_path / "X.npy", np.zeros((3, 2), dtype=np.float32))
    np.save(tmp_path / "y.npy", np.array([0, 1, 0]))
    assert_clean_error(*run_cli(["model", "evaluate", path, tmp_path / "X.npy", tmp_path / "y.npy"], capsys), match="class list")


def test_a_failed_json_run_prints_nothing_to_stdout(scoring, tmp_path, capsys):
    model, X, _ = scoring
    np.save(tmp_path / "unknown.npy", np.array(["nope"] * 8))
    code, out, err = run_cli(["model", "evaluate", model, X, tmp_path / "unknown.npy", "--json"], capsys)
    assert_clean_error(code, out, err)


def test_malformed_invocations_are_argparse_usage_errors(scoring, capsys):
    model, X, y = scoring
    for argv in (["model", "evaluate"], ["model", "evaluate", model], ["model", "evaluate", model, X, y, "extra"],
                 ["model", "evaluate", model, X, y, "--device", "tpu"], ["model", "evaluate", model, X, y, "--batch-size", "x"]):
        with pytest.raises(SystemExit) as exc:
            cli_main([str(a) for a in argv])
        assert exc.value.code == 2
        assert "Traceback" not in capsys.readouterr().err


# ============================================================================================ fresh process


def test_fresh_process_agrees_with_the_python_api(scoring, m116_artifact, tmp_path):
    model, X, y = scoring
    held_X, held_y = big_target_data(60, seed=9)
    np.save(tmp_path / "hX.npy", held_X)
    np.save(tmp_path / "hy.npy", held_y)

    def run(*args):
        completed = subprocess.run(
            [sys.executable, "-m", "forge", "model", "evaluate", *map(str, args), "--json"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    assert_matches_result(run(model, X, y), load_predictor(model).evaluate(_X, np.array(_TRUE)))
    assert_matches_result(run(m116_artifact, tmp_path / "hX.npy", tmp_path / "hy.npy"),
                          load_predictor(m116_artifact).evaluate(held_X, held_y))


def test_fresh_process_errors_have_no_traceback_and_a_nonzero_status(scoring, tmp_path):
    model, X, _ = scoring
    completed = subprocess.run(
        [sys.executable, "-m", "forge", "model", "evaluate", model, str(X)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 1 and completed.stdout == ""
    assert completed.stderr.startswith("Error: ") and "Traceback" not in completed.stderr
