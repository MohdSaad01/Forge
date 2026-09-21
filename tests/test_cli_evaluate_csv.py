"""Milestone 118 tests: `forge model evaluate MODEL DATA.csv --target COLUMN`.

```text
forge model evaluate MODEL DATA.csv --target COLUMN
    -> forge.data.load_csv() -> (X, y) -> load_predictor() -> ArtifactPredictor.evaluate() -> printed result
```

CSV is an *input format* for the M117 command, so the contract is that nothing else about evaluation moved:

- **parity** -- the CSV run's numbers are the `.npy` run's (JSON equal, printed text equal) and the Python API's,
  and are checked against hand-computed values, not just against each other;
- **thinness** -- a sentinel replaces `ArtifactPredictor.evaluate()`; whatever it returns is what the CLI prints
  and the arrays it receives are exactly the CSV's values (no CLI-side preprocessing, metrics or class logic);
- **independence** -- the reader knows nothing of artifacts; an M116 `target_transform="standardize"` artifact is
  reported in native units with no CSV-side involvement, and the class order is the artifact's, never the file's;
- **errors** -- every user mistake is one `Error: ...` line naming the file/line/column, exit 1, empty stdout
  (also under `--json`), never a traceback;
- **compatibility** -- `.npy` and image evaluation, and artifacts saved before this milestone, behave as before;
- **read-only** -- the artifact and the CSV are byte-identical afterwards.

CUDA parity lives in `tests/test_csv_tabular_workflow_cuda.py`; the installed-wheel run in `tests/test_packaging_smoke.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.training import ClassificationEvaluationResult, load_predictor
from forge.training.inference import ArtifactPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_artifact_evaluation import (  # noqa: E402  (shared hand-computed fixtures)
    _CLASSES, _CONFUSION, _TRUE, _X, _image_root, _save_image_artifact, _save_scoring_artifact,
)
from test_cli_evaluate import (  # noqa: E402
    assert_clean_error, assert_matches_result, evaluate_json, regression_data, run_cli, save_regression_artifact,
    sha256, train_standardized_regressor, tree_state,
)
from test_csv_tabular_workflow import write_csv  # noqa: E402
from test_target_transform import big_target_data, independent_native_prediction  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_CLASSIFIER = REPO_ROOT / "models" / "tabular_classifier" / "diabetes_classifier.forge"
PIMA_HOLDOUT = REPO_ROOT / "examples" / "tabular_diabetes" / "data" / "diabetes_holdout_eval.csv"


def evaluate_csv_json(capsys, model, csv_path, *extra, target="label"):
    return evaluate_json(capsys, model, csv_path, "--target", target, *extra)


def scoring_csv(tmp_path, labels=None, *, name="scoring.csv", order=None):
    """`_X` (2 features) with a label column *between* them, so the target is neither first nor last."""
    labels = list(_TRUE) if labels is None else labels
    rows = np.arange(len(_X)) if order is None else np.asarray(order)
    return write_csv(tmp_path / name, {
        "x0": _X[rows, 0].astype(np.float64), "label": np.array(labels, dtype=object)[rows], "x1": _X[rows, 1].astype(np.float64),
    })


@pytest.fixture()
def scoring(tmp_path):
    return _save_scoring_artifact(tmp_path), scoring_csv(tmp_path)


@pytest.fixture(scope="module")
def m116_artifact(tmp_path_factory) -> str:
    return train_standardized_regressor(tmp_path_factory.mktemp("cli_csv_m116"))


def housing_like_csv(tmp_path, X, y, name="held_out.csv"):
    """The target is the FIRST column, as in the StatLib California housing file."""
    return write_csv(tmp_path / name, {"median_house_value": y, **{f"f{i}": X[:, i] for i in range(X.shape[1])}})


# ============================================================================================ classification


def test_csv_classification_matches_hand_computation_the_npy_run_and_the_python_api(scoring, tmp_path, capsys):
    model, csv_path = scoring
    payload = evaluate_csv_json(capsys, model, csv_path)

    assert payload["task"] == "tabular_classification" and payload["samples"] == 8
    assert payload["classes"] == _CLASSES                                   # the artifact's order, not the file's
    assert payload["confusion_matrix"] == _CONFUSION.tolist()
    assert payload["support"] == [4, 3, 1, 0]
    assert payload["accuracy"] == pytest.approx(6 / 8)                      # only true if the persisted Normalize ran
    assert payload["baseline_accuracy"] == pytest.approx(4 / 8)
    assert payload["precision"] == pytest.approx([3 / 3, 2 / 3, 1 / 2, 0.0])
    assert payload["recall"] == pytest.approx([3 / 4, 2 / 3, 1 / 1, 0.0])

    np.save(tmp_path / "X.npy", _X)
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    assert payload == evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy")     # CSV == .npy, exactly
    assert_matches_result(payload, load_predictor(model).evaluate(_X, np.array(_TRUE)))


def test_the_target_column_is_left_out_of_the_features(scoring, capsys):
    """Mutation 1: a target left among the features would fail the feature-count check (or, worse, not)."""
    model, csv_path = scoring
    assert csv_path.read_text().splitlines()[0] == "x0,label,x1"
    assert evaluate_csv_json(capsys, model, csv_path)["samples"] == 8      # 2 features, as the artifact expects


def test_feature_column_order_is_the_files_order(scoring, tmp_path, capsys):
    """Mutation 2: this artifact scores zebra=z0, apple=z1, so swapping the two columns changes the answer."""
    model, csv_path = scoring
    swapped = write_csv(tmp_path / "swapped.csv", {
        "x1": _X[:, 1].astype(np.float64), "x0": _X[:, 0].astype(np.float64), "label": np.array(_TRUE, dtype=object),
    })
    same, different = evaluate_csv_json(capsys, model, csv_path), evaluate_csv_json(capsys, model, swapped)
    assert same["confusion_matrix"] != different["confusion_matrix"]
    # ... and an alphabetical sort of the columns ("label", "x0", "x1" -> x0, x1) would have kept them right:
    # here the *file* order (x1, x0) is what is used, so the two differ, exactly as the NumPy arrays would.
    np.save(tmp_path / "Xs.npy", _X[:, ::-1])
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    assert different == evaluate_json(capsys, model, tmp_path / "Xs.npy", tmp_path / "y.npy")


def test_integer_and_whole_number_float_labels_are_class_indices(scoring, tmp_path, capsys):
    model, csv_path = scoring
    expected = evaluate_csv_json(capsys, model, csv_path)
    indices = [_CLASSES.index(t) for t in _TRUE]
    as_int = scoring_csv(tmp_path, [str(i) for i in indices], name="idx.csv")
    as_float = scoring_csv(tmp_path, [f"{i}.0" for i in indices], name="idx_float.csv")
    assert evaluate_csv_json(capsys, model, as_int) == expected
    assert evaluate_csv_json(capsys, model, as_float) == expected


def test_the_class_order_is_the_artifacts_whatever_order_the_rows_come_in(scoring, tmp_path, capsys):
    model, csv_path = scoring
    expected = evaluate_csv_json(capsys, model, csv_path)
    shuffled = scoring_csv(tmp_path, name="shuffled.csv", order=[6, 2, 7, 0, 5, 3, 1, 4])   # 'apple' is not first any more
    got = evaluate_csv_json(capsys, model, shuffled)
    assert got["classes"] == _CLASSES and got == expected                   # the confusion matrix is indexed by the artifact


def test_human_readable_output_is_the_same_text_as_the_npy_run(scoring, tmp_path, capsys):
    model, csv_path = scoring
    code, csv_out, err = run_cli(["model", "evaluate", model, csv_path, "--target", "label"], capsys)
    assert code == 0, err
    np.save(tmp_path / "X.npy", _X)
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    assert run_cli(["model", "evaluate", model, tmp_path / "X.npy", tmp_path / "y.npy"], capsys)[1] == csv_out
    assert "Accuracy: 75.00%" in csv_out and "Confusion matrix" in csv_out


def test_batch_size_device_and_json_flags_work_with_csv(scoring, capsys):
    model, csv_path = scoring
    base = evaluate_csv_json(capsys, model, csv_path)
    assert evaluate_csv_json(capsys, model, csv_path, "--batch-size", "3", "--device", "cpu") == base


# ============================================================================================ regression


def test_csv_regression_matches_numpy_the_npy_run_and_the_python_api(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data()
    csv_path = housing_like_csv(tmp_path, X, y)
    payload = evaluate_csv_json(capsys, model, csv_path, target="median_house_value")

    from forge.serialization import load_model, load_preprocessing
    from forge.tensor.tensor import Tensor
    from forge.training import predict

    pre, net = load_preprocessing(model), load_model(model, device="cpu")
    out = predict(net, pre(Tensor(X.astype(np.float32)))).numpy()[:, 0].astype(np.float64)
    assert payload["mse"] == pytest.approx(float(np.mean((out - y) ** 2)), rel=1e-5)
    assert payload["mae"] == pytest.approx(float(np.mean(np.abs(out - y))), rel=1e-5)
    assert payload["baseline_mse"] == pytest.approx(float(np.mean((y - y.mean()) ** 2)), rel=1e-5)

    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)
    assert payload == evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy")
    assert_matches_result(payload, load_predictor(model).evaluate(X, y))


def test_a_standardized_regression_artifact_is_reported_in_native_units_from_a_csv(m116_artifact, tmp_path, capsys):
    """M116 x M118. The model speaks z-scores (|output| < ~20); the CSV target is ~2e5. Neither the reader nor the
    CLI has transform code, yet the metrics are in dollars-squared, equal to a by-hand NumPy calculation."""
    held_X, held_y = big_target_data(120, seed=42)
    csv_path = housing_like_csv(tmp_path, held_X, held_y)
    payload = evaluate_csv_json(capsys, m116_artifact, csv_path, target="median_house_value")

    predictions = independent_native_prediction(m116_artifact, held_X)[:, 0]
    assert isinstance(load_predictor(m116_artifact).target_transform, forge.data.StandardizeTarget)
    assert payload["baseline_mse"] > 1e9 and payload["mse"] > 1e3           # z-space numbers would be < ~10
    assert payload["mse"] == pytest.approx(float(np.mean((predictions - held_y) ** 2)), rel=1e-5)
    assert payload["mae"] == pytest.approx(float(np.mean(np.abs(predictions - held_y))), rel=1e-5)
    assert payload["baseline_mse"] == pytest.approx(float(np.mean((held_y - held_y.mean()) ** 2)), rel=1e-12)
    assert payload["mse"] < 0.2 * payload["baseline_mse"]

    np.save(tmp_path / "X.npy", held_X)
    np.save(tmp_path / "y.npy", held_y)
    assert payload == evaluate_json(capsys, m116_artifact, tmp_path / "X.npy", tmp_path / "y.npy")
    assert_matches_result(payload, load_predictor(m116_artifact).evaluate(held_X, held_y))


def test_the_standardized_text_output_is_native_units_too(m116_artifact, tmp_path, capsys):
    held_X, held_y = big_target_data(120, seed=42)
    csv_path = housing_like_csv(tmp_path, held_X, held_y)
    code, out, err = run_cli(["model", "evaluate", m116_artifact, csv_path, "--target", "median_house_value"], capsys)
    assert code == 0, err
    printed = {ln.split(": ")[0]: float(ln.split(": ")[1]) for ln in out.splitlines() if ln.startswith(("MSE", "MAE"))}
    predictions = independent_native_prediction(m116_artifact, held_X)[:, 0]
    assert printed["MSE"] == pytest.approx(float(np.mean((predictions - held_y) ** 2)), rel=1e-4)
    assert printed["MAE"] == pytest.approx(float(np.mean(np.abs(predictions - held_y))), rel=1e-4)


def test_the_reader_source_knows_nothing_about_artifacts():
    """Data ingestion and artifact semantics stay independent: none of these concepts appear in the reader's code."""
    from test_csv_reader import reader_code_without_docstrings

    code = reader_code_without_docstrings()
    for concept in ("target_transform", "StandardizeTarget", "load_predictor", "ArtifactPredictor", "preprocessing",
                    "Normalize", "classes", "save_model", "inspect_model", "forge.training", "forge.serialization"):
        assert concept not in code, concept


def test_a_multi_output_artifact_does_not_take_a_single_csv_target(tmp_path, capsys):
    model = save_regression_artifact(tmp_path, outputs=2)
    X, y = regression_data()
    code, out, err = run_cli(["model", "evaluate", model, housing_like_csv(tmp_path, X, y), "--target", "median_house_value"], capsys)
    assert_clean_error(code, out, err, match="ArtifactPredictor.evaluate()")      # the evaluation API's own message


# ============================================================================================ thinness


def test_the_cli_prints_whatever_the_api_returns_and_hands_it_the_csv_values_untouched(monkeypatch, scoring, capsys):
    """Replace `ArtifactPredictor.evaluate()` with a sentinel. Any CLI-side metric, preprocessing, class ordering or
    conversion would change the printed numbers or the arrays received."""
    model, csv_path = scoring
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
    payload = evaluate_csv_json(capsys, model, csv_path)
    assert_matches_result(payload, sentinel)

    assert len(seen) == 1
    features, targets, kwargs = seen[0]
    np.testing.assert_array_equal(features, _X.astype(np.float64))         # raw file values: no Normalize, target removed
    # Milestone 119: the one thing the CLI now also forwards is the header names of those columns (target removed,
    # file order) -- it still never reorders or drops anything itself; `ArtifactPredictor` matches them by name.
    assert features.shape == (8, 2) and kwargs == {"feature_names": ["x0", "x1"]}
    assert targets.tolist() == list(_TRUE)                                # verbatim names: no class-index mapping either


def test_a_regression_target_reaches_evaluate_as_the_raw_native_numbers(monkeypatch, m116_artifact, tmp_path, capsys):
    from forge.training import RegressionEvaluationResult

    held_X, held_y = big_target_data(30, seed=5)
    csv_path = housing_like_csv(tmp_path, held_X, held_y)
    seen = []
    sentinel = RegressionEvaluationResult(task="regression", samples=99, loss=1.5, mse=2.5, mae=3.5, baseline_mse=4.5)

    def spy(self, features, targets=None, **kwargs):
        seen.append((features, targets))
        return sentinel

    monkeypatch.setattr(ArtifactPredictor, "evaluate", spy)
    assert_matches_result(evaluate_csv_json(capsys, m116_artifact, csv_path, target="median_house_value"), sentinel)
    np.testing.assert_array_equal(seen[0][1], held_y)                     # never standardized by the CSV path
    np.testing.assert_array_equal(seen[0][0], held_X)
    assert seen[0][1].dtype == np.float64


def test_the_artifact_is_loaded_and_evaluated_exactly_once(monkeypatch, scoring, capsys):
    import forge.cli.model as cli_model

    real_load_predictor, real_evaluate = cli_model.load_predictor, ArtifactPredictor.evaluate
    loads, evaluations = [], []

    def counting_load_predictor(*args, **kwargs):
        loads.append(args)
        return real_load_predictor(*args, **kwargs)

    def counting_evaluate(self, *args, **kwargs):
        evaluations.append(args)
        return real_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(cli_model, "load_predictor", counting_load_predictor)
    monkeypatch.setattr(ArtifactPredictor, "evaluate", counting_evaluate)
    model, csv_path = scoring
    assert evaluate_csv_json(capsys, model, csv_path, "--batch-size", "1")["samples"] == 8
    assert len(loads) == 1 and len(evaluations) == 1


def test_the_cli_reads_the_csv_through_the_public_reader_once(monkeypatch, scoring, capsys):
    import forge.cli.model as cli_model

    calls = []
    real = cli_model.load_csv
    monkeypatch.setattr(cli_model, "load_csv", lambda *a, **k: calls.append((a, k)) or real(*a, **k))
    model, csv_path = scoring
    evaluate_csv_json(capsys, model, csv_path)
    # Milestone 119: also asks the reader for the header names (`return_feature_names=True`); still one read.
    assert len(calls) == 1 and calls[0][1] == {"target": "label", "labels": True, "return_feature_names": True}


# ============================================================================================ dispatch + compatibility


def test_the_extension_alone_picks_the_reader(scoring, tmp_path, capsys):
    model, csv_path = scoring
    upper = tmp_path / "SCORING.CSV"
    upper.write_bytes(csv_path.read_bytes())
    assert evaluate_csv_json(capsys, model, upper) == evaluate_csv_json(capsys, model, csv_path)
    # CSV *text* in a file that is not called .csv is not sniffed: it is read as .npy, and refused as such.
    txt = tmp_path / "scoring.txt"
    txt.write_bytes(csv_path.read_bytes())
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    assert_clean_error(*run_cli(["model", "evaluate", model, txt, tmp_path / "y.npy"], capsys), match="readable .npy")
    # and .npy bytes in a file called .csv are read as CSV (and refused as CSV).
    fake = tmp_path / "fake.csv"
    np.save(tmp_path / "real.npy", _X)
    fake.write_bytes((tmp_path / "real.npy").read_bytes())
    assert_clean_error(*run_cli(["model", "evaluate", model, fake, "--target", "label"], capsys), match="fake.csv")


def test_npy_evaluation_is_unchanged_by_csv_support(scoring, tmp_path, capsys):
    model, _ = scoring
    np.save(tmp_path / "X.npy", _X)
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    payload = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy")
    assert payload["confusion_matrix"] == _CONFUSION.tolist() and payload["accuracy"] == pytest.approx(0.75)
    assert set(payload) == {"task", "samples", "loss", "accuracy", "baseline_accuracy", "classes", "confusion_matrix",
                            "precision", "recall", "support"}


def test_the_json_keys_are_the_result_dataclasses_and_not_csv_metadata(scoring, tmp_path, capsys):
    """CSV must not add keys (file name, columns, target name, row counts) to the M117 JSON contract."""
    model, csv_path = scoring
    payload = evaluate_csv_json(capsys, model, csv_path)
    assert set(payload) == {"task", "samples", "loss", "accuracy", "baseline_accuracy", "classes", "confusion_matrix",
                            "precision", "recall", "support"}
    regression = save_regression_artifact(tmp_path)
    X, y = regression_data()
    payload = evaluate_csv_json(capsys, regression, housing_like_csv(tmp_path, X, y), target="median_house_value")
    assert set(payload) == {"task", "samples", "loss", "mse", "mae", "baseline_mse"}


def test_an_artifact_saved_before_this_milestone_evaluates_a_csv_like_its_npy_files(capsys, tmp_path):
    """The bundled Pima classifier (M92-era) and the bundled 154-row holdout CSV -- real data, unchanged files."""
    holdout = np.genfromtxt(PIMA_HOLDOUT, delimiter=",", skip_header=1)
    np.save(tmp_path / "X.npy", holdout[:, :-1])
    np.save(tmp_path / "y.npy", holdout[:, -1].astype(int))
    before = (sha256(BUNDLED_CLASSIFIER), sha256(PIMA_HOLDOUT))
    payload = evaluate_json(capsys, BUNDLED_CLASSIFIER, PIMA_HOLDOUT, "--target", "Outcome")
    assert payload == evaluate_json(capsys, BUNDLED_CLASSIFIER, tmp_path / "X.npy", tmp_path / "y.npy")
    assert payload["samples"] == 154 and payload["baseline_accuracy"] == pytest.approx(96 / 154)
    assert payload["accuracy"] > payload["baseline_accuracy"]
    assert_matches_result(payload, load_predictor(str(BUNDLED_CLASSIFIER)).evaluate(holdout[:, :-1], holdout[:, -1].astype(int)))
    assert (sha256(BUNDLED_CLASSIFIER), sha256(PIMA_HOLDOUT)) == before


def test_image_evaluation_is_unchanged_and_refuses_csv_arguments(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    root = _image_root(tmp_path)
    assert run_cli(["model", "evaluate", model, root, "--json"], capsys)[0] == 0
    assert_clean_error(*run_cli(["model", "evaluate", model, root, "--target", "label"], capsys), match="--target")
    csv_path = write_csv(tmp_path / "labels.csv", {"a": np.array([1.0, 2.0]), "label": np.array(["cat", "dog"], dtype=object)})
    assert_clean_error(*run_cli(["model", "evaluate", model, csv_path, "--target", "label"], capsys), match="--target")
    assert_clean_error(*run_cli(["model", "evaluate", model, csv_path], capsys), match="not a directory")


# ============================================================================================ read-only


def test_evaluating_a_csv_leaves_every_file_untouched(scoring, capsys):
    model, csv_path = scoring
    root = Path(model).parent
    before = tree_state(root)
    for extra in ([], ["--json"], ["--device", "cpu"]):
        assert run_cli(["model", "evaluate", model, csv_path, "--target", "label", *extra], capsys)[0] == 0
    assert tree_state(root) == before


def test_repeated_csv_evaluation_is_deterministic(scoring, capsys):
    model, csv_path = scoring
    assert evaluate_csv_json(capsys, model, csv_path) == evaluate_csv_json(capsys, model, csv_path)


# ============================================================================================ errors


def bad(tmp_path, text, name="bad.csv", *, raw=None):
    path = tmp_path / name
    path.write_bytes(raw if raw is not None else text.encode("utf-8"))
    return path


def clean_error(capsys, model, *argv, match):
    """The same mistake with and without --json: one `Error:` line, exit 1, nothing on stdout."""
    for extra in ([], ["--json"]):
        code, out, err = run_cli(["model", "evaluate", model, *argv, *extra], capsys)
        assert_clean_error(code, out, err, match=match)
        assert err.count("\n") == 1 and err.endswith("\n")             # a single line


@pytest.mark.parametrize("name, text, match", [
    ("header_only", "x0,label,x1\n", "no data rows"),
    ("empty", "", "is empty"),
    ("no_header", "1,0,2.5\n3,1,4\n", "looks like data, not a header"),
    ("duplicate", "x0,x0,label\n1,2,zebra\n", "duplicate column name"),
    ("two_targets", "x0,label,label\n1,zebra,apple\n", "'label' appears 2 times"),
    ("no_target_col", "x0,x1,kind\n1,2,zebra\n", "target column 'label' is not in the header"),
    ("target_only", "label\nzebra\n", "no feature columns"),
    ("ragged", "x0,label,x1\n1,zebra,2\n3,apple\n", "line 3: 2 field(s)"),
    ("text_feature", "x0,label,x1\n1,zebra,two\n", "'two' is not a number"),
    ("empty_field", "x0,label,x1\n1,zebra,\n", "empty feature"),
    ("empty_label", "x0,label,x1\n1,,2\n", "empty target"),
    ("nan", "x0,label,x1\n1,zebra,nan\n", "not a finite number"),
    ("inf", "x0,label,x1\n-inf,zebra,2\n", "not a finite number"),
    ("bad_quote", 'x0,label,x1\n1,"zebra,2\n', "malformed CSV"),
    ("mixed_labels", "x0,label,x1\n1,zebra,2\n3,4,5\n", "mixes numbers and text"),
    ("fractional_label", "x0,label,x1\n1,0.5,2\n", "not a class index"),
    ("semicolons", "x0;label;x1\n1;zebra;2\n", "comma-delimited"),
])
def test_bad_csv_files_are_one_clean_named_error(scoring, tmp_path, capsys, name, text, match):
    model, _ = scoring
    path = bad(tmp_path, text, f"{name}.csv")
    clean_error(capsys, model, path, "--target", "label", match=match)
    clean_error(capsys, model, path, "--target", "label", match=f"{name}.csv")


def test_invalid_utf8_and_a_directory_and_a_missing_file(scoring, tmp_path, capsys):
    model, _ = scoring
    clean_error(capsys, model, bad(tmp_path, "", "latin1.csv", raw=b"x0,label,x1\n1,caf\xe9,2\n"), "--target", "label",
                match="not valid UTF-8")
    (tmp_path / "folder.csv").mkdir()
    clean_error(capsys, model, tmp_path / "folder.csv", "--target", "label", match="not a file")
    clean_error(capsys, model, tmp_path / "missing.csv", "--target", "label", match="CSV file not found")


def test_argument_combinations_that_make_no_sense(scoring, tmp_path, capsys):
    model, csv_path = scoring
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    np.save(tmp_path / "X.npy", _X)
    clean_error(capsys, model, csv_path, match="needs --target COLUMN")
    clean_error(capsys, model, csv_path, tmp_path / "y.npy", "--target", "label", match="carries its own targets")
    clean_error(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--target", "label", match="names a column of a .csv INPUT")
    clean_error(capsys, model, tmp_path / "X.npy", "--target", "label", match="names a column of a .csv INPUT")


def test_an_unknown_target_name_lists_what_the_file_has(scoring, capsys):
    model, csv_path = scoring
    code, out, err = run_cli(["model", "evaluate", model, csv_path, "--target", "Label"], capsys)
    assert_clean_error(code, out, err, match="'Label' is not in the header")
    assert "'x0', 'label', 'x1'" in err


def test_a_csv_with_the_wrong_number_of_feature_columns_is_the_evaluation_apis_error(scoring, tmp_path, capsys):
    model, _ = scoring
    three = write_csv(tmp_path / "three.csv", {
        "a": np.zeros(3), "b": np.ones(3), "c": np.ones(3), "label": np.array(["zebra", "apple", "mango"], dtype=object),
    })
    code, out, err = run_cli(["model", "evaluate", model, three, "--target", "label"], capsys)
    assert_clean_error(code, out, err, match="feature")
    assert "2" in err and "3" in err


def test_class_names_the_artifact_does_not_have(scoring, tmp_path, capsys):
    model, _ = scoring
    unknown = scoring_csv(tmp_path, ["zebra", "apple", "mango", "okapi", "zebra", "zebra", "apple", "zebra"], name="okapi.csv")
    clean_error(capsys, model, unknown, "--target", "label", match="okapi")
    out_of_range = scoring_csv(tmp_path, ["0", "1", "2", "4", "0", "0", "1", "0"], name="range.csv")
    clean_error(capsys, model, out_of_range, "--target", "label", match="outside")


def test_a_regression_csv_with_a_text_target_names_the_cell(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data(8)
    csv_path = write_csv(tmp_path / "text_target.csv", {
        "price": np.array([str(v) for v in y[:7]] + ["expensive"], dtype=object), **{f"f{i}": X[:, i] for i in range(4)},
    })
    for extra in ([], ["--json"]):
        code, out, err = run_cli(["model", "evaluate", model, csv_path, "--target", "price", *extra], capsys)
        assert_clean_error(code, out, err, match="line 9, column 'price': 'expensive' is not a number")
    assert "a numeric target was required" in err


def test_a_nan_in_a_regression_target_is_refused_before_evaluation(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data(8)
    values = [str(v) for v in y]
    values[2] = "NaN"
    csv_path = write_csv(tmp_path / "nan_target.csv", {"price": np.array(values, dtype=object), **{f"f{i}": X[:, i] for i in range(4)}})
    clean_error(capsys, model, csv_path, "--target", "price", match="line 4, column 'price': 'NaN' is not a finite number")


def test_a_classification_artifact_needs_labels_and_a_regression_one_needs_numbers(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data(8)
    labelled = write_csv(tmp_path / "labelled.csv", {"kind": np.array(["a", "b"] * 4, dtype=object), **{f"f{i}": X[:, i] for i in range(4)}})
    clean_error(capsys, model, labelled, "--target", "kind", match="'a' is not a number")


def test_the_image_artifact_needs_no_csv_columns(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    clean_error(capsys, model, tmp_path / "x.csv", "--target", "label", match="--target")


# ============================================================================================ fresh process


def test_fresh_process_csv_evaluation_agrees_with_the_python_api(scoring, m116_artifact, tmp_path):
    import json

    model, csv_path = scoring
    held_X, held_y = big_target_data(60, seed=9)
    housing = housing_like_csv(tmp_path, held_X, held_y)

    def run(*args):
        completed = subprocess.run(
            [sys.executable, "-m", "forge", "model", "evaluate", *map(str, args), "--json"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    assert_matches_result(run(model, csv_path, "--target", "label"), load_predictor(model).evaluate(_X, np.array(_TRUE)))
    assert_matches_result(run(m116_artifact, housing, "--target", "median_house_value"),
                          load_predictor(m116_artifact).evaluate(held_X, held_y))


def test_fresh_process_csv_errors_have_no_traceback_and_a_nonzero_status(scoring, tmp_path):
    model, _ = scoring
    path = bad(tmp_path, "x0,label,x1\n1,zebra,oops\n")
    completed = subprocess.run(
        [sys.executable, "-m", "forge", "model", "evaluate", model, str(path), "--target", "label"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert completed.returncode == 1 and completed.stdout == ""
    assert completed.stderr.startswith("Error: ") and "Traceback" not in completed.stderr
    assert "line 2, column 'x1': 'oops' is not a number" in completed.stderr
