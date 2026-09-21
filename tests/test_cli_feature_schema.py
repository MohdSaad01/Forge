"""Milestone 119 tests: `forge model evaluate` / `forge model predict` with named CSV columns.

The CLI is an input boundary. It reads a CSV (`load_csv(..., return_feature_names=True)` /
`load_csv_features()`), hands the arrays *and their header names* to the one place that knows the rule
(`ArtifactPredictor`), and prints what comes back. So this file pins three things:

- **the behaviour** -- a reordered CSV scores/predicts exactly like the ordered one; an unknown, missing or extra
  (`id`, or a left-in target) column is one `Error:` line, exit 1, empty stdout (also under `--json`);
- **thinness** -- the CLI never reorders, drops or renames a column itself: it forwards the header names untouched;
- **one rule** -- `predict` and `evaluate` produce the same complaint for the same defect.

`.npy` and JSON inputs have no column names, so they are unchanged (and a reordered `.npy` is, as documented,
accepted unchecked).
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
from forge.training.inference import ArtifactPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema_support import (  # noqa: E402
    CLASSES, FEATURES, make_data, permute, train_classifier, train_regressor, write_named_csv,
)

X, Y_REG, Y_CLS = make_data()
PERM = ["rooms", "income", "distance", "age"]
NPY_PERM = ["distance", "rooms", "age", "income"]


def run_cli(argv, capsys):
    code = cli_main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli_schema")
    train_regressor(root / "reg.forge")
    train_classifier(root / "cls.forge")
    train_regressor(root / "reg_plain.forge", names=None)
    train_classifier(root / "cls_plain.forge", names=None)

    targets = {"regression": ("target", Y_REG), "tabular_classification": ("target", Y_CLS)}
    files = {}
    for task, target in targets.items():
        files[task] = {
            "ordered": write_named_csv(root / f"{task}_ordered.csv", FEATURES, X, target),
            "reordered": write_named_csv(root / f"{task}_reordered.csv", PERM, permute(X, FEATURES, PERM), target),
            "unknown": write_named_csv(root / f"{task}_unknown.csv", ["income", "age", "bedrooms", "distance"], X, target),
            "extra_id": write_named_csv(root / f"{task}_id.csv", ["id", *FEATURES], np.column_stack([np.arange(len(X)), X]), target),
            "id_replaces": write_named_csv(root / f"{task}_idr.csv", ["id", *FEATURES[:3]], np.column_stack([np.arange(len(X)), X[:, :3]]), target),
            "missing": write_named_csv(root / f"{task}_missing.csv", FEATURES[:3], X[:, :3], target),
            "case": write_named_csv(root / f"{task}_case.csv", ["Income", "age", "rooms", "distance"], X, target),
        }
    # feature-only files for `predict`
    predict = {
        "ordered": write_named_csv(root / "p_ordered.csv", FEATURES, X[:5]),
        "reordered": write_named_csv(root / "p_reordered.csv", PERM, permute(X[:5], FEATURES, PERM)),
        "unknown": write_named_csv(root / "p_unknown.csv", ["income", "age", "bedrooms", "distance"], X[:5]),
        "extra_id": write_named_csv(root / "p_id.csv", ["id", *FEATURES], np.column_stack([np.arange(5), X[:5]])),
        "with_target": write_named_csv(root / "p_target.csv", FEATURES, X[:5], ("target", Y_REG[:5])),
        "missing": write_named_csv(root / "p_missing.csv", FEATURES[:3], X[:5, :3]),
        "case": write_named_csv(root / "p_case.csv", ["Income", "age", "rooms", "distance"], X[:5]),
    }
    # The CSV reader strips a header cell's surrounding whitespace (its documented M118 contract), so through a CSV a
    # name can only differ from the artifact's by whitespace *inside* it: 'dis tance' is not 'distance'.
    (root / "p_inner.csv").write_text("income,age,rooms,dis tance\n" + ",".join(["1"] * 4) + "\n")
    predict["inner_space"] = root / "p_inner.csv"
    return {"root": root, "eval": files, "predict": predict}


def model(world, task, named=True):
    stem = "reg" if task == "regression" else "cls"
    return world["root"] / f"{stem}{'' if named else '_plain'}.forge"


TASKS = ["regression", "tabular_classification"]


def evaluate(capsys, world, task, csv, *flags, named=True):
    return run_cli(["model", "evaluate", model(world, task, named), world["eval"][task][csv], "--target", "target", *flags], capsys)


def predict(capsys, world, task, name, *flags, named=True):
    return run_cli(["model", "predict", model(world, task, named), world["predict"][name], *flags], capsys)


def assert_clean_error(code, out, err, *expected):
    assert code == 1 and out == "", (code, out)
    assert err.startswith("Error: ") and "Traceback" not in err and err.count("\n") == 1, err
    for text in expected:
        assert text in err, err


# ============================================================================== evaluate: reordered CSV


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("flags", [["--json"], []], ids=["json", "text"])
def test_evaluating_a_reordered_csv_prints_exactly_what_the_ordered_csv_prints(capsys, world, task, flags):
    ordered = evaluate(capsys, world, task, "ordered", *flags)
    reordered = evaluate(capsys, world, task, "reordered", *flags)
    assert ordered[0] == reordered[0] == 0 and ordered[2] == reordered[2] == ""        # no warning: the artifact has names
    assert ordered[1] == reordered[1] and ordered[1] != ""


@pytest.mark.parametrize("task", TASKS)
def test_the_reordered_csv_result_equals_the_python_api_on_the_ordered_arrays(capsys, world, task):
    code, out, _ = evaluate(capsys, world, task, "reordered", "--json")
    y = Y_REG if task == "regression" else Y_CLS
    reference = forge.load_predictor(str(model(world, task))).evaluate(X, y)
    payload = json.loads(out)
    assert code == 0 and payload["samples"] == len(X)
    for key in ("loss", "accuracy", "mse", "mae"):
        if key in payload:
            assert payload[key] == getattr(reference, key), key


@pytest.mark.parametrize("task", TASKS)
def test_the_same_csv_columns_reordered_would_have_scored_worse_without_the_names(capsys, world, task):
    """Non-vacuous: through the unnamed `.npy` door the same reordered values are accepted and scored wrongly."""
    y = Y_REG if task == "regression" else Y_CLS
    np.save(world["root"] / "moved.npy", permute(X, FEATURES, NPY_PERM))
    np.save(world["root"] / f"{task}_y.npy", y)
    code, out, err = run_cli(["model", "evaluate", model(world, task), world["root"] / "moved.npy",
                              world["root"] / f"{task}_y.npy", "--json"], capsys)
    assert code == 0 and err == ""                                        # unchecked: the documented .npy limitation
    _, ordered, _ = evaluate(capsys, world, task, "ordered", "--json")
    assert json.loads(out) != json.loads(ordered)


# ============================================================================== evaluate: rejected


@pytest.mark.parametrize("flags", [[], ["--json"]], ids=["text", "json"])
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("csv, expected", [
    ("unknown", ("missing", "['rooms']", "unexpected", "['bedrooms']")),
    ("extra_id", ("unexpected", "['id']", "never drops")),
    ("id_replaces", ("missing", "['distance']", "unexpected", "['id']")),
    ("missing", ("missing", "['distance']")),
    ("case", ("'Income' vs 'income' differ only in case/whitespace",)),
])
def test_a_csv_that_does_not_have_the_artifacts_columns_is_one_clear_error(capsys, world, task, csv, expected, flags):
    code, out, err = evaluate(capsys, world, task, csv, *flags)
    assert_clean_error(code, out, err, "ArtifactPredictor.evaluate()", "feature names", *expected)


def test_a_duplicate_csv_header_is_still_refused_at_the_reader(capsys, world):
    bad = world["root"] / "dup.csv"
    bad.write_text("income,income,age,rooms,distance,target\n1,2,3,4,5,6\n")
    code, out, err = run_cli(["model", "evaluate", model(world, "regression"), bad, "--target", "target"], capsys)
    assert_clean_error(code, out, err, "duplicate column name(s) ['income']")


def test_a_malformed_csv_is_still_one_reader_error(capsys, world):
    bad = world["root"] / "bad.csv"
    bad.write_text("income,age,rooms,distance,target\n1,2,3,oops,6\n")
    code, out, err = run_cli(["model", "evaluate", model(world, "regression"), bad, "--target", "target"], capsys)
    assert_clean_error(code, out, err, "line 2, column 'distance': 'oops' is not a number")


# ============================================================================== evaluate: thinness (the CLI never aligns)


@pytest.mark.parametrize("task", TASKS)
def test_the_cli_forwards_the_header_names_and_the_raw_csv_order_untouched(monkeypatch, capsys, world, task):
    seen = {}

    def spy(self, X_, y_=None, *, feature_names=None, **kwargs):
        seen.update(X=np.asarray(X_), names=feature_names)
        raise forge.DataError("stop here")

    monkeypatch.setattr(ArtifactPredictor, "evaluate", spy)
    evaluate(capsys, world, task, "reordered")
    assert seen["names"] == PERM                                            # the header, target removed, file order
    assert np.array_equal(seen["X"], permute(X, FEATURES, PERM))            # values NOT reordered by the CLI


def test_a_npy_evaluation_passes_no_names_at_all(monkeypatch, capsys, world):
    seen = {}

    def spy(self, X_, y_=None, *, feature_names="unset", **kwargs):
        seen["names"] = feature_names
        raise forge.DataError("stop here")

    monkeypatch.setattr(ArtifactPredictor, "evaluate", spy)
    np.save(world["root"] / "x.npy", X)
    np.save(world["root"] / "y.npy", Y_REG)
    run_cli(["model", "evaluate", model(world, "regression"), world["root"] / "x.npy", world["root"] / "y.npy"], capsys)
    assert seen["names"] == "unset"                                         # not passed at all: reaches the API exactly as in M117


@pytest.mark.parametrize("task", TASKS)
def test_npy_evaluation_of_a_named_artifact_is_unchanged(capsys, world, task):
    y = Y_REG if task == "regression" else Y_CLS
    np.save(world["root"] / "x.npy", X)
    np.save(world["root"] / "y.npy", y)
    code, out, err = run_cli(["model", "evaluate", model(world, task), world["root"] / "x.npy", world["root"] / "y.npy", "--json"], capsys)
    assert code == 0 and err == ""
    reference = forge.load_predictor(str(model(world, task))).evaluate(X, y)
    assert json.loads(out)["samples"] == reference.samples == len(X)


# ============================================================================== predict: CSV input


def _values(out, task):
    payload = json.loads(out)
    return payload["prediction"] if task == "regression" else payload["predictions"]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("flags", [["--json"], []], ids=["json", "text"])
def test_predicting_a_reordered_csv_prints_exactly_what_the_ordered_csv_prints(capsys, world, task, flags):
    ordered = predict(capsys, world, task, "ordered", *flags)
    reordered = predict(capsys, world, task, "reordered", *flags)
    assert ordered[0] == reordered[0] == 0 and ordered[2] == reordered[2] == ""
    assert ordered[1] == reordered[1] and ordered[1] != ""


@pytest.mark.parametrize("task", TASKS)
def test_csv_predictions_equal_the_python_api_on_the_ordered_rows(capsys, world, task):
    _, out, _ = predict(capsys, world, task, "reordered", "--json")
    expected = forge.load_predictor(str(model(world, task))).predict(X[:5])
    if task == "regression":
        assert np.array_equal(np.array(_values(out, task), dtype=np.float32), expected.numpy())
    else:
        assert [p["class"] for p in _values(out, task)] == [p.label for p in expected]
        assert [p["confidence"] for p in _values(out, task)] == [p.confidence for p in expected]


@pytest.mark.parametrize("flags", [[], ["--json"]], ids=["text", "json"])
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("name, expected", [
    ("unknown", ("['rooms']", "['bedrooms']")),
    ("extra_id", ("unexpected", "['id']")),
    ("with_target", ("unexpected", "['target']")),         # a target left in the file is an extra column, not silently dropped
    ("missing", ("missing", "['distance']")),
    ("case", ("'Income' vs 'income' differ only in case/whitespace",)),
    ("inner_space", ("'dis tance' vs 'distance' differ only in case/whitespace",)),
])
def test_predict_rejects_the_same_defects(capsys, world, task, name, expected, flags):
    code, out, err = predict(capsys, world, task, name, *flags)
    assert_clean_error(code, out, err, "feature names", *expected)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("name", ["unknown", "extra_id", "missing", "case"])
def test_predict_and_evaluate_apply_one_rule_and_say_the_same_thing(capsys, world, task, name):
    """The same defect gives the same complaint on both commands, up to the name of the function that noticed it."""
    _, _, predict_err = predict(capsys, world, task, name)
    _, _, evaluate_err = evaluate(capsys, world, task, name)

    def tail(message):
        return message.split("() ", 1)[1]

    assert tail(predict_err) == tail(evaluate_err) and "feature names" in tail(evaluate_err)


def test_the_cli_forwards_predict_names_untouched(monkeypatch, capsys, world):
    seen = {}
    import forge.cli.model as cli_model

    def spy(path, data, *, feature_names="unset", **kwargs):
        seen.update(data=np.asarray(data), names=feature_names)
        raise forge.DataError("stop here")

    monkeypatch.setattr(cli_model, "predict_model", spy)
    predict(capsys, world, "regression", "reordered")
    assert seen["names"] == PERM and np.array_equal(seen["data"], permute(X[:5], FEATURES, PERM))


@pytest.mark.parametrize("task", TASKS)
def test_json_input_to_predict_is_unchanged_and_carries_no_names(monkeypatch, capsys, world, task):
    row = world["root"] / "row.json"
    row.write_text(json.dumps(X[:1].tolist()))
    seen = {}
    real = forge.cli.model.predict_model

    def spy(path, data, **kwargs):
        seen.update(kwargs)
        return real(path, data, **kwargs)

    monkeypatch.setattr(forge.cli.model, "predict_model", spy)
    code, out, err = run_cli(["model", "predict", model(world, task), row, "--json"], capsys)
    assert code == 0 and err == "" and "feature_names" not in seen          # not passed at all: reaches the API as before
    expected = forge.load_predictor(str(model(world, task))).predict(X[:1].astype(np.float32))
    if task == "regression":
        assert np.allclose(json.loads(out)["prediction"], expected.numpy())
    else:
        assert json.loads(out)["predictions"][0]["class"] == expected[0].label


def test_a_multi_row_csv_prints_one_block_per_row(capsys, world):
    code, out, _ = predict(capsys, world, "tabular_classification", "reordered")
    assert code == 0 and out.count("Prediction:") == 5 and "Sample 0:" in out and "Sample 4:" in out


def test_a_non_csv_input_is_read_as_json_exactly_as_before(capsys, world):
    (world["root"] / "not.txt").write_text("income,age\n1,2\n")
    code, out, err = run_cli(["model", "predict", model(world, "regression"), world["root"] / "not.txt"], capsys)
    assert_clean_error(code, out, err, "input must contain numeric JSON data.")


# ============================================================================== an artifact with no names


@pytest.mark.parametrize("task", TASKS)
def test_a_csv_against_an_unnamed_artifact_works_and_warns_on_stderr_only(capsys, world, task):
    """M118 behaviour is preserved (the CSV is used as given) plus one honest line: the order could not be verified."""
    code, out, err = evaluate(capsys, world, task, "ordered", "--json", named=False)
    assert code == 0
    assert err.startswith("Warning: ") and err.count("\n") == 1 and "records no feature names" in err
    assert "UserWarning" not in err and ".py" not in err                    # a note, not Python's file:line noise
    json.loads(out)                                                          # stdout is still valid JSON
    assert out == run_cli(["model", "evaluate", model(world, task, named=False),
                           world["eval"][task]["ordered"], "--target", "target", "--json"], capsys)[1]


@pytest.mark.parametrize("task", TASKS)
def test_predict_csv_against_an_unnamed_artifact_warns_too(capsys, world, task):
    code, out, err = predict(capsys, world, task, "ordered", "--json", named=False)
    assert code == 0 and err.startswith("Warning: ") and "records no feature names" in err
    json.loads(out)


@pytest.mark.parametrize("task", TASKS)
def test_an_unnamed_artifact_still_only_checks_the_width_for_a_csv(capsys, world, task):
    """Unchanged M118 behaviour for artifacts saved before names: an extra column is a width error, a swap is accepted."""
    code, out, err = evaluate(capsys, world, task, "extra_id", named=False)
    assert_clean_error(code, out, err, "expected 4 input feature(s), received 5")
    swapped = evaluate(capsys, world, task, "reordered", "--json", named=False)
    assert swapped[0] == 0                                                   # accepted, unchecked: the documented limitation


def test_the_warning_reaches_a_real_subprocess_as_one_plain_line(world):
    """End to end through the console entry point (`python -m forge`), not in-process."""
    completed = subprocess.run(
        [sys.executable, "-m", "forge", "model", "evaluate", str(model(world, "regression", named=False)),
         str(world["eval"]["regression"]["ordered"]), "--target", "target", "--json"],
        capture_output=True, text=True, cwd=str(world["root"]),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.startswith("Warning: ") and completed.stderr.count("\n") == 1
    json.loads(completed.stdout)


# ============================================================================== read-only, convert


@pytest.mark.parametrize("task", TASKS)
def test_nothing_is_written(capsys, world, task):
    before = (sha(model(world, task)), sha(world["eval"][task]["reordered"]), sha(world["predict"]["reordered"]))
    evaluate(capsys, world, task, "reordered")
    predict(capsys, world, task, "reordered")
    assert (sha(model(world, task)), sha(world["eval"][task]["reordered"]), sha(world["predict"]["reordered"])) == before


def test_a_converted_artifact_still_checks_columns(capsys, world):
    converted = world["root"] / "converted.forge"
    code, _, err = run_cli(["model", "convert", model(world, "regression"), "--device", "cpu", "--output", converted], capsys)
    assert code == 0, err
    ok = run_cli(["model", "evaluate", converted, world["eval"]["regression"]["reordered"], "--target", "target", "--json"], capsys)
    assert ok[0] == 0 and ok[2] == ""
    bad = run_cli(["model", "evaluate", converted, world["eval"]["regression"]["extra_id"], "--target", "target"], capsys)
    assert_clean_error(*bad, "['id']")
