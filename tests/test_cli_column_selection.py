"""Milestone 120 tests: `forge model evaluate` / `forge model predict --columns` and `forge model convert --feature-names`.

Two independent CLI additions, both closing the same real-world friction (a production CSV that carries a
non-feature column, e.g. `id`) without inventing a second reordering mechanism:

- `--columns NAME [NAME ...]` on `evaluate`/`predict`: selects and orders exactly those CSV header columns as
  features (`forge.data.load_csv(..., columns=...)` / `load_csv_features(..., columns=...)`), then the *same*
  M119 name-alignment (`_column_order()`) that already runs for every named input takes over -- selection and
  alignment stay two separate steps, never fused, and `predict`/`evaluate` cannot apply different rules because
  neither implements one.
- `--feature-names NAME [NAME ...]` on `convert`: attaches (or replaces) an artifact's feature names without
  retraining -- a metadata-only edit riding on `convert`'s existing reload/re-save, so it is safe by
  construction (the model weights it reloads and re-saves are simply not touched) and is proven so below.

Named-input rejection semantics (unknown/missing/extra/case-different column) are already pinned by
`test_cli_feature_schema.py` and are not re-tested here -- these tests are about *selection*, which sits one
step earlier in the pipeline.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.cli.main import main as cli_main

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema_support import FEATURES, make_data, metadata_of, train_classifier, train_regressor  # noqa: E402

X, Y_REG, Y_CLS = make_data()
TASKS = ["regression", "tabular_classification"]


def run_cli(argv, capsys):
    code = cli_main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_csv(path: Path, header, rows, target=None) -> Path:
    names = list(header) + ([target[0]] if target else [])
    lines = [",".join(names)]
    for i, row in enumerate(rows):
        cells = [repr(float(v)) for v in row]
        if target:
            cells.append(str(target[1][i]) if target[1].dtype.kind in "iu" else repr(float(target[1][i])))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli_columns")
    train_regressor(root / "reg.forge")
    train_classifier(root / "cls.forge")
    train_regressor(root / "reg_plain.forge", names=None)
    train_classifier(root / "cls_plain.forge", names=None)

    ids = np.arange(len(X))
    targets = {"regression": ("target", Y_REG), "tabular_classification": ("target", Y_CLS)}
    files = {}
    for task, target in targets.items():
        files[task] = {
            "with_id": write_csv(root / f"{task}_id.csv", ["id", *FEATURES], np.column_stack([ids, X]), target),
            "with_id_reordered": write_csv(
                root / f"{task}_id_reordered.csv", ["id", *reversed(FEATURES)],
                np.column_stack([ids, X[:, ::-1]]), target,
            ),
            "ordered": write_csv(root / f"{task}_ordered.csv", FEATURES, X, target),
        }
    predict = {
        "with_id": write_csv(root / "p_id.csv", ["id", *FEATURES], np.column_stack([ids[:5], X[:5]])),
        "json": root / "p.json",
    }
    predict["json"].write_text("[[1.0, 2.0, 3.0, 4.0]]")
    return {"root": root, "eval": files, "predict": predict}


def model(world, task, named=True):
    stem = "reg" if task == "regression" else "cls"
    return world["root"] / f"{stem}{'' if named else '_plain'}.forge"


def assert_clean_error(code, out, err, *expected):
    assert code == 1 and out == "", (code, out)
    assert err.startswith("Error: ") and "Traceback" not in err and err.count("\n") == 1, err
    for text in expected:
        assert text in err, err


# ============================================================================== evaluate --columns


@pytest.mark.parametrize("task", TASKS)
def test_evaluate_without_columns_on_an_id_csv_is_rejected(capsys, world, task):
    code, out, err = run_cli(
        ["model", "evaluate", model(world, task), world["eval"][task]["with_id"], "--target", "target"], capsys,
    )
    assert_clean_error(code, out, err, "unexpected", "'id'")


@pytest.mark.parametrize("task", TASKS)
def test_evaluate_with_columns_excludes_the_id_and_matches_the_id_free_csv(capsys, world, task):
    code, out, err = run_cli(
        ["model", "evaluate", model(world, task), world["eval"][task]["with_id"], "--target", "target",
         "--columns", *FEATURES, "--json"], capsys,
    )
    assert code == 0 and err == ""
    ordered_code, ordered_out, _ = run_cli(
        ["model", "evaluate", model(world, task), world["eval"][task]["ordered"], "--target", "target", "--json"],
        capsys,
    )
    assert ordered_code == 0 and out == ordered_out


@pytest.mark.parametrize("task", TASKS)
def test_evaluate_columns_order_is_authoritative_not_file_order(capsys, world, task):
    """The id-and-reversed-header CSV, selected in FEATURES order, must equal the ordered CSV -- selection re-orders."""
    code, out, err = run_cli(
        ["model", "evaluate", model(world, task), world["eval"][task]["with_id_reordered"], "--target", "target",
         "--columns", *FEATURES, "--json"], capsys,
    )
    ordered_code, ordered_out, _ = run_cli(
        ["model", "evaluate", model(world, task), world["eval"][task]["ordered"], "--target", "target", "--json"],
        capsys,
    )
    assert code == 0 and err == "" and out == ordered_out


def test_evaluate_columns_duplicate_is_one_clear_error(capsys, world):
    code, out, err = run_cli(
        ["model", "evaluate", model(world, "regression"), world["eval"]["regression"]["with_id"], "--target", "target",
         "--columns", *FEATURES, FEATURES[0]], capsys,
    )
    assert_clean_error(code, out, err, "unique")


def test_evaluate_columns_unknown_is_one_clear_error(capsys, world):
    code, out, err = run_cli(
        ["model", "evaluate", model(world, "regression"), world["eval"]["regression"]["with_id"], "--target", "target",
         "--columns", *FEATURES, "nope"], capsys,
    )
    assert_clean_error(code, out, err, "'nope'")


def test_evaluate_columns_including_the_target_is_rejected(capsys, world):
    code, out, err = run_cli(
        ["model", "evaluate", model(world, "regression"), world["eval"]["regression"]["with_id"], "--target", "target",
         "--columns", *FEATURES, "target"], capsys,
    )
    assert_clean_error(code, out, err, "target column")


def test_evaluate_columns_rejected_for_npy_input(capsys, world, tmp_path):
    np.save(tmp_path / "x.npy", X)
    np.save(tmp_path / "y.npy", Y_REG)
    code, out, err = run_cli(
        ["model", "evaluate", model(world, "regression"), tmp_path / "x.npy", tmp_path / "y.npy",
         "--columns", *FEATURES], capsys,
    )
    assert_clean_error(code, out, err, "--columns", ".npy")


# ============================================================================== predict --columns


def test_predict_without_columns_on_an_id_csv_is_rejected(capsys, world):
    code, out, err = run_cli(["model", "predict", model(world, "regression"), world["predict"]["with_id"]], capsys)
    assert_clean_error(code, out, err, "unexpected", "'id'")


def test_predict_with_columns_excludes_the_id(capsys, world):
    code, out, err = run_cli(
        ["model", "predict", model(world, "regression"), world["predict"]["with_id"], "--columns", *FEATURES, "--json"],
        capsys,
    )
    assert code == 0 and err == ""


def test_predict_columns_rejected_for_json_input(capsys, world):
    code, out, err = run_cli(
        ["model", "predict", model(world, "regression"), world["predict"]["json"], "--columns", *FEATURES], capsys,
    )
    assert_clean_error(code, out, err, "--columns", "JSON")


def test_predict_columns_rejected_when_task_has_no_columns(capsys, tmp_path):
    """An artifact whose task is not a numeric-feature one (here: no task at all) is refused before file I/O."""
    import forge.nn as nn
    from forge import save_model

    m = nn.Sequential(nn.Linear(4, 2))
    path = tmp_path / "untaksed.forge"
    save_model(m, str(path))
    code, out, err = run_cli(["model", "predict", path, "x.csv", "--columns", "a", "b"], capsys)
    assert code == 1 and out == "" and "task" in err


# ============================================================================== convert --feature-names (retrofit)


def test_convert_feature_names_attaches_names_to_an_unnamed_artifact(capsys, world, tmp_path):
    out_path = tmp_path / "retro.forge"
    code, out, err = run_cli(
        ["model", "convert", model(world, "regression", named=False), "--device", "cpu", "--output", out_path,
         "--feature-names", *FEATURES], capsys,
    )
    assert code == 0 and err == ""
    info = forge.inspect_model(str(out_path))
    assert info.input_schema.feature_names == tuple(FEATURES)


def test_convert_feature_names_wrong_count_is_rejected_and_writes_nothing(capsys, world, tmp_path):
    out_path = tmp_path / "bad.forge"
    code, out, err = run_cli(
        ["model", "convert", model(world, "regression", named=False), "--device", "cpu", "--output", out_path,
         "--feature-names", "a", "b"], capsys,
    )
    assert code == 1 and "4 input feature" in err
    assert not out_path.exists()


def test_convert_feature_names_duplicate_is_rejected(capsys, world, tmp_path):
    out_path = tmp_path / "bad2.forge"
    code, out, err = run_cli(
        ["model", "convert", model(world, "regression", named=False), "--device", "cpu", "--output", out_path,
         "--feature-names", "a", "a", "b", "c"], capsys,
    )
    assert code == 1 and "unique" in err
    assert not out_path.exists()


def test_convert_feature_names_changes_no_weights_and_no_predictions(world, tmp_path):
    src = model(world, "regression", named=False)
    out_path = tmp_path / "retro2.forge"
    cli_main(["model", "convert", str(src), "--device", "cpu", "--output", str(out_path),
              "--feature-names", *FEATURES])

    def param_hashes(path):
        import hashlib
        with zipfile.ZipFile(str(path)) as zf:
            return {n: hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist() if n != "metadata.json"}

    assert param_hashes(src) == param_hashes(out_path)

    before, after = forge.load_predictor(str(src)), forge.load_predictor(str(out_path))
    p1 = before.predict(X).numpy()
    p2 = after.predict(X).numpy()
    assert np.array_equal(p1, p2)


def test_convert_without_feature_names_keeps_whatever_the_artifact_already_had(capsys, world, tmp_path):
    named_out = tmp_path / "still_named.forge"
    run_cli(["model", "convert", model(world, "regression", named=True), "--device", "cpu", "--output", named_out], capsys)
    assert forge.inspect_model(str(named_out)).input_schema.feature_names == tuple(FEATURES)

    unnamed_out = tmp_path / "still_unnamed.forge"
    run_cli(["model", "convert", model(world, "regression", named=False), "--device", "cpu", "--output", unnamed_out], capsys)
    schema = forge.inspect_model(str(unnamed_out)).input_schema
    assert schema is None or schema.feature_names is None


def test_convert_feature_names_replaces_existing_names(capsys, world, tmp_path):
    out_path = tmp_path / "renamed.forge"
    new_names = [f"col{i}" for i in range(len(FEATURES))]
    code, out, err = run_cli(
        ["model", "convert", model(world, "regression", named=True), "--device", "cpu", "--output", out_path,
         "--feature-names", *new_names], capsys,
    )
    assert code == 0 and err == ""
    assert forge.inspect_model(str(out_path)).input_schema.feature_names == tuple(new_names)
