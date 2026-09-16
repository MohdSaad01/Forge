"""Milestone 101 tests: the portable structural input contract --

```text
save_model(..., task="regression"/"tabular_classification")
    -> inspect_model().input_schema (ModelInfo.input_schema, InputSchema)
    -> predict_tensor_artifact() / predict_tabular_classification_artifact()
       validate input_data's last-axis width against it, before
       preprocessing or the model ever run
```

Covers: `InputSchema` derivation from already-persisted architecture
metadata (no format change); the explicit task/architecture gating that
keeps this from being fabricated for classification/segmentation/sequence
artifacts or legacy (no-`task=`) artifacts; the real, reproduced failure
modes this closes (a wrong feature count previously reached `Normalize`/
`Linear` and failed there with a confusing shape error) and the one it does
**not** close (same-length feature reordering -- Milestone 101's mandatory
semantic-honesty test); CLI surfacing; artifact immutability; and a genuine
fresh-process check. See `forge/serialization/model.py::InputSchema`/
`_leading_linear_in_features()`, `forge/training/inference.py::
_validate_feature_count()`, and `docs/architecture/persistence.md`'s
**Portable input-contract validation** section.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.cli.main import main as cli_main
from forge.data.transforms import Normalize
from forge.exceptions import DataError
from forge.nn import Flatten, Linear, ReLU, Sequential
from forge.serialization import InputSchema, ModelInfo, inspect_model, save_model
from forge.training import predict_model, predict_tabular_classification_artifact, predict_tensor_artifact

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.dataset import N_FEATURES as DIABETES_N_FEATURES, load_raw  # noqa: E402
from examples.tabular_diabetes.train import main as diabetes_train_main  # noqa: E402
from examples.regression.dataset import N_FEATURES as REGRESSION_N_FEATURES  # noqa: E402
from examples.regression.train import main as regression_train_main  # noqa: E402

_DIABETES_SMALL = ["--batch-size", "16"]
_REGRESSION_SMALL = ["--n-train", "60", "--n-val", "20", "--n-test", "20", "--batch-size", "16"]


# -- InputSchema derivation: unit-level -------------------------------------


def test_input_schema_present_for_regression_task_bare_linear(tmp_path):
    path = tmp_path / "model.forge"
    save_model(Linear(8, 1), str(path), task="regression")
    info = inspect_model(str(path))
    assert info.input_schema == InputSchema(feature_count=8)


def test_input_schema_present_for_tabular_classification_task_sequential(tmp_path):
    path = tmp_path / "model.forge"
    model = Sequential(Linear(5, 16), ReLU(), Linear(16, 3))
    save_model(model, str(path), task="tabular_classification", classes=["a", "b", "c"])
    info = inspect_model(str(path))
    assert info.input_schema == InputSchema(feature_count=5)


def test_input_schema_absent_with_no_task(tmp_path):
    """Legacy/no-task artifacts get no input_schema, even with an
    architecture that would otherwise resolve cleanly -- never invented for
    an artifact that never declared what kind of workflow it is."""
    path = tmp_path / "model.forge"
    save_model(Linear(8, 1), str(path))
    assert inspect_model(str(path)).input_schema is None


def test_input_schema_absent_for_classification_task_even_with_a_linear_head(tmp_path):
    """A Linear classifier head deep inside a CNN must not be mistaken for a
    fixed-width numeric-vector input contract -- gating on task="regression"/
    "tabular_classification" first means a Conv2d-first architecture is
    never even inspected for this."""
    from forge.nn import Conv2d, MaxPool2d

    path = tmp_path / "model.forge"
    model = Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 3 * 3, 2),
    )
    save_model(model, str(path), task="classification", classes=["cat", "dog"])
    assert inspect_model(str(path)).input_schema is None


def test_input_schema_absent_for_segmentation_and_sequence_tasks(tmp_path):
    seg_path = tmp_path / "seg.forge"
    save_model(Linear(4, 4), str(seg_path), task="segmentation")
    assert inspect_model(str(seg_path)).input_schema is None

    from forge.nn import RNNCell

    seq_path = tmp_path / "seq.forge"
    save_model(RNNCell(3, 8), str(seq_path), task="sequence", classes=["a", "b", "c"])
    assert inspect_model(str(seq_path)).input_schema is None


def test_input_schema_absent_when_sequential_first_child_is_not_linear(tmp_path):
    """Only the literal first child of a Sequential is consulted -- a Linear
    that exists but is not immediately first is not searched for, per
    _leading_linear_in_features()'s own documented, deliberately narrow scope."""
    path = tmp_path / "model.forge"
    model = Sequential(ReLU(), Linear(4, 2))
    save_model(model, str(path), task="regression")
    assert inspect_model(str(path)).input_schema is None


def test_input_schema_is_a_frozen_dataclass_with_one_field():
    schema = InputSchema(feature_count=8)
    assert schema.feature_count == 8
    with pytest.raises(Exception):
        schema.feature_count = 9  # frozen


def test_model_info_str_includes_input_line_when_present(tmp_path):
    path = tmp_path / "model.forge"
    save_model(Linear(8, 1), str(path), task="regression")
    text = str(inspect_model(str(path)))
    assert "Input: 8 feature(s)" in text


def test_model_info_str_shows_n_a_when_input_schema_absent(tmp_path):
    path = tmp_path / "model.forge"
    save_model(Linear(8, 1), str(path))
    text = str(inspect_model(str(path)))
    assert "Input: n/a" in text


def test_input_schema_reexported_from_serialization_package():
    assert forge.serialization.InputSchema is InputSchema
    assert isinstance(ModelInfo, type)


# -- validation: predict_tensor_artifact() -----------------------------------


def _bare_regression_model(n_features=8, path=None):
    save_model(Linear(n_features, 1), str(path), task="regression")


def test_predict_tensor_artifact_accepts_correct_feature_count(tmp_path):
    path = tmp_path / "model.forge"
    _bare_regression_model(8, path)
    result = predict_tensor_artifact(str(path), np.zeros((3, 8), dtype=np.float32))
    assert result.shape == (3, 1)


def test_predict_tensor_artifact_accepts_single_unbatched_sample(tmp_path):
    path = tmp_path / "model.forge"
    _bare_regression_model(8, path)
    result = predict_tensor_artifact(str(path), np.zeros((8,), dtype=np.float32))
    assert result.shape == (1,)


def test_predict_tensor_artifact_rejects_too_few_features(tmp_path):
    path = tmp_path / "model.forge"
    _bare_regression_model(8, path)
    with pytest.raises(DataError, match=r"expected 8 input feature\(s\), received 7"):
        predict_tensor_artifact(str(path), np.zeros((1, 7), dtype=np.float32))


def test_predict_tensor_artifact_rejects_too_many_features(tmp_path):
    path = tmp_path / "model.forge"
    _bare_regression_model(8, path)
    with pytest.raises(DataError, match=r"expected 8 input feature\(s\), received 9"):
        predict_tensor_artifact(str(path), np.zeros((1, 9), dtype=np.float32))


def test_predict_tensor_artifact_error_names_the_public_function_not_internals(tmp_path):
    path = tmp_path / "model.forge"
    _bare_regression_model(8, path)
    with pytest.raises(DataError) as exc_info:
        predict_tensor_artifact(str(path), np.zeros((1, 7), dtype=np.float32))
    message = str(exc_info.value)
    assert "predict_tensor_artifact()" in message
    assert "Linear" not in message
    assert "broadcastable" not in message


def test_predict_tensor_artifact_validation_fires_before_preprocessing_runs(tmp_path):
    """Without the Milestone 101 fix, a wrong feature count reached
    Normalize's broadcast arithmetic and failed there instead -- this test
    locks in that the new, clear error fires first."""
    path = tmp_path / "model.forge"
    transform = Normalize(mean=np.zeros(8), std=np.ones(8))
    model = Linear(8, 1)
    save_model(model, str(path), preprocessing=transform, task="regression")
    with pytest.raises(DataError, match="predict_tensor_artifact"):
        predict_tensor_artifact(str(path), np.zeros((1, 7), dtype=np.float32))


def test_predict_tensor_artifact_no_task_falls_back_to_pre_m101_behavior(tmp_path):
    """An artifact with no task= gets no input_schema, so a wrong feature
    count is left to fail exactly as it did before this milestone (a
    ShapeMismatchError from Linear, not the new DataError) -- proving no
    behavior change for artifacts this milestone deliberately does not
    build a contract for."""
    from forge.exceptions import ShapeMismatchError

    path = tmp_path / "model.forge"
    save_model(Linear(8, 1), str(path))  # no task=
    assert inspect_model(str(path)).input_schema is None
    with pytest.raises(ShapeMismatchError):
        predict_tensor_artifact(str(path), np.zeros((1, 7), dtype=np.float32))


# -- validation: predict_tabular_classification_artifact() ------------------


def test_predict_tabular_classification_artifact_rejects_wrong_feature_count(tmp_path):
    path = tmp_path / "model.forge"
    model = Sequential(Linear(5, 8), ReLU(), Linear(8, 2))
    save_model(model, str(path), task="tabular_classification", classes=["no", "yes"])

    with pytest.raises(DataError, match=r"expected 5 input feature\(s\), received 4"):
        predict_tabular_classification_artifact(str(path), np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(DataError, match=r"expected 5 input feature\(s\), received 6"):
        predict_tabular_classification_artifact(str(path), np.zeros((1, 6), dtype=np.float32))


def test_predict_tabular_classification_artifact_accepts_correct_feature_count(tmp_path):
    path = tmp_path / "model.forge"
    model = Sequential(Linear(5, 8), ReLU(), Linear(8, 2))
    save_model(model, str(path), task="tabular_classification", classes=["no", "yes"])

    results = predict_tabular_classification_artifact(str(path), np.zeros((2, 5), dtype=np.float32))
    assert len(results) == 2


def test_predict_model_dispatch_also_validates(tmp_path):
    """predict_model() needs no separate change: it delegates straight to
    predict_tensor_artifact()/predict_tabular_classification_artifact(),
    which already validate."""
    path = tmp_path / "model.forge"
    save_model(Linear(8, 1), str(path), task="regression")
    with pytest.raises(DataError, match="expected 8 input feature"):
        predict_model(str(path), np.zeros((1, 3), dtype=np.float32))


# -- real workload: examples/tabular_diabetes (Phase 8) ----------------------


@pytest.fixture(scope="module")
def diabetes_artifact(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("diabetes_artifacts")
    diabetes_train_main(_DIABETES_SMALL + ["--epochs", "2", "--seed", "0", "--output-dir", str(out_dir)])
    return out_dir / "tabular_diabetes_model.forge"


def test_diabetes_artifact_has_the_expected_input_schema(diabetes_artifact):
    info = inspect_model(str(diabetes_artifact))
    assert info.task == "tabular_classification"
    assert info.input_schema == InputSchema(feature_count=DIABETES_N_FEATURES)


def test_diabetes_artifact_valid_input_succeeds(diabetes_artifact):
    X_raw, _ = load_raw()
    row = X_raw[0].reshape(1, DIABETES_N_FEATURES).astype(np.float32)
    result = predict_model(str(diabetes_artifact), row)
    assert len(result) == 1


def test_diabetes_artifact_rejects_too_few_features(diabetes_artifact):
    X_raw, _ = load_raw()
    row = X_raw[0, :-1].reshape(1, DIABETES_N_FEATURES - 1).astype(np.float32)
    with pytest.raises(DataError, match=rf"expected {DIABETES_N_FEATURES} input feature\(s\), received {DIABETES_N_FEATURES - 1}"):
        predict_model(str(diabetes_artifact), row)


def test_diabetes_artifact_rejects_too_many_features(diabetes_artifact):
    X_raw, _ = load_raw()
    row = np.concatenate([X_raw[0], [99.0]]).reshape(1, DIABETES_N_FEATURES + 1).astype(np.float32)
    with pytest.raises(DataError, match=rf"expected {DIABETES_N_FEATURES} input feature\(s\), received {DIABETES_N_FEATURES + 1}"):
        predict_model(str(diabetes_artifact), row)


def test_diabetes_artifact_reordered_features_is_not_detected(diabetes_artifact):
    """Milestone 101's mandatory semantic-honesty test (Phase 25): a real
    reordering of two real feature columns (Pregnancies/Glucose, indices 0
    and 1) has the exact same shape and passes silently -- this is the
    documented, tested limit of structural validation, not a bug. If this
    test starts failing because predict_model() started raising, that would
    mean a *new* capability was added that must be documented; if it starts
    failing because the two predictions became identical, that would mean
    the reordering stopped being a real semantic difference for this
    artifact, and the test data should be revisited."""
    X_raw, _ = load_raw()
    correct = X_raw[0].reshape(1, DIABETES_N_FEATURES).astype(np.float32)
    reordered = correct.copy()
    reordered[:, [0, 1]] = reordered[:, [1, 0]]

    # No error at all -- both structurally valid.
    correct_result = predict_model(str(diabetes_artifact), correct)[0]
    reordered_result = predict_model(str(diabetes_artifact), reordered)[0]

    # The prediction is computed from meaningfully different input (the
    # confidence differs even where the predicted label happens to agree),
    # proving this genuinely reached the model with swapped semantics rather
    # than being coincidentally a no-op reorder.
    assert correct_result.confidence != pytest.approx(reordered_result.confidence, abs=1e-9)


def test_diabetes_artifact_immutable_across_valid_and_invalid_predictions(diabetes_artifact):
    import hashlib

    def _hash():
        return hashlib.sha256(diabetes_artifact.read_bytes()).hexdigest()

    before = _hash()
    X_raw, _ = load_raw()
    valid = X_raw[0].reshape(1, DIABETES_N_FEATURES).astype(np.float32)
    predict_model(str(diabetes_artifact), valid)
    try:
        predict_model(str(diabetes_artifact), valid[:, :-1])
    except DataError:
        pass
    inspect_model(str(diabetes_artifact))
    after = _hash()
    assert before == after


def test_diabetes_artifact_input_schema_deterministic_across_identical_training_runs(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    diabetes_train_main(_DIABETES_SMALL + ["--epochs", "1", "--seed", "7", "--output-dir", str(out_a)])
    diabetes_train_main(_DIABETES_SMALL + ["--epochs", "1", "--seed", "7", "--output-dir", str(out_b)])

    info_a = inspect_model(str(out_a / "tabular_diabetes_model.forge"))
    info_b = inspect_model(str(out_b / "tabular_diabetes_model.forge"))
    assert info_a.input_schema == info_b.input_schema == InputSchema(feature_count=DIABETES_N_FEATURES)


def test_diabetes_artifact_cli_predict_rejects_wrong_feature_count(diabetes_artifact, tmp_path, capsys):
    X_raw, _ = load_raw()
    bad_row = X_raw[0, :-1].tolist()
    input_path = tmp_path / "bad_input.json"
    input_path.write_text(json.dumps(bad_row))

    exit_code = cli_main(["model", "predict", str(diabetes_artifact), str(input_path)])
    assert exit_code != 0


# -- real workload: examples/regression (Phase 9) ----------------------------


@pytest.fixture(scope="module")
def regression_artifact(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("regression_artifacts")
    regression_train_main(_REGRESSION_SMALL + ["--epochs", "1", "--seed", "0", "--output-dir", str(out_dir)])
    return out_dir / "regression_model.forge"


def test_regression_artifact_has_the_expected_input_schema(regression_artifact):
    info = inspect_model(str(regression_artifact))
    assert info.task == "regression"
    assert info.input_schema == InputSchema(feature_count=REGRESSION_N_FEATURES)


def test_regression_artifact_valid_input_succeeds(regression_artifact):
    result = predict_model(str(regression_artifact), np.zeros((1, REGRESSION_N_FEATURES), dtype=np.float32))
    assert result.shape == (1, 1)


def test_regression_artifact_rejects_wrong_feature_count(regression_artifact):
    with pytest.raises(DataError, match=rf"expected {REGRESSION_N_FEATURES} input feature\(s\)"):
        predict_model(str(regression_artifact), np.zeros((1, REGRESSION_N_FEATURES - 1), dtype=np.float32))


# -- CLI ----------------------------------------------------------------------


def test_cli_inspect_prints_input_line_when_present(diabetes_artifact, capsys):
    exit_code = cli_main(["model", "inspect", str(diabetes_artifact)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"Input: {DIABETES_N_FEATURES} feature(s)" in out


def test_cli_inspect_json_includes_input_feature_count(diabetes_artifact, capsys):
    exit_code = cli_main(["model", "inspect", str(diabetes_artifact), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["input_feature_count"] == DIABETES_N_FEATURES


def test_cli_inspect_omits_input_line_when_absent(tmp_path, capsys):
    path = tmp_path / "model.forge"
    save_model(Linear(4, 2), str(path))
    exit_code = cli_main(["model", "inspect", str(path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Input:" not in out

    exit_code = cli_main(["model", "inspect", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["input_feature_count"] is None


# -- backward compatibility: image/sequence/segmentation real artifacts -----


def test_real_sequence_artifact_has_no_input_schema(tmp_path):
    from forge.nn import RNNCell

    path = tmp_path / "seq.forge"
    save_model(RNNCell(4, 8), str(path), task="sequence", classes=list("abcd"))
    info = inspect_model(str(path))
    assert info.task == "sequence"
    assert info.input_schema is None


# -- fresh process ------------------------------------------------------------


def test_input_contract_validation_works_from_a_genuinely_separate_process(diabetes_artifact):
    X_raw, _ = load_raw()
    bad_row = X_raw[0, :-1].tolist()

    script = (
        "import numpy as np\n"
        "import forge\n"
        f"row = np.array({bad_row!r}, dtype=np.float32).reshape(1, {DIABETES_N_FEATURES - 1})\n"
        "try:\n"
        f"    forge.predict_model({str(diabetes_artifact)!r}, row)\n"
        "    print('NO_ERROR')\n"
        "except forge.DataError as e:\n"
        "    print(f'DataError: {e}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert "DataError" in result.stdout
    assert f"expected {DIABETES_N_FEATURES} input feature(s), received {DIABETES_N_FEATURES - 1}" in result.stdout
