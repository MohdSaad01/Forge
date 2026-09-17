"""Milestone 91 tests: the end-to-end tabular classification artifact workflow --

```text
examples.tabular_classification.dataset/model -> forge.train_and_save()
    -> verified .forge artifact -> fresh process
    -> forge.predict_tabular_classification_artifact() / forge.predict_model()
    -> ClassificationPrediction
```

This is the literal M91 acceptance test: a genuinely separate OS process,
given only the `.forge` file this test's own process trained and saved,
must produce a classification prediction that agrees with this process's
own prediction on the same raw input -- proving the artifact (model +
persisted `Normalize` preprocessing + `classes=` + `task=
"tabular_classification"`) is actually portable. Mirrors
`tests/test_regression_artifact_workflow.py`'s structure for the numeric-
input artifact shape, and `tests/test_sequence_artifact_prediction.py`'s
unit-level coverage of the artifact-shape-specific function itself.

Also covers the real, reproduced Milestone 91 blocker directly: before
`task="tabular_classification"`/`predict_tabular_classification_artifact()`
existed, this exact artifact shape (a `Linear`-terminated classifier over
numeric tabular input, saved with `classes=`) raised `forge.DataError` from
deep inside `predict_artifact()` (`... requires image to be a file path
..., got ndarray`) the moment a developer tried to predict on it -- see
`docs/development/m91-tabular-classification-artifact-inference.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.exceptions import DataError, PersistenceError
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import save_model
from forge.training import ClassificationPrediction, predict_model, predict_tabular_classification_artifact

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_classification.dataset import CLASS_NAMES, N_FEATURES, generate_raw  # noqa: E402
from examples.tabular_classification.train import main as train_main  # noqa: E402

_SMALL = [
    "--samples-per-class-train", "40", "--samples-per-class-val", "10", "--samples-per-class-test", "10",
    "--batch-size", "16",
]


# -- unit-level: predict_tabular_classification_artifact() --------------------


def _tiny_model(num_classes=3, n_features=4):
    return Sequential(Linear(n_features, 6), ReLU(), Linear(6, num_classes))


def test_predict_tabular_classification_artifact_returns_one_result_per_row(tmp_path):
    forge.random.seed(0)
    model = _tiny_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), classes=["a", "b", "c"], task="tabular_classification")

    batch = np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)
    results = predict_tabular_classification_artifact(str(path), batch)
    assert isinstance(results, list) and len(results) == 5
    assert all(isinstance(r, ClassificationPrediction) for r in results)
    assert all(r.label in ("a", "b", "c") for r in results)


def test_predict_tabular_classification_artifact_without_classes_returns_raw_indices(tmp_path):
    forge.random.seed(0)
    model = _tiny_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="tabular_classification")

    batch = np.random.default_rng(2).standard_normal((3, 4)).astype(np.float32)
    results = predict_tabular_classification_artifact(str(path), batch)
    assert isinstance(results, list) and len(results) == 3
    assert all(isinstance(r, int) for r in results)


def test_predict_tabular_classification_artifact_preprocessing_is_optional(tmp_path):
    forge.random.seed(0)
    model = _tiny_model()
    no_preprocessing_path = tmp_path / "no_preprocessing.forge"
    save_model(model, str(no_preprocessing_path), classes=["a", "b", "c"], task="tabular_classification")

    batch = np.random.default_rng(3).standard_normal((2, 4)).astype(np.float32)
    # No preprocessing= saved -- input_data must be passed to the model as-is,
    # not silently rejected (unlike predict_artifact()'s mandatory preprocessing).
    results = predict_tabular_classification_artifact(str(no_preprocessing_path), batch)
    assert len(results) == 2


def test_predict_tabular_classification_artifact_applies_saved_preprocessing(tmp_path):
    from forge.data import Normalize

    forge.random.seed(0)
    model = _tiny_model()
    path = tmp_path / "model.forge"
    transform = Normalize(mean=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), std=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
    save_model(model, str(path), preprocessing=transform, classes=["a", "b", "c"], task="tabular_classification")

    raw = np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    via_api = predict_tabular_classification_artifact(str(path), raw)

    from forge.serialization import load_model

    reloaded = load_model(str(path))
    standardized = (raw - transform.mean) / np.asarray(transform.std)
    with forge.no_grad():
        manual_output = reloaded(forge.Tensor(standardized.astype(np.float32)))
    manual_index = int(np.argmax(manual_output.numpy(), axis=1)[0])

    assert via_api[0].index == manual_index


def test_predict_tabular_classification_artifact_rejects_invalid_input_type(tmp_path):
    forge.random.seed(0)
    model = _tiny_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), classes=["a", "b", "c"], task="tabular_classification")

    with pytest.raises(DataError):
        predict_tabular_classification_artifact(str(path), "not/an/array")


def test_predict_tabular_classification_artifact_missing_file_raises_persistence_error(tmp_path):
    with pytest.raises(PersistenceError):
        predict_tabular_classification_artifact(str(tmp_path / "nope.forge"), np.zeros((1, 4), dtype=np.float32))


def test_predict_tabular_classification_artifact_rejects_unbatched_1d_input_clearly(tmp_path):
    """Milestone 105: a genuinely reproduced external-developer blocker --
    an unbatched 1-D single row (the same shape `predict_tensor_artifact()`'s
    regression path legitimately accepts, see `_validate_feature_count()`'s
    own docstring) used to slip past feature-count validation here and fail
    deep inside `interpret_classification()` with an internals-revealing
    `TrainerError` naming a function this caller never called directly.
    Must now raise a clear `forge.DataError` before the model ever runs,
    naming the exact fix (wrap the row in an extra list)."""
    forge.random.seed(0)
    model = _tiny_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), classes=["a", "b", "c"], task="tabular_classification")

    with pytest.raises(DataError, match="requires a batched input"):
        predict_tabular_classification_artifact(str(path), [0.1, 0.2, 0.3, 0.4])

    # The equivalent artifact with no saved classes (the raw np.argmax(...,
    # axis=1) fallback) must reject the same shape just as clearly, not with
    # a raw numpy AxisError.
    no_classes_path = tmp_path / "model_no_classes.forge"
    save_model(model, str(no_classes_path), task="tabular_classification")
    with pytest.raises(DataError, match="requires a batched input"):
        predict_tabular_classification_artifact(str(no_classes_path), [0.1, 0.2, 0.3, 0.4])

    # A correctly-batched single-row input (the documented contract) still works.
    results = predict_tabular_classification_artifact(str(path), [[0.1, 0.2, 0.3, 0.4]])
    assert len(results) == 1


# -- end-to-end: real example, fresh process -----------------------------------


def test_train_and_save_writes_a_tabular_classification_artifact_with_full_metadata(tmp_path):
    """The fresh training path must save the fitted `Normalize` transform,
    the class vocabulary, and `task="tabular_classification"` together --
    without all three, a fresh process holding only the `.forge` file could
    not standardize a raw feature vector, interpret the output, or route
    through `predict_model()` at all."""
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "1", "--output-dir", str(out_dir)])

    model_path = out_dir / "tabular_classification_model.forge"
    assert model_path.is_file()

    from forge.serialization import inspect_model, load_classes, load_preprocessing

    assert load_preprocessing(str(model_path)) is not None
    assert load_classes(str(model_path)) == CLASS_NAMES
    assert inspect_model(str(model_path)).task == "tabular_classification"


def test_real_example_achieves_well_above_trivial_baseline_accuracy(tmp_path, capsys):
    """Confirms genuine learning, not merely "the code runs" -- the real
    Section-6 baseline-workload requirement."""
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "15", "--seed", "2", "--output-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert "Trivial baseline accuracy was 25.0%" in out

    import re
    match = re.search(r"Final test evaluation: loss=[\d.]+, accuracy=([\d.]+)%", out)
    assert match is not None, out
    test_accuracy = float(match.group(1))
    assert test_accuracy > 50.0, f"expected well above the 25% trivial baseline, got {test_accuracy}%"


def test_predict_tabular_classification_artifact_agrees_with_a_genuinely_separate_process(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "3", "--seed", "42", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_classification_model.forge"

    # A brand-new raw feature vector, never part of train/val/test (seed
    # deliberately outside the training run's own seed-derived data stream).
    raw_x, _ = generate_raw(1, seed=99999)
    raw_x_literal = raw_x.tolist()

    expected = predict_tabular_classification_artifact(str(model_path), raw_x)[0]

    script = (
        "import numpy as np\n"
        "import forge\n"
        f"raw_x = np.array({raw_x_literal!r}, dtype=np.float32)\n"
        f"result = forge.predict_tabular_classification_artifact({str(model_path)!r}, raw_x)[0]\n"
        "print(f'label={result.label} index={result.index} confidence={result.confidence:.6f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert f"label={expected.label} index={expected.index}" in result.stdout


def test_predict_model_agrees_with_predict_tabular_classification_artifact_after_real_training(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "5", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_classification_model.forge"

    # generate_raw(samples_per_class, seed) returns N_CLASSES * samples_per_class
    # rows (unlike examples/regression's generate_raw(n_samples, seed), which
    # takes a total count directly) -- samples_per_class=1 here for a small,
    # exact-length batch.
    raw_x, _ = generate_raw(1, seed=777)
    via_unified = predict_model(str(model_path), raw_x)
    via_direct = predict_tabular_classification_artifact(str(model_path), raw_x)

    assert len(via_unified) == len(via_direct) == len(raw_x)
    for u, d in zip(via_unified, via_direct):
        assert u.label == d.label
        assert u.index == d.index
