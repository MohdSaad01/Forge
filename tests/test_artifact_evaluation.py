"""Milestone 113 tests: `ArtifactPredictor.evaluate()` -- first-class evaluation of a saved artifact.

```text
saved .forge artifact -> forge.load_predictor() -> predictor.evaluate(X, y)
    -> raw X -> persisted preprocessing -> model -> task-specific metrics
```

The contract under test (not the implementation): metrics are correct against
hand-computed values; the artifact's persisted preprocessing is what runs
(a fixture whose preprocessing *changes the answer*, not an identity one); the
Pima diabetes artifact agrees with the independent oracle
(`examples/tabular_diabetes/evaluate.py`) on the real holdout; invalid input
raises a `DataError` (never `ShapeMismatchError`/`KeyError`/`IndexError`);
evaluation never mutates the artifact or the model; a fresh process agrees.
CUDA parity lives in `tests/test_artifact_evaluation_cuda.py`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.data.transforms import Compose, Normalize, Resize
from forge.exceptions import DataError, PersistenceError, ShapeMismatchError
from forge.nn import Conv1d, Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU, RNNCell, Sequential
from forge.nn.parameter import Parameter
from forge.serialization import register_module, save_model
from forge.training import (
    ClassificationEvaluationResult,
    RegressionEvaluationResult,
    load_predictor,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.evaluate import evaluate_artifact, load_labeled_csv  # noqa: E402
from examples.tabular_diabetes.train import main as diabetes_train_main  # noqa: E402

_PIMA_HOLDOUT = _REPO_ROOT / "examples" / "tabular_diabetes" / "data" / "diabetes_holdout_eval.csv"

# Deliberately not alphabetical, and with a class ("unused") no model output can ever select.
_CLASSES = ["zebra", "apple", "mango", "unused"]
_MEAN = np.array([10.0, 0.0], dtype=np.float32)


def _set_linear(layer: Linear, weight, bias) -> None:
    layer.weight = Parameter(np.asarray(weight, dtype=np.float32))
    layer.bias = Parameter(np.asarray(bias, dtype=np.float32))


def _scoring_model() -> Sequential:
    """Scores for z = preprocessed row: zebra=z0, apple=z1, mango=0.5, unused=-100. (Linear computes z @ W + b.)"""
    layer = Linear(2, 4)
    _set_linear(layer, [[1, 0, 0, 0], [0, 1, 0, 0]], [0, 0, 0.5, -100.0])
    return Sequential(layer)


# Raw rows; the artifact's Normalize subtracts 10 from column 0. Every row has x0 >= x1, so a model
# fed the RAW row always answers "zebra" -- only the persisted preprocessing yields the answers below.
_X = np.array(
    [[12, 0], [13, 1], [10, 3], [10, 2], [10, 0], [10.2, 0.1], [11, 0], [10, 5]], dtype=np.float32,
)
_PREDICTED = ["zebra", "zebra", "apple", "apple", "mango", "mango", "zebra", "apple"]
_TRUE = ["zebra", "zebra", "apple", "zebra", "mango", "apple", "zebra", "apple"]
# rows = true, columns = predicted, both in _CLASSES order.
_CONFUSION = np.array([[3, 1, 0, 0], [0, 2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])


def _save_scoring_artifact(tmp_path, *, preprocessing=True, classes=_CLASSES, name="scoring.forge") -> str:
    path = tmp_path / name
    save_model(
        _scoring_model(), str(path),
        preprocessing=Normalize(mean=_MEAN, std=np.ones(2, dtype=np.float32)) if preprocessing else None,
        classes=classes, task="tabular_classification",
    )
    return str(path)


def _reference_cross_entropy(x: np.ndarray, label_indices: np.ndarray) -> float:
    z = (x - _MEAN).astype(np.float64)
    logits = np.stack([z[:, 0], z[:, 1], np.full(len(z), 0.5), np.full(len(z), -100.0)], axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(z)), label_indices].mean())


# -- classification metrics ---------------------------------------------------


def test_classification_metrics_match_hand_computation(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))

    result = predictor.evaluate(_X, _TRUE)

    assert isinstance(result, ClassificationEvaluationResult)
    assert result.task == "tabular_classification"
    assert result.samples == 8
    assert result.accuracy == pytest.approx(6 / 8)
    assert result.baseline_accuracy == pytest.approx(4 / 8)  # zebra is the majority of the evaluated rows
    np.testing.assert_array_equal(result.confusion_matrix, _CONFUSION)
    assert result.support == (4, 3, 1, 0)
    assert result.precision == pytest.approx((3 / 3, 2 / 3, 1 / 2, 0.0))
    assert result.recall == pytest.approx((3 / 4, 2 / 3, 1 / 1, 0.0))
    assert result.metrics == {"accuracy": result.accuracy}
    indices = np.array([_CLASSES.index(t) for t in _TRUE])
    assert result.loss == pytest.approx(_reference_cross_entropy(_X, indices), rel=1e-5)


def test_class_order_is_the_artifacts_not_the_data_or_alphabetical(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    result = predictor.evaluate(_X, _TRUE)
    assert result.classes == tuple(_CLASSES)
    assert list(result.classes) != sorted(result.classes)
    # Row/column 0 is "zebra" (the artifact's first class), though "apple" sorts first and appears first in y.
    assert result.confusion_matrix[0, 0] == 3


def test_integer_and_name_labels_give_identical_results(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    by_name = predictor.evaluate(_X, _TRUE)
    by_index = predictor.evaluate(_X, [_CLASSES.index(t) for t in _TRUE])
    by_array = predictor.evaluate(_X, np.array([_CLASSES.index(t) for t in _TRUE], dtype=np.int64))
    for other in (by_index, by_array):
        np.testing.assert_array_equal(other.confusion_matrix, by_name.confusion_matrix)
        assert other.accuracy == by_name.accuracy
        assert other.loss == by_name.loss


def test_zero_division_is_zero_and_raises_no_warning(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a NumPy divide/invalid RuntimeWarning would fail the test
        result = predictor.evaluate(_X, _TRUE)
    # "unused" is never predicted (precision) and never true (recall): both defined as 0.0, not NaN.
    assert result.precision[3] == 0.0 and result.recall[3] == 0.0
    assert all(np.isfinite(result.precision)) and all(np.isfinite(result.recall))


def test_a_class_predicted_but_never_true_has_zero_precision_not_nan(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    # Every row is truly "zebra": apple/mango are predicted for some rows but have no support (recall 0/0 -> 0.0).
    result = predictor.evaluate(_X, ["zebra"] * 8)
    assert result.support == (8, 0, 0, 0)
    assert result.recall == pytest.approx((3 / 8, 0.0, 0.0, 0.0))
    assert result.precision == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert result.baseline_accuracy == 1.0


def test_persisted_preprocessing_is_what_runs(tmp_path):
    """The M97 trap: the same model without the artifact's Normalize answers "zebra" for every row."""
    with_prep = load_predictor(_save_scoring_artifact(tmp_path, preprocessing=True, name="with.forge"))
    without_prep = load_predictor(_save_scoring_artifact(tmp_path, preprocessing=False, name="without.forge"))

    correct = with_prep.evaluate(_X, _TRUE)
    naive = without_prep.evaluate(_X, _TRUE)

    assert correct.accuracy == pytest.approx(6 / 8)
    assert naive.accuracy == pytest.approx(4 / 8)          # only the four true zebra rows
    assert naive.confusion_matrix[:, 0].sum() == 8          # everything predicted zebra
    assert correct.accuracy != naive.accuracy


def test_evaluate_and_predict_agree_on_every_row(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    predicted = [p.label for p in predictor.predict(_X)]
    assert predicted == _PREDICTED
    result = predictor.evaluate(_X, predicted)  # scoring predict()'s own labels must be perfect
    assert result.accuracy == 1.0
    np.testing.assert_array_equal(np.diag(result.confusion_matrix), result.support)


def test_result_does_not_depend_on_batch_size(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    reference = predictor.evaluate(_X, _TRUE, batch_size=8)
    for batch_size in (1, 3, 5, 1000):
        result = predictor.evaluate(_X, _TRUE, batch_size=batch_size)
        np.testing.assert_array_equal(result.confusion_matrix, reference.confusion_matrix)
        assert result.accuracy == reference.accuracy
        assert result.loss == pytest.approx(reference.loss, rel=1e-6)


def test_result_is_frozen_and_confusion_matrix_read_only(tmp_path):
    result = load_predictor(_save_scoring_artifact(tmp_path)).evaluate(_X, _TRUE)
    with pytest.raises(Exception):
        result.accuracy = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        result.confusion_matrix[0, 0] = 99


def test_x_may_be_a_nested_list_or_tensor(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    reference = predictor.evaluate(_X, _TRUE)
    for x in (_X.tolist(), forge.Tensor(_X)):
        result = predictor.evaluate(x, _TRUE)
        np.testing.assert_array_equal(result.confusion_matrix, reference.confusion_matrix)


# -- regression metrics -------------------------------------------------------

_REG_X = np.array([[11, 1], [12, 2], [10, 4], [14, 0]], dtype=np.float32)
_REG_Y = np.array([2.0, 4.5, 4.0, 5.0], dtype=np.float64)


def _save_regression_artifact(tmp_path, *, outputs=1, name="regression.forge") -> str:
    layer = Linear(2, outputs)
    # z = x - [10, 0]; output 0 = z0 + 2*z1 + 0.5; the optional output 1 = z0.  (Linear computes z @ W + b.)
    weight = [[1, 1], [2, 0]] if outputs == 2 else [[1], [2]]
    _set_linear(layer, weight, [0.5, 0.0][:outputs])
    path = tmp_path / name
    save_model(
        Sequential(layer), str(path), preprocessing=Normalize(mean=_MEAN, std=np.ones(2, dtype=np.float32)),
        task="regression",
    )
    return str(path)


def test_regression_metrics_match_hand_computation(tmp_path):
    predictor = load_predictor(_save_regression_artifact(tmp_path))
    z = _REG_X.astype(np.float64) - _MEAN
    predictions = z[:, 0] * 1 + z[:, 1] * 2 + 0.5   # [3.5, 6.5, 1.5, 4.5]

    result = predictor.evaluate(_REG_X, _REG_Y)

    assert isinstance(result, RegressionEvaluationResult)
    assert result.task == "regression"
    assert result.samples == 4
    assert result.mse == pytest.approx(np.mean((predictions - _REG_Y) ** 2))
    assert result.mae == pytest.approx(np.mean(np.abs(predictions - _REG_Y)))
    assert result.loss == pytest.approx(result.mse, rel=1e-6)
    assert result.baseline_mse == pytest.approx(np.var(_REG_Y))
    assert result.metrics == {"mse": result.mse, "mae": result.mae}


def test_regression_accepts_column_vector_targets(tmp_path):
    predictor = load_predictor(_save_regression_artifact(tmp_path))
    flat = predictor.evaluate(_REG_X, _REG_Y)
    column = predictor.evaluate(_REG_X, _REG_Y[:, None])
    assert (column.mse, column.mae, column.samples) == (flat.mse, flat.mae, flat.samples)


def test_regression_multi_output_averages_over_samples_and_outputs(tmp_path):
    predictor = load_predictor(_save_regression_artifact(tmp_path, outputs=2, name="multi.forge"))
    z = _REG_X.astype(np.float64) - _MEAN
    predictions = np.stack([z[:, 0] * 1 + z[:, 1] * 2 + 0.5, z[:, 0] * 1 + z[:, 1] * 0], axis=1)
    targets = np.stack([_REG_Y, _REG_Y * 2], axis=1)

    result = predictor.evaluate(_REG_X, targets)

    assert result.mse == pytest.approx(np.mean((predictions - targets) ** 2))
    assert result.mae == pytest.approx(np.mean(np.abs(predictions - targets)))
    assert result.baseline_mse == pytest.approx(np.mean((targets - targets.mean(axis=0)) ** 2))


def test_regression_applies_persisted_preprocessing(tmp_path):
    predictor = load_predictor(_save_regression_artifact(tmp_path))
    raw_predictions = _REG_X[:, 0] * 1 + _REG_X[:, 1] * 2 + 0.5   # what a caller who skipped Normalize would compute
    raw_mse = float(np.mean((raw_predictions - _REG_Y) ** 2))
    result = predictor.evaluate(_REG_X, _REG_Y)
    assert result.mse != pytest.approx(raw_mse)
    predicted = predictor.predict(_REG_X).numpy().ravel()
    assert result.mse == pytest.approx(np.mean((predicted - _REG_Y) ** 2))


# -- Pima diabetes: real dataset, parity with the independent oracle ----------


@pytest.fixture(scope="module")
def pima_model(tmp_path_factory) -> Path:
    """The README's reference Pima artifact, trained here rather than read from `examples/`.

    `examples/tabular_diabetes/artifacts*/` is gitignored (generated, machine-local), so a fresh
    checkout has no such file. The reference recipe is deterministic and bit-identical across
    platforms, so every known-number assertion below holds on a freshly trained copy.
    """
    out_dir = tmp_path_factory.mktemp("pima_artifact")
    diabetes_train_main(["--epochs", "60", "--seed", "0", "--device", "cpu", "--output-dir", str(out_dir)])
    return out_dir / "tabular_diabetes_model.forge"


def _oracle_metrics(model_path: str, csv_path: str) -> dict:
    """The oracle's own composition (`load_model` + `load_preprocessing` + `forge.predict`), plus plain NumPy."""
    X, y = load_labeled_csv(csv_path)
    model = forge.load_model(model_path)
    preprocessing = forge.load_preprocessing(model_path)
    logits = forge.predict(model, preprocessing(forge.Tensor(X))).numpy().astype(np.float64)
    predicted = logits.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y, predicted):
        confusion[t, p] += 1
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return {
        "confusion": confusion,
        "loss": float(-log_probs[np.arange(len(y)), y].mean()),
        "precision": confusion.diagonal() / confusion.sum(axis=0),
        "recall": confusion.diagonal() / confusion.sum(axis=1),
    }


def test_pima_evaluate_matches_the_existing_oracle(pima_model):
    X, y = load_labeled_csv(str(_PIMA_HOLDOUT))
    summary = evaluate_artifact(str(pima_model), str(_PIMA_HOLDOUT))
    oracle = _oracle_metrics(str(pima_model), str(_PIMA_HOLDOUT))

    result = load_predictor(str(pima_model)).evaluate(X, y)

    assert result.samples == summary.samples == 154
    assert result.accuracy == pytest.approx(summary.accuracy, abs=1e-12)
    assert result.baseline_accuracy == pytest.approx(summary.baseline_accuracy, abs=1e-12)
    np.testing.assert_array_equal(result.confusion_matrix, oracle["confusion"])
    assert result.loss == pytest.approx(oracle["loss"], rel=1e-5)
    assert result.precision == pytest.approx(tuple(oracle["precision"]), abs=1e-12)
    assert result.recall == pytest.approx(tuple(oracle["recall"]), abs=1e-12)
    assert result.classes == ("no_diabetes", "diabetes")
    # The known holdout numbers (M97/M112): 72.1% vs the holdout's own 62.3% majority baseline.
    assert result.accuracy == pytest.approx(0.7208, abs=5e-4)
    assert result.baseline_accuracy == pytest.approx(0.6234, abs=5e-4)


def test_pima_evaluate_applies_the_persisted_replacevalue_and_normalize(pima_model):
    """Raw sentinel-zero rows through the bare model reproduce the M97 wrong number; evaluate() does not."""
    X, y = load_labeled_csv(str(_PIMA_HOLDOUT))
    bare = forge.load_model(str(pima_model))
    raw_logits = forge.predict(bare, forge.Tensor(X)).numpy()
    raw_accuracy = float((raw_logits.argmax(axis=1) == y).mean())

    result = load_predictor(str(pima_model)).evaluate(X, y)

    assert raw_accuracy < 0.5 < result.accuracy
    assert repr(load_predictor(str(pima_model))._preprocessing) == "Compose([ReplaceValue, Normalize])"


def test_pima_evaluate_with_string_labels_and_wrong_feature_count(pima_model):
    X, y = load_labeled_csv(str(_PIMA_HOLDOUT))
    predictor = load_predictor(str(pima_model))
    by_name = predictor.evaluate(X, [predictor.classes[i] for i in y])
    assert by_name.accuracy == pytest.approx(predictor.evaluate(X, y).accuracy)
    with pytest.raises(DataError, match="expected 8 input feature"):
        predictor.evaluate(X[:, :7], y)


_FRESH_SCRIPT = """
import csv, json, sys
import numpy as np
import forge

# Deliberately nothing from examples/: just forge, the artifact, and the CSV.
with open(sys.argv[2], newline="") as f:
    reader = csv.reader(f); next(reader)
    data = np.array(list(reader), dtype=np.float32)
X, y = data[:, :-1], data[:, -1].astype(np.int64)
r = forge.load_predictor(sys.argv[1]).evaluate(X, y)
print(json.dumps({
    "samples": r.samples, "accuracy": r.accuracy, "baseline": r.baseline_accuracy, "loss": r.loss,
    "confusion": r.confusion_matrix.tolist(), "precision": list(r.precision), "recall": list(r.recall),
    "support": list(r.support), "classes": list(r.classes),
}))
"""


def test_fresh_processes_agree_with_each_other_and_with_this_process(pima_model):
    args = f' "{pima_model}" "{_PIMA_HOLDOUT}"'
    script = "import sys; sys.argv = ['x'] + sys.argv[1:]\n" + _FRESH_SCRIPT

    def run_once() -> dict:
        completed = subprocess.run(
            [sys.executable, "-c", _FRESH_SCRIPT, str(pima_model), str(_PIMA_HOLDOUT)],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=120,
        )
        assert completed.returncode == 0, f"subprocess failed:\n{completed.stderr}"
        return json.loads(completed.stdout.strip().splitlines()[-1])

    process_a, process_b = run_once(), run_once()
    assert process_a == process_b

    X, y = load_labeled_csv(str(_PIMA_HOLDOUT))
    here = load_predictor(str(pima_model)).evaluate(X, y)
    assert process_a["samples"] == here.samples == 154
    assert process_a["accuracy"] == here.accuracy
    assert process_a["loss"] == here.loss
    assert process_a["confusion"] == here.confusion_matrix.tolist()
    assert process_a["precision"] == list(here.precision)
    assert process_a["recall"] == list(here.recall)
    assert process_a["classes"] == list(here.classes)


# -- image-classification artifacts (ImageFolder layout) ----------------------

_IMAGE_SIZE = (8, 8)


def _make_image(path: Path, fill: int) -> None:
    Image.fromarray(np.full((8, 8, 3), fill, dtype=np.uint8), mode="RGB").save(path)


def _save_image_artifact(tmp_path, *, classes=("dog", "cat"), name="image.forge") -> str:
    forge.random.seed(0)
    model = Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2), Flatten(),
        Linear(4 * 4 * 4, len(classes)),
    )
    path = tmp_path / name
    save_model(
        model, str(path), preprocessing=Compose([Resize(_IMAGE_SIZE), Normalize(mean=0.0, std=255.0)]),
        classes=list(classes), task="classification",
    )
    return str(path)


def _image_root(tmp_path, per_class=None) -> Path:
    per_class = per_class or {"cat": [30, 60, 90], "dog": [150, 200, 250, 255]}
    root = tmp_path / "images"
    for class_name, fills in per_class.items():
        (root / class_name).mkdir(parents=True)
        for i, fill in enumerate(fills):
            _make_image(root / class_name / f"{i}.png", fill)
    return root


def test_image_folder_labels_match_the_artifacts_classes_by_name_not_order(tmp_path):
    predictor = load_predictor(_save_image_artifact(tmp_path))       # artifact order: dog, cat
    root = _image_root(tmp_path)                                      # ImageFolder order: cat, dog

    result = predictor.evaluate(str(root))

    expected = np.zeros((2, 2), dtype=np.int64)
    for class_dir in sorted(root.iterdir()):
        for image in sorted(class_dir.iterdir()):
            expected[predictor.classes.index(class_dir.name), predictor.predict(str(image)).index] += 1
    assert result.classes == ("dog", "cat")
    np.testing.assert_array_equal(result.confusion_matrix, expected)
    assert result.support == (4, 3)               # dog first, per the artifact -- not cat first
    assert result.samples == 7
    assert result.accuracy == pytest.approx(np.trace(expected) / 7)
    assert result.baseline_accuracy == pytest.approx(4 / 7)
    assert np.isfinite(result.loss)


def test_image_folder_may_omit_a_class_the_artifact_has(tmp_path):
    predictor = load_predictor(_save_image_artifact(tmp_path, classes=("dog", "cat", "bird")))
    result = predictor.evaluate(str(_image_root(tmp_path)))
    assert result.classes == ("dog", "cat", "bird")
    assert result.support == (4, 3, 0)
    assert result.recall[2] == 0.0


def test_image_folder_unknown_class_folder_names_both_vocabularies(tmp_path):
    predictor = load_predictor(_save_image_artifact(tmp_path))
    root = _image_root(tmp_path, {"cat": [30], "dog": [200], "wolf": [128]})
    with pytest.raises(DataError, match=r"wolf.*\['dog', 'cat'\]"):
        predictor.evaluate(str(root))


def test_image_folder_rejects_y_a_non_path_and_an_unreadable_image(tmp_path):
    predictor = load_predictor(_save_image_artifact(tmp_path))
    root = _image_root(tmp_path)
    with pytest.raises(DataError, match="does not take y"):
        predictor.evaluate(str(root), ["cat"])
    with pytest.raises(DataError, match="directory path"):
        predictor.evaluate(np.zeros((2, 3, 8, 8), dtype=np.float32))
    with pytest.raises(DataError, match="does not exist or is not a directory"):
        predictor.evaluate(str(tmp_path / "missing"))
    (root / "cat" / "broken.png").write_bytes(b"not an image")
    with pytest.raises(DataError, match="could not read image"):
        predictor.evaluate(str(root))


# -- validation: clear domain errors, never internals -------------------------


def test_mismatched_sample_counts_are_a_data_error(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    with pytest.raises(DataError, match="8 sample.*7"):
        predictor.evaluate(_X, _TRUE[:-1])
    reg = load_predictor(_save_regression_artifact(tmp_path))
    with pytest.raises(DataError, match="4 sample.*3"):
        reg.evaluate(_REG_X, _REG_Y[:-1])


def test_wrong_feature_count_is_a_data_error(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    with pytest.raises(DataError, match="expected 2 input feature"):
        predictor.evaluate(np.zeros((8, 3), dtype=np.float32), _TRUE)
    reg = load_predictor(_save_regression_artifact(tmp_path))
    with pytest.raises(DataError, match="expected 2 input feature"):
        reg.evaluate(np.zeros((4, 5), dtype=np.float32), _REG_Y)


def test_wrong_input_shape_for_an_artifact_without_a_schema_is_a_data_error_not_shape_mismatch(tmp_path):
    # Conv1d-first => no InputSchema, so the early feature-count check cannot catch this; the model itself would
    # raise ShapeMismatchError. evaluate() must surface a DataError instead.
    forge.random.seed(0)
    model = Sequential(Conv1d(1, 2, kernel_size=3), Flatten(), Linear(2 * 6, 2))
    path = tmp_path / "conv1d.forge"
    save_model(model, str(path), classes=["a", "b"], task="tabular_classification")
    predictor = load_predictor(str(path))
    assert predictor.input_schema is None

    good = predictor.evaluate(np.zeros((4, 1, 8), dtype=np.float32), ["a", "b", "a", "b"])
    assert good.samples == 4
    with pytest.raises(DataError) as excinfo:
        predictor.evaluate(np.zeros((4, 1, 9), dtype=np.float32), ["a", "b", "a", "b"])
    assert not isinstance(excinfo.value, ShapeMismatchError)
    assert "not compatible" in str(excinfo.value)


@pytest.mark.parametrize("task", ["segmentation", "sequence"])
def test_unsupported_tasks_raise_a_data_error(tmp_path, task):
    forge.random.seed(0)
    if task == "segmentation":
        model = Sequential(Conv2d(3, 1, kernel_size=3, padding=1))
        save_model(model, str(tmp_path / "m.forge"), preprocessing=Normalize(mean=0.0, std=255.0), task="segmentation")
    else:
        class _Step(Module):
            def __init__(self):
                super().__init__()
                self.cell = RNNCell(4, 4)
                self.out = Linear(4, 4)

            def step(self, x, h):
                h = self.cell(x, h)
                return self.out(h), h

            def init_hidden(self, batch_size, device="cpu"):
                return self.cell.init_hidden(batch_size, device=device)

        register_module("EvalStepModelM113", _Step, get_config=lambda m: {})
        save_model(_Step(), str(tmp_path / "m.forge"), classes=list("abcd"), task="sequence")

    predictor = load_predictor(str(tmp_path / "m.forge"))
    with pytest.raises(DataError, match=f"does not support task '{task}'"):
        predictor.evaluate(np.zeros((2, 3), dtype=np.float32), [0, 1])


def test_missing_labels_and_bad_labels_are_data_errors(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    with pytest.raises(DataError, match="requires y"):
        predictor.evaluate(_X)
    with pytest.raises(DataError, match=r"\['banana'\]"):
        predictor.evaluate(_X, ["banana"] + _TRUE[1:])
    with pytest.raises(DataError, match="outside"):
        predictor.evaluate(_X, [0, 1, 2, 3, 4, 0, 1, 2])
    with pytest.raises(DataError, match="outside"):
        predictor.evaluate(_X, [-1, 1, 2, 0, 0, 0, 1, 2])
    with pytest.raises(DataError, match="class names .* or integer"):
        predictor.evaluate(_X, np.array([0.0, 1.0] * 4))
    with pytest.raises(DataError, match="class names .* or integer"):
        predictor.evaluate(_X, [True, False] * 4)
    with pytest.raises(DataError, match="1-D"):
        predictor.evaluate(_X, np.zeros((8, 1), dtype=np.int64))
    with pytest.raises(DataError, match="no labels"):
        predictor.evaluate(_X, [])
    reg = load_predictor(_save_regression_artifact(tmp_path))
    with pytest.raises(DataError, match="requires y"):
        reg.evaluate(_REG_X)


def test_bad_features_are_data_errors(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    with pytest.raises(DataError, match="batched"):
        predictor.evaluate(_X[0], _TRUE[:1])                          # a single unbatched row
    with pytest.raises(DataError, match="no samples"):
        predictor.evaluate(np.zeros((0, 2), dtype=np.float32), [])
    with pytest.raises(DataError, match="Tensor, NumPy array, or list"):
        predictor.evaluate("not features", _TRUE)
    nan_rows = _X.copy()
    nan_rows[2, 1] = np.nan
    with pytest.raises(DataError, match="non-finite"):
        predictor.evaluate(nan_rows, _TRUE)
    with pytest.raises(DataError):
        predictor.evaluate([[1.0, 2.0], [3.0]], ["zebra", "apple"])   # ragged rows


def test_bad_regression_targets_are_data_errors(tmp_path):
    reg = load_predictor(_save_regression_artifact(tmp_path))
    with pytest.raises(DataError, match="non-finite"):
        reg.evaluate(_REG_X, np.array([1.0, np.nan, 2.0, 3.0]))
    with pytest.raises(DataError, match="numeric"):
        reg.evaluate(_REG_X, ["a", "b", "c", "d"])
    with pytest.raises(DataError, match="one target per model output"):
        reg.evaluate(_REG_X, np.zeros((4, 3)))
    with pytest.raises(DataError, match=r"\(n,\) or \(n, outputs\)"):
        reg.evaluate(_REG_X, np.zeros((4, 1, 1)))


def test_invalid_batch_size_is_a_data_error(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    for bad in (0, -1, 2.5, True, None):
        with pytest.raises(DataError, match="batch_size"):
            predictor.evaluate(_X, _TRUE, batch_size=bad)


def test_artifact_without_classes_cannot_be_evaluated_as_a_classifier(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path, classes=None, name="nocls.forge"))
    with pytest.raises(PersistenceError, match="no class list"):
        predictor.evaluate(_X, [0, 1, 2, 0, 0, 1, 0, 1])


def test_classes_that_disagree_with_the_model_output_are_a_persistence_error(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path, classes=["a", "b", "c"], name="bad.forge"))
    with pytest.raises(PersistenceError, match="inconsistent"):
        predictor.evaluate(_X, [0, 1, 2, 0, 0, 1, 0, 1])


# -- read-only behaviour and reuse of the loaded predictor --------------------


def _parameter_snapshot(model) -> list:
    return [p.to("cpu").numpy().copy() for p in model.parameters()]


def test_evaluate_leaves_the_artifact_model_and_metadata_untouched(tmp_path):
    path = _save_scoring_artifact(tmp_path)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    predictor = load_predictor(path)
    before = _parameter_snapshot(predictor.model)
    classes_before, task_before, schema_before = list(predictor.classes), predictor.task, predictor.input_schema
    preprocessing_before = repr(predictor._preprocessing)

    predictor.evaluate(_X, _TRUE)
    predictor.evaluate(_X, _TRUE, batch_size=3)

    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for a, b in zip(before, _parameter_snapshot(predictor.model)):
        np.testing.assert_array_equal(a, b)
    assert (list(predictor.classes), predictor.task, predictor.input_schema) == (classes_before, task_before, schema_before)
    assert repr(predictor._preprocessing) == preprocessing_before


@pytest.mark.parametrize("was_training", [True, False])
def test_evaluate_restores_the_models_train_eval_mode(tmp_path, was_training):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    predictor.model.train(was_training)
    predictor.evaluate(_X, _TRUE)
    assert predictor.model.training is was_training


def test_evaluation_runs_in_eval_mode_and_consumes_no_rng(tmp_path):
    forge.random.seed(0)
    model = Sequential(Linear(2, 8), Dropout(0.9), Linear(8, 2))
    path = tmp_path / "dropout.forge"
    save_model(model, str(path), classes=["a", "b"], task="tabular_classification")
    predictor = load_predictor(str(path))
    predictor.model.train(True)     # if evaluate() left Dropout active, results would differ between calls
    rng_state = forge.random.get_state()

    first = predictor.evaluate(_X, ["a", "b"] * 4)
    second = predictor.evaluate(_X, ["a", "b"] * 4)

    assert forge.random.get_state() == rng_state
    np.testing.assert_array_equal(first.confusion_matrix, second.confusion_matrix)
    assert first.loss == second.loss and first.accuracy == second.accuracy


def test_repeated_evaluation_is_identical(tmp_path):
    predictor = load_predictor(_save_scoring_artifact(tmp_path))
    first = predictor.evaluate(_X, _TRUE)
    for _ in range(3):
        again = predictor.evaluate(_X, _TRUE)
        np.testing.assert_array_equal(again.confusion_matrix, first.confusion_matrix)
        assert (again.loss, again.accuracy, again.precision, again.recall) == (
            first.loss, first.accuracy, first.precision, first.recall,
        )


def test_evaluate_never_reloads_the_artifact(tmp_path, monkeypatch):
    import forge.serialization.model as model_module

    predictor = load_predictor(_save_scoring_artifact(tmp_path))

    def _forbidden(*args, **kwargs):
        raise AssertionError("evaluate() must not reopen or reconstruct the artifact")

    for name in ("inspect_model", "load_model", "load_preprocessing", "load_classes"):
        monkeypatch.setattr(model_module, name, _forbidden)

    assert predictor.evaluate(_X, _TRUE, batch_size=2).samples == 8
