"""Milestone 86 tests: `forge.predict_model()`, the unified portable-artifact
prediction entry point over `predict_artifact()`/`predict_tensor_artifact()`/
`predict_image_artifact()` (Milestones 82-84).

Covers workflow determination from an artifact's own persisted metadata
(`_determine_workflow()`, `forge/training/inference.py`), delegation
correctness against each task-specific function, input-type validation,
the deliberate refusal to guess for an undetermined artifact, backward
compatibility of the three existing functions, and a genuine fresh-process
run. See `forge/training/inference.py::predict_model()` and
`docs/development/m86-unified-artifact-prediction.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.data.transforms import Compose, Normalize, Resize
from forge.exceptions import DataError, PersistenceError
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.serialization import save_model
from forge.training import (
    ClassificationPrediction,
    predict_artifact,
    predict_image_artifact,
    predict_model,
    predict_tensor_artifact,
)
from forge.training.inference import predict_model as predict_model_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESIZE_SIZE = (8, 8)
_N_FEATURES = 4


# -- shared fixtures, one per real supported artifact shape -------------------


def _classification_transform():
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


def _classification_model(num_classes=2):
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, num_classes),
    )


def _regression_transform():
    return Normalize(
        mean=np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
        std=np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32),
    )


def _regression_model():
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, 1))


def _segmentation_transform():
    return Compose([Normalize(mean=0.0, std=255.0)])


def _segmentation_model():
    return Sequential(Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), Conv2d(4, 1, kernel_size=3, padding=1))


def _make_image(path: Path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _saved_classification_model(tmp_path, *, classes=("cat", "dog"), seed=0) -> Path:
    forge.random.seed(seed)
    model = _classification_model(num_classes=len(classes))
    path = tmp_path / "classification.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=list(classes))
    return path


def _saved_regression_model(tmp_path, *, seed=0) -> Path:
    forge.random.seed(seed)
    model = _regression_model()
    path = tmp_path / "regression.forge"
    save_model(model, str(path), preprocessing=_regression_transform())
    return path


def _saved_segmentation_model(tmp_path, *, seed=0) -> Path:
    forge.random.seed(seed)
    model = _segmentation_model()
    path = tmp_path / "segmentation.forge"
    save_model(model, str(path), preprocessing=_segmentation_transform())
    return path


# -- basic API shape ----------------------------------------------------------


def test_predict_model_is_reexported_consistently():
    assert forge.predict_model is predict_model
    assert forge.training.predict_model is predict_model
    assert predict_model is predict_model_direct


# -- dispatch -----------------------------------------------------------------


def test_predict_model_dispatches_classification_artifact(tmp_path):
    model_path = _saved_classification_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_model(str(model_path), str(image_path))
    assert isinstance(result, ClassificationPrediction)
    assert result.label in ("cat", "dog")


def test_predict_model_dispatches_regression_artifact(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    batch = np.random.default_rng(1).standard_normal((3, _N_FEATURES)).astype(np.float32)

    result = predict_model(str(model_path), batch)
    assert isinstance(result, forge.Tensor)
    assert result.shape == (3, 1)


def test_predict_model_dispatches_segmentation_artifact(tmp_path):
    model_path = _saved_segmentation_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_model(str(model_path), str(image_path))
    assert isinstance(result, forge.Tensor)
    assert result.shape == (1, 8, 8)
    assert set(np.unique(result.numpy()).tolist()) <= {0.0, 1.0}


# -- delegation correctness ----------------------------------------------------


def test_predict_model_matches_predict_artifact_bit_for_bit(tmp_path):
    model_path = _saved_classification_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path, fill=200)

    unified = predict_model(str(model_path), str(image_path))
    direct = predict_artifact(str(model_path), str(image_path))

    assert unified.label == direct.label
    assert unified.index == direct.index
    assert unified.confidence == pytest.approx(direct.confidence, abs=1e-6)


def test_predict_model_matches_predict_tensor_artifact_bit_for_bit(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    batch = np.random.default_rng(2).standard_normal((5, _N_FEATURES)).astype(np.float32)

    unified = predict_model(str(model_path), batch)
    direct = predict_tensor_artifact(str(model_path), batch)

    np.testing.assert_array_equal(unified.numpy(), direct.numpy())


def test_predict_model_matches_predict_image_artifact_bit_for_bit(tmp_path):
    model_path = _saved_segmentation_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path, fill=210)

    unified = predict_model(str(model_path), str(image_path))
    direct = predict_image_artifact(str(model_path), str(image_path))

    np.testing.assert_array_equal(unified.numpy(), direct.numpy())


def test_predict_model_documented_limitation_classification_without_classes_misdispatches(tmp_path):
    """A classification model saved with `classes=None` is a real, valid
    state (`predict_artifact()`'s own docstring) -- but it is also
    `Linear`-terminated, exactly like a regression model, and has no
    `classes` to disambiguate it. `_determine_workflow()` documents this as
    a known limitation rather than silently guessing right sometimes; this
    test pins that documented behavior down so a future change to the
    dispatch signal shows up here, not as a surprise. Callers who saved a
    classification artifact this way should call `predict_artifact()`
    directly (as `predict_artifact()` still supports unchanged -- see the
    backward-compatibility tests below)."""
    forge.random.seed(0)
    model = _classification_model(num_classes=2)
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=_classification_transform(), classes=None)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    with pytest.raises(DataError):
        predict_model(str(model_path), str(image_path))

    # The task-specific function itself is unaffected by this limitation.
    result = predict_artifact(str(model_path), str(image_path))
    assert isinstance(result, int)


# -- input validation -----------------------------------------------------


def test_predict_model_rejects_a_tensor_for_a_classification_artifact(tmp_path):
    model_path = _saved_classification_model(tmp_path)

    with pytest.raises(DataError):
        predict_model(str(model_path), forge.Tensor(np.zeros((3, 8, 8), dtype=np.float32)))


def test_predict_model_rejects_an_image_path_for_a_regression_artifact(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    with pytest.raises(DataError):
        predict_model(str(model_path), str(image_path))


def test_predict_model_rejects_a_tensor_for_a_segmentation_artifact(tmp_path):
    model_path = _saved_segmentation_model(tmp_path)

    with pytest.raises(DataError):
        predict_model(str(model_path), forge.Tensor(np.zeros((1, _N_FEATURES), dtype=np.float32)))


# -- unsupported artifacts ------------------------------------------------


def test_predict_model_raises_clearly_for_an_undeterminable_artifact(tmp_path):
    """No saved `classes`, and an architecture with neither a `Linear` nor a
    `Conv2d` layer -- `_determine_workflow()` has no signal left to use, and
    must fail clearly rather than guessing one of the three workflows."""
    forge.random.seed(0)
    model = Sequential(ReLU())
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path))

    with pytest.raises(PersistenceError, match="could not determine"):
        predict_model(str(model_path), np.zeros((1, _N_FEATURES), dtype=np.float32))


def test_predict_model_missing_file_raises_persistence_error(tmp_path):
    with pytest.raises(PersistenceError):
        predict_model(str(tmp_path / "nope.forge"), np.zeros((1, _N_FEATURES), dtype=np.float32))


# -- backward compatibility ------------------------------------------------


def test_predict_artifact_predict_tensor_artifact_predict_image_artifact_unchanged(tmp_path):
    """predict_model() must add no new argument or behavior to the three
    functions it delegates to -- call each directly, exactly as Milestones
    82-84 established, and confirm they still behave identically."""
    classification_path = _saved_classification_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)
    result = predict_artifact(str(classification_path), str(image_path))
    assert isinstance(result, ClassificationPrediction)

    regression_path = _saved_regression_model(tmp_path)
    batch = np.random.default_rng(3).standard_normal((2, _N_FEATURES)).astype(np.float32)
    result = predict_tensor_artifact(str(regression_path), batch)
    assert isinstance(result, forge.Tensor) and result.shape == (2, 1)

    segmentation_path = _saved_segmentation_model(tmp_path)
    result = predict_image_artifact(str(segmentation_path), str(image_path))
    assert isinstance(result, forge.Tensor) and result.shape == (1, 8, 8)


# -- fresh process ----------------------------------------------------------


def test_predict_model_works_from_a_genuinely_separate_process_for_all_three_artifact_shapes(tmp_path):
    """The whole point of Milestone 86 is that `forge.predict_model()` alone
    -- with no other knowledge of the artifact -- produces a useful
    prediction. Prove it for all three supported artifact shapes, each
    saved to its own file, in one real, separate OS process."""
    classification_path = _saved_classification_model(tmp_path, seed=10)
    image_path = tmp_path / "query.png"
    _make_image(image_path, size=(30, 20))

    regression_path = _saved_regression_model(tmp_path, seed=11)
    raw_x = np.random.default_rng(12).standard_normal((1, _N_FEATURES)).astype(np.float32)

    segmentation_path = _saved_segmentation_model(tmp_path, seed=13)
    seg_image_path = tmp_path / "seg_query.png"
    _make_image(seg_image_path, size=(8, 8), fill=210)

    script = (
        "import numpy as np\n"
        "import forge\n"
        f"raw_x = np.array({raw_x.tolist()!r}, dtype=np.float32)\n"
        f"cls_result = forge.predict_model({str(classification_path)!r}, {str(image_path)!r})\n"
        f"reg_result = forge.predict_model({str(regression_path)!r}, raw_x)\n"
        f"seg_result = forge.predict_model({str(segmentation_path)!r}, {str(seg_image_path)!r})\n"
        "print(f'cls label={cls_result.label} index={cls_result.index} confidence={cls_result.confidence:.6f}')\n"
        "print(f'reg value={reg_result.numpy()[0, 0]:.8f}')\n"
        "print(f'seg shape={seg_result.shape} unique={sorted(set(seg_result.numpy().ravel().tolist()))}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    expected_cls = predict_model(str(classification_path), str(image_path))
    expected_reg = predict_model(str(regression_path), raw_x)
    expected_seg = predict_model(str(segmentation_path), str(seg_image_path))

    assert f"cls label={expected_cls.label} index={expected_cls.index}" in result.stdout
    assert f"reg value={float(expected_reg.numpy()[0, 0]):.8f}" in result.stdout
    assert f"seg shape={expected_seg.shape}" in result.stdout
