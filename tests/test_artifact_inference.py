"""Milestone 82 tests: `forge.predict_artifact()`, the portable-artifact
inference API.

Covers turning a `.forge` file + one new image file directly into a
`ClassificationPrediction` (or a raw class index, when no `classes=` was
saved) -- composing `load_model()`/`load_preprocessing()`/`load_classes()`/
`predict()`/`interpret_classification()` in one call, with no manual
reconstruction of the training-time preprocessing/interpretation pipeline.
See `forge/training/inference.py::predict_artifact()` and
`docs/development/m82-artifact-inference.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.data import ImageFolder
from forge.data.transforms import Compose, Normalize, Resize
from forge.exceptions import DataError, PersistenceError
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential
from forge.serialization import load_classes, load_model, load_preprocessing, save_model
from forge.training import ClassificationPrediction, interpret_classification, predict, predict_artifact
from forge.training.inference import predict_artifact as predict_artifact_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESIZE_SIZE = (8, 8)


class _TinyCNN(Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.fc = Linear(4 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(self.flatten(x))


from forge.serialization import register_module  # noqa: E402

register_module(
    "_TinyCNN_M82Test",
    _TinyCNN,
    get_config=lambda m: {"num_classes": m.fc.out_features},
)


def _build_transform():
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


def _make_image(path: Path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _saved_tiny_model(tmp_path, *, preprocessing=None, classes=None, seed=0):
    forge.random.seed(seed)
    model = _TinyCNN(num_classes=len(classes) if classes else 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing, classes=classes)
    return path


# -- basic API shape ----------------------------------------------------------


def test_predict_artifact_is_reexported_consistently():
    assert forge.predict_artifact is predict_artifact
    assert forge.training.predict_artifact is predict_artifact
    assert predict_artifact is predict_artifact_direct


def test_predict_artifact_returns_classification_prediction_with_label_and_confidence(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_artifact(str(model_path), str(image_path))
    assert isinstance(result, ClassificationPrediction)
    assert result.label in ("cat", "dog")
    assert result.index in (0, 1)
    assert 0.0 <= result.confidence <= 1.0


def test_predict_artifact_accepts_a_pathlib_path_for_the_image(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_artifact(str(model_path), image_path)
    assert isinstance(result, ClassificationPrediction)


def test_predict_artifact_without_classes_returns_raw_index(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=None)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_artifact(str(model_path), str(image_path))
    assert isinstance(result, int)
    assert result in (0, 1)


def test_predict_artifact_matches_the_manual_load_predict_interpret_pipeline(tmp_path):
    """predict_artifact() must be a pure composition -- bit-for-bit identical
    to hand-assembling load_model()/load_preprocessing()/load_classes()/
    predict()/interpret_classification() (the exact sequence it replaces)."""
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path, fill=200)

    result = predict_artifact(str(model_path), str(image_path))

    preprocessing = load_preprocessing(str(model_path))
    model = load_model(str(model_path))
    raw = ImageFolder._load_image(image_path)
    batch = preprocessing(raw).reshape(1, 3, *_RESIZE_SIZE)
    expected = interpret_classification(predict(model, batch), load_classes(str(model_path)))[0]

    assert result.label == expected.label
    assert result.index == expected.index
    assert result.confidence == pytest.approx(expected.confidence, abs=1e-6)


def test_predict_artifact_applies_persisted_resize_for_a_different_resolution_image(tmp_path):
    """The new image's resolution deliberately differs from the persisted
    Resize target -- proving the saved preprocessing pipeline is actually
    exercised, not just a same-shape passthrough."""
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path, size=(37, 51))  # != _RESIZE_SIZE

    result = predict_artifact(str(model_path), str(image_path))
    assert isinstance(result, ClassificationPrediction)


def test_predict_artifact_device_override_accepted_on_cpu_only_machine(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_artifact(str(model_path), str(image_path), device="cpu")
    assert isinstance(result, ClassificationPrediction)


# -- error handling -------------------------------------------------------


def test_predict_artifact_without_preprocessing_raises_persistence_error(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=None, classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    with pytest.raises(PersistenceError, match="preprocessing"):
        predict_artifact(str(model_path), str(image_path))


def test_predict_artifact_missing_model_file_raises_persistence_error(tmp_path):
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    with pytest.raises(PersistenceError):
        predict_artifact(str(tmp_path / "nope.forge"), str(image_path))


def test_predict_artifact_missing_image_file_raises_data_error(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])

    with pytest.raises(DataError):
        predict_artifact(str(model_path), str(tmp_path / "nope.png"))


def test_predict_artifact_rejects_unsupported_input_types(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])

    with pytest.raises(DataError):
        predict_artifact(str(model_path), 12345)

    with pytest.raises(DataError):
        predict_artifact(str(model_path), forge.Tensor(np.zeros((3, 8, 8), dtype=np.float32)))


def test_predict_artifact_corrupt_image_file_raises_data_error(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform(), classes=["cat", "dog"])
    bad_image = tmp_path / "not_an_image.png"
    bad_image.write_bytes(b"not a real png file")

    with pytest.raises(DataError):
        predict_artifact(str(model_path), str(bad_image))


# -- fresh process ----------------------------------------------------------


def test_predict_artifact_works_from_a_genuinely_separate_process(tmp_path):
    """The whole point of Milestone 82 is that a `.forge` file alone is
    enough -- prove it by launching a real, separate OS process that only
    imports `forge` and calls `forge.predict_artifact()`, with no shared
    memory, module cache, or import state with this test process.

    Built entirely from pre-registered `forge.nn` types (`Sequential`,
    `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, `Linear`) -- unlike this file's
    other tests, a custom `register_module()` call (like `_TinyCNN` above)
    would not exist in the fresh subprocess, which never imports this test
    module.
    """
    forge.random.seed(0)
    model = Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, 2),
    )
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path, size=(30, 20))  # a resolution never "trained" at

    script = (
        "import sys\n"
        "import forge\n"
        f"result = forge.predict_artifact({str(model_path)!r}, {str(image_path)!r})\n"
        "print(f'label={result.label} index={result.index} confidence={result.confidence:.6f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    expected = predict_artifact(str(model_path), str(image_path))
    assert f"label={expected.label} index={expected.index}" in result.stdout
