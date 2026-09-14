"""Milestone 82 CUDA tests: `forge.predict_artifact()` on the real CUDA backend.

Mirrors `tests/test_artifact_inference.py`'s CPU coverage for the
device-specific behavior only: an artifact saved from a CUDA model round-trips
back onto CUDA by default, an explicit `device="cpu"` override still works,
and CPU/CUDA predictions on the same weights agree within the project's
established tolerance. See `forge/training/inference.py::predict_artifact()`.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import forge
from forge.backend.cuda import is_cuda_available
from forge.data.transforms import Compose, Normalize, Resize
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.serialization import save_model
from forge.training import ClassificationPrediction, predict_artifact

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_RESIZE_SIZE = (8, 8)
TOL = dict(rtol=1e-4, atol=1e-4)


def _build_transform():
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


def _make_image(path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _make_grayscale_image(path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0]), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _tiny_model():
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, 2),
    )


def _tiny_grayscale_model():
    """Milestone 94: the Conv2d(1, ...)-first shape, hardware-verified on CUDA."""
    return Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, 2),
    )


def test_predict_artifact_grayscale_model_on_cuda(tmp_path):
    """Milestone 94: a Conv2d(1, ...) model + grayscale image must predict
    correctly on the real CUDA device, hardware-verified on the 940MX."""
    forge.random.seed(3)
    model = _tiny_grayscale_model().to("cuda")
    model_path = tmp_path / "grayscale_model.forge"
    save_model(model, str(model_path), preprocessing=_build_transform(), classes=["zero", "one"])
    image_path = tmp_path / "query.png"
    _make_grayscale_image(image_path)

    result = predict_artifact(str(model_path), str(image_path))
    assert isinstance(result, ClassificationPrediction)
    assert result.label in ("zero", "one")


def test_predict_artifact_grayscale_model_cpu_and_cuda_agree(tmp_path):
    """CPU/CUDA parity for the grayscale-decode path specifically, mirroring
    `test_predict_artifact_cpu_and_cuda_saved_artifacts_agree_on_the_same_weights`."""
    forge.random.seed(4)
    model = _tiny_grayscale_model()
    image_path = tmp_path / "query.png"
    _make_grayscale_image(image_path, size=(19, 23))

    cpu_path = tmp_path / "cpu_model.forge"
    save_model(model, str(cpu_path), preprocessing=_build_transform(), classes=["zero", "one"])

    model.to("cuda")
    cuda_path = tmp_path / "cuda_model.forge"
    save_model(model, str(cuda_path), preprocessing=_build_transform(), classes=["zero", "one"])

    cpu_result = predict_artifact(str(cpu_path), str(image_path))
    cuda_result = predict_artifact(str(cuda_path), str(image_path))

    assert cpu_result.index == cuda_result.index
    assert cpu_result.label == cuda_result.label
    np.testing.assert_allclose(cpu_result.confidence, cuda_result.confidence, **TOL)


def test_predict_artifact_saved_from_cuda_model_restores_onto_cuda_by_default(tmp_path):
    forge.random.seed(0)
    model = _tiny_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_artifact(str(model_path), str(image_path))
    assert isinstance(result, ClassificationPrediction)
    assert result.label in ("cat", "dog")


def test_predict_artifact_device_override_converts_cuda_artifact_to_cpu(tmp_path):
    forge.random.seed(1)
    model = _tiny_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=_build_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_artifact(str(model_path), str(image_path), device="cpu")
    assert isinstance(result, ClassificationPrediction)


def test_predict_artifact_cpu_and_cuda_saved_artifacts_agree_on_the_same_weights(tmp_path):
    """Save the identical weights once from CPU and once from CUDA (`Module.to()`
    moves Parameters in place -- Milestone 9), predict_artifact() on each
    saved file, and confirm the confidences agree within CUDA/CPU tolerance."""
    forge.random.seed(2)
    model = _tiny_model()
    image_path = tmp_path / "query.png"
    _make_image(image_path, size=(19, 23))  # a resolution never "trained" at

    cpu_path = tmp_path / "cpu_model.forge"
    save_model(model, str(cpu_path), preprocessing=_build_transform(), classes=["cat", "dog"])

    model.to("cuda")
    cuda_path = tmp_path / "cuda_model.forge"
    save_model(model, str(cuda_path), preprocessing=_build_transform(), classes=["cat", "dog"])

    cpu_result = predict_artifact(str(cpu_path), str(image_path))
    cuda_result = predict_artifact(str(cuda_path), str(image_path))

    assert cpu_result.index == cuda_result.index
    assert cpu_result.label == cuda_result.label
    np.testing.assert_allclose(cpu_result.confidence, cuda_result.confidence, **TOL)
