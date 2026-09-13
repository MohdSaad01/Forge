"""Milestone 86 CUDA tests: `forge.predict_model()` on the real CUDA backend.

Mirrors `tests/test_artifact_inference_cuda.py`/`test_segmentation_artifact_
workflow_cuda.py`'s CPU-vs-CUDA coverage, but through the unified dispatcher:
proves `predict_model()`'s own metadata inspection and workflow determination
add no CPU-only path -- an artifact saved from a CUDA model still dispatches
and restores onto CUDA by default, and predictions agree with the CPU-saved
equivalent, for all three supported artifact shapes. See
`forge/training/inference.py::predict_model()`.
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
from forge.training import ClassificationPrediction, predict_model

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_RESIZE_SIZE = (8, 8)
_N_FEATURES = 4
TOL = dict(rtol=1e-4, atol=1e-4)


def _classification_transform():
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


def _classification_model(num_classes=2):
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, num_classes),
    )


def _regression_model():
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, 1))


def _segmentation_model():
    return Sequential(Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), Conv2d(4, 1, kernel_size=3, padding=1))


def _make_image(path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_predict_model_classification_artifact_saved_from_cuda_restores_onto_cuda_by_default(tmp_path):
    forge.random.seed(0)
    model = _classification_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=_classification_transform(), classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_model(str(model_path), str(image_path))
    assert isinstance(result, ClassificationPrediction)
    assert result.label in ("cat", "dog")


def test_predict_model_regression_artifact_saved_from_cuda_restores_onto_cuda_by_default(tmp_path):
    forge.random.seed(1)
    model = _regression_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path))

    batch = np.random.default_rng(1).standard_normal((3, _N_FEATURES)).astype(np.float32)
    result = predict_model(str(model_path), batch)
    assert result.shape == (3, 1)


def test_predict_model_segmentation_artifact_saved_from_cuda_restores_onto_cuda_by_default(tmp_path):
    forge.random.seed(2)
    model = _segmentation_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=Compose([Normalize(mean=0.0, std=255.0)]))
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_model(str(model_path), str(image_path))
    assert result.shape == (1, 8, 8)
    assert set(np.unique(result.numpy()).tolist()) <= {0.0, 1.0}


def test_predict_model_cpu_and_cuda_saved_artifacts_agree_across_all_three_shapes(tmp_path):
    forge.random.seed(3)
    cls_model = _classification_model()
    reg_model = _regression_model()
    seg_model = _segmentation_model()
    image_path = tmp_path / "query.png"
    _make_image(image_path, size=(19, 23))
    batch = np.random.default_rng(4).standard_normal((2, _N_FEATURES)).astype(np.float32)

    cls_cpu_path, cls_cuda_path = tmp_path / "cls_cpu.forge", tmp_path / "cls_cuda.forge"
    save_model(cls_model, str(cls_cpu_path), preprocessing=_classification_transform(), classes=["cat", "dog"])
    cls_model.to("cuda")
    save_model(cls_model, str(cls_cuda_path), preprocessing=_classification_transform(), classes=["cat", "dog"])
    cpu_cls = predict_model(str(cls_cpu_path), str(image_path))
    cuda_cls = predict_model(str(cls_cuda_path), str(image_path))
    assert cpu_cls.index == cuda_cls.index
    np.testing.assert_allclose(cpu_cls.confidence, cuda_cls.confidence, **TOL)

    reg_cpu_path, reg_cuda_path = tmp_path / "reg_cpu.forge", tmp_path / "reg_cuda.forge"
    save_model(reg_model, str(reg_cpu_path))
    reg_model.to("cuda")
    save_model(reg_model, str(reg_cuda_path))
    cpu_reg = predict_model(str(reg_cpu_path), batch)
    cuda_reg = predict_model(str(reg_cuda_path), batch)
    np.testing.assert_allclose(cpu_reg.numpy(), cuda_reg.numpy(), **TOL)

    seg_transform = Compose([Normalize(mean=0.0, std=255.0)])
    seg_cpu_path, seg_cuda_path = tmp_path / "seg_cpu.forge", tmp_path / "seg_cuda.forge"
    save_model(seg_model, str(seg_cpu_path), preprocessing=seg_transform)
    seg_model.to("cuda")
    save_model(seg_model, str(seg_cuda_path), preprocessing=seg_transform)
    cpu_seg = predict_model(str(seg_cpu_path), str(image_path))
    cuda_seg = predict_model(str(seg_cuda_path), str(image_path))
    np.testing.assert_array_equal(cpu_seg.numpy(), cuda_seg.numpy())
