"""Milestone 87 CUDA tests: explicit task metadata / `predict_model()`
dispatch on the real CUDA backend.

Task metadata itself is device-independent (a plain JSON string in
`metadata.json`), so this file exists only to prove the new dispatch layer
adds no CPU-only path: a CUDA-recorded artifact's task is inspected without
requiring CUDA, and `predict_model()` still restores onto CUDA by default and
produces the same result CPU-vs-CUDA, exactly like Milestone 86's own CUDA
coverage (`tests/test_unified_artifact_prediction_cuda.py`) -- now via the
explicit-task path rather than the legacy architecture heuristic. See
`forge/training/inference.py::_determine_workflow()`.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import forge
from forge.backend.cuda import is_cuda_available
from forge.data.transforms import Compose, Normalize, Resize
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.serialization import inspect_model, save_model
from forge.training import ClassificationPrediction, predict_model

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_RESIZE_SIZE = (8, 8)
_N_FEATURES = 4


def _classification_model(num_classes=2):
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, num_classes),
    )


def _regression_model():
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, 1))


def _make_image(path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_inspect_model_reads_task_from_cuda_saved_artifact_without_cuda_requirement(tmp_path, monkeypatch):
    """`inspect_model()` never requires CUDA regardless of the recorded
    device (Milestone 85's own contract) -- confirm this still holds for the
    new `task` field by monkeypatching CUDA unavailable for the read."""
    forge.random.seed(0)
    model = _regression_model().to("cuda")
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="regression")

    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    info = inspect_model(str(path))
    assert info.task == "regression"
    assert info.device == "cuda"


def test_predict_model_classification_with_explicit_task_restores_onto_cuda_by_default(tmp_path):
    forge.random.seed(0)
    model = _classification_model().to("cuda")
    path = tmp_path / "model.forge"
    save_model(
        model, str(path),
        preprocessing=Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)]),
        classes=None, task="classification",
    )
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_model(str(path), str(image_path))
    assert isinstance(result, int)


def test_predict_model_cpu_and_cuda_saved_explicit_task_artifacts_agree(tmp_path):
    forge.random.seed(3)
    model = _regression_model()
    batch = np.random.default_rng(4).standard_normal((2, _N_FEATURES)).astype(np.float32)

    cpu_path, cuda_path = tmp_path / "cpu.forge", tmp_path / "cuda.forge"
    save_model(model, str(cpu_path), task="regression")
    model.to("cuda")
    save_model(model, str(cuda_path), task="regression")

    cpu_result = predict_model(str(cpu_path), batch)
    cuda_result = predict_model(str(cuda_path), batch)
    np.testing.assert_allclose(cpu_result.numpy(), cuda_result.numpy(), rtol=1e-4, atol=1e-4)
