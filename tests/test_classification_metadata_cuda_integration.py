"""Milestone 72 CUDA test: class-label metadata + `interpret_classification()`
with a CUDA-loaded model.

Mirrors `tests/test_preprocessing_persistence_cuda_integration.py`'s own
skip/structure conventions. `classes`/`interpret_classification()` are
host-only by construction (a plain JSON list plus NumPy softmax over an
already-`.to("cpu")`'d `predict()` output -- see `forge/training/
inference.py`), so the only genuinely CUDA-relevant question is whether
this composes correctly with a model reloaded onto `device="cuda"`, and
that a CPU-loaded and CUDA-loaded copy of the same model produce the same
interpreted label. Skips cleanly without a working CUDA backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.data.transforms import Compose, Normalize, Resize
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU
from forge.serialization import load_classes, load_model, load_preprocessing, register_module, save_model
from forge.training import ClassificationPrediction, interpret_classification, predict

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


class _TinyCUDACNN(Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.fc = Linear(4 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(self.flatten(x))


register_module(
    "_TinyCUDACNN_M72Test",
    _TinyCUDACNN,
    get_config=lambda m: {"num_classes": m.fc.out_features},
)


def test_cuda_model_interprets_classes_saved_alongside_it(tmp_path):
    model = _TinyCUDACNN(num_classes=2).to("cuda")
    preprocessing = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])
    classes = ["cat", "dog"]

    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing, classes=classes)

    reloaded_model = load_model(str(path), device="cuda")
    reloaded_preprocessing = load_preprocessing(str(path))
    reloaded_classes = load_classes(str(path))
    assert reloaded_classes == classes

    raw = Tensor(np.full((3, 40, 32), 128.0, dtype=np.float32))
    prepared = reloaded_preprocessing(raw)
    batch = prepared.reshape(1, *prepared.shape)

    output = predict(reloaded_model, batch)  # a CUDA forward pass, CPU result
    assert output.device.type == "cpu"

    result = interpret_classification(output, reloaded_classes)[0]
    assert isinstance(result, ClassificationPrediction)
    assert result.label in classes
    assert 0.0 <= result.confidence <= 1.0


def test_cpu_and_cuda_loaded_models_agree_on_interpreted_label(tmp_path):
    model = _TinyCUDACNN(num_classes=2).to("cuda")
    preprocessing = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])
    classes = ["circle", "square"]

    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing, classes=classes)

    raw = Tensor(np.full((3, 20, 24), 90.0, dtype=np.float32))
    prepared = load_preprocessing(str(path))(raw)
    batch = prepared.reshape(1, *prepared.shape)

    cpu_model = load_model(str(path), device="cpu")
    cuda_model = load_model(str(path), device="cuda")
    cpu_result = interpret_classification(predict(cpu_model, batch), load_classes(str(path)))[0]
    cuda_result = interpret_classification(predict(cuda_model, batch), load_classes(str(path)))[0]

    assert cpu_result.label == cuda_result.label
    assert cpu_result.index == cuda_result.index
    np.testing.assert_allclose(cpu_result.confidence, cuda_result.confidence, atol=1e-4)
