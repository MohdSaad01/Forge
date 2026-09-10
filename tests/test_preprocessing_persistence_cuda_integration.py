"""Milestone 71 CUDA test: preprocessing persistence + a CUDA-loaded model.

`forge.data`/`forge.data.transforms` (and therefore preprocessing
persistence, `forge/serialization/transforms.py`) are CPU-only by design --
see `docs/architecture/data-pipeline.md`'s **Constraints** section and
`docs/architecture/persistence.md`'s **Preprocessing metadata** section.
The only genuinely CUDA-relevant question this milestone raises is whether
`load_preprocessing()` (host-only) still composes correctly with a model
reloaded onto `device="cuda"` via `load_model()` -- i.e. that saving both
together and loading each independently produces a working CPU-preprocess
-> CUDA-model -> `forge.predict()` pipeline. Mirrors `tests/
test_image_folder_classification_cuda_integration.py`'s own skip/structure
conventions. Skips cleanly without a working CUDA backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge.backend.cuda import is_cuda_available
from forge.data.transforms import Compose, Normalize, Resize
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU
from forge.serialization import load_model, load_preprocessing, register_module, save_model
from forge.training import predict

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
    "_TinyCUDACNN_M71Test",
    _TinyCUDACNN,
    get_config=lambda m: {"num_classes": m.fc.out_features},
)


def test_cuda_model_uses_persisted_cpu_preprocessing(tmp_path):
    model = _TinyCUDACNN(num_classes=2).to("cuda")
    preprocessing = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])

    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing)

    reloaded_model = load_model(str(path), device="cuda")
    reloaded_preprocessing = load_preprocessing(str(path))

    from forge import Tensor

    raw = Tensor(np.full((3, 40, 32), 128.0, dtype=np.float32))  # a different source resolution
    prepared = reloaded_preprocessing(raw)  # runs on CPU regardless of the model's device
    assert prepared.shape == (3, 16, 16)
    assert prepared.device.type == "cpu"

    batch = prepared.reshape(1, *prepared.shape)
    prediction = predict(reloaded_model, batch)  # predict() moves the batch to the model's device
    assert prediction.device.type == "cpu"  # predict() always returns a CPU Tensor
    assert prediction.shape == (1, 2)
    assert np.all(np.isfinite(prediction.numpy()))


def test_cuda_and_cpu_loaded_models_agree_given_identical_preprocessing(tmp_path):
    # (16, 16) -> (matches `_TinyCUDACNN`'s fixed `Linear(4 * 8 * 8, ...)` after
    # one MaxPool2d(2)) -- must stay consistent with the model's expected
    # flattened feature count, unlike the CPU test file's own model (which
    # uses a different Resize target and is unaffected by this).
    model = _TinyCUDACNN(num_classes=2).to("cuda")
    preprocessing = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])

    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing)

    from forge import Tensor

    raw = Tensor(np.full((3, 20, 24), 90.0, dtype=np.float32))
    prepared = load_preprocessing(str(path))(raw)
    assert prepared.shape == (3, 16, 16)
    batch = prepared.reshape(1, *prepared.shape)

    cpu_model = load_model(str(path), device="cpu")
    cuda_model = load_model(str(path), device="cuda")
    cpu_pred = predict(cpu_model, batch).numpy()
    cuda_pred = predict(cuda_model, batch).numpy()
    np.testing.assert_allclose(cpu_pred, cuda_pred, atol=1e-4)
