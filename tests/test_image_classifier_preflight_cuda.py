"""Milestone 115 CUDA tests: `forge.train_image_classifier()` preflight on the real CUDA backend.

Device-specific behaviour only (the CPU file, `tests/test_image_classifier_preflight.py`,
covers the contract): the caller's model is moved to the requested device *before* it is
probed, so the probe runs where training will, and the artifact that comes out is usable
on both devices with the same result. Skips cleanly without a working CUDA backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.backend.cuda import is_cuda_available
from forge.nn import Flatten, Linear, Module, ReLU, Sequential
from forge.training import Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

SIZE = (32, 32)
FLAT = 3 * SIZE[0] * SIZE[1]


def _make_two_class_root(tmp_path: Path, per_class: int = 8) -> Path:
    root = tmp_path / "data"
    for cls, base in [("cat", (200, 0)), ("dog", (0, 200))]:
        (root / cls).mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (20, 20), (base[0], base[1], i * 5)).save(root / cls / f"{cls}_{i}.png")
    return root


def _train(root, path, **kwargs):
    return forge.train_image_classifier(
        root, path=path, epochs=1, batch_size=4, image_size=SIZE, val_fraction=0.25, device="cuda",
        verbose=False, **kwargs,
    )


def _flat_model(outputs: int) -> Sequential:
    return Sequential(Flatten(), Linear(FLAT, 8), ReLU(), Linear(8, outputs))


class Unregistered(Module):
    """A working model `save_model()` cannot serialise; records the device its probe input arrived on."""

    def __init__(self):
        super().__init__()
        self.fc = Linear(FLAT, 2)
        self.input_devices = []

    def forward(self, x):
        self.input_devices.append(x.device.type)
        return self.fc(x.reshape(x.shape[0], FLAT))  # explicit size: CUDA reshape does not take -1


@pytest.fixture
def no_training(monkeypatch):
    def tripwire(self, *args, **kwargs):
        raise AssertionError("training started: a preflight check ran too late")

    monkeypatch.setattr(Trainer, "fit", tripwire)


def test_a_wrong_output_width_is_rejected_before_epoch_1_on_cuda(tmp_path, no_training):
    root = _make_two_class_root(tmp_path)
    with pytest.raises(forge.TrainerError, match=r"needs \(batch, 2\)"):
        _train(root, str(tmp_path / "m.forge"), model=_flat_model(5))
    assert not (tmp_path / "m.forge").exists()


def test_a_bad_path_is_rejected_before_epoch_1_on_cuda(tmp_path, no_training):
    root = _make_two_class_root(tmp_path)
    with pytest.raises(forge.PersistenceError, match="does not exist"):
        _train(root, str(tmp_path / "nope" / "m.forge"))


def test_a_cpu_model_is_moved_to_cuda_before_the_probe(tmp_path, no_training):
    root = _make_two_class_root(tmp_path)
    model = Unregistered()
    assert model.device is None or model.device.type == "cpu"

    with pytest.raises(forge.PersistenceError, match="not registered"):  # passes the probe, fails the dry-run save
        _train(root, str(tmp_path / "m.forge"), model=model)

    assert model.input_devices == ["cuda"]  # the probe ran where training will


def test_a_cpu_model_trains_on_cuda_with_the_preflight(tmp_path):
    root = _make_two_class_root(tmp_path)

    result = _train(root, str(tmp_path / "m.forge"), model=_flat_model(2))

    assert result.model.device.type == "cuda"
    assert forge.inspect_model(str(tmp_path / "m.forge")).device == "cuda"


def test_the_same_saved_weights_evaluate_identically_on_cpu_and_cuda(tmp_path):
    root = _make_two_class_root(tmp_path)
    path = str(tmp_path / "m.forge")
    _train(root, path, model=_flat_model(2))

    on_cuda = forge.load_predictor(path, device="cuda").evaluate(root)
    on_cpu = forge.load_predictor(path, device="cpu").evaluate(root)

    assert on_cuda.accuracy == on_cpu.accuracy
    assert np.array_equal(on_cuda.confusion_matrix, on_cpu.confusion_matrix)
    assert on_cuda.loss == pytest.approx(on_cpu.loss, abs=1e-4)
