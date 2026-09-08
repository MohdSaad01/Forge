"""Milestone 64 integration tests: the `examples/segmentation` pipeline on CUDA.

Mirrors `tests/test_segmentation_example_integration.py` (same small
synthetic dataset) but drives `Trainer(..., device="cuda")`, additionally
verifying CUDA residency of parameters/gradients/Adam state and CPU/CUDA
prediction parity. Skips cleanly when CUDA is unavailable; hardware-verified
on the development machine's GeForce 940MX (CC 5.0) per
`docs/development/development-environment.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.data import DataLoader
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.segmentation.dataset import IMAGE_SIZE, NUM_CHANNELS, make_datasets  # noqa: E402
from examples.segmentation.metrics import IoU, PixelAccuracy  # noqa: E402
from examples.segmentation.model import build_model  # noqa: E402
from examples.segmentation.train import majority_class_baseline  # noqa: E402


def test_full_pipeline_trains_and_beats_baseline_on_cuda():
    forge.random.seed(0)
    train_ds, test_ds = make_datasets(300, 60, seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(2))
    test_loader = DataLoader(test_ds, batch_size=16)

    baseline_accuracy, baseline_iou = majority_class_baseline(test_ds)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", metrics=[PixelAccuracy(), IoU()], verbose=False)

    trainer.fit(train_loader, epochs=8, validation_loader=test_loader)

    eval_result = trainer.evaluate(test_loader)
    assert eval_result.metrics["pixel_accuracy"] > baseline_accuracy
    assert eval_result.metrics["iou"] > baseline_iou + 0.3


def test_cuda_residency_of_parameters_gradients_and_adam_state():
    forge.random.seed(5)
    train_ds, _ = make_datasets(32, 8, seed=6)
    loader = DataLoader(train_ds, batch_size=8, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=1)

    for name, param in model.named_parameters():
        assert isinstance(param._data, CUDAStorage), f"parameter '{name}' is not CUDA-resident"
        assert param.grad is not None, f"parameter '{name}' has no gradient"
        assert isinstance(param.grad._data, CUDAStorage), f"gradient for '{name}' is not CUDA-resident"
        state = optimizer.state[param]
        assert isinstance(state.m, CUDAStorage), f"Adam 'm' for '{name}' is not CUDA-resident"
        assert isinstance(state.v, CUDAStorage), f"Adam 'v' for '{name}' is not CUDA-resident"


def test_checkpoint_save_and_resume_on_cuda(tmp_path):
    forge.random.seed(10)
    train_ds, _ = make_datasets(48, 10, seed=11)
    loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    checkpoint_path = tmp_path / "segmentation_tiny_cuda.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path), device="cuda")
    assert checkpoint.model.device.type == "cuda"
    for _, param in checkpoint.model.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    resumed_trainer = Trainer(checkpoint.model, MSELoss(), checkpoint.optimizer, device="cuda", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.model.device.type == "cuda"


def test_model_persistence_preserves_predictions_on_cuda(tmp_path):
    forge.random.seed(20)
    train_ds, _ = make_datasets(32, 8, seed=21)
    loader = DataLoader(train_ds, batch_size=8, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    query = Tensor(
        np.random.default_rng(22).random((4, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32),
        device="cuda",
    )
    with forge.no_grad():
        pre_save = model(query).to("cpu").numpy()

    model_path = tmp_path / "segmentation_tiny_cuda.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cuda")
    for _, param in reloaded.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    with forge.no_grad():
        post_load = reloaded(query).to("cpu").numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-5)


def test_cpu_and_cuda_prediction_parity_after_training():
    """Same seed/data trained identically on CPU vs. CUDA should predict closely (parity check)."""
    train_ds, _ = make_datasets(64, 4, seed=1)
    query = Tensor(
        np.random.default_rng(99).random((8, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    )

    def run(device: str) -> np.ndarray:
        forge.random.seed(1)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(1))
        model = build_model().to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(model, MSELoss(), optimizer, device=device, verbose=False)
        trainer.fit(loader, epochs=2)
        with forge.no_grad():
            return model(query.to(device)).to("cpu").numpy()

    cpu_pred = run("cpu")
    cuda_pred = run("cuda")
    np.testing.assert_allclose(cuda_pred, cpu_pred, rtol=1e-3, atol=1e-3)
