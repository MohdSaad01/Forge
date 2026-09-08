"""Milestone 63 integration tests: the `examples/autoencoder` pipeline on CUDA.

Mirrors `tests/test_autoencoder_example_integration.py` (same tiny
synthetic `(N, 1, 28, 28)` stand-in dataset) but drives `Trainer(...,
device="cuda")`, additionally verifying CUDA residency of parameters/
gradients/Adam state (including the new `UpsampleNearest2d` decoder path's
own gradient) and CPU/CUDA prediction parity. Skips cleanly when CUDA is
unavailable; hardware-verified on the development machine's GeForce 940MX
(CC 5.0) per `docs/development/development-environment.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.autoencoder.model import build_model  # noqa: E402

_NUM_CLASSES = 10


def _make_synthetic_images(n: int, seed: int) -> np.ndarray:
    """A tiny `(N, 1, 28, 28)` float32 stand-in with real learnable structure.

    Same "class L -> a bright vertical stripe at columns [2L, 2L+2)" motif
    `tests/test_autoencoder_example_integration.py`/`test_mnist_example_
    cuda_integration.py` use -- pure i.i.d. noise has no structure for any
    autoencoder to learn (a model cannot beat "predict the mean" on truly
    random pixels), so this dataset needs the same class-correlated
    structure the CPU test uses, not just a matching shape.
    """
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, 1, 28, 28)) * 0.05).astype(np.float32)
    for i in range(n):
        label = i % _NUM_CLASSES
        col = label * 2
        stripe = 1.0 + rng.standard_normal((28, 2)).astype(np.float32) * 0.05
        X[i, 0, :, col : col + 2] = stripe
    return np.clip(X, 0.0, 1.0)


def _make_reconstruction_dataset(n: int, seed: int) -> TensorDataset:
    images = Tensor(_make_synthetic_images(n, seed))
    return TensorDataset(images, images)


def test_full_pipeline_trains_and_beats_baseline_on_cuda():
    forge.random.seed(0)
    train_ds = _make_reconstruction_dataset(200, seed=1)
    test_ds = _make_reconstruction_dataset(60, seed=2)
    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True, generator=np.random.default_rng(3))
    test_loader = DataLoader(test_ds, batch_size=20)

    train_images = np.stack([train_ds[i][0].numpy() for i in range(len(train_ds))])
    test_images = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))])
    baseline_mse = float(((test_images - train_images.mean(axis=0)) ** 2).mean())

    model = build_model(latent_dim=8).to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)

    history = trainer.fit(train_loader, epochs=15, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    assert eval_result.loss < baseline_mse * 0.5


def test_cuda_residency_of_parameters_gradients_and_adam_state():
    forge.random.seed(5)
    loader = DataLoader(_make_reconstruction_dataset(64, seed=6), batch_size=8, shuffle=False)

    model = build_model(latent_dim=8).to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
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
    loader = DataLoader(_make_reconstruction_dataset(64, seed=11), batch_size=16, shuffle=False)

    model = build_model(latent_dim=8).to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    checkpoint_path = tmp_path / "autoencoder_tiny_cuda.ckpt"
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
    loader = DataLoader(_make_reconstruction_dataset(48, seed=21), batch_size=8, shuffle=False)

    model = build_model(latent_dim=8).to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    query = Tensor(np.random.default_rng(22).random((4, 1, 28, 28)).astype(np.float32), device="cuda")
    with forge.no_grad():
        pre_save = model(query).to("cpu").numpy()

    model_path = tmp_path / "autoencoder_tiny_cuda.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cuda")
    for _, param in reloaded.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    with forge.no_grad():
        post_load = reloaded(query).to("cpu").numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-5)


def test_cpu_and_cuda_prediction_parity_after_training():
    """Same seed/data trained identically on CPU vs. CUDA should predict closely (parity check)."""
    train_ds = _make_reconstruction_dataset(128, seed=1)
    query = Tensor(np.random.default_rng(99).random((8, 1, 28, 28)).astype(np.float32))

    def run(device: str) -> np.ndarray:
        forge.random.seed(1)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(1))
        model = build_model(latent_dim=8).to(device)
        optimizer = Adam(model.parameters(), lr=5e-3)
        trainer = Trainer(model, MSELoss(), optimizer, device=device, verbose=False)
        trainer.fit(loader, epochs=1)
        with forge.no_grad():
            return model(query.to(device)).to("cpu").numpy()

    cpu_pred = run("cpu")
    cuda_pred = run("cuda")
    np.testing.assert_allclose(cuda_pred, cpu_pred, rtol=1e-3, atol=1e-3)
