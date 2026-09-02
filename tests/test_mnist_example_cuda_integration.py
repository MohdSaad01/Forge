"""Milestone 20 integration tests: the `examples/mnist` pipeline on CUDA.

Mirrors `tests/test_mnist_example_integration.py` (same tiny synthetic
`(N, 1, 28, 28)` dataset, no real MNIST download) but drives `Trainer(...,
device="cuda")`, additionally verifying CUDA residency of parameters,
gradients, and Adam optimizer state -- the acceptance criteria specific to
Milestone 20's CUDA path. Skips cleanly when CUDA is unavailable; hardware-
verified on the development machine's GeForce 940MX (CC 5.0) per
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
from forge.data import DataLoader, TensorDataset
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Accuracy, Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.mnist.model import build_model  # noqa: E402  (path setup above)

_NUM_CLASSES = 10


def _make_synthetic_mnist(n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, 1, 28, 28)) * 0.05).astype(np.float32)
    Y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        label = i % _NUM_CLASSES
        Y[i] = label
        col = label * 2
        stripe = 1.0 + rng.standard_normal((28, 2)).astype(np.float32) * 0.05
        X[i, 0, :, col : col + 2] = stripe
    return X, Y


def _make_dataset(n: int, seed: int) -> TensorDataset:
    X, Y = _make_synthetic_mnist(n, seed)
    return TensorDataset(Tensor(X), Tensor(Y))


def test_full_pipeline_trains_and_learns_on_cuda():
    forge.random.seed(0)
    train_ds = _make_dataset(80, seed=1)
    test_ds = _make_dataset(40, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(3))
    test_loader = DataLoader(test_ds, batch_size=16)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", metrics=[Accuracy()], verbose=False)

    history = trainer.fit(train_loader, epochs=8, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    assert eval_result.metrics["accuracy"] >= 0.8


def test_cuda_residency_of_parameters_gradients_and_adam_state():
    forge.random.seed(5)
    loader = DataLoader(_make_dataset(32, seed=6), batch_size=8, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
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
    loader = DataLoader(_make_dataset(48, seed=11), batch_size=16, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    checkpoint_path = tmp_path / "mnist_tiny_cuda.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path), device="cuda")
    assert checkpoint.model.device.type == "cuda"
    for _, param in checkpoint.model.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    resumed_trainer = Trainer(checkpoint.model, CrossEntropyLoss(), checkpoint.optimizer, device="cuda", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.model.device.type == "cuda"


def test_model_persistence_preserves_predictions_on_cuda(tmp_path):
    forge.random.seed(20)
    loader = DataLoader(_make_dataset(32, seed=21), batch_size=8, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    query = Tensor(np.random.default_rng(22).standard_normal((4, 1, 28, 28)).astype(np.float32), device="cuda")
    with forge.no_grad():
        pre_save = model(query).to("cpu").numpy()

    model_path = tmp_path / "mnist_tiny_cuda.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cuda")
    for _, param in reloaded.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    with forge.no_grad():
        post_load = reloaded(query).to("cpu").numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-5)
