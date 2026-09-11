"""Milestone 79 CUDA test: `forge.train()` on the real CUDA backend.

This is orchestration around `DataLoader`/`Trainer` -- it introduces no new
CUDA kernel and no new device-dispatch logic of its own (`device=` resolves
via `Device.parse`/`Module.to()`, both already CUDA-tested elsewhere, then
delegates straight to `Trainer(device=...)`). This file is the minimal
hardware-verified proof that `device="cuda"` actually moves a CPU-built model
and trains it for real on the GPU, matching this repo's convention of gating
real-hardware tests behind `is_cuda_available()`
(`tests/test_training_session_cuda.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.data import TensorDataset
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import MSELoss
from forge.optim import Adam
from forge.training import train

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


class _CudaMLP(Module):
    def __init__(self, in_features=2, hidden=4, out_features=1):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _dataset(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    return TensorDataset(Tensor(X), Tensor(y))


def _params(model) -> "dict[str, np.ndarray]":
    return {name: p.to("cpu").numpy().copy() for name, p in model.named_parameters()}


def test_train_moves_a_cpu_built_model_to_cuda_and_trains():
    model = _CudaMLP()  # constructed on CPU by default, like every real example
    assert str(model.device) == "cpu"
    optimizer = Adam(model.parameters(), lr=0.05)  # built before the device move

    history = train(
        model, _dataset(), loss=MSELoss(), optimizer=optimizer,
        epochs=1, device="cuda", verbose=False,
    )

    assert str(model.device) == "cuda"
    assert history[0].device == "cuda"


def test_train_on_cuda_actually_updates_parameters():
    model = _CudaMLP()
    optimizer = Adam(model.parameters(), lr=0.1)
    before = _params(model)

    train(
        model, _dataset(n=64), loss=MSELoss(), optimizer=optimizer,
        epochs=5, device="cuda", verbose=False,
    )

    after = _params(model)
    assert any(not np.array_equal(before[name], after[name]) for name in before)


def test_train_on_cuda_reports_validation():
    model = _CudaMLP()
    optimizer = Adam(model.parameters(), lr=0.05)
    history = train(
        model, _dataset(n=32, seed=0), loss=MSELoss(), optimizer=optimizer,
        epochs=2, device="cuda", validation_dataset=_dataset(n=16, seed=1), verbose=False,
    )
    assert history[-1].val_loss is not None
