"""Milestone 68 CUDA tests: `forge.predict()` on the real CUDA backend.

Mirrors `tests/test_inference.py`'s CPU coverage for the device-specific
behavior only: a CPU-resident input Tensor handed to a CUDA model is moved
automatically (no explicit `.to("cuda")` required from the caller), the
returned prediction always lands back on CPU, and CPU/CUDA predictions agree
within the project's established tolerance. See
`forge/training/inference.py` and `tests/test_cuda_consistency.py` for the
CPU/CUDA parity convention this file follows.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.backend.cuda import is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.nn import Linear, Module, ReLU
from forge.training import predict

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


class MLP(Module):
    def __init__(self, in_features=4, hidden=8, out_features=3):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _features(n=11, in_features=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, in_features)).astype(np.float32)


def test_predict_moves_cpu_input_to_cuda_model_automatically():
    forge.random.seed(0)
    model = MLP()
    x_np = _features(n=6)

    with no_grad():
        expected = model(Tensor(x_np)).numpy()

    model.to("cuda")  # same Parameter values, moved in place -- Milestone 9
    cpu_input = Tensor(x_np)  # deliberately left on CPU
    assert str(cpu_input.device) == "cpu"

    result = predict(model, cpu_input)
    assert str(result.device) == "cpu"
    np.testing.assert_allclose(result.numpy(), expected, **TOL)


def test_predict_over_cuda_loader_concatenates_correctly():
    forge.random.seed(1)
    model = MLP().to("cuda")
    x_np = _features(n=13)
    dataset = TensorDataset(Tensor(x_np))
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    result = predict(model, loader)
    assert result.shape == (13, 3)
    assert str(result.device) == "cpu"

    with no_grad():
        expected = model(Tensor(x_np, device="cuda")).to("cpu").numpy()
    np.testing.assert_allclose(result.numpy(), expected, **TOL)


def test_predict_device_defaults_to_cuda_model_device():
    forge.random.seed(2)
    model = MLP().to("cuda")
    result = predict(model, Tensor(_features(n=4)))
    # The output always comes back on CPU regardless of the model's device --
    # predict() infers the *compute* device from the model, not the return
    # device, which is always CPU for direct usability (.numpy(), display).
    assert str(result.device) == "cpu"
    assert str(model.device) == "cuda"
