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
from forge.nn import Linear, Module, ReLU, RNNCell
from forge.training import generate_sequence, predict

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


# -- generate_sequence() (Milestone 75) --------------------------------------


class TinyStepModel(Module):
    """Mirrors `tests/test_inference.py::TinyStepModel` -- see that file's docstring."""

    def __init__(self, vocab_size=5, hidden_size=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.cell = RNNCell(vocab_size, hidden_size)
        self.output = Linear(hidden_size, vocab_size)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.output(h), h

    def init_hidden(self, batch_size, device="cpu"):
        return self.cell.init_hidden(batch_size, device=device)


def _one_hot(index: int, size: int) -> np.ndarray:
    row = np.zeros((1, size), dtype=np.float32)
    row[0, index] = 1.0
    return row


def test_generate_sequence_runs_on_cuda_model_with_cpu_built_encode_inputs():
    forge.random.seed(3)
    model = TinyStepModel().to("cuda")
    # `encode` builds a plain CPU Tensor (as every real Forge example does)
    # -- generate_sequence() must move it to the model's CUDA device itself,
    # the same automatic-transfer contract `predict()` already guarantees.
    result = generate_sequence(
        model,
        seed=[0, 1],
        encode=lambda t: Tensor(_one_hot(t, model.vocab_size)),
        decode=lambda idx: idx,
        length=10,
        rng=np.random.default_rng(0),
    )
    assert result[:2] == [0, 1]
    assert len(result) == 12
    assert all(0 <= t < model.vocab_size for t in result)


def test_generate_sequence_cpu_and_cuda_agree_given_the_same_rng_and_weights():
    forge.random.seed(4)
    model = TinyStepModel()
    # `Module.to()` moves the same Parameters in place (Milestone 9) -- run
    # once on CPU, move the identical model to CUDA, run again, so both
    # results come from exactly the same weights, isolating device as the
    # only variable (the same pattern `test_predict_moves_cpu_input_to_cuda_
    # model_automatically` above uses).
    kwargs = dict(seed=[2], decode=lambda idx: idx, length=15)
    cpu_result = generate_sequence(
        model, encode=lambda t: Tensor(_one_hot(t, model.vocab_size)), rng=np.random.default_rng(7), **kwargs
    )
    model.to("cuda")
    cuda_result = generate_sequence(
        model, encode=lambda t: Tensor(_one_hot(t, model.vocab_size)), rng=np.random.default_rng(7), **kwargs
    )
    assert cpu_result == cuda_result
