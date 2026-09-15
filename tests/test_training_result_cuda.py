"""Milestone 99 CUDA test: `TrainingResult`/`TrainAndSaveResult.artifact_path`
on the real CUDA backend. See `forge/training/api.py`.

`TrainingResult` introduces no new CUDA kernel or device-dispatch logic of
its own -- it is a plain Python wrapper constructed after `Trainer.fit()`
returns. This file is the minimal hardware-verified proof that the result
object built on top of a CUDA-trained model is semantically consistent with
the same workload trained on CPU (final loss agrees within the same
tolerance every other CPU/CUDA numerical-consistency test in this repo uses
-- see `tests/test_cuda_consistency.py` -- never bit-for-bit, since
accumulation order differs across backends), and that `artifact_path`/
`model` identity hold on CUDA exactly as they do on CPU.
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
from forge.optim import SGD
from forge.serialization import register_module
from forge.training import TrainingHistory, TrainingResult, train, train_and_save

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-5)


class _CudaMLP(Module):
    def __init__(self, in_features=2, hidden=4, out_features=1):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


register_module("_CudaMLP_M99Test", _CudaMLP, get_config=lambda m: {
    "in_features": m.fc1.in_features, "hidden": m.fc1.out_features, "out_features": m.fc2.out_features,
})


def _dataset(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    return TensorDataset(Tensor(X), Tensor(y))


def _train_on(device: str) -> TrainingResult:
    forge.random.seed(0)
    model = _CudaMLP()
    return train(
        model, _dataset(n=32), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1),
        epochs=5, batch_size=8, shuffle=False, device=device, verbose=False,
    )


def test_training_result_on_cuda_is_still_a_training_history():
    result = _train_on("cuda")
    assert isinstance(result, TrainingHistory)
    assert isinstance(result, TrainingResult)
    assert len(result) == 5


def test_training_result_on_cuda_exposes_the_trained_model_on_cuda():
    result = _train_on("cuda")
    assert str(result.model.device) == "cuda"
    assert result.final_train_loss == result[-1].train_loss


def test_training_result_final_train_loss_agrees_cpu_vs_cuda():
    """Same seed/architecture/data/batch order on CPU vs CUDA -- final
    training loss must agree within numerical tolerance, matching the
    established deterministic-workload convention (M98's own CPU/CUDA
    evaluation parity), not bit-for-bit (see module docstring)."""
    cpu_result = _train_on("cpu")
    cuda_result = _train_on("cuda")
    np.testing.assert_allclose(cuda_result.final_train_loss, cpu_result.final_train_loss, **TOL)


def test_train_and_save_result_artifact_path_on_cuda(tmp_path):
    forge.random.seed(0)
    model = _CudaMLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    path = str(tmp_path / "model.forge")

    result = train_and_save(
        model, _dataset(n=32), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1),
        epochs=2, device="cuda", verbose=False, path=path, sample=x,
    )

    assert result.artifact_path == path
    assert str(result.history.model.device) == "cuda"
    reloaded = forge.load_model(result.artifact_path)
    with forge.no_grad():
        expected = model(x.to("cuda")).to("cpu").numpy()
        actual = reloaded(x.to("cuda")).to("cpu").numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5)
