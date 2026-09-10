"""Milestone 73 CUDA test: `start_training_session()`/`TrainingSession` on the
real CUDA backend.

This is orchestration around `Trainer`/`load_checkpoint()`/a plain
`numpy.random.Generator` -- it introduces no new CUDA kernel and no new
device-dispatch logic of its own (`device=` is passed straight through to
`Trainer(...)` and `load_checkpoint(..., device=...)`, both already
CUDA-tested elsewhere). This file is the minimal hardware-verified proof that
the fresh-session and resume-equivalence paths work end-to-end on `device=
"cuda"`, matching this repo's convention of gating real-hardware tests behind
`is_cuda_available()` (`tests/test_trainer_cuda.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import MSELoss
from forge.optim import Adam
from forge.serialization import register_module
from forge.training import start_training_session

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


class _CudaMLP(Module):
    def __init__(self, in_features=2, hidden=4, out_features=1):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


register_module(
    "_CudaMLP_M73_session_test",
    _CudaMLP,
    get_config=lambda m: {
        "in_features": m.fc1.in_features,
        "hidden": m.fc1.out_features,
        "out_features": m.fc2.out_features,
    },
)


def _build_model():
    return _CudaMLP()


def _build_optimizer(params):
    return Adam(params, lr=1e-2)


def _dataset(n=64, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    return TensorDataset(Tensor(X), Tensor(y))


def _params(model) -> "dict[str, np.ndarray]":
    return {name: p.to("cpu").numpy().copy() for name, p in model.named_parameters()}


def test_fresh_cuda_session_trains_and_reports_cuda_device():
    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=0, device="cuda"
    )
    assert str(session.trainer.device) == "cuda"
    assert str(session.trainer.model.device) == "cuda"

    loader = DataLoader(_dataset(), batch_size=8, shuffle=False)
    history = session.trainer.fit(loader, epochs=1)
    assert history[0].device == "cuda"


def test_cuda_resume_equivalence_matches_continuous_training(tmp_path):
    """The same resume-equivalence guarantee as
    tests/test_training_session.py::test_resume_equivalence_full_training_matches_partial_then_resume_with_shuffle,
    hardware-verified on real CUDA -- since Trainer's device transfer and
    load_checkpoint's device placement are both real CUDA paths, not just
    CPU-tested logic reused unmodified.
    """
    dataset = _dataset(n=64, seed=2)

    full_session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=21, device="cuda"
    )
    full_loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=full_session.data_loader_rng)
    full_session.trainer.fit(full_loader, epochs=4)

    ckpt_path = tmp_path / "cuda_partial.forge"
    partial_session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=21, device="cuda"
    )
    partial_loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=partial_session.data_loader_rng)
    partial_session.trainer.fit(partial_loader, epochs=2)
    partial_session.save_checkpoint(str(ckpt_path))

    resumed_session = start_training_session(
        build_model=_build_model,
        build_optimizer=_build_optimizer,
        loss_fn=MSELoss(),
        seed=21,
        device="cuda",
        resume=str(ckpt_path),
    )
    resumed_loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=resumed_session.data_loader_rng)
    resumed_session.trainer.fit(resumed_loader, epochs=2)

    full_params = _params(full_session.trainer.model)
    resumed_params = _params(resumed_session.trainer.model)
    for name in full_params:
        np.testing.assert_allclose(full_params[name], resumed_params[name], atol=1e-5)
