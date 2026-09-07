"""Milestone 53 integration test: `examples.mnist.model.build_model_bn()` on CUDA.

Mirrors `tests/test_mnist_bn_example_integration.py` (same tiny synthetic
`(N, 1, 28, 28)` dataset) but drives `Trainer(..., device="cuda")`, proving
the CUDA fused BatchNorm2d kernel (`Tensor.batch_norm2d`,
`CUDABackend.batch_norm2d{,_backward}`) trains correctly end-to-end and that
its buffers stay CUDA-resident through persistence. Skips cleanly when CUDA
is unavailable; hardware-verified on the development machine's GeForce 940MX
(CC 5.0) per `docs/development/development-environment.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_model, save_model
from forge.training import Accuracy, Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.mnist.model import build_model_bn  # noqa: E402  (path setup above)

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


def test_full_pipeline_with_batchnorm_trains_and_learns_on_cuda():
    forge.random.seed(0)
    train_ds = _make_dataset(80, seed=1)
    test_ds = _make_dataset(40, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(3))
    test_loader = DataLoader(test_ds, batch_size=16)

    model = build_model_bn().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", metrics=[Accuracy()], verbose=False)

    history = trainer.fit(train_loader, epochs=8, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    assert eval_result.metrics["accuracy"] >= 0.8


def test_batchnorm_buffers_and_gradients_are_cuda_resident():
    forge.random.seed(5)
    loader = DataLoader(_make_dataset(32, seed=6), batch_size=8, shuffle=False)

    model = build_model_bn().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=1)

    bn = model._modules["1"]
    assert isinstance(bn.running_mean._data, CUDAStorage)
    assert isinstance(bn.running_var._data, CUDAStorage)
    assert isinstance(bn.weight._data, CUDAStorage)
    assert bn.weight.grad is not None
    assert isinstance(bn.weight.grad._data, CUDAStorage)


def test_train_eval_mode_difference_holds_on_cuda():
    forge.random.seed(4)
    train_loader = DataLoader(_make_dataset(64, seed=5), batch_size=16, shuffle=True, generator=np.random.default_rng(6))
    model = build_model_bn().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(train_loader, epochs=3)

    x = Tensor(np.random.default_rng(7).standard_normal((8, 1, 28, 28)).astype(np.float32), device="cuda")

    model.train()
    with no_grad():
        train_mode_out = model(x).to("cpu").numpy()

    model.eval()
    with no_grad():
        eval_mode_out = model(x).to("cpu").numpy()

    assert not np.allclose(train_mode_out, eval_mode_out)


def test_bn_model_persistence_preserves_predictions_on_cuda(tmp_path):
    forge.random.seed(8)
    train_loader = DataLoader(_make_dataset(48, seed=9), batch_size=16, shuffle=True, generator=np.random.default_rng(10))

    model = build_model_bn().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(train_loader, epochs=3)
    model.eval()

    query = Tensor(np.random.default_rng(11).standard_normal((4, 1, 28, 28)).astype(np.float32), device="cuda")
    with no_grad():
        pre_save = model(query).to("cpu").numpy()

    model_path = tmp_path / "mnist_bn_tiny_cuda.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cuda")
    assert reloaded.training is False
    assert isinstance(reloaded._modules["1"].running_mean._data, CUDAStorage)

    with no_grad():
        post_load = reloaded(query).to("cpu").numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-5)
