"""Milestone 53 integration test: `examples.mnist.model.build_model_bn()`.

Exercises `Conv2d -> BatchNorm2d -> ReLU -> ...` (M52's proposed validation
workload, `docs/development/m52-product-direction.md` Section 8) through the
same `DataLoader -> Trainer -> CrossEntropyLoss -> Adam` pipeline
`tests/test_mnist_example_integration.py` already validates for the
BatchNorm-free architecture -- against the same tiny synthetic `(N, 1, 28,
28)` stand-in dataset, for the same reason (no real MNIST download needed).
This file adds only what that one does not already cover: that inserting
BatchNorm2d doesn't break training, that train/eval-mode behavior actually
differs, and that BatchNorm's buffers round-trip through `save_model`/
`load_model` inside a real trained model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import forge
from forge import Tensor, no_grad
from forge.data import DataLoader, TensorDataset
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_model, save_model
from forge.training import Accuracy, Trainer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.mnist.model import build_model_bn  # noqa: E402  (path setup above)

_NUM_CLASSES = 10


def _make_synthetic_mnist(n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Same synthetic stand-in as `tests/test_mnist_example_integration.py`."""
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


def test_bn_model_forward_shape_matches_ten_class_output():
    model = build_model_bn()
    x = Tensor(np.zeros((5, 1, 28, 28), dtype=np.float32))
    y = model(x)
    assert y.shape == (5, _NUM_CLASSES)


def test_bn_model_has_a_batchnorm2d_child_with_expected_channels():
    from forge.nn import BatchNorm2d

    model = build_model_bn()
    bn = model._modules["1"]
    assert isinstance(bn, BatchNorm2d)
    assert bn.num_features == 8


def test_full_pipeline_with_batchnorm_trains_and_learns_on_cpu():
    forge.random.seed(0)
    train_ds = _make_dataset(80, seed=1)
    test_ds = _make_dataset(40, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(3))
    test_loader = DataLoader(test_ds, batch_size=16)

    model = build_model_bn()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", metrics=[Accuracy()], verbose=False)

    history = trainer.fit(train_loader, epochs=8, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    assert eval_result.metrics["accuracy"] >= 0.8


def test_train_mode_and_eval_mode_produce_different_outputs():
    """A trained model's train-mode output (batch statistics) should differ
    from its eval-mode output (running statistics) on the same input --
    Section 11's explicit train/eval behavioral-difference requirement."""
    forge.random.seed(4)
    train_loader = DataLoader(_make_dataset(64, seed=5), batch_size=16, shuffle=True, generator=np.random.default_rng(6))
    model = build_model_bn()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=3)

    x = Tensor(np.random.default_rng(7).standard_normal((8, 1, 28, 28)).astype(np.float32))

    model.train()
    with no_grad():
        train_mode_out = model(x).numpy()

    model.eval()
    with no_grad():
        eval_mode_out = model(x).numpy()

    assert not np.allclose(train_mode_out, eval_mode_out)


def test_bn_model_persistence_preserves_predictions_in_eval_mode(tmp_path):
    forge.random.seed(8)
    train_loader = DataLoader(_make_dataset(48, seed=9), batch_size=16, shuffle=True, generator=np.random.default_rng(10))

    model = build_model_bn()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=3)
    model.eval()

    query = Tensor(np.random.default_rng(11).standard_normal((4, 1, 28, 28)).astype(np.float32))
    with no_grad():
        pre_save = model(query).numpy()

    model_path = tmp_path / "mnist_bn_tiny.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path))
    assert reloaded.training is False

    with no_grad():
        post_load = reloaded(query).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)
