"""Milestone 16 acceptance test: `Sequential` + `Flatten` + `Dropout` CNN, end-to-end.

Builds

```text
Sequential(
    Conv2d(1, 4, 3), ReLU(), MaxPool2d(2), Flatten(),
    Linear(36, 8), ReLU(), Dropout(0.2), Linear(8, 2),
)
```

through the existing `Dataset -> DataLoader -> Trainer -> ... -> SGD` flow,
unmodified -- the same deterministic "bright top half / bright bottom half"
two-class image dataset `tests/test_conv_trainer_integration.py` uses for
the Milestone 15 acceptance test. No accuracy threshold is asserted (per the
milestone brief) -- only that training loss falls substantially, proving the
composed model (including a stochastic `Dropout` layer) actually learns.
The CUDA variant is skipped cleanly when CUDA is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.nn.loss import CrossEntropyLoss
from forge.optim import SGD
from forge.training import Accuracy, Trainer


def _make_dataset(n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, 8, 8), dtype=np.float32)
    Y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        label = i % 2
        Y[i] = label
        if label == 0:
            X[i, 0, :4, :] = 1.0 + rng.standard_normal((4, 8)).astype(np.float32) * 0.05
            X[i, 0, 4:, :] = rng.standard_normal((4, 8)).astype(np.float32) * 0.05
        else:
            X[i, 0, 4:, :] = 1.0 + rng.standard_normal((4, 8)).astype(np.float32) * 0.05
            X[i, 0, :4, :] = rng.standard_normal((4, 8)).astype(np.float32) * 0.05
    return X, Y


def _build_model() -> Sequential:
    # Conv2d(k=3,s=1,p=0) on 8x8 -> 6x6; MaxPool2d(2) -> 3x3 -> Flatten -> 4*3*3=36.
    return Sequential(
        Conv2d(1, 4, kernel_size=3),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(4 * 3 * 3, 8),
        ReLU(),
        Dropout(0.2),
        Linear(8, 2),
    )


def test_sequential_cnn_forward_shape_matches_flatten_arithmetic():
    model = _build_model()
    x = Tensor(np.zeros((5, 1, 8, 8), dtype=np.float32))
    y = model(x)
    assert y.shape == (5, 2)


def test_sequential_cnn_all_parameters_receive_gradients():
    forge.random.seed(1)
    X, Y = _make_dataset(n=16, seed=2)
    model = _build_model()
    loss_fn = CrossEntropyLoss()

    pred = model(Tensor(X))
    loss = loss_fn(pred, Y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} did not receive a gradient"
        assert param.grad.shape == param.shape


def test_sequential_cnn_trains_via_trainer_on_cpu():
    forge.random.seed(0)
    X, Y = _make_dataset(n=40, seed=1)

    dataset = TensorDataset(Tensor(X), Tensor(Y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = _build_model()
    loss_fn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.2)
    trainer = Trainer(model, loss_fn, opt, device="cpu", metrics=[Accuracy()], verbose=False)

    history = trainer.fit(loader, epochs=20)

    first_loss = history[0].train_loss
    last_loss = history[-1].train_loss
    assert last_loss < first_loss * 0.6

    eval_result = trainer.evaluate(loader)
    assert eval_result.metrics["accuracy"] >= 0.85


def test_sequential_cnn_dropout_is_inactive_during_trainer_evaluate():
    """`Trainer.evaluate()` switches the model to eval mode -- the nested
    `Dropout` layer must behave deterministically (identity) there."""
    forge.random.seed(3)
    X, Y = _make_dataset(n=16, seed=4)
    dataset = TensorDataset(Tensor(X), Tensor(Y))
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = _build_model()
    loss_fn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, loss_fn, opt, device="cpu", verbose=False)

    result1 = trainer.evaluate(loader)
    result2 = trainer.evaluate(loader)
    assert result1.loss == pytest.approx(result2.loss)
    assert model.training is True  # evaluate() restores training mode afterward


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")
def test_sequential_cnn_trains_via_trainer_on_cuda():
    forge.random.seed(0)
    X, Y = _make_dataset(n=40, seed=1)

    dataset = TensorDataset(Tensor(X), Tensor(Y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = _build_model().to("cuda")
    loss_fn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.2)
    trainer = Trainer(model, loss_fn, opt, device="cuda", metrics=[Accuracy()], verbose=False)

    history = trainer.fit(loader, epochs=20)

    first_loss = history[0].train_loss
    last_loss = history[-1].train_loss
    assert last_loss < first_loss * 0.6

    eval_result = trainer.evaluate(loader)
    assert eval_result.metrics["accuracy"] >= 0.85
