"""Milestone 5 integration test: Dataset -> DataLoader -> model-compatible batches.

Verifies TensorDataset/DataLoader batches feed directly into the existing
Linear + MSELoss + backward() stack established in M1-M4, with no Trainer
orchestration -- just enough of a loop to prove batch compatibility.
"""

from __future__ import annotations

import numpy as np

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.data.transforms import Normalize
from forge.nn import Linear
from forge.nn.loss import MSELoss
from forge.optim import SGD


def test_dataset_dataloader_feeds_linear_mse_backward():
    forge.random.seed(0)
    data_rng = np.random.default_rng(0)

    X = data_rng.uniform(-1, 1, size=(16, 3))
    y = (2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)

    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=np.random.default_rng(1))

    model = Linear(3, 1)
    loss_fn = MSELoss()

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 3)
    assert batch_y.shape == (4, 1)

    prediction = model(batch_x)
    loss = loss_fn(prediction, batch_y)
    loss.backward()

    assert model.weight.grad is not None
    assert model.weight.grad.shape == model.weight.shape
    assert model.bias.grad is not None


def test_full_epoch_training_loop_reduces_loss():
    """A hand-written loop (no Trainer) over DataLoader batches, proving the
    Dataset -> DataLoader -> Linear -> MSELoss -> SGD sequence works end to end."""
    forge.random.seed(1)
    data_rng = np.random.default_rng(2)

    X = data_rng.uniform(-1, 1, size=(64, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1.0).reshape(-1, 1)

    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(3))

    model = Linear(2, 1)
    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=0.3)

    def epoch_loss() -> float:
        total, count = 0.0, 0
        for bx, by in loader:
            pred = model(bx)
            l = loss_fn(pred, by)
            total += float(l.numpy()) * bx.shape[0]
            count += bx.shape[0]
        return total / count

    first_epoch_loss = epoch_loss()

    for _ in range(30):
        for bx, by in loader:
            opt.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()

    last_epoch_loss = epoch_loss()
    assert last_epoch_loss < first_epoch_loss * 0.1


def test_transform_pipeline_feeds_model_compatible_batches():
    forge.random.seed(5)
    X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
    y = np.array([[1.0], [0.0], [1.0], [0.0]])

    dataset = TensorDataset(
        Tensor(X), Tensor(y), transform=Normalize(mean=[40.0, 50.0], std=[20.0, 20.0])
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = Linear(2, 1)
    for bx, by in loader:
        assert bx.shape == (2, 2)
        pred = model(bx)
        assert pred.shape == (2, 1)
