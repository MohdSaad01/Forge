"""Milestone 6 manual verification: Dataset -> DataLoader -> Trainer -> Model.

Run directly: `python examples/trainer_demo.py`

Demonstrates the full public training-engine workflow from
`docs/architecture/training-engine.md`, replacing the hand-written
```python
optimizer.zero_grad()
prediction = model(x)
loss = loss_fn(prediction, target)
loss.backward()
optimizer.step()
```
loop (still required in Milestones 1-5) with a single `trainer.fit(...)`
call -- covering dataset/loader construction, model/loss/optimizer/trainer
construction, multi-epoch training with progress output, and evaluation via
a held-out validation split, for both a regression and a classification
task.
"""

from __future__ import annotations

import numpy as np

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset, random_split
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD
from forge.training import Accuracy, MeanAbsoluteError, Trainer


class MLP(Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def regression_demo() -> None:
    print("=" * 70)
    print("Regression: Linear -> MSELoss -> SGD, y = 3*x1 - 2*x2 + 1")
    print("=" * 70)

    forge.random.seed(0)
    data_rng = np.random.default_rng(0)

    # 1. Dataset.
    n_samples = 200
    X = data_rng.uniform(-1, 1, size=(n_samples, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    train_ds, val_ds = random_split(dataset, [160, 40], generator=np.random.default_rng(1))

    # 2. DataLoaders.
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(2))
    val_loader = DataLoader(val_ds, batch_size=16)

    # 3-5. Model, loss, optimizer.
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    # 6. Trainer -- no manual zero_grad/backward/step loop below.
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        metrics=[MeanAbsoluteError()],
    )

    # 7-8. Multi-epoch training with progress output and a programmatic history.
    history = trainer.fit(train_loader, epochs=15, validation_loader=val_loader)

    print(f"\nFirst-epoch train loss: {history[0].train_loss:.4f}")
    print(f"Last-epoch train loss:  {history[-1].train_loss:.4f}")
    print(f"Last-epoch val loss:    {history[-1].val_loss:.4f}")
    print(f"Learned weight: {model.weight.numpy().ravel()} (true: [3.0, -2.0])")
    print(f"Learned bias:   {model.bias.numpy()} (true: [1.0])")

    # 9-10. Standalone evaluation and a meaningful-improvement check.
    final_eval = trainer.evaluate(val_loader)
    print(f"\nFinal held-out evaluation: loss={final_eval.loss:.4f}, mae={final_eval.metrics['mae']:.4f}")
    assert history[-1].train_loss < history[0].train_loss * 0.1, "expected meaningful loss reduction"


def classification_demo() -> None:
    print("\n" + "=" * 70)
    print("Classification: Linear -> ReLU -> Linear -> CrossEntropyLoss -> SGD")
    print("=" * 70)

    forge.random.seed(7)
    data_rng = np.random.default_rng(7)

    # 1. Dataset: two separable Gaussian blobs.
    n = 60
    class0 = data_rng.normal(loc=[-1.5, -1.5], scale=0.4, size=(n, 2))
    class1 = data_rng.normal(loc=[1.5, 1.5], scale=0.4, size=(n, 2))
    X = np.vstack([class0, class1])
    targets = np.array([0] * n + [1] * n)
    dataset = TensorDataset(Tensor(X), Tensor(targets))
    train_ds, val_ds = random_split(dataset, [100, 20], generator=np.random.default_rng(3))

    # 2. DataLoaders.
    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True, generator=np.random.default_rng(4))
    val_loader = DataLoader(val_ds, batch_size=20)

    # 3-5. Model, loss, optimizer.
    model = MLP(2, 8, 2)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.5)

    # 6-8. Trainer, multi-epoch training, progress + history.
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        metrics=[Accuracy()],
    )
    history = trainer.fit(train_loader, epochs=30, validation_loader=val_loader)

    print(f"\nFirst-epoch train accuracy: {history[0].train_metrics['accuracy']:.2%}")
    print(f"Last-epoch train accuracy:  {history[-1].train_metrics['accuracy']:.2%}")
    print(f"Last-epoch val accuracy:    {history[-1].val_metrics['accuracy']:.2%}")

    # 9-10. Standalone evaluation and a meaningful-improvement check.
    final_eval = trainer.evaluate(val_loader)
    print(f"\nFinal held-out evaluation: loss={final_eval.loss:.4f}, accuracy={final_eval.metrics['accuracy']:.2%}")
    assert final_eval.metrics["accuracy"] >= 0.85, "expected meaningful classification accuracy"


if __name__ == "__main__":
    regression_demo()
    classification_demo()
