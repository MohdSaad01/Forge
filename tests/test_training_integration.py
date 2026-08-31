"""Milestone 4 integration tests: forward -> loss -> backward -> optimizer.step().

Covers the full training sequence documented in
`docs/architecture/optimization.md`, and a deterministic synthetic learning
experiment demonstrating meaningful loss reduction (not a fragile exact
final-loss threshold).
"""

from __future__ import annotations

import numpy as np

import forge
from forge import Tensor
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD


class MLP(Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# -- forward -> loss -> backward -> step actually changes parameters --------


def test_single_training_step_changes_all_parameters():
    forge.random.seed(0)
    model = Linear(3, 2)
    x = Tensor(np.array([[1.0, -1.0, 2.0], [0.5, 0.5, -0.5]]))
    target = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))

    before = {name: p.numpy().copy() for name, p in model.named_parameters()}

    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=0.1)

    opt.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, target)
    loss.backward()
    opt.step()

    for name, param in model.named_parameters():
        assert not np.allclose(param.numpy(), before[name]), f"{name} did not change"


def test_zero_grad_before_backward_prevents_accumulation_across_steps():
    forge.random.seed(1)
    model = Linear(2, 1)
    opt = SGD(model.parameters(), lr=0.01)
    x = Tensor([[1.0, 1.0]])
    target = Tensor([[0.0]])
    loss_fn = MSELoss()

    opt.zero_grad()
    loss1 = loss_fn(model(x), target)
    loss1.backward()
    grad_first_step = model.weight.grad.numpy().copy()

    opt.step()
    opt.zero_grad()
    assert model.weight.grad is None

    loss2 = loss_fn(model(x), target)
    loss2.backward()
    grad_second_step = model.weight.grad.numpy().copy()

    # Without zero_grad, the second backward would add onto the first
    # gradient; since we cleared it, the second gradient stands alone and
    # need not equal double the first.
    assert grad_second_step.shape == grad_first_step.shape


def test_all_output_and_intermediate_values_stay_finite():
    forge.random.seed(2)
    model = MLP(4, 6, 2)
    x = Tensor(np.random.default_rng(3).normal(size=(5, 4)))
    target = Tensor(np.zeros((5, 2)))
    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=0.05)

    for _ in range(10):
        opt.zero_grad()
        loss = loss_fn(model(x), target)
        assert np.isfinite(loss.numpy())
        loss.backward()
        opt.step()
        for param in model.parameters():
            assert np.all(np.isfinite(param.numpy()))


# -- deterministic regression learning experiment ----------------------------


def test_linear_regression_loss_decreases_and_recovers_target_function():
    """Linear -> MSELoss -> SGD learns y = 3*x1 - 2*x2 + 1 from synthetic data."""
    forge.random.seed(42)
    data_rng = np.random.default_rng(42)

    X = data_rng.uniform(-1, 1, size=(32, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1
    x_t = Tensor(X)
    y_t = Tensor(y.reshape(-1, 1))

    model = Linear(2, 1)
    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=0.1)

    initial_weight = model.weight.numpy().copy()
    initial_bias = model.bias.numpy().copy()

    losses = []
    for _ in range(200):
        opt.zero_grad()
        prediction = model(x_t)
        loss = loss_fn(prediction, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.numpy()))

    assert all(np.isfinite(l) for l in losses)

    # Meaningful improvement from the initial state, not a fragile exact
    # threshold: final loss is at least two orders of magnitude smaller.
    assert losses[-1] < losses[0] * 1e-2

    assert not np.allclose(model.weight.numpy(), initial_weight)
    assert not np.allclose(model.bias.numpy(), initial_bias)

    # The learned function is close to the true generating function.
    np.testing.assert_allclose(model.weight.numpy().ravel(), [3.0, -2.0], atol=0.05)
    np.testing.assert_allclose(model.bias.numpy(), [1.0], atol=0.05)


def test_classification_mlp_loss_decreases_and_separates_classes():
    """Linear -> ReLU -> Linear -> CrossEntropyLoss -> SGD on a linearly separable set."""
    forge.random.seed(7)
    data_rng = np.random.default_rng(7)

    n = 20
    class0 = data_rng.normal(loc=[-1.5, -1.5], scale=0.3, size=(n, 2))
    class1 = data_rng.normal(loc=[1.5, 1.5], scale=0.3, size=(n, 2))
    X = np.vstack([class0, class1])
    targets = np.array([0] * n + [1] * n)
    x_t = Tensor(X)

    model = MLP(2, 8, 2)
    loss_fn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.5)

    losses = []
    for _ in range(100):
        opt.zero_grad()
        logits = model(x_t)
        loss = loss_fn(logits, targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.numpy()))

    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < losses[0] * 0.5

    predictions = np.argmax(model(x_t).numpy(), axis=1)
    accuracy = float(np.mean(predictions == targets))
    assert accuracy >= 0.9
