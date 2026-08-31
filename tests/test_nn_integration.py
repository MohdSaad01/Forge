import numpy as np

from forge import Tensor
from forge.nn import Linear, Module, ReLU


class MLP(Module):
    """Linear -> ReLU -> Linear, the milestone's reference integration shape."""

    def __init__(self, in_features=4, hidden=6, out_features=2):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_mlp_forward_backward_all_parameters_receive_gradients():
    model = MLP(in_features=4, hidden=6, out_features=2)
    x = Tensor(np.array([[1.0, -2.0, 3.0, 0.5], [0.1, 0.2, -0.3, 0.4]]))

    output = model(x)
    assert output.shape == (2, 2)

    loss = output.sum()
    loss.backward()

    params = dict(model.named_parameters())
    assert set(params) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    for name, param in params.items():
        assert param.grad is not None, f"{name} did not receive a gradient"
        assert param.grad.shape == param.shape


def test_mlp_gradient_matches_numerical_gradient_for_fc1_weight():
    model = MLP(in_features=3, hidden=4, out_features=1)
    x_data = np.array([1.0, -0.5, 2.0])

    def forward_numpy(w1, b1, w2, b2):
        h = np.maximum(x_data @ w1 + b1, 0.0)
        return float(np.sum(h @ w2 + b2))

    w1 = model.fc1.weight.numpy().copy()
    b1 = model.fc1.bias.numpy().copy()
    w2 = model.fc2.weight.numpy().copy()
    b2 = model.fc2.bias.numpy().copy()

    x = Tensor(x_data)
    loss = model(x).sum()
    loss.backward()

    eps = 1e-4
    expected = np.zeros_like(w1)
    it = np.nditer(w1, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = w1[idx]
        w1[idx] = orig + eps
        plus = forward_numpy(w1, b1, w2, b2)
        w1[idx] = orig - eps
        minus = forward_numpy(w1, b1, w2, b2)
        w1[idx] = orig
        expected[idx] = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(model.fc1.weight.grad.numpy(), expected, rtol=1e-3, atol=1e-3)


def test_mlp_eval_mode_propagates_and_does_not_affect_forward_shape():
    model = MLP()
    model.eval()
    assert model.training is False
    assert model.fc1.training is False
    assert model.relu.training is False

    x = Tensor(np.ones((3, 4)))
    y = model(x)
    assert y.shape == (3, 2)
