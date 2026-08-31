import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import Linear

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = fn(x)
        x[idx] = orig - eps
        minus = fn(x)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def fixed_linear(in_features=3, out_features=2, bias=True):
    """A Linear layer with deterministic, hand-chosen (non-random) parameters."""
    layer = Linear(in_features, out_features, bias=bias)
    weight = np.arange(in_features * out_features, dtype=np.float64).reshape(
        in_features, out_features
    ) * 0.1
    layer.weight = type(layer.weight)(weight)
    if bias:
        layer.bias = type(layer.bias)(np.arange(out_features, dtype=np.float64) * 0.1 + 1.0)
    return layer


# -- shapes ---------------------------------------------------------------


def test_linear_parameter_shapes():
    layer = Linear(3, 5)
    assert layer.weight.shape == (3, 5)
    assert layer.bias.shape == (5,)


def test_linear_without_bias_has_no_bias_parameter():
    layer = Linear(3, 5, bias=False)
    assert layer.bias is None
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"weight"}


def test_linear_output_shape_single_sample():
    layer = Linear(4, 3)
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = layer(x)
    assert y.shape == (3,)


def test_linear_output_shape_batched():
    layer = Linear(4, 3)
    x = Tensor(np.zeros((10, 4)))
    y = layer(x)
    assert y.shape == (10, 3)


# -- forward correctness ----------------------------------------------------


def test_linear_forward_matches_manual_matmul():
    layer = fixed_linear(3, 2)
    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = Tensor(x_data)
    y = layer(x)
    expected = x_data @ layer.weight.numpy() + layer.bias.numpy()
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


def test_linear_forward_without_bias():
    layer = fixed_linear(3, 2, bias=False)
    x_data = np.array([1.0, 2.0, 3.0])
    x = Tensor(x_data)
    y = layer(x)
    expected = x_data @ layer.weight.numpy()
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


# -- invalid input ----------------------------------------------------------


def test_linear_rejects_wrong_input_feature_dim():
    layer = Linear(4, 3)
    x = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ShapeMismatchError):
        layer(x)


def test_linear_rejects_higher_rank_input():
    layer = Linear(4, 3)
    x = Tensor(np.zeros((2, 2, 4)))
    with pytest.raises(ShapeMismatchError):
        layer(x)


def test_linear_rejects_non_positive_dimensions():
    with pytest.raises(ShapeMismatchError):
        Linear(0, 4)
    with pytest.raises(ShapeMismatchError):
        Linear(4, -1)


# -- backward / gradients ----------------------------------------------------


def test_linear_backward_produces_weight_and_bias_gradients():
    layer = fixed_linear(3, 2)
    x = Tensor([1.0, 2.0, 3.0])
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_linear_weight_gradient_matches_numerical_gradient():
    layer = fixed_linear(3, 2)
    x_data = np.array([1.0, 2.0, 3.0])
    weight_data = layer.weight.numpy().copy()
    bias_data = layer.bias.numpy().copy()

    def fn(w):
        return float(np.sum(x_data @ w + bias_data))

    x = Tensor(x_data)
    y = layer(x).sum()
    y.backward()

    expected = numerical_grad(fn, weight_data.copy())
    np.testing.assert_allclose(layer.weight.grad.numpy(), expected, **FD_TOL)


def test_linear_bias_gradient_is_ones_when_summed():
    layer = fixed_linear(3, 2)
    x = Tensor([1.0, 2.0, 3.0])
    y = layer(x).sum()
    y.backward()
    np.testing.assert_allclose(layer.bias.grad.numpy(), [1.0, 1.0], **TOL)


def test_linear_input_gradient_flows_through():
    layer = fixed_linear(3, 2)
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = layer(x).sum()
    y.backward()
    expected = layer.weight.numpy().sum(axis=1)
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)
