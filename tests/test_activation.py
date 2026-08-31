import numpy as np

from forge import Tensor
from forge.nn import ReLU

TOL = dict(rtol=1e-6, atol=1e-6)


def test_relu_positive_values_pass_through():
    x = Tensor([1.0, 2.5, 100.0])
    y = ReLU()(x)
    np.testing.assert_allclose(y.numpy(), [1.0, 2.5, 100.0], **TOL)


def test_relu_negative_values_become_zero():
    x = Tensor([-1.0, -0.5, -100.0])
    y = ReLU()(x)
    np.testing.assert_allclose(y.numpy(), [0.0, 0.0, 0.0], **TOL)


def test_relu_mixed_values():
    x = Tensor([-2.0, 0.0, 3.0])
    y = ReLU()(x)
    np.testing.assert_allclose(y.numpy(), [0.0, 0.0, 3.0], **TOL)


def test_relu_is_a_module_without_parameters():
    relu = ReLU()
    assert list(relu.parameters()) == []


# -- backward ---------------------------------------------------------------


def test_relu_backward_zeroes_gradient_for_negative_inputs():
    x = Tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    y = ReLU()(x).sum()
    y.backward()
    np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0, 0.0, 1.0], **TOL)


def test_relu_backward_at_zero_is_zero():
    x = Tensor([0.0], requires_grad=True)
    y = ReLU()(x).sum()
    y.backward()
    np.testing.assert_allclose(x.grad.numpy(), [0.0], **TOL)


def test_relu_backward_scales_with_upstream_gradient():
    x = Tensor([1.0, -1.0], requires_grad=True)
    y = ReLU()(x)
    y.backward(Tensor([5.0, 5.0]))
    np.testing.assert_allclose(x.grad.numpy(), [5.0, 0.0], **TOL)


def test_relu_result_is_leaf_when_input_does_not_require_grad():
    x = Tensor([1.0, -1.0])
    y = ReLU()(x)
    assert y.requires_grad is False
    assert y.grad_fn is None
