"""`Tensor.sqrt()` / `Tensor.__truediv__` (Milestone 53): CPU forward/backward correctness.

Added for `nn.BatchNorm2d`'s `std = sqrt(var + eps)` / `xhat = diff / std` --
see `docs/development/m53-batchnorm.md`. Mirrors `tests/test_tanh.py`'s shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor

TOL = dict(rtol=1e-6, atol=1e-6)


# -- sqrt ------------------------------------------------------------------


def test_sqrt_forward_matches_numpy():
    x = Tensor([1.0, 4.0, 9.0, 16.0])
    result = x.sqrt()
    np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0, 4.0], **TOL)


def test_sqrt_result_is_leaf_when_input_does_not_require_grad():
    y = Tensor([4.0]).sqrt()
    assert y.requires_grad is False
    assert y.grad_fn is None


def test_sqrt_backward_matches_analytic_derivative():
    x = Tensor([1.0, 4.0, 9.0], requires_grad=True)
    x.sqrt().sum().backward()
    expected = 0.5 / np.sqrt([1.0, 4.0, 9.0])
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


def test_sqrt_backward_matches_finite_difference():
    eps = 1e-6
    x0 = np.array([1.3, 4.2, 9.9])
    x = Tensor(x0, dtype="float64", requires_grad=True)
    x.sqrt().sum().backward()
    analytic = x.grad.numpy()

    numeric = np.empty_like(x0)
    for i in range(len(x0)):
        plus, minus = x0.copy(), x0.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric[i] = (np.sqrt(plus).sum() - np.sqrt(minus).sum()) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)


def test_sqrt_backward_scales_with_upstream_gradient():
    x = Tensor([4.0, 9.0], requires_grad=True)
    y = x.sqrt()
    y.backward(Tensor([2.0, 3.0]))
    expected = np.array([2.0, 3.0]) * 0.5 / np.sqrt([4.0, 9.0])
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


# -- div (__truediv__) ------------------------------------------------------


def test_div_forward_matches_numpy():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([2.0, 4.0, 5.0])
    np.testing.assert_allclose((a / b).numpy(), [0.5, 0.5, 0.6], **TOL)


def test_div_by_python_scalar():
    a = Tensor([2.0, 4.0])
    np.testing.assert_allclose((a / 2.0).numpy(), [1.0, 2.0], **TOL)


def test_rdiv_python_scalar_by_tensor():
    a = Tensor([2.0, 4.0])
    np.testing.assert_allclose((2.0 / a).numpy(), [1.0, 0.5], **TOL)


def test_div_backward_matches_analytic_derivative():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 2.0, 2.0], requires_grad=True)
    (a / b).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), [0.5, 0.5, 0.5], **TOL)
    np.testing.assert_allclose(b.grad.numpy(), -np.array([1.0, 2.0, 3.0]) / 4.0, **TOL)


def test_div_backward_matches_finite_difference():
    eps = 1e-6
    a0 = np.array([1.3, -2.1, 0.7])
    b0 = np.array([2.2, 1.4, -3.3])

    def loss(a_arr, b_arr):
        return (a_arr / b_arr).sum()

    a = Tensor(a0, dtype="float64", requires_grad=True)
    b = Tensor(b0, dtype="float64", requires_grad=True)
    (a / b).sum().backward()

    for i in range(len(a0)):
        ap, am = a0.copy(), a0.copy()
        ap[i] += eps
        am[i] -= eps
        numeric = (loss(ap, b0) - loss(am, b0)) / (2 * eps)
        np.testing.assert_allclose(a.grad.numpy()[i], numeric, rtol=1e-5, atol=1e-5)

    for i in range(len(b0)):
        bp, bm = b0.copy(), b0.copy()
        bp[i] += eps
        bm[i] -= eps
        numeric = (loss(a0, bp) - loss(a0, bm)) / (2 * eps)
        np.testing.assert_allclose(b.grad.numpy()[i], numeric, rtol=1e-5, atol=1e-5)


def test_div_broadcasts_on_cpu():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([2.0, 4.0])
    np.testing.assert_allclose((a / b).numpy(), [[0.5, 0.5], [1.5, 1.0]], **TOL)


def test_div_broadcast_backward_reduces_to_operand_shape():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([2.0, 4.0], requires_grad=True)
    (a / b).sum().backward()
    assert a.grad.shape == (2, 2)
    assert b.grad.shape == (2,)


def test_div_device_mismatch_raises():
    from forge.backend.cuda import is_cuda_available
    from forge.exceptions import UnsupportedDeviceError

    if not is_cuda_available():
        pytest.skip("CUDA is not available on this machine")
    a = Tensor([1.0, 2.0])
    b = Tensor([1.0, 2.0]).to("cuda")
    with pytest.raises(UnsupportedDeviceError):
        _ = a / b
