import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import GradientStateError, ShapeMismatchError

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference gradient of a scalar-valued fn(np.ndarray) -> float."""
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


# -- requires_grad / leaf semantics -----------------------------------------


def test_requires_grad_defaults_false():
    x = Tensor([1.0, 2.0])
    assert x.requires_grad is False
    assert x.grad is None


def test_requires_grad_true_is_leaf():
    x = Tensor([1.0, 2.0], requires_grad=True)
    assert x.requires_grad is True
    assert x.is_leaf is True
    assert x.grad_fn is None


def test_requires_grad_rejects_non_float_dtype():
    with pytest.raises(GradientStateError):
        Tensor([1, 2, 3], dtype="int64", requires_grad=True)


def test_result_of_op_is_not_leaf():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x + x
    assert y.is_leaf is False
    assert y.requires_grad is True
    assert y.grad_fn is not None


def test_op_between_non_grad_tensors_does_not_require_grad():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    y = a + b
    assert y.requires_grad is False
    assert y.is_leaf is True
    assert y.grad_fn is None


def test_backward_on_non_grad_tensor_raises():
    x = Tensor([1.0, 2.0])
    with pytest.raises(GradientStateError):
        x.backward()


# -- basic gradients: add / sub / mul ---------------------------------------


def test_add_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    z = (x + y).sum()
    z.backward()
    np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0], **TOL)
    np.testing.assert_allclose(y.grad.numpy(), [1.0, 1.0, 1.0], **TOL)


def test_sub_backward():
    x = Tensor([5.0, 7.0, 9.0], requires_grad=True)
    y = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = (x - y).sum()
    z.backward()
    np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0], **TOL)
    np.testing.assert_allclose(y.grad.numpy(), [-1.0, -1.0, -1.0], **TOL)


def test_mul_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = (x * y).sum()
    z.backward()
    np.testing.assert_allclose(x.grad.numpy(), [4.0, 5.0, 6.0], **TOL)
    np.testing.assert_allclose(y.grad.numpy(), [1.0, 2.0, 3.0], **TOL)


def test_mul_self_backward_matches_numerical_gradient():
    # y = (x * x).sum() -> dy/dx = 2x
    data = np.array([1.0, 2.0, 3.0, 4.0])
    x = Tensor(data.copy(), requires_grad=True)
    y = (x * x).sum()
    y.backward()

    def fn(arr):
        return float(np.sum(arr * arr))

    expected = numerical_grad(fn, data.copy())
    np.testing.assert_allclose(x.grad.numpy(), expected, **FD_TOL)
    np.testing.assert_allclose(x.grad.numpy(), 2 * data, **TOL)


# -- broadcasting -------------------------------------------------------


def test_broadcast_add_backward_reduces_grad_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)
    b = Tensor([10.0, 20.0], requires_grad=True)  # (2,)
    y = (x + b).sum()
    y.backward()
    assert x.grad.shape == (2, 2)
    assert b.grad.shape == (2,)
    np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], **TOL)
    np.testing.assert_allclose(b.grad.numpy(), [2.0, 2.0], **TOL)


def test_broadcast_sub_backward_reduces_grad_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)
    y = (x - b).sum()
    y.backward()
    assert b.grad.shape == (2,)
    np.testing.assert_allclose(b.grad.numpy(), [-2.0, -2.0], **TOL)


def test_broadcast_mul_backward_reduces_grad_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)
    b = Tensor([10.0, 20.0], requires_grad=True)  # (2,)
    y = (x * b).sum()
    y.backward()
    assert x.grad.shape == (2, 2)
    assert b.grad.shape == (2,)
    # dy/dx = b broadcast; dy/db = sum over rows of x
    np.testing.assert_allclose(x.grad.numpy(), [[10.0, 20.0], [10.0, 20.0]], **TOL)
    np.testing.assert_allclose(b.grad.numpy(), [4.0, 6.0], **TOL)


def test_broadcast_scalar_add_backward():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x + 5.0).sum()
    y.backward()
    np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0], **TOL)


# -- sum: full, axis, keepdims ----------------------------------------------


def test_sum_full_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum()
    y.backward()
    np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], **TOL)


def test_sum_axis_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum(axis=0)  # shape (2,)
    y.backward(Tensor([1.0, 1.0]))
    np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], **TOL)


def test_sum_axis_keepdims_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum(axis=1, keepdims=True)  # shape (2, 1)
    y.backward(Tensor([[1.0], [1.0]]))
    np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], **TOL)


def test_sum_backward_matches_numerical_gradient():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = Tensor(data.copy(), requires_grad=True)
    y = x.sum()
    y.backward()

    def fn(arr):
        return float(np.sum(arr))

    expected = numerical_grad(fn, data.copy())
    np.testing.assert_allclose(x.grad.numpy(), expected, **FD_TOL)


# -- reshape ------------------------------------------------------------


def test_reshape_backward_restores_shape():
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    y = x.reshape(2, 3)
    z = y.sum()
    z.backward()
    assert x.grad.shape == (6,)
    np.testing.assert_allclose(x.grad.numpy(), [1.0] * 6, **TOL)


def test_reshape_backward_preserves_gradient_values():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.reshape(4)
    y.backward(Tensor([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(x.grad.numpy(), [[1.0, 2.0], [3.0, 4.0]], **TOL)


# -- matmul ---------------------------------------------------------------


def test_matmul_2d_backward():
    a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_data = np.array([[5.0, 6.0], [7.0, 8.0]])
    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)
    y = (a @ b).sum()
    y.backward()

    # d(sum(A@B))/dA = ones @ B.T ; d/dB = A.T @ ones
    expected_grad_a = np.ones((2, 2)) @ b_data.T
    expected_grad_b = a_data.T @ np.ones((2, 2))
    np.testing.assert_allclose(a.grad.numpy(), expected_grad_a, **TOL)
    np.testing.assert_allclose(b.grad.numpy(), expected_grad_b, **TOL)


def test_matmul_2d_backward_matches_numerical_gradient():
    a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_data = np.array([[5.0, 6.0], [7.0, 8.0]])

    def fn(a_arr):
        return float(np.sum(a_arr @ b_data))

    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)
    y = (a @ b).sum()
    y.backward()

    expected = numerical_grad(fn, a_data.copy())
    np.testing.assert_allclose(a.grad.numpy(), expected, **FD_TOL)


def test_matmul_1d_dot_product_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    y = a @ b  # scalar
    y.backward()
    np.testing.assert_allclose(a.grad.numpy(), [4.0, 5.0, 6.0], **TOL)
    np.testing.assert_allclose(b.grad.numpy(), [1.0, 2.0, 3.0], **TOL)


def test_matmul_matrix_vector_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)
    b = Tensor([1.0, 1.0], requires_grad=True)  # (2,)
    y = (a @ b).sum()  # scalar, y = a @ b then summed
    y.backward()
    # d(sum(a@b))/da = outer(ones(2), b); d/db = a.T @ ones(2)
    np.testing.assert_allclose(a.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], **TOL)
    np.testing.assert_allclose(b.grad.numpy(), [4.0, 6.0], **TOL)


def test_matmul_dimension_mismatch_still_raises_shape_error():
    a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    b = Tensor([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(ShapeMismatchError):
        a @ b


# -- graph behavior: chaining, multiple uses, accumulation -------------------


def test_chained_operations_backward():
    x = Tensor([2.0], requires_grad=True)
    y = x * 3.0
    z = y + 1.0
    w = z.sum()
    w.backward()
    # w = 3x + 1 -> dw/dx = 3
    np.testing.assert_allclose(x.grad.numpy(), [3.0], **TOL)


def test_multiple_use_of_same_tensor_accumulates():
    x = Tensor([2.0], requires_grad=True)
    a = x * 2.0
    b = x * 3.0
    c = (a + b).sum()
    c.backward()
    # c = 2x + 3x = 5x -> dc/dx = 5
    np.testing.assert_allclose(x.grad.numpy(), [5.0], **TOL)


def test_gradient_accumulation_explicit():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    z = y.sum()
    z.backward()
    first_grad = x.grad.numpy().copy()
    np.testing.assert_allclose(first_grad, [2.0, 4.0], **TOL)

    # A fresh forward pass + backward accumulates onto the existing grad.
    y2 = x * x
    z2 = y2.sum()
    z2.backward()
    np.testing.assert_allclose(x.grad.numpy(), first_grad * 2, **TOL)


# -- non-scalar backward behavior --------------------------------------


def test_non_scalar_backward_without_gradient_raises():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2.0
    with pytest.raises(GradientStateError):
        y.backward()


def test_non_scalar_backward_with_explicit_gradient_succeeds():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2.0
    y.backward(Tensor([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0], **TOL)


def test_backward_with_mismatched_gradient_shape_raises():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2.0
    with pytest.raises(ShapeMismatchError):
        y.backward(Tensor([1.0, 1.0]))


def test_scalar_backward_defaults_to_gradient_of_one():
    x = Tensor([2.0], requires_grad=True)
    y = (x * x).sum()
    assert y.shape == ()
    y.backward()
    np.testing.assert_allclose(x.grad.numpy(), [4.0], **TOL)


# -- gradient lifecycle: repeated backward, zero_grad ------------------------


def test_backward_twice_on_same_non_leaf_output_raises():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    with pytest.raises(GradientStateError):
        y.backward()


def test_backward_twice_on_leaf_accumulates():
    x = Tensor([2.0], requires_grad=True)
    x.backward(Tensor([1.0]))
    x.backward(Tensor([1.0]))
    np.testing.assert_allclose(x.grad.numpy(), [2.0], **TOL)


def test_zero_grad_clears_gradient():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None


def test_graph_is_freed_after_backward():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    z = y.sum()
    assert y.grad_fn is not None
    z.backward()
    assert y.grad_fn is None


# -- manual verification example from the milestone spec --------------------


def test_manual_verification_example():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    # d(sum(x^2))/dx = 2x
    np.testing.assert_allclose(x.grad.numpy(), [[2.0, 4.0], [6.0, 8.0]], **TOL)
