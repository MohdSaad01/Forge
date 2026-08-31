import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import ShapeMismatchError

TOL = dict(rtol=1e-6, atol=1e-6)


def test_add_elementwise():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([10.0, 20.0, 30.0])
    result = (a + b).numpy()
    np.testing.assert_allclose(result, [11.0, 22.0, 33.0], **TOL)


def test_add_broadcasts():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([10.0, 20.0])
    result = (a + b).numpy()
    np.testing.assert_allclose(result, [[11.0, 22.0], [13.0, 24.0]], **TOL)


def test_add_with_python_scalar():
    a = Tensor([1.0, 2.0, 3.0])
    result = (a + 1).numpy()
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0], **TOL)


def test_radd_with_python_scalar():
    a = Tensor([1.0, 2.0, 3.0])
    result = (1 + a).numpy()
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0], **TOL)


def test_sub_elementwise():
    a = Tensor([5.0, 7.0, 9.0])
    b = Tensor([1.0, 2.0, 3.0])
    result = (a - b).numpy()
    np.testing.assert_allclose(result, [4.0, 5.0, 6.0], **TOL)


def test_rsub_with_python_scalar():
    a = Tensor([1.0, 2.0, 3.0])
    result = (10 - a).numpy()
    np.testing.assert_allclose(result, [9.0, 8.0, 7.0], **TOL)


def test_mul_elementwise():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([2.0, 2.0, 2.0])
    result = (a * b).numpy()
    np.testing.assert_allclose(result, [2.0, 4.0, 6.0], **TOL)


def test_add_shape_mismatch_raises_clearly():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([1.0, 2.0])
    with pytest.raises(ShapeMismatchError, match=r"\(3,\).*\(2,\)"):
        a + b


def test_sub_shape_mismatch_raises_clearly():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ShapeMismatchError):
        a - b


def test_matmul_2d():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    result = (a @ b).numpy()
    np.testing.assert_allclose(result, [[19.0, 22.0], [43.0, 50.0]], **TOL)


def test_matmul_vector_dot_product():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    result = (a @ b).numpy()
    np.testing.assert_allclose(result, 32.0, **TOL)


def test_matmul_inner_dimension_mismatch_raises_clearly():
    a = Tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)
    b = Tensor([[1.0, 2.0]])  # shape (1, 2)
    with pytest.raises(ShapeMismatchError):
        a @ b


def test_matmul_rejects_higher_rank_tensors():
    a = Tensor(np.ones((2, 2, 2)))
    b = Tensor(np.ones((2, 2, 2)))
    with pytest.raises(ShapeMismatchError):
        a @ b


def test_sum_all_elements():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    result = t.sum()
    assert result.shape == ()
    np.testing.assert_allclose(result.numpy(), 10.0, **TOL)


def test_sum_along_axis():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    result = t.sum(axis=0)
    np.testing.assert_allclose(result.numpy(), [4.0, 6.0], **TOL)


def test_sum_keepdims():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    result = t.sum(axis=1, keepdims=True)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result.numpy(), [[3.0], [7.0]], **TOL)


def test_sum_invalid_axis_raises_clearly():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ShapeMismatchError):
        t.sum(axis=5)


def test_reshape_valid():
    t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = t.reshape(2, 3)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], **TOL)


def test_reshape_accepts_tuple_argument():
    t = Tensor([1.0, 2.0, 3.0, 4.0])
    result = t.reshape((2, 2))
    assert result.shape == (2, 2)


def test_reshape_invalid_size_raises_clearly():
    t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ShapeMismatchError):
        t.reshape(2, 3)


def test_operations_return_forge_tensors():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    assert isinstance(a + b, Tensor)
    assert isinstance(a - b, Tensor)
    assert isinstance(a * b, Tensor)
    assert isinstance(a @ b, Tensor)
    assert isinstance(a.sum(), Tensor)
    assert isinstance(a.reshape(2, 1), Tensor)
