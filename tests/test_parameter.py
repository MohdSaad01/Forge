import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import GradientStateError
from forge.nn import Parameter

TOL = dict(rtol=1e-6, atol=1e-6)


def test_parameter_is_a_tensor():
    p = Parameter([1.0, 2.0, 3.0])
    assert isinstance(p, Tensor)


def test_parameter_requires_grad_by_default():
    p = Parameter([1.0, 2.0])
    assert p.requires_grad is True
    assert p.is_leaf is True
    assert p.grad_fn is None


def test_parameter_can_opt_out_of_grad():
    p = Parameter([1.0, 2.0], requires_grad=False)
    assert p.requires_grad is False


def test_parameter_rejects_non_float_dtype():
    with pytest.raises(GradientStateError):
        Parameter([1, 2, 3], dtype="int64")


def test_parameter_data_accessible():
    p = Parameter([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(p.numpy(), [[1.0, 2.0], [3.0, 4.0]], **TOL)
    assert p.shape == (2, 2)


def test_parameter_tracks_gradient_through_ops():
    p = Parameter([1.0, 2.0, 3.0])
    y = (p * p).sum()
    y.backward()
    np.testing.assert_allclose(p.grad.numpy(), [2.0, 4.0, 6.0], **TOL)


def test_op_result_on_parameter_is_plain_tensor_not_parameter():
    p = Parameter([1.0, 2.0])
    y = p + p
    assert isinstance(y, Tensor)
    assert not isinstance(y, Parameter)
