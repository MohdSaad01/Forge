import numpy as np
import pytest

from forge import DEFAULT_DTYPE, Device, DType, Tensor
from forge.exceptions import UnsupportedDTypeError


def test_create_from_list():
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    np.testing.assert_array_equal(t.numpy(), np.array([1, 2, 3], dtype=np.float32))


def test_create_from_nested_list():
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)


def test_create_from_numpy_array_preserves_dtype():
    arr = np.array([1, 2, 3], dtype=np.int64)
    t = Tensor(arr)
    assert t.dtype == DType.INT64


def test_create_from_tensor():
    original = Tensor([1.0, 2.0], dtype="float64")
    copy = Tensor(original)
    assert copy.dtype == DType.FLOAT64
    np.testing.assert_array_equal(copy.numpy(), original.numpy())


def test_default_dtype_is_float32_for_float_data():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.dtype == DEFAULT_DTYPE
    assert t.dtype == DType.FLOAT32


def test_explicit_dtype_via_string():
    t = Tensor([1, 2, 3], dtype="int32")
    assert t.dtype == DType.INT32


def test_explicit_dtype_via_enum():
    t = Tensor([1, 2, 3], dtype=DType.INT64)
    assert t.dtype == DType.INT64


def test_bool_tensor():
    t = Tensor([True, False, True], dtype="bool")
    assert t.dtype == DType.BOOL


def test_default_device_is_cpu():
    t = Tensor([1, 2, 3])
    assert t.device == Device("cpu")
    assert str(t.device) == "cpu"


def test_repr_contains_dtype_and_device():
    t = Tensor([1, 2, 3], dtype="int32")
    text = repr(t)
    assert "int32" in text
    assert "cpu" in text


def test_unsupported_dtype_string_raises_clearly():
    with pytest.raises(UnsupportedDTypeError, match="complex64"):
        Tensor([1, 2, 3], dtype="complex64")


def test_unsupported_numpy_dtype_raises_clearly():
    with pytest.raises(UnsupportedDTypeError):
        Tensor(np.array([1, 2, 3], dtype=np.complex128))


def test_len_reports_first_dimension():
    t = Tensor([[1, 2], [3, 4], [5, 6]])
    assert len(t) == 3
