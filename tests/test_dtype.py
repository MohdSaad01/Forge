import numpy as np
import pytest

from forge import DType, Tensor
from forge.exceptions import UnsupportedDTypeError
from forge.tensor.dtype import dtype_from_numpy, resolve_dtype


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (DType.FLOAT32, np.dtype(np.float32)),
        (DType.FLOAT64, np.dtype(np.float64)),
        (DType.INT32, np.dtype(np.int32)),
        (DType.INT64, np.dtype(np.int64)),
        (DType.BOOL, np.dtype(np.bool_)),
    ],
)
def test_dtype_maps_to_expected_numpy_dtype(dtype, expected):
    assert dtype.numpy_dtype == expected


def test_dtype_from_numpy_round_trips():
    assert dtype_from_numpy(np.float32) == DType.FLOAT32
    assert dtype_from_numpy(np.dtype("int64")) == DType.INT64


def test_dtype_from_numpy_rejects_unsupported():
    with pytest.raises(UnsupportedDTypeError):
        dtype_from_numpy(np.float16)


def test_resolve_dtype_none_means_infer():
    assert resolve_dtype(None) is None


def test_resolve_dtype_accepts_raw_python_type():
    assert resolve_dtype(bool) == np.dtype(np.bool_)


def test_matmul_requires_matching_dtypes_is_not_enforced_but_promotes():
    # Forge lets NumPy's own promotion rules apply for mixed-dtype ops; it
    # does not silently truncate. int64 * float32 promotes to float64.
    a = Tensor([1, 2, 3], dtype="int64")
    b = Tensor([1.0, 2.0, 3.0], dtype="float32")
    result = a * b
    assert result.dtype == DType.FLOAT64
