"""Forge's dtype identity.

Forge exposes its own small ``DType`` enum rather than leaking raw NumPy
dtype objects through the public Tensor API, while still using NumPy dtypes
as the underlying storage representation on the CPU backend.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..exceptions import UnsupportedDTypeError


class DType(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"

    @property
    def numpy_dtype(self) -> np.dtype:
        return _TO_NUMPY[self]

    def __str__(self) -> str:
        return self.value


_TO_NUMPY: dict[DType, np.dtype] = {
    DType.FLOAT32: np.dtype(np.float32),
    DType.FLOAT64: np.dtype(np.float64),
    DType.INT32: np.dtype(np.int32),
    DType.INT64: np.dtype(np.int64),
    DType.BOOL: np.dtype(np.bool_),
}

_FROM_NUMPY: dict[np.dtype, DType] = {v: k for k, v in _TO_NUMPY.items()}

DEFAULT_DTYPE = DType.FLOAT32
DEFAULT_INT_DTYPE = DType.INT64

_DEFAULT_BY_KIND = {
    "f": DEFAULT_DTYPE.numpy_dtype,
    "i": DEFAULT_INT_DTYPE.numpy_dtype,
    "u": DEFAULT_INT_DTYPE.numpy_dtype,
    "b": DType.BOOL.numpy_dtype,
}


def infer_default_dtype(probe_dtype) -> np.dtype:
    """Map a NumPy-inferred dtype (from raw Python data) to Forge's default.

    Plain Python data has no dtype of its own to preserve, so Forge picks a
    stable default per kind (float32 for floats, int64 for integers,
    regardless of platform) instead of NumPy's platform-dependent inference
    (e.g. plain Python int lists default to int32 on Windows, int64 on Linux).
    """
    probe_dtype = np.dtype(probe_dtype)
    mapped = _DEFAULT_BY_KIND.get(probe_dtype.kind)
    if mapped is None:
        supported = ", ".join(d.value for d in DType)
        raise UnsupportedDTypeError(
            f"Cannot infer a supported dtype from data with dtype '{probe_dtype}'. "
            f"Supported dtypes: {supported}."
        )
    return mapped


def dtype_from_numpy(np_dtype) -> DType:
    """Resolve a concrete NumPy dtype to a Forge ``DType``, or raise clearly."""
    resolved = _FROM_NUMPY.get(np.dtype(np_dtype))
    if resolved is None:
        supported = ", ".join(d.value for d in DType)
        raise UnsupportedDTypeError(
            f"Unsupported dtype '{np.dtype(np_dtype)}'. Supported dtypes: {supported}."
        )
    return resolved


def resolve_dtype(dtype) -> "np.dtype | None":
    """Resolve a user-supplied dtype spec to a NumPy dtype, or None to infer from data.

    Accepts a ``DType``, a Forge dtype name (e.g. ``"float32"``), or a raw
    NumPy-compatible dtype/type (e.g. ``np.float32``, ``float``, ``bool``).
    """
    if dtype is None:
        return None
    if isinstance(dtype, DType):
        return dtype.numpy_dtype
    if isinstance(dtype, str):
        try:
            return DType(dtype).numpy_dtype
        except ValueError:
            supported = ", ".join(d.value for d in DType)
            raise UnsupportedDTypeError(
                f"Unsupported dtype '{dtype}'. Supported dtypes: {supported}."
            ) from None
    try:
        np_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise UnsupportedDTypeError(f"Unsupported dtype: {dtype!r}") from exc
    return dtype_from_numpy(np_dtype).numpy_dtype
