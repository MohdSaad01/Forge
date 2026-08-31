"""The Forge Tensor abstraction.

A Tensor is a typed, device-associated, multidimensional numerical value.
This module establishes the tensor/operation surface that autograd will
later attach to (Milestone 2+); no computation graph exists yet, and this
class intentionally does not add graph/gradient state ahead of that need.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.exceptions import AxisError

from ..backend import get_backend
from ..backend.device import Device
from ..exceptions import ShapeMismatchError, UnsupportedDeviceError, UnsupportedDTypeError
from .dtype import DType, dtype_from_numpy, infer_default_dtype, resolve_dtype

Shape = tuple[int, ...]


class Tensor:
    """A typed multidimensional numerical value associated with a device."""

    def __init__(self, data: Any, dtype: Any = None, device: "str | Device" = "cpu"):
        self._device = Device.parse(device)
        backend = get_backend(self._device)

        if isinstance(data, Tensor):
            data = data._data

        np_dtype = resolve_dtype(dtype)
        if np_dtype is None and not isinstance(data, np.ndarray):
            # Raw Python data (lists/scalars) has no dtype of its own to
            # preserve, so apply Forge's stable default instead of NumPy's
            # platform-dependent inference.
            np_dtype = infer_default_dtype(np.asarray(data).dtype)
        try:
            array = backend.from_array(data, np_dtype)
        except (TypeError, ValueError) as exc:
            raise UnsupportedDTypeError(
                f"Could not build a tensor from the given data: {exc}"
            ) from exc

        self._dtype = dtype_from_numpy(array.dtype)
        self._data = array

    @classmethod
    def _wrap(cls, array: np.ndarray, device: Device) -> "Tensor":
        """Build a Tensor directly from backend storage, skipping re-validation."""
        instance = object.__new__(cls)
        instance._data = array
        instance._dtype = dtype_from_numpy(array.dtype)
        instance._device = device
        return instance

    # -- Introspection --------------------------------------------------

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def numpy(self) -> np.ndarray:
        """Return the underlying CPU storage as a NumPy array (shares memory)."""
        if self._device.type != "cpu":
            raise UnsupportedDeviceError(
                f"Cannot access NumPy storage for a tensor on device '{self._device}'."
            )
        return self._data

    def __repr__(self) -> str:
        return f"Tensor({self._data!r}, dtype={self._dtype}, device={self._device})"

    def __len__(self) -> int:
        return len(self._data)

    # -- Elementwise binary ops ------------------------------------------

    def _coerce(self, other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, device=self._device)

    def _binary_op(self, other: Any, backend_method: str, op_symbol: str) -> "Tensor":
        other_t = self._coerce(other)
        if other_t._device != self._device:
            raise UnsupportedDeviceError(
                f"Cannot apply '{op_symbol}' to tensors on different devices: "
                f"{self._device} and {other_t._device}."
            )
        try:
            np.broadcast_shapes(self.shape, other_t.shape)
        except ValueError as exc:
            raise ShapeMismatchError(
                f"Cannot apply '{op_symbol}' to shapes {self.shape} and {other_t.shape}: "
                "not broadcastable."
            ) from exc

        backend = get_backend(self._device)
        result = getattr(backend, backend_method)(self._data, other_t._data)
        return Tensor._wrap(result, self._device)

    def __add__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "add", "+")

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "sub", "-")

    def __rsub__(self, other: Any) -> "Tensor":
        return self._coerce(other)._binary_op(self, "sub", "-")

    def __mul__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "mul", "*")

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    # -- Matrix multiplication -------------------------------------------

    def __matmul__(self, other: Any) -> "Tensor":
        other_t = self._coerce(other)
        if other_t._device != self._device:
            raise UnsupportedDeviceError(
                f"Cannot apply '@' to tensors on different devices: "
                f"{self._device} and {other_t._device}."
            )
        if self.ndim not in (1, 2) or other_t.ndim not in (1, 2):
            raise ShapeMismatchError(
                "matmul supports 1D and 2D tensors only, got shapes "
                f"{self.shape} and {other_t.shape}."
            )

        backend = get_backend(self._device)
        try:
            result = backend.matmul(self._data, other_t._data)
        except ValueError as exc:
            raise ShapeMismatchError(
                f"Cannot matrix-multiply shapes {self.shape} and {other_t.shape}: {exc}"
            ) from exc
        return Tensor._wrap(result, self._device)

    # -- Reductions and reshaping -----------------------------------------

    def sum(self, axis: "int | tuple[int, ...] | None" = None, keepdims: bool = False) -> "Tensor":
        backend = get_backend(self._device)
        try:
            result = backend.sum(self._data, axis, keepdims)
        except AxisError as exc:
            raise ShapeMismatchError(
                f"Cannot sum over axis {axis} for tensor of shape {self.shape}: {exc}"
            ) from exc
        return Tensor._wrap(result, self._device)

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        backend = get_backend(self._device)
        try:
            result = backend.reshape(self._data, shape)
        except ValueError as exc:
            raise ShapeMismatchError(
                f"Cannot reshape tensor of shape {self.shape} "
                f"({self._data.size} elements) to {shape}: {exc}"
            ) from exc
        return Tensor._wrap(result, self._device)
