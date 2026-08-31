"""The Forge Tensor abstraction.

A Tensor is a typed, device-associated, multidimensional numerical value
that can optionally track gradient information. Differentiable operations
attach a ``forge.autograd.Node`` (the tensor's ``grad_fn``) recording enough
information to compute input gradients; ``Tensor.backward()`` triggers
reverse-mode traversal of that graph. See ``docs/architecture/autograd.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.exceptions import AxisError

from ..autograd import Node, is_grad_enabled, run_backward
from ..autograd.functions import matmul_backward, reduce_grad_to_shape, sum_backward
from ..backend import get_backend
from ..backend.device import Device
from ..exceptions import (
    GradientStateError,
    ShapeMismatchError,
    UnsupportedDeviceError,
    UnsupportedDTypeError,
)
from .dtype import DType, dtype_from_numpy, infer_default_dtype, resolve_dtype

Shape = tuple[int, ...]

_GRAD_CAPABLE_DTYPES = (DType.FLOAT32, DType.FLOAT64)


class Tensor:
    """A typed multidimensional numerical value associated with a device."""

    def __init__(
        self,
        data: Any,
        dtype: Any = None,
        device: "str | Device" = "cpu",
        requires_grad: bool = False,
    ):
        self._device = Device.parse(device)
        backend = get_backend(self._device)

        if isinstance(data, Tensor):
            if data._device != self._device:
                raise UnsupportedDeviceError(
                    f"Cannot construct a Tensor on device '{self._device}' directly from a "
                    f"Tensor already on device '{data._device}'. Call "
                    f"tensor.to('{self._device}') to move it explicitly first."
                )
            data = data._data

        np_dtype = resolve_dtype(dtype)
        if np_dtype is None and not hasattr(data, "dtype"):
            # Raw Python data (lists/scalars) has no dtype of its own to
            # preserve, so apply Forge's stable default instead of NumPy's
            # platform-dependent inference. Anything that already carries a
            # concrete dtype (a NumPy array/scalar, or another backend's
            # storage, e.g. CUDAStorage) keeps it instead.
            np_dtype = infer_default_dtype(np.asarray(data).dtype)
        try:
            array = backend.from_array(data, np_dtype)
        except (TypeError, ValueError) as exc:
            raise UnsupportedDTypeError(
                f"Could not build a tensor from the given data: {exc}"
            ) from exc

        self._dtype = dtype_from_numpy(array.dtype)
        self._data = array

        self._requires_grad = bool(requires_grad)
        if self._requires_grad and self._dtype not in _GRAD_CAPABLE_DTYPES:
            raise GradientStateError(
                f"requires_grad=True requires a floating-point dtype, got '{self._dtype}'."
            )
        self._grad_fn: "Node | None" = None
        self._is_leaf = True
        self.grad: "Tensor | None" = None

    @classmethod
    def _wrap(
        cls,
        array: np.ndarray,
        device: Device,
        requires_grad: bool = False,
        grad_fn: "Node | None" = None,
    ) -> "Tensor":
        """Build a Tensor directly from backend storage, skipping re-validation."""
        instance = object.__new__(cls)
        instance._data = array
        instance._dtype = dtype_from_numpy(array.dtype)
        instance._device = device
        instance._requires_grad = requires_grad
        instance._grad_fn = grad_fn
        instance._is_leaf = grad_fn is None
        instance.grad = None
        return instance

    def _differentiable_wrap(
        self,
        array: np.ndarray,
        inputs: "tuple[Tensor, ...]",
        backward_fn,
        name: str,
    ) -> "Tensor":
        """Wrap a forward result, attaching a grad_fn if any input requires grad.

        Skipped entirely inside `forge.no_grad()` (`is_grad_enabled()` is
        `False`): the result is a plain, non-grad-requiring Tensor regardless
        of its inputs, so no graph is built for a forward pass that will
        never be differentiated.
        """
        requires_grad = is_grad_enabled() and any(t._requires_grad for t in inputs)
        if requires_grad and self._device.type != "cpu":
            raise UnsupportedDeviceError(
                "Automatic differentiation is only supported on the 'cpu' device in this "
                f"milestone; got a differentiable '{name}' operation on device "
                f"'{self._device}'. Construct tensors with requires_grad=False to run this "
                "operation on this device, or perform gradient-tracked operations on CPU. "
                "See docs/architecture/cuda-backend.md."
            )
        grad_fn = Node(inputs, backward_fn, name) if requires_grad else None
        return Tensor._wrap(array, self._device, requires_grad=requires_grad, grad_fn=grad_fn)

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

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def is_leaf(self) -> bool:
        """True for tensors not produced by a tracked differentiable operation."""
        return self._is_leaf

    @property
    def grad_fn(self) -> "Node | None":
        return self._grad_fn

    def numpy(self) -> np.ndarray:
        """Return the underlying CPU storage as a NumPy array (shares memory)."""
        if self._device.type != "cpu":
            raise UnsupportedDeviceError(
                f"Cannot access NumPy storage for a tensor on device '{self._device}'."
            )
        return self._data

    def __repr__(self) -> str:
        extra = ", requires_grad=True" if self._requires_grad else ""
        return f"Tensor({self._data!r}, dtype={self._dtype}, device={self._device}{extra})"

    def __len__(self) -> int:
        return len(self._data)

    # -- Elementwise binary ops ------------------------------------------

    def _coerce(self, other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, device=self._device)

    def _binary_op(self, other: Any, backend_method: str, op_symbol: str, local_grads) -> "Tensor":
        """Apply a broadcasting elementwise op.

        ``local_grads(grad_output, a_data, b_data)`` returns the raw
        (pre-broadcast-reduction) gradients for ``self`` and ``other``; this
        method reduces each back down to its input's shape.
        """
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

        a_shape, b_shape = self.shape, other_t.shape
        a_data, b_data = self._data, other_t._data

        def backward_fn(grad_output: np.ndarray):
            grad_a_raw, grad_b_raw = local_grads(grad_output, a_data, b_data)
            return (
                reduce_grad_to_shape(grad_a_raw, a_shape),
                reduce_grad_to_shape(grad_b_raw, b_shape),
            )

        return self._differentiable_wrap(result, (self, other_t), backward_fn, op_symbol)

    def __add__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "add", "+", lambda g, a, b: (g, g))

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "sub", "-", lambda g, a, b: (g, -g))

    def __rsub__(self, other: Any) -> "Tensor":
        return self._coerce(other)._binary_op(self, "sub", "-", lambda g, a, b: (g, -g))

    def __mul__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "mul", "*", lambda g, a, b: (g * b, g * a))

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

        a_data, b_data = self._data, other_t._data

        def backward_fn(grad_output: np.ndarray):
            return matmul_backward(grad_output, a_data, b_data)

        return self._differentiable_wrap(result, (self, other_t), backward_fn, "@")

    # -- Reductions and reshaping -----------------------------------------

    def sum(self, axis: "int | tuple[int, ...] | None" = None, keepdims: bool = False) -> "Tensor":
        backend = get_backend(self._device)
        try:
            result = backend.sum(self._data, axis, keepdims)
        except AxisError as exc:
            raise ShapeMismatchError(
                f"Cannot sum over axis {axis} for tensor of shape {self.shape}: {exc}"
            ) from exc

        original_shape, ndim = self.shape, self.ndim

        def backward_fn(grad_output: np.ndarray):
            return (sum_backward(grad_output, original_shape, ndim, axis, keepdims),)

        return self._differentiable_wrap(result, (self,), backward_fn, "sum")

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

        original_shape = self.shape

        def backward_fn(grad_output: np.ndarray):
            return (grad_output.reshape(original_shape),)

        return self._differentiable_wrap(result, (self,), backward_fn, "reshape")

    def relu(self) -> "Tensor":
        backend = get_backend(self._device)
        result = backend.relu(self._data)

        positive_mask = self._data > 0

        def backward_fn(grad_output: np.ndarray):
            return (np.where(positive_mask, grad_output, 0).astype(grad_output.dtype, copy=False),)

        return self._differentiable_wrap(result, (self,), backward_fn, "relu")

    def exp(self) -> "Tensor":
        backend = get_backend(self._device)
        result = backend.exp(self._data)

        def backward_fn(grad_output: np.ndarray):
            return (grad_output * result,)

        return self._differentiable_wrap(result, (self,), backward_fn, "exp")

    def log(self) -> "Tensor":
        backend = get_backend(self._device)
        result = backend.log(self._data)

        input_data = self._data

        def backward_fn(grad_output: np.ndarray):
            return (grad_output / input_data,)

        return self._differentiable_wrap(result, (self,), backward_fn, "log")

    # -- Device transfer -----------------------------------------------------

    def to(self, device: "str | Device") -> "Tensor":
        """Explicitly move this tensor's data to another device, copying it.

        Never happens implicitly as a side effect of another operation (see
        `_binary_op`'s device-mismatch check) -- this is the one sanctioned
        way to cross devices. The result is always a fresh leaf tensor with
        `requires_grad=False`, regardless of this tensor's own gradient
        state: Forge does not support automatic differentiation across a
        device transfer in this milestone (see
        `docs/architecture/cuda-backend.md`), and never fakes it by quietly
        keeping the source tensor's graph alive.
        """
        target = Device.parse(device)
        if target == self._device:
            return self

        target_backend = get_backend(target)
        source_backend = get_backend(self._device)
        host_array = source_backend.to_numpy(self._data)
        new_storage = target_backend.from_array(host_array, host_array.dtype)
        return Tensor._wrap(new_storage, target, requires_grad=False)

    # -- Autograd ----------------------------------------------------------

    def backward(self, gradient: "Tensor | Any | None" = None) -> None:
        """Run reverse-mode autodiff from this tensor, accumulating into leaf `.grad`s.

        For a non-scalar tensor, ``gradient`` must be supplied explicitly and
        match this tensor's shape; a scalar tensor defaults to an upstream
        gradient of 1. Gradients accumulate into each leaf's ``.grad`` across
        separate ``backward()`` calls (use ``zero_grad()`` to reset). The
        graph feeding this tensor is freed as it is consumed, so calling
        ``backward()`` again on the same non-leaf output raises
        ``GradientStateError``; build a new forward pass instead.
        """
        if self._device.type != "cpu":
            raise UnsupportedDeviceError(
                f"backward() is only supported for tensors on device 'cpu' in this milestone; "
                f"got device '{self._device}'. See docs/architecture/cuda-backend.md."
            )
        if not self._requires_grad:
            raise GradientStateError("backward() called on a tensor that does not require grad.")
        if not self._is_leaf and self._grad_fn is None:
            raise GradientStateError(
                "backward() was already called on this graph and it has been freed. "
                "Run a new forward pass to compute gradients again."
            )

        if gradient is None:
            if self.shape != ():
                raise GradientStateError(
                    "backward() requires an explicit upstream gradient for a non-scalar "
                    f"output of shape {self.shape}; only scalar outputs default to a "
                    "gradient of 1."
                )
            grad_array = np.ones((), dtype=self._data.dtype)
        else:
            grad_tensor = gradient if isinstance(gradient, Tensor) else Tensor(gradient, device=self._device)
            if grad_tensor.shape != self.shape:
                raise ShapeMismatchError(
                    f"Upstream gradient shape {grad_tensor.shape} does not match "
                    f"output shape {self.shape}."
                )
            grad_array = grad_tensor._data.astype(self._data.dtype, copy=False)

        run_backward(self, grad_array)

    def zero_grad(self) -> None:
        """Clear this tensor's accumulated gradient."""
        self.grad = None

    def _accumulate_grad(self, grad_array: np.ndarray) -> None:
        if self.grad is None:
            self.grad = Tensor._wrap(np.array(grad_array, dtype=self._data.dtype), self._device)
        else:
            self.grad = Tensor._wrap(self.grad._data + grad_array, self._device)
