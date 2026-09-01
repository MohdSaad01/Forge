"""CPU backend: the reference implementation (see ADR-002).

Thin wrappers over NumPy. No shape/dtype validation happens here -- that is
the Tensor layer's responsibility, so error messages stay in terms of Forge
tensors rather than raw NumPy internals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Backend


def _reduce_grad_to_shape(grad: np.ndarray, shape: "tuple[int, ...]") -> np.ndarray:
    """Undo NumPy broadcasting: sum an upstream gradient down to ``shape``.

    Mirrors NumPy's broadcasting rule (dimensions align from the right): any
    leading extra dimensions are summed away, then any dimension that was
    broadcast from size 1 is summed back down to size 1.
    """
    ndim_diff = grad.ndim - len(shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class CPUBackend(Backend):
    device_type = "cpu"

    def from_array(self, data: Any, dtype: "np.dtype | None") -> np.ndarray:
        return np.array(data, dtype=dtype)

    def to_numpy(self, storage: Any) -> np.ndarray:
        return storage

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.add(a, b)

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.subtract(a, b)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.multiply(a, b)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def sum(self, a: np.ndarray, axis, keepdims: bool) -> np.ndarray:
        return np.asarray(np.sum(a, axis=axis, keepdims=keepdims))

    def reshape(self, a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(a, shape)

    def relu(self, a: np.ndarray) -> np.ndarray:
        return np.maximum(a, 0)

    def exp(self, a: np.ndarray) -> np.ndarray:
        return np.exp(a)

    def log(self, a: np.ndarray) -> np.ndarray:
        return np.log(a)

    # -- backward (Milestone 10) -------------------------------------------

    def add_backward(self, grad_output: np.ndarray, a: np.ndarray, b: np.ndarray):
        return (
            _reduce_grad_to_shape(grad_output, a.shape),
            _reduce_grad_to_shape(grad_output, b.shape),
        )

    def sub_backward(self, grad_output: np.ndarray, a: np.ndarray, b: np.ndarray):
        return (
            _reduce_grad_to_shape(grad_output, a.shape),
            _reduce_grad_to_shape(-grad_output, b.shape),
        )

    def mul_backward(self, grad_output: np.ndarray, a: np.ndarray, b: np.ndarray):
        return (
            _reduce_grad_to_shape(grad_output * b, a.shape),
            _reduce_grad_to_shape(grad_output * a, b.shape),
        )

    def matmul_backward(self, grad_output: np.ndarray, a: np.ndarray, b: np.ndarray):
        if a.ndim == 1 and b.ndim == 1:
            return grad_output * b, grad_output * a
        if a.ndim == 2 and b.ndim == 1:
            return np.outer(grad_output, b), a.T @ grad_output
        if a.ndim == 1 and b.ndim == 2:
            return b @ grad_output, np.outer(a, grad_output)
        return grad_output @ b.T, a.T @ grad_output

    def sum_backward(
        self, grad_output: np.ndarray, original_shape: "tuple[int, ...]", ndim: int, axis, keepdims: bool
    ) -> np.ndarray:
        grad = grad_output
        if axis is not None and not keepdims:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            for ax in sorted(a % ndim for a in axes):
                grad = np.expand_dims(grad, ax)
        return np.broadcast_to(grad, original_shape).copy()

    def reshape_backward(self, grad_output: np.ndarray, original_shape: "tuple[int, ...]") -> np.ndarray:
        return grad_output.reshape(original_shape)

    def relu_backward(self, grad_output: np.ndarray, a: np.ndarray) -> np.ndarray:
        positive_mask = a > 0
        return np.where(positive_mask, grad_output, 0).astype(grad_output.dtype, copy=False)

    # -- optimizer (Milestone 10) ------------------------------------------

    def sgd_step(self, data: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
        data -= lr * grad
        return data
