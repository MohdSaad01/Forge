"""Local backward math for Milestone 1's differentiable operations.

Pure NumPy helpers with no Tensor dependency, kept separate from the graph
engine (``forge.autograd.engine``) so each operation's backward rule is easy
to find, read, and test in isolation.
"""

from __future__ import annotations

import numpy as np


def reduce_grad_to_shape(grad: np.ndarray, shape: "tuple[int, ...]") -> np.ndarray:
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


def matmul_backward(
    grad_output: np.ndarray, a: np.ndarray, b: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """Gradients for ``a @ b``, matching NumPy's 1D/2D matmul semantics."""
    if a.ndim == 1 and b.ndim == 1:
        return grad_output * b, grad_output * a
    if a.ndim == 2 and b.ndim == 1:
        return np.outer(grad_output, b), a.T @ grad_output
    if a.ndim == 1 and b.ndim == 2:
        return b @ grad_output, np.outer(a, grad_output)
    # a.ndim == 2 and b.ndim == 2
    return grad_output @ b.T, a.T @ grad_output


def sum_backward(
    grad_output: np.ndarray,
    original_shape: "tuple[int, ...]",
    ndim: int,
    axis,
    keepdims: bool,
) -> np.ndarray:
    """Gradient for ``x.sum(axis=axis, keepdims=keepdims)``: broadcast back to ``x``'s shape."""
    grad = grad_output
    if axis is not None and not keepdims:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        for ax in sorted(a % ndim for a in axes):
            grad = np.expand_dims(grad, ax)
    return np.broadcast_to(grad, original_shape).copy()
