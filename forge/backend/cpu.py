"""CPU backend: the reference implementation (see ADR-002).

Thin wrappers over NumPy. No shape/dtype validation happens here -- that is
the Tensor layer's responsibility, so error messages stay in terms of Forge
tensors rather than raw NumPy internals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Backend


def _pad_nchw(a: np.ndarray, ph: int, pw: int, value: float) -> np.ndarray:
    """Zero- (or `value`-) pad the last two (H, W) axes of an NCHW array."""
    if ph == 0 and pw == 0:
        return a
    return np.pad(a, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant", constant_values=value)


def _sliding_windows(padded: np.ndarray, kh: int, kw: int, sh: int, sw: int) -> np.ndarray:
    """(N, C, Hp, Wp) -> (N, C, H_out, W_out, kh, kw) strided window view (no copy)."""
    windows = sliding_window_view(padded, (kh, kw), axis=(2, 3))
    return windows[:, :, ::sh, ::sw, :, :]


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

    def exp_backward(self, grad_output: np.ndarray, result: np.ndarray) -> np.ndarray:
        return grad_output * result

    def log_backward(self, grad_output: np.ndarray, a: np.ndarray) -> np.ndarray:
        return grad_output / a

    # -- optimizer (Milestone 10) ------------------------------------------

    def sgd_step(self, data: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
        data -= lr * grad
        return data

    # -- CrossEntropyLoss support (Milestone 14) ----------------------------

    def max_axis1(self, a: np.ndarray) -> np.ndarray:
        return np.max(a, axis=1, keepdims=True)

    # -- Conv2d / MaxPool2d (Milestone 15) -----------------------------------
    #
    # Both implementations are im2col-style: build a strided (zero-copy)
    # sliding-window view of the (padded) input via `sliding_window_view`,
    # then reduce it with vectorized NumPy/BLAS ops (a big matmul for
    # `conv2d`, a max reduction for `max_pool2d`) rather than a Python loop
    # over every output element. Backward recomputes the same window view
    # from the saved forward input (`x`/`a`) -- the same "recompute from a
    # saved input" convention `relu_backward`/`exp_backward` already use --
    # and scatters each window position's contribution back with a small
    # (kh*kw-iteration) loop of strided `+=` accumulation, which correctly
    # sums overlapping windows when `stride < kernel_size`.

    def conv2d(self, x: np.ndarray, weight: np.ndarray, bias: "np.ndarray | None",
               stride: "tuple[int, int]", padding: "tuple[int, int]") -> np.ndarray:
        N, Cin, H, W = x.shape
        Cout, _, KH, KW = weight.shape
        SH, SW = stride
        PH, PW = padding
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1

        padded = _pad_nchw(x, PH, PW, 0.0)
        windows = _sliding_windows(padded, KH, KW, SH, SW)  # (N, Cin, H_out, W_out, KH, KW)
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N, H_out * W_out, Cin * KH * KW)
        w_flat = weight.reshape(Cout, Cin * KH * KW)

        out = cols @ w_flat.T  # (N, H_out*W_out, Cout), batched matmul
        out = out.transpose(0, 2, 1).reshape(N, Cout, H_out, W_out)
        if bias is not None:
            out = out + bias.reshape(1, Cout, 1, 1)
        return np.ascontiguousarray(out)

    def conv2d_backward(
        self, grad_output: np.ndarray, x: np.ndarray, weight: np.ndarray, bias: "np.ndarray | None",
        stride: "tuple[int, int]", padding: "tuple[int, int]",
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray | None]":
        N, Cin, H, W = x.shape
        Cout, _, KH, KW = weight.shape
        SH, SW = stride
        PH, PW = padding
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]

        padded = _pad_nchw(x, PH, PW, 0.0)
        windows = _sliding_windows(padded, KH, KW, SH, SW)  # (N, Cin, H_out, W_out, KH, KW)
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N, H_out * W_out, Cin * KH * KW)

        grad_out_rows = grad_output.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, Cout)
        cols_rows = cols.reshape(N * H_out * W_out, Cin * KH * KW)

        grad_weight = (grad_out_rows.T @ cols_rows).reshape(Cout, Cin, KH, KW)
        grad_bias = grad_output.sum(axis=(0, 2, 3)) if bias is not None else None

        w_flat = weight.reshape(Cout, Cin * KH * KW)
        grad_cols = (grad_out_rows @ w_flat).reshape(N, H_out, W_out, Cin, KH, KW)

        grad_padded = np.zeros((N, Cin, H + 2 * PH, W + 2 * PW), dtype=x.dtype)
        for di in range(KH):
            for dj in range(KW):
                grad_padded[:, :, di : di + SH * H_out : SH, dj : dj + SW * W_out : SW] += (
                    grad_cols[:, :, :, :, di, dj].transpose(0, 3, 1, 2)
                )
        grad_x = np.ascontiguousarray(grad_padded[:, :, PH : PH + H, PW : PW + W])
        return grad_x, grad_weight, grad_bias

    def max_pool2d(
        self, a: np.ndarray, kernel_size: "tuple[int, int]", stride: "tuple[int, int]", padding: "tuple[int, int]"
    ) -> np.ndarray:
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding
        padded = _pad_nchw(a, PH, PW, -np.inf)
        windows = _sliding_windows(padded, KH, KW, SH, SW)  # (N, C, H_out, W_out, KH, KW)
        return np.ascontiguousarray(windows.max(axis=(4, 5)))

    def max_pool2d_backward(
        self, grad_output: np.ndarray, a: np.ndarray, kernel_size: "tuple[int, int]",
        stride: "tuple[int, int]", padding: "tuple[int, int]",
    ) -> np.ndarray:
        N, C, H, W = a.shape
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]

        padded = _pad_nchw(a, PH, PW, -np.inf)
        windows = _sliding_windows(padded, KH, KW, SH, SW)  # (N, C, H_out, W_out, KH, KW)
        # First occurrence (in row-major kh-then-kw scan order) of each
        # window's maximum wins ties -- `np.argmax`'s own documented
        # tie-breaking rule, applied to the window flattened in that order.
        flat = windows.reshape(N, C, H_out, W_out, KH * KW)
        argmax_idx = np.argmax(flat, axis=-1)
        onehot = np.zeros_like(flat)
        np.put_along_axis(onehot, argmax_idx[..., None], 1.0, axis=-1)
        onehot = onehot.reshape(N, C, H_out, W_out, KH, KW)

        grad_padded = np.zeros((N, C, H + 2 * PH, W + 2 * PW), dtype=a.dtype)
        for di in range(KH):
            for dj in range(KW):
                grad_padded[:, :, di : di + SH * H_out : SH, dj : dj + SW * W_out : SW] += (
                    onehot[:, :, :, :, di, dj] * grad_output
                )
        return np.ascontiguousarray(grad_padded[:, :, PH : PH + H, PW : PW + W])
