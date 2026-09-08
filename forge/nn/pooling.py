"""2D max pooling layer."""

from __future__ import annotations

from ..exceptions import ShapeMismatchError
from ..tensor.tensor import Tensor
from ._shape_utils import pad_pair, pair
from .module import Module


class MaxPool2d(Module):
    """2D max pooling over an NCHW tensor. No trainable parameters.

    Defaults `stride` to `kernel_size` when `stride=None`, matching the
    conventional (non-overlapping) pooling window. Ties within a window break
    deterministically to the first maximum in row-major (top-to-bottom, then
    left-to-right) scan order -- see `Tensor.max_pool2d`.
    """

    def __init__(
        self,
        kernel_size: "int | tuple[int, int]",
        stride: "int | tuple[int, int] | None" = None,
        padding: "int | tuple[int, int]" = 0,
    ):
        super().__init__()
        kh, kw = pair(kernel_size, "kernel_size")
        sh, sw = (kh, kw) if stride is None else pair(stride, "stride")
        ph, pw = pad_pair(padding, "padding")

        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.padding = (ph, pw)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ShapeMismatchError(
                f"MaxPool2d expects a 4D (N, C, H, W) input, got shape {x.shape}."
            )
        return x.max_pool2d(self.kernel_size, self.stride, self.padding)

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class MaxPool1d(Module):
    """1D max pooling over an `(N, C, L)` tensor. No trainable parameters.

    Milestone 62: implemented by reshaping to a dummy `(N, C, 1, L)` 4D
    tensor and dispatching to `Tensor.max_pool2d` (`kernel_size=(1, k)`,
    `stride=(1, s)`, `padding=(0, p)`), then reshaping the result back to
    `(N, C, L_out)` -- the same reshape-and-reuse convention `nn.Conv1d`
    uses for `Conv2d`. No new `Backend` method or CUDA kernel; ties within a
    window break the same way `MaxPool2d`'s do (see `Tensor.max_pool2d`).

    Defaults `stride` to `kernel_size` when `stride=None`, matching
    `MaxPool2d`.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: "int | None" = None,
        padding: int = 0,
    ):
        super().__init__()
        k = _positive_int_1d(kernel_size, "kernel_size")
        s = k if stride is None else _positive_int_1d(stride, "stride")
        p = _nonneg_int_1d(padding, "padding")

        self.kernel_size = k
        self.stride = s
        self.padding = p

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ShapeMismatchError(
                f"MaxPool1d expects a 3D (N, C, L) input, got shape {x.shape}."
            )
        N, C, L = x.shape
        x4 = x.reshape(N, C, 1, L)
        y4 = x4.max_pool2d((1, self.kernel_size), (1, self.stride), (0, self.padding))
        _, _, _, L_out = y4.shape
        return y4.reshape(N, C, L_out)

    def __repr__(self) -> str:
        return f"MaxPool1d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


def _positive_int_1d(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShapeMismatchError(f"{name} must be a positive int, got {value!r}.")
    return value


def _nonneg_int_1d(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShapeMismatchError(f"{name} must be a non-negative int, got {value!r}.")
    return value


__all__ = ["MaxPool2d", "MaxPool1d"]
