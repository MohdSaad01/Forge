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


__all__ = ["MaxPool2d"]
