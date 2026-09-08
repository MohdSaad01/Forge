"""Nearest-neighbor spatial upsampling."""

from __future__ import annotations

from ..exceptions import ShapeMismatchError
from ..tensor.tensor import Tensor
from ._shape_utils import pair
from .module import Module


class UpsampleNearest2d(Module):
    """Nearest-neighbor upsample over an NCHW tensor. No trainable parameters.

    `(N, C, H, W) -> (N, C, H*sh, W*sw)`: each input element is repeated into
    an `(sh, sw)` block of identical output elements -- the shape-inverse of
    a `(sh, sw)` `MaxPool2d`. Added in Milestone 63 as the decoder half of
    `examples/autoencoder/`'s convolutional autoencoder: composing this with
    `Conv2d` (upsample, then convolve) is a standard, deliberate way to grow
    a spatial feature map back up without a transposed convolution -- it
    avoids the checkerboard-artifact failure mode transposed convolution is
    well known for. See `Tensor.upsample_nearest2d`/`Backend.upsample_nearest2d`.
    """

    def __init__(self, scale_factor: "int | tuple[int, int]" = 2):
        super().__init__()
        self.scale_factor = pair(scale_factor, "scale_factor")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ShapeMismatchError(
                f"UpsampleNearest2d expects a 4D (N, C, H, W) input, got shape {x.shape}."
            )
        return x.upsample_nearest2d(self.scale_factor)

    def __repr__(self) -> str:
        return f"UpsampleNearest2d(scale_factor={self.scale_factor})"


__all__ = ["UpsampleNearest2d"]
