"""2D convolution layer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import random as forge_random
from ..exceptions import ShapeMismatchError
from ..tensor.dtype import DEFAULT_DTYPE
from ..tensor.tensor import Tensor
from ._shape_utils import pad_pair, pair
from .module import Module
from .parameter import Parameter


class Conv2d(Module):
    """2D cross-correlation over an NCHW input (the conventional neural-network "convolution").

    `weight` has shape `(out_channels, in_channels, kernel_height, kernel_width)`
    and `bias` (when enabled) has shape `(out_channels,)`. Restricted to
    integer stride, integer symmetric zero padding, no dilation, no groups,
    no transposed convolution -- see `docs/architecture/backend-architecture.md`.

    ## Output shape
    For input `(N, C_in, H, W)`, the output is `(N, C_out, H_out, W_out)` with
    `H_out = floor((H + 2*padding_h - kernel_h) / stride_h) + 1` (and
    correspondingly for `W_out`).

    ## Initialization
    Both `weight` and `bias` are drawn from `Uniform(-1/sqrt(fan_in),
    1/sqrt(fan_in))` where `fan_in = in_channels * kernel_height *
    kernel_width` -- the direct Conv2d analog of `Linear`'s
    `1/sqrt(in_features)` bound (each of the `fan_in` summed terms in one
    output element has variance `~1/(3*fan_in)`, so the sum stays `O(1)`
    regardless of kernel size or channel count). Draws come from
    `forge.random.default_generator()` unless a `generator` is passed
    explicitly, so seeding `forge.random.seed(...)` makes construction
    deterministic.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: "int | tuple[int, int]",
        stride: "int | tuple[int, int]" = 1,
        padding: "int | tuple[int, int]" = 0,
        bias: bool = True,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ShapeMismatchError(
                f"Conv2d requires positive in_channels/out_channels, got "
                f"in_channels={in_channels}, out_channels={out_channels}."
            )
        kh, kw = pair(kernel_size, "kernel_size")
        sh, sw = pair(stride, "stride")
        ph, pw = pad_pair(padding, "padding")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.padding = (ph, pw)

        param_dtype = dtype if dtype is not None else DEFAULT_DTYPE
        rng = generator if generator is not None else forge_random.default_generator()
        fan_in = in_channels * kh * kw
        bound = 1.0 / math.sqrt(fan_in)

        weight_data = rng.uniform(-bound, bound, size=(out_channels, in_channels, kh, kw))
        self.weight = Parameter(weight_data, dtype=param_dtype, device=device)

        if bias:
            bias_data = rng.uniform(-bound, bound, size=(out_channels,))
            self.bias = Parameter(bias_data, dtype=param_dtype, device=device)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ShapeMismatchError(
                f"Conv2d expects a 4D (N, C_in, H, W) input, got shape {x.shape}."
            )
        if x.shape[1] != self.in_channels:
            raise ShapeMismatchError(
                f"Conv2d(in_channels={self.in_channels}) cannot accept input with "
                f"{x.shape[1]} channels (input shape {x.shape})."
            )
        return x.conv2d(self.weight, self.bias, self.stride, self.padding)

    def __repr__(self) -> str:
        return (
            f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None})"
        )


__all__ = ["Conv2d"]
