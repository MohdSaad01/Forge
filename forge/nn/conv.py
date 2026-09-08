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


class Conv1d(Module):
    """1D cross-correlation over an `(N, C_in, L)` input -- temporal/sequence convolution.

    `weight` has shape `(out_channels, in_channels, kernel_size)` and `bias`
    (when enabled) has shape `(out_channels,)`, the direct 1D analogs of
    `Conv2d`'s parameter shapes. Same restrictions as `Conv2d`: integer
    stride, integer symmetric zero padding, no dilation, no groups, no
    transposed convolution.

    ## Implementation

    Milestone 62: implemented entirely by reshaping to a dummy `(N, C_in, 1,
    L)` 4D tensor and dispatching to the existing `Tensor.conv2d`/`Conv2d`
    machinery (`stride=(1, stride)`, `padding=(0, padding)`, a `(C_out,
    C_in, 1, kernel_size)` weight view), then reshaping the `(N, C_out, 1,
    L_out)` result back down to `(N, C_out, L_out)` -- the same "compose from
    an existing differentiable op, no per-device code" convention
    `nn.Flatten` already uses for `Tensor.reshape`. `Tensor.reshape` is
    already real and differentiable on both CPU and CUDA
    (`docs/architecture/cuda-backend.md`), so `Conv1d` needs no new `Backend`
    method, no new CUDA kernel, and inherits `Conv2d`'s already CPU/CUDA
    tested forward+backward correctness (including its optimized CUDA
    kernels) automatically -- there is no separate "Conv1d kernel" to keep in
    parity. `weight`/`bias` are still real, independent `Parameter`s stored
    in their natural 3D/1D shapes; only the *view* passed into `conv2d` is
    reshaped, once per `forward()` call.

    ## Output shape
    For input `(N, C_in, L)`, the output is `(N, C_out, L_out)` with
    `L_out = floor((L + 2*padding - kernel_size) / stride) + 1`.

    ## Initialization
    Same `Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))` scheme as `Conv2d`, with
    `fan_in = in_channels * kernel_size`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ShapeMismatchError(
                f"Conv1d requires positive in_channels/out_channels, got "
                f"in_channels={in_channels}, out_channels={out_channels}."
            )
        k = _positive_int(kernel_size, "kernel_size")
        s = _positive_int(stride, "stride")
        p = _nonneg_int(padding, "padding")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.stride = s
        self.padding = p

        param_dtype = dtype if dtype is not None else DEFAULT_DTYPE
        rng = generator if generator is not None else forge_random.default_generator()
        fan_in = in_channels * k
        bound = 1.0 / math.sqrt(fan_in)

        weight_data = rng.uniform(-bound, bound, size=(out_channels, in_channels, k))
        self.weight = Parameter(weight_data, dtype=param_dtype, device=device)

        if bias:
            bias_data = rng.uniform(-bound, bound, size=(out_channels,))
            self.bias = Parameter(bias_data, dtype=param_dtype, device=device)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ShapeMismatchError(
                f"Conv1d expects a 3D (N, C_in, L) input, got shape {x.shape}."
            )
        if x.shape[1] != self.in_channels:
            raise ShapeMismatchError(
                f"Conv1d(in_channels={self.in_channels}) cannot accept input with "
                f"{x.shape[1]} channels (input shape {x.shape})."
            )
        N, Cin, L = x.shape
        x4 = x.reshape(N, Cin, 1, L)
        w4 = self.weight.reshape(self.out_channels, self.in_channels, 1, self.kernel_size)
        y4 = x4.conv2d(w4, self.bias, (1, self.stride), (0, self.padding))
        _, Cout, _, L_out = y4.shape
        return y4.reshape(N, Cout, L_out)

    def __repr__(self) -> str:
        return (
            f"Conv1d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None})"
        )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShapeMismatchError(f"{name} must be a positive int, got {value!r}.")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShapeMismatchError(f"{name} must be a non-negative int, got {value!r}.")
    return value


__all__ = ["Conv2d", "Conv1d"]
