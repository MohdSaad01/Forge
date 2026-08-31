"""Fully connected (affine) layer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import random as forge_random
from ..exceptions import ShapeMismatchError
from ..tensor.dtype import DEFAULT_DTYPE
from ..tensor.tensor import Tensor
from .module import Module
from .parameter import Parameter


class Linear(Module):
    """`y = x @ weight + bias`, with `weight` shape `(in_features, out_features)`.

    Accepts a single sample `x` of shape `(in_features,)` or a batch of shape
    `(batch, in_features)` -- both are ordinary matmul shapes already
    supported by `Tensor.__matmul__`, so no batching logic is needed here.

    ## Initialization
    Both `weight` and `bias` are drawn from
    `Uniform(-1/sqrt(in_features), 1/sqrt(in_features))`. This keeps the
    initial output scale roughly independent of `in_features` (each of the
    `in_features` summed terms has variance `~1/(3*in_features)`, so the sum
    stays `O(1)`), which is the same bound PyTorch's default `nn.Linear`
    init reduces to. Draws come from `forge.random.default_generator()`
    unless a `generator` is passed explicitly, so seeding
    `forge.random.seed(...)` makes construction deterministic.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ShapeMismatchError(
                f"Linear requires positive in_features/out_features, got "
                f"in_features={in_features}, out_features={out_features}."
            )

        self.in_features = in_features
        self.out_features = out_features

        param_dtype = dtype if dtype is not None else DEFAULT_DTYPE
        rng = generator if generator is not None else forge_random.default_generator()
        bound = 1.0 / math.sqrt(in_features)

        weight_data = rng.uniform(-bound, bound, size=(in_features, out_features))
        self.weight = Parameter(weight_data, dtype=param_dtype, device=device)

        if bias:
            bias_data = rng.uniform(-bound, bound, size=(out_features,))
            self.bias = Parameter(bias_data, dtype=param_dtype, device=device)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in (1, 2):
            raise ShapeMismatchError(
                f"Linear expects a 1D (in_features,) or 2D (batch, in_features) "
                f"input, got shape {x.shape}."
            )
        if x.shape[-1] != self.in_features:
            raise ShapeMismatchError(
                f"Linear(in_features={self.in_features}) cannot accept input of "
                f"shape {x.shape}: last dimension must be {self.in_features}."
            )

        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )


__all__ = ["Linear"]
