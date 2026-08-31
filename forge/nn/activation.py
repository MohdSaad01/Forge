"""Activation modules."""

from __future__ import annotations

from ..tensor.tensor import Tensor
from .module import Module


class ReLU(Module):
    """`relu(x) = max(x, 0)`, elementwise.

    Delegates to `Tensor.relu()` (backed by `Backend.relu`), so it has no
    parameters of its own and participates in autograd purely through the
    Tensor-level primitive -- no gradient math lives in this class.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


__all__ = ["ReLU"]
