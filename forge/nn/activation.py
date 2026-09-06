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


class Tanh(Module):
    """`tanh(x)`, elementwise (Milestone 50: added for `nn.RNNCell`'s recurrence).

    Delegates to `Tensor.tanh()` (backed by `Backend.tanh`), so it has no
    parameters of its own -- the same shape as `ReLU` above.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


__all__ = ["ReLU", "Tanh"]
