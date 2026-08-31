"""The base optimizer abstraction.

Responsibility boundary: autograd computes gradients; the optimizer only
consumes them (`step()`) and clears them (`zero_grad()`). It never triggers
a forward/backward pass and never computes a gradient itself.
"""

from __future__ import annotations

from typing import Iterable

from ..exceptions import OptimizerError
from ..nn.parameter import Parameter


class Optimizer:
    """Owns a list of `Parameter`s and updates them from their `.grad`.

    Constructed from an iterable of `Parameter`s (typically
    `model.parameters()`), not a model instance -- this keeps the optimizer
    independent of `Module` internals.
    """

    def __init__(self, parameters: "Iterable[Parameter]"):
        self.parameters: list[Parameter] = list(parameters)

    def zero_grad(self) -> None:
        """Clear every parameter's accumulated gradient.

        Delegates to `Parameter.zero_grad()` (inherited from `Tensor`) for
        each owned parameter rather than duplicating gradient-clearing logic.
        """
        for param in self.parameters:
            param.zero_grad()

    def step(self) -> None:
        raise OptimizerError(f"{type(self).__name__} does not implement step().")


__all__ = ["Optimizer"]
