"""Stochastic gradient descent."""

from __future__ import annotations

import math
from typing import Iterable

from ..backend import get_backend
from ..exceptions import OptimizerError
from ..nn.parameter import Parameter
from .optimizer import Optimizer


class SGD(Optimizer):
    """Plain stochastic gradient descent: `parameter -= lr * parameter.grad`.

    No momentum, weight decay, or learning-rate schedule -- see
    `docs/architecture/optimization.md` for the scope of this milestone.
    Parameters with no gradient (`.grad is None`, e.g. unused in the current
    forward pass) are left unchanged rather than treated as an error.

    `step()` dispatches to `Backend.sgd_step()` (Milestone 10) -- a direct,
    in-place storage update on whichever device the parameter lives on
    (`np.ndarray` arithmetic for CPU, a real CUDA kernel for CUDA), rather
    than going through Tensor's differentiable operators. It is a plain
    state change either way: it does not attach a `grad_fn` or otherwise
    extend the autograd graph, and a CUDA parameter is updated without any
    host round-trip.
    """

    def __init__(self, parameters: "Iterable[Parameter]", lr: float):
        super().__init__(parameters)
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or math.isnan(lr) or lr <= 0:
            raise OptimizerError(f"SGD requires a positive learning rate, got {lr!r}.")
        self.lr = float(lr)

    def step(self) -> None:
        for param in self.parameters:
            if param.grad is None:
                continue
            backend = get_backend(param._device)
            param._data = backend.sgd_step(param._data, param.grad._data, self.lr)


__all__ = ["SGD"]
