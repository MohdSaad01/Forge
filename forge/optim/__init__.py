"""Forge's optimizer package (Milestone 4).

An `Optimizer` owns a flat list of `Parameter`s (typically
`model.parameters()`) and updates them in place from their `.grad` --
computed entirely by the existing Tensor/autograd system, never by the
optimizer itself. See `docs/architecture/optimization.md`.
"""

from .optimizer import Optimizer
from .sgd import SGD

__all__ = ["Optimizer", "SGD"]
