"""Forge's neural-network composition layer (Milestone 3).

Built entirely on the existing Tensor/autograd system: `Module`/`Parameter`
provide composition and discovery, while every numerical computation in
`Linear`/`ReLU` runs through ordinary differentiable Tensor operations. See
`docs/architecture/modules.md`.
"""

from .activation import ReLU
from .linear import Linear
from .module import Module
from .parameter import Parameter

__all__ = ["Module", "Parameter", "Linear", "ReLU"]
