"""Forge's neural-network composition layer (Milestone 3 + 4).

Built entirely on the existing Tensor/autograd system: `Module`/`Parameter`
provide composition and discovery, while every numerical computation in
`Linear`/`ReLU`/the loss functions runs through ordinary differentiable
Tensor operations. See `docs/architecture/modules.md` and
`docs/architecture/optimization.md`.
"""

from .activation import ReLU
from .conv import Conv2d
from .linear import Linear
from .loss import CrossEntropyLoss, Loss, MSELoss
from .module import Module
from .parameter import Parameter
from .pooling import MaxPool2d

__all__ = [
    "Module", "Parameter", "Linear", "ReLU", "Conv2d", "MaxPool2d",
    "Loss", "MSELoss", "CrossEntropyLoss",
]
