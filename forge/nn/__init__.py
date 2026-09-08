"""Forge's neural-network composition layer (Milestone 3 + 4).

Built entirely on the existing Tensor/autograd system: `Module`/`Parameter`
provide composition and discovery, while every numerical computation in
`Linear`/`ReLU`/the loss functions runs through ordinary differentiable
Tensor operations. See `docs/architecture/modules.md` and
`docs/architecture/optimization.md`.
"""

from .activation import ReLU, Tanh
from .batchnorm import BatchNorm2d
from .container import Sequential
from .conv import Conv1d, Conv2d
from .dropout import Dropout
from .embedding import Embedding
from .flatten import Flatten
from .linear import Linear
from .loss import CrossEntropyLoss, Loss, MSELoss
from .module import Module
from .parameter import Parameter
from .pooling import MaxPool1d, MaxPool2d
from .rnn import RNNCell

__all__ = [
    "Module", "Parameter", "Linear", "ReLU", "Tanh", "Conv2d", "MaxPool2d",
    "Conv1d", "MaxPool1d",
    "Sequential", "Flatten", "Dropout", "RNNCell", "BatchNorm2d", "Embedding",
    "Loss", "MSELoss", "CrossEntropyLoss",
]
