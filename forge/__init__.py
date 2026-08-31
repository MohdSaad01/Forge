"""Forge: a from-scratch deep-learning framework.

Milestone 1 established the Tensor abstraction and the CPU execution
boundary. Milestone 2 added gradient tracking and reverse-mode autodiff on
top of that Tensor. Milestone 3 added the `nn` module/parameter composition
layer built on top of both. Milestone 4 added loss functions (`nn.MSELoss`,
`nn.CrossEntropyLoss`) and the `optim` optimizer package (`optim.SGD`) that
consumes `Parameter` gradients to update model state. Milestone 5 added the
`data` package (`Dataset`, `TensorDataset`, `DataLoader`, transforms) for
representing training data and producing model-ready batches, independent of
the model/loss/optimizer stack. Milestone 6 adds the `training` package
(`Trainer`, metrics, `TrainingHistory`) that orchestrates all of the above
into a reusable training/evaluation workflow, plus `no_grad()` -- a minimal
autograd extension that suspends graph construction during evaluation.
"""

from . import data, nn, optim, random, training
from .autograd import no_grad
from .backend.device import Device
from .exceptions import (
    DataError,
    ForgeError,
    GradientStateError,
    LossError,
    ModuleError,
    OptimizerError,
    ShapeMismatchError,
    TrainerError,
    UnsupportedDeviceError,
    UnsupportedDTypeError,
)
from .tensor import DEFAULT_DTYPE, DType, Tensor

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "DType",
    "DEFAULT_DTYPE",
    "Device",
    "ForgeError",
    "ShapeMismatchError",
    "UnsupportedDTypeError",
    "UnsupportedDeviceError",
    "GradientStateError",
    "ModuleError",
    "LossError",
    "OptimizerError",
    "DataError",
    "TrainerError",
    "no_grad",
    "nn",
    "optim",
    "random",
    "data",
    "training",
]
