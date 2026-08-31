"""Forge: a from-scratch deep-learning framework.

Milestone 1 established the Tensor abstraction and the CPU execution
boundary. Milestone 2 added gradient tracking and reverse-mode autodiff on
top of that Tensor. Milestone 3 added the `nn` module/parameter composition
layer built on top of both. Milestone 4 adds loss functions (`nn.MSELoss`,
`nn.CrossEntropyLoss`) and the `optim` optimizer package (`optim.SGD`) that
consumes `Parameter` gradients to update model state.
"""

from . import nn, optim, random
from .backend.device import Device
from .exceptions import (
    ForgeError,
    GradientStateError,
    LossError,
    ModuleError,
    OptimizerError,
    ShapeMismatchError,
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
    "nn",
    "optim",
    "random",
]
