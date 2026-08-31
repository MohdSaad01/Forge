"""Forge: a from-scratch deep-learning framework.

Milestone 1 established the Tensor abstraction and the CPU execution
boundary. Milestone 2 adds gradient tracking and reverse-mode autodiff on
top of that Tensor.
"""

from .backend.device import Device
from .exceptions import (
    ForgeError,
    GradientStateError,
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
]
