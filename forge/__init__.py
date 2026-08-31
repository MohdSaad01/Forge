"""Forge: a from-scratch deep-learning framework.

Milestone 1 exposes the Tensor abstraction and the CPU execution boundary.
"""

from .backend.device import Device
from .exceptions import ForgeError, ShapeMismatchError, UnsupportedDeviceError, UnsupportedDTypeError
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
]
