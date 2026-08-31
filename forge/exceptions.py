"""Forge's error hierarchy.

Kept small and explicit: callers should be able to catch ``ForgeError`` for
any Forge-specific failure, or a specific subtype when they care about the
distinction between a bad shape, a bad dtype, and a bad device.
"""


class ForgeError(Exception):
    """Base class for all Forge-specific errors."""


class ShapeMismatchError(ForgeError):
    """Raised when an operation is applied to incompatible tensor shapes."""


class UnsupportedDTypeError(ForgeError):
    """Raised when a dtype is not one Forge implements."""


class UnsupportedDeviceError(ForgeError):
    """Raised when a device is not recognized or not yet executable."""


class GradientStateError(ForgeError):
    """Raised for invalid autograd usage or an inconsistent gradient state.

    Examples: calling ``backward()`` on a non-scalar tensor without an
    explicit upstream gradient, calling ``backward()`` on a tensor that does
    not require grad, requesting ``requires_grad`` on a non-floating dtype,
    or calling ``backward()`` again on a graph already freed by a previous
    call.
    """
