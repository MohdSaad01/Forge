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


class ModuleError(ForgeError):
    """Raised for invalid ``Module``/``Parameter`` composition or configuration.

    Examples: assigning a `Parameter` or child `Module` before calling
    `Module.__init__` (`super().__init__()`), or invoking a module whose
    `forward()` has not been implemented.
    """


class LossError(ForgeError):
    """Raised for invalid loss-function inputs.

    Examples: a prediction/target shape mismatch, a classification target
    with the wrong shape or a non-integer dtype, an out-of-range class
    index, or invoking a loss whose `forward()` has not been implemented.
    """


class OptimizerError(ForgeError):
    """Raised for invalid optimizer configuration or usage.

    Examples: a non-positive learning rate, or invoking an optimizer whose
    `step()` has not been implemented.
    """
