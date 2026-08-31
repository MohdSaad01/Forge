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


class DataError(ForgeError):
    """Raised for invalid dataset/DataLoader/transform configuration or usage.

    Examples: mismatched sample counts between tensors in a `TensorDataset`,
    an out-of-range or invalid dataset index, an invalid `DataLoader` batch
    size or configuration, `random_split` lengths that do not sum to the
    dataset size, a transform applied to an incompatible sample, or invoking
    a `Dataset`/`Transform` base method that has not been implemented.
    """


class TrainerError(ForgeError):
    """Raised for invalid Trainer/metric configuration or usage.

    Examples: a missing/invalid model, loss, or optimizer; an unsupported
    device; a non-positive epoch count; a DataLoader that yields no batches;
    a batch that is not a `(features, target)` tuple; a metric given
    mismatched prediction/target shapes; or a metric's `compute()` called
    with no samples seen.
    """
