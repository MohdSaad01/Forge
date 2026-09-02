"""Task-evaluation metrics, kept separate from loss functions.

A `Loss` (`forge.nn.Loss`) is the differentiable training objective; a
`Metric` is a non-differentiable measurement computed for reporting only --
it never participates in the autograd graph and never modifies model
parameters. `Metric.update()` is called once per batch and accumulates raw
running totals (not per-batch means), so `compute()` after a full epoch
reflects every sample seen, correctly weighted even when batches have
different sizes. See `docs/architecture/training-engine.md`.

As of Milestone 12, `_as_numpy` (below) transfers a non-CPU `Tensor` to CPU
before reading it, so every built-in metric works unchanged for CUDA
predictions/targets -- each is a small, non-differentiable NumPy reduction
that was never a good fit for a dedicated CUDA kernel (see
`docs/architecture/training-engine.md`'s **CUDA metrics** section). This is
a one-way, read-only transfer of already-computed values for reporting; it
never feeds back into training and never touches `CPUBackend`'s compute
methods (`Tensor.to()` calls `CPUBackend.from_array`, a transfer primitive,
not a compute one).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import TrainerError
from ..tensor.tensor import Tensor


def _as_numpy(value: "Tensor | Any") -> np.ndarray:
    if isinstance(value, Tensor):
        return value.to("cpu").numpy()
    return np.asarray(value)


class Metric:
    """Base class for a batch-aggregating evaluation metric.

    Subclasses accumulate state in `update()` across batches and reduce it
    in `compute()`; `reset()` clears accumulated state at the start of a new
    epoch/phase. The base methods raise `TrainerError`, matching the
    "must implement" pattern already used by `Module.forward`/`Loss.forward`/
    `Optimizer.step`/`Dataset.__getitem__`.
    """

    name: str = "metric"

    def reset(self) -> None:
        raise TrainerError(f"{type(self).__name__} does not implement reset().")

    def update(self, prediction: "Tensor | Any", target: "Tensor | Any") -> None:
        raise TrainerError(f"{type(self).__name__} does not implement update().")

    def compute(self) -> float:
        raise TrainerError(f"{type(self).__name__} does not implement compute().")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class MeanSquaredError(Metric):
    """Mean squared error over every element seen: `mean((prediction - target)^2)`.

    `prediction`/`target` must have exactly the same shape (matching
    `MSELoss`'s convention). Accumulates a running `(sum of squared error,
    element count)` pair rather than averaging per-batch means, so batches
    of unequal size contribute proportionally to the final `compute()`
    value.
    """

    name = "mse"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum_squared_error = 0.0
        self._count = 0

    def update(self, prediction: "Tensor | Any", target: "Tensor | Any") -> None:
        pred = _as_numpy(prediction)
        targ = _as_numpy(target)
        if pred.shape != targ.shape:
            raise TrainerError(
                f"'{self.name}' requires prediction and target to have the same "
                f"shape, got {pred.shape} and {targ.shape}."
            )
        diff = pred.astype(np.float64) - targ.astype(np.float64)
        self._sum_squared_error += float(np.sum(diff * diff))
        self._count += diff.size

    def compute(self) -> float:
        if self._count == 0:
            raise TrainerError(f"'{self.name}'.compute() called with no samples seen.")
        return self._sum_squared_error / self._count


class MeanAbsoluteError(Metric):
    """Mean absolute error over every element seen: `mean(|prediction - target|)`.

    Same shape requirement and weighted-aggregation behavior as
    `MeanSquaredError`.
    """

    name = "mae"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum_absolute_error = 0.0
        self._count = 0

    def update(self, prediction: "Tensor | Any", target: "Tensor | Any") -> None:
        pred = _as_numpy(prediction)
        targ = _as_numpy(target)
        if pred.shape != targ.shape:
            raise TrainerError(
                f"'{self.name}' requires prediction and target to have the same "
                f"shape, got {pred.shape} and {targ.shape}."
            )
        diff = pred.astype(np.float64) - targ.astype(np.float64)
        self._sum_absolute_error += float(np.sum(np.abs(diff)))
        self._count += diff.size

    def compute(self) -> float:
        if self._count == 0:
            raise TrainerError(f"'{self.name}'.compute() called with no samples seen.")
        return self._sum_absolute_error / self._count


class Accuracy(Metric):
    """Classification accuracy: fraction of predicted classes matching target.

    `prediction` is `(batch_size, num_classes)` scores/logits -- the
    predicted class is `argmax` over the class axis, matching
    `CrossEntropyLoss`'s convention. `target` is `(batch_size,)` integer
    class indices. Accumulates a running `(correct count, sample count)`
    pair so batches of unequal size are weighted correctly.
    """

    name = "accuracy"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, prediction: "Tensor | Any", target: "Tensor | Any") -> None:
        pred = _as_numpy(prediction)
        targ = _as_numpy(target)
        if pred.ndim != 2:
            raise TrainerError(
                f"'{self.name}' expects prediction of shape (batch_size, "
                f"num_classes), got shape {pred.shape}."
            )
        if targ.ndim != 1 or targ.shape[0] != pred.shape[0]:
            raise TrainerError(
                f"'{self.name}' expects target of shape (batch_size,) = "
                f"({pred.shape[0]},), got shape {targ.shape}."
            )
        predicted_classes = np.argmax(pred, axis=1)
        self._correct += int(np.sum(predicted_classes == targ))
        self._total += int(targ.shape[0])

    def compute(self) -> float:
        if self._total == 0:
            raise TrainerError(f"'{self.name}'.compute() called with no samples seen.")
        return self._correct / self._total


__all__ = ["Metric", "MeanSquaredError", "MeanAbsoluteError", "Accuracy"]
