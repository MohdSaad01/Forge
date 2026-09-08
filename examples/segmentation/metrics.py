"""Pixel accuracy and IoU: dense-prediction evaluation metrics (Milestone 64).

`forge.training.metrics.Metric` is deliberately designed for exactly this
kind of extension (see its module docstring: "a non-differentiable
measurement computed for reporting only"). Every built-in `Metric`
(`Accuracy`, `MeanSquaredError`, `MeanAbsoluteError`) assumes either a
`(batch, num_classes)` classification shape or a plain elementwise
comparison -- none fits a per-pixel binary segmentation mask, so this
example subclasses `Metric` locally rather than requiring a framework
change; `Trainer(..., metrics=[...])` accepts any `Metric` instance
regardless of where it is defined.

`prediction` is the model's raw (unbounded, no final activation -- see
`model.py`) `(N, 1, H, W)` output; it is thresholded at `0.5` here (not
inside the model or the loss) to obtain a binary predicted mask, matching
`train.py`'s own evaluation convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge.exceptions import TrainerError
from forge.tensor.tensor import Tensor
from forge.training.metrics import Metric

_THRESHOLD = 0.5


def _as_binary_numpy(value: "Tensor | Any") -> np.ndarray:
    array = value.to("cpu").numpy() if isinstance(value, Tensor) else np.asarray(value)
    if array.ndim != 4 or array.shape[1] != 1:
        raise TrainerError(
            f"expected a (batch, 1, H, W) tensor for segmentation metrics, got shape {array.shape}."
        )
    return array


class PixelAccuracy(Metric):
    """Fraction of pixels whose thresholded prediction matches the binary target mask."""

    name = "pixel_accuracy"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, prediction: "Tensor | Any", target: "Tensor | Any") -> None:
        pred = _as_binary_numpy(prediction) >= _THRESHOLD
        targ = _as_binary_numpy(target) >= _THRESHOLD
        if pred.shape != targ.shape:
            raise TrainerError(
                f"'{self.name}' requires prediction and target to have the same shape, "
                f"got {pred.shape} and {targ.shape}."
            )
        self._correct += int(np.sum(pred == targ))
        self._total += int(pred.size)

    def compute(self) -> float:
        if self._total == 0:
            raise TrainerError(f"'{self.name}'.compute() called with no samples seen.")
        return self._correct / self._total


class IoU(Metric):
    """Intersection-over-union of the thresholded foreground prediction against the target mask.

    Accumulates a running `(intersection, union)` pixel count across every
    batch (not a per-batch mean of per-image IoU), so batches of unequal
    size, and images with no foreground pixels at all, are handled without a
    special case. If the accumulated union is `0` (no foreground pixels
    predicted or present across the whole evaluation), `compute()` returns
    `1.0` -- correctly-predicted "nothing here" agreement, not an undefined
    `0/0`.
    """

    name = "iou"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._intersection = 0
        self._union = 0

    def update(self, prediction: "Tensor | Any", target: "Tensor | Any") -> None:
        pred = _as_binary_numpy(prediction) >= _THRESHOLD
        targ = _as_binary_numpy(target) >= _THRESHOLD
        if pred.shape != targ.shape:
            raise TrainerError(
                f"'{self.name}' requires prediction and target to have the same shape, "
                f"got {pred.shape} and {targ.shape}."
            )
        self._intersection += int(np.sum(pred & targ))
        self._union += int(np.sum(pred | targ))

    def compute(self) -> float:
        if self._union == 0:
            return 1.0
        return self._intersection / self._union


__all__ = ["PixelAccuracy", "IoU"]
