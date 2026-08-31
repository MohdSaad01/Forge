"""Loss functions: differentiable Tensor-valued comparisons between predictions and targets.

Built entirely from ordinary Tensor operations (`-`, `*`, `.sum()`, and the
`.exp()`/`.log()` primitives added in this milestone), so gradients flow
through the existing autograd graph -- a loss attaches no backward math of
its own. See `docs/architecture/optimization.md`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import LossError
from ..tensor.tensor import Tensor


class Loss:
    """Base class for Forge loss functions.

    A `Loss` is a composable, callable component -- `loss_fn(prediction, target)`
    -- mirroring `Module`'s `__call__`-delegates-to-`forward()` shape. It is
    deliberately not a `Module` itself: a loss owns no parameters and is not
    part of a model's module tree.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise LossError(f"{type(self).__name__} does not implement forward().")


class MSELoss(Loss):
    """Mean squared error: `mean((prediction - target)^2)`.

    `prediction` and `target` must have exactly the same shape -- the mean is
    taken over every element of that shape (e.g. for a `(batch, features)`
    prediction, this averages over both batch and feature dimensions).
    """

    def forward(self, prediction: Tensor, target: "Tensor | Any") -> Tensor:
        if not isinstance(target, Tensor):
            target = Tensor(target, dtype=prediction.dtype, device=prediction.device)
        if prediction.shape != target.shape:
            raise LossError(
                "MSELoss requires prediction and target to have the same shape, "
                f"got prediction shape {prediction.shape} and target shape {target.shape}."
            )

        diff = prediction - target
        squared = diff * diff
        n = int(np.prod(prediction.shape)) if prediction.shape else 1
        return squared.sum() * (1.0 / n)


class CrossEntropyLoss(Loss):
    """Numerically stable multiclass cross-entropy for classification.

    `logits`: Tensor of shape `(batch_size, num_classes)` -- unnormalized,
    per-class scores.
    `target`: Tensor or array-like of shape `(batch_size,)` containing
    integer class indices in `[0, num_classes)`.

    Computes `-mean(log_softmax(logits)[i, target[i]])`. `log_softmax` is
    computed via the log-sum-exp trick (subtracting each row's max logit,
    which cancels out analytically, before exponentiating) so it never
    overflows regardless of the input scale. The per-row max is a plain
    NumPy constant (not differentiated), which is exact -- the identity
    `log_softmax(x - c) == log_softmax(x)` holds for any `c`, so treating it
    as a constant does not change the gradient. Target selection uses a
    one-hot multiply (also a plain, non-differentiable constant) rather than
    a new indexing/gather primitive, so this loss needs only the `.exp()`
    and `.log()` Tensor primitives added in this milestone, plus existing
    ops.
    """

    def forward(self, logits: Tensor, target: "Tensor | Any") -> Tensor:
        if logits.ndim != 2:
            raise LossError(
                "CrossEntropyLoss expects logits of shape (batch_size, num_classes), "
                f"got shape {logits.shape}."
            )
        batch_size, num_classes = logits.shape

        target_array = target.numpy() if isinstance(target, Tensor) else np.asarray(target)
        if target_array.ndim != 1 or target_array.shape[0] != batch_size:
            raise LossError(
                f"CrossEntropyLoss expects target of shape (batch_size,) = ({batch_size},), "
                f"got shape {target_array.shape}."
            )
        if not np.issubdtype(target_array.dtype, np.integer):
            raise LossError(
                "CrossEntropyLoss expects integer class-index targets, got dtype "
                f"'{target_array.dtype}'."
            )
        if target_array.size and (
            int(target_array.min()) < 0 or int(target_array.max()) >= num_classes
        ):
            raise LossError(
                f"CrossEntropyLoss target values must be valid class indices in "
                f"[0, {num_classes}), got values in "
                f"[{int(target_array.min())}, {int(target_array.max())}]."
            )

        logits_np = logits.numpy()
        max_vals = np.max(logits_np, axis=1, keepdims=True)
        shift = Tensor(max_vals, dtype=logits.dtype, device=logits.device)
        shifted = logits - shift

        log_sum_exp = shifted.exp().sum(axis=1, keepdims=True).log()
        log_probs = shifted - log_sum_exp

        one_hot = np.zeros((batch_size, num_classes), dtype=logits_np.dtype)
        one_hot[np.arange(batch_size), target_array] = 1.0
        one_hot_t = Tensor(one_hot, dtype=logits.dtype, device=logits.device)

        picked = (log_probs * one_hot_t).sum(axis=1)
        return picked.sum() * (-1.0 / batch_size)


__all__ = ["Loss", "MSELoss", "CrossEntropyLoss"]
