"""Loss functions: differentiable Tensor-valued comparisons between predictions and targets.

Built entirely from ordinary Tensor operations (`-`, `*`, `.sum()`, and the
`.exp()`/`.log()` primitives added in Milestone 8), so gradients flow through
the existing autograd graph -- a loss attaches no backward math of its own.
See `docs/architecture/optimization.md`.

As of Milestone 12, `MSELoss` is CUDA-compatible: it composes only `-`
(exact-shape sub), `*` (exact-shape mul), and `.sum()` (full reduction,
`axis=None`) -- every one of which `CUDABackend` already implements forward
*and* backward (Milestones 8-10). No new CUDA kernel was needed; see
`docs/architecture/cuda-backend.md`'s **CUDA losses** section.

As of Milestone 14, `CrossEntropyLoss` is also CUDA-compatible (no
`CUDACrossEntropyLoss` subclass, no second autograd engine). As of
Milestone 31, its numerically-stable log-sum-exp computation is a single
fused `Backend.cross_entropy`/`cross_entropy_backward` primitive
(`Tensor.cross_entropy()`, `forge/tensor/tensor.py`) rather than ~9 composed
Tensor ops -- Milestone 21/31 profiling found the composed version costing
more wall-clock time on CUDA than the entire M20 CNN's forward pass, purely
from per-kernel-launch dispatch overhead on a tensor too small for that cost
to be hidden by actual arithmetic. See `docs/performance/pipeline-profiling.md`
and that method's docstring for the full justification; this class's own
validation (shape/dtype/target-range checks, below) is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend import get_backend
from ..exceptions import LossError, UnsupportedDeviceError
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

    Device-agnostic by construction: `prediction`/`target` on different
    devices is rejected by `-` itself (`Tensor._binary_op`'s existing
    device-consistency check), never silently reconciled. Works unmodified
    on CUDA as of Milestone 12 -- see the module docstring.
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
        # `Tensor(1.0 / n, ...)` is built with `prediction`'s own dtype
        # explicitly, rather than multiplying by a bare Python float (which
        # `Tensor._coerce` would infer as float32 regardless of
        # `prediction`'s dtype): CPU tolerates that mismatch by upcasting
        # via NumPy, but `CUDABackend`'s elementwise ops require the two
        # operands to already share a dtype, so a float64 CUDA loss would
        # otherwise raise `CUDAError` here.
        scale = Tensor(1.0 / n, dtype=prediction.dtype, device=prediction.device)
        return squared.sum() * scale


class CrossEntropyLoss(Loss):
    """Numerically stable multiclass cross-entropy for classification.

    `logits`: Tensor of shape `(batch_size, num_classes)` -- unnormalized,
    per-class scores. `target`: Tensor or array-like of shape `(batch_size,)`
    containing integer class indices in `[0, num_classes)`.

    Computes `-mean(log_softmax(logits)[i, target[i]])`. `log_softmax` is
    computed via the log-sum-exp trick (subtracting each row's max logit,
    which cancels out analytically, before exponentiating) so it never
    overflows regardless of the input scale. The per-row max is a plain
    backend-computed constant (not differentiated), which is exact -- the
    identity `log_softmax(x - c) == log_softmax(x)` holds for any per-row `c`,
    so treating it as a constant does not change the gradient. Target
    selection uses a one-hot multiply (also a plain, non-differentiable
    constant) rather than a new indexing/gather primitive, so this loss needs
    only the `.exp()`/`.log()`/`.sum(axis=1, ...)` Tensor primitives plus
    ordinary elementwise ops.

    Works unmodified on CUDA as of Milestone 14 -- see
    `docs/architecture/cuda-backend.md`'s **CUDA CrossEntropyLoss** section.
    `logits` and `target` (when `target` is itself a Tensor) must already be
    on the same device -- see **Loss device validation** in that section;
    this is never silently reconciled.
    """

    def forward(self, logits: Tensor, target: "Tensor | Any") -> Tensor:
        if logits.ndim != 2:
            raise LossError(
                "CrossEntropyLoss expects logits of shape (batch_size, num_classes), "
                f"got shape {logits.shape}."
            )
        batch_size, num_classes = logits.shape

        if isinstance(target, Tensor):
            if target.device != logits.device:
                raise UnsupportedDeviceError(
                    "CrossEntropyLoss requires target on the same device as logits; got "
                    f"logits on device '{logits.device}' and target on device "
                    f"'{target.device}'. Move target explicitly with "
                    f".to('{logits.device}') first -- this is never done automatically."
                )
            # A host materialization of the (small, integer) target indices
            # for validation below -- exactly the same kind of read-only
            # transfer `Trainer`/`Metric`/persistence already use for
            # non-computational purposes (see `docs/architecture/
            # cuda-backend.md`'s **No CPU fallback** section). The actual
            # loss *computation* (log-sum-exp, the gather, the reduction)
            # always runs through backend dispatch below, never here.
            target_array = get_backend(target.device).to_numpy(target._data)
        else:
            target_array = np.asarray(target)

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

        # Milestone 31: dispatches to `Tensor.cross_entropy()`, a fused
        # `Backend` primitive (`max_axis1` + log-sum-exp + gather + mean in
        # one or two real kernel launches on CUDA, one vectorized NumPy call
        # on CPU) rather than composing ~9 separate Tensor ops -- see that
        # method's docstring and `docs/performance/pipeline-profiling.md`
        # for why. `target_array` (already fully validated above, including
        # the class-index range check) only ever needs to cross the
        # host/device boundary as small integer indices now -- the one-hot
        # float matrix the pre-Milestone-31 implementation built and
        # transferred here is gone entirely.
        target_tensor = Tensor(target_array.astype(np.int64, copy=False), device=logits.device)
        return logits.cross_entropy(target_tensor)


__all__ = ["Loss", "MSELoss", "CrossEntropyLoss"]
