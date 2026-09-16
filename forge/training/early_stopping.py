"""`EarlyStopping` (Milestone 100): stop `Trainer.fit()` when a monitored
validation quantity stops improving, optionally restoring the best-seen
model state.

Milestone 98 measured a real overfitting curve on `examples/tabular_diabetes`:
validation loss reaches its minimum around epoch 9-14 while training loss
keeps falling through epoch 60. Milestone 99 exposed `TrainingResult` so a
caller can see that after the fact; this milestone lets `Trainer.fit()` act
on it *during* training instead of always running every requested epoch:

```python
result = forge.train(
    model, train_dataset,
    loss=loss_fn, optimizer=optimizer, epochs=60,
    validation_dataset=val_dataset,
    early_stopping=forge.training.EarlyStopping(patience=5, restore_best=True),
)
result.stopped_early          # True if patience was exhausted before epoch 60
result.best_epoch             # the global epoch number with the best val_loss
result.best_monitored_value   # that epoch's val_loss
```

## Monitor

`monitor` (default `"val_loss"`) names a validation quantity, always
prefixed `"val_"` -- either the literal `"val_loss"` or `"val_<metric
name>"` for any `Metric` passed to `Trainer(..., metrics=[...])` (e.g.
`"val_accuracy"` for `Accuracy()`). There is no expression language beyond
this one lookup: a monitor naming a metric that was not actually supplied
raises `TrainerError` the first time `fit()` tries to read it, naming what
*is* available.

## Mode

Whether `monitor` improves by getting smaller or larger is not inferred --
`mode="min"` (the default, correct for `"val_loss"` and any other error/loss
-shaped metric) or `mode="max"` (for `"val_accuracy"` and similar) is an
explicit, required choice for anything other than the default, rather than
a name-sniffing heuristic (`docs/architecture/training-engine.md`'s own
precedent for avoiding this kind of guessing -- see M87's `task=`).

## Patience and min_delta

`patience` (default 5) is the number of *consecutive* validation epochs
that fail to improve before training stops -- see `Trainer.fit()`'s own
docstring for the exact off-by-one definition and
`docs/architecture/training-engine.md` for a worked example. `min_delta`
(default `0.0`) is the minimum change that counts as an improvement:

```text
mode="min":  new_value < best_value - min_delta
mode="max":  new_value > best_value + min_delta
```

The very first validation result always establishes the initial best value
(there is nothing to compare it against yet).

## restore_best

When `True` (the default), `Trainer.fit()` restores the model's
parameters/buffers to whatever they were right after the best-scoring
epoch, once the loop ends -- whether it ended by exhausting `patience` or by
reaching `epochs`. When `False`, the model is left exactly as the last
completed epoch's training left it, and `fit()`'s only behavioral change is
stopping the loop early.

Restoration reuses `Module.named_parameters()`/`named_buffers()` (the exact
traversal `save_model()`/`save_checkpoint()` already use to walk a module
tree) plus each Tensor's own backend `to_numpy()`/`from_array()` transfer
primitives -- a plain host-memory copy taken after every new best epoch,
restored in place at the end. This is deliberately not a second
serialization system: nothing is written to disk, and no new archive format
is introduced (`docs/architecture/persistence.md` is unchanged). Only the
best snapshot is ever held in memory (a stale one is dropped, not
accumulated) -- one clone of the model's numeric state, not one per epoch.

## Optimizer state is not restored

Only `Module` parameters/buffers are snapshotted and restored -- the
optimizer's own per-parameter state (e.g. Adam's `m`/`v` moments) keeps
whatever it accumulated by the time the loop actually stopped. This matches
the scope `forge.train()`/`forge.train_and_save()` already documented for
themselves (`docs/architecture/training-engine.md`'s **What this does not
add** section): neither function has ever produced or consumed a checkpoint,
so there was no optimizer/model pairing contract to preserve across an
early-stopped run in the first place. A caller who checkpoints a
`Trainer` after an early-stopped `fit()` call (via
`Trainer.save_checkpoint()`/`TrainingSession`) gets whatever the model
currently holds -- the restored best parameters when `restore_best=True` --
paired with the optimizer's state as of the *last* epoch that actually ran;
resuming continued training from that checkpoint is a valid but
non-continuous restart (the optimizer's momentum/moments were computed
against a different point in the parameter trajectory than the parameters
they are now paired with), not a mismatch `load_checkpoint()` itself detects
or rejects. This is a documented boundary, not a defect -- see
`docs/architecture/training-engine.md`'s **Early stopping** section.

## Requires validation data

`early_stopping=` with no `validation_loader`/`validation_dataset` raises
`TrainerError` immediately, before any epoch runs -- there is no quantity to
monitor otherwise, and Forge never silently substitutes training loss for a
missing validation signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..backend import get_backend
from ..exceptions import TrainerError

if TYPE_CHECKING:
    from ..nn.module import Module
    from .trainer import EpochResult

_VALID_MODES = ("min", "max")


@dataclass
class EarlyStopping:
    """Early-stopping configuration for `Trainer.fit()`/`forge.train()`.

    See this module's own docstring for the full contract (monitor, mode,
    patience, min_delta, restore_best, and what is/is not restored).
    """

    monitor: str = "val_loss"
    patience: int = 5
    min_delta: float = 0.0
    restore_best: bool = True
    mode: str = "min"

    def __post_init__(self) -> None:
        if not isinstance(self.monitor, str) or not self.monitor.startswith("val_") or self.monitor == "val_":
            raise TrainerError(
                "EarlyStopping monitor must be a validation quantity name starting with "
                f"'val_' (e.g. 'val_loss', 'val_accuracy'), got {self.monitor!r}."
            )
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1:
            raise TrainerError(f"EarlyStopping patience must be a positive int, got {self.patience!r}.")
        if isinstance(self.min_delta, bool) or not isinstance(self.min_delta, (int, float)) or self.min_delta < 0:
            raise TrainerError(f"EarlyStopping min_delta must be a number >= 0, got {self.min_delta!r}.")
        if self.mode not in _VALID_MODES:
            raise TrainerError(f"EarlyStopping mode must be one of {_VALID_MODES}, got {self.mode!r}.")

    def _extract(self, record: "EpochResult") -> float:
        """Read this config's monitored quantity off one `EpochResult`.

        Raises `TrainerError` for a metric name that was never computed --
        e.g. `monitor="val_accuracy"` when `Trainer` was built with no
        `Accuracy()` metric -- naming what validation quantities actually
        are available for this run.
        """
        if self.monitor == "val_loss":
            return record.val_loss
        metric_name = self.monitor[len("val_"):]
        if metric_name not in record.val_metrics:
            available = ", ".join(["val_loss"] + [f"val_{name}" for name in record.val_metrics])
            raise TrainerError(
                f"EarlyStopping monitor='{self.monitor}' is not available; this Trainer's "
                f"validation results only expose: {available}."
            )
        return record.val_metrics[metric_name]

    def _is_improvement(self, value: float, best: float) -> bool:
        if self.mode == "min":
            return value < best - self.min_delta
        return value > best + self.min_delta


def _snapshot_model_state(model: "Module") -> "dict[str, np.ndarray]":
    """A plain in-memory copy of every Parameter/buffer `model` owns, keyed by dotted name.

    Walks `model.named_parameters()`/`named_buffers()` -- the same
    traversal `save_model()`/`save_checkpoint()` already use -- and reads
    each Tensor's data via its own backend's `to_numpy()`, exactly the
    transfer primitive persistence already uses to get a host-memory copy
    before writing an archive. No file I/O, no archive format: this is a
    runtime copy held only for the duration of one `fit()` call.
    """
    snapshot: "dict[str, np.ndarray]" = {}
    for name, param in model.named_parameters():
        snapshot[f"param:{name}"] = get_backend(param.device).to_numpy(param._data).copy()
    for name, buf in model.named_buffers():
        snapshot[f"buffer:{name}"] = get_backend(buf.device).to_numpy(buf._data).copy()
    return snapshot


def _restore_model_state(model: "Module", snapshot: "dict[str, np.ndarray]") -> None:
    """Write a `_snapshot_model_state()` result back into `model`, in place.

    Restores each Parameter/buffer's *data* by dotted name -- object
    identity, shape, dtype, device, and `requires_grad` are untouched
    (mirrors `Tensor._move_storage_()`'s own in-place-data-replacement
    discipline, minus the device change, since the model never moves during
    this). Any accumulated `.grad` on a restored Parameter is cleared: it
    was computed against the parameter values training has since moved past.
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for key, array in snapshot.items():
        kind, name = key.split(":", 1)
        target = params[name] if kind == "param" else buffers[name]
        backend = get_backend(target.device)
        target._data = backend.from_array(array, array.dtype)
        if kind == "param":
            target.grad = None


__all__ = ["EarlyStopping"]
