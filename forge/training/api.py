"""`forge.train()`/`forge.train_and_save()`: the single-call high-level
training and train-to-artifact entry points (Milestones 79/81).

Every Trainer-based Forge example's `train.py` still starts with the same
hand-assembled sequence before it ever calls `.fit()`:

```python
loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
history = trainer.fit(loader, epochs=epochs, validation_loader=val_loader)
```

`train()` is that sequence, written once, as a thin orchestration layer over
`DataLoader`/`Trainer` -- it builds no gradients, updates no parameters, and
implements no batching of its own; it just wires the pieces `Trainer` already
requires from a `Dataset` a caller already has:

```python
history = forge.train(
    model, train_dataset,
    loss=CrossEntropyLoss(),
    optimizer=Adam(model.parameters(), lr=1e-3),
    epochs=10,
    validation_dataset=val_dataset,
    device="cuda",
    metrics=[Accuracy()],
)
```

## What this adds over `Trainer`

- **Dataset, not DataLoader.** `dataset`/`validation_dataset` accept a plain
  `forge.data.Dataset` directly -- `train()` builds the `DataLoader`(s)
  itself (`batch_size`/`shuffle` are the only two settings every example
  actually varies; anything else -- `drop_last`, a custom `generator`, CUDA
  prefetch -- means the caller wants `Trainer` directly, so an already-built
  `DataLoader` is accepted too and used as-is, unwrapped).
- **Device is one keyword, not a `model.to(device)` the caller writes by
  hand first.** `device="cuda"` (or an explicit `forge.Device`) moves `model`
  there *in place* before training -- see **Device** below for why this is
  safe and why it is the one deliberate difference from `Trainer`'s own
  "validate, never move" policy.

## What this does not add

Everything else is `Trainer`, unmodified: `loss`/`optimizer` must already be
real `forge.nn.Loss`/`forge.optim.Optimizer` instances (no loss/optimizer
inference -- see `docs/development/m79-high-level-training-api.md`), `epochs`
means exactly what `Trainer.fit(epochs=...)` means, and the returned
`TrainingHistory` is `Trainer.fit()`'s own return value, not a new result
type. `train()` never constructs a `Trainer` subclass and adds no callback,
scheduler, or configuration-object surface.

**No checkpoint/resume.** `TrainingSession`/`start_training_session()`
(Milestone 73) already own that workflow, and its resume path fundamentally
replaces the model/optimizer a caller passed in with whatever the checkpoint
saved -- exactly the "second checkpoint abstraction" this milestone's brief
warns against building. `train()`'s whole shape is "you already built
`model`/`optimizer`, train them"; grafting resume onto that would mean either
silently discarding the caller's `model`/`optimizer` for a resumed run (a
surprising, easy-to-miss behavior change) or reintroducing
`start_training_session()`'s `build_model`/`build_optimizer` factories (which
would make `train()` no simpler than the workflow it exists to replace). A
resumable training run still uses `start_training_session()` + `Trainer`
directly -- see `examples/image_folder_classification/train.py` for both
paths side by side.

## `train_and_save()`: `train()` + `save_and_verify()` in one call (Milestone 81)

Both `examples/mnist/train.py` and `examples/image_folder_classification/
train.py` call `train()` and then immediately `save_and_verify()`
(`forge/training/inference.py`, Milestone 78) on the result -- the identical
two-call sequence, in the same order, in both scripts. `train_and_save()` is
that sequence, written once, returning a `TrainAndSaveResult` (`history`,
`val_loss`/`val_metrics` from the final epoch, and the freshly reloaded,
verified `model`) instead of a plain `TrainingHistory`. It adds no
validation/training/persistence logic beyond calling `train()` then
`save_and_verify()` exactly once each -- see that function's own docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..backend.device import Device
from ..data.dataloader import DataLoader
from ..data.dataset import Dataset
from ..exceptions import DataError, TrainerError
from ..nn.loss import Loss
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..tensor.tensor import Tensor
from .inference import save_and_verify
from .metrics import Metric
from .trainer import Trainer, TrainingHistory


def _resolve_loader(data: "Dataset | DataLoader", batch_size: int, shuffle: bool, name: str) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    if not isinstance(data, Dataset):
        raise DataError(
            f"forge.train() requires {name} to be a forge.data.Dataset or a "
            f"forge.data.DataLoader, got {type(data).__name__}."
        )
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train(
    model: Module,
    dataset: "Dataset | DataLoader",
    *,
    loss: Loss,
    optimizer: Optimizer,
    epochs: int,
    batch_size: int = 32,
    shuffle: bool = True,
    validation_dataset: "Dataset | DataLoader | None" = None,
    device: "str | Device | None" = None,
    metrics: "Iterable[Metric] | None" = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Train `model` on `dataset` for `epochs` epochs -- the common case, in one call.

    ```python
    history = forge.train(
        model, train_dataset,
        loss=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=1e-3),
        epochs=10,
        validation_dataset=val_dataset,
        device="cuda",
        metrics=[Accuracy()],
    )
    ```

    `loss`/`optimizer`/`epochs` are required and keyword-only -- there is no
    automatic loss or optimizer selection (see this module's docstring); a
    missing one is a `TypeError` at the call site, the same as any other
    required Python keyword argument. `model` must already be a
    `forge.nn.Module`, `loss` a `forge.nn.Loss`, and `optimizer` a
    `forge.optim.Optimizer` -- validated by the `Trainer` this function
    constructs internally, raising `forge.TrainerError` exactly as a direct
    `Trainer(...)` call would.

    **Dataset input.** `dataset`/`validation_dataset` each accept either a
    plain `forge.data.Dataset` (wrapped in a fresh `DataLoader(batch_size=
    batch_size, shuffle=...)` -- `shuffle` for the training set, always
    `False` for validation, matching every existing example's own
    convention) or an already-constructed `DataLoader`, used exactly as
    given (`batch_size`/`shuffle` are then ignored for that loader -- pass a
    `DataLoader` directly whenever `drop_last`, a custom `generator`, or CUDA
    prefetching (`loader.prefetch(...)`) is needed; `train()` deliberately
    does not re-expose `DataLoader`'s full constructor). Anything else raises
    `forge.DataError`.

    **Device.** `device` defaults to `model`'s current device (`"cpu"` if
    `model` owns no `Parameter`s yet); an explicit `device=` calls
    `model.to(device)` **in place** before training. This is the one
    deliberate difference from `Trainer`'s own "validate, never move"
    policy (`Trainer` itself is unmodified) -- it is safe because
    `Module.to()` moves every `Parameter` in place, preserving Python
    identity, so an `optimizer` already constructed from `model.parameters()`
    before this call remains valid afterward. This is exactly the
    `model = build_model().to(args.device)` line every existing example
    already writes by hand, folded into `train()` so `device="cuda"` alone
    is enough.

    **Validation.** `validation_dataset`, when given, is evaluated once per
    epoch via `Trainer.fit(..., validation_loader=...)` -- identical
    semantics, reflected in the returned `TrainingHistory`'s
    `val_loss`/`val_metrics` per `EpochResult`.

    Returns `Trainer.fit()`'s own `TrainingHistory` -- there is no separate
    "high-level result" type. `model`/`optimizer` are mutated in place by
    training (the same convention `Trainer.fit()` itself uses); this
    function does not return the model, since the caller already holds it.
    """
    if not isinstance(model, Module):
        raise TrainerError(f"forge.train() requires a forge.nn.Module model, got {type(model).__name__}.")

    resolved_device = Device.parse(device) if device is not None else (model.device or Device.parse("cpu"))
    model.to(resolved_device)

    train_loader = _resolve_loader(dataset, batch_size, shuffle, "dataset")
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = _resolve_loader(validation_dataset, batch_size, False, "validation_dataset")

    trainer = Trainer(
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        device=resolved_device,
        metrics=metrics,
        verbose=verbose,
    )
    return trainer.fit(train_loader, epochs=epochs, validation_loader=validation_loader)


@dataclass(frozen=True)
class TrainAndSaveResult:
    """`train_and_save()`'s return value (Milestone 81).

    The same three things a caller gets from `train()` + `save_and_verify()`
    separately, standing side by side rather than split across two return
    values: `history` is `train()`'s own `TrainingHistory`, `model` is
    `save_and_verify()`'s freshly reloaded, verified `Module` (not the
    original -- see that function's docstring for why), and `val_loss`/
    `val_metrics` are the *last* epoch's validation result -- already
    computed once per epoch via `validation_dataset=` (`None`/`{}` when no
    `validation_dataset` was given), copied here so a caller doesn't need to
    know to index `history[-1]` to find the final evaluation result. No
    second evaluation pass runs to produce these -- see `train_and_save()`'s
    own docstring.
    """

    history: TrainingHistory
    val_loss: "float | None"
    val_metrics: "dict[str, float]"
    model: Module


def train_and_save(
    model: Module,
    dataset: "Dataset | DataLoader",
    *,
    loss: Loss,
    optimizer: Optimizer,
    epochs: int,
    path: str,
    sample: Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    validation_dataset: "Dataset | DataLoader | None" = None,
    device: "str | Device | None" = None,
    metrics: "Iterable[Metric] | None" = None,
    verbose: bool = True,
    preprocessing: "Any | None" = None,
    classes: "list[str] | None" = None,
    atol: float = 1e-5,
) -> TrainAndSaveResult:
    """Train `model`, then save + verify it as a portable artifact, in one call (Milestone 81).

    ```python
    result = forge.train_and_save(
        model, train_dataset,
        loss=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=1e-3),
        epochs=10,
        validation_dataset=val_dataset,
        device="cuda",
        metrics=[Accuracy()],
        path="model.forge",
        sample=query_x_batch,
        preprocessing=build_transform(),
        classes=full_dataset.classes,
    )
    result.history       # the TrainingHistory train() returned
    result.val_metrics    # the last epoch's validation metrics, e.g. {"accuracy": 0.97}
    result.model          # the freshly reloaded, verified Module
    ```

    `examples/mnist/train.py` and `examples/image_folder_classification/
    train.py` each independently called `train()` then immediately
    `save_and_verify()` on its result -- the same two-call sequence, in the
    same order, with no framework logic between them beyond a hand-picked
    `sample`. `train_and_save()` is that sequence, written once: `train(model,
    dataset, loss=loss, optimizer=optimizer, epochs=epochs, ...)` followed by
    `save_and_verify(model, path, sample, preprocessing=preprocessing,
    classes=classes, atol=atol)`, with the final epoch's validation result
    copied onto the return value so a caller does not need to know
    `TrainingHistory`'s own indexing convention to find it.

    This is composition only: `train()` and `save_and_verify()` are each
    called exactly once, unmodified, with no new validation, training, or
    persistence logic of its own -- everything either function itself
    documents (required `loss`/`optimizer`, no automatic loss/optimizer/
    architecture selection, `sample` must already be a batched `Tensor`
    matching `predict()`'s own calling convention, `TrainerError`/`DataError`
    /`PersistenceError` on the same conditions those two functions already
    raise) applies here unchanged. `device`, when given, is resolved and the
    model moved by `train()`; `save_and_verify()` is then called with no
    explicit `device=`, so it defaults to `model.device` -- the same device
    `train()` just moved the model to.

    **Scope.** Like `train()`, this has no `checkpoint=`/`resume=` concept --
    checkpoint persistence remains a separate, resumable-training concern
    handled by `forge.save_checkpoint()`/`TrainingSession` alongside this
    call, exactly as every retrofitted example already does (see `train()`'s
    own docstring for why checkpoint/resume was deliberately kept out of the
    high-level training call). `Trainer`/`train()`/`save_and_verify()` remain
    fully available, unmodified, for a caller needing checkpoint/resume, CUDA
    prefetch, a custom `DataLoader`, or a training/verification boundary this
    function does not expose.
    """
    history = train(
        model, dataset,
        loss=loss, optimizer=optimizer, epochs=epochs,
        batch_size=batch_size, shuffle=shuffle,
        validation_dataset=validation_dataset, device=device, metrics=metrics, verbose=verbose,
    )
    reloaded = save_and_verify(
        model, path, sample, preprocessing=preprocessing, classes=classes, atol=atol,
    )
    last = history[-1]
    return TrainAndSaveResult(
        history=history, val_loss=last.val_loss, val_metrics=last.val_metrics, model=reloaded,
    )


__all__ = ["train", "train_and_save", "TrainAndSaveResult"]
