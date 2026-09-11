"""`forge.train()`: the single-call high-level training entry point (Milestone 79).

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
"""

from __future__ import annotations

from typing import Iterable

from ..backend.device import Device
from ..data.dataloader import DataLoader
from ..data.dataset import Dataset
from ..exceptions import DataError, TrainerError
from ..nn.loss import Loss
from ..nn.module import Module
from ..optim.optimizer import Optimizer
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


__all__ = ["train"]
