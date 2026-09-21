"""`forge.train()`/`forge.train_and_save()`: the single-call high-level
training and train-to-artifact entry points (Milestones 79/81), returning
`TrainingResult`/`TrainAndSaveResult` (extended in Milestone 99 -- see that
section below).

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
`train_loss`/`train_metrics`/`val_loss`/`val_metrics` from the final epoch,
the freshly reloaded, verified `model`, and the `artifact_path` actually
written) instead of a plain `TrainingHistory`. It adds no
validation/training/persistence logic beyond calling `train()` then
`save_and_verify()` exactly once each -- see that function's own docstring.

## `TrainingResult`: what `train()` actually returns (Milestone 99)

Before Milestone 99, `train()` returned `Trainer.fit()`'s own bare
`TrainingHistory`, and a caller who wanted the trained model, the final
epoch's loss/metrics, or (for `train_and_save()`) the artifact path it just
wrote had to keep its own local variables for all of it -- exactly the
"scattered objects" problem `experiment/run_experiment.py` (Milestone 98)
hand-rolls today (`model_path = output_dir / ...` tracked separately,
`history[-1].train_loss`/`history[-1].val_metrics["accuracy"]` indexed by
hand into the returned history). `TrainingResult` (subclasses
`TrainingHistory` -- see its own docstring) and `TrainAndSaveResult`'s new
`artifact_path` field close that gap: `train()`/`train_and_save()` now
return everything a caller needs immediately after one training operation,
without giving up any existing behavior (`isinstance(result,
TrainingHistory)` is still `True`; indexing/iteration/`len()` are unchanged).
This is a training-*result* capability only -- it adds no experiment
tracking, no persistence of its own (`TrainingResult`/`TrainAndSaveResult`
are never written into a `.forge` file), and no new training semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..backend.device import Device
from ..data.dataloader import DataLoader
from ..data.dataset import Dataset
from ..exceptions import DataError, TrainerError
from ..nn.loss import Loss
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..tensor.tensor import Tensor
from .early_stopping import EarlyStopping
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


class TrainingResult(TrainingHistory):
    """`train()`'s return value (Milestone 99): a `TrainingHistory` plus the trained model.

    `TrainingResult` *is* a `TrainingHistory` -- it subclasses it rather than
    wrapping it, so every existing caller written against `history =
    forge.train(...)` (`len(history)`, `for record in history`, `history[i]`,
    `history.train_losses`/`history.val_losses`) keeps working completely
    unchanged; `isinstance(result, TrainingHistory)` is `True`. `result.history`
    returns `result` itself -- there is no second, separate history object to
    keep in sync with `train()`'s own docstring's "no second `history`
    representation" rule.

    What it adds over a plain `TrainingHistory`:

    - `model` -- the trained `Module`, i.e. the exact object `train()` was
      called with (already mutated in place by training, per `train()`'s own
      docstring) -- not a copy or reconstruction. Exposed here so
      `result.model` answers "what did I just train?" without the caller
      needing to have kept its own reference.
    - `epochs_completed` -- `len(self)`, spelled out for readability; Forge
      training has no early stopping, so this always equals the requested
      `epochs`, but a caller answering "how many epochs actually ran?"
      should not have to know that `TrainingHistory` supports `len()`.
    - `final_train_loss`/`final_train_metrics`/`final_val_loss`/
      `final_val_metrics` -- the last completed epoch's own `EpochResult`
      fields, copied here so "what was the final performance?" does not
      require knowing `history[-1]`'s indexing convention.
      `final_val_loss`/`final_val_metrics` are `None`/`{}` when `train()` was
      called without a `validation_dataset` -- exactly `EpochResult`'s own
      convention, never fabricated.
    - `stopped_early`/`best_epoch`/`best_monitored_value`/
      `monitored_quantity` (Milestone 100) -- copied from the underlying
      `Trainer.fit()` call's own `TrainingHistory`; see `EarlyStopping`'s
      docstring. All four are `False`/`None`/`None`/`None` when `train()`
      was called with no `early_stopping=`.

    Never constructed directly by a caller -- `train()` is the only producer.
    """

    def __init__(self, history: TrainingHistory, model: Module) -> None:
        super().__init__()
        self.records = history.records
        self.model = model
        self.stopped_early = history.stopped_early
        self.best_epoch = history.best_epoch
        self.best_monitored_value = history.best_monitored_value
        self.monitored_quantity = history.monitored_quantity

    @property
    def history(self) -> "TrainingResult":
        return self

    @property
    def epochs_completed(self) -> int:
        return len(self.records)

    @property
    def final_train_loss(self) -> float:
        return self.records[-1].train_loss

    @property
    def final_train_metrics(self) -> "dict[str, float]":
        return self.records[-1].train_metrics

    @property
    def final_val_loss(self) -> "float | None":
        return self.records[-1].val_loss

    @property
    def final_val_metrics(self) -> "dict[str, float]":
        return self.records[-1].val_metrics

    def __repr__(self) -> str:
        return (
            f"TrainingResult(epochs_completed={self.epochs_completed}, "
            f"final_train_loss={self.final_train_loss:.4f}, "
            f"final_val_loss={self.final_val_loss})"
        )


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
    early_stopping: "EarlyStopping | None" = None,
) -> TrainingResult:
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
    semantics, reflected in the returned `TrainingResult`'s
    `val_loss`/`val_metrics` per `EpochResult`.

    **Early stopping (Milestone 100).** `early_stopping`, when given, is
    passed straight through to `Trainer.fit(..., early_stopping=...)` --
    requires `validation_dataset` (raises `forge.TrainerError` otherwise, the
    same error `Trainer.fit()` itself raises). See `forge.training.
    EarlyStopping`'s own docstring for `monitor`/`patience`/`min_delta`/
    `restore_best` semantics. `result.stopped_early`/`result.best_epoch`/
    `result.best_monitored_value` report the outcome; `result.model` (and,
    for `train_and_save()`, the saved artifact) already reflects the
    restored best parameters when `restore_best=True` fired, since `fit()`
    performs that restoration in place before returning.

    Returns a `TrainingResult` (Milestone 99) -- a `TrainingHistory` (exactly
    `Trainer.fit()`'s own record-per-epoch object; no second history
    representation) with the trained `model` and final-epoch convenience
    accessors (`final_train_loss`, `final_val_loss`, ...) attached -- see
    `TrainingResult`'s own docstring. `model`/`optimizer` are still mutated in
    place by training (the same convention `Trainer.fit()` itself uses);
    `result.model` is that same object, not a copy, for a caller that would
    rather read it off the result than keep its own reference.
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
    history = trainer.fit(
        train_loader, epochs=epochs, validation_loader=validation_loader, early_stopping=early_stopping,
    )
    return TrainingResult(history, model)


@dataclass(frozen=True)
class TrainAndSaveResult:
    """`train_and_save()`'s return value (Milestone 81, extended Milestone 99).

    Everything a caller gets from `train()` + `save_and_verify()` separately,
    standing side by side rather than split across two return values and two
    manually-tracked local variables: `history` is `train()`'s own
    `TrainingResult` (a `TrainingHistory` plus the trained model and
    final-epoch accessors -- see that class's own docstring), `model` is
    `save_and_verify()`'s freshly reloaded, verified `Module` (not the
    original, and not `history.model` either -- see that function's docstring
    for why they differ), `artifact_path` is the `path` this call actually
    wrote and verified (Milestone 99 -- previously only available from the
    caller's own `path` variable), and `train_loss`/`train_metrics`/
    `val_loss`/`val_metrics` are the *last* epoch's training/validation
    result -- already computed once per epoch via `validation_dataset=`
    (`val_loss`/`val_metrics` are `None`/`{}` when no `validation_dataset`
    was given), copied here so a caller doesn't need to know to index
    `history[-1]` to find the final result. No second training or evaluation
    pass runs to produce these -- see `train_and_save()`'s own docstring.
    """

    history: TrainingResult
    train_loss: float
    train_metrics: "dict[str, float]"
    val_loss: "float | None"
    val_metrics: "dict[str, float]"
    model: Module
    artifact_path: str
    stopped_early: bool = False
    best_epoch: "int | None" = None
    best_monitored_value: "float | None" = None


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
    task: "str | None" = None,
    atol: float = 1e-5,
    early_stopping: "EarlyStopping | None" = None,
    target_transform: "Any | None" = None,
    feature_names: "Sequence[str] | None" = None,
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
        task="classification",
    )
    result.history       # the TrainingResult train() returned
    result.val_metrics   # the last epoch's validation metrics, e.g. {"accuracy": 0.97}
    result.model         # the freshly reloaded, verified Module
    result.artifact_path # the path that was actually written and verified
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

    `task` (Milestone 87) is passed straight through to `save_and_verify()` --
    the recommended way to produce a correctly self-describing artifact in
    one call: `forge.train_and_save(..., path="model.forge",
    task="classification")` needs no separate `save_model()` call solely to
    attach task metadata. See `save_model()`'s own docstring for the exact
    vocabulary; omitting it (the default) writes no task metadata, exactly
    like omitting `classes`/`preprocessing`.

    **Early stopping (Milestone 100).** `early_stopping` is passed straight
    through to `train()`. Because `train()`'s own `Trainer.fit()` call
    restores the best-epoch parameters/buffers *in place* before `train()`
    returns (when `early_stopping.restore_best` fired), `model` already
    holds the restored best state by the time `save_and_verify()` runs --
    the artifact this call saves and verifies is the restored best model,
    never the later, possibly-worse final-epoch state. `result.
    stopped_early`/`result.best_epoch`/`result.best_monitored_value` mirror
    `TrainingResult`'s own fields for a caller who only kept this result.

    **Target transform (Milestone 116).** `target_transform` is a
    `forge.data.StandardizeTarget` passed straight through to `save_and_verify()`
    / `save_model()` (`task="regression"` only). It is **not** applied to `dataset`:
    the caller trains on the already-transformed targets and this records the
    transform in the artifact so `load_predictor()` returns native units. Because the
    training is on transformed targets, everything this call reports --
    `train_loss`/`val_loss`/`*_metrics`, `history`, `best_monitored_value` -- is in
    the *transformed* space; use `forge.load_predictor(path).evaluate(X, y)` for
    native-unit metrics (`forge.train_tabular_regressor()` does exactly that).

    **Feature names (Milestone 119).** `feature_names` -- one name per input column, for
    `task="regression"`/`"tabular_classification"` -- is passed straight through to
    `save_and_verify()` / `save_model()`, which records it in the artifact and reads it back.
    Nothing is trained differently: names are identity metadata about the *raw* input
    columns, never an input to the model or to preprocessing.
    """
    history = train(
        model, dataset,
        loss=loss, optimizer=optimizer, epochs=epochs,
        batch_size=batch_size, shuffle=shuffle,
        validation_dataset=validation_dataset, device=device, metrics=metrics, verbose=verbose,
        early_stopping=early_stopping,
    )
    reloaded = save_and_verify(
        model, path, sample, preprocessing=preprocessing, classes=classes, task=task, atol=atol,
        target_transform=target_transform, feature_names=feature_names,
    )
    last = history[-1]
    return TrainAndSaveResult(
        history=history,
        train_loss=last.train_loss,
        train_metrics=last.train_metrics,
        val_loss=last.val_loss,
        val_metrics=last.val_metrics,
        model=reloaded,
        artifact_path=path,
        stopped_early=history.stopped_early,
        best_epoch=history.best_epoch,
        best_monitored_value=history.best_monitored_value,
    )


__all__ = ["train", "train_and_save", "TrainingResult", "TrainAndSaveResult"]
