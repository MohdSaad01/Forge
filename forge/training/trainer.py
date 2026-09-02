"""The Trainer: orchestrates DataLoader/Module/Loss/Autograd/Optimizer.

`Trainer` owns none of the systems it coordinates -- it does not compute
gradients, update parameters, implement a loss, implement an optimizer
algorithm, or implement DataLoader batching. Each training step is exactly
the manual sequence Milestones 1-5 already required by hand:

```text
optimizer.zero_grad() -> model(x) -> loss_fn(prediction, target)
    -> loss.backward() -> optimizer.step()
```

`Trainer` just runs that sequence over every batch of every epoch, and
records what happened. See `docs/architecture/training-engine.md`.

As of Milestone 12, that sequence runs unmodified on CUDA: `Trainer` owns
device placement the same minimal way it owns everything else -- it
validates the model's device (never moves it) and explicitly transfers each
CPU batch `DataLoader` yields to `self.device` before the forward pass.
`DataLoader` itself is never made device-aware. See the **Device**
paragraphs on the `Trainer` class docstring below and
`docs/architecture/training-engine.md`'s **Device semantics** section.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from ..autograd import no_grad
from ..backend import get_backend
from ..backend.device import Device
from ..exceptions import TrainerError, UnsupportedDeviceError
from ..nn.loss import Loss
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..tensor.tensor import Tensor
from .metrics import Metric


@dataclass(frozen=True)
class EvaluationResult:
    """The result of `Trainer.evaluate()` over one DataLoader."""

    loss: float
    metrics: "dict[str, float]"
    samples: int
    duration: float
    device: str


@dataclass(frozen=True)
class EpochResult:
    """One `TrainingHistory` record: everything observed during one `fit()` epoch.

    `val_loss`/`val_metrics` reflect a `validation_loader` evaluated after
    this epoch's training; when `fit()` was called without one, `val_loss`
    is `None` and `val_metrics` is empty -- classification-only fields
    (e.g. accuracy) are never hard-coded here, they simply appear inside
    `train_metrics`/`val_metrics` when a classification metric was supplied.
    """

    epoch: int
    train_loss: float
    train_metrics: "dict[str, float]"
    val_loss: "float | None"
    val_metrics: "dict[str, float]"
    duration: float
    samples: int
    device: str


class TrainingHistory:
    """The ordered, programmatically-accessible record `Trainer.fit()` returns.

    One `EpochResult` per completed epoch, in the order epochs ran. Supports
    `len()`, iteration, and integer indexing directly.
    """

    def __init__(self) -> None:
        self.records: "list[EpochResult]" = []

    def append(self, record: EpochResult) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> "Iterator[EpochResult]":
        return iter(self.records)

    def __getitem__(self, index: int) -> EpochResult:
        return self.records[index]

    @property
    def train_losses(self) -> "list[float]":
        """Per-epoch training loss, in epoch order."""
        return [r.train_loss for r in self.records]

    @property
    def val_losses(self) -> "list[float | None]":
        """Per-epoch validation loss, in epoch order (`None` for an epoch run without validation)."""
        return [r.val_loss for r in self.records]

    def __repr__(self) -> str:
        return f"TrainingHistory(epochs={len(self.records)})"


class Trainer:
    """Coordinates repeated forward/loss/backward/update steps over a DataLoader.

    ```python
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device="cpu")
    history = trainer.fit(train_loader, epochs=10, validation_loader=val_loader)
    results = trainer.evaluate(test_loader)
    ```

    `model`/`loss_fn`/`optimizer` must be actual Forge `Module`/`Loss`/
    `Optimizer` instances -- `Trainer` composes them, it does not duplicate
    what they do. `metrics` is an optional iterable of `Metric` instances
    (each supplying a unique `.name`), reported alongside the loss but never
    replacing it as the training objective.

    **Device (Milestone 12).** `device` must resolve (via `Device.parse`) to
    `"cpu"` or `"cuda"`; anything else raises `UnsupportedDeviceError`
    immediately, and `device="cuda"` on a machine without a working CUDA
    backend raises `CUDAError` immediately (construction actually probes the
    backend via `get_backend()` rather than deferring the failure to the
    first batch). There is no CUDA-specific `Trainer` subclass -- one class
    handles both devices, per `docs/architecture/training-engine.md`'s
    **Device semantics** section.

    `Trainer` never moves the model: it *validates*, at the start of
    `fit()`/`evaluate()`, that every `Parameter` the model owns already sits
    on `self.device` (via `Module.device`), raising `UnsupportedDeviceError`
    with a clear message naming the required `model.to(device)` call
    otherwise. Call `model.to(device)` yourself before constructing/using a
    `Trainer` configured for that device. A model with no `Parameter`s
    (`Module.device is None`) is exempt -- there is nothing to validate.

    `Trainer` *does* move each batch explicitly: `fit()`/`evaluate()` call
    `x.to(self.device)` (and `y.to(self.device)` when `y` is a `Tensor`) on
    every batch `DataLoader` yields, before the forward pass. `DataLoader`
    itself remains entirely CPU-side and is never made device-aware -- see
    **Batch movement** below.
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Loss,
        optimizer: Optimizer,
        device: "str | Device" = "cpu",
        metrics: "Iterable[Metric] | None" = None,
        verbose: bool = True,
    ):
        if not isinstance(model, Module):
            raise TrainerError(
                f"Trainer requires a forge.nn.Module model, got {type(model).__name__}."
            )
        if not isinstance(loss_fn, Loss):
            raise TrainerError(
                f"Trainer requires a forge.nn.Loss loss_fn, got {type(loss_fn).__name__}."
            )
        if not isinstance(optimizer, Optimizer):
            raise TrainerError(
                f"Trainer requires a forge.optim.Optimizer optimizer, got "
                f"{type(optimizer).__name__}."
            )

        resolved_device = Device.parse(device)
        # Probes the backend now (raises `CUDAError` immediately for
        # `device="cuda"` on a machine without a working CUDA backend)
        # rather than deferring the failure to the first batch.
        get_backend(resolved_device)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = resolved_device
        self.verbose = bool(verbose)
        self.metrics = self._validate_metrics(metrics)
        # Training-progress counters (Milestone 18): persisted/restored by
        # `save_checkpoint()`/`resume()` so `fit()` continues epoch numbering
        # and `global_step` counting across a checkpoint resume, rather than
        # restarting from 1/0. Plain mutable attributes -- not part of any
        # constructor validation -- mirroring how `Trainer` already exposes
        # `self.model`/`self.optimizer` for direct inspection/reassignment.
        self.epoch = 0
        self.global_step = 0

    @staticmethod
    def _validate_metrics(metrics: "Iterable[Metric] | None") -> "dict[str, Metric]":
        if metrics is None:
            return {}
        result: "dict[str, Metric]" = {}
        for m in metrics:
            if not isinstance(m, Metric):
                raise TrainerError(
                    f"Trainer metrics must be forge.training.Metric instances, got "
                    f"{type(m).__name__}."
                )
            if m.name in result:
                raise TrainerError(f"Duplicate metric name '{m.name}'.")
            result[m.name] = m
        return result

    # -- batch handling -------------------------------------------------

    @staticmethod
    def _unpack_batch(batch: Any) -> "tuple[Tensor, Any]":
        if not isinstance(batch, tuple) or len(batch) != 2:
            got = f"{len(batch)}-tuple" if isinstance(batch, tuple) else type(batch).__name__
            raise TrainerError(
                "Trainer requires each DataLoader batch to be a (features, target) "
                f"tuple, got {got}."
            )
        x, y = batch
        if not isinstance(x, Tensor):
            raise TrainerError(
                f"Trainer requires batch features to be a Tensor, got {type(x).__name__}."
            )
        if x.ndim == 0:
            raise TrainerError(
                "Trainer requires batch features with a leading batch dimension, "
                "got a scalar Tensor."
            )
        return x, y

    def _to_device_batch(self, batch: Any) -> "tuple[Tensor, Any]":
        """`_unpack_batch`, then explicit `x.to(device)`/`y.to(device)` (Milestone 12).

        The one place a CPU `DataLoader` batch becomes a `self.device` batch.
        `x` is always a Tensor (`_unpack_batch` already enforces this) and is
        always moved. `y` is moved only if it is itself a Tensor (matching
        `DataLoader`'s own contract: a `TensorDataset`-backed loader always
        yields a Tensor target, but Trainer does not assume every `Dataset`
        does) -- a non-Tensor `y` (e.g. a raw NumPy target array) is passed
        through unchanged, since `.to()` has no meaning for it and each
        `Loss`/`Metric` already accepts raw array-like targets directly. Both
        calls are no-ops (return the original object) when the batch is
        already on `self.device`, so this costs nothing extra for a CPU
        `Trainer`.
        """
        x, y = self._unpack_batch(batch)
        x = x.to(self.device)
        if isinstance(y, Tensor):
            y = y.to(self.device)
        return x, y

    def _check_model_device(self) -> None:
        """Validate (never move) that the model already sits on `self.device` (Milestone 12).

        `Trainer`'s chosen device policy: validate, don't relocate -- see the
        class docstring's **Device** section and
        `docs/architecture/training-engine.md`. A model with no `Parameter`s
        (`Module.device is None`, e.g. a bare activation-only module) is
        exempt: there is no device to be inconsistent with.
        """
        model_device = self.model.device
        if model_device is not None and model_device != self.device:
            raise UnsupportedDeviceError(
                f"Trainer is configured for device '{self.device}' but the model's "
                f"Parameters are on device '{model_device}'. Trainer validates model "
                "placement rather than moving it implicitly -- call "
                f"model.to('{self.device}') explicitly first."
            )

    # -- loader / config validation --------------------------------------

    @staticmethod
    def _validate_epochs(epochs: int) -> None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise TrainerError(f"epochs must be a positive int, got {epochs!r}.")

    @staticmethod
    def _validate_loader(loader: Any, name: str) -> None:
        try:
            n_batches = len(loader)
        except TypeError as exc:
            raise TrainerError(
                f"{name} must support len(); {type(loader).__name__} does not."
            ) from exc
        if n_batches == 0:
            raise TrainerError(
                f"{name} produced no batches; Trainer requires at least one batch "
                "per epoch."
            )

    # -- core phases ------------------------------------------------------

    def _run_training_epoch(self, loader: Any) -> "tuple[float, dict[str, float], int]":
        self.model.train()
        for m in self.metrics.values():
            m.reset()

        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            x, y = self._to_device_batch(batch)
            batch_size = x.shape[0]

            self.optimizer.zero_grad()
            prediction = self.model(x)
            loss = self.loss_fn(prediction, y)
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            total_loss += float(loss.to("cpu").numpy()) * batch_size
            total_samples += batch_size
            for m in self.metrics.values():
                m.update(prediction, y)

        if total_samples == 0:
            raise TrainerError("train_loader yielded no samples for this epoch.")

        train_loss = total_loss / total_samples
        train_metrics = {name: m.compute() for name, m in self.metrics.items()}
        return train_loss, train_metrics, total_samples

    def evaluate(self, loader: Any) -> EvaluationResult:
        """Run `loader` through the model without updating parameters.

        Switches the model to evaluation mode (`model.eval()`, propagated
        to every nested child module) for the duration of the call, then
        restores whatever mode it was in beforehand -- calling `evaluate()`
        mid-`fit()` returns the model to training mode for the next epoch,
        while calling it standalone after training leaves the model's mode
        exactly as the caller left it. The forward pass runs inside
        `forge.no_grad()`, so it never builds an autograd graph and
        `optimizer.step()` is never called -- parameters cannot change as a
        result of evaluation.
        """
        self._validate_loader(loader, "loader")
        self._check_model_device()

        was_training = self.model.training
        self.model.eval()
        for m in self.metrics.values():
            m.reset()

        start = time.perf_counter()
        total_loss = 0.0
        total_samples = 0
        try:
            with no_grad():
                for batch in loader:
                    x, y = self._to_device_batch(batch)
                    batch_size = x.shape[0]

                    prediction = self.model(x)
                    loss = self.loss_fn(prediction, y)

                    total_loss += float(loss.to("cpu").numpy()) * batch_size
                    total_samples += batch_size
                    for m in self.metrics.values():
                        m.update(prediction, y)
        finally:
            self.model.train(was_training)

        if total_samples == 0:
            raise TrainerError("loader yielded no samples to evaluate.")

        duration = time.perf_counter() - start
        loss_value = total_loss / total_samples
        metric_values = {name: m.compute() for name, m in self.metrics.items()}
        return EvaluationResult(
            loss=loss_value,
            metrics=metric_values,
            samples=total_samples,
            duration=duration,
            device=str(self.device),
        )

    def fit(
        self,
        train_loader: Any,
        epochs: int,
        validation_loader: "Any | None" = None,
    ) -> TrainingHistory:
        """Train `self.model` for `epochs` epochs over `train_loader`.

        One epoch is exactly one full pass over `train_loader` (as many
        batches as it yields). `epochs` must be a positive int -- `epochs
        <= 0` raises `TrainerError` immediately rather than silently
        training zero epochs and returning an empty history. `train_loader`
        (and `validation_loader`, if given) must yield at least one batch;
        an empty loader raises `TrainerError` rather than dividing by zero
        samples.

        Every epoch that runs is recorded in the returned `TrainingHistory`,
        in order -- there is no partial/early-stopping skip in this
        milestone. If `validation_loader` is supplied, it is evaluated via
        `self.evaluate()` once at the end of each training epoch (never
        before, never mid-epoch), and its loss/metrics are attached to that
        epoch's `EpochResult`. Progress for each epoch is printed unless
        `Trainer(..., verbose=False)` was set at construction.
        """
        self._validate_epochs(epochs)
        self._validate_loader(train_loader, "train_loader")
        if validation_loader is not None:
            self._validate_loader(validation_loader, "validation_loader")
        self._check_model_device()

        history = TrainingHistory()
        for local_epoch in range(1, epochs + 1):
            self.epoch += 1
            start = time.perf_counter()
            train_loss, train_metrics, samples = self._run_training_epoch(train_loader)

            val_loss = None
            val_metrics: "dict[str, float]" = {}
            if validation_loader is not None:
                eval_result = self.evaluate(validation_loader)
                val_loss = eval_result.loss
                val_metrics = eval_result.metrics

            duration = time.perf_counter() - start
            record = EpochResult(
                epoch=self.epoch,
                train_loss=train_loss,
                train_metrics=train_metrics,
                val_loss=val_loss,
                val_metrics=val_metrics,
                duration=duration,
                samples=samples,
                device=str(self.device),
            )
            history.append(record)
            if self.verbose:
                self._report(record, local_epoch, epochs)

        return history

    # -- checkpointing (Milestone 18) ----------------------------------------

    def save_checkpoint(self, path: str, *, extra: "dict[str, Any] | None" = None) -> None:
        """Save this Trainer's `model`/`optimizer` state and training progress to `path`.

        A thin wrapper over `forge.serialization.save_checkpoint()` -- fills
        in `epoch`/`global_step` from `self.epoch`/`self.global_step`
        automatically. See `forge.serialization.checkpoint` for the format.
        """
        from ..serialization.checkpoint import save_checkpoint as _save_checkpoint

        _save_checkpoint(path, self.model, self.optimizer, epoch=self.epoch, global_step=self.global_step, extra=extra)

    def resume(self, checkpoint: "Any") -> None:
        """Adopt a `load_checkpoint()`-produced `Checkpoint`'s state in place.

        Replaces `self.model`, `self.optimizer`, `self.epoch`, and
        `self.global_step` with the checkpoint's; `self.loss_fn`/
        `self.device`/`self.metrics`/`self.verbose` are untouched. `fit()`
        called afterward resumes the normal
        `DataLoader -> forward -> loss -> backward -> optimizer.step()`
        sequence exactly as ordinary training, continuing epoch numbering
        and `global_step` counting from the checkpoint's values.

        `checkpoint.model` must already sit on `self.device` -- matching
        Trainer's existing "validate, never move" device policy
        (`_check_model_device`), pass the matching `device=` to
        `load_checkpoint()` first if needed.
        """
        from ..serialization.checkpoint import Checkpoint as _Checkpoint

        if not isinstance(checkpoint, _Checkpoint):
            raise TrainerError(f"Trainer.resume() requires a forge.serialization.Checkpoint, got {type(checkpoint).__name__}.")

        model_device = checkpoint.model.device
        if model_device is not None and model_device != self.device:
            raise TrainerError(
                f"Cannot resume: checkpoint model is on device '{model_device}' but this "
                f"Trainer is configured for device '{self.device}'. Call "
                f"load_checkpoint(path, device='{self.device}') first."
            )

        self.model = checkpoint.model
        self.optimizer = checkpoint.optimizer
        self.epoch = checkpoint.epoch
        self.global_step = checkpoint.global_step

    # -- progress reporting -------------------------------------------------

    def _report(self, record: EpochResult, local_epoch: int, epochs: int) -> None:
        """Print one epoch's summary. Format is informational only -- not a stable contract."""
        print(f"Epoch {local_epoch}/{epochs} (global epoch {record.epoch})")
        print(f"loss: {record.train_loss:.4f}")
        for name, value in record.train_metrics.items():
            print(f"{name}: {value:.4f}")
        if record.val_loss is not None:
            print(f"val_loss: {record.val_loss:.4f}")
            for name, value in record.val_metrics.items():
                print(f"val_{name}: {value:.4f}")
        samples_per_sec = record.samples / record.duration if record.duration > 0 else float("inf")
        print(f"time: {record.duration:.2f}s")
        print(f"samples/sec: {samples_per_sec:.0f}")
        print(f"device: {record.device}")


__all__ = ["Trainer", "TrainingHistory", "EpochResult", "EvaluationResult"]
