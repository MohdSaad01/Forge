# Training Engine (Milestone 6)

## Package layout
```
forge/
    training/
        trainer.py    Trainer, EpochResult, EvaluationResult, TrainingHistory
        metrics.py    Metric, MeanSquaredError, MeanAbsoluteError, Accuracy
    autograd/engine.py  no_grad, is_grad_enabled (new in this milestone)
```
`forge.training` is exposed as a submodule of `forge` (`forge.training.Trainer`),
alongside `forge.nn`/`forge.optim`/`forge.data`/`forge.random`. `forge.no_grad`
is exposed at the top level (`forge.no_grad()`), alongside `forge.Tensor`.

## Trainer responsibility
`Trainer` (`forge/training/trainer.py`) orchestrates the existing
`Dataset`/`DataLoader`, `Module`, `Loss`, autograd, and `Optimizer` systems
into a reusable training/evaluation workflow. It owns no numerical
computation of its own:

```text
DataLoader
    v
Trainer
    +-- Model     (forward pass -- Trainer does not compute predictions itself)
    +-- Loss      (objective   -- Trainer does not implement loss math)
    +-- Optimizer (parameter update -- Trainer does not compute or apply gradients)
          v
       training
          v
       metrics/history
```

Concretely, every training batch runs exactly the sequence Milestones 1-5
already required by hand:
```text
optimizer.zero_grad() -> model(x) -> loss_fn(prediction, target)
    -> loss.backward() -> optimizer.step()
```
`Trainer` never calls a gradient computation directly (that is
`Tensor.backward()`/`forge.autograd`'s job), never mutates a parameter
directly (that is `Optimizer.step()`'s job), and never re-implements
batching/shuffling (that is `DataLoader`'s job). It sequences those calls
over every batch of every epoch and records what happened.

## Public API
```python
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device="cpu",
    metrics=[MeanSquaredError()],   # optional
    verbose=True,                    # optional, default True
)

history = trainer.fit(train_loader, epochs=10, validation_loader=val_loader)
results = trainer.evaluate(test_loader)
```
Construction validates every component eagerly and raises `TrainerError`
(or `UnsupportedDeviceError` for the device) rather than failing later with
an unrelated `AttributeError`:
- `model` must be a `forge.nn.Module` instance.
- `loss_fn` must be a `forge.nn.Loss` instance.
- `optimizer` must be a `forge.optim.Optimizer` instance.
- `device` must resolve (via `forge.Device.parse`) to `"cpu"` -- see
  **Device semantics** below.
- `metrics`, if given, must be `Metric` instances with unique `.name`s.

## Training lifecycle
`fit()` runs `_run_training_epoch` once per epoch:
```python
model.train()
for batch in train_loader:
    x, y = batch                 # (features, target), enforced -- see below
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
```
`model.train()` is called once per epoch, before the batch loop, so a model
switched to `eval()` mode by a prior `evaluate()` call is always restored to
training mode before the next epoch's batches run.

**Batch structure.** Trainer requires every batch to be a `(features,
target)` 2-tuple with a Tensor-valued `features` (exactly what `DataLoader`
produces for a `TensorDataset`-style dataset, per
`docs/architecture/data-system.md`). A batch that is a single Tensor, a
tuple of the wrong length, or has non-Tensor features raises `TrainerError`
identifying the mismatch rather than failing deeper inside `model()`/
`loss_fn()`.

**Loss/metric aggregation.** `MSELoss`/`CrossEntropyLoss` each return a
*mean* over their batch (per-element or per-sample, respectively). Trainer
accumulates `sum(batch_loss * batch_size)` and `sum(batch_size)` across all
batches in an epoch and divides at the end, so a batch of 6 samples
contributes proportionally less than a batch of 32 -- never a naive mean of
per-batch means. `Metric.update()`/`compute()` follow the identical pattern
(see **Metrics** below).

## Evaluation lifecycle
```python
results = trainer.evaluate(loader)
# results.loss, results.metrics, results.samples, results.duration, results.device
```
`evaluate()`:
1. Records the model's current mode (`was_training = model.training`).
2. Calls `model.eval()` -- propagated to every nested child module by the
   existing M3 `Module.train()`/`eval()` recursion.
3. Runs the forward pass over every batch inside `forge.no_grad()` (see
   **`no_grad`** below) -- no `loss.backward()`, no `optimizer.step()`, and
   no autograd graph is built for the predictions.
4. Restores the model's prior mode (`model.train(was_training)`), in a
   `finally` block so a mid-evaluation exception cannot leave the model
   stuck in eval mode.

This means calling `evaluate()` mid-`fit()` (after a training epoch, model
still in train mode) returns the model to train mode for the next epoch;
calling it standalone (e.g. after training completes, with the model however
the caller left it) preserves that mode exactly.

### `no_grad`
Forge had no context for suspending autograd graph construction before this
milestone. `forge.no_grad()` (`forge/autograd/engine.py`) is the minimal
addition: a single global flag (`is_grad_enabled()`), checked by
`Tensor._differentiable_wrap` in exactly the place it already decides
whether an operation's result requires grad:
```python
requires_grad = is_grad_enabled() and any(t._requires_grad for t in inputs)
```
Inside `with no_grad():`, every differentiable operation produces a plain,
non-grad-requiring result with no `grad_fn`, regardless of its inputs'
`requires_grad` -- so an evaluation forward pass through a model whose
parameters all require grad still builds no graph. This is not a
context-management framework: it is one flag, toggled by one context
manager, restored in `__exit__` (including on exception), with no
per-tensor or per-thread state. See `docs/architecture/autograd.md`.

## Metrics
`Metric` (`forge/training/metrics.py`) is a separate abstraction from
`Loss`:
```text
Loss   -> training objective (differentiable, drives optimizer.step())
Metric -> measurement/reporting (non-differentiable, never touches params)
```
```python
class Metric:
    def reset(self) -> None: ...
    def update(self, prediction, target) -> None: ...
    def compute(self) -> float: ...
```
Trainer calls `reset()` once at the start of a phase (a training epoch or
an `evaluate()` call), `update()` once per batch, and `compute()` once at
the end of the phase. Each built-in metric accumulates a running total
(sum-of-error and element/sample count) rather than a per-batch mean, so
`compute()` after batches of unequal size is exact -- see
**Numerical correctness** below.

Built-in metrics:
- **`MeanSquaredError`** (`name="mse"`): `mean((prediction - target)^2)`
  over every element seen. Requires `prediction.shape == target.shape`.
- **`MeanAbsoluteError`** (`name="mae"`): `mean(|prediction - target|)`,
  same shape requirement and aggregation.
- **`Accuracy`** (`name="accuracy"`): fraction of `argmax(prediction,
  axis=1) == target` over every sample seen. `prediction` is `(batch_size,
  num_classes)`, `target` is `(batch_size,)` integer class indices --
  matching `CrossEntropyLoss`'s convention.

All three raise `TrainerError` on a shape mismatch or on `compute()` with
zero samples seen, rather than returning `nan` silently. None mutate
`prediction`/`target` or touch model parameters.

## Training history
`fit()` returns a `TrainingHistory`: an ordered, indexable, iterable
sequence of `EpochResult` records, one per completed epoch:
```python
@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    train_metrics: dict[str, float]
    val_loss: float | None        # None if no validation_loader was given
    val_metrics: dict[str, float]  # {} if no validation_loader was given
    duration: float                 # wall-clock seconds for the epoch
    samples: int                    # samples processed during training this epoch
    device: str
```
`val_loss`/`val_metrics` are `None`/`{}` (not hard-coded classification- or
regression-specific fields) when `fit()` was called without a
`validation_loader`, so a purely regression or purely classification run
never carries an irrelevant field. `TrainingHistory.train_losses` /
`.val_losses` are convenience per-epoch value lists; `history[i]`,
`len(history)`, and `for record in history` are all supported directly.

`evaluate()` returns the analogous but distinct `EvaluationResult` (`loss`,
`metrics`, `samples`, `duration`, `device`) -- it is not appended to any
history, since a standalone `evaluate()` call is not part of a `fit()` run.

## Progress reporting
Unless `Trainer(..., verbose=False)`, `fit()` prints one block per epoch
after that epoch (training, and validation if configured) completes:
```text
Epoch 1/10
loss: 0.8421
mse: 0.8421
val_loss: 0.7103
val_mse: 0.7103
time: 0.12s
samples/sec: 830
device: cpu
```
This is informational output only, not a stable machine-readable format --
tests assert that output exists/is suppressible, never its exact text.
`verbose` affects printing only; it never changes what is trained or what
`TrainingHistory` records (a `verbose=True` and a `verbose=False` run over
identical data produce identical histories).

## Validation
```python
trainer.fit(train_loader, epochs=10, validation_loader=val_loader)
```
`validation_loader` is optional. When given, it is evaluated via
`self.evaluate(validation_loader)` exactly once, at the end of each training
epoch (never before, never mid-epoch) -- so validation always reflects that
epoch's just-updated parameters. No early stopping and no checkpointing are
implemented on top of this (explicitly out of scope for this milestone).

## Device semantics
Milestone 6 remains CPU-only, matching the rest of Forge at this milestone.
`Trainer(..., device=...)` resolves the device via the existing
`forge.Device.parse` and requires `device.type == "cpu"`; anything else
(`"cuda"`, `"cuda:0"`, or an unrecognized string) raises
`UnsupportedDeviceError` immediately at construction. Trainer never moves a
tensor to another device merely because a device string was accepted
elsewhere in Forge, and never claims to execute on CUDA.

## Epoch semantics
- One epoch is exactly one full pass over `train_loader` -- as many batches
  as it yields (`len(train_loader)`), respecting whatever
  shuffling/`drop_last` that `DataLoader` was configured with. Trainer does
  not reshuffle independently; shuffling is entirely `DataLoader`'s
  responsibility (see `docs/architecture/data-system.md`).
- `epochs <= 0` (or a non-`int`, e.g. a `float` or `bool`) raises
  `TrainerError` immediately -- `fit()` never silently trains zero epochs
  and returns an empty history.
- An empty loader (`len(loader) == 0`) passed to `fit()` (as either
  `train_loader` or `validation_loader`) or to `evaluate()` raises
  `TrainerError` before any batch is processed, rather than producing a
  `nan`/divide-by-zero loss.
- Every epoch that runs is appended to `TrainingHistory` -- there is no
  partial-epoch or early-stopping skip in this milestone.

## Known limitations
Explicitly out of scope for Milestone 6 (see `docs/product/scope.md` and
the milestone's own non-goals): CUDA execution, distributed training, mixed
precision, checkpointing, early stopping, learning-rate schedulers,
hyperparameter tuning, multiprocessing DataLoader workers, a general
logging/observability platform, experiment tracking, a CLI, model
serialization, and a callbacks system. `no_grad()` is a single global flag,
not a general context-management system -- no `retain_graph` equivalent,
no per-tensor/per-thread grad state, no nesting-depth tracking beyond plain
save/restore.
