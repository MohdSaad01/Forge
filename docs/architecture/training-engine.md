# Training Engine (Milestone 6; CUDA device support as of Milestone 12; CUDA classification via `CrossEntropyLoss` as of Milestone 14; checkpointing/resume as of Milestone 18; standalone inference via `predict()` as of Milestone 68; prediction interpretation via `interpret_classification()` as of Milestone 72; fresh-or-resumed session construction via `start_training_session()` as of Milestone 73; portable-artifact save+verify via `save_and_verify()` as of Milestone 78; single-call high-level training via `train()` as of Milestone 79; train-to-verified-artifact via `train_and_save()` as of Milestone 81; single-call portable-artifact inference via `predict_artifact()` as of Milestone 82)

## Package layout
```
forge/
    training/
        trainer.py     Trainer, EpochResult, EvaluationResult, TrainingHistory
        metrics.py     Metric, MeanSquaredError, MeanAbsoluteError, Accuracy
        inference.py   predict() (Milestone 68), interpret_classification(), ClassificationPrediction (Milestone 72), save_and_verify() (Milestone 78), predict_artifact() (Milestone 82)
        session.py     TrainingSession, start_training_session() (Milestone 73)
        api.py         train() (Milestone 79), train_and_save(), TrainAndSaveResult (Milestone 81)
    autograd/engine.py  no_grad, is_grad_enabled (new in this milestone)
```
`forge.training` is exposed as a submodule of `forge` (`forge.training.Trainer`),
alongside `forge.nn`/`forge.optim`/`forge.data`/`forge.random`. `forge.no_grad`
and `forge.predict` are exposed at the top level, alongside `forge.Tensor`.

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
(or `UnsupportedDeviceError`/`CUDAError` for the device) rather than failing
later with an unrelated `AttributeError`:
- `model` must be a `forge.nn.Module` instance.
- `loss_fn` must be a `forge.nn.Loss` instance.
- `optimizer` must be a `forge.optim.Optimizer` instance.
- `device` must resolve (via `forge.Device.parse`) to `"cpu"` or `"cuda"`,
  and that backend must actually be usable right now -- see **Device
  semantics** below.
- `metrics`, if given, must be `Metric` instances with unique `.name`s.

Construction deliberately does **not** validate that `model` already sits on
`device` -- that is checked lazily, at the start of `fit()`/`evaluate()`
(see **Device semantics**), so building a `Trainer` and moving the model
(`model.to(device)`) can happen in either order.

## Training lifecycle
`fit()` runs `_run_training_epoch` once per epoch:
```python
model.train()
for batch in train_loader:
    x, y = batch                 # (features, target), enforced -- see below
    x, y = x.to(device), (y.to(device) if isinstance(y, Tensor) else y)  # M12
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
`loss_fn()`. This validation (`Trainer._unpack_batch`) runs *before* the
Milestone 12 device transfer described above, so an unsupported batch shape
fails with the same clear `TrainerError` regardless of `self.device`.

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
1. Validates the model's device against `self.device` (Milestone 12 -- see
   **Device semantics** below), before touching the loader or the model's
   mode.
2. Records the model's current mode (`was_training = model.training`).
3. Calls `model.eval()` -- propagated to every nested child module by the
   existing M3 `Module.train()`/`eval()` recursion.
4. Runs the forward pass over every batch inside `forge.no_grad()` (see
   **`no_grad`** below) -- no `loss.backward()`, no `optimizer.step()`, and
   no autograd graph is built for the predictions. Each batch is
   device-transferred first, exactly like `fit()`'s training loop.
5. Restores the model's prior mode (`model.train(was_training)`), in a
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

### CUDA metrics (Milestone 12)
Each built-in metric's `update()` is a small, non-differentiable NumPy
reduction (`_as_numpy` in `forge/training/metrics.py`) -- not a good fit for
a dedicated CUDA kernel, and metrics never participate in the autograd graph
in the first place. Rather than leaving them CUDA-incompatible, `_as_numpy`
transfers a `Tensor` argument to CPU first (`value.to("cpu").numpy()`) and
computes exactly as before; a non-`Tensor` argument (e.g. a raw class-index
array) is used as-is. This is a one-way, read-only transfer of
already-computed prediction/target values purely for reporting -- it never
feeds anything back into training, and it never invokes a `CPUBackend`
*compute* method (`Tensor.to("cpu")` is a transfer, going through
`CUDABackend.to_numpy` and `CPUBackend.from_array`, not `CPUBackend.add`/
`matmul`/etc.), so it does not violate Trainer's no-CPU-fallback guarantee
for training computation -- see **Device semantics** below and
`docs/architecture/cuda-backend.md`. All three built-in metrics therefore
work unmodified for a CUDA `Trainer`; there is no CUDA-incompatible
built-in metric in this milestone.

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
Milestone 6 shipped `Trainer` as CPU-only. **As of Milestone 12, `Trainer`
executes a real training/evaluation workflow on CUDA** through the exact
same `Trainer` class -- there is no `CUDATrainer` subclass, and the
`optimizer.zero_grad() -> model(x) -> loss_fn -> loss.backward() ->
optimizer.step()` sequence above is unchanged on either device. This
section documents the three device-related decisions that extension
required: what `device=` validates at construction, how `Trainer` decides
whether the model is where it needs to be, and how a batch produced by a
CPU-only `DataLoader` reaches a CUDA model.

### Construction: device *support*, not model placement
`Trainer(..., device=...)` resolves the device via `forge.Device.parse` (as
before) and now additionally calls `forge.backend.get_backend(device)` to
confirm that backend can actually be used right now:
- An unrecognized device string (anything other than `"cpu"`/`"cuda"`, e.g.
  `"tpu"`) raises `UnsupportedDeviceError` immediately, unchanged from
  Milestone 6.
- `device="cuda"` on a machine without a working CUDA backend (no driver, no
  compatible device, a failed kernel compile) raises `CUDAError`
  immediately -- the same failure `Tensor(..., device="cuda")` or
  `Module.to("cuda")` would raise, surfaced at `Trainer` construction rather
  than deferred to the first batch.

Construction does **not** touch `model` at all: it neither inspects nor
moves it. Building a CUDA `Trainer` around a still-CPU-resident model is
valid; see the next section for when that becomes an error.

### Model placement: Trainer validates, it never moves
Two policies were available (per the milestone brief): have `Trainer`
silently call `model.to(device)` on the caller's behalf, or have `Trainer`
require the model already be there and fail clearly otherwise. **`Trainer`
validates; it does not move.** `Trainer._check_model_device()` -- called at
the start of both `fit()` and `evaluate()`, before any batch is touched --
compares `self.model.device` (the single device shared by every `Parameter`
the model owns; `None` for a `Parameter`-less model, which is exempt from
this check) against `self.device`, raising `UnsupportedDeviceError` naming
the exact `model.to(device)` call needed if they differ.

This was the only policy compatible with an existing Milestone 9 test
(`tests/test_module_cuda.py::test_trainer_configured_for_cpu_rejects_a_cuda_model`,
renamed from Milestone 9's `..._fails_clearly_on_forward` but unchanged in
intent): a `device="cpu"` `Trainer` fed a CUDA-resident model must still
fail clearly, never silently repatriate the model to CPU to make training
"work". Auto-moving would have made that call succeed by quietly relocating
the model -- exactly the "silently move a tensor outside the documented
Trainer lifecycle" behavior the milestone brief prohibits. Validating
instead keeps `Trainer`'s behavior identical however the caller reaches a
device mismatch (constructed with the wrong `device=`, or `model.to()`'d to
the wrong place afterward), and keeps `Module.to(device)` -- already the
one sanctioned, explicit way to move a model (`docs/architecture/modules.md`)
-- the *only* way, with no second, implicit code path duplicating it.

### Batch movement: Trainer transfers, DataLoader never does
`DataLoader` is explicitly **not** made device-aware (per the milestone
brief's non-goals: no GPU `DataLoader`, no automatic batching to a device).
It continues to yield ordinary CPU Tensors, exactly as `docs/architecture/
data-system.md` describes, regardless of what device a `Trainer` consuming
it is configured for.

`Trainer._to_device_batch()` -- the Milestone 12 replacement for a bare
`_unpack_batch()` call inside `fit()`'s training loop and `evaluate()` --
unpacks the batch as before and then explicitly transfers it:
```python
x, y = self._unpack_batch(batch)
x = x.to(self.device)
if isinstance(y, Tensor):
    y = y.to(self.device)
return x, y
```
`x` (always a `Tensor`, enforced by `_unpack_batch`) is always transferred.
`y` is transferred only if it is itself a `Tensor` -- a raw array-like
target (which `MSELoss`/`CrossEntropyLoss`/the built-in metrics already
accept directly) is passed through unchanged, since `.to()` has no meaning
for it. `Tensor.to(device)` is a no-op returning the original object when
already on `device` (`docs/architecture/cuda-backend.md`), so this costs a
CPU `Trainer` nothing extra and introduces no new code path for the
CPU-only case. This is the *only* place in Forge a `DataLoader`-produced
batch crosses a device boundary -- never inside `Dataset`/`DataLoader`
itself, matching the target flow:
```text
Dataset -> CPU DataLoader -> Trainer -> explicit x.to(device)/y.to(device)
    -> CUDA Module -> CUDA Loss -> CUDA autograd -> CUDA SGD
```

### CUDA losses through Trainer
`Trainer` calls `self.loss_fn(prediction, y)` exactly as before -- it has no
CUDA-specific loss-handling code. Whether that succeeds on CUDA is entirely
the loss's own concern: both built-in losses now work unmodified --
`MSELoss` since Milestone 12, `CrossEntropyLoss` since Milestone 14 (see
`docs/architecture/cuda-backend.md`'s **CUDA losses**/**CUDA
CrossEntropyLoss** sections). `Trainer(device="cuda",
loss_fn=CrossEntropyLoss())` therefore trains a real classification model
end-to-end on CUDA -- `Trainer` never needed to know which loss it was
calling, in either milestone.

### Reporting the loss/metrics still touches CPU -- deliberately
`total_loss += float(loss.to("cpu").numpy()) * batch_size` (both `fit()`'s
training loop and `evaluate()`) and metrics' `_as_numpy` (see **CUDA
metrics** above) both call `.to("cpu")` on a CUDA scalar/tensor. This is the
one sanctioned host round-trip in the whole CUDA training path: extracting
an already-computed Python `float`/NumPy value for bookkeeping
(`EpochResult`/`EvaluationResult`/metric accumulation), never recomputing
anything on CPU. `Tensor.to("cpu")` goes through `CUDABackend.to_numpy` (a
`cudaMemcpy` D2H) and `CPUBackend.from_array` (a transfer primitive) -- it
never calls a `CPUBackend` *compute* method (`add`/`sub`/`mul`/`matmul`/
`sum`/`reshape`/`relu`/`exp`/`log`/`sgd_step`), so it does not compromise
the no-CPU-fallback guarantee below.

### No CPU fallback
For a `Trainer(device="cuda")`, every computation in the
`zero_grad -> forward -> loss -> backward -> step` sequence executes as a
real CUDA operation: `CUDABackend` forward/backward kernels for the model
and loss, the existing backend-aware autograd engine
(`docs/architecture/autograd.md`) dispatching to `CUDABackend`, and
`CUDABackend.sgd_step` for the optimizer update. `tests/test_trainer_cuda.py`
asserts this structurally (monkeypatching every `CPUBackend` compute method,
the same technique `tests/test_module_cuda.py`/`tests/test_cuda_autograd.py`
already use) across a full multi-epoch `fit()` call with validation and
metrics: zero `CPUBackend` compute-method calls occur.

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
- As of Milestone 18, `record.epoch` is `self.epoch` (a persistent counter
  that survives across multiple `fit()` calls and across a checkpoint
  resume), not a call-local `1..epochs` index -- see **Checkpointing and
  resume** below.

## Checkpointing and resume (Milestone 18)
`Trainer` owns two plain, mutable training-progress counters, initialized
to `0` at construction:
```text
self.epoch          # completed epochs, across every fit() call so far
self.global_step     # completed optimizer.step() calls, across every fit() call so far
```
`fit()` increments `self.epoch` once per epoch (rather than counting a
call-local `1..epochs` range) and records that running value as each
`EpochResult.epoch`; `_run_training_epoch` increments `self.global_step`
once per training batch, immediately after `optimizer.step()`. This makes a
second `fit()` call on the same `Trainer` -- whether a plain continuation or
after `resume()` -- number its epochs and count its steps starting from
wherever the previous call left off, with no separate "resume" code path
inside `fit()` itself: it is the exact same
`DataLoader -> forward -> loss -> backward -> optimizer.step()` sequence
either way.

```python
trainer.save_checkpoint(path)                 # -> forge.serialization.save_checkpoint(path, trainer.model, trainer.optimizer, epoch=trainer.epoch, global_step=trainer.global_step)
checkpoint = forge.load_checkpoint(path, device=...)
trainer2 = Trainer(model=..., loss_fn=..., optimizer=..., device=...)  # any model/optimizer, replaced next
trainer2.resume(checkpoint)                    # trainer2.model/optimizer/epoch/global_step <- checkpoint's
trainer2.fit(loader, epochs=2)                 # epochs continue from trainer2.epoch, not from 1
```
`resume()` validates that `checkpoint.model`'s device matches `self.device`
(the same "validate, never move" policy `_check_model_device` already
enforces for ordinary `fit()`/`evaluate()`) and otherwise only reassigns
`self.model`/`self.optimizer`/`self.epoch`/`self.global_step` -- `loss_fn`,
`device`, `metrics`, and `verbose` stay whatever the `Trainer` was
constructed with. See `docs/architecture/persistence.md`'s
**Checkpointing** section for the full save/load format, the RNG/
determinism policy, and what `TrainingHistory` does *not* carry across a
resume (a fresh `TrainingHistory` starts at the resumed epoch numbers; nothing
concatenates it with a prior call's history automatically).

## Inference (Milestone 68)
`Trainer` deliberately requires a `Loss` and an `Optimizer` at construction
-- both are meaningless once a model is trained, but every `mnist`/
`regression`/`resnet`/`segmentation`/`autoencoder`/`waveform_classification`
example that reached the "load a saved model and run it on new data" point
had independently hand-rolled the same four-line sequence to do it without
constructing a full `Trainer` just to call it once:
```python
with no_grad():
    prediction = model(x.to(device)).to("cpu").numpy()
```
`predict()` (`forge/training/inference.py`) is that sequence, written once,
as a free function -- not a `Trainer` method (no `Loss`/`Optimizer` to own)
and not a `Module.predict()` (would blur the "what a Module computes" vs.
"how it's orchestrated" boundary `Trainer` itself exists to preserve):
```python
model = forge.load_model("model.forge", device="cuda")
prediction = forge.predict(model, x)              # x: a single Tensor
predictions = forge.predict(model, test_loader)    # a DataLoader
```
1. Validates `model` is a `forge.nn.Module` (`TrainerError` otherwise).
2. Resolves the compute device: an explicit `device=` argument, else
   `model.device` (inferred, not assumed), else `"cpu"` for a
   Parameter-less model -- the same edge-case exemption
   `Trainer._check_model_device` already uses.
3. Records `model.training`, calls `model.eval()`.
4. Runs the forward pass(es) inside `forge.no_grad()` -- identical to
   `evaluate()`'s own inference discipline.
5. Restores the model's prior training mode in a `finally` block, exactly
   like `evaluate()`.
6. Returns the result as a `Tensor` already on `"cpu"` -- ready for
   `.numpy()`, comparison, or display with no further device/autograd
   bookkeeping.

**Two input shapes.** A single `Tensor` returns a single `Tensor` (one
forward pass). An iterable of batches (a `DataLoader`, or anything yielding
a bare `Tensor` or a `(features, ...)` tuple) runs the model over every
batch and returns the per-batch outputs concatenated into one `Tensor`, in
iteration order. Concatenation happens on already-materialized NumPy arrays
(`np.concatenate`, then wrapped back into one `Tensor`) -- there is no
`Tensor.cat` primitive in Forge, and adding one purely for this
non-differentiable, inference-time convenience would be exactly the kind of
speculative core primitive `docs/development/m49-capability-assessment.md`
already rejected for an unrelated op; `Metric`'s own `_as_numpy` host-side
reductions are the same precedent applied to a different consumer.

**Not a `Trainer` method.** `Trainer.fit()`/`evaluate()` both need a `Loss`
to report a meaningful loss value; pure post-training inference has no loss
to compute and, per `docs/product/vision.md`'s own workflow
(`load -> give it new data -> receive a prediction`), should not require
constructing one. `predict()` therefore takes only `model`
(+ `inputs`/`device`) and reuses none of `Trainer`'s internal state --
callers who already have a live `Trainer` can call `forge.predict(trainer.model,
x)` exactly as anyone else would.

**Real consumers.** Every single-forward-per-step example's own
model-persistence round-trip check (`mnist`, `regression`, `resnet`,
`segmentation`, `autoencoder`, `waveform_classification`) now calls
`predict()` instead of hand-rolling the block above -- see each example's
`train.py`. The three hand-written-training-loop examples (`char_rnn`,
`word_rnn`, `long_range_recall`) do not use `predict()`: their per-timestep
`.step()` inference does not fit `predict()`'s single-forward-call shape,
and forcing it to would be exactly the kind of speculative generalization
`docs/product/scope.md` warns against building without a real consumer.
`char_rnn`/`word_rnn` do share a different, purpose-built inference helper
for their own multi-step shape -- see **Sequence generation** below.

## Sequence generation: `generate_sequence()` (Milestone 75)
`predict()` covers one forward pass per call; a stepwise recurrent language
model's actual inference need is *autoregressive generation* -- prime a
hidden state over a seed sequence, then repeatedly sample a next token from
the model's own output distribution and feed it back in as the next input.
`examples/char_rnn/train.py` and `examples/word_rnn/train.py` each
independently hand-wrote this exact loop (`generate()`); once `nn.Embedding`
made `word_rnn` a real second consumer, the two functions turned out
structurally identical except for how a single token is encoded into a
model input and decoded back out of a sampled class index. `forge.training.
generate_sequence()` (`forge/training/inference.py`) is that shared loop,
extracted once:
```python
generated = forge.generate_sequence(
    model, seed=list("a tensor"),
    encode=lambda ch: Tensor(one_hot(vocab.encode(ch), vocab.size)),
    decode=lambda idx: vocab.decode([idx]),
    length=200,
)
```
`model` must implement the informal stepwise-recurrence protocol every
Forge sequence example already follows: `model.init_hidden(batch_size,
device=...) -> state` and `model.step(x, state) -> (logits, state)` for a
single timestep, batch size 1. This is deliberately duck-typed rather than a
new base class or `typing.Protocol` -- two independent, pre-existing
consumers already share the exact shape, which is what justifies extracting
it at all, not a new formalized interface no third consumer has asked for.

`encode`/`decode` stay caller-supplied callables rather than framework
machinery: `char_rnn` one-hot-encodes a character and joins characters back
into a string, `word_rnn` looks up a word's embedding index and returns a
single decoded word -- genuinely different per model, and composition
(passing two small functions) is simpler than inventing a vocabulary
abstraction neither example's own `Vocab` class needs replacing. Sampling
softmaxes `model.step()`'s logits on the host (plain NumPy, mirroring
`interpret_classification()`'s and `Metric`'s own non-differentiable
host-side reductions) and draws from that distribution via `rng.choice`
-- greedy argmax decoding was never what either example did. `rng` defaults
to `forge.random.default_generator()` (the same default-argument convention
`random_split()` already uses) when omitted. Runs under eval mode and
`forge.no_grad()`, restoring the model's prior training mode afterward,
exactly like `predict()`.

`examples/char_rnn/train.py::generate()` and
`examples/word_rnn/train.py::generate()` are now both thin wrappers over
`generate_sequence()` -- their existing test suites
(`tests/test_char_rnn_example_integration.py`,
`tests/test_word_rnn_example_integration.py`) pass unmodified against the
refactor, and both examples' real end-to-end runs (`python -m
examples.char_rnn.train` / `python -m examples.word_rnn.train`) produce the
same shape of output as before.

## Prediction interpretation: `interpret_classification()` (Milestone 72)
`predict()` deliberately stops at a raw `Tensor` -- it has no way to know
what a classification model's output *indices mean*. `forge.training.
interpret_classification(output, classes)` is the next step in the pipeline,
turning that raw `Tensor` plus a class-name vocabulary (`forge.
load_classes()`, or any `list[str]` such as `ImageFolder.classes`) into
`ClassificationPrediction(label, index, confidence)` per row -- see
`docs/architecture/persistence.md`'s **Class-label metadata** section for
the full contract, including why its `confidence` (a softmax probability) is
a legitimate reading of `output` rather than an invented display value, and
why the check "does `output`'s width match `len(classes)`" belongs here
(interpretation time) rather than in `save_model()` (save time, when the
model's actual output width is not yet knowable from a class list alone).
`examples/image_folder_classification/train.py`/`infer.py` and `forge model
predict` (`forge/cli/model.py`) are its real consumers.

## Reusable training sessions: `start_training_session()` (Milestone 73)
By Milestone 72, seven of Forge's checkpoint-capable examples (`mnist`,
`regression`, `resnet`, `segmentation`, `autoencoder`,
`waveform_classification`, `image_folder_classification`) each hand-rolled
an identical ~10-line branch in their own `train.py`:
```python
if args.resume:
    checkpoint = load_checkpoint(args.resume, device=args.device)
    model, optimizer = checkpoint.model, checkpoint.optimizer
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[...])
    trainer.resume(checkpoint)
else:
    model = build_model().to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[...])
```
`forge.training.start_training_session()` (`forge/training/session.py`) is
that branch, written once, returning a `TrainingSession(trainer,
data_loader_rng, resumed, checkpoint)`:
```python
session = start_training_session(
    build_model=lambda: build_model(num_classes=...),
    build_optimizer=lambda params: Adam(params, lr=args.lr),
    loss_fn=CrossEntropyLoss(),
    seed=args.seed,
    device=args.device,
    metrics=[Accuracy()],
    resume=args.resume,
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           generator=session.data_loader_rng)
history = session.trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
session.save_checkpoint(str(checkpoint_path))
```
`build_model`/`build_optimizer` are factories (`build_optimizer` receives
`model.parameters()`) called only on the fresh-run path -- a resumed session
takes its model/optimizer from the checkpoint instead, exactly as the
hand-written branch above did. `loss_fn`/`device`/`metrics`/`verbose`/
`prefetch`/`prefetch_size` pass straight through to `Trainer(...)`; this
function adds no configuration surface of its own beyond what `Trainer`
already validates.

**Closing a second, latent bug.** Milestone 65 found and fixed a real
reproducibility gap in `regression/train.py`: with `shuffle=True` (every
example's actual default), resuming re-seeded a *fresh* `--seed`-derived
`DataLoader` generator instead of continuing the interrupted run's shuffle
stream, silently diverging from what continuous training would have
produced. The fix -- save `data_rng.bit_generator.state` into
`save_checkpoint(..., extra=...)`, restore it on resume -- was copied by
hand into `resnet` but never into `mnist`, `segmentation`, `autoencoder`,
`waveform_classification`, or `image_folder_classification`, each of which
carried the same latent bug M65 had already diagnosed once.
`TrainingSession.data_loader_rng` is built the same way
(`numpy.random.default_rng(seed)`), and `TrainingSession.save_checkpoint()`
folds its current `bit_generator.state` into the checkpoint's `extra` dict
automatically (raising `TrainerError` if the caller's own `extra` already
defines that key) -- restored automatically by the next
`start_training_session(..., resume=path)` call. An example that builds its
training `DataLoader` from `session.data_loader_rng` and saves through
`session.save_checkpoint()` gets M65's fix for free, with nothing to
remember per example. A checkpoint saved via plain `Trainer.
save_checkpoint()` (bypassing `TrainingSession`) has no
`"data_loader_rng_state"` key -- resuming from one leaves `data_loader_rng`
at its freshly-`seed`-derived state, matching every example's pre-Milestone-
73 behavior when no such key was ever saved.

This reuses Milestone 65's existing mechanism (a JSON-safe `numpy.random.
Generator` state living in `save_checkpoint(..., extra=...)`'s pre-existing
caller-defined dict) rather than building a second reproducibility system --
see `forge/serialization/checkpoint.py`'s **RNG / determinism policy**
section for what a checkpoint's own `forge.random` state does and does not
cover (exactly the gap `data_loader_rng` closes).

**What this does not do.** `start_training_session()` never constructs a
`DataLoader`, never calls `fit()`/`evaluate()`, and never reads a dataset --
it returns a ready `Trainer` (already `.resume()`d, when resuming) plus a
`data_loader_rng`; the caller still builds its own `DataLoader`s and drives
training itself. It is not a config object or a YAML-driven system -- every
argument is a plain keyword matching an existing `Trainer`/`load_checkpoint`
parameter, not a new configuration surface. `forge/training/trainer.py`
itself is unmodified.

**Real consumers.** `examples/image_folder_classification/train.py` (this
milestone's primary, required consumer per its own brief) and
`examples/regression/train.py` (retrofitted second consumer, chosen because
it already had the hand-written `data_loader_rng_state` logic this function
absorbs -- its existing dedicated reproducibility test suite,
`tests/test_regression_reproducible_training.py`, passes unmodified against
the refactored script, proving behavioral equivalence). `mnist`, `resnet`,
`segmentation`, `autoencoder`, and `waveform_classification` were not
retrofitted this milestone (mechanical, low-risk, and left as a natural
follow-up) but `regression`'s retrofit demonstrates the identical pattern.

## Portable-artifact save + verify: `save_and_verify()` (Milestone 78)
Every Trainer-based example (`mnist`, `regression`, `resnet`, `autoencoder`,
`segmentation`, `waveform_classification`, `image_folder_classification`)
independently hand-wrote the identical closing sequence once training
finished: `save_model()`, then `load_model()` a fresh copy back, then
`predict()` each and compare with `numpy.allclose(..., atol=1e-5)` behind a
bare `assert` -- the literal proof that `docs/product/vision.md`'s "save as a
portable artifact... load it later... receive a useful result" workflow
actually holds for the file just written, not just the in-memory model.
`forge.training.save_and_verify()` (`forge/training/inference.py`) is that
sequence, written once:
```python
reloaded = forge.save_and_verify(
    trainer.model, str(model_path), query_x,
    preprocessing=build_transform(), classes=full_dataset.classes,
)
result = interpret_classification(predict(reloaded, new_image), reloaded_classes)
```
1. `predict(model, sample, device=...)` -- the pre-save prediction, from the
   live, just-trained model.
2. `save_model(model, path, preprocessing=preprocessing, classes=classes)`
   (Milestones 71/72's unmodified persistence call -- no new serialization
   logic).
3. `load_model(path, device=...)` -- a genuinely fresh reconstruction.
4. `predict(reloaded, sample)` -- the post-load prediction.
5. `numpy.allclose(pre_save, post_load, atol=atol)`; a mismatch raises
   `PersistenceError` (a real, catchable error -- not an `assert` a caller
   could lose under `python -O`) naming the max absolute difference.

Returns the freshly **reloaded** `Module`, not the original -- so whatever a
caller does next (an interpretation demo, a reconstruction image) genuinely
exercises the file on disk, not lingering in-memory state.

**Scope.** Composes exactly `save_model()` + `load_model()` + `predict()`,
so it covers exactly `predict()`'s calling convention (one batched `Tensor`
forward pass) -- it is orthogonal to, and does not touch,
`Trainer.save_checkpoint()`/`TrainingSession.save_checkpoint()` (checkpoint
persistence is a separate, resumable-training concern; a caller calls both,
exactly as every retrofitted example's `train.py` does). It does not cover
the stepwise-recurrence sequence models (`char_rnn`/`word_rnn`/
`long_range_recall`), which verify via `model.step(x, state)` -- a different
calling convention `predict()` itself was never built for (see **Inference**
above) -- those three examples keep their own independent hand-written check
rather than being forced into a shape that does not fit them.

**Real consumers.** All seven Trainer-based examples were retrofitted --
`image_folder_classification` (the primary, required consumer per this
milestone's own brief) and `mnist` (Forge's flagship example, and the first
place `save_and_verify()`'s return value is reused for interpretation, not
just discarded after the check) are the two most representative;
`regression`/`resnet`/`autoencoder`/`segmentation`/`waveform_classification`
confirm the same call shape holds across every remaining architecture and
persistence-metadata combination (with/without `preprocessing=`,
with/without `classes=`, `Sequential` and custom-registered `Module` trees).

**Fresh-process verification (Milestone 80).** `save_and_verify()` itself
only proves an artifact survives a reload *within the same process*.
Milestones 71/72/77/78's own reports additionally described `infer.py`
(`mnist`/`image_folder_classification`) as verified in "a genuinely separate
process" -- true of the manual runs recorded in each milestone's own
session, but not, until Milestone 80, of anything the automated test suite
itself enforced: every existing `infer.py` test imported `infer.run()` into
the *same* test process rather than launching a real second one.
`tests/test_mnist_example_integration.py::
test_infer_cli_runs_in_a_genuinely_separate_process` and `tests/
test_image_folder_classification_integration.py::
test_infer_cli_runs_in_a_genuinely_separate_process` close that gap: each
launches `python -m examples.<name>.infer` via `subprocess.run()` -- a real,
separate OS process with no shared memory, module cache, or import state --
and asserts its stdout matches `predict()`/`interpret_classification()`
computed independently in the test process against the same file. The
`image_folder_classification` version also drives the real `train.py::
main()` entry point first (not a hand-built `Trainer`), so it additionally
exercises the script's own argument parsing and wiring, which no prior test
touched.

## Single-call high-level training: `train()` (Milestone 79)
Every Trainer-based example's `train.py` still starts with the same
hand-assembled sequence before it ever calls `.fit()`:
```python
loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
history = trainer.fit(loader, epochs=epochs, validation_loader=val_loader)
```
`forge.training.train()` (`forge/training/api.py`) is that sequence, written
once, as a thin orchestration layer over `DataLoader`/`Trainer` -- it builds
no gradients, updates no parameters, and implements no batching of its own:
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
`loss`/`optimizer`/`epochs` are required and keyword-only -- no automatic
loss or optimizer selection, matching this milestone's own explicit scope.
`model` must be a `forge.nn.Module`, `loss` a `forge.nn.Loss`, `optimizer` a
`forge.optim.Optimizer` -- validated by the `Trainer` this function
constructs internally, raising `TrainerError` exactly as a direct
`Trainer(...)` call would.

**Dataset, not just DataLoader.** `dataset`/`validation_dataset` each accept
either a plain `forge.data.Dataset` (wrapped in a fresh
`DataLoader(batch_size=batch_size, shuffle=...)` -- `shuffle` for the
training set, always `False` for validation) or an already-built
`DataLoader`, used exactly as given. This is deliberately the *only* other
shape accepted -- `train()` does not re-expose `DataLoader`'s full
constructor (`drop_last`, a custom `generator`, CUDA prefetch); a caller
needing any of those builds the `DataLoader` itself and passes it in place
of a `Dataset`, exactly like `examples/mnist/train.py`'s own retrofit does
to keep its `data_rng`-seeded shuffle generator independent of `forge.
random`'s own stream (see **Real consumer** below).

**Device is one keyword.** `device` defaults to `model`'s current device
(`"cpu"` for a `Parameter`-less fresh model); an explicit `device=` calls
`model.to(device)` **in place** before training -- the one deliberate
difference from `Trainer`'s own "validate, never move" policy (`Trainer`
itself is unmodified by this milestone). This is safe because `Module.to()`
moves every `Parameter` in place, preserving Python identity
(`docs/architecture/modules.md`), so an `optimizer` already constructed from
`model.parameters()` before the `train()` call remains valid afterward. This
is exactly the `model = build_model().to(args.device)` line every existing
example already writes by hand, folded into one keyword.

**Returns `Trainer.fit()`'s own `TrainingHistory`** -- there is no separate
high-level result type, and `train()` does not return the model (the caller
already holds the reference it passed in; `model`/`optimizer` are mutated in
place by training, the same convention `Trainer.fit()` itself uses).

### What `train()` does not cover: checkpoint/resume
`TrainingSession`/`start_training_session()` (Milestone 73) already own the
fresh-or-resumed-`Trainer` workflow, and its resume path fundamentally
*replaces* the model/optimizer a caller passed in with whatever the
checkpoint saved. Grafting that onto `train()`'s "you already built
`model`/`optimizer`, train them" shape would mean either silently discarding
the caller's `model`/`optimizer` on a resumed call (a surprising, easy-to-
miss behavior change for a function whose whole premise is "the model you
passed is the model that trains") or reintroducing
`start_training_session()`'s `build_model`/`build_optimizer` factories --
which would make `train()` no simpler than the workflow it exists to
replace, i.e. a second checkpoint abstraction, exactly what this milestone's
brief warns against building. `train()` therefore has no `checkpoint=`/
`resume=` parameter at all: a resumable run still uses
`start_training_session()` + `Trainer` directly.

### Real consumers: `examples/mnist/train.py` and `examples/image_folder_classification/train.py`
Forge's flagship example is `train()`'s first real consumer. A fresh
(non-`--resume`) run now trains through `forge.train()`; `--resume` keeps
using a plain `Trainer` + `trainer.resume(checkpoint)` exactly as before --
the two paths live side by side in one script, the clearest real
demonstration of where the high-level and low-level training APIs each
apply.

Because `train()` does not track `epoch`/`global_step`, `examples/mnist/
train.py`'s fresh branch derives them from the returned `TrainingHistory`
(`epoch = len(history)`, `global_step = epoch * len(train_loader)` -- exactly
what a fresh `Trainer.fit(epochs=...)` would have counted) and calls
`forge.save_checkpoint()` (the free function, not `Trainer.save_checkpoint
()`) directly -- still the same checkpoint format, no new persistence logic.
The script's final-evaluation print also now reads `history[-1].val_loss`/
`val_metrics` (already computed once per epoch via `validation_dataset=
test_loader`) instead of a second, redundant `trainer.evaluate(test_loader)`
call recomputing the identical number.

**Milestone 80 extended this to `image_folder_classification`** -- the other
candidate `train()`'s own Milestone 79 brief suggested, and originally
**not** retrofitted there because, since Milestone 73, its `--resume` path
gets exact resume-equivalence via `start_training_session()`'s
`data_loader_rng_state` checkpoint field (**Reusable training sessions**
above), and a naive `train()` retrofit of the *whole* script (both branches,
as `start_training_session()` handled them together) would have produced a
checkpoint with no such field, silently downgrading the first `--resume`
after a fresh run to lose exact shuffle-continuity -- exactly the M65-class
regression this codebase has fixed once already and takes seriously.

The resolution: split this script into the same `--resume`-or-fresh
two-branch shape `mnist` already has, matching `mnist`'s finding that only
the *fresh* half needs retrofitting. The fresh branch builds its own
`data_loader_rng = numpy.random.default_rng(seed)` (exactly what
`start_training_session()` would have built internally), trains via
`forge.train(model, DataLoader(train_ds, generator=data_loader_rng), ...)`,
and saves its checkpoint via plain `forge.save_checkpoint(..., extra=
{"data_loader_rng_state": data_loader_rng.bit_generator.state})` -- the
identical key `TrainingSession.save_checkpoint()` itself writes. The
`--resume` branch is untouched: it still calls `start_training_session
(resume=args.resume, ...)`, whose resume path reads
`checkpoint.extra["data_loader_rng_state"]` regardless of which of the two
call sites wrote it. This closes the fresh-path -> resume shuffle-continuity
gap with **zero new persistence logic and zero change to `start_training_
session()`/`Trainer`/`train()` themselves** -- a pure composition of already
-public building blocks -- and is verified directly by
`tests/test_image_folder_classification_integration.py::
test_resume_after_a_forge_train_fresh_run_matches_continuous_training`,
which trains a fresh run through `forge.train()`, resumes it through
`start_training_session()`, and confirms the result is bit-for-bit
equivalent (`atol=1e-5`) to one continuous run at `shuffle=True` -- this
example's real default, and the specific case Milestone 65 originally found
broken elsewhere. See `docs/development/m80-train-to-artifact-workflow.md`.

## Train, evaluate, persist, verify in one call: `train_and_save()` (Milestone 81)

Both `examples/mnist/train.py` and `examples/image_folder_classification/
train.py`'s fresh (non-`--resume`) paths call `train()` and then immediately
`save_and_verify()` on the result -- the same two calls, in the same order,
in both scripts (with `forge.save_checkpoint()` in between, an orthogonal,
resumable-training concern -- see **Portable-artifact save + verify** above).
`forge.training.train_and_save()` (`forge/training/api.py`) is that pair,
written once:
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
result.history       # train()'s own TrainingHistory
result.val_metrics    # the last epoch's validation metrics, e.g. {"accuracy": 0.97}
result.model          # save_and_verify()'s freshly reloaded, verified Module
```
Calls `train()` then `save_and_verify()` exactly once each, in that order,
with no new validation/training/persistence logic -- every condition either
function documents (required `loss`/`optimizer`, `sample` already batched,
`TrainerError`/`DataError`/`PersistenceError` on the same conditions) applies
unchanged. `TrainAndSaveResult.val_loss`/`val_metrics` are copied from the
*last* `EpochResult` in `result.history` -- already computed once per epoch
via `validation_dataset=`, not recomputed by a second evaluation pass (a
`forge.evaluate()`-style standalone post-training evaluation call was
considered and rejected in Milestone 80's own report for exactly this
reason: every real consumer already gets per-epoch evaluation for free).

**Real consumers.** Both `examples/mnist/train.py` and `examples/
image_folder_classification/train.py`'s fresh paths now call
`train_and_save()` in place of their separate `train()` + `save_and_verify()`
calls; `--resume` in both scripts (which has no `train()` call for
`train_and_save()` to wrap) keeps calling `save_and_verify()` directly.

**A real hazard found and fixed along the way.** Retrofitting
`image_folder_classification` (whose model uses `Dropout`) surfaced a real,
previously-invisible bug: `load_model()` reconstructs a module tree by
calling each registered type's ordinary constructor (e.g. `Conv2d.__init__`),
which draws an initial-weights sample from `forge.random.default_generator()`
before the loaded parameter values overwrite it -- a wasted draw that
nonetheless *advances* the global generator, silently shifting whatever a
caller draws next (e.g. the next `Dropout` step in an ongoing training run).
Moving `save_and_verify()`'s reload to run immediately after `train()`
(inside `train_and_save()`), instead of after a separate `forge.
save_checkpoint()` call as both examples previously had it, changed exactly
what `forge.random` state that checkpoint recorded -- and broke
`tests/test_image_folder_classification_integration.py::
test_resume_after_a_forge_train_fresh_run_matches_continuous_training`'s
bit-for-bit resume-equivalence guarantee. The real fix was in `load_model()`
itself (`forge/serialization/model.py`): snapshot `forge.random`'s state
before reconstructing the module tree and restore it immediately after (the
same `get_state()`/`set_state()` mechanism `forge.serialization.checkpoint`
already uses) -- loading a model for inference/verification is now a pure
read with no observable effect on unrelated future random draws, regardless
of when it happens relative to a checkpoint save. See
`tests/test_serialization.py`'s **load_model() must not perturb forge.random**
tests and `docs/development/m81-train-to-verified-artifact-workflow.md`.

## Portable-artifact inference in one call: `predict_artifact()` (Milestone 82)

`examples/image_folder_classification/infer.py` and `forge model predict`
(`forge/cli/model.py`) each independently hand-wrote the identical closing
sequence a developer holding a `.forge` file otherwise has to reconstruct by
hand: `load_preprocessing()`, `load_model()`, decode the new image via
`ImageFolder._load_image()`, apply the preprocessing, add a batch dimension,
`predict()`, `load_classes()`, `interpret_classification()` -- exactly the
internal framework knowledge `docs/product/vision.md`'s portable-artifact
workflow is supposed to hide from a consumer of the file, not just its
producer. `forge.training.predict_artifact()` (`forge/training/inference.py`)
is that sequence, written once:
```python
result = forge.predict_artifact("model.forge", "new_photo.jpg")
print(f"Prediction: {result.label}")
print(f"Confidence: {result.confidence:.1%}")
```
1. `load_preprocessing(path)` -- raises `PersistenceError` if `path` was
   saved with no preprocessing configuration; there is no automatic way to
   prepare an arbitrary new image otherwise (the same hidden-assumption
   failure mode Milestone 71 closed).
2. `load_model(path, device=device)` -- a fresh reconstruction, honoring
   `load_model()`'s own default-to-saved-device / explicit-override policy.
3. `ImageFolder._load_image(Path(image))` -- the same single-image decode
   step `ImageFolder.__getitem__` itself uses (see that method's own
   docstring: "the right place to reuse... if a caller ever needs to
   preprocess one arbitrary image file the exact same way `ImageFolder`
   does").
4. The reconstructed `preprocessing(raw)`, then a leading batch dimension.
5. `predict(model, batch)`.
6. `load_classes(path)` -- when present, `interpret_classification(output,
   classes)[0]` turns the raw output into a `ClassificationPrediction`; when
   absent, the raw predicted class index is returned as a plain `int`
   instead of fabricating a placeholder label for an artifact that never
   recorded a class vocabulary (a real, valid state -- see
   `load_classes()`'s own docstring -- not an error condition).

**Input.** `image` must be a path (`str` or `os.PathLike`) to one image file
on disk -- the one input shape a saved artifact's persisted preprocessing can
already fully describe end-to-end. Anything else raises `forge.DataError`
before any file I/O, rather than failing deep inside image decoding.

**Scope.** Composes exactly `load_model()` + `load_preprocessing()` +
`load_classes()` + `ImageFolder._load_image()` + `predict()` +
`interpret_classification()`, each called unchanged -- no new artifact
format, generic input abstraction, task registry, or model-serving machinery.
This is deliberately narrower than "any Forge artifact": it covers the one
artifact shape Forge can currently fully describe end-to-end
(image-classification models saved with `preprocessing=`), not a generic
runtime for arbitrary model/input combinations (tabular rows, raw sequences,
and stepwise-recurrence models all shape differently, and none has a single
canonical "new input file" convention the way an image does).

**Real consumers.** `examples/image_folder_classification/infer.py::run()`
and `forge/cli/model.py::cmd_predict()` both now delegate to
`predict_artifact()` instead of independently hand-assembling the same
sequence -- the Python API and the CLI share the exact same inference path.
`examples/image_folder_classification/train.py`'s own end-of-run demo on a
brand-new, never-trained-on-resolution image (Section 12) also now calls
`forge.predict_artifact(model_path, new_image_path)` directly instead of
manually reloading preprocessing and re-running `predict()`/
`interpret_classification()` itself.

**Fresh-process verification.** `tests/test_artifact_inference.py::
test_predict_artifact_works_from_a_genuinely_separate_process` launches a
real `subprocess` that only imports `forge` and calls `forge.
predict_artifact()` directly (built from pre-registered `forge.nn` types, so
no custom `register_module()` call from the test module needs to exist in the
fresh process) -- proving the `.forge` file alone, with no in-memory state
from training, is enough. `tests/test_image_folder_classification_
integration.py::test_infer_cli_runs_in_a_genuinely_separate_process`
(Milestone 80) independently exercises the same code path via `infer.py`'s
own subprocess.

See `docs/development/m82-artifact-inference.md`.

## Known limitations
Explicitly out of scope for Milestone 6 (see `docs/product/scope.md` and
the milestone's own non-goals): distributed training, mixed precision,
early stopping, learning-rate schedulers, hyperparameter tuning,
multiprocessing DataLoader workers, a general logging/observability
platform, experiment tracking, a CLI, and a callbacks system. As of
Milestone 18, checkpointing/training-resume is also no longer on this list
(see **Checkpointing and resume** above) -- but `TrainingHistory` itself is
still not persisted or resumed automatically. `no_grad()` is a single
global flag, not a general
context-management system -- no `retain_graph` equivalent, no
per-tensor/per-thread grad state, no nesting-depth tracking beyond plain
save/restore.

As of Milestone 12, CUDA execution is no longer on this list, but the CUDA
training path itself has real, deliberate limits:
- **No GPU `DataLoader`, pinned memory, async prefetch, or multiprocessing
  workers** -- explicitly out of scope per the milestone brief; `DataLoader`
  stays exactly as capable (and exactly as CPU-only) as Milestone 5 left it.
- **CUDA persistence (Milestone 13).** `save_model()` now saves a
  CUDA-trained model directly, with no `model.to("cpu")` step required
  first; `load_model()` restores it back onto CUDA by default (or `"cpu"`/
  `"cuda"` explicitly via `device=`). `Trainer` itself gained no new
  persistence behavior -- saving/loading a `Trainer`-trained model is still
  a call the caller makes explicitly, before/after `fit()`, never something
  `Trainer` does automatically. See `docs/architecture/persistence.md` and
  `docs/architecture/cuda-backend.md`'s **CUDA model persistence** section.
- **No automatic device placement beyond the validate-not-move policy** --
  `Trainer` never calls `model.to(device)` on the caller's behalf, on either
  device, for any reason.
- **Single GPU, index 0 only**, inherited unchanged from
  `docs/architecture/cuda-backend.md`'s existing CUDA backend limitation --
  `Trainer` introduces no new multi-GPU behavior.
