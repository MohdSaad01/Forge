# M79 — High-Level Training API: `forge.train()`

## 1. Executive summary

M79's brief demanded a real product milestone -- not another readiness
assessment, repository survey, duplication cleanup, or documentation-only
milestone -- answering: *what currently forces an ordinary Forge developer
to understand Forge's internal training machinery when they simply want to
train a model?* The answer, confirmed by a targeted inspection of
`forge/training/trainer.py`, `forge/training/session.py`,
`forge/data/dataloader.py`, and every `examples/*/train.py`: **every single
script still hand-assembles `DataLoader(...)` + `Trainer(...)` +
`trainer.fit(...)` itself** before it can train anything, even though
`start_training_session()` (M73) already removed the fresh-vs-resumed
`Trainer`-construction branch and `save_and_verify()` (M78) already removed
the post-training save-and-verify round trip.

M79 built `forge.train()` (`forge/training/api.py`) -- a single-call,
thin orchestration layer over the existing `DataLoader`/`Trainer`, not a
new training engine -- and retrofitted `examples/mnist/train.py` (Forge's
flagship example) to use it for its common, non-`--resume` path.

## 2. Final public API

```python
forge.train(
    model: Module,
    dataset: Dataset | DataLoader,
    *,
    loss: Loss,
    optimizer: Optimizer,
    epochs: int,
    batch_size: int = 32,
    shuffle: bool = True,
    validation_dataset: Dataset | DataLoader | None = None,
    device: str | Device | None = None,
    metrics: Iterable[Metric] | None = None,
    verbose: bool = True,
) -> TrainingHistory
```

```python
model = MyModel()
history = forge.train(
    model, train_dataset,
    loss=CrossEntropyLoss(),
    optimizer=Adam(model.parameters(), lr=1e-3),
    epochs=10,
    validation_dataset=val_dataset,
    device="cuda",
    metrics=[Accuracy()],
)
forge.save_and_verify(model, "model.forge", sample=...)
```

Re-exported as `forge.train` and `forge.training.train`, matching
`predict()`/`save_and_verify()`/`generate_sequence()`'s existing top-level
re-export precedent.

## 3. Why this API was chosen

**Dataset input.** `dataset`/`validation_dataset` accept either a plain
`forge.data.Dataset` (wrapped in a fresh `DataLoader(batch_size=batch_size,
shuffle=...)`) or an already-built `DataLoader`, used exactly as given.
`batch_size`/`shuffle` are the only two `DataLoader` settings exposed --
anything else (`drop_last`, a custom shuffle `generator`, CUDA prefetch)
means passing a pre-built `DataLoader` in `dataset`'s place instead of
re-exposing `DataLoader`'s full constructor. This was not a hypothetical
design choice: the real consumer (`examples/mnist/train.py`) needed exactly
this escape hatch, since it deliberately keeps model-initialization
(`forge.random`) and batch-order (`data_rng`) on two independent generator
streams -- see §5.

**Loss/optimizer are required, not inferred.** Both are keyword-only with no
default, matching the brief's explicit "require explicit loss/optimizer, no
automatic inference" instruction. A missing one is a plain `TypeError` at
the call site.

**Device is one keyword.** `device="cuda"` calls `model.to(device)` **in
place** before training. This is the one deliberate difference from
`Trainer`'s own "validate, never move" policy (`Trainer` itself is
unmodified). It is safe because `Module.to()` moves every `Parameter` in
place, preserving Python identity (`forge/nn/module.py`), so an `optimizer`
already constructed from `model.parameters()` before the `train()` call
remains valid afterward -- verified directly by
`test_train_keeps_optimizer_parameter_identity_valid_across_a_device_move`
(CPU) and the CUDA equivalent. This single keyword replaces the
`model = build_model().to(args.device)` line every example already writes
by hand.

**No checkpoint/resume.** Investigated per the brief's own instruction to
determine this, not assume it. `TrainingSession`'s resume path fundamentally
*replaces* the model/optimizer a caller passed in with whatever the
checkpoint saved -- exactly the "second checkpoint abstraction" the brief
warns against. Grafting that onto `train()`'s "you already built this model,
train it" shape would mean either silently discarding the caller's
`model`/`optimizer` on a resumed call, or reintroducing
`start_training_session()`'s `build_model`/`build_optimizer` factories
(making `train()` no simpler than what it replaces). `train()` has no
`checkpoint=`/`resume=` parameter; a resumable run still uses
`start_training_session()` + `Trainer` directly (demonstrated side by side
in the real consumer, §5).

**Note on M78's own prior finding.** M78's report (§18, "Rejected
alternatives") already investigated and rejected "`forge.train()` (a
monolithic, dataset-to-artifact function)" -- collapsing dataset
construction, preprocessing, *and* model definition into one function's
parameter list. **M79's `train()` is deliberately not that function.** It
takes an already-constructed `model` and an already-constructed `optimizer`
as arguments; it does not construct a dataset, does not define
preprocessing, and does not define a model architecture. Every example's
`build_model()`/`build_transform()`/`build_datasets()` triple remains
exactly as bespoke as M78 found it necessary to keep. M79's `train()` only
replaces the narrower, still-real duplication M78 didn't address: the
`DataLoader`+`Trainer`+`fit()` glue between an already-built model/dataset
and a trained result.

## 4. How it integrates with `Trainer`/`TrainingSession`

`train()` constructs exactly one `Trainer` internally and calls
`Trainer.fit()` once -- `forge/training/trainer.py` is completely
unmodified, and every validation `Trainer.__init__`/`fit()` already performs
(model/loss/optimizer type checks, device probing, `epochs` validation,
empty-loader checks) fires exactly as it would for a hand-written
`Trainer(...)` call, with the same exception types (`TrainerError`,
`UnsupportedDeviceError`, `CUDAError`). `train()` adds no validation logic
of its own beyond resolving `dataset`/`validation_dataset` into
`DataLoader`s and resolving/moving the device.

`forge/training/session.py` is also completely unmodified. `train()` and
`TrainingSession` are siblings, not a hierarchy: `train()` is for a caller
that already has a `model`/`optimizer` and wants the common, non-resuming
case in one call; `TrainingSession` is for a caller that wants
fresh-or-resumed `Trainer` construction from factories. Neither is built on
top of the other.

## 5. Real consumer: `examples/mnist/train.py`

**Why `mnist`, not `image_folder_classification`** (the brief's suggested
candidate) -- verified against the actual repository rather than assumed:

A grep of every `examples/*/train.py`'s `--resume` handling found all seven
`Trainer`-based examples still support `--resume`, and two of them
(`regression`, `image_folder_classification`) already get exact
`DataLoader`-shuffle resume-equivalence via `start_training_session()`'s
`data_loader_rng_state` checkpoint field (M65/M73). Retrofitting
`image_folder_classification`'s fresh path to `forge.train()` (which has no
such field) would have **silently downgraded that specific, already-fixed
reproducibility guarantee**: the first `--resume` after a `forge.train()`
-trained, `forge.save_checkpoint()`-written checkpoint would lose exact
shuffle continuity. `mnist`, by contrast, was one of the five examples M73
explicitly left un-retrofitted to `TrainingSession` (confirmed by grep: its
`train.py` has never called `start_training_session()`) -- its `--resume`
path never had that guarantee to begin with. Retrofitting `mnist`'s fresh
path is therefore a genuine simplification with **zero reproducibility
regression**, while still being Forge's flagship, most complete example.

**The retrofit:**

```python
# train_loader/test_loader are still hand-built (data_rng stays independent
# of forge.random's own stream, per this script's own documented policy):
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_rng)
test_loader = DataLoader(test_ds, batch_size=args.batch_size)

if args.resume:
    # unchanged: plain Trainer + trainer.resume(checkpoint)
    ...
    history = trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
    trainer.save_checkpoint(str(checkpoint_path))
else:
    model = build_model().to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    history = forge.train(
        model, train_loader, loss=loss_fn, optimizer=optimizer,
        epochs=args.epochs, validation_dataset=test_loader,
        device=args.device, metrics=[Accuracy()],
    )
    epoch, global_step = len(history), len(history) * len(train_loader)
    forge.save_checkpoint(str(checkpoint_path), model, optimizer, epoch=epoch, global_step=global_step)
```

Both branches feed into the unchanged `save_and_verify()` ->
`interpret_classification()`/`predict()` -> fresh-process `infer.py`
pipeline below them.

The fresh path's checkpoint is written via the free `forge.save_checkpoint()`
function (not `Trainer.save_checkpoint()`, since `train()` returns no
`Trainer`) with `epoch`/`global_step` derived from the returned
`TrainingHistory` -- exactly what a fresh `Trainer.fit(epochs=N)` would have
counted (`epoch=N`, `global_step=N * len(train_loader)`), verified by
resuming the produced checkpoint end to end (§7). The final-evaluation print
now reads `history[-1].val_loss`/`val_metrics` (already computed once per
epoch via `validation_dataset=test_loader`) instead of a second, redundant
`trainer.evaluate(test_loader)` call recomputing the identical number.

## 6. End-to-end workflow demonstrated

```text
real MNIST (IDX files) -> MNISTDataset -> DataLoader
    -> forge.train(model, train_loader, loss=..., optimizer=..., epochs=..., validation_dataset=test_loader, device=...)
    -> TrainingHistory (per-epoch train/val loss + accuracy)
    -> forge.save_checkpoint(epoch, global_step derived from history)
    -> save_and_verify(model, path, sample, preprocessing=..., classes=...)
         -> save_model() -> load_model() -> predict() x2 -> PersistenceError on mismatch
    -> interpret_classification(predict(reloaded, sample), classes)
    -> human-readable prediction + confidence
    -> a brand-new image, decoded + preprocessed via load_preprocessing()
    -> examples/mnist/infer.py, run as a genuinely separate process
```

Run for real (not just unit-tested), both devices, against real downloaded
MNIST data (`examples/mnist/data`, already present):

- `python -m examples.mnist.train --epochs 2 --device cpu` -- 2 epochs, real
  60,000/10,000-sample MNIST, loss 0.3456 -> 0.0948, val accuracy 98.08%,
  checkpoint + model saved, in-process inference correct (digit 7, 100%
  confidence), new-image inference correct (digit 2, 100% confidence).
- `python -m examples.mnist.train --resume <checkpoint from above> --epochs 1`
  -- resumed at `epoch=2, global_step=938` (`938 = ceil(60000/128) * 2`,
  confirming the fresh path's derived `global_step` was exactly right),
  continued to epoch 3, val accuracy 98.28%.
- `python -m examples.mnist.infer --model <model> --image <new image>` -- run
  as a genuinely separate process, correctly predicted digit 2 (99.9%
  confidence), matching the in-process result.
- `python -m examples.mnist.train --epochs 1 --device cuda` -- real CUDA
  training on the 940MX: loss 0.3453, val accuracy 97.02% (1 epoch),
  1701 train samples/sec vs. 792 samples/sec on CPU (~2.1x), save_and_verify
  + inference all correct.

## 7. Files changed

**Framework:**
- `forge/training/api.py` -- new file, `train()`.
- `forge/training/__init__.py`, `forge/__init__.py` -- re-exports +
  docstring updates.

**Real consumer:**
- `examples/mnist/train.py` -- fresh-run path now calls `forge.train()`;
  `--resume` path unchanged (plain `Trainer` + `trainer.resume()`).

**Tests:**
- `tests/test_training_api.py` -- 21 new CPU tests.
- `tests/test_training_api_cuda.py` -- 3 new CUDA tests (hardware-verified
  on the 940MX).

**Documentation:**
- `docs/architecture/training-engine.md` -- new **Single-call high-level
  training: `train()`** section (API contract, device-move safety argument,
  the checkpoint/resume boundary, the real-consumer decision), package
  -layout and header updates.
- `README.md` -- **First model** section now uses `forge.train()`;
  `forge.training` bullet and **Training, checkpointing, and persistence**
  section updated.
- `examples/README.md` -- **The shared training workflow** section updated
  to describe all three shared abstractions (`start_training_session()`,
  `train()`, `save_and_verify()`) and which examples use which; `mnist` row
  updated.
- `examples/mnist/README.md` -- pipeline diagram, CUDA-training paragraph,
  and checkpoint/resume section updated to describe the `forge.train()` /
  plain-`Trainer` split.
- `docs/development/progress.md` -- M79 entry appended.
- `docs/development/m79-high-level-training-api.md` -- this report (new
  file).

No `forge/tensor`, `forge/autograd`, `forge/backend`, `forge/nn`,
`forge/optim`, `forge/data`, `forge/serialization`,
`forge/training/trainer.py`, or `forge/training/session.py` file was
touched.

## 8. Tests added and full-suite result

**`tests/test_training_api.py`** (21 CPU tests): `TrainingHistory` return
type and length; parameter updates actually happen; loss decreases over
epochs; bit-exact equivalence against a hand-built `DataLoader`+`Trainer`
given identical seeded initial parameters and unshuffled data (proving
`train()` is pure orchestration, not a second training implementation);
`Dataset` input; already-built-`DataLoader` input (including `drop_last`);
`batch_size` respected; rejection of a non-`Dataset`/non-`DataLoader`;
validation reported when `validation_dataset` given (`Dataset` and
`DataLoader` forms), absent (`None`/`{}`) otherwise; `metrics` passed
through; device defaults to the model's own device; explicit `device=`
moves the model in place; optimizer-parameter-identity safety across a
device move (the specific claim in §3); `TypeError` for missing
`loss`/`optimizer`/`epochs`; `TrainerError` for a non-`Module`/non-`Loss`/
non-`Optimizer`/invalid `epochs` (delegated to `Trainer`'s own validation);
top-level re-export identity.

**`tests/test_training_api_cuda.py`** (3 CUDA tests, hardware-gated via
`pytest.mark.skipif(not is_cuda_available(), ...)`): a CPU-constructed model
moves to CUDA and trains; CUDA parameters actually update; CUDA validation
reporting.

**Full suite:** `python -m pytest tests/ -q` -> **2,289 collected, 2,288
passed, 1 failed** (2,265 + 24 new). The one
failure is `tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory` (`CUDA active bytes
grew: 320 -> 288` -- note this is a *shrink*, not a growth, a measurement
-noise artifact of the same pre-existing CUDA-allocator-measurement flake
documented since M63), re-run in isolation immediately after
(`python -m pytest tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory -q`) and passed
cleanly -- confirmed no M79 regression.

`examples/mnist/train.py`'s own pre-existing integration test suites
(`tests/test_mnist_example_integration.py`,
`tests/test_mnist_example_cuda_integration.py`) pass **unmodified** against
the retrofit -- they exercise `build_model()`/`build_transform()` directly
against a hand-built `Trainer`, not `main()`, so they were unaffected by the
`main()`-internal refactor; the real end-to-end `main()` runs in §6 are the
verification that `main()` itself still works correctly.

## 9. CPU/CUDA verification

Both devices hardware-verified in this session, not merely written:

- CPU: full `tests/test_training_api.py` (21/21 passed) plus a real
  `python -m examples.mnist.train --epochs 2 --device cpu` run against real
  MNIST (§6).
- CUDA (940MX): full `tests/test_training_api_cuda.py` (3/3 passed) plus a
  real `python -m examples.mnist.train --epochs 1 --device cuda` run against
  real MNIST (§6), confirming `forge.train(..., device="cuda")` actually
  moves a CPU-constructed model and trains it through real CUDA kernels
  (unmodified `Trainer`/`CUDABackend` dispatch -- `train()` introduces no
  new CUDA code path of its own).

## 10. What developers can now do that they could not reasonably do before

Before M79, training any Forge model -- even the common, non-resuming case
-- required knowing and writing:
```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
model.to(device)
trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, metrics=metrics)
history = trainer.fit(loader, epochs=epochs, validation_loader=val_loader)
```
After M79:
```python
history = forge.train(model, dataset, loss=loss_fn, optimizer=optimizer,
                       epochs=epochs, validation_dataset=val_dataset,
                       device=device, metrics=metrics)
```
A developer can now train a Forge model through one documented,
type-annotated, top-level function call without ever constructing a
`DataLoader` or a `Trainer` by hand -- while `Trainer`/`TrainingSession`
remain fully available, unmodified, and are what `train()` itself is built
on, for anything requiring checkpoint/resume, CUDA prefetch, or custom
`DataLoader` settings.

## 11. Explicitly out of scope

Per the brief's own non-goals, none of the following were built:
automatic architecture/loss/optimizer selection, hyperparameter tuning,
callbacks, schedulers, distributed/multi-GPU training, mixed precision,
experiment tracking, configuration files/YAML, model zoos, cloud training,
serving infrastructure, a generalized pipeline DSL, or a second training
engine. `train()` also deliberately does not expose `drop_last` or a custom
`DataLoader` `generator` (pass a pre-built `DataLoader` instead) and does
not expose checkpoint/resume (§3).

## 12. Suggested commit message

```
feat: add forge.train(), a single-call high-level training entry point

Adds forge.training.train() (forge/training/api.py), a thin orchestration
layer over the existing DataLoader/Trainer: accepts a Dataset or DataLoader
directly, moves the model to device= in place, and drives Trainer.fit()
underneath, returning the same TrainingHistory. Trainer/TrainingSession are
unmodified and remain the lower-level API for checkpoint/resume, CUDA
prefetch, or custom DataLoader settings train() does not expose.

Retrofits examples/mnist/train.py's fresh (non-resume) path to forge.train()
-- chosen over image_folder_classification (the brief's suggested candidate)
after confirming, by direct inspection, that image_folder_classification's
--resume path already depends on start_training_session()'s shuffle-
continuity guarantee, which forge.train() cannot provide without duplicating
TrainingSession; mnist never had that guarantee, so the retrofit is a real
simplification with zero reproducibility regression.

24 new tests (21 CPU + 3 CUDA, hardware-verified on the 940MX). Full suite:
2,289 collected, 2,288 passed, 1 pre-existing CUDA-allocator-measurement
flake (documented since M63), reproduced passing in isolation.
```

## 13. Next product-level gap (not automatically M80)

`train()` covers the common, single-`fit()`-call case; every remaining
`Trainer`-based example (`regression`, `resnet`, `segmentation`,
`autoencoder`, `waveform_classification`, `image_folder_classification`)
still hand-assembles its fresh-training path, though (per §5's finding) most
of them depend on `start_training_session()`'s resume-equivalence guarantee
and are therefore not safe to retrofit to `train()` without the same
analysis this milestone applied to `mnist`. A concrete, evidence-backed
follow-up would be: for each of those five, directly check (not assume)
whether its fresh path actually needs `data_loader_rng_state` continuity
(i.e., whether its own tests or documented usage ever chain a `--generate`/
fresh run into a later `--resume`) -- the ones that don't are safe `train()`
retrofits by the same reasoning as `mnist`; the ones that do are not, absent
a future extension to `train()` itself. Not started here, and not assumed
to be M80's automatic next step.
