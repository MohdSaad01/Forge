# M81 — High-Level Training-to-Artifact Workflow: `forge.train_and_save()`

## 1. Executive summary

M81's brief demanded a real product milestone -- not another readiness
assessment, capability survey, or documentation-only pass -- answering one
narrow question: *what manual framework-level work does a developer still
have to perform after `forge.train()` finishes in order to produce a
verified, portable model artifact?*

A targeted inspection of `forge.train()`, `TrainingHistory`,
`Trainer.evaluate()`, `save_model()`/`save_and_verify()`, `load_model()`/
`load_preprocessing()`/`load_classes()`, `predict()`/
`interpret_classification()`, and both `examples/mnist/train.py` and
`examples/image_folder_classification/train.py` (post-M80) found that the
"evaluate" and "persist + verify" steps the brief's own workflow diagram
lists were **already individually solved** -- `Trainer.fit(...,
validation_dataset=...)` already reports per-epoch evaluation for free
(Milestone 6/79), and `save_and_verify()` (Milestone 78) already saves,
reloads, and proves a model's portability in one call. But both real
`train.py` scripts still called `forge.train()` and then, several lines
later (after computing `epoch`/`global_step` and saving a checkpoint),
called `save_and_verify()` separately -- the identical two-call sequence, in
the same order, in both scripts, with the same hand-picked, manually-batched
`sample` each time.

M81 built `forge.train_and_save()` (`forge/training/api.py`) -- a thin
orchestration layer that calls `train()` then `save_and_verify()` exactly
once each, returning a `TrainAndSaveResult` (history, final validation
result, reloaded+verified model) -- and retrofitted both real consumers'
fresh (non-`--resume`) paths to use it. Along the way, retrofitting
`image_folder_classification` (whose model uses `Dropout`) surfaced and fixed
a real, previously-invisible bug in `load_model()`: reconstructing a module
tree draws (and discards) random initialization values, silently advancing
`forge.random`'s global generator -- a leak that broke a bit-for-bit
resume-equivalence test the moment `save_and_verify()`'s reload was moved to
run before, instead of after, a checkpoint save.

## 2. What was actually missing (evidence, not assumption)

Per the brief's own explicit distinction from M78 ("if inspection shows that
`save_and_verify()` already provides the correct persistence operation, then
build the missing workflow composition around it, rather than another
persistence function"), this milestone's job was to find what wraps
`train()` + `save_and_verify()`, not to modify either.

M80's own report (`docs/development/m80-train-to-artifact-workflow.md`,
section 11) had already considered and rejected "a generic 'train and save'
pipeline function (`forge.train_and_save()` or similar)" at the time it
retrofitted `image_folder_classification` to `forge.train()`, on the
grounds that "no evidence showed the two-call sequence... is itself a real
duplication problem." Re-examining that decision under M81's brief (which
explicitly invites reconsidering exactly this composition, "only if the
actual consumer demonstrates that such composition is genuinely useful"):
the evidence available to M80 at the time was a single retrofit performed
in the same milestone. By M81, that pattern has now survived unchanged
across two independent scripts (`mnist/train.py` since M79, `image_folder_
classification/train.py` since M80) with byte-for-byte identical shape:

```python
history = forge.train(model, ..., loss=loss_fn, optimizer=optimizer,
                       epochs=..., validation_dataset=..., device=...,
                       metrics=[Accuracy()])
# ... checkpoint bookkeeping (epoch/global_step, forge.save_checkpoint) ...
query_x, query_y = test_ds[0]
query_x = query_x.to(device).reshape(1, *shape)
reloaded = save_and_verify(model, str(model_path), query_x,
                            preprocessing=build_transform(), classes=classes)
```

This is exactly the "two independent, already-existing consumers share the
exact same shape" bar this codebase has used to justify every other
extraction in this arc (`start_training_session()` in M73, `generate_
sequence()` in M75, `save_and_verify()` itself in M78) -- the bar M80 found
not yet met for this specific pair, and which this milestone found is now
met.

**What the "evaluate" step in the brief's workflow diagram already needed
no new code for:** `forge.train(..., validation_dataset=test_loader)`
already evaluates once per epoch via `Trainer.evaluate()` internally
(`Trainer.fit()`, `forge/training/trainer.py`), and the resulting
`TrainingHistory`'s last `EpochResult` already carries the final
`val_loss`/`val_metrics` -- both real `train.py` scripts already print this
directly, with an explicit comment ("`history[-1].val_* already reflects
test_loader evaluated after the last epoch's parameter update`") noting that
a second, standalone `trainer.evaluate(test_loader)` call would be
redundant. M80's own report (section 11) separately considered and rejected
a standalone `forge.evaluate()` free function for the same reason: no real
consumer needs a second, post-hoc evaluation pass. M81 did not re-open that
question -- `TrainAndSaveResult` simply copies the *already-computed* final
epoch's result onto the return value (`val_loss`, `val_metrics`), so a
caller does not need to know `TrainingHistory`'s own indexing convention to
find it, without recomputing anything.

## 3. What was implemented

### 3.1 `forge.training.train_and_save()` and `TrainAndSaveResult`

`forge/training/api.py`:

```python
@dataclass(frozen=True)
class TrainAndSaveResult:
    history: TrainingHistory
    val_loss: "float | None"
    val_metrics: "dict[str, float]"
    model: Module


def train_and_save(
    model, dataset, *,
    loss, optimizer, epochs,
    path, sample,
    batch_size=32, shuffle=True,
    validation_dataset=None,
    device=None, metrics=None, verbose=True,
    preprocessing=None, classes=None, atol=1e-5,
) -> TrainAndSaveResult:
    history = train(model, dataset, loss=loss, optimizer=optimizer, epochs=epochs,
                     batch_size=batch_size, shuffle=shuffle,
                     validation_dataset=validation_dataset, device=device,
                     metrics=metrics, verbose=verbose)
    reloaded = save_and_verify(model, path, sample, preprocessing=preprocessing,
                                classes=classes, atol=atol)
    last = history[-1]
    return TrainAndSaveResult(history=history, val_loss=last.val_loss,
                               val_metrics=last.val_metrics, model=reloaded)
```

Every parameter is either forwarded straight to `train()` (`dataset`,
`loss`, `optimizer`, `epochs`, `batch_size`, `shuffle`, `validation_dataset`,
`device`, `metrics`, `verbose`) or straight to `save_and_verify()` (`path`,
`sample`, `preprocessing`, `classes`, `atol`) -- no new validation logic,
no new training logic, no new persistence logic. `path`/`sample` are
required and keyword-only (a missing one is a plain `TypeError`, matching
`train()`'s own `loss`/`optimizer`/`epochs`).

### 3.2 The root-cause fix: `load_model()` no longer perturbs `forge.random`

`forge/serialization/model.py`'s `load_model()` reconstructs a module tree
by calling each registered type's ordinary constructor (`spec.from_config()`,
which for `Conv2d`/`Linear`/`Embedding`/etc. defaults to `cls(**config)`).
Every one of those constructors draws an initial-weights sample from
`forge.random.default_generator()` -- a draw `load_model()` then discards a
moment later by overwriting the parameter with the file's own saved values.
That draw still *advances* the global generator's stream position, though,
and nothing previously undid this:

```python
>>> forge.random.seed(0)
>>> model = Sequential(Conv2d(1, 2, kernel_size=3), Flatten(), Linear(8, 3))
>>> before = forge.random.default_generator().bit_generator.state
>>> save_and_verify(model, "m.forge", x)   # calls load_model() internally
>>> after = forge.random.default_generator().bit_generator.state
>>> before == after
False   # <- the bug: loading a model has an observable side effect
```

This surfaced as a real test failure while retrofitting `image_folder_
classification` (whose model uses `Dropout`, which also draws from
`forge.random.default_generator()` -- see 4.2 below), not as a hypothetical
concern. The fix (`forge/serialization/model.py::load_model()`):

```python
random_state = forge_random.get_state()
try:
    return _build_load_node(root, prefix="", arrays=arrays, path=path, target_device=target_device)
except PersistenceError:
    raise
except (TypeError, ValueError, KeyError, AttributeError) as exc:
    raise PersistenceError(...) from exc
finally:
    forge_random.set_state(random_state)
```

Snapshotting/restoring `forge.random`'s state around reconstruction is the
same `get_state()`/`set_state()` mechanism `forge.serialization.checkpoint`
already uses for exact training-resume determinism (`forge/random.py`) --
no new RNG machinery, just applying the existing one at a second call site
that needed it. `load_model()` is now a pure read with respect to
`forge.random`: it can be called anywhere, any number of times, with zero
observable effect on any later random draw, regardless of what constructors
its own reconstruction happens to invoke.

### 3.3 Real consumers retrofitted

`examples/mnist/train.py` and `examples/image_folder_classification/
train.py`'s fresh (non-`--resume`) branches now call `train_and_save()` in
place of their separate `train()` + (later) `save_and_verify()` calls;
`--resume` in both scripts (which has no `train()` call for
`train_and_save()` to wrap) keeps calling `save_and_verify()` directly,
unchanged. The `sample`/`query_x` a caller must supply is now computed once,
up front (before the `--resume`-or-fresh branch), since both branches need
it.

## 4. Why this implementation was chosen

### 4.1 Composition only, no new validation/training/persistence logic

`train_and_save()` calls `train()` and `save_and_verify()` exactly once
each, with every parameter forwarded unchanged. Every failure mode either
function already documents (`TrainerError` for a non-`Module`/non-`Loss`/
non-`Optimizer`/invalid `epochs`, `DataError` for a non-`Dataset`/non-
`DataLoader` or non-`Tensor` `sample`, `PersistenceError` for a save/reload
prediction mismatch) applies unchanged -- `tests/test_training_api.py`'s new
tests verify this by construction (e.g.
`test_train_and_save_trains_exactly_like_train` proves bit-for-bit training
equivalence against a direct `train()` call given identical seeded state).

### 4.2 The real hazard this retrofit found, and why the fix belongs in `load_model()`, not `train_and_save()`

Retrofitting `image_folder_classification`'s fresh path (whose CNN has
`Dropout(0.3)` -- see `examples/image_folder_classification/model.py`)
broke `tests/test_image_folder_classification_integration.py::
test_resume_after_a_forge_train_fresh_run_matches_continuous_training`: a
continuous 4-epoch run and a 2-epoch-fresh + 2-epoch-`--resume` run, which
must produce bit-for-bit identical parameters (Milestone 65/73's shuffle-
resume-equivalence guarantee), diverged by ~1e-3 in nearly every weight
element -- the signature of a different random stream, not a training-loop
bug.

Root cause: both scripts had always called `forge.save_checkpoint()`
*before* `save_and_verify()` (checkpoint bookkeeping happens right after
`forge.train()` returns; the save/verify/inference demo happens several
print statements later). `train_and_save()` collapses `train()` immediately
into `save_and_verify()`, so the checkpoint save in the retrofitted fresh
branch now happens *after* `save_and_verify()`'s internal `load_model()`
call instead of before it. Because `load_model()` was silently advancing
`forge.random`'s global state (4.2 above), the checkpoint now recorded a
different `forge.random` state than before -- and since `Dropout` draws
from that same generator during the subsequent `--resume` run, the resumed
training diverged.

Two fixes were possible:
- Avoid ever reordering checkpoint-save relative to `save_and_verify()` --
  i.e., don't build `train_and_save()`, or don't retrofit
  `image_folder_classification` to it. Rejected: this treats a real bug in
  `load_model()` as a permanent constraint on API composition order, which
  would make `train_and_save()` (and by extension any future caller who
  reloads a model mid-training-run for any reason) fragile in a way with no
  visible warning.
- Fix `load_model()` itself so reconstruction has no observable side effect
  on unrelated future random draws. **Chosen.** This is the actual root
  cause (a model file's reconstruction should not be able to affect a
  completely unrelated part of a training run just because it happens to
  run before a later random draw), it is a two-line, fully isolated fix
  using a mechanism (`forge.random.get_state()`/`set_state()`) this codebase
  already has and already uses for exactly this kind of state boundary, and
  it makes `load_model()` -- and every composition built on it, including
  `save_and_verify()`/`train_and_save()` -- safe to call at any point in a
  training script without this class of hazard, not just in the one
  ordering the two examples happened to use before.

This is a `forge/serialization/model.py` change, one layer away from
`forge/training/api.py` where M81's headline capability lives -- but it is
the direct, load-bearing consequence of implementing this milestone's real
consumer honestly (retrofitting and testing both examples, not just
building the function in isolation), and CLAUDE.md's own instruction to
"identify root causes and fix underlying issues" applies squarely here.

### 4.3 Why not extend `save_and_verify()` or `train()` instead

Considered and rejected, per the brief's own explicit warning not to
"rename, duplicate, or cosmetically wrap" `save_and_verify()`, and M79's
already-established reasoning against adding parameters to `train()`
(checkpoint/artifact concerns would make it "no simpler than the workflow
it exists to replace"). `train_and_save()` is a new, separate, small
function that composes both, exactly matching the shape `start_training_
session()`/`generate_sequence()`/`save_and_verify()` themselves already
established for "two real consumers share an identical multi-call sequence."

## 5. Real consumers

`examples/mnist/train.py` and `examples/image_folder_classification/
train.py` -- both fresh (non-`--resume`) paths, chosen because both are the
exact scripts whose duplicated `train()` -> `save_and_verify()` sequence is
the evidence this milestone's capability is built from. `image_folder_
classification` remains the brief's preferred richer consumer (real image
files, persisted preprocessing, persisted class vocabulary); `mnist` is
included because it demonstrates the same real duplication independently,
which is what justifies the extraction in the first place (this codebase's
own established bar, see section 2).

## 6. End-to-end workflow demonstrated

```text
ImageFolder/MNISTDataset -> DataLoader
    -> forge.train_and_save(model, train_loader, loss=CrossEntropyLoss(),
                             optimizer=Adam(...), epochs=..., validation_dataset=test_loader,
                             device=..., metrics=[Accuracy()],
                             path="model.forge", sample=query_x_batch,
                             preprocessing=build_transform(), classes=full_dataset.classes)
        -> train() -> TrainingHistory (per-epoch train/val loss + accuracy)
        -> save_and_verify() -> save_model() -> load_model() -> predict() x2
                                -> PersistenceError on mismatch
    -> TrainAndSaveResult(history, val_loss, val_metrics, model=reloaded)
    -> forge.save_checkpoint(..., extra={"data_loader_rng_state": ...})   # orthogonal, unchanged
    -> interpret_classification(predict(reloaded, sample), classes) -> human-readable prediction
    -> [--resume] start_training_session(resume=checkpoint_path) + save_and_verify() directly
    -> python -m examples.<name>.infer, launched as a real subprocess (Milestone 80, unchanged)
```

Run for real, both devices:

- CPU: `python -m examples.image_folder_classification.train --generate
  --epochs 8 --device cpu` -- fresh path via `train_and_save()`, checkpoint +
  model saved, save+verify passed, in-process and new-image inference demos
  correct.
- CPU: `python -m examples.mnist.train --download --epochs 3 --device cpu`
  -- same, against real MNIST.
- CUDA (940MX): `tests/test_training_api_cuda.py::
  test_train_and_save_on_cuda_produces_a_verified_reloadable_artifact`,
  `tests/test_image_folder_classification_cuda_integration.py::
  test_main_fresh_path_uses_forge_train_on_cuda`, and
  `tests/test_mnist_example_cuda_integration.py` all pass with the retrofitted
  fresh path, hardware-verified (13/13 CUDA tests in the touched files).
- `--resume` continues to work identically for both examples (unchanged
  code path), and `image_folder_classification`'s bit-for-bit resume-
  equivalence test passes again once the `load_model()` fix landed.

## 7. Files changed

**Framework:**
- `forge/training/api.py` -- `train_and_save()`, `TrainAndSaveResult` (new).
- `forge/training/__init__.py`, `forge/__init__.py` -- re-exports +
  docstring updates.
- `forge/serialization/model.py` -- `load_model()` now snapshots/restores
  `forge.random`'s state around module-tree reconstruction (root-cause fix,
  section 3.2/4.2).

**Real consumers:**
- `examples/mnist/train.py` -- fresh path now calls `forge.train_and_save()`
  in place of `forge.train()` + a later `save_and_verify()` call; `--resume`
  now calls `save_and_verify()` directly (unchanged behavior, moved earlier
  in the branch). Module docstring updated.
- `examples/image_folder_classification/train.py` -- same retrofit; the
  fresh branch's `data_loader_rng`/checkpoint logic is otherwise unchanged.
  Module docstring updated.

**Tests:**
- `tests/test_training_api.py` -- 11 new CPU tests for `train_and_save()`.
- `tests/test_training_api_cuda.py` -- 1 new CUDA test (hardware-verified).
- `tests/test_serialization.py` -- 2 new tests for the `load_model()`
  RNG-isolation fix.
- `tests/test_image_folder_classification_integration.py`,
  `tests/test_mnist_example_integration.py` -- existing integration/`main()`
  -level/subprocess tests re-verified against the retrofit (no test changes
  needed; they assert on file contents and subprocess behavior, not internal
  call structure).

**Documentation:**
- `docs/architecture/training-engine.md` -- new **Train, evaluate, persist,
  verify in one call: `train_and_save()`** section; package-layout and
  header updates.
- `docs/architecture/persistence.md` -- **Architecture reconstruction**
  section gained an **RNG isolation** subsection documenting the
  `load_model()` fix.
- `README.md`, `examples/README.md`, `examples/mnist/README.md`,
  `examples/image_folder_classification/README.md` -- updated to describe
  `train_and_save()` as the fresh-path entry point, alongside `train()`/
  `save_and_verify()`/`Trainer`/`TrainingSession` for lower-level control.
- `docs/development/progress.md` -- M81 entry appended.
- `docs/development/m81-train-to-verified-artifact-workflow.md` -- this
  report (new file).

No `forge/tensor`, `forge/autograd`, `forge/backend`, `forge/nn`,
`forge/optim`, `forge/data`, `forge/training/trainer.py`, or
`forge/training/session.py` file was touched. `Trainer`, `TrainingSession`,
`train()`, and `save_and_verify()`'s own signatures are byte-for-byte
unchanged.

## 8. Tests added and full-suite result

- `tests/test_training_api.py` (+11, CPU): return type/history length;
  bit-for-bit training equivalence against a direct `train()` call; a
  working, reloadable artifact is written; `preprocessing=`/`classes=`
  pass through; the final epoch's validation result is exposed on
  `TrainAndSaveResult` without recomputation; `None`/`{}` when no
  `validation_dataset`; `PersistenceError` propagated on a simulated
  reload mismatch; `TrainerError`/`DataError` for a non-`Module`/non-`Tensor`
  `sample`; `TypeError` for missing `path`/`sample`; top-level re-export
  identity.
- `tests/test_training_api_cuda.py` (+1, CUDA, hardware-gated): a
  CPU-built model trains via `train_and_save(..., device="cuda")` and
  produces a CUDA-resident, reloadable artifact.
- `tests/test_serialization.py` (+2, CPU): `load_model()` does not advance
  `forge.random`'s state; the same guarantee holds when reconstruction
  raises partway through (the `finally` path).

**Full suite:** `python -m pytest tests/ -q` -> **2,308 collected, 2,307
passed, 1 failed** (2,294 + 14 new). The one failure is `tests/
test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`
(`CUDA active bytes grew: 320 -> 288` -- a *shrink*, the same
measurement-noise artifact documented since M63), re-run in isolation
immediately after and passed cleanly -- confirmed no M81 regression.

## 9. CPU/CUDA verification

Both devices hardware-verified in this session:

- CPU: `tests/test_training_api.py` (32/32, including 13 new), `tests/
  test_serialization.py` (60/60, including 2 new), `tests/
  test_image_folder_classification_integration.py` (22/22, including the
  resume-equivalence test that surfaced and then confirmed the
  `load_model()` fix), `tests/test_mnist_example_integration.py` (13/13).
- CUDA (940MX): `tests/test_training_api_cuda.py` (4/4, including the new
  `train_and_save()` test), `tests/
  test_image_folder_classification_cuda_integration.py` (5/5), `tests/
  test_mnist_example_cuda_integration.py` -- all passed against real CUDA
  kernels, confirming the retrofitted fresh path still moves a CPU-built
  model to CUDA and produces a CUDA-resident, reloadable artifact through
  unmodified `Trainer`/`CUDABackend` dispatch.
- Full suite (`python -m pytest tests/ -q`): 2,308 collected, 2,307 passed,
  1 pre-existing flake (section 8), reproduced passing in isolation.

## 10. Limitations

- `train_and_save()` has no checkpoint/resume concept, matching `train()`'s
  own deliberate scope boundary -- a resumable run still uses
  `start_training_session()` + `Trainer` + a direct `save_and_verify()`
  call, exactly as both retrofitted examples' `--resume` branches do.
- `regression`, `resnet`, `segmentation`, `autoencoder`, and
  `waveform_classification` still hand-assemble their fresh training +
  save/verify path (unchanged by this milestone) -- each would need the
  same "does its `--resume` path depend on `data_loader_rng_state`
  continuity" check M79/M80 already flagged before a safe `train()`/
  `train_and_save()` retrofit, which this milestone did not re-open.
- The `load_model()` RNG-isolation fix (section 3.2) closes the specific
  hazard this milestone's retrofit surfaced (a wasted-but-observable draw
  from `forge.random.default_generator()` during reconstruction); it does
  not add any new RNG-state tracking beyond the existing `get_state()`/
  `set_state()` mechanism, and does not change what `load_model()` restores
  onto the reconstructed model itself (parameters/buffers were always
  correct; only the *side effect* on unrelated future draws was fixed).

## 11. Rejected alternatives

- **A monolithic `forge.train_and_save()` that also builds the dataset,
  preprocessing, or model** -- rejected per the brief's own explicit
  constraint and M78's own prior rejection of exactly this shape; every
  input (`model`, `dataset`, `path`, `sample`, `preprocessing`, `classes`)
  remains an explicit, already-constructed argument.
- **Adding `path=`/`sample=`/`preprocessing=`/`classes=` parameters directly
  to `forge.train()`** -- rejected per the brief's own "strong architectural
  constraint" against inflating `train()`'s surface with artifact/
  persistence concerns; `train_and_save()` is a separate function that
  composes `train()`, not a modification of it.
- **A standalone `forge.evaluate()` free function** -- not re-considered;
  M80's report already rejected this for lack of a real consumer needing a
  second, post-hoc evaluation pass, and nothing in this milestone's
  inspection changed that finding. `TrainAndSaveResult.val_loss`/
  `val_metrics` expose the *already-computed* final-epoch result instead of
  recomputing it.
- **Leaving `image_folder_classification`'s fresh path on separate `train()`
  + `save_and_verify()` calls to avoid the checkpoint-reordering hazard**
  -- rejected once the actual root cause (`load_model()`'s RNG leak) was
  found and fixed; working around a bug by constraining API composition
  order permanently would have left the same hazard latent for any future
  caller.

## 12. What can an external Forge developer do now that they could not realistically do before M81?

Before M81, training a model through `forge.train()` and turning the result
into a verified, portable `.forge` artifact required two separate calls in
a specific order, with the developer responsible for remembering to make
the second one (`save_and_verify()`) and for picking/batching a sample
input by hand each time -- both real examples had already independently
written this exact sequence, a duplication no different in kind from the
ones `start_training_session()`/`generate_sequence()`/`save_and_verify()`
were each built to remove.

**After M81:** a developer can call one function --
`forge.train_and_save(model, dataset, loss=..., optimizer=..., epochs=...,
path=..., sample=..., preprocessing=..., classes=...)` -- and get back a
`TrainAndSaveResult` holding the completed training history, the final
epoch's validation result, and a freshly reloaded, verified model, with
`forge.PersistenceError` raised automatically if the saved artifact and the
in-memory model ever disagree. `Trainer`, `train()`, and `save_and_verify()`
remain fully available, unmodified, for checkpoint/resume, CUDA prefetch, or
any finer-grained control. Separately, and load-bearing for any script that
reloads a model mid-run for any reason, `load_model()` no longer has a
hidden side effect on `forge.random`'s global state -- a real, previously
undetected correctness gap this milestone's real-consumer retrofit found and
closed at its root.
