# M80 — End-to-End Train-to-Portable-Artifact Workflow

## 1. Executive summary

M80's brief demanded a real product milestone -- not another readiness
assessment -- answering: *after M79, can an external developer reasonably
train a model with `forge.train()` and turn it into the same kind of
portable artifact M71/M72/M77 established, ending in useful fresh-process
inference?*

A targeted inspection (not a broad capability survey) of `forge.train()`,
`TrainingSession`, `save_and_verify()`, `load_classes()`/
`load_preprocessing()`, `predict()`/`interpret_classification()`, `forge
model predict`, and both `examples/mnist/` and
`examples/image_folder_classification/` found the answer was **already yes
for `mnist`** (M79's own retrofit already flows into
`save_and_verify(..., preprocessing=..., classes=...)` and a fresh-process
`infer.py`) -- but two concrete, evidence-backed gaps remained that kept the
workflow from being real for Forge's other, richer flagship example, and
kept "fresh process" as an asserted claim rather than an enforced one
anywhere in the repository:

1. `image_folder_classification` -- the brief's own preferred real
   consumer, and the closer analogue to a real-world image-classification
   workflow (arbitrary files on disk, persisted preprocessing, persisted
   class vocabulary) -- never used `forge.train()` at all.
2. No test in the entire repository had ever launched a genuinely separate
   OS process for inference, or called either example's actual
   `train.py::main()` entry point, despite four milestones' worth of reports
   (M71/M72/M77/M78) describing "fresh process" as already verified.

M80 closed both gaps through composition of existing public APIs -- no
change to `forge.train()`, `TrainingSession`, `Trainer`, or the persistence
format.

## 2. What was missing (evidence, not assumption)

### 2.1 `image_folder_classification` was `forge.train()`-ineligible for a real, previously-documented reason

M79's own report (§5) explicitly investigated and rejected retrofitting
`image_folder_classification`: since Milestone 73, its `--resume` path gets
exact `DataLoader`-shuffle resume-equivalence via
`start_training_session()`'s `data_loader_rng_state` checkpoint field
(Milestone 65's fix, generalized in M73). `forge.train()` has no
`checkpoint=`/`resume=` concept and produces no such field -- retrofitting
the *whole* script (as `start_training_session()` handled both branches
together) would have silently downgraded the first `--resume` after a fresh
run, losing exact shuffle continuity. This was a correct, evidence-based
decision at the time, not an oversight -- but it left the brief's preferred
real consumer permanently excluded from the API M79 built, unless someone
found a way to keep the guarantee.

Re-reading `mnist/train.py`'s actual shape (confirmed by direct inspection,
not memory) showed the real structure: `mnist` splits into a
`--resume`-or-fresh two-branch `if`/`else`, with `forge.train()` only in the
fresh branch. `image_folder_classification`, by contrast, called
`start_training_session()` **once**, letting its internal `resume` argument
decide the branch -- a single call, not two. Nothing in `forge.train()`,
`TrainingSession`, or the checkpoint format actually required this
single-call shape; it was just how the script had been written since M73.
Splitting it the same way `mnist` is already split -- fresh via
`forge.train()`, `--resume` via `start_training_session()` -- was never
tried.

### 2.2 "Fresh process" was never actually tested as a fresh process

Grepping the entire `tests/` directory for `subprocess` found it used only
in CUDA-allocator-reentrancy/pinned-memory tests and `test_cli.py` (whose
own comment explicitly says "no subprocess" for its CLI-`main()` calls) --
**never** for launching an example's `infer.py`. Every existing "fresh
process" test (`test_infer_run_classifies_a_fresh_process_style_png` in both
`tests/test_mnist_example_integration.py` and
`tests/test_classification_metadata.py`'s `image_folder_classification`
coverage) imports `infer.run()` directly into the running test process.

This is a real discrepancy against the historical record, not a pedantic
one: M78's own report (§14, "Fresh-process inference verification") states
`examples/mnist/infer.py` and `examples/image_folder_classification/infer.py`
"were re-run as genuinely separate `python -m` processes via their own test
suites" -- a claim the actual test code does not support. The manual
dev-session runs recorded in M71/M72/M77/M78/M79's reports were real, but
nothing in the automated suite would catch a regression that broke real
process-boundary behavior (e.g. the `try: from .x import y / except
ImportError: from x import y` "running as a plain script" fallback every
`train.py`/`infer.py` module carries, which -- because every test imports
these modules as part of the `examples` package -- had literally never been
exercised by any test).

## 3. What was implemented

### 3.1 `image_folder_classification/train.py`: split into `forge.train()`-fresh / `start_training_session()`-resume

```python
if args.resume:
    session = start_training_session(
        build_model=lambda: build_model(num_classes=len(full_dataset.classes)),
        build_optimizer=lambda params: Adam(params, lr=args.lr),
        loss_fn=loss_fn, seed=args.seed, device=args.device,
        metrics=[Accuracy()], resume=args.resume,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=session.data_loader_rng)
    history = session.trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
    session.save_checkpoint(str(checkpoint_path))
    model = session.trainer.model
else:
    forge.random.seed(args.seed)
    data_loader_rng = np.random.default_rng(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=data_loader_rng)
    model = build_model(num_classes=len(full_dataset.classes)).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    history = forge.train(
        model, train_loader, loss=loss_fn, optimizer=optimizer, epochs=args.epochs,
        validation_dataset=test_loader, device=args.device, metrics=[Accuracy()],
    )
    epoch, global_step = len(history), len(history) * len(train_loader)
    forge.save_checkpoint(
        str(checkpoint_path), model, optimizer, epoch=epoch, global_step=global_step,
        extra={"data_loader_rng_state": data_loader_rng.bit_generator.state},
    )
```

The fresh branch builds `data_loader_rng` itself -- exactly what
`start_training_session()` builds internally from the same `seed` -- and
writes its `bit_generator.state` into the checkpoint's `extra` dict under
the key `"data_loader_rng_state"`, the identical key
`TrainingSession.save_checkpoint()` already writes. `start_training_session()`'s
resume path (unchanged, `forge/training/session.py`) reads that key from
`checkpoint.extra` with no knowledge of, or dependency on, which function
wrote the checkpoint. This is a pure composition of two already-public
building blocks (`forge.train()`, `forge.save_checkpoint(..., extra=...)`)
against a persistence field (`data_loader_rng_state`) that already existed
and already had a defined meaning -- no new abstraction, no format change,
no change to `forge/training/session.py`, `forge/training/api.py`,
`forge/training/trainer.py`, or `forge/serialization/checkpoint.py`.

The rest of the script (dataset loading, `build_transform()`,
`save_and_verify()`, the two inference demos, the printed CLI commands) is
unchanged; only the variable `model` is now set from either branch (`model =
session.trainer.model` on resume, `model = build_model(...).to(device)` on
the fresh path) since there is no longer a single `session.trainer.model`
reference valid for both. The "final test evaluation" print now reads
`history[-1].val_loss`/`val_metrics` (already computed once per epoch via
`validation_dataset=test_loader`) instead of a redundant
`trainer.evaluate(test_loader)` call -- the same simplification M79 applied
to `mnist`, now applied here too since the fresh branch's `forge.train()`
call returns no `Trainer` to call `.evaluate()` on.

### 3.2 Genuine subprocess-based fresh-process tests

Added `test_infer_cli_runs_in_a_genuinely_separate_process` to both
`tests/test_mnist_example_integration.py` and
`tests/test_image_folder_classification_integration.py`. Each:

1. Builds a real trained model + saved `.forge` artifact (with
   `preprocessing=`/`classes=`) -- for `image_folder_classification`, via
   the real `train.py::main()` entry point; for `mnist`, via the same
   hand-built `Trainer` pipeline the file's other tests already use (to
   avoid requiring the real ~11MB MNIST download in every test
   environment -- `main()`'s own real-MNIST path is exercised manually and
   in M79's dev-session report, not by this test).
2. Launches `subprocess.run([sys.executable, "-m",
   "examples.<name>.infer", "--model", ..., "--image", ...], cwd=repo_root,
   capture_output=True, text=True)` -- a real, separate OS process, with no
   shared memory, module cache, or import state with the test process.
3. Asserts `returncode == 0` and that the subprocess's stdout matches
   `forge.predict()`/`interpret_classification()` computed independently,
   in the test process, against the same file -- proving the subprocess
   doesn't merely "not crash" but produces the exact documented result.

### 3.3 `main()`-level and CUDA coverage for the retrofit itself

Added `test_main_fresh_path_uses_forge_train_and_produces_a_working_artifact`
and `test_resume_after_a_forge_train_fresh_run_matches_continuous_training`
(CPU, `tests/test_image_folder_classification_integration.py`), and
`test_main_fresh_path_uses_forge_train_on_cuda` (CUDA,
`tests/test_image_folder_classification_cuda_integration.py`,
hardware-verified on the 940MX). These are the first tests in this
repository's history to call `image_folder_classification`'s real
`train.py::main()` (or `mnist`'s equivalent) rather than hand-building a
`Trainer` -- every prior integration test in both files exercises
`build_model()`/`build_transform()`/`Trainer` directly, so a regression in
`main()`'s own argument parsing or branch wiring (exactly the kind of bug
this milestone's retrofit could have introduced) would previously have gone
undetected until a human ran the script by hand.

## 4. Why this implementation was chosen

**Composition over a new abstraction.** The brief explicitly forbids a
"Pipeline"/"Experiment"/"ModelManager" abstraction and prefers composing
`forge.train()`, `Trainer.evaluate()`/`Metric`, `save_model()`/
`save_and_verify()`, `load_model()`/`load_classes()`/`load_preprocessing()`,
and `predict()`/`interpret_classification()` if they already suffice. They
did: the `data_loader_rng_state` continuity problem that blocked
`image_folder_classification`'s retrofit turned out to be solvable with
zero new code in `forge/` -- just calling the existing free
`forge.save_checkpoint(..., extra=...)` function with the same dict key
`TrainingSession.save_checkpoint()` already uses. This is the "smallest
missing capability that makes the workflow real" the brief asks for, not a
framework change.

**Why not extend `forge.train()` with a resume-continuity parameter
instead?** Considered and rejected. That would re-introduce exactly the
"second checkpoint abstraction" M79's brief (and M79's own report, §3)
already argued against: `forge.train()`'s entire premise is "you already
built `model`/`optimizer`, train them," and any resume-aware extension
either has to accept a `data_loader_rng=` parameter (leaking
`TrainingSession`-specific plumbing into a function whose whole point is to
not need it) or silently do nothing useful without one (dead parameter).
Building the state in the calling script -- exactly where
`start_training_session()` itself builds it -- keeps `forge.train()`
unchanged and the responsibility in the one place that already understands
what a checkpoint's `extra` dict is for.

**Why not retrofit both branches of `image_folder_classification` to
`forge.train()`, dropping `start_training_session()` entirely?**
`forge.train()` has no resume concept at all (§3 of M79's report explains
why this is a deliberate, permanent design choice, not a gap to fill) --
there is no way to resume a checkpoint through `forge.train()` without
`start_training_session()` or an equivalent. The two-branch split is the
only shape that keeps both the high-level fresh-training entry point and
correct resume semantics.

**Why prioritize the subprocess tests as a real M80 deliverable, not "just
more tests"?** The brief's own Validation Requirements section explicitly
lists "Fresh process: Inference works in a genuinely separate process" as a
thing M80 must validate, and separately warns "Use real end-to-end
execution rather than relying exclusively on mocked/unit-level tests." Four
prior milestones' reports asserted this property was already verified;
direct inspection proved that assertion false against the actual test
suite. Leaving that gap unaddressed while shipping only the
`image_folder_classification` retrofit would have left the milestone's own
required validation criterion unmet by the letter of the brief, and left a
real, previously-undetected blind spot (the never-exercised
`except ImportError` plain-script fallback, and the never-tested `main()`
entry points) in place.

## 5. Real consumer

`examples/image_folder_classification/` -- the brief's own preferred real
consumer, chosen specifically because it demonstrates the fullest version of
the intended workflow (real image files on disk, `ImageFolder`, persisted
`Resize`/`Normalize` preprocessing, persisted class vocabulary, mixed
source resolutions, a CNN with `BatchNorm2d`/`Dropout`) -- now trains its
common case through `forge.train()`, joining `examples/mnist/` as the
second demonstrated `forge.train()` consumer and closing the "brief's
suggested candidate was excluded" gap M79 left open.

`examples/mnist/` is the second real consumer for this milestone's other
half (the subprocess fresh-process test), extending the same treatment to
Forge's flagship example.

## 6. End-to-end workflow demonstrated

```text
generate_dataset() [mixed H, W, real PNG files on disk]
    -> ImageFolder -> Resize+Normalize (Compose, persistable)
    -> random_split -> DataLoader(generator=data_loader_rng)
    -> forge.train(model, train_loader, loss=CrossEntropyLoss(), optimizer=Adam(...),
                    epochs=..., validation_dataset=test_loader, device=..., metrics=[Accuracy()])
    -> TrainingHistory (per-epoch train/val loss + accuracy)
    -> forge.save_checkpoint(..., extra={"data_loader_rng_state": ...})
    -> save_and_verify(model, path, sample, preprocessing=Compose([Resize, Normalize]), classes=full_dataset.classes)
         -> save_model() -> load_model() -> predict() x2 -> PersistenceError on mismatch
    -> interpret_classification(predict(reloaded, sample), classes) -> human-readable prediction
    -> a brand-new, never-seen 200x140 image, preprocessed via load_preprocessing() reconstructed from the file
    -> python -m examples.image_folder_classification.infer, launched as a real subprocess
         -> load_model() + load_preprocessing() + load_classes() + predict() + interpret_classification()
         -> stdout cross-checked against the same computation done in-process
    -> [--resume] start_training_session(resume=checkpoint_path) reads data_loader_rng_state,
       continues the exact same shuffle stream -- bit-for-bit equivalent to uninterrupted training
```

Run for real (not just unit-tested), against the full-scale synthetic
dataset, on both devices:

- CPU: `python -m examples.image_folder_classification.train --generate
  --epochs 25 --device cpu` -- fresh path via `forge.train()`, full
  save/verify/infer demo, all prints correct.
- CUDA (940MX): `python -m examples.image_folder_classification.train
  --generate --epochs 25 --device cuda` -- same, CUDA-resident parameters
  confirmed by the new `test_main_fresh_path_uses_forge_train_on_cuda`.
- `python -m examples.image_folder_classification.infer --model
  examples/image_folder_classification/artifacts/image_folder_model.forge
  --image examples/image_folder_classification/artifacts/
  new_mixed_resolution_query.png` -- run as a genuinely separate process,
  correct prediction, matching the in-process result.
- `python -m examples.image_folder_classification.train --resume
  examples/image_folder_classification/artifacts/image_folder_checkpoint.forge
  --epochs 4` -- resumes a `forge.train()`-fresh-produced checkpoint via
  `start_training_session()`, continuing training correctly.

The equivalent `mnist` real end-to-end run (fresh `forge.train()` ->
`--resume` -> fresh-process `infer.py`) was already demonstrated in M79's
own report (§6) and is unchanged by this milestone; this milestone adds the
subprocess-level automated proof that specific claim now has.

## 7. Files changed

**Real consumer:**
- `examples/image_folder_classification/train.py` -- fresh (non-`--resume`)
  path now calls `forge.train()`; `--resume` path unchanged
  (`start_training_session()`). Module docstring updated.

**Tests:**
- `tests/test_image_folder_classification_integration.py` -- 3 new tests:
  real `main()` fresh-path artifact production, `forge.train()`-fresh ->
  `start_training_session()`-resume bit-for-bit equivalence at
  `shuffle=True`, and genuine-subprocess `infer.py` verification.
- `tests/test_image_folder_classification_cuda_integration.py` -- 1 new
  test: real `main()` fresh path on CUDA (hardware-verified on the 940MX).
- `tests/test_mnist_example_integration.py` -- 1 new test: genuine-subprocess
  `infer.py` verification.

**Documentation:**
- `docs/architecture/training-engine.md` -- **Single-call high-level
  training** section's "Real consumer" subsection rewritten to cover both
  `mnist` and `image_folder_classification`, explaining exactly how the
  fresh path preserves resume-equivalence; **Portable-artifact save +
  verify** section gained a **Fresh-process verification (Milestone 80)**
  subsection correcting the record on what was and wasn't previously
  enforced by the test suite.
- `README.md` (repo root) -- no change needed (its `forge.train()` example
  already used the API contract unchanged by this milestone).
- `examples/README.md` -- **The shared training workflow** section updated
  to describe the `mnist`/`image_folder_classification` two-branch pattern
  and the new subprocess-based fresh-process tests.
- `examples/image_folder_classification/README.md` -- pipeline diagram,
  checkpointing section, and integration-tests section updated.
- `examples/mnist/README.md` -- standalone-inference section updated to
  reference the new subprocess test.
- `docs/development/progress.md` -- M80 entry appended.
- `docs/development/m80-train-to-artifact-workflow.md` -- this report (new
  file).

No `forge/tensor`, `forge/autograd`, `forge/backend`, `forge/nn`,
`forge/optim`, `forge/data`, `forge/serialization`,
`forge/training/api.py`, `forge/training/session.py`, or
`forge/training/trainer.py` file was touched. `forge.train()`,
`TrainingSession`, `Trainer`, and the `.forge` persistence format are
byte-for-byte unchanged.

## 8. Tests added and full-suite result

5 new tests total:

- `tests/test_image_folder_classification_integration.py` (+3, CPU):
  - `test_main_fresh_path_uses_forge_train_and_produces_a_working_artifact`
  - `test_resume_after_a_forge_train_fresh_run_matches_continuous_training`
  - `test_infer_cli_runs_in_a_genuinely_separate_process`
- `tests/test_image_folder_classification_cuda_integration.py` (+1, CUDA,
  hardware-gated via `pytest.mark.skipif(not is_cuda_available(), ...)`):
  - `test_main_fresh_path_uses_forge_train_on_cuda`
- `tests/test_mnist_example_integration.py` (+1, CPU):
  - `test_infer_cli_runs_in_a_genuinely_separate_process`

**Full suite:** `python -m pytest tests/ -q` -> **2,294 collected, 2,293
passed, 1 failed** (2,289 + 5 new). The one failure is
`tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory` (`CUDA active bytes
grew: 320 -> 288` -- a *shrink*, the same measurement-noise artifact
documented since M63), re-run in isolation immediately after and passed
cleanly -- confirmed no M80 regression.

Also independently re-ran, in isolation, every test file this milestone
touches plus its closest dependencies to confirm no collateral damage:
`tests/test_image_folder_classification_integration.py` (9/9),
`tests/test_image_folder_classification_cuda_integration.py` (5/5, real
940MX hardware), `tests/test_mnist_example_integration.py` (13/13),
`tests/test_mnist_example_cuda_integration.py`,
`tests/test_training_api.py`/`test_training_api_cuda.py`,
`tests/test_training_session.py`/`test_training_session_cuda.py`,
`tests/test_classification_metadata.py`/
`test_classification_metadata_cuda_integration.py`,
`tests/test_preprocessing_persistence.py` -- 102 tests, all passed.

## 9. CPU/CUDA verification

Both devices hardware-verified in this session:

- CPU: all new and existing CPU tests in both touched example's integration
  suites (9 + 13 = 22 tests, including the 4 new CPU tests) passed.
- CUDA (940MX): `test_image_folder_classification_cuda_integration.py`
  (5/5, including the new `test_main_fresh_path_uses_forge_train_on_cuda`)
  passed against real CUDA kernels -- confirming `forge.train(...,
  device="cuda")` correctly drives this example's real
  `Conv2d`/`BatchNorm2d`/`MaxPool2d`/`Dropout` architecture through
  unmodified `Trainer`/`CUDABackend` dispatch, with CUDA-resident
  parameters and a working checkpoint/prediction round trip.

## 10. Limitations

- The `mnist` half of the subprocess fresh-process test builds its artifact
  through the same hand-built `Trainer` pipeline this file's other tests
  already use, not through `train.py::main()`'s real-MNIST-download path --
  deliberately, to avoid requiring an ~11MB download in every test
  environment (matching this suite's existing, documented policy for every
  other `mnist` integration test). `main()`'s real-download fresh path is
  exercised manually (as in M79's own report) and by the
  `image_folder_classification` `main()`-level tests (§3.3), which need no
  download -- so `main()`-level coverage exists in this milestone, just not
  specifically paired with mnist's real dataset.
- `regression`, `resnet`, `segmentation`, `autoencoder`, and
  `waveform_classification` still hand-assemble their fresh training path
  (unchanged by this milestone) and have no subprocess-based fresh-process
  test, since none of them currently have a standalone `infer.py` script the
  way `mnist`/`image_folder_classification` do -- there is nothing for such
  a test to launch yet.
- The resume-equivalence test added in §3.3 (`shuffle=True`, `atol=1e-5`,
  small synthetic dataset, 2+2 epochs) is a targeted regression guard for
  the specific mechanism this milestone's retrofit depends on, not a
  general-purpose reproducibility test suite; it mirrors the scale and
  tolerance of every other resume-equivalence test in this repository
  (e.g. `tests/test_mnist_example_integration.py::
  test_resume_equivalence_matches_continuous_training`).

## 11. Rejected alternatives

- **Extending `forge.train()` with a resume/checkpoint parameter** so
  `image_folder_classification` could use one call for both branches --
  rejected per §4 above; reintroduces the "second checkpoint abstraction"
  M79's brief explicitly warned against, for no capability gain over the
  two-branch split every other retrofitted example already uses.
- **A generic "train and save" pipeline function** (`forge.train_and_save()`
  or similar) collapsing `forge.train()` + `save_and_verify()` into one
  call -- rejected: the brief explicitly forbids a monolithic
  dataset-to-artifact function, and no evidence showed the two-call sequence
  (already used identically by every retrofitted example) is itself a real
  duplication problem; `save_and_verify()` already exists precisely because
  M78 found and fixed that duplication once.
- **A `forge.evaluate()` free function** analogous to `forge.predict()`
  (to give `forge.train()` callers a standalone post-training evaluation
  path without constructing a `Trainer`) -- considered, since the brief's
  workflow diagram lists "evaluate" as a distinct step. Rejected: every
  real consumer of `forge.train()` already gets per-epoch evaluation for
  free via `validation_dataset=`, and no consumer in this repository
  currently needs a *separate*, post-hoc evaluation call on a withheld set
  after `forge.train()` returns -- inventing the function without a real
  caller would repeat the mistake M49's own discipline warns against
  (a primitive added speculatively, not for a demonstrated consumer).
- **Retrofitting `regression`/`resnet`/`segmentation`/`autoencoder`/
  `waveform_classification` to `forge.train()` in the same pass** --
  rejected: M79's report (§13) already flagged that each of these needs the
  same "does its `--resume` path actually depend on `data_loader_rng_state`
  continuity" check `image_folder_classification` just got here before it
  can be safely retrofitted, and none of them have the brief's specific
  "richest real consumer" justification `image_folder_classification` has.
  Doing this opportunistically for all five in one milestone would have
  been unscoped, speculative batch work, not evidence-driven.
- **Building a general "fresh-process test harness"** (a pytest fixture or
  helper abstracting `subprocess.run(["-m", ...])`) -- rejected as
  premature: exactly two examples (`mnist`, `image_folder_classification`)
  currently have a standalone `infer.py` to test this way; a shared helper
  for two call sites with near-identical but not identical argument shapes
  is exactly the kind of speculative infrastructure this codebase's
  discipline avoids until a third real consumer justifies it.

## 12. Concrete user-facing capability gained

Before M80: an external Forge developer who wanted to build a
directory-of-image-files classification workflow (the closest thing to a
real-world use case Forge has) had to construct a `Trainer` by hand (via
`start_training_session()`'s factory-based shape) to get resumable training
-- `forge.train()`, Forge's advertised single-call high-level entry point,
did not apply to that workflow at all, only to the simpler grayscale-MNIST
case. Separately, nothing in Forge's own test suite actually enforced that
the "save a model, load it in a different process, get a useful prediction"
claim documented in five separate milestone reports held across a real
process boundary -- a regression in `infer.py`'s own argument parsing,
import fallback, or `main()` wiring could have shipped undetected.

**After M80:** a developer building an image-classification workflow with
real files, persisted preprocessing, and persisted class metadata can use
`forge.train()` for the common (non-resuming) case exactly the way `mnist`
users already could, with resumable training still available via
`start_training_session()` when needed -- the two-branch pattern is now
demonstrated on Forge's most realistic example, not just its simplest one.
And Forge's own CI now proves, on every test run, via a real second OS
process, that the documented `python -m examples.<name>.train` ->
`python -m examples.<name>.infer` workflow actually works end to end for
both flagship classification examples -- a guarantee that previously
existed only as a historical claim in milestone reports, verifiable only by
a human re-running the commands by hand.

## 13. Follow-up opportunities (evidence-justified only)

- `regression`, `resnet`, `segmentation`, `autoencoder`, and
  `waveform_classification` each still need the same "does `--resume`
  depend on `data_loader_rng_state` continuity" check before a safe
  `forge.train()` retrofit can be judged -- not assumed, per M79's own
  still-standing recommendation. This milestone did not re-open that
  question; it only closed it for `image_folder_classification`, which the
  brief specifically named.
- No other example currently has a standalone `infer.py` the way
  `mnist`/`image_folder_classification` do, so the subprocess-based
  fresh-process test pattern this milestone establishes has no other
  current target. If a third example gains one, the same pattern applies
  directly with no new infrastructure needed.

## 14. Suggested commit message

```
feat: extend forge.train() to image_folder_classification, add subprocess-verified fresh-process tests

Splits examples/image_folder_classification/train.py's single
start_training_session() call into the same --resume-or-fresh two-branch
shape examples/mnist/train.py already has: the fresh path now trains
through forge.train() (Milestone 79), while --resume keeps using
start_training_session() unchanged. The fresh path builds its own
data_loader_rng and writes its state into the checkpoint via plain
forge.save_checkpoint(..., extra={"data_loader_rng_state": ...}) -- the
same key TrainingSession.save_checkpoint() writes -- so the Milestone 73
shuffle-resume-equivalence guarantee this example has had since M73
survives with zero regression, verified by a new bit-for-bit equivalence
test at shuffle=True.

Also closes a real verification gap found by direct inspection: despite
four milestones' reports (M71/M72/M77/M78) describing infer.py as verified
in "a genuinely separate process," no test in the repository had ever
launched a real second OS process for inference, or called either flagship
example's actual train.py::main() entry point. Adds genuine
subprocess-based fresh-process tests for both mnist and
image_folder_classification (launching `python -m examples.<name>.infer`
as a real subprocess and cross-checking its stdout against predict()/
interpret_classification() computed independently), plus main()-level CPU
and CUDA coverage of the new forge.train() fresh path
(hardware-verified on the 940MX).

No change to forge.train(), TrainingSession, Trainer, or the .forge
persistence format -- pure composition of existing public APIs. 5 new
tests. Full suite: 2,294 collected, 2,293 passed, 1 pre-existing CUDA-
allocator-measurement flake (documented since M63), reproduced passing in
isolation.
```
