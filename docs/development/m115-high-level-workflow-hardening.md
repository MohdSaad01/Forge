# M115 — High-level workflow hardening and real-world robustness

An evidence-first audit of Forge's four first-class workflows
(`train_image_classifier()`, `ArtifactPredictor.evaluate()`, `train_tabular_classifier()`,
`train_tabular_regressor()`), run the way a developer would use them, to decide whether
either M114-identified limitation justifies a 1.x change.

**Outcome: one targeted change, one deliberate non-change.**

| Candidate | Decision | Class |
|---|---|---|
| Image-classifier preflight | **Retrofitted** (reproduced, fixed, regression-tested, real-data validated) | A — fix now |
| Regression target scaling | **Not justified currently** (measured on real data; needs an architecture decision) | B — document/defer |

## 1. Executive summary

- **Tested:** the three training workflows and `evaluate()` on real data (25,000-image
  petimages, Pima diabetes, UCI Concrete, and — for the scaling question only — the
  StatLib California housing data), on CPU and the 940MX; the three bundled artifacts
  from fresh processes with CUDA hidden and the working directory outside the repo; a
  68-case invalid-input audit run on both the pristine M114 code and the changed code.
- **Changed:** `train_image_classifier()` now rejects a bad `path=`, an incompatible
  `model=` and an unserialisable model **before epoch 1** (path checks before the dataset
  scan), using two small helpers shared with the tabular workflows
  (`forge/training/_preflight.py`). 24 new tests. Before the change a typo in `path=` was
  reported after an estimated ~55 minutes on a real 25,000-image run (extrapolated from
  1,000-image timings, section 3.1), and a `model=` with the wrong
  number of outputs was never reported at all: it trained, saved, verified, and produced
  an artifact that failed on every `predict()`.
- **Not changed:** target scaling. Real data shows a genuine, gradual degradation with
  target magnitude (section 8) but the API behaves as documented, the result is usable
  and its shortfall is visible in the result object, a complete caller-side workaround
  exists, and a proper fix needs an architecture decision (a persisted output-side
  transform) that existing documentation does not settle.
- **Regressions:** none. All 164 M114 tabular tests pass unmodified after the shared-helper
  refactor; M114's real-workload numbers reproduce exactly (Pima 72.7% / `[[80,16],[26,32]]`;
  Concrete MSE 23.21, R² 0.926 on CPU).

## 2. Baseline

Public surface at HEAD `0443a93`: `forge.train_image_classifier()` (M107),
`ArtifactPredictor.evaluate()` (M113), `forge.train_tabular_classifier()` /
`forge.train_tabular_regressor()` (M114), with `load_predictor()` / `predict()` as the
consumer side and three bundled artifacts under `models/`. M114 baseline:
**3,009 passed, 0 failed, 0 skipped, CUDA available**. The implementation matches the
API documented in M111–M114 (checked against `forge/training/image_classifier.py`,
`forge/training/tabular.py`, `docs/development/m114-tabular-workflows.md`, and by
running them).

## 3. Known limitation reproduction

### 3.1 Image-classifier preflight — reproduced

Real petimages subset (100 images/class), 2 epochs, CPU, `Trainer.fit` wrapped in a
timer (training still runs; the wrapper records whether and when it started). Run on the
**unmodified M114 code**:

| Input | Failure boundary | Actual |
|---|---|---|
| `path=` in a missing directory | after all epochs (fit ran 17.3 s) | `PersistenceError` |
| `path=` is a directory | after all epochs (16.4 s) | `PersistenceError: [WinError 5] Access is denied` — unhelpful |
| `path=` parent is a file | after all epochs (16.9 s) | `PersistenceError` |
| `model=` unserialisable `Module` | after all epochs (6.5 s) | `PersistenceError` |
| `model=` 5 logits for 2 classes | **never** — trained, saved, verified | artifact fails every `predict()`: `TrainerError: output has 5 class score(s) but 2 c…` |
| `model=` 1 logit for 2 classes | epoch 1, first batch | `LossError: CrossEntropyLoss target values must be valid class indices…` (not domain-level) |
| `model=` wrong input width / 1-channel | epoch 1, first batch | `ShapeMismatchError` from deep inside a layer |
| `model=` not a `Module` | before training, but after the dataset scan | `TrainerError` |

**Cost at realistic scale**, measured on the 940MX machine (1,000 real images, extrapolated
linearly to the 25,000-image M107 run):

| Stage | Measured | 25,000 images, 5 epochs |
|---|---|---|
| `on_error="skip"` dataset scan (decodes every image once) | 25.1 ms/image | ~10.5 min |
| Training, CUDA | 26.4 ms/image/epoch | ~44 min |
| Training, CPU | 50.7 ms/image/epoch | ~85 min |

So a bad `path=` is reported after **≈55 minutes on CUDA (≈95 minutes on CPU)**, and the
same mistake in `model=` (wrong output count) is never reported. All of these are
predictable before training, and the tabular workflows (M114) already reject the same
inputs before epoch 1. Users reasonably expect one contract from all high-level training
APIs. **Classification: A** — a meaningful reliability problem with a small, well-understood fix.

### 3.2 Regression target scaling — measured on real data

Public API only (`train_tabular_regressor()` at its defaults; `load_predictor().predict()`
for the test R²; 3 seeds unless noted; CPU). Splits are fixed and the test rows are never
seen in training.

**UCI Concrete** (the file was re-fetched and verified byte-identical to M114's: source
`.xls` SHA-256 `710076c6…`, CSV `733f7eb8…`), target multiplied by k (M114's experiment,
re-run):

| target scale | test R² (mean, min–max) | val MSE / baseline |
|---|---|---|
| ×1 (MPa, mean 36) | 0.905 (0.895–0.911) | 0.124 |
| ×30 | 0.794 (0.710–0.840) | 0.302 |
| ×1000 | 0.668 (0.664–0.672) | 0.458 |

**California housing, StatLib original** (`lib.stat.cmu.edu/datasets/houses.zip`; 20,640
rows, 8 features, median house value in **dollars**: mean 206,856, std 115,393; all
finite). A *real* workload whose target is naturally large, not a rescaled one: 3,000-row
training pool, 2,000 test rows (`default_rng(123)` permutation), seeds 0–2:

| target as given to Forge | test R² | epochs run | s/run |
|---|---|---|---|
| dollars (~2×10⁵) | 0.665 (0.664–0.666) | 500, 500, 500 (cap) | 113 |
| ×10 (~2×10⁶), seed 0 | 0.643 | 500 (cap) | 97 |
| ×100 (~2×10⁷), seed 0 | 0.492 | 500 (cap) | 95 |
| $100k units (÷10⁵) | 0.752 (0.747–0.758) | 145, 130, 98 (early-stopped) | 33 |
| dollars, caller z-scores `y`, inverts by hand | 0.752 (0.746–0.759) | — | — |

Reading: target scale is the whole difference on this dataset (the hand-scaled runs
match the ÷10⁵ runs), costing ~0.09 R² and ~3.4× the training time at dollar scale, and
the loss is gradual (0.49 at ×100), never unstable or non-finite. Assessment in section 8.

## 4. Real-world workflow audit

Everything below uses only the public API; no workflow was rebuilt by hand.

**Image classification** (after the change): `forge.train_image_classifier()` on 2,000
real petimages (files 0–999 of each class, numeric order), 3 epochs, CUDA, 204 s. Classes
`['cat','dog']`, 1,599/400 train/val, **1 unreadable image skipped and reported**
(the training range includes the dataset's known corrupt `cat/666.jpg`), val accuracy 0.728. Then `load_predictor()` → `predict()`
(cat 12300 → dog 57.8% — wrong, dog 12300 → dog 93.4% — correct) → `evaluate()` on 400
**held-out** images (the last-numbered files, never trained on): accuracy **0.728 vs 0.500
majority baseline**, loss 0.5502, confusion `[[154,46],[63,137]]`. A fresh process with
`CUDA_VISIBLE_DEVICES=-1` and cwd outside the repo reproduced the confusion matrix and loss
exactly (accuracy 0.7275, loss 0.55017). Accuracy is modest by design (3 epochs on 8% of
the data); the point is that the chain works and the numbers agree across processes and devices.

**Tabular classification, Pima** (`tests/real_world/tabular_workflows.py`, M114's protocol):
CPU and CUDA both give holdout accuracy **72.7% vs 62.3% baseline**, confusion
`[[80,16],[26,32]]`, 85 epochs, best epoch 75 — identical to M114.

**Tabular regression, Concrete** (same script, with the verified CSV): CPU test MSE
**23.21**, MAE 3.39 MPa, baseline 312.23, **R² 0.926** (500 epochs, 33 s); CUDA MSE 24.05,
R² 0.923 (489 epochs, 119 s). Identical to M114; CUDA is not bit-identical to CPU and
about 3.6× slower here (launch-bound at batch 32), as M114 recorded.

Predict and evaluate ran through `load_predictor()` in every case above, on raw inputs.

## 5. Error boundary audit

68 predictable invalid configurations (a standalone script outside the suite, public API only,
`Trainer.fit` wrapped to record whether training started, stray files checked), run on the **pristine M114
tree** and on the changed tree. All 59 tabular and other rows have identical outcomes in both.
The nine image rows that changed:

| Input | Before (M114) | After (M115) |
|---|---|---|
| `path=` missing directory | after training | before the dataset scan, `PersistenceError` |
| `path=` is a directory | after training, `[WinError 5]` | before the scan, `PersistenceError: … is a directory` |
| `path=""` | after training, `[WinError 3]` | before the scan, `DataError` (same message as tabular) |
| `path=` parent is a file | after training | before the scan, `PersistenceError` |
| `model=` 5 logits / 2 classes | **trained and saved; unusable artifact** | before epoch 1, `TrainerError: … output of shape (2, 5) … needs (batch, 2): one raw score per class, for the 2 classes ['cat', 'dog']` |
| `model=` 1 logit | `LossError` at first batch | before epoch 1, `TrainerError` (same message shape) |
| `model=` wrong input width | `ShapeMismatchError` at first batch | before epoch 1, `TrainerError: … could not process a (2, 3, 64, 64) batch … must take (batch, 3, 64, 64) input` |
| `model=` 1-channel on RGB | `ShapeMismatchError` at first batch | same `TrainerError` |
| `model=` unserialisable | after training | before epoch 1, `PersistenceError` |

Rows that were and remain correct, all **before epoch 1** with a domain-level error, for
all three APIs unless noted: `model=` not a `Module`; NaN/Inf in `X` (and in regression
`y`); 1-D/3-D `X`; empty `X`; `len(X) != len(y)`; multi-column or continuous classification
`y`; single class; a class absent from the training split (tabular); string or 3-D regression
`y`; missing/empty/single-class image directory; `image_size` `(0,0)`, `64`, or too small for
the default CNN; `batch_size=0`; negative or NaN `learning_rate`; `val_fraction` out of range.

Remaining imperfections are minor and classified in section 11.

## 6. Consistency analysis

| Concern | Image classification | Tabular classification | Tabular regression |
|---|---|---|---|
| Preflight before epoch 1 | yes (M115) | yes | yes |
| Invalid `path` | before scan / before epoch 1 | before epoch 1 | before epoch 1 |
| `model=` validation | probed on 2 real images | probed on 2 real rows | probed on 2 real rows |
| Unserialisable `model=` | dry-run save | dry-run save | dry-run save |
| NaN/Inf | n/a (decoded pixels are finite); unreadable files skipped or raised | `DataError` | `DataError` (`X` and `y`) |
| Preprocessing | fixed `Resize` + `Normalize(0, 255)`, built once | `Normalize` fitted on training rows, opt-in `ReplaceValue` | same |
| Persistence | `save_and_verify`, `task="classification"`, classes | `task="tabular_classification"`, classes | `task="regression"` |
| Result numbers | final-epoch training metrics | metrics of the *saved artifact* via `evaluate()` (+ baseline) | same |
| `evaluate()` / `predict()` | directory / image path | rows | rows |
| Argument validation | mixed types (below) | uniform `DataError` | uniform `DataError` |
| `path` type | `str` or `PathLike`, echoed as given | normalised to `str` | normalised to `str` |
| Absent class in training split | not checked | `DataError` | n/a |
| Early stopping / `verbose` | none / on | on (`patience`) / off | same |

Justified differences: fixed vs fitted preprocessing (pixels have a known scale), no early
stopping and verbose progress (epochs take minutes, not milliseconds), result metrics
source. **Unjustified but minor, and left alone (section 11, class B):** the image API's
argument validation is uneven (`epochs=0` is a `TrainerError` raised on entry to `fit`,
`learning_rate` an `OptimizerError`, `seed=-1` a raw NumPy `ValueError`), it does not
reject a class that lost all its images to the validation split, and it echoes `path` as
given. None fails late or silently produces a wrong artifact.

## 7. Changes implemented

### Image-classifier preflight

- **Problem.** Predictable mistakes surfaced after training (bad `path=`, unserialisable
  model) or never (`model=` with the wrong number of outputs).
- **Evidence.** Section 3.1: ~55 min (CUDA) to ~95 min (CPU) to report a typo'd path on a
  real run; an unusable artifact from a 5-logit model; the same inputs already rejected by
  the tabular APIs.
- **Fix.** `train_image_classifier()` now: (1) before the dataset scan — checks `path` is a
  non-empty path whose directory exists and that is not a directory, and that `model=` is a
  `Module`; (2) after the split — moves the model to its device, and for a caller-supplied
  `model=` *runs it once* on two real preprocessed images and requires `(batch,
  len(classes))` scores (the M114 approach: probe with real input, no architecture
  introspection); (3) dry-runs `save_model()` on the untrained model into a temporary
  sibling of `path`, removed immediately. The final artifact is still written only after
  training. Shared code lives in `forge/training/_preflight.py`
  (`check_save_path`, `preflight_save`, `check_model_contract`); M114's `tabular.py` now
  calls them instead of carrying its own copy, with byte-identical messages.
- **Why this scope.** The two helpers are *moved*, not duplicated: the tabular
  `_preflight_save` was already workflow-agnostic and its probe differed only in
  wording. Each workflow still decides what to check and in what order — no preflight
  framework, no callbacks, no changes to `Trainer`, serialisation, preprocessing or results.
  The default CNN is not probed (it is built for `image_size` in the same function). The
  path check runs before the scan because `on_error="skip"` decodes every image
  once (~10 min at 25,000 images), so that is the dominant cost of a typo.
- **Tests.** `tests/test_image_classifier_preflight.py` (19, CPU) and
  `tests/test_image_classifier_preflight_cuda.py` (5, CUDA); see section 10. They fail on the
  M114 code (15 of 19 CPU tests fail there; the other 4 are "still works" tests) and pass now.
- **Real-world validation.** Section 4 (2,000 real images, CUDA, held-out evaluation,
  fresh-process CPU reproduction) and `tests/real_world/petimages_smoke.py`
  (160/160 images, 0 skipped, 2 epochs on CPU in 10.2 s, artifact verified, inference valid).

Behaviour changes to note: errors that used to be `LossError`/`ShapeMismatchError` at the
first batch are now `TrainerError` before epoch 1; `path=""`/`None` is a `DataError` instead
of a late `PersistenceError`; a `model=` that produced the wrong number of outputs now raises
instead of silently producing an unusable artifact. No successful call changes: a same-seed
run is bit-identical before and after (loss values and a SHA-256 over every parameter and buffer,
default and custom `model=`, compared across the M114 and M115 trees).

Documentation updated for this change only: the module and function docstrings of
`image_classifier.py`, `docs/architecture/training-engine.md` (Preflight paragraph),
`README.md`.

## 8. Target scaling decision

**Not justified currently — class B (meaningful capability gap, evidence-backed, blocked on a
design decision).**

The brief's test is whether the limitation is (A) a concrete user-facing defect, (B) a
meaningful but currently unneeded capability, or (C) theoretical. Section 3.2 shows it is
not C: on a real dataset with a naturally large target, unscaled training costs ~0.09 R²
and ~3.4× the time. It is not A, because:

1. **The API does what it documents.** `train_tabular_regressor()`'s docstring, the README and
   M114 all say targets are trained in their own units and that very large ones degrade;
   nothing misbehaves relative to that contract. In `maintenance.md`'s classification this is a
   feature request, not a bug.
2. **Training is stable and the result is usable.** No non-finite loss, no divergence, R² well
   above the baseline at every magnitude tested (0.49 even at ~2×10⁷). Concrete, the one
   regression workload Forge validates end to end, needs nothing.
3. **The shortfall is visible.** `epochs_completed` equals the cap with `stopped_early=False`
   and `validation_mse / baseline_mse` is 0.33–0.49 rather than ~0.12, in the result object.
4. **A complete workaround exists and was measured to recover everything** (0.752 = 0.752).
   Its cost is real: the artifact then predicts in standardised units and must be inverted
   with constants the caller keeps out of band — exactly what the "self-contained artifact"
   principle exists to avoid.
5. **A correct fix is an architecture decision, not a small change.** The artifact carries
   *input* preprocessing only. A persisted, inverted target transform means either extending
   the artifact format with an output-side step that `predict()`, `evaluate()`, the CLI and
   `inspect_model()` all honour, or folding the affine map into the final `Linear` layer after
   training (self-contained with no format change, but it only works for models ending in a
   `Linear`, so `model=` would be treated differently, and the saved weights would no longer be
   the trained ones). Nothing in the docs chooses between these, `CLAUDE.md` says to stop and
   report rather than improvise such a choice, and the brief rules out redesigning
   serialisation inside this milestone.

**Trigger for reconsidering** (no milestone number assigned): a real regression workload
where either (a) the default result is unusable — validation R² near the baseline, or a
non-finite loss, that scaling `y` by hand fixes — or (b) the caller cannot apply the
workaround because the artifact is consumed by code they do not control (`forge model
predict`, a bundled consumer script) and must return native units. If either occurs, the
first thing to settle is which of the two designs above to adopt; the requirements would then
be exactly the brief's: fit on training targets only, persist the transform, `predict()` and
`evaluate()` in the user's units, fresh-process and CPU/CUDA parity, no second preprocessing
system.

This finding is stronger than M114's (which used a rescaled Concrete), and the decision is a
judgement call; the evidence above is what to revisit if you weigh it differently.

## 9. Preflight decision

**Retrofitted** (section 7). Evidence: reproduced late and silent failures (3.1),
a measured 55–95 minute cost on a real run, the same contract already shipped for the two
tabular workflows, and a fix that reuses M114's mechanism rather than adding one.

## 10. Tests

**New**

- `tests/test_image_classifier_preflight.py` — 19 CPU tests: each bad-path variant fails
  before epoch 1 (tripwire on `Trainer.fit`), the missing-directory / directory / empty-path /
  non-`Module` cases fail *before the dataset scan* (tripwire on `ImageFolder`), nothing is
  written and no `.forge-preflight-*` file is left behind, the final artifact does not exist at
  epoch 1; wrong output width (5 and 1 for 2 classes), wrong input width, 1-channel model and
  a non-reducing model are `TrainerError` naming the classes and expected shape; the probe
  sees a real preprocessed `(2, 3, H, W)` float32 batch in `[0, 1]`; a rejected model keeps its
  training flag and BatchNorm statistics; unserialisable model; a valid custom model and the
  default model still train, save, predict, and `evaluate()`; same seed is bit-identical.
- `tests/test_image_classifier_preflight_cuda.py` — 5 CUDA tests (skip cleanly without CUDA):
  wrong output width and bad path rejected before epoch 1 on CUDA; a CPU model is on CUDA when the
  probe runs (the probe records the device its input arrived on) and then trains there; the same
  saved weights evaluate identically on CPU and CUDA.

**Existing:** the 24 M107 image tests and all 164 M114 tabular tests pass unmodified (the
tabular tests exercise the code that was moved into `_preflight.py`). 15 of the 19 new CPU
tests fail on the M114 code; the other 4 are "still works" tests that must pass on both.

**Full suite** (`python -m pytest tests/ -q`, CUDA available, 940MX):
**3,033 passed, 0 failed, 0 skipped** in 537 s. The delta from M114's 3,009 is exactly the 24
new tests (19 CPU + 5 CUDA); no existing test was modified.

History, for the record: an earlier full run (3,032 passed) predated a late edit to one CUDA
test, and the next run failed that one test (`test_a_cpu_model_is_moved_to_cuda_before_the_probe`)
because its own toy model called `reshape(n, -1)` on a CUDA tensor, which CUDA rejects (finding in
section 11). The defect was in the test model, not in `forge/`; it was fixed and the suite re-run
clean.

## 11. Remaining findings

| Finding | Class | Note |
|---|---|---|
| Image API: a class whose images all land in the validation split is not rejected (tabular rejects it) | B | Silent but needs a tiny class; a new rejection would change 1.x behaviour; no real workload hit it |
| Image API argument validation uneven (`epochs=0` `TrainerError` on entry to `fit`; `learning_rate` `OptimizerError`; `seed=-1` raw NumPy `ValueError`) | B | All fail before epoch 1, cheaply; only the exception type differs |
| `ImageClassifierResult.artifact_path` echoes a `Path` as given (tabular normalises to `str`) | B | Cosmetic; `Path` works |
| `predict()` on a ragged nested list raises a plain `ValueError`, not a `ForgeError` (M113/M114 finding; hit again in the consumer audit) | B | Immediate, correctly rejected, message understandable; brief's three-part bar for fixing it in M115 not met (not clearly worth changing on its own) |
| Regressor `predict()` accepts a 1-D row (returns shape `(1,)`) while the classifier raises `DataError` | B | Correct value either way |
| `predict()` on 3-D input raises `ShapeMismatchError` from `Linear` (a `ForgeError`, but not a `DataError`) | B | Clear enough; correct rejection |
| `predict()` on 0 rows returns an empty result | C | Empty in, empty out |
| Regression target scaling | B | Section 8, with trigger |
| `Tensor.reshape(n, -1)` works on CPU but raises on CUDA (found while writing a CUDA test model; identical on the M114 tree) | B | Model-authoring API, not the high-level workflows; a caller's `model=` using `-1` fails on CUDA -- and the new probe now reports that as a `TrainerError` before epoch 1 instead of mid-training |
| Small MLPs train slower on CUDA than on this CPU | C | Recorded in M114; launch-bound, supported |
| Bundled image model device default | D | Fixed in I1; consumer ran with CUDA hidden |
| `.gitignore` contradicting `models/` | D | Disproved in I1; `git check-ignore` empty for all three artifacts, no ignored-but-tracked files |
| `maintenance.md` smoke-test reference | D | Fixed in I1; `tests/real_world/petimages_smoke.py` is tracked and referenced |
| `maintenance.md` regression baseline was stale (2,802) | D | Updated to this run's figures (section 10) |
| `Trainer.evaluate()` has no persisted-preprocessing hook | B | Documented since M97; `ArtifactPredictor.evaluate()` is the supported path |

## 12. Product state

The high-level surface is coherent and complete for the intended 1.x scope: three training
workflows and saved-artifact evaluation, one contract for what is rejected before epoch 1, one
consumer path (`load_predictor()` → `predict()` / `evaluate()`), artifacts that work from a fresh
CUDA-less process, and identical results across CPU and CUDA for the same saved weights. The audit
found no other defect that fails late or silently produces a wrong artifact. The one capability
with real evidence behind it — target scaling — is documented with a measured trigger.

## 13. Next step

No scheduled next milestone. Return to real-world usage and maintenance.

The one open decision, for you rather than a new milestone: whether to adopt a persisted
target transform, and if so which of the two designs in section 8. The audit did not find
evidence that it is needed to make the current workflows *work*, only that it would make
one class of regression workloads work better.
