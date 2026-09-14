# Milestone 91 — Tabular-Classification-Artifact Inference: `forge.predict_tabular_classification_artifact()`

## Objective

Stress the training and data-preparation side of Forge with a realistic new
workload, discover whatever real limitation it exposes, and fix only that --
no architecture survey, no feature chosen in advance. This is the second
workload-driven milestone (after M90's sequence-generation workload); the
brief explicitly forbade starting with "Forge needs feature X" and required
implementing the baseline workload first, with existing public APIs, before
writing any framework code.

## Selected workload

**Tabular classification**: a synthetic device-telemetry health classifier.
Ten continuous sensor-style features per device; four health states
(`normal`/`warning`/`critical`/`fault`). Six features are genuinely
discriminative (drawn around one of four fixed per-class center vectors,
with deliberate class overlap); four are pure distractors, drawn independent
of class -- mirroring `examples/regression/dataset.py`'s own `x7`
distractor. Built entirely in-process from a fixed seed, no download, no
third-party dependency -- the same precedent `char_rnn`/`word_rnn`/
`regression` already established.

## Why this workload

`examples/regression` already exercises tabular (non-image) input, but only
for *regression* output. `mnist`/`image_folder_classification` already
exercise *classification* output, but always over image input. No existing
example combined tabular input with classification output -- a materially
different, previously unvalidated combination, and (per the brief's Section
4 criteria) realistic, reproducible, small, and understandable to an
external developer. It also directly stresses `forge.data.random_split`
under a condition `examples/regression` never needed: class-blocked raw
data (see **Data pipeline finding** below).

## External developer goal

Train a tabular classifier end-to-end using only Forge's public API
(`forge.train_and_save()`, the preferred high-level entry point per the
brief's Section 9), save a portable artifact, and predict on a brand-new raw
feature vector using the same `forge.predict_model()` entry point every
other artifact shape already supports -- with no manual reconstruction of
preprocessing, no private-module access, and no need to already know
Forge's internals.

## Baseline workflow (as actually attempted)

```text
examples.tabular_classification.dataset.make_datasets()
    -> DataLoader(shuffle=True)
    -> Sequential(Linear/ReLU/Linear/ReLU/Linear)
    -> CrossEntropyLoss + Adam
    -> forge.train_and_save(..., classes=CLASS_NAMES, task="classification")
    -> forge.predict_model(model_path, raw_query_array)
```

Run for real (`python -m examples.tabular_classification.train --epochs 30
--device cpu`), not simulated. Training worked with zero framework changes:
**82.1% test accuracy against a 25.0% trivial (uniform-random) baseline** --
genuine learning, not merely "the code runs" (98.6%-style reduction claims
would be dishonest here; this dataset has deliberate class overlap by
design, so 82% against a 25% floor is the right evidence bar).

The very next step -- predicting on a brand-new raw feature vector via
`forge.predict_model()`, exactly the pattern `examples/regression/train.py`'s
own end-of-run demo already establishes for the regression shape -- failed
immediately:

```text
Traceback (most recent call last):
  File "examples\tabular_classification\train.py", line 158, in main
    result = forge.predict_model(str(model_path), raw_query.astype(np.float32))
  File "forge\training\inference.py", line 853, in predict_model
    return predict_artifact(path, input_data, device=device)
  File "forge\training\inference.py", line 356, in predict_artifact
    raise DataError(
forge.exceptions.DataError: predict_artifact() requires image to be a file path (str or os.PathLike), got ndarray.
```

## Existing Forge capabilities used (worked unmodified)

- `forge.data.TensorDataset`, `forge.data.Normalize` (feature standardization
  fit on the training split only).
- `forge.data.random_split` (see **Data pipeline finding** below).
- `forge.data.DataLoader` (shuffling, batching, the final partial batch).
- `forge.nn.Sequential`/`Linear`/`ReLU`, `forge.nn.CrossEntropyLoss`.
- `forge.optim.Adam`.
- `forge.train_and_save()` (`forge/training/api.py`) -- the fresh-training
  path used directly, no manual `Trainer` assembly, no `--resume`/checkpoint
  branch (see **Checkpointing** below).
- `forge.training.Accuracy` metric.
- `forge.save_model(..., preprocessing=..., classes=..., task=...)` /
  `forge.inspect_model()` / `forge model inspect`.

## Data pipeline finding (investigated, not a blocker)

This workload's raw data is generated in **class-block order** (every
`normal` row, then every `warning` row, ...) -- deliberately, to mirror a
common real-world shape (a CSV exported one class at a time). Section 12 of
the brief requires checking whether the existing split utilities are
sufficient for this shape before assuming they are. `examples/regression`'s
`sequential_split` (consecutive index blocks, no RNG) would be actively
wrong here: it would put entire classes only in the training split, with
none in validation/test. `forge.data.random_split` (permute-then-slice,
already existing, already used elsewhere for order-meaningful data e.g.
`trainer_demo.py`) handles this correctly with no new capability needed --
confirmed directly by this milestone's own training run, whose actual
training-split class counts (`[309, 296, 300, 295]` out of 1,200 training
samples at the README's reference settings) are close to balanced despite
`random_split` being deliberately *not* stratified. No stratified-split
capability was added -- per the brief's Section 12, this workload does not
demonstrate a need for exact per-class balance, only a representative mix,
which plain `random_split` already provides.

## Problems actually discovered

| # | Step | Observed behavior | Root cause | Severity |
|---|---|---|---|---|
| 1 | Predicting on a saved tabular classification artifact | `forge.DataError: predict_artifact() requires image to be a file path (str or os.PathLike), got ndarray` | `forge.predict_model()`'s only classification workflow (`predict_artifact()`, M82) is scoped to image input by design -- there was no artifact-level classification inference path for numeric input at all | **Blocker** |
| 2 | Class-blocked raw data + train/val/test split | Not a failure -- confirmed `sequential_split` would be wrong here, `random_split` is correct | N/A (existing capability sufficient, investigated per Section 12) | Not friction |

Only one real problem was found. Per the brief's Section 7 ("M91 should
normally have one primary capability/fix"), this milestone has exactly one.

## Root cause

Not "the artifact predicts wrong" -- it never reaches prediction at all.
The actual cause, traced through `forge/training/inference.py`:
`_determine_workflow()` returns `"classification"` for any artifact with
`info.task == "classification"` (or, absent an explicit task, one with a
saved `classes` vocabulary); `predict_model()` then calls
`predict_artifact(path, input_data, device=device)` **unconditionally** for
that workflow. `predict_artifact()` (M82) was deliberately, correctly scoped
to image-classification artifacts only -- its own docstring says so
explicitly ("Milestone 82's scope is image-classification artifacts
specifically, not a generic input/artifact runtime") -- and requires `image`
to be a file path, decoded via `ImageFolder._load_image()`. There is no
architectural ambiguity or bug in `predict_artifact()` itself: the actual
gap is that **no sibling function was ever built for a classification model
whose input is numeric, not a file** -- the exact gap `predict_tensor_
artifact()` (M83) already closed for *regression*, never extended to
classification. Training a tabular classifier was never blocked (`Trainer`/
`DataLoader`/`CrossEntropyLoss`/`classes=` all already support it fully);
only its portable-artifact *inference* path was missing.

## Primary problem selected and why

Problem #1 (the blocker) -- the only real problem this milestone found, and
by definition the highest-value one: it makes the entire workload's natural
final step ("predict on a new sample from the saved file alone") impossible
through any documented Forge API.

## Implementation

**Smallest complete fix**, following the exact precedent M83 (regression)
and M90 (sequence) already established for their own new artifact shapes:

1. **`forge/serialization/model.py`**: `TASK_TYPES` gains a fifth value,
   `"tabular_classification"` (`("classification", "regression",
   "segmentation", "sequence", "tabular_classification")`). Like `"sequence"`
   (M90), it has **no legacy-architecture fallback** -- `"tabular_
   classification"` is architecturally identical to `"classification"` (both
   may be `Linear`-terminated with a `classes=` vocabulary), so no guess
   from `ModelSummary.module_types` could ever safely tell them apart; an
   explicit `task=` declaration is the only reliable signal, following M87's
   own "never guess, always declare" philosophy. Unlike `"sequence"`,
   `classes=` remains **optional** for this task (mirroring
   `"classification"`'s own policy) -- `_validate_task()` needed no new
   restriction, since the existing `task in ("regression", "segmentation")`
   classes-rejection clause already excludes it by construction.
2. **`forge/training/inference.py`**: new `predict_tabular_classification_
   artifact(path, input_data, *, device=None)` -- composes `load_model()` +
   `load_preprocessing()` (optional, like `predict_tensor_artifact()`) +
   `predict()` + `load_classes()` (optional, like `predict_artifact()`) +
   `interpret_classification()`.
   **One deliberate design decision, made and documented explicitly**:
   `input_data` follows `predict_tensor_artifact()`'s "already batched, any
   batch size" convention -- not `predict_artifact()`'s "always exactly one
   image file" convention. Returning only the first row's result (mirroring
   `predict_artifact()`'s `[0]` indexing) would have silently discarded real
   caller data for any batch size greater than one -- a genuine correctness
   risk this milestone deliberately avoided rather than copying the nearest
   precedent uncritically. The function therefore returns **one result per
   input row** (`list[ClassificationPrediction] | list[int]`), always a
   list, even for a single-sample call.
   `_determine_workflow()`/`predict_model()` gained one new `elif` branch
   dispatching `task="tabular_classification"` to this function -- no change
   to the dispatch function's shape.
3. **`forge/cli/model.py`**: `cmd_predict` gained a `task ==
   "tabular_classification"` branch. Reuses `regression`'s numeric-JSON-file
   input parsing (the private helper was renamed `_parse_regression_input`
   -> `_parse_numeric_input`, since it is now shared by two tasks with
   identical input shapes but different output interpretation) but prints
   classification-shaped output via a new `_print_classification_results()`
   (plural -- distinct from the existing singular `_print_classification_
   result()` used by `task == "classification"`, since this task's result is
   always a list): a single-row input prints identically to `task ==
   "classification"`'s own single-image text output; a genuinely multi-row
   JSON input prints one numbered `Sample N:` block per row, and `--json`
   emits `{"task": "tabular_classification", "predictions": [...]}` (a list,
   one entry per row) rather than a flat single-result payload.
4. Exports: `predict_tabular_classification_artifact` added to
   `forge/training/__init__.py` and `forge/__init__.py`'s public surface
   (alongside the existing four artifact-shape functions), plus the
   corresponding docstring/`TASK_TYPES`-comment updates in both files.

## Public API changes

- New: `forge.predict_tabular_classification_artifact(path, input_data, *,
  device=None) -> list[ClassificationPrediction] | list[int]`.
- New: `"tabular_classification"` added to `forge.serialization.model.
  TASK_TYPES` and therefore to `forge.save_model(..., task=...)`'s accepted
  values.
- `forge.predict_model()` gained one new dispatch branch; no change to its
  existing signature or behavior for the other four tasks.
- **No breaking change** to any existing public function. `predict_artifact()`
  /`predict_tensor_artifact()`/`predict_image_artifact()`/
  `predict_sequence_artifact()` are byte-for-byte unchanged.

## CLI changes

- `forge model predict` accepts `task="tabular_classification"` artifacts:
  same JSON-numeric-file input convention as `task="regression"`, printing
  classification-shaped output (label/confidence, or numbered per-row blocks
  for a multi-row input).
- Internal-only rename: `_parse_regression_input` -> `_parse_numeric_input`
  (a private CLI helper, not part of the public API; the three existing
  tests that imported it by name were updated accordingly).
- No change to any existing task's CLI input/output shape or exit codes.

## Artifact changes

No format change (`FORMAT_VERSION` unchanged) -- `"tabular_classification"`
is just a new accepted string for the already-existing, already-optional
`"task"` metadata key, exactly like M90's `"sequence"` addition. Files saved
before this milestone remain fully loadable and behave identically.

## Example changes

New `examples/tabular_classification/` (`__init__.py`, `dataset.py`,
`model.py`, `train.py`, `infer.py`, `README.md`) -- the first example
combining tabular input with classification output. `train.py`'s fresh
(only) training path uses `forge.train_and_save(..., classes=CLASS_NAMES,
task="tabular_classification")` directly; deliberately **no `--resume`/
checkpoint path** -- this workload's dataset generation and training are
both fast enough (single-digit seconds for the README's reference run) that
interrupted-training resume is not a realistic concern (brief Section 15),
so it was not built. `infer.py` is a standalone, fresh-process script
(mirroring `examples/segmentation/infer.py`'s independence from `train.py`)
whose `--input` is a literal JSON array on the command line, matching
`examples/char_rnn/infer.py`'s own "short input as literal text, not a
file" precedent for a Python-API example (distinct from the CLI's own
JSON-*file* convention).

## Exact files changed

- `forge/serialization/model.py` -- `TASK_TYPES`, `_validate_task`/
  `_validate_classes`/`save_model()`/`inspect_model()` docstrings.
- `forge/training/inference.py` -- new `predict_tabular_classification_
  artifact()`; `_determine_workflow()`/`predict_model()` dispatch branch and
  docstrings; module docstring.
- `forge/training/__init__.py`, `forge/__init__.py` -- exports.
- `forge/cli/model.py` -- `_parse_regression_input` renamed to
  `_parse_numeric_input`; new `_print_classification_results()`; new
  `cmd_predict` branch; module/argparse-help docstring updates.
- `examples/tabular_classification/__init__.py`, `dataset.py`, `model.py`,
  `train.py`, `infer.py`, `README.md` -- new.
- `examples/README.md` -- new table row; portable-artifact-inference-
  function-count paragraph updated (three -> five, adding the M90/M91
  functions that paragraph had not been updated for).
- `docs/architecture/persistence.md` -- **Task metadata** section updated
  for the fifth task value; new **Tabular-classification-artifact
  prediction** section.
- `docs/development/cli.md` -- **Model prediction** section: new
  `tabular_classification` bullet, `--json` example, legacy-artifacts
  paragraph.
- `docs/development/progress.md` -- this milestone's entry.
- `docs/development/m91-tabular-classification-artifact-inference.md` --
  this file.
- `tests/test_tabular_classification_artifact_prediction.py`,
  `tests/test_tabular_classification_artifact_prediction_cuda.py` -- new.
- `tests/test_unified_artifact_prediction.py`, `tests/test_cli_predict.py`,
  `tests/test_task_metadata.py` -- extended/updated (see **Tests** below).

## Tests added/modified

- `tests/test_tabular_classification_artifact_prediction.py` (new, CPU, 10
  tests) -- unit-level coverage of `predict_tabular_classification_
  artifact()` (per-row results, optional classes, optional preprocessing,
  preprocessing correctness against a hand-computed reference, invalid
  input type, missing file) and end-to-end coverage using the real example
  (artifact metadata completeness, genuine learning vs. trivial baseline,
  a genuinely separate-process `subprocess` prediction-agreement check,
  `predict_model()`/`predict_tabular_classification_artifact()` agreement
  after real training).
- `tests/test_tabular_classification_artifact_prediction_cuda.py` (new,
  CUDA, 3 tests, hardware-verified on the reference GeForce 940MX) --
  predicts-on-CUDA-by-default, `predict_model()` CUDA dispatch, and
  CPU-vs-CUDA prediction parity for the same CPU-saved artifact.
- `tests/test_unified_artifact_prediction.py` (+3 CPU tests) -- dispatch,
  bit-for-bit delegation match, and input-type-rejection coverage for the
  new workflow; existing backward-compatibility and fresh-process tests
  extended to include this fifth artifact shape.
- `tests/test_cli_predict.py` (+4 CPU, +1 CUDA) -- CLI routing, `--json`
  output (with and without saved classes), multi-row input reporting, CUDA
  device handling; existing fresh-process subprocess test extended to
  include this fourth CLI task shape; three existing tests updated for the
  `_parse_regression_input` -> `_parse_numeric_input` rename.
- `tests/test_task_metadata.py` -- `test_task_types_is_the_documented_
  vocabulary` updated to the new 5-value tuple (`test_save_model_accepts_
  each_documented_task`, already parametrized over `TASK_TYPES`, covers the
  new value with no code change of its own).

## Full-suite result

2,526 collected (22 new over M90's 2,504); **2,525 passed, 1 failed** -- the
same pre-existing `test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory` allocator-measurement
flake documented since ~M63 (an unrelated CUDA memory-accounting timing
issue, not touched by this milestone). No M91 regression.

## Fresh-process verification

- This milestone's own new artifact shape: `tests/test_tabular_
  classification_artifact_prediction.py::test_predict_tabular_
  classification_artifact_agrees_with_a_genuinely_separate_process` trains
  and saves a real artifact in one process, then launches a genuine
  `subprocess.run([sys.executable, "-c", ...])` that only imports
  `forge`/`numpy` and calls `forge.predict_tabular_classification_
  artifact()` directly, confirming its printed prediction matches this
  process's own prediction on the identical raw input.
- `examples/tabular_classification/infer.py` was also run directly against
  a freshly trained artifact from a separate `python -m` invocation and
  matched `train.py`'s own end-of-run prediction exactly (see
  **End-to-end workload result** below).
- `tests/test_unified_artifact_prediction.py::test_predict_model_works_
  from_a_genuinely_separate_process_for_all_four_artifact_shapes` and
  `tests/test_cli_predict.py::test_cli_predict_fresh_process_all_four_
  task_shapes` extend their existing subprocess-based checks to include
  this fifth/fourth shape respectively.

## CPU verification

Every workflow above (training, inspection, prediction, both new test
files, the extended existing test files, the full suite) ran and passed on
this machine's CPU backend. The real example was also retrained and
manually verified end-to-end via direct `python -m examples.
tabular_classification.train`/`.infer` invocations and the `forge` CLI (see
below), not only through the automated test suite.

## CUDA verification

`tests/test_tabular_classification_artifact_prediction_cuda.py` (3 tests)
and `tests/test_cli_predict.py::
test_cli_predict_tabular_classification_reaches_cuda_inference_path` (1
test) all ran and passed on the reference GeForce 940MX (CC 5.0, CUDA
Toolkit 12.6) -- confirming a `task="tabular_classification"` model
trained/saved on CUDA predicts on CUDA by default, `forge.predict_model()`
dispatches correctly on CUDA, and a CPU-saved artifact produces matching
predictions whether explicitly loaded onto CPU or CUDA.

## End-to-end workload result

Manually verified, not just via the automated test suite:

```text
python -m examples.tabular_classification.train --epochs 30 --device cpu
  -> Final test evaluation: accuracy=82.1% (baseline 25.0%)
  -> Saved + verified model + preprocessing -> tabular_classification_model.forge
  -> Predicting a brand-new raw sample via forge.predict_model():
       Prediction: fault (confidence 92.6%) -- true label was 'fault'

python -m examples.tabular_classification.infer --model ... --input '[...]'
  -> Prediction: fault (confidence 92.6%)   # matches train.py's own demo exactly

python -m forge model inspect tabular_classification_model.forge
  -> Task: tabular_classification
     Preprocessing: yes (Normalize(...))
     Classes: normal, warning, critical, fault

python -m forge model predict tabular_classification_model.forge input.json
  -> Prediction: fault
     Confidence: 92.6%

python -m forge model predict tabular_classification_model.forge input.json --json
  -> {"task": "tabular_classification", "predictions": [{"class": "fault", "confidence": 0.926...}]}
```

The full documented pipeline -- dataset -> training -> artifact ->
`forge model inspect` -> `forge model predict` / `forge.predict_model()` /
`examples/tabular_classification/infer.py` -- completed successfully using
only public APIs, with the Python API and both CLI forms agreeing exactly.

## Existing workflow regression result

Each of the four pre-existing task workflows was **retrained fresh** (not
just re-run from stale artifacts) and re-verified end-to-end via `forge
model inspect`/`forge model predict`/each example's own `infer.py`:

- **Classification** (`image_folder_classification`, `--generate
  --samples-per-class 20 --epochs 3`): `forge model inspect` reports `Task:
  classification`; `forge model predict` on a new mixed-resolution query
  image matches `train.py`'s own printed prediction exactly (`triangle`,
  39.6%).
- **Regression** (`examples/regression`, `--epochs 3 --n-train 200`): `forge
  model inspect` reports `Task: regression`; `forge model predict` on a raw
  JSON feature vector succeeds and returns a numeric prediction.
- **Segmentation** (`examples/segmentation`, `--epochs 2 --n-train 40`):
  `forge model inspect` reports `Task: segmentation`; `forge model predict
  --output mask.png` succeeds and writes a real mask file.
- **Sequence** (`examples/char_rnn`, `--epochs 3`): `forge model inspect`
  reports `Task: sequence`; `examples/char_rnn/infer.py` generates text
  successfully from the fresh artifact; the bare `forge model predict` CLI
  still fails with the same, already-documented, unchanged `CharRNN` custom-
  module-registration limitation (`"it is not registered for persistence in
  this process"`) -- confirming this pre-existing, intentional restriction
  was not altered by this milestone.

No regression in any of the four. This is in addition to, not a replacement
for, each workflow's own existing automated test suite, all of which also
passed unchanged in the full-suite run above.

## Remaining limitations

- **No stratified splitting.** `random_split` is not class-stratified;
  per-class training counts vary slightly (e.g. `[309, 296, 300, 295]` for
  1,200 requested training samples at the README's reference settings, not
  exactly `[300, 300, 300, 300]`). Not added -- this workload does not
  demonstrate a need for exact balance, only a representative mix, which
  `random_split` already provides (brief Section 12's explicit instruction).
- **No CSV/tabular file loading.** Like `examples/regression`, this
  workload's data is synthetic and in-process; no `Dataset` for reading a
  real CSV/Parquet file was built or needed. Not exercised as a real need by
  this workload.
- **`predict_tabular_classification_artifact()`'s list-per-row return is a
  deliberate divergence from `predict_artifact()`'s single-result
  convention** -- documented extensively (see **Implementation** above), not
  an oversight; a caller must index `[0]` for the common single-sample case.

## Explicitly deferred features

Per the brief's Section 24 and this milestone's own findings, none of the
following were added, because no exercised workflow demonstrated a concrete
need: batch/directory prediction beyond what per-row JSON batching already
provides, CSV/NDJSON tabular data loading, stratified/k-fold/grouped/
time-aware splitting, model serving/REST/HTTP inference, a model registry,
distributed/multi-GPU training, AMP, a generalized "any classification
input" runtime replacing the per-artifact-shape function discipline, or any
other speculative capability.

## What an external developer can now do

Train a tabular classifier on structured, non-image data using
`forge.train_and_save(..., classes=[...], task="tabular_classification")`
-- the same high-level entry point every other task already uses -- and get
a correct, human-readable classification prediction on a brand-new raw
feature vector from the saved `.forge` file alone, via `forge.predict_model()`,
`forge.predict_tabular_classification_artifact()` directly, or `forge model
predict` from a completely separate process, without ever needing to
reconstruct preprocessing, class interpretation, or model loading by hand --
closing the one real gap this milestone's workload exposed between what
Forge could already *train* and what it could previously *predict*.

## Suggested commit message

```
feat: add forge.predict_tabular_classification_artifact() and task="tabular_classification" metadata (M91 tabular classification workload)
```
