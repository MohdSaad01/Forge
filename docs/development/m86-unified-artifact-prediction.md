# M86 — Unified Portable Artifact Prediction: `forge.predict_model()`

## 1. Executive summary

M82-84 gave Forge three task-specific, one-call inference functions over a
saved `.forge` artifact -- `predict_artifact()` (image classification),
`predict_tensor_artifact()` (numeric regression), `predict_image_artifact()`
(image-to-image/segmentation) -- and M85 gave a developer holding just the
file a way to answer "what *is* this artifact?" (`inspect_model()`). What
remained was the step in between: a developer still had to read
`inspect_model()`'s result themselves, or already know out of band, which
of the three functions to call.

Built `forge.predict_model(path, input_data, *, device=None)`
(`forge/training/inference.py`): it calls `inspect_model()`, determines
which of the three workflows the artifact's own persisted metadata
describes, and delegates to the matching function **unchanged**, returning
its result unchanged. No new artifact format, task registry, or
input-conversion machinery -- exactly the "thin dispatcher" shape the brief
asked for.

## 2. Final public API

```python
result = forge.predict_model("model.forge", input_data, device=None)
```

- `path` -- a `.forge` file saved by `forge.save_model()`.
- `input_data` -- whatever the *selected* workflow's own function expects:
  a `str`/`os.PathLike` image path (classification, segmentation), or a
  `Tensor`/NumPy array/nested list (regression). Validated exactly as
  strictly as calling that function directly.
- `device` -- passed through unchanged; same default as the other three
  functions (the device recorded in the archive).
- Returns exactly what the selected function would have returned:
  `ClassificationPrediction | int` (classification), `Tensor` (regression
  or segmentation).
- Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
  artifact, or when the artifact's own metadata does not reliably identify
  one of the three supported workflows.

## 3. How artifact workflow detection works

A new private helper, `_determine_workflow(info: ModelInfo) -> str`
(`forge/training/inference.py`), reads only fields `inspect_model()`
(Milestone 85) already exposes:

1. `info.classes is not None` -> `"classification"`.
2. `info.classes is None`:
   - `"Linear" in info.model.module_types` -> `"regression"`.
   - `"Linear" not in info.model.module_types` and
     `"Conv2d" in info.model.module_types` -> `"segmentation"`.
   - Neither -> raises `forge.PersistenceError`, naming what metadata was
     available and which workflows `predict_model()` supports.

`predict_model()` itself is four lines: call `inspect_model()`, call
`_determine_workflow()`, call the matching function.

## 4. Why the chosen detection signal is reliable

**`classes` presence for classification.** `save_model()`
(`forge/serialization/model.py`) never populates the `"classes"` metadata
entry for anything but a classification model -- it is written only when a
caller explicitly passes `classes=...`, and every classification example in
this repo (`mnist`, `image_folder_classification`) always does. This is not
an inferred signal; it is the artifact author's own explicit statement of
intent, the same one `predict_artifact()` itself already treats as
authoritative for "does this artifact have a class vocabulary."

**`module_types` for regression vs. segmentation, once `classes` is absent.**
Verified directly against the real architectures, not assumed:

- `examples/regression/model.py`: `Linear(8, 64) -> ReLU -> Linear(64, 32)
  -> ReLU -> Linear(32, 1)` -- `Linear`-only, no `Conv2d` anywhere.
- `examples/segmentation/model.py`: `Conv2d(3, 16, ...) -> ReLU ->
  MaxPool2d(2) -> Conv2d(16, 32, ...) -> ReLU -> UpsampleNearest2d ->
  Conv2d(32, 1, ...)` -- fully convolutional, **no `Linear` layer at all**.
- `examples/image_folder_classification/model.py` (for contrast, though
  this path is reached via `classes` first): `Conv2d -> ... -> Flatten ->
  Linear -> ... -> Linear` -- `Conv2d` **and** `Linear` both present.

This is not incidental: a dense, per-pixel prediction head (segmentation)
must preserve spatial structure end-to-end, so it has no reason to ever
flatten into a fixed-size fully-connected layer; a classification or
regression head produces a fixed-size output vector (class logits, or a
scalar/vector prediction), which is exactly what a `Linear` layer is for.
`"Linear" in module_types` is therefore a real architectural signature, not
a coincidence of today's three example files.

**`ModelSummary.type` (the root type) was deliberately not used.** Every
model in every current Forge example is built via `Sequential`, so
`info.model.type` is `"Sequential"` in all three cases and carries zero
discriminating power by itself -- this is exactly what the brief's "do not
assume model type alone is sufficient" warning refers to, and it was
confirmed, not just assumed, by inspecting every example's `model.py`.

**No weight introspection, no trial forward pass, no tensor-dimension
guessing.** Every signal used (`classes`, `module_types`) was already a
public, stable `ModelInfo` field before M86 (Milestone 85) -- `_determine_
workflow()` adds no new persisted data and reconstructs no live `Module`.

## 5. Documented limitation: `classes=None` classification

A classification model saved with `classes=None` is a real, valid state
(`predict_artifact()`'s own docstring: "a classification model with no name
for its outputs... is a real, valid artifact state") -- but it is still
`Linear`-terminated, exactly like a regression model, and has no `classes`
left to disambiguate it from one. `_determine_workflow()` misidentifies it
as `"regression"`, and calling `predict_model()` on such an artifact with an
image path raises `forge.DataError` (from `predict_tensor_artifact()`
rejecting a non-numeric input) rather than classifying the image.

No current Forge example's own `train.py` produces this state -- every
classification example always saves `classes=`. It **is**, however, a real,
already-tested state: `tests/test_classification_metadata.py::
test_cli_predict_without_classes_prints_index` builds exactly this artifact
and exercises it through the CLI's direct `predict_artifact()` call, which
is unaffected. `tests/test_unified_artifact_prediction.py::
test_predict_model_documented_limitation_classification_without_classes_misdispatches`
pins this behavior down explicitly, so a future change to the dispatch
signal surfaces here rather than as a silent regression.

Resolving this would need a persisted task flag `save_model()` does not
write today (see Section 12, Rejected alternatives) -- not added
speculatively for a state no real training script currently produces.

## 6. Which existing prediction APIs are delegated to

`predict_model()` calls exactly one of, unchanged:

- `forge.training.predict_artifact()` (Milestone 82)
- `forge.training.predict_tensor_artifact()` (Milestone 83)
- `forge.training.predict_image_artifact()` (Milestone 84)

No internals of any of the three are duplicated or modified.

## 7. Real consumer(s)

Three real, independently-trained artifact shapes now go through
`forge.predict_model()` instead of a caller naming the task-specific
function directly:

- `examples/image_folder_classification/infer.py::run()` -- classification,
  `classes` always present.
- `examples/segmentation/infer.py::run()` -- segmentation, `Conv2d`-only,
  no `Linear`.
- `examples/regression/train.py`'s own end-of-run demo -- regression,
  `Linear`-only.

All three were retrained end-to-end with small smoke configurations
(`--epochs 1`-`2`, tens of samples) after the retrofit and produced correct
predictions through the new call path (see Section 9).

`forge model predict` (the CLI) was evaluated and **deliberately not**
retrofitted -- see Section 12.

## 8. Files changed

- `forge/training/inference.py` -- added `_determine_workflow()` and
  `predict_model()`; updated the module docstring.
- `forge/training/__init__.py`, `forge/__init__.py` -- exported
  `predict_model` at both levels; updated docstrings.
- `examples/image_folder_classification/infer.py` -- `run()` now calls
  `forge.predict_model()`.
- `examples/segmentation/infer.py` -- `run()` now calls
  `forge.predict_model()`.
- `examples/regression/train.py` -- end-of-run demo now calls
  `forge.predict_model()`.
- `forge/cli/model.py` -- docstring only, recording that the retrofit was
  evaluated and rejected; `cmd_predict()` itself is unchanged.
- `docs/architecture/training-engine.md` -- new **Unified portable-artifact
  prediction: `predict_model()` (Milestone 86)** section; milestone list in
  the file's own title; package-layout listing.
- `docs/architecture/persistence.md` -- one-paragraph forward pointer in
  the **Model inspection (Milestone 85)** section.
- `docs/development/progress.md` -- new M86 entry.
- `docs/development/m86-unified-artifact-prediction.md` -- this report.
- `tests/test_unified_artifact_prediction.py` (new, 15 tests, CPU).
- `tests/test_unified_artifact_prediction_cuda.py` (new, 4 tests, CUDA,
  hardware-verified).

## 9. Architecture impact

None to `Tensor`, autograd, CUDA kernels, `Trainer`, `TrainingSession`,
`DataLoader`, or the model serialization format. `predict_model()` is a
pure composition over `inspect_model()` + the three existing `predict_*`
functions, living entirely inside `forge/training/inference.py`. No new
persisted metadata, no `FORMAT_VERSION` bump, no task registry, no serving
infrastructure.

## 10. Tests added

`tests/test_unified_artifact_prediction.py` (CPU, 15 tests):
- Re-export consistency (`forge.predict_model is forge.training.predict_model
  is` the direct function).
- Dispatch: classification, regression, segmentation artifacts each route
  correctly.
- Delegation correctness: `predict_model()`'s result matches the
  corresponding direct `predict_artifact()`/`predict_tensor_artifact()`/
  `predict_image_artifact()` call bit-for-bit, for all three shapes.
- The documented `classes=None` limitation, pinned down explicitly.
- Input validation: a `Tensor` rejected for a classification artifact, an
  image path rejected for a regression artifact, a `Tensor` rejected for a
  segmentation artifact -- each via the delegated function's own existing
  `DataError`.
- Unsupported artifacts: an architecture with neither `Linear` nor `Conv2d`
  and no `classes` raises `PersistenceError` naming what was available;
  a missing file still raises `PersistenceError`.
- Backward compatibility: all three existing functions called directly,
  confirming unchanged behavior.
- Fresh process: one `subprocess.run([sys.executable, "-c", ...])` call
  predicts through all three artifact shapes in a single separate process,
  compared bit-for-bit / index-for-index against this process's own calls.

`tests/test_unified_artifact_prediction_cuda.py` (CUDA, 4 tests,
hardware-verified on the 940MX): CUDA-saved-artifact restore-onto-CUDA-by
-default for each of the three shapes, plus a CPU/CUDA agreement check
across all three.

## 11. Full-suite result

`python -m pytest tests/`: **2,409 collected, 2,408 passed, 1 failed**
(2,390 pre-M86 + 19 new). The one failure is
`test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`
-- the same pre-existing CUDA allocator-measurement flake documented since
M63 (see `[[forge-allocator-deadlock-bug]]`-adjacent memory and every prior
milestone's own full-suite report); re-run in isolation immediately after,
it passed cleanly (`1 passed in 0.99s`), confirming it is unrelated to M86.
Every pre-existing artifact-inference and CLI test
(`test_artifact_inference.py`, `test_tensor_artifact_inference.py`,
`test_segmentation_artifact_workflow.py`, `test_classification_metadata.py`,
`test_model_inspection.py`) and every retrofitted example's integration
test (`test_image_folder_classification_integration.py`,
`test_segmentation_artifact_workflow.py`, `test_regression_artifact_workflow.py`)
re-ran green, unchanged.

## 12. Fresh-process verification

`tests/test_unified_artifact_prediction.py::
test_predict_model_works_from_a_genuinely_separate_process_for_all_three_artifact_shapes`
saves a classification, a regression, and a segmentation artifact to three
separate files, launches one real `subprocess.run([sys.executable, "-c",
...])` process that imports only `forge`/`numpy` and calls
`forge.predict_model()` on each of the three, and compares every result
against this test process's own `predict_model()` call on the same files --
bit-for-bit for the regression `Tensor`, index/label-for-index/label for
the classification result, shape-for-shape for the segmentation mask.

## 13. CUDA verification

`tests/test_unified_artifact_prediction_cuda.py` (4 tests) ran directly
against the real 940MX (CUDA 12.6): a classification, a regression, and a
segmentation artifact each saved from a CUDA-resident model restore onto
CUDA by default through `predict_model()`, and a fourth test confirms
CPU-saved and CUDA-saved artifacts of the identical weights agree (exactly,
for the deterministic-threshold segmentation mask and classification
index/label; within `rtol=atol=1e-4` for the classification confidence and
regression value, the same tolerance `test_artifact_inference_cuda.py`/
`test_segmentation_artifact_workflow_cuda.py` already established).

## 14. Limitations

- **`classes=None` classification artifacts are misidentified as
  regression** -- see Section 5. Documented, tested, and does not affect
  `predict_artifact()` called directly.
- **No fourth workflow.** An artifact whose architecture has neither a
  `Linear` nor a `Conv2d` layer (or any future task Forge does not yet have
  a `predict_*_artifact()` function for) is explicitly unsupported --
  `predict_model()` raises rather than guessing. This is by design, not an
  oversight: the brief's own "reliable dispatch beats clever inference"
  principle.
- **No input-shape validation beyond what the delegated function already
  does.** `predict_model()` performs no independent input checking; a
  caller relying on it for input validation gets exactly `predict_artifact()`
  /`predict_tensor_artifact()`/`predict_image_artifact()`'s own existing
  checks, no more, no less.
- **`forge model predict` still requires knowing the artifact is a
  classification one** -- unchanged from before M86 (see Section 12,
  Rejected alternatives, and Section 5).

## 15. Rejected/deferred alternatives

- **A generic task-registry/plugin system for dispatch.** Rejected: three
  known, already-implemented workflows need no registry; a fourth workflow
  would need its own persisted-signal design work regardless of whether a
  registry exists, so a registry buys nothing today.
- **Dispatching primarily on `input_data`'s type** (`str` -> classification,
  `Tensor` -> regression). Rejected for the same reason M83's own report
  rejected it inside `predict_artifact()`: the artifact's own metadata, not
  the caller's input, determines which workflow applies; input type is only
  ever used (by the delegated function) to validate compatibility with the
  workflow the artifact's metadata already selected.
- **Introspecting parameter values, or running a trial forward pass, to
  infer task type from output shape.** Rejected: `inspect_model()`
  deliberately never reconstructs a live `Module` or touches weights
  (Milestone 85's whole point), and doing so here would reintroduce
  exactly the CUDA/module-registry requirements that function was built to
  avoid.
- **Retrofitting `forge model predict` onto `predict_model()`.** Evaluated
  directly -- routing this command's existing `--image` input through
  `predict_model()` would silently break `tests/
  test_classification_metadata.py::test_cli_predict_without_classes_prints_index`,
  a real, already-passing test of a real, valid artifact state (Section 5).
  The command's own `--image`-only contract was never ambiguous about which
  workflow applies, so there was no real gain to weigh against that cost.
  Deferred, not abandoned: if `forge model predict` ever needs to support
  non-classification artifacts, it needs its own output-handling redesign
  (printing a `Tensor` mask or numeric value, not just
  `ClassificationPrediction`/`int`) independent of this question.
- **A persisted `task=` metadata flag in `save_model()`.** Would fully
  resolve Section 5's limitation, but is a real, separate persistence-format
  decision (a new `FORMAT_VERSION`-relevant metadata entry every existing
  artifact would lack) that no real consumer has asked for yet -- deferred
  as an evidence-based candidate, not spec'd out speculatively.

## 16. What an external Forge developer can now do

Before M86: hold a `.forge` file, call `forge.inspect_model()`, read
`.classes`/`.preprocessing`/`.model.module_types`, reason about which of
`predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`
applies, then call it.

After M86:

```python
result = forge.predict_model("trained_model.forge", input_data)
```

with no prior knowledge of whether `trained_model.forge` is a
classification, regression, or segmentation artifact -- Forge determines
this from the artifact's own persisted metadata and returns the same useful
result the correct task-specific function would have. An artifact
`predict_model()` cannot reliably classify still fails with a clear,
actionable error rather than a silently wrong prediction.

## 17. Suggested commit message

```
feat: add forge.predict_model(), unified dispatch over the three portable-artifact prediction workflows
```
