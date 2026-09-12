# M82 — Portable Artifact Inference Workflow: `forge.predict_artifact()`

## 1. Executive summary

M81 finished the *producer* side of the portable-artifact story:
`forge.train_and_save()` turns a `Module` + `Dataset` into a verified
`.forge` file in one call. M82's brief asked about the *consumer* side: a
developer who receives a `.forge` file still had to know, and correctly
sequence, seven separate framework calls (`load_model()`,
`load_preprocessing()`, `load_classes()`, decode the image, apply the
preprocessing, `predict()`, `interpret_classification()`) before getting a
usable prediction -- exactly the internal framework knowledge
`docs/product/vision.md`'s portable-artifact workflow is supposed to hide.

A targeted inspection (`forge/serialization/model.py`,
`forge/training/inference.py`, `forge/data/image_folder.py`,
`forge/cli/model.py`, `examples/image_folder_classification/infer.py`) found
that this exact seven-step sequence was already independently hand-written
**twice** -- once in `examples/image_folder_classification/infer.py::run()`
and once in `forge/cli/model.py::cmd_predict()` -- in the same order, with
the same "no classes -> fall back to a raw index instead of erroring"
behavior in both places. That is the same "two real consumers, same shape"
extraction bar `train_and_save()` itself was built against in M81.

M82 built `forge.training.predict_artifact(path, image, *, device=None)`
(`forge/training/inference.py`) -- a pure composition of the existing
lower-level functions, with no new artifact format, input abstraction, or
model-serving machinery -- and retrofitted both real consumers plus
`examples/image_folder_classification/train.py`'s own new-image inference
demo to delegate to it.

## 2. What was actually missing

`save_and_verify()`/`train_and_save()` (M78/M81) already solved "is the file
portable" for the *training* process. Nothing in the codebase solved "can a
process that only has the file get a prediction without also having the
training-time code" as a single call -- every place that answered this
question did so by re-deriving the same seven-step sequence by hand:

```python
preprocessing = load_preprocessing(path)
if preprocessing is None:
    raise PersistenceError(...)
model = load_model(path, device=device)
raw = ImageFolder._load_image(Path(image_path))
prepared = preprocessing(raw)
batch = prepared.reshape(1, *prepared.shape)
output = predict(model, batch)
classes = load_classes(path)
if classes is not None:
    return interpret_classification(output, classes)[0]
return int(np.argmax(output.numpy(), axis=1)[0])
```

verified by diffing `examples/image_folder_classification/infer.py::run()`
against `forge/cli/model.py::cmd_predict()` (pre-M82): identical body, modulo
`args.image`/`args.model` vs. `image_path`/`model_path` naming and how the
final result is printed.

## 3. What was implemented

### 3.1 `forge.training.predict_artifact()`

`forge/training/inference.py`, alongside `predict()`/`save_and_verify()`/
`interpret_classification()`:

```python
def predict_artifact(
    path: str,
    image: "str | os.PathLike",
    *,
    device: "str | Device | None" = None,
) -> "ClassificationPrediction | int":
    ...
```

Behavior, exactly matching the sequence above:

1. `image` must be a `str`/`os.PathLike` -- anything else raises
   `forge.DataError` immediately, before any file I/O.
2. `load_preprocessing(path)` -- `None` raises `forge.PersistenceError`
   naming the missing configuration (there is no automatic way to prepare an
   arbitrary new image otherwise; the same hidden-assumption failure mode
   M71 closed).
3. `load_model(path, device=device)`.
4. `ImageFolder._load_image(Path(image))` -- the same single-image decode
   step `ImageFolder.__getitem__` itself uses.
5. The reconstructed `preprocessing(raw)`, batched, through `predict()`.
6. `load_classes(path)`: present -> `interpret_classification(output,
   classes)[0]` (a `ClassificationPrediction`); absent -> the raw predicted
   class index as a plain `int`.

Step 6's "return an `int`, not an error" branch was a deliberate choice, not
an oversight -- see Section 4.2.

### 3.2 Real consumers retrofitted

- `examples/image_folder_classification/infer.py::run()` is now a one-line
  call to `forge.predict_artifact()`.
- `forge/cli/model.py::cmd_predict()` now calls `predict_artifact()` and
  formats its result -- the CLI and the Python API share one inference path,
  per the brief's own Section 9.
- `examples/image_folder_classification/train.py`'s Section 12 (the
  end-of-run demo on a brand-new image file at a resolution never trained
  at) now calls `forge.predict_artifact(model_path, new_image_path)`
  directly instead of manually reloading preprocessing and re-running
  `predict()`/`interpret_classification()`. Section 11 (a demo on an
  in-memory held-out test-split *sample*, not a file) is unchanged -- there
  is no file for `predict_artifact()` to point at in that case.

## 4. Why this implementation was chosen

### 4.1 Composition only

`predict_artifact()` adds no new validation, serialization, or inference
logic. Every condition it can raise is a condition one of `load_model()`,
`load_preprocessing()`, `load_classes()`, `ImageFolder._load_image()`, or
`interpret_classification()` already raises -- `predict_artifact()` only
sequences the calls and decides which of the last two outcomes (a
`ClassificationPrediction` or a bare `int`) applies.

### 4.2 No-classes returns an `int`, not an error

The brief's own Section 8 warns against "pretending" an artifact is a
classification artifact it is not. The chosen reading: fabricating a
placeholder label (`"class_0"`) for an artifact with no class vocabulary
would be the actual pretense; returning the real, structurally honest
information Forge has -- a class *index* with no name for it -- is not. This
also preserves the exact behavior both pre-existing consumers already had
and already had a passing test for
(`tests/test_classification_metadata.py::
test_cli_predict_without_classes_prints_index`) -- refactoring the CLI to
delegate to a new function that changed this specific behavior would have
been an undocumented, untested behavior break for no stated benefit.

### 4.3 Input scoped to a file path, not a generic input type

`image` accepts only `str`/`os.PathLike`. The brief's Section 4/5 explicitly
warn against a generic artifact-input abstraction; Forge's other example
workloads (tabular rows, raw audio/waveform arrays, stepwise-recurrence
token sequences) have no single canonical "new input file" convention the
way an image does, so there is nothing a wider input contract could
honestly support yet. A caller with an in-memory `Tensor` already has
`predict()` + `interpret_classification()` directly, unchanged.

## 5. Files changed

- `forge/training/inference.py` -- added `predict_artifact()`; updated the
  module docstring.
- `forge/training/__init__.py`, `forge/__init__.py` -- exported
  `predict_artifact` at both levels.
- `forge/cli/model.py` -- `cmd_predict()` now delegates to
  `predict_artifact()`; removed the now-duplicated image-decode/preprocess/
  predict/interpret logic and its now-unused imports.
- `examples/image_folder_classification/infer.py` -- `run()` now delegates
  to `forge.predict_artifact()`; removed its own copy of the same sequence.
- `examples/image_folder_classification/train.py` -- Section 12's new-image
  demo now calls `forge.predict_artifact()`; removed the manual
  `load_preprocessing()`/decode/predict/interpret sequence it replaced, and
  the now-unused `load_preprocessing` import.
- `tests/test_artifact_inference.py` (new), `tests/
  test_artifact_inference_cuda.py` (new).
- `docs/architecture/training-engine.md` -- new **Portable-artifact
  inference in one call** section; package-layout table and milestone
  header updated.
- `examples/image_folder_classification/README.md` -- updated to describe
  the `predict_artifact()`-based workflow.
- `docs/development/progress.md` -- new M82 entry.

No changes to `Tensor`/autograd, CUDA kernels, `Trainer`, `TrainingSession`,
optimizer infrastructure, or the checkpoint/model file format.

## 6. Tests added

`tests/test_artifact_inference.py` (13 tests, CPU): re-export identity;
returns a correct `ClassificationPrediction`; accepts a `Path` image
argument; no-classes returns a raw `int`; bit-for-bit equivalence against
the manual `load_model()`/`load_preprocessing()`/`load_classes()`/
`predict()`/`interpret_classification()` sequence it replaces; a
differently-sized image proves the persisted `Resize` is actually
exercised; explicit `device="cpu"` override; missing-preprocessing ->
`PersistenceError`; missing model file -> `PersistenceError`; missing image
file -> `DataError`; unsupported input types (`int`, a bare `Tensor`) ->
`DataError`; a corrupt image file -> `DataError`; and a genuine `subprocess`
fresh-process test (built from only pre-registered `forge.nn` types, since
the fresh process never imports this test module's own
`register_module()` call).

`tests/test_artifact_inference_cuda.py` (3 tests, hardware-gated): an
artifact saved from a CUDA model restores onto CUDA by default; an explicit
`device="cpu"` override converts it; and CPU-saved vs. CUDA-saved copies of
the identical weights agree within CUDA/CPU tolerance.

## 7. Full-suite result and CPU/CUDA verification

Full suite: **2,324 collected, 2,323 passed, 1 failed** (2,308 pre-M82 +
16 new: 13 CPU in `tests/test_artifact_inference.py`, 3 CUDA in `tests/
test_artifact_inference_cuda.py`). The one failure,
`tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, is the same
pre-existing CUDA-allocator-measurement flake documented since M63 --
reproduced passing cleanly in isolation immediately afterward
(`before: 320 -> after: 288` bytes in the full-suite run vs. a clean pass
standalone), confirming no M82 regression.

Every pre-existing test the refactor touches passed unmodified:
`tests/test_classification_metadata.py`'s CLI-predict tests (9/9, including
the no-classes-falls-back-to-index case), `tests/
test_image_folder_classification_integration.py`'s fresh-process `infer.py`
subprocess test, and `tests/test_image_folder_classification_cuda_
integration.py` (5/5) -- proving the CLI/`infer.py` retrofit is
behavior-preserving on both CPU and CUDA. The 3 CUDA tests in `tests/
test_artifact_inference_cuda.py` were run and passed directly against the
reference 940MX (`docs/development/development-environment.md`).

Verified end-to-end with a real run of
`examples/image_folder_classification/train.py`, `python -m forge model
predict`, and `python -m examples.image_folder_classification.infer`
against the same freshly-trained artifact, confirming all three print the
identical prediction.

## 8. Limitations

- Scoped to image-classification artifacts saved with `preprocessing=`
  (`classes=` optional) -- the one artifact shape Forge can currently fully
  describe end-to-end. Tabular/waveform/sequence-model artifacts, and the
  stepwise-recurrence models (`char_rnn`/`word_rnn`/`long_range_recall`),
  are not covered; there is no single canonical "new input" convention for
  them yet to build a `predict_artifact()`-equivalent against honestly.
- `image` accepts only a file path, not an already-decoded array/`Tensor` --
  a caller with in-memory pixel data uses `predict()` +
  `interpret_classification()` directly.
- No batch-of-images convenience (one image in, one result out) -- matching
  every existing single-image consumer (`infer.py`, `forge model predict`)
  this replaces.

## 9. Explicitly rejected/deferred

Per the brief's own Section 15: no generic artifact-input schema/task
registry, no model-serving/HTTP layer, no batch-serving infrastructure, no
automatic architecture/preprocessing inference. None of these were needed to
make the one real workflow (`.forge` file + one new image -> prediction)
genuinely one call.

## 10. What can an external Forge developer do now that they could not before M82?

Before: turning a `.forge` file into a prediction on a brand-new image
required knowing, and correctly ordering, seven separate framework calls,
including reaching into `ImageFolder._load_image` (a "protected"-by-
convention helper) to decode the image the same way training did.

After:
```python
result = forge.predict_artifact("model.forge", "new_photo.jpg")
print(f"Prediction: {result.label} ({result.confidence:.1%})")
```
one call, with `forge/cli/model.py`'s `predict` subcommand and
`examples/image_folder_classification/infer.py` both now built on the exact
same function -- proving it is the real inference path, not a parallel
convenience wrapper.

## 11. Suggested commit message

```
feat: add forge.predict_artifact(), one-call portable-artifact inference

Composes load_model()/load_preprocessing()/load_classes()/predict()/
interpret_classification() into a single call that turns a .forge file
and one new image file into a ClassificationPrediction (or a raw class
index, when no classes were saved). Refactors forge/cli/model.py's
`predict` subcommand and examples/image_folder_classification/infer.py
to delegate to it instead of independently hand-rolling the identical
sequence, and simplifies train.py's own new-image inference demo to use
it directly.
```
