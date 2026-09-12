# M83 — End-to-End Regression Workflow: `forge.predict_tensor_artifact()`

## 1. Executive summary

M82 proved Forge's high-level train -> save -> fresh-process -> predict
workflow for image classification. M83's brief asked whether that workflow
generalizes beyond classification, or whether Forge had merely built one
very polished classification path. The answer had to come from building a
second, materially different, complete workflow around `examples/
regression/` -- Forge's existing tabular-regression example -- not from
another readiness assessment.

A targeted inspection of `examples/regression/`, `forge/training/`, and
`forge/serialization/` found two real gaps, both on the persistence/
inference side (the training side -- `Dataset`/`DataLoader`/`Trainer`/
checkpoint/resume/reproducibility -- was already solid, per M60/M65):

1. **The example never used `forge.train_and_save()`.** Unlike `mnist`/
   `image_folder_classification` (M80/M81), `examples/regression/train.py`
   still built its `Trainer` via `start_training_session()` on *every* run,
   fresh or resumed, because M60 predates the fresh/resume split M80
   introduced and no later milestone had revisited it.
2. **The saved model never carried its preprocessing.** `dataset.py`'s
   `make_datasets()` fits a `Normalize(mean, std)` transform on the training
   split and bakes it into every split's `TensorDataset`, but `train.py`
   called `save_and_verify(model, path, query_x)` with no `preprocessing=` --
   the artifact recorded only the trained weights. A developer holding just
   `regression_model.forge` had no way to standardize a brand-new *raw*
   feature vector the same way training data was standardized; the
   framework's own `preprocessing=` mechanism (M71) already existed and
   already worked for this exact shape (`Normalize` is not image-specific --
   see Section 2), it just was never used here.

M83 fixed both, then added the one piece that did not yet exist anywhere:
`forge.training.predict_tensor_artifact(path, input_data, *, device=None)`
-- the non-classification counterpart to `predict_artifact()` (M82), for an
artifact whose input is a plain numeric array rather than an image file.

## 2. What was actually missing (and what was not)

`forge/data/transforms.py::Normalize` was already fully generic --
`(x - mean) * (1/std)`, broadcasting like any Tensor arithmetic, with no
image-specific assumption anywhere in it. `forge/serialization/model.py`'s
`preprocessing=`/`load_preprocessing()` mechanism (M71) is also fully
generic: it persists *any* registered `Transform`'s configuration, and
`Normalize` has been a registered transform since M71. **No persistence
format change was needed at all** -- the regression artifact shape was
already fully supported by existing infrastructure; it just was never
exercised for a non-image model.

What genuinely did not exist: a one-call way to turn `(path, raw_input)`
into a prediction for this shape. `predict_artifact()` (M82) is scoped,
deliberately and explicitly in its own docstring, to image files with
mandatory preprocessing and optional class interpretation -- none of which
fits a numeric input with optional preprocessing and no class concept.

## 3. What was implemented

### 3.1 `forge.training.predict_tensor_artifact()`

`forge/training/inference.py`, alongside `predict_artifact()`:

```python
def predict_tensor_artifact(
    path: str,
    input_data: "Tensor | np.ndarray | Sequence[Any]",
    *,
    device: "str | Device | None" = None,
) -> Tensor:
    ...
```

Behavior:

1. `input_data` must already be batched (a `Tensor`, or a NumPy array /
   nested list-or-tuple of numbers convertible to one via
   `Tensor(input_data)`) -- the same convention `forge.predict()` itself
   requires. Anything else raises `forge.DataError` immediately.
2. `load_preprocessing(path)` -- **optional**, unlike `predict_artifact()`:
   `None` is a real, valid state (not every numeric model needs
   standardization), so `input_data` is used unchanged when absent, and run
   through the reconstructed transform when present.
3. `load_model(path, device=device)`.
4. `predict(model, prepared)` -- the raw output `Tensor`, returned directly.
   No class vocabulary is ever consulted; there is no `classes=` analogue
   for a numeric result.

### 3.2 `examples/regression/dataset.py`

`make_datasets()`'s returned `stats` dict gained one new key, `"transform"`
-- the exact fitted `Normalize` instance already constructed internally and
baked into every split's `TensorDataset`. This is exposure, not new logic:
`train.py` now passes `stats["transform"]` straight to
`save_model(..., preprocessing=...)` instead of reconstructing an
equivalent `Normalize(mean=stats["mean"], std=stats["std"])` from the raw
arrays that were already there.

### 3.3 `examples/regression/train.py`

Split into the same `--resume`-or-fresh two-branch shape `mnist`/
`image_folder_classification` already have (M80/M81):

- **Fresh path**: builds `data_loader_rng` itself, trains through
  `forge.train_and_save(..., preprocessing=stats["transform"])`, and writes
  `data_loader_rng_state` into the checkpoint via plain
  `forge.save_checkpoint(..., extra=...)` (the same key
  `start_training_session()`'s resume path already reads) -- exactly
  `image_folder_classification/train.py`'s own M80 pattern.
- **`--resume` path**: unchanged in shape (`start_training_session()` +
  `Trainer.fit()`), except `save_and_verify()` now also receives
  `preprocessing=stats["transform"]`.
- Both branches still compute a separate held-out **test**-split evaluation
  after training/validation -- `regression`'s genuine three-way train/val/
  test split (unlike `mnist`/`image_folder_classification`'s two-way split,
  where validation and "final evaluation" are the same set) is a real
  methodological property of this example, preserved rather than collapsed
  to fit `train_and_save()`'s exact shape. The fresh branch gets this via a
  throwaway `Trainer(model, loss_fn, optimizer, ...).evaluate(test_loader)`
  after `train_and_save()` returns -- reusing `Trainer.evaluate()` unchanged,
  not new evaluation logic.
- The end-of-run demo now calls `forge.predict_tensor_artifact(model_path,
  new_raw_feature_vector)` on a brand-new *raw* sample (never standardized
  by this process), mirroring `image_folder_classification/train.py`'s own
  `predict_artifact()` demo on a brand-new image file.

## 4. Why this implementation was chosen

### 4.1 A separate function, not a generalized `predict_artifact()`

Considered and rejected: dispatching inside `predict_artifact()` on the type
of its second argument (a file path vs. a `Tensor`/array). Rejected because
the two contracts differ in more than input type -- `predict_artifact()`
makes preprocessing *mandatory* and always decodes a file; the numeric case
makes preprocessing *optional* and never touches a filesystem; the return
types are entirely different shapes (`ClassificationPrediction`/`int` vs. a
raw `Tensor`). Branching one function across two contracts this different
was judged to be exactly the "awkward task detection" the brief's Section 8
warns against, even though the dispatch key itself (argument type) is not a
stored task flag. A second, small, honestly-scoped function was the smaller,
clearer design -- Option B from the brief's own Section 8.

### 4.2 No batching convenience, no single-sample convention

Unlike a decoded image file (always exactly one sample, so
`predict_artifact()` adds the batch dimension itself), a numeric input has
no canonical "this is one unbatched sample" shape -- `(8,)` could mean one
8-feature sample or a batch of 8 scalars for a different model. Rather than
inventing a new, model-shape-dependent convention, `predict_tensor_artifact()`
requires the same "already batched" contract `forge.predict()` itself
already documents and requires. This is not a missing convenience; it is
avoiding a genuinely ambiguous one.

### 4.3 No format change; `classes` never fabricated

The saved-artifact format is untouched -- `preprocessing`/`classes` were
already independent, optional, sibling metadata entries (M71/M72);
`regression`'s artifacts simply now populate the first and correctly never
populate the second. `predict_tensor_artifact()` never consults
`load_classes()` at all -- fabricating a placeholder class interpretation
for a regression output would be precisely the "pretend every model is a
classification model" failure mode the brief's Section 4 warns against.

## 5. Files changed

- `forge/training/inference.py` -- added `predict_tensor_artifact()`;
  updated the module docstring.
- `forge/training/__init__.py`, `forge/__init__.py` -- exported
  `predict_tensor_artifact` at both levels; updated docstrings.
- `examples/regression/dataset.py` -- `make_datasets()`'s `stats` dict gained
  a `"transform"` key (the already-constructed `Normalize` instance).
- `examples/regression/train.py` -- split into fresh/`--resume` branches;
  fresh branch now trains via `forge.train_and_save(...,
  preprocessing=stats["transform"])`; `--resume` branch's `save_and_verify()`
  call now also passes `preprocessing=stats["transform"]`; new end-of-run
  `forge.predict_tensor_artifact()` demo on a brand-new raw feature vector.
- `tests/test_tensor_artifact_inference.py` (new) -- general contract tests
  for `predict_tensor_artifact()`, independent of the regression example.
- `tests/test_regression_artifact_workflow.py` (new) -- the real M83
  acceptance test: `examples.regression` dataset/model ->
  `forge.train_and_save()` -> artifact -> genuinely separate OS process ->
  `forge.predict_tensor_artifact()` -> numerical agreement.
- `docs/architecture/training-engine.md` -- new **Portable-artifact
  inference for numeric input** section; package-layout table and milestone
  header updated.
- `examples/regression/README.md`, `examples/README.md`, `README.md` --
  updated to describe the new artifact-level inference workflow.
- `docs/development/progress.md` -- new M83 entry.

No changes to `Tensor`/autograd, CUDA kernels, `Trainer`, `TrainingSession`,
optimizer infrastructure, or the persisted model-file format
(`FORMAT_VERSION` unchanged).

## 6. Tests added

`tests/test_tensor_artifact_inference.py` (13 tests, CPU): re-export
identity; returns a raw `Tensor` with no class interpretation; accepts
`Tensor`/`np.ndarray`/nested-list input equivalently; bit-for-bit
equivalence against the manual `load_model()`/`load_preprocessing()`/
`predict()` sequence it replaces; a saved artifact with **no** preprocessing
passes input through unchanged (the optional-preprocessing contract,
deliberately different from `predict_artifact()`'s mandatory one);
pre-save vs. post-load numerical equivalence; explicit `device="cpu"`
override; unsupported input types and a missing model file raise
`DataError`/`PersistenceError`; and a genuine `subprocess` fresh-process
test (built from pre-registered `forge.nn`/`Normalize` types only).

`tests/test_regression_artifact_workflow.py` (4 tests, CPU): the fresh
(non-`--resume`) path's artifact carries persisted `Normalize` preprocessing
and no fabricated `classes`; a genuinely separate `subprocess` process,
given only the trained `.forge` file, reproduces this process's own
`predict_tensor_artifact()` prediction on an identical brand-new raw
feature vector to 8 decimal places; the `--resume` path also persists
preprocessing (not just the fresh path); and a hand-computed reference
(raw input standardized with `make_datasets()`'s own recorded mean/std, fed
directly through the reloaded model) matches `predict_tensor_artifact()`'s
result within `atol=1e-5`.

## 7. Full-suite result

Full suite: **2,338 collected, 2,337 passed, 1 failed** (2,308 pre-M83 + 4
retrofitted-but-otherwise-unmodified regression test files still passing +
17 new: 13 in `tests/test_tensor_artifact_inference.py`, 4 in `tests/
test_regression_artifact_workflow.py`). The one failure, `tests/
test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, is the same
pre-existing CUDA-allocator-measurement flake documented since M63 --
reproduced passing cleanly in isolation immediately afterward, confirming no
M83 regression.

Every pre-existing regression test passed unmodified after the `train.py`
retrofit: `tests/test_regression_example_integration.py` (16/16, including
resume-equivalence and model-persistence tests), `tests/
test_regression_reproducible_training.py` (5/5, including the
`shuffle=True` full-vs-resumed bitwise-parameter-equivalence test -- the
exact guarantee M65 introduced and M80/M81 already proved survives this
fresh/resume split for `image_folder_classification`), and `tests/
test_regression_example_cuda_integration.py` (6/6, hardware-verified on the
reference 940MX).

## 8. Real regression training verification

```bash
python -m examples.regression.train --n-train 200 --n-val 40 --n-test 40 --epochs 3 --device cpu
```
trained, saved a checkpoint + verified model + run record, then printed:
```
New raw feature vector [...], preprocessing reconstructed from '.../regression_model.forge':
Prediction: 0.0504
```
-- confirming the artifact-level demo runs end-to-end on a real training
run, not just inside a test. A subsequent `--resume` invocation against the
same checkpoint also completed and re-verified its (now preprocessing-
carrying) artifact.

## 9. Fresh-process inference verification

`tests/test_regression_artifact_workflow.py::
test_predict_tensor_artifact_agrees_with_a_genuinely_separate_process`
trains a real regression model via `examples.regression.train.main()`
(exercising the actual `forge.train_and_save()` fresh path), then launches
`subprocess.run([sys.executable, "-c", script], ...)` where `script` only
`import`s `forge`/`numpy` and calls `forge.predict_tensor_artifact()` --
never importing `examples.regression` or this test module, so no in-memory
training state or custom registration is available to it. The subprocess's
printed prediction is compared against this process's own
`predict_tensor_artifact()` call on the identical raw input.

## 10. Numerical equivalence verification

Three independent checks, all passing:
- Pre-save (still-in-memory model) vs. post-load
  (`predict_tensor_artifact()`) prediction on the same input:
  `atol=1e-5` (`tests/test_tensor_artifact_inference.py::
  test_predict_tensor_artifact_prediction_equals_pre_save_prediction`) --
  the tolerance matches `save_and_verify()`'s own default `atol` for the
  identical float32-CPU-round-trip class of comparison.
- Fresh-process vs. same-process prediction on the identical raw input,
  formatted to 8 significant decimal digits (float32 precision, fully
  deterministic inference with no dropout/randomness) -- exact string match
  required, the tightest check performed.
- Hand-computed reference (`(raw - mean) / std` via plain NumPy, fed
  directly through `load_model()`'s reloaded module) vs.
  `predict_tensor_artifact()`'s result: `atol=1e-5` (`tests/
  test_regression_artifact_workflow.py::
  test_predict_tensor_artifact_standardizes_raw_input_the_same_way_training_did`).

All three tolerances are float32-CPU round-trip tolerances (matching
`save_and_verify()`'s own precedent), not arbitrarily loosened to make a
test pass.

## 11. CUDA verification

No CUDA-specific code was added -- `predict_tensor_artifact()` composes
`load_model()`/`load_preprocessing()`/`predict()` unchanged, all of which
are already CUDA-tested. `examples/regression/train.py`'s CUDA path
(`tests/test_regression_example_cuda_integration.py`, 6/6) was re-verified
on the reference 940MX after the fresh/resume split, confirming the
retrofit is behavior-preserving on CUDA too. `predict_tensor_artifact()`
itself was not additionally exercised against CUDA in this milestone (no
CUDA-specific behavior to verify beyond what `load_model()`/`predict()`
already cover) -- `device="cuda"` is accepted and would dispatch through the
same paths `predict_artifact()`'s own CUDA tests already cover.

## 12. Limitations

- `predict_tensor_artifact()` requires an already-batched input, matching
  `predict()`'s own contract -- there is no "add a batch dimension for a
  single raw sample" convenience (see Section 4.2 for why this was not
  built).
- No CLI command was added (`forge model predict --image ...` is
  intentionally image-specific; a generic `--input` representation is a
  separate product problem the brief explicitly excluded).
- Scoped, like `predict_artifact()`, to models whose calling convention is
  a single `model(x)` forward pass on a batched `Tensor` -- the
  stepwise-recurrence models (`char_rnn`/`word_rnn`/`long_range_recall`)
  are not covered by either function.

## 13. Explicitly rejected/deferred

Per the brief's own Section 17: no generic ML task abstraction, no
universal artifact schema, no tabular-data framework/pandas dependency, no
automatic feature scaling/model/loss selection, no generic task detection
or CLI input protocol, no model serving/HTTP/batch-serving infrastructure.
None of these were needed to make the one real workflow (`.forge` file +
one new raw feature vector -> numeric prediction) genuinely one call.
Section 4.1 above records the one design alternative (generalizing
`predict_artifact()`) considered and rejected in favor of a separate
function.

## 14. What can an external Forge developer do now that they could not before M83?

Before: an external regression developer could train through
`start_training_session()`/`Trainer`, but a saved `regression_model.forge`
carried no preprocessing -- a fresh process could reload the model but had
no way to know how to standardize a brand-new raw feature vector before
handing it to the model, short of re-deriving the training split's own
mean/std by hand.

After:
```python
prediction = forge.predict_tensor_artifact("price_model.forge", input_batch)
```
one call, exercised by `examples/regression/train.py`'s own end-of-run demo
and proven portable by a genuinely separate OS process in
`tests/test_regression_artifact_workflow.py` -- the same "train through the
high-level API, save a verified artifact, move it to a fresh process, get a
useful prediction with no manual `load_model()`/`predict()` reconstruction"
guarantee `image_folder_classification` established for classification in
M82, now also true for a materially different (non-classification, numeric)
workload.

## 15. Suggested commit message

```
feat: add forge.predict_tensor_artifact(), one-call numeric-artifact inference

Composes load_model()/load_preprocessing()/predict() into a single call
that turns a .forge file and one already-batched numeric input directly
into a raw prediction Tensor, with preprocessing optional and no class-
vocabulary concept -- the non-classification counterpart to
predict_artifact() (M82). Retrofits examples/regression/train.py to save
its fitted Normalize transform as preprocessing= (via a fresh/--resume
split mirroring mnist/image_folder_classification's own M80/M81 pattern)
and to demonstrate predict_tensor_artifact() on a brand-new raw feature
vector, closing the one real gap between Forge's classification and
regression high-level workflows.
```
