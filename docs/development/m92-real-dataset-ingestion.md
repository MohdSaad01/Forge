# Milestone 92 — Real Dataset Ingestion and Portable Tabular Workflow

## Objective

M91 proved Forge's tabular-classification workflow end-to-end on a
**synthetic** dataset generated in-process. M92 asks whether that same
workflow survives contact with an ordinary real-world tabular file an
external developer would actually download -- following the brief's
explicit rule: attempt the real workload with existing public APIs first,
document actual friction, then implement only the smallest capability the
friction demonstrates. Do not start by building CSV support, pandas
integration, or a missing-value framework because those "sound useful."

## Selected dataset

The [Pima Indians Diabetes dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
(NIDDK/UCI-sourced, a widely-used real benchmark, not synthetic or a toy
set): 768 real patients, 8 numeric diagnostic measurements
(`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
`BMI`, `DiabetesPedigreeFunction`, `Age`), one binary target (`Outcome`:
1 = diabetes diagnosed within 5 years, 0 = not).

## Why this dataset

- **Numerical features, meaningful target, nontrivial size**: 768 rows,
  8 continuous features, real class imbalance (500 negative / 268
  positive -- 65.1%/34.9%), matching the brief's Section 2 preferences.
- **A genuine, well-documented real-world preprocessing requirement**:
  five features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
  `BMI`) cannot legitimately be `0` in a living patient, but the source
  data encodes "not measured" as literal `0` rather than a blank field --
  a real, commonly-cited data-quality issue with this exact dataset, not
  a constructed edge case. Measured directly from the bundled CSV:

  | column | zeros | % of 768 rows |
  |---|---|---|
  | Glucose | 5 | 0.7% |
  | BloodPressure | 35 | 4.6% |
  | SkinThickness | 227 | 29.6% |
  | Insulin | 374 | 48.7% |
  | BMI | 11 | 1.4% |

  (`Pregnancies == 0` is a real, valid value and was left alone;
  `DiabetesPedigreeFunction`/`Age` have no zeros.)
- **No categorical features, no genuinely missing (blank/NaN) fields** --
  confirms the brief's Section 11/10 "do not build categorical encoding /
  a general missing-value framework unless demonstrated" guidance from the
  other direction: this dataset's real missing-value shape is narrower
  (sentinel-coded, specific columns) than either of those speculative
  features would address, and needed neither.
- No cloud infrastructure, no runtime network dependency: the CSV
  (`https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`)
  was downloaded once during development and committed as
  `examples/tabular_diabetes/data/diabetes.csv` -- the example needs no
  network access to run (brief Section 3).

## External developer goal

Train a real tabular classifier end-to-end using only Forge's public API,
save a portable artifact, and predict on a brand-new raw patient row in a
fresh process -- the same goal M91 validated for synthetic data, now
against a file with real-world data-quality problems.

## Baseline: attempted with existing public APIs, no Forge changes

```text
diabetes.csv -> csv.reader + NumPy (stdlib, ~10 lines)
    -> TensorDataset -> random_split -> Normalize (fit on train)
    -> DataLoader -> MLP -> CrossEntropyLoss -> Adam
    -> forge.train_and_save(task="tabular_classification")
    -> forge.predict_model()
```

Every step ran to completion with **zero Forge changes**. Concretely
measured (throwaway script, not part of the final example):

- CSV ingestion: `csv.reader` + `np.array(rows, dtype=np.float32)` -- no
  friction at all. `np.genfromtxt`/stdlib `csv` already make "raw CSV ->
  NumPy array" trivial; this confirms the brief's Section 4 expectation
  that manual CSV parsing is often reasonable, not a real blocker.
- Splitting: `random_split` worked correctly (row order is not
  class-sorted, unlike M91's synthetic dataset, but `random_split` remains
  the safer default with an explicit representativeness guarantee).
- Training: `Sequential`/`CrossEntropyLoss`/`Adam`/`forge.train_and_save()`
  all worked unmodified. Baseline (zeros passed straight through
  `Normalize`, no imputation): **71.0% test accuracy** vs. a 65.1%
  majority-class baseline.
- Fresh-process inference: `forge.predict_model()` on a raw row with a
  real sentinel-zero `Insulin` reading did **not** crash or raise. It
  produced a confident, plausible-looking prediction.

**Conclusion so far**: Forge's existing public data/training/persistence
API is already sufficient to *run* this real dataset end-to-end. No
framework change is justified by "it doesn't work."

## The real problem: silently wrong, not broken

Manually imputing the sentinel zeros in raw NumPy *before* Forge ever sees
the array (training-split median per affected column) also runs fine, and
measurably improves accuracy: **72.9% vs. 71.0%** test accuracy (+1.9pp,
same seed/split/epochs) in the baseline comparison -- a real, positive
signal that correct handling of this dataset's real preprocessing
requirement matters, not just a hypothetical concern.

But that manual imputation step has **no representation in
`forge.data.Normalize`** (an affine `(x - mean) / std`, nothing
conditional) or in any other registered preprocessing transform
(`forge/serialization/transforms.py`: `Resize`, `Compose`, `Normalize`,
`ToTensor`, `Reshape`, `Flatten` -- an exhaustive list before this
milestone). So the moment a model trained this way is saved with
`preprocessing=Normalize(...)`, the imputation step is **silently lost**:
the artifact only remembers "subtract mean, divide by std." A fresh raw
inference row's sentinel zero is standardized as if it were a real,
extreme-low measurement, not "not measured."

### Reproduced directly against a real trained artifact

Training a real model with manually-imputed data, saving `preprocessing=
Normalize(mean, std)` (fit on the imputed training data), then predicting
on the **same raw, un-imputed row** two different ways:

```python
raw_row = [[2, 130.0, 70.0, 0.0, 0.0, 28.5, 0.5, 35.0]]  # SkinThickness/Insulin both genuinely missing (0)

forge.predict_model(model_path, raw_row)[0]
# -> no_diabetes (100.0% confidence)   -- sentinel zeros standardized as literal low readings

manually_imputed_row = [[2, 130.0, 70.0, 29.0, 125.0, 28.5, 0.5, 35.0]]  # dev hand-copies training medians
forge.predict_model(model_path, manually_imputed_row)[0]
# -> diabetes (70.2% confidence)   -- what training actually assumed for a missing value
```

**The predicted class flips**, purely based on whether the developer
happens to still have the exact training-time imputation constants lying
around -- constants that are not recoverable from the `.forge` file at
all. This is exactly the brief's Section 8 "preprocessing portability"
failure mode: "the external developer should not have to manually
recreate training preprocessing," and here they'd have no way to even if
they wanted to.

## Root cause

Preprocessing persistence (`forge/serialization/transforms.py`, Milestone
71) only covers transforms whose entire behavior is already flat,
JSON-safe constructor arguments. `Normalize` is deliberately affine-only;
there was no transform capable of expressing "conditionally replace a
specific value in a specific column with a fixed constant," so an
imputation step -- however it was written -- had no path to travel with the
artifact.

## Problem selected and why

This is the highest-value, and only real, actionable finding from the
baseline: CSV ingestion needed no fix (already trivial); splitting needed
no fix (`random_split` already correct); there are no categorical features
and no genuinely-blank fields (nothing to build there); the CLI/inference
input shape is unchanged from M91 (a flat numeric row, already supported).
Preprocessing-portability for a real, demonstrated, *measured* accuracy-
relevant preprocessing step is the one gap with concrete evidence: a
21.9-percentage-point observed confidence swing (100% -> 70.2%, on a class
flip) traced to exactly one missing capability.

## Implementation

### `forge.data.ReplaceValue` (`forge/data/transforms.py`)

A new `Transform`: replaces a sentinel value with a fixed per-column fill
value, on explicit column indices only (never a blanket "replace this
value everywhere" -- `Pregnancies == 0` is valid data, so replacement must
stay opt-in per column). Constructor: `ReplaceValue(sentinel, columns,
fill)`, where `columns`/`fill` are parallel lists. Operates along the last
axis of its input, so the same instance works whether it is applied to a
single unbatched `(F,)` sample (`TensorDataset.__getitem__`'s calling
convention) or an already-batched `(N, F)` array
(`predict_tabular_classification_artifact()`'s calling convention on
`preprocessing(prepared)`) -- exactly mirroring how `Normalize`'s own
broadcast already works in both shapes. This dual-shape requirement was
verified by reading both call sites, not assumed.

### Registration (`forge/serialization/transforms.py`)

`register_transform("ReplaceValue", ReplaceValue, get_config=...)` --
config is `{"sentinel": float, "columns": [int, ...], "fill": [float,
...]}`, all JSON-safe scalars, reconstructed via the default `cls(**config)`
path (no custom `from_config` needed, unlike `Reshape`'s positional-varargs
case). `Compose([ReplaceValue(...), Normalize(...)])` round-trips
recursively for free through `Compose`'s existing serialization -- no
changes needed there.

### What was deliberately *not* built

- No general missing-value/imputation framework: `ReplaceValue` has no
  strategy selection (mean/median/mode), no `NaN` handling, no automatic
  column detection -- it does exactly "replace this value, in these
  columns, with these constants," fit externally by the caller (in
  `dataset.py::make_datasets()`, from training-split statistics), the same
  division of responsibility `Normalize` already has (Forge fits nothing
  itself; callers fit and pass constants).
- No feature-schema/column-name metadata: a fixed, documented column order
  (`FEATURE_NAMES` in `dataset.py`) was sufficient, matching M91's
  precedent -- not demonstrated as a blocker for this workload.
- No categorical encoding, no CSV/pandas framework, no new artifact task
  type, no CLI flag changes: none demonstrated as necessary. `task=
  "tabular_classification"` (M91) already fully describes this workload's
  inference contract; `forge model predict`'s existing JSON-numeric-input
  parsing already handles a raw feature row unchanged.

## Example: `examples/tabular_diabetes/`

`dataset.py` (real CSV ingestion + two-stage preprocessing fit: `
ReplaceValue` fill values from training-split non-zero medians, then
`Normalize` mean/std from the imputed training split), `model.py` (smaller
MLP than `tabular_classification`'s, given the smaller real dataset),
`train.py` (`forge.train_and_save(..., preprocessing=Compose([ReplaceValue,
Normalize]), classes=CLASS_NAMES, task="tabular_classification")`, no
`--resume`/checkpoint -- training finishes in seconds), `infer.py`
(standalone fresh-process script, mirrors `tabular_classification/infer.py`
exactly), `README.md`, `data/diabetes.csv` (committed).

## Files changed

- `forge/data/transforms.py` -- new `ReplaceValue` class; `__all__` updated.
- `forge/data/__init__.py` -- re-export `ReplaceValue`.
- `forge/serialization/transforms.py` -- register `ReplaceValue`; docstring
  updated.
- `examples/tabular_diabetes/` -- new (`__init__.py`, `dataset.py`,
  `model.py`, `train.py`, `infer.py`, `README.md`, `data/diabetes.csv`).
- `examples/README.md` -- new table row.
- `.gitignore` -- `examples/tabular_diabetes/artifacts*/` entry (mirroring
  every other example's generated-artifacts exclusion; `data/` is
  intentionally *not* ignored).
- `tests/test_transforms.py` -- 12 new `ReplaceValue` unit tests.
- `tests/test_preprocessing_persistence.py` -- 3 new round-trip tests
  (`ReplaceValue` alone, `Compose([ReplaceValue, Normalize])`, full
  `save_model`/`load_preprocessing` round trip).
- `tests/test_tabular_diabetes_workflow.py` -- new, 12 CPU tests.
- `tests/test_tabular_diabetes_workflow_cuda.py` -- new, 3 CUDA tests.
- `docs/architecture/data-system.md`, `docs/architecture/data-pipeline.md`,
  `docs/architecture/persistence.md` -- transform-registry lists updated.
- No `FORMAT_VERSION` bump -- fully backward/forward compatible (a legacy
  artifact with no `ReplaceValue` step simply never references it).

## Tests

30 new tests total, all passing:

- `tests/test_transforms.py` (12): construction validation (mismatched
  lengths, empty/duplicate/negative/out-of-range columns), single-sample
  vs. batched application, columns left untouched even when their own
  value equals the sentinel, dtype/device preservation, `repr`, composition
  with `Normalize`.
- `tests/test_preprocessing_persistence.py` (3): `ReplaceValue` alone round
  trips; `Compose([ReplaceValue, Normalize])` round trips with a numeric
  check against the expected post-pipeline value; full `save_model()`/
  `load_preprocessing()` artifact-level round trip.
- `tests/test_tabular_diabetes_workflow.py` (12, CPU): real CSV loads
  deterministically with correct shape/class balance/header validation;
  `make_datasets()` fits `ReplaceValue` from training-split medians only
  (never touching `Pregnancies`); splits don't overlap; the two-stage fit
  is deterministic per seed; a direct check that `Pregnancies` stays
  untouched while `Insulin` gets imputed through the real fitted pipeline;
  `train_and_save()` writes full artifact metadata; real test accuracy
  clears the (real, imbalanced) 65.1% majority-class baseline;
  genuinely-separate-subprocess prediction agreement on a real CSV row;
  `predict_model()`/`predict_tabular_classification_artifact()` agreement;
  **a direct reproduction of the M92 finding** -- two artifacts differing
  only in whether their saved preprocessing includes `ReplaceValue` produce
  different standardized inputs (and can produce different predictions)
  for the identical raw row with a missing reading.
- `tests/test_tabular_diabetes_workflow_cuda.py` (3, CUDA, hardware-gated):
  real example trains/saves on CUDA; predicts on CUDA by default; CPU vs.
  CUDA prediction parity for the new preprocessing pipeline specifically
  (exercises `ReplaceValue`'s sentinel-comparison branch on both backends).

## Full-suite result

2,556 collected (30 new over M91's 2,526); full suite: 2,555 passed, 1
failed -- the same pre-existing, unrelated `test_dataloader_prefetch.py`
CUDA allocator-measurement flake documented since ~M63 (reproduced passing
in isolation immediately afterward; not a regression).

## Fresh-process verification

`tests/test_tabular_diabetes_workflow.py::
test_predict_model_agrees_with_a_genuinely_separate_process` launches a
real `subprocess.run([sys.executable, "-c", ...])` that imports only
`forge`/`numpy`, loads the trained artifact by path alone, and predicts on
a real CSV row (including its genuine sentinel-zero readings) -- its output
is asserted to match this test process's own prediction.

## CPU verification

`python -m examples.tabular_diabetes.train --epochs 60 --device cpu`:
72.1% test accuracy vs. 65.1% majority-class baseline. `python -m forge
model inspect`/`predict` and `python -m examples.tabular_diabetes.infer`
both verified manually against the resulting artifact, including a raw row
with real sentinel-zero readings.

## CUDA verification

`python -m examples.tabular_diabetes.train --epochs 60 --device cuda`:
72.7% test accuracy, hardware-verified on the reference GeForce 940MX (CUDA
12.6). `tests/test_tabular_diabetes_workflow_cuda.py` (3 tests) passing on
the same hardware, including CPU/CUDA prediction parity through the new
`ReplaceValue` transform.

## Real dataset training result

CPU: 72.1% test accuracy (60/20/20 split, seed 0, 60 epochs) vs. 65.1%
majority-class baseline. CUDA: 72.7% (same hyperparameters). Both well
above trivial, confirming genuine learning on real, noisy, imbalanced data
-- not state-of-the-art (not attempted; out of scope per brief Section 13),
but honest evidence the pipeline works.

## Real artifact inference result

A brand-new raw patient row (including a genuine sentinel-zero reading) is
correctly imputed, standardized, and classified in one `forge.
predict_model()` call, in a fresh process, with no manual preprocessing by
the caller -- the concrete developer-facing outcome this milestone was
built to produce.

## Existing-workflow regression result

Retrained fresh and re-verified (CLI `inspect`/`predict`, or fresh-process
`infer.py`) for all five artifact task types after this change:

- `regression` -- retrained (5 epochs), `task=regression`, CLI predict
  returned a numeric prediction.
- `segmentation` -- retrained (3 epochs), `task=segmentation`, CLI predict
  (`--output`) produced a mask file.
- `classification` (`image_folder_classification --generate`) -- retrained
  (3 epochs), `task=classification`, CLI predict returned a label.
- `sequence` (`char_rnn`) -- retrained (3 epochs), `task=sequence`, fresh-
  process `infer.py` generated text.
- `tabular_classification` (M91's own example) -- unchanged, its own test
  suite (22 tests) still passes.

No regressions. `tests/test_unified_artifact_prediction.py`,
`tests/test_cli_predict.py`, `tests/test_task_metadata.py`,
`tests/test_model_inspection.py` (117 tests combined) all pass unchanged.

## Remaining limitations

- `ReplaceValue`'s sentinel comparison is exact (`==`), not
  tolerance-based -- correct for this dataset (integer-valued sentinel `0`
  in float columns) but would need a tolerance parameter for a workload
  whose sentinel is itself a float subject to rounding. Not built --no
  workload has demonstrated the need.
- Class imbalance (65.1%/34.9%) was measured and reported (majority-class
  baseline) but not corrected for (no class weighting, no resampling) --
  matches the brief's "do not optimize for state-of-the-art accuracy."
- No feature-schema/column-name enforcement at inference time -- a caller
  who passes columns in the wrong order gets a silently wrong prediction,
  identical to M91's own documented limitation. Not demonstrated as a
  blocker by this workload either.

## Deferred capabilities

Every item in the brief's Section 25 "Explicitly Do Not Build" list:
pandas/DataFrame integration, a generic CSV framework, CSV/TSV CLI support,
a categorical-encoding framework, a general missing-value framework, a
feature-schema registry, dataset versioning, data validation, experiment
tracking, hyperparameter search, cross-validation, model serving, cloud/
distributed training, additional task types, a model registry -- none
demonstrated as necessary by this workload.

## What an external developer can now do

Take an ordinary real-world tabular CSV with sentinel-coded missing values,
parse it with a few lines of stdlib code, fit imputation and standardization
once on the training split, train a classifier with `forge.train_and_save()`,
and hand the resulting `.forge` file to anyone -- who can predict on a
brand-new **raw** row (missing readings included) with one
`forge.predict_model()` call or one `forge model predict` CLI invocation,
with no need to know that imputation happened at all, let alone reimplement
it.

## Suggested commit message

```
feat: add forge.data.ReplaceValue for persistable sentinel-value imputation, validated against a real diabetes dataset (M92)
```
