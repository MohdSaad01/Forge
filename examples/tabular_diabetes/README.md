# Forge Real-Dataset Tabular Classification Example (Milestone 92)

A workload-driven validation of Forge's tabular workflow against a real,
externally-sourced dataset rather than a synthetic one --
`examples/tabular_classification` (Milestone 91) proved the workflow with
in-process-generated data; this example asks whether it survives contact
with ordinary real-world tabular data.

```text
diabetes.csv -> load_raw() -> make_datasets() -> DataLoader
    -> MLP (Linear/ReLU/Linear/ReLU/Linear) -> CrossEntropyLoss -> Adam
    -> forge.train_and_save() -> forge.predict_model()
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`).

## The dataset

The [Pima Indians Diabetes dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv):
768 real patients, 8 numeric diagnostic measurements, binary `Outcome`
(1 = diabetes diagnosed within 5 years, 0 = not). Committed as
`data/diabetes.csv` so the example needs no network access to run. See
`dataset.py`'s module docstring for the full column list and class balance
(500 negative / 268 positive -- 65.1% / 34.9%).

## The real problem this dataset exposes: sentinel-coded missing values

Five features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
`BMI`) are physically impossible at exactly `0`, but the source data
encodes "not measured" as literal `0` rather than a blank field --
`Insulin` alone is `0` in 48.7% of rows. `Pregnancies == 0` is left alone
(a real, valid value).

**What Milestone 92 found**: `forge.data.Normalize` alone lets the whole
pipeline train, save, and predict without ever raising an error -- Forge's
existing CSV-to-`TensorDataset`-to-`DataLoader`-to-`train_and_save()`
workflow needed *zero* changes to run end-to-end on this real file. But it
is silently wrong. Manually replacing sentinel zeros with a training-set
median in raw NumPy *before* Forge sees the array also runs fine -- but
that imputation step has no representation in `forge.data.Normalize`, so
it is lost the instant the model is saved: `preprocessing=` only ever
persisted "subtract mean, divide by std." A fresh raw inference row with a
genuine missing reading gets standardized as if `0` were a real
measurement, not "not measured."

Reproduced directly against a real trained artifact
(`docs/development/m92-real-dataset-ingestion.md`): the *same* patient row
predicts `no_diabetes` at 100% confidence when the artifact's preprocessing
doesn't know about imputation, and `diabetes` at 70% confidence when the
developer remembers to hand-reimplement the exact training-time imputation
constants themselves -- constants that are not recoverable from the `.forge`
file at all. **The predicted class flips**, purely based on whether the
developer happens to still have that code lying around.

## The fix: `forge.data.ReplaceValue`

A small, persistable transform (`forge/data/transforms.py`, registered in
`forge/serialization/transforms.py`) that replaces a sentinel value with a
fixed per-column fill value, on specific columns only:

```python
from forge.data import Compose, Normalize, ReplaceValue

preprocessing = Compose([
    ReplaceValue(sentinel=0.0, columns=[1, 2, 3, 4, 5], fill=training_medians),
    Normalize(mean=..., std=...),
])
```

Fit once (both stages, from the training split only -- see
`dataset.py::make_datasets()`), then saved as one `preprocessing=` value.
Because `ReplaceValue` is registered for persistence exactly like
`Normalize`/`Resize`/`Reshape`/`Flatten`, the whole `Compose` pipeline
-- imputation *and* standardization -- travels with the artifact and applies
automatically and identically to every future raw inference row. No
general missing-value framework was added: `ReplaceValue` is scoped to
"replace one sentinel with a fixed per-column value," nothing more
(no strategy selection, no auto-detection of which columns need it, no
`NaN` handling) -- the smallest capability this real, demonstrated blocker
required.

## Model

```text
(N, 8) -> Linear(8, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 2) -> (N, 2)
```

Smaller than `tabular_classification`'s MLP -- 768 real rows is a much
smaller dataset than that example's synthetic one, and a smaller network
overfits less.

## CPU training

```bash
python -m examples.tabular_diabetes.train --epochs 60 --device cpu
```

Reference numbers from this repository's CPU (i5-7200U, `--seed 0`, default
hyperparameters, 60 epochs, 60/20/20 train/val/test split): final test
accuracy **72.1%** against a **65.1%** majority-class baseline (always
predicting `no_diabetes`) -- genuine learning on real, noisy, imbalanced
data, not merely "the code runs." This is *not* a uniform-random 50%
baseline: with 65.1%/34.9% class imbalance, "always guess the majority
class" is the real triviality bar a real dataset like this one sets, unlike
`tabular_classification`'s roughly-balanced synthetic four-class problem.
Exact numbers vary by hardware/seed; only "above the majority-class
baseline" is a stability guarantee
(`tests/test_tabular_diabetes_workflow.py` checks this directly).

## CUDA training

```bash
python -m examples.tabular_diabetes.train --epochs 60 --device cuda
```

Identical model/optimizer/data pipeline; hardware-verified on the reference
GeForce 940MX.

## No checkpoint/resume

Matching `tabular_classification` -- this dataset is small enough (768 rows)
that training finishes in a few seconds, so interrupted-training resume is
not a realistic concern.

## Determinism

`--seed` (default `0`) governs `forge.random.seed()` (`Linear` parameter
initialization), `make_datasets()`'s own `numpy.random.default_rng(seed +
1)` (the train/val/test `random_split` permutation), and `DataLoader`'s
explicit shuffle generator.

## Model persistence

`train.py` trains and saves through one `forge.train_and_save()` call,
including the fitted `Compose([ReplaceValue, Normalize])` preprocessing
pipeline, the class vocabulary (`classes=CLASS_NAMES`), and
`task="tabular_classification"` (the same task Milestone 91 introduced --
this dataset's input/output shape is the identical inference contract, so
no new task type was added; see the Milestone 92 brief's own "Do not
confuse data loading with artifact semantics" section).

## Portable-artifact inference

A developer holding only `tabular_diabetes_model.forge` gets a correct
prediction on a **brand-new raw patient row, sentinel zeros and all**, in
one call -- no manual imputation required:

```python
import forge
import numpy as np

# A new patient row with a genuinely-missing Insulin reading (0) --
# order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
# DiabetesPedigreeFunction, Age.
raw_row = np.array([[2, 130, 70, 25, 0, 28.5, 0.5, 35]], dtype=np.float32)
results = forge.predict_model("examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge", raw_row)
print(results[0].label, results[0].confidence)
```

`predict_model()` reads the artifact's `task="tabular_classification"`
metadata and dispatches to `forge.predict_tabular_classification_artifact()`
(Milestone 91), which loads and applies the saved `Compose([ReplaceValue,
Normalize])` preprocessing -- including the imputation -- before the
forward pass, exactly like it already does for `Normalize` alone.

## CLI inspection and prediction

```bash
python -m forge model inspect examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge
```

```bash
echo '[2, 130, 70, 25, 0, 28.5, 0.5, 35]' > input.json
python -m forge model predict examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge input.json
```

`forge model inspect` renders the full preprocessing pipeline, including
the new transform, via `Compose`'s existing `__repr__`:

```text
Preprocessing detail: ReplaceValue(sentinel=0.0, columns=[1, 2, 3, 4, 5], fill=[...]) -> Normalize(mean=[...], std=[...])
```

## Fresh-process inference

```bash
python -m examples.tabular_diabetes.infer \
    --model examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge \
    --input '[2, 130, 70, 25, 0, 28.5, 0.5, 35]'
```

A standalone script independent of `train.py` -- see `infer.py`'s own
module docstring.

## Multi-row inference without reloading the artifact per row (Milestone 102)

```bash
python -m examples.tabular_diabetes.app \
    --model examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge \
    --input '[[2, 130, 70, 25, 0, 28.5, 0.5, 35], [1, 85, 66, 29, 0, 26.6, 0.351, 31]]'
```

```text
Loaded examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge once (tabular_classification artifact, 2 classes) -- predicting 2 row(s).
Row 0: no_diabetes (confidence 64.4%)
Row 1: no_diabetes (confidence 75.7%)
```

`infer.py` is the right tool for one patient row (`forge.predict_model()`
reloads the artifact for that one call). A real application predicting on
many rows should not pay that reload cost per row -- `app.py` loads the
artifact exactly once via `forge.load_predictor()`, then calls
`predictor.predict(row)` per row, reusing the same cached model,
preprocessing, and `InputSchema` for every prediction. A structurally
invalid row (wrong feature count) is reported inline rather than aborting
the rest of the batch. See `forge.load_predictor()`'s own docstring for the
full reusable-inference contract, and `app.py`'s own module docstring for
this script's independence from `train.py`/`dataset.py`.

## Evaluating the artifact on unseen labeled data (Milestone 97)

```bash
python -m examples.tabular_diabetes.evaluate \
    --model examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge \
    --data examples/tabular_diabetes/data/diabetes_holdout_eval.csv
```

```text
Model:             examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge
Test samples:      154
Accuracy:          72.1%
Baseline accuracy: 62.3%
Improvement:       +9.7 percentage points
-> model is meaningfully better than the majority-class baseline.
```

A standalone script, independent of `train.py`/`dataset.py`/`model.py`
exactly like `infer.py` -- it composes `forge.load_model()` +
`forge.load_preprocessing()` + `forge.predict()` + `forge.training.Accuracy`
directly, deliberately **not** `Trainer.evaluate()` (see that method's own
updated docstring, `forge/training/trainer.py`, for why: it has no hook for
persisted preprocessing and silently produces a badly wrong, worse-than-
baseline number on this dataset's raw rows). `data/diabetes_holdout_eval.csv`
is `dataset.make_datasets(seed=0)`'s own `test_ds` split, exported once to a
plain CSV so it is reachable without importing any producer code -- see
`evaluate.py`'s own module docstring for exact reproduction.

### The library equivalent (Milestone 113)

`forge.load_predictor(path).evaluate(X, y)` is the first-class version of what
`evaluate.py` composes by hand: it applies the artifact's persisted
`ReplaceValue`/`Normalize` and returns accuracy, baseline, loss, a confusion
matrix, and per-class precision/recall in one call.

```python
predictor = forge.load_predictor("examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge")
result = predictor.evaluate(X, y)      # X: raw (n, 8) rows, y: 0/1 or "no_diabetes"/"diabetes"
```

```text
samples 154 | accuracy 72.1% | baseline 62.3% | loss 0.586
                 precision  recall  support
no_diabetes         0.757    0.812       96
diabetes            0.647    0.569       58
confusion matrix (rows = true):  [[78 18]
                                  [25 33]]
```

`evaluate.py` is deliberately left as it is: an independent implementation
that `tests/test_artifact_evaluation.py` uses as the numeric oracle, so the
library API is checked against something that does not share its code.

## Early stopping (Milestone 100)

Milestone 98's own experiment comparison found this workload's validation
loss visibly bottoms out well before the default 60 epochs. `--early-stopping`
(off by default, so plain `python -m examples.tabular_diabetes.train` behaves
exactly as before) turns on `forge.training.EarlyStopping`:

```bash
python -m examples.tabular_diabetes.train --epochs 60 --seed 0 --device cpu \
    --early-stopping --patience 5
```

Reference run (this repository, `--seed 0`, `patience=5`, `min_delta=0.0`,
`restore_best=True`, default hyperparameters otherwise) versus a plain fixed
60-epoch run with the same seed:

```text
                    fixed 60 epochs   60-epoch max + early stopping
epochs completed    60                12
best epoch          --                7
best val_loss       --                0.5204
final val_loss       0.5775           0.5211 (epoch 12, not restored)
test loss            0.5858           0.5115
test accuracy         72.1%            72.1%
demo-row confidence   99.8%            90.4%
```

Early stopping activated for real (`stopped_early=True`) and restored epoch
7's parameters -- test *accuracy* happened to match the fixed run exactly at
this sample size, but test *loss* is meaningfully lower and the restored
model is visibly less overconfident on the demo row (90.4% vs. 99.8%),
consistent with the fixed run having overfit past its best validation point.
This is a single-seed, single-dataset comparison, not a claim that early
stopping is universally better here -- see `docs/development/progress.md`'s
Milestone 100 entry. `--patience`/`--min-delta` are also exposed for
experimentation; `--early-stopping` requires no other flag changes since
`validation_dataset=val_loader` is always already passed to
`forge.train_and_save()`.

## Column order

Like `tabular_classification`, this example does not introduce a feature-
schema abstraction: a raw inference row must supply its 8 values in exactly
the order `FEATURE_NAMES` documents (`dataset.py`). This was not a
demonstrated blocker for this workload -- a single, documented, fixed
column order was sufficient -- so no schema metadata was added (Milestone
92 brief, Section 9).

## What this milestone did *not* need to build

Per the Milestone 92 brief's own scope discipline: no CSV-parsing
framework (stdlib `csv` + NumPy was already sufficient, see `dataset.py::
load_raw()`), no pandas/DataFrame integration, no categorical-encoding
support (this dataset has none), no general missing-value framework (just
the one narrow `ReplaceValue` transform the workload demonstrated), and no
new artifact task type (`task="tabular_classification"` already fully
describes this workload's inference contract).

## Integration tests

`tests/test_tabular_diabetes_workflow.py` (CPU) and
`tests/test_tabular_diabetes_workflow_cuda.py` (CUDA; skips cleanly without
a working CUDA backend) exercise this pipeline end-to-end, including a
direct reproduction of the Milestone 92 finding (a raw row with a missing
reading gets a materially different prediction depending on whether
`ReplaceValue` is part of the saved preprocessing). Run them with:

```bash
python -m pytest tests/test_tabular_diabetes_workflow.py tests/test_tabular_diabetes_workflow_cuda.py
```

`ReplaceValue` itself (the reusable transform, independent of this example)
is covered by `tests/test_transforms.py` and its persistence round-trip by
`tests/test_preprocessing_persistence.py`.
