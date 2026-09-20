# M114 — First-class tabular classification and regression workflows

`forge.train_tabular_classifier()` and `forge.train_tabular_regressor()`: a numeric
feature matrix and targets in, a verified portable `.forge` artifact and a result
object out, composed entirely from existing Forge machinery. Reference for the
behaviour is the module docstring of `forge/training/tabular.py`; this document is
the milestone report and the record of what was measured.

## 1. Summary

- Two explicit workflow functions plus two frozen result dataclasses, exported from
  `forge` and `forge.training`. No second training loop, persistence path or
  prediction path: `random_split` -> preprocessing fitted on the training rows ->
  `TensorDataset`/`DataLoader` -> default MLP (or `model=`) -> `train_and_save()` ->
  `load_predictor().evaluate()` (M113) for the numbers in the result.
- Everything predictable is rejected **before epoch 1**: data shape/finiteness/
  target validity, argument ranges, a caller's `model=` against the task (a probe
  forward pass), and the output `path` plus serialisability (a real `save_model()`
  dry run of the untrained model into a temporary sibling file).
- Validated on two real datasets: Pima diabetes (holdout 72.7% vs 62.3% baseline)
  and UCI Concrete Compressive Strength (test R^2 0.926, targets unscaled), on CPU
  and on the 940MX, and from a wheel-only environment in a fresh process.
- One pre-existing defect found by the CUDA run and fixed at its root: a
  CUDA-loaded artifact rejected the float64/integer NumPy arrays that
  `np.genfromtxt` and integer columns produce (section 12).
- 164 new tests; two bundled artifacts with consumer scripts.

## 2. Public API

```python
forge.train_tabular_classifier(
    X, y, *, path, classes=None, epochs=100, batch_size=32, learning_rate=1e-3,
    val_fraction=0.2, missing_columns=None, missing_value=0.0, patience=10,
    model=None, device=None, seed=0, verbose=False,
) -> TabularClassificationResult

forge.train_tabular_regressor(
    X, y, *, path, epochs=500, batch_size=32, learning_rate=1e-3,
    val_fraction=0.2, missing_columns=None, missing_value=0.0, patience=30,
    model=None, device=None, seed=0, verbose=False,
) -> TabularRegressionResult
```

`X`: numeric `(samples, features)` (`ndarray`, nested list, `Tensor`). Classification
`y`: class names (`str`) or integer indices (integer-valued floats and booleans
accepted); regression `y`: numeric `(n,)`, `(n, 1)` or `(n, outputs)`. `path` may be
`str` or `os.PathLike`.

Argument decisions (each made from the existing conventions, not copied from the
image API): `val_fraction` keeps `train_image_classifier`'s name; `patience` rather
than an `EarlyStopping` object (tabular users want "stop after N stagnant epochs";
the existing `EarlyStopping(patience, restore_best=True)` is what runs underneath);
`verbose` defaults to `False` (tabular epochs take milliseconds, so per-epoch lines
are noise where the image API's are progress); loss and optimizer are not exposed
(`CrossEntropyLoss`/`MSELoss`, `Adam`); no `validation_data=` (module docstring,
**Validation split**: for i.i.d. rows the seeded random split is right, and a dedicated
held-out set is scored with `predictor.evaluate()`); `missing_columns=`/
`missing_value=` expose only the existing `ReplaceValue`, off by default.

Results (frozen dataclasses; `eq` default like `ImageClassifierResult`):

| `TabularClassificationResult` | `TabularRegressionResult` |
|---|---|
| `task, samples, features, train_samples, validation_samples` | `task, samples, features, outputs, train_samples, validation_samples` |
| `classes` (output order, also persisted) | -- |
| `epochs_completed, stopped_early, best_epoch, best_monitored_value` | same |
| `train_loss, validation_loss` (cross-entropy) | `train_loss, validation_loss` (`MSELoss`, == mse to float32 rounding) |
| `train_accuracy, validation_accuracy, baseline_accuracy` | `train_mse, validation_mse, train_mae, validation_mae, baseline_mse` |
| `artifact_path, history` (`TrainingResult`), `model` (reloaded, verified) | same |

`train_*`/`validation_*` are `ArtifactPredictor.evaluate()` of the **saved artifact**
on the raw training / validation rows -- not the running per-epoch numbers, which
describe a model that is still changing (and, with early stopping, not the model
that was saved). `baseline_accuracy` is the majority share of the *validation* rows;
`baseline_mse` the MSE of predicting the *validation* mean -- the M113 definitions.

## 3. Architecture

```
X, y --validate--> random_split(seed) --> fit ReplaceValue?/Normalize on TRAIN rows
   --> Compose (the object that is also saved as preprocessing=)
   --> TensorDataset/DataLoader --> default MLP | model= --> CrossEntropyLoss|MSELoss + Adam
   --> train_and_save(task=..., classes=..., preprocessing=..., early_stopping=...)
   --> load_predictor(path).evaluate(train rows / validation rows) --> result
```

- **Split before fit.** The split is `random_split` on a `TensorDataset`; statistics
  are computed from `X[train_indices]` only. Tests reproduce the split from its
  documented rule and check the persisted mean/std against NumPy on those rows, and a
  test overwrites every validation row with `1e9` and asserts the persisted
  preprocessing is unchanged.
- **Preprocessing.** `Normalize` per feature; a constant (std ~ 0) feature gets
  `std = 1`. Optional `ReplaceValue` first (sentinel -> training-split median of that
  column's non-sentinel entries; a column that is entirely sentinel in the training
  split is a `DataError`). Features are transformed once, in batch, with the same
  fitted `Compose` that is saved -- the transform already accepts `(N, F)`.
- **Default models.** Classification `Linear(F,32)-ReLU-Linear(32,16)-ReLU-Linear(16,K)`
  (the Pima model's widths); regression `Linear(F,64)-ReLU-Linear(64,32)-ReLU-Linear(32,d)`
  (`examples/regression`'s). Both `Sequential` of registered layers, so the artifact
  gets an `InputSchema`. Widths were checked, not assumed (section 5, 6).
- **Labels follow M113's `evaluate()` convention** (names or indices), so the `y` a
  user trained with is the `y` they pass to `evaluate()`. Names are sorted; integer
  labels must be exactly `0..K-1` (otherwise an index label `1` could silently mean
  the class `"2"` at evaluation time); `classes=[...]` fixes the order explicitly.

## 4. Preflight validation (before epoch 1)

Each row is tested with `Trainer.fit` replaced by a tripwire that fails the test if
training starts, and (where relevant) that nothing was written to the directory.

| Check | Error | Why before epoch 1 |
|---|---|---|
| `X` not numeric / not 2-D / empty; NaN or Inf (checked again after the float32 cast, so a float64 that overflows is caught) | `DataError` naming the first bad index | M111/M112: NaN trained silently to `loss = nan` |
| `len(X) != len(y)`; `y` 2-D / non-integer floats / mixed / a single class / an unused class / names not in `classes=` | `DataError` | the failure would otherwise surface as a shape error or a dead output |
| Regression `y` NaN/Inf, non-numeric, 3-D, wrong length | `DataError` | same |
| A class with no rows in the training split | `DataError` | it could never be learned |
| `model=` not a `Module`; cannot process `(rows, F)`; output width != K / target width | `TrainerError` | M111: a 5-output model trained on 2 classes, saved, verified, and failed on every `predict()` |
| `path`: directory missing, is a directory, parent is a file, invalid filename, not writable | `PersistenceError` | M111: failed only after all epochs |
| Model/preprocessing/classes not serialisable (e.g. an unregistered `Module`) | `PersistenceError` | same |
| `epochs`, `batch_size`, `learning_rate`, `val_fraction`, `patience`, `seed`, `missing_*` out of range | `DataError` | cheap |

The `model=` contract is checked by **running the model once** on two real
preprocessed rows, not by reading its structure: it verifies input width and output
width exactly and works for any `Module` (no architecture-inference framework, no
guessing). The save check calls the real `save_model()` on the untrained model with a
temporary name in `path`'s directory that embeds `path`'s own filename -- so an invalid
filename fails there exactly as it would at the end -- and removes it; the final
artifact is not written until training finishes. The directory must already exist,
matching `save_model()`'s own contract; nothing is created implicitly.

A non-finite loss *during* training (finite data, absurd learning rate) is
`Trainer`'s existing `TrainerError`; `learning_rate=1e30` is tested to raise it and to
leave no artifact.

## 5. Tabular classification: Pima Indians Diabetes

- **Data:** the bundled `examples/tabular_diabetes/data/diabetes.csv`, unchanged:
  768 rows x 8 features, binary `Outcome`; zero is "not measured" in Glucose,
  BloodPressure, SkinThickness, Insulin, BMI -> `missing_columns=[1, 2, 3, 4, 5]`
  (the M92 semantics; without it the workflow is silently wrong, as M92 showed).
- **Protocol:** the M97/M113 154-row `diabetes_holdout_eval.csv` (62.3% majority
  share) is the test set; the other 614 rows (an exact complement: the main file has
  no duplicate rows) go to `train_tabular_classifier()`, which splits them 491 / 123.
- **Result (CPU, seed 0):** 85 epochs, early-stopped, best epoch 75. Train 85.7%,
  validation 77.2% (baseline 63.4%). **Holdout 72.7% vs 62.3% baseline**, loss 0.5852,
  confusion `[[80, 16], [26, 32]]`, precision `(0.755, 0.667)`, recall `(0.833, 0.552)`.
  For reference the M113 artifact scored 72.1% (`[[78, 18], [25, 33]]`) on the same
  holdout; not a claim of improvement -- different training rows and stochastic
  training.
- **Width check** (5 seeds, holdout accuracy): 32/16 mean 73.4%, 64/32 mean 73.8%;
  no measurable difference, so the smaller Pima model stays.
- **Artifact:** `models/tabular_classifier/diabetes_classifier.forge` (5.3 KB,
  `device: cpu`, `task: tabular_classification`, classes `no_diabetes`/`diabetes`,
  `ReplaceValue -> Normalize` persisted).

## 6. Tabular regression: UCI Concrete Compressive Strength

**Verified file properties** (inspected before training; not taken from M112):

| | |
|---|---|
| Source | `https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip` -> `Concrete_Data.xls` (legacy binary Excel), SHA-256 `710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4` |
| Sheets | `Sheet1` 1031 x 9 (1 header + **1030 data rows**, 9 columns); `Sheet2`, `Sheet3` empty |
| Cell types | every data cell is numeric (xlrd type 2); no blank, text or error cells; **no NaN/Inf** |
| Features (8) | Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate (kg/m^3), Age (days) |
| Target (last col) | compressive strength, MPa: min 2.3318, max 82.5992, mean 35.8178, std 16.6976 |
| Feature ranges | cement 102-540; slag 0-359.4; fly ash 0-200.1; water 121.75-247; superplasticizer 0-32.2; coarse 801-1145; fine 594-992.6; age 1-365 (14 distinct values) |
| Zeros | slag 466, fly ash 566, superplasticizer 379 rows: the ingredient is absent, a real quantity. The dataset's readme states "Missing Attribute Values: None". **Not** treated as missing (`missing_columns` unused) |
| Other | 25 exact duplicate rows (kept); no constant column; header text contains parentheses and a comma, so the CSV quotes it |

**Transformation applied:** format conversion only. `xlrd` (installed into a
throwaway directory, not into the project environment and not a Forge dependency)
read `Sheet1`; each value was written with `repr(float(v))`, header text verbatim ->
`concrete.csv`, SHA-256 `733f7eb8570d2612748da66e9f858032c6202cc2f10980cefeb03c60da3f6b38`.
No row dropped, no value changed. The CSV is not committed (the repo does not ship
datasets it does not need); `tests/real_world/tabular_workflows.py` takes it via
`--concrete-csv` and skips the regression half, saying so, without it.

- **Protocol:** fixed 206-row test set `default_rng(123).permutation(1030)[:206]`; the
  other 824 rows to `train_tabular_regressor()` (659 train / 165 validation).
- **Target scaling was not needed** (the brief's stop-and-document condition did not
  trigger). Straight numeric regression, defaults except as noted: 200 epochs at
  lr 1e-3 reached test R^2 0.884 and was still improving (best epoch 196) -- so the
  default budget was raised, not the target scaled.
- **Defaults were chosen on validation MSE, 3 seeds** (test shown for confirmation):

  | lr | epochs / patience | epochs run | val MSE | test R^2 |
  |---|---|---|---|---|
  | 1e-3 | 500 / 30 | 500, 418, 446 | 29.5 | 0.910 |
  | 3e-3 | 300 / 30 | 300, 190, 277 | 26.9 | 0.908 |
  | 1e-2 | 300 / 30 | 91, 105, 184 | 28.3 | 0.916 |

  All within noise, so `lr = 1e-3` (the family-wide default) with `epochs=500,
  patience=30` (~15 s on the i5-7200U). Widths 64/32 vs 32/16 (200 epochs): test MSE
  36.3 vs 37.2; 64/32 kept.
- **Result (CPU, seed 0, the bundled artifact):** 500 epochs (cap reached, best epoch
  491). Train MSE 13.85 / MAE 2.80; validation MSE 28.21 (baseline 261.88);
  **test MSE 23.21, MAE 3.39 MPa, baseline 312.23, R^2 0.926.**
- **Artifact:** `models/tabular_regressor/concrete_strength_regressor.forge` (12 KB,
  `device: cpu`, `task: regression`, `Normalize` persisted, no target transform).
- **Target-scale sensitivity (measured, a limitation):** the same real data with the
  target multiplied by 1 / 30 / 1000 (defaults, same split): test R^2 **0.926 / 0.680 /
  0.666**. Unscaled targets work at Concrete's magnitude and degrade beyond it. Fixing
  that needs a persisted, inverted target transform through `predict`/`evaluate` --
  the design decision the brief says not to improvise.

## 7. Evaluation

The result numbers are `evaluate()`'s, and tests assert equality with an independent
`predictor.evaluate(X[val_idx], y[val_idx])` (exact, including the float64 regression
targets), and with NumPy on the artifact's own predictions (regression MSE/MAE to
1e-4 relative) and on the baselines (`np.bincount(...).max()/n`; `np.mean((y - y.mean())**2)`).
Cross-checks: `best_monitored_value` (from `Trainer` during training) and the
artifact's re-evaluated validation loss agree to float32 rounding (0.44536970 vs
0.44536966 on the first Pima run) -- the training-time and the saved-artifact paths
score the same model the same way. Mutation: saving the same weights *without* the
persisted preprocessing drops validation accuracy by > 0.25 on the awkward-scale
fixture and multiplies regression MSE by > 20 (both tested).

## 8. Artifacts

```
models/tabular_classifier/diabetes_classifier.forge   predict.py
models/tabular_regressor/concrete_strength_regressor.forge   predict.py
```

Regenerate with `python tests/real_world/tabular_workflows.py --concrete-csv <csv>
--models-dir models` (CPU-trained: an artifact recording `device: cuda` will not load
on a machine without CUDA, the I1 lesson). Consumers use only public APIs
(`forge.load_predictor(path, device="cpu")`, `predict`), never rebuild the model, run
from the model directory or the repository root, take 8 numbers as separate arguments
or one comma-separated argument, print ASCII only, and turn wrong counts, non-numeric
text, NaN and Inf into a one-line error and exit 1. Example output:

```
Forge Tabular Classifier          Forge Tabular Regressor
------------------------          -----------------------
Prediction: diabetes              Predicted compressive strength: 65.8 MPa
Confidence: 72.8%
```

Both print that they demonstrate the workflow and are not a medical tool / engineering
estimate; the example rows are the first rows of their datasets, not a quality claim.

## 9. Tests

164 new: `tests/test_tabular_workflows.py` (138, CPU), `tests/test_tabular_workflows_cuda.py`
(9), `tests/test_bundled_tabular_models.py` (17). Public behaviour throughout; the
split is reproduced from its documented rule so leakage/baselines are checked against
NumPy, not the implementation. 

**Full suite (`python -m pytest tests/ -q`, CUDA available: 940MX): 3009 passed, 0 failed, 0 skipped**, 754.6 s -- the previous 2,845 plus these 164; no existing test was modified.

Mutation checks (implementation deliberately broken, then restored; each turns the
named tests red):

| Mutation | Tests failing |
|---|---|
| save without persisted preprocessing | 15 (raw-row predict/evaluate accuracy, persisted-mean, fresh-process...) |
| finite check disabled | 6 (NaN/Inf in `X` and `y`, both APIs) |
| `model=` contract check skipped | 5 (wrong output width, wrong input width, regression widths) |
| save preflight skipped | 9 (missing/invalid path, parent is file, unserialisable model) |
| statistics fitted on all rows (precise version) | 4 (the leakage tests) |

The first version of the last mutation was too crude (it broke row counts, failing ~50
tests, proving nothing about leakage); it was redone to change only which rows the
statistics see.

## 10. CPU / CUDA

CUDA hardware was available (940MX, CC 5.0, CUDA 12.6) and used. Real workloads, same
protocol, `device="cuda"`:

| | CPU | CUDA |
|---|---|---|
| Pima holdout | 72.7%, `[[80,16],[26,32]]`, 85 epochs, 1.9 s | 72.7%, `[[80,16],[26,32]]`, 85 epochs, 7.1 s |
| Concrete test | MSE 23.21, R^2 0.926, 500 epochs, 16.0 s | MSE 24.05, R^2 0.923, 489 epochs, 82.8 s |

Save -> load (artifact records `device: cuda`) -> predict -> evaluate all ran on the
GPU. CUDA training is not bit-identical to CPU (pre-existing, documented in M107);
tests therefore compare the device-independent parts exactly (split, fitted
preprocessing, baselines), *the same saved weights* run on both devices (identical
confusion matrix / accuracy, loss within 1e-4, regression predictions within 1e-4
relative), and only require independently trained models to be comparably good.
**Finding:** these small MLPs train ~5x *slower* on CUDA than on this CPU
(launch-bound at batch 32) -- supported and tested, not a speed-up here.

## 11. Fresh-process validation

- Pytest: both artifacts consumed by a `subprocess` (CUDA hidden) that loads,
  predicts and evaluates; its labels/accuracy/confusion matrix/MSE/MAE/baselines are
  asserted equal to the in-process values. The consumer scripts run from a working
  directory outside the repository, from the model directory, and from the repo root.
- **Wheel-only environment:** `python -m build --wheel`; a new venv with only that
  wheel and its dependencies (NumPy 2.5.3, Pillow 12.3.0); a directory containing
  *only* `tabular_classifier/`, `tabular_regressor/` and two CSVs; `CUDA_VISIBLE_DEVICES=-1`;
  `forge.__file__` resolved to the venv's `site-packages`. Both `predict.py` scripts ran
  (72.8% `diabetes`; 65.8 MPa) and `load_predictor().evaluate()` reproduced the in-repo
  numbers exactly: Pima holdout 72.7% vs 62.3%, `[[80,16],[26,32]]`; Concrete test MSE
  23.21, MAE 3.39, R^2 0.926.
- The existing CLI reads the new artifacts (`forge model inspect`, `forge model
  predict`); tested for both tasks.

## 12. Findings

**M114 defects fixed**

1. *(pre-existing, found by this milestone's CUDA run)* `predict()`/`evaluate()` on an
   artifact loaded onto CUDA raised `CUDAError: CUDA 'matmul' requires matching dtypes,
   got ['float64', 'float32']` for a float64 NumPy array (what `np.genfromtxt`/`loadtxt`
   return) and for integer arrays/lists; CPU silently promoted and returned float64.
   Reproduced with **only pre-existing APIs** (a hand-saved model + `Normalize`), then
   fixed in the one shared boundary `_coerce_numeric_input()`: arrays/lists become
   float32 (`DEFAULT_DTYPE`) before preprocessing; explicit `Tensor`s are untouched.
   The 244 existing inference/artifact/evaluation/CLI/bundled-model tests pass
   unchanged. **Behaviour change to note:** CPU predictions from float64 input are now
   float32 (values agree to 1e-6 relative). Tested on CPU and CUDA.
2. *(in-milestone, caught by tests)* result numbers were first computed against
   float32-rounded regression targets and differed from a later `evaluate(X_val, y_val)`
   at the 1e-7 relative level; targets are now kept in float64 for scoring.

**Pre-existing, not fixed (out of this brief's scope)**

- `train_image_classifier()` still has the late-failure modes M111 found (bad `path=`
  directory after all epochs; unvalidated `model=`). M114's brief scopes the preflight
  to the two new APIs; retrofitting the image API is a separate change.
- A ragged nested list passed to `predict()` leaks a raw NumPy `ValueError` from
  `Tensor()` (M113 finding; `evaluate()`/`train_tabular_*` wrap it in `DataError`).
- `Trainer.evaluate()` still has no persisted-preprocessing hook (documented since M97).

**Deferred / observed**

- Target scaling (section 6 measurements) -- needs a persisted, inverted transform.
- No stratified split; the class-absent-from-training check is the guard.
- 25 duplicate Concrete rows can straddle a split (mild optimism); kept and recorded.
- The Concrete CSV is not shipped, so the regression *acceptance script* needs an
  external file; the bundled regressor artifact and its tests are the durable evidence.
- Small-MLP CUDA slowness (section 10).

## 13. Limitations

- Unscaled regression targets degrade at large magnitudes (R^2 0.93 -> 0.68 at 30x
  Concrete's scale); very large targets should be rescaled by the caller for now.
- Random split only (no time-series, grouped or stratified split); features are
  positional -- the artifact records the feature *count*, not names or order
  (`InputSchema`'s existing, documented limit).
- The regression default budget (500 epochs) can still be reached before early
  stopping fires (it was on the Concrete seed-0 run), costing ~15 s on this CPU.
- MLP only for the default; `model=` covers everything else that maps
  `(batch, F)` -> `(batch, K or d)`.
- A caller-supplied `model=` is probed with the training features only; a model that
  behaves differently in train vs eval mode is not detected.

## 14. Next step

The A-class surface planned in M111/M112 -- image classification (M107), saved-artifact
evaluation (M113), tabular classification and regression (M114) -- is now complete.
Nothing further is *scheduled*. Two candidates have concrete evidence behind them and
are recorded for a future decision, not started: (1) a persisted, inverted target
transform for large-magnitude regression targets (measured above), and (2) retrofitting
M114's preflight to `train_image_classifier()` (M111's original finding). B-class
workflows (forecasting, segmentation, text classification) remain trigger-based.

**M114 Decision: ACCEPTED**
