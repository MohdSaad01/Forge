# M121 — First-class tabular CSV training workflow

Status: **IMPLEMENTED.** M118 gave the tabular workflow a CSV *reader* (`forge.data.load_csv()`); M120 taught
that reader to select and order specific feature columns explicitly (`columns=`), closing the "an id column
forces a file rewrite" gap for *prediction* and *evaluation*. Training was the one door M120 deliberately left
closed: a developer with a production-style CSV still had to write `load_csv(..., columns=..., return_
feature_names=True)` followed by `train_tabular_classifier(X, y, ..., feature_names=names)` by hand, threading
`names` from the first call into the second themselves -- exactly the sequence
`tests/test_csv_column_selection_training.py` already wrote out longhand for every one of its own tests. M121
closes that gap with two new functions, `forge.train_tabular_classifier_csv()` / `forge.train_tabular_
regressor_csv()`, and one new CLI command, `forge model train`, both thin orchestration layers over the
existing, unmodified pipeline.

## 1. Objective

Close the remaining concrete gap in Forge's high-level tabular workflow: a developer with a production-style
CSV should be able to explicitly select the feature columns and target column, train a tabular model, persist
the feature schema and preprocessing, and receive a portable artifact without manually writing CSV-loading
glue code.

## 2. Existing gap

Before this milestone, the only way to train from a CSV was:

```python
X, y, names = forge.data.load_csv(
    "pima.csv", target="Outcome", labels=True,
    columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
             "DiabetesPedigreeFunction", "Age"],
    return_feature_names=True,
)
result = forge.train_tabular_classifier(X, y, path="pima.forge", feature_names=names, classes=[...])
```

Two calls, in a fixed order, with `names` manually threaded from the first into the second. Every M120 test
that exercises the full CSV-to-artifact path (`tests/test_csv_column_selection_training.py`) writes exactly
this sequence out longhand. `forge model train` did not exist at all -- `forge model predict`/`forge model
evaluate` could consume a CSV, but training one required a Python script; there was no CLI door onto training
symmetrical with the CLI doors onto prediction/evaluation.

## 3. API design

```python
forge.train_tabular_classifier_csv(
    "patients.csv", target="Outcome", path="patients.forge",
    columns=["Pregnancies", "Glucose", "BMI"],
    classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5],
)
forge.train_tabular_regressor_csv(
    "housing.csv", target="median_house_value", path="housing.forge",
    columns=["median_income", "housing_median_age", "total_rooms"],
    target_transform="standardize",
)
```

New module `forge/training/tabular_csv.py` (mirroring `forge/data/csv_reader.py`'s own precedent of a
dedicated boundary module, rather than folding CSV-awareness into `forge/training/tabular.py`). Each function
is exactly two calls:

```python
X, y, names = load_csv(csv_path, target=target, labels=..., columns=columns, return_feature_names=True)
return train_tabular_classifier(X, y, ..., feature_names=names)   # or train_tabular_regressor
```

No new validation, no new defaults, no new result type. Every keyword `train_tabular_classifier()`/
`train_tabular_regressor()` documents (`classes`, `epochs`, `batch_size`, `learning_rate`, `val_fraction`,
`missing_columns`, `missing_value`, `patience`, `model`, `device`, `seed`, `verbose`, and, for regression,
`target_transform`) is a keyword on the CSV wrapper too, with the identical default, spelled out explicitly
(not `**kwargs`) so an unrecognized keyword is an ordinary Python `TypeError`, not silently swallowed. There is
deliberately **no `feature_names=` parameter** on either wrapper -- the CSV's own selected column names are
threaded through automatically, which is the entire point of the milestone (§7 below).

`train_tabular_classifier(X, y, ...)` / `train_tabular_regressor(X, y, ...)` are completely unmodified: not one
line of `forge/training/tabular.py` changed. `csv_path`'s CSV-loading errors surface under `load_csv()`'s own
name and message; everything downstream surfaces under `train_tabular_classifier()`/`train_tabular_
regressor()`'s own name -- exactly as if a caller had written the two calls by hand, because that is
literally what happens.

## 4. CLI design

```bash
forge model train DATA.csv --task {classification,regression} --target COLUMN --output PATH \
    [--columns NAME [NAME ...]] [--device {cpu,cuda}] [--epochs N] [--batch-size N] \
    [--learning-rate LR] [--seed N] [--target-transform standardize] [--json]
```

`forge/cli/model.py::cmd_train()`. `--task` alone dispatches to one of the two Python functions above (never
an architecture/data-driven guess); every other flag is forwarded **only when the caller actually gave it**
(`_forwarded_training_kwargs()`), so an omitted flag is exactly that Python function's own default, never a
second, CLI-specific default that could silently drift from it. `DATA` must be a `.csv` file (extension-based
dispatch, matching `predict`/`evaluate`'s own convention) -- no other format trains through this command.

**CLI scope (§11 of the brief, decided explicitly, not by omission):** exposed -- `task`, `target`, `columns`,
`output`, `device`, `epochs`, `batch-size`, `learning-rate`, `seed`, and (regression only) `target-transform`.
**Not** exposed -- `classes`, `val_fraction`, `missing_columns`, `missing_value`, `patience`, `model=`,
`verbose`. Each of the excluded parameters either has no stable, generically-useful CLI representation
(`model=` is a live Python object; `missing_columns=` is a list of *positional* column indices a CLI caller
would have to compute from the CSV header by hand) or has no demonstrated CLI-specific need yet (`classes=`,
`val_fraction=`, `patience=`, `verbose=` all have well-tested defaults). This mirrors `forge model evaluate`'s
own precedent of exposing a deliberately narrower surface than the Python API and documenting the omissions
rather than papering over every parameter.

`--target-transform standardize` is rejected outright (a `CLIError`, before any file is read) when combined
with `--task classification`. `--json` prints one document built from the returned result's own fields (not
the full dataclass -- `history`/`model` are Python objects, not JSON); text mode prints a short human summary
(samples/features/epochs, then classes+accuracy or MSE+MAE as appropriate).

## 5. Architecture

```text
CSV
 │
 ├── train_tabular_classifier_csv() / train_tabular_regressor_csv()      <- NEW (forge/training/tabular_csv.py)
 │     │
 │     ├── load_csv(..., columns=, return_feature_names=True)             <- UNCHANGED (M118/M120)
 │     │
 │     └── train_tabular_classifier() / train_tabular_regressor()        <- UNCHANGED (M114/M116/M119)
 │           │
 │           ├── validation/preflight, split, preprocessing               <- UNCHANGED
 │           ├── default MLP or caller's model=, training, early stop     <- UNCHANGED
 │           └── train_and_save() -> save_model()                        <- UNCHANGED
 │
 └── forge model train                                                    <- NEW (forge/cli/model.py::cmd_train)
       └── calls the two functions above; no separate CLI training logic
```

No new CSV parser, preprocessing pipeline, training loop, artifact format, feature-schema representation, or
model type exists anywhere in this milestone -- verified directly (§9), not only argued.

## 6. CSV contract (reused unchanged)

`columns=None` (default): every column except `target` is a feature, in file order (M118). `columns=[...]`:
exactly those header columns, in exactly that order; every other column (an `id`, a timestamp) is never read,
never validated (M120). `target` may never appear in `columns` -- rejected by name, never silently dropped
(`load_csv()`'s own `_check_columns_against_header()`, unmodified). Empty/duplicate/unknown `columns` entries
are `load_csv()`'s own `DataError`s, reached through the wrapper with no re-validation. There is still no
automatic id/timestamp/target detection anywhere: selection is exactly as explicit as `load_csv()` already
makes it.

## 7. Feature-schema propagation

The wrapper always calls `load_csv(..., return_feature_names=True)` and threads the result straight into
`feature_names=` on the underlying trainer -- there is no code path through either wrapper that omits it, and
no way for a caller to opt out (an array caller using `train_tabular_classifier()`/`train_tabular_regressor()`
directly still opts in with `feature_names=`, unchanged). `tests/test_train_tabular_csv.py::
test_feature_names_persist_automatically_with_no_feature_names_argument` pins both halves of this: the
wrapper's signature has no `feature_names` parameter at all (`inspect.signature`), and the artifact still
records the selected names. The persisted names are exactly what M119's alignment (`_column_order()`, in
`forge/training/inference.py`, untouched) later checks a reordered CSV or array against --
`test_reordered_feature_only_csv_predicts_identically_after_csv_training` and the real-data script (§14) prove
a CSV-trained artifact participates in the M119 alignment path exactly like an array-trained one.

## 8. Classification workflow

`train_tabular_classifier_csv()` reads the target as **labels** (`load_csv(..., labels=True)`) and calls
`train_tabular_classifier()` unchanged. `tests/test_train_tabular_csv.py::
test_classifier_csv_explicit_columns_excludes_the_id_and_matches_the_manual_pipeline` proves every result
field and every trained parameter (SHA-256 over the reloaded model's named parameters) are identical to the
same `load_csv()` + `train_tabular_classifier(..., feature_names=...)` calls written by hand. A dedicated test
with **text** class labels (`test_classifier_csv_reads_text_class_labels_not_numeric_targets`) exists because a
0/1-integer-coded target cannot distinguish `labels=True` from `labels=False` at all (an integer-valued float
is accepted as a class index either way) -- this was found directly during mutation testing (§19, mutation 4)
and the missing test was added rather than left as a gap.

## 9. Regression workflow

`train_tabular_regressor_csv()` reads the target as **numbers** (`labels=False`) and calls
`train_tabular_regressor()` unchanged, including its default `epochs=500`/`patience=30` (deliberately
different from the classifier's `epochs=100`/`patience=10` -- both wrappers preserve their own underlying
function's defaults rather than picking one shared default, verified by
`test_csv_wrapper_defaults_match_the_array_trainer_defaults`, which compares the two signatures' defaults
directly via `inspect.signature`).

## 10. Target-transform behavior

`target_transform="standardize"` (M116) is passed straight through to `train_tabular_regressor()` with no
scaling logic in this module at all -- training happens on standardized targets fitted on the training split
only, and the fitted `forge.data.StandardizeTarget` is saved in the artifact so `predict()`/`evaluate()` and
every number in the returned result are in native units, exactly as for an array caller.
`test_regressor_csv_target_transform_standardize_persists_and_is_native_units` and the real housing validation
(§14, R² 0.737/0.731) both confirm this end-to-end on real, id-bearing data.

## 11. Artifact compatibility

No format-version change. `feature_names` is the same optional M119 metadata key `save_model()` already
writes; a CSV-trained artifact is byte-for-byte indistinguishable in format from an array-trained one with the
same `feature_names=` given (proven by the parameter-hash equality checks throughout §8/§14, not merely
argued). Existing named and unnamed artifacts, and every M119/M120 prediction/evaluation path, are untouched --
`forge/training/tabular.py`, `forge/training/inference.py`, and `forge/data/csv_reader.py` have zero diff
against pristine `HEAD` (confirmed via `git diff --stat`, §17).

## 12. Test coverage

- `tests/test_train_tabular_csv.py` (22) -- API-level: default/explicit column selection and order, target
  exclusion (rejected, never silently dropped), unknown/duplicate/empty `columns`, automatic feature-name
  persistence with no `feature_names=` parameter on the wrapper at all, reordered-CSV-predicts-identically
  (M119 alignment intact for a CSV-trained artifact), `missing_columns=` sentinel replacement through the CSV
  door, custom `model=` forwarding and contract-checking, `target_transform="standardize"` persistence and
  native units, text-label classification (the mutation-4 finding, §19), defaults-match-the-array-API,
  thin-wrapper verification (no `csv`/`open()` call inside `tabular_csv.py` itself, via `ast`), existing array
  APIs unaffected, and a real Pima id-bearing-CSV end-to-end identity check.
- `tests/test_cli_train.py` (17) -- CLI: classifier/regression training, `--json`, default column selection,
  `--target-transform` interaction and rejection for classification, every required-argument/argparse error,
  invalid target/column-selection errors, non-`.csv` input rejection, missing output directory, and a
  fresh-process consumption check (`forge.load_predictor()` plus a second `model predict` call on the
  CLI-trained artifact).
- `tests/test_train_tabular_csv_cuda.py` (3) -- CUDA: classifier CSV training on the real GeForce 940MX
  (parameter-level accuracy parity with CPU), regressor CSV training with `target_transform="standardize"` on
  CUDA in native units, and `forge model train --device cuda` producing a CUDA-recorded artifact.
- `tests/test_packaging_smoke.py` (+4) -- installed-wheel, fresh-process, outside-the-repository: the M121
  wrapper functions trained directly from a real id-bearing Pima CSV (identical to the M118/M120 array
  pipeline run in the dev-tree process), `forge model train` on the same file matching the Python wrapper's own
  call, a standardized regressor from a realistic-scale id-bearing CSV via both the wrapper and the CLI, and
  clean `Error:`-line-only reporting for three bad invocations. Everything else in this file (29 -> 33 tests;
  M93/M117-M120's own) is unmodified and still passes.

## 13. Mutation results

Seven mutations applied directly to `forge/training/tabular_csv.py`, each confirmed to turn a specific test
(or set of tests) red, then reverted (diffed byte-identical against a pre-mutation backup) before the next:

| # | Mutation | Killed by |
|---|---|---|
| 1 | Ignore `columns=` in the classifier wrapper (always pass `columns=None` to `load_csv()`) | 11 tests, incl. `test_classifier_csv_explicit_columns_excludes_the_id_and_matches_the_manual_pipeline` |
| 2 | Reverse the selected column order before calling `load_csv()` | 6 tests, incl. `test_selected_column_order_is_authoritative_not_file_order` |
| 3 | Drop `feature_names=` forwarding in the classifier wrapper (pass `None`) | 4 tests, incl. `test_feature_names_persist_automatically_with_no_feature_names_argument` |
| 4 | Swap `labels=True` -> `labels=False` in the classifier wrapper (target miscategorized) | **survived** against integer 0/1 labels (both readings accept an integer-valued float as a class index); a new test with **text** class labels was added and confirmed to kill it |
| 5 | Drop `model=` forwarding in the classifier wrapper (`model=None`) | 1 test, `test_custom_model_is_forwarded_and_contract_checked` (the wrong-output-width contract check no longer fires because the custom model is silently dropped) |
| 6 | Drop `target_transform=` forwarding in the regressor wrapper | 2 tests, incl. `test_regressor_csv_target_transform_standardize_persists_and_is_native_units` |
| 7 | Ignore `columns=` in the regressor wrapper | 3 tests, incl. `test_regressor_csv_explicit_columns_matches_the_manual_pipeline` |

Mutation 4 is the one genuine finding of this pass: it exposed that the existing test suite had no case that
could distinguish "read the target as class labels" from "read the target as numbers" when the labels happen
to be 0/1-integer-coded, because `train_tabular_classifier()`'s own label-coercion accepts an integer-valued
float as a class index regardless of which `load_csv()` reading produced it. The fix was not to weaken the
mutation but to add the missing test (`test_classifier_csv_reads_text_class_labels_not_numeric_targets`, using
text class names, which `labels=False` cannot parse as numbers at all) -- confirmed to kill the mutation, then
the mutation was reverted and the full file re-verified green (§20).

All seven mutations were applied and reverted one at a time; every revert was diffed byte-identical against a
saved pre-mutation copy of `forge/training/tabular_csv.py` before moving to the next.

## 14. Real-data validation

`tests/real_world/csv_training_workflows.py` (new acceptance script, mirroring `csv_workflows.py`'s own
structure): an `id`-bearing production-style CSV -- a Pima diabetes file with a synthetic `patient_id` column
prepended, and a StatLib California housing file with a synthetic `parcel_id` column prepended -- trained
directly through `train_tabular_classifier_csv()`/`train_tabular_regressor_csv()` with explicit `columns=`,
compared against the id-free `load_csv()` + array-trainer pipeline, and against `forge model train` run on the
same file.

**Pima diabetes** (768 rows, real, bundled `examples/tabular_diabetes/data/diabetes.csv`, `patient_id` values
`10000..10767` prepended):

```text
wrapper vs manual pipeline: every result field and every trained parameter identical (SHA-256 dabce1a3a514a23f...)
trained on 'cpu': validation accuracy 78.6% vs 67.5% majority baseline
forge model train CLI (no missing_columns=, not exposed on the CLI surface): identical to the same API call
reordered feature-only CSV predicts identically (M119 alignment intact)
```

**California housing** (StatLib `cadata.txt`, 20,640 rows, downloaded fresh for this milestone -- SHA-256
`5e407d03adc03eb6aa34360c2976ee534d8309287dd49e3c22688630bddc9271` -- 3,000 train / 2,000 held-out rows, the
M115/M118/M120 `default_rng(123)` split, `parcel_id` values `900000..` prepended to both files,
`target_transform="standardize"`):

```text
wrapper vs manual pipeline: every result field and every trained parameter identical (SHA-256 a270e22c30ae1b80...)
trained on 'cpu': held-out (2000 rows), native dollars: MSE 3.506e+09, MAE 40501, R^2 0.737
forge model train CLI: identical result to the Python API
id-bearing held-out CSV evaluated through the CLI with --columns: identical to the API's evaluate()
```

Both halves re-run with `--device cuda` on the development machine's real GeForce 940MX:

```text
Pima:    validation accuracy 78.6% vs 67.5% (identical to CPU); parameter-hash identity confirmed cuda-vs-cuda
Housing: held-out MSE 3.585e+09, MAE 41158, R^2 0.731 (CPU 0.737 -- the same CPU/CUDA gap M118's own housing
         run reported, not a new discrepancy)
```

`ALL OK` on both the CPU and the `--device cuda` invocation.

## 15. Fresh-process / installed-wheel validation

`tests/test_packaging_smoke.py`'s new M121 section (§12 above) builds the real wheel (`python -m build`),
installs it into a throwaway venv (`numpy`+`Pillow` only -- no `pandas`, asserted elsewhere in the same file),
and runs every M121 check as a genuine `subprocess` from a directory outside the repository:
`train_tabular_classifier_csv()` on a real id-bearing Pima CSV, `forge model train` on the same file, a
standardized regressor via both the Python wrapper and the CLI, and clean `Error:`-only reporting for three bad
invocations. All 6 new tests pass; the file's other 27 pre-existing tests (M93/M117-M120) are untouched and
still pass -- **33/33** in this file.

## 16. CPU/CUDA status

CPU tests need no CUDA. CUDA was available and confirmed (`forge.cuda.is_cuda_available()` -> `True`, GeForce
940MX) for the full development-machine run; `tests/test_train_tabular_csv_cuda.py`'s 3 tests, and the real-data
script's `--device cuda` invocation (§14), both ran against real hardware -- no CUDA behavior was skipped or
simulated.

## 17. Performance

Not a concern for this milestone: the wrapper adds one Python-level function call and forwards its arguments;
`load_csv()`'s own parsing cost (measured and accepted in M118) is unchanged, and there is no second pass over
the file, no copy of the arrays beyond what `load_csv()` itself already returns, and no duplicate parsing.
`forge/training/tabular.py` (the training pipeline itself) has a zero diff against pristine `HEAD`, so its own
cost is unchanged by definition.

## 18. Files changed

```text
forge/training/tabular_csv.py        NEW  -- train_tabular_classifier_csv() / train_tabular_regressor_csv()
forge/training/__init__.py           +2 lines  -- export the two new functions
forge/__init__.py                    +2 lines export, docstring update  -- top-level re-export + module docstring
forge/cli/model.py                   +docstring, +train subparser, +cmd_train()  -- `forge model train`
docs/development/cli.md              +Model training section, help-list line
docs/development/m121-tabular-csv-training.md   NEW (this report)
docs/development/progress.md         +M121 entry
tests/test_train_tabular_csv.py      NEW (22 tests)
tests/test_cli_train.py              NEW (17 tests)
tests/test_train_tabular_csv_cuda.py NEW (3 tests)
tests/test_packaging_smoke.py        +4 tests (+ 1 import: `csv`)
tests/real_world/csv_training_workflows.py   NEW (real-data acceptance script)
```

`forge/training/tabular.py`, `forge/training/inference.py`, `forge/data/csv_reader.py`,
`forge/data/feature_names.py`, and `forge/serialization/*` have **zero diff** against pristine `HEAD` --
confirmed by `git diff --stat` before writing this report.

## 19. Limitations

- **No new training capability.** `train_tabular_classifier_csv()`/`train_tabular_regressor_csv()` cannot do
  anything `load_csv()` + `train_tabular_classifier()`/`train_tabular_regressor()` could not already do
  together -- this milestone is convenience, not capability, exactly as designed.
- **CSV-only training door.** There is no way to train from an in-memory array through these two functions (by
  design -- that remains `train_tabular_classifier()`/`train_tabular_regressor()` directly) and no way to train
  from any file format but `.csv` through `forge model train`.
- **The CLI surface is deliberately narrower than the Python API** (§4) -- `classes`, `val_fraction`,
  `missing_columns`, `missing_value`, `patience`, `model=`, `verbose` are not exposed; a workflow that needs one
  of them uses the Python wrapper (or the array API) directly.
- **No automatic column-kind detection anywhere** (inherited from M118/M120, unchanged): an unselected `id`
  column is simply not read, never recognized as "the kind of thing that should be excluded."
- **Mutation coverage is targeted (7 mutations), not exhaustive** -- a deliberate scope decision consistent
  with M120's own precedent, not a gap discovered and left open. One mutation (4) initially survived and was
  used to find and fill a genuine test gap (text-label classification) rather than being weakened or ignored.

## 20. Deferred work

`forge model train` exposing `missing_columns=`/`model=` if a real workload demands it; a `--drop` inverse of
`--columns` (still not demonstrated, per M120's own deferral); training from formats other than `.csv`; a full
exhaustive mutation matrix for this milestone's new code, if a future change to `tabular_csv.py` raises the
value of one. None has a demonstrated workload yet -- consistent with the guiding principle closing this
report.

## 21. Full-suite result

Baseline (pristine `HEAD`, matching M120's own reported figure): **3,838 passed, 0 failed, 0 skipped.**
(A first attempt to measure this baseline via `git stash` was contaminated by this milestone's own untracked
new test files, which `git stash` does not touch without `-u`, and showed 3,880 -- not a real discrepancy, just
a measurement artifact; 3,880 minus this milestone's 42 array/CLI/CUDA tests is exactly 3,838.)

`python -m pytest tests/ -q` after this milestone's changes, on the development machine (i5-7200U, GeForce
940MX, CUDA 12.6, Python 3.13.5):

```text
3884 passed in 727.91s (0:12:07)
```

**3,838 -> 3,884 passed, 0 failed, 0 skipped** -- exactly the 46 new tests (22 API + 17 CLI + 3 CUDA + 4
installed-wheel). CUDA was available throughout (`forge.cuda.is_cuda_available()` -> `True`, GeForce 940MX) and
no CUDA test was skipped or simulated.

## 22. Acceptance criteria

- CSV classifier training is first-class: `forge.train_tabular_classifier_csv()`. **Met.**
- CSV regressor training is first-class: `forge.train_tabular_regressor_csv()`. **Met.**
- Existing M118 column semantics reused unchanged. **Met** (`forge/data/csv_reader.py` zero diff).
- Existing M119 feature-schema semantics reused unchanged. **Met** (§7, §11).
- Existing M116 target-transform semantics reused unchanged. **Met** (§10).
- Selected feature names automatically persisted. **Met** (§7, pinned by test).
- No duplicate CSV/training implementation. **Met** (§5, §17 zero-diff, and a direct `ast`-based test that
  `tabular_csv.py` contains no `csv`/`open()` call of its own).
- Existing array APIs remain compatible. **Met** (`forge/training/tabular.py` zero diff; a dedicated test
  checks the array functions' own signatures are unaffected).
- Real Pima workflow passes. **Met** (§14, CPU and CUDA).
- Real California Housing workflow passes. **Met** (§14, CPU and CUDA, R² 0.737/0.731).
- Reordered inference passes. **Met** (§7, §14).
- Fresh-process consumption passes. **Met** (`tests/test_cli_train.py`'s fresh-predictor-load test).
- Installed-wheel validation passes. **Met** (§15, 6/6 new + 33/33 total in the packaging suite).
- CPU passes. **Met.**
- CUDA passes where applicable. **Met** (§16).
- Targeted mutations pass. **Met** (§13, 7/7 killed after one gap was found and closed).
- Full test suite passes with 0 failures and 0 skips. **Met** (§21: 3,838 -> 3,884, 0 failed, 0 skipped).
- Documentation updated. **Met** (this report, `docs/development/cli.md`, `docs/development/progress.md`,
  `forge/__init__.py`/`forge/cli/model.py` docstrings).
- No unnecessary artifact-format bump. **Met** (§11, no format-version change).

## 23. Final disposition

```text
Decision: IMPLEMENTED

Summary:
- forge.train_tabular_classifier_csv() / forge.train_tabular_regressor_csv() (forge/training/tabular_csv.py):
  two thin wrappers, each exactly load_csv(..., columns=, return_feature_names=True) followed by the unmodified
  train_tabular_classifier()/train_tabular_regressor()(..., feature_names=names). No new validation, training,
  persistence or feature-schema logic anywhere.
- forge model train DATA.csv --task ... --target ... --output ...: a thin CLI adapter over the two functions
  above, with a deliberately scoped flag surface (task/target/columns/output/device/epochs/batch-size/
  learning-rate/seed/target-transform), documented explicitly rather than exposing every Python-API parameter.
- Feature names are persisted automatically -- neither wrapper has a feature_names= parameter at all, and there
  is no way to omit the artifact's schema when training through this door.
- Existing array APIs (train_tabular_classifier/regressor), the M118 CSV reader, the M119 alignment mechanism
  and the M116 target-transform mechanism are all unmodified (verified: zero diff against pristine HEAD).

Tests:
- baseline: 3,838 passed, 0 failed, 0 skipped (pristine HEAD, matching M120's own reported figure)
- final: 3,884 passed, 0 failed, 0 skipped (727.91s / 12m07s, this development machine)
- new: 46 (22 API + 17 CLI + 3 CUDA + 4 installed-wheel)
- failures: 0
- skipped: 0 (CUDA available and exercised throughout)

Mutations: 7 applied and reverted one at a time (byte-diff-confirmed); all 7 killed. One (labels=True/False
  confused in the classifier wrapper) initially survived against integer-coded labels -- closed by adding a
  text-class-label test, not by weakening the mutation.

Real datasets:
- Pima diabetes (768 rows, real, bundled), id-bearing CSV: CPU validation accuracy 78.6% vs 67.5% baseline;
  CUDA identical. Wrapper-vs-manual-pipeline parameter hashes identical; CLI-vs-wrapper results identical.
- California housing (StatLib, 20,640 rows, freshly downloaded, SHA-256 5e407d03...bd9271), id-bearing CSV,
  target_transform=standardize: CPU held-out R^2 0.737 (MSE 3.506e9, MAE $40,501); CUDA R^2 0.731 (the same
  CPU/CUDA gap M118 already reported, not a new discrepancy). Wrapper-vs-manual and CLI-vs-wrapper identical.

Installed-wheel status: 6 new tests, all passing, inside the real built wheel, clean venv (no pandas), outside
  the repository, fresh subprocess per invocation; the file's other 27 pre-existing tests (M93/M117-M120)
  remain green (33/33 total).

CPU/CUDA status: CPU tests need no CUDA; CUDA available and exercised (GeForce 940MX) throughout, including the
  real-data acceptance script's --device cuda run; no CUDA test skipped or simulated.

Compatibility status: no artifact-format change; forge/training/tabular.py, forge/training/inference.py,
  forge/data/csv_reader.py and forge/data/feature_names.py have zero diff against pristine HEAD.

Known limitations: CSV-only training door (by design); CLI surface deliberately narrower than the Python API
  (classes=/val_fraction=/missing_columns=/missing_value=/patience=/model=/verbose= not exposed); no automatic
  column-kind detection anywhere (inherited, unchanged); mutation coverage targeted (7), not exhaustive.

Concrete trigger for next work: a real workload needing one of the CLI's currently-unexposed parameters
  (missing_columns=/model=), or --drop as the inverse of --columns (still undemonstrated, per M120's own
  deferral).
```
