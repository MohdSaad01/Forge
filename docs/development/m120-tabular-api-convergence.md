# M120 — Tabular API convergence: explicit column selection and feature-name retrofit

Status: **IMPLEMENTED.** M119 closed the *identity* half of the tabular contract (an artifact can know
*which* feature each input column is). M120 closes the one concrete gap M119 deliberately left open and
documented as deferred: a real CSV that carries a column that is not a model feature at all (an `id`, say)
still had to be removed from the file by hand before Forge would read it. The fix is narrow and additive:
one new keyword (`columns=`) on the two existing CSV readers, one new CLI flag (`--columns`) shared
identically by `predict` and `evaluate`, and one narrow, safety-checked retrofit operation
(`forge model convert --feature-names`) for an existing artifact that has no names to attach.

```text
CSV with an id column
        │
   columns=[...]                  <- Milestone 120: explicit selection (this milestone)
        │
  named feature matrix (X, names)
        │
  feature_names=names             <- Milestone 119: identity, unchanged
        │
   portable artifact
        │
  named CSV (with or without the id column, --columns as needed)
        │
  _column_order() alignment       <- Milestone 119: unchanged
        │
  predict() / evaluate()
```

## 1. Current tabular API inventory (Phase 1, read before changing anything)

| Piece | Behaviour at HEAD (`22e8a34`, before this milestone) |
|---|---|
| `forge.data.load_csv(path, *, target, labels=False, return_feature_names=False)` | every non-target column is a feature, in file order; no way to exclude one |
| `forge.data.load_csv_features(path)` | every column is a feature, in file order; no `target=`, no selection |
| `train_tabular_classifier/regressor(X, y, ..., feature_names=None)` | arrays in, optional identity metadata out (M119); untouched by this milestone |
| `train_and_save()` / `save_and_verify()` | pass `feature_names=` through unmodified; untouched |
| `ArtifactPredictor.predict()/.evaluate()` | `feature_names=` aligns named input to the artifact's recorded names (`_column_order()`); untouched |
| `forge model predict` | `.csv` (feature-only, `load_csv_features()`) or JSON; forwards CSV header names as `feature_names=` |
| `forge model evaluate` | `.csv` (`--target COLUMN`, `load_csv(..., return_feature_names=True)`), `.npy`, or an image directory |
| `forge model inspect` | reports `input_feature_names` (list or `null`) from `InputSchema.feature_names` |
| `forge model convert` | reloads + re-saves, preserving `preprocessing`/`classes`/`task`/`target_transform`/`feature_names` unchanged; no way to *add* names to an artifact that has none |
| Artifact format | version 2 (3 only with a target transform); `feature_names` is an optional metadata key, no format bump (M119) |

Nothing here was assumed from the M119 report without being re-read from the actual `HEAD` source
(`forge/data/csv_reader.py`, `forge/data/feature_names.py`, `forge/serialization/model.py`,
`forge/training/{tabular,inference}.py`, `forge/cli/model.py`) before any change was made.

## 2. M119's remaining friction (Phase 1)

M119's own report names the gap directly (§14, §23): *"an `id` column is an unexpected column and is
rejected; nothing is special-cased and nothing is dropped"*, with the explicit deferred trigger *"a real
workload where the id/target column is legitimately present in the data file and removing it by hand is
the friction"*. Section 3 below reproduces that friction on real data and confirms it is exactly what M119
predicted -- not a new failure mode, and not something that needed guessing at.

The API-surface list in the brief (`load_csv()`, `load_csv_features()`, `train_tabular_*()`,
`ArtifactPredictor`, `forge model predict/evaluate/inspect/convert`) was re-inspected in full; the *only*
piece of it with a genuine, reproducible gap is the CSV reader's all-or-nothing column contract. Training,
prediction, evaluation and inspection already compose correctly around named arrays (M119); nothing there
needed to change.

## 3. Real-world reproduction

A real 768-row Pima CSV (`examples/tabular_diabetes/data/diabetes.csv`) with a `patient_id` column
prepended (values `10000..10767`, plausible synthetic patient IDs) and a real 20,640-row StatLib California
housing CSV (`docs`/M118's recipe) with a `parcel_id` column prepended (`900000..920639`), both realistic
production-style files:

```text
patient_id,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
10000,6,148,72,35,0,33.6,0.627,50,1
...
```

Before this milestone, both are unreadable by `load_csv()`/`forge model evaluate` without first deleting
the id column from the file (an integer id column happens to parse as a "feature" and silently reach the
model at the wrong width or, worse, at the *right* width if a real feature was miscounted -- exactly the
M119 §3/§16/§17 failure mode; a non-numeric id, tested below, fails outright with `is not a number`).

With `columns=`, both work unmodified:

```text
$ python -m forge model evaluate diabetes_m120.forge diabetes_with_id.csv --target Outcome
Error: ArtifactPredictor.evaluate() the input's columns do not match this artifact's feature names --
  unexpected (the input has them, the artifact does not): ['patient_id']. ...
$ python -m forge model evaluate diabetes_m120.forge diabetes_with_id.csv --target Outcome \
    --columns Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age
Task: tabular_classification
Samples: 768
Accuracy: 78.39%
Baseline accuracy: 65.10%
```

California housing (`target_transform="standardize"`, id-column CSV, 2,000-row held-out set), evaluated
through the real CLI, `--columns` in the *training* order and again in a *different* order (selection then
M119 alignment composing):

```text
$ python -m forge model evaluate housing_m120.forge housing_test_with_id.csv --target median_house_value
Error: ... unexpected ...: ['parcel_id']. ...
$ python -m forge model evaluate housing_m120.forge housing_test_with_id.csv --target median_house_value \
    --columns median_income housing_median_age total_rooms total_bedrooms population households latitude longitude
MSE: 3.5055e+09   MAE: 40501.1   Baseline MSE: 1.33322e+10
$ python -m forge model evaluate housing_m120.forge housing_test_with_id.csv --target median_house_value \
    --columns longitude latitude households population total_bedrooms total_rooms housing_median_age median_income
MSE: 3.5055e+09   MAE: 40501.1   Baseline MSE: 1.33322e+10    # identical -- selection order is irrelevant once aligned
```

Confirms the brief's premise: the friction is real, the fix is exactly "explicit column selection," and it
composes with M119 alignment rather than duplicating it.

## 4. Column-selection design

`forge.data.load_csv(path, *, target, labels=False, return_feature_names=False, columns=None)` and
`forge.data.load_csv_features(path, *, columns=None)`. `columns=None` (default) is the pre-M120 contract,
byte-for-byte: every column except `target` is a feature, in file order. `columns=[...]` (a non-empty
list/tuple of distinct, non-blank header names -- **the exact same shape** `feature_names=`
already requires, via the existing `forge.data.feature_names.validate_feature_names()`, reused rather than
reinvented, with the error message reworded for `columns` rather than `feature_names`) selects **exactly**
those columns as features, **in exactly that order**, and every other column is never read: not parsed,
not validated, not even required to be numeric. A single `_read_csv()` code path handles both cases via one
`feature_indices` list (the header positions of the columns to read) -- there is no second reader and no
branch duplicating the row-parsing loop.

Rules (all from the brief, all enforced, each with its own test):

| Rule | Enforced by |
|---|---|
| order is authoritative (`columns` order, not file order) | `feature_indices` built from `columns`, not `names` |
| duplicates rejected | `validate_feature_names()` (shared with `feature_names=`) |
| unknown column rejected, named | `_check_columns_against_header()` |
| target in `columns` rejected, not dropped | `_check_columns_against_header()` |
| empty selection rejected | `validate_feature_names()` |
| selecting every non-target column works, no special flag | tested directly; identical arrays to the unselected default |
| no automatic id/timestamp/target detection | `columns=None` is the only default; nothing is ever inferred from a name |

## 5. API design

No new function. `load_csv()`/`load_csv_features()` were extended, not duplicated -- the brief explicitly
warned against `load_csv_columns()`/`load_csv_selected_features()` and there was no genuine semantic
distinction to justify one: selection is one more thing the *existing* reader can be asked to do,
symmetrical with `target=` (which one column to remove) and `return_feature_names=` (whether to also return
the resulting names). `train_tabular_classifier()`/`train_tabular_regressor()` were **not** touched --
still `(X, y, ..., feature_names=None)`, arrays only, exactly as the brief required; the end-to-end Python
workflow is:

```python
X, y, names = forge.data.load_csv(
    "diabetes.csv", target="Outcome", labels=True,
    columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    return_feature_names=True,
)
forge.train_tabular_classifier(X, y, path="diabetes.forge", feature_names=names, classes=["no_diabetes", "diabetes"])
```

`return_feature_names=True` still opts in to getting `names` back (unchanged default); when `columns=` is
given, the returned names are exactly `columns`, from the same read.

## 6. CLI design

`forge model evaluate MODEL DATA.csv --target COLUMN --columns NAME [NAME ...]` and `forge model predict
MODEL DATA.csv --columns NAME [NAME ...]` -- identical flag, identical semantics, on both commands, because
neither implements column selection itself: both forward `args.columns` to `load_csv()`/
`load_csv_features()` and nothing else changes downstream. `--columns` is rejected (a clear `CLIError`,
never silently ignored) for a JSON `predict` input, a `.npy`/image `evaluate` input, and for a `predict`
task with no CSV input at all (classification/segmentation/sequence) -- there is nothing to select from in
any of those. Errors distinguish "unknown column" (names it), "duplicate requested column" (M119's
existing wording, reused), and "missing artifact feature" (M119's existing `_column_order()` message,
unmodified) -- no internal function name leaks into a normal CLI error (verified: every new rejection
above is a plain `Error: ...` line, not a traceback).

## 7. Feature-name integration / M119 alignment integration

Deliberately kept as **two separate steps in one pipeline**, per the brief's explicit instruction not to
fuse them: `columns=`/`--columns` decides *which raw file columns become `X`*, entirely inside the CSV
reader, with no knowledge of any artifact; `_column_order()` (`forge/training/inference.py`, unmodified by
this milestone) decides *what order the artifact needs them in*, entirely inside the prediction/evaluation
path, with no knowledge of any file. The CLI wires them together by calling the reader first and handing
its output to the (unmodified) named-input path -- it does not itself reorder, drop or rename anything.
Proven, not just designed: California housing above, selected in two different orders, scores identically;
a Pima artifact trained by selecting `columns=` in training order and then queried through a feature-only
CSV selected in the *reverse* order predicts bit-identically to the correctly-ordered array call
(`tests/test_csv_column_selection_training.py::test_a_selected_then_reordered_csv_still_predicts_correctly_m119_alignment_intact`).

## 8. Training behaviour

Unmodified. `train_tabular_classifier()`/`train_tabular_regressor()` still take `(X, y, feature_names=)`
arrays; `_validate_feature_names()`, `preflight_save()`, the split/preprocessing pipeline and
`train_and_save()` are untouched by this milestone. What changed is only how `X`/`y`/`names` are produced
from a file *before* that call -- proven end-to-end (Pima classification, California regression with
`target_transform="standardize"`, §3 and §9) that a selected-column `X` trains, saves and evaluates exactly
like any other array-sourced `X`: `result.features` equals the selected column count, not the file's; the
id column never reaches `_coerce_features()` (checked directly: `Xs.shape[1] == 4` in the small synthetic
fixture, `== 8` on real Pima).

## 9. Prediction / 10. Evaluation behaviour

Both converge on the unmodified M119 path once the CSV boundary has produced `(X, names)`/`(X, y, names)`.
Tested directly, real and synthetic data: named CSV (ordered / selected / selected-and-reordered), an id
column present without `--columns` (rejected, `'id'`/`'patient_id'`/`'parcel_id'` named), `.npy` unchanged
(rejected for `--columns`), JSON unchanged (rejected for `--columns`). `forge model evaluate`'s JSON output
for a selection-based call is byte-identical to the same rows read from an id-free file
(`test_installed_cli_evaluate_columns_excludes_the_id_and_matches_the_correct_csv`, installed wheel).

## 11. Target-transform interaction (M116)

Selection happens entirely inside the CSV reader, before `train_tabular_regressor()` (or `predict()`/
`evaluate()`) ever sees the array -- `StandardizeTarget` is fitted/applied downstream of it and never
touches column identity. Verified on California housing (§3: native-unit MSE/MAE identical across
selection orders) and with a synthetic fixture
(`test_target_transform_composes_with_selection_native_units`): predictions from a selected-column artifact
remain in native units, exactly as an array-sourced one would.

## 12. ID-column behaviour

No automatic detection exists anywhere in this milestone, by design and by test: omitting `columns=`
still treats every non-target column as a feature (`test_omitting_columns_still_treats_every_non_target_column_as_a_feature`),
so a numeric `id` column silently becomes a (wrong) feature exactly as before M120, and a non-numeric one
still fails with `is not a number` (`test_a_non_numeric_id_column_is_rejected_without_explicit_selection`).
Only an *explicit* `columns=`/`--columns` excludes it -- and does so unconditionally, whether the excluded
column is numeric or not (`test_an_unselected_column_is_ignored_even_if_non_numeric`), because it is simply
never read.

## 13. Artifact-retrofit investigation

M119 §23 named this as a deferred concrete trigger: *"a user who must add names to an artifact they cannot
retrain."* Investigated against the existing conversion/persistence architecture rather than built
speculatively: `forge model convert` already reloads a model, its preprocessing, its classes, its task and
its target transform, and re-saves all of them unchanged via `save_model()` -- the exact "load once, write
back with one field different" shape a safe retrofit needs, with `save_model()` itself already enforcing
every M119 safety rule on `feature_names=` (task must be tabular, count must equal the model's actual input
width read from its architecture, names must be `validate_feature_names()`-valid). Implemented as
`--feature-names NAME [NAME ...]` on the existing command -- no new command, no new persistence path.

**Why this is safe by construction, not merely by testing:** `convert` was already exactly "reload the
model's weights, re-save the model's weights, plus whatever metadata is passed to `save_model()`." Adding
one more optional metadata argument to that existing call cannot touch the weights that are reloaded and
re-saved either side of it -- there is no code path in `cmd_convert()` that reads `--feature-names` before
the model is loaded or writes it into anything but the metadata dict `save_model()` receives. Verified
directly (not only argued): the converted file's per-parameter archive entries are SHA-256-identical to the
source file's, and `predict()` on a batch of random rows returns bit-identical output before and after
retrofit, both manually (`diabetes_unnamed.forge` vs `diabetes_retrofit.forge`, real Pima data) and by test
(`test_convert_feature_names_changes_no_weights_and_no_predictions`,
`test_installed_cli_convert_feature_names_retrofit_changes_no_weights` in the installed wheel).

## 14. Compatibility behaviour

No format change. `--feature-names` writes through the same `save_model(..., feature_names=...)` call M119
already made safe (optional key, format version 2, bumped to 3 only alongside a target transform); retrofit
produces an artifact indistinguishable in format from one that was *trained* with `feature_names=`. Omitting
`--feature-names` on `convert` is unchanged from before this milestone: whatever names (or lack of them) the
source artifact had are carried over as-is (`test_convert_without_feature_names_keeps_whatever_the_artifact_already_had`).
`columns=`/`--columns` add no new artifact metadata and no new file format at all -- they are pure input
readers, so there is nothing new to be compatible or incompatible with; an M120 build's `load_csv()` with no
`columns=` reads a file exactly as an M119/M118 build would (`test_selecting_all_columns_in_file_order_matches_the_unselected_default`,
and the two pre-existing pinned-argument CLI tests updated below).

## 15. Pima validation / 16. California Housing validation / 17. M116 target-transform validation

Covered together in §3 (real-data CLI reproduction), §8-11 (end-to-end training/prediction/evaluation
tests) and §12 (target-transform native units on California housing). Both real datasets: an artifact
trained via `columns=` on a real id-bearing CSV scores identically to one trained on the same rows without
an id column present at all, through both the Python API and the installed-wheel CLI.

## 18. External consumer validation

`tests/test_packaging_smoke.py` (M93's real-wheel/clean-venv/outside-repository harness, reused, not
duplicated) gained three tests, appended to its existing Milestone-119 section rather than building a new
fixture: `test_installed_cli_evaluate_columns_excludes_the_id_and_matches_the_correct_csv`,
`test_installed_cli_predict_columns_excludes_the_id`, and
`test_installed_cli_convert_feature_names_retrofit_changes_no_weights` (parameter-hash equality proven
inside the actual installed wheel, not just in the dev tree). All three reuse the existing `named_pima`
fixture (a real Pima artifact trained from a header-named CSV, in the clean venv, in a fresh process) and
its already-written `extra_id`/`features_only` CSV layouts -- `--columns` is a strictly narrower reading of
CSV files M119's own fixture already produced, so no new fixture data was needed. Confirmed passing inside
the actual built wheel (`python -m build`), in a venv containing only `forge`+`numpy`+`Pillow` (`pandas`
absence already asserted by `test_the_consumer_environment_has_no_dataframe_library`, unmodified), from a
directory outside the repository, each CLI call a fresh process. (Run alongside, not interleaved with, the
full CPU/CUDA suite below, to avoid the load-induced timing sensitivity the CUDA suite has shown before --
see `docs/development/m95-*`/testing-conventions memory.)

## 19. Performance

Not optimised (no caching, no schema object). Measured on the development machine (i5-7200U): reading the
3,000-row California-housing training CSV with `columns=` selecting 8 of 9 columns is **not slower** than
reading it with no selection (99.4 ms vs 111.2 ms, best-of-20) -- selection does strictly less parsing work
(one column's cells are never even touched), so there is no overhead to amortise. The one-time
`_validate_columns_argument()`/`_check_columns_against_header()` cost is the same order of magnitude as
M119's already-measured `_column_order()` name-matching (microseconds for a handful of names), dwarfed by
CSV parsing and completely dwarfed by training/inference. The retrofit path (`convert --feature-names`) does
one extra `save_model()` metadata write, already paid by `convert` on every call; no additional cost was
introduced.

## 20. Tests

53 new tests in three new files plus three appended to the existing packaging-smoke suite:

- `tests/test_csv_column_selection.py` (25) -- the CSV reader in isolation: selection, order-authoritative,
  ignored/non-numeric unselected columns, all rejections (duplicate, unknown, target-overlap, empty,
  non-list, blank name), `load_csv_features(..., columns=)`, labels, missing-value interaction.
- `tests/test_csv_column_selection_training.py` (5) -- the real end-to-end workflow: names reach the
  trained artifact, the id column never reaches training, a selected-then-reordered CSV still predicts
  correctly (M119 alignment intact), `target_transform="standardize"` composes, classification workflow.
- `tests/test_cli_column_selection.py` (20) -- `--columns` on `evaluate`/`predict` (id exclusion, order
  authoritative, duplicate/unknown/target-overlap rejection, rejected for `.npy`/JSON/no-CSV-task), `convert
  --feature-names` (attach, wrong count refused and writes nothing, duplicate refused, weight/prediction
  parity, default behaviour unchanged, replacing existing names).
- `tests/test_packaging_smoke.py` (+3) -- the same `--columns`/`--feature-names` behaviour inside the real
  installed wheel, outside the repository (§18).

**Existing tests updated: two assertions, both a genuine, minimal contract change** (the same discipline
M119 followed): `test_load_csv_features_has_no_target_argument_and_no_labels` now expects the parameter list
`["path", "columns"]` (it added a parameter; it still has no `target=`/`labels=`); `test_cli_evaluate_csv.py`'s
pinned-call assertion for the CLI's `load_csv(...)` invocation now includes `"columns": None` (the CLI now
always forwards `columns=` -- `None` when `--columns` was not given -- exactly the same "no keyword drifted
silently" pattern M119's own sibling assertion already established). No other existing test changed.

## 21. Mutation results

Not a full independent mutation-testing pass on the scale of M119's 39 (schedule/effort tradeoff for this
milestone, documented rather than hidden). Three targeted mutations were applied directly to
`forge/data/csv_reader.py` and each confirmed to turn a specific new test red before being reverted:

| Mutation | Killed by |
|---|---|
| target-overlap check disabled (`columns` containing `target` silently accepted) | `test_target_listed_in_columns_is_rejected_not_silently_dropped`, `test_evaluate_columns_including_the_target_is_rejected` |
| order-authoritative broken (`feature_indices` sorted into header order instead of `columns` order) | `test_order_is_authoritative_not_file_order`, `test_load_csv_features_columns_selects_and_orders` |
| unknown-column check disabled (`unknown = []`) | `test_unknown_column_is_rejected_and_named`, `test_load_csv_features_unknown_column_is_rejected` |

All three were confirmed reverted (file hash-compared against the pre-mutation backup) and the full
targeted test set re-run green before proceeding. The retrofit safety property (§13) was verified directly
rather than by mutation: SHA-256 parameter-file equality and bit-identical `predict()` output, both in the
dev tree and inside the installed wheel.

## 22. Full-suite result

Baseline (M119): `3,785 passed, 0 failed, 0 skipped` (CUDA available, GeForce 940MX).

`python -m pytest tests/ -q` after this milestone's changes, on the development machine (i5-7200U,
GeForce 940MX, CUDA 12.6, Python 3.13.5):

```text
3838 passed in 662.25s (0:11:02)
```

**3,838 passed, 0 failed, 0 skipped** -- 3,785 -> 3,838, exactly the 53 new tests (§20); every pre-existing
test still passes unmodified except the two pinned-call assertions updated for the documented contract
change. CUDA was available throughout (`forge.cuda.is_cuda_available()` confirmed `True` on this machine)
and no CUDA test was skipped or simulated. (Run in two passes for practical reasons, not two different
results: the main run was started before `tests/test_packaging_smoke.py` gained its three new M120 tests
--pytest collects once at process start, so a file edited after a background run has begun is not picked
up by that run -- so a second, complete run was made after those tests were added, and it is the number
reported here; the first run's 3,835/3,835 is consistent with it, differing by exactly those 3 tests.)

## 23. Limitations

- **Column selection has no automatic detection of anything** -- an `id`/timestamp/target column left out
  of `columns=` is simply a column that was not selected, not a recognised "kind" of column; the brief
  required this and it is enforced by test (§12).
- **Selection is per-file, not per-artifact.** `columns=`/`--columns` select from the *file*; they carry no
  awareness of what an artifact expects (that remains `_column_order()`'s job, unchanged). A `columns=` list
  that selects the wrong set of columns for a given artifact still fails at the (unchanged) M119 alignment
  step, with M119's own error text.
- **Retrofit cannot discover names** -- `--feature-names` only attaches names the caller already knows and
  supplies explicitly; there remains no way to guess them from a CSV header (deliberately, per the brief).
- **Retrofit requires a readable input width**, exactly as M119's own `feature_names=` did: an artifact
  whose architecture is not a `Sequential` starting with `Linear`, or whose task is not
  regression/tabular_classification, cannot have names attached (`save_model()`'s existing check, unchanged).
- **Missing-value and categorical-feature behaviour are unchanged** -- selection does not enable imputation
  (an empty cell in a *selected* column is still rejected; an unselected column's missing values are simply
  never looked at, tested directly) and a selected string-valued column is still rejected as non-numeric,
  exactly as the brief required.
- **Mutation coverage of the new code is targeted (3 mutations), not exhaustive** (§21) -- a deliberate
  scope decision for this milestone, not a gap discovered and left open.

## 24. Deferred work

`--drop COLUMNS` (the inverse of `--columns`: name what to exclude rather than what to keep) -- no
demonstrated workload needed it over `--columns` itself, and the brief explicitly favoured explicit
selection; addressing `missing_columns=` by name instead of position; recording the *target* column's name
in the artifact; a full 39-style mutation matrix for this milestone's new code, if a future change to the
same files raises the value of one; automatic inference of `--feature-names` from a CSV header on `convert`
(explicitly ruled out by the brief). None has a demonstrated workload yet.

## 25. Final decision

**IMPLEMENTED.** The friction M119 identified and deferred -- a real CSV's non-feature column forcing a
file rewrite -- is real (§3, reproduced on two real datasets) and is closed by the smallest addition that
does not duplicate the M119 alignment mechanism: one new reader keyword, one new identical CLI flag on both
`predict` and `evaluate`, and one safety-checked retrofit operation riding on `convert`'s existing
reload/re-save. No DataFrame, no schema framework, no categorical/missing-value handling, no new public
type, and no format-version change were needed or added.

```text
Decision: IMPLEMENTED
```

```text
Framework changes: none beyond the CSV/CLI layers below -- train_tabular_classifier/regressor, save_model,
  InputSchema, _column_order() and every M119 prediction/evaluation path are unmodified.
CSV changes: forge.data.load_csv(..., columns=None) and load_csv_features(path, *, columns=None) (Milestone 120):
  explicit, order-authoritative column selection sharing validate_feature_names(); one _read_csv() path for both
  the default and the selected case via a single feature_indices list; unselected columns are never parsed,
  never validated.
Training changes: none. Selected arrays reach train_tabular_classifier/regressor exactly as any other array does.
Prediction changes: none to the alignment mechanism. forge model predict gained --columns, sharing
  _read_numeric_input() -> load_csv_features(..., columns=).
Evaluation changes: none to the alignment mechanism. forge model evaluate gained --columns, sharing the same
  load_csv(..., columns=) call already used for --target.
Artifact changes: forge model convert gained --feature-names (retrofit), riding on its existing reload/re-save;
  no format change; safety enforced by the existing save_model() count/task checks.
Tests added: 53 (25 CSV reader, 5 end-to-end training, 20 CLI, 3 installed-wheel); 2 existing pinned-call
  assertions updated for a genuine, minimal contract change (both documented in Section 20).
Full-suite result: 3,838 passed, 0 failed, 0 skipped (baseline 3,785; CUDA available, GeForce 940MX; 11m02s)
Pima result: a real id-bearing CSV, unreadable without --columns, evaluates/predicts identically to the
  id-free file once --columns is given (accuracy 78.39% vs 65.10% baseline on the 768-row set).
Regression result: California housing (target_transform=standardize), id-bearing CSV: MSE 3.5055e9 / MAE 40501
  identical regardless of --columns order, matching the id-free file.
M116 target-transform result: native units preserved through selection on both real data and a synthetic
  fixture; selection never observes the transform.
External consumer status: three new tests pass inside the real built wheel, clean venv (no pandas), outside
  the repository, fresh process per CLI call, reusing the existing Milestone-119 fixture's CSV layouts.
CPU/CUDA status: CPU tests need no CUDA; CUDA available and confirmed (GeForce 940MX) for the full run; no CUDA
  test skipped or simulated. No CUDA-specific code was added by this milestone (the new logic is CSV-text and
  CLI-argument handling only), so no new CUDA-specific test was needed.
Compatibility status: no format change; columns=/--columns are pure input readers with no persisted trace;
  --feature-names writes through the unmodified Milestone-119 save_model() path; omitting either flag is
  byte-identical to before this milestone.
Remaining limitations: no automatic column-kind detection (by design); retrofit cannot discover names from a
  CSV; retrofit needs a Linear-first architecture and a tabular task, as M119's own feature_names= already
  required; targeted (not exhaustive) mutation coverage of the new code.
Concrete trigger for next work: a real workload needing --drop (name what to exclude) over --columns (name
  what to keep), or a workload needing missing_columns= addressed by name instead of position.
```
