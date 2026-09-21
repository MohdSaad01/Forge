# M119 — Persisted tabular feature schema

Status: **IMPLEMENTED.** A saved tabular model can now know *which* feature each input column is, not only how many
there are. The smallest thing that provides that guarantee: one optional metadata key, one optional keyword on the
existing functions, one rule in one place. No new tabular type, no DataFrame, no format-version change, and every
existing artifact and every NumPy call behaves exactly as before.

```text
CSV header ─► load_csv(..., return_feature_names=True) ─► X, y, names ─► train_tabular_*(..., feature_names=names)
                                                                              │
                                                              artifact records ["Pregnancies","Glucose",...]   (raw columns, in model order)
                                                                              │
named CSV ─► load_csv_features() / load_csv() ─► X, names ─► predict(X, feature_names=names) / evaluate(...)
                                                                              │
                              same names, any order ─► reorder to the artifact's order ─► preprocessing ─► model (one path)
                              anything else         ─► DataError naming the missing / unexpected columns
bare NumPy array / .npy / JSON ─► no names exist ─► width check only, exactly as before
```

## 1. Problem

Feature identity is part of a tabular model's input contract. An artifact that records `feature_count = 8` cannot tell
`Pregnancies,Glucose,...,Age` from `Glucose,Pregnancies,...,Age`: the shape is right, the model runs, and the answer is
wrong with no error. The distinction the milestone is built on:

- **shape correctness** — the input has the width the model takes. Forge has checked this since M101.
- **semantic correctness** — each value is in the column the model was trained to read it in. Nothing checked this.

A caller who passes an already-ordered NumPy array has made a claim Forge cannot see and is not blamed for it. The defect
is specifically about **named** tabular data — a CSV whose header *says* what each column is — and an artifact's inability
to use that information.

## 2. The M118 limitation

M118 (`forge.data.load_csv`, `forge model evaluate DATA.csv --target`) preserved CSV column order into the model input
exactly, and recorded the consequence as a limitation: *"artifacts persist a feature count only, so reordered CSV columns
are undetectable"*. That statement is accurate at HEAD and is the starting point, verified below rather than assumed.

## 3. Real-world reproduction (pristine HEAD `c0d580b`, before any change)

Real Pima artifact (`train_tabular_classifier` on the 768-row `diabetes.csv`, `missing_columns=[1..5]`, seed 0; validation
accuracy 76.6% against a 67.5% baseline), evaluated through `forge model evaluate` on the 154-row bundled holdout
(majority baseline 62.3%), with the *same rows* in different column layouts. Run against `git archive HEAD` in a neutral
directory (`forge.__file__` confirmed to be the pristine copy):

| holdout CSV | exit | accuracy | loss | what happened |
|---|---|---|---|---|
| columns in training order | 0 | **74.0%** | 0.493 | correct |
| `Glucose,Pregnancies,...` (two columns swapped) | **0** | **51.9%** | 1.726 | accepted; below the 62.3% baseline; `84` of `154` predicted labels change |
| `id` + all 8 features | 1 | — | — | rejected **only because 9 ≠ 8**: `expected 8 input feature(s), received 9` |
| `id` replacing the last feature (still 8 feature columns) | **0** | **39.0%** | 13.28 | accepted: `id`'s values were fed to the model as `Age` |
| last feature renamed `Age → Weight`, values untouched | **0** | 74.0% | 0.493 | accepted: the artifact cannot see names at all |

Same feature count, different predictions, and nothing the artifact could use to notice: that is (1)–(4) of the brief.
The ID-column question is answered by the fourth row: the workflow accepted it *when the width happened to match*, and
what happened was silent garbage; the third row's rejection is an accident of counting. `forge model predict` has no
CSV reader at all at HEAD (`Error: input must contain numeric JSON data.`), so the CLI half of the brief needed one.

California housing (StatLib, 3,000 train / 2,000 held out, `target_transform="standardize"`, unnamed artifact, M118
behaviour) shows how bad "silent" can be on a regression target in dollars: correct MSE 3.51×10⁹; two columns swapped
1.47×10¹¹ (**42×** worse); the columns shuffled 2.64×10¹⁴ (**75,000×**); an `id` replacing the last feature 1.05×10¹⁴ —
all exit status 0. Section 17.

## 4. Current architecture (read from HEAD, before modifying it)

| Piece | Behaviour at HEAD |
|---|---|
| `InputSchema` (`forge/serialization/model.py`) | frozen dataclass, one field `feature_count`. **Not persisted**: `inspect_model()` *derives* it from `_leading_linear_in_features(root)` — the first `Linear.in_features` of a `Sequential` — and only when `task` is `regression`/`tabular_classification` |
| `ModelInfo` | `model`, `preprocessing`, `classes`, `task`, `format_version`, `device`, `input_schema`, `target_transform` |
| `save_model(model, path, preprocessing, classes, task, target_transform)` | metadata keys `forge_format_version`, `device`, `root`, `preprocessing`, `classes`, `task`, plus `target_transform` **only** when given |
| Format versions | `2`; `3` **only** when a `target_transform` is present (M116). Rule since M71: optional key, no bump — *unless an older reader ignoring the key would misinterpret the file*, which a target transform (wrong-unit numbers) does |
| `_predict_tensor_core()` (M102) | coerce → batch check → **width check** → preprocessing → finiteness → `predict()` → target-transform inverse. Called by `predict_tensor_artifact`, `predict_tabular_classification_artifact`, `ArtifactPredictor.predict` and (per batch) `.evaluate` |
| `predict_model()` | dispatches on the artifact's `task` to those functions |
| `train_tabular_*()` (M114) | numeric arrays only → validation → split → preprocessing fitted on training rows → `preflight_save()` → `train_and_save()` → `save_and_verify()` → `save_model()` |
| `forge.data.load_csv()` (M118) | `(X, y)`; header cells stripped, unique, non-empty; target by name; features in file order |
| `forge model evaluate` | `.npy` pair, `.csv` + `--target`, or image directory |
| `forge model predict` | JSON numeric file only (no CSV) |
| Metadata precedent | `"preprocessing"`, `"classes"`, `"task"`, `"target_transform"` are sibling keys; an absent key is the ordinary case |

One thing that did **not** match the brief's assumptions: `forge model predict MODEL INPUT.csv` does not exist at HEAD.
Named CLI prediction therefore required adding a feature-only CSV input (section 10).

## 5. Feature-schema design alternatives

**What the artifact records.**

| Option | Verdict |
|---|---|
| **A. `InputSchema.feature_names: tuple[str, ...] \| None = None`**, persisted as an optional list | **selected** |
| B. Always-present names, defaulting to `feature_0, feature_1, ...` | rejected: fabricates identity for every array-trained model and every old artifact; a fabricated name is indistinguishable from a real one |
| C. A `TabularSchema` (names + dtypes + target + roles + categorical maps) | rejected: no workload needs more than names; this is the "general schema framework" the brief rules out |
| D. `{name: index}` mapping | rejected: redundant with order, and a dict invites "order doesn't matter" reading |
| E. A hash/fingerprint of the names | rejected: detects a mismatch but cannot align and cannot say *which* column is wrong |

**How names enter Forge.**

| Option | Verdict |
|---|---|
| A. CSV-derived only | rejected as the *only* route: the training functions take arrays, so names need an entry point there |
| **B. `feature_names=[...]` on `train_tabular_*`** (optional), with the CSV reader able to return the header names | **selected** |
| C. A `TabularData(X, feature_names)` object | rejected: a new public type; every existing `train_tabular_*(X, y)` caller would face a second way to pass data; nothing here needs it |

`feature_names=` is a plain list, so `load_csv(..., return_feature_names=True)` feeds it directly, and so does
`list(df.columns)` for someone who has a DataFrame elsewhere (Forge itself imports none).

**Named-input policy.**

| Option | Verdict |
|---|---|
| **Reorder-and-align when the names are exactly the artifact's; reject otherwise** | **selected** |
| Reject any input whose order differs | rejected: forces every caller to reorder files by hand when Forge can do it with no ambiguity |
| Reorder, but drop extras / fill missing / match case-insensitively | rejected: each is a guess about which column is which |

Reordering is safe *because* the names are exact and unique: given the artifact's names `A` and an input's names `N`
with `set(N) == set(A)`, the permutation is determined — there is nothing to guess. Evaluated against the brief's list:
*preprocessing* (alignment runs first, so `Normalize`/`ReplaceValue` see the artifact's positions), *input schema* (the
width check still runs after it), *performance* (a permutation of a few dozen ints: section 21), *ambiguity* (none:
exact, unique), *artifact semantics* (order is authoritative, so the artifact's order is the target).

**Name rules (Phase 3).**

| Question | Decision |
|---|---|
| Names optional? | yes — `None` is every existing artifact and every array-trained one |
| Order authoritative? | yes — `feature_names[i]` is input column `i` |
| Unique? | yes — duplicates rejected (`must be unique, but 'age' appear(s) more than once`) |
| Strings? | yes — `str` only (a `numpy.str_` becomes a plain `str`); `bytes`, numbers, `None` rejected |
| Empty names? | rejected (empty or whitespace-only) |
| Case | significant — `Age` ≠ `age` |
| Whitespace | preserved — `Age` ≠ ` Age`; **nothing is stripped or normalised**. (The one existing strip is `load_csv()` removing whitespace around a CSV *header cell*, an M118 file-format rule; the schema itself never normalises) |
| Container | `list` or `tuple`, non-empty; a bare `str` (would be a list of characters) is rejected |

**Extra columns / ID column (Phase 6).** Rejected, as the brief guessed — but evaluated rather than assumed. Ignoring an
extra column is a guess that the column is harmless; the third and fourth rows of section 3 show what that guess costs
(`id` fed to the model as a feature, 39.0% accuracy). Special-casing a column *called* `id` would guess again (`ID`?
`index`? `patient_id`? a real feature that happens to be called `id`?). Rejecting is loud, names the column, and the fix is
one edit to the data; ignoring is silent and undiscoverable. A `--drop` option is the natural way to make it convenient
later and is deferred (section 23).

**Format version (Phase 4 / artifact format).** Not bumped. The rule this codebase has followed since M71, made explicit
by M116, is: bump only when an older reader that ignores the new key would *misinterpret* the artifact. An M118 build
reading a named artifact ignores the key and does what it always did with every artifact — check the width — and returns
the same numbers. Verified, not argued: a pristine M118 build (`git archive HEAD`) loads a named artifact and its raw
float32 outputs, accuracy and loss are **bit-identical** to the new build's (`np.array_equal` on the `(154, 2)` output).
A bump to version 4 was rejected because it would (a) make every named artifact unreadable to an older build even for
`.npy` use, (b) need a combined scheme with M116's version 3 (`v3 ⇒ target_transform required` today), and (c) protect
against a loss, not a misreading. **The one residual cost is documented, not hidden:** an older build's
`forge model convert` rewrites the artifact *without* the names (measured: names `None` after an M118 `convert`) — a valid
unnamed artifact, not a misread one, with column checking lost. The new build's `convert` preserves them.

**Unnamed input, named artifact / named input, unnamed artifact.** A bare array carries no names and gets the existing
width check (documented, tested to still accept a reordered array — that limitation cannot be closed without pretending
an array knows something it does not). Names given to an artifact that records none cannot be verified. Raising would
break `forge model evaluate legacy.forge data.csv --target y`, which M118 documents and tests; silently accepting would
imply a check that did not happen. So: the input is used as given and a `UserWarning` says the order could not be verified
— emitted only after the call *succeeds* (a warning next to an error that rejects the input anyway is noise; found and
fixed during implementation, section 19). The CLI prints it as one `Warning:` line on stderr; stdout, including `--json`,
is untouched.

## 6. Selected design

| Layer | Change |
|---|---|
| `forge/data/feature_names.py` (new, 60 lines) | `validate_feature_names(names) -> tuple[str, ...]`: the one definition of a valid list, used by training, saving, loading and prediction |
| `InputSchema` | `feature_names: tuple[str, ...] \| None = None` (equality with the old width-only form preserved) |
| `save_model(..., feature_names=None)` | writes optional `"feature_names"`; requires a tabular task and exactly one name per input column of the model's first `Linear`; **key absent when not given** (unnamed files byte-identical to before) |
| `inspect_model()` | reads and re-validates names (malformed / duplicate / wrong length / wrong task / no derivable width → `PersistenceError`, so `load_predictor()` refuses too); `ModelInfo.__str__` prints `Feature names: ...` only when present |
| `save_and_verify(..., feature_names=)`, `train_and_save(..., feature_names=)` | pass through; `save_and_verify` reads the names back and fails loudly if they did not survive |
| `train_tabular_classifier/regressor(..., feature_names=None)` | validated (one per column of `X`) before epoch 1; preflight saves the untrained model with them, so an unnamed-able `model=` is refused before training |
| `_predict_tensor_core(..., feature_names=)` + `_column_order()` | **the** rule: validate names → width of names == width of input → match against the artifact's → permutation or `DataError`; then the existing width check, preprocessing, model |
| `predict_tensor_artifact`, `predict_tabular_classification_artifact`, `predict_model`, `ArtifactPredictor.predict`, `.evaluate` | optional `feature_names=`; naming the columns of a non-tabular artifact is a `DataError`, not ignored |
| `forge.data.load_csv(..., return_feature_names=False)` | `True` → `(X, y, names)` from the *same read* |
| `forge.data.load_csv_features(path)` (new) | a CSV with no target → `(X, names)`; every column is a feature |
| CLI `evaluate` | reads names with the CSV, forwards them; passes no kwarg for `.npy`/image (arguments unchanged from M117) |
| CLI `predict` | accepts a `.csv` of feature columns for the two tabular tasks; JSON unchanged |
| CLI `inspect` / `convert` | names shown (`--json`: `input_feature_names`, `null` when absent); `convert` preserves them |

`forge model predict` and `forge model evaluate` cannot apply different rules because neither implements one: both hand
`feature_names` to the same function. Measured by a test that compares their error text for the same defect.

## 7. Rejected alternatives

`TabularData`/`TabularDataset`; fabricated default names; case-insensitive or whitespace-normalising matching;
ignoring extra columns; automatic `id` detection; auto-dropping a target column left in a `predict` file; a sidecar
`.schema` file for `.npy`; a format-version bump; failing (rather than warning) when names are given to an unnamed
artifact; putting the alignment in the CLI or in `load_csv()` (the reader must stay ignorant of artifacts); a
`feature_names` field on the training result dataclasses (the artifact is the record); pandas/DataFrame input.
Reasons are in section 5.

## 8. Compatibility strategy

| Situation | Behaviour |
|---|---|
| Artifact saved before M119, new build | loads; `feature_names is None`; width check only; **the bundled `models/tabular_classifier/diabetes_classifier.forge` is tested** |
| Artifact trained from arrays, new build | identical to the above (no key is written) |
| Unnamed artifact written by the new build vs by an M118 build | **identical**: measured with the same seed, every archive entry (`metadata.json` and each parameter array) has the same SHA-256, for a classifier and for a regressor with an M116 target transform (the key is absent, not null) |
| Named artifact, **older** build | loads and predicts **bit-identically**, width check only (measured); older `convert` drops the names (measured) |
| Named artifact, bare array / `.npy` / JSON | unchanged: width check only |
| Named artifact, named input | aligned or rejected |
| Unnamed artifact, named input | used as given + one warning after success |
| Tampered names (wrong count, duplicates, non-list, wrong task, no derivable width) | `PersistenceError` from `inspect_model()`, `load_predictor()`, `predict_model()`; an explicit `null` means "no names" |
| Names + M116 target transform | both present, version 3, both applied, tested |
| `save_model` argument order | `feature_names` appended last: no positional caller changes |

## 9. Training integration

`train_tabular_regressor(X, y, ..., feature_names=None)` and the classifier. The arrays API is untouched; a call without
`feature_names` produces the artifact it always did. Names are identity metadata, so **the trained model, the fitted
preprocessing and every reported number are unchanged by them** — tested with the same seed: identical parameter SHA-256,
identical result fields, identical metadata apart from the one key. A CSV-derived list excludes the target column
(`A,B,C,target` → `[A,B,C]`, tested with the target first, middle and last). Invalid lists are `DataError`s raised before
epoch 1 with nothing written; a `model=` whose input width cannot be read from its saved architecture (not a `Sequential`
starting with `Linear`) is a `PersistenceError` from the preflight, before epoch 1, with the preflight's temporary file
removed. `missing_columns=` still addresses columns by position.

## 10. Prediction behaviour

Routes verified to converge on `_predict_tensor_core()`: `ArtifactPredictor.predict()`, `predict_tensor_artifact()`,
`predict_tabular_classification_artifact()`, `predict_model()`, `forge model predict`. There is no separate execution
path for named input: alignment runs, then the same width check, preprocessing, model and target-transform inverse as
before. Every case below is parametrised across all routes and both tasks.

| Named input | Result |
|---|---|
| the artifact's names, its order | identical to the unnamed call |
| the artifact's names, any other order | predictions **bit-identical** to the correctly ordered input; checked against an independent oracle (artifact preprocessing + model applied by hand) |
| a name the artifact does not have | `DataError`: `missing (...): ['rooms']; unexpected (...): ['bedrooms']` |
| a missing column | `DataError` naming it |
| an extra column, incl. `id` | `DataError`: `unexpected ... ['id']` — never dropped |
| an `id` replacing a feature (width still right) | `DataError` naming both |
| duplicate names / empty / blank / non-string / a bare string | `DataError` naming the problem |
| wrong number of names for the columns given | `DataError` (`3 name(s) but the input has 4 column(s)`) |
| `Income` for `income`, ` age`, `dis tance` | `DataError` + a hint (`'Income' vs 'income' differ only in case/whitespace, which counts`); matching stays exact |
| 3-D input with names | `DataError` |
| a single unbatched row (regression) | reordered; a classifier's unbatched-row error still comes first |
| `Tensor`, list, float64 array, tuple of names | all accepted; the caller's array is never modified |
| names for an image/segmentation artifact | `DataError`, not ignored |
| names, unnamed artifact | used as given; `UserWarning` after success; malformed names / wrong count still rejected |

`forge model predict MODEL rows.csv` (new): a header row and **only feature columns** (`load_csv_features()`); the same
rule. A target column left in the file is an extra column. JSON input is read exactly as before and carries no names.

## 11. Evaluation behaviour

`ArtifactPredictor.evaluate(X, y, feature_names=)` aligns `X` **once** (not per batch) with the same `_column_order()`,
so what is scored is what `predict()` would predict. A reordered `X` scores **exactly** like the ordered one (every field
of the result, classification and regression); bad names are rejected before any row is scored; `batch_size` does not
change a named result. **M116**: a `target_transform="standardize"` artifact evaluated on a reordered CSV returns the
same native-unit MSE as the ordered one, and predictions equal the independent inverse-transform oracle to `1e-5`
(alignment is on the input side, the inverse on the output side; both compose). `forge model evaluate` on a reordered CSV
prints exactly what the ordered CSV prints — text and `--json`, classification and regression.

## 12. CSV behaviour

`load_csv(path, *, target, labels=False, return_feature_names=False)`: `True` returns `(X, y, names)` from the same read —
the names of exactly the columns of `X` (target excluded, file order). Default return is still `(X, y)`; errors are
byte-identical with or without the flag. `load_csv_features(path)` shares the reader (`_read_csv`) and the whole M118
file contract; there is nothing to drop, so a target/`id` column left in the file is a feature column *to the reader* and
an unexpected column *to a named artifact*. Names are the header cells with surrounding whitespace stripped (the M118
rule); `Age` and `age` are two columns with two names; duplicate headers are rejected by the reader, so CSV-derived names
are always valid. The reader still imports nothing from `training`/`serialization` (tested by AST).

## 13. `.npy` behaviour

A `.npy` file has no column names, so nothing can be verified by name. Unchanged: shape (width) validation only, no
warning, no error. A test pins the limitation instead of hiding it: the same values in a different order through the
`.npy` door exit 0 and score differently. No sidecar file format was created. **The guarantee is: names checked when there
are names; arrays are the caller's responsibility, exactly as before.**

## 14. ID-column behaviour

Decided in section 5, demonstrated on real data in sections 16–17: an `id` column is an *unexpected column* and is
rejected; nothing is special-cased and nothing is dropped. `id` + all features → `unexpected ['id']`; `id` replacing the
last feature (the case the old width check *could not* catch) → `missing ['Age']; unexpected ['id']`; a target column left
in a `predict` file → `unexpected ['Outcome']`.

## 15. Preprocessing interaction

Names belong to the **raw** input positions. Alignment runs first, so the artifact's persisted `Normalize`/`ReplaceValue`
see the columns in the artifact's order — the property a wrong ordering of these two steps would break, and a mutation
proves the tests notice (M12). Tested: a `missing_columns=[1]` artifact, an input with `age == 0` in *some* rows, columns
shuffled and named → predictions identical to the ordered input (the sentinel is replaced in the artifact's column, not
the input's). Preprocessing is not renamed, not responsible for identity, and is byte-equal between a named and an unnamed
training run (`repr` equal; parameter SHA-256 equal). No normalisation statistic is a feature name.

## 16. Real Pima validation

`python tests/real_world/feature_schema_workflows.py` (CPU, and `--device cuda`; both `ALL OK`). The 614-row training pool
(rows of `diabetes.csv` not in the holdout), trained twice with the same seed: once from arrays (unnamed, the M118
artifact) and once with the CSV header's names. Every row is the same 154 holdout rows, only the column layout changes,
evaluated through the real `forge model evaluate`:

| holdout CSV | unnamed artifact (M118 behaviour) | named artifact (M119) |
|---|---|---|
| correct order | accuracy 72.7% | accuracy 72.7% (**identical**) |
| two columns swapped | **38.96%** — accepted, silently wrong | **72.7% — exactly the correct order's numbers** |
| all eight columns shuffled | **37.66%** — accepted | **72.7%** |
| `id` + all features | rejected by width only (`expected 8 ... received 9`) | rejected: `unexpected ... ['id']` |
| `id` replacing the last feature | **38.31%** — accepted, silently wrong | rejected: `missing ['Age']; unexpected ['id']` |
| last feature renamed `Age → Weight` | 72.7% — accepted | rejected: `missing ['Age']; unexpected ['Weight']` |
| `Pregnancies` written `PREGNANCIES` | 72.7% — accepted | rejected: `differ only in case/whitespace` |

Also shown by the script: the named and unnamed artifacts from the same seed have **identical trained parameters**
(SHA-256 `c9c1261f5ecf39a1…`) and identical validation accuracy (77.2% against a 63.4% baseline) — names change nothing
but the one metadata key; `forge model predict` on a reordered feature-only CSV prints exactly what the ordered one
prints, and a left-in `Outcome` column is rejected as unexpected; the **bundled pre-M119 artifact**
(`models/tabular_classifier/diabetes_classifier.forge`) still evaluates the holdout CSV (72.7%, confusion matrix equal to
the API's on arrays) with one `Warning:` line on stderr and pure JSON on stdout. Pristine-HEAD reproduction of the same
defect on a different trained model is section 3.

## 17. California housing validation

Same script with `--cadata` (StatLib, 20,640 rows; 3,000 train / 2,000 held out; `target_transform="standardize"`; target
is the *first* column of the CSV). Named header names: `median_income … longitude` — the target is **not** in them.
Held-out MSE in dollars², CPU:

| holdout CSV | unnamed artifact | named artifact |
|---|---|---|
| correct order | 3.51×10⁹ | 3.51×10⁹ |
| two columns swapped | 1.47×10¹¹ (**42×**, accepted) | **3.51×10⁹** |
| columns shuffled | 2.64×10¹⁴ (**75,000×**, accepted) | **3.51×10⁹** |
| `id` replacing the last feature | 1.05×10¹⁴ (accepted) | rejected, names both columns |
| renamed / case-changed column | accepted | rejected |

M116 is intact alongside the names: `target_transform` is still in the artifact (format 3), results are still native dollars
(R² 0.737, MAE ≈ 40,500), and parameters are identical to the unnamed run (CPU). The same run on the GeForce 940MX was
`ALL OK`.

## 18. External consumer validation

`tests/test_packaging_smoke.py` builds the real wheel (`python -m build --wheel`), installs it into a fresh venv (never
`pip install -e`), and every step below runs from a directory outside the repository, each CLI call in a fresh process.
Eight tests were added (the fixture trains a **named** Pima artifact inside the clean venv from a CSV):

1. `forge.__file__` is under `site-packages`, not the repository (asserted in the training consumer);
2. the artifact records the eight header names and `forge model inspect` shows them (text and `--json`), format version 2;
3. a **named CSV in the correct order** and a **reordered** one print byte-identical `--json` and empty stderr;
4. **unknown-column**, **extra `id`**, **`id` replacing the last feature** and **case-different** CSVs are each one
   `Error:` line, exit 1, empty stdout, no traceback, naming the columns (text and `--json`);
5. `forge model predict` on a feature-only CSV: reordered == ordered, 154 predictions; a target column left in the file is
   rejected as unexpected;
6. the **bundled pre-M119 artifact** evaluates a CSV in the wheel with exactly one `Warning:` line on stderr and JSON equal to
   a standalone reference consumer (`forge.load_predictor().evaluate()`).

All eight pass in the full suite. A separate, narrated run of the same flow (`python -m build --wheel` -> `forge-0.1.0-py3-none-any.whl`, which contains `forge/data/feature_names.py`; a fresh venv whose `pip list` shows only `forge`, `numpy` and `pillow` -- no pandas; every command run from a directory outside the repository) printed:

```text
forge.__file__ = ...\wheel_run\env\Lib\site-packages\forge\__init__.py
trained; artifact records: ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')
correct vs reordered CSV -> stdout byte-identical: True | exit 0 0 | stderr empty: True     (accuracy 0.7597)
$ forge model evaluate named.forge unknown.csv --target Outcome --json
  exit 1   Error: ArtifactPredictor.evaluate() the input's columns do not match this artifact's feature names -- missing ...: ['Age']; unexpected ...: ['Weight']. ...
$ forge model evaluate named.forge extra_id.csv --target Outcome
  exit 1   Error: ... unexpected (the input has them, the artifact does not): ['id']. Names are matched exactly; ...
predict feature-only CSV, ordered vs reordered -> identical: True | predictions: 154
$ forge model evaluate pima_old.forge correct.csv --target Outcome --json      (the bundled pre-M119 artifact)
  exit 0   accuracy 0.7273   stderr: Warning: ... records no feature names ... cannot be verified ...
```

## 19. Tests and mutations

**Added: 447 tests** (3,338 → 3,785): `test_feature_names_persistence.py` 73, `test_named_input.py` 192,
`test_cli_feature_schema.py` 88, `test_training_feature_names.py` 39, `test_csv_feature_names.py` 40,
`test_feature_schema_cuda.py` 7 (real GPU, skipped cleanly without CUDA), and 8 installed-wheel tests in
`test_packaging_smoke.py`; shared fixtures in `tests/feature_schema_support.py` (not collected). The reorder tests are
non-vacuous by construction (each first asserts that the same columns in another order, *unnamed*, changes the prediction),
alignment is checked against an independent oracle (artifact preprocessing + model applied by hand), and every rejection is
parametrised over all five routes and both tasks. `tests/real_world/feature_schema_workflows.py` is the real-data script.

**Existing tests changed: two assertions, one file (`test_cli_evaluate_csv.py`), both a genuine contract change.** M118
pinned the exact arguments the CLI passes (`load_csv(target=..., labels=...)` and `evaluate(features, targets)` with no
keyword arguments). The CLI now also asks the reader for the header names and forwards them as `feature_names=`. The
sibling M117 assertion (`kwargs == {}` for `.npy`) is **untouched**: the CLI passes no `feature_names` kwarg at all
for `.npy`/image input, which is the stronger form of "unchanged".

**Mutation testing — 39 deliberate bugs, each applied to a scratch copy of `forge/`, the M119 tests run, expecting failure
(the unmutated copy: 439 passed, and `forge.__file__` proved to be the mutated copy):**

| # | brief item | mutation | killed by |
|---|---|---|---|
| M01, M33 | 1 names not persisted | names not written / not passed from training | `..._written_as_a_json_list...`, `test_inspect_model_exposes_the_names...` |
| M02, M03 | 2 names ignored | named predict / named evaluate input never reordered | `test_reordered_named_input_predicts_...`, evaluate reorder tests |
| M13, M17, M18, M39 | 2 names ignored | `predict_model` / CLI evaluate / CLI (both) / classifier path drop the names | route- and CLI-parametrised tests |
| M04 | 3 wrong reorder | inverse permutation | `..._like_the_correctly_ordered_input[...rotated/shuffled]` |
| M05 | 4 unknown accepted | unknown name mapped to column 0 | `TestRejections::test_an_unknown_column...` |
| M06 | 5 missing accepted | missing name filled from column 0 | `TestRejections::test_a_missing_column...` |
| M07, M27 | 6 duplicates accepted | duplicate check removed; bare string accepted | `test_invalid_names_are_rejected_with_the_reason` |
| M08 | 7 id silently dropped | extra column ignored | `test_an_extra_column_is_rejected_never_silently_dropped` |
| M09, M10, M11 | 8 old artifacts break | unnamed artifact raises / names to unnamed artifact raise / no warning | pre-M119 real-artifact tests, warning tests |
| M12 | 9 preprocessing changes identity | preprocessing runs **before** alignment | `..._like_the_correctly_ordered_input`, `missing_columns` sentinel test |
| M21, M22 | 10 predict vs evaluate differ | `predict` (only) / `evaluate` (only) lower-cases names | `test_predict_rejects_the_same_defects[case]`, `..._one_rule_...`, evaluate case test |
| M19, M20 | exact preservation | names stripped on entry / matched case-insensitively | case & whitespace tests |
| M14, M36 | scope | names for an image artifact ignored / accepted at save for a non-tabular task | `test_names_for_an_image_artifact_are_refused...`, `test_names_require_a_tabular_task` |
| M15, M26, M30, M37 | validation | inspect count check / save count check / names-vs-width check / rank check removed | tamper tests, `..._bad_names_...`, `..._malformed_name_list...`, `..._1d_and_2d...` |
| M16, M29 | CSV | target leaks into names / column 0 mistaken for a target | `test_the_target_is_excluded...`, CLI predict tests |
| M23, M24, M25, M35 | artifact | `convert` drops names / `save_and_verify` stops reading back / names bump the version / names sorted | convert, round-trip, version, `positional_documentation` tests |
| M28, M34 | CLI | `Warning:` handler removed / `inspect --json` omits names | subprocess + `inspect` tests |
| M32 | training | names not preflighted (a bad `model=` reaches epoch 1) | `..._refused_before_epoch_1` |
| M38 | ordering | the "not verified" warning fires before later errors | `..._only_the_error_never_a_warning_beside_it` |
| M31 | CUDA | a reordered CUDA tensor loses its device | see below |

**All 39 killed. Two findings the mutation work and the tests surfaced, both fixed:**

- *A warning beside an error.* The first implementation warned "could not verify the order" as soon as it saw an unnamed
  artifact. That put a `Warning:` line in front of the `Error:` line for inputs rejected anyway (a wrong-width CSV, a `y`
  whose shape does not match the outputs) — breaking M118's "a mistake is one `Error:` line" and failing an existing test
  (`test_a_multi_output_artifact_does_not_take_a_single_csv_target`). Fixed at the root (the warning is emitted only after the
  call succeeds, in one helper) rather than by ordering individual checks; M38 keeps it fixed.
- *A test that claimed more than it checked.* M31 survived at first: `predict()` moves any input to the model's device, so
  losing the device on the reordered tensor changes no result, and my CUDA test named "without leaving the device" only
  compared results. The test now observes the tensor that reaches the model call (CUDA in, CUDA at the model); M31 is killed.

A further mutation-driven improvement: the `predict` command's CSV variants originally lacked a case-different file, so M21
(lower-casing only in `predict`) would have survived; the variants (`case`, inner-space `dis tance`) were added before the run.

## 20. Full-suite result

`python -m pytest tests/ -q` on the development machine (i5-7200U, GeForce 940MX, CUDA 12.6, Python 3.13.5), run with no other heavy job in parallel:

```text
3785 passed in 1302.83s (0:21:42)
```

**3,785 passed, 0 failed, 0 skipped**; M118 ended at 3,338, so +447. CUDA is available and the CUDA tests ran on real hardware (none skipped, none simulated). The M119 files alone (six new files + the eight wheel tests) are 447 of those; the rest is the untouched M1-M118 suite, apart from the two assertions described in section 19. The known load-induced timing flake in `test_cuda_streams.py` did not occur. Real-data scripts (not part of the suite): `tests/real_world/feature_schema_workflows.py --cadata ...` -- `ALL OK` on CPU and with `--device cuda`.

## 21. Performance

Not optimised (no caching, compiled schema, or special data structure). Measured on the development machine
(i5-7200U): **matching 8 names costs 6–14 µs** (`_column_order`, best of 200; a 64-name list is proportionally the same
order). `evaluate()` on Pima ×20 (3,080 rows × 8): unnamed 5.2 ms · named, already in order 5.4 ms · named, reversed
5.9 ms. On California (20,640 × 8): 50 ms · 42 ms · 35 ms — the differences between the three are smaller than the
run-to-run noise (the "already in order" path does *no* extra work: `_column_order` returns `None` and nothing is copied).
The reordering itself is one `np.ascontiguousarray(X[:, order])`, measured on an idle machine: 0.009 ms for 154 x 8, 0.058 ms for 3,080 x 8, **1.1 ms for 20,640 x 8 (about 2% of that evaluation)**; at 20,640 x 64 it is 20 ms and matching 64 names 60 us -- a memory copy proportional to the data, far below inference cost, so nothing was optimised. GPU (940MX) timings are noisier still — thermal drift on this machine is documented since M47 — and showed no
systematic effect. The script asserts the reversed path stays within 1.5× + 5 ms of unnamed.

## 22. Limitations

- **Arrays carry no names.** A NumPy array, `.npy` file or JSON file is checked for width only; a reordered one is
  accepted (pinned by a test). This cannot be closed without inventing information; no sidecar format was made.
- **Artifacts trained before M119, or from unnamed arrays, cannot check columns**, and the CLI/API say so with one warning
  after success. Existing artifacts are not upgraded (no in-place "add names" tool; retrain with `feature_names=`).
- **Names are only supplied through `train_tabular_*`, `save_model`/`save_and_verify`/`train_and_save`.** `forge.train()`'s
  `Dataset`/`DataLoader` path carries no names.
- **Feature names require a readable input width**: a custom `model=` that is not a `Sequential` starting with `Linear`
  cannot record names (`PersistenceError` before epoch 1); `InputSchema` has always had the same restriction.
- **An older Forge's `convert` drops the names** of a named artifact (valid unnamed result; measured). No older build
  misreads a named artifact.
- **Names must be a `list`/`tuple`** (a NumPy array or pandas `Index` is rejected with a clear message).
- **Extra columns are rejected, not dropped.** An `id`/target column must be removed from the file; there is no `--drop`.
- `missing_columns=` still addresses columns by **position**, not by name.
- The CSV reader's header strip (M118) is the only normalisation of a name anywhere; an explicit `feature_names=` list is
  stored exactly as given, so `[" Age"]` at training and a CSV header `Age` at inference mismatch by design.

## 23. Deferred work

`--drop COLUMNS` / `drop=` for `id`-style columns; addressing `missing_columns=` by name; naming a `Dataset`/`DataLoader`
workflow's columns; an in-place "add names to an existing artifact" command; names on non-`Linear`-first architectures;
accepting NumPy string arrays / pandas `Index` for `feature_names`; recording the target column's name in the artifact.
None has a demonstrated workload yet.

## 24. Final decision

**IMPLEMENTED.** The answer to the milestone's question is *yes, a saved tabular model can know what each input column
represents*, and the smallest schema that provides the guarantee is an optional, exactly-preserved, ordered list of raw
feature names recorded in the artifact, matched to named input by one function. The decision rests on evidence, not
elegance: on real data an artifact that knows only a count accepted swapped columns (accuracy 72.7% → 39.0% on Pima,
MSE 42× worse on California housing) and an `id` column standing in for a feature, with exit status 0; with names the same
inputs score identically or are refused with the columns named. It did not need a new public type, a format version, a DataFrame
or a change to any NumPy caller, and it costs microseconds.

```text
Decision: IMPLEMENTED
```

```text
Framework changes: forge.data.feature_names.validate_feature_names (new, the one name-validity rule); optional feature_names= on
  train_tabular_classifier/regressor, train_and_save, save_and_verify, save_model, predict_tensor_artifact,
  predict_tabular_classification_artifact, predict_model, ArtifactPredictor.predict/evaluate; one alignment rule,
  inference._column_order(), inside the existing _predict_tensor_core path; named-input errors for non-tabular artifacts.
Schema changes: InputSchema gains feature_names: tuple[str, ...] | None = None; optional "feature_names" metadata list, written only
  when given, format version NOT bumped (measured: a pristine M118 build reads a named artifact bit-identically); unnamed
  artifacts byte-identical to before (every archive entry SHA-equal, old vs new build); inspect_model/ModelInfo expose names.
CSV changes: load_csv(..., return_feature_names=False) -> (X, y, names) when True; new load_csv_features(path) -> (X, names) for a
  target-less file; shared _read_csv; M118 contract and errors unchanged.
CLI changes: evaluate forwards the CSV header names; predict accepts a .csv of feature columns (tabular tasks) with the same rule;
  inspect shows Feature names / input_feature_names; convert preserves names; one "Warning:" stderr line for an unnamed artifact.
Tests added: 447 (439 in six new files + 8 installed-wheel), 39/39 mutations killed, two M118 assertions updated for a genuine contract change.
Full-suite result: 3,785 passed, 0 failed, 0 skipped (baseline 3,338; CUDA available, GeForce 940MX; 21m42s)
Pima result: two swapped columns 72.7% -> 38.96% (unnamed, accepted) vs 72.7% identical (named); id-for-a-feature 38.31% accepted vs rejected;
  named and unnamed training give identical parameters.
Regression result: California housing (target_transform=standardize): swapped columns MSE 3.51e9 -> 1.47e11 (unnamed) vs 3.51e9 (named);
  shuffled 2.64e14 vs 3.51e9; native units and R^2 0.737 intact.
External consumer status: installed wheel, fresh venv, outside the repo, fresh process per call: named CSV, reordered CSV (byte-identical
  output), unknown / extra id / id-replaces-feature / case CSVs (one Error line each), predict CSV, pre-M119 artifact (one Warning line).
CPU/CUDA status: CPU tests need no CUDA; 7 CUDA tests and the real-data script run on the real GeForce 940MX (not skipped, not simulated).
Compatibility status: every pre-M119 artifact and every NumPy call unchanged; the bundled real artifact tested; named artifacts readable by
  an M118 build (width check only); an M118 build's `forge model convert` would drop the names (documented).
Remaining limitations: arrays/.npy/JSON carry no names and are width-checked only; unnamed artifacts cannot check columns; extra columns
  are rejected (no --drop); names are a list/tuple; forge.train() carries no names.
Concrete trigger for next work: a real workload where the id/target column is legitimately present in the data file and removing it by hand is
  the friction (=> a deliberate `drop=`), or a user who must add names to an artifact they cannot retrain (=> an explicit "add names" operation).
```
