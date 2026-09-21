# M118 — First-class CSV tabular workflow

Status: **IMPLEMENTED.** One small public function, one CLI input format, **no change to any training, evaluation,
preprocessing, target-transform or serialization code.** Baseline 3,178 tests → see section 18 for the final count.

```text
CSV file ──► forge.data.load_csv() ──► (X, y) NumPy arrays ──┬──► train_tabular_classifier() / _regressor()   (unchanged)
                                                              └──► ArtifactPredictor.evaluate()               (unchanged)
forge model evaluate MODEL DATA.csv --target COLUMN  ==  the CLI reading DATA.csv through load_csv(), then M117
```

## 1. Problem

A developer with an ordinary tabular CSV had to write their own CSV → NumPy conversion before Forge's tabular workflow
(`train_tabular_*()`, `forge model evaluate`) could touch it. Forge's own repository already carried three ad-hoc copies
of that conversion: `examples/tabular_diabetes/dataset.py::load_raw` and `evaluate.py::load_labeled_csv` (`csv` module,
fixed column positions) and `tests/real_world/tabular_workflows.py::_load_csv` (`np.genfromtxt(skip_header=1)`, which
silently turns an unparsable cell into `nan` and cannot tell a header from data).

## 2. Current limitation from M117 (verified at HEAD, not taken from the report)

- `forge model evaluate MODEL INPUT [TARGETS]` read `.npy` (`allow_pickle=False`) and ImageFolder directories only;
  a CSV was reported as "not a readable .npy file". `docs/development/cli.md` and `m117-...md` listed CSV as deferred
  ("a user who cannot produce `.npy`").
- `forge/training/tabular.py` documented "no DataFrame or CSV support ... read the file with ordinary Python/NumPy".
- `forge.data` had no CSV reader (`Dataset`, `TensorDataset`, `ImageFolder`, transforms, loaders).

Current behaviour that the design had to respect (all read from the code at HEAD):

| Area | Behaviour at HEAD |
|---|---|
| `train_tabular_classifier/regressor(X, y, *, path, ...)` | array APIs; `_coerce_features` → finite `float32` `(n, F)`; preflight before epoch 1; split → preprocessing fitted on the training rows → `train_and_save`; results from `load_predictor().evaluate()` on the saved artifact |
| Labels | classifier `y`: class names (`str`) or integer indices `0..K-1`; **integer-valued floats and bools accepted** |
| `ArtifactPredictor.evaluate(X, y)` | `encode_class_labels`: names or **integer** indices only (a `float64` `y` is refused even if integral); regression `y` numeric, native units; applies persisted preprocessing and target-transform inverse |
| `InputSchema` (M101) | persists a **feature count** only — no feature names — so column *reordering* is undetectable by an artifact |
| Artifact | `preprocessing`, `classes`, `task`, `target_transform` (M116) are persisted inside the `.forge` file |
| CLI errors | `CLIError`/`ForgeError` → one `Error: ...` line on stderr, exit 1, no traceback; stdout empty on failure |
| Dependencies | `numpy>=1.26`, `Pillow>=10.0`; no pandas (pandas 2.3.1 *is* installed in the dev env, which is why "Forge does not import it" is tested in a subprocess) |

## 3. Real CSV workload

Real datasets already used by Forge, no new dataset chosen to justify the feature:

- **Pima diabetes** — bundled `examples/tabular_diabetes/data/diabetes.csv` (768 × 9, 23.9 kB) and the 154-row
  `diabetes_holdout_eval.csv`; the 614-row training pool is the rows of the first that are not in the second.
- **California housing** — StatLib `cadata.txt` (`lib.stat.cmu.edu/datasets/houses.zip`, 20,640 × 9, target median house
  value in dollars, **first** column). It is not naturally CSV, so a *controlled* CSV was made from it: header names
  `median_house_value, median_income, housing_median_age, total_rooms, total_bedrooms, population, households, latitude,
  longitude`, values written with `repr(float)` (exact round-trip), no value or row changed (1,237,466 bytes). 3,000
  training / 2,000 held-out rows from `default_rng(123).permutation` (the M115–M117 recipe). Reproduce with
  `python tests/real_world/csv_workflows.py --cadata cadata.txt`.

## 4. CSV contract

Implemented and enforced by `forge/data/csv_reader.py`; every violation is a `DataError` naming file, line and column.

| Question | Decision |
|---|---|
| Header | **Required**, first non-blank row, never sniffed. A file whose first row is all numbers is rejected as "looks like data, not a header" |
| Delimiter / quoting | comma; standard `"` quoting (quoted commas/quotes/newlines fine); malformed quoting is an error (`csv` `strict=True`). No dialect detection; a `;`/tab/`\|` one-column header gets a targeted message |
| Encoding | UTF-8, leading BOM accepted (Windows tools write one — M89 lesson); anything else is a "not valid UTF-8" error naming the byte offset |
| Target column | **explicit** `target="Name"`, exact match, never "last column"; removed from the features. Housing has it *first*, Pima *last* |
| Feature columns | every other column, **numeric**, **in file order** (never sorted) |
| Column names | stripped of whitespace; must be non-empty and **unique** (duplicates, and a target appearing twice, are separate errors) |
| Extra columns / ID column | not silently dropped: an ID column is a non-numeric or meaningless feature and must be removed from the file (no `--drop`, see 20) |
| Numbers | ASCII decimal floats only (`12`, `-3.5`, `.5`, `1e-3`), whitespace ignored. **Rejected:** text, `1_000`, hex, non-ASCII digits (all of which `float()` alone would accept or half-accept), `nan`/`inf`/`infinity` (named as non-finite), overflow to infinity |
| Missing values | **not supported, never imputed.** An empty cell is an error naming line and column. `ReplaceValue` (`missing_columns=`) replaces a numeric *sentinel* such as `0` — an ordinary number to the reader — and cannot represent "empty"; no CSV-specific rule invented |
| Categorical / string features, one-hot, dates | unsupported; a text feature cell is an error that says so |
| Row lengths | every row must have the header's field count; blank lines are skipped, `,,` is not blank |
| Target values | `labels=False` (default): numbers → `float64`. `labels=True`: whole numbers → `int64` class **indices** (`1.0` is 1, `0.5` is an error) or text → `str` class **names**; a mixed column is an error |
| Class order | **never** taken from the file: the reader returns labels; the class list belongs to `train_tabular_classifier` (sorted names or `classes=`) and to the artifact at evaluation |
| Too few samples | the reader accepts any positive row count; the *training* preflight ("at least 3 samples", "at least 2 classes") still speaks |
| Feature-name schema | none. The artifact records a feature **count**; a CSV whose columns are in another order is not detectable. Documented, not papered over |

## 5. Architecture

`forge/data/csv_reader.py` (281 lines: a 68-line contract docstring, then ~190 non-blank lines of code and function docstrings; stdlib `csv` + `re` + NumPy) is the whole implementation. It imports nothing from
`forge.training`, `forge.serialization` or `forge.nn`, and its code never mentions preprocessing, classes, target transforms
or artifacts (an AST test enforces that, docstrings excluded). Everything after the arrays exist is existing code.

- Parsing: `csv.reader(strict=True)` over `open(..., encoding="utf-8-sig", newline="")`; cells parsed by an ASCII-only
  regex *then* `float()`, so `float`'s lenient forms never enter; `X` is `float64` (every digit of the file kept; Forge
  casts to `float32` itself, as it does for any float64 array).
- No `eval`, no `pickle`, no `np.genfromtxt` (which coerces to `nan`); tested by source inspection.
- The whole file is read into memory (see 15).

## 6. Public reader vs internal CLI helper — decision: **public** (`forge.data.load_csv`)

Option A. Reasons, in order of weight:

1. **The training API takes arrays.** M118's training integration *is* `X, y = load_csv(...)` followed by the unchanged
   call; an internal-only reader would leave every Python user (and the three copies in section 1) hand-rolling the same
   parse the CLI already does — and the brief's success test ("without first writing their own CSV-to-NumPy layer") is
   about Python developers too.
2. **The contract is small and stable:** `load_csv(path, *, target, labels=False) -> (X, y)`, `float64`/`int64`/`str`
   arrays, `DataError` for every rejection. That is the whole public commitment.
3. It is not symmetry-for-its-own-sake: nothing else was made public (`TARGET_TYPES`, parsing helpers stay private), no
   `Dataset`/table class exists, and the training functions did **not** grow a `csv=`/path argument.

An early draft had `target_type="auto"` (guess numeric-vs-text from the whole column). It was replaced by the explicit
`labels=` flag after two findings (section 10): `evaluate()` refuses float labels while the trainer accepts them, and a
guessed column type would turn one typo (`O` for `0`) into a third class. The reader now never guesses.

## 7. Training integration

**No training code changed** (`forge/training/tabular.py` differs from HEAD only in two docstring sentences that said
"no CSV support"). Composition:

```python
X, y = forge.data.load_csv("diabetes.csv", target="Outcome", labels=True)
result = forge.train_tabular_classifier(X, y, path="diabetes.forge", classes=[...], missing_columns=[1, 2, 3, 4, 5])
X, y = forge.data.load_csv("housing.csv", target="median_house_value")
result = forge.train_tabular_regressor(X, y, path="housing.forge", target_transform="standardize")
```

So finite-value validation, preflight (before epoch 1), train-only preprocessing, train-only target-transform fitting,
artifact verification and the `evaluate()`-derived result numbers are the same code paths as for any array.
`tests/test_csv_tabular_workflow.py` asserts the signatures still start `(X, y)` with no path/CSV parameter and that the
training module does not import the reader.

## 8. CLI integration

```bash
forge model evaluate MODEL DATA.csv --target COLUMN [--device {cpu,cuda}] [--batch-size N] [--json]
```

- **Dispatch is by extension, deterministically:** `.csv` (case-insensitive) → CSV reader; `.npy` → M117 reader;
  directory → image reader. No content sniffing (tested both ways: CSV text in `x.txt` is refused as `.npy`; `.npy`
  bytes in `x.csv` are refused as CSV).
- `--target COLUMN` is required for a CSV, forbidden for `.npy`/image; a positional `TARGETS` is refused with a CSV
  (the targets are in the file) — each with its own message. `--target` was chosen over overloading the positional
  `TARGETS` (a file path) with a column name: the two would be indistinguishable to the eye.
- The task only decides how the target column is *read*: `labels=True` for `tabular_classification`, numbers for
  `regression` (so a bad regression cell is named). The CLI still makes exactly one `load_predictor()` and one
  `evaluate()` call (tested with counting wrappers) and computes no metric.
- JSON is unchanged: it is `dataclasses.fields(result)`; no CSV metadata key exists (tested against the exact key sets).

## 9. Validation / error semantics

Every case exits 1 with a single `Error: ...` line, no traceback, empty stdout with and without `--json`; messages name the
file, line and column. Reader tests (80) + CLI tests cover: missing file, directory, empty file, header-only, headerless,
semicolon/tab-delimited, duplicate columns, target appearing twice, missing target (lists the columns, case-sensitive),
unnamed column, target-only file, short/long rows (line number counts blank lines), 19 non-numeric spellings (`abc`, `1_000`,
`0x1F`, Arabic-Indic digits, `$5`, `N/A`, ...), empty/whitespace-only cells, empty target, NaN/Inf in features and targets
(8 spellings each), `1e999`, three malformed-quoting shapes, invalid UTF-8, binary garbage, mixed/fractional labels, and
the evaluation API's own errors reached through a CSV (wrong feature count, unknown class name, index out of range).
Errors that `ArtifactPredictor.evaluate()` raises keep its wording (it names `evaluate()`), as in M117.

## 10. Pima results (real, CPU and CUDA)

`python tests/real_world/csv_workflows.py` (CPU: 1.8 s train; CUDA 940MX: 6.2 s):

- `diabetes.csv` → `X (768, 8)`, `y (768,)` **bit-identical** to `np.genfromtxt` (rows, feature order, values, labels).
- Training pool (614 rows), `classes=["no_diabetes","diabetes"]`, `missing_columns=[1,2,3,4,5]`, seed 0: CSV workflow and
  NumPy workflow give **identical** results — every result field and the SHA-256 of all trained parameters
  (`c9c1261f5ecf39a1…`); 85 epochs, validation accuracy 77.2 % vs 63.4 % majority baseline. Compared with equality,
  not a tolerance. (Artifact *files* differ in zip timestamps only; every member is byte-equal.)
- 154-row holdout through `forge model evaluate ... holdout.csv --target Outcome`: accuracy 72.73 %, baseline 62.34 %
  (the M113 figure), loss 0.5852, confusion `[[80, 16], [26, 32]]`; **JSON byte-identical** to the `.npy` form.
- The pre-existing bundled classifier (`models/tabular_classifier/`, saved long before M118) evaluates the bundled holdout
  CSV to the same numbers as its `.npy` form (a test, and an installed-wheel test).

**Finding (pre-existing, not changed):** `train_tabular_classifier` accepts integer-valued float labels (`1.0`), but
`ArtifactPredictor.evaluate` refuses a `float64` `y` ("requires y to be class names (str) or integer class indices").
A first CSV run of the CLI hit it. The reader therefore returns `int64` for `labels=True`. Aligning `evaluate()` with
the trainer would change public API behaviour and is outside M118; it is recorded in 19.

## 11. California housing results (real, CPU and CUDA)

- `housing_full.csv` (1.24 MB, 20,640 rows) → arrays **bit-identical** to the `cadata.txt` values; target is the first
  column and is selected by name (a last-column assumption would have trained on latitude).
- 3,000-row training CSV, `target_transform="standardize"`, seed 0: CSV and NumPy workflows **identical** (all result
  fields, parameter SHA-256 `a270e22c30ae1b80…`), 79 epochs, transform `mean=210,786.5 std=116,963.9` fitted on the
  training split only; 8.1 s CPU (CUDA: 42.1 s, 75 epochs — CUDA is supported, not faster here, as in M114/M116).
- 2,000 held-out rows via `forge model evaluate housing.forge held_out.csv --target median_house_value`, **native
  dollars**: MSE 3.506×10⁹, MAE 40,501, baseline MSE 1.333×10¹⁰, R² 0.737 (CUDA-trained: 3.585×10⁹ / 41,158 / 0.731).
  Consistent with M116/M117's 0.749 (same recipe, seed and dataset; the exact held-out rows were not recorded then, so the
  baselines differ slightly). JSON byte-identical to the `.npy` form.

## 12. M116 target-transform validation

The reader never sees a transform. On the awkward M116 data (target ≈ 2×10⁵, model output z-scores) the CSV path's `mse`,
`mae`, `loss`, `baseline_mse` equal an independent NumPy calculation from the artifact's raw `metadata.json` (`rel=1e-5`),
are ≈10⁹ (z-space would be < 10), equal the API, equal the `.npy` run, and the text output is checked against the same oracle.
A sentinel test shows the regression target reaches `evaluate()` as the raw `float64` native numbers. Mutation 5 (ignore the
artifact's transform metadata) is killed. The reader's source is asserted not to contain `target_transform`,
`StandardizeTarget`, `preprocessing`, `classes`, `load_predictor`, ... .

## 13. API / CLI parity

CLI JSON == `.npy` JSON == Python API result, per field, on: the hand-computed 8-row fixture (confusion matrix, per-class
precision/recall, artifact class order that is neither alphabetical nor row order), a regression artifact (vs NumPy),
the M116 artifact, the bundled real artifact + real holdout, and Pima/California in the acceptance script. Printed text is
identical between the CSV and `.npy` forms. Columns swapped in the file change the answer exactly as the swapped NumPy array
does (file order is honoured; the artifact cannot object — limitation in 19).

## 14. External installed-wheel validation

Wheel built from the working tree (`python -m build --wheel`; every `.py` in it byte-equal to the tree), installed into a
fresh venv containing **only** `forge`, `numpy 2.5.3`, `pillow 12.3.0` (`pip list`; `import pandas` fails), cwd outside the
repository, `forge.__file__` = `…\venv\Lib\site-packages\forge\__init__.py`, fresh processes, the `forge` console script:

- training consumer: `load_csv("pima_pool.csv")` → `train_tabular_classifier` (85 epochs, 77.24 % vs 63.41 %) and
  `load_csv("housing_train.csv")` → `train_tabular_regressor(target_transform="standardize")` (79 epochs, val MSE 3.444×10⁹
  vs baseline 1.239×10¹⁰); no CSV→NumPy conversion by the consumer; `"pandas" not in sys.modules`.
- `forge model evaluate pima.forge holdout.csv --target Outcome` → 72.73 %; `... housing.forge housing_test.csv --target
  median_house_value --json` → MSE 3505503930.98 — **the same digits as the in-repo run**.
- a bad CSV → `Error: CSV file 'bad.csv', line 2, column 'b': 'two' is not a number. ...`, exit 1, no traceback.

Automated in `tests/test_packaging_smoke.py` (5 tests, oracle = an independent standalone script that parses with the
`csv` module and calls only `load_predictor().evaluate()`; also asserts the venv has no pandas).

## 15. Performance measurement

`load_csv` best of 5 (i5-7200U); NumPy's own parsers for scale. Linear in file size, ~3.5 MB/s.

| File | Size | Rows | `load_csv` | `np.genfromtxt` | `np.loadtxt` | peak Python alloc |
|---|---|---|---|---|---|---|
| Pima | 24 kB | 768 | 12 ms | 4 ms | 1 ms | 0.4 MB |
| Housing train | 180 kB | 3,000 | 48 ms | 15 ms | 6 ms | 1.5 MB |
| Housing full | 1.24 MB | 20,640 | **347 ms** | 108 ms | 39 ms | 10 MB |
| Housing ×10 (synthetic tiling) | 12.4 MB | 206,400 | 3.6 s | 1.2 s | 0.4 s | 102 MB |
| Housing ×50 (synthetic tiling) | 61.9 MB | 1,032,000 | 18.4 s | 6.0 s | 2.0 s | 508 MB |

Against the work that follows: the 3,000-row housing training file parses in 48 ms and then trains for 8.1 s (CPU), so
parsing is 0.6 % of that workflow (the full 20,640-row file: 0.35 s). Not optimised, and no streaming built, per the brief: the real workloads are
nowhere near a problem. Honest limits: it is ~3× slower than `genfromtxt` and ~9× slower than `loadtxt`, and peak
memory is ≈8× the final array (a list of Python floats before `np.asarray`), so a multi-million-row file would need a
different reader. Deferred (20).

## 16. Tests added (160; 3,178 → 3,338 collected)

| File | Tests | Covers |
|---|---|---|
| `tests/test_csv_reader.py` | 80 | contract, exact values, target removal/position, file-order columns, labels, every rejection above, no-pandas subprocess, no `eval`/`pickle` in source |
| `tests/test_csv_tabular_workflow.py` | 20 | CSV ≡ NumPy training (equality of every field and parameter), train-only preprocessing and target transform, preflight still before epoch 1, float32-overflow refusal, thin-boundary checks, real Pima |
| `tests/test_cli_evaluate_csv.py` | 51 | hand-computed parity, `.npy` parity, M116 native units, sentinel thinness, load/evaluate-once, dispatch, compatibility (bundled real artifact, image, `.npy`), 17+ error cases × {text, `--json`}, fresh process |
| `tests/test_csv_tabular_workflow_cuda.py` | 4 | CUDA CSV evaluation ≡ `.npy` ≡ API ≡ CPU; M116 native units on GPU; CSV → CUDA training → verified artifact; read-only |
| `tests/test_packaging_smoke.py` (+5) | 5 | installed wheel: no-pandas env, real artifact, M116 regression CSV, training consumer, bad-CSV errors |
| `tests/real_world/csv_workflows.py` | script | Pima + California acceptance, CPU/CUDA (not collected by pytest) |

One existing test was edited (3 lines in `tests/test_cli_evaluate.py::test_corrupted_and_unsupported_input_files`): it used
a text file named `text.csv` as "a file that is not `.npy`"; a `.csv` INPUT is now a supported format, so the file is
named `text.txt` (same intent, same assertions).

## 17. Mutation results (17 mutations; all killed)

Each mutation edits one line, runs the new test files (and the M117 CLI tests where relevant), and restores the file
(checked by hash). Script: a small harness in the milestone scratchpad.

| # | Mutation | Result |
|---|---|---|
| 1 | target column kept as a feature | killed (reader tests) |
| 2 | feature columns reversed | killed |
| 3a | reader lets NaN/Inf through | killed |
| 3b | training's post-float32 finite check removed (a CSV value finite in float64 only) | **first run: survived** — a later layer's "non-finite" message satisfied a loose `match`; test tightened to the first check's positional message → killed |
| 3c | reader "repairs" an unparsable cell to `0.0` | killed |
| 4a / 4b | target transform / input preprocessing fitted on all rows | killed |
| 5 | artifact target-transform metadata ignored at evaluation | killed (CSV regression ≡ NumPy) |
| 6 | CLI recomputes one metric itself (a *correct* recomputation) | killed by the sentinel test |
| 7a / 7b | `.npy` inputs routed to the CSV reader / `allow_pickle=True` | killed |
| 8a / 8b / 8c | reader accepts `float()`-parsable text / CLI drops `labels=` / mixed label column accepted | killed |
| 9 / 10 / 11 | headerless file accepted / empty cell → 0 / last column assumed as target | killed |

## 18. Compatibility results

Existing `.npy` and image evaluation tests (all of `tests/test_cli_evaluate.py`, including its image tests, apart from the
one renamed file in 16) pass unchanged; `.npy` behaviour also asserted directly (`--target` with `.npy` refused, key sets unchanged); the pre-M118
bundled artifact evaluates a CSV like its `.npy` form; old artifacts are untouched (byte hashes before/after).
Full suite (`python -m pytest tests/ -q`, CUDA available, 940MX): **3,338 passed, 0 failed, 0 skipped** in 9 min 46 s. (A first
full run had one failure — my own test's source filter tripped on the docstring I had just added to `tabular.py`; the
test was rewritten with an AST docstring-stripper, re-checked with a mutation, and the whole suite re-run green.)

## 19. Limitations

- **Feature names are not persisted**: an artifact records a feature *count*. A CSV whose feature columns are in a
  different order than the training CSV evaluates without complaint (and wrongly). Documented in the reader, `cli.md`, and
  tested as "file order is honoured". Trigger for a fix: a real workload that reorders columns (feature-name schema).
- A class whose **name is a number** (`"1"`, `"2"`) cannot be expressed in a CSV label column (it reads as an index); use the
  array API.
- No categorical/string features, no missing-value handling (empty cell = error), no dates, no other delimiters/encodings,
  no ID/ignore-column option: a CSV with an ID or a text column must be edited first.
- Pre-existing, unchanged: `ArtifactPredictor.evaluate()` refuses float labels the trainer accepts (10).
- In-memory only; ~3.5 MB/s and ≈8× peak memory (15).
- `evaluate()`-originated errors still name `ArtifactPredictor.evaluate()` (as in M117).
- Pima uses `0` as its missing-value sentinel: that is `missing_columns=`, not the reader.

## 20. Deferred work

`--drop`/column selection; feature-name schema; categorical features; missing-value imputation; a faster or streaming reader;
`evaluate()`/trainer label-type alignment; `.npz`; R². None is started; none was needed by Pima or California housing.

## 21. Final decision

CSV became a thin, predictable boundary into the existing tabular workflow: same arrays as the NumPy reference, identical
training and evaluation results, artifact semantics and target transforms untouched, useful errors, installed-wheel verified.

Decision: IMPLEMENTED
