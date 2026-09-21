# M117 — First-class artifact evaluation CLI

**Decision: IMPLEMENTED** — `forge model evaluate MODEL INPUT [TARGETS]`, a thin command-line interface over
`forge.load_predictor(MODEL).evaluate(...)`.

## 1. Problem

M113 made `forge.load_predictor("model.forge").evaluate(X, y)` the correct way to score a saved artifact, and
M116 made that evaluation report native units for a persisted target transform. Using it still required writing
a Python program. An external developer holding a `.forge` file and their evaluation data could `inspect` and
`predict` from the shell (M19/M88) but not answer the question *"how good is this model on my data?"*
without `import forge`.

## 2. Existing API capability (inspected at HEAD `d58bca4`, not assumed from earlier reports)

- `load_predictor(path, *, device=None)` (`forge/training/inference.py`) does `inspect_model()` +
  `_determine_workflow()` + `load_model()` + `load_preprocessing()` + `load_classes()` once and returns an
  `ArtifactPredictor` holding the model, preprocessing, classes, `InputSchema` and `target_transform`.
- `ArtifactPredictor.evaluate(X, y=None, *, batch_size=256)` supports exactly three tasks —
  `tabular_classification`, `regression`, image `classification` — and raises `DataError` for `segmentation`
  and `sequence`. Every numeric batch runs through the same `_predict_tensor_core()` `predict()` uses:
  input-schema check → persisted preprocessing → non-finite check → model → (M116) `_to_native_units()`.
  Classification needs a persisted class list (`PersistenceError` otherwise); the confusion matrix is
  indexed by the artifact's class order; image folders are matched to classes **by name**.
- The results are two frozen dataclasses. `ClassificationEvaluationResult`: `task, samples, loss, accuracy,
  baseline_accuracy, classes, confusion_matrix (read-only int64 ndarray), precision, recall, support`.
  `RegressionEvaluationResult`: `task, samples, loss, mse, mae, baseline_mse`. **There is no R² field.**
- CLI conventions (`forge/cli/main.py`, `model.py`): `CLIError` and `ForgeError` are both caught in one
  place → `Error: <message>` on stderr, exit 1, no traceback; argparse errors exit 2. `model predict` routes
  on `inspect_model(path).task` and refuses a task-less legacy artifact rather than falling back to
  `_legacy_infer_workflow()` (which guesses `regression` for an unlabelled `Linear` classifier).
  `--json` in `inspect`/`predict` is `print(json.dumps(payload, indent=2))` built inline per command — there
  is **no shared JSON helper** to reuse, so nothing competes with the new one.
- Forge has **no** CSV/DataFrame/`.npy` reader in its data layer (`.npy` appears only inside the archive
  format). Discrepancy with the request: the CLI reference is `docs/development/cli.md`; there is no
  `docs/cli.md`, so the former was updated.

## 3. CLI design

```text
forge model evaluate MODEL INPUT [TARGETS] [--device {cpu,cuda}] [--batch-size N] [--json]
```

`cmd_evaluate()` in `forge/cli/model.py`, in order:

1. `MODEL` must exist → else `CLIError`.
2. `inspect_model(MODEL).task` decides only **how to read `INPUT`** (`.npy` vs image directory). A task-less
   artifact, or a task with no evaluation semantics, is refused before any data is read.
3. Read `INPUT`/`TARGETS` (`np.load(..., allow_pickle=False)`) or check the directory; argument-shape mistakes
   (missing/extra `TARGETS`) are CLI errors.
4. `load_predictor(MODEL, device=...)` **once**, then `.evaluate(features, targets)` **once**.
5. Print the result: text, or JSON.

`--device` mirrors `model predict` (optional; default = the device the file was saved from; no auto-selection,
no fallback). `--batch-size` is a pass-through of `evaluate()`'s only parameter (`None` → the API default is
not restated in the CLI). It was not in the request; it is included because a bounded batch is what makes a
large `.npy` practical on an 8 GB machine, and it adds no logic.

## 4. Supported input formats

| Task | `INPUT` | `TARGETS` |
|---|---|---|
| `tabular_classification` | `.npy` `(samples, features)` | `.npy` `(samples,)` of class names (str) or integer indices |
| `regression` | `.npy` `(samples, features)` | `.npy` `(samples,)`, `(samples, 1)` or `(samples, outputs)`, native units |
| `classification` (image) | directory `root/<class>/<images>` | not accepted |

Not supported, and not silently inferred: CSV, DataFrames, `.npz`, pickled arrays, JSON. A CSV reader would be
a data-ingestion subsystem (delimiter/header/encoding/missing-value policy, string labels) that Forge does not
have; `numpy.save()` is one line and is unambiguous about dtype and shape.

Minimal external example:

```bash
forge model evaluate diabetes.forge X.npy y.npy
forge model evaluate housing.forge X.npy y.npy --json
forge model evaluate pets.forge held_out_dir --device cpu
```

JSON schema: one document on stdout whose keys are exactly the result dataclass's fields, in field order
(classification: `task, samples, loss, accuracy, baseline_accuracy, classes, confusion_matrix, precision,
recall, support`; regression: `task, samples, loss, mse, mae, baseline_mse`). `confusion_matrix` is a list of
lists of ints (row = true class, column = predicted class, both in `classes` order). Documented in
`docs/development/cli.md`.

## 5. Architecture

```text
forge model evaluate ─► argparse ─► cmd_evaluate ─► inspect_model(task)  [routing of file formats only]
                                        │
                                        ├─► _load_npy / directory check
                                        └─► load_predictor()  ─► ArtifactPredictor.evaluate()  ─► result dataclass
                                                                                                        │
                                             text: _print_evaluation   |   JSON: _evaluation_payload ◄──┘
```

What the CLI contains: a `.npy` reader, a text formatter, and a JSON converter that iterates
`dataclasses.fields(result)` (so a field added to a result type appears in the JSON with no CLI change, and the
CLI cannot drift from the API). What it deliberately does **not** contain: prediction, preprocessing, target
transforms, class mapping, baselines, confusion matrix, or any metric. The CLI never mentions
`target_transform`: M116 works because `evaluate()` already applies the inverse.

One piece of knowledge is duplicated and is the price of task-specific file formats:
`_NUMERIC_EVALUATION_TASKS`/`_IMAGE_EVALUATION_TASKS` restate which tasks `evaluate()` supports. If `evaluate()`
gains a task, the CLI must be taught how to read its input; until then it reports the task as unsupported.
`cmd_predict()` has the same shape.

### R² — decision

Not added. `RegressionEvaluationResult` has no R²; adding it means extending a public frozen dataclass and
defining a policy for constant targets (`baseline_mse == 0`) — an API decision, not a CLI detail, and the
request forbids computing it only in the CLI. It is derivable from what is printed:
`R² = 1 - MSE / Baseline MSE` (single-output). Deferred (section 17).

## 6. API/CLI parity methodology

Every parity test compares the CLI's JSON to `predictor.evaluate(...)` **field by field**
(`assert_matches_result`: the key set must equal the dataclass's fields; arrays with `assert_array_equal`,
floats within `rel=1e-6`) — success alone is never the assertion. Because "both agree" is vacuous if both are
wrong, each is *also* checked against an independent oracle: hand-computed confusion matrix / precision /
recall / support for a deliberately non-alphabetical class list, and NumPy MSE/MAE/baseline from the artifact's
raw `metadata.json` (`independent_native_prediction`). A separate sentinel test replaces
`ArtifactPredictor.evaluate` with a function returning made-up numbers and asserts they are exactly what the CLI
prints and that it received exactly the files' arrays (no caller-side conversion).

## 7. Classification validation

Tabular: the 8-row hand-computed fixture from M113 (classes `zebra, apple, mango, unused`; a class no output
can select; raw rows that answer "zebra" for everything if preprocessing were skipped). CLI = hand values =
API, for class-name and class-index targets, in text and JSON. Image: folders `cat`/`dog` against artifact
order `dog, cat` — the matrix, support and per-class rows follow the artifact order; an artifact class with no
folder is allowed (support 0); an unexpected folder is refused, naming it. Real data (section 11): Pima
holdout via the installed wheel printed **72.08% / 62.34% / loss 0.5858 / `[[78,18],[25,33]]`**, identical to the
README's Python-API numbers.

## 8. Regression validation

Plain regression with a persisted `Normalize`: `(n,)` and `(n, 1)` targets give identical results; a 2-output
model is scored over samples and outputs (matches NumPy); text output format checked line by line.

## 9. M116 native-unit validation (mandatory)

Artifact: `train_tabular_regressor(..., target_transform="standardize")` on a target of ~2×10⁵ (the M116 test
data), so the model's own output is z-scores (|·| < ~20). The CLI's `mse`, `mae`, `loss`, `baseline_mse` equal
the NumPy calculation on the hand-inverted predictions (`rel=1e-5`), equal the API, and are ~10⁹ (a z-space
number would be < 10). The **text** output is checked against the same oracle (initially it only asserted
`MSE > 1e3`, which still holds when the inverse is skipped — mutation testing exposed that; tightened).
Real data (section 11): California housing, 3,000 train / 2,000 held-out, seed 0 → **MSE 3.50×10⁹, MAE
$42,632, baseline 1.39×10¹⁰ (R² = 0.749, the figure M116 measured)**, all in dollars.

## 10. Error-handling validation

Each case asserts exit ≠ 0, stderr starting `Error: `, no `Traceback`, **empty stdout** (also under `--json`):
missing artifact; garbage artifact; missing input / targets file; tabular artifact with no `TARGETS`;
wrong feature count; 1-D input; NaN/±Inf in `X` and in `y`; `y` 2-D / too short / wrong width / 3-D; unknown
class name; out-of-range class index; garbage `.npy`, CSV, `.npz`, pickled-object `.npy`, empty file; image
input that is a file / missing / has no class folders / no images / an unreadable image / an unexpected class
folder; `TARGETS` given for an image artifact; segmentation artifact; task-less artifact; classification
artifact without a class list; argparse misuse (exit 2, no traceback); a fresh-process error. Messages are the
evaluation API's own `DataError`/`PersistenceError` text where the API detects the problem, so some name
`ArtifactPredictor.evaluate()` (limitation).

## 11. External installed-wheel validation

Wheel built from the working tree (`python -m build --wheel`), installed into a fresh venv; cwd outside the
repository; the `forge` **console script** from the venv; `forge.__file__` =
`…\venv\Lib\site-packages\forge\__init__.py`. A standalone `reference_evaluate.py` (only `load_predictor()` +
`.evaluate()`) is the oracle. Three real artifacts, each compared field-for-field, CLI vs oracle: Pima
tabular classifier (154 held-out rows), M116 standardized California-housing regressor (2,000 rows), and the
bundled cat/dog image classifier on 200 real petimages (100 per class; the artifact's training split is not
recorded, so the 89.0% is a pipeline check, not a generalisation claim) — **no mismatching field in any of the
three**. The same is automated in `tests/test_packaging_smoke.py` (synthetic artifacts, so it needs no external
data): 5 tests, including a bad-invocation test, all through the installed console script.

## 12. CPU / CUDA validation

Installed wheel, real CUDA (940MX): for all three artifacts `--device cuda` equals the oracle run on CUDA, and
`--device cuda` equals `--device cpu` within the project's 1e-4 tolerance (counts exact). In-repo
`tests/test_cli_evaluate_cuda.py` (6 tests, hardware-run): tabular, regression, M116 native units, image,
CUDA-saved artifact evaluated with explicit `--device cpu` (and default → CUDA, the saved device), and
artifact hash unchanged after CUDA evaluation. Explicit `--device cuda` without a CUDA backend is the existing
`load_model()` error (no fallback); not re-tested here because this machine has CUDA and the path is not
CLI code.

## 13. Artifact immutability validation

SHA-256 of the artifact **and** of the inputs, plus the set of files in the directory, is identical after
text, `--json`, `--device cpu` and image runs (in-process); the M116 artifact hash is unchanged across repeated
evaluations that return identical results; the installed-wheel runs record hashes before/after for all three
real artifacts (`diff` empty), also after the CUDA runs. Evaluation runs in a fresh process every time, so no
model/mode/RNG state can carry over; the in-process repeatability test covers determinism.

Repeated-load check: `load_model` is counted by a spy (exactly 1 call, even with `--batch-size 1` over 8
samples). Wall times (idle machine, installed wheel): housing 2,000 rows — import 0.21 s, load 0.007 s,
evaluate 0.005 s, CLI total 0.56 s; 200 images — import 0.19 s, load 0.48 s, evaluate 2.14 s, CLI total
3.24 s (the difference is interpreter start-up and argument parsing). Nothing to fix.

## 14. Tests added (57)

| File | Tests | What |
|---|---|---|
| `tests/test_cli_evaluate.py` | 46 | parity, hand/NumPy oracles, text + JSON, batch size, M116 native units, image by-name, JSON validity (numpy scalars, non-finite), thin-interface sentinel, single load, read-only, the error matrix of section 10, fresh process |
| `tests/test_cli_evaluate_cuda.py` | 6 | CUDA parity with API and CPU, M116 on CUDA, image on CUDA, CUDA-saved → CPU, hash unchanged |
| `tests/test_packaging_smoke.py` | +5 | installed console script outside the repo vs a standalone reference evaluator (3 tasks), install identity, bad invocation |

No existing test was modified (the two edits to `test_packaging_smoke.py` besides the new tests are two added
imports).

### Mutation testing

Each mutation was applied to the real source, the new CPU test file run, and the source restored
(`git diff` afterwards shows only the intended change). All 7 turn the suite red:

| # | Mutation | Red tests |
|---|---|---|
| 1 | CLI re-implements the tabular-classification evaluation itself (same numbers, bypassing `evaluate()`) | 5 (the sentinel test is the one that catches a *correct* bypass) |
| 2 | target-transform inverse skipped in `_predict_tensor_core` | 2 (M116 JSON and text) |
| 3 | persisted preprocessing skipped | 6 |
| 4 | class list sorted alphabetically | 7 |
| 5 | JSON: ndarray passed through / NaN allowed / numpy scalar passed through | 15 / 1 / 1 |
| 6 | CLI perturbs `accuracy`/`mse` after evaluation | 8 |
| 7 | CLI appends a byte to the artifact | 3 |

Findings from the mutation pass, both fixed in the tests: the M116 text-output assertion was too loose (#2
initially failed only 1 test), and a first version of an image-output test matched `"cat"` inside the word
`classification`.

## 15. Full-suite result

`python -m pytest tests/ -q`: **3,178 passed, 0 failed, 0 skipped**, CUDA available (9 m 25 s). Baseline
3,121 + 57 new = 3,178 (`--collect-only` agrees).

## 16. Limitations

- `.npy` and image directories only; every other format is refused (by design, section 4).
- Error text often names `ArtifactPredictor.evaluate()`: the API's own messages are shown verbatim rather
  than rewritten (rewriting would be a second copy of them).
- The CLI's list of evaluable tasks is a copy of `evaluate()`'s (section 5).
- No R² (section 5); `--json` output for large confusion matrices is tall (`indent=2`, as `predict` does).
- Text output is a report for humans, not a stable format; `--json` is the stable one.
- Only the tasks `evaluate()` supports; a legacy task-less artifact must be resaved with `task=`.

## 17. Deferred work

R² on `RegressionEvaluationResult` (public-API change); CSV/`.npz` input (data-ingestion subsystem); a single
`--targets` inferred from a column. None is started.

## 18. Final decision

Implemented as an interface layer with no second evaluation engine.

Decision: IMPLEMENTED

Framework changes: none (no file under `forge/` other than `forge/cli/model.py` changed; `ArtifactPredictor`,
result dataclasses, artifact format untouched)
CLI changes: new `forge model evaluate MODEL INPUT [TARGETS] [--device] [--batch-size] [--json]` in
`forge/cli/model.py` (`cmd_evaluate`, `_load_npy`, `_jsonable`, `_evaluation_payload`, `_print_evaluation`)
Tests added: 57 (46 CPU CLI, 6 CUDA CLI, 5 installed-wheel), plus a 7-mutation protection check
Full-suite result: 3,178 passed, 0 failed, 0 skipped (baseline 3,121), CUDA available
Classification result: tabular (Pima holdout 72.08%, baseline 62.34%, loss 0.5858, `[[78,18],[25,33]]`) and
image (folder names matched by class name) equal the Python API and independent oracles
Regression result: 1-D, `(n, 1)` and multi-output targets equal NumPy and the API; California housing native
MSE 3.50×10⁹ / MAE 42,632 / baseline 1.39×10¹⁰
M116 target-transform result: `standardize` artifact reported in native units through the CLI, which contains
no target-transform code; text and JSON equal the independent NumPy oracle and the API
External consumer status: PASSED — clean venv, installed wheel, `forge` console script, cwd outside the repo,
`forge.__file__` in site-packages; tabular + M116 regression + image artifacts match a standalone reference
evaluator with no mismatching field
CPU/CUDA status: PASSED on real hardware (940MX) — CLI CUDA = oracle on CUDA, CLI CUDA = CLI CPU within 1e-4;
CUDA-saved artifact evaluated on CPU
Artifact immutability: PASSED — SHA-256 of artifact and inputs unchanged after every evaluation, CPU and CUDA,
in-process and installed
Remaining limitations: `.npy`/image-directory inputs only; API-worded error text; no R²; task list copied from
`evaluate()`
Concrete trigger for next work: a user who cannot produce `.npy` (their data is only a CSV), or who needs R²
from the evaluation result, filing that as a concrete request
