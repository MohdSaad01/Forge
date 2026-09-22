# Forge Command-Line Interface (Milestone 19, extended in 72, 117 and 121)

## Purpose
A thin command-line adapter over Forge's existing public persistence and
benchmark APIs -- `forge model ...`, `forge checkpoint ...`, `forge
benchmark ...`. It exposes operations a Python caller could already do with
`forge.save_model()`/`forge.load_model()`/`forge.save_checkpoint()`/
`forge.load_checkpoint()` and the `benchmarks` package, for use without
writing a script. It implements no new framework logic: every command either
reads existing archive metadata read-only, or calls one of the functions
above directly. See `forge/cli/main.py`'s module docstring for the package
layout.

## Installation / entry point
```bash
python -m forge ...          # always available once `forge` is importable
forge ...                    # available after `pip install -e .` (or a
                              # normal install), via the `forge` console-script
                              # entry point declared in pyproject.toml
```
`import forge` never imports `forge.cli` -- the CLI package is only reached
through one of the two entry points above, so a plain library import pays no
CLI-related cost.

## Help
```bash
forge --help
forge model --help
forge model inspect --help
forge model convert --help
forge model predict --help
forge model evaluate --help
forge model train --help
forge checkpoint --help
forge checkpoint inspect --help
forge checkpoint convert --help
forge benchmark --help        # forwarded to `python -m benchmarks --help`
```

## Model inspection
```bash
forge model inspect PATH [--json]
```
Reports the saved format version, the module hierarchy (types and dotted
names), every parameter's name/shape/dtype, total parameter count, the
recorded device, the saved training/eval mode, whether a preprocessing
pipeline was saved (`forge.save_model(..., preprocessing=...)`, Milestone
71), and the saved class-name vocabulary if any (`forge.save_model(...,
classes=...)`, Milestone 72). A regression artifact trained on standardised
targets also reports its target transform (a `Target transform:` line, and
`"target_transform"` in `--json`, `null` otherwise; Milestone 116) -- `forge
model predict` on such an artifact already returns native units, and `forge model
convert` carries the transform over. A tabular artifact trained with `feature_names=`
(Milestone 119) also lists them -- a `Feature names: Pregnancies, Glucose, ...` line after
`Input: 8 feature(s)`, and `"input_feature_names": [...]` in `--json` (`null` for an artifact
that records none, which is every artifact saved before M119 and any trained from unnamed
arrays; names are never invented) -- and `convert` carries them over. Never prints tensor values.

The module tree and parameter list (text and `--json`) are always ordered by
construction order (e.g. a `Sequential`'s children as `0, 1, 2, ..., 12`, not
`0, 1, 10, 11, 12, 2, ...`) regardless of the archive's on-disk key order --
see Milestone 89.

**Never requires CUDA, regardless of the model's recorded device**, and
never reconstructs a live `Module` or runs any computation: it reads only
`metadata.json` from the archive (via `forge.serialization.archive
.read_archive`, the same primitive `load_model()` itself uses internally),
so a CUDA-saved model can be inspected on a CPU-only machine. This also
means `inspect` does not require the saved module types to be registered in
the current process -- unlike `load_model()`/`model convert`, which do.

## Checkpoint inspection
```bash
forge checkpoint inspect PATH [--json]
```
Reports the checkpoint format version, the model architecture (same as
`model inspect`), the optimizer type and hyperparameter configuration, epoch
and global step, the recorded device, whether RNG state is present, and how
many of the optimizer's parameters have saved per-parameter state (e.g. Adam
`m`/`v`) -- a count, never the state values themselves.

Like `model inspect`, this reads only `metadata.json` and never requires
CUDA. It deliberately never calls `forge.load_checkpoint()`: that function's
documented contract includes **overwriting `forge.random`'s process-global
RNG state** as part of restoring a checkpoint (the mechanism that makes
resumed training deterministic) -- exactly the kind of state mutation an
`inspect` command must never trigger. `checkpoint convert` (below), which is
not a read-only operation, does call `load_checkpoint()`/`save_checkpoint()`
directly and inherits that documented RNG side effect, same as any Python
caller of those functions would.

## Model conversion
```bash
forge model convert MODEL --device {cpu,cuda} --output OUTPUT [--feature-names NAME [NAME ...]]
```
Loads `MODEL` explicitly onto `--device` (`forge.load_model(MODEL,
device=...)`) and saves the result to `OUTPUT` (`forge.save_model(...)`).
Performs no computation beyond the transfer `load_model`/`save_model`
already do. `--device` is required -- there is no default and no "use CUDA
if convenient" behavior; requesting `--device cuda` with no CUDA backend
available fails with a clear error and a non-zero exit status rather than
silently falling back to CPU.

**`--feature-names` (Milestone 120): attach feature names without retraining.** A tabular
artifact saved before Milestone 119, or trained from unnamed arrays, cannot check a named CSV's
column order (M119's own limitation). If the caller knows exactly what its input columns are,
`--feature-names` retrofits them onto `OUTPUT`:
```bash
forge model convert diabetes_old.forge --device cpu --output diabetes_named.forge \
    --feature-names Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age
```
This is a metadata-only edit riding on `convert`'s existing reload/re-save: `MODEL`'s weights are
loaded and saved back unchanged (`convert` was already exactly this sequence; only the
`feature_names=` argument to `save_model()` differs), so no prediction changes for a correctly
ordered NumPy input. `save_model()` itself still enforces the Milestone 119 rules -- the artifact
must be a tabular task (`regression`/`tabular_classification`) whose input width Forge can read
(a `Sequential` starting with `Linear`), and the names must be exactly one per input column, all
distinct and non-empty -- so a wrong count or a duplicate is refused before anything is written:
```text
Error: save_model() feature_names= has 3 name(s), but the model's first Linear layer takes 8 input feature(s).
```
Omitting `--feature-names` keeps `convert`'s pre-existing behavior exactly: whatever feature names
(or lack of them) `MODEL` already had are carried over to `OUTPUT` unchanged. There is no automatic
way to *discover* the right names (e.g. from a CSV) -- they must be supplied explicitly, by someone
who knows the model's column order.

## Model prediction (Milestone 72, made task-aware in Milestone 88, extended to sequence generation in Milestone 90 and tabular classification in Milestone 91)
```bash
forge model predict MODEL INPUT [--device {cpu,cuda}] [--output PATH] [--length N] [--columns NAME [NAME ...]] [--json]
```
Predicts from a saved `.forge` artifact using only what the file itself
already carries -- the developer never has to know or pass which of
classification/regression/segmentation/sequence/tabular_classification
`MODEL` is. This command
reads the artifact's own persisted `task` metadata (`forge.save_model(...,
task=...)`, Milestone 87, via `forge.inspect_model()`) and delegates
straight to `forge.predict_model()` (Milestone 86), which dispatches to the
matching task-specific function unchanged. No new inference logic lives
here.

**`INPUT`'s shape depends on the artifact's task:**

- **classification** -- `INPUT` is an image file path, decoded the same way
  `ImageFolder` does. With a saved class vocabulary (`classes=...`):
  ```text
  Prediction: dog
  Confidence: 94.2%
  ```
  Without one, falls back to the raw index (no confidence is claimed, since
  there is no label to attach it to):
  ```text
  Prediction: class index 3 (no class-name vocabulary was saved with this model)
  ```
- **regression** -- `INPUT` is a path to a JSON file of numeric data: a flat
  list (`[1.2, 3.4, 5.6, 7.8]`) is treated as one unbatched sample and given
  a leading batch dimension; a nested list (`[[1.2, 3.4], [5.6, 7.8]]`) is
  treated as already batched. A leading UTF-8 byte-order mark (BOM) is
  tolerated and stripped (Milestone 89) -- common on Windows, where
  `Out-File`/`>`/Notepad's "UTF-8" option all write one by default. Prints
  the raw numeric prediction:
  ```text
  Prediction: [[0.8134]]
  ```
- **segmentation** -- `INPUT` is an image file path; `--output PATH` is
  **required** and receives the predicted mask, written via
  `forge.data.save_image()`:
  ```text
  Predicted mask saved to: mask.png
  ```
- **sequence** (Milestone 90) -- `INPUT` is literal seed text on the command
  line, not a file path (there is nothing to decode); tokenized as
  individual characters (`list(INPUT)`), matching the char-level vocabulary
  `examples/char_rnn` uses. `--length N` (default 200) is the number of new
  characters to generate. Prints the full generated text, seed included:
  ```text
  Generated: a tensor produces a sample...
  ```
  A word-level (or other custom-tokenized) sequence artifact is not
  representable through this command's char-level convention -- call
  `forge.predict_sequence_artifact()` directly with a pre-tokenized seed
  list instead.
- **tabular_classification** (Milestone 91) -- `INPUT` is a path to a JSON
  file of numeric data, parsed exactly like **regression**'s (a flat list is
  one sample; a nested list is already batched), but the output is
  classification-shaped -- one prediction per input row, since the input may
  legitimately be more than one sample:
  ```text
  Prediction: warning
  Confidence: 87.1%
  ```
  A multi-row input prints one numbered block per row instead:
  ```text
  Sample 0: Prediction: normal
  Sample 0: Confidence: 91.0%
  Sample 1: Prediction: fault
  Sample 1: Confidence: 76.4%
  ```
  Without a saved class vocabulary, falls back to raw indices per row, the
  same as **classification**. See `docs/development/
  m91-tabular-classification-artifact-inference.md` for why this needed its
  own task value: `task="classification"`'s `INPUT` has always meant "an
  image file path" (`predict_artifact()`), and a tabular classification
  artifact's input is an already-batched numeric feature vector instead --
  before this task existed, this exact artifact shape (numeric input,
  `task="classification"`) failed with `predict_artifact() requires image to
  be a file path (str or os.PathLike), got ndarray`.

**A `.csv` `INPUT` for the two tabular tasks (Milestone 119).** `regression` and
`tabular_classification` also accept a CSV file (chosen by the `.csv` extension alone, as for
`evaluate`) with a header row and **only feature columns** -- there is no target to name, and
every column is a feature (`forge.data.load_csv_features()`). The header gives the columns'
*names*, and the artifact then matches them to the model's by name -- the same rule, in the same
place, as `forge model evaluate` (below): an artifact that records feature names reorders a CSV
that has exactly those names in any order, and rejects anything else (a missing, unknown, extra
-- including an `id`, or a target column left in the file -- or case/whitespace-different name)
with one `Error:` line, exit 1, empty stdout. Nothing is dropped, renamed or guessed. A JSON
`INPUT` has no column names, so it is read exactly as before and checked for width only. An
artifact that records no names (saved before M119, or trained from arrays) takes the CSV as
given and prints one `Warning: ... records no feature names ... cannot be verified` line on
stderr after the result -- stdout, including `--json`, is unchanged.

**`--columns` (Milestone 120): select which CSV columns are features.** A real CSV file often
carries a column that is not a model feature (an `id`, say). Without `--columns`, every column of
a `.csv` `INPUT` is a feature (`load_csv_features()`'s default) -- an `id` column left in the file
is then an unexpected column, rejected by the name matching above. `--columns NAME [NAME ...]`
selects **exactly** those header columns as the features, in **exactly** that order, regardless of
where they sit in the file; every other column (the `id` included) is never read, never validated:
```bash
forge model predict diabetes.forge new_patients.csv \
    --columns Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age
```
Selection (`--columns`, which raw file columns become `X`) and the Milestone 119 name-alignment
above (what order the artifact needs them in) are two separate steps in one pipeline -- selection
never reorders for a specific artifact, alignment never reads the file. There is still no automatic
id/timestamp detection: a column not listed in `--columns` is simply never read, whatever it is
named. `--columns` is rejected for a JSON `INPUT` (there is no header to select from) and for a
task with no CSV input at all (classification/segmentation/sequence).

`--json` prints a stable, machine-readable result instead of the text above,
e.g. `{"task": "classification", "class": "dog", "confidence": 0.942}` (or
`{"task": "regression", "prediction": [[0.8134]]}` /
`{"task": "segmentation", "output_path": "mask.png"}` /
`{"task": "sequence", "seed": "a tensor", "generated": "a tensor produces..."}` /
`{"task": "tabular_classification", "predictions": [{"class": "warning", "confidence": 0.871}]}`).

Requires `MODEL` to have been saved with `preprocessing=...` for
classification/segmentation -- there is nothing to reproduce automatically
otherwise, and this command never guesses or silently skips preprocessing;
it fails with a clear error instead. A sequence artifact has no
preprocessing concept; it requires `classes=...` instead (the saved token
vocabulary -- see **Model prediction** limitations below). A
tabular_classification artifact's `preprocessing=`/`classes=` are both
optional, exactly like regression's/classification's own.

**Custom model classes.** `MODEL` must be built entirely from module types
already registered in the running `forge` CLI process (Forge's built-ins --
`Linear`, `Conv2d`, `Sequential`, etc. -- are pre-registered; a hand-written
composite class is not, per `docs/architecture/persistence.md`'s **Known
limitations**). This is a pre-existing, general persistence requirement, not
specific to any one task -- it already applied to `model convert` before
Milestone 90. It does mean the bare CLI cannot `predict`/`convert` a
custom-class artifact like `examples/char_rnn`'s `CharRNN`: that example's
`infer.py` imports its own `model.py` (registering `CharRNN` as a side
effect) before calling the same `forge.predict_model()` this command uses
internally -- see that script and `examples/resnet`/`examples/autoencoder`'s
READMEs, which document the identical constraint for their own custom
classes.

**Legacy artifacts (no `task=`, saved before Milestone 87).** This command
never falls back to an architecture-based guess to pick a task: doing so
could silently misidentify a classification artifact saved with
`classes=None` as a regression artifact (both are `Linear`-terminated with
no saved `classes` -- the exact ambiguity Milestones 86/87 documented and
refused to ship a CLI dispatcher around). A legacy artifact with no `task=`
fails clearly instead:
```text
Error: artifact does not declare a task.
Use the task-specific prediction API or resave the model with task metadata.
```
Resave the model with `forge.save_model(..., task=...)` (or
`train_and_save()`/`save_and_verify()`'s own `task=`), or call the
task-specific Python API directly (`forge.predict_artifact()`/
`forge.predict_tensor_artifact()`/`forge.predict_image_artifact()`/
`forge.predict_tabular_classification_artifact()`), which are unaffected by
this limitation.

This is a thin adapter over exactly the same Python sequence
`examples/image_folder_classification/infer.py`, `examples/segmentation/
infer.py`, and `examples/regression/train.py`'s own demo each already
demonstrate as standalone scripts.

## Model evaluation (Milestone 117; CSV input Milestone 118)
```bash
forge model evaluate MODEL INPUT [TARGETS] [--target COLUMN] [--columns NAME [NAME ...]] [--device {cpu,cuda}] [--batch-size N] [--json]
```
Scores a saved `.forge` artifact on labeled data and prints the metrics -- the
command-line form of `forge.load_predictor(MODEL).evaluate(X, y)` (Milestone
113). It is an interface, not a second evaluation system: the command reads
`INPUT`/`TARGETS` from disk, calls `load_predictor()` once and `evaluate()` once,
and prints the result dataclass it gets back. Preprocessing, the class order, a
persisted regression target transform (Milestone 116) and every metric come from
the artifact and the existing evaluation code; nothing is recomputed here, so the
CLI's numbers are the Python API's.

**Inputs** (the artifact's own `task` decides how they are read; a task-less
legacy artifact is refused, exactly as for `predict`):

| Artifact task | `INPUT` | `TARGETS` |
|---|---|---|
| `tabular_classification` | `.csv` with a header (targets in the `--target` column), or `.npy`, shape `(samples, features)` | `.npy`, shape `(samples,)`: class **names** (strings, each in the artifact's classes) or integer class **indices**; not accepted with a `.csv` |
| `regression` | `.csv` with a header (targets in the `--target` column), or `.npy`, shape `(samples, features)` | `.npy`, shape `(samples,)`, `(samples, 1)` or `(samples, outputs)`, always in the caller's **native units**; not accepted with a `.csv` |
| `classification` (image) | a directory laid out as `root/<class_name>/<image files>` | **not accepted** -- labels are the folder names, matched to the artifact's classes by name |
| `segmentation`, `sequence` | no evaluation semantics -- use `forge model predict` | |

`.npy` means a single array written by `numpy.save()`. The file is loaded with
`allow_pickle=False` (data, never code), so an object array is refused; `.npz`
archives and other formats are not read. Make the files from your data with
`numpy.save("X.npy", X)`. Shape, dtype, finiteness and label checks are the
evaluation API's own.

**CSV input** (Milestone 118): an `INPUT` whose extension is `.csv` (case-insensitive;
the extension alone picks the reader, the content is never sniffed) is read by
`forge.data.load_csv()` into the same `(X, y)` arrays a `.npy` pair would give, and
`--target COLUMN` names the target column:

```bash
forge model evaluate diabetes.forge holdout.csv --target Outcome
forge model evaluate housing.forge held_out.csv --target median_house_value --json
```

- The first row is the header (never guessed); the file is comma-delimited UTF-8 with
  standard double-quote quoting. `--target` is matched exactly and is required -- there
  is no "last column" default -- and is removed from the features. **Every other column
  is a numeric feature, in file order** -- unless `--columns` (below) is given.
- A regression artifact's target column must be numbers (kept in native units, exactly as
  in a `.npy`). A classification artifact's is integers (class **indices**, into the
  *artifact's* class list) or text (class **names**, matched to the artifact's classes
  by name); the class order is never taken from the file.
- Nothing is repaired: an empty cell, `NaN`/`Inf`, text in a feature column, a row with
  the wrong number of fields, duplicate column names, a missing/duplicated target
  column, a header-less or non-comma-delimited file, malformed quoting or a non-UTF-8
  file each stop the run with one `Error:` line naming the file, line and column.
  Forge's one missing-value mechanism (`missing_columns=` at training time) replaces a
  numeric sentinel such as `0`, which is an ordinary number to the reader.
- `TARGETS` is not accepted together with a `.csv` (the targets are in the file), and
  `--target` is not accepted with a `.npy` or an image directory.
- **Column names are checked against the artifact's, when it has any (Milestone 119).** The
  CLI passes the header names of the feature columns (the `--target` excluded, file order)
  to `ArtifactPredictor.evaluate(..., feature_names=)` untouched; it never reorders, drops
  or renames a column itself. Against an artifact trained with `feature_names=`, a CSV with
  exactly its names in another order is scored exactly like the correctly ordered one, and
  anything else -- a missing, unknown or extra column (an `id` is an extra column: it is
  never recognised or dropped, so remove it from the file), a duplicate, a name that differs
  by case or whitespace -- is one `Error:` line naming the columns, before any row is scored.
  An artifact saved before M119, or trained from unnamed arrays, records only a feature
  *count*: it cannot detect a reordered CSV, uses the columns as given, and after a
  successful run prints one `Warning:` line on stderr saying so (stdout is unchanged).
  A `.npy` input has no column names and is checked for width only -- as before.
- **`--columns` selects which CSV columns are features (Milestone 120).** A real CSV file
  often carries a column that is not a model feature (an `id`, say); without `--columns` it
  is read as every other column, matched against the artifact's names above (rejected as an
  unexpected column if the artifact has names; rejected outright if it is not numeric).
  `--columns NAME [NAME ...]` selects **exactly** those header columns as the features, in
  **exactly** that order, regardless of where they sit in the file -- every other column,
  the `id` included, is never read, never validated:
  ```bash
  forge model evaluate diabetes.forge diabetes_with_id.csv --target Outcome \
      --columns Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age
  ```
  `--target` must not appear in `--columns` (the target is always removed automatically,
  never listed as a feature too); an unknown or duplicated name in `--columns` is one
  `Error:` line naming it. Selection and the name-alignment bullet above stay two separate
  steps: `--columns` decides which raw file columns become `X` (unchanged by which artifact
  they will be scored against); alignment then decides what order the artifact needs them in
  (unchanged by how they were selected). `--columns` is rejected for a `.npy` input or an
  image-classification directory, neither of which has a header to select from. `forge model
  predict` accepts the identical `--columns` flag with the identical rule (above) -- the two
  commands cannot differ because neither implements column selection itself.

The output, the `--json` keys and the numbers are exactly those of the `.npy` form:
the CSV path is an input reader and adds no key, metric or preprocessing.

```bash
forge model evaluate diabetes.forge X.npy y.npy
```
```text
Task: tabular_classification
Samples: 154
Accuracy: 72.08%
Baseline accuracy: 62.34%
Loss: 0.5858

Classes:
  class        precision  recall  support
  no_diabetes     0.7573  0.8125       96
  diabetes        0.6471  0.5690       58

Confusion matrix (rows = true class, columns = predicted class):
               no_diabetes  diabetes
  no_diabetes           78        18
  diabetes              25        33
```
```bash
forge model evaluate housing.forge X.npy y.npy
```
```text
Task: regression
Samples: 2000
MSE: 3.50082e+09
MAE: 42632.2
Baseline MSE: 1.39342e+10
Loss: 3.50082e+09
```
For an artifact saved with `target_transform="standardize"` these are native
units (here dollars, squared for MSE): no flag or inverse transform is involved.
`Baseline accuracy` / `Baseline MSE` are what "always predict the evaluated
data's majority class / mean target" would score on exactly these samples.
R-squared is not printed (it is not part of the evaluation result); for a single-output model it is
`1 - MSE / Baseline MSE`.

**`--json`** prints one JSON document, and nothing else, to stdout. Its keys are
exactly the fields of the result dataclass (`forge.training.
ClassificationEvaluationResult` / `RegressionEvaluationResult`), in field order:

- classification: `task`, `samples`, `loss`, `accuracy`, `baseline_accuracy`,
  `classes` (list of names, the artifact's order), `confusion_matrix` (list of
  lists of ints; row = true class, column = predicted class, both in `classes`
  order), `precision`, `recall`, `support` (lists parallel to `classes`);
- regression: `task`, `samples`, `loss`, `mse`, `mae`, `baseline_mse`.

Numbers are plain JSON numbers (never `NaN`/`Infinity`: a non-finite metric is an
error instead). Errors never touch stdout, so a pipeline can trust that stdout is
either one valid document or empty.

`--device` follows `model predict`: optional, and by default the model is loaded
onto the device it was saved from; `--device cpu` evaluates a CUDA-saved artifact
on CPU, and `--device cuda` requires a real CUDA backend (no fallback).
`--batch-size N` only bounds memory (the API's default is used when omitted).
Evaluation is read-only: the artifact, the inputs and the directory around them
are unchanged afterwards.

## Model training (Milestone 121)
```bash
forge model train DATA.csv --task {classification,regression} --target COLUMN --output PATH \
    [--columns NAME [NAME ...]] [--device {cpu,cuda}] [--epochs N] [--batch-size N] \
    [--learning-rate LR] [--seed N] [--target-transform standardize] [--json]
```
The command-line door onto `forge.train_tabular_classifier_csv()` / `forge.train_tabular_regressor_csv()`
(Milestone 121) -- the CSV-to-artifact counterpart of `model predict`/`model evaluate`. It is a thin adapter
and nothing else: `--task` alone picks which of the two Python functions to call (there is no
architecture/data-driven guess, and no third training system lives here), and every other flag is forwarded
to that function **only when the caller actually passed it** -- an omitted flag is exactly that function's own
default (100 epochs for classification, 500 for regression, batch size 32, learning rate `1e-3`, seed `0`,
`cpu`), never a second, CLI-specific default that could drift from the Python API's.

```bash
forge model train patients.csv --task classification --target Outcome --output patients.forge \
    --columns Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age
```
```text
Trained tabular_classification model -> 'patients.forge'
Samples: 768 (train 614, validation 154)
Features: 8
Epochs completed: 100
Classes: no_diabetes, diabetes
Validation accuracy: 78.39% (baseline 65.10%)
```
```bash
forge model train housing.csv --task regression --target median_house_value --output housing.forge \
    --columns median_income housing_median_age total_rooms total_bedrooms population households latitude longitude \
    --target-transform standardize
```
```text
Trained regression model -> 'housing.forge'
Samples: 3000 (train 2400, validation 600)
Features: 8
Epochs completed: 500
Validation MSE: 3.5e+09 (baseline 1.33e+10)
Validation MAE: 40501.1
```

**`DATA` must be a `.csv` file** (chosen by extension alone, exactly as `predict`/`evaluate` choose their CSV
path) -- no other input format trains through this command yet; `--target` (the header name of the target
column, removed from the features automatically) is required. `--columns` is the identical Milestone 120
selection `predict`/`evaluate` already use: omit it and every non-target column becomes a feature, in file
order (the M118 default); give it and **exactly** those header columns become the features, in **exactly**
that order -- so a production CSV carrying an `id` column needs no rewriting to train from, exactly as it
needs none to predict/evaluate from:
```bash
forge model train diabetes_with_id.csv --task classification --target Outcome --output pima.forge \
    --columns Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age
```
`--target` must not appear in `--columns` (the same `load_csv()` rejection `predict`/`evaluate` already
surface, naming the column); an unknown or duplicated `--columns` entry is likewise the reader's own error.

**Feature names are persisted automatically -- there is no `--feature-names` flag on this command.** The CSV's
own selected column names (in the order they end up in) become the artifact's `InputSchema.feature_names`
(Milestone 119) with no extra step: a later reordered CSV or named array is aligned by
`predict`/`evaluate`/`ArtifactPredictor` exactly as for any other named artifact.

**`--target-transform standardize`** (regression only, Milestone 116): train on standardized targets, fitted
on the training split only, with the fitted transform saved in the artifact so `predict`/`evaluate` and every
number in the printed/`--json` result are in native units. Rejected with `--task classification`.

**CLI scope.** Only the flags above are exposed -- `task`, `target`, `columns`, `output`, `device`, `epochs`,
`batch-size`, `learning-rate`, `seed`, and (regression only) `target-transform`. Every other Python-API
keyword (`classes`, `val_fraction`, `missing_columns`, `missing_value`, `patience`, `model=`, `verbose`) has a
stable, well-tested default that is not yet a demonstrated CLI need; a workflow that needs one of them calls
`forge.train_tabular_classifier_csv()`/`forge.train_tabular_regressor_csv()` (or the array
`train_tabular_classifier()`/`train_tabular_regressor()`) directly. This is a deliberate scope decision, not
an oversight -- the brief this milestone implements explicitly warns against turning the CLI into "a giant
argument surface merely because the underlying Python API has many parameters."

`--json` prints one JSON document built from the returned `TabularClassificationResult`/
`TabularRegressionResult`'s own fields (`task`, `artifact_path`, `samples`, `features`, `train_samples`,
`validation_samples`, `epochs_completed`, plus `classes`/`validation_accuracy`/`baseline_accuracy` for
classification or `outputs`/`validation_mse`/`validation_mae`/`baseline_mse` for regression) -- not the full
dataclass (`history`/`model` are Python objects, not JSON).

`--output`'s directory must already exist (the same rule `model predict --output`/`model convert --output`
already enforce); the file itself is created by this command. `--device` follows `model predict`/`model
evaluate`: optional, and CUDA is required only when `--device cuda` is passed explicitly (no fallback).

## Checkpoint conversion
```bash
forge checkpoint convert CHECKPOINT --device {cpu,cuda} --output OUTPUT
```
Loads `CHECKPOINT` explicitly onto `--device` (`forge.load_checkpoint()`)
and saves the reconstructed model, optimizer (type, hyperparameters, and
per-parameter state), and training progress (epoch, global step) back out
(`forge.save_checkpoint()`), preserving everything the checkpoint format
itself preserves. Same explicit-device/no-fallback policy as `model
convert`.

## Benchmark invocation
```bash
forge benchmark [any argument accepted by `python -m benchmarks`]
```
A pure pass-through: every argument after `benchmark` is forwarded unparsed
to `benchmarks.run.main()`, so `--categories`/`--output`/benchmark selection
logic lives in exactly one place (`benchmarks/run.py`), not duplicated here.
See `docs/performance/benchmarking.md` for the underlying suite.

`benchmarks/` is a top-level package outside `forge`, deliberately excluded
from `forge`'s own installation (`import forge` never touches it, matching
Milestone 11's original design). `forge benchmark` therefore **only works
when run from within the Forge repository** -- against an installed `forge`
package with no `benchmarks/` directory alongside it, it fails with a clear
"benchmark suite is not available" error rather than a raw `ImportError`.

## Device behavior
Every device-affecting command that changes a model's device (`model
convert`, `checkpoint convert`) requires an explicit `--device {cpu,cuda}`
-- Forge's existing "no implicit device fallback" policy
(`docs/architecture/persistence.md`), unchanged by the CLI. `model predict`'s and `model evaluate`'s
`--device` is optional, matching `forge.load_model()`'s/`forge.predict()`'s
own default (the device the model was saved from) -- prediction does not
*convert* the file, so there is no ambiguity to force an explicit choice
about. Inspection commands never touch a device or a backend at all.

## CUDA requirements
- `model inspect` / `checkpoint inspect`: never require CUDA, for any input.
- `model convert` / `checkpoint convert` with `--device cpu`: never require
  CUDA, even for a CUDA-recorded input file.
- `model convert` / `checkpoint convert` with `--device cuda`: require a
  real CUDA backend; if unavailable, fail with a clear error and exit
  status 1 (never a silent CPU fallback).
- `model predict` / `model evaluate`: require CUDA only if `--device cuda` is passed explicitly,
  or if omitted and the model was saved from `"cuda"` -- identical policy to
  `forge.load_model()`'s own (**Device semantics**,
  `docs/architecture/persistence.md`).
- `model train` (Milestone 121): requires CUDA only if `--device cuda` is passed explicitly (omitted, it
  trains on `cpu` -- `forge.train_tabular_classifier_csv()`/`forge.train_tabular_regressor_csv()`'s own
  default); no fallback if CUDA is requested and unavailable.
- `benchmark`: CUDA-dependent categories (`transfer`, and the CUDA half of
  `forward`/`backward`/`training`) produce no results when CUDA is
  unavailable, exactly as `python -m benchmarks` already behaves
  unmodified.

## Error behavior
All CLI-facing errors -- missing file, malformed/corrupt archive,
unsupported format version, invalid `--device` choice, unavailable CUDA,
unknown command, missing required argument -- print a concise `Error: ...`
message to stderr and exit non-zero. Argument-parsing errors (an invalid
`--device` choice, a missing required flag, an unknown subcommand) are
reported by `argparse` itself (exit status 2, its own convention); every
other user-facing failure is a `forge.exceptions.ForgeError` (most commonly
`PersistenceError`) or a CLI-specific `CLIError`, both caught by
`forge/cli/main.py`'s single handler and printed without a Python traceback.
An unexpected internal exception (not one of the above) is left to propagate
with its full traceback, for diagnosability during development -- it is
never silently swallowed.

## Limitations
- **No automatic training resume.** The checkpoint format does not capture
  enough to reconstruct an arbitrary caller's dataset/model-construction
  code, and the CLI deliberately does not invent a second training engine to
  paper over that gap (Milestone 19's explicit non-goal). `forge checkpoint
  inspect`/`convert` are the CLI's checkpoint story; resuming an actual
  training run remains a Python `forge.load_checkpoint()` call followed by
  the caller's own `Trainer`/training-loop code, exactly as before this
  milestone.
- **`forge benchmark` is a development-only command** (see **Benchmark
  invocation** above) -- it is not available from a `pip`-installed `forge`
  package used outside the Forge repository.
- No experiment tracking, config/YAML system, hyperparameter sweeps,
  distributed-training commands, cloud storage, dataset downloading, job
  scheduling, daemon/server mode, model serving, or REST API -- all
  explicitly out of scope for this milestone.
- **`model predict` supports exactly the four tasks `forge.predict_model()`
  does** (Milestones 88/90): classification, regression, segmentation,
  sequence. There is no generic `forge predict` command spanning every
  Forge workload (autoencoders, ...) -- those have a different natural
  input shape with no single saved-artifact-describable file convention yet.
- **`model train` (Milestone 121) trains only from a `.csv` file**, and only the two tabular tasks
  (`classification`/`regression`) -- there is no CLI training door for image classification, segmentation or
  sequence workloads, and no `.npy`/array input (an in-memory array trains through the Python
  `train_tabular_classifier()`/`train_tabular_regressor()` directly). Its flag surface is a deliberately
  narrower subset of the Python API's -- `classes=`, `val_fraction=`, `missing_columns=`, `missing_value=`,
  `patience=`, `model=` and `verbose=` are not exposed; a workflow needing one of them calls
  `forge.train_tabular_classifier_csv()`/`forge.train_tabular_regressor_csv()` (or the array API) directly.
- **`model evaluate` reads `.csv` files (Milestone 118), `.npy` files and image
  directories only.** A CSV is a header row plus numeric columns and one named
  target: no categorical/string features, no missing-value handling, no dates, no
  other delimiters or encodings, no DataFrames, no `.npz`, no train/holdout
  splitting. Column selection (`--columns`, Milestone 120) is explicit only -- there
  is still no automatic id/timestamp/target detection, and no `--drop` (the inverse
  of `--columns`: name what to keep, not what to discard). It evaluates exactly
  the three tasks `ArtifactPredictor.evaluate()` does (tabular classification,
  regression, image classification). Its error messages are the evaluation API's
  own `DataError` text, so they may name `ArtifactPredictor.evaluate()`.
- **`model predict`'s sequence support is char-level-tokenization-only**
  (Milestone 90): the CLI splits `INPUT` into individual characters; a
  vocabulary tokenized a different way (e.g. `examples/word_rnn`'s
  whole-word vocabulary) needs `forge.predict_sequence_artifact()` called
  directly with a pre-tokenized seed list -- see **Model prediction** above.
- **`model predict` requires explicit `task=` metadata** (Milestone 88): a
  legacy artifact saved before Milestone 87 (no `task=`) is not predictable
  through this command -- see **Model prediction** above for why, and
  `docs/development/m88-unified-artifact-prediction-cli.md` for the full
  reasoning.
