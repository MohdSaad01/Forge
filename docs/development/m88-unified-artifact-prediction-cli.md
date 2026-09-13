# Milestone 88 — Unified Artifact Prediction CLI

## Objective

Bring `forge.predict_model()` (Milestone 86) and its authoritative `task=`
metadata (Milestone 87) to the actual end-user boundary: `forge model
predict`. Before this milestone, the CLI's `predict` subcommand was
hard-wired to "classify one image" (`--image IMAGE`, always via
`forge.predict_artifact()`) -- Milestones 86/87 both evaluated retrofitting
it onto the unified dispatcher and both declined, because a classification
artifact saved with `classes=None` and no `task=` is architecturally
indistinguishable from a legacy regression artifact, and `predict_model()`'s
legacy fallback would have silently misidentified it. Milestone 87's
explicit `task=` metadata is what finally makes a safe generic CLI
dispatcher possible, and this milestone builds it.

## What changed

`forge model predict MODEL INPUT` is now genuinely task-aware:

```bash
forge model predict model.forge image.jpg                       # classification
forge model predict regression.forge input.json                 # regression
forge model predict segmentation.forge image.png --output mask.png  # segmentation
```

The command reads `forge.inspect_model(MODEL).task` -- the same public,
read-only inspection API `model inspect`'s own "Task" line already uses --
and uses it as the **sole** routing signal:

- `task` present -> delegate straight to `forge.predict_model()` (Milestone
  86), which returns the matching workflow immediately, since an explicit
  `task` is authoritative and skips architecture inspection entirely (see
  `forge/training/inference.py::_determine_workflow()`).
- `task` absent (a genuinely legacy artifact, saved before Milestone 87) ->
  fail clearly. This command never falls back to `predict_model()`'s own
  `_legacy_infer_workflow()` architecture heuristic -- doing so would
  reintroduce exactly the `classes=None`-misidentified-as-regression problem
  Milestones 86/87 refused to ship a CLI dispatcher around.

No new inference logic was written: `cmd_predict` (`forge/cli/model.py`)
composes `inspect_model()` + `predict_model()` + (for regression) a small
JSON-to-`Tensor` input parser + (for segmentation) `forge.data.save_image()`
-- the same three functions a Python caller would use directly.

## CLI syntax

```text
forge model predict MODEL INPUT [--device {cpu,cuda}] [--output PATH] [--json]
```

- `MODEL` -- path to a `.forge` artifact. Must declare `task=` (Milestone
  87) -- see **Legacy artifacts** below.
- `INPUT` -- an image file path for classification/segmentation, or a path
  to a JSON file of numeric data for regression.
- `--device {cpu,cuda}` -- optional, matches `forge.load_model()`'s own
  default (the device the model was saved from). No CPU fallback if `cuda`
  is requested but unavailable.
- `--output PATH` -- required only for a segmentation artifact; the
  predicted mask is written here via `forge.data.save_image()`.
- `--json` -- machine-readable output instead of the human-readable text
  below.

No `--task` flag exists or is needed -- the artifact's own persisted
metadata determines the task, by design (Milestone 87's whole point).

## Supported tasks and input/output formats

### Classification
`INPUT` is an image file, decoded the same way `ImageFolder` does. With a
saved class vocabulary (`classes=...`):
```text
Prediction: cat
Confidence: 94.2%
```
Without one (`classes=None` is a real, valid state -- see
`predict_artifact()`'s own docstring):
```text
Prediction: class index 3 (no class-name vocabulary was saved with this model)
```
`--json`:
```json
{"task": "classification", "class": "cat", "confidence": 0.942}
```
or, with no saved vocabulary, `"class"` and `"confidence"` are `null` and an
`"index"` field is included instead.

### Regression
`INPUT` is a path to a JSON file of numeric data:
- a flat list (`[1.2, 3.4, 5.6, 7.8]`) is treated as **one** unbatched
  sample and given a leading batch dimension;
- a nested list (`[[1.2, 3.4], [5.6, 7.8]]`) is treated as **already
  batched** and passed through as-is.

```text
Prediction: [[0.4309830963611603]]
```
`--json`:
```json
{"task": "regression", "prediction": [[0.4309830963611603]]}
```
Malformed JSON, non-numeric JSON, or a non-JSON file (e.g. an image) all
produce the same clear error rather than a traceback:
```text
Error: regression input must contain numeric JSON data.
```

### Segmentation
`INPUT` is an image file; `--output PATH` is **required**:
```text
Predicted mask saved to: mask.png
```
`--json`:
```json
{"task": "segmentation", "output_path": "mask.png"}
```
Omitting `--output` fails clearly (`--output` is named in the error);
writing to a directory that does not exist fails clearly too, mirroring
`model convert`'s own `--output` validation.

## Legacy artifact behavior

An artifact saved before Milestone 87 has no `task` key at all. Rather than
guessing (which could silently misidentify a `classes=None` classification
artifact as regression -- the exact scenario Milestones 86/87 both
document), this command fails clearly:
```text
Error: artifact does not declare a task.
Use the task-specific prediction API or resave the model with task metadata.
```
The fix is either to resave the model (`forge.save_model(..., task=...)`,
or `train_and_save()`/`save_and_verify()`'s own `task=`), or to call the
task-specific Python API directly (`forge.predict_artifact()`/
`forge.predict_tensor_artifact()`/`forge.predict_image_artifact()`), all
three of which are completely unaffected by this limitation.

## Public API changes

None. `forge.predict_model()`, `forge.inspect_model()`,
`forge.predict_artifact()`/`predict_tensor_artifact()`/
`predict_image_artifact()`, and `forge.data.save_image()` are all
Milestone 82-87 APIs used exactly as they already existed. The CLI's own
`_parse_regression_input()`/`_print_classification_result()` helpers
(`forge/cli/model.py`) are internal to the CLI adapter, not part of the
public `forge` API surface.

## CLI syntax change (breaking)

The old `forge model predict MODEL --image IMAGE` syntax is replaced by
`forge model predict MODEL INPUT` (a plain positional, since `INPUT` is no
longer always an image). This is a deliberate, milestone-scoped breaking
change to the CLI's own argument surface -- the only place `--image` was
ever required. `tests/test_classification_metadata.py`'s pre-existing CLI
tests were updated to the new positional syntax and to declare
`task="classification"` on their fixtures (see **Tests** below for why).

## Files changed

- `forge/cli/model.py` -- `cmd_predict()` rewritten to be task-aware;
  `add_parser()`'s `predict` subcommand: `--image` replaced by a positional
  `input`, plus new `--output`/`--json`; new `_parse_regression_input()`/
  `_print_classification_result()` helpers; module docstring rewritten.
- `tests/test_classification_metadata.py` -- CLI predict tests updated to
  the new positional syntax, `Prediction:`/`Prediction: class index` wording,
  and explicit `task="classification"` fixtures; one new test for the
  genuinely-ambiguous legacy (no `task` at all) case.
- `tests/test_cli_predict.py` (new) -- the bulk of this milestone's test
  coverage (see **Tests**).
- `examples/image_folder_classification/train.py` -- its own printed
  "inspect the generated artifacts" hint updated to the new positional
  syntax.
- `examples/image_folder_classification/README.md`, `docs/development/cli.md`,
  `README.md` -- documentation updated for the new syntax/behavior.
- `docs/development/progress.md` -- this milestone's entry.
- `docs/development/m88-unified-artifact-prediction-cli.md` -- this file.

## Architecture

```text
forge model predict MODEL INPUT
        |
        v
inspect_model(MODEL) -> .task
        |
   task is None?  --yes--> CLIError("does not declare a task")
        | no
        v
task == "classification"/"regression"/"segmentation"?
        |
        v
predict_model(MODEL, parsed_input, device=...)   <- same public dispatcher
        |                                            a Python caller uses
        v
format + print (text or --json); segmentation also save_image()s the mask
```

The CLI never re-implements `_determine_workflow()` -- it only decides
whether calling it at all is safe (i.e., whether `task` is already known).

## Tests

- `tests/test_cli_predict.py` (new, 22 tests, all CPU except one CUDA test):
  command routing for all three tasks; regression input parsing (flat list,
  nested list, malformed JSON, non-numeric JSON, an image given where JSON
  is expected, missing input file); segmentation `--output` requirement and
  invalid output directory; missing artifact; legacy (no `task`) artifact
  failing clearly; a simulated future/unsupported task (via monkeypatching
  the CLI's own `inspect_model` call, since a real artifact can never carry
  a task outside `TASK_TYPES`); `--json` output for all three tasks
  (including the classes-less classification case); explicit `--device cpu`;
  `--device cuda` with CUDA unavailable; a genuine `subprocess`-launched
  fresh-process run (`python -m forge model predict ...`) covering all three
  task shapes in one test; one CUDA-hardware-gated test verifying `--device
  cuda` reaches the real CUDA inference path (skips cleanly without CUDA).
- `tests/test_classification_metadata.py` -- pre-existing CLI predict tests
  updated in place (positional syntax, new wording, explicit
  `task="classification"`), plus one new test for the legacy no-task case.

## Full-suite result

2,476 collected, 2,475 passed, 1 pre-existing `test_dataloader_prefetch.py`
allocator-measurement flake (documented since ~M63, isolation-verified, no
M88 regression).

## Fresh-process verification

`tests/test_cli_predict.py::test_cli_predict_fresh_process_all_three_task_shapes`
launches `python -m forge model predict ...` as a genuine, separate OS
process (`subprocess.run`) for all three artifact shapes -- proving the
workflow depends on nothing but the saved `.forge` file, not any in-process
state.

## CPU verification

All tests above ran on this machine's CPU backend; the three example train
scripts (`examples/regression`, `examples/segmentation`,
`examples/image_folder_classification --generate`) were each retrained fresh
(small/smoke configurations) and their resulting artifacts predicted
end-to-end through the real `forge model predict` CLI (not just tests) --
see **Real end-to-end examples demonstrated** below.

## CUDA verification

`test_cli_predict_reaches_cuda_inference_path` ran (not skipped) on the
940MX and passed: `forge model predict regression.forge input.json --device
cuda` produces a real CUDA-backed prediction. No new CUDA kernels were
added or needed.

## Real end-to-end examples demonstrated

All three retrained fresh via their own `train.py` (smoke configurations),
then predicted via the real, installed CLI (not a test):

**Classification** (`examples/image_folder_classification --generate`):
```text
$ python -m forge model predict image_folder_model.forge new_mixed_resolution_query.png
Prediction: triangle
Confidence: 47.5%
```
(matches `train.py`'s own end-of-run `predict_artifact()` demo exactly)

**Regression** (`examples/regression`):
```text
$ python -m forge model predict regression_model.forge input.json
Prediction: [[0.4309830963611603]]
```
(matches `train.py`'s own end-of-run demo prediction, `0.4310`, to the
printed precision)

**Segmentation** (`examples/segmentation`):
```text
$ python -m forge model predict segmentation_model.forge segmentation_new_image.png --output cli_predicted_mask.png
Predicted mask saved to: cli_predicted_mask.png
```

## Limitations

- **Legacy artifacts (no `task=`) cannot be predicted through this command**
  at all, by design -- see **Legacy artifact behavior**. This is a hard,
  documented restriction, not an oversight.
- **Regression input is single-file JSON only.** There is no `--input-format
  csv` or similar; a caller with tabular data in another format must convert
  it to JSON first, or call `forge.predict_tensor_artifact()` directly with
  a NumPy array.
- **`--json` is per-task-shaped, not a generic schema.** Each task's JSON
  payload has its own fields (documented above); there is no unified
  envelope across tasks beyond the shared `"task"` key.
- **No batch/directory input.** `INPUT` is always exactly one file (one
  image, or one JSON document); predicting many inputs in one invocation
  still requires a Python loop over `forge.predict_model()` or repeated CLI
  invocations.

## Deferred ideas

- A `--input-format` option for regression (CSV, NDJSON) if a real workload
  ever needs one -- no such need exists today.
- Batch prediction (`INPUT` as a directory or a list of paths) if a real
  workload demonstrates the current one-call-per-input shape is
  insufficient.
- Segmentation overlay/visualization output beyond the raw `{0,1}` mask --
  explicitly out of scope per this milestone's brief.

## Suggested commit message

```
feat: make `forge model predict` task-aware over classification/regression/segmentation artifacts
```
