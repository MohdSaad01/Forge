# Milestone 89 — External Developer Workflow Validation and Friction Removal

## Objective

Every prior milestone since M78 built the pieces of a developer-facing
workflow: train -> portable `.forge` artifact (M78/M81) -> inspect (M85) ->
predict, unified across classification/regression/segmentation (M86/M87) ->
the same, task-aware, at the CLI (M88). M89 does not add a new piece. It asks
whether those pieces, used exactly as documented by someone who did not build
them, actually form a working product. Concretely: **use Forge like an
external developer would, discover real friction, and fix only what is
actually found** -- no framework readiness survey, no speculative feature
work, no code cleanup for its own sake.

## Exact workflows exercised

For each of the three task families, the full documented pipeline was run
literally, using only public APIs/CLI commands/saved files:

```text
dataset -> train.py --output-dir ... -> .forge artifact
    -> forge model inspect (fresh venv, outside the repo)
    -> forge model predict (fresh venv, outside the repo)
```

- **Classification** -- `examples/image_folder_classification` (`--generate
  --samples-per-class 60 --epochs 5`), then `forge model predict
  image_folder_model.forge query_shape.png` on a new, never-trained-on image.
- **Regression** -- `examples/regression` (`--epochs 8`), then `forge model
  predict regression_model.forge input.json`.
- **Segmentation** -- `examples/segmentation` (`--epochs 3`), then `forge
  model predict segmentation_model.forge query_image.png --output mask.png`.

Also exercised: `forge.train()`'s own README "First model" snippet, run
verbatim; the fresh-consumer artifact-copy workflow (below); CUDA
inspect/predict on a CUDA-trained artifact; and a deliberate set of
user-mistake error cases (Section 12 of the milestone brief).

## Environment used

This repository's own development machine (Windows 10, Python 3.13.5,
i5-7200U/8GB RAM, NVIDIA 940MX/CUDA 12.6 -- see
`docs/development/development-environment.md`), plus a genuinely separate,
freshly created Python virtual environment (see below) for the
fresh-consumer half of the validation.

## Fresh-environment/process setup

```powershell
python -m venv freshvenv
freshvenv\Scripts\pip.exe install -e C:\Zeus\Forge     # the README's documented install step, run verbatim
freshvenv\Scripts\python.exe -c "import forge; print(forge.__version__)"
```

Result: `forge 0.1.0`, dependency set exactly `numpy`, `pillow`, `pip`, `forge`
-- no `pytest`/`matplotlib` leakage from the dev environment, confirming
`pyproject.toml`'s declared dependency set is complete and CPU-only
installation pulls in nothing extra.

Artifacts (`.forge` files) and query inputs (a PNG, a JSON file) produced by
training runs in the main repository were then **copied into an isolated
directory outside the repository** (`.../scratchpad/m89/consumer/`), with no
copy of `examples/`'s source code alongside them. Every `forge model
inspect`/`forge model predict` invocation below was run from that directory,
through `freshvenv`'s own installed `forge` console script -- a genuine
fresh-process, fresh-environment, no-training-process-alive, no-Forge-source-
visible consumer, per the milestone brief's Section 3/4 requirement.

## Classification workflow result

Clean. `forge model inspect image_folder_model.forge` correctly reported
task, preprocessing (`Resize(64,64) -> Normalize`), and the `circle/square/
triangle` class vocabulary; `forge model predict image_folder_model.forge
query_shape.png` and its `--json` form both produced a correct, confident
prediction from the fresh venv with no access to `train.py`/`generate_dataset
.py`. One friction found here (see below): the printed module tree
(13 layers) was structurally scrambled.

## Regression workflow result

**Blocked**, then fixed. `forge model predict regression_model.forge
input.json`, with `input.json` written the ordinary way a Windows developer
would (PowerShell's `Out-File -Encoding utf8`), failed with `Error:
regression input must contain numeric JSON data.` -- despite the file
containing exactly the documented, valid flat-list format. See **Friction
discovered** below.

## Segmentation workflow result

Clean. `forge model inspect segmentation_model.forge` and `forge model
predict segmentation_model.forge query_image.png --output mask.png` (and
`--json`) both worked correctly from the fresh venv/consumer directory,
producing a real mask file. No issues found.

## Artifact portability result

All three artifacts round-tripped correctly through: training process ->
`.forge` file -> copied to an unrelated directory -> fresh venv's `forge`
CLI, with no manual reconstruction of preprocessing, classes, or task
required in any case -- confirming M78/M81/M85/M87's portability contracts
hold under a genuinely external consumer, not just the same-process/same-repo
tests that already covered this. A CUDA-trained regression artifact was
additionally copied into the same fresh venv/consumer directory and
predicted correctly with `--device cuda`, `--device cpu` (a live transfer),
and the omitted-`--device` default (uses the recorded `"cuda"`) -- all three
agreeing exactly, and `forge model inspect` on the same file required no
CUDA at all.

## CLI result

All three tasks' documented `forge model predict` invocations work as
written, with the one regression-input fix below. Error handling (missing
artifact, missing input, invalid `--device`, missing/invalid segmentation
`--output`) was already clear and specific in every case tested -- see
**Errors intentionally tested**.

## Python API result

`forge.predict_model()`/`forge.inspect_model()`/`forge.train()` all matched
their documented contracts and the CLI's own behavior for every artifact with
`task=` set. One deliberate, already-documented (M86/M87/M88) divergence was
re-confirmed, not newly discovered: `forge.predict_model()` (Python) still
applies its architecture-based legacy-workflow guess for an artifact with no
`task=`, while `forge model predict` (CLI) refuses outright rather than
guess. This is intentional -- the CLI dispatcher has no way to show a caller
the ambiguity the way a Python caller inspecting the artifact's own code
might -- and is unchanged by this milestone; see **Friction discovered**
below for why it was not "fixed" into agreement.

## Documentation-following result

The README's "First model" snippet was copied verbatim into the fresh venv
and run unmodified -- reproduced the documented loss curve shape and printed
a prediction, no drift found. `examples/regression`'s, `examples/segmentation
`'s, and `examples/image_folder_classification`'s READMEs' documented
commands (`train.py --output-dir`, `forge model inspect`, `forge model
predict`) were followed literally throughout this validation with no
deviation required.

## Errors intentionally tested

| Case | Result |
|---|---|
| Wrong artifact path | `Error: artifact not found: ...` |
| Wrong input path | `Error: input file not found: ...` |
| Wrong input type (image given for regression) | `Error: regression input must contain numeric JSON data.` |
| Malformed JSON | Same clear message as above |
| Missing task metadata (legacy artifact) | `Error: artifact does not declare a task. Use the task-specific prediction API or resave the model with task metadata.` |
| Invalid `--device` | argparse: `error: argument --device: invalid choice: 'tpu' (choose from cpu, cuda)` |
| Missing segmentation `--output` | `Error: segmentation prediction requires --output <path> to save the predicted mask.` |
| Invalid output directory | `Error: cannot write to '...': directory '...' does not exist.` |

Every case above was already clear before this milestone -- no changes were
needed or made to this error surface.

## Friction discovered

1. **[Blocker] Regression JSON input with a UTF-8 BOM fails with a
   misleading error.** `forge/cli/model.py::_parse_regression_input()`
   opened the input file with `encoding="utf-8"`. A file written the
   ordinary way on this project's own primary, documented development
   platform (Windows -- PowerShell's `Out-File`/`>`, or Notepad's "UTF-8"
   save option) carries a leading UTF-8 byte-order mark. `json.load()` on
   such a file raises `json.JSONDecodeError`, which the existing `except
   (OSError, ValueError, UnicodeDecodeError)` clause already caught -- but
   only to report the same generic "not numeric JSON" message a truly
   malformed file gets, giving no hint that the data was in fact valid. This
   silently blocks the documented `forge model predict regression.forge
   input.json` workflow for a large, ordinary class of Windows-authored
   input files.
2. **[Significant friction] `forge model inspect`/`checkpoint inspect`
   scramble a model's module/parameter order once it has 10+ numbered
   children.** `forge/serialization/archive.py` writes `metadata.json` with
   `json.dumps(..., sort_keys=True)` (a deliberate, reasonable choice for
   stable diffs of the file itself). `forge/cli/_archive_info.py::
   walk_modules()`/`walk_parameters()` then iterated that already-string-
   sorted `children` dict directly, so any `Sequential` with 10+ children --
   an entirely ordinary case, demonstrated by this milestone's own retrained
   `examples/image_folder_classification` artifact (13 layers) -- printed
   its architecture as `0, 1, 10, 11, 12, 2, 3, ...` in both the text and
   `--json` output. This directly undermines Section 11's core promise
   ("what model is this?") for any nontrivial-sized model; it is a display
   bug only (`load_model()`'s actual reconstruction keys children by name,
   not iteration order, so no model was ever computed incorrectly -- verified
   by this milestone's own correct predictions throughout).
3. **[Investigated, not new friction] CLI vs. Python API legacy-artifact
   divergence.** `forge.predict_model()` (Python) still falls back to its
   documented, tested architecture heuristic for an artifact with no `task=`
   metadata; `forge model predict` (CLI) refuses with a clear error instead.
   This is a real, demonstrated divergence between the two entry points --
   but it is the *intended*, extensively documented outcome of M86/M87/M88
   (see `docs/development/m88-unified-artifact-prediction-cli.md`'s "Legacy
   artifact behavior" section): the CLI dispatcher has no way to show a
   caller the `classes=None`-vs-regression ambiguity the way a Python caller
   reading `predict_model()`'s own docstring might, so it deliberately never
   guesses there, while the Python function's fallback remains a documented,
   tested, permanent (if narrow) capability. Forcing agreement here would
   either reintroduce the exact misdispatch risk M86/87/88 refused to ship,
   or remove a working, tested Python capability with no complaint driving
   it -- neither is a fix. No change made.

No other blocker or significant friction was found across classification,
regression, segmentation, artifact portability, fresh-process/fresh-venv
inference, CPU/CUDA parity, or the error cases tested.

## Severity of each friction

- Finding 1: **Blocker** (the documented workflow cannot complete for a large
  class of real, ordinary input files).
- Finding 2: **Significant friction** (the workflow completes, but its answer
  to "what is this model?" is actively wrong for any 10+-layer model).
- Finding 3: **Not friction** -- an intentional, already-documented design
  decision, re-confirmed rather than newly discovered.

## Which friction was fixed

Both Finding 1 and Finding 2. Finding 3 was investigated and deliberately
left unchanged (see above).

## Why each fix was selected

Both are small, evidence-based, single-layer fixes with no architectural
impact -- exactly the M89 mandate. Finding 1 blocks the single most
externally-visible regression workflow (`forge model predict`) for anyone
authoring input on this project's own primary development platform. Finding
2 was demonstrated against this milestone's own real retrained artifact, not
a constructed edge case -- 10+ `Sequential` children is the common case for
any CNN of realistic depth, not a rare one.

## Files changed

- `forge/cli/model.py` -- `_parse_regression_input()` now opens the input
  file with `encoding="utf-8-sig"` instead of `"utf-8"`, transparently
  stripping a leading BOM if present (identical behavior otherwise).
- `forge/cli/_archive_info.py` -- new `_child_sort_key()`/`_sorted_children()`
  helpers; `walk_modules()`/`walk_parameters()` now iterate a module's
  children in construction order (numeric-aware for `Sequential`-style
  digit-string names) instead of the archive's on-disk (alphabetically
  sorted) key order.
- `tests/test_cli_predict.py` -- new
  `test_cli_predict_regression_accepts_utf8_bom`.
- `tests/test_cli.py` -- new
  `test_model_inspect_orders_ten_or_more_sequential_children_numerically`
  (text and `--json`).
- `docs/development/cli.md` -- documents BOM tolerance for regression input
  and construction-order guarantee for module/parameter listings.
- `docs/development/progress.md` -- this milestone's entry.
- `docs/development/m89-external-developer-workflow-validation.md` -- this
  file.

## Public API changes

None. Both fixes are internal to `forge/cli/`; `forge.predict_model()`,
`forge.inspect_model()`, `forge.train()`/`forge.train_and_save()`, and every
other public function used during this validation are unchanged.

## CLI changes

`forge model predict` (regression) now accepts a UTF-8-BOM-prefixed JSON
input file identically to one without a BOM. `forge model inspect`/`forge
checkpoint inspect` (text and `--json`) now report module/parameter order
correctly for 10+-child `Sequential` modules. No flag, argument, or exit-code
behavior changed.

## Tests added/modified

- `tests/test_cli_predict.py::test_cli_predict_regression_accepts_utf8_bom`
  (new) -- a JSON file with a leading `\xef\xbb\xbf` BOM parses identically
  to one without.
- `tests/test_cli.py::test_model_inspect_orders_ten_or_more_sequential_children_numerically`
  (new) -- a 13-child `Sequential` is reported in construction order (`0`
  through `12`) in both text and `--json` `forge model inspect` output.

## Full-suite result

2,478 collected, 2,477 passed, 1 failed -- the same pre-existing
`test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`
allocator-measurement flake documented since ~M63 (reproduced passing cleanly
in isolation immediately after the full-suite run; no M89 regression).

## Fresh-process result

All three task workflows (classification/regression/segmentation) were
predicted successfully from a genuinely separate OS process (a fresh venv's
installed `forge` console script) against artifacts copied outside the
repository, with no access to `examples/`' source code -- see **Fresh-
environment/process setup** above. This is in addition to, not a replacement
for, `tests/test_cli_predict.py`'s/`test_model_inspection.py`'s own existing
`subprocess`-based fresh-process tests.

## CPU verification

Every workflow above (training, inspection, prediction, both fixes'
regression tests, the full suite) ran and passed on this machine's CPU
backend.

## CUDA verification

A regression artifact was trained on CUDA (`--device cuda`, 940MX,
hardware-verified), copied into the same fresh venv/consumer directory as the
CPU artifacts, and predicted correctly via `forge model predict` with the
default device (uses the recorded `"cuda"`), explicit `--device cuda`, and
explicit `--device cpu` (a live device transfer) -- all three producing the
identical prediction. `forge model inspect` on the same CUDA-recorded file
required no CUDA backend at all, confirming M85's documented guarantee.

## End-to-end acceptance result

For all three task families:
```text
dataset -> train.py --output-dir ... -> .forge artifact
    -> copied to an isolated directory, no repo source alongside it
    -> fresh venv's `forge model inspect` -> correct architecture/task/preprocessing/classes description
    -> fresh venv's `forge model predict` -> correct, useful prediction
```
completed successfully using only documented public APIs/CLI commands and
saved files -- no Tensor/autograd/CUDA-kernel/serialization-internals/private
module access was needed anywhere in this validation.

## Remaining friction

- **Legacy (pre-M87) artifacts remain unpredictable through the CLI by
  design** (Finding 3 above, and M88's own documented limitation) -- a
  developer holding one must resave it with `task=` or drop to the
  task-specific Python API. This is an intentional, permanent restriction,
  not an oversight this milestone left unfixed.
- **Regression input is still single-file JSON only** -- no CSV/NDJSON. No
  workflow exercised during this validation demonstrated a need for another
  format; per the milestone brief, this remains explicitly deferred.
- **No batch/directory prediction.** Not exercised as a need during this
  validation; remains explicitly deferred.

## Explicitly deferred features

Per the milestone brief's Section 18 and this validation's own findings, none
of the following were added, because no exercised workflow demonstrated a
concrete need for them: CSV/NDJSON regression input, batch/directory
prediction, segmentation overlays/visualization, model serving/REST/HTTP
inference, a model registry/versioning/zoo, or any other speculative
capability.

## What an external developer can now do

Take any of Forge's three vision-named workload families (classification,
regression, segmentation), train it with the documented `train.py`/
`forge.train_and_save()` path, hand the resulting single `.forge` file to a
completely separate machine/environment/process, `pip install -e .` Forge
there (or use an existing install), and get a correct, human-readable
prediction via `forge model predict` -- including on a Windows-authored JSON
input file with a BOM, and with an architecturally correct `forge model
inspect` description regardless of how many layers the model has -- without
ever reading Forge's own source code or importing anything outside `forge`'s
public surface.

## Suggested commit message

```
fix: strip UTF-8 BOM from regression predict input and report module/parameter order correctly for 10+-child Sequential models (M89 external-workflow validation)
```
