# M122 — Unified model training CLI

Status: **IMPLEMENTED.** M121 gave `forge model train` a real command name with only a CSV-shaped
capability behind it: `--task {classification,regression}` dispatched to the two tabular CSV
trainers, but Forge's mature, already-shipping `forge.train_image_classifier()` had no CLI door at
all. M122 closes that specific mismatch -- `forge model train` now also accepts `--task
image-classification`, dispatching an `ImageFolder`-layout directory straight to the existing
`train_image_classifier()` -- and nothing else. No new training engine, no automatic task
detection, no fourth training family.

## 1. Objective

Turn `forge model train` into what its name already implied: a single CLI front door onto every
mature, already-shipping Forge training capability, not just the CSV-tabular one M121 built. Concretely:
give `--task image-classification` a working CLI path onto `forge.train_image_classifier()`, while
leaving M121's tabular contract untouched.

## 2. Existing CLI gap

Before this milestone:

```text
Python API                              CLI
├── train_image_classifier()            (no CLI door)
├── train_tabular_classifier_csv()      forge model train --task classification
└── train_tabular_regressor_csv()       forge model train --task regression
```

`forge.train_image_classifier()` (Milestone 107) is a mature, hardware-verified, real-dataset
(`petimages`) high-level workflow -- exactly as mature as the two tabular CSV wrappers `forge model
train` already exposed -- but reaching it required a Python script. `model predict`/`model evaluate`
already treat image classification as a first-class task (Milestones 88/117); training was the one
command that did not.

## 3. Supported training families after this milestone

| Input | `--task` | Backend API |
|---|---|---|
| `.csv` file | `classification` | `forge.train_tabular_classifier_csv()` (Milestone 121, unchanged) |
| `.csv` file | `regression` | `forge.train_tabular_regressor_csv()` (Milestone 121, unchanged) |
| `ImageFolder`-layout directory | `image-classification` | `forge.train_image_classifier()` (Milestone 107, unchanged) |

Segmentation and sequence training remain Python-only -- see **Deferred training families** below.

## 4. CLI contract

```bash
forge model train DATA.csv --task {classification,regression} --target COLUMN --output PATH \
    [--columns NAME [NAME ...]] [--device {cpu,cuda}] [--epochs N] [--batch-size N] \
    [--learning-rate LR] [--seed N] [--target-transform standardize] [--json]

forge model train IMAGE_DIR --task image-classification --output PATH \
    [--device {cpu,cuda}] [--epochs N] [--batch-size N] [--learning-rate LR] [--seed N] [--json]
```

`--task` is the sole, explicit, authoritative dispatch signal -- never inferred from whether `DATA`
is a file or a directory (see **Task vocabulary** and **Input validation** below). Every other flag
is forwarded to the selected Python function **only when the caller actually passed it**, so an
omitted flag is that function's own default, never a second CLI-specific one -- identical to M121's
own rule, now applied to a third function.

## 5. Task vocabulary

`--task` gained one new choice: `image-classification`, alongside M121's unchanged `classification`/
`regression`. These three values are the CLI's own dispatch vocabulary and are deliberately distinct
from the artifact's *persisted* `task=` metadata that `model inspect`/`model predict`/`model
evaluate` already report -- `--task classification` (CLI) produces an artifact with
`task="tabular_classification"` (persisted), and `--task image-classification` (CLI) produces one
with `task="classification"` (persisted, the same value every other image classifier already uses).
This is not a new inconsistency: M121 already established that `--task classification` (CLI) maps to
persisted `"tabular_classification"`, not `"classification"` -- M122 only adds a third CLI value onto
an existing artifact-vocabulary/CLI-vocabulary split, it did not introduce the split.

## 6. Input validation

`--task image-classification` requires `DATA` to be a directory (`root/<class_name>/<image files>`,
`ImageFolder`'s own contract); `--task classification`/`--task regression` require a `.csv` file, as
before. Every invalid combination the brief called out is rejected before training, naming the
declared task, the detected input, and the incompatible argument:

- a `.csv` file with `--task image-classification` -> `"'<path>' is not a directory"`.
- a directory with `--task classification`/`--task regression` -> `"--task <task> reads a .csv file
  ...; '<path>' is a directory. Use --task image-classification ... instead"`.
- `--target`/`--columns`/`--target-transform` with `--task image-classification` -> each rejected by
  name, explaining what CSV concept it names and that an image directory/task has none.
- a missing `--target` with `--task classification`/`--task regression` -> unchanged from M121: an
  argparse usage error (`SystemExit`, exit status 2), not a `CLIError`, since `--target`'s
  requiredness is conditional on `--task` and argparse's own `required=` cannot express that (see
  `forge/cli/model.py`'s `_parser` default on the `train` subparser and `_cmd_train_tabular()`'s
  first check -- this is the same usage-error path argparse would take itself, not a second, ad hoc
  exit mechanism).

No task is ever guessed from `DATA`'s shape -- `--task` decides which function runs; the file/
directory check only confirms that choice is consistent with what was actually given.

## 7. API routing

`forge/cli/model.py::cmd_train()` is now two lines: it dispatches on `args.task ==
"image-classification"` to `_cmd_train_image()`, and everything else to `_cmd_train_tabular()`
(M121's original body, unmodified except for the relocated `--target`-required check and a clearer
directory-vs-file error message). Each helper is a thin translator -- CLI arguments in, one existing
Python call out, that call's own result dataclass printed. No `GenericTrainer`, no
`TrainingBackend`, no dispatcher abstraction: the two helper functions *are* the routing.

## 8. Tabular compatibility (M121, unchanged)

`_cmd_train_tabular()` is exactly M121's `cmd_train()` body -- same validation order, same two
function calls, same JSON/text output. The only behavioral additions are (1) the `--target`-required
check moved from "argparse `required=True`" to an explicit `args._parser.error(...)` call so its
requiredness can depend on `--task`, which still raises `SystemExit` exactly as before, and (2) a
directory given where a `.csv` file was expected now names the mistake explicitly instead of
reporting "not found". All 17 pre-existing `tests/test_cli_train.py` tests pass unmodified.

## 9. Image classification integration

`_cmd_train_image()` calls `forge.train_image_classifier(data_dir, path=output, **forwarded_kwargs)`
-- the same `--device`/`--epochs`/`--batch-size`/`--learning-rate`/`--seed` forwarding helper M121
already used (`_forwarded_training_kwargs()`, unchanged, since the four shared kwarg names happen to
match both APIs exactly). One deliberate deviation from "never a second CLI-specific default":
`verbose=False` is always passed, overriding `train_image_classifier()`'s own `verbose=True` default.
Justification: `train_image_classifier(verbose=True)` prints per-epoch and skipped-file lines
directly to stdout, which would corrupt `--json`'s "exactly one JSON document on stdout" contract
(the same contract every other `--json` command in this file already guarantees) and would be
inconsistent with the CLI's own single-summary convention even without `--json`. The tabular CSV
wrappers needed no equivalent override because their own Python default is already `verbose=False`.

`ImageFolder`/`Resize`/`Normalize`/the default CNN/`Adam`/`CrossEntropyLoss`/`train_and_save()` are
all reached exactly as a direct `train_image_classifier()` call would reach them -- nothing about
image decoding, splitting, model construction or artifact saving is reimplemented here.

## 10. JSON output

`--json` for `--task image-classification` reports fields drawn straight from `ImageClassifierResult`
(no full-dataclass dump -- `history`/`model` are Python objects, not JSON, matching M121's own rule):
`task` (always `"classification"`, the artifact's persisted value), `artifact_path`, `dataset_size`,
`train_size`, `val_size`, `classes`, `epochs_completed` (`result.history.epochs_completed`),
`skipped_images` (a count), `validation_accuracy` (`result.val_metrics.get("accuracy")`). Tested for
both text and `--json` output in `tests/test_cli_train_image.py`.

## 11. Artifact behavior

A CLI-trained image artifact is saved through the ordinary `train_image_classifier()` ->
`train_and_save()` -> `save_model(..., task="classification", classes=...)` path -- there is no
CLI-only artifact variant. `tests/test_cli_train_image.py::
test_a_cli_trained_image_artifact_is_consumable_from_a_fresh_predictor` trains via the CLI, then
consumes the artifact through `forge.load_predictor()`, `forge model predict`, and `forge model
evaluate`, each as a fresh, independent call with no reference to the training invocation.

## 12. Real petimages validation

Ran the complete chain through the actual installed CLI (`python -m forge model train ... --task
image-classification --json`, subprocess, not the Python API) against the real petimages dataset
(`C:\Zeus\model_training\image_classifier\petimages\{cat,dog}`, ~25,000 images total -- Forge's
established real image-classification dataset, per `docs/development/maintenance.md` section 3),
using the same deterministic-160-image-subset approach as `tests/real_world/petimages_smoke.py`
(first 80 files per class, numeric sort):

```text
forge model train <subset> --task image-classification --output pets.forge \
    --epochs 2 --batch-size 16 --seed 0 --json
-> classes ["cat", "dog"], dataset_size 160, train_size 128, val_size 32

forge model inspect pets.forge --json    -> task "classification", classes ["cat", "dog"]
forge model predict pets.forge <held-out cat/81.jpg> --json  -> {"class": "cat", ...}
forge model predict pets.forge <held-out dog/81.jpg> --json  -> {"class": "cat", ...}
forge model evaluate pets.forge <subset dir> --json           -> accuracy 0.5, baseline 0.5
```

Every step exited 0 and produced valid JSON matching the artifact's own reported classes. 50%
accuracy after 2 epochs on 160 images is expected and not a regression -- as with M107/M115's own
quick real-dataset checks, the subset and epoch count are chosen for workflow-correctness speed, not
a meaningful accuracy number (the full ~25,000-image / real-epoch-count run is `tests/real_world/
petimages_smoke.py`'s own established territory, unmodified and untouched by this milestone).

## 13. Real tabular regression/classification validation

Unchanged: M121's own real-dataset validation (Pima diabetes, StatLib housing) exercised
`_cmd_train_tabular()`'s code path, which is byte-for-byte the same code this milestone kept in
place (see **Tabular compatibility** above). Re-run as part of the full suite below; no new
tabular-specific real-dataset validation was needed since no tabular behavior changed.

## 14. CPU validation

`tests/test_cli_train_image.py` (17 tests, CPU-only): happy path (text + `--json`), routing/mutation
kills (image task never reaches the tabular trainers and vice versa, verified with `monkeypatch`
spies around the real functions -- not mocks that skip training), argument forwarding, omitted-flag
defaults, every invalid combination from the brief, output-directory validation, and fresh-process
artifact consumption (`predict`/`evaluate` on the CLI-trained artifact). Also a same-seed CLI-vs-
Python-API parameter-identity check (`test_cli_image_classification_matches_the_python_api_for_the_
same_seed`).

## 15. CUDA validation

`tests/test_cli_train_image_cuda.py` (1 test, skips cleanly without CUDA): `--device cuda` actually
trains on CUDA (asserts `metadata["device"] == "cuda"` read directly from the saved archive, not
inferred), then the artifact is consumed on CPU through `forge model predict` with no reference to
the training run. **Hardware-verified** on this machine's reference GeForce 940MX
(`forge.backend.cuda.is_cuda_available()` -> `True`) -- 1 passed.

## 16. Fresh-process validation

Every CLI test in `tests/test_cli_train_image.py`/`test_cli_train_image_cuda.py` that checks artifact
consumption does so via `forge.load_predictor()`/`forge model predict`/`forge model evaluate` called
independently of the training invocation, matching M121's own convention (`tests/test_cli_train.py`'s
`test_a_cli_trained_artifact_is_consumable_from_a_freshly_loaded_predictor`).

## 17. Installed-wheel validation

Extended `tests/test_packaging_smoke.py` (the Milestone 93 real-wheel/clean-venv/outside-the-repo
harness) with two new tests:

- `test_installed_cli_trains_an_image_classifier_and_matches_the_python_api` -- builds the real
  wheel, installs into a fresh venv, trains an image classifier both via a direct
  `forge.train_image_classifier()` subprocess call and via `forge model train --task
  image-classification --json` (the installed console script), and asserts identical `classes`/
  `dataset_size`/`train_size`/`val_size`/`validation_accuracy`. A third, independent subprocess then
  runs `model inspect`/`model predict`/`model evaluate` against the CLI-trained artifact.
- `test_installed_cli_model_train_rejects_bad_image_classification_invocations_without_a_traceback`
  -- the same four invalid-combination cases as `test_cli_train_image.py`, run against the installed
  console script, confirming no `Traceback` reaches stderr outside the repository either.

All 35 tests in `test_packaging_smoke.py` pass (33 pre-existing + 2 new).

## 18. Test coverage

- `tests/test_cli_train_image.py` -- 17 new CPU tests.
- `tests/test_cli_train_image_cuda.py` -- 1 new CUDA test (hardware-verified, passing).
- `tests/test_packaging_smoke.py` -- 2 new installed-wheel tests (33 -> 35).
- `tests/test_cli_train.py` -- 17 pre-existing M121 tests, unmodified, all still passing.

## 19. Mutation results

Targeted against the brief's own list (section 18), primarily via real (not mocked) `monkeypatch`
spies that still call through to the genuine training function:

1. **Route image classification to the tabular trainer** -- killed:
   `test_cli_image_classification_never_calls_the_tabular_trainers` monkeypatches both tabular
   functions to raise `AssertionError` if called at all, then runs a real `--task
   image-classification` training end to end.
2. **Ignore the selected task** -- killed by the same test plus its symmetric counterpart,
   `test_cli_tabular_classification_never_calls_the_image_trainer` (monkeypatches
   `train_image_classifier` to raise, runs a real `--task classification` CSV training).
3. **Ignore `--device`** -- killed: `test_cli_forwards_image_training_arguments_to_train_image_
   classifier` spies on the real call and asserts `captured["device"] == "cpu"`; the CUDA test
   independently asserts the saved artifact's `device` metadata is `"cuda"`.
4. **Drop `--output`** -- killed: the same spy asserts `captured["path"] == str(output)`, and every
   happy-path test asserts the file actually exists at that path.
5. **Ignore `--columns` for tabular training** -- already covered and still killed by M121's own
   `tests/test_cli_train.py` (unmodified).
6. **Drop `target_transform` for regression** -- already covered and still killed by M121's own
   `tests/test_cli_train.py` (unmodified).
7. **Create an artifact through a separate image-training implementation** -- killed by the same
   spy-and-delegate test (it calls the real `train_image_classifier()`, so a second implementation
   swapped in would have to reproduce its exact return shape) and by the cross-API parity test
   (`test_cli_image_classification_matches_the_python_api_for_the_same_seed`, which asserts
   byte-identical trained parameters between a CLI run and a direct API call at the same seed).
8. **Accept an invalid task/input combination** -- killed: one test per combination listed in the
   brief (CSV + image-classification, image directory + tabular classification/regression, image
   directory + `--target`, image directory + `--columns`, image directory + `--target-transform`),
   each asserting exit status 1 and a specific, non-generic error substring.

## 20. Files changed

- `forge/cli/model.py` -- `train` subparser: new `image-classification` `--task` choice, `--target`
  relaxed to conditionally required (validated in code, not argparse, via a stored parser reference),
  updated help text. `cmd_train()` split into `_cmd_train_image()` (new) and `_cmd_train_tabular()`
  (M121's original body, relocated). New import: `train_image_classifier`.
- `tests/test_cli_train_image.py` -- new, 17 tests.
- `tests/test_cli_train_image_cuda.py` -- new, 1 test.
- `tests/test_packaging_smoke.py` -- 2 new tests appended.
- `docs/development/cli.md` -- **Model training** section extended with the image-classification
  contract, dispatch table, rejected-flag list and JSON payload; CUDA-requirements and Limitations
  sections updated.
- `README.md` -- CLI command list and a short **Model training** example block added (also fixed an
  M121 documentation gap: `forge model train` had no README mention at all before this milestone).
- `docs/development/m122-unified-model-training-cli.md` -- this report.
- `docs/development/progress.md` -- M122 entry appended.

No file under `forge/training/`, `forge/data/`, or `forge/serialization/` changed -- every training,
preprocessing, and artifact-format behavior this milestone touches was already correct and unmodified.

## 21. Performance observations

No new computation is introduced -- `_cmd_train_image()` is argument translation plus one existing
call. The real-petimages CLI validation (section 12) trained 128 images / 2 epochs / CPU in
well under a minute, consistent with `train_image_classifier()`'s own already-measured performance
(unchanged by this milestone).

## 22. Limitations

- **`--image-size`, `--on-error`, `--val-fraction`, `model=`, `classes=`, `verbose=`** are not
  exposed for `--task image-classification`, mirroring M121's identical scope decision for the
  tabular tasks -- a workflow needing one of them calls `forge.train_image_classifier()` directly.
- **`verbose` is forced to `False`** for `--task image-classification` regardless of
  `train_image_classifier()`'s own `verbose=True` default (see **Image classification integration**
  above) -- the one place this milestone's CLI forwarding rule has a documented exception, for
  `--json` correctness.
- **No automatic task detection.** `--task` is always required and authoritative; a directory given
  under `--task classification` is refused, never silently retried as `--task image-classification`.
- Segmentation and sequence training remain Python-only (see below).

## 23. Deferred training families

Segmentation and sequence-model training have no CLI training door, and none was added here --
neither has the kind of single, stable, "one directory/file in, one artifact out" contract this
command's three existing tasks share (`docs/development/maintenance.md`'s maintenance boundary lists
`forge.train_image_classifier()`/`forge.train_tabular_classifier()`/`forge.train_tabular_regressor()`
as the supported high-level training surface; nothing about segmentation or sequence training is
listed as mature enough for this kind of thin CLI wrapper today). Per this milestone's own guiding
principle, further training-CLI expansion should wait for a concrete, mature capability and a real
workflow need, not be added for symmetry alone.

## 24. Full-suite result

```text
python -m pytest tests/ -q
Total:   3,904
Passed:  3,904
Failed:  0
Skipped: 0
(1067.79s / 0:17:47, CUDA present)
```

Run with every M122 change in place (`forge/cli/model.py`, `tests/test_cli_train_image.py`,
`tests/test_cli_train_image_cuda.py`, `tests/test_packaging_smoke.py`'s two additions), in a single
pass. 3,884 at the M121 baseline + 20 new (17 CPU CLI + 1 CUDA CLI + 2 installed-wheel) = 3,904.

## 25. Final disposition

Decision: IMPLEMENTED

Summary:
- `forge model train` now supports image classification (`--task image-classification`) alongside
  M121's unchanged tabular classification/regression, through the existing
  `forge.train_image_classifier()` -- no second training implementation.
- CLI routing is two explicit, thin functions (`_cmd_train_image()`/`_cmd_train_tabular()`); no
  generic trainer/dispatcher abstraction was introduced.
- Every invalid task/input combination the brief listed is rejected before training, with a message
  naming the task, the input, and the incompatible argument.

Tests:
- baseline: 3,884 passed, 0 failed, 0 skipped (M121)
- final: 3,904 passed, 0 failed, 0 skipped (single full-suite run, all changes in place)
- failures: 0
- skipped: 0

CUDA:
- Hardware-verified on the reference GeForce 940MX: `--device cuda` trains on CUDA (saved artifact
  metadata confirms), and the resulting artifact is consumed correctly on CPU.

Real datasets:
- Real petimages (`C:\Zeus\model_training\image_classifier\petimages`) trained/inspected/predicted/
  evaluated entirely through the installed CLI, end to end, all steps exit 0.
- M121's real Pima/StatLib-housing tabular validation is unaffected (code path unchanged).

Known limitations:
- `--image-size`/`--on-error`/`--val-fraction`/`model=`/`classes=`/`verbose=` are not exposed for
  image-classification training, matching M121's tabular scope decision.
- No CLI training door for segmentation or sequence workloads (deferred, not mature enough yet).
