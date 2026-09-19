# Milestone 111 — High-Level API Surface Audit and 1.x Workflow Design

Audit/design milestone. **No framework code, tests, or public APIs changed.**
Everything below was derived from the repository at HEAD `9f1158e`
(2026-09-19), not from milestone history; where a claim rests on something
run rather than read, it is marked **[probed]** (throwaway scripts outside the
repo; nothing under `forge/` or `tests/` was touched). Focused reference tests
were re-run at HEAD before writing this: `test_image_classifier.py`,
`test_reusable_predictor.py`, `test_tabular_classification_artifact_prediction.py`,
`test_regression_artifact_workflow.py` -- 70 passed.

## 1. Executive summary

Forge's convenience surface is lopsided. **Consuming** an artifact is already
unified across all five task types (`load_predictor()`, `predict_model()`,
`forge model predict`). **Producing** one in a single call exists for exactly
one workflow: `train_image_classifier()` (folder of images -> artifact).

The gap that remains is not "more examples need APIs". It is narrower:

1. **Tabular classification and tabular regression** -- the two most common
   non-image workflows -- already have complete, hardware-verified,
   real-data-validated end-to-end paths (M83, M91, M92, M100-M102; Pima
   diabetes CSV committed in the repo). But each still costs a developer
   ~180-290 lines of `train.py` plus ~150-215 lines of `dataset.py`, and the
   plumbing is the same plumbing M107 removed for images: a fitted `Normalize`
   built by hand and kept in sync with the persisted `preprocessing=`, a
   hand-computed split, a hand-picked batched `sample`, a hand-sized MLP, a
   hand-computed baseline.
2. **Evaluating a saved artifact on new labeled data** has no first-class path.
   `Trainer.evaluate()` silently skips persisted preprocessing; M97 measured
   the consequence (37.0% accuracy vs a 62.3% majority baseline, no error).
   `examples/tabular_diabetes/evaluate.py` hand-composes four calls to avoid it.
3. **The existing high-level API itself has fail-late and unvalidated-escape-
   hatch defects** (section 1.1) that any new API family would copy unless the
   conventions are fixed first.

**Recommendation: three additions, in two milestones, and nothing else.**
`train_tabular_classifier()`, `train_tabular_regressor()` (one milestone -- they
share ~90% of their body), and `ArtifactPredictor.evaluate()` (one milestone).
Every other workload in `examples/` stays a lower-level example. Net new
top-level names: 4 (two functions, two result types); `ArtifactPredictor.evaluate`
is a method.

### 1.1 Findings from the inspection that shape the plan

| # | Finding | Evidence |
|---|---|---|
| F1 | **Save-time failures surface after training finishes.** A nonexistent output directory, or a non-persistable `preprocessing=` (e.g. `Lambda`), raises `PersistenceError` only after every epoch has run; the trained model is discarded. On the reference real workload (5 epochs, ~25k images, 940MX) that is tens of minutes lost to a typo in `path=`. | **[probed]** `train_image_classifier(..., path="<missing dir>/m.forge", epochs=2, verbose=True)` printed both epochs, then raised. Same for `Lambda` through `train_and_save`. |
| F2 | **The `model=` escape hatch is unvalidated against the task.** A 5-logit model given for a 2-class dataset trains and passes `save_and_verify()` (which only proves save/reload agreement), then every `predict()` raises `TrainerError: output has 5 class score(s) but 2 class label(s)`. A wrong input width surfaces as a raw `ShapeMismatchError` from `Linear`. | **[probed]** |
| F3 | **NaN features train silently to NaN loss** (`final_train_loss = nan`, no error), and `ReplaceValue` cannot express NaN-as-missing (`ReplaceValue(sentinel=nan)` is a no-op, since `nan != nan`). | **[probed]** |
| F4 | **The 1D-waveform example's artifact cannot use the unified inference API.** It is saved with `task="classification"` (image workflow) but has no preprocessing/classes; `load_predictor()` raises `PersistenceError ... no automatic way to prepare the input image`. The autoencoder artifact needs example-local `ConvAutoencoder` registered before it loads and has no task type. | **[probed]** |
| F5 | `ImageClassifierResult` omits `stopped_early`/`best_epoch`/`best_monitored_value`, which `TrainAndSaveResult` carries, and `train_image_classifier()` does not expose `early_stopping=` -- though early stopping is a supported, example-validated feature (tabular_diabetes). | Code read: `image_classifier.py:186-224`, `api.py:326-356` |
| F6 | The bundled `models/image_classifier/image_model.forge` (1.19 MB) records `device: cuda`, so `predict.py` (which calls `load_predictor(MODEL_PATH)` with no device) needs a working CUDA setup. The artifact itself loads and predicts fine on CPU with `device="cpu"`. | **[probed]** `inspect_model()`; CPU load + prediction succeeded |
| F7 | Housekeeping: `.gitignore` lists both `models/image_classifier/...` files under "ignore local testing" although both are tracked (contradictory; harmless today). `maintenance.md` section 3 names `sandbox/petimages_smoke.py` and `sandbox/t1_cat_dog/train_high_level.py`, but `sandbox/` is empty and was never tracked (`git log --all -- sandbox` is empty); the dataset and training script now live at `C:\Zeus\model_training\image_classifier\`, outside the repo. The documented real-world smoke test is not reproducible from a clone. | `git ls-files`, `git log --all`, directory listing |

Valid behavior confirmed **[probed]**: unknown `device=` gives a clear
`UnsupportedDeviceError` listing supported types; `epochs=0`, `batch_size=0`,
`learning_rate<0`, `image_size=64` (non-tuple), `val_fraction` out of range,
`<2 classes`, missing `data_dir`, and corrupt image files (`on_error="skip"`)
all give domain-level errors; `path=` accepts a `pathlib.Path` despite the
`str` annotation; inference errors (missing file, non-image file, array instead
of a path, wrong tabular feature count, unbatched 1-D tabular row) are all
clear `DataError`s.

## 2. Current high-level APIs

| Surface | Names | Covers |
|---|---|---|
| Train (general) | `train()`, `train_and_save()`, `save_and_verify()`, `EarlyStopping` | Any `Module` on any `Dataset`/`DataLoader`; caller supplies loss/optimizer/`sample`/preprocessing/classes/task |
| Train (workflow) | `train_image_classifier()` -> `ImageClassifierResult` | Folder-per-class images -> verified classification artifact. **The only workflow-level trainer.** |
| Train (advanced) | `Trainer`, `start_training_session()`, `save_checkpoint()`/`load_checkpoint()` | Resume with exact `DataLoader` shuffle equivalence; hand-written loops |
| Results | `TrainingResult`, `TrainAndSaveResult`, `ImageClassifierResult` | History + final metrics + model + `artifact_path` |
| Predict (one-shot) | `predict_model()`; `predict_artifact()`, `predict_tensor_artifact()`, `predict_image_artifact()`, `predict_sequence_artifact()`, `predict_tabular_classification_artifact()` | One function per artifact shape, plus a dispatcher over `task` metadata |
| Predict (reusable) | `load_predictor()` -> `ArtifactPredictor` | Load once, predict many; all five tasks |
| Introspect | `inspect_model()` -> `ModelInfo` (task, classes, preprocessing, `InputSchema`) | No model reconstruction, no CUDA needed |
| Low-level inference | `predict()`, `interpret_classification()`, `generate_sequence()` | Live-model inference |
| CLI | `forge model inspect|convert|predict`, `forge checkpoint inspect|convert`, `forge benchmark` | Post-training only; **no training command** (deliberately rejected in M107) |

`forge.__all__` has 54 names. The predict side is already feature-complete;
this audit proposes no change to it beyond `evaluate()`.

## 3. Workflow inventory

Derived from `examples/`, `forge/`, `models/`, `tests/`, and the CLI.
"HL API" = an existing workflow-level convenience API.

| Workflow | Task (`TASK_TYPES`) | Example(s) | Data | Model | Train path | Eval path | Persist | Inference | HL API | Audience | Stable? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Image classification (folder) | `classification` | `image_folder_classification` (synthetic PNGs); real `petimages` (external, ~25k); `models/image_classifier` | Directory-per-class files via `ImageFolder` | 3-block CNN | `train_image_classifier()` -> `train_and_save()` | val split in-call only | `.forge` + `Resize/Normalize` + classes | `load_predictor()`, `predict_model()`, CLI | **Yes** | Ordinary | Yes (M107, real-validated, externally consumed) |
| Image classification (bundled IDX) | `classification` | `mnist`, `resnet` | Downloaded MNIST IDX | CNN / residual CNN | `train_and_save()` (fresh) / `Trainer` (resume) | `Trainer.evaluate()` | `.forge` | `predict_artifact()` | No | Advanced/benchmark | Yes -- but a benchmark, not a user workflow |
| Tabular classification | `tabular_classification` | `tabular_classification` (synthetic); `tabular_diabetes` (**real** Pima CSV, committed) | NumPy arrays from CSV / generated | MLP | `train_and_save()` + hand-built dataset/split/`Normalize` | `evaluate.py` (hand-composed) | `.forge` + `Compose([ReplaceValue?, Normalize])` + classes + `InputSchema` | `load_predictor()`, `app.py` | No | **Ordinary** | Yes (M91, M92, M100-102) |
| Tabular regression | `regression` | `regression` (synthetic only) | Generated arrays | MLP | `train_and_save()` + hand-built dataset | `Trainer.evaluate()` | `.forge` + `Normalize` + `InputSchema` | `predict_tensor_artifact()` | No | **Ordinary** | Path yes (M83); **no real dataset in repo** |
| Segmentation | `segmentation` | `segmentation` (synthetic shapes) | Generated (image, mask) pairs | Encoder/decoder CNN | `Trainer` + `save_and_verify()` | example-local `PixelAccuracy`/`IoU` | `.forge` + `Normalize` | `predict_image_artifact()` | No | Advanced | Path yes; no real data, no paired-data abstraction |
| 1D signal classification | `classification` (mis-fit, F4) | `waveform_classification` | Generated | `Conv1d` CNN | `Trainer` + `save_and_verify()` | `Trainer.evaluate()` | `.forge` (no classes/preprocessing) | **None usable** (F4) | No | Advanced | Training yes; artifact story incomplete |
| Autoencoder | none | `autoencoder` | MNIST | Example-local `ConvAutoencoder` | `Trainer` | reconstruction MSE | `.forge` (needs custom-module registration) | None | No | Research/demo | Example only |
| Sequence generation (RNN LM) | `sequence` | `char_rnn`, `word_rnn` | Generated corpora | `RNNCell` (+`Embedding`) | **hand-written loop** (`Trainer` cannot express stepwise recurrence) | sample text | `.forge` + vocab | `predict_sequence_artifact()` | No | Advanced | Path yes; toy corpora |
| Long-range recall | none | `long_range_recall` | Synthetic | `RNNCell` vs `LSTMCell` | hand-written loop | recall accuracy | model persistence | None | No | Diagnostic | Experiment |
| Checkpoint/resume | (cross-cutting) | `mnist`, `regression`, `image_folder_classification` | -- | -- | `start_training_session()` | -- | checkpoint | -- | (`train()` excludes it by design) | Advanced | Yes |
| Evaluate a saved artifact | (cross-cutting) | `tabular_diabetes/evaluate.py` | Held-out CSV | -- | -- | **hand-composed** `load_model`+`load_preprocessing`+`predict`+`Accuracy` | -- | -- | No | **Ordinary** | Pieces stable; no single path |
| Teaching scripts | -- | `trainer_demo.py`, `data_pipeline_demo.py`, `persistence_demo.py` | Tiny generated | `Linear`/MLP | `Trainer` | -- | -- | -- | -- | Learners | n/a |

## 4. API promotion analysis

### 4.1 What makes `train_image_classifier()` the right reference

- **Input:** one path. Classes, sample count, resolution handling, and corrupt
  files are discovered, not declared.
- **Configuration:** only knobs an ordinary user can reason about --
  `epochs`, `batch_size`, `learning_rate`, `val_fraction`, `image_size`,
  `device`, `seed`, `on_error`, `model`, `path`, `verbose`. No optimizer, loss,
  metric, scheduler, shuffle, prefetch, or `atol`.
- **Defaults:** a *previously validated* architecture and hyperparameters
  (`t1_cat_dog` on real data), not a new design. Loss/optimizer are fixed
  because image-folder classification has one obvious choice.
- **Composition:** `ImageFolder` -> one `Compose` built once and used as both
  `transform=` and persisted `preprocessing=` (eliminating build-twice drift
  by construction) -> `random_split` -> `DataLoader` -> model -> `train_and_save()`.
  No new loop, format, or dataset type.
- **Result:** developer-facing facts -- final metrics, classes, dataset/split
  sizes, skipped files, artifact path -- not internal objects.
- **Persistence:** always train + save + verify; `path=` is required. The
  artifact is self-describing (task, classes, preprocessing), so consumption
  needs no configuration.
- **Errors:** domain-level `DataError`s naming the problem and the escape hatch.
  Its gaps are F1, F2, F5.
- **Extensibility:** `model=` replaces the architecture; everything else
  remains reachable by dropping to `ImageFolder` + `train_and_save()`.

Principles extracted (used in section 6): *compose, never reimplement;
fit-and-persist preprocessing once; validated defaults over invented ones;
always yield a self-describing artifact; the escape hatch is a lower API, not
a bag of flags.*

### 4.2 Criteria matrix

Y = met, P = partly, N = not met. C1 commonness, C2 identity, C3 workflow
stability, C4 API simplicity, C5 existing infrastructure, C6 user benefit,
C7 repetition, C8 validation feasibility (real dataset -> artifact -> fresh
process).

| Candidate | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | Class |
|---|---|---|---|---|---|---|---|---|---|
| Image classification, folder (reference) | Y | Y | Y | Y | Y | Y | Y | Y | done |
| **Tabular classification** | Y | Y | Y | Y | Y | Y | Y | Y | **A** |
| **Tabular regression** | Y | Y | Y | Y | Y | Y | Y | P | **A** (conditional, see 8) |
| **Artifact evaluation** | Y | Y | P | P | Y | Y | P | Y | **A** (weakest A) |
| Segmentation | P | Y | P | N | P | P | P | N | **B** |
| Image classification, IDX (mnist/resnet) | P | P | Y | P | Y | N | N | Y | **C** |
| 1D signal classification | P | Y | P | P | P | P | P | N | **C** |
| Sequence generation (RNN LM) | P | Y | P | N | N | P | N | N | **C** |
| Checkpoint/resume | P | Y | Y | N | P | P | P | Y | **C** |
| Teaching scripts | -- | -- | -- | -- | -- | -- | -- | -- | **C** |
| Autoencoder | N | P | N | N | N | N | N | N | **D** |
| Long-range recall | N | N | N | N | N | N | N | N | **D** |

### 4.3 Per-workflow analysis

| Workflow | Current example | Existing low-level path | User value | API complexity | Class | Reasoning |
|---|---|---|---|---|---|---|
| Tabular classification | `tabular_diabetes` (real), `tabular_classification` | `TensorDataset` + hand `random_split` + hand-fit `Normalize` (+ optional `ReplaceValue`) + MLP + `train_and_save(task="tabular_classification", classes=, preprocessing=, sample=)` | **High.** `dataset.py`+`train.py` = 396 lines for Pima, of which the generic part (split, train-only statistics, std-zero guard, sample pick, baseline) is what every user rewrites. The M92 finding (fitted preprocessing silently lost unless persisted) is exactly the class of mistake an API prevents. | **Low.** Input is `(X, y)`; everything else is already a Forge object. Only dataset-specific piece (sentinel imputation) fits an optional `transform=`. | **A** | Meets all eight criteria; the only workflow with a *committed* real dataset, a holdout file, and a fresh-process consumer. Highest-value, lowest-risk promotion. |
| Tabular regression | `regression` | Same as above with `MSELoss`, MSE/MAE metrics, `task="regression"` | **High** (UC2 in `use-cases.md`). Same plumbing as classification. | **Low**, but shares one open risk: targets cannot be rescaled without new persistence machinery (section 6.6). | **A**, conditional | Identical shape to tabular classification, so it is cheaper to ship together than apart. C8 is only partial: the repo has no real regression dataset. Fallback if real-data validation forces target scaling: ship classification alone and split regression out. |
| Artifact evaluation | `tabular_diabetes/evaluate.py` | `load_model` + `load_preprocessing` + `predict` + `Accuracy` by hand; `Trainer.evaluate()` is a trap (no preprocessing hook) | **High correctness value:** a silent 37% vs 62% wrong number (M97). Every model eventually meets new labeled data. | **Medium.** Input differs per task (folder for images, arrays for numeric) -- the same per-task input rule `predict()` already has. Metrics only; the artifact does not record its loss, so no loss is reported. | **A** (weakest) | Justified by a demonstrated correctness trap, not by convenience; only one example currently hand-rolls it (hence C7 = P). Lands last and can be dropped without affecting the tabular APIs. |
| Segmentation | `segmentation` | `Trainer` + `save_and_verify(task="segmentation")` | Moderate | **High.** Needs a paired image/mask input (no `Dataset` for it; the example generates data in-process) and `PixelAccuracy`/`IoU`, which are example-local, not in `forge.training`. | **B** | A legitimate common task, but promoting it now would require a new data abstraction and new metrics with no real dataset to validate on. Promote when a real paired dataset workload exists. |
| Image classification, IDX | `mnist`, `resnet` | `train_and_save()` | Low -- MNIST is a benchmark, and the same task is served by `train_image_classifier()` on any folder | -- | **C** | A dataset format, not a workflow. Stays the flagship low-level example. |
| 1D signal classification | `waveform_classification` | `Trainer` + `save_and_verify` | Low-moderate | Medium; an input shape (`(N,1,L)`) the tabular API would not cover | **C** | Synthetic only, and its artifact has no working inference path (F4). Fixing F4 needs a new task type (a Minor, additive change) that no real workload currently justifies. |
| Sequence generation | `char_rnn`, `word_rnn` | hand-written loop | Low | **High.** `Trainer` cannot express stepwise recurrence; a workflow API would need a second training engine (a stated non-goal). | **C** | Toy corpora; educational; inference already unified. |
| Checkpoint/resume | three examples | `start_training_session()` | Moderate | High; `train()` deliberately excludes it because resume replaces the caller's model/optimizer | **C** | Already has a first-class API at the right (lower) level. |
| Autoencoder | `autoencoder` | `Trainer` | Low | High; example-local module means artifacts need custom imports (F4) | **D** | Specialized; no task type, no inference path, no user demand. |
| Long-range recall | `long_range_recall` | hand-written loop | None | -- | **D** | A diagnostic of a framework property (vanishing gradients), not a user workflow. |

## 5. Recommended 1.x API surface

Only three items are justified. Signatures are proposals for M112/M113 to
finalize; none are implemented here.

### 5.1 `forge.train_tabular_classifier(X, y, *, path, ...)`

- **Purpose:** numeric feature table + labels -> verified, self-describing
  classification artifact, in one call.
- **Typical invocation:**
  ```python
  result = forge.train_tabular_classifier(X, y, path="model.forge", epochs=60)
  result.val_metrics["accuracy"], result.baseline_accuracy
  ```
- **Important arguments:** `X` (`(n, f)` array-like), `y` (`(n,)` int or str
  labels), `path` (required), `epochs`, `batch_size`, `learning_rate`,
  `val_fraction`, `model`, `transform`, `classes`, `early_stopping`, `device`,
  `seed`, `verbose`. Defaults come from the values validated on Pima
  (`epochs=60`, `batch_size=32`, `learning_rate=1e-3`, MLP 32/16); the milestone
  re-validates them on the synthetic four-class example rather than inventing new ones.
- **Result (`TabularClassifierResult`, frozen dataclass):** `history`,
  `train_loss`/`train_metrics`/`val_loss`/`val_metrics`, `model` (the reloaded,
  verified one), `artifact_path`, `classes`, `n_features`,
  `dataset_size`/`train_size`/`val_size`, `baseline_accuracy` (majority class,
  measured on the validation split -- M92 showed accuracy is only meaningful
  against it), `stopped_early`/`best_epoch`/`best_monitored_value`.
- **Artifact behavior:** `train_and_save(..., task="tabular_classification",
  classes=..., preprocessing=<one fitted Compose>)`. `Normalize` statistics are
  fit on the training split only, after any user `transform=`. `InputSchema`
  (feature count) is derived by the existing M101 mechanism.
- **Inference behavior:** unchanged -- `load_predictor()`, `predict_model()`,
  `forge model predict`.

### 5.2 `forge.train_tabular_regressor(X, y, *, path, ...)`

Same as 5.1 with: `y` is `(n,)` or `(n, 1)` float; `MSELoss`; metrics
`MeanSquaredError` + `MeanAbsoluteError`; no `classes`; `task="regression"`;
result `TabularRegressorResult` with `baseline_mse` (predict-the-train-mean,
on the validation split) in place of `baseline_accuracy`. **Targets are not
rescaled** (see 6.6). Artifact/inference unchanged (`predict_tensor_artifact()`,
`load_predictor()`).

### 5.3 `ArtifactPredictor.evaluate(...)`

- **Purpose:** score a saved artifact on new labeled data, with the artifact's
  own persisted preprocessing applied -- closing the M97 silent-wrong-number trap.
- **Typical invocation:**
  ```python
  predictor = forge.load_predictor("model.forge")
  predictor.evaluate(X_new, y_new)        # tabular_classification / regression
  predictor.evaluate("holdout_folder/")   # classification (folder-per-class)
  ```
- **Important arguments:** input follows `predictor.task`, exactly as
  `predict()` already does. For folders, class *names* are matched against the
  artifact's `classes` (never by discovery order); an unknown class is a
  `DataError` naming both vocabularies. Other tasks raise `DataError`
  ("evaluate() is not supported for task ...").
- **Result:** a small frozen result: `metrics` (`accuracy`, or `mse`/`mae`),
  `samples`, and the same baseline field as training. No `loss` (not persisted).
- **Artifact behavior:** read-only; the artifact is not modified.
- **Not proposed:** `forge.evaluate()`/`forge.evaluate_artifact()` top-level
  functions, or `forge model evaluate` -- the method on the already-loaded
  predictor is the smallest surface.

## 6. Common API conventions

These apply to `train_image_classifier()` (retrofitted, additively) and to every
future workflow API.

### 6.1 Naming
`train_<input-kind>_<role>` -> `train_image_classifier`, `train_tabular_classifier`,
`train_tabular_regressor`. The input-kind qualifier is what keeps each function
honest about its input contract; M107 already rejected a dataset-agnostic
`train_classifier()` for looking more general than it is. The artifact `task=`
string is a separate, existing vocabulary and is not renamed
(`tabular_classification`, `regression`, ...). Results are
`<InputKind><Role>Result`. One function per workflow -- no `task=` switch
(M107 rejected that for `train()`). Tabular classification and regression stay
two public functions over one private core.

### 6.2 Inputs
- Directory workflows: a path with the exact `ImageFolder` contract.
- Numeric workflows: **NumPy arrays / nested lists / `Tensor`**, converted once.
- **No** DataFrame parameter (pandas is not a dependency; `df.to_numpy()` is
  one call), **no** CSV reader (M92 decided against one: `np.loadtxt` is one
  line), **no** new input abstraction. A caller holding a `Dataset` already has
  `train_and_save()`.
- Validate up front, with domain errors: `X`/`y` length mismatch, non-2-D `X`,
  ragged rows, **non-finite values (F3)**, fewer than 2 classes,
  `n_samples` too small for the split.

### 6.3 Device
`device=None` -> the model's current device (`cpu` for a fresh model);
`"cpu"`/`"cuda"` explicit. **No `"auto"`**: it contradicts Forge's stated
"never a fallback" policy (`forge model convert --device` is always explicit;
`CLAUDE.md`: CUDA must be real), and on a 2 GB card an implicit choice can
turn into an OOM. Unknown device strings already give a clear
`UnsupportedDeviceError` **[probed]**.

### 6.4 Training configuration
Shared surface: `epochs`, `batch_size`, `learning_rate`, `val_fraction`,
`seed`, `device`, `model`, `early_stopping`, `verbose`. Per-workflow defaults
are taken from that workflow's validated example (image `epochs=5`, tabular
`epochs=60`), not from a global default. **Not exposed:** optimizer, loss,
metrics, scheduler, `shuffle`, prefetch, `drop_last`, checkpoint/resume,
`atol`. `seed` governs the split, shuffling, and (only when the default model
is built) parameter initialization -- documented, as today. Validation is
always on (`0 < val_fraction < 1`, at least one sample per side); there is no
held-out *test* split -- that is what `evaluate()` is for.

### 6.5 Model selection
Validated default architecture; `model=` accepts any `forge.nn.Module`.
**New requirement (F2):** before the first epoch, run one forward pass on a
real sample batch and check the output against the task contract
(`len(classes)` logits, or 1 regression output); on mismatch raise a
`DataError` naming expected vs actual and stating the required input/output
shape. This also converts the raw `ShapeMismatchError` from a wrong input
width into a domain error.

### 6.6 Preprocessing
One `Compose`, built once, used as both the dataset `transform=` and the
persisted `preprocessing=`; statistics fit on the training split only
(the M92 leakage rule); reuse `Normalize`/`Resize`/`ReplaceValue`; no parallel
preprocessing system. An optional `transform=` is applied **before** the
built-in standardization, which is then fit on its output (the order
`tabular_diabetes` already proves). **Do not build NaN imputation** -- reject
non-finite input clearly and let the user clean it (`np.nan_to_num`, or
`ReplaceValue` for a non-NaN sentinel). **Regression target scaling is the one
place regression cannot mirror classification:** rescaling `y` needs a
persisted inverse transform, which is new artifact machinery (not an existing
mechanism). Convention: the API does not rescale targets and documents that;
M112's real regression dataset must show whether that is workable (section 8).

### 6.7 Results
Frozen dataclasses. Shared field names (a test asserts them on every result
type): `history`, `train_loss`, `train_metrics`, `val_loss`, `val_metrics`,
`model`, `artifact_path`, `dataset_size`, `train_size`, `val_size`,
`stopped_early`, `best_epoch`, `best_monitored_value`. Workflow-specific fields
only where meaningful: `classes` (classification), `skipped_images` (image
only), `n_features` (tabular), a baseline (`baseline_accuracy` /
`baseline_mse`). **No inheritance hierarchy** -- a naming convention plus a
test, not a base class.

### 6.8 Persistence
Always **train + save + verify** via `train_and_save()`; `path=` is required
(`train()` remains the train-only entry point); artifact is always
self-describing (task, classes where applicable, preprocessing). No new
format. `path` accepts `str | os.PathLike` (already works **[probed]**).
**New requirement (F1) -- preflight before epoch 1:** verify the output
directory exists and is writable, and that `preprocessing` (including any
user `transform=`) is serializable, *before* training starts. A late
`PersistenceError` after a long run is the worst failure mode a
"one-call" API can have.

### 6.9 Errors
Domain-level `DataError`/`TrainerError`/`PersistenceError` that name the
argument and the fix; never a raw `ShapeMismatchError`, `IndexError`,
`KeyError`, or `AttributeError` from a path a high-level API can pre-check.
No exception-hierarchy redesign.

### 6.10 CLI
Python is the primary surface. **No `forge train ...` commands** (M107's
reasoning stands: the CLI wraps trained-artifact operations only, and no real
workflow needs CLI-only training). `forge model predict` already covers all
five tasks; no `forge model evaluate` unless a real CI/shell workflow asks.

## 7. Model artifact plan

Datasets stay outside the repo. Committed artifacts must be small, produced by
a supported public workflow, loadable through the public API with no custom
imports, and paired with a consumer that uses only `forge.load_predictor()`.

| Path | Verdict | Why | Size | Custom imports? | Consumer |
|---|---|---|---|---|---|
| `models/image_classifier/` (exists) | **Keep**; fix F6 in `predict.py` (accept `--device`, default `cpu`; no binary change) | Real dataset, external consumer validated | 1.19 MB | No | `predict.py` |
| `models/tabular_classifier/` (M112) | **Add** -- Pima diabetes, trained through `train_tabular_classifier()` | Real committed dataset; proves the tabular API on real data; tiny | ~5 KB (existing example artifact is 5.3 KB) | No | `predict.py` (JSON row in, label + confidence out) |
| `models/tabular_regressor/` (M112) | **Add** -- real regression dataset chosen in M112 | Only real regression demonstration | ~12 KB (existing example artifact is 12 KB) | No | `predict.py` |
| Segmentation | **Do not add** | Synthetic-only; no real dataset | -- | No | -- |
| Waveform | **Do not add** | Artifact not consumable via public API (F4) | -- | No | -- |
| Autoencoder | **Do not add** | Needs `ConvAutoencoder` registered (custom import) | -- | **Yes** | -- |
| Char/word RNN | **Do not add** | Toy corpora; example artifacts remain generated locally | -- | No | -- |
| MNIST/ResNet | **Do not add** | Reproducible from the example; adds nothing over the bundled image model | -- | No | -- |

New artifacts should be saved from a **CPU** run so they load on any machine
(the bundled image model's CUDA provenance is F6); CUDA verification of the
training path stays in the test suite, per `CLAUDE.md`.

## 8. Proposed milestone sequence

Two milestones. Neither is started here.

### M112 — Tabular high-level training family

- **Purpose:** close the tabular training gap and fix the two conventions
  (F1, F2) that any new API would otherwise inherit.
- **Workflows:** tabular classification, tabular regression.
- **Framework changes:** new `forge/training/tabular.py` (composition only)
  with one private core; no changes to `Trainer`, serialization, or artifact
  format. Additive retrofits to `train_image_classifier()` only: (a) preflight
  of output path and preprocessing persistability (F1), (b) model-contract check
  on a sample batch (F2), (c) `early_stopping=` and the three result fields (F5),
  (d) `path: str | os.PathLike`. The preflight and the model-contract check are
  shared private helpers used by all three functions.
- **Public APIs:** `train_tabular_classifier`, `train_tabular_regressor`,
  `TabularClassifierResult`, `TabularRegressorResult`; four new `forge.*` exports.
- **Tests:** CPU-only unless marked; input validation (length mismatch, 1-D `X`,
  NaN/Inf, ragged, single class, tiny dataset, `val_fraction`); determinism by
  `seed`; artifact round-trips in a **fresh subprocess**; persisted
  preprocessing equals the training preprocessing and its statistics come from
  the training split only; `model=` contract errors; preflight failures happen
  *before* the first epoch (asserted by a model that records whether it was
  called); early stopping and best-model restoration; a cross-family test that
  every result type carries the shared fields; CUDA tests skip cleanly and
  are hardware-verified on the 940MX; existing image-classifier tests unchanged.
- **Real workload:** (1) Pima diabetes CSV, **training on the rows not in
  `diabetes_holdout_eval.csv`** (the holdout is the example's test split, so the
  API's own random split must not train on it); (2) one real, small, public
  regression dataset acquired externally, as M92 did for Pima (committed only
  if roughly <= 100 KB; a candidate such as UCI Auto MPG is *unverified* and the
  choice must not be made to dodge target scale).
- **Artifact/example:** `models/tabular_classifier/`, `models/tabular_regressor/`
  (section 7); no change to existing `examples/`.
- **Acceptance criteria:** (i) the new call on Pima beats the 65.1% majority
  baseline on the holdout and lands within 3 percentage points of
  `examples/tabular_diabetes`' reported accuracy; (ii) the fresh-process
  consumer reproduces the in-process prediction on the same row; (iii) the
  regression artifact beats predict-the-mean MSE on held-out rows, **or** the
  milestone documents that target scaling blocked it and splits regression out
  (no new persistence machinery is added to force it through); (iv) 100% of new
  error paths in 6.2/6.5/6.8 have a test; (v) full suite passes with no
  previously passing test changed (baseline 2,765 at M110, plus the new tests);
  (vi) `README.md`, `examples/README.md`, `training-engine.md`, `progress.md`
  updated for the changed surface only.
- **Why this milestone exists:** tabular classification/regression are the most
  common non-image workflows, already fully working; the plumbing is the same
  M107 plumbing; and the F1/F2 defects must be fixed before the API family
  grows or they get copied.

### M113 — Artifact evaluation: `ArtifactPredictor.evaluate()`

- **Purpose:** first-class, preprocessing-correct evaluation of a saved artifact.
- **Workflows:** classification (folder), tabular classification, regression.
- **Framework changes:** one method on `ArtifactPredictor` over the existing
  predict core and existing metrics (`Accuracy`, `MeanSquaredError`,
  `MeanAbsoluteError`); `Trainer.evaluate()` unchanged (its docstring already
  warns about the trap).
- **Public APIs:** `ArtifactPredictor.evaluate()` plus one small frozen result type.
- **Tests:** numeric parity with `examples/tabular_diabetes/evaluate.py` on the
  holdout CSV (the oracle); a regression test showing the M92 sentinel rows do
  **not** reproduce the 37% figure; class-name mismatch and unsupported-task
  errors; regression metrics against a hand-computed value; CUDA tests skip cleanly.
- **Real workload:** diabetes holdout CSV; the M112 regression artifact on
  held-out rows; the bundled cat/dog model on held-out `petimages` images (external
  dataset, manual/skip-if-absent, as the M110 smoke test is designed).
- **Artifact/example:** none new; `evaluate.py` is left untouched as the oracle.
- **Acceptance criteria:** exact metric parity with `evaluate.py`; evaluation
  applies persisted preprocessing on every task it supports; unsupported tasks
  fail with a domain `DataError`; no change to any existing signature.
- **Why this milestone exists:** the M97 silent-wrong-number trap is a
  correctness hazard, and after M112 the "train, save, predict" loop is
  complete except for "evaluate". It is separate from M112 because it is an
  inference-side capability and can be dropped without affecting the tabular APIs.

### Patch-level housekeeping surfaced by the audit (not milestones)

Independent, documentation/consumer-only, each its own Patch:
`models/image_classifier/predict.py` device handling (F6); the contradictory
`.gitignore` entries (F7); `maintenance.md` section 3's dangling `sandbox/`
references (F7 -- either restore a smoke script under version control or
correct the section).

## 9. Explicit non-goals

None of these has evidence in an existing Forge workload; none should be built
in this phase.

- AutoML, model/architecture search, hyperparameter optimization.
- Model registry, model zoo, plugin ecosystem.
- Cloud training, hosted inference, serving infrastructure.
- Distributed or multi-GPU training; AMP (no real workload needs it).
- A new serialization format, a second training engine, a second
  preprocessing system, a `Trainer` fork for stepwise/RNN training.
- Automatic device selection; NaN imputation; CSV/DataFrame input abstractions;
  target-scaling/output-postprocessing persistence (unless M112 proves it
  unavoidable, in which case stop and report per `CLAUDE.md`).
- High-level APIs for segmentation, waveform, autoencoder, RNN language
  modeling, long-range recall, or MNIST/ResNet.
- A `forge train ...` CLI or `forge model evaluate`.
- Consolidating or renaming the five `predict_*_artifact()` functions.

## 10. Definition of done

The high-level API expansion phase is complete when **all** hold:

1. `forge.train_tabular_classifier`, `forge.train_tabular_regressor` (or
   regression explicitly split out per M112 criterion iii), and
   `ArtifactPredictor.evaluate` exist, are exported, and are documented in
   `README.md` and `training-engine.md`.
2. `forge.__all__` grows by exactly four names (the two trainers and their two
   result types from sections 5.1/5.2) and no new predict function; no existing
   public signature changed (additive keywords only) and no existing test was
   modified.
3. Every workflow-level trainer (image + tabular) satisfies the section 6
   conventions, enforced by tests: shared result fields, preflight-before-epoch-1,
   `model=` contract validation, early stopping, non-finite input rejection.
4. Each promoted API has passed its real workload end to end: dataset -> real
   training -> artifact -> **fresh process** inference (`models/` consumer) ->
   held-out evaluation. The CPU path is tested in CI; the CUDA path is
   hardware-verified on the reference GPU.
5. `models/` holds exactly: `image_classifier/`, `tabular_classifier/`, and
   `tabular_regressor/` (if not split out) -- each < 2 MB, each loadable on
   CPU with no custom imports, none with datasets inside.
6. The full suite passes (`python -m pytest tests/ -q`) with the new baseline
   recorded in `maintenance.md`, and the F6/F7 housekeeping Patches are closed.
7. Every workflow in section 3 not promoted is still runnable as an example,
   unchanged, and no unpromoted workflow gained an API.

**M111 Decision: ACCEPTED**
