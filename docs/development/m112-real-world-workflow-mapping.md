# Milestone 112 — Real-World ML Workflow Mapping and 1.x API Scope

Audit/design/research milestone. **No framework code, tests, examples, models, or
public APIs changed.** Everything below was derived from the repository at HEAD
`6cf1b37` (2026-09-19). Evidence is tagged:

| Tag | Meaning |
|---|---|
| **[S]** | Source-supported fact -- stated by an external page I actually retrieved (section 3.2 lists them, including what I could *not* retrieve). |
| **[I]** | My interpretation of sources or of the field. Not a fact. |
| **[F]** | Forge-specific conclusion, drawn from the repository, not from the outside world. |
| **[probed]** | Observed by running a throwaway script at HEAD (scripts lived in the session scratchpad; nothing under `forge/`, `tests/`, `examples/`, or `models/` was touched, and they are not committed). |

Validation: the four reference test files (`test_image_classifier.py`,
`test_reusable_predictor.py`, `test_tabular_classification_artifact_prediction.py`,
`test_regression_artifact_workflow.py`) were re-run at HEAD: **70 passed**. The full
suite was deliberately not run -- nothing executable changed.

M111 (`m111-high-level-api-audit.md`, ACCEPTED) is the baseline this builds on. It
asked "which *examples* deserve APIs?". M112 asks "which *problems* does Forge
make easy?" -- and therefore re-checks M111's sequence instead of adopting it.

---

## 1. Executive summary

**What M112 found.**

1. **The field's common workflows are few, and Forge already covers the core of
   them.** Across Keras, fastai, Hugging Face, scikit-learn, and Vertex AI
   documentation **[S]**, the recurring problem families are: classification and
   regression over *images*, *tabular rows*, and *text*; plus time-series
   forecasting, segmentation, object detection, and generation. **[I]** The
   intersection that a small from-scratch framework can serve with an end-to-end
   workflow is much smaller than that union.
2. **Forge can serve exactly three of those as complete workflows today** --
   image classification (has a one-call API), tabular classification, tabular
   regression (both fully working, no one-call API). Everything else is either a
   lower-level capability (1D signals, text generation, autoencoders) or has a
   real infrastructure gap (text classification, segmentation on real data,
   detection, generation, multi-label).
3. **Evaluation is the one cross-cutting gap**, not a per-workflow one. Forge has
   no way to score a saved artifact on new labeled data with its own persisted
   preprocessing, and its metric vocabulary is only accuracy/MSE/MAE
   **[probed]** -- so even a correct evaluation of an imbalanced classifier
   (Pima: 62.3% holdout majority) is uninformative without a confusion matrix.
4. **The recommended 1.x high-level surface is four items**: the existing
   `train_image_classifier()`, plus `train_tabular_classifier()`,
   `train_tabular_regressor()`, and `ArtifactPredictor.evaluate()`. Everything
   else is a lower-level example or is explicitly deferred/excluded.

**What changed relative to M111** (this is why M112 was worth doing before
implementing):

| # | M111 said | M112 finding | Consequence |
|---|---|---|---|
| R1 | Implement tabular first (M112), evaluate last (M113) | Evaluate defines the metric vocabulary (`accuracy`/`mse`/`mae`) and the "baseline is computed on the evaluated data" rule that the tabular result types should then copy; it also works today on artifacts users *already hold*, and it is the measurement every tabular acceptance criterion needs. | **Swap the order:** evaluate is M113, tabular is M114. Same two milestones, re-scoped. |
| R2 | Evaluation reports `accuracy` (or `mse`/`mae`) and a baseline | Only accuracy/MSE/MAE exist in `forge.training` **[probed]**. On an imbalanced target, accuracy alone hides the failure mode. | `evaluate()` also returns a confusion matrix and per-class precision/recall, computed in NumPy inside `evaluate()` -- **not** new `Metric` classes. |
| R3 | Tabular APIs split with `val_fraction` only | A random split is *wrong* for time-ordered/windowed data: overlapping windows leak across a random split. Forecasting by direct windowed regression works through the existing regression artifact **[probed]**, so this is the only thing standing between the tabular regressor and a valid forecasting recipe. It is also what M111's own "do not train on the holdout rows" requirement needs. | Add `validation_data=(X_val, y_val)`, mutually exclusive with `val_fraction`. No windowing helper, no forecasting API. |
| R4 | F4: the waveform artifact "needs a new task type" to use the unified predictor | A `Conv1d` classifier saved with `task="tabular_classification"` loads and predicts through `load_predictor()` today **[probed]**; `InputSchema` is simply `None` for a non-`Linear`-first model. The example's problem is that it saves `task="classification"` (the *image* workflow). | **No new task type.** 1D signals stay class C; the label is a naming wart, not a capability gap. |
| R5 | Regression's unscaled target might block the API (6.6 open risk) | On a Concrete-strength-shaped synthetic target (mean 35, std 17), an unscaled-target MLP reached 7% of the predict-the-mean MSE at 60 epochs and 4% at 150 **[probed, synthetic]**. | Risk substantially retired; still confirmed on real data in M114. The split-out fallback stays in the acceptance criteria but is now unlikely to trigger. |
| R6 | Real regression dataset: "e.g. UCI Auto MPG (unverified)" | Auto MPG has missing `horsepower` values and a categorical `car_name` **[S]** -- it would run straight into the NaN gap (F3). **UCI Concrete Compressive Strength** is 1,030 rows, 8 numeric features, *no missing values*, CC BY 4.0 **[S]**. | Recommend Concrete (subject to M114 acquiring and re-verifying it). |
| R7 | M111 acceptance (i): beat the "65.1% majority baseline" on the holdout | 65.1% is the full-dataset majority share (500/768). The *holdout's* majority is **62.3%**, and the existing example artifact scores **72.1%** on it **[probed]** via `evaluate.py`. | Correct the criterion: compare against the majority of the *evaluated split*. |

**Recommendation.** ACCEPTED. Two implementation milestones, in this order:
M113 `ArtifactPredictor.evaluate()`, then M114 the tabular training family plus the
shared workflow conventions and two committed reference artifacts. Nothing else is
promoted (section 9).

**An honest caveat about tabular.** **[S]** Grinsztajn, Oyallon & Varoquaux
(arXiv 2207.08815) report that tree-based models remain state-of-the-art on
medium-sized (~10K sample) tabular data across 45 datasets. **[F]** Forge has no
tree models and should not claim tabular MLPs are the best tool for tabular data.
The tabular APIs are justified by *workflow completeness in a deep-learning
framework* (UC1/UC2 in `docs/product/use-cases.md`; HF and fastai both list
tabular tasks **[S]**), not by accuracy. The API docs and the model README must say
so, and `evaluate()` reporting a baseline is part of keeping that honest.

---

## 2. Forge's current capability map

Inspected at HEAD: `forge/`, `examples/`, `models/`, `tests/`, `docs/`, `README.md`,
`pyproject.toml`, the CLI, and the committed artifacts.

### 2.1 Building blocks (what actually exists)

| Layer | Present | Verified absent |
|---|---|---|
| Layers (`forge.nn`) | `Linear`, `Conv1d`, `Conv2d`, `MaxPool1d`, `MaxPool2d`, `UpsampleNearest2d`, `BatchNorm2d`, `RNNCell`, `LSTMCell`, `Embedding`, `Dropout`, `Flatten`, `ReLU`, `Tanh`, `Sequential` | GRU, LayerNorm, BatchNorm1d, average/global pooling, attention, `nn.Sigmoid`/`nn.Softmax` modules (a search of `forge/` for these names returned nothing) |
| Losses | `MSELoss`, `CrossEntropyLoss` | BCE / any multi-label loss |
| Optimizers | `SGD`, `Adam` | schedulers |
| Metrics (`forge.training`) | `Accuracy`, `MeanSquaredError` (`"mse"`), `MeanAbsoluteError` (`"mae"`) | precision, recall, F1, confusion matrix, IoU (`PixelAccuracy`/`IoU` exist only inside `examples/segmentation/`) |
| Data | `Dataset`, `TensorDataset`, `Subset`, `random_split`, `sequential_split`, `ImageFolder` (`on_error=`), `DataLoader`, `CUDAPrefetchLoader` | text dataset, paired image/mask dataset, windowing helper, CSV/DataFrame reader (M92: deliberate) |
| Transforms | `Compose`, `ToTensor`, `Normalize`, `ReplaceValue`, `Reshape`, `Flatten`, `Resize` (persistable); `Lambda` (**not** persistable **[probed]**) | tokenizer/vocab transform, NaN imputation |
| Training | `Trainer`, `train()`, `train_and_save()`, `start_training_session()`, `EarlyStopping`, `train_image_classifier()` | a stepwise-recurrence trainer (RNN examples use hand-written loops) |
| Artifacts | `.forge` with architecture, weights, `preprocessing`, `classes`, `task` (one of 5 `TASK_TYPES`), `InputSchema` | target/output post-processing, thresholds, tokenizers |
| Inference | `predict()`, five `predict_*_artifact()` functions, `predict_model()`, `load_predictor()`/`ArtifactPredictor`, `forge model predict` | evaluation of a saved artifact |

`forge.__all__` has 54 names.

### 2.2 End-to-end chains per workload (what is actually complete)

Columns are the stages of the desired end state (`data -> train -> evaluate ->
artifact -> fresh-process inference`).

| Workload | One-call train | Evaluate a *saved* artifact | Artifact `task` | Fresh-process inference | Real data validated |
|---|---|---|---|---|---|
| Image classification (folder) | **Yes** (`train_image_classifier`) | No | `classification` | Yes (external wheel consumer, M107) | Yes (~25k cat/dog; external, not in repo) |
| Tabular classification | No (~400 lines by hand) | Hand-composed (`evaluate.py`) | `tabular_classification` | Yes (`infer.py`, `app.py`) | Yes (Pima CSV, committed) |
| Tabular regression | No | `Trainer.evaluate()` only (silently skips preprocessing) | `regression` | Yes (`predict_tensor_artifact`) | **No** (synthetic only) |
| Segmentation | No | example-local metrics | `segmentation` | Yes | No (synthetic) |
| 1D signal classification | No | `Trainer.evaluate()` | saved as `classification` (mislabelled, R4) | **No** as saved; **Yes** if saved as `tabular_classification` **[probed]** | No (synthetic) |
| Sequence generation | No | sample text | `sequence` | Yes | No (toy corpora) |
| Autoencoder / long-range recall | No | reconstruction MSE / recall accuracy | none | No | MNIST only / synthetic |

### 2.3 Committed artifacts and their state

| Path | Size | Task | Recorded device | Test coverage |
|---|---|---|---|---|
| `models/image_classifier/image_model.forge` | 1.19 MB | `classification` (`cat`, `dog`) | `cuda` | **None** -- no test loads it (grep of `tests/`: only `docs/` and `README.md` mention it) |
| `examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge` | 5.3 KB | `tabular_classification`, `InputSchema(feature_count=8)` | `cpu` | via the example's tests |

---

## 3. Real-world workflow taxonomy

### 3.1 Problems, not architectures

Model families are **implementations**; the taxonomy below is by **user problem**.

| Model family | Problems it can serve in Forge |
|---|---|
| CNN (`Conv2d`) | image classification; segmentation; (regression on images -- unexercised) |
| `Conv1d` | 1D signal classification; fixed-window time-series work |
| MLP (`Linear`/`ReLU`) | tabular classification; tabular regression; **direct** windowed forecasting; bag-of-features text; reconstruction/anomaly scoring |
| `RNNCell`/`LSTMCell` | generation; (sequence classification and stepwise forecasting need a hand-written loop) |
| `Embedding` | text/token models; categorical ids |
| Autoencoder (`Conv2d`+`UpsampleNearest2d`, or `Linear`) | reconstruction; anomaly scoring; representation learning |

The rule for the rest of this document: an API is named for the *problem* and its
*input kind* (`train_tabular_regressor`), never for the architecture
(`train_mlp`) and never because an example exists.

### 3.2 External evidence (what was and was not established)

Retrieved 2026-09-19:

| # | Source | What it establishes **[S]** |
|---|---|---|
| S1 | Keras data-loading API (`keras.io/api/data_loading/`) | A major framework's data layer treats four input kinds as first-class: `image_dataset_from_directory`, `text_dataset_from_directory`, `audio_dataset_from_directory`, `timeseries_dataset_from_array` (plus `pad_sequences`). |
| S2 | fastai docs (`docs.fast.ai`) | Lists application areas: vision (classification, segmentation, GANs), text (classification), **tabular**, **collaborative filtering**, and medical vision/text -- with a "much the same" pattern across them. |
| S3 | Hugging Face Tasks (`huggingface.co/tasks`) | Names Tabular Classification and Tabular Regression as their own category; also Text Classification, Text Generation, Image Classification, Image Segmentation, Object Detection, Unconditional Image Generation, Audio Classification. **Time-series forecasting and anomaly detection do not appear on that page.** |
| S4 | scikit-learn estimator map | Organizes the problem space as classification, regression, clustering, dimensionality reduction. |
| S5 | Google Vertex AI AutoML documentation | Tabular classification/regression/forecasting; image classification and object detection; text classification. **Caveat:** obtained through a search summary; the page itself returned 404/redirect on direct fetch, and the model-evaluation page's metric list could not be retrieved. Treat as weak support. |
| S6 | Grinsztajn, Oyallon, Varoquaux, arXiv 2207.08815 | Tree-based models remain state-of-the-art on medium-sized (~10K) tabular data across 45 datasets. |
| S7 | UCI ML Repository, datasets 165 and 9 | Concrete: 1,030 rows, 8 numeric features, no missing values, CC BY 4.0, donated 2007-08-02. Auto MPG: 398 rows, `horsepower` has missing values, `car_name` categorical, CC BY 4.0. |

**Not established.** I looked for an authoritative ranking of workflow prevalence
(e.g. a survey giving the share of practitioners by data type). What surfaced was
a secondary blog paraphrase ("50-90% use tabular data") with no verifiable primary
figure, so **no prevalence number is used or claimed anywhere in this report.**

**[I]** What the sources support is *co-occurrence*, not ranking: image, tabular,
and text classification/regression appear in essentially every taxonomy; forecasting
appears in Keras (S1) and Vertex (S5) but not HF (S3); detection and generation
appear where the framework/platform has pretrained models and large compute (S2, S3,
S5). "Common" is therefore established for the core set only, and is *never* by
itself a reason to build (section 5).

### 3.3 Families considered

Image classification, tabular classification, tabular regression, time-series
forecasting, time-series/1D-signal classification, image segmentation, text
classification, text generation, anomaly detection, representation learning /
autoencoding, object detection, image generation, multi-label classification, audio
classification, recommender/collaborative filtering, and clustering/dimensionality
reduction (scikit-learn territory, not a deep-learning workflow). Sequence
classification over variable-length inputs is treated separately from fixed-window
signals because the two differ in Forge's ability to train them.

---

## 4. Workflow-to-Forge mapping

Readiness: **Strong** (complete chain, real data) / **Mostly ready** (complete chain,
missing one convenience or real-data proof) / **Partial** (works with material
manual work or a semantic hole) / **Infrastructure gap** (needs a new primitive,
transform, input kind, or metric) / **Outside** (needs new architecture or is not a
deep-learning workflow).

| Real-world workflow | Typical user problem | Input | Output | Forge capability | Existing example | Probe / evidence | Readiness |
|---|---|---|---|---|---|---|---|
| Image classification (folder) | "Sort my photos into classes" | folder-per-class images | class + confidence | `train_image_classifier` -> `classification` artifact -> `load_predictor` | `image_folder_classification`; `models/image_classifier` | reference tests pass; real 25k run (M107) | **Strong** (no saved-artifact evaluate) |
| Tabular classification | "Predict a category from columns" | `(n,f)` numbers + labels | class + confidence | MLP + `Normalize`/`ReplaceValue` -> `tabular_classification` artifact | `tabular_diabetes` (real), `tabular_classification` | holdout 72.1% vs 62.3% **[probed]** | **Mostly ready** (no one-call API, no evaluate) |
| Tabular regression | "Predict a number from columns" | `(n,f)` numbers + targets | float(s) | MLP + `Normalize` -> `regression` artifact | `regression` (synthetic) | unscaled target converges **[probed, synthetic]**; `y` of shape `(n,)` raises `LossError` against a `(n,1)` output **[probed]** | **Mostly ready** (no real dataset, no one-call API) |
| Time-series forecasting (direct, fixed window) | "Predict the next k values" | 1D/2D series -> `(n,window)` | `(n,1)` or `(n,k)` | windowed MLP regression; artifact round-trips **[probed]** | none | val MSE 0.023 vs mean-baseline 0.85; `(n,4)` multi-step output works **[probed]** | **Partial** (must window by hand; random split leaks, R3; no recursive rollout at inference; no walk-forward evaluation) |
| 1D signal / time-series classification | "Classify a waveform/sensor trace" | `(n,1,L)` | class | `Conv1d` CNN via `Trainer` | `waveform_classification` (synthetic) | loads via `tabular_classification` **[probed]**; the example's `classification` artifact does **not** (`PersistenceError`, F4) | **Partial** (works; mislabelled; no real data) |
| Image segmentation | "Mask the region of interest" | image + mask pairs | per-pixel mask | encoder/decoder CNN; `segmentation` artifact; `predict_image_artifact` | `segmentation` (synthetic) | `MSELoss` on a thresholded mask; metrics are example-local | **Infrastructure gap** (no paired dataset abstraction, no in-library seg metrics, no real dataset chosen) |
| Text classification | "Label this review/ticket" | strings | class | `Embedding` + `Flatten` + `Linear` trains on fixed-length ids **[probed]** | none | predictor rejects a `str` **[probed]**; `Lambda` tokenizer not persistable **[probed]**; no tokenizer/vocab transform; no variable-length pooling; no text dataset | **Infrastructure gap** |
| Text / sequence generation | "Generate text in this style" | token seed | tokens | `RNNCell`/`Embedding`, hand-written loop, `sequence` artifact | `char_rnn`, `word_rnn` | inference already unified | **Partial** (toy corpora; `Trainer` cannot express recurrence) |
| Anomaly detection | "Flag unusual rows" | `(n,f)`, usually unlabeled | score / flag | tabular autoencoder (`target=input`) as a `regression` artifact **[probed]**: normal recon-MSE 0.45 vs anomalous 10.4 | none | threshold is *not* persisted; no evaluation semantics without labeled anomalies | **Partial** (expressible; no artifact concept for a threshold) |
| Representation learning / autoencoding | "Compress / embed my data" | images or rows | latent / reconstruction | conv autoencoder | `autoencoder` (MNIST) | needs example-local `ConvAutoencoder` registered before load (F4) | **Partial** (example only) |
| Multi-label classification | "Tag with several labels" | any | label set | none | none | no BCE loss, no `nn.Sigmoid` **[probed]** | **Infrastructure gap** |
| Object detection | "Find and box objects" | images + boxes | boxes + labels | none | none | no box format, NMS, anchor/IoU losses, mAP | **Outside** |
| Image generation | "Synthesize images" | noise/text | images | none (autoencoder is not generative) | none | needs GAN/diffusion machinery and heavy compute | **Outside** |
| Audio classification | "Classify sounds" | audio files | class | none | none | Forge decodes images (Pillow), not audio | **Outside** |
| Recommender / collaborative filtering | "Suggest items" | id interactions | rankings | `Embedding` only | none | no sparse-id dataset, no ranking metrics | **Outside** |
| Clustering / dimensionality reduction | "Group / project my data" | rows | labels / coords | none | none | scikit-learn's domain (S4) | **Outside** (not deep learning) |

**Can Forge currently support the workflow as a coherent end-to-end chain?**
Yes for image classification (except saved-artifact evaluation), tabular
classification, tabular regression. Direct forecasting and 1D-signal classification
work *technically*, but only through undocumented recipes. Everything else: no.

---

## 5. API promotion classification

### 5.1 Ten-criterion matrix

Y met, P partly, N not met. Criteria: **1** real-world relevance, **2** Forge
identity, **3** existing capability, **4** end-to-end coherence (data -> preprocess
-> train -> evaluate -> artifact -> fresh-process inference, no second framework),
**5** user simplicity, **6** natural data model, **7** sensible evaluation
semantics, **8** persistable in the existing artifact, **9** validatable on a real
dataset with reasonable infrastructure, **10** scope cost acceptable.

| Workflow | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Image classification (folder) | Y | Y | Y | Y | Y | Y | P | Y | Y | Y | done |
| **Tabular classification** | Y | Y | Y | Y | Y | Y | P | Y | Y | Y | **A** |
| **Tabular regression** | Y | Y | Y | Y | Y | Y | P | Y | P | Y | **A** |
| **Saved-artifact evaluation** | Y | Y | Y | Y | Y | P | Y | Y | Y | Y | **A** |
| Time-series forecasting | Y | Y | P | P | P | P | N | P | P | P | **B** |
| Image segmentation | Y | Y | P | P | N | N | P | Y | N | P | **B** |
| Text classification | Y | Y | P | N | P | P | Y | N | P | N | **B** |
| 1D signal classification | P | Y | Y | P | P | P | Y | P | N | Y | **C** |
| Text/sequence generation | P | Y | P | N | N | Y | N | Y | N | N | **C** |
| Anomaly detection | P | P | P | N | P | Y | N | N | P | N | **D** |
| Autoencoding / representation | N | P | P | N | N | Y | N | N | N | N | **D** |
| Multi-label classification | P | Y | N | N | P | Y | Y | P | P | N | **D** |
| Object detection | Y | P | N | N | N | N | N | N | N | N | **D** |
| Image generation | Y | N | N | N | N | Y | N | N | N | N | **D** |
| Audio / recsys / clustering | P | N | N | N | N | N | N | N | N | N | **D** |

(criterion 1 for the `P` rows: relevant in some sources, absent from the ones I
could verify -- for example, HF's task page omits anomaly detection and forecasting
**[S]**; I make no prevalence claim, see 3.2.)

### 5.2 Classification with reasoning

**A -- first-class 1.x API candidates**

- **Tabular classification.** The plumbing a developer rewrites (split; fit
  `Normalize` on train only; keep it in sync with persisted `preprocessing=`;
  hand-pick a batched `sample`; hand-size an MLP; compute the baseline) is exactly
  what `train_image_classifier()` removed for images. Already hardware- and
  real-data-validated, artifact-complete, and the only workflow with a committed
  real dataset and holdout.
- **Tabular regression.** Same body as classification with a different loss/metric.
  Criterion 9 is `P` only because the real dataset is not yet in hand; the
  unscaled-target risk is substantially retired **[probed]**.
- **Saved-artifact evaluation.** Justified by a *demonstrated silent wrong number*
  (M97: 37.0% vs 62.3% baseline, no error), not by convenience. It is the only A
  item that is a cross-cutting capability rather than a workflow.

**B -- important, defer** (each with a *promotion trigger*, per `maintenance.md`
section 7 -- none is scheduled):

- **Time-series forecasting.** Common **[S: S1, S5]** and technically reachable as
  windowed regression **[probed]**. Deferred because a real forecasting API needs
  design Forge does not have: an ordered-split contract, a windowing/`timeseries_
  dataset_from_array`-like helper (S1 ships one), recursive multi-step rollout at
  inference, and walk-forward evaluation. The direct-forecast subset is served now
  by `train_tabular_regressor(validation_data=...)` plus a documented recipe.
  *Trigger:* a real forecasting workload where the recipe measurably fails, or
  repeated user windowing boilerplate.
- **Image segmentation.** Common **[S: S2, S3]**; `segmentation` task and
  `predict_image_artifact` exist. Deferred because there is no paired image/mask
  dataset abstraction, IoU/pixel-accuracy live only in an example, and no real
  dataset has been chosen. *Trigger:* a real paired dataset workload.
- **Text classification.** Very common **[S: S1, S2, S3, S5]**, Forge fit is real
  (`Embedding`). Deferred because it needs *three* new primitives at once -- a
  persistable tokenizer/vocab transform, string input at `ArtifactPredictor.predict`,
  and variable-length handling (no pooling layer exists) -- plus a text dataset
  and a real corpus. *Trigger:* a concrete text workload; the design must start
  from the persistence question (a tokenizer that does not travel with the artifact
  breaks the "fresh process" guarantee).

**C -- supported capability; keep as lower-level example**

- **1D signal classification.** Trains through `Trainer`; artifact works if saved
  as `tabular_classification` **[probed]**. A `train_signal_classifier()` would be a
  third near-copy with no real dataset behind it.
- **Text/sequence generation.** Toy corpora, hand-written loops; a workflow API
  would need a second training engine (a stated non-goal). Inference is already
  unified.
- Also C: MNIST/ResNet (a dataset format, not a workflow), checkpoint/resume (already
  first-class at the right, lower, level), teaching scripts.

**D -- outside current 1.x scope**

- **Anomaly detection.** Expressible at the low level **[probed]** but a *first-class*
  workflow needs a persisted threshold (new artifact metadata) and evaluation
  semantics that only exist with labeled anomalies. **[I]** It is widely needed in
  practice, but I found no source in this audit that ranks it, and no Forge workload
  demands it. D here means "no 1.x API and no promise", not "impossible".
- **Autoencoding/representation learning, multi-label, object detection, image
  generation, audio, recommenders, clustering:** see section 10.

---

## 6. Recommended 1.x high-level API surface

Only what section 5 justifies. Signatures are design proposals for M113/M114 to
finalize; **none is implemented in M112.**

### 6.0 The surface as a whole

```python
# train (one function per problem x input kind)
forge.train_image_classifier(data_dir, *, path, ...)       # exists (M107)
forge.train_tabular_classifier(X, y, *, path, ...)         # M114
forge.train_tabular_regressor(X, y, *, path, ...)          # M114

# use
predictor = forge.load_predictor(path)                     # exists
predictor.predict(x)                                       # exists
predictor.evaluate(...)                                    # M113
```

Net new top-level names: **4** (two functions, two result types); `evaluate()` is a
method, and its result type is exported from `forge.training` only if a caller needs
to annotate it.

### 6.1 `forge.train_tabular_classifier(X, y, *, path, ...)`

| Item | Design |
|---|---|
| Purpose | Numeric feature table + labels -> verified, self-describing classification artifact in one call. |
| Natural input | `X`: `(n,f)` NumPy array / nested list / `Tensor`. `y`: `(n,)` ints or strings. No DataFrame parameter, no CSV reader (M92/M111). |
| Target/output | Class label + confidence at inference. |
| Important configuration | `path` (required), `epochs=60`, `batch_size=32`, `learning_rate=1e-3`, `val_fraction=0.2` **or** `validation_data=(X_val, y_val)`, `model=`, `transform=`, `classes=`, `early_stopping=`, `device=`, `seed=0`, `verbose=True`. |
| Reasonable defaults | Pima-validated MLP (32/16) and hyperparameters; re-validated on the synthetic four-class example, not invented. |
| Result | Frozen `TabularClassifierResult`: shared fields (6.7) + `classes`, `n_features`, `baseline_accuracy` (majority of the validation split). |
| Artifact | `train_and_save(..., task="tabular_classification", classes=..., preprocessing=<one fitted Compose>)`; statistics fit on the training split only, after any user `transform=`; `InputSchema` via the existing M101 mechanism. |
| Inference | Unchanged: `load_predictor`, `predict_model`, `forge model predict`. |
| Evaluation | `predictor.evaluate(X_new, y_new)` (M113). |

### 6.2 `forge.train_tabular_regressor(X, y, *, path, ...)`

Same as 6.1 except: `y` may be `(n,)`, `(n,1)`, or `(n,k)` -- normalized internally
(a raw `(n,)` target against a `(n,1)` model raises `LossError` today
**[probed]**, so the API must reshape); the default model's output width is
`y.shape[1]`; `MSELoss`; metrics `mse`/`mae`; `task="regression"`; no `classes`;
result `TabularRegressorResult` with `baseline_mse` (predict-the-train-mean,
evaluated on the validation split) instead of `baseline_accuracy`.
**Targets are not rescaled** (R5; synthetic probe supports this; M114 confirms on
Concrete). `(n,k)` is accepted because it costs nothing (the multi-step artifact
round-trips **[probed]**) and is what makes the direct-forecast recipe work; it is
**not** a forecasting API.

### 6.3 `ArtifactPredictor.evaluate(...)`

| Item | Design |
|---|---|
| Purpose | Score a saved artifact on new labeled data with the artifact's own persisted preprocessing -- closing the M97 silent-wrong-number trap. |
| Input | Follows `predictor.task`, exactly as `predict()` already does (see 6.4). |
| Result | Frozen dataclass: `task`, `samples`, `metrics` (`accuracy` \| `mse`,`mae`), `baseline`, and for classification `classes`, `confusion_matrix`, `per_class` (`precision`, `recall`, `support`). Metric keys deliberately match the training `val_metrics` keys. No `loss` (not persisted). |
| Artifact | Read-only; never modifies the file. |
| Not proposed | `forge.evaluate()`, `forge.evaluate_artifact()`, `forge model evaluate` -- the method on the already-loaded predictor is the smallest surface. |

### 6.4 Evaluation as a first-class workflow (what "evaluate a saved model" means)

`evaluate` = **artifact + held-out labeled data + persisted preprocessing + task-
specific metrics**, with the persisted preprocessing applied by the library, never
by the caller. Per task:

| Task | Input | Label handling | Metrics | Baseline (computed on the *evaluated* data, not read from the artifact) | Verdict |
|---|---|---|---|---|---|
| `classification` (image) | folder-per-class path | class **names** matched against `predictor.classes`; an unknown class is a `DataError` naming both vocabularies; never by discovery order | accuracy, confusion matrix, per-class precision/recall | majority-class share | supported |
| `tabular_classification` | `(X, y)` | `y` interpreted exactly as `train_tabular_classifier` interprets it; strings must be members of `classes`; integer indices must be in range | same | majority-class share | supported |
| `regression` | `(X, y)` with `y` `(n,)`/`(n,1)`/`(n,k)` | -- | `mse`, `mae` (+ `r2` as the normalized form of the baseline comparison) | predict-the-evaluated-mean MSE | supported |
| `segmentation` | image + mask pairs | -- | IoU / pixel accuracy | -- | **not supported**: raises `DataError` (blocked on B item) |
| `sequence` | -- | -- | -- | -- | **not supported** (would need a loss/perplexity semantics the artifact does not persist) |

Why a baseline computed on the evaluated data: the artifact does not record its
training distribution, and the existing oracle (`evaluate.py`) already defines it
this way (62.3% = the holdout's own majority **[probed]**). It is a *floor to beat*,
not a claim about the training set, and the docstring must say so.

Why a method, not a function: M111's reasoning stands -- the predictor already
holds the model, preprocessing, classes, and `InputSchema`, so `evaluate` adds one
verb to an object that has all its nouns. A CLI `forge model evaluate` is not
justified until a shell/CI workflow asks for it (a tabular CLI would need a CSV
contract Forge deliberately does not own).

---

## 7. Shared API conventions

These apply to `train_image_classifier()` (retrofitted additively) and every future
workflow trainer. Items marked **(new)** differ from or add to M111.

| Dimension | Convention |
|---|---|
| **Naming** | `train_<input-kind>_<role>` -> `train_image_classifier`, `train_tabular_classifier`, `train_tabular_regressor`. Named for the problem and the input contract, never the architecture. No `task=`/`mode=`/`workflow=` switch. The artifact `task` strings are a separate, existing vocabulary and are not renamed (R4). Results are `<InputKind><Role>Result`. |
| **Input forms** | Directory workflows: a path with the exact `ImageFolder` contract. Numeric workflows: NumPy array / nested list / `Tensor`, converted once. No DataFrame parameter, no CSV reader, no new input abstraction. Validate up front, with domain errors: length mismatch, non-2-D `X`, ragged rows, **non-finite values** (F3 reproduced **[probed]**: silent `nan` loss), fewer than 2 classes, too few samples for the split. |
| **Validation data (new)** | `val_fraction` (random `random_split`) **or** `validation_data=(X_val, y_val)`; passing both is an error. The random split is invalid for ordered/windowed data (R3), so the explicit form is required, not optional. No held-out *test* split in training -- that is `evaluate()`. |
| **Model selection** | A validated default architecture per workflow; `model=` accepts any `forge.nn.Module`. Before epoch 1, run one forward pass on a real sample batch and check the output against the task contract (`len(classes)` logits / `y.shape[1]` outputs); mismatch -> `DataError` naming expected vs actual (F2). |
| **Preprocessing** | One `Compose`, built once, used as both the dataset `transform=` and the persisted `preprocessing=`; statistics fit on the training split only; reuse `Normalize`/`Resize`/`ReplaceValue`; an optional user `transform=` runs *before* the built-in standardization, which is fit on its output. **No NaN imputation** (reject and let the user clean). No target rescaling. No second preprocessing system. |
| **Training configuration** | Exposed: `epochs`, `batch_size`, `learning_rate`, `val_fraction`/`validation_data`, `seed`, `device`, `model`, `early_stopping`, `verbose`. Per-workflow defaults come from that workflow's *validated* example. Not exposed: optimizer, loss, metrics, scheduler, shuffle, prefetch, `drop_last`, checkpoint/resume, `atol`, hyperparameter search. |
| **Device** | `None` -> the model's current device; `"cpu"`/`"cuda"` explicit. **No `"auto"`** (contradicts Forge's never-fallback policy and can turn into an OOM on a 2 GB card). |
| **Results** | Frozen dataclasses, shared field names asserted by one cross-family test: `history`, `train_loss`, `train_metrics`, `val_loss`, `val_metrics`, `model`, `artifact_path`, `dataset_size`, `train_size`, `val_size`, `stopped_early`, `best_epoch`, `best_monitored_value`; workflow-specific fields only where meaningful (`classes`, `skipped_images`, `n_features`, a baseline). No base class. Metric keys are the existing `accuracy`/`mse`/`mae` everywhere, including `evaluate()`. |
| **Persistence** | Always train + save + verify via `train_and_save()`; `path` required, `str | os.PathLike`. Artifacts are self-describing (task, classes, preprocessing). No new format. **Preflight before epoch 1** (F1): output directory exists and is writable; `preprocessing` (including user `transform=`) is serializable -- so a typo or a `Lambda` costs seconds, not a full run. |
| **Evaluation** | `predictor.evaluate()` only (6.3/6.4). Never `Trainer.evaluate()` for a saved artifact. |
| **Errors** | Domain-level `DataError`/`TrainerError`/`PersistenceError` naming the argument and the fix; never a raw `ShapeMismatchError`, `LossError`, `KeyError`, or `AttributeError` from a path a high-level API can pre-check. No exception-hierarchy redesign. |
| **CLI** | Python is the primary surface. No `forge train ...`; `forge model predict` already covers all five tasks. |

### Where the pattern stops

The family is defined by three tests, all of which must hold before a fourth trainer
is proposed: (1) the natural data form is a path or an array Forge already
understands, with no new dataset/input abstraction; (2) the result is one of the
five existing artifact tasks with an existing inference path; (3) a real dataset
validates it end to end. A workflow that fails any test stays an example or waits
for its trigger. This is why the family is **three trainers**, not one per example
(`train_segmenter`, `train_autoencoder`, `train_char_rnn`, `train_lstm`,
`train_resnet`, `train_waveform_classifier` are all explicitly rejected).

---

## 8. Model artifact plan

`models/` holds *finished, useful, representative* artifacts, not every example
model. Datasets stay external.

| Path | Workflow | Verdict | Why | Size | Custom imports? | Consumer |
|---|---|---|---|---|---|---|
| `models/image_classifier/` | image classification | **Keep**; Patch F6: `predict.py` accepts `--device`, default `cpu` (the artifact records `cuda` **[probed]**, so a clone without CUDA currently fails); **add a test** that loads it in a fresh subprocess on CPU (none exists) | Real 25k-image dataset; externally consumed | 1.19 MB | No | `predict.py` |
| `models/tabular_classifier/` | tabular classification | **Add in M114** -- Pima, trained through `train_tabular_classifier()` **on the rows not in `diabetes_holdout_eval.csv`**, saved from a CPU run | Only real tabular dataset already in the repo; proves the API and the fresh-process path; tiny | ~5 KB | No | `predict.py`: JSON row in -> label + confidence out |
| `models/tabular_regressor/` | tabular regression | **Add in M114** -- UCI Concrete Compressive Strength (CC BY 4.0 **[S]**), CPU run; attribution in the consumer's docstring | The only real regression demonstration; clean numeric data | ~12 KB (existing regression example artifact is 12 KB) | No | `predict.py`: JSON row in -> MPa out |
| segmentation | -- | **Do not add** | synthetic-only; blocked on the B item | -- | -- | -- |
| waveform | -- | **Do not add** | synthetic; consumable only via the `tabular_classification` label (R4) | -- | -- | -- |
| autoencoder | -- | **Do not add** | needs `ConvAutoencoder` registered (custom import) | -- | Yes | -- |
| char/word RNN | -- | **Do not add** | toy corpora | -- | -- | -- |
| MNIST / ResNet | -- | **Do not add** | reproducible from the example; adds nothing over the bundled image model | -- | -- | -- |
| text / forecasting | -- | **Do not add** | no supported workflow | -- | -- | -- |

Rules for any committed artifact: produced by a supported public workflow; loadable
through the public API with **no custom imports**; paired with a consumer that uses
only `forge.load_predictor()`; saved from a **CPU** run so it loads anywhere
(CUDA correctness of the training path is verified in the test suite on the
940MX, per `CLAUDE.md`, not by the artifact's provenance); under 2 MB; covered by a
CPU test that loads it in a fresh subprocess and predicts one row; never a dataset
inside.

End state: exactly three directories under `models/`.

---

## 9. Proposed post-M111 milestone sequence

Smallest coherent sequence: **two milestones**, reordered from M111 (R1) and
re-scoped by R2-R7. Neither is started here. Housekeeping stays as Patches.

### M113 — Saved-artifact evaluation: `ArtifactPredictor.evaluate()`

- **Purpose:** first-class, preprocessing-correct evaluation of any saved artifact;
  fixes the metric vocabulary and baseline semantics the tabular results will copy.
- **Workflows:** image classification (folder), tabular classification, regression.
- **Public APIs:** `ArtifactPredictor.evaluate()` + one frozen result type.
- **Framework changes:** one method over the existing predict core; confusion matrix
  and per-class precision/recall computed in NumPy inside it; reuses
  `Accuracy`/`MeanSquaredError`/`MeanAbsoluteError` for the headline numbers. **No
  new `Metric` classes, no `Trainer` change.** Unsupported tasks raise `DataError`.
- **Real dataset/workload:** Pima holdout CSV (`evaluate.py` is the numeric oracle);
  the bundled cat/dog model on held-out `petimages` images (external, manual/skip-
  if-absent like the M110 smoke test); an existing synthetic regression artifact.
- **Artifacts:** none new.
- **Tests:** exact parity with `evaluate.py` (72.1% / 62.3% on the 154-row holdout);
  a regression test that raw sentinel rows fed to `Trainer.evaluate()` reproduce the
  M97 trap while `evaluate()` does not; confusion matrix against a hand-computed
  4-row case; class-name mismatch and unsupported-task errors; regression metrics
  against hand-computed values; string vs int label handling; artifact file bytes
  unchanged after `evaluate()`.
- **Fresh-process validation:** `load_predictor` + `evaluate` in a subprocess that
  imports nothing from `examples/`.
- **CPU/CUDA:** CPU tests unconditional; CUDA-loaded predictor test skips cleanly and
  is hardware-verified on the 940MX.
- **Acceptance criteria:** metric parity with `evaluate.py`; persisted preprocessing
  applied on every supported task; per-class output present and correct; unsupported
  tasks fail with a domain error; no existing signature changed; no existing test
  modified.
- **Why now:** a demonstrated silent-wrong-number hazard on artifacts users already
  hold; it is the measurement M114's acceptance criteria are stated in.

### M114 — Tabular training family and shared workflow conventions

- **Purpose:** close the tabular training gap with two functions that share one
  private core, and fix the conventions (F1, F2, F5, R3) before the family grows.
- **Workflows:** tabular classification, tabular regression.
- **Public APIs:** `train_tabular_classifier`, `train_tabular_regressor`,
  `TabularClassifierResult`, `TabularRegressorResult`; `forge.__all__` grows by
  exactly four names.
- **Framework changes:** new `forge/training/tabular.py` (composition only); shared
  private helpers for preflight (F1), model-contract check (F2), non-finite/shape
  input validation (F3), and `validation_data=`. Additive retrofits to
  `train_image_classifier()`: preflight, model-contract check, `early_stopping=`, the
  three missing result fields (F5), `path: str | os.PathLike`. No change to
  `Trainer`, serialization, or the artifact format.
- **Real dataset/workload:** (1) Pima, training on the rows **not** in
  `diabetes_holdout_eval.csv`; (2) UCI Concrete Compressive Strength, acquired
  externally and re-verified (M111 did this for Pima; Concrete's stated facts in
  S7 are unverified against the file itself until then). The regression run must
  *not* be chosen or tuned to dodge target scale.
- **Artifacts:** `models/tabular_classifier/`, `models/tabular_regressor/`
  (section 8) plus the fresh-subprocess load test; `models/image_classifier/`
  device patch (F6).
- **Tests:** CPU-only unless marked; input validation (length mismatch, 1-D `X`,
  NaN/Inf, ragged, single class, tiny dataset, `val_fraction`, `val_fraction` +
  `validation_data` together, `y` of shape `(n,)`/`(n,1)`/`(n,k)`); determinism by
  `seed`; artifact round-trip in a fresh subprocess; persisted preprocessing equals
  training preprocessing and its statistics come from the training split only;
  preflight failures occur *before* the first epoch (asserted by a model that
  records whether it was called); `model=` contract errors; early stopping and best-
  model restoration; a cross-family test that every result type carries the shared
  fields; an ordered-split test showing `validation_data=` keeps a windowed series
  leak-free; existing image-classifier tests unchanged.
- **Fresh-process validation:** each committed artifact loaded and predicted in a
  subprocess by its own `predict.py`.
- **CPU/CUDA:** CPU is the default and what the committed artifacts use; CUDA tests
  skip cleanly and are hardware-verified on the 940MX.
- **Acceptance criteria:** (i) on the Pima holdout, `evaluate()` (M113) reports
  accuracy above the **holdout's** majority baseline (62.3%) and within 3 points of
  the existing example's 72.1%; (ii) the fresh-process consumer reproduces the
  in-process prediction on the same row; (iii) the regression artifact beats
  predict-the-mean MSE on held-out rows, **or** the milestone documents that target
  scaling blocked it and splits regression out (no new persistence machinery added
  to force it through); (iv) every new error path in section 7 has a test; (v) the
  full suite passes with no previously passing test changed (baseline 2,765 at M110,
  plus new tests); (vi) `README.md`, `examples/README.md`, `training-engine.md`,
  `progress.md` updated for the changed surface only, with the "MLPs are a baseline
  on tabular data" caveat (S6) in the API docstring.

### Patches (not milestones)

Each is independently documentation/consumer-only:

- `models/image_classifier/predict.py` device handling (F6) -- may fold into M114.
- `.gitignore` contradictory `models/image_classifier/...` entries; `maintenance.md`
  section 3's dangling `sandbox/` references (F7).
- `examples/waveform_classification` README: state that its artifact is consumed by
  saving with `task="tabular_classification"` (R4), or fix the example's `task=`
  in a separate example-only change. **No new task type.**

### Trigger-based backlog (deliberately unscheduled)

| Item | Class | Promote when | Prerequisites named by this audit |
|---|---|---|---|
| Forecasting | B | a real forecasting workload shows the recipe failing, or repeated windowing boilerplate | window helper, ordered-split contract, recursive rollout, walk-forward evaluation |
| Segmentation | B | a real paired image/mask dataset workload exists | paired dataset abstraction, IoU/pixel-accuracy in `forge.training`, `evaluate()` extension |
| Text classification | B | a concrete text workload exists | persistable tokenizer/vocab transform, string input in `predict`, sequence pooling, text dataset, real corpus |

Per `maintenance.md` section 7, no milestone number is reserved for any of these.

---

## 10. Explicit non-goals

Rules of thumb: a non-goal is either already out of scope in `docs/product/scope.md`
(cloud ML infrastructure, distributed training, large-model training, enterprise
orchestration), or fails the three "where the pattern stops" tests in section 7.

| Item | Verdict | Reasoning |
|---|---|---|
| AutoML | **Out** | Model/architecture search is a second product; `train_*` already exposes `epochs`/`learning_rate`/`model=` for a caller to loop over. Vertex AutoML **[S: S5]** is a hosted-platform feature. |
| Hyperparameter optimization | **Out** | Same. The fixed, validated defaults are a feature, not a limitation to search around. |
| Object detection | **Out** (D) | Needs box formats, NMS, anchor/IoU losses, mAP: multiple new subsystems and a heavy-compute dataset. Common **[S]** but impractical on a 2 GB card. |
| Image generation | **Out** (D) | GAN/diffusion machinery and compute; the autoencoder is not generative. |
| Large-scale NLP | **Out** | `scope.md`: no large-model training. No attention layers exist. |
| Distributed training / multi-GPU orchestration | **Out** | `scope.md`; one reference GPU. |
| Model serving | **Out** | `scope.md` excludes API wrappers/serving; `load_predictor()` is the in-process ceiling. |
| Cloud training | **Out** | `scope.md`; `CLAUDE.md`: no cloud services or paid dependencies. |
| Model registry / marketplace | **Out** | `models/` is a repo directory of three artifacts, not a registry. |
| Plugin ecosystem | **Out** | `register_module`/`register_transform` already exist for custom classes; an ecosystem is unjustified for a single-maintainer project. |
| Pretrained model zoo / transfer learning | **Out** | **[I]** fastai/HF workflows lean heavily on pretrained weights **[S: S2, S3]**; Forge has none, and importing others' formats is already unsupported (`README`). This is a real limit on from-scratch image accuracy; it is documented, not something to paper over with an API. |
| Multi-label, audio, recommenders, clustering | **Out** | Each fails at least tests (1) and (3) (new loss/metric or new data kind; no workload). |
| `forge train ...` CLI; DataFrame/CSV abstractions; a `train()` mega-function taking `task=` | **Out** | Rejected in M107/M111/M92; nothing in this audit reverses them. |
| A second training engine (stepwise/RNN `Trainer`) | **Out** | Stated non-goal; keeps generation/sequence classification as examples. |

---

## 11. Definition of done

The high-level API expansion phase is complete when **all** of the following hold:

1. `ArtifactPredictor.evaluate` (M113) exists and is documented; it supports exactly
   `classification`, `tabular_classification`, `regression`, and raises a domain
   `DataError` for the rest.
2. `forge.train_tabular_classifier` and `forge.train_tabular_regressor` (or
   regression explicitly split out per M114 criterion iii) exist, are exported, and
   are documented in `README.md` and `training-engine.md`.
3. `forge.__all__` grows by exactly four names, no new `predict_*` function is
   added, no existing public signature changes meaning (additive keywords only), and
   no existing test is modified.
4. All three trainers satisfy the section 7 conventions, *enforced by tests*: shared
   result fields, preflight-before-epoch-1, `model=` contract check, `early_stopping=`,
   non-finite input rejection, `validation_data=` exclusive with `val_fraction`.
5. Every promoted workflow has passed its real workload end to end: dataset ->
   real training -> artifact -> **fresh-process** inference via a `models/`
   consumer -> held-out `evaluate()`. CPU is tested in CI; CUDA is hardware-verified
   on the reference GPU.
6. `models/` holds exactly `image_classifier/`, `tabular_classifier/`, and
   `tabular_regressor/` (if not split out): each under 2 MB, CPU-loadable with no
   custom imports, no dataset inside, and each loaded by a CPU test that runs in a
   fresh subprocess.
7. The full suite passes (`python -m pytest tests/ -q`), with the new baseline
   recorded in `maintenance.md`, and the F6/F7 Patches are closed.
8. Every workload in section 4 that is not promoted is still runnable exactly as it
   is today, no unpromoted workflow gained an API, and the three B items remain
   unscheduled until their triggers are met.

---

## Appendix A — Probe log

All at HEAD `6cf1b37`, CPU, throwaway scripts outside the repo. "GAP" means the
existing public API could not express the step; it is a finding, not a defect
fixed here.

| # | Probe | Result |
|---|---|---|
| P1 | Direct 1-step forecasting: windows -> `train_and_save(task="regression")` -> `load_predictor().predict(raw windows)` | **PASS.** Val MSE 0.023 vs predict-mean 0.851; preprocessing applied from the artifact. |
| P2 | Direct multi-step forecasting, `y` shaped `(n,4)` | **PASS.** Output `(2,4)`. |
| P3 | Regression with 1-D `y` `(n,)` against `Linear(...,1)` | **GAP.** `LossError: MSELoss requires prediction and target to have the same shape, got prediction shape (32, 1) and target shape (32,)`. |
| P4 | `Conv1d` classifier saved as `task="tabular_classification"` -> `load_predictor` -> predict `(N,1,L)` | **PASS.** `InputSchema=None`; labels returned. (R4) |
| P5 | Same model saved as `task="classification"` (what the example does) | **GAP.** `PersistenceError ... no automatic way to prepare the input image` (F4 reconfirmed). |
| P6 | `Embedding`->`Flatten`->`Linear` on fixed-length token ids, train + save | **PASS** (trains and persists). |
| P7 | Text artifact given a raw string at `predict()` | **GAP.** `DataError: ... requires input_data to be a Tensor, NumPy array, or list/tuple of numbers, got str`. |
| P8 | `Lambda` tokenizer as `preprocessing=` | **GAP.** `PersistenceError: Cannot serialize ... 'forge.data.transforms.Lambda': it is not registered for persistence`. |
| P9 | Tabular autoencoder (`target=input`) as a `regression` artifact | **PASS** as a capability: normal recon-MSE 0.45 vs anomalous 10.4. The threshold is not persisted by any artifact field. |
| P10 | Classification metrics beyond accuracy | **GAP.** Only `Accuracy`, `MeanSquaredError`, `MeanAbsoluteError` exist in `forge.training`. |
| P11 | Multi-label building blocks | **GAP.** No BCE loss and no `nn.Sigmoid` (only `Tensor.sigmoid()`). |
| P12 | NaN in a feature column through `forge.train()` | **GAP (silent).** `final_train_loss = nan`, no error (F3 reconfirmed). |
| P13 | Unscaled-target MLP, target mean 35 / std 17, 800 train rows, Adam 1e-3 | val MSE 21.8 (60 ep, 7% of baseline 306.7) and 12.5 (150 ep, 4%). A second "absorb mean/std into the last layer" variant in the same script was a **no-op** (numbers identical to default) and is discarded, not cited. Synthetic data; confirms the risk is small, not that it is zero. |
| P14 | `python -m examples.tabular_diabetes.evaluate` on the committed artifact and holdout | 154 rows: accuracy **72.1%**, holdout-majority baseline **62.3%**, +9.7 points (the M111 "65.1%" is the full-dataset majority share, R7). |
| P15 | `inspect_model()` on both committed artifacts | Pima: `tabular_classification`, `InputSchema(feature_count=8)`, device `cpu`. Bundled image model: `classification`, `['cat','dog']`, device **`cuda`** (F6). |
| P16 | Four reference test files at HEAD | 70 passed. |

M111's findings F1, F2, F5, F6, F7 were **carried forward, not re-probed** (F3 was
re-probed as P12; F4 as P4/P5). Probes P1-P13 use synthetic data and prove
*expressibility*, not accuracy.

---

**M112 Decision:**
**ACCEPTED**

Smallest recommended post-M111 implementation sequence:

1. **M113** -- `ArtifactPredictor.evaluate()` (classification folder, tabular
   classification, regression; confusion matrix + per-class output; unsupported
   tasks raise `DataError`).
2. **M114** -- `train_tabular_classifier()` + `train_tabular_regressor()` (with
   `validation_data=`, shared preflight/model-contract/non-finite validation, additive
   `train_image_classifier()` retrofits) + `models/tabular_classifier/` and
   `models/tabular_regressor/` (Pima on non-holdout rows; UCI Concrete).

Forecasting, segmentation, and text classification are deferred (B) with named
promotion triggers and **no** milestone number. Nothing here is implemented yet.
