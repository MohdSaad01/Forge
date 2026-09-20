# M116 — Persisted regression target transforms

**Decision: IMPLEMENTED** — `train_tabular_regressor(..., target_transform="standardize")`, backed by a
persisted, artifact-level `forge.data.StandardizeTarget`.

Question this milestone had to answer: *when a regression model is trained on transformed targets, can
Forge preserve the transformation and give an external developer correct predictions and evaluation
results in the original target units?* Answer: yes, and the measured benefit on a real workload is large
enough to justify it (section 5).

## 1. Problem

Forge's artifact carries the **input** side of a model's contract (`preprocessing=`, M71) and nothing on
the **target** side. A regression model trained on a target of large magnitude (California house prices in
dollars, ~2×10⁵) trains measurably worse, and the only remedy — standardising `y` by hand — leaves an
artifact that predicts *z-scores*. Whoever consumes it (`forge model predict`, a bundled `predict.py`, any
code that only has the `.forge` file) receives `-0.6` where the truth is `$89,400`, and must know a mean and
standard deviation that live nowhere in the file.

## 2. M115 evidence (reproduced, not assumed)

M115 measured, and deliberately did not fix, the effect. It was **re-run on the M116 starting tree** (pristine
`HEAD`, `git archive` into a scratch directory, neutral cwd, `forge.__file__` confirmed) with the same
3,000-row training pool / 2,000 held-out test rows (`default_rng(123)` permutation of StatLib
`cadata.txt`, 20,640 rows, target median house value in dollars), seeds 0–2, `train_tabular_regressor()` at its
defaults:

| pristine HEAD, test rows in **native dollars** | R² | MSE | MAE | epochs | cap (500) reached | s/run |
|---|---|---|---|---|---|---|
| native target (as given) | **0.665** (0.664–0.666) | 4.378×10⁹ | 47,988 | 500, 500, 500 | **yes, 3/3** | 72 |
| caller z-scores `y`, inverts by hand | **0.752** (0.746–0.759) | 3.241×10⁹ | 39,872 | 133, 96, 178 | no | 19 |

Baseline MSE (predict the test mean) 1.307×10¹⁰. These reproduce M115's 0.665 / 0.752 to three digits.

## 3. Current architecture (what HEAD actually is)

Read before changing anything; M115's summary matched HEAD.

- **`train_tabular_regressor()`** (`forge/training/tabular.py`): validate → `random_split` (seeded) → fit
  `Normalize` (optionally after `ReplaceValue`) on the **training rows only** → `TensorDataset`/`DataLoader` →
  default MLP or caller `model=` (run once on real rows to check its output width) → `MSELoss`/`Adam` →
  `train_and_save(task="regression")` → `load_predictor(path).evaluate()` on the saved artifact for every number in
  the result. Targets were used as given.
- **Artifact** (`forge/serialization/model.py`, `archive.py`): a ZIP of `metadata.json` + `.npy` parameters.
  Metadata keys: `forge_format_version` (**2**, bumped once, M53), `device`, `root` (module tree), `preprocessing`
  (a registry-serialised `Transform`), `classes`, `task`. Every optional key so far (preprocessing M71, classes
  M72, task M87) was added **without** a version bump. `load_model()`/`inspect_model()` accept exactly one version.
- **Prediction** — one shared body: `_predict_tensor_core()` (coerce → `InputSchema` feature-count check →
  persisted preprocessing → finite check → `predict()`) is called by `predict_tensor_artifact()`,
  `predict_tabular_classification_artifact()`, `ArtifactPredictor.predict()` **and** `ArtifactPredictor.evaluate()`
  (`_evaluate_features`). `predict_model()` dispatches by `ModelInfo.task`. `load_predictor()` composes
  `inspect_model()` + `load_model()` + `load_preprocessing()` + `load_classes()` once.
- **`InputSchema`**: only `feature_count` of the leading `Linear`; structural, input-side.
- **Existing transform abstractions**: `forge.data.transforms.Transform` (Tensor → Tensor, **no inverse**, an input
  pipeline) plus a public registry in `forge.serialization.transforms` (`register_transform`). Neither is shaped for
  a NumPy, invertible, output-side map.
- **Results**: `TrainingResult` / `TrainAndSaveResult` carry per-epoch losses and metrics in whatever space the
  model trained in; `RegressionEvaluationResult` (`mse`, `mae`, `loss`, `baseline_mse`) is computed from
  `predict()` output against the caller's `y`.
- **Consumers that re-save**: `forge model convert` calls `load_model` + `save_model(preprocessing, classes, task)` —
  it would silently drop any *new* metadata. `forge model inspect`/`predict` read `inspect_model()`/`predict_model()`.

## 4. Real workload methodology

Same rows, split, seeds, model family (default `Linear(8,64)-ReLU-Linear(64,32)-ReLU-Linear(32,1)`), optimizer,
learning rate, patience and input preprocessing throughout; the only variable is the target. Every metric is
computed **in native dollars on the 2,000 held-out rows** by an independent NumPy path from
`predictor.predict()` output (then cross-checked against `predictor.evaluate()`: relative difference ≤ 1.4×10⁻⁸).
Transform statistics are fitted from training targets only (by hand in section 2 from the 3,000-row pool, never the
test rows; by Forge itself in section 5 from the internal *training split* of that pool).

## 5. Target-transform experiment (M116 tree)

`target_transform=None` is bit-for-bit the pre-M116 behaviour (same MSEs to every printed digit as section 2; an
identical SHA-256 over the artifact's parameters and metadata under HEAD and the M116 tree, section 12).

| M116 tree, test rows in **native dollars** | R² | MSE | MAE | val MSE / baseline | epochs | cap reached | s/run |
|---|---|---|---|---|---|---|---|
| `target_transform=None` | 0.665 (0.664–0.666) | 4.378×10⁹ | 47,988 | 0.31–0.35 | 500, 500, 500 | **3/3** | 65 |
| `target_transform="standardize"` | **0.749** (0.748–0.750) | 3.279×10⁹ | 40,205 | 0.23–0.27 | 133, 96, 97 | **0/3** | 16 |

**Materially better**: +0.084 R², −25% test MSE, −16% MAE, training ~4× faster and *no longer exhausting the epoch
cap* — while the built-in transform matches the hand-scaled workaround (0.749 vs 0.752; the remaining gap is fitting
on the 2,400-row training split rather than the 3,000-row pool). The claim is not the smaller *training* loss (which is
in a different unit and meaningless to compare); it is the native-dollar held-out numbers above.

The trigger M115 recorded — *a consumer that must return native units* — is also demonstrated, section 13: the
hand-scaled artifact, given to an ordinary consumer, prints `[-0.6, 2.4, -1.1, -0.7, 1.7]` for house values.

## 6. Selected design

```text
Selected design:  A — a persisted, output-side target transform. `forge.data.StandardizeTarget`
                  (fit / transform / inverse_transform, per target column) is stored as an optional
                  "target_transform" metadata entry beside "preprocessing"; the ONE shared
                  _predict_tensor_core() applies its inverse to the model's raw output.
Why:              The artifact — not the caller's memory — converts back, and the model stays exactly what
                  was trained. It mirrors the existing input-side design (preprocessing is already an
                  artifact-level property, not baked into weights), works for any `model=` (a
                  non-Linear last layer, multi-output with per-column scales), keeps the saved weights
                  equal to the trained/early-stopping-restored weights, and the mean/std are visible in
                  `inspect_model()` / `forge model inspect`. One application point means no prediction or
                  evaluation path can return un-inverted numbers.
Rejected alternative: B — fold `y = z·std + mean` into the final Linear after training.
Why:              (1) Only valid when the last layer is a `Linear`: a ReLU/Sigmoid/Tanh head, a custom
                  Module, or any future architecture is unsupported, so `model=` would be split into
                  two classes of behaviour. (2) It hides a *preprocessing decision inside model
                  parameters*: saved weights ≠ trained weights, the target statistics are not recoverable
                  from the file, the change cannot be inspected, undone or audited, and early-stopping's
                  restored best weights would be rewritten after the fact. (3) The model would no longer
                  represent what was trained, so any lower-level use of the saved model (`Trainer.evaluate`,
                  resuming from a checkpoint) mixes two target spaces with no marker. (4) It still needs the
                  same per-column bookkeeping for multi-output. B's one real advantage — no format change, and
                  an *older* Forge would read it correctly — is a compatibility convenience, not a semantic one,
                  and is what section 8 addresses instead.
Compatibility consequence: Existing artifacts are untouched: no key + version 2 = identity, read and predicted
                  exactly as before, no regeneration. An artifact WITH a target transform is written as
                  format version 3, which a pre-M116 Forge refuses ("unsupported format version 3") rather
                  than misreading — that forward incompatibility is the deliberate cost of A.
```

## 7. Rejected design (detail)

See B above. Not implemented in any form; there is no "fold" code path. Also considered and rejected as *scope*:
a `Transform`-style registry entry (`register_transform` for targets), a `TargetTransform` base class, `minmax` /
`log` / `robust` variants, and per-column choice of transform. None has a workload behind it; `"standardize"` is a
closed vocabulary (`TARGET_TRANSFORM_TYPES`) so adding one later is an explicit format decision, not a plugin.

## 8. Artifact compatibility strategy

- **Old → new (must keep working).** A version-2 file without `"target_transform"` loads as the identity. Verified
  on the bundled `models/tabular_regressor/concrete_strength_regressor.forge` (predictions exactly equal to a manual raw
  forward pass; `inspect` text output unchanged, and `--json` gains only a `"target_transform": null` key), on an artifact
  written by pristine HEAD (read by the M116 tree, identical predictions) and by the existing tests, all unmodified.
- **Default writes are byte-for-byte what they were.** No transform → version 2, no new key (an identical SHA-256
  over parameters + metadata under HEAD and the M116 tree).
- **New → old (must fail loudly).** With a transform the file is **version 3**; a HEAD build reading it raises
  `PersistenceError: … unsupported format version 3`. Without a version bump it would have ignored the unknown key and
  returned z-scores as if they were dollars — silent misinterpretation, the exact failure this milestone exists to
  remove. Verified against pristine HEAD (section 12).
- **Strict in both directions inside this build.** Version 3 *requires* the entry; version 2 *forbids* it. A stripped
  key, a smuggled key, an unknown type, `std ≤ 0`, non-finite or length-mismatched values, or a transform on a
  non-regression task are `PersistenceError`s from `inspect_model`, `load_target_transform`, `load_predictor` and
  `predict_tensor_artifact` alike.
- Under `maintenance.md`'s versioning this is **additive/Minor**: no existing call, artifact, default or file format
  changes meaning; `FORMAT_VERSION` (the version every non-transform artifact is written with) stays 2.

## 9. Implementation

Framework changes (small, one new module):

- `forge/data/target_transform.py` (new) — `StandardizeTarget(mean, std)`, `.fit(y)` (classmethod, population std),
  `.transform(y)`, `.inverse_transform(z)`, `to_config()` / `from_config()`, equality. Host NumPy float64. A **constant**
  target column is a `DataError` (Forge's convention for a caller-chosen operation that cannot be defined; features
  get `std = 1` because a constant feature is harmless, a constant target means nothing to learn or wrong data).
  Exported as `forge.data.StandardizeTarget`.
- `forge/serialization/model.py` — `save_model(..., target_transform=None)`; versions 2/3
  (`SUPPORTED_FORMAT_VERSIONS`, `TARGET_TRANSFORM_FORMAT_VERSION`); `load_target_transform()`;
  `ModelInfo.target_transform`; strict metadata validation shared by `inspect_model` and `load_target_transform`.
- `forge/training/inference.py` — `_predict_tensor_core(..., target_transform=)` applies the inverse; the four callers
  (`predict_tensor_artifact`, `ArtifactPredictor.predict`, `ArtifactPredictor.evaluate`, and through it
  `predict_model`) pass the artifact's; `ArtifactPredictor.target_transform`; `save_and_verify(..., target_transform=)`
  also proves the transform survived the round trip. `load_predictor()` needed **no change**: `inspect_model()` already
  loads it.
- `forge/training/api.py` — `train_and_save(..., target_transform=)`, a pass-through.
- `forge/training/tabular.py` — `train_tabular_regressor(..., target_transform=None | "standardize")`; the transform is
  fitted after the split on training targets only, validation targets are transformed with those statistics, the model
  trains on z-scores, and `TabularRegressionResult` gains `target_transform`. All result metrics still come from the saved
  artifact's `evaluate()` against **native** targets. `_preflight.preflight_save` dry-runs the save with the transform.
- `forge/cli/model.py`, `forge/cli/_archive_info.py` — `inspect` reports it; **`convert` carries it over** (it would
  otherwise have silently produced z-score artifacts); the CLI accepts version 3.
- Docs: README, `train_tabular_regressor()` / `tabular.py` docstrings, `docs/architecture/persistence.md`
  (**Target transform metadata**, Versioning), `docs/architecture/training-engine.md`, `docs/development/cli.md`,
  `progress.md`, this file.

Not done, on purpose: a new prediction path, arbitrary callables, pickle, a transform registry/plugin system, a
default-on transform (the default stays `None`, so nothing changes for existing callers), CUDA kernels (the arithmetic is
host-side), changes to `Trainer`/`TrainingResult`/`ArtifactPredictor`'s constructor, forecasting/DataFrame/AMP.

**Semantics as shipped.** `predict()` returns native units (a float32 `Tensor`, same type as before). `evaluate()`'s
`mse`/`mae`/`loss`/`baseline_mse` are native units (squared for the squared ones); `y` passed to it is always native.
`TabularRegressionResult.*_mse/_mae/_loss/baseline_mse` are native. What is **not** native and is documented as such:
`result.history` and `best_monitored_value` (per-epoch curves and early stopping's validation loss, in z-scores) and the
bare `result.model` / `load_model()` `Module` (its output is the training space). `train_and_save()` /
`TrainAndSaveResult` are generic and unchanged: they report the space the caller trained in, which the docstring now says.

## 10. Test results

| suite | result |
|---|---|
| `tests/test_target_transform.py` (new, CPU) | **81 passed** |
| `tests/test_target_transform_cuda.py` (new, CUDA hardware) | **7 passed** on the 940MX |
| M114/M115/M113/CLI/inspection/serialization/bundled-model tests, **unmodified** | pass |
| Full suite `python -m pytest tests/ -q` (12 min 26 s) | **3,121 passed, 0 failed, 0 skipped**, CUDA available (M115 baseline 3,033 + 88 new) |

No existing test was edited. The new tests recompute expectations **independently**: the expected native prediction is
a hand-run `load_model()` forward pass plus `z·std + mean` with `mean`/`std` read straight from `metadata.json`
(no `StandardizeTarget`, no `ArtifactPredictor`); the split is reproduced from `random_split`'s documented rule; metrics
are NumPy. Coverage: unit behaviour of `StandardizeTarget` (fit/transform/inverse, multi-column, unbatched, ints,
`Tensor`, bad shapes/types/non-finite/empty/constant, construction errors, config round trip); artifact format (v2 without
key, v3 with, save-time rejections, 11 corrupt/missing-metadata cases each through four APIs, unknown version, model/
transform width mismatch); native-unit prediction through `predict`, `predict_tensor_artifact`, `predict_model` and
`forge model predict`; unbatched rows; wrong feature count; native `evaluate()`; training (default unchanged and writes
nothing new, fit on the training split only, native result metrics, multi-output, custom non-Linear-ending model,
unsupported names, constant / non-finite / empty / mis-shaped / mismatched targets before epoch 1); `save_and_verify`; the
bundled pre-M116 artifact; `forge model convert`/`inspect`; a fresh subprocess outside the repo.

**Mutation checks** (each applied to the source, the new test file run, the source restored — every one caught):

| mutation | tests failing |
|---|---|
| remove target-transform persistence (`save_model` ignores it) | 16 (+14 fixture errors) |
| remove the inverse transform | 10, incl. fresh-process, native predict, native evaluate |
| fit the transform on **all** targets (leakage) | 3 (incl. `fitted_on_the_training_split_only`) |
| fit the transform on the **validation** targets | 3 |
| `evaluate()` does not convert to native units | 3 (incl. native result metrics) |
| an old (no-key, v2) artifact stops reading as identity | 4 (incl. bundled artifact) |
| a v3 file with the key stripped is accepted | 1 |
| a constant target silently gets `std = 1` | 4 |
| `forge model convert` drops the transform | 1 |
| `predict_tensor_artifact` forgets the transform | 1 |
| transform written but version left at 2 | 15 (+14 errors) |

## 11. CPU / CUDA results

Target transformation is host-side NumPy arithmetic on the copied-back model output; it adds no device-specific behaviour
and no CUDA kernel.

**Real workload on the 940MX** (California housing, same rows/split, seed 0, `device="cuda"`; CPU rows from section 5):

| device | target | R² (test, native $) | MAE | epochs | cap reached | s/run |
|---|---|---|---|---|---|---|
| CUDA | `None` | 0.665 | 48,225 | 500 | yes | 417 |
| CUDA | `"standardize"` | **0.742** | 40,010 | 120 | no | 94 |
| CPU | `None` | 0.665 | 48,226 | 500 | yes | 64 |
| CPU | `"standardize"` | **0.748** | 39,740 | 133 | no | 18 |

Same conclusion on both devices. Independently trained CPU and CUDA models are not bit-identical (pre-existing, documented
in M114) — 0.742 vs 0.748 — and, as in M114, this small MLP trains ~5× *faster* on CPU than on the 940MX; CUDA is supported and
hardware-tested, not a speed-up here. The `None` runs reproduce the CPU numbers (0.6647 vs 0.6647).

**Same artifact, both devices** (`tests/test_target_transform_cuda.py`, 7 tests, real hardware, plus the experiment): loading one
saved artifact onto CPU and CUDA and predicting native dollars agrees to `rtol=1e-4, atol=1e-2` (the tolerance
`test_tabular_workflows_cuda.py` already uses); on 2,000 test rows the largest disagreement was **0.094 dollars** on values of order
10⁵ (relative 2.7×10⁻⁷). Also verified on CUDA: a CUDA-trained artifact predicts native units and its result metrics equal an
independent computation (`rtol=1e-3`); a CPU-trained artifact predicts identically on CUDA; `forge model convert --device cpu`
from a CUDA artifact keeps the transform; a fresh process loads the CUDA artifact onto CUDA and returns native units; the
returned prediction `Tensor` is on the CPU (the inverse ran host-side).

## 12. Fresh-process and compatibility results

- **Fresh process** (`tests/test_target_transform.py`, subprocess, `cwd` outside the repo, `CUDA_VISIBLE_DEVICES=-1`):
  predictions equal the in-process predictions **exactly**, equal the process's own independent
  `z·std + mean` recomputation to 1×10⁻⁶ relative, and `evaluate()` reproduces `mse`/`mae`/`baseline_mse` exactly.
- **Default training, pristine HEAD vs M116 tree** (same seed, 400 California rows, 60 epochs): identical SHA-256
  over all parameters + metadata (`b22b286c…`), identical keys (`classes, device, forge_format_version, preprocessing,
  root, task`), version 2, identical predictions.
- **M116 tree reads a HEAD-written artifact**: predictions identical to HEAD's (the bundled Concrete artifact is covered by a
  test against a manual forward pass).
- **HEAD reads a v3 artifact**: `PersistenceError: Cannot inspect model … unsupported format version 3`.

## 13. Consumer results (installed wheel, clean venv, fresh process, outside the repo)

`python -m build --wheel` → a brand-new `venv` containing only `forge` (the wheel), `numpy`, `Pillow` — no repo on
`sys.path`, no pytest (`forge.__file__` under the venv's `site-packages`), cwd a temp directory. The consumer:

```python
import json, sys
import forge

predictor = forge.load_predictor(sys.argv[1], device="cpu")
features = json.load(open(sys.argv[2]))

prediction = predictor.predict(features)

print(json.dumps({"forge_from": forge.__file__, "prediction": [round(v, 1) for v in prediction.numpy()[:, 0].tolist()]}))
```

It contains no `mean`, `std`, `* std + mean` or any inverse. Five held-out California blocks (true median value →
prediction from the wheel-installed consumer): `125,000 → 141,306`, `500,001 → 486,215`, `89,400 → 89,511`,
`120,800 → 133,843`, `500,001 → 411,287` — native dollars, identical to the in-repo prediction. `forge model predict
… --json` from the venv's `forge.exe` printed the same values, and `forge model inspect` reported format version 3.
**Control:** the same consumer against the hand-z-scored artifact (what a user had to build before M116) printed
`[-0.6, 2.4, -1.1, -0.7, 1.7]`.

## 14. Limitations

- **Float32 output.** `predict()` still returns a float32 `Tensor`, so native values carry float32 resolution
  (~0.016 at 2×10⁵). A target whose *offset dwarfs its spread by more than ~10⁵×* (e.g. 10⁹ ± 1) would be quantised by
  that; the transform arithmetic itself is float64 and `StandardizeTarget.inverse_transform` returns float64 for callers who
  need it. Not measured on a real workload; not changed.
- **Training-space numbers are not converted.** `result.history` / `best_monitored_value` / `result.model` stay in z-scores
  (documented in the result and the docstrings); for multi-output targets no single scalar converts a z-space loss to native.
- **Only `"standardize"`.** No min-max/log/robust transform; no per-column choice; one vocabulary entry per format decision.
- **A pre-M116 Forge cannot open a transformed artifact** (by design, see section 8).
- **Not the default.** Callers with large-magnitude targets must opt in; whether a default-on policy would help or hurt at
  small magnitudes (e.g. Concrete, ~36 MPa) was **not measured** here.
- Only measured on one real dataset (California housing) plus synthetic data in tests; CUDA training is not faster than CPU
  for these small MLPs on the reference 940MX (pre-existing, M114).

## 15. Deferred items

Default-on target standardisation; other transforms; converting per-epoch history to native units; surfacing the transform in
`predict_model()`'s return; a float64 prediction option; time-series/windowed targets. None is needed by the evidence here.

## 16. Final decision

Implement, narrowly. The capability materially improves the real workflow (R² 0.665 → 0.749, epoch cap no longer hit,
~4× faster), removes an out-of-band constants requirement that a real consumer could not satisfy, and fits the existing
artifact model without a second prediction path. Old artifacts are untouched; new ones are refused by old readers rather
than misread.

Decision: IMPLEMENTED
