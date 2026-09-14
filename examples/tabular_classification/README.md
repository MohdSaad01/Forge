# Forge Tabular Classification Example (Milestone 91)

A workload-driven validation of Forge's training/data pipeline for **tabular
classification** -- structured, non-image input mapped to a class label --
the one combination of "tabular input" (already exercised by
`examples/regression`) and "classification output" (already exercised by
`mnist`/`image_folder_classification`, always over image input) no prior
example combined:

```text
make_datasets() -> DataLoader -> MLP (Linear/ReLU/Linear/ReLU/Linear)
    -> CrossEntropyLoss -> Adam -> forge.train_and_save() -> forge.predict_model()
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`). Nothing here is new
framework logic beyond this milestone's own fix (see below) -- this example
generates a deterministic synthetic dataset and assembles a small MLP from
existing `forge.nn` layers, exactly like `examples/regression`.

## The classification problem

A synthetic device-telemetry health classifier: 10 continuous sensor-style
features per device, four health states (`normal`/`warning`/`critical`/
`fault`). Six of the ten features are genuinely discriminative (drawn around
one of four fixed per-class center vectors, with deliberate class overlap so
the problem is learnable but not trivially separable); the remaining four
are pure distractors, drawn independent of class -- mirroring
`examples/regression/dataset.py`'s own `x7` distractor feature. See
`dataset.py`'s module docstring for the full generation contract.

## Why `random_split`, not `sequential_split`

Unlike `examples/regression` (i.i.d. samples, so `sequential_split` and
`random_split` are equivalent and the simpler one is used), this example's
raw data is generated in **class-block order** -- every `normal` row, then
every `warning` row, and so on -- mirroring an ordinary real-world shape (a
CSV exported one class at a time). `forge.data.sequential_split` would put
entire classes only in the training split, with none in validation/test; a
real, demonstrated reason to use `forge.data.random_split` instead, which
already handles this correctly with no new splitting capability needed --
see `dataset.py::make_datasets()`'s own docstring for the full reasoning
(Milestone 91 brief, Section 12).

## Model

```text
(N, 10) -> Linear(10, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 4) -> (N, 4)
```

Raw per-class logits (`CrossEntropyLoss`'s expected input) -- the one
architectural difference from `examples/regression`'s otherwise identical
shape.

## CPU training

```bash
python -m examples.tabular_classification.train --epochs 30 --device cpu
```

Trains the MLP above with Adam (`lr=1e-3`) over `4 * 300 = 1,200` synthetic
training samples (`4 * 60 = 240` validation, `4 * 60 = 240` test -- see
`--samples-per-class-*`), reports per-epoch training/validation loss and
accuracy, then saves a verified model file under `--output-dir` (default
`examples/tabular_classification/artifacts`). Reference numbers from this
repository's CPU (i5-7200U, `--seed 0`, default hyperparameters, 30 epochs):
final test accuracy **82.1%** against a **25.0%** trivial (uniform-random)
baseline -- genuine learning, not merely "the code runs." Exact numbers vary
by hardware/seed; only "well above the trivial baseline" is a stability
guarantee (`tests/test_tabular_classification_artifact_prediction.py`
checks this directly on a smaller sample count/epoch budget for test speed).

## CUDA training

```bash
python -m examples.tabular_classification.train --epochs 30 --device cuda
```

Identical model/optimizer/data pipeline; hardware-verified on the reference
GeForce 940MX.

## No checkpoint/resume

Unlike `examples/regression`, this script has a single fresh-training path
only -- no `--resume`. This dataset generates and trains fast enough
(single-digit seconds for the full run above) that interrupted-training
resume is not a realistic concern for this workload (Milestone 91 brief,
Section 15) -- checkpoint/resume remains available for any workload that
needs it via `forge.save_checkpoint()`/`TrainingSession`, unmodified.

## Determinism

`--seed` (default `0`) governs four independent, deliberately separate
generators: `forge.random.seed()` (`Linear` parameter initialization),
`dataset.py::generate_raw()`'s own `numpy.random.default_rng(seed)` (the
synthetic features/noise), `make_datasets()`'s own
`numpy.random.default_rng(seed + 1)` (the train/val/test `random_split`
permutation), and `DataLoader`'s explicit shuffle generator -- matching
`examples/regression/train.py`'s documented policy.

## Model persistence

`train.py` trains and saves through one `forge.train_and_save()` call,
including the fitted, training-split `Normalize` feature-standardization
transform (`preprocessing=`), the class vocabulary (`classes=CLASS_NAMES`),
and `task="tabular_classification"` (**not** `task="classification"` --
see below). The model is saved, then reloaded fresh and its prediction
confirmed to match the pre-save model, raising `forge.PersistenceError`
instead if the reload ever disagrees -- the same `save_and_verify()`
guarantee every other `forge.train_and_save()`-based example gets.

## Portable-artifact inference: the Milestone 91 fix

A developer holding only `tabular_classification_model.forge` gets a
classification prediction on a brand-new raw feature vector in one call:

```python
import forge
import numpy as np

raw_features = np.array([[0.5, -1.2, 0.3, 1.8, -0.4, 2.0, -1.5, 0.9, 0.1, -0.2]], dtype=np.float32)
results = forge.predict_model("examples/tabular_classification/artifacts/tabular_classification_model.forge", raw_features)
print(results[0].label, results[0].confidence)
```

**This did not work before Milestone 91.** Before this milestone, every
classification artifact -- regardless of whether its input was an image file
or a numeric feature vector -- was saved with `task="classification"` (the
only classification task that existed), which `forge.predict_model()`
always routed to `forge.predict_artifact()`. That function requires `image`
to be a file path; calling it with a NumPy array raised:

```text
forge.exceptions.DataError: predict_artifact() requires image to be a file path (str or os.PathLike), got ndarray.
```

This is the real, reproduced blocker this milestone found and fixed: a
tabular classification model could already be fully *trained* with
`Trainer`/`DataLoader`/`CrossEntropyLoss`/`classes=` (nothing about training
needed to change), but had no supported *inference* path at all. The fix is
a new task value, `task="tabular_classification"` (`forge.serialization.
model.TASK_TYPES`), and its own artifact-shape function,
`forge.predict_tabular_classification_artifact()` -- composing
`load_model()` + `load_preprocessing()` (optional) + `predict()` +
`load_classes()` (optional) + `interpret_classification()`, the same
composition pattern `predict_artifact()`/`predict_tensor_artifact()`
already established for their own artifact shapes. See
`docs/development/m91-tabular-classification-artifact-inference.md` for the
full investigation.

**Returns one result per input row.** Unlike `predict_artifact()` (always
exactly one image in, one prediction out), `predict_tabular_classification_
artifact()`'s input follows `predict_tensor_artifact()`'s "already batched,
any batch size" convention -- so it returns a list, one `ClassificationPrediction`
per row, even for a single-sample query (`results[0]` above).

`tests/test_tabular_classification_artifact_prediction.py` proves this
end-to-end with a genuinely separate OS process: it trains and saves a real
artifact in one process, then launches a `subprocess` that only imports
`forge`/`numpy` and calls `forge.predict_tabular_classification_artifact()`
directly, confirming its printed prediction matches this process's own
prediction on the identical raw input.

## CLI inspection and prediction

```bash
python -m forge model inspect examples/tabular_classification/artifacts/tabular_classification_model.forge
```

```bash
echo '[0.5, -1.2, 0.3, 1.8, -0.4, 2.0, -1.5, 0.9, 0.1, -0.2]' > input.json
python -m forge model predict examples/tabular_classification/artifacts/tabular_classification_model.forge input.json
```

`forge model predict` shares `regression`'s JSON-numeric-file input parsing
(a flat list is one sample; a nested list is already batched) but prints
classification-shaped output -- one block per input row for a multi-row
JSON file, since the input may legitimately batch more than one sample.

## Fresh-process inference

```bash
python -m examples.tabular_classification.infer \
    --model examples/tabular_classification/artifacts/tabular_classification_model.forge \
    --input '[0.5, -1.2, 0.3, 1.8, -0.4, 2.0, -1.5, 0.9, 0.1, -0.2]'
```

A standalone script independent of `train.py` -- see `infer.py`'s own module
docstring.

## Integration tests

`tests/test_tabular_classification_artifact_prediction.py` (CPU) and
`tests/test_tabular_classification_artifact_prediction_cuda.py` (CUDA;
skips cleanly without a working CUDA backend) exercise this exact pipeline
end-to-end, covering: deterministic dataset generation, the `random_split`
class-representativeness property, model construction/forward shape,
training loss reduction well below the trivial baseline, model save/load
prediction consistency, CPU/CUDA prediction parity, and CLI inspection/
prediction of generated artifacts. Run them with:

```bash
python -m pytest tests/test_tabular_classification_artifact_prediction.py tests/test_tabular_classification_artifact_prediction_cuda.py
```
