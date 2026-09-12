# Forge Image-Folder Classification Example (Milestone 69, extended in 70/71/72/80/81)

An end-to-end validation of `forge.data.ImageFolder` -- Forge's first
dataset that discovers labeled samples from **ordinary image files on
disk**, rather than a bundled binary format (`examples/mnist/`'s IDX files)
or in-memory arrays -- extended (Milestone 70) so the source images are
genuinely **mixed-resolution**, made batchable via
`forge.data.transforms.Resize`, extended again (Milestone 71) so the
saved model file carries the exact preprocessing configuration a new image
must go through, reconstructed automatically in a *separate inference
process*, and extended once more (Milestone 72) so the saved model file
*also* carries the class-name vocabulary its predicted indices refer to --
no more manually remembering or re-implementing `Resize`/pixel scaling, and
no more a hand-written `classes.json` sidecar file, at inference time:

```text
generate_dataset() [mixed H, W] -> ImageFolder -> Resize+Normalize -> random_split -> DataLoader
    -> forge.train_and_save() -> CNN (Conv2d/BatchNorm2d/MaxPool2d/Dropout) -> CrossEntropyLoss
    -> Adam -> save + verify (model + preprocessing + classes) -> forge.predict() -> interpret_classification()
                                                                ^
                                infer.py / `forge model predict`: fresh process, everything from one file
```

Milestone 80 retrofitted this script's fresh (non-`--resume`) path to train
through `forge.train()` (Milestone 79's single-call high-level training
entry point) instead of hand-assembling `Trainer(...)` + `trainer.fit(...)`
itself -- the same retrofit Milestone 79 applied to `examples/mnist/
train.py`. Milestone 81 replaced that fresh path's `forge.train()` call
followed by a separate `save_and_verify()` call with one
`forge.training.train_and_save()` call -- the identical two-call sequence
`examples/mnist/train.py`'s fresh path also used, now written once. `--resume`
still uses `forge.training.start_training_session()` (Milestone 73) followed
by a direct `save_and_verify()` call, preserving this example's exact
`DataLoader`-shuffle resume-equivalence guarantee with zero regression --
see `docs/architecture/training-engine.md`'s **Single-call high-level
training** and **Train, evaluate, persist, verify in one call** sections for
exactly how the fresh path's checkpoint stays resumable.

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.save_model`/`save_checkpoint`/
`load_model`/`load_checkpoint`/`load_preprocessing`/`load_classes`/
`interpret_classification`). Nothing here is new framework logic -- this
example only generates a small synthetic image dataset and assembles a
small CNN from existing `forge.nn` layers.

## Files

- `generate_dataset.py` -- writes a synthetic `circle`/`square`/`triangle`
  shape-image dataset to disk as PNG files, in an `ImageFolder`-compatible
  directory tree. Each image's height and width are drawn independently
  from `[--min-size, --max-size]` (default `48`-`128`), so the dataset is
  mixed-resolution by construction.
- `model.py` -- `build_model()`, the CNN architecture (expects `(3, 64, 64)`
  input, matching `train.py`'s `Resize((64, 64))`).
- `train.py` -- the runnable example: dataset generation, `Resize`/
  `Normalize` preprocessing, training, evaluation, checkpointing, resume,
  model + preprocessing + classes persistence, and standalone single-image
  inference on both a held-out test sample and a brand-new image at a
  resolution never seen in training.
- `infer.py` (Milestone 71, extended in 72) -- a **separate, standalone
  script**: given only a saved model path and an image path, reconstructs
  the exact preprocessing used during training from the model file itself
  (`forge.load_preprocessing()`) and the class-name vocabulary
  (`forge.load_classes()`), then produces a human-readable prediction. Does
  not import `train.py`'s `build_transform()` or any other in-memory
  training-time state -- see its own module docstring. (Before Milestone
  72 this also needed a separate `--classes path/to/classes.json` argument;
  the model file now carries that itself.)

## Prerequisites

- Forge installed (`pip install -e .` from the repo root), including its
  `Pillow` dependency (used by `ImageFolder` for image decoding -- see
  `docs/development/m69-image-folder.md`).
- For CUDA training: a working Forge CUDA backend (see
  `docs/architecture/cuda-backend.md`).

## The dataset

Nothing is downloaded. `train.py --generate` (or
`generate_dataset.generate_dataset(root, ...)` directly) writes 300 PNG
images per class (900 total by default) into:

```text
examples/image_folder_classification/data/
    circle/
        circle_0000.png
        ...
    square/
        ...
    triangle/
        ...
```

Each image is an RGB shape rendered on a light background, with random
**position**, **scale**, **rotation** (square/triangle), **shape color**
(always meaningfully darker than the background, but the exact hue varies
per image so color can't be memorized), per-pixel Gaussian noise, and
(Milestone 70) **height and width**, each drawn independently and uniformly
from `[--min-size, --max-size]` (default `48`-`128`) -- real
position/scale/color/resolution variation, not a hard-coded pixel-location
mapping or a uniformly-sized dataset. See `generate_dataset.py`'s module
docstring for the exact rendering procedure. Fully reproducible from
`--seed`.

Because source images are mixed-resolution, `ImageFolder` alone cannot be
batched by `DataLoader` (`_stack` requires every sample in a batch to share
one shape). `train.py` applies `forge.data.transforms.Resize((64, 64))`
before any pixel-scaling transform, normalizing every sample to `(3, 64,
64)` regardless of its source file's dimensions -- see
`docs/development/m70-image-preprocessing.md` for the full investigation
and `forge/data/transforms.py`'s `Resize` docstring for the transform
itself.

## CPU training

```bash
python -m examples.image_folder_classification.train --generate --device cpu
```

`ImageFolder("examples/image_folder_classification/data", transform=Resize((64, 64))
-> Normalize(mean=0.0, std=255.0))` discovers the 3 classes and normalizes every
mixed-resolution sample to `(3, 64, 64)`; `forge.data.random_split` (an
existing primitive, not new example logic) carves an 80/20 train/test split
from the single directory, since a freshly generated dataset has no
separate split on disk the way MNIST's four files do. Trains a small CNN
(`Conv2d(3,16,3) -> BatchNorm2d -> ReLU -> MaxPool2d(2) -> Conv2d(16,32,3)
-> BatchNorm2d -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(6272,64) -> ReLU
-> Dropout(0.3) -> Linear(64,3)`) with Adam (`lr=4e-4`), reports per-epoch
training/validation loss and accuracy, then saves a checkpoint and model
file.

### Expected approximate behavior (reference: this repository's CPU, i5-7200U)

| Epoch | Train loss | Train acc | Val acc |
|------:|-----------:|----------:|--------:|
| 1     | 1.117      | ~40%      | ~36%    |
| 25    | 0.077      | ~99%      | ~58%    |

~400s total for 25 epochs on the reference CPU -- meaningfully slower than
Milestone 69's uniformly-32x32 run (~68-75s), expected since every sample is
now decoded from a larger (48-128px) source file, resized to 64x64 (4x the
pixel count of 32x32), and the model's final `Linear` layer grew
proportionally (1152 -> 6272 input features). Measured directly: dataset
`__getitem__` (decode+resize+scale) over all 900 samples took ~1.4s/epoch
(~9% of a ~16s epoch) -- image preprocessing is not the bottleneck; the
larger model's forward/backward pass dominates. See
`docs/development/m70-image-preprocessing.md`'s Performance section.
Exact numbers vary run to run by hardware/library version but are not a
stability guarantee -- only "loss drops sharply, validation accuracy ends
well above the 3-class 33% chance baseline" is the guaranteed, tested
property (`tests/test_image_folder_classification_integration.py`, on a
smaller/faster synthetic dataset than this full run). The visible gap
between train accuracy (~99%) and validation accuracy (~58%) reflects a
genuinely nontrivial task at this dataset size/model size, not a bug -- see
`docs/development/m69-image-folder.md`'s Training Results section for why
`BatchNorm2d`/`Dropout` were added to reach this from an initial
near-chance result; Resize's fixed-size preprocessing did not remove that
existing gap, and this milestone did not attempt to.

## CUDA training

```bash
python -m examples.image_folder_classification.train --generate --device cuda
```

Identical model/optimizer/data pipeline; only `Trainer(..., device="cuda")`
and `model.to("cuda")` differ. `forge.data`/`ImageFolder`/`Resize`
themselves always produce plain CPU Tensors -- `Trainer` is the layer that
moves each batch to CUDA, exactly as it does for every other Forge dataset
(`docs/architecture/data-system.md`'s "Device behavior" section).
Hardware-verified on the reference GeForce 940MX: 109.1s for 25 epochs
(~3.7x the CPU run's 402.4s), first-epoch loss closely matching the CPU
run's (1.1095 vs. 1.1173, same `--seed`). See
`docs/development/m70-image-preprocessing.md` for full measurements.

## Determinism

`--seed` (default `0`) governs three independent things: `forge.random.seed()`
(model parameter initialization), the synthetic dataset generator (which
shapes/colors/positions get drawn), and the `random_split`/`DataLoader`
shuffling generators (`numpy.random.Generator`, seed-derived but independent
of Forge's own default generator) -- matching every other Forge example's
documented RNG policy.

## Checkpointing, resume, and persistence

Identical pattern to every other Forge example
(`docs/architecture/persistence.md`): `train.py` saves a checkpoint capturing
model + Adam state + epoch/global_step + RNG state, supports `--resume`, and
saves + verifies the model through `forge.training.save_and_verify()`
(Milestone 78) -- called directly on `--resume`, or as the second half of
`forge.train_and_save()` (Milestone 81) on the fresh path -- which saves the
model with its `preprocessing=Compose([Resize(...), Normalize(...)])`
pipeline (Milestone 71) and `classes=full_dataset.classes` vocabulary
(Milestone 72) as two sibling metadata entries in the same `.forge` model
file (`save_model(model, path, preprocessing=..., classes=...)` -- no
separate sidecar file), then reloads it fresh and confirms the reload's
prediction matches the pre-save model, raising `forge.PersistenceError`
instead if the reload ever disagrees -- see `docs/architecture/
persistence.md`'s **Preprocessing metadata** and **Class-label metadata**
sections, and `docs/architecture/training-engine.md`'s **Portable-artifact
save + verify** and **Train, evaluate, persist, verify in one call**
sections. (Milestone 81: the fresh path trains, saves, and verifies the
`.forge` model file via one `forge.train_and_save()` call, then writes its
checkpoint via plain `forge.save_checkpoint()`; `--resume` still goes through
`start_training_session()` followed by a direct `save_and_verify()` call --
see the pipeline diagram above and `docs/architecture/training-engine.md`'s
**Single-call high-level training** section for why reordering the
checkpoint write after `save_and_verify()`'s reload still preserves exact
resume-equivalence, now that `load_model()` itself no longer leaks a random
draw into `forge.random`'s global state -- see **Train, evaluate, persist,
verify in one call**'s "real hazard found and fixed" note.)

## Inference demonstration

At the end of every run, `train.py` takes one held-out test-split image,
runs it through `forge.predict()`, and turns the raw output into a
human-readable class name + confidence via
`forge.interpret_classification()`, using the class vocabulary reconstructed
from the just-saved model file (`forge.load_classes()`) rather than the
in-memory `ImageFolder.classes` this process still happens to have -- proving
the file alone is enough:

```text
Inference demo (test sample 0, true class: square):
Prediction: triangle
Confidence: 71.4%
```

(A misclassification here is expected some fraction of the time at this
model's ~58% validation accuracy -- not a pipeline bug.)

It then generates one brand-new image that was never part of the dataset,
at a resolution (`200x140`) deliberately outside the `--min-size`/
`--max-size` training range, loads it via the same `ImageFolder._load_image`
decode step, applies the preprocessing pipeline **reconstructed from the
just-saved model file** via `forge.load_preprocessing()` (Milestone 71 --
not the in-process `build_transform()` function), and predicts:

```text
New mixed-resolution image (200x140, true class: circle, never seen during training), preprocessing + classes reconstructed from '...image_folder_model.forge':
Prediction: circle
Confidence: 88.9%
```

This demonstrates the milestones' core claim: a caller does not need to
manually pre-resize a new image to the model's trained resolution, remember
which transform was used, or keep a separate class-name file next to the
model -- everything is reconstructed automatically from the one saved file.
Both demos use the same `forge.predict()`/`forge.interpret_classification()`
used by every other Forge classification workflow -- no separate
hand-written inference implementation.

### Fresh-process inference (`infer.py`, Milestone 71/72; `forge model predict` CLI, Milestone 72)

`train.py` prints the exact commands at the end of its run. Both work from
nothing but the saved model file and a path to any image -- no separate
`classes.json` sidecar file to keep track of:

```bash
python -m examples.image_folder_classification.infer \
    --model examples/image_folder_classification/artifacts/image_folder_model.forge \
    --image examples/image_folder_classification/artifacts/new_mixed_resolution_query.png
```

```text
Prediction: circle
Confidence: 88.9%
```

Or, equivalently, via Forge's own CLI (`forge/cli/model.py`'s `predict`
subcommand -- no Python script needed at all):

```bash
python -m forge model predict examples/image_folder_classification/artifacts/image_folder_model.forge \
    --image examples/image_folder_classification/artifacts/new_mixed_resolution_query.png
```

```text
Predicted class: circle
Confidence: 88.9%
```

`infer.py` never imports `train.py`'s `build_transform()` or reuses any
in-memory object from the training run -- it calls `forge.load_model()`,
`forge.load_preprocessing()`, and `forge.load_classes()` against the same
file and nothing else, demonstrating the workflow across a genuine process
boundary. Passing the path to a model saved *without* `preprocessing=`
raises a clear `PersistenceError` rather than guessing or silently skipping
preprocessing; a model saved without `classes=` still predicts, falling back
to printing the raw class index (see `run()`'s own docstring).

## Integration tests (no dataset download required, generated on the fly)

`tests/test_image_folder_classification_integration.py` (CPU) and
`tests/test_image_folder_classification_cuda_integration.py` (CUDA; skips
cleanly without a working CUDA backend) exercise this exact pipeline against
a smaller/faster generated dataset than the full example run above --
covering dataset generation, model shape, training loss reduction and
above-chance accuracy on real image files, checkpoint save/resume, and
model save/load/predict consistency. Milestone 80 added three more: `train.
main()`'s real fresh path (the exact `python -m examples.image_folder_
classification.train` entry point, not a hand-built `Trainer`) produces a
working artifact (Milestone 81: now via `forge.train_and_save()`); a fresh
run -> `start_training_session()`-resume sequence matches one continuous run
bit-for-bit at this script's real `shuffle=True` default (the
resume-equivalence guarantee the retrofit had to preserve -- still enforced
unchanged after Milestone 81's retrofit, see
`test_resume_after_a_forge_train_fresh_run_matches_continuous_training`);
and `infer.py` is launched as a genuine `subprocess` -- a real separate OS
process, not
just a fresh Python import -- and its stdout is cross-checked against
`forge.predict()`/`interpret_classification()` computed independently
against the same file. Run them with:

```bash
python -m pytest tests/test_image_folder_classification_integration.py tests/test_image_folder_classification_cuda_integration.py
```

Also see `tests/test_image_folder.py` for `forge.data.ImageFolder`'s own
unit tests (discovery, image conversion, dataset behavior, `DataLoader`
integration) -- independent of this example -- `tests/
test_preprocessing_persistence.py` (Milestone 71) for the preprocessing-
persistence mechanism's own tests, including a real `ImageFolder` ->
train -> save -> reload -> `forge.predict()` round trip against files on
disk -- and `tests/test_classification_metadata.py` (Milestone 72) for
`save_model(..., classes=...)`/`load_classes()`/`interpret_classification()`,
the `forge model predict` CLI command, and this example's own
`infer.py`/`train.py` fresh-process workflow.
