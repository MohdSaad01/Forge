# Forge Image-Folder Classification Example (Milestone 69, extended in 70)

An end-to-end validation of `forge.data.ImageFolder` -- Forge's first
dataset that discovers labeled samples from **ordinary image files on
disk**, rather than a bundled binary format (`examples/mnist/`'s IDX files)
or in-memory arrays -- now extended (Milestone 70) so the source images are
genuinely **mixed-resolution**, made batchable via
`forge.data.transforms.Resize`:

```text
generate_dataset() [mixed H, W] -> ImageFolder -> Resize -> random_split -> DataLoader
    -> Trainer -> CNN (Conv2d/BatchNorm2d/MaxPool2d/Dropout) -> CrossEntropyLoss
    -> Adam -> save/load -> forge.predict()
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.save_model`/`save_checkpoint`/
`load_model`/`load_checkpoint`). Nothing here is new framework logic --
this example only generates a small synthetic image dataset and assembles a
small CNN from existing `forge.nn` layers.

## Files

- `generate_dataset.py` -- writes a synthetic `circle`/`square`/`triangle`
  shape-image dataset to disk as PNG files, in an `ImageFolder`-compatible
  directory tree. Each image's height and width are drawn independently
  from `[--min-size, --max-size]` (default `48`-`128`), so the dataset is
  mixed-resolution by construction.
- `model.py` -- `build_model()`, the CNN architecture (expects `(3, 64, 64)`
  input, matching `train.py`'s `Resize((64, 64))`).
- `train.py` -- the runnable example: dataset generation, `Resize`
  preprocessing, training, evaluation, checkpointing, resume, model
  persistence, and standalone single-image inference on both a held-out
  test sample and a brand-new image at a resolution never seen in training.

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
-> pixel-scale Lambda)` discovers the 3 classes and normalizes every
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
after training verifies the save/load round trip reproduces the same
prediction via `forge.predict()`.

## Inference demonstration

At the end of every run, `train.py` takes one held-out test-split image,
runs it through `forge.predict()`, and maps the predicted class index back
to a human-readable class name via `ImageFolder.classes`:

```text
Inference demo (test sample 0, true class: square):
Prediction: triangle
```

(A misclassification here is expected some fraction of the time at this
model's ~58% validation accuracy -- not a pipeline bug.)

It then (Milestone 70) generates one brand-new image that was never part of
the dataset, at a resolution (`200x140`) deliberately outside the
`--min-size`/`--max-size` training range, loads it via the same
`ImageFolder._load_image` decode step, applies the *same* `Resize`/
pixel-scale transform used during training, and predicts:

```text
New mixed-resolution image (200x140, true class: circle, never seen during training):
Prediction: circle
```

This demonstrates the milestone's core claim: a caller does not need to
manually pre-resize a new image to the model's trained resolution -- the
same public `Resize` transform used for training data works unchanged at
inference time. Both demos use the same `forge.predict()` used by every
other Forge example -- no separate hand-written inference implementation.

## Integration tests (no dataset download required, generated on the fly)

`tests/test_image_folder_classification_integration.py` (CPU) and
`tests/test_image_folder_classification_cuda_integration.py` (CUDA; skips
cleanly without a working CUDA backend) exercise this exact pipeline against
a smaller/faster generated dataset than the full example run above --
covering dataset generation, model shape, training loss reduction and
above-chance accuracy on real image files, checkpoint save/resume, and
model save/load/predict consistency. Run them with:

```bash
python -m pytest tests/test_image_folder_classification_integration.py tests/test_image_folder_classification_cuda_integration.py
```

Also see `tests/test_image_folder.py` for `forge.data.ImageFolder`'s own
unit tests (discovery, image conversion, dataset behavior, `DataLoader`
integration) -- independent of this example.
