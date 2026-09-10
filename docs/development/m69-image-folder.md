# M69 — Directory-Based Image Dataset: First Real-World Data Ingestion Layer

## 1. Objective

Implement a small, production-quality `ImageFolder`-style dataset in
`forge.data` that discovers labeled images from a directory-per-class tree
(`root/class_a/*.jpg`, `root/class_b/*.jpg`, ...) and exposes
`(image_tensor, class_index)` samples through Forge's existing
`Dataset`/`DataLoader` pipeline -- a real framework capability, not an
example-only helper -- then prove it with a genuine end-to-end consumer
(generate a dataset -> train -> save/load -> `forge.predict()` -> map the
predicted class back to a name).

## 2. Repository Findings

- `forge/data/dataset.py`'s `Dataset` is a minimal two-method protocol
  (`__len__`/`__getitem__`, both raising `DataError` unimplemented); a
  sample may be any representation a custom `Dataset` documents for itself.
  `TensorDataset` established the `transform`/`target_transform` separation
  (feature transform never reaches the label) that `ImageFolder` reuses
  exactly.
- `forge/data/dataloader.py`'s `_stack` batches same-shape/same-dtype
  Tensor components via `np.stack` -- meaning every sample `ImageFolder`
  yields for one dataset must share `(H, W)`; there is no resize step
  anywhere in the pipeline.
- The only prior "image loading" precedent, `examples/mnist/dataset.py`,
  parses MNIST's IDX binary format directly (raw bytes -> NumPy) -- **not**
  an image-file decoder. It does establish the "raw `[0, 255]` range,
  scaling left to the caller's `transform`" convention `ImageFolder` also
  follows.
- `docs/architecture/data-pipeline.md`'s own "Constraints" section had
  already explicitly anticipated this exact milestone: *"External libraries
  may handle specialized parsing/image decoding while Forge owns the
  dataset/transform/batching contracts."* This settled the "write an image
  decoder from scratch vs. use a library" question before implementation
  started -- confirmed as the smallest correct choice per Section 7's own
  guardrail against hand-rolled decoders.
- Pillow (12.1.1) was already present in the dev environment but **not** a
  declared `pyproject.toml` dependency anywhere in the codebase; no
  optional-dependency/lazy-import convention exists for a *required*
  capability like this (the one precedent, `forge/cli/benchmark.py`, lazily
  imports an *internal, non-packaged* module, not a third-party library).
- `forge.exceptions.DataError` already exactly covers "invalid dataset
  configuration or usage" -- no new exception type was needed.
- `forge/data/__init__.py`/`forge/__init__.py` are flat re-export lists;
  adding `ImageFolder` followed the exact pattern every prior `forge.data`
  addition (`TensorDataset`, `Subset`, `random_split`) already used.
- No `Resize`/augmentation transform exists in `forge/data/transforms.py`
  (`Normalize`/`Reshape`/`Flatten`/`Lambda`/`ToTensor`/`Compose` only) --
  confirmed a genuine, pre-existing gap, not something to silently work
  around inside `ImageFolder`.

## 3. Existing Capabilities Reused

`Dataset` base class and its `transform`/`target_transform` contract,
`DataLoader`/`_collate`/`_stack` batching (unmodified), `forge.data.
random_split` (used by the example to carve train/test from one directory,
rather than inventing a new split convention), `forge.exceptions.DataError`,
`forge.nn.BatchNorm2d` (M53) and `forge.nn.Dropout` (M4) in the example
model, `forge.training.Trainer`/`Accuracy`, `forge.save_model`/`load_model`/
`save_checkpoint`/`load_checkpoint`, and `forge.predict()` (M68) for the
inference demonstration. Zero changes to `Dataset`, `DataLoader`, `Trainer`,
autograd, Tensor internals, or any CUDA kernel.

## 4. Missing Capability Identified

No file-backed image dataset of any kind existed; every prior example
either parsed a bundled binary format (MNIST) or generated data directly
into in-memory NumPy arrays (`regression`, `waveform_classification`,
`segmentation`, `long_range_recall`). "An ordinary developer hands Forge a
directory of image files" had no path through the framework at all.

## 5. Design

`ImageFolder` lives entirely in `forge/data/image_folder.py`, a new module
alongside `dataset.py`/`dataloader.py`/`transforms.py` -- no existing file
touched except the package `__init__.py` re-export lists. It is a thin
`Dataset` subclass: construction does directory discovery and validation
once (eager), `__getitem__` decodes one image file per call (lazy, no
caching -- see Section 16, Performance). The single image-decode step is
factored into a reusable `@staticmethod _load_image(path)` specifically so
nothing else (a future single-image-inference helper, e.g.) would need to
duplicate it -- `__getitem__` is its only caller in this milestone, but the
factoring was already necessary internally and costs nothing extra.

## 6. API

```python
from forge.data import ImageFolder, IMAGE_EXTENSIONS

dataset = ImageFolder("data/cats-dogs", transform=preprocess)
image, label = dataset[0]     # image: Tensor(3, H, W) float32, [0, 255]
dataset.classes                # ['cats', 'dogs'] -- sorted, deterministic
dataset.class_to_idx           # {'cats': 0, 'dogs': 1}
dataset.samples                 # [(Path(...), 0), ...] -- deterministic order
```

`transform`/`target_transform` behave exactly like `TensorDataset`'s:
`transform` applied to the image `Tensor`, `target_transform` to the
integer-label `Tensor`, configured separately (never one callable trusted
to leave the label alone). `extensions=` overrides the default
`IMAGE_EXTENSIONS` tuple. No metadata beyond `classes`/`class_to_idx`/
`samples` was added -- each is directly used by the example (`classes` for
the final class-name mapping, `samples` implicitly via `__getitem__`); no
metadata was added merely because another framework exposes it.

## 7. Directory Semantics

- Each **immediate** subdirectory of `root` is one class; classes are
  sorted by name for deterministic indices (`classes`/`class_to_idx`) --
  construction never depends on filesystem iteration order.
- Only files **directly inside** a class directory are scanned; a class
  directory's own subdirectories are **not** recursed into. Deliberately
  rejected, not an oversight -- "one class = one flat directory of image
  files" is the entire discovery contract, verified by
  `test_nested_subdirectory_files_are_not_discovered`.
- A file counts as a sample only if its suffix (case-insensitive) is one of
  `IMAGE_EXTENSIONS` (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`, `.pgm`,
  `.tif`, `.tiff`, `.webp`) or a caller's `extensions=` override; every
  other file is silently skipped, never treated as a sample. Files within
  one class are sorted by name -- fully deterministic sample order.
- `root` must exist and be a directory (`DataError` otherwise); must
  contain at least one class subdirectory (`DataError` otherwise); the
  **total** sample count across all classes must be at least one
  (`DataError` otherwise, mirroring `TensorDataset`'s "at least one sample"
  rule -- a zero-sample dataset would otherwise surface as a confusing
  failure later in `DataLoader`/`Trainer`).
- An **individual** class directory with zero matching files is valid: it
  still becomes a real, indexable class in `classes`/`class_to_idx`, just
  contributing zero samples -- directory structure alone establishes the
  class list, independent of a class's current image count.

## 8. Image Representation

Every image decodes (via Pillow) to a `(3, H, W)` float32 `Tensor`, raw
pixel values in `[0, 255]`, always 3 channels in RGB order: grayscale
(`L`)/palette/CMYK/etc. converted to RGB (grayscale replicated across
channels), RGBA alpha **discarded** (no compositing against a background),
via Pillow's own `Image.convert("RGB")`. This keeps every sample the same
channel count regardless of source file mode, which `DataLoader` batching
requires. `ImageFolder` does **not** resize -- every file under one `root`
must already share `(H, W)`, or `transform=` must normalize that (no
`Resize` transform exists yet; see Section 18).

## 9. Transform Integration

`transform`/`target_transform` are ordinary constructor arguments applied
inside `__getitem__`, exactly like `TensorDataset`/`MNISTDataset` --
`Compose`/`Lambda`/`Normalize`/etc. all work unmodified (verified by
`test_transform_applied_to_image_only`/`test_target_transform_applied_to_
label_only`). No new transform type was added; no augmentation machinery
was added.

## 10. Example Workload

`examples/image_folder_classification/`: `generate_dataset.py` renders a
synthetic `circle`/`square`/`triangle` dataset (Pillow `ImageDraw` + NumPy;
varying position, scale, rotation, per-image foreground color, and
per-pixel Gaussian noise -- no download, no MNIST reuse, per the brief's
explicit instruction). `model.py` builds a small CNN. `train.py` wires
`ImageFolder -> random_split -> DataLoader -> Trainer -> CrossEntropyLoss/
Adam -> checkpoint/model save -> reload -> forge.predict()`.

## 11. Training Results

**Honest finding, not smoothed over.** The first working version (plain
`Conv2d`/`ReLU`/`MaxPool2d`, independently-random background *and*
foreground colors per image) trained -- loss fell steadily -- but did
**not** generalize: validation accuracy stayed pinned at the 33% chance
baseline even after 20 epochs, because color polarity (which of the two
random colors was brighter) varied unpredictably per image, and a 2-conv
network with no normalization could not learn a polarity-invariant edge
detector from a few hundred samples. Diagnosed by first confirming the
pipeline itself had no bug (the model *could* reach 100% train accuracy on
a 30-sample dataset it was allowed to memorize -- ruling out a data/label
mismatch), then fixed by two changes, in this order:

1. **Generator fix**: kept the shape color genuinely random per image but
   constrained the background to always be light and the foreground to
   always be meaningfully darker (`_MIN_LUMINANCE_GAP = 90`) -- removing
   the polarity-invariance requirement while keeping real per-image color
   variation (color still can't be memorized as a shortcut).
2. **Model fix**: added `BatchNorm2d` after each `Conv2d` (closed most of
   the remaining chance-baseline gap) and `Dropout(0.3)` before the final
   `Linear` (closed most of the resulting train/validation overfitting gap
   -- train accuracy alone reached ~99% within a few epochs without it,
   with validation accuracy far behind). Both are pre-existing `forge.nn`
   layers; zero new framework code.

Final measured result (`--seed 0`, defaults: 300 images/class = 900 total,
80/20 split, 25 epochs, `lr=4e-4`, batch size 32):

| Epoch | Train loss | Train acc | Val acc |
|------:|-----------:|----------:|--------:|
| 1     | 1.098      | ~34%      | ~34%    |
| 25    | 0.160      | ~97%      | 70.00%  |

Final test evaluation: loss 0.689, accuracy 70.00% -- well above the
3-class 33% chance baseline. 68.2s total on the reference i5-7200U CPU
(~264 samples/sec). A visible train/validation accuracy gap remains
(~97% vs. ~70%) -- reported honestly as a real property of this dataset
size/model size, not tuned away further (per the milestone's "do not
optimize prematurely" scope).

## 12. Inference Results

`train.py` takes one held-out test-split image, runs `forge.predict()`,
and maps the predicted index back through `ImageFolder.classes`:

```text
Inference demo (test sample 0, true class: square):
Prediction: square
```

Uses the same `forge.predict()` every other Forge example uses -- no
separate hand-written inference path.

## 13. Persistence Results

`save_model()` -> `load_model()` -> `forge.predict()` on the same query
image reproduces the pre-save prediction to `atol=1e-5`/`1e-6` (CPU and
CUDA), verified both in `train.py`'s own run-time assertion and in
`tests/test_image_folder_classification_integration.py::
test_model_persistence_and_predict_map_back_to_class_name`. Checkpoint
save/resume (model + Adam state + epoch/global_step) verified the same way
every prior example verifies it.

## 14. CPU/CUDA Verification

CPU: full example run above, plus `tests/test_image_folder.py` (29 tests,
no CUDA dependency) and `tests/test_image_folder_classification_
integration.py` (5 tests). CUDA (reference GeForce 940MX, CC 5.0, CUDA
12.6): full example run reached 71.11% validation accuracy (loss curve
1.090 -> 0.159, closely tracking CPU's 1.098 -> 0.160) in 23.1s (~3x CPU
throughput, ~780 samples/sec) -- `forge.data`/`ImageFolder` itself always
produces plain CPU Tensors; `Trainer` is the layer that moves each batch to
CUDA, exactly as documented for every other Forge dataset
(`docs/architecture/data-system.md`'s "Device behavior" section, unchanged).
`tests/test_image_folder_classification_cuda_integration.py` (4 tests,
skips cleanly without CUDA) additionally verifies CUDA residency of
parameters/gradients/Adam state and checkpoint/persistence round trips on
CUDA -- all passing on the reference hardware.

## 15. Tests

38 new tests total:

- `tests/test_image_folder.py` (29, CPU): class discovery and deterministic
  ordering (classes and files), extension filtering (case-insensitive,
  custom `extensions=` override), unsupported-file skipping, empty-class-
  directory and nested-subdirectory semantics, missing-root/no-class-dirs/
  zero-total-samples error paths, multi-class discovery, RGB/grayscale/
  RGBA image-conversion shape/dtype/value-range/channel checks, a corrupt-
  file `DataError` path, `len`/indexing (negative and out-of-range and
  non-int), `transform`/`target_transform` separation, `repr`, the exported
  `IMAGE_EXTENSIONS` constant, and `DataLoader` integration (batching,
  reproducible shuffling with an explicit generator, feeding a real
  `Conv2d`-based model).
- `tests/test_image_folder_classification_integration.py` (5, CPU):
  dataset-generation validity, model forward shape, full pipeline training
  (loss reduction + above-chance accuracy + parameter-change check) against
  **real PNG files on disk**, checkpoint save/resume, and model persistence
  with class-index-to-name mapping.
- `tests/test_image_folder_classification_cuda_integration.py` (4, CUDA
  hardware-gated): full pipeline training on CUDA, CUDA residency of
  parameters/gradients/Adam state, checkpoint save/resume on CUDA, model
  persistence on CUDA.

Full suite: **2,113 tests collected, 2,112 passed, 1 failed**
(`test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_
pinned_memory`, the same pre-existing allocator-measurement flake
documented since M63 -- reproduced passing cleanly in isolation
immediately afterward; not an M69 regression). Hardware-verified on the
reference 940MX (CUDA 12.6, driver 582.53).

## 16. Performance Observations

Images decode fresh on every `__getitem__`, no caching/preloading -- the
simplest behavior consistent with the existing architecture (`DataLoader`
has no caching layer for any dataset). No measurement in this milestone
showed repeated decoding as a bottleneck at this dataset's scale (900
images, 32x32); revisit only if a future real workload measures otherwise.
CPU training throughput (~264 samples/sec) was noticeably lower than
MNIST's comparable architecture (~1000-1200 samples/sec) -- expected, since
this model is larger (`BatchNorm2d` x2, more channels) and every sample
goes through a real Pillow file decode per epoch, unlike MNIST's fully
preloaded in-memory NumPy array. Not optimized further per the milestone's
explicit "do not optimize prematurely" scope.

## 17. Architecture Impact

Zero changes to `Dataset`, `DataLoader`, `Trainer`, autograd, Tensor
internals, optimizers, or any CUDA kernel. One new file
(`forge/data/image_folder.py`), two `__init__.py` re-export-list edits, one
new packaging dependency (`Pillow>=10.0`), and one new `examples/`
directory. `forge.data` remains CPU-only exactly as before -- `ImageFolder`
introduces no device awareness of its own.

## 18. Limitations

- No `Resize` transform -- every image under one `ImageFolder` root must
  already share `(H, W)`, or a caller-supplied `transform=` must normalize
  it. A genuinely mixed-size real-world dataset (unlike this milestone's
  uniformly-generated 32x32 synthetic one) cannot be loaded as-is today.
- No file-level caching/preloading -- acceptable at this milestone's scale;
  unmeasured at a much larger (e.g. tens-of-thousands-of-images) scale.
- No multi-label support (one label per sample only, inherited from
  `Dataset`'s general shape, not `ImageFolder`-specific).
- No streaming/lazy directory listing for very large directories --
  `__init__` eagerly walks the full tree once.

## 19. Deferred Capabilities

- A `Resize` transform (genuine, identified gap -- not added here because
  no real consumer in this milestone needed it; the synthetic dataset is
  uniformly sized by construction).
- Data augmentation transforms (random crop/flip/rotation) -- explicitly
  out of scope per the milestone's own guardrails.
- A generic model-serving API, distributed/multi-GPU training, automatic
  architecture selection -- all explicitly out of scope, unrelated to this
  milestone's findings.

## 20. Future Dependency Identified

If a future milestone needs to train on images of genuinely mixed
dimensions (a real downloaded dataset, not a uniformly-generated synthetic
one), a `Resize` transform becomes a real blocker, not a speculative one --
this is the clearest concrete next dependency this milestone's own example
exposed but did not need to solve.

## 21. Practical Product Impact

Before this milestone, "give Forge a directory of labeled images" had no
path through the framework at all -- every image-classification example
either parsed a bundled binary format or synthesized data directly into
arrays. After this milestone, an ordinary developer can point
`ImageFolder` at `root/class_a/`, `root/class_b/`, ... and reach a trained,
saved, reloadable, `forge.predict()`-served model without writing any
image-decoding or batching code themselves -- verified end-to-end against
real files on disk (not mocked), including a training run that legitimately
struggled and was legitimately fixed with existing framework layers
(`BatchNorm2d`, `Dropout`), not just a toy dataset that happened to work on
the first attempt.

## 22. Suggested Next Direction

A `forge.data.Resize` (or equivalently-scoped) transform, once a real
(non-synthetic) mixed-size image dataset creates concrete pressure for it
-- following this codebase's own established discipline of building a
primitive only once a genuine consumer needs it, not speculatively.
