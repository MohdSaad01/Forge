# Forge Data Pipeline

## Target flow

```text
Raw source
   ↓
Dataset
   ↓
Transforms
   ↓
DataLoader
   ↓
Batches
   ↓
Trainer
```

Implemented through `Batches` as of Milestone 5 (`forge.data`). As of
Milestone 6, `forge.training.Trainer` consumes `DataLoader` output to run
the training/evaluation workflow -- see
`docs/architecture/training-engine.md`. Full details:
`docs/architecture/data-system.md`.

## Dataset
`forge.data.Dataset` supports direct indexing (`__getitem__`) and length
(`__len__`) over samples, without assuming a specific file format. A sample
may be a single `Tensor`, a tuple such as `(features, target)`, or another
representation a custom `Dataset` documents for itself.

## Built-in sources
- NumPy/array-backed data: `forge.data.TensorDataset` (M5).
- `forge.data.Subset` / `forge.data.random_split` for deterministic dataset
  splitting (M5); `forge.data.sequential_split` (M74) is the same shape
  without a permutation -- a contiguous, no-RNG-needed split for a dataset
  whose order already carries no meaning, replacing the manual
  `X[:n_train]`/`X[n_train:n_train+n_val]`/`X[n_train+n_val:]` array-slicing
  `examples/regression`/`examples/waveform_classification` used to duplicate
  independently. See `docs/development/m74-data-workflow.md`.
- Directory-based image classification: `forge.data.ImageFolder` (M69) --
  discovers `(image, label)` samples from a `root/class_x/*.jpg` directory
  tree. See `docs/development/m69-image-folder.md` and
  `forge/data/image_folder.py`'s module docstring for the full contract.
- Practical file-backed datasets (beyond `ImageFolder`): not yet implemented.
- Tabular data conveniences: not yet implemented.

## Transforms
`forge.data.Transform`/`Compose` (M5) are composable and focused on
training-relevant preprocessing: `Normalize`, `Reshape`, `Flatten`,
`ToTensor`, `Resize` (M70), `Lambda`. A transform operates on one sample
component (typically the features Tensor), not a whole `(features,
target)` tuple -- `TensorDataset` wires `transform`/`target_transform` to
the feature/target positions separately so a feature transform cannot
silently reach a label. `Resize` (`forge/data/transforms.py`) resizes a
`(C, H, W)` image Tensor to an explicit `(height, width)` via Pillow's
bilinear filter -- added specifically so mixed-resolution `ImageFolder`
datasets can be batched by the existing `DataLoader` unmodified; see
`docs/development/m70-image-preprocessing.md`. A transform's *configuration*
(not the transform object itself, and never `Lambda`) can optionally travel
with a saved model via `forge.save_model(..., preprocessing=...)`/
`forge.load_preprocessing()` (M71) -- see `docs/architecture/persistence.md`'s
**Preprocessing metadata** section; `forge.data`/`forge.data.transforms`
themselves are unchanged by this (the registry/serialization logic lives in
`forge.serialization.transforms`, preserving this document's layering).

## DataLoader
`forge.data.DataLoader` (M5) is responsible for batching, optional
shuffling (deterministic given a supplied `numpy.random.Generator`, or
`forge.random`'s process-global generator otherwise), and synchronous
iteration -- independent of `Module`/`Loss`/`Optimizer`. No multiprocessing
workers or asynchronous prefetching yet.

## Constraints
Avoid recreating pandas/scikit-learn/Pillow ecosystems. External libraries may handle specialized parsing/image decoding while Forge owns the dataset/transform/batching contracts -- `ImageFolder` (M69) is the first concrete instance of this: it depends on Pillow for image decoding only, not for anything dataset/transform/batching-shaped.
