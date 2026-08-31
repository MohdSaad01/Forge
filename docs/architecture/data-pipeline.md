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
  splitting (M5).
- Practical file-backed datasets: not yet implemented.
- Image dataset conveniences: not yet implemented.
- Tabular data conveniences: not yet implemented.

## Transforms
`forge.data.Transform`/`Compose` (M5) are composable and focused on
training-relevant preprocessing: `Normalize`, `Reshape`, `Flatten`,
`ToTensor`, `Lambda`. A transform operates on one sample component
(typically the features Tensor), not a whole `(features, target)` tuple --
`TensorDataset` wires `transform`/`target_transform` to the feature/target
positions separately so a feature transform cannot silently reach a label.

## DataLoader
`forge.data.DataLoader` (M5) is responsible for batching, optional
shuffling (deterministic given a supplied `numpy.random.Generator`, or
`forge.random`'s process-global generator otherwise), and synchronous
iteration -- independent of `Module`/`Loss`/`Optimizer`. No multiprocessing
workers or asynchronous prefetching yet.

## Constraints
Avoid recreating pandas/scikit-learn/Pillow ecosystems. External libraries may handle specialized parsing/image decoding while Forge owns the dataset/transform/batching contracts.
