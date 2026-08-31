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

## Dataset
The dataset abstraction should support direct indexing/iteration over samples without assuming a specific file format.

## Built-in sources
- NumPy/array-backed data.
- Practical file-backed datasets as needed.
- Image dataset conveniences later.
- Tabular data conveniences later.

## Transforms
Transforms should be composable and focused on training-relevant preprocessing such as normalization, resizing, encoding, and conversion to tensors.

## DataLoader
Responsible for batching, optional shuffling, and iteration. Keep it independent of the model.

## Constraints
Avoid recreating pandas/scikit-learn/Pillow ecosystems. External libraries may handle specialized parsing/image decoding while Forge owns the dataset/transform/batching contracts.
