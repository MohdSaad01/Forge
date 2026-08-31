"""Milestone 5 manual verification: Dataset -> Transforms -> DataLoader -> Model.

Run directly: `python examples/data_pipeline_demo.py`

Demonstrates the target flow from `docs/architecture/data-pipeline.md`:
    Raw data -> Dataset -> Transforms -> DataLoader -> Batches -> Model
No training engine/Trainer is involved -- this only proves that batches
produced by `forge.data` feed directly into the existing `forge.nn` stack.
"""

from __future__ import annotations

import numpy as np

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.data.transforms import Normalize
from forge.nn import Linear


def main() -> None:
    forge.random.seed(0)
    data_rng = np.random.default_rng(0)

    # 1. Raw feature/target data: y = 2*x0 - x1 + noise.
    n_samples = 20
    X = data_rng.uniform(-10, 10, size=(n_samples, 2))
    y = (2 * X[:, 0] - X[:, 1]).reshape(-1, 1)

    # 2. Dataset, with a per-feature normalization transform.
    dataset = TensorDataset(
        Tensor(X),
        Tensor(y),
        transform=Normalize(mean=X.mean(axis=0), std=X.std(axis=0)),
    )
    print(f"Dataset size: {len(dataset)}")
    sample_x, sample_y = dataset[0]
    print(f"Sample 0: features shape={sample_x.shape}, target shape={sample_y.shape}")

    # 3. DataLoader: batching + shuffling.
    loader = DataLoader(dataset, batch_size=6, shuffle=False, drop_last=False)
    print(f"\nUnshuffled batches ({len(loader)} total):")
    for i, (batch_x, batch_y) in enumerate(loader):
        print(f"  batch {i}: batch_x.shape={batch_x.shape}, batch_y.shape={batch_y.shape}")

    # 4. Deterministic shuffling: same generator seed reproduces the same order.
    order_a = [
        b[0].numpy().tolist()
        for b in DataLoader(dataset, batch_size=6, shuffle=True, generator=np.random.default_rng(42))
    ]
    order_b = [
        b[0].numpy().tolist()
        for b in DataLoader(dataset, batch_size=6, shuffle=True, generator=np.random.default_rng(42))
    ]
    print(f"\nDeterministic shuffle (same seed) reproducible: {order_a == order_b}")

    # 5. Feed a batch through an existing Forge model.
    model = Linear(2, 1)
    batch_x, batch_y = next(iter(loader))
    prediction = model(batch_x)
    print(f"\nLinear(2, 1) forward on a batch: input {batch_x.shape} -> output {prediction.shape}")


if __name__ == "__main__":
    main()
