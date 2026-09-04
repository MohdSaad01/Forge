"""Forge's dataset/transform/batching foundation (Milestone 5).

```text
Raw data -> Dataset -> Transforms -> DataLoader -> Batches -> Model / Trainer
```

Independent of `Module`/`Loss`/`Optimizer`/a training engine -- `forge.data`
only produces model-ready `Tensor` batches; it does not run a forward pass or
know about gradients. See `docs/architecture/data-system.md`.
"""

from .dataloader import DataLoader
from .dataset import Dataset, Subset, TensorDataset, random_split
from .prefetch import CUDAPrefetchLoader
from .transforms import Compose, Flatten, Lambda, Normalize, Reshape, ToTensor, Transform

__all__ = [
    "Dataset",
    "TensorDataset",
    "Subset",
    "random_split",
    "DataLoader",
    "CUDAPrefetchLoader",
    "Transform",
    "Compose",
    "ToTensor",
    "Normalize",
    "Reshape",
    "Flatten",
    "Lambda",
]
