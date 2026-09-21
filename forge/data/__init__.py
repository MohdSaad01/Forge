"""Forge's dataset/transform/batching foundation (Milestone 5).

```text
Raw data -> Dataset -> Transforms -> DataLoader -> Batches -> Model / Trainer
```

Independent of `Module`/`Loss`/`Optimizer`/a training engine -- `forge.data`
only produces model-ready `Tensor` batches; it does not run a forward pass or
know about gradients. See `docs/architecture/data-system.md`.
"""

from .csv_reader import load_csv, load_csv_features
from .dataloader import DataLoader
from .dataset import Dataset, Subset, TensorDataset, random_split, sequential_split
from .image_folder import IMAGE_EXTENSIONS, ImageFolder, save_image
from .prefetch import CUDAPrefetchLoader
from .target_transform import StandardizeTarget
from .transforms import Compose, Flatten, Lambda, Normalize, ReplaceValue, Reshape, Resize, ToTensor, Transform

__all__ = [
    "Dataset",
    "TensorDataset",
    "Subset",
    "random_split",
    "sequential_split",
    "ImageFolder",
    "IMAGE_EXTENSIONS",
    "save_image",
    "load_csv",
    "load_csv_features",
    "DataLoader",
    "CUDAPrefetchLoader",
    "Transform",
    "Compose",
    "ToTensor",
    "Normalize",
    "ReplaceValue",
    "Reshape",
    "Flatten",
    "Resize",
    "Lambda",
    "StandardizeTarget",
]
