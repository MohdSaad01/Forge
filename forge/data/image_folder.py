"""`ImageFolder`: a directory-per-class image classification dataset (Milestone 69).

Forge's data-pipeline documentation (`docs/architecture/data-pipeline.md`)
had explicitly deferred "image dataset conveniences" and permitted an
external library for image *decoding* specifically ("External libraries may
handle specialized parsing/image decoding while Forge owns the
dataset/transform/batching contracts"). `ImageFolder` is that convenience:
a `forge.data.Dataset` that discovers labeled images from an ordinary
directory tree

```text
root/
    class_a/
        image1.jpg
        image2.png
    class_b/
        image3.jpg
```

and turns them into `(image, label)` `Tensor` samples through the exact
same `Dataset` -> transform -> `DataLoader` -> `Trainer` pipeline every
other Forge dataset already goes through -- no parallel loading machinery.

## Directory semantics

- Each immediate subdirectory of `root` is one class. Classes are sorted by
  name and receive deterministic integer indices in that order (`classes`,
  `class_to_idx`) -- construction never depends on filesystem iteration
  order.
- Only files directly inside a class directory are scanned; a class
  directory's own subdirectories (if any) are **not** recursed into. This is
  a deliberate, documented choice, not an oversight -- "one class = one
  directory of image files" is the entire discovery contract.
- A file counts as a sample only if its suffix (case-insensitively) is one
  of `IMAGE_EXTENSIONS` (overridable via `extensions=`); every other file is
  silently skipped, never treated as a sample. Files within one class are
  sorted by name, so sample order is fully deterministic for a given
  directory tree.
- `root` must exist and be a directory, and must contain at least one class
  subdirectory with at least one supported image file -- both are `DataError`
  otherwise (mirroring `TensorDataset`'s "at least one sample" requirement,
  since a zero-sample dataset would only surface as a confusing failure
  later in `DataLoader`/`Trainer`). An *individual* class directory with zero
  matching files is allowed (it still becomes a valid, indexable class in
  `classes`/`class_to_idx`; it just contributes no samples) -- directory
  structure alone establishes the class list, independent of the class's
  current image count.

## Image representation

Every image is decoded (via Pillow) and converted to `Tensor` as:

- shape `(3, H, W)` -- channels-first, matching every convolutional model in
  Forge (`Conv2d` expects `(N, C, H, W)`).
- dtype `float32`, raw pixel values in `[0, 255]` (scaling/normalization is
  left to `transform=`, exactly like `examples/mnist/dataset.py`'s
  `MNISTDataset`).
- always 3 channels, RGB order: grayscale (`L`)/palette/CMYK/etc. images are
  converted to RGB (grayscale replicated across channels); RGBA images have
  their alpha channel **discarded** (no compositing against a background) --
  via Pillow's own `Image.convert("RGB")`. This keeps every sample the same
  shape regardless of the source files' original modes, which `DataLoader`
  batching requires.

`ImageFolder` does not resize images -- every file under `root` must already
share the same `(H, W)`, or `transform=` must normalize that (Forge has no
`Resize` transform yet; see the module's own limitations note in
`docs/development/m69-image-folder.md`). Images are decoded fresh on every
`__getitem__` call (no caching/preloading) -- the simplest behavior, and
consistent with "measure before optimizing" (`docs/architecture/
data-pipeline.md`'s scope); revisit only if a real workload measures this as
a bottleneck.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image, UnidentifiedImageError

from ..exceptions import DataError
from ..tensor.tensor import Tensor
from .dataset import Dataset, _normalize_index

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp")


class ImageFolder(Dataset):
    """A directory-per-class image classification dataset.

    ```python
    dataset = ImageFolder("data/cats-dogs", transform=preprocess)
    image, label = dataset[0]        # image: Tensor(3, H, W) float32 [0, 255]
    dataset.classes                  # ['cats', 'dogs'] -- sorted, deterministic
    dataset.class_to_idx              # {'cats': 0, 'dogs': 1}
    dataset.samples                   # [(Path(...), 0), (Path(...), 0), ...]
    ```

    See the module docstring for the full directory/image-representation
    contract. `transform`/`target_transform` are applied exactly like
    `TensorDataset`/`MNISTDataset`: `transform` to the image `Tensor`,
    `target_transform` to the integer-label `Tensor`.
    """

    def __init__(
        self,
        root: "str | Path",
        transform: "Any | None" = None,
        target_transform: "Any | None" = None,
        extensions: "tuple[str, ...]" = IMAGE_EXTENSIONS,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise DataError(
                f"ImageFolder root '{self.root}' does not exist or is not a directory."
            )

        self.transform = transform
        self.target_transform = target_transform
        self.extensions = tuple(e.lower() for e in extensions)

        class_dirs = sorted(
            (p for p in self.root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
        if not class_dirs:
            raise DataError(
                f"ImageFolder root '{self.root}' contains no class subdirectories. "
                "Expected a structure like root/class_a/, root/class_b/, ..."
            )

        self.classes = [p.name for p in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        samples: "list[tuple[Path, int]]" = []
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            files = sorted(
                (
                    f
                    for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.extensions
                ),
                key=lambda f: f.name,
            )
            for f in files:
                samples.append((f, class_idx))

        if not samples:
            raise DataError(
                f"ImageFolder root '{self.root}' contains no images with a supported "
                f"extension {self.extensions} under any class subdirectory."
            )

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image(path: Path) -> Tensor:
        """Decode `path` into a `(3, H, W)` float32 `Tensor`, raw `[0, 255]` range.

        The single, shared image-decode step -- `__getitem__` calls this
        directly, and it is the right place to reuse (not duplicate) if a
        caller ever needs to preprocess one arbitrary image file the exact
        same way `ImageFolder` does (e.g. single-image inference on a file
        that isn't part of the dataset).
        """
        try:
            with Image.open(path) as img:
                array = np.array(img.convert("RGB"), dtype=np.uint8)
        except (OSError, UnidentifiedImageError) as exc:
            raise DataError(f"ImageFolder could not read image '{path}': {exc}") from exc

        chw = np.ascontiguousarray(array.transpose(2, 0, 1).astype(np.float32))
        return Tensor(chw)

    def __getitem__(self, index: int) -> "tuple[Tensor, Tensor]":
        idx = _normalize_index(index, len(self), "ImageFolder")
        path, class_idx = self.samples[idx]

        image = self._load_image(path)
        if self.transform is not None:
            image = self.transform(image)

        label = Tensor(int(class_idx), dtype="int64")
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __repr__(self) -> str:
        return f"ImageFolder(root={str(self.root)!r}, classes={self.classes}, size={len(self)})"


__all__ = ["ImageFolder", "IMAGE_EXTENSIONS"]
