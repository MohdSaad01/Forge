"""MNIST dataset loading for the Forge end-to-end example (Milestone 20).

Forge's core `forge.data` package has no MNIST-specific knowledge -- this
module is the example-side `forge.data.Dataset` implementation that adapts
the standard MNIST IDX file format into Forge `Tensor` samples.

## Source

The four standard MNIST files, in the original IDX format introduced by
Yann LeCun's site (now offline):

```text
train-images-idx3-ubyte.gz   60,000 training images
train-labels-idx1-ubyte.gz   60,000 training labels
t10k-images-idx3-ubyte.gz    10,000 test images
t10k-labels-idx1-ubyte.gz    10,000 test labels
```

`download_mnist()` fetches these from a well-known public mirror of the
original files (`DEFAULT_BASE_URL`, the same mirror `torchvision` uses),
over plain `urllib` -- no third-party ML framework dependency. Downloading
never happens implicitly: `MNISTDataset(..., download=False)` (the default)
raises `DataError` with the expected file location if the files are not
already present, and a caller must pass `download=True` or call
`download_mnist()` directly to fetch them.

## Expected location

`root/train-images-idx3-ubyte(.gz)`, `root/train-labels-idx1-ubyte(.gz)`,
etc. -- gzip-compressed or already-decompressed files are both accepted (the
gzip magic bytes are sniffed directly, not inferred from the file name).

## Format

Each IDX file is a big-endian binary header (a magic number identifying
images-vs-labels, then the array's dimension sizes as `uint32`s) followed by
raw `uint8` array data -- see `_read_idx_images`/`_read_idx_labels` below.

## Preprocessing

`MNISTDataset.__getitem__` returns `(image, label)`:

- `image`: a `forge.Tensor` of shape `(1, 28, 28)`, `float32`, raw pixel
  values in `[0, 255]` -- scaling/normalization is left to the caller's
  `transform` (see `examples/mnist/train.py`, which applies the conventional
  `/255` scale then `Normalize(mean=0.1307, std=0.3081)` via
  `forge.data.transforms`), matching the milestone brief's "use the existing
  transformation infrastructure" rather than baking preprocessing into the
  dataset itself.
- `label`: a scalar (`shape == ()`) `int64` `forge.Tensor` holding the class
  index `0-9`; `DataLoader` batches these into a `(batch_size,)` tensor, the
  shape `CrossEntropyLoss` expects.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from forge import Tensor
from forge.data import Dataset
from forge.exceptions import DataError

DEFAULT_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049

_FILES_BY_SPLIT = {
    True: ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    False: ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def _open_maybe_gzip(path: Path):
    with open(path, "rb") as f:
        is_gzip = f.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if is_gzip else open(path, "rb")


def _read_idx_images(path: Path) -> np.ndarray:
    """Parse an IDX3 image file into a `(N, rows, cols)` `uint8` array."""
    with _open_maybe_gzip(path) as f:
        data = f.read()
    if len(data) < 16:
        raise DataError(f"'{path}' is too short to be a valid MNIST IDX image file.")
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != _IMAGE_MAGIC:
        raise DataError(
            f"'{path}' is not a valid MNIST image file: expected IDX magic number "
            f"{_IMAGE_MAGIC}, got {magic}."
        )
    pixels = np.frombuffer(data, dtype=np.uint8, offset=16)
    expected = n * rows * cols
    if pixels.size != expected:
        raise DataError(
            f"'{path}' is truncated: header declares {n} images of {rows}x{cols}, "
            f"expected {expected} pixel bytes, found {pixels.size}."
        )
    return pixels.reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Parse an IDX1 label file into a `(N,)` `uint8` array."""
    with _open_maybe_gzip(path) as f:
        data = f.read()
    if len(data) < 8:
        raise DataError(f"'{path}' is too short to be a valid MNIST IDX label file.")
    magic, n = struct.unpack(">II", data[:8])
    if magic != _LABEL_MAGIC:
        raise DataError(
            f"'{path}' is not a valid MNIST label file: expected IDX magic number "
            f"{_LABEL_MAGIC}, got {magic}."
        )
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    if labels.size != n:
        raise DataError(
            f"'{path}' is truncated: header declares {n} labels, found {labels.size}."
        )
    return labels


def download_mnist(root: "str | Path", base_url: str = DEFAULT_BASE_URL) -> None:
    """Explicitly fetch the four standard MNIST IDX files into `root`.

    Never called implicitly -- `MNISTDataset` only downloads when constructed
    with `download=True`, and this function itself is always an explicit,
    caller-initiated action. A file already present under `root` (compressed
    or not) is left untouched rather than re-fetched. Uses `urllib.request`
    only -- no third-party HTTP or ML-framework dependency.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    names: "set[str]" = set()
    for images_name, labels_name in _FILES_BY_SPLIT.values():
        names.add(images_name)
        names.add(labels_name)

    for name in sorted(names):
        if (root / name).is_file() or (root / f"{name}.gz").is_file():
            continue
        url = base_url.rstrip("/") + "/" + f"{name}.gz"
        dest = root / f"{name}.gz"
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)


class MNISTDataset(Dataset):
    """MNIST digit classification, conforming to `forge.data.Dataset`.

    ```python
    train_ds = MNISTDataset("data/mnist", train=True, download=True, transform=preprocess)
    test_ds = MNISTDataset("data/mnist", train=False, transform=preprocess)
    ```

    `transform` is applied to the `(1, 28, 28)` float32 image sample;
    `target_transform` (rarely needed) to the integer label sample -- the
    same feature/target separation `forge.data.TensorDataset` already
    establishes. See the module docstring for file format/location/source.
    """

    def __init__(
        self,
        root: "str | Path",
        train: bool = True,
        transform: "Any | None" = None,
        target_transform: "Any | None" = None,
        download: bool = False,
    ):
        self.root = Path(root)
        self.train = bool(train)
        self.transform = transform
        self.target_transform = target_transform

        images_name, labels_name = _FILES_BY_SPLIT[self.train]
        images_path = self._resolve(images_name)
        labels_path = self._resolve(labels_name)

        if (images_path is None or labels_path is None) and download:
            download_mnist(self.root)
            images_path = self._resolve(images_name)
            labels_path = self._resolve(labels_name)

        if images_path is None or labels_path is None:
            split = "training" if self.train else "test"
            raise DataError(
                f"MNIST {split} files not found under '{self.root}'. Expected "
                f"'{images_name}(.gz)' and '{labels_name}(.gz)'. Construct with "
                "download=True, call examples.mnist.dataset.download_mnist(root) "
                f"first, or place the files there manually (source: {DEFAULT_BASE_URL})."
            )

        images = _read_idx_images(images_path)
        labels = _read_idx_labels(labels_path)
        if images.shape[0] != labels.shape[0]:
            raise DataError(
                f"MNIST image count ({images.shape[0]}) does not match label count "
                f"({labels.shape[0]}) under '{self.root}'."
            )

        self.images = images
        self.labels = labels

    def _resolve(self, name: str) -> "Path | None":
        gz_path = self.root / f"{name}.gz"
        raw_path = self.root / name
        if gz_path.is_file():
            return gz_path
        if raw_path.is_file():
            return raw_path
        return None

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> "tuple[Tensor, Tensor]":
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise DataError(f"MNISTDataset index must be an int, got {type(index).__name__}.")
        idx = int(index)
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise DataError(f"Index {index} out of range for MNISTDataset of size {len(self)}.")

        image = Tensor(self.images[idx].astype(np.float32).reshape(1, 28, 28))
        if self.transform is not None:
            image = self.transform(image)

        label = Tensor(int(self.labels[idx]), dtype="int64")
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __repr__(self) -> str:
        split = "train" if self.train else "test"
        return f"MNISTDataset(root={str(self.root)!r}, split={split!r}, size={len(self)})"


__all__ = ["MNISTDataset", "download_mnist", "DEFAULT_BASE_URL"]
