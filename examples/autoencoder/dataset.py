"""MNIST reconstruction pairs for the convolutional autoencoder example (Milestone 63).

`AutoencoderDataset` wraps `examples.mnist.dataset.MNISTDataset` -- reusing
its already-tested IDX parsing/download code rather than duplicating it --
and discards the label, returning `(image, image)` pairs instead of
`(image, label)`: reconstruction training compares a model's output against
its own (transformed) input, so the "target" `Trainer.fit()` expects is
simply the input itself. This is the one piece of example-specific glue
needed; nothing about MNIST file parsing/downloading is reimplemented here.

`root` defaults to `examples/mnist/data` -- both examples need the exact
same raw MNIST files, so a user who has already run `examples/mnist` does
not need a second ~11MB download; pass a different `--data-root` (see
`train.py`) if a fully independent copy is wanted.

Labels are not used for training, but `label_at()` exposes them for
`train.py`'s qualitative latent-space evaluation (do images of the same
digit end up with nearby latent codes?) -- a legitimate use of the labels
Forge's own `Dataset` contract does not forbid, just never fed to the loss.
"""

from __future__ import annotations

from typing import Any

from forge.data import Dataset

try:
    from ..mnist.dataset import DEFAULT_BASE_URL, MNISTDataset, download_mnist
except ImportError:  # running as a plain script, not a package
    from examples.mnist.dataset import DEFAULT_BASE_URL, MNISTDataset, download_mnist

DEFAULT_ROOT = "examples/mnist/data"


class AutoencoderDataset(Dataset):
    """`(image, image)` reconstruction pairs over MNIST -- see module docstring."""

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        train: bool = True,
        transform: "Any | None" = None,
        download: bool = False,
    ):
        self._mnist = MNISTDataset(root, train=train, transform=transform, download=download)

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, index: int) -> "tuple[Any, Any]":
        image, _label = self._mnist[index]
        return image, image

    def label_at(self, index: int) -> int:
        """The original MNIST class label at `index` -- evaluation only, never used for training."""
        _, label = self._mnist[index]
        return int(label.numpy())

    def __repr__(self) -> str:
        split = "train" if self._mnist.train else "test"
        return f"AutoencoderDataset(split={split!r}, size={len(self)})"


__all__ = ["AutoencoderDataset", "DEFAULT_ROOT", "DEFAULT_BASE_URL", "download_mnist"]
