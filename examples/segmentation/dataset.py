"""A deterministic, in-process synthetic segmentation dataset (Milestone 64).

Every prior Forge image example (`mnist`, `autoencoder`) reduces an image to
a single vector output (a class index, a latent code) or reconstructs the
whole image back (autoencoder). None of them produce a *per-pixel* output --
this dataset exists to give the segmentation example a genuine dense
-prediction target: for each `(3, H, W)` synthetic image, a same-resolution
`(1, H, W)` binary mask marking exactly which pixels belong to one randomly
placed foreground shape (a filled circle or square) against a noisy
background.

Following `examples/regression/dataset.py`'s and `examples/char_rnn`'s
precedent (`vision.md`/`use-cases.md` name segmentation-style dense
prediction as a legitimate task shape but require no *specific* dataset),
everything is generated in-process from a fixed seed -- no download, no
third-party dependency beyond NumPy.

## The task

A `size x size` RGB image with a solid, low-noise background color and one
randomly sized/positioned/colored foreground shape (circle or square). The
foreground color is drawn from a wide hue range but rejected and redrawn
until it differs from the (fixed) background color by a minimum contrast
margin -- this guarantees the task is *learnable* by a small local-receptive
-field CNN (a real color/edge cue exists at every foreground/background
boundary) while still requiring genuine per-pixel spatial reasoning: shape
identity, size, position, and exact color all vary every sample, so a model
cannot memorize a fixed mask or a single global threshold -- it must learn
that "this pixel's color differs from the background" from local evidence,
exactly the kind of signal a small `Conv2d` stack is built to extract.

The background color itself is intentionally fixed across the whole dataset
(not randomized per sample): this keeps the task solvable by the
deliberately tiny model `examples/segmentation/model.py` uses (Milestone
64's brief calls for "the smallest architecture that genuinely demonstrates
dense prediction," not a task calibrated to need a deep network with a large
receptive field). Shape type, size, position, and color are all randomized
per sample, so the dataset is not a fixed handful of images repeated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge import Tensor
from forge.data import Dataset

IMAGE_SIZE = 32
NUM_CHANNELS = 3

_BACKGROUND_COLOR = np.array([0.82, 0.82, 0.85], dtype=np.float32)
_BACKGROUND_NOISE_STD = 0.03
_FOREGROUND_NOISE_STD = 0.03
_MIN_CONTRAST = 0.35  # minimum mean-abs-channel-difference between fg and bg color
_MIN_RADIUS = 4
_MAX_RADIUS = 10


def _draw_foreground_color(rng: np.random.Generator) -> np.ndarray:
    """A random RGB color at least `_MIN_CONTRAST` away (mean abs channel diff) from the background."""
    while True:
        color = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
        if float(np.abs(color - _BACKGROUND_COLOR).mean()) >= _MIN_CONTRAST:
            return color


def _generate_one(rng: np.random.Generator, size: int) -> "tuple[np.ndarray, np.ndarray]":
    """One `(image, mask)` pair: `image` is `(3, H, W)` float32 in `[0, 1]`, `mask` is `(1, H, W)` float32 in `{0, 1}`."""
    noise = rng.normal(0.0, _BACKGROUND_NOISE_STD, size=(size, size, 3)).astype(np.float32)
    image_hwc = np.clip(_BACKGROUND_COLOR[None, None, :] + noise, 0.0, 1.0)

    shape_kind = rng.choice(("circle", "square"))
    radius = int(rng.integers(_MIN_RADIUS, _MAX_RADIUS + 1))
    cy = int(rng.integers(radius, size - radius))
    cx = int(rng.integers(radius, size - radius))

    yy, xx = np.mgrid[0:size, 0:size]
    if shape_kind == "circle":
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    else:
        mask = (np.abs(yy - cy) <= radius) & (np.abs(xx - cx) <= radius)

    fg_color = _draw_foreground_color(rng)
    fg_noise = rng.normal(0.0, _FOREGROUND_NOISE_STD, size=(size, size, 3)).astype(np.float32)
    image_hwc[mask] = np.clip(fg_color[None, :] + fg_noise[mask], 0.0, 1.0)

    image_chw = np.transpose(image_hwc, (2, 0, 1)).astype(np.float32)
    mask_chw = mask[None, :, :].astype(np.float32)
    return image_chw, mask_chw


def generate_raw(n_samples: int, seed: int, size: int = IMAGE_SIZE) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministically draw `n_samples` `(image, mask)` pairs from a fixed `seed`.

    Returns `(images, masks)`: `images` is `(n_samples, 3, size, size)` float32
    in `[0, 1]`; `masks` is `(n_samples, 1, size, size)` float32 in `{0, 1}`.
    The same `seed` always reproduces identical arrays, matching
    `examples/regression/dataset.py::generate_raw`'s determinism contract.
    """
    if n_samples <= 0:
        raise ValueError(f"generate_raw requires n_samples > 0, got {n_samples}.")
    rng = np.random.default_rng(seed)
    images = np.empty((n_samples, 3, size, size), dtype=np.float32)
    masks = np.empty((n_samples, 1, size, size), dtype=np.float32)
    for i in range(n_samples):
        images[i], masks[i] = _generate_one(rng, size)
    return images, masks


class SegmentationDataset(Dataset):
    """`(image, mask)` pairs -- see module docstring. Generated once at construction, held in memory."""

    def __init__(self, n_samples: int, seed: int, size: int = IMAGE_SIZE):
        self._images, self._masks = generate_raw(n_samples, seed, size)

    def __len__(self) -> int:
        return self._images.shape[0]

    def __getitem__(self, index: int) -> "tuple[Tensor, Tensor]":
        return Tensor(self._images[index]), Tensor(self._masks[index])

    def __repr__(self) -> str:
        return f"SegmentationDataset(size={len(self)})"


def make_datasets(
    n_train: int, n_test: int, seed: int = 0, size: int = IMAGE_SIZE
) -> "tuple[SegmentationDataset, SegmentationDataset]":
    """Build deterministic, non-overlapping train/test `SegmentationDataset`s.

    Train and test draw from independent generator streams (`seed` and
    `seed + 1`) rather than slicing one combined draw -- unlike
    `examples/regression`'s fixed-length-feature-vector dataset, there is no
    natural "generate everything, then slice" shape here since each sample
    is an independently rendered image.
    """
    train_ds = SegmentationDataset(n_train, seed=seed, size=size)
    test_ds = SegmentationDataset(n_test, seed=seed + 1, size=size)
    return train_ds, test_ds


__all__ = ["IMAGE_SIZE", "NUM_CHANNELS", "generate_raw", "SegmentationDataset", "make_datasets"]
