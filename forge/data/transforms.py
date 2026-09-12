"""A small, composable transform abstraction for per-sample preprocessing.

Transforms operate on a single sample component (a `Tensor`, or whatever a
custom `Dataset` chooses to pass through), not on a whole `(features,
target)` tuple. This keeps a transform meant only for features from
accidentally reaching labels: `TensorDataset` wires `transform` to the
features position and `target_transform` to the target position separately
(see `forge/data/dataset.py`), rather than handing every transform the full
sample and trusting it to leave the label alone.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from ..exceptions import DataError
from ..tensor.tensor import Tensor


class Transform:
    """Base class for a single-sample-component preprocessing step."""

    def __call__(self, sample: Any) -> Any:
        raise DataError(f"{type(self).__name__} does not implement __call__().")


class Compose(Transform):
    """Apply a sequence of transforms in order, threading the result through each."""

    def __init__(self, transforms: Iterable[Callable[[Any], Any]]):
        self.transforms = list(transforms)
        for t in self.transforms:
            if not callable(t):
                raise DataError(
                    f"Compose requires every transform to be callable, got {t!r}."
                )

    def __call__(self, sample: Any) -> Any:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        inner = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose([{inner}])"


class ToTensor(Transform):
    """Convert array-like data to a Tensor, e.g. as the first step in a Compose."""

    def __init__(self, dtype: Any = None, device: str = "cpu"):
        self.dtype = dtype
        self.device = device

    def __call__(self, sample: Any) -> Tensor:
        return Tensor(sample, dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:
        return f"ToTensor(dtype={self.dtype!r}, device={self.device!r})"


class Normalize(Transform):
    """Elementwise `(x - mean) / std`, applied to a single Tensor sample.

    `mean`/`std` broadcast against the sample the same way Tensor arithmetic
    broadcasts elsewhere in Forge (e.g. a scalar `mean`/`std` normalizes
    every element identically; a per-channel `mean`/`std` broadcasts across
    the remaining dimensions). Implemented as `(x - mean) * (1/std)` since
    Forge's Tensor has no division operator.
    """

    def __init__(self, mean: Any, std: Any):
        std_arr = np.asarray(std, dtype=np.float64)
        if np.any(std_arr == 0):
            raise DataError("Normalize requires every std value to be non-zero.")
        self.mean = mean
        # `std` itself (not just its reciprocal) is kept so a serializer
        # (`forge.serialization.transforms`) can round-trip the exact
        # constructor arguments rather than reconstructing `std` from
        # `1 / self._inv_std`, which would introduce needless float error.
        self.std = std
        self._inv_std = 1.0 / std_arr

    def __call__(self, sample: Tensor) -> Tensor:
        if not isinstance(sample, Tensor):
            raise DataError(
                f"Normalize expects a Tensor sample, got {type(sample).__name__}."
            )
        mean_t = Tensor(self.mean, dtype=sample.dtype, device=sample.device)
        inv_std_t = Tensor(self._inv_std, dtype=sample.dtype, device=sample.device)
        return (sample - mean_t) * inv_std_t

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean!r}, std={self.std!r})"


class Reshape(Transform):
    """Reshape a single Tensor sample to the given shape (see `Tensor.reshape`)."""

    def __init__(self, *shape: int):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        self.shape = shape

    def __call__(self, sample: Tensor) -> Tensor:
        if not isinstance(sample, Tensor):
            raise DataError(
                f"Reshape expects a Tensor sample, got {type(sample).__name__}."
            )
        return sample.reshape(*self.shape)

    def __repr__(self) -> str:
        return f"Reshape(shape={self.shape!r})"


class Flatten(Transform):
    """Reshape a single Tensor sample to one dimension."""

    def __call__(self, sample: Tensor) -> Tensor:
        if not isinstance(sample, Tensor):
            raise DataError(
                f"Flatten expects a Tensor sample, got {type(sample).__name__}."
            )
        size = 1
        for d in sample.shape:
            size *= d
        return sample.reshape(size)

    def __repr__(self) -> str:
        return "Flatten()"


class Resize(Transform):
    """Resize a `(C, H, W)` image Tensor to an explicit `(height, width)` (Milestone 70).

    Exists to make mixed-resolution `ImageFolder` datasets batchable by
    `DataLoader`, whose `_stack` requires every sample in a batch to share
    one shape -- see `forge/data/image_folder.py`'s own documented
    limitation. `Resize` is deliberately an ordinary preprocessing
    transform, not a differentiable Tensor op: it runs via Pillow on a
    Tensor's raw pixel data, outside the autograd graph, exactly like
    `ImageFolder._load_image`'s own Pillow-based decode step.

    ```python
    dataset = ImageFolder(root, transform=Resize((64, 64)))
    ```

    - `size`: a `(height, width)` tuple of positive ints. The output
      Tensor's shape is always `(C, height, width)`.
    - Input: a `(C, H, W)` Tensor with `C` of 1 (grayscale) or 3 (RGB),
      values in Pillow's native `[0, 255]` 8-bit range -- i.e. the
      representation `ImageFolder` produces directly. Apply `Resize` before
      any transform that rescales pixel values (e.g. a `Lambda` dividing by
      255), not after: resizing reinterprets whatever values are present as
      8-bit pixel intensities.
    - Output: `(C, height, width)` Tensor, same dtype and device as the
      input (matching every other transform in this module).
    - Interpolation: Pillow's bilinear filter (`Image.BILINEAR|Resampling.
      BILINEAR`) -- a reasonable general-purpose default; not configurable,
      since no consumer has needed another filter yet.
    """

    def __init__(self, size: "tuple[int, int]"):
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise DataError(f"Resize requires a (height, width) tuple, got {size!r}.")
        height, width = size
        for value in (height, width):
            if isinstance(value, bool) or not isinstance(value, int):
                raise DataError(f"Resize requires integer (height, width), got {size!r}.")
            if value <= 0:
                raise DataError(f"Resize requires positive (height, width), got {size!r}.")
        self.size = (int(height), int(width))

    def __call__(self, sample: Tensor) -> Tensor:
        if not isinstance(sample, Tensor):
            raise DataError(f"Resize expects a Tensor sample, got {type(sample).__name__}.")
        if sample.ndim != 3 or sample.shape[0] not in (1, 3):
            raise DataError(
                f"Resize expects a (C, H, W) Tensor with C in (1, 3), got shape {sample.shape}."
            )

        channels = sample.shape[0]
        height, width = self.size
        chw = np.clip(sample.numpy(), 0, 255).astype(np.uint8)
        hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))

        if channels == 1:
            image = Image.fromarray(hwc[:, :, 0], mode="L")
        else:
            image = Image.fromarray(hwc, mode="RGB")
        resized = image.resize((width, height), Image.BILINEAR)

        resized_array = np.array(resized, dtype=np.float32)
        if channels == 1:
            resized_array = resized_array[np.newaxis, :, :]
        else:
            resized_array = resized_array.transpose(2, 0, 1)
        resized_array = np.ascontiguousarray(resized_array)

        return Tensor(resized_array, dtype=sample.dtype, device=sample.device)

    def __repr__(self) -> str:
        return f"Resize(size={self.size})"


class Lambda(Transform):
    """Wrap an arbitrary callable as a Transform."""

    def __init__(self, fn: Callable[[Any], Any]):
        if not callable(fn):
            raise DataError(f"Lambda requires a callable, got {fn!r}.")
        self.fn = fn

    def __call__(self, sample: Any) -> Any:
        return self.fn(sample)


__all__ = ["Transform", "Compose", "ToTensor", "Normalize", "Reshape", "Flatten", "Resize", "Lambda"]
