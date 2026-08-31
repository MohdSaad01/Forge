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
        self._inv_std = 1.0 / std_arr

    def __call__(self, sample: Tensor) -> Tensor:
        if not isinstance(sample, Tensor):
            raise DataError(
                f"Normalize expects a Tensor sample, got {type(sample).__name__}."
            )
        mean_t = Tensor(self.mean, dtype=sample.dtype, device=sample.device)
        inv_std_t = Tensor(self._inv_std, dtype=sample.dtype, device=sample.device)
        return (sample - mean_t) * inv_std_t


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


class Lambda(Transform):
    """Wrap an arbitrary callable as a Transform."""

    def __init__(self, fn: Callable[[Any], Any]):
        if not callable(fn):
            raise DataError(f"Lambda requires a callable, got {fn!r}.")
        self.fn = fn

    def __call__(self, sample: Any) -> Any:
        return self.fn(sample)


__all__ = ["Transform", "Compose", "ToTensor", "Normalize", "Reshape", "Flatten", "Lambda"]
