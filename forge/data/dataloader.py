"""Batching: iterate a Dataset and produce model-ready Tensor batches.

`DataLoader` is independent of `Module`/`Loss`/`Optimizer` -- it only knows
how to ask a `Dataset` for samples by index and stack the results into
batches. No multiprocessing workers, no asynchronous prefetching (see
`docs/architecture/data-pipeline.md`); iteration is plain synchronous Python.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np

from .. import random as forge_random
from ..exceptions import DataError
from ..tensor.tensor import Tensor
from .dataset import Dataset


class DataLoader:
    """Iterate `dataset` in batches of `batch_size`, optionally shuffled.

    ```python
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_x, batch_y in loader:
        ...
    ```

    - `batch_size`: must be a positive int. Default `1`.
    - `shuffle`: if `True`, sample order is permuted independently each time
      iteration starts (`for batch in loader`), drawing from `generator` if
      given, otherwise `forge.random.default_generator()` -- the same
      process-global generator `Linear` draws from, so `forge.random.seed()`
      makes an unshuffled run's *and* a shuffled run's ordering
      reproducible. Passing an explicit `generator` (e.g.
      `np.random.default_rng(0)`) makes a given loader's ordering
      reproducible independently of the global generator's state.
    - `drop_last`: if `True`, a final batch smaller than `batch_size` is
      dropped rather than yielded. Default `False`.

    A dataset item that is a single `Tensor` batches into a single `Tensor`
    of shape `(batch_size, *sample.shape)`. A dataset item that is a tuple
    (e.g. `(features, target)`) batches into a tuple of such Tensors, one per
    component, preserving the feature/target correspondence at matching
    batch indices.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: "np.random.Generator | None" = None,
    ):
        try:
            len(dataset)
        except TypeError as exc:
            raise DataError(
                "DataLoader requires a dataset supporting len(); "
                f"{type(dataset).__name__} does not."
            ) from exc

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise DataError(f"batch_size must be a positive int, got {batch_size!r}.")
        if not isinstance(shuffle, bool):
            raise DataError(f"shuffle must be a bool, got {shuffle!r}.")
        if not isinstance(drop_last, bool):
            raise DataError(f"drop_last must be a bool, got {drop_last!r}.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            rng = self.generator if self.generator is not None else forge_random.default_generator()
            rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            samples = [self.dataset[int(i)] for i in batch_indices]
            yield _collate(samples)

    def __repr__(self) -> str:
        return (
            f"DataLoader(batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"drop_last={self.drop_last})"
        )


def _collate(samples: list) -> Any:
    """Stack a list of dataset samples (Tensors, or tuples of Tensors) into a batch."""
    first = samples[0]
    if isinstance(first, tuple):
        for s in samples:
            if not isinstance(s, tuple) or len(s) != len(first):
                raise DataError(
                    "Cannot batch samples with inconsistent structure: expected every "
                    f"sample to be a {len(first)}-tuple like the first sample."
                )
        return tuple(_stack([s[i] for s in samples]) for i in range(len(first)))
    return _stack(samples)


def _stack(components: list) -> Tensor:
    first = components[0]
    if not isinstance(first, Tensor):
        raise DataError(
            f"Cannot batch a sample component of type {type(first).__name__}; "
            "DataLoader batches Tensor-valued samples."
        )
    arrays = []
    for c in components:
        if not isinstance(c, Tensor):
            raise DataError(
                f"Cannot batch a sample component of type {type(c).__name__}; "
                "DataLoader batches Tensor-valued samples."
            )
        if c.dtype != first.dtype:
            raise DataError(
                f"Cannot batch samples with differing dtypes: {first.dtype} and {c.dtype}."
            )
        if c.shape != first.shape:
            raise DataError(
                f"Cannot batch samples with differing shapes: {first.shape} and {c.shape}."
            )
        arrays.append(c.numpy())
    stacked = np.stack(arrays, axis=0)
    return Tensor(stacked, dtype=first.dtype, device=first.device)


__all__ = ["DataLoader"]
