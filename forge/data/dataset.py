"""The Dataset abstraction and the built-in array-backed dataset.

`Dataset` is a minimal, model-independent protocol -- a source of samples
addressable by length and index. It intentionally has no dependency on
`Module`, `Loss`, `Optimizer`, or a training engine (see
`docs/architecture/data-pipeline.md`); a `Dataset` only ever needs to know
how many samples it has and how to produce one.

A sample may be a single `Tensor`, a tuple such as `(features, target)`, or
another representation a custom `Dataset` documents for itself -- Forge does
not force every dataset into one hard-coded schema.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .. import random as forge_random
from ..exceptions import DataError
from ..tensor.tensor import Tensor


class Dataset:
    """Base class for a source of samples: `__len__` and `__getitem__`.

    Subclasses implement both. The base implementations raise `DataError`
    rather than `NotImplementedError`, matching the "must implement" pattern
    already used by `Module.forward`/`Loss.forward`/`Optimizer.step`.
    """

    def __len__(self) -> int:
        raise DataError(f"{type(self).__name__} does not implement __len__().")

    def __getitem__(self, index: int) -> Any:
        raise DataError(f"{type(self).__name__} does not implement __getitem__().")


def _normalize_index(index: Any, length: int, owner: str) -> int:
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise DataError(f"{owner} index must be an int, got {type(index).__name__}.")
    idx = int(index)
    if idx < 0:
        idx += length
    if not (0 <= idx < length):
        raise DataError(f"Index {index} out of range for {owner} of size {length}.")
    return idx


class TensorDataset(Dataset):
    """An in-memory dataset backed by one or more aligned Tensors.

    ```python
    dataset = TensorDataset(features, targets)
    x, y = dataset[0]
    ```

    Every tensor must share the same size along its first (sample) axis;
    a mismatch raises `DataError` at construction. Indexing returns a tuple
    of per-sample Tensors (one component per input tensor, in the same
    order), each preserving its source tensor's dtype and device. A dataset
    built from a single tensor returns that sample directly rather than a
    1-tuple.

    `transform` is applied to the first tensor's sample (the conventional
    "features" position); `target_transform` is applied to the second
    tensor's sample (the conventional "target" position), if present. This
    keeps a transform meant only for features from silently reaching labels
    -- feature and target preprocessing are always configured separately.
    Transforms beyond the first two tensors are not supported.
    """

    def __init__(
        self,
        *tensors: "Tensor | Any",
        transform: "Any | None" = None,
        target_transform: "Any | None" = None,
    ):
        if not tensors:
            raise DataError("TensorDataset requires at least one tensor.")

        wrapped = tuple(t if isinstance(t, Tensor) else Tensor(t) for t in tensors)
        for i, t in enumerate(wrapped):
            if t.ndim == 0:
                raise DataError(
                    f"TensorDataset tensor {i} has shape () -- every tensor needs at "
                    "least one dimension to serve as the sample axis."
                )

        size = wrapped[0].shape[0]
        for i, t in enumerate(wrapped):
            if t.shape[0] != size:
                raise DataError(
                    "TensorDataset requires every tensor to have the same number of "
                    f"samples; tensor 0 has {size} but tensor {i} has {t.shape[0]}."
                )
        if size == 0:
            raise DataError("TensorDataset requires at least one sample.")
        if target_transform is not None and len(wrapped) < 2:
            raise DataError(
                "target_transform requires at least two tensors (features, target)."
            )

        self.tensors = wrapped
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int) -> "Tensor | tuple[Tensor, ...]":
        idx = _normalize_index(index, len(self), "TensorDataset")

        samples = [self._extract(t, idx) for t in self.tensors]
        if self.transform is not None:
            samples[0] = self.transform(samples[0])
        if self.target_transform is not None:
            samples[1] = self.target_transform(samples[1])

        if len(samples) == 1:
            return samples[0]
        return tuple(samples)

    @staticmethod
    def _extract(tensor: Tensor, idx: int) -> Tensor:
        return Tensor(tensor.numpy()[idx], dtype=tensor.dtype, device=tensor.device)

    def __repr__(self) -> str:
        return f"TensorDataset(n_tensors={len(self.tensors)}, size={len(self)})"


class Subset(Dataset):
    """A dataset view over a subset of another dataset's indices, order preserved."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        idx = _normalize_index(index, len(self), "Subset")
        return self.dataset[self.indices[idx]]

    def __repr__(self) -> str:
        return f"Subset(size={len(self)})"


def random_split(
    dataset: Dataset,
    lengths: Iterable[int],
    generator: "np.random.Generator | None" = None,
) -> list[Subset]:
    """Split `dataset` into disjoint `Subset`s of the given sizes.

    `lengths` must be non-negative and sum to exactly `len(dataset)`. Indices
    are permuted once (via `generator`, or `forge.random.default_generator()`
    if omitted) and sliced into consecutive blocks, so each returned `Subset`
    preserves feature/target sample correspondence from the source dataset.
    Deterministic for a given generator/seed, matching `forge.random`'s
    reproducibility model (see `docs/architecture/modules.md`).
    """
    lengths = [int(n) for n in lengths]
    if any(n < 0 for n in lengths):
        raise DataError(f"random_split lengths must be non-negative, got {lengths}.")

    total = len(dataset)
    if sum(lengths) != total:
        raise DataError(
            f"random_split lengths must sum to the dataset size ({total}), "
            f"got {lengths} summing to {sum(lengths)}."
        )

    rng = generator if generator is not None else forge_random.default_generator()
    permutation = rng.permutation(total)

    subsets = []
    offset = 0
    for n in lengths:
        subsets.append(Subset(dataset, permutation[offset : offset + n]))
        offset += n
    return subsets


__all__ = ["Dataset", "TensorDataset", "Subset", "random_split"]
