"""A deterministic, in-process synthetic tabular *classification* dataset (Milestone 91).

`examples/regression` already exercises Forge's data pipeline for tabular
*regression*; no existing example combines tabular (non-image) input with
*classification* output. Framed as a small device-telemetry health
classifier: `N_FEATURES = 10` sensor-style readings per device, four health
states (`CLASS_NAMES`).

## The classification problem

Six of the ten features (`N_INFORMATIVE`) are genuinely discriminative --
drawn `Normal(center_c, _CLASS_STD)` around one of four fixed per-class
center vectors (`_CLASS_CENTERS`), with deliberate overlap between
neighboring classes so the problem is learnable but not trivially separable.
The remaining four features are pure distractors -- drawn `Uniform(-3, 3)`
independent of class, exactly mirroring `examples/regression/dataset.py`'s
`x7` distractor feature (a model that has genuinely learned the boundary
must implicitly down-weight them).

## Determinism

`generate_raw(samples_per_class, seed)` draws everything from one
`numpy.random.default_rng(seed)` instance, in class-block order (every
class-0 row, then every class-1 row, ...) -- deliberately **not** shuffled at
generation time. This mirrors a common real-world shape: a tabular dataset
read from a CSV that happens to be sorted by label (e.g. exported one class
at a time). See `make_datasets()`'s docstring for why this specific shape is
the reason this example uses `forge.data.random_split` rather than
`examples/regression`'s `sequential_split` -- a genuine, demonstrated
Section-12 "are the existing split utilities sufficient?" check, not an
arbitrary choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge import Tensor
from forge.data import Normalize, Subset, TensorDataset, random_split

N_FEATURES = 10
N_INFORMATIVE = 6
N_CLASSES = 4
CLASS_NAMES = ["normal", "warning", "critical", "fault"]

# Fixed per-class center vectors in the informative-feature subspace --
# hardcoded (not seed-derived) so the class structure itself is stable and
# reviewable across runs; only the noisy samples drawn around it vary with
# `seed`. Chosen with deliberate pairwise overlap (adjacent classes ~2-3
# units apart under _CLASS_STD=1.5) so the task requires real learning.
_CLASS_CENTERS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.5, 2.5, 0.0, 0.0, 0.0, 0.0],
        [2.5, -2.5, 2.5, 0.0, 0.0, 0.0],
        [-2.5, 0.0, -2.5, 2.5, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_CLASS_STD = 1.5
_DISTRACTOR_LOW, _DISTRACTOR_HIGH = -3.0, 3.0

assert _CLASS_CENTERS.shape == (N_CLASSES, N_INFORMATIVE)


def generate_raw(samples_per_class: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministically draw `N_CLASSES * samples_per_class` `(X, y)` rows from a fixed `seed`.

    `X`: `(N_CLASSES * samples_per_class, N_FEATURES)` float32, in class-block
    order (see module docstring). `y`: matching `(N,)` int64 class indices
    `0..N_CLASSES-1`. The same `seed` always reproduces identical arrays.
    """
    if samples_per_class <= 0:
        raise ValueError(f"generate_raw requires samples_per_class > 0, got {samples_per_class}.")
    rng = np.random.default_rng(seed)

    blocks = []
    labels = []
    for class_index, center in enumerate(_CLASS_CENTERS):
        informative = rng.normal(loc=center, scale=_CLASS_STD, size=(samples_per_class, N_INFORMATIVE))
        distractor = rng.uniform(
            _DISTRACTOR_LOW, _DISTRACTOR_HIGH, size=(samples_per_class, N_FEATURES - N_INFORMATIVE)
        )
        blocks.append(np.concatenate([informative, distractor], axis=1))
        labels.append(np.full(samples_per_class, class_index, dtype=np.int64))

    X = np.concatenate(blocks, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0)
    return X, y


def make_datasets(
    samples_per_class_train: int,
    samples_per_class_val: int,
    samples_per_class_test: int,
    seed: int = 0,
) -> "tuple[Subset, Subset, Subset, dict[str, Any]]":
    """Build deterministic, non-overlapping, class-balanced-in-expectation train/val/test splits.

    Draws `samples_per_class_train + samples_per_class_val +
    samples_per_class_test` rows per class in one `generate_raw()` call (so
    the raw array is laid out in contiguous class blocks -- see the module
    docstring), then splits with `forge.data.random_split` rather than
    `examples/regression`'s `sequential_split`: `sequential_split` takes
    consecutive index blocks, which for class-blocked data would put entire
    classes only in the train split (and none in val/test) -- a genuine
    correctness problem `random_split`'s existing permute-then-slice
    behavior already solves, with no new splitting capability needed.

    Feature standardization (`forge.data.Normalize`) is fit on the *training*
    split's mean/std only -- computed from the actual training indices
    `random_split` returned (not a fixed prefix of the raw array, unlike
    `examples/regression/dataset.py`, since the training split here is a
    random subset, not a contiguous block) -- then applied identically to
    all three splits via one shared `TensorDataset` transform, exactly like
    `examples/regression/dataset.py`. Class labels `y` are left as raw
    integer indices (`CrossEntropyLoss`'s expected target shape), never
    normalized or one-hot encoded.

    Returns `(train_ds, val_ds, test_ds, stats)`; `stats["transform"]` is the
    fitted `Normalize` instance, reused as-is for `preprocessing=` at
    persistence time (mirroring `examples/regression`'s own `stats["transform"]`
    contract), and `stats["train_class_counts"]` reports the actual per-class
    count the random split produced (not exactly equal across classes, since
    `random_split` is not stratified -- see the module docstring's Section-12
    note: no stratification was added because this workload does not need
    exact per-class balance to train successfully, only a representative mix).
    """
    samples_per_class = samples_per_class_train + samples_per_class_val + samples_per_class_test
    X, y = generate_raw(samples_per_class, seed)

    n_train = N_CLASSES * samples_per_class_train
    n_val = N_CLASSES * samples_per_class_val
    n_test = N_CLASSES * samples_per_class_test

    full_ds_raw = TensorDataset(Tensor(X), Tensor(y))
    # A separate generator from generate_raw()'s own rng (mirroring
    # examples/regression/dataset.py's independent-stream-per-concern
    # policy), so the split permutation and the feature/noise draws are two
    # independent deterministic streams for a given seed.
    split_rng = np.random.default_rng(seed + 1)
    train_split, val_split, test_split = random_split(full_ds_raw, [n_train, n_val, n_test], generator=split_rng)

    train_X = X[train_split.indices]
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    transform = Normalize(mean=mean, std=std)

    full_ds = TensorDataset(Tensor(X), Tensor(y), transform=transform)
    train_ds = Subset(full_ds, train_split.indices)
    val_ds = Subset(full_ds, val_split.indices)
    test_ds = Subset(full_ds, test_split.indices)

    train_class_counts = np.bincount(y[train_split.indices], minlength=N_CLASSES).tolist()

    stats = {
        "mean": mean,
        "std": std,
        "transform": transform,
        "train_class_counts": train_class_counts,
    }
    return train_ds, val_ds, test_ds, stats


__all__ = ["N_FEATURES", "N_INFORMATIVE", "N_CLASSES", "CLASS_NAMES", "generate_raw", "make_datasets"]
