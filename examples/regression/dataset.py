"""A deterministic, in-process synthetic tabular regression dataset (Milestone 60).

`vision.md`/`use-cases.md` name tabular regression (UC2) as one of Forge's
three core workload families, but never require a *specific* dataset --
unlike MNIST (a real external corpus), this example follows `char_rnn`/
`word_rnn`'s precedent of generating its data entirely in-process from a
fixed seed, so the example needs no download and no third-party dependency.

## The regression problem

`N_FEATURES = 8` continuous features, each drawn i.i.d. `Uniform(-2, 2)`.
The target is a fixed function of six of them, mixing linear, interaction,
and quadratic terms so a plain linear model cannot fit it exactly (motivating
the `ReLU` MLP in `model.py` rather than a bare `Linear`):

```text
y = 3.0*x0 - 2.0*x1 + 1.5*x2 + 0.5*x3      (linear terms)
    + 1.2 * x4 * x5                          (interaction term)
    + 0.8 * x6**2                            (quadratic term)
    + noise,  noise ~ Normal(0, 0.5)
```

`x7` is a pure distractor: drawn from the same distribution as every other
feature but never used by `true_function`, so a model that has genuinely
learned the relationship must implicitly learn to down-weight it, rather
than the dataset being solvable by memorizing a trivial one-to-one mapping.

## Determinism

`generate_raw(n_samples, seed)` draws everything (features, noise) from one
`numpy.random.default_rng(seed)` instance in a fixed sequence -- the same
`seed` always reproduces the exact same `(X, y)` arrays, independent of
`forge.random`'s own generator (which instead governs model parameter
initialization -- see `train.py`'s Determinism section).

## Splits and preprocessing

`make_datasets()` draws `n_train + n_val + n_test` samples in one
`generate_raw()` call, wraps them in a single `TensorDataset`, then carves
train/val/test blocks out with `forge.data.sequential_split` (Milestone 74)
rather than hand-slicing the underlying NumPy arrays -- this example and
`examples/waveform_classification/dataset.py` used to duplicate the same
`X[:n_train]` / `X[n_train:n_train+n_val]` / `X[n_train+n_val:]` bookkeeping
independently; see `docs/development/m74-data-workflow.md`. Because every row
is an independent identically-distributed draw, a contiguous block is
equivalent to a random split (no shuffling is needed here, unlike
`forge.data.random_split`'s use in `examples/trainer_demo.py` for a dataset
with meaningful sample order). Feature standardization (`forge.data.
Normalize`) is fit on the *training* split's mean/std only (still a direct
`X[:n_train]` slice of the raw array -- computing train-only statistics is a
modeling concern `sequential_split` itself has no reason to know about),
then applied identically to all three splits via the shared `TensorDataset`'s
one `transform` -- validation/test statistics never leak into the transform.
The target `y` is left in its natural scale (not normalized), so reported
MSE/RMSE are directly interpretable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge import Tensor
from forge.data import Normalize, TensorDataset, sequential_split

N_FEATURES = 8

# Coefficients for x0..x3 (pure linear terms).
_LINEAR_COEFFICIENTS = (3.0, -2.0, 1.5, 0.5)
_INTERACTION_COEFFICIENT = 1.2  # x4 * x5
_QUADRATIC_COEFFICIENT = 0.8  # x6 ** 2
# x7 is a deliberate distractor -- see module docstring.
_NOISE_STD = 0.5
_FEATURE_LOW, _FEATURE_HIGH = -2.0, 2.0


def true_function(X: np.ndarray) -> np.ndarray:
    """The noiseless target `f(X)` used by `generate_raw()` -- see module docstring."""
    c0, c1, c2, c3 = _LINEAR_COEFFICIENTS
    return (
        c0 * X[:, 0]
        + c1 * X[:, 1]
        + c2 * X[:, 2]
        + c3 * X[:, 3]
        + _INTERACTION_COEFFICIENT * X[:, 4] * X[:, 5]
        + _QUADRATIC_COEFFICIENT * X[:, 6] ** 2
    )


def generate_raw(n_samples: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministically draw `n_samples` `(X, y)` pairs from a fixed `seed`.

    `X`: `(n_samples, N_FEATURES)` float32, i.i.d. `Uniform(-2, 2)`.
    `y`: `(n_samples, 1)` float32, `true_function(X) + Normal(0, NOISE_STD)`.
    The same `seed` always reproduces identical arrays.
    """
    if n_samples <= 0:
        raise ValueError(f"generate_raw requires n_samples > 0, got {n_samples}.")
    rng = np.random.default_rng(seed)
    X = rng.uniform(_FEATURE_LOW, _FEATURE_HIGH, size=(n_samples, N_FEATURES)).astype(np.float32)
    noise = rng.normal(0.0, _NOISE_STD, size=(n_samples,)).astype(np.float32)
    y = (true_function(X) + noise).astype(np.float32).reshape(-1, 1)
    return X, y


def make_datasets(
    n_train: int, n_val: int, n_test: int, seed: int = 0
) -> "tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, Any]]":
    """Build deterministic, non-overlapping train/val/test `TensorDataset`s.

    Returns `(train_ds, val_ds, test_ds, stats)`, where `stats` holds the
    training-split feature `mean`/`std`, the `transform` (the same fitted
    `Normalize` instance baked into each split's `TensorDataset`, Milestone
    83 -- reused as-is for `preprocessing=` at persistence time rather than
    reconstructed from `mean`/`std`), and `y_train_mean`/`y_train_var` (used
    by `train.py` to report a trivial predict-the-mean baseline MSE).
    """
    total = n_train + n_val + n_test
    X, y = generate_raw(total, seed)

    X_train, y_train = X[:n_train], y[:n_train]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    transform = Normalize(mean=mean, std=std)

    full_ds = TensorDataset(Tensor(X), Tensor(y), transform=transform)
    train_ds, val_ds, test_ds = sequential_split(full_ds, [n_train, n_val, n_test])

    stats = {
        "mean": mean,
        "std": std,
        "y_train_mean": float(y_train.mean()),
        "y_train_var": float(y_train.var()),
        # Milestone 83: the exact fitted Normalize instance every split's
        # TensorDataset already applies -- exposed so train.py can hand it to
        # forge.save_model(..., preprocessing=...)/train_and_save(...,
        # preprocessing=...) unchanged, rather than reconstructing an
        # equivalent Normalize(mean=stats["mean"], std=stats["std"]) from the
        # raw arrays above.
        "transform": transform,
    }
    return train_ds, val_ds, test_ds, stats


__all__ = ["N_FEATURES", "true_function", "generate_raw", "make_datasets"]
