"""A deterministic, in-process synthetic waveform classification dataset (Milestone 62).

Follows `examples/regression/dataset.py`'s precedent exactly: generated
entirely in-process from a fixed seed (no download, no third-party
dependency), an i.i.d. draw per sample so a contiguous train/val/test slice
is equivalent to a random split with no separate shuffle needed.

## The classification problem

Each sample is a length-`LENGTH` (default 64) single-channel time series
belonging to one of `NUM_CLASSES` (4) waveform shapes -- sine, square,
sawtooth, triangle -- drawn with a random frequency, phase, and amplitude,
then corrupted with additive Gaussian noise:

```text
x(t) = amplitude * shape(2*pi*freq*t + phase) + noise,  noise ~ Normal(0, 0.25)
```

`freq` (`Uniform(2, 6)` cycles across the window) and `phase`
(`Uniform(0, 2*pi)`) vary per sample -- a model that only memorized one fixed
phase/frequency per class could not classify a held-out sample, so this
requires learning genuine shape-invariant-to-shift features, exactly the
kind of local, translation-sensitive pattern `nn.Conv1d`'s sliding kernel is
suited to (`Linear`-only-in-time-domain models are hurt far more by phase
misalignment). This is a `UC1` ("train a classifier",
`docs/product/use-cases.md`) workload expressed with a model family -- 1D
temporal convolution -- none of Forge's other examples (`mnist`'s 2D `Conv2d`,
`char_rnn`/`word_rnn`'s `RNNCell`, `regression`'s plain MLP) cover.

At `NOISE_STD = 0.25` (comparable to the `~0.7-1.3` amplitude range), classes
are visually distinguishable but not trivially so -- particularly square vs.
sawtooth/triangle at unfavorable phase -- giving a non-trivial task with a
well-defined baseline (uniform random guessing = `1 / NUM_CLASSES = 25%`
accuracy).

## Determinism

`generate_raw(n_samples, seed)` draws everything (labels, frequency, phase,
amplitude, noise) from one `numpy.random.default_rng(seed)` instance in a
fixed sequence -- the same `seed` always reproduces the exact same `(X, y)`
arrays, independent of `forge.random`'s own generator (which instead governs
model parameter initialization -- see `train.py`'s Determinism section).

## Splits

`make_datasets()` draws `n_train + n_val + n_test` samples in one
`generate_raw()` call and slices them into contiguous train/val/test blocks,
exactly like `examples/regression/dataset.py::make_datasets`. No feature
normalization is applied -- the signal is already zero-centered with a
bounded, comparable-to-1 amplitude, unlike `regression`'s raw `Uniform(-2, 2)`
features.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge import Tensor
from forge.data import TensorDataset

LENGTH = 64
NUM_CLASSES = 4
CLASS_NAMES = ("sine", "square", "sawtooth", "triangle")

_FREQ_LOW, _FREQ_HIGH = 2.0, 6.0  # cycles spanned by the window
_AMPLITUDE_LOW, _AMPLITUDE_HIGH = 0.7, 1.3
_NOISE_STD = 0.25


def _waveform(label: int, phase_t: np.ndarray) -> np.ndarray:
    """The noiseless, unit-amplitude shape for `label` at phase-angle `phase_t` (radians)."""
    if label == 0:  # sine
        return np.sin(phase_t)
    if label == 1:  # square
        return np.sign(np.sin(phase_t))
    cycle_frac = (phase_t / (2 * np.pi)) % 1.0
    if label == 2:  # sawtooth: linear ramp -1 -> 1 over each cycle
        return 2.0 * cycle_frac - 1.0
    if label == 3:  # triangle: linear ramp -1 -> 1 -> -1 over each cycle
        return 2.0 * np.abs(2.0 * ((cycle_frac + 0.25) % 1.0) - 1.0) - 1.0
    raise ValueError(f"Unknown waveform label {label}, expected one of 0..{NUM_CLASSES - 1}.")


def generate_raw(n_samples: int, seed: int, length: int = LENGTH) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministically draw `n_samples` `(X, y)` pairs from a fixed `seed`.

    `X`: `(n_samples, 1, length)` float32 waveforms. `y`: `(n_samples,)`
    int64 class indices in `[0, NUM_CLASSES)`. The same `seed` always
    reproduces identical arrays.
    """
    if n_samples <= 0:
        raise ValueError(f"generate_raw requires n_samples > 0, got {n_samples}.")
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, NUM_CLASSES, size=n_samples)
    freqs = rng.uniform(_FREQ_LOW, _FREQ_HIGH, size=n_samples)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_samples)
    amplitudes = rng.uniform(_AMPLITUDE_LOW, _AMPLITUDE_HIGH, size=n_samples)
    noise = rng.normal(0.0, _NOISE_STD, size=(n_samples, length))

    t = np.arange(length, dtype=np.float64) / length
    X = np.empty((n_samples, length), dtype=np.float64)
    for i in range(n_samples):
        phase_t = 2 * np.pi * freqs[i] * t + phases[i]
        X[i] = amplitudes[i] * _waveform(int(labels[i]), phase_t)
    X = (X + noise).astype(np.float32).reshape(n_samples, 1, length)
    y = labels.astype(np.int64)
    return X, y


def make_datasets(
    n_train: int, n_val: int, n_test: int, seed: int = 0, length: int = LENGTH
) -> "tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, Any]]":
    """Build deterministic, non-overlapping train/val/test `TensorDataset`s.

    Returns `(train_ds, val_ds, test_ds, stats)`; `stats` holds
    `num_classes`, `class_names`, and `majority_baseline_accuracy` (uniform
    random guessing over `NUM_CLASSES` balanced-in-expectation classes),
    used by `train.py` to report a trivial baseline.
    """
    total = n_train + n_val + n_test
    X, y = generate_raw(total, seed, length=length)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    train_ds = TensorDataset(Tensor(X_train), Tensor(y_train))
    val_ds = TensorDataset(Tensor(X_val), Tensor(y_val))
    test_ds = TensorDataset(Tensor(X_test), Tensor(y_test))

    stats = {
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "majority_baseline_accuracy": 1.0 / NUM_CLASSES,
    }
    return train_ds, val_ds, test_ds, stats


__all__ = ["LENGTH", "NUM_CLASSES", "CLASS_NAMES", "generate_raw", "make_datasets"]
