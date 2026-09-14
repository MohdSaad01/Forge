"""A real, externally-sourced tabular classification dataset (Milestone 92).

`examples/tabular_classification` (Milestone 91) proved Forge's tabular-
classification workflow end-to-end, but on a **synthetic** dataset generated
in-process. Milestone 92 asks whether that same workflow survives contact
with an ordinary real-world tabular file a developer would actually
download.

## The dataset

The Pima Indians Diabetes dataset: 768 female patients of Pima Indian
heritage, 8 numeric diagnostic measurements each, and a binary `Outcome`
(1 = diabetes diagnosed within 5 years, 0 = not) -- a genuine, widely-used
UCI/NIDDK-sourced benchmark, not a toy or synthetic set. Retrieved once as
`data/diabetes.csv` (with header) from
`https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`
-- see this file's own header row for the exact column names -- and
committed here so the example needs no network access at run time (Milestone
92 brief, Section 3: acquire externally, do not build a hosting mechanism).

```text
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age  ->  Outcome (0 = no diabetes, 1 = diabetes)
```

## Class balance

500 negative / 268 positive (65.1% / 34.9%) -- meaningfully imbalanced,
unlike `tabular_classification`'s synthetic, roughly-balanced four-class
problem. This changes what "trivial baseline" means: a uniform coin flip
gets 50%, but always predicting the majority class ("no diabetes") gets
65.1% for free. `train.py` reports the majority-class baseline, not a
uniform-random one, so a reported accuracy is only meaningful evidence of
real learning once it clears 65.1%, not 50%.

## The real preprocessing requirement this dataset exposes

Five of the eight features can never legitimately be zero in a living
patient -- `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` --
but the source data encodes an unmeasured reading as literal `0`, not a
blank field or `NaN`, a genuinely common real-world missing-value encoding:

```text
column                zeros   % of 768 rows
Glucose                   5    0.7%
BloodPressure             35    4.6%
SkinThickness            227   29.6%
Insulin                  374   48.7%
BMI                       11    1.4%
```

(`Pregnancies == 0` is left alone -- zero pregnancies is a real, valid
value, not a missing reading; `DiabetesPedigreeFunction`/`Age` have no
zeros at all.)

Milestone 92's investigation (`docs/development/
m92-real-dataset-ingestion.md`) found that `forge.data.Normalize` alone
lets the whole workflow *run* end-to-end without error, but is silently
wrong: a value of `0` in `Insulin` is standardized as if zero insulin were
a real, extreme-low measurement, not flagged as "not measured." Manually
replacing those sentinel zeros with a training-set median *before* Forge
ever sees the array also runs -- but that step has no representation in
`forge.data.Normalize`, so it cannot be part of `preprocessing=`, and is
silently lost the moment the model is saved: a fresh raw inference row
would be standardized with the sentinel intact, giving a **different
predicted class** than the exact same patient's data would get if the
developer remembered to reimplement the imputation constants by hand (see
that doc's reproduced example -- the same row flips from `no_diabetes`
100% to `diabetes` 70% purely based on whether imputation happened).

That is the real, demonstrated Milestone 92 fix: `forge.data.ReplaceValue`
(`forge/data/transforms.py`), a small persistable transform that replaces a
sentinel value with a fixed per-column fill on specific columns -- fit once
on the training split's non-zero medians, then composed with `Normalize`
into one `Compose([ReplaceValue(...), Normalize(...)])` pipeline that is
saved as `preprocessing=` and therefore applies identically and
automatically to every future raw inference row, exactly like `Normalize`
already does on its own.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from forge import Tensor
from forge.data import Compose, Normalize, ReplaceValue, Subset, TensorDataset, random_split

_DATA_PATH = Path(__file__).parent / "data" / "diabetes.csv"

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
N_FEATURES = len(FEATURE_NAMES)
CLASS_NAMES = ["no_diabetes", "diabetes"]
N_CLASSES = len(CLASS_NAMES)

# Columns where the source data encodes "not measured" as a literal 0 --
# see the module docstring's zero-count table. Pregnancies (index 0) is
# deliberately excluded: 0 is a real, valid value there.
_MISSING_SENTINEL = 0.0
_SENTINEL_COLUMNS = [FEATURE_NAMES.index(name) for name in
                     ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")]


def load_raw(path: "str | Path | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """Parse the bundled CSV into `(X, y)` NumPy arrays -- ordinary stdlib `csv` + NumPy, no Forge involved yet.

    `X`: `(768, 8)` float32, column order matching `FEATURE_NAMES`. `y`:
    `(768,)` int64, `0`/`1`. This is the "manual CSV parsing -> NumPy
    conversion" step the Milestone 92 brief's Section 4 explicitly
    anticipates as reasonable, ordinary developer work -- not something
    Forge needs a CSV-reading feature for; `numpy`/`csv` already make it a
    ~10-line function.
    """
    csv_path = Path(path) if path is not None else _DATA_PATH
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if header[:-1] != FEATURE_NAMES or header[-1] != "Outcome":
        raise ValueError(f"Unexpected diabetes.csv header {header!r}; expected {FEATURE_NAMES + ['Outcome']!r}.")

    data = np.array(rows, dtype=np.float32)
    X = data[:, :-1]
    y = data[:, -1].astype(np.int64)
    return X, y


def make_datasets(
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> "tuple[Subset, Subset, Subset, dict[str, Any]]":
    """Build deterministic train/val/test splits with fitted, portable preprocessing.

    Row order in the source CSV carries no known meaning (not sorted by
    outcome -- confirmed by inspection, see `docs/development/
    m92-real-dataset-ingestion.md`), but `random_split` is used anyway
    rather than `sequential_split`, matching `tabular_classification`'s own
    precedent: a random permutation is the only one of the two that
    provides an explicit, checkable representativeness guarantee rather
    than relying on the source file happening to already be shuffled.

    Preprocessing is fit in two stages, both from the training split only
    (never validation/test, avoiding leakage):

    1. `ReplaceValue` fill values -- the training split's per-column median
       computed over its *non-sentinel* entries only (so the sentinel zeros
       themselves don't pull the median toward zero).
    2. `Normalize` mean/std -- computed on the training split *after*
       imputation, so standardization reflects the corrected distribution,
       not one still containing sentinel zeros.

    Returns `(train_ds, val_ds, test_ds, stats)`; `stats["transform"]` is
    the fitted `Compose([ReplaceValue, Normalize])`, reused as-is for
    `preprocessing=` at persistence time.
    """
    X, y = load_raw()
    n = len(X)
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    n_train = n - n_val - n_test

    full_ds_raw = TensorDataset(Tensor(X), Tensor(y))
    split_rng = np.random.default_rng(seed + 1)
    train_split, val_split, test_split = random_split(full_ds_raw, [n_train, n_val, n_test], generator=split_rng)

    train_X = X[train_split.indices]

    fill = []
    for c in _SENTINEL_COLUMNS:
        column = train_X[:, c]
        non_missing = column[column != _MISSING_SENTINEL]
        fill.append(float(np.median(non_missing)))
    replace_missing = ReplaceValue(sentinel=_MISSING_SENTINEL, columns=_SENTINEL_COLUMNS, fill=fill)

    imputed_train_X = train_X.copy()
    for c, fill_value in zip(_SENTINEL_COLUMNS, fill):
        column = imputed_train_X[:, c]
        imputed_train_X[:, c] = np.where(column == _MISSING_SENTINEL, fill_value, column)
    mean = imputed_train_X.mean(axis=0)
    std = imputed_train_X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    normalize = Normalize(mean=mean, std=std)

    transform = Compose([replace_missing, normalize])

    full_ds = TensorDataset(Tensor(X), Tensor(y), transform=transform)
    train_ds = Subset(full_ds, train_split.indices)
    val_ds = Subset(full_ds, val_split.indices)
    test_ds = Subset(full_ds, test_split.indices)

    train_class_counts = np.bincount(y[train_split.indices], minlength=N_CLASSES).tolist()
    majority_baseline = max(np.bincount(y, minlength=N_CLASSES)) / n

    stats = {
        "fill": fill,
        "mean": mean,
        "std": std,
        "transform": transform,
        "train_class_counts": train_class_counts,
        "majority_baseline": majority_baseline,
    }
    return train_ds, val_ds, test_ds, stats


__all__ = [
    "FEATURE_NAMES", "N_FEATURES", "CLASS_NAMES", "N_CLASSES",
    "load_raw", "make_datasets",
]
