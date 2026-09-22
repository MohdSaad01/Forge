"""`forge.train_tabular_classifier_csv()` / `forge.train_tabular_regressor_csv()`: CSV-to-artifact training in one call (Milestone 121).

M118 gave the tabular workflow a CSV *reader* (`forge.data.load_csv()`); M120 taught that reader to select
and order specific feature columns explicitly (`columns=`), so a real production CSV carrying an `id` column
no longer needs to be rewritten by hand first. What both milestones left the caller doing by hand is the
bridge between them:

```text
CSV -> load_csv(..., columns=[...], return_feature_names=True) -> X, y, names -> train_tabular_*(..., feature_names=names)
```

Two calls, in a fixed order, with `names` threaded from the first call into the second -- exactly the sequence
`tests/test_csv_column_selection_training.py` already writes by hand for every one of its tests. These two
functions are that sequence, written once:

```python
result = forge.train_tabular_classifier_csv(
    "patients.csv", target="Outcome", path="patients.forge",
    columns=["Pregnancies", "Glucose", "BMI"],
)
predictor = forge.load_predictor(result.artifact_path)
predictor.predict([[6, 148, 33.6]])[0].label
```

## What this composes (nothing new underneath)

Exactly two existing calls, in this order, and nothing else:

1. `forge.data.load_csv(csv_path, target=target, labels=..., columns=columns, return_feature_names=True)` --
   unmodified (Milestones 118/120): reads the file, removes `target`, selects and orders `columns` (or every
   other column, in file order, when `columns=None`), and returns the resulting feature names alongside the
   arrays.
2. `forge.train_tabular_classifier(X, y, ..., feature_names=names)` /
   `forge.train_tabular_regressor(X, y, ..., feature_names=names)` -- unmodified (Milestones 114/116/119):
   split, preprocessing, the default MLP or a caller's `model=`, training, early stopping, and a verified
   `.forge` artifact, exactly as for any array a caller already had in memory. `names` (the CSV's own header
   cells, in the order `X`'s columns actually ended up in) is threaded straight into `feature_names=` --
   the artifact's persisted schema is the CSV's own column names, with no second place a caller has to repeat
   them.

There is no new CSV parser, no new preprocessing pipeline, no new training loop and no new artifact format
here -- every failure this can raise is one of those two functions' own documented exceptions, under their
own names (`load_csv()`'s `DataError` for anything wrong with the file or the column selection;
`train_tabular_classifier()`/`train_tabular_regressor()`'s `DataError`/`TrainerError`/`PersistenceError` for
anything wrong with the resulting arrays, a caller's `model=`, or `path`). This module adds no validation of
its own beyond Python's normal `TypeError` for an unrecognized keyword argument.

## Column selection and the target (Milestones 118/120, reused unchanged)

`columns=None` (the default): every column except `target` becomes a feature, in file order -- the M118
default, unchanged. `columns=[...]`: **exactly** those header columns become the features, in **exactly**
that order, and every other column (an `id`, a timestamp, anything not selected) is never read -- the M120
contract, unchanged. `target` may never appear in `columns` -- listing it is a `DataError` naming it
(`load_csv()`'s own check); the target is always removed from the features automatically and never has to be
excluded by hand. There is still no automatic id/timestamp/target detection anywhere in this module: selection
is exactly as explicit as `load_csv()` already makes it.

## Feature names are automatic, not a second argument (Milestone 119, reused unchanged)

`load_csv(..., return_feature_names=True)` already returns the exact names of `X`'s columns, in `X`'s own
column order -- selected-and-reordered by `columns=` when given. This module always passes `return_feature_names=True`
and threads the result straight into `feature_names=` on the underlying trainer: a caller of
`train_tabular_classifier_csv()`/`train_tabular_regressor_csv()` never writes `feature_names=` themselves, and
there is no way to *omit* the artifact's schema when training from a CSV through this door (an array caller
using `train_tabular_classifier()`/`train_tabular_regressor()` directly still opts in with `feature_names=`,
unchanged). The persisted names are exactly what M119's alignment (`_column_order()`, in
`forge/training/inference.py`, untouched by this milestone) later checks a named CSV or array against --
proven directly by `tests/test_train_tabular_csv.py`'s reordered-CSV-predicts-identically tests.

## Everything else is exactly the array API (Milestones 114/116/119, reused unchanged)

Every keyword `train_tabular_classifier()`/`train_tabular_regressor()` documents -- `classes`, `epochs`,
`batch_size`, `learning_rate`, `val_fraction`, `missing_columns`, `missing_value`, `patience`, `model`,
`device`, `seed`, `verbose`, and (regression only) `target_transform` -- is a keyword here too, with the
identical default, forwarded unchanged. Nothing here reimplements the split, the preprocessing, the default
MLP, early stopping, or the M116 target-transform mechanism; see `forge/training/tabular.py`'s own module
docstring for what each of those keywords means. This module cannot silently drift from the array API's
defaults because it never states them twice -- see `tests/test_train_tabular_csv.py::
test_csv_wrapper_defaults_match_the_array_trainer_defaults`, which compares the two signatures directly.

## Return value

Exactly `train_tabular_classifier()`'s/`train_tabular_regressor()`'s own `TabularClassificationResult`/
`TabularRegressionResult` -- there is no third result type. `result.features` is the number of *selected*
feature columns (never the file's total column count); `result.artifact_path` is `path`, verified exactly as
for any array-sourced artifact.

## Not here

No DataFrame, no categorical/missing-value handling beyond the existing `missing_columns=` sentinel mechanism,
no automatic column-kind detection, no new artifact format, no CLI (that is `forge model train`, a thin
adapter over these two functions -- `forge/cli/model.py`), and no way to train from anything but a `.csv`
file: an in-memory array is still `train_tabular_classifier()`/`train_tabular_regressor()` directly. See
`docs/development/m121-tabular-csv-training.md`.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ..backend.device import Device
from ..data.csv_reader import load_csv
from ..nn.module import Module
from .tabular import TabularClassificationResult, TabularRegressionResult, train_tabular_classifier, train_tabular_regressor

_CLASSIFIER_CSV = "train_tabular_classifier_csv"
_REGRESSOR_CSV = "train_tabular_regressor_csv"


def train_tabular_classifier_csv(
    csv_path: "str | os.PathLike",
    *,
    target: str,
    path: "str | os.PathLike",
    columns: "Sequence[str] | None" = None,
    classes: "Sequence[str] | None" = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    missing_columns: "Sequence[int] | None" = None,
    missing_value: float = 0.0,
    patience: "int | None" = 10,
    model: "Module | None" = None,
    device: "str | Device | None" = None,
    seed: int = 0,
    verbose: bool = False,
) -> TabularClassificationResult:
    """Train a tabular classifier directly from a CSV file and save it as a portable artifact, in one call.

    ```python
    result = forge.train_tabular_classifier_csv(
        "patients.csv", target="Outcome", path="patients.forge",
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                 "DiabetesPedigreeFunction", "Age"],
        classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5],
    )
    predictor = forge.load_predictor(result.artifact_path)
    predictor.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])[0].label
    ```

    `forge.data.load_csv(csv_path, target=target, labels=True, columns=columns, return_feature_names=True)`
    (Milestones 118/120, unmodified) followed by `forge.train_tabular_classifier(X, y, ...,
    feature_names=names)` (Milestones 114/119, unmodified) -- see this module's own docstring for exactly what
    that means and what is (and is not) validated here.

    - `csv_path` -- the CSV file: a header row, comma-delimited UTF-8 (module `forge.data.csv_reader`'s
      contract). `target` -- the header name of the target column; removed from the features automatically
      and rejected if also listed in `columns`.
    - `columns` -- `None` (default): every column except `target` is a feature, in file order. Otherwise
      (Milestone 120): exactly those header columns, in exactly that order; every other column (an `id`, say)
      is never read.
    - `path` -- where the `.forge` artifact is written (the *output*, distinct from `csv_path`, the *input*).
    - Every other keyword -- `classes`, `epochs`, `batch_size`, `learning_rate`, `val_fraction`,
      `missing_columns`, `missing_value`, `patience`, `model`, `device`, `seed`, `verbose` -- means exactly
      what it means on `forge.train_tabular_classifier()`, with the identical default, and is forwarded to it
      unchanged; see that function's own docstring. There is no `feature_names=` here: the CSV's own selected
      column names are threaded through automatically (module docstring, **Feature names are automatic**).

    Returns a `TabularClassificationResult` -- exactly what `train_tabular_classifier()` itself returns.
    Raises `forge.DataError` from `load_csv()` (a malformed file, an unknown/duplicate/target-overlapping
    `columns` entry, ...) or from `train_tabular_classifier()` (bad arguments, non-finite data, too few
    classes, ...), `forge.TrainerError` (a bad `model=`), or `forge.PersistenceError` (a bad `path`) -- all
    before epoch 1, all under the raising function's own name, exactly as a caller who wrote the two calls by
    hand would see.
    """
    X, y, names = load_csv(csv_path, target=target, labels=True, columns=columns, return_feature_names=True)
    return train_tabular_classifier(
        X, y,
        path=path,
        classes=classes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        missing_columns=missing_columns,
        missing_value=missing_value,
        patience=patience,
        model=model,
        device=device,
        seed=seed,
        verbose=verbose,
        feature_names=names,
    )


def train_tabular_regressor_csv(
    csv_path: "str | os.PathLike",
    *,
    target: str,
    path: "str | os.PathLike",
    columns: "Sequence[str] | None" = None,
    epochs: int = 500,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    missing_columns: "Sequence[int] | None" = None,
    missing_value: float = 0.0,
    patience: "int | None" = 30,
    model: "Module | None" = None,
    device: "str | Device | None" = None,
    seed: int = 0,
    verbose: bool = False,
    target_transform: "str | None" = None,
) -> TabularRegressionResult:
    """Train a tabular regressor directly from a CSV file and save it as a portable artifact, in one call.

    ```python
    result = forge.train_tabular_regressor_csv(
        "housing.csv", target="median_house_value", path="housing.forge",
        columns=["median_income", "housing_median_age", "total_rooms"],
        target_transform="standardize",
    )
    predictor = forge.load_predictor(result.artifact_path)
    predictor.predict([[8.3, 41.0, 880.0]])   # native units, not a z-score
    ```

    Same contract as `train_tabular_classifier_csv()` (see it and this module's own docstring), except the
    target column is read as numbers (`load_csv(..., labels=False)`) and every keyword --
    `epochs`, `batch_size`, `learning_rate`, `val_fraction`, `missing_columns`, `missing_value`, `patience`,
    `model`, `device`, `seed`, `verbose`, `target_transform` -- means exactly what it means on
    `forge.train_tabular_regressor()`, with the identical default (`epochs=500`, `patience=30`), forwarded
    unchanged. There is no `classes=` and no `feature_names=` (the latter for the same reason as the
    classifier: the CSV's own selected column names are threaded through automatically).

    `target_transform="standardize"` (Milestone 116) is passed straight through to
    `train_tabular_regressor()` -- training happens on standardized targets, fitted on the training split only,
    and the fitted `forge.data.StandardizeTarget` is saved in the artifact so `predict()`/`evaluate()` and
    every number in the returned result stay in native units, exactly as for an array caller. This module does
    not scale the target itself.

    Returns a `TabularRegressionResult` -- exactly what `train_tabular_regressor()` itself returns.
    """
    X, y, names = load_csv(csv_path, target=target, labels=False, columns=columns, return_feature_names=True)
    return train_tabular_regressor(
        X, y,
        path=path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        missing_columns=missing_columns,
        missing_value=missing_value,
        patience=patience,
        model=model,
        device=device,
        seed=seed,
        verbose=verbose,
        target_transform=target_transform,
        feature_names=names,
    )


__all__ = ["train_tabular_classifier_csv", "train_tabular_regressor_csv"]
