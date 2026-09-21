"""`forge.train_tabular_classifier()` / `forge.train_tabular_regressor()`: rows-of-numbers-to-artifact convenience calls (Milestone 114).

Two ordinary questions about a table of numbers -- *which class does this row
belong to?* and *what continuous value should this row predict?* -- were
already answerable with Forge's existing public API (`examples/tabular_diabetes`,
`examples/regression`), but only by hand-assembling the whole pipeline: split
the rows, fit preprocessing on the training rows only, build `Dataset`s and
`DataLoader`s, pick a model, a loss and an optimizer, train, save, verify, and
remember to persist the *fitted* preprocessing so a later raw row is prepared
the way training rows were. These two functions are that pipeline, written
once, for numeric arrays a developer already has in memory:

```python
result = forge.train_tabular_classifier(X, y, path="model.forge")
predictor = forge.load_predictor(result.artifact_path)
predictor.predict(X_new)               # raw rows in; preprocessing is inside the artifact
predictor.evaluate(X_test, y_test)     # M113
```

`X` is a numeric `(samples, features)` array-like, `y` the targets. There is
no DataFrame support and no new data abstraction: these functions take arrays,
never a path to a data file. Read a CSV with `forge.data.load_csv()` (Milestone
118: a numeric CSV in, plain `(X, y)` arrays out) or with ordinary Python/NumPy,
and pass the arrays.

## What this composes (nothing new underneath)

Exactly the existing stack, in this order:

1. Validate everything a caller can get wrong (see **Preflight**).
2. `forge.data.random_split()` the rows into train/validation, seeded.
3. Fit preprocessing on the **training rows only** -- `Normalize` (feature
   standardisation), optionally preceded by `ReplaceValue` (see **Preprocessing**)
   -- and apply it to both splits. The fitted `Compose` is the very object saved
   as the artifact's `preprocessing=`, so training and inference cannot drift.
4. `TensorDataset`/`DataLoader`, a default MLP of `Linear`/`ReLU` (or the
   caller's own `model=`), `CrossEntropyLoss` or `MSELoss`, `Adam`.
5. `forge.train_and_save()`, unmodified -- no second training loop, no second
   persistence path -- with `task="tabular_classification"` / `"regression"`.
6. `forge.load_predictor(path).evaluate()` (M113), unmodified, on the saved
   artifact for the numbers in the result.

## Preflight: what is rejected before epoch 1

Every failure below is a clear `forge` exception raised before any training:

- `X` not numeric or not 2-D; `len(X) != len(y)`; any NaN/Inf in `X` or in a
  regression `y` (`DataError`). Missing values must be handled by the caller
  before calling: fill or drop them. Forge never trains through NaN.
- Classification labels that are not a set of >= 2 class names/indices; a class
  that has no rows in the training split (`DataError`).
- `model=` that is not a `forge.nn.Module`, cannot process a `(rows, features)`
  batch, or whose output width is not the number of classes / target width. This
  is checked by running the model once on real (preprocessed) rows, not by
  guessing from its structure (`TrainerError`).
- Arguments out of range (`epochs`, `batch_size`, `learning_rate`, `val_fraction`,
  `patience`, `seed`, `missing_*`) (`DataError`).
- The output `path`: its directory must exist and be writable, its filename must
  be valid, and the model, fitted preprocessing and class list must serialise. All
  of it is proven by really saving the *untrained* model to a temporary sibling
  file (removed immediately) through the same `save_model()` the final save uses;
  the final artifact is not written until training has finished (`PersistenceError`).

## Preprocessing

`Normalize(mean, std)` per feature, fitted on the training split. A constant
(zero-variance) training feature gets `std = 1`, so it becomes 0 instead of
dividing by zero. Targets are **not** scaled or transformed unless a regression
caller asks for it (see **Target transform**).

`missing_columns=[...]` (with `missing_value=`, default `0.0`) opts specific
columns into `ReplaceValue`: entries equal to the sentinel are replaced with the
training split's median of that column's non-sentinel entries *before*
standardisation. This is the existing Pima-diabetes mechanism (`0` means "not
measured" in five of its columns); it is off by default because a sentinel
that means "missing" in one column is a legitimate value in another. It is not
a general missing-value framework: NaN is not a sentinel (`ReplaceValue` cannot
express it) and is rejected as above.

## Target transform (regression only, opt-in; Milestone 116)

`train_tabular_regressor(..., target_transform="standardize")` trains on
`z = (y - mean) / std` instead of `y`, with `mean`/`std` fitted **per target column on
the training split only** (the validation rows, which early stopping and the
result's validation numbers use, are transformed with those training statistics and
never contribute to them). The fitted `forge.data.StandardizeTarget` is saved in the
artifact next to the preprocessing, so the artifact -- not the caller's memory --
converts back: `predict()` and `evaluate()` speak the caller's native units, and
every `train_*`/`validation_*`/`baseline_mse` number in the result is a native-unit
number computed on the saved artifact, exactly as without a transform. Why it exists
(a target of magnitude ~2x10^5 needs 500 epochs to reach R^2 0.665 where standardised
it reaches 0.746 in ~130) and why it is an artifact-level output transform rather than
a rescaling of the last `Linear` layer: `docs/development/m116-persisted-target-transforms.md`.

Off by default (`target_transform=None`): the default pipeline and the artifacts it
writes are byte-for-byte what they were before this milestone. A constant target
column has no scale to standardise by and is a `DataError`, not a silent divide.
Anything that reads `result.model` / `result.history` directly sees the *training*
space (z-scores); only the artifact and the result's metrics are in native units.

## Validation split

`val_fraction` (default `0.2`) of the rows are held out with `random_split`, seeded
by `seed`. There is no `validation_data=`: for i.i.d. rows the random split is
the right tool, and a caller with a dedicated held-out set trains on all the rows
they want and scores the held-out set with `predictor.evaluate()`. (An explicit
validation set matters for ordered data such as windowed time series, which this
milestone does not address.) No stratification: a class that ends up with no
training rows is rejected rather than silently untrainable.

## Early stopping

On by default (`patience=10`): training stops once validation loss has not
improved for 10 epochs and the best epoch's weights are what gets saved
(`EarlyStopping(patience=..., restore_best=True)`, unmodified). `epochs` is still
the upper bound. `patience=None` runs every epoch and saves the last one.

## Result numbers

`train_*`/`validation_*` in the results are computed by `evaluate()` on the
*saved artifact* -- the model that is actually on disk -- against the raw training
and validation rows, so they are comparable to a later
`predictor.evaluate(X_test, y_test)` and include the M113 baselines (majority-class
share of the validation rows; MSE of always predicting the validation mean). They
are not the running per-epoch training-pass numbers in `result.history`, which
are computed while the weights are still changing.

## Not here

No hyperparameter search, schedulers, callbacks, DataFrame or file handling (a CSV is
read into arrays by `forge.data.load_csv()` before the call),
target transforms other than `"standardize"`, time-series behaviour or CLI. Anything else: build the pipeline
from `train_and_save()` directly, exactly as `examples/tabular_diabetes` does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .. import random as forge_random
from ..backend.device import Device
from ..data.dataloader import DataLoader
from ..data.dataset import TensorDataset, random_split
from ..data.target_transform import TARGET_TRANSFORM_TYPES, StandardizeTarget
from ..data.transforms import Compose, Normalize, ReplaceValue
from ..exceptions import DataError, TrainerError
from ..nn.activation import ReLU
from ..nn.container import Sequential
from ..nn.linear import Linear
from ..nn.loss import CrossEntropyLoss, MSELoss
from ..nn.module import Module
from ..optim.adam import Adam
from ..tensor.tensor import Tensor
from ._preflight import check_model_contract, preflight_save
from .api import TrainingResult, train_and_save
from .early_stopping import EarlyStopping
from .evaluation import encode_class_labels, encode_regression_targets
from .inference import load_predictor
from .metrics import Accuracy, MeanAbsoluteError, MeanSquaredError

_CLASSIFIER = "train_tabular_classifier"
_REGRESSOR = "train_tabular_regressor"

# Hidden widths of the default MLPs. Classification: the Pima-diabetes model's
# widths (a small, noisy dataset overfits a wider net); regression: the widths of
# `examples/regression`. See `docs/development/m114-tabular-workflows.md` for the
# comparison against the alternatives that fixed them.
_CLASSIFIER_HIDDEN = (32, 16)
_REGRESSOR_HIDDEN = (64, 32)


def _default_mlp(features: int, outputs: int, hidden: "tuple[int, int]") -> Sequential:
    first, second = hidden
    return Sequential(
        Linear(features, first),
        ReLU(),
        Linear(first, second),
        ReLU(),
        Linear(second, outputs),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_int(value: Any, name: str, fn: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise DataError(f"{fn}() {name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _validate_arguments(
    fn: str, path: Any, epochs: Any, batch_size: Any, learning_rate: Any, val_fraction: Any,
    patience: Any, seed: Any, missing_columns: Any, missing_value: Any,
) -> "tuple[str, list[int] | None]":
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path:
        raise DataError(f"{fn}() path must be a non-empty file path (str or os.PathLike), got {path!r}.")
    _require_int(epochs, "epochs", fn)
    _require_int(batch_size, "batch_size", fn)
    _require_int(seed, "seed", fn, minimum=0)
    if patience is not None:
        _require_int(patience, "patience", fn)
    if (
        isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float, np.number))
        or not np.isfinite(learning_rate) or learning_rate <= 0
    ):
        raise DataError(f"{fn}() learning_rate must be a finite number > 0, got {learning_rate!r}.")
    if (
        isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float, np.number))
        or not 0.0 < val_fraction < 1.0
    ):
        raise DataError(f"{fn}() val_fraction must be strictly between 0 and 1, got {val_fraction!r}.")

    columns = None
    if missing_columns is not None:
        try:
            columns = [_require_int(c, "missing_columns entries", fn, minimum=0) for c in missing_columns]
        except TypeError:
            raise DataError(
                f"{fn}() missing_columns must be a sequence of column indices, got {missing_columns!r}."
            ) from None
        if not columns or len(set(columns)) != len(columns):
            raise DataError(
                f"{fn}() missing_columns must be a non-empty list of unique column indices, "
                f"got {missing_columns!r}."
            )
    if (
        isinstance(missing_value, bool) or not isinstance(missing_value, (int, float, np.number))
        or not np.isfinite(missing_value)
    ):
        raise DataError(f"{fn}() missing_value must be a finite number, got {missing_value!r}.")
    return path, columns


def _validate_target_transform(value: Any, fn: str) -> "str | None":
    if value is None:
        return None
    if not isinstance(value, str) or value not in TARGET_TRANSFORM_TYPES:
        raise DataError(
            f"{fn}() target_transform must be None or one of {TARGET_TRANSFORM_TYPES!r}, got {value!r}."
        )
    return value


def _to_host_array(data: Any, name: str, fn: str) -> np.ndarray:
    if isinstance(data, Tensor):
        data = data.to("cpu").numpy()
    try:
        return np.asarray(data)
    except ValueError as exc:  # e.g. ragged nested lists
        raise DataError(f"{fn}() could not interpret {name} as an array: {exc}") from exc


def _require_all_finite(values: np.ndarray, name: str, fn: str, what: str) -> None:
    bad = ~np.isfinite(values)
    if bad.any():
        first = tuple(int(i) for i in np.argwhere(bad)[0])
        raise DataError(
            f"{fn}() {name} contains {int(bad.sum())} non-finite value(s) (NaN/Inf), first at index "
            f"{first}. Forge does not train on NaN/Inf {what}: fill or drop them before calling "
            "(e.g. replace NaN with a per-column median, or drop those rows)."
        )


def _coerce_features(X: Any, fn: str) -> np.ndarray:
    """`X` as a finite `(samples, features)` float32 array, or `DataError`."""
    features = _to_host_array(X, "X", fn)
    if features.dtype.kind not in ("f", "i", "u"):
        raise DataError(f"{fn}() requires X to be numeric, got dtype '{features.dtype}'.")
    if features.ndim != 2:
        raise DataError(
            f"{fn}() requires X to be 2-D (samples, features), got shape {features.shape}. "
            "A single row is [[...]], not [...]."
        )
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise DataError(f"{fn}() requires a non-empty X, got shape {features.shape}.")
    if features.dtype.kind == "f":
        _require_all_finite(features, "X", fn, "features")
    with np.errstate(over="ignore"):  # an overflow is reported just below, with its position
        features = features.astype(np.float32)
    _require_all_finite(features, "X", fn, "features")  # float64 values too large for float32
    return features


def _coerce_class_targets(
    y: Any, n_samples: int, classes: "Sequence[str] | None", fn: str,
) -> "tuple[np.ndarray, list[str]]":
    """`(labels, classes)`: `labels` `(n,)` `int64` indices into the deterministic class list.

    Follows `ArtifactPredictor.evaluate()`'s own label convention so a saved
    artifact is scored with the same `y`: class **names** (`str`) or integer class
    **indices**. Names are sorted (unless `classes=` fixes the order); indices must
    then be exactly `0..K-1`. Integer-valued floats and booleans (what
    `np.loadtxt` and a CSV column of 0/1 give) are accepted as indices.
    """
    labels = _to_host_array(y, "y", fn)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise DataError(
            f"{fn}() requires y to be 1-D (one class label per sample), got shape {labels.shape}."
        )
    if labels.shape[0] != n_samples:
        raise DataError(f"{fn}() X has {n_samples} sample(s) but y has {labels.shape[0]}.")

    kind = labels.dtype.kind
    if kind == "O":
        if not all(isinstance(v, str) for v in labels.tolist()):
            raise DataError(f"{fn}() y must be class names (str) or integer class indices, got mixed/object values.")
        labels = labels.astype(str)
    elif kind == "S":
        labels = labels.astype(str)
    elif kind == "b":
        labels = labels.astype(np.int64)
    elif kind == "f":
        _require_all_finite(labels, "y", fn, "class labels")
        if not np.array_equal(labels, np.floor(labels)):
            raise DataError(
                f"{fn}() y holds non-integer floats, which are not class labels. Class labels are "
                "names (str) or integer class indices; for a continuous target use "
                "train_tabular_regressor()."
            )
        labels = labels.astype(np.int64)
    elif kind not in ("U", "i", "u"):
        raise DataError(f"{fn}() y must be class names (str) or integer class indices, got dtype '{labels.dtype}'.")

    is_names = labels.dtype.kind == "U"
    if classes is None:
        if is_names:
            class_list = sorted(set(labels.tolist()))
        else:
            if labels.min() < 0 or set(np.unique(labels).tolist()) != set(range(int(labels.max()) + 1)):
                raise DataError(
                    f"{fn}() integer labels must be exactly the class indices 0..K-1 (each used at least "
                    f"once), got values {np.unique(labels).tolist()[:12]!r}. Pass class names (str) or "
                    "classes=[...] to name them."
                )
            class_list = [str(i) for i in range(int(labels.max()) + 1)]
    else:
        if isinstance(classes, (str, bytes)) or not isinstance(classes, (list, tuple)) or not all(
            isinstance(c, str) and c.strip() for c in classes
        ):
            raise DataError(f"{fn}() classes must be a list of non-empty strings, got {classes!r}.")
        class_list = list(classes)
        if len(set(class_list)) != len(class_list):
            raise DataError(f"{fn}() classes must not contain duplicates, got {class_list!r}.")

    if len(class_list) < 2:
        raise DataError(f"{fn}() needs at least 2 classes, found {len(class_list)}: {class_list!r}.")
    indices = encode_class_labels(labels, class_list, fn)
    unused = [class_list[i] for i in np.flatnonzero(np.bincount(indices, minlength=len(class_list)) == 0)]
    if unused:
        raise DataError(f"{fn}() class(es) {unused!r} have no rows in y; every class must occur at least once.")
    return indices, class_list


def _coerce_regression_targets(y: Any, n_samples: int, fn: str) -> np.ndarray:
    """`y` as a finite `(n, outputs)` float64 array (`(n,)` becomes a single column).

    Kept in float64 -- the caller's own precision -- so the result's numbers are
    exactly what a later `predictor.evaluate(X, y)` on the same rows reports;
    training casts to float32 itself.
    """
    raw = _to_host_array(y, "y", fn)
    if raw.dtype.kind == "f":
        _require_all_finite(raw, "y", fn, "targets")  # before encode_regression_targets' evaluate()-worded message
    targets = encode_regression_targets(raw, fn)  # numeric, 1-D/2-D, non-empty
    if targets.shape[0] != n_samples:
        raise DataError(f"{fn}() X has {n_samples} sample(s) but y has {targets.shape[0]}.")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.shape[1] == 0:
        raise DataError(f"{fn}() requires y to have at least one target column, got shape {targets.shape}.")
    with np.errstate(over="ignore"):
        as_float32 = targets.astype(np.float32)
    _require_all_finite(as_float32, "y", fn, "targets")  # float64 values too large for float32
    return targets


# ---------------------------------------------------------------------------
# Split + preprocessing
# ---------------------------------------------------------------------------


@dataclass
class _Prepared:
    """One split of the rows with preprocessing fitted on its training half only."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    transform: Compose
    train_features: np.ndarray  # preprocessed, model-ready float32
    val_features: np.ndarray


def _split_and_preprocess(
    X: np.ndarray, targets: np.ndarray, val_fraction: float, seed: int,
    missing_columns: "list[int] | None", missing_value: float, fn: str,
) -> _Prepared:
    n_samples, n_features = X.shape
    n_val = max(1, round(n_samples * val_fraction))
    n_train = n_samples - n_val
    if n_train < 2:
        raise DataError(
            f"{fn}() needs at least 3 samples to hold out a validation split, got {n_samples} "
            f"(val_fraction={val_fraction!r} leaves {n_train} for training)."
        )

    # Split first: nothing below sees a validation row.
    train_subset, val_subset = random_split(
        TensorDataset(Tensor(X), Tensor(targets)), [n_train, n_val], generator=np.random.default_rng(seed),
    )
    train_indices = np.asarray(train_subset.indices, dtype=np.int64)
    val_indices = np.asarray(val_subset.indices, dtype=np.int64)
    train_X = X[train_indices]

    steps = []
    if missing_columns is not None:
        out_of_range = [c for c in missing_columns if c >= n_features]
        if out_of_range:
            raise DataError(f"{fn}() missing_columns {out_of_range!r} out of range for {n_features} feature(s).")
        fill = []
        for c in missing_columns:
            present = train_X[:, c][train_X[:, c] != np.float32(missing_value)]
            if present.size == 0:
                raise DataError(
                    f"{fn}() column {c} is entirely {missing_value!r} in the training split, so there is "
                    "nothing to compute its replacement value from."
                )
            fill.append(float(np.median(present)))
        steps.append(ReplaceValue(sentinel=missing_value, columns=missing_columns, fill=fill))

    imputed = Compose(steps)(Tensor(train_X)) if steps else Tensor(train_X)
    imputed_np = imputed.numpy().astype(np.float64)
    mean = imputed_np.mean(axis=0)
    std = imputed_np.std(axis=0)
    # A constant feature (std ~ 0, allowing for float rounding) is centred, not scaled.
    std = np.where(std > 1e-10 * np.maximum(1.0, np.abs(mean)), std, 1.0)
    steps.append(Normalize(mean=mean.tolist(), std=std.tolist()))
    transform = Compose(steps)

    train_features = transform(Tensor(train_X)).numpy()
    val_features = transform(Tensor(X[val_indices])).numpy()
    for name, values in (("training", train_features), ("validation", val_features)):
        if not np.isfinite(values).all():
            raise DataError(
                f"{fn}() preprocessing produced non-finite {name} features (feature values are too "
                "large to standardise in float32). Rescale X before calling."
            )
    return _Prepared(train_indices, val_indices, transform, train_features, val_features)


# ---------------------------------------------------------------------------
# Model resolution (the save preflight is in `_preflight.py`)
# ---------------------------------------------------------------------------


def _resolve_model(
    model: "Module | None", features: int, outputs: int, hidden: "tuple[int, int]",
    device: "str | Device | None", seed: int, probe_rows: Tensor, output_meaning: str, fn: str,
) -> Module:
    """The model to train, moved to its device, contract-checked if caller-supplied."""
    supplied = model is not None
    if supplied:
        if not isinstance(model, Module):
            raise TrainerError(f"{fn}() requires model= to be a forge.nn.Module, got {type(model).__name__}.")
    else:
        forge_random.seed(seed)
        model = _default_mlp(features, outputs, hidden)

    resolved = Device.parse(device) if device is not None else (model.device or Device.parse("cpu"))
    model.to(resolved)

    if supplied:
        check_model_contract(
            model, probe_rows, outputs, "features", f"(batch, {features})", output_meaning, fn,
        )
    return model


def _early_stopping(patience: "int | None") -> "EarlyStopping | None":
    return EarlyStopping(patience=patience, restore_best=True) if patience is not None else None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularClassificationResult:
    """`train_tabular_classifier()`'s return value (Milestone 114).

    ```python
    result.classes                       # ['no_diabetes', 'diabetes']
    result.validation_accuracy, result.baseline_accuracy
    result.artifact_path                 # feed to forge.load_predictor()
    ```

    - `task` -- `"tabular_classification"`, the task saved in the artifact.
    - `samples`/`features`/`train_samples`/`validation_samples` -- row counts (all
      rows; the two splits) and the feature count.
    - `classes` -- the class names in output order (`classes[i]` is model output
      `i`), also persisted in the artifact. Sorted names, or `0..K-1` as strings,
      or exactly the `classes=` given -- never discovery order.
    - `train_loss`/`train_accuracy`/`validation_loss`/`validation_accuracy` --
      cross-entropy loss and accuracy of the **saved model** on each split, from
      `ArtifactPredictor.evaluate()` (see the module docstring).
    - `baseline_accuracy` -- the share of the majority class **in the validation
      rows**: what always predicting one class scores on exactly those rows (the
      M113 definition). `validation_accuracy` is only evidence of learning if it
      beats this.
    - `epochs_completed`/`stopped_early`/`best_epoch`/`best_monitored_value` --
      from training; `best_epoch`/`best_monitored_value` are `None` when
      `patience=None`. With early stopping the saved weights are those of
      `best_epoch`.
    - `artifact_path` -- the verified `.forge` file.
    - `history` -- the underlying `TrainingResult` (per-epoch curves); `model` --
      the freshly reloaded, verified `Module`.
    """

    task: str
    samples: int
    features: int
    train_samples: int
    validation_samples: int
    classes: "list[str]"
    epochs_completed: int
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    baseline_accuracy: float
    artifact_path: str
    stopped_early: bool
    best_epoch: "int | None"
    best_monitored_value: "float | None"
    history: TrainingResult
    model: Module


@dataclass(frozen=True)
class TabularRegressionResult:
    """`train_tabular_regressor()`'s return value (Milestone 114).

    ```python
    result.validation_mse, result.baseline_mse
    result.validation_mae                # in the target's own units
    ```

    Fields mirror `TabularClassificationResult`, with `outputs` (target columns) in
    place of `classes`. `train_*`/`validation_*` are of the **saved model** on each
    split, from `ArtifactPredictor.evaluate()`; `*_loss` is the `MSELoss` of that
    evaluation -- the same quantity as `*_mse` up to float32 rounding. Targets are
    unscaled, so MSE/MAE are in the target's own (squared) units.
    `baseline_mse` is the MSE of always predicting the **validation** targets' own
    per-output mean (the M113 definition); `validation_mse` is only evidence of
    learning if it is well below this.

    With `target_transform="standardize"` (Milestone 116) `target_transform` is the
    fitted `forge.data.StandardizeTarget` (else `None`). Every `*_mse`/`*_mae`/
    `*_loss`/`baseline_mse` above is still in native units -- computed by the saved
    artifact's `evaluate()` on native targets. What is *not* native: `history` (the
    per-epoch curves), `best_monitored_value` (early stopping's validation loss) and
    `model` (the bare reloaded `Module`, whose output is the training space) -- they
    are in z-scores; use `forge.load_predictor(result.artifact_path)` for predictions.
    """

    task: str
    samples: int
    features: int
    outputs: int
    train_samples: int
    validation_samples: int
    epochs_completed: int
    train_loss: float
    validation_loss: float
    train_mse: float
    validation_mse: float
    train_mae: float
    validation_mae: float
    baseline_mse: float
    artifact_path: str
    stopped_early: bool
    best_epoch: "int | None"
    best_monitored_value: "float | None"
    history: TrainingResult
    model: Module
    target_transform: "StandardizeTarget | None" = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def train_tabular_classifier(
    X: Any,
    y: Any,
    *,
    path: "str | os.PathLike",
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
    """Train a classifier on a numeric feature matrix and save it as a portable artifact, in one call.

    ```python
    result = forge.train_tabular_classifier(
        X, y, path="diabetes.forge",
        classes=["no_diabetes", "diabetes"],      # y holds 0/1
        missing_columns=[1, 2, 3, 4, 5],          # 0 means "not measured" here
    )
    predictor = forge.load_predictor(result.artifact_path)
    predictor.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])[0].label
    predictor.evaluate(X_test, y_test).accuracy
    ```

    - `X` -- numeric `(samples, features)` array-like (`ndarray`, nested list,
      `Tensor`). NaN/Inf raise `DataError`.
    - `y` -- one label per row: class **names** (`str`) or integer class
      **indices** (integer-valued floats and booleans are accepted). Without
      `classes=`, names are sorted and become the class order; indices must be
      exactly `0..K-1` and the classes are named `"0".."K-1"`. With `classes=[...]`
      the class order is exactly that list (so `y` holds names from it, or indices
      into it). At least 2 classes, each used at least once.
    - `path` -- where the `.forge` artifact is written; its directory must already
      exist.
    - `epochs` (upper bound), `batch_size`, `learning_rate` (Adam), `seed` (split,
      shuffling and the default model's initialisation), `device`
      (`"cpu"`/`"cuda"`), `verbose` (per-epoch lines; off by default -- tabular
      epochs take milliseconds).
    - `val_fraction`, `patience` -- see the module docstring (**Validation split**,
      **Early stopping**). `patience=None` disables early stopping.
    - `missing_columns`/`missing_value` -- opt columns into median replacement of a
      sentinel (module docstring, **Preprocessing**).
    - `model` -- your own `forge.nn.Module` in place of the default MLP
      (`Linear(F,32)-ReLU-Linear(32,16)-ReLU-Linear(16,K)`). It is used as given (not
      reseeded) and must map `(batch, features)` to `(batch, K)` raw class scores;
      that is verified on real rows before epoch 1 (`TrainerError`).

    Loss is `CrossEntropyLoss`, optimizer `Adam`. Everything else -- see the
    module docstring for the full pipeline and what is validated before training.

    Returns a `TabularClassificationResult`. Raises `forge.DataError` (data or
    arguments), `forge.TrainerError` (`model=`), `forge.PersistenceError` (`path`),
    all before epoch 1; `forge.TrainerError` if training itself diverges to NaN/Inf.
    """
    fn = _CLASSIFIER
    path, columns = _validate_arguments(
        fn, path, epochs, batch_size, learning_rate, val_fraction, patience, seed, missing_columns, missing_value,
    )
    features = _coerce_features(X, fn)
    labels, class_list = _coerce_class_targets(y, features.shape[0], classes, fn)

    prepared = _split_and_preprocess(features, labels, val_fraction, seed, columns, missing_value, fn)
    train_labels, val_labels = labels[prepared.train_indices], labels[prepared.val_indices]
    absent = [class_list[i] for i in np.flatnonzero(np.bincount(train_labels, minlength=len(class_list)) == 0)]
    if absent:
        raise DataError(
            f"{fn}() class(es) {absent!r} have no rows in the training split (too few rows for "
            f"val_fraction={val_fraction!r}). Add data for them, lower val_fraction, or drop the class."
        )

    n_features = features.shape[1]
    net = _resolve_model(
        model, n_features, len(class_list), _CLASSIFIER_HIDDEN, device, seed,
        Tensor(prepared.train_features[:2]), f"one raw score per class, for the {len(class_list)} classes {class_list!r}", fn,
    )
    preflight_save(net, path, prepared.transform, class_list, "tabular_classification", fn)

    train_loader = DataLoader(
        TensorDataset(Tensor(prepared.train_features), Tensor(train_labels)),
        batch_size=batch_size, shuffle=True, generator=np.random.default_rng(seed),
    )
    val_loader = DataLoader(TensorDataset(Tensor(prepared.val_features), Tensor(val_labels)), batch_size=batch_size)
    saved = train_and_save(
        net, train_loader,
        loss=CrossEntropyLoss(),
        optimizer=Adam(net.parameters(), lr=learning_rate),
        epochs=epochs,
        validation_dataset=val_loader,
        device=device,
        metrics=[Accuracy()],
        verbose=verbose,
        path=path,
        sample=Tensor(prepared.val_features[:1]),
        preprocessing=prepared.transform,
        classes=class_list,
        task="tabular_classification",
        early_stopping=_early_stopping(patience),
    )

    predictor = load_predictor(path)
    train_eval = predictor.evaluate(features[prepared.train_indices], train_labels)
    val_eval = predictor.evaluate(features[prepared.val_indices], val_labels)
    return TabularClassificationResult(
        task="tabular_classification",
        samples=features.shape[0],
        features=n_features,
        train_samples=len(prepared.train_indices),
        validation_samples=len(prepared.val_indices),
        classes=class_list,
        epochs_completed=saved.history.epochs_completed,
        train_loss=train_eval.loss,
        validation_loss=val_eval.loss,
        train_accuracy=train_eval.accuracy,
        validation_accuracy=val_eval.accuracy,
        baseline_accuracy=val_eval.baseline_accuracy,
        artifact_path=saved.artifact_path,
        stopped_early=saved.stopped_early,
        best_epoch=saved.best_epoch,
        best_monitored_value=saved.best_monitored_value,
        history=saved.history,
        model=saved.model,
    )


def train_tabular_regressor(
    X: Any,
    y: Any,
    *,
    path: "str | os.PathLike",
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
    """Train a regressor on a numeric feature matrix and save it as a portable artifact, in one call.

    ```python
    result = forge.train_tabular_regressor(X, y, path="strength.forge")
    predictor = forge.load_predictor(result.artifact_path)
    predictor.predict([[540.0, 0.0, 0.0, 162.0, 2.5, 1040.0, 676.0, 28.0]])   # Tensor, shape (1, 1)
    predictor.evaluate(X_test, y_test).mse
    ```

    Same contract and arguments as `train_tabular_classifier()` (see it and the
    module docstring), except:

    - `y` -- a numeric, finite target: `(n,)`, `(n, 1)` or `(n, outputs)`. By default
      it is trained on in its own units, and the artifact predicts in them. NaN/Inf
      raise `DataError`. Large targets train worse (M115, 3,000 real California house
      prices: R^2 0.665 in dollars, ~200,000, against 0.75 once `y` was standardised, and
      the 500-epoch default was exhausted) -- pass `target_transform="standardize"` for
      those. Do **not** rescale `y` yourself and train on that: the artifact would then
      predict in the rescaled units and not know how to convert back.
    - `target_transform` -- `None` (default: train on `y` as given) or `"standardize"`:
      train on `(y - mean) / std` with `mean`/`std` fitted per target column on the
      **training split only** and saved in the artifact, so `predict()` and `evaluate()`
      return and compare **native** units and no caller ever inverts anything. A
      constant training target column is a `DataError`. See the module docstring
      (**Target transform**); the result's metrics are native either way.
    - There is no `classes=`.
    - Defaults are `epochs=500`, `patience=30`: an unscaled target starts a
      randomly-initialised network far from the answer and needs longer (on
      Concrete, 200 epochs stopped while validation loss was still falling).
    - Default model: `Linear(F,64)-ReLU-Linear(64,32)-ReLU-Linear(32,outputs)`. A
      `model=` must map `(batch, features)` to `(batch, outputs)` -- that is, its
      last layer must have as many outputs as `y` has columns (1 for an `(n,)`
      target), checked on real rows before epoch 1.
    - Loss is `MSELoss`.

    A regression artifact's `predict()` returns a `(rows, outputs)` `Tensor`.
    Returns a `TabularRegressionResult`.
    """
    fn = _REGRESSOR
    path, columns = _validate_arguments(
        fn, path, epochs, batch_size, learning_rate, val_fraction, patience, seed, missing_columns, missing_value,
    )
    target_transform = _validate_target_transform(target_transform, fn)
    features = _coerce_features(X, fn)
    targets = _coerce_regression_targets(y, features.shape[0], fn)
    n_outputs = targets.shape[1]

    prepared = _split_and_preprocess(
        features, targets.astype(np.float32), val_fraction, seed, columns, missing_value, fn,
    )
    train_targets, val_targets = targets[prepared.train_indices], targets[prepared.val_indices]

    # Fitted on the TRAINING targets only (after the split, like the input preprocessing);
    # the validation targets are transformed with those statistics and never contribute.
    fitted = None
    train_fit, val_fit = train_targets, val_targets  # what the model is trained/early-stopped on
    if target_transform == "standardize":
        try:
            fitted = StandardizeTarget.fit(train_targets)
        except DataError as exc:
            raise DataError(f"{fn}() target_transform='standardize' cannot be fitted: {exc}") from exc
        train_fit, val_fit = fitted.transform(train_targets), fitted.transform(val_targets)
        for name, values in (("training", train_fit), ("validation", val_fit)):
            with np.errstate(over="ignore"):
                as_float32 = values.astype(np.float32)
            _require_all_finite(as_float32, f"{name} targets after target_transform", fn, "targets")

    n_features = features.shape[1]
    net = _resolve_model(
        model, n_features, n_outputs, _REGRESSOR_HIDDEN, device, seed,
        Tensor(prepared.train_features[:2]), f"one value per target column, for a {n_outputs}-column y", fn,
    )
    preflight_save(net, path, prepared.transform, None, "regression", fn, target_transform=fitted)

    train_loader = DataLoader(
        TensorDataset(Tensor(prepared.train_features), Tensor(train_fit.astype(np.float32))),
        batch_size=batch_size, shuffle=True, generator=np.random.default_rng(seed),
    )
    val_loader = DataLoader(
        TensorDataset(Tensor(prepared.val_features), Tensor(val_fit.astype(np.float32))), batch_size=batch_size,
    )
    saved = train_and_save(
        net, train_loader,
        loss=MSELoss(),
        optimizer=Adam(net.parameters(), lr=learning_rate),
        epochs=epochs,
        validation_dataset=val_loader,
        device=device,
        metrics=[MeanSquaredError(), MeanAbsoluteError()],
        verbose=verbose,
        path=path,
        sample=Tensor(prepared.val_features[:1]),
        preprocessing=prepared.transform,
        task="regression",
        early_stopping=_early_stopping(patience),
        target_transform=fitted,
    )

    predictor = load_predictor(path)
    train_eval = predictor.evaluate(features[prepared.train_indices], train_targets)
    val_eval = predictor.evaluate(features[prepared.val_indices], val_targets)
    return TabularRegressionResult(
        task="regression",
        samples=features.shape[0],
        features=n_features,
        outputs=n_outputs,
        train_samples=len(prepared.train_indices),
        validation_samples=len(prepared.val_indices),
        epochs_completed=saved.history.epochs_completed,
        train_loss=train_eval.loss,
        validation_loss=val_eval.loss,
        train_mse=train_eval.mse,
        validation_mse=val_eval.mse,
        train_mae=train_eval.mae,
        validation_mae=val_eval.mae,
        baseline_mse=val_eval.baseline_mse,
        artifact_path=saved.artifact_path,
        stopped_early=saved.stopped_early,
        best_epoch=saved.best_epoch,
        best_monitored_value=saved.best_monitored_value,
        history=saved.history,
        model=saved.model,
        target_transform=fitted,
    )


__all__ = [
    "train_tabular_classifier",
    "train_tabular_regressor",
    "TabularClassificationResult",
    "TabularRegressionResult",
]
