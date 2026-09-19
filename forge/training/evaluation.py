"""Result types and metric computation for `ArtifactPredictor.evaluate()` (Milestone 113).

`forge.load_predictor(path).evaluate(X, y)` scores a *saved* artifact on
held-out labeled data, applying the artifact's own persisted preprocessing
(see `ArtifactPredictor.evaluate()` in `forge/training/inference.py` for the
workflow). This module holds only the two pieces that have nothing to do with
loading or running a model: the frozen result dataclasses, and the small
NumPy reductions that turn model output plus labels into metrics. It imports
nothing from `inference.py`, so there is no cycle.

## No new metric framework

Headline numbers reuse what already exists -- `forge.training.Accuracy`,
`MeanSquaredError`, `MeanAbsoluteError` for the metrics, and
`forge.nn.CrossEntropyLoss` / `MSELoss` for `loss` (the only two losses Forge
has, and the ones every classification / regression workflow trains with).
The confusion matrix and per-class precision/recall are a handful of NumPy
lines computed here, not new `Metric` classes.

## Metric definitions

Classification (`ClassificationEvaluationResult`), for `n` samples and `k`
persisted classes:

- `accuracy` -- `correct / n`, `argmax` over the model's `k` output scores.
- `loss` -- mean `CrossEntropyLoss` over all `n` samples (computed on the
  concatenated outputs, so it is sample-weighted, independent of batch size).
- `confusion_matrix` -- `(k, k)` `int64`; **row = true class, column =
  predicted class**, both indexed by the artifact's persisted class order
  (`classes[i]` labels row/column `i`). The order is never inferred from the
  evaluation data, so a class absent from the data still has its row/column.
- `precision[c]` -- `TP_c / (TP_c + FP_c)` = diagonal / column sum.
- `recall[c]` -- `TP_c / (TP_c + FN_c)` = diagonal / row sum (= `support[c]`).
- **Zero division** is defined, not warned about: a class the model never
  predicted has `precision == 0.0`; a class with no true samples in the
  evaluated data has `recall == 0.0` (scikit-learn's `zero_division=0`
  convention). No NaN and no NumPy warning can reach the result.
- `baseline_accuracy` -- the share of the *evaluated data's* most common
  class: what "always predict the majority class" scores on exactly these
  samples. It is a floor to beat, not a statement about the training set (the
  artifact does not record its training distribution).

Regression (`RegressionEvaluationResult`):

- `mse` / `mae` -- mean squared / absolute error over every element
  (`MeanSquaredError` / `MeanAbsoluteError`); for a multi-output model the mean
  is over samples *and* outputs.
- `loss` -- `MSELoss` over the same outputs. Numerically `mse`, computed the
  way training computes it (single precision), so the two agree to float32
  rounding rather than bit-for-bit.
- `baseline_mse` -- the MSE of always predicting the evaluated targets' own
  per-output mean (what `r2 == 0` corresponds to). Same "floor to beat"
  reading as `baseline_accuracy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..exceptions import DataError, PersistenceError
from ..nn.loss import CrossEntropyLoss, MSELoss
from ..tensor.tensor import Tensor
from .metrics import Accuracy, MeanAbsoluteError, MeanSquaredError


@dataclass(frozen=True, eq=False)
class ClassificationEvaluationResult:
    """What `ArtifactPredictor.evaluate()` returns for a classification artifact.

    ```python
    result = predictor.evaluate(X_test, y_test)
    result.accuracy, result.baseline_accuracy
    result.classes                      # ('no_diabetes', 'diabetes')
    result.confusion_matrix             # rows = true class, columns = predicted class
    dict(zip(result.classes, result.recall))
    ```

    `classes`, `precision`, `recall`, `support` are parallel: index `i` of each
    describes `classes[i]`, which is also row/column `i` of
    `confusion_matrix` (a read-only `int64` array; rows are the true class).
    `support[i]` is the number of evaluated samples whose true class is
    `classes[i]`. See `forge/training/evaluation.py` for the exact metric
    definitions, including the zero-division rule. Equality is identity
    (`eq=False`) because a dataclass `==` over an array field is ambiguous.
    """

    task: str
    samples: int
    loss: float
    accuracy: float
    baseline_accuracy: float
    classes: "tuple[str, ...]"
    confusion_matrix: np.ndarray
    precision: "tuple[float, ...]"
    recall: "tuple[float, ...]"
    support: "tuple[int, ...]"

    @property
    def metrics(self) -> "dict[str, float]":
        """`{"accuracy": ...}` -- the same key `TrainingResult.val_metrics` uses."""
        return {"accuracy": self.accuracy}


@dataclass(frozen=True)
class RegressionEvaluationResult:
    """What `ArtifactPredictor.evaluate()` returns for a regression artifact.

    `loss` and `mse` are the same quantity (see `forge/training/evaluation.py`);
    `baseline_mse` is the MSE of always predicting the evaluated targets' mean.
    """

    task: str
    samples: int
    loss: float
    mse: float
    mae: float
    baseline_mse: float

    @property
    def metrics(self) -> "dict[str, float]":
        """`{"mse": ..., "mae": ...}` -- the same keys `TrainingResult.val_metrics` uses."""
        return {"mse": self.mse, "mae": self.mae}


def require_evaluation_classes(path: str, classes: "list[str] | None") -> "list[str]":
    """The artifact's persisted class list, or a clear `PersistenceError` if it has none."""
    if not classes:
        raise PersistenceError(
            f"'{path}' was saved with no class list (see forge.save_model(..., classes=...)) -- "
            "ArtifactPredictor.evaluate() needs the persisted class order to index the confusion "
            "matrix and per-class metrics, and will not infer one from the evaluation data."
        )
    return list(classes)


def encode_class_labels(y: Any, classes: "Sequence[str]", fn_name: str) -> np.ndarray:
    """Turn `y` into a 1-D `int64` array of indices into `classes`.

    `y` is either class **names** (every one must be in `classes`) or integer
    class **indices** in `[0, len(classes))` -- the two conventions the rest of
    Forge already uses (`classes=` at save time; integer targets for
    `CrossEntropyLoss`). Anything else is a `DataError` naming the expected
    vocabulary, never a raw `KeyError`/`IndexError`.
    """
    if isinstance(y, Tensor):
        y = y.to("cpu").numpy()
    try:
        labels = np.asarray(y)
    except ValueError as exc:
        raise DataError(f"{fn_name}() could not interpret y as a 1-D list of labels: {exc}") from exc

    if labels.ndim != 1:
        raise DataError(
            f"{fn_name}() requires y to be 1-D (one class label per sample), got shape {labels.shape}."
        )
    if labels.shape[0] == 0:
        raise DataError(f"{fn_name}() received no labels.")

    class_list = list(classes)
    if labels.dtype.kind in ("U", "S"):
        index_of = {name: i for i, name in enumerate(class_list)}
        names = labels.astype(str)
        unknown = sorted({str(n) for n in names if str(n) not in index_of})
        if unknown:
            raise DataError(
                f"{fn_name}() y contains label(s) {unknown!r} that are not among this artifact's "
                f"classes {class_list!r}."
            )
        return np.fromiter((index_of[str(n)] for n in names), dtype=np.int64, count=names.shape[0])

    if labels.dtype.kind in ("i", "u"):
        bad = (labels < 0) | (labels >= len(class_list))
        if bad.any():
            raise DataError(
                f"{fn_name}() y contains class index value(s) {sorted(set(labels[bad].tolist()))!r} "
                f"outside [0, {len(class_list)}) for this artifact's classes {class_list!r}."
            )
        return labels.astype(np.int64)

    raise DataError(
        f"{fn_name}() requires y to be class names (str) or integer class indices, got dtype "
        f"'{labels.dtype}'."
    )


def encode_regression_targets(y: Any, fn_name: str) -> np.ndarray:
    """Turn `y` into a finite floating-point array, or raise `DataError`."""
    if isinstance(y, Tensor):
        y = y.to("cpu").numpy()
    try:
        targets = np.asarray(y)
    except ValueError as exc:
        raise DataError(f"{fn_name}() could not interpret y as a numeric array: {exc}") from exc
    if targets.dtype.kind not in ("f", "i", "u"):
        raise DataError(f"{fn_name}() requires numeric regression targets, got dtype '{targets.dtype}'.")
    if targets.ndim not in (1, 2):
        raise DataError(
            f"{fn_name}() requires y of shape (n,) or (n, outputs), got shape {targets.shape}."
        )
    if targets.shape[0] == 0:
        raise DataError(f"{fn_name}() received no targets.")
    targets = targets.astype(np.float64)
    if not np.isfinite(targets).all():
        raise DataError(
            f"{fn_name}() received {int((~np.isfinite(targets)).sum())} non-finite value(s) (NaN/Inf) "
            "in y -- replace them with real values before evaluating."
        )
    return targets


def require_finite_output(output: np.ndarray, fn_name: str) -> None:
    """Refuse to score a model output that already contains NaN/Inf.

    A finite, preprocessed input can still overflow to a non-finite output;
    scoring it would put NaN into `loss`/`mse` with no explanation.
    """
    if not np.isfinite(output).all():
        raise DataError(
            f"{fn_name}() the model produced non-finite (NaN/Inf) output for this data, so no "
            "meaningful metrics exist. Check the inputs' scale against the artifact's preprocessing."
        )


def build_classification_result(
    task: str, output: Tensor, labels: np.ndarray, classes: "Sequence[str]", path: str,
) -> ClassificationEvaluationResult:
    """Metrics for `output` (`(n, k)` raw scores, CPU) against `labels` (`(n,)` class indices)."""
    class_names = tuple(classes)
    k = len(class_names)
    if output.ndim != 2 or output.shape[1] != k:
        raise PersistenceError(
            f"'{path}' is inconsistent: its model produced output of shape {output.shape}, but the "
            f"artifact lists {k} class(es) {list(class_names)!r}."
        )

    n = int(labels.shape[0])
    scores = output.numpy()
    predicted = np.argmax(scores, axis=1)

    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    confusion.setflags(write=False)

    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(k, dtype=np.float64), where=predicted_count > 0,
    )
    recall = np.divide(true_positive, support, out=np.zeros(k, dtype=np.float64), where=support > 0)

    accuracy = Accuracy()
    accuracy.update(output, labels)

    return ClassificationEvaluationResult(
        task=task,
        samples=n,
        loss=float(CrossEntropyLoss()(output, labels).numpy()),
        accuracy=accuracy.compute(),
        baseline_accuracy=float(support.max()) / n,
        classes=class_names,
        confusion_matrix=confusion,
        precision=tuple(float(v) for v in precision),
        recall=tuple(float(v) for v in recall),
        support=tuple(int(v) for v in support),
    )


def build_regression_result(task: str, output: Tensor, targets: np.ndarray, fn_name: str) -> RegressionEvaluationResult:
    """Metrics for `output` (CPU) against `targets` (`(n,)` or `(n, outputs)`, float64)."""
    predictions = output.numpy()
    # A model with a single output column is scored against a plain (n,) target
    # (and vice versa) -- the one shape ambiguity a caller can't be expected to guess.
    if targets.ndim == 1 and predictions.ndim == 2 and predictions.shape[1] == 1:
        targets = targets[:, None]
    elif targets.ndim == 2 and targets.shape[1] == 1 and predictions.ndim == 1:
        targets = targets[:, 0]
    if targets.shape != predictions.shape:
        raise DataError(
            f"{fn_name}() y has shape {targets.shape}, but the model's output has shape "
            f"{predictions.shape} -- y must have one target per model output."
        )

    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    mse.update(predictions, targets)
    mae.update(predictions, targets)
    baseline = float(np.mean((targets - targets.mean(axis=0)) ** 2))

    return RegressionEvaluationResult(
        task=task,
        samples=int(targets.shape[0]),
        loss=float(MSELoss()(output, targets).numpy()),
        mse=mse.compute(),
        mae=mae.compute(),
        baseline_mse=baseline,
    )


__all__ = ["ClassificationEvaluationResult", "RegressionEvaluationResult"]
