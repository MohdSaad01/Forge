"""`StandardizeTarget`: a persisted, invertible transform of a regression *target* (Milestone 116).

`forge.data.transforms` prepare a model's **inputs**; nothing in Forge prepared its
**targets**, so a regression model trained on a target of large magnitude (California
house prices in dollars, ~2x10^5) trained worse, and the caller's only recourse --
standardising `y` by hand -- left an artifact that predicts in standardised units and
must be inverted with constants kept out of band (M115, `docs/development/m116-persisted-target-transforms.md`).

```text
native y --transform--> training target --> model --> prediction --inverse_transform--> native y
```

`StandardizeTarget` is that pair of maps for `z = (y - mean) / std`, per target column.
It is saved *inside the artifact* (`save_model(..., target_transform=...)`), so
`predict()` and `evaluate()` on the artifact speak the user's units and a consumer never
needs to know the model was trained on z-scores.

This is deliberately **not** a transform framework: one concrete, closed type, host-side
NumPy arithmetic in float64, no registry and no user-supplied callables (nothing here can
be pickled or executed from a file). `forge.data.transforms.Transform` is the *input*
pipeline (Tensor in, Tensor out, no inverse) and is unrelated to this class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import DataError
from ..tensor.tensor import Tensor

# The closed vocabulary of persistable target transforms (also the `target_transform=`
# strings of `forge.train_tabular_regressor()`).
TARGET_TRANSFORM_TYPES = ("standardize",)


def _as_float64_vector(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataError(f"StandardizeTarget {name} must be a number or a list of numbers ({exc}).") from exc
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.shape[0] == 0:
        raise DataError(f"StandardizeTarget {name} must be a non-empty 1-D list (one entry per target column), got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise DataError(f"StandardizeTarget {name} must be finite, got {array.tolist()!r}.")
    return array


class StandardizeTarget:
    """`z = (y - mean) / std` per target column, and its inverse `y = z * std + mean`.

    ```python
    target = StandardizeTarget.fit(y_train)     # statistics from TRAINING targets only
    z = target.transform(y_train)               # what the model is trained on
    y = target.inverse_transform(model_output)  # native units again
    ```

    `mean`/`std` are per target column (a 1-column target has length-1 lists); `std`
    must be finite and > 0. Arrays are NumPy, arithmetic is float64 (the caller's own
    precision), and the last axis of an input is the target-column axis: `(n,)` for a
    1-column target, `(n, outputs)` in general, or an unbatched `(outputs,)`.

    Construct with fitted values (`StandardizeTarget(mean, std)`, as `Normalize` is) or
    fit them with `StandardizeTarget.fit(y_train)`. Two instances are equal when their
    parameters are.
    """

    type_name = "standardize"

    def __init__(self, mean: Any, std: Any) -> None:
        mean_array = _as_float64_vector(mean, "mean")
        std_array = _as_float64_vector(std, "std")
        if mean_array.shape != std_array.shape:
            raise DataError(
                f"StandardizeTarget mean and std must have one entry per target column and the same "
                f"length, got {mean_array.shape[0]} and {std_array.shape[0]}."
            )
        if (std_array <= 0).any():
            raise DataError(
                f"StandardizeTarget std must be > 0 for every target column, got {std_array.tolist()!r}."
            )
        self.mean = mean_array
        self.std = std_array

    @property
    def outputs(self) -> int:
        """The number of target columns this transform was fitted for."""
        return int(self.mean.shape[0])

    @classmethod
    def fit(cls, y: Any) -> "StandardizeTarget":
        """Fit `mean`/`std` to `y`: `(n,)` (one column) or `(n, outputs)`; population std (`ddof=0`).

        Pass **training** targets only -- a transform fitted on validation or test
        targets leaks their distribution into training. Raises `DataError` for a
        non-numeric, non-finite, empty or wrongly shaped `y`, and for a **constant**
        column (`max == min`): its standard deviation is 0, so there is nothing to
        standardise by, and Forge does not invent a scale (features get `std = 1` in
        that case because a constant *feature* is harmless; a constant *target* means
        there is nothing to learn or the data is wrong).
        """
        array = _as_target_array(y, "StandardizeTarget.fit()")
        constant = np.flatnonzero(array.max(axis=0) == array.min(axis=0))
        if constant.size:
            raise DataError(
                f"StandardizeTarget.fit() target column(s) {constant.tolist()} are constant "
                f"(value {array[0, constant].tolist()!r}) in the data it was fitted on, so their "
                "standard deviation is 0 and there is no scale to standardise by. Drop the constant "
                "target column or train without target_transform."
            )
        std = array.std(axis=0)
        if not np.isfinite(std).all() or (std <= 0).any():
            raise DataError(
                "StandardizeTarget.fit() could not compute a finite positive standard deviation for "
                "the targets (their magnitude is too large or their spread too small to standardise)."
            )
        return cls(array.mean(axis=0), std)

    def transform(self, y: Any) -> np.ndarray:
        """`(y - mean) / std`, float64, same shape as `y`."""
        return (self._checked(y, "transform") - self.mean) / self.std

    def inverse_transform(self, z: Any) -> np.ndarray:
        """`z * std + mean`, float64, same shape as `z`: a model output back in native units."""
        return self._checked(z, "inverse_transform") * self.std + self.mean

    def _checked(self, values: Any, fn: str) -> np.ndarray:
        array = _as_numeric_array(values, f"StandardizeTarget.{fn}()")
        if array.ndim == 0:
            raise DataError(f"StandardizeTarget.{fn}() requires an array, got a scalar.")
        if self.outputs != 1 and array.shape[-1] != self.outputs:
            raise DataError(
                f"StandardizeTarget.{fn}() expected {self.outputs} target column(s) along the last "
                f"axis, got shape {array.shape}."
            )
        return array

    def to_config(self) -> dict:
        """The JSON-safe `{"type": "standardize", "mean": [...], "std": [...]}` node saved in an artifact."""
        return {"type": self.type_name, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_config(cls, node: Any) -> "StandardizeTarget":
        """Inverse of `to_config()`; raises `DataError` for values that are not a valid fitted transform."""
        return cls(node["mean"], node["std"])

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StandardizeTarget)
            and np.array_equal(self.mean, other.mean)
            and np.array_equal(self.std, other.std)
        )

    def __hash__(self) -> int:
        return hash((self.mean.tobytes(), self.std.tobytes()))

    def __repr__(self) -> str:
        return f"StandardizeTarget(mean={self.mean.tolist()}, std={self.std.tolist()})"


def _as_numeric_array(values: Any, fn: str) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.to("cpu").numpy()
    try:
        array = np.asarray(values)
    except ValueError as exc:
        raise DataError(f"{fn} could not interpret its input as a numeric array: {exc}") from exc
    if array.dtype.kind not in ("f", "i", "u"):
        raise DataError(f"{fn} requires numeric values, got dtype '{array.dtype}'.")
    return array.astype(np.float64)


def _as_target_array(y: Any, fn: str) -> np.ndarray:
    array = _as_numeric_array(y, fn)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise DataError(f"{fn} requires y of shape (n,) or (n, outputs), got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DataError(f"{fn} requires a non-empty y, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise DataError(
            f"{fn} received {int((~np.isfinite(array)).sum())} non-finite value(s) (NaN/Inf) in y -- "
            "replace them with real values first."
        )
    return array


__all__ = ["StandardizeTarget", "TARGET_TRANSFORM_TYPES"]
