"""`forge.data.load_csv()`: a numeric CSV file to the `(X, y)` arrays Forge's tabular APIs take (Milestone 118).

Forge's tabular workflow is defined on in-memory arrays --
`train_tabular_classifier(X, y)`, `train_tabular_regressor(X, y)`,
`ArtifactPredictor.evaluate(X, y)`. A CSV file is only a way to get those arrays,
so this module is a boundary and nothing more: it reads text, checks that the text
means exactly one thing, and hands back ordinary NumPy arrays. It has no idea what
a preprocessing step, a target transform, a class list or an artifact is, and it
never will -- everything after the arrays exist is the existing code, unchanged.

```python
X, y = forge.data.load_csv("diabetes.csv", target="Outcome", labels=True)
result = forge.train_tabular_classifier(X, y, path="diabetes.forge", classes=["no_diabetes", "diabetes"])
```

## The contract

A CSV file is accepted if, and only if, all of this holds. Anything else is a
`DataError` that names the file, the line and the column -- nothing is guessed:

- **Comma-delimited, UTF-8** (a leading byte-order mark is accepted), standard
  double-quote quoting (a quoted field may hold commas, quotes and newlines);
  malformed quoting is an error. No other delimiter is tried or detected.
- **The first non-blank row is the header.** There is no header sniffing: a file
  without one is rejected (its first data row would be read as column names, which
  are then not unique/not found). Header names are stripped of surrounding
  whitespace, must be non-empty and **unique**.
- **`target=` names one column, explicitly** -- never "the last column". It is
  removed from the features. **Every other column is a numeric feature**, in file
  order (never sorted, never reordered) -- unless `columns=` is given (Milestone
  120): then **exactly those columns**, in the order given, are the features, and
  every other column (an `id`, a timestamp, anything not explicitly selected) is
  ignored -- not read, not validated, not guessed. There is still no automatic
  drop/ignore list: a column that is not a feature is either removed from the
  file or left out of `columns=`; Forge never infers that a column named `id` is
  not a feature.
- **Every row has as many fields as the header.** Blank lines are skipped;
  a row of empty fields is not blank, it is missing data.
- **Feature cells are decimal numbers** (`12`, `-3.5`, `.5`, `1e-3`), surrounding
  whitespace ignored. Not numbers: text, `1_000`, hex, non-ASCII digits, and
  `nan`/`inf` (which are finite-check failures, reported as such).
- **No missing values.** An empty cell is an error, never `0`, never a mean. Forge's
  one missing-value mechanism is `ReplaceValue` (`missing_columns=` on the training
  functions), which replaces a numeric *sentinel* such as `0`; it cannot represent
  "empty", and this reader does not invent an imputation rule.
- **The target** is read one of two explicit ways, chosen by the caller, never guessed:
  by default (`labels=False`) it is **numbers** -> a `float64` array (regression);
  with `labels=True` it is **class labels** -> integers (`0`, `1`, `2`; `1.0` is the
  integer 1, `0.5` is an error) as an `int64` array of class *indices*, or text as a
  `str` array of class *names* (cells verbatim after whitespace stripping). A label column
  that mixes integers and text is an error, so one stray `abc` in a column of `0`/`1`
  cannot quietly become a third class.

Deliberately not supported, and not approximated: categorical/one-hot features,
string features, dates, missing-value imputation, other delimiters or encodings,
files with comments, streaming. Ask for none of them and the file is rejected.

## What comes back

`X` is `(rows, features)` `float64`; `y` is `(rows,)` `float64` (numbers), `int64` (integer
labels) or `str` (text labels). `float64` keeps every digit of the file; Forge's tabular
functions cast to `float32` themselves, exactly as for any NumPy array a caller passes.
The whole file is read into memory.

**Explicit column selection (`columns=`, Milestone 120).** A real CSV often carries a
column that is not a model feature -- an `id`, a timestamp, some other piece of
metadata -- and until this milestone the only way to read it was to remove that
column from the file first. `load_csv(path, target=..., columns=["Age", "Glucose",
"BMI"])` (and `load_csv_features(path, columns=[...])`, which has no target) instead
selects exactly those columns, in exactly that order, regardless of where they sit in
the file; everything else in the file -- an `id` column included -- is never read.
`columns=` is validated the same way `feature_names=` is (`forge.data.feature_names.
validate_feature_names`: a non-empty list/tuple of distinct, non-blank strings), plus
two CSV-specific rules: every name must be a real header column (an unknown one is a
`DataError` naming it), and (for `load_csv()`) `target` must not appear in `columns`
(the target is always removed automatically; listing it too is rejected, not silently
dropped). There is still no automatic id/timestamp detection -- selection is always
explicit, and omitting `columns=` is exactly the pre-Milestone-120 "every other column
is a feature" contract, unchanged.

Column **names are not in the arrays** -- an array has none. `load_csv(...,
return_feature_names=True)` returns them beside `X` (Milestone 119), and
`load_csv_features(path)` reads a file with no target at all. Handed on as
`feature_names=`, they let a saved artifact match a CSV's columns to the model's by name
instead of by position (`train_tabular_*(..., feature_names=)` records them;
`predict()`/`evaluate(..., feature_names=)` check against them). An artifact trained
without names records only its feature *count* and cannot detect a reordered CSV: for
those, keep the feature columns in one order.

Integer labels are class **indices** and text labels class **names**, exactly as for any
NumPy `y` -- see `train_tabular_classifier()` / `ArtifactPredictor.evaluate()`, which
alone know the class order. A class whose *name* is a number (`"1"`, `"2"`) cannot be
written in a CSV label column (it reads as an index); use the array API for that.
"""

from __future__ import annotations

import csv
import math
import os
import re

import numpy as np

from ..exceptions import DataError
from .feature_names import validate_feature_names

# Decimal floats only, ASCII digits only. `float()` alone would also accept "1_000", "١٢٣" and "infinity".
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z", re.ASCII)
_NON_FINITE = re.compile(r"[+-]?(?:nan|inf|infinity)\Z", re.ASCII | re.IGNORECASE)

_MAX_LISTED = 12  # column names shown in a "not found" message


def _quote(text: str, limit: int = 40) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _parse_number(cell: str, where: str, role: str) -> float:
    """`cell` as a finite float, or a `DataError` saying which cell of `where` and why."""
    if cell == "":
        raise DataError(
            f"{where}: empty {role}. Forge does not impute missing values -- fill it, drop the row, or "
            "(for a sentinel such as 0) use missing_columns= in the training call."
        )
    if _NON_FINITE.match(cell):
        raise DataError(f"{where}: {_quote(cell)} is not a finite number (NaN/Inf). Forge does not train on or score NaN/Inf.")
    if not _NUMBER.match(cell):
        hint = (
            "Forge reads every column except the target as a numeric feature (no categorical/string "
            "features, no dates); remove or convert this column."
            if role == "feature" else "a numeric target was required."
        )
        raise DataError(f"{where}: {_quote(cell)} is not a number. {hint}")
    value = float(cell)
    if not math.isfinite(value):
        raise DataError(f"{where}: {_quote(cell)} is too large to represent (it overflows to infinity).")
    return value


def _read_header(row: "list[str]", path: str, target: "str | None") -> "tuple[list[str], int]":
    """The stripped header names and the target's index (`-1` when `target` is `None`: every column is a feature)."""
    names = [cell.strip() for cell in row]
    where = f"CSV file '{path}'"
    empty = [i + 1 for i, name in enumerate(names) if name == ""]
    if empty:
        raise DataError(
            f"{where}: header column(s) {empty} have no name. Every column needs a name (an unnamed index "
            "column, as some tools write, must be removed from the file)."
        )
    if all(_NUMBER.match(name) for name in names):
        raise DataError(
            f"{where}: the first row {names[:6]!r} looks like data, not a header. The file must start with a "
            "header row of column names; Forge does not guess whether there is one."
        )
    if len(names) == 1 and any(sep in names[0] for sep in (";", "\t", "|")):
        raise DataError(
            f"{where}: the header is one column, {_quote(names[0])}. Forge reads comma-delimited CSV only "
            "(no other delimiter is detected)."
        )
    if target is not None and names.count(target) > 1:
        positions = [i + 1 for i, name in enumerate(names) if name == target]
        raise DataError(
            f"{where}: the target column {target!r} appears {len(positions)} times in the header (columns "
            f"{positions}); column names must be unique so the target cannot be confused with a feature."
        )
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise DataError(f"{where}: duplicate column name(s) {duplicated!r}; column names must be unique.")
    if target is None:
        return names, -1
    if target not in names:
        shown = ", ".join(repr(n) for n in names[:_MAX_LISTED]) + (", ..." if len(names) > _MAX_LISTED else "")
        raise DataError(
            f"{where}: target column {target!r} is not in the header. Columns are matched exactly "
            f"(case-sensitive); the file has {len(names)}: {shown}."
        )
    if len(names) == 1:
        raise DataError(
            f"{where}: the only column is the target {target!r}; there are no feature columns to learn from."
        )
    return names, names.index(target)


def _validate_columns_argument(columns: Any, fn: str) -> "tuple[str, ...]":
    """`columns=` as a tuple of distinct, non-blank column names, or a `DataError` naming what is wrong (Milestone 120).

    Same structural shape `forge.data.feature_names.validate_feature_names` already defines for
    `feature_names=` (a non-empty `list`/`tuple` of distinct, non-empty, non-`str` types rejected) --
    reused rather than re-invented, since a valid column selection and a valid `feature_names=` list
    are the same kind of thing. The message names `columns`, not `feature_names`: the two arguments
    are validated identically but mean different things (one selects input columns from a file, the
    other records an artifact's schema) and should not be confused in an error a caller reads.
    """
    try:
        return validate_feature_names(columns)
    except DataError as exc:
        # validate_feature_names()'s own message always starts with the literal word "feature_names";
        # reword it for this argument rather than exposing the internal helper's own name.
        reworded = str(exc).replace("feature_names", "columns", 1)
        raise DataError(f"{fn}() {reworded}") from exc


def _check_columns_against_header(
    columns: "tuple[str, ...]", names: "list[str]", target: "str | None", where: str,
) -> None:
    """`columns=` against the file's actual header: every name must exist, and (with a target) none may be it (Milestone 120)."""
    if target is not None and target in columns:
        raise DataError(
            f"{where}: columns includes the target column {target!r}. The target is always removed from "
            "the features automatically; list only feature columns in columns."
        )
    header = set(names)
    unknown = [c for c in columns if c not in header]
    if unknown:
        shown = ", ".join(repr(c) for c in unknown[:_MAX_LISTED]) + (", ..." if len(unknown) > _MAX_LISTED else "")
        header_shown = ", ".join(repr(n) for n in names[:_MAX_LISTED]) + (", ..." if len(names) > _MAX_LISTED else "")
        raise DataError(
            f"{where}: columns names column(s) not in the header: {shown}. Columns are matched exactly "
            f"(case-sensitive); the file has {len(names)}: {header_shown}."
        )


def load_csv(
    path: "str | os.PathLike",
    *,
    target: str,
    labels: bool = False,
    return_feature_names: bool = False,
    columns: "Sequence[str] | None" = None,
) -> "tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]":
    """Read a numeric CSV file into `(X, y)` NumPy arrays (Milestone 118).

    ```python
    X, y = forge.data.load_csv("housing.csv", target="median_house_value")
    result = forge.train_tabular_regressor(X, y, path="housing.forge", target_transform="standardize")

    # Milestone 119: keep the column names, so the artifact can check them later
    X, y, names = forge.data.load_csv("housing.csv", target="median_house_value", return_feature_names=True)
    result = forge.train_tabular_regressor(X, y, path="housing.forge", feature_names=names)

    # Milestone 120: a real CSV that also carries an id column -- select the features explicitly
    X, y, names = forge.data.load_csv(
        "diabetes.csv", target="Outcome", columns=["Pregnancies", "Glucose", "BMI"], return_feature_names=True,
    )
    ```

    - `path` -- the CSV file: comma-delimited UTF-8 with a header row (module docstring,
      **The contract**).
    - `target` -- the header name of the target column, matched exactly. It is removed from
      the features; every other column is a numeric feature, kept in file order -- unless
      `columns=` is given.
    - `labels` -- `False` (default): the target is numbers, `y` is `float64` (regression; a
      non-numeric cell is named). `True`: the target is class labels (classification): a
      column of integers gives an `int64` `y` of class indices, a column of text a `str`
      `y` of class names, a mixture or a non-integer number is an error.
    - `columns` -- `None` (default): every column except `target` is a feature, in file
      order (unchanged). Otherwise (Milestone 120) a non-empty list/tuple of distinct header
      names: **exactly** those columns become the features, in **exactly** the order given --
      regardless of where they sit in the file -- and every other column (an `id`, a
      timestamp, anything else not selected) is ignored outright: never read, never
      validated, never guessed. `target` must not appear in `columns` (it is always removed
      automatically -- listing it too is a `DataError`, not silently dropped); a name not in
      the file's header is a `DataError` naming it; a duplicate name is a `DataError`.
    - `return_feature_names` -- `False` (default): return `(X, y)`. `True` (Milestone 119): return
      `(X, y, feature_names)`, the header names of the columns of `X` (the target excluded), in the
      same order, from the *same read* of the file -- exactly `columns`, when given. Pass them to
      `train_tabular_*(..., feature_names=)` to record them in the artifact, and to
      `predict()`/`evaluate(..., feature_names=)` to have a named input checked against it. They are
      the header cells with surrounding whitespace stripped -- nothing else about a name is changed.

    Returns `(X, y)`: `X` `(rows, features)` `float64`, `y` `(rows,)`. They are plain
    arrays for `train_tabular_classifier()` / `train_tabular_regressor()` /
    `ArtifactPredictor.evaluate()`, which then validate them exactly as any other
    array. Raises `forge.DataError` for anything the contract rejects, naming the file,
    line and column; nothing is repaired or skipped.
    """
    _check_path(path, "load_csv")
    if not isinstance(target, str) or not target.strip():
        raise DataError(f"load_csv() target must be the name of a column (a non-empty str), got {target!r}.")
    if not isinstance(labels, bool):
        raise DataError(f"load_csv() labels must be True or False, got {labels!r}.")
    if not isinstance(return_feature_names, bool):
        raise DataError(f"load_csv() return_feature_names must be True or False, got {return_feature_names!r}.")
    checked_columns = _validate_columns_argument(columns, "load_csv") if columns is not None else None
    X, y, feature_names = _read_csv(os.fspath(path), target.strip(), labels, checked_columns)
    return (X, y, feature_names) if return_feature_names else (X, y)


def load_csv_features(
    path: "str | os.PathLike", *, columns: "Sequence[str] | None" = None,
) -> "tuple[np.ndarray, list[str]]":
    """Read a CSV file that has *no target column* into `(X, feature_names)` (Milestone 119).

    ```python
    X, names = forge.data.load_csv_features("new_patients.csv")
    forge.load_predictor("diabetes.forge").predict(X, feature_names=names)

    # Milestone 120: the file also has an id column -- select the features explicitly
    X, names = forge.data.load_csv_features("new_patients.csv", columns=["Pregnancies", "Glucose", "BMI"])
    ```

    The prediction-time counterpart of `load_csv()`: the same file contract (header row,
    comma-delimited UTF-8, unique non-empty names, numeric cells, no missing values),
    except that there is no target column at all.

    - `columns` -- `None` (default): **every** column is a feature, in file order (unchanged --
      there is nothing to drop; a target or `id` column left in the file is a feature column as
      far as this reader knows, and a named artifact then rejects it as an unexpected column).
      Otherwise (Milestone 120), the same explicit selection `load_csv(..., columns=...)`
      documents: exactly those columns, in exactly that order; every other column is ignored.

    Returns `X` `(rows, features)` `float64` and the header names (whitespace stripped, or
    exactly `columns` when given) as a list. Raises `forge.DataError`, as `load_csv()` does.
    """
    _check_path(path, "load_csv_features")
    checked_columns = _validate_columns_argument(columns, "load_csv_features") if columns is not None else None
    X, _, feature_names = _read_csv(os.fspath(path), None, False, checked_columns)
    return X, feature_names


def _check_path(path: "str | os.PathLike", fn: str) -> None:
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise DataError(f"{fn}() path must be a non-empty file path (str or os.PathLike), got {path!r}.")


def _read_csv(
    path: str, target: "str | None", labels: bool, columns: "tuple[str, ...] | None" = None,
) -> "tuple[np.ndarray, np.ndarray | None, list[str]]":
    """The one reader behind `load_csv()`/`load_csv_features()`: `(X, y, feature_names)`, `y` is `None` without a target.

    `columns` (Milestone 120), when given, is already-validated (`_validate_columns_argument()`): the
    feature columns to select, in the order to select them in. Selection and target-removal share one
    mechanism -- `feature_indices`, the header positions of `feature_names` -- so there is exactly one
    code path whether or not `columns` was given, not two.
    """
    if not os.path.isfile(path):
        raise DataError(f"CSV file not found: {path}" if not os.path.exists(path) else f"CSV path is not a file: {path}")

    where = f"CSV file '{path}'"
    features: "list[list[float]]" = []
    target_cells: "list[str]" = []
    target_lines: "list[int]" = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            names: "list[str] | None" = None
            target_index = -1
            feature_indices: "list[int]" = []
            for row in reader:
                if not row:  # a blank line
                    continue
                line = reader.line_num
                if names is None:
                    names, target_index = _read_header(row, path, target)
                    if columns is not None:
                        _check_columns_against_header(columns, names, target, where)
                        feature_names = list(columns)
                    else:
                        feature_names = [n for i, n in enumerate(names) if i != target_index]
                    feature_indices = [names.index(n) for n in feature_names]
                    continue
                if len(row) != len(names):
                    raise DataError(
                        f"{where}, line {line}: {len(row)} field(s), but the header has {len(names)} column(s) "
                        f"({names[0]!r}, ..., {names[-1]!r}). Every row needs one field per column."
                    )
                cells = [cell.strip() for cell in row]
                features.append([
                    _parse_number(cells[i], f"{where}, line {line}, column {name!r}", "feature")
                    for i, name in zip(feature_indices, feature_names)
                ])
                if target is None:
                    continue
                target_cell = cells[target_index]
                if target_cell == "":
                    raise DataError(
                        f"{where}, line {line}, column {target!r}: empty target. A row without a target cannot "
                        "be trained on or scored; fill it or drop the row."
                    )
                if labels:
                    target_cells.append(target_cell)
                else:
                    target_cells.append(_parse_number(target_cell, f"{where}, line {line}, column {target!r}", "target"))
                target_lines.append(line)
    except UnicodeDecodeError as exc:
        raise DataError(
            f"{where} is not valid UTF-8 text (byte offset {exc.start}). Forge reads CSV as UTF-8 "
            "(a leading BOM is fine); re-save the file as UTF-8."
        ) from exc
    except csv.Error as exc:
        raise DataError(f"{where}, line {reader.line_num}: malformed CSV ({exc}).") from exc
    except OSError as exc:
        raise DataError(f"cannot read CSV file '{path}': {exc.strerror or exc}") from exc

    if names is None:
        raise DataError(f"{where} is empty: it has no header row.")
    if not features:
        raise DataError(f"{where} has a header but no data rows.")

    if target is None:
        y = None
    elif labels:
        y = _label_column(target_cells, target_lines, target, where)
    else:
        y = np.asarray(target_cells, dtype=np.float64)
    return np.asarray(features, dtype=np.float64), y, feature_names


def _label_column(cells: "list[str]", lines: "list[int]", target: str, where: str) -> np.ndarray:
    """Class labels: all integers -> `int64` indices; all text -> `str` names; a mixture, a fraction or NaN/Inf -> `DataError`."""
    numeric = [bool(_NUMBER.match(cell)) for cell in cells]
    if all(numeric):
        indices = []
        for cell, line in zip(cells, lines):
            value = _parse_number(cell, f"{where}, line {line}, column {target!r}", "target")
            if value != math.floor(value) or abs(value) > 2.0 ** 53:
                raise DataError(
                    f"{where}, line {line}, column {target!r}: {_quote(cell)} is not a class index. Numeric class "
                    "labels are whole numbers (0, 1, 2, ...); for a continuous target read the file without labels=True."
                )
            indices.append(int(value))
        return np.array(indices, dtype=np.int64)
    for cell, line in zip(cells, lines):
        if _NON_FINITE.match(cell):
            raise DataError(
                f"{where}, line {line}, column {target!r}: {_quote(cell)} is a NaN/Inf marker, not a class label."
            )
    if any(numeric):
        other = numeric.index(not numeric[0])  # first cell of the other kind than cell 0
        raise DataError(
            f"{where}, line {lines[other]}, column {target!r}: the label column mixes numbers and text "
            f"(e.g. {_quote(cells[0])} and {_quote(cells[other])}). Labels are all whole numbers "
            "(class indices) or all text (class names); Forge will not decide which cells are typos."
        )
    return np.array(cells, dtype=str)


__all__ = ["load_csv", "load_csv_features"]
