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
  order (never sorted, never reordered). There is no drop/ignore list: an ID or
  any other column that is not a feature has to be removed from the file.
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

Column **names are not carried into the arrays and a saved artifact records only its
feature *count***: an artifact cannot detect a CSV whose columns are in a different
order from the one it was trained on. Keep the feature columns in one order.

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


def _read_header(row: "list[str]", path: str, target: str) -> "tuple[list[str], int]":
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
    if names.count(target) > 1:
        positions = [i + 1 for i, name in enumerate(names) if name == target]
        raise DataError(
            f"{where}: the target column {target!r} appears {len(positions)} times in the header (columns "
            f"{positions}); column names must be unique so the target cannot be confused with a feature."
        )
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise DataError(f"{where}: duplicate column name(s) {duplicated!r}; column names must be unique.")
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


def load_csv(
    path: "str | os.PathLike",
    *,
    target: str,
    labels: bool = False,
) -> "tuple[np.ndarray, np.ndarray]":
    """Read a numeric CSV file into `(X, y)` NumPy arrays (Milestone 118).

    ```python
    X, y = forge.data.load_csv("housing.csv", target="median_house_value")
    result = forge.train_tabular_regressor(X, y, path="housing.forge", target_transform="standardize")
    ```

    - `path` -- the CSV file: comma-delimited UTF-8 with a header row (module docstring,
      **The contract**).
    - `target` -- the header name of the target column, matched exactly. It is removed from
      the features; every other column is a numeric feature, kept in file order.
    - `labels` -- `False` (default): the target is numbers, `y` is `float64` (regression; a
      non-numeric cell is named). `True`: the target is class labels (classification): a
      column of integers gives an `int64` `y` of class indices, a column of text a `str`
      `y` of class names, a mixture or a non-integer number is an error.

    Returns `(X, y)`: `X` `(rows, features)` `float64`, `y` `(rows,)`. They are plain
    arrays for `train_tabular_classifier()` / `train_tabular_regressor()` /
    `ArtifactPredictor.evaluate()`, which then validate them exactly as any other
    array. Raises `forge.DataError` for anything the contract rejects, naming the file,
    line and column; nothing is repaired or skipped.
    """
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path:
        raise DataError(f"load_csv() path must be a non-empty file path (str or os.PathLike), got {path!r}.")
    if not isinstance(target, str) or not target.strip():
        raise DataError(f"load_csv() target must be the name of a column (a non-empty str), got {target!r}.")
    target = target.strip()
    if not isinstance(labels, bool):
        raise DataError(f"load_csv() labels must be True or False, got {labels!r}.")
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
            for row in reader:
                if not row:  # a blank line
                    continue
                line = reader.line_num
                if names is None:
                    names, target_index = _read_header(row, path, target)
                    feature_names = [n for i, n in enumerate(names) if i != target_index]
                    continue
                if len(row) != len(names):
                    raise DataError(
                        f"{where}, line {line}: {len(row)} field(s), but the header has {len(names)} column(s) "
                        f"({names[0]!r}, ..., {names[-1]!r}). Every row needs one field per column."
                    )
                cells = [cell.strip() for cell in row]
                target_cell = cells.pop(target_index)
                features.append([
                    _parse_number(cell, f"{where}, line {line}, column {name!r}", "feature")
                    for cell, name in zip(cells, feature_names)
                ])
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

    y = _label_column(target_cells, target_lines, target, where) if labels else np.asarray(target_cells, dtype=np.float64)
    return np.asarray(features, dtype=np.float64), y


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


__all__ = ["load_csv"]
