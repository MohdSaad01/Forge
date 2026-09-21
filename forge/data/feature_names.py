"""The one definition of what a valid list of tabular feature names is (Milestone 119).

A saved tabular artifact can record *which* feature each input column is, not only how many
there are (`save_model(..., feature_names=...)`, `InputSchema.feature_names`), and named
input (`forge.data.load_csv(..., return_feature_names=True)`, `predict(..., feature_names=...)`)
is checked against that record. Training, saving, loading and prediction all accept a list of
names, and they must agree on what a valid one is; this function is that agreement, so a
name list that can be saved can always be loaded and matched, and vice versa.

The rules are deliberately few, and none of them rewrites a name:

- a `list` or `tuple` (not a bare `str`, which would be read as a list of characters) with at
  least one entry;
- every entry a `str` that is not empty and not only whitespace;
- every entry different from every other, compared **exactly** -- `"Age"` and `"age"` are two
  names, and `"Age"` and `" Age"` are two names.

Names are never stripped, lower-cased or otherwise normalised: whatever is given is what is
stored and what is later compared. (The one place a name is stripped is upstream of this:
`forge.data.load_csv()` has always stripped the whitespace around a CSV *header cell*, a
documented part of reading the file.) Raising here is a `DataError` whose message begins with
`feature_names` so a caller can prefix it with its own name.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import DataError

_MAX_LISTED = 6


def validate_feature_names(names: Any) -> "tuple[str, ...]":
    """`names` as a tuple of plain `str`, or a `DataError` saying exactly what is wrong."""
    if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)):
        raise DataError(f"feature_names must be a list of strings, got {type(names).__name__}: {names!r}.")
    if not names:
        raise DataError("feature_names must not be empty.")
    for position, name in enumerate(names):
        if not isinstance(name, str):
            raise DataError(
                f"feature_names must contain only strings, got {type(name).__name__} {name!r} at position {position}."
            )
        if not name.strip():
            raise DataError(f"feature_names must not contain an empty or blank name, got {name!r} at position {position}.")
    plain = tuple(str(name) for name in names)  # a numpy.str_ becomes a plain str
    seen: "set[str]" = set()
    duplicated: "list[str]" = []
    for name in plain:
        if name in seen and name not in duplicated:
            duplicated.append(name)
        seen.add(name)
    if duplicated:
        shown = ", ".join(repr(n) for n in duplicated[:_MAX_LISTED]) + (", ..." if len(duplicated) > _MAX_LISTED else "")
        raise DataError(f"feature_names must be unique, but {shown} appear(s) more than once.")
    return plain


__all__ = ["validate_feature_names"]
