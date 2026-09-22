"""Milestone 120 tests: `forge.data.load_csv(..., columns=...)` / `load_csv_features(..., columns=...)`.

The one new capability this milestone adds to the CSV reader: explicit column selection, so a real CSV
that carries a non-feature column (an `id`, say) can be read without first rewriting the file. These
tests pin exactly what the milestone brief specifies and nothing more:

- **order is authoritative** -- the returned features are in `columns`' order, not the file's;
- **selection, not detection** -- an unselected column (an `id`, even a non-numeric one) is never read,
  never validated, never guessed at; there is still no automatic id/timestamp/target recognition;
- **rejections** -- unknown column, duplicate column, the target listed in `columns`, an empty
  selection, a non-list `columns`, all clearly named -- and that omitting `columns=` is byte-identical
  to the pre-Milestone-120 "every other column is a feature" contract (`test_csv_reader.py`).

`load_csv()`'s M119 `return_feature_names=True` and `load_csv_features()`'s own names are covered
together here since selection and naming compose (the returned names are exactly `columns`, in order).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge.data import load_csv, load_csv_features
from forge.exceptions import DataError

BASIC = "id,a,target,b,c\n1,1,10,2,3\n2,4,20,5,6\n3,7,30,8,9\n"


def write(tmp_path, text: str, name: str = "d.csv") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# ================================================================================== load_csv(..., columns=...)


def test_columns_selects_exactly_those_columns(tmp_path):
    X, y = load_csv(write(tmp_path, BASIC), target="target", columns=["a", "b"])
    np.testing.assert_array_equal(X, [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    assert y.tolist() == [10.0, 20.0, 30.0]


def test_order_is_authoritative_not_file_order(tmp_path):
    """`columns=["c", "a"]` on a file whose header is `a, ..., c` must return `[c, a]`, not `[a, c]`."""
    X, y, names = load_csv(write(tmp_path, BASIC), target="target", columns=["c", "a"], return_feature_names=True)
    assert names == ["c", "a"]
    np.testing.assert_array_equal(X, [[3.0, 1.0], [6.0, 4.0], [9.0, 7.0]])


def test_return_feature_names_is_exactly_columns_in_order(tmp_path):
    _, _, names = load_csv(write(tmp_path, BASIC), target="target", columns=["b", "a"], return_feature_names=True)
    assert names == ["b", "a"]


def test_an_unselected_column_is_ignored_even_if_non_numeric(tmp_path):
    """The whole point: an id column need not even be numeric when it is not selected."""
    text = "id,a,target,b\nPT-1,1,10,2\nPT-2,4,20,5\n"
    X, y = load_csv(write(tmp_path, text), target="target", columns=["a", "b"])
    np.testing.assert_array_equal(X, [[1.0, 2.0], [4.0, 5.0]])
    assert y.tolist() == [10.0, 20.0]


def test_all_non_target_columns_can_be_selected_explicitly_no_special_flag(tmp_path):
    X, y = load_csv(write(tmp_path, BASIC), target="target", columns=["id", "a", "b", "c"])
    np.testing.assert_array_equal(X, [[1.0, 1.0, 2.0, 3.0], [2.0, 4.0, 5.0, 6.0], [3.0, 7.0, 8.0, 9.0]])


def test_selecting_all_columns_in_file_order_matches_the_unselected_default(tmp_path):
    path = write(tmp_path, BASIC)
    X1, y1 = load_csv(path, target="target")
    X2, y2 = load_csv(path, target="target", columns=["id", "a", "b", "c"])
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_omitting_columns_still_treats_every_non_target_column_as_a_feature(tmp_path):
    """`columns=None` (the default) is byte-identical to the pre-Milestone-120 contract: no automatic exclusion."""
    path = write(tmp_path, BASIC)
    X, y, names = load_csv(path, target="target", return_feature_names=True)
    assert names == ["id", "a", "b", "c"]  # 'id' is read as a feature -- there is no automatic id detection
    np.testing.assert_array_equal(X, [[1.0, 1.0, 2.0, 3.0], [2.0, 4.0, 5.0, 6.0], [3.0, 7.0, 8.0, 9.0]])


def test_a_non_numeric_id_column_is_rejected_without_explicit_selection(tmp_path):
    """Without columns=, a non-numeric id is a real numeric-feature violation, not silently skipped."""
    text = "id,a,target\nPT-1,1,10\n"
    with pytest.raises(DataError, match="not a number"):
        load_csv(write(tmp_path, text, "d2.csv"), target="target")


# -------------------------------------------------------------------------------------------- rejections


def test_duplicate_column_is_rejected(tmp_path):
    with pytest.raises(DataError, match=r"columns must be unique.*'a'"):
        load_csv(write(tmp_path, BASIC), target="target", columns=["a", "a", "b"])


def test_unknown_column_is_rejected_and_named(tmp_path):
    with pytest.raises(DataError, match=r"'NotAColumn'"):
        load_csv(write(tmp_path, BASIC), target="target", columns=["a", "NotAColumn"])


def test_target_listed_in_columns_is_rejected_not_silently_dropped(tmp_path):
    with pytest.raises(DataError, match=r"includes the target column 'target'"):
        load_csv(write(tmp_path, BASIC), target="target", columns=["a", "target"])


def test_empty_selection_is_rejected(tmp_path):
    with pytest.raises(DataError, match="must not be empty"):
        load_csv(write(tmp_path, BASIC), target="target", columns=[])


def test_a_bare_string_is_rejected_not_read_as_a_list_of_characters(tmp_path):
    with pytest.raises(DataError, match="must be a list"):
        load_csv(write(tmp_path, BASIC), target="target", columns="ab")


def test_a_non_list_is_rejected(tmp_path):
    with pytest.raises(DataError, match="must be a list"):
        load_csv(write(tmp_path, BASIC), target="target", columns=42)


def test_a_blank_name_is_rejected(tmp_path):
    with pytest.raises(DataError, match="empty or blank"):
        load_csv(write(tmp_path, BASIC), target="target", columns=["a", "  "])


def test_error_names_the_columns_file_has(tmp_path):
    message = None
    try:
        load_csv(write(tmp_path, BASIC), target="target", columns=["nope"])
    except DataError as exc:
        message = str(exc)
    assert message is not None and "'id'" in message and "'a'" in message


# ================================================================================== load_csv_features(..., columns=...)


def test_load_csv_features_default_is_every_column(tmp_path):
    text = "id,a,b\n1,2,3\n4,5,6\n"
    X, names = load_csv_features(write(tmp_path, text))
    assert names == ["id", "a", "b"]
    np.testing.assert_array_equal(X, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_load_csv_features_columns_selects_and_orders(tmp_path):
    text = "id,a,b\n1,2,3\n4,5,6\n"
    X, names = load_csv_features(write(tmp_path, text), columns=["b", "a"])
    assert names == ["b", "a"]
    np.testing.assert_array_equal(X, [[3.0, 2.0], [6.0, 5.0]])


def test_load_csv_features_ignores_an_unselected_non_numeric_id(tmp_path):
    text = "id,a,b\nPT-1,2,3\nPT-2,5,6\n"
    X, names = load_csv_features(write(tmp_path, text), columns=["a", "b"])
    assert names == ["a", "b"]
    np.testing.assert_array_equal(X, [[2.0, 3.0], [5.0, 6.0]])


def test_load_csv_features_unknown_column_is_rejected(tmp_path):
    text = "id,a,b\n1,2,3\n"
    with pytest.raises(DataError, match="'nope'"):
        load_csv_features(write(tmp_path, text), columns=["a", "nope"])


def test_load_csv_features_duplicate_is_rejected(tmp_path):
    text = "id,a,b\n1,2,3\n"
    with pytest.raises(DataError, match="unique"):
        load_csv_features(write(tmp_path, text), columns=["a", "a"])


def test_load_csv_features_empty_selection_is_rejected(tmp_path):
    text = "id,a,b\n1,2,3\n"
    with pytest.raises(DataError, match="must not be empty"):
        load_csv_features(write(tmp_path, text), columns=[])


# ================================================================================== labels + missing values unaffected


def test_columns_works_with_class_labels(tmp_path):
    text = "id,f1,label,f2\nA,1,yes,2\nB,3,no,4\n"
    X, y, names = load_csv(
        write(tmp_path, text), target="label", labels=True, columns=["f1", "f2"], return_feature_names=True,
    )
    assert names == ["f1", "f2"] and y.tolist() == ["yes", "no"]
    np.testing.assert_array_equal(X, [[1.0, 2.0], [3.0, 4.0]])


def test_missing_value_in_a_selected_column_still_rejected(tmp_path):
    text = "id,a,target,b\n1,,10,2\n"
    with pytest.raises(DataError, match="empty"):
        load_csv(write(tmp_path, text), target="target", columns=["a", "b"])


def test_missing_value_in_an_unselected_column_is_not_even_looked_at(tmp_path):
    """columns= does not enable imputation -- it simply never reads a column that was not selected."""
    text = "id,a,target,b\n,1,10,2\n"
    X, y = load_csv(write(tmp_path, text), target="target", columns=["a", "b"])
    np.testing.assert_array_equal(X, [[1.0, 2.0]])
