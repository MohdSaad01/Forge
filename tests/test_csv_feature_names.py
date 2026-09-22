"""Milestone 119 tests: the CSV reader hands back column *names* (`return_feature_names=`, `load_csv_features()`).

The reader stays a boundary: it reads text, checks it means one thing, and returns arrays plus the header
names of exactly those arrays' columns. It knows nothing about artifacts. The M118 contract (`test_csv_reader.py`)
is untouched -- these tests add only what is new, and pin that the default return is still `(X, y)`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from forge.data import load_csv, load_csv_features
from forge.data.feature_names import validate_feature_names
from forge.exceptions import DataError


def write(tmp_path, text: str, name: str = "d.csv", *, newline="\n") -> Path:
    path = tmp_path / name
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return path


BASIC = "a,target,b,c\n1,10,2,3\n4,20,5,6\n7,30,8,9\n"


# ============================================================================== load_csv(..., return_feature_names=)


def test_the_default_return_is_still_a_pair(tmp_path):
    result = load_csv(write(tmp_path, BASIC), target="target")
    assert isinstance(result, tuple) and len(result) == 2


def test_return_feature_names_appends_the_names_of_exactly_the_columns_of_x(tmp_path):
    X, y, names = load_csv(write(tmp_path, BASIC), target="target", return_feature_names=True)
    assert names == ["a", "b", "c"] and X.shape == (3, 3) and y.tolist() == [10.0, 20.0, 30.0]
    assert X[:, names.index("b")].tolist() == [2.0, 5.0, 8.0]              # the name really is that column


@pytest.mark.parametrize("target, expected", [("a", ["target", "b", "c"]), ("target", ["a", "b", "c"]), ("c", ["a", "target", "b"])])
def test_the_target_is_excluded_wherever_it_is_and_the_rest_keep_file_order(tmp_path, target, expected):
    _, _, names = load_csv(write(tmp_path, BASIC), target=target, return_feature_names=True)
    assert names == expected


def test_asking_for_the_names_changes_no_value(tmp_path):
    path = write(tmp_path, BASIC)
    X1, y1 = load_csv(path, target="target")
    X2, y2, _ = load_csv(path, target="target", return_feature_names=True)
    assert X1.dtype == X2.dtype and np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_names_come_with_class_labels_too(tmp_path):
    path = write(tmp_path, "f1,label,f2\n1,yes,2\n3,no,4\n")
    X, y, names = load_csv(path, target="label", labels=True, return_feature_names=True)
    assert names == ["f1", "f2"] and y.tolist() == ["yes", "no"]


def test_header_whitespace_is_stripped_and_nothing_else_about_a_name_changes(tmp_path):
    """The M118 header contract (cells are stripped) is the reader's, documented; case and inner spaces stay."""
    path = write(tmp_path, "  Age  , Blood Pressure ,target\n1,2,3\n")
    _, _, names = load_csv(path, target="target", return_feature_names=True)
    assert names == ["Age", "Blood Pressure"]


def test_names_differing_only_by_case_are_two_columns_with_two_names(tmp_path):
    path = write(tmp_path, "Age,age,target\n1,2,3\n")
    _, _, names = load_csv(path, target="target", return_feature_names=True)
    assert names == ["Age", "age"] and validate_feature_names(names) == ("Age", "age")


def test_a_quoted_header_name_with_a_comma_survives_exactly(tmp_path):
    path = write(tmp_path, 'a,"b, with comma",target\n1,2,3\n')
    _, _, names = load_csv(path, target="target", return_feature_names=True)
    assert names == ["a", "b, with comma"]


def test_a_utf8_bom_and_crlf_do_not_leak_into_the_first_name(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"\xef\xbb\xbfalpha,target\r\n1,2\r\n")
    _, _, names = load_csv(path, target="target", return_feature_names=True)
    assert names == ["alpha"]


def test_returned_names_are_always_a_valid_feature_name_list(tmp_path):
    """Whatever a CSV yields can be handed straight to `feature_names=`: same rules, one definition."""
    _, _, names = load_csv(write(tmp_path, BASIC), target="target", return_feature_names=True)
    assert validate_feature_names(names) == tuple(names)


@pytest.mark.parametrize("flag", [1, 0, "yes", None, np.bool_(True)])
def test_return_feature_names_must_be_a_bool(tmp_path, flag):
    with pytest.raises(DataError, match="return_feature_names must be True or False"):
        load_csv(write(tmp_path, BASIC), target="target", return_feature_names=flag)


def test_a_duplicate_header_is_still_refused_so_names_are_always_unique(tmp_path):
    with pytest.raises(DataError, match=r"duplicate column name\(s\) \['a'\]"):
        load_csv(write(tmp_path, "a,a,target\n1,2,3\n"), target="target", return_feature_names=True)


def test_errors_are_unchanged_by_asking_for_names(tmp_path):
    path = write(tmp_path, "a,target\nx,1\n")
    with pytest.raises(DataError) as plain:
        load_csv(path, target="target")
    with pytest.raises(DataError) as named:
        load_csv(path, target="target", return_feature_names=True)
    assert str(plain.value) == str(named.value)


# ============================================================================== load_csv_features


def test_load_csv_features_returns_every_column_in_file_order(tmp_path):
    X, names = load_csv_features(write(tmp_path, BASIC))
    assert names == ["a", "target", "b", "c"] and X.shape == (3, 4) and X.dtype == np.float64
    assert X[:, 1].tolist() == [10.0, 20.0, 30.0]                          # 'target' is just another column here


def test_load_csv_features_agrees_with_load_csv_on_the_shared_columns(tmp_path):
    path = write(tmp_path, BASIC)
    X_all, names_all = load_csv_features(path)
    X, _, names = load_csv(path, target="target", return_feature_names=True)
    keep = [names_all.index(n) for n in names]
    assert np.array_equal(X_all[:, keep], X)


def test_load_csv_features_accepts_a_single_column(tmp_path):
    X, names = load_csv_features(write(tmp_path, "only\n1\n2\n"))
    assert names == ["only"] and X.tolist() == [[1.0], [2.0]]


def test_load_csv_features_takes_a_pathlike_and_a_str(tmp_path):
    path = write(tmp_path, BASIC)
    assert load_csv_features(str(path))[1] == load_csv_features(path)[1] == ["a", "target", "b", "c"]


@pytest.mark.parametrize("text, message", [
    ("a,a,b\n1,2,3\n", r"duplicate column name\(s\) \['a'\]"),
    ("a,,b\n1,2,3\n", r"header column\(s\) \[2\] have no name"),
    ("1,2,3\n4,5,6\n", "looks like data, not a header"),
    ("a,b\n", "has a header but no data rows"),
    ("", "is empty: it has no header row"),
    ("a,b\n1\n", "line 2: 1 field\\(s\\), but the header has 2"),
    ("a,b\n1,\n", "line 2, column 'b': empty feature"),
    ("a,b\n1,oops\n", "line 2, column 'b': 'oops' is not a number"),
    ("a,b\n1,nan\n", "not a finite number"),
    ("a,b\n1,1_000\n", "is not a number"),
    ("a;b\n1;2\n", "Forge reads comma-delimited CSV only"),
    ('a,"b\n1,2\n', "malformed CSV"),
])
def test_load_csv_features_applies_the_same_file_contract(tmp_path, text, message):
    with pytest.raises(DataError, match=message):
        load_csv_features(write(tmp_path, text))


def test_load_csv_features_reports_the_file(tmp_path):
    with pytest.raises(DataError, match="CSV file not found"):
        load_csv_features(tmp_path / "missing.csv")
    with pytest.raises(DataError, match="CSV path is not a file"):
        load_csv_features(tmp_path)
    for bad in ("", None, 5, b"x.csv"):
        with pytest.raises(DataError, match=r"load_csv_features\(\) path must be a non-empty file path"):
            load_csv_features(bad)


def test_load_csv_features_has_no_target_argument_and_no_labels():
    """Milestone 120 added `columns=` (column selection); there is still no `target=`/`labels=`."""
    import inspect

    assert list(inspect.signature(load_csv_features).parameters) == ["path", "columns"]


def test_non_utf8_input_is_reported_not_traceback(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes(b"a,b\n1,\xff\n")
    with pytest.raises(DataError, match="not valid UTF-8"):
        load_csv_features(path)


def test_blank_lines_are_skipped_as_in_load_csv(tmp_path):
    X, names = load_csv_features(write(tmp_path, "a,b\n\n1,2\n\n3,4\n"))
    assert names == ["a", "b"] and X.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_the_reader_is_still_independent_of_artifacts_and_training():
    """The M118 boundary: neither reader imports (or is told about) an artifact, a predictor or a trainer."""
    import ast

    source = Path(__file__).resolve().parents[1] / "forge" / "data" / "csv_reader.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("forge.training", "forge.serialization", ".training", ".serialization")) or m in
                   ("..training", "..serialization") for m in imported), imported
    assert os.path.basename(str(source)) == "csv_reader.py"
