"""Milestone 118 tests: `forge.data.load_csv()`, the CSV -> `(X, y)` boundary.

The reader is tested as a boundary and nothing else: what text becomes which array, and that everything the
documented contract rejects is rejected with a message that names the file, line and column. Every expected
array is written out by hand; nothing is checked by round-tripping through the code under test. Training,
evaluation and the CLI over these arrays are `tests/test_csv_tabular_workflow.py` / `tests/test_cli_evaluate_csv.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.data import load_csv
from forge.exceptions import DataError

REPO_ROOT = Path(__file__).resolve().parents[1]
PIMA = REPO_ROOT / "examples" / "tabular_diabetes" / "data" / "diabetes.csv"


def write(tmp_path, text, name="data.csv", *, raw=None):
    path = tmp_path / name
    path.write_bytes(raw if raw is not None else text.encode("utf-8"))
    return path


def rejected(tmp_path, text, *, match, target="y", labels=False, raw=None, name="data.csv"):
    path = write(tmp_path, text, name, raw=raw)
    with pytest.raises(DataError) as info:
        load_csv(path, target=target, labels=labels)
    message = str(info.value)
    assert match in message, message
    return message


# ------------------------------------------------------------------------------------------------ the happy path


def test_public_surface():
    assert forge.data.load_csv is load_csv and "load_csv" in forge.data.__all__


def test_reads_features_and_target_exactly(tmp_path):
    path = write(tmp_path, "a,b,y\n1,2.5,10\n-3,.5,20.25\n+4.,1e-3,-30\n")
    X, y = load_csv(path, target="y")
    np.testing.assert_array_equal(X, np.array([[1.0, 2.5], [-3.0, 0.5], [4.0, 0.001]]))
    np.testing.assert_array_equal(y, np.array([10.0, 20.25, -30.0]))
    assert X.dtype == np.float64 and y.dtype == np.float64 and X.shape == (3, 2) and y.shape == (3,)


def test_values_are_bit_exact_doubles_not_rounded(tmp_path):
    path = write(tmp_path, "a,y\n0.1,1\n0.30000000000000004,2\n123456789.12345678,3\n4.5260000000000000e+005,4\n")
    X, _ = load_csv(path, target="y")
    assert X[:, 0].tolist() == [0.1, 0.30000000000000004, 123456789.12345678, 452600.0]


def test_the_target_is_removed_from_the_features_wherever_it_is(tmp_path):
    """Mutation 1: a target left in X leaks the answer into the features."""
    for header, target, feature_cols in (
        ("t,f1,f2", "t", [1, 2]), ("f1,t,f2", "t", [0, 2]), ("f1,f2,t", "t", [0, 1]),
    ):
        rows = [[7.0, 1.0, 2.0], [8.0, 3.0, 4.0]]
        text = header + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n"
        X, y = load_csv(write(tmp_path, text), target=target)
        data = np.array(rows)
        t = header.split(",").index("t")
        np.testing.assert_array_equal(y, data[:, t])
        np.testing.assert_array_equal(X, data[:, [i for i in range(3) if i != t]])
        assert X.shape == (2, 2) and not np.isin(y, X).any()


def test_feature_columns_keep_file_order_not_alphabetical(tmp_path):
    """Mutation 2: reordering (sorting) the columns changes what each model input means."""
    path = write(tmp_path, "zeta,alpha,mid,y\n1,2,3,9\n4,5,6,9\n")
    X, _ = load_csv(path, target="y")
    np.testing.assert_array_equal(X, [[1, 2, 3], [4, 5, 6]])


def test_the_target_must_be_named_it_is_not_the_last_column(tmp_path):
    path = write(tmp_path, "price,area,rooms\n100,10,2\n200,20,3\n")
    X, y = load_csv(path, target="price")
    np.testing.assert_array_equal(y, [100, 200])
    np.testing.assert_array_equal(X, [[10, 2], [20, 3]])
    X2, y2 = load_csv(path, target="rooms")
    np.testing.assert_array_equal(y2, [2, 3])
    np.testing.assert_array_equal(X2, [[100, 10], [200, 20]])


def test_a_real_file_pima(tmp_path):
    X, y = load_csv(PIMA, target="Outcome", labels=True)
    assert X.shape == (768, 8) and y.shape == (768,) and y.dtype == np.int64
    assert set(np.unique(y)) == {0, 1} and int((y == 1).sum()) == 268
    reference = np.genfromtxt(PIMA, delimiter=",", skip_header=1)
    np.testing.assert_array_equal(X, reference[:, :-1])           # bit-identical to the NumPy reference
    np.testing.assert_array_equal(y, reference[:, -1].astype(np.int64))


def test_accepts_str_and_pathlike(tmp_path):
    path = write(tmp_path, "a,y\n1,2\n")
    for given in (path, str(path)):
        X, y = load_csv(given, target="y")
        assert X.tolist() == [[1.0]] and y.tolist() == [2.0]


def test_returns_fresh_arrays_and_never_touches_the_file(tmp_path):
    path = write(tmp_path, "a,y\n1,2\n3,4\n")
    before = path.read_bytes()
    first = load_csv(path, target="y")
    first[0][0, 0] = 99.0
    second = load_csv(path, target="y")
    assert second[0][0, 0] == 1.0 and path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.csv"]


# ------------------------------------------------------------------------------------------------ format details


def test_bom_crlf_blank_lines_whitespace_and_no_trailing_newline(tmp_path):
    raw = b"\xef\xbb\xbfa , b ,y\r\n\r\n 1 , 2 ,3\r\n\r\n4,5,6"
    X, y = load_csv(write(tmp_path, "", raw=raw), target="y")
    np.testing.assert_array_equal(X, [[1, 2], [4, 5]])
    np.testing.assert_array_equal(y, [3, 6])


def test_quoted_header_names_and_quoted_numbers(tmp_path):
    path = write(tmp_path, '"strength (MPa), 28d","a ""quoted"" name",y\n"1.5","2",3\n')
    X, y = load_csv(path, target="y")
    np.testing.assert_array_equal(X, [[1.5, 2.0]])
    with pytest.raises(DataError, match="strength"):     # the whole quoted name, comma and all, is one column
        load_csv(path, target="strength")
    assert load_csv(path, target="strength (MPa), 28d")[1].tolist() == [1.5]
    assert load_csv(path, target='a "quoted" name')[1].tolist() == [2.0]


def test_a_quoted_text_label_may_contain_commas_and_newlines(tmp_path):
    path = write(tmp_path, 'f,label\n1,"a, b"\n2,"line1\nline2"\n3,plain\n')
    _, y = load_csv(path, target="label", labels=True)
    assert y.tolist() == ["a, b", "line1\nline2", "plain"]


# ------------------------------------------------------------------------------------------------ labels


def test_integer_labels_become_int64_indices(tmp_path):
    _, y = load_csv(write(tmp_path, "f,c\n1,0\n2,1\n3,2\n4,1\n"), target="c", labels=True)
    assert y.dtype == np.int64 and y.tolist() == [0, 1, 2, 1]


def test_whole_number_floats_are_the_same_integers(tmp_path):
    _, y = load_csv(write(tmp_path, "f,c\n1,0.0\n2,1.\n3,1e0\n4,+2\n"), target="c", labels=True)
    assert y.dtype == np.int64 and y.tolist() == [0, 1, 1, 2]


def test_text_labels_are_verbatim_strings(tmp_path):
    _, y = load_csv(write(tmp_path, "f,species\n1, versicolor \n2,setosa\n3,Setosa\n"), target="species", labels=True)
    assert y.dtype.kind == "U" and y.tolist() == ["versicolor", "setosa", "Setosa"]   # stripped, case kept


def test_label_order_is_never_inferred_from_row_order(tmp_path):
    """The reader returns labels only; the class list belongs to the training/evaluation API."""
    a = load_csv(write(tmp_path, "f,c\n1,zebra\n2,apple\n", name="a.csv"), target="c", labels=True)[1]
    b = load_csv(write(tmp_path, "f,c\n1,apple\n2,zebra\n", name="b.csv"), target="c", labels=True)[1]
    assert a.tolist() == ["zebra", "apple"] and b.tolist() == ["apple", "zebra"]


def test_a_label_column_that_mixes_numbers_and_text_is_an_error(tmp_path):
    message = rejected(tmp_path, "f,c\n1,0\n2,1\n3,O\n", target="c", labels=True, match="mixes numbers and text")
    assert "line 4" in message and "'O'" in message


def test_a_fractional_number_is_not_a_class_index(tmp_path):
    message = rejected(tmp_path, "f,c\n1,0\n2,0.5\n", target="c", labels=True, match="not a class index")
    assert "line 3" in message and "0.5" in message


@pytest.mark.parametrize("cell", ["nan", "NaN", "inf", "-Infinity"])
def test_a_nan_marker_is_never_a_class_name(tmp_path, cell):
    rejected(tmp_path, f"f,c\n1,cat\n2,{cell}\n", target="c", labels=True, match="NaN/Inf")


def test_numeric_targets_are_the_default_and_text_is_named(tmp_path):
    """Mutation 8: silent coercion. A regression target with text in it names the cell, never becomes 0/NaN."""
    message = rejected(tmp_path, "f,y\n1,5\n2,n/a\n", match="not a number")
    assert "line 3" in message and "column 'y'" in message and "'n/a'" in message


# ------------------------------------------------------------------------------------------------ what is rejected


def test_missing_file_directory_and_bad_arguments(tmp_path):
    with pytest.raises(DataError, match="CSV file not found"):
        load_csv(tmp_path / "nope.csv", target="y")
    with pytest.raises(DataError, match="not a file"):
        load_csv(tmp_path, target="y")
    path = write(tmp_path, "a,y\n1,2\n")
    for bad_path in (None, 5, ""):
        with pytest.raises(DataError, match="path"):
            load_csv(bad_path, target="y")
    for bad_target in (None, 3, "", "  "):
        with pytest.raises(DataError, match="target"):
            load_csv(path, target=bad_target)
    for bad_labels in (None, 1, "yes"):
        with pytest.raises(DataError, match="labels"):
            load_csv(path, target="y", labels=bad_labels)


@pytest.mark.parametrize("text", ["", "\n\n", "\r\n"])
def test_empty_file(tmp_path, text):
    rejected(tmp_path, text, match="is empty")


def test_a_whitespace_only_first_line_is_an_unnamed_header_not_an_empty_file(tmp_path):
    rejected(tmp_path, "  \n", match="have no name")


def test_a_header_only_file_has_no_data(tmp_path):
    rejected(tmp_path, "a,b,y\n", match="no data rows")
    rejected(tmp_path, "a,b,y", match="no data rows", name="no_newline.csv")
    rejected(tmp_path, "a,b,y\n\n\n", match="no data rows", name="blank_tail.csv")


def test_a_file_without_a_header_is_not_guessed_at(tmp_path):
    message = rejected(tmp_path, "6,148,72,1\n1,85,66,0\n", match="looks like data, not a header")
    assert "header row" in message
    rejected(tmp_path, "1.5,2,3\n4,5,6\n", match="looks like data", target="3")


def test_another_delimiter_is_recognised_as_such_not_parsed(tmp_path):
    rejected(tmp_path, "a;b;y\n1;2;3\n", match="comma-delimited")
    rejected(tmp_path, "a\tb\ty\n1\t2\t3\n", match="comma-delimited", name="tabs.csv")


def test_duplicate_column_names(tmp_path):
    message = rejected(tmp_path, "a,b,a,y\n1,2,3,4\n", match="duplicate column name(s) ['a']")
    assert "must be unique" in message
    rejected(tmp_path, "a, a ,y\n1,2,3\n", match="duplicate", name="stripped.csv")     # equal after whitespace stripping


def test_the_target_column_appearing_twice(tmp_path):
    message = rejected(tmp_path, "a,y,b,y\n1,2,3,4\n", match="appears 2 times")
    assert "'y'" in message and "[2, 4]" in message


def test_a_missing_target_column_lists_the_columns_it_did_find(tmp_path):
    message = rejected(tmp_path, "Glucose,BMI,Outcome\n1,2,3\n", target="outcome", match="'outcome' is not in the header")
    assert "'Glucose', 'BMI', 'Outcome'" in message and "case-sensitive" in message


def test_a_long_header_is_abbreviated_in_the_missing_target_message(tmp_path):
    header = ",".join(f"c{i}" for i in range(40))
    message = rejected(tmp_path, header + "\n" + ",".join("1" for _ in range(40)) + "\n", match="not in the header")
    assert "40" in message and "c39" not in message and "..." in message


def test_an_unnamed_column(tmp_path):
    message = rejected(tmp_path, ",a,y\n0,1,2\n", match="have no name")
    assert "[1]" in message


def test_a_target_only_file_has_no_features(tmp_path):
    rejected(tmp_path, "y\n1\n2\n", match="no feature columns")


@pytest.mark.parametrize("row, line, count", [("1,2", 3, 2), ("1,2,3,4", 3, 4), ("1", 3, 1)])
def test_inconsistent_row_lengths_name_the_line(tmp_path, row, line, count):
    message = rejected(tmp_path, f"a,b,y\n1,2,3\n{row}\n4,5,6\n", match=f"line {line}")
    assert f"{count} field(s)" in message and "3 column(s)" in message


def test_the_line_number_counts_physical_lines_including_blank_ones(tmp_path):
    rejected(tmp_path, "a,y\n\n1,2\n\n3\n", match="line 5")


@pytest.mark.parametrize("cell", ["abc", "12abc", "1,5x", "0x1F", "1_000", "١٢٣", "--1", "1e", "e5", ".", "+", "1 2", "$5", "5%", "1.2.3", "true", "None", "N/A", "-"])
def test_no_non_numeric_feature_is_ever_coerced(tmp_path, cell):
    """Mutation 8. `float()` alone would accept `1_000` and the Arabic-Indic digits; the reader must not."""
    text = f'a,b,y\n1,2,3\n4,"{cell}",6\n'
    message = rejected(tmp_path, text, match="is not a number")
    assert "line 3" in message and "column 'b'" in message


def test_a_non_numeric_feature_message_says_what_to_do(tmp_path):
    message = rejected(tmp_path, "Id,species,y\nA17,setosa,1\n", match="'A17' is not a number")
    assert "column 'Id'" in message and "no categorical/string features" in message and "remove or convert" in message


def test_empty_numeric_fields_are_errors_never_zero_or_a_mean(tmp_path):
    message = rejected(tmp_path, "a,b,y\n1,2,3\n4,,6\n", match="empty feature")
    assert "line 3" in message and "column 'b'" in message and "does not impute" in message
    rejected(tmp_path, "a,b,y\n1,2,3\n4,   ,6\n", match="empty feature", name="spaces.csv")
    rejected(tmp_path, "a,b,y\n1,2,3\n,,\n", match="empty feature", name="all_empty.csv")     # not a blank line
    rejected(tmp_path, "a,b,y\n1,2,\n", match="empty target", name="empty_target.csv")


@pytest.mark.parametrize("cell", ["nan", "NaN", "NAN", "inf", "-inf", "+Inf", "infinity", "-Infinity"])
def test_nan_and_inf_are_named_as_such(tmp_path, cell):
    for header_position in ("feature", "target"):
        text = f"a,y\n{cell},1\n" if header_position == "feature" else f"a,y\n1,{cell}\n"
        message = rejected(tmp_path, text, match="not a finite number")
        assert "NaN/Inf" in message and cell in message


def test_a_value_that_overflows_double_precision_is_an_error(tmp_path):
    rejected(tmp_path, "a,y\n1e999,1\n", match="overflows to infinity")
    rejected(tmp_path, "a,y\n1,-1e999\n", match="overflows to infinity", name="target.csv")


@pytest.mark.parametrize("text, match", [
    ('a,y\n"1,2\n', "malformed CSV"),                       # an unterminated quote
    ('a,y\n"1"x,2\n', "malformed CSV"),                     # text after a closing quote
    ('a,"b"c,y\n1,2,3\n', "malformed CSV"),                 # ... in the header
])
def test_malformed_quoting_is_a_named_error(tmp_path, text, match):
    message = rejected(tmp_path, text, match=match)
    assert "line " in message and "data.csv" in message


def test_invalid_utf8_is_named(tmp_path):
    message = rejected(tmp_path, "", raw=b"a,y\n1,2\n\xff\xfe3,4\n", match="not valid UTF-8")
    assert "byte offset" in message and "re-save" in message


def test_a_binary_file_is_an_error_not_a_crash(tmp_path):
    with pytest.raises(DataError):
        load_csv(write(tmp_path, "", raw=bytes(range(256)) * 4), target="y")


def test_every_error_names_the_file(tmp_path):
    for text in ("", "a,y\n", "a,y\n1,x\n", "a,y\n1\n", "a;y\n1;2\n"):
        path = write(tmp_path, text, name="the_file.csv")
        with pytest.raises(DataError) as info:
            load_csv(path, target="y")
        assert "the_file.csv" in str(info.value), str(info.value)


def test_errors_are_forge_errors_so_the_cli_prints_them_without_a_traceback(tmp_path):
    with pytest.raises(forge.exceptions.ForgeError):
        load_csv(write(tmp_path, "a,y\n1,x\n"), target="y")


# ------------------------------------------------------------------------------------------------ dependencies


def test_reading_a_csv_imports_no_dataframe_library(tmp_path):
    """pandas is installed in the dev environment, so this proves Forge does not use it (not that it is absent)."""
    path = write(tmp_path, "a,y\n1,2\n3,4\n")
    code = (
        "import sys, forge\n"
        f"X, y = forge.data.load_csv(r'{path}', target='y')\n"
        "assert X.tolist() == [[1.0], [3.0]]\n"
        "banned = [m for m in ('pandas', 'polars', 'pyarrow', 'openpyxl') if m in sys.modules]\n"
        "assert not banned, banned\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)})
    assert result.returncode == 0, result.stderr


def reader_code_without_docstrings() -> str:
    """The reader's executable code (identifiers, calls, string literals), with every docstring removed."""
    import ast

    tree = ast.parse((REPO_ROOT / "forge" / "data" / "csv_reader.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_no_eval_pickle_or_dataframe_in_the_reader_source():
    code = reader_code_without_docstrings()
    for banned in ("eval(", "exec(", "pickle", "pandas", "genfromtxt", "loadtxt", "allow_pickle", "DataFrame"):
        assert banned not in code, banned
