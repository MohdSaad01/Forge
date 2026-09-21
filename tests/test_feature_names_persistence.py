"""Milestone 119 tests: what a saved artifact records about its input *columns*.

`InputSchema` used to be a width derived from the first `Linear`; now it can also carry
`feature_names`, persisted by `save_model(..., feature_names=)`. This file covers the artifact
side -- the name rules, the metadata written to disk, compatibility with every artifact that has
none, and the read side (`inspect_model()`, `forge model inspect`, `forge model convert`).
Using names to align input is `test_named_input.py`; training them in is
`test_training_feature_names.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.cli.main import main as cli_main
from forge.data.feature_names import validate_feature_names
from forge.exceptions import DataError, PersistenceError
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import InputSchema, inspect_model, save_model
from forge.serialization.model import FORMAT_VERSION, TARGET_TRANSFORM_FORMAT_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema_support import (  # noqa: E402
    FEATURES, metadata_of, tamper, train_classifier, train_regressor,
)


def mlp(width: int = 4) -> Sequential:
    return Sequential(Linear(width, 5), ReLU(), Linear(5, 1))


# ==================================================================== the name rules (one definition)


@pytest.mark.parametrize("names", [["a"], ["a", "b"], ("a", "b"), ["Age", "age"], [" Age", "Age"], ["a b", "a  b"]])
def test_valid_names_are_returned_exactly_as_given(names):
    assert validate_feature_names(names) == tuple(names)


def test_numpy_strings_become_plain_strings():
    checked = validate_feature_names(list(np.array(["a", "b"])))
    assert checked == ("a", "b") and all(type(n) is str for n in checked)


@pytest.mark.parametrize("bad, message", [
    ("abc", "must be a list of strings, got str"),
    (b"ab", "must be a list of strings, got bytes"),
    (None, "must be a list of strings, got NoneType"),
    (5, "must be a list of strings, got int"),
    ({"a", "b"}, "must be a list of strings, got set"),
    (np.array(["a", "b"]), "must be a list of strings, got ndarray"),
    ([], "must not be empty"),
    (["a", 3], "only strings, got int 3 at position 1"),
    ([None], "only strings, got NoneType"),
    ([b"a"], "only strings, got bytes"),
    (["a", ""], "empty or blank name, got '' at position 1"),
    (["a", "   "], "empty or blank name, got '   ' at position 1"),
    (["a", "\t"], "empty or blank name"),
    (["a", "b", "a"], "must be unique, but 'a' appear(s) more than once"),
    (["a", "a", "b", "b"], "'a', 'b' appear(s) more than once"),
])
def test_invalid_names_are_rejected_with_the_reason(bad, message):
    with pytest.raises(DataError) as excinfo:
        validate_feature_names(bad)
    assert message in str(excinfo.value) and str(excinfo.value).startswith("feature_names")


@pytest.mark.parametrize("pair", [("Age", "age"), ("Age", " Age"), ("Age", "Age "), ("a b", "a  b"), ("é", "é")])
def test_names_differing_only_by_case_or_whitespace_are_different_names(pair):
    """Exact preservation: there is no normalisation, so these are two distinct, both-valid, names."""
    assert validate_feature_names(list(pair)) == pair


# ==================================================================== save_model: what is written


def test_names_are_written_as_a_json_list_exactly_as_given(tmp_path):
    save_model(mlp(), str(tmp_path / "m.forge"), task="regression", feature_names=["Age", " age ", "b", "c"])
    md = metadata_of(tmp_path / "m.forge")
    assert md["feature_names"] == ["Age", " age ", "b", "c"]                  # case + whitespace untouched


def test_an_unnamed_artifact_has_no_feature_names_key_at_all(tmp_path):
    """Byte-for-byte what M118 wrote: the key is absent, not null -- so an unnamed file is unchanged."""
    save_model(mlp(), str(tmp_path / "m.forge"), task="regression")
    assert "feature_names" not in metadata_of(tmp_path / "m.forge")


def test_naming_an_artifact_adds_exactly_one_metadata_key(tmp_path):
    model = mlp()
    save_model(model, str(tmp_path / "plain.forge"), task="regression")
    save_model(model, str(tmp_path / "named.forge"), task="regression", feature_names=FEATURES)
    plain, named = metadata_of(tmp_path / "plain.forge"), metadata_of(tmp_path / "named.forge")
    assert set(named) - set(plain) == {"feature_names"} and set(plain) <= set(named)
    assert {k: v for k, v in named.items() if k != "feature_names"} == plain


def test_format_version_is_not_bumped_by_names(tmp_path):
    """Names do not change what any output *means*, so an older reader ignoring them is safe (see the M119 report)."""
    save_model(mlp(), str(tmp_path / "m.forge"), task="regression", feature_names=FEATURES)
    assert metadata_of(tmp_path / "m.forge")["forge_format_version"] == FORMAT_VERSION == 2


def test_names_and_a_target_transform_coexist_and_stay_version_3(tmp_path):
    result = train_regressor(tmp_path / "m.forge", target_transform="standardize")
    md = metadata_of(tmp_path / "m.forge")
    assert md["forge_format_version"] == TARGET_TRANSFORM_FORMAT_VERSION
    assert md["feature_names"] == FEATURES and "target_transform" in md
    info = inspect_model(str(tmp_path / "m.forge"))
    assert info.input_schema.feature_names == tuple(FEATURES) and info.target_transform == result.target_transform


def test_names_are_the_raw_columns_whatever_preprocessing_does(tmp_path):
    """4 in, 4 out, in the given order: ReplaceValue/Normalize neither rename nor add a name."""
    train_regressor(tmp_path / "m.forge", missing_columns=[1])
    assert metadata_of(tmp_path / "m.forge")["feature_names"] == FEATURES


# ==================================================================== save_model: rejected, and nothing written


@pytest.mark.parametrize("names", [
    ["a", "b", "c"], ["a", "b", "c", "d", "e"],            # wrong count
    ["a", "a", "b", "c"],                                  # duplicate
    ["a", "b", "c", ""], ["a", "b", "c", "  "],            # empty / blank
    ["a", "b", "c", 4], "abcd", [],                        # not strings / a bare str / empty
])
def test_bad_names_raise_persistence_error_and_write_no_file(tmp_path, names):
    with pytest.raises(PersistenceError, match="save_model\\(\\) "):
        save_model(mlp(), str(tmp_path / "m.forge"), task="regression", feature_names=names)
    assert not (tmp_path / "m.forge").exists() and list(tmp_path.iterdir()) == []


def test_wrong_count_message_names_both_counts(tmp_path):
    with pytest.raises(PersistenceError, match=r"3 name\(s\), but the model's first Linear layer takes 4"):
        save_model(mlp(), str(tmp_path / "m.forge"), task="regression", feature_names=["a", "b", "c"])


@pytest.mark.parametrize("task", [None, "classification", "segmentation", "sequence"])
def test_names_require_a_tabular_task(tmp_path, task):
    classes = ["a", "b"] if task in ("classification", "sequence") else None
    with pytest.raises(PersistenceError, match="feature_names= requires task in"):
        save_model(mlp(), str(tmp_path / "m.forge"), task=task, classes=classes, feature_names=FEATURES)
    assert list(tmp_path.iterdir()) == []


def test_names_need_a_model_whose_input_width_is_known(tmp_path):
    from forge.nn import Flatten

    model = Sequential(Flatten(), Linear(4, 1))          # first layer is not Linear: width not derivable
    with pytest.raises(PersistenceError, match="input width Forge can read"):
        save_model(model, str(tmp_path / "m.forge"), task="regression", feature_names=FEATURES)
    save_model(model, str(tmp_path / "ok.forge"), task="regression")   # ... but it is still saveable unnamed
    assert inspect_model(str(tmp_path / "ok.forge")).input_schema is None


# ==================================================================== reading: inspect_model / InputSchema


def test_inspect_model_exposes_the_names_as_a_tuple(tmp_path):
    train_regressor(tmp_path / "r.forge")
    train_classifier(tmp_path / "c.forge")
    for name in ("r.forge", "c.forge"):
        schema = inspect_model(str(tmp_path / name)).input_schema
        assert schema == InputSchema(feature_count=4, feature_names=tuple(FEATURES))
        assert isinstance(schema.feature_names, tuple)


def test_an_unnamed_artifact_reports_no_names_never_invented_ones(tmp_path):
    train_regressor(tmp_path / "m.forge", names=None)
    schema = inspect_model(str(tmp_path / "m.forge")).input_schema
    assert schema == InputSchema(feature_count=4) and schema.feature_names is None


def test_input_schema_still_compares_equal_to_the_old_width_only_form():
    assert InputSchema(feature_count=8) == InputSchema(feature_count=8, feature_names=None)
    assert InputSchema(1) != InputSchema(1, ("a",))
    assert InputSchema(feature_count=2, feature_names=("a", "b")) != InputSchema(feature_count=2, feature_names=("b", "a"))


def test_the_bundled_pre_m119_real_artifact_loads_as_unnamed():
    """A real artifact saved milestones ago: valid, width-only, and prediction is unchanged."""
    path = Path(__file__).resolve().parents[1] / "models" / "tabular_classifier" / "diabetes_classifier.forge"
    info = inspect_model(str(path))
    assert info.input_schema == InputSchema(feature_count=8, feature_names=None)
    assert "Feature names" not in str(info)
    assert forge.load_predictor(str(path)).predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])[0].label in ("no_diabetes", "diabetes")


def test_model_info_text_lists_names_only_when_present(tmp_path):
    train_classifier(tmp_path / "named.forge")
    train_classifier(tmp_path / "plain.forge", names=None)
    named, plain = str(inspect_model(str(tmp_path / "named.forge"))), str(inspect_model(str(tmp_path / "plain.forge")))
    assert "Feature names: income, age, rooms, distance" in named
    assert "Feature names" not in plain and "Input: 4 feature(s)" in plain      # unchanged for an unnamed artifact


# ==================================================================== reading a tampered / inconsistent file


@pytest.fixture()
def named_artifact(tmp_path):
    train_regressor(tmp_path / "m.forge")
    return tmp_path / "m.forge"


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m.update(feature_names=["a", "b", "c"]), "lists 3 name\\(s\\) but the saved model takes 4"),
    (lambda m: m.update(feature_names=["a", "b", "c", "d", "e"]), "lists 5 name\\(s\\) but the saved model takes 4"),
    (lambda m: m.update(feature_names="income"), "malformed 'feature_names' metadata"),
    (lambda m: m.update(feature_names={"0": "a"}), "malformed 'feature_names' metadata"),
    (lambda m: m.update(feature_names=["a", "b", "b", "c"]), "must be unique"),
    (lambda m: m.update(feature_names=["a", "b", 3, "c"]), "only strings"),
    (lambda m: m.update(feature_names=["a", "b", "", "c"]), "empty or blank"),
    (lambda m: m.update(feature_names=[]), "must not be empty"),
    (lambda m: m.update(task="classification", classes=["x", "y"]), "only .* artifacts"),
    (lambda m: m.pop("task"), "has a 'feature_names' entry but task=None"),
])
def test_a_feature_names_entry_that_save_model_could_not_have_written_is_refused(named_artifact, mutate, message):
    bad = tamper(named_artifact, mutate)
    with pytest.raises(PersistenceError, match=message):
        inspect_model(str(bad))
    with pytest.raises(PersistenceError, match=message):        # and therefore load_predictor() (which inspects first)
        forge.load_predictor(str(bad))


def test_names_on_an_architecture_with_no_derivable_width_are_refused(named_artifact):
    def flatten_first(md):
        md["root"]["children"]["0"]["type"] = "Flatten"          # the width can no longer be read from the tree

    with pytest.raises(PersistenceError, match="no input width to check it against"):
        inspect_model(str(tamper(named_artifact, flatten_first)))


def test_an_explicit_null_entry_is_the_same_as_no_entry(named_artifact):
    """`save_model()` never writes null; a hand-edited null means 'no names', not an error and not a guess."""
    schema = inspect_model(str(tamper(named_artifact, lambda m: m.update(feature_names=None)))).input_schema
    assert schema == InputSchema(feature_count=4, feature_names=None)


def test_load_model_does_not_depend_on_the_names(named_artifact):
    """The model itself is untouched by the names: stripping them changes no parameter."""
    stripped = tamper(named_artifact, lambda m: m.pop("feature_names"))
    for (na, pa), (nb, pb) in zip(forge.load_model(str(named_artifact)).named_parameters(),
                                  forge.load_model(str(stripped)).named_parameters()):
        assert na == nb and np.array_equal(pa.numpy(), pb.numpy())


# ==================================================================== forge model inspect / convert


def run_cli(argv, capsys):
    code = cli_main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_inspect_text_lists_the_feature_names(named_artifact, capsys):
    code, out, err = run_cli(["model", "inspect", named_artifact], capsys)
    assert code == 0 and err == ""
    assert "Input: 4 feature(s)\nFeature names: income, age, rooms, distance\n" in out.replace("\r\n", "\n")


def test_cli_inspect_json_has_the_names_and_is_valid_json(named_artifact, capsys):
    code, out, err = run_cli(["model", "inspect", named_artifact, "--json"], capsys)
    payload = json.loads(out)
    assert code == 0 and payload["input_feature_count"] == 4 and payload["input_feature_names"] == FEATURES


def test_cli_inspect_json_of_an_unnamed_artifact_has_null_names(tmp_path, capsys):
    train_regressor(tmp_path / "m.forge", names=None)
    code, out, _ = run_cli(["model", "inspect", tmp_path / "m.forge", "--json"], capsys)
    payload = json.loads(out)
    assert payload["input_feature_count"] == 4 and payload["input_feature_names"] is None


def test_cli_inspect_text_of_an_unnamed_artifact_is_unchanged(tmp_path, capsys):
    train_regressor(tmp_path / "m.forge", names=None)
    _, out, _ = run_cli(["model", "inspect", tmp_path / "m.forge"], capsys)
    assert "Input: 4 feature(s)" in out and "Feature names" not in out


def test_cli_inspect_json_of_a_non_tabular_artifact_has_null_names(tmp_path, capsys):
    save_model(mlp(), str(tmp_path / "m.forge"))                      # no task at all
    _, out, _ = run_cli(["model", "inspect", tmp_path / "m.forge", "--json"], capsys)
    payload = json.loads(out)
    assert payload["input_feature_count"] is None and payload["input_feature_names"] is None


def test_cli_convert_preserves_the_names(named_artifact, tmp_path, capsys):
    """Dropping them here would silently turn a column-checked artifact into a width-checked one."""
    out_path = tmp_path / "converted.forge"
    code, _, err = run_cli(["model", "convert", named_artifact, "--device", "cpu", "--output", out_path], capsys)
    assert code == 0, err
    assert metadata_of(out_path)["feature_names"] == FEATURES
    assert inspect_model(str(out_path)).input_schema == inspect_model(str(named_artifact)).input_schema


def test_cli_convert_of_an_unnamed_artifact_stays_unnamed(tmp_path, capsys):
    train_regressor(tmp_path / "m.forge", names=None)
    code, _, err = run_cli(["model", "convert", tmp_path / "m.forge", "--device", "cpu", "--output", tmp_path / "o.forge"], capsys)
    assert code == 0, err
    assert "feature_names" not in metadata_of(tmp_path / "o.forge")


def test_convert_preserves_names_alongside_a_target_transform(tmp_path, capsys):
    train_regressor(tmp_path / "m.forge", target_transform="standardize")
    code, _, err = run_cli(["model", "convert", tmp_path / "m.forge", "--device", "cpu", "--output", tmp_path / "o.forge"], capsys)
    assert code == 0, err
    md = metadata_of(tmp_path / "o.forge")
    assert md["feature_names"] == FEATURES and md["forge_format_version"] == TARGET_TRANSFORM_FORMAT_VERSION
