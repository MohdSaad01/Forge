"""Milestone 119 tests: a *named* tabular input is aligned to the artifact's feature names, or rejected.

The contract (`forge.training.inference._column_order()`), for an artifact that records feature names:

- exactly the artifact's names, in any order -> reordered into the artifact's order (names decide identity);
- anything else -> `DataError` naming what is missing / unexpected: a missing column, an unknown one, an extra
  one (an `id` is an extra column -- never recognised, never dropped), a duplicate, a name that differs by
  case or whitespace. Nothing is normalised, renamed, ignored or guessed;
- an artifact that records no names cannot verify anything: the input is used as given and a `UserWarning` says so;
- an unnamed array (`feature_names=None`) is checked for width only, exactly as before.

It is one rule reached by every route -- `ArtifactPredictor.predict()`, `predict_model()`,
`predict_tensor_artifact()`, `predict_tabular_classification_artifact()` and `ArtifactPredictor.evaluate()`.
The reorder tests are *non-vacuous*: each asserts first that the same columns in another order, unnamed,
really does change the prediction.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.exceptions import DataError
from forge.tensor.tensor import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema_support import (  # noqa: E402
    FEATURES, assert_order_matters, make_data, permute, result_fields_equal, rows, same_prediction, tamper,
    train_classifier, train_regressor,
)
from test_artifact_evaluation import _save_image_artifact  # noqa: E402

PERMUTATIONS = {
    "reversed": ["distance", "rooms", "age", "income"],
    "swap_first_two": ["age", "income", "rooms", "distance"],
    "rotated": ["age", "rooms", "distance", "income"],
    "shuffled": ["rooms", "income", "distance", "age"],
}
X, Y_REG, Y_CLS = make_data()


# ------------------------------------------------------------------------------------------- routes to predict

def _via_predictor(path, data, names):
    return forge.load_predictor(path).predict(data, feature_names=names)


def _via_predict_model(path, data, names):
    return forge.predict_model(path, data, feature_names=names)


def _via_task_function(task):
    fn = forge.predict_tensor_artifact if task == "regression" else forge.predict_tabular_classification_artifact
    return lambda path, data, names: fn(path, data, feature_names=names)


def routes(task):
    return [("predictor", _via_predictor), ("predict_model", _via_predict_model), ("task_function", _via_task_function(task))]


TASKS = ["regression", "tabular_classification"]
TASK_ROUTES = [pytest.param(task, fn, id=f"{task}-{name}") for task in TASKS for name, fn in routes(task)]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("named_input")
    train_regressor(root / "reg.forge")
    train_classifier(root / "cls.forge")
    train_regressor(root / "reg_plain.forge", names=None)
    train_classifier(root / "cls_plain.forge", names=None)
    return {
        "regression": str(root / "reg.forge"), "tabular_classification": str(root / "cls.forge"),
        "regression_plain": str(root / "reg_plain.forge"), "tabular_classification_plain": str(root / "cls_plain.forge"),
    }


# ============================================================================== reorder: aligned to the artifact


@pytest.mark.parametrize("task, route", TASK_ROUTES)
def test_named_input_in_the_artifacts_own_order_predicts_exactly_like_the_unnamed_array(artifacts, task, route):
    path = artifacts[task]
    assert same_prediction(route(path, X, FEATURES), forge.load_predictor(path).predict(X))


@pytest.mark.parametrize("perm", PERMUTATIONS.values(), ids=PERMUTATIONS.keys())
@pytest.mark.parametrize("task, route", TASK_ROUTES)
def test_reordered_named_input_predicts_exactly_like_the_correctly_ordered_input(artifacts, task, route, perm):
    path = artifacts[task]
    assert_order_matters(forge.load_predictor(path), X, FEATURES, perm)            # else this test proves nothing
    moved = permute(X, FEATURES, perm)
    assert same_prediction(route(path, moved, perm), forge.load_predictor(path).predict(X))


@pytest.mark.parametrize("task", TASKS)
def test_the_alignment_agrees_with_an_independent_oracle(artifacts, task):
    """Not the same code path: apply the artifact's own preprocessing + model to the correctly ordered rows by hand."""
    path = artifacts[task]
    perm = PERMUTATIONS["shuffled"]
    named = forge.load_predictor(path).predict(permute(X, FEATURES, perm), feature_names=perm)
    pre, model = forge.load_preprocessing(path), forge.load_model(path)
    raw = forge.predict(model, pre(Tensor(X.astype(np.float32)))).numpy()
    if task == "regression":
        assert np.array_equal(named.numpy(), raw)
    else:
        assert [p.index for p in named] == list(np.argmax(raw, axis=1))


@pytest.mark.parametrize("task", TASKS)
def test_without_names_a_reordered_array_is_still_accepted_unchecked(artifacts, task):
    """`.npy`/NumPy input carries no names, so nothing can be verified: unchanged pre-M119 behaviour, documented."""
    path = artifacts[task]
    predictor = forge.load_predictor(path)
    moved = permute(X, FEATURES, PERMUTATIONS["reversed"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # no warning, no error: silently width-checked
        wrong = predictor.predict(moved)
    assert not same_prediction(wrong, predictor.predict(X))
    with pytest.raises(DataError, match="expected 4 input feature\\(s\\), received 3"):
        predictor.predict(moved[:, :3])                                  # the width check is unchanged


def test_a_single_unbatched_row_is_reordered_too(artifacts):
    path = artifacts["regression"]
    perm = PERMUTATIONS["rotated"]
    row = X[3]
    named = forge.load_predictor(path).predict(row[[FEATURES.index(n) for n in perm]], feature_names=perm)
    assert same_prediction(named, forge.load_predictor(path).predict(row))


def test_an_unbatched_row_to_a_classifier_still_gets_the_batching_error_first(artifacts):
    with pytest.raises(DataError, match="requires a batched input"):
        forge.load_predictor(artifacts["tabular_classification"]).predict(X[0], feature_names=FEATURES)


@pytest.mark.parametrize("kind", ["tensor", "list", "float64_array", "tuple_names"])
def test_input_kinds_and_name_containers(artifacts, kind):
    path, perm = artifacts["regression"], PERMUTATIONS["shuffled"]
    moved = permute(X, FEATURES, perm)
    data, names = {
        "tensor": (Tensor(moved.astype(np.float32)), perm),
        "list": (moved.tolist(), perm),
        "float64_array": (moved.astype(np.float64), perm),
        "tuple_names": (moved, tuple(perm)),
    }[kind]
    assert same_prediction(forge.load_predictor(path).predict(data, feature_names=names), forge.load_predictor(path).predict(X))


def test_the_callers_array_is_never_modified(artifacts):
    moved = permute(X, FEATURES, PERMUTATIONS["reversed"])
    snapshot = moved.copy()
    forge.load_predictor(artifacts["regression"]).predict(moved, feature_names=PERMUTATIONS["reversed"])
    assert np.array_equal(moved, snapshot)


def test_names_apply_to_1d_and_2d_inputs_only(artifacts):
    with pytest.raises(DataError, match="feature_names= needs a 1-D row or a 2-D"):
        forge.load_predictor(artifacts["regression"]).predict(X.reshape(2, 80, 4), feature_names=FEATURES)


# ============================================================================== rejection: same rule on every route


def _reject(route, path, data, names, message):
    with pytest.raises(DataError, match=message):
        route(path, data, names)


@pytest.mark.parametrize("task, route", TASK_ROUTES)
class TestRejections:
    def test_an_unknown_column_is_rejected_not_positionally_assumed(self, artifacts, task, route):
        names = ["income", "age", "bedrooms", "distance"]                       # 'rooms' renamed: same width, same values
        _reject(route, artifacts[task], X, names, r"missing \(the artifact expects them.*\['rooms'\].*unexpected \(the input has them.*\['bedrooms'\]")

    def test_a_missing_column_is_rejected_by_name(self, artifacts, task, route):
        _reject(route, artifacts[task], X[:, :3], FEATURES[:3], r"missing .*\['distance'\]")

    def test_an_extra_column_is_rejected_never_silently_dropped(self, artifacts, task, route):
        extra = np.column_stack([np.arange(len(X)), X])
        _reject(route, artifacts[task], extra, ["id", *FEATURES], r"unexpected .*\['id'\]")

    def test_an_id_column_replacing_a_feature_is_rejected_although_the_width_matches(self, artifacts, task, route):
        swapped = np.column_stack([np.arange(len(X)), X[:, :3]])              # 4 columns: the width check alone would pass
        _reject(route, artifacts[task], swapped, ["id", *FEATURES[:3]], r"missing .*\['distance'\].*unexpected .*\['id'\]")

    def test_duplicate_names_are_rejected_not_deduplicated(self, artifacts, task, route):
        _reject(route, artifacts[task], X, ["income", "age", "age", "distance"], r"must be unique, but 'age'")

    @pytest.mark.parametrize("names, message", [
        (["income", "age", "rooms", ""], "empty or blank name"),
        (["income", "age", "rooms", "  "], "empty or blank name"),
        (["income", "age", "rooms", 3], "only strings"),
        ("income", "must be a list of strings"),
        (None.__class__, "must be a list of strings"),
        ([], "must not be empty"),
        (FEATURES[:3], r"feature_names has 3 name\(s\) but the input has 4 column\(s\)"),
        (FEATURES + ["extra"], r"feature_names has 5 name\(s\) but the input has 4 column\(s\)"),
    ])
    def test_a_malformed_name_list_is_rejected(self, artifacts, task, route, names, message):
        _reject(route, artifacts[task], X, names, message)

    @pytest.mark.parametrize("wrong, right", [("Income", "income"), ("AGE", "age"), (" age", "age"), ("age ", "age"),
                                              ("rooms\t", "rooms"), ("Distance", "distance")])
    def test_case_and_whitespace_differences_are_different_names_and_the_message_says_so(self, artifacts, task, route, wrong, right):
        names = [wrong if n == right else n for n in FEATURES]
        _reject(route, artifacts[task], X, names, re.escape(f"{wrong!r} vs {right!r} differ only in case/whitespace"))

    def test_the_error_lists_what_the_artifact_expects(self, artifacts, task, route):
        with pytest.raises(DataError) as excinfo:
            route(artifacts[task], X, ["a", "b", "c", "d"])
        assert "The artifact's features are ['income', 'age', 'rooms', 'distance']" in str(excinfo.value)


def test_every_route_applies_the_same_rule_and_says_the_same_thing(artifacts):
    """One rule, not four: the messages are identical except for the name of the function that was called."""
    bad = ["id", *FEATURES[:3]]
    data = np.column_stack([np.arange(len(X)), X[:, :3]])
    messages = set()
    for task in TASKS:
        for _, route in routes(task):
            with pytest.raises(DataError) as excinfo:
                route(artifacts[task], data, bad)
            messages.add(str(excinfo.value).split("() ", 1)[1])
    assert len(messages) == 1, messages


@pytest.mark.parametrize("task", TASKS)
def test_rejection_happens_before_any_model_or_preprocessing_runs(artifacts, task, monkeypatch):
    import forge.training.inference as inference

    def boom(*a, **k):
        raise AssertionError("the model ran on an input that should have been rejected")

    monkeypatch.setattr(inference, "predict", boom)
    with pytest.raises(DataError, match="unexpected"):
        forge.load_predictor(artifacts[task]).predict(np.column_stack([np.arange(len(X)), X]), feature_names=["id", *FEATURES])


# ============================================================================== an artifact without names


@pytest.mark.parametrize("task", TASKS)
def test_names_given_to_an_unnamed_artifact_cannot_be_verified_and_say_so(artifacts, task):
    path = artifacts[f"{task}_plain"]
    moved = permute(X, FEATURES, PERMUTATIONS["reversed"])
    with pytest.warns(UserWarning, match="records no feature names.*column order cannot be verified"):
        got = forge.load_predictor(path).predict(moved, feature_names=PERMUTATIONS["reversed"])
    assert same_prediction(got, forge.load_predictor(path).predict(moved))    # used exactly as given, in the order given


@pytest.mark.parametrize("task", TASKS)
def test_an_unnamed_artifact_with_no_names_given_is_silent_as_ever(artifacts, task):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forge.load_predictor(artifacts[f"{task}_plain"]).predict(X)


@pytest.mark.parametrize("task", TASKS)
def test_names_given_to_an_unnamed_artifact_must_still_be_well_formed_and_match_the_width(artifacts, task):
    predictor = forge.load_predictor(artifacts[f"{task}_plain"])
    with pytest.raises(DataError, match="must be unique"):
        predictor.predict(X, feature_names=["a", "a", "b", "c"])
    with pytest.raises(DataError, match=r"3 name\(s\) but the input has 4 column"):
        predictor.predict(X, feature_names=["a", "b", "c"])


@pytest.mark.parametrize("task, y", [("regression", Y_REG), ("tabular_classification", Y_CLS)])
def test_a_rejected_input_to_an_unnamed_artifact_is_only_the_error_never_a_warning_beside_it(artifacts, task, y):
    """A "could not verify the order" note next to an error that rejects the input anyway would be noise (M118: one line)."""
    predictor = forge.load_predictor(artifacts[f"{task}_plain"])
    wider = np.column_stack([np.arange(len(X)), X])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # any warning becomes a different exception
        with pytest.raises(DataError, match=r"expected 4 input feature\(s\), received 5"):
            predictor.predict(wider, feature_names=["id", *FEATURES])
        with pytest.raises(DataError, match=r"expected 4 input feature\(s\), received 5"):
            predictor.evaluate(wider, y, feature_names=["id", *FEATURES])
        with pytest.raises(DataError, match=r"X has 160 sample\(s\) but y has 5"):
            predictor.evaluate(X, y[:5], feature_names=FEATURES)          # a bad y is reported before the names note


def test_a_pre_m119_real_artifact_takes_names_with_a_warning_and_never_reorders():
    """The bundled Pima classifier predates names: a named CSV works exactly as in M118, with the warning."""
    path = Path(__file__).resolve().parents[1] / "models" / "tabular_classifier" / "diabetes_classifier.forge"
    Xp, yp, names = forge.data.load_csv(
        Path(__file__).resolve().parents[1] / "examples" / "tabular_diabetes" / "data" / "diabetes_holdout_eval.csv",
        target="Outcome", labels=True, return_feature_names=True,
    )
    predictor = forge.load_predictor(str(path))
    with pytest.warns(UserWarning, match="records no feature names"):
        named = predictor.evaluate(Xp, yp, feature_names=names)
    assert result_fields_equal(named, predictor.evaluate(Xp, yp))


# ============================================================================== not tabular


def test_names_for_an_image_artifact_are_refused_not_ignored(tmp_path):
    predictor = forge.load_predictor(_save_image_artifact(tmp_path))
    with pytest.raises(DataError, match="feature_names= applies only to tabular artifacts"):
        predictor.predict("photo.png", feature_names=["a"])
    with pytest.raises(DataError, match="feature_names= applies only to tabular artifacts"):
        predictor.evaluate(str(tmp_path), feature_names=["a"])
    with pytest.raises(DataError, match="feature_names= applies only to tabular artifacts"):
        forge.predict_model(_save_image_artifact(tmp_path, name="other.forge"), "photo.png", feature_names=["a"])


# ============================================================================== evaluate


@pytest.mark.parametrize("perm", PERMUTATIONS.values(), ids=PERMUTATIONS.keys())
def test_evaluating_a_reordered_named_x_scores_exactly_like_the_correctly_ordered_x(artifacts, perm):
    for task, y in (("tabular_classification", Y_CLS), ("regression", Y_REG)):
        predictor = forge.load_predictor(artifacts[task])
        assert_order_matters(predictor, X, FEATURES, perm)
        reference = predictor.evaluate(X, y)
        moved = predictor.evaluate(permute(X, FEATURES, perm), y, feature_names=perm)
        assert result_fields_equal(moved, reference), (task, moved, reference)


@pytest.mark.parametrize("task, y", [("tabular_classification", Y_CLS), ("regression", Y_REG)])
def test_evaluating_a_reordered_x_without_names_scores_the_wrong_thing(artifacts, task, y):
    """The failure M119 exists for, at the API: same shape, no error, different (worse) numbers."""
    predictor = forge.load_predictor(artifacts[task])
    reference = predictor.evaluate(X, y)
    wrong = predictor.evaluate(permute(X, FEATURES, PERMUTATIONS["reversed"]), y)
    assert not result_fields_equal(wrong, reference)


@pytest.mark.parametrize("task, y", [("tabular_classification", Y_CLS), ("regression", Y_REG)])
def test_evaluate_rejects_the_same_bad_inputs_before_scoring_anything(artifacts, task, y):
    predictor = forge.load_predictor(artifacts[task])
    with pytest.raises(DataError, match=r"unexpected .*\['id'\]"):
        predictor.evaluate(np.column_stack([np.arange(len(X)), X]), y, feature_names=["id", *FEATURES])
    with pytest.raises(DataError, match=r"missing .*\['distance'\]"):
        predictor.evaluate(X[:, :3], y, feature_names=FEATURES[:3])
    with pytest.raises(DataError, match="must be unique"):
        predictor.evaluate(X, y, feature_names=["income", "age", "age", "distance"])
    with pytest.raises(DataError, match="differ only in case/whitespace"):
        predictor.evaluate(X, y, feature_names=["Income", "age", "rooms", "distance"])


def test_evaluate_batching_does_not_change_a_named_result(artifacts):
    predictor = forge.load_predictor(artifacts["tabular_classification"])
    perm = PERMUTATIONS["shuffled"]
    moved = permute(X, FEATURES, perm)
    small = predictor.evaluate(moved, Y_CLS, feature_names=perm, batch_size=7)
    assert small.accuracy == predictor.evaluate(X, Y_CLS, batch_size=7).accuracy
    assert np.array_equal(small.confusion_matrix, predictor.evaluate(X, Y_CLS).confusion_matrix)


def test_evaluate_and_predict_align_identically(artifacts):
    """What evaluate() scores is what predict() predicts: the alignment is the same one."""
    predictor = forge.load_predictor(artifacts["regression"])
    perm = PERMUTATIONS["rotated"]
    moved = permute(X, FEATURES, perm)
    predicted = predictor.predict(moved, feature_names=perm).numpy().reshape(-1)
    result = predictor.evaluate(moved, Y_REG, feature_names=perm)
    assert result.mse == pytest.approx(float(np.mean((predicted.astype(np.float64) - Y_REG) ** 2)), rel=1e-5)


def test_evaluate_needs_a_2d_x_when_names_are_given(artifacts):
    with pytest.raises(DataError, match="needs a 2-D"):
        forge.load_predictor(artifacts["regression"]).evaluate(X.reshape(2, 80, 4), Y_REG.reshape(2, 80), feature_names=FEATURES)


def test_evaluate_warns_for_an_unnamed_artifact_and_then_scores_as_given(artifacts):
    predictor = forge.load_predictor(artifacts["regression_plain"])
    with pytest.warns(UserWarning, match="records no feature names"):
        named = predictor.evaluate(X, Y_REG, feature_names=FEATURES)
    assert result_fields_equal(named, predictor.evaluate(X, Y_REG))


# ============================================================================== M116 target transform, preprocessing


@pytest.fixture(scope="module")
def standardized(tmp_path_factory):
    path = tmp_path_factory.mktemp("named_tt") / "tt.forge"
    train_regressor(path, target_transform="standardize")
    return str(path)


def test_target_transform_still_returns_native_units_after_a_reorder(standardized):
    """Alignment is on the input side; the M116 inverse on the output side is untouched (and both compose)."""
    perm = PERMUTATIONS["shuffled"]
    predictor = forge.load_predictor(standardized)
    assert predictor.target_transform is not None
    moved = permute(X, FEATURES, perm)
    named = predictor.predict(moved, feature_names=perm)
    assert same_prediction(named, predictor.predict(X))
    z = forge.predict(forge.load_model(standardized), forge.load_preprocessing(standardized)(Tensor(X.astype(np.float32)))).numpy()
    expected = predictor.target_transform.inverse_transform(z.astype(np.float64))
    np.testing.assert_allclose(named.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_target_transform_evaluation_of_a_reordered_x_equals_the_ordered_one(standardized):
    perm = PERMUTATIONS["reversed"]
    predictor = forge.load_predictor(standardized)
    reference = predictor.evaluate(X, Y_REG)
    moved = predictor.evaluate(permute(X, FEATURES, perm), Y_REG, feature_names=perm)
    assert result_fields_equal(moved, reference) and reference.mse < reference.baseline_mse


def test_preprocessing_sees_the_columns_in_the_artifacts_order_and_a_sentinel_hits_the_right_column(tmp_path):
    """`missing_columns=[1]` is 'age' *in the artifact*. A reordered input must not move which column ReplaceValue fixes."""
    path = tmp_path / "missing.forge"
    train_regressor(path, missing_columns=[1])
    perm = PERMUTATIONS["shuffled"]
    with_sentinel = X.copy()
    with_sentinel[::4, FEATURES.index("age")] = 0.0                       # 'age' == 0 means "not measured"
    predictor = forge.load_predictor(str(path))
    reference = predictor.predict(with_sentinel)
    assert not same_prediction(reference, predictor.predict(X))            # the sentinel is really being replaced/used
    got = predictor.predict(permute(with_sentinel, FEATURES, perm), feature_names=perm)
    assert same_prediction(got, reference)


def test_preprocessing_is_unchanged_by_names(tmp_path):
    """Names never touch the fitted transform: the persisted preprocessing is byte-equal named vs unnamed."""
    train_regressor(tmp_path / "a.forge")
    train_regressor(tmp_path / "b.forge", names=None)
    a, b = forge.load_preprocessing(str(tmp_path / "a.forge")), forge.load_preprocessing(str(tmp_path / "b.forge"))
    assert repr(a) == repr(b)


# ============================================================================== corrupt artifacts fail at load, not at predict


def test_a_named_artifact_with_inconsistent_names_cannot_even_be_loaded_as_a_predictor(artifacts, tmp_path):
    bad = tamper(Path(artifacts["regression"]), lambda m: m.update(feature_names=["a", "b", "c"]))
    with pytest.raises(forge.PersistenceError, match="lists 3 name"):
        forge.load_predictor(str(bad))
    with pytest.raises(forge.PersistenceError, match="lists 3 name"):
        forge.predict_model(str(bad), X, feature_names=FEATURES)
