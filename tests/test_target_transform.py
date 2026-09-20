"""Milestone 116 tests: persisted regression target transforms on CPU.

`StandardizeTarget`, `save_model(..., target_transform=)` / `load_target_transform()`,
`forge.train_tabular_regressor(..., target_transform="standardize")`, and the
artifact-level inverse that `load_predictor().predict()/evaluate()`,
`predict_tensor_artifact()`, `predict_model()` and `forge model predict` share.

Nothing here trusts one Forge function to check another: the expected native prediction
is recomputed by hand from the artifact's raw `metadata.json` numbers plus a manual
`load_model()` forward pass; the split is reproduced from its documented rule; metrics
come from NumPy. The data is deliberately awkward -- a target around 2x10^5 whose
z-scores are O(1) -- so a model whose output is not inverted, or metrics that are not
in native units, are off by five orders of magnitude and cannot pass by accident.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.cli.main import main as cli_main
from forge.data import StandardizeTarget
from forge.exceptions import DataError, PersistenceError
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import inspect_model, load_model, load_preprocessing, load_target_transform, save_model
from forge.serialization.archive import METADATA_ENTRY
from forge.tensor.tensor import Tensor
from forge.training import Trainer, predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tabular_workflows import OFFSETS, SCALES, split_indices  # noqa: E402  (shared data recipe)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_REGRESSOR = REPO_ROOT / "models" / "tabular_regressor" / "concrete_strength_regressor.forge"
EPOCHS = 150
SEED = 3


def big_target_data(n: int = 300, seed: int = 0, outputs: int = 1) -> "tuple[np.ndarray, np.ndarray]":
    """Features at scales 1000 / 0.01 / 5; target ~2e5 +- 1e5 (first column), optionally a second ~0.5 +- 0.02 one."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    y = 2.0e5 + 1.0e5 * (z[:, 0] - 0.5 * z[:, 1]) + 3.0e3 * rng.normal(size=n)
    if outputs == 2:
        y = np.stack([y, 0.5 + 0.02 * z[:, 2] + 0.001 * rng.normal(size=n)], axis=1)
    return z * SCALES + OFFSETS, y


@pytest.fixture
def no_training(monkeypatch):
    def tripwire(self, *args, **kwargs):
        raise AssertionError("training started: a preflight check ran too late")

    monkeypatch.setattr(Trainer, "fit", tripwire)


@pytest.fixture(scope="module")
def tt_run(tmp_path_factory):
    X, y = big_target_data()
    path = tmp_path_factory.mktemp("tt") / "model.forge"
    result = forge.train_tabular_regressor(
        X, y, path=path, seed=SEED, epochs=EPOCHS, target_transform="standardize",
    )
    return X, y, path, result


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory):
    X, y = big_target_data()
    path = tmp_path_factory.mktemp("plain") / "model.forge"
    return X, y, path, forge.train_tabular_regressor(X, y, path=path, seed=SEED, epochs=EPOCHS)


# ------------------------------------------------------------------ independent reference helpers


def raw_metadata(path) -> dict:
    with zipfile.ZipFile(str(path)) as zf:
        return json.loads(zf.read(METADATA_ENTRY))


def independent_native_prediction(path, rows: np.ndarray) -> np.ndarray:
    """A native-unit prediction computed by hand: persisted preprocessing, raw model forward,
    then `z * std + mean` with `mean`/`std` read straight out of `metadata.json`.
    No `StandardizeTarget`, no `ArtifactPredictor`."""
    node = raw_metadata(path).get("target_transform")
    model, pre = load_model(str(path), device="cpu"), load_preprocessing(str(path))
    z = predict(model, pre(Tensor(np.asarray(rows, dtype=np.float32)))).numpy().astype(np.float64)
    if node is None:
        return z
    return z * np.asarray(node["std"]) + np.asarray(node["mean"])


def tamper(path: Path, mutate) -> Path:
    with zipfile.ZipFile(str(path)) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries[METADATA_ENTRY] = json.dumps(mutate(json.loads(entries[METADATA_ENTRY]))).encode("utf-8")
    out = path.parent / f"tampered_{len(list(path.parent.iterdir()))}_{path.name}"
    with zipfile.ZipFile(str(out), "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return out


# ============================================================================ StandardizeTarget


def test_fit_uses_population_statistics_per_column():
    y = np.random.default_rng(0).normal(5.0, 3.0, size=(50, 2)) * [1.0, 1000.0]
    t = StandardizeTarget.fit(y)
    np.testing.assert_allclose(t.mean, y.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(t.std, y.std(axis=0, ddof=0), rtol=1e-14)
    assert t.outputs == 2
    one = StandardizeTarget.fit(y[:, 0])  # (n,) is a single column
    assert one.outputs == 1 and one.mean.shape == (1,)


def test_transform_and_inverse_have_known_values_and_round_trip():
    t = StandardizeTarget.fit([10.0, 20.0, 30.0])
    std = np.sqrt(200.0 / 3.0)
    np.testing.assert_allclose(t.transform([10.0, 20.0, 30.0]), [-10 / std, 0.0, 10 / std], rtol=1e-14)
    np.testing.assert_allclose(t.inverse_transform([-10 / std, 0.0, 10 / std]), [10.0, 20.0, 30.0], rtol=1e-14)
    y = np.random.default_rng(1).normal(1e6, 1e4, size=(40, 3))
    multi = StandardizeTarget.fit(y)
    np.testing.assert_allclose(multi.inverse_transform(multi.transform(y)), y, rtol=1e-12)
    assert abs(multi.transform(y).mean(axis=0)).max() < 1e-12 and np.allclose(multi.transform(y).std(axis=0), 1.0)


def test_inputs_may_be_unbatched_integer_or_tensor():
    t = StandardizeTarget([100.0, 5.0], [10.0, 2.0])
    np.testing.assert_allclose(t.inverse_transform([1.0, 1.0]), [110.0, 7.0])  # an unbatched (outputs,) row
    np.testing.assert_allclose(t.transform(np.array([[110, 7]])), [[1.0, 1.0]])  # ints
    np.testing.assert_allclose(t.inverse_transform(Tensor(np.array([[0.0, 1.0]], dtype=np.float32))), [[100.0, 7.0]])
    scalar_col = StandardizeTarget([100.0], [10.0])
    np.testing.assert_allclose(scalar_col.inverse_transform([0.0, 1.0, 2.0]), [100.0, 110.0, 120.0])


@pytest.mark.parametrize("bad", [np.zeros((4, 3)), [[1.0]], np.zeros((2, 2, 3)), "abc", 5.0])
def test_transform_rejects_wrong_shapes_and_types(bad):
    t = StandardizeTarget([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(DataError):
        t.transform(bad)
    with pytest.raises(DataError):
        t.inverse_transform(bad)


@pytest.mark.parametrize(
    "bad", [[], np.empty((0, 2)), [1.0, float("nan")], [[1.0], [float("inf")]], np.zeros((2, 2, 2)), ["a", "b"], 3.0],
)
def test_fit_rejects_bad_targets(bad):
    with pytest.raises(DataError):
        StandardizeTarget.fit(bad)


def test_fit_rejects_a_constant_column_instead_of_dividing_by_zero():
    with pytest.raises(DataError, match="constant"):
        StandardizeTarget.fit([5.0, 5.0, 5.0])
    with pytest.raises(DataError, match=r"column\(s\) \[1\]"):
        StandardizeTarget.fit(np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]]))
    # A tiny but real spread is a valid scale; only exactly-constant columns are refused.
    t = StandardizeTarget.fit([1.0, 1.0 + 1e-9])
    assert t.std[0] > 0 and np.isfinite(t.transform([1.0, 1.0 + 1e-9])).all()


@pytest.mark.parametrize(
    "mean,std",
    [([0.0], [0.0]), ([0.0], [-1.0]), ([float("nan")], [1.0]), ([0.0], [float("inf")]), ([0.0, 1.0], [1.0]), ([], [])],
)
def test_constructor_rejects_invalid_parameters(mean, std):
    with pytest.raises(DataError):
        StandardizeTarget(mean, std)


def test_config_round_trip_and_equality():
    t = StandardizeTarget([1.5, -2.0], [0.25, 8.0])
    node = t.to_config()
    assert node == {"type": "standardize", "mean": [1.5, -2.0], "std": [0.25, 8.0]}
    assert StandardizeTarget.from_config(json.loads(json.dumps(node))) == t
    assert t != StandardizeTarget([1.5, -2.0], [0.25, 9.0]) and t != "standardize"
    assert hash(t) == hash(StandardizeTarget([1.5, -2.0], [0.25, 8.0]))


# ============================================================================ artifact format


def _model():
    return Sequential(Linear(3, 4), ReLU(), Linear(4, 1))


def test_an_artifact_without_a_target_transform_is_written_exactly_as_before(tmp_path):
    path = tmp_path / "plain.forge"
    save_model(_model(), str(path), task="regression")
    meta = raw_metadata(path)
    assert meta["forge_format_version"] == 2 and "target_transform" not in meta
    assert load_target_transform(str(path)) is None and inspect_model(str(path)).target_transform is None


def test_an_artifact_with_a_target_transform_is_version_3_and_round_trips(tmp_path):
    path = tmp_path / "tt.forge"
    tt = StandardizeTarget([206856.5], [115393.25])
    save_model(_model(), str(path), task="regression", target_transform=tt)
    meta = raw_metadata(path)
    assert meta["forge_format_version"] == 3
    assert meta["target_transform"] == {"type": "standardize", "mean": [206856.5], "std": [115393.25]}
    assert load_target_transform(str(path)) == tt
    info = inspect_model(str(path))
    assert info.target_transform == tt and info.format_version == 3
    assert "Target transform: standardize" in str(info)
    assert isinstance(load_model(str(path)), Sequential)  # load_model() reads version 3 too


@pytest.mark.parametrize("bad", [{"mean": [0.0], "std": [1.0]}, "standardize", lambda z: z, np.array([1.0])])
def test_save_model_accepts_only_a_standardize_target(tmp_path, bad):
    with pytest.raises(PersistenceError, match="StandardizeTarget"):
        save_model(_model(), str(tmp_path / "x.forge"), task="regression", target_transform=bad)
    assert not (tmp_path / "x.forge").exists()


@pytest.mark.parametrize("task", [None, "tabular_classification", "classification", "segmentation"])
def test_save_model_rejects_a_target_transform_on_a_non_regression_task(tmp_path, task):
    with pytest.raises(PersistenceError, match="task='regression'"):
        save_model(_model(), str(tmp_path / "x.forge"), task=task, target_transform=StandardizeTarget([0.0], [1.0]))
    assert not (tmp_path / "x.forge").exists()


def _tt_artifact(tmp_path) -> Path:
    path = tmp_path / "tt.forge"
    save_model(_model(), str(path), task="regression", target_transform=StandardizeTarget([1e5], [1e4]))
    return path


def _set_node(node):
    def mutate(meta):
        meta["target_transform"] = node
        return meta
    return mutate


def _drop_key(meta):
    del meta["target_transform"]
    return meta


def _downgrade(meta):
    meta["forge_format_version"] = 2
    return meta


def _classification_task(meta):
    meta["task"] = "tabular_classification"
    return meta


@pytest.mark.parametrize(
    "mutate,match",
    [
        (_drop_key, "missing"),                                                             # stripped from a v3 file
        (_downgrade, "requires version 3"),                                                 # smuggled into a v2 file
        (_set_node({"type": "minmax", "mean": [0.0], "std": [1.0]}), "malformed"),          # unsupported name
        (_set_node({"type": "standardize", "mean": [0.0], "std": [0.0]}), "invalid"),       # would divide by zero
        (_set_node({"type": "standardize", "mean": [0.0], "std": [-1.0]}), "invalid"),
        (_set_node({"type": "standardize", "mean": [0.0, 1.0], "std": [1.0]}), "invalid"),  # width disagreement
        (_set_node({"type": "standardize", "mean": [float("nan")], "std": [1.0]}), "invalid"),
        (_set_node({"type": "standardize", "mean": [0.0]}), "malformed"),                   # no std
        (_set_node(None), "malformed"),
        (_set_node("standardize"), "malformed"),
        (_classification_task, "task='regression'|task=None|declares task"),
    ],
)
def test_corrupt_or_missing_transform_metadata_is_an_error_everywhere(tmp_path, mutate, match):
    bad = tamper(_tt_artifact(tmp_path), mutate)
    for call in (
        lambda: inspect_model(str(bad)),
        lambda: load_target_transform(str(bad)),
        lambda: forge.load_predictor(str(bad)),
        lambda: forge.predict_tensor_artifact(str(bad), [[1.0, 2.0, 3.0]]),
    ):
        with pytest.raises(PersistenceError, match=match):
            call()


def test_an_unknown_format_version_is_still_rejected(tmp_path):
    bad = tamper(_tt_artifact(tmp_path), lambda m: {**m, "forge_format_version": 4})
    for call in (lambda: inspect_model(str(bad)), lambda: load_model(str(bad)), lambda: load_target_transform(str(bad))):
        with pytest.raises(PersistenceError, match="unsupported format version 4"):
            call()


def test_a_model_and_transform_that_disagree_about_the_output_width_are_reported(tmp_path):
    path = tmp_path / "mismatch.forge"  # a 1-output model saved with a 2-column transform
    save_model(_model(), str(path), task="regression", target_transform=StandardizeTarget([0.0, 0.0], [1.0, 1.0]))
    with pytest.raises(PersistenceError, match="inconsistent"):
        forge.load_predictor(str(path)).predict([[1.0, 2.0, 3.0]])


# ============================================================================ prediction in native units


def test_predict_returns_native_units_matching_an_independent_calculation(tt_run):
    X, y, path, result = tt_run
    rows = X[:8]
    expected = independent_native_prediction(path, rows)
    raw = predict(load_model(str(path)), load_preprocessing(str(path))(Tensor(rows.astype(np.float32)))).numpy()
    assert np.abs(raw).max() < 20 and expected.mean() > 5e4  # the model itself speaks z-scores; the artifact must not
    got = forge.load_predictor(str(path)).predict(rows)
    assert got.shape == (8, 1) and got.dtype == forge.DEFAULT_DTYPE
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)
    assert np.corrcoef(got.numpy()[:, 0], y[:8])[0, 1] > 0.9  # and they really are the targets' units


def test_every_prediction_entry_point_returns_the_same_native_numbers(tt_run, tmp_path, capsys):
    X, _, path, _ = tt_run
    rows = X[:4]
    expected = independent_native_prediction(path, rows)
    predictor = forge.load_predictor(str(path))
    for got in (
        predictor.predict(rows),
        forge.predict_tensor_artifact(str(path), rows),
        forge.predict_model(str(path), rows),
    ):
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)
    payload = tmp_path / "rows.json"
    payload.write_text(json.dumps(rows.tolist()))
    assert cli_main(["model", "predict", str(path), str(payload), "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["task"] == "regression"
    np.testing.assert_allclose(np.array(printed["prediction"]), expected, rtol=1e-6)


def test_a_single_unbatched_row_is_predicted_in_native_units_too(tt_run):
    X, _, path, _ = tt_run
    predictor = forge.load_predictor(str(path))
    single = predictor.predict(X[0])
    assert single.shape == (1,)
    np.testing.assert_allclose(single.numpy(), independent_native_prediction(path, X[:1])[0], rtol=1e-6)


def test_the_predictor_and_inspection_expose_the_transform_read_only(tt_run):
    _, _, path, result = tt_run
    predictor = forge.load_predictor(str(path))
    node = raw_metadata(path)["target_transform"]
    assert predictor.target_transform == StandardizeTarget(node["mean"], node["std"]) == result.target_transform
    assert forge.inspect_model(str(path)).target_transform == predictor.target_transform


def test_a_wrong_feature_count_is_a_data_error_for_predict_and_evaluate(tt_run):
    X, y, path, _ = tt_run
    predictor = forge.load_predictor(str(path))
    with pytest.raises(DataError, match="expected 3 input feature"):
        predictor.predict(X[:2, :2])
    with pytest.raises(DataError, match="expected 3 input feature"):
        predictor.evaluate(X[:5, :2], y[:5])
    with pytest.raises(DataError, match="expected 3 input feature"):
        forge.predict_tensor_artifact(str(path), X[:2, :2])


# ============================================================================ evaluation in native units


def test_evaluate_reports_native_unit_metrics(tt_run):
    X, y, path, _ = tt_run
    held_X, held_y = big_target_data(120, seed=42)
    predictions = independent_native_prediction(path, held_X)[:, 0]
    mse = float(np.mean((predictions - held_y) ** 2))
    mae = float(np.mean(np.abs(predictions - held_y)))
    baseline = float(np.mean((held_y - held_y.mean()) ** 2))
    got = forge.load_predictor(str(path)).evaluate(held_X, held_y)
    assert baseline > 1e9 and mse > 1e3  # z-space numbers would be < ~10
    assert got.mse == pytest.approx(mse, rel=1e-5)
    assert got.mae == pytest.approx(mae, rel=1e-5)
    assert got.loss == pytest.approx(mse, rel=1e-4)
    assert got.baseline_mse == pytest.approx(baseline, rel=1e-12)
    assert got.mse < 0.2 * got.baseline_mse  # and the model really learned the native target


def test_evaluate_accepts_1d_and_column_targets_alike(tt_run):
    X, y, path, _ = tt_run
    predictor = forge.load_predictor(str(path))
    a, b = predictor.evaluate(X[:50], y[:50]), predictor.evaluate(X[:50], y[:50, None])
    assert (a.mse, a.mae, a.baseline_mse) == (b.mse, b.mae, b.baseline_mse)


# ============================================================================ training workflow


def test_default_training_is_unchanged_and_writes_no_transform(plain_run):
    _, _, path, result = plain_run
    meta = raw_metadata(path)
    assert meta["forge_format_version"] == 2 and "target_transform" not in meta
    assert result.target_transform is None and forge.load_predictor(str(path)).target_transform is None


def test_the_transform_is_fitted_on_the_training_split_only(tt_run):
    X, y, path, result = tt_run
    train_idx, val_idx = split_indices(len(X), SEED)
    node = raw_metadata(path)["target_transform"]
    np.testing.assert_allclose(node["mean"], [y[train_idx].mean()], rtol=1e-12)
    np.testing.assert_allclose(node["std"], [y[train_idx].std()], rtol=1e-12)
    # The test has teeth: the statistics that leaking would have produced are measurably different.
    assert abs(y.mean() - y[train_idx].mean()) > 1e-4 * y.std()
    assert abs(y[val_idx].mean() - y[train_idx].mean()) > 1e-3 * y.std()
    assert abs(y.std() - y[train_idx].std()) > 1e-5 * y.std()


def test_result_metrics_are_native_units_computed_from_the_saved_artifact(tt_run):
    X, y, path, result = tt_run
    train_idx, val_idx = split_indices(len(X), SEED)
    predictions = independent_native_prediction(path, X)[:, 0]
    for split, idx in (("train", train_idx), ("validation", val_idx)):
        err = predictions[idx] - y[idx]
        assert getattr(result, f"{split}_mse") == pytest.approx(float(np.mean(err ** 2)), rel=1e-5)
        assert getattr(result, f"{split}_mae") == pytest.approx(float(np.mean(np.abs(err))), rel=1e-5)
    assert result.baseline_mse == pytest.approx(float(np.var(y[val_idx])), rel=1e-9)
    assert result.validation_mse > 1e3 and result.validation_loss == pytest.approx(result.validation_mse, rel=1e-4)
    assert result.validation_mse < 0.2 * result.baseline_mse


def test_the_bare_training_history_stays_in_the_training_space(tt_run):
    # Documented: `history` / `best_monitored_value` are z-scores, only the result's metrics are native.
    _, _, _, result = tt_run
    assert result.history[-1].val_loss < 50 < result.validation_mse
    assert result.best_monitored_value < 50


def test_standardising_rescues_a_large_target_the_default_cannot_learn_in_the_same_budget(tt_run, plain_run):
    _, _, _, with_transform = tt_run
    _, _, _, without = plain_run
    assert with_transform.validation_mse < 0.2 * with_transform.baseline_mse
    assert without.validation_mse > 0.9 * without.baseline_mse  # 150 epochs of Adam never reaches a 2e5 output
    assert with_transform.baseline_mse == without.baseline_mse  # same split, same native baseline


def test_multi_output_targets_get_a_per_column_transform(tmp_path):
    X, y = big_target_data(outputs=2)
    path = tmp_path / "multi.forge"
    result = forge.train_tabular_regressor(X, y, path=path, seed=SEED, epochs=100, target_transform="standardize")
    train_idx, _ = split_indices(len(X), SEED)
    node = raw_metadata(path)["target_transform"]
    assert result.outputs == 2 and len(node["mean"]) == 2
    np.testing.assert_allclose(node["mean"], y[train_idx].mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(node["std"], y[train_idx].std(axis=0), rtol=1e-12)
    got = forge.load_predictor(str(path)).predict(X[:6]).numpy()
    assert got.shape == (6, 2)
    np.testing.assert_allclose(got, independent_native_prediction(path, X[:6]), rtol=1e-5)
    assert got[:, 0].mean() > 5e4 and 0.0 < got[:, 1].mean() < 1.0  # each column in its own native units


def test_a_custom_model_with_a_non_linear_last_layer_still_round_trips(tmp_path):
    # An artifact-level transform never touches the model, so any architecture works (folding the
    # affine map into a final Linear could not have handled this one).
    X, y = big_target_data()
    net = Sequential(Linear(3, 8), ReLU(), Linear(8, 1), ReLU())
    path = tmp_path / "custom.forge"
    forge.train_tabular_regressor(X, y, path=path, model=net, seed=SEED, epochs=5, target_transform="standardize")
    np.testing.assert_allclose(
        forge.load_predictor(str(path)).predict(X[:5]).numpy(), independent_native_prediction(path, X[:5]), rtol=1e-6,
    )


@pytest.mark.parametrize("bad", ["minmax", "Standardize", "", 1, True, ["standardize"], StandardizeTarget([0.0], [1.0])])
def test_an_unsupported_target_transform_is_rejected_before_training(no_training, tmp_path, bad):
    X, y = big_target_data(60)
    with pytest.raises(DataError, match="target_transform must be None or one of"):
        forge.train_tabular_regressor(X, y, path=tmp_path / "m.forge", target_transform=bad)
    assert not (tmp_path / "m.forge").exists()


@pytest.mark.parametrize(
    "make_y",
    [
        lambda y: np.full_like(y, 42.0),                                   # constant target
        lambda y: np.stack([y, np.full_like(y, 7.0)], axis=1),             # one constant column of two
    ],
    ids=["constant", "one-constant-column"],
)
def test_a_constant_target_is_refused_before_training(no_training, tmp_path, make_y):
    X, y = big_target_data(60)
    with pytest.raises(DataError, match="constant"):
        forge.train_tabular_regressor(X, make_y(y), path=tmp_path / "m.forge", target_transform="standardize")
    assert not (tmp_path / "m.forge").exists()


def test_a_target_constant_only_in_the_training_split_is_refused(no_training, tmp_path):
    # The statistics come from the training rows: here they alone are constant, the full y is not.
    X, y = big_target_data(60)
    train_idx, val_idx = split_indices(len(X), 0)
    y = y.copy()
    y[train_idx] = 5.0
    assert np.ptp(y) > 0
    with pytest.raises(DataError, match="constant"):
        forge.train_tabular_regressor(X, y, path=tmp_path / "m.forge", target_transform="standardize", seed=0)


@pytest.mark.parametrize(
    "make_y,match",
    [
        (lambda y: np.where(np.arange(len(y)) == 3, np.nan, y), "non-finite"),
        (lambda y: np.where(np.arange(len(y)) == 3, np.inf, y), "non-finite"),
        (lambda y: y[:0], "no targets|non-empty|has 0"),
        (lambda y: y.reshape(len(y), 1, 1), "shape"),
        (lambda y: y[:-1], "sample"),
        (lambda y: np.array(["a"] * len(y)), "numeric"),
    ],
    ids=["nan", "inf", "empty", "3d", "length-mismatch", "strings"],
)
def test_bad_targets_are_rejected_before_training_with_a_transform_too(no_training, tmp_path, make_y, match):
    X, y = big_target_data(60)
    with pytest.raises(DataError, match=match):
        forge.train_tabular_regressor(X, make_y(y), path=tmp_path / "m.forge", target_transform="standardize")


def test_save_and_verify_confirms_the_transform_survived_the_round_trip(tmp_path, monkeypatch):
    model, tt = _model(), StandardizeTarget([1e5], [1e4])
    sample = Tensor(np.zeros((1, 3), dtype=np.float32))
    forge.save_and_verify(model, str(tmp_path / "ok.forge"), sample, task="regression", target_transform=tt)
    assert load_target_transform(str(tmp_path / "ok.forge")) == tt
    monkeypatch.setattr(
        "forge.serialization.model.load_target_transform", lambda path: StandardizeTarget([1e5], [2e4]),
    )
    with pytest.raises(PersistenceError, match="target transform read back"):
        forge.save_and_verify(model, str(tmp_path / "bad.forge"), sample, task="regression", target_transform=tt)


# ============================================================================ old artifacts / CLI


def test_a_bundled_pre_m116_regression_artifact_is_read_as_the_identity():
    meta = raw_metadata(BUNDLED_REGRESSOR)
    assert meta["forge_format_version"] == 2 and "target_transform" not in meta
    info = inspect_model(str(BUNDLED_REGRESSOR))
    assert info.target_transform is None and "Target transform" not in str(info)
    predictor = forge.load_predictor(str(BUNDLED_REGRESSOR))
    assert predictor.target_transform is None
    rows = np.array([[540.0, 0.0, 0.0, 162.0, 2.5, 1040.0, 676.0, 28.0], [332.5, 142.5, 0.0, 228.0, 0.0, 932.0, 594.0, 270.0]])
    by_hand = independent_native_prediction(BUNDLED_REGRESSOR, rows)  # raw model output, nothing inverted
    np.testing.assert_array_equal(predictor.predict(rows).numpy(), by_hand.astype(np.float32))
    np.testing.assert_array_equal(forge.predict_tensor_artifact(str(BUNDLED_REGRESSOR), rows).numpy(), by_hand.astype(np.float32))


def test_convert_carries_the_target_transform_and_leaves_old_artifacts_at_version_2(tt_run, tmp_path):
    X, _, path, _ = tt_run
    converted = tmp_path / "converted.forge"
    assert cli_main(["model", "convert", str(path), "--device", "cpu", "--output", str(converted)]) == 0
    assert raw_metadata(converted)["target_transform"] == raw_metadata(path)["target_transform"]
    assert raw_metadata(converted)["forge_format_version"] == 3
    np.testing.assert_array_equal(
        forge.load_predictor(str(converted)).predict(X[:5]).numpy(), forge.load_predictor(str(path)).predict(X[:5]).numpy(),
    )
    old = tmp_path / "old.forge"
    assert cli_main(["model", "convert", str(BUNDLED_REGRESSOR), "--device", "cpu", "--output", str(old)]) == 0
    assert raw_metadata(old)["forge_format_version"] == 2 and "target_transform" not in raw_metadata(old)


def test_cli_inspect_reports_the_target_transform(tt_run, capsys):
    _, _, path, _ = tt_run
    assert cli_main(["model", "inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format_version"] == 3 and payload["target_transform"] == raw_metadata(path)["target_transform"]
    assert cli_main(["model", "inspect", str(path)]) == 0
    assert "Target transform: StandardizeTarget(" in capsys.readouterr().out
    assert cli_main(["model", "inspect", str(BUNDLED_REGRESSOR), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["target_transform"] is None


# ============================================================================ fresh process


_FRESH = """
import json, sys, zipfile
import numpy as np
import forge
d = np.load(sys.argv[2])
p = forge.load_predictor(sys.argv[1])
pred = p.predict(d["X"][:5]).numpy()[:, 0]
ev = p.evaluate(d["X"], d["y"])
# independent of StandardizeTarget/ArtifactPredictor: raw model + preprocessing + the numbers in metadata.json
node = json.loads(zipfile.ZipFile(sys.argv[1]).read("metadata.json"))["target_transform"]
model = forge.load_model(sys.argv[1], device="cpu")
pre = forge.load_preprocessing(sys.argv[1])
z = forge.predict(model, pre(forge.Tensor(d["X"][:5].astype(np.float32)))).numpy()[:, 0].astype(np.float64)
print(json.dumps({"pred": pred.tolist(), "by_hand": (z * node["std"][0] + node["mean"][0]).tolist(),
                  "mse": ev.mse, "mae": ev.mae, "base": ev.baseline_mse, "tt": p.target_transform.to_config()}))
"""


def test_a_fresh_process_predicts_and_evaluates_in_native_units(tt_run, tmp_path):
    _, _, path, _ = tt_run
    held_X, held_y = big_target_data(120, seed=42)
    data = tmp_path / "eval.npz"
    np.savez(data, X=held_X, y=held_y)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run(
        [sys.executable, "-c", _FRESH, str(path), str(data)], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr
    fresh = json.loads(out.stdout)
    predictor = forge.load_predictor(str(path))
    here = predictor.evaluate(held_X, held_y)
    assert fresh["pred"] == predictor.predict(held_X[:5]).numpy()[:, 0].tolist()
    np.testing.assert_allclose(fresh["pred"], fresh["by_hand"], rtol=1e-6)
    assert (fresh["mse"], fresh["mae"], fresh["base"]) == (here.mse, here.mae, here.baseline_mse)
    assert fresh["tt"] == raw_metadata(path)["target_transform"] and fresh["mse"] > 1e3
