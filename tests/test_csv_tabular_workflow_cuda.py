"""Milestone 118 CUDA tests: the CSV workflow on the real CUDA backend.

Device behaviour only -- the CSV contract, parity, thinness and error semantics are CPU tests
(`tests/test_csv_reader.py`, `tests/test_csv_tabular_workflow.py`, `tests/test_cli_evaluate_csv.py`). The reader has no device
code (it returns NumPy arrays); what is checked here is that those arrays go through the existing CUDA training and
evaluation paths exactly as NumPy arrays do:

- `forge model evaluate DATA.csv --target ... --device cuda` equals the same run with `.npy` files on CUDA, the
  Python API on CUDA, and CPU (counts exactly, floating metrics within the project's CPU/CUDA tolerance);
- an M116 standardized-target artifact is still reported in native units on the GPU from a CSV;
- `train_tabular_classifier(*load_csv(...), device="cuda")` trains and saves a verified artifact whose result
  numbers are the saved artifact's own `evaluate()` numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.data import load_csv
from forge.training import load_predictor

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_artifact_evaluation import _CONFUSION, _TRUE, _X, _save_scoring_artifact  # noqa: E402
from test_cli_evaluate import assert_matches_result, evaluate_json, sha256, train_standardized_regressor  # noqa: E402
from test_cli_evaluate_csv import housing_like_csv, scoring_csv  # noqa: E402
from test_csv_tabular_workflow import feature_columns, write_csv  # noqa: E402
from test_tabular_workflows import classification_data  # noqa: E402
from test_target_transform import big_target_data, independent_native_prediction  # noqa: E402

TOL = dict(rel=1e-4, abs=1e-5)


@pytest.fixture(autouse=True)
def _release_cached_cuda_memory():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def assert_close(cuda: dict, cpu: dict) -> None:
    assert cuda.keys() == cpu.keys()
    for key, expected in cpu.items():
        if isinstance(expected, float) or (isinstance(expected, list) and expected and isinstance(expected[0], float)):
            assert cuda[key] == pytest.approx(expected, **TOL), key
        else:
            assert cuda[key] == expected, key


def test_csv_classification_on_cuda_matches_npy_the_api_and_cpu(tmp_path, capsys):
    model = _save_scoring_artifact(tmp_path)
    csv_path = scoring_csv(tmp_path)
    np.save(tmp_path / "X.npy", _X)
    np.save(tmp_path / "y.npy", np.array(_TRUE))

    on_cuda = evaluate_json(capsys, model, csv_path, "--target", "label", "--device", "cuda")
    npy_on_cuda = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cuda")
    on_cpu = evaluate_json(capsys, model, csv_path, "--target", "label", "--device", "cpu")

    assert on_cuda["confusion_matrix"] == _CONFUSION.tolist()
    assert on_cuda == npy_on_cuda                                         # same arrays, same kernels: identical
    assert_close(on_cuda, on_cpu)
    assert_matches_result(on_cuda, load_predictor(model, device="cuda").evaluate(_X, np.array(_TRUE)))


def test_a_standardized_csv_regression_is_in_native_units_on_cuda(tmp_path, capsys):
    model = train_standardized_regressor(tmp_path)
    held_X, held_y = big_target_data(120, seed=42)
    csv_path = housing_like_csv(tmp_path, held_X, held_y)

    on_cuda = evaluate_json(capsys, model, csv_path, "--target", "median_house_value", "--device", "cuda")
    on_cpu = evaluate_json(capsys, model, csv_path, "--target", "median_house_value", "--device", "cpu")

    oracle = independent_native_prediction(model, held_X)[:, 0]
    assert on_cuda["baseline_mse"] > 1e9 and on_cuda["mse"] > 1e3         # z-space numbers would be < ~10
    assert on_cuda["mse"] == pytest.approx(float(np.mean((oracle - held_y) ** 2)), rel=1e-3)
    assert_close(on_cuda, on_cpu)
    assert_matches_result(on_cuda, load_predictor(model, device="cuda").evaluate(held_X, held_y))


def test_training_from_a_csv_on_cuda_produces_a_verified_artifact(tmp_path):
    X, y = classification_data(120)
    csv_path = write_csv(tmp_path / "d.csv", {**feature_columns(X), "label": y})
    Xc, yc = load_csv(csv_path, target="label", labels=True)
    path = tmp_path / "cuda.forge"

    result = forge.train_tabular_classifier(Xc, yc, path=path, classes=["neg", "pos"], device="cuda", seed=3, epochs=20)

    predictor = load_predictor(str(path))
    assert predictor.model.device.type == "cuda" and predictor.classes == ["neg", "pos"]
    assert result.artifact_path == str(path) and np.isfinite(result.validation_loss)
    assert result.validation_accuracy > result.baseline_accuracy - 0.05   # 20 epochs: a sanity bound, not a benchmark
    train_rows = np.random.default_rng(3).permutation(120)[: 120 - 24]
    assert result.train_accuracy == predictor.evaluate(X[train_rows], y[train_rows]).accuracy   # the saved artifact's own number


def test_cuda_csv_evaluation_leaves_artifact_and_csv_untouched(tmp_path, capsys):
    model = _save_scoring_artifact(tmp_path)
    csv_path = scoring_csv(tmp_path)
    before = (sha256(model), sha256(csv_path))
    assert evaluate_json(capsys, model, csv_path, "--target", "label", "--device", "cuda")["samples"] == 8
    assert (sha256(model), sha256(csv_path)) == before
