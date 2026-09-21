"""Milestone 117 CUDA tests: `forge model evaluate --device cuda` on the real CUDA backend.

Device behaviour only (the CLI/API parity, thinness, error and read-only contracts are covered on CPU in
`tests/test_cli_evaluate.py`): the CLI's CUDA evaluation equals the Python API's CUDA evaluation and agrees with
CPU (exactly for counts, within the project's float tolerance for loss/MSE); a target-transform artifact is still
in native units on CUDA; and a CUDA-saved artifact can be evaluated on CPU with an explicit `--device cpu`.
No CUDA-specific evaluation code exists in the CLI -- these exercise the existing predictor and kernels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.data.transforms import Normalize
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import save_model
from forge.training import load_predictor

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_artifact_evaluation import _CONFUSION, _TRUE, _X, _image_root, _save_image_artifact, _save_scoring_artifact  # noqa: E402
from test_cli_evaluate import (  # noqa: E402
    assert_matches_result, evaluate_json, regression_data, run_cli, save_regression_artifact, sha256,
    train_standardized_regressor,
)
from test_target_transform import big_target_data, independent_native_prediction  # noqa: E402

TOL = dict(rel=1e-4, abs=1e-5)


@pytest.fixture(autouse=True)
def _release_cached_cuda_memory():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def assert_close(cuda: dict, cpu: dict) -> None:
    """Counts and labels exactly; floating metrics within the project's CPU/CUDA tolerance."""
    assert cuda.keys() == cpu.keys()
    for key, expected in cpu.items():
        if isinstance(expected, float):
            assert cuda[key] == pytest.approx(expected, **TOL), key
        elif isinstance(expected, list) and expected and isinstance(expected[0], float):
            assert cuda[key] == pytest.approx(expected, **TOL), key
        else:
            assert cuda[key] == expected, key


def test_tabular_classification_on_cuda_matches_the_api_and_cpu(tmp_path, capsys):
    model = _save_scoring_artifact(tmp_path)
    np.save(tmp_path / "X.npy", _X)
    np.save(tmp_path / "y.npy", np.array(_TRUE))

    on_cuda = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cuda")
    on_cpu = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cpu")

    assert on_cuda["confusion_matrix"] == _CONFUSION.tolist()
    assert_close(on_cuda, on_cpu)
    assert_matches_result(on_cuda, load_predictor(model, device="cuda").evaluate(_X, np.array(_TRUE)))


def test_regression_on_cuda_matches_the_api_and_cpu(tmp_path, capsys):
    model = save_regression_artifact(tmp_path)
    X, y = regression_data()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)

    on_cuda = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cuda", "--batch-size", "16")
    on_cpu = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cpu")

    assert_close(on_cuda, on_cpu)
    assert_matches_result(on_cuda, load_predictor(model, device="cuda").evaluate(X, y, batch_size=16))


def test_standardized_regression_is_in_native_units_on_cuda(tmp_path, capsys):
    """M116 on the GPU: still dollars-squared, still equal to an independent NumPy calculation."""
    model = train_standardized_regressor(tmp_path)
    held_X, held_y = big_target_data(120, seed=42)
    np.save(tmp_path / "X.npy", held_X)
    np.save(tmp_path / "y.npy", held_y)

    on_cuda = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cuda")
    on_cpu = evaluate_json(capsys, model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cpu")

    oracle = independent_native_prediction(model, held_X)[:, 0]
    assert on_cuda["baseline_mse"] > 1e9 and on_cuda["mse"] > 1e3
    assert on_cuda["mse"] == pytest.approx(float(np.mean((oracle - held_y) ** 2)), rel=1e-3)
    assert_close(on_cuda, on_cpu)
    assert_matches_result(on_cuda, load_predictor(model, device="cuda").evaluate(held_X, held_y))


def test_image_classification_on_cuda_matches_the_api_and_cpu(tmp_path, capsys):
    model = _save_image_artifact(tmp_path)
    root = _image_root(tmp_path)
    on_cuda = evaluate_json(capsys, model, root, "--device", "cuda")
    assert_close(on_cuda, evaluate_json(capsys, model, root, "--device", "cpu"))
    assert_matches_result(on_cuda, load_predictor(model, device="cuda").evaluate(str(root)))


def test_a_cuda_saved_artifact_evaluates_on_cpu_when_asked_and_on_cuda_by_default(tmp_path, capsys):
    """Saved from a CUDA model: an explicit `--device cpu` works (the artifact is portable), and with no
    `--device` the predictor loads onto the device the file recorded -- the same rule as `forge model predict`."""
    forge.random.seed(0)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 1)).to("cuda")
    path = tmp_path / "cuda_saved.forge"
    save_model(
        model, str(path), task="regression",
        preprocessing=Normalize(mean=np.zeros(4, dtype=np.float32), std=np.full(4, 2.0, dtype=np.float32)),
    )
    X, y = regression_data()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)

    on_cpu = evaluate_json(capsys, path, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cpu")
    by_default = evaluate_json(capsys, path, tmp_path / "X.npy", tmp_path / "y.npy")

    assert load_predictor(str(path)).model.device.type == "cuda"     # what "default" means for this file
    assert_close(by_default, on_cpu)
    assert_matches_result(on_cpu, load_predictor(str(path), device="cpu").evaluate(X, y))


def test_cuda_evaluation_leaves_the_artifact_untouched(tmp_path, capsys):
    model = _save_scoring_artifact(tmp_path)
    np.save(tmp_path / "X.npy", _X)
    np.save(tmp_path / "y.npy", np.array(_TRUE))
    before = sha256(model)
    assert run_cli(["model", "evaluate", model, tmp_path / "X.npy", tmp_path / "y.npy", "--device", "cuda"], capsys)[0] == 0
    assert sha256(model) == before
