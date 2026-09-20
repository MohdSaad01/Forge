"""Milestone 116 CUDA tests: persisted regression target transforms on the real CUDA backend.

Device-specific behaviour only; the format, semantics and negative cases are covered on
CPU in `tests/test_target_transform.py`. The target transform itself is host-side NumPy
arithmetic and must not introduce device-specific behaviour: these tests check that the
model runs on CUDA, the inverse is applied to the copied-back output, and the *same saved
artifact* gives the same native-unit answers on both devices within float tolerance
(the tolerance `tests/test_tabular_workflows_cuda.py` already uses). Independently trained
CPU and CUDA models are only required to be comparably good, never bit-identical.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.cli.main import main as cli_main

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_target_transform import (  # noqa: E402  (shared data recipe and independent reference helpers)
    EPOCHS, REPO_ROOT, SEED, big_target_data, independent_native_prediction, raw_metadata,
)
from test_tabular_workflows import split_indices  # noqa: E402


@pytest.fixture(autouse=True)
def _empty_cuda_cache():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


@pytest.fixture(scope="module")
def cuda_tt(tmp_path_factory):
    X, y = big_target_data()
    path = tmp_path_factory.mktemp("cuda_tt") / "model.forge"
    result = forge.train_tabular_regressor(
        X, y, path=path, device="cuda", seed=SEED, epochs=EPOCHS, target_transform="standardize",
    )
    return X, y, path, result


@pytest.fixture(scope="module")
def cpu_tt(tmp_path_factory):
    X, y = big_target_data()
    path = tmp_path_factory.mktemp("cpu_tt") / "model.forge"
    result = forge.train_tabular_regressor(
        X, y, path=path, device="cpu", seed=SEED, epochs=EPOCHS, target_transform="standardize",
    )
    return X, y, path, result


def test_trains_on_cuda_and_the_artifact_predicts_in_native_units(cuda_tt):
    X, y, path, result = cuda_tt
    assert forge.inspect_model(str(path)).device == "cuda" and raw_metadata(path)["forge_format_version"] == 3
    predictor = forge.load_predictor(str(path))  # lands on CUDA, the device recorded in the artifact
    assert next(iter(predictor.model.parameters())).device.type == "cuda"
    got = predictor.predict(X[:8])
    assert got.device.type == "cpu"  # the (host-side) inverse ran on the copied-back output
    np.testing.assert_allclose(got.numpy(), independent_native_prediction(path, X[:8]), rtol=1e-4, atol=1e-2)
    assert result.validation_mse < 0.2 * result.baseline_mse


def test_result_metrics_are_native_units_on_cuda(cuda_tt):
    X, y, path, result = cuda_tt
    train_idx, val_idx = split_indices(len(X), SEED)
    predictions = independent_native_prediction(path, X)[:, 0]
    err = predictions[val_idx] - y[val_idx]
    assert result.validation_mse == pytest.approx(float(np.mean(err ** 2)), rel=1e-3)
    assert result.validation_mae == pytest.approx(float(np.mean(np.abs(err))), rel=1e-3)
    assert result.baseline_mse == pytest.approx(float(np.var(y[val_idx])), rel=1e-9)
    assert result.validation_mse > 1e3  # native units, not z-scores


def test_the_same_artifact_agrees_across_cpu_and_cuda(cuda_tt):
    X, y, path, _ = cuda_tt
    on_cuda = forge.load_predictor(str(path), device="cuda")
    on_cpu = forge.load_predictor(str(path), device="cpu")
    np.testing.assert_allclose(on_cuda.predict(X[:60]).numpy(), on_cpu.predict(X[:60]).numpy(), rtol=1e-4, atol=1e-2)
    a, b = on_cuda.evaluate(X, y), on_cpu.evaluate(X, y)
    assert a.mse == pytest.approx(b.mse, rel=1e-3) and a.mae == pytest.approx(b.mae, rel=1e-3)
    assert a.baseline_mse == b.baseline_mse  # device-independent: NumPy on the native targets
    assert on_cuda.target_transform == on_cpu.target_transform


def test_a_cpu_trained_artifact_predicts_the_same_native_values_on_cuda(cpu_tt):
    X, y, path, _ = cpu_tt
    on_cuda = forge.load_predictor(str(path), device="cuda")
    on_cpu = forge.load_predictor(str(path), device="cpu")
    np.testing.assert_allclose(on_cuda.predict(X[:60]).numpy(), on_cpu.predict(X[:60]).numpy(), rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(
        forge.predict_tensor_artifact(str(path), X[:10], device="cuda").numpy(),
        independent_native_prediction(path, X[:10]), rtol=1e-4, atol=1e-2,
    )


def test_cpu_and_cuda_training_are_comparably_good(cuda_tt, cpu_tt):
    # Not bit-identical (documented, pre-existing); the transform must not make either device worse.
    _, _, _, gpu = cuda_tt
    _, _, _, cpu = cpu_tt
    assert gpu.baseline_mse == cpu.baseline_mse
    assert gpu.validation_mse < 0.2 * gpu.baseline_mse and cpu.validation_mse < 0.2 * cpu.baseline_mse
    assert gpu.target_transform == cpu.target_transform  # fitted on host, so identical


def test_convert_from_cuda_to_cpu_keeps_the_transform(cuda_tt, tmp_path):
    X, _, path, _ = cuda_tt
    converted = tmp_path / "converted_cpu.forge"
    assert cli_main(["model", "convert", str(path), "--device", "cpu", "--output", str(converted)]) == 0
    assert forge.inspect_model(str(converted)).device == "cpu"
    assert raw_metadata(converted)["target_transform"] == raw_metadata(path)["target_transform"]
    np.testing.assert_allclose(
        forge.load_predictor(str(converted)).predict(X[:20]).numpy(),
        forge.load_predictor(str(path), device="cuda").predict(X[:20]).numpy(), rtol=1e-4, atol=1e-2,
    )


_FRESH = """
import json, sys
import numpy as np
import forge
d = np.load(sys.argv[2])
p = forge.load_predictor(sys.argv[1])
print(json.dumps({"device": next(iter(p.model.parameters())).device.type,
                  "pred": p.predict(d["X"][:6]).numpy()[:, 0].tolist(), "mse": p.evaluate(d["X"], d["y"]).mse}))
"""


def test_a_fresh_process_loads_the_cuda_artifact_onto_cuda_and_predicts_native_units(cuda_tt, tmp_path):
    X, y, path, _ = cuda_tt
    held_X, held_y = big_target_data(100, seed=42)
    data = tmp_path / "eval.npz"
    np.savez(data, X=held_X, y=held_y)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run(
        [sys.executable, "-c", _FRESH, str(path), str(data)], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    fresh = json.loads(out.stdout)
    assert fresh["device"] == "cuda"
    np.testing.assert_allclose(fresh["pred"], independent_native_prediction(path, held_X[:6])[:, 0], rtol=1e-4, atol=1e-2)
    assert fresh["mse"] == pytest.approx(forge.load_predictor(str(path)).evaluate(held_X, held_y).mse, rel=1e-6)
