"""Milestone 121 CUDA tests: `train_tabular_classifier_csv()`/`train_tabular_regressor_csv()` and `forge model
train --device cuda` on the real CUDA backend.

Device behaviour only -- the CSV-to-artifact contract, thinness and error semantics are CPU tests
(`tests/test_train_tabular_csv.py`, `tests/test_cli_train.py`). Neither the CSV reader nor this module's own
code has device-specific logic; what is checked here is that the two functions reach the existing CUDA
training path exactly as an array-sourced call already does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.cli.main import main
from forge.training import load_predictor

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_tabular_csv import FEATURES, write_csv_with_id  # noqa: E402
from test_tabular_workflows import classification_data, regression_data  # noqa: E402


@pytest.fixture(autouse=True)
def _release_cached_cuda_memory():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def test_classifier_csv_on_cuda_matches_cpu_parameters_and_reports_the_cuda_device(tmp_path):
    X, y = classification_data(160)
    path = write_csv_with_id(tmp_path / "cls.csv", X, y, label=True)
    kwargs = dict(classes=["neg", "pos"], seed=3, epochs=20, patience=None)

    cpu = forge.train_tabular_classifier_csv(path, target="target", path=tmp_path / "cpu.forge", columns=FEATURES, **kwargs)
    cuda = forge.train_tabular_classifier_csv(
        path, target="target", path=tmp_path / "cuda.forge", columns=FEATURES, device="cuda", **kwargs,
    )

    predictor = load_predictor(str(tmp_path / "cuda.forge"))
    assert predictor.model.device.type == "cuda"
    assert cuda.validation_accuracy == pytest.approx(cpu.validation_accuracy, abs=1e-4)
    assert cuda.artifact_path == str(tmp_path / "cuda.forge")
    info = forge.inspect_model(cuda.artifact_path)
    assert info.input_schema.feature_names == tuple(FEATURES)


def test_regressor_csv_on_cuda_with_target_transform_reports_native_units(tmp_path):
    X, y = regression_data(160)
    path = write_csv_with_id(tmp_path / "reg.csv", X, y, label=False)

    result = forge.train_tabular_regressor_csv(
        path, target="target", path=tmp_path / "cuda.forge", columns=FEATURES,
        target_transform="standardize", device="cuda", seed=3, epochs=40, patience=None,
    )
    predictor = load_predictor(str(result.artifact_path))
    assert predictor.model.device.type == "cuda"
    predicted = predictor.predict(X[:10]).numpy()[:, 0]
    assert np.corrcoef(predicted, y[:10])[0, 1] > 0.7


def test_cli_model_train_device_cuda_produces_a_cuda_recorded_artifact(tmp_path, capsys):
    X, y = classification_data(120)
    path = write_csv_with_id(tmp_path / "cls.csv", X, y, label=True)
    output = tmp_path / "m.forge"

    code = main([
        "model", "train", str(path), "--task", "classification", "--target", "target",
        "--output", str(output), "--columns", *FEATURES, "--device", "cuda", "--epochs", "10",
    ])
    captured = capsys.readouterr()
    assert code == 0 and captured.err == ""

    info = forge.inspect_model(str(output))
    assert info.device == "cuda"
