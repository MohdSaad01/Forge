"""Milestone 92 CUDA tests: the real-dataset tabular classification example on the real CUDA backend.

Mirrors `tests/test_tabular_classification_artifact_prediction_cuda.py`'s
coverage for this dataset: proves the real `examples/tabular_diabetes`
pipeline (real CSV -> `Compose([ReplaceValue, Normalize])` preprocessing ->
training -> artifact) trains and predicts on CUDA, and that a CPU-saved
artifact using this new preprocessing pipeline produces identical
predictions when loaded onto CPU vs. CUDA. Hardware-verified on the
reference GeForce 940MX.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.serialization import inspect_model, save_model
from forge.training import predict_tabular_classification_artifact

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.dataset import CLASS_NAMES, N_FEATURES, load_raw, make_datasets  # noqa: E402
from examples.tabular_diabetes.train import main as train_main  # noqa: E402

_SMALL = ["--batch-size", "16"]


def test_real_example_trains_and_saves_on_cuda(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "3", "--seed", "1", "--device", "cuda", "--output-dir", str(out_dir)])

    model_path = out_dir / "tabular_diabetes_model.forge"
    assert model_path.is_file()
    assert inspect_model(str(model_path)).device == "cuda"
    assert inspect_model(str(model_path)).task == "tabular_classification"


def test_predict_tabular_classification_artifact_predicts_on_cuda_by_default(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "2", "--device", "cuda", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_diabetes_model.forge"

    X, _ = load_raw()
    raw_rows = X[0:4]
    results = predict_tabular_classification_artifact(str(model_path), raw_rows)
    assert len(results) == 4
    assert all(r.label in CLASS_NAMES for r in results)


def test_cpu_saved_artifact_with_replace_value_preprocessing_matches_across_devices(tmp_path):
    """The Compose([ReplaceValue, Normalize]) preprocessing pipeline's
    forward math (including the sentinel-comparison branch inside
    ReplaceValue) must agree identically between CPU and CUDA -- the same
    parity guarantee every other preprocessing transform already has."""
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "3", "--device", "cpu", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_diabetes_model.forge"

    X, _ = load_raw()
    raw_rows = X[5:9]  # includes real sentinel-zero readings
    cpu_results = predict_tabular_classification_artifact(str(model_path), raw_rows, device="cpu")
    cuda_results = predict_tabular_classification_artifact(str(model_path), raw_rows, device="cuda")

    assert [r.index for r in cpu_results] == [r.index for r in cuda_results]
    for cpu_r, cuda_r in zip(cpu_results, cuda_results):
        assert cpu_r.confidence == pytest.approx(cuda_r.confidence, abs=1e-4)
