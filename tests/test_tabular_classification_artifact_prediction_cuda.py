"""Milestone 91 CUDA tests: `forge.predict_tabular_classification_artifact()`/
`forge.predict_model()` on the real CUDA backend.

Mirrors `tests/test_unified_artifact_prediction_cuda.py`'s CPU-vs-CUDA
coverage for the fifth artifact shape: proves a `task="tabular_classification"`
model trained/saved on CUDA predicts on CUDA by default, and that a
CPU-saved artifact explicitly loaded onto CUDA produces the same prediction
as loading it on CPU. Hardware-verified on the reference GeForce 940MX --
see `forge/training/inference.py::predict_tabular_classification_artifact()`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.data import Normalize
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import inspect_model, save_model
from forge.training import predict_model, predict_tabular_classification_artifact

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_CLASSES = ["normal", "warning", "critical"]
_N_FEATURES = 4


def _tiny_model(device="cpu"):
    return Sequential(Linear(_N_FEATURES, 6, device=device), ReLU(), Linear(6, len(_CLASSES), device=device))


def test_predict_tabular_classification_artifact_predicts_on_cuda_by_default(tmp_path):
    forge.random.seed(0)
    model = _tiny_model(device="cuda")
    path = tmp_path / "tabular_cuda.forge"
    save_model(model, str(path), classes=_CLASSES, task="tabular_classification")
    assert inspect_model(str(path)).device == "cuda"

    batch = np.random.default_rng(0).standard_normal((3, _N_FEATURES)).astype(np.float32)
    results = predict_tabular_classification_artifact(str(path), batch)
    assert len(results) == 3
    assert all(r.label in _CLASSES for r in results)


def test_predict_model_dispatches_tabular_classification_artifact_on_cuda(tmp_path):
    forge.random.seed(1)
    model = _tiny_model(device="cuda")
    path = tmp_path / "tabular_cuda.forge"
    save_model(model, str(path), classes=_CLASSES, task="tabular_classification")

    batch = np.random.default_rng(1).standard_normal((2, _N_FEATURES)).astype(np.float32)
    results = predict_model(str(path), batch)
    assert len(results) == 2


def test_cpu_saved_tabular_classification_artifact_matches_across_cpu_and_cuda_devices(tmp_path):
    """The same CPU-saved artifact, explicitly loaded onto CPU vs. CUDA, must
    produce the same prediction -- the forward math must agree between
    backends, exactly as every other CPU/CUDA parity test in this repo
    already establishes for `Linear`/`ReLU` individually."""
    forge.random.seed(2)
    model = _tiny_model(device="cpu")
    path = tmp_path / "tabular_cpu.forge"
    transform = Normalize(
        mean=np.array([0.5, -0.5, 1.0, 0.0], dtype=np.float32),
        std=np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float32),
    )
    save_model(model, str(path), preprocessing=transform, classes=_CLASSES, task="tabular_classification")

    raw = np.random.default_rng(9).standard_normal((4, _N_FEATURES)).astype(np.float32)
    cpu_results = predict_tabular_classification_artifact(str(path), raw, device="cpu")
    cuda_results = predict_tabular_classification_artifact(str(path), raw, device="cuda")

    assert [r.index for r in cpu_results] == [r.index for r in cuda_results]
    for cpu_r, cuda_r in zip(cpu_results, cuda_results):
        assert cpu_r.confidence == pytest.approx(cuda_r.confidence, abs=1e-4)
