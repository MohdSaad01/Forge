"""Milestone 102 CUDA tests: `forge.load_predictor()`/`ArtifactPredictor` on
the real CUDA backend.

Mirrors `tests/test_reusable_predictor.py`'s CPU coverage for the
device-specific behavior only: a predictor loaded with `device="cuda"` runs
every subsequent `predict()` on that already-loaded CUDA model (no per-call
device move, no reconstruction), CPU/CUDA predictions on the same weights
agree within the project's established tolerance, and an artifact saved from
a CUDA model round-trips back onto CUDA by default. See
`forge/training/inference.py::ArtifactPredictor`/`load_predictor()`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.data.transforms import Normalize
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import save_model
from forge.training import ClassificationPrediction, load_predictor, predict_tabular_classification_artifact

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)
_N_FEATURES = 4


def _transform():
    return Normalize(
        mean=np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
        std=np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32),
    )


def _tabular_model(num_classes=3):
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, num_classes))


def test_load_predictor_saved_from_cuda_model_restores_onto_cuda_by_default(tmp_path):
    forge.random.seed(0)
    model = _tabular_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(
        model, str(model_path), preprocessing=_transform(), classes=["normal", "warning", "critical"],
        task="tabular_classification",
    )

    predictor = load_predictor(str(model_path))
    assert predictor.model.device.type == "cuda"

    batch = np.random.default_rng(0).standard_normal((2, _N_FEATURES)).astype(np.float32)
    results = predictor.predict(batch)
    assert len(results) == 2
    assert all(isinstance(r, ClassificationPrediction) for r in results)


def test_load_predictor_device_override_converts_cuda_artifact_to_cpu(tmp_path):
    forge.random.seed(1)
    model = _tabular_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(
        model, str(model_path), preprocessing=_transform(), classes=["normal", "warning", "critical"],
        task="tabular_classification",
    )

    predictor = load_predictor(str(model_path), device="cpu")
    assert predictor.model.device.type == "cpu"

    batch = np.random.default_rng(1).standard_normal((1, _N_FEATURES)).astype(np.float32)
    result = predictor.predict(batch)
    assert isinstance(result[0], ClassificationPrediction)


def test_predictor_stays_on_cuda_across_repeated_predictions(tmp_path):
    """No per-call device move: the model's device is fixed at load time and
    every subsequent `predict()` runs on it unchanged."""
    forge.random.seed(2)
    model = _tabular_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(
        model, str(model_path), preprocessing=_transform(), classes=["normal", "warning", "critical"],
        task="tabular_classification",
    )

    predictor = load_predictor(str(model_path))
    batch = np.random.default_rng(2).standard_normal((1, _N_FEATURES)).astype(np.float32)
    for _ in range(5):
        predictor.predict(batch)
        assert predictor.model.device.type == "cuda"


def test_predictor_cpu_and_cuda_predictions_agree(tmp_path):
    forge.random.seed(3)
    model = _tabular_model()
    batch = np.random.default_rng(3).standard_normal((3, _N_FEATURES)).astype(np.float32)

    cpu_path = tmp_path / "cpu_model.forge"
    save_model(
        model, str(cpu_path), preprocessing=_transform(), classes=["normal", "warning", "critical"],
        task="tabular_classification",
    )

    model.to("cuda")
    cuda_path = tmp_path / "cuda_model.forge"
    save_model(
        model, str(cuda_path), preprocessing=_transform(), classes=["normal", "warning", "critical"],
        task="tabular_classification",
    )

    cpu_predictor = load_predictor(str(cpu_path))
    cuda_predictor = load_predictor(str(cuda_path))

    cpu_results = cpu_predictor.predict(batch)
    cuda_results = cuda_predictor.predict(batch)

    for cpu_result, cuda_result in zip(cpu_results, cuda_results):
        assert cpu_result.index == cuda_result.index
        assert cpu_result.label == cuda_result.label
        np.testing.assert_allclose(cpu_result.confidence, cuda_result.confidence, **TOL)


def test_predictor_predict_matches_predict_tabular_classification_artifact_on_cuda(tmp_path):
    forge.random.seed(4)
    model = _tabular_model().to("cuda")
    model_path = tmp_path / "model.forge"
    save_model(
        model, str(model_path), preprocessing=_transform(), classes=["normal", "warning", "critical"],
        task="tabular_classification",
    )
    batch = np.random.default_rng(4).standard_normal((2, _N_FEATURES)).astype(np.float32)

    predictor = load_predictor(str(model_path))
    via_predictor = predictor.predict(batch)
    direct = predict_tabular_classification_artifact(str(model_path), batch)

    for u, d in zip(via_predictor, direct):
        assert u.label == d.label
        assert u.index == d.index
        assert u.confidence == pytest.approx(d.confidence, abs=1e-6)
