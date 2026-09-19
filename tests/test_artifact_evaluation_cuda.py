"""Milestone 113 CUDA tests: `ArtifactPredictor.evaluate()` on the real CUDA backend.

Device-specific behaviour only (the metric/validation contract is covered on
CPU in `tests/test_artifact_evaluation.py`): a predictor loaded with
`device="cuda"` evaluates on the CUDA model, and CPU and CUDA evaluation of the
same artifact agree -- exactly (confusion matrix, accuracy) or within the
project's float tolerance (loss). Uses the committed Pima diabetes artifact and
holdout, plus a synthetic regression artifact. No CUDA-specific evaluation code
exists; these tests exercise the existing predictor/CUDA kernels.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.evaluate import load_labeled_csv  # noqa: E402

_PIMA_MODEL = str(_REPO_ROOT / "examples" / "tabular_diabetes" / "artifacts" / "tabular_diabetes_model.forge")
_PIMA_HOLDOUT = str(_REPO_ROOT / "examples" / "tabular_diabetes" / "data" / "diabetes_holdout_eval.csv")
TOL = dict(rel=1e-4, abs=1e-5)


def test_pima_cuda_evaluation_agrees_with_cpu():
    X, y = load_labeled_csv(_PIMA_HOLDOUT)
    cpu = load_predictor(_PIMA_MODEL, device="cpu")
    cuda = load_predictor(_PIMA_MODEL, device="cuda")
    assert cpu.model.device.type == "cpu" and cuda.model.device.type == "cuda"

    on_cpu = cpu.evaluate(X, y)
    on_cuda = cuda.evaluate(X, y)

    assert on_cuda.samples == on_cpu.samples == 154
    np.testing.assert_array_equal(on_cuda.confusion_matrix, on_cpu.confusion_matrix)
    assert on_cuda.accuracy == on_cpu.accuracy
    assert on_cuda.baseline_accuracy == on_cpu.baseline_accuracy
    assert on_cuda.loss == pytest.approx(on_cpu.loss, **TOL)
    assert on_cuda.precision == pytest.approx(on_cpu.precision, **TOL)
    assert on_cuda.recall == pytest.approx(on_cpu.recall, **TOL)
    assert on_cuda.accuracy == pytest.approx(0.7208, abs=5e-4)


def test_cuda_evaluation_is_repeatable_and_batch_size_independent():
    X, y = load_labeled_csv(_PIMA_HOLDOUT)
    cuda = load_predictor(_PIMA_MODEL, device="cuda")
    first = cuda.evaluate(X, y)
    for batch_size in (1, 32, 1000):
        again = cuda.evaluate(X, y, batch_size=batch_size)
        np.testing.assert_array_equal(again.confusion_matrix, first.confusion_matrix)
        assert again.loss == pytest.approx(first.loss, **TOL)


def test_cuda_evaluation_leaves_the_model_on_cuda_and_unchanged():
    X, y = load_labeled_csv(_PIMA_HOLDOUT)
    cuda = load_predictor(_PIMA_MODEL, device="cuda")
    before = [p.to("cpu").numpy().copy() for p in cuda.model.parameters()]
    cuda.model.train(True)
    cuda.evaluate(X, y)
    assert cuda.model.device.type == "cuda"
    assert cuda.model.training is True
    for a, p in zip(before, cuda.model.parameters()):
        np.testing.assert_array_equal(a, p.to("cpu").numpy())


def test_regression_cuda_evaluation_agrees_with_cpu(tmp_path):
    forge.random.seed(0)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 1))
    path = str(tmp_path / "regression.forge")
    save_model(
        model, path,
        preprocessing=Normalize(
            mean=np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
            std=np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32),
        ),
        task="regression",
    )
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 4)).astype(np.float32)
    y = rng.standard_normal(64).astype(np.float32)

    on_cpu = load_predictor(path, device="cpu").evaluate(X, y)
    on_cuda = load_predictor(path, device="cuda").evaluate(X, y, batch_size=16)

    assert on_cuda.samples == on_cpu.samples == 64
    assert on_cuda.mse == pytest.approx(on_cpu.mse, **TOL)
    assert on_cuda.mae == pytest.approx(on_cpu.mae, **TOL)
    assert on_cuda.loss == pytest.approx(on_cpu.loss, **TOL)
    assert on_cuda.baseline_mse == on_cpu.baseline_mse
