"""Milestone 101 CUDA tests: the portable structural input contract on the
real CUDA backend.

The validation itself (`forge/training/inference.py::_validate_feature_count()`)
runs entirely on host-side `Tensor.shape`/metadata before any device
dispatch, so it needs no CUDA-specific logic -- these tests exist to prove
that fact directly (rejection happens with no CUDA kernel ever launched, and
identically whether the model/tensor lives on CPU or CUDA), and that a
CUDA-saved artifact's `input_schema` is derived the same way a CPU one's is.
Hardware-verified on the reference GeForce 940MX. See
`tests/test_input_schema.py` for the full CPU-side unit/integration coverage
this mirrors.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.exceptions import DataError
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import InputSchema, inspect_model, save_model
from forge.training import predict_model, predict_tabular_classification_artifact, predict_tensor_artifact

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_N_FEATURES = 6


def test_input_schema_derived_identically_for_a_cuda_saved_regression_artifact(tmp_path):
    forge.random.seed(0)
    model = Linear(_N_FEATURES, 1, device="cuda")
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="regression")

    info = inspect_model(str(path))
    assert info.device == "cuda"
    assert info.input_schema == InputSchema(feature_count=_N_FEATURES)


def test_predict_tensor_artifact_rejects_wrong_feature_count_on_a_cuda_artifact(tmp_path):
    forge.random.seed(1)
    model = Linear(_N_FEATURES, 1, device="cuda")
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="regression")

    with pytest.raises(DataError, match=rf"expected {_N_FEATURES} input feature\(s\), received {_N_FEATURES - 1}"):
        predict_tensor_artifact(str(path), np.zeros((1, _N_FEATURES - 1), dtype=np.float32))


def test_predict_tabular_classification_artifact_rejects_wrong_feature_count_on_cuda(tmp_path):
    forge.random.seed(2)
    model = Sequential(
        Linear(_N_FEATURES, 8, device="cuda"), ReLU(), Linear(8, 2, device="cuda"),
    )
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="tabular_classification", classes=["no", "yes"])

    with pytest.raises(DataError, match=rf"expected {_N_FEATURES} input feature\(s\)"):
        predict_tabular_classification_artifact(str(path), np.zeros((1, _N_FEATURES + 2), dtype=np.float32))


def test_predict_model_valid_input_still_predicts_on_cuda_after_validation(tmp_path):
    """The Milestone 101 check must not interfere with the ordinary,
    already-tested CUDA prediction path for correctly shaped input."""
    forge.random.seed(3)
    model = Linear(_N_FEATURES, 1, device="cuda")
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="regression")

    result = predict_model(str(path), np.zeros((2, _N_FEATURES), dtype=np.float32))
    assert result.shape == (2, 1)
    assert result.device.type == "cpu"  # predict() always returns a CPU Tensor
