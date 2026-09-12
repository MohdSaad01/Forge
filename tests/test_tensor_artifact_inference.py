"""Milestone 83 tests: `forge.predict_tensor_artifact()`, the portable-artifact
inference API for models whose input is a plain numeric array rather than an
image file (e.g. regression).

Covers turning a `.forge` file + one already-batched numeric input directly
into a raw prediction `Tensor` -- composing `load_model()`/
`load_preprocessing()`/`predict()` in one call, with preprocessing optional
(unlike `predict_artifact()`, where it is mandatory) and no class-vocabulary
concept. See `forge/training/inference.py::predict_tensor_artifact()` and
`docs/development/m83-regression-artifact-workflow.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.data.transforms import Normalize
from forge.exceptions import DataError, PersistenceError
from forge.nn import Linear, ReLU, Sequential
from forge.serialization import load_model, load_preprocessing, save_model
from forge.training import predict, predict_tensor_artifact
from forge.training.inference import predict_tensor_artifact as predict_tensor_artifact_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
_N_FEATURES = 4


def _build_tiny_model() -> Sequential:
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, 1))


def _build_transform() -> Normalize:
    return Normalize(mean=np.array([1.0, -1.0, 0.5, 0.0]), std=np.array([2.0, 3.0, 1.0, 4.0]))


def _saved_tiny_model(tmp_path, *, preprocessing=None, seed=0) -> Path:
    forge.random.seed(seed)
    model = _build_tiny_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing)
    return path


# -- basic API shape ----------------------------------------------------------


def test_predict_tensor_artifact_is_reexported_consistently():
    assert forge.predict_tensor_artifact is predict_tensor_artifact
    assert forge.training.predict_tensor_artifact is predict_tensor_artifact
    assert predict_tensor_artifact is predict_tensor_artifact_direct


def test_predict_tensor_artifact_returns_a_raw_tensor_with_no_classes_involved(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    batch = np.random.default_rng(1).standard_normal((3, _N_FEATURES)).astype(np.float32)

    result = predict_tensor_artifact(str(model_path), batch)
    assert isinstance(result, forge.Tensor)
    assert result.shape == (3, 1)


def test_predict_tensor_artifact_accepts_tensor_ndarray_and_nested_list(tmp_path):
    model_path = _saved_tiny_model(tmp_path)
    raw = [[0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0]]

    from_list = predict_tensor_artifact(str(model_path), raw)
    from_array = predict_tensor_artifact(str(model_path), np.array(raw, dtype=np.float32))
    from_tensor = predict_tensor_artifact(str(model_path), forge.Tensor(np.array(raw, dtype=np.float32)))

    np.testing.assert_allclose(from_list.numpy(), from_array.numpy(), atol=1e-6)
    np.testing.assert_allclose(from_list.numpy(), from_tensor.numpy(), atol=1e-6)


def test_predict_tensor_artifact_matches_the_manual_load_predict_pipeline(tmp_path):
    """predict_tensor_artifact() must be a pure composition -- bit-for-bit
    identical to hand-assembling load_model()/load_preprocessing()/predict()
    (the exact sequence it replaces)."""
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    batch = np.random.default_rng(2).standard_normal((5, _N_FEATURES)).astype(np.float32)

    result = predict_tensor_artifact(str(model_path), batch)

    preprocessing = load_preprocessing(str(model_path))
    model = load_model(str(model_path))
    expected = predict(model, preprocessing(forge.Tensor(batch)))

    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


def test_predict_tensor_artifact_with_no_saved_preprocessing_uses_input_as_is(tmp_path):
    """Unlike predict_artifact() (mandatory preprocessing), a numeric artifact
    saved with no preprocessing is a real, valid state -- the input is passed
    to the model unchanged rather than raising."""
    model_path = _saved_tiny_model(tmp_path, preprocessing=None)
    batch = np.random.default_rng(3).standard_normal((2, _N_FEATURES)).astype(np.float32)

    result = predict_tensor_artifact(str(model_path), batch)

    model = load_model(str(model_path))
    expected = predict(model, forge.Tensor(batch))
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


def test_predict_tensor_artifact_prediction_equals_pre_save_prediction(tmp_path):
    """The literal M83 numerical-equivalence requirement: the artifact's
    prediction on a sample must agree with the same, still-in-memory model's
    prediction on that sample computed before it was ever saved."""
    forge.random.seed(11)
    model = _build_tiny_model()
    transform = _build_transform()
    raw_batch = np.random.default_rng(4).standard_normal((6, _N_FEATURES)).astype(np.float32)

    pre_save = predict(model, transform(forge.Tensor(raw_batch))).numpy()

    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=transform)
    post_load = predict_tensor_artifact(str(model_path), raw_batch).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-5)


def test_predict_tensor_artifact_device_override_accepted_on_cpu_only_machine(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    batch = np.zeros((1, _N_FEATURES), dtype=np.float32)

    result = predict_tensor_artifact(str(model_path), batch, device="cpu")
    assert isinstance(result, forge.Tensor)


# -- error handling -------------------------------------------------------


def test_predict_tensor_artifact_rejects_unsupported_input_types(tmp_path):
    model_path = _saved_tiny_model(tmp_path)

    with pytest.raises(DataError):
        predict_tensor_artifact(str(model_path), "not a tensor")

    with pytest.raises(DataError):
        predict_tensor_artifact(str(model_path), 12345)


def test_predict_tensor_artifact_missing_model_file_raises_persistence_error(tmp_path):
    batch = np.zeros((1, _N_FEATURES), dtype=np.float32)

    with pytest.raises(PersistenceError):
        predict_tensor_artifact(str(tmp_path / "nope.forge"), batch)


# -- fresh process ----------------------------------------------------------


def test_predict_tensor_artifact_works_from_a_genuinely_separate_process(tmp_path):
    """The whole point of Milestone 83 is that a `.forge` file alone is
    enough -- prove it by launching a real, separate OS process that only
    imports `forge`/`numpy` and calls `forge.predict_tensor_artifact()`, with
    no shared memory, module cache, or import state with this test process.

    Built entirely from pre-registered `forge.nn` types (`Sequential`,
    `Linear`, `ReLU`) and the pre-registered `Normalize` transform, so the
    fresh subprocess (which never imports this test module) can reconstruct
    both without any custom `register_module()`/`register_transform()` call.
    """
    forge.random.seed(5)
    model = _build_tiny_model()
    transform = _build_transform()
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=transform)

    raw_x = np.random.default_rng(6).standard_normal((1, _N_FEATURES)).astype(np.float32)
    raw_x_literal = raw_x.tolist()

    script = (
        "import numpy as np\n"
        "import forge\n"
        f"raw_x = np.array({raw_x_literal!r}, dtype=np.float32)\n"
        f"prediction = forge.predict_tensor_artifact({str(model_path)!r}, raw_x)\n"
        "print(f'prediction={prediction.numpy()[0, 0]:.8f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    expected = predict_tensor_artifact(str(model_path), raw_x)
    expected_value = float(expected.numpy()[0, 0])
    assert f"prediction={expected_value:.8f}" in result.stdout
