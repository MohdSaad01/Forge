"""Milestone 119 CUDA tests: named-column alignment on the real CUDA backend.

The schema rule is device-free host logic (names are compared as strings and the columns permuted before
preprocessing and the model), so its behaviour is a CPU test (`test_named_input.py`, `test_cli_feature_schema.py`).
What is checked here is only that the alignment composes with the GPU paths: a CUDA-resident input `Tensor` is
reordered without leaving its device, a CUDA-loaded predictor scores a reordered named X exactly like the ordered one,
and a CPU-trained named artifact behaves the same when loaded onto CUDA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.cli.main import main as cli_main
from forge.exceptions import DataError
from forge.tensor.tensor import Tensor

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema_support import (  # noqa: E402
    FEATURES, assert_order_matters, make_data, permute, result_fields_equal, same_prediction, train_classifier,
    train_regressor, write_named_csv,
)

X, Y_REG, Y_CLS = make_data()
PERM = ["rooms", "income", "distance", "age"]


@pytest.fixture(scope="module")
def cuda_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("schema_cuda")
    train_regressor(root / "reg.forge", device="cuda")
    train_classifier(root / "cls.forge", device="cuda")
    return root


def test_a_cuda_trained_artifact_records_the_names_and_is_a_cuda_artifact(cuda_artifacts):
    info = forge.inspect_model(str(cuda_artifacts / "reg.forge"))
    assert info.device == "cuda" and info.input_schema.feature_names == tuple(FEATURES)


@pytest.mark.parametrize("name, y", [("reg", Y_REG), ("cls", Y_CLS)])
def test_a_cuda_predictor_scores_a_reordered_named_x_exactly_like_the_ordered_one(cuda_artifacts, name, y):
    predictor = forge.load_predictor(str(cuda_artifacts / f"{name}.forge"))
    assert predictor.model.device.type == "cuda"
    assert_order_matters(predictor, X, FEATURES, PERM)
    assert same_prediction(predictor.predict(permute(X, FEATURES, PERM), feature_names=PERM), predictor.predict(X))
    reference = predictor.evaluate(X, y)
    assert result_fields_equal(predictor.evaluate(permute(X, FEATURES, PERM), y, feature_names=PERM), reference)


def test_a_cuda_resident_input_tensor_is_reordered_without_leaving_the_device(cuda_artifacts, monkeypatch):
    """The results alone cannot show this (`predict()` moves any input to the model's device), so observe the
    tensor that actually reaches the model call: it must still be the CUDA one, not a host round-trip copy."""
    import forge.training.inference as inference

    seen = []
    real_predict = inference.predict
    monkeypatch.setattr(inference, "predict", lambda model, data, *a, **k: seen.append(data.device.type) or real_predict(model, data, *a, **k))
    predictor = forge.load_predictor(str(cuda_artifacts / "reg.forge"))
    moved = Tensor(permute(X, FEATURES, PERM).astype(np.float32), device="cuda")
    ordered = Tensor(X.astype(np.float32), device="cuda")
    assert same_prediction(predictor.predict(moved, feature_names=PERM), predictor.predict(ordered))
    assert seen == ["cuda", "cuda"]


def test_a_cuda_input_tensor_with_bad_names_is_still_rejected_before_any_kernel_runs(cuda_artifacts):
    predictor = forge.load_predictor(str(cuda_artifacts / "reg.forge"))
    with pytest.raises(DataError, match=r"unexpected .*\['id'\]"):
        predictor.predict(Tensor(np.column_stack([np.arange(len(X)), X]).astype(np.float32), device="cuda"),
                          feature_names=["id", *FEATURES])


def test_a_cpu_trained_named_artifact_aligns_the_same_when_loaded_onto_cuda(tmp_path):
    train_regressor(tmp_path / "cpu.forge")
    cpu = forge.load_predictor(str(tmp_path / "cpu.forge"), device="cpu")
    gpu = forge.load_predictor(str(tmp_path / "cpu.forge"), device="cuda")
    moved = permute(X, FEATURES, PERM)
    np.testing.assert_allclose(gpu.predict(moved, feature_names=PERM).numpy(),
                               cpu.predict(moved, feature_names=PERM).numpy(), rtol=1e-4, atol=1e-5)
    assert same_prediction(gpu.predict(moved, feature_names=PERM), gpu.predict(X))


def test_the_cli_evaluates_a_reordered_csv_on_cuda_like_the_ordered_one(cuda_artifacts, capsys):
    ordered = write_named_csv(cuda_artifacts / "o.csv", FEATURES, X, ("target", Y_REG))
    reordered = write_named_csv(cuda_artifacts / "r.csv", PERM, permute(X, FEATURES, PERM), ("target", Y_REG))
    outputs = []
    for csv in (ordered, reordered):
        code = cli_main(["model", "evaluate", str(cuda_artifacts / "reg.forge"), str(csv), "--target", "target",
                         "--device", "cuda", "--json"])
        captured = capsys.readouterr()
        assert code == 0 and captured.err == "", captured.err
        outputs.append(json.loads(captured.out))
    assert outputs[0] == outputs[1]
