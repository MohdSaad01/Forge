"""Milestone 90 CUDA tests: `forge.predict_sequence_artifact()`/`forge.predict_model()`
on the real CUDA backend.

Mirrors `tests/test_unified_artifact_prediction_cuda.py`'s CPU-vs-CUDA
coverage for the fourth artifact shape: proves a `task="sequence"` model
trained/saved on CUDA reloads and generates on CUDA by default, and that a
CPU-saved artifact explicitly loaded onto CUDA (`device="cuda"`) produces
the same generated tokens as loading it on CPU, given the same explicit
`rng`. Hardware-verified on the reference GeForce 940MX -- see
`forge/training/inference.py::predict_sequence_artifact()`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.nn import Linear, Module, RNNCell
from forge.serialization import inspect_model, register_module, save_model
from forge.training import predict_model, predict_sequence_artifact

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_VOCAB = list("abcde ")


class TinyStepModelCUDA(Module):
    """Mirrors `tests/test_sequence_artifact_prediction.py::TinyStepModel` --
    duplicated (not imported) so this CUDA-gated file has no import-time
    dependency on the always-run CPU test file, matching this repo's
    established CUDA-test-file split convention."""

    def __init__(self, vocab_size=len(_VOCAB), hidden_size=8, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RNNCell(vocab_size, hidden_size, device=device)
        self.output = Linear(hidden_size, vocab_size, device=device)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.output(h), h

    def init_hidden(self, batch_size, device="cpu"):
        return self.cell.init_hidden(batch_size, device=device)


register_module(
    "TinyStepModelCUDA",
    TinyStepModelCUDA,
    get_config=lambda m: {"vocab_size": m.vocab_size, "hidden_size": m.hidden_size},
)


def test_predict_sequence_artifact_generates_on_cuda_by_default(tmp_path):
    forge.random.seed(0)
    model = TinyStepModelCUDA(device="cuda")
    path = tmp_path / "sequence_cuda.forge"
    save_model(model, str(path), classes=list(_VOCAB), task="sequence")
    assert inspect_model(str(path)).device == "cuda"

    generated = predict_sequence_artifact(str(path), ["a", "b"], length=15, rng=np.random.default_rng(0))
    assert len(generated) == 17
    assert set(generated) <= set(_VOCAB)


def test_predict_model_dispatches_sequence_artifact_on_cuda(tmp_path):
    forge.random.seed(1)
    model = TinyStepModelCUDA(device="cuda")
    path = tmp_path / "sequence_cuda.forge"
    save_model(model, str(path), classes=list(_VOCAB), task="sequence")

    result = predict_model(str(path), ["a"], length=10)
    assert isinstance(result, list)
    assert len(result) == 11


def test_cpu_saved_sequence_artifact_matches_across_cpu_and_cuda_devices(tmp_path):
    """The same CPU-saved artifact, explicitly loaded onto CPU vs. CUDA with
    the same explicit `rng` seed, must generate the same tokens -- the
    stepwise recurrence's forward math must agree between backends, exactly
    as every other CPU/CUDA parity test in this repo already establishes for
    `RNNCell`/`Linear` individually (`tests/test_rnn_cuda.py`)."""
    forge.random.seed(2)
    model = TinyStepModelCUDA(device="cpu")
    path = tmp_path / "sequence_cpu.forge"
    save_model(model, str(path), classes=list(_VOCAB), task="sequence")

    cpu_result = predict_sequence_artifact(
        str(path), ["a", "c"], length=12, device="cpu", rng=np.random.default_rng(5)
    )
    cuda_result = predict_sequence_artifact(
        str(path), ["a", "c"], length=12, device="cuda", rng=np.random.default_rng(5)
    )
    assert cpu_result == cuda_result
