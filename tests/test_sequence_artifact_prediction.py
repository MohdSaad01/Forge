"""Milestone 90 tests: `forge.predict_sequence_artifact()`, the fourth
portable-artifact prediction workflow (alongside `predict_artifact()`/
`predict_tensor_artifact()`/`predict_image_artifact()`, Milestones 82-84),
and its wiring into `forge.predict_model()`/`forge model predict`.

Covers `save_model(..., task="sequence")` validation (requires `classes=`,
the persisted token vocabulary), `predict_sequence_artifact()` itself
(generation shape/vocabulary, unknown-token/empty-seed errors, the
missing-vocabulary error, and the clear failure for a loaded model that does
not implement the stepwise-recurrence protocol), `predict_model()` dispatch
(including the new `length=` parameter), CLI `forge model predict` for a
sequence artifact, and a genuine fresh-process run. See
`forge/training/inference.py::predict_sequence_artifact()`,
`forge/cli/model.py::cmd_predict()`, and
`docs/development/m90-sequence-artifact-inference.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.exceptions import DataError, PersistenceError
from forge.nn import Linear, Module, RNNCell, ReLU, Sequential
from forge.serialization import inspect_model, load_classes, register_module, save_model
from forge.training import predict_model, predict_sequence_artifact
from forge.training.inference import predict_sequence_artifact as predict_sequence_artifact_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VOCAB = list("abcde ")


class TinyStepModel(Module):
    """A minimal stepwise recurrent model: `RNNCell` -> `Linear`, one-hot input.

    Mirrors `examples/char_rnn/model.py::CharRNN`'s `step()`/`init_hidden()`
    shape exactly (the protocol `generate_sequence()`/`predict_sequence_
    artifact()` document) -- same pattern `tests/test_inference.py::
    TinyStepModel` already uses for `generate_sequence()`'s own tests,
    duplicated here (rather than imported) so this file has no cross-file
    test dependency.
    """

    def __init__(self, vocab_size=6, hidden_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RNNCell(vocab_size, hidden_size)
        self.output = Linear(hidden_size, vocab_size)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.output(h), h

    def init_hidden(self, batch_size, device="cpu"):
        return self.cell.init_hidden(batch_size, device=device)


register_module(
    "TinyStepModelM90",
    TinyStepModel,
    get_config=lambda m: {"vocab_size": m.vocab_size, "hidden_size": m.hidden_size},
)


def _step_model(seed=0, vocab_size=len(_VOCAB), hidden_size=8) -> TinyStepModel:
    forge.random.seed(seed)
    return TinyStepModel(vocab_size=vocab_size, hidden_size=hidden_size)


def _saved_sequence_model(tmp_path, *, vocab=None, seed=0, name="sequence.forge") -> Path:
    vocab = list(_VOCAB) if vocab is None else vocab
    model = _step_model(seed=seed, vocab_size=len(vocab))
    path = tmp_path / name
    save_model(model, str(path), classes=vocab, task="sequence")
    return path


# -- save_model(task="sequence") validation ----------------------------------


def test_save_model_task_sequence_requires_classes(tmp_path):
    model = _step_model()
    with pytest.raises(PersistenceError, match="requires classes="):
        save_model(model, str(tmp_path / "model.forge"), task="sequence")


def test_save_model_task_sequence_with_classes_round_trips_metadata(tmp_path):
    path = _saved_sequence_model(tmp_path)
    info = inspect_model(str(path))
    assert info.task == "sequence"
    assert info.classes == _VOCAB
    assert load_classes(str(path)) == _VOCAB


def test_task_types_includes_sequence():
    from forge.serialization.model import TASK_TYPES
    assert "sequence" in TASK_TYPES


# -- predict_sequence_artifact() ---------------------------------------------


def test_predict_sequence_artifact_generates_requested_length_from_vocab(tmp_path):
    path = _saved_sequence_model(tmp_path)
    seed = ["a", "b", "c"]
    rng = np.random.default_rng(0)

    generated = predict_sequence_artifact(str(path), seed, length=20, rng=rng)

    assert len(generated) == len(seed) + 20
    assert generated[: len(seed)] == seed
    assert set(generated) <= set(_VOCAB)


def test_predict_sequence_artifact_is_reexported_consistently():
    assert forge.predict_sequence_artifact is predict_sequence_artifact
    assert forge.training.predict_sequence_artifact is predict_sequence_artifact
    assert predict_sequence_artifact is predict_sequence_artifact_direct


def test_predict_sequence_artifact_rejects_empty_seed(tmp_path):
    path = _saved_sequence_model(tmp_path)
    with pytest.raises(DataError):
        predict_sequence_artifact(str(path), [], length=5)


def test_predict_sequence_artifact_rejects_unknown_seed_token(tmp_path):
    path = _saved_sequence_model(tmp_path)
    with pytest.raises(DataError, match="not in"):
        predict_sequence_artifact(str(path), ["a", "Z"], length=5)


def test_predict_sequence_artifact_requires_saved_vocabulary(tmp_path):
    model = _step_model()
    path = tmp_path / "no_vocab.forge"
    save_model(model, str(path))  # no classes=, no task=

    with pytest.raises(PersistenceError, match="no vocabulary"):
        predict_sequence_artifact(str(path), ["a"], length=5)


def test_predict_sequence_artifact_requires_stepwise_protocol(tmp_path):
    """A `task='sequence'` artifact whose model has no `step()`/`init_hidden()`
    (an ordinary `forward()`-based model) must fail with a clear
    `PersistenceError` naming the missing protocol, not a raw
    `AttributeError` from inside `generate_sequence()`."""
    forge.random.seed(0)
    model = Sequential(Linear(len(_VOCAB), len(_VOCAB)), ReLU())
    path = tmp_path / "not_stepwise.forge"
    save_model(model, str(path), classes=list(_VOCAB), task="sequence")

    with pytest.raises(PersistenceError, match="stepwise-recurrence protocol"):
        predict_sequence_artifact(str(path), ["a"], length=5)


def test_predict_sequence_artifact_deterministic_with_explicit_rng(tmp_path):
    path = _saved_sequence_model(tmp_path)
    first = predict_sequence_artifact(str(path), ["a", "b"], length=15, rng=np.random.default_rng(7))
    second = predict_sequence_artifact(str(path), ["a", "b"], length=15, rng=np.random.default_rng(7))
    assert first == second


def test_predict_sequence_artifact_does_not_mutate_the_model_on_disk(tmp_path):
    """Reloading and generating twice from the same artifact must produce
    the same result -- predict_sequence_artifact() reloads a fresh model
    each call and never writes back to `path`."""
    path = _saved_sequence_model(tmp_path)
    first = predict_sequence_artifact(str(path), ["a"], length=10, rng=np.random.default_rng(3))
    second = predict_sequence_artifact(str(path), ["a"], length=10, rng=np.random.default_rng(3))
    assert first == second


# -- predict_model() dispatch --------------------------------------------


def test_predict_model_dispatches_sequence_artifact(tmp_path):
    path = _saved_sequence_model(tmp_path)
    result = predict_model(str(path), ["a", "b"], length=10)
    assert isinstance(result, list)
    assert len(result) == 12
    assert set(result) <= set(_VOCAB)


def test_predict_model_requires_length_for_sequence_artifact(tmp_path):
    path = _saved_sequence_model(tmp_path)
    with pytest.raises(DataError, match="length="):
        predict_model(str(path), ["a", "b"])


def test_predict_model_matches_predict_sequence_artifact_bit_for_bit(tmp_path):
    path = _saved_sequence_model(tmp_path)

    forge.random.seed(100)
    via_predict_model = predict_model(str(path), ["a", "c"], length=12)

    forge.random.seed(100)
    via_direct = predict_sequence_artifact(str(path), ["a", "c"], length=12)

    assert via_predict_model == via_direct


def test_predict_model_rejects_a_tensor_for_a_sequence_artifact(tmp_path):
    path = _saved_sequence_model(tmp_path)
    with pytest.raises(DataError):
        predict_model(str(path), forge.Tensor(np.zeros((1, len(_VOCAB)), dtype=np.float32)), length=5)


# -- existing workflows unaffected -------------------------------------------


def test_classification_regression_segmentation_predict_model_unaffected_by_length_param(tmp_path):
    """`length=` is new, optional, and ignored for the other three workflows
    -- a regression call with no `length=` must behave exactly as before."""
    from forge.nn import Conv2d, Flatten, MaxPool2d

    forge.random.seed(0)
    reg_model = Sequential(Linear(4, 8), ReLU(), Linear(8, 1))
    reg_path = tmp_path / "regression.forge"
    save_model(reg_model, str(reg_path), task="regression")
    batch = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)

    result = predict_model(str(reg_path), batch)
    assert isinstance(result, forge.Tensor)
    assert result.shape == (2, 1)


# -- CLI ----------------------------------------------------------------


def test_cli_predict_sequence_generates_text(tmp_path, capsys):
    from forge.cli.main import main as cli_main

    path = _saved_sequence_model(tmp_path)
    exit_code = cli_main(["model", "predict", str(path), "ab", "--length", "10"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.startswith("Generated: ab")


def test_cli_predict_sequence_json_output(tmp_path, capsys):
    import json

    from forge.cli.main import main as cli_main

    path = _saved_sequence_model(tmp_path)
    exit_code = cli_main(["model", "predict", str(path), "ab", "--length", "10", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "sequence"
    assert payload["seed"] == "ab"
    assert payload["generated"].startswith("ab")
    assert len(payload["generated"]) == 12


def test_cli_predict_sequence_rejects_empty_seed(tmp_path, capsys):
    from forge.cli.main import main as cli_main

    path = _saved_sequence_model(tmp_path)
    exit_code = cli_main(["model", "predict", str(path), ""])
    assert exit_code == 1
    assert "non-empty seed" in capsys.readouterr().err


def test_cli_predict_sequence_does_not_require_input_to_be_a_file(tmp_path):
    """Unlike classification/regression/segmentation, the CLI's sequence
    input is literal seed text, not a file path -- a seed string that does
    not name any file on disk must still succeed (proving the CLI's
    file-existence check is skipped for the sequence task)."""
    from forge.cli.main import main as cli_main

    path = _saved_sequence_model(tmp_path)
    seed = "abc"  # every character is in _VOCAB, but no such file exists
    assert not (tmp_path / seed).exists()

    exit_code = cli_main(["model", "predict", str(path), seed])
    assert exit_code == 0


def test_cli_predict_missing_task_fails_clearly(tmp_path, capsys):
    from forge.cli.main import main as cli_main

    forge.random.seed(0)
    vocab_no_space = [c for c in _VOCAB if c.strip()]  # classification-style classes= rejects whitespace
    model = _step_model(vocab_size=len(vocab_no_space))
    path = tmp_path / "no_task.forge"
    save_model(model, str(path), classes=vocab_no_space)  # classes but no task

    exit_code = cli_main(["model", "predict", str(path), "ab"])
    assert exit_code == 1
    assert "does not declare a task" in capsys.readouterr().err


# -- fresh process --------------------------------------------------------


def test_predict_sequence_artifact_works_from_a_genuinely_separate_process(tmp_path):
    """The whole point of Milestone 90 is that `forge.predict_model()` alone
    -- with no access to `train.py`'s in-memory `Vocab`/model -- generates
    text from a sequence artifact. Prove it in one real, separate OS
    process, exactly like `test_unified_artifact_prediction.py` does for
    the other three artifact shapes."""
    path = _saved_sequence_model(tmp_path, seed=42)

    # The subprocess is a genuinely separate interpreter with no import of
    # this test module -- it registers its own, independently-defined
    # `TinyStepModelM90` class before loading, exactly as a real external
    # consumer's own model.py (e.g. examples/char_rnn/model.py) would
    # register itself on import. Reconstruction only needs a registered
    # class of the same name/config/parameter shapes, not the same Python
    # object -- see `forge/serialization/registry.py`.
    script = (
        "import forge\n"
        "from forge.nn import Linear, Module, RNNCell\n"
        "from forge.serialization import register_module\n"
        "class TinyStepModelM90(Module):\n"
        "    def __init__(self, vocab_size, hidden_size):\n"
        "        super().__init__()\n"
        "        self.vocab_size = vocab_size\n"
        "        self.hidden_size = hidden_size\n"
        "        self.cell = RNNCell(vocab_size, hidden_size)\n"
        "        self.output = Linear(hidden_size, vocab_size)\n"
        "    def step(self, x, h):\n"
        "        h = self.cell(x, h)\n"
        "        return self.output(h), h\n"
        "    def init_hidden(self, batch_size, device='cpu'):\n"
        "        return self.cell.init_hidden(batch_size, device=device)\n"
        "register_module('TinyStepModelM90', TinyStepModelM90, "
        "get_config=lambda m: {'vocab_size': m.vocab_size, 'hidden_size': m.hidden_size})\n"
        f"result = forge.predict_model({str(path)!r}, ['a', 'b'], length=10)\n"
        "print('generated=' + ''.join(result))\n"
        "print('length=' + str(len(result)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert "length=12" in result.stdout
    assert "generated=ab" in result.stdout
