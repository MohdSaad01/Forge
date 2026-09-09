"""Milestone 67 integration tests: the `examples/long_range_recall` pipeline, CPU-only.

Exercises the exact pipeline `examples/long_range_recall/train.py` runs --
`dataset.build_dataset()` -> `DataLoader` -> `RecallModel` (`RNNCell` or
`LSTMCell` -> `Linear`) -> `CrossEntropyLoss` -> `Adam` -- at small,
fast settings (short `seq_len`, few epochs) rather than the milestone
report's full-scale numbers, mirroring `test_char_rnn_example_integration.py`'s
convention. Covers: dataset shape, both cells' forward shape, multi-step
training reducing loss below the untrained chance baseline for both cells,
parameter updates, model save/load prediction consistency, and the
gradient-norm-decay diagnostic (`train.py::gradient_probe`) itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import no_grad
from forge.data import DataLoader
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_model, save_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.long_range_recall.dataset import VOCAB_SIZE, build_dataset  # noqa: E402
from examples.long_range_recall.model import RecallModel, build_model  # noqa: E402
from examples.long_range_recall.train import _forward_sequence, evaluate, train_one_epoch  # noqa: E402


# -- dataset --------------------------------------------------------------------


def test_build_dataset_shapes_and_label_alignment():
    dataset = build_dataset(n_sequences=20, seq_len=6, seed=0)
    inputs, labels = dataset.tensors
    assert inputs.shape == (20, 6, VOCAB_SIZE)
    assert labels.shape == (20,)
    one_hot = inputs.numpy()
    # timestep 0's one-hot argmax must equal the label at every sequence.
    np.testing.assert_array_equal(np.argmax(one_hot[:, 0, :], axis=1), labels.numpy())


def test_build_dataset_rejects_non_positive_sizes():
    with pytest.raises(ValueError):
        build_dataset(n_sequences=0, seq_len=5)
    with pytest.raises(ValueError):
        build_dataset(n_sequences=5, seq_len=0)


# -- model shape ------------------------------------------------------------------


@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
def test_recall_model_forward_shape(cell_type):
    model = build_model(cell_type, VOCAB_SIZE, hidden_size=8)
    batch_np = build_dataset(n_sequences=4, seq_len=5, seed=1).tensors[0].numpy()
    logits = _forward_sequence(model, batch_np, device="cpu")
    assert logits.shape == (4, VOCAB_SIZE)


# -- end-to-end training ----------------------------------------------------------


def _train(cell_type: str, seed: int = 0, epochs: int = 15, hidden_size: int = 16, seq_len: int = 8, batch_size: int = 16):
    forge.random.seed(seed)
    data_rng = np.random.default_rng(seed)
    dataset = build_dataset(n_sequences=256, seq_len=seq_len, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=data_rng)
    model = build_model(cell_type, VOCAB_SIZE, hidden_size=hidden_size, device="cpu")
    optimizer = Adam(model.parameters(), lr=2e-2)
    loss_fn = CrossEntropyLoss()

    losses = [train_one_epoch(model, loader, optimizer, loss_fn, "cpu") for _ in range(epochs)]
    return model, losses


@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
def test_training_loss_drops_well_below_untrained_chance_baseline(cell_type):
    model, losses = _train(cell_type)
    chance_baseline = np.log(VOCAB_SIZE)
    assert losses[0] < chance_baseline * 1.5  # sanity: first-epoch loss is in a plausible range
    assert losses[-1] < chance_baseline * 0.3, (
        f"expected meaningful learning: final loss {losses[-1]:.3f} should be well below the "
        f"untrained chance baseline {chance_baseline:.3f}"
    )
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
def test_training_updates_every_parameter(cell_type):
    forge.random.seed(0)
    dataset = build_dataset(n_sequences=64, seq_len=8, seed=0)
    model = build_model(cell_type, VOCAB_SIZE, hidden_size=8, device="cpu")
    before = {name: p.numpy().copy() for name, p in model.named_parameters()}

    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(0))
    optimizer = Adam(model.parameters(), lr=1e-2)
    train_one_epoch(model, loader, optimizer, CrossEntropyLoss(), "cpu")

    for name, param in model.named_parameters():
        assert not np.allclose(param.numpy(), before[name]), f"parameter '{name}' did not update"


@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
def test_trained_model_beats_chance_on_held_out_evaluation(cell_type):
    model, _ = _train(cell_type, epochs=25)
    eval_dataset = build_dataset(n_sequences=128, seq_len=8, seed=99)
    acc = evaluate(model, eval_dataset, "cpu")
    assert acc > 1.0 / VOCAB_SIZE + 0.3


# -- persistence ----------------------------------------------------------------


@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
def test_model_persistence_preserves_predictions(cell_type, tmp_path):
    model, _ = _train(cell_type, epochs=3)
    model_path = tmp_path / f"recall_{cell_type}_model.forge"
    batch_np = build_dataset(n_sequences=4, seq_len=8, seed=7).tensors[0].numpy()

    with no_grad():
        pre_logits = _forward_sequence(model, batch_np, "cpu").numpy()

    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cpu")
    assert isinstance(reloaded, RecallModel)
    assert reloaded.cell_type == cell_type

    with no_grad():
        post_logits = _forward_sequence(reloaded, batch_np, "cpu").numpy()

    np.testing.assert_allclose(pre_logits, post_logits, atol=1e-6)


# -- gradient-norm-decay diagnostic ------------------------------------------------


def test_gradient_probe_shows_lstm_surviving_further_than_rnn():
    """`train.py::gradient_probe`'s underlying mechanism, checked directly
    (not just printed): at a fixed sequence length, the gradient `LSTMCell`
    delivers to the first timestep must be many orders of magnitude larger
    (i.e. less vanished) than what `RNNCell` delivers, the core evidence
    motivating this milestone (`docs/development/m67-lstm-long-range-recall.md`)."""
    from forge import Tensor

    seq_len = 25
    hidden_size = 16
    batch_size = 8

    def first_timestep_grad_norm(cell_type: str) -> float:
        forge.random.seed(0)
        gen = forge.random.default_generator()
        model = build_model(cell_type, VOCAB_SIZE, hidden_size, generator=gen)
        loss_fn = CrossEntropyLoss()
        dataset = build_dataset(n_sequences=batch_size, seq_len=seq_len, seed=0)
        batch_np, labels_np = dataset.tensors[0].numpy(), dataset.tensors[1].numpy()

        state = model.init_state(batch_size)
        x0 = Tensor(batch_np[:, 0, :], requires_grad=True)
        state = model.step(x0, state)
        for t in range(1, seq_len):
            state = model.step(Tensor(batch_np[:, t, :]), state)
        logits = model.predict(state)
        loss_fn(logits, Tensor(labels_np)).backward()
        return float(np.linalg.norm(x0.grad.numpy()))

    rnn_grad = first_timestep_grad_norm("rnn")
    lstm_grad = first_timestep_grad_norm("lstm")
    assert lstm_grad > rnn_grad * 1e3, (
        f"expected LSTMCell's surviving gradient ({lstm_grad:.3e}) to be orders of magnitude "
        f"larger than RNNCell's ({rnn_grad:.3e}) at seq_len={seq_len}"
    )
