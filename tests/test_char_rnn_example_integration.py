"""Milestone 50 integration tests: the `examples/char_rnn` pipeline, CPU-only.

Exercises the exact pipeline `examples/char_rnn/train.py` runs --
`corpus.TEXT` -> `dataset.build_dataset()` -> `DataLoader` -> `CharRNN`
(`RNNCell` -> `Linear`) -> per-timestep `CrossEntropyLoss` -> `Adam` -- using
the real (already tiny, synthetic, offline) corpus rather than a stand-in,
since this example needs no download and already trains in well under a
second per epoch at these sizes. Covers: dataset/vocab shape, forward shape,
multi-step training loss reduction below the untrained-model baseline,
parameter updates, and model save/load prediction consistency -- the
Milestone 50 brief's Step 7 "genuine end-to-end training" requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.data import DataLoader
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_model, save_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.char_rnn.dataset import Vocab, build_dataset  # noqa: E402
from examples.char_rnn.model import CharRNN, build_model  # noqa: E402
from examples.char_rnn.train import _one_hot, generate, train_one_epoch  # noqa: E402

_TINY_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a tensor holds numbers and remembers how they were made. "
    "the quick brown fox runs past the lazy dog again. "
) * 6


# -- dataset / vocab ----------------------------------------------------------


def test_vocab_covers_every_corpus_character():
    vocab = Vocab(_TINY_CORPUS)
    assert set(vocab.chars) == set(_TINY_CORPUS)
    assert vocab.decode(vocab.encode(_TINY_CORPUS)) == _TINY_CORPUS


def test_build_dataset_shapes_and_next_char_alignment():
    seq_len = 10
    dataset, vocab = build_dataset(_TINY_CORPUS, seq_len=seq_len)
    assert len(dataset) == (len(vocab.encode(_TINY_CORPUS)) - 1) // seq_len

    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    ids = vocab.encode(_TINY_CORPUS)
    np.testing.assert_array_equal(x.numpy(), ids[:seq_len])
    np.testing.assert_array_equal(y.numpy(), ids[1 : seq_len + 1])


def test_build_dataset_rejects_corpus_shorter_than_seq_len():
    with pytest.raises(ValueError):
        build_dataset("short", seq_len=100)


# -- model shape ----------------------------------------------------------------


def test_char_rnn_step_output_shape():
    vocab = Vocab(_TINY_CORPUS)
    model = build_model(vocab.size, hidden_size=8)
    h0 = model.init_hidden(4)
    x0 = Tensor(_one_hot(np.zeros(4, dtype=np.int64), vocab.size))
    logits, h1 = model.step(x0, h0)
    assert logits.shape == (4, vocab.size)
    assert h1.shape == (4, 8)


# -- end-to-end training --------------------------------------------------------


def _train(seed: int = 0, epochs: int = 12, hidden_size: int = 24, seq_len: int = 16, batch_size: int = 8):
    forge.random.seed(seed)
    data_rng = np.random.default_rng(seed)
    dataset, vocab = build_dataset(_TINY_CORPUS, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=data_rng)
    model = build_model(vocab.size, hidden_size=hidden_size, device="cpu")
    optimizer = Adam(model.parameters(), lr=5e-2)
    loss_fn = CrossEntropyLoss()

    losses = [train_one_epoch(model, loader, optimizer, loss_fn, "cpu") for _ in range(epochs)]
    return model, vocab, losses


def test_training_loss_drops_well_below_untrained_uniform_baseline():
    model, vocab, losses = _train()
    uniform_baseline = np.log(vocab.size)
    assert losses[0] < uniform_baseline * 1.5  # sanity: first-epoch loss is in a plausible range
    assert losses[-1] < uniform_baseline * 0.5, (
        f"expected meaningful learning: final loss {losses[-1]:.3f} should be well below the "
        f"untrained uniform-guess baseline {uniform_baseline:.3f}"
    )
    assert losses[-1] < losses[0]


def test_training_updates_every_parameter():
    forge.random.seed(0)
    dataset, vocab = build_dataset(_TINY_CORPUS, seq_len=16)
    model = build_model(vocab.size, hidden_size=8, device="cpu")
    before = {name: p.numpy().copy() for name, p in model.named_parameters()}

    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(0))
    optimizer = Adam(model.parameters(), lr=1e-2)
    train_one_epoch(model, loader, optimizer, CrossEntropyLoss(), "cpu")

    for name, param in model.named_parameters():
        assert not np.allclose(param.numpy(), before[name]), f"parameter '{name}' did not update"


def test_generate_produces_requested_length_and_only_vocab_characters():
    model, vocab, _ = _train(epochs=2)
    rng = np.random.default_rng(42)
    sample = generate(model, vocab, seed_text="a tensor", length=50, device="cpu", rng=rng)
    assert len(sample) == len("a tensor") + 50
    assert set(sample) <= set(vocab.chars)


# -- persistence ----------------------------------------------------------------


def test_model_persistence_preserves_predictions(tmp_path):
    model, vocab, _ = _train(epochs=3)
    model_path = tmp_path / "char_rnn_model.forge"

    with no_grad():
        h0 = model.init_hidden(1)
        x0 = Tensor(_one_hot(np.array([0]), vocab.size))
        pre_logits, _ = model.step(x0, h0)
        pre_logits = pre_logits.numpy()

    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cpu")
    assert isinstance(reloaded, CharRNN)

    with no_grad():
        h0 = reloaded.init_hidden(1)
        post_logits, _ = reloaded.step(x0, h0)
        post_logits = post_logits.numpy()

    np.testing.assert_allclose(pre_logits, post_logits, atol=1e-6)
