"""Milestone 54 integration tests: the `examples/word_rnn` pipeline, CPU-only.

Exercises the exact pipeline `examples/word_rnn/train.py` runs --
`corpus.TEXT`/`VOCAB_WORDS` -> `dataset.build_dataset()` -> `DataLoader` ->
`WordRNN` (`Embedding` -> `RNNCell` -> `Linear`) -> per-timestep
`CrossEntropyLoss` -> `Adam` -- mirroring `tests/
test_char_rnn_example_integration.py`'s structure exactly. Uses a small,
hand-written tiny vocabulary/corpus (not the example's full ~1,800-word
one) so this suite stays fast; the full-scale corpus is exercised directly
by running `examples/word_rnn/train.py` itself (see
`docs/development/m54-product-direction.md`'s real-workload results), not
by the test suite.
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

from examples.word_rnn.dataset import Vocab, build_dataset  # noqa: E402
from examples.word_rnn.model import WordRNN, build_model  # noqa: E402
from examples.word_rnn.train import generate, train_one_epoch  # noqa: E402

_TINY_VOCAB = [
    "cat", "dog", "bird", "fish",
    "runs", "jumps", "sleeps", "eats",
    "fast", "slow", "well", "loudly",
    ".",
]
_TINY_CORPUS = (
    "cat runs fast . dog jumps well . bird sleeps slow . fish eats loudly . "
    "dog runs fast . cat jumps well . fish sleeps slow . bird eats loudly . "
) * 8


# -- dataset / vocab ----------------------------------------------------------


def test_vocab_covers_every_declared_word():
    vocab = Vocab(_TINY_VOCAB)
    assert vocab.size == len(_TINY_VOCAB)
    assert vocab.decode(vocab.encode(_TINY_VOCAB)) == _TINY_VOCAB


def test_build_dataset_shapes_and_next_word_alignment():
    seq_len = 8
    dataset, vocab = build_dataset(_TINY_CORPUS, _TINY_VOCAB, seq_len=seq_len)
    ids = vocab.encode(_TINY_CORPUS.split())
    assert len(dataset) == (len(ids) - 1) // seq_len

    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    np.testing.assert_array_equal(x.numpy(), ids[:seq_len])
    np.testing.assert_array_equal(y.numpy(), ids[1 : seq_len + 1])


def test_build_dataset_rejects_corpus_shorter_than_seq_len():
    with pytest.raises(ValueError):
        build_dataset("cat runs", _TINY_VOCAB, seq_len=100)


# -- model shape ----------------------------------------------------------------


def test_word_rnn_step_output_shape():
    vocab = Vocab(_TINY_VOCAB)
    model = build_model(vocab.size, embedding_dim=6, hidden_size=8)
    h0 = model.init_hidden(4)
    x0 = Tensor(np.zeros(4, dtype=np.int64))
    logits, h1 = model.step(x0, h0)
    assert logits.shape == (4, vocab.size)
    assert h1.shape == (4, 8)


# -- end-to-end training --------------------------------------------------------


def _train(seed: int = 0, epochs: int = 15, embedding_dim: int = 8, hidden_size: int = 24, seq_len: int = 8, batch_size: int = 4):
    forge.random.seed(seed)
    data_rng = np.random.default_rng(seed)
    dataset, vocab = build_dataset(_TINY_CORPUS, _TINY_VOCAB, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=data_rng)
    model = build_model(vocab.size, embedding_dim=embedding_dim, hidden_size=hidden_size, device="cpu")
    optimizer = Adam(model.parameters(), lr=5e-2)
    loss_fn = CrossEntropyLoss()

    losses = [train_one_epoch(model, loader, optimizer, loss_fn, "cpu") for _ in range(epochs)]
    return model, vocab, losses


def test_training_loss_drops_well_below_untrained_uniform_baseline():
    model, vocab, losses = _train()
    uniform_baseline = np.log(vocab.size)
    assert losses[0] < uniform_baseline * 1.5
    assert losses[-1] < uniform_baseline * 0.5, (
        f"expected meaningful learning: final loss {losses[-1]:.3f} should be well below the "
        f"untrained uniform-guess baseline {uniform_baseline:.3f}"
    )
    assert losses[-1] < losses[0]


def test_training_updates_every_parameter_including_embedding_table():
    forge.random.seed(0)
    dataset, vocab = build_dataset(_TINY_CORPUS, _TINY_VOCAB, seq_len=8)
    model = build_model(vocab.size, embedding_dim=6, hidden_size=8, device="cpu")
    before = {name: p.numpy().copy() for name, p in model.named_parameters()}

    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=np.random.default_rng(0))
    optimizer = Adam(model.parameters(), lr=1e-2)
    train_one_epoch(model, loader, optimizer, CrossEntropyLoss(), "cpu")

    updated_any_embedding_row = False
    for name, param in model.named_parameters():
        if name == "embedding.weight":
            # Only rows for words that actually appeared in this epoch's
            # batches receive a nonzero gradient/update -- proving *some*
            # row changed (not necessarily every row) is the correct claim
            # for a sparse embedding table, unlike every other parameter.
            updated_any_embedding_row = not np.allclose(param.numpy(), before[name])
            continue
        assert not np.allclose(param.numpy(), before[name]), f"parameter '{name}' did not update"
    assert updated_any_embedding_row, "no embedding row was updated by a full epoch"


def test_generate_produces_requested_length_and_only_vocab_words():
    model, vocab, _ = _train(epochs=3)
    rng = np.random.default_rng(42)
    sample = generate(model, vocab, seed_words=["cat", "runs"], length=20, device="cpu", rng=rng)
    assert len(sample) == 2 + 20
    assert set(sample) <= set(vocab.words)


# -- persistence ----------------------------------------------------------------


def test_model_persistence_preserves_predictions(tmp_path):
    model, vocab, _ = _train(epochs=3)
    model_path = tmp_path / "word_rnn_model.forge"

    with no_grad():
        h0 = model.init_hidden(1)
        x0 = Tensor(np.array([0], dtype=np.int64))
        pre_logits, _ = model.step(x0, h0)
        pre_logits = pre_logits.numpy()

    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cpu")
    assert isinstance(reloaded, WordRNN)

    with no_grad():
        h0 = reloaded.init_hidden(1)
        post_logits, _ = reloaded.step(x0, h0)
        post_logits = post_logits.numpy()

    np.testing.assert_allclose(pre_logits, post_logits, atol=1e-6)
