"""Word-level dataset construction for the word-RNN example (Milestone 54).

Mirrors `examples/char_rnn/dataset.py` exactly (`Vocab` + non-overlapping
fixed-length `(input, target)` index-sequence pairs wrapped in the existing
`forge.data.TensorDataset`/`DataLoader`), tokenizing by whitespace instead
of by character. The one deliberate difference: `Vocab` here is built from
the corpus module's explicit `VOCAB_WORDS` list rather than empirically from
whichever words happen to appear in `TEXT` -- the vocabulary is itself the
thing under test (`nn.Embedding` at a realistic multi-hundred/low-thousands
scale), so it must have its full, fixed size regardless of the generated
corpus's actual word-frequency distribution.
"""

from __future__ import annotations

import numpy as np

from forge import Tensor
from forge.data import TensorDataset


class Vocab:
    """A fixed word<->index mapping, built from an explicit word list."""

    def __init__(self, words: "list[str]"):
        if not words:
            raise ValueError("Vocab requires a non-empty word list.")
        self.words = list(dict.fromkeys(words))  # de-duplicate, preserve order
        self.size = len(self.words)
        self._word_to_idx = {w: i for i, w in enumerate(self.words)}
        self._idx_to_word = {i: w for i, w in enumerate(self.words)}

    def encode(self, tokens) -> np.ndarray:
        return np.array([self._word_to_idx[t] for t in tokens], dtype=np.int64)

    def decode(self, indices) -> "list[str]":
        return [self._idx_to_word[int(i)] for i in indices]


def build_dataset(text: str, vocab_words: "list[str]", seq_len: int) -> "tuple[TensorDataset, Vocab]":
    """Non-overlapping `(input, target)` word-index-sequence pairs, each of shape `(seq_len,)`."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    vocab = Vocab(vocab_words)
    ids = vocab.encode(text.split())
    n_sequences = (len(ids) - 1) // seq_len
    if n_sequences < 1:
        raise ValueError(f"Corpus too short ({len(ids)} tokens) for seq_len={seq_len}.")

    inputs = np.stack([ids[i * seq_len : (i + 1) * seq_len] for i in range(n_sequences)])
    targets = np.stack([ids[i * seq_len + 1 : (i + 1) * seq_len + 1] for i in range(n_sequences)])
    dataset = TensorDataset(Tensor(inputs), Tensor(targets))
    return dataset, vocab


__all__ = ["Vocab", "build_dataset"]
