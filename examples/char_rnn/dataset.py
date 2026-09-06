"""Character-level dataset construction for the char-RNN example (Milestone 50).

Builds a vocabulary directly from a text corpus and turns it into fixed-
length, non-overlapping `(input, target)` integer-index sequence pairs --
`target[i]` is always `input[i]` shifted one character ahead (next-character
prediction) -- wrapped in the existing `forge.data.TensorDataset`/
`DataLoader` so batching needs no new data-pipeline code. One-hot encoding
happens later, per timestep, directly from a batch's raw indices
(`train.py`'s `_one_hot`): Forge has no embedding/gather Tensor primitive,
and none is needed at this vocabulary size (see
`docs/development/m50-char-rnn.md`, Step 3 -- a documented workaround, not a
blocker).
"""

from __future__ import annotations

import numpy as np

from forge import Tensor
from forge.data import TensorDataset


class Vocab:
    """A fixed character<->index mapping, built once from a text corpus."""

    def __init__(self, text: str):
        if not text:
            raise ValueError("Vocab requires a non-empty corpus.")
        self.chars = sorted(set(text))
        self.size = len(self.chars)
        self._char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self._idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> np.ndarray:
        return np.array([self._char_to_idx[c] for c in text], dtype=np.int64)

    def decode(self, indices) -> str:
        return "".join(self._idx_to_char[int(i)] for i in indices)


def build_dataset(text: str, seq_len: int) -> "tuple[TensorDataset, Vocab]":
    """Non-overlapping `(input, target)` index-sequence pairs, each of shape `(seq_len,)`."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    vocab = Vocab(text)
    ids = vocab.encode(text)
    n_sequences = (len(ids) - 1) // seq_len
    if n_sequences < 1:
        raise ValueError(f"Corpus too short ({len(ids)} chars) for seq_len={seq_len}.")

    inputs = np.stack([ids[i * seq_len : (i + 1) * seq_len] for i in range(n_sequences)])
    targets = np.stack([ids[i * seq_len + 1 : (i + 1) * seq_len + 1] for i in range(n_sequences)])
    dataset = TensorDataset(Tensor(inputs), Tensor(targets))
    return dataset, vocab


__all__ = ["Vocab", "build_dataset"]
