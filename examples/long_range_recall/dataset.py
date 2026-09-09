"""The long-range-recall task (Milestone 67).

A synthetic binary sequence-classification task, structurally the same as
the "temporal order"/"copy" tasks Hochreiter & Schmidhuber (1997) and later
RNN-benchmark papers use to demonstrate the vanilla-RNN vanishing-gradient
problem: every sequence has `seq_len` timesteps, each a random one-hot
`{0, 1}` symbol, EXCEPT timestep 0, which is set to the true label. A model
must predict the label from its FINAL hidden state alone, after seeing
`seq_len - 1` further random (label-independent) symbols -- this forces
whatever recurrence processes the sequence to carry information from step 0
all the way to step `seq_len - 1`, which is exactly the mechanism a vanilla
RNN's repeated `tanh`-bounded multiplicative recurrence provably degrades
geometrically with `seq_len` (measured directly in
`docs/development/m67-lstm-long-range-recall.md`).

Random (rather than a constant "blank") distractor symbols at every other
step is the harder of the two classic task variants -- a constant blank
lets a recurrence settle into an easy fixed point; a genuinely resampled
distractor forces the model to keep discriminating "new information" from
"noise" at every step, which is the more realistic long-sequence setting.
"""

from __future__ import annotations

import numpy as np

from forge.data import TensorDataset
from forge.tensor.tensor import Tensor

VOCAB_SIZE = 2


def build_dataset(n_sequences: int, seq_len: int, seed: int = 0) -> TensorDataset:
    """`n_sequences` independent recall sequences, each `(seq_len, VOCAB_SIZE)` one-hot.

    Returns a `TensorDataset(inputs, labels)`: `inputs` has shape
    `(n_sequences, seq_len, VOCAB_SIZE)` (float32), `labels` has shape
    `(n_sequences,)` (int64, the timestep-0 symbol each sequence must
    reproduce at its output).
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be positive, got {n_sequences}.")

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, VOCAB_SIZE, size=n_sequences)
    symbols = rng.integers(0, VOCAB_SIZE, size=(n_sequences, seq_len))
    symbols[:, 0] = labels

    one_hot = np.zeros((n_sequences, seq_len, VOCAB_SIZE), dtype=np.float32)
    seq_idx = np.arange(n_sequences)[:, None]
    t_idx = np.arange(seq_len)[None, :]
    one_hot[seq_idx, t_idx, symbols] = 1.0

    return TensorDataset(Tensor(one_hot), Tensor(labels.astype(np.int64)))


__all__ = ["VOCAB_SIZE", "build_dataset"]
