"""`WordRNN`: a small recurrent word-level language model (Milestone 54).

Structurally identical to `examples/char_rnn/model.py`'s `CharRNN` -- the
only architectural difference is the first layer:

```text
CharRNN:  one-hot(char_t)      (batch, vocab_size)       -> RNNCell -> Linear
WordRNN:  Embedding(word_t)    (batch, embedding_dim)     -> RNNCell -> Linear
```

`CharRNN` fed a one-hot vector directly into `RNNCell.i2h` (an ordinary
`Linear(vocab_size, hidden_size)`), which needed no embedding/gather
primitive because a one-hot-vector-times-Linear-weight is mathematically a
row lookup already -- just computed as a full `(batch, vocab_size) @
(vocab_size, hidden_size)` matmul instead of a real `O(batch,
embedding_dim)` gather. At `CharRNN`'s few-dozen-character vocabulary that
distinction was immaterial (`docs/development/m50-char-rnn.md`). At
`WordRNN`'s ~1,800-word vocabulary it is not -- see
`docs/development/m54-product-direction.md` for the direct measurement --
so `WordRNN` uses `nn.Embedding` (Milestone 54) instead, feeding its
`embedding_dim`-wide output into `RNNCell` exactly like `CharRNN` fed its
one-hot vector in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge.nn import Embedding, Linear, Module, RNNCell
from forge.serialization import register_module
from forge.tensor.tensor import Tensor


class WordRNN(Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = Embedding(vocab_size, embedding_dim, dtype=dtype, device=device, generator=generator)
        self.cell = RNNCell(embedding_dim, hidden_size, dtype=dtype, device=device, generator=generator)
        self.output = Linear(hidden_size, vocab_size, dtype=dtype, device=device, generator=generator)

    def step(self, token_ids: Tensor, h: Tensor) -> "tuple[Tensor, Tensor]":
        """One recurrence step: `(batch,)` int64 token ids in, `(batch, hidden_size)` hidden state in."""
        x = self.embedding(token_ids)
        h = self.cell(x, h)
        logits = self.output(h)
        return logits, h

    def init_hidden(self, batch_size: int, device: str = "cpu") -> Tensor:
        return self.cell.init_hidden(batch_size, device=device)

    def __repr__(self) -> str:
        return (
            f"WordRNN(vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, "
            f"hidden_size={self.hidden_size})"
        )


register_module(
    "WordRNN",
    WordRNN,
    get_config=lambda m: {
        "vocab_size": m.vocab_size,
        "embedding_dim": m.embedding_dim,
        "hidden_size": m.hidden_size,
    },
)


def build_model(
    vocab_size: int, embedding_dim: int = 32, hidden_size: int = 128, device: str = "cpu", generator=None
) -> WordRNN:
    return WordRNN(vocab_size, embedding_dim, hidden_size, device=device, generator=generator)


__all__ = ["WordRNN", "build_model"]
