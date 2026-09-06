"""`CharRNN`: the small recurrent architecture used by the char-RNN example (Milestone 50).

Built from exactly two existing/newly-added `forge.nn` layers -- no
example-specific numerical logic:

```text
one-hot(char_t)  (batch, vocab_size)
    -> RNNCell(vocab_size, hidden_size)  -> h_t  (batch, hidden_size)
    -> Linear(hidden_size, vocab_size)   -> logits_t  (batch, vocab_size)
```

`step()` runs one timestep; `train.py` owns the Python-level loop that
unrolls this over a sequence, carrying `h` forward and accumulating one
`CrossEntropyLoss` per timestep -- see that module's docstring for why this
is a hand-written loop rather than `forge.training.Trainer.fit()` (a
deliberate, documented workaround, not a framework gap: `Trainer` assumes
one `forward(batch) -> prediction` call per step, which a multi-timestep
recurrence does not fit, and Forge's M1-M5 hand-written
`zero_grad->forward->loss->backward->step` loop remains a fully supported,
precedented way to train outside that assumption).

`CharRNN` is a custom composite `Module` (not one of the library's built-in
registered types), so it registers itself for persistence via
`forge.serialization.register_module()` at import time -- the same
`docs/architecture/persistence.md` ADR-003 pattern any hand-written
multi-layer model must follow.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge.nn import Linear, Module, RNNCell
from forge.serialization import register_module
from forge.tensor.tensor import Tensor


class CharRNN(Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RNNCell(vocab_size, hidden_size, dtype=dtype, device=device, generator=generator)
        self.output = Linear(hidden_size, vocab_size, dtype=dtype, device=device, generator=generator)

    def step(self, x: Tensor, h: Tensor) -> "tuple[Tensor, Tensor]":
        """One recurrence step: `(batch, vocab_size)` input, `(batch, hidden_size)` hidden state in."""
        h = self.cell(x, h)
        logits = self.output(h)
        return logits, h

    def init_hidden(self, batch_size: int, device: str = "cpu") -> Tensor:
        return self.cell.init_hidden(batch_size, device=device)

    def __repr__(self) -> str:
        return f"CharRNN(vocab_size={self.vocab_size}, hidden_size={self.hidden_size})"


register_module(
    "CharRNN",
    CharRNN,
    get_config=lambda m: {"vocab_size": m.vocab_size, "hidden_size": m.hidden_size},
)


def build_model(vocab_size: int, hidden_size: int = 64, device: str = "cpu", generator=None) -> CharRNN:
    return CharRNN(vocab_size, hidden_size, device=device, generator=generator)


__all__ = ["CharRNN", "build_model"]
