"""`RecallModel`: a single recurrent cell (vanilla `RNNCell` or `LSTMCell`)
feeding a `Linear` classifier head, used to compare the two cells on
`examples/long_range_recall`'s long-range-dependency task (Milestone 67).

```text
one-hot(symbol_t)  (batch, vocab_size)
    -> RNNCell|LSTMCell(vocab_size, hidden_size) -> state_t
    -> [after the final timestep] Linear(hidden_size, vocab_size) -> logits
```

`step()` mirrors `examples/char_rnn/model.py::CharRNN.step()`'s one-timestep
shape, generalized to an opaque `state` object so `train.py`'s unroll loop
does not need to know whether `state` is a bare hidden Tensor (`RNNCell`) or
an `(h, c)` pair (`LSTMCell`) -- the two cells' *shapes* differ, but nothing
about training/evaluating a classifier on top of either one does.

Registers itself for persistence the same way `CharRNN`/`WordRNN` do (a
hand-written composite `Module`, not one of the library's built-in
registered types).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge.exceptions import ModuleError
from forge.nn import LSTMCell, Linear, Module, RNNCell
from forge.serialization import register_module
from forge.tensor.tensor import Tensor

_CELL_TYPES = ("rnn", "lstm")


class RecallModel(Module):
    def __init__(
        self,
        cell_type: str,
        vocab_size: int,
        hidden_size: int,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        if cell_type not in _CELL_TYPES:
            raise ModuleError(f"RecallModel cell_type must be one of {_CELL_TYPES}, got {cell_type!r}.")
        self.cell_type = cell_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        if cell_type == "rnn":
            self.cell = RNNCell(vocab_size, hidden_size, dtype=dtype, device=device, generator=generator)
        else:
            self.cell = LSTMCell(vocab_size, hidden_size, dtype=dtype, device=device, generator=generator)
        self.head = Linear(hidden_size, vocab_size, dtype=dtype, device=device, generator=generator)

    def init_state(self, batch_size: int, device: str = "cpu"):
        """An opaque initial recurrent state: a Tensor for `RNNCell`, an `(h, c)` pair for `LSTMCell`."""
        return self.cell.init_hidden(batch_size, device=device)

    def step(self, x: Tensor, state):
        """One recurrence step: consumes one timestep's input, returns the next opaque `state`.

        Unlike `CharRNN.step()` (which returns `(logits, h)` every step,
        since char-level language modeling predicts at every timestep),
        `RecallModel` only ever needs a prediction from the *final* state --
        the recall task's whole point is that the label is not recoverable
        from any intermediate state alone. So `step()` only advances the
        recurrence; call `predict()` once, after the last timestep.
        """
        if self.cell_type == "rnn":
            h = self.cell(x, state)
            return h
        h, c = self.cell(x, state[0], state[1])
        return (h, c)

    def hidden_of(self, state) -> Tensor:
        """The hidden-state component of `state` (`state` itself for `RNNCell`, `state[0]` for `LSTMCell`)."""
        return state if self.cell_type == "rnn" else state[0]

    def predict(self, state) -> Tensor:
        return self.head(self.hidden_of(state))

    def __repr__(self) -> str:
        return (
            f"RecallModel(cell_type={self.cell_type!r}, vocab_size={self.vocab_size}, "
            f"hidden_size={self.hidden_size})"
        )


register_module(
    "RecallModel",
    RecallModel,
    get_config=lambda m: {
        "cell_type": m.cell_type,
        "vocab_size": m.vocab_size,
        "hidden_size": m.hidden_size,
    },
)


def build_model(
    cell_type: str, vocab_size: int, hidden_size: int = 32, device: str = "cpu", generator=None
) -> RecallModel:
    return RecallModel(cell_type, vocab_size, hidden_size, device=device, generator=generator)


__all__ = ["RecallModel", "build_model"]
