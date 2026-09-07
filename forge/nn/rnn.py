"""A vanilla (Elman) recurrent cell (Milestone 50).

Added as the one genuine blocker discovered while attempting a small
character-level RNN language model with Forge as it existed after M49 --
see `docs/development/m50-char-rnn.md`. Composed entirely from two existing
`Linear` layers and the new `Tensor.tanh()` primitive (`forge/tensor/
tensor.py`), following the same "compose from existing Tensor ops, no
dedicated backward rule" convention `Dropout`/`ReLU` already use.

`RNNCell` computes one recurrence step; unrolling over a sequence (the
Python-level `for t in range(seq_len): h = cell(x_t, h)` loop) is the
caller's responsibility -- Forge has no sequence/time-axis abstraction, and
this milestone's model does not need one (see the module doc above for why
a 3D "batch of sequences" Tensor was deliberately not introduced). Reusing
the same `RNNCell` instance (and therefore the same `i2h`/`h2h` `Parameter`s)
at every timestep is ordinary Forge autograd weight-sharing -- `forge.
autograd.engine.run_backward` already accumulates gradients correctly for a
leaf tensor used as input to more than one graph node (verified directly
before this milestone added any new code; see the M50 report's Step 3).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import ShapeMismatchError
from ..tensor.tensor import Tensor
from .linear import Linear
from .module import Module


class RNNCell(Module):
    """One step of a vanilla RNN: `h' = tanh(x @ W_ih + b_ih + h @ W_hh)`.

    `input_size`/`hidden_size` fix the expected shapes of `x` (`(batch,
    input_size)`) and `h` (`(batch, hidden_size)`); the returned hidden state
    has shape `(batch, hidden_size)`. Only one bias term exists (on the
    input projection, `i2h`) -- `i2h`'s bias and a second `h2h` bias would be
    redundant (their sum is what matters, since both feed the same `+`
    before `tanh`), so `h2h` is constructed with `bias=False`, mirroring how
    `Conv2d`'s single bias is not duplicated elsewhere in Forge.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = Linear(input_size, hidden_size, bias=True, dtype=dtype, device=device, generator=generator)
        self.h2h = Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device, generator=generator)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ShapeMismatchError(
                f"RNNCell(input_size={self.input_size}) expects x of shape "
                f"(batch, {self.input_size}), got shape {x.shape}."
            )
        if h.ndim != 2 or h.shape[-1] != self.hidden_size:
            raise ShapeMismatchError(
                f"RNNCell(hidden_size={self.hidden_size}) expects h of shape "
                f"(batch, {self.hidden_size}), got shape {h.shape}."
            )
        if x.shape[0] != h.shape[0]:
            raise ShapeMismatchError(
                f"RNNCell got mismatched batch sizes: x has {x.shape[0]}, h has {h.shape[0]}."
            )
        if x.device.type == "cuda":
            # Milestone 55: a real sequence-model training step calls this
            # once per timestep (M50/M54's hand-written unrolled loop) --
            # composing from `Linear`/`+`/`.tanh()` costs ~4 forward + ~8
            # backward CUDA kernel launches per call, deep in the
            # launch-overhead-bound regime at Forge's actual RNN shapes (see
            # `docs/development/m55-post-m54-assessment.md`). Dispatches to
            # one fused forward+backward primitive instead, mirroring
            # `nn.BatchNorm2d`'s CUDA-only-fused / CPU-composed split.
            return x.rnn_cell(self.i2h.weight, self.i2h.bias, self.h2h.weight, h)
        return (self.i2h(x) + self.h2h(h)).tanh()

    def init_hidden(self, batch_size: int, dtype: Any = None, device: str = "cpu") -> Tensor:
        """A fresh all-zero initial hidden state of shape `(batch_size, hidden_size)`."""
        param_dtype = dtype if dtype is not None else self.i2h.weight.dtype
        return Tensor(np.zeros((batch_size, self.hidden_size)), dtype=param_dtype, device=device)

    def __repr__(self) -> str:
        return f"RNNCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


__all__ = ["RNNCell"]
