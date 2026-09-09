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
from ..tensor.dtype import DEFAULT_DTYPE
from ..tensor.tensor import Tensor
from .linear import Linear
from .module import Module
from .parameter import Parameter


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


class LSTMCell(Module):
    """One step of an LSTM (Hochreiter & Schmidhuber 1997): input/forget/output
    gates plus a candidate cell update, gating how much of the running cell
    state `c` is kept vs. overwritten at each step.

    ```text
    i = sigmoid(x @ Wxi + h @ Whi + bi)   # input gate
    f = sigmoid(x @ Wxf + h @ Whf + bf)   # forget gate
    g = tanh(x @ Wxg + h @ Whg + bg)      # candidate cell update
    o = sigmoid(x @ Wxo + h @ Who + bo)   # output gate
    c' = f * c + i * g
    h' = o * tanh(c')
    ```

    Added in Milestone 67 as the smallest capability that closes a directly
    measured gap: `nn.RNNCell` (the vanilla/Elman recurrence Forge has had
    since M50) provably fails a long-range-dependency task (Forge's own
    `examples/long_range_recall/`) once the dependency spans more than
    roughly 100-150 timesteps, because its backpropagated gradient decays by
    many orders of magnitude per step (measured directly, not assumed -- see
    `docs/development/m67-lstm-long-range-recall.md`). The additive
    `c' = f*c + i*g` update gives the cell-state gradient a path with no
    repeated multiplication by a squashed (`tanh`-bounded) activation, which
    is exactly the mechanism that fixes this in the classic construction.

    Composed entirely from eight existing `Linear` layers (one `i2h`/`h2h`
    pair per gate, mirroring `RNNCell`'s own `i2h`/`h2h` split -- `h2h` has
    no bias, for the same "redundant with i2h's bias" reason `RNNCell`
    documents) plus the `+`/`*`/`.sigmoid()`/`.tanh()` Tensor primitives, so
    it needs no dedicated backward rule and no CUDA kernel of its own --
    every op it uses already has CPU+CUDA forward/backward support. This
    mirrors `RNNCell`'s *original* M50 form (a dedicated fused CUDA kernel
    was added later, in M55, only once real per-timestep launch-overhead
    pressure was measured; no such measurement has been made for
    `LSTMCell` yet, so no fused kernel is added here -- see the M67 report's
    Limitations section).

    A single `Linear(input_size, 4*hidden_size)` "combined gate" projection
    (computing all four gates' pre-activations in one matmul) would be more
    efficient than four separate `Linear`s per branch, but Forge's `Tensor`
    has no slicing/indexing primitive to split that combined output back
    into four `(batch, hidden_size)` chunks (confirmed absent by direct
    inspection: no `__getitem__`/`split`/`chunk` in `forge/tensor/tensor.py`
    as of this milestone) -- adding one solely to enable this micro-fusion,
    with no other consumer, would violate the "no dead/speculative
    infrastructure" rule this codebase holds itself to. Eight small
    `Linear`s, composed the same way `RNNCell` already composes two, is the
    smallest change that needs no new primitive at all.

    The forget-gate bias `bf` is initialized to `+1.0` on top of `Linear`'s
    ordinary `Uniform(-1/sqrt(in), 1/sqrt(in))` draw (Jozefowicz et al. 2015,
    "An Empirical Exploration of Recurrent Network Architectures") so the
    cell starts in a "mostly remember" regime rather than "mostly forget"
    -- without this, `sigmoid(~0) = 0.5` erases roughly half the cell state
    every step from the very first gradient update, which matters
    disproportionately for exactly the long-range task this cell exists to
    solve.
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
        param_dtype = dtype if dtype is not None else DEFAULT_DTYPE

        def make_gate(forget_bias_offset: float = 0.0) -> "tuple[Linear, Linear]":
            i2h = Linear(input_size, hidden_size, bias=True, dtype=dtype, device=device, generator=generator)
            h2h = Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device, generator=generator)
            if forget_bias_offset:
                bias_np = i2h.bias.numpy() if i2h.bias.device.type == "cpu" else i2h.bias.to("cpu").numpy()
                i2h.bias = Parameter(bias_np + forget_bias_offset, dtype=param_dtype, device=device)
            return i2h, h2h

        self.i2h_i, self.h2h_i = make_gate()
        self.i2h_f, self.h2h_f = make_gate(forget_bias_offset=1.0)
        self.i2h_g, self.h2h_g = make_gate()
        self.i2h_o, self.h2h_o = make_gate()

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> "tuple[Tensor, Tensor]":
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ShapeMismatchError(
                f"LSTMCell(input_size={self.input_size}) expects x of shape "
                f"(batch, {self.input_size}), got shape {x.shape}."
            )
        if h.ndim != 2 or h.shape[-1] != self.hidden_size:
            raise ShapeMismatchError(
                f"LSTMCell(hidden_size={self.hidden_size}) expects h of shape "
                f"(batch, {self.hidden_size}), got shape {h.shape}."
            )
        if c.ndim != 2 or c.shape[-1] != self.hidden_size:
            raise ShapeMismatchError(
                f"LSTMCell(hidden_size={self.hidden_size}) expects c of shape "
                f"(batch, {self.hidden_size}), got shape {c.shape}."
            )
        if not (x.shape[0] == h.shape[0] == c.shape[0]):
            raise ShapeMismatchError(
                f"LSTMCell got mismatched batch sizes: x has {x.shape[0]}, "
                f"h has {h.shape[0]}, c has {c.shape[0]}."
            )

        i_gate = (self.i2h_i(x) + self.h2h_i(h)).sigmoid()
        f_gate = (self.i2h_f(x) + self.h2h_f(h)).sigmoid()
        g_gate = (self.i2h_g(x) + self.h2h_g(h)).tanh()
        o_gate = (self.i2h_o(x) + self.h2h_o(h)).sigmoid()

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * c_new.tanh()
        return h_new, c_new

    def init_hidden(self, batch_size: int, dtype: Any = None, device: str = "cpu") -> "tuple[Tensor, Tensor]":
        """A fresh all-zero `(h0, c0)` pair, each of shape `(batch_size, hidden_size)`."""
        param_dtype = dtype if dtype is not None else self.i2h_i.weight.dtype
        h0 = Tensor(np.zeros((batch_size, self.hidden_size)), dtype=param_dtype, device=device)
        c0 = Tensor(np.zeros((batch_size, self.hidden_size)), dtype=param_dtype, device=device)
        return h0, c0

    def __repr__(self) -> str:
        return f"LSTMCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


__all__ = ["RNNCell", "LSTMCell"]
