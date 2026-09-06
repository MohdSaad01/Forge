"""`forge.nn.RNNCell` (Milestone 50): the vanilla-RNN recurrence step.

Added as the one genuine blocker discovered while attempting a small
character-level RNN language model with Forge as it existed after M49 --
see `docs/development/m50-char-rnn.md`. `RNNCell` is composed entirely from
two existing `Linear` layers and the new `Tensor.tanh()` primitive, so most
of its correctness already follows from `test_linear.py`/`test_tanh.py`;
this file focuses on what is genuinely new: multi-timestep unrolling with
the *same* cell instance (Parameter weight-sharing across an autograd
graph), which is exactly what `examples/char_rnn/train.py` does.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import RNNCell

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)


def fixed_cell(input_size=3, hidden_size=2):
    """An RNNCell with deterministic, hand-chosen (non-random) parameters."""
    cell = RNNCell(input_size, hidden_size)
    w_ih = np.arange(input_size * hidden_size, dtype=np.float64).reshape(input_size, hidden_size) * 0.1
    cell.i2h.weight = type(cell.i2h.weight)(w_ih)
    cell.i2h.bias = type(cell.i2h.bias)(np.arange(hidden_size, dtype=np.float64) * 0.1 + 0.5)
    w_hh = np.arange(hidden_size * hidden_size, dtype=np.float64).reshape(hidden_size, hidden_size) * 0.1
    cell.h2h.weight = type(cell.h2h.weight)(w_hh)
    return cell


# -- shapes / construction ---------------------------------------------------


def test_rnn_cell_parameter_shapes():
    cell = RNNCell(4, 5)
    assert cell.i2h.weight.shape == (4, 5)
    assert cell.i2h.bias.shape == (5,)
    assert cell.h2h.weight.shape == (5, 5)
    assert cell.h2h.bias is None  # single bias term, on i2h only


def test_rnn_cell_output_shape():
    cell = RNNCell(4, 6)
    x = Tensor(np.zeros((3, 4)))
    h0 = cell.init_hidden(3)
    h1 = cell(x, h0)
    assert h1.shape == (3, 6)


def test_rnn_cell_init_hidden_is_all_zero():
    cell = RNNCell(4, 6)
    h0 = cell.init_hidden(5)
    np.testing.assert_allclose(h0.numpy(), np.zeros((5, 6)))


def test_rnn_cell_rejects_wrong_input_feature_dim():
    cell = RNNCell(4, 6)
    x = Tensor(np.zeros((2, 3)))
    h0 = cell.init_hidden(2)
    with pytest.raises(ShapeMismatchError):
        cell(x, h0)


def test_rnn_cell_rejects_wrong_hidden_dim():
    cell = RNNCell(4, 6)
    x = Tensor(np.zeros((2, 4)))
    h0 = Tensor(np.zeros((2, 5)))
    with pytest.raises(ShapeMismatchError):
        cell(x, h0)


def test_rnn_cell_rejects_mismatched_batch_sizes():
    cell = RNNCell(4, 6)
    x = Tensor(np.zeros((2, 4)))
    h0 = Tensor(np.zeros((3, 6)))
    with pytest.raises(ShapeMismatchError):
        cell(x, h0)


# -- forward correctness ------------------------------------------------------


def test_rnn_cell_forward_matches_manual_computation():
    cell = fixed_cell(3, 2)
    x_data = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    h_data = np.array([[0.1, -0.2], [0.0, 0.3]])
    h1 = cell(Tensor(x_data), Tensor(h_data))

    expected = np.tanh(
        x_data @ cell.i2h.weight.numpy() + cell.i2h.bias.numpy() + h_data @ cell.h2h.weight.numpy()
    )
    np.testing.assert_allclose(h1.numpy(), expected, **TOL)


def test_rnn_cell_zero_input_and_hidden_gives_tanh_of_bias():
    cell = fixed_cell(3, 2)
    x = Tensor(np.zeros((1, 3)))
    h0 = cell.init_hidden(1)
    h1 = cell(x, h0)
    np.testing.assert_allclose(h1.numpy(), np.tanh(cell.i2h.bias.numpy())[None, :], **TOL)


# -- multi-timestep unrolling (the property this model actually needs) -------


def test_rnn_cell_reused_across_timesteps_accumulates_weight_gradients():
    """The same cell (and therefore the same Parameters) used at every timestep.

    This is exactly `examples/char_rnn/train.py`'s unroll loop. Verified
    directly before this milestone wrote any new code
    (`forge/autograd/engine.py`'s reverse-topological-order accumulation
    already handles a leaf reused as input to multiple graph nodes) -- this
    test locks that property in against regression.
    """
    cell = fixed_cell(2, 2)
    batch = 2
    seq_len = 4
    rng = np.random.default_rng(0)
    inputs = [Tensor(rng.standard_normal((batch, 2))) for _ in range(seq_len)]

    h = cell.init_hidden(batch, dtype="float64")
    for x_t in inputs:
        h = cell(x_t, h)
    loss = h.sum()
    loss.backward()

    assert cell.i2h.weight.grad is not None
    assert cell.i2h.bias.grad is not None
    assert cell.h2h.weight.grad is not None
    assert cell.i2h.weight.grad.shape == cell.i2h.weight.shape
    assert cell.h2h.weight.grad.shape == cell.h2h.weight.shape
    # A gradient contribution came from more than one timestep: the gradient
    # is not what a *single*-timestep backward through the same op would
    # produce (a weak but cheap "accumulation actually happened" check).
    assert np.any(cell.h2h.weight.grad.numpy() != 0.0)


def test_rnn_cell_unrolled_weight_gradient_matches_finite_difference():
    """Numerically verify the *shared* i2h weight's gradient over a 3-step unroll."""
    eps = 1e-5
    hidden_size = 2
    input_size = 2
    batch = 1
    seq_len = 3
    rng = np.random.default_rng(1)
    x_data = [rng.standard_normal((batch, input_size)) for _ in range(seq_len)]
    w_ih0 = rng.standard_normal((input_size, hidden_size)) * 0.3
    b_ih0 = rng.standard_normal((hidden_size,)) * 0.1
    w_hh0 = rng.standard_normal((hidden_size, hidden_size)) * 0.3

    def forward(w_ih: np.ndarray) -> float:
        cell = RNNCell(input_size, hidden_size, dtype="float64")
        cell.i2h.weight = type(cell.i2h.weight)(w_ih)
        cell.i2h.bias = type(cell.i2h.bias)(b_ih0)
        cell.h2h.weight = type(cell.h2h.weight)(w_hh0)
        h = cell.init_hidden(batch, dtype="float64")
        for x_t in x_data:
            h = cell(Tensor(x_t, dtype="float64"), h)
        return float(h.numpy().sum())

    numeric = np.empty_like(w_ih0)
    it = np.nditer(w_ih0, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus, minus = w_ih0.copy(), w_ih0.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric[idx] = (forward(plus) - forward(minus)) / (2 * eps)

    cell = RNNCell(input_size, hidden_size, dtype="float64")
    cell.i2h.weight = type(cell.i2h.weight)(w_ih0)
    cell.i2h.bias = type(cell.i2h.bias)(b_ih0)
    cell.h2h.weight = type(cell.h2h.weight)(w_hh0)
    h = cell.init_hidden(batch, dtype="float64")
    for x_t in x_data:
        h = cell(Tensor(x_t, dtype="float64"), h)
    h.sum().backward()

    np.testing.assert_allclose(cell.i2h.weight.grad.numpy(), numeric, **FD_TOL)


def test_rnn_cell_is_registered_for_persistence():
    from forge.serialization.registry import spec_for_class

    spec = spec_for_class(RNNCell)
    cell = RNNCell(4, 6)
    config = spec.get_config(cell)
    rebuilt = spec.from_config(config)
    assert rebuilt.input_size == 4
    assert rebuilt.hidden_size == 6
