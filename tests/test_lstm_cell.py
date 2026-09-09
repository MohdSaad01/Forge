"""`forge.nn.LSTMCell` (Milestone 67): the gated-recurrence step.

Added to close a directly measured gap: `nn.RNNCell`'s vanilla/Elman
recurrence loses its backpropagated gradient exponentially fast over long
sequences (see `docs/development/m67-lstm-long-range-recall.md`).
`LSTMCell` is composed entirely from eight existing `Linear` layers and the
`+`/`*`/`.sigmoid()`/`.tanh()` Tensor primitives, so most of its correctness
already follows from `test_linear.py`/`test_sigmoid.py`/`test_tanh.py` --
this file focuses on what is genuinely new: the gating equations themselves,
multi-timestep unrolling (shared-Parameter weight sharing, mirroring
`test_rnn.py`'s own RNNCell coverage), and persistence registration.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import LSTMCell
from forge.nn.parameter import Parameter

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def fixed_cell(input_size=3, hidden_size=2):
    """An LSTMCell with deterministic, hand-chosen (non-random) parameters."""
    cell = LSTMCell(input_size, hidden_size, dtype="float64")
    rng = np.random.default_rng(0)
    for name in ["i2h_i", "h2h_i", "i2h_f", "h2h_f", "i2h_g", "h2h_g", "i2h_o", "h2h_o"]:
        lin = getattr(cell, name)
        lin.weight = Parameter(rng.standard_normal(lin.weight.shape) * 0.3, dtype="float64")
        if lin.bias is not None:
            lin.bias = Parameter(rng.standard_normal(lin.bias.shape) * 0.1, dtype="float64")
    return cell


def manual_forward(cell, x, h, c):
    def gate(i2h, h2h, activation):
        z = x @ i2h.weight.numpy() + i2h.bias.numpy() + h @ h2h.weight.numpy()
        return activation(z)

    i = gate(cell.i2h_i, cell.h2h_i, _sigmoid_np)
    f = gate(cell.i2h_f, cell.h2h_f, _sigmoid_np)
    g = gate(cell.i2h_g, cell.h2h_g, np.tanh)
    o = gate(cell.i2h_o, cell.h2h_o, _sigmoid_np)
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


# -- shapes / construction ----------------------------------------------------


def test_lstm_cell_parameter_shapes():
    cell = LSTMCell(4, 5)
    for name in ["i2h_i", "i2h_f", "i2h_g", "i2h_o"]:
        lin = getattr(cell, name)
        assert lin.weight.shape == (4, 5)
        assert lin.bias.shape == (5,)
    for name in ["h2h_i", "h2h_f", "h2h_g", "h2h_o"]:
        lin = getattr(cell, name)
        assert lin.weight.shape == (5, 5)
        assert lin.bias is None  # redundant with i2h's bias, same reasoning as RNNCell


def test_lstm_cell_output_shapes():
    cell = LSTMCell(4, 6)
    x = Tensor(np.zeros((3, 4)))
    h0, c0 = cell.init_hidden(3)
    h1, c1 = cell(x, h0, c0)
    assert h1.shape == (3, 6)
    assert c1.shape == (3, 6)


def test_lstm_cell_init_hidden_is_all_zero():
    cell = LSTMCell(4, 6)
    h0, c0 = cell.init_hidden(5)
    np.testing.assert_allclose(h0.numpy(), np.zeros((5, 6)))
    np.testing.assert_allclose(c0.numpy(), np.zeros((5, 6)))


def test_lstm_cell_forget_gate_bias_starts_above_default_init():
    """Jozefowicz et al. 2015's forget-gate-bias-`+1` trick -- see `LSTMCell`'s
    docstring for why."""
    cell = LSTMCell(4, 32)
    other_gate_bias = cell.i2h_i.bias.numpy()
    forget_bias = cell.i2h_f.bias.numpy()
    # Both start from the same Uniform(-1/sqrt(4), 1/sqrt(4)) draw range; the
    # forget gate's mean should sit about 1.0 above the input gate's.
    assert forget_bias.mean() - other_gate_bias.mean() > 0.5


@pytest.mark.parametrize("bad_kwargs", [dict(x_batch=3), dict(h_batch=3), dict(c_batch=3)])
def test_lstm_cell_rejects_mismatched_batch_sizes(bad_kwargs):
    cell = LSTMCell(4, 6)
    batch = dict(x_batch=2, h_batch=2, c_batch=2)
    batch.update(bad_kwargs)
    x = Tensor(np.zeros((batch["x_batch"], 4)))
    h = Tensor(np.zeros((batch["h_batch"], 6)))
    c = Tensor(np.zeros((batch["c_batch"], 6)))
    with pytest.raises(ShapeMismatchError):
        cell(x, h, c)


def test_lstm_cell_rejects_wrong_input_feature_dim():
    cell = LSTMCell(4, 6)
    x = Tensor(np.zeros((2, 3)))
    h0, c0 = cell.init_hidden(2)
    with pytest.raises(ShapeMismatchError):
        cell(x, h0, c0)


def test_lstm_cell_rejects_wrong_hidden_dim():
    cell = LSTMCell(4, 6)
    x = Tensor(np.zeros((2, 4)))
    h0 = Tensor(np.zeros((2, 5)))
    c0 = Tensor(np.zeros((2, 6)))
    with pytest.raises(ShapeMismatchError):
        cell(x, h0, c0)


# -- forward correctness -------------------------------------------------------


def test_lstm_cell_forward_matches_manual_gate_computation():
    cell = fixed_cell(3, 2)
    x_data = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    h_data = np.array([[0.1, -0.2], [0.0, 0.3]])
    c_data = np.array([[0.05, 0.1], [-0.05, 0.2]])

    h1, c1 = cell(Tensor(x_data, dtype="float64"), Tensor(h_data, dtype="float64"), Tensor(c_data, dtype="float64"))
    expected_h, expected_c = manual_forward(cell, x_data, h_data, c_data)
    np.testing.assert_allclose(h1.numpy(), expected_h, **TOL)
    np.testing.assert_allclose(c1.numpy(), expected_c, **TOL)


def test_lstm_cell_zero_forget_and_input_gate_retains_nothing_or_everything():
    """Sanity/mechanism check: driving the forget gate fully open (bias `+inf`-like)
    and the input gate fully closed should leave `c` unchanged from its input."""
    cell = LSTMCell(2, 3, dtype="float64")
    for name in ["i2h_i", "h2h_i", "i2h_f", "h2h_f", "i2h_g", "h2h_g", "i2h_o", "h2h_o"]:
        lin = getattr(cell, name)
        lin.weight = Parameter(np.zeros_like(lin.weight.numpy()), dtype="float64")
        if lin.bias is not None:
            lin.bias = Parameter(np.zeros_like(lin.bias.numpy()), dtype="float64")
    cell.i2h_f.bias = Parameter(np.full(3, 30.0), dtype="float64")  # forget gate -> sigmoid(30) ~= 1
    cell.i2h_i.bias = Parameter(np.full(3, -30.0), dtype="float64")  # input gate -> sigmoid(-30) ~= 0
    cell.i2h_o.bias = Parameter(np.full(3, 30.0), dtype="float64")  # output gate open

    x = Tensor(np.ones((1, 2)), dtype="float64")
    c0 = Tensor(np.array([[0.3, -0.4, 0.7]]), dtype="float64")
    h0 = Tensor(np.zeros((1, 3)), dtype="float64")
    h1, c1 = cell(x, h0, c0)
    np.testing.assert_allclose(c1.numpy(), c0.numpy(), atol=1e-8)
    np.testing.assert_allclose(h1.numpy(), np.tanh(c0.numpy()), atol=1e-8)


# -- multi-timestep unrolling (the property real sequence models need) --------


def test_lstm_cell_reused_across_timesteps_accumulates_weight_gradients():
    cell = fixed_cell(2, 2)
    batch = 2
    seq_len = 4
    rng = np.random.default_rng(0)
    inputs = [Tensor(rng.standard_normal((batch, 2)), dtype="float64") for _ in range(seq_len)]

    h, c = cell.init_hidden(batch, dtype="float64")
    for x_t in inputs:
        h, c = cell(x_t, h, c)
    loss = h.sum() + c.sum()
    loss.backward()

    for name in ["i2h_i", "h2h_i", "i2h_f", "h2h_f", "i2h_g", "h2h_g", "i2h_o", "h2h_o"]:
        lin = getattr(cell, name)
        assert lin.weight.grad is not None
        assert lin.weight.grad.shape == lin.weight.shape
        if lin.bias is not None:
            assert lin.bias.grad is not None
    assert np.any(cell.h2h_f.weight.grad.numpy() != 0.0)


def test_lstm_cell_unrolled_weight_gradient_matches_finite_difference():
    """Numerically verify the *shared* input-gate weight's gradient over a 3-step unroll."""
    eps = 1e-5
    hidden_size = 2
    input_size = 2
    batch = 1
    seq_len = 3
    rng = np.random.default_rng(1)
    x_data = [rng.standard_normal((batch, input_size)) for _ in range(seq_len)]

    def build(w_i2h_i):
        cell = fixed_cell(input_size, hidden_size)
        cell.i2h_i.weight = Parameter(w_i2h_i, dtype="float64")
        return cell

    w0 = fixed_cell(input_size, hidden_size).i2h_i.weight.numpy().copy()

    def forward(w_i2h_i):
        cell = build(w_i2h_i)
        h, c = cell.init_hidden(batch, dtype="float64")
        for x_t in x_data:
            h, c = cell(Tensor(x_t, dtype="float64"), h, c)
        return float((h.sum() + c.sum()).numpy())

    numeric = np.empty_like(w0)
    it = np.nditer(w0, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus, minus = w0.copy(), w0.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric[idx] = (forward(plus) - forward(minus)) / (2 * eps)

    cell = build(w0)
    h, c = cell.init_hidden(batch, dtype="float64")
    for x_t in x_data:
        h, c = cell(Tensor(x_t, dtype="float64"), h, c)
    (h.sum() + c.sum()).backward()

    np.testing.assert_allclose(cell.i2h_i.weight.grad.numpy(), numeric, **FD_TOL)


# -- persistence ----------------------------------------------------------------


def test_lstm_cell_is_registered_for_persistence():
    from forge.serialization.registry import spec_for_class

    spec = spec_for_class(LSTMCell)
    cell = LSTMCell(4, 6)
    config = spec.get_config(cell)
    rebuilt = spec.from_config(config)
    assert rebuilt.input_size == 4
    assert rebuilt.hidden_size == 6
