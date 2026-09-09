"""`forge.nn.LSTMCell` on CUDA (Milestone 67).

Skipped cleanly without a working CUDA backend via the module-level
`pytestmark`, matching every other `tests/test_cuda_*.py` file. `LSTMCell`
adds no CUDA-specific code of its own -- it composes `Linear` (CUDA-capable
since Milestone 9) and `Tensor.sigmoid()`/`Tensor.tanh()` (elementwise CUDA
kernels; see `test_cuda_backend.py` for primitive-level coverage) -- so these
tests exercise the composition end-to-end via the real `nn.LSTMCell` API,
including a multi-timestep unroll, mirroring `test_rnn_cuda.py`'s own
`RNNCell` coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import LSTMCell

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-5, atol=1e-5)

_GATES = ["i2h_i", "h2h_i", "i2h_f", "h2h_f", "i2h_g", "h2h_g", "i2h_o", "h2h_o"]


def _matched_cells(input_size=3, hidden_size=4):
    cpu_cell = LSTMCell(input_size, hidden_size)
    cuda_cell = LSTMCell(input_size, hidden_size)
    for name in _GATES:
        cpu_lin = getattr(cpu_cell, name)
        cuda_lin = getattr(cuda_cell, name)
        cuda_lin.weight._data = np.array(cpu_lin.weight._data, copy=True)
        if cpu_lin.bias is not None:
            cuda_lin.bias._data = np.array(cpu_lin.bias._data, copy=True)
    cuda_cell.to("cuda")
    return cpu_cell, cuda_cell


def test_lstm_cell_single_step_matches_cpu():
    cpu_cell, cuda_cell = _matched_cells()
    x = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    x_cpu, x_cuda = Tensor(x), Tensor(x).to("cuda")
    h0_cpu, c0_cpu = cpu_cell.init_hidden(5)
    h0_cuda, c0_cuda = cuda_cell.init_hidden(5, device="cuda")

    h1_cpu, c1_cpu = cpu_cell(x_cpu, h0_cpu, c0_cpu)
    h1_cuda, c1_cuda = cuda_cell(x_cuda, h0_cuda, c0_cuda)

    assert h1_cuda.device.type == "cuda"
    assert isinstance(h1_cuda._data, CUDAStorage)
    assert isinstance(c1_cuda._data, CUDAStorage)
    np.testing.assert_allclose(h1_cuda.to("cpu").numpy(), h1_cpu.numpy(), **TOL)
    np.testing.assert_allclose(c1_cuda.to("cpu").numpy(), c1_cpu.numpy(), **TOL)


def test_lstm_cell_multi_step_unroll_matches_cpu():
    """The shape `examples/long_range_recall/train.py` uses: one cell, several timesteps."""
    cpu_cell, cuda_cell = _matched_cells(input_size=3, hidden_size=4)
    rng = np.random.default_rng(1)
    xs = [rng.standard_normal((2, 3)).astype(np.float32) for _ in range(6)]

    h_cpu, c_cpu = cpu_cell.init_hidden(2)
    h_cuda, c_cuda = cuda_cell.init_hidden(2, device="cuda")
    for x in xs:
        h_cpu, c_cpu = cpu_cell(Tensor(x), h_cpu, c_cpu)
        h_cuda, c_cuda = cuda_cell(Tensor(x).to("cuda"), h_cuda, c_cuda)

    np.testing.assert_allclose(h_cuda.to("cpu").numpy(), h_cpu.numpy(), **TOL)
    np.testing.assert_allclose(c_cuda.to("cpu").numpy(), c_cpu.numpy(), **TOL)


def test_lstm_cell_unrolled_backward_weight_gradients_match_cpu():
    cpu_cell, cuda_cell = _matched_cells(input_size=2, hidden_size=2)
    rng = np.random.default_rng(2)
    xs = [rng.standard_normal((3, 2)).astype(np.float32) for _ in range(4)]

    h_cpu, c_cpu = cpu_cell.init_hidden(3)
    h_cuda, c_cuda = cuda_cell.init_hidden(3, device="cuda")
    for x in xs:
        h_cpu, c_cpu = cpu_cell(Tensor(x), h_cpu, c_cpu)
        h_cuda, c_cuda = cuda_cell(Tensor(x).to("cuda"), h_cuda, c_cuda)

    (h_cpu.sum() + c_cpu.sum()).backward()
    (h_cuda.sum() + c_cuda.sum()).backward()

    for name in _GATES:
        cpu_lin = getattr(cpu_cell, name)
        cuda_lin = getattr(cuda_cell, name)
        assert cuda_lin.weight.grad.device.type == "cuda"
        np.testing.assert_allclose(cuda_lin.weight.grad.to("cpu").numpy(), cpu_lin.weight.grad.numpy(), **TOL)
        if cpu_lin.bias is not None:
            np.testing.assert_allclose(cuda_lin.bias.grad.to("cpu").numpy(), cpu_lin.bias.grad.numpy(), **TOL)


def test_lstm_cell_parameters_and_init_hidden_are_cuda_resident():
    cell = LSTMCell(4, 6).to("cuda")
    for name in _GATES:
        assert getattr(cell, name).weight.device.type == "cuda"
    h0, c0 = cell.init_hidden(2, device="cuda")
    assert h0.device.type == "cuda"
    assert c0.device.type == "cuda"
    assert isinstance(h0._data, CUDAStorage)
    assert isinstance(c0._data, CUDAStorage)
