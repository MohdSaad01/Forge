"""`forge.nn.RNNCell` on CUDA (Milestone 50).

Skipped cleanly without a working CUDA backend via the module-level
`pytestmark`, matching every other `tests/test_cuda_*.py` file. `RNNCell`
adds no CUDA-specific code of its own -- it composes `Linear` (CUDA-capable
since Milestone 9) and `Tensor.tanh()` (this milestone's new CUDA kernel,
see `test_cuda_backend.py`/`test_cuda_consistency.py`/`test_cuda_autograd.py`
for the primitive-level coverage) -- so these tests exercise the composition
end-to-end via the real `nn.RNNCell` API, including a multi-timestep unroll
(the actual shape `examples/char_rnn/train.py` uses), not just the
underlying primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import RNNCell

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-5, atol=1e-5)


def _matched_cells(input_size=3, hidden_size=4):
    cpu_cell = RNNCell(input_size, hidden_size)
    cuda_cell = RNNCell(input_size, hidden_size)
    cuda_cell.i2h.weight._data = np.array(cpu_cell.i2h.weight._data, copy=True)
    cuda_cell.i2h.bias._data = np.array(cpu_cell.i2h.bias._data, copy=True)
    cuda_cell.h2h.weight._data = np.array(cpu_cell.h2h.weight._data, copy=True)
    cuda_cell.to("cuda")
    return cpu_cell, cuda_cell


def test_rnn_cell_single_step_matches_cpu():
    cpu_cell, cuda_cell = _matched_cells()
    x = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    x_cpu, x_cuda = Tensor(x), Tensor(x).to("cuda")
    h0_cpu = cpu_cell.init_hidden(5)
    h0_cuda = cuda_cell.init_hidden(5, device="cuda")

    h1_cpu = cpu_cell(x_cpu, h0_cpu)
    h1_cuda = cuda_cell(x_cuda, h0_cuda)

    assert h1_cuda.device.type == "cuda"
    assert isinstance(h1_cuda._data, CUDAStorage)
    np.testing.assert_allclose(h1_cuda.to("cpu").numpy(), h1_cpu.numpy(), **TOL)


def test_rnn_cell_multi_step_unroll_matches_cpu():
    """The actual shape `examples/char_rnn/train.py` uses: one cell, several timesteps."""
    cpu_cell, cuda_cell = _matched_cells(input_size=3, hidden_size=4)
    rng = np.random.default_rng(1)
    xs = [rng.standard_normal((2, 3)).astype(np.float32) for _ in range(5)]

    h_cpu = cpu_cell.init_hidden(2)
    h_cuda = cuda_cell.init_hidden(2, device="cuda")
    for x in xs:
        h_cpu = cpu_cell(Tensor(x), h_cpu)
        h_cuda = cuda_cell(Tensor(x).to("cuda"), h_cuda)

    np.testing.assert_allclose(h_cuda.to("cpu").numpy(), h_cpu.numpy(), **TOL)


def test_rnn_cell_unrolled_backward_weight_gradients_match_cpu():
    cpu_cell, cuda_cell = _matched_cells(input_size=2, hidden_size=2)
    rng = np.random.default_rng(2)
    xs = [rng.standard_normal((3, 2)).astype(np.float32) for _ in range(4)]

    h_cpu = cpu_cell.init_hidden(3)
    h_cuda = cuda_cell.init_hidden(3, device="cuda")
    for x in xs:
        h_cpu = cpu_cell(Tensor(x), h_cpu)
        h_cuda = cuda_cell(Tensor(x).to("cuda"), h_cuda)

    h_cpu.sum().backward()
    h_cuda.sum().backward()

    assert cuda_cell.i2h.weight.grad.device.type == "cuda"
    assert cuda_cell.h2h.weight.grad.device.type == "cuda"
    np.testing.assert_allclose(
        cuda_cell.i2h.weight.grad.to("cpu").numpy(), cpu_cell.i2h.weight.grad.numpy(), **TOL
    )
    np.testing.assert_allclose(
        cuda_cell.h2h.weight.grad.to("cpu").numpy(), cpu_cell.h2h.weight.grad.numpy(), **TOL
    )
    np.testing.assert_allclose(
        cuda_cell.i2h.bias.grad.to("cpu").numpy(), cpu_cell.i2h.bias.grad.numpy(), **TOL
    )


def test_rnn_cell_parameters_and_init_hidden_are_cuda_resident():
    cell = RNNCell(4, 6).to("cuda")
    assert cell.i2h.weight.device.type == "cuda"
    assert cell.h2h.weight.device.type == "cuda"
    h0 = cell.init_hidden(2, device="cuda")
    assert h0.device.type == "cuda"
    assert isinstance(h0._data, CUDAStorage)
