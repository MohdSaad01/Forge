"""CUDA `nn.RNNCell` (Milestone 55): the fused-kernel path (`Tensor.rnn_cell()`,
`CUDABackend.rnn_cell`/`rnn_cell_backward`), parity with CPU.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise, matching `tests/test_batchnorm_cuda.py`'s/`tests/
test_cuda_embedding.py`'s convention. See `docs/development/
m55-post-m54-assessment.md` for the full design writeup (why a fused
primitive was added: a real char-RNN/word-RNN training step calls
`RNNCell.forward()` once per timestep, and composing it from `Linear`/`+`/
`.tanh()` costs ~4 forward + ~8 backward CUDA kernel launches per call --
deep in the launch-overhead-bound regime at Forge's actual RNN shapes).
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.exceptions import ShapeMismatchError, UnsupportedDeviceError
from forge.nn import Parameter, RNNCell

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-5)


def _make_pair(batch=4, input_size=5, hidden_size=7, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    w_ih = rng.standard_normal((input_size, hidden_size)).astype(dtype)
    b_ih = rng.standard_normal((hidden_size,)).astype(dtype)
    w_hh = rng.standard_normal((hidden_size, hidden_size)).astype(dtype)

    cpu = RNNCell(input_size, hidden_size, dtype=dtype)
    cpu.i2h.weight = Parameter(w_ih.copy(), dtype=dtype)
    cpu.i2h.bias = Parameter(b_ih.copy(), dtype=dtype)
    cpu.h2h.weight = Parameter(w_hh.copy(), dtype=dtype)

    cuda = RNNCell(input_size, hidden_size, dtype=dtype, device="cuda")
    cuda.i2h.weight = Parameter(w_ih.copy(), dtype=dtype, device="cuda")
    cuda.i2h.bias = Parameter(b_ih.copy(), dtype=dtype, device="cuda")
    cuda.h2h.weight = Parameter(w_hh.copy(), dtype=dtype, device="cuda")
    return cpu, cuda


# -- Tensor.rnn_cell() device/shape validation --------------------------------


def test_rnn_cell_tensor_method_rejects_cpu_input():
    x = Tensor(np.zeros((2, 3), dtype=np.float32))
    w_ih = Tensor(np.zeros((3, 4), dtype=np.float32))
    b_ih = Tensor(np.zeros((4,), dtype=np.float32))
    w_hh = Tensor(np.zeros((4, 4), dtype=np.float32))
    h = Tensor(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(UnsupportedDeviceError):
        x.rnn_cell(w_ih, b_ih, w_hh, h)


def test_rnn_cell_rejects_1d_x():
    x = Tensor(np.zeros((3,), dtype=np.float32), device="cuda")
    w_ih = Tensor(np.zeros((3, 4), dtype=np.float32), device="cuda")
    b_ih = Tensor(np.zeros((4,), dtype=np.float32), device="cuda")
    w_hh = Tensor(np.zeros((4, 4), dtype=np.float32), device="cuda")
    h = Tensor(np.zeros((1, 4), dtype=np.float32), device="cuda")
    with pytest.raises(ShapeMismatchError):
        x.rnn_cell(w_ih, b_ih, w_hh, h)


@pytest.mark.parametrize(
    "bad_shapes",
    [
        dict(w_ih=(5, 4)),  # wrong input_size dimension
        dict(b_ih=(5,)),  # wrong hidden_size
        dict(w_hh=(4, 5)),  # not square
        dict(h=(3, 4)),  # wrong batch size
    ],
)
def test_rnn_cell_rejects_mismatched_operand_shapes(bad_shapes):
    batch, input_size, hidden_size = 2, 3, 4
    shapes = dict(x=(batch, input_size), w_ih=(input_size, hidden_size), b_ih=(hidden_size,),
                  w_hh=(hidden_size, hidden_size), h=(batch, hidden_size))
    shapes.update(bad_shapes)
    x = Tensor(np.zeros(shapes["x"], dtype=np.float32), device="cuda")
    w_ih = Tensor(np.zeros(shapes["w_ih"], dtype=np.float32), device="cuda")
    b_ih = Tensor(np.zeros(shapes["b_ih"], dtype=np.float32), device="cuda")
    w_hh = Tensor(np.zeros(shapes["w_hh"], dtype=np.float32), device="cuda")
    h = Tensor(np.zeros(shapes["h"], dtype=np.float32), device="cuda")
    with pytest.raises(ShapeMismatchError):
        x.rnn_cell(w_ih, b_ih, w_hh, h)


def test_rnn_cell_rejects_device_mismatch():
    x = Tensor(np.zeros((2, 3), dtype=np.float32), device="cuda")
    w_ih = Tensor(np.zeros((3, 4), dtype=np.float32))  # CPU
    b_ih = Tensor(np.zeros((4,), dtype=np.float32), device="cuda")
    w_hh = Tensor(np.zeros((4, 4), dtype=np.float32), device="cuda")
    h = Tensor(np.zeros((2, 4), dtype=np.float32), device="cuda")
    with pytest.raises(UnsupportedDeviceError):
        x.rnn_cell(w_ih, b_ih, w_hh, h)


# -- Forward/backward parity with CPU -----------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rnn_cell_forward_matches_cpu(dtype):
    batch, input_size, hidden_size = 6, 5, 8
    cpu, cuda = _make_pair(batch, input_size, hidden_size, dtype=dtype)
    rng = np.random.default_rng(2)
    x_arr = rng.standard_normal((batch, input_size)).astype(dtype)
    h_arr = rng.standard_normal((batch, hidden_size)).astype(dtype)

    out_cpu = cpu(Tensor(x_arr, dtype=dtype), Tensor(h_arr, dtype=dtype))
    out_cuda = cuda(Tensor(x_arr, dtype=dtype, device="cuda"), Tensor(h_arr, dtype=dtype, device="cuda"))
    assert out_cuda.device.type == "cuda"
    tol = TOL if dtype == np.float32 else dict(rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(out_cuda.to("cpu").numpy(), out_cpu.numpy(), **tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rnn_cell_backward_matches_cpu_for_every_gradient(dtype):
    batch, input_size, hidden_size = 6, 5, 8
    cpu, cuda = _make_pair(batch, input_size, hidden_size, dtype=dtype)
    rng = np.random.default_rng(3)
    x_arr = rng.standard_normal((batch, input_size)).astype(dtype)
    h_arr = rng.standard_normal((batch, hidden_size)).astype(dtype)
    grad_arr = rng.standard_normal((batch, hidden_size)).astype(dtype)

    x_cpu = Tensor(x_arr.copy(), dtype=dtype, requires_grad=True)
    h_cpu = Tensor(h_arr.copy(), dtype=dtype, requires_grad=True)
    out_cpu = cpu(x_cpu, h_cpu)
    out_cpu.backward(Tensor(grad_arr.copy(), dtype=dtype))

    x_cuda = Tensor(x_arr.copy(), dtype=dtype, requires_grad=True, device="cuda")
    h_cuda = Tensor(h_arr.copy(), dtype=dtype, requires_grad=True, device="cuda")
    out_cuda = cuda(x_cuda, h_cuda)
    out_cuda.backward(Tensor(grad_arr.copy(), dtype=dtype, device="cuda"))

    tol = TOL if dtype == np.float32 else dict(rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **tol)
    np.testing.assert_allclose(h_cuda.grad.to("cpu").numpy(), h_cpu.grad.numpy(), **tol)
    np.testing.assert_allclose(cuda.i2h.weight.grad.to("cpu").numpy(), cpu.i2h.weight.grad.numpy(), **tol)
    np.testing.assert_allclose(cuda.i2h.bias.grad.to("cpu").numpy(), cpu.i2h.bias.grad.numpy(), **tol)
    np.testing.assert_allclose(cuda.h2h.weight.grad.to("cpu").numpy(), cpu.h2h.weight.grad.numpy(), **tol)


def test_rnn_cell_backward_accumulates_across_repeated_timesteps():
    """A shared RNNCell instance used at every timestep of an unrolled sequence
    must accumulate `i2h`/`h2h` gradients across every timestep's contribution
    -- the exact weight-sharing pattern `examples/char_rnn`/`examples/
    word_rnn`'s training loop relies on (M50's own verification, re-confirmed
    here for the fused CUDA path specifically)."""
    batch, input_size, hidden_size, seq_len = 3, 4, 5, 4
    cpu, cuda = _make_pair(batch, input_size, hidden_size, seed=7)
    rng = np.random.default_rng(8)
    xs = [rng.standard_normal((batch, input_size)).astype(np.float32) for _ in range(seq_len)]

    def run(cell, device):
        h = cell.init_hidden(batch, device=device)
        for x_arr in xs:
            x = Tensor(x_arr, device=device)
            h = cell(x, h)
        h.backward(Tensor(np.ones((batch, hidden_size), dtype=np.float32), device=device))

    run(cpu, "cpu")
    run(cuda, "cuda")

    np.testing.assert_allclose(cuda.i2h.weight.grad.to("cpu").numpy(), cpu.i2h.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda.i2h.bias.grad.to("cpu").numpy(), cpu.i2h.bias.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda.h2h.weight.grad.to("cpu").numpy(), cpu.h2h.weight.grad.numpy(), **TOL)


def test_rnn_cell_module_dispatches_to_fused_primitive_on_cuda():
    """Structural check: `nn.RNNCell.forward()` on a CUDA tensor must call
    `Tensor.rnn_cell()` (one CUDA call), never the composed `Linear`/`+`/
    `.tanh()` path -- mirrors `test_cuda_embedding.py`'s own
    zero-CPUBackend-calls structural check."""
    cell = RNNCell(4, 5, device="cuda")
    x = Tensor(np.zeros((2, 4), dtype=np.float32), device="cuda")
    h = cell.init_hidden(2, device="cuda")

    calls = []
    original = Tensor.rnn_cell

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    Tensor.rnn_cell = spy
    try:
        cell(x, h)
    finally:
        Tensor.rnn_cell = original
    assert calls == [1]


def test_rnn_cell_repeated_forward_backward_has_stable_memory():
    """200 repeated forward/backward iterations should settle at a fixed
    `allocated_bytes` (steady-state cache reuse), not grow per-iteration --
    the same convention `tests/test_cuda_embedding.py`'s own memory-safety
    check follows."""
    cell = RNNCell(8, 16, device="cuda")
    x_arr = np.random.default_rng(9).standard_normal((4, 8)).astype(np.float32)
    h0 = cell.init_hidden(4, device="cuda")

    def one_iteration():
        x = Tensor(x_arr, device="cuda", requires_grad=True)
        h_prev = Tensor(h0.to("cpu").numpy(), device="cuda", requires_grad=True)
        out = cell(x, h_prev)
        out.backward(Tensor(np.ones((4, 16), dtype=np.float32), device="cuda"))

    for _ in range(10):
        one_iteration()
    forge.cuda.empty_cache()
    for _ in range(10):
        one_iteration()
    stats_after_warmup = forge.cuda.memory_stats()
    baseline = stats_after_warmup.allocated_bytes

    for _ in range(200):
        one_iteration()
    stats_after = forge.cuda.memory_stats()
    assert stats_after.allocated_bytes == baseline
    assert stats_after.cache_hit_count > stats_after_warmup.cache_hit_count
