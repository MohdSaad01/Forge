"""Milestone 32 tests: optimized `k_conv2d_backward_input` (CUDA Conv2d backward).

M32 profiled the real `conv2d_backward` CUDA kernels
(`benchmarks/conv2d_backward_profile.py`) and found `dInput`
(`cf_conv2d_backward_input_*`) doing `Cout`-fold redundant integer
division/modulo work per thread -- the `(kh, ho)`/`(kw, wo)` validity
resolution in `k_conv2d_backward_input` (`kernels.cu`) never depended on the
`co` loop it sat inside. The fix hoists that resolution into two small
per-thread local tables computed once, leaving the `co` loop free of
division. See `docs/architecture/cuda-backend.md`'s **CUDA Conv2d backward:
input optimization (Milestone 32)** section for the full writeup.

`tests/test_cuda_conv.py` already covers Conv2d forward/backward vs. CPU
across stride/padding/kernel-shape combinations (unmodified, still passing
after this change -- the rewritten kernel computes the exact same
`(co, kh, ho, kw, wo)` contributions as before, just without recomputing
`co`-independent work `Cout` times) and an input-gradient finite-difference
check. This module covers what M32 specifically needs: weight/bias finite
differences (not previously covered), explicit-stream and cross-stream
correctness for the *backward* pass specifically (only forward was covered
on an explicit stream before), and repeated-use memory safety -- the same
angles `tests/test_cuda_cross_entropy_fusion.py` covered for M31's fusion.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = fn(x)
        x[idx] = orig - eps
        minus = fn(x)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


# -- Finite-difference: weight and bias (input already covered by test_cuda_conv.py) --


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_cuda_conv2d_finite_difference_weight(stride, padding):
    forge.random.seed(4)
    layer = Conv2d(2, 3, kernel_size=3, stride=stride, padding=padding)
    x_data = np.random.default_rng(20).standard_normal((2, 2, 7, 7)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    def loss(wd):
        out = Tensor(x_data).conv2d(Tensor(wd), Tensor(b_data), layer.stride, layer.padding)
        return float((out.numpy() ** 2).sum())

    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda", requires_grad=True)
    b_cuda = Tensor(b_data.copy(), device="cuda")
    out = x_cuda.conv2d(w_cuda, b_cuda, layer.stride, layer.padding)
    (out * out).sum().backward()

    np.testing.assert_allclose(
        w_cuda.grad.to("cpu").numpy(), numerical_grad(loss, w_data.copy()), rtol=1e-2, atol=1e-2
    )


def test_cuda_conv2d_finite_difference_bias():
    forge.random.seed(5)
    layer = Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(21).standard_normal((2, 2, 6, 6)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    def loss(bd):
        out = Tensor(x_data).conv2d(Tensor(w_data), Tensor(bd), layer.stride, layer.padding)
        return float((out.numpy() ** 2).sum())

    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    b_cuda = Tensor(b_data.copy(), device="cuda", requires_grad=True)
    out = x_cuda.conv2d(w_cuda, b_cuda, layer.stride, layer.padding)
    (out * out).sum().backward()

    np.testing.assert_allclose(
        b_cuda.grad.to("cpu").numpy(), numerical_grad(loss, b_data.copy()), rtol=1e-2, atol=1e-2
    )


# -- Explicit-stream (async mode) correctness for the backward pass ----------


def test_cuda_conv2d_backward_on_explicit_stream_matches_cpu():
    rng = np.random.default_rng(30)
    x_np = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)

    forge.random.seed(6)
    cpu_layer = Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
    x_cpu = Tensor(x_np.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    forge.random.seed(6)
    cuda_layer = Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_np.copy(), device="cuda", requires_grad=True)
        cuda_layer(x_cuda).sum().backward()
    s.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)
    assert isinstance(x_cuda.grad._data, CUDAStorage)


# -- Cross-stream correctness (producer stream != conv2d_backward's stream) --


def test_cuda_conv2d_backward_correct_when_inputs_from_different_streams():
    """x, weight, and the upstream gradient each produced on a distinct stream."""
    stream_x = forge.cuda.Stream()
    stream_w = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    rng = np.random.default_rng(31)
    x_data = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)

    forge.random.seed(7)
    cpu_layer = Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    forge.random.seed(7)
    layer = Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    with forge.cuda.stream(stream_x):
        x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    with forge.cuda.stream(stream_w):
        w_cuda = Tensor(w_data.copy(), device="cuda", requires_grad=True)
        b_cuda = Tensor(b_data.copy(), device="cuda", requires_grad=True)

    with forge.cuda.stream(stream_compute):
        out = x_cuda.conv2d(w_cuda, b_cuda, (1, 1), (1, 1))
        out.sum().backward()
    stream_compute.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(w_cuda.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(b_cuda.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)


def test_cuda_conv2d_backward_correct_when_grad_output_from_different_stream():
    """The upstream gradient reaching conv2d_backward was itself last produced on another stream."""
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    rng = np.random.default_rng(32)
    x_data = rng.standard_normal((2, 2, 6, 6)).astype(np.float32)

    forge.random.seed(8)
    cpu_layer = Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    y_cpu = cpu_layer(x_cpu)
    (y_cpu + y_cpu).sum().backward()

    forge.random.seed(8)
    cuda_layer = Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")

    with forge.cuda.stream(stream_a):
        x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
        out = cuda_layer(x_cuda)
    with forge.cuda.stream(stream_b):
        # CUDA has no scalar-broadcast `mul` (see `_elementwise`'s exact-shape
        # restriction) -- an exact-shape `add` produces the same "doubled"
        # upstream gradient while still exercising a real cross-stream
        # producer for conv2d_backward's `grad_output`.
        scaled = out + out
    with forge.cuda.stream(stream_a):
        scaled.sum().backward()
    stream_a.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


# -- Resource lifetime / memory safety ----------------------------------------


def test_cuda_conv2d_backward_repeated_use_does_not_grow_active_memory():
    x_data = np.random.default_rng(33).standard_normal((4, 3, 10, 10)).astype(np.float32)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    layer = Conv2d(3, 6, kernel_size=3, stride=1, padding=1).to("cuda")
    for _ in range(30):
        x = Tensor(x_data.copy(), device="cuda", requires_grad=True)
        layer.weight.zero_grad()
        layer.bias.zero_grad()
        out = layer(x)
        out.sum().backward()

    del x, out, layer
    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0
