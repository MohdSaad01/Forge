"""Milestone 34 tests: `dWeight` via im2col + the existing tiled GEMM (adopted for large shapes).

M33 rejected cooperative reduction and named im2col + GEMM (reusing the
existing M11 shared-memory-tiled `k_matmul`) as the next structurally
different `dWeight` candidate. `benchmarks/conv2d_backward_weight_im2col_
profile.py` and `benchmarks/conv2d_backward_im2col_pipeline_profile.py`
measured it 1.12-1.59x faster end-to-end at every representative shape with
>= 1,152 weight elements, and slower at the one shape below the existing
256-element threshold (`mnist_conv1`, where the block-reduce path already
wins per M33) -- see `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 34** section for the full report. `CUDABackend.conv2d_backward`
(`backend.py`) now dispatches `dWeight` to `forge.backend.cuda.
experimental_conv_im2col.dweight_im2col_gemm` at/above
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD` (256 weight elements, matching
`kernels.cu`'s existing `CONV2D_WEIGHT_REDUCE_THRESHOLD`) and keeps the
original kernel below it.

This module has two parts: direct correctness coverage for the
`dweight_im2col_gemm` pipeline itself (CPU comparison across shapes, finite
difference, explicit-stream, cross-stream, repeated-use memory safety --
mirroring `tests/test_cuda_conv2d_backward_weight_cooperative.py`'s (M33)
and `tests/test_cuda_conv2d_backward_optimization.py`'s (M32) existing
coverage patterns), and **production dispatch** coverage: correctness
through the ordinary `Tensor.conv2d`/`nn.Conv2d` API at shapes that actually
cross `_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD`, since `tests/test_cuda_conv.py`'s
existing shapes (`Cin`/`Cout` <= 4) never exceed ~144 weight elements and so
never exercised the new dispatch branch before this milestone.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import get_cuda_backend
from forge.backend.cuda.experimental_conv_im2col import dweight_im2col_gemm
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _reference_and_experimental_grad_w(x_data, w_data, b_data, stride, padding, upstream=None):
    """CPU-reference `grad_weight` vs. the experimental CUDA im2col+GEMM path, same inputs."""
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    out_cpu = x_cpu.conv2d(w_cpu, b_cpu, stride, padding)
    if upstream is None:
        upstream = np.random.default_rng(99).standard_normal(out_cpu.shape).astype(x_data.dtype)
    out_cpu.backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_im2col_gemm(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, stride, padding)
    actual = backend.to_numpy(result)
    return expected, actual


# -- Correctness vs. CPU across shapes / stride / padding / kernel size ------

SHAPES = [
    # (Cin, Cout, H, W, K, stride, padding) -- spans the M32/M33 threshold
    # (256 weight elements) and the stride/padding combos M32's own test
    # file covers.
    (2, 3, 6, 6, 3, 1, 0),      # 54 elements, no padding
    (2, 3, 6, 6, 3, 1, 1),      # 54 elements, padded
    (8, 16, 9, 9, 3, 1, 1),     # 1,152 elements -- at/above threshold
    (16, 32, 7, 7, 3, 2, 1),    # 4,608 elements, strided
    (3, 4, 9, 9, 3, 2, 1),      # asymmetric N (see repeated-use test below), strided
    (2, 3, 5, 5, 2, 1, 0),      # even kernel size, odd spatial size
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_im2col_gemm_dweight_matches_cpu(cin, cout, h, w, k, s, p):
    forge.random.seed(50)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(51).standard_normal((3, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected, actual = _reference_and_experimental_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    np.testing.assert_allclose(actual, expected, **TOL)


def test_im2col_gemm_dweight_matches_cpu_float64():
    forge.random.seed(52)
    layer = Conv2d(4, 6, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(53).standard_normal((2, 4, 8, 8)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    expected, actual = _reference_and_experimental_grad_w(x_data, w_data, b_data, (1, 1), (1, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


# -- Finite difference (Section 25 of the milestone brief) -------------------


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


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_im2col_gemm_dweight_finite_difference(stride, padding):
    forge.random.seed(54)
    layer = Conv2d(2, 3, kernel_size=3, stride=stride, padding=padding)
    x_data = np.random.default_rng(55).standard_normal((2, 2, 7, 7)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    def loss(wd):
        out = Tensor(x_data).conv2d(Tensor(wd), Tensor(b_data), (stride, stride), (padding, padding))
        return float((out.numpy() ** 2).sum())

    upstream = 2.0 * Tensor(x_data).conv2d(
        Tensor(w_data), Tensor(b_data), (stride, stride), (padding, padding)
    ).numpy()

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()
    result = dweight_im2col_gemm(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (stride, stride), (padding, padding)
    )
    actual = backend.to_numpy(result)

    np.testing.assert_allclose(actual, numerical_grad(loss, w_data.copy()), rtol=1e-2, atol=1e-2)


# -- Explicit-stream (async mode) correctness ---------------------------------


def test_im2col_gemm_dweight_on_explicit_stream_matches_cpu():
    forge.random.seed(56)
    layer = Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    x_data = np.random.default_rng(57).standard_normal((2, 3, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(58).standard_normal((2, 5, 5, 5)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    x_cpu.conv2d(w_cpu, b_cpu, (2, 2), (1, 1)).backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda")
        w_cuda = Tensor(w_data.copy(), device="cuda")
        grad_out_cuda = Tensor(upstream.copy(), device="cuda")
        result = dweight_im2col_gemm(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (2, 2), (1, 1)
        )
    s.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Cross-stream correctness (producer streams != compute stream) -----------


def test_im2col_gemm_dweight_correct_when_inputs_from_different_streams():
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    forge.random.seed(59)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(60).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(61).standard_normal((2, 4, 8, 8)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    x_cpu.conv2d(w_cpu, b_cpu, (1, 1), (1, 1)).backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    with forge.cuda.stream(stream_x):
        x_cuda = Tensor(x_data.copy(), device="cuda")
        w_cuda = Tensor(w_data.copy(), device="cuda")
    with forge.cuda.stream(stream_g):
        grad_out_cuda = Tensor(upstream.copy(), device="cuda")

    with forge.cuda.stream(stream_compute):
        result = dweight_im2col_gemm(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Memory safety (repeated use) ---------------------------------------------


def test_im2col_gemm_dweight_repeated_use_does_not_grow_active_memory():
    x_data = np.random.default_rng(62).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(63).standard_normal((16, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(64).standard_normal((4, 16, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    gc.disable()
    try:
        for _ in range(20):
            x_cuda = Tensor(x_data.copy(), device="cuda")
            w_cuda = Tensor(w_data.copy(), device="cuda")
            grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
            result = dweight_im2col_gemm(
                backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
            )
            del x_cuda, w_cuda, grad_out_cuda, result
    finally:
        gc.enable()

    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0


# -- Production dispatch: through the ordinary Tensor.conv2d / nn.Conv2d API --
#
# `tests/test_cuda_conv.py`'s existing shapes (Cin/Cout <= 4, K=3) top out at
# ~144 weight elements -- always below `_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD`
# (256) -- so none of them exercise `CUDABackend.conv2d_backward`'s new
# im2col+GEMM dispatch branch. These tests use `Cout*Cin*KH*KW >= 256`
# shapes specifically to cover that branch through the real, public API
# (rather than only via direct `dweight_im2col_gemm` calls above).


def _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding, bias=True, seed=70):
    forge.random.seed(seed)
    cpu_layer = Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
    cuda_layer = Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    if bias:
        cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")
    return cpu_layer, cuda_layer


@pytest.mark.parametrize(
    "in_ch,out_ch,kernel_size,stride,padding",
    [
        (8, 16, 3, 1, 1),    # 1,152 elements -- at/above threshold
        (16, 32, 3, 2, 1),   # 4,608 elements, strided -- above threshold
        (2, 3, 3, 1, 1),     # 54 elements -- below threshold, unaffected regression check
    ],
)
def test_production_conv2d_backward_matches_cpu_across_threshold(in_ch, out_ch, kernel_size, stride, padding):
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding)
    x_data = np.random.default_rng(71).standard_normal((2, in_ch, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cpu_layer(x_cpu).sum().backward()
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)


def test_production_conv2d_weight_reuse_accumulates_matching_cpu_above_threshold():
    """Gradient accumulation across repeated backward() calls, mirroring test_cuda_conv.py's
    small-shape version but at a weight-element count that exercises the im2col+GEMM path."""
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 1, 1, bias=False, seed=72)
    x1 = np.random.default_rng(73).standard_normal((2, 8, 9, 9)).astype(np.float32)
    x2 = np.random.default_rng(74).standard_normal((2, 8, 9, 9)).astype(np.float32)

    (cpu_layer(Tensor(x1)).sum() + cpu_layer(Tensor(x2)).sum()).backward()
    (cuda_layer(Tensor(x1, device="cuda")).sum() + cuda_layer(Tensor(x2, device="cuda")).sum()).backward()

    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_on_explicit_stream_above_threshold():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 2, 1, seed=75)
    x_data = np.random.default_rng(76).standard_normal((2, 8, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
        cuda_layer(x_cuda).sum().backward()
    s.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_repeated_use_does_not_grow_active_memory_above_threshold():
    x_data = np.random.default_rng(77).standard_normal((2, 8, 9, 9)).astype(np.float32)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    layer = Conv2d(8, 16, kernel_size=3, stride=1, padding=1).to("cuda")
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
