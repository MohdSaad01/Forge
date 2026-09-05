"""Milestone 38 tests: `dWeight` half-fused split-K GEMM (Candidate B, partial production).

M38 investigated whether `im2col` (`k_im2col_conv2d`) -- measured by M37 as
54-63% of `dWeight` pipeline time, more than the GEMM itself -- could be
eliminated or reduced. M37 had already tried fusing *both* gathers
(`im2col`'s and `grad_output_permute`'s) directly into the GEMM's tile
loads (`experimental_conv_fused`, Candidates A/C) and rejected it: losing at
every shape with >= 18 GEMM blocks, because a block-tiled GEMM's `tile_a`
load is redundantly regathered `blocks_x` times (once per unique
`blockIdx.x`) and `tile_b` is redundantly regathered `blocks_y` times (once
per unique `blockIdx.y`) -- fusing *both* pays both taxes. M38 measured
`Cout <= 32` (`blocks_y = ceil(Cout/16) <= 2`) at every representative
shape while `Cin*KH*KW` reaches 144 (`blocks_x` up to 9), so fusing only
`im2col`'s gather (`tile_b`, this milestone's actual target) -- keeping
`grad_output_permute`'s cheap, already-materialized `dYcolT` as `tile_a`'s
source -- pays at most the cheap `blocks_y` tax, never the expensive
`blocks_x` one.

`dweight_halffused_gemm_splitk` (`experimental_conv_halffused.py`,
Candidate B) implements exactly that: no `Xcol` buffer, `dYcolT`
materialized as before. Measured (`Cout` in {8,16,17,24,32,48,64} at a
fixed base shape, interleaved A/B, CUDA events): a clean, monotonic win at
`blocks_y == 1` (`Cout <= 16`: 1.29-1.46x) that flips to a real,
reproducible regression at `blocks_y >= 2` (`Cout` 17-32: ~0.92-0.93x;
Cout=48/64: 0.68-0.76x). `CUDABackend.conv2d_backward` (`backend.py`) now
dispatches to Candidate B only when `weight_elements >=
_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD` *and* `ceil(Cout/16) == 1` -- exactly
the regime it measurably wins in -- and keeps M37's `dweight_im2col_gemm_
splitk` everywhere else, including every shape it would regress. See
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 38** section
for the complete evidence.

This module mirrors `tests/test_cuda_conv2d_backward_weight_splitk_gemm.py`'s
(M37) structure: direct correctness (CPU comparison across shapes spanning
both the `blocks_y == 1` and `blocks_y >= 2` regimes, finite difference,
explicit-stream, cross-stream, repeated-use memory safety) plus
production-dispatch coverage confirming both branches of the new `Cout`-
based condition through the real `Tensor.conv2d`/`nn.Conv2d` API.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import _MATMUL_TILE, get_cuda_backend
from forge.backend.cuda.experimental_conv_halffused import dweight_halffused_gemm_splitk
from forge.backend.cuda.experimental_conv_im2col import dweight_im2col_gemm_splitk
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _reference_and_halffused_grad_w(x_data, w_data, b_data, stride, padding, upstream=None):
    """CPU-reference `grad_weight` vs. the M38 half-fused split-K GEMM path, same inputs."""
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    out_cpu = x_cpu.conv2d(w_cpu, b_cpu, stride, padding)
    if upstream is None:
        upstream = np.random.default_rng(299).standard_normal(out_cpu.shape).astype(x_data.dtype)
    out_cpu.backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_halffused_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, stride, padding)
    actual = backend.to_numpy(result)
    return expected, actual


# -- Correctness vs. CPU across shapes / stride / padding / kernel size ------
# Spans both `blocks_y == 1` (Cout <= 16, Candidate B's winning regime) and
# `blocks_y >= 2` (Cout > 16, Candidate B is still correct there -- just not
# production-dispatched) so the kernel itself is validated everywhere it can
# run, independent of the dispatch decision.

SHAPES = [
    (2, 3, 6, 6, 3, 1, 0),      # 54 elements, no padding, below threshold, blocks_y=1
    (2, 3, 6, 6, 3, 1, 1),      # 54 elements, padded, below threshold, blocks_y=1
    (8, 16, 9, 9, 3, 1, 1),     # 1,152 elements, Cout=16 -- blocks_y=1 (dispatched here)
    (16, 32, 7, 7, 3, 2, 1),    # 4,608 elements, Cout=32 -- blocks_y=2 (not dispatched, still correct)
    (3, 4, 9, 9, 3, 2, 1),      # asymmetric N, strided, blocks_y=1
    (2, 3, 5, 5, 2, 1, 0),      # even kernel size, odd spatial size
    (1, 1, 5, 5, 3, 1, 1),      # Cin=Cout=1 -- smallest possible GEMM dims
    (4, 17, 8, 8, 3, 1, 1),     # Cout=17 -- just crosses into blocks_y=2
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_halffused_gemm_dweight_matches_cpu(cin, cout, h, w, k, s, p):
    forge.random.seed(250)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(251).standard_normal((3, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected, actual = _reference_and_halffused_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    np.testing.assert_allclose(actual, expected, **TOL)


def test_halffused_gemm_dweight_matches_cpu_float64():
    forge.random.seed(252)
    layer = Conv2d(4, 6, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(253).standard_normal((2, 4, 8, 8)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    expected, actual = _reference_and_halffused_grad_w(x_data, w_data, b_data, (1, 1), (1, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_halffused_gemm_dweight_matches_splitk_baseline():
    """Both kernels compute the identical mathematical operation -- same result
    within floating-point-reassociation tolerance, from the *same* inputs."""
    forge.random.seed(254)
    layer = Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(255).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    upstream = np.random.default_rng(256).standard_normal((4, 16, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    halffused = backend.to_numpy(
        dweight_halffused_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    )
    splitk = backend.to_numpy(
        dweight_im2col_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    )
    np.testing.assert_allclose(halffused, splitk, **TOL)


# -- Finite difference --------------------------------------------------------


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
def test_halffused_gemm_dweight_finite_difference(stride, padding):
    forge.random.seed(257)
    layer = Conv2d(2, 3, kernel_size=3, stride=stride, padding=padding)
    x_data = np.random.default_rng(258).standard_normal((2, 2, 7, 7)).astype(np.float64)
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
    result = dweight_halffused_gemm_splitk(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (stride, stride), (padding, padding)
    )
    actual = backend.to_numpy(result)

    np.testing.assert_allclose(actual, numerical_grad(loss, w_data.copy()), rtol=1e-2, atol=1e-2)


# -- Explicit-stream (async mode) correctness ---------------------------------


def test_halffused_gemm_dweight_on_explicit_stream_matches_cpu():
    forge.random.seed(259)
    layer = Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    x_data = np.random.default_rng(260).standard_normal((2, 3, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(261).standard_normal((2, 5, 5, 5)).astype(np.float32)

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
        result = dweight_halffused_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (2, 2), (1, 1)
        )
    s.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Cross-stream correctness (producer streams != compute stream) -----------


def test_halffused_gemm_dweight_correct_when_inputs_from_different_streams():
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    forge.random.seed(262)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(263).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(264).standard_normal((2, 4, 8, 8)).astype(np.float32)

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
        result = dweight_halffused_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


def test_halffused_gemm_dweight_reverse_stream_direction():
    stream_compute = forge.cuda.Stream()
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()

    forge.random.seed(265)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(266).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(267).standard_normal((2, 4, 8, 8)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    x_cpu.conv2d(w_cpu, b_cpu, (1, 1), (1, 1)).backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    with forge.cuda.stream(stream_g):
        grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    with forge.cuda.stream(stream_x):
        x_cuda = Tensor(x_data.copy(), device="cuda")
        w_cuda = Tensor(w_data.copy(), device="cuda")

    with forge.cuda.stream(stream_compute):
        result = dweight_halffused_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Memory safety (repeated use) ---------------------------------------------


def test_halffused_gemm_dweight_repeated_use_does_not_grow_active_memory():
    x_data = np.random.default_rng(268).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(269).standard_normal((16, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(270).standard_normal((4, 16, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    gc.disable()
    try:
        for _ in range(100):
            x_cuda = Tensor(x_data.copy(), device="cuda")
            w_cuda = Tensor(w_data.copy(), device="cuda")
            grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
            result = dweight_halffused_gemm_splitk(
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


def test_halffused_gemm_dweight_cache_hit_rate_matches_production_reuse():
    x_data = np.random.default_rng(271).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(272).standard_normal((16, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(273).standard_normal((4, 16, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    forge.cuda.empty_cache()

    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
    result = dweight_halffused_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    del result
    forge.cuda.synchronize()

    before = forge.cuda.memory_stats().cache_hit_count
    for _ in range(10):
        result = dweight_halffused_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
        del result
    forge.cuda.synchronize()
    after = forge.cuda.memory_stats().cache_hit_count

    assert after > before


# -- Production dispatch: through the ordinary Tensor.conv2d / nn.Conv2d API --
# Covers both branches of the new Cout-based condition (`_MATMUL_TILE`
# imported directly from `backend.py` so the test derives `blocks_y` the
# same way production dispatch does, rather than hardcoding "16").


def _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding, bias=True, seed=270):
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
        (8, 16, 3, 1, 1),    # 1,152 elements, Cout=16 -- blocks_y=1, Candidate B dispatched
        (4, 17, 3, 1, 1),    # 612 elements, Cout=17 -- blocks_y=2, splitk (unfused) dispatched
        (16, 32, 3, 2, 1),   # 4,608 elements, Cout=32 -- blocks_y=2, splitk (unfused) dispatched
        (2, 3, 3, 1, 1),     # 54 elements -- below threshold, unaffected regression check
    ],
)
def test_production_conv2d_backward_matches_cpu_via_m38_dispatch(in_ch, out_ch, kernel_size, stride, padding):
    assert (out_ch + _MATMUL_TILE - 1) // _MATMUL_TILE == (1 if out_ch <= 16 else 2)  # sanity-check the test's own premise
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding)
    x_data = np.random.default_rng(271).standard_normal((2, in_ch, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cpu_layer(x_cpu).sum().backward()
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)


def test_production_conv2d_weight_reuse_accumulates_matching_cpu_via_m38_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 1, 1, bias=False, seed=272)
    x1 = np.random.default_rng(273).standard_normal((2, 8, 9, 9)).astype(np.float32)
    x2 = np.random.default_rng(274).standard_normal((2, 8, 9, 9)).astype(np.float32)

    (cpu_layer(Tensor(x1)).sum() + cpu_layer(Tensor(x2)).sum()).backward()
    (cuda_layer(Tensor(x1, device="cuda")).sum() + cuda_layer(Tensor(x2, device="cuda")).sum()).backward()

    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_on_explicit_stream_via_m38_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 2, 1, seed=275)
    x_data = np.random.default_rng(276).standard_normal((2, 8, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
        cuda_layer(x_cuda).sum().backward()
    s.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_repeated_use_does_not_grow_active_memory_via_m38_dispatch():
    x_data = np.random.default_rng(277).standard_normal((2, 8, 9, 9)).astype(np.float32)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    layer = Conv2d(8, 16, kernel_size=3, stride=1, padding=1).to("cuda")
    for _ in range(50):
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
