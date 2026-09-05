"""Milestone 37 tests: `dWeight` split-K GEMM (production, supersedes M34's plain GEMM call).

M37 decomposed the M34 im2col+GEMM `dWeight` path
(`forge.backend.cuda.experimental_conv_im2col`) and found the GEMM call
itself (`cf_matmul_*`, `k_matmul`) launches as few as 5 of the 940MX's
24-resident-block device capacity at several representative shapes -- the
huge `N*Hout*Wout` reduction lives entirely inside each block's serial inner
loop, invisible to block count, while `Cout`/`Cin*KH*KW` (the GEMM's actual
`M`/`N` dimensions) stay small. Two structurally different fixes (Candidates
A/C, `forge.backend.cuda.experimental_conv_fused`, fusing the im2col/permute
gathers directly into the GEMM's tile loads) were measured *slower* than the
M34 baseline at every shape with >= 18 GEMM blocks -- recomputing gather
indices costs more than the occupancy fix bought back once occupancy was no
longer the bottleneck. Candidate E (`dweight_im2col_gemm_splitk`, this
module's target) isolates the occupancy fix alone: M34's already-fast,
cache-friendly `Xcol`/`dYcolT` buffers (`im2col`/`grad_output_permute`, both
completely unmodified) are fed to a split-K GEMM (`cf_matmul_splitk_*`, a
new, narrowly-scoped kernel -- `k_matmul` itself stays untouched) that
splits the reduction dimension across `num_k_splits` blocks, each
atomically accumulating its partial sum into a pre-zeroed output. Measured
2.7-9.0x faster than M34 at every one of the 7 representative shapes, with
no regression -- see `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 37** section for the complete evidence.
`CUDABackend.conv2d_backward` (`backend.py`) now dispatches to
`dweight_im2col_gemm_splitk` at/above `_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD`
(unchanged from M34) instead of the plain `dweight_im2col_gemm`.

This module mirrors `tests/test_cuda_conv2d_backward_weight_im2col_gemm.py`'s
(M34) structure exactly: direct correctness (CPU comparison across shapes,
finite difference, explicit-stream, cross-stream, repeated-use memory
safety) plus production-dispatch coverage through the real `Tensor.conv2d`/
`nn.Conv2d` API.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import get_cuda_backend
from forge.backend.cuda.experimental_conv_im2col import dweight_im2col_gemm_splitk, recommended_num_k_splits
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _reference_and_splitk_grad_w(x_data, w_data, b_data, stride, padding, upstream=None):
    """CPU-reference `grad_weight` vs. the M37 split-K GEMM path, same inputs."""
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    out_cpu = x_cpu.conv2d(w_cpu, b_cpu, stride, padding)
    if upstream is None:
        upstream = np.random.default_rng(199).standard_normal(out_cpu.shape).astype(x_data.dtype)
    out_cpu.backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_im2col_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, stride, padding)
    actual = backend.to_numpy(result)
    return expected, actual


# -- Correctness vs. CPU across shapes / stride / padding / kernel size ------

SHAPES = [
    # (Cin, Cout, H, W, K, stride, padding) -- spans the M32/M33/M34
    # threshold (256 weight elements) and a range of split-count regimes
    # (small M -> few reduction tiles -> num_k_splits clamped down; large
    # M -> the full 16-way split).
    (2, 3, 6, 6, 3, 1, 0),      # 54 elements, no padding, below threshold
    (2, 3, 6, 6, 3, 1, 1),      # 54 elements, padded, below threshold
    (8, 16, 9, 9, 3, 1, 1),     # 1,152 elements -- at/above threshold
    (16, 32, 7, 7, 3, 2, 1),    # 4,608 elements, strided
    (3, 4, 9, 9, 3, 2, 1),      # asymmetric N, strided
    (2, 3, 5, 5, 2, 1, 0),      # even kernel size, odd spatial size
    (1, 1, 5, 5, 3, 1, 1),      # Cin=Cout=1 -- smallest possible GEMM M/N dims
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_splitk_gemm_dweight_matches_cpu(cin, cout, h, w, k, s, p):
    forge.random.seed(150)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(151).standard_normal((3, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected, actual = _reference_and_splitk_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    np.testing.assert_allclose(actual, expected, **TOL)


def test_splitk_gemm_dweight_matches_cpu_float64():
    forge.random.seed(152)
    layer = Conv2d(4, 6, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(153).standard_normal((2, 4, 8, 8)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    expected, actual = _reference_and_splitk_grad_w(x_data, w_data, b_data, (1, 1), (1, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


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
def test_splitk_gemm_dweight_finite_difference(stride, padding):
    forge.random.seed(154)
    layer = Conv2d(2, 3, kernel_size=3, stride=stride, padding=padding)
    x_data = np.random.default_rng(155).standard_normal((2, 2, 7, 7)).astype(np.float64)
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
    result = dweight_im2col_gemm_splitk(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (stride, stride), (padding, padding)
    )
    actual = backend.to_numpy(result)

    np.testing.assert_allclose(actual, numerical_grad(loss, w_data.copy()), rtol=1e-2, atol=1e-2)


# -- Explicit-stream (async mode) correctness ---------------------------------


def test_splitk_gemm_dweight_on_explicit_stream_matches_cpu():
    forge.random.seed(156)
    layer = Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    x_data = np.random.default_rng(157).standard_normal((2, 3, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(158).standard_normal((2, 5, 5, 5)).astype(np.float32)

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
        result = dweight_im2col_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (2, 2), (1, 1)
        )
    s.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Cross-stream correctness (producer streams != compute stream) -----------


def test_splitk_gemm_dweight_correct_when_inputs_from_different_streams():
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    forge.random.seed(159)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(160).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(161).standard_normal((2, 4, 8, 8)).astype(np.float32)

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
        result = dweight_im2col_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


def test_splitk_gemm_dweight_reverse_stream_direction():
    """Same as the cross-stream test above but with producer/consumer stream roles swapped
    (Section on 'reverse stream direction' correctness, Phase 7 of the M37 brief)."""
    stream_compute = forge.cuda.Stream()
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()

    forge.random.seed(162)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(163).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(164).standard_normal((2, 4, 8, 8)).astype(np.float32)

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
        result = dweight_im2col_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Memory safety (repeated use) ---------------------------------------------


def test_splitk_gemm_dweight_repeated_use_does_not_grow_active_memory():
    x_data = np.random.default_rng(165).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(166).standard_normal((16, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(167).standard_normal((4, 16, 9, 9)).astype(np.float32)

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
            result = dweight_im2col_gemm_splitk(
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


def test_splitk_gemm_dweight_cache_hit_rate_matches_production_reuse():
    """`cf_matmul_splitk_*`'s `cudaMemsetAsync` writes into the same allocator-cached
    output buffer every call -- confirms the M25 caching allocator still reuses it
    (Section 'allocator cache reuse', Phase 7) rather than the memset path forcing a
    fresh allocation every time."""
    x_data = np.random.default_rng(168).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(169).standard_normal((16, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(170).standard_normal((4, 16, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    forge.cuda.empty_cache()

    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
    result = dweight_im2col_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    del result
    forge.cuda.synchronize()

    before = forge.cuda.memory_stats().cache_hit_count
    for _ in range(10):
        result = dweight_im2col_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
        del result
    forge.cuda.synchronize()
    after = forge.cuda.memory_stats().cache_hit_count

    assert after > before


# -- `recommended_num_k_splits` unit coverage ---------------------------------


@pytest.mark.parametrize(
    "m_reduction,expected",
    [
        (1, 1),        # < 1 tile -- clamp to 1, never 0
        (16, 1),       # exactly 1 tile
        (17, 2),       # just over 1 tile -- 2 tiles needed
        (256, 16),     # exactly 16 tiles -- splits == tiles
        (10816, 16),   # mnist_conv2's own M -- far more tiles than the 16 cap
        (100352, 16),  # large_spatial's own M
    ],
)
def test_recommended_num_k_splits(m_reduction, expected):
    assert recommended_num_k_splits(m_reduction) == expected


# -- Production dispatch: through the ordinary Tensor.conv2d / nn.Conv2d API --


def _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding, bias=True, seed=170):
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
def test_production_conv2d_backward_matches_cpu_via_splitk_dispatch(in_ch, out_ch, kernel_size, stride, padding):
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding)
    x_data = np.random.default_rng(171).standard_normal((2, in_ch, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cpu_layer(x_cpu).sum().backward()
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)


def test_production_conv2d_weight_reuse_accumulates_matching_cpu_via_splitk_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 1, 1, bias=False, seed=172)
    x1 = np.random.default_rng(173).standard_normal((2, 8, 9, 9)).astype(np.float32)
    x2 = np.random.default_rng(174).standard_normal((2, 8, 9, 9)).astype(np.float32)

    (cpu_layer(Tensor(x1)).sum() + cpu_layer(Tensor(x2)).sum()).backward()
    (cuda_layer(Tensor(x1, device="cuda")).sum() + cuda_layer(Tensor(x2, device="cuda")).sum()).backward()

    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_on_explicit_stream_via_splitk_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 2, 1, seed=175)
    x_data = np.random.default_rng(176).standard_normal((2, 8, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
        cuda_layer(x_cuda).sum().backward()
    s.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_repeated_use_does_not_grow_active_memory_via_splitk_dispatch():
    x_data = np.random.default_rng(177).standard_normal((2, 8, 9, 9)).astype(np.float32)

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
