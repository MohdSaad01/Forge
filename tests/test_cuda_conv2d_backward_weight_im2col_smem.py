"""Milestone 39 tests: `dWeight` im2col shared-memory input-plane staging.

M38 left `k_im2col_conv2d` itself as the dominant remaining `dWeight` cost
at every `blocks_y >= 2` shape (`Cout > 16` -- the regime M38's half-fused
GEMM cannot help, since its redundant-regather tax grows with `blocks_y`).
`nvcc -Xptxas -v` on `k_im2col_conv2d` (unchanged since M34): 32
registers/thread, 0 bytes stack frame, 0 bytes spill -- not a local-memory
problem (unlike M32's original `dInput`), and (one thread per `Xcol` output
element, `M*K` threads at every shape) not occupancy-limited either. Two
structurally different candidates were designed and measured
(`benchmarks/m39_im2col_reuse_profile.py`):

- Candidate A (`im2col_indexed`): removes the redundant per-thread integer
  division `k_im2col_conv2d` pays for its `m`/`k` decompositions (hoisted
  `m` decomposition, table-lookup `k` decomposition) -- shape-dependent,
  regressing badly at small `K` (0.29-0.48x at `mnist_conv1`/`K=1`) despite
  winning at larger `K` -- **rejected** for production.
- Candidate B (`im2col_smem`): stages an entire `x[n,ci,:,:]` plane into
  shared memory once per block, serving every `Xcol` write for that plane
  from shared memory instead of a fresh global load -- won consistently at
  *every* representative shape and sweep point tested (1.05-1.96x, no
  regression anywhere), including the `K=1`/`reuse_factor=1.0` edge case.
  **Accepted** for the `blocks_y >= 2` production branch.

`dweight_im2col_smem_gemm_splitk` (`experimental_conv_im2col_reuse.py`)
implements exactly that: identical to M37's `dweight_im2col_gemm_splitk`
except its `im2col` stage uses Candidate B whenever the per-block
shared-memory request fits this device's cap, falling back to the
unmodified M34 `im2col` otherwise. `CUDABackend.conv2d_backward`
(`backend.py`) now calls it in place of `dweight_im2col_gemm_splitk` for the
whole `blocks_y >= 2` branch (Candidate A was never dispatched -- it is
tested here for correctness only, matching M33/M36's convention for a
rejected-but-kept-for-reproducibility candidate). See
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 39** section
for the complete evidence.

This module mirrors `tests/test_cuda_conv2d_backward_weight_halffused_gemm.py`'s
(M38) structure: direct correctness (CPU comparison across shapes spanning
kernel size/stride/padding/dtype, finite difference, explicit-stream,
cross-stream, repeated-use memory safety, allocator reuse) plus
production-dispatch coverage through the real `Tensor.conv2d`/`nn.Conv2d`
API, and correctness-only coverage for the rejected Candidate A.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import _MATMUL_TILE, get_cuda_backend
from forge.backend.cuda.experimental_conv_im2col import dweight_im2col_gemm_splitk, im2col
from forge.backend.cuda.experimental_conv_im2col_reuse import (
    build_k_decomposition_tables,
    dweight_im2col_smem_gemm_splitk,
    im2col_indexed,
    im2col_smem,
    im2col_smem_fits,
)
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _reference_and_actual_grad_w(x_data, w_data, b_data, stride, padding, upstream=None):
    """CPU-reference `grad_weight` vs. the M39 production dWeight path, same inputs."""
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    out_cpu = x_cpu.conv2d(w_cpu, b_cpu, stride, padding)
    if upstream is None:
        upstream = np.random.default_rng(399).standard_normal(out_cpu.shape).astype(x_data.dtype)
    out_cpu.backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_im2col_smem_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, stride, padding)
    actual = backend.to_numpy(result)
    return expected, actual


# -- Correctness vs. CPU across shapes / stride / padding / kernel size ------
# Spans `Cin=1` (smallest possible shared-memory plane), even/odd kernel
# sizes, strided/padded/unpadded configurations, and both sides of the
# `blocks_y` boundary (the production dispatch only reaches this function at
# `blocks_y >= 2`, but the kernel itself is correct at `blocks_y == 1` too).

SHAPES = [
    (1, 20, 6, 6, 3, 1, 0),      # Cin=1 -- smallest possible shared-memory plane
    (2, 3, 6, 6, 3, 1, 1),       # blocks_y=1 (not production-dispatched here, still correct)
    (8, 32, 9, 9, 3, 1, 1),      # Cout=32 -- blocks_y=2, production-dispatched
    (16, 48, 7, 7, 3, 2, 1),     # Cout=48 -- blocks_y=3, strided
    (3, 17, 9, 9, 3, 2, 1),      # Cout=17 -- just crosses into blocks_y=2
    (2, 32, 5, 5, 2, 1, 0),      # even kernel size, no padding
    (2, 32, 8, 8, 1, 1, 0),      # 1x1 kernel -- zero nominal reuse
    (4, 32, 13, 13, 5, 1, 2),    # 5x5 kernel -- large per-plane reuse
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_im2col_smem_dweight_matches_cpu(cin, cout, h, w, k, s, p):
    forge.random.seed(350)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(351).standard_normal((3, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected, actual = _reference_and_actual_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    np.testing.assert_allclose(actual, expected, **TOL)


def test_im2col_smem_dweight_matches_cpu_float64():
    forge.random.seed(352)
    layer = Conv2d(4, 24, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(353).standard_normal((2, 4, 8, 8)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    expected, actual = _reference_and_actual_grad_w(x_data, w_data, b_data, (1, 1), (1, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_im2col_smem_dweight_matches_splitk_baseline():
    """Both kernels compute the identical mathematical operation -- same result
    within floating-point-reassociation tolerance, from the *same* inputs."""
    forge.random.seed(354)
    layer = Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(355).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    upstream = np.random.default_rng(356).standard_normal((4, 32, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    smem = backend.to_numpy(
        dweight_im2col_smem_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    )
    splitk = backend.to_numpy(
        dweight_im2col_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    )
    np.testing.assert_allclose(smem, splitk, **TOL)


# -- im2col_smem in isolation vs. the unmodified M34 im2col -------------------


@pytest.mark.parametrize("cin,h,w,kh,kw,sh,sw,ph,pw", [
    (1, 5, 5, 3, 3, 1, 1, 1, 1),
    (3, 7, 7, 3, 3, 2, 2, 1, 1),
    (16, 28, 28, 3, 3, 1, 1, 1, 1),
    (8, 13, 13, 5, 5, 1, 2, 2, 1),
    (4, 6, 6, 1, 1, 1, 1, 0, 0),
])
def test_im2col_smem_matches_baseline_im2col(cin, h, w, kh, kw, sh, sw, ph, pw):
    def hout_wout(dim, k, s, p):
        return (dim + 2 * p - k) // s + 1

    Hout, Wout = hout_wout(h, kh, sh, ph), hout_wout(w, kw, sw, pw)
    backend = get_cuda_backend()
    x_data = np.random.default_rng(357).standard_normal((2, cin, h, w)).astype(np.float32)
    x = Tensor(x_data, device="cuda")._data
    forge.cuda.synchronize()

    baseline = backend.to_numpy(im2col(backend, x, 2, cin, h, w, kh, kw, sh, sw, ph, pw, Hout, Wout))
    smem = backend.to_numpy(im2col_smem(backend, x, 2, cin, h, w, kh, kw, sh, sw, ph, pw, Hout, Wout))
    np.testing.assert_allclose(smem, baseline, rtol=1e-6, atol=1e-6)


def test_im2col_smem_fits_reports_false_above_device_cap():
    assert im2col_smem_fits(np.float32, 28, 28) is True
    assert im2col_smem_fits(np.float32, 200, 200) is False  # 160,000 elements -- far over 48KB/4 bytes


# -- Candidate A (rejected): correctness-only, so shipped-but-unused code ----
# does not silently bit-rot (M33/M36's convention for a rejected candidate).


def test_build_k_decomposition_tables_matches_kernel_convention():
    ci, kh, kw = build_k_decomposition_tables(Cin=2, KH=3, KW=3)
    # k = ci*KH*KW + kh*KW + kw, matching k_im2col_conv2d's own decomposition.
    for k in range(2 * 3 * 3):
        expected_kw = k % 3
        expected_kh = (k // 3) % 3
        expected_ci = (k // 3) // 3
        assert (ci[k], kh[k], kw[k]) == (expected_ci, expected_kh, expected_kw)


@pytest.mark.parametrize("cin,h,w,kh,kw,sh,sw,ph,pw", [
    (1, 5, 5, 3, 3, 1, 1, 1, 1),
    (3, 7, 7, 3, 3, 2, 2, 1, 1),
    (16, 28, 28, 3, 3, 1, 1, 1, 1),
    (16, 28, 28, 5, 5, 1, 1, 2, 2),
])
def test_im2col_indexed_matches_baseline_im2col(cin, h, w, kh, kw, sh, sw, ph, pw):
    def hout_wout(dim, k, s, p):
        return (dim + 2 * p - k) // s + 1

    Hout, Wout = hout_wout(h, kh, sh, ph), hout_wout(w, kw, sw, pw)
    backend = get_cuda_backend()
    x_data = np.random.default_rng(358).standard_normal((2, cin, h, w)).astype(np.float32)
    x = Tensor(x_data, device="cuda")._data
    forge.cuda.synchronize()

    baseline = backend.to_numpy(im2col(backend, x, 2, cin, h, w, kh, kw, sh, sw, ph, pw, Hout, Wout))
    indexed = backend.to_numpy(im2col_indexed(backend, x, 2, cin, h, w, kh, kw, sh, sw, ph, pw, Hout, Wout))
    np.testing.assert_allclose(indexed, baseline, rtol=1e-6, atol=1e-6)


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
def test_im2col_smem_dweight_finite_difference(stride, padding):
    forge.random.seed(360)
    layer = Conv2d(2, 20, kernel_size=3, stride=stride, padding=padding)
    x_data = np.random.default_rng(361).standard_normal((2, 2, 7, 7)).astype(np.float64)
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
    result = dweight_im2col_smem_gemm_splitk(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (stride, stride), (padding, padding)
    )
    actual = backend.to_numpy(result)

    np.testing.assert_allclose(actual, numerical_grad(loss, w_data.copy()), rtol=1e-2, atol=1e-2)


# -- Explicit-stream (async mode) correctness ---------------------------------


def test_im2col_smem_dweight_on_explicit_stream_matches_cpu():
    forge.random.seed(362)
    layer = Conv2d(3, 20, kernel_size=3, stride=2, padding=1)
    x_data = np.random.default_rng(363).standard_normal((2, 3, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(364).standard_normal((2, 20, 5, 5)).astype(np.float32)

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
        result = dweight_im2col_smem_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (2, 2), (1, 1)
        )
    s.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Cross-stream correctness (producer streams != compute stream) -----------


def test_im2col_smem_dweight_correct_when_inputs_from_different_streams():
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    forge.random.seed(365)
    layer = Conv2d(2, 20, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(366).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(367).standard_normal((2, 20, 8, 8)).astype(np.float32)

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
        result = dweight_im2col_smem_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


def test_im2col_smem_dweight_reverse_stream_direction():
    stream_compute = forge.cuda.Stream()
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()

    forge.random.seed(368)
    layer = Conv2d(2, 20, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(369).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(370).standard_normal((2, 20, 8, 8)).astype(np.float32)

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
        result = dweight_im2col_smem_gemm_splitk(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Memory safety (repeated use) ---------------------------------------------


def test_im2col_smem_dweight_repeated_use_does_not_grow_active_memory():
    x_data = np.random.default_rng(371).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(372).standard_normal((32, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(373).standard_normal((4, 32, 9, 9)).astype(np.float32)

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
            result = dweight_im2col_smem_gemm_splitk(
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


def test_im2col_smem_dweight_cache_hit_rate_matches_production_reuse():
    x_data = np.random.default_rng(374).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(375).standard_normal((32, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(376).standard_normal((4, 32, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    forge.cuda.empty_cache()

    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
    result = dweight_im2col_smem_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    del result
    forge.cuda.synchronize()

    before = forge.cuda.memory_stats().cache_hit_count
    for _ in range(10):
        result = dweight_im2col_smem_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
        del result
    forge.cuda.synchronize()
    after = forge.cuda.memory_stats().cache_hit_count

    assert after > before


# -- Production dispatch: through the ordinary Tensor.conv2d / nn.Conv2d API --


def _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding, bias=True, seed=380):
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
        (4, 17, 3, 1, 1),    # 612 elements, Cout=17 -- blocks_y=2, M39 dispatched
        (16, 32, 3, 2, 1),   # 4,608 elements, Cout=32 -- blocks_y=2, M39 dispatched
        (8, 48, 3, 1, 1),    # 3,456 elements, Cout=48 -- blocks_y=3, M39 dispatched
        (8, 16, 3, 1, 1),    # blocks_y=1 -- routed to M38 half-fused, unaffected regression check
        (2, 3, 3, 1, 1),     # below threshold -- unaffected regression check
    ],
)
def test_production_conv2d_backward_matches_cpu_via_m39_dispatch(in_ch, out_ch, kernel_size, stride, padding):
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding)
    x_data = np.random.default_rng(381).standard_normal((2, in_ch, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cpu_layer(x_cpu).sum().backward()
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)


def test_production_conv2d_weight_reuse_accumulates_matching_cpu_via_m39_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 32, 3, 1, 1, bias=False, seed=382)
    x1 = np.random.default_rng(383).standard_normal((2, 8, 9, 9)).astype(np.float32)
    x2 = np.random.default_rng(384).standard_normal((2, 8, 9, 9)).astype(np.float32)

    (cpu_layer(Tensor(x1)).sum() + cpu_layer(Tensor(x2)).sum()).backward()
    (cuda_layer(Tensor(x1, device="cuda")).sum() + cuda_layer(Tensor(x2, device="cuda")).sum()).backward()

    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_on_explicit_stream_via_m39_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 32, 3, 2, 1, seed=385)
    x_data = np.random.default_rng(386).standard_normal((2, 8, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
        cuda_layer(x_cuda).sum().backward()
    s.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_repeated_use_does_not_grow_active_memory_via_m39_dispatch():
    x_data = np.random.default_rng(387).standard_normal((2, 8, 9, 9)).astype(np.float32)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    layer = Conv2d(8, 32, kernel_size=3, stride=1, padding=1).to("cuda")
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


def test_production_dispatch_sanity_check_blocks_y():
    """Sanity-checks this test module's own premise about the dispatch boundary."""
    assert (17 + _MATMUL_TILE - 1) // _MATMUL_TILE == 2
    assert (16 + _MATMUL_TILE - 1) // _MATMUL_TILE == 1
    assert (48 + _MATMUL_TILE - 1) // _MATMUL_TILE == 3
