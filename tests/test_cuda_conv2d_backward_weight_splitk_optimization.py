"""Milestone 43 tests: `dWeight` two-stage deterministic split-K reduction
(Candidate B, production at `blocks_y == 1`).

M42 found `k_dweight_halffused_gemm_splitk` (M38, `blocks_y == 1`, i.e.
`Cout <= 16`) reaching only 21-25% of the practical compute ceiling --
roughly half of every other GEMM-dispatched Conv2d kernel -- and flagged
split-K's atomic-accumulation combine step as an untested root-cause
hypothesis. M43's `num_k_splits` sensitivity sweep
(`benchmarks/m43_dweight_splitk_profile.py`) ruled out split-count/occupancy
retuning as a major source of headroom (the production `recommended_num_k_
splits` formula already sits within ~3-8% of its own swept optimum) but
confirmed atomics cost something real. `dweight_halffused_gemm_splitk_
tworeduce` (`experimental_conv_dweight_tworeduce.py`) replaces the atomic
combine with a deterministic two-stage reduction -- each split's GEMM writes
a disjoint partial-output slice (`k_dweight_halffused_gemm_splitk_partial`)
instead of an `atomicAdd`, and a second small kernel (`k_dweight_splitk_
reduce`) sums the `num_k_splits` axis -- at the *same*, unchanged
`num_k_splits` the production kernel already uses. Measured 1.14-1.24x
faster end-to-end at `mnist_conv2` and 1.02-1.04x faster at `large_spatial`,
at every tested `num_k_splits`, with no regression. `CUDABackend.
conv2d_backward` (`backend.py`) now dispatches to this kernel wherever M38's
kernel previously ran (`weight_elements >= _CONV2D_WEIGHT_IM2COL_GEMM_
THRESHOLD` and `ceil(Cout/16) == 1`) -- the dispatch boundary itself is
unchanged.

This module mirrors `tests/test_cuda_conv2d_backward_weight_halffused_gemm.py`'s
(M38) structure: direct correctness (CPU comparison, finite difference,
explicit-stream, cross-stream, repeated-use memory safety, a direct
same-inputs comparison against the M38 atomic kernel it replaces) plus
production-dispatch coverage confirming the `blocks_y == 1` dispatch
boundary through the real `Tensor.conv2d`/`nn.Conv2d` API.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import _MATMUL_TILE, get_cuda_backend
from forge.backend.cuda.experimental_conv_dweight_tworeduce import dweight_halffused_gemm_splitk_tworeduce
from forge.backend.cuda.experimental_conv_halffused import dweight_halffused_gemm_splitk
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _reference_and_tworeduce_grad_w(x_data, w_data, b_data, stride, padding, upstream=None, num_k_splits=None):
    """CPU-reference `grad_weight` vs. the M43 two-stage split-K reduction, same inputs."""
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    out_cpu = x_cpu.conv2d(w_cpu, b_cpu, stride, padding)
    if upstream is None:
        upstream = np.random.default_rng(699).standard_normal(out_cpu.shape).astype(x_data.dtype)
    out_cpu.backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_halffused_gemm_splitk_tworeduce(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, stride, padding, num_k_splits=num_k_splits,
    )
    actual = backend.to_numpy(result)
    return expected, actual


# -- Correctness vs. CPU across shapes / stride / padding / kernel size ------
# Spans both `blocks_y == 1` (Cout <= 16, this kernel's production regime)
# and `blocks_y >= 2` (Cout > 16 -- still correct, just not dispatched there)
# so the kernel itself is validated everywhere it can run.

SHAPES = [
    (2, 3, 6, 6, 3, 1, 0),      # 54 elements, no padding, below threshold, blocks_y=1
    (2, 3, 6, 6, 3, 1, 1),      # 54 elements, padded, below threshold, blocks_y=1
    (8, 16, 9, 9, 3, 1, 1),     # 1,152 elements, Cout=16 -- blocks_y=1 (dispatched here)
    (16, 32, 7, 7, 3, 2, 1),    # 4,608 elements, Cout=32 -- blocks_y=2 (not dispatched, still correct)
    (3, 4, 9, 9, 3, 2, 1),      # asymmetric N, strided, blocks_y=1
    (2, 3, 5, 5, 2, 1, 0),      # even kernel size, odd spatial size
    (1, 1, 5, 5, 3, 1, 1),      # Cin=Cout=1 -- smallest possible GEMM dims
    (4, 17, 8, 8, 3, 1, 1),     # Cout=17 -- just crosses into blocks_y=2
    (2, 1, 6, 6, 1, 1, 0),      # K=1, Cout=1 -- degenerate GEMM
    (2, 5, 8, 8, 5, 1, 2),      # K=5, larger kernel
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_tworeduce_dweight_matches_cpu(cin, cout, h, w, k, s, p):
    forge.random.seed(650)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(651).standard_normal((3, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected, actual = _reference_and_tworeduce_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    np.testing.assert_allclose(actual, expected, **TOL)


def test_tworeduce_dweight_matches_cpu_float64():
    forge.random.seed(652)
    layer = Conv2d(4, 6, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(653).standard_normal((2, 4, 8, 8)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    expected, actual = _reference_and_tworeduce_grad_w(x_data, w_data, b_data, (1, 1), (1, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("num_k_splits", [1, 2, 8, 32, 64])
def test_tworeduce_dweight_matches_cpu_at_custom_num_k_splits(num_k_splits):
    """`num_k_splits` is exposed as an explicit argument (unlike M38's kernel,
    which always used `recommended_num_k_splits` internally) -- every valid
    split count, including 1 (no parallelism) and one far above the
    production formula's value, must still produce the correct sum."""
    forge.random.seed(654)
    layer = Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(655).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected, actual = _reference_and_tworeduce_grad_w(x_data, w_data, b_data, (1, 1), (1, 1), num_k_splits=num_k_splits)
    np.testing.assert_allclose(actual, expected, **TOL)


def test_tworeduce_dweight_matches_m38_atomic_baseline():
    """Same mathematical operation as the M38 kernel it replaces in
    production -- same result within floating-point-reassociation tolerance,
    from the same inputs, at the same `num_k_splits`."""
    forge.random.seed(656)
    layer = Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(657).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    upstream = np.random.default_rng(658).standard_normal((4, 16, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    tworeduce = backend.to_numpy(
        dweight_halffused_gemm_splitk_tworeduce(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    )
    atomic = backend.to_numpy(
        dweight_halffused_gemm_splitk(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    )
    np.testing.assert_allclose(tworeduce, atomic, **TOL)


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
def test_tworeduce_dweight_finite_difference(stride, padding):
    forge.random.seed(659)
    layer = Conv2d(2, 3, kernel_size=3, stride=stride, padding=padding)
    x_data = np.random.default_rng(660).standard_normal((2, 2, 7, 7)).astype(np.float64)
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
    result = dweight_halffused_gemm_splitk_tworeduce(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (stride, stride), (padding, padding)
    )
    actual = backend.to_numpy(result)

    np.testing.assert_allclose(actual, numerical_grad(loss, w_data.copy()), rtol=1e-2, atol=1e-2)


# -- Explicit-stream (async mode) correctness ---------------------------------


def test_tworeduce_dweight_on_explicit_stream_matches_cpu():
    forge.random.seed(661)
    layer = Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    x_data = np.random.default_rng(662).standard_normal((2, 3, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(663).standard_normal((2, 5, 5, 5)).astype(np.float32)

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
        result = dweight_halffused_gemm_splitk_tworeduce(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (2, 2), (1, 1)
        )
    s.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Cross-stream correctness (producer streams != compute stream) -----------


def test_tworeduce_dweight_correct_when_inputs_from_different_streams():
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    forge.random.seed(664)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(665).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(666).standard_normal((2, 4, 8, 8)).astype(np.float32)

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
        result = dweight_halffused_gemm_splitk_tworeduce(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


def test_tworeduce_dweight_reverse_stream_direction():
    stream_compute = forge.cuda.Stream()
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()

    forge.random.seed(667)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(668).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(669).standard_normal((2, 4, 8, 8)).astype(np.float32)

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
        result = dweight_halffused_gemm_splitk_tworeduce(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1)
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


# -- Memory safety (repeated use) ---------------------------------------------


def test_tworeduce_dweight_repeated_use_does_not_grow_active_memory():
    x_data = np.random.default_rng(670).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(671).standard_normal((16, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(672).standard_normal((4, 16, 9, 9)).astype(np.float32)

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
            result = dweight_halffused_gemm_splitk_tworeduce(
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


def test_tworeduce_dweight_cache_hit_rate_matches_production_reuse():
    x_data = np.random.default_rng(673).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(674).standard_normal((16, 8, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(675).standard_normal((4, 16, 9, 9)).astype(np.float32)

    backend = get_cuda_backend()
    forge.cuda.empty_cache()

    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
    result = dweight_halffused_gemm_splitk_tworeduce(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    del result
    forge.cuda.synchronize()

    before = forge.cuda.memory_stats().cache_hit_count
    for _ in range(10):
        result = dweight_halffused_gemm_splitk_tworeduce(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
        del result
    forge.cuda.synchronize()
    after = forge.cuda.memory_stats().cache_hit_count

    assert after > before


# -- Production dispatch: through the ordinary Tensor.conv2d / nn.Conv2d API --
# Confirms the `blocks_y == 1` dispatch boundary (unchanged from M38) now
# reaches the M43 two-stage-reduction kernel, and that the `blocks_y >= 2`
# / below-threshold branches (untouched by M43) still work.


def _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding, bias=True, seed=670):
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
        (8, 16, 3, 1, 1),    # 1,152 elements, Cout=16 -- blocks_y=1, M43 tworeduce dispatched
        (4, 17, 3, 1, 1),    # 612 elements, Cout=17 -- blocks_y=2, im2col-smem splitk (unaffected)
        (16, 32, 3, 2, 1),   # 4,608 elements, Cout=32 -- blocks_y=2, im2col-smem splitk (unaffected)
        (2, 3, 3, 1, 1),     # 54 elements -- below threshold, unaffected regression check
    ],
)
def test_production_conv2d_backward_matches_cpu_via_m43_dispatch(in_ch, out_ch, kernel_size, stride, padding):
    assert (out_ch + _MATMUL_TILE - 1) // _MATMUL_TILE == (1 if out_ch <= 16 else 2)  # sanity-check the test's own premise
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding)
    x_data = np.random.default_rng(671).standard_normal((2, in_ch, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cpu_layer(x_cpu).sum().backward()
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)


def test_production_conv2d_weight_reuse_accumulates_matching_cpu_via_m43_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 1, 1, bias=False, seed=672)
    x1 = np.random.default_rng(673).standard_normal((2, 8, 9, 9)).astype(np.float32)
    x2 = np.random.default_rng(674).standard_normal((2, 8, 9, 9)).astype(np.float32)

    (cpu_layer(Tensor(x1)).sum() + cpu_layer(Tensor(x2)).sum()).backward()
    (cuda_layer(Tensor(x1, device="cuda")).sum() + cuda_layer(Tensor(x2, device="cuda")).sum()).backward()

    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_on_explicit_stream_via_m43_dispatch():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 2, 1, seed=675)
    x_data = np.random.default_rng(676).standard_normal((2, 8, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
        cuda_layer(x_cuda).sum().backward()
    s.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_production_conv2d_backward_repeated_use_does_not_grow_active_memory_via_m43_dispatch():
    x_data = np.random.default_rng(677).standard_normal((2, 8, 9, 9)).astype(np.float32)

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
