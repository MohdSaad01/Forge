"""Milestone 45 tests: dWeight below-256-weight-element warp-shuffle candidates.

M44 found the below-`CONV2D_WEIGHT_REDUCE_THRESHOLD` (256) dWeight dispatch
path (`k_conv2d_backward_weight_reduce`, still production, unchanged since
M21) the single largest contributor to the real MNIST training step, and
recommended a warp-shuffle-based cooperative reduction as a genuinely
untried angle. M45 benchmarked two such candidates -- M33's existing
`k_conv2d_backward_weight_warp` (one full 32-lane warp per weight element,
`cf_conv2d_backward_weight_warpreduce_*`) and a new sub-warp-group variant
(`k_conv2d_backward_weight_warp_subgroup`, `cf_conv2d_backward_weight_
warpsubgroup_*`, configurable 8/16/32-lane groups) -- and REJECTED both:
neither cleared the milestone's 1.15x acceptance bar at `mnist_conv1`'s own
shape or across an independent weight-element-count/reduction-size/K sweep,
and several sweep points regressed relative to the current production
kernel. See `docs/performance/conv2d-backward-profiling.md`'s **Milestone
45** section for the complete benchmark evidence and root-cause discussion.

This module is the correctness coverage for the new profiling-only kernel
code this milestone adds (`cf_conv2d_backward_weight_warpsubgroup_*` --
`warpreduce` is already covered by `tests/test_cuda_conv2d_backward_weight_
cooperative.py`, M33) -- never called by `CUDABackend` itself, production
dispatch and both production kernels are byte-for-byte unchanged. Also
extends `warpreduce` coverage to the below-256 regime specifically (M33's
own coverage focused on the >= 1,152-weight-element regime it was
investigating) and adds the boundary/finite-difference/explicit-stream
checks the milestone brief's Sections 9-10 ask for.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import CUDAStorage, get_cuda_backend
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL_F32 = dict(rtol=1e-4, atol=1e-4)
TOL_F64 = dict(rtol=1e-6, atol=1e-6)

CONV2D_WEIGHT_REDUCE_THRESHOLD = 256  # must match kernels.cu's dispatch


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _hout_wout(H, K, S, P):
    return (H + 2 * P - K) // S + 1


def _cpu_grad_w(x_data, w_data, b_data, stride, padding):
    layer_x = Tensor(x_data.copy(), requires_grad=True)
    w = Tensor(w_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)
    out = layer_x.conv2d(w, b, stride, padding)
    out.sum().backward()
    return w.grad.numpy()


def _raw_grad_w_warpsubgroup(x_data, w_data, b_data, stride, padding, warps_per_block, group_size,
                              dtype=np.float32, stream_handle=None):
    """Call `cf_conv2d_backward_weight_warpsubgroup_*` directly (profiling-only)."""
    backend = get_cuda_backend()
    lib = backend._lib
    N, Cin, H, W = x_data.shape
    Cout, _, KH, KW = w_data.shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = _hout_wout(H, KH, SH, PH), _hout_wout(W, KW, SW, PW)

    x = Tensor(x_data.astype(dtype).copy(), device="cuda")
    grad_out = Tensor(np.ones((N, Cout, Hout, Wout), dtype=dtype), device="cuda")
    forge.cuda.synchronize()

    itemsize = np.dtype(dtype).itemsize
    grad_w_ptr = backend._alloc(Cout * Cin * KH * KW * itemsize)
    # Wrap immediately in a `CUDAStorage` (the M25 caching allocator's own
    # release path) -- see `forge_hardware_quirks`/M33's own memory test
    # comment for why a raw `cf_free` here would corrupt later tests'
    # allocator-bookkeeping assertions.
    grad_w_storage = CUDAStorage(grad_w_ptr, (Cout, Cin, KH, KW), dtype, lib)
    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    suffix = "f32" if dtype == np.float32 else "f64"
    fn = getattr(lib, f"cf_conv2d_backward_weight_warpsubgroup_{suffix}")
    code = fn(grad_out._data.ptr, x._data.ptr, grad_w_ptr, *shape_args,
              ctypes.c_int(warps_per_block), ctypes.c_int(group_size), stream_handle)
    assert code == 0, f"warpsubgroup launch failed with code {code}"
    return backend.to_numpy(grad_w_storage)


# Below-256-weight-element shapes, spanning the M45 sweep's own range
# (9-252 weight elements) plus a couple of stride/padding variations.
BELOW_256_SHAPES = [
    # (Cin, Cout, H, W, K, stride, padding, weight_elements)
    (1, 1, 8, 8, 3, 1, 1, 9),         # smallest tested (M45 we_16 point)
    (1, 8, 28, 28, 3, 1, 1, 72),      # mnist_conv1's own exact shape
    (2, 8, 9, 9, 3, 2, 1, 144),       # strided
    (3, 7, 13, 13, 3, 1, 0, 189),     # no padding
    (15, 17, 12, 12, 1, 1, 0, 255),   # one below the M21 threshold, K=1
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p,we", BELOW_256_SHAPES)
@pytest.mark.parametrize("group_size", [8, 16, 32])
@pytest.mark.parametrize("warps_per_block", [1, 2, 4])
def test_warpsubgroup_matches_cpu_f32(cin, cout, h, w, k, s, p, we, group_size, warps_per_block):
    forge.random.seed(450)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(451).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    actual = _raw_grad_w_warpsubgroup(x_data, w_data, b_data, (s, s), (p, p), warps_per_block, group_size)
    np.testing.assert_allclose(actual, expected, **TOL_F32)


@pytest.mark.parametrize("cin,cout,h,w,k,s,p,we", BELOW_256_SHAPES[:3])
def test_warpsubgroup_matches_cpu_f64(cin, cout, h, w, k, s, p, we):
    forge.random.seed(452)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(453).standard_normal((2, cin, h, w))
    w_data = layer.weight.numpy().astype(np.float64)
    b_data = layer.bias.numpy().astype(np.float64)

    expected = _cpu_grad_w(x_data.astype(np.float64), w_data, b_data, (s, s), (p, p))
    actual = _raw_grad_w_warpsubgroup(
        x_data, w_data, b_data, (s, s), (p, p), warps_per_block=4, group_size=16, dtype=np.float64,
    )
    np.testing.assert_allclose(actual, expected, **TOL_F64)


# -- `warpreduce` (M33's existing kernel) at the below-256 regime specifically --
# (M33's own test file exercised it only at 54/1,152/4,608-element shapes; this
# extends coverage across the M45 sweep's own below-256 shapes.)


def _raw_grad_w_warpreduce(x_data, w_data, b_data, stride, padding, warps_per_block, dtype=np.float32):
    backend = get_cuda_backend()
    lib = backend._lib
    N, Cin, H, W = x_data.shape
    Cout, _, KH, KW = w_data.shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = _hout_wout(H, KH, SH, PH), _hout_wout(W, KW, SW, PW)

    x = Tensor(x_data.astype(dtype).copy(), device="cuda")
    grad_out = Tensor(np.ones((N, Cout, Hout, Wout), dtype=dtype), device="cuda")
    forge.cuda.synchronize()

    itemsize = np.dtype(dtype).itemsize
    grad_w_ptr = backend._alloc(Cout * Cin * KH * KW * itemsize)
    grad_w_storage = CUDAStorage(grad_w_ptr, (Cout, Cin, KH, KW), dtype, lib)
    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    suffix = "f32" if dtype == np.float32 else "f64"
    fn = getattr(lib, f"cf_conv2d_backward_weight_warpreduce_{suffix}")
    code = fn(grad_out._data.ptr, x._data.ptr, grad_w_ptr, *shape_args, ctypes.c_int(warps_per_block), None)
    assert code == 0, f"warpreduce launch failed with code {code}"
    return backend.to_numpy(grad_w_storage)


@pytest.mark.parametrize("cin,cout,h,w,k,s,p,we", BELOW_256_SHAPES)
@pytest.mark.parametrize("warps_per_block", [1, 4, 16])
def test_warpreduce_below256_matches_cpu(cin, cout, h, w, k, s, p, we, warps_per_block):
    forge.random.seed(454)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(455).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    actual = _raw_grad_w_warpreduce(x_data, w_data, b_data, (s, s), (p, p), warps_per_block)
    np.testing.assert_allclose(actual, expected, **TOL_F32)


# -- Finite-difference (Section 9) -------------------------------------------


def test_warpsubgroup_finite_difference():
    """Direct finite-difference check of the warpsubgroup candidate's own
    grad_weight output (not just an analytic-vs-CPU comparison)."""
    rng = np.random.default_rng(456)
    N, Cin, Cout, H, W, K, S, P = 2, 1, 8, 10, 10, 3, 1, 1
    x_data = rng.standard_normal((N, Cin, H, W)).astype(np.float64)
    w_data = rng.standard_normal((Cout, Cin, K, K)).astype(np.float64) * 0.1
    Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
    grad_out_ones = np.ones((N, Cout, Hout, Wout), dtype=np.float64)

    b_data = np.zeros((Cout,), dtype=np.float64)
    analytic = _raw_grad_w_warpsubgroup(
        x_data, w_data, b_data, (S, S), (P, P), warps_per_block=2, group_size=8, dtype=np.float64,
    )

    eps = 1e-6
    numeric = np.zeros_like(w_data)
    layer_x = Tensor(x_data.copy())
    w_flat = w_data.ravel()
    for idx in range(w_flat.size):
        w_plus = w_data.copy().ravel()
        w_plus[idx] += eps
        w_minus = w_data.copy().ravel()
        w_minus[idx] -= eps
        out_plus = layer_x.conv2d(Tensor(w_plus.reshape(w_data.shape)), Tensor(b_data), (S, S), (P, P))
        out_minus = layer_x.conv2d(Tensor(w_minus.reshape(w_data.shape)), Tensor(b_data), (S, S), (P, P))
        loss_plus = out_plus.numpy().sum()
        loss_minus = out_minus.numpy().sum()
        numeric.ravel()[idx] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-2, atol=1e-2)


# -- Explicit-stream execution (Section 9) -----------------------------------


def test_warpsubgroup_explicit_stream_matches_cpu():
    forge.random.seed(457)
    cin, cout, h, w, k, s, p = 1, 8, 28, 28, 3, 1, 1
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(458).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p))

    stream = forge.cuda.Stream()
    actual = _raw_grad_w_warpsubgroup(
        x_data, w_data, b_data, (s, s), (p, p), warps_per_block=2, group_size=16,
        stream_handle=stream.handle,
    )
    stream.synchronize()
    np.testing.assert_allclose(actual, expected, **TOL_F32)


# -- Boundary testing (Section 10): 255 stays below-256 dispatch, unchanged --


def test_production_dispatch_unaffected_at_and_above_threshold():
    """`cf_conv2d_backward_weight_*` (the actual production dispatcher) must
    still route weight_elements=255 to block-reduce and 256/257 to
    per-thread -- confirms this milestone's new profiling-only kernel
    additions did not alter the M21 dispatch threshold at all."""
    backend = get_cuda_backend()
    lib = backend._lib
    fn_current = getattr(lib, "cf_conv2d_backward_weight_f32")

    def _run_current(x_data, cin, cout, k=1):
        N, H, W, S, P = 2, 6, 6, 1, 0
        Hout, Wout = _hout_wout(H, k, S, P), _hout_wout(W, k, S, P)
        x = Tensor(x_data.copy(), device="cuda")
        grad_out = Tensor(np.ones((N, cout, Hout, Wout), dtype=np.float32), device="cuda")
        forge.cuda.synchronize()
        grad_w_ptr = backend._alloc(cout * cin * k * k * 4)
        grad_w_storage = CUDAStorage(grad_w_ptr, (cout, cin, k, k), np.float32, lib)
        shape_args = (
            ctypes.c_int(N), ctypes.c_int(cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(cout), ctypes.c_int(k), ctypes.c_int(k),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        code = fn_current(grad_out._data.ptr, x._data.ptr, grad_w_ptr, *shape_args, None)
        assert code == 0
        return backend.to_numpy(grad_w_storage)

    # Just confirms the production entry point still runs correctly and
    # matches CPU at both sides of the threshold -- the dispatch *decision*
    # itself is unreachable from Python, so this is the closest black-box
    # proxy: correctness holds regardless of which internal kernel served it.
    for cin, cout in [(15, 17), (16, 16), (1, 257)]:  # 255 / 256 / 257
        layer = Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)
        x_data = np.random.default_rng(1).standard_normal((2, cin, 6, 6)).astype(np.float32)
        w_data = layer.weight.numpy().copy()
        b_data = layer.bias.numpy().copy()
        expected = _cpu_grad_w(x_data, w_data, b_data, (1, 1), (0, 0))
        actual = _run_current(x_data, cin, cout, k=1)
        np.testing.assert_allclose(actual, expected, **TOL_F32)


# -- Memory safety (repeated use, Section 12) --------------------------------


def test_warpsubgroup_repeated_use_does_not_grow_active_memory():
    import gc

    x_data = np.random.default_rng(459).standard_normal((2, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(460).standard_normal((16, 8, 3, 3)).astype(np.float32)
    b_data = np.zeros((16,), dtype=np.float32)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    gc.disable()
    try:
        for _ in range(20):
            _raw_grad_w_warpsubgroup(x_data, w_data, b_data, (1, 1), (1, 1), warps_per_block=4, group_size=8)
            _raw_grad_w_warpsubgroup(x_data, w_data, b_data, (1, 1), (1, 1), warps_per_block=2, group_size=16)
    finally:
        gc.enable()

    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0
