"""Milestone 33 tests: dWeight cooperative-reduction candidate kernels.

M33 profiled whether a cooperative (multi-thread-per-weight-element)
reduction could beat the existing per-thread `dWeight` kernel
(`k_conv2d_backward_weight`) at the weight-element counts (>= 1,152) where
production currently uses it (see `docs/performance/conv2d-backward-
profiling.md`'s **Milestone 33** section). Two cooperative designs were
added to `kernels.cu` purely for that measurement -- `k_conv2d_backward_
weight_reduce`, forced regardless of the M21 threshold via
`cf_conv2d_backward_weight_blockreduce_*` (one block per weight element,
shared-memory tree reduction, already production code at small weight
counts) and the new `k_conv2d_backward_weight_warp` via `cf_conv2d_backward_
weight_warpreduce_*` (one warp per weight element, `__shfl_down_sync`
reduction, multiple weights per block) -- both measured 3-4x *slower* than
the existing per-thread kernel at every M32/M33 shape with >= 1,152 weight
elements, so neither replaced production dispatch (`CUDABackend.
conv2d_backward` and `cf_conv2d_backward_weight_*`'s own threshold are
unchanged this milestone).

This module is the correctness coverage for that new, profiling-only kernel
code (`cf_conv2d_backward_weight_{perthread,blockreduce,warpreduce}_*`,
never called by `CUDABackend` itself) -- each is compared directly against
`CPUBackend`'s `conv2d_backward` weight gradient, mirroring `tests/
test_cuda_conv.py`'s existing CPU-vs-CUDA comparison pattern.
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

TOL = dict(rtol=1e-4, atol=1e-4)


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


def _raw_grad_w(fn_name, x_data, w_data, b_data, stride, padding, extra_arg=None):
    """Call one of the M33 profiling-only forced-dispatch kernels directly."""
    backend = get_cuda_backend()
    lib = backend._lib
    N, Cin, H, W = x_data.shape
    Cout, _, KH, KW = w_data.shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = _hout_wout(H, KH, SH, PH), _hout_wout(W, KW, SW, PW)

    x = Tensor(x_data.copy(), device="cuda")
    w = Tensor(w_data.copy(), device="cuda")
    grad_out = Tensor(np.ones((N, Cout, Hout, Wout), dtype=np.float32), device="cuda")
    forge.cuda.synchronize()

    grad_w_ptr = backend._alloc(Cout * Cin * KH * KW * 4)
    # Wrap the raw pointer in a `CUDAStorage` immediately -- it was obtained
    # through the M25 caching allocator (`backend._alloc`), and releasing it
    # any other way (e.g. a raw `cf_free`) bypasses that allocator's own
    # `allocated_bytes` bookkeeping, corrupting later tests' memory-safety
    # assertions (`forge.cuda.memory_stats()`) even though nothing was
    # actually leaked at the driver level. `CUDAStorage.__del__` is the one
    # correct release path, exactly as every other CUDA-resident buffer in
    # Forge uses.
    grad_w_storage = CUDAStorage(grad_w_ptr, (Cout, Cin, KH, KW), np.float32, lib)
    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    fn = getattr(lib, fn_name)
    if extra_arg is None:
        code = fn(grad_out._data.ptr, x._data.ptr, grad_w_ptr, *shape_args, None)
    else:
        code = fn(grad_out._data.ptr, x._data.ptr, grad_w_ptr, *shape_args, ctypes.c_int(extra_arg), None)
    assert code == 0, f"{fn_name} launch failed with code {code}"

    return backend.to_numpy(grad_w_storage)


# `grad_output` is all-ones above, so the CPU reference must match: build it
# the same way (an all-ones upstream gradient, i.e. `out.sum().backward()`).
SHAPES = [
    # (Cin, Cout, H, W, K, stride, padding) -- one below, several at/above
    # the M21 threshold (weight_elements = Cout*Cin*K*K, 256).
    (2, 3, 6, 6, 3, 1, 1),      # 54 elements -- below threshold
    (8, 16, 9, 9, 3, 1, 1),     # 1,152 elements -- at/above threshold
    (16, 32, 7, 7, 3, 2, 1),    # 4,608 elements, strided
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_perthread_forced_matches_cpu(cin, cout, h, w, k, s, p):
    forge.random.seed(40)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(41).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    actual = _raw_grad_w("cf_conv2d_backward_weight_perthread_f32", x_data, w_data, b_data, (s, s), (p, p))
    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
@pytest.mark.parametrize("threads_per_block", [64, 128, 256])
def test_blockreduce_forced_matches_cpu(cin, cout, h, w, k, s, p, threads_per_block):
    forge.random.seed(42)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(43).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    actual = _raw_grad_w(
        "cf_conv2d_backward_weight_blockreduce_f32", x_data, w_data, b_data, (s, s), (p, p),
        extra_arg=threads_per_block,
    )
    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
@pytest.mark.parametrize("warps_per_block", [2, 4, 8])
def test_warpreduce_forced_matches_cpu(cin, cout, h, w, k, s, p, warps_per_block):
    forge.random.seed(44)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(45).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p))
    actual = _raw_grad_w(
        "cf_conv2d_backward_weight_warpreduce_f32", x_data, w_data, b_data, (s, s), (p, p),
        extra_arg=warps_per_block,
    )
    np.testing.assert_allclose(actual, expected, **TOL)


# -- Memory safety (Section 24 of the milestone brief) ------------------------


def test_cooperative_candidate_kernels_repeated_use_does_not_grow_active_memory():
    """Repeated calls to all three M33 profiling-only kernels leave allocator counters bounded."""
    import gc

    x_data = np.random.default_rng(46).standard_normal((2, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(47).standard_normal((16, 8, 3, 3)).astype(np.float32)
    b_data = np.zeros((16,), dtype=np.float32)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    gc.disable()
    try:
        for _ in range(20):
            _raw_grad_w("cf_conv2d_backward_weight_perthread_f32", x_data, w_data, b_data, (1, 1), (1, 1))
            _raw_grad_w(
                "cf_conv2d_backward_weight_blockreduce_f32", x_data, w_data, b_data, (1, 1), (1, 1), extra_arg=128
            )
            _raw_grad_w(
                "cf_conv2d_backward_weight_warpreduce_f32", x_data, w_data, b_data, (1, 1), (1, 1), extra_arg=4
            )
    finally:
        gc.enable()

    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0
