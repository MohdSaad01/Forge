"""Milestone 46 tests: dWeight below-256-weight-element grid-split
block-reduce candidate (rejected).

M45 rejected warp-shuffle reduction for the below-`CONV2D_WEIGHT_REDUCE_
THRESHOLD` (256) dWeight path (`k_conv2d_backward_weight_reduce`, still
production, unchanged since M21) and corrected M44's diagnosis: the
production kernel already launches `weight_elements x 256` threads -- more
than either rejected warp candidate. M46 investigated the remaining
untried axis directly: grid-level parallelism. `k_conv2d_backward_weight_
reduce_gridsplit` launches `num_splits` blocks per weight element (instead
of one), each reducing a disjoint slice of the `N*Hout*Wout` dimension;
`k_dweight_splitk_reduce` (M43's existing combine kernel, reused unmodified
-- a flat weight vector is just `Cout=1, Kdim=weight_elements` of the same
`(num_k_splits, rows*cols)` shape it already sums) combines the partial
results. `forge.backend.cuda.experimental_conv_dweight_gridsplit.
dweight_below256_gridsplit` wraps both calls with buffer management.

**Result: rejected.** A real `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
query found both the production kernel and this candidate land at the same
5 resident blocks/SM (register-bound) -- grid-splitting cannot raise the
940MX's per-SM concurrent-block ceiling, only launch more/smaller blocks
against it. A same-session, round-robin-interleaved CUDA-event A/B (see
`benchmarks/m46_dweight_below256_gridsplit_profile.py`'s own methodology-
correction note: an initial block-sequential timing attempt measured an
illusory 1.47x "win" at `mnist_conv1` that could not be reproduced once
properly interleaved, landing at 1.02x instead) found the best speedup
anywhere in the sweep topped out at ~1.07x -- never clearing the milestone's
1.15x acceptance bar -- and short reductions (`reduction_small`, 256
elements) regressed outright as split count grew. See `docs/performance/
conv2d-backward-profiling.md`'s **Milestone 46** section for the complete
evidence. Production dispatch and `k_conv2d_backward_weight_reduce` are
byte-for-byte unchanged; this module is the correctness coverage for the
new profiling-only kernel code this milestone adds.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import get_cuda_backend
from forge.backend.cuda.experimental_conv_dweight_gridsplit import dweight_below256_gridsplit
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


def _reference_and_gridsplit_grad_w(x_data, w_data, b_data, stride, padding, num_splits, upstream=None):
    """CPU-reference `grad_weight` vs. the M46 grid-split candidate, same inputs."""
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    out_cpu = x_cpu.conv2d(w_cpu, b_cpu, stride, padding)
    if upstream is None:
        upstream = np.random.default_rng(760).standard_normal(out_cpu.shape).astype(x_data.dtype)
    out_cpu.backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_below256_gridsplit(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, stride, padding, num_splits,
    )
    actual = backend.to_numpy(result)
    return expected, actual


# -- Correctness vs. CPU across shapes / stride / padding / kernel size / -----
# -- num_splits, spanning the M46 sweep's own below-256 shapes ---------------

BELOW_256_SHAPES = [
    # (Cin, Cout, H, W, K, stride, padding, weight_elements)
    (1, 1, 8, 8, 3, 1, 1, 9),         # smallest tested
    (1, 8, 28, 28, 3, 1, 1, 72),      # mnist_conv1's own exact shape
    (2, 8, 9, 9, 3, 2, 1, 144),       # strided
    (3, 7, 13, 13, 3, 1, 0, 189),     # no padding
    (15, 17, 12, 12, 1, 1, 0, 255),   # one below the M21 threshold, K=1
]


@pytest.mark.parametrize("cin,cout,h,w,k,s,p,we", BELOW_256_SHAPES)
@pytest.mark.parametrize("num_splits", [1, 2, 4, 8])
def test_gridsplit_matches_cpu_f32(cin, cout, h, w, k, s, p, we, num_splits):
    forge.random.seed(760)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(761).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    expected, actual = _reference_and_gridsplit_grad_w(x_data, w_data, b_data, (s, s), (p, p), num_splits)
    np.testing.assert_allclose(actual, expected, **TOL_F32)


@pytest.mark.parametrize("num_splits", [1, 2, 4, 8])
def test_gridsplit_matches_cpu_f64(num_splits):
    forge.random.seed(762)
    cin, cout, h, w, k, s, p = 2, 8, 9, 9, 3, 2, 1
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(763).standard_normal((2, cin, h, w))
    w_data = layer.weight.numpy().astype(np.float64)
    b_data = layer.bias.numpy().astype(np.float64)

    expected, actual = _reference_and_gridsplit_grad_w(
        x_data.astype(np.float64), w_data, b_data, (s, s), (p, p), num_splits,
    )
    np.testing.assert_allclose(actual, expected, **TOL_F64)


# -- Direct same-inputs comparison against the production kernel -------------


def test_gridsplit_matches_production_atomic_free_kernel():
    """`num_splits` in {1,2,4,8} must all agree with the production
    `k_conv2d_backward_weight_reduce` kernel on identical inputs (not just
    each independently matching CPU) -- rules out a shared bug that happens
    to cancel against the CPU reference."""
    from forge.backend.cuda.backend import CUDAStorage
    import ctypes

    forge.random.seed(764)
    cin, cout, h, w, k, s, p = 1, 8, 28, 28, 3, 1, 1
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(765).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()

    backend = get_cuda_backend()
    lib = backend._lib
    Hout = (h + 2 * p - k) // s + 1
    Wout = (w + 2 * p - k) // s + 1
    x_cuda = Tensor(x_data.copy(), device="cuda")
    grad_out_cuda = Tensor(np.random.default_rng(766).standard_normal((2, cout, Hout, Wout)).astype(np.float32), device="cuda")
    forge.cuda.synchronize()

    weight_elements = cout * cin * k * k
    shape_args = (
        ctypes.c_int(2), ctypes.c_int(cin), ctypes.c_int(h), ctypes.c_int(w),
        ctypes.c_int(cout), ctypes.c_int(k), ctypes.c_int(k),
        ctypes.c_int(s), ctypes.c_int(s), ctypes.c_int(p), ctypes.c_int(p),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    grad_w_ptr = backend._alloc(weight_elements * 4)
    fn_production = getattr(lib, "cf_conv2d_backward_weight_f32")
    code = fn_production(grad_out_cuda._data.ptr, x_cuda._data.ptr, grad_w_ptr, *shape_args, None)
    assert code == 0
    forge.cuda.synchronize()
    production = backend.to_numpy(CUDAStorage(grad_w_ptr, (weight_elements,), np.float32, lib)).copy()

    for num_splits in (1, 2, 4, 8):
        result = dweight_below256_gridsplit(
            backend, grad_out_cuda._data, x_cuda._data, (cout, cin, k, k), (s, s), (p, p), num_splits,
        )
        actual = backend.to_numpy(result).reshape(-1)
        np.testing.assert_allclose(actual, production, rtol=1e-4, atol=1e-4)


# -- Finite-difference ---------------------------------------------------------


def test_gridsplit_finite_difference():
    rng = np.random.default_rng(767)
    N, Cin, Cout, H, W, K, S, P = 2, 1, 8, 10, 10, 3, 1, 1
    x_data = rng.standard_normal((N, Cin, H, W)).astype(np.float64)
    w_data = rng.standard_normal((Cout, Cin, K, K)).astype(np.float64) * 0.1
    Hout, Wout = (H + 2 * P - K) // S + 1, (W + 2 * P - K) // S + 1
    b_data = np.zeros((Cout,), dtype=np.float64)

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_ones = Tensor(np.ones((N, Cout, Hout, Wout), dtype=np.float64), device="cuda")
    forge.cuda.synchronize()

    analytic = backend.to_numpy(
        dweight_below256_gridsplit(backend, grad_out_ones._data, x_cuda._data, w_cuda._data.shape, (S, S), (P, P), 4)
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


# -- Explicit-stream execution -------------------------------------------------


def test_gridsplit_on_explicit_stream_matches_cpu():
    forge.random.seed(768)
    layer = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(769).standard_normal((2, 1, 28, 28)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(770).standard_normal((2, 8, 28, 28)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    x_cpu.conv2d(w_cpu, b_cpu, (1, 1), (1, 1)).backward(Tensor(upstream.copy()))
    expected = w_cpu.grad.numpy()

    backend = get_cuda_backend()
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda")
        w_cuda = Tensor(w_data.copy(), device="cuda")
        grad_out_cuda = Tensor(upstream.copy(), device="cuda")
        result = dweight_below256_gridsplit(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1), 2,
        )
    s.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL_F32)


# -- Cross-stream correctness (producer streams != compute stream) -----------


def test_gridsplit_correct_when_inputs_from_different_streams():
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    forge.random.seed(771)
    layer = Conv2d(2, 5, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(772).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    upstream = np.random.default_rng(773).standard_normal((2, 5, 8, 8)).astype(np.float32)

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
        result = dweight_below256_gridsplit(
            backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1), 4,
        )
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL_F32)


# -- Boundary testing: 255 stays below-256 dispatch, unchanged ---------------


def test_production_dispatch_unaffected_at_and_above_threshold():
    """Confirms this milestone's new profiling-only kernel additions did not
    alter the M21 dispatch threshold at all -- production entry point still
    correct on both sides of 255/256/257."""
    import ctypes

    from forge.backend.cuda.backend import CUDAStorage

    backend = get_cuda_backend()
    lib = backend._lib
    fn_current = getattr(lib, "cf_conv2d_backward_weight_f32")

    def _run_current(x_data, cin, cout, k=1):
        N, H, W, S, P = 2, 6, 6, 1, 0
        Hout, Wout = (H - k) // S + 1, (W - k) // S + 1
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

    def _cpu_grad_w(x_data, w_data, b_data, stride, padding):
        layer_x = Tensor(x_data.copy(), requires_grad=True)
        w = Tensor(w_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        out = layer_x.conv2d(w, b, stride, padding)
        out.sum().backward()
        return w.grad.numpy()

    for cin, cout in [(15, 17), (16, 16), (1, 257)]:  # 255 / 256 / 257
        layer = Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)
        x_data = np.random.default_rng(1).standard_normal((2, cin, 6, 6)).astype(np.float32)
        w_data = layer.weight.numpy().copy()
        b_data = layer.bias.numpy().copy()
        expected = _cpu_grad_w(x_data, w_data, b_data, (1, 1), (0, 0))
        actual = _run_current(x_data, cin, cout, k=1)
        np.testing.assert_allclose(actual, expected, **TOL_F32)


# -- Memory safety (repeated use) ---------------------------------------------


def test_gridsplit_repeated_use_does_not_grow_active_memory():
    x_data = np.random.default_rng(774).standard_normal((2, 1, 28, 28)).astype(np.float32)
    w_data = np.random.default_rng(775).standard_normal((8, 1, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(776).standard_normal((2, 8, 28, 28)).astype(np.float32)

    backend = get_cuda_backend()
    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    gc.disable()
    try:
        for _ in range(50):
            x_cuda = Tensor(x_data.copy(), device="cuda")
            grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
            result = dweight_below256_gridsplit(
                backend, grad_out_cuda._data, x_cuda._data, w_data.shape, (1, 1), (1, 1), 2,
            )
            del result, x_cuda, grad_out_cuda
    finally:
        gc.enable()

    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0


def test_gridsplit_allocator_reuse_across_split_counts():
    """Different `num_splits` values allocate different-sized partial
    buffers -- confirms the caching allocator handles repeated
    alloc/free of varying sizes without leaking or growing unbounded."""
    x_data = np.random.default_rng(777).standard_normal((2, 1, 28, 28)).astype(np.float32)
    w_data = np.random.default_rng(778).standard_normal((8, 1, 3, 3)).astype(np.float32)
    grad_out_data = np.random.default_rng(779).standard_normal((2, 8, 28, 28)).astype(np.float32)

    backend = get_cuda_backend()
    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    gc.disable()
    try:
        for _ in range(10):
            for num_splits in (1, 2, 4, 8):
                x_cuda = Tensor(x_data.copy(), device="cuda")
                grad_out_cuda = Tensor(grad_out_data.copy(), device="cuda")
                result = dweight_below256_gridsplit(
                    backend, grad_out_cuda._data, x_cuda._data, w_data.shape, (1, 1), (1, 1), num_splits,
                )
                del result, x_cuda, grad_out_cuda
    finally:
        gc.enable()

    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0
