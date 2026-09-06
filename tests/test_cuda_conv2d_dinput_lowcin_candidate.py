"""Milestone 48 tests: CUDA Conv2d `dInput` low-Cin-specialized kernel
(accepted; production).

M47 found the M36 channel-fused `dInput` kernel (`k_conv2d_backward_input_
channelfused`) drops from 29.0% of the practical compute ceiling at `Cin=16`
to only 4.7% at `Cin=1` -- `mnist_conv1`'s own real shape, Forge's only real
`Cin=1` workload (`examples/mnist/model.py`'s first conv layer). M48's own
`nvcc -Xptxas -v`/occupancy investigation found this is *not* primarily a
register-pressure/occupancy problem (f32 register count only drops 54->48,
occupancy only rises 4->5 blocks/SM going from `MAX_CIN_REG=16` to a
`Cin=1`-specialized accumulator array) but per-thread instruction overhead:
the channel-fused kernel's `#pragma unroll`ed accumulator loop still emits
16 runtime-checked iterations at every `Cout*KH*KW` step even when only the
first is ever useful at `Cin=1`.

`k_conv2d_backward_input_channelfused_lowcin<T, CIN_MAX>` (`kernels.cu`) is
the identical kernel body with `CIN_MAX` as a real template parameter,
instantiated at `CIN_MAX=1`. A same-session, round-robin-interleaved,
order-independent A/B (`benchmarks/m48_dinput_value_assessment.py`) measured
a reproducible 2.24x-2.62x isolated speedup over the unspecialized
channel-fused kernel at `Cin=1`, bit-exact correct. `cf_conv2d_backward_
input_*` (the production entry point) now dispatches to it whenever `Cin <=
CONV2D_DINPUT_LOWCIN_MAX_CIN` (1) -- the one real Forge shape it was
measured at -- ahead of the unchanged M36 channel-fused path (`Cin <= 16`)
and the M32 fallback (`Cin > 16`). See `docs/performance/
m48-dinput-value-assessment.md` for the complete evidence and Amdahl
analysis.

`tests/test_cuda_conv2d_dinput_optimization.py` (M36) already exercises the
production dispatch end-to-end at `Cin=1` (still passing unmodified after
this milestone, now exercising the new lowcin1 path instead of the M36
channel-fused path it previously reached) -- this module adds what M48
specifically needs: a direct kernel-level parity check between the
lowcin1 candidate and the unspecialized channel-fused kernel, correctness at
and across the new `Cin <= CONV2D_DINPUT_LOWCIN_MAX_CIN` dispatch boundary,
and memory-lifecycle/stream coverage for the newly-promoted kernel.
"""

from __future__ import annotations

import ctypes
import gc

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import CUDAStorage, get_cuda_backend
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

CONV2D_DINPUT_LOWCIN_MAX_CIN = 1  # must match kernels.cu's dispatch


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    gc.collect()
    forge.cuda.empty_cache()


def _hout(H: int, K: int, S: int, P: int) -> int:
    return (H + 2 * P - K) // S + 1


def _run_kernel(fn_name: str, N, Cin, Cout, H, W, K, S, P, dtype) -> np.ndarray:
    backend = get_cuda_backend()
    lib = backend._lib
    Hout, Wout = _hout(H, K, S, P), _hout(W, K, S, P)

    rng = np.random.default_rng(42)
    w_np = rng.standard_normal((Cout, Cin, K, K)).astype(dtype)
    go_np = rng.standard_normal((N, Cout, Hout, Wout)).astype(dtype)

    w = forge.Tensor(w_np, device="cuda")._data
    go = forge.Tensor(go_np, device="cuda")._data
    forge.cuda.synchronize()

    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
        ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    gx_ptr = backend._alloc(N * Cin * H * W * w_np.itemsize)
    gx_storage = CUDAStorage(gx_ptr, (N, Cin, H, W), dtype, lib)
    fn = getattr(lib, fn_name)
    code = fn(go.ptr, w.ptr, gx_ptr, *shape_args, None)
    assert code == 0, f"{fn_name} launch failed with code {code}"
    forge.cuda.synchronize()

    out = np.empty((N, Cin, H, W), dtype=dtype)
    lib.cf_memcpy_d2h(out.ctypes.data_as(ctypes.c_void_p), gx_ptr, ctypes.c_size_t(out.nbytes))
    del gx_storage
    return out


# -- Direct kernel-level parity: channel-fused (M36) vs. lowcin1 (M48) -------


@pytest.mark.parametrize(
    "N,Cout,H,W,K,S,P",
    [
        (2, 4, 7, 7, 3, 1, 1),   # baseline shape, stride=1, padding>0
        (2, 4, 7, 7, 3, 1, 0),   # stride=1, padding=0
        (2, 4, 9, 9, 3, 2, 1),   # stride>1
        (64, 8, 28, 28, 3, 1, 1),  # mnist_conv1's own exact shape
        (1, 27, 28, 28, 3, 1, 1),  # we_243-style, high Cout
        (2, 5, 10, 10, 5, 1, 2),   # K=5
        (3, 3, 11, 13, 5, 2, 2),   # K=5, stride=2, non-square H/W
        (2, 4, 7, 7, 2, 1, 0),     # even kernel size
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lowcin1_matches_channelfused_kernel(N, Cout, H, W, K, S, P, dtype):
    Cin = 1
    suffix = "f32" if dtype == np.float32 else "f64"
    baseline = _run_kernel(f"cf_conv2d_backward_input_channelfused_{suffix}", N, Cin, Cout, H, W, K, S, P, dtype)
    candidate = _run_kernel(f"cf_conv2d_backward_input_channelfused_lowcin1_{suffix}", N, Cin, Cout, H, W, K, S, P, dtype)
    tol = dict(rtol=1e-3, atol=1e-4) if dtype == np.float32 else dict(rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(candidate, baseline, **tol)


def test_lowcin1_matches_baseline_original_kernel():
    """Also parity-check against the original (pre-M36) `k_conv2d_backward_input`,
    not just the M36 channel-fused kernel -- transitivity isn't assumed."""
    N, Cin, Cout, H, W, K, S, P = 64, 1, 8, 28, 28, 3, 1, 1
    original = _run_kernel("cf_conv2d_backward_input_f32", N, Cin, Cout, H, W, K, S, P, np.float32)
    candidate = _run_kernel("cf_conv2d_backward_input_channelfused_lowcin1_f32", N, Cin, Cout, H, W, K, S, P, np.float32)
    np.testing.assert_allclose(candidate, original, rtol=1e-3, atol=1e-4)


# -- Production dispatch boundary: Cin<=1 (lowcin1) vs. Cin==2 (channelfused) --


@pytest.mark.parametrize("cin", [CONV2D_DINPUT_LOWCIN_MAX_CIN, CONV2D_DINPUT_LOWCIN_MAX_CIN + 1])
def test_production_dispatch_correct_at_lowcin_boundary(cin):
    """`Cin == 1` takes the new lowcin1 path; `Cin == 2` takes the unchanged M36
    channel-fused path -- both must match a CPU reference through the real
    `nn.Conv2d`/autograd API."""
    rng = np.random.default_rng(51)
    x_data = rng.standard_normal((2, cin, 6, 6)).astype(np.float32)

    forge.random.seed(11)
    cpu_layer = Conv2d(cin, 3, kernel_size=3, stride=1, padding=1)
    x_cpu = forge.Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    forge.random.seed(11)
    cuda_layer = Conv2d(cin, 3, kernel_size=3, stride=1, padding=1)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")

    x_cuda = forge.Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("cin,cout,n,h,w,k,s,p", [
    (1, 8, 64, 28, 28, 3, 1, 1),   # mnist_conv1 exact shape
    (1, 4, 4, 10, 12, 3, 2, 1),    # smaller, strided
    (1, 16, 8, 13, 13, 1, 1, 0),   # 1x1 kernel
    (1, 27, 2, 9, 9, 3, 1, 0),     # high Cout, no padding
])
def test_production_dispatch_lowcin1_matches_cpu_across_shapes(cin, cout, n, h, w, k, s, p):
    rng = np.random.default_rng(52)
    x_data = rng.standard_normal((n, cin, h, w)).astype(np.float32)

    forge.random.seed(12)
    cpu_layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_cpu = forge.Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    forge.random.seed(12)
    cuda_layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")

    x_cuda = forge.Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), rtol=1e-4, atol=1e-4
    )


def test_lowcin1_f64_matches_cpu():
    rng = np.random.default_rng(53)
    x_data = rng.standard_normal((2, 1, 8, 8)).astype(np.float64)

    forge.random.seed(13)
    cpu_layer = Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
    cpu_layer.weight._data = cpu_layer.weight._data.astype(np.float64)
    cpu_layer.bias._data = cpu_layer.bias._data.astype(np.float64)
    x_cpu = forge.Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    cuda_layer = Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")

    x_cuda = forge.Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), rtol=1e-8, atol=1e-10)


# -- Explicit stream / cross-stream correctness -------------------------------


def test_lowcin1_on_explicit_stream_matches_cpu():
    rng = np.random.default_rng(54)
    x_data = rng.standard_normal((2, 1, 8, 8)).astype(np.float32)

    forge.random.seed(14)
    cpu_model = Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
    x_cpu = forge.Tensor(x_data.copy(), requires_grad=True)
    cpu_model(x_cpu).sum().backward()

    forge.random.seed(14)
    cuda_model = Conv2d(1, 4, kernel_size=3, stride=1, padding=1).to("cuda")
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = forge.Tensor(x_data.copy(), device="cuda", requires_grad=True)
        cuda_model(x_cuda).sum().backward()
    s.synchronize()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), rtol=1e-3, atol=1e-3)


def test_lowcin1_correct_when_inputs_from_different_streams():
    stream_x = forge.cuda.Stream()
    stream_g = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    backend = get_cuda_backend()
    lib = backend._lib

    N, Cin, Cout, H, W, K, S, P = 2, 1, 5, 8, 8, 3, 1, 1
    Hout, Wout = _hout(H, K, S, P), _hout(W, K, S, P)
    rng = np.random.default_rng(55)
    w_np = rng.standard_normal((Cout, Cin, K, K)).astype(np.float32)
    go_np = rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32)

    with forge.cuda.stream(stream_x):
        w = forge.Tensor(w_np.copy(), device="cuda")._data
    with forge.cuda.stream(stream_g):
        go = forge.Tensor(go_np.copy(), device="cuda")._data

    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
        ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    with forge.cuda.stream(stream_compute):
        gx_ptr = backend._alloc(N * Cin * H * W * 4)
        gx_storage = CUDAStorage(gx_ptr, (N, Cin, H, W), np.float32, lib)
        fn = getattr(lib, "cf_conv2d_backward_input_channelfused_lowcin1_f32")
        code = fn(go.ptr, w.ptr, gx_ptr, *shape_args, stream_compute.handle)
        assert code == 0
    stream_compute.synchronize()

    actual = backend.to_numpy(gx_storage).copy()
    del gx_storage

    x_cpu = forge.Tensor(np.zeros((N, Cin, H, W), dtype=np.float32), requires_grad=True)
    w_cpu = forge.Tensor(w_np.copy(), requires_grad=True)
    b_cpu = forge.Tensor(np.zeros((Cout,), dtype=np.float32), requires_grad=True)
    out_cpu = x_cpu.conv2d(w_cpu, b_cpu, (S, S), (P, P))
    out_cpu.backward(forge.Tensor(go_np.copy()))
    expected = x_cpu.grad.numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


# -- Memory safety (repeated use, allocator reuse) ----------------------------


def test_lowcin1_repeated_use_does_not_grow_active_memory():
    backend = get_cuda_backend()
    lib = backend._lib
    N, Cin, Cout, H, W, K, S, P = 8, 1, 8, 16, 16, 3, 1, 1
    Hout, Wout = _hout(H, K, S, P), _hout(W, K, S, P)

    rng = np.random.default_rng(56)
    w = forge.Tensor(rng.standard_normal((Cout, Cin, K, K)).astype(np.float32), device="cuda")._data
    go = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")._data
    forge.cuda.synchronize()
    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
        ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    fn = getattr(lib, "cf_conv2d_backward_input_channelfused_lowcin1_f32")

    before = forge.cuda.memory_stats()
    for _ in range(20):
        gx_ptr = backend._alloc(N * Cin * H * W * 4)
        gx_storage = CUDAStorage(gx_ptr, (N, Cin, H, W), np.float32, lib)
        code = fn(go.ptr, w.ptr, gx_ptr, *shape_args, None)
        assert code == 0
        forge.cuda.synchronize()
        del gx_storage
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes


def test_lowcin1_allocator_reuse_across_repeated_calls():
    """Repeated same-size allocations across calls should hit the caching
    allocator, not grow total cached memory -- mirrors the reuse-safety
    convention this codebase uses for every new CUDA kernel."""
    backend = get_cuda_backend()
    lib = backend._lib
    N, Cin, Cout, H, W, K, S, P = 4, 1, 4, 12, 12, 3, 1, 1
    Hout, Wout = _hout(H, K, S, P), _hout(W, K, S, P)

    rng = np.random.default_rng(57)
    w = forge.Tensor(rng.standard_normal((Cout, Cin, K, K)).astype(np.float32), device="cuda")._data
    go = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")._data
    forge.cuda.synchronize()
    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
        ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )
    fn = getattr(lib, "cf_conv2d_backward_input_channelfused_lowcin1_f32")

    gx_ptr = backend._alloc(N * Cin * H * W * 4)
    gx_storage = CUDAStorage(gx_ptr, (N, Cin, H, W), np.float32, lib)
    code = fn(go.ptr, w.ptr, gx_ptr, *shape_args, None)
    assert code == 0
    forge.cuda.synchronize()
    del gx_storage

    before_hits = forge.cuda.memory_stats().cache_hit_count
    for _ in range(10):
        gx_ptr2 = backend._alloc(N * Cin * H * W * 4)
        gx_storage2 = CUDAStorage(gx_ptr2, (N, Cin, H, W), np.float32, lib)
        code = fn(go.ptr, w.ptr, gx_ptr2, *shape_args, None)
        assert code == 0
        forge.cuda.synchronize()
        del gx_storage2
    after_hits = forge.cuda.memory_stats().cache_hit_count

    assert after_hits > before_hits


# -- Occupancy / register-count sanity (guards the M48 diagnostic claim) -----


def test_lowcin1_occupancy_export_reports_at_least_production():
    """Real `cudaOccupancyMaxActiveBlocksPerMultiprocessor` query -- guards
    against a future kernels.cu change silently regressing the occupancy
    finding this milestone's acceptance decision was based on."""
    backend = get_cuda_backend()
    lib = backend._lib

    out_prod = ctypes.c_int(0)
    code1 = lib.cf_occupancy_conv2d_backward_input_channelfused_f32(ctypes.byref(out_prod))
    out_low = ctypes.c_int(0)
    code2 = lib.cf_occupancy_conv2d_backward_input_channelfused_lowcin1_f32(ctypes.byref(out_low))
    assert code1 == 0 and code2 == 0
    assert out_low.value >= out_prod.value
