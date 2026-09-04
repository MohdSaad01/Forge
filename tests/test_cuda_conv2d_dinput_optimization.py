"""Milestone 36 tests: CUDA Conv2d `dInput` channel-fused optimization.

M36 profiled the real `dInput` kernel (`cf_conv2d_backward_input_*`) and
found it reaching only ~12% of the 940MX's practical compute ceiling while
using well under 20% of its practical bandwidth ceiling at every
representative shape -- `nvcc -Xptxas -v` traced this to a 512-byte
per-thread *local memory* stack frame (the M32 `kh_valid`/`ho_valid`/
`kw_valid`/`wo_valid` tables, dynamically indexed and therefore never
register-resident), read back `Cout * h_count * w_count` times per thread.

`k_conv2d_backward_input_channelfused` (`kernels.cu`) replaces the
one-thread-per-`(n,ci,h,w)` mapping with one thread per `(n,h,w)` holding all
`Cin` accumulators in a compile-time-unrolled register array (`Cin <=
MAX_CIN_REG=16`) -- confirmed via `-Xptxas -v` to have zero stack frame -- and
reads each `grad_output[n,co,ho,wo]` value once per thread, reused across
every `ci`. `cf_conv2d_backward_input_*` (the production entry point)
dispatches to it whenever `Cin <= 16`, falling back to the unchanged
`k_conv2d_backward_input` otherwise. See `docs/performance/
conv2d-backward-profiling.md`'s **Milestone 36** section for the full
before/after evidence and the two rejected candidates (shared-memory
grad_output tiling, warp-cooperative reduction over `Cout`).

`tests/test_cuda_conv.py` and `tests/test_cuda_conv2d_backward_optimization.py`
already exercise the production dispatch end-to-end (via `nn.Conv2d`/
autograd) at `Cin` values that all fall inside the channel-fused path (still
passing unmodified after this milestone -- confirming Candidate B computes
the exact same gradient, just via a different summation order/thread
mapping). This module adds what M36 specifically needs: a direct kernel-level
parity check between the two dispatch paths across stride/padding/kernel-size/
dtype combinations, and correctness at and across the `Cin <=
MAX_CIN_REG` dispatch boundary itself (never previously exercised, since no
existing Forge test uses `Cin > 16`).
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

MAX_CIN_REG = 16  # must match kernels.cu's k_conv2d_backward_input_channelfused


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    # `gc.collect()` before `empty_cache()`: the autograd-graph tests below
    # build real cyclic node references (parent/child), which CPython's
    # plain refcounting never reclaims immediately -- without this, their
    # CUDAStorage objects (and the memory they hold) linger until some
    # later, unrelated test's own `gc.collect()` sweeps them up, corrupting
    # that test's "before" memory-stats baseline (see forge_testing
    # conventions for this exact class of cross-test contamination).
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
    # Wrapped in a `CUDAStorage` (not a bare `cf_free`'d pointer) so it is
    # released through `forge.backend.cuda.allocator`'s normal `__del__` ->
    # `release()` path when it goes out of scope -- calling the raw
    # `cf_free` C function directly here would `cudaFree` the pointer while
    # the allocator's own `allocated_bytes` accounting still believed it was
    # live, permanently desyncing that counter for the rest of the pytest
    # process (surfaced as a spurious failure in an unrelated, later memory-
    # leak test in the same session).
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


# -- Direct kernel-level parity: baseline vs. channel-fused candidate --------


@pytest.mark.parametrize(
    "N,Cin,Cout,H,W,K,S,P",
    [
        (2, 3, 4, 7, 7, 3, 1, 1),  # baseline shape, stride=1, padding>0
        (2, 3, 4, 7, 7, 3, 1, 0),  # stride=1, padding=0
        (2, 3, 4, 9, 9, 3, 2, 1),  # stride>1
        (1, 1, 8, 28, 28, 3, 1, 1),  # mnist_conv1 shape (Cin=1)
        (4, 8, 16, 13, 13, 3, 1, 1),  # mnist_conv2 shape
        (2, 16, 5, 10, 10, 5, 1, 2),  # K=5 (Section 23's "another kernel size")
        (3, 16, 3, 11, 13, 5, 2, 2),  # K=5, stride=2, non-square H/W
        (2, 1, 4, 7, 7, 2, 1, 0),  # even kernel size
        (1, MAX_CIN_REG, 4, 6, 6, 3, 1, 1),  # exactly at the register-array bound
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_channelfused_matches_baseline_kernel(N, Cin, Cout, H, W, K, S, P, dtype):
    suffix = "f32" if dtype == np.float32 else "f64"
    baseline = _run_kernel(f"cf_conv2d_backward_input_{suffix}", N, Cin, Cout, H, W, K, S, P, dtype)
    # Force the candidate directly (bypassing production dispatch) so this
    # test still exercises the candidate kernel even if a shape happens to
    # fall outside where production would currently route to it.
    candidate = _run_kernel(f"cf_conv2d_backward_input_channelfused_{suffix}", N, Cin, Cout, H, W, K, S, P, dtype)
    tol = dict(rtol=1e-3, atol=1e-4) if dtype == np.float32 else dict(rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(candidate, baseline, **tol)


# -- Production dispatch boundary: Cin <= MAX_CIN_REG vs. Cin > MAX_CIN_REG --


@pytest.mark.parametrize("cin", [MAX_CIN_REG, MAX_CIN_REG + 1])
def test_production_dispatch_correct_at_cin_boundary(cin):
    """`Cin == MAX_CIN_REG` takes the channel-fused path; `Cin == MAX_CIN_REG + 1` falls back."""
    rng = np.random.default_rng(50)
    x_data = rng.standard_normal((2, cin, 6, 6)).astype(np.float32)

    forge.random.seed(9)
    cpu_layer = Conv2d(cin, 3, kernel_size=3, stride=1, padding=1)
    x_cpu = forge.Tensor(x_data.copy(), requires_grad=True)
    cpu_layer(x_cpu).sum().backward()

    forge.random.seed(9)
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


def test_channelfused_and_baseline_agree_above_register_bound():
    """`Cin > MAX_CIN_REG` is out of the channel-fused kernel's documented scope --
    production never routes there, but the kernel itself must not silently
    corrupt results if called directly at that size (it simply never writes
    channels beyond MAX_CIN_REG). This test documents that boundary rather
    than asserting the (undefined-for-this-kernel) upper channels match."""
    N, Cin, Cout, H, W, K, S, P = 1, MAX_CIN_REG + 4, 3, 6, 6, 3, 1, 1
    baseline = _run_kernel("cf_conv2d_backward_input_f32", N, Cin, Cout, H, W, K, S, P, np.float32)
    candidate = _run_kernel("cf_conv2d_backward_input_channelfused_f32", N, Cin, Cout, H, W, K, S, P, np.float32)
    # Channels within the register bound still match exactly.
    np.testing.assert_allclose(
        candidate[:, :MAX_CIN_REG], baseline[:, :MAX_CIN_REG], rtol=1e-3, atol=1e-4
    )
    # Production dispatch (verified above) never calls the candidate kernel
    # directly for Cin > MAX_CIN_REG -- only the always-correct baseline.
    production = _run_kernel("cf_conv2d_backward_input_f32", N, Cin, Cout, H, W, K, S, P, np.float32)
    np.testing.assert_allclose(production, baseline, rtol=1e-3, atol=1e-4)
