"""Milestone 37 tests: rejected fused-gather GEMM `dWeight` candidates A/C.

`forge.backend.cuda.experimental_conv_fused` holds two `dWeight` candidates
measured and **rejected** for production during M37 (see that module's
docstring and `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 37** section): folding `im2col`/`grad_output_permute`'s gathers
directly into the GEMM's tile-load step measured *slower* than the M34
baseline at every shape with >= 18 GEMM blocks, so neither is wired into
`CUDABackend.conv2d_backward`. Per the milestone brief's Phase 4 ("verify
numerical correctness against CPUBackend" for every serious candidate),
this module keeps minimal correctness coverage for both -- guarding against
silent bit rot in shipped-but-unused profiling code, not exhaustive
production-grade coverage (that level of coverage lives in
`tests/test_cuda_conv2d_backward_weight_splitk_gemm.py`, for the candidate
that *was* selected).
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import get_cuda_backend
from forge.backend.cuda.experimental_conv_fused import dweight_fused_gemm, dweight_fused_gemm_splitk
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)

SHAPES = [
    # (Cin, Cout, H, W, K, stride, padding)
    (2, 3, 6, 6, 3, 1, 0),
    (8, 16, 9, 9, 3, 1, 1),
    (16, 32, 7, 7, 3, 2, 1),
    (1, 1, 5, 5, 3, 1, 1),
]


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _cpu_grad_w(x_data, w_data, b_data, stride, padding, upstream):
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    w_cpu = Tensor(w_data.copy(), requires_grad=True)
    b_cpu = Tensor(b_data.copy(), requires_grad=True)
    x_cpu.conv2d(w_cpu, b_cpu, stride, padding).backward(Tensor(upstream.copy()))
    return w_cpu.grad.numpy()


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_candidate_a_fused_gemm_matches_cpu(cin, cout, h, w, k, s, p):
    forge.random.seed(250)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(251).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    out_shape = Tensor(x_data).conv2d(Tensor(w_data), Tensor(b_data), (s, s), (p, p)).shape
    upstream = np.random.default_rng(252).standard_normal(out_shape).astype(np.float32)
    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p), upstream)

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_fused_gemm(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (s, s), (p, p))
    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
@pytest.mark.parametrize("num_k_splits", [1, 4, 16])
def test_candidate_c_fused_gemm_splitk_matches_cpu(cin, cout, h, w, k, s, p, num_k_splits):
    forge.random.seed(253)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(254).standard_normal((2, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    out_shape = Tensor(x_data).conv2d(Tensor(w_data), Tensor(b_data), (s, s), (p, p)).shape
    upstream = np.random.default_rng(255).standard_normal(out_shape).astype(np.float32)
    expected = _cpu_grad_w(x_data, w_data, b_data, (s, s), (p, p), upstream)

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_fused_gemm_splitk(
        backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (s, s), (p, p), num_k_splits
    )
    np.testing.assert_allclose(backend.to_numpy(result), expected, **TOL)


def test_candidate_a_fused_gemm_matches_cpu_float64():
    forge.random.seed(256)
    layer = Conv2d(4, 6, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(257).standard_normal((2, 4, 8, 8)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()
    out_shape = Tensor(x_data).conv2d(Tensor(w_data), Tensor(b_data), (1, 1), (1, 1)).shape
    upstream = np.random.default_rng(258).standard_normal(out_shape).astype(np.float64)
    expected = _cpu_grad_w(x_data, w_data, b_data, (1, 1), (1, 1), upstream)

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")
    w_cuda = Tensor(w_data.copy(), device="cuda")
    grad_out_cuda = Tensor(upstream.copy(), device="cuda")
    forge.cuda.synchronize()

    result = dweight_fused_gemm(backend, grad_out_cuda._data, x_cuda._data, w_cuda._data.shape, (1, 1), (1, 1))
    np.testing.assert_allclose(backend.to_numpy(result), expected, rtol=1e-8, atol=1e-8)
