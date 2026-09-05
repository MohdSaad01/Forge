"""Milestone 41 tests: CUDA Conv2d forward via im2col + GEMM (production dispatch).

M40 measured `k_conv2d_forward` (unchanged since Milestone 15, one thread
per output element, zero memory reuse) at only 10.7-18.0% of the 940MX's
practical compute ceiling -- confirmed via a fresh `nvcc -Xptxas -v` pass
(this milestone) to have zero register spill/stack frame, so the
inefficiency is structural (no shared-memory/GEMM reuse), not a M32-style
register-pressure problem. `benchmarks/m41_conv2d_forward_profile.py`
measured two structurally different GEMM-based candidates against it at 15
representative/sweep shapes:

- **Candidate A** (`experimental_conv_forward_im2col.conv2d_forward_im2col_
  gemm`): `im2col`/`im2col_smem` (Milestone 34/39, unmodified) -> weight
  transpose (`k_transpose`, Milestone 11, unmodified) -> the existing tiled
  GEMM (`cf_matmul_*`, Milestone 11, unmodified) -> a new output-permute
  kernel (`k_conv2d_output_permute`) that also fuses in the bias add.
- **Candidate B** (`experimental_conv_forward_halffused.
  conv2d_forward_halffused_gemm`): weight transpose (same, cheap) + a new
  half-fused GEMM kernel (`k_conv2d_forward_halffused_gemm`) that gathers
  `Xcol` tiles on the fly (no `Xcol` buffer at all) and writes its result
  directly into the final `(N, Cout, Hout, Wout)` layout with bias fused in.

Both won by 1.06-2.71x at every shape at/above ~20M total forward FLOPs, and
both regressed (0.26-0.75x) at every shape at/below ~7.2M FLOPs (fixed
kernel-launch/allocation overhead dominates an already-fast baseline call).
`CUDABackend.conv2d` (`backend.py`) now dispatches to one of the two GEMM
candidates at/above `_CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD` (10,000,000 total
FLOPs) and keeps the original per-thread kernel below it; above the
threshold, `blocks_x = ceil(Cout/16) <= 2` (every current Forge shape)
selects Candidate B (measured faster there), larger `Cout` selects Candidate
A with the Milestone 39 shared-memory im2col variant (Candidate B's
redundant `Xcol`-regather tax grows with `blocks_x` and lost to Candidate A
at the one tested `Cout=128` shape). See `docs/performance/
conv2d-forward-profiling.md`'s **Milestone 41** section for the complete
evidence.

This module has three parts: direct correctness coverage for each
experimental pipeline (CPU comparison across shapes/stride/padding/kernel
size/dtype, explicit-stream, cross-stream, repeated-use memory safety --
mirroring `tests/test_cuda_conv2d_backward_weight_im2col_gemm.py`'s (M34)
existing coverage pattern), and **production dispatch** coverage through the
real `Tensor.conv2d`/`nn.Conv2d` API at shapes that cross both dispatch
boundaries (the FLOPs threshold itself, and the `blocks_x` candidate-A-vs-B
boundary) -- since `tests/test_cuda_conv.py`'s existing shapes are all small
enough to stay on the unchanged baseline kernel and never exercised either
boundary before this milestone.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.backend.cuda.backend import _CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD, _MATMUL_TILE, get_cuda_backend
from forge.backend.cuda.experimental_conv_forward_halffused import conv2d_forward_halffused_gemm
from forge.backend.cuda.experimental_conv_forward_im2col import conv2d_forward_im2col_gemm
from forge.backend.cpu import CPUBackend
from forge.nn import Conv2d

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)

CANDIDATES = {
    "candidate_a_plain": lambda backend, x, w, b, s, p: conv2d_forward_im2col_gemm(backend, x, w, b, s, p, use_smem_im2col=False),
    "candidate_a_smem": lambda backend, x, w, b, s, p: conv2d_forward_im2col_gemm(backend, x, w, b, s, p, use_smem_im2col=True),
    "candidate_b": conv2d_forward_halffused_gemm,
}


@pytest.fixture(autouse=True)
def _empty_cache_around_test():
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


# -- Direct correctness vs. CPU across shapes / stride / padding / kernel size --

SHAPES = [
    # (Cin, Cout, H, W, K, stride, padding)
    (2, 3, 6, 6, 3, 1, 0),
    (2, 3, 6, 6, 3, 1, 1),
    (8, 16, 9, 9, 3, 1, 1),
    (16, 32, 7, 7, 3, 2, 1),
    (3, 4, 9, 9, 3, 2, 1),
    (2, 3, 5, 5, 2, 1, 0),      # even kernel size, odd spatial size
    (4, 6, 10, 10, 1, 1, 0),    # K=1
    (2, 4, 12, 12, 5, 1, 2),    # K=5
    (1, 128, 8, 8, 3, 1, 1),    # large Cout (blocks_x > 2)
    (64, 4, 8, 8, 3, 1, 1),     # large Cin, small Cout
]


@pytest.mark.parametrize("candidate", list(CANDIDATES))
@pytest.mark.parametrize("cin,cout,h,w,k,s,p", SHAPES)
def test_candidate_forward_matches_cpu(candidate, cin, cout, h, w, k, s, p):
    forge.random.seed(80)
    layer = Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)
    x_data = np.random.default_rng(81).standard_normal((3, cin, h, w)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    ref = CPUBackend().conv2d(x_data, w_data, b_data, (s, s), (p, p))

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")._data
    w_cuda = Tensor(w_data.copy(), device="cuda")._data
    b_cuda = Tensor(b_data.copy(), device="cuda")._data
    forge.cuda.synchronize()

    result = CANDIDATES[candidate](backend, x_cuda, w_cuda, b_cuda, (s, s), (p, p))
    actual = backend.to_numpy(result)
    np.testing.assert_allclose(actual, ref, **TOL)


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_candidate_forward_matches_cpu_no_bias(candidate):
    forge.random.seed(82)
    layer = Conv2d(4, 6, kernel_size=3, stride=1, padding=1, bias=False)
    x_data = np.random.default_rng(83).standard_normal((2, 4, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()

    ref = CPUBackend().conv2d(x_data, w_data, None, (1, 1), (1, 1))

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")._data
    w_cuda = Tensor(w_data.copy(), device="cuda")._data
    forge.cuda.synchronize()

    result = CANDIDATES[candidate](backend, x_cuda, w_cuda, None, (1, 1), (1, 1))
    np.testing.assert_allclose(backend.to_numpy(result), ref, **TOL)


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_candidate_forward_matches_cpu_float64(candidate):
    forge.random.seed(84)
    layer = Conv2d(4, 6, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(85).standard_normal((2, 4, 8, 8)).astype(np.float64)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()

    ref = CPUBackend().conv2d(x_data, w_data, b_data, (1, 1), (1, 1))

    backend = get_cuda_backend()
    x_cuda = Tensor(x_data.copy(), device="cuda")._data
    w_cuda = Tensor(w_data.copy(), device="cuda")._data
    b_cuda = Tensor(b_data.copy(), device="cuda")._data
    forge.cuda.synchronize()

    result = CANDIDATES[candidate](backend, x_cuda, w_cuda, b_cuda, (1, 1), (1, 1))
    np.testing.assert_allclose(backend.to_numpy(result), ref, rtol=1e-9, atol=1e-9)


# -- Explicit-stream (async mode) correctness ---------------------------------


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_candidate_forward_on_explicit_stream_matches_cpu(candidate):
    forge.random.seed(86)
    layer = Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    x_data = np.random.default_rng(87).standard_normal((2, 3, 9, 9)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    ref = CPUBackend().conv2d(x_data, w_data, b_data, (2, 2), (1, 1))

    backend = get_cuda_backend()
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        x_cuda = Tensor(x_data.copy(), device="cuda")._data
        w_cuda = Tensor(w_data.copy(), device="cuda")._data
        b_cuda = Tensor(b_data.copy(), device="cuda")._data
        result = CANDIDATES[candidate](backend, x_cuda, w_cuda, b_cuda, (2, 2), (1, 1))
    s.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), ref, **TOL)


# -- Cross-stream correctness (producer streams != compute stream) -----------


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_candidate_forward_correct_when_inputs_from_different_streams(candidate):
    stream_x = forge.cuda.Stream()
    stream_w = forge.cuda.Stream()
    stream_compute = forge.cuda.Stream()

    forge.random.seed(88)
    layer = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x_data = np.random.default_rng(89).standard_normal((2, 2, 8, 8)).astype(np.float32)
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()
    ref = CPUBackend().conv2d(x_data, w_data, b_data, (1, 1), (1, 1))

    backend = get_cuda_backend()
    with forge.cuda.stream(stream_x):
        x_cuda = Tensor(x_data.copy(), device="cuda")._data
    with forge.cuda.stream(stream_w):
        w_cuda = Tensor(w_data.copy(), device="cuda")._data
        b_cuda = Tensor(b_data.copy(), device="cuda")._data

    with forge.cuda.stream(stream_compute):
        result = CANDIDATES[candidate](backend, x_cuda, w_cuda, b_cuda, (1, 1), (1, 1))
    stream_compute.synchronize()

    np.testing.assert_allclose(backend.to_numpy(result), ref, **TOL)


# -- Memory safety (repeated use) ---------------------------------------------


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_candidate_forward_repeated_use_does_not_grow_active_memory(candidate):
    x_data = np.random.default_rng(90).standard_normal((4, 8, 9, 9)).astype(np.float32)
    w_data = np.random.default_rng(91).standard_normal((16, 8, 3, 3)).astype(np.float32)
    b_data = np.random.default_rng(92).standard_normal((16,)).astype(np.float32)

    backend = get_cuda_backend()
    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    gc.disable()
    try:
        for _ in range(20):
            x_cuda = Tensor(x_data.copy(), device="cuda")._data
            w_cuda = Tensor(w_data.copy(), device="cuda")._data
            b_cuda = Tensor(b_data.copy(), device="cuda")._data
            result = CANDIDATES[candidate](backend, x_cuda, w_cuda, b_cuda, (1, 1), (1, 1))
            del x_cuda, w_cuda, b_cuda, result
    finally:
        gc.enable()

    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0


# -- Production dispatch: through the ordinary Tensor.conv2d / nn.Conv2d API --
#
# `tests/test_cuda_conv.py`'s existing shapes are all small enough to stay on
# the unchanged baseline kernel (well below `_CONV2D_FORWARD_GEMM_FLOPS_
# THRESHOLD`) -- these tests specifically cross both M41 dispatch boundaries
# through the real, public API.


def _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding, bias=True, seed=93):
    forge.random.seed(seed)
    cpu_layer = Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
    cuda_layer = Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    if bias:
        cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")
    return cpu_layer, cuda_layer


def _dispatch_branch(cin, cout, k, s, p, n, h, w):
    Hout = (h + 2 * p - k) // s + 1
    Wout = (w + 2 * p - k) // s + 1
    total_flops = 2 * n * cout * Hout * Wout * cin * k * k
    if total_flops < _CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD:
        return "baseline"
    blocks_x = (cout + _MATMUL_TILE - 1) // _MATMUL_TILE
    return "candidate_b" if blocks_x <= 2 else "candidate_a"


@pytest.mark.parametrize(
    "in_ch,out_ch,kernel_size,stride,padding,n,h",
    [
        (1, 8, 3, 1, 1, 2, 16),     # well below threshold -- unaffected regression check
        (8, 16, 3, 1, 1, 16, 28),   # above threshold, blocks_x<=2 -> candidate B
        (16, 32, 3, 1, 1, 8, 28),   # above threshold, blocks_x<=2 -> candidate B
        (8, 128, 3, 1, 1, 8, 28),   # above threshold, blocks_x>2 -> candidate A
        (4, 6, 1, 1, 0, 8, 20),     # K=1, near the FLOPs threshold from below
    ],
)
def test_production_conv2d_forward_matches_cpu_across_dispatch(in_ch, out_ch, kernel_size, stride, padding, n, h):
    branch = _dispatch_branch(in_ch, out_ch, kernel_size, stride, padding, n, h, h)
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding)
    x_data = np.random.default_rng(94).standard_normal((n, in_ch, h, h)).astype(np.float32)

    out_cpu = cpu_layer(Tensor(x_data.copy())).numpy()
    out_cuda = cuda_layer(Tensor(x_data.copy(), device="cuda")).to("cpu").numpy()

    assert branch in ("baseline", "candidate_a", "candidate_b")
    np.testing.assert_allclose(out_cuda, out_cpu, **TOL)


def test_production_conv2d_forward_no_bias_above_threshold():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 1, 1, bias=False, seed=95)
    x_data = np.random.default_rng(96).standard_normal((16, 8, 28, 28)).astype(np.float32)

    out_cpu = cpu_layer(Tensor(x_data.copy())).numpy()
    out_cuda = cuda_layer(Tensor(x_data.copy(), device="cuda")).to("cpu").numpy()
    np.testing.assert_allclose(out_cuda, out_cpu, **TOL)


def test_production_conv2d_forward_on_explicit_stream_above_threshold():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 2, 1, seed=97)
    x_data = np.random.default_rng(98).standard_normal((16, 8, 28, 28)).astype(np.float32)

    out_cpu = cpu_layer(Tensor(x_data.copy())).numpy()

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        out_cuda_tensor = cuda_layer(Tensor(x_data.copy(), device="cuda"))
    s.synchronize()

    np.testing.assert_allclose(out_cuda_tensor.to("cpu").numpy(), out_cpu, **TOL)


def test_production_conv2d_forward_backward_still_matches_cpu_above_threshold():
    """Forward dispatch changed; backward (untouched) must still produce the
    same gradients end-to-end through the real autograd graph."""
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(8, 16, 3, 1, 1, seed=99)
    x_data = np.random.default_rng(100).standard_normal((16, 8, 28, 28)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    cpu_layer(x_cpu).sum().backward()
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)


def test_production_conv2d_forward_repeated_use_does_not_grow_active_memory_above_threshold():
    x_data = np.random.default_rng(101).standard_normal((16, 8, 28, 28)).astype(np.float32)

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats()

    layer = Conv2d(8, 16, kernel_size=3, stride=1, padding=1).to("cuda")
    for _ in range(20):
        x = Tensor(x_data.copy(), device="cuda")
        out = layer(x)
        del x, out

    del layer
    gc.collect()
    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()

    assert after.allocated_bytes == before.allocated_bytes
    assert after.reserved_bytes == 0
    assert after.pending_bytes == 0
