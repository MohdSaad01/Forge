"""Conv2d forward via im2col + the existing tiled GEMM (Milestone 41, Candidate A).

M40 found `k_conv2d_forward` (unchanged since Milestone 15, one thread per
output element, zero memory reuse) achieving only 10.7-18.0% of Forge's
practical compute ceiling -- meaningfully less efficient than either backward
Conv2d kernel despite identical FLOP counts, and recommended applying the
same im2col + existing tiled GEMM technique M34 already validated for
`dWeight`. This module is that reformulation, oriented for forward:

    Xcol    = im2col(x)                 (M, K) = (N*Hout*Wout, Cin*KH*KW)
    weightT = transpose(weight)         (K, Cout) -- weight is (Cout, K) contiguous
    out_mat = Xcol @ weightT            (M, Cout), via the EXISTING unmodified k_matmul
    out     = permute(out_mat) + bias   (N, Cout, Hout, Wout)

## Orientation (verified against `CPUBackend.conv2d`)

`CPUBackend.conv2d` (`forge/backend/cpu.py`) computes exactly this:
`cols @ w_flat.T` producing `(N, Hout*Wout, Cout)`, then
`.transpose(0, 2, 1).reshape(N, Cout, Hout, Wout)`. This module reproduces
that same mathematical orientation on the GPU, but avoids NumPy-style lazy
transposes (which `k_matmul` cannot consume -- it requires literal row-major
operands, no transpose flag): `weight` is `(Cout, Cin*KH*KW)` contiguous, so
producing `weightT` as literal `(Cin*KH*KW, Cout)` memory needs one real
transpose (`k_transpose`, Milestone 11, unmodified -- cheap, since
`Cout*Cin*KH*KW` is Forge's small weight-element count, always far smaller
than `Xcol`). `out_mat = Xcol @ weightT` then lands in `(M, Cout)` row-major,
which needs the same "reshape+permute" `CPUBackend.conv2d` does via NumPy's
`.transpose(0, 2, 1)` -- done here by the one genuinely new kernel this
module needs, `k_conv2d_output_permute` (`kernels.cu`), which also fuses in
the bias add so no fifth kernel launch is needed.

## No hidden synchronization

`im2col`, `transpose`, the GEMM call, and `output_permute` are all issued on
the current Forge stream (`backend._stream_handle()`) with no
`cudaDeviceSynchronize()`/`cudaStreamSynchronize()` between them -- ordinary
CUDA stream program order (all four launches enqueued on the same stream) is
what keeps each stage's writes visible to the next stage's reads, not an
inserted host-side wait. `conv2d_forward_im2col_gemm` calls
`backend._maybe_synchronize()` exactly once, at the very end -- a no-op on
an explicit `CUDAStream` (async mode), matching every other `CUDABackend`
method's contract.

## Production status

See `docs/performance/conv2d-forward-profiling.md`'s **Milestone 41**
section for the measured benchmark comparison against both the existing
production `k_conv2d_forward` and this module's sibling
`experimental_conv_forward_halffused` (Candidate B).
"""

from __future__ import annotations

import ctypes
from typing import Any

from .backend import CUDAStorage, _SUFFIX
from .experimental_conv_im2col import im2col
from .experimental_conv_im2col_reuse import im2col_smem, im2col_smem_fits


def transpose_weight(backend: Any, weight: CUDAStorage, Cout: int, K: int) -> CUDAStorage:
    """`weightT`, shape `(K, Cout)`, via the existing generic `k_transpose` (Milestone 11)."""
    dtype = weight.dtype
    out_ptr = backend._alloc(K * Cout * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_transpose_{_SUFFIX[dtype]}")
    code = fn(weight.ptr, out_ptr, ctypes.c_longlong(Cout), ctypes.c_longlong(K), backend._stream_handle())
    backend._check(code, "weight transpose (conv2d forward im2col+GEMM, Milestone 41)")
    return CUDAStorage(out_ptr, (K, Cout), dtype, backend._lib)


def output_permute(
    backend: Any, out_mat: CUDAStorage, bias: "CUDAStorage | None",
    N: int, Cout: int, Hout: int, Wout: int,
) -> CUDAStorage:
    """`out`, shape `(N, Cout, Hout, Wout)`, gathered from `out_mat`'s `(M, Cout)`
    row-major layout via the new `k_conv2d_output_permute`, with bias fused in."""
    dtype = out_mat.dtype
    out_ptr = backend._alloc(N * Cout * Hout * Wout * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_conv2d_output_permute_{_SUFFIX[dtype]}")
    code = fn(
        out_mat.ptr, bias.ptr if bias is not None else None, out_ptr,
        ctypes.c_int(N), ctypes.c_int(Cout), ctypes.c_int(Hout), ctypes.c_int(Wout),
        ctypes.c_int(1 if bias is not None else 0),
        backend._stream_handle(),
    )
    backend._check(code, "output permute (conv2d forward im2col+GEMM, Milestone 41)")
    return CUDAStorage(out_ptr, (N, Cout, Hout, Wout), dtype, backend._lib)


def conv2d_forward_im2col_gemm(
    backend: Any, x: CUDAStorage, weight: CUDAStorage, bias: "CUDAStorage | None",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
    use_smem_im2col: bool = True,
) -> CUDAStorage:
    """The complete Candidate A forward pipeline: im2col -> weight transpose
    -> existing tiled GEMM -> output permute (+bias).

    `use_smem_im2col` mirrors the current production `dWeight` dispatch
    (`dweight_im2col_smem_gemm_splitk`): prefer the Milestone 39
    shared-memory input-plane-staging im2col variant when it fits this
    device's conservative 48KB-per-block cap, falling back to the plain
    Milestone 34 `im2col` otherwise. Set `False` to force the plain variant
    (used by `benchmarks/m41_conv2d_forward_profile.py` to measure both).
    """
    storages = (x, weight) if bias is None else (x, weight, bias)
    dtype = backend._require_compute_dtype(*storages, op="conv2d forward (experimental im2col+GEMM, Milestone 41)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    Hout = (H + 2 * PH - KH) // SH + 1
    Wout = (W + 2 * PW - KW) // SW + 1
    K = Cin * KH * KW
    M = N * Hout * Wout

    if use_smem_im2col and im2col_smem_fits(dtype, H, W):
        xcol = im2col_smem(backend, x, N, Cin, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout)
    else:
        xcol = im2col(backend, x, N, Cin, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout)

    weightT = transpose_weight(backend, weight, Cout, K)

    out_mat_ptr = backend._alloc(M * Cout * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_matmul_{_SUFFIX[dtype]}")
    code = fn(
        xcol.ptr, weightT.ptr, out_mat_ptr,
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(Cout),
        backend._stream_handle(),
    )
    backend._check(code, "GEMM (conv2d forward im2col+GEMM, Milestone 41)")
    out_mat = CUDAStorage(out_mat_ptr, (M, Cout), dtype, backend._lib)

    out = output_permute(backend, out_mat, bias, N, Cout, Hout, Wout)
    backend._maybe_synchronize("conv2d forward (experimental im2col+GEMM, Milestone 41)")
    return out


__all__ = ["transpose_weight", "output_permute", "conv2d_forward_im2col_gemm"]
