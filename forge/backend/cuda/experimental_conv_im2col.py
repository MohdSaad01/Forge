"""Conv2d `dWeight` via im2col + the existing tiled GEMM (Milestone 34).

M33 rejected cooperative reduction as a `dWeight` optimization and named
im2col + GEMM -- reusing Forge's existing M11 shared-memory-tiled `k_matmul`
-- as the next structurally different candidate worth measuring
(`docs/performance/conv2d-backward-profiling.md`'s M33 **Limitations**
section). This module builds `Xcol` and a permuted `grad_output` via the two
gather kernels in `kernels.cu`'s matching "dWeight im2col + existing tiled
GEMM candidate" section, then issues one ordinary `cf_matmul_*` call -- **no
second GEMM implementation** (forbidden by the milestone brief's Section 43;
the existing tiled GEMM is used completely unmodified).

**Production status (Milestone 34 decision): adopted, above a weight-element
threshold.** `benchmarks/conv2d_backward_weight_im2col_profile.py` and
`benchmarks/conv2d_backward_im2col_pipeline_profile.py` measured this path
1.12-1.59x faster than the existing per-thread `dWeight` kernel, end-to-end
within `conv2d_backward`, at every representative shape with >= 1,152 weight
elements (`weight_elements = Cout*Cin*KH*KW`), with modest peak-memory
overhead (3-68MB extra reserved, well inside the 940MX's 2GB budget) and no
new synchronization. It measured slower at the one tested shape below the
existing `CONV2D_WEIGHT_REDUCE_THRESHOLD = 256` boundary (`mnist_conv1`, 72
elements, where the existing block-reduce kernel already wins per M33). Per
Sections 29/30/40 of the milestone brief, `CUDABackend.conv2d_backward`
(`backend.py`) now dispatches to `dweight_im2col_gemm` (this module) at/above
that threshold and keeps the original kernel below it -- see `backend.py`'s
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD` for the exact dispatch point and
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 34** section
for the complete evidence, including the documented caveat that no shape
between 256 and 1,152 weight elements was tested. `dInput`/`dBias` are
completely untouched -- only `dWeight`'s own kernel choice changed.

## Orientation (verified against `CPUBackend.conv2d_backward`)

Forge's actual weight-gradient GEMM, already computed on CPU via NumPy/BLAS
in `forge/backend/cpu.py`, is::

    grad_weight_mat = grad_out_rows.T @ cols_rows   # (Cout, M) @ (M, K) -> (Cout, K)

where `M = N*Hout*Wout` (the reduction dimension) and `K = Cin*KH*KW`. This
is **not** the `Xcol^T @ dYmat -> (K, Cout)` orientation a literal reading of
a naive im2col writeup would suggest -- that produces the transpose of what
Forge's `(Cout, Cin, KH, KW)` weight layout needs. This module follows the
verified orientation: `dycolT` (`Cout x M`) as GEMM operand A, `xcol`
(`M x K`) as operand B, producing `(Cout, K)` directly -- already exactly
`grad_weight`'s contiguous memory layout, reshaped to `(Cout, Cin, KH, KW)`
at zero cost (no copy, `CUDAStorage` construction only).

## No hidden synchronization

`im2col`, `grad_output_permute`, and the GEMM call are all issued on the
current Forge stream (`backend._stream_handle()`) with no
`cudaDeviceSynchronize()`/`cudaStreamSynchronize()` between them -- ordinary
CUDA stream program order (all three launches enqueued on the same stream)
is what keeps `im2col`'s writes visible to the GEMM read, not an inserted
host-side wait. `dweight_im2col_gemm` calls `backend._maybe_synchronize()`
exactly once, at the very end -- a no-op on an explicit `CUDAStream` (async
mode), matching every other `CUDABackend` method's contract.
"""

from __future__ import annotations

import ctypes
from typing import Any

from .backend import CUDAStorage, _SUFFIX


def im2col(
    backend: Any, x: CUDAStorage,
    N: int, Cin: int, H: int, W: int, KH: int, KW: int,
    SH: int, SW: int, PH: int, PW: int, Hout: int, Wout: int,
) -> CUDAStorage:
    """Build `Xcol`, shape `(N*Hout*Wout, Cin*KH*KW)`, row-major, via `k_im2col_conv2d`."""
    dtype = x.dtype
    M = N * Hout * Wout
    K = Cin * KH * KW
    ptr = backend._alloc(M * K * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_im2col_conv2d_{_SUFFIX[dtype]}")
    code = fn(
        x.ptr, ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(KH), ctypes.c_int(KW), ctypes.c_int(SH), ctypes.c_int(SW),
        ctypes.c_int(PH), ctypes.c_int(PW), ctypes.c_int(Hout), ctypes.c_int(Wout),
        backend._stream_handle(),
    )
    backend._check(code, "im2col (experimental dWeight, Milestone 34)")
    return CUDAStorage(ptr, (M, K), dtype, backend._lib)


def grad_output_permute(backend: Any, grad_output: CUDAStorage, N: int, Cout: int, Hout: int, Wout: int) -> CUDAStorage:
    """Build `dYcolT`, shape `(Cout, N*Hout*Wout)`, row-major, via `k_conv2d_grad_output_permute`."""
    dtype = grad_output.dtype
    M = N * Hout * Wout
    ptr = backend._alloc(Cout * M * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_conv2d_grad_output_permute_{_SUFFIX[dtype]}")
    code = fn(
        grad_output.ptr, ptr,
        ctypes.c_int(N), ctypes.c_int(Cout), ctypes.c_int(Hout), ctypes.c_int(Wout),
        backend._stream_handle(),
    )
    backend._check(code, "grad_output permute (experimental dWeight, Milestone 34)")
    return CUDAStorage(ptr, (Cout, M), dtype, backend._lib)


def dweight_im2col_gemm(
    backend: Any, grad_output: CUDAStorage, x: CUDAStorage,
    weight_shape: "tuple[int, int, int, int]",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
) -> CUDAStorage:
    """The complete experimental `dWeight` formulation: im2col -> grad_output permute -> existing tiled GEMM.

    Mirrors `CUDABackend.conv2d_backward`'s own shape derivation exactly, but
    computes only `grad_weight` (never `dInput`/`dBias` -- Section 6 of the
    milestone brief scopes this to weight-gradient only). Returns a
    `CUDAStorage` shaped `weight_shape`, matching `cf_conv2d_backward_weight_*`'s
    own return shape so it is a drop-in comparison target.
    """
    dtype = backend._require_compute_dtype(grad_output, x, op="conv2d dWeight (experimental im2col+GEMM)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = grad_output.shape[2], grad_output.shape[3]

    xcol = im2col(backend, x, N, Cin, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout)
    dycolT = grad_output_permute(backend, grad_output, N, Cout, Hout, Wout)

    M = N * Hout * Wout
    K = Cin * KH * KW
    out_ptr = backend._alloc(Cout * K * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_matmul_{_SUFFIX[dtype]}")
    code = fn(
        dycolT.ptr, xcol.ptr, out_ptr,
        ctypes.c_int(Cout), ctypes.c_int(M), ctypes.c_int(K),
        backend._stream_handle(),
    )
    backend._check(code, "GEMM (experimental dWeight, Milestone 34)")
    backend._maybe_synchronize("conv2d dWeight (experimental im2col+GEMM, Milestone 34)")
    return CUDAStorage(out_ptr, weight_shape, dtype, backend._lib)


__all__ = ["im2col", "grad_output_permute", "dweight_im2col_gemm"]
