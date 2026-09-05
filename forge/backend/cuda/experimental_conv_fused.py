"""Conv2d `dWeight` fused-gather GEMM candidates A/C (Milestone 37, rejected).

`benchmarks/m37_dweight_profile.py` decomposed the M34 im2col+GEMM `dWeight`
path (`forge.backend.cuda.experimental_conv_im2col`) across the 7
representative shapes and found two distinct, independently-measured
bottlenecks:

1. `k_im2col_conv2d` + `k_conv2d_grad_output_permute` (pure data-movement,
   zero FLOPs) together cost 54-63% of total dWeight time at every
   production shape -- more than the GEMM itself.
2. The GEMM's own launch geometry (`ceil(Cout/16) * ceil(Cin*KH*KW/16)`
   16x16 blocks) launches as few as 5 of the 940MX's 24-resident-block
   device capacity, and measured achieved-GFLOP/s-as-fraction-of-ceiling
   tracks that occupancy shortfall almost exactly.

This module wraps `kernels.cu`'s two matching profiling-only kernels, both
targeting bottleneck 1 (and, for Candidate C, bottleneck 2 as well):

* `dweight_fused_gemm` (Candidate A) -- folds both gathers directly into a
  `k_matmul`-shaped tiled GEMM's shared-memory tile loads, eliminating the
  `Xcol`/`dYcolT` intermediate buffers and their two kernel launches
  entirely. Same block geometry as M34's GEMM call (occupancy unchanged).
* `dweight_fused_gemm_splitk` (Candidate C) -- Candidate A plus splitting
  the huge `N*Hout*Wout` reduction across `num_k_splits` blocks along
  `blockIdx.z`, each atomically accumulating its partial sum into a
  pre-zeroed output -- directly targeting the occupancy shortfall.

**Measured and rejected** (`benchmarks/m37_dweight_candidates_profile.py`):
both lost to the M34 baseline at every shape with >= 18 GEMM blocks (already
75% of device block capacity) -- recomputing gather indices via integer
div/mod on every GEMM tile-loop iteration costs more than either bottleneck
fix buys back once occupancy is no longer the limiting factor. Candidate E
(`forge.backend.cuda.experimental_conv_im2col.dweight_im2col_gemm_splitk`)
isolates bottleneck 2 alone -- keeping M34's already-fast, cache-friendly
buffer reads and adding *only* the split-K occupancy fix -- and measured
2.7-9.0x faster than M34 at every shape, with no regression; it is the
Milestone 37 production dispatch. See `docs/performance/
conv2d-backward-profiling.md`'s **Milestone 37** section for the complete
comparison. Neither candidate in *this* module is wired into
`CUDABackend.conv2d_backward`. `k_matmul` and the M34 im2col/permute
kernels are completely unmodified by any of the three candidates.
"""

from __future__ import annotations

import ctypes
from typing import Any

from .backend import CUDAStorage, _SUFFIX


def dweight_fused_gemm(
    backend: Any, grad_output: CUDAStorage, x: CUDAStorage,
    weight_shape: "tuple[int, int, int, int]",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
) -> CUDAStorage:
    """Candidate A: single fused kernel, no intermediate `Xcol`/`dYcolT` buffers."""
    dtype = backend._require_compute_dtype(grad_output, x, op="conv2d dWeight (experimental fused GEMM, M37 candidate A)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = grad_output.shape[2], grad_output.shape[3]

    out_ptr = backend._alloc(Cout * Cin * KH * KW * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_dweight_fused_gemm_{_SUFFIX[dtype]}")
    code = fn(
        grad_output.ptr, x.ptr, out_ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
        backend._stream_handle(),
    )
    backend._check(code, "conv2d dWeight (experimental fused GEMM, M37 candidate A)")
    backend._maybe_synchronize("conv2d dWeight (experimental fused GEMM, M37 candidate A)")
    return CUDAStorage(out_ptr, weight_shape, dtype, backend._lib)


def dweight_fused_gemm_splitk(
    backend: Any, grad_output: CUDAStorage, x: CUDAStorage,
    weight_shape: "tuple[int, int, int, int]",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
    num_k_splits: int,
) -> CUDAStorage:
    """Candidate C: Candidate A's fused gather plus a split-K reduction for occupancy."""
    dtype = backend._require_compute_dtype(grad_output, x, op="conv2d dWeight (experimental fused GEMM splitk, M37 candidate C)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = grad_output.shape[2], grad_output.shape[3]

    out_ptr = backend._alloc(Cout * Cin * KH * KW * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_dweight_fused_gemm_splitk_{_SUFFIX[dtype]}")
    code = fn(
        grad_output.ptr, x.ptr, out_ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
        ctypes.c_int(num_k_splits),
        backend._stream_handle(),
    )
    backend._check(code, "conv2d dWeight (experimental fused GEMM splitk, M37 candidate C)")
    backend._maybe_synchronize("conv2d dWeight (experimental fused GEMM splitk, M37 candidate C)")
    return CUDAStorage(out_ptr, weight_shape, dtype, backend._lib)


__all__ = ["dweight_fused_gemm", "dweight_fused_gemm_splitk"]
