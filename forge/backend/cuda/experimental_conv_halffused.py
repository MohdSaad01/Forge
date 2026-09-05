"""Conv2d `dWeight` half-fused split-K GEMM candidate (Milestone 38, Candidate B).

M38 investigates whether `im2col` (`k_im2col_conv2d`) -- the single most
expensive stage of the M34/M37 `dWeight` pipeline (54-63% of total pipeline
time, per M37's decomposition) -- can be eliminated or reduced without
regressing the already-accepted M37 production dispatch (`experimental_conv_
im2col.dweight_im2col_gemm_splitk`).

## Why M37's Candidate A/C do not answer this question

M37 already tried "eliminate the gathers via a fused implicit GEMM"
(`experimental_conv_fused.dweight_fused_gemm[_splitk]`) and **rejected** it:
losing at every shape with >= 18 GEMM blocks. The measured cause was
recomputed gather-index arithmetic costing more than the eliminated buffer
traffic saved -- but that candidate fused *both* gathers (`im2col` and
`grad_output_permute`) symmetrically. A block-tiled GEMM's `tile_a` load
depends only on `(row, a_m)`, not on `blockIdx.x`, so every block sharing a
`blockIdx.y` redundantly regathers the same `tile_a` data `blocks_x` times;
symmetrically `tile_b` is redundantly regathered `blocks_y` times per unique
`blockIdx.x`. At every one of M38's 7 representative shapes, `Cout <= 32`
(`blocks_y = ceil(Cout/16) <= 2`) while `Cin*KH*KW` reaches 144 (`blocks_x`
up to 9) -- so M37's "fuse everything" candidate paid up to a **9x**
redundant-regather tax on the *cheap* `grad_output` gather (`permute` is
only 7-8% of pipeline time) while barely touching the *expensive* `im2col`
gather's own redundancy (at most 2x, since `blocks_y <= 2` everywhere
tested).

## Candidate B: fuse only the expensive gather

This module fuses *only* `im2col`'s gather (`tile_b`, eliminating the
`Xcol` buffer entirely -- this milestone's actual target) directly into the
GEMM's tile load, while keeping `grad_output_permute`'s cheap, already-fast
materialized `dYcolT` buffer (`k_conv2d_grad_output_permute`, unchanged
since M34) as `tile_a`'s source. Unlike M37's symmetric fusion, this pays
the redundant-regather tax only where it is cheap (`blocks_y <= 2`) and
never where it was expensive (`blocks_x` up to 9). Split-K (M37) is folded
in unconditionally, matching the accepted production baseline's own
occupancy fix, so the comparison isolates exactly one variable: does
eliminating the `Xcol` buffer (keeping everything else, including the
split-K occupancy fix, identical) win end-to-end?

See `benchmarks/m38_im2col_profile.py` for the measured comparison against
the M37 production baseline (`dweight_im2col_gemm_splitk`) at all 7
representative shapes, and `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 38** section for the full decision record.
"""

from __future__ import annotations

import ctypes
from typing import Any

from .backend import CUDAStorage, _SUFFIX
from .experimental_conv_im2col import grad_output_permute, recommended_num_k_splits


def dweight_halffused_gemm_splitk(
    backend: Any, grad_output: CUDAStorage, x: CUDAStorage,
    weight_shape: "tuple[int, int, int, int]",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
) -> CUDAStorage:
    """Candidate B: materialized `dYcolT` (unchanged M34 kernel) + on-the-fly
    `Xcol` gather fused into a split-K GEMM tile load -- no `Xcol` buffer.
    """
    dtype = backend._require_compute_dtype(grad_output, x, op="conv2d dWeight (half-fused split-K GEMM, M38 candidate B)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = grad_output.shape[2], grad_output.shape[3]

    dycolT = grad_output_permute(backend, grad_output, N, Cout, Hout, Wout)

    M = N * Hout * Wout
    K = Cin * KH * KW
    num_k_splits = recommended_num_k_splits(M)
    out_ptr = backend._alloc(Cout * K * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_dweight_halffused_gemm_splitk_{_SUFFIX[dtype]}")
    code = fn(
        dycolT.ptr, x.ptr, out_ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
        ctypes.c_int(num_k_splits),
        backend._stream_handle(),
    )
    backend._check(code, "conv2d dWeight (half-fused split-K GEMM, M38 candidate B)")
    backend._maybe_synchronize("conv2d dWeight (half-fused split-K GEMM, M38 candidate B)")
    return CUDAStorage(out_ptr, weight_shape, dtype, backend._lib)


__all__ = ["dweight_halffused_gemm_splitk"]
