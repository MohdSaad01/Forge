"""Conv2d `dWeight` two-stage deterministic split-K reduction (Milestone 43,
Candidate B).

**Production status (Milestone 43 decision): adopted**, replacing M38's
`dweight_halffused_gemm_splitk` (`experimental_conv_halffused.py`, still
kept for benchmark comparison, now dead in production) at the `blocks_y==1`
dWeight dispatch (`CUDABackend.conv2d_backward`, `Cout<=16`).

M42 flagged split-K's atomic-accumulation combine step as an untested root-
cause hypothesis for why `k_dweight_halffused_gemm_splitk` (`blocks_y==1`)
reaches only ~half the roofline efficiency of the structurally identical
(non-split-K) forward half-fused GEMM. M43's own `num_k_splits` sensitivity
sweep (`benchmarks/m43_dweight_splitk_profile.py`) found the production
`recommended_num_k_splits` formula already sits within ~3-8% of its own
swept optimum (ruling out split-count/occupancy retuning as a source of
major headroom) but confirmed atomics still cost something real: pushing
`num_k_splits` far past the production value degrades monotonically (e.g.
`mnist_conv2`: 1.00ms at 64 splits vs 1.37ms at 676). This module replaces
the atomic combine with a deterministic two-stage reduction (each split
writes a disjoint partial-output slice; a second small kernel sums the
`num_k_splits` axis) at the *same*, unchanged `num_k_splits` the production
kernel already uses -- isolating the atomic-vs-plain-write cost directly.
Measured 1.14-1.24x faster end-to-end than the atomic kernel at
`mnist_conv2` and 1.02-1.04x faster at `large_spatial`, at every tested
`num_k_splits`, with no regression at any tested shape/split count.

See `benchmarks/m43_dweight_splitk_profile.py`'s Candidate B comparison and
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 43** section
for the complete measured evidence.
"""

from __future__ import annotations

import ctypes
from typing import Any

from .backend import CUDAStorage, _SUFFIX
from .experimental_conv_im2col import grad_output_permute, recommended_num_k_splits


def dweight_halffused_gemm_splitk_tworeduce(
    backend: Any, grad_output: CUDAStorage, x: CUDAStorage,
    weight_shape: "tuple[int, int, int, int]",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
    num_k_splits: "int | None" = None,
) -> CUDAStorage:
    """Candidate B: same tile loads/gather as the production M38 kernel, but
    each split writes its own disjoint partial slice (no atomic) and a
    second small reduction kernel sums the `num_k_splits` axis."""
    dtype = backend._require_compute_dtype(grad_output, x, op="conv2d dWeight (two-stage split-K reduction, M43 candidate B)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = grad_output.shape[2], grad_output.shape[3]

    dycolT = grad_output_permute(backend, grad_output, N, Cout, Hout, Wout)

    M = N * Hout * Wout
    K = Cin * KH * KW
    if num_k_splits is None:
        num_k_splits = recommended_num_k_splits(M)

    partial_ptr = backend._alloc(num_k_splits * Cout * K * dtype.itemsize)
    # Wrapped in a `CUDAStorage` immediately (like `dycolT` above) purely so
    # its `__del__` frees it back to the allocator when this function
    # returns -- it is never used as a tensor, only as an owned buffer.
    partial = CUDAStorage(partial_ptr, (num_k_splits, Cout, K), dtype, backend._lib)
    out_ptr = backend._alloc(Cout * K * dtype.itemsize)

    fn_partial = getattr(backend._lib, f"cf_dweight_halffused_gemm_splitk_partial_{_SUFFIX[dtype]}")
    code = fn_partial(
        dycolT.ptr, x.ptr, partial.ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
        ctypes.c_int(num_k_splits),
        backend._stream_handle(),
    )
    backend._check(code, "conv2d dWeight (two-stage split-K reduction, M43 candidate B, GEMM partials)")

    fn_reduce = getattr(backend._lib, f"cf_dweight_splitk_reduce_{_SUFFIX[dtype]}")
    code = fn_reduce(
        partial.ptr, out_ptr,
        ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(num_k_splits),
        backend._stream_handle(),
    )
    backend._check(code, "conv2d dWeight (two-stage split-K reduction, M43 candidate B, reduce)")
    backend._maybe_synchronize("conv2d dWeight (two-stage split-K reduction, M43 candidate B)")
    return CUDAStorage(out_ptr, weight_shape, dtype, backend._lib)


__all__ = ["dweight_halffused_gemm_splitk_tworeduce"]
