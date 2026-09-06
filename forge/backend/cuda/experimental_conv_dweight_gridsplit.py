"""Conv2d `dWeight` below-256-weight-element grid-split block-reduce
candidate (Milestone 46).

**Production status (Milestone 46 decision): rejected.** `CUDABackend.
conv2d_backward`'s below-`CONV2D_WEIGHT_REDUCE_THRESHOLD` (256) dispatch and
`k_conv2d_backward_weight_reduce` (production since M21) are byte-for-byte
unchanged. This module exists purely so this milestone's benchmark
(`benchmarks/m46_dweight_below256_gridsplit_profile.py`) and test module
(`tests/test_cuda_conv2d_backward_weight_below256_gridsplit.py`) have a
clean, buffer-managed entry point instead of repeating raw `ctypes` buffer
plumbing in both places -- the same role `experimental_conv_dweight_
tworeduce.py` plays for M43's (accepted) two-stage split-K candidate.

M45 corrected M44's "insufficient threads" diagnosis: the production
block-reduce kernel already launches `weight_elements x 256` threads, more
than either of M45's own (rejected) warp-shuffle candidates. M46 asked the
remaining question directly -- is *grid-level* parallelism (blocks, not
threads) the limiting factor? -- by launching `num_splits` blocks per weight
element instead of one, each reducing a disjoint slice of the
`N*Hout*Wout` dimension (`k_conv2d_backward_weight_reduce_gridsplit`), then
combining the `(num_splits, weight_elements)` partial buffer with the
existing M43 combine kernel (`k_dweight_splitk_reduce`, already general
enough for a flat vector via `Cout=1, Kdim=weight_elements`).

A real `cudaOccupancyMaxActiveBlocksPerMultiprocessor` query
(`cf_occupancy_conv2d_backward_weight_reduce[_gridsplit]_f32`) found both
kernels land at the *same* 5 resident blocks/SM (register-bound, not
thread- or shared-memory-bound) -- grid-splitting does not raise the 940MX's
per-SM concurrent-block ceiling, it only launches more, smaller blocks
against that same ceiling. A same-session, round-robin-**interleaved**
CUDA-event A/B (the milestone's own corrected methodology -- an initial
block-sequential timing attempt measured an illusory 1.47x "win" at
`mnist_conv1` that a true interleaved rerun could not reproduce, 1.02x
instead, within noise) confirmed the occupancy diagnosis: the best speedup
found across every representative/sweep shape topped out at ~1.07x, never
reaching the milestone's own 1.15x acceptance bar, and reduction lengths
short enough for one split's slice to be nearly free (`reduction_small`,
256 elements) regressed outright (down to 0.58x at `num_splits=8`) as the
extra per-block launch/sync/second-kernel overhead dominated. See
`docs/performance/conv2d-backward-profiling.md`'s **Milestone 46** section
for the complete evidence.
"""

from __future__ import annotations

import ctypes
from typing import Any

from .backend import CUDAStorage, _SUFFIX


def dweight_below256_gridsplit(
    backend: Any, grad_output: CUDAStorage, x: CUDAStorage,
    weight_shape: "tuple[int, int, int, int]",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
    num_splits: int,
) -> CUDAStorage:
    """Rejected M46 candidate: `num_splits` blocks per weight element, each
    reducing a disjoint slice of `N*Hout*Wout`, combined by the existing M43
    `k_dweight_splitk_reduce` kernel. Never called by `CUDABackend` --
    profiling/test entry point only."""
    dtype = backend._require_compute_dtype(grad_output, x, op="conv2d dWeight (M46 grid-split candidate, rejected)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = grad_output.shape[2], grad_output.shape[3]
    weight_elements = Cout * Cin * KH * KW

    shape_args = (
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout),
    )

    partial_ptr = backend._alloc(num_splits * weight_elements * dtype.itemsize)
    # Wrapped in a `CUDAStorage` immediately (M43's own convention) purely so
    # its `__del__` frees it back to the allocator when this function
    # returns -- it is never used as a tensor, only as an owned buffer.
    partial = CUDAStorage(partial_ptr, (num_splits, weight_elements), dtype, backend._lib)
    out_ptr = backend._alloc(weight_elements * dtype.itemsize)

    fn_partial = getattr(backend._lib, f"cf_conv2d_backward_weight_gridsplit_partial_{_SUFFIX[dtype]}")
    code = fn_partial(
        grad_output.ptr, x.ptr, partial.ptr, *shape_args,
        ctypes.c_int(num_splits), backend._stream_handle(),
    )
    backend._check(code, "conv2d dWeight (M46 grid-split candidate, partial)")

    fn_reduce = getattr(backend._lib, f"cf_dweight_splitk_reduce_{_SUFFIX[dtype]}")
    code = fn_reduce(
        partial.ptr, out_ptr,
        ctypes.c_int(1), ctypes.c_int(weight_elements), ctypes.c_int(num_splits),
        backend._stream_handle(),
    )
    backend._check(code, "conv2d dWeight (M46 grid-split candidate, reduce)")
    backend._maybe_synchronize("conv2d dWeight (M46 grid-split candidate)")
    return CUDAStorage(out_ptr, weight_shape, dtype, backend._lib)


__all__ = ["dweight_below256_gridsplit"]
