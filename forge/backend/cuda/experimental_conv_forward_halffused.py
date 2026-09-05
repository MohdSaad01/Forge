"""Conv2d forward via a half-fused GEMM (Milestone 41, Candidate B).

Candidate A (`experimental_conv_forward_im2col`) materializes `Xcol`, the
large `(N*Hout*Wout, Cin*KH*KW)` gathered-window buffer, before handing it to
the existing tiled GEMM. This module is a structurally different candidate:
it eliminates the `Xcol` buffer entirely by fusing its gather directly into
the GEMM's own shared-memory tile load -- the same "half-fused" idea
Milestone 38 used for `dWeight` (`experimental_conv_halffused`), with the
fused/materialized roles reversed to match forward's own GEMM orientation.

`weightT` (`(Cin*KH*KW, Cout)`, via the existing `k_transpose`, unmodified)
stays materialized -- it is small (`Cout*Cin*KH*KW` elements, Forge's own
weight-element count, always far smaller than `Xcol`) and cheap to build
once per forward call. Only the *expensive* operand's gather (`Xcol`,
`tile_a`) is fused into `k_conv2d_forward_halffused_gemm`'s tile load; the
kernel also writes its result directly into `out`'s real
`(N, Cout, Hout, Wout)` memory layout with bias fused in, so unlike
Candidate A this pipeline needs no `out_mat`/`Xcol` intermediate buffers at
all -- only `weightT` and the final output.

No split-K: forward's GEMM has `M=N*Hout*Wout` (huge) as a *block-count*
dimension (`blocks_y=ceil(M/16)`), unlike `dWeight`'s GEMM where `M` was the
*reduction* dimension and `Cout`/`Cin*KH*KW` (small) were the block-count
dimensions -- the exact occupancy shortfall that motivated Milestone 37's
split-K fix. Forward's own `blocks_y` is already large at every Forge shape,
so this kernel needs no third grid dimension or atomic accumulation.

See `docs/performance/conv2d-forward-profiling.md`'s **Milestone 41**
section for the measured comparison against Candidate A and the existing
production `k_conv2d_forward`.
"""

from __future__ import annotations

import ctypes
from typing import Any

from .backend import CUDAStorage, _SUFFIX
from .experimental_conv_forward_im2col import transpose_weight


def conv2d_forward_halffused_gemm(
    backend: Any, x: CUDAStorage, weight: CUDAStorage, bias: "CUDAStorage | None",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
) -> CUDAStorage:
    """The complete Candidate B forward pipeline: weight transpose -> fused
    im2col-gather GEMM, writing directly into the final `(N, Cout, Hout,
    Wout)` output with bias fused in."""
    storages = (x, weight) if bias is None else (x, weight, bias)
    dtype = backend._require_compute_dtype(*storages, op="conv2d forward (experimental half-fused GEMM, Milestone 41)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    Hout = (H + 2 * PH - KH) // SH + 1
    Wout = (W + 2 * PW - KW) // SW + 1
    K = Cin * KH * KW

    weightT = transpose_weight(backend, weight, Cout, K)

    out_ptr = backend._alloc(N * Cout * Hout * Wout * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_conv2d_forward_halffused_gemm_{_SUFFIX[dtype]}")
    code = fn(
        weightT.ptr, x.ptr, bias.ptr if bias is not None else None, out_ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(Cout), ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout), ctypes.c_int(1 if bias is not None else 0),
        backend._stream_handle(),
    )
    backend._check(code, "half-fused GEMM (conv2d forward, Milestone 41 candidate B)")
    out = CUDAStorage(out_ptr, (N, Cout, Hout, Wout), dtype, backend._lib)
    backend._maybe_synchronize("conv2d forward (experimental half-fused GEMM, Milestone 41)")
    return out


__all__ = ["conv2d_forward_halffused_gemm"]
