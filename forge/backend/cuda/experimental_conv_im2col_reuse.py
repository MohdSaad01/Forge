"""Milestone 39: `im2col` materialization candidates (profiling-only).

M38 left `k_im2col_conv2d` (`experimental_conv_im2col.im2col`) as the
dominant remaining `dWeight` cost at every `blocks_y >= 2` shape (`Cout >
16` -- the regime M38's half-fused GEMM cannot help). This module implements
the two candidates `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 39** section measures against that baseline:

- `im2col_indexed` (**Candidate A**): removes the redundant per-thread
  integer division `k_im2col_conv2d` pays for its `m -> (n,ho,wo)` and
  `k -> (ci,kh,kw)` decompositions (the former redundant `K`-fold per output
  position, the latter redundant `M`-fold per `k` value) by hoisting the
  `m` decomposition to one thread per block (broadcast via shared memory)
  and looking `k`'s decomposition up from a tiny host-built table instead of
  computing it via division. See `kernels.cu`'s matching section for the
  full root-cause writeup.
- `im2col_smem` (**Candidate B**): stages an entire `x[n,ci,:,:]` plane into
  shared memory once per block and serves every `Xcol` write for that plane
  from shared memory instead of a fresh global load -- the milestone brief's
  literal "shared-memory input tile reuse" hypothesis.

Both are reached only through this module (`CUDABackend`/
`experimental_conv_im2col.im2col` never call them directly) -- the same
profiling-only convention every prior milestone's rejected/accepted
candidate set uses. See `benchmarks/m39_im2col_reuse_profile.py` for the
measured comparison.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from .backend import CUDAStorage, _SUFFIX
from .experimental_conv_im2col import grad_output_permute, im2col, recommended_num_k_splits

_INT32 = np.dtype(np.int32)

# `k_im2col_conv2d_smem` stages one entire `x[n,ci,:,:]` plane into dynamic
# shared memory per block. CC 5.0 (the verified 940MX) has a 48KB-per-block
# shared-memory ceiling with no opt-in to raise it (that mechanism -
# `cudaFuncAttributeMaxDynamicSharedMemorySize` - did not exist before
# Volta/CC 7.0), and this is the *lowest* such ceiling any CUDA-capable
# device exposes, so using it as a fixed, conservative safety cap is correct
# on every device Forge might run on (a newer GPU with a larger real budget
# simply never gets close to it for any Forge-scale Conv2d layer). Every one
# of Forge's own representative/test shapes (largest: 56x56 = 3,136
# elements, 12.5KB at float32 / 25.1KB at float64) sits far under this --
# the guard exists for a hypothetical larger spatial layer, not because any
# shape Forge currently exercises is close to it.
_MAX_SMEM_BYTES_PER_BLOCK = 48 * 1024


def _upload_int32(backend: Any, host: np.ndarray) -> CUDAStorage:
    """Synchronous small host->device upload -- mirrors `CUDABackend.from_array`'s
    own `cf_memcpy_h2d` call exactly, sized for a tiny (<=288-element)
    per-call index table, never a hot-path bulk transfer.

    Returns a `CUDAStorage` (never a bare pointer) so the buffer is released
    back through the M25 caching allocator like every other Forge CUDA
    buffer when it goes out of scope -- a bare `backend._alloc()` pointer
    with no owning `CUDAStorage` is never freed. An earlier draft of this
    function returned the raw pointer directly and leaked exactly this way
    (caught by this module's own repeated-use memory-safety test) -- the
    same bookkeeping mistake `docs/performance/conv2d-backward-profiling.md`'s
    **Milestone 33** section documents from that milestone's own profiling
    kernels ("an earlier draft released the candidate kernels' output buffer
    via a raw `cf_free` instead of through `CUDAStorage`... which silently
    corrupted `allocated_bytes` bookkeeping for *later*, unrelated tests in
    the same pytest session").
    """
    host = np.ascontiguousarray(host, dtype=_INT32)
    nbytes = max(host.nbytes, 1)
    ptr = backend._alloc(nbytes)
    if host.nbytes > 0:
        code = backend._lib.cf_memcpy_h2d(ptr, host.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(host.nbytes))
        backend._check(code, "im2col index-table upload (Milestone 39, Candidate A)")
    return CUDAStorage(ptr, (host.size,), _INT32, backend._lib)


def build_k_decomposition_tables(Cin: int, KH: int, KW: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """`k -> (ci, kh, kw)` for `k` in `[0, Cin*KH*KW)`, matching `k_im2col_conv2d`'s
    own `kw_ = k % KW; k1 = k / KW; kh = k1 % KH; ci = k1 / KH` decomposition exactly.
    """
    k = np.arange(Cin * KH * KW, dtype=np.int64)
    kw = k % KW
    k1 = k // KW
    kh = k1 % KH
    ci = k1 // KH
    return ci.astype(np.int32), kh.astype(np.int32), kw.astype(np.int32)


def _threads_for_k(K: int) -> int:
    """`min(256, K rounded up to a warp multiple)` -- `k_im2col_conv2d_indexed`
    launches exactly one thread per `k` value (`blockIdx.y` covers any
    remainder above 256), so a fixed 256-thread block wastes
    `1 - K/256` of every block's threads whenever `K < 256` (`K` is 9-288
    across every Forge shape -- often far below 256). Rounding up to a
    32-thread (one warp) multiple avoids partial-warp waste without
    over-provisioning for the common case where one block covers all of `K`.
    """
    return max(32, min(256, ((K + 31) // 32) * 32))


def im2col_indexed(
    backend: Any, x: CUDAStorage,
    N: int, Cin: int, H: int, W: int, KH: int, KW: int,
    SH: int, SW: int, PH: int, PW: int, Hout: int, Wout: int,
    threads_per_block: "int | None" = None,
) -> CUDAStorage:
    """Candidate A: block-per-output-position im2col with hoisted index decomposition."""
    dtype = x.dtype
    M = N * Hout * Wout
    K = Cin * KH * KW
    if threads_per_block is None:
        threads_per_block = _threads_for_k(K)

    ci_tab, kh_tab, kw_tab = build_k_decomposition_tables(Cin, KH, KW)
    k_ci = _upload_int32(backend, ci_tab)
    k_kh = _upload_int32(backend, kh_tab)
    k_kw = _upload_int32(backend, kw_tab)

    ptr = backend._alloc(M * K * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_im2col_conv2d_indexed_{_SUFFIX[dtype]}")
    code = fn(
        x.ptr, ptr, k_ci.ptr, k_kh.ptr, k_kw.ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout), ctypes.c_int(K),
        ctypes.c_int(threads_per_block),
        backend._stream_handle(),
    )
    backend._check(code, "im2col indexed (Milestone 39, Candidate A)")
    del k_ci, k_kh, k_kw
    return CUDAStorage(ptr, (M, K), dtype, backend._lib)


_SMEM_THREADS_PER_BLOCK = 256


def im2col_smem(
    backend: Any, x: CUDAStorage,
    N: int, Cin: int, H: int, W: int, KH: int, KW: int,
    SH: int, SW: int, PH: int, PW: int, Hout: int, Wout: int,
    threads_per_block: int = _SMEM_THREADS_PER_BLOCK,
) -> CUDAStorage:
    """Candidate B: one block per `(n, ci)` plane, staged into shared memory once."""
    dtype = x.dtype
    M = N * Hout * Wout
    K = Cin * KH * KW

    ptr = backend._alloc(M * K * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_im2col_conv2d_smem_{_SUFFIX[dtype]}")
    code = fn(
        x.ptr, ptr,
        ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(KH), ctypes.c_int(KW),
        ctypes.c_int(SH), ctypes.c_int(SW), ctypes.c_int(PH), ctypes.c_int(PW),
        ctypes.c_int(Hout), ctypes.c_int(Wout), ctypes.c_int(K),
        ctypes.c_int(threads_per_block),
        backend._stream_handle(),
    )
    backend._check(code, "im2col smem (Milestone 39, Candidate B)")
    return CUDAStorage(ptr, (M, K), dtype, backend._lib)


def im2col_smem_fits(dtype: np.dtype, H: int, W: int) -> bool:
    """Whether `k_im2col_conv2d_smem`'s per-block dynamic shared-memory
    request (`H*W*itemsize`, one full input plane) fits CC 5.0's 48KB static
    per-block ceiling -- see `_MAX_SMEM_BYTES_PER_BLOCK` above."""
    return H * W * np.dtype(dtype).itemsize <= _MAX_SMEM_BYTES_PER_BLOCK


def dweight_im2col_smem_gemm_splitk(
    backend: Any, grad_output: CUDAStorage, x: CUDAStorage,
    weight_shape: "tuple[int, int, int, int]",
    stride: "tuple[int, int]", padding: "tuple[int, int]",
) -> CUDAStorage:
    """Milestone 39 production `dWeight` (the `blocks_y >= 2` branch,
    `Cout > 16`): identical to M37's `dweight_im2col_gemm_splitk` except its
    `im2col` stage uses Candidate B (`im2col_smem`, shared-memory input-plane
    staging) whenever the per-block shared-memory footprint fits this
    device's conservative cap (`im2col_smem_fits`), falling back to the
    unmodified M34 `im2col` otherwise so no shape can ever fail to launch.
    `grad_output_permute` and the split-K GEMM (`cf_matmul_splitk_*`) are
    completely unchanged from M37 -- only the `im2col` call differs.

    See `docs/performance/conv2d-backward-profiling.md`'s **Milestone 39**
    section for the measured evidence (a real, reproducible full-pipeline
    win of 1.05-1.24x at every representative shape in this dispatch
    branch, with no observed regression across the full shape/kernel-size/
    stride sweep tested).
    """
    dtype = backend._require_compute_dtype(grad_output, x, op="conv2d dWeight (im2col+smem + split-K GEMM, Milestone 39)")
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    Hout, Wout = grad_output.shape[2], grad_output.shape[3]

    if im2col_smem_fits(dtype, H, W):
        xcol = im2col_smem(backend, x, N, Cin, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout)
    else:
        xcol = im2col(backend, x, N, Cin, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout)
    dycolT = grad_output_permute(backend, grad_output, N, Cout, Hout, Wout)

    M = N * Hout * Wout
    K = Cin * KH * KW
    num_k_splits = recommended_num_k_splits(M)
    out_ptr = backend._alloc(Cout * K * dtype.itemsize)
    fn = getattr(backend._lib, f"cf_matmul_splitk_{_SUFFIX[dtype]}")
    code = fn(
        dycolT.ptr, xcol.ptr, out_ptr,
        ctypes.c_int(Cout), ctypes.c_int(M), ctypes.c_int(K),
        ctypes.c_int(num_k_splits),
        backend._stream_handle(),
    )
    backend._check(code, "split-K GEMM (dWeight, im2col+smem, Milestone 39)")
    backend._maybe_synchronize("conv2d dWeight (im2col+smem + split-K GEMM, Milestone 39)")
    return CUDAStorage(out_ptr, weight_shape, dtype, backend._lib)


__all__ = [
    "build_k_decomposition_tables", "im2col_indexed", "im2col_smem",
    "im2col_smem_fits", "dweight_im2col_smem_gemm_splitk",
]
