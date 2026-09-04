"""Kernel-level roofline characterization (Milestone 35).

    python -m benchmarks.m35_kernels

Covers Sections 12-21 and 31-32 of the M35 brief: elementwise, reduction,
GEMM, Conv2d (forward/dInput/dWeight-direct/dWeight-im2col+GEMM), MaxPool2d,
Dropout, and the optimizers, at small/medium/large sizes, plus the M34
256-1152-weight-element threshold region and its memory/performance
tradeoff. Every isolated-kernel timing uses the timing-enabled `TimedEvent`
convention (`forge.backend.cuda.profiling_events`), reusing
`m35_hardware._time_raw` for the actual measurement loop.

Conv2d forward/dInput/dWeight-direct is measured by importing and reusing
`conv2d_backward_profile._RawConv2dBackward`/`_time_phase` directly (no
reimplementation of that isolation logic); the dWeight im2col+GEMM path and
its memory/threshold analysis reuses
`conv2d_backward_weight_im2col_profile._profile_shape` directly, which
already reports `xcol_mb`/`dycolT_mb`/`current_vs_experimental_speedup` --
exactly what Sections 17/31/32 ask for. Per Section 31, the new
256-1152-weight-element shapes are **measurement only**: `backend.py`'s
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD` is never touched.

Every result is run through `benchmarks.roofline`'s FLOP/byte/AI model and
classified against `benchmarks/results/m35_hardware.json`'s measured
practical ceilings (run `m35_hardware` first).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import _SUFFIX, get_cuda_backend, is_cuda_available

from . import roofline as rf
from .conv2d_backward_profile import (
    BATCH_SIZES,
    BATCH_SWEEP_BASE,
    SHAPES as CONV_SHAPES,
    _check_fits_in_vram,
    _hout_wout,
    _RawConv2dBackward,
    _time_phase as _conv_time_phase,
)
from .conv2d_backward_weight_im2col_profile import _profile_shape as _im2col_profile_shape
from .environment import collect_environment
from .m35_hardware import _time_raw
from .timing import time_cpu, time_cuda

WARMUP = 5
ITERATIONS = 20

# Self-contained size tiers (deliberately not touching `sizes.py`'s existing
# tiny/small/medium tiers used by the stable `python -m benchmarks` suite --
# M35 wants a true "large" tier to find the memory-bound plateau, Section 11).
EW_SIZES = {"small": 16_384, "medium": 1_048_576, "large": 16_000_000}
CE_SIZES = {"small": (64, 10), "medium": (1_024, 10), "large": (16_384, 10)}
GEMM_SHAPES = {
    "square_small": (128, 128, 128),
    "square_medium": (512, 512, 512),
    "square_large": (1536, 1536, 1536),
    "tall_skinny": (8192, 64, 64),
    "wide_flat": (64, 8192, 64),
}
MAXPOOL_CONFIGS = {
    "small": {"N": 8, "C": 8, "H": 32, "W": 32},
    "medium": {"N": 16, "C": 16, "H": 64, "W": 64},
    "large": {"N": 32, "C": 32, "H": 96, "W": 96},
}
POOL_K = POOL_S = 2

# M34's untested 256-1152-weight-element region (Section 31), all at
# mnist_conv1's spatial scale (N=64, H=W=28, K=3, S=1, P=1) so only the
# channel-count product varies -- weight_elements = Cin*Cout*9.
THRESHOLD_SHAPES = {
    "thresh_288": {"N": 64, "Cin": 4, "Cout": 8, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
    "thresh_576": {"N": 64, "Cin": 8, "Cout": 8, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
    "thresh_864": {"N": 64, "Cin": 8, "Cout": 12, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
    "thresh_1008": {"N": 64, "Cin": 8, "Cout": 14, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
}


def _classify_result(gflops: float, elapsed_s: float, ai: float, gbps: float, ceilings: "rf.Ceilings | None") -> dict:
    if ceilings is None:
        return {}
    c = rf.classify(gflops, elapsed_s, ai, ceilings, achieved_gbps=gbps)
    return {"classification": c.label, "roofline_ceiling_gflops": c.roofline_ceiling_gflops,
            "fraction_of_ceiling": c.fraction_of_ceiling, "classification_note": c.note}


def _record(op: str, scale: str, shape: dict, timing: "dict[str, float]", flops: int, nbytes: int,
            ceilings: "rf.Ceilings | None") -> dict:
    elapsed = timing["mean_s"]
    gflops = (flops / 1e9) / elapsed if elapsed > 0 else 0.0
    gbps = (nbytes / 1e9) / elapsed if elapsed > 0 else 0.0
    ai = rf.arithmetic_intensity(flops, nbytes)
    rec = {
        "op": op, "scale": scale, "shape": shape, "timing": timing,
        "flops": flops, "bytes": nbytes, "arithmetic_intensity": ai,
        "achieved_gflops": gflops, "achieved_gbps": gbps,
    }
    rec.update(_classify_result(gflops, elapsed, ai, gbps, ceilings))
    return rec


# -- elementwise + reduction ------------------------------------------------


def _raw_binary_fn(name: str):
    backend = get_cuda_backend()
    return getattr(backend._lib, f"cf_{name}_{_SUFFIX[np.dtype(np.float32)]}")


def _raw_unary_fn(name: str):
    backend = get_cuda_backend()
    return getattr(backend._lib, f"cf_{name}_{_SUFFIX[np.dtype(np.float32)]}")


def _profile_elementwise(ceilings) -> "list[dict]":
    results = []
    backend = get_cuda_backend()
    for scale, n in EW_SIZES.items():
        rng = np.random.default_rng(0)
        a = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
        b = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
        c = forge.Tensor((rng.random(n).astype(np.float32) + 0.1), device="cuda")  # positive, for exp/log
        forge.cuda.synchronize()

        for op in ("add", "sub", "mul"):
            fn = _raw_binary_fn(op)
            out_ptr = backend._alloc(n * 4)
            args = (a._data.ptr, b._data.ptr, out_ptr, ctypes.c_longlong(n))
            timing = _time_raw(lambda fn=fn, args=args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
            results.append(_record(op, scale, {"n": n}, timing,
                                    rf.flops_elementwise(n), rf.bytes_elementwise_binary(n), ceilings))
            backend._lib.cf_free(out_ptr)

        for op in ("relu", "exp", "log"):
            fn = _raw_unary_fn(op)
            src = a if op == "relu" else c
            out_ptr = backend._alloc(n * 4)
            args = (src._data.ptr, out_ptr, ctypes.c_longlong(n))
            timing = _time_raw(lambda fn=fn, args=args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
            flops = rf.flops_relu(n) if op == "relu" else rf.flops_elementwise(n)
            results.append(_record(op, scale, {"n": n}, timing,
                                    flops, rf.bytes_elementwise_unary(n), ceilings))
            backend._lib.cf_free(out_ptr)

        forge.cuda.empty_cache()
    return results


def _profile_reduction(ceilings) -> "list[dict]":
    results = []
    backend = get_cuda_backend()
    suffix = _SUFFIX[np.dtype(np.float32)]

    for scale, n in EW_SIZES.items():
        rng = np.random.default_rng(1)
        a = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        fn = getattr(backend._lib, f"cf_sum_{suffix}")
        out_ptr = backend._alloc(4)
        args = (a._data.ptr, out_ptr, ctypes.c_longlong(n))
        timing = _time_raw(lambda fn=fn, args=args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("sum", scale, {"n": n}, timing,
                                rf.flops_reduction(n), rf.bytes_reduction(n), ceilings))
        backend._lib.cf_free(out_ptr)
        forge.cuda.empty_cache()

    for scale, (batch, classes) in CE_SIZES.items():
        rng = np.random.default_rng(2)
        logits = forge.Tensor(rng.standard_normal((batch, classes)).astype(np.float32), device="cuda")
        target = forge.Tensor(rng.integers(0, classes, size=(batch,)).astype(np.int64), device="cuda")
        grad_out = forge.Tensor(np.ones((), dtype=np.float32), device="cuda")
        forge.cuda.synchronize()

        fwd_fn = getattr(backend._lib, f"cf_cross_entropy_forward_{suffix}")
        per_row_ptr = backend._alloc(batch * 4)
        fwd_args = (logits._data.ptr, target._data.ptr, per_row_ptr,
                    ctypes.c_longlong(batch), ctypes.c_longlong(classes), ctypes.c_double(1.0 / batch))
        timing = _time_raw(lambda fn=fwd_fn, args=fwd_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("cross_entropy_forward", scale, {"batch": batch, "classes": classes}, timing,
                                rf.flops_cross_entropy_forward(batch, classes),
                                rf.bytes_cross_entropy_forward(batch, classes), ceilings))
        backend._lib.cf_free(per_row_ptr)

        bwd_fn = getattr(backend._lib, f"cf_cross_entropy_backward_{suffix}")
        grad_in_ptr = backend._alloc(batch * classes * 4)
        bwd_args = (grad_out._data.ptr, logits._data.ptr, target._data.ptr, grad_in_ptr,
                    ctypes.c_longlong(batch), ctypes.c_longlong(classes), ctypes.c_double(1.0 / batch))
        timing = _time_raw(lambda fn=bwd_fn, args=bwd_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("cross_entropy_backward", scale, {"batch": batch, "classes": classes}, timing,
                                rf.flops_cross_entropy_backward(batch, classes),
                                rf.bytes_cross_entropy_backward(batch, classes), ceilings))
        backend._lib.cf_free(grad_in_ptr)
        forge.cuda.empty_cache()

    return results


# -- GEMM ---------------------------------------------------------------


def _profile_gemm(ceilings) -> "list[dict]":
    results = []
    backend = get_cuda_backend()
    fn = getattr(backend._lib, f"cf_matmul_{_SUFFIX[np.dtype(np.float32)]}")

    for name, (M, K, N) in GEMM_SHAPES.items():
        nbytes_total = (M * K + K * N + M * N) * 4
        assert nbytes_total < 300 * 1024 * 1024, f"GEMM shape {name} too large for a single benchmark"
        rng = np.random.default_rng(3)
        a = forge.Tensor(rng.standard_normal((M, K)).astype(np.float32), device="cuda")
        b = forge.Tensor(rng.standard_normal((K, N)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        out_ptr = backend._alloc(M * N * 4)
        args = (a._data.ptr, b._data.ptr, out_ptr, ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N))
        timing = _time_raw(lambda fn=fn, args=args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("matmul", name, {"M": M, "K": K, "N": N}, timing,
                                rf.flops_matmul(M, N, K), rf.bytes_matmul_minimum(M, N, K), ceilings))
        backend._lib.cf_free(out_ptr)
        forge.cuda.empty_cache()
    return results


# -- Conv2d forward / dInput / dWeight-direct (reuses conv2d_backward_profile) --


def _profile_conv2d(ceilings) -> "list[dict]":
    results = []
    all_shapes = dict(CONV_SHAPES)
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        all_shapes[f"batch_{n}"] = cfg

    for name, cfg in all_shapes.items():
        _check_fits_in_vram(cfg)
        raw = _RawConv2dBackward(cfg)
        Hout, Wout = raw.Hout, raw.Wout
        N, Cin, Cout, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["K"]

        fwd_ms = _conv_time_phase(raw.forward, iterations=ITERATIONS, warmup=WARMUP)
        dinput_ms = _conv_time_phase(raw.backward_input, iterations=ITERATIONS, warmup=WARMUP)
        dweight_ms = _conv_time_phase(raw.backward_weight, iterations=ITERATIONS, warmup=WARMUP)

        shape = {"N": N, "Cin": Cin, "Cout": Cout, "H": cfg["H"], "W": cfg["W"], "K": K, "Hout": Hout, "Wout": Wout}
        results.append(_record("conv2d_forward", name, shape,
                                {"mean_s": fwd_ms["mean_ms"] / 1000, **{k: v / 1000 for k, v in fwd_ms.items() if k != "mean_ms"}},
                                rf.flops_conv2d_forward(N, Cout, Hout, Wout, Cin, K, K),
                                rf.bytes_conv2d_forward(N, Cin, cfg["H"], cfg["W"], Cout, K, K, Hout, Wout), ceilings))
        results.append(_record("conv2d_dinput", name, shape,
                                {"mean_s": dinput_ms["mean_ms"] / 1000, **{k: v / 1000 for k, v in dinput_ms.items() if k != "mean_ms"}},
                                rf.flops_conv2d_dinput(N, Cin, cfg["H"], cfg["W"], Cout, K, K),
                                rf.bytes_conv2d_dinput(N, Cin, cfg["H"], cfg["W"], Cout, K, K, Hout, Wout), ceilings))
        results.append(_record("conv2d_dweight_direct", name, shape,
                                {"mean_s": dweight_ms["mean_ms"] / 1000, **{k: v / 1000 for k, v in dweight_ms.items() if k != "mean_ms"}},
                                rf.flops_conv2d_dweight(Cout, Cin, K, K, N, Hout, Wout),
                                rf.bytes_conv2d_dweight(N, Cin, cfg["H"], cfg["W"], Cout, K, K, Hout, Wout), ceilings))
        forge.cuda.empty_cache()
    return results


# -- dWeight im2col+GEMM, M34 threshold sweep, memory tradeoff (Sections 17, 31, 32) --


def _profile_dweight_im2col_and_threshold(ceilings) -> "list[dict]":
    results = []
    all_shapes = dict(CONV_SHAPES)
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        all_shapes[f"batch_{n}"] = cfg
    all_shapes.update(THRESHOLD_SHAPES)

    for name, cfg in all_shapes.items():
        raw_profile = _im2col_profile_shape(name, cfg)
        N, Cin, Cout, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(cfg["H"], K, S, P), _hout_wout(cfg["W"], K, S, P)
        flops = rf.flops_conv2d_dweight(Cout, Cin, K, K, N, Hout, Wout)
        nbytes = rf.bytes_conv2d_dweight(N, Cin, cfg["H"], cfg["W"], Cout, K, K, Hout, Wout)

        gemm_timing = {"mean_s": raw_profile["gemm_ms"]["mean_ms"] / 1000}
        total_timing = {"mean_s": raw_profile["summed_total_experimental_ms"] / 1000}

        results.append(_record("conv2d_dweight_im2col_gemm_only", name,
                                {"weight_elements": raw_profile["weight_elements"]}, gemm_timing,
                                flops, nbytes, ceilings))
        results.append(_record("conv2d_dweight_im2col_total", name,
                                {"weight_elements": raw_profile["weight_elements"], "in_256_1152_region": name in THRESHOLD_SHAPES},
                                total_timing, flops, nbytes, ceilings))

        results[-1]["im2col_gemm_vs_direct_speedup"] = raw_profile["current_vs_experimental_speedup"]
        results[-1]["memory_overhead_mb"] = raw_profile["xcol_mb"] + raw_profile["dycolT_mb"]
        results[-1]["current_direct_ms"] = raw_profile["current_ms"]["mean_ms"]
        results[-1]["im2col_phase_ms"] = raw_profile["im2col_ms"]["mean_ms"]
        results[-1]["permute_phase_ms"] = raw_profile["permute_ms"]["mean_ms"]
        results[-1]["gemm_phase_ms"] = raw_profile["gemm_ms"]["mean_ms"]

        forge.cuda.empty_cache()
    return results


# -- MaxPool2d ------------------------------------------------------------


def _profile_maxpool(ceilings) -> "list[dict]":
    results = []
    backend = get_cuda_backend()
    suffix = _SUFFIX[np.dtype(np.float32)]

    for scale, cfg in MAXPOOL_CONFIGS.items():
        N, C, H, W = cfg["N"], cfg["C"], cfg["H"], cfg["W"]
        Hout = (H - POOL_K) // POOL_S + 1
        Wout = (W - POOL_K) // POOL_S + 1
        rng = np.random.default_rng(4)
        x = forge.Tensor(rng.standard_normal((N, C, H, W)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, C, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()

        shape_args = (ctypes.c_int(N), ctypes.c_int(C), ctypes.c_int(H), ctypes.c_int(W),
                      ctypes.c_int(POOL_K), ctypes.c_int(POOL_K), ctypes.c_int(POOL_S), ctypes.c_int(POOL_S),
                      ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(Hout), ctypes.c_int(Wout))

        fwd_fn = getattr(backend._lib, f"cf_maxpool2d_forward_{suffix}")
        out_ptr = backend._alloc(N * C * Hout * Wout * 4)
        fwd_args = (x._data.ptr, out_ptr, *shape_args)
        timing = _time_raw(lambda fn=fwd_fn, args=fwd_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("maxpool2d_forward", scale, {"N": N, "C": C, "H": H, "W": W}, timing,
                                rf.flops_maxpool(N * C * Hout * Wout, POOL_K, POOL_K),
                                rf.bytes_maxpool_forward(N, C, Hout, Wout, POOL_K, POOL_K), ceilings))

        bwd_fn = getattr(backend._lib, f"cf_maxpool2d_backward_{suffix}")
        grad_x_ptr = backend._alloc(N * C * H * W * 4)
        bwd_args = (x._data.ptr, grad_out._data.ptr, grad_x_ptr, *shape_args)
        timing = _time_raw(lambda fn=bwd_fn, args=bwd_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("maxpool2d_backward", scale, {"N": N, "C": C, "H": H, "W": W}, timing,
                                rf.flops_maxpool(N * C * Hout * Wout, POOL_K, POOL_K),
                                rf.bytes_maxpool_backward(N, C, H, W, Hout, Wout), ceilings))

        backend._lib.cf_free(out_ptr)
        backend._lib.cf_free(grad_x_ptr)
        forge.cuda.empty_cache()
    return results


# -- Dropout (mask generation vs. elementwise multiply, Section 19) --------


def _profile_dropout(ceilings) -> "list[dict]":
    results = []
    backend = get_cuda_backend()
    suffix = _SUFFIX[np.dtype(np.float32)]
    p = 0.5

    for scale, n in EW_SIZES.items():
        rng = np.random.default_rng(5)
        x = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
        forge.cuda.synchronize()

        mask_fn = getattr(backend._lib, f"cf_dropout_mask_{suffix}")
        mask_ptr = backend._alloc(n * 4)
        mask_args = (mask_ptr, ctypes.c_longlong(n), ctypes.c_double(p), ctypes.c_uint64(0))
        timing = _time_raw(lambda fn=mask_fn, args=mask_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("dropout_mask_generation", scale, {"n": n}, timing,
                                0, n * 4, ceilings))  # write-only: mask generated in-register, 0 FLOPs

        mul_fn = _raw_binary_fn("mul")
        mul_out_ptr = backend._alloc(n * 4)
        mul_args = (x._data.ptr, mask_ptr, mul_out_ptr, ctypes.c_longlong(n))
        timing = _time_raw(lambda fn=mul_fn, args=mul_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("dropout_elementwise_multiply", scale, {"n": n}, timing,
                                rf.flops_dropout(n), rf.bytes_dropout(n), ceilings))

        backend._lib.cf_free(mask_ptr)
        backend._lib.cf_free(mul_out_ptr)
        forge.cuda.empty_cache()

    # Eval-mode forward: `nn.Dropout` in eval mode is a pure pass-through --
    # no CUDA kernel launch at all (Section 19: "determine whether RNG/mask
    # generation is significant" -- eval mode's answer is "zero cost, by
    # construction, since nothing is launched").
    import forge.nn as nn
    dropout = nn.Dropout(p)
    dropout.eval()
    x = forge.Tensor(np.random.default_rng(5).standard_normal(EW_SIZES["medium"]).astype(np.float32), device="cuda")
    timing = time_cuda(lambda: dropout(x), warmup=WARMUP, iterations=ITERATIONS)
    results.append({
        "op": "dropout_eval_mode_forward", "scale": "medium", "shape": {"n": EW_SIZES["medium"]},
        "timing": {"mean_s": timing.mean, "median_s": timing.median, "min_s": timing.min,
                   "max_s": timing.max, "stdev_s": timing.stdev},
        "flops": 0, "bytes": 0, "arithmetic_intensity": 0.0, "achieved_gflops": 0.0, "achieved_gbps": 0.0,
        "note": "eval-mode Dropout launches no CUDA kernel -- pure Python pass-through.",
    })
    return results


# -- Optimizers (SGD/Adam) -------------------------------------------------


def _profile_optimizers(ceilings) -> "list[dict]":
    results = []
    backend = get_cuda_backend()
    suffix = _SUFFIX[np.dtype(np.float32)]

    for scale, n in EW_SIZES.items():
        rng = np.random.default_rng(6)
        param = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
        grad = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
        forge.cuda.synchronize()

        sgd_fn = getattr(backend._lib, f"cf_sgd_step_{suffix}")
        sgd_args = (param._data.ptr, grad._data.ptr, ctypes.c_double(1e-3), ctypes.c_longlong(n))
        timing = _time_raw(lambda fn=sgd_fn, args=sgd_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("sgd_step", scale, {"n": n}, timing, rf.flops_sgd(n), rf.bytes_sgd(n), ceilings))

        m = forge.Tensor(np.zeros(n, dtype=np.float32), device="cuda")
        v = forge.Tensor(np.zeros(n, dtype=np.float32), device="cuda")
        forge.cuda.synchronize()
        adam_fn = getattr(backend._lib, f"cf_adam_step_{suffix}")
        bc1, bc2 = 1.0 - 0.9, 1.0 - 0.999
        adam_args = (param._data.ptr, grad._data.ptr, m._data.ptr, v._data.ptr,
                     ctypes.c_double(1e-3), ctypes.c_double(0.9), ctypes.c_double(0.999),
                     ctypes.c_double(1e-8), ctypes.c_double(0.0),
                     ctypes.c_double(bc1), ctypes.c_double(bc2), ctypes.c_longlong(n))
        timing = _time_raw(lambda fn=adam_fn, args=adam_args: fn(*args, None), iterations=ITERATIONS, warmup=WARMUP)
        results.append(_record("adam_step", scale, {"n": n}, timing, rf.flops_adam(n), rf.bytes_adam(n), ceilings))
        forge.cuda.empty_cache()

    return results


# -- CPU vs. CUDA context (Section 35) -------------------------------------


def _profile_cpu_vs_cuda() -> "list[dict]":
    """Representative CPU/CUDA wall-time pairs at the "medium" scale -- context, not a roofline point."""
    results = []
    rng = np.random.default_rng(7)
    n = EW_SIZES["medium"]
    for device in ("cpu", "cuda"):
        a = forge.Tensor(rng.standard_normal(n).astype(np.float32), device=device)
        b = forge.Tensor(rng.standard_normal(n).astype(np.float32), device=device)
        timer = time_cpu if device == "cpu" else time_cuda
        add_t = timer(lambda a=a, b=b: a + b, warmup=WARMUP, iterations=ITERATIONS)
        results.append({"op": "add", "device": device, "n": n, "mean_s": add_t.mean})

        dim = 512
        m1 = forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device=device)
        m2 = forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device=device)
        mm_t = timer(lambda m1=m1, m2=m2: m1 @ m2, warmup=WARMUP, iterations=ITERATIONS)
        results.append({"op": "matmul", "device": device, "dim": dim, "mean_s": mm_t.mean})

        cfg = CONV_SHAPES["mnist_conv1"]
        x = forge.Tensor(rng.standard_normal((cfg["N"], cfg["Cin"], cfg["H"], cfg["W"])).astype(np.float32), device=device)
        w = forge.Tensor(rng.standard_normal((cfg["Cout"], cfg["Cin"], cfg["K"], cfg["K"])).astype(np.float32), device=device)
        b_ = forge.Tensor(rng.standard_normal((cfg["Cout"],)).astype(np.float32), device=device)
        conv_t = timer(lambda x=x, w=w, b_=b_: x.conv2d(w, b_, (1, 1), (0, 0)), warmup=WARMUP, iterations=ITERATIONS)
        results.append({"op": "conv2d_forward", "device": device, "shape": "mnist_conv1", "mean_s": conv_t.mean})
    return results


def _run(ceilings) -> dict:
    return {
        "elementwise": _profile_elementwise(ceilings),
        "reduction": _profile_reduction(ceilings),
        "gemm": _profile_gemm(ceilings),
        "conv2d": _profile_conv2d(ceilings),
        "conv2d_dweight_im2col_and_threshold": _profile_dweight_im2col_and_threshold(ceilings),
        "maxpool2d": _profile_maxpool(ceilings),
        "dropout": _profile_dropout(ceilings),
        "optimizers": _profile_optimizers(ceilings),
        "cpu_vs_cuda": _profile_cpu_vs_cuda(),
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M35 kernel-level roofline characterization (940MX, real CUDA) ===", ""]
    for section in ("elementwise", "reduction", "gemm", "conv2d", "maxpool2d", "dropout", "optimizers"):
        lines.append(f"-- {section} --")
        for r in profile[section]:
            cls = r.get("classification", "n/a")
            lines.append(
                f"  {r['op']:<32}{str(r['scale']):<10}{r['timing']['mean_s']*1e3:10.4f}ms  "
                f"{r['achieved_gflops']:9.3f} GFLOP/s  {r['achieved_gbps']:9.3f} GB/s  "
                f"AI={r['arithmetic_intensity']:7.3f}  {cls}"
            )
        lines.append("")
    lines.append("-- conv2d dWeight im2col+GEMM / threshold sweep --")
    for r in profile["conv2d_dweight_im2col_and_threshold"]:
        if r["op"] == "conv2d_dweight_im2col_total":
            lines.append(
                f"  {r['scale']:<14} weight#={r['shape']['weight_elements']:>5}  "
                f"speedup={r['im2col_gemm_vs_direct_speedup']:.2f}x  mem_overhead={r['memory_overhead_mb']:.2f}MB  "
                f"{r.get('classification', 'n/a')}"
            )
    lines.append("")
    lines.append("-- CPU vs. CUDA context (Section 35) --")
    for r in profile["cpu_vs_cuda"]:
        lines.append(f"  {r['op']:<16}{r['device']:<6}{r['mean_s']*1e3:10.4f}ms")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m35_kernels.json")
    parser.add_argument("--ceilings", default="benchmarks/results/m35_hardware.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m35_kernels requires real CUDA hardware.")
        return

    ceilings = None
    if Path(args.ceilings).exists():
        ceilings = rf.load_ceilings(args.ceilings)
    else:
        print(f"Warning: {args.ceilings} not found -- run m35_hardware first. Proceeding without classification.")

    profile = _run(ceilings)
    print(_render_report(profile))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"environment": collect_environment(), "profile": profile}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved profile -> {output_path}")


if __name__ == "__main__":
    main()
