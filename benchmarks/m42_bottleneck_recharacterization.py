"""M42: fresh post-M41 CUDA bottleneck re-characterization (measurement-only).

M41 changed `CUDABackend.conv2d`'s forward dispatch substantially (a
shape-based im2col+GEMM/half-fused-GEMM dispatch replacing a single direct
kernel below a FLOPs threshold) -- every earlier milestone's Conv2d-forward
percentages and optimization priorities are now stale. M40's own backward
re-characterization is *not* stale (M41 touched forward only), but no
milestone since M40 has measured the *whole* pipeline fresh against the
CURRENT production dispatch on both the forward and backward sides at once.

This script does exactly that, by combining -- not reimplementing --
M40's backward decomposition (`m40_bottleneck_recharacterization`) with
M41's forward decomposition (`m41_conv2d_forward_profile`), both against a
dispatch decision this script recomputes itself by mirroring `CUDABackend.
conv2d`/`conv2d_backward` exactly (Section 6 of the M42 brief: "do not rely
on historical documentation for dispatch decisions -- inspect the actual
production dispatch code"):

1. **Dispatch verification**: for every representative shape, compute both
   the forward (`total_flops`/`blocks_x`) and dWeight (`weight_elements`/
   `blocks_y`) dispatch decisions exactly as `backend.py` does, and record
   which function each one actually reaches.
2. **Full decomposition of the CURRENT production path**: forward (via
   `m41_conv2d_forward_profile._ForwardCandidates`, measuring whichever
   candidate -- baseline / half-fused / im2col+GEMM -- dispatch actually
   selects at that shape, with sub-stage decomposition when the im2col+GEMM
   path is selected), dInput/dWeight/dBias (via `m40_bottleneck_
   recharacterization`'s `_RawConv2dBackward`/`_RawDweightCurrent`), plus a
   real end-to-end `CUDABackend.conv2d()`/`conv2d_backward()` call at every
   shape as a cross-check that the decomposed sub-stage sum matches the
   actual production entry point.
3. **Fresh roofline ceilings** (M35's own methodology, this session) and
   classification of forward/dInput/dWeight/GEMM/CrossEntropy.
4. **Resource analysis** (`nvcc -Xptxas -v`) over the union of forward- and
   backward-relevant kernels.
5. **Pipeline health + memory**: reuses `pipeline_profile._profile_async_
   epoch`/`_profile_allocator_and_pinned`/`_profile_transfer_sizes` and CPU
   component costs directly, plus an explicit measurement of the im2col/GEMM
   forward path's temporary-buffer footprint (Xcol/weightT/out_mat) at the
   largest representative shape.
6. **MNIST kernel-contribution ranking + CrossEntropy roofline**: reuses
   `m35_mnist._run` (fresh) and `m35_kernels._profile_reduction` (fresh,
   filtered to its CrossEntropy forward/backward records) directly.
7. **Amdahl analysis** using this session's own measured fractions, not
   M35/M40's historical ones.

No production code is modified or reimplemented differently than production
already does -- every stage call below is the same function `backend.py`
itself calls (either directly, or through the same experimental module
`backend.py` imports).

    python -m benchmarks.m42_bottleneck_recharacterization
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import forge
from forge.backend.cuda.backend import (
    _CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD,
    _MATMUL_TILE,
    is_cuda_available,
)
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from . import m35_hardware, m35_kernels, m35_mnist, roofline
from .conv2d_backward_profile import (
    BATCH_SIZES,
    BATCH_SWEEP_BASE,
    SHAPES,
    _RawConv2dBackward,
    _check_fits_in_vram,
    _hout_wout,
)
from .environment import collect_environment
from .m40_bottleneck_recharacterization import _RawDweightCurrent
from .m40_bottleneck_recharacterization import _dispatch_decision as _dweight_dispatch_decision
from .m41_conv2d_forward_profile import SWEEP_SHAPES, _ForwardCandidates
from .m41_conv2d_forward_profile import _interleaved_compare as _fwd_interleaved_compare
from .m41_conv2d_forward_profile import _time_calls as _fwd_time_calls
from .pipeline_profile import _profile_allocator_and_pinned, _profile_async_epoch

WARMUP = 5
ITERATIONS = 30


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    """CUDA-event timing of a zero-arg `call() -> int errcode` (M40's exact
    convention -- used for every backward sub-stage below; callers embed any
    `stream_handle` argument in their own lambda, matching M40's own
    `_profile_shape`)."""
    for _ in range(warmup):
        code = call()
        assert code == 0, f"kernel launch failed with code {code}"
        forge.cuda.synchronize()
    pairs = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(None)
        code = call()
        assert code == 0, f"kernel launch failed with code {code}"
        end = TimedEvent()
        end.record(None)
        pairs.append((start, end))
    forge.cuda.synchronize()
    samples = [elapsed_ms(s, e) for s, e in pairs]
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


# -- Section 6/Section 3: forward dispatch verification, mirrors `CUDABackend.conv2d` exactly --


def _forward_dispatch_decision(cfg: "dict[str, int]") -> dict:
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    S, P = cfg["S"], cfg["P"]
    Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
    total_flops = 2 * N * Cout * Hout * Wout * Cin * K * K
    if total_flops < _CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD:
        return {"total_flops": total_flops, "blocks_x": None, "path": "cf_conv2d_forward (M15 per-thread, unchanged)"}
    blocks_x = (Cout + _MATMUL_TILE - 1) // _MATMUL_TILE
    if blocks_x <= 2:
        path = "conv2d_forward_halffused_gemm (M41 Candidate B)"
    else:
        path = "conv2d_forward_im2col_gemm (M41 Candidate A, smem-im2col + transpose + GEMM + permute)"
    return {"total_flops": total_flops, "blocks_x": blocks_x, "path": path}


def _real_conv2d_call(cand: "_ForwardCandidates"):
    """A zero-arg wrapper around the real, public `CUDABackend.conv2d()` entry
    point -- not a candidate's own internal pipeline call -- for the
    end-to-end cross-check every shape below performs."""

    def call() -> int:
        cand._real_fwd_out = cand.backend.conv2d(cand.x, cand.w, cand.b, (cand.S, cand.S), (cand.P, cand.P))
        return 0

    return call


def _real_conv2d_backward_call(raw: "_RawConv2dBackward", cfg: "dict[str, int]"):
    S, P = cfg["S"], cfg["P"]

    def call() -> int:
        raw._real_bwd_out = raw.backend.conv2d_backward(raw.grad_out, raw.x, raw.w, raw.b, (S, S), (P, P))
        return 0

    return call


def _classify(ms: float, flops: int, nbytes: int, ceilings: "roofline.Ceilings") -> dict:
    seconds = ms / 1000.0
    ai = roofline.arithmetic_intensity(flops, nbytes)
    gflops = (flops / seconds) / 1e9 if seconds > 0 else 0.0
    gbps = (nbytes / seconds) / 1e9 if seconds > 0 else 0.0
    cls = roofline.classify(gflops, seconds, ai, ceilings)
    return {
        "achieved_gflops": gflops, "achieved_gbps": gbps, "arithmetic_intensity": ai,
        "fraction_of_compute_ceiling": gflops / ceilings.compute_gflops if ceilings.compute_gflops else 0.0,
        "classification": cls.label, "note": cls.note,
    }


def _profile_shape_full(name: str, cfg: "dict[str, int]", ceilings: "roofline.Ceilings") -> dict:
    """Complete current-production forward + backward decomposition for one
    shape, plus real end-to-end cross-check calls for both directions."""
    _check_fits_in_vram(cfg)
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]

    # -- forward --------------------------------------------------------
    fwd_cand = _ForwardCandidates(cfg)
    Hout, Wout = fwd_cand.Hout, fwd_cand.Wout
    fwd_decision = _forward_dispatch_decision(cfg)

    fwd_stages = {}
    if fwd_decision["blocks_x"] is None:
        forward_ms = _fwd_time_calls(fwd_cand.baseline)
    elif fwd_decision["blocks_x"] <= 2:
        forward_ms = _fwd_time_calls(fwd_cand.candidate_b)
    else:
        forward_ms = _fwd_time_calls(fwd_cand.candidate_a_smem)
        stages = _fwd_interleaved_compare({"im2col": fwd_cand.stage_im2col, "transpose": fwd_cand.stage_transpose})
        stages["matmul"] = _fwd_time_calls(fwd_cand.stage_matmul)
        stages["permute"] = _fwd_time_calls(fwd_cand.stage_permute)
        fwd_stages = stages

    real_conv2d_ms = _fwd_time_calls(_real_conv2d_call(fwd_cand))

    # -- backward ---------------------------------------------------------
    raw = _RawConv2dBackward(cfg)
    dweight_decision = _dweight_dispatch_decision(cfg)

    d_input = _time_phase(lambda: raw.backward_input(None))
    d_bias = _time_phase(lambda: raw.backward_bias(None))

    dweight_stages = {}
    if dweight_decision["blocks_y"] is None:
        d_weight = _time_phase(lambda: raw.backward_weight(None))
        dweight_total_ms = d_weight["mean_ms"]
    else:
        current = _RawDweightCurrent(cfg)
        permute_t = _time_phase(current.permute)
        if dweight_decision["blocks_y"] == 1:
            gemm_t = _time_phase(current.fused_gemm)
            dweight_stages = {"permute_ms": permute_t, "fused_gemm_ms": gemm_t}
            dweight_total_ms = permute_t["mean_ms"] + gemm_t["mean_ms"]
        else:
            im2col_t = _time_phase(current.im2col_stage)
            gemm_t = _time_phase(current.splitk_gemm)
            dweight_stages = {"permute_ms": permute_t, "im2col_ms": im2col_t, "splitk_gemm_ms": gemm_t}
            dweight_total_ms = permute_t["mean_ms"] + im2col_t["mean_ms"] + gemm_t["mean_ms"]
        d_weight = {"mean_ms": dweight_total_ms}

    backward_total_ms = d_input["mean_ms"] + dweight_total_ms + d_bias["mean_ms"]
    real_conv2d_backward_ms = _time_phase(_real_conv2d_backward_call(raw, cfg))

    fwd_flops = roofline.flops_conv2d_forward(N, Cout, Hout, Wout, Cin, K, K)
    fwd_bytes = roofline.bytes_conv2d_forward(N, Cin, H, W, Cout, K, K, Hout, Wout)
    din_flops = roofline.flops_conv2d_dinput(N, Cin, H, W, Cout, K, K)
    din_bytes = roofline.bytes_conv2d_dinput(N, Cin, H, W, Cout, K, K, Hout, Wout)
    dw_flops = roofline.flops_conv2d_dweight(Cout, Cin, K, K, N, Hout, Wout)
    dw_bytes = roofline.bytes_conv2d_dweight(N, Cin, H, W, Cout, K, K, Hout, Wout)

    return {
        "name": name, "config": cfg, "Hout": Hout, "Wout": Wout,
        "forward_dispatch": fwd_decision, "dweight_dispatch": dweight_decision,
        "forward_ms": forward_ms, "forward_stages_ms": fwd_stages,
        "real_conv2d_call_ms": real_conv2d_ms,
        "d_input_ms": d_input, "d_weight_ms": d_weight, "d_bias_ms": d_bias,
        "dweight_stages_ms": dweight_stages,
        "backward_total_ms": backward_total_ms,
        "real_conv2d_backward_call_ms": real_conv2d_backward_ms,
        "d_input_pct_of_backward": 100.0 * d_input["mean_ms"] / backward_total_ms,
        "d_weight_pct_of_backward": 100.0 * dweight_total_ms / backward_total_ms,
        "d_bias_pct_of_backward": 100.0 * d_bias["mean_ms"] / backward_total_ms,
        "roofline_forward": _classify(forward_ms["mean_ms"], fwd_flops, fwd_bytes, ceilings),
        "roofline_dinput": _classify(d_input["mean_ms"], din_flops, din_bytes, ceilings),
        "roofline_dweight": _classify(dweight_total_ms, dw_flops, dw_bytes, ceilings),
    }


def _profile_forward_only(name: str, cfg: "dict[str, int]", ceilings: "roofline.Ceilings") -> dict:
    """Forward-only decomposition, for the M41 K/stride/channel sweep shapes
    (Section 4 of the M42 brief) -- these are not part of the 7 backward
    representative shapes and are never fed through `_check_fits_in_vram`'s
    backward-buffer sizing, matching `m41_conv2d_forward_profile`'s own
    handling of the same shapes."""
    _check_fits_in_vram(cfg)
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    cand = _ForwardCandidates(cfg)
    Hout, Wout = cand.Hout, cand.Wout
    decision = _forward_dispatch_decision(cfg)

    if decision["blocks_x"] is None:
        forward_ms = _fwd_time_calls(cand.baseline)
    elif decision["blocks_x"] <= 2:
        forward_ms = _fwd_time_calls(cand.candidate_b)
    else:
        forward_ms = _fwd_time_calls(cand.candidate_a_smem)

    fwd_flops = roofline.flops_conv2d_forward(N, Cout, Hout, Wout, Cin, K, K)
    fwd_bytes = roofline.bytes_conv2d_forward(N, Cin, H, W, Cout, K, K, Hout, Wout)
    return {
        "name": name, "config": cfg, "Hout": Hout, "Wout": Wout,
        "forward_dispatch": decision, "forward_ms": forward_ms,
        "roofline_forward": _classify(forward_ms["mean_ms"], fwd_flops, fwd_bytes, ceilings),
    }


# -- Section 8: nvcc -Xptxas -v resource analysis, forward + backward kernels --


def _run_nvcc_ptxas_verbose() -> dict:
    from forge.backend.cuda import build as _build

    src = _build._SOURCE
    arch = _build._ARCH
    scratch = Path(src).parent / "_m42_ptxas_scratch.obj"
    cmd = ["nvcc", "-Xptxas", "-v", "-arch=" + arch, "-c", str(src), "-o", str(scratch)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        msvc_bin = _build._find_msvc_bin()
        if msvc_bin is not None:
            env = os.environ.copy()
            env["PATH"] = str(msvc_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    text = result.stdout + result.stderr

    kernels_of_interest = [
        "k_conv2d_forward", "k_conv2d_output_permute", "k_conv2d_forward_halffused_gemm",
        "k_matmul", "k_matmul_splitk", "k_im2col_conv2d", "k_im2col_conv2d_smem",
        "k_conv2d_backward_input_channelfused", "k_conv2d_backward_weight_reduce",
        "k_dweight_halffused_gemm_splitk", "k_conv2d_grad_output_permute",
    ]
    per_kernel: "dict[str, list]" = {name: [] for name in kernels_of_interest}
    current = None
    for line in text.splitlines():
        for name in kernels_of_interest:
            if name in line and ("_Z" in line or "for '" in line or name + "<" in line or name + "(" in line):
                current = name
        if "registers" in line and current is not None:
            per_kernel[current].append(line.strip())

    if scratch.exists():
        scratch.unlink()
    return {"per_kernel_lines": per_kernel, "returncode": result.returncode}


# -- Section 10: memory characterization of the im2col/GEMM forward path's temporaries --


def _forward_temp_buffer_bytes(cfg: "dict[str, int]", itemsize: int = 4) -> dict:
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    S, P = cfg["S"], cfg["P"]
    Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
    M = N * Hout * Wout
    Kdim = Cin * K * K
    xcol_bytes = M * Kdim * itemsize
    weightT_bytes = Kdim * Cout * itemsize
    out_mat_bytes = M * Cout * itemsize
    input_bytes = N * Cin * H * W * itemsize
    return {
        "xcol_bytes": xcol_bytes, "weightT_bytes": weightT_bytes, "out_mat_bytes": out_mat_bytes,
        "total_temp_bytes": xcol_bytes + weightT_bytes + out_mat_bytes,
        "xcol_vs_input_ratio": xcol_bytes / input_bytes if input_bytes else 0.0,
    }


def _profile_forward_memory(largest_shape_name: str, cfg: "dict[str, int]") -> dict:
    import gc

    gc.collect()
    forge.cuda.empty_cache()
    before = forge.cuda.memory_stats().as_dict()

    cand = _ForwardCandidates(cfg)
    decision = _forward_dispatch_decision(cfg)
    if decision["blocks_x"] is not None and decision["blocks_x"] > 2:
        for _ in range(3):
            cand.candidate_a_smem()
    forge.cuda.synchronize()
    after = forge.cuda.memory_stats().as_dict()

    gc.collect()
    forge.cuda.empty_cache()
    after_gc = forge.cuda.memory_stats().as_dict()

    return {
        "shape": largest_shape_name, "dispatch": decision,
        "modeled_temp_buffers": _forward_temp_buffer_bytes(cfg),
        "allocator_before": before, "allocator_after_3_calls": after,
        "allocator_after_gc_and_empty_cache": after_gc,
    }


# -- Amdahl (mirrors M40's formula) ------------------------------------------


def _amdahl(fractions: "dict[str, float]", speedups=(1.5, 2.0, 3.0)) -> "dict[str, dict[str, float]]":
    out = {}
    for name, frac in fractions.items():
        out[name] = {f"{s}x": 1.0 / ((1 - frac) + frac / s) for s in speedups}
    return out


def _run() -> dict:
    print("-- Phase 0: fresh hardware ceilings (M35 methodology, this session) --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s, "
          f"bandwidth ceiling: {ceilings.bandwidth_gbps:.2f} GB/s")

    print("-- Phase 1: nvcc -Xptxas -v (forward + backward kernel resource analysis) --")
    ptxas = _run_nvcc_ptxas_verbose()
    for name, lines in ptxas["per_kernel_lines"].items():
        print(f"  {name}: {len(lines)} launch-config line(s) found")

    print("-- Phase 2: current-production full forward+backward decomposition (7 shapes + batch sweep) --")
    shapes = [_profile_shape_full(name, cfg, ceilings) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape_full(f"batch_{n}", cfg, ceilings))

    print("-- Phase 3: forward-only K/stride/channel sweep (M41 shapes, fresh) --")
    sweep_results = [_profile_forward_only(name, cfg, ceilings) for name, cfg in SWEEP_SHAPES.items()]

    print("-- Phase 4: memory characterization of the im2col/GEMM forward path's temporaries --")
    # `SHAPES`/`BATCH_SWEEP_BASE`'s Cout<=32 never reaches `blocks_x>2` (Candidate
    # A, the only forward path with an `Xcol` buffer at all) -- every one of them
    # dispatches to Candidate B (no `Xcol`), per M41's own finding. `cout_high`
    # (Cout=128) is the one sweep shape that actually exercises Candidate A, so
    # it is the only shape where this section's question is even applicable.
    forward_memory = _profile_forward_memory("cout_high", SWEEP_SHAPES["cout_high"])

    print("-- Phase 5: MNIST kernel-contribution ranking (m35_mnist, fresh) --")
    mnist_ranking_profile = m35_mnist._run(ceilings)

    print("-- Phase 6: CrossEntropy roofline (m35_kernels reduction/CE profiler, fresh) --")
    reduction_profile = m35_kernels._profile_reduction(ceilings)
    cross_entropy = [r for r in reduction_profile if r["op"].startswith("cross_entropy")]

    print("-- Phase 7: async pipeline profile + allocator/pinned-memory characterization --")
    async_batch_sweep = [_profile_async_epoch(bs, prefetch_size=2, n_samples=1024) for bs in (32, 64, 128)]
    prefetch_depth_sweep = [_profile_async_epoch(64, prefetch_size=d, n_samples=1024) for d in (1, 2, 3)]
    allocator_and_pinned = _profile_allocator_and_pinned(64, 1024)

    # -- Amdahl analysis, using this session's own freshly measured fractions --
    ranking = mnist_ranking_profile["kernel_ranking"]
    total_step_s = sum(r["mean_seconds"] for r in ranking)
    by_op = {r["op"]: r["mean_seconds"] for r in ranking}
    conv_bwd_s = by_op.get("backward:conv2d", 0.0)
    conv_fwd_s = by_op.get("forward:Conv2d", 0.0)
    matmul_bwd_s = by_op.get("backward:@", 0.0)

    mnist_shapes = [s for s in shapes if s["name"] in ("mnist_conv1", "mnist_conv2")]
    din_share = sum(s["d_input_ms"]["mean_ms"] for s in mnist_shapes)
    dw_share = sum(s["d_weight_ms"]["mean_ms"] for s in mnist_shapes)
    db_share = sum(s["d_bias_ms"]["mean_ms"] for s in mnist_shapes)
    fwd_share = sum(s["forward_ms"]["mean_ms"] for s in mnist_shapes)
    conv_bwd_total = din_share + dw_share + db_share

    fractions = {
        "conv2d forward (of full step)": conv_fwd_s / total_step_s if total_step_s else 0.0,
        "dInput (of full step)": (conv_bwd_s * (din_share / conv_bwd_total)) / total_step_s if total_step_s and conv_bwd_total else 0.0,
        "dWeight (of full step)": (conv_bwd_s * (dw_share / conv_bwd_total)) / total_step_s if total_step_s and conv_bwd_total else 0.0,
        "conv2d backward total (of full step)": conv_bwd_s / total_step_s if total_step_s else 0.0,
        "matmul backward (of full step)": matmul_bwd_s / total_step_s if total_step_s else 0.0,
    }
    amdahl = _amdahl(fractions)

    return {
        "hardware_ceilings": hw_profile,
        "ptxas": ptxas,
        "shapes": shapes,
        "batch_sweep": batch_sweep,
        "sweep_shapes": sweep_results,
        "forward_memory": forward_memory,
        "mnist_kernel_ranking": mnist_ranking_profile,
        "cross_entropy": cross_entropy,
        "async_batch_sweep": async_batch_sweep,
        "prefetch_depth_sweep": prefetch_depth_sweep,
        "allocator_and_pinned": allocator_and_pinned,
        "amdahl_fractions": fractions,
        "amdahl_projection": amdahl,
        "mnist_fwd_share_ms": fwd_share,
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M42 fresh post-M41 bottleneck re-characterization (940MX, real CUDA) ===", ""]
    lines.append("-- Dispatch verification + full decomposition --")
    header = (f"{'shape':<14}{'fwd_flops':>11}{'blk_x':>6}{'w#':>7}{'blk_y':>6}"
              f"{'fwd(ms)':>9}{'dIn(ms)':>9}{'dW(ms)':>9}{'dB(ms)':>8}{'bwd(ms)':>9}")
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        fd, dd = r["forward_dispatch"], r["dweight_dispatch"]
        lines.append(
            f"{r['name']:<14}{fd['total_flops']:>11}{str(fd['blocks_x']):>6}"
            f"{dd['weight_elements']:>7}{str(dd['blocks_y']):>6}"
            f"{r['forward_ms']['mean_ms']:>9.4f}{r['d_input_ms']['mean_ms']:>9.4f}"
            f"{r['d_weight_ms']['mean_ms']:>9.4f}{r['d_bias_ms']['mean_ms']:>8.4f}"
            f"{r['backward_total_ms']:>9.4f}"
        )
    lines.append("")
    lines.append("-- Roofline classification --")
    for r in profile["shapes"] + profile["batch_sweep"]:
        rf_ = r["roofline_forward"]
        di = r["roofline_dinput"]
        dw = r["roofline_dweight"]
        lines.append(
            f"  {r['name']:<14} fwd={rf_['fraction_of_compute_ceiling']*100:5.1f}%({rf_['classification']:<24})  "
            f"dIn={di['fraction_of_compute_ceiling']*100:5.1f}%({di['classification']:<24})  "
            f"dW={dw['fraction_of_compute_ceiling']*100:5.1f}%({dw['classification']})"
        )
    lines.append("")
    lines.append("-- MNIST kernel-contribution ranking (this session) --")
    for r in profile["mnist_kernel_ranking"]["kernel_ranking"][:8]:
        lines.append(f"  {r['op']:<22}{r['percent_of_step']:6.2f}%  {r['mean_seconds']*1e3:8.4f}ms  {r.get('classification','n/a')}")
    lines.append("")
    lines.append("-- CrossEntropy roofline --")
    for r in profile["cross_entropy"]:
        lines.append(f"  {r['op']:<24}{str(r['scale']):<20}{r['timing']['mean_s']*1e3:8.4f}ms  {r.get('classification','n/a')}")
    lines.append("")
    lines.append("-- Amdahl projection (hypothetical per-component speedup -> overall step speedup) --")
    for name, proj in profile["amdahl_projection"].items():
        frac = profile["amdahl_fractions"][name]
        lines.append(f"  {name:<38} frac={frac*100:5.2f}%  " + "  ".join(f"{k}->{v:.3f}x" for k, v in proj.items()))
    lines.append("")
    lines.append("-- Async pipeline batch sweep --")
    for r in profile["async_batch_sweep"]:
        lines.append(f"  batch={r['batch_size']:>4}  {r['samples_per_sec']:9.0f} samples/sec  util={r['compute_stream_utilization']*100:5.1f}%")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m42_bottleneck_recharacterization.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m42_bottleneck_recharacterization requires real CUDA hardware.")
        return

    profile = _run()
    print(_render_report(profile))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"environment": collect_environment(), "profile": profile}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved profile -> {output_path}")


if __name__ == "__main__":
    main()
