"""M44: fresh post-M43 CUDA bottleneck re-characterization (measurement-only).

M43 replaced the `blocks_y==1` dWeight path's atomic split-K combine
(`k_dweight_halffused_gemm_splitk`, M38) with a deterministic two-stage
reduction (`k_dweight_halffused_gemm_splitk_partial` + `k_dweight_splitk_
reduce`, M43) -- a measured 1.11-1.24x component win at `mnist_conv2`,
1.02-1.04x at `large_spatial`. `CUDABackend.conv2d_backward`'s `blocks_y==1`
branch now calls `experimental_conv_dweight_tworeduce.dweight_halffused_
gemm_splitk_tworeduce` in place of M38's kernel; the dispatch *condition*
itself (`weight_elements`/`blocks_y` thresholds) is unchanged. No milestone
since M42 has re-measured the whole pipeline fresh against this new
production path, and M42's own `_RawDweightCurrent` dWeight sub-stage
helper (`m40_bottleneck_recharacterization.py`) still decomposes
`blocks_y==1` as "permute + fused-GEMM" (M38) -- stale after M43.

This script does exactly that:

1. **Dispatch verification** (read fresh from `backend.py`/`kernels.cu`,
   not from documentation): forward FLOPs threshold + `blocks_x` regime
   (unchanged since M41), dWeight `weight_elements`/`blocks_y` thresholds
   with the `blocks_y==1` path corrected to M43's two-stage reduction, and
   dInput's `Cin<=16` channel-fused/fallback boundary (unchanged since M36,
   confirmed directly against `kernels.cu`'s `CONV2D_DINPUT_CHANNELFUSED_
   MAX_CIN` launcher).
2. **Full decomposition of the CURRENT production path** at the 7
   established representative shapes + batch sweep: forward (M41 dispatch,
   unchanged), dInput, dWeight's *actual* current sub-stages (permute +
   partial-GEMM + reduce for `blocks_y==1`; im2col[-smem] + permute +
   split-K GEMM for `blocks_y>=2`; the unchanged M21 kernel below the
   weight-element threshold), dBias, plus real end-to-end `CUDABackend.
   conv2d()`/`conv2d_backward()` calls for cross-check -- mirrors M42's own
   `_profile_shape_full` exactly, with the `blocks_y==1` branch corrected.
3. **M43 stage analysis** (Section 5 of the M44 brief): profiles the
   partial-GEMM and reduce stages *separately* (not just their sum) at
   every `blocks_y==1` shape, plus a fresh same-session interleaved A/B
   (M38 atomic vs. M43 two-stage) at `mnist_conv2`/`large_spatial` reusing
   `m43_dweight_splitk_profile._candidate_b_comparison` directly -- so this
   milestone's own numbers confirm M43's win still holds, rather than
   trusting M43's report at face value.
4. **Targeted sweeps** (Section 3 of the M44 brief): a dWeight `Cout` sweep
   spanning both `blocks_y==1` and `blocks_y>=2` at a fixed base shape, a
   dedicated below-256-weight-element sweep (never touched by M43, twice
   rejected in M33/M34), and a dInput `Cin` sweep straddling the M36
   channel-fused/fallback boundary (`Cin` in {1, 8, 16, 17, 32, 64}).
5. **Fresh roofline ceilings** (M35 methodology, this session) and
   classification of every candidate above.
6. **MNIST kernel-contribution ranking + CrossEntropy roofline**: reuses
   `m35_mnist._run`/`m35_kernels._profile_reduction` directly (fresh).
7. **Pipeline health**: reuses `pipeline_profile._profile_async_epoch`/
   `_profile_allocator_and_pinned` directly (fresh) -- confirms M43
   introduced no synchronization/allocator/prefetch regression.
8. **Bottleneck ranking + Amdahl** at 1.25x/1.5x/2x per the M44 brief (M42
   used 1.5x/2x/3x) using this session's own measured fractions, with
   dWeight's `blocks_y==1` GEMM path and the below-256 block-reduce path
   now ranked as *separate* candidates (M42 combined them into one
   "dWeight" row).

No production code is modified or reimplemented differently than
production already does -- every stage call below is the same function
`backend.py` itself calls (either directly, or through the same
experimental module `backend.py` imports).

    python -m benchmarks.m44_bottleneck_recharacterization
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import forge
from forge.backend.cuda.backend import (
    _CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD,
    _MATMUL_TILE,
    is_cuda_available,
)
from forge.backend.cuda.experimental_conv_im2col import recommended_num_k_splits

from . import m35_hardware, m35_kernels, m35_mnist, roofline
from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _RawConv2dBackward, _check_fits_in_vram, _hout_wout
from .environment import collect_environment
from .m40_bottleneck_recharacterization import _RawDweightCurrent
from .m40_bottleneck_recharacterization import _dispatch_decision as _m38_style_dweight_dispatch_decision
from .m41_conv2d_forward_profile import SWEEP_SHAPES, _ForwardCandidates
from .m41_conv2d_forward_profile import _interleaved_compare as _fwd_interleaved_compare
from .m41_conv2d_forward_profile import _time_calls as _fwd_time_calls
from .m42_bottleneck_recharacterization import (
    _classify,
    _forward_dispatch_decision,
    _forward_temp_buffer_bytes,
    _profile_forward_memory,
    _profile_forward_only,
    _real_conv2d_backward_call,
    _real_conv2d_call,
    _run_nvcc_ptxas_verbose,
    _time_phase,
)
from .m43_dweight_splitk_profile import _RawDweightTworeduce, _candidate_b_comparison
from .pipeline_profile import _profile_allocator_and_pinned, _profile_async_epoch

WARMUP = 5
ITERATIONS = 30


# -- Section 1/2: dWeight dispatch verification, corrected for M43 ----------


def _dweight_dispatch_decision(cfg: "dict[str, int]") -> dict:
    """Mirrors `CUDABackend.conv2d_backward`'s dWeight dispatch exactly
    (same threshold/`blocks_y` arithmetic as M40's own `_dispatch_decision`),
    but with the `blocks_y==1` path label corrected to M43's production
    function -- M40/M42's own decision helper still names M38's (now dead)
    atomic-combine kernel."""
    decision = dict(_m38_style_dweight_dispatch_decision(cfg))
    if decision["blocks_y"] == 1:
        decision["path"] = "dweight_halffused_gemm_splitk_tworeduce (M43, permute + partial-GEMM + reduce)"
    return decision


def _dinput_dispatch_decision(cfg: "dict[str, int]") -> dict:
    """Mirrors `cf_conv2d_backward_input_*`'s own internal C-level dispatch
    (`kernels.cu`'s `CONV2D_DINPUT_CHANNELFUSED_MAX_CIN = 16`, unchanged
    since M36) -- confirmed directly against the compiled kernel source,
    not inferred from documentation."""
    Cin = cfg["Cin"]
    if Cin <= 16:
        return {"Cin": Cin, "path": "k_conv2d_backward_input_channelfused (M36)"}
    return {"Cin": Cin, "path": "k_conv2d_backward_input (M32 original, fallback -- unused by any real Forge shape)"}


# -- Section 2: dWeight sub-stage timing for the CURRENT (post-M43) blocks_y==1 path --


def _dweight_tworeduce_stages(cfg: "dict[str, int]") -> dict:
    """Isolated permute / partial-GEMM / reduce timing for the current
    production `blocks_y==1` dWeight path, at the production
    `recommended_num_k_splits` value -- the exact three sub-stages M44's
    brief (Section 5) asks be profiled separately."""
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    S, P = cfg["S"], cfg["P"]
    Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
    M = N * Hout * Wout
    recommended = recommended_num_k_splits(M)

    inst = _RawDweightTworeduce(cfg, max_splits=recommended)
    permute_t = _time_phase(lambda: inst.permute(None))
    partial_t = _time_phase(
        lambda: inst.fn_partial(
            inst.dycolT_ptr, inst.x.ptr, inst.partial_ptr, *inst.shape_args,
            ctypes.c_int(recommended), None,
        )
    )
    reduce_t = _time_phase(
        lambda: inst.fn_reduce(
            inst.partial_ptr, inst.out_ptr,
            ctypes.c_int(inst.Cout), ctypes.c_int(inst.Kdim), ctypes.c_int(recommended), None,
        )
    )
    total_ms = permute_t["mean_ms"] + partial_t["mean_ms"] + reduce_t["mean_ms"]
    return {
        "recommended_num_k_splits": recommended,
        "permute_ms": permute_t, "partial_gemm_ms": partial_t, "reduce_ms": reduce_t,
        "total_ms": total_ms,
        "partial_gemm_pct_of_dweight": 100.0 * partial_t["mean_ms"] / total_ms,
        "reduce_pct_of_dweight": 100.0 * reduce_t["mean_ms"] / total_ms,
        "permute_pct_of_dweight": 100.0 * permute_t["mean_ms"] / total_ms,
    }


# -- Section 2: full forward+backward decomposition at the 7 shapes + batch sweep --


def _profile_shape_full(name: str, cfg: "dict[str, int]", ceilings: "roofline.Ceilings") -> dict:
    _check_fits_in_vram(cfg)
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]

    # -- forward (M41 dispatch, unchanged by M43) ------------------------
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

    # -- backward -----------------------------------------------------------
    raw = _RawConv2dBackward(cfg)
    dweight_decision = _dweight_dispatch_decision(cfg)
    dinput_decision = _dinput_dispatch_decision(cfg)

    d_input = _time_phase(lambda: raw.backward_input(None))
    d_bias = _time_phase(lambda: raw.backward_bias(None))

    dweight_stages = {}
    if dweight_decision["blocks_y"] is None:
        d_weight = _time_phase(lambda: raw.backward_weight(None))
        dweight_total_ms = d_weight["mean_ms"]
    elif dweight_decision["blocks_y"] == 1:
        tw = _dweight_tworeduce_stages(cfg)
        dweight_stages = {"permute_ms": tw["permute_ms"], "partial_gemm_ms": tw["partial_gemm_ms"], "reduce_ms": tw["reduce_ms"]}
        dweight_total_ms = tw["total_ms"]
        d_weight = {"mean_ms": dweight_total_ms}
    else:
        current = _RawDweightCurrent(cfg)
        permute_t = _time_phase(current.permute)
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
        "forward_dispatch": fwd_decision, "dweight_dispatch": dweight_decision, "dinput_dispatch": dinput_decision,
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


# -- Section 3: dWeight Cout sweep spanning both blocks_y regimes -----------


def _dweight_cout_sweep(ceilings: "roofline.Ceilings") -> list:
    """`Cout` in {2,4,8,16} (`blocks_y==1`) and {32,64,128} (`blocks_y>=2`)
    at a fixed `Cin=16,H=W=13,K=3` base (`weight_elements = 144*Cout`,
    always >= 256 -- stays in the GEMM dispatch regime throughout)."""
    base = {"N": 64, "Cin": 16, "H": 13, "W": 13, "K": 3, "S": 1, "P": 1}
    results = []
    for cout in (2, 4, 8, 16, 32, 64, 128):
        cfg = dict(base)
        cfg["Cout"] = cout
        _check_fits_in_vram(cfg)
        decision = _dweight_dispatch_decision(cfg)
        Hout, Wout = _hout_wout(cfg["H"], cfg["K"], cfg["S"], cfg["P"]), _hout_wout(cfg["W"], cfg["K"], cfg["S"], cfg["P"])

        if decision["blocks_y"] == 1:
            tw = _dweight_tworeduce_stages(cfg)
            total_ms = tw["total_ms"]
            stages_ms = {"permute_ms": tw["permute_ms"]["mean_ms"], "partial_gemm_ms": tw["partial_gemm_ms"]["mean_ms"], "reduce_ms": tw["reduce_ms"]["mean_ms"]}
        else:
            current = _RawDweightCurrent(cfg)
            permute_t = _time_phase(current.permute)
            im2col_t = _time_phase(current.im2col_stage)
            gemm_t = _time_phase(current.splitk_gemm)
            total_ms = permute_t["mean_ms"] + im2col_t["mean_ms"] + gemm_t["mean_ms"]
            stages_ms = {"permute_ms": permute_t["mean_ms"], "im2col_ms": im2col_t["mean_ms"], "splitk_gemm_ms": gemm_t["mean_ms"]}

        flops = roofline.flops_conv2d_dweight(cout, cfg["Cin"], cfg["K"], cfg["K"], cfg["N"], Hout, Wout)
        nbytes = roofline.bytes_conv2d_dweight(cfg["N"], cfg["Cin"], cfg["H"], cfg["W"], cout, cfg["K"], cfg["K"], Hout, Wout)
        results.append({
            "Cout": cout, "config": cfg, "dispatch": decision,
            "stages_ms": stages_ms, "total_ms": total_ms,
            "roofline": _classify(total_ms, flops, nbytes, ceilings),
        })
    return results


# -- Section 3: below-256 weight-element sweep (never touched by M43) -------


def _dweight_below_threshold_sweep(ceilings: "roofline.Ceilings") -> list:
    """`weight_elements = Cout*Cin*K*K` in {72, 144, 144, 216, 243}, all
    below `_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD=256` -- the M21 per-thread/
    block-reduce kernel path, twice investigated (M33 cooperative reduction,
    M34 im2col+GEMM) and rejected both times, never touched by M43. Fixed
    `N=64,H=W=28,K=3` (mnist_conv1's own spatial shape)."""
    base = {"N": 64, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1}
    points = [
        ("mnist_conv1 (we=72)", 1, 8),
        ("we_144a", 1, 16),
        ("we_144b", 2, 8),
        ("we_216", 3, 8),
        ("we_243", 1, 27),
    ]
    results = []
    for label, cin, cout in points:
        cfg = dict(base)
        cfg["Cin"], cfg["Cout"] = cin, cout
        _check_fits_in_vram(cfg)
        decision = _dweight_dispatch_decision(cfg)
        assert decision["blocks_y"] is None, f"{label} unexpectedly reached GEMM dispatch: {decision}"

        raw = _RawConv2dBackward(cfg)
        d_weight = _time_phase(lambda: raw.backward_weight(None))
        Hout, Wout = raw.Hout, raw.Wout
        flops = roofline.flops_conv2d_dweight(cout, cin, cfg["K"], cfg["K"], cfg["N"], Hout, Wout)
        nbytes = roofline.bytes_conv2d_dweight(cfg["N"], cin, cfg["H"], cfg["W"], cout, cfg["K"], cfg["K"], Hout, Wout)
        results.append({
            "label": label, "config": cfg, "weight_elements": decision["weight_elements"],
            "d_weight_ms": d_weight, "roofline": _classify(d_weight["mean_ms"], flops, nbytes, ceilings),
        })
    return results


# -- Section 3: dInput Cin sweep straddling the M36 channel-fused boundary --


def _dinput_cin_sweep(ceilings: "roofline.Ceilings") -> list:
    """`Cin` in {1, 8, 16, 17, 32, 64} -- brackets `CONV2D_DINPUT_
    CHANNELFUSED_MAX_CIN=16` on both sides. Fixed `N=16,Cout=16,H=W=28,K=3`
    (M41's own `_SWEEP_BASE` shape)."""
    base = {"N": 16, "Cout": 16, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1}
    results = []
    for cin in (1, 8, 16, 17, 32, 64):
        cfg = dict(base)
        cfg["Cin"] = cin
        _check_fits_in_vram(cfg)
        decision = _dinput_dispatch_decision(cfg)

        raw = _RawConv2dBackward(cfg)
        d_input = _time_phase(lambda: raw.backward_input(None))
        Hout, Wout = raw.Hout, raw.Wout
        flops = roofline.flops_conv2d_dinput(cfg["N"], cin, cfg["H"], cfg["W"], cfg["Cout"], cfg["K"], cfg["K"])
        nbytes = roofline.bytes_conv2d_dinput(cfg["N"], cin, cfg["H"], cfg["W"], cfg["Cout"], cfg["K"], cfg["K"], Hout, Wout)
        results.append({
            "Cin": cin, "config": cfg, "dispatch": decision,
            "d_input_ms": d_input, "roofline": _classify(d_input["mean_ms"], flops, nbytes, ceilings),
        })
    return results


# -- Amdahl (M44 brief: 1.25x/1.5x/2x, not M42's 1.5x/2x/3x) ----------------


def _amdahl(fractions: "dict[str, float]", speedups=(1.25, 1.5, 2.0)) -> "dict[str, dict[str, float]]":
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

    print("-- Phase 2: current-production full forward+backward decomposition (7 shapes + batch sweep) --")
    shapes = [_profile_shape_full(name, cfg, ceilings) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape_full(f"batch_{n}", cfg, ceilings))

    print("-- Phase 3: M43 stage analysis (partial-GEMM vs. reduce, + fresh interleaved M38-vs-M43 A/B) --")
    m43_ab = [
        _candidate_b_comparison(name, SHAPES[name])
        for name in ("mnist_conv2", "large_spatial")
        if _dweight_dispatch_decision(SHAPES[name])["blocks_y"] == 1
    ]

    print("-- Phase 4: targeted sweeps (dWeight Cout, below-256 weight-elements, dInput Cin) --")
    dweight_cout_sweep = _dweight_cout_sweep(ceilings)
    dweight_below_threshold = _dweight_below_threshold_sweep(ceilings)
    dinput_cin_sweep = _dinput_cin_sweep(ceilings)

    print("-- Phase 5: forward-only K/stride/channel sweep (M41 shapes, fresh) --")
    sweep_results = [_profile_forward_only(name, cfg, ceilings) for name, cfg in SWEEP_SHAPES.items()]

    print("-- Phase 6: forward im2col/GEMM temp-buffer memory characterization --")
    forward_memory = _profile_forward_memory("cout_high", SWEEP_SHAPES["cout_high"])

    print("-- Phase 7: MNIST kernel-contribution ranking (m35_mnist, fresh) --")
    mnist_ranking_profile = m35_mnist._run(ceilings)

    print("-- Phase 8: CrossEntropy roofline (m35_kernels reduction/CE profiler, fresh) --")
    reduction_profile = m35_kernels._profile_reduction(ceilings)
    cross_entropy = [r for r in reduction_profile if r["op"].startswith("cross_entropy")]

    print("-- Phase 9: async pipeline profile + allocator/pinned-memory characterization --")
    async_batch_sweep = [_profile_async_epoch(bs, prefetch_size=2, n_samples=1024) for bs in (32, 64, 128)]
    prefetch_depth_sweep = [_profile_async_epoch(64, prefetch_size=d, n_samples=1024) for d in (1, 2, 3)]
    allocator_and_pinned = _profile_allocator_and_pinned(64, 1024)

    # -- bottleneck ranking + Amdahl, using this session's own fresh fractions --
    ranking = mnist_ranking_profile["kernel_ranking"]
    total_step_s = sum(r["mean_seconds"] for r in ranking)
    by_op = {r["op"]: r["mean_seconds"] for r in ranking}
    conv_bwd_s = by_op.get("backward:conv2d", 0.0)
    conv_fwd_s = by_op.get("forward:Conv2d", 0.0)
    matmul_bwd_s = by_op.get("backward:@", 0.0)
    ce_bwd_s = by_op.get("backward:cross_entropy", 0.0)

    mnist_conv1 = next(s for s in shapes if s["name"] == "mnist_conv1")  # below-256 block-reduce
    mnist_conv2 = next(s for s in shapes if s["name"] == "mnist_conv2")  # blocks_y==1 GEMM
    din_share = mnist_conv1["d_input_ms"]["mean_ms"] + mnist_conv2["d_input_ms"]["mean_ms"]
    dw_below256_share = mnist_conv1["d_weight_ms"]["mean_ms"]
    dw_blocksy1_share = mnist_conv2["d_weight_ms"]["mean_ms"]
    db_share = mnist_conv1["d_bias_ms"]["mean_ms"] + mnist_conv2["d_bias_ms"]["mean_ms"]
    conv_bwd_total = din_share + dw_below256_share + dw_blocksy1_share + db_share

    def _frac(share: float) -> float:
        return (conv_bwd_s * (share / conv_bwd_total)) / total_step_s if total_step_s and conv_bwd_total else 0.0

    fractions = {
        "conv2d forward (of full step)": conv_fwd_s / total_step_s if total_step_s else 0.0,
        "dInput (of full step)": _frac(din_share),
        "dWeight blocks_y==1 (M43 tworeduce, of full step)": _frac(dw_blocksy1_share),
        "dWeight below-256 block-reduce (of full step)": _frac(dw_below256_share),
        "dBias (of full step)": _frac(db_share),
        "conv2d backward total (of full step)": conv_bwd_s / total_step_s if total_step_s else 0.0,
        "matmul backward (of full step)": matmul_bwd_s / total_step_s if total_step_s else 0.0,
        "CrossEntropy backward (of full step)": ce_bwd_s / total_step_s if total_step_s else 0.0,
    }
    amdahl = _amdahl(fractions)

    return {
        "hardware_ceilings": hw_profile,
        "ptxas": ptxas,
        "shapes": shapes,
        "batch_sweep": batch_sweep,
        "m43_stage_analysis": m43_ab,
        "dweight_cout_sweep": dweight_cout_sweep,
        "dweight_below_threshold_sweep": dweight_below_threshold,
        "dinput_cin_sweep": dinput_cin_sweep,
        "sweep_shapes": sweep_results,
        "forward_memory": forward_memory,
        "mnist_kernel_ranking": mnist_ranking_profile,
        "cross_entropy": cross_entropy,
        "async_batch_sweep": async_batch_sweep,
        "prefetch_depth_sweep": prefetch_depth_sweep,
        "allocator_and_pinned": allocator_and_pinned,
        "amdahl_fractions": fractions,
        "amdahl_projection": amdahl,
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M44 fresh post-M43 bottleneck re-characterization (940MX, real CUDA) ===", ""]
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
    lines.append("-- M43 stage analysis (mnist_conv2 / large_spatial, blocks_y==1) --")
    for r in profile["m43_stage_analysis"]:
        lines.append(f"  {r['name']}: recommended_num_k_splits={r['recommended_num_k_splits']}")
        for row in r["sweep"]:
            lines.append(
                f"    splits={row['num_k_splits']:>4}  gemm-only speedup={row['speedup_gemm_only']:.3f}x  "
                f"full speedup={row['speedup_full']:.3f}x"
            )
    lines.append("")
    lines.append("-- dWeight Cout sweep (blocks_y==1 vs. blocks_y>=2) --")
    for r in profile["dweight_cout_sweep"]:
        lines.append(
            f"  Cout={r['Cout']:>4}  blocks_y={r['dispatch']['blocks_y']}  total={r['total_ms']:.4f}ms  "
            f"{r['roofline']['fraction_of_compute_ceiling']*100:5.1f}% of ceiling"
        )
    lines.append("")
    lines.append("-- dWeight below-256 weight-element sweep --")
    for r in profile["dweight_below_threshold_sweep"]:
        lines.append(
            f"  {r['label']:<20} we={r['weight_elements']:>4}  {r['d_weight_ms']['mean_ms']:.4f}ms  "
            f"{r['roofline']['fraction_of_compute_ceiling']*100:5.1f}% of ceiling"
        )
    lines.append("")
    lines.append("-- dInput Cin sweep (channel-fused boundary at Cin<=16) --")
    for r in profile["dinput_cin_sweep"]:
        lines.append(
            f"  Cin={r['Cin']:>3}  {r['dispatch']['path']:<60}  {r['d_input_ms']['mean_ms']:.4f}ms  "
            f"{r['roofline']['fraction_of_compute_ceiling']*100:5.1f}% of ceiling"
        )
    lines.append("")
    lines.append("-- MNIST kernel-contribution ranking (this session) --")
    for r in profile["mnist_kernel_ranking"]["kernel_ranking"][:8]:
        lines.append(f"  {r['op']:<22}{r['percent_of_step']:6.2f}%  {r['mean_seconds']*1e3:8.4f}ms  {r.get('classification','n/a')}")
    lines.append("")
    lines.append("-- Amdahl projection (hypothetical per-component speedup -> overall step speedup) --")
    for name, proj in profile["amdahl_projection"].items():
        frac = profile["amdahl_fractions"][name]
        lines.append(f"  {name:<48} frac={frac*100:5.2f}%  " + "  ".join(f"{k}->{v:.3f}x" for k, v in proj.items()))
    lines.append("")
    lines.append("-- Async pipeline batch sweep --")
    for r in profile["async_batch_sweep"]:
        lines.append(f"  batch={r['batch_size']:>4}  {r['samples_per_sec']:9.0f} samples/sec  util={r['compute_stream_utilization']*100:5.1f}%")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m44_bottleneck_recharacterization.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m44_bottleneck_recharacterization requires real CUDA hardware.")
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
