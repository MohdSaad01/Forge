"""M47: fresh post-M46 CUDA bottleneck re-characterization (measurement-only).

M46 rejected grid-level splitting of the below-`CONV2D_WEIGHT_REDUCE_
THRESHOLD` (256) dWeight block-reduce kernel (`k_conv2d_backward_weight_
reduce`, unchanged since M21): a real `cudaOccupancyMaxActiveBlocksPerMulti
processor` query found production and the grid-split candidate land at
*identical* occupancy (5 blocks/SM, register-bound, 62.5%), so grid-splitting
cannot raise the 940MX's per-SM concurrent-block ceiling. Combined with M45's
prior rejection of warp-shuffle cooperation, the below-256 kernel has now
been attacked from four angles across M21/M33/M45/M46 without a clean win.

This milestone does **not** begin by assuming the below-256 kernel should be
attacked a fifth time. Instead it re-measures the whole CUDA training
pipeline fresh (production dispatch is byte-for-byte unchanged since M43, so
this both confirms M44's own numbers still hold within thermal variance and
gives every candidate an equal, fresh look) and adds exactly one new
measurement M44/M45/M46 never produced: a single-session, three-way
interleaved comparison of production against *both* rejected below-256
candidates (M45's best warp-shuffle configuration and M46's best grid-split
configuration) at once, plus a fresh occupancy re-query -- the combined
evidence needed to formally settle the below-256 kernel's status (Section 7
of the M47 brief) rather than re-litigating either prior milestone
one-candidate-at-a-time.

No production code, kernel, dispatch, or public API is modified.

    python -m benchmarks.m47_bottleneck_recharacterization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.backend.cuda.backend import is_cuda_available

from . import m35_hardware, m35_kernels, m35_mnist, roofline
from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _check_fits_in_vram, _hout_wout
from .environment import collect_environment
from .m44_bottleneck_recharacterization import (
    _amdahl,
    _dinput_cin_sweep,
    _dweight_below_threshold_sweep,
    _dweight_cout_sweep,
    _dweight_dispatch_decision,
    _profile_shape_full,
)
from .m41_conv2d_forward_profile import SWEEP_SHAPES
from .m42_bottleneck_recharacterization import (
    _profile_forward_memory,
    _profile_forward_only,
    _run_nvcc_ptxas_verbose as _m42_ptxas,
)
from .m43_dweight_splitk_profile import _candidate_b_comparison
from .m45_dweight_below256_profile import (
    SUBGROUP_CONFIGS,
    WARPS_PER_BLOCK_SWEEP,
    _RawDWeightBelow256,
    _run_nvcc_ptxas_verbose as _m45_ptxas,
)
from .m46_dweight_below256_gridsplit_profile import (
    NUM_SPLITS_SWEEP,
    _occupancy_query,
    _RawDWeightGridsplit,
    _interleaved_multi_time,
    _run_nvcc_ptxas_verbose as _m46_ptxas,
)
from .pipeline_profile import _profile_allocator_and_pinned, _profile_async_epoch

WARMUP = 5
ITERATIONS = 30


# -- Section 10 (new): three-way below-256 reassessment ----------------------
# Neither M45 nor M46 ever timed their own candidate against the *other*
# milestone's candidate in the same interleaved run -- each was only ever
# compared against production alone. This closes that gap with one fresh,
# same-session, round-robin-interleaved comparison (M46's own
# `_interleaved_multi_time`, the corrected methodology after M46's
# block-sequential-timing retraction) across all three variants at once.

_BELOW256_REASSESS_SHAPES = {
    # mirrors m44's own `_dweight_below_threshold_sweep` base points so this
    # reassessment's shapes line up 1:1 with this session's own fresh
    # below-256 roofline sweep (Section 4/8 of the report).
    "mnist_conv1 (we=72)": {"N": 64, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1, "Cin": 1, "Cout": 8},
    "we_144": {"N": 64, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1, "Cin": 1, "Cout": 16},
    "we_243": {"N": 64, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1, "Cin": 1, "Cout": 27},
}


def _classify_dweight(mean_ms: float, cfg: "dict[str, int]", ceilings: "roofline.Ceilings") -> dict:
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    Hout, Wout = _hout_wout(H, K, cfg["S"], cfg["P"]), _hout_wout(W, K, cfg["S"], cfg["P"])
    flops = roofline.flops_conv2d_dweight(Cout, Cin, K, K, N, Hout, Wout)
    nbytes = roofline.bytes_conv2d_dweight(N, Cin, H, W, Cout, K, K, Hout, Wout)
    ai = roofline.arithmetic_intensity(flops, nbytes)
    achieved_gflops = flops / (mean_ms / 1000) / 1e9 if mean_ms > 0 else 0.0
    ceiling = ceilings.roofline_ceiling_gflops(ai)
    return {"achieved_gflops": achieved_gflops, "ceiling_gflops": ceiling,
            "fraction_of_ceiling": achieved_gflops / ceiling if ceiling > 0 else 0.0}


def _below256_three_way_reassessment(ceilings: "roofline.Ceilings") -> list:
    results = []
    for label, cfg in _BELOW256_REASSESS_SHAPES.items():
        _check_fits_in_vram(cfg)
        warp_raw = _RawDWeightBelow256(cfg)
        grid_raw = _RawDWeightGridsplit(cfg)
        assert warp_raw.weight_elements < 256 and grid_raw.weight_elements < 256

        calls = {"production": warp_raw.current}
        calls.update({
            f"warpreduce_wp{wp}": (lambda sh, wp=wp: warp_raw.warpreduce(wp, sh))
            for wp in WARPS_PER_BLOCK_SWEEP
        })
        calls.update({
            f"warpsubgroup_gs{gs}_wp{wp}": (lambda sh, wp=wp, gs=gs: warp_raw.warpsubgroup(wp, gs, sh))
            for gs, wp in SUBGROUP_CONFIGS
        })
        calls.update({
            f"gridsplit_ns{ns}": (lambda sh, ns=ns: grid_raw.gridsplit(ns, sh))
            for ns in NUM_SPLITS_SWEEP
        })

        timed = _interleaved_multi_time(calls, iterations=ITERATIONS, warmup=WARMUP)
        production_ms = timed["production"]["mean_ms"]
        best_name = min(timed, key=lambda k: timed[k]["mean_ms"])
        best_ms = timed[best_name]["mean_ms"]

        results.append({
            "label": label,
            "config": cfg,
            "weight_elements": warp_raw.weight_elements,
            "timed_ms": timed,
            "production_ms": production_ms,
            "production_classification": _classify_dweight(production_ms, cfg, ceilings),
            "best_variant": best_name,
            "best_ms": best_ms,
            "best_classification": _classify_dweight(best_ms, cfg, ceilings),
            "speedup_production_vs_best": production_ms / best_ms if best_name != "production" else 1.0,
        })
    return results


def _merge_ptxas(*dicts: dict) -> dict:
    merged: "dict[str, list]" = {}
    for d in dicts:
        for name, lines in d["per_kernel_lines"].items():
            merged.setdefault(name, [])
            if lines and lines not in (merged[name],):
                for line in lines:
                    if line not in merged[name]:
                        merged[name].append(line)
    return {"per_kernel_lines": merged}


def _run() -> dict:
    print("-- Phase 0: fresh hardware ceilings (M35 methodology, this session) --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s, "
          f"bandwidth ceiling: {ceilings.bandwidth_gbps:.2f} GB/s")

    print("-- Phase 1: nvcc -Xptxas -v (forward + backward + below-256-candidate kernels) --")
    ptxas = _merge_ptxas(_m42_ptxas(), _m45_ptxas(), _m46_ptxas())

    print("-- Phase 2: current-production full forward+backward decomposition (7 shapes + batch sweep) --")
    shapes = [_profile_shape_full(name, cfg, ceilings) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape_full(f"batch_{n}", cfg, ceilings))

    print("-- Phase 3: M43 blocks_y==1 stage re-confirmation (fresh interleaved A/B) --")
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

    print("-- Phase 10 (new): below-256 three-way reassessment (production vs. M45 vs. M46, one interleaved run) --")
    below256_reassessment = _below256_three_way_reassessment(ceilings)
    below256_occupancy = _occupancy_query()

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
        "below256_three_way_reassessment": below256_reassessment,
        "below256_occupancy": below256_occupancy,
        "amdahl_fractions": fractions,
        "amdahl_projection": amdahl,
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M47 fresh post-M46 bottleneck re-characterization (940MX, real CUDA) ===", ""]
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
    lines.append("-- below-256 three-way reassessment (production vs. M45 warp-shuffle vs. M46 grid-split) --")
    for r in profile["below256_three_way_reassessment"]:
        lines.append(
            f"  {r['label']:<20} we={r['weight_elements']:>4}  production={r['production_ms']:.4f}ms  "
            f"best={r['best_variant']} ({r['best_ms']:.4f}ms)  speedup={r['speedup_production_vs_best']:.3f}x  "
            f"prod%ceil={r['production_classification']['fraction_of_ceiling']*100:.1f}%"
        )
    lines.append("")
    occ = profile["below256_occupancy"]
    lines.append(f"  occupancy (fresh): production={occ['max_active_blocks_per_sm_reduce']} blocks/SM "
                 f"({occ['occupancy_fraction_reduce']*100:.1f}%), gridsplit={occ['max_active_blocks_per_sm_gridsplit']} "
                 f"blocks/SM ({occ['occupancy_fraction_gridsplit']*100:.1f}%)")
    lines.append("")
    lines.append("-- dWeight below-256 weight-element sweep (roofline) --")
    for r in profile["dweight_below_threshold_sweep"]:
        lines.append(
            f"  {r['label']:<20} we={r['weight_elements']:>4}  {r['d_weight_ms']['mean_ms']:.4f}ms  "
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
    parser.add_argument("--output", default="benchmarks/results/m47_bottleneck_recharacterization.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m47_bottleneck_recharacterization requires real CUDA hardware.")
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
