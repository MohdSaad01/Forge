"""Milestone 38: root-cause + candidate evaluation for `dWeight`'s `im2col` cost.

M37 decomposed the M34 im2col+GEMM `dWeight` pipeline and found `im2col`
(`k_im2col_conv2d`) + `grad_output` permute together cost 54-63% of total
pipeline time -- more than the GEMM itself -- and fixed the GEMM's own
occupancy shortfall via a split-K variant (`dweight_im2col_gemm_splitk`,
the current production dispatch above `_CONV2D_WEIGHT_IM2COL_GEMM_
THRESHOLD`). This script reproduces that baseline fresh (im2col/permute are
byte-for-byte unchanged since M34, so the isolated-stage numbers are
expected to match M37's within run-to-run hardware variance) and measures
Milestone 38's Candidate B (`experimental_conv_halffused.
dweight_halffused_gemm_splitk` -- eliminates the `Xcol` buffer, keeps
`dYcolT`) against the M37 baseline at all 7 representative shapes, plus a
`Cout` sweep isolating the `blocks_y = ceil(Cout/16)` discriminator that
determines whether Candidate B wins or regresses.

    python -m benchmarks.m38_im2col_profile
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import _MATMUL_TILE, get_cuda_backend, is_cuda_available
from forge.backend.cuda.experimental_conv_halffused import dweight_halffused_gemm_splitk
from forge.backend.cuda.experimental_conv_im2col import (
    dweight_im2col_gemm_splitk,
    grad_output_permute,
    im2col,
)
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _check_fits_in_vram, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30


def _time_call(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    for _ in range(warmup):
        call()
        forge.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(None)
        call()
        end = TimedEvent()
        end.record(None)
        forge.cuda.synchronize()
        samples.append(elapsed_ms(start, end))
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _make_inputs(cfg: "dict[str, int]"):
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    S, P = cfg["S"], cfg["P"]
    Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
    rng = np.random.default_rng(0)
    x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")._data
    go = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")._data
    forge.cuda.synchronize()
    return x, go, (Cout, Cin, K, K), (S, S), (P, P), Hout, Wout


def _peak_reserved_mb(call, iterations: int = 20) -> float:
    gc.collect()
    forge.cuda.empty_cache()
    forge.cuda.reset_peak_memory_stats()
    for _ in range(iterations):
        result = call()
        del result
    forge.cuda.synchronize()
    peak = forge.cuda.memory_stats().peak_reserved_bytes
    gc.collect()
    forge.cuda.empty_cache()
    return peak / (1024 * 1024)


def _profile_shape(name: str, cfg: "dict[str, int]") -> dict:
    _check_fits_in_vram(cfg)
    x, go, weight_shape, stride, padding, Hout, Wout = _make_inputs(cfg)
    backend = get_cuda_backend()
    N, Cin, Cout, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["K"]
    M = N * Hout * Wout
    Kdim = Cin * K * K
    blocks_y = (Cout + _MATMUL_TILE - 1) // _MATMUL_TILE
    blocks_x = (Kdim + _MATMUL_TILE - 1) // _MATMUL_TILE

    im2col_t = _time_call(lambda: im2col(backend, x, N, Cin, cfg["H"], cfg["W"], K, K, stride[0], stride[1], padding[0], padding[1], Hout, Wout))
    permute_t = _time_call(lambda: grad_output_permute(backend, go, N, Cout, Hout, Wout))
    baseline_t = _time_call(lambda: dweight_im2col_gemm_splitk(backend, go, x, weight_shape, stride, padding))
    candidateB_t = _time_call(lambda: dweight_halffused_gemm_splitk(backend, go, x, weight_shape, stride, padding))
    full_backward_t = _time_call(
        lambda: backend.conv2d_backward(
            go, x,
            forge.Tensor(np.zeros(weight_shape, dtype=np.float32), device="cuda")._data,
            forge.Tensor(np.zeros(weight_shape[0], dtype=np.float32), device="cuda")._data,
            stride, padding,
        )
    )

    peak_baseline_mb = _peak_reserved_mb(lambda: dweight_im2col_gemm_splitk(backend, go, x, weight_shape, stride, padding))
    peak_candidateB_mb = _peak_reserved_mb(lambda: dweight_halffused_gemm_splitk(backend, go, x, weight_shape, stride, padding))

    return {
        "name": name,
        "config": cfg,
        "weight_elements": Cout * Kdim,
        "M": M,
        "K": Kdim,
        "Cout": Cout,
        "blocks_y": blocks_y,
        "blocks_x": blocks_x,
        "total_gemm_blocks": blocks_y * blocks_x,
        "im2col_ms": im2col_t,
        "permute_ms": permute_t,
        "baseline_splitk_ms": baseline_t,
        "candidate_b_halffused_ms": candidateB_t,
        "speedup_candidate_b_over_baseline": baseline_t["mean_ms"] / candidateB_t["mean_ms"],
        "full_conv2d_backward_ms": full_backward_t,
        "peak_reserved_mb_baseline": peak_baseline_mb,
        "peak_reserved_mb_candidate_b": peak_candidateB_mb,
    }


def _cout_sweep() -> list:
    """Isolates the `blocks_y = ceil(Cout/16)` discriminator directly."""
    base = dict(BATCH_SWEEP_BASE)
    base["N"] = 64
    backend = get_cuda_backend()
    results = []
    for cout in (8, 16, 17, 24, 32, 48, 64):
        cfg = dict(base)
        cfg["Cout"] = cout
        _check_fits_in_vram(cfg)
        x, go, weight_shape, stride, padding, Hout, Wout = _make_inputs(cfg)
        blocks_y = (cout + _MATMUL_TILE - 1) // _MATMUL_TILE
        baseline_t = _time_call(lambda: dweight_im2col_gemm_splitk(backend, go, x, weight_shape, stride, padding))
        candidateB_t = _time_call(lambda: dweight_halffused_gemm_splitk(backend, go, x, weight_shape, stride, padding))
        results.append({
            "cout": cout,
            "blocks_y": blocks_y,
            "baseline_ms": baseline_t["mean_ms"],
            "candidate_b_ms": candidateB_t["mean_ms"],
            "speedup": baseline_t["mean_ms"] / candidateB_t["mean_ms"],
        })
    return results


def _render_report(profile: dict) -> str:
    lines = ["=== M38 im2col root-cause + Candidate B evaluation (940MX, real CUDA) ===", ""]
    header = f"{'shape':<14}{'weight#':>8}{'blocks_y':>9}{'blocks_x':>9}{'im2col(ms)':>12}{'baseline(ms)':>14}{'candB(ms)':>11}{'speedup':>9}"
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        lines.append(
            f"{r['name']:<14}{r['weight_elements']:>8}{r['blocks_y']:>9}{r['blocks_x']:>9}"
            f"{r['im2col_ms']['mean_ms']:>12.3f}{r['baseline_splitk_ms']['mean_ms']:>14.3f}"
            f"{r['candidate_b_halffused_ms']['mean_ms']:>11.3f}{r['speedup_candidate_b_over_baseline']:>8.3f}x"
        )
    lines.append("")
    lines.append("Cout sweep (blocks_y discriminator, fixed Cin=16,H=W=28,K=3,N=64):")
    lines.append(f"{'Cout':>6}{'blocks_y':>9}{'baseline(ms)':>14}{'candB(ms)':>11}{'speedup':>9}")
    for r in profile["cout_sweep"]:
        lines.append(f"{r['cout']:>6}{r['blocks_y']:>9}{r['baseline_ms']:>14.3f}{r['candidate_b_ms']:>11.3f}{r['speedup']:>8.3f}x")
    return "\n".join(lines)


def _run() -> dict:
    shapes = [_profile_shape(name, cfg) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape(f"batch_{n}", cfg))
    cout_sweep = _cout_sweep()
    return {"shapes": shapes, "batch_sweep": batch_sweep, "cout_sweep": cout_sweep}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m38_im2col_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m38_im2col_profile requires real CUDA hardware.")
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
