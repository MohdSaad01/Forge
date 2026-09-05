"""M37 Phase 1/2: quantitative decomposition of the M34 im2col+GEMM `dWeight`
path, plus a GEMM-shape/occupancy diagnosis.

M36's roofline update named `dWeight` (im2col + the existing M11 tiled
`k_matmul`, unmodified since M34) as the now-dominant component of Conv2d
backward. This script asks M37's central question directly: is the M34
*algorithm* itself the bottleneck, or is one of its three stages -- im2col
construction, `grad_output` permutation, or the GEMM call -- disproportionately
expensive relative to the others and to the hardware's roofline ceilings?

Reuses, rather than re-derives:

* `benchmarks.conv2d_backward_weight_im2col_profile._RawIm2colGemm` (M34) for
  every isolated-stage timing (`current`/`im2col`/`permute`/`gemm`/`total`),
  at the same 7 representative shapes (`conv2d_backward_profile.SHAPES` +
  `BATCH_SWEEP_BASE`/`BATCH_SIZES`).
* `benchmarks.roofline` (M35) for FLOP/byte-traffic counting, arithmetic
  intensity, and the practical compute/bandwidth ceilings already measured
  in `benchmarks/results/m35_summary.json`.
* `CUDABackend.conv2d_backward` directly (unmodified) for the *actual*
  production "complete Conv2d backward" time, through the real M34 dispatch
  -- not the M32 profiler's raw single-kernel-only timing.

New in this script: a GEMM launch-geometry model. `k_matmul`'s 16x16 tiling
(`MATMUL_TILE`, `kernels.cu`) launches `ceil(GEMM_M/16) * ceil(GEMM_N/16)`
thread blocks, each 256 threads. Forge's dWeight GEMM orientation
(`dweight_im2col_gemm`) is `cf_matmul(A=dYcolT, B=Xcol, GEMM_M=Cout,
GEMM_K=N*Hout*Wout, GEMM_N=Cin*KH*KW)` -- so `GEMM_M`/`GEMM_N` are the *small*
Conv2d weight dimensions (8-64 output channels; 9-288 `Cin*KH*KW`) while the
huge `N*Hout*Wout` reduction lives entirely inside `GEMM_K`, invisible to
block count. The 940MX has 3 SMs and (Maxwell/CC5.0) a 2048-resident-thread
cap per SM, i.e. 8 resident 256-thread blocks/SM, 24 blocks device-wide --
so a GEMM whose `ceil(M/16)*ceil(N/16)` block count is well below 24 cannot
occupy the whole device even though its total FLOP count is large, regardless
of how efficient each individual block's inner loop is. This is measured
directly (not inferred from elapsed time alone, per the milestone brief) by
comparing the *isolated-GEMM* achieved GFLOP/s against a small-square-GEMM
control at matched total FLOPs but balanced M/N, and by reporting occupancy
launch geometry explicitly for every shape.

    python -m benchmarks.m37_dweight_profile
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import _SUFFIX, get_cuda_backend, is_cuda_available
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from . import roofline
from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _check_fits_in_vram, _hout_wout
from .conv2d_backward_weight_im2col_profile import _RawIm2colGemm
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30

# -- 940MX launch-geometry constants (measured directly via the CUDA driver
# API's `cuDeviceGetAttribute`, not a spec sheet transcription -- see the
# M37 report's Phase 2 for the exact query). -------------------------------
SM_COUNT = 3
MAX_THREADS_PER_SM = 2048
MATMUL_TILE = 16
THREADS_PER_BLOCK = MATMUL_TILE * MATMUL_TILE  # 256
BLOCKS_PER_SM_CAP = MAX_THREADS_PER_SM // THREADS_PER_BLOCK  # 8 (thread-count-limited, not register/shared-mem)
DEVICE_BLOCK_CAPACITY = SM_COUNT * BLOCKS_PER_SM_CAP  # 24


def _gemm_launch_geometry(gemm_m: int, gemm_n: int) -> dict:
    blocks_y = (gemm_m + MATMUL_TILE - 1) // MATMUL_TILE
    blocks_x = (gemm_n + MATMUL_TILE - 1) // MATMUL_TILE
    total_blocks = blocks_x * blocks_y
    return {
        "gemm_m": gemm_m,
        "gemm_n": gemm_n,
        "blocks_y": blocks_y,
        "blocks_x": blocks_x,
        "total_blocks": total_blocks,
        "device_block_capacity": DEVICE_BLOCK_CAPACITY,
        "occupancy_fraction": min(1.0, total_blocks / DEVICE_BLOCK_CAPACITY),
    }


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    stream_handle = None
    for _ in range(warmup):
        code = call(stream_handle)
        assert code == 0, f"kernel launch failed with code {code}"
        forge.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(stream_handle)
        code = call(stream_handle)
        assert code == 0, f"kernel launch failed with code {code}"
        end = TimedEvent()
        end.record(stream_handle)
        forge.cuda.synchronize()
        samples.append(elapsed_ms(start, end))

    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _time_production_conv2d_backward(cfg: "dict[str, int]", iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    """Times the real `CUDABackend.conv2d_backward` production dispatch end-to-end."""
    backend = get_cuda_backend()
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    S, P = cfg["S"], cfg["P"]
    Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)

    rng = np.random.default_rng(0)
    x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
    w = forge.Tensor(rng.standard_normal((Cout, Cin, K, K)).astype(np.float32), device="cuda")
    b = forge.Tensor(rng.standard_normal((Cout,)).astype(np.float32), device="cuda")
    grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
    forge.cuda.synchronize()

    def call(_stream_handle):
        backend.conv2d_backward(grad_out._data, x._data, w._data, b._data, (S, S), (P, P))
        return 0

    for _ in range(warmup):
        call(None)
        forge.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(None)
        call(None)
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


def _profile_shape(name: str, cfg: "dict[str, int]", ceilings: roofline.Ceilings) -> dict:
    _check_fits_in_vram(cfg)
    raw = _RawIm2colGemm(cfg)

    current = _time_phase(raw.current)
    im2col_t = _time_phase(raw.im2col)
    permute_t = _time_phase(raw.permute)
    gemm_t = _time_phase(raw.gemm)
    total_t = _time_phase(raw.total)
    full_backward = _time_production_conv2d_backward(cfg)

    total_experimental_ms = im2col_t["mean_ms"] + permute_t["mean_ms"] + gemm_t["mean_ms"]

    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    Hout, Wout = raw.Hout, raw.Wout

    flops = roofline.flops_conv2d_dweight(Cout, Cin, K, K, N, Hout, Wout)
    nbytes = roofline.bytes_conv2d_dweight(N, Cin, H, W, Cout, K, K, Hout, Wout, itemsize=4)
    ai = roofline.arithmetic_intensity(flops, nbytes)

    def _perf(ms: float) -> dict:
        seconds = ms / 1000.0
        gflops = (flops / seconds) / 1e9 if seconds > 0 else 0.0
        gbps = (nbytes / seconds) / 1e9 if seconds > 0 else 0.0
        cls = roofline.classify(gflops, seconds, ai, ceilings)
        return {
            "achieved_gflops": gflops,
            "achieved_gbps": gbps,
            "fraction_of_compute_ceiling": gflops / ceilings.compute_gflops if ceilings.compute_gflops > 0 else 0.0,
            "fraction_of_bandwidth_ceiling": gbps / ceilings.bandwidth_gbps if ceilings.bandwidth_gbps > 0 else 0.0,
            "classification": cls.label,
        }

    geometry = _gemm_launch_geometry(raw.Cout, raw.Kdim)

    # Isolated-GEMM-only roofline: GEMM's own FLOPs are the same conv2d
    # dweight FLOP count (im2col/permute are pure data movement, 0 FLOPs);
    # its own byte traffic is the minimum GEMM traffic on the *already built*
    # Xcol/dYcolT operands (Section 8: never blame the GEMM for gather cost).
    gemm_nbytes = roofline.bytes_matmul_minimum(raw.Cout, raw.Kdim, raw.M, itemsize=4)
    gemm_ai = roofline.arithmetic_intensity(flops, gemm_nbytes)

    def _gemm_perf(ms: float) -> dict:
        seconds = ms / 1000.0
        gflops = (flops / seconds) / 1e9 if seconds > 0 else 0.0
        gbps = (gemm_nbytes / seconds) / 1e9 if seconds > 0 else 0.0
        cls = roofline.classify(gflops, seconds, gemm_ai, ceilings)
        return {
            "achieved_gflops": gflops,
            "achieved_gbps": gbps,
            "fraction_of_compute_ceiling": gflops / ceilings.compute_gflops if ceilings.compute_gflops > 0 else 0.0,
            "fraction_of_bandwidth_ceiling": gbps / ceilings.bandwidth_gbps if ceilings.bandwidth_gbps > 0 else 0.0,
            "classification": cls.label,
        }

    return {
        "name": name,
        "config": cfg,
        "weight_elements": raw.weight_elements,
        "M": raw.M,
        "K": raw.Kdim,
        "Cout": raw.Cout,
        "gemm_launch_geometry": geometry,
        "flops": flops,
        "bytes_minimum": nbytes,
        "arithmetic_intensity": ai,
        "gemm_arithmetic_intensity": gemm_ai,
        "current_kernel_ms": current,
        "im2col_ms": im2col_t,
        "permute_ms": permute_t,
        "gemm_ms": gemm_t,
        "summed_total_experimental_ms": total_experimental_ms,
        "measured_total_pipeline_ms": total_t,
        "full_conv2d_backward_ms": full_backward,
        "stage_share_pct": {
            "im2col": 100.0 * im2col_t["mean_ms"] / total_experimental_ms,
            "permute": 100.0 * permute_t["mean_ms"] / total_experimental_ms,
            "gemm": 100.0 * gemm_t["mean_ms"] / total_experimental_ms,
        },
        "dweight_pct_of_full_backward": 100.0 * total_t["mean_ms"] / full_backward["mean_ms"],
        "roofline_total_pipeline": _perf(total_t["mean_ms"]),
        "roofline_gemm_only": _gemm_perf(gemm_t["mean_ms"]),
        "roofline_current_kernel": _perf(current["mean_ms"]),
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M37 dWeight decomposition + roofline (940MX, real CUDA) ===", ""]
    header = (
        f"{'shape':<14}{'weight#':>8}{'blocks':>7}{'occ%':>6}"
        f"{'im2col%':>9}{'permute%':>9}{'gemm%':>7}"
        f"{'GFLOP/s':>9}{'%compute':>9}{'class':>18}"
    )
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        rl = r["roofline_total_pipeline"]
        sh = r["stage_share_pct"]
        geo = r["gemm_launch_geometry"]
        lines.append(
            f"{r['name']:<14}{r['weight_elements']:>8}{geo['total_blocks']:>7}"
            f"{geo['occupancy_fraction'] * 100:>5.1f}%"
            f"{sh['im2col']:>8.1f}%{sh['permute']:>8.1f}%{sh['gemm']:>6.1f}%"
            f"{rl['achieved_gflops']:>9.2f}{rl['fraction_of_compute_ceiling'] * 100:>8.1f}%"
            f"{rl['classification']:>18}"
        )
    lines.append("")
    lines.append(
        "blocks = GEMM launch block count (ceil(Cout/16)*ceil(Cin*KH*KW/16)); "
        f"device_block_capacity={DEVICE_BLOCK_CAPACITY} (3 SMs x 8 resident 256-thread blocks/SM). "
        "occ% = total_blocks/device_block_capacity, clamped at 100%."
    )
    return "\n".join(lines)


def _run() -> dict:
    ceilings = roofline.load_ceilings("benchmarks/results/m35_summary.json")
    shapes = [_profile_shape(name, cfg, ceilings) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape(f"batch_{n}", cfg, ceilings))
    return {
        "ceilings": {"compute_gflops": ceilings.compute_gflops, "bandwidth_gbps": ceilings.bandwidth_gbps},
        "shapes": shapes,
        "batch_sweep": batch_sweep,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m37_dweight_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m37_dweight_profile requires real CUDA hardware.")
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
