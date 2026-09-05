"""Milestone 39: `im2col` materialization -- reuse/instruction-bound investigation.

M38 left `k_im2col_conv2d` (`experimental_conv_im2col.im2col`) as the
dominant remaining `dWeight` cost at every `blocks_y >= 2` shape (`Cout >
16` -- the regime M38's half-fused GEMM cannot help, since its
redundant-regather tax grows with `blocks_y`). This script:

1. Reproduces the M38 baseline `im2col` timings fresh at the 7 representative
   shapes (`im2col` is byte-for-byte unchanged since M34).
2. Quantifies the input-patch reuse available at each shape
   (`reuse_factor = Hout*Wout*KH*KW / (H*W)`, the average number of times a
   non-boundary input pixel is read across the whole `Xcol` construction).
3. Benchmarks the two candidates in `experimental_conv_im2col_reuse.py`
   (Candidate A: index-hoisted/table-lookup; Candidate B: shared-memory
   input-plane staging) against that baseline, interleaved, at the 7
   shapes plus a `KH=KW` sweep (1, 3, 5) and a `stride` sweep (1, 2) at a
   fixed base shape isolating exactly how reuse opportunity changes speed.
4. Reports peak reserved memory for each candidate (Candidate A allocates
   three small `K`-sized int32 tables per call in addition to `Xcol`;
   Candidate B allocates only `Xcol`, identical to the baseline).

    python -m benchmarks.m39_im2col_reuse_profile
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available
from forge.backend.cuda.experimental_conv_im2col import im2col
from forge.backend.cuda.experimental_conv_im2col_reuse import im2col_indexed, im2col_smem
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


def _make_x(N, Cin, H, W, seed=0):
    rng = np.random.default_rng(seed)
    x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")._data
    forge.cuda.synchronize()
    return x


def _reuse_factor(H, W, KH, KW, Hout, Wout) -> float:
    """Average number of times a (non-boundary) input pixel is read while
    building `Xcol` -- `Hout*Wout*KH*KW / (H*W)`, exact for `stride=1`
    interior pixels, an upper bound near boundaries/`stride>1` (some
    (kh,kw,ho,wo) combinations fall outside the input and read nothing)."""
    return (Hout * Wout * KH * KW) / (H * W)


def _profile_shape(name: str, cfg: "dict[str, int]") -> dict:
    _check_fits_in_vram(cfg)
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    S, P = cfg["S"], cfg["P"]
    Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
    backend = get_cuda_backend()
    x = _make_x(N, Cin, H, W)

    args = (N, Cin, H, W, K, K, S, S, P, P, Hout, Wout)
    baseline_t = _time_call(lambda: im2col(backend, x, *args))
    indexed_t = _time_call(lambda: im2col_indexed(backend, x, *args))
    smem_t = _time_call(lambda: im2col_smem(backend, x, *args))

    peak_baseline = _peak_reserved_mb(lambda: im2col(backend, x, *args))
    peak_indexed = _peak_reserved_mb(lambda: im2col_indexed(backend, x, *args))
    peak_smem = _peak_reserved_mb(lambda: im2col_smem(backend, x, *args))

    M = N * Hout * Wout
    Kdim = Cin * K * K
    xcol_mb = (M * Kdim * 4) / (1024 * 1024)

    return {
        "name": name,
        "config": cfg,
        "weight_elements": Cout * Kdim,
        "M": M,
        "K": Kdim,
        "Hout": Hout,
        "Wout": Wout,
        "xcol_mb": xcol_mb,
        "reuse_factor": _reuse_factor(H, W, K, K, Hout, Wout),
        "baseline_ms": baseline_t,
        "indexed_ms": indexed_t,
        "smem_ms": smem_t,
        "speedup_indexed": baseline_t["mean_ms"] / indexed_t["mean_ms"],
        "speedup_smem": baseline_t["mean_ms"] / smem_t["mean_ms"],
        "peak_reserved_mb_baseline": peak_baseline,
        "peak_reserved_mb_indexed": peak_indexed,
        "peak_reserved_mb_smem": peak_smem,
    }


def _kernel_size_sweep() -> list:
    """Isolates reuse-factor as a function of KH=KW at a fixed base spatial/channel shape."""
    N, Cin, H, W = 64, 16, 28, 28
    backend = get_cuda_backend()
    x = _make_x(N, Cin, H, W)
    results = []
    for K in (1, 3, 5):
        P = K // 2
        Hout, Wout = _hout_wout(H, K, 1, P), _hout_wout(W, K, 1, P)
        args = (N, Cin, H, W, K, K, 1, 1, P, P, Hout, Wout)
        baseline_t = _time_call(lambda: im2col(backend, x, *args))
        indexed_t = _time_call(lambda: im2col_indexed(backend, x, *args))
        smem_t = _time_call(lambda: im2col_smem(backend, x, *args))
        results.append({
            "K": K,
            "reuse_factor": _reuse_factor(H, W, K, K, Hout, Wout),
            "baseline_ms": baseline_t["mean_ms"],
            "indexed_ms": indexed_t["mean_ms"],
            "smem_ms": smem_t["mean_ms"],
            "speedup_indexed": baseline_t["mean_ms"] / indexed_t["mean_ms"],
            "speedup_smem": baseline_t["mean_ms"] / smem_t["mean_ms"],
        })
    return results


def _stride_sweep() -> list:
    """Isolates reuse-factor as a function of stride at a fixed K=3 base shape."""
    N, Cin, H, W, K = 64, 16, 28, 28, 3
    backend = get_cuda_backend()
    x = _make_x(N, Cin, H, W)
    results = []
    for S in (1, 2):
        P = 1
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        args = (N, Cin, H, W, K, K, S, S, P, P, Hout, Wout)
        baseline_t = _time_call(lambda: im2col(backend, x, *args))
        indexed_t = _time_call(lambda: im2col_indexed(backend, x, *args))
        smem_t = _time_call(lambda: im2col_smem(backend, x, *args))
        results.append({
            "stride": S,
            "reuse_factor": _reuse_factor(H, W, K, K, Hout, Wout),
            "baseline_ms": baseline_t["mean_ms"],
            "indexed_ms": indexed_t["mean_ms"],
            "smem_ms": smem_t["mean_ms"],
            "speedup_indexed": baseline_t["mean_ms"] / indexed_t["mean_ms"],
            "speedup_smem": baseline_t["mean_ms"] / smem_t["mean_ms"],
        })
    return results


def _cout_dispatch_regime_sweep() -> list:
    """Cout sweep restricted to the blocks_y>=2 regime `im2col` is actually
    reached in production (Cout > 16) -- confirms the candidates' behavior
    does not depend on Cout at all (im2col never reads/writes grad_output),
    included for completeness/documentation rather than as new evidence."""
    base = dict(BATCH_SWEEP_BASE)
    base["N"] = 64
    backend = get_cuda_backend()
    results = []
    for cout in (17, 32, 64):
        cfg = dict(base)
        cfg["Cout"] = cout
        _check_fits_in_vram(cfg)
        results.append(_profile_shape(f"cout_{cout}", cfg))
    return results


def _render_report(profile: dict) -> str:
    lines = ["=== M39 im2col reuse/instruction-bound profile (940MX, real CUDA) ===", ""]
    header = f"{'shape':<14}{'weight#':>8}{'reuse':>7}{'base(ms)':>10}{'idx(ms)':>9}{'smem(ms)':>10}{'idx_x':>8}{'smem_x':>8}"
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        lines.append(
            f"{r['name']:<14}{r['weight_elements']:>8}{r['reuse_factor']:>7.2f}"
            f"{r['baseline_ms']['mean_ms']:>10.3f}{r['indexed_ms']['mean_ms']:>9.3f}{r['smem_ms']['mean_ms']:>10.3f}"
            f"{r['speedup_indexed']:>7.3f}x{r['speedup_smem']:>7.3f}x"
        )
    lines.append("")
    lines.append("Kernel-size sweep (Cin=16,H=W=28,N=64,stride=1):")
    lines.append(f"{'K':>4}{'reuse':>8}{'base(ms)':>10}{'idx(ms)':>9}{'smem(ms)':>10}{'idx_x':>8}{'smem_x':>8}")
    for r in profile["kernel_size_sweep"]:
        lines.append(
            f"{r['K']:>4}{r['reuse_factor']:>8.2f}{r['baseline_ms']:>10.3f}{r['indexed_ms']:>9.3f}{r['smem_ms']:>10.3f}"
            f"{r['speedup_indexed']:>7.3f}x{r['speedup_smem']:>7.3f}x"
        )
    lines.append("")
    lines.append("Stride sweep (Cin=16,H=W=28,N=64,K=3):")
    lines.append(f"{'stride':>6}{'reuse':>8}{'base(ms)':>10}{'idx(ms)':>9}{'smem(ms)':>10}{'idx_x':>8}{'smem_x':>8}")
    for r in profile["stride_sweep"]:
        lines.append(
            f"{r['stride']:>6}{r['reuse_factor']:>8.2f}{r['baseline_ms']:>10.3f}{r['indexed_ms']:>9.3f}{r['smem_ms']:>10.3f}"
            f"{r['speedup_indexed']:>7.3f}x{r['speedup_smem']:>7.3f}x"
        )
    lines.append("")
    lines.append("Peak reserved MB (baseline / indexed / smem):")
    for r in profile["shapes"]:
        lines.append(
            f"  {r['name']:<14}{r['peak_reserved_mb_baseline']:>8.2f} / "
            f"{r['peak_reserved_mb_indexed']:>8.2f} / {r['peak_reserved_mb_smem']:>8.2f}"
        )
    return "\n".join(lines)


def _run() -> dict:
    shapes = [_profile_shape(name, cfg) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape(f"batch_{n}", cfg))
    return {
        "shapes": shapes,
        "batch_sweep": batch_sweep,
        "kernel_size_sweep": _kernel_size_sweep(),
        "stride_sweep": _stride_sweep(),
        "cout_regime_sweep": _cout_dispatch_regime_sweep(),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m39_im2col_reuse_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m39_im2col_reuse_profile requires real CUDA hardware.")
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
