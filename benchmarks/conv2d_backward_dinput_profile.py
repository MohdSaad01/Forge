"""Dedicated CUDA Conv2d `dInput` structural-candidate profiler (Milestone 36).

M35 (`docs/performance/m35-roofline-characterization.md`) measured `dInput`
reaching only ~12% of the 940MX's practical compute ceiling (104.57 GFLOP/s)
while using well under 20% of its practical bandwidth ceiling (15.09 GB/s) at
every representative shape -- far below *both* ceilings despite an
arithmetic intensity (4-48 FLOPs/byte) that places most shapes decisively
compute-bound by the roofline model (ridge point ~6.93 FLOPs/byte). Per
Section 14/15 of the M36 brief, that combination points away from a memory-
bandwidth-reuse problem and toward instruction efficiency: `nvcc -Xptxas -v`
on the existing `k_conv2d_backward_input` (`kernels.cu`) confirms a 512-byte
per-thread stack frame -- the M32 `kh_valid`/`ho_valid`/`kw_valid`/`wo_valid`
local arrays, dynamically indexed and therefore never register-resident,
read back `Cout * h_count * w_count` times per thread from local memory.

This script isolates and times three structurally different candidates
against the unchanged production kernel, at the same 7 representative shapes
`conv2d_backward_profile.py` (M32) established:

  * `current`         -- the existing `cf_conv2d_backward_input_*` kernel
                          (unchanged -- see kernels.cu's M32 section)
  * `smem@T`           -- Candidate A: shared-memory grad_output row-tile
                           reuse across `Cin`, for `T in {64, 128, 256}`
                           threads/block (Section 20's block-size experiment)
  * `channelfused`     -- Candidate B: one thread per `(n, hi, wi)` across
                           all `Cin` at once, register-resident accumulators,
                           no local-memory index tables
  * `warpreduce@W`     -- Candidate C: one warp (32 lanes) per output
                           element, cooperative reduction over `Cout`, for
                           `W in {2, 4, 8}` warps/block

    python -m benchmarks.conv2d_backward_dinput_profile
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _check_fits_in_vram, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30
BLOCK_SIZES = (64, 128, 256)
WARPS_PER_BLOCK = (2, 4, 8)
MAX_CIN_REG = 16  # must match kernels.cu's k_conv2d_backward_input_channelfused


class _RawDInput:
    """Directly-callable, isolated `dInput` kernel variants (profiling-only)."""

    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        self.N, self.Cin, self.Cout, self.H, self.W, self.K = N, Cin, Cout, H, W, K
        self.Hout, self.Wout = Hout, Wout
        self.supports_channelfused = Cin <= MAX_CIN_REG

        rng = np.random.default_rng(0)
        w = forge.Tensor(rng.standard_normal((Cout, Cin, K, K)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()

        self.w, self.grad_out = w._data, grad_out._data
        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.grad_x_ptr = self.backend._alloc(N * Cin * H * W * 4)
        self.fn_current = getattr(self.lib, "cf_conv2d_backward_input_f32")
        self.fn_smem = getattr(self.lib, "cf_conv2d_backward_input_smem_f32")
        self.fn_channelfused = getattr(self.lib, "cf_conv2d_backward_input_channelfused_f32")
        self.fn_warpreduce = getattr(self.lib, "cf_conv2d_backward_input_warpreduce_f32")

    def current(self, stream_handle) -> int:
        return self.fn_current(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, stream_handle)

    def smem(self, threads_per_block: int, stream_handle) -> int:
        return self.fn_smem(
            self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args,
            ctypes.c_int(threads_per_block), stream_handle,
        )

    def channelfused(self, stream_handle) -> int:
        return self.fn_channelfused(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, stream_handle)

    def warpreduce(self, warps_per_block: int, stream_handle) -> int:
        return self.fn_warpreduce(
            self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args,
            ctypes.c_int(warps_per_block), stream_handle,
        )


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    # Per-iteration synchronization (not just after warmup): several
    # candidates are far slower than production at some shapes -- queuing
    # many such launches back-to-back on the null stream risks tripping
    # Windows WDDM's ~2s TDR watchdog, the same reasoning M33's profiler
    # documents.
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


def _profile_shape(name: str, cfg: "dict[str, int]") -> dict:
    _check_fits_in_vram(cfg)
    raw = _RawDInput(cfg)

    current = _time_phase(raw.current)
    smem = {bs: _time_phase(lambda sh, bs=bs: raw.smem(bs, sh)) for bs in BLOCK_SIZES}
    channelfused = _time_phase(raw.channelfused) if raw.supports_channelfused else None
    warpreduce = {wp: _time_phase(lambda sh, wp=wp: raw.warpreduce(wp, sh)) for wp in WARPS_PER_BLOCK}

    best_smem_bs = min(smem, key=lambda bs: smem[bs]["mean_ms"])
    best_warp_wp = min(warpreduce, key=lambda wp: warpreduce[wp]["mean_ms"])

    candidates_ms = {"smem": smem[best_smem_bs]["mean_ms"], "warpreduce": warpreduce[best_warp_wp]["mean_ms"]}
    if channelfused is not None:
        candidates_ms["channelfused"] = channelfused["mean_ms"]
    best_candidate = min(candidates_ms, key=lambda k: candidates_ms[k])

    return {
        "name": name,
        "config": cfg,
        "Hout": raw.Hout,
        "Wout": raw.Wout,
        "supports_channelfused": raw.supports_channelfused,
        "current_ms": current,
        "smem_ms": {str(bs): v for bs, v in smem.items()},
        "channelfused_ms": channelfused,
        "warpreduce_ms": {str(wp): v for wp, v in warpreduce.items()},
        "best_smem_block_size": best_smem_bs,
        "best_smem_ms": smem[best_smem_bs]["mean_ms"],
        "best_warpreduce_warps_per_block": best_warp_wp,
        "best_warpreduce_ms": warpreduce[best_warp_wp]["mean_ms"],
        "best_candidate": best_candidate,
        "best_candidate_ms": candidates_ms[best_candidate],
        "best_candidate_speedup_vs_current": current["mean_ms"] / candidates_ms[best_candidate],
        "current_vs_smem_speedup": current["mean_ms"] / smem[best_smem_bs]["mean_ms"],
        "current_vs_channelfused_speedup": (
            current["mean_ms"] / channelfused["mean_ms"] if channelfused is not None else None
        ),
        "current_vs_warpreduce_speedup": current["mean_ms"] / warpreduce[best_warp_wp]["mean_ms"],
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M36 dInput structural-candidate profile (940MX, real CUDA) ===", ""]
    header = (
        f"{'shape':<14}{'current(ms)':>13}{'smem(ms)':>11}{'chanfuse(ms)':>13}{'warp(ms)':>11}"
        f"{'best':>14}{'speedup':>9}"
    )
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        cf_ms = r["channelfused_ms"]["mean_ms"] if r["channelfused_ms"] is not None else float("nan")
        lines.append(
            f"{r['name']:<14}{r['current_ms']['mean_ms']:>13.4f}{r['best_smem_ms']:>11.4f}"
            f"{cf_ms:>13.4f}{r['best_warpreduce_ms']:>11.4f}"
            f"{r['best_candidate']:>14}{r['best_candidate_speedup_vs_current']:>8.2f}x"
        )
    return "\n".join(lines)


def _run() -> dict:
    shapes = [_profile_shape(name, cfg) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape(f"batch_{n}", cfg))
    return {"shapes": shapes, "batch_sweep": batch_sweep}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m36_dinput_candidates_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- conv2d_backward_dinput_profile requires real CUDA hardware.")
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
