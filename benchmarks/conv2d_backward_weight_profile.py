"""Dedicated CUDA Conv2d `dWeight` cooperative-reduction profiler (Milestone 33).

M32 (`benchmarks/conv2d_backward_profile.py`) isolated and timed all three
`conv2d_backward` kernels and found `dWeight` -- unchanged since Milestone 21
-- now the single largest `conv2d_backward` contributor at 6 of the 7
representative shapes (`docs/performance/conv2d-backward-profiling.md`'s
**New bottleneck ranking**). This script investigates *why*: it isolates
`dWeight`'s two candidate kernels (`k_conv2d_backward_weight`, one thread per
weight element with a full serial reduction; `k_conv2d_backward_weight_reduce`,
one block per weight element with a shared-memory tree reduction) directly,
independent of the M21 dispatch threshold (`CONV2D_WEIGHT_REDUCE_THRESHOLD =
256` weight elements) that currently picks between them -- via the
profiling-only `cf_conv2d_backward_weight_{perthread,blockreduce}_*` exports
added to `kernels.cu` this milestone (see that file's matching comment).

Reuses the exact M32 `SHAPES`/`BATCH_SWEEP_BASE`/`BATCH_SIZES` (Section 5 of
the milestone brief: "do not replace the M32 workloads") so results stay
directly comparable. For each shape, measures:

  * `current`      -- the existing `cf_conv2d_backward_weight_*` dispatcher
                       (whichever kernel the M21 threshold currently picks)
  * `perthread`     -- `k_conv2d_backward_weight` forced, regardless of shape
  * `blockreduce@T` -- `k_conv2d_backward_weight_reduce` forced, for
                        `T in {64, 128, 256}` threads/block (Section 14's
                        block-size experiment)

    python -m benchmarks.conv2d_backward_weight_profile
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
WARPS_PER_BLOCK = (2, 4, 8)  # 64/128/256 threads/block, warp-cooperative candidate
CONV2D_WEIGHT_REDUCE_THRESHOLD = 256  # must match kernels.cu -- see that file's dispatch comment


class _RawDWeight:
    """Directly-callable, isolated `dWeight` kernel variants (profiling-only)."""

    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        self.weight_elements = Cout * Cin * K * K
        self.reduction_size = N * Hout * Wout

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()

        self.x, self.grad_out = x._data, grad_out._data
        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.grad_w_ptr = self.backend._alloc(Cout * Cin * K * K * 4)
        self.fn_current = getattr(self.lib, "cf_conv2d_backward_weight_f32")
        self.fn_perthread = getattr(self.lib, "cf_conv2d_backward_weight_perthread_f32")
        self.fn_blockreduce = getattr(self.lib, "cf_conv2d_backward_weight_blockreduce_f32")
        self.fn_warpreduce = getattr(self.lib, "cf_conv2d_backward_weight_warpreduce_f32")

    def current(self, stream_handle) -> int:
        return self.fn_current(self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args, stream_handle)

    def perthread(self, stream_handle) -> int:
        return self.fn_perthread(self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args, stream_handle)

    def blockreduce(self, threads_per_block: int, stream_handle) -> int:
        return self.fn_blockreduce(
            self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args,
            ctypes.c_int(threads_per_block), stream_handle,
        )

    def warpreduce(self, warps_per_block: int, stream_handle) -> int:
        return self.fn_warpreduce(
            self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args,
            ctypes.c_int(warps_per_block), stream_handle,
        )


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    # Synchronize after *every* launch, warmup and timed alike -- unlike
    # `conv2d_backward_profile.py`'s `_time_phase` (which only needs this for
    # warmup, since every kernel it measures is fast), this module also
    # forces the block-reduce dWeight kernel at weight-element counts far
    # outside its production range (thousands of blocks at large shapes),
    # which measures far slower than either production kernel at that scale
    # (see the M33 report's **Cooperative Strategy Evaluated** section) --
    # queuing 30 such launches back-to-back on the null stream before a
    # single sync reliably exceeded Windows WDDM's ~2s TDR watchdog
    # (observed directly while writing this profiler). Per-iteration
    # synchronization does not affect measurement accuracy here: each
    # sample's elapsed time still comes from GPU-side `cudaEventElapsedTime`
    # between that iteration's own start/end events, not from host wall time.
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
    raw = _RawDWeight(cfg)

    current = _time_phase(raw.current)
    perthread = _time_phase(raw.perthread)
    blockreduce = {bs: _time_phase(lambda sh, bs=bs: raw.blockreduce(bs, sh)) for bs in BLOCK_SIZES}
    warpreduce = {wp: _time_phase(lambda sh, wp=wp: raw.warpreduce(wp, sh)) for wp in WARPS_PER_BLOCK}

    current_path = "blockreduce(256)" if raw.weight_elements < CONV2D_WEIGHT_REDUCE_THRESHOLD else "perthread"
    best_blockreduce_bs = min(blockreduce, key=lambda bs: blockreduce[bs]["mean_ms"])
    best_warpreduce_wp = min(warpreduce, key=lambda wp: warpreduce[wp]["mean_ms"])
    best_cooperative_ms = min(blockreduce[best_blockreduce_bs]["mean_ms"], warpreduce[best_warpreduce_wp]["mean_ms"])

    return {
        "name": name,
        "config": cfg,
        "weight_elements": raw.weight_elements,
        "reduction_elements_per_weight": raw.reduction_size,
        "current_dispatch_path": current_path,
        "current_ms": current,
        "perthread_ms": perthread,
        "blockreduce_ms": {str(bs): v for bs, v in blockreduce.items()},
        "warpreduce_ms": {str(wp): v for wp, v in warpreduce.items()},
        "best_blockreduce_block_size": best_blockreduce_bs,
        "best_blockreduce_ms": blockreduce[best_blockreduce_bs]["mean_ms"],
        "best_warpreduce_warps_per_block": best_warpreduce_wp,
        "best_warpreduce_ms": warpreduce[best_warpreduce_wp]["mean_ms"],
        "perthread_vs_best_cooperative_speedup": perthread["mean_ms"] / best_cooperative_ms,
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M33 dWeight cooperative-reduction profile (940MX, real CUDA) ===", ""]
    header = (
        f"{'shape':<14}{'weight#':>8}{'reduce#':>9}{'path':>16}"
        f"{'current(ms)':>13}{'perthr(ms)':>12}"
        f"{'bred64':>9}{'bred128':>9}{'bred256':>9}"
        f"{'wred64':>9}{'wred128':>9}{'wred256':>9}{'speedup':>9}"
    )
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        br = r["blockreduce_ms"]
        wr = r["warpreduce_ms"]
        lines.append(
            f"{r['name']:<14}{r['weight_elements']:>8}{r['reduction_elements_per_weight']:>9}"
            f"{r['current_dispatch_path']:>16}{r['current_ms']['mean_ms']:>13.4f}"
            f"{r['perthread_ms']['mean_ms']:>12.4f}"
            f"{br['64']['mean_ms']:>9.4f}{br['128']['mean_ms']:>9.4f}{br['256']['mean_ms']:>9.4f}"
            f"{wr['2']['mean_ms']:>9.4f}{wr['4']['mean_ms']:>9.4f}{wr['8']['mean_ms']:>9.4f}"
            f"{r['perthread_vs_best_cooperative_speedup']:>8.2f}x"
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
    parser.add_argument("--output", default="benchmarks/results/m33_conv2d_backward_weight_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- conv2d_backward_weight_profile requires real CUDA hardware.")
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
