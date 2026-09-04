"""Dedicated CUDA Conv2d-backward profiler (Milestone 32).

`benchmarks/mnist_profile.py` (Milestone 21) times `conv2d`'s *combined*
backward op (`node.backward_fn` for the whole `conv2d` node) as one number.
That is not fine-grained enough to answer M32's question: `CUDABackend.
conv2d_backward` (`forge/backend/cuda/backend.py`) already issues three
*separate* kernel launches per call --
`cf_conv2d_backward_{input,weight,bias}_{f32,f64}` -- each already followed
by its own `_maybe_synchronize()`. This script isolates and times each of
the three directly, at a set of representative shapes (not just the two
fixed M20 MNIST layer shapes M21 measured), using the same timing-enabled
`TimedEvent` (`forge.backend.cuda.profiling_events`) convention
`benchmarks/pipeline_profile.py` established -- CUDA events for GPU-side
timing, not `time.perf_counter()` bracketing, per Section 37 of the M32
brief.

No production code is touched or reimplemented to get this isolation: the
three backward kernels are already independently launchable through
`CUDABackend`'s own compiled `ctypes` functions (`self._lib.
cf_conv2d_backward_{input,weight,bias}_{f32,f64}`, with `argtypes` already
configured by `_configure_signatures`) -- this script calls them directly,
mirroring exactly the argument construction `CUDABackend.conv2d_backward`
itself uses, the same way `mnist_profile.py`'s `_profiled_run_backward`
mirrors (rather than modifies) `forge.autograd.engine.run_backward`.

    python -m benchmarks.conv2d_backward_profile
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

from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30

# -- representative shapes (Section 5/36 of the M32 brief) -------------------
#
# Every shape uses stride=1, padding=1 (`Hout=H`, `Wout=W`) except where
# noted -- this isolates channel-count/spatial-size/batch-size effects from
# each other one variable at a time, rather than changing stride/padding too.
SHAPES: "dict[str, dict[str, int]]" = {
    # MNIST conv1: the real M20 first-layer shape (Cout*Cin*K*K = 72 weight
    # elements -- below CONV2D_WEIGHT_REDUCE_THRESHOLD, block-reduce kernel).
    "mnist_conv1": {"N": 64, "Cin": 1, "Cout": 8, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
    # MNIST conv2 (post-pool, 13x13): the real M20 second-layer shape
    # (Cout*Cin*K*K = 1,152 -- at/above threshold, one-thread-per-weight
    # kernel, M21's own "regression" shape).
    "mnist_conv2": {"N": 64, "Cin": 8, "Cout": 16, "H": 13, "W": 13, "K": 3, "S": 1, "P": 1},
    # Larger channel workload: a plausible deeper-CNN layer, same spatial
    # size as conv1 (Cout*Cin*K*K = 4,608 weight elements -- well above
    # threshold, and each thread's serial reduction is N*Hout*Wout=50,176 --
    # 6.5x conv2's own 7,744, at 4x the element/thread count).
    "large_channel": {"N": 64, "Cin": 16, "Cout": 32, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
    # Larger spatial workload: conv2's channel shape at 4x the linear
    # resolution (56x56 instead of 13x13) -- fits comfortably in the 940MX's
    # 2GB VRAM budget (see this module's `_check_fits_in_vram`).
    "large_spatial": {"N": 32, "Cin": 8, "Cout": 16, "H": 56, "W": 56, "K": 3, "S": 1, "P": 1},
}

# Batch-size sweep at a fixed "large_channel"-shaped layer -- Section 5's
# "batch-size dependent?" question, isolated from channel/spatial variation.
BATCH_SWEEP_BASE = {"Cin": 16, "Cout": 32, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1}
BATCH_SIZES = (32, 64, 128)


def _hout_wout(H: int, K: int, S: int, P: int) -> int:
    return (H + 2 * P - K) // S + 1


def _check_fits_in_vram(cfg: "dict[str, int]") -> None:
    """~940MX 2GB sanity check -- all live buffers for one backward call, float32."""
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    Hout = _hout_wout(H, K, cfg["S"], cfg["P"])
    Wout = _hout_wout(W, K, cfg["S"], cfg["P"])
    total_elems = (
        N * Cin * H * W * 2  # x + grad_x
        + N * Cout * Hout * Wout * 2  # out + grad_out
        + Cout * Cin * K * K * 2  # weight + grad_weight
        + Cout * 2  # bias + grad_bias
    )
    nbytes = total_elems * 4
    assert nbytes < 256 * 1024 * 1024, f"shape {cfg} would use ~{nbytes / 1e6:.1f}MB -- too large for this sweep"


class _RawConv2dBackward:
    """Directly-callable, isolated CUDA Conv2d backward kernels (profiling-only).

    Mirrors `CUDABackend.conv2d_backward`'s own argument construction
    exactly (same `shape_args` tuple, same suffix dispatch) but issues the
    three kernel launches as three independently-timeable calls instead of
    one Python-level call that returns all three gradients together.
    """

    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        self.N, self.Cin, self.Cout, self.H, self.W, self.K = N, Cin, Cout, H, W, K
        self.Hout, self.Wout = Hout, Wout

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        w = forge.Tensor(rng.standard_normal((Cout, Cin, K, K)).astype(np.float32), device="cuda")
        b = forge.Tensor(rng.standard_normal((Cout,)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")

        # Force materialization on the default (synchronous) stream so every
        # timed launch below starts from a clean, already-complete state --
        # matches `pipeline_profile.py`'s own warmup-then-synchronize recipe.
        forge.cuda.synchronize()

        self.x, self.w, self.b, self.grad_out = x._data, w._data, b._data, grad_out._data
        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.fn_fwd = getattr(self.lib, "cf_conv2d_forward_f32")
        self.fn_gx = getattr(self.lib, "cf_conv2d_backward_input_f32")
        self.fn_gw = getattr(self.lib, "cf_conv2d_backward_weight_f32")
        self.fn_gb = getattr(self.lib, "cf_conv2d_backward_bias_f32")

        self.grad_x_ptr = self.backend._alloc(N * Cin * H * W * 4)
        self.grad_w_ptr = self.backend._alloc(Cout * Cin * K * K * 4)
        self.grad_b_ptr = self.backend._alloc(Cout * 4)
        self.out_ptr = self.backend._alloc(N * Cout * Hout * Wout * 4)

    def forward(self, stream_handle) -> int:
        return self.fn_fwd(
            self.x.ptr, self.w.ptr, self.b.ptr, self.out_ptr,
            *self.shape_args, ctypes.c_int(1), stream_handle,
        )

    def backward_input(self, stream_handle) -> int:
        return self.fn_gx(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, stream_handle)

    def backward_weight(self, stream_handle) -> int:
        return self.fn_gw(self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args, stream_handle)

    def backward_bias(self, stream_handle) -> int:
        return self.fn_gb(
            self.grad_out.ptr, self.grad_b_ptr,
            ctypes.c_int(self.N), ctypes.c_int(self.Cout), ctypes.c_int(self.Hout), ctypes.c_int(self.Wout),
            stream_handle,
        )


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    stream_handle = None  # default stream: every launch already fully host-synchronous
    # Synchronize after *every* warmup launch, not once at the end: several
    # of these kernels (the naive one-thread-per-weight `dWeight` path, at
    # this milestone's larger shapes) are slow enough that queuing `warmup`
    # of them back-to-back on the null stream before a single sync risks
    # tripping Windows WDDM's ~2s TDR watchdog (this was observed directly
    # while writing this profiler -- see the M32 report's "Kernel Launch
    # Analysis"/"Baseline Timings" sections).
    for _ in range(warmup):
        code = call(stream_handle)
        assert code == 0, f"kernel launch failed with code {code}"
        forge.cuda.synchronize()

    pairs = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(stream_handle)
        code = call(stream_handle)
        assert code == 0, f"kernel launch failed with code {code}"
        end = TimedEvent()
        end.record(stream_handle)
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


def _profile_shape(name: str, cfg: "dict[str, int]") -> dict:
    _check_fits_in_vram(cfg)
    raw = _RawConv2dBackward(cfg)

    forward = _time_phase(raw.forward)
    d_input = _time_phase(raw.backward_input)
    d_weight = _time_phase(raw.backward_weight)
    d_bias = _time_phase(raw.backward_bias)

    backward_total_ms = d_input["mean_ms"] + d_weight["mean_ms"] + d_bias["mean_ms"]
    weight_elems = cfg["Cout"] * cfg["Cin"] * cfg["K"] * cfg["K"]
    reduce_size = cfg["N"] * raw.Hout * raw.Wout

    return {
        "name": name,
        "config": cfg,
        "Hout": raw.Hout,
        "Wout": raw.Wout,
        "weight_elements": weight_elems,
        "dweight_reduction_size": reduce_size,
        "forward_ms": forward,
        "d_input_ms": d_input,
        "d_weight_ms": d_weight,
        "d_bias_ms": d_bias,
        "backward_total_ms": backward_total_ms,
        "d_input_pct_of_backward": 100.0 * d_input["mean_ms"] / backward_total_ms,
        "d_weight_pct_of_backward": 100.0 * d_weight["mean_ms"] / backward_total_ms,
        "d_bias_pct_of_backward": 100.0 * d_bias["mean_ms"] / backward_total_ms,
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M32 Conv2d backward profile (940MX, real CUDA) ===", ""]
    lines.append(f"{'shape':<16}{'weight#':>9}{'reduce#':>10}{'fwd(ms)':>10}{'dIn(ms)':>10}{'dW(ms)':>10}{'dB(ms)':>10}{'bwd(ms)':>10}   ranking")
    for r in profile["shapes"]:
        ranked = sorted(
            [("dInput", r["d_input_ms"]["mean_ms"]), ("dWeight", r["d_weight_ms"]["mean_ms"]), ("dBias", r["d_bias_ms"]["mean_ms"])],
            key=lambda kv: -kv[1],
        )
        rank_str = " > ".join(f"{k}({v:.2f}ms)" for k, v in ranked)
        lines.append(
            f"{r['name']:<16}{r['weight_elements']:>9}{r['dweight_reduction_size']:>10}"
            f"{r['forward_ms']['mean_ms']:>10.4f}{r['d_input_ms']['mean_ms']:>10.4f}"
            f"{r['d_weight_ms']['mean_ms']:>10.4f}{r['d_bias_ms']['mean_ms']:>10.4f}"
            f"{r['backward_total_ms']:>10.4f}   {rank_str}"
        )
    lines.append("")
    lines.append(f"{'batch':<16}{'weight#':>9}{'reduce#':>10}{'fwd(ms)':>10}{'dIn(ms)':>10}{'dW(ms)':>10}{'dB(ms)':>10}{'bwd(ms)':>10}")
    for r in profile["batch_sweep"]:
        lines.append(
            f"N={r['config']['N']:<14}{r['weight_elements']:>9}{r['dweight_reduction_size']:>10}"
            f"{r['forward_ms']['mean_ms']:>10.4f}{r['d_input_ms']['mean_ms']:>10.4f}"
            f"{r['d_weight_ms']['mean_ms']:>10.4f}{r['d_bias_ms']['mean_ms']:>10.4f}"
            f"{r['backward_total_ms']:>10.4f}"
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
    parser.add_argument("--output", default="benchmarks/results/conv2d_backward_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- conv2d_backward_profile requires real CUDA hardware.")
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
