"""Full `conv2d_backward` comparison: production vs. experimental-dWeight pipeline (Milestone 34).

`conv2d_backward_weight_im2col_profile.py` isolated `dWeight` alone.
Section 34 of the milestone brief asks for the *complete* `conv2d_backward`
comparison -- production `dInput` + production `dBias` held fixed, only
`dWeight` swapped for the experimental im2col+GEMM path -- since that is the
actually relevant end-to-end number (a fast `dWeight` alone does not matter
if `conv2d_backward` overall does not improve). This script also captures
Section 11/18/19's allocator/peak-memory accounting: `forge.cuda.
memory_stats()` across a repeated-iteration run of each variant with
temporaries explicitly released every iteration (so the caching allocator
sees its designed steady-state, same-size-repeat workload), reporting peak
reserved bytes and cache hit/miss counts -- so the extra `Xcol`/`dYcolT`
temporary allocations' cost is visible, not just asserted.

Timing (`_time_phase`) pre-allocates every output/temporary buffer once in
`_RawFullBackward.__init__` and reuses those same pointers across warmup and
measured iterations -- the same convention `conv2d_backward_profile.py` (M32)
and `conv2d_backward_weight_profile.py` (M33) already use, so kernel time is
isolated from allocator overhead. Memory accounting (`_measure_memory`) uses
a *separate*, fresh-allocate-then-release-every-iteration path instead, since
that is what actually exercises cache hit/miss behavior.

    python -m benchmarks.conv2d_backward_im2col_pipeline_profile
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda import allocator as _allocator
from forge.backend.cuda.backend import _SUFFIX, get_cuda_backend, is_cuda_available
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _check_fits_in_vram, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30
MEMORY_ITERATIONS = 20


class _RawFullBackward:
    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        self.N, self.Cin, self.Cout, self.H, self.W, self.K = N, Cin, Cout, H, W, K
        self.Hout, self.Wout = Hout, Wout
        self.M = N * Hout * Wout
        self.Kdim = Cin * K * K

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        w = forge.Tensor(rng.standard_normal((Cout, Cin, K, K)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        self.x, self.w, self.grad_out = x._data, w._data, grad_out._data

        suffix = _SUFFIX[np.dtype(np.float32)]
        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.im2col_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(K), ctypes.c_int(K), ctypes.c_int(S), ctypes.c_int(S),
            ctypes.c_int(P), ctypes.c_int(P), ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.permute_args = (ctypes.c_int(N), ctypes.c_int(Cout), ctypes.c_int(Hout), ctypes.c_int(Wout))

        self.fn_gx = getattr(self.lib, f"cf_conv2d_backward_input_{suffix}")
        self.fn_gw = getattr(self.lib, f"cf_conv2d_backward_weight_{suffix}")
        self.fn_gb = getattr(self.lib, f"cf_conv2d_backward_bias_{suffix}")
        self.fn_im2col = getattr(self.lib, f"cf_im2col_conv2d_{suffix}")
        self.fn_permute = getattr(self.lib, f"cf_conv2d_grad_output_permute_{suffix}")
        self.fn_matmul = getattr(self.lib, f"cf_matmul_{suffix}")

        # Pre-allocated once, reused every timed call (M32/M33 convention) --
        # isolates kernel time from allocator overhead.
        self.grad_x_ptr = self.backend._alloc(N * Cin * H * W * 4)
        self.grad_w_ptr = self.backend._alloc(Cout * Cin * K * K * 4)
        self.grad_b_ptr = self.backend._alloc(Cout * 4)
        self.xcol_ptr = self.backend._alloc(self.M * self.Kdim * 4)
        self.dycolT_ptr = self.backend._alloc(Cout * self.M * 4)

        self.output_sizes_production = [N * Cin * H * W * 4, Cout * Cin * K * K * 4, Cout * 4]
        self.output_sizes_experimental = self.output_sizes_production + [self.M * self.Kdim * 4, Cout * self.M * 4]

    # -- timing (fixed, pre-allocated buffers) --------------------------------

    def production(self, stream_handle) -> int:
        c1 = self.fn_gx(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, stream_handle)
        c2 = self.fn_gw(self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args, stream_handle)
        c3 = self.fn_gb(
            self.grad_out.ptr, self.grad_b_ptr,
            ctypes.c_int(self.N), ctypes.c_int(self.Cout), ctypes.c_int(self.Hout), ctypes.c_int(self.Wout),
            stream_handle,
        )
        return c1 or c2 or c3

    def experimental(self, stream_handle) -> int:
        c1 = self.fn_gx(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, stream_handle)
        c2 = self.fn_im2col(self.x.ptr, self.xcol_ptr, *self.im2col_args, stream_handle)
        c3 = self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, stream_handle)
        c4 = self.fn_matmul(
            self.dycolT_ptr, self.xcol_ptr, self.grad_w_ptr,
            ctypes.c_int(self.Cout), ctypes.c_int(self.M), ctypes.c_int(self.Kdim),
            stream_handle,
        )
        c5 = self.fn_gb(
            self.grad_out.ptr, self.grad_b_ptr,
            ctypes.c_int(self.N), ctypes.c_int(self.Cout), ctypes.c_int(self.Hout), ctypes.c_int(self.Wout),
            stream_handle,
        )
        return c1 or c2 or c3 or c4 or c5

    # -- fresh-allocate-then-release, for allocator/memory accounting --------

    def production_fresh(self, stream_handle) -> int:
        grad_x_ptr = self.backend._alloc(self.N * self.Cin * self.H * self.W * 4)
        grad_w_ptr = self.backend._alloc(self.Cout * self.Cin * self.K * self.K * 4)
        grad_b_ptr = self.backend._alloc(self.Cout * 4)
        c1 = self.fn_gx(self.grad_out.ptr, self.w.ptr, grad_x_ptr, *self.shape_args, stream_handle)
        c2 = self.fn_gw(self.grad_out.ptr, self.x.ptr, grad_w_ptr, *self.shape_args, stream_handle)
        c3 = self.fn_gb(
            self.grad_out.ptr, grad_b_ptr,
            ctypes.c_int(self.N), ctypes.c_int(self.Cout), ctypes.c_int(self.Hout), ctypes.c_int(self.Wout),
            stream_handle,
        )
        forge.cuda.synchronize()
        for ptr, nbytes in zip((grad_x_ptr, grad_w_ptr, grad_b_ptr), self.output_sizes_production):
            _allocator.release(nbytes, ptr)
        return c1 or c2 or c3

    def experimental_fresh(self, stream_handle) -> int:
        grad_x_ptr = self.backend._alloc(self.N * self.Cin * self.H * self.W * 4)
        grad_w_ptr = self.backend._alloc(self.Cout * self.Cin * self.K * self.K * 4)
        grad_b_ptr = self.backend._alloc(self.Cout * 4)
        xcol_ptr = self.backend._alloc(self.M * self.Kdim * 4)
        dycolT_ptr = self.backend._alloc(self.Cout * self.M * 4)
        c1 = self.fn_gx(self.grad_out.ptr, self.w.ptr, grad_x_ptr, *self.shape_args, stream_handle)
        c2 = self.fn_im2col(self.x.ptr, xcol_ptr, *self.im2col_args, stream_handle)
        c3 = self.fn_permute(self.grad_out.ptr, dycolT_ptr, *self.permute_args, stream_handle)
        c4 = self.fn_matmul(
            dycolT_ptr, xcol_ptr, grad_w_ptr,
            ctypes.c_int(self.Cout), ctypes.c_int(self.M), ctypes.c_int(self.Kdim),
            stream_handle,
        )
        c5 = self.fn_gb(
            self.grad_out.ptr, grad_b_ptr,
            ctypes.c_int(self.N), ctypes.c_int(self.Cout), ctypes.c_int(self.Hout), ctypes.c_int(self.Wout),
            stream_handle,
        )
        forge.cuda.synchronize()
        for ptr, nbytes in zip(
            (grad_x_ptr, grad_w_ptr, grad_b_ptr, xcol_ptr, dycolT_ptr), self.output_sizes_experimental
        ):
            _allocator.release(nbytes, ptr)
        return c1 or c2 or c3 or c4 or c5


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


def _measure_memory(call) -> dict:
    """Repeated fresh-alloc-then-release: peak bytes, and cache hit ratio in steady state."""
    stream_handle = None
    forge.cuda.empty_cache()
    forge.cuda.reset_peak_memory_stats()
    before = forge.cuda.memory_stats()

    for _ in range(MEMORY_ITERATIONS):
        code = call(stream_handle)
        assert code == 0

    after_steady = forge.cuda.memory_stats()
    forge.cuda.empty_cache()
    after_cache_cleared = forge.cuda.memory_stats()

    return {
        "peak_allocated_bytes": after_steady.peak_allocated_bytes,
        "peak_reserved_bytes": after_steady.peak_reserved_bytes,
        "cache_hit_count_delta": after_steady.cache_hit_count - before.cache_hit_count,
        "cache_miss_count_delta": after_steady.cache_miss_count - before.cache_miss_count,
        "reserved_bytes_after_empty_cache": after_cache_cleared.reserved_bytes,
        "allocated_bytes_after_empty_cache": after_cache_cleared.allocated_bytes,
    }


def _profile_shape(name: str, cfg: "dict[str, int]") -> dict:
    _check_fits_in_vram(cfg)
    raw = _RawFullBackward(cfg)

    production_time = _time_phase(raw.production)
    experimental_time = _time_phase(raw.experimental)

    production_mem = _measure_memory(raw.production_fresh)
    experimental_mem = _measure_memory(raw.experimental_fresh)

    return {
        "name": name,
        "config": cfg,
        "production_backward_ms": production_time,
        "experimental_backward_ms": experimental_time,
        "speedup": production_time["mean_ms"] / experimental_time["mean_ms"],
        "production_memory": production_mem,
        "experimental_memory": experimental_mem,
        "extra_peak_reserved_mb": (
            (experimental_mem["peak_reserved_bytes"] - production_mem["peak_reserved_bytes"]) / (1024 * 1024)
        ),
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M34 full conv2d_backward: production vs. experimental-dWeight pipeline ===", ""]
    header = (
        f"{'shape':<14}{'prod(ms)':>10}{'exp(ms)':>10}{'speedup':>9}"
        f"{'prod_peak(MB)':>15}{'exp_peak(MB)':>14}{'extra(MB)':>11}"
    )
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        pm = r["production_memory"]["peak_reserved_bytes"] / (1024 * 1024)
        em = r["experimental_memory"]["peak_reserved_bytes"] / (1024 * 1024)
        lines.append(
            f"{r['name']:<14}{r['production_backward_ms']['mean_ms']:>10.4f}"
            f"{r['experimental_backward_ms']['mean_ms']:>10.4f}{r['speedup']:>8.2f}x"
            f"{pm:>15.2f}{em:>14.2f}{r['extra_peak_reserved_mb']:>11.2f}"
        )
    lines.append("")
    lines.append("speedup = production_ms / experimental_ms -- >1.0 means experimental is faster.")
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
    parser.add_argument("--output", default="benchmarks/results/m34_pipeline_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- conv2d_backward_im2col_pipeline_profile requires real CUDA hardware.")
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
