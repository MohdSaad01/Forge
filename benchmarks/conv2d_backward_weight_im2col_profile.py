"""Dedicated CUDA `dWeight` im2col + existing tiled GEMM profiler (Milestone 34).

M33 rejected cooperative reduction and named im2col + GEMM (reusing the
existing M11 shared-memory-tiled `k_matmul`) as the next structurally
different `dWeight` candidate worth measuring. This script isolates and
times every phase of that candidate -- `im2col` construction, the
`grad_output` permute (`dYcolT`), the GEMM call itself, and the complete
pipeline -- at the same 7 M32/M33 representative shapes
(`benchmarks/conv2d_backward_profile.py`'s `SHAPES`/`BATCH_SWEEP_BASE`/
`BATCH_SIZES`, imported directly rather than duplicated), and compares
against the existing production `dWeight` kernel in isolation.

Per the milestone brief's Section 8 ("do not compare the current kernel
against only GEMM time"), the headline comparison is always **current
dWeight vs. total experimental time** (`im2col + permute + gemm`), never
GEMM time alone. Section 12 ("compare three approaches") is covered by:

  * `current_ms`     -- the existing `cf_conv2d_backward_weight_*` dispatcher
  * `im2col_ms`       -- `cf_im2col_conv2d_*` alone (materialization cost)
  * `permute_ms`      -- `cf_conv2d_grad_output_permute_*` alone
  * `gemm_ms`          -- `cf_matmul_*` alone, on already-built `Xcol`/`dYcolT`
  * `total_experimental_ms` -- im2col + permute + gemm, summed

    python -m benchmarks.conv2d_backward_weight_im2col_profile
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

from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _check_fits_in_vram, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30


class _RawIm2colGemm:
    """Directly-callable, per-phase-isolated im2col+GEMM `dWeight` pipeline (profiling-only)."""

    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        self.N, self.Cin, self.Cout, self.H, self.W, self.K = N, Cin, Cout, H, W, K
        self.S, self.P, self.Hout, self.Wout = S, P, Hout, Wout
        self.M = N * Hout * Wout
        self.Kdim = Cin * K * K
        self.weight_elements = Cout * Cin * K * K

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()

        self.x, self.grad_out = x._data, grad_out._data
        suffix = _SUFFIX[np.dtype(np.float32)]
        self.fn_current = getattr(self.lib, f"cf_conv2d_backward_weight_{suffix}")
        self.fn_im2col = getattr(self.lib, f"cf_im2col_conv2d_{suffix}")
        self.fn_permute = getattr(self.lib, f"cf_conv2d_grad_output_permute_{suffix}")
        self.fn_matmul = getattr(self.lib, f"cf_matmul_{suffix}")

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

        self.grad_w_ptr = self.backend._alloc(Cout * Cin * K * K * 4)
        self.xcol_ptr = self.backend._alloc(self.M * self.Kdim * 4)
        self.xcol_nbytes = self.M * self.Kdim * 4
        self.dycolT_ptr = self.backend._alloc(Cout * self.M * 4)
        self.dycolT_nbytes = Cout * self.M * 4

        # Pre-populate Xcol/dYcolT once (outside timing) so the isolated
        # GEMM-only phase measures GEMM time against real, already-correct
        # operand data -- not garbage from a freshly `cudaMalloc`'d buffer.
        code = self.fn_im2col(self.x.ptr, self.xcol_ptr, *self.im2col_args, None)
        assert code == 0, f"im2col priming launch failed with code {code}"
        code = self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, None)
        assert code == 0, f"permute priming launch failed with code {code}"
        forge.cuda.synchronize()

    def current(self, stream_handle) -> int:
        return self.fn_current(self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args, stream_handle)

    def im2col(self, stream_handle) -> int:
        return self.fn_im2col(self.x.ptr, self.xcol_ptr, *self.im2col_args, stream_handle)

    def permute(self, stream_handle) -> int:
        return self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, stream_handle)

    def gemm(self, stream_handle) -> int:
        return self.fn_matmul(
            self.dycolT_ptr, self.xcol_ptr, self.grad_w_ptr,
            ctypes.c_int(self.Cout), ctypes.c_int(self.M), ctypes.c_int(self.Kdim),
            stream_handle,
        )

    def total(self, stream_handle) -> int:
        """The complete, un-synchronized-between-stages pipeline (Section 9/21: stream order only)."""
        code = self.fn_im2col(self.x.ptr, self.xcol_ptr, *self.im2col_args, stream_handle)
        if code != 0:
            return code
        code = self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, stream_handle)
        if code != 0:
            return code
        return self.fn_matmul(
            self.dycolT_ptr, self.xcol_ptr, self.grad_w_ptr,
            ctypes.c_int(self.Cout), ctypes.c_int(self.M), ctypes.c_int(self.Kdim),
            stream_handle,
        )


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


def _profile_shape(name: str, cfg: "dict[str, int]") -> dict:
    _check_fits_in_vram(cfg)
    raw = _RawIm2colGemm(cfg)

    current = _time_phase(raw.current)
    im2col_t = _time_phase(raw.im2col)
    permute_t = _time_phase(raw.permute)
    gemm_t = _time_phase(raw.gemm)
    total_t = _time_phase(raw.total)

    total_experimental_ms = im2col_t["mean_ms"] + permute_t["mean_ms"] + gemm_t["mean_ms"]
    xcol_mb = raw.xcol_nbytes / (1024 * 1024)
    dycolT_mb = raw.dycolT_nbytes / (1024 * 1024)

    return {
        "name": name,
        "config": cfg,
        "weight_elements": raw.weight_elements,
        "M": raw.M,  # GEMM reduction dim: N*Hout*Wout
        "K": raw.Kdim,  # GEMM operand dim: Cin*KH*KW
        "Cout": raw.Cout,
        "gemm_dims": {"A_rows_M": raw.Cout, "A_cols_K": raw.M, "B_cols_N": raw.Kdim},
        "xcol_shape": [raw.M, raw.Kdim],
        "xcol_mb": xcol_mb,
        "dycolT_shape": [raw.Cout, raw.M],
        "dycolT_mb": dycolT_mb,
        "current_ms": current,
        "im2col_ms": im2col_t,
        "permute_ms": permute_t,
        "gemm_ms": gemm_t,
        "measured_total_ms": total_t,  # whole pipeline, one synchronize at the end
        "summed_total_experimental_ms": total_experimental_ms,  # sum of isolated phases
        "current_vs_experimental_speedup": total_experimental_ms / current["mean_ms"],
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M34 dWeight im2col+GEMM profile (940MX, real CUDA) ===", ""]
    header = (
        f"{'shape':<14}{'weight#':>8}{'M(=N*Ho*Wo)':>12}{'K':>6}"
        f"{'current(ms)':>13}{'im2col(ms)':>11}{'permute(ms)':>12}{'gemm(ms)':>10}"
        f"{'total(ms)':>11}{'xcol(MB)':>10}{'ratio':>8}"
    )
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        lines.append(
            f"{r['name']:<14}{r['weight_elements']:>8}{r['M']:>12}{r['K']:>6}"
            f"{r['current_ms']['mean_ms']:>13.4f}{r['im2col_ms']['mean_ms']:>11.4f}"
            f"{r['permute_ms']['mean_ms']:>12.4f}{r['gemm_ms']['mean_ms']:>10.4f}"
            f"{r['summed_total_experimental_ms']:>11.4f}{r['xcol_mb']:>10.2f}"
            f"{r['current_vs_experimental_speedup']:>7.2f}x"
        )
    lines.append("")
    lines.append("ratio = experimental_total / current -- <1.0 means experimental is faster.")
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
    parser.add_argument("--output", default="benchmarks/results/m34_dweight_im2col_gemm.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- conv2d_backward_weight_im2col_profile requires real CUDA hardware.")
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
