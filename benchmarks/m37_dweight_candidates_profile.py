"""M37 Phase 4: benchmark the two fused-gather GEMM `dWeight` candidates
against the M34 im2col+GEMM baseline, at all 7 representative shapes.

Candidate A (`dweight_fused_gemm`) folds `k_im2col_conv2d` +
`k_conv2d_grad_output_permute` directly into the tiled-GEMM tile-load step,
eliminating both intermediate buffers/launches (measured 54-63% of M34's
total dWeight time, see `benchmarks/m37_dweight_profile.py`). Candidate C
(`dweight_fused_gemm_splitk`) additionally splits the GEMM's reduction
dimension across more blocks to address the measured occupancy shortfall
(as low as 5 of 24 device-resident block slots at several representative
shapes). Both are profiling-only (`forge.backend.cuda.experimental_conv_fused`)
-- neither is wired into `CUDABackend.conv2d_backward`.

For Candidate C this script sweeps `num_k_splits` over a fixed grid at every
shape (never trusting `recommended_num_k_splits` blindly) so the eventual
production decision is based on swept measurements, not a formula.

    python -m benchmarks.m37_dweight_candidates_profile
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
from .conv2d_backward_weight_im2col_profile import _RawIm2colGemm
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30
SPLIT_GRID = (1, 2, 4, 8, 16)


class _RawSplitkOnBuffers:
    """Candidate E: M34's own `Xcol`/`dYcolT` buffers fed to `cf_matmul_splitk_*`.

    `splitk()` rebuilds `Xcol`/`dYcolT` on *every* call (`im2col` + `permute`
    + split-K GEMM, three launches) -- matching `_RawIm2colGemm.total`'s own
    "complete pipeline" methodology exactly, so `candidate_e_best_ms` is
    directly comparable to `m34_total_pipeline_ms` (both are full-pipeline
    times, not GEMM-only times). An earlier version of this class reused a
    single pre-built `Xcol`/`dYcolT` across the timed loop and therefore
    measured GEMM-only time -- caught by comparing against the real
    production `CUDABackend.conv2d_backward` end-to-end time, which did not
    show the improvement this (bugged) isolated number implied.
    """

    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        self.M = N * Hout * Wout
        self.Kdim = Cin * K * K
        self.Cout = Cout

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()

        self.x, self.grad_out = x._data, grad_out._data
        suffix = _SUFFIX[np.dtype(np.float32)]
        self.fn_im2col = getattr(self.lib, f"cf_im2col_conv2d_{suffix}")
        self.fn_permute = getattr(self.lib, f"cf_conv2d_grad_output_permute_{suffix}")
        self.fn_splitk = getattr(self.lib, f"cf_matmul_splitk_{suffix}")

        self.xcol_ptr = self.backend._alloc(self.M * self.Kdim * 4)
        self.dycolT_ptr = self.backend._alloc(Cout * self.M * 4)
        self.out_ptr = self.backend._alloc(Cout * self.Kdim * 4)

        self.im2col_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(K), ctypes.c_int(K), ctypes.c_int(S), ctypes.c_int(S),
            ctypes.c_int(P), ctypes.c_int(P), ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.permute_args = (ctypes.c_int(N), ctypes.c_int(Cout), ctypes.c_int(Hout), ctypes.c_int(Wout))

        # Prime once outside timing, exactly like `_RawIm2colGemm`, purely so
        # a GEMM-only isolated number can also be reported for diagnosis.
        code = self.fn_im2col(self.x.ptr, self.xcol_ptr, *self.im2col_args, None)
        assert code == 0
        code = self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, None)
        assert code == 0
        forge.cuda.synchronize()

    def splitk_gemm_only(self, num_k_splits: int):
        """GEMM-only, reusing the primed `Xcol`/`dYcolT` -- a diagnostic number, not the headline."""
        def call(stream_handle) -> int:
            return self.fn_splitk(
                self.dycolT_ptr, self.xcol_ptr, self.out_ptr,
                ctypes.c_int(self.Cout), ctypes.c_int(self.M), ctypes.c_int(self.Kdim),
                ctypes.c_int(num_k_splits), stream_handle,
            )
        return call

    def splitk(self, num_k_splits: int):
        """Full pipeline: im2col + permute + split-K GEMM, all three rebuilt every call."""
        def call(stream_handle) -> int:
            code = self.fn_im2col(self.x.ptr, self.xcol_ptr, *self.im2col_args, stream_handle)
            if code != 0:
                return code
            code = self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, stream_handle)
            if code != 0:
                return code
            return self.fn_splitk(
                self.dycolT_ptr, self.xcol_ptr, self.out_ptr,
                ctypes.c_int(self.Cout), ctypes.c_int(self.M), ctypes.c_int(self.Kdim),
                ctypes.c_int(num_k_splits), stream_handle,
            )
        return call


class _RawFusedCandidates:
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
        self.fn_fused = getattr(self.lib, f"cf_dweight_fused_gemm_{suffix}")
        self.fn_splitk = getattr(self.lib, f"cf_dweight_fused_gemm_splitk_{suffix}")

        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.out_ptr = self.backend._alloc(Cout * Cin * K * K * 4)

    def fused(self, stream_handle) -> int:
        return self.fn_fused(self.grad_out.ptr, self.x.ptr, self.out_ptr, *self.shape_args, stream_handle)

    def splitk(self, num_k_splits: int):
        def call(stream_handle) -> int:
            return self.fn_splitk(
                self.grad_out.ptr, self.x.ptr, self.out_ptr, *self.shape_args,
                ctypes.c_int(num_k_splits), stream_handle,
            )
        return call


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
    baseline_raw = _RawIm2colGemm(cfg)
    fused_raw = _RawFusedCandidates(cfg)
    splitk_buf_raw = _RawSplitkOnBuffers(cfg)

    m34_current = _time_phase(baseline_raw.current)
    m34_total = _time_phase(baseline_raw.total)
    candidate_a = _time_phase(fused_raw.fused)

    def _sweep(raw_splitk_call_factory, total_m: int) -> "dict[int, dict]":
        results = {}
        total_tiles = (total_m + 15) // 16
        for splits in SPLIT_GRID:
            if splits > total_tiles:
                continue
            results[splits] = _time_phase(raw_splitk_call_factory(splits))
        return results

    splitk_fused_results = _sweep(fused_raw.splitk, fused_raw.M)
    best_split_c = min(splitk_fused_results, key=lambda s: splitk_fused_results[s]["mean_ms"])
    candidate_c_best = splitk_fused_results[best_split_c]

    splitk_buf_results = _sweep(splitk_buf_raw.splitk, splitk_buf_raw.M)
    best_split_e = min(splitk_buf_results, key=lambda s: splitk_buf_results[s]["mean_ms"])
    candidate_e_best = splitk_buf_results[best_split_e]

    return {
        "name": name,
        "config": cfg,
        "weight_elements": fused_raw.weight_elements,
        "M": fused_raw.M,
        "K": fused_raw.Kdim,
        "Cout": fused_raw.Cout,
        "m34_current_kernel_ms": m34_current,
        "m34_total_pipeline_ms": m34_total,
        "candidate_a_fused_ms": candidate_a,
        "candidate_a_vs_m34_speedup": m34_total["mean_ms"] / candidate_a["mean_ms"],
        "candidate_c_splitk_sweep_ms": {str(k): v for k, v in splitk_fused_results.items()},
        "candidate_c_best_split": best_split_c,
        "candidate_c_best_ms": candidate_c_best,
        "candidate_c_vs_m34_speedup": m34_total["mean_ms"] / candidate_c_best["mean_ms"],
        "candidate_c_vs_candidate_a_speedup": candidate_a["mean_ms"] / candidate_c_best["mean_ms"],
        "candidate_e_splitk_sweep_ms": {str(k): v for k, v in splitk_buf_results.items()},
        "candidate_e_best_split": best_split_e,
        "candidate_e_best_ms": candidate_e_best,
        "candidate_e_vs_m34_speedup": m34_total["mean_ms"] / candidate_e_best["mean_ms"],
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M37 dWeight candidates A/C/E vs M34 baseline (940MX, real CUDA) ===", ""]
    header = (
        f"{'shape':<14}{'weight#':>8}{'M34(ms)':>10}{'A(ms)':>9}{'A_spd':>7}"
        f"{'C_split':>8}{'C(ms)':>9}{'C_spd':>7}{'E_split':>8}{'E(ms)':>9}{'E_spd':>7}"
    )
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        lines.append(
            f"{r['name']:<14}{r['weight_elements']:>8}{r['m34_total_pipeline_ms']['mean_ms']:>10.4f}"
            f"{r['candidate_a_fused_ms']['mean_ms']:>9.4f}{r['candidate_a_vs_m34_speedup']:>6.2f}x"
            f"{r['candidate_c_best_split']:>8}{r['candidate_c_best_ms']['mean_ms']:>9.4f}{r['candidate_c_vs_m34_speedup']:>6.2f}x"
            f"{r['candidate_e_best_split']:>8}{r['candidate_e_best_ms']['mean_ms']:>9.4f}{r['candidate_e_vs_m34_speedup']:>6.2f}x"
        )
    lines.append("")
    lines.append("speedup = m34_total_pipeline / candidate -- >1.0 means the candidate is faster.")
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
    parser.add_argument("--output", default="benchmarks/results/m37_dweight_candidates_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m37_dweight_candidates_profile requires real CUDA hardware.")
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
