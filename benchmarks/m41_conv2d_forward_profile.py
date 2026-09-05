"""Milestone 41: CUDA Conv2d forward -- PROFILE + BENCHMARK.

M40 found `k_conv2d_forward` (unchanged since Milestone 15) achieving only
10.7-18.0% of Forge's practical compute ceiling -- the least efficient
Conv2d-adjacent kernel measured, despite an identical FLOP count to
`dInput`/`dWeight` (which reach 22-44%). This script:

1. **PROFILE**: times the current production `cf_conv2d_forward_*` in
   isolation at the 7 existing representative shapes
   (`conv2d_backward_profile.SHAPES` + `BATCH_SWEEP_BASE`/`BATCH_SIZES`) plus
   a dedicated K/stride/channel sweep, computing FLOPs/bytes/arithmetic
   intensity/roofline classification for each via `benchmarks.roofline`
   (same conventions M35/M40 already established).
2. **BENCHMARK**: interleaved A/B/C comparison of the current production
   kernel against Milestone 41's two candidates (`experimental_conv_forward_
   im2col.conv2d_forward_im2col_gemm` -- Candidate A; `experimental_conv_
   forward_halffused.conv2d_forward_halffused_gemm` -- Candidate B),
   measuring the *complete* forward pipeline for each (not just an internal
   GEMM stage -- the exact benchmarking mistake M37's own report flags),
   plus a stage-by-stage decomposition of Candidate A (im2col / weight
   transpose / GEMM / output permute).

No production code path is changed by running this script -- `CUDABackend.
conv2d` is untouched; both candidates are reached only through their own
experimental modules, exactly like every prior "profiling-only candidate"
milestone (M37/M38/M39's own scripts).

    python -m benchmarks.m41_conv2d_forward_profile
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import _SUFFIX, get_cuda_backend, is_cuda_available
from forge.backend.cuda.experimental_conv_forward_halffused import conv2d_forward_halffused_gemm
from forge.backend.cuda.experimental_conv_forward_im2col import (
    conv2d_forward_im2col_gemm,
    output_permute,
    transpose_weight,
)
from forge.backend.cuda.experimental_conv_im2col import im2col
from forge.backend.cuda.experimental_conv_im2col_reuse import im2col_smem, im2col_smem_fits
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from . import m35_hardware, roofline
from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _check_fits_in_vram, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30

# -- Section 1: K/stride/channel sweep (Section 1 of the M41 brief) ---------
# stride=1/padding=1 unless noted; each entry varies exactly one axis from a
# common Cin=8/Cout=16/28x28/N=16 base, mirroring `conv2d_backward_profile.
# SHAPES`'s own "one variable at a time" convention.
_SWEEP_BASE = {"N": 16, "Cin": 8, "Cout": 16, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1}

SWEEP_SHAPES: "dict[str, dict[str, int]]" = {
    "k1_s1": {**_SWEEP_BASE, "K": 1, "P": 0},
    "k3_s1": {**_SWEEP_BASE, "K": 3, "P": 1},
    "k5_s1": {**_SWEEP_BASE, "K": 5, "P": 2},
    "k3_s2": {**_SWEEP_BASE, "K": 3, "P": 1, "S": 2},
    "cin_low": {**_SWEEP_BASE, "Cin": 1},
    "cin_high": {**_SWEEP_BASE, "Cin": 64},
    "cout_low": {**_SWEEP_BASE, "Cout": 4},
    "cout_high": {**_SWEEP_BASE, "Cout": 128},
}

ALL_SHAPES: "dict[str, dict[str, int]]" = {**SHAPES, **SWEEP_SHAPES}
for _bs in BATCH_SIZES:
    ALL_SHAPES[f"batch_{_bs}"] = {**BATCH_SWEEP_BASE, "N": _bs}


def _time_calls(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    """CUDA-event timing of a zero-arg `call() -> int errcode`, default stream."""
    for _ in range(warmup):
        code = call()
        assert code == 0, f"kernel launch failed with code {code}"
        forge.cuda.synchronize()

    pairs = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(None)
        code = call()
        assert code == 0, f"kernel launch failed with code {code}"
        end = TimedEvent()
        end.record(None)
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


def _interleaved_compare(calls: "dict[str, object]", iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, dict[str, float]]":
    """Section 5's "interleaved A/B runs" requirement: warm up every named
    zero-arg callable, then alternate one call per name per round (not all
    of A then all of B) so run-to-run thermal/clock drift affects every
    candidate equally rather than biasing whichever ran first or last."""
    names = list(calls.keys())
    for name in names:
        for _ in range(warmup):
            code = calls[name]()
            assert code == 0, f"{name} kernel launch failed with code {code}"
            forge.cuda.synchronize()

    pairs: "dict[str, list]" = {name: [] for name in names}
    for _ in range(iterations):
        for name in names:
            start = TimedEvent()
            start.record(None)
            code = calls[name]()
            assert code == 0, f"{name} kernel launch failed with code {code}"
            end = TimedEvent()
            end.record(None)
            pairs[name].append((start, end))
    forge.cuda.synchronize()

    out = {}
    for name in names:
        samples = [elapsed_ms(s, e) for s, e in pairs[name]]
        out[name] = {
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        }
    return out


class _ForwardCandidates:
    """Directly-callable, complete-pipeline forward implementations for one
    shape: production baseline, Candidate A (im2col+GEMM, both im2col
    variants), Candidate B (half-fused GEMM). Every call allocates its own
    output buffer(s) fresh (matching real usage -- `CUDABackend.conv2d`
    never reuses a buffer across calls either) so timing includes real
    allocator cost, not an artificially reused buffer.
    """

    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
        self.N, self.Cin, self.Cout, self.H, self.W, self.K = N, Cin, Cout, H, W, K
        self.S, self.P = S, P
        self.Hout, self.Wout = Hout, Wout
        self.M = N * Hout * Wout
        self.Kdim = Cin * K * K

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        w = forge.Tensor(rng.standard_normal((Cout, Cin, K, K)).astype(np.float32), device="cuda")
        b = forge.Tensor(rng.standard_normal((Cout,)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        self.x, self.w, self.b = x._data, w._data, b._data

        self.fn_fwd = getattr(self.lib, "cf_conv2d_forward_f32")
        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )
        self.out_ptr = self.backend._alloc(N * Cout * Hout * Wout * 4)

    # -- complete pipelines (each returns 0 on success, matches _time_calls) --
    def baseline(self) -> int:
        return self.fn_fwd(self.x.ptr, self.w.ptr, self.b.ptr, self.out_ptr, *self.shape_args, ctypes.c_int(1), None)

    def candidate_a_plain(self) -> int:
        self._last = conv2d_forward_im2col_gemm(self.backend, self.x, self.w, self.b, (self.S, self.S), (self.P, self.P), use_smem_im2col=False)
        return 0

    def candidate_a_smem(self) -> int:
        self._last = conv2d_forward_im2col_gemm(self.backend, self.x, self.w, self.b, (self.S, self.S), (self.P, self.P), use_smem_im2col=True)
        return 0

    def candidate_b(self) -> int:
        self._last = conv2d_forward_halffused_gemm(self.backend, self.x, self.w, self.b, (self.S, self.S), (self.P, self.P))
        return 0

    # -- Candidate A sub-stages, for isolated-stage decomposition ------------
    def stage_im2col(self) -> int:
        if im2col_smem_fits(np.float32, self.H, self.W):
            self._xcol = im2col_smem(self.backend, self.x, self.N, self.Cin, self.H, self.W, self.K, self.K, self.S, self.S, self.P, self.P, self.Hout, self.Wout)
        else:
            self._xcol = im2col(self.backend, self.x, self.N, self.Cin, self.H, self.W, self.K, self.K, self.S, self.S, self.P, self.P, self.Hout, self.Wout)
        return 0

    def stage_transpose(self) -> int:
        self._weightT = transpose_weight(self.backend, self.w, self.Cout, self.Kdim)
        return 0

    def stage_matmul(self) -> int:
        from forge.backend.cuda.backend import CUDAStorage

        out_ptr = self.backend._alloc(self.M * self.Cout * 4)
        fn = getattr(self.lib, "cf_matmul_f32")
        code = fn(self._xcol.ptr, self._weightT.ptr, out_ptr, ctypes.c_int(self.M), ctypes.c_int(self.Kdim), ctypes.c_int(self.Cout), None)
        # Reassigning `self._out_mat` (rather than stashing a raw pointer)
        # lets the *previous* iteration's buffer be freed by ordinary
        # refcounting as soon as it's replaced -- an earlier draft stashed
        # only the raw pointer and re-wrapped it in a fresh `CUDAStorage`
        # inside `stage_permute` on every call, which double-frees the same
        # pointer from the second call onward (only one `CUDAStorage` may
        # ever own a given pointer) and produced a wildly inflated
        # `batch_128` permute timing from the resulting allocator/driver
        # corruption -- the exact "leaked/dangling buffer" mistake M33's own
        # profiling kernels made once, per that milestone's report.
        self._out_mat = CUDAStorage(out_ptr, (self.M, self.Cout), np.dtype(np.float32), self.lib)
        return code

    def stage_permute(self) -> int:
        self._last = output_permute(self.backend, self._out_mat, self.b, self.N, self.Cout, self.Hout, self.Wout)
        return 0


def _run_nvcc_ptxas_verbose() -> "dict[str, dict]":
    """`nvcc -Xptxas -v` against the unmodified `kernels.cu`, filtered down to
    the forward-Conv2d-relevant kernels (Section 2 of the M41 brief).

    Mirrors `build.ensure_kernel_library`'s own MSVC-`cl.exe`-on-PATH retry
    (Windows `nvcc` delegates host compilation to `cl.exe`, not normally on
    PATH outside a Developer Command Prompt) -- an earlier version of this
    function invoked `nvcc` directly and silently failed (`returncode=1`,
    "Cannot find compiler 'cl.exe'"), which produced an empty `per_kernel_
    lines` result with no visible error in the console summary.
    """
    import os

    from forge.backend.cuda import build as _build

    src = _build._SOURCE
    arch = _build._ARCH
    cmd = [
        "nvcc", "-Xptxas", "-v", "-arch=" + arch, "-c", str(src),
        "-o", str(Path(src).parent / "_m41_ptxas_scratch.obj"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        msvc_bin = _build._find_msvc_bin()
        if msvc_bin is not None:
            env = os.environ.copy()
            env["PATH"] = str(msvc_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    text = result.stdout + result.stderr

    kernels_of_interest = [
        "k_conv2d_forward", "k_conv2d_output_permute", "k_conv2d_forward_halffused_gemm", "k_matmul", "k_im2col_conv2d",
    ]
    per_kernel: "dict[str, list]" = {name: [] for name in kernels_of_interest}
    current = None
    for line in text.splitlines():
        for name in kernels_of_interest:
            if name in line and ("_Z" in line or "for '" in line or name + "<" in line or name + "(" in line):
                current = name
        if "registers" in line and current is not None:
            per_kernel[current].append(line.strip())

    scratch = Path(src).parent / "_m41_ptxas_scratch.obj"
    if scratch.exists():
        scratch.unlink()

    return {"raw_stdout": result.stdout, "raw_stderr": result.stderr, "per_kernel_lines": per_kernel, "returncode": result.returncode}


def _classify(ms: float, flops: int, nbytes: int, ceilings: "roofline.Ceilings") -> dict:
    seconds = ms / 1000.0
    ai = roofline.arithmetic_intensity(flops, nbytes)
    gflops = (flops / seconds) / 1e9 if seconds > 0 else 0.0
    gbps = (nbytes / seconds) / 1e9 if seconds > 0 else 0.0
    cls = roofline.classify(gflops, seconds, ai, ceilings)
    return {
        "achieved_gflops": gflops, "achieved_gbps": gbps, "arithmetic_intensity": ai,
        "fraction_of_compute_ceiling": gflops / ceilings.compute_gflops if ceilings.compute_gflops else 0.0,
        "classification": cls.label, "note": cls.note,
    }


def _profile_and_benchmark_shape(name: str, cfg: "dict[str, int]", ceilings: "roofline.Ceilings") -> dict:
    _check_fits_in_vram(cfg)
    cand = _ForwardCandidates(cfg)
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    Hout, Wout = cand.Hout, cand.Wout

    isolated_baseline = _time_calls(cand.baseline)
    fwd_flops = roofline.flops_conv2d_forward(N, Cout, Hout, Wout, Cin, K, K)
    fwd_bytes = roofline.bytes_conv2d_forward(N, Cin, H, W, Cout, K, K, Hout, Wout)
    roofline_baseline = _classify(isolated_baseline["mean_ms"], fwd_flops, fwd_bytes, ceilings)

    complete = _interleaved_compare({
        "baseline": cand.baseline,
        "candidate_a_plain": cand.candidate_a_plain,
        "candidate_a_smem": cand.candidate_a_smem,
        "candidate_b": cand.candidate_b,
    })

    stages = _interleaved_compare({
        "im2col": cand.stage_im2col,
        "transpose": cand.stage_transpose,
    })
    stages["matmul"] = _time_calls(cand.stage_matmul)
    stages["permute"] = _time_calls(cand.stage_permute)
    stage_total_ms = sum(stages[s]["mean_ms"] for s in ("im2col", "transpose", "matmul", "permute"))

    return {
        "name": name, "config": cfg, "Hout": Hout, "Wout": Wout,
        "isolated_baseline_ms": isolated_baseline,
        "roofline_baseline": roofline_baseline,
        "complete_pipeline_ms": complete,
        "candidate_a_stages_ms": stages,
        "candidate_a_stage_total_ms": stage_total_ms,
        "speedup_candidate_a_plain": complete["baseline"]["mean_ms"] / complete["candidate_a_plain"]["mean_ms"],
        "speedup_candidate_a_smem": complete["baseline"]["mean_ms"] / complete["candidate_a_smem"]["mean_ms"],
        "speedup_candidate_b": complete["baseline"]["mean_ms"] / complete["candidate_b"]["mean_ms"],
    }


def _run() -> dict:
    if not is_cuda_available():
        raise SystemExit("CUDA is not available on this machine; M41 profiling requires real hardware.")

    print("-- nvcc -Xptxas -v (register/occupancy analysis) --")
    ptxas = _run_nvcc_ptxas_verbose()
    for name, lines in ptxas["per_kernel_lines"].items():
        print(f"  {name}:")
        for line in lines:
            print(f"    {line}")

    print("\n-- fresh hardware ceilings (M35 methodology, this session) --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s, bandwidth: {ceilings.bandwidth_gbps:.2f} GB/s")

    results = {}
    for name, cfg in ALL_SHAPES.items():
        print(f"\n-- {name}: {cfg} --")
        r = _profile_and_benchmark_shape(name, cfg, ceilings)
        results[name] = r
        b = r["complete_pipeline_ms"]["baseline"]["mean_ms"]
        a_plain = r["complete_pipeline_ms"]["candidate_a_plain"]["mean_ms"]
        a_smem = r["complete_pipeline_ms"]["candidate_a_smem"]["mean_ms"]
        cb = r["complete_pipeline_ms"]["candidate_b"]["mean_ms"]
        print(f"   baseline={b:.4f}ms  A(plain)={a_plain:.4f}ms ({r['speedup_candidate_a_plain']:.2f}x)  "
              f"A(smem)={a_smem:.4f}ms ({r['speedup_candidate_a_smem']:.2f}x)  B={cb:.4f}ms ({r['speedup_candidate_b']:.2f}x)  "
              f"[baseline {r['roofline_baseline']['fraction_of_compute_ceiling']*100:.1f}% of ceiling, {r['roofline_baseline']['classification']}]")

    payload = {
        "environment": collect_environment(),
        "ceilings": {"practical_compute_gflops": ceilings.compute_gflops, "practical_bandwidth_gbps": ceilings.bandwidth_gbps},
        "ptxas": {"per_kernel_lines": ptxas["per_kernel_lines"], "returncode": ptxas["returncode"]},
        "shapes": results,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/m41_conv2d_forward_profile.json"))
    args = parser.parse_args()

    payload = _run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
