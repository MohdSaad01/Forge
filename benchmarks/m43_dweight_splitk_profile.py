"""M43 Phase: profile the dWeight `blocks_y==1` split-K reduction (production
`k_dweight_halffused_gemm_splitk`, M38) to confirm or reject M42's atomic-
reduction root-cause hypothesis before any optimization is attempted.

M42 found this kernel reaches only 21.7-24.8% of the practical compute
ceiling -- roughly half of every other GEMM-dispatched Conv2d kernel,
including the structurally similar (non-split-K) forward half-fused GEMM
(43.5-48.7%). The one documented architectural difference is split-K's
atomic-accumulation combine step. This script gathers the evidence M42
explicitly deferred:

1. Isolated kernel time + full-pipeline time (permute + GEMM) at an
   expanded shape sweep: the two real `blocks_y==1` representative shapes
   (`mnist_conv2`, `large_spatial`) plus a dedicated `Cout in {1,2,4,8,16}`
   sweep at small/medium/large reduction lengths (`N*Hout*Wout`), so the
   `Cout=16`-only evidence M42 flagged as a limitation is no longer the
   only data point.
2. A `num_k_splits` sensitivity sweep at each of those shapes -- if
   atomic-accumulation contention (not occupancy) is the bottleneck, time
   should *not* improve monotonically with more splits past the point
   where occupancy is already saturated; if occupancy is still the
   limiter, more splits should keep helping.
3. `nvcc -Xptxas -v` resource re-analysis (register/shared-memory/spill)
   of the target kernel plus two comparison kernels (`k_conv2d_forward_
   halffused_gemm`, structurally identical minus split-K; `k_matmul_splitk`,
   the non-fused split-K GEMM used by the `blocks_y>=2` dWeight path).
4. A closed-form atomic-instruction-count model (one `atomicAdd` per
   thread with `row<Cout, col<Kdim` per k-split whose tile range is
   non-empty -- exactly what `k_dweight_halffused_gemm_splitk`'s own final
   `if` guard executes): `num_k_splits_active * Cout * Kdim`, where
   `num_k_splits_active = min(num_k_splits, total_tiles)`. Reported
   alongside each shape/split measurement so contention scaling can be
   read directly off the atomic count rather than inferred.

Fresh roofline ceilings (M35 methodology) classify every measurement.
No production code is modified -- this is measurement only, calling the
same compiled `cf_dweight_halffused_gemm_splitk_*`/`cf_conv2d_grad_output_
permute_*` functions `experimental_conv_halffused.dweight_halffused_gemm_
splitk` itself calls, with `num_k_splits` exposed directly (mirroring
`m37_dweight_candidates_profile.py`'s own raw-ctypes-call convention).

    python -m benchmarks.m43_dweight_splitk_profile
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import _MATMUL_TILE, _SUFFIX, get_cuda_backend, is_cuda_available
from forge.backend.cuda.experimental_conv_im2col import grad_output_permute, recommended_num_k_splits
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from . import m35_hardware, roofline
from .conv2d_backward_profile import SHAPES, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30
SPLIT_GRID = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def _hw(cfg: "dict[str, int]") -> "tuple[int, int]":
    return _hout_wout(cfg["H"], cfg["K"], cfg["S"], cfg["P"]), _hout_wout(cfg["W"], cfg["K"], cfg["S"], cfg["P"])


class _RawDweightHalffusedSplitk:
    """Raw ctypes access to the production `blocks_y==1` dWeight kernel
    (`cf_dweight_halffused_gemm_splitk_*`) with `num_k_splits` exposed as a
    direct argument, plus the permute stage it depends on -- same two
    compiled functions `experimental_conv_halffused.dweight_halffused_gemm_
    splitk` calls, no reimplementation."""

    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hw(cfg)
        self.N, self.Cin, self.Cout, self.H, self.W, self.K = N, Cin, Cout, H, W, K
        self.S, self.P, self.Hout, self.Wout = S, P, Hout, Wout
        self.M = N * Hout * Wout
        self.Kdim = Cin * K * K
        self.total_tiles = (self.M + _MATMUL_TILE - 1) // _MATMUL_TILE
        self.blocks_y = (Cout + _MATMUL_TILE - 1) // _MATMUL_TILE
        self.blocks_x = (self.Kdim + _MATMUL_TILE - 1) // _MATMUL_TILE

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        self.x, self.grad_out = x._data, grad_out._data

        suffix = _SUFFIX[np.dtype(np.float32)]
        self.fn_permute = getattr(self.lib, f"cf_conv2d_grad_output_permute_{suffix}")
        self.fn_gemm = getattr(self.lib, f"cf_dweight_halffused_gemm_splitk_{suffix}")

        self.dycolT_ptr = self.backend._alloc(Cout * self.M * 4)
        self.out_ptr = self.backend._alloc(Cout * self.Kdim * 4)
        self.permute_args = (ctypes.c_int(N), ctypes.c_int(Cout), ctypes.c_int(Hout), ctypes.c_int(Wout))
        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )

        code = self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, None)
        assert code == 0
        forge.cuda.synchronize()

    def permute(self, stream_handle) -> int:
        return self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, stream_handle)

    def gemm_only(self, num_k_splits: int):
        def call(stream_handle) -> int:
            return self.fn_gemm(
                self.dycolT_ptr, self.x.ptr, self.out_ptr, *self.shape_args,
                ctypes.c_int(num_k_splits), stream_handle,
            )
        return call

    def full(self, num_k_splits: int):
        def call(stream_handle) -> int:
            code = self.permute(stream_handle)
            if code != 0:
                return code
            return self.fn_gemm(
                self.dycolT_ptr, self.x.ptr, self.out_ptr, *self.shape_args,
                ctypes.c_int(num_k_splits), stream_handle,
            )
        return call

    def atomic_count(self, num_k_splits: int) -> int:
        active_splits = min(num_k_splits, self.total_tiles)
        return active_splits * self.Cout * self.Kdim


class _RawDweightTworeduce:
    """Raw ctypes access to Milestone 43 Candidate B (`experimental_conv_
    dweight_tworeduce`): same GEMM tile loads as the production kernel, but
    each split writes a disjoint partial slice (no atomic) and a second
    small kernel sums the `num_k_splits` axis. Allocates the largest partial
    buffer this instance will ever need once (`max_splits`), so the timed
    `full`/`gemm_only` calls never allocate."""

    def __init__(self, cfg: "dict[str, int]", max_splits: int):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hw(cfg)
        self.Cout, self.Kdim = Cout, Cin * K * K
        self.M = N * Hout * Wout

        rng = np.random.default_rng(0)
        x = forge.Tensor(rng.standard_normal((N, Cin, H, W)).astype(np.float32), device="cuda")
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        self.x, self.grad_out = x._data, grad_out._data

        suffix = _SUFFIX[np.dtype(np.float32)]
        self.fn_permute = getattr(self.lib, f"cf_conv2d_grad_output_permute_{suffix}")
        self.fn_partial = getattr(self.lib, f"cf_dweight_halffused_gemm_splitk_partial_{suffix}")
        self.fn_reduce = getattr(self.lib, f"cf_dweight_splitk_reduce_{suffix}")

        self.dycolT_ptr = self.backend._alloc(Cout * self.M * 4)
        self.partial_ptr = self.backend._alloc(max_splits * Cout * self.Kdim * 4)
        self.out_ptr = self.backend._alloc(Cout * self.Kdim * 4)
        self.permute_args = (ctypes.c_int(N), ctypes.c_int(Cout), ctypes.c_int(Hout), ctypes.c_int(Wout))
        self.shape_args = (
            ctypes.c_int(N), ctypes.c_int(Cin), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_int(Cout), ctypes.c_int(K), ctypes.c_int(K),
            ctypes.c_int(S), ctypes.c_int(S), ctypes.c_int(P), ctypes.c_int(P),
            ctypes.c_int(Hout), ctypes.c_int(Wout),
        )

        code = self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, None)
        assert code == 0
        forge.cuda.synchronize()

    def permute(self, stream_handle) -> int:
        return self.fn_permute(self.grad_out.ptr, self.dycolT_ptr, *self.permute_args, stream_handle)

    def gemm_only(self, num_k_splits: int):
        """Partial-GEMM + reduce, no permute -- comparable to `_RawDweightHalffusedSplitk.gemm_only`."""
        def call(stream_handle) -> int:
            code = self.fn_partial(
                self.dycolT_ptr, self.x.ptr, self.partial_ptr, *self.shape_args,
                ctypes.c_int(num_k_splits), stream_handle,
            )
            if code != 0:
                return code
            return self.fn_reduce(
                self.partial_ptr, self.out_ptr,
                ctypes.c_int(self.Cout), ctypes.c_int(self.Kdim), ctypes.c_int(num_k_splits), stream_handle,
            )
        return call

    def full(self, num_k_splits: int):
        def call(stream_handle) -> int:
            code = self.permute(stream_handle)
            if code != 0:
                return code
            return self.gemm_only(num_k_splits)(stream_handle)
        return call


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    stream_handle = None
    for _ in range(warmup):
        code = call(stream_handle)
        assert code == 0, f"kernel launch failed with code {code}"
        forge.cuda.synchronize()
    pairs = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(None)
        code = call(stream_handle)
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
    """Alternates one call per candidate per round (not all of A then all of
    B) so run-to-run thermal/clock drift affects every candidate equally --
    mirrors `m41_conv2d_forward_profile._interleaved_compare` exactly."""
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


def _candidate_b_comparison(name: str, cfg: "dict[str, int]") -> dict:
    """Interleaved A/B: production atomic-combine kernel vs. Candidate B's
    two-stage deterministic reduction, at several `num_k_splits` (including
    the production `recommended_num_k_splits` value and higher points where
    the extended sweep showed the atomic kernel starting to degrade)."""
    max_splits = 256
    atomic = _RawDweightHalffusedSplitk(cfg)
    tworeduce = _RawDweightTworeduce(cfg, max_splits=max_splits)
    recommended = recommended_num_k_splits(atomic.M)
    split_points = sorted(set(s for s in (recommended, 32, 64, 96, 128, 256) if s <= max_splits))

    results = []
    for splits in split_points:
        cmp = _interleaved_compare({
            "atomic_gemm_only": (lambda c=atomic.gemm_only(splits): c(None)),
            "tworeduce_gemm_only": (lambda c=tworeduce.gemm_only(splits): c(None)),
            "atomic_full": (lambda c=atomic.full(splits): c(None)),
            "tworeduce_full": (lambda c=tworeduce.full(splits): c(None)),
        })
        results.append({
            "num_k_splits": splits,
            "atomic_gemm_only_ms": cmp["atomic_gemm_only"]["mean_ms"],
            "tworeduce_gemm_only_ms": cmp["tworeduce_gemm_only"]["mean_ms"],
            "atomic_full_ms": cmp["atomic_full"]["mean_ms"],
            "tworeduce_full_ms": cmp["tworeduce_full"]["mean_ms"],
            "speedup_gemm_only": cmp["atomic_gemm_only"]["mean_ms"] / cmp["tworeduce_gemm_only"]["mean_ms"],
            "speedup_full": cmp["atomic_full"]["mean_ms"] / cmp["tworeduce_full"]["mean_ms"],
        })
    return {"name": name, "recommended_num_k_splits": recommended, "sweep": results}


def _classify(cfg: "dict[str, int]", ms: float, ceilings: roofline.Ceilings) -> dict:
    Hout, Wout = _hw(cfg)
    flops = roofline.flops_conv2d_dweight(cfg["Cout"], cfg["Cin"], cfg["K"], cfg["K"], cfg["N"], Hout, Wout)
    nbytes = roofline.bytes_conv2d_dweight(cfg["N"], cfg["Cin"], cfg["H"], cfg["W"], cfg["Cout"], cfg["K"], cfg["K"], Hout, Wout)
    seconds = ms / 1000.0
    gflops = flops / seconds / 1e9 if seconds > 0 else 0.0
    ai = roofline.arithmetic_intensity(flops, nbytes)
    c = roofline.classify(gflops, seconds, ai, ceilings)
    return {
        "gflops": gflops, "fraction_of_ceiling": c.fraction_of_ceiling, "label": c.label,
    }


def _profile_shape(name: str, cfg: "dict[str, int]", ceilings: roofline.Ceilings) -> dict:
    raw = _RawDweightHalffusedSplitk(cfg)
    recommended = recommended_num_k_splits(raw.M)

    permute_t = _time_phase(lambda sh: raw.permute(sh))
    gemm_t = _time_phase(raw.gemm_only(recommended))
    full_t = _time_phase(raw.full(recommended))

    split_sweep = []
    for splits in SPLIT_GRID:
        gemm_only_t = _time_phase(raw.gemm_only(splits))
        split_sweep.append({
            "num_k_splits": splits,
            "active_splits": min(splits, raw.total_tiles),
            "atomic_count": raw.atomic_count(splits),
            "gemm_only_ms": gemm_only_t["mean_ms"],
            "gemm_only_stdev_ms": gemm_only_t["stdev_ms"],
        })
    best = min(split_sweep, key=lambda r: r["gemm_only_ms"])

    return {
        "name": name, "config": cfg, "M": raw.M, "Kdim": raw.Kdim, "Cout": raw.Cout,
        "blocks_y": raw.blocks_y, "blocks_x": raw.blocks_x, "total_tiles": raw.total_tiles,
        "recommended_num_k_splits": recommended,
        "permute_ms": permute_t, "gemm_only_ms": gemm_t, "full_pipeline_ms": full_t,
        "classification_full_pipeline": _classify(cfg, full_t["mean_ms"], ceilings),
        "num_k_splits_sweep": split_sweep,
        "best_num_k_splits": best["num_k_splits"],
        "best_gemm_only_ms": best["gemm_only_ms"],
        "recommended_vs_best_gap": (
            next(r["gemm_only_ms"] for r in split_sweep if r["num_k_splits"] == recommended) / best["gemm_only_ms"]
        ),
    }


# -- Cout sweep at small/medium/large reduction lengths -----------------------


def _cout_reduction_sweep() -> list:
    """`Cout in {1,2,4,8,16}` (all `blocks_y==1`) x `N in {small,medium,large}`
    reduction length (`M = N*Hout*Wout`), fixed `Cin=8,H=W=13,K=3,S=1,P=1`
    (mnist_conv2's own spatial shape) -- isolates whether the efficiency gap
    is `Cout`-dependent, reduction-length-dependent, or both/neither."""
    base = {"Cin": 8, "H": 13, "W": 13, "K": 3, "S": 1, "P": 1}
    results = []
    for reduction_label, n in (("small", 8), ("medium", 64), ("large", 512)):
        for cout in (1, 2, 4, 8, 16):
            cfg = dict(base)
            cfg["N"], cfg["Cout"] = n, cout
            raw = _RawDweightHalffusedSplitk(cfg)
            recommended = recommended_num_k_splits(raw.M)
            gemm_t = _time_phase(raw.gemm_only(recommended))
            results.append({
                "reduction_label": reduction_label, "N": n, "Cout": cout, "M": raw.M,
                "recommended_num_k_splits": recommended, "gemm_only_ms": gemm_t["mean_ms"],
            })
    return results


# -- K / reduction sweep: vary N, Hout, Wout, Cin, KH, KW independently -------


def _k_reduction_sweep() -> list:
    """Varies `N`, `H`/`W` (-> `Hout`/`Wout`), `Cin`, `K` one at a time from a
    fixed `blocks_y==1` base (`mnist_conv2`-shaped) so `N*Hout*Wout` spans
    small/medium/large reduction sizes independently of `Cout`, per Section 3
    of the M43 brief."""
    base = {"N": 64, "Cin": 8, "Cout": 16, "H": 13, "W": 13, "K": 3, "S": 1, "P": 1}
    sweeps = {
        "n_small": {"N": 8}, "n_large": {"N": 256},
        "spatial_small": {"H": 7, "W": 7}, "spatial_large": {"H": 32, "W": 32},
        "cin_low": {"Cin": 1}, "cin_high": {"Cin": 16},
        "k1": {"K": 1}, "k5": {"K": 5},
    }
    results = []
    for label, override in sweeps.items():
        cfg = dict(base)
        cfg.update(override)
        raw = _RawDweightHalffusedSplitk(cfg)
        recommended = recommended_num_k_splits(raw.M)
        gemm_t = _time_phase(raw.gemm_only(recommended))
        results.append({
            "label": label, "config": cfg, "M": raw.M, "Kdim": raw.Kdim,
            "recommended_num_k_splits": recommended, "gemm_only_ms": gemm_t["mean_ms"],
        })
    return results


# -- nvcc -Xptxas -v resource analysis ----------------------------------------


def _run_nvcc_ptxas_verbose() -> dict:
    from forge.backend.cuda import build as _build

    src = _build._SOURCE
    arch = _build._ARCH
    scratch = Path(src).parent / "_m43_ptxas_scratch.obj"
    cmd = ["nvcc", "-Xptxas", "-v", "-arch=" + arch, "-c", str(src), "-o", str(scratch)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        msvc_bin = _build._find_msvc_bin()
        if msvc_bin is not None:
            env = os.environ.copy()
            env["PATH"] = str(msvc_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    text = result.stdout + result.stderr

    kernels_of_interest = [
        "k_dweight_halffused_gemm_splitk", "k_dweight_halffused_gemm_splitk_partial",
        "k_dweight_splitk_reduce", "k_conv2d_forward_halffused_gemm", "k_matmul_splitk",
    ]
    per_kernel: "dict[str, list]" = {name: [] for name in kernels_of_interest}
    current = None
    for line in text.splitlines():
        for name in kernels_of_interest:
            if name in line and ("_Z" in line or "for '" in line or name + "<" in line or name + "(" in line):
                current = name
        if "registers" in line and current is not None:
            per_kernel[current].append(line.strip())

    if scratch.exists():
        scratch.unlink()
    return {"per_kernel_lines": per_kernel, "returncode": result.returncode}


def _render_report(profile: dict) -> str:
    lines = ["=== M43 dWeight blocks_y==1 split-K reduction profile (940MX, real CUDA) ===", ""]
    lines.append("-- nvcc -Xptxas -v --")
    for name, ptxlines in profile["ptxas"]["per_kernel_lines"].items():
        lines.append(f"  {name}:")
        for line in ptxlines:
            lines.append(f"    {line}")
    lines.append("")
    lines.append("-- representative shapes --")
    header = f"{'shape':<14}{'Cout':>5}{'M':>10}{'Kdim':>7}{'blocks_x':>9}{'rec_splits':>11}{'gemm(ms)':>10}{'full(ms)':>10}{'%ceil':>8}{'best_splits':>12}{'rec/best':>9}"
    lines.append(header)
    for r in profile["shapes"]:
        lines.append(
            f"{r['name']:<14}{r['Cout']:>5}{r['M']:>10}{r['Kdim']:>7}{r['blocks_x']:>9}"
            f"{r['recommended_num_k_splits']:>11}{r['gemm_only_ms']['mean_ms']:>10.4f}"
            f"{r['full_pipeline_ms']['mean_ms']:>10.4f}{r['classification_full_pipeline']['fraction_of_ceiling']*100:>7.1f}%"
            f"{r['best_num_k_splits']:>12}{r['recommended_vs_best_gap']:>8.3f}x"
        )
    lines.append("")
    lines.append("-- num_k_splits sensitivity (mnist_conv2) --")
    mnist2 = next(r for r in profile["shapes"] if r["name"] == "mnist_conv2")
    lines.append(f"{'splits':>8}{'active':>8}{'atomics':>10}{'gemm(ms)':>10}")
    for r in mnist2["num_k_splits_sweep"]:
        lines.append(f"{r['num_k_splits']:>8}{r['active_splits']:>8}{r['atomic_count']:>10}{r['gemm_only_ms']:>10.4f}")
    lines.append("")
    lines.append("-- Cout x reduction-length sweep --")
    lines.append(f"{'reduction':<10}{'N':>6}{'Cout':>6}{'M':>10}{'rec_splits':>11}{'gemm(ms)':>10}")
    for r in profile["cout_reduction_sweep"]:
        lines.append(f"{r['reduction_label']:<10}{r['N']:>6}{r['Cout']:>6}{r['M']:>10}{r['recommended_num_k_splits']:>11}{r['gemm_only_ms']:>10.4f}")
    lines.append("")
    lines.append("-- K/reduction sweep --")
    lines.append(f"{'label':<16}{'M':>10}{'Kdim':>7}{'rec_splits':>11}{'gemm(ms)':>10}")
    for r in profile["k_reduction_sweep"]:
        lines.append(f"{r['label']:<16}{r['M']:>10}{r['Kdim']:>7}{r['recommended_num_k_splits']:>11}{r['gemm_only_ms']:>10.4f}")
    lines.append("")
    lines.append("-- Candidate B (two-stage reduce) vs. production atomic combine, interleaved A/B --")
    for cb in profile["candidate_b_comparison"]:
        lines.append(f"  {cb['name']} (recommended_num_k_splits={cb['recommended_num_k_splits']}):")
        lines.append(f"  {'splits':>8}{'atomic_gemm':>13}{'2reduce_gemm':>14}{'speedup_gemm':>13}{'atomic_full':>13}{'2reduce_full':>14}{'speedup_full':>13}")
        for r in cb["sweep"]:
            lines.append(
                f"  {r['num_k_splits']:>8}{r['atomic_gemm_only_ms']:>13.4f}{r['tworeduce_gemm_only_ms']:>14.4f}"
                f"{r['speedup_gemm_only']:>12.3f}x{r['atomic_full_ms']:>13.4f}{r['tworeduce_full_ms']:>14.4f}{r['speedup_full']:>12.3f}x"
            )
    return "\n".join(lines)


def _run() -> dict:
    # nvcc runs first, and deliberately before any GPU timing: it is a
    # CPU-only subprocess that takes tens of seconds, and the 940MX's WDDM
    # driver aggressively downclocks an idle GPU -- if it ran *between* two
    # GPU-timing phases, the phase right after it would measure artificially
    # slow (cold-clock) numbers relative to a phase measured back-to-back
    # with prior GPU activity. Confirmed empirically this session: an
    # earlier draft's isolated `gemm_only` measurement (~3.5ms) disagreed
    # with the *same* call's own `num_k_splits==16` sweep entry (~1.0ms,
    # measured moments later with no intervening CPU-only gap) by ~3.4x.
    # Keeping every GPU-timing phase contiguous (nvcc first, then ceilings,
    # then all shape/sweep measurements back-to-back) removes the gap.
    print("-- Phase 1: nvcc -Xptxas -v --")
    ptxas = _run_nvcc_ptxas_verbose()

    print("-- Phase 0: fresh hardware ceilings (M35 methodology, this session) --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s")

    print("-- Phase 2: representative shapes (mnist_conv2, large_spatial) + num_k_splits sweep --")
    shapes = []
    for name in ("mnist_conv2", "large_spatial"):
        shapes.append(_profile_shape(name, SHAPES[name], ceilings))

    print("-- Phase 3: Cout x reduction-length sweep --")
    cout_reduction_sweep = _cout_reduction_sweep()

    print("-- Phase 4: K/reduction sweep --")
    k_reduction_sweep = _k_reduction_sweep()

    print("-- Phase 5: Candidate B (two-stage split-K reduction) vs. production atomic combine --")
    candidate_b = [_candidate_b_comparison(name, SHAPES[name]) for name in ("mnist_conv2", "large_spatial")]

    return {
        "ceilings": {"compute_gflops": ceilings.compute_gflops, "bandwidth_gbps": ceilings.bandwidth_gbps},
        "ptxas": ptxas,
        "shapes": shapes,
        "cout_reduction_sweep": cout_reduction_sweep,
        "k_reduction_sweep": k_reduction_sweep,
        "candidate_b_comparison": candidate_b,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m43_dweight_splitk_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m43_dweight_splitk_profile requires real CUDA hardware.")
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
