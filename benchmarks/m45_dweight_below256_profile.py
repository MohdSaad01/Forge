"""M45 — CUDA dWeight below-256-weight-element warp-shuffle reduction
(profile -> analyze -> design -> benchmark -> select).

M44 found the below-`CONV2D_WEIGHT_REDUCE_THRESHOLD` (256) dWeight dispatch
path (`k_conv2d_backward_weight_reduce`, one 256-thread block per weight
element, shared-memory tree reduction, unchanged since M21) is now the single
largest contributor to the real MNIST training step (20.19%), while reaching
only 3.1-3.2% of the practical compute ceiling across every below-threshold
shape tested -- the worst roofline efficiency of any candidate measured to
date, and unchanged by two prior investigations (M33 shared-memory
cooperative reduction at a *different* weight-element regime, M34
im2col+GEMM). M44 recommended a genuinely untried angle at this specific
target: warp-shuffle-based cooperative reduction.

This script does NOT assume that recommendation is correct -- it profiles
and benchmarks the baseline (Phases 0-1), then benchmarks two warp-shuffle
candidates against it, before any acceptance decision is made:

  * `warpreduce` (M33's `k_conv2d_backward_weight_warp`, unmodified, already
    in `kernels.cu` -- one full 32-lane warp per weight element, multiple
    warps/weights packed per block via `warps_per_block`). M33 already
    measured this at `mnist_conv1` (972-2.12ms vs. the 2.19ms production
    baseline in that session) but never turned it into an accept/reject
    decision at this specific target, since M33's own focus was the
    >= 1,152-weight-element regime.
  * `warpsubgroup` (new this milestone, `k_conv2d_backward_weight_warp_
    subgroup`) -- a configurable sub-warp group size (8/16/32 lanes per
    weight element) so more independent weight-groups can share one warp
    when a full 32-lane group is more parallelism than one weight element's
    reduction usefully absorbs. `group_size=32` degenerates to exactly
    `warpreduce`'s own algorithm, so only `group_size in (8, 16)` are swept
    here (32 is already covered by `warpreduce` itself -- not re-tested).

Both candidates are reused/added as profiling-only kernels
(`cf_conv2d_backward_weight_{warpreduce,warpsubgroup}_*`), never called by
`CUDABackend` -- production dispatch is untouched by this script, per the
milestone's own "do not modify production dispatch until independently
benchmarked" instruction.

    python -m benchmarks.m45_dweight_below256_profile
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
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from . import m35_hardware, m35_mnist, roofline
from .conv2d_backward_profile import _RawConv2dBackward, _check_fits_in_vram, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30
CONV2D_WEIGHT_REDUCE_THRESHOLD = 256  # must match kernels.cu

WARPS_PER_BLOCK_SWEEP = (1, 2, 4, 8, 16)  # warpreduce: threads/block = 32*this
SUBGROUP_CONFIGS = tuple(
    (group_size, warps_per_block)
    for group_size in (8, 16)  # 32 degenerates to warpreduce itself -- not retested
    for warps_per_block in (2, 4, 8)
)


# -- Phase 0/1: fresh ceilings + nvcc -Xptxas -v resource analysis -----------


def _run_nvcc_ptxas_verbose() -> dict:
    from forge.backend.cuda import build as _build

    src = _build._SOURCE
    arch = _build._ARCH
    scratch = Path(src).parent / "_m45_ptxas_scratch.obj"
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
        "k_conv2d_backward_weight_reduce", "k_conv2d_backward_weight_warp",
        "k_conv2d_backward_weight_warp_subgroup", "k_conv2d_backward_weight",
    ]
    per_kernel: "dict[str, list]" = {name: [] for name in kernels_of_interest}
    current = None
    for line in text.splitlines():
        # `kernels.cu` defines dozens of unrelated `__global__` kernels
        # between these four (M15-M44 history) -- nvcc's ptxas output is not
        # grouped by source order, so `current` must be reset at *every*
        # "Compiling entry function" boundary (matched or not), never left
        # sticky across an unrelated kernel's own header. Otherwise an
        # unrelated kernel's "Used N registers" line silently gets
        # misattributed to whichever target kernel was last actually seen
        # (a real bug caught while writing this script: without the reset,
        # `k_conv2d_backward_weight_reduce` picked up ~60 unrelated lines
        # from every kernel compiled between it and the next target match).
        # Longest-name-first match so `k_conv2d_backward_weight_warp_subgroup`
        # is not misattributed to the `k_conv2d_backward_weight_warp` prefix,
        # and `k_conv2d_backward_weight_reduce`/`_warp*` are not misattributed
        # to the bare `k_conv2d_backward_weight` prefix. Itanium name mangling
        # embeds each identifier as plain ASCII (length-prefixed), so a
        # substring match against the mangled "Compiling entry function"
        # line is reliable.
        if "Compiling entry function" in line:
            current = None
            for name in sorted(kernels_of_interest, key=len, reverse=True):
                if name in line:
                    current = name
                    break
        if "registers" in line and current is not None:
            per_kernel[current].append(line.strip())

    if scratch.exists():
        scratch.unlink()
    return {"per_kernel_lines": per_kernel, "returncode": result.returncode}


# -- Raw, directly-callable below-256 dWeight kernel variants ----------------


class _RawDWeightBelow256:
    """Directly-callable below-256 dWeight kernel variants (profiling-only).

    Mirrors `benchmarks/conv2d_backward_weight_profile.py`'s `_RawDWeight`
    exactly for `current`/`warpreduce`, and adds the new `warpsubgroup`
    candidate -- a separate class (rather than extending that M33 file) to
    keep each milestone's own profiling-only scope self-contained, matching
    M40/M43/M44's own convention of defining a fresh `_Raw*` class per
    milestone rather than editing a prior one.
    """

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
        self.fn_warpreduce = getattr(self.lib, "cf_conv2d_backward_weight_warpreduce_f32")
        self.fn_warpsubgroup = getattr(self.lib, "cf_conv2d_backward_weight_warpsubgroup_f32")

    def current(self, stream_handle) -> int:
        """The actual production dispatcher (picks block-reduce below 256, as here)."""
        return self.fn_current(self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args, stream_handle)

    def warpreduce(self, warps_per_block: int, stream_handle) -> int:
        return self.fn_warpreduce(
            self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args,
            ctypes.c_int(warps_per_block), stream_handle,
        )

    def warpsubgroup(self, warps_per_block: int, group_size: int, stream_handle) -> int:
        return self.fn_warpsubgroup(
            self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args,
            ctypes.c_int(warps_per_block), ctypes.c_int(group_size), stream_handle,
        )


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    # Per-iteration synchronization (matching `conv2d_backward_weight_profile.
    # py`'s own `_time_phase`) -- below-256 shapes never approach WDDM's ~2s
    # TDR watchdog, but this keeps both scripts' timing methodology identical.
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


def _classify(mean_ms: float, cfg: "dict[str, int]", ceilings: "roofline.Ceilings") -> dict:
    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    Hout, Wout = _hout_wout(H, K, cfg["S"], cfg["P"]), _hout_wout(W, K, cfg["S"], cfg["P"])
    flops = roofline.flops_conv2d_dweight(Cout, Cin, K, K, N, Hout, Wout)
    nbytes = roofline.bytes_conv2d_dweight(N, Cin, H, W, Cout, K, K, Hout, Wout)
    ai = roofline.arithmetic_intensity(flops, nbytes)
    achieved_gflops = flops / (mean_ms / 1000) / 1e9 if mean_ms > 0 else 0.0
    ceiling = ceilings.roofline_ceiling_gflops(ai)
    return {"achieved_gflops": achieved_gflops, "ceiling_gflops": ceiling,
            "fraction_of_ceiling": achieved_gflops / ceiling if ceiling > 0 else 0.0}


def _profile_shape(name: str, cfg: "dict[str, int]", ceilings: "roofline.Ceilings") -> dict:
    _check_fits_in_vram(cfg)
    raw = _RawDWeightBelow256(cfg)
    assert raw.weight_elements < CONV2D_WEIGHT_REDUCE_THRESHOLD, (
        f"{name}: weight_elements={raw.weight_elements} is not below the M45 target threshold"
    )

    current = _time_phase(raw.current)
    warpreduce = {wp: _time_phase(lambda sh, wp=wp: raw.warpreduce(wp, sh)) for wp in WARPS_PER_BLOCK_SWEEP}
    warpsubgroup = {
        f"{gs}x{wp}": _time_phase(lambda sh, wp=wp, gs=gs: raw.warpsubgroup(wp, gs, sh))
        for gs, wp in SUBGROUP_CONFIGS
    }

    best_warpreduce_wp = min(warpreduce, key=lambda wp: warpreduce[wp]["mean_ms"])
    best_warpsubgroup_cfg = min(warpsubgroup, key=lambda k: warpsubgroup[k]["mean_ms"])
    best_candidate_ms = min(warpreduce[best_warpreduce_wp]["mean_ms"], warpsubgroup[best_warpsubgroup_cfg]["mean_ms"])

    return {
        "name": name,
        "config": cfg,
        "weight_elements": raw.weight_elements,
        "reduction_size": raw.reduction_size,
        "current_ms": current,
        "current_classification": _classify(current["mean_ms"], cfg, ceilings),
        "warpreduce_ms": {str(wp): v for wp, v in warpreduce.items()},
        "warpsubgroup_ms": warpsubgroup,
        "best_warpreduce_warps_per_block": best_warpreduce_wp,
        "best_warpreduce_ms": warpreduce[best_warpreduce_wp]["mean_ms"],
        "best_warpsubgroup_config": best_warpsubgroup_cfg,
        "best_warpsubgroup_ms": warpsubgroup[best_warpsubgroup_cfg]["mean_ms"],
        "best_candidate_ms": best_candidate_ms,
        "best_candidate_classification": _classify(best_candidate_ms, cfg, ceilings),
        "speedup_current_vs_best_candidate": current["mean_ms"] / best_candidate_ms,
    }


# -- Phase 2: weight-element-count sweep (fixed MNIST-like reduction) --------

_MNIST_LIKE = {"N": 64, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1}

WEIGHT_ELEMENT_SWEEP = {
    # label -> (Cin, Cout) at K=3 -> weight_elements = 9*Cin*Cout
    "we_16": (1, 1),     # ceil: 9*1*1=9 -- "very small" anchor below every
                          # other point; Cin*Cout chosen to land near, not
                          # exactly on, each requested label (K=3 forces
                          # multiples of 9) -- exact we recorded in results.
    "we_32": (1, 3),      # 27
    "we_64": (1, 7),      # 63
    "we_72": (1, 8),      # 72 -- mnist_conv1 itself
    "we_128": (2, 7),     # 126
    "we_192": (3, 7),     # 189
    "we_255": (4, 7),     # 252 -- closest multiple of 9 below 256
}


def _weight_element_sweep(ceilings: "roofline.Ceilings") -> list:
    results = []
    for label, (cin, cout) in WEIGHT_ELEMENT_SWEEP.items():
        cfg = dict(_MNIST_LIKE, Cin=cin, Cout=cout)
        results.append(_profile_shape(label, cfg, ceilings))
    return results


# -- Phase 3: reduction-size sweep (fixed weight_elements=72, mnist_conv1's --
# -- own Cin/Cout/K split) ----------------------------------------------------

REDUCTION_SIZE_SWEEP = {
    # label -> (N, H, W) at Cin=1, Cout=8, K=3, S=1, P=1 (we=72 fixed)
    "reduction_small": (4, 8, 8),        # N*Hout*Wout = 256
    "reduction_medium": (16, 16, 16),    # 4,096
    "reduction_mnist": (64, 28, 28),     # 50,176 -- mnist_conv1 itself
    "reduction_large": (128, 40, 40),    # 204,800
}


def _reduction_size_sweep(ceilings: "roofline.Ceilings") -> list:
    results = []
    for label, (n, h, w) in REDUCTION_SIZE_SWEEP.items():
        cfg = {"N": n, "Cin": 1, "Cout": 8, "H": h, "W": w, "K": 3, "S": 1, "P": 1}
        results.append(_profile_shape(label, cfg, ceilings))
    return results


# -- Phase 4: channel/kernel-size configuration sweep (K=1/3/5) --------------

CHANNEL_KERNEL_SWEEP = {
    # label -> (Cin, Cout, K), N=H=W chosen so weight_elements stays < 256
    "k1_wide": (4, 16, 1),     # we=64, 1x1 conv (no spatial reuse)
    "k3_mnist": (1, 8, 3),     # we=72, mnist_conv1 itself (duplicate check)
    "k5_narrow": (1, 4, 5),    # we=100, 5x5 conv
}


def _channel_kernel_sweep(ceilings: "roofline.Ceilings") -> list:
    results = []
    for label, (cin, cout, k) in CHANNEL_KERNEL_SWEEP.items():
        cfg = {"N": 64, "Cin": cin, "Cout": cout, "H": 28, "W": 28, "K": k, "S": 1, "P": 1}
        results.append(_profile_shape(label, cfg, ceilings))
    return results


# -- Phase 5: complete `conv2d_backward` (dInput + dWeight + dBias) A/B ------


class _RawConv2dBackwardCandidate(_RawConv2dBackward):
    """`_RawConv2dBackward` (M32, unmodified base class) with dWeight's call
    swapped for one of this milestone's candidates -- dInput/dBias stay the
    exact same production kernels either way, so this isolates dWeight's own
    contribution to the complete backward pass without touching production
    dispatch."""

    def __init__(self, cfg: "dict[str, int]", warps_per_block: int, group_size: "int | None"):
        super().__init__(cfg)
        self.fn_warpreduce = getattr(self.lib, "cf_conv2d_backward_weight_warpreduce_f32")
        self.fn_warpsubgroup = getattr(self.lib, "cf_conv2d_backward_weight_warpsubgroup_f32")
        self._warps_per_block = warps_per_block
        self._group_size = group_size

    def dweight_candidate(self, stream_handle) -> int:
        if self._group_size is None:
            return self.fn_warpreduce(
                self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args,
                ctypes.c_int(self._warps_per_block), stream_handle,
            )
        return self.fn_warpsubgroup(
            self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args,
            ctypes.c_int(self._warps_per_block), ctypes.c_int(self._group_size), stream_handle,
        )


def _complete_backward_comparison(name: str, cfg: "dict[str, int]", best_wp: int, best_gs: "int | None") -> dict:
    raw = _RawConv2dBackwardCandidate(cfg, best_wp, best_gs)

    def _full_current(stream_handle):
        c1 = raw.backward_input(stream_handle)
        c2 = raw.backward_weight(stream_handle)
        c3 = raw.backward_bias(stream_handle)
        return c1 or c2 or c3

    def _full_candidate(stream_handle):
        c1 = raw.backward_input(stream_handle)
        c2 = raw.dweight_candidate(stream_handle)
        c3 = raw.backward_bias(stream_handle)
        return c1 or c2 or c3

    current = _time_phase(_full_current)
    candidate = _time_phase(_full_candidate)
    return {
        "name": name,
        "current_full_backward_ms": current,
        "candidate_full_backward_ms": candidate,
        "speedup": current["mean_ms"] / candidate["mean_ms"],
    }


# -- Phase 6: boundary test (255 / 256 / 257 weight elements) ----------------


def _boundary_test() -> list:
    results = []
    # Exact boundary values require weight_elements land exactly on 255/256/257.
    # K=3 forces multiples of 9 (Cin*Cout*9), so K=1 is used here instead to hit
    # the boundary exactly: weight_elements = Cin*Cout at K=1.
    for cin, cout, label, expected_we in [
        (15, 17, "we_255_below", 255),
        (16, 16, "we_256_exact", 256),
        (1, 257, "we_257_above", 257),
    ]:
        cfg = {"N": 8, "Cin": cin, "Cout": cout, "H": 12, "W": 12, "K": 1, "S": 1, "P": 1}
        we = cout * cin * 1 * 1
        assert we == expected_we, f"{label}: expected {expected_we}, got {we}"
        dispatched_path = "block-reduce (below 256)" if we < CONV2D_WEIGHT_REDUCE_THRESHOLD else "per-thread (>= 256)"
        raw = _RawDWeightBelow256(cfg) if we < CONV2D_WEIGHT_REDUCE_THRESHOLD else None
        entry = {"name": label, "weight_elements": we, "dispatched_path": dispatched_path}
        if raw is not None:
            entry["current_ms"] = _time_phase(raw.current)["mean_ms"]
        results.append(entry)
    return results


# -- Phase 7: fresh Amdahl fraction (mnist_conv1's own dWeight share of the --
# -- real, fresh MNIST training step) ----------------------------------------


def _amdahl_fraction(ceilings: "roofline.Ceilings", mnist_conv1_current_ms: float) -> dict:
    mnist_profile = m35_mnist._run(ceilings)
    ranking = mnist_profile["kernel_ranking"]
    total_step_ms = sum(r["mean_seconds"] for r in ranking) * 1000.0
    fraction = mnist_conv1_current_ms / total_step_ms if total_step_ms > 0 else 0.0
    conv2d_backward_entry = next((r for r in ranking if r["op"] == "backward:conv2d"), None)
    return {
        "total_step_ms": total_step_ms,
        "mnist_conv1_dweight_below256_ms": mnist_conv1_current_ms,
        "mnist_conv1_dweight_below256_fraction_of_step": fraction,
        "backward_conv2d_full_fraction_of_step": (
            conv2d_backward_entry["percent_of_step"] / 100.0 if conv2d_backward_entry else None
        ),
    }


def _amdahl_projection(fraction: float, speedups=(1.15, 1.25, 1.5, 2.0)) -> "dict[str, float]":
    return {f"{s}x": 1.0 / ((1 - fraction) + fraction / s) for s in speedups}


# -- Report rendering ---------------------------------------------------------


def _render_report(profile: dict) -> str:
    lines = ["=== M45 dWeight below-256 warp-shuffle profile (940MX, real CUDA) ===", ""]
    lines.append("-- nvcc -Xptxas -v --")
    for name, ptxlines in profile["ptxas"]["per_kernel_lines"].items():
        lines.append(f"  {name}:")
        for line in ptxlines:
            lines.append(f"    {line}")
    lines.append("")

    for section_name, key in [
        ("weight-element sweep (fixed mnist-like reduction)", "weight_element_sweep"),
        ("reduction-size sweep (fixed weight_elements=72)", "reduction_size_sweep"),
        ("channel/kernel-size sweep", "channel_kernel_sweep"),
    ]:
        lines.append(f"-- {section_name} --")
        header = f"{'shape':<16}{'weight#':>8}{'reduce#':>10}{'current(ms)':>13}{'best_cand(ms)':>15}{'speedup':>9}{'cur%ceil':>10}{'cand%ceil':>10}"
        lines.append(header)
        for r in profile[key]:
            lines.append(
                f"{r['name']:<16}{r['weight_elements']:>8}{r['reduction_size']:>10}"
                f"{r['current_ms']['mean_ms']:>13.4f}{r['best_candidate_ms']:>15.4f}"
                f"{r['speedup_current_vs_best_candidate']:>8.2f}x"
                f"{r['current_classification']['fraction_of_ceiling']*100:>9.1f}%"
                f"{r['best_candidate_classification']['fraction_of_ceiling']*100:>9.1f}%"
            )
        lines.append("")

    lines.append("-- complete conv2d_backward (dInput+dWeight+dBias) A/B, best candidate per shape --")
    for r in profile["complete_backward"]:
        lines.append(f"  {r['name']}: current={r['current_full_backward_ms']['mean_ms']:.4f}ms "
                      f"candidate={r['candidate_full_backward_ms']['mean_ms']:.4f}ms speedup={r['speedup']:.3f}x")
    lines.append("")

    lines.append("-- boundary test (255 / 256 / 257 weight elements) --")
    for r in profile["boundary"]:
        extra = f" current={r['current_ms']:.4f}ms" if "current_ms" in r else " (>=256, not this milestone's target)"
        lines.append(f"  {r['name']}: weight_elements={r['weight_elements']} -> {r['dispatched_path']}{extra}")
    lines.append("")

    lines.append("-- Amdahl (fresh, this session) --")
    a = profile["amdahl"]
    lines.append(f"  mnist_conv1 dWeight below-256 fraction of full step: {a['mnist_conv1_dweight_below256_fraction_of_step']*100:.2f}%")
    lines.append(f"  (backward:conv2d whole-op fraction, for context: "
                  f"{a['backward_conv2d_full_fraction_of_step']*100 if a['backward_conv2d_full_fraction_of_step'] else float('nan'):.2f}%)")
    for s, v in profile["amdahl_projection"].items():
        lines.append(f"  {s} kernel speedup -> {v:.3f}x whole-step speedup")
    lines.append("")

    lines.append("-- Acceptance decision --")
    lines.append(profile["decision"]["summary"])
    return "\n".join(lines)


# -- Decision -----------------------------------------------------------------

ACCEPTANCE_SPEEDUP_BAR = 1.15


def _decide(profile: dict) -> dict:
    mnist_row = next(r for r in profile["weight_element_sweep"] if r["name"] == "we_72")
    mnist_speedup = mnist_row["speedup_current_vs_best_candidate"]
    all_rows = profile["weight_element_sweep"] + profile["reduction_size_sweep"] + profile["channel_kernel_sweep"]
    min_speedup = min(r["speedup_current_vs_best_candidate"] for r in all_rows)
    regressions = [r["name"] for r in all_rows if r["speedup_current_vs_best_candidate"] < 0.95]

    accepted = mnist_speedup >= ACCEPTANCE_SPEEDUP_BAR and not regressions
    summary = (
        f"mnist_conv1-shape speedup (isolated kernel) = {mnist_speedup:.3f}x "
        f"(bar: >= {ACCEPTANCE_SPEEDUP_BAR}x). Min speedup across all below-256 sweep shapes: "
        f"{min_speedup:.3f}x. Regressions (>5% slower than production): {regressions or 'none'}. "
        f"Decision: {'ACCEPT' if accepted else 'REJECT'} -- "
        + ("clears the acceptance bar with no regression." if accepted else
           "does not clear the acceptance bar (or a regression was found); "
           "production dispatch left unchanged.")
    )
    return {"accepted": accepted, "mnist_conv1_speedup": mnist_speedup, "min_speedup": min_speedup,
            "regressions": regressions, "summary": summary}


def _run() -> dict:
    print("-- Phase 0: fresh practical hardware ceilings --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s, "
          f"bandwidth ceiling: {ceilings.bandwidth_gbps:.2f} GB/s")

    print("-- Phase 1: nvcc -Xptxas -v resource analysis --")
    ptxas = _run_nvcc_ptxas_verbose()

    print("-- Phase 2: weight-element-count sweep --")
    weight_element_sweep = _weight_element_sweep(ceilings)

    print("-- Phase 3: reduction-size sweep --")
    reduction_size_sweep = _reduction_size_sweep(ceilings)

    print("-- Phase 4: channel/kernel-size sweep --")
    channel_kernel_sweep = _channel_kernel_sweep(ceilings)

    print("-- Phase 5: complete conv2d_backward A/B (mnist_conv1 + we_192 + reduction_large) --")
    mnist_conv1_row = next(r for r in weight_element_sweep if r["name"] == "we_72")
    we_192_row = next(r for r in weight_element_sweep if r["name"] == "we_192")
    reduction_large_row = next(r for r in reduction_size_sweep if r["name"] == "reduction_large")
    complete_backward = []
    for label, cfg, row in [
        ("mnist_conv1", dict(_MNIST_LIKE, Cin=1, Cout=8), mnist_conv1_row),
        ("we_192", dict(_MNIST_LIKE, Cin=3, Cout=7), we_192_row),
        ("reduction_large", {"N": 128, "Cin": 1, "Cout": 8, "H": 40, "W": 40, "K": 3, "S": 1, "P": 1}, reduction_large_row),
    ]:
        best_wp_str = row["best_warpreduce_warps_per_block"]
        best_wr_ms = row["best_warpreduce_ms"]
        best_sg_ms = row["best_warpsubgroup_ms"]
        if best_wr_ms <= best_sg_ms:
            complete_backward.append(_complete_backward_comparison(label, cfg, int(best_wp_str), None))
        else:
            gs_str, wp_str = row["best_warpsubgroup_config"].split("x")
            complete_backward.append(_complete_backward_comparison(label, cfg, int(wp_str), int(gs_str)))

    print("-- Phase 6: boundary test (255/256/257 weight elements) --")
    boundary = _boundary_test()

    print("-- Phase 7: fresh Amdahl fraction (real MNIST training step, this session) --")
    amdahl = _amdahl_fraction(ceilings, mnist_conv1_row["current_ms"]["mean_ms"])
    amdahl_projection = _amdahl_projection(amdahl["mnist_conv1_dweight_below256_fraction_of_step"])

    profile = {
        "ptxas": ptxas,
        "weight_element_sweep": weight_element_sweep,
        "reduction_size_sweep": reduction_size_sweep,
        "channel_kernel_sweep": channel_kernel_sweep,
        "complete_backward": complete_backward,
        "boundary": boundary,
        "amdahl": amdahl,
        "amdahl_projection": amdahl_projection,
    }
    profile["decision"] = _decide(profile)
    return profile


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m45_dweight_below256_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m45_dweight_below256_profile requires real CUDA hardware.")
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
