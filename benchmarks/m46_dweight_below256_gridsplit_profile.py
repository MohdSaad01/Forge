"""M46 -- CUDA dWeight below-256-weight-element grid-split reduction
(PROFILE -> ANALYZE -> DESIGN -> BENCHMARK -> SELECT).

M45 rejected warp-shuffle cooperative reduction for the below-
`CONV2D_WEIGHT_REDUCE_THRESHOLD` (256) dWeight path (`k_conv2d_backward_
weight_reduce`, one 256-thread block per weight element, unchanged since
M21) and corrected M44's diagnosis: the production kernel already launches
`weight_elements x 256` threads -- both warp candidates *reduced* total
launched parallelism (32 or 8/16 threads per weight element), so
"insufficient threads" was never the right framing. M45 named the untried
axis directly: grid-level parallelism. The production kernel launches
exactly one *block* per weight element; this script investigates whether
launching several blocks per weight element (each reducing a disjoint slice
of the `N*Hout*Wout` reduction dimension, combined by a cheap second-stage
reduction) helps, and does not assume it does -- Phase 1's occupancy
diagnostics are collected and interpreted *before* any benchmark is run.

    python -m benchmarks.m46_dweight_below256_gridsplit_profile
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
from forge.backend.cuda.backend import CUDAStorage, get_cuda_backend, is_cuda_available
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from . import m35_hardware, m35_mnist, roofline
from .conv2d_backward_profile import _RawConv2dBackward, _check_fits_in_vram, _hout_wout
from .environment import collect_environment

WARMUP = 5
ITERATIONS = 30
CONV2D_WEIGHT_REDUCE_THRESHOLD = 256  # must match kernels.cu
CONV2D_REDUCE_THREADS = 256  # must match kernels.cu

NUM_SPLITS_SWEEP = (1, 2, 4, 8)  # 1 is a same-kernel sanity check (should equal production, modulo split overhead)

# -- Maxwell (compute capability 5.0) architectural limits, used only to
# -- interpret the real `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
# -- query below -- never used in place of it.
CC50_MAX_THREADS_PER_SM = 2048
CC50_MAX_WARPS_PER_SM = 64
CC50_SM_COUNT = 3


# -- Phase 0/1: fresh ceilings + nvcc -Xptxas -v + real occupancy query ------


def _run_nvcc_ptxas_verbose() -> dict:
    from forge.backend.cuda import build as _build

    src = _build._SOURCE
    arch = _build._ARCH
    scratch = Path(src).parent / "_m46_ptxas_scratch.obj"
    cmd = ["nvcc", "-Xptxas", "-v", "-arch=" + arch, "-c", str(src), "-o", str(scratch)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        msvc_bin = _build._find_msvc_bin()
        if msvc_bin is not None:
            env = os.environ.copy()
            env["PATH"] = str(msvc_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    text = result.stdout + result.stderr

    kernels_of_interest = ["k_conv2d_backward_weight_reduce_gridsplit", "k_conv2d_backward_weight_reduce"]
    per_kernel: "dict[str, list]" = {name: [] for name in kernels_of_interest}
    current = None
    for line in text.splitlines():
        # See M45's identical comment (`benchmarks/m45_dweight_below256_
        # profile.py`) for why `current` must reset at every "Compiling
        # entry function" boundary and matches are checked longest-name-first.
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


def _occupancy_query() -> dict:
    """Real `cudaOccupancyMaxActiveBlocksPerMultiprocessor` query (f32, the
    exact launch configuration each kernel actually uses: 256 threads/block,
    `256*sizeof(float)` shared memory) -- not a hand-derived estimate."""
    backend = get_cuda_backend()
    lib = backend._lib
    shared_mem_bytes = CONV2D_REDUCE_THREADS * 4  # sizeof(float)

    out = ctypes.c_int(0)
    fn_reduce = getattr(lib, "cf_occupancy_conv2d_backward_weight_reduce_f32")
    code1 = fn_reduce(ctypes.c_int(shared_mem_bytes), ctypes.byref(out))
    max_active_blocks_reduce = out.value

    out2 = ctypes.c_int(0)
    fn_gridsplit = getattr(lib, "cf_occupancy_conv2d_backward_weight_reduce_gridsplit_f32")
    code2 = fn_gridsplit(ctypes.c_int(shared_mem_bytes), ctypes.byref(out2))
    max_active_blocks_gridsplit = out2.value

    assert code1 == 0 and code2 == 0, f"occupancy query failed: {code1}, {code2}"

    return {
        "shared_mem_bytes": shared_mem_bytes,
        "threads_per_block": CONV2D_REDUCE_THREADS,
        "max_active_blocks_per_sm_reduce": max_active_blocks_reduce,
        "max_active_blocks_per_sm_gridsplit": max_active_blocks_gridsplit,
        "sm_count": CC50_SM_COUNT,
        "max_resident_blocks_total_reduce": max_active_blocks_reduce * CC50_SM_COUNT,
        "max_resident_blocks_total_gridsplit": max_active_blocks_gridsplit * CC50_SM_COUNT,
        "occupancy_fraction_reduce": (max_active_blocks_reduce * CONV2D_REDUCE_THREADS) / CC50_MAX_THREADS_PER_SM,
        "occupancy_fraction_gridsplit": (max_active_blocks_gridsplit * CONV2D_REDUCE_THREADS) / CC50_MAX_THREADS_PER_SM,
        "note": (
            "Both kernels are register-bound (not shared-memory- or thread-count-bound) at this launch "
            "configuration -- see the ptxas register counts alongside this. Identical (or near-identical) "
            "occupancy between the two confirms grid-splitting does not change per-SM concurrent-block "
            "capacity; it only changes total block count and per-block work size."
        ),
    }


# -- Raw, directly-callable kernel variants -----------------------------------


class _RawDWeightGridsplit:
    """Directly-callable production + grid-split dWeight kernel variants
    (profiling-only). Mirrors `m45_dweight_below256_profile._RawDWeightBelow256`."""

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
        self.grad_w_ptr = self.backend._alloc(self.weight_elements * 4)
        self.fn_current = getattr(self.lib, "cf_conv2d_backward_weight_f32")
        self.fn_partial = getattr(self.lib, "cf_conv2d_backward_weight_gridsplit_partial_f32")
        self.fn_reduce = getattr(self.lib, "cf_dweight_splitk_reduce_f32")

        # Largest sweep num_splits determines the partial buffer's max size --
        # allocated once, reused across every `gridsplit` call in this instance
        # (same convention as `_RawConv2dBackward`'s own preallocated output buffers).
        max_splits = max(NUM_SPLITS_SWEEP)
        self.partial_ptr = self.backend._alloc(max_splits * self.weight_elements * 4)
        self.out_ptr = self.backend._alloc(self.weight_elements * 4)

    def current(self, stream_handle) -> int:
        return self.fn_current(self.grad_out.ptr, self.x.ptr, self.grad_w_ptr, *self.shape_args, stream_handle)

    def gridsplit(self, num_splits: int, stream_handle) -> int:
        c1 = self.fn_partial(
            self.grad_out.ptr, self.x.ptr, self.partial_ptr, *self.shape_args,
            ctypes.c_int(num_splits), stream_handle,
        )
        c2 = self.fn_reduce(
            self.partial_ptr, self.out_ptr,
            ctypes.c_int(1), ctypes.c_int(self.weight_elements), ctypes.c_int(num_splits),
            stream_handle,
        )
        return c1 or c2

    def check_correctness(self, num_splits: int) -> float:
        """Returns max abs error between `gridsplit(num_splits)` and `current()`."""
        code = self.current(None)
        assert code == 0
        forge.cuda.synchronize()
        ref = self.backend.to_numpy(CUDAStorage(self.grad_w_ptr, (self.weight_elements,), self.grad_out.dtype, self.lib)).copy()

        code = self.gridsplit(num_splits, None)
        assert code == 0
        forge.cuda.synchronize()
        out = self.backend.to_numpy(CUDAStorage(self.out_ptr, (self.weight_elements,), self.grad_out.dtype, self.lib))
        return float(np.abs(out - ref).max())


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    """Single-call timing -- kept only for callers that genuinely have just
    one thing to time (e.g. the boundary test). Any A/B comparison must use
    `_interleaved_multi_time` below instead: a first same-session run of
    this script timed `current` then each candidate as separate
    fully-sequential blocks (this function, called once per variant) and
    found a 1.47x "win" at `mnist_conv1` that a true round-robin-interleaved
    rerun (`_interleaved_multi_time`) could not reproduce (1.02x, within
    noise) -- the 940MX's clock/power state visibly drifts over a
    multi-second block of repeated launches (see the M46 report's own
    **Methodology correction** section), so block-sequential timing biases
    whichever variant is measured later. This is the same class of mistake
    M37 was caught making (an apples-to-oranges timing setup producing an
    illusory win) via a different mechanism (clock drift vs. buffer reuse) --
    same lesson: cross-check any single-block-timed win against an
    interleaved rerun before trusting it."""
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


def _interleaved_multi_time(calls: "dict[str, object]", iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, dict]":
    """True round-robin-interleaved A/B/.../N timing: every named call is
    warmed up first, then each of `iterations` rounds times *every* call
    once, in the same fixed order, before moving to the next round. Any
    slow GPU clock/power-state drift over the run affects every variant
    almost equally within each round, instead of biasing whichever variant
    happens to be timed in a later block -- the fix for the block-sequential
    `_time_phase`-per-variant methodology this milestone's own first attempt
    used and had to retract (see `_time_phase`'s docstring)."""
    stream_handle = None
    for fn in calls.values():
        for _ in range(warmup):
            code = fn(stream_handle)
            assert code == 0, f"kernel launch failed with code {code}"
            forge.cuda.synchronize()

    samples: "dict[str, list]" = {name: [] for name in calls}
    for _ in range(iterations):
        for name, fn in calls.items():
            start = TimedEvent()
            start.record(stream_handle)
            code = fn(stream_handle)
            assert code == 0, f"kernel launch failed with code {code}"
            end = TimedEvent()
            end.record(stream_handle)
            forge.cuda.synchronize()
            samples[name].append(elapsed_ms(start, end))

    return {
        name: {
            "mean_ms": statistics.mean(vals),
            "median_ms": statistics.median(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
            "stdev_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
        for name, vals in samples.items()
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
    raw = _RawDWeightGridsplit(cfg)
    assert raw.weight_elements < CONV2D_WEIGHT_REDUCE_THRESHOLD, (
        f"{name}: weight_elements={raw.weight_elements} is not below the M46 target threshold"
    )

    # Correctness check for every split count before any timing -- a wrong
    # kernel timing fast is worthless.
    max_errs = {ns: raw.check_correctness(ns) for ns in NUM_SPLITS_SWEEP}

    # True round-robin-interleaved A/B (see `_interleaved_multi_time`'s
    # docstring) -- `current` and every `num_splits` candidate are timed in
    # the same fixed round-robin order, `ITERATIONS` rounds, so GPU
    # clock/power-state drift over the run cannot bias one variant over
    # another the way block-sequential timing did in this milestone's first
    # (retracted) attempt.
    calls = {"current": raw.current}
    calls.update({f"ns{ns}": (lambda sh, ns=ns: raw.gridsplit(ns, sh)) for ns in NUM_SPLITS_SWEEP})
    timed = _interleaved_multi_time(calls)

    current = timed["current"]
    gridsplit = {ns: timed[f"ns{ns}"] for ns in NUM_SPLITS_SWEEP}

    best_ns = min(gridsplit, key=lambda ns: gridsplit[ns]["mean_ms"])
    best_candidate_ms = gridsplit[best_ns]["mean_ms"]

    return {
        "name": name,
        "config": cfg,
        "weight_elements": raw.weight_elements,
        "reduction_size": raw.reduction_size,
        "max_abs_errors": max_errs,
        "current_ms": current,
        "current_classification": _classify(current["mean_ms"], cfg, ceilings),
        "gridsplit_ms": {str(ns): v for ns, v in gridsplit.items()},
        "best_num_splits": best_ns,
        "best_candidate_ms": best_candidate_ms,
        "best_candidate_classification": _classify(best_candidate_ms, cfg, ceilings),
        "speedup_current_vs_best_candidate": current["mean_ms"] / best_candidate_ms,
    }


# -- Phase 2: weight-element-count sweep (fixed MNIST-like reduction) --------

_MNIST_LIKE = {"N": 64, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1}

WEIGHT_ELEMENT_SWEEP = {
    # label -> (Cin, Cout) at K=3 -> weight_elements = 9*Cin*Cout
    "we_72": (1, 8),      # 72 -- mnist_conv1 itself
    "we_96": (2, 5),      # 90 -- closest multiple of 9 to 96
    "we_128": (2, 7),     # 126
    "we_144": (2, 8),     # 144
    "we_192": (3, 7),     # 189
    "we_216": (3, 8),     # 216
    "we_240": (4, 7),     # 252 -- reused for both "240" and "255" anchors below (K=3 forces multiples of 9)
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
    swapped for this milestone's grid-split candidate -- dInput/dBias stay
    the exact same production kernels either way."""

    def __init__(self, cfg: "dict[str, int]", num_splits: int):
        super().__init__(cfg)
        weight_elements = self.Cout * self.Cin * self.K * self.K
        self.fn_partial = getattr(self.lib, "cf_conv2d_backward_weight_gridsplit_partial_f32")
        self.fn_reduce = getattr(self.lib, "cf_dweight_splitk_reduce_f32")
        self.partial_ptr = self.backend._alloc(num_splits * weight_elements * 4)
        self._num_splits = num_splits
        self._weight_elements = weight_elements

    def dweight_candidate(self, stream_handle) -> int:
        c1 = self.fn_partial(
            self.grad_out.ptr, self.x.ptr, self.partial_ptr, *self.shape_args,
            ctypes.c_int(self._num_splits), stream_handle,
        )
        c2 = self.fn_reduce(
            self.partial_ptr, self.grad_w_ptr,
            ctypes.c_int(1), ctypes.c_int(self._weight_elements), ctypes.c_int(self._num_splits),
            stream_handle,
        )
        return c1 or c2


def _complete_backward_comparison(name: str, cfg: "dict[str, int]", best_num_splits: int) -> dict:
    raw = _RawConv2dBackwardCandidate(cfg, best_num_splits)

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

    timed = _interleaved_multi_time({"current": _full_current, "candidate": _full_candidate})
    current, candidate = timed["current"], timed["candidate"]
    return {
        "name": name,
        "num_splits": best_num_splits,
        "current_full_backward_ms": current,
        "candidate_full_backward_ms": candidate,
        "speedup": current["mean_ms"] / candidate["mean_ms"],
    }


# -- Phase 6: boundary test (255 / 256 / 257 weight elements) ----------------


def _boundary_test() -> list:
    results = []
    for cin, cout, label, expected_we in [
        (15, 17, "we_255_below", 255),
        (16, 16, "we_256_exact", 256),
        (1, 257, "we_257_above", 257),
    ]:
        cfg = {"N": 8, "Cin": cin, "Cout": cout, "H": 12, "W": 12, "K": 1, "S": 1, "P": 1}
        we = cout * cin * 1 * 1
        assert we == expected_we, f"{label}: expected {expected_we}, got {we}"
        dispatched_path = "block-reduce (below 256)" if we < CONV2D_WEIGHT_REDUCE_THRESHOLD else "per-thread (>= 256)"
        raw = _RawDWeightGridsplit(cfg) if we < CONV2D_WEIGHT_REDUCE_THRESHOLD else None
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
    lines = ["=== M46 dWeight below-256 grid-split reduction profile (940MX, real CUDA) ===", ""]
    lines.append("-- nvcc -Xptxas -v --")
    for name, ptxlines in profile["ptxas"]["per_kernel_lines"].items():
        lines.append(f"  {name}:")
        for line in ptxlines:
            lines.append(f"    {line}")
    lines.append("")

    lines.append("-- real occupancy query (cudaOccupancyMaxActiveBlocksPerMultiprocessor) --")
    occ = profile["occupancy"]
    lines.append(f"  threads/block={occ['threads_per_block']}  shared_mem_bytes={occ['shared_mem_bytes']}")
    lines.append(f"  production (k_conv2d_backward_weight_reduce): {occ['max_active_blocks_per_sm_reduce']} blocks/SM "
                  f"({occ['occupancy_fraction_reduce']*100:.1f}% occupancy), "
                  f"{occ['max_resident_blocks_total_reduce']} resident blocks across {occ['sm_count']} SMs")
    lines.append(f"  gridsplit candidate: {occ['max_active_blocks_per_sm_gridsplit']} blocks/SM "
                  f"({occ['occupancy_fraction_gridsplit']*100:.1f}% occupancy), "
                  f"{occ['max_resident_blocks_total_gridsplit']} resident blocks across {occ['sm_count']} SMs")
    lines.append(f"  {occ['note']}")
    lines.append("")

    for section_name, key in [
        ("weight-element sweep (fixed mnist-like reduction)", "weight_element_sweep"),
        ("reduction-size sweep (fixed weight_elements=72)", "reduction_size_sweep"),
        ("channel/kernel-size sweep", "channel_kernel_sweep"),
    ]:
        lines.append(f"-- {section_name} --")
        header = f"{'shape':<16}{'weight#':>8}{'reduce#':>10}{'current(ms)':>13}{'best_cand(ms)':>15}{'best_ns':>8}{'speedup':>9}{'cur%ceil':>10}{'cand%ceil':>10}"
        lines.append(header)
        for r in profile[key]:
            lines.append(
                f"{r['name']:<16}{r['weight_elements']:>8}{r['reduction_size']:>10}"
                f"{r['current_ms']['mean_ms']:>13.4f}{r['best_candidate_ms']:>15.4f}"
                f"{r['best_num_splits']:>8}"
                f"{r['speedup_current_vs_best_candidate']:>8.2f}x"
                f"{r['current_classification']['fraction_of_ceiling']*100:>9.1f}%"
                f"{r['best_candidate_classification']['fraction_of_ceiling']*100:>9.1f}%"
            )
        lines.append("")

    lines.append("-- complete conv2d_backward (dInput+dWeight+dBias) A/B, best candidate per shape --")
    for r in profile["complete_backward"]:
        lines.append(f"  {r['name']} (num_splits={r['num_splits']}): current={r['current_full_backward_ms']['mean_ms']:.4f}ms "
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

    lines.append("-- Correctness (max abs error vs production, per split count) --")
    for r in profile["weight_element_sweep"]:
        errs = ", ".join(f"ns={ns}: {e:.2e}" for ns, e in r["max_abs_errors"].items())
        lines.append(f"  {r['name']}: {errs}")
    lines.append("")

    lines.append("-- Acceptance decision --")
    lines.append(profile["decision"]["summary"])
    return "\n".join(lines)


# -- Decision -----------------------------------------------------------------

ACCEPTANCE_SPEEDUP_BAR = 1.15
CORRECTNESS_ATOL = 1e-3  # float32 accumulation-order noise across split counts


def _decide(profile: dict) -> dict:
    mnist_row = next(r for r in profile["weight_element_sweep"] if r["name"] == "we_72")
    mnist_speedup = mnist_row["speedup_current_vs_best_candidate"]
    all_rows = profile["weight_element_sweep"] + profile["reduction_size_sweep"] + profile["channel_kernel_sweep"]
    min_speedup = min(r["speedup_current_vs_best_candidate"] for r in all_rows)
    regressions = [r["name"] for r in all_rows if r["speedup_current_vs_best_candidate"] < 0.95]
    max_err = max(max(r["max_abs_errors"].values()) for r in all_rows)
    correctness_ok = max_err < CORRECTNESS_ATOL

    accepted = mnist_speedup >= ACCEPTANCE_SPEEDUP_BAR and not regressions and correctness_ok
    summary = (
        f"mnist_conv1-shape speedup (isolated kernel) = {mnist_speedup:.3f}x "
        f"(bar: >= {ACCEPTANCE_SPEEDUP_BAR}x). Min speedup across all below-256 sweep shapes: "
        f"{min_speedup:.3f}x. Regressions (>5% slower than production): {regressions or 'none'}. "
        f"Max correctness error across all shapes/splits: {max_err:.2e} "
        f"(bar: < {CORRECTNESS_ATOL:.0e}). "
        f"Decision: {'ACCEPT' if accepted else 'REJECT'} -- "
        + ("clears the acceptance bar with no regression." if accepted else
           "does not clear the acceptance bar (or a regression/correctness issue was found); "
           "production dispatch left unchanged.")
    )
    return {"accepted": accepted, "mnist_conv1_speedup": mnist_speedup, "min_speedup": min_speedup,
            "regressions": regressions, "max_correctness_error": max_err, "summary": summary}


def _run() -> dict:
    print("-- Phase 0: fresh practical hardware ceilings --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s, "
          f"bandwidth ceiling: {ceilings.bandwidth_gbps:.2f} GB/s")

    print("-- Phase 1: nvcc -Xptxas -v resource analysis + real occupancy query --")
    ptxas = _run_nvcc_ptxas_verbose()
    occupancy = _occupancy_query()

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
        complete_backward.append(_complete_backward_comparison(label, cfg, row["best_num_splits"]))

    print("-- Phase 6: boundary test (255/256/257 weight elements) --")
    boundary = _boundary_test()

    print("-- Phase 7: fresh Amdahl fraction (real MNIST training step, this session) --")
    amdahl = _amdahl_fraction(ceilings, mnist_conv1_row["current_ms"]["mean_ms"])
    amdahl_projection = _amdahl_projection(amdahl["mnist_conv1_dweight_below256_fraction_of_step"])

    profile = {
        "ptxas": ptxas,
        "occupancy": occupancy,
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
    parser.add_argument("--output", default="benchmarks/results/m46_dweight_below256_gridsplit_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m46_dweight_below256_gridsplit_profile requires real CUDA hardware.")
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
