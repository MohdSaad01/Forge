"""M48: CUDA Conv2d dInput low-Cin value assessment (PROFILE -> ANALYZE ->
VALUE-ASSESS -> DESIGN -> BENCHMARK -> DECIDE -> VALIDATE).

M47 selected the M36 channel-fused `dInput` kernel's low-`Cin` regime
(`Cin=1`, `mnist_conv1`'s own real shape) as the next candidate: worst
live-candidate roofline efficiency measured (4.7% of the practical compute
ceiling), with a genuinely untried structural angle (M36's channel-fusion
technique amortizes a fixed register cost over multiple `Cin` accumulators;
at `Cin=1` there is only one).

This milestone does not assume that finding makes the target correct to
optimize. It first establishes real-workload relevance (grepping Forge's
own examples/tests/benchmarks for every `Conv2d` shape actually exercised),
then a fresh baseline (dispatch verification, `nvcc -Xptxas -v`, a real
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` query) *before* writing any
candidate kernel, per the brief's PROFILE-before-DESIGN discipline.

The occupancy/register evidence turned out more nuanced than M47's own
hypothesis: reducing the channel-fused kernel's accumulator array from
`MAX_CIN_REG=16` down to a `Cin=1`-specialized `CIN_MAX=1` only drops f32
register usage 54->48 and occupancy 4->5 blocks/SM (25%, not the dramatic
jump a pure register-pressure story would predict) -- yet a same-session,
round-robin-interleaved, order-independent A/B measured a reproducible
2.24x-2.62x *isolated* kernel speedup anyway, attributable to per-thread
instruction overhead (the unspecialized kernel's `#pragma unroll`ed
accumulator loop still emits 16 runtime-checked iterations per `Cout*KH*KW`
step even when only the first is ever useful). This cleared the acceptance
bar by a wide margin, so `k_conv2d_backward_input_channelfused_lowcin<T,1>`
(`kernels.cu`) was promoted to production, dispatched at
`Cin<=CONV2D_DINPUT_LOWCIN_MAX_CIN=1` -- see `docs/performance/
m48-dinput-value-assessment.md` for the full report and decision.

    python -m benchmarks.m48_dinput_value_assessment
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
from .conv2d_backward_profile import SHAPES, _RawConv2dBackward, _check_fits_in_vram, _hout_wout
from .environment import collect_environment
from .m44_bottleneck_recharacterization import _amdahl, _dinput_cin_sweep, _profile_shape_full

WARMUP = 8
ITERATIONS = 40

CC50_MAX_THREADS_PER_SM = 2048
CC50_SM_COUNT = 3

# -- Real-workload relevance (Section 4 of the brief) ------------------------
#
# Established by grepping `examples/`, `tests/`, `benchmarks/` for every
# `Conv2d(...)` instantiation this session (not re-derived at runtime --
# these are file/line facts about the repository as of this milestone).
REAL_WORKLOAD_RELEVANCE = {
    "real_forge_networks_with_conv2d": [
        {
            "file": "examples/mnist/model.py",
            "layers": ["Conv2d(1, 8, kernel_size=3)", "Conv2d(8, 16, kernel_size=3)"],
            "note": "Forge's only real (non-test, non-benchmark-synthetic) trained network. "
            "Its first layer is the only real Cin=1 shape anywhere in the repository.",
        },
    ],
    "synthetic_cin1_shapes": [
        "tests/test_conv.py, tests/test_cuda_conv.py, tests/test_cuda_streams.py, "
        "tests/test_lifetime.py, tests/test_serialization.py, tests/test_cuda_persistence.py, "
        "tests/test_sequential_flatten_dropout_integration.py, tests/test_conv_trainer_integration.py: "
        "small Cin=1 Conv2d layers used purely for correctness/API coverage on tiny inputs "
        "(6x6-8x8 spatial, N<=4), never as a performance workload.",
        "benchmarks/*.py sweep files (conv2d_backward_*, m4x_*): Cin=1 appears only as one point "
        "in a deliberate Cin sweep (1,2,4,8,16,...), constructed specifically to characterize the "
        "kernel across Cin -- not evidence of a second real Cin=1 workload.",
    ],
    "conclusion": (
        "Cin=1 matters to exactly one real Forge workload: mnist_conv1, the first layer of the "
        "one real trained network in the repository. It is not a broadly representative CNN input "
        "pattern within Forge's own scope today -- optimizing it helps MNIST specifically, not a "
        "class of real workloads."
    ),
}


# -- Fresh dispatch verification (read directly from kernels.cu/backend.py) --


def _dinput_dispatch_decision(cfg: "dict[str, int]") -> dict:
    Cin = cfg["Cin"]
    if Cin <= 1:
        return {"Cin": Cin, "path": "k_conv2d_backward_input_channelfused_lowcin<T,1> (M48, NEW production)"}
    if Cin <= 16:
        return {"Cin": Cin, "path": "k_conv2d_backward_input_channelfused (M36, unchanged)"}
    return {"Cin": Cin, "path": "k_conv2d_backward_input (M32 original, fallback -- unused by any real Forge shape)"}


# -- nvcc -Xptxas -v (Section 5/6) -------------------------------------------


def _run_nvcc_ptxas_verbose() -> dict:
    from forge.backend.cuda import build as _build

    src = _build._SOURCE
    arch = _build._ARCH
    scratch = Path(src).parent / "_m48_ptxas_scratch.obj"
    cmd = ["nvcc", "-Xptxas", "-v", "-arch=" + arch, "-c", str(src), "-o", str(scratch)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        msvc_bin = _build._find_msvc_bin()
        if msvc_bin is not None:
            env = os.environ.copy()
            env["PATH"] = str(msvc_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    text = result.stdout + result.stderr

    kernels_of_interest = ["k_conv2d_backward_input_channelfused_lowcin", "k_conv2d_backward_input_channelfused"]
    per_kernel: "dict[str, list]" = {name: [] for name in kernels_of_interest}
    current = None
    for line in text.splitlines():
        if "Compiling entry function" in line:
            current = None
            for name in sorted(kernels_of_interest, key=len, reverse=True):
                if name in line:
                    current = name
                    break
        if ("registers" in line or "stack frame" in line) and current is not None:
            per_kernel[current].append(line.strip())

    if scratch.exists():
        scratch.unlink()
    return {"per_kernel_lines": per_kernel, "returncode": result.returncode}


# -- Real occupancy query (Section 5/6) --------------------------------------


def _occupancy_query() -> dict:
    backend = get_cuda_backend()
    lib = backend._lib

    out1 = ctypes.c_int(0)
    code1 = lib.cf_occupancy_conv2d_backward_input_channelfused_f32(ctypes.byref(out1))
    out2 = ctypes.c_int(0)
    code2 = lib.cf_occupancy_conv2d_backward_input_channelfused_lowcin1_f32(ctypes.byref(out2))
    assert code1 == 0 and code2 == 0, f"occupancy query failed: {code1}, {code2}"

    threads_per_block = 256
    return {
        "threads_per_block": threads_per_block,
        "max_active_blocks_per_sm_channelfused": out1.value,
        "max_active_blocks_per_sm_lowcin1": out2.value,
        "occupancy_fraction_channelfused": (out1.value * threads_per_block) / CC50_MAX_THREADS_PER_SM,
        "occupancy_fraction_lowcin1": (out2.value * threads_per_block) / CC50_MAX_THREADS_PER_SM,
        "sm_count": CC50_SM_COUNT,
    }


# -- Isolated interleaved A/B: production (lowcin1) vs. pre-M48 baseline -----
# -- (forced channelfused call) at Cin=1, plus a Cin=2 no-regression check ---

_CIN1_AB_SHAPES = {
    "mnist_conv1 (N=64,Cin=1,Cout=8,H=W=28,K=3,P=1)": {"N": 64, "Cin": 1, "Cout": 8, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
    "cin1_large_spatial (N=16,Cin=1,Cout=32,H=W=64,K=3,P=1)": {"N": 16, "Cin": 1, "Cout": 32, "H": 64, "W": 64, "K": 3, "S": 1, "P": 1},
    "cin1_batch64_cout16 (N=64,Cin=1,Cout=16,H=W=28,K=3,P=1)": {"N": 64, "Cin": 1, "Cout": 16, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1},
    "cin1_strided (N=32,Cin=1,Cout=8,H=W=32,K=3,S=2,P=1)": {"N": 32, "Cin": 1, "Cout": 8, "H": 32, "W": 32, "K": 3, "S": 2, "P": 1},
}


class _RawDInputLowcin:
    def __init__(self, cfg: "dict[str, int]"):
        self.backend = get_cuda_backend()
        self.lib = self.backend._lib
        N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
        S, P = cfg["S"], cfg["P"]
        Hout, Wout = _hout_wout(H, K, S, P), _hout_wout(W, K, S, P)
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
        self.fn_production = getattr(self.lib, "cf_conv2d_backward_input_f32")  # live dispatch (includes M48)
        self.fn_channelfused_forced = getattr(self.lib, "cf_conv2d_backward_input_channelfused_f32")  # pre-M48 baseline
        self.fn_lowcin1_forced = getattr(self.lib, "cf_conv2d_backward_input_channelfused_lowcin1_f32")

    def production(self, sh) -> int:
        return self.fn_production(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, sh)

    def channelfused_forced(self, sh) -> int:
        return self.fn_channelfused_forced(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, sh)

    def lowcin1_forced(self, sh) -> int:
        return self.fn_lowcin1_forced(self.grad_out.ptr, self.w.ptr, self.grad_x_ptr, *self.shape_args, sh)


def _interleaved_multi_time(calls: "dict[str, object]", iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, dict]":
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
            "mean_ms": statistics.mean(vals), "median_ms": statistics.median(vals),
            "min_ms": min(vals), "max_ms": max(vals),
            "stdev_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
        for name, vals in samples.items()
    }


def _cin1_ab_sweep() -> list:
    results = []
    for label, cfg in _CIN1_AB_SHAPES.items():
        _check_fits_in_vram(cfg)
        raw = _RawDInputLowcin(cfg)
        timed = _interleaved_multi_time({
            "production (lowcin1)": raw.production,
            "pre-M48 baseline (channelfused, forced)": raw.channelfused_forced,
        })
        prod_ms = timed["production (lowcin1)"]["mean_ms"]
        base_ms = timed["pre-M48 baseline (channelfused, forced)"]["mean_ms"]
        results.append({
            "label": label, "config": cfg, "timed_ms": timed,
            "production_ms": prod_ms, "pre_m48_baseline_ms": base_ms,
            "speedup": base_ms / prod_ms,
        })
    return results


_CIN2_REGRESSION_SHAPE = {"N": 32, "Cin": 2, "Cout": 8, "H": 28, "W": 28, "K": 3, "S": 1, "P": 1}


def _cin2_no_regression_check() -> dict:
    """Confirms M48 did not disturb the Cin=2 (still M36 channel-fused) path --
    production dispatch is byte-for-byte unchanged there."""
    _check_fits_in_vram(_CIN2_REGRESSION_SHAPE)
    raw = _RawConv2dBackward(_CIN2_REGRESSION_SHAPE)
    from .m42_bottleneck_recharacterization import _time_phase
    d_input = _time_phase(lambda: raw.backward_input(None))
    return {"config": _CIN2_REGRESSION_SHAPE, "d_input_ms": d_input}


# -- End-to-end conv2d_backward() delta (Section 10: real, not just Amdahl) --


def _end_to_end_backward_delta(ceilings: "roofline.Ceilings") -> dict:
    """Real `_profile_shape_full` at mnist_conv1/mnist_conv2 (reflects the
    NEW, live production dispatch) vs. a same-session reconstruction of the
    pre-M48 total (same dWeight/dBias isolated timings -- untouched by this
    milestone -- with mnist_conv1's dInput isolated timing swapped for the
    forced pre-M48 baseline kernel instead of the live production call).
    Also derives a same-session pre-/post-M48 dInput step-fraction estimate
    (Section 13 of M48's own brief), mirroring M44/M47's own `_frac` method,
    rather than only carrying M47's own prior-session fraction forward."""
    cfg1 = SHAPES["mnist_conv1"]
    cfg2 = SHAPES["mnist_conv2"]
    new_full1 = _profile_shape_full("mnist_conv1", cfg1, ceilings)
    new_full2 = _profile_shape_full("mnist_conv2", cfg2, ceilings)  # Cin=16, unaffected by M48

    raw = _RawDInputLowcin(cfg1)
    from .m42_bottleneck_recharacterization import _time_phase
    old_d_input1 = _time_phase(lambda: raw.channelfused_forced(None))

    new_backward_total_ms = new_full1["backward_total_ms"]
    old_backward_total_ms = old_d_input1["mean_ms"] + (new_backward_total_ms - new_full1["d_input_ms"]["mean_ms"])

    new_din_share = new_full1["d_input_ms"]["mean_ms"] + new_full2["d_input_ms"]["mean_ms"]
    old_din_share = old_d_input1["mean_ms"] + new_full2["d_input_ms"]["mean_ms"]
    dw_share = new_full1["d_weight_ms"]["mean_ms"] + new_full2["d_weight_ms"]["mean_ms"]
    db_share = new_full1["d_bias_ms"]["mean_ms"] + new_full2["d_bias_ms"]["mean_ms"]
    conv_bwd_total_new = new_din_share + dw_share + db_share
    conv_bwd_total_old = old_din_share + dw_share + db_share

    return {
        "config": cfg1,
        "new_d_input_ms": new_full1["d_input_ms"]["mean_ms"],
        "old_d_input_ms": old_d_input1["mean_ms"],
        "d_weight_ms": new_full1["d_weight_ms"]["mean_ms"],
        "d_bias_ms": new_full1["d_bias_ms"]["mean_ms"],
        "new_backward_total_ms": new_backward_total_ms,
        "old_backward_total_ms": old_backward_total_ms,
        "backward_total_speedup": old_backward_total_ms / new_backward_total_ms,
        "new_forward_ms": new_full1["forward_ms"]["mean_ms"],
        # same-session (mnist_conv1 + mnist_conv2) share reconstruction, used
        # only to derive a fresh dInput-of-conv2d-backward RATIO -- combined
        # with the fresh whole-step `conv_bwd_frac_mean` (Phase 8) below to
        # estimate the dInput-of-full-step fraction before/after M48.
        "din_share_of_conv_bwd_new": new_din_share / conv_bwd_total_new,
        "din_share_of_conv_bwd_old": old_din_share / conv_bwd_total_old,
    }


# -- MNIST fresh step-fraction measurement (post-M48, live dispatch) --------


def _mnist_fraction_trials(ceilings: "roofline.Ceilings", n_trials: int = 5) -> dict:
    """Averages `dInput`'s fraction of the full training step across several
    fresh trials -- the M47 thermal-drift lesson (single draws vary by
    several percentage points on this hardware)."""
    din_fracs = []
    conv_bwd_fracs = []
    for _ in range(n_trials):
        profile = m35_mnist._run(ceilings)
        ranking = profile["kernel_ranking"]
        total_s = sum(r["mean_seconds"] for r in ranking)
        by_op = {r["op"]: r["mean_seconds"] for r in ranking}
        conv_bwd_s = by_op.get("backward:conv2d", 0.0)
        if total_s <= 0 or conv_bwd_s <= 0:
            continue
        conv_bwd_fracs.append(conv_bwd_s / total_s)
    return {
        "n_trials": len(conv_bwd_fracs),
        "conv_bwd_frac_mean": statistics.mean(conv_bwd_fracs) if conv_bwd_fracs else 0.0,
        "conv_bwd_frac_range": (min(conv_bwd_fracs), max(conv_bwd_fracs)) if conv_bwd_fracs else (0.0, 0.0),
    }


def _run() -> dict:
    print("-- Phase 0: fresh hardware ceilings (M35 methodology, this session) --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s, "
          f"bandwidth ceiling: {ceilings.bandwidth_gbps:.2f} GB/s")

    print("-- Phase 1: dispatch verification (Cin<=1 -> lowcin1, Cin<=16 -> channelfused, else fallback) --")
    dispatch_mnist_conv1 = _dinput_dispatch_decision(SHAPES["mnist_conv1"])
    dispatch_mnist_conv2 = _dinput_dispatch_decision(SHAPES["mnist_conv2"])

    print("-- Phase 2: nvcc -Xptxas -v (channelfused vs. lowcin1) --")
    ptxas = _run_nvcc_ptxas_verbose()

    print("-- Phase 3: real occupancy query (channelfused vs. lowcin1) --")
    occupancy = _occupancy_query()

    print("-- Phase 4: fresh dInput Cin sweep (post-M48 live dispatch) --")
    dinput_cin_sweep = _dinput_cin_sweep(ceilings)
    # `_dinput_cin_sweep` (m44_bottleneck_recharacterization) times the live,
    # real production kernel call correctly, but its own `dispatch.path`
    # label predates M48 and does not know about the new Cin<=1 band --
    # relabel with M48's own dispatch decision so the saved JSON/report text
    # is not stale (the *measured* d_input_ms values are unaffected either way).
    for r in dinput_cin_sweep:
        r["dispatch"] = _dinput_dispatch_decision({"Cin": r["Cin"]})

    print("-- Phase 5: isolated interleaved A/B (production vs. pre-M48 baseline) at Cin=1 shapes --")
    cin1_ab = _cin1_ab_sweep()

    print("-- Phase 6: Cin=2 no-regression check (path unaffected by M48) --")
    cin2_check = _cin2_no_regression_check()

    print("-- Phase 7: end-to-end conv2d_backward() delta at mnist_conv1 --")
    end_to_end = _end_to_end_backward_delta(ceilings)

    print("-- Phase 8: fresh MNIST step-fraction measurement (5 trials, post-M48) --")
    mnist_fractions = _mnist_fraction_trials(ceilings, n_trials=5)

    # -- Amdahl: two independent fraction estimates for the "before" input --
    # (1) M47's own last-measured fraction (11.75%, prior session, carried
    #     forward since production dispatch can no longer be reverted
    #     in-process to re-derive it identically);
    # (2) a fresh, same-session reconstruction: this session's own measured
    #     whole-step `conv2d backward` fraction (Phase 8, averaged over 5
    #     trials) times this session's own dInput-of-conv2d-backward RATIO
    #     (Phase 7, before/after M48) -- triangulates M47's carried-over
    #     figure against fresh evidence rather than trusting it blindly.
    pre_m48_dinput_fraction_m47 = 0.1175  # M47 Section 13
    conv_bwd_frac = mnist_fractions["conv_bwd_frac_mean"]
    fresh_din_fraction_old = conv_bwd_frac * end_to_end["din_share_of_conv_bwd_old"]
    fresh_din_fraction_new = conv_bwd_frac * end_to_end["din_share_of_conv_bwd_new"]

    measured_speedups = sorted(set(round(r["speedup"], 2) for r in cin1_ab))
    amdahl_measured = _amdahl(
        {
            "dInput (pre-M48 fraction, M47 carried-over)": pre_m48_dinput_fraction_m47,
            "dInput (pre-M48 fraction, fresh this-session reconstruction)": fresh_din_fraction_old,
        },
        speedups=tuple(measured_speedups),
    )
    amdahl_hypothetical = _amdahl({"dInput (pre-M48 fraction, M47 carried-over)": pre_m48_dinput_fraction_m47})

    return {
        "hardware_ceilings": hw_profile,
        "real_workload_relevance": REAL_WORKLOAD_RELEVANCE,
        "dispatch": {"mnist_conv1": dispatch_mnist_conv1, "mnist_conv2": dispatch_mnist_conv2},
        "ptxas": ptxas,
        "occupancy": occupancy,
        "dinput_cin_sweep": dinput_cin_sweep,
        "cin1_ab_sweep": cin1_ab,
        "cin2_no_regression_check": cin2_check,
        "end_to_end_backward_delta": end_to_end,
        "mnist_fractions": mnist_fractions,
        "amdahl_pre_m48_fraction_m47_carried_over": pre_m48_dinput_fraction_m47,
        "fresh_dinput_fraction_old": fresh_din_fraction_old,
        "fresh_dinput_fraction_new": fresh_din_fraction_new,
        "amdahl_at_measured_speedups": amdahl_measured,
        "amdahl_at_hypothetical_speedups": amdahl_hypothetical,
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M48 dInput low-Cin value assessment (940MX, real CUDA) ===", ""]
    lines.append("-- Dispatch verification --")
    lines.append(f"  mnist_conv1 (Cin=1): {profile['dispatch']['mnist_conv1']['path']}")
    lines.append(f"  mnist_conv2 (Cin=16): {profile['dispatch']['mnist_conv2']['path']}")
    lines.append("")
    lines.append("-- Occupancy (real cudaOccupancyMaxActiveBlocksPerMultiprocessor) --")
    occ = profile["occupancy"]
    lines.append(f"  channelfused: {occ['max_active_blocks_per_sm_channelfused']} blocks/SM "
                 f"({occ['occupancy_fraction_channelfused']*100:.1f}%)")
    lines.append(f"  lowcin1:      {occ['max_active_blocks_per_sm_lowcin1']} blocks/SM "
                 f"({occ['occupancy_fraction_lowcin1']*100:.1f}%)")
    lines.append("")
    lines.append("-- dInput Cin sweep (post-M48 live dispatch) --")
    for r in profile["dinput_cin_sweep"]:
        lines.append(f"  Cin={r['Cin']:>3}  {r['dispatch']['path']:<60}  {r['d_input_ms']['mean_ms']:.4f}ms  "
                     f"{r['roofline']['fraction_of_compute_ceiling']*100:5.1f}% of ceiling")
    lines.append("")
    lines.append("-- Isolated interleaved A/B: production (lowcin1) vs. pre-M48 baseline --")
    for r in profile["cin1_ab_sweep"]:
        lines.append(f"  {r['label']}: production={r['production_ms']:.4f}ms "
                     f"pre-M48={r['pre_m48_baseline_ms']:.4f}ms speedup={r['speedup']:.3f}x")
    lines.append("")
    e2e = profile["end_to_end_backward_delta"]
    lines.append("-- End-to-end conv2d_backward() delta at mnist_conv1 --")
    lines.append(f"  old backward_total={e2e['old_backward_total_ms']:.4f}ms  "
                 f"new backward_total={e2e['new_backward_total_ms']:.4f}ms  "
                 f"speedup={e2e['backward_total_speedup']:.3f}x")
    lines.append("")
    lines.append(f"-- Fresh this-session dInput fraction reconstruction (conv_bwd_frac_mean="
                 f"{profile['mnist_fractions']['conv_bwd_frac_mean']*100:.1f}%, "
                 f"n={profile['mnist_fractions']['n_trials']} trials) --")
    lines.append(f"  pre-M48 (reconstructed): {profile['fresh_dinput_fraction_old']*100:.2f}% of full step")
    lines.append(f"  post-M48 (measured):     {profile['fresh_dinput_fraction_new']*100:.2f}% of full step")
    lines.append("")
    lines.append("-- Amdahl at the REAL measured kernel-level speedups --")
    for name, proj in profile["amdahl_at_measured_speedups"].items():
        lines.append(f"  {name}: " + "  ".join(f"{k}->{v:.4f}x" for k, v in proj.items()))
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m48_dinput_value_assessment.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m48_dinput_value_assessment requires real CUDA hardware.")
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
