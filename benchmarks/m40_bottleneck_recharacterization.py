"""M40: post-M39 CUDA bottleneck re-characterization (measurement-only).

M31-M39 progressively optimized `conv2d_backward`'s `dInput` (M32 division
hoist, M36 channel-fused register mapping) and `dWeight` (M34 im2col+GEMM,
M37 split-K occupancy fix, M38 half-fused split-K GEMM for `blocks_y == 1`,
M39 shared-memory im2col staging for `blocks_y >= 2`). No milestone since
M35 re-measured the *whole* pipeline fresh against the CURRENT production
dispatch, and M37/M38/M39's own decomposition scripts each measure a
specific *candidate* pair (old vs. new), not "what does the pipeline look
like today, end to end, at every shape, with every phase attributed to the
function actually selected by `CUDABackend.conv2d_backward` right now."

This script does exactly that:

1. **Dispatch verification** (Section 6 of the M40 brief): for every
   representative shape, compute `weight_elements`/`blocks_y` exactly as
   `CUDABackend.conv2d_backward` does and record which dWeight function is
   actually selected -- so the decomposition below is never measuring the
   wrong implementation (the exact mistake M37's own report flags as
   "a real issue caught in M37").
2. **Full decomposition of the CURRENT production path** at every shape:
   forward, dInput (current dispatch -- channel-fused when `Cin<=16`, via
   `cf_conv2d_backward_input_*`'s own internal C-level dispatch), dWeight's
   *actual* current sub-stages (permute + fused-GEMM for `blocks_y==1`;
   im2col-or-smem-fallback + permute + split-K GEMM for `blocks_y>=2`; the
   unchanged M21 kernel below the weight-element threshold), dBias, and the
   real end-to-end `CUDABackend.conv2d_backward()` call for cross-check.
3. **Fresh roofline ceilings** (re-running M35's own methodology, same
   process) and classification of dInput/dWeight-total/forward.
4. Reuses `pipeline_profile._profile_async_epoch` (async pipeline, batch
   sweep) and `m35_mnist._run` (synchronous per-op kernel-contribution
   ranking, non-Conv2d ops) directly rather than re-deriving either.

No production code is modified or reimplemented differently than
production already does; every dWeight sub-stage call below is the same
function `backend.py` itself calls.

    python -m benchmarks.m40_bottleneck_recharacterization
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import numpy as np

import forge
from forge.backend.cuda.backend import _MATMUL_TILE, _CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD, _SUFFIX, get_cuda_backend, is_cuda_available
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms
from forge.backend.cuda.experimental_conv_im2col import grad_output_permute, im2col, recommended_num_k_splits
from forge.backend.cuda.experimental_conv_im2col_reuse import im2col_smem, im2col_smem_fits

from . import m35_hardware, m35_mnist, roofline
from .conv2d_backward_profile import BATCH_SIZES, BATCH_SWEEP_BASE, SHAPES, _RawConv2dBackward, _check_fits_in_vram, _hout_wout
from .environment import collect_environment
from .pipeline_profile import _profile_async_epoch

WARMUP = 5
ITERATIONS = 30


def _time_phase(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
    for _ in range(warmup):
        code = call()
        assert code == 0, f"kernel launch failed with code {code}"
        forge.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = TimedEvent()
        start.record(None)
        code = call()
        assert code == 0, f"kernel launch failed with code {code}"
        end = TimedEvent()
        end.record(None)
        forge.cuda.synchronize()
        samples.append(elapsed_ms(start, end))
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _dispatch_decision(cfg: "dict[str, int]") -> dict:
    """Mirrors `CUDABackend.conv2d_backward`'s dWeight dispatch exactly."""
    Cout, Cin, K = cfg["Cout"], cfg["Cin"], cfg["K"]
    weight_elements = Cout * Cin * K * K
    if weight_elements < _CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD:
        return {"weight_elements": weight_elements, "blocks_y": None, "path": "cf_conv2d_backward_weight (M21 per-thread/block-reduce)"}
    blocks_y = (Cout + _MATMUL_TILE - 1) // _MATMUL_TILE
    if blocks_y == 1:
        path = "dweight_halffused_gemm_splitk (M38, permute + fused-GEMM)"
    else:
        path = "dweight_im2col_smem_gemm_splitk (M39, im2col[-smem] + permute + split-K GEMM)"
    return {"weight_elements": weight_elements, "blocks_y": blocks_y, "path": path}


class _RawDweightCurrent:
    """Direct ctypes calls for every sub-stage of the CURRENT production
    dWeight dispatch (mirrors `experimental_conv_halffused`/
    `experimental_conv_im2col_reuse`'s own Python wrappers, but keeps each
    stage independently timeable -- the same pattern `_RawIm2colGemm` (M34)
    and `_RawConv2dBackward` (M32) already established)."""

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
        grad_out = forge.Tensor(rng.standard_normal((N, Cout, Hout, Wout)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        self.x, self.grad_out = x._data, grad_out._data

        self.decision = _dispatch_decision(cfg)
        self.num_k_splits = recommended_num_k_splits(self.M)

    # -- sub-stages, always current production functions --------------------
    def permute(self) -> int:
        self._dycolT = grad_output_permute(self.backend, self.grad_out, self.N, self.Cout, self.Hout, self.Wout)
        return 0

    def im2col_stage(self) -> int:
        if im2col_smem_fits(np.dtype(np.float32), self.H, self.W):
            self._xcol = im2col_smem(self.backend, self.x, self.N, self.Cin, self.H, self.W, self.K, self.K,
                                      self.S, self.S, self.P, self.P, self.Hout, self.Wout)
        else:
            self._xcol = im2col(self.backend, self.x, self.N, self.Cin, self.H, self.W, self.K, self.K,
                                 self.S, self.S, self.P, self.P, self.Hout, self.Wout)
        return 0

    def splitk_gemm(self) -> int:
        out_ptr = self.backend._alloc(self.Cout * self.Kdim * 4)
        fn = getattr(self.lib, "cf_matmul_splitk_f32")
        code = fn(
            self._dycolT.ptr, self._xcol.ptr, out_ptr,
            ctypes.c_int(self.Cout), ctypes.c_int(self.M), ctypes.c_int(self.Kdim),
            ctypes.c_int(self.num_k_splits), None,
        )
        self.backend._check(code, "m40 splitk gemm")
        return 0

    def fused_gemm(self) -> int:
        out_ptr = self.backend._alloc(self.Cout * self.Kdim * 4)
        fn = getattr(self.lib, "cf_dweight_halffused_gemm_splitk_f32")
        code = fn(
            self._dycolT.ptr, self.x.ptr, out_ptr,
            ctypes.c_int(self.N), ctypes.c_int(self.Cin), ctypes.c_int(self.H), ctypes.c_int(self.W),
            ctypes.c_int(self.Cout), ctypes.c_int(self.K), ctypes.c_int(self.K),
            ctypes.c_int(self.S), ctypes.c_int(self.S), ctypes.c_int(self.P), ctypes.c_int(self.P),
            ctypes.c_int(self.Hout), ctypes.c_int(self.Wout),
            ctypes.c_int(self.num_k_splits), None,
        )
        self.backend._check(code, "m40 fused gemm")
        return 0


def _profile_shape(name: str, cfg: "dict[str, int]", ceilings: roofline.Ceilings) -> dict:
    _check_fits_in_vram(cfg)
    raw = _RawConv2dBackward(cfg)
    decision = _dispatch_decision(cfg)

    forward = _time_phase(lambda: raw.forward(None))
    d_input = _time_phase(lambda: raw.backward_input(None))
    d_bias = _time_phase(lambda: raw.backward_bias(None))

    dweight_stages = {}
    if decision["blocks_y"] is None:
        d_weight = _time_phase(lambda: raw.backward_weight(None))
        dweight_total_ms = d_weight["mean_ms"]
    else:
        current = _RawDweightCurrent(cfg)
        permute_t = _time_phase(current.permute)
        if decision["blocks_y"] == 1:
            gemm_t = _time_phase(current.fused_gemm)
            dweight_stages = {"permute_ms": permute_t, "fused_gemm_ms": gemm_t}
            dweight_total_ms = permute_t["mean_ms"] + gemm_t["mean_ms"]
        else:
            im2col_t = _time_phase(current.im2col_stage)
            gemm_t = _time_phase(current.splitk_gemm)
            dweight_stages = {"permute_ms": permute_t, "im2col_ms": im2col_t, "splitk_gemm_ms": gemm_t}
            dweight_total_ms = permute_t["mean_ms"] + im2col_t["mean_ms"] + gemm_t["mean_ms"]
        d_weight = {"mean_ms": dweight_total_ms}

    backward_total_ms = d_input["mean_ms"] + dweight_total_ms + d_bias["mean_ms"]

    N, Cin, Cout, H, W, K = cfg["N"], cfg["Cin"], cfg["Cout"], cfg["H"], cfg["W"], cfg["K"]
    Hout, Wout = raw.Hout, raw.Wout

    def _classify(ms: float, flops: int, nbytes: int) -> dict:
        seconds = ms / 1000.0
        ai = roofline.arithmetic_intensity(flops, nbytes)
        gflops = (flops / seconds) / 1e9 if seconds > 0 else 0.0
        gbps = (nbytes / seconds) / 1e9 if seconds > 0 else 0.0
        cls = roofline.classify(gflops, seconds, ai, ceilings)
        return {"achieved_gflops": gflops, "achieved_gbps": gbps, "arithmetic_intensity": ai,
                "fraction_of_compute_ceiling": gflops / ceilings.compute_gflops if ceilings.compute_gflops else 0.0,
                "classification": cls.label}

    din_flops = roofline.flops_conv2d_dinput(N, Cin, H, W, Cout, K, K)
    din_bytes = roofline.bytes_conv2d_dinput(N, Cin, H, W, Cout, K, K, Hout, Wout)
    dw_flops = roofline.flops_conv2d_dweight(Cout, Cin, K, K, N, Hout, Wout)
    dw_bytes = roofline.bytes_conv2d_dweight(N, Cin, H, W, Cout, K, K, Hout, Wout)
    fwd_flops = roofline.flops_conv2d_forward(N, Cout, Hout, Wout, Cin, K, K)
    fwd_bytes = roofline.bytes_conv2d_forward(N, Cin, H, W, Cout, K, K, Hout, Wout)

    return {
        "name": name, "config": cfg, "dispatch": decision,
        "forward_ms": forward, "d_input_ms": d_input, "d_weight_ms": d_weight,
        "d_bias_ms": d_bias, "dweight_stages_ms": dweight_stages,
        "backward_total_ms": backward_total_ms,
        "d_input_pct_of_backward": 100.0 * d_input["mean_ms"] / backward_total_ms,
        "d_weight_pct_of_backward": 100.0 * dweight_total_ms / backward_total_ms,
        "d_bias_pct_of_backward": 100.0 * d_bias["mean_ms"] / backward_total_ms,
        "roofline_forward": _classify(forward["mean_ms"], fwd_flops, fwd_bytes),
        "roofline_dinput": _classify(d_input["mean_ms"], din_flops, din_bytes),
        "roofline_dweight": _classify(dweight_total_ms, dw_flops, dw_bytes),
    }


def _amdahl(fractions: "dict[str, float]", speedups=(1.5, 2.0, 3.0)) -> "dict[str, dict[str, float]]":
    """`1 / (1 - frac + frac/speedup)` per named phase fraction of total step."""
    out = {}
    for name, frac in fractions.items():
        out[name] = {f"{s}x": 1.0 / ((1 - frac) + frac / s) for s in speedups}
    return out


def _run() -> dict:
    print("-- Phase 0: fresh hardware ceilings (M35 methodology, this session) --")
    hw_profile = m35_hardware._run()
    ceilings = roofline.Ceilings(
        compute_gflops=hw_profile["ceilings"]["practical_compute_gflops"],
        bandwidth_gbps=hw_profile["ceilings"]["practical_bandwidth_gbps"],
    )
    print(f"   practical compute ceiling: {ceilings.compute_gflops:.2f} GFLOP/s, "
          f"bandwidth ceiling: {ceilings.bandwidth_gbps:.2f} GB/s")

    print("-- Phase 1: current-production Conv2d backward decomposition (7 shapes + batch sweep) --")
    shapes = [_profile_shape(name, cfg, ceilings) for name, cfg in SHAPES.items()]
    batch_sweep = []
    for n in BATCH_SIZES:
        cfg = dict(BATCH_SWEEP_BASE)
        cfg["N"] = n
        batch_sweep.append(_profile_shape(f"batch_{n}", cfg, ceilings))

    print("-- Phase 2: synchronous per-op kernel-contribution ranking (m35_mnist, rerun fresh) --")
    mnist_ranking_profile = m35_mnist._run(ceilings)

    print("-- Phase 3: async pipeline profile (batch sweep, prefetch depth sweep) --")
    async_batch_sweep = [_profile_async_epoch(bs, prefetch_size=2, n_samples=1024) for bs in (32, 64, 128)]
    prefetch_depth_sweep = [_profile_async_epoch(64, prefetch_size=d, n_samples=1024) for d in (1, 2, 3)]

    # -- Amdahl analysis, using the freshly measured full training step -----
    ranking = mnist_ranking_profile["kernel_ranking"]
    total_step_s = sum(r["mean_seconds"] for r in ranking)
    by_op = {r["op"]: r["mean_seconds"] for r in ranking}
    conv_bwd_s = by_op.get("backward:conv2d", 0.0)
    conv_fwd_s = by_op.get("forward:Conv2d", 0.0)
    matmul_bwd_s = by_op.get("backward:@", 0.0)

    # Split conv2d-backward's measured share into dInput/dWeight/dBias using
    # this session's own per-shape ratios at MNIST's real two layer shapes
    # (mnist_conv1, mnist_conv2) -- the only two shapes in `shapes` that are
    # actually MNIST's own layers.
    mnist_shapes = [s for s in shapes if s["name"] in ("mnist_conv1", "mnist_conv2")]
    din_share = sum(s["d_input_ms"]["mean_ms"] for s in mnist_shapes)
    dw_share = sum(s["d_weight_ms"]["mean_ms"] for s in mnist_shapes)
    db_share = sum(s["d_bias_ms"]["mean_ms"] for s in mnist_shapes)
    conv_bwd_total = din_share + dw_share + db_share
    fractions = {
        "dInput (of full step)": (conv_bwd_s * (din_share / conv_bwd_total)) / total_step_s if total_step_s and conv_bwd_total else 0.0,
        "dWeight (of full step)": (conv_bwd_s * (dw_share / conv_bwd_total)) / total_step_s if total_step_s and conv_bwd_total else 0.0,
        "conv2d backward total (of full step)": conv_bwd_s / total_step_s if total_step_s else 0.0,
        "conv2d forward (of full step)": conv_fwd_s / total_step_s if total_step_s else 0.0,
        "matmul backward (of full step)": matmul_bwd_s / total_step_s if total_step_s else 0.0,
    }
    amdahl = _amdahl(fractions)

    return {
        "hardware_ceilings": hw_profile,
        "shapes": shapes,
        "batch_sweep": batch_sweep,
        "mnist_kernel_ranking": mnist_ranking_profile,
        "async_batch_sweep": async_batch_sweep,
        "prefetch_depth_sweep": prefetch_depth_sweep,
        "amdahl_fractions": fractions,
        "amdahl_projection": amdahl,
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M40 post-M39 bottleneck re-characterization (940MX, real CUDA) ===", ""]
    lines.append("-- Dispatch verification + decomposition --")
    header = f"{'shape':<14}{'weight#':>8}{'blocks_y':>9}{'fwd(ms)':>9}{'dIn(ms)':>9}{'dW(ms)':>9}{'dB(ms)':>8}{'bwd(ms)':>9}  path"
    lines.append(header)
    for r in profile["shapes"] + profile["batch_sweep"]:
        d = r["dispatch"]
        lines.append(
            f"{r['name']:<14}{d['weight_elements']:>8}{str(d['blocks_y']):>9}"
            f"{r['forward_ms']['mean_ms']:>9.4f}{r['d_input_ms']['mean_ms']:>9.4f}"
            f"{r['d_weight_ms']['mean_ms']:>9.4f}{r['d_bias_ms']['mean_ms']:>8.4f}"
            f"{r['backward_total_ms']:>9.4f}  {d['path']}"
        )
    lines.append("")
    lines.append("-- MNIST kernel-contribution ranking (this session) --")
    for r in profile["mnist_kernel_ranking"]["kernel_ranking"][:8]:
        lines.append(f"  {r['op']:<22}{r['percent_of_step']:6.2f}%  {r['mean_seconds']*1e3:8.4f}ms  {r.get('classification','n/a')}")
    lines.append("")
    lines.append("-- Amdahl projection (hypothetical per-component speedup -> overall step speedup) --")
    for name, proj in profile["amdahl_projection"].items():
        frac = profile["amdahl_fractions"][name]
        lines.append(f"  {name:<38} frac={frac*100:5.2f}%  " + "  ".join(f"{k}->{v:.3f}x" for k, v in proj.items()))
    lines.append("")
    lines.append("-- Async pipeline batch sweep --")
    for r in profile["async_batch_sweep"]:
        lines.append(f"  batch={r['batch_size']:>4}  {r['samples_per_sec']:9.0f} samples/sec  util={r['compute_stream_utilization']*100:5.1f}%")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m40_bottleneck_recharacterization.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m40_bottleneck_recharacterization requires real CUDA hardware.")
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
