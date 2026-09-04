"""Hardware characterization + practical compute/bandwidth ceilings (Milestone 35).

    python -m benchmarks.m35_hardware

Section 5 (hardware characterization) and Section 6 (practical hardware
ceilings) of the M35 brief. Two distinct kinds of number are produced and
kept clearly labeled apart everywhere they are used (Section 5's "clearly
distinguish theoretical specification from measured practical ceiling"):

- **theoretical**: publicly documented 940MX (GM108, Maxwell) specs --
  3 SMs x 128 CUDA cores/SM = 384 CUDA cores, a 64-bit GDDR5 memory bus --
  combined with this run's own `nvidia-smi`-reported clocks to derive a
  textbook peak FLOP/s and peak bandwidth. Never presented as an achievable
  Forge target.
- **practical**: measured directly, on this process's real CUDA context, by
  timing Forge's own existing kernels at large sizes with the timing-enabled
  `TimedEvent` (`forge.backend.cuda.profiling_events`) -- `cf_matmul_f32`
  (compute-dense, large square GEMM) for the compute ceiling, `cf_add_f32`
  (a 2-read-1-write streaming elementwise op) for the bandwidth ceiling.
  Deliberately reuses Forge's *existing* compiled kernels rather than adding
  new hand-tuned peak-throughput microkernels to `kernels.cu` -- keeping
  this milestone's default of zero production CUDA changes (Section 41)
  while still satisfying Section 6's "simple arithmetic-heavy workload" /
  "sufficiently large streaming workload" requirements. This is documented
  as a limitation (see the M35 report's Limitations section): a dedicated
  hand-written peak microbenchmark could refine these ceilings further.

Writes `benchmarks/results/m35_hardware.json`, consumed by
`benchmarks.roofline.load_ceilings` from every other `m35_*` script.
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
from forge.backend.cuda.profiling_events import TimedEvent, elapsed_ms

from .environment import collect_environment

WARMUP = 5
ITERATIONS = 15

# Square GEMM sizes for the compute-ceiling sweep -- large enough for the
# tiled kernel to reach steady state, small enough to stay well inside the
# 940MX's 2GB VRAM budget (3 * dim^2 * 4 bytes; 2048^2 -> ~50MB total).
_COMPUTE_CEILING_DIMS = (512, 1024, 2048)

# Element count for the bandwidth-ceiling streaming-add sweep: 3 buffers *
# 20,000,000 * 4 bytes = ~240MB, comfortably inside the 940MX's 2GB budget
# and the documented ~256MB single-benchmark safety margin (see
# `conv2d_backward_profile.py`'s `_check_fits_in_vram`).
_BANDWIDTH_CEILING_ELEMENTS = 20_000_000

# Public 940MX (GM108, Maxwell) specification constants -- theoretical only.
_THEORETICAL_SM_COUNT = 3
_THEORETICAL_CUDA_CORES_PER_SM = 128
_THEORETICAL_MEMORY_BUS_BITS = 64


def _nvidia_smi_clocks() -> "dict[str, float | None]":
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            sm_mhz, mem_mhz = (float(v.strip()) for v in result.stdout.strip().splitlines()[0].split(","))
            return {"sm_clock_mhz": sm_mhz, "mem_clock_mhz": mem_mhz}
    except Exception:
        pass
    return {"sm_clock_mhz": None, "mem_clock_mhz": None}


def _theoretical_specs() -> dict:
    clocks = _nvidia_smi_clocks()
    cores = _THEORETICAL_SM_COUNT * _THEORETICAL_CUDA_CORES_PER_SM
    theoretical_gflops = None
    theoretical_gbps = None
    if clocks["sm_clock_mhz"] is not None:
        # 1 FMA (2 FLOPs) per core per cycle -- the standard textbook peak-FLOP formula.
        theoretical_gflops = cores * (clocks["sm_clock_mhz"] * 1e6) * 2 / 1e9
    if clocks["mem_clock_mhz"] is not None:
        # GDDR5 double-data-rate: 2 transfers/cycle, `_THEORETICAL_MEMORY_BUS_BITS`-bit bus.
        theoretical_gbps = (clocks["mem_clock_mhz"] * 1e6) * 2 * (_THEORETICAL_MEMORY_BUS_BITS / 8) / 1e9
    return {
        "gpu": "NVIDIA GeForce 940MX (GM108, Maxwell) -- public specification, not runtime-queried",
        "sm_count": _THEORETICAL_SM_COUNT,
        "cuda_cores_per_sm": _THEORETICAL_CUDA_CORES_PER_SM,
        "cuda_cores_total": cores,
        "memory_bus_bits": _THEORETICAL_MEMORY_BUS_BITS,
        "sm_clock_mhz": clocks["sm_clock_mhz"],
        "mem_clock_mhz": clocks["mem_clock_mhz"],
        "theoretical_fp32_gflops": theoretical_gflops,
        "theoretical_memory_gbps": theoretical_gbps,
        "note": "Derived from public GM108 core-count specs + this run's nvidia-smi max clocks. "
                "Not an achievable Forge target -- see practical_ceilings for measured numbers.",
    }


def _time_raw(call, iterations: int = ITERATIONS, warmup: int = WARMUP) -> "dict[str, float]":
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
        samples.append(elapsed_ms(start, end) / 1000.0)  # seconds

    return {
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _measure_compute_ceiling() -> dict:
    backend = get_cuda_backend()
    lib = backend._lib
    suffix = _SUFFIX[np.dtype(np.float32)]
    fn = getattr(lib, f"cf_matmul_{suffix}")

    sweep = []
    for dim in _COMPUTE_CEILING_DIMS:
        rng = np.random.default_rng(0)
        a = forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device="cuda")
        b = forge.Tensor(rng.standard_normal((dim, dim)).astype(np.float32), device="cuda")
        forge.cuda.synchronize()
        out_ptr = backend._alloc(dim * dim * 4)
        args = (a._data.ptr, b._data.ptr, out_ptr, ctypes.c_int(dim), ctypes.c_int(dim), ctypes.c_int(dim))

        timing = _time_raw(lambda: fn(*args, None))
        flops = 2 * dim * dim * dim
        gflops = (flops / 1e9) / timing["mean_s"]
        sweep.append({"dim": dim, "timing": timing, "achieved_gflops": gflops})
        forge.cuda.empty_cache()

    best = max(sweep, key=lambda r: r["achieved_gflops"])
    return {"sweep": sweep, "practical_compute_gflops": best["achieved_gflops"], "best_dim": best["dim"]}


def _measure_bandwidth_ceiling() -> dict:
    backend = get_cuda_backend()
    lib = backend._lib
    suffix = _SUFFIX[np.dtype(np.float32)]
    fn = getattr(lib, f"cf_add_{suffix}")

    n = _BANDWIDTH_CEILING_ELEMENTS
    rng = np.random.default_rng(0)
    a = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
    b = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
    forge.cuda.synchronize()
    out_ptr = backend._alloc(n * 4)
    args = (a._data.ptr, b._data.ptr, out_ptr, ctypes.c_longlong(n))

    timing = _time_raw(lambda: fn(*args, None))
    nbytes = 3 * n * 4  # read a, read b, write out
    gbps = (nbytes / 1e9) / timing["mean_s"]
    forge.cuda.empty_cache()

    return {"elements": n, "bytes_moved": nbytes, "timing": timing, "practical_bandwidth_gbps": gbps}


def _run() -> dict:
    theoretical = _theoretical_specs()
    compute = _measure_compute_ceiling()
    bandwidth = _measure_bandwidth_ceiling()
    return {
        "theoretical_specs": theoretical,
        "compute_ceiling_sweep": compute,
        "bandwidth_ceiling_measurement": bandwidth,
        "ceilings": {
            "practical_compute_gflops": compute["practical_compute_gflops"],
            "practical_bandwidth_gbps": bandwidth["practical_bandwidth_gbps"],
        },
    }


def _render_report(profile: dict) -> str:
    lines = ["=== M35 hardware characterization + practical ceilings (940MX, real CUDA) ===", ""]
    t = profile["theoretical_specs"]
    lines.append("-- Theoretical specification (public GM108 spec + measured clocks) --")
    lines.append(f"  SMs={t['sm_count']}  cores/SM={t['cuda_cores_per_sm']}  total cores={t['cuda_cores_total']}")
    lines.append(f"  sm_clock={t['sm_clock_mhz']}MHz  mem_clock={t['mem_clock_mhz']}MHz  bus={t['memory_bus_bits']}-bit")
    lines.append(f"  theoretical FP32: {t['theoretical_fp32_gflops']:.1f} GFLOP/s" if t['theoretical_fp32_gflops'] else "  theoretical FP32: unavailable")
    lines.append(f"  theoretical bandwidth: {t['theoretical_memory_gbps']:.2f} GB/s" if t['theoretical_memory_gbps'] else "  theoretical bandwidth: unavailable")
    lines.append("")
    lines.append("-- Practical compute ceiling (cf_matmul_f32, square GEMM sweep) --")
    for r in profile["compute_ceiling_sweep"]["sweep"]:
        lines.append(f"  dim={r['dim']:>5}  {r['timing']['mean_s']*1000:9.4f}ms  {r['achieved_gflops']:8.2f} GFLOP/s")
    lines.append(f"  practical_compute_gflops = {profile['ceilings']['practical_compute_gflops']:.2f} GFLOP/s")
    lines.append("")
    bw = profile["bandwidth_ceiling_measurement"]
    lines.append("-- Practical bandwidth ceiling (cf_add_f32, large streaming add) --")
    lines.append(f"  n={bw['elements']:,}  {bw['timing']['mean_s']*1000:9.4f}ms  {bw['practical_bandwidth_gbps']:8.3f} GB/s")
    lines.append("")
    lines.append("These are Forge's own practical benchmark ceilings for this environment, not a claim")
    lines.append("that every possible CUDA workload could reach them (Section 6).")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m35_hardware.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m35_hardware requires real CUDA hardware.")
        return

    profile = _run()
    print(_render_report(profile))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"environment": collect_environment(), **profile}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved profile -> {output_path}")


if __name__ == "__main__":
    main()
