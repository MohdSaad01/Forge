"""Full MNIST workload characterization (Milestone 35).

    python -m benchmarks.m35_mnist

Covers Sections 24-26 (full MNIST workload, kernel-runtime ranking),
Section 33 (batch-size scaling), and Section 37 (profiling overhead) of
the M35 brief, by reusing the existing M21/M31 MNIST profilers directly
rather than reimplementing them:

- **Synchronous per-layer/per-op breakdown**: `mnist_profile._profile_device
  ("cuda")` (unchanged, batch_size=64 from `MNIST_PROFILE_CONFIG`).
- **Async per-phase GPU-busy breakdown + batch-size scaling**:
  `pipeline_profile._profile_async_epoch(batch_size, ...)`, called directly
  at N=32/64/128 (Section 33) -- this is exactly M31's own batch-size sweep,
  rerun fresh against the post-M34 kernel set.
- **Profiling overhead** (Section 37): times the same fixed number of
  training steps two ways -- a plain, uninstrumented `loss.backward()` loop
  (what `Trainer`/ordinary training code actually runs) vs.
  `mnist_profile._profiled_run_backward`'s per-op-timed walk (what every
  `mnist_profile`/M35 measurement above actually used) -- to verify
  instrumentation itself is not materially altering the numbers reported.

The kernel-contribution ranking (Section 26) applies `roofline.py`'s
FLOP/byte model to the real M20 CNN's known, fixed layer shapes (`examples/
mnist/model.py`: Conv2d(1,8,k=3) -> pool -> Conv2d(8,16,k=3) -> pool ->
Linear(400,64) -> Linear(64,10), batch=64) to turn `mnist_profile`'s
per-op-name wall-clock breakdown into GFLOP/s and a roofline classification
per named op -- aggregated across both Conv2d layers per op name (the
`_profiled_run_backward` walker aggregates by `node.name`, not by layer
identity, matching M31's own "conv2d backward is 55-57% of total time"
aggregate framing).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import forge
import forge.nn as nn
import forge.optim as optim
from forge.backend import get_backend
from forge.backend.cuda.backend import is_cuda_available

from . import roofline as rf
from .environment import collect_environment
from .mnist_profile import _profile_device, _profiled_run_backward, _sync
from .pipeline_profile import _profile_async_epoch
from .sizes import MNIST_PROFILE_CONFIG

BATCH_SIZES = (32, 64, 128)
N_SAMPLES = 1024

# The real M20 CNN's fixed shapes (`examples/mnist/model.py`'s own docstring
# trace, batch=64): used only to attach a FLOP/byte roofline model to
# `mnist_profile`'s per-op-name wall-clock breakdown (Section 26). Both
# Conv2d/Linear/MaxPool2d instances are summed per op name, matching how
# `_profiled_run_backward` itself aggregates (by `node.name`, not identity).
_BATCH = MNIST_PROFILE_CONFIG["batch_size"]
_CONV1 = {"N": _BATCH, "Cin": 1, "Cout": 8, "H": 28, "W": 28, "K": 3, "Hout": 26, "Wout": 26}
_CONV2 = {"N": _BATCH, "Cin": 8, "Cout": 16, "H": 13, "W": 13, "K": 3, "Hout": 11, "Wout": 11}
_POOL1 = {"N": _BATCH, "C": 8, "H": 26, "W": 26, "Hout": 13, "Wout": 13, "K": 2}
_POOL2 = {"N": _BATCH, "C": 16, "H": 11, "W": 11, "Hout": 5, "Wout": 5, "K": 2}
_LINEAR1 = {"M": _BATCH, "K": 400, "N": 64}
_LINEAR2 = {"M": _BATCH, "K": 64, "N": 10}


def _conv_forward_flops_bytes(c):
    return (rf.flops_conv2d_forward(c["N"], c["Cout"], c["Hout"], c["Wout"], c["Cin"], c["K"], c["K"]),
            rf.bytes_conv2d_forward(c["N"], c["Cin"], c["H"], c["W"], c["Cout"], c["K"], c["K"], c["Hout"], c["Wout"]))


def _conv_backward_flops_bytes(c):
    """dInput + dWeight combined (the two kernels `conv2d`'s single backward node triggers)."""
    f = rf.flops_conv2d_dinput(c["N"], c["Cin"], c["H"], c["W"], c["Cout"], c["K"], c["K"])
    f += rf.flops_conv2d_dweight(c["Cout"], c["Cin"], c["K"], c["K"], c["N"], c["Hout"], c["Wout"])
    b = rf.bytes_conv2d_dinput(c["N"], c["Cin"], c["H"], c["W"], c["Cout"], c["K"], c["K"], c["Hout"], c["Wout"])
    b += rf.bytes_conv2d_dweight(c["N"], c["Cin"], c["H"], c["W"], c["Cout"], c["K"], c["K"], c["Hout"], c["Wout"])
    return f, b


def _pool_forward_bytes(p):
    return rf.bytes_maxpool_forward(p["N"], p["C"], p["Hout"], p["Wout"], p["K"], p["K"])


def _pool_backward_bytes(p):
    return rf.bytes_maxpool_backward(p["N"], p["C"], p["H"], p["W"], p["Hout"], p["Wout"])


def _linear_forward_flops_bytes(l):
    return rf.flops_matmul(l["M"], l["N"], l["K"]), rf.bytes_matmul_minimum(l["M"], l["N"], l["K"])


def _forward_model() -> "dict[str, tuple[int, int]]":
    """`class-name -> (flops, bytes)`, matching `forward_by_layer_mean_seconds`'s keys."""
    conv1_f, conv1_b = _conv_forward_flops_bytes(_CONV1)
    conv2_f, conv2_b = _conv_forward_flops_bytes(_CONV2)
    lin1_f, lin1_b = _linear_forward_flops_bytes(_LINEAR1)
    lin2_f, lin2_b = _linear_forward_flops_bytes(_LINEAR2)
    return {
        "Conv2d": (conv1_f + conv2_f, conv1_b + conv2_b),
        "MaxPool2d": (0, _pool_forward_bytes(_POOL1) + _pool_forward_bytes(_POOL2)),
        "Linear": (lin1_f + lin2_f, lin1_b + lin2_b),
        "ReLU": (0, 0),  # size-dependent; deliberately not modeled (mixed shapes at each ReLU site)
        "Flatten": (0, 0),
    }


def _backward_model() -> "dict[str, tuple[int, int]]":
    """`node.name -> (flops, bytes)`, matching `backward_by_op_mean_seconds`'s keys."""
    conv1_f, conv1_b = _conv_backward_flops_bytes(_CONV1)
    conv2_f, conv2_b = _conv_backward_flops_bytes(_CONV2)
    lin1_f, lin1_b = _linear_forward_flops_bytes(_LINEAR1)  # matmul backward: 2 matmuls, same total MAC count
    lin2_f, lin2_b = _linear_forward_flops_bytes(_LINEAR2)
    return {
        "conv2d": (conv1_f + conv2_f, conv1_b + conv2_b),
        "max_pool2d": (0, _pool_backward_bytes(_POOL1) + _pool_backward_bytes(_POOL2)),
        "@": (2 * (lin1_f + lin2_f), 2 * (lin1_b + lin2_b)),  # matmul backward issues 2 matmuls (grad wrt each operand)
    }


def _apply_roofline(name: str, mean_seconds: float, flops: int, nbytes: int, ceilings) -> dict:
    gflops = (flops / 1e9) / mean_seconds if mean_seconds > 0 and flops > 0 else 0.0
    gbps = (nbytes / 1e9) / mean_seconds if mean_seconds > 0 and nbytes > 0 else 0.0
    ai = rf.arithmetic_intensity(flops, nbytes)
    rec = {"op": name, "mean_seconds": mean_seconds, "flops": flops, "bytes": nbytes,
           "arithmetic_intensity": ai, "achieved_gflops": gflops, "achieved_gbps": gbps}
    if ceilings is not None and mean_seconds > 0:
        c = rf.classify(gflops, mean_seconds, ai, ceilings, achieved_gbps=gbps if gbps > 0 else None)
        rec.update({"classification": c.label, "fraction_of_ceiling": c.fraction_of_ceiling})
    return rec


def _profile_overhead(iterations: int = 30) -> dict:
    """Section 37: plain (uninstrumented) `.backward()` vs. the per-op-timed walker."""
    from examples.mnist.model import build_model

    forge.random.seed(0)
    model = build_model().to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    x_data = rng.standard_normal((64, 1, 28, 28)).astype(np.float32)
    y_data = rng.integers(0, 10, size=(64,)).astype(np.int64)

    def plain_step():
        x = forge.Tensor(x_data, device="cuda")
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y_data)
        loss.backward()
        optimizer.step()

    def instrumented_step():
        x = forge.Tensor(x_data, device="cuda")
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y_data)
        grad_seed = get_backend(loss.device).from_array(np.ones((), dtype=loss._data.dtype), loss._data.dtype)
        _profiled_run_backward(loss, grad_seed, "cuda", {})
        optimizer.step()

    for step in (plain_step, instrumented_step):
        for _ in range(5):
            step()
        _sync("cuda")

    plain_times, instrumented_times = [], []
    for _ in range(iterations):
        _sync("cuda")
        t0 = time.perf_counter()
        plain_step()
        _sync("cuda")
        plain_times.append(time.perf_counter() - t0)
    for _ in range(iterations):
        _sync("cuda")
        t0 = time.perf_counter()
        instrumented_step()
        _sync("cuda")
        instrumented_times.append(time.perf_counter() - t0)

    plain_mean = statistics.mean(plain_times)
    instrumented_mean = statistics.mean(instrumented_times)
    return {
        "plain_mean_seconds": plain_mean,
        "instrumented_mean_seconds": instrumented_mean,
        "overhead_fraction": (instrumented_mean - plain_mean) / plain_mean if plain_mean > 0 else float("inf"),
    }


def _run(ceilings) -> dict:
    sync_profile = _profile_device_wrapper()
    batch_sweep = [_profile_async_epoch(bs, prefetch_size=2, n_samples=N_SAMPLES) for bs in BATCH_SIZES]
    overhead = _profile_overhead()

    forward_model = _forward_model()
    backward_model = _backward_model()
    ranking = []
    total_seconds = sum(sync_profile["forward_by_layer_mean_seconds"].values()) + sum(
        sync_profile["backward_by_op_mean_seconds"].values()
    )
    for name, mean_s in sync_profile["forward_by_layer_mean_seconds"].items():
        flops, nbytes = forward_model.get(name, (0, 0))
        rec = _apply_roofline(f"forward:{name}", mean_s, flops, nbytes, ceilings)
        rec["percent_of_step"] = (mean_s / total_seconds * 100) if total_seconds > 0 else 0.0
        ranking.append(rec)
    for name, mean_s in sync_profile["backward_by_op_mean_seconds"].items():
        flops, nbytes = backward_model.get(name, (0, 0))
        rec = _apply_roofline(f"backward:{name}", mean_s, flops, nbytes, ceilings)
        rec["percent_of_step"] = (mean_s / total_seconds * 100) if total_seconds > 0 else 0.0
        ranking.append(rec)
    ranking.sort(key=lambda r: r["mean_seconds"], reverse=True)

    return {
        "sync_per_layer_per_op": sync_profile,
        "batch_size_sweep": batch_sweep,
        "profiling_overhead": overhead,
        "kernel_ranking": ranking,
    }


def _profile_device_wrapper() -> dict:
    return _profile_device("cuda")


def _render_report(profile: dict) -> str:
    lines = ["=== M35 MNIST workload characterization (940MX, real CUDA) ===", ""]
    lines.append("-- Kernel contribution ranking (Section 26) --")
    for r in profile["kernel_ranking"][:10]:
        cls = r.get("classification", "n/a")
        lines.append(
            f"  {r['op']:<22}{r['percent_of_step']:6.2f}%  {r['mean_seconds']*1e3:8.4f}ms  "
            f"{r['achieved_gflops']:8.3f} GFLOP/s  {r['achieved_gbps']:8.3f} GB/s  {cls}"
        )
    lines.append("")

    lines.append("-- Batch-size scaling (Section 33) --")
    for r in profile["batch_size_sweep"]:
        lines.append(
            f"  batch={r['batch_size']:>4}  {r['samples_per_sec']:9.0f} samples/sec  "
            f"compute_util={r['compute_stream_utilization']*100:5.1f}%  "
            f"fwd={r['gpu_forward_ms']:.4f}ms bwd={r['gpu_backward_ms']:.4f}ms opt={r['gpu_optimizer_ms']:.4f}ms"
        )
    lines.append("")

    o = profile["profiling_overhead"]
    lines.append("-- Profiling overhead (Section 37) --")
    lines.append(f"  plain step:        {o['plain_mean_seconds']*1e3:.4f}ms")
    lines.append(f"  instrumented step: {o['instrumented_mean_seconds']*1e3:.4f}ms")
    lines.append(f"  overhead:          {o['overhead_fraction']*100:.1f}%")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m35_mnist.json")
    parser.add_argument("--ceilings", default="benchmarks/results/m35_hardware.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- m35_mnist requires real CUDA hardware.")
        return

    ceilings = None
    if Path(args.ceilings).exists():
        ceilings = rf.load_ceilings(args.ceilings)
    else:
        print(f"Warning: {args.ceilings} not found -- run m35_hardware first. Proceeding without classification.")

    profile = _run(ceilings)
    print(_render_report(profile))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"environment": collect_environment(), "profile": profile}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved profile -> {output_path}")


if __name__ == "__main__":
    main()
