"""CUDA allocation-behavior profiling (Milestone 24).

A diagnostic script, not a benchmark-suite category with a stable schema --
mirrors `benchmarks/mnist_profile.py`'s own status (Milestone 21) exactly,
but for *allocation behavior* instead of *timing breakdown*. Uses
`forge.cuda.profiler` (Milestone 24) to record allocation/free events, then
`benchmarks/alloc_analysis.py`'s pure functions to summarize them.

Four sections, matching the M24 brief:

1. `_profile_mnist_workload` -- Section 6: the real M20 CNN, warmup vs.
   steady-state, phase-tagged (transfer/forward/loss/backward/optimizer).
2. `_profile_operations` -- Section 7/8: representative op-level forward and
   backward allocation traffic, one scale each (allocation *counts* barely
   depend on tensor size -- one representative scale is enough to see the
   pattern; `benchmarks/ops_bench.py`/`backward_bench.py` already cover the
   multi-scale *timing* question).
3. `_profile_transfers` -- Section 9: CPU<->CUDA transfer allocation
   behavior at the existing `TRANSFER_SIZES` scales.
4. `_measure_alloc_free_overhead` -- Section 10: direct `cudaMalloc`/
   `cudaFree` host-API wall-clock cost, measured by timing
   `CUDABackend._alloc`/`cf_free` directly (bypassing `CUDAStorage`, whose
   `__del__` would otherwise be the only free path) -- these are classic,
   host-blocking CUDA Runtime API calls (no kernel launch, no queue), so a
   plain `time.perf_counter()` bracket around each call already measures the
   real host-observable cost with no asynchronous-execution ambiguity to
   correct for (see `docs/architecture/cuda-memory-allocator.md`'s
   "cudaMalloc/cudaFree overhead" section for why this differs from
   `benchmarks/timing.py`'s synchronize-bracketed kernel methodology).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

import forge
import forge.nn as nn
import forge.optim as optim
from forge.backend.cuda.backend import is_cuda_available

from . import alloc_analysis as aa
from .environment import collect_environment
from .sizes import (
    ADAM_PARAM_SIZES,
    CONV2D_CONFIGS,
    DROPOUT_P,
    ELEMENTWISE_SIZES,
    LOSS_CONFIGS,
    MATMUL_DIMS,
    MNIST_PROFILE_CONFIG,
    POOL2D_KERNEL,
    TRANSFER_SIZES,
)

_OP_WARMUP = 5
_OP_ITERATIONS = 30


def _sync() -> None:
    from forge.backend.cuda.backend import get_cuda_backend

    get_cuda_backend().synchronize()


def _summarize_events(events, iterations: "int | None" = None) -> dict:
    """The common per-trace summary reused by every section below."""
    alloc_dist = aa.size_distribution(events, kind="alloc")
    free_dist = aa.size_distribution(events, kind="free")
    lifetimes = aa.lifetime_distribution(events)
    split = aa.persistent_vs_temporary(events)
    reuse = aa.reuse_opportunity(events)
    sim_exact = aa.simulate_caching_allocator(events, policy="exact")
    sim_size_class = aa.simulate_caching_allocator(events, policy="size_class")

    by_category: "dict[str, dict[str, int]]" = {}
    for e in events:
        key = e.category or "(untagged)"
        bucket = by_category.setdefault(key, {"alloc_count": 0, "alloc_bytes": 0, "free_count": 0, "free_bytes": 0})
        if e.kind == "alloc":
            bucket["alloc_count"] += 1
            bucket["alloc_bytes"] += e.nbytes
        else:
            bucket["free_count"] += 1
            bucket["free_bytes"] += e.nbytes

    summary = {
        "total_events": len(events),
        "alloc_size_distribution": alloc_dist,
        "free_size_distribution": free_dist,
        "lifetime_distribution": lifetimes,
        "persistent_vs_temporary": split,
        "reuse_opportunity": reuse,
        "cache_simulation_exact": sim_exact,
        "cache_simulation_size_class": sim_size_class,
        "by_category": by_category,
    }
    if iterations:
        summary["per_iteration"] = {
            "alloc_count": alloc_dist["count"] / iterations,
            "alloc_bytes": alloc_dist["total_bytes"] / iterations,
            "free_count": free_dist["count"] / iterations,
        }
    return summary


# -- 1. M20 MNIST workload (Section 6) -------------------------------------------


def _profile_mnist_workload() -> dict:
    from examples.mnist.model import build_model

    cfg = MNIST_PROFILE_CONFIG
    batch_size = cfg["batch_size"]

    forge.random.seed(0)
    model = build_model().to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    rng = np.random.default_rng(0)
    x_data = rng.standard_normal((batch_size, 1, 28, 28)).astype(np.float32)
    y_data = rng.integers(0, 10, size=(batch_size,)).astype(np.int64)

    def one_step() -> None:
        with forge.cuda.profiler.tag("transfer"):
            x = forge.Tensor(x_data, device="cuda")
        optimizer.zero_grad()
        with forge.cuda.profiler.tag("forward"):
            out = model(x)
        with forge.cuda.profiler.tag("loss"):
            loss = loss_fn(out, y_data)
        with forge.cuda.profiler.tag("backward"):
            loss.backward()
        with forge.cuda.profiler.tag("optimizer"):
            optimizer.step()

    for _ in range(cfg["warmup_iterations"]):
        one_step()
    _sync()
    gc.collect()

    # Clean (unprofiled) wall-clock timing of one full steady-state
    # iteration, measured *without* the profiler running -- the allocation
    # trace below is captured in a separate pass so the profiler's own
    # per-event bookkeeping cost never inflates this number (see Section 10's
    # overhead comparison, which uses this as the "full iteration" baseline).
    _sync()
    t0 = time.perf_counter()
    for _ in range(cfg["iterations"]):
        one_step()
    _sync()
    t1 = time.perf_counter()
    mean_iteration_seconds = (t1 - t0) / cfg["iterations"]
    gc.collect()

    forge.cuda.reset_peak_memory_stats()
    mem_before = forge.cuda.memory_stats()
    forge.cuda.profiler.reset()
    forge.cuda.profiler.start()
    for _ in range(cfg["iterations"]):
        one_step()
    _sync()
    forge.cuda.profiler.stop()
    events = forge.cuda.profiler.events()
    mem_after = forge.cuda.memory_stats()
    gc.collect()
    mem_after_gc = forge.cuda.memory_stats()

    summary = _summarize_events(events, iterations=cfg["iterations"])
    summary.update({
        "batch_size": batch_size,
        "iterations": cfg["iterations"],
        "warmup_iterations": cfg["warmup_iterations"],
        "mean_iteration_seconds": mean_iteration_seconds,
        "peak_allocated_bytes": mem_after.peak_allocated_bytes,
        "allocated_bytes_before": mem_before.allocated_bytes,
        "allocated_bytes_after_no_gc": mem_after.allocated_bytes,
        "allocated_bytes_after_gc": mem_after_gc.allocated_bytes,
    })
    return summary


# -- 2. Representative operations, forward and backward (Section 7/8) -----------


def _build_op_cases() -> "dict[str, dict]":
    """One representative ("small") scale per operation -- see module docstring."""
    n = ELEMENTWISE_SIZES["small"]
    dim = MATMUL_DIMS["small"]
    conv_cfg = CONV2D_CONFIGS["small"]
    loss_cfg = LOSS_CONFIGS["small"]
    adam_n = ADAM_PARAM_SIZES["small"]
    rng = np.random.default_rng(42)

    def leaf_vec(seed):
        r = np.random.default_rng(seed)
        return lambda: forge.Tensor(r.standard_normal(n).astype(np.float32), device="cuda", requires_grad=True)

    def leaf_pos_vec(seed):
        r = np.random.default_rng(seed)
        return lambda: forge.Tensor((r.random(n).astype(np.float32) + 0.1), device="cuda", requires_grad=True)

    def leaf_mat(seed):
        r = np.random.default_rng(seed)
        return lambda: forge.Tensor(r.standard_normal((dim, dim)).astype(np.float32), device="cuda", requires_grad=True)

    cases: "dict[str, dict]" = {}

    for name, fn in (("add", lambda a, b: a + b), ("sub", lambda a, b: a - b), ("mul", lambda a, b: a * b)):
        make_a, make_b = leaf_vec(1), leaf_vec(2)
        cases[name] = {"forward": lambda fn=fn, make_a=make_a, make_b=make_b: fn(make_a(), make_b())}

    make_ma, make_mb = leaf_mat(3), leaf_mat(4)
    cases["matmul"] = {"forward": lambda make_ma=make_ma, make_mb=make_mb: make_ma() @ make_mb()}

    make_v = leaf_vec(5)
    cases["sum"] = {"forward": lambda make_v=make_v: make_v().sum()}
    cases["reshape"] = {"forward": lambda make_v=make_v, n=n: make_v().reshape(n)}
    cases["relu"] = {"forward": lambda make_v=make_v: make_v().relu()}

    make_pos = leaf_pos_vec(6)
    cases["exp"] = {"forward": lambda make_pos=make_pos: make_pos().exp()}
    cases["log"] = {"forward": lambda make_pos=make_pos: make_pos().log()}

    def make_conv_inputs():
        x = forge.Tensor(
            rng.standard_normal((conv_cfg["N"], conv_cfg["Cin"], conv_cfg["H"], conv_cfg["W"])).astype(np.float32),
            device="cuda", requires_grad=True,
        )
        w = forge.Tensor(
            rng.standard_normal((conv_cfg["Cout"], conv_cfg["Cin"], conv_cfg["K"], conv_cfg["K"])).astype(np.float32),
            device="cuda", requires_grad=True,
        )
        b = forge.Tensor(rng.standard_normal((conv_cfg["Cout"],)).astype(np.float32), device="cuda", requires_grad=True)
        return x, w, b

    def conv_forward():
        x, w, b = make_conv_inputs()
        return x.conv2d(w, b, (1, 1), (0, 0))

    cases["conv2d"] = {"forward": conv_forward}

    def make_pool_input():
        h_out, w_out = conv_cfg["H"] - conv_cfg["K"] + 1, conv_cfg["W"] - conv_cfg["K"] + 1
        return forge.Tensor(
            rng.standard_normal((conv_cfg["N"], conv_cfg["Cout"], h_out, w_out)).astype(np.float32),
            device="cuda", requires_grad=True,
        )

    pool = nn.MaxPool2d(POOL2D_KERNEL)
    cases["max_pool2d"] = {"forward": lambda: pool(make_pool_input())}

    dropout = nn.Dropout(DROPOUT_P)
    dropout.train()
    cases["dropout"] = {"forward": lambda make_v=make_v: dropout(make_v())}

    batch, classes = loss_cfg["batch"], loss_cfg["classes"]
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def make_reg():
        pred = forge.Tensor(rng.standard_normal((batch, classes)).astype(np.float32), device="cuda", requires_grad=True)
        target = forge.Tensor(rng.standard_normal((batch, classes)).astype(np.float32), device="cuda")
        return pred, target

    cases["mse_loss"] = {"forward": lambda: mse(*make_reg())}

    def make_cls():
        logits = forge.Tensor(rng.standard_normal((batch, classes)).astype(np.float32), device="cuda", requires_grad=True)
        target = rng.integers(0, classes, size=(batch,)).astype(np.int64)
        return logits, target

    cases["cross_entropy_loss"] = {"forward": lambda: ce(*make_cls())}

    return cases


def _profile_forward_allocations(make_fn) -> "tuple":
    for _ in range(_OP_WARMUP):
        make_fn()
    _sync()
    gc.collect()
    forge.cuda.profiler.reset()
    forge.cuda.profiler.start()
    for _ in range(_OP_ITERATIONS):
        make_fn()
    _sync()
    forge.cuda.profiler.stop()
    return forge.cuda.profiler.events()


def _profile_backward_allocations(make_fn) -> "tuple":
    """Build `_OP_ITERATIONS` fresh forward passes (untracked), then profile only `.backward()`.

    Mirrors `benchmarks/backward_bench.py`'s `_build_calls` pattern: a
    non-leaf output's graph is consumed by its one and only `backward()`
    call, so allocation traffic *from backward* can only be isolated by
    building each iteration's forward pass outside the profiling window.
    """
    calls = []
    for _ in range(_OP_WARMUP + _OP_ITERATIONS):
        y = make_fn()
        grad = None
        if y.shape != ():
            grad = forge.Tensor(np.ones(y.shape, dtype=np.float32), device="cuda")
        calls.append((lambda y=y, grad=grad: y.backward(grad)))

    for call in calls[:_OP_WARMUP]:
        call()
    _sync()
    gc.collect()

    forge.cuda.profiler.reset()
    forge.cuda.profiler.start()
    for call in calls[_OP_WARMUP:]:
        call()
    _sync()
    forge.cuda.profiler.stop()
    return forge.cuda.profiler.events()


def _profile_optimizer_allocations() -> dict:
    results = {}
    for name, opt_cls, kwargs in (("sgd_step", optim.SGD, {"lr": 0.01}), ("adam_step", optim.Adam, {"lr": 0.01})):
        n = ADAM_PARAM_SIZES["small"]
        rng = np.random.default_rng(20)
        param = nn.Parameter(rng.standard_normal(n).astype(np.float32), device="cuda")
        param.grad = forge.Tensor(rng.standard_normal(n).astype(np.float32), device="cuda")
        opt = opt_cls([param], **kwargs)

        for _ in range(_OP_WARMUP):
            opt.step()
        _sync()
        gc.collect()

        forge.cuda.profiler.reset()
        forge.cuda.profiler.start()
        for _ in range(_OP_ITERATIONS):
            opt.step()
        _sync()
        forge.cuda.profiler.stop()
        results[name] = _summarize_events(forge.cuda.profiler.events(), iterations=_OP_ITERATIONS)
    return results


def _profile_operations() -> dict:
    cases = _build_op_cases()
    results: "dict[str, dict]" = {}
    for name, funcs in cases.items():
        forward_events = _profile_forward_allocations(funcs["forward"])
        backward_events = _profile_backward_allocations(funcs["forward"])
        results[name] = {
            "forward": _summarize_events(forward_events, iterations=_OP_ITERATIONS),
            "backward": _summarize_events(backward_events, iterations=_OP_ITERATIONS),
        }
    results.update(_profile_optimizer_allocations())
    return results


# -- 3. CPU <-> CUDA transfers (Section 9) ---------------------------------------


def _profile_transfers() -> dict:
    results = {}
    for scale, n in TRANSFER_SIZES.items():
        rng = np.random.default_rng(30)
        cpu_data = rng.standard_normal(n).astype(np.float32)
        cuda_t = forge.Tensor(cpu_data, device="cuda")

        def h2d():
            forge.Tensor(cpu_data, device="cuda")

        def d2h(cuda_t=cuda_t):
            cuda_t.to("cpu")

        for _ in range(_OP_WARMUP):
            h2d()
        _sync()
        gc.collect()
        forge.cuda.profiler.reset()
        forge.cuda.profiler.start()
        for _ in range(_OP_ITERATIONS):
            h2d()
        _sync()
        forge.cuda.profiler.stop()
        h2d_events = forge.cuda.profiler.events()

        for _ in range(_OP_WARMUP):
            d2h()
        gc.collect()
        forge.cuda.profiler.reset()
        forge.cuda.profiler.start()
        for _ in range(_OP_ITERATIONS):
            d2h()
        _sync()
        forge.cuda.profiler.stop()
        d2h_events = forge.cuda.profiler.events()

        results[scale] = {
            "nbytes": n * 4,
            "h2d": _summarize_events(h2d_events, iterations=_OP_ITERATIONS),
            "d2h": _summarize_events(d2h_events, iterations=_OP_ITERATIONS),
        }
    return results


# -- 4. cudaMalloc / cudaFree overhead (Section 10/11) ---------------------------


def _measure_alloc_free_overhead() -> dict:
    """Direct host-API timing -- see module docstring for why no synchronize bracket is needed."""
    from forge.backend.cuda import memory as _memory
    from forge.backend.cuda.backend import get_cuda_backend

    backend = get_cuda_backend()
    warmup, iterations = 10, 200
    results = {}
    for scale, n in ELEMENTWISE_SIZES.items():
        nbytes = n * 4

        for _ in range(warmup):
            ptr = backend._alloc(nbytes)
            code = backend._lib.cf_free(ptr)
            _memory.record_free(nbytes, ptr.value or 0)

        alloc_times, free_times = [], []
        for _ in range(iterations):
            t0 = time.perf_counter()
            ptr = backend._alloc(nbytes)
            t1 = time.perf_counter()
            backend._lib.cf_free(ptr)
            t2 = time.perf_counter()
            _memory.record_free(nbytes, ptr.value or 0)
            alloc_times.append(t1 - t0)
            free_times.append(t2 - t1)

        results[scale] = {
            "nbytes": nbytes,
            "mean_alloc_seconds": statistics.mean(alloc_times),
            "median_alloc_seconds": statistics.median(alloc_times),
            "mean_free_seconds": statistics.mean(free_times),
            "median_free_seconds": statistics.median(free_times),
            "iterations": iterations,
        }
    return results


# -- report rendering --------------------------------------------------------------


def _render_summary(profile: dict) -> str:
    lines = []
    mnist = profile["mnist_workload"]
    lines.append(
        f"MNIST workload (batch={mnist['batch_size']}, {mnist['iterations']} steady-state iterations, "
        f"{mnist['warmup_iterations']} warmup):"
    )
    lines.append(f"  mean iteration time:      {mnist['mean_iteration_seconds'] * 1000:.4f} ms")
    lines.append(f"  allocations/iteration:    {mnist['per_iteration']['alloc_count']:.1f}")
    lines.append(f"  alloc bytes/iteration:    {mnist['per_iteration']['alloc_bytes']:.0f}")
    lines.append(f"  peak allocated bytes:     {mnist['peak_allocated_bytes']}")
    lines.append(
        f"  persistent bytes:         {mnist['persistent_vs_temporary']['persistent_bytes']} "
        f"({mnist['persistent_vs_temporary']['persistent_count']} blocks)"
    )
    lines.append(
        f"  temporary bytes:          {mnist['persistent_vs_temporary']['temporary_bytes']} "
        f"({mnist['persistent_vs_temporary']['temporary_count']} blocks)"
    )
    lines.append(
        f"  exact-size cache sim hit rate: {mnist['cache_simulation_exact']['cache_hit_rate'] * 100:.1f}%"
    )
    lines.append("")

    overhead = profile["alloc_free_overhead"]
    lines.append("cudaMalloc / cudaFree host-API overhead (direct timing, no kernel involved):")
    for scale, r in overhead.items():
        lines.append(
            f"  {scale:<8} ({r['nbytes']:>9} bytes): "
            f"alloc {r['mean_alloc_seconds'] * 1e6:8.2f} us, free {r['mean_free_seconds'] * 1e6:8.2f} us"
        )
    lines.append("")

    lines.append("Per-operation allocation counts (forward / backward, small scale):")
    for name, r in profile["operations"].items():
        fwd = r.get("forward")
        bwd = r.get("backward")
        if fwd is not None:
            lines.append(
                f"  {name:<20} forward: {fwd['per_iteration']['alloc_count']:5.1f} allocs/call, "
                f"backward: {bwd['per_iteration']['alloc_count']:5.1f} allocs/call"
            )
        else:
            lines.append(f"  {name:<20} step: {r['per_iteration']['alloc_count']:5.1f} allocs/call")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/alloc_profile.json")
    args = parser.parse_args(argv)

    if not is_cuda_available():
        print("CUDA is not available on this machine -- allocation profiling requires real CUDA hardware.")
        return

    environment = collect_environment()
    profile = {
        "mnist_workload": _profile_mnist_workload(),
        "operations": _profile_operations(),
        "transfers": _profile_transfers(),
        "alloc_free_overhead": _measure_alloc_free_overhead(),
    }

    print(_render_summary(profile))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"environment": environment, "profile": profile}, indent=2), encoding="utf-8")
    print(f"\nSaved allocation profile -> {output_path}")


if __name__ == "__main__":
    main()
