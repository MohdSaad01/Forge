"""MNIST workload profiling (Milestone 21, Section 4).

Breaks one training iteration of the real M20 CNN
(`examples.mnist.model.build_model()`) down into:

```text
data transfer -> forward (per-layer-type) -> loss -> backward (per-op) -> optimizer
```

so the dominant cost can be identified from measurement rather than assumed.
This is a diagnostic tool, not a benchmark-suite category with a stable
schema: it prints a breakdown table and (optionally) writes a JSON dump, but
its output is not part of `benchmarks/results/latest.json`'s
`BenchmarkResult` list.

## How the breakdown is obtained
**Forward**, per layer: `Sequential.forward()` (`forge/nn/container.py`) is
just `for child in self.named_children(): x = child(x)` -- this module runs
that exact loop by hand, with an explicit synchronization point (a no-op on
CPU) bracketing each child call, and accumulates elapsed time keyed by each
child's class name (`Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, `Linear`).
Since this is the same sequence of Tensor operations `Sequential.forward()`
itself performs, unpacking the loop changes nothing about what is computed.

**Backward**, per op: `Tensor.backward()` ultimately calls
`forge.autograd.engine.run_backward`, which walks the graph in one call and
returns only the final leaf gradients -- it has no per-operation timing hook
of its own. `_profiled_run_backward` below is a small, self-contained
re-implementation of that same topological-order graph walk (using the
public `Node`/`_topological_order` primitives `forge/autograd/engine.py`
already exposes), timing each `node.backward_fn(...)` call individually and
aggregating by `node.name` (the same op-name strings `Tensor`'s forward
methods already attach, e.g. `"conv2d"`, `"relu"`, `"max_pool2d"`,
`"reshape"`, `"@"`, `"sum"`). This adds synchronization overhead between
every backward op on CUDA (extra `cudaDeviceSynchronize()` calls a normal
`backward()` does not make) -- appropriate for a profiling tool that needs
per-op resolution, not for the hot training path, so this instrumented
walker is never used outside this diagnostic script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import forge
import forge.nn as nn
import forge.optim as optim
from forge.autograd.engine import Node, _topological_order
from forge.backend import get_backend
from forge.backend.cuda.backend import is_cuda_available

from .environment import collect_environment
from .sizes import MNIST_PROFILE_CONFIG


def _sync(device: str) -> None:
    if device == "cuda":
        from forge.backend.cuda.backend import get_cuda_backend

        get_cuda_backend().synchronize()


def _profiled_run_backward(root, grad_array, device: str, op_times: "dict[str, float]") -> None:
    """`forge.autograd.engine.run_backward`, instrumented to time each op by name.

    Mirrors that function's algorithm exactly (topological order, pending
    gradient accumulation via `Backend.add()`, graph freed on use) -- see
    this module's docstring for why a separate copy is used instead of
    modifying the real engine.
    """
    order = _topological_order(root)
    pending: "dict[int, object]" = {id(root): grad_array}
    sync = (lambda: _sync(device))

    for tensor in reversed(order):
        grad_output = pending.pop(id(tensor), None)
        if grad_output is None:
            continue

        node = tensor._grad_fn
        if node is None:
            if tensor.is_leaf:
                tensor._accumulate_grad(grad_output)
            continue

        sync()
        t0 = time.perf_counter()
        input_grads = node.backward_fn(grad_output)
        sync()
        t1 = time.perf_counter()
        op_times[node.name] = op_times.get(node.name, 0.0) + (t1 - t0)

        for inp, grad in zip(node.inputs, input_grads):
            if grad is None or not inp.requires_grad:
                continue
            if id(inp) in pending:
                pending[id(inp)] = get_backend(inp.device).add(pending[id(inp)], grad)
            else:
                pending[id(inp)] = grad

        tensor._grad_fn = None


def _profile_device(device: str) -> dict:
    from examples.mnist.model import build_model

    cfg = MNIST_PROFILE_CONFIG
    batch_size = cfg["batch_size"]

    forge.random.seed(0)
    model = build_model()
    if device == "cuda":
        model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    rng = np.random.default_rng(0)
    x_data = rng.standard_normal((batch_size, 1, 28, 28)).astype(np.float32)
    y_data = rng.integers(0, 10, size=(batch_size,)).astype(np.int64)

    phase_totals = {"transfer": 0.0, "forward": 0.0, "loss": 0.0, "backward": 0.0, "optimizer": 0.0}
    forward_by_layer: "dict[str, float]" = {}
    backward_by_op: "dict[str, float]" = {}

    def one_step(record: bool) -> None:
        _sync(device)
        t0 = time.perf_counter()
        x = forge.Tensor(x_data, device=device)
        _sync(device)
        t1 = time.perf_counter()

        optimizer.zero_grad()
        out = x
        for _, child in model.named_children():
            _sync(device)
            layer_t0 = time.perf_counter()
            out = child(out)
            _sync(device)
            layer_t1 = time.perf_counter()
            if record:
                key = type(child).__name__
                forward_by_layer[key] = forward_by_layer.get(key, 0.0) + (layer_t1 - layer_t0)
        _sync(device)
        t2 = time.perf_counter()

        loss = loss_fn(out, y_data)
        _sync(device)
        t3 = time.perf_counter()

        if record:
            # Mirrors `Tensor.backward()`'s own default-gradient construction
            # exactly (`forge/tensor/tensor.py`) for a scalar loss output.
            grad_seed = get_backend(loss.device).from_array(
                np.ones((), dtype=loss._data.dtype), loss._data.dtype
            )
            _profiled_run_backward(loss, grad_seed, device, backward_by_op)
        else:
            loss.backward()
        _sync(device)
        t4 = time.perf_counter()

        optimizer.step()
        _sync(device)
        t5 = time.perf_counter()

        if record:
            phase_totals["transfer"] += t1 - t0
            phase_totals["forward"] += t2 - t1
            phase_totals["loss"] += t3 - t2
            phase_totals["backward"] += t4 - t3
            phase_totals["optimizer"] += t5 - t4

    for _ in range(cfg["warmup_iterations"]):
        one_step(record=False)
    _sync(device)

    for _ in range(cfg["iterations"]):
        one_step(record=True)

    n = cfg["iterations"]
    return {
        "device": device,
        "batch_size": batch_size,
        "iterations": n,
        "warmup_iterations": cfg["warmup_iterations"],
        "phase_mean_seconds": {k: v / n for k, v in phase_totals.items()},
        "forward_by_layer_mean_seconds": {k: v / n for k, v in forward_by_layer.items()},
        "backward_by_op_mean_seconds": {k: v / n for k, v in backward_by_op.items()},
    }


def _render_report(profile: dict) -> str:
    lines = [f"Device: {profile['device']}  (batch_size={profile['batch_size']}, iterations={profile['iterations']})", ""]
    lines.append("Phase breakdown (mean per training step):")
    total = sum(profile["phase_mean_seconds"].values())
    for phase, seconds in profile["phase_mean_seconds"].items():
        pct = (seconds / total * 100) if total > 0 else 0.0
        lines.append(f"  {phase:<12} {seconds * 1000:8.4f} ms  ({pct:5.1f}%)")
    lines.append(f"  {'TOTAL':<12} {total * 1000:8.4f} ms")
    lines.append("")
    lines.append("Forward, by layer type (mean per training step):")
    for layer, seconds in sorted(profile["forward_by_layer_mean_seconds"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {layer:<12} {seconds * 1000:8.4f} ms")
    lines.append("")
    lines.append("Backward, by op (mean per training step):")
    for op, seconds in sorted(profile["backward_by_op_mean_seconds"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {op:<12} {seconds * 1000:8.4f} ms")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/mnist_profile.json")
    args = parser.parse_args(argv)

    environment = collect_environment()
    devices = ["cpu"] + (["cuda"] if is_cuda_available() else [])
    profiles = [_profile_device(d) for d in devices]

    for profile in profiles:
        print(_render_report(profile))
        print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"environment": environment, "profiles": profiles}, indent=2), encoding="utf-8"
    )
    print(f"Saved profile -> {output_path}")


if __name__ == "__main__":
    main()
