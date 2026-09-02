"""Structured benchmark output: JSON + a human-readable summary table (Milestone 11).

`BenchmarkResult` is a plain, JSON-serializable record -- not a hard-coded
table format. `render_table` is one reasonable text rendering of a list of
results; a caller wanting a different representation (CSV, a dataframe) can
build one from the same `BenchmarkResult` list without touching this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .timing import Timing


@dataclass(frozen=True)
class BenchmarkResult:
    category: str  # "forward" | "backward" | "transfer" | "training"
    operation: str  # "add", "matmul", "h2d", "full_step", ...
    device: str  # "cpu" | "cuda"
    scale: str  # "tiny" | "small" | "medium" | ...
    shape: str  # human-readable shape/size descriptor
    dtype: str
    mean_seconds: float
    median_seconds: float
    stdev_seconds: float
    min_seconds: float
    max_seconds: float
    warmup: int
    iterations: int
    extra: "dict[str, Any]" = field(default_factory=dict)

    @classmethod
    def from_timing(
        cls,
        *,
        category: str,
        operation: str,
        device: str,
        scale: str,
        shape: str,
        dtype: str,
        timing: Timing,
        extra: "dict[str, Any] | None" = None,
    ) -> "BenchmarkResult":
        return cls(
            category=category,
            operation=operation,
            device=device,
            scale=scale,
            shape=shape,
            dtype=dtype,
            mean_seconds=timing.mean,
            median_seconds=timing.median,
            stdev_seconds=timing.stdev,
            min_seconds=timing.min,
            max_seconds=timing.max,
            warmup=timing.warmup,
            iterations=timing.iterations,
            extra=dict(extra) if extra else {},
        )


def save_json(results: "list[BenchmarkResult]", environment: dict, path: "str | Path") -> None:
    payload = {"environment": environment, "results": [asdict(r) for r in results]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_table(results: "list[BenchmarkResult]") -> str:
    """A fixed-width text table, grouped in the order results were given."""
    headers = ["Category", "Operation", "Device", "Scale", "Shape", "Dtype", "Mean(ms)", "Std(ms)", "Iters"]
    if not results:
        return "  ".join(headers) + "\n(no results)"

    rows = [
        [
            r.category,
            r.operation,
            r.device,
            r.scale,
            r.shape,
            r.dtype,
            f"{r.mean_seconds * 1000:.4f}",
            f"{r.stdev_seconds * 1000:.4f}",
            str(r.iterations),
        ]
        for r in results
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt_row(cells: "list[str]") -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)
