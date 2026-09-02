"""Hardware/software environment capture for benchmark reproducibility (Milestone 11).

Every benchmark run records enough metadata to interpret its own numbers
later, per `docs/performance/benchmarking.md`'s reproducibility
requirement: Forge never claims a benchmark result generalizes beyond the
environment it was measured in.
"""

from __future__ import annotations

import platform
import subprocess
import sys

import numpy as np

import forge


def _git_commit() -> "str | None":
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _cuda_environment() -> dict:
    from forge.backend.cuda.backend import is_cuda_available

    if not is_cuda_available():
        return {"available": False}

    from forge.backend.cuda.backend import get_cuda_backend

    backend = get_cuda_backend()
    info: dict = {"available": True, "device_count": backend.device_count}

    # Best-effort extras via nvidia-smi; absence of the tool (or a parse
    # failure) never fails the benchmark run -- this is descriptive
    # metadata, not something correctness depends on.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            fields = [f.strip() for f in result.stdout.strip().splitlines()[0].split(",")]
            if len(fields) == 4:
                info["gpu_name"], info["driver_version"], info["memory_total"], info["compute_capability"] = fields
    except Exception:
        pass

    return info


def collect_environment() -> dict:
    """A JSON-safe dict describing the machine/software this run executed on."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "forge_version": forge.__version__,
        "forge_commit": _git_commit(),
        "cuda": _cuda_environment(),
    }
