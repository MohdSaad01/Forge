"""Milestone 97 CUDA tests: artifact evaluation on the real CUDA backend.

Mirrors `tests/test_tabular_diabetes_workflow_cuda.py`'s coverage: proves
`evaluate.py::evaluate_artifact()` (Milestone 97) produces identical
accuracy whether the artifact is loaded onto CPU or CUDA. Hardware-verified
on the reference GeForce 940MX.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from forge.backend.cuda import is_cuda_available

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.dataset import load_raw  # noqa: E402
from examples.tabular_diabetes.evaluate import evaluate_artifact  # noqa: E402
from examples.tabular_diabetes.train import main as train_main  # noqa: E402

_SMALL = ["--batch-size", "16"]


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(X.shape[1])] + ["Outcome"])
        for row, label in zip(X, y):
            writer.writerow(list(row) + [int(label)])


def test_evaluate_artifact_agrees_between_cpu_and_cuda(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "3", "--seed", "9", "--device", "cuda", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_diabetes_model.forge"

    X, y = load_raw()
    csv_path = tmp_path / "eval.csv"
    _write_csv(csv_path, X[:40], y[:40])

    cpu_summary = evaluate_artifact(str(model_path), str(csv_path), device="cpu")
    cuda_summary = evaluate_artifact(str(model_path), str(csv_path), device="cuda")

    assert cpu_summary.samples == cuda_summary.samples
    assert cpu_summary.accuracy == pytest.approx(cuda_summary.accuracy, abs=1e-6)
    assert cpu_summary.baseline_accuracy == cuda_summary.baseline_accuracy
