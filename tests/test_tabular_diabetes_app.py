"""Milestone 102 tests: `examples/tabular_diabetes/app.py`, the real
multi-row consumer application built around `forge.load_predictor()`.

```text
tabular_diabetes_model.forge (from Milestone 92's train.py)
    + a batch of raw patient rows
    -> app.py's run(): forge.load_predictor() once, predictor.predict() per row
```

Covers: the artifact is loaded exactly once for a whole batch (not once per
row -- the actual product requirement this milestone exists to demonstrate),
predictions agree with `forge.predict_model()`'s own one-shot result for the
same row, a single flat row is accepted the same way multiple rows are, a
structurally invalid row (Milestone 101's `InputSchema` check) is reported
inline instead of aborting the whole batch, and a genuine fresh-process run.
See `examples/tabular_diabetes/app.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.exceptions import DataError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.app import main, run  # noqa: E402
from examples.tabular_diabetes.dataset import load_raw  # noqa: E402
from examples.tabular_diabetes.train import main as train_main  # noqa: E402

_SMALL = ["--batch-size", "16"]


@pytest.fixture()
def small_artifact(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "3", "--seed", "11", "--output-dir", str(out_dir)])
    return out_dir / "tabular_diabetes_model.forge"


@pytest.fixture()
def sample_rows():
    X, _ = load_raw()
    return X[:3].tolist()


# -- run(): loads once, predicts per row --------------------------------------


def test_run_loads_the_artifact_exactly_once_for_a_whole_batch(small_artifact, sample_rows, monkeypatch):
    import forge.serialization.model as model_module

    calls = {"load_model": 0}
    original = model_module.load_model

    def _counted(*args, **kwargs):
        calls["load_model"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "load_model", _counted)

    predictor, results = run(str(small_artifact), json.dumps(sample_rows))
    assert len(results) == len(sample_rows)
    assert calls["load_model"] == 1


def test_run_returns_the_loaded_predictor(small_artifact, sample_rows):
    predictor, _results = run(str(small_artifact), json.dumps(sample_rows))
    assert predictor.task == "tabular_classification"
    assert predictor.classes == ["no_diabetes", "diabetes"]


def test_run_accepts_a_single_flat_row(small_artifact, sample_rows):
    _predictor, results = run(str(small_artifact), json.dumps(sample_rows[0]))
    assert len(results) == 1


def test_run_matches_predict_model_for_each_row(small_artifact, sample_rows):
    _predictor, results = run(str(small_artifact), json.dumps(sample_rows))
    for row, result in zip(sample_rows, results):
        expected = forge.predict_model(str(small_artifact), np.array([row], dtype=np.float32))[0]
        assert result.label == expected.label
        assert result.index == expected.index
        assert result.confidence == pytest.approx(expected.confidence, abs=1e-6)


def test_run_reports_an_invalid_row_without_discarding_the_rest(small_artifact, sample_rows):
    bad_row = sample_rows[0][:-1]  # 7 features instead of 8
    rows = [sample_rows[0], bad_row, sample_rows[1]]
    _predictor, results = run(str(small_artifact), json.dumps(rows))

    assert len(results) == 3
    assert not isinstance(results[0], DataError)
    assert isinstance(results[1], DataError)
    assert not isinstance(results[2], DataError)


# -- main(): CLI surface -------------------------------------------------------


def test_main_prints_one_line_per_row(small_artifact, sample_rows, capsys):
    main(["--model", str(small_artifact), "--input", json.dumps(sample_rows)])
    out = capsys.readouterr().out
    assert "Loaded" in out
    assert out.count("Row ") == len(sample_rows)


def test_main_reports_invalid_row_inline(small_artifact, sample_rows, capsys):
    bad_row = sample_rows[0][:-1]
    main(["--model", str(small_artifact), "--input", json.dumps([bad_row])])
    out = capsys.readouterr().out
    assert "invalid input" in out


# -- fresh process --------------------------------------------------------------


def test_app_works_from_a_genuinely_separate_process(small_artifact, sample_rows):
    script = (
        "from examples.tabular_diabetes.app import run\n"
        "import json\n"
        f"predictor, results = run({str(small_artifact)!r}, {json.dumps(sample_rows)!r})\n"
        "for r in results:\n"
        "    print(f'{r.label} {r.confidence:.6f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    _predictor, expected = run(str(small_artifact), json.dumps(sample_rows))
    for r in expected:
        assert f"{r.label} {r.confidence:.6f}" in result.stdout
