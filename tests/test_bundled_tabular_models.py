"""Milestone 114 tests: the committed `models/tabular_classifier/` and `models/tabular_regressor/`.

Run as an external user would (the I1 lesson for `models/image_classifier`): the
consumer script in a separate process, a working directory outside the repository,
stdout through a pipe, `CUDA_VISIBLE_DEVICES=-1`. The artifacts are CPU-trained
(`device: cpu`), so they must load on any machine. No CUDA hardware is needed.
The scripts use only public Forge APIs; nothing here imports `examples/` code.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge

_MODELS = Path(__file__).resolve().parents[1] / "models"
_CLS_DIR, _REG_DIR = _MODELS / "tabular_classifier", _MODELS / "tabular_regressor"
_CLS_MODEL, _REG_MODEL = _CLS_DIR / "diabetes_classifier.forge", _REG_DIR / "concrete_strength_regressor.forge"
_HOLDOUT = Path(__file__).resolve().parents[1] / "examples" / "tabular_diabetes" / "data" / "diabetes_holdout_eval.csv"

_PATIENT = "6 148 72 35 0 33.6 0.627 50"  # first row of the Pima CSV, sentinel zero and all
_MIX = "540 0 0 162 2.5 1040 676 28"      # first row of the Concrete table


def _run(directory: Path, *args, cwd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    return subprocess.run(
        [sys.executable, str(directory / "predict.py"), *map(str, args)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
    )


def test_bundled_artifacts_are_portable_and_self_describing():
    cls = forge.inspect_model(str(_CLS_MODEL))
    assert (cls.task, cls.classes, cls.device, cls.input_schema.feature_count) == (
        "tabular_classification", ["no_diabetes", "diabetes"], "cpu", 8)
    assert cls.preprocessing is not None and "ReplaceValue" in cls.preprocessing.description
    reg = forge.inspect_model(str(_REG_MODEL))
    assert (reg.task, reg.classes, reg.device, reg.input_schema.feature_count) == ("regression", None, "cpu", 8)
    assert reg.preprocessing is not None and "Normalize" in reg.preprocessing.description


def test_bundled_classifier_still_beats_the_baseline_on_the_m113_holdout():
    holdout = np.genfromtxt(_HOLDOUT, delimiter=",", skip_header=1)
    result = forge.load_predictor(str(_CLS_MODEL)).evaluate(holdout[:, :-1], holdout[:, -1].astype(int))
    assert result.samples == 154 and result.baseline_accuracy == pytest.approx(0.6233766, abs=1e-6)
    assert result.accuracy >= 0.70 and np.isfinite(result.loss)


def test_classifier_script_predicts_from_outside_the_repo(tmp_path):
    result = _run(_CLS_DIR, *_PATIENT.split(), cwd=tmp_path)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Forge Tabular Classifier" in result.stdout
    label = next(line for line in result.stdout.splitlines() if line.startswith("Prediction: "))[len("Prediction: "):]
    assert label in ("no_diabetes", "diabetes") and "Confidence:" in result.stdout
    result.stdout.encode("ascii")  # portable to any console encoding
    expected = forge.load_predictor(str(_CLS_MODEL), device="cpu").predict([[float(v) for v in _PATIENT.split()]])[0]
    assert label == expected.label and f"{expected.confidence:.1%}" in result.stdout


def test_classifier_script_accepts_one_comma_separated_argument(tmp_path):
    a = _run(_CLS_DIR, "6,148,72,35,0,33.6,0.627,50", cwd=tmp_path)
    b = _run(_CLS_DIR, *_PATIENT.split(), cwd=tmp_path)
    assert a.returncode == b.returncode == 0 and a.stdout == b.stdout


def test_regressor_script_predicts_from_outside_the_repo(tmp_path):
    result = _run(_REG_DIR, *_MIX.split(), cwd=tmp_path)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Forge Tabular Regressor" in result.stdout
    line = next(l for l in result.stdout.splitlines() if l.startswith("Predicted compressive strength: "))
    expected = float(forge.load_predictor(str(_REG_MODEL), device="cpu").predict([[float(v) for v in _MIX.split()]]).numpy()[0, 0])
    assert line == f"Predicted compressive strength: {expected:.1f} MPa"
    assert 20.0 < expected < 100.0  # a plausible strength for this mixture, not a benchmark
    result.stdout.encode("ascii")


@pytest.mark.parametrize("directory, good", [(_CLS_DIR, _PATIENT), (_REG_DIR, _MIX)])
@pytest.mark.parametrize("bad, message", [
    ("1 2 3", "expected 8 numbers, got 3"),
    ("a b c d e f g h", "must be a number"),
    ("1 2 nan 4 5 6 7 8", "non-finite"),
    ("1 2 inf 4 5 6 7 8", "non-finite"),
])
def test_scripts_report_bad_input_cleanly(tmp_path, directory, good, bad, message):
    result = _run(directory, *bad.split(), cwd=tmp_path)
    assert result.returncode == 1
    assert message in result.stdout
    assert "Traceback" not in result.stderr and "Traceback" not in result.stdout


@pytest.mark.parametrize("directory", [_CLS_DIR, _REG_DIR])
def test_scripts_with_no_arguments_print_usage(tmp_path, directory):
    result = _run(directory, cwd=tmp_path)
    assert result.returncode == 1 and "Usage: python predict.py" in result.stdout and "Example:" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("directory", [_CLS_DIR, _REG_DIR])
def test_scripts_run_from_their_own_directory_too(directory):
    good = _PATIENT if directory == _CLS_DIR else _MIX
    result = _run(directory, *good.split(), cwd=directory)
    assert result.returncode == 0, result.stderr
