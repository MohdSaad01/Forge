"""Issue I1 tests: the committed `models/image_classifier/` model is consumable on any machine.

The artifact was saved from a CUDA run (`device: cuda` in its metadata), and
Forge deliberately never moves a CUDA-saved model onto the CPU implicitly. The
committed consumer script therefore has to choose the device itself; before
I1 it did not, so on a machine without CUDA it died with a `PersistenceError`
traceback. It also printed non-ASCII box-drawing characters, which raised
`UnicodeEncodeError` whenever stdout was piped/redirected on Windows (cp1252).

These run the script as an external user would -- a separate process, a
working directory outside the repository, stdout captured through a pipe --
with `CUDA_VISIBLE_DEVICES=-1` so the CPU-only path is exercised even on a
machine that has a GPU. No CUDA hardware is needed or used.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "image_classifier"
_MODEL = _MODEL_DIR / "image_model.forge"
_SCRIPT = _MODEL_DIR / "predict.py"


def _make_image(path: Path) -> Path:
    pixels = (np.random.default_rng(0).random((80, 80, 3)) * 255).astype("uint8")
    Image.fromarray(pixels).save(path)
    return path


def _run(*args, cwd, cpu_only=True):
    env = dict(os.environ)
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *map(str, args)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
    )


def test_bundled_artifact_metadata_matches_its_documented_use():
    info = forge.inspect_model(str(_MODEL))
    assert info.task == "classification"
    assert info.classes == ["cat", "dog"]
    assert info.preprocessing


def test_consumer_script_predicts_on_a_machine_without_cuda(tmp_path):
    image = _make_image(tmp_path / "sample.png")
    result = _run(image, cwd=tmp_path)  # cwd outside the repository; stdout is a pipe
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Forge Image Classifier" in result.stdout
    assert "Prediction: cat" in result.stdout or "Prediction: dog" in result.stdout
    assert "Confidence:" in result.stdout
    result.stdout.encode("ascii")  # portable to any console encoding


def test_consumer_script_reports_a_missing_image_cleanly(tmp_path):
    result = _run(tmp_path / "nope.png", cwd=tmp_path)
    assert result.returncode == 1
    assert "Image not found" in result.stdout
    assert "Traceback" not in result.stderr


def test_default_load_on_a_machine_without_cuda_still_raises_a_clear_error(tmp_path):
    """The framework contract is unchanged: no silent CUDA->CPU move."""
    script = (
        "import forge\n"
        f"forge.load_predictor({str(_MODEL)!r})\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "PersistenceError" in result.stderr
    assert "device='cpu'" in result.stderr


def test_explicit_cpu_load_matches_on_cpu_only_and_gpu_machines(tmp_path):
    image = _make_image(tmp_path / "sample.png")
    prediction = forge.load_predictor(str(_MODEL), device="cpu").predict(image)
    assert prediction.label in ("cat", "dog")
    assert 0.0 <= prediction.confidence <= 1.0


@pytest.mark.skipif(
    not forge.cuda.is_cuda_available(), reason="CUDA is not available on this machine"
)
def test_explicit_cuda_load_still_works_and_agrees_with_cpu(tmp_path):
    image = _make_image(tmp_path / "sample.png")
    cpu = forge.load_predictor(str(_MODEL), device="cpu").predict(image)
    cuda = forge.load_predictor(str(_MODEL), device="cuda").predict(image)
    default = forge.load_predictor(str(_MODEL)).predict(image)  # recorded device: cuda
    assert cpu.label == cuda.label == default.label
    assert cuda.confidence == pytest.approx(cpu.confidence, abs=1e-4)
