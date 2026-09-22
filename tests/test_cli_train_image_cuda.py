"""Milestone 122 CUDA test: `forge model train DIR --task image-classification --device cuda`.

Mirrors `tests/test_cli_train_image.py`'s CPU coverage for the device-specific behavior only: the
CLI forwards `--device cuda` to `forge.train_image_classifier()` and the resulting artifact really
trained on CUDA (not a silent CPU fallback), consumable afterward through `forge model predict` on
CPU. Skips cleanly without a working CUDA backend, matching every other `*_cuda.py` test in this
suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import forge
from forge.backend.cuda import is_cuda_available
from forge.cli.main import main

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


def _make_image(path: Path, size=(64, 64), color=(10, 20, 30)) -> None:
    Image.new("RGB", size, color).save(path)


def _make_two_class_root(tmp_path: Path, per_class: int = 8, size=(64, 64)) -> Path:
    root = tmp_path / "data"
    for cls, base_color in [("cat", (200, 0, 0)), ("dog", (0, 200, 0))]:
        class_dir = root / cls
        class_dir.mkdir(parents=True)
        for i in range(per_class):
            _make_image(class_dir / f"{cls}_{i}.png", size=size, color=(base_color[0], base_color[1], i * 5))
    return root


def run(argv, capsys):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_trains_an_image_classifier_on_cuda(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    output = tmp_path / "pets.forge"
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--output", str(output),
         "--epochs", "2", "--batch-size", "4", "--seed", "0", "--device", "cuda"],
        capsys,
    )
    assert code == 0 and err == ""
    assert output.is_file()
    assert forge.inspect_model(str(output)).task == "classification"

    import json as _json
    import zipfile
    with zipfile.ZipFile(str(output)) as zf:
        saved_metadata = _json.loads(zf.read("metadata.json"))
    assert saved_metadata["device"] == "cuda"

    # Consumption on CPU must work with no reference to the CUDA training run.
    predictor = forge.load_predictor(str(output), device="cpu")
    assert predictor.classes == ["cat", "dog"]
    sample_image = next((root / "cat").glob("*.png"))
    predict_code, predict_out, predict_err = run(["model", "predict", str(output), str(sample_image)], capsys)
    assert predict_code == 0 and predict_err == ""
    assert "Prediction:" in predict_out
