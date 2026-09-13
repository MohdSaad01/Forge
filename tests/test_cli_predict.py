"""Milestone 88 tests: `forge model predict`, made task-aware.

Covers task-based command routing (classification/regression/segmentation),
each task's input parsing (image files; JSON numeric data, including
malformed/non-numeric/missing input), `--output` for segmentation,
`--json` machine-readable output, `--device`, missing/unsupported/absent
task metadata, ordinary file-not-found errors, a genuine fresh-process
(subprocess) invocation, and CUDA device handling. See
`forge/cli/model.py::cmd_predict` and
`docs/development/m88-unified-artifact-prediction-cli.md`.

`tests/test_classification_metadata.py` keeps the CLI's original
classification-specific tests (label/confidence text, classes=None, missing
preprocessing) -- this file covers the new task-dispatch behavior Milestone
88 adds on top, including the two non-classification tasks the CLI never
supported before.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.backend.cuda import is_cuda_available
from forge.cli.main import main as cli_main
from forge.data.transforms import Compose, Normalize, Resize
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.serialization import save_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESIZE_SIZE = (8, 8)
_N_FEATURES = 4


# -- shared fixtures, mirroring tests/test_unified_artifact_prediction.py -----


def _classification_transform():
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


def _classification_model(num_classes=2):
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, num_classes),
    )


def _regression_transform():
    return Normalize(
        mean=np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
        std=np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32),
    )


def _regression_model():
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, 1))


def _segmentation_transform():
    return Compose([Normalize(mean=0.0, std=255.0)])


def _segmentation_model():
    return Sequential(Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), Conv2d(4, 1, kernel_size=3, padding=1))


def _make_image(path: Path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _saved_classification_model(tmp_path, *, classes=("cat", "dog"), task="classification", seed=0) -> Path:
    forge.random.seed(seed)
    model = _classification_model(num_classes=len(classes) if classes else 2)
    path = tmp_path / "classification.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=list(classes) if classes else None, task=task)
    return path


def _saved_regression_model(tmp_path, *, task="regression", seed=0) -> Path:
    forge.random.seed(seed)
    model = _regression_model()
    path = tmp_path / "regression.forge"
    save_model(model, str(path), preprocessing=_regression_transform(), task=task)
    return path


def _saved_segmentation_model(tmp_path, *, task="segmentation", seed=0) -> Path:
    forge.random.seed(seed)
    model = _segmentation_model()
    path = tmp_path / "segmentation.forge"
    save_model(model, str(path), preprocessing=_segmentation_transform(), task=task)
    return path


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# -- command routing ----------------------------------------------------------


def test_cli_predict_routes_classification_artifact(tmp_path, capsys):
    model_path = _saved_classification_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), str(image_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Prediction:" in out
    assert "Confidence:" in out


def test_cli_predict_routes_regression_artifact(tmp_path, capsys):
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.2, 3.4, 5.6, 7.8])

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.startswith("Prediction:")


def test_cli_predict_routes_segmentation_artifact(tmp_path, capsys):
    model_path = _saved_segmentation_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)
    output_path = tmp_path / "mask.png"

    exit_code = cli_main(["model", "predict", str(model_path), str(image_path), "--output", str(output_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"Predicted mask saved to: {output_path}" in out
    assert output_path.is_file()


# -- regression input parsing --------------------------------------------------


def test_cli_predict_regression_accepts_flat_list_as_one_sample(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.2, 3.4, 5.6, 7.8])

    from forge.cli.model import _parse_regression_input
    array = _parse_regression_input(str(input_path))
    assert array.shape == (1, 4)


def test_cli_predict_regression_accepts_nested_list_as_already_batched(tmp_path):
    from forge.cli.model import _parse_regression_input

    input_path = tmp_path / "input.json"
    _write_json(input_path, [[1.2, 3.4, 0.1, 0.2], [5.6, 7.8, 0.3, 0.4]])
    array = _parse_regression_input(str(input_path))
    assert array.shape == (2, 4)


def test_cli_predict_regression_accepts_utf8_bom(tmp_path):
    """Milestone 89 external-workflow finding: Windows tools (PowerShell's
    `Out-File`/`>`, Notepad's "UTF-8" save option) commonly write a leading
    UTF-8 BOM. A numerically valid JSON file with a BOM must parse the same
    as one without -- not fail with the generic "not numeric JSON" error."""
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b"\xef\xbb\xbf" + json.dumps([1.2, 3.4, 5.6, 7.8]).encode("utf-8"))

    from forge.cli.model import _parse_regression_input
    array = _parse_regression_input(str(input_path))
    assert array.shape == (1, 4)
    np.testing.assert_allclose(array, [[1.2, 3.4, 5.6, 7.8]], rtol=1e-6)


def test_cli_predict_regression_malformed_json_fails_clearly(tmp_path, capsys):
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    input_path.write_text("{not valid json", encoding="utf-8")

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "numeric JSON data" in err


def test_cli_predict_regression_nonnumeric_json_fails_clearly(tmp_path, capsys):
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, ["cat", "dog"])

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "numeric JSON data" in err


def test_cli_predict_regression_given_an_image_fails_clearly(tmp_path, capsys):
    """Section 10's "Invalid input type" case: a non-JSON file given to a
    regression artifact must produce a clear message, not a traceback."""
    model_path = _saved_regression_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), str(image_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "numeric JSON data" in err


def test_cli_predict_missing_input_file_fails(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    exit_code = cli_main(["model", "predict", str(model_path), str(tmp_path / "nope.json")])
    assert exit_code == 1


# -- segmentation output --------------------------------------------------------


def test_cli_predict_segmentation_requires_output(tmp_path, capsys):
    model_path = _saved_segmentation_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), str(image_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "--output" in err


def test_cli_predict_segmentation_invalid_output_directory_fails_clearly(tmp_path, capsys):
    model_path = _saved_segmentation_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)
    output_path = tmp_path / "does_not_exist" / "mask.png"

    exit_code = cli_main(["model", "predict", str(model_path), str(image_path), "--output", str(output_path)])
    assert exit_code == 1
    assert "Error" in capsys.readouterr().err


# -- metadata: explicit task / missing task / unsupported task ----------------


def test_cli_predict_missing_artifact_fails_clearly(tmp_path, capsys):
    exit_code = cli_main(["model", "predict", str(tmp_path / "nope.forge"), str(tmp_path / "x.png")])
    assert exit_code == 1
    assert "artifact not found" in capsys.readouterr().err


def test_cli_predict_legacy_no_task_fails_clearly(tmp_path, capsys):
    model_path = _saved_regression_model(tmp_path, task=None)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.0, 2.0, 3.0, 4.0])

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "does not declare a task" in err
    assert "resave" in err.lower()


def test_cli_predict_unsupported_task_fails_clearly(tmp_path, capsys, monkeypatch):
    """A hypothetical future task value `inspect_model()` would accept but
    this command has not been taught to route -- simulated by monkeypatching
    the CLI's own `inspect_model` call, since a real artifact can never carry
    a task outside `TASK_TYPES` (`inspect_model()` itself rejects that)."""
    from dataclasses import dataclass

    import forge.cli.model as cli_model

    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.0, 2.0, 3.0, 4.0])

    @dataclass
    class _FakeInfo:
        task: str

    monkeypatch.setattr(cli_model, "inspect_model", lambda path: _FakeInfo(task="clustering"))

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "unsupported artifact task" in err
    assert "clustering" in err


# -- --json output --------------------------------------------------------------


def test_cli_predict_classification_json_output(tmp_path, capsys):
    model_path = _saved_classification_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), str(image_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "classification"
    assert payload["class"] in ("cat", "dog")
    assert 0.0 <= payload["confidence"] <= 1.0


def test_cli_predict_classification_without_classes_json_output(tmp_path, capsys):
    model_path = _saved_classification_model(tmp_path, classes=None)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), str(image_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "classification"
    assert payload["class"] is None
    assert isinstance(payload["index"], int)
    assert payload["confidence"] is None


def test_cli_predict_regression_json_output(tmp_path, capsys):
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.2, 3.4, 5.6, 7.8])

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "regression"
    assert isinstance(payload["prediction"], list)


def test_cli_predict_segmentation_json_output(tmp_path, capsys):
    model_path = _saved_segmentation_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)
    output_path = tmp_path / "mask.png"

    exit_code = cli_main(
        ["model", "predict", str(model_path), str(image_path), "--output", str(output_path), "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"task": "segmentation", "output_path": str(output_path)}


# -- device -----------------------------------------------------------------


def test_cli_predict_explicit_cpu_device(tmp_path, capsys):
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.0, 2.0, 3.0, 4.0])

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path), "--device", "cpu"])
    assert exit_code == 0


def test_cli_predict_cuda_unavailable_returns_clear_error(tmp_path, capsys, monkeypatch):
    import forge.backend.cuda as cuda_module

    monkeypatch.setattr(cuda_module, "is_cuda_available", lambda: False)
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.0, 2.0, 3.0, 4.0])

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path), "--device", "cuda"])
    assert exit_code == 1
    assert "CUDA" in capsys.readouterr().err


# -- fresh process ------------------------------------------------------------


def test_cli_predict_fresh_process_all_three_task_shapes(tmp_path):
    classification_path = _saved_classification_model(tmp_path, seed=10)
    image_path = tmp_path / "query.png"
    _make_image(image_path, size=(30, 20))

    regression_path = _saved_regression_model(tmp_path, seed=11)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [0.5, -0.5, 0.25, 0.1])

    segmentation_path = _saved_segmentation_model(tmp_path, seed=13)
    seg_image_path = tmp_path / "seg_query.png"
    _make_image(seg_image_path, size=(8, 8), fill=210)
    mask_path = tmp_path / "mask.png"

    for args in (
        [str(classification_path), str(image_path)],
        [str(regression_path), str(input_path)],
        [str(segmentation_path), str(seg_image_path), "--output", str(mask_path)],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "forge", "model", "predict", *args],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"subprocess failed for {args}:\n{result.stderr}"

    assert mask_path.is_file()


# -- CUDA-hardware-verified (skips cleanly without CUDA) -----------------------


pytestmark_cuda = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytestmark_cuda
def test_cli_predict_reaches_cuda_inference_path(tmp_path, capsys):
    model_path = _saved_regression_model(tmp_path)
    input_path = tmp_path / "input.json"
    _write_json(input_path, [1.0, 2.0, 3.0, 4.0])

    exit_code = cli_main(["model", "predict", str(model_path), str(input_path), "--device", "cuda"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.startswith("Prediction:")
