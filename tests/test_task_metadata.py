"""Milestone 87 tests: explicit artifact task metadata and reliable
`forge.predict_model()` dispatch.

Covers `save_model(..., task=...)`'s validation (valid/invalid task values,
`task`/`classes` interaction), `inspect_model()`'s `ModelInfo.task` field
(including legacy artifacts with no `"task"` key and malformed/tampered
`"task"` values), `predict_model()`'s explicit-task-first dispatch
(`_determine_workflow()`/`_legacy_infer_workflow()`,
`forge/training/inference.py`) -- most importantly, that a classification
artifact saved with `classes=None` but an explicit `task="classification"`
is dispatched correctly rather than misidentified as regression (the M86
ambiguity this milestone closes) -- the `forge model inspect` CLI's "Task"
line/JSON field, and a genuine fresh-process check. See
`forge/serialization/model.py`'s `save_model()`/`inspect_model()` and
`docs/architecture/persistence.md`'s **Task metadata** section.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.cli.main import main as cli_main
from forge.data.transforms import Compose, Normalize, Resize
from forge.exceptions import DataError, PersistenceError
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.serialization import TASK_TYPES, inspect_model, load_classes, save_model
from forge.serialization.archive import METADATA_ENTRY
from forge.training import ClassificationPrediction, predict_artifact, predict_model
from forge.training.inference import predict_model as predict_model_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESIZE_SIZE = (8, 8)
_N_FEATURES = 4


def _tamper_metadata(path: Path, mutate) -> Path:
    with zipfile.ZipFile(str(path), "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    metadata = json.loads(entries[METADATA_ENTRY])
    metadata = mutate(metadata)
    entries[METADATA_ENTRY] = json.dumps(metadata).encode("utf-8")
    tampered = path.parent / f"tampered_{path.name}"
    with zipfile.ZipFile(str(tampered), "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return tampered


def _classification_transform():
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


def _classification_model(num_classes=2):
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, num_classes),
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


# -- save_model(..., task=...) validation -------------------------------------


def test_task_types_is_the_documented_vocabulary():
    # Milestone 90 added "sequence" as a fourth documented task -- see
    # docs/architecture/persistence.md's Task metadata section.
    assert TASK_TYPES == ("classification", "regression", "segmentation", "sequence")


@pytest.mark.parametrize("task", list(TASK_TYPES))
def test_save_model_accepts_each_documented_task(tmp_path, task):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    # task="sequence" requires classes= (its token vocabulary) -- see
    # tests/test_sequence_artifact_prediction.py for its own dedicated
    # coverage; every other task is validated with no classes= here.
    classes = ["a", "b", "c"] if task == "sequence" else None
    save_model(model, str(path), task=task, classes=classes)
    assert inspect_model(str(path)).task == task


def test_save_model_without_task_defaults_to_none(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))
    assert inspect_model(str(path)).task is None


def test_save_model_task_none_explicit_matches_default(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task=None)
    assert inspect_model(str(path)).task is None


@pytest.mark.parametrize("bad_task", ["Classification", "regression ", "detection", "", 1, ["classification"]])
def test_save_model_invalid_task_raises_before_writing(tmp_path, bad_task):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    with pytest.raises(PersistenceError):
        save_model(model, str(path), task=bad_task)
    assert not path.exists()


def test_save_model_classification_task_allows_classes_none(tmp_path):
    """The exact M86 gap: classification + classes=None is a real, valid
    state, and task= must not reject it."""
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="classification", classes=None)
    assert inspect_model(str(path)).task == "classification"
    assert load_classes(str(path)) is None


def test_save_model_classification_task_allows_classes_present(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="classification", classes=["cat", "dog", "bird"])
    assert inspect_model(str(path)).task == "classification"
    assert load_classes(str(path)) == ["cat", "dog", "bird"]


@pytest.mark.parametrize("task", ["regression", "segmentation"])
def test_save_model_rejects_classes_with_non_classification_task(tmp_path, task):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    with pytest.raises(PersistenceError):
        save_model(model, str(path), task=task, classes=["a", "b"])
    assert not path.exists()


@pytest.mark.parametrize("task", ["regression", "segmentation"])
def test_save_model_accepts_non_classification_task_without_classes(tmp_path, task):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task=task)
    assert inspect_model(str(path)).task == task
    assert load_classes(str(path)) is None


# -- inspect_model(): ModelInfo.task -------------------------------------------


def test_inspect_model_task_is_none_for_legacy_no_task_key(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    legacy = _tamper_metadata(path, lambda m: (m.pop("task", None), m)[1])
    assert inspect_model(str(legacy)).task is None
    # load_model() is unaffected by task metadata entirely.
    assert isinstance(forge.load_model(str(legacy)), Linear)


def test_inspect_model_raises_for_malformed_task_metadata(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    tampered = _tamper_metadata(path, lambda m: {**m, "task": "not-a-real-task"})
    with pytest.raises(PersistenceError):
        inspect_model(str(tampered))


def test_model_info_str_includes_task_line(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="regression")
    text = str(inspect_model(str(path)))
    assert "Task: regression" in text


def test_model_info_str_reports_unknown_task_for_legacy_artifact(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))
    text = str(inspect_model(str(path)))
    assert "Task: unknown" in text


# -- forge model inspect CLI ---------------------------------------------------


def test_cli_inspect_reports_task_line(tmp_path, capsys):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="segmentation")

    exit_code = cli_main(["model", "inspect", str(path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Task: segmentation" in out


def test_cli_inspect_json_reports_task_field(tmp_path, capsys):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="classification", classes=["a", "b"])

    exit_code = cli_main(["model", "inspect", str(path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "classification"


def test_cli_inspect_json_reports_null_task_for_legacy_artifact(tmp_path, capsys):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    exit_code = cli_main(["model", "inspect", str(path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] is None


def test_cli_convert_preserves_task(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="regression")
    output_path = tmp_path / "converted.forge"

    exit_code = cli_main(["model", "convert", str(path), "--device", "cpu", "--output", str(output_path)])
    assert exit_code == 0
    assert inspect_model(str(output_path)).task == "regression"


# -- predict_model() dispatch: explicit task is authoritative ------------------


def test_predict_model_dispatches_classification_without_classes_via_explicit_task(tmp_path):
    """The specific M86 bug M87 must eliminate: task=classification,
    classes=None must dispatch to classification, not regression, even
    though the architecture is Linear-terminated exactly like a regression
    model."""
    forge.random.seed(0)
    model = _classification_model(num_classes=2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=None, task="classification")
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    # Without the fix, this would raise DataError (misdispatched to
    # predict_tensor_artifact(), which rejects a path input) -- see
    # test_unified_artifact_prediction.py's own documented-limitation test
    # for the pre-M87/legacy-artifact behavior this is deliberately NOT.
    result = predict_model(str(path), str(image_path))
    assert isinstance(result, int)


def test_predict_model_dispatches_classification_with_classes_via_explicit_task(tmp_path):
    forge.random.seed(0)
    model = _classification_model(num_classes=2)
    path = tmp_path / "model.forge"
    save_model(
        model, str(path), preprocessing=_classification_transform(), classes=["cat", "dog"], task="classification",
    )
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_model(str(path), str(image_path))
    assert isinstance(result, ClassificationPrediction)
    assert result.label in ("cat", "dog")


def test_predict_model_dispatches_regression_via_explicit_task(tmp_path):
    forge.random.seed(1)
    model = _regression_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), task="regression")
    batch = np.random.default_rng(1).standard_normal((3, _N_FEATURES)).astype(np.float32)

    result = predict_model(str(path), batch)
    assert isinstance(result, forge.Tensor)
    assert result.shape == (3, 1)


def test_predict_model_dispatches_segmentation_via_explicit_task(tmp_path):
    forge.random.seed(2)
    model = _segmentation_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=_segmentation_transform(), task="segmentation")
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    result = predict_model(str(path), str(image_path))
    assert isinstance(result, forge.Tensor)
    assert result.shape == (1, 8, 8)


def test_predict_model_explicit_task_overrides_architecture_ambiguity(tmp_path):
    """A classification model with a Linear-terminated architecture (the
    exact architecture shape a legacy/no-task artifact cannot disambiguate
    from regression) still dispatches correctly to classification when
    task= is explicit -- proving task, not architecture, decides dispatch."""
    forge.random.seed(0)
    model = _classification_model(num_classes=2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=None, task="classification")

    info = inspect_model(str(path))
    assert "Linear" in info.model.module_types  # the architecture signal that would say "regression"
    assert info.task == "classification"  # but the explicit signal wins

    image_path = tmp_path / "query.png"
    _make_image(image_path)
    # predict_tensor_artifact() would accept a Tensor/ndarray, not a path --
    # confirming this really did dispatch through predict_artifact().
    with pytest.raises(DataError):
        predict_model(str(path), np.zeros((1, _N_FEATURES), dtype=np.float32))
    result = predict_model(str(path), str(image_path))
    assert isinstance(result, int)


def test_predict_model_matches_predict_artifact_for_explicit_task_artifact(tmp_path):
    forge.random.seed(5)
    model = _classification_model(num_classes=2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=None, task="classification")
    image_path = tmp_path / "query.png"
    _make_image(image_path, fill=180)

    unified = predict_model(str(path), str(image_path))
    direct = predict_artifact(str(path), str(image_path))
    assert unified == direct


# -- legacy fallback is unchanged and isolated ---------------------------------


def test_predict_model_legacy_no_task_still_uses_architecture_heuristic(tmp_path):
    """A genuinely legacy artifact (no task=, the pre-M87 shape) keeps
    exactly the M86 behavior -- classes present -> classification."""
    forge.random.seed(0)
    model = _classification_model(num_classes=2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=["cat", "dog"])
    assert inspect_model(str(path)).task is None

    image_path = tmp_path / "query.png"
    _make_image(image_path)
    result = predict_model(str(path), str(image_path))
    assert isinstance(result, ClassificationPrediction)


def test_predict_model_legacy_classification_without_classes_still_misdispatches(tmp_path):
    """Pinned, documented pre-existing limitation (Milestone 86): with no
    explicit task and no classes, a classification model is architecturally
    indistinguishable from regression, so it is still misidentified here --
    exactly the behavior tests/test_unified_artifact_prediction.py already
    pins for the legacy shape. This is the one case Milestone 87 could not
    retroactively fix without the caller supplying task= at save time."""
    forge.random.seed(0)
    model = _classification_model(num_classes=2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=None)
    assert inspect_model(str(path)).task is None
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    with pytest.raises(DataError):
        predict_model(str(path), str(image_path))


def test_predict_model_raises_clearly_for_undeterminable_legacy_artifact(tmp_path):
    forge.random.seed(0)
    model = Sequential(ReLU())
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    with pytest.raises(PersistenceError, match="no explicit task metadata"):
        predict_model(str(path), np.zeros((1, _N_FEATURES), dtype=np.float32))


# -- fresh process --------------------------------------------------------------


def test_predict_model_dispatches_via_explicit_task_from_a_genuinely_separate_process(tmp_path):
    forge.random.seed(7)
    model = _classification_model(num_classes=2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=_classification_transform(), classes=None, task="classification")
    image_path = tmp_path / "query.png"
    _make_image(image_path, size=(12, 9))

    script = (
        "import forge\n"
        f"result = forge.predict_model({str(path)!r}, {str(image_path)!r})\n"
        "print(f'index={result}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"

    expected = predict_model(str(path), str(image_path))
    assert f"index={expected}" in proc.stdout


# -- basic re-export shape -------------------------------------------------------


def test_predict_model_still_reexported_consistently():
    assert forge.predict_model is predict_model
    assert predict_model is predict_model_direct
