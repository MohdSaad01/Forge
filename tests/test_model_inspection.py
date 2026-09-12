"""Milestone 85 tests: `forge.inspect_model()`, the portable-artifact
inspection API.

Covers the structured `ModelInfo`/`ModelSummary`/`PreprocessingInfo` result
returned for artifacts with/without preprocessing, with/without classes, and
legacy (pre-M71/M72) archives; error handling for missing/corrupt/malformed
files; that inspection never reconstructs a live model, never requires CUDA,
and never mutates `forge.random` state; `forge model inspect` CLI delegation;
and real-consumer verification against the three existing artifact
workflows (classification, regression, segmentation) plus a genuine
fresh-process check. See `forge/serialization/model.py::inspect_model()` and
`docs/architecture/persistence.md`'s **Model inspection** section.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.cli.main import main as cli_main
from forge.data.transforms import Compose, Flatten as DataFlatten, Normalize, Reshape, Resize, ToTensor
from forge.exceptions import PersistenceError
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.serialization import (
    ModelInfo,
    ModelSummary,
    PreprocessingInfo,
    inspect_model,
    load_model,
    save_model,
)
from forge.serialization.archive import METADATA_ENTRY
from forge.serialization.model import inspect_model as inspect_model_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _tamper_metadata(path: Path, mutate) -> Path:
    """Rewrite `path`'s metadata.json via `mutate(dict) -> dict`, matching
    `tests/test_serialization.py`'s/`test_preprocessing_persistence.py`'s own
    tampering helper convention."""
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


# -- basic API shape ----------------------------------------------------------


def test_inspect_model_is_reexported_consistently():
    assert forge.inspect_model is inspect_model
    assert forge.serialization.inspect_model is inspect_model
    assert inspect_model is inspect_model_direct


def test_inspect_model_returns_dataclass_instances(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    info = inspect_model(str(path))
    assert isinstance(info, ModelInfo)
    assert isinstance(info.model, ModelSummary)
    assert info.preprocessing is None
    assert info.classes is None


def test_inspect_model_basic_fields_no_preprocessing_no_classes(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    info = inspect_model(str(path))
    assert info.model.type == "Linear"
    expected_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    assert info.model.parameter_count == expected_params
    assert info.preprocessing is None
    assert info.classes is None
    assert info.format_version == 2
    assert info.device == "cpu"


# -- preprocessing --------------------------------------------------------


def test_inspect_model_with_preprocessing_describes_the_pipeline(tmp_path):
    model = Linear(4, 2)
    pre = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=pre)

    info = inspect_model(str(path))
    assert info.preprocessing is not None
    assert isinstance(info.preprocessing, PreprocessingInfo)
    assert "Resize" in info.preprocessing.description
    assert "Normalize" in info.preprocessing.description
    assert info.preprocessing.description.index("Resize") < info.preprocessing.description.index("Normalize")
    assert " -> " in info.preprocessing.description

    # The reconstructed transform is usable programmatically, not just for display.
    x = Tensor(np.full((3, 20, 24), 180.0, dtype=np.float32))
    np.testing.assert_allclose(pre(x).numpy(), info.preprocessing.transform(x).numpy(), atol=1e-6)


def test_inspect_model_single_transform_description(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=Normalize(mean=0.0, std=1.0))

    info = inspect_model(str(path))
    assert info.preprocessing.description == "Normalize(mean=0.0, std=1.0)"


@pytest.mark.parametrize(
    "transform",
    [Reshape(2, 6), DataFlatten(), ToTensor(dtype="float32", device="cpu")],
)
def test_inspect_model_every_registered_transform_has_a_readable_repr(tmp_path, transform):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=transform)

    info = inspect_model(str(path))
    assert "object at 0x" not in info.preprocessing.description


def test_inspect_model_without_preprocessing_is_none(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path))
    assert inspect_model(str(path)).preprocessing is None


# -- classes ----------------------------------------------------------------


def test_inspect_model_with_classes(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), classes=["cat", "dog", "bird"])
    assert inspect_model(str(path)).classes == ["cat", "dog", "bird"]


def test_inspect_model_without_classes_is_none(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))
    assert inspect_model(str(path)).classes is None


def test_inspect_model_distinguishes_classification_vs_non_classification_shapes(tmp_path):
    classification_path = tmp_path / "classification.forge"
    save_model(
        Linear(4, 2), str(classification_path),
        preprocessing=Normalize(mean=0.0, std=1.0), classes=["a", "b"],
    )
    regression_shaped_path = tmp_path / "regression.forge"
    save_model(Linear(4, 1), str(regression_shaped_path), preprocessing=Normalize(mean=0.0, std=1.0))

    classification_info = inspect_model(str(classification_path))
    regression_info = inspect_model(str(regression_shaped_path))

    assert classification_info.preprocessing is not None and classification_info.classes is not None
    assert regression_info.preprocessing is not None and regression_info.classes is None


# -- module tree identification ---------------------------------------------


def test_inspect_model_module_types_lists_the_full_tree_in_order(tmp_path):
    model = Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, 2),
    )
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    info = inspect_model(str(path))
    assert info.model.type == "Sequential"
    assert info.model.module_types[0] == "Sequential"
    assert sorted(info.model.module_types[1:]) == sorted(["Conv2d", "ReLU", "MaxPool2d", "Flatten", "Linear"])
    expected_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    assert info.model.parameter_count == expected_params


# -- str() rendering ----------------------------------------------------------


def test_model_info_str_contains_expected_lines(tmp_path):
    model = Linear(4, 2)
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=pre, classes=["cat", "dog"])

    text = str(inspect_model(str(path)))
    assert "Model: Linear" in text
    assert "Input preprocessing: Resize" in text
    assert "Classes: cat, dog" in text
    assert "Artifact format: version 2" in text


def test_model_info_str_shows_none_for_absent_preprocessing_and_classes(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path))
    text = str(inspect_model(str(path)))
    assert "Input preprocessing: none" in text
    assert "Classes: none" in text


# -- backward compatibility (legacy archives) --------------------------------


def test_inspect_model_legacy_archive_with_no_preprocessing_or_classes_keys(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    def _strip(metadata):
        del metadata["preprocessing"]
        del metadata["classes"]
        return metadata

    legacy = _tamper_metadata(path, _strip)
    info = inspect_model(str(legacy))
    assert info.preprocessing is None
    assert info.classes is None
    assert info.model.type == "Linear"
    # load_model() must remain unaffected -- inspect_model() reads the same
    # legacy file load_model()/load_preprocessing()/load_classes() already
    # tolerate.
    assert isinstance(load_model(str(legacy)), Linear)


def test_inspect_model_does_not_require_the_root_module_type_to_be_registered(tmp_path):
    """Unlike load_model(), which must reconstruct a live Module (and thus
    needs the type registered in this process), inspect_model() only reads
    metadata -- an unregistered/unknown root type must not block it."""
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    def _rename_root_type(metadata):
        metadata["root"]["type"] = "SomeModelTypeNeverRegisteredInThisProcess"
        return metadata

    tampered = _tamper_metadata(path, _rename_root_type)

    info = inspect_model(str(tampered))
    assert info.model.type == "SomeModelTypeNeverRegisteredInThisProcess"

    with pytest.raises(PersistenceError):
        load_model(str(tampered))


# -- no CUDA requirement, no RNG mutation -------------------------------------


def test_inspect_model_does_not_mutate_random_state(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)]),
               classes=["a", "b"])

    forge.random.seed(123)
    before = forge.random.get_state()
    inspect_model(str(path))
    after = forge.random.get_state()
    assert before == after


def test_inspect_model_works_on_a_cuda_saved_artifact_description_without_cuda(tmp_path, monkeypatch):
    """inspect_model() must never require CUDA, even for a CUDA-recorded
    archive -- mirroring `forge/cli/_archive_info.py`'s own documented reason
    for reading metadata.json directly rather than through load_model()."""
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    def _mark_cuda(metadata):
        metadata["device"] = "cuda"
        return metadata

    tampered = _tamper_metadata(path, _mark_cuda)
    info = inspect_model(str(tampered))
    assert info.device == "cuda"


# -- error handling -----------------------------------------------------------


def test_inspect_model_missing_file_raises():
    with pytest.raises(PersistenceError):
        inspect_model("does_not_exist.forge")


def test_inspect_model_corrupt_zip_raises(tmp_path):
    bad = tmp_path / "bad.forge"
    bad.write_bytes(b"not a real forge archive")
    with pytest.raises(PersistenceError):
        inspect_model(str(bad))


def test_inspect_model_unsupported_format_version_raises(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    tampered = _tamper_metadata(path, lambda m: {**m, "forge_format_version": 999})
    with pytest.raises(PersistenceError, match="format version"):
        inspect_model(str(tampered))


def test_inspect_model_malformed_classes_raises(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    tampered = _tamper_metadata(path, lambda m: {**m, "classes": {"not": "a list"}})
    with pytest.raises(PersistenceError):
        inspect_model(str(tampered))


def test_inspect_model_missing_root_raises(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    def _drop_root(metadata):
        del metadata["root"]
        return metadata

    tampered = _tamper_metadata(path, _drop_root)
    with pytest.raises(PersistenceError):
        inspect_model(str(tampered))


# -- CLI delegation ------------------------------------------------------------


def test_cli_inspect_reports_preprocessing_detail_matching_the_public_api(tmp_path, capsys):
    model = Linear(4, 2)
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=pre, classes=["cat", "dog"])

    expected = inspect_model(str(path)).preprocessing.description

    exit_code = cli_main(["model", "inspect", str(path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"Preprocessing detail: {expected}" in out
    # Existing, previously-tested lines must be unchanged.
    assert "Preprocessing: yes" in out
    assert "Classes: cat, dog" in out


def test_cli_inspect_json_includes_preprocessing_description(tmp_path, capsys):
    model = Linear(4, 2)
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=pre)

    expected = inspect_model(str(path)).preprocessing.description
    exit_code = cli_main(["model", "inspect", str(path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["preprocessing_description"] == expected
    assert payload["has_preprocessing"] is True


def test_cli_inspect_no_preprocessing_detail_line_when_absent(tmp_path, capsys):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    exit_code = cli_main(["model", "inspect", str(path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Preprocessing detail:" not in out

    exit_code = cli_main(["model", "inspect", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["preprocessing_description"] is None


# -- fresh process ------------------------------------------------------------


def test_inspect_model_works_from_a_genuinely_separate_process(tmp_path):
    """Built entirely from pre-registered `forge.nn` types (mirroring
    `tests/test_artifact_inference.py`'s own fresh-process test): a real,
    separate OS process must be able to inspect the artifact from the file
    alone, with no shared import state or module cache with this test
    process."""
    forge.random.seed(0)
    model = Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, 2),
    )
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=pre, classes=["cat", "dog"])

    script = (
        "import forge\n"
        f"info = forge.inspect_model({str(path)!r})\n"
        "print(f'type={info.model.type} params={info.model.parameter_count} "
        "classes={info.classes} format_version={info.format_version} device={info.device}')\n"
        "print(f'preprocessing={info.preprocessing.description}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    expected = inspect_model(str(path))
    assert (
        f"type={expected.model.type} params={expected.model.parameter_count} "
        f"classes={expected.classes} format_version={expected.format_version} device={expected.device}"
    ) in result.stdout
    assert f"preprocessing={expected.preprocessing.description}" in result.stdout


# -- real consumers: the three existing artifact workflows (M82-84) ----------


def test_inspect_model_on_a_real_regression_artifact(tmp_path):
    from examples.regression.train import main as regression_train_main

    out_dir = tmp_path / "artifacts"
    regression_train_main([
        "--n-train", "60", "--n-val", "20", "--n-test", "20", "--batch-size", "16",
        "--epochs", "1", "--seed", "1", "--output-dir", str(out_dir),
    ])
    model_path = out_dir / "regression_model.forge"
    assert model_path.is_file()

    info = inspect_model(str(model_path))
    assert info.preprocessing is not None
    assert "Normalize" in info.preprocessing.description
    assert info.classes is None
    assert info.model.parameter_count > 0


def test_inspect_model_on_a_real_segmentation_artifact(tmp_path):
    from examples.segmentation.train import main as segmentation_train_main

    out_dir = tmp_path / "artifacts"
    segmentation_train_main([
        "--n-train", "80", "--n-test", "20", "--batch-size", "16", "--epochs", "1",
        "--seed", "1", "--output-dir", str(out_dir),
    ])
    model_path = out_dir / "segmentation_model.forge"
    assert model_path.is_file()

    info = inspect_model(str(model_path))
    assert info.preprocessing is not None
    assert info.classes is None
    assert info.model.type == "Sequential"


def test_inspect_model_on_a_real_classification_artifact(tmp_path):
    from examples.image_folder_classification import train as image_folder_train_module

    out_dir = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    image_folder_train_module.main([
        "--data-root", str(data_root), "--generate", "--samples-per-class", "15",
        "--min-size", "24", "--max-size", "48", "--device", "cpu",
        "--epochs", "1", "--batch-size", "8", "--seed", "1", "--output-dir", str(out_dir),
    ])
    model_path = out_dir / "image_folder_model.forge"
    assert model_path.is_file()

    info = inspect_model(str(model_path))
    assert info.preprocessing is not None
    assert "Resize" in info.preprocessing.description
    assert "Normalize" in info.preprocessing.description
    assert info.classes is not None and len(info.classes) >= 2


def test_inspect_model_distinguishes_all_three_real_workflows_metadata_shape(tmp_path):
    """The literal M85 acceptance shape: three real artifact-producing
    workflows, one inspection API, correctly distinguished metadata --
    none of it hard-coded into inspect_model() itself."""
    from examples.image_folder_classification import train as image_folder_train_module
    from examples.regression.train import main as regression_train_main
    from examples.segmentation.train import main as segmentation_train_main

    classification_dir = tmp_path / "classification"
    image_folder_train_module.main([
        "--data-root", str(tmp_path / "data"), "--generate", "--samples-per-class", "15",
        "--min-size", "24", "--max-size", "48", "--device", "cpu",
        "--epochs", "1", "--batch-size", "8", "--seed", "2", "--output-dir", str(classification_dir),
    ])
    regression_dir = tmp_path / "regression"
    regression_train_main([
        "--n-train", "60", "--n-val", "20", "--n-test", "20", "--batch-size", "16",
        "--epochs", "1", "--seed", "2", "--output-dir", str(regression_dir),
    ])
    segmentation_dir = tmp_path / "segmentation"
    segmentation_train_main([
        "--n-train", "80", "--n-test", "20", "--batch-size", "16", "--epochs", "1",
        "--seed", "2", "--output-dir", str(segmentation_dir),
    ])

    classification_info = inspect_model(str(classification_dir / "image_folder_model.forge"))
    regression_info = inspect_model(str(regression_dir / "regression_model.forge"))
    segmentation_info = inspect_model(str(segmentation_dir / "segmentation_model.forge"))

    assert classification_info.preprocessing is not None and classification_info.classes is not None
    assert regression_info.preprocessing is not None and regression_info.classes is None
    assert segmentation_info.preprocessing is not None and segmentation_info.classes is None
