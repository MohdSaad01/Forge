"""Milestone 72 tests: class-label metadata persistence and prediction interpretation.

Covers `save_model(..., classes=...)`/`load_classes()` (the model-file-level
API, mirroring `tests/test_preprocessing_persistence.py`'s own conventions
for `preprocessing=`/`load_preprocessing()`), `forge.training.
interpret_classification()` (turning a raw prediction `Tensor` into a
human-readable `ClassificationPrediction`), the `forge model predict` CLI
command, and a real end-to-end `ImageFolder` -> train -> save -> fresh-process
`infer.py` workflow with no `classes.json` sidecar file. See
`docs/architecture/persistence.md`'s **Class-label metadata** section and
`docs/development/m72-classification-metadata.md`.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge import Tensor
from forge.cli.main import main as cli_main
from forge.data.transforms import Compose, Normalize, Resize
from forge.exceptions import PersistenceError, TrainerError
from forge.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, MaxPool2d, Module, ReLU
from forge.optim import Adam
from forge.serialization import load_classes, load_model, load_preprocessing, save_model
from forge.serialization.archive import METADATA_ENTRY
from forge.training import ClassificationPrediction, Trainer, interpret_classification, predict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# -- save_model(..., classes=...) / load_classes() ------------------------


def test_save_load_classes_round_trip(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), classes=["cat", "dog", "bird"])
    assert load_classes(str(path)) == ["cat", "dog", "bird"]
    # load_model() is unaffected -- still returns a bare, working Module.
    assert isinstance(load_model(str(path)), Linear)


def test_save_model_without_classes_defaults_to_none(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))
    assert load_classes(str(path)) is None


def test_save_model_classes_none_explicit_matches_default(tmp_path):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path), classes=None)
    assert load_classes(str(path)) is None


def test_save_model_and_preprocessing_and_classes_together(tmp_path):
    model = Linear(4, 3)
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=pre, classes=["a", "b", "c"])
    assert load_classes(str(path)) == ["a", "b", "c"]
    assert load_preprocessing(str(path)) is not None


@pytest.mark.parametrize(
    "bad_classes",
    [
        [],
        "cat,dog",
        ["cat", 2],
        ["cat", ""],
        ["cat", "  "],
        ["cat", "dog", "cat"],
        {"0": "cat"},
    ],
)
def test_save_model_invalid_classes_raises_before_writing(tmp_path, bad_classes):
    model = Linear(4, 3)
    path = tmp_path / "model.forge"
    with pytest.raises(PersistenceError):
        save_model(model, str(path), classes=bad_classes)
    assert not path.exists()


def test_backward_compatible_file_has_no_classes_key(tmp_path):
    """A file saved with no `classes=` argument at all (the pre-M72 call
    shape) must still load cleanly through both `load_model()` and the new
    `load_classes()` -- the latter returning `None`, not raising."""
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    with zipfile.ZipFile(str(path), "r") as zf:
        metadata = json.loads(zf.read(METADATA_ENTRY))
    assert metadata["classes"] is None

    # Simulate a genuinely pre-M72 file (no "classes" key present at all).
    del metadata["classes"]
    legacy = tmp_path / "legacy.forge"
    with zipfile.ZipFile(str(path), "r") as src, zipfile.ZipFile(str(legacy), "w") as dst:
        for item in src.infolist():
            if item.filename == METADATA_ENTRY:
                dst.writestr(item, json.dumps(metadata))
            else:
                dst.writestr(item, src.read(item.filename))

    assert load_classes(str(legacy)) is None
    reloaded = load_model(str(legacy))
    assert isinstance(reloaded, Linear)


def test_load_classes_missing_file_raises():
    with pytest.raises(PersistenceError):
        load_classes("does_not_exist.forge")


def test_load_classes_malformed_metadata_raises(tmp_path):
    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    with zipfile.ZipFile(str(path), "r") as zf:
        metadata = json.loads(zf.read(METADATA_ENTRY))
    metadata["classes"] = {"not": "a list"}
    with zipfile.ZipFile(str(path), "r") as src, zipfile.ZipFile(str(tmp_path / "bad.forge"), "w") as dst:
        for item in src.infolist():
            if item.filename == METADATA_ENTRY:
                dst.writestr(item, json.dumps(metadata))
            else:
                dst.writestr(item, src.read(item.filename))

    with pytest.raises(PersistenceError):
        load_classes(str(tmp_path / "bad.forge"))


# -- interpret_classification() --------------------------------------------


def test_interpret_classification_basic():
    # Row 0: class 1 clearly dominant. Row 1: class 0 clearly dominant.
    output = Tensor([[0.0, 5.0, 0.0], [3.0, 0.0, 0.0]])
    results = interpret_classification(output, ["cat", "dog", "bird"])
    assert len(results) == 2
    assert all(isinstance(r, ClassificationPrediction) for r in results)
    assert results[0].label == "dog"
    assert results[0].index == 1
    assert results[1].label == "cat"
    assert results[1].index == 0
    # A dominant logit (5.0 vs 0.0) should produce high confidence.
    assert results[0].confidence > 0.95
    assert 0.0 <= results[1].confidence <= 1.0


def test_interpret_classification_confidence_matches_manual_softmax():
    logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    output = Tensor(logits)
    result = interpret_classification(output, ["a", "b", "c"])[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    assert result.index == 1
    np.testing.assert_allclose(result.confidence, expected[0, 1], atol=1e-6)


def test_interpret_classification_wrong_output_dim_raises():
    output = Tensor([[1.0, 2.0, 3.0]])  # 3 class scores
    with pytest.raises(TrainerError, match="dimension"):
        interpret_classification(output, ["cat", "dog"])  # only 2 labels


def test_interpret_classification_non_2d_output_raises():
    output = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(TrainerError):
        interpret_classification(output, ["a", "b", "c"])


def test_interpret_classification_empty_classes_raises():
    output = Tensor([[1.0, 2.0]])
    with pytest.raises(TrainerError):
        interpret_classification(output, [])


def test_interpret_classification_accepts_plain_list_not_just_saved_classes():
    # classes need not come from load_classes() -- any list[str] works,
    # e.g. ImageFolder.classes directly.
    output = Tensor([[2.0, 1.0]])
    results = interpret_classification(output, ["x", "y"])
    assert results[0].label == "x"


# -- forge model predict CLI ------------------------------------------------


class _TinyCNN(Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.fc = Linear(4 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(self.flatten(x))


from forge.serialization import register_module  # noqa: E402  (after class def, mirrors test_preprocessing_persistence.py)

register_module(
    "_TinyCNN_M72Test",
    _TinyCNN,
    get_config=lambda m: {"num_classes": m.fc.out_features},
)


def _make_image(path: Path, size=(8, 8), fill=100):
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _saved_tiny_model(tmp_path, *, preprocessing=None, classes=None):
    forge.random.seed(0)
    model = _TinyCNN(num_classes=len(classes) if classes else 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing, classes=classes)
    return path


def test_cli_predict_prints_label_and_confidence(tmp_path, capsys):
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    model_path = _saved_tiny_model(tmp_path, preprocessing=pre, classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), "--image", str(image_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Predicted class:" in out
    assert ("cat" in out) or ("dog" in out)
    assert "Confidence:" in out


def test_cli_predict_without_classes_prints_index(tmp_path, capsys):
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    model_path = _saved_tiny_model(tmp_path, preprocessing=pre, classes=None)
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), "--image", str(image_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Predicted class index:" in out
    assert "Confidence" not in out


def test_cli_predict_without_preprocessing_fails_clearly(tmp_path, capsys):
    model_path = _saved_tiny_model(tmp_path, preprocessing=None, classes=["cat", "dog"])
    image_path = tmp_path / "query.png"
    _make_image(image_path)

    exit_code = cli_main(["model", "predict", str(model_path), "--image", str(image_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "preprocessing" in err.lower()


def test_cli_predict_missing_model_file_fails(tmp_path):
    image_path = tmp_path / "query.png"
    _make_image(image_path)
    exit_code = cli_main(["model", "predict", str(tmp_path / "nope.forge"), "--image", str(image_path)])
    assert exit_code == 1


def test_cli_predict_missing_image_file_fails(tmp_path):
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    model_path = _saved_tiny_model(tmp_path, preprocessing=pre, classes=["cat", "dog"])
    exit_code = cli_main(["model", "predict", str(model_path), "--image", str(tmp_path / "nope.png")])
    assert exit_code == 1


def test_cli_inspect_reports_classes_and_preprocessing(tmp_path, capsys):
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    model_path = _saved_tiny_model(tmp_path, preprocessing=pre, classes=["cat", "dog"])

    exit_code = cli_main(["model", "inspect", str(model_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Preprocessing: yes" in out
    assert "Classes: cat, dog" in out


def test_cli_inspect_json_reports_classes(tmp_path, capsys):
    model_path = _saved_tiny_model(tmp_path, preprocessing=None, classes=["cat", "dog"])
    exit_code = cli_main(["model", "inspect", str(model_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classes"] == ["cat", "dog"]
    assert payload["has_preprocessing"] is False


def test_cli_convert_preserves_preprocessing_and_classes(tmp_path):
    pre = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    model_path = _saved_tiny_model(tmp_path, preprocessing=pre, classes=["cat", "dog"])
    output_path = tmp_path / "converted.forge"

    exit_code = cli_main(["model", "convert", str(model_path), "--device", "cpu", "--output", str(output_path)])
    assert exit_code == 0
    assert load_classes(str(output_path)) == ["cat", "dog"]
    assert load_preprocessing(str(output_path)) is not None


def test_cli_inspect_no_classes_shows_none(tmp_path, capsys):
    model_path = _saved_tiny_model(tmp_path, preprocessing=None, classes=None)
    exit_code = cli_main(["model", "inspect", str(model_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Classes: (none)" in out


# -- end-to-end: ImageFolder + train + save(classes=...) + fresh infer.py --


def _make_dataset(root: Path, per_class: int = 4):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for cls, base in (("a", 40), ("b", 210)):
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(per_class):
            h = int(rng.integers(20, 30))
            w = int(rng.integers(20, 30))
            fill = base + int(rng.integers(-5, 5))
            arr = np.clip(np.full((h, w, 3), fill), 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="RGB").save(cls_dir / f"{cls}_{i}.png")


def test_end_to_end_train_save_classes_then_fresh_process_infer(tmp_path):
    from forge.data import DataLoader, ImageFolder

    data_root = tmp_path / "data"
    _make_dataset(data_root, per_class=5)

    preprocessing = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    dataset = ImageFolder(str(data_root), transform=preprocessing)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    forge.random.seed(0)
    model = _TinyCNN(num_classes=len(dataset.classes))
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, loss_fn=CrossEntropyLoss(), optimizer=optimizer, device="cpu")
    trainer.fit(loader, epochs=1)

    model_path = tmp_path / "artifact.forge"
    save_model(model, str(model_path), preprocessing=preprocessing, classes=dataset.classes)

    # Simulate a fresh process: forget everything except the file path.
    del preprocessing, model, dataset, loader, trainer

    # examples/image_folder_classification/infer.py's own run() function --
    # imported directly to exercise the exact fresh-process workflow it
    # implements, no --classes sidecar argument needed.
    from examples.image_folder_classification import infer as infer_module

    new_path = tmp_path / "new_query.png"
    Image.fromarray(np.full((17, 15, 3), 205, dtype=np.uint8), mode="RGB").save(new_path)

    result = infer_module.run(str(model_path), str(new_path))
    assert isinstance(result, ClassificationPrediction)
    assert result.label in ("a", "b")
    assert 0.0 <= result.confidence <= 1.0

    # And the equivalent via the CLI, against the same artifact.
    exit_code = cli_main(["model", "predict", str(model_path), "--image", str(new_path)])
    assert exit_code == 0
