"""Milestone 107 tests: `forge.data.ImageFolder(on_error=...)` and
`forge.train_image_classifier()`/`ImageClassifierResult`, CPU-only.

Covers the two concrete DX gaps Milestone 107 found while reconstructing
`sandbox/t1_cat_dog/train.py`'s real ~25,000-image workflow: (1)
`ImageFolder` had no public way to skip unreadable files (fixed by
`on_error=`), and (2) training an image-folder classifier required manually
assembling `Resize`/`Normalize`/`random_split`/`DataLoader`/a hand-sized CNN/
a hand-picked `save_and_verify()` sample (fixed by `train_image_classifier()`
composing all of it). See `forge/data/image_folder.py` and
`forge/training/image_classifier.py` for the full documented contract this
tests against; `tests/test_image_classifier_cuda.py` for the CUDA-specific
counterpart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.data import DataLoader, ImageFolder
from forge.exceptions import DataError, TrainerError
from forge.nn import Flatten, Linear, ReLU, Sequential
from forge.training import ImageClassifierResult
from forge.training.image_classifier import _default_image_classifier_cnn


def _make_image(path: Path, size=(20, 20), color=(10, 20, 30)):
    Image.new("RGB", size, color).save(path)


def _make_two_class_root(tmp_path: Path, per_class: int = 6, size=(20, 20)) -> Path:
    root = tmp_path / "data"
    for cls, base_color in [("cat", (200, 0, 0)), ("dog", (0, 200, 0))]:
        class_dir = root / cls
        class_dir.mkdir(parents=True)
        for i in range(per_class):
            _make_image(class_dir / f"{cls}_{i}.png", size=size, color=(base_color[0], base_color[1], i * 5))
    return root


def _corrupt(path: Path) -> None:
    path.write_bytes(b"not a real image file")


# -- ImageFolder(on_error=...) --------------------------------------------


def test_on_error_defaults_to_raise_and_does_not_scan_at_construction(tmp_path):
    root = _make_two_class_root(tmp_path)
    _corrupt(root / "cat" / "corrupt.jpg")

    dataset = ImageFolder(str(root))
    assert dataset.on_error == "raise"
    assert dataset.skipped_samples == []
    assert len(dataset) == 13  # 6 + 6 good + 1 corrupt, none filtered

    # The corrupt sample is still present and fails lazily, not eagerly.
    with pytest.raises(DataError):
        for i in range(len(dataset)):
            dataset[i]


def test_on_error_skip_excludes_unreadable_files_and_records_them(tmp_path):
    root = _make_two_class_root(tmp_path)
    corrupt_path = root / "cat" / "corrupt.jpg"
    _corrupt(corrupt_path)

    dataset = ImageFolder(str(root), on_error="skip")
    assert len(dataset) == 12
    assert len(dataset.skipped_samples) == 1
    skipped_path, reason = dataset.skipped_samples[0]
    assert Path(skipped_path) == corrupt_path
    assert "could not read image" in reason

    # Every remaining sample decodes cleanly (DataLoader must never hit the bad file).
    for i in range(len(dataset)):
        dataset[i]


def test_on_error_skip_still_discovers_classes_from_directory_structure(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=1)
    _corrupt(root / "cat" / "corrupt.jpg")
    dataset = ImageFolder(str(root), on_error="skip")
    assert dataset.classes == ["cat", "dog"]


def test_on_error_skip_raises_if_every_candidate_file_is_unreadable(tmp_path):
    root = tmp_path / "data"
    (root / "cat").mkdir(parents=True)
    (root / "dog").mkdir(parents=True)
    _corrupt(root / "cat" / "bad1.jpg")
    _corrupt(root / "dog" / "bad2.jpg")

    with pytest.raises(DataError, match="no readable images"):
        ImageFolder(str(root), on_error="skip")


def test_on_error_invalid_value_raises(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=1)
    with pytest.raises(DataError, match="on_error"):
        ImageFolder(str(root), on_error="ignore")


def test_dataloader_over_skip_mode_dataset_never_raises(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=4)
    _corrupt(root / "dog" / "corrupt.jpg")
    dataset = ImageFolder(str(root), on_error="skip")
    loader = DataLoader(dataset, batch_size=3)
    batches = list(loader)
    assert sum(b[0].shape[0] for b in batches) == len(dataset)


# -- _default_image_classifier_cnn -----------------------------------------


def test_default_cnn_matches_the_validated_t1_flattened_size_at_64x64():
    model = _default_image_classifier_cnn(num_classes=2, image_size=(64, 64))
    x = forge.Tensor(np.zeros((1, 3, 64, 64), dtype=np.float32))
    out = model(x)
    assert out.shape == (1, 2)


def test_default_cnn_too_small_image_size_raises_clear_error():
    with pytest.raises(DataError, match="too small"):
        _default_image_classifier_cnn(num_classes=2, image_size=(8, 8))


# -- train_image_classifier: happy path -------------------------------------


def test_train_image_classifier_end_to_end_default_model(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=8, size=(32, 32))
    model_path = tmp_path / "model.forge"

    result = forge.train_image_classifier(
        str(root), path=str(model_path), epochs=2, batch_size=4,
        image_size=(32, 32), val_fraction=0.25, seed=0, device="cpu", verbose=False,
    )

    assert isinstance(result, ImageClassifierResult)
    assert result.classes == ["cat", "dog"]
    assert result.dataset_size == 16
    assert result.train_size == 12
    assert result.val_size == 4
    assert result.skipped_images == []
    assert result.artifact_path == str(model_path)
    assert model_path.exists()
    assert "accuracy" in result.val_metrics
    assert len(result.history) == 2

    info = forge.inspect_model(str(model_path))
    assert info.task == "classification"
    assert info.classes == ["cat", "dog"]


def test_train_image_classifier_skips_corrupt_images_by_default(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=8, size=(32, 32))
    corrupt_path = root / "cat" / "corrupt.jpg"
    _corrupt(corrupt_path)
    model_path = tmp_path / "model.forge"

    result = forge.train_image_classifier(
        str(root), path=str(model_path), epochs=1, batch_size=4,
        image_size=(32, 32), val_fraction=0.25, seed=0, device="cpu", verbose=False,
    )

    assert result.dataset_size == 16  # the corrupt file never counted
    assert len(result.skipped_images) == 1
    assert Path(result.skipped_images[0][0]) == corrupt_path


def test_train_image_classifier_on_error_raise_propagates_dataerror(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=8, size=(32, 32))
    _corrupt(root / "cat" / "corrupt.jpg")
    model_path = tmp_path / "model.forge"

    with pytest.raises(DataError):
        forge.train_image_classifier(
            str(root), path=str(model_path), epochs=1, batch_size=4,
            image_size=(32, 32), on_error="raise", device="cpu", verbose=False,
        )


def test_train_image_classifier_accepts_a_custom_model(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=8, size=(16, 16))
    model_path = tmp_path / "model.forge"

    custom = Sequential(Flatten(), Linear(3 * 16 * 16, 8), ReLU(), Linear(8, 2))
    result = forge.train_image_classifier(
        str(root), path=str(model_path), epochs=1, batch_size=4,
        image_size=(16, 16), val_fraction=0.25, model=custom, device="cpu", verbose=False,
    )
    assert result.model is not custom  # save_and_verify() returns the freshly reloaded model
    assert result.artifact_path == str(model_path)


def test_train_image_classifier_artifact_predicts_a_new_image(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=8, size=(32, 32))
    model_path = tmp_path / "model.forge"
    forge.train_image_classifier(
        str(root), path=str(model_path), epochs=1, batch_size=4,
        image_size=(32, 32), val_fraction=0.25, device="cpu", verbose=False,
    )

    new_image = root / "cat" / "cat_0.png"  # reuse an existing sample as a stand-in new file
    prediction = forge.predict_artifact(str(model_path), str(new_image))
    assert prediction.label in ("cat", "dog")


# -- train_image_classifier: invalid input ----------------------------------


def test_train_image_classifier_rejects_invalid_val_fraction(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=4)
    with pytest.raises(DataError, match="val_fraction"):
        forge.train_image_classifier(str(root), path=str(tmp_path / "m.forge"), val_fraction=1.5)
    with pytest.raises(DataError, match="val_fraction"):
        forge.train_image_classifier(str(root), path=str(tmp_path / "m.forge"), val_fraction=0.0)


def test_train_image_classifier_rejects_image_size_too_small_for_default_cnn(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=4)
    with pytest.raises(DataError, match="too small"):
        forge.train_image_classifier(str(root), path=str(tmp_path / "m.forge"), image_size=(8, 8))


def test_train_image_classifier_rejects_non_module_model(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=4)
    with pytest.raises(TrainerError):
        forge.train_image_classifier(str(root), path=str(tmp_path / "m.forge"), model=object())


def test_train_image_classifier_missing_data_dir_raises_dataerror(tmp_path):
    with pytest.raises(DataError):
        forge.train_image_classifier(str(tmp_path / "nope"), path=str(tmp_path / "m.forge"))


def test_train_image_classifier_requires_at_least_two_classes(tmp_path):
    root = tmp_path / "data"
    (root / "onlyclass").mkdir(parents=True)
    _make_image(root / "onlyclass" / "a.png")
    with pytest.raises(DataError, match="at least 2 classes"):
        forge.train_image_classifier(str(root), path=str(tmp_path / "m.forge"))


def test_train_image_classifier_rejects_invalid_on_error_value(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=4)
    with pytest.raises(DataError, match="on_error"):
        forge.train_image_classifier(str(root), path=str(tmp_path / "m.forge"), on_error="ignore")


def test_train_image_classifier_empty_class_directory_contributes_no_samples(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=4)
    (root / "empty_third_class").mkdir()
    result = forge.train_image_classifier(
        str(root), path=str(tmp_path / "m.forge"), epochs=1, batch_size=2,
        image_size=(32, 32), val_fraction=0.25, device="cpu", verbose=False,
    )
    assert result.classes == ["cat", "dog", "empty_third_class"]
    assert result.dataset_size == 8  # the empty class contributes zero samples


# -- determinism -------------------------------------------------------------


def test_train_image_classifier_same_seed_reproduces_the_same_split_and_result(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=8, size=(32, 32))

    result_a = forge.train_image_classifier(
        str(root), path=str(tmp_path / "a.forge"), epochs=1, batch_size=4,
        image_size=(32, 32), val_fraction=0.25, seed=7, device="cpu", verbose=False,
    )
    result_b = forge.train_image_classifier(
        str(root), path=str(tmp_path / "b.forge"), epochs=1, batch_size=4,
        image_size=(32, 32), val_fraction=0.25, seed=7, device="cpu", verbose=False,
    )
    assert result_a.train_loss == pytest.approx(result_b.train_loss)
    assert result_a.val_metrics["accuracy"] == pytest.approx(result_b.val_metrics["accuracy"])
