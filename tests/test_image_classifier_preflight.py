"""Milestone 115 tests: `forge.train_image_classifier()` preflight (CPU).

M111 found, and M114 re-confirmed, that the image workflow discovered predictable
mistakes only after training: a bad `path=` after every epoch, and a `model=` with the
wrong number of outputs never (it trained, saved and verified, and failed on the first
`predict()`). M114's tabular workflows already reject both before epoch 1; these tests
pin the same contract for the image workflow, via the shared `forge/training/_preflight.py`.

`Trainer.fit` is replaced by a tripwire wherever a test claims "rejected before epoch 1",
and `ImageFolder` by one wherever it claims "rejected before the dataset scan": if the
work starts, the test fails with `AssertionError`, not with the expected error.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.nn import BatchNorm2d, Conv2d, Flatten, Linear, Module, ReLU, Sequential
from forge.training import Trainer
from forge.training import image_classifier as image_classifier_module

SIZE = (32, 32)  # the default 3-block CNN needs at least 22 px
FLAT = 3 * SIZE[0] * SIZE[1]
CONV_FLAT = 4 * (SIZE[0] - 2) * (SIZE[1] - 2)  # 4 channels after one 3x3 convolution


def _make_two_class_root(tmp_path: Path, per_class: int = 8) -> Path:
    root = tmp_path / "data"
    for cls, base in [("cat", (200, 0)), ("dog", (0, 200))]:
        (root / cls).mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (20, 20), (base[0], base[1], i * 5)).save(root / cls / f"{cls}_{i}.png")
    return root


@pytest.fixture
def root(tmp_path) -> Path:
    return _make_two_class_root(tmp_path)


@pytest.fixture
def no_training(monkeypatch):
    def tripwire(self, *args, **kwargs):
        raise AssertionError("training started: a preflight check ran too late")

    monkeypatch.setattr(Trainer, "fit", tripwire)


@pytest.fixture
def no_scan(monkeypatch):
    def tripwire(*args, **kwargs):
        raise AssertionError("the dataset was scanned: a cheap check ran too late")

    monkeypatch.setattr(image_classifier_module, "ImageFolder", tripwire)


def train(root, path, **kwargs):
    kwargs.setdefault("epochs", 1)
    return forge.train_image_classifier(
        root, path=path, batch_size=4, image_size=SIZE, val_fraction=0.25, device="cpu", verbose=False, **kwargs,
    )


def flat_model(outputs: int) -> Sequential:
    return Sequential(Flatten(), Linear(FLAT, 8), ReLU(), Linear(8, outputs))


class Unregistered(Module):
    """A working model that `save_model()` cannot serialise (not registered)."""

    def __init__(self, outputs: int = 2):
        super().__init__()
        self.fc = Linear(FLAT, outputs)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.numpy().copy())
        return self.fc(x.reshape(x.shape[0], -1))


def leftovers(directory: Path) -> "list[str]":
    return sorted(p.name for p in directory.iterdir() if p.name.startswith(".forge-preflight"))


# -- output path --------------------------------------------------------------


class TestOutputPath:
    def test_missing_directory_fails_before_the_dataset_scan(self, root, tmp_path, no_scan, no_training):
        with pytest.raises(forge.PersistenceError, match="does not exist"):
            train(root, str(tmp_path / "nope" / "m.forge"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]

    def test_path_is_a_directory_fails_before_the_dataset_scan(self, root, tmp_path, no_scan, no_training):
        with pytest.raises(forge.PersistenceError, match="is a directory"):
            train(root, str(tmp_path))

    def test_parent_is_a_file(self, root, tmp_path, no_training):
        (tmp_path / "file.txt").write_text("x")
        with pytest.raises(forge.PersistenceError):
            train(root, str(tmp_path / "file.txt" / "m.forge"))

    @pytest.mark.skipif(os.name != "nt", reason="'|' is only an invalid filename character on Windows")
    def test_invalid_filename_fails_before_epoch_1_and_writes_nothing(self, root, tmp_path, no_training):
        with pytest.raises(forge.PersistenceError):
            train(root, str(tmp_path / "a|b.forge"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]

    @pytest.mark.parametrize("path", ["", None])
    def test_empty_or_missing_path_fails_before_the_dataset_scan(self, root, no_scan, no_training, path):
        with pytest.raises(forge.DataError, match="path"):
            train(root, path)

    def test_a_model_that_cannot_be_serialised_fails_before_epoch_1(self, root, tmp_path, no_training):
        with pytest.raises(forge.PersistenceError, match="not registered"):
            train(root, str(tmp_path / "m.forge"), model=Unregistered())
        assert not (tmp_path / "m.forge").exists()
        assert leftovers(tmp_path) == []

    def test_the_final_artifact_is_not_written_until_training_has_finished(self, root, tmp_path, monkeypatch):
        artifact = tmp_path / "m.forge"
        seen = {}
        original_fit = Trainer.fit

        def spy(self, *args, **kwargs):
            seen["exists_at_epoch_1"] = artifact.exists()
            seen["leftovers_at_epoch_1"] = leftovers(tmp_path)
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(Trainer, "fit", spy)
        train(root, str(artifact))
        assert seen == {"exists_at_epoch_1": False, "leftovers_at_epoch_1": []}
        assert artifact.exists()


# -- model= contract ----------------------------------------------------------


class TestModelContract:
    def test_a_non_module_model_fails_before_the_dataset_scan(self, root, tmp_path, no_scan, no_training):
        with pytest.raises(forge.TrainerError, match="forge.nn.Module"):
            train(root, str(tmp_path / "m.forge"), model="resnet")

    def test_too_many_outputs_is_rejected_instead_of_saving_an_unusable_artifact(self, root, tmp_path, no_training):
        with pytest.raises(forge.TrainerError, match=r"needs \(batch, 2\)") as info:
            train(root, str(tmp_path / "m.forge"), model=flat_model(5))
        assert "(2, 5)" in str(info.value) and "'cat'" in str(info.value) and "'dog'" in str(info.value)
        assert not (tmp_path / "m.forge").exists()

    def test_too_few_outputs_is_rejected(self, root, tmp_path, no_training):
        with pytest.raises(forge.TrainerError, match=r"needs \(batch, 2\)"):
            train(root, str(tmp_path / "m.forge"), model=flat_model(1))

    def test_wrong_input_width_is_a_trainer_error_naming_the_expected_input(self, root, tmp_path, no_training):
        model = Sequential(Flatten(), Linear(999, 2))
        with pytest.raises(forge.TrainerError, match=rf"could not process a \(2, 3, {SIZE[0]}, {SIZE[1]}\).*\(batch, 3, {SIZE[0]}, {SIZE[1]}\)"):
            train(root, str(tmp_path / "m.forge"), model=model)

    def test_a_single_channel_model_is_rejected_for_rgb_input(self, root, tmp_path, no_training):
        model = Sequential(Conv2d(1, 4, kernel_size=3), Flatten(), Linear(CONV_FLAT, 2))
        with pytest.raises(forge.TrainerError, match="could not process"):
            train(root, str(tmp_path / "m.forge"), model=model)

    def test_a_model_that_does_not_reduce_to_class_scores_is_rejected(self, root, tmp_path, no_training):
        with pytest.raises(forge.TrainerError, match=r"needs \(batch, 2\)"):
            train(root, str(tmp_path / "m.forge"), model=Sequential(Conv2d(3, 2, kernel_size=3)))

    def test_the_probe_is_a_real_preprocessed_batch(self, root, tmp_path, no_training):
        model = Unregistered()
        with pytest.raises(forge.PersistenceError):  # it passes the contract, then fails serialisation
            train(root, str(tmp_path / "m.forge"), model=model)
        assert len(model.seen) == 1
        batch = model.seen[0]
        assert batch.shape == (2, 3, *SIZE)
        assert batch.dtype == np.float32 and 0.0 <= batch.min() and batch.max() <= 1.0  # Normalize(0, 255)
        assert batch.max() > 0.0  # real pixels, not a zeros placeholder

    def test_a_rejected_model_is_left_exactly_as_it_was(self, root, tmp_path, no_training):
        norm = BatchNorm2d(4)
        model = Sequential(Conv2d(3, 4, kernel_size=3), norm, Flatten(), Linear(CONV_FLAT, 5))
        running_mean = norm.running_mean.numpy().copy()
        with pytest.raises(forge.TrainerError):
            train(root, str(tmp_path / "m.forge"), model=model)
        assert model.training  # the probe runs in eval mode and restores the mode
        np.testing.assert_array_equal(norm.running_mean.numpy(), running_mean)


# -- what still works ---------------------------------------------------------


def test_a_valid_custom_model_still_trains_saves_predicts_and_evaluates(root, tmp_path):
    path = tmp_path / "m.forge"
    result = train(root, str(path), model=flat_model(2), epochs=2)

    assert result.artifact_path == str(path) and path.exists()
    assert leftovers(tmp_path) == []
    assert result.classes == ["cat", "dog"]

    predictor = forge.load_predictor(str(path), device="cpu")
    prediction = predictor.predict(str(root / "cat" / "cat_0.png"))
    assert prediction.label in ("cat", "dog")
    evaluation = predictor.evaluate(root)
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert list(evaluation.classes) == ["cat", "dog"] and int(np.sum(evaluation.confusion_matrix)) == 16


def test_the_default_model_still_trains_and_leaves_no_temporary_file(root, tmp_path):
    path = tmp_path / "m.forge"
    result = train(root, str(path))
    assert path.exists() and leftovers(tmp_path) == []
    assert result.model is not None and result.val_size == 4


def test_preflight_does_not_change_what_training_produces(root, tmp_path):
    """Same seed, same result as before the preflight existed: it consumes no RNG and changes no weights."""
    first = train(root, str(tmp_path / "a.forge"), seed=5, epochs=2)
    second = train(root, str(tmp_path / "b.forge"), seed=5, epochs=2)
    assert first.train_loss == second.train_loss and first.val_metrics == second.val_metrics
    x = forge.Tensor(np.random.default_rng(0).random((3, 3, *SIZE)).astype(np.float32))
    np.testing.assert_array_equal(forge.predict(first.model, x).numpy(), forge.predict(second.model, x).numpy())
