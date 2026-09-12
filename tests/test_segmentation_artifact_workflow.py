"""Milestone 84 tests: `forge.predict_image_artifact()`, the portable
image-to-image (dense-prediction) artifact inference API, and the complete
`examples/segmentation` file-based artifact workflow it completes.

```text
examples.segmentation.dataset/model -> forge.train_and_save()-style artifact
    -> fresh process -> forge.predict_image_artifact() -> predicted mask
    -> forge.data.save_image()
```

Mirrors `tests/test_artifact_inference.py` (Milestone 82, classification)
and `tests/test_regression_artifact_workflow.py` (Milestone 83, numeric
regression) for this third artifact shape. See
`forge/training/inference.py::predict_image_artifact()`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.data import ImageFolder
from forge.data.transforms import Compose, Normalize
from forge.exceptions import DataError, PersistenceError
from forge.nn import Conv2d, Module, ReLU, Sequential
from forge.serialization import load_preprocessing, save_model
from forge.training import predict, predict_image_artifact
from forge.training.inference import predict_image_artifact as predict_image_artifact_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.segmentation.dataset import IMAGE_SIZE, generate_raw  # noqa: E402
from examples.segmentation.train import build_transform  # noqa: E402
from examples.segmentation.train import main as train_main  # noqa: E402

_SMALL_TRAIN = ["--n-train", "200", "--n-test", "40", "--batch-size", "16", "--epochs", "3"]


def _build_transform():
    return Compose([Normalize(mean=0.0, std=255.0)])


def _tiny_mask_model() -> Sequential:
    """A tiny (3, H, W) -> (1, H, W) same-resolution model -- fast to train/save/load, no maxpool/upsample needed."""
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1),
        ReLU(),
        Conv2d(4, 1, kernel_size=3, padding=1),
    )


def _make_rgb_image(path: Path, size=(8, 8), fill=(200, 30, 30)) -> None:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = fill
    Image.fromarray(arr, mode="RGB").save(path)


def _saved_tiny_model(tmp_path, *, preprocessing=None, seed=0):
    forge.random.seed(seed)
    model = _tiny_mask_model()
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=preprocessing)
    return path


# -- basic API shape ----------------------------------------------------------


def test_predict_image_artifact_is_reexported_consistently():
    assert forge.predict_image_artifact is predict_image_artifact
    assert forge.training.predict_image_artifact is predict_image_artifact
    assert predict_image_artifact is predict_image_artifact_direct


def test_predict_image_artifact_returns_a_binary_chw_mask(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path, size=(8, 8))

    result = predict_image_artifact(str(model_path), str(image_path))
    assert isinstance(result, forge.Tensor)
    assert result.shape == (1, 8, 8)
    values = set(np.unique(result.numpy()).tolist())
    assert values <= {0.0, 1.0}


def test_predict_image_artifact_output_is_directly_savable(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path, size=(10, 6))

    result = predict_image_artifact(str(model_path), str(image_path))
    out_path = tmp_path / "prediction.png"
    forge.data.save_image(result, str(out_path))
    assert out_path.is_file()

    with Image.open(out_path) as img:
        assert img.mode == "L"
        assert img.size == (10, 6)
        array = np.array(img)
    assert set(np.unique(array).tolist()) <= {0, 255}


def test_predict_image_artifact_accepts_a_pathlib_path_for_the_image(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path)

    result = predict_image_artifact(str(model_path), image_path)
    assert isinstance(result, forge.Tensor)


def test_predict_image_artifact_matches_the_manual_load_predict_threshold_pipeline(tmp_path):
    """predict_image_artifact() must be a pure composition -- bit-for-bit
    identical to hand-assembling load_model()/load_preprocessing()/
    ImageFolder._load_image()/predict()/threshold, the exact sequence it
    replaces, on the same on-disk image file."""
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path, fill=(80, 150, 40))

    result = predict_image_artifact(str(model_path), str(image_path))

    preprocessing = load_preprocessing(str(model_path))
    model = forge.load_model(str(model_path))
    raw = ImageFolder._load_image(image_path)
    batch = preprocessing(raw).reshape(1, 3, 8, 8)
    expected = (predict(model, batch).numpy() >= 0.5).astype(np.float32)[0]

    np.testing.assert_array_equal(result.numpy(), expected)


def test_predict_image_artifact_custom_threshold_changes_the_result(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path)

    low = predict_image_artifact(str(model_path), str(image_path), threshold=-1e9)
    high = predict_image_artifact(str(model_path), str(image_path), threshold=1e9)
    assert np.all(low.numpy() == 1.0)
    assert np.all(high.numpy() == 0.0)


def test_predict_image_artifact_device_override_accepted_on_cpu_only_machine(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path)

    result = predict_image_artifact(str(model_path), str(image_path), device="cpu")
    assert isinstance(result, forge.Tensor)


# -- error handling -------------------------------------------------------


def test_predict_image_artifact_without_preprocessing_raises_persistence_error(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=None)
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path)

    with pytest.raises(PersistenceError, match="preprocessing"):
        predict_image_artifact(str(model_path), str(image_path))


def test_predict_image_artifact_missing_model_file_raises_persistence_error(tmp_path):
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path)

    with pytest.raises(PersistenceError):
        predict_image_artifact(str(tmp_path / "nope.forge"), str(image_path))


def test_predict_image_artifact_missing_image_file_raises_data_error(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())

    with pytest.raises(DataError):
        predict_image_artifact(str(model_path), str(tmp_path / "nope.png"))


def test_predict_image_artifact_rejects_unsupported_input_types(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())

    with pytest.raises(DataError):
        predict_image_artifact(str(model_path), 12345)


def test_predict_image_artifact_corrupt_image_file_raises_data_error(tmp_path):
    model_path = _saved_tiny_model(tmp_path, preprocessing=_build_transform())
    bad_image = tmp_path / "not_an_image.png"
    bad_image.write_bytes(b"not a real png file")

    with pytest.raises(DataError):
        predict_image_artifact(str(model_path), str(bad_image))


# -- real consumer: examples/segmentation, end to end ----------------------


def test_segmentation_train_saves_preprocessing_and_the_new_file_based_demo_artifacts(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL_TRAIN + ["--seed", "1", "--output-dir", str(out_dir)])

    model_path = out_dir / "segmentation_model.forge"
    assert model_path.is_file()
    assert load_preprocessing(str(model_path)) is not None

    for name in (
        "segmentation_new_image.png",
        "segmentation_new_ground_truth_mask.png",
        "segmentation_new_predicted_mask.png",
    ):
        assert (out_dir / name).is_file(), f"missing {name}"

    with Image.open(out_dir / "segmentation_new_predicted_mask.png") as img:
        assert img.size == (IMAGE_SIZE, IMAGE_SIZE)


def test_predict_image_artifact_reproduces_trains_own_saved_prediction(tmp_path):
    """train.py's own `predict_image_artifact()` call and a fresh call made
    here, on the same file, from this same process, must agree exactly --
    the artifact on disk is the single source of truth, not whatever
    training happened to hold in memory."""
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL_TRAIN + ["--seed", "2", "--output-dir", str(out_dir)])

    model_path = out_dir / "segmentation_model.forge"
    new_image_path = out_dir / "segmentation_new_image.png"
    saved_prediction_path = out_dir / "segmentation_new_predicted_mask.png"

    recomputed = predict_image_artifact(str(model_path), str(new_image_path))
    with Image.open(saved_prediction_path) as img:
        saved = np.array(img).astype(np.float32) / 255.0

    np.testing.assert_array_equal(recomputed.numpy()[0], saved)


def test_segmentation_artifact_prediction_beats_all_background_on_a_learnable_shape(tmp_path):
    """A weak sanity check on real product output, not just plumbing: after a
    few real epochs, the artifact's predicted mask on a brand-new image
    should overlap the true shape better than predicting nothing at all."""
    out_dir = tmp_path / "artifacts"
    seed = 3
    train_main([
        "--n-train", "300", "--n-test", "60", "--batch-size", "16", "--epochs", "8",
        "--seed", str(seed), "--output-dir", str(out_dir),
    ])

    model_path = out_dir / "segmentation_model.forge"
    _, ground_truth = generate_raw(1, seed=seed + 2, size=IMAGE_SIZE)  # matches train.py's `args.seed + 2`
    new_image_path = out_dir / "segmentation_new_image.png"

    predicted = predict_image_artifact(str(model_path), str(new_image_path)).numpy() >= 0.5
    target = ground_truth[0] >= 0.5

    intersection = int(np.sum(predicted & target))
    union = int(np.sum(predicted | target))
    iou = 1.0 if union == 0 else intersection / union
    assert iou > 0.3


# -- fresh process ----------------------------------------------------------


def test_examples_segmentation_infer_works_from_a_genuinely_separate_process(tmp_path):
    """The whole point of Milestone 84 is that a `.forge` file plus one new
    image file are enough -- prove it by launching a real, separate OS
    process that runs `examples/segmentation/infer.py` as a script, with no
    shared memory, module cache, or import state with this test process."""
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL_TRAIN + ["--seed", "4", "--output-dir", str(out_dir)])

    model_path = out_dir / "segmentation_model.forge"
    new_image_path = out_dir / "segmentation_new_image.png"
    fresh_output_path = tmp_path / "fresh_prediction.png"

    result = subprocess.run(
        [
            sys.executable, "-m", "examples.segmentation.infer",
            "--model", str(model_path),
            "--image", str(new_image_path),
            "--output", str(fresh_output_path),
        ],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert fresh_output_path.is_file()

    expected = predict_image_artifact(str(model_path), str(new_image_path))
    with Image.open(fresh_output_path) as img:
        assert img.mode == "L"
        assert img.size == (IMAGE_SIZE, IMAGE_SIZE)
        produced = np.array(img).astype(np.float32) / 255.0

    np.testing.assert_array_equal(produced, expected.numpy()[0])


def test_predict_image_artifact_works_from_a_genuinely_separate_process_with_a_tiny_model(tmp_path):
    """Same fresh-process guarantee as `test_artifact_inference.py`'s
    equivalent, but for this artifact shape: built entirely from
    pre-registered `forge.nn` types, so the fresh subprocess (which never
    imports this test module) can reconstruct it."""
    forge.random.seed(0)
    model = _tiny_mask_model()
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), preprocessing=_build_transform())
    image_path = tmp_path / "query.png"
    _make_rgb_image(image_path, size=(12, 9))

    script = (
        "import forge\n"
        f"result = forge.predict_image_artifact({str(model_path)!r}, {str(image_path)!r})\n"
        "print(f'shape={result.shape} sum={float(result.numpy().sum()):.4f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    expected = predict_image_artifact(str(model_path), str(image_path))
    assert f"shape={expected.shape} sum={float(expected.numpy().sum()):.4f}" in result.stdout
