"""Milestone 76 tests: `forge.data.save_image`.

The write-side counterpart to `ImageFolder`'s read-side tests
(`test_image_folder.py`): writes a Tensor to a real file via Pillow and
reads it back to verify pixel content, rather than mocking the filesystem or
Pillow. See `forge/data/image_folder.py::save_image`'s own docstring for the
documented `[0, 1]`-scaled-input, `(C, H, W)`-with-`C`-in-`(1, 3)` contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge import Tensor
from forge.data import save_image
from forge.exceptions import DataError


def _tensor(array: np.ndarray) -> Tensor:
    return Tensor(array.astype(np.float32))


# -- normal operation -----------------------------------------------------


def test_save_image_rgb_round_trips_pixel_values(tmp_path):
    array = np.zeros((3, 4, 5), dtype=np.float32)
    array[0, :, :] = 1.0   # full red
    array[1, :, :] = 0.5   # half green
    array[2, :, :] = 0.0   # no blue
    path = tmp_path / "out.png"

    save_image(_tensor(array), path)

    assert path.is_file()
    with Image.open(path) as img:
        assert img.mode == "RGB"
        assert img.size == (5, 4)  # PIL size is (width, height)
        pixel = img.getpixel((0, 0))
    assert pixel == (255, 128, 0)


def test_save_image_grayscale_round_trips_pixel_values(tmp_path):
    array = np.full((1, 6, 7), 0.2, dtype=np.float32)
    path = tmp_path / "gray.png"

    save_image(_tensor(array), path)

    with Image.open(path) as img:
        assert img.mode == "L"
        assert img.size == (7, 6)
        pixel = img.getpixel((0, 0))
    assert pixel == round(0.2 * 255.0)


def test_save_image_accepts_path_object_and_str(tmp_path):
    array = np.zeros((3, 2, 2), dtype=np.float32)
    save_image(_tensor(array), tmp_path / "as_path.png")
    save_image(_tensor(array), str(tmp_path / "as_str.png"))
    assert (tmp_path / "as_path.png").is_file()
    assert (tmp_path / "as_str.png").is_file()


# -- clipping (unbounded model output) -------------------------------------


def test_save_image_clips_values_above_one(tmp_path):
    array = np.full((1, 2, 2), 5.0, dtype=np.float32)
    path = tmp_path / "clipped_high.png"
    save_image(_tensor(array), path)
    with Image.open(path) as img:
        assert img.getpixel((0, 0)) == 255


def test_save_image_clips_values_below_zero(tmp_path):
    array = np.full((1, 2, 2), -5.0, dtype=np.float32)
    path = tmp_path / "clipped_low.png"
    save_image(_tensor(array), path)
    with Image.open(path) as img:
        assert img.getpixel((0, 0)) == 0


# -- invalid inputs ---------------------------------------------------------


def test_save_image_rejects_non_tensor(tmp_path):
    with pytest.raises(DataError):
        save_image(np.zeros((3, 2, 2), dtype=np.float32), tmp_path / "out.png")


def test_save_image_rejects_wrong_ndim(tmp_path):
    with pytest.raises(DataError):
        save_image(_tensor(np.zeros((2, 2), dtype=np.float32)), tmp_path / "out.png")


def test_save_image_rejects_unsupported_channel_count(tmp_path):
    with pytest.raises(DataError):
        save_image(_tensor(np.zeros((2, 4, 4), dtype=np.float32)), tmp_path / "out.png")


def test_save_image_rejects_missing_parent_directory(tmp_path):
    with pytest.raises(DataError):
        save_image(_tensor(np.zeros((3, 2, 2), dtype=np.float32)), tmp_path / "missing_dir" / "out.png")


def test_save_image_error_message_names_the_path(tmp_path):
    bad_path = tmp_path / "nope" / "out.png"
    with pytest.raises(DataError, match="nope"):
        save_image(_tensor(np.zeros((3, 2, 2), dtype=np.float32)), bad_path)


# -- shape edge cases --------------------------------------------------------


def test_save_image_single_pixel(tmp_path):
    path = tmp_path / "tiny.png"
    save_image(_tensor(np.array([[[1.0]], [[0.0]], [[0.5]]], dtype=np.float32)), path)
    with Image.open(path) as img:
        assert img.size == (1, 1)
