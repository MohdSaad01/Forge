"""Milestone 69 tests: `forge.data.ImageFolder`.

Covers discovery, image conversion, dataset behavior, and `DataLoader`
integration against real image files written to `tmp_path` -- no external
dataset download. See `forge/data/image_folder.py`'s module docstring for
the documented contract this tests against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge import DType, Tensor
from forge.data import DataLoader, ImageFolder
from forge.data.image_folder import IMAGE_EXTENSIONS
from forge.data.transforms import Compose, Lambda, Normalize, Resize
from forge.exceptions import DataError


def _make_image(path: Path, size=(8, 8), color=(10, 20, 30), mode="RGB"):
    Image.new(mode, size, color).save(path)


def _make_two_class_root(tmp_path: Path, per_class: int = 3) -> Path:
    root = tmp_path / "data"
    for cls, base_color in [("cats", (200, 0, 0)), ("dogs", (0, 200, 0))]:
        class_dir = root / cls
        class_dir.mkdir(parents=True)
        for i in range(per_class):
            _make_image(class_dir / f"{cls}_{i}.png", color=(base_color[0], base_color[1], i * 10))
    return root


# -- discovery ----------------------------------------------------------


def test_discovers_classes_sorted_deterministically(tmp_path):
    root = tmp_path / "data"
    for cls in ["zebra", "apple", "mango"]:
        (root / cls).mkdir(parents=True)
        _make_image(root / cls / "img.png")
    ds = ImageFolder(str(root))
    assert ds.classes == ["apple", "mango", "zebra"]
    assert ds.class_to_idx == {"apple": 0, "mango": 1, "zebra": 2}


def test_class_indices_are_deterministic_across_construction(tmp_path):
    root = _make_two_class_root(tmp_path)
    ds1 = ImageFolder(str(root))
    ds2 = ImageFolder(str(root))
    assert ds1.classes == ds2.classes
    assert ds1.class_to_idx == ds2.class_to_idx
    assert [p.name for p, _ in ds1.samples] == [p.name for p, _ in ds2.samples]


def test_file_ordering_within_class_is_deterministic(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    for name in ["c.png", "a.png", "b.png"]:
        _make_image(class_dir / name)
    ds = ImageFolder(str(root))
    names = [p.name for p, _ in ds.samples]
    assert names == ["a.png", "b.png", "c.png"]


def test_supported_extensions_are_case_insensitive(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    _make_image(class_dir / "upper.PNG")
    _make_image(class_dir / "lower.jpg")
    ds = ImageFolder(str(root))
    assert len(ds) == 2


def test_unsupported_files_are_not_treated_as_samples(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    _make_image(class_dir / "real.png")
    (class_dir / "notes.txt").write_text("not an image")
    (class_dir / "README.md").write_text("also not an image")
    ds = ImageFolder(str(root))
    assert len(ds) == 1
    assert ds.samples[0][0].name == "real.png"


def test_custom_extensions_override_default(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    _make_image(class_dir / "img.png")
    with pytest.raises(DataError):
        ImageFolder(str(root), extensions=(".jpg",))


def test_empty_class_directory_is_a_valid_class_with_zero_samples(tmp_path):
    root = tmp_path / "data"
    (root / "has_images").mkdir(parents=True)
    _make_image(root / "has_images" / "img.png")
    (root / "empty_class").mkdir(parents=True)

    ds = ImageFolder(str(root))
    assert ds.classes == ["empty_class", "has_images"]
    assert ds.class_to_idx["empty_class"] == 0
    assert len(ds) == 1  # only has_images contributed a sample
    assert all(idx == ds.class_to_idx["has_images"] for _, idx in ds.samples)


def test_nested_subdirectory_files_are_not_discovered(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    nested = class_dir / "nested"
    nested.mkdir(parents=True)
    _make_image(class_dir / "top.png")
    _make_image(nested / "deep.png")

    ds = ImageFolder(str(root))
    assert len(ds) == 1
    assert ds.samples[0][0].name == "top.png"


def test_missing_root_raises_data_error(tmp_path):
    with pytest.raises(DataError):
        ImageFolder(str(tmp_path / "does_not_exist"))


def test_root_that_is_a_file_raises_data_error(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    with pytest.raises(DataError):
        ImageFolder(str(f))


def test_root_with_no_class_subdirectories_raises_data_error(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _make_image(root / "stray.png")  # a loose file, not inside a class dir
    with pytest.raises(DataError):
        ImageFolder(str(root))


def test_root_with_zero_total_samples_raises_data_error(tmp_path):
    root = tmp_path / "data"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    with pytest.raises(DataError):
        ImageFolder(str(root))


def test_multiple_classes_all_discovered(tmp_path):
    root = tmp_path / "data"
    for cls in ["a", "b", "c", "d"]:
        (root / cls).mkdir(parents=True)
        _make_image(root / cls / "img.png")
    ds = ImageFolder(str(root))
    assert ds.classes == ["a", "b", "c", "d"]
    assert len(ds) == 4


# -- image conversion -----------------------------------------------------


def test_rgb_image_shape_dtype_and_range(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    _make_image(class_dir / "img.png", size=(5, 7), color=(10, 20, 30), mode="RGB")
    ds = ImageFolder(str(root))
    image, _ = ds[0]
    assert isinstance(image, Tensor)
    assert image.shape == (3, 7, 5)  # (C, H, W)
    assert image.dtype == DType.FLOAT32
    arr = image.numpy()
    assert arr.min() >= 0.0 and arr.max() <= 255.0
    np.testing.assert_allclose(arr[:, 0, 0], [10.0, 20.0, 30.0])


def test_grayscale_image_converted_to_three_channels(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    _make_image(class_dir / "gray.png", size=(4, 4), color=128, mode="L")
    ds = ImageFolder(str(root))
    image, _ = ds[0]
    assert image.shape == (3, 4, 4)
    arr = image.numpy()
    # Grayscale replicated identically across all 3 channels.
    np.testing.assert_allclose(arr[0], arr[1])
    np.testing.assert_allclose(arr[1], arr[2])
    np.testing.assert_allclose(arr[0, 0, 0], 128.0)


def test_rgba_image_alpha_is_discarded(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    _make_image(class_dir / "rgba.png", size=(4, 4), color=(50, 60, 70, 5), mode="RGBA")
    ds = ImageFolder(str(root))
    image, _ = ds[0]
    assert image.shape == (3, 4, 4)  # no alpha channel present
    arr = image.numpy()
    np.testing.assert_allclose(arr[:, 0, 0], [50.0, 60.0, 70.0])


def test_corrupt_image_file_raises_data_error(tmp_path):
    root = tmp_path / "data"
    class_dir = root / "a"
    class_dir.mkdir(parents=True)
    bad = class_dir / "corrupt.png"
    bad.write_bytes(b"not actually a png")
    ds = ImageFolder(str(root))
    with pytest.raises(DataError):
        ds[0]


# -- dataset behavior -------------------------------------------------------


def test_len_matches_discovered_sample_count(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=5)
    ds = ImageFolder(str(root))
    assert len(ds) == 10


def test_getitem_returns_tensor_and_int64_label(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=2)
    ds = ImageFolder(str(root))
    image, label = ds[0]
    assert isinstance(image, Tensor)
    assert isinstance(label, Tensor)
    assert label.shape == ()
    assert label.dtype == DType.INT64
    assert int(label.numpy()) in (0, 1)


def test_getitem_negative_index(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=2)
    ds = ImageFolder(str(root))
    last_via_negative = ds[-1]
    last_via_positive = ds[len(ds) - 1]
    np.testing.assert_allclose(last_via_negative[0].numpy(), last_via_positive[0].numpy())
    assert int(last_via_negative[1].numpy()) == int(last_via_positive[1].numpy())


def test_getitem_out_of_range_raises_data_error(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=2)
    ds = ImageFolder(str(root))
    with pytest.raises(DataError):
        ds[len(ds)]


def test_getitem_non_int_index_raises_data_error(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=2)
    ds = ImageFolder(str(root))
    with pytest.raises(DataError):
        ds["0"]


def test_transform_applied_to_image_only(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=2)
    transform = Compose([Lambda(lambda x: x * (1.0 / 255.0)), Normalize(mean=0.5, std=0.5)])
    ds = ImageFolder(str(root), transform=transform)
    image, label = ds[0]
    arr = image.numpy()
    assert arr.min() >= -1.0001 and arr.max() <= 1.0001
    assert label.dtype == DType.INT64  # target untouched by the feature transform


def test_target_transform_applied_to_label_only(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=2)
    ds = ImageFolder(str(root), target_transform=lambda t: Tensor(int(t.numpy()) + 100, dtype="int64"))
    image, label = ds[0]
    assert int(label.numpy()) >= 100
    assert image.numpy().max() > 1.0  # feature untouched (raw [0, 255] range)


def test_repr_contains_classes_and_size(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=2)
    ds = ImageFolder(str(root))
    text = repr(ds)
    assert "cats" in text and "dogs" in text
    assert "size=4" in text


def test_default_extensions_constant_exported():
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS


# -- DataLoader integration --------------------------------------------------


def test_dataloader_batches_image_folder_samples(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=6)
    ds = ImageFolder(str(root))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3  # 12 samples / batch_size 4
    for bx, by in batches[:-1]:
        assert bx.shape[0] == 4
        assert bx.shape[1:] == (3, 8, 8)
        assert by.shape == (4,)
        assert by.dtype == DType.INT64


def test_dataloader_shuffle_is_reproducible_with_explicit_generator(tmp_path):
    root = _make_two_class_root(tmp_path, per_class=6)
    ds = ImageFolder(str(root))

    loader1 = DataLoader(ds, batch_size=3, shuffle=True, generator=np.random.default_rng(0))
    loader2 = DataLoader(ds, batch_size=3, shuffle=True, generator=np.random.default_rng(0))

    labels1 = [by.numpy().tolist() for _, by in loader1]
    labels2 = [by.numpy().tolist() for _, by in loader2]
    assert labels1 == labels2


def test_dataloader_feeds_conv2d_compatible_batches(tmp_path):
    from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

    root = _make_two_class_root(tmp_path, per_class=4)
    ds = ImageFolder(str(root))
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = Sequential(
        Conv2d(3, 4, kernel_size=3),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(4 * 3 * 3, 2),
    )
    bx, by = next(iter(loader))
    out = model(bx)
    assert out.shape == (4, 2)


# -- mixed-resolution ImageFolder + Resize (Milestone 70) --------------------


def _make_mixed_resolution_root(tmp_path: Path) -> Path:
    """Mirror the M70 brief's own example: a root of differently-sized files."""
    root = tmp_path / "data"
    sizes = {
        "circles": [(43, 61), (80, 72), (128, 96)],
        "squares": [(55, 70), (91, 64), (120, 110)],
        "triangles": [(64, 45), (96, 128), (72, 90)],
    }
    for cls, dims in sizes.items():
        class_dir = root / cls
        class_dir.mkdir(parents=True)
        for i, (w, h) in enumerate(dims):
            _make_image(class_dir / f"{cls}_{i}.png", size=(w, h))
    return root


def test_mixed_resolution_dataset_has_varying_sample_shapes_without_resize(tmp_path):
    root = _make_mixed_resolution_root(tmp_path)
    ds = ImageFolder(str(root))
    shapes = {ds[i][0].shape for i in range(len(ds))}
    assert len(shapes) > 1


def test_mixed_resolution_dataset_batching_fails_without_resize(tmp_path):
    root = _make_mixed_resolution_root(tmp_path)
    ds = ImageFolder(str(root))
    loader = DataLoader(ds, batch_size=2)
    with pytest.raises(DataError):
        list(loader)


def test_resize_transform_normalizes_mixed_resolution_samples_to_one_shape(tmp_path):
    root = _make_mixed_resolution_root(tmp_path)
    ds = ImageFolder(str(root), transform=Resize((32, 32)))
    shapes = {ds[i][0].shape for i in range(len(ds))}
    assert shapes == {(3, 32, 32)}


def test_dataloader_batches_mixed_resolution_dataset_after_resize(tmp_path):
    root = _make_mixed_resolution_root(tmp_path)
    ds = ImageFolder(str(root), transform=Resize((16, 16)))
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3  # 9 samples / batch_size 3
    for bx, by in batches:
        assert bx.shape == (3, 3, 16, 16)
        assert by.shape == (3,)


def test_resize_composes_with_pixel_scaling_transform(tmp_path):
    root = _make_mixed_resolution_root(tmp_path)
    transform = Compose([Resize((16, 16)), Lambda(lambda x: x * (1.0 / 255.0))])
    ds = ImageFolder(str(root), transform=transform)
    loader = DataLoader(ds, batch_size=9, shuffle=False)
    bx, by = next(iter(loader))
    assert bx.shape == (9, 3, 16, 16)
    assert bx.numpy().min() >= 0.0
    assert bx.numpy().max() <= 1.0
