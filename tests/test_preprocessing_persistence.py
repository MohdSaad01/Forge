"""Milestone 71 tests: preprocessing-transform configuration persistence.

Covers `forge.serialization.transforms` (the transform registry, `serialize_
transform`/`deserialize_transform`, and rejection of unregistered/malformed
transforms) and `save_model(..., preprocessing=...)`/`load_preprocessing()`
(the model-file-level API), plus a real end-to-end `ImageFolder` ->
`Resize`/`Normalize` -> train -> save -> fresh-process load -> `forge.
predict()` workflow using real PNG files on disk (mirroring `tests/
test_image_folder_classification_integration.py`'s own conventions). See
`docs/architecture/persistence.md`'s **Preprocessing metadata** section.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge import Tensor
from forge.data.transforms import (
    Compose,
    Flatten,
    Lambda,
    Normalize,
    Reshape,
    Resize,
    ToTensor,
    Transform,
)
from forge.exceptions import PersistenceError
from forge.nn import Conv2d, CrossEntropyLoss, Flatten as NNFlatten, Linear, MaxPool2d, Module, ReLU
from forge.optim import Adam
from forge.serialization import load_model, load_preprocessing, register_module, save_model
from forge.serialization.transforms import (
    deserialize_transform,
    register_transform,
    serialize_transform,
    spec_for_class,
    spec_for_name,
)
from forge.training import Trainer, predict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _image_tensor(channels, height, width, fill=100.0):
    return Tensor(np.full((channels, height, width), fill, dtype=np.float32))


# -- serialize_transform / deserialize_transform -------------------------


def test_resize_round_trips():
    t = Resize((32, 48))
    node = serialize_transform(t)
    assert node == {"type": "Resize", "config": {"size": [32, 48]}}
    t2 = deserialize_transform(node)
    assert isinstance(t2, Resize)
    assert t2.size == (32, 48)


def test_normalize_scalar_round_trips():
    t = Normalize(mean=0.0, std=255.0)
    node = serialize_transform(t)
    t2 = deserialize_transform(node)
    x = _image_tensor(3, 4, 4, fill=127.5)
    np.testing.assert_allclose(t(x).numpy(), t2(x).numpy())


def test_normalize_per_channel_round_trips():
    # (C, 1, 1) so mean/std broadcast against a (C, H, W) sample the same
    # way Normalize's own docstring documents.
    mean = [[[0.1]], [[0.2]], [[0.3]]]
    std = [[[0.9]], [[0.8]], [[0.7]]]
    t = Normalize(mean=mean, std=std)
    node = serialize_transform(t)
    t2 = deserialize_transform(node)
    x = Tensor(np.random.default_rng(0).uniform(0, 1, size=(3, 5, 5)).astype(np.float32))
    np.testing.assert_allclose(t(x).numpy(), t2(x).numpy(), atol=1e-6)


def test_reshape_round_trips():
    t = Reshape(2, 6)
    node = serialize_transform(t)
    t2 = deserialize_transform(node)
    x = Tensor(np.arange(12, dtype=np.float32))
    assert t2(x).shape == (2, 6)
    np.testing.assert_allclose(t(x).numpy(), t2(x).numpy())


def test_flatten_round_trips():
    t = Flatten()
    node = serialize_transform(t)
    assert node == {"type": "Flatten", "config": {}}
    t2 = deserialize_transform(node)
    x = _image_tensor(3, 2, 2)
    assert t2(x).shape == (12,)


def test_to_tensor_round_trips():
    t = ToTensor(dtype="float32", device="cpu")
    node = serialize_transform(t)
    assert node == {"type": "ToTensor", "config": {"dtype": "float32", "device": "cpu"}}
    t2 = deserialize_transform(node)
    out = t2(np.array([1, 2, 3]))
    assert isinstance(out, Tensor)
    assert str(out.dtype) == "float32"


def test_to_tensor_none_dtype_round_trips():
    t = ToTensor()
    node = serialize_transform(t)
    assert node["config"]["dtype"] is None
    t2 = deserialize_transform(node)
    assert t2.dtype is None


def test_compose_round_trips_recursively():
    t = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])
    node = serialize_transform(t)
    assert node["type"] == "Compose"
    assert [c["type"] for c in node["config"]["transforms"]] == ["Resize", "Normalize"]

    t2 = deserialize_transform(node)
    assert isinstance(t2, Compose)
    x = _image_tensor(3, 20, 24, fill=200.0)
    np.testing.assert_allclose(t(x).numpy(), t2(x).numpy(), atol=1e-6)


def test_nested_compose_round_trips():
    inner = Compose([Resize((8, 8))])
    outer = Compose([inner, Normalize(mean=0.0, std=255.0)])
    node = serialize_transform(outer)
    t2 = deserialize_transform(node)
    x = _image_tensor(3, 10, 10, fill=50.0)
    np.testing.assert_allclose(outer(x).numpy(), t2(x).numpy(), atol=1e-6)


# -- rejection of unsupported / malformed transforms ----------------------


def test_lambda_is_not_registered():
    with pytest.raises(PersistenceError):
        spec_for_class(Lambda)


def test_serialize_lambda_raises_clearly():
    with pytest.raises(PersistenceError, match="Lambda|not registered"):
        serialize_transform(Lambda(lambda x: x))


def test_compose_containing_lambda_raises_at_serialize_time():
    t = Compose([Resize((8, 8)), Lambda(lambda x: x * 2.0)])
    with pytest.raises(PersistenceError):
        serialize_transform(t)


def test_deserialize_unknown_type_raises():
    with pytest.raises(PersistenceError):
        deserialize_transform({"type": "NotARealTransform", "config": {}})


def test_deserialize_malformed_node_raises():
    with pytest.raises(PersistenceError):
        deserialize_transform("not a dict")
    with pytest.raises(PersistenceError):
        deserialize_transform({"type": "Resize"})  # missing 'config'
    with pytest.raises(PersistenceError):
        deserialize_transform({"type": "Resize", "config": "not a dict"})


def test_deserialize_invalid_config_raises_persistence_error():
    # Resize's own validation (DataError) must surface as a PersistenceError
    # through this reconstruction path, not leak the underlying exception type.
    with pytest.raises(PersistenceError):
        deserialize_transform({"type": "Resize", "config": {"size": [0, 10]}})


def test_spec_for_name_unknown_raises():
    with pytest.raises(PersistenceError):
        spec_for_name("TotallyUnknownTransformType")


def test_register_transform_conflicting_name_raises():
    class _FakeResize(Transform):
        def __call__(self, sample):
            return sample

    with pytest.raises(PersistenceError):
        register_transform("Resize", _FakeResize, get_config=lambda m: {})


def test_register_transform_same_pair_is_noop_safe():
    # Re-registering Resize under its own existing name/class must not raise.
    register_transform("Resize", Resize, get_config=lambda m: {"size": list(m.size)})


# -- save_model(..., preprocessing=...) / load_preprocessing() ------------


def test_save_load_preprocessing_round_trip(tmp_path):
    model = Linear(4, 2)
    pre = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=pre)

    loaded_pre = load_preprocessing(str(path))
    assert isinstance(loaded_pre, Compose)
    x = _image_tensor(3, 20, 24, fill=180.0)
    np.testing.assert_allclose(pre(x).numpy(), loaded_pre(x).numpy(), atol=1e-6)

    # load_model() is unaffected -- still returns a bare, working Module.
    loaded_model = load_model(str(path))
    assert isinstance(loaded_model, Linear)


def test_save_model_without_preprocessing_defaults_to_none(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path))
    assert load_preprocessing(str(path)) is None


def test_save_model_preprocessing_none_explicit_matches_default(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    save_model(model, str(path), preprocessing=None)
    assert load_preprocessing(str(path)) is None


def test_save_model_with_lambda_preprocessing_raises_before_writing(tmp_path):
    model = Linear(4, 2)
    path = tmp_path / "model.forge"
    with pytest.raises(PersistenceError):
        save_model(model, str(path), preprocessing=Lambda(lambda x: x))
    assert not path.exists()


def test_backward_compatible_file_has_no_preprocessing_key(tmp_path):
    """A file saved with no `preprocessing=` argument at all (the pre-M71
    call shape) must still load cleanly through both `load_model()` and the
    new `load_preprocessing()` -- the latter returning `None`, not raising."""
    import json
    import zipfile

    from forge.serialization.archive import METADATA_ENTRY

    model = Linear(3, 3)
    path = tmp_path / "model.forge"
    save_model(model, str(path))

    with zipfile.ZipFile(str(path), "r") as zf:
        metadata = json.loads(zf.read(METADATA_ENTRY))
    assert metadata["preprocessing"] is None

    # Simulate a genuinely pre-M71 file (no "preprocessing" key present at
    # all, not even null) and confirm both APIs still handle it.
    del metadata["preprocessing"]
    tmp_zip = tmp_path / "legacy.forge"
    with zipfile.ZipFile(str(path), "r") as src, zipfile.ZipFile(str(tmp_zip), "w") as dst:
        for item in src.infolist():
            if item.filename == METADATA_ENTRY:
                dst.writestr(item, json.dumps(metadata))
            else:
                dst.writestr(item, src.read(item.filename))

    assert load_preprocessing(str(tmp_zip)) is None
    reloaded = load_model(str(tmp_zip))
    assert isinstance(reloaded, Linear)


def test_load_preprocessing_missing_file_raises():
    with pytest.raises(PersistenceError):
        load_preprocessing("does_not_exist.forge")


# -- end-to-end: ImageFolder + Resize/Normalize + model + fresh load ------


def _make_dataset(root: Path, per_class: int = 4):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for cls, base in (("a", 40), ("b", 210)):
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(per_class):
            h = int(rng.integers(20, 40))
            w = int(rng.integers(20, 40))
            fill = base + int(rng.integers(-5, 5))
            arr = np.clip(np.full((h, w, 3), fill), 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="RGB").save(cls_dir / f"{cls}_{i}.png")


class _TinyCNN(Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2)
        self.flatten = NNFlatten()
        self.fc = Linear(4 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(self.flatten(x))


register_module(
    "_TinyCNN_M71Test",
    _TinyCNN,
    get_config=lambda m: {"num_classes": m.fc.out_features},
)


def test_end_to_end_imagefolder_preprocessing_persists_and_reproduces(tmp_path):
    from forge.data import DataLoader, ImageFolder

    data_root = tmp_path / "data"
    _make_dataset(data_root, per_class=5)

    preprocessing = Compose([Resize((16, 16)), Normalize(mean=0.0, std=255.0)])
    dataset = ImageFolder(str(data_root), transform=preprocessing)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    forge.random.seed(0)
    model = _TinyCNN(num_classes=len(dataset.classes))
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, loss_fn=CrossEntropyLoss(), optimizer=optimizer, device="cpu")
    trainer.fit(loader, epochs=1)

    model_path = tmp_path / "artifact.forge"
    save_model(model, str(model_path), preprocessing=preprocessing)

    # Simulate a fresh process: only the file path is known from here on,
    # no in-memory reference to `preprocessing`/`model`/`dataset` is reused.
    del preprocessing, model, dataset, loader, trainer

    reloaded_model = load_model(str(model_path), device="cpu")
    reloaded_preprocessing = load_preprocessing(str(model_path))
    assert isinstance(reloaded_preprocessing, Compose)

    # A brand-new image, a different source resolution than anything in the
    # training set, decoded fresh via ImageFolder's own loader.
    new_path = tmp_path / "new_query.png"
    Image.fromarray(np.full((77, 61, 3), 205, dtype=np.uint8), mode="RGB").save(new_path)
    raw = ImageFolder._load_image(new_path)
    assert raw.shape != (3, 16, 16)  # genuinely a different source resolution

    preprocessed = reloaded_preprocessing(raw)
    assert preprocessed.shape == (3, 16, 16)

    batch = preprocessed.reshape(1, *preprocessed.shape)
    prediction = predict(reloaded_model, batch)
    assert prediction.shape == (1, 2)
    # A finite, real prediction -- not NaN/inf from a mis-scaled input.
    assert np.all(np.isfinite(prediction.numpy()))


def test_end_to_end_training_and_inference_time_preprocessing_are_identical(tmp_path):
    """Training-time (in-DataLoader) and inference-time (post-load,
    single-image) preprocessing must be the exact same reconstructed
    transform applied identically -- not two independently-written paths
    that merely happen to agree."""
    from forge.data import ImageFolder

    data_root = tmp_path / "data"
    _make_dataset(data_root, per_class=3)

    preprocessing = Compose([Resize((12, 12)), Normalize(mean=0.0, std=255.0)])
    dataset = ImageFolder(str(data_root), transform=preprocessing)
    model = _TinyCNN(num_classes=len(dataset.classes))

    model_path = tmp_path / "artifact.forge"
    save_model(model, str(model_path), preprocessing=preprocessing)

    reloaded_preprocessing = load_preprocessing(str(model_path))

    sample_path, _ = dataset.samples[0]
    raw = ImageFolder._load_image(sample_path)
    train_time = preprocessing(raw)
    inference_time = reloaded_preprocessing(raw)
    np.testing.assert_allclose(train_time.numpy(), inference_time.numpy(), atol=1e-6)
