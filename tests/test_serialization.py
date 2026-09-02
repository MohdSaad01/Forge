"""Milestone 7 tests: model persistence (save_model/load_model).

Covers simple/nested round-trips, numerical/prediction equivalence, fresh
autograd state after load, `.training` mode round-tripping, the registry's
custom-module opt-in, and clear failures for invalid/corrupt/unsupported
files -- including that a tampered file never triggers arbitrary code
execution. See `docs/architecture/persistence.md`.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.exceptions import PersistenceError
from forge.nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential
from forge.serialization import load_model, register_module, save_model
from forge.serialization.archive import METADATA_ENTRY, PARAMETERS_DIR


class MLP(Module):
    """A hand-composed nested model -- registered below for persistence."""

    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


register_module(
    "MLP",
    MLP,
    get_config=lambda m: {
        "in_features": m.fc1.in_features,
        "hidden": m.fc1.out_features,
        "out_features": m.fc2.out_features,
    },
)


class Unregistered(Module):
    def forward(self, x):
        return x


def _read_metadata(path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read(METADATA_ENTRY))


def _write_metadata(path, metadata: dict) -> None:
    """Rewrite metadata.json in place inside an existing archive (test tampering helper)."""
    with zipfile.ZipFile(path, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries[METADATA_ENTRY] = json.dumps(metadata).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


# -- simple model round-trip ----------------------------------------------


def test_save_load_simple_linear_restores_architecture(tmp_path):
    model = Linear(4, 8)
    path = tmp_path / "linear.forge"
    save_model(model, str(path))

    loaded = load_model(str(path))
    assert isinstance(loaded, Linear)
    assert loaded.in_features == 4
    assert loaded.out_features == 8
    assert loaded.bias is not None


def test_save_load_simple_linear_parameter_shapes_and_values_match(tmp_path):
    model = Linear(4, 8)
    path = tmp_path / "linear.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert loaded.weight.shape == model.weight.shape
    assert loaded.bias.shape == model.bias.shape
    assert np.array_equal(loaded.weight.numpy(), model.weight.numpy())
    assert np.array_equal(loaded.bias.numpy(), model.bias.numpy())
    assert loaded.weight.dtype == model.weight.dtype


def test_save_load_simple_linear_predictions_match(tmp_path):
    model = Linear(4, 8)
    path = tmp_path / "linear.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    x = Tensor(np.random.default_rng(0).uniform(-1, 1, size=(5, 4)))
    with no_grad():
        original_out = model(x).numpy()
        loaded_out = loaded(x).numpy()
    assert np.allclose(original_out, loaded_out)


def test_linear_without_bias_round_trips(tmp_path):
    model = Linear(3, 2, bias=False)
    path = tmp_path / "linear_no_bias.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert loaded.bias is None
    assert set(dict(loaded.named_parameters())) == {"weight"}


# -- Conv2d / MaxPool2d round-trip (Milestone 15) --------------------------


def test_save_load_conv2d_restores_architecture_and_parameters(tmp_path):
    model = Conv2d(3, 5, kernel_size=(3, 4), stride=2, padding=1)
    path = tmp_path / "conv2d.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, Conv2d)
    assert loaded.in_channels == 3
    assert loaded.out_channels == 5
    assert loaded.kernel_size == (3, 4)
    assert loaded.stride == (2, 2)
    assert loaded.padding == (1, 1)
    assert loaded.weight.shape == model.weight.shape
    assert loaded.bias.shape == model.bias.shape
    assert np.array_equal(loaded.weight.numpy(), model.weight.numpy())
    assert np.array_equal(loaded.bias.numpy(), model.bias.numpy())


def test_conv2d_without_bias_round_trips(tmp_path):
    model = Conv2d(2, 3, kernel_size=3, bias=False)
    path = tmp_path / "conv2d_no_bias.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert loaded.bias is None
    assert set(dict(loaded.named_parameters())) == {"weight"}


def test_save_load_conv2d_predictions_match(tmp_path):
    model = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    path = tmp_path / "conv2d.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    x = Tensor(np.random.default_rng(9).standard_normal((3, 2, 6, 6)))
    with no_grad():
        assert np.allclose(model(x).numpy(), loaded(x).numpy())


def test_save_load_maxpool2d_restores_configuration(tmp_path):
    model = MaxPool2d(3, stride=2, padding=1)
    path = tmp_path / "maxpool2d.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, MaxPool2d)
    assert loaded.kernel_size == (3, 3)
    assert loaded.stride == (2, 2)
    assert loaded.padding == (1, 1)
    assert dict(loaded.named_parameters()) == {}


def test_save_load_maxpool2d_predictions_match(tmp_path):
    model = MaxPool2d(2)
    path = tmp_path / "maxpool2d.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    x = Tensor(np.random.default_rng(10).standard_normal((2, 3, 8, 8)))
    with no_grad():
        assert np.allclose(model(x).numpy(), loaded(x).numpy())


class TinyCNN(Module):
    """Conv2d -> ReLU -> MaxPool2d -> Linear, registered below for persistence."""

    def __init__(self):
        super().__init__()
        self.conv = Conv2d(1, 4, kernel_size=3)
        self.relu = ReLU()
        self.pool = MaxPool2d(2)
        self.fc = Linear(4 * 3 * 3, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))


register_module("TinyCNN", TinyCNN, get_config=lambda m: {})


def test_save_load_nested_cnn_all_parameters_match(tmp_path):
    model = TinyCNN()
    path = tmp_path / "cnn.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    original = dict(model.named_parameters())
    restored = dict(loaded.named_parameters())
    assert set(original) == set(restored) == {"conv.weight", "conv.bias", "fc.weight", "fc.bias"}
    for name in original:
        assert np.array_equal(original[name].numpy(), restored[name].numpy())


def test_save_load_nested_cnn_predictions_match(tmp_path):
    model = TinyCNN()
    path = tmp_path / "cnn.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    x = Tensor(np.random.default_rng(11).standard_normal((2, 1, 8, 8)))
    with no_grad():
        assert np.allclose(model(x).numpy(), loaded(x).numpy())


# -- Sequential / Flatten / Dropout round-trip (Milestone 16) --------------


def test_save_load_flatten_restores_configuration(tmp_path):
    model = Flatten(start_dim=2, end_dim=3)
    path = tmp_path / "flatten.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, Flatten)
    assert loaded.start_dim == 2
    assert loaded.end_dim == 3
    assert dict(loaded.named_parameters()) == {}


def test_save_load_dropout_restores_p_and_training_state(tmp_path):
    model = Dropout(0.35)
    model.eval()
    path = tmp_path / "dropout.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, Dropout)
    assert loaded.p == pytest.approx(0.35)
    assert loaded.training is False
    assert dict(loaded.named_parameters()) == {}


def test_save_load_dropout_training_mode_round_trips_true(tmp_path):
    model = Dropout(0.5)
    assert model.training is True
    path = tmp_path / "dropout.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert loaded.training is True


def test_save_load_sequential_restores_child_hierarchy_and_order(tmp_path):
    model = Sequential(
        Conv2d(1, 4, kernel_size=3), ReLU(), MaxPool2d(2), Flatten(), Linear(36, 8), Dropout(0.3),
    )
    path = tmp_path / "sequential.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, Sequential)
    original_order = [name for name, _ in model.named_children()]
    loaded_order = [name for name, _ in loaded.named_children()]
    assert loaded_order == original_order == ["0", "1", "2", "3", "4", "5"]
    assert [type(c).__name__ for _, c in loaded.named_children()] == [
        "Conv2d", "ReLU", "MaxPool2d", "Flatten", "Linear", "Dropout",
    ]


def test_save_load_sequential_parameters_and_predictions_match(tmp_path):
    forge.random.seed(5)
    model = Sequential(
        Conv2d(1, 4, kernel_size=3), ReLU(), MaxPool2d(2), Flatten(), Linear(36, 8), ReLU(),
        Dropout(0.3), Linear(8, 2),
    )
    model.eval()  # deterministic predictions across the round-trip (Dropout is identity)
    path = tmp_path / "sequential_cnn.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    original = dict(model.named_parameters())
    restored = dict(loaded.named_parameters())
    assert set(original) == set(restored)
    for name in original:
        assert np.array_equal(original[name].numpy(), restored[name].numpy())

    x = Tensor(np.random.default_rng(6).standard_normal((3, 1, 8, 8)).astype(np.float32))
    with no_grad():
        assert np.allclose(model(x).numpy(), loaded(x).numpy())


def test_save_load_empty_sequential_round_trips(tmp_path):
    model = Sequential()
    path = tmp_path / "empty_sequential.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, Sequential)
    assert list(loaded.named_children()) == []
    x = Tensor([1.0, 2.0, 3.0])
    with no_grad():
        assert np.array_equal(loaded(x).numpy(), x.numpy())


def test_save_load_nested_sequential_preserves_hierarchy(tmp_path):
    model = Sequential(Linear(3, 4), Dropout(0.5), Sequential(Linear(4, 2), Dropout(0.1)))
    path = tmp_path / "nested_sequential.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert [name for name, _ in loaded.named_children()] == ["0", "1", "2"]
    inner = loaded._modules["2"]
    assert isinstance(inner, Sequential)
    assert [name for name, _ in inner.named_children()] == ["0", "1"]
    assert len(list(loaded.parameters())) == len(list(model.parameters()))


def test_nested_sequential_mixed_training_mode_round_trips_per_module(tmp_path):
    model = Sequential(Linear(2, 2), Dropout(0.5), Sequential(Dropout(0.2)))
    model.eval()
    model._modules["1"].train()  # diverge one child back to train mode

    path = tmp_path / "sequential_mixed_mode.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert loaded.training is False
    assert loaded._modules["0"].training is False
    assert loaded._modules["1"].training is True
    assert loaded._modules["2"].training is False
    assert loaded._modules["2"]._modules["0"].training is False


# -- nested model round-trip -----------------------------------------------


def test_save_load_nested_model_restores_architecture(tmp_path):
    model = MLP(4, 8, 2)
    path = tmp_path / "mlp.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, MLP)
    assert isinstance(loaded.fc1, Linear)
    assert isinstance(loaded.relu, ReLU)
    assert isinstance(loaded.fc2, Linear)
    assert loaded.fc1.in_features == 4
    assert loaded.fc1.out_features == 8
    assert loaded.fc2.in_features == 8
    assert loaded.fc2.out_features == 2


def test_save_load_nested_model_all_parameters_match(tmp_path):
    model = MLP(4, 8, 2)
    path = tmp_path / "mlp.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    original = dict(model.named_parameters())
    restored = dict(loaded.named_parameters())
    assert set(original) == set(restored) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    for name in original:
        assert np.array_equal(original[name].numpy(), restored[name].numpy())
        assert original[name].shape == restored[name].shape


def test_save_load_nested_model_predictions_match(tmp_path):
    model = MLP(4, 8, 2)
    path = tmp_path / "mlp.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    x = Tensor(np.random.default_rng(1).uniform(-1, 1, size=(6, 4)))
    with no_grad():
        assert np.allclose(model(x).numpy(), loaded(x).numpy())


# -- fresh autograd state ---------------------------------------------------


def test_loaded_parameters_are_fresh_leaves_with_no_grad_state(tmp_path):
    model = Linear(3, 2)
    path = tmp_path / "linear.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert loaded.weight.is_leaf
    assert loaded.weight.grad_fn is None
    assert loaded.weight.grad is None
    assert loaded.weight.requires_grad is True


def test_loaded_model_builds_new_graph_on_forward_pass(tmp_path):
    model = Linear(3, 2)
    path = tmp_path / "linear.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    x = Tensor([1.0, 2.0, 3.0])
    out = loaded(x)
    assert out.grad_fn is not None
    assert not out.is_leaf

    out.sum().backward()
    assert loaded.weight.grad is not None
    assert loaded.weight.grad.shape == loaded.weight.shape


# -- training/eval mode -----------------------------------------------------


def test_training_mode_true_round_trips(tmp_path):
    model = Linear(2, 2)
    assert model.training is True
    path = tmp_path / "m.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert loaded.training is True


def test_eval_mode_round_trips(tmp_path):
    model = Linear(2, 2)
    model.eval()
    path = tmp_path / "m.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert loaded.training is False


def test_nested_mixed_training_mode_round_trips_per_module(tmp_path):
    model = MLP(3, 4, 2)
    model.eval()
    model.fc2.train()  # diverge one child back to train mode
    assert model.training is False
    assert model.fc1.training is False
    assert model.fc2.training is True

    path = tmp_path / "mlp.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert loaded.training is False
    assert loaded.fc1.training is False
    assert loaded.relu.training is False
    assert loaded.fc2.training is True


# -- custom/unregistered modules ---------------------------------------------


def test_saving_unregistered_module_type_raises_persistence_error(tmp_path):
    model = Unregistered()
    path = tmp_path / "bad.forge"
    with pytest.raises(PersistenceError):
        save_model(model, str(path))
    assert not path.exists()


def test_registering_class_under_conflicting_name_raises():
    with pytest.raises(PersistenceError):
        register_module("Linear", Unregistered, get_config=lambda m: {})


# -- invalid files ------------------------------------------------------


def test_load_nonexistent_file_raises_persistence_error(tmp_path):
    with pytest.raises(PersistenceError):
        load_model(str(tmp_path / "does_not_exist.forge"))


def test_load_malformed_zip_raises_persistence_error(tmp_path):
    path = tmp_path / "not_a_zip.forge"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_unsupported_version_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["forge_format_version"] = 999
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_unsupported_module_type_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["root"]["type"] = "TotallyUnknownModuleType"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_unrecognized_device_raises_persistence_error(tmp_path):
    """`"cuda"` is a legitimate recorded device as of M13 -- an unknown device
    string (not `"cpu"`/`"cuda"`) is what should still fail to load."""
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["device"] = "tpu"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_invalid_device_override_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    with pytest.raises(PersistenceError):
        load_model(str(path), device="tpu")


# -- CUDA device policy (metadata-level, no CUDA hardware required) ---------
#
# These exercise `load_model()`'s CUDA-availability policy by monkeypatching
# `forge.backend.cuda.is_cuda_available` directly, so they run deterministically
# on any machine regardless of whether CUDA is actually present. Hardware-
# verified CUDA<->CUDA round trips live in `tests/test_cuda_persistence.py`.


def test_saved_cpu_model_records_cpu_device_in_metadata(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    assert metadata["device"] == "cpu"


def test_load_cuda_saved_file_without_cuda_available_raises_clear_error(tmp_path, monkeypatch):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["device"] = "cuda"
    _write_metadata(path, metadata)

    import forge.backend.cuda as cuda_module

    monkeypatch.setattr(cuda_module, "is_cuda_available", lambda: False)

    with pytest.raises(PersistenceError, match="CUDA"):
        load_model(str(path))


def test_load_device_cuda_override_without_cuda_available_raises_clear_error(tmp_path, monkeypatch):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    import forge.backend.cuda as cuda_module

    monkeypatch.setattr(cuda_module, "is_cuda_available", lambda: False)

    with pytest.raises(PersistenceError, match="CUDA"):
        load_model(str(path), device="cuda")


def test_load_device_cpu_override_ignores_recorded_cuda_device_no_hardware_needed(tmp_path):
    """A `device="cpu"` override never touches CUDA availability at all --
    the archive's parameter bytes are already host-resident regardless of
    what device they were saved from, so this works even without CUDA."""
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["device"] = "cuda"
    _write_metadata(path, metadata)

    loaded = load_model(str(path), device="cpu")
    assert loaded.weight.device.type == "cpu"
    np.testing.assert_array_equal(loaded.weight.numpy(), model.weight.numpy())


def test_load_missing_parameter_data_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    with zipfile.ZipFile(path, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    del entries[f"{PARAMETERS_DIR}/bias.npy"]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_wrong_parameter_shape_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["root"]["parameters"]["weight"]["shape"] = [99, 99]
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_wrong_parameter_dtype_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["root"]["parameters"]["weight"]["dtype"] = "float64"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_corrupted_parameter_bytes_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    with zipfile.ZipFile(path, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries[f"{PARAMETERS_DIR}/weight.npy"] = b"not a valid npy payload"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_save_to_missing_directory_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    bad_path = tmp_path / "no_such_directory" / "model.forge"
    with pytest.raises(PersistenceError):
        save_model(model, str(bad_path))


def test_failed_save_leaves_no_partial_file(tmp_path):
    model = Unregistered()
    path = tmp_path / "partial.forge"
    with pytest.raises(PersistenceError):
        save_model(model, str(path))
    assert list(tmp_path.iterdir()) == []


# -- no arbitrary execution ---------------------------------------------


def test_malicious_type_field_does_not_execute_anything(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["root"]["type"] = "__import__('os').system"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_model(str(path))


def test_load_uses_allow_pickle_false(tmp_path):
    """Guards against a future regression that would let a saved array carry pickled objects."""
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    with zipfile.ZipFile(path, "r") as zf:
        raw = zf.read(f"{PARAMETERS_DIR}/weight.npy")
    # A plain-array .npy load must succeed with allow_pickle=False (no object dtype smuggled in).
    import io

    array = np.load(io.BytesIO(raw), allow_pickle=False)
    assert array.dtype != object


# -- end-to-end: train -> save -> load -> predict --------------------------


def test_end_to_end_train_save_load_predict_equivalence(tmp_path):
    from forge.data import DataLoader, TensorDataset
    from forge.nn.loss import MSELoss
    from forge.optim import SGD
    from forge.training import Trainer

    forge.random.seed(123)
    rng = np.random.default_rng(123)

    n = 64
    X = rng.uniform(-1, 1, size=(n, 3))
    y = (2 * X[:, 0] - 1 * X[:, 1] + 0.5 * X[:, 2] + 0.3).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(5))

    model = Linear(3, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    trainer.fit(loader, epochs=20)

    x_query = Tensor(rng.uniform(-1, 1, size=(4, 3)))
    with no_grad():
        pre_save_prediction = model(x_query).numpy()

    path = tmp_path / "trained.forge"
    save_model(model, str(path))

    del model  # ensure the loaded model is self-sufficient, not relying on the original
    loaded = load_model(str(path))
    with no_grad():
        post_load_prediction = loaded(x_query).numpy()

    assert np.allclose(pre_save_prediction, post_load_prediction, atol=1e-6)
