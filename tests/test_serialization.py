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
from forge.nn import Linear, Module, ReLU
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


def test_load_unsupported_device_raises_persistence_error(tmp_path):
    model = Linear(2, 2)
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    metadata = _read_metadata(path)
    metadata["device"] = "cuda"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_model(str(path))


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
