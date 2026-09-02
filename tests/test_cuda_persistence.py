"""Milestone 13 tests: CUDA model persistence.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py`/`tests/test_module_cuda.py`/
`tests/test_trainer_cuda.py`. CPU-only persistence behavior (including the
metadata-level CUDA-unavailable policy checks that don't require real
hardware) lives in `tests/test_serialization.py`. See
`docs/architecture/persistence.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential
from forge.serialization import load_model, register_module, save_model

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


class CudaMLP(Module):
    """A hand-composed nested model, distinct from `test_serialization.py`'s
    `MLP` to avoid a registry name collision across test modules."""

    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


register_module(
    "CudaMLP",
    CudaMLP,
    get_config=lambda m: {
        "in_features": m.fc1.in_features,
        "hidden": m.fc1.out_features,
        "out_features": m.fc2.out_features,
    },
)


# -- CUDA -> CUDA round trip ------------------------------------------------


def test_cuda_saved_metadata_records_cuda_device(tmp_path):
    import json
    import zipfile

    from forge.serialization.archive import METADATA_ENTRY

    model = Linear(4, 8).to("cuda")
    path = tmp_path / "m.forge"
    save_model(model, str(path))

    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read(METADATA_ENTRY))
    assert metadata["device"] == "cuda"


def test_cuda_to_cuda_round_trip_parameters_are_cuda_and_match_values(tmp_path):
    model = Linear(4, 8).to("cuda")
    original = {name: np.array(p.to("cpu").numpy(), copy=True) for name, p in model.named_parameters()}

    path = tmp_path / "m.forge"
    save_model(model, str(path))
    del model
    loaded = load_model(str(path))

    for name, param in loaded.named_parameters():
        assert param.device.type == "cuda"
        assert isinstance(param._data, CUDAStorage)
        np.testing.assert_allclose(param.to("cpu").numpy(), original[name], **TOL)


def test_cuda_to_cuda_round_trip_shapes_dtypes_and_requires_grad_match(tmp_path):
    model = Linear(4, 8).to("cuda")
    path = tmp_path / "m.forge"

    save_model(model, str(path))
    loaded = load_model(str(path))
    for name, param in loaded.named_parameters():
        original = dict(model.named_parameters())[name]
        assert param.shape == original.shape
        assert param.dtype == original.dtype
        assert param.requires_grad == original.requires_grad


def test_cuda_to_cuda_round_trip_forward_output_matches(tmp_path):
    forge.random.seed(42)
    model = CudaMLP(4, 8, 2).to("cuda")
    path = tmp_path / "m.forge"

    x = Tensor(np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)).to("cuda")
    with no_grad():
        original_out = model(x).to("cpu").numpy()

    save_model(model, str(path))
    del model
    loaded = load_model(str(path))
    assert loaded.fc1.weight.device.type == "cuda"

    with no_grad():
        loaded_out = loaded(x).to("cpu").numpy()
    np.testing.assert_allclose(original_out, loaded_out, **TOL)


def test_cuda_nested_model_hierarchy_and_training_mode_round_trip(tmp_path):
    model = CudaMLP(3, 4, 2).to("cuda")
    model.eval()
    model.fc2.train()

    path = tmp_path / "m.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert isinstance(loaded, CudaMLP)
    assert isinstance(loaded.fc1, Linear)
    assert isinstance(loaded.relu, ReLU)
    assert isinstance(loaded.fc2, Linear)
    assert loaded.training is False
    assert loaded.fc1.training is False
    assert loaded.fc2.training is True
    for _, param in loaded.named_parameters():
        assert param.device.type == "cuda"


def test_loaded_cuda_parameters_are_fresh_leaves_with_no_grad_state(tmp_path):
    model = Linear(3, 2).to("cuda")
    path = tmp_path / "m.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))

    assert loaded.weight.is_leaf
    assert loaded.weight.grad_fn is None
    assert loaded.weight.grad is None
    assert loaded.weight.requires_grad is True
    assert loaded.weight.device.type == "cuda"


# -- Sequential / Flatten / Dropout CUDA round trip (Milestone 16) -----------


def test_cuda_sequential_cnn_round_trip_predictions_match(tmp_path):
    forge.random.seed(9)
    model = Sequential(
        Conv2d(1, 4, kernel_size=3), ReLU(), MaxPool2d(2), Flatten(), Linear(36, 8), ReLU(),
        Dropout(0.3), Linear(8, 2),
    ).to("cuda")
    model.eval()  # deterministic predictions across the round-trip (Dropout is identity)

    x = Tensor(np.random.default_rng(10).standard_normal((3, 1, 8, 8)).astype(np.float32)).to("cuda")
    with no_grad():
        original_out = model(x).to("cpu").numpy()

    path = tmp_path / "sequential_cnn.forge"
    save_model(model, str(path))
    del model
    loaded = load_model(str(path))

    assert isinstance(loaded, Sequential)
    assert [name for name, _ in loaded.named_children()] == ["0", "1", "2", "3", "4", "5", "6", "7"]
    for _, param in loaded.named_parameters():
        assert param.device.type == "cuda"
        assert isinstance(param._data, CUDAStorage)

    with no_grad():
        loaded_out = loaded(x).to("cpu").numpy()
    np.testing.assert_allclose(original_out, loaded_out, **TOL)


# -- Explicit device-override conversions -----------------------------------


def test_cuda_to_cpu_explicit_override_round_trip_matches(tmp_path):
    model = Linear(4, 8).to("cuda")
    original = {name: np.array(p.to("cpu").numpy(), copy=True) for name, p in model.named_parameters()}

    path = tmp_path / "m.forge"
    save_model(model, str(path))
    loaded = load_model(str(path), device="cpu")

    for name, param in loaded.named_parameters():
        assert param.device.type == "cpu"
        assert isinstance(param._data, np.ndarray)
        np.testing.assert_allclose(param.numpy(), original[name], **TOL)


def test_cpu_to_cuda_explicit_override_round_trip_matches(tmp_path):
    model = Linear(4, 8)  # CPU model
    original = {name: np.array(p.numpy(), copy=True) for name, p in model.named_parameters()}

    path = tmp_path / "m.forge"
    save_model(model, str(path))
    loaded = load_model(str(path), device="cuda")

    for name, param in loaded.named_parameters():
        assert param.device.type == "cuda"
        assert isinstance(param._data, CUDAStorage)
        np.testing.assert_allclose(param.to("cpu").numpy(), original[name], **TOL)


# -- No CPU computational fallback -------------------------------------------


def test_save_load_cuda_model_never_calls_cpu_backend_compute_ops(tmp_path, monkeypatch):
    """Structural proof that saving/loading a CUDA model transfers bytes
    (`Backend.to_numpy`/`Backend.from_array`) without ever running a
    `CPUBackend` *computation* op."""
    model = Linear(4, 8).to("cuda")
    x = Tensor(np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)).to("cuda")

    calls: list[str] = []
    compute_ops = ("add", "sub", "mul", "matmul", "sum", "reshape", "relu", "exp", "log", "sgd_step")
    for name in compute_ops:
        original = getattr(CPUBackend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    path = tmp_path / "m.forge"
    save_model(model, str(path))
    loaded = load_model(str(path))
    with no_grad():
        loaded(x)

    assert calls == []


# -- Real training -> save -> load -> predict --------------------------------


def test_cuda_trained_model_can_be_saved_and_reloaded_after_training(tmp_path):
    from forge.data import DataLoader, TensorDataset
    from forge.nn.loss import MSELoss
    from forge.optim import SGD
    from forge.training import Trainer

    forge.random.seed(7)
    rng = np.random.default_rng(7)

    n = 32
    X = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    y = (2 * X[:, 0] - 1 * X[:, 1] + 0.5 * X[:, 2] + 0.3).reshape(-1, 1).astype(np.float32)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(5))

    model = Linear(3, 1).to("cuda")
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=10)

    x_query = Tensor(rng.uniform(-1, 1, size=(4, 3)).astype(np.float32)).to("cuda")
    with no_grad():
        pre_save_prediction = model(x_query).to("cpu").numpy()

    path = tmp_path / "trained.forge"
    save_model(model, str(path))

    del model
    loaded = load_model(str(path))
    assert loaded.weight.device.type == "cuda"
    with no_grad():
        post_load_prediction = loaded(x_query).to("cpu").numpy()

    np.testing.assert_allclose(pre_save_prediction, post_load_prediction, atol=1e-6)
