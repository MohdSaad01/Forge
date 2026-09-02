"""Milestone 18 tests: CUDA training checkpoint save/load.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_persistence.py`. CPU-only checkpoint
behavior (including CUDA-unavailable metadata-level policy checks) lives in
`tests/test_checkpoint.py`. See `docs/architecture/persistence.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import Linear
from forge.nn.loss import MSELoss
from forge.optim import SGD, Adam
from forge.serialization import load_checkpoint, save_checkpoint

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


def _train_steps(model, optimizer, x, n):
    for _ in range(n):
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()


# -- CUDA -> CUDA round trip -------------------------------------------------


def test_cuda_checkpoint_records_cuda_device(tmp_path):
    import json
    import zipfile

    from forge.serialization.archive import METADATA_ENTRY

    model = Linear(4, 3).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)).to("cuda")
    _train_steps(model, optimizer, x, 2)

    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)
    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read(METADATA_ENTRY))
    assert metadata["device"] == "cuda"


def test_cuda_to_cuda_adam_state_round_trip(tmp_path):
    forge.random.seed(5)
    model = Linear(4, 3).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99), weight_decay=0.001)
    x = Tensor(np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)).to("cuda")
    _train_steps(model, optimizer, x, 4)

    path = tmp_path / "cuda_adam.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=4)
    checkpoint = load_checkpoint(str(path))

    assert checkpoint.epoch == 1
    assert checkpoint.global_step == 4

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        assert param.device.type == "cuda"
        assert isinstance(param._data, CUDAStorage)
        original = original_by_name[name]
        np.testing.assert_allclose(param.to("cpu").numpy(), original.to("cpu").numpy(), **TOL)

        original_state = optimizer.state[original]
        restored_state = checkpoint.optimizer.state[param]
        assert isinstance(restored_state.m, CUDAStorage)
        assert isinstance(restored_state.v, CUDAStorage)
        assert restored_state.step == original_state.step
        from forge.backend import get_backend

        backend = get_backend(original_state.device)
        np.testing.assert_allclose(
            backend.to_numpy(restored_state.m), backend.to_numpy(original_state.m), **TOL
        )
        np.testing.assert_allclose(
            backend.to_numpy(restored_state.v), backend.to_numpy(original_state.v), **TOL
        )


def test_cuda_to_cuda_one_more_step_matches_continuing_original(tmp_path):
    forge.random.seed(6)
    model = Linear(4, 2).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.02)
    x = Tensor(np.random.default_rng(2).standard_normal((5, 4)).astype(np.float32)).to("cuda")
    _train_steps(model, optimizer, x, 4)

    path = tmp_path / "cuda_next.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=4)

    _train_steps(model, optimizer, x, 1)

    checkpoint = load_checkpoint(str(path))
    _train_steps(checkpoint.model, checkpoint.optimizer, x, 1)

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(
            param.to("cpu").numpy(), original_by_name[name].to("cpu").numpy(), **TOL
        )


def test_cuda_sgd_checkpoint_round_trip(tmp_path):
    model = Linear(3, 2).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.1)
    x = Tensor(np.random.default_rng(3).standard_normal((4, 3)).astype(np.float32)).to("cuda")
    _train_steps(model, optimizer, x, 3)

    path = tmp_path / "cuda_sgd.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=3)
    checkpoint = load_checkpoint(str(path))

    assert isinstance(checkpoint.optimizer, SGD)
    assert checkpoint.optimizer.lr == pytest.approx(0.1)
    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        assert param.device.type == "cuda"
        np.testing.assert_allclose(
            param.to("cpu").numpy(), original_by_name[name].to("cpu").numpy(), **TOL
        )


# -- cross-device conversion (section 13) ------------------------------------


def test_cuda_to_cpu_explicit_override(tmp_path):
    model = Linear(4, 3).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(4).standard_normal((4, 4)).astype(np.float32)).to("cuda")
    _train_steps(model, optimizer, x, 3)

    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=3)
    checkpoint = load_checkpoint(str(path), device="cpu")

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        assert param.device.type == "cpu"
        assert isinstance(param._data, np.ndarray)
        np.testing.assert_allclose(param.numpy(), original_by_name[name].to("cpu").numpy(), **TOL)

        original_state = optimizer.state[original_by_name[name]]
        restored_state = checkpoint.optimizer.state[param]
        assert isinstance(restored_state.m, np.ndarray)
        assert isinstance(restored_state.v, np.ndarray)
        from forge.backend import get_backend

        backend = get_backend(original_state.device)
        np.testing.assert_allclose(backend.to_numpy(original_state.m), restored_state.m, **TOL)
        np.testing.assert_allclose(backend.to_numpy(original_state.v), restored_state.v, **TOL)
        assert original_state.step == restored_state.step


def test_cpu_to_cuda_explicit_override(tmp_path):
    model = Linear(4, 3)  # CPU model
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(5).standard_normal((4, 4)).astype(np.float32))
    _train_steps(model, optimizer, x, 3)

    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=3)
    checkpoint = load_checkpoint(str(path), device="cuda")

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        assert param.device.type == "cuda"
        assert isinstance(param._data, CUDAStorage)
        np.testing.assert_allclose(param.to("cpu").numpy(), original_by_name[name].numpy(), **TOL)

        original_state = optimizer.state[original_by_name[name]]
        restored_state = checkpoint.optimizer.state[param]
        assert isinstance(restored_state.m, CUDAStorage)
        from forge.backend import get_backend

        restored_backend = get_backend(restored_state.device)
        np.testing.assert_allclose(
            restored_backend.to_numpy(restored_state.m), original_state.m, **TOL
        )


# -- no CPU computational fallback (section 14) ------------------------------


def test_save_load_cuda_checkpoint_never_calls_cpu_backend_compute_ops(tmp_path, monkeypatch):
    model = Linear(4, 3).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(6).standard_normal((3, 4)).astype(np.float32)).to("cuda")
    _train_steps(model, optimizer, x, 2)

    calls: list[str] = []
    compute_ops = (
        "add", "sub", "mul", "matmul", "sum", "reshape", "relu", "exp", "log",
        "sgd_step", "adam_step",
    )
    for name in compute_ops:
        original = getattr(CPUBackend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=2)
    checkpoint = load_checkpoint(str(path))
    with forge.no_grad():
        checkpoint.model(x)

    assert calls == []


# -- resume equivalence (section 19, CUDA + Adam) ----------------------------


def test_resume_equivalence_cuda_adam(tmp_path):
    forge.random.seed(77)
    model = Linear(4, 2).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)).to("cuda")

    _train_steps(model, optimizer, x, 5)

    path = tmp_path / "resume.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=5)

    _train_steps(model, optimizer, x, 3)

    checkpoint = load_checkpoint(str(path))
    _train_steps(checkpoint.model, checkpoint.optimizer, x, 3)

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        original = original_by_name[name]
        np.testing.assert_allclose(param.to("cpu").numpy(), original.to("cpu").numpy(), **TOL)

        original_state = optimizer.state[original]
        restored_state = checkpoint.optimizer.state[param]
        assert original_state.step == restored_state.step


# -- Trainer + CUDA -----------------------------------------------------------


def test_cuda_trainer_checkpoint_and_resume(tmp_path):
    from forge.data import DataLoader, TensorDataset
    from forge.training import Trainer

    forge.random.seed(9)
    rng = np.random.default_rng(9)
    n = 32
    X = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    y = (2 * X[:, 0] - X[:, 1] + 0.3).reshape(-1, 1).astype(np.float32)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = Linear(3, 1).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    path = tmp_path / "trainer_cuda.ckpt"
    trainer.save_checkpoint(path=str(path))

    checkpoint = load_checkpoint(str(path))
    fresh_model = Linear(3, 1).to("cuda")
    fresh_optimizer = Adam(fresh_model.parameters(), lr=0.05)
    trainer2 = Trainer(model=fresh_model, loss_fn=MSELoss(), optimizer=fresh_optimizer, device="cuda", verbose=False)
    trainer2.resume(checkpoint)
    assert trainer2.epoch == 2

    history = trainer2.fit(loader, epochs=2)
    assert [r.epoch for r in history] == [3, 4]
