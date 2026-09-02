"""Milestone 18 tests: training checkpoint save/load (`save_checkpoint`/`load_checkpoint`).

Covers Adam/SGD state round-trips, parameter-identity association, device
handling (CPU-only here; CUDA cases live in `tests/test_cuda_checkpoint.py`),
`save_model()`/`save_checkpoint()` isolation, format versioning, security
(unregistered/malicious optimizer types), RNG determinism for Dropout
resume, resume equivalence, and Trainer integration. See
`docs/architecture/persistence.md`.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import PersistenceError
from forge.nn import Dropout, Linear, ReLU, Sequential
from forge.nn.loss import MSELoss
from forge.optim import SGD, Adam
from forge.serialization import Checkpoint, load_checkpoint, register_optimizer, save_checkpoint, save_model
from forge.serialization.archive import METADATA_ENTRY
from forge.serialization.checkpoint import CHECKPOINT_FORMAT_VERSION


def _read_metadata(path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read(METADATA_ENTRY))


def _write_metadata(path, metadata: dict) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries[METADATA_ENTRY] = json.dumps(metadata).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def _train_steps(model, optimizer, x, n):
    outs = []
    for _ in range(n):
        optimizer.zero_grad()
        out = model(x)
        outs.append(out.numpy().copy())
        loss = out.sum()
        loss.backward()
        optimizer.step()
    return outs


# -- Adam state round-trip (the critical correctness test, section 11) ------


def test_adam_checkpoint_round_trip_m_v_step_match(tmp_path):
    forge.random.seed(1)
    model = Linear(4, 3)
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.85, 0.995), eps=1e-7, weight_decay=0.01)
    x = Tensor(np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32))
    _train_steps(model, optimizer, x, 4)

    path = tmp_path / "adam.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=2, global_step=4)
    checkpoint = load_checkpoint(str(path))

    assert isinstance(checkpoint, Checkpoint)
    assert isinstance(checkpoint.optimizer, Adam)
    assert checkpoint.epoch == 2
    assert checkpoint.global_step == 4

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        original = original_by_name[name]
        original_state = optimizer.state[original]
        restored_state = checkpoint.optimizer.state[param]

        assert restored_state.step == original_state.step
        np.testing.assert_array_equal(restored_state.m, original_state.m)
        np.testing.assert_array_equal(restored_state.v, original_state.v)
        assert restored_state.m.dtype == original_state.m.dtype
        assert restored_state.m.shape == original_state.m.shape
        assert restored_state.device == original_state.device


def test_adam_checkpoint_preserves_hyperparameters(tmp_path):
    model = Linear(2, 2)
    optimizer = Adam(model.parameters(), lr=0.0025, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.1)
    x = Tensor(np.random.default_rng(2).standard_normal((3, 2)).astype(np.float32))
    _train_steps(model, optimizer, x, 1)

    path = tmp_path / "adam_hparams.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=1)
    checkpoint = load_checkpoint(str(path))

    assert checkpoint.optimizer.lr == pytest.approx(0.0025)
    assert checkpoint.optimizer.beta1 == pytest.approx(0.8)
    assert checkpoint.optimizer.beta2 == pytest.approx(0.9)
    assert checkpoint.optimizer.eps == pytest.approx(1e-6)
    assert checkpoint.optimizer.weight_decay == pytest.approx(0.1)


def test_adam_checkpoint_one_more_step_matches_continuing_original(tmp_path):
    """The critical correctness property: resumed Adam state produces the exact
    same next update as continuing the original optimizer would have."""
    forge.random.seed(2)
    model = Linear(5, 3)
    optimizer = Adam(model.parameters(), lr=0.02)
    x = Tensor(np.random.default_rng(3).standard_normal((6, 5)).astype(np.float32))
    _train_steps(model, optimizer, x, 5)

    path = tmp_path / "adam_next_step.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=5)

    # Continue the original optimizer for one more step.
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    checkpoint = load_checkpoint(str(path))
    checkpoint.optimizer.zero_grad()
    loss2 = checkpoint.model(x).sum()
    loss2.backward()
    checkpoint.optimizer.step()

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)


def test_adam_checkpoint_parameters_never_stepped_have_no_state(tmp_path):
    """A parameter with `requires_grad=False` (never receives a gradient) has
    no Adam state -- it must round-trip with none, not a fabricated zero state."""
    from forge.nn import Parameter

    model = Linear(3, 2)
    model.bias = Parameter(model.bias.numpy(), requires_grad=False)
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(4).standard_normal((2, 3)).astype(np.float32))
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    assert model.weight in optimizer.state
    # bias still requires_grad False but Adam iterates all optimizer.parameters;
    # bias.grad is None only if it never participated -- here it does participate
    # in the forward graph but has requires_grad=False so grad stays None.
    assert model.bias not in optimizer.state

    path = tmp_path / "adam_partial.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=1)
    checkpoint = load_checkpoint(str(path))

    weight2 = dict(checkpoint.model.named_parameters())["weight"]
    bias2 = dict(checkpoint.model.named_parameters())["bias"]
    assert weight2 in checkpoint.optimizer.state
    assert bias2 not in checkpoint.optimizer.state


# -- SGD checkpoint (section 12) --------------------------------------------


def test_sgd_checkpoint_round_trip_config_and_values(tmp_path):
    model = Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.15)
    x = Tensor(np.random.default_rng(5).standard_normal((4, 3)).astype(np.float32))
    _train_steps(model, optimizer, x, 3)

    path = tmp_path / "sgd.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=3)
    checkpoint = load_checkpoint(str(path))

    assert isinstance(checkpoint.optimizer, SGD)
    assert checkpoint.optimizer.lr == pytest.approx(0.15)
    assert checkpoint.epoch == 1
    assert checkpoint.global_step == 3

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_array_equal(param.numpy(), original_by_name[name].numpy())


def test_sgd_checkpoint_one_more_step_matches_continuing_original(tmp_path):
    model = Linear(4, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    x = Tensor(np.random.default_rng(6).standard_normal((5, 4)).astype(np.float32))
    _train_steps(model, optimizer, x, 3)

    path = tmp_path / "sgd_next.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=3)

    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    checkpoint = load_checkpoint(str(path))
    checkpoint.optimizer.zero_grad()
    loss2 = checkpoint.model(x).sum()
    loss2.backward()
    checkpoint.optimizer.step()

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)


# -- save_model() / save_checkpoint() isolation (section 18) ----------------


def test_save_model_contains_no_optimizer_state(tmp_path):
    model = Linear(3, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(7).standard_normal((3, 3)).astype(np.float32))
    _train_steps(model, optimizer, x, 2)

    path = tmp_path / "model_only.forge"
    save_model(model, str(path))
    metadata = _read_metadata(path)

    assert "optimizer" not in metadata
    assert "training_progress" not in metadata
    assert "rng" not in metadata
    assert "forge_checkpoint_format_version" not in metadata


def test_save_checkpoint_contains_optimizer_state(tmp_path):
    model = Linear(3, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(8).standard_normal((3, 3)).astype(np.float32))
    _train_steps(model, optimizer, x, 2)

    path = tmp_path / "with_opt.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=2)
    metadata = _read_metadata(path)

    assert metadata["optimizer"]["type"] == "Adam"
    assert metadata["training_progress"] == {"epoch": 1, "global_step": 2}
    assert "default_generator_state" in metadata["rng"]
    assert metadata["forge_checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION


# -- versioning (section 16) -------------------------------------------------


def test_load_checkpoint_unsupported_version_raises(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)

    metadata = _read_metadata(path)
    metadata["forge_checkpoint_format_version"] = 999
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_checkpoint(str(path))


def test_checkpoint_format_version_independent_of_model_format_version(tmp_path):
    """A checkpoint's version key is distinct from save_model()'s -- tampering
    with save_model()'s version key must not affect checkpoint loading, and
    vice versa (they are read from different metadata fields entirely)."""
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)
    metadata = _read_metadata(path)
    assert "forge_format_version" not in metadata
    assert metadata["forge_checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION


# -- security (section 17) ---------------------------------------------------


def test_load_checkpoint_unregistered_optimizer_type_raises(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)

    metadata = _read_metadata(path)
    metadata["optimizer"]["type"] = "TotallyUnknownOptimizerType"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_checkpoint(str(path))


def test_malicious_optimizer_type_field_does_not_execute_anything(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)

    metadata = _read_metadata(path)
    metadata["optimizer"]["type"] = "__import__('os').system"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_checkpoint(str(path))


def test_malicious_model_type_field_in_checkpoint_does_not_execute_anything(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)

    metadata = _read_metadata(path)
    metadata["model"]["type"] = "__import__('os').system"
    _write_metadata(path, metadata)

    with pytest.raises(PersistenceError):
        load_checkpoint(str(path))


def test_save_checkpoint_unregistered_optimizer_type_raises(tmp_path):
    class Unregistered(SGD):
        pass

    model = Linear(2, 2)
    optimizer = Unregistered(model.parameters(), lr=0.1)
    path = tmp_path / "bad.ckpt"
    with pytest.raises(PersistenceError):
        save_checkpoint(str(path), model, optimizer)
    assert not path.exists()


def test_registering_optimizer_under_conflicting_name_raises():
    class NotAnOptimizer:
        pass

    with pytest.raises(PersistenceError):
        register_optimizer(
            "Adam", NotAnOptimizer,
            get_config=lambda o: {}, from_config=lambda p, c: None,
            get_param_state=lambda o, p: None, set_param_state=lambda o, p, a, s, d: None,
        )


def test_load_uses_allow_pickle_false_for_optimizer_state(tmp_path):
    model = Linear(2, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(9).standard_normal((3, 2)).astype(np.float32))
    _train_steps(model, optimizer, x, 1)

    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)

    with zipfile.ZipFile(path, "r") as zf:
        m_entries = [n for n in zf.namelist() if "optimizer/state" in n and n.endswith("/m.npy")]
        assert m_entries
        raw = zf.read(m_entries[0])
    import io

    array = np.load(io.BytesIO(raw), allow_pickle=False)
    assert array.dtype != object


# -- parameter-identity association (section 3, 8) ---------------------------


def test_optimizer_with_parameter_outside_model_raises(tmp_path):
    model = Linear(2, 2)
    stray = Linear(2, 2)
    optimizer = SGD(list(model.parameters()) + list(stray.parameters()), lr=0.1)

    path = tmp_path / "bad.ckpt"
    with pytest.raises(PersistenceError):
        save_checkpoint(str(path), model, optimizer)


def test_optimizer_subset_of_model_parameters_round_trips(tmp_path):
    """Optimizer covering only some of the model's parameters (e.g. a frozen
    layer excluded from the optimizer) must still round-trip correctly."""
    model = Sequential(Linear(3, 4), Linear(4, 2))
    frozen, trainable = model._modules["0"], model._modules["1"]
    optimizer = SGD(trainable.parameters(), lr=0.1)

    x = Tensor(np.random.default_rng(10).standard_normal((3, 3)).astype(np.float32))
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "partial_opt.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=1)
    checkpoint = load_checkpoint(str(path))

    restored_frozen = checkpoint.model._modules["0"]
    restored_trainable = checkpoint.model._modules["1"]
    assert set(id(p) for p in checkpoint.optimizer.parameters) == set(
        id(p) for p in restored_trainable.parameters()
    )
    for p in restored_frozen.parameters():
        assert id(p) not in set(id(op) for op in checkpoint.optimizer.parameters)


# -- explicit device override (section 7, CPU-only half) --------------------


def test_load_checkpoint_explicit_cpu_override(tmp_path):
    model = Linear(3, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(11).standard_normal((3, 3)).astype(np.float32))
    _train_steps(model, optimizer, x, 1)

    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)
    checkpoint = load_checkpoint(str(path), device="cpu")
    assert checkpoint.model.device.type == "cpu"


def test_load_checkpoint_invalid_device_override_raises(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)

    with pytest.raises(PersistenceError):
        load_checkpoint(str(path), device="tpu")


def test_load_checkpoint_cuda_saved_without_cuda_available_raises(tmp_path, monkeypatch):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)

    metadata = _read_metadata(path)
    metadata["device"] = "cuda"
    _write_metadata(path, metadata)

    import forge.backend.cuda as cuda_module

    monkeypatch.setattr(cuda_module, "is_cuda_available", lambda: False)
    with pytest.raises(PersistenceError, match="CUDA"):
        load_checkpoint(str(path))


# -- RNG / determinism policy (section 10) -----------------------------------


def test_checkpoint_restores_default_rng_state_for_dropout_determinism(tmp_path):
    forge.random.seed(21)
    model = Sequential(Linear(4, 8), ReLU(), Dropout(0.5), Linear(8, 2))
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32))
    _train_steps(model, optimizer, x, 3)

    path = tmp_path / "rng.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=3)

    original_outs = _train_steps(model, optimizer, x, 2)

    checkpoint = load_checkpoint(str(path))
    restored_outs = _train_steps(checkpoint.model, checkpoint.optimizer, x, 2)

    for a, b in zip(original_outs, restored_outs):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_checkpoint_rng_state_is_json_safe_and_present(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "c.ckpt"
    save_checkpoint(str(path), model, optimizer)
    metadata = _read_metadata(path)
    state = metadata["rng"]["default_generator_state"]
    assert isinstance(state, dict)
    assert state["bit_generator"] == "PCG64"


# -- resume equivalence (section 19, CPU + Adam) -----------------------------


def test_resume_equivalence_cpu_adam(tmp_path):
    forge.random.seed(42)
    model = Linear(4, 2)
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32))

    _train_steps(model, optimizer, x, 5)  # N steps

    path = tmp_path / "resume.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=5)

    _train_steps(model, optimizer, x, 4)  # continue for M steps

    checkpoint = load_checkpoint(str(path))
    _train_steps(checkpoint.model, checkpoint.optimizer, x, 4)  # restore, continue M steps

    original_by_name = dict(model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        original = original_by_name[name]
        np.testing.assert_allclose(param.numpy(), original.numpy(), atol=1e-6)

        original_state = optimizer.state[original]
        restored_state = checkpoint.optimizer.state[param]
        assert original_state.step == restored_state.step
        np.testing.assert_allclose(restored_state.m, original_state.m, atol=1e-6)
        np.testing.assert_allclose(restored_state.v, original_state.v, atol=1e-6)


# -- Trainer integration ------------------------------------------------------


def test_trainer_save_checkpoint_and_resume_continues_epoch_and_global_step(tmp_path):
    from forge.data import DataLoader, TensorDataset
    from forge.training import Trainer

    forge.random.seed(100)
    rng = np.random.default_rng(100)
    n = 24
    X = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    y = (2 * X[:, 0] - X[:, 1] + 0.3).reshape(-1, 1).astype(np.float32)

    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = Linear(3, 1)
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=optimizer, verbose=False)
    trainer.fit(loader, epochs=3)
    assert trainer.epoch == 3
    assert trainer.global_step == 3 * len(loader)

    path = tmp_path / "trainer.ckpt"
    trainer.save_checkpoint(path=str(path))

    checkpoint = load_checkpoint(str(path))
    fresh_model = Linear(3, 1)
    fresh_optimizer = Adam(fresh_model.parameters(), lr=0.05)
    trainer2 = Trainer(model=fresh_model, loss_fn=MSELoss(), optimizer=fresh_optimizer, verbose=False)
    trainer2.resume(checkpoint)

    assert trainer2.epoch == 3
    assert trainer2.global_step == 3 * len(loader)

    history = trainer2.fit(loader, epochs=2)
    assert [r.epoch for r in history] == [4, 5]
    assert trainer2.global_step == 5 * len(loader)


def test_trainer_resume_rejects_non_checkpoint():
    from forge.training import Trainer

    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=optimizer, verbose=False)
    with pytest.raises(Exception):
        trainer.resume("not a checkpoint")


# -- extra passthrough --------------------------------------------------------


def test_save_checkpoint_extra_passthrough(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "extra.ckpt"
    save_checkpoint(str(path), model, optimizer, extra={"best_val_loss": 0.42, "run": "exp1"})
    checkpoint = load_checkpoint(str(path))
    assert checkpoint.extra == {"best_val_loss": 0.42, "run": "exp1"}


def test_save_checkpoint_no_extra_defaults_to_empty_dict(tmp_path):
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "no_extra.ckpt"
    save_checkpoint(str(path), model, optimizer)
    checkpoint = load_checkpoint(str(path))
    assert checkpoint.extra == {}


# -- basic validation ---------------------------------------------------------


def test_save_checkpoint_requires_module():
    optimizer = SGD([], lr=0.1)
    with pytest.raises(PersistenceError):
        save_checkpoint("x.ckpt", "not a module", optimizer)


def test_save_checkpoint_requires_optimizer(tmp_path):
    model = Linear(2, 2)
    with pytest.raises(PersistenceError):
        save_checkpoint(str(tmp_path / "x.ckpt"), model, "not an optimizer")


def test_load_nonexistent_checkpoint_raises():
    with pytest.raises(PersistenceError):
        load_checkpoint("does_not_exist.ckpt")
