"""Milestone 20 integration tests: the `examples/mnist` pipeline, CPU-only.

Exercises the exact pipeline `examples/mnist/train.py` runs against real
MNIST -- `examples.mnist.model.build_model()` -> `DataLoader` -> `Trainer`
-> `CrossEntropyLoss` -> `Adam`, plus checkpoint save/resume (including
resume equivalence against continuous training), model save/load, and CLI
inspection -- against a tiny synthetic `(N, 1, 28, 28)` dataset so this
suite never needs the ~11MB real MNIST download (see
`examples/mnist/dataset.py` for that; this file covers Milestone 20's
"Integration Tests" section instead: a fast, deterministic stand-in with the
same documented shape).

See `tests/test_conv_trainer_integration.py` for the Milestone 15 precedent
this follows (a small labeled-image synthetic dataset trained end-to-end
through `Trainer`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.cli.main import main as cli_main
from forge.data import DataLoader, TensorDataset
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Accuracy, Trainer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.mnist.model import build_model  # noqa: E402  (path setup above)

_NUM_CLASSES = 10


def _make_synthetic_mnist(n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """A tiny `(N, 1, 28, 28)` float32 / `(N,)` int64 stand-in for real MNIST.

    Each of the 10 classes is a distinct bright 2-pixel-wide vertical stripe
    (class `L` at columns `[2L, 2L+2)`) over low-amplitude noise -- easily
    separable by the same `Conv2d`/`MaxPool2d` architecture `build_model()`
    uses for real digits, without requiring any actual image data.
    """
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, 1, 28, 28)) * 0.05).astype(np.float32)
    Y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        label = i % _NUM_CLASSES
        Y[i] = label
        col = label * 2
        stripe = 1.0 + rng.standard_normal((28, 2)).astype(np.float32) * 0.05
        X[i, 0, :, col : col + 2] = stripe
    return X, Y


def _make_dataset(n: int, seed: int) -> TensorDataset:
    X, Y = _make_synthetic_mnist(n, seed)
    return TensorDataset(Tensor(X), Tensor(Y))


# -- shape / architecture -----------------------------------------------------


def test_synthetic_dataset_matches_documented_nchw_shape():
    X, Y = _make_synthetic_mnist(6, seed=0)
    assert X.shape == (6, 1, 28, 28)
    assert X.dtype == np.float32
    assert Y.shape == (6,)
    assert Y.dtype == np.int64


def test_model_forward_shape_matches_ten_class_output():
    model = build_model()
    x = Tensor(np.zeros((5, 1, 28, 28), dtype=np.float32))
    y = model(x)
    assert y.shape == (5, _NUM_CLASSES)


# -- CPU training: loss decreases, accuracy exceeds chance, params change ----


def test_full_pipeline_trains_and_learns_on_cpu():
    forge.random.seed(0)
    train_ds = _make_dataset(80, seed=1)
    test_ds = _make_dataset(40, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(3))
    test_loader = DataLoader(test_ds, batch_size=16)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", metrics=[Accuracy()], verbose=False)

    initial_params = [p.numpy().copy() for p in model.parameters()]
    history = trainer.fit(train_loader, epochs=8, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    # 10-class chance accuracy is 0.10; this checks substantially above chance.
    assert eval_result.metrics["accuracy"] >= 0.8

    for before, after in zip(initial_params, model.parameters()):
        assert not np.allclose(before, after.numpy()), "a parameter did not change during training"


# -- checkpoint save/resume ----------------------------------------------------


def test_checkpoint_save_and_resume_restores_state_and_continues_training(tmp_path):
    forge.random.seed(10)
    train_loader = DataLoader(_make_dataset(64, seed=11), batch_size=16, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=2)

    checkpoint_path = tmp_path / "mnist_tiny.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    assert checkpoint.epoch == trainer.epoch == 2
    assert checkpoint.global_step == trainer.global_step

    # Model state restores.
    original_by_name = dict(trainer.model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)
        assert param.device == original_by_name[name].device

    # Adam state restores.
    for name, param in checkpoint.model.named_parameters():
        original_param = original_by_name[name]
        original_state = trainer.optimizer.state[original_param]
        restored_state = checkpoint.optimizer.state[param]
        assert restored_state.step == original_state.step
        np.testing.assert_allclose(restored_state.m, original_state.m, atol=1e-6)
        np.testing.assert_allclose(restored_state.v, original_state.v, atol=1e-6)

    # Training continues: epoch/global_step advance further, on the same device.
    resumed_trainer = Trainer(checkpoint.model, CrossEntropyLoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(train_loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.global_step == trainer.global_step + len(train_loader)
    assert resumed_trainer.model.device.type == "cpu"


def test_resume_equivalence_matches_continuous_training(tmp_path):
    """`N` epochs -> checkpoint -> reload -> `M` epochs == continuous `N+M` epochs.

    Uses `shuffle=False` so both runs see an identical batch sequence with no
    caller-owned `DataLoader` generator to restore -- isolating this
    comparison to exactly what a checkpoint itself covers (model/optimizer
    state, RNG), per `forge.serialization.checkpoint`'s documented RNG
    policy. This architecture has no `Dropout`, so no other source of
    training-time randomness exists to diverge either way.
    """
    n_epochs, m_epochs = 2, 2

    def _make_trainer(seed: int) -> "tuple[Trainer, DataLoader]":
        forge.random.seed(seed)
        loader = DataLoader(_make_dataset(48, seed=seed + 1), batch_size=8, shuffle=False)
        model = build_model()
        optimizer = Adam(model.parameters(), lr=5e-3)
        return Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False), loader

    # Continuous: N + M epochs in one run.
    continuous_trainer, continuous_loader = _make_trainer(seed=42)
    continuous_trainer.fit(continuous_loader, epochs=n_epochs + m_epochs)

    # Checkpointed: N epochs -> save -> reload -> M more epochs.
    checkpointed_trainer, checkpointed_loader = _make_trainer(seed=42)
    checkpointed_trainer.fit(checkpointed_loader, epochs=n_epochs)
    checkpoint_path = tmp_path / "resume_equivalence.ckpt"
    checkpointed_trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    resumed_trainer = Trainer(checkpoint.model, CrossEntropyLoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(checkpointed_loader, epochs=m_epochs)

    continuous_by_name = dict(continuous_trainer.model.named_parameters())
    for name, param in resumed_trainer.model.named_parameters():
        np.testing.assert_allclose(
            param.numpy(), continuous_by_name[name].numpy(), atol=1e-5,
            err_msg=f"resumed vs. continuous training diverged for parameter '{name}'",
        )


# -- model persistence ---------------------------------------------------------


def test_model_persistence_preserves_predictions(tmp_path):
    forge.random.seed(20)
    train_loader = DataLoader(_make_dataset(48, seed=21), batch_size=16, shuffle=True, generator=np.random.default_rng(22))

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=3)

    query = Tensor(np.random.default_rng(23).standard_normal((4, 1, 28, 28)).astype(np.float32))
    with no_grad():
        pre_save = model(query).numpy()

    model_path = tmp_path / "mnist_tiny.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path))

    with no_grad():
        post_load = reloaded(query).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)


# -- CLI inspection on generated artifacts -------------------------------------


def test_cli_inspects_generated_mnist_model_and_checkpoint(tmp_path, capsys):
    forge.random.seed(30)
    train_loader = DataLoader(_make_dataset(32, seed=31), batch_size=8, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=1)

    model_path = tmp_path / "mnist_tiny.forge"
    checkpoint_path = tmp_path / "mnist_tiny.ckpt"
    save_model(model, str(model_path))
    trainer.save_checkpoint(str(checkpoint_path))

    code = cli_main(["model", "inspect", str(model_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Conv2d" in out and "MaxPool2d" in out and "Sequential" in out
    assert "Total parameters: 27562" in out

    code = cli_main(["checkpoint", "inspect", str(checkpoint_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Optimizer: Adam" in out
    assert "Epoch: 1" in out
