"""Milestone 60 integration tests: the `examples/regression` pipeline, CPU-only.

Exercises the exact pipeline `examples/regression/train.py` runs --
`examples.regression.dataset.make_datasets()` -> `DataLoader` -> `Trainer`
-> `examples.regression.model.build_model()` -> `MSELoss` -> `Adam` -- plus
deterministic dataset generation, checkpoint save/resume (including resume
equivalence against continuous training), model save/load, and CLI
inspection. Mirrors `tests/test_mnist_example_integration.py`'s structure,
using this example's own small dataset directly (it is already fast and
synthetic, so no separate "tiny stand-in" dataset is needed the way MNIST's
real download is stood in for).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.cli.main import main as cli_main
from forge.data import DataLoader
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import MeanAbsoluteError, Trainer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.regression.dataset import N_FEATURES, generate_raw, make_datasets  # noqa: E402
from examples.regression.model import build_model  # noqa: E402


# -- deterministic dataset generation -----------------------------------------


def test_generate_raw_is_reproducible_for_a_fixed_seed():
    X1, y1 = generate_raw(50, seed=7)
    X2, y2 = generate_raw(50, seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_raw_differs_across_seeds():
    X1, y1 = generate_raw(50, seed=1)
    X2, y2 = generate_raw(50, seed=2)
    assert not np.array_equal(X1, X2)
    assert not np.array_equal(y1, y2)


def test_generate_raw_matches_documented_shape_and_dtype():
    X, y = generate_raw(20, seed=0)
    assert X.shape == (20, N_FEATURES)
    assert X.dtype == np.float32
    assert y.shape == (20, 1)
    assert y.dtype == np.float32


def test_make_datasets_splits_are_disjoint_and_correctly_sized():
    train_ds, val_ds, test_ds, stats = make_datasets(60, 20, 20, seed=3)
    assert len(train_ds) == 60
    assert len(val_ds) == 20
    assert len(test_ds) == 20
    assert "mean" in stats and "std" in stats
    assert stats["mean"].shape == (N_FEATURES,)
    assert stats["std"].shape == (N_FEATURES,)
    assert stats["y_train_var"] > 0.0


def test_make_datasets_normalizes_training_features_to_near_zero_mean_unit_std():
    train_ds, _, _, _ = make_datasets(500, 50, 50, seed=4)
    X = np.stack([train_ds[i][0].numpy() for i in range(len(train_ds))])
    np.testing.assert_allclose(X.mean(axis=0), np.zeros(N_FEATURES), atol=0.15)
    np.testing.assert_allclose(X.std(axis=0), np.ones(N_FEATURES), atol=0.15)


# -- model construction --------------------------------------------------------


def test_model_forward_shape_matches_single_regression_output():
    model = build_model()
    x = Tensor(np.zeros((5, N_FEATURES), dtype=np.float32))
    y = model(x)
    assert y.shape == (5, 1)


# -- CPU training: loss decreases and beats the trivial baseline --------------


def test_full_pipeline_trains_and_beats_baseline_on_cpu():
    forge.random.seed(0)
    train_ds, val_ds, _, stats = make_datasets(300, 60, 60, seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(2))
    val_loader = DataLoader(val_ds, batch_size=16)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", metrics=[MeanAbsoluteError()], verbose=False)

    initial_params = [p.numpy().copy() for p in model.parameters()]
    history = trainer.fit(train_loader, epochs=25, validation_loader=val_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.2
    eval_result = trainer.evaluate(val_loader)
    # A trivial predict-the-training-mean baseline has MSE == stats["y_train_var"].
    assert eval_result.loss < stats["y_train_var"] * 0.3

    for before, after in zip(initial_params, model.parameters()):
        assert not np.allclose(before, after.numpy()), "a parameter did not change during training"


# -- checkpoint save/resume ----------------------------------------------------


def test_checkpoint_save_and_resume_restores_state_and_continues_training(tmp_path):
    forge.random.seed(10)
    train_ds, _, _, _ = make_datasets(80, 10, 10, seed=11)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=2)

    checkpoint_path = tmp_path / "regression_tiny.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    assert checkpoint.epoch == trainer.epoch == 2
    assert checkpoint.global_step == trainer.global_step

    original_by_name = dict(trainer.model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)
        assert param.device == original_by_name[name].device

    for name, param in checkpoint.model.named_parameters():
        original_param = original_by_name[name]
        original_state = trainer.optimizer.state[original_param]
        restored_state = checkpoint.optimizer.state[param]
        assert restored_state.step == original_state.step
        np.testing.assert_allclose(restored_state.m, original_state.m, atol=1e-6)
        np.testing.assert_allclose(restored_state.v, original_state.v, atol=1e-6)

    resumed_trainer = Trainer(checkpoint.model, MSELoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(train_loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.global_step == trainer.global_step + len(train_loader)
    assert resumed_trainer.model.device.type == "cpu"


def test_resume_equivalence_matches_continuous_training(tmp_path):
    """`N` epochs -> checkpoint -> reload -> `M` epochs == continuous `N+M` epochs.

    Uses `shuffle=False` so both runs see an identical batch sequence with no
    caller-owned `DataLoader` generator to restore, matching
    `test_mnist_example_integration.py`'s own resume-equivalence test.
    """
    n_epochs, m_epochs = 2, 2

    def _make_trainer(seed: int) -> "tuple[Trainer, DataLoader]":
        forge.random.seed(seed)
        train_ds, _, _, _ = make_datasets(64, 10, 10, seed=seed + 1)
        loader = DataLoader(train_ds, batch_size=8, shuffle=False)
        model = build_model()
        optimizer = Adam(model.parameters(), lr=5e-3)
        return Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False), loader

    continuous_trainer, continuous_loader = _make_trainer(seed=42)
    continuous_trainer.fit(continuous_loader, epochs=n_epochs + m_epochs)

    checkpointed_trainer, checkpointed_loader = _make_trainer(seed=42)
    checkpointed_trainer.fit(checkpointed_loader, epochs=n_epochs)
    checkpoint_path = tmp_path / "resume_equivalence.ckpt"
    checkpointed_trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    resumed_trainer = Trainer(checkpoint.model, MSELoss(), checkpoint.optimizer, device="cpu", verbose=False)
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
    train_ds, _, _, _ = make_datasets(48, 10, 10, seed=21)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(22))

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=3)

    query = Tensor(np.random.default_rng(23).standard_normal((4, N_FEATURES)).astype(np.float32))
    with no_grad():
        pre_save = model(query).numpy()

    model_path = tmp_path / "regression_tiny.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path))

    with no_grad():
        post_load = reloaded(query).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)


# -- CLI inspection on generated artifacts -------------------------------------


def test_cli_inspects_generated_regression_model_and_checkpoint(tmp_path, capsys):
    forge.random.seed(30)
    train_ds, _, _, _ = make_datasets(32, 8, 8, seed=31)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=1)

    model_path = tmp_path / "regression_tiny.forge"
    checkpoint_path = tmp_path / "regression_tiny.ckpt"
    save_model(model, str(model_path))
    trainer.save_checkpoint(str(checkpoint_path))

    code = cli_main(["model", "inspect", str(model_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Sequential" in out and "Linear" in out
    assert "Total parameters: 2689" in out

    code = cli_main(["checkpoint", "inspect", str(checkpoint_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Optimizer: Adam" in out
    assert "Epoch: 1" in out
