"""Milestone 62 integration tests: the `examples/waveform_classification` pipeline, CPU-only.

Exercises the exact pipeline `examples/waveform_classification/train.py` runs
-- `examples.waveform_classification.dataset.make_datasets()` -> `DataLoader`
-> `Trainer` -> `examples.waveform_classification.model.build_model()`
(`Conv1d`/`ReLU`/`MaxPool1d`/`Flatten`/`Linear`) -> `CrossEntropyLoss` ->
`Adam` -- plus deterministic dataset generation, checkpoint save/resume,
model save/load, and CLI inspection. Mirrors
`tests/test_regression_example_integration.py`'s structure; this example's
dataset is synthetic and fast to generate, so no separate "tiny stand-in"
dataset is needed.
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
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Accuracy, Trainer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.waveform_classification.dataset import (  # noqa: E402
    LENGTH,
    NUM_CLASSES,
    generate_raw,
    make_datasets,
)
from examples.waveform_classification.model import build_model  # noqa: E402


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


def test_generate_raw_matches_documented_shape_dtype_and_label_range():
    X, y = generate_raw(20, seed=0)
    assert X.shape == (20, 1, LENGTH)
    assert X.dtype == np.float32
    assert y.shape == (20,)
    assert y.dtype == np.int64
    assert y.min() >= 0 and y.max() < NUM_CLASSES


def test_make_datasets_splits_are_disjoint_and_correctly_sized():
    train_ds, val_ds, test_ds, stats = make_datasets(60, 20, 20, seed=3)
    assert len(train_ds) == 60
    assert len(val_ds) == 20
    assert len(test_ds) == 20
    assert stats["num_classes"] == NUM_CLASSES
    assert stats["majority_baseline_accuracy"] == pytest.approx(1.0 / NUM_CLASSES)


def test_make_datasets_classes_are_roughly_balanced():
    """Labels are drawn i.i.d. uniform -- not exactly balanced, but not skewed either."""
    _, _, _, _ = make_datasets(1, 1, 1, seed=0)  # smoke: doesn't raise for tiny splits
    X, y = generate_raw(2000, seed=4)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    assert counts.min() > 2000 / NUM_CLASSES * 0.7  # loose balance check, not exact


# -- model construction --------------------------------------------------------


def test_model_forward_shape_matches_num_classes():
    model = build_model()
    x = Tensor(np.zeros((5, 1, LENGTH), dtype=np.float32))
    y = model(x)
    assert y.shape == (5, NUM_CLASSES)


# -- CPU training: loss decreases and beats the trivial baseline --------------


def test_full_pipeline_trains_and_beats_baseline_on_cpu():
    forge.random.seed(0)
    train_ds, val_ds, _, stats = make_datasets(1200, 200, 200, seed=1)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=np.random.default_rng(2))
    val_loader = DataLoader(val_ds, batch_size=32)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", metrics=[Accuracy()], verbose=False)

    initial_params = [p.numpy().copy() for p in model.parameters()]
    history = trainer.fit(train_loader, epochs=8, validation_loader=val_loader)

    assert history[-1].train_loss < history[0].train_loss
    eval_result = trainer.evaluate(val_loader)
    assert eval_result.metrics["accuracy"] > stats["majority_baseline_accuracy"] * 2

    for before, after in zip(initial_params, model.parameters()):
        assert not np.allclose(before, after.numpy()), "a parameter did not change during training"


# -- checkpoint save/resume ----------------------------------------------------


def test_checkpoint_save_and_resume_restores_state_and_continues_training(tmp_path):
    forge.random.seed(10)
    train_ds, _, _, _ = make_datasets(80, 10, 10, seed=11)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=2)

    checkpoint_path = tmp_path / "waveform_tiny.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    assert checkpoint.epoch == trainer.epoch == 2
    assert checkpoint.global_step == trainer.global_step

    original_by_name = dict(trainer.model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)
        assert param.device == original_by_name[name].device

    resumed_trainer = Trainer(checkpoint.model, CrossEntropyLoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(train_loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.global_step == trainer.global_step + len(train_loader)
    assert resumed_trainer.model.device.type == "cpu"


def test_resume_equivalence_matches_continuous_training(tmp_path):
    n_epochs, m_epochs = 2, 2

    def _make_trainer(seed: int) -> "tuple[Trainer, DataLoader]":
        forge.random.seed(seed)
        train_ds, _, _, _ = make_datasets(64, 10, 10, seed=seed + 1)
        loader = DataLoader(train_ds, batch_size=8, shuffle=False)
        model = build_model()
        optimizer = Adam(model.parameters(), lr=1e-3)
        return Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False), loader

    continuous_trainer, continuous_loader = _make_trainer(seed=42)
    continuous_trainer.fit(continuous_loader, epochs=n_epochs + m_epochs)

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
    train_ds, _, _, _ = make_datasets(48, 10, 10, seed=21)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(22))

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=3)

    query = Tensor(np.random.default_rng(23).standard_normal((4, 1, LENGTH)).astype(np.float32))
    with no_grad():
        pre_save = model(query).numpy()

    model_path = tmp_path / "waveform_tiny.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path))

    with no_grad():
        post_load = reloaded(query).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)


# -- CLI inspection on generated artifacts -------------------------------------


def test_cli_inspects_generated_waveform_model_and_checkpoint(tmp_path, capsys):
    forge.random.seed(30)
    train_ds, _, _, _ = make_datasets(32, 8, 8, seed=31)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=1)

    model_path = tmp_path / "waveform_tiny.forge"
    checkpoint_path = tmp_path / "waveform_tiny.ckpt"
    save_model(model, str(model_path))
    trainer.save_checkpoint(str(checkpoint_path))

    code = cli_main(["model", "inspect", str(model_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Sequential" in out and "Conv1d" in out

    code = cli_main(["checkpoint", "inspect", str(checkpoint_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Optimizer: Adam" in out
    assert "Epoch: 1" in out
