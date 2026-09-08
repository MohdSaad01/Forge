"""Milestone 64 integration tests: the `examples/segmentation` pipeline, CPU-only.

Exercises the exact pipeline `examples/segmentation/train.py` runs --
`examples.segmentation.dataset.make_datasets()` -> `DataLoader` -> `Trainer`
-> `examples.segmentation.model.build_model()` -> `MSELoss` -> `Adam` --
plus deterministic dataset generation, the example-local `PixelAccuracy`/
`IoU` metrics, checkpoint save/resume (including resume equivalence against
continuous training), model save/load, and CLI inspection. Mirrors
`tests/test_regression_example_integration.py`'s structure, using this
example's own small, fast, in-process-generated dataset directly (no
external download to stand in for, matching `regression`'s own precedent).
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
from forge.training import Trainer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.segmentation.dataset import (  # noqa: E402
    IMAGE_SIZE,
    NUM_CHANNELS,
    generate_raw,
    make_datasets,
)
from examples.segmentation.metrics import IoU, PixelAccuracy  # noqa: E402
from examples.segmentation.model import build_model  # noqa: E402
from examples.segmentation.train import majority_class_baseline  # noqa: E402


# -- deterministic dataset generation -----------------------------------------


def test_generate_raw_is_reproducible_for_a_fixed_seed():
    images1, masks1 = generate_raw(20, seed=7)
    images2, masks2 = generate_raw(20, seed=7)
    np.testing.assert_array_equal(images1, images2)
    np.testing.assert_array_equal(masks1, masks2)


def test_generate_raw_differs_across_seeds():
    images1, masks1 = generate_raw(20, seed=1)
    images2, masks2 = generate_raw(20, seed=2)
    assert not np.array_equal(images1, images2)
    assert not np.array_equal(masks1, masks2)


def test_generate_raw_matches_documented_shape_dtype_and_value_ranges():
    images, masks = generate_raw(30, seed=0)
    assert images.shape == (30, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    assert images.dtype == np.float32
    assert masks.shape == (30, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert masks.dtype == np.float32
    assert images.min() >= 0.0 and images.max() <= 1.0
    assert set(np.unique(masks).tolist()) <= {0.0, 1.0}


def test_generate_raw_every_mask_has_a_nonempty_nontrivial_foreground_region():
    """Every synthetic image has a real (non-empty, non-full-image) shape -- not a degenerate all-bg/all-fg mask."""
    _, masks = generate_raw(50, seed=3)
    for i in range(masks.shape[0]):
        foreground_fraction = float(masks[i].mean())
        assert 0.0 < foreground_fraction < 0.9


def test_make_datasets_splits_are_disjoint_and_correctly_sized():
    train_ds, test_ds = make_datasets(60, 20, seed=5)
    assert len(train_ds) == 60
    assert len(test_ds) == 20
    # Independent generator streams (seed, seed + 1) -- verify they are not
    # simply repeating the same images.
    train_img0 = train_ds[0][0].numpy()
    test_img0 = test_ds[0][0].numpy()
    assert not np.array_equal(train_img0, test_img0)


# -- model construction --------------------------------------------------------


def test_model_forward_shape_matches_dense_per_pixel_output():
    model = build_model()
    x = Tensor(np.zeros((4, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
    y = model(x)
    assert y.shape == (4, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_model_parameter_count_matches_documented_total():
    model = build_model()
    total = sum(p.numpy().size for p in model.parameters())
    assert total == 5377


# -- PixelAccuracy / IoU metrics -----------------------------------------------


def test_pixel_accuracy_is_exact_on_hand_crafted_predictions():
    metric = PixelAccuracy()
    pred = Tensor(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))  # 1 correct-pattern image
    target = Tensor(np.array([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=np.float32))  # differs at one pixel
    metric.update(pred, target)
    assert metric.compute() == pytest.approx(3 / 4)


def test_pixel_accuracy_thresholds_raw_unbounded_predictions_at_half():
    metric = PixelAccuracy()
    pred = Tensor(np.array([[[[2.0, -1.0], [0.6, 0.4]]]], dtype=np.float32))
    target = Tensor(np.array([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=np.float32))
    metric.update(pred, target)
    # thresholded pred: [1, 0, 1, 0] vs target [1, 0, 1, 0] -- all correct
    assert metric.compute() == pytest.approx(1.0)


def test_pixel_accuracy_accumulates_across_batches_weighted_by_size():
    metric = PixelAccuracy()
    perfect = Tensor(np.ones((2, 1, 2, 2), dtype=np.float32))
    metric.update(perfect, perfect)
    wrong = Tensor(np.zeros((1, 1, 2, 2), dtype=np.float32))
    target = Tensor(np.ones((1, 1, 2, 2), dtype=np.float32))
    metric.update(wrong, target)
    # 2 perfect images (8 correct/8) + 1 fully wrong image (0 correct/4) = 8/12
    assert metric.compute() == pytest.approx(8 / 12)


def test_iou_perfect_overlap_scores_one():
    metric = IoU()
    mask = Tensor(np.array([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=np.float32))
    metric.update(mask, mask)
    assert metric.compute() == pytest.approx(1.0)


def test_iou_no_overlap_scores_zero():
    metric = IoU()
    pred = Tensor(np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32))
    target = Tensor(np.array([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))
    metric.update(pred, target)
    assert metric.compute() == pytest.approx(0.0)


def test_iou_partial_overlap_matches_hand_computed_fraction():
    metric = IoU()
    # pred foreground: {(0,0), (0,1)}; target foreground: {(0,0), (1,0)}
    pred = Tensor(np.array([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=np.float32))
    target = Tensor(np.array([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=np.float32))
    metric.update(pred, target)
    # intersection = 1 ((0,0)), union = 3 ((0,0),(0,1),(1,0))
    assert metric.compute() == pytest.approx(1 / 3)


def test_iou_with_no_foreground_anywhere_scores_one_not_undefined():
    metric = IoU()
    zeros = Tensor(np.zeros((1, 1, 2, 2), dtype=np.float32))
    metric.update(zeros, zeros)
    assert metric.compute() == pytest.approx(1.0)


def test_majority_class_baseline_matches_manual_computation():
    train_ds, test_ds = make_datasets(4, 20, seed=9)
    accuracy, iou = majority_class_baseline(test_ds)

    masks = np.stack([test_ds[i][1].numpy() for i in range(len(test_ds))])
    expected_accuracy = float((masks < 0.5).mean())
    assert accuracy == pytest.approx(expected_accuracy)
    assert iou == pytest.approx(0.0)  # every test image has a nonempty shape (see the dataset test above)


# -- CPU training: loss decreases and beats the trivial baseline --------------


def test_full_pipeline_trains_and_beats_baseline_on_cpu():
    forge.random.seed(0)
    train_ds, test_ds = make_datasets(300, 60, seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(2))
    test_loader = DataLoader(test_ds, batch_size=16)

    baseline_accuracy, baseline_iou = majority_class_baseline(test_ds)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", metrics=[PixelAccuracy(), IoU()], verbose=False)

    initial_params = [p.numpy().copy() for p in model.parameters()]
    trainer.fit(train_loader, epochs=8, validation_loader=test_loader)

    eval_result = trainer.evaluate(test_loader)
    assert eval_result.metrics["pixel_accuracy"] > baseline_accuracy
    assert eval_result.metrics["iou"] > baseline_iou + 0.3  # baseline IoU is 0.0; require real overlap

    for before, after in zip(initial_params, model.parameters()):
        assert not np.allclose(before, after.numpy()), "a parameter did not change during training"


# -- checkpoint save/resume ----------------------------------------------------


def test_checkpoint_save_and_resume_restores_state_and_continues_training(tmp_path):
    forge.random.seed(10)
    train_ds, _ = make_datasets(48, 10, seed=11)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=2)

    checkpoint_path = tmp_path / "segmentation_tiny.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    assert checkpoint.epoch == trainer.epoch == 2
    assert checkpoint.global_step == trainer.global_step

    original_by_name = dict(trainer.model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)
        assert param.device == original_by_name[name].device

    resumed_trainer = Trainer(checkpoint.model, MSELoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(train_loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.global_step == trainer.global_step + len(train_loader)
    assert resumed_trainer.model.device.type == "cpu"


def test_resume_equivalence_matches_continuous_training(tmp_path):
    n_epochs, m_epochs = 2, 2

    def _make_trainer(seed: int) -> "tuple[Trainer, DataLoader]":
        forge.random.seed(seed)
        train_ds, _ = make_datasets(32, 8, seed=seed + 1)
        loader = DataLoader(train_ds, batch_size=8, shuffle=False)
        model = build_model()
        optimizer = Adam(model.parameters(), lr=1e-3)
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
    train_ds, _ = make_datasets(32, 8, seed=21)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, generator=np.random.default_rng(22))

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=2)

    query = Tensor(np.random.default_rng(23).random((4, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32))
    with no_grad():
        pre_save = model(query).numpy()

    model_path = tmp_path / "segmentation_tiny.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path))

    with no_grad():
        post_load = reloaded(query).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)


# -- CLI inspection on generated artifacts -------------------------------------


def test_cli_inspects_generated_segmentation_model_and_checkpoint(tmp_path, capsys):
    forge.random.seed(30)
    train_ds, _ = make_datasets(16, 4, seed=31)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=1)

    model_path = tmp_path / "segmentation_tiny.forge"
    checkpoint_path = tmp_path / "segmentation_tiny.ckpt"
    save_model(model, str(model_path))
    trainer.save_checkpoint(str(checkpoint_path))

    code = cli_main(["model", "inspect", str(model_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Sequential" in out and "Conv2d" in out and "UpsampleNearest2d" in out
    assert "Total parameters: 5377" in out

    code = cli_main(["checkpoint", "inspect", str(checkpoint_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Optimizer: Adam" in out
    assert "Epoch: 1" in out
