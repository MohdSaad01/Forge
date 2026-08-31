"""Tests for `forge.training.metrics`: MSE, MAE, Accuracy and their aggregation.

Metrics must be mathematically correct across unequal batch sizes -- an
epoch-level score must be computed from all samples together, not from an
average of per-batch scores. See `docs/architecture/training-engine.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import TrainerError
from forge.training import Accuracy, MeanAbsoluteError, MeanSquaredError


# -- MeanSquaredError --------------------------------------------------------


def test_mse_single_batch():
    metric = MeanSquaredError()
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[1.0, 0.0], [3.0, 0.0]])
    metric.update(pred, target)
    # errors: 0, 2, 0, 4 -> squared: 0, 4, 0, 16 -> mean = 5.0
    assert metric.compute() == pytest.approx(5.0)


def test_mse_weighted_aggregation_across_unequal_batches():
    metric = MeanSquaredError()
    # batch 1: 32 samples of error 1.0 each (squared error 1.0)
    metric.update(Tensor(np.ones((32, 1))), Tensor(np.zeros((32, 1))))
    # batch 2: 32 samples of error 1.0 each
    metric.update(Tensor(np.ones((32, 1))), Tensor(np.zeros((32, 1))))
    # batch 3: 6 samples of error 3.0 each (squared error 9.0)
    metric.update(Tensor(np.full((6, 1), 3.0)), Tensor(np.zeros((6, 1))))

    total_elements = 32 + 32 + 6
    expected = (32 * 1.0 + 32 * 1.0 + 6 * 9.0) / total_elements
    assert metric.compute() == pytest.approx(expected)
    # sanity: naive mean-of-batch-means would be wrong here
    naive = (1.0 + 1.0 + 9.0) / 3
    assert metric.compute() != pytest.approx(naive)


def test_mse_reset_clears_state():
    metric = MeanSquaredError()
    metric.update(Tensor([1.0]), Tensor([0.0]))
    metric.reset()
    with pytest.raises(TrainerError):
        metric.compute()


def test_mse_shape_mismatch_raises():
    metric = MeanSquaredError()
    with pytest.raises(TrainerError):
        metric.update(Tensor([[1.0, 2.0]]), Tensor([1.0, 2.0]))


def test_mse_compute_before_update_raises():
    metric = MeanSquaredError()
    with pytest.raises(TrainerError):
        metric.compute()


# -- MeanAbsoluteError --------------------------------------------------------


def test_mae_single_batch():
    metric = MeanAbsoluteError()
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[1.0, 0.0], [3.0, 0.0]])
    metric.update(pred, target)
    # abs errors: 0, 2, 0, 4 -> mean = 1.5
    assert metric.compute() == pytest.approx(1.5)


def test_mae_weighted_aggregation_across_unequal_batches():
    metric = MeanAbsoluteError()
    metric.update(Tensor(np.ones((10, 1))), Tensor(np.zeros((10, 1))))  # abs err 1.0 x10
    metric.update(Tensor(np.full((2, 1), 5.0)), Tensor(np.zeros((2, 1))))  # abs err 5.0 x2
    expected = (10 * 1.0 + 2 * 5.0) / 12
    assert metric.compute() == pytest.approx(expected)


def test_mae_shape_mismatch_raises():
    metric = MeanAbsoluteError()
    with pytest.raises(TrainerError):
        metric.update(Tensor([[1.0]]), Tensor([1.0, 2.0]))


# -- Accuracy ------------------------------------------------------------


def test_accuracy_single_batch():
    metric = Accuracy()
    logits = Tensor([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0]])  # predicted: 0, 1, 0
    target = Tensor(np.array([0, 1, 1]))
    metric.update(logits, target)
    assert metric.compute() == pytest.approx(2 / 3)


def test_accuracy_weighted_aggregation_across_unequal_batches():
    metric = Accuracy()
    # batch 1: 32 samples, all correct
    logits1 = Tensor(np.tile([[10.0, 0.0]], (32, 1)))
    target1 = Tensor(np.zeros(32, dtype=np.int64))
    metric.update(logits1, target1)
    # batch 2: 6 samples, all wrong
    logits2 = Tensor(np.tile([[10.0, 0.0]], (6, 1)))
    target2 = Tensor(np.ones(6, dtype=np.int64))
    metric.update(logits2, target2)

    expected = 32 / (32 + 6)
    assert metric.compute() == pytest.approx(expected)


def test_accuracy_invalid_prediction_ndim_raises():
    metric = Accuracy()
    with pytest.raises(TrainerError):
        metric.update(Tensor([1.0, 2.0]), Tensor(np.array([0])))


def test_accuracy_invalid_target_shape_raises():
    metric = Accuracy()
    with pytest.raises(TrainerError):
        metric.update(Tensor([[1.0, 2.0]]), Tensor(np.array([[0]])))


def test_metrics_do_not_modify_model_parameters():
    forge.random.seed(0)
    from forge.nn import Linear

    model = Linear(2, 2)
    before = model.weight.numpy().copy()

    metric = Accuracy()
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    prediction = model(x)
    target = Tensor(np.array([0, 1]))
    metric.update(prediction, target)
    metric.compute()

    np.testing.assert_array_equal(model.weight.numpy(), before)
