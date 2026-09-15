"""Milestone 99 tests: `TrainingResult`/`TrainAndSaveResult.artifact_path` --
the first-class training-run result `forge.train()`/`forge.train_and_save()`
now return. See `forge/training/api.py`.

Covers: `TrainingResult` construction/backward-compatibility with
`TrainingHistory`, final-epoch convenience accessors (with and without
validation), model identity, `TrainAndSaveResult.artifact_path`, the new
`train_loss`/`train_metrics` fields, determinism, repeated-access stability,
and public-API import surface.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.data import TensorDataset
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD, Adam
from forge.serialization import register_module
from forge.training import Accuracy, TrainAndSaveResult, Trainer, TrainingHistory, TrainingResult, train, train_and_save


class _MLP(Module):
    def __init__(self, in_features=2, hidden=4, out_features=1):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


register_module("_MLP_M99Test", _MLP, get_config=lambda m: {
    "in_features": m.fc1.in_features, "hidden": m.fc1.out_features, "out_features": m.fc2.out_features,
})


def _regression_dataset(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    return TensorDataset(Tensor(X), Tensor(y))


def _classification_dataset(n=32, seed=0, num_classes=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(n,)).astype(np.int64)
    return TensorDataset(Tensor(X), Tensor(y))


# -- public API surface -------------------------------------------------------


def test_training_result_is_exported_at_top_level():
    assert forge.training.TrainingResult is TrainingResult
    assert forge.TrainingResult is TrainingResult


# -- train() -> TrainingResult -------------------------------------------------


def test_train_returns_a_training_result():
    model = _MLP()
    result = train(
        model, _regression_dataset(),
        loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05), epochs=3, verbose=False,
    )
    assert isinstance(result, TrainingResult)


def test_training_result_is_still_a_training_history():
    """Backward compatibility (Milestone 99 Phase 10): every existing caller
    written against a bare `TrainingHistory` -- len(), iteration, indexing,
    train_losses/val_losses -- must keep working unchanged."""
    model = _MLP()
    result = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=3, verbose=False,
    )
    assert isinstance(result, TrainingHistory)
    assert len(result) == 3
    assert len(list(result)) == 3
    assert result[0].epoch == 1
    assert result[-1].epoch == 3
    assert len(result.train_losses) == 3
    assert len(result.val_losses) == 3


def test_training_result_history_property_returns_self():
    model = _MLP()
    result = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=2, verbose=False,
    )
    assert result.history is result


def test_training_result_exposes_the_trained_model_by_identity():
    model = _MLP()
    result = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=2, verbose=False,
    )
    assert result.model is model


def test_training_result_epochs_completed():
    model = _MLP()
    result = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=5, verbose=False,
    )
    assert result.epochs_completed == 5 == len(result)


def test_training_result_final_metrics_match_last_epoch_record():
    model = _MLP(in_features=2, out_features=3)
    result = train(
        model, _classification_dataset(), loss=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.01), epochs=4,
        validation_dataset=_classification_dataset(seed=1),
        metrics=[Accuracy()], verbose=False,
    )
    last = result[-1]
    assert result.final_train_loss == last.train_loss
    assert result.final_train_metrics == last.train_metrics
    assert result.final_val_loss == last.val_loss
    assert result.final_val_metrics == last.val_metrics
    assert "accuracy" in result.final_train_metrics
    assert "accuracy" in result.final_val_metrics


def test_training_result_final_val_fields_are_none_without_validation():
    model = _MLP()
    result = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=1, verbose=False,
    )
    assert result.final_val_loss is None
    assert result.final_val_metrics == {}


def test_training_result_repeated_access_is_stable():
    model = _MLP()
    result = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=3, verbose=False,
    )
    first = (result.final_train_loss, result.epochs_completed, len(result))
    second = (result.final_train_loss, result.epochs_completed, len(result))
    assert first == second


def test_training_result_repr_does_not_raise():
    model = _MLP()
    result = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=1, verbose=False,
    )
    assert "TrainingResult" in repr(result)


def test_train_deterministic_final_metrics_for_same_seed():
    def _run():
        forge.random.seed(0)
        model = _MLP()
        return train(
            model, _regression_dataset(n=32), loss=MSELoss(),
            optimizer=SGD(model.parameters(), lr=0.1),
            epochs=5, batch_size=8, shuffle=False, verbose=False,
        )

    result_a = _run()
    result_b = _run()
    assert result_a.final_train_loss == result_b.final_train_loss


# -- train_and_save() -> TrainAndSaveResult.artifact_path ---------------------


def test_train_and_save_result_reports_the_artifact_path(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    path = str(tmp_path / "model.forge")

    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=2, path=path, sample=x, verbose=False,
    )
    assert result.artifact_path == path
    assert os.path.exists(result.artifact_path)


def test_train_and_save_result_artifact_is_independently_loadable(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    path = str(tmp_path / "model.forge")

    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=2, path=path, sample=x, verbose=False,
    )
    reloaded = forge.load_model(result.artifact_path)
    with forge.no_grad():
        expected = model(x).numpy()
        actual = reloaded(x).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_train_and_save_result_history_is_a_training_result(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=3, path=str(tmp_path / "model.forge"), sample=x, verbose=False,
    )
    assert isinstance(result.history, TrainingResult)
    assert isinstance(result.history, TrainingHistory)
    # history.model is the pre-save training-time model; result.model is the
    # freshly reloaded, verified model -- deliberately different objects.
    assert result.history.model is model
    assert result.model is not model


def test_train_and_save_result_exposes_final_train_loss_and_metrics(tmp_path):
    model = _MLP(in_features=2, out_features=3)
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _classification_dataset(), loss=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.01), epochs=3,
        metrics=[Accuracy()], verbose=False,
        path=str(tmp_path / "model.forge"), sample=x,
    )
    assert result.train_loss == result.history[-1].train_loss
    assert result.train_metrics == result.history[-1].train_metrics
    assert "accuracy" in result.train_metrics


def test_train_and_save_result_train_loss_present_without_validation(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=1, path=str(tmp_path / "model.forge"), sample=x, verbose=False,
    )
    assert isinstance(result.train_loss, float)
    assert result.val_loss is None
    assert result.val_metrics == {}


def test_train_and_save_result_repeated_access_is_stable(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=2, path=str(tmp_path / "model.forge"), sample=x, verbose=False,
    )
    assert result.artifact_path == result.artifact_path
    assert result.train_loss == result.train_loss
    assert result.history is result.history


def test_train_and_save_result_is_frozen(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=1, path=str(tmp_path / "model.forge"), sample=x, verbose=False,
    )
    with pytest.raises(Exception):
        result.artifact_path = "somewhere-else.forge"


def test_train_and_save_deterministic_final_metrics_for_same_seed(tmp_path):
    def _run(name):
        forge.random.seed(0)
        model = _MLP()
        x = Tensor(np.zeros((1, 2), dtype=np.float32))
        return train_and_save(
            model, _regression_dataset(n=32), loss=MSELoss(),
            optimizer=SGD(model.parameters(), lr=0.1),
            epochs=4, batch_size=8, shuffle=False, verbose=False,
            path=str(tmp_path / f"{name}.forge"), sample=x,
        )

    result_a = _run("a")
    result_b = _run("b")
    assert result_a.train_loss == result_b.train_loss
