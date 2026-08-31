"""Milestone 6 tests: Trainer orchestration of DataLoader/Module/Loss/Optimizer.

Covers construction validation, the training lifecycle, DataLoader
integration, evaluation semantics, validation-loader support, training
history, progress-output suppressibility, and end-to-end deterministic
regression/classification experiments. See
`docs/architecture/training-engine.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.exceptions import TrainerError, UnsupportedDeviceError
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD
from forge.training import Accuracy, MeanAbsoluteError, MeanSquaredError, Trainer


class MLP(Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _regression_loader(n=32, batch_size=8, shuffle=False, seed=42):
    forge.random.seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _classification_loader(n=20, batch_size=8, seed=7):
    forge.random.seed(seed)
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=[-1.5, -1.5], scale=0.3, size=(n, 2))
    class1 = rng.normal(loc=[1.5, 1.5], scale=0.3, size=(n, 2))
    X = np.vstack([class0, class1])
    targets = np.array([0] * n + [1] * n)
    dataset = TensorDataset(Tensor(X), Tensor(targets))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _trainer_components():
    forge.random.seed(0)
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    return model, loss_fn, optimizer


# -- Trainer construction -----------------------------------------------------


def test_valid_construction():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device="cpu")
    assert trainer.model is model
    assert str(trainer.device) == "cpu"


def test_construction_rejects_non_module_model():
    _, loss_fn, optimizer = _trainer_components()
    with pytest.raises(TrainerError):
        Trainer(model=object(), loss_fn=loss_fn, optimizer=optimizer)


def test_construction_rejects_non_loss():
    model, _, optimizer = _trainer_components()
    with pytest.raises(TrainerError):
        Trainer(model=model, loss_fn=lambda p, t: p, optimizer=optimizer)


def test_construction_rejects_non_optimizer():
    model, loss_fn, _ = _trainer_components()
    with pytest.raises(TrainerError):
        Trainer(model=model, loss_fn=loss_fn, optimizer=object())


def test_construction_rejects_cuda_device():
    model, loss_fn, optimizer = _trainer_components()
    with pytest.raises(UnsupportedDeviceError):
        Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device="cuda")


def test_construction_rejects_unknown_device_string():
    model, loss_fn, optimizer = _trainer_components()
    with pytest.raises(UnsupportedDeviceError):
        Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device="tpu")


def test_construction_rejects_duplicate_metric_names():
    model, loss_fn, optimizer = _trainer_components()
    with pytest.raises(TrainerError):
        Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=[MeanSquaredError(), MeanSquaredError()],
        )


def test_construction_rejects_non_metric_instance():
    model, loss_fn, optimizer = _trainer_components()
    with pytest.raises(TrainerError):
        Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, metrics=["not a metric"])


def test_fit_rejects_zero_epochs():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()
    with pytest.raises(TrainerError):
        trainer.fit(loader, epochs=0)


def test_fit_rejects_negative_epochs():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()
    with pytest.raises(TrainerError):
        trainer.fit(loader, epochs=-3)


def test_fit_rejects_non_int_epochs():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()
    with pytest.raises(TrainerError):
        trainer.fit(loader, epochs=2.5)


def test_fit_rejects_empty_loader():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    empty_dataset = TensorDataset(Tensor(np.zeros((1, 2))), Tensor(np.zeros((1, 1))))
    empty_loader = DataLoader(empty_dataset, batch_size=4, drop_last=True)
    assert len(empty_loader) == 0
    with pytest.raises(TrainerError):
        trainer.fit(empty_loader, epochs=1)


# -- Training -----------------------------------------------------------


def test_one_epoch_training_runs_and_updates_parameters():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    before = model.weight.numpy().copy()

    loader = _regression_loader()
    history = trainer.fit(loader, epochs=1)

    assert len(history) == 1
    assert not np.allclose(model.weight.numpy(), before)


def test_multiple_epochs_all_recorded():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()
    history = trainer.fit(loader, epochs=5)

    assert len(history) == 5
    assert [r.epoch for r in history] == [1, 2, 3, 4, 5]


def test_optimizer_step_actually_changes_parameters_each_epoch():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()

    snapshots = [model.weight.numpy().copy()]
    for _ in range(3):
        trainer.fit(loader, epochs=1)
        snapshots.append(model.weight.numpy().copy())

    for before, after in zip(snapshots, snapshots[1:]):
        assert not np.allclose(before, after)


def test_loss_decreases_on_deterministic_synthetic_data():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader(n=32, batch_size=32)

    history = trainer.fit(loader, epochs=100)
    assert history.train_losses[-1] < history.train_losses[0] * 0.1


def test_model_remains_usable_after_training():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()
    trainer.fit(loader, epochs=2)

    x = Tensor([[0.1, 0.2]])
    prediction = model(x)
    assert prediction.shape == (1, 1)
    assert np.isfinite(prediction.numpy()).all()


# -- DataLoader integration -----------------------------------------------


def test_multiple_batches_are_all_consumed():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader(n=32, batch_size=8)
    assert len(loader) == 4

    history = trainer.fit(loader, epochs=1)
    assert history[0].samples == 32


def test_final_partial_batch_is_included_by_default():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader(n=10, batch_size=4)  # batches of 4, 4, 2
    assert len(loader) == 3

    history = trainer.fit(loader, epochs=1)
    assert history[0].samples == 10


def test_shuffled_dataloader_trains_successfully():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader(n=32, batch_size=8, shuffle=True)

    history = trainer.fit(loader, epochs=3)
    assert len(history) == 3
    assert all(np.isfinite(r.train_loss) for r in history)


def test_rejects_non_tuple_batch():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)

    class SingleTensorDataset:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return Tensor([1.0, 2.0])

    loader = DataLoader(SingleTensorDataset(), batch_size=2)
    with pytest.raises(TrainerError):
        trainer.fit(loader, epochs=1)


# -- Evaluation -----------------------------------------------------------


def test_evaluate_does_not_update_parameters():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()
    before = model.weight.numpy().copy()

    trainer.evaluate(loader)
    np.testing.assert_array_equal(model.weight.numpy(), before)


def test_evaluate_does_not_call_optimizer_step(monkeypatch):
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()

    calls = []
    monkeypatch.setattr(optimizer, "step", lambda: calls.append("step"))
    trainer.evaluate(loader)
    assert calls == []


def test_evaluate_switches_to_eval_mode_during_forward_pass():
    modes_seen = []

    class ModeSpy(Module):
        def __init__(self):
            super().__init__()
            self.fc = Linear(2, 1)

        def forward(self, x):
            modes_seen.append(self.training)
            return self.fc(x)

    forge.random.seed(0)
    model = ModeSpy()
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), verbose=False)
    loader = _regression_loader()

    trainer.evaluate(loader)
    assert all(mode is False for mode in modes_seen)


def test_evaluate_restores_prior_training_mode():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()

    model.train()
    trainer.evaluate(loader)
    assert model.training is True

    model.eval()
    trainer.evaluate(loader)
    assert model.training is False


def test_evaluate_propagates_eval_mode_to_nested_modules():
    forge.random.seed(0)
    model = MLP(2, 4, 2)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _classification_loader()

    seen_modes = {}

    original_forward = model.fc1.forward

    def spy_forward(x):
        seen_modes["fc1"] = model.fc1.training
        seen_modes["fc2"] = model.fc2.training
        return original_forward(x)

    model.fc1.forward = spy_forward
    trainer.evaluate(loader)
    assert seen_modes == {"fc1": False, "fc2": False}


def test_evaluate_metrics_aggregate_correctly_across_unequal_batches():
    forge.random.seed(0)
    model = Linear(1, 1)
    # Force a known linear function: y = x (weight=1, bias=0).
    model.weight._data[...] = 1.0
    model.bias._data[...] = 0.0

    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=[MeanAbsoluteError()],
        verbose=False,
    )

    X = np.concatenate([np.zeros((32, 1)), np.full((6, 1), 5.0)])
    y = np.concatenate([np.ones((32, 1)), np.zeros((6, 1))])  # errors: 32x1.0, 6x5.0
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=32)  # batches of 32, 6

    result = trainer.evaluate(loader)
    expected_mae = (32 * 1.0 + 6 * 5.0) / 38
    assert result.metrics["mae"] == pytest.approx(expected_mae)
    assert result.samples == 38


def test_evaluate_rejects_empty_loader():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    empty_dataset = TensorDataset(Tensor(np.zeros((1, 2))), Tensor(np.zeros((1, 1))))
    empty_loader = DataLoader(empty_dataset, batch_size=4, drop_last=True)
    with pytest.raises(TrainerError):
        trainer.evaluate(empty_loader)


# -- Validation ----------------------------------------------------------


def test_fit_with_validation_loader_records_val_loss_and_metrics():
    forge.random.seed(0)
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=[MeanAbsoluteError()],
        verbose=False,
    )

    train_loader = _regression_loader(n=32, batch_size=8, seed=1)
    val_loader = _regression_loader(n=16, batch_size=8, seed=2)

    history = trainer.fit(train_loader, epochs=3, validation_loader=val_loader)

    for record in history:
        assert record.val_loss is not None
        assert "mae" in record.val_metrics
        assert "mae" in record.train_metrics


def test_fit_without_validation_loader_leaves_val_fields_empty():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()

    history = trainer.fit(loader, epochs=2)
    for record in history:
        assert record.val_loss is None
        assert record.val_metrics == {}


def test_train_and_validation_metrics_are_distinguishable():
    forge.random.seed(0)
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=[MeanSquaredError()],
        verbose=False,
    )

    train_loader = _regression_loader(n=32, batch_size=8, seed=1)
    val_loader = _regression_loader(n=32, batch_size=8, seed=99)

    history = trainer.fit(train_loader, epochs=1, validation_loader=val_loader)
    record = history[0]
    # Different underlying data (different seeds) should produce different values;
    # more importantly, the two are stored under clearly separate fields.
    assert record.train_loss != record.val_loss or record.train_metrics != record.val_metrics
    assert "mse" in record.train_metrics and "mse" in record.val_metrics


def test_fit_rejects_empty_validation_loader():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    train_loader = _regression_loader()
    empty_dataset = TensorDataset(Tensor(np.zeros((1, 2))), Tensor(np.zeros((1, 1))))
    empty_val_loader = DataLoader(empty_dataset, batch_size=4, drop_last=True)

    with pytest.raises(TrainerError):
        trainer.fit(train_loader, epochs=1, validation_loader=empty_val_loader)


# -- History ---------------------------------------------------------------


def test_history_programmatic_access():
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()

    history = trainer.fit(loader, epochs=3)
    assert len(history) == 3
    assert len(history.train_losses) == 3
    assert len(history.val_losses) == 3
    for record in history:
        assert record.device == "cpu"
        assert record.samples == 32
        assert record.duration >= 0
    # indexing
    assert history[0].epoch == 1
    assert history[-1].epoch == 3


# -- Progress ---------------------------------------------------------------


def test_progress_output_can_be_disabled(capsys):
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
    loader = _regression_loader()

    trainer.fit(loader, epochs=2)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_progress_output_can_be_enabled(capsys):
    model, loss_fn, optimizer = _trainer_components()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=True)
    loader = _regression_loader()

    trainer.fit(loader, epochs=1)
    captured = capsys.readouterr()
    assert "Epoch 1/1" in captured.out
    assert "loss:" in captured.out


def test_training_works_identically_regardless_of_verbosity():
    forge.random.seed(5)
    loader_a = _regression_loader(seed=5)
    forge.random.seed(5)
    model_a = Linear(2, 1)
    trainer_a = Trainer(model=model_a, loss_fn=MSELoss(), optimizer=SGD(model_a.parameters(), lr=0.1), verbose=False)
    history_a = trainer_a.fit(loader_a, epochs=5)

    forge.random.seed(5)
    loader_b = _regression_loader(seed=5)
    forge.random.seed(5)
    model_b = Linear(2, 1)
    trainer_b = Trainer(model=model_b, loss_fn=MSELoss(), optimizer=SGD(model_b.parameters(), lr=0.1), verbose=True)
    history_b = trainer_b.fit(loader_b, epochs=5)

    np.testing.assert_allclose(history_a.train_losses, history_b.train_losses)


# -- End-to-end integration ------------------------------------------------


def test_end_to_end_regression_training_reduces_loss_and_recovers_function():
    forge.random.seed(42)
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(64, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=np.random.default_rng(0))

    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=[MeanSquaredError()],
        verbose=False,
    )

    history = trainer.fit(loader, epochs=150)

    assert history.train_losses[-1] < history.train_losses[0] * 1e-2
    np.testing.assert_allclose(model.weight.numpy().ravel(), [3.0, -2.0], atol=0.1)
    np.testing.assert_allclose(model.bias.numpy(), [1.0], atol=0.1)


def test_end_to_end_classification_training_improves_accuracy():
    forge.random.seed(7)
    rng = np.random.default_rng(7)
    n = 20
    class0 = rng.normal(loc=[-1.5, -1.5], scale=0.3, size=(n, 2))
    class1 = rng.normal(loc=[1.5, 1.5], scale=0.3, size=(n, 2))
    X = np.vstack([class0, class1])
    targets = np.array([0] * n + [1] * n)
    dataset = TensorDataset(Tensor(X), Tensor(targets))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(1))

    model = MLP(2, 8, 2)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.5)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=[Accuracy()],
        verbose=False,
    )

    history = trainer.fit(loader, epochs=60)

    assert history.train_losses[-1] < history.train_losses[0] * 0.5
    assert history[-1].train_metrics["accuracy"] >= 0.85

    eval_result = trainer.evaluate(DataLoader(dataset, batch_size=8))
    assert eval_result.metrics["accuracy"] >= 0.85
