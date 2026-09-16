"""Milestone 100 tests: `EarlyStopping` -- `Trainer.fit()` stopping training
and optionally restoring the best-seen model state based on a monitored
validation quantity. See `docs/architecture/training-engine.md`'s
**Early stopping** section and `forge/training/early_stopping.py`.

Deterministic coverage uses a monkeypatched `Trainer.evaluate()` to script
an exact sequence of validation losses per epoch (real training still runs
underneath, so the model's actual parameters change every epoch) -- this
makes patience/min_delta/best-epoch logic fully deterministic and testable
without depending on real training dynamics. `test_val_accuracy_monitor_...`
and `test_determinism_...` additionally exercise real (unscripted)
validation end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.exceptions import TrainerError
from forge.nn import Linear, Module, ReLU, Sequential
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD
from forge.training import Accuracy, EarlyStopping, EvaluationResult, Trainer, train, train_and_save


class _MLP(Module):
    def __init__(self, in_features=2, hidden=4, out_features=1):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _regression_loader(n=16, batch_size=8, seed=0):
    forge.random.seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _trainer(metrics=None, seed=0):
    forge.random.seed(seed)
    model = _MLP()
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05)
    return Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, verbose=False, metrics=metrics)


def _script_validation(trainer, values):
    """Replace `trainer.evaluate` with a stub returning `values[i]` as val_loss on call i.

    Real training (`_run_training_epoch`) is untouched -- only the reported
    validation loss is scripted, so parameter updates each epoch remain real
    and deterministic.
    """
    it = iter(values)

    def fake_evaluate(loader):
        return EvaluationResult(loss=next(it), metrics={}, samples=1, duration=0.0, device="cpu")

    trainer.evaluate = fake_evaluate


# -- EarlyStopping configuration ----------------------------------------------


def test_early_stopping_default_construction():
    es = EarlyStopping()
    assert es.monitor == "val_loss"
    assert es.patience == 5
    assert es.min_delta == 0.0
    assert es.restore_best is True
    assert es.mode == "min"


def test_early_stopping_rejects_monitor_without_val_prefix():
    with pytest.raises(TrainerError):
        EarlyStopping(monitor="loss")


def test_early_stopping_rejects_bare_val_prefix():
    with pytest.raises(TrainerError):
        EarlyStopping(monitor="val_")


def test_early_stopping_rejects_nonpositive_patience():
    with pytest.raises(TrainerError):
        EarlyStopping(patience=0)
    with pytest.raises(TrainerError):
        EarlyStopping(patience=-1)


def test_early_stopping_rejects_non_int_patience():
    with pytest.raises(TrainerError):
        EarlyStopping(patience=2.5)


def test_early_stopping_rejects_negative_min_delta():
    with pytest.raises(TrainerError):
        EarlyStopping(min_delta=-0.1)


def test_early_stopping_rejects_bad_mode():
    with pytest.raises(TrainerError):
        EarlyStopping(mode="sideways")


def test_forge_top_level_and_training_module_expose_early_stopping():
    assert forge.EarlyStopping is EarlyStopping
    assert forge.training.EarlyStopping is EarlyStopping


# -- Validation-data requirement ------------------------------------------------


def test_fit_rejects_early_stopping_without_validation_loader():
    trainer = _trainer()
    loader = _regression_loader()
    with pytest.raises(TrainerError):
        trainer.fit(loader, epochs=5, early_stopping=EarlyStopping())


def test_fit_rejects_non_early_stopping_object():
    trainer = _trainer()
    loader = _regression_loader()
    with pytest.raises(TrainerError):
        trainer.fit(loader, epochs=5, validation_loader=loader, early_stopping=object())


def test_train_rejects_early_stopping_without_validation_dataset():
    forge.random.seed(1)
    model = _MLP()
    optimizer = SGD(model.parameters(), lr=0.05)
    loader = _regression_loader(seed=1)
    with pytest.raises(TrainerError):
        train(
            model, loader.dataset, loss=MSELoss(), optimizer=optimizer, epochs=5,
            batch_size=8, early_stopping=EarlyStopping(), verbose=False,
        )


def test_unknown_monitor_metric_raises_clearly():
    trainer = _trainer(metrics=None)
    loader = _regression_loader()
    _script_validation(trainer, [1.0, 0.9])
    with pytest.raises(TrainerError):
        trainer.fit(
            loader, epochs=2, validation_loader=loader,
            early_stopping=EarlyStopping(monitor="val_accuracy", mode="max"),
        )


# -- Backward compatibility ------------------------------------------------------


def test_history_defaults_when_no_early_stopping():
    trainer = _trainer()
    loader = _regression_loader()
    history = trainer.fit(loader, epochs=3)
    assert len(history) == 3
    assert history.stopped_early is False
    assert history.best_epoch is None
    assert history.best_monitored_value is None
    assert history.monitored_quantity is None


# -- Core stopping semantics ------------------------------------------------------


def test_first_validation_result_establishes_baseline():
    trainer = _trainer()
    loader = _regression_loader()
    _script_validation(trainer, [2.5])
    history = trainer.fit(loader, epochs=1, validation_loader=loader, early_stopping=EarlyStopping(patience=5))
    assert history.best_epoch == 1
    assert history.best_monitored_value == pytest.approx(2.5)
    assert history.stopped_early is False


def test_continuous_improvement_runs_every_requested_epoch():
    trainer = _trainer()
    loader = _regression_loader()
    values = [1.0 - 0.1 * i for i in range(10)]
    _script_validation(trainer, values)
    history = trainer.fit(loader, epochs=10, validation_loader=loader, early_stopping=EarlyStopping(patience=3))
    assert len(history) == 10
    assert history.stopped_early is False
    assert history.best_epoch == 10
    assert history.best_monitored_value == pytest.approx(values[-1])
    assert history.monitored_quantity == "val_loss"


def test_no_improvement_stops_after_patience_consecutive_non_improving_epochs():
    """patience=2: baseline at epoch 1, then exactly 2 non-improving epochs stop training
    after epoch 3 -- not epoch 2 (patience counts *consecutive* non-improvements, and the
    epoch that exhausts patience still itself completes and is recorded)."""
    trainer = _trainer()
    loader = _regression_loader()
    _script_validation(trainer, [1.0] * 20)
    history = trainer.fit(loader, epochs=20, validation_loader=loader, early_stopping=EarlyStopping(patience=2))
    assert len(history) == 3
    assert history.stopped_early is True
    assert history.best_epoch == 1
    assert history.best_monitored_value == pytest.approx(1.0)


def test_min_delta_prevents_tiny_improvements_from_resetting_patience():
    trainer = _trainer()
    loader = _regression_loader()
    # Each step "improves" by 0.001 -- below min_delta=0.01, so never counts.
    values = [1.0, 0.999, 0.998, 0.997, 0.996]
    _script_validation(trainer, values)
    history = trainer.fit(
        loader, epochs=10, validation_loader=loader,
        early_stopping=EarlyStopping(patience=2, min_delta=0.01),
    )
    assert len(history) == 3
    assert history.stopped_early is True
    assert history.best_epoch == 1
    assert history.best_monitored_value == pytest.approx(1.0)


def test_min_delta_still_allows_large_enough_improvements():
    trainer = _trainer()
    loader = _regression_loader()
    values = [1.0, 0.8, 0.5, 0.49, 0.48]
    _script_validation(trainer, values)
    history = trainer.fit(
        loader, epochs=10, validation_loader=loader,
        early_stopping=EarlyStopping(patience=2, min_delta=0.1),
    )
    # epoch1 baseline=1.0; epoch2 0.8 improves (>=0.1) -> best=0.8; epoch3 0.5 improves -> best=0.5;
    # epoch4 0.49 does not improve (needs < 0.4); epoch5 0.48 does not improve either -> stop.
    assert len(history) == 5
    assert history.stopped_early is True
    assert history.best_epoch == 3
    assert history.best_monitored_value == pytest.approx(0.5)


def test_max_mode_improvement_direction():
    trainer = _trainer()
    loader = _regression_loader()  # data itself is irrelevant; validation is fully scripted
    _script_validation(trainer, [0.5])

    it_values = iter([0.5, 0.6, 0.55, 0.5])

    def fake_evaluate(_loader):
        return EvaluationResult(loss=0.0, metrics={"accuracy": next(it_values)}, samples=1, duration=0.0, device="cpu")

    trainer.evaluate = fake_evaluate
    history = trainer.fit(
        loader, epochs=10, validation_loader=loader,
        early_stopping=EarlyStopping(monitor="val_accuracy", mode="max", patience=2),
    )
    # epoch1 baseline=0.5; epoch2 0.6 improves -> best=0.6, epoch=2; epoch3 0.55 no improve (count=1);
    # epoch4 0.5 no improve (count=2>=patience) -> stop after epoch4.
    assert len(history) == 4
    assert history.stopped_early is True
    assert history.best_epoch == 2
    assert history.best_monitored_value == pytest.approx(0.6)
    assert history.monitored_quantity == "val_accuracy"


# -- Best-model restoration -------------------------------------------------------


def test_restore_best_restores_parameters_from_the_actual_best_epoch():
    trainer = _trainer(seed=5)
    loader = _regression_loader(seed=5)
    values = [1.0, 0.5, 0.6, 0.7]  # best is epoch 2; training continues (for real) through epoch 4
    _script_validation(trainer, values)
    history = trainer.fit(
        loader, epochs=10, validation_loader=loader,
        early_stopping=EarlyStopping(patience=2, restore_best=True),
    )
    assert history.best_epoch == 2
    assert history.stopped_early is True
    assert len(history) == 4
    restored_params = [p.numpy().copy() for p in trainer.model.parameters()]

    # Ground truth: an independent run, identical seed/data/optimizer, trained for
    # exactly the 2 real epochs that produced the best snapshot.
    ref_trainer = _trainer(seed=5)
    ref_loader = _regression_loader(seed=5)
    ref_trainer.fit(ref_loader, epochs=2)
    expected_params = [p.numpy().copy() for p in ref_trainer.model.parameters()]

    assert len(restored_params) == len(expected_params) > 0
    for restored, expected in zip(restored_params, expected_params):
        np.testing.assert_allclose(restored, expected, atol=1e-6)


def test_restore_best_false_keeps_the_final_completed_epochs_parameters():
    trainer = _trainer(seed=6)
    loader = _regression_loader(seed=6)
    values = [1.0, 0.5, 0.6, 0.7]
    _script_validation(trainer, values)
    history = trainer.fit(
        loader, epochs=10, validation_loader=loader,
        early_stopping=EarlyStopping(patience=2, restore_best=False),
    )
    assert history.best_epoch == 2
    assert len(history) == 4
    final_params = [p.numpy().copy() for p in trainer.model.parameters()]

    ref_trainer = _trainer(seed=6)
    ref_loader = _regression_loader(seed=6)
    ref_trainer.fit(ref_loader, epochs=4)  # all 4 epochs that actually ran, no restoration
    expected_params = [p.numpy().copy() for p in ref_trainer.model.parameters()]

    for final, expected in zip(final_params, expected_params):
        np.testing.assert_allclose(final, expected, atol=1e-6)


# -- Real (unscripted) validation end-to-end --------------------------------------


def test_val_accuracy_monitor_works_end_to_end_with_real_validation():
    forge.random.seed(3)
    rng = np.random.default_rng(3)
    class0 = rng.normal(loc=[-1.5, -1.5], scale=0.3, size=(20, 2))
    class1 = rng.normal(loc=[1.5, 1.5], scale=0.3, size=(20, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 20 + [1] * 20)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = _MLP(in_features=2, hidden=8, out_features=2)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, loss_fn=CrossEntropyLoss(), optimizer=optimizer, verbose=False, metrics=[Accuracy()])
    history = trainer.fit(
        loader, epochs=15, validation_loader=loader,
        early_stopping=EarlyStopping(monitor="val_accuracy", mode="max", patience=5),
    )
    assert history.monitored_quantity == "val_accuracy"
    assert history.best_epoch is not None
    assert 0.0 <= history.best_monitored_value <= 1.0


def test_determinism_same_config_same_stopping_outcome():
    def run():
        forge.random.seed(9)
        model = _MLP()
        optimizer = SGD(model.parameters(), lr=0.05)
        loader = _regression_loader(seed=9)
        return train(
            model, loader.dataset,
            loss=MSELoss(), optimizer=optimizer, epochs=25,
            validation_dataset=loader.dataset, batch_size=8, shuffle=False,
            early_stopping=EarlyStopping(patience=3, restore_best=True),
            verbose=False,
        )

    r1 = run()
    r2 = run()
    assert r1.epochs_completed == r2.epochs_completed
    assert r1.stopped_early == r2.stopped_early
    assert r1.best_epoch == r2.best_epoch
    assert r1.best_monitored_value == pytest.approx(r2.best_monitored_value)
    p1 = next(r1.model.parameters()).numpy()
    p2 = next(r2.model.parameters()).numpy()
    np.testing.assert_allclose(p1, p2, atol=1e-8)


# -- train_and_save() integration -------------------------------------------------


def test_train_and_save_saves_the_restored_best_model(tmp_path):
    forge.random.seed(11)
    # A registered Sequential/Linear/ReLU tree, not the test-local _MLP class --
    # save_and_verify() requires every module type to be registered for persistence.
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    optimizer = SGD(model.parameters(), lr=0.05)
    loader = _regression_loader(seed=11)
    sample_x, _ = loader.dataset[0]
    sample_x = sample_x.reshape(1, 2)

    result = train_and_save(
        model, loader.dataset,
        loss=MSELoss(), optimizer=optimizer, epochs=25,
        validation_dataset=loader.dataset, batch_size=8, shuffle=False,
        path=str(tmp_path / "model.forge"), sample=sample_x,
        early_stopping=EarlyStopping(patience=3, restore_best=True),
        verbose=False,
    )

    assert result.artifact_path == str(tmp_path / "model.forge")
    assert result.best_epoch == result.history.best_epoch
    assert result.stopped_early == result.history.stopped_early
    # save_and_verify() already asserts the reloaded artifact reproduces the live model's
    # predictions bit-for-bit (raising PersistenceError otherwise) -- the correctness of
    # *which* epoch's parameters ended up live is proven independently above
    # (test_restore_best_restores_parameters_from_the_actual_best_epoch).
    live_pred = forge.predict(model, sample_x)
    reloaded_pred = forge.predict(result.model, sample_x)
    np.testing.assert_allclose(live_pred.numpy(), reloaded_pred.numpy(), atol=1e-5)
