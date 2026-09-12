"""Milestone 79 tests: `forge.train()`, the single-call high-level training
entry point over `Trainer`/`DataLoader`. See `forge/training/api.py`.

Covers: basic training through the new API, correct parameter updates,
Dataset vs. already-built-DataLoader input, device resolution/movement
(including that `Module.to()`'s in-place-identity guarantee keeps an
optimizer built before the call valid afterward), validation-dataset
behavior, compatibility with `Trainer`'s own return type, and error handling
for invalid required arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.exceptions import DataError, PersistenceError, TrainerError
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD, Adam
from forge.serialization import load_classes, load_preprocessing, register_module
from forge.training import Accuracy, TrainAndSaveResult, Trainer, TrainingHistory, train, train_and_save


class _MLP(Module):
    def __init__(self, in_features=2, hidden=4, out_features=1):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Registered for persistence -- train_and_save()'s tests below need to
# actually save/reload this class (matches tests/test_inference.py's own
# _MLP_M78Test registration precedent for save_and_verify()).
register_module("_MLP_M81Test", _MLP, get_config=lambda m: {
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


def _params(model) -> "dict[str, np.ndarray]":
    return {name: p.numpy().copy() for name, p in model.named_parameters()}


# -- basic training / parameter updates ---------------------------------------


def test_train_returns_a_training_history():
    model = _MLP()
    history = train(
        model, _regression_dataset(),
        loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05), epochs=3,
    )
    assert isinstance(history, TrainingHistory)
    assert len(history) == 3


def test_train_actually_updates_model_parameters():
    model = _MLP()
    before = _params(model)
    train(
        model, _regression_dataset(),
        loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), epochs=5,
    )
    after = _params(model)
    assert any(not np.array_equal(before[name], after[name]) for name in before)


def test_train_reduces_loss_over_epochs():
    model = _MLP(hidden=8)
    history = train(
        model, _regression_dataset(n=64),
        loss=MSELoss(), optimizer=Adam(model.parameters(), lr=0.05), epochs=20, verbose=False,
    )
    assert history[-1].train_loss < history[0].train_loss


def test_train_is_equivalent_to_manual_trainer_construction():
    """forge.train() must be pure orchestration: training a model through
    train() must produce bit-identical results to building the same
    DataLoader+Trainer by hand, given identical initial parameters, data,
    and (unshuffled) batch order."""
    dataset = _regression_dataset(n=32)

    forge.random.seed(0)
    model_a = _MLP()
    train(
        model_a, dataset, loss=MSELoss(), optimizer=SGD(model_a.parameters(), lr=0.1),
        epochs=3, batch_size=8, shuffle=False, verbose=False,
    )

    forge.random.seed(0)
    model_b = _MLP()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    trainer = Trainer(model=model_b, loss_fn=MSELoss(), optimizer=SGD(model_b.parameters(), lr=0.1), verbose=False)
    trainer.fit(loader, epochs=3)

    params_a = _params(model_a)
    params_b = _params(model_b)
    for name in params_a:
        np.testing.assert_array_equal(params_a[name], params_b[name])


# -- dataset / dataloader integration ------------------------------------------


def test_train_accepts_a_plain_dataset():
    model = _MLP()
    history = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=2,
    )
    assert len(history) == 2


def test_train_accepts_an_already_built_dataloader():
    model = _MLP()
    loader = DataLoader(_regression_dataset(), batch_size=4, shuffle=True, drop_last=True)
    history = train(
        model, loader, loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05), epochs=2,
    )
    assert len(history) == 2


def test_train_respects_batch_size_for_a_plain_dataset():
    model = _MLP()
    dataset = _regression_dataset(n=17)
    history = train(
        model, dataset, loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.01),
        epochs=1, batch_size=5, verbose=False,
    )
    assert history[0].samples == 17  # 4 batches of 5 + 1 of 2, drop_last=False


def test_train_rejects_a_non_dataset_non_dataloader():
    model = _MLP()
    with pytest.raises(DataError):
        train(model, [1, 2, 3], loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.01), epochs=1)


# -- validation -----------------------------------------------------------------


def test_train_reports_validation_when_a_validation_dataset_is_given():
    model = _MLP()
    history = train(
        model, _regression_dataset(n=32, seed=0), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=2,
        validation_dataset=_regression_dataset(n=16, seed=1), verbose=False,
    )
    assert history[0].val_loss is not None
    assert history[-1].val_loss is not None


def test_train_validation_is_none_without_a_validation_dataset():
    model = _MLP()
    history = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=1, verbose=False,
    )
    assert history[0].val_loss is None
    assert history[0].val_metrics == {}


def test_train_validation_dataset_accepts_a_dataloader_too():
    model = _MLP()
    val_loader = DataLoader(_regression_dataset(n=16, seed=1), batch_size=4)
    history = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=1,
        validation_dataset=val_loader, verbose=False,
    )
    assert history[0].val_loss is not None


# -- metrics ----------------------------------------------------------------


def test_train_passes_metrics_through_to_trainer():
    model = _MLP(in_features=2, out_features=3)
    history = train(
        model, _classification_dataset(), loss=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.01), epochs=1,
        metrics=[Accuracy()], verbose=False,
    )
    assert "accuracy" in history[0].train_metrics


# -- device -------------------------------------------------------------------


def test_train_defaults_device_to_cpu_for_a_fresh_model():
    model = _MLP()
    history = train(
        model, _regression_dataset(), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=1, verbose=False,
    )
    assert history[0].device == "cpu"
    assert str(model.device) == "cpu"


def test_train_explicit_device_moves_the_model_in_place():
    model = _MLP()
    optimizer = SGD(model.parameters(), lr=0.05)  # built before the device move
    history = train(
        model, _regression_dataset(), loss=MSELoss(), optimizer=optimizer,
        epochs=1, device="cpu", verbose=False,
    )
    assert str(model.device) == "cpu"
    assert history[0].device == "cpu"


def test_train_keeps_optimizer_parameter_identity_valid_across_a_device_move():
    """optimizer=Adam(model.parameters(), ...) is built BEFORE train() moves
    the model -- Module.to()'s in-place-identity guarantee is what makes this
    safe; if it broke, training would silently update stale, detached
    Parameter copies and the model's real parameters would never change."""
    model = _MLP()
    optimizer = Adam(model.parameters(), lr=0.1)
    before = _params(model)
    train(
        model, _regression_dataset(n=64), loss=MSELoss(), optimizer=optimizer,
        epochs=5, device="cpu", verbose=False,
    )
    after = _params(model)
    assert any(not np.array_equal(before[name], after[name]) for name in before)


# -- error handling for invalid required arguments -----------------------------


def test_train_requires_loss_optimizer_epochs_as_keywords():
    model = _MLP()
    with pytest.raises(TypeError):
        train(model, _regression_dataset())  # missing loss/optimizer/epochs


def test_train_rejects_a_non_module_model():
    with pytest.raises(TrainerError):
        train(object(), _regression_dataset(), loss=MSELoss(), optimizer=SGD([], lr=0.1), epochs=1)


def test_train_rejects_a_non_loss():
    model = _MLP()
    with pytest.raises(TrainerError):
        train(model, _regression_dataset(), loss=lambda p, t: p, optimizer=SGD(model.parameters(), lr=0.1), epochs=1)


def test_train_rejects_a_non_optimizer():
    model = _MLP()
    with pytest.raises(TrainerError):
        train(model, _regression_dataset(), loss=MSELoss(), optimizer=object(), epochs=1)


def test_train_rejects_invalid_epochs():
    model = _MLP()
    with pytest.raises(TrainerError):
        train(model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), epochs=0)


def test_train_is_exported_at_top_level():
    assert forge.train is train


# -- train_and_save() (Milestone 81) -------------------------------------------


def test_train_and_save_is_reexported_consistently():
    assert forge.train_and_save is train_and_save
    assert forge.training.train_and_save is train_and_save


def test_train_and_save_returns_a_train_and_save_result(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=3, path=str(tmp_path / "model.forge"), sample=x, verbose=False,
    )
    assert isinstance(result, TrainAndSaveResult)
    assert isinstance(result.history, TrainingHistory)
    assert len(result.history) == 3


def test_train_and_save_trains_exactly_like_train(tmp_path):
    """train_and_save() must be pure composition: training must be
    bit-identical to a direct train() call given identical seeded initial
    parameters, data, and (unshuffled) batch order."""
    dataset = _regression_dataset(n=32)
    x = Tensor(np.zeros((1, 2), dtype=np.float32))

    forge.random.seed(0)
    model_a = _MLP()
    train(
        model_a, dataset, loss=MSELoss(), optimizer=SGD(model_a.parameters(), lr=0.1),
        epochs=3, batch_size=8, shuffle=False, verbose=False,
    )

    forge.random.seed(0)
    model_b = _MLP()
    train_and_save(
        model_b, dataset, loss=MSELoss(), optimizer=SGD(model_b.parameters(), lr=0.1),
        epochs=3, batch_size=8, shuffle=False, verbose=False,
        path=str(tmp_path / "model.forge"), sample=x,
    )

    params_a = _params(model_a)
    params_b = _params(model_b)
    for name in params_a:
        np.testing.assert_array_equal(params_a[name], params_b[name])


def test_train_and_save_writes_a_working_reloadable_artifact(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    path = str(tmp_path / "model.forge")

    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=2, path=path, sample=x, verbose=False,
    )

    assert isinstance(result.model, _MLP)
    assert result.model is not model
    reloaded = forge.load_model(path)
    with forge.no_grad():
        expected = model(x).numpy()
        actual = reloaded(x).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_train_and_save_passes_through_preprocessing_and_classes(tmp_path):
    model = _MLP(in_features=2, out_features=3)
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    path = str(tmp_path / "model.forge")

    train_and_save(
        model, _classification_dataset(), loss=CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=0.01), epochs=1, path=path, sample=x, verbose=False,
        preprocessing=None, classes=["a", "b", "c"],
    )

    assert load_classes(path) == ["a", "b", "c"]


def test_train_and_save_exposes_the_final_epochs_validation_result(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _regression_dataset(n=32, seed=0), loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=0.05), epochs=3, verbose=False,
        validation_dataset=_regression_dataset(n=16, seed=1),
        path=str(tmp_path / "model.forge"), sample=x,
    )
    assert result.val_loss == result.history[-1].val_loss
    assert result.val_metrics == result.history[-1].val_metrics
    assert result.val_loss is not None


def test_train_and_save_validation_is_none_without_a_validation_dataset(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    result = train_and_save(
        model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
        epochs=1, verbose=False, path=str(tmp_path / "model.forge"), sample=x,
    )
    assert result.val_loss is None
    assert result.val_metrics == {}


def test_train_and_save_raises_persistence_error_on_prediction_mismatch(tmp_path, monkeypatch):
    """A reloaded model that diverges from the just-trained model must be
    caught -- the exact condition save_and_verify() exists to catch,
    propagated unchanged through train_and_save()."""
    from forge.training import api as api_module
    from forge.training import inference as inference_module

    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    path = str(tmp_path / "model.forge")

    real_predict = inference_module.predict
    call_count = {"n": 0}

    def _flaky_predict(m, inputs, device=None):
        call_count["n"] += 1
        result = real_predict(m, inputs, device=device)
        if call_count["n"] == 2:
            return Tensor(result.numpy() + 100.0)
        return result

    monkeypatch.setattr(inference_module, "predict", _flaky_predict)
    with pytest.raises(PersistenceError):
        api_module.train_and_save(
            model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.05),
            epochs=1, path=path, sample=x, verbose=False,
        )


def test_train_and_save_rejects_a_non_module_model(tmp_path):
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(TrainerError):
        train_and_save(
            object(), _regression_dataset(), loss=MSELoss(), optimizer=SGD([], lr=0.1),
            epochs=1, path=str(tmp_path / "model.forge"), sample=x,
        )


def test_train_and_save_rejects_a_non_tensor_sample(tmp_path):
    model = _MLP()
    with pytest.raises(DataError):
        train_and_save(
            model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1),
            epochs=1, path=str(tmp_path / "model.forge"), sample=np.zeros((1, 2), dtype=np.float32),
        )


def test_train_and_save_requires_path_and_sample_as_keywords(tmp_path):
    model = _MLP()
    x = Tensor(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(TypeError):
        train_and_save(
            model, _regression_dataset(), loss=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), epochs=1,
        )  # missing path/sample
