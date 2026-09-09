"""Milestone 68 tests: `forge.predict()` / `forge.training.predict()`.

Covers the standalone post-training inference path -- single-`Tensor` and
batched-iterable input, eval-mode/no_grad semantics, training-mode
restoration, device handling, and validation errors. See
`forge/training/inference.py` and `docs/architecture/training-engine.md`'s
**Inference** section.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.data import DataLoader, TensorDataset
from forge.exceptions import DataError, TrainerError
from forge.nn import Dropout, Linear, Module, ReLU
from forge.training import Trainer, predict
from forge.training.inference import predict as predict_direct


class MLP(Module):
    def __init__(self, in_features=4, hidden=8, out_features=3):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _model(seed=0):
    forge.random.seed(seed)
    return MLP()


def _features(n=11, in_features=4, seed=0):
    rng = np.random.default_rng(seed)
    return Tensor(rng.uniform(-1, 1, size=(n, in_features)).astype(np.float32))


def test_predict_is_reexported_consistently():
    assert forge.predict is predict
    assert forge.training.predict is predict
    assert predict is predict_direct


def test_predict_single_tensor_matches_manual_forward():
    model = _model()
    x = _features(n=6)
    with no_grad():
        expected = model(x).numpy()
    result = predict(model, x)
    assert isinstance(result, Tensor)
    assert result.device.type == "cpu"
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_predict_over_iterable_concatenates_in_order():
    model = _model()
    x = _features(n=11)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    result = predict(model, loader)
    assert result.shape == (11, 3)

    with no_grad():
        expected = model(x).numpy()
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_predict_over_iterable_of_feature_target_tuples():
    model = _model()
    x = _features(n=9)
    y = Tensor(np.zeros((9,), dtype=np.int64))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    result = predict(model, loader)
    with no_grad():
        expected = model(x).numpy()
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_predict_restores_training_mode():
    model = _model()
    model.train()
    predict(model, _features())
    assert model.training is True

    model.eval()
    predict(model, _features())
    assert model.training is False


class DropoutMLP(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(4, 16)
        self.drop = Dropout(p=0.9)

    def forward(self, x):
        return self.drop(self.fc(x))


def test_predict_disables_dropout_even_when_model_left_in_train_mode():
    forge.random.seed(0)
    model = DropoutMLP()
    model.train()

    x = _features(n=5)
    first = predict(model, x).numpy()
    second = predict(model, x).numpy()
    # Eval-mode Dropout is a deterministic identity -- two calls must agree
    # exactly, proving predict() actually switched to eval mode rather than
    # running the model's own (train-mode) flag.
    np.testing.assert_array_equal(first, second)
    assert model.training is True


def test_predict_output_has_no_grad_graph():
    model = _model()
    result = predict(model, _features())
    assert result.requires_grad is False


def test_predict_output_always_on_cpu():
    model = _model()
    result = predict(model, _features())
    assert str(result.device) == "cpu"


def test_predict_rejects_non_module():
    with pytest.raises(TrainerError):
        predict("not a module", _features())


def test_predict_rejects_non_tensor_batch_element():
    model = _model()
    with pytest.raises(DataError):
        predict(model, [np.zeros((2, 4), dtype=np.float32)])


def test_predict_rejects_empty_iterable():
    model = _model()
    with pytest.raises(DataError):
        predict(model, [])


def test_predict_device_defaults_to_model_device():
    model = _model()
    assert str(model.device) == "cpu"
    result = predict(model, _features())
    assert str(result.device) == "cpu"


def test_predict_explicit_device_override_accepted_on_cpu_only_machine():
    model = _model()
    result = predict(model, _features(), device="cpu")
    np.testing.assert_allclose(
        result.numpy(),
        predict(model, _features(seed=0)).numpy(),
    )


def test_predict_parameterless_model_defaults_to_cpu():
    relu = ReLU()
    x = _features(n=4)
    result = predict(relu, x)
    assert str(result.device) == "cpu"
    np.testing.assert_allclose(result.numpy(), np.maximum(x.numpy(), 0.0), atol=1e-6)


def test_predict_matches_trainer_evaluate_predictions_on_same_data():
    """predict() and Trainer's own forward pass must agree bit-for-bit."""
    from forge.nn.loss import MSELoss
    from forge.optim import SGD

    model = _model()
    x = _features(n=8)
    y = Tensor(np.zeros((8, 3), dtype=np.float32))
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.01), verbose=False)
    with no_grad():
        trainer_forward = model(x).numpy()

    result = predict(model, x)
    np.testing.assert_allclose(result.numpy(), trainer_forward, atol=1e-6)
    # Trainer.evaluate() must not have mutated parameters, and predict()
    # must not either -- both are inference-only.
    trainer.evaluate(loader)
    with no_grad():
        after = model(x).numpy()
    np.testing.assert_allclose(trainer_forward, after, atol=1e-6)
