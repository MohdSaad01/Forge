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
from forge.nn import Dropout, Linear, Module, ReLU, RNNCell
from forge.training import Trainer, generate_sequence, predict
from forge.training.inference import generate_sequence as generate_sequence_direct
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


# -- generate_sequence() (Milestone 75) --------------------------------------


class TinyStepModel(Module):
    """A minimal stepwise recurrent model: `RNNCell` -> `Linear`, one-hot input.

    Mirrors `examples/char_rnn/model.py::CharRNN`'s `step()`/`init_hidden()`
    shape exactly (the protocol `generate_sequence()` documents), without
    depending on the `examples` package.
    """

    def __init__(self, vocab_size=5, hidden_size=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.cell = RNNCell(vocab_size, hidden_size)
        self.output = Linear(hidden_size, vocab_size)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.output(h), h

    def init_hidden(self, batch_size, device="cpu"):
        return self.cell.init_hidden(batch_size, device=device)


def _one_hot(index: int, size: int) -> np.ndarray:
    row = np.zeros((1, size), dtype=np.float32)
    row[0, index] = 1.0
    return row


def _step_model(seed=0, vocab_size=5, hidden_size=6):
    forge.random.seed(seed)
    return TinyStepModel(vocab_size=vocab_size, hidden_size=hidden_size)


def _encode(token, vocab_size):
    return Tensor(_one_hot(token, vocab_size))


def test_generate_sequence_is_reexported_consistently():
    assert forge.generate_sequence is generate_sequence
    assert forge.training.generate_sequence is generate_sequence
    assert generate_sequence is generate_sequence_direct


def test_generate_sequence_output_length_includes_seed():
    model = _step_model()
    rng = np.random.default_rng(0)
    result = generate_sequence(
        model,
        seed=[0, 1],
        encode=lambda t: _encode(t, model.vocab_size),
        decode=lambda idx: idx,
        length=7,
        rng=rng,
    )
    assert result[:2] == [0, 1]
    assert len(result) == 2 + 7


def test_generate_sequence_only_produces_valid_indices():
    model = _step_model()
    rng = np.random.default_rng(1)
    result = generate_sequence(
        model,
        seed=[2],
        encode=lambda t: _encode(t, model.vocab_size),
        decode=lambda idx: idx,
        length=25,
        rng=rng,
    )
    assert all(0 <= t < model.vocab_size for t in result)


def test_generate_sequence_deterministic_given_same_rng_state():
    model = _step_model()
    kwargs = dict(seed=[0], encode=lambda t: _encode(t, model.vocab_size), decode=lambda idx: idx, length=10)
    first = generate_sequence(model, rng=np.random.default_rng(42), **kwargs)
    second = generate_sequence(model, rng=np.random.default_rng(42), **kwargs)
    assert first == second


def test_generate_sequence_defaults_to_forge_default_generator():
    model = _step_model()
    forge.random.seed(123)
    kwargs = dict(seed=[0], encode=lambda t: _encode(t, model.vocab_size), decode=lambda idx: idx, length=5)
    forge.random.seed(7)
    first = generate_sequence(model, **kwargs)
    forge.random.seed(7)
    second = generate_sequence(model, **kwargs)
    assert first == second


def test_generate_sequence_restores_training_mode():
    model = _step_model()
    model.train()
    kwargs = dict(seed=[0], encode=lambda t: _encode(t, model.vocab_size), decode=lambda idx: idx, length=3)
    generate_sequence(model, rng=np.random.default_rng(0), **kwargs)
    assert model.training is True

    model.eval()
    generate_sequence(model, rng=np.random.default_rng(0), **kwargs)
    assert model.training is False


def test_generate_sequence_rejects_non_module():
    with pytest.raises(TrainerError):
        generate_sequence("not a module", seed=[0], encode=lambda t: t, decode=lambda i: i, length=1)


def test_generate_sequence_rejects_empty_seed():
    model = _step_model()
    with pytest.raises(DataError):
        generate_sequence(model, seed=[], encode=lambda t: t, decode=lambda i: i, length=1)


def test_generate_sequence_rejects_negative_length():
    model = _step_model()
    with pytest.raises(DataError):
        generate_sequence(
            model, seed=[0], encode=lambda t: _encode(t, model.vocab_size), decode=lambda idx: idx, length=-1
        )


def test_generate_sequence_zero_length_returns_only_seed():
    model = _step_model()
    result = generate_sequence(
        model,
        seed=[0, 1, 2],
        encode=lambda t: _encode(t, model.vocab_size),
        decode=lambda idx: idx,
        length=0,
        rng=np.random.default_rng(0),
    )
    assert result == [0, 1, 2]


def test_generate_sequence_matches_manual_reference_loop():
    """generate_sequence()'s sampling loop must match a hand-written reference exactly."""
    model = _step_model(seed=5)
    vocab_size = model.vocab_size
    seed_tokens = [1, 3]
    length = 6

    def run_manual(rng):
        with no_grad():
            h = model.init_hidden(1)
            for token in seed_tokens[:-1]:
                _, h = model.step(_encode(token, vocab_size), h)
            generated = list(seed_tokens)
            current = seed_tokens[-1]
            for _ in range(length):
                logits, h = model.step(_encode(current, vocab_size), h)
                probs = np.exp(logits.numpy()[0])
                probs = probs / probs.sum()
                next_idx = int(rng.choice(vocab_size, p=probs))
                current = next_idx
                generated.append(current)
        return generated

    expected = run_manual(np.random.default_rng(99))
    actual = generate_sequence(
        model,
        seed=seed_tokens,
        encode=lambda t: _encode(t, vocab_size),
        decode=lambda idx: idx,
        length=length,
        rng=np.random.default_rng(99),
    )
    assert actual == expected
