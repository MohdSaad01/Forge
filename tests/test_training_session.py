"""Milestone 73 tests: `forge.training.start_training_session()`/`TrainingSession`
-- the fresh-or-resumed `Trainer` construction workflow extracted from the
identical ~10-line branch every checkpoint-capable example (`mnist`,
`regression`, `resnet`, `segmentation`, `autoencoder`,
`waveform_classification`, `image_folder_classification`) used to hand-roll.
See `forge/training/session.py` and `docs/development/m73-reusable-training-workflow.md`.

Covers: fresh-session construction, resumed-session construction (model/
optimizer/epoch/global_step carried from the checkpoint), the automatic
`data_loader_rng_state` round trip through `TrainingSession.save_checkpoint()`
(including the resume-equivalence guarantee this closes for examples that
never got Milestone 65's fix), backward compatibility with a checkpoint saved
by plain `Trainer.save_checkpoint()` (no `data_loader_rng_state` key), and
construction-time validation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.exceptions import TrainerError
from forge.nn import Linear, Module, ReLU
from forge.nn.loss import MSELoss
from forge.optim import SGD, Adam
from forge.serialization import load_checkpoint, register_module
from forge.training import MeanSquaredError, TrainingSession, start_training_session


class _MLP(Module):
    def __init__(self, in_features=2, hidden=4, out_features=1):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


register_module(
    "_MLP_M73_session_test",
    _MLP,
    get_config=lambda m: {
        "in_features": m.fc1.in_features,
        "hidden": m.fc1.out_features,
        "out_features": m.fc2.out_features,
    },
)


def _build_model():
    return _MLP()


def _build_optimizer(params):
    return Adam(params, lr=1e-2)


def _dataset(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    return TensorDataset(Tensor(X), Tensor(y))


def _params(model) -> "dict[str, np.ndarray]":
    return {name: p.numpy().copy() for name, p in model.named_parameters()}


# -- fresh sessions -----------------------------------------------------------


def test_fresh_session_builds_model_and_optimizer_via_factories():
    session = start_training_session(
        build_model=_build_model,
        build_optimizer=_build_optimizer,
        loss_fn=MSELoss(),
        seed=0,
    )
    assert isinstance(session, TrainingSession)
    assert session.resumed is False
    assert session.checkpoint is None
    assert isinstance(session.trainer.model, _MLP)
    assert isinstance(session.trainer.optimizer, Adam)
    assert session.trainer.epoch == 0
    assert session.trainer.global_step == 0


def test_fresh_session_seeds_forge_random_before_building_the_model():
    session_a = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=42
    )
    session_b = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=42
    )
    params_a = _params(session_a.trainer.model)
    params_b = _params(session_b.trainer.model)
    for name in params_a:
        np.testing.assert_array_equal(params_a[name], params_b[name])


def test_fresh_session_data_loader_rng_matches_a_plain_default_rng_of_the_same_seed():
    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=7
    )
    expected = np.random.default_rng(7)
    assert session.data_loader_rng.bit_generator.state == expected.bit_generator.state


def test_session_moves_model_to_requested_device():
    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=0, device="cpu"
    )
    assert str(session.trainer.device) == "cpu"
    assert str(session.trainer.model.device) == "cpu"


def test_session_passes_metrics_through_to_trainer():
    session = start_training_session(
        build_model=_build_model,
        build_optimizer=_build_optimizer,
        loss_fn=MSELoss(),
        seed=0,
        metrics=[MeanSquaredError()],
    )
    assert "mse" in session.trainer.metrics


# -- construction validation ---------------------------------------------------


def test_rejects_non_callable_build_model():
    with pytest.raises(TrainerError):
        start_training_session(
            build_model=object(), build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=0
        )


def test_rejects_non_callable_build_optimizer():
    with pytest.raises(TrainerError):
        start_training_session(
            build_model=_build_model, build_optimizer=object(), loss_fn=MSELoss(), seed=0
        )


def test_invalid_loss_fn_still_raises_trainer_error_via_trainer_construction():
    with pytest.raises(TrainerError):
        start_training_session(
            build_model=_build_model,
            build_optimizer=_build_optimizer,
            loss_fn=lambda p, t: p,
            seed=0,
        )


# -- save_checkpoint() / resume round trip -------------------------------------


def test_save_checkpoint_merges_data_loader_rng_state(tmp_path):
    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=3
    )
    path = tmp_path / "ckpt.forge"
    session.save_checkpoint(str(path), extra={"note": "hello"})

    checkpoint = load_checkpoint(str(path))
    assert checkpoint.extra["note"] == "hello"
    assert "data_loader_rng_state" in checkpoint.extra
    assert checkpoint.extra["data_loader_rng_state"] == session.data_loader_rng.bit_generator.state
    json.dumps(checkpoint.extra)  # must be JSON-safe


def test_save_checkpoint_rejects_caller_supplied_data_loader_rng_state_key(tmp_path):
    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=0
    )
    with pytest.raises(TrainerError):
        session.save_checkpoint(str(tmp_path / "ckpt.forge"), extra={"data_loader_rng_state": {}})


def test_resume_restores_model_optimizer_epoch_and_global_step(tmp_path):
    path = tmp_path / "ckpt.forge"
    loader = DataLoader(_dataset(), batch_size=8, shuffle=False)

    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=0
    )
    session.trainer.fit(loader, epochs=2)
    session.save_checkpoint(str(path))

    resumed = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=0, resume=str(path)
    )
    assert resumed.resumed is True
    assert resumed.checkpoint is not None
    assert resumed.trainer.epoch == 2
    assert resumed.trainer.global_step == session.trainer.global_step
    np.testing.assert_array_equal(
        _params(resumed.trainer.model)["fc1.weight"], _params(session.trainer.model)["fc1.weight"]
    )


def test_resume_without_a_saved_data_loader_rng_state_falls_back_to_a_fresh_generator(tmp_path):
    """A checkpoint saved via plain trainer.save_checkpoint() (not
    TrainingSession.save_checkpoint()) has no 'data_loader_rng_state' key --
    resuming from it must not crash, and data_loader_rng is left at its
    freshly-seed-derived state (matching every example's pre-M73 behavior).
    """
    path = tmp_path / "ckpt.forge"
    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=5
    )
    session.trainer.save_checkpoint(str(path))  # bypasses TrainingSession.save_checkpoint()

    resumed = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=5, resume=str(path)
    )
    expected = np.random.default_rng(5)
    assert resumed.data_loader_rng.bit_generator.state == expected.bit_generator.state


def test_resume_equivalence_full_training_matches_partial_then_resume_with_shuffle(tmp_path):
    """The M65 guarantee, reproduced through the new abstraction: continuous
    training and interrupt-then-resume must reach identical final parameters
    even with shuffle=True, because session.data_loader_rng's stream position
    is saved/restored automatically by TrainingSession.save_checkpoint()."""
    dataset = _dataset(n=64, seed=1)

    full_session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=11
    )
    full_loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=full_session.data_loader_rng)
    full_session.trainer.fit(full_loader, epochs=5)

    ckpt_path = tmp_path / "partial.forge"
    partial_session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=11
    )
    partial_loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=partial_session.data_loader_rng)
    partial_session.trainer.fit(partial_loader, epochs=3)
    partial_session.save_checkpoint(str(ckpt_path))

    resumed_session = start_training_session(
        build_model=_build_model,
        build_optimizer=_build_optimizer,
        loss_fn=MSELoss(),
        seed=11,
        resume=str(ckpt_path),
    )
    resumed_loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=resumed_session.data_loader_rng)
    resumed_session.trainer.fit(resumed_loader, epochs=2)

    full_params = _params(full_session.trainer.model)
    resumed_params = _params(resumed_session.trainer.model)
    for name in full_params:
        np.testing.assert_allclose(full_params[name], resumed_params[name], atol=1e-6)


def test_resume_onto_cuda_converts_a_cpu_saved_checkpoint(tmp_path):
    """start_training_session() passes device= to load_checkpoint() itself,
    so resuming onto a different device than the checkpoint was saved from
    converts it (load_checkpoint's own documented behavior) rather than
    raising -- there is no device-mismatch failure mode reachable through
    this API, unlike constructing a Trainer around an already-loaded,
    un-converted checkpoint by hand.
    """
    path = tmp_path / "ckpt.forge"
    session = start_training_session(
        build_model=_build_model, build_optimizer=_build_optimizer, loss_fn=MSELoss(), seed=0, device="cpu"
    )
    session.save_checkpoint(str(path))

    from forge.backend.cuda import is_cuda_available

    if not is_cuda_available():
        pytest.skip("CUDA is not available on this machine")

    resumed = start_training_session(
        build_model=_build_model,
        build_optimizer=_build_optimizer,
        loss_fn=MSELoss(),
        seed=0,
        device="cuda",
        resume=str(path),
    )
    assert str(resumed.trainer.device) == "cuda"
    assert str(resumed.trainer.model.device) == "cuda"
