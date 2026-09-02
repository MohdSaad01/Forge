"""CUDA `Trainer`/loss integration tests (Milestone 12).

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py`/`tests/test_cuda_autograd.py`/
`tests/test_module_cuda.py`/`tests/test_cuda_loss.py`. These prove the full

    Dataset -> CPU DataLoader -> Trainer(device="cuda") -> explicit batch
    transfer -> CUDA Module -> CUDA Loss -> CUDA autograd -> CUDA SGD

path documented in `docs/architecture/training-engine.md`'s **Device
semantics** section: `DataLoader` stays CPU-side, `Trainer` validates (never
moves) the model and explicitly transfers each batch, and no `CPUBackend`
compute method is ever invoked along the way.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.data import DataLoader, TensorDataset
from forge.exceptions import CUDAError, TrainerError, UnsupportedDeviceError
from forge.nn import Linear, MSELoss
from forge.optim import SGD
from forge.training import MeanAbsoluteError, MeanSquaredError, Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-3, atol=1e-3)


def _regression_loader(n=32, batch_size=8, shuffle=False, seed=42):
    """A plain CPU `DataLoader` -- `Trainer` is the only thing that ever moves this to CUDA."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _matched_linear_models():
    """Two `Linear(2, 1)`s with numerically identical parameters, one about to move to CUDA."""
    forge.random.seed(123)
    cpu_model = Linear(2, 1)
    cuda_model = Linear(2, 1)
    for (_, p_cpu), (_, p_cuda) in zip(cpu_model.named_parameters(), cuda_model.named_parameters()):
        p_cuda._data = np.array(p_cpu._data, copy=True)
    return cpu_model, cuda_model


# -- Trainer CUDA configuration ------------------------------------------------


def test_cuda_trainer_valid_construction():
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda")
    assert str(trainer.device) == "cuda"
    assert trainer.model is model


def test_cuda_trainer_construction_does_not_require_model_on_cuda():
    """Construction validates the *backend*, not the model's placement -- model
    placement is validated lazily at fit()/evaluate() time (see below)."""
    model = Linear(2, 1)  # still CPU-resident
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda")
    assert str(trainer.device) == "cuda"


def test_cuda_trainer_fit_rejects_cpu_model():
    model = Linear(2, 1)  # never moved to cuda
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False
    )
    with pytest.raises(UnsupportedDeviceError):
        trainer.fit(_regression_loader(), epochs=1)


def test_cuda_trainer_evaluate_rejects_cpu_model():
    model = Linear(2, 1)
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False
    )
    with pytest.raises(UnsupportedDeviceError):
        trainer.evaluate(_regression_loader())


def test_cuda_trainer_does_not_move_the_model():
    """Trainer's policy is validate-not-move: constructing/using a CUDA Trainer
    with a CPU model must never mutate that model's device as a side effect."""
    model = Linear(2, 1)
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False
    )
    with pytest.raises(UnsupportedDeviceError):
        trainer.fit(_regression_loader(), epochs=1)
    assert model.device.type == "cpu"


def test_cuda_trainer_construction_rejects_unknown_device_string():
    model = Linear(2, 1)
    with pytest.raises(UnsupportedDeviceError):
        Trainer(model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="tpu")


# -- Batch movement -------------------------------------------------------------


def test_cuda_trainer_moves_cpu_batches_to_cuda():
    """DataLoader yields CPU tensors; Trainer explicitly transfers them before the
    forward pass -- observed here by recording the Tensor the model actually saw."""
    model = Linear(2, 1).to("cuda")
    seen = {}

    real_forward = Linear.forward

    def spying_forward(self, x):
        seen["x"] = x
        return real_forward(self, x)

    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False)
    loader = _regression_loader()
    for batch in loader:
        x, y = batch
        assert x.device.type == "cpu"  # DataLoader itself never produces CUDA tensors
        break

    Linear.forward = spying_forward
    try:
        trainer.fit(loader, epochs=1)
    finally:
        Linear.forward = real_forward

    assert seen["x"].device.type == "cuda"
    assert isinstance(seen["x"]._data, CUDAStorage)


def test_dataloader_itself_never_produces_cuda_tensors():
    """No implicit GPU behavior in DataLoader -- every batch it yields, on its own
    (never passed through a Trainer), stays on CPU regardless of Trainer's device."""
    loader = _regression_loader()
    for x, y in loader:
        assert x.device.type == "cpu"
        assert y.device.type == "cpu"


# -- CUDA Trainer training / evaluation lifecycle -----------------------------


def test_cuda_trainer_fit_returns_history_with_cuda_device():
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1),
        device="cuda", metrics=[MeanSquaredError()], verbose=False,
    )
    history = trainer.fit(_regression_loader(), epochs=3)
    assert len(history) == 3
    for record in history:
        assert record.device == "cuda"
        assert "mse" in record.train_metrics
        assert np.isfinite(record.train_loss)


def test_cuda_trainer_fit_with_validation_loader():
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1),
        device="cuda", metrics=[MeanSquaredError(), MeanAbsoluteError()], verbose=False,
    )
    train_loader = _regression_loader(n=32, seed=1)
    val_loader = _regression_loader(n=16, seed=2)
    history = trainer.fit(train_loader, epochs=3, validation_loader=val_loader)
    for record in history:
        assert record.val_loss is not None
        assert np.isfinite(record.val_loss)
        assert "mse" in record.val_metrics and "mae" in record.val_metrics


def test_cuda_trainer_evaluate_standalone():
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1),
        device="cuda", metrics=[MeanSquaredError()], verbose=False,
    )
    result = trainer.evaluate(_regression_loader())
    assert result.device == "cuda"
    assert np.isfinite(result.loss)
    assert "mse" in result.metrics
    assert result.samples == 32


def test_cuda_trainer_evaluate_does_not_change_parameters():
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False
    )
    before = {name: p.to("cpu").numpy().copy() for name, p in model.named_parameters()}
    trainer.evaluate(_regression_loader())
    after = {name: p.to("cpu").numpy().copy() for name, p in model.named_parameters()}
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_cuda_trainer_evaluate_builds_no_graph():
    """`evaluate()` runs the forward pass inside `no_grad()` -- the prediction it
    computes must carry no CUDA graph, even though the model's Parameters
    still have `requires_grad=True`."""

    class RecordingLinear(Linear):
        last_output = None

        def forward(self, x):
            out = super().forward(x)
            RecordingLinear.last_output = out
            return out

    model = RecordingLinear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False
    )
    trainer.evaluate(_regression_loader())
    assert RecordingLinear.last_output is not None
    assert RecordingLinear.last_output.requires_grad is False
    assert RecordingLinear.last_output.grad_fn is None


def test_cuda_trainer_evaluate_restores_training_mode():
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False
    )
    assert model.training is True
    trainer.evaluate(_regression_loader())
    assert model.training is True


def test_cuda_trainer_fit_rejects_unsupported_batch_structure():
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1), device="cuda", verbose=False
    )

    class BadLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            yield Tensor([[1.0, 2.0]])  # not a (features, target) tuple

    with pytest.raises(TrainerError):
        trainer.fit(BadLoader(), epochs=1)


# -- CUDA CrossEntropyLoss is a clean, explicit failure ------------------------


def test_cuda_trainer_fit_with_cross_entropy_fails_clearly():
    from forge.nn import CrossEntropyLoss

    model = Linear(2, 2).to("cuda")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 2)).astype(np.float32)
    targets = rng.integers(0, 2, size=16)
    loader = DataLoader(TensorDataset(Tensor(X), Tensor(targets)), batch_size=4)

    trainer = Trainer(
        model=model, loss_fn=CrossEntropyLoss(), optimizer=SGD(model.parameters(), lr=0.1),
        device="cuda", verbose=False,
    )
    with pytest.raises(forge.LossError, match="CPU-only"):
        trainer.fit(loader, epochs=1)


# -- No CPU fallback ------------------------------------------------------------


def test_cuda_trainer_fit_never_calls_cpu_backend_compute_ops(monkeypatch):
    model = Linear(2, 1).to("cuda")
    trainer = Trainer(
        model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.1),
        device="cuda", metrics=[MeanSquaredError()], verbose=False,
    )

    calls: list[str] = []
    compute_ops = ("add", "sub", "mul", "matmul", "sum", "reshape", "relu", "exp", "log", "sgd_step")
    for name in compute_ops:
        original = getattr(CPUBackend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    train_loader = _regression_loader(n=16, seed=3)
    val_loader = _regression_loader(n=8, seed=4)
    trainer.fit(train_loader, epochs=2, validation_loader=val_loader)

    assert calls == []


# -- End-to-end CUDA regression: demonstrates real learning --------------------


def test_cuda_trainer_end_to_end_regression_learns():
    forge.random.seed(7)
    model = Linear(2, 1).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.3)
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=optimizer, device="cuda", verbose=False)

    loader = _regression_loader(n=64, batch_size=16, seed=11)
    history = trainer.fit(loader, epochs=60)

    assert history.train_losses[0] > history.train_losses[-1] * 10  # loss drops substantially
    assert history.train_losses[-1] < 0.05

    weight, bias = model.weight.to("cpu").numpy(), model.bias.to("cpu").numpy()
    np.testing.assert_allclose(weight.flatten(), [3.0, -2.0], atol=0.1)
    np.testing.assert_allclose(bias.flatten(), [1.0], atol=0.1)

    for name, param in model.named_parameters():
        assert param.device.type == "cuda", name
        assert param.grad is not None, name
        assert param.grad.device.type == "cuda", name


# -- CPU/CUDA training consistency ---------------------------------------------


def test_cpu_cuda_trainer_training_consistency():
    cpu_model, cuda_model = _matched_linear_models()
    cuda_model.to("cuda")

    cpu_trainer = Trainer(
        model=cpu_model, loss_fn=MSELoss(), optimizer=SGD(cpu_model.parameters(), lr=0.2), device="cpu", verbose=False
    )
    cuda_trainer = Trainer(
        model=cuda_model, loss_fn=MSELoss(), optimizer=SGD(cuda_model.parameters(), lr=0.2), device="cuda", verbose=False
    )

    cpu_history = cpu_trainer.fit(_regression_loader(n=32, batch_size=8, seed=99), epochs=15)
    cuda_history = cuda_trainer.fit(_regression_loader(n=32, batch_size=8, seed=99), epochs=15)

    np.testing.assert_allclose(cpu_history.train_losses, cuda_history.train_losses, **TOL)
    for (_, p_cpu), (_, p_cuda) in zip(cpu_model.named_parameters(), cuda_model.named_parameters()):
        np.testing.assert_allclose(p_cuda.to("cpu").numpy(), p_cpu.numpy(), **TOL)
