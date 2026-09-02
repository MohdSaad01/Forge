"""Milestone 17 tests: CUDA Adam.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py`/`tests/test_cuda_persistence.py`.
CPU-only Adam behavior (reference-value correctness, hyperparameter
validation, state bookkeeping, Trainer integration, persistence isolation)
lives in `tests/test_optimizer.py`. See `docs/architecture/optimization.md`
and `docs/architecture/cuda-backend.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.exceptions import OptimizerError
from forge.nn import Linear, Parameter
from forge.optim import Adam

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


# -- CUDA state is real CUDA storage -----------------------------------------


def test_adam_cuda_state_is_cuda_resident():
    p = Parameter([1.0, 2.0], device="cuda")
    p.grad = Tensor([0.1, -0.2], device="cuda")
    opt = Adam([p], lr=0.1)
    opt.step()

    state = opt.state[p]
    assert isinstance(state.m, CUDAStorage)
    assert isinstance(state.v, CUDAStorage)
    assert not isinstance(state.m, np.ndarray)
    assert not isinstance(state.v, np.ndarray)
    assert state.device.type == "cuda"


def test_adam_cuda_parameter_updates_happen_on_cuda():
    p = Parameter([1.0, 2.0], device="cuda")
    p.grad = Tensor([0.1, -0.2], device="cuda")
    opt = Adam([p], lr=0.1)
    opt.step()
    assert p.device.type == "cuda"
    assert isinstance(p._data, CUDAStorage)


# -- CPU/CUDA numerical agreement --------------------------------------------


def test_adam_cpu_cuda_agree_single_step():
    theta0 = [1.0, -2.0, 0.5, 3.0]
    grad0 = [0.1, 0.2, -0.1, 0.05]

    p_cpu = Parameter(theta0, device="cpu")
    p_cuda = Parameter(theta0, device="cuda")
    p_cpu.grad = Tensor(grad0, device="cpu")
    p_cuda.grad = Tensor(grad0, device="cuda")

    Adam([p_cpu], lr=0.05).step()
    Adam([p_cuda], lr=0.05).step()

    np.testing.assert_allclose(p_cpu.numpy(), p_cuda.to("cpu").numpy(), **TOL)


def test_adam_cpu_cuda_agree_multiple_steps_with_weight_decay():
    rng = np.random.default_rng(0)
    theta0 = rng.standard_normal((4, 3)).astype(np.float32).tolist()

    p_cpu = Parameter(theta0, device="cpu")
    p_cuda = Parameter(theta0, device="cuda")
    opt_cpu = Adam([p_cpu], lr=0.05, weight_decay=0.01)
    opt_cuda = Adam([p_cuda], lr=0.05, weight_decay=0.01)

    for _ in range(8):
        g = rng.standard_normal((4, 3)).astype(np.float32).tolist()
        p_cpu.grad = Tensor(g, device="cpu")
        p_cuda.grad = Tensor(g, device="cuda")
        opt_cpu.step()
        opt_cuda.step()

    np.testing.assert_allclose(p_cpu.numpy(), p_cuda.to("cpu").numpy(), **TOL)


def test_adam_cpu_cuda_agree_on_real_model():
    forge.random.seed(0)
    layer_cpu = Linear(4, 3)
    layer_cuda = Linear(4, 3)
    # Force identical initial weights across devices.
    w0 = layer_cpu.weight.numpy().copy()
    b0 = layer_cpu.bias.numpy().copy()
    layer_cuda.weight = Parameter(w0.tolist(), device="cuda")
    layer_cuda.bias = Parameter(b0.tolist(), device="cuda")

    opt_cpu = Adam(layer_cpu.parameters(), lr=0.05)
    opt_cuda = Adam(layer_cuda.parameters(), lr=0.05)

    x_np = np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)
    x_cpu = Tensor(x_np, device="cpu")
    x_cuda = Tensor(x_np, device="cuda")

    for _ in range(5):
        opt_cpu.zero_grad()
        opt_cuda.zero_grad()
        y_cpu = layer_cpu(x_cpu).sum()
        y_cuda = layer_cuda(x_cuda).sum()
        y_cpu.backward()
        y_cuda.backward()
        opt_cpu.step()
        opt_cuda.step()

    np.testing.assert_allclose(layer_cpu.weight.numpy(), layer_cuda.weight.to("cpu").numpy(), **TOL)
    np.testing.assert_allclose(layer_cpu.bias.numpy(), layer_cuda.bias.to("cpu").numpy(), **TOL)


# -- no CPU fallback -----------------------------------------------------------


def test_adam_cuda_grad_and_param_never_leave_device_mid_step(monkeypatch):
    """Structural proof: `CPUBackend.adam_step` is never invoked for a CUDA
    parameter -- the CUDA Adam path must dispatch to `CUDABackend.adam_step`
    exclusively, never fall back to NumPy arithmetic."""
    from forge.backend.cpu import CPUBackend

    calls = []
    original = CPUBackend.adam_step

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CPUBackend, "adam_step", spy)

    p = Parameter([1.0, 2.0], device="cuda")
    p.grad = Tensor([0.1, -0.2], device="cuda")
    opt = Adam([p], lr=0.1)
    opt.step()

    assert calls == []


def test_adam_cuda_state_never_constructed_as_numpy():
    p = Parameter([1.0, 2.0, 3.0], device="cuda")
    p.grad = Tensor([0.1, 0.2, 0.3], device="cuda")
    opt = Adam([p], lr=0.1)
    opt.step()
    state = opt.state[p]
    assert type(state.m).__name__ == "CUDAStorage"
    assert type(state.v).__name__ == "CUDAStorage"


# -- device transfer after optimizer state exists (Policy A) -----------------


def test_adam_state_device_mismatch_after_module_to_raises():
    layer = Linear(2, 2)  # starts on cpu
    opt = Adam(layer.parameters(), lr=0.01)
    x_cpu = Tensor([[1.0, 2.0]])
    opt.zero_grad()
    layer(x_cpu).sum().backward()
    opt.step()
    assert opt.state[layer.weight].device.type == "cpu"

    layer.to("cuda")
    x_cuda = Tensor([[1.0, 2.0]], device="cuda")
    opt.zero_grad()
    layer(x_cuda).sum().backward()

    with pytest.raises(OptimizerError, match="device"):
        opt.step()


def test_adam_state_cleared_after_move_reinitializes_on_new_device():
    layer = Linear(2, 2)
    opt = Adam(layer.parameters(), lr=0.01)
    x_cpu = Tensor([[1.0, 2.0]])
    opt.zero_grad()
    layer(x_cpu).sum().backward()
    opt.step()

    layer.to("cuda")
    opt.state.clear()

    x_cuda = Tensor([[1.0, 2.0]], device="cuda")
    opt.zero_grad()
    layer(x_cuda).sum().backward()
    opt.step()  # should not raise

    assert opt.state[layer.weight].device.type == "cuda"
    assert opt.state[layer.weight].step == 1
    assert isinstance(opt.state[layer.weight].m, CUDAStorage)


# -- end-to-end CUDA training ---------------------------------------------------


def test_adam_end_to_end_cuda_training_decreases_loss():
    from forge.nn.loss import MSELoss

    forge.random.seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(32, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)

    model = Linear(2, 1)
    model.to("cuda")
    loss_fn = MSELoss()
    opt = Adam(model.parameters(), lr=0.05)

    x_t = Tensor(X, device="cuda")
    y_t = Tensor(y, device="cuda")

    losses = []
    for _ in range(50):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.to("cpu").numpy()))

    assert losses[-1] < losses[0]
    assert model.weight.device.type == "cuda"
    assert isinstance(model.weight._data, CUDAStorage)
    assert isinstance(opt.state[model.weight].m, CUDAStorage)
    assert isinstance(model.weight.grad._data, CUDAStorage)


def test_trainer_works_with_adam_on_cuda():
    from forge.data import DataLoader, TensorDataset
    from forge.nn.loss import MSELoss
    from forge.training import Trainer

    forge.random.seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(16, 2)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1).astype(np.float32)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=4)

    model = Linear(2, 1)
    model.to("cuda")
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=optimizer, device="cuda", verbose=False)

    history = trainer.fit(loader, epochs=5)
    assert history.train_losses[-1] < history.train_losses[0]
