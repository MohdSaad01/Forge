"""Milestone 63 CUDA tests: `nn.UpsampleNearest2d` real CUDA kernels.

A new, dedicated forward/backward CUDA kernel pair (`k_upsample_nearest2d_
forward`/`k_upsample_nearest2d_backward`, `forge/backend/cuda/kernels.cu`) --
unlike `Conv1d`/`MaxPool1d` (Milestone 62), this primitive could not be
composed from an existing CUDA op (no general N-D broadcast/repeat exists on
CUDA -- see `base.py`'s `upsample_nearest2d` docstring), so it needed real
kernel work. Mirrors `tests/test_cuda_conv.py`'s `MaxPool2d` section:
forward/backward parity with CPU, tie/overlap-free correctness by
construction, a finite-difference check, device movement, `Sequential`
composition with `Conv2d` (the actual decoder pattern), and a spy-based
"never falls back to CPUBackend" guard. Every test requires an actual
working CUDA backend and is skipped cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import Conv2d, Sequential, UpsampleNearest2d
from forge.optim import SGD

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = fn(x)
        x[idx] = orig - eps
        minus = fn(x)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


# -- forward matches CPU, real CUDA storage ------------------------------------


@pytest.mark.parametrize("scale_factor", [1, 2, 3, (2, 3)])
def test_cuda_upsample_nearest2d_forward_matches_cpu(scale_factor):
    layer = UpsampleNearest2d(scale_factor)
    x_data = np.random.default_rng(8).standard_normal((2, 3, 5, 6)).astype(np.float32)

    y_cpu = layer(Tensor(x_data.copy()))
    y_cuda = layer(Tensor(x_data.copy(), device="cuda"))

    assert y_cuda.device.type == "cuda"
    assert isinstance(y_cuda._data, CUDAStorage)
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


# -- backward matches CPU -------------------------------------------------------


@pytest.mark.parametrize("scale_factor", [2, 3, (2, 3), (3, 1)])
def test_cuda_upsample_nearest2d_backward_matches_cpu(scale_factor):
    layer = UpsampleNearest2d(scale_factor)
    x_data = np.random.default_rng(9).standard_normal((2, 3, 4, 5)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)

    layer(x_cpu).sum().backward()
    layer(x_cuda).sum().backward()

    assert isinstance(x_cuda.grad._data, CUDAStorage)
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)


def test_cuda_upsample_nearest2d_finite_difference():
    layer = UpsampleNearest2d((2, 3))
    rng = np.random.default_rng(10)
    x_data = rng.standard_normal((2, 2, 4, 5))

    def loss(xd):
        return float((layer(Tensor(xd)).numpy() ** 2).sum())

    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    out = layer(x_cuda)
    (out * out).sum().backward()

    np.testing.assert_allclose(
        x_cuda.grad.to("cpu").numpy(), numerical_grad(loss, x_data.copy()), rtol=1e-2, atol=1e-2
    )


def test_cuda_upsample_nearest2d_never_calls_cpu_backend(monkeypatch):
    layer = UpsampleNearest2d(2)
    x_cuda = Tensor(
        np.random.default_rng(11).standard_normal((2, 3, 4, 4)).astype(np.float32),
        device="cuda", requires_grad=True,
    )

    calls: list[str] = []
    for name in dir(CPUBackend):
        if name.startswith("_"):
            continue
        original = getattr(CPUBackend, name)
        if not callable(original):
            continue

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    layer(x_cuda).sum().backward()
    assert calls == []


# -- device movement ------------------------------------------------------------


def test_upsample_nearest2d_has_no_state_to_move():
    # No Parameters/buffers -- `.to("cuda")` on a Sequential containing this
    # layer moves only its Conv2d neighbor; this test documents that the
    # layer itself participates correctly (a no-op) in that traversal.
    model = Sequential(UpsampleNearest2d(2), Conv2d(2, 2, kernel_size=3, padding=1))
    model.to("cuda")
    assert model.device.type == "cuda"
    x = Tensor(np.random.default_rng(5).standard_normal((1, 2, 4, 4)).astype(np.float32), device="cuda")
    y = model(x)
    assert isinstance(y._data, CUDAStorage)
    assert y.shape == (1, 2, 8, 8)


# -- Sequential composition with Conv2d: the actual decoder pattern, trains on CUDA --


def test_cuda_upsample_then_conv2d_trains_and_reduces_loss():
    forge.random.seed(0)
    model = Sequential(
        UpsampleNearest2d(2),
        Conv2d(2, 2, kernel_size=3, padding=1),
    ).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(6).standard_normal((4, 2, 4, 4)).astype(np.float32), device="cuda")
    target = Tensor(np.zeros((4, 2, 8, 8), dtype=np.float32), device="cuda")
    n = 4 * 2 * 8 * 8

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        pred = model(x)
        diff = pred - target
        loss = (diff * diff).sum() * Tensor(1.0 / n, device="cuda")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.to("cpu").numpy()))

    assert losses[-1] < losses[0]
