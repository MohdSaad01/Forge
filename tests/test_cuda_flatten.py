"""Milestone 16 CUDA test: `Flatten` real CUDA execution.

`Flatten` has no backend code of its own -- it composes `Tensor.reshape`,
already a real CUDA operation (a device-to-device copy, `CUDABackend.reshape`)
with no CPU fallback. These tests confirm that composition holds on real
hardware: shape/dtype correctness, autograd, and no `CPUBackend` involvement.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import Flatten, Linear, ReLU, Sequential

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


def test_cuda_flatten_forward_matches_cpu():
    x_data = np.random.default_rng(0).standard_normal((4, 3, 5, 6)).astype(np.float32)
    cpu_out = Flatten()(Tensor(x_data.copy())).numpy()
    cuda_out = Flatten()(Tensor(x_data.copy(), device="cuda"))

    assert cuda_out.device.type == "cuda"
    assert isinstance(cuda_out._data, CUDAStorage)
    np.testing.assert_array_equal(cuda_out.to("cpu").numpy(), cpu_out)


def test_cuda_flatten_backward_matches_cpu():
    x_data = np.random.default_rng(1).standard_normal((2, 3, 4)).astype(np.float32)
    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)

    Flatten()(x_cpu).sum().backward()
    Flatten()(x_cuda).sum().backward()

    assert isinstance(x_cuda.grad._data, CUDAStorage)
    np.testing.assert_array_equal(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy())


def test_cuda_flatten_never_calls_cpu_backend(monkeypatch):
    x = Tensor(
        np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32),
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

    Flatten()(x).sum().backward()
    assert calls == []


def test_cuda_sequential_with_flatten_and_linear_runs_end_to_end():
    forge.random.seed(0)
    model = Sequential(Flatten(), Linear(12, 4), ReLU()).to("cuda")
    x = Tensor(np.random.default_rng(3).standard_normal((3, 3, 4)).astype(np.float32), device="cuda")
    out = model(x)
    assert out.device.type == "cuda"
    assert out.shape == (3, 4)
