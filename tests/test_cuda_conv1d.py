"""Milestone 62 CUDA tests: `nn.Conv1d`/`nn.MaxPool1d`.

`Conv1d`/`MaxPool1d` are implemented entirely by reshaping to a dummy 4D
tensor and dispatching to `Conv2d`/`MaxPool2d`'s existing, already
CUDA-hardware-verified machinery (see `forge/nn/conv.py`'s `Conv1d`
docstring) -- no new `Backend` method, no new CUDA kernel. These tests
therefore focus on what is actually new: that the reshape-and-reuse
composition itself produces real `CUDAStorage` end to end (never a silent
CPU round trip) and that its forward/backward numerically agree with the
CPU path, plus device movement and an end-to-end tiny classification model,
mirroring `tests/test_cuda_conv.py`'s structure. Every test requires an
actual working CUDA backend and is skipped cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend import get_backend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import Conv1d, Flatten, Linear, MaxPool1d, Module, ReLU, Sequential
from forge.optim import SGD

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


def _to_cuda_conv1d(layer_cpu: Conv1d) -> Conv1d:
    """A CUDA `Conv1d` with numerically identical parameters to `layer_cpu`."""
    layer_cuda = Conv1d(
        layer_cpu.in_channels, layer_cpu.out_channels, layer_cpu.kernel_size,
        stride=layer_cpu.stride, padding=layer_cpu.padding, bias=layer_cpu.bias is not None,
        device="cuda",
    )
    backend = get_backend(layer_cuda.weight.device)
    layer_cuda.weight._data = backend.from_array(layer_cpu.weight.numpy().copy(), layer_cpu.weight.numpy().dtype)
    if layer_cpu.bias is not None:
        layer_cuda.bias._data = backend.from_array(layer_cpu.bias.numpy().copy(), layer_cpu.bias.numpy().dtype)
    return layer_cuda


# -- forward/backward parity with CPU ------------------------------------------


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1), (2, 0)])
def test_conv1d_forward_matches_cpu(stride, padding):
    forge.random.seed(0)
    layer_cpu = Conv1d(3, 4, kernel_size=3, stride=stride, padding=padding)
    layer_cuda = _to_cuda_conv1d(layer_cpu)

    x0 = np.random.default_rng(1).standard_normal((2, 3, 12)).astype(np.float32)
    y_cpu = layer_cpu(Tensor(x0))
    y_cuda = layer_cuda(Tensor(x0, device="cuda"))

    assert isinstance(y_cuda._data, CUDAStorage)
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


def test_conv1d_backward_matches_cpu_for_input_weight_and_bias():
    forge.random.seed(0)
    layer_cpu = Conv1d(2, 3, kernel_size=3, padding=1)
    layer_cuda = _to_cuda_conv1d(layer_cpu)

    x0 = np.random.default_rng(2).standard_normal((3, 2, 10)).astype(np.float32)
    x_cpu = Tensor(x0, requires_grad=True)
    x_cuda = Tensor(x0, device="cuda", requires_grad=True)

    layer_cpu(x_cpu).sum().backward()
    layer_cuda(x_cuda).sum().backward()

    assert isinstance(x_cuda.grad._data, CUDAStorage)
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(layer_cuda.weight.grad.to("cpu").numpy(), layer_cpu.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(layer_cuda.bias.grad.to("cpu").numpy(), layer_cpu.bias.grad.numpy(), **TOL)


def test_conv1d_without_bias_matches_cpu():
    forge.random.seed(0)
    layer_cpu = Conv1d(2, 3, kernel_size=3, bias=False)
    layer_cuda = _to_cuda_conv1d(layer_cpu)

    x0 = np.random.default_rng(3).standard_normal((2, 2, 9)).astype(np.float32)
    y_cpu = layer_cpu(Tensor(x0))
    y_cuda = layer_cuda(Tensor(x0, device="cuda"))
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


def test_maxpool1d_forward_matches_cpu():
    x0 = np.random.default_rng(4).standard_normal((2, 3, 11)).astype(np.float32)
    layer = MaxPool1d(3, stride=2, padding=1)
    y_cpu = layer(Tensor(x0))
    y_cuda = layer(Tensor(x0, device="cuda"))
    assert isinstance(y_cuda._data, CUDAStorage)
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


def test_maxpool1d_backward_matches_cpu():
    x0 = np.random.default_rng(5).standard_normal((2, 2, 8)).astype(np.float32)
    layer = MaxPool1d(2, stride=1)
    x_cpu = Tensor(x0, requires_grad=True)
    x_cuda = Tensor(x0, device="cuda", requires_grad=True)

    layer(x_cpu).sum().backward()
    layer(x_cuda).sum().backward()

    assert isinstance(x_cuda.grad._data, CUDAStorage)
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)


# -- device movement ------------------------------------------------------------


def test_conv1d_to_cuda_moves_weight_and_bias():
    layer = Conv1d(2, 4, kernel_size=3).to("cuda")
    assert layer.weight.device.type == "cuda"
    assert layer.bias.device.type == "cuda"
    assert isinstance(layer.weight._data, CUDAStorage)


def test_conv1d_to_cpu_moves_a_cuda_layer_back():
    layer = Conv1d(2, 4, kernel_size=3, device="cuda")
    layer.to("cpu")
    assert layer.weight.device.type == "cpu"
    assert isinstance(layer.weight._data, np.ndarray)


# -- end-to-end CUDA classification model: Conv1d -> ReLU -> MaxPool1d -> Linear --


class TinyTCN(Module):
    """Conv1d(1,4,k=3,padding=1) -> ReLU -> MaxPool1d(2) -> Linear, for a length-16 input.

    padding=1 keeps the conv output length at 16; MaxPool1d(2) (stride
    defaults to kernel_size) halves it to 8, so the flattened size feeding
    `Linear` is `4 * 8 = 32`.
    """

    def __init__(self):
        super().__init__()
        self.conv = Conv1d(1, 4, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool1d(2)
        self.fc = Linear(4 * 8, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        n = x.shape[0]
        return self.fc(x.reshape(n, x.shape[1] * x.shape[2]))


def test_tiny_tcn_trains_on_cuda_end_to_end():
    forge.random.seed(0)
    model = TinyTCN().to("cuda")
    x = Tensor(np.random.default_rng(7).standard_normal((5, 1, 16)).astype(np.float32), device="cuda")
    y = model(x)
    assert y.shape == (5, 3)
    assert isinstance(y._data, CUDAStorage)

    opt = SGD(model.parameters(), lr=0.05)
    before = model.conv.weight.to("cpu").numpy().copy()
    y.sum().backward()
    opt.step()
    assert not np.allclose(before, model.conv.weight.to("cpu").numpy())


def test_conv1d_maxpool1d_inside_sequential_on_cuda():
    forge.random.seed(0)
    model = Sequential(
        Conv1d(1, 4, kernel_size=3, padding=1), ReLU(), MaxPool1d(2), Flatten(), Linear(4 * 8, 2),
    ).to("cuda")
    x = Tensor(np.random.default_rng(8).standard_normal((3, 1, 16)).astype(np.float32), device="cuda", requires_grad=True)
    y = model(x)
    assert y.device.type == "cuda"
    y.sum().backward()
    conv = model._modules["0"]
    assert conv.weight.grad is not None
    assert conv.weight.grad.device.type == "cuda"
