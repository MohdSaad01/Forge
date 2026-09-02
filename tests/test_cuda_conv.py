"""Milestone 15 CUDA tests: `Conv2d`/`MaxPool2d` real CUDA kernels.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise, matching the convention in
`tests/test_cuda_backend.py`/`tests/test_cuda_autograd.py`. These prove:
CUDA `conv2d`/`max_pool2d` forward and backward execute as real kernels
(structurally distinct from `CPUBackend`, real `CUDAStorage` throughout),
CUDA outputs/gradients agree with CPU within tolerance, a CUDA finite-
difference check passes, and no CUDA `Conv2d`/`MaxPool2d` computation ever
falls back to `CPUBackend` (via a monkeypatch spy on every `CPUBackend`
method).
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.exceptions import CUDAError
from forge.nn import Conv2d, Linear, MaxPool2d, Module, ReLU
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


# -- Conv2d: forward matches CPU, real CUDA storage --------------------------


def test_cuda_conv2d_forward_matches_cpu_and_uses_real_storage():
    forge.random.seed(0)
    layer = Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
    x_data = np.random.default_rng(1).standard_normal((2, 3, 9, 9)).astype(np.float32)

    x_cpu = Tensor(x_data.copy())
    y_cpu = layer(x_cpu)

    cuda_layer = Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
    cuda_layer.weight._data = np.array(layer.weight._data, copy=True)
    cuda_layer.bias._data = np.array(layer.bias._data, copy=True)
    cuda_layer.to("cuda")
    x_cuda = Tensor(x_data.copy(), device="cuda")
    y_cuda = cuda_layer(x_cuda)

    assert y_cuda.device.type == "cuda"
    assert isinstance(y_cuda._data, CUDAStorage)
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


def test_cuda_conv2d_forward_without_bias_matches_cpu():
    forge.random.seed(1)
    layer = Conv2d(2, 3, kernel_size=3, bias=False)
    x_data = np.random.default_rng(2).standard_normal((1, 2, 6, 6)).astype(np.float32)

    y_cpu = layer(Tensor(x_data.copy()))

    cuda_layer = Conv2d(2, 3, kernel_size=3, bias=False)
    cuda_layer.weight._data = np.array(layer.weight._data, copy=True)
    cuda_layer.to("cuda")
    y_cuda = cuda_layer(Tensor(x_data.copy(), device="cuda"))
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


# -- Conv2d: backward matches CPU ---------------------------------------------


def _matched_conv_cpu_cuda(in_ch, out_ch, kernel_size, stride, padding, bias=True):
    forge.random.seed(2)
    cpu_layer = Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
    cuda_layer = Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
    cuda_layer.weight._data = np.array(cpu_layer.weight._data, copy=True)
    if bias:
        cuda_layer.bias._data = np.array(cpu_layer.bias._data, copy=True)
    cuda_layer.to("cuda")
    return cpu_layer, cuda_layer


@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [(3, 1, 0), (3, 1, 1), (3, 2, 1), ((3, 2), (2, 1), (1, 0))],
)
def test_cuda_conv2d_backward_matches_cpu(kernel_size, stride, padding):
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(3, 4, kernel_size, stride, padding)
    x_data = np.random.default_rng(3).standard_normal((2, 3, 8, 8)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)

    cpu_layer(x_cpu).sum().backward()
    cuda_layer(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(cuda_layer.bias.grad.to("cpu").numpy(), cpu_layer.bias.grad.numpy(), **TOL)
    assert isinstance(x_cuda.grad._data, CUDAStorage)
    assert isinstance(cuda_layer.weight.grad._data, CUDAStorage)
    assert isinstance(cuda_layer.bias.grad._data, CUDAStorage)


def test_cuda_conv2d_weight_reuse_accumulates_matching_cpu():
    cpu_layer, cuda_layer = _matched_conv_cpu_cuda(2, 3, 3, 1, 1, bias=False)
    x1 = np.random.default_rng(4).standard_normal((1, 2, 6, 6)).astype(np.float32)
    x2 = np.random.default_rng(5).standard_normal((1, 2, 6, 6)).astype(np.float32)

    (cpu_layer(Tensor(x1)).sum() + cpu_layer(Tensor(x2)).sum()).backward()
    (cuda_layer(Tensor(x1, device="cuda")).sum() + cuda_layer(Tensor(x2, device="cuda")).sum()).backward()

    np.testing.assert_allclose(cuda_layer.weight.grad.to("cpu").numpy(), cpu_layer.weight.grad.numpy(), **TOL)


def test_cuda_conv2d_finite_difference_input():
    forge.random.seed(3)
    layer = Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
    w_data = layer.weight.numpy().astype(np.float64).copy()
    b_data = layer.bias.numpy().astype(np.float64).copy()
    x_data = np.random.default_rng(6).standard_normal((1, 2, 5, 5)).astype(np.float64)

    def loss(xd):
        return float(
            (Tensor(xd).conv2d(Tensor(w_data), Tensor(b_data), layer.stride, layer.padding).numpy() ** 2).sum()
        )

    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    w_cuda = Tensor(w_data.copy(), device="cuda")
    b_cuda = Tensor(b_data.copy(), device="cuda")
    out = x_cuda.conv2d(w_cuda, b_cuda, layer.stride, layer.padding)
    (out * out).sum().backward()

    np.testing.assert_allclose(
        x_cuda.grad.to("cpu").numpy(), numerical_grad(loss, x_data.copy()), rtol=1e-2, atol=1e-2
    )


def test_cuda_conv2d_never_calls_cpu_backend(monkeypatch):
    cuda_layer = Conv2d(3, 4, kernel_size=3, stride=2, padding=1).to("cuda")
    x_cuda = Tensor(np.random.default_rng(7).standard_normal((2, 3, 8, 8)).astype(np.float32), device="cuda")

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

    out = cuda_layer(x_cuda)
    out.sum().backward()

    assert calls == []
    assert isinstance(cuda_layer.weight.grad._data, CUDAStorage)
    assert isinstance(cuda_layer.bias.grad._data, CUDAStorage)


# -- MaxPool2d: forward matches CPU, real CUDA storage ------------------------


@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [(2, 2, 0), (2, 1, 0), (3, 2, 1)],
)
def test_cuda_maxpool2d_forward_matches_cpu(kernel_size, stride, padding):
    layer = MaxPool2d(kernel_size, stride=stride, padding=padding)
    x_data = np.random.default_rng(8).standard_normal((2, 3, 8, 8)).astype(np.float32)

    y_cpu = layer(Tensor(x_data.copy()))
    y_cuda = layer(Tensor(x_data.copy(), device="cuda"))

    assert y_cuda.device.type == "cuda"
    assert isinstance(y_cuda._data, CUDAStorage)
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


# -- MaxPool2d: backward matches CPU, including overlapping windows -----------


@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [(2, 2, 0), (3, 1, 1), (3, 2, 1)],
)
def test_cuda_maxpool2d_backward_matches_cpu(kernel_size, stride, padding):
    layer = MaxPool2d(kernel_size, stride=stride, padding=padding)
    x_data = np.random.default_rng(9).standard_normal((2, 3, 7, 7)).astype(np.float32)

    x_cpu = Tensor(x_data.copy(), requires_grad=True)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)

    layer(x_cpu).sum().backward()
    layer(x_cuda).sum().backward()

    assert isinstance(x_cuda.grad._data, CUDAStorage)
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)


def test_cuda_maxpool2d_tie_break_matches_cpu_convention():
    x_data = np.array([[[[5.0, 5.0], [5.0, 5.0]]]], dtype=np.float32)
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    MaxPool2d(2)(x_cuda).sum().backward()
    expected = np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), expected, **TOL)


def test_cuda_maxpool2d_overlapping_windows_accumulate_matching_cpu():
    x_data = np.zeros((1, 1, 3, 3), dtype=np.float32)
    x_data[0, 0, 1, 1] = 10.0
    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    MaxPool2d(2, stride=1)(x_cuda).sum().backward()
    expected = np.zeros((1, 1, 3, 3), dtype=np.float32)
    expected[0, 0, 1, 1] = 4.0
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), expected, **TOL)


def test_cuda_maxpool2d_finite_difference():
    layer = MaxPool2d(2, stride=1, padding=1)
    rng = np.random.default_rng(10)
    x_data = rng.permutation(2 * 2 * 5 * 5).astype(np.float64).reshape(2, 2, 5, 5)
    x_data = x_data + rng.standard_normal(x_data.shape) * 1e-3

    def loss(xd):
        return float((Tensor(xd).max_pool2d(layer.kernel_size, layer.stride, layer.padding).numpy() ** 2).sum())

    x_cuda = Tensor(x_data.copy(), device="cuda", requires_grad=True)
    out = x_cuda.max_pool2d(layer.kernel_size, layer.stride, layer.padding)
    (out * out).sum().backward()

    np.testing.assert_allclose(
        x_cuda.grad.to("cpu").numpy(), numerical_grad(loss, x_data.copy()), rtol=1e-2, atol=1e-2
    )


def test_cuda_maxpool2d_never_calls_cpu_backend(monkeypatch):
    layer = MaxPool2d(2)
    x_cuda = Tensor(np.random.default_rng(11).standard_normal((2, 3, 6, 6)).astype(np.float32), device="cuda", requires_grad=True)

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


# -- Conv2d validation errors are CUDAError/ShapeMismatchError, not raw crashes --


def test_cuda_conv2d_channel_mismatch_raises_shape_error():
    layer = Conv2d(3, 2, kernel_size=3).to("cuda")
    x = Tensor(np.zeros((1, 2, 8, 8), dtype=np.float32), device="cuda")
    with pytest.raises(forge.ShapeMismatchError):
        layer(x)


def test_cuda_conv2d_unsupported_dtype_raises_cuda_error():
    weight = Tensor(np.zeros((2, 1, 3, 3), dtype=np.int32), device="cuda")
    x = Tensor(np.zeros((1, 1, 5, 5), dtype=np.int32), device="cuda")
    with pytest.raises(CUDAError):
        x.conv2d(weight, None, (1, 1), (0, 0))


# -- End-to-end CUDA classification model: Conv2d -> ReLU -> MaxPool2d -> Linear --


class TinyCNN(Module):
    """Conv2d(1,4,k=3) -> ReLU -> MaxPool2d(2) -> Linear, for an 8x8 input.

    Conv2d(kernel=3, stride=1, padding=0) on 8x8 gives 6x6; MaxPool2d(2)
    (stride defaults to kernel_size) gives 3x3 -- so the flattened size
    feeding `Linear` is `4 * 3 * 3 = 36`, computed once here rather than via
    a lazy shape-inference forward pass.
    """

    def __init__(self):
        super().__init__()
        self.conv = Conv2d(1, 4, kernel_size=3)
        self.relu = ReLU()
        self.pool = MaxPool2d(2)
        self.fc = Linear(4 * 3 * 3, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        n = x.shape[0]
        flat = x.reshape(n, x.shape[1] * x.shape[2] * x.shape[3])
        return self.fc(flat)


def test_cuda_small_cnn_trains_and_reduces_loss():
    forge.random.seed(0)
    rng = np.random.default_rng(0)
    N = 40
    X = np.zeros((N, 1, 8, 8), dtype=np.float32)
    Y = np.zeros((N,), dtype=np.int64)
    for i in range(N):
        label = i % 2
        Y[i] = label
        if label == 0:
            X[i, 0, :4, :] = 1.0 + rng.standard_normal((4, 8)).astype(np.float32) * 0.05
        else:
            X[i, 0, 4:, :] = 1.0 + rng.standard_normal((4, 8)).astype(np.float32) * 0.05

    model = TinyCNN().to("cuda")
    x_cuda_full = Tensor(X, device="cuda")

    loss_fn = forge.nn.CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.1)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        pred = model(x_cuda_full)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        losses.append(loss.to("cpu").numpy().item())

    assert losses[-1] < losses[0] * 0.5
