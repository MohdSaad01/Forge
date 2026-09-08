"""Milestone 62 tests: `nn.Conv1d` (CPU).

`Conv1d` is implemented entirely by reshaping to a dummy `(N, C, 1, L)` 4D
tensor and dispatching to the existing, already-tested `Tensor.conv2d`
machinery (see `forge/nn/conv.py`'s `Conv1d` docstring) -- no new `Backend`
method. These tests therefore focus on: configuration validation, parameter
shapes/initialization, forward correctness against an independent reference
implementation (not `Conv2d`, so a bug shared between the two would still be
caught), shape validation errors, gradient accumulation, and finite-difference
gradient checks. See `tests/test_cuda_conv1d.py` for the CUDA counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import Conv1d
from forge.optim import SGD
from forge.serialization import load_model, save_model

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-2, atol=1e-2)


def reference_conv1d(x: np.ndarray, w: np.ndarray, b, stride: int, padding: int) -> np.ndarray:
    """Independent (triple-loop) ground truth, deliberately not sharing code with Conv2d."""
    N, Cin, L = x.shape
    Cout, _, K = w.shape
    L_out = (L + 2 * padding - K) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    out = np.zeros((N, Cout, L_out), dtype=np.float64)
    for n in range(N):
        for co in range(Cout):
            for lo in range(L_out):
                li0 = lo * stride
                patch = padded[n, :, li0 : li0 + K]
                out[n, co, lo] = np.sum(patch * w[co]) + (b[co] if b is not None else 0.0)
    return out


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
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


# -- configuration validation -------------------------------------------------


def test_conv1d_rejects_non_positive_channels():
    with pytest.raises(ShapeMismatchError):
        Conv1d(0, 4, kernel_size=3)
    with pytest.raises(ShapeMismatchError):
        Conv1d(3, -1, kernel_size=3)


@pytest.mark.parametrize("kernel_size", [0, -1, 3.0, "3", (3,)])
def test_conv1d_rejects_invalid_kernel_size(kernel_size):
    with pytest.raises(ShapeMismatchError):
        Conv1d(1, 2, kernel_size=kernel_size)


@pytest.mark.parametrize("stride", [0, -1, 1.5])
def test_conv1d_rejects_invalid_stride(stride):
    with pytest.raises(ShapeMismatchError):
        Conv1d(1, 2, kernel_size=3, stride=stride)


@pytest.mark.parametrize("padding", [-1, -2])
def test_conv1d_rejects_invalid_padding(padding):
    with pytest.raises(ShapeMismatchError):
        Conv1d(1, 2, kernel_size=3, padding=padding)


# -- parameter shapes / initialization ----------------------------------------


def test_conv1d_parameter_shapes():
    layer = Conv1d(3, 8, kernel_size=5)
    assert layer.weight.shape == (8, 3, 5)
    assert layer.bias.shape == (8,)


def test_conv1d_without_bias_has_no_bias_parameter():
    layer = Conv1d(3, 8, kernel_size=3, bias=False)
    assert layer.bias is None
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"weight"}


def test_conv1d_init_bound_matches_fan_in():
    forge.random.seed(0)
    k, cin = 5, 4
    layer = Conv1d(cin, 6, kernel_size=k)
    bound = 1.0 / np.sqrt(cin * k)
    assert np.all(np.abs(layer.weight.numpy()) <= bound)
    assert np.all(np.abs(layer.bias.numpy()) <= bound)


def test_conv1d_construction_is_deterministic_under_seed():
    forge.random.seed(7)
    a = Conv1d(2, 3, kernel_size=3)
    forge.random.seed(7)
    b = Conv1d(2, 3, kernel_size=3)
    np.testing.assert_array_equal(a.weight.numpy(), b.weight.numpy())
    np.testing.assert_array_equal(a.bias.numpy(), b.bias.numpy())


# -- output shape ---------------------------------------------------------------


@pytest.mark.parametrize(
    "L,kernel_size,stride,padding,expected",
    [
        (10, 3, 1, 0, 8),
        (10, 3, 1, 1, 10),
        (10, 3, 2, 1, 5),
        (9, 5, 1, 0, 5),
    ],
)
def test_conv1d_output_shape_formula(L, kernel_size, stride, padding, expected):
    layer = Conv1d(2, 4, kernel_size=kernel_size, stride=stride, padding=padding)
    x = Tensor(np.zeros((3, 2, L)))
    y = layer(x)
    assert y.shape == (3, 4, expected)


# -- forward correctness vs. an independent reference -------------------------


def _fixed_conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    layer = Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    w = np.linspace(-1, 1, out_channels * in_channels * kernel_size).reshape(out_channels, in_channels, kernel_size)
    layer.weight = type(layer.weight)(w)
    if bias:
        b = np.linspace(0.1, 0.5, out_channels)
        layer.bias = type(layer.bias)(b)
    return layer


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1), (2, 0)])
def test_conv1d_forward_matches_reference(stride, padding):
    layer = _fixed_conv1d(3, 2, kernel_size=3, stride=stride, padding=padding)
    x_data = np.random.default_rng(0).standard_normal((2, 3, 12))
    y = layer(Tensor(x_data))
    expected = reference_conv1d(x_data, layer.weight.numpy(), layer.bias.numpy(), layer.stride, layer.padding)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


def test_conv1d_forward_without_bias_matches_reference():
    layer = _fixed_conv1d(2, 3, kernel_size=3, bias=False)
    x_data = np.random.default_rng(1).standard_normal((2, 2, 8))
    y = layer(Tensor(x_data))
    expected = reference_conv1d(x_data, layer.weight.numpy(), None, layer.stride, layer.padding)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


def test_conv1d_forward_single_sample_batch_of_one():
    layer = _fixed_conv1d(1, 2, kernel_size=3)
    x_data = np.random.default_rng(2).standard_normal((1, 1, 7))
    y = layer(Tensor(x_data))
    expected = reference_conv1d(x_data, layer.weight.numpy(), layer.bias.numpy(), layer.stride, layer.padding)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


# -- runtime shape validation ---------------------------------------------------


def test_conv1d_rejects_wrong_input_ndim():
    layer = Conv1d(1, 2, kernel_size=3)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 4))))
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 1, 4, 4))))


def test_conv1d_rejects_channel_mismatch():
    layer = Conv1d(3, 2, kernel_size=3)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((1, 2, 8))))


def test_conv1d_rejects_kernel_larger_than_padded_input():
    layer = Conv1d(1, 2, kernel_size=5, padding=0)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((1, 1, 4))))


# -- gradient accumulation ------------------------------------------------------


def test_conv1d_weight_used_once_receives_gradient():
    layer = _fixed_conv1d(2, 3, kernel_size=3)
    x = Tensor(np.random.default_rng(3).standard_normal((2, 2, 8)))
    layer(x).sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad is not None


def test_conv1d_weight_used_multiple_times_accumulates():
    layer = _fixed_conv1d(2, 3, kernel_size=3, bias=False)
    x1 = Tensor(np.random.default_rng(4).standard_normal((1, 2, 8)))
    x2 = Tensor(np.random.default_rng(5).standard_normal((1, 2, 8)))

    combined = layer(x1).sum() + layer(x2).sum()
    combined.backward()
    combined_grad = layer.weight.grad.numpy().copy()

    layer.weight.zero_grad()
    layer(x1).sum().backward()
    grad1 = layer.weight.grad.numpy().copy()
    layer.weight.zero_grad()
    layer(x2).sum().backward()
    grad2 = layer.weight.grad.numpy().copy()

    np.testing.assert_allclose(combined_grad, grad1 + grad2, **TOL)


def test_conv1d_integrates_with_sgd():
    layer = Conv1d(2, 3, kernel_size=3)
    x = Tensor(np.random.default_rng(6).standard_normal((2, 2, 8)))
    before = layer.weight.numpy().copy()
    opt = SGD(layer.parameters(), lr=0.1)
    layer(x).sum().backward()
    opt.step()
    assert not np.allclose(before, layer.weight.numpy())


# -- finite-difference gradient checks ------------------------------------------


@pytest.mark.parametrize(
    "N,Cin,Cout,L,kernel_size,stride,padding",
    [
        (1, 1, 1, 6, 3, 1, 0),
        (1, 2, 3, 6, 3, 1, 1),
        (2, 2, 2, 8, 3, 2, 1),
        (2, 3, 2, 9, 5, 2, 0),
    ],
)
def test_conv1d_finite_difference_input_weight_bias(N, Cin, Cout, L, kernel_size, stride, padding):
    forge.random.seed(0)
    rng = np.random.default_rng(42)
    layer = Conv1d(Cin, Cout, kernel_size, stride=stride, padding=padding)
    x_data = rng.standard_normal((N, Cin, L))
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    def _forward(xd, wd, bd):
        # Mirrors `Conv1d.forward`'s reshape-then-`conv2d` composition directly
        # (rather than constructing a fresh `Conv1d`, which would draw new
        # random init weights this test immediately has to overwrite).
        n, cin, length = xd.shape
        x4 = Tensor(xd).reshape(n, cin, 1, length)
        w4 = Tensor(wd).reshape(Cout, Cin, 1, kernel_size)
        y4 = x4.conv2d(w4, Tensor(bd), (1, stride), (0, padding))
        return y4

    def loss_x(xd):
        return float((_forward(xd, w_data, b_data).numpy() ** 2).sum())

    def loss_w(wd):
        return float((_forward(x_data, wd, b_data).numpy() ** 2).sum())

    def loss_b(bd):
        return float((_forward(x_data, w_data, bd).numpy() ** 2).sum())

    x = Tensor(x_data.copy(), requires_grad=True)
    layer.weight = type(layer.weight)(w_data.copy())
    layer.bias = type(layer.bias)(b_data.copy())
    out = layer(x)
    (out * out).sum().backward()

    np.testing.assert_allclose(x.grad.numpy(), numerical_grad(loss_x, x_data.copy()), **FD_TOL)
    np.testing.assert_allclose(layer.weight.grad.numpy(), numerical_grad(loss_w, w_data.copy()), **FD_TOL)
    np.testing.assert_allclose(layer.bias.grad.numpy(), numerical_grad(loss_b, b_data.copy()), **FD_TOL)


# -- serialization ----------------------------------------------------------------


def test_conv1d_serialization_round_trip_preserves_predictions(tmp_path):
    forge.random.seed(0)
    layer = Conv1d(2, 4, kernel_size=3, padding=1)
    x = Tensor(np.random.default_rng(9).standard_normal((3, 2, 10)).astype(np.float32))
    with forge.no_grad():
        pre_save = layer(x).numpy()

    path = tmp_path / "conv1d.forge"
    save_model(layer, str(path))
    reloaded = load_model(str(path))
    with forge.no_grad():
        post_load = reloaded(x).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)
    assert reloaded.kernel_size == layer.kernel_size
    assert reloaded.stride == layer.stride
    assert reloaded.padding == layer.padding
