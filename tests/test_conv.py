"""Milestone 15 tests: `nn.Conv2d` (CPU).

Covers configuration validation, parameter shapes/initialization, forward
correctness against an independent reference implementation, shape
validation errors, gradient accumulation (parameter/input reuse, multiple
graph paths), and finite-difference gradient checks for input/weight/bias
across padding, stride>1, multiple channels, and batch size > 1. See
`tests/test_cuda_conv.py` for the CUDA counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import Conv2d
from forge.optim import SGD

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-2, atol=1e-2)


def reference_conv2d(x: np.ndarray, w: np.ndarray, b, stride, padding) -> np.ndarray:
    """Independent (triple-loop) ground truth, deliberately not sharing code with the backend."""
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    SH, SW = stride
    PH, PW = padding
    H_out = (H + 2 * PH - KH) // SH + 1
    W_out = (W + 2 * PW - KW) // SW + 1
    padded = np.pad(x, ((0, 0), (0, 0), (PH, PH), (PW, PW)))
    out = np.zeros((N, Cout, H_out, W_out), dtype=np.float64)
    for n in range(N):
        for co in range(Cout):
            for ho in range(H_out):
                for wo in range(W_out):
                    hi0, wi0 = ho * SH, wo * SW
                    patch = padded[n, :, hi0 : hi0 + KH, wi0 : wi0 + KW]
                    out[n, co, ho, wo] = np.sum(patch * w[co]) + (b[co] if b is not None else 0.0)
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


def test_conv2d_rejects_non_positive_channels():
    with pytest.raises(ShapeMismatchError):
        Conv2d(0, 4, kernel_size=3)
    with pytest.raises(ShapeMismatchError):
        Conv2d(3, -1, kernel_size=3)


@pytest.mark.parametrize("kernel_size", [0, -1, (0, 3), (3, -2), (3,), "3", 3.0])
def test_conv2d_rejects_invalid_kernel_size(kernel_size):
    with pytest.raises(ShapeMismatchError):
        Conv2d(1, 2, kernel_size=kernel_size)


@pytest.mark.parametrize("stride", [0, -1, (0, 1), (1, -1)])
def test_conv2d_rejects_invalid_stride(stride):
    with pytest.raises(ShapeMismatchError):
        Conv2d(1, 2, kernel_size=3, stride=stride)


@pytest.mark.parametrize("padding", [-1, (-1, 0), (0, -2)])
def test_conv2d_rejects_invalid_padding(padding):
    with pytest.raises(ShapeMismatchError):
        Conv2d(1, 2, kernel_size=3, padding=padding)


def test_conv2d_accepts_rectangular_kernel_size():
    layer = Conv2d(1, 2, kernel_size=(3, 5))
    assert layer.kernel_size == (3, 5)
    assert layer.weight.shape == (2, 1, 3, 5)


# -- parameter shapes / initialization ----------------------------------------


def test_conv2d_parameter_shapes():
    layer = Conv2d(3, 8, kernel_size=3)
    assert layer.weight.shape == (8, 3, 3, 3)
    assert layer.bias.shape == (8,)


def test_conv2d_without_bias_has_no_bias_parameter():
    layer = Conv2d(3, 8, kernel_size=3, bias=False)
    assert layer.bias is None
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"weight"}


def test_conv2d_init_bound_matches_fan_in():
    forge.random.seed(0)
    kh, kw, cin = 3, 3, 4
    layer = Conv2d(cin, 6, kernel_size=(kh, kw))
    bound = 1.0 / np.sqrt(cin * kh * kw)
    assert np.all(np.abs(layer.weight.numpy()) <= bound)
    assert np.all(np.abs(layer.bias.numpy()) <= bound)


def test_conv2d_construction_is_deterministic_under_seed():
    forge.random.seed(7)
    a = Conv2d(2, 3, kernel_size=3)
    forge.random.seed(7)
    b = Conv2d(2, 3, kernel_size=3)
    np.testing.assert_array_equal(a.weight.numpy(), b.weight.numpy())
    np.testing.assert_array_equal(a.bias.numpy(), b.bias.numpy())


# -- output shape ---------------------------------------------------------------


@pytest.mark.parametrize(
    "H,W,kernel_size,stride,padding,expected",
    [
        (8, 8, 3, 1, 0, (6, 6)),
        (8, 8, 3, 1, 1, (8, 8)),
        (8, 8, 3, 2, 1, (4, 4)),
        (7, 9, (3, 5), (2, 1), (1, 2), (4, 9)),
        (5, 5, 5, 1, 0, (1, 1)),
    ],
)
def test_conv2d_output_shape_formula(H, W, kernel_size, stride, padding, expected):
    layer = Conv2d(2, 4, kernel_size=kernel_size, stride=stride, padding=padding)
    x = Tensor(np.zeros((3, 2, H, W)))
    y = layer(x)
    assert y.shape == (3, 4) + expected


# -- forward correctness vs. an independent reference -------------------------


def _fixed_conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    layer = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    kh, kw = layer.kernel_size
    w = np.linspace(-1, 1, out_channels * in_channels * kh * kw).reshape(out_channels, in_channels, kh, kw)
    layer.weight = type(layer.weight)(w)
    if bias:
        b = np.linspace(0.1, 0.5, out_channels)
        layer.bias = type(layer.bias)(b)
    return layer


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1), (2, 0)])
def test_conv2d_forward_matches_reference(stride, padding):
    layer = _fixed_conv(3, 2, kernel_size=3, stride=stride, padding=padding)
    rng = np.random.default_rng(0)
    x_data = rng.standard_normal((2, 3, 9, 9))
    y = layer(Tensor(x_data))
    expected = reference_conv2d(
        x_data, layer.weight.numpy(), layer.bias.numpy(), layer.stride, layer.padding
    )
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


def test_conv2d_forward_without_bias_matches_reference():
    layer = _fixed_conv(2, 3, kernel_size=3, bias=False)
    rng = np.random.default_rng(1)
    x_data = rng.standard_normal((2, 2, 6, 6))
    y = layer(Tensor(x_data))
    expected = reference_conv2d(x_data, layer.weight.numpy(), None, layer.stride, layer.padding)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


def test_conv2d_forward_single_sample_batch_of_one():
    layer = _fixed_conv(1, 2, kernel_size=3)
    x_data = np.random.default_rng(2).standard_normal((1, 1, 5, 5))
    y = layer(Tensor(x_data))
    expected = reference_conv2d(x_data, layer.weight.numpy(), layer.bias.numpy(), layer.stride, layer.padding)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


# -- runtime shape validation ---------------------------------------------------


def test_conv2d_rejects_wrong_input_ndim():
    layer = Conv2d(1, 2, kernel_size=3)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 4))))


def test_conv2d_rejects_channel_mismatch():
    layer = Conv2d(3, 2, kernel_size=3)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((1, 2, 8, 8))))


def test_conv2d_rejects_kernel_larger_than_padded_input():
    layer = Conv2d(1, 2, kernel_size=5, padding=0)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((1, 1, 4, 4))))


def test_conv2d_bias_shape_mismatch_raises():
    layer = Conv2d(1, 2, kernel_size=3)
    x = Tensor(np.zeros((1, 1, 5, 5)))
    bad_bias = Tensor(np.zeros(3))
    with pytest.raises(ShapeMismatchError):
        x.conv2d(layer.weight, bad_bias, layer.stride, layer.padding)


# -- gradient accumulation ------------------------------------------------------


def test_conv2d_weight_used_once_receives_gradient():
    layer = _fixed_conv(2, 3, kernel_size=3)
    x = Tensor(np.random.default_rng(3).standard_normal((2, 2, 6, 6)))
    layer(x).sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad is not None


def test_conv2d_weight_used_multiple_times_accumulates():
    """Same weight applied to two different inputs -> gradients add, matching two independent calls summed."""
    layer = _fixed_conv(2, 3, kernel_size=3, bias=False)
    x1 = Tensor(np.random.default_rng(4).standard_normal((1, 2, 6, 6)))
    x2 = Tensor(np.random.default_rng(5).standard_normal((1, 2, 6, 6)))

    combined = (layer(x1).sum() + layer(x2).sum())
    combined.backward()
    combined_grad = layer.weight.grad.numpy().copy()

    layer.weight.zero_grad()
    layer(x1).sum().backward()
    grad1 = layer.weight.grad.numpy().copy()
    layer.weight.zero_grad()
    layer(x2).sum().backward()
    grad2 = layer.weight.grad.numpy().copy()

    np.testing.assert_allclose(combined_grad, grad1 + grad2, **TOL)


def test_conv2d_input_used_multiple_times_accumulates():
    """Same input tensor fed through two different Conv2d layers -> gradients add through both paths."""
    layer1 = _fixed_conv(1, 2, kernel_size=3, bias=False)
    layer2 = _fixed_conv(1, 2, kernel_size=3, bias=False)
    x = Tensor(np.random.default_rng(6).standard_normal((1, 1, 6, 6)), requires_grad=True)

    (layer1(x).sum() + layer2(x).sum()).backward()
    combined_grad = x.grad.numpy().copy()

    x1 = Tensor(x.numpy(), requires_grad=True)
    layer1(x1).sum().backward()
    x2 = Tensor(x.numpy(), requires_grad=True)
    layer2(x2).sum().backward()

    np.testing.assert_allclose(combined_grad, x1.grad.numpy() + x2.grad.numpy(), **TOL)


def test_conv2d_integrates_with_sgd():
    layer = Conv2d(2, 3, kernel_size=3)
    x = Tensor(np.random.default_rng(7).standard_normal((2, 2, 6, 6)))
    before = layer.weight.numpy().copy()
    opt = SGD(layer.parameters(), lr=0.1)
    layer(x).sum().backward()
    opt.step()
    assert not np.allclose(before, layer.weight.numpy())


# -- finite-difference gradient checks ------------------------------------------


@pytest.mark.parametrize(
    "N,Cin,Cout,H,W,kernel_size,stride,padding",
    [
        (1, 1, 1, 5, 5, 3, 1, 0),
        (1, 2, 3, 5, 5, 3, 1, 1),
        (2, 2, 2, 6, 6, 3, 2, 1),
        (2, 3, 2, 7, 7, (3, 3), (2, 1), (1, 0)),
    ],
)
def test_conv2d_finite_difference_input_weight_bias(N, Cin, Cout, H, W, kernel_size, stride, padding):
    forge.random.seed(0)
    rng = np.random.default_rng(42)
    layer = Conv2d(Cin, Cout, kernel_size, stride=stride, padding=padding)
    x_data = rng.standard_normal((N, Cin, H, W))
    w_data = layer.weight.numpy().copy()
    b_data = layer.bias.numpy().copy()

    def loss_x(xd):
        return float((Tensor(xd).conv2d(Tensor(w_data), Tensor(b_data), layer.stride, layer.padding).numpy() ** 2).sum())

    def loss_w(wd):
        return float((Tensor(x_data).conv2d(Tensor(wd), Tensor(b_data), layer.stride, layer.padding).numpy() ** 2).sum())

    def loss_b(bd):
        return float((Tensor(x_data).conv2d(Tensor(w_data), Tensor(bd), layer.stride, layer.padding).numpy() ** 2).sum())

    x = Tensor(x_data.copy(), requires_grad=True)
    w = Tensor(w_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)
    out = x.conv2d(w, b, layer.stride, layer.padding)
    (out * out).sum().backward()

    np.testing.assert_allclose(x.grad.numpy(), numerical_grad(loss_x, x_data.copy()), **FD_TOL)
    np.testing.assert_allclose(w.grad.numpy(), numerical_grad(loss_w, w_data.copy()), **FD_TOL)
    np.testing.assert_allclose(b.grad.numpy(), numerical_grad(loss_b, b_data.copy()), **FD_TOL)
