"""Milestone 15 tests: `nn.MaxPool2d` (CPU).

Covers configuration validation/defaults, forward correctness against an
independent reference implementation, deterministic tie-breaking, runtime
shape validation, gradient accumulation under overlapping windows
(stride < kernel_size), and finite-difference gradient checks. See
`tests/test_cuda_conv.py` for the CUDA counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import MaxPool2d

TOL = dict(rtol=1e-6, atol=1e-6)
FD_TOL = dict(rtol=1e-2, atol=1e-2)


def reference_max_pool2d(x: np.ndarray, kernel_size, stride, padding) -> np.ndarray:
    N, C, H, W = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding
    H_out = (H + 2 * PH - KH) // SH + 1
    W_out = (W + 2 * PW - KW) // SW + 1
    padded = np.pad(x, ((0, 0), (0, 0), (PH, PH), (PW, PW)), constant_values=-np.inf)
    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for n in range(N):
        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):
                    hi0, wi0 = ho * SH, wo * SW
                    out[n, c, ho, wo] = padded[n, c, hi0 : hi0 + KH, wi0 : wi0 + KW].max()
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


# -- configuration ----------------------------------------------------------


def test_maxpool2d_defaults_stride_to_kernel_size():
    layer = MaxPool2d(3)
    assert layer.kernel_size == (3, 3)
    assert layer.stride == (3, 3)
    assert layer.padding == (0, 0)


def test_maxpool2d_explicit_stride_overrides_default():
    layer = MaxPool2d(3, stride=1, padding=1)
    assert layer.stride == (1, 1)
    assert layer.padding == (1, 1)


def test_maxpool2d_has_no_parameters():
    layer = MaxPool2d(2)
    assert list(layer.parameters()) == []


@pytest.mark.parametrize("kernel_size", [0, -1, (0, 2), (2,), "2"])
def test_maxpool2d_rejects_invalid_kernel_size(kernel_size):
    with pytest.raises(ShapeMismatchError):
        MaxPool2d(kernel_size)


@pytest.mark.parametrize("stride", [0, -1, (0, 1)])
def test_maxpool2d_rejects_invalid_stride(stride):
    with pytest.raises(ShapeMismatchError):
        MaxPool2d(2, stride=stride)


@pytest.mark.parametrize("padding", [-1, (-1, 0)])
def test_maxpool2d_rejects_invalid_padding(padding):
    with pytest.raises(ShapeMismatchError):
        MaxPool2d(2, padding=padding)


# -- output shape -------------------------------------------------------------


def test_maxpool2d_output_shape_non_overlapping():
    layer = MaxPool2d(2)
    y = layer(Tensor(np.zeros((3, 4, 8, 8))))
    assert y.shape == (3, 4, 4, 4)


def test_maxpool2d_output_shape_with_stride_and_padding():
    layer = MaxPool2d(3, stride=2, padding=1)
    y = layer(Tensor(np.zeros((2, 3, 7, 7))))
    assert y.shape == (2, 3, 4, 4)


def test_maxpool2d_rejects_kernel_larger_than_padded_input():
    layer = MaxPool2d(5)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((1, 1, 4, 4))))


def test_maxpool2d_rejects_wrong_input_ndim():
    layer = MaxPool2d(2)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 4))))


# -- forward correctness vs. an independent reference -------------------------


@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [(2, 2, 0), (2, 1, 0), (3, 2, 1), (3, 1, 1)],
)
def test_maxpool2d_forward_matches_reference(kernel_size, stride, padding):
    layer = MaxPool2d(kernel_size, stride=stride, padding=padding)
    x_data = np.random.default_rng(0).standard_normal((2, 3, 8, 8))
    y = layer(Tensor(x_data))
    expected = reference_max_pool2d(x_data, layer.kernel_size, layer.stride, layer.padding)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


# -- tie-breaking: deterministic, first occurrence in row-major scan order ----


def test_maxpool2d_tie_breaks_to_first_occurrence_top_left():
    x = Tensor(np.array([[[[5.0, 5.0], [5.0, 5.0]]]]), requires_grad=True)
    out = MaxPool2d(2)(x)
    assert out.numpy().item() == 5.0
    out.sum().backward()
    expected = np.array([[[[1.0, 0.0], [0.0, 0.0]]]])
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


def test_maxpool2d_tie_breaks_row_major_not_column_major():
    """A tie between a horizontal and vertical neighbor -- row-major picks the earlier row first."""
    x = Tensor(np.array([[[[1.0, 5.0], [5.0, 1.0]]]]), requires_grad=True)
    out = MaxPool2d(2)(x)
    out.sum().backward()
    # Row-major flatten order is [ (0,0), (0,1), (1,0), (1,1) ] -> index 1 = (0,1) wins over index 2 = (1,0).
    expected = np.array([[[[0.0, 1.0], [0.0, 0.0]]]])
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


def test_maxpool2d_tie_break_is_reproducible_across_calls():
    x_data = np.full((1, 1, 4, 4), 3.0)
    layer = MaxPool2d(2)
    grads = []
    for _ in range(3):
        x = Tensor(x_data.copy(), requires_grad=True)
        layer(x).sum().backward()
        grads.append(x.grad.numpy().copy())
    np.testing.assert_array_equal(grads[0], grads[1])
    np.testing.assert_array_equal(grads[0], grads[2])


# -- gradient shape / accumulation under overlap -------------------------------


def test_maxpool2d_backward_grad_shape_matches_input():
    x = Tensor(np.random.default_rng(1).standard_normal((2, 3, 6, 6)), requires_grad=True)
    MaxPool2d(2)(x).sum().backward()
    assert x.grad.shape == x.shape


def test_maxpool2d_overlapping_windows_accumulate_gradient():
    """stride < kernel_size: an input element can be the argmax for more than one output window."""
    x_data = np.zeros((1, 1, 3, 3))
    x_data[0, 0, 1, 1] = 10.0  # the unique max, inside every overlapping 2x2 window that covers it
    x = Tensor(x_data, requires_grad=True)
    out = MaxPool2d(2, stride=1)(x)  # (1,1,2,2): all four windows contain (1,1)
    assert out.shape == (1, 1, 2, 2)
    out.sum().backward()
    expected = np.zeros((1, 1, 3, 3))
    expected[0, 0, 1, 1] = 4.0  # selected as the max by all 4 overlapping windows
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


# -- finite-difference gradient checks ------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W,kernel_size,stride,padding",
    [
        (1, 1, 5, 5, 2, 2, 0),
        (1, 2, 5, 5, 3, 2, 1),
        (2, 2, 6, 6, 2, 1, 0),
        (2, 3, 7, 7, (3, 2), (2, 1), (1, 0)),
    ],
)
def test_maxpool2d_finite_difference(N, C, H, W, kernel_size, stride, padding):
    rng = np.random.default_rng(3)
    # Distinct values everywhere avoid ties landing exactly on a finite-difference probe.
    x_data = rng.permutation(N * C * H * W).astype(np.float64).reshape(N, C, H, W)
    x_data += rng.standard_normal(x_data.shape) * 1e-3

    layer = MaxPool2d(kernel_size, stride=stride, padding=padding)

    def loss(xd):
        return float((Tensor(xd).max_pool2d(layer.kernel_size, layer.stride, layer.padding).numpy() ** 2).sum())

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.max_pool2d(layer.kernel_size, layer.stride, layer.padding)
    (out * out).sum().backward()

    np.testing.assert_allclose(x.grad.numpy(), numerical_grad(loss, x_data.copy()), **FD_TOL)
