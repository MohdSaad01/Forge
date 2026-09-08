"""Milestone 62 tests: `nn.MaxPool1d` (CPU).

`MaxPool1d` is implemented entirely by reshaping to a dummy `(N, C, 1, L)`
4D tensor and dispatching to the existing, already-tested `Tensor.max_pool2d`
(see `forge/nn/pooling.py`'s `MaxPool1d` docstring) -- no new `Backend`
method. Mirrors `tests/test_pooling.py`'s structure adapted to 1D. See
`tests/test_cuda_conv1d.py` for the CUDA counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import MaxPool1d
from forge.serialization import load_model, save_model

TOL = dict(rtol=1e-6, atol=1e-6)
FD_TOL = dict(rtol=1e-2, atol=1e-2)


def reference_max_pool1d(x: np.ndarray, kernel_size, stride, padding) -> np.ndarray:
    N, C, L = x.shape
    L_out = (L + 2 * padding - kernel_size) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), constant_values=-np.inf)
    out = np.zeros((N, C, L_out), dtype=np.float64)
    for n in range(N):
        for c in range(C):
            for lo in range(L_out):
                li0 = lo * stride
                out[n, c, lo] = padded[n, c, li0 : li0 + kernel_size].max()
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


def test_maxpool1d_defaults_stride_to_kernel_size():
    layer = MaxPool1d(3)
    assert layer.kernel_size == 3
    assert layer.stride == 3
    assert layer.padding == 0


def test_maxpool1d_explicit_stride_overrides_default():
    layer = MaxPool1d(3, stride=1, padding=1)
    assert layer.stride == 1
    assert layer.padding == 1


def test_maxpool1d_has_no_parameters():
    layer = MaxPool1d(2)
    assert list(layer.parameters()) == []


@pytest.mark.parametrize("kernel_size", [0, -1, 2.0, "2"])
def test_maxpool1d_rejects_invalid_kernel_size(kernel_size):
    with pytest.raises(ShapeMismatchError):
        MaxPool1d(kernel_size)


@pytest.mark.parametrize("stride", [0, -1])
def test_maxpool1d_rejects_invalid_stride(stride):
    with pytest.raises(ShapeMismatchError):
        MaxPool1d(2, stride=stride)


@pytest.mark.parametrize("padding", [-1, -2])
def test_maxpool1d_rejects_invalid_padding(padding):
    with pytest.raises(ShapeMismatchError):
        MaxPool1d(2, padding=padding)


# -- output shape -------------------------------------------------------------


def test_maxpool1d_output_shape_non_overlapping():
    layer = MaxPool1d(2)
    y = layer(Tensor(np.zeros((3, 4, 8))))
    assert y.shape == (3, 4, 4)


def test_maxpool1d_output_shape_with_stride_and_padding():
    layer = MaxPool1d(3, stride=2, padding=1)
    y = layer(Tensor(np.zeros((2, 3, 7))))
    assert y.shape == (2, 3, 4)


def test_maxpool1d_rejects_kernel_larger_than_padded_input():
    layer = MaxPool1d(5)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((1, 1, 4))))


def test_maxpool1d_rejects_wrong_input_ndim():
    layer = MaxPool1d(2)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 4))))
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 1, 4, 4))))


# -- forward correctness vs. an independent reference -------------------------


@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [(2, 2, 0), (2, 1, 0), (3, 2, 1), (3, 1, 1)],
)
def test_maxpool1d_forward_matches_reference(kernel_size, stride, padding):
    layer = MaxPool1d(kernel_size, stride=stride, padding=padding)
    x_data = np.random.default_rng(0).standard_normal((2, 3, 10))
    y = layer(Tensor(x_data))
    expected = reference_max_pool1d(x_data, layer.kernel_size, layer.stride, layer.padding)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


# -- tie-breaking: deterministic, first occurrence -----------------------------


def test_maxpool1d_tie_breaks_to_first_occurrence():
    x = Tensor(np.array([[[5.0, 5.0, 5.0]]]), requires_grad=True)
    out = MaxPool1d(3)(x)
    assert out.numpy().item() == 5.0
    out.sum().backward()
    expected = np.array([[[1.0, 0.0, 0.0]]])
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


# -- gradient shape / accumulation under overlap -------------------------------


def test_maxpool1d_backward_grad_shape_matches_input():
    x = Tensor(np.random.default_rng(1).standard_normal((2, 3, 6)), requires_grad=True)
    MaxPool1d(2)(x).sum().backward()
    assert x.grad.shape == x.shape


def test_maxpool1d_overlapping_windows_accumulate_gradient():
    x_data = np.zeros((1, 1, 3))
    x_data[0, 0, 1] = 10.0  # unique max, inside every overlapping window of size 2 that covers it
    x = Tensor(x_data, requires_grad=True)
    out = MaxPool1d(2, stride=1)(x)  # (1,1,2): both windows contain index 1
    assert out.shape == (1, 1, 2)
    out.sum().backward()
    expected = np.zeros((1, 1, 3))
    expected[0, 0, 1] = 2.0
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


# -- finite-difference gradient checks ------------------------------------------


@pytest.mark.parametrize(
    "N,C,L,kernel_size,stride,padding",
    [
        (1, 1, 8, 2, 2, 0),
        (1, 2, 9, 3, 2, 1),
        (2, 2, 10, 2, 1, 0),
        (2, 3, 11, 3, 2, 1),
    ],
)
def test_maxpool1d_finite_difference(N, C, L, kernel_size, stride, padding):
    rng = np.random.default_rng(3)
    # Distinct values everywhere avoid ties landing exactly on a finite-difference probe.
    x_data = rng.permutation(N * C * L).astype(np.float64).reshape(N, C, L)
    x_data += rng.standard_normal(x_data.shape) * 1e-3

    layer = MaxPool1d(kernel_size, stride=stride, padding=padding)

    def _forward(xd):
        n, c, length = xd.shape
        x4 = Tensor(xd).reshape(n, c, 1, length)
        return x4.max_pool2d((1, layer.kernel_size), (1, layer.stride), (0, layer.padding))

    def loss(xd):
        return float((_forward(xd).numpy() ** 2).sum())

    x = Tensor(x_data.copy(), requires_grad=True)
    out = layer(x)
    (out * out).sum().backward()

    np.testing.assert_allclose(x.grad.numpy(), numerical_grad(loss, x_data.copy()), **FD_TOL)


# -- serialization ----------------------------------------------------------------


def test_maxpool1d_serialization_round_trip_preserves_predictions(tmp_path):
    layer = MaxPool1d(3, stride=2, padding=1)
    x = Tensor(np.random.default_rng(9).standard_normal((3, 2, 10)).astype(np.float32))
    with forge.no_grad():
        pre_save = layer(x).numpy()

    path = tmp_path / "maxpool1d.forge"
    save_model(layer, str(path))
    reloaded = load_model(str(path))
    with forge.no_grad():
        post_load = reloaded(x).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)
    assert reloaded.kernel_size == layer.kernel_size
    assert reloaded.stride == layer.stride
    assert reloaded.padding == layer.padding
