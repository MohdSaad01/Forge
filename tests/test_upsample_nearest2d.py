"""Milestone 63 tests: `nn.UpsampleNearest2d` (CPU).

A new fused `Backend` primitive (`Tensor.upsample_nearest2d`,
`Backend.upsample_nearest2d`/`upsample_nearest2d_backward`) -- added as the
decoder half of `examples/autoencoder/`'s convolutional autoencoder, the
first Forge primitive that can grow an NCHW spatial map back up (the inverse
of `MaxPool2d`'s shrink). Mirrors `tests/test_pooling.py`/`test_maxpool1d.py`'s
structure. See `tests/test_cuda_upsample_nearest2d.py` for the CUDA
counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import UpsampleNearest2d
from forge.serialization import load_model, save_model

TOL = dict(rtol=1e-6, atol=1e-6)
FD_TOL = dict(rtol=1e-2, atol=1e-2)


def reference_upsample_nearest2d(x: np.ndarray, sh: int, sw: int) -> np.ndarray:
    return np.repeat(np.repeat(x, sh, axis=2), sw, axis=3)


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


# -- configuration ------------------------------------------------------------


def test_upsample_nearest2d_default_scale_factor():
    layer = UpsampleNearest2d()
    assert layer.scale_factor == (2, 2)


def test_upsample_nearest2d_int_scale_factor_applies_to_both_dims():
    layer = UpsampleNearest2d(3)
    assert layer.scale_factor == (3, 3)


def test_upsample_nearest2d_pair_scale_factor():
    layer = UpsampleNearest2d((2, 3))
    assert layer.scale_factor == (2, 3)


def test_upsample_nearest2d_has_no_parameters():
    assert list(UpsampleNearest2d().parameters()) == []


@pytest.mark.parametrize("scale_factor", [0, -1, 2.0, "2"])
def test_upsample_nearest2d_rejects_invalid_scale_factor(scale_factor):
    with pytest.raises(ShapeMismatchError):
        UpsampleNearest2d(scale_factor)


def test_upsample_nearest2d_rejects_wrong_input_ndim():
    layer = UpsampleNearest2d(2)
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 4))))
    with pytest.raises(ShapeMismatchError):
        layer(Tensor(np.zeros((4, 1, 4, 4, 1))))


# -- output shape ---------------------------------------------------------------


def test_upsample_nearest2d_output_shape():
    layer = UpsampleNearest2d(2)
    y = layer(Tensor(np.zeros((3, 4, 5, 7))))
    assert y.shape == (3, 4, 10, 14)


def test_upsample_nearest2d_asymmetric_output_shape():
    layer = UpsampleNearest2d((2, 3))
    y = layer(Tensor(np.zeros((1, 2, 4, 5))))
    assert y.shape == (1, 2, 8, 15)


# -- forward correctness vs. an independent reference -----------------------------


@pytest.mark.parametrize("scale_factor", [1, 2, 3, (2, 3), (3, 1)])
def test_upsample_nearest2d_forward_matches_reference(scale_factor):
    layer = UpsampleNearest2d(scale_factor)
    x_data = np.random.default_rng(0).standard_normal((2, 3, 4, 5))
    y = layer(Tensor(x_data))
    expected = reference_upsample_nearest2d(x_data, *layer.scale_factor)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


def test_upsample_nearest2d_repeats_each_element_into_a_block():
    x = Tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]]))  # (1, 1, 2, 2)
    y = UpsampleNearest2d(2)(x)
    expected = np.array(
        [[[[1.0, 1.0, 2.0, 2.0],
           [1.0, 1.0, 2.0, 2.0],
           [3.0, 3.0, 4.0, 4.0],
           [3.0, 3.0, 4.0, 4.0]]]]
    )
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


# -- gradient shape / accumulation -------------------------------------------------


def test_upsample_nearest2d_backward_grad_shape_matches_input():
    x = Tensor(np.random.default_rng(1).standard_normal((2, 3, 4, 5)), requires_grad=True)
    UpsampleNearest2d(2)(x).sum().backward()
    assert x.grad.shape == x.shape


def test_upsample_nearest2d_backward_sums_each_output_block():
    # Every element of the (sh, sw) output block traces back to one input
    # element -- summing the upstream gradient (here all-ones, via `.sum()`)
    # must therefore multiply each input element's gradient by sh*sw.
    x = Tensor(np.random.default_rng(2).standard_normal((1, 2, 3, 4)), requires_grad=True)
    UpsampleNearest2d((2, 3))(x).sum().backward()
    expected = np.full((1, 2, 3, 4), 2 * 3, dtype=x.grad.numpy().dtype)
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


# -- finite-difference gradient checks ----------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W,scale_factor",
    [
        (1, 1, 4, 4, 2),
        (1, 2, 3, 5, (2, 3)),
        (2, 2, 4, 6, 1),
        (2, 3, 5, 4, (3, 2)),
    ],
)
def test_upsample_nearest2d_finite_difference(N, C, H, W, scale_factor):
    rng = np.random.default_rng(3)
    x_data = rng.standard_normal((N, C, H, W))
    layer = UpsampleNearest2d(scale_factor)

    def loss(xd):
        return float((layer(Tensor(xd)).numpy() ** 2).sum())

    x = Tensor(x_data.copy(), requires_grad=True)
    out = layer(x)
    (out * out).sum().backward()

    np.testing.assert_allclose(x.grad.numpy(), numerical_grad(loss, x_data.copy()), **FD_TOL)


# -- composition with Conv2d (the actual decoder pattern) -------------------------------


def test_upsample_nearest2d_composes_with_conv2d_and_trains():
    from forge.nn import Conv2d, Sequential
    from forge.optim import SGD

    forge.random.seed(0)
    model = Sequential(UpsampleNearest2d(2), Conv2d(2, 2, kernel_size=3, padding=1))
    x = Tensor(np.random.default_rng(4).standard_normal((2, 2, 4, 4)).astype(np.float32), requires_grad=False)
    optimizer = SGD(model.parameters(), lr=0.1)

    conv = model._modules["1"]
    before = conv.weight.numpy().copy()
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    after = conv.weight.numpy()

    assert not np.allclose(before, after)


# -- serialization --------------------------------------------------------------------


def test_upsample_nearest2d_serialization_round_trip_preserves_predictions(tmp_path):
    layer = UpsampleNearest2d((2, 3))
    x = Tensor(np.random.default_rng(9).standard_normal((3, 2, 4, 5)).astype(np.float32))
    with forge.no_grad():
        pre_save = layer(x).numpy()

    path = tmp_path / "upsample_nearest2d.forge"
    save_model(layer, str(path))
    reloaded = load_model(str(path))
    with forge.no_grad():
        post_load = reloaded(x).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)
    assert reloaded.scale_factor == layer.scale_factor
