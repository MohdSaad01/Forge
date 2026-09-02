"""Milestone 16 tests: `Flatten` (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import Flatten, Linear, Module, ReLU, Sequential


# -- default behavior: (N, C, H, W) -> (N, C*H*W) ----------------------------


def test_default_flatten_collapses_all_but_batch_dim():
    x = Tensor(np.zeros((5, 3, 4, 6), dtype=np.float32))
    out = Flatten()(x)
    assert out.shape == (5, 3 * 4 * 6)


def test_default_flatten_values_match_reshape():
    x_data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    x = Tensor(x_data)
    out = Flatten()(x)
    np.testing.assert_array_equal(out.numpy(), x_data.reshape(2, 60))


def test_flatten_has_no_parameters():
    assert dict(Flatten().named_parameters()) == {}


# -- start_dim / end_dim generalization ---------------------------------------


def test_flatten_custom_middle_range():
    x = Tensor(np.zeros((2, 3, 4, 5), dtype=np.float32))
    out = Flatten(start_dim=1, end_dim=2)(x)
    assert out.shape == (2, 12, 5)


def test_flatten_full_collapse_including_batch():
    x = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    out = Flatten(start_dim=0, end_dim=-1)(x)
    assert out.shape == (24,)


def test_flatten_negative_end_dim_matches_last_axis():
    x = Tensor(np.zeros((2, 3, 4, 5), dtype=np.float32))
    out_neg = Flatten(start_dim=1, end_dim=-1)(x)
    out_pos = Flatten(start_dim=1, end_dim=3)(x)
    assert out_neg.shape == out_pos.shape == (2, 60)


def test_flatten_single_dim_range_is_a_no_op_shape():
    x = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    out = Flatten(start_dim=1, end_dim=1)(x)
    assert out.shape == (2, 3, 4)


# -- validation --------------------------------------------------------------


def test_flatten_start_after_end_raises_shape_error():
    x = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ShapeMismatchError):
        Flatten(start_dim=2, end_dim=1)(x)


def test_flatten_end_dim_out_of_range_raises_shape_error():
    x = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ShapeMismatchError):
        Flatten(start_dim=0, end_dim=5)(x)


def test_flatten_start_dim_out_of_range_raises_shape_error():
    x = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ShapeMismatchError):
        Flatten(start_dim=-10, end_dim=-1)(x)


# -- autograd -----------------------------------------------------------------


def test_flatten_participates_in_autograd_gradient_shape_matches_input():
    x = Tensor(np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32), requires_grad=True)
    out = Flatten()(x)
    out.sum().backward()
    assert x.grad.shape == x.shape
    np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(x.numpy()))


def test_flatten_backward_matches_manual_reshape_gradient():
    x_data = np.random.default_rng(1).standard_normal((2, 3, 4)).astype(np.float32)
    x1 = Tensor(x_data.copy(), requires_grad=True)
    x2 = Tensor(x_data.copy(), requires_grad=True)

    Flatten()(x1).sum().backward()
    x2.reshape(2, 12).sum().backward()
    np.testing.assert_array_equal(x1.grad.numpy(), x2.grad.numpy())


# -- integration with Sequential / Linear -------------------------------------


def test_flatten_feeds_linear_through_sequential():
    forge.random.seed(0)
    model = Sequential(Flatten(), Linear(12, 4), ReLU())
    x = Tensor(np.random.default_rng(2).standard_normal((3, 3, 4)).astype(np.float32))
    out = model(x)
    assert out.shape == (3, 4)


# -- serialization-relevant config attributes exist as plain ints -----------


def test_flatten_repr_reports_configured_dims():
    f = Flatten(start_dim=2, end_dim=3)
    assert "start_dim=2" in repr(f)
    assert "end_dim=3" in repr(f)
