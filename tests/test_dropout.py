"""Milestone 16 tests: `Dropout` (CPU).

Determinism comes from explicitly seeding `forge.random` (or passing an
explicit `numpy.random.Generator`) -- never from asserting exact values
against an unseeded draw. Statistical tests use a large tensor and a
generous tolerance so they don't flake.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ModuleError
from forge.nn import Dropout


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5, -1.0])
def test_dropout_rejects_p_out_of_range(p):
    with pytest.raises(ModuleError):
        Dropout(p)


@pytest.mark.parametrize("p", [0.0, 0.5, 0.999])
def test_dropout_accepts_valid_p(p):
    Dropout(p)  # must not raise


def test_dropout_default_p_is_one_half():
    assert Dropout().p == 0.5


def test_dropout_has_no_parameters():
    assert dict(Dropout(0.5).named_parameters()) == {}


# -- training: statistical behavior --------------------------------------------


def test_dropout_training_zeroes_approximately_p_fraction():
    forge.random.seed(1)
    p = 0.3
    x = Tensor(np.ones((500, 500), dtype=np.float32))
    out = Dropout(p)(x).numpy()

    frac_zero = float((out == 0).mean())
    assert abs(frac_zero - p) < 0.02


def test_dropout_training_preserves_mean_within_tolerance():
    forge.random.seed(2)
    p = 0.4
    x_data = np.random.default_rng(3).standard_normal((400, 400)).astype(np.float32) + 5.0
    x = Tensor(x_data)
    out = Dropout(p)(x).numpy()

    assert abs(out.mean() - x_data.mean()) < 0.05 * abs(x_data.mean())


def test_dropout_nonzero_elements_are_scaled_by_one_over_one_minus_p():
    forge.random.seed(4)
    p = 0.25
    x = Tensor(np.ones((200, 200), dtype=np.float32))
    out = Dropout(p)(x).numpy()

    nonzero = out[out != 0]
    assert nonzero.size > 0
    np.testing.assert_allclose(nonzero, 1.0 / (1.0 - p), rtol=1e-5)


def test_dropout_p_zero_is_exact_identity():
    forge.random.seed(5)
    x_data = np.random.default_rng(6).standard_normal((10, 10)).astype(np.float32)
    x = Tensor(x_data)
    out = Dropout(0.0)(x).numpy()
    np.testing.assert_array_equal(out, x_data)


# -- evaluation: exact identity -------------------------------------------------


def test_dropout_eval_mode_is_identity():
    d = Dropout(0.5)
    d.eval()
    x_data = np.random.default_rng(7).standard_normal((8, 8)).astype(np.float32)
    x = Tensor(x_data)
    out = d(x)
    np.testing.assert_array_equal(out.numpy(), x_data)


def test_dropout_eval_mode_returns_the_same_tensor_object():
    """No new graph node is inserted during eval -- `forward()` returns `x` itself."""
    d = Dropout(0.5)
    d.eval()
    x = Tensor([1.0, 2.0, 3.0])
    assert d(x) is x


def test_dropout_eval_mode_is_deterministic_across_calls():
    d = Dropout(0.9)
    d.eval()
    x = Tensor(np.random.default_rng(8).standard_normal((5, 5)).astype(np.float32))
    out1 = d(x).numpy()
    out2 = d(x).numpy()
    np.testing.assert_array_equal(out1, out2)


# -- train() / eval() transition -------------------------------------------------


def test_dropout_switching_back_to_train_restores_stochastic_behavior():
    forge.random.seed(9)
    d = Dropout(0.5)
    x = Tensor(np.ones((300, 300), dtype=np.float32))

    d.eval()
    assert (d(x).numpy() == 1.0).all()

    d.train()
    out = d(x).numpy()
    assert (out == 0).any()  # overwhelmingly likely at p=0.5, 90000 elements


# -- autograd: backward reuses the forward mask ----------------------------------


def test_dropout_backward_gradient_matches_forward_mask_pattern():
    forge.random.seed(10)
    x = Tensor(np.ones((50, 50), dtype=np.float32), requires_grad=True)
    d = Dropout(0.4)
    out = d(x)
    out.sum().backward()

    out_np = out.numpy()
    grad_np = x.grad.numpy()
    # grad is nonzero exactly where output is nonzero, same scale (input was all-ones).
    np.testing.assert_array_equal(out_np != 0, grad_np != 0)
    np.testing.assert_allclose(grad_np[grad_np != 0], out_np[out_np != 0])


def test_dropout_backward_does_not_redraw_a_different_mask():
    """Two separate backward-eligible forward passes must use independently
    drawn masks (not literally the same array object across calls), but each
    forward's own backward must reuse *that* forward's mask exactly."""
    forge.random.seed(11)
    d = Dropout(0.5)
    x1 = Tensor(np.ones((30, 30), dtype=np.float32), requires_grad=True)
    x2 = Tensor(np.ones((30, 30), dtype=np.float32), requires_grad=True)

    out1 = d(x1)
    out2 = d(x2)
    out1.sum().backward()
    out2.sum().backward()

    np.testing.assert_array_equal(out1.numpy() != 0, x1.grad.numpy() != 0)
    np.testing.assert_array_equal(out2.numpy() != 0, x2.grad.numpy() != 0)


def test_dropout_eval_mode_gradient_is_identity():
    x = Tensor(np.random.default_rng(12).standard_normal((6, 6)).astype(np.float32), requires_grad=True)
    d = Dropout(0.5)
    d.eval()
    out = d(x)
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(x.numpy()))


# -- determinism under explicit seeding -----------------------------------------


def test_dropout_reproducible_via_forge_random_seed():
    x_data = np.ones((20, 20), dtype=np.float32)

    forge.random.seed(42)
    out1 = Dropout(0.3)(Tensor(x_data)).numpy()

    forge.random.seed(42)
    out2 = Dropout(0.3)(Tensor(x_data)).numpy()

    np.testing.assert_array_equal(out1, out2)


def test_dropout_reproducible_via_explicit_generator():
    x_data = np.ones((20, 20), dtype=np.float32)

    out1 = Dropout(0.3, generator=np.random.default_rng(99))(Tensor(x_data)).numpy()
    out2 = Dropout(0.3, generator=np.random.default_rng(99))(Tensor(x_data)).numpy()

    np.testing.assert_array_equal(out1, out2)


def test_dropout_explicit_generator_independent_of_global_forge_random_state():
    x_data = np.ones((20, 20), dtype=np.float32)

    forge.random.seed(1)
    out1 = Dropout(0.3, generator=np.random.default_rng(7))(Tensor(x_data)).numpy()

    forge.random.seed(999)  # different global state
    out2 = Dropout(0.3, generator=np.random.default_rng(7))(Tensor(x_data)).numpy()

    np.testing.assert_array_equal(out1, out2)


def test_dropout_two_forward_calls_draw_different_masks():
    """Successive forward calls in one training loop must not repeat the same mask."""
    forge.random.seed(13)
    d = Dropout(0.5)
    x = Tensor(np.ones((100, 100), dtype=np.float32))
    out1 = d(x).numpy()
    out2 = d(x).numpy()
    assert not np.array_equal(out1, out2)


# -- dtype / shape preservation --------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dropout_preserves_dtype(dtype):
    forge.random.seed(14)
    x = Tensor(np.ones((5, 5)), dtype=dtype)
    out = Dropout(0.5)(x)
    assert str(out.dtype) == dtype


def test_dropout_preserves_shape():
    forge.random.seed(15)
    x = Tensor(np.ones((3, 4, 5), dtype=np.float32))
    out = Dropout(0.5)(x)
    assert out.shape == x.shape
