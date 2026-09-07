"""`nn.BatchNorm2d` (Milestone 53): CPU forward/backward/training-eval semantics.

CPU's `BatchNorm2d.forward()` composes entirely from general Tensor
primitives (`sum`/`sqrt`/`div`/`reshape`, `forge/nn/batchnorm.py`), so this
suite validates the *composed* behavior end-to-end -- there is no separate
CPU backward rule to test in isolation. See `tests/test_batchnorm_cuda.py`
for the CUDA fused-kernel parity suite and `docs/development/
m53-batchnorm.md` for the full design writeup.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import BatchNorm2d, Conv2d, Sequential

TOL = dict(rtol=1e-5, atol=1e-6)


def _reference_batchnorm(x: np.ndarray, weight, bias, eps: float) -> np.ndarray:
    """An independent NumPy reference for training-mode BatchNorm2d, per-channel."""
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)  # biased (ddof=0), matching training normalization
    xhat = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        C = x.shape[1]
        xhat = xhat * weight.reshape(1, C, 1, 1) + bias.reshape(1, C, 1, 1)
    return xhat


# -- construction / validation ----------------------------------------------


def test_batchnorm2d_rejects_non_positive_num_features():
    with pytest.raises(ShapeMismatchError):
        BatchNorm2d(0)


def test_batchnorm2d_rejects_out_of_range_momentum():
    with pytest.raises(ShapeMismatchError):
        BatchNorm2d(3, momentum=1.5)


def test_batchnorm2d_rejects_wrong_channel_count():
    bn = BatchNorm2d(4)
    x = Tensor(np.zeros((2, 3, 5, 5), dtype=np.float32))
    with pytest.raises(ShapeMismatchError):
        bn(x)


def test_batchnorm2d_rejects_non_4d_input():
    bn = BatchNorm2d(3)
    with pytest.raises(ShapeMismatchError):
        bn(Tensor(np.zeros((3, 5), dtype=np.float32)))


def test_batchnorm2d_default_running_stats_are_zero_and_one():
    bn = BatchNorm2d(4)
    np.testing.assert_array_equal(bn.running_mean.numpy(), np.zeros(4))
    np.testing.assert_array_equal(bn.running_var.numpy(), np.ones(4))


def test_batchnorm2d_default_affine_params_are_identity():
    bn = BatchNorm2d(4)
    np.testing.assert_array_equal(bn.weight.numpy(), np.ones(4))
    np.testing.assert_array_equal(bn.bias.numpy(), np.zeros(4))


def test_batchnorm2d_affine_false_has_no_weight_or_bias():
    bn = BatchNorm2d(4, affine=False)
    assert bn.weight is None
    assert bn.bias is None
    assert "weight" not in dict(bn.named_parameters())


# -- training forward: matches an independent reference ----------------------


def test_training_forward_matches_numpy_reference():
    forge.random.seed(0)
    bn = BatchNorm2d(3)
    x0 = np.random.default_rng(1).standard_normal((5, 3, 4, 4)).astype(np.float64)
    x = Tensor(x0, dtype="float64")
    y = bn(x)
    expected = _reference_batchnorm(x0, bn.weight.numpy(), bn.bias.numpy(), bn.eps)
    np.testing.assert_allclose(y.numpy(), expected, **TOL)


def test_training_forward_zero_mean_unit_variance_per_channel_before_affine():
    bn = BatchNorm2d(2, affine=False)
    x0 = np.random.default_rng(2).standard_normal((6, 2, 5, 5)).astype(np.float64)
    y = bn(Tensor(x0, dtype="float64")).numpy()
    np.testing.assert_allclose(y.mean(axis=(0, 2, 3)), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(y.var(axis=(0, 2, 3)), [1.0, 1.0], atol=1e-4)


# -- running statistics --------------------------------------------------------


def test_running_stats_update_with_correct_momentum_and_unbiased_variance():
    bn = BatchNorm2d(2, momentum=0.3)
    x0 = np.random.default_rng(3).standard_normal((4, 2, 3, 3)).astype(np.float64)
    bn(Tensor(x0, dtype="float64"))

    count = 4 * 3 * 3
    batch_mean = x0.mean(axis=(0, 2, 3))
    batch_var = x0.var(axis=(0, 2, 3))  # biased
    unbiased_var = batch_var * count / (count - 1)
    expected_running_mean = 0.7 * 0.0 + 0.3 * batch_mean
    expected_running_var = 0.7 * 1.0 + 0.3 * unbiased_var

    np.testing.assert_allclose(bn.running_mean.numpy(), expected_running_mean, **TOL)
    np.testing.assert_allclose(bn.running_var.numpy(), expected_running_var, **TOL)


def test_running_stats_accumulate_across_repeated_training_batches():
    bn = BatchNorm2d(2, momentum=0.5)
    rng = np.random.default_rng(4)
    for _ in range(3):
        bn(Tensor(rng.standard_normal((4, 2, 3, 3)).astype(np.float32)))
    # After several updates the running stats should have moved away from
    # their init (0, 1) -- a coarse but device-independent liveness check.
    assert not np.allclose(bn.running_mean.numpy(), 0.0)
    assert not np.allclose(bn.running_var.numpy(), 1.0)


def test_eval_mode_does_not_update_running_stats():
    bn = BatchNorm2d(2)
    bn.eval()
    before_mean = bn.running_mean.numpy().copy()
    before_var = bn.running_var.numpy().copy()
    bn(Tensor(np.random.default_rng(5).standard_normal((4, 2, 3, 3)).astype(np.float32)))
    np.testing.assert_array_equal(bn.running_mean.numpy(), before_mean)
    np.testing.assert_array_equal(bn.running_var.numpy(), before_var)


def test_eval_mode_uses_running_stats_not_batch_stats():
    bn = BatchNorm2d(2, affine=False)
    bn.running_mean._data = np.array([5.0, -5.0])
    bn.running_var._data = np.array([2.0, 3.0])
    bn.eval()
    x = Tensor(np.random.default_rng(6).standard_normal((4, 2, 3, 3)).astype(np.float32))
    y = bn(x).numpy()
    expected = (x.numpy() - np.array([5.0, -5.0]).reshape(1, 2, 1, 1)) / np.sqrt(
        np.array([2.0, 3.0]).reshape(1, 2, 1, 1) + bn.eps
    )
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)


def test_eval_output_does_not_depend_on_current_batch():
    bn = BatchNorm2d(2)
    bn(Tensor(np.random.default_rng(7).standard_normal((8, 2, 4, 4)).astype(np.float32)))  # warm up running stats
    bn.eval()
    x1 = Tensor(np.random.default_rng(8).standard_normal((3, 2, 4, 4)).astype(np.float32))
    x2 = Tensor(np.random.default_rng(9).standard_normal((3, 2, 4, 4)).astype(np.float32))
    # Same running stats, different batches -> normalization formula is
    # identical per-element regardless of the other samples in the batch.
    single_x1 = Tensor(x1.numpy()[:1])
    joint_first_row = bn(x1).numpy()[0]
    solo_first_row = bn(single_x1).numpy()[0]
    np.testing.assert_allclose(joint_first_row, solo_first_row, atol=1e-5)


# -- eps ------------------------------------------------------------------


def test_eps_changes_normalization_output():
    x = Tensor(np.random.default_rng(10).standard_normal((4, 2, 3, 3)).astype(np.float64), dtype="float64")
    small_eps = BatchNorm2d(2, eps=1e-8, affine=False, dtype="float64")
    large_eps = BatchNorm2d(2, eps=1.0, affine=False, dtype="float64")
    y_small = small_eps(x).numpy()
    y_large = large_eps(x).numpy()
    assert not np.allclose(y_small, y_large)


# -- gradients: finite-difference validation ----------------------------------


def _finite_diff_grad(f, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.empty_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus, minus = x0.copy(), x0.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (f(plus) - f(minus)) / (2 * eps)
    return grad


def test_input_gradient_matches_finite_difference():
    x0 = np.random.default_rng(11).standard_normal((3, 2, 3, 3)).astype(np.float64)
    w0 = np.array([1.3, 0.7])
    b0 = np.array([0.1, -0.2])

    def loss_fn(x_arr):
        bn = BatchNorm2d(2, dtype="float64")
        bn.weight._data = w0.copy()
        bn.bias._data = b0.copy()
        y = bn(Tensor(x_arr, dtype="float64"))
        return (y * y).sum().numpy()

    bn = BatchNorm2d(2, dtype="float64")
    bn.weight._data = w0.copy()
    bn.bias._data = b0.copy()
    x = Tensor(x0, dtype="float64", requires_grad=True)
    y = bn(x)
    (y * y).sum().backward()

    numeric = _finite_diff_grad(loss_fn, x0)
    np.testing.assert_allclose(x.grad.numpy(), numeric, rtol=1e-4, atol=1e-5)


def test_weight_and_bias_gradients_match_finite_difference():
    x0 = np.random.default_rng(12).standard_normal((3, 2, 3, 3)).astype(np.float64)
    w0 = np.array([1.3, 0.7])
    b0 = np.array([0.1, -0.2])

    def loss_of(w_arr, b_arr):
        bn = BatchNorm2d(2, dtype="float64")
        bn.weight._data = w_arr.copy()
        bn.bias._data = b_arr.copy()
        y = bn(Tensor(x0, dtype="float64"))
        return (y * y).sum().numpy()

    bn = BatchNorm2d(2, dtype="float64")
    bn.weight._data = w0.copy()
    bn.bias._data = b0.copy()
    y = bn(Tensor(x0, dtype="float64"))
    (y * y).sum().backward()

    numeric_w = _finite_diff_grad(lambda w: loss_of(w, b0), w0)
    numeric_b = _finite_diff_grad(lambda b: loss_of(w0, b), b0)
    np.testing.assert_allclose(bn.weight.grad.numpy(), numeric_w, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(bn.bias.grad.numpy(), numeric_b, rtol=1e-4, atol=1e-5)


def test_non_affine_input_gradient_matches_finite_difference():
    x0 = np.random.default_rng(13).standard_normal((3, 2, 3, 3)).astype(np.float64)

    def loss_fn(x_arr):
        bn = BatchNorm2d(2, affine=False, dtype="float64")
        y = bn(Tensor(x_arr, dtype="float64"))
        return (y * y).sum().numpy()

    bn = BatchNorm2d(2, affine=False, dtype="float64")
    x = Tensor(x0, dtype="float64", requires_grad=True)
    y = bn(x)
    (y * y).sum().backward()

    numeric = _finite_diff_grad(loss_fn, x0)
    np.testing.assert_allclose(x.grad.numpy(), numeric, rtol=1e-4, atol=1e-5)


def test_no_gradient_flows_to_running_stats():
    bn = BatchNorm2d(2)
    x = Tensor(np.random.default_rng(14).standard_normal((4, 2, 3, 3)).astype(np.float32), requires_grad=True)
    bn(x).sum().backward()
    assert bn.running_mean.grad is None
    assert bn.running_var.grad is None


# -- repeated use / mode transitions ------------------------------------------


def test_repeated_forward_backward_accumulates_gradients_correctly():
    bn = BatchNorm2d(2)
    x = Tensor(np.random.default_rng(15).standard_normal((4, 2, 3, 3)).astype(np.float32), requires_grad=True)

    bn(x).sum().backward()
    first_weight_grad = bn.weight.grad.numpy().copy()
    x.grad = None
    bn(x).sum().backward()
    second_weight_grad = bn.weight.grad.numpy().copy()

    # zero_grad was not called between calls -- gradients accumulate.
    np.testing.assert_allclose(second_weight_grad, 2 * first_weight_grad, rtol=1e-4)


def test_train_then_eval_then_train_again_produces_consistent_shapes():
    bn = BatchNorm2d(2)
    x = Tensor(np.random.default_rng(16).standard_normal((4, 2, 3, 3)).astype(np.float32))
    y1 = bn(x)
    bn.eval()
    y2 = bn(x)
    bn.train()
    y3 = bn(x)
    assert y1.shape == y2.shape == y3.shape == x.shape


# -- integration: Sequential + Conv2d -----------------------------------------


def test_batchnorm2d_inside_sequential_with_conv2d():
    forge.random.seed(0)
    model = Sequential(Conv2d(1, 4, kernel_size=3), BatchNorm2d(4))
    x = Tensor(np.random.default_rng(17).standard_normal((3, 1, 8, 8)).astype(np.float32), requires_grad=True)
    y = model(x)
    assert y.shape == (3, 4, 6, 6)
    y.sum().backward()
    bn = model._modules["1"]
    assert bn.weight.grad is not None
    assert x.grad is not None


def test_model_train_eval_propagates_to_nested_batchnorm2d():
    model = Sequential(Conv2d(1, 4, kernel_size=3), BatchNorm2d(4))
    model.eval()
    assert model._modules["1"].training is False
    model.train()
    assert model._modules["1"].training is True
