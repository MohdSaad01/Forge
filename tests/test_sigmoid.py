"""`Tensor.sigmoid()` (Milestone 67): CPU forward/backward correctness.

Added as the smallest primitive needed for `nn.LSTMCell`'s input/forget/
output gates -- see `docs/development/m67-lstm-long-range-recall.md`.
Mirrors the existing `tanh` Tensor-primitive test shape (`tests/test_tanh.py`).
"""

from __future__ import annotations

import numpy as np

from forge import Tensor

TOL = dict(rtol=1e-6, atol=1e-6)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def test_sigmoid_forward_matches_reference():
    x = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    result = x.sigmoid()
    np.testing.assert_allclose(result.numpy(), _sigmoid_np([-2.0, -0.5, 0.0, 0.5, 2.0]), **TOL)


def test_sigmoid_forward_2d():
    data = [[-1.0, 0.0], [1.0, 3.0]]
    result = Tensor(data).sigmoid()
    np.testing.assert_allclose(result.numpy(), _sigmoid_np(data), **TOL)


def test_sigmoid_result_is_leaf_when_input_does_not_require_grad():
    x = Tensor([1.0, -1.0])
    y = x.sigmoid()
    assert y.requires_grad is False
    assert y.grad_fn is None


def test_sigmoid_backward_matches_analytic_derivative():
    x = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    y = x.sigmoid().sum()
    y.backward()
    s = _sigmoid_np([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = s * (1 - s)
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


def test_sigmoid_backward_scales_with_upstream_gradient():
    x = Tensor([0.0, 1.0], requires_grad=True)
    y = x.sigmoid()
    y.backward(Tensor([2.0, 3.0]))
    s = _sigmoid_np([0.0, 1.0])
    expected = np.array([2.0, 3.0]) * s * (1 - s)
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


def test_sigmoid_backward_matches_finite_difference():
    eps = 1e-5
    x0 = np.array([-1.3, 0.2, 0.9])

    x = Tensor(x0, dtype="float64", requires_grad=True)
    x.sigmoid().sum().backward()
    analytic = x.grad.numpy()

    numeric = np.empty_like(x0)
    for i in range(len(x0)):
        plus = x0.copy()
        plus[i] += eps
        minus = x0.copy()
        minus[i] -= eps
        numeric[i] = (_sigmoid_np(plus).sum() - _sigmoid_np(minus).sum()) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_sigmoid_saturates_towards_zero_and_one():
    x = Tensor([-100.0, 100.0])
    result = x.sigmoid().numpy()
    np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-9)


def test_sigmoid_stays_finite_for_large_magnitude_inputs():
    """Large-magnitude inputs must not overflow through the naive `exp(-x)`
    formula -- see `CPUBackend.sigmoid`'s numerically-stable per-sign branch."""
    x = Tensor([-1000.0, 1000.0])
    result = x.sigmoid().numpy()
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)
