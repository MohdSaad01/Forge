"""`Tensor.tanh()` (Milestone 50): CPU forward/backward correctness.

Added as the one genuine blocker `nn.RNNCell` (`examples/char_rnn/`)
exposed -- see `docs/development/m50-char-rnn.md`. Mirrors the existing
`relu`/`exp` Tensor-primitive test shape (no dedicated CPU-only `exp`/`log`
test file exists in this repo -- those are exercised indirectly via
`CrossEntropyLoss`; `tanh` gets its own file since it has no such indirect
consumer at the Tensor level).
"""

from __future__ import annotations

import numpy as np

from forge import Tensor

TOL = dict(rtol=1e-6, atol=1e-6)


def test_tanh_forward_matches_numpy():
    x = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    result = x.tanh()
    np.testing.assert_allclose(result.numpy(), np.tanh([-2.0, -0.5, 0.0, 0.5, 2.0]), **TOL)


def test_tanh_forward_2d():
    data = [[-1.0, 0.0], [1.0, 3.0]]
    result = Tensor(data).tanh()
    np.testing.assert_allclose(result.numpy(), np.tanh(data), **TOL)


def test_tanh_result_is_leaf_when_input_does_not_require_grad():
    x = Tensor([1.0, -1.0])
    y = x.tanh()
    assert y.requires_grad is False
    assert y.grad_fn is None


def test_tanh_backward_matches_analytic_derivative():
    x = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    y = x.tanh().sum()
    y.backward()
    expected = 1 - np.tanh([-2.0, -0.5, 0.0, 0.5, 2.0]) ** 2
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


def test_tanh_backward_scales_with_upstream_gradient():
    x = Tensor([0.0, 1.0], requires_grad=True)
    y = x.tanh()
    y.backward(Tensor([2.0, 3.0]))
    expected = np.array([2.0, 3.0]) * (1 - np.tanh([0.0, 1.0]) ** 2)
    np.testing.assert_allclose(x.grad.numpy(), expected, **TOL)


def test_tanh_backward_matches_finite_difference():
    eps = 1e-5
    x0 = np.array([-1.3, 0.2, 0.9])

    x = Tensor(x0, dtype="float64", requires_grad=True)
    x.tanh().sum().backward()
    analytic = x.grad.numpy()

    numeric = np.empty_like(x0)
    for i in range(len(x0)):
        plus = x0.copy()
        plus[i] += eps
        minus = x0.copy()
        minus[i] -= eps
        numeric[i] = (np.tanh(plus).sum() - np.tanh(minus).sum()) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_tanh_saturates_towards_plus_and_minus_one():
    x = Tensor([-100.0, 100.0])
    result = x.tanh().numpy()
    np.testing.assert_allclose(result, [-1.0, 1.0], atol=1e-9)
