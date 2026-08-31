import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import OptimizerError
from forge.nn import Linear, Parameter
from forge.optim import SGD


# -- construction / validation ------------------------------------------


def test_sgd_construction_stores_parameters():
    p1 = Parameter([1.0, 2.0])
    p2 = Parameter([3.0])
    opt = SGD([p1, p2], lr=0.1)
    assert opt.parameters == [p1, p2]
    assert opt.lr == 0.1


def test_sgd_accepts_generator_of_parameters():
    layer = Linear(3, 2)
    opt = SGD(layer.parameters(), lr=0.01)
    assert len(opt.parameters) == 2


@pytest.mark.parametrize("bad_lr", [0.0, -0.1, -1, float("nan")])
def test_sgd_rejects_invalid_learning_rate(bad_lr):
    p = Parameter([1.0])
    with pytest.raises(OptimizerError):
        SGD([p], lr=bad_lr)


def test_sgd_rejects_non_numeric_learning_rate():
    p = Parameter([1.0])
    with pytest.raises(OptimizerError):
        SGD([p], lr="0.1")


def test_sgd_accepts_positive_learning_rate():
    p = Parameter([1.0])
    opt = SGD([p], lr=1e-6)
    assert opt.lr == 1e-6


# -- parameter update ------------------------------------------------------


def test_sgd_step_matches_manual_calculation():
    p = Parameter([2.0])
    p.grad = Tensor([0.5])
    opt = SGD([p], lr=0.1)
    opt.step()
    np.testing.assert_allclose(p.numpy(), [1.95], rtol=1e-6, atol=1e-6)


def test_sgd_step_updates_multiple_parameters_independently():
    w = Parameter([[1.0, 2.0], [3.0, 4.0]])
    b = Parameter([0.5, -0.5])
    w.grad = Tensor([[1.0, 1.0], [1.0, 1.0]])
    b.grad = Tensor([2.0, 2.0])
    opt = SGD([w, b], lr=0.1)
    opt.step()
    np.testing.assert_allclose(w.numpy(), [[0.9, 1.9], [2.9, 3.9]], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.numpy(), [0.3, -0.7], rtol=1e-6, atol=1e-6)


def test_sgd_step_does_not_create_autograd_graph():
    p = Parameter([2.0])
    p.grad = Tensor([0.5])
    opt = SGD([p], lr=0.1)
    opt.step()
    assert p.grad_fn is None
    assert p.is_leaf is True


def test_sgd_step_leaves_parameter_without_gradient_unchanged():
    p = Parameter([2.0])
    assert p.grad is None
    opt = SGD([p], lr=0.1)
    opt.step()
    np.testing.assert_allclose(p.numpy(), [2.0], rtol=1e-6, atol=1e-6)


def test_sgd_step_skips_ungraded_parameter_but_updates_others():
    p1 = Parameter([2.0])  # no grad
    p2 = Parameter([2.0])
    p2.grad = Tensor([1.0])
    opt = SGD([p1, p2], lr=0.1)
    opt.step()
    np.testing.assert_allclose(p1.numpy(), [2.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(p2.numpy(), [1.9], rtol=1e-6, atol=1e-6)


# -- zero_grad ---------------------------------------------------------------


def test_zero_grad_clears_all_parameter_gradients():
    p1 = Parameter([1.0])
    p2 = Parameter([2.0])
    p1.grad = Tensor([1.0])
    p2.grad = Tensor([2.0])
    opt = SGD([p1, p2], lr=0.1)
    opt.zero_grad()
    assert p1.grad is None
    assert p2.grad is None


def test_zero_grad_is_safe_when_no_gradient_present():
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    opt.zero_grad()  # should not raise
    assert p.grad is None


# -- repeated steps ------------------------------------------------------


def test_repeated_steps_move_parameter_monotonically_toward_zero_grad():
    p = Parameter([10.0])
    opt = SGD([p], lr=0.1)
    for _ in range(5):
        p.grad = Tensor([1.0])  # constant positive gradient
        opt.step()
    # each step: p -= 0.1 * 1.0 -> after 5 steps, p = 10 - 0.5 = 9.5
    np.testing.assert_allclose(p.numpy(), [9.5], rtol=1e-6, atol=1e-6)


def test_repeated_zero_grad_step_cycle_on_real_forward_pass():
    layer = Linear(2, 1)
    opt = SGD(layer.parameters(), lr=0.1)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    initial_weight = layer.weight.numpy().copy()
    for _ in range(3):
        opt.zero_grad()
        y = layer(x).sum()
        y.backward()
        opt.step()

    assert not np.allclose(layer.weight.numpy(), initial_weight)


# -- base Optimizer ---------------------------------------------------------


def test_base_optimizer_step_raises():
    from forge.optim import Optimizer

    opt = Optimizer([Parameter([1.0])])
    with pytest.raises(OptimizerError):
        opt.step()
