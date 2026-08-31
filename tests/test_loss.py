import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import LossError
from forge.nn import CrossEntropyLoss, MSELoss

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference gradient of a scalar-valued fn(np.ndarray) -> float."""
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


# -- MSELoss: forward -----------------------------------------------------


def test_mse_forward_value():
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.5, 2.5, 2.5])
    loss = MSELoss()(pred, target)
    assert loss.shape == ()
    np.testing.assert_allclose(loss.numpy(), 0.25, **TOL)


def test_mse_forward_zero_when_equal():
    pred = Tensor([1.0, -2.0, 3.5])
    loss = MSELoss()(pred, pred)
    np.testing.assert_allclose(loss.numpy(), 0.0, **TOL)


def test_mse_accepts_raw_array_target():
    pred = Tensor([1.0, 2.0])
    loss = MSELoss()(pred, [0.0, 0.0])
    np.testing.assert_allclose(loss.numpy(), 2.5, **TOL)


def test_mse_batched_shape_averages_over_all_elements():
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[0.0, 0.0], [0.0, 0.0]])
    loss = MSELoss()(pred, target)
    # mean(1 + 4 + 9 + 16) = 30/4
    np.testing.assert_allclose(loss.numpy(), 7.5, **TOL)


# -- MSELoss: shape validation ----------------------------------------------


def test_mse_rejects_mismatched_shapes():
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.0, 2.0])
    with pytest.raises(LossError):
        MSELoss()(pred, target)


# -- MSELoss: gradients -----------------------------------------------------


def test_mse_gradient_analytical():
    pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 2.5])
    loss = MSELoss()(pred, target)
    loss.backward()
    # d(mean((p-t)^2))/dp = 2*(p-t)/n
    expected = 2 * (np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 2.5])) / 3
    np.testing.assert_allclose(pred.grad.numpy(), expected, **TOL)


def test_mse_gradient_matches_finite_difference():
    pred_data = np.array([0.3, -1.2, 2.7, 0.1])
    target_data = np.array([0.0, 1.0, 2.0, -0.5])

    def fn(p):
        return float(np.mean((p - target_data) ** 2))

    pred = Tensor(pred_data.copy(), requires_grad=True)
    loss = MSELoss()(pred, Tensor(target_data))
    loss.backward()

    expected = numerical_grad(fn, pred_data.copy())
    np.testing.assert_allclose(pred.grad.numpy(), expected, **FD_TOL)


# -- CrossEntropyLoss: forward -----------------------------------------------


def test_cross_entropy_forward_value():
    logits = Tensor([[1.0, 2.0, 3.0]])
    target = np.array([2])
    loss = CrossEntropyLoss()(logits, target)

    exp_logits = np.exp([1.0, 2.0, 3.0])
    probs = exp_logits / exp_logits.sum()
    expected = -np.log(probs[2])
    np.testing.assert_allclose(loss.numpy(), expected, **TOL)


def test_cross_entropy_uniform_logits_gives_log_num_classes():
    logits = Tensor([[0.0, 0.0, 0.0, 0.0]])
    loss = CrossEntropyLoss()(logits, np.array([1]))
    np.testing.assert_allclose(loss.numpy(), np.log(4), **TOL)


def test_cross_entropy_batched_forward_matches_manual_mean():
    logits_data = np.array([[2.0, 1.0, 0.1], [0.1, 0.2, 3.0]])
    targets = np.array([0, 2])

    def manual_ce(logits, targets):
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(targets)), targets].mean()

    loss = CrossEntropyLoss()(Tensor(logits_data), targets)
    np.testing.assert_allclose(loss.numpy(), manual_ce(logits_data, targets), **TOL)


# -- CrossEntropyLoss: target/shape validation -------------------------------


def test_cross_entropy_rejects_non_2d_logits():
    logits = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(LossError):
        CrossEntropyLoss()(logits, np.array([0]))


def test_cross_entropy_rejects_wrong_target_length():
    logits = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(LossError):
        CrossEntropyLoss()(logits, np.array([0, 1, 0]))


def test_cross_entropy_rejects_non_integer_target_dtype():
    logits = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(LossError):
        CrossEntropyLoss()(logits, np.array([0.0, 1.0]))


# -- CrossEntropyLoss: class-index validation --------------------------------


def test_cross_entropy_rejects_negative_class_index():
    logits = Tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(LossError):
        CrossEntropyLoss()(logits, np.array([-1]))


def test_cross_entropy_rejects_out_of_range_class_index():
    logits = Tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(LossError):
        CrossEntropyLoss()(logits, np.array([3]))


# -- CrossEntropyLoss: numerical stability -----------------------------------


def test_cross_entropy_stable_for_large_logits():
    logits = Tensor([[1000.0, 1001.0, 1002.0]])
    loss = CrossEntropyLoss()(logits, np.array([2]))
    assert np.isfinite(loss.numpy())
    # shift-invariant: same as small logits with the same differences
    small = Tensor([[0.0, 1.0, 2.0]])
    small_loss = CrossEntropyLoss()(small, np.array([2]))
    np.testing.assert_allclose(loss.numpy(), small_loss.numpy(), **TOL)


def test_cross_entropy_stable_for_very_negative_logits():
    logits = Tensor([[-1000.0, -999.0, -998.0]])
    loss = CrossEntropyLoss()(logits, np.array([0]))
    assert np.isfinite(loss.numpy())


# -- CrossEntropyLoss: gradients ---------------------------------------------


def test_cross_entropy_gradient_matches_softmax_minus_onehot():
    logits_data = np.array([[2.0, 1.0, 0.1]])
    targets = np.array([0])
    logits = Tensor(logits_data.copy(), requires_grad=True)
    loss = CrossEntropyLoss()(logits, targets)
    loss.backward()

    shifted = logits_data - logits_data.max(axis=1, keepdims=True)
    softmax = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(softmax)
    one_hot[0, targets[0]] = 1.0
    expected = (softmax - one_hot) / len(targets)
    np.testing.assert_allclose(logits.grad.numpy(), expected, **TOL)


def test_cross_entropy_gradient_matches_finite_difference():
    logits_data = np.array([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    targets = np.array([1, 0])

    def fn(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(len(targets)), targets].mean())

    logits = Tensor(logits_data.copy(), requires_grad=True)
    loss = CrossEntropyLoss()(logits, targets)
    loss.backward()

    expected = numerical_grad(fn, logits_data.copy())
    np.testing.assert_allclose(logits.grad.numpy(), expected, **FD_TOL)


# -- base Loss ---------------------------------------------------------------


def test_base_loss_forward_raises():
    from forge.nn import Loss

    with pytest.raises(LossError):
        Loss()(Tensor([1.0]), Tensor([1.0]))
