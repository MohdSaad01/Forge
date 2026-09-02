import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import OptimizerError
from forge.nn import Linear, Parameter
from forge.optim import SGD, Adam


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


# ============================================================================
# Adam (Milestone 17)
# ============================================================================


def _adam_reference(theta, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """A small NumPy-only reference Adam implementation (float64 throughout).

    Mirrors the exact update Forge's `Adam` is specified to perform. Used to
    verify Forge's CPU/CUDA Adam without depending on an external ML
    framework at test time.
    """
    theta = np.array(theta, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t, g in enumerate(grads, start=1):
        g = np.array(g, dtype=np.float64)
        if weight_decay != 0.0:
            g = g + weight_decay * theta
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta


# -- construction / validation ------------------------------------------


def test_adam_construction_defaults():
    p = Parameter([1.0])
    opt = Adam([p])
    assert opt.parameters == [p]
    assert opt.lr == 1e-3
    assert opt.beta1 == 0.9
    assert opt.beta2 == 0.999
    assert opt.eps == 1e-8
    assert opt.weight_decay == 0.0
    assert opt.state == {}


def test_adam_construction_custom_hyperparameters():
    p = Parameter([1.0])
    opt = Adam([p], lr=0.01, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.001)
    assert opt.lr == 0.01
    assert opt.beta1 == 0.8
    assert opt.beta2 == 0.99
    assert opt.eps == 1e-6
    assert opt.weight_decay == 0.001


def test_adam_accepts_generator_of_parameters():
    layer = Linear(3, 2)
    opt = Adam(layer.parameters())
    assert len(opt.parameters) == 2


@pytest.mark.parametrize("bad_lr", [0.0, -0.1, -1, float("nan")])
def test_adam_rejects_invalid_learning_rate(bad_lr):
    with pytest.raises(OptimizerError):
        Adam([Parameter([1.0])], lr=bad_lr)


@pytest.mark.parametrize("bad_betas", [(1.0, 0.9), (0.9, 1.0), (-0.1, 0.9), (0.9, -0.1), (float("nan"), 0.9)])
def test_adam_rejects_invalid_betas(bad_betas):
    with pytest.raises(OptimizerError):
        Adam([Parameter([1.0])], betas=bad_betas)


def test_adam_rejects_malformed_betas():
    with pytest.raises(OptimizerError):
        Adam([Parameter([1.0])], betas=(0.9,))
    with pytest.raises(OptimizerError):
        Adam([Parameter([1.0])], betas="0.9,0.999")


@pytest.mark.parametrize("bad_eps", [0.0, -1e-8, float("nan")])
def test_adam_rejects_invalid_eps(bad_eps):
    with pytest.raises(OptimizerError):
        Adam([Parameter([1.0])], eps=bad_eps)


@pytest.mark.parametrize("bad_wd", [-0.1, -1, float("nan")])
def test_adam_rejects_invalid_weight_decay(bad_wd):
    with pytest.raises(OptimizerError):
        Adam([Parameter([1.0])], weight_decay=bad_wd)


def test_adam_accepts_zero_weight_decay_boundary():
    Adam([Parameter([1.0])], weight_decay=0.0)  # boundary; should not raise
    Adam([Parameter([1.0])], betas=(0.0, 0.0))  # boundary; should not raise


# -- CPU reference correctness -----------------------------------------------


def test_adam_first_step_matches_reference():
    p = Parameter([1.0, 2.0])
    p.grad = Tensor([0.1, -0.2])
    opt = Adam([p], lr=0.1)
    opt.step()
    expected = _adam_reference([1.0, 2.0], [[0.1, -0.2]], lr=0.1)
    np.testing.assert_allclose(p.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_adam_multiple_steps_match_reference():
    theta0 = [1.0, 2.0]
    grads = [[0.1, -0.2], [0.05, 0.3], [-0.1, 0.1], [0.2, -0.05]]
    p = Parameter(theta0)
    opt = Adam([p], lr=0.05)
    for g in grads:
        p.grad = Tensor(g)
        opt.step()
    expected = _adam_reference(theta0, grads, lr=0.05)
    np.testing.assert_allclose(p.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_adam_weight_decay_matches_reference():
    theta0 = [1.0, -2.0, 0.5]
    grads = [[0.1, 0.2, -0.1], [0.05, -0.1, 0.2], [-0.05, 0.05, 0.05]]
    p = Parameter(theta0)
    opt = Adam([p], lr=0.02, weight_decay=0.1)
    for g in grads:
        p.grad = Tensor(g)
        opt.step()
    expected = _adam_reference(theta0, grads, lr=0.02, weight_decay=0.1)
    np.testing.assert_allclose(p.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_adam_bias_correction_makes_first_step_larger_than_late_steps():
    """With a constant gradient, bias correction inflates early m_hat/v_hat,
    so the first-step update magnitude is not simply lr (uncorrected m_hat
    would give exactly `lr` on step 1); this checks the step-1 update
    matches the exact bias-corrected formula rather than an uncorrected one.
    """
    p = Parameter([0.0])
    p.grad = Tensor([1.0])
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    opt.step()
    # m_hat = 1.0 (bias-corrected first moment of a constant grad of 1 is exact)
    # v_hat = 1.0 similarly -> update = lr * 1 / (1 + eps) ~= lr
    np.testing.assert_allclose(p.numpy(), [-0.1], rtol=1e-4, atol=1e-6)


def test_adam_state_accumulates_across_steps():
    p = Parameter([0.0])
    opt = Adam([p], lr=0.1)
    p.grad = Tensor([1.0])
    opt.step()
    state = opt.state[p]
    assert state.step == 1
    m_after_1, v_after_1 = state.m.copy(), state.v.copy()

    p.grad = Tensor([1.0])
    opt.step()
    state = opt.state[p]
    assert state.step == 2
    assert not np.allclose(state.m, m_after_1)
    assert not np.allclose(state.v, v_after_1)


def test_adam_zero_gradient_leaves_parameter_unchanged():
    p = Parameter([3.0])
    opt = Adam([p], lr=0.1)
    p.grad = Tensor([0.0])
    opt.step()
    np.testing.assert_allclose(p.numpy(), [3.0], rtol=1e-6, atol=1e-6)


def test_adam_missing_gradient_is_skipped_and_allocates_no_state():
    p = Parameter([2.0])
    opt = Adam([p], lr=0.1)
    assert p.grad is None
    opt.step()
    np.testing.assert_allclose(p.numpy(), [2.0], rtol=1e-6, atol=1e-6)
    assert p not in opt.state


def test_adam_skips_ungraded_parameter_but_updates_others():
    p1 = Parameter([2.0])  # no grad
    p2 = Parameter([2.0])
    p2.grad = Tensor([1.0])
    opt = Adam([p1, p2], lr=0.1)
    opt.step()
    np.testing.assert_allclose(p1.numpy(), [2.0], rtol=1e-6, atol=1e-6)
    assert p1 not in opt.state
    assert p2 in opt.state
    expected = _adam_reference([2.0], [[1.0]], lr=0.1)
    np.testing.assert_allclose(p2.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_adam_updates_multiple_parameters_independently():
    w = Parameter([[1.0, 2.0], [3.0, 4.0]])
    b = Parameter([0.5, -0.5])
    w.grad = Tensor([[1.0, 1.0], [1.0, 1.0]])
    b.grad = Tensor([2.0, 2.0])
    opt = Adam([w, b], lr=0.1)
    opt.step()
    expected_w = _adam_reference([[1.0, 2.0], [3.0, 4.0]], [[[1.0, 1.0], [1.0, 1.0]]], lr=0.1)
    expected_b = _adam_reference([0.5, -0.5], [[2.0, 2.0]], lr=0.1)
    np.testing.assert_allclose(w.numpy(), expected_w, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(b.numpy(), expected_b, rtol=1e-5, atol=1e-6)


# -- gradient/state validation ------------------------------------------------


def test_adam_rejects_gradient_shape_mismatch():
    p = Parameter([1.0, 2.0])
    p.grad = Tensor([1.0])
    opt = Adam([p])
    with pytest.raises(OptimizerError):
        opt.step()


def test_adam_rejects_gradient_dtype_mismatch():
    p = Parameter([1.0, 2.0])  # float32 by default
    p.grad = Tensor([1.0, 2.0], dtype="float64")
    opt = Adam([p])
    with pytest.raises(OptimizerError):
        opt.step()


# -- state identity / no autograd graph ---------------------------------------


def test_adam_step_does_not_create_autograd_graph():
    p = Parameter([2.0])
    p.grad = Tensor([0.5])
    opt = Adam([p], lr=0.1)
    opt.step()
    assert p.grad_fn is None
    assert p.is_leaf is True


def test_adam_state_keyed_by_parameter_identity_not_name():
    """State follows the Parameter object even if reachable under a
    different attribute name / module hierarchy, per the milestone spec."""
    p = Parameter([1.0])
    opt = Adam([p], lr=0.1)
    p.grad = Tensor([1.0])
    opt.step()
    assert p in opt.state

    class Wrapper:
        pass

    holder = Wrapper()
    holder.renamed_param = p  # same object, different name/context
    assert holder.renamed_param in opt.state
    assert opt.state[holder.renamed_param] is opt.state[p]


def test_adam_preserves_parameter_object_identity_across_steps():
    layer = Linear(2, 2)
    opt = Adam(layer.parameters(), lr=0.01)
    weight_id = id(layer.weight)
    x = Tensor([[1.0, 2.0]])
    for _ in range(3):
        opt.zero_grad()
        y = layer(x).sum()
        y.backward()
        opt.step()
    assert id(layer.weight) == weight_id


def test_adam_state_uses_parameter_shape_and_dtype():
    p = Parameter([[1.0, 2.0], [3.0, 4.0]])
    p.grad = Tensor([[1.0, 1.0], [1.0, 1.0]])
    opt = Adam([p], lr=0.1)
    opt.step()
    state = opt.state[p]
    assert state.m.shape == p.shape
    assert state.v.shape == p.shape
    assert state.m.dtype == p.numpy().dtype
    assert state.v.dtype == p.numpy().dtype


# -- SGD compatibility ---------------------------------------------------------


def test_sgd_unaffected_by_adam_addition():
    p = Parameter([2.0])
    p.grad = Tensor([0.5])
    opt = SGD([p], lr=0.1)
    opt.step()
    np.testing.assert_allclose(p.numpy(), [1.95], rtol=1e-6, atol=1e-6)


# -- end-to-end CPU training ---------------------------------------------------


def test_adam_end_to_end_cpu_regression_training_decreases_loss():
    from forge.nn.loss import MSELoss

    forge.random.seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(32, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
    x_t = Tensor(X)
    y_t = Tensor(y)

    model = Linear(2, 1)
    loss_fn = MSELoss()
    opt = Adam(model.parameters(), lr=0.05)

    initial_weight = model.weight.numpy().copy()
    losses = []
    for _ in range(50):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.numpy()))

    assert losses[-1] < losses[0]
    assert not np.allclose(model.weight.numpy(), initial_weight)


def test_adam_deterministic_under_fixed_seed():
    def run():
        forge.random.seed(123)
        model = Linear(2, 1)
        opt = Adam(model.parameters(), lr=0.05)
        x = Tensor([[1.0, -1.0], [0.5, 0.5]])
        for _ in range(5):
            opt.zero_grad()
            y = model(x).sum()
            y.backward()
            opt.step()
        return model.weight.numpy().copy()

    w1 = run()
    w2 = run()
    np.testing.assert_allclose(w1, w2, rtol=1e-7, atol=1e-7)


# -- Trainer integration --------------------------------------------------------


def test_trainer_works_with_adam_without_modification():
    from forge.data import DataLoader, TensorDataset
    from forge.nn.loss import MSELoss
    from forge.training import Trainer

    forge.random.seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(16, 2))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=4)

    model = Linear(2, 1)
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=optimizer, device="cpu", verbose=False)

    history = trainer.fit(loader, epochs=5)
    assert history.train_losses[-1] < history.train_losses[0]


# -- model persistence isolation -------------------------------------------------


def test_save_model_does_not_serialize_optimizer_state(tmp_path):
    layer = Linear(2, 2)
    opt = Adam(layer.parameters(), lr=0.01)
    x = Tensor([[1.0, 2.0]])
    opt.zero_grad()
    y = layer(x).sum()
    y.backward()
    opt.step()
    assert len(opt.state) == 2  # weight + bias both received gradients

    path = str(tmp_path / "model.forge")
    forge.save_model(layer, path)

    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert not any("adam" in n.lower() or "state" in n.lower() for n in names)

    loaded = forge.load_model(path)
    np.testing.assert_allclose(loaded.weight.numpy(), layer.weight.numpy(), rtol=1e-6, atol=1e-6)
