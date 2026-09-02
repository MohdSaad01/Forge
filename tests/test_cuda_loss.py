"""CUDA loss tests (Milestone 12; `CrossEntropyLoss` CUDA support in Milestone 14).

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py`/`tests/test_cuda_consistency.py`/
`tests/test_cuda_autograd.py`/`tests/test_module_cuda.py`. `MSELoss` is
CUDA-compatible as of Milestone 12 -- it composes only ops CUDA already
implements forward *and* backward (`-`, `*`, `.sum()`), so no new CUDA
kernel was needed (see `docs/architecture/cuda-backend.md`'s **CUDA
losses** section). `CrossEntropyLoss` is CUDA-compatible as of Milestone 14
(see that same document's **CUDA CrossEntropyLoss** section): this file's
CrossEntropyLoss tests cover CUDA/CPU forward agreement (including
numerically difficult logits), CUDA backward correctness (analytic vs. CPU,
and vs. finite differences), reduction semantics, device validation, and the
absence of any CPU computational fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import is_cuda_available
from forge.exceptions import LossError, UnsupportedDeviceError
from forge.nn import CrossEntropyLoss, MSELoss

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-5, atol=1e-5)
LOOSE_TOL = dict(rtol=1e-4, atol=1e-4)


# -- MSELoss: CUDA forward correctness ---------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_mse_cuda_forward_matches_cpu(dtype):
    pred_data = [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]]
    target_data = [[1.5, 2.5, 2.5], [3.0, 0.0, 1.0]]

    pred_cpu = Tensor(pred_data, dtype=dtype)
    target_cpu = Tensor(target_data, dtype=dtype)
    pred_cuda = pred_cpu.to("cuda")
    target_cuda = target_cpu.to("cuda")

    cpu_loss = MSELoss()(pred_cpu, target_cpu)
    cuda_loss = MSELoss()(pred_cuda, target_cuda)

    assert cuda_loss.device.type == "cuda"
    assert cuda_loss.shape == ()
    np.testing.assert_allclose(cuda_loss.to("cpu").numpy(), cpu_loss.numpy(), **TOL)


def test_mse_cuda_forward_zero_when_equal():
    pred = Tensor([1.0, -2.0, 3.5]).to("cuda")
    loss = MSELoss()(pred, pred)
    np.testing.assert_allclose(loss.to("cpu").numpy(), 0.0, **TOL)


def test_mse_cuda_accepts_raw_array_target():
    """A non-Tensor target is constructed fresh on the prediction's device --
    never a silent transfer of an existing device tensor (see the module's
    **Loss device validation** requirement)."""
    pred = Tensor([1.0, 2.0]).to("cuda")
    loss = MSELoss()(pred, [0.0, 0.0])
    assert loss.device.type == "cuda"
    np.testing.assert_allclose(loss.to("cpu").numpy(), 2.5, **TOL)


# -- MSELoss: CUDA backward correctness / gradient residency ----------------


def test_mse_cuda_backward_gradient_matches_cpu_and_is_cuda_resident():
    pred_data = np.array([0.3, -1.2, 2.7, 0.1], dtype=np.float32)
    target_data = np.array([0.0, 1.0, 2.0, -0.5], dtype=np.float32)

    pred_cpu = Tensor(pred_data.copy(), requires_grad=True)
    MSELoss()(pred_cpu, Tensor(target_data.copy())).backward()

    pred_cuda = Tensor(pred_data.copy(), device="cuda", requires_grad=True)
    target_cuda = Tensor(target_data.copy(), device="cuda")
    cuda_loss = MSELoss()(pred_cuda, target_cuda)
    cuda_loss.backward()

    assert pred_cuda.grad is not None
    assert pred_cuda.grad.device.type == "cuda"
    np.testing.assert_allclose(pred_cuda.grad.to("cpu").numpy(), pred_cpu.grad.numpy(), **TOL)


def test_mse_cuda_backward_analytical():
    pred = Tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
    target = Tensor([1.5, 2.5, 2.5], device="cuda")
    loss = MSELoss()(pred, target)
    loss.backward()
    expected = 2 * (np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 2.5])) / 3
    np.testing.assert_allclose(pred.grad.to("cpu").numpy(), expected, **TOL)


def test_mse_cuda_batched_backward_matches_cpu():
    rng = np.random.default_rng(0)
    pred_data = rng.standard_normal((5, 3)).astype(np.float32)
    target_data = rng.standard_normal((5, 3)).astype(np.float32)

    pred_cpu = Tensor(pred_data.copy(), requires_grad=True)
    MSELoss()(pred_cpu, Tensor(target_data.copy())).backward()

    pred_cuda = Tensor(pred_data.copy(), device="cuda", requires_grad=True)
    MSELoss()(pred_cuda, Tensor(target_data.copy(), device="cuda")).backward()

    np.testing.assert_allclose(pred_cuda.grad.to("cpu").numpy(), pred_cpu.grad.numpy(), **TOL)


# -- MSELoss: device validation ----------------------------------------------


def test_mse_rejects_cuda_prediction_with_cpu_target():
    pred = Tensor([1.0, 2.0, 3.0]).to("cuda")
    target = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(UnsupportedDeviceError):
        MSELoss()(pred, target)


def test_mse_rejects_cpu_prediction_with_cuda_target():
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.0, 2.0, 3.0]).to("cuda")
    with pytest.raises(UnsupportedDeviceError):
        MSELoss()(pred, target)


# -- MSELoss: no CPU fallback -------------------------------------------------


def test_mse_cuda_forward_and_backward_never_call_cpu_backend(monkeypatch):
    pred = Tensor([1.0, 2.0, 3.0, -4.0], device="cuda", requires_grad=True)
    target = Tensor([1.5, 2.5, 2.5, -3.0], device="cuda")

    calls: list[str] = []
    for name in ("add", "sub", "mul", "matmul", "sum", "reshape", "relu", "exp", "log"):
        original = getattr(CPUBackend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    loss = MSELoss()(pred, target)
    loss.backward()

    assert calls == []
    assert loss.device.type == "cuda"
    assert pred.grad.device.type == "cuda"


# -- CrossEntropyLoss: CUDA forward correctness (Milestone 14) ---------------


def test_cross_entropy_still_works_on_cpu_after_moving_explicitly():
    logits = Tensor([[1.0, 2.0, 3.0]], device="cuda").to("cpu")
    loss = CrossEntropyLoss()(logits, np.array([2]))
    assert loss.device.type == "cpu"
    assert np.isfinite(loss.numpy())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cross_entropy_cuda_forward_matches_cpu(dtype):
    logits_data = [[1.0, 2.0, 3.0], [0.5, -1.0, 0.2], [4.0, 4.0, -2.0]]
    target = np.array([2, 0, 1])

    logits_cpu = Tensor(logits_data, dtype=dtype)
    logits_cuda = logits_cpu.to("cuda")

    cpu_loss = CrossEntropyLoss()(logits_cpu, target)
    cuda_loss = CrossEntropyLoss()(logits_cuda, target)

    assert cuda_loss.device.type == "cuda"
    assert cuda_loss.shape == ()
    np.testing.assert_allclose(cuda_loss.to("cpu").numpy(), cpu_loss.numpy(), **TOL)


def test_cross_entropy_cuda_accepts_target_as_cuda_tensor():
    logits = Tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.2]]).to("cuda")
    target = Tensor([2, 0], dtype="int64").to("cuda")
    loss = CrossEntropyLoss()(logits, target)
    assert loss.device.type == "cuda"
    expected = CrossEntropyLoss()(logits.to("cpu"), np.array([2, 0]))
    np.testing.assert_allclose(loss.to("cpu").numpy(), expected.numpy(), **TOL)


@pytest.mark.parametrize(
    "logits_data,target",
    [
        pytest.param([[1000.0, 1.0, -1000.0]], [0], id="large_positive"),
        pytest.param([[-1000.0, -999.0, -1000.0]], [1], id="large_negative"),
        pytest.param([[0.0, 500.0, -500.0]], [1], id="large_difference"),
        pytest.param([[3.0, 3.0, 3.0, 3.0]], [2], id="repeated_equal_logits"),
        pytest.param([[1.0, -1.0]], [0], id="single_sample_batch"),
        pytest.param(
            [[0.1, -0.2, 5.0, -3.0, 2.2, -1.1, 0.0]], [4], id="many_classes"
        ),
    ],
)
def test_cross_entropy_cuda_numerically_difficult_logits_match_cpu(logits_data, target):
    target = np.array(target)
    logits_cpu = Tensor(logits_data)
    logits_cuda = logits_cpu.to("cuda")

    cpu_loss = CrossEntropyLoss()(logits_cpu, target)
    cuda_loss = CrossEntropyLoss()(logits_cuda, target)

    assert np.isfinite(cuda_loss.to("cpu").numpy())
    np.testing.assert_allclose(cuda_loss.to("cpu").numpy(), cpu_loss.numpy(), **LOOSE_TOL)


# -- CrossEntropyLoss: CUDA backward correctness / gradient residency --------


def test_cross_entropy_cuda_backward_matches_cpu_and_is_cuda_resident():
    logits_data = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.2], [4.0, 4.0, -2.0]], dtype=np.float32)
    target = np.array([2, 0, 1])

    logits_cpu = Tensor(logits_data.copy(), requires_grad=True)
    CrossEntropyLoss()(logits_cpu, target).backward()

    logits_cuda = Tensor(logits_data.copy(), device="cuda", requires_grad=True)
    CrossEntropyLoss()(logits_cuda, target).backward()

    assert logits_cuda.grad is not None
    assert logits_cuda.grad.device.type == "cuda"
    np.testing.assert_allclose(logits_cuda.grad.to("cpu").numpy(), logits_cpu.grad.numpy(), **TOL)


def test_cross_entropy_cuda_backward_analytical_matches_softmax_minus_one_hot():
    """Backward should equal `(softmax(logits) - one_hot(target)) / batch_size`."""
    logits_data = np.array([[2.0, 1.0, 0.1], [0.0, 1.0, 2.0]], dtype=np.float32)
    target = np.array([0, 2])

    logits = Tensor(logits_data.copy(), device="cuda", requires_grad=True)
    CrossEntropyLoss()(logits, target).backward()

    exp = np.exp(logits_data - logits_data.max(axis=1, keepdims=True))
    softmax = exp / exp.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(logits_data)
    one_hot[np.arange(2), target] = 1.0
    expected = (softmax - one_hot) / 2

    np.testing.assert_allclose(logits.grad.to("cpu").numpy(), expected, **TOL)


@pytest.mark.parametrize("batch_size,num_classes", [(1, 3), (4, 2), (5, 6)])
def test_cross_entropy_cuda_backward_finite_difference_check(batch_size, num_classes):
    rng = np.random.default_rng(42)
    logits_data = rng.standard_normal((batch_size, num_classes)).astype(np.float64)
    target = rng.integers(0, num_classes, size=batch_size)

    logits = Tensor(logits_data.copy(), dtype="float64", device="cuda", requires_grad=True)
    CrossEntropyLoss()(logits, target).backward()
    analytical = logits.grad.to("cpu").numpy()

    def loss_value(data):
        return float(CrossEntropyLoss()(Tensor(data, dtype="float64"), target).numpy())

    eps = 1e-5
    numerical = np.zeros_like(logits_data)
    it = np.nditer(logits_data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus, minus = logits_data.copy(), logits_data.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numerical[idx] = (loss_value(plus) - loss_value(minus)) / (2 * eps)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-3)


# -- CrossEntropyLoss: reduction semantics (mean, not sum) -------------------


def test_cross_entropy_cuda_reduction_is_mean_matching_cpu():
    logits_data = [[1.0, 2.0, 3.0], [0.5, -1.0, 0.2], [4.0, 4.0, -2.0], [0.0, 0.0, 1.0]]
    target = np.array([2, 0, 1, 2])

    logits_cuda = Tensor(logits_data).to("cuda")
    loss = CrossEntropyLoss()(logits_cuda, target).to("cpu").numpy()

    per_example = []
    for row, t in zip(logits_data, target):
        per_example.append(float(CrossEntropyLoss()(Tensor([row]), np.array([t])).numpy()))
    np.testing.assert_allclose(loss, np.mean(per_example), **TOL)


def test_cross_entropy_cuda_backward_scales_by_one_over_batch_size():
    """Doubling the batch (by duplicating rows) should halve each row's gradient contribution."""
    logits_data = np.array([[1.0, 2.0, 0.5], [0.2, -0.3, 1.1]], dtype=np.float32)
    target = np.array([1, 0])

    small_logits = Tensor(logits_data.copy(), device="cuda", requires_grad=True)
    CrossEntropyLoss()(small_logits, target).backward()

    doubled_data = np.concatenate([logits_data, logits_data], axis=0)
    doubled_target = np.concatenate([target, target])
    big_logits = Tensor(doubled_data.copy(), device="cuda", requires_grad=True)
    CrossEntropyLoss()(big_logits, doubled_target).backward()

    small_grad = small_logits.grad.to("cpu").numpy()
    big_grad = big_logits.grad.to("cpu").numpy()
    np.testing.assert_allclose(big_grad[:2], small_grad / 2, **TOL)
    np.testing.assert_allclose(big_grad[2:], small_grad / 2, **TOL)


# -- CrossEntropyLoss: device validation --------------------------------------


def test_cross_entropy_rejects_cuda_logits_with_cpu_target_tensor():
    logits = Tensor([[1.0, 2.0, 3.0]]).to("cuda")
    target = Tensor([2], dtype="int64")
    with pytest.raises(UnsupportedDeviceError):
        CrossEntropyLoss()(logits, target)


def test_cross_entropy_rejects_cpu_logits_with_cuda_target_tensor():
    logits = Tensor([[1.0, 2.0, 3.0]])
    target = Tensor([2], dtype="int64").to("cuda")
    with pytest.raises(UnsupportedDeviceError):
        CrossEntropyLoss()(logits, target)


def test_cross_entropy_cuda_rejection_builds_no_graph():
    """A rejected forward call must not leave a half-built CUDA graph behind."""
    logits = Tensor([[1.0, 2.0, 3.0]], device="cuda", requires_grad=True)
    target = Tensor([2], dtype="int64")  # CPU target, CUDA logits: device mismatch
    with pytest.raises(UnsupportedDeviceError):
        CrossEntropyLoss()(logits, target)
    assert logits.grad_fn is None


def test_cross_entropy_cuda_target_out_of_range_still_raises_loss_error():
    logits = Tensor([[1.0, 2.0, 3.0]]).to("cuda")
    with pytest.raises(LossError):
        CrossEntropyLoss()(logits, np.array([5]))


# -- CrossEntropyLoss: no CPU fallback ----------------------------------------


def test_cross_entropy_cuda_forward_and_backward_never_call_cpu_backend(monkeypatch):
    logits_data = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.2], [4.0, 4.0, -2.0]], dtype=np.float32)
    target = np.array([2, 0, 1])
    logits = Tensor(logits_data, device="cuda", requires_grad=True)

    calls: list[str] = []
    compute_ops = (
        "add", "sub", "mul", "matmul", "sum", "reshape", "relu", "exp", "log",
        "exp_backward", "log_backward", "max_axis1",
    )
    for name in compute_ops:
        original = getattr(CPUBackend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    loss = CrossEntropyLoss()(logits, target)
    loss.backward()

    assert calls == []
    assert loss.device.type == "cuda"
    assert logits.grad.device.type == "cuda"
