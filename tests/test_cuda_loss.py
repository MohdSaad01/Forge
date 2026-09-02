"""CUDA loss tests (Milestone 12).

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py`/`tests/test_cuda_consistency.py`/
`tests/test_cuda_autograd.py`/`tests/test_module_cuda.py`. `MSELoss` is
CUDA-compatible as of this milestone -- it composes only ops CUDA already
implements forward *and* backward (`-`, `*`, `.sum()`), so no new CUDA
kernel was needed (see `docs/architecture/cuda-backend.md`'s **CUDA
losses** section). `CrossEntropyLoss` remains CPU-only by deliberate
deferral; this file also proves that deferral fails clearly rather than
silently falling back to CPU.
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


# -- CrossEntropyLoss: deferred to CPU, fails clearly on CUDA ----------------


def test_cross_entropy_rejects_cuda_logits_clearly():
    logits = Tensor([[1.0, 2.0, 3.0]]).to("cuda")
    target = np.array([2])
    with pytest.raises(LossError, match="CPU-only"):
        CrossEntropyLoss()(logits, target)


def test_cross_entropy_cuda_rejection_builds_no_graph():
    """A rejected forward call must not leave a half-built CUDA graph behind."""
    logits = Tensor([[1.0, 2.0, 3.0]], device="cuda", requires_grad=True)
    target = np.array([2])
    with pytest.raises(LossError):
        CrossEntropyLoss()(logits, target)
    assert logits.grad_fn is None


def test_cross_entropy_still_works_on_cpu_after_moving_explicitly():
    """The documented escape hatch: `.to('cpu')` first, then CrossEntropyLoss works."""
    logits = Tensor([[1.0, 2.0, 3.0]], device="cuda").to("cpu")
    loss = CrossEntropyLoss()(logits, np.array([2]))
    assert loss.device.type == "cpu"
    assert np.isfinite(loss.numpy())
