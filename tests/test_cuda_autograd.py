"""CUDA autograd tests (Milestone 10).

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py`/`tests/test_cuda_consistency.py`/
`tests/test_module_cuda.py`. These prove CUDA tensors participate in
reverse-mode autograd for real: every backward computation for a supported
operation executes as a genuine CUDA kernel (never a disguised CPU
fallback), CUDA gradients agree with CPU gradients within tolerance, CUDA
gradients are themselves CUDA-resident, device/dtype mismatches on
`backward()` fail clearly, and unsupported CUDA operations (`exp`/`log`)
still fail clearly rather than silently building a broken graph. See
`docs/architecture/autograd.md` and `docs/architecture/cuda-backend.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.exceptions import CUDAError, ShapeMismatchError, UnsupportedDeviceError
from forge.nn import Linear, Module, ReLU
from forge.optim import SGD

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


def _cpu_and_cuda_leaves(*arrays: np.ndarray) -> "tuple[list[Tensor], list[Tensor]]":
    """Build matching CPU/CUDA leaf tensors (float32, requires_grad=True) from the same data."""
    cpu = [Tensor(a.astype(np.float32).copy(), requires_grad=True) for a in arrays]
    cuda = [Tensor(a.astype(np.float32).copy(), device="cuda", requires_grad=True) for a in arrays]
    return cpu, cuda


def _assert_matching_grads(cpu_tensors, cuda_tensors) -> None:
    for cpu_t, cuda_t in zip(cpu_tensors, cuda_tensors):
        assert cuda_t.grad is not None
        assert cuda_t.grad.device.type == "cuda"
        np.testing.assert_allclose(cuda_t.grad.to("cpu").numpy(), cpu_t.grad.numpy(), **TOL)


# -- elementwise: add / sub / mul, exact shape and row-broadcast -------------------


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_elementwise_backward_matches_cpu(op):
    a = np.array([1.0, -2.0, 3.5])
    b = np.array([10.0, 20.0, -30.0])
    (a_cpu, b_cpu), (a_cuda, b_cuda) = _cpu_and_cuda_leaves(a, b)

    if op == "add":
        cpu_out, cuda_out = (a_cpu + b_cpu).sum(), (a_cuda + b_cuda).sum()
    elif op == "sub":
        cpu_out, cuda_out = (a_cpu - b_cpu).sum(), (a_cuda - b_cuda).sum()
    else:
        cpu_out, cuda_out = (a_cpu * b_cpu).sum(), (a_cuda * b_cuda).sum()

    cpu_out.backward()
    cuda_out.backward()
    _assert_matching_grads([a_cpu, b_cpu], [a_cuda, b_cuda])


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_row_broadcast_backward_matches_cpu(op):
    """(rows, cols) op (cols,) -- the shape CUDA Linear's bias add needs."""
    mat = np.random.default_rng(0).standard_normal((4, 3))
    vec = np.random.default_rng(1).standard_normal((3,))
    (mat_cpu, vec_cpu), (mat_cuda, vec_cuda) = _cpu_and_cuda_leaves(mat, vec)

    if op == "add":
        cpu_out, cuda_out = (mat_cpu + vec_cpu).sum(), (mat_cuda + vec_cuda).sum()
    elif op == "sub":
        cpu_out, cuda_out = (mat_cpu - vec_cpu).sum(), (mat_cuda - vec_cuda).sum()
    else:
        cpu_out, cuda_out = (mat_cpu * vec_cpu).sum(), (mat_cuda * vec_cuda).sum()

    cpu_out.backward()
    cuda_out.backward()
    assert mat_cuda.grad.shape == (4, 3)
    assert vec_cuda.grad.shape == (3,)
    _assert_matching_grads([mat_cpu, vec_cpu], [mat_cuda, vec_cuda])


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_row_broadcast_backward_reversed_operand_order_matches_cpu(op):
    """(cols,) op (rows, cols) -- the reversed operand order, per CUDA forward's own coverage."""
    vec = np.random.default_rng(2).standard_normal((3,))
    mat = np.random.default_rng(3).standard_normal((4, 3))
    (vec_cpu, mat_cpu), (vec_cuda, mat_cuda) = _cpu_and_cuda_leaves(vec, mat)

    if op == "add":
        cpu_out, cuda_out = (vec_cpu + mat_cpu).sum(), (vec_cuda + mat_cuda).sum()
    elif op == "sub":
        cpu_out, cuda_out = (vec_cpu - mat_cpu).sum(), (vec_cuda - mat_cuda).sum()
    else:
        cpu_out, cuda_out = (vec_cpu * mat_cpu).sum(), (vec_cuda * mat_cuda).sum()

    cpu_out.backward()
    cuda_out.backward()
    _assert_matching_grads([vec_cpu, mat_cpu], [vec_cuda, mat_cuda])


def test_multiple_use_of_same_cuda_tensor_accumulates():
    """Two consumers contributing to the same tensor's gradient -- exercises the
    backend-dispatched accumulation path in `forge.autograd.engine.run_backward`
    (a CUDAStorage has no `__add__`; the engine must route this through
    `Backend.add`)."""
    (x_cpu,), (x_cuda,) = _cpu_and_cuda_leaves(np.array([2.0, 3.0]))
    two = Tensor([2.0, 2.0], device="cpu")
    three = Tensor([3.0, 3.0], device="cpu")
    two_cuda, three_cuda = two.to("cuda"), three.to("cuda")
    cpu_out = (x_cpu * two + x_cpu * three).sum()
    cuda_out = (x_cuda * two_cuda + x_cuda * three_cuda).sum()
    cpu_out.backward()
    cuda_out.backward()
    _assert_matching_grads([x_cpu], [x_cuda])
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), [5.0, 5.0], **TOL)


# -- matmul: all four supported 1D/2D shape combinations ----------------------------


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((4,), (4,)),  # vector . vector
        ((5, 4), (4,)),  # matrix . vector
        ((4,), (4, 3)),  # vector . matrix
        ((5, 4), (4, 3)),  # matrix . matrix
    ],
)
def test_matmul_backward_matches_cpu(a_shape, b_shape):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(a_shape)
    b = rng.standard_normal(b_shape)
    (a_cpu, b_cpu), (a_cuda, b_cuda) = _cpu_and_cuda_leaves(a, b)

    cpu_result = a_cpu @ b_cpu
    cuda_result = a_cuda @ b_cuda
    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)

    if cpu_result.shape == ():
        cpu_result.backward()
        cuda_result.backward()
    else:
        grad = rng.standard_normal(cpu_result.shape).astype(np.float32)
        cpu_result.backward(Tensor(grad))
        cuda_result.backward(Tensor(grad, device="cuda"))

    _assert_matching_grads([a_cpu, b_cpu], [a_cuda, b_cuda])


def test_matmul_backward_matches_numerical_gradient():
    """Finite-difference check for CUDA matmul backward (matrix.matrix case)."""
    rng = np.random.default_rng(5)
    a_data = rng.standard_normal((3, 2)).astype(np.float32)
    b_data = rng.standard_normal((2, 4)).astype(np.float32)

    a_cuda = Tensor(a_data.copy(), device="cuda", requires_grad=True)
    b_cuda = Tensor(b_data.copy(), device="cuda", requires_grad=True)
    (a_cuda @ b_cuda).sum().backward()
    analytical = a_cuda.grad.to("cpu").numpy()

    eps = 1e-3
    numerical = np.zeros_like(a_data)
    it = np.nditer(a_data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        a_plus, a_minus = a_data.copy(), a_data.copy()
        a_plus[idx] += eps
        a_minus[idx] -= eps
        plus = float(np.sum(a_plus @ b_data))
        minus = float(np.sum(a_minus @ b_data))
        numerical[idx] = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-2)


# -- sum / reshape ------------------------------------------------------------------


def test_sum_full_reduction_backward_matches_cpu():
    data = np.random.default_rng(6).standard_normal((3, 4))
    (x_cpu,), (x_cuda,) = _cpu_and_cuda_leaves(data)
    x_cpu.sum().backward()
    x_cuda.sum().backward()
    _assert_matching_grads([x_cpu], [x_cuda])
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), np.ones((3, 4)), **TOL)


def test_sum_axis_is_unsupported_on_cuda_backward_too():
    """Forward already refuses axis-wise CUDA sum; there is no reachable backward path."""
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", requires_grad=True)
    with pytest.raises(CUDAError, match="axis"):
        t.sum(axis=0)


def test_reshape_backward_matches_cpu():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    (x_cpu,), (x_cuda,) = _cpu_and_cuda_leaves(data)
    x_cpu.reshape(2, 3).sum().backward()
    x_cuda.reshape(2, 3).sum().backward()
    _assert_matching_grads([x_cpu], [x_cuda])


# -- ReLU: positive, negative, zero, mixed -------------------------------------------


def test_relu_backward_matches_cpu_mixed_signs():
    data = np.array([-3.0, 2.5, 0.0, 1.5, -0.1, 4.0, -0.001])
    (x_cpu,), (x_cuda,) = _cpu_and_cuda_leaves(data)
    x_cpu.relu().sum().backward()
    x_cuda.relu().sum().backward()
    _assert_matching_grads([x_cpu], [x_cuda])
    # Strict-inequality convention: gradient at exactly 0 is 0.
    grad = x_cuda.grad.to("cpu").numpy()
    np.testing.assert_allclose(grad, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], **TOL)


def test_relu_backward_all_positive():
    (x_cpu,), (x_cuda,) = _cpu_and_cuda_leaves(np.array([1.0, 5.0, 0.5]))
    x_cpu.relu().sum().backward()
    x_cuda.relu().sum().backward()
    _assert_matching_grads([x_cpu], [x_cuda])
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), [1.0, 1.0, 1.0], **TOL)


def test_relu_backward_all_negative():
    (x_cpu,), (x_cuda,) = _cpu_and_cuda_leaves(np.array([-1.0, -5.0, -0.5]))
    x_cpu.relu().sum().backward()
    x_cuda.relu().sum().backward()
    _assert_matching_grads([x_cpu], [x_cuda])
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), [0.0, 0.0, 0.0], **TOL)


# -- CUDA Linear backward ------------------------------------------------------------


def test_cuda_linear_backward_weight_and_bias_match_cpu_and_are_cuda_resident():
    forge.random.seed(42)
    cpu_model = Linear(4, 3)
    cuda_model = Linear(4, 3)
    cuda_model.weight._data = np.array(cpu_model.weight._data, copy=True)
    cuda_model.bias._data = np.array(cpu_model.bias._data, copy=True)
    cuda_model.to("cuda")

    x = np.random.default_rng(7).standard_normal((5, 4)).astype(np.float32)
    x_cpu, x_cuda = Tensor(x), Tensor(x, device="cuda")

    cpu_model(x_cpu).sum().backward()
    cuda_model(x_cuda).sum().backward()

    assert cuda_model.weight.grad.device.type == "cuda"
    assert cuda_model.bias.grad.device.type == "cuda"
    np.testing.assert_allclose(
        cuda_model.weight.grad.to("cpu").numpy(), cpu_model.weight.grad.numpy(), **TOL
    )
    np.testing.assert_allclose(
        cuda_model.bias.grad.to("cpu").numpy(), cpu_model.bias.grad.numpy(), **TOL
    )


def test_cuda_linear_backward_never_calls_cpu_backend(monkeypatch):
    model = Linear(4, 3).to("cuda")
    x = Tensor(np.random.default_rng(8).standard_normal((5, 4)).astype(np.float32), device="cuda")

    calls: list[str] = []
    for name in ("matmul", "add", "sub", "mul", "relu", "sum", "reshape", "matmul_backward",
                 "add_backward", "sub_backward", "mul_backward", "relu_backward", "sum_backward",
                 "reshape_backward", "sgd_step"):
        original = getattr(CPUBackend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    out = model(x)
    out.sum().backward()

    assert calls == []
    assert isinstance(model.weight.grad._data, CUDAStorage)
    assert isinstance(model.bias.grad._data, CUDAStorage)


# -- CUDA ReLU through a real Linear -> ReLU -> Linear model --------------------------


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 6)
        self.relu = ReLU()
        self.fc2 = Linear(6, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _matched_mlp() -> "tuple[MLP, MLP]":
    forge.random.seed(123)
    cpu_model = MLP()
    cuda_model = MLP()
    for (_, p_cpu), (_, p_cuda) in zip(cpu_model.named_parameters(), cuda_model.named_parameters()):
        p_cuda._data = np.array(p_cpu._data, copy=True)
    return cpu_model, cuda_model


def test_cuda_multilayer_model_backward_matches_cpu():
    cpu_model, cuda_model = _matched_mlp()
    cuda_model.to("cuda")

    x = np.random.default_rng(9).standard_normal((6, 4)).astype(np.float32)
    x_cpu, x_cuda = Tensor(x), Tensor(x, device="cuda")

    cpu_out = cpu_model(x_cpu)
    cuda_out = cuda_model(x_cuda)
    np.testing.assert_allclose(cuda_out.to("cpu").numpy(), cpu_out.numpy(), **TOL)

    cpu_out.sum().backward()
    cuda_out.sum().backward()

    for (name, p_cpu), (_, p_cuda) in zip(cpu_model.named_parameters(), cuda_model.named_parameters()):
        assert p_cuda.grad is not None, name
        assert p_cuda.grad.device.type == "cuda", name
        np.testing.assert_allclose(p_cuda.grad.to("cpu").numpy(), p_cpu.grad.numpy(), **TOL)


def test_cuda_multilayer_model_backward_never_calls_cpu_backend(monkeypatch):
    _, cuda_model = _matched_mlp()
    cuda_model.to("cuda")
    x_cuda = Tensor(np.random.default_rng(10).standard_normal((5, 4)).astype(np.float32), device="cuda")

    calls: list[str] = []
    for name in dir(CPUBackend):
        if name.startswith("_"):
            continue
        original = getattr(CPUBackend, name)
        if not callable(original):
            continue

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    out = cuda_model(x_cuda)
    out.sum().backward()

    assert calls == []


# -- device / dtype mismatch on backward() -------------------------------------------


def test_backward_with_cpu_gradient_on_cuda_tensor_raises_clearly():
    t = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
    with pytest.raises(UnsupportedDeviceError):
        t.backward(Tensor([1.0, 1.0]))  # CPU gradient, no explicit .to("cuda")


def test_backward_with_cuda_gradient_on_cpu_tensor_raises_clearly():
    t = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(UnsupportedDeviceError):
        t.backward(Tensor([1.0, 1.0], device="cuda"))


def test_backward_with_mismatched_shape_gradient_on_cuda_still_raises_shape_error():
    t = Tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
    with pytest.raises(ShapeMismatchError):
        t.backward(Tensor([1.0, 1.0], device="cuda"))


# -- unsupported CUDA operations fail clearly, no fallback ----------------------------


def test_exp_on_grad_requiring_cuda_tensor_raises_cuda_error_not_silent_fallback():
    t = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
    with pytest.raises(CUDAError):
        t.exp()
    # No graph was built by the failed call -- the tensor is untouched.
    assert t.is_leaf is True
    assert t.grad_fn is None


def test_log_on_grad_requiring_cuda_tensor_raises_cuda_error_not_silent_fallback():
    t = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
    with pytest.raises(CUDAError):
        t.log()
    assert t.is_leaf is True
    assert t.grad_fn is None


# -- no_grad still suspends CUDA graph construction ------------------------------------


def test_no_grad_still_suspends_cuda_graph_construction():
    with forge.no_grad():
        a = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
        b = Tensor([3.0, 4.0], device="cuda", requires_grad=True)
        c = a + b
    assert c.requires_grad is False
    assert c.grad_fn is None
    assert c.is_leaf is True


# -- SGD on CUDA parameters -----------------------------------------------------------


def test_sgd_step_updates_cuda_parameter_in_place_without_graph():
    model = Linear(4, 3).to("cuda")
    x = Tensor(np.random.default_rng(11).standard_normal((5, 4)).astype(np.float32), device="cuda")
    opt = SGD(model.parameters(), lr=0.1)

    before_weight = model.weight.to("cpu").numpy().copy()
    model(x).sum().backward()
    opt.step()
    after_weight = model.weight.to("cpu").numpy()

    assert model.weight.device.type == "cuda"
    assert isinstance(model.weight._data, CUDAStorage)
    assert not np.allclose(before_weight, after_weight)
    assert model.weight.grad_fn is None
    assert model.weight.is_leaf is True


def test_sgd_step_on_cuda_matches_cpu_step():
    forge.random.seed(7)
    cpu_model = Linear(3, 2)
    cuda_model = Linear(3, 2)
    cuda_model.weight._data = np.array(cpu_model.weight._data, copy=True)
    cuda_model.bias._data = np.array(cpu_model.bias._data, copy=True)
    cuda_model.to("cuda")

    x = np.random.default_rng(12).standard_normal((4, 3)).astype(np.float32)
    cpu_model(Tensor(x)).sum().backward()
    cuda_model(Tensor(x, device="cuda")).sum().backward()

    SGD(cpu_model.parameters(), lr=0.05).step()
    SGD(cuda_model.parameters(), lr=0.05).step()

    np.testing.assert_allclose(cuda_model.weight.to("cpu").numpy(), cpu_model.weight.numpy(), **TOL)
    np.testing.assert_allclose(cuda_model.bias.to("cpu").numpy(), cpu_model.bias.numpy(), **TOL)


def test_zero_grad_clears_cuda_gradients_without_cpu_transfer():
    model = Linear(4, 3).to("cuda")
    x = Tensor(np.random.default_rng(13).standard_normal((5, 4)).astype(np.float32), device="cuda")
    opt = SGD(model.parameters(), lr=0.1)
    model(x).sum().backward()
    assert model.weight.grad is not None
    opt.zero_grad()
    assert model.weight.grad is None
    assert model.bias.grad is None


# -- small real CUDA training loop (TensorDataset -> DataLoader -> SGD) ---------------


def test_small_cuda_training_loop_reduces_loss_and_recovers_weights():
    forge.random.seed(0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 3)).astype(np.float32)
    true_w = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    Y = (X @ true_w + 0.3).reshape(-1, 1).astype(np.float32)

    model = Linear(3, 1)
    model.to("cuda")
    opt = SGD(model.parameters(), lr=0.05)
    loss_fn = forge.nn.MSELoss()

    dataset = forge.data.TensorDataset(Tensor(X), Tensor(Y))
    loader = forge.data.DataLoader(dataset, batch_size=16, shuffle=True)

    first_loss = None
    last_loss = None
    for _epoch in range(20):
        epoch_loss, n = 0.0, 0
        for xb, yb in loader:
            xb_cuda, yb_cuda = xb.to("cuda"), yb.to("cuda")
            opt.zero_grad()
            pred = model(xb_cuda)
            loss = loss_fn(pred, yb_cuda)
            loss.backward()
            opt.step()
            epoch_loss += loss.to("cpu").numpy().item() * xb.shape[0]
            n += xb.shape[0]
        avg = epoch_loss / n
        if first_loss is None:
            first_loss = avg
        last_loss = avg

    assert last_loss < first_loss * 0.05
    np.testing.assert_allclose(model.weight.to("cpu").numpy().flatten(), true_w, atol=0.05)
    np.testing.assert_allclose(model.bias.to("cpu").numpy(), [0.3], atol=0.05)
