"""CUDA `nn.Module` execution tests (Milestone 9).

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py`/`tests/test_cuda_consistency.py`.
These prove the *high-level* Module/Parameter/Linear/ReLU path executes on
real CUDA hardware -- `tests/test_cuda_consistency.py` already covers the raw
Tensor-level operations these are built from. See
`docs/architecture/cuda-backend.md` and `docs/architecture/modules.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.exceptions import CUDAError, ModuleError, UnsupportedDeviceError
from forge.nn import Linear, Module, Parameter, ReLU
from forge.optim import SGD
from forge.training import Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


class MLP(Module):
    """`Linear -> ReLU -> Linear`, the exact shape M9 targets."""

    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.relu = ReLU()
        self.fc2 = Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _matched_models() -> tuple[MLP, MLP]:
    """Two `MLP`s with numerically identical parameters, one about to move to CUDA.

    Building two independently-initialized models would make a CPU/CUDA
    output comparison meaningless (different random weights) -- per
    `docs/architecture/cuda-backend.md`'s numerical-consistency rule, both
    models must start from the *same* parameter values.
    """
    forge.random.seed(123)
    cpu_model = MLP()
    cuda_model = MLP()
    for (_, p_cpu), (_, p_cuda) in zip(cpu_model.named_parameters(), cuda_model.named_parameters()):
        p_cuda._data = np.array(p_cpu._data, copy=True)
    return cpu_model, cuda_model


# -- Module.to() ---------------------------------------------------------------


def test_to_cuda_moves_all_parameters_to_cuda_storage():
    model = MLP()
    model.to("cuda")
    for _, param in model.named_parameters():
        assert param.device.type == "cuda"
        assert isinstance(param._data, CUDAStorage)
        assert not isinstance(param._data, np.ndarray)


def test_to_cpu_moves_a_cuda_model_back():
    model = MLP().to("cuda")
    model.to("cpu")
    for _, param in model.named_parameters():
        assert param.device.type == "cpu"
        assert isinstance(param._data, np.ndarray)


def test_to_recurses_into_every_child_module():
    model = MLP().to("cuda")
    assert model.fc1.weight.device.type == "cuda"
    assert model.fc1.bias.device.type == "cuda"
    assert model.fc2.weight.device.type == "cuda"
    assert model.fc2.bias.device.type == "cuda"


def test_to_preserves_shape_dtype_and_requires_grad():
    model = MLP()
    before = {name: (p.shape, p.dtype, p.requires_grad) for name, p in model.named_parameters()}
    model.to("cuda")
    after = {name: (p.shape, p.dtype, p.requires_grad) for name, p in model.named_parameters()}
    assert before == after


def test_to_preserves_parameter_identity():
    model = MLP()
    before = {name: id(p) for name, p in model.named_parameters()}
    model.to("cuda")
    after = {name: id(p) for name, p in model.named_parameters()}
    assert before == after
    assert model.fc1.weight is model._modules["fc1"]._parameters["weight"]


def test_to_preserves_registration_and_mutates_returns_self():
    model = MLP()
    result = model.to("cuda")
    assert result is model
    assert set(name for name, _ in model.named_parameters()) == {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
    }
    assert "fc1" in dict(model.named_children())


def test_to_produces_leaf_parameters_with_no_graph():
    model = MLP().to("cuda")
    for _, param in model.named_parameters():
        assert param.is_leaf is True
        assert param.grad_fn is None


def test_to_clears_stale_grad():
    model = MLP()
    x = Tensor(np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32))
    model(x).sum().backward()
    assert model.fc1.weight.grad is not None
    model.to("cuda")
    assert model.fc1.weight.grad is None


def test_to_invalid_device_string_raises_clearly():
    model = MLP()
    with pytest.raises(UnsupportedDeviceError):
        model.to("not-a-real-device")


def test_to_cuda_then_cpu_round_trip_preserves_values():
    model = MLP()
    original = {name: np.array(p.numpy(), copy=True) for name, p in model.named_parameters()}
    model.to("cuda")
    model.to("cpu")
    for name, param in model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original[name], **TOL)


# -- Module device introspection ------------------------------------------------


def test_module_device_reports_common_device():
    model = MLP()
    assert model.device.type == "cpu"
    model.to("cuda")
    assert model.device.type == "cuda"


def test_module_device_none_for_parameterless_module():
    assert ReLU().device is None


def test_module_device_raises_for_manually_mixed_parameters():
    model = MLP().to("cuda")
    # Manually reassign one parameter back to CPU, bypassing Module.to() --
    # a legitimate way to construct a mixed-device tree for this test.
    model.fc2.weight = Parameter(np.zeros((8, 3), dtype=np.float32), device="cpu")
    with pytest.raises(ModuleError):
        _ = model.device


# -- Device mismatch -------------------------------------------------------------


def test_cuda_model_rejects_cpu_input():
    model = MLP().to("cuda")
    x_cpu = Tensor(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(UnsupportedDeviceError):
        with forge.no_grad():
            model(x_cpu)


def test_cpu_model_rejects_cuda_input():
    model = MLP()
    x_cuda = Tensor(np.zeros((2, 4), dtype=np.float32)).to("cuda")
    with pytest.raises(UnsupportedDeviceError):
        model(x_cuda)


# -- CUDA Linear -----------------------------------------------------------------


def test_cuda_linear_single_sample_output_shape_and_values():
    cpu_model, cuda_model = _matched_models()
    cuda_model.to("cuda")
    x = np.random.default_rng(5).standard_normal((4,)).astype(np.float32)
    x_cpu, x_cuda = Tensor(x), Tensor(x).to("cuda")

    cpu_out = cpu_model.fc1(x_cpu)
    with forge.no_grad():
        cuda_out = cuda_model.fc1(x_cuda)

    assert cuda_out.shape == (8,)
    assert cuda_out.device.type == "cuda"
    np.testing.assert_allclose(cuda_out.to("cpu").numpy(), cpu_out.numpy(), **TOL)


def test_cuda_linear_batched_input_output_shape_and_values():
    cpu_model, cuda_model = _matched_models()
    cuda_model.to("cuda")
    x = np.random.default_rng(6).standard_normal((10, 4)).astype(np.float32)
    x_cpu, x_cuda = Tensor(x), Tensor(x).to("cuda")

    cpu_out = cpu_model.fc1(x_cpu)
    with forge.no_grad():
        cuda_out = cuda_model.fc1(x_cuda)

    assert cuda_out.shape == (10, 8)
    np.testing.assert_allclose(cuda_out.to("cpu").numpy(), cpu_out.numpy(), **TOL)


def test_cuda_linear_weight_and_bias_are_cuda_resident():
    model = Linear(4, 8).to("cuda")
    assert model.weight.device.type == "cuda"
    assert model.bias.device.type == "cuda"


# -- High-level model: Linear -> ReLU -> Linear on CUDA --------------------------


def test_cuda_full_model_forward_matches_cpu():
    cpu_model, cuda_model = _matched_models()
    cuda_model.to("cuda")
    x = np.random.default_rng(7).standard_normal((6, 4)).astype(np.float32)
    x_cpu, x_cuda = Tensor(x), Tensor(x).to("cuda")

    cpu_out = cpu_model(x_cpu)
    with forge.no_grad():
        cuda_out = cuda_model(x_cuda)

    assert cuda_out.device.type == "cuda"
    assert isinstance(cuda_out._data, CUDAStorage)
    assert cuda_out.shape == cpu_out.shape
    np.testing.assert_allclose(cuda_out.to("cpu").numpy(), cpu_out.numpy(), **TOL)


def test_cuda_full_model_forward_without_no_grad_raises_immediately():
    """Parameters keep requires_grad=True after `.to()`; building a graph on CUDA is
    still unsupported (unchanged M8 boundary), so a bare (non-`no_grad`) forward call
    raises immediately rather than silently succeeding or building a broken graph."""
    model = MLP().to("cuda")
    x_cuda = Tensor(np.zeros((2, 4), dtype=np.float32)).to("cuda")
    with pytest.raises(UnsupportedDeviceError, match="[Aa]utomatic differentiation"):
        model(x_cuda)


def test_cuda_model_forward_does_not_call_cpu_backend(monkeypatch):
    """Structural proof the CUDA model path never quietly delegates to CPUBackend."""
    _, cuda_model = _matched_models()
    cuda_model.to("cuda")
    x_cuda = Tensor(np.random.default_rng(8).standard_normal((5, 4)).astype(np.float32)).to("cuda")

    calls: list[str] = []
    for name in ("matmul", "add", "sub", "mul", "relu", "sum", "reshape"):
        original = getattr(CPUBackend, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(CPUBackend, name, spy)

    with forge.no_grad():
        out = cuda_model(x_cuda)

    assert calls == []
    assert isinstance(out._data, CUDAStorage)


# -- Autograd boundary -----------------------------------------------------------


def test_cuda_model_backward_fails_clearly():
    model = MLP().to("cuda")
    x_cuda = Tensor(np.random.default_rng(9).standard_normal((3, 4)).astype(np.float32)).to("cuda")
    with forge.no_grad():
        out = model(x_cuda)
    with pytest.raises(UnsupportedDeviceError, match="cpu"):
        out.sum().backward()


# -- Persistence / Trainer boundaries (documented, not implemented in M9) --------


def test_saving_a_cuda_model_is_rejected_not_silently_copied(tmp_path):
    model = MLP().to("cuda")
    with pytest.raises(forge.ForgeError):
        forge.save_model(model, str(tmp_path / "model.forge"))


def test_trainer_with_a_cuda_model_fails_clearly_on_forward():
    """Trainer itself stays CPU-only (unchanged M6 boundary); a CUDA-moved model
    fed CPU batches by a CPU DataLoader fails at the first device-mismatched
    forward op, never silently training or silently transferring anything."""
    model = MLP().to("cuda")
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = forge.nn.MSELoss()
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device="cpu", verbose=False)

    x = Tensor(np.random.default_rng(10).standard_normal((6, 4)).astype(np.float32))
    y = Tensor(np.random.default_rng(11).standard_normal((6, 3)).astype(np.float32))
    loader = forge.data.DataLoader(forge.data.TensorDataset(x, y), batch_size=2)

    with pytest.raises(UnsupportedDeviceError):
        trainer.fit(loader, epochs=1)
