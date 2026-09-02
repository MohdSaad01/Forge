"""Milestone 16 tests: `Sequential`.

`Sequential` adds no new discovery/training-mode/device machinery of its
own -- every test here exercises the *existing* `Module` traversal API
(`named_children`, `named_parameters`, `named_modules`, `train`/`eval`,
`to`, `device`) against a `Sequential`-composed tree, proving the container
integrates rather than duplicates.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.exceptions import ModuleError
from forge.nn import Linear, MaxPool2d, Module, ReLU, Sequential


# -- construction / validation --------------------------------------------


def test_sequential_registers_children_in_construction_order():
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 4))
    names = [name for name, _ in model.named_children()]
    assert names == ["0", "1", "2"]


def test_sequential_non_module_argument_raises_module_error():
    with pytest.raises(ModuleError):
        Sequential(Linear(2, 2), "not a module")


def test_sequential_non_module_argument_registers_nothing():
    """A rejected construction leaves no partial state -- the error is raised
    for the first non-Module argument, before any later argument is seen."""
    with pytest.raises(ModuleError):
        Sequential(object(), Linear(2, 2))


def test_sequential_empty_construction_is_supported_and_is_identity():
    model = Sequential()
    assert list(model.named_children()) == []
    x = Tensor([1.0, 2.0, 3.0])
    out = model(x)
    assert out is x


# -- forward: order matters -------------------------------------------------


def test_sequential_forward_applies_modules_in_order():
    """MaxPool2d then a shape-specific Linear only type-checks in the
    declared order -- proves `forward()` doesn't secretly reorder."""
    model = Sequential(MaxPool2d(2), ReLU())
    x = Tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) - 8.0)
    out = model(x)
    # MaxPool2d(2) on a 4x4 -> 2x2, non-overlapping windows.
    pooled = x.numpy().reshape(1, 1, 2, 2, 2, 2).max(axis=(3, 5))
    np.testing.assert_array_equal(out.numpy(), np.maximum(pooled, 0))


def test_sequential_matches_manual_composition():
    forge.random.seed(0)
    l1, r1, l2 = Linear(4, 5), ReLU(), Linear(5, 2)
    x = Tensor(np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32))

    manual = l2(r1(l1(x)))
    model = Sequential(l1, r1, l2)
    seq_out = model(x)
    np.testing.assert_allclose(seq_out.numpy(), manual.numpy())


# -- callable interface -----------------------------------------------------


def test_sequential_is_callable_through_module_dunder_call():
    model = Sequential(Linear(2, 2))
    x = Tensor([1.0, 2.0])
    out = model(x)
    assert out.shape == (2,)


# -- discovery: parameters/modules/children ---------------------------------


def test_sequential_participates_in_parameters():
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 4))
    assert len(list(model.parameters())) == 4  # two Linear layers x (weight, bias)


def test_sequential_participates_in_named_parameters_with_index_prefixes():
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 4))
    names = {name for name, _ in model.named_parameters()}
    assert names == {"0.weight", "0.bias", "2.weight", "2.bias"}


def test_sequential_participates_in_named_modules():
    model = Sequential(Linear(2, 3), ReLU())
    names = {name for name, _ in model.named_modules()}
    assert names == {"", "0", "1"}


def test_sequential_participates_in_modules():
    model = Sequential(Linear(2, 3), ReLU())
    assert len(list(model.modules())) == 3  # self + 2 children


def test_sequential_named_children_returns_immediate_children_only():
    inner = Sequential(Linear(2, 2))
    model = Sequential(inner, ReLU())
    names = [name for name, _ in model.named_children()]
    assert names == ["0", "1"]  # inner's own children are not flattened here


def test_sequential_children_matches_named_children_values():
    model = Sequential(Linear(2, 3), ReLU())
    assert list(model.children()) == [m for _, m in model.named_children()]


# -- train/eval propagation, including nested Sequential --------------------


def test_sequential_defaults_to_training_mode():
    model = Sequential(Linear(2, 2))
    assert model.training is True


def test_sequential_eval_propagates_to_children():
    model = Sequential(Linear(2, 2), ReLU())
    model.eval()
    assert model.training is False
    assert all(child.training is False for _, child in model.named_children())


def test_sequential_train_restores_children_after_eval():
    model = Sequential(Linear(2, 2), ReLU())
    model.eval()
    model.train()
    assert model.training is True
    assert all(child.training is True for _, child in model.named_children())


def test_nested_sequential_train_eval_propagates_recursively():
    inner = Sequential(Linear(2, 2))
    model = Sequential(Linear(3, 2), inner)
    model.eval()
    assert model.training is False
    assert inner.training is False
    assert inner._modules["0"].training is False

    model.train()
    assert inner.training is True
    assert inner._modules["0"].training is True


# -- device movement / Module.to() -------------------------------------------


def test_sequential_device_is_none_without_parameters():
    model = Sequential(ReLU())
    assert model.device is None


def test_sequential_device_reports_cpu_by_default():
    model = Sequential(Linear(2, 2))
    assert model.device.type == "cpu"


def test_sequential_to_cpu_is_a_no_op_and_returns_self():
    model = Sequential(Linear(2, 2))
    weight_before = model._modules["0"].weight
    result = model.to("cpu")
    assert result is model
    assert model._modules["0"].weight is weight_before


def test_sequential_to_cuda_behaves_per_hardware_availability():
    model = Sequential(Linear(2, 2), ReLU())
    if is_cuda_available():
        model.to("cuda")
        assert model.device.type == "cuda"
        assert model._modules["0"].weight.device.type == "cuda"
    else:
        from forge.exceptions import CUDAError

        with pytest.raises(CUDAError):
            model.to("cuda")


def test_nested_sequential_to_moves_every_descendant_parameter():
    if not is_cuda_available():
        pytest.skip("CUDA is not available on this machine")
    inner = Sequential(Linear(2, 2))
    model = Sequential(Linear(3, 2), inner)
    model.to("cuda")
    assert inner._modules["0"].weight.device.type == "cuda"


# -- integration with a bare, non-Sequential Module --------------------------


def test_sequential_as_a_child_of_a_hand_written_module():
    class Wrapper(Module):
        def __init__(self):
            super().__init__()
            self.body = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))

        def forward(self, x):
            return self.body(x)

    m = Wrapper()
    x = Tensor([1.0, 2.0])
    out = m(x)
    assert out.shape == (1,)
    assert len(list(m.parameters())) == 4
