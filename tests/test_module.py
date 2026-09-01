import pytest

from forge.backend.cuda import is_cuda_available
from forge.exceptions import ModuleError, UnsupportedDeviceError
from forge.nn import Linear, Module, Parameter


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter([1.0, 2.0])

    def forward(self, x):
        return x + self.weight


class TwoLayer(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(3, 4)
        self.layer2 = Linear(4, 2)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class SharedWeight(Module):
    """Two attributes intentionally referencing the same Parameter."""

    def __init__(self):
        super().__init__()
        w = Parameter([1.0, 2.0, 3.0])
        self.a = w
        self.b = w

    def forward(self, x):
        return x


# -- parameter registration ---------------------------------------------


def test_module_registers_parameter_attribute():
    m = Simple()
    names = dict(m.named_parameters())
    assert "weight" in names
    assert names["weight"] is m.weight


def test_module_forgetting_super_init_raises_clearly():
    class Broken(Module):
        def __init__(self):
            self.weight = Parameter([1.0])  # no super().__init__()

    with pytest.raises(ModuleError):
        Broken()


# -- child modules and recursive discovery -------------------------------


def test_module_registers_child_modules():
    m = TwoLayer()
    children = dict(m.named_children())
    assert set(children) == {"layer1", "layer2"}
    assert children["layer1"] is m.layer1


def test_recursive_parameter_discovery_finds_all_nested_parameters():
    m = TwoLayer()
    names = {name for name, _ in m.named_parameters()}
    assert names == {"layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"}


def test_parameters_returns_four_params_for_two_linear_layers():
    m = TwoLayer()
    assert len(list(m.parameters())) == 4


def test_named_parameters_paths_reflect_hierarchy():
    m = TwoLayer()
    names = [name for name, _ in m.named_parameters()]
    assert "layer1.weight" in names
    assert "layer1.bias" in names
    assert "layer2.weight" in names
    assert "layer2.bias" in names


def test_duplicate_parameter_reference_not_returned_twice():
    m = SharedWeight()
    params = list(m.parameters())
    assert len(params) == 1
    names = [name for name, _ in m.named_parameters()]
    assert names == ["a"]  # first-seen name only, "b" is the same object


def test_named_modules_includes_self_and_children():
    m = TwoLayer()
    names = {name for name, _ in m.named_modules()}
    assert names == {"", "layer1", "layer2"}


# -- callable interface ---------------------------------------------------


def test_calling_module_invokes_forward():
    m = Simple()
    from forge import Tensor

    out = m(Tensor([10.0, 20.0]))
    assert out.numpy().tolist() == [11.0, 22.0]


def test_base_module_forward_not_implemented_raises_clearly():
    m = Module()
    with pytest.raises(ModuleError):
        m()


# -- training / evaluation mode -------------------------------------------


def test_module_defaults_to_training_mode():
    m = Simple()
    assert m.training is True


def test_eval_sets_training_false():
    m = Simple()
    m.eval()
    assert m.training is False


def test_train_sets_training_true():
    m = Simple()
    m.eval()
    m.train()
    assert m.training is True


def test_mode_propagates_to_child_modules():
    m = TwoLayer()
    m.eval()
    assert m.training is False
    assert m.layer1.training is False
    assert m.layer2.training is False

    m.train()
    assert m.layer1.training is True
    assert m.layer2.training is True


def test_train_accepts_explicit_mode_argument():
    m = TwoLayer()
    m.train(False)
    assert m.training is False
    assert m.layer1.training is False


# -- device movement (Milestone 9) ----------------------------------------


def test_to_cpu_is_a_no_op_and_returns_self():
    m = TwoLayer()
    weight_before = m.layer1.weight
    result = m.to("cpu")
    assert result is m
    assert m.layer1.weight is weight_before
    assert m.layer1.weight.device.type == "cpu"


def test_to_invalid_device_string_raises_clearly():
    m = TwoLayer()
    with pytest.raises(UnsupportedDeviceError):
        m.to("not-a-real-device")


def test_to_returns_self_matching_train_eval_convention():
    m = Simple()
    assert m.to("cpu") is m


def test_module_device_is_none_without_parameters():
    class NoParams(Module):
        def forward(self, x):
            return x

    assert NoParams().device is None


def test_module_device_reports_cpu_by_default():
    m = TwoLayer()
    assert m.device.type == "cpu"


def test_module_to_cuda_behaves_per_hardware_availability():
    """Branches on hardware availability so this CPU-tier test never requires CUDA
    -- see tests/test_module_cuda.py for the full CUDA-hardware verification."""
    m = TwoLayer()
    if is_cuda_available():
        m.to("cuda")
        assert m.device.type == "cuda"
    else:
        from forge.exceptions import CUDAError

        with pytest.raises(CUDAError):
            m.to("cuda")
