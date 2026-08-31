import pytest

import forge
from forge import Device, Tensor
from forge.exceptions import UnsupportedDeviceError


def test_parse_bare_cpu():
    d = Device.parse("cpu")
    assert d.type == "cpu"
    assert d.index is None
    assert str(d) == "cpu"


def test_parse_cuda_with_index():
    d = Device.parse("cuda:0")
    assert d.type == "cuda"
    assert d.index == 0
    assert str(d) == "cuda:0"


def test_parse_is_case_insensitive():
    d = Device.parse("CPU")
    assert d.type == "cpu"


def test_parse_unknown_device_type_raises_clearly():
    with pytest.raises(UnsupportedDeviceError, match="tpu"):
        Device.parse("tpu")


def test_parse_bad_index_raises_clearly():
    with pytest.raises(UnsupportedDeviceError):
        Device.parse("cuda:x")


def test_device_equality():
    assert Device.parse("cpu") == Device.parse("cpu")
    assert Device.parse("cuda:0") != Device.parse("cuda:1")


def test_tensor_on_cpu_succeeds():
    t = Tensor([1, 2, 3], device="cpu")
    assert t.device.type == "cpu"


def test_tensor_on_cuda_behaves_per_hardware_availability():
    # The device string itself always parses (Device.parse succeeds). Whether
    # constructing a tensor on it actually *works* depends on hardware/toolchain
    # availability -- this CPU-tier test must not require CUDA to run, so it
    # branches on `is_cuda_available()` rather than assuming either outcome.
    # See tests/test_cuda_backend.py for CUDA-only hardware verification.
    from forge.backend.cuda import is_cuda_available

    if is_cuda_available():
        t = Tensor([1, 2, 3], device="cuda")
        assert t.device.type == "cuda"
    else:
        with pytest.raises(forge.CUDAError):
            Tensor([1, 2, 3], device="cuda")


def test_unknown_device_string_on_tensor_raises_clearly():
    with pytest.raises(UnsupportedDeviceError):
        Tensor([1, 2, 3], device="not-a-real-device")
