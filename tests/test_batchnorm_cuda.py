"""CUDA `nn.BatchNorm2d` (Milestone 53): parity with CPU, plus the CUDA-only
fused-kernel path (`Tensor.batch_norm2d()`, `CUDABackend.batch_norm2d{,_backward}`).

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise, matching `tests/test_cuda_backend.py`'s convention. See
`tests/test_batchnorm.py` for the CPU-only reference/finite-difference suite
this mirrors, and `docs/development/m53-batchnorm.md` for the CUDA kernel
design (mean/var reduction, fused normalize+affine, backward reduction,
fused dx).
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend import get_backend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.exceptions import UnsupportedDeviceError
from forge.nn import BatchNorm2d, Conv2d, Sequential

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-4)


def _to_cuda_bn(bn_cpu: BatchNorm2d, num_features: int, affine: bool) -> BatchNorm2d:
    """A CUDA BatchNorm2d with numerically identical parameters/buffers to `bn_cpu`."""
    bn_cuda = BatchNorm2d(num_features, eps=bn_cpu.eps, momentum=bn_cpu.momentum, affine=affine, device="cuda")
    backend = get_backend(bn_cuda.running_mean.device)
    bn_cuda.running_mean._data = backend.from_array(bn_cpu.running_mean.numpy().copy(), bn_cpu.running_mean.numpy().dtype)
    bn_cuda.running_var._data = backend.from_array(bn_cpu.running_var.numpy().copy(), bn_cpu.running_var.numpy().dtype)
    if affine:
        bn_cuda.weight._data = backend.from_array(bn_cpu.weight.numpy().copy(), bn_cpu.weight.numpy().dtype)
        bn_cuda.bias._data = backend.from_array(bn_cpu.bias.numpy().copy(), bn_cpu.bias.numpy().dtype)
    return bn_cuda


# -- Tensor.batch_norm2d() device restriction ---------------------------------


def test_batch_norm2d_tensor_method_rejects_cpu_input():
    x = Tensor(np.zeros((2, 3, 4, 4), dtype=np.float32))
    running_mean = Tensor(np.zeros(3, dtype=np.float32))
    running_var = Tensor(np.ones(3, dtype=np.float32))
    with pytest.raises(UnsupportedDeviceError):
        x.batch_norm2d(None, None, running_mean, running_var, training=True, momentum=0.1, eps=1e-5)


def test_nn_batchnorm2d_forward_dispatches_to_fused_op_on_cuda():
    bn = BatchNorm2d(3, device="cuda")
    x = Tensor(np.random.default_rng(0).standard_normal((2, 3, 4, 4)).astype(np.float32), device="cuda")
    y = bn(x)
    assert isinstance(y._data, CUDAStorage)


# -- training forward/backward parity -----------------------------------------


def test_training_forward_matches_cpu():
    forge.random.seed(0)
    bn_cpu = BatchNorm2d(3)
    bn_cuda = _to_cuda_bn(bn_cpu, 3, affine=True)

    x0 = np.random.default_rng(1).standard_normal((4, 3, 5, 5)).astype(np.float32)
    y_cpu = bn_cpu(Tensor(x0))
    y_cuda = bn_cuda(Tensor(x0, device="cuda"))
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)


def test_training_backward_matches_cpu_for_input_weight_and_bias():
    forge.random.seed(0)
    bn_cpu = BatchNorm2d(3)
    bn_cuda = _to_cuda_bn(bn_cpu, 3, affine=True)

    x0 = np.random.default_rng(2).standard_normal((4, 3, 5, 5)).astype(np.float32)
    x_cpu = Tensor(x0, requires_grad=True)
    x_cuda = Tensor(x0, device="cuda", requires_grad=True)

    bn_cpu(x_cpu).sum().backward()
    bn_cuda(x_cuda).sum().backward()

    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(bn_cuda.weight.grad.to("cpu").numpy(), bn_cpu.weight.grad.numpy(), **TOL)
    np.testing.assert_allclose(bn_cuda.bias.grad.to("cpu").numpy(), bn_cpu.bias.grad.numpy(), **TOL)


def test_running_stats_update_matches_cpu():
    forge.random.seed(0)
    bn_cpu = BatchNorm2d(3, momentum=0.3)
    bn_cuda = _to_cuda_bn(bn_cpu, 3, affine=True)

    x0 = np.random.default_rng(3).standard_normal((4, 3, 5, 5)).astype(np.float32)
    bn_cpu(Tensor(x0))
    bn_cuda(Tensor(x0, device="cuda"))

    np.testing.assert_allclose(bn_cuda.running_mean.to("cpu").numpy(), bn_cpu.running_mean.numpy(), **TOL)
    np.testing.assert_allclose(bn_cuda.running_var.to("cpu").numpy(), bn_cpu.running_var.numpy(), **TOL)


def test_non_affine_forward_and_backward_match_cpu():
    forge.random.seed(0)
    bn_cpu = BatchNorm2d(2, affine=False)
    bn_cuda = _to_cuda_bn(bn_cpu, 2, affine=False)

    x0 = np.random.default_rng(4).standard_normal((3, 2, 4, 4)).astype(np.float32)
    x_cpu = Tensor(x0, requires_grad=True)
    x_cuda = Tensor(x0, device="cuda", requires_grad=True)

    y_cpu = bn_cpu(x_cpu)
    y_cuda = bn_cuda(x_cuda)
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)

    y_cpu.sum().backward()
    y_cuda.sum().backward()
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)


# -- eval mode parity -----------------------------------------------------------


def test_eval_forward_and_backward_match_cpu():
    bn_cpu = BatchNorm2d(2)
    bn_cpu.running_mean._data = np.array([0.3, -0.2], dtype=np.float32)
    bn_cpu.running_var._data = np.array([1.5, 0.6], dtype=np.float32)
    bn_cpu.eval()
    bn_cuda = _to_cuda_bn(bn_cpu, 2, affine=True)
    bn_cuda.eval()

    x0 = np.random.default_rng(5).standard_normal((3, 2, 4, 4)).astype(np.float32)
    x_cpu = Tensor(x0, requires_grad=True)
    x_cuda = Tensor(x0, device="cuda", requires_grad=True)

    y_cpu = bn_cpu(x_cpu)
    y_cuda = bn_cuda(x_cuda)
    np.testing.assert_allclose(y_cuda.to("cpu").numpy(), y_cpu.numpy(), **TOL)

    y_cpu.sum().backward()
    y_cuda.sum().backward()
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)


def test_eval_mode_does_not_update_running_stats_on_cuda():
    bn = BatchNorm2d(2, device="cuda")
    bn.eval()
    before_mean = bn.running_mean.to("cpu").numpy().copy()
    before_var = bn.running_var.to("cpu").numpy().copy()
    x = Tensor(np.random.default_rng(6).standard_normal((4, 2, 3, 3)).astype(np.float32), device="cuda")
    bn(x)
    np.testing.assert_array_equal(bn.running_mean.to("cpu").numpy(), before_mean)
    np.testing.assert_array_equal(bn.running_var.to("cpu").numpy(), before_var)


def test_no_gradient_flows_to_running_stats_on_cuda():
    bn = BatchNorm2d(2, device="cuda")
    x = Tensor(np.random.default_rng(7).standard_normal((4, 2, 3, 3)).astype(np.float32), device="cuda", requires_grad=True)
    bn(x).sum().backward()
    assert bn.running_mean.grad is None
    assert bn.running_var.grad is None


# -- device movement ------------------------------------------------------------


def test_to_cuda_moves_batchnorm_buffers_and_parameters():
    bn = BatchNorm2d(3).to("cuda")
    assert bn.running_mean.device.type == "cuda"
    assert bn.running_var.device.type == "cuda"
    assert bn.weight.device.type == "cuda"
    assert bn.bias.device.type == "cuda"


def test_to_cpu_moves_a_cuda_batchnorm_back():
    bn = BatchNorm2d(3, device="cuda")
    bn.to("cpu")
    assert bn.running_mean.device.type == "cpu"
    assert isinstance(bn.running_mean._data, np.ndarray)


# -- integration: Sequential + Conv2d on CUDA ----------------------------------


def test_batchnorm2d_inside_sequential_with_conv2d_on_cuda():
    forge.random.seed(0)
    model = Sequential(Conv2d(1, 4, kernel_size=3), BatchNorm2d(4)).to("cuda")
    x = Tensor(np.random.default_rng(8).standard_normal((3, 1, 8, 8)).astype(np.float32), device="cuda", requires_grad=True)
    y = model(x)
    assert y.device.type == "cuda"
    y.sum().backward()
    bn = model._modules["1"]
    assert bn.weight.grad is not None
    assert bn.weight.grad.device.type == "cuda"
