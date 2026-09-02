"""CUDA-only hardware tests (Milestone 8).

Every test in this module requires an actual working CUDA backend (a
CUDA-capable GPU, driver, and `nvcc`+MSVC toolchain) and is skipped cleanly
otherwise via the module-level `pytestmark`. These are the tests that prove
CUDA execution is real: device dispatch is structurally distinct from CPU,
tensors are backed by genuine device memory (not a NumPy array wearing a
"cuda" label), kernels actually execute and synchronize, and unsupported
operations fail with a clear `CUDAError` rather than silently running on
CPU. `relu` is a real CUDA kernel as of Milestone 9 (see `test_module_cuda.py`
for high-level `nn.Module`/`Linear`/`ReLU` CUDA execution tests); `exp`/`log`
and `sum(axis=1)` are real CUDA kernels as of Milestone 14 (see
`tests/test_cuda_loss.py` for `CrossEntropyLoss`, which is what they exist
for). As of Milestone 10, CUDA tensors support reverse-mode
autograd for this module's supported forward operations -- see
`tests/test_cuda_autograd.py` -- so this file's own forward-only tests build
their differentiable-op examples with `requires_grad=False` leaves (the
default) to stay focused on forward execution. See
`docs/architecture/cuda-backend.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend import get_backend
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDABackend, CUDAStorage, is_cuda_available
from forge.backend.device import Device
from forge.exceptions import CUDAError, UnsupportedDeviceError

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-5, atol=1e-5)


# -- 1. CUDA initializes -----------------------------------------------------


def test_cuda_backend_initializes():
    backend = get_backend(Device.parse("cuda"))
    assert isinstance(backend, CUDABackend)
    assert backend.device_count >= 1


def test_backend_dispatch_is_structurally_distinct_from_cpu():
    """Proves `device="cuda"` cannot simply be invoking `CPUBackend`."""
    cpu_backend = get_backend(Device.parse("cpu"))
    cuda_backend = get_backend(Device.parse("cuda"))
    assert isinstance(cpu_backend, CPUBackend)
    assert isinstance(cuda_backend, CUDABackend)
    assert type(cpu_backend) is not type(cuda_backend)
    assert not isinstance(cuda_backend, CPUBackend)


def test_invalid_cuda_device_index_raises_clearly():
    with pytest.raises(CUDAError, match="index"):
        get_backend(Device.parse("cuda:1"))


# -- 2 & 4. CPU -> CUDA -> CPU transfer --------------------------------------


def test_tensor_moves_cpu_to_cuda():
    cpu_t = Tensor([1.0, 2.0, 3.0])
    cuda_t = cpu_t.to("cuda")
    assert cuda_t.device.type == "cuda"
    assert cuda_t.device.type != cpu_t.device.type
    # Structural proof it is real device storage, never a relabeled NumPy array.
    assert isinstance(cuda_t._data, CUDAStorage)
    assert not isinstance(cuda_t._data, np.ndarray)


def test_cuda_to_cpu_transfer_returns_correct_values():
    original = [1.0, -2.5, 3.25, 4.0]
    cuda_t = Tensor(original).to("cuda")
    back = cuda_t.to("cpu")
    assert back.device.type == "cpu"
    np.testing.assert_allclose(back.numpy(), original, **TOL)


def test_transfer_preserves_dtype():
    cpu_t = Tensor([1.0, 2.0], dtype="float64")
    cuda_t = cpu_t.to("cuda")
    assert str(cuda_t.dtype) == "float64"
    back = cuda_t.to("cpu")
    assert str(back.dtype) == "float64"


def test_to_same_device_is_a_no_op():
    cpu_t = Tensor([1.0, 2.0])
    assert cpu_t.to("cpu") is cpu_t
    cuda_t = cpu_t.to("cuda")
    assert cuda_t.to("cuda") is cuda_t


def test_transfer_does_not_share_memory_between_devices():
    """A CUDA tensor must own its own device buffer, not alias CPU memory."""
    cpu_t = Tensor([1.0, 2.0, 3.0])
    cuda_t = cpu_t.to("cuda")
    back = cuda_t.to("cpu")
    back.numpy()[0] = 999.0
    # Mutating the round-tripped copy must not affect the original CPU tensor
    # or a fresh read-back from the GPU buffer.
    assert cpu_t.numpy()[0] == 1.0
    np.testing.assert_allclose(cuda_t.to("cpu").numpy(), [1.0, 2.0, 3.0], **TOL)


# -- 3 & 5. Elementwise kernel execution --------------------------------------


def test_elementwise_add_executes_on_gpu():
    a = Tensor([1.0, 2.0, 3.0]).to("cuda")
    b = Tensor([10.0, 20.0, 30.0]).to("cuda")
    result = a + b
    assert result.device.type == "cuda"
    assert isinstance(result._data, CUDAStorage)
    np.testing.assert_allclose(result.to("cpu").numpy(), [11.0, 22.0, 33.0], **TOL)


def test_elementwise_sub_and_mul_execute_on_gpu():
    a = Tensor([5.0, 7.0, 9.0]).to("cuda")
    b = Tensor([1.0, 2.0, 3.0]).to("cuda")
    np.testing.assert_allclose((a - b).to("cpu").numpy(), [4.0, 5.0, 6.0], **TOL)
    np.testing.assert_allclose((a * b).to("cpu").numpy(), [5.0, 14.0, 27.0], **TOL)


def test_row_broadcast_add_matches_bias_shape_executes_on_gpu():
    """Milestone 9: (rows, cols) + (cols,) is supported (needed for Linear's bias add)."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    b = Tensor([10.0, 20.0]).to("cuda")
    result = a + b
    assert result.device.type == "cuda"
    assert isinstance(result._data, CUDAStorage)
    np.testing.assert_allclose(result.to("cpu").numpy(), [[11.0, 22.0], [13.0, 24.0]], **TOL)


def test_row_broadcast_reversed_operand_order_on_cuda():
    a = Tensor([10.0, 20.0]).to("cuda")
    b = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    result = a + b
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.to("cpu").numpy(), [[11.0, 22.0], [13.0, 24.0]], **TOL)


def test_row_broadcast_sub_respects_operand_order_on_cuda():
    mat = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    vec = Tensor([10.0, 20.0]).to("cuda")
    np.testing.assert_allclose((mat - vec).to("cpu").numpy(), [[-9.0, -18.0], [-7.0, -16.0]], **TOL)
    np.testing.assert_allclose((vec - mat).to("cpu").numpy(), [[9.0, 18.0], [7.0, 16.0]], **TOL)


def test_column_broadcast_sub_matches_cross_entropy_shift_shape_on_gpu():
    """Milestone 14: (rows, cols) - (rows, 1) is supported for `sub` only --
    the shape CrossEntropyLoss's log-sum-exp shift needs."""
    mat = Tensor([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]]).to("cuda")
    colvec = Tensor([[10.0], [20.0]]).to("cuda")
    result = mat - colvec
    assert result.device.type == "cuda"
    assert isinstance(result._data, CUDAStorage)
    np.testing.assert_allclose(
        result.to("cpu").numpy(), [[-9.0, -8.0, -7.0], [-16.0, -21.0, -19.5]], **TOL
    )


def test_column_broadcast_sub_reversed_operand_order_on_cuda():
    colvec = Tensor([[10.0], [20.0]]).to("cuda")
    mat = Tensor([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]]).to("cuda")
    result = colvec - mat
    np.testing.assert_allclose(
        result.to("cpu").numpy(), [[9.0, 8.0, 7.0], [16.0, 21.0, 19.5]], **TOL
    )


def test_column_broadcast_add_and_mul_remain_unsupported_on_cuda():
    """Only `sub` gained column-broadcast support -- add/mul with a (rows, 1)
    operand still isn't a shape either backend needs to handle."""
    mat = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    colvec = Tensor([[10.0], [20.0]]).to("cuda")
    with pytest.raises(CUDAError, match="broadcast"):
        mat + colvec
    with pytest.raises(CUDAError, match="broadcast"):
        mat * colvec


def test_elementwise_broadcasting_beyond_the_row_case_is_unsupported_on_cuda():
    """General N-D broadcasting is still out of scope -- only the row-vector case is supported."""
    a = Tensor([[[1.0, 2.0], [3.0, 4.0]]]).to("cuda")  # (1, 2, 2)
    b = Tensor([10.0, 20.0]).to("cuda")  # (2,), does not fit the (rows, cols)+(cols,) pattern
    with pytest.raises(CUDAError, match="broadcast"):
        a + b


def test_elementwise_shape_mismatch_raises_cuda_error_not_a_crash():
    a = Tensor([1.0, 2.0, 3.0]).to("cuda")
    b = Tensor([1.0, 2.0]).to("cuda")
    # The Tensor-level broadcast check already rejects fully incompatible
    # shapes before reaching the backend.
    with pytest.raises(forge.ShapeMismatchError):
        a + b


# -- 6. Matrix multiplication -------------------------------------------------


def test_matmul_2d_executes_on_gpu():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    b = Tensor([[5.0, 6.0], [7.0, 8.0]]).to("cuda")
    result = (a @ b).to("cpu")
    np.testing.assert_allclose(result.numpy(), [[19.0, 22.0], [43.0, 50.0]], **TOL)


def test_matmul_vector_dot_product_on_gpu():
    a = Tensor([1.0, 2.0, 3.0]).to("cuda")
    b = Tensor([4.0, 5.0, 6.0]).to("cuda")
    result = (a @ b).to("cpu")
    assert result.shape == ()
    np.testing.assert_allclose(result.numpy(), 32.0, **TOL)


def test_matmul_matrix_vector_on_gpu():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    b = Tensor([1.0, 1.0]).to("cuda")
    result = (a @ b).to("cpu")
    np.testing.assert_allclose(result.numpy(), [3.0, 7.0], **TOL)


def test_matmul_vector_matrix_on_gpu():
    a = Tensor([1.0, 1.0]).to("cuda")
    b = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    result = (a @ b).to("cpu")
    np.testing.assert_allclose(result.numpy(), [4.0, 6.0], **TOL)


def test_matmul_inner_dimension_mismatch_raises_shape_error_on_cuda():
    a = Tensor([[1.0, 2.0, 3.0]]).to("cuda")  # (1, 3)
    b = Tensor([[1.0, 2.0]]).to("cuda")  # (1, 2)
    with pytest.raises(forge.ShapeMismatchError):
        a @ b


# -- 7. Reduction --------------------------------------------------------------


def test_sum_full_reduction_on_gpu():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    result = t.sum().to("cpu")
    assert result.shape == ()
    np.testing.assert_allclose(result.numpy(), 10.0, **TOL)


def test_sum_axis0_is_unsupported_on_cuda():
    """Milestone 14 adds axis=1 support only -- axis=0 (or any other axis) still isn't."""
    t = Tensor([[1.0, 2.0], [3.0, 4.0]]).to("cuda")
    with pytest.raises(CUDAError, match="axis"):
        t.sum(axis=0)


def test_sum_axis1_executes_on_gpu_and_produces_correct_values():
    """Milestone 14: sum(axis=1) is a real CUDA kernel, needed by CrossEntropyLoss."""
    t = Tensor([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]]).to("cuda")
    result = t.sum(axis=1)
    assert result.device.type == "cuda"
    assert isinstance(result._data, CUDAStorage)
    assert result.shape == (2,)
    np.testing.assert_allclose(result.to("cpu").numpy(), [6.0, 3.5], **TOL)


def test_sum_axis1_keepdims_on_gpu():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]]).to("cuda")
    result = t.sum(axis=1, keepdims=True)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result.to("cpu").numpy(), [[6.0], [3.5]], **TOL)


def test_reshape_on_gpu_preserves_values():
    t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to("cuda")
    result = t.reshape(2, 3).to("cpu")
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], **TOL)


def test_reshape_invalid_size_raises_shape_error_on_cuda():
    t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to("cuda")
    with pytest.raises(forge.ShapeMismatchError):
        t.reshape(2, 3)


# -- 8. Unsupported operations fail clearly -----------------------------------


def test_relu_executes_on_gpu_and_produces_correct_values():
    """Milestone 9: relu is a real CUDA kernel, not unsupported."""
    t = Tensor([1.0, -2.0, 0.0, 3.5, -0.001]).to("cuda")
    result = t.relu()
    assert result.device.type == "cuda"
    assert isinstance(result._data, CUDAStorage)
    np.testing.assert_allclose(result.to("cpu").numpy(), [1.0, 0.0, 0.0, 3.5, 0.0], **TOL)


def test_relu_on_all_negative_and_all_positive_inputs():
    negative = Tensor([-1.0, -5.0, -0.5]).to("cuda")
    positive = Tensor([1.0, 5.0, 0.5]).to("cuda")
    np.testing.assert_allclose(negative.relu().to("cpu").numpy(), [0.0, 0.0, 0.0], **TOL)
    np.testing.assert_allclose(positive.relu().to("cpu").numpy(), [1.0, 5.0, 0.5], **TOL)


def test_relu_unsupported_dtype_raises_clearly_on_cuda():
    t = Tensor([1, -2, 3], dtype="int32").to("cuda")
    with pytest.raises(CUDAError, match="dtype"):
        t.relu()


def test_exp_executes_on_gpu_and_produces_correct_values():
    """Milestone 14: exp is a real CUDA kernel, needed by CrossEntropyLoss."""
    t = Tensor([0.0, 1.0, -1.0, 2.0]).to("cuda")
    result = t.exp()
    assert result.device.type == "cuda"
    assert isinstance(result._data, CUDAStorage)
    np.testing.assert_allclose(result.to("cpu").numpy(), np.exp([0.0, 1.0, -1.0, 2.0]), **TOL)


def test_log_executes_on_gpu_and_produces_correct_values():
    t = Tensor([1.0, np.e, 0.5, 10.0]).to("cuda")
    result = t.log()
    assert result.device.type == "cuda"
    assert isinstance(result._data, CUDAStorage)
    np.testing.assert_allclose(result.to("cpu").numpy(), np.log([1.0, np.e, 0.5, 10.0]), **TOL)


def test_exp_unsupported_dtype_raises_clearly_on_cuda():
    t = Tensor([1, 2, 3], dtype="int32").to("cuda")
    with pytest.raises(CUDAError, match="dtype"):
        t.exp()


def test_log_unsupported_dtype_raises_clearly_on_cuda():
    t = Tensor([1, 2, 3], dtype="int32").to("cuda")
    with pytest.raises(CUDAError, match="dtype"):
        t.log()


def test_int_dtype_arithmetic_is_unsupported_on_cuda():
    a = Tensor([1, 2, 3], dtype="int32").to("cuda")
    b = Tensor([4, 5, 6], dtype="int32").to("cuda")
    with pytest.raises(CUDAError, match="dtype"):
        a + b


def test_int_dtype_transfer_itself_is_supported():
    """Transfer/storage is dtype-generic even though compute kernels are float-only."""
    a = Tensor([1, 2, 3], dtype="int32")
    cuda_a = a.to("cuda")
    assert str(cuda_a.dtype) == "int32"
    back = cuda_a.to("cpu")
    np.testing.assert_array_equal(back.numpy(), [1, 2, 3])


# -- Device mismatch -----------------------------------------------------------


def test_cpu_cuda_device_mismatch_raises_clearly():
    cpu_t = Tensor([1.0, 2.0, 3.0])
    cuda_t = cpu_t.to("cuda")
    with pytest.raises(UnsupportedDeviceError):
        cpu_t + cuda_t


def test_constructing_tensor_across_devices_without_to_raises_clearly():
    cuda_t = Tensor([1.0, 2.0]).to("cuda")
    with pytest.raises(UnsupportedDeviceError):
        Tensor(cuda_t, device="cpu")


# -- Autograd: forward and backward supported for this module's ops (Milestone 10) --
#
# See `tests/test_cuda_autograd.py` for the full CUDA autograd test suite
# (numerical CPU/CUDA gradient comparisons, Linear/ReLU/multi-layer backward,
# device-mismatch errors, exp/log's clear CUDAError). These two tests just
# confirm the *this-module's-scope* forward-only boundary from Milestones 8/9
# is now gone: a differentiable op on a grad-requiring CUDA tensor no longer
# raises, and `backward()` is no longer restricted to `cpu`.


def test_differentiable_op_on_cuda_tensor_no_longer_raises():
    a = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
    b = Tensor([3.0, 4.0], device="cuda", requires_grad=True)
    c = a + b
    assert c.requires_grad is True
    assert c.grad_fn is not None
    assert c.device.type == "cuda"


def test_backward_on_cuda_tensor_no_longer_raises():
    t = Tensor([1.0, 2.0], device="cuda", requires_grad=True)
    t.backward(Tensor([1.0, 1.0], device="cuda"))
    assert t.grad is not None
    assert t.grad.device.type == "cuda"
    np.testing.assert_allclose(t.grad.to("cpu").numpy(), [1.0, 1.0], **TOL)


def test_transfer_never_carries_requires_grad_across_devices():
    cpu_t = Tensor([1.0, 2.0], requires_grad=True)
    cuda_t = cpu_t.to("cuda")
    assert cuda_t.requires_grad is False
    assert cuda_t.is_leaf is True
