"""CUDA `Tensor.sqrt()` / `Tensor.__truediv__` (Milestone 53): parity with CPU.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise, matching the convention in `tests/test_cuda_backend.py`. See
`tests/test_sqrt_div.py` for the CPU-only forward/backward/finite-difference
suite; this file only proves CUDA reaches the same numbers, plus the
CUDA-specific scope restriction (`div` supports exact-shape operands only).
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.exceptions import CUDAError

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-5, atol=1e-5)


def test_cuda_sqrt_forward_matches_cpu():
    data = [1.0, 4.0, 9.0, 16.0]
    cpu = Tensor(data).sqrt()
    cuda = Tensor(data, device="cuda").sqrt()
    np.testing.assert_allclose(cuda.to("cpu").numpy(), cpu.numpy(), **TOL)


def test_cuda_sqrt_backward_matches_cpu():
    data = [1.3, 4.2, 9.9]
    x_cpu = Tensor(data, requires_grad=True)
    x_cuda = Tensor(data, device="cuda", requires_grad=True)
    x_cpu.sqrt().sum().backward()
    x_cuda.sqrt().sum().backward()
    np.testing.assert_allclose(x_cuda.grad.to("cpu").numpy(), x_cpu.grad.numpy(), **TOL)


def test_cuda_div_forward_matches_cpu():
    a_data, b_data = [1.0, 2.0, 3.0], [2.0, 4.0, 5.0]
    cpu = Tensor(a_data) / Tensor(b_data)
    cuda = Tensor(a_data, device="cuda") / Tensor(b_data, device="cuda")
    np.testing.assert_allclose(cuda.to("cpu").numpy(), cpu.numpy(), **TOL)


def test_cuda_div_backward_matches_cpu():
    a_data, b_data = [1.3, -2.1, 0.7], [2.2, 1.4, -3.3]
    a_cpu = Tensor(a_data, requires_grad=True)
    b_cpu = Tensor(b_data, requires_grad=True)
    (a_cpu / b_cpu).sum().backward()

    a_cuda = Tensor(a_data, device="cuda", requires_grad=True)
    b_cuda = Tensor(b_data, device="cuda", requires_grad=True)
    (a_cuda / b_cuda).sum().backward()

    np.testing.assert_allclose(a_cuda.grad.to("cpu").numpy(), a_cpu.grad.numpy(), **TOL)
    np.testing.assert_allclose(b_cuda.grad.to("cpu").numpy(), b_cpu.grad.numpy(), **TOL)


def test_cuda_div_rejects_broadcasting_shapes():
    """No CUDA consumer needs a broadcasting division yet -- see `CUDABackend.div`."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    b = Tensor([2.0, 4.0], device="cuda")
    with pytest.raises(CUDAError):
        _ = a / b
