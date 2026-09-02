"""CPU/CUDA numerical consistency tests (Milestone 8).

For every operation implemented on both backends, the CUDA result must
match the CPU reference result (ADR-002) within floating-point tolerance --
never bit-for-bit equality. Skipped cleanly when CUDA is unavailable. See
`docs/architecture/cuda-backend.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.backend.cuda import is_cuda_available

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-5, atol=1e-5)


def _both(data, dtype):
    cpu = Tensor(data, dtype=dtype)
    cuda = cpu.to("cuda")
    return cpu, cuda


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_elementwise_consistency(op, dtype):
    a_data = [[1.0, -2.0, 3.5], [4.0, 5.5, -6.0]]
    b_data = [[10.0, 20.0, -30.0], [1.0, -2.0, 3.0]]
    a_cpu, a_cuda = _both(a_data, dtype)
    b_cpu, b_cuda = _both(b_data, dtype)

    if op == "add":
        cpu_result, cuda_result = a_cpu + b_cpu, a_cuda + b_cuda
    elif op == "sub":
        cpu_result, cuda_result = a_cpu - b_cpu, a_cuda - b_cuda
    else:
        cpu_result, cuda_result = a_cpu * b_cpu, a_cuda * b_cuda

    assert cuda_result.dtype == cpu_result.dtype
    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((3,), (3,)),  # vector . vector
        ((4, 3), (3,)),  # matrix . vector
        ((3,), (3, 5)),  # vector . matrix
        ((4, 3), (3, 5)),  # matrix . matrix
    ],
)
def test_matmul_consistency(dtype, a_shape, b_shape):
    rng = np.random.default_rng(0)
    a_data = rng.standard_normal(a_shape)
    b_data = rng.standard_normal(b_shape)
    a_cpu, a_cuda = _both(a_data, dtype)
    b_cpu, b_cuda = _both(b_data, dtype)

    cpu_result = a_cpu @ b_cpu
    cuda_result = a_cuda @ b_cuda

    assert cuda_result.shape == cpu_result.shape
    assert cuda_result.dtype == cpu_result.dtype
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 4)])
def test_sum_full_reduction_consistency(dtype, shape):
    rng = np.random.default_rng(1)
    data = rng.standard_normal(shape)
    cpu, cuda = _both(data, dtype)

    cpu_result = cpu.sum()
    cuda_result = cuda.sum()

    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(3, 4), (1, 5), (6, 1)])
def test_sum_axis1_consistency(dtype, shape):
    """Milestone 14: axis=1 reduction on a 2D tensor."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal(shape)
    cpu, cuda = _both(data, dtype)

    for keepdims in (False, True):
        cpu_result = cpu.sum(axis=1, keepdims=keepdims)
        cuda_result = cuda.sum(axis=1, keepdims=keepdims)
        assert cuda_result.shape == cpu_result.shape
        np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exp_consistency(dtype):
    data = [[-3.0, 2.5, 0.0], [1.5, -0.1, 4.0]]
    cpu, cuda = _both(data, dtype)
    cpu_result = cpu.exp()
    cuda_result = cuda.exp()
    assert cuda_result.dtype == cpu_result.dtype
    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_log_consistency(dtype):
    data = [[0.5, 2.5, 1.0], [1.5, 10.0, 4.0]]
    cpu, cuda = _both(data, dtype)
    cpu_result = cpu.log()
    cuda_result = cuda.log()
    assert cuda_result.dtype == cpu_result.dtype
    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_relu_consistency(dtype):
    data = [[-3.0, 2.5, 0.0], [1.5, -0.1, 4.0]]
    cpu, cuda = _both(data, dtype)
    cpu_result = cpu.relu()
    cuda_result = cuda.relu()
    assert cuda_result.dtype == cpu_result.dtype
    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reshape_consistency(dtype):
    cpu, cuda = _both([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)
    cpu_result = cpu.reshape(2, 3)
    cuda_result = cuda.reshape(2, 3)
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


def test_chained_ops_consistency():
    """A small chain of ops, matching the kind of expression a model forward pass builds."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    w = rng.standard_normal((3, 2)).astype(np.float32)
    b = rng.standard_normal((2,)).astype(np.float32)

    x_cpu, x_cuda = Tensor(x), Tensor(x).to("cuda")
    w_cpu, w_cuda = Tensor(w), Tensor(w).to("cuda")

    cpu_result = (x_cpu @ w_cpu).sum()
    cuda_result = (x_cuda @ w_cuda).sum()

    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)


def test_chained_linear_relu_linear_consistency():
    """Matches the exact op sequence a `Linear -> ReLU -> Linear` model forward runs."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    w1 = rng.standard_normal((4, 6)).astype(np.float32)
    b1 = rng.standard_normal((6,)).astype(np.float32)
    w2 = rng.standard_normal((6, 2)).astype(np.float32)
    b2 = rng.standard_normal((2,)).astype(np.float32)

    def run(to_device):
        xt, w1t, b1t, w2t, b2t = (to_device(Tensor(v)) for v in (x, w1, b1, w2, b2))
        h = (xt @ w1t + b1t).relu()
        return h @ w2t + b2t

    cpu_result = run(lambda t: t)
    cuda_result = run(lambda t: t.to("cuda"))

    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result.to("cpu").numpy(), cpu_result.numpy(), **TOL)
