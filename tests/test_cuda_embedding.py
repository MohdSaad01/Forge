"""CUDA `nn.Embedding` / `Tensor.embedding_lookup()` (Milestone 54): parity with CPU.

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise, matching the convention in `tests/test_cuda_sqrt_div.py`/
`tests/test_batchnorm_cuda.py`. See `tests/test_embedding.py` for the
CPU-only forward/backward/finite-difference/validation suite and
`docs/development/m54-product-direction.md` for the full design writeup.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.exceptions import CUDAError
from forge.nn import Embedding, Parameter

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

TOL = dict(rtol=1e-4, atol=1e-5)


def _make_pair(num_embeddings=12, embedding_dim=5, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    weight_data = rng.standard_normal((num_embeddings, embedding_dim)).astype(dtype)
    cpu = Embedding(num_embeddings, embedding_dim, dtype=dtype)
    cpu.weight = Parameter(weight_data.copy(), dtype=dtype)
    cuda = Embedding(num_embeddings, embedding_dim, dtype=dtype, device="cuda")
    cuda.weight = Parameter(weight_data.copy(), dtype=dtype, device="cuda")
    return cpu, cuda


def test_cuda_embedding_forward_matches_cpu_1d_indices():
    cpu, cuda = _make_pair()
    idx_array = np.array([0, 5, 11, 5, 2], dtype=np.int64)
    out_cpu = cpu(Tensor(idx_array))
    out_cuda = cuda(Tensor(idx_array, device="cuda"))
    assert out_cuda.device.type == "cuda"
    np.testing.assert_allclose(out_cuda.to("cpu").numpy(), out_cpu.numpy(), **TOL)


def test_cuda_embedding_forward_matches_cpu_2d_indices():
    cpu, cuda = _make_pair(num_embeddings=30, embedding_dim=8)
    idx_array = np.random.default_rng(3).integers(0, 30, size=(4, 6)).astype(np.int64)
    out_cpu = cpu(Tensor(idx_array))
    out_cuda = cuda(Tensor(idx_array, device="cuda"))
    assert out_cuda.shape == (4, 6, 8)
    np.testing.assert_allclose(out_cuda.to("cpu").numpy(), out_cpu.numpy(), **TOL)


def test_cuda_embedding_backward_matches_cpu_with_repeated_indices():
    cpu, cuda = _make_pair()
    idx_array = np.array([1, 2, 1, 1, 11, 0], dtype=np.int64)
    upstream = np.random.default_rng(4).standard_normal((6, 5)).astype(np.float32)

    out_cpu = cpu(Tensor(idx_array))
    out_cpu.backward(Tensor(upstream))

    out_cuda = cuda(Tensor(idx_array, device="cuda"))
    out_cuda.backward(Tensor(upstream, device="cuda"))

    np.testing.assert_allclose(cuda.weight.grad.to("cpu").numpy(), cpu.weight.grad.numpy(), **TOL)


def test_cuda_embedding_backward_matches_cpu_float64():
    cpu, cuda = _make_pair(dtype=np.float64)
    idx_array = np.array([3, 3, 7, 9], dtype=np.int64)
    upstream = np.random.default_rng(5).standard_normal((4, 5))

    out_cpu = cpu(Tensor(idx_array))
    out_cpu.backward(Tensor(upstream, dtype=np.float64))
    out_cuda = cuda(Tensor(idx_array, device="cuda"))
    out_cuda.backward(Tensor(upstream, dtype=np.float64, device="cuda"))

    np.testing.assert_allclose(
        cuda.weight.grad.to("cpu").numpy(), cpu.weight.grad.numpy(), rtol=1e-8, atol=1e-9
    )


def test_cuda_embedding_grad_never_reaches_indices():
    _, cuda = _make_pair()
    idx = Tensor(np.array([0, 1], dtype=np.int64), device="cuda")
    out = cuda(idx)
    out.backward(Tensor(np.ones(out.shape, dtype=np.float32), device="cuda"))
    assert idx.grad is None


def test_cuda_embedding_lookup_requires_int64_indices():
    weight = Tensor(np.zeros((5, 3), dtype=np.float32), device="cuda", requires_grad=True)
    bad_indices = Tensor(np.array([0.0, 1.0], dtype=np.float32), device="cuda")
    from forge.backend.cuda.backend import get_cuda_backend
    backend = get_cuda_backend()
    with pytest.raises(CUDAError):
        backend.embedding_lookup(weight._data, bad_indices._data)


def test_cuda_embedding_full_model_no_cpu_fallback(monkeypatch):
    """A structural check: an Embedding->RNNCell forward/backward on CUDA never touches CPUBackend."""
    from forge.backend.cpu import CPUBackend
    from forge.nn import RNNCell, Linear

    cuda_embedding = Embedding(15, 6, device="cuda")
    cell = RNNCell(6, 8, device="cuda")
    out_layer = Linear(8, 15, device="cuda")

    calls = []
    original_methods = {}
    for name in ("add", "sub", "mul", "matmul", "sum", "tanh"):
        original = getattr(CPUBackend, name)
        original_methods[name] = original

        def make_wrapper(n, orig):
            def wrapper(self, *args, **kwargs):
                calls.append(n)
                return orig(self, *args, **kwargs)
            return wrapper

        monkeypatch.setattr(CPUBackend, name, make_wrapper(name, original))

    idx = Tensor(np.array([1, 2, 3], dtype=np.int64), device="cuda")
    h = cell.init_hidden(3, device="cuda")
    x = cuda_embedding(idx)
    h = cell(x, h)
    logits = out_layer(h)
    logits.sum().backward()

    assert calls == [], f"Unexpected CPUBackend calls: {calls}"
