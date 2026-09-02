"""Milestone 16 CUDA tests: `Dropout` real CUDA execution.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise, matching the convention in
`tests/test_cuda_conv.py`/`tests/test_cuda_backend.py`. Per the milestone
brief (see `docs/architecture/cuda-backend.md`'s **CUDA Dropout** section),
CPU and CUDA masks are never expected to match element-for-element (they
draw from different RNG streams/algorithms by construction) -- these tests
compare shape, dtype, statistical behavior, gradient correctness, and
structural CUDA-only execution instead.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cpu import CPUBackend
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.nn import Dropout

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


# -- forward: real CUDA storage, correct shape/dtype -------------------------


def test_cuda_dropout_forward_produces_real_cuda_storage():
    forge.random.seed(0)
    x = Tensor(np.ones((100, 100), dtype=np.float32), device="cuda")
    out = Dropout(0.5)(x)
    assert out.device.type == "cuda"
    assert isinstance(out._data, CUDAStorage)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_dropout_preserves_dtype(dtype):
    forge.random.seed(1)
    x = Tensor(np.ones((20, 20)), dtype=dtype, device="cuda")
    out = Dropout(0.3)(x)
    assert str(out.dtype) == dtype


# -- statistical behavior on real hardware -----------------------------------


def test_cuda_dropout_zeroes_approximately_p_fraction():
    forge.random.seed(2)
    p = 0.3
    x = Tensor(np.ones((500, 500), dtype=np.float32), device="cuda")
    out = Dropout(p)(x).to("cpu").numpy()
    frac_zero = float((out == 0).mean())
    assert abs(frac_zero - p) < 0.02


def test_cuda_dropout_preserves_mean_within_tolerance():
    forge.random.seed(3)
    p = 0.4
    x_data = np.random.default_rng(4).standard_normal((400, 400)).astype(np.float32) + 5.0
    out = Dropout(p)(Tensor(x_data, device="cuda")).to("cpu").numpy()
    assert abs(out.mean() - x_data.mean()) < 0.05 * abs(x_data.mean())


def test_cuda_dropout_nonzero_elements_scaled_by_one_over_one_minus_p():
    forge.random.seed(5)
    p = 0.25
    x = Tensor(np.ones((200, 200), dtype=np.float32), device="cuda")
    out = Dropout(p)(x).to("cpu").numpy()
    nonzero = out[out != 0]
    assert nonzero.size > 0
    np.testing.assert_allclose(nonzero, 1.0 / (1.0 - p), rtol=1e-5)


def test_cuda_dropout_two_forward_calls_draw_different_masks():
    forge.random.seed(6)
    d = Dropout(0.5)
    x = Tensor(np.ones((100, 100), dtype=np.float32), device="cuda")
    out1 = d(x).to("cpu").numpy()
    out2 = d(x).to("cpu").numpy()
    assert not np.array_equal(out1, out2)


# -- eval mode: exact identity -------------------------------------------------


def test_cuda_dropout_eval_mode_is_identity():
    d = Dropout(0.5)
    d.eval()
    x_data = np.random.default_rng(7).standard_normal((8, 8)).astype(np.float32)
    x = Tensor(x_data, device="cuda")
    out = d(x)
    assert out is x
    np.testing.assert_array_equal(out.to("cpu").numpy(), x_data)


# -- autograd: backward reuses the forward mask --------------------------------


def test_cuda_dropout_backward_gradient_matches_forward_mask_pattern():
    forge.random.seed(8)
    x = Tensor(np.ones((64, 64), dtype=np.float32), device="cuda", requires_grad=True)
    out = Dropout(0.4)(x)
    out.sum().backward()

    out_np = out.to("cpu").numpy()
    grad_np = x.grad.to("cpu").numpy()
    np.testing.assert_array_equal(out_np != 0, grad_np != 0)
    np.testing.assert_allclose(grad_np[grad_np != 0], out_np[out_np != 0])


def test_cuda_dropout_eval_mode_gradient_is_identity():
    x = Tensor(
        np.random.default_rng(9).standard_normal((6, 6)).astype(np.float32),
        device="cuda", requires_grad=True,
    )
    d = Dropout(0.5)
    d.eval()
    out = d(x)
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.to("cpu").numpy(), np.ones((6, 6), dtype=np.float32))


# -- CPU vs CUDA: statistical agreement, not bit-for-bit -----------------------


def test_cpu_and_cuda_dropout_agree_statistically_not_bitwise():
    p = 0.3
    x_data = np.ones((300, 300), dtype=np.float32)

    forge.random.seed(10)
    cpu_out = Dropout(p)(Tensor(x_data.copy())).numpy()
    forge.random.seed(10)
    cuda_out = Dropout(p)(Tensor(x_data.copy(), device="cuda")).to("cpu").numpy()

    # Different RNG streams by construction -- masks are not expected to match.
    assert not np.array_equal(cpu_out, cuda_out)
    # But both realize the same distribution.
    assert abs(float((cpu_out == 0).mean()) - p) < 0.03
    assert abs(float((cuda_out == 0).mean()) - p) < 0.03


# -- no CPU fallback ------------------------------------------------------------


def test_cuda_dropout_never_calls_cpu_backend(monkeypatch):
    forge.random.seed(11)
    x = Tensor(np.ones((50, 50), dtype=np.float32), device="cuda", requires_grad=True)

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

    out = Dropout(0.4)(x)
    out.sum().backward()

    assert calls == []
    assert isinstance(out._data, CUDAStorage)
    assert isinstance(x.grad._data, CUDAStorage)
