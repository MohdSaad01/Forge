"""`nn.Embedding` / `Tensor.embedding_lookup()` (Milestone 54): CPU forward/backward/validation.

CPU's `Tensor.embedding_lookup()` is a fused primitive backed by NumPy
fancy indexing (forward) and `np.add.at` scatter-add (backward), like
`cross_entropy`/`batch_norm2d` -- there is no separate "composed from other
ops" CPU path to test the way `BatchNorm2d` has. See
`tests/test_cuda_embedding.py` for CUDA parity and
`docs/development/m54-product-direction.md` for the full design writeup and
the measured evidence this was built against.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.exceptions import ShapeMismatchError
from forge.nn import Embedding, Parameter

TOL = dict(rtol=1e-6, atol=1e-7)


# -- construction / validation ------------------------------------------------


def test_embedding_rejects_non_positive_num_embeddings():
    with pytest.raises(ShapeMismatchError):
        Embedding(0, 4)


def test_embedding_rejects_non_positive_embedding_dim():
    with pytest.raises(ShapeMismatchError):
        Embedding(10, 0)


def test_embedding_weight_shape():
    emb = Embedding(10, 4)
    assert emb.weight.shape == (10, 4)
    assert isinstance(emb.weight, Parameter)
    assert emb.weight.requires_grad


def test_embedding_rejects_out_of_range_index():
    emb = Embedding(5, 3)
    idx = Tensor(np.array([0, 5], dtype=np.int64))
    with pytest.raises(ShapeMismatchError):
        emb(idx)


def test_embedding_rejects_negative_index():
    emb = Embedding(5, 3)
    idx = Tensor(np.array([-1, 2], dtype=np.int64))
    with pytest.raises(ShapeMismatchError):
        emb(idx)


def test_embedding_rejects_non_integer_indices():
    emb = Embedding(5, 3)
    idx = Tensor(np.array([0.0, 1.0], dtype=np.float32))
    with pytest.raises(ShapeMismatchError):
        emb(idx)


def test_tensor_embedding_lookup_rejects_non_2d_table():
    table = Tensor(np.zeros((5, 3, 2), dtype=np.float32), requires_grad=True)
    idx = Tensor(np.array([0], dtype=np.int64))
    with pytest.raises(ShapeMismatchError):
        table.embedding_lookup(idx)


# -- forward correctness -------------------------------------------------------


def test_embedding_forward_selects_correct_rows():
    forge.random.seed(0)
    emb = Embedding(6, 4)
    idx = Tensor(np.array([0, 3, 5, 3], dtype=np.int64))
    out = emb(idx)
    assert out.shape == (4, 4)
    expected = emb.weight.numpy()[np.array([0, 3, 5, 3])]
    np.testing.assert_allclose(out.numpy(), expected, **TOL)


def test_embedding_forward_arbitrary_index_shape():
    forge.random.seed(0)
    emb = Embedding(20, 6)
    idx_array = np.random.default_rng(0).integers(0, 20, size=(5, 7))
    idx = Tensor(idx_array.astype(np.int64))
    out = emb(idx)
    assert out.shape == (5, 7, 6)
    np.testing.assert_allclose(out.numpy(), emb.weight.numpy()[idx_array], **TOL)


def test_embedding_forward_accepts_1d_int32_indices():
    forge.random.seed(0)
    emb = Embedding(6, 4)
    idx = Tensor(np.array([1, 2], dtype=np.int32))
    out = emb(idx)
    np.testing.assert_allclose(out.numpy(), emb.weight.numpy()[[1, 2]], **TOL)


# -- backward correctness -------------------------------------------------------


def test_embedding_backward_accumulates_repeated_indices():
    forge.random.seed(0)
    emb = Embedding(10, 4)
    idx = Tensor(np.array([1, 2, 1, 9], dtype=np.int64))
    out = emb(idx)
    out.backward(Tensor(np.ones(out.shape, dtype=np.float32)))
    grad = emb.weight.grad.numpy()
    np.testing.assert_allclose(grad[1], np.full(4, 2.0), **TOL)
    np.testing.assert_allclose(grad[9], np.full(4, 1.0), **TOL)
    np.testing.assert_allclose(grad[0], np.zeros(4), **TOL)
    np.testing.assert_allclose(grad[3], np.zeros(4), **TOL)


def test_embedding_backward_finite_difference():
    rng = np.random.default_rng(2)
    weight_data = rng.standard_normal((8, 5))
    emb = Embedding(8, 5, dtype=np.float64)
    emb.weight = Parameter(weight_data, dtype=np.float64)
    idx_array = rng.integers(0, 8, size=(6,))
    idx = Tensor(idx_array.astype(np.int64))
    upstream = rng.standard_normal((6, 5))

    out = emb(idx)
    out.backward(Tensor(upstream, dtype=np.float64))
    analytic = emb.weight.grad.numpy().copy()

    eps = 1e-6
    numeric = np.zeros_like(weight_data)
    for i in range(weight_data.shape[0]):
        for j in range(weight_data.shape[1]):
            wp = weight_data.copy(); wp[i, j] += eps
            wm = weight_data.copy(); wm[i, j] -= eps
            lp = np.sum(wp[idx_array] * upstream)
            lm = np.sum(wm[idx_array] * upstream)
            numeric[i, j] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_embedding_no_gradient_reaches_indices():
    emb = Embedding(5, 3)
    idx = Tensor(np.array([0, 1], dtype=np.int64))
    out = emb(idx)
    out.backward(Tensor(np.ones(out.shape, dtype=np.float32)))
    assert idx.grad is None
    assert idx.requires_grad is False


# -- integration with the rest of Forge ------------------------------------------


def test_embedding_composes_with_module_to_device_moves_weight_only():
    emb = Embedding(5, 3)
    original_weight = emb.weight.numpy().copy()
    assert emb.device is not None  # Parameter present, single device
    moved = emb.to("cpu")  # no-op move, proves .to() still works with an Embedding present
    np.testing.assert_allclose(moved.weight.numpy(), original_weight)


def test_embedding_repr():
    emb = Embedding(10, 4)
    assert "num_embeddings=10" in repr(emb)
    assert "embedding_dim=4" in repr(emb)
