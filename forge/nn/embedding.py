"""Embedding-table lookup layer (Milestone 54).

Added as the one genuine framework gap M54's fresh capability assessment
found and measured: `examples/char_rnn` (Milestone 50) works around the
absence of an embedding/gather primitive with a one-hot-vector-plus-Linear
input encoding, explicitly documented at the time as adequate only "at this
vocabulary size" and a real limitation "at a vocabulary large enough for
one-hot's O(vocab_size) per-step cost to bite (thousands+)". A direct probe
against Forge's real CPU backend (`docs/development/m54-product-direction.md`)
measured that cost directly: 40x at a 30-token vocabulary, growing to
roughly 300x-3500x by a few thousand tokens -- confirming the predicted
threshold with real numbers rather than an estimate. `nn.Embedding` closes
that gap for any vocabulary size, built on the new `Tensor.embedding_lookup()`
fused primitive (`forge/tensor/tensor.py`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import random as forge_random
from ..backend import get_backend
from ..exceptions import ShapeMismatchError
from ..tensor.dtype import DEFAULT_DTYPE
from ..tensor.tensor import Tensor
from .module import Module
from .parameter import Parameter


class Embedding(Module):
    """A lookup table of `num_embeddings` learned vectors, each `embedding_dim` wide.

    `forward(indices)` returns `self.weight[indices]` via the fused
    `Tensor.embedding_lookup()` primitive -- `indices` may be any shape
    (a single `(batch,)` timestep of token ids, or a whole `(batch,
    seq_len)` block looked up in one call); the result has shape
    `indices.shape + (embedding_dim,)`.

    ## Initialization
    `weight` is drawn from `N(0, 1)` (the standard embedding-table init used
    by, e.g., PyTorch's `nn.Embedding` default) -- unlike `Linear`/`Conv2d`'s
    fan-in-scaled uniform bound, there is no "fan-in" for a lookup table (no
    weighted sum of inputs happens inside this layer at all), so there is no
    equivalent scale-preservation argument to make; `N(0, 1)` is simply a
    reasonable, well-precedented starting scale for a table that gradient
    descent will reshape entirely on its own.

    ## Index validation
    `indices` must hold integer values in `[0, num_embeddings)`. Unlike
    `Tensor.embedding_lookup()` itself (which trusts its caller, mirroring
    `Tensor.cross_entropy()`'s own division of responsibility), `Embedding.
    forward()` validates this explicitly -- the same "validate at the
    user-facing layer, trust the fused Tensor primitive" split
    `CrossEntropyLoss`/`Tensor.cross_entropy()` already established.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ShapeMismatchError(
                f"Embedding requires positive num_embeddings/embedding_dim, got "
                f"num_embeddings={num_embeddings}, embedding_dim={embedding_dim}."
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        param_dtype = dtype if dtype is not None else DEFAULT_DTYPE
        rng = generator if generator is not None else forge_random.default_generator()
        weight_data = rng.standard_normal(size=(num_embeddings, embedding_dim))
        self.weight = Parameter(weight_data, dtype=param_dtype, device=device)

    def forward(self, indices: Tensor) -> Tensor:
        if indices.device != self.weight.device:
            raise ShapeMismatchError(
                f"Embedding requires indices on the same device as its weight table; got weight on "
                f"'{self.weight.device}' and indices on '{indices.device}'. Move indices explicitly "
                f"with .to('{self.weight.device}') first -- this is never done automatically."
            )
        # A host read for validation only, device-agnostic (the same
        # `get_backend(device).to_numpy()` convention `CrossEntropyLoss`
        # uses for its own integer target) -- the actual lookup always runs
        # through `Tensor.embedding_lookup()`'s backend dispatch below.
        index_array = get_backend(indices.device).to_numpy(indices._data)
        if not np.issubdtype(index_array.dtype, np.integer):
            raise ShapeMismatchError(
                f"Embedding requires integer-dtype indices, got dtype '{index_array.dtype}'."
            )
        if index_array.size and (
            int(index_array.min()) < 0 or int(index_array.max()) >= self.num_embeddings
        ):
            raise ShapeMismatchError(
                f"Embedding indices must be in [0, {self.num_embeddings}), got values in "
                f"[{int(index_array.min())}, {int(index_array.max())}]."
            )
        if indices.dtype.numpy_dtype != np.int64:
            indices = Tensor(index_array.astype(np.int64, copy=False), device=indices.device)
        return self.weight.embedding_lookup(indices)

    def __repr__(self) -> str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"


__all__ = ["Embedding"]
