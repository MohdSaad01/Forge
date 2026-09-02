"""`Dropout`: the first training/evaluation-dependent stochastic Module."""

from __future__ import annotations

import numpy as np

from .. import random as forge_random
from ..exceptions import ModuleError
from ..tensor.tensor import Tensor
from .module import Module


class Dropout(Module):
    """Inverted dropout: zeroes each element independently with probability `p` while training.

    ```text
    training: mask ~ Bernoulli(1-p); output = input * mask / (1-p)
    eval:     output = input
    ```

    The `/ (1-p)` rescaling happens during training (not at eval time --
    "inverted" dropout, the standard convention), so `eval()`'s forward pass
    needs no compensating scale and is a plain identity.

    Reads its own `self.training` flag (`Module.train()`/`Module.eval()`,
    `docs/architecture/modules.md`) -- no new state mechanism. No
    parameters: dropout has nothing to learn.

    ## Randomness
    Draws come from `forge.random.default_generator()` unless a
    `numpy.random.Generator` is passed explicitly via `Dropout(...,
    generator=...)`, fetched fresh on every `forward()` call (not
    snapshotted at construction, unlike `Linear`/`Conv2d`'s one-time
    initialization draw) so a `forge.random.seed(...)` call always governs
    whatever Dropout draws happen after it, including across many training
    steps. No second global RNG is introduced.

    ## Autograd
    `forward()` composes exactly one Tensor operation:
    `x * x.dropout_mask(p, rng)` (`Tensor.dropout_mask`,
    `forge/tensor/tensor.py`). The mask is a plain `requires_grad=False`
    leaf with the `/(1-p)` scaling already baked into its `0`/`1/(1-p)`
    values, so ordinary `mul` autograd gives exactly the required forward
    and backward behavior (`grad_input = grad_output * mask / (1-p)`) with
    no Dropout-specific backward rule, and the *same* mask instance
    computed in forward is what `mul`'s backward closure captures and
    reuses -- backward never redraws the mask. During evaluation,
    `forward()` returns `x` itself unchanged (no new graph node), so
    gradients flow through exactly as if Dropout were not present.

    ## CUDA
    Works unmodified on CUDA: `Tensor.dropout_mask` dispatches to
    `Backend.dropout_mask`, which for `CUDABackend` generates every mask
    element with a real on-device kernel (never NumPy, never a host round
    trip) -- see `docs/architecture/cuda-backend.md`'s **CUDA Dropout**
    section. Exact per-element mask values are not expected to match CPU
    (different RNG streams by construction); only statistical/semantic
    behavior is compared across backends.
    """

    def __init__(self, p: float = 0.5, generator: "np.random.Generator | None" = None):
        super().__init__()
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 <= p < 1.0):
            raise ModuleError(f"Dropout requires 0 <= p < 1, got p={p!r}.")
        self.p = float(p)
        self._generator = generator

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        rng = self._generator if self._generator is not None else forge_random.default_generator()
        mask = x.dropout_mask(self.p, rng)
        return x * mask

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


__all__ = ["Dropout"]
