"""`BatchNorm2d`: Forge's first buffer-owning, train/eval-mode-dependent normalization layer.

See `docs/development/m53-batchnorm.md` for the full design writeup (why
`Module` needed a buffer concept, the CPU/CUDA implementation split, and the
autograd/serialization/device-movement validation). Summary:

- `running_mean`/`running_var` are `Module` buffers (`register_buffer()`,
  `forge/nn/module.py`) -- non-trainable, non-differentiable, but moved by
  `.to(device)` and persisted by `save_model`/`save_checkpoint` exactly like
  `weight`/`bias`.
- On CPU, the forward pass composes entirely from existing/newly-added
  general `Tensor` primitives (`sum`, `-`, `*`, `.sqrt()`, `/`, `.reshape()`)
  -- CPU already supports the `(0, 2, 3)`-axis reduction and `(1, C, 1, 1)`
  vs. `(N, C, H, W)` broadcasting these compose from, so backward is entirely
  automatic (no BatchNorm-specific CPU backward rule is written anywhere).
- On CUDA, the same composition is not possible (`CUDABackend.sum()` has no
  multi-axis reduction and CUDA broadcasting is scoped to two shapes neither
  of which is this one), so the forward pass instead calls one dedicated
  fused primitive, `Tensor.batch_norm2d()` (CUDA-only, `forge/tensor/
  tensor.py`), backed by a real forward+backward CUDA kernel pair
  (`forge/backend/cuda/backend.py`/`kernels.cu`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..autograd import no_grad
from ..exceptions import ShapeMismatchError
from ..tensor.dtype import DEFAULT_DTYPE
from ..tensor.tensor import Tensor
from .module import Module
from .parameter import Parameter


class BatchNorm2d(Module):
    """Per-channel batch normalization over an `(N, C, H, W)` input.

    ```text
    training: mu_c, var_c = batch mean/(biased) variance of x[:, c, :, :] over (N, H, W)
              xhat = (x - mu) / sqrt(var + eps)
              running_mean <- (1 - momentum) * running_mean + momentum * mu
              running_var  <- (1 - momentum) * running_var  + momentum * var_unbiased
    eval:     xhat = (x - running_mean) / sqrt(running_var + eps)
    output:   affine=True  -> weight * xhat + bias   (weight/bias: Parameters, shape (C,))
              affine=False -> xhat
    ```

    `running_var`'s momentum update uses the *unbiased* estimator
    (`var * m / (m - 1)`, `m = N*H*W`) while normalization itself uses the
    *biased* estimator (dividing by `m`) -- the same convention most
    normalization implementations use (an unbiased running estimate of the
    population variance, but the standard biased in-batch normalization).
    `m <= 1` skips the unbiased correction (falls back to the biased value)
    rather than dividing by zero.

    Reads `self.training` (`Module.train()`/`Module.eval()`) to choose
    batch-vs-running statistics -- the second consumer of that
    train/eval-mode-dependent-forward pattern after `Dropout` (M16).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        dtype: Any = None,
        device: str = "cpu",
    ):
        super().__init__()
        if num_features <= 0:
            raise ShapeMismatchError(f"BatchNorm2d requires num_features > 0, got {num_features}.")
        if not (0.0 <= momentum <= 1.0):
            raise ShapeMismatchError(f"BatchNorm2d requires 0 <= momentum <= 1, got {momentum}.")

        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)

        param_dtype = dtype if dtype is not None else DEFAULT_DTYPE
        if self.affine:
            self.weight = Parameter(np.ones(num_features), dtype=param_dtype, device=device)
            self.bias = Parameter(np.zeros(num_features), dtype=param_dtype, device=device)
        else:
            self.weight = None
            self.bias = None

        self.register_buffer(
            "running_mean", Tensor(np.zeros(num_features), dtype=param_dtype, device=device)
        )
        self.register_buffer(
            "running_var", Tensor(np.ones(num_features), dtype=param_dtype, device=device)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ShapeMismatchError(f"BatchNorm2d expects a 4D (N, C, H, W) input, got shape {x.shape}.")
        if x.shape[1] != self.num_features:
            raise ShapeMismatchError(
                f"BatchNorm2d(num_features={self.num_features}) cannot accept input with "
                f"{x.shape[1]} channels (input shape {x.shape})."
            )

        if x.device.type == "cuda":
            # See the module docstring: CUDA has no general multi-axis
            # reduction or (1,C,1,1)-vs-(N,C,H,W) broadcast to compose this
            # from, so this dispatches to one dedicated fused CUDA primitive
            # instead (forward+backward kernels, `docs/development/
            # m53-batchnorm.md`'s **CUDA implementation** section).
            return x.batch_norm2d(
                self.weight, self.bias, self.running_mean, self.running_var,
                training=self.training, momentum=self.momentum, eps=self.eps,
            )

        C = self.num_features
        N, _, H, W = x.shape
        count = N * H * W

        if self.training:
            mean = x.sum(axis=(0, 2, 3), keepdims=True) * (1.0 / count)
            diff = x - mean
            var = (diff * diff).sum(axis=(0, 2, 3), keepdims=True) * (1.0 / count)

            with no_grad():
                mean_flat = mean.reshape(C)
                var_flat = var.reshape(C)
                unbiased_scale = count / (count - 1) if count > 1 else 1.0
                new_running_mean = self.running_mean * (1.0 - self.momentum) + mean_flat * self.momentum
                new_running_var = (
                    self.running_var * (1.0 - self.momentum) + (var_flat * unbiased_scale) * self.momentum
                )
                self.running_mean._data = new_running_mean._data
                self.running_var._data = new_running_var._data

            std = (var + self.eps).sqrt()
            xhat = diff / std
        else:
            mean = self.running_mean.reshape(1, C, 1, 1)
            var = self.running_var.reshape(1, C, 1, 1)
            std = (var + self.eps).sqrt()
            xhat = (x - mean) / std

        if self.affine:
            return xhat * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)
        return xhat

    def __repr__(self) -> str:
        return (
            f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
        )


__all__ = ["BatchNorm2d"]
