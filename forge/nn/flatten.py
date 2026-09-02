"""`Flatten`: collapses a contiguous run of dimensions into one."""

from __future__ import annotations

from ..exceptions import ShapeMismatchError
from ..tensor.tensor import Tensor
from .module import Module


class Flatten(Module):
    """Collapses dims `[start_dim, end_dim]` (inclusive) of the input into a single dim.

    The primary use case -- and the only one Milestone 15's CNN layers
    (`Conv2d`/`MaxPool2d`) need -- is the default `Flatten(start_dim=1,
    end_dim=-1)`, collapsing an `(N, C, H, W)` feature map down to `(N, C*H*W)`
    ahead of a `Linear` layer, batch dimension untouched:

    ```text
    (N, C, H, W) -> (N, C*H*W)
    ```

    `start_dim`/`end_dim` follow the same convention as `Tensor.sum`'s
    `axis`/NumPy's own negative-indexing: negative values count from the
    end (`end_dim=-1` is the last dimension). Both are resolved against the
    input's actual `ndim` on every `forward()` call (not fixed at
    construction), since a `Flatten` instance may be reused across inputs
    of different rank in principle, though every module in Forge that feeds
    one -- `Conv2d`/`MaxPool2d` -- always produces a fixed-rank `(N, C, H,
    W)` output.

    Built entirely on `Tensor.reshape` (already differentiable and already
    real on both CPU and CUDA, `docs/architecture/cuda-backend.md`), so this
    module has no parameters, no backward rule of its own, and no per-device
    code: a `Flatten` participates in autograd purely through
    `Tensor.reshape`'s existing backward rule, exactly like `ReLU` through
    `Tensor.relu()`.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        ndim = x.ndim
        start = self.start_dim if self.start_dim >= 0 else self.start_dim + ndim
        end = self.end_dim if self.end_dim >= 0 else self.end_dim + ndim

        if not (0 <= start <= end < ndim):
            raise ShapeMismatchError(
                f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim}) is not valid "
                f"for a {ndim}D input of shape {x.shape}: resolved start_dim={start}, "
                f"end_dim={end} must satisfy 0 <= start_dim <= end_dim < {ndim}."
            )

        shape = x.shape
        flat_size = 1
        for dim in shape[start : end + 1]:
            flat_size *= dim
        new_shape = shape[:start] + (flat_size,) + shape[end + 1 :]
        return x.reshape(new_shape)

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


__all__ = ["Flatten"]
