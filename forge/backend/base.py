"""The numerical execution boundary a backend must implement.

The Tensor layer validates shapes/dtypes/devices and raises Forge's own
errors; a ``Backend`` just performs the raw numerical work for one device
type. This keeps high-level Tensor code free of backend-specific details,
per the documented boundary in ``docs/architecture/backend-architecture.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """Numerical implementation behind Tensor operations for one device type."""

    device_type: str

    @abstractmethod
    def from_array(self, data: Any, dtype: "np.dtype | None") -> np.ndarray:
        """Build backend storage from array-like data."""

    @abstractmethod
    def to_numpy(self, storage: Any) -> np.ndarray:
        """Materialize backend storage as a host NumPy array (used by `Tensor.to()`)."""

    @abstractmethod
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def sum(self, a: np.ndarray, axis, keepdims: bool) -> np.ndarray: ...

    @abstractmethod
    def reshape(self, a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray: ...

    @abstractmethod
    def relu(self, a: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def exp(self, a: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def log(self, a: np.ndarray) -> np.ndarray: ...

    # -- backward (Milestone 10) ------------------------------------------
    #
    # Operation-specific gradient math, one implementation per backend, so
    # `Tensor`'s backward closures (`forge/tensor/tensor.py`) never contain
    # backend-specific (NumPy vs. CUDAStorage) code themselves -- they just
    # call `get_backend(device).<op>_backward(...)`. `a`/`b`/`grad_output`
    # are raw backend storage (a `np.ndarray` for `CPUBackend`, a
    # `CUDAStorage` for `CUDABackend`), matching every other `Backend` method.

    @abstractmethod
    def add_backward(self, grad_output: Any, a: Any, b: Any) -> "tuple[Any, Any]": ...

    @abstractmethod
    def sub_backward(self, grad_output: Any, a: Any, b: Any) -> "tuple[Any, Any]": ...

    @abstractmethod
    def mul_backward(self, grad_output: Any, a: Any, b: Any) -> "tuple[Any, Any]": ...

    @abstractmethod
    def matmul_backward(self, grad_output: Any, a: Any, b: Any) -> "tuple[Any, Any]": ...

    @abstractmethod
    def sum_backward(
        self, grad_output: Any, original_shape: "tuple[int, ...]", ndim: int, axis, keepdims: bool
    ) -> Any: ...

    @abstractmethod
    def reshape_backward(self, grad_output: Any, original_shape: "tuple[int, ...]") -> Any: ...

    @abstractmethod
    def relu_backward(self, grad_output: Any, a: Any) -> Any: ...

    @abstractmethod
    def exp_backward(self, grad_output: Any, result: Any) -> Any:
        """`d(exp(x))/dx = exp(x)`, i.e. `grad_output * result` (`result` is exp's own saved output)."""

    @abstractmethod
    def log_backward(self, grad_output: Any, a: Any) -> Any:
        """`d(log(x))/dx = 1/x`, i.e. `grad_output / a` (`a` is log's saved input)."""

    # -- optimizer (Milestone 10) ------------------------------------------

    @abstractmethod
    def sgd_step(self, data: Any, grad: Any, lr: float) -> Any:
        """In-place `data -= lr * grad`; returns `data` (same object where possible)."""

    # -- Adam optimizer (Milestone 17) --------------------------------------
    #
    # `m`/`v` are the running first/second moment estimates -- raw backend
    # storage matching `data`'s shape/dtype/device exactly, the same
    # convention `data`/`grad` themselves use (an `np.ndarray` for
    # `CPUBackend`, a `CUDAStorage` for `CUDABackend`; never a NumPy array
    # standing in for CUDA state). `step` is the 1-indexed step count for
    # *this* parameter, used for bias correction (`1 - beta{1,2}**step`).
    # All numerical Adam work -- moment updates, bias correction, the
    # parameter update itself -- happens inside this one call; the optimizer
    # (`forge/optim/adam.py`) only handles parameter iteration, state
    # bookkeeping, and hyperparameters.

    @abstractmethod
    def adam_step(
        self, data: Any, grad: Any, m: Any, v: Any,
        lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, step: int,
    ) -> "tuple[Any, Any, Any]":
        """In-place Adam update; returns `(data, m, v)` (same objects where possible)."""

    # -- CrossEntropyLoss support (Milestone 14) ----------------------------

    @abstractmethod
    def max_axis1(self, a: Any) -> Any:
        """Row-wise max reduction for a 2D array: shape (rows, cols) -> (rows, 1).

        Used internally by `CrossEntropyLoss`'s log-sum-exp numerical-
        stability shift (`docs/architecture/optimization.md`). Not exposed as
        a public differentiable Tensor operation -- the result is always
        treated as a constant (the log-sum-exp identity `logsumexp(x - c) ==
        logsumexp(x) - c` holds for any per-row constant `c`, so no gradient
        needs to flow through the max itself), so there is no
        `max_axis1_backward` counterpart.
        """

    # -- Conv2d / MaxPool2d (Milestone 15) ----------------------------------
    #
    # `x`/`weight`/`bias` are raw backend storage (a `np.ndarray` for
    # `CPUBackend`, a `CUDAStorage` for `CUDABackend`) in NCHW / (C_out,
    # C_in, KH, KW) / (C_out,) layout, matching every other `Backend` method.
    # `stride`/`padding` are always `(height, width)` int pairs by the time
    # they reach the backend -- `Tensor.conv2d`/`Tensor.max_pool2d` normalize
    # a bare int before calling here. `bias` may be `None` (no bias term).

    @abstractmethod
    def conv2d(self, x: Any, weight: Any, bias: "Any | None", stride: "tuple[int, int]", padding: "tuple[int, int]") -> Any:
        """2D cross-correlation forward: NCHW `x`, (C_out, C_in, KH, KW) `weight`."""

    @abstractmethod
    def conv2d_backward(
        self, grad_output: Any, x: Any, weight: Any, bias: "Any | None",
        stride: "tuple[int, int]", padding: "tuple[int, int]",
    ) -> "tuple[Any, Any, Any | None]":
        """Returns `(grad_x, grad_weight, grad_bias)`; `grad_bias` is `None` iff `bias` is."""

    @abstractmethod
    def max_pool2d(
        self, x: Any, kernel_size: "tuple[int, int]", stride: "tuple[int, int]", padding: "tuple[int, int]"
    ) -> Any:
        """2D max pooling forward over NCHW `x`."""

    @abstractmethod
    def max_pool2d_backward(
        self, grad_output: Any, x: Any, kernel_size: "tuple[int, int]",
        stride: "tuple[int, int]", padding: "tuple[int, int]",
    ) -> Any:
        """Gradient w.r.t. `x`, recomputing each window's argmax from the saved input `x`."""

    # -- Dropout (Milestone 16) ----------------------------------------------
    #
    # `a` is only ever read for its shape/dtype -- its values never affect
    # the mask. `rng` is a `numpy.random.Generator` (`forge.random`'s
    # process-global generator by default, or an explicit one); a CPU
    # implementation draws directly from it (`docs/architecture/modules.md`),
    # while a CUDA implementation draws exactly one integer seed from it
    # (a cheap host-side scalar draw, not per-element randomness) and
    # generates every element's Bernoulli draw on-device from that seed --
    # see `docs/architecture/cuda-backend.md`'s **CUDA Dropout** section.

    @abstractmethod
    def dropout_mask(self, a: Any, p: float, rng: Any) -> Any:
        """A mask matching `a`'s shape/dtype: `1/(1-p)` with probability `1-p`, else `0`.

        Scaling is baked into the mask itself (inverted dropout), so
        `Tensor.dropout_mask()`'s caller implements both forward
        (`x * mask`) and backward (`grad_output * mask`, via ordinary `mul`
        autograd) with one multiply and no Dropout-specific backward rule.
        """

    # -- CrossEntropyLoss (Milestone 31) --------------------------------------
    #
    # Milestone 21/31 profiling found `nn.CrossEntropyLoss`'s forward pass
    # (previously composed from ~9 ordinary Tensor primitives: max_axis1, two
    # row-broadcast subs, exp, sum(axis=1), log, a mul against a freshly
    # host-transferred one-hot matrix, a full sum, and a final scale) costing
    # *more* wall-clock time on CUDA than the entire M20 CNN's forward pass,
    # despite operating on a tensor orders of magnitude smaller -- each of
    # those ~9 forward + ~7 backward primitives pays a real per-launch
    # dispatch cost regardless of how little arithmetic it does (see
    # `docs/performance/pipeline-profiling.md`). `x` (logits, floating-point)
    # and `target` (class indices, always int64 on CUDA -- see
    # `CUDABackend.cross_entropy`) are raw backend storage, matching every
    # other `Backend` method; `Tensor.cross_entropy()` (`tensor.py`) is the
    # one call site, reached from `nn.CrossEntropyLoss.forward()` after all
    # of that class's existing shape/dtype/range validation.

    @abstractmethod
    def cross_entropy(self, x: Any, target: Any) -> Any:
        """Fused forward: `mean(-log_softmax(x)[i, target[i]])` over `x`'s shape (batch, classes)."""

    @abstractmethod
    def cross_entropy_backward(self, grad_output: Any, x: Any, target: Any) -> Any:
        """Gradient w.r.t. `x`: `grad_output/batch_size * (softmax(x) - one_hot(target))`.

        Recomputes softmax from the saved forward input `x` (and `target`)
        rather than caching intermediate forward state -- the same
        "recompute from a saved input" convention `relu`/`exp`/`log`
        backward already use elsewhere in this interface.
        """
