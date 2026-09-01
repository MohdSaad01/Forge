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

    # -- optimizer (Milestone 10) ------------------------------------------

    @abstractmethod
    def sgd_step(self, data: Any, grad: Any, lr: float) -> Any:
        """In-place `data -= lr * grad`; returns `data` (same object where possible)."""
