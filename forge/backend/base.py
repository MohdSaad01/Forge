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
