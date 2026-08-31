"""CPU backend: the reference implementation (see ADR-002).

Thin wrappers over NumPy. No shape/dtype validation happens here -- that is
the Tensor layer's responsibility, so error messages stay in terms of Forge
tensors rather than raw NumPy internals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Backend


class CPUBackend(Backend):
    device_type = "cpu"

    def from_array(self, data: Any, dtype: "np.dtype | None") -> np.ndarray:
        return np.array(data, dtype=dtype)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.add(a, b)

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.subtract(a, b)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.multiply(a, b)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def sum(self, a: np.ndarray, axis, keepdims: bool) -> np.ndarray:
        return np.asarray(np.sum(a, axis=axis, keepdims=keepdims))

    def reshape(self, a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(a, shape)
