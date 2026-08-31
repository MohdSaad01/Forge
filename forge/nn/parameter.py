"""Trainable tensor state owned by a Module."""

from __future__ import annotations

from typing import Any

from ..tensor.tensor import Tensor


class Parameter(Tensor):
    """A leaf `Tensor` that requires grad by default, marking trainable state.

    `Parameter` adds no numerical behavior of its own -- it is a thin `Tensor`
    subclass so it plugs directly into the existing autograd system.
    Operations applied to a `Parameter` run through Tensor's ordinary
    differentiable ops and accumulate gradients into `.grad` exactly like any
    other leaf tensor. `Module.__setattr__` recognizes this type to register
    it automatically (see `forge.nn.Module`).
    """

    def __init__(
        self,
        data: Any,
        dtype: Any = None,
        device: "str" = "cpu",
        requires_grad: bool = True,
    ):
        super().__init__(data, dtype=dtype, device=device, requires_grad=requires_grad)

    def __repr__(self) -> str:
        extra = "" if self._requires_grad else ", requires_grad=False"
        return f"Parameter({self._data!r}, dtype={self._dtype}, device={self._device}{extra})"


__all__ = ["Parameter"]
