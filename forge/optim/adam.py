"""Adam: adaptive-moment-estimation optimizer (Milestone 17)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..backend import get_backend
from ..backend.device import Device
from ..exceptions import OptimizerError
from ..nn.parameter import Parameter
from .optimizer import Optimizer


@dataclass
class _AdamState:
    """Per-`Parameter` Adam state: running moment estimates plus a step count.

    `m`/`v` are raw backend storage (an `np.ndarray` for a CPU `Parameter`, a
    `CUDAStorage` for a CUDA one) matching the parameter's shape/dtype
    exactly -- the same convention `Parameter._data` itself uses, never a
    NumPy array standing in for CUDA state. `device` is recorded explicitly
    (rather than re-derived from `m`'s type) so `Adam.step()` can cheaply
    detect a parameter that moved device after this state was created --
    see `Adam`'s docstring, "Device transfer after state exists".
    """

    m: Any
    v: Any
    device: Device
    step: int = 0


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015): per-parameter adaptive learning rates from
    running first/second gradient-moment estimates.

    ```text
    m_t     = beta1 * m_(t-1) + (1 - beta1) * g_t
    v_t     = beta2 * v_(t-1) + (1 - beta2) * g_t**2
    m_hat   = m_t / (1 - beta1**t)
    v_hat   = v_t / (1 - beta2**t)
    theta   = theta - lr * m_hat / (sqrt(v_hat) + eps)
    ```

    All numerical work happens in `Backend.adam_step()` (CPU: NumPy
    in-place arithmetic; CUDA: one kernel launch operating on `data`/`grad`/
    `m`/`v` entirely on-device) -- `Adam` itself only iterates parameters,
    owns hyperparameters, and manages per-parameter state, mirroring the
    `Tensor -> Backend` boundary every other Forge operation uses (see
    `docs/architecture/optimization.md`).

    **`weight_decay`** (default `0.0`, disabled) is classic L2
    regularization folded directly into the gradient before the moment
    updates: `g_t <- g_t + weight_decay * theta`. This is the original
    Adam paper's semantics, *not* AdamW's decoupled decay (which instead
    subtracts `lr * weight_decay * theta` from the parameter directly,
    outside the moment estimates). Forge implements only the former under
    the name `Adam`; AdamW is out of scope for this milestone.

    **State identity.** `self.state` maps `Parameter` -> `_AdamState`.
    `Parameter` defines no custom `__eq__`/`__hash__`, so this is ordinary
    Python object-identity dict keying -- state follows the same Parameter
    object regardless of which Module hierarchy currently references it or
    what attribute name it is assigned to, per the spec. State is allocated
    lazily on a parameter's first `step()` call with a non-`None` `.grad`
    (never eagerly at construction), matching `SGD`'s "no gradient, no
    update" convention and avoiding unnecessary CUDA allocation for
    parameters that never train.

    **Device transfer after state exists.** `Module.to(device)` moves a
    `Parameter`'s storage in place (same Python object, new device) but has
    no knowledge of any optimizer -- by design, `Optimizer` stays decoupled
    from `Module` internals (`docs/architecture/optimization.md`). Adam
    therefore does *not* migrate `m`/`v` automatically (Policy A): if
    `step()` finds existing state whose `device` no longer matches the
    parameter's current device, it raises `OptimizerError` rather than
    silently pairing (say) a CUDA parameter with CPU moment buffers or
    computing on the wrong device. The state dict is the explicit migration
    mechanism -- clear the stale entry (`del optimizer.state[param]`, or
    `optimizer.state.clear()` for every parameter) to let the next `step()`
    lazily reinitialize fresh, zeroed state on the parameter's new device;
    this deliberately restarts that parameter's moment estimates and step
    count rather than guessing how to transfer them.
    """

    def __init__(
        self,
        parameters: "Iterable[Parameter]",
        lr: float = 1e-3,
        betas: "tuple[float, float]" = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters)

        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or math.isnan(lr) or lr <= 0:
            raise OptimizerError(f"Adam requires a positive learning rate, got {lr!r}.")

        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or not all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in betas)
        ):
            raise OptimizerError(f"Adam requires betas=(beta1, beta2) as a 2-tuple of numbers, got {betas!r}.")
        beta1, beta2 = betas
        if math.isnan(beta1) or not (0 <= beta1 < 1):
            raise OptimizerError(f"Adam requires 0 <= beta1 < 1, got {beta1!r}.")
        if math.isnan(beta2) or not (0 <= beta2 < 1):
            raise OptimizerError(f"Adam requires 0 <= beta2 < 1, got {beta2!r}.")

        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or math.isnan(eps) or eps <= 0:
            raise OptimizerError(f"Adam requires a positive eps, got {eps!r}.")

        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, (int, float))
            or math.isnan(weight_decay)
            or weight_decay < 0
        ):
            raise OptimizerError(f"Adam requires weight_decay >= 0, got {weight_decay!r}.")

        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state: "dict[Parameter, _AdamState]" = {}

    def step(self) -> None:
        for param in self.parameters:
            if param.grad is None:
                continue

            grad = param.grad
            if grad.shape != param.shape:
                raise OptimizerError(
                    f"Adam: gradient shape {grad.shape} does not match parameter shape {param.shape}."
                )
            if grad.dtype != param.dtype:
                raise OptimizerError(
                    f"Adam: gradient dtype '{grad.dtype}' does not match parameter dtype '{param.dtype}'."
                )
            if grad.device != param.device:
                raise OptimizerError(
                    f"Adam: gradient device '{grad.device}' does not match parameter device '{param.device}'."
                )

            backend = get_backend(param._device)
            state = self.state.get(param)

            if state is not None and state.device != param._device:
                raise OptimizerError(
                    f"Adam: parameter moved from device '{state.device}' to '{param._device}' after "
                    "optimizer state was created for it; Adam does not migrate state automatically "
                    "(see Adam's docstring, 'Device transfer after state exists'). Clear the stale "
                    "entry -- `del optimizer.state[param]` or `optimizer.state.clear()` -- to "
                    "reinitialize fresh state on the parameter's current device."
                )

            if state is None:
                zeros_host = np.zeros(param._data.shape, dtype=param._data.dtype)
                m = backend.from_array(zeros_host, param._data.dtype)
                v = backend.from_array(zeros_host, param._data.dtype)
                state = _AdamState(m=m, v=v, device=param._device)
                self.state[param] = state
            elif state.m.shape != param._data.shape or state.m.dtype != param._data.dtype:
                raise OptimizerError(
                    f"Adam: existing optimizer state for a parameter has shape/dtype "
                    f"{state.m.shape}/{state.m.dtype}, incompatible with the parameter's current "
                    f"{param._data.shape}/{param._data.dtype}."
                )

            state.step += 1
            data, m, v = backend.adam_step(
                param._data, grad._data, state.m, state.v,
                self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, state.step,
            )
            param._data = data
            state.m = m
            state.v = v


__all__ = ["Adam"]
