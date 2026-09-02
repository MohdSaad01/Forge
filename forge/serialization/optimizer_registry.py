"""The explicit optimizer-type registry used by `forge.serialization.checkpoint`.

Mirrors `forge.serialization.registry` (the `Module`-type registry) exactly,
for exactly the same reason: `load_checkpoint()` must never instantiate an
optimizer class based solely on a string read from an untrusted archive. A
checkpoint's `"optimizer"."type"` field is only ever used as a lookup key
into this in-process registry -- never `eval`'d, never resolved via dynamic
import or attribute lookup. `SGD` and `Adam` are registered here as Forge's
two built-in supported optimizer types (`docs/architecture/persistence.md`).

An `OptimizerSpec` bundles four callables:

- `get_config(optimizer) -> dict`: JSON-safe hyperparameters (never
  per-parameter state).
- `from_config(parameters, config) -> optimizer`: reconstructs a fresh
  optimizer bound to the given (already-reconstructed) `Parameter`s.
- `get_param_state(optimizer, param) -> dict | None`: this optimizer's
  serializable state for one `Parameter` (`None` if the optimizer keeps no
  state for it, e.g. SGD always, or an Adam parameter never `step()`-ed).
  Values that are `np.ndarray` are archived as `.npy` array entries; every
  other value (`int`/`float`/`bool`) is stored as JSON metadata.
- `set_param_state(optimizer, param, arrays, scalars, device) -> None`:
  the inverse -- reconstructs this optimizer's state for `param` (already a
  fresh `Parameter` on `device`) from host `arrays`/`scalars`, and installs
  it into `optimizer.state`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from ..backend import get_backend
from ..backend.device import Device
from ..exceptions import PersistenceError
from ..nn.parameter import Parameter
from ..optim.optimizer import Optimizer


@dataclass(frozen=True)
class OptimizerSpec:
    """How to persist and reconstruct one registered `Optimizer` type."""

    type_name: str
    cls: type
    get_config: "Callable[[Optimizer], dict]"
    from_config: "Callable[[Iterable[Parameter], dict], Optimizer]"
    get_param_state: "Callable[[Optimizer, Parameter], dict | None]"
    set_param_state: "Callable[[Optimizer, Parameter, dict, dict, Device], None]"


_BY_NAME: "dict[str, OptimizerSpec]" = {}
_BY_CLASS: "dict[type, OptimizerSpec]" = {}


def register_optimizer(
    type_name: str,
    cls: type,
    get_config: "Callable[[Optimizer], dict]",
    from_config: "Callable[[Iterable[Parameter], dict], Optimizer]",
    get_param_state: "Callable[[Optimizer, Parameter], dict | None]",
    set_param_state: "Callable[[Optimizer, Parameter, dict, dict, Device], None]",
) -> OptimizerSpec:
    """Register `cls` as a checkpoint-able optimizer type under `type_name`.

    Same re-registration rules as `forge.serialization.register_module`: a
    `type_name`/`cls` pair may be (re-)registered consistently, but claiming
    an already-taken name (or re-registering a class under a different name)
    raises `PersistenceError`.
    """
    if not isinstance(type_name, str) or not type_name.strip():
        raise PersistenceError(
            f"register_optimizer() requires a non-empty string type_name, got {type_name!r}."
        )
    if not isinstance(cls, type):
        raise PersistenceError(f"register_optimizer() requires a class, got {cls!r}.")

    existing_by_name = _BY_NAME.get(type_name)
    if existing_by_name is not None and existing_by_name.cls is not cls:
        raise PersistenceError(
            f"Cannot register {cls!r} as optimizer type '{type_name}': that name is already "
            f"registered for a different class ({existing_by_name.cls!r})."
        )
    existing_by_class = _BY_CLASS.get(cls)
    if existing_by_class is not None and existing_by_class.type_name != type_name:
        raise PersistenceError(
            f"Cannot register {cls!r} under optimizer type '{type_name}': it is already "
            f"registered under a different type name ('{existing_by_class.type_name}')."
        )

    spec = OptimizerSpec(
        type_name=type_name,
        cls=cls,
        get_config=get_config,
        from_config=from_config,
        get_param_state=get_param_state,
        set_param_state=set_param_state,
    )
    _BY_NAME[type_name] = spec
    _BY_CLASS[cls] = spec
    return spec


def spec_for_class(cls: type) -> OptimizerSpec:
    """Look up the registered spec for an optimizer class, or raise `PersistenceError`."""
    spec = _BY_CLASS.get(cls)
    if spec is None:
        raise PersistenceError(
            f"Cannot save a checkpoint for optimizer type '{cls.__module__}.{cls.__qualname__}': "
            "it is not registered for checkpointing. Register it first with "
            "forge.serialization.optimizer_registry.register_optimizer(...)."
        )
    return spec


def spec_for_name(type_name: str) -> OptimizerSpec:
    """Look up the registered spec for a checkpoint's saved optimizer type name.

    This is the only place a checkpoint file's `"optimizer"."type"` string is
    used: as a key into this process's registry. An unknown name never
    triggers an import, an attribute lookup, or any other attempt to resolve
    it dynamically -- it raises `PersistenceError` instead.
    """
    spec = _BY_NAME.get(type_name)
    if spec is None:
        raise PersistenceError(
            f"Cannot load checkpoint: optimizer type '{type_name}' is not registered for "
            "checkpointing in this process. Register the corresponding class first with "
            "forge.serialization.optimizer_registry.register_optimizer(...)."
        )
    return spec


def _register_builtins() -> None:
    from ..optim.adam import Adam, _AdamState
    from ..optim.sgd import SGD

    # -- SGD: hyperparameters only, no per-parameter state ------------------

    def _sgd_get_config(opt: SGD) -> dict:
        return {"lr": opt.lr}

    def _sgd_from_config(parameters: "Iterable[Parameter]", config: dict) -> SGD:
        return SGD(parameters, lr=float(config["lr"]))

    def _sgd_get_param_state(opt: SGD, param: Parameter):
        return None

    def _sgd_set_param_state(opt: SGD, param: Parameter, arrays: dict, scalars: dict, device: Device) -> None:
        pass

    register_optimizer(
        "SGD", SGD,
        get_config=_sgd_get_config,
        from_config=_sgd_from_config,
        get_param_state=_sgd_get_param_state,
        set_param_state=_sgd_set_param_state,
    )

    # -- Adam: hyperparameters plus per-parameter (m, v, step) --------------

    def _adam_get_config(opt: Adam) -> dict:
        return {
            "lr": opt.lr,
            "betas": [opt.beta1, opt.beta2],
            "eps": opt.eps,
            "weight_decay": opt.weight_decay,
        }

    def _adam_from_config(parameters: "Iterable[Parameter]", config: dict) -> Adam:
        betas = config["betas"]
        return Adam(
            parameters,
            lr=float(config["lr"]),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(config["eps"]),
            weight_decay=float(config["weight_decay"]),
        )

    def _adam_get_param_state(opt: Adam, param: Parameter):
        state = opt.state.get(param)
        if state is None:
            return None
        backend = get_backend(state.device)
        return {
            "m": np.array(backend.to_numpy(state.m), copy=True),
            "v": np.array(backend.to_numpy(state.v), copy=True),
            "step": int(state.step),
        }

    def _adam_set_param_state(opt: Adam, param: Parameter, arrays: dict, scalars: dict, device: Device) -> None:
        backend = get_backend(device)
        m = backend.from_array(arrays["m"], param._data.dtype)
        v = backend.from_array(arrays["v"], param._data.dtype)
        opt.state[param] = _AdamState(m=m, v=v, device=device, step=int(scalars["step"]))

    register_optimizer(
        "Adam", Adam,
        get_config=_adam_get_config,
        from_config=_adam_from_config,
        get_param_state=_adam_get_param_state,
        set_param_state=_adam_set_param_state,
    )


_register_builtins()


__all__ = ["OptimizerSpec", "register_optimizer", "spec_for_class", "spec_for_name"]
