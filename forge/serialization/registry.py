"""The explicit module-type registry used to save/load `Module` trees.

Reconstruction never inspects, imports, or evaluates anything named inside a
model file. A file's `"type"` field is only ever used as a lookup key into
this in-process registry -- if the key is not present, loading fails with a
clear `PersistenceError` rather than attempting to locate or construct
anything on its own. This is the mechanism that keeps `load_model()` free of
arbitrary code execution: every class `from_config` can construct is a class
that this process has already imported and explicitly opted in via
`register_module()`, never one named by the file itself.

`Linear` and `ReLU` are registered here as Forge's built-in supported types.
A composite/custom `Module` subclass (anything with its own `__init__`
assembling child modules, e.g. a hand-written multi-layer model) is not
automatically persistable -- it must be registered the same way, by calling
`register_module()` before saving or loading it. See
`docs/architecture/persistence.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import PersistenceError


@dataclass(frozen=True)
class ModuleSpec:
    """How to persist and reconstruct one registered `Module` type."""

    type_name: str
    cls: type
    get_config: "Callable[[Any], dict]"
    from_config: "Callable[[dict], Any]"


_BY_NAME: "dict[str, ModuleSpec]" = {}
_BY_CLASS: "dict[type, ModuleSpec]" = {}


def register_module(
    type_name: str,
    cls: type,
    get_config: "Callable[[Any], dict]",
    from_config: "Callable[[dict], Any] | None" = None,
) -> ModuleSpec:
    """Register `cls` as a persistable module type under the stable name `type_name`.

    `get_config(module_instance) -> dict` must extract a JSON-safe
    (str/int/float/bool/None/list/dict) constructor-configuration mapping --
    architecture, not parameter values (those are saved/restored
    separately). `from_config(config) -> module_instance` reconstructs a
    fresh, unloaded instance from that mapping; if omitted it defaults to
    `cls(**config)`, which is sufficient for any class whose `__init__`
    accepts exactly the keys `get_config` returns.

    A `type_name`/`cls` pair may only be registered once, consistently --
    re-registering the same class under the same name is a no-op-safe
    re-registration, but registering a name or class against a *different*
    counterpart raises `PersistenceError`, since that would make save/load
    ambiguous about which class a file's `"type"` refers to.
    """
    if not isinstance(type_name, str) or not type_name.strip():
        raise PersistenceError(
            f"register_module() requires a non-empty string type_name, got {type_name!r}."
        )
    if not isinstance(cls, type):
        raise PersistenceError(f"register_module() requires a class, got {cls!r}.")

    existing_by_name = _BY_NAME.get(type_name)
    if existing_by_name is not None and existing_by_name.cls is not cls:
        raise PersistenceError(
            f"Cannot register {cls!r} as '{type_name}': that name is already registered "
            f"for a different class ({existing_by_name.cls!r})."
        )
    existing_by_class = _BY_CLASS.get(cls)
    if existing_by_class is not None and existing_by_class.type_name != type_name:
        raise PersistenceError(
            f"Cannot register {cls!r} under '{type_name}': it is already registered "
            f"under a different type name ('{existing_by_class.type_name}')."
        )

    resolved_from_config = from_config if from_config is not None else (lambda config, _cls=cls: _cls(**config))
    spec = ModuleSpec(type_name=type_name, cls=cls, get_config=get_config, from_config=resolved_from_config)
    _BY_NAME[type_name] = spec
    _BY_CLASS[cls] = spec
    return spec


def spec_for_class(cls: type) -> ModuleSpec:
    """Look up the registered spec for a module class, or raise `PersistenceError`.

    Lookup is by exact class -- a subclass of a registered type is not
    automatically considered persistable, since its `forward()`/`__init__`
    behavior may differ from the registered class's.
    """
    spec = _BY_CLASS.get(cls)
    if spec is None:
        raise PersistenceError(
            f"Cannot save a module of type '{cls.__module__}.{cls.__qualname__}': it is not "
            "registered for persistence. Register it first with "
            "forge.serialization.register_module(type_name, cls, get_config)."
        )
    return spec


def spec_for_name(type_name: str) -> ModuleSpec:
    """Look up the registered spec for a saved type name, or raise `PersistenceError`.

    This is the only place a model file's `"type"` string is used: as a key
    into this process's registry. An unknown name never triggers an import,
    an attribute lookup, or any other attempt to resolve it dynamically.
    """
    spec = _BY_NAME.get(type_name)
    if spec is None:
        raise PersistenceError(
            f"Cannot load module type '{type_name}': it is not registered for persistence "
            "in this process. Register the corresponding class first with "
            "forge.serialization.register_module(type_name, cls, get_config)."
        )
    return spec


def _register_builtins() -> None:
    from ..nn.activation import ReLU
    from ..nn.container import Sequential
    from ..nn.conv import Conv2d
    from ..nn.dropout import Dropout
    from ..nn.flatten import Flatten
    from ..nn.linear import Linear
    from ..nn.module import Module
    from ..nn.pooling import MaxPool2d

    register_module(
        "Linear",
        Linear,
        get_config=lambda m: {
            "in_features": m.in_features,
            "out_features": m.out_features,
            "bias": m.bias is not None,
        },
    )
    register_module("ReLU", ReLU, get_config=lambda m: {})
    register_module(
        "Conv2d",
        Conv2d,
        get_config=lambda m: {
            "in_channels": m.in_channels,
            "out_channels": m.out_channels,
            "kernel_size": list(m.kernel_size),
            "stride": list(m.stride),
            "padding": list(m.padding),
            "bias": m.bias is not None,
        },
    )
    register_module(
        "MaxPool2d",
        MaxPool2d,
        get_config=lambda m: {
            "kernel_size": list(m.kernel_size),
            "stride": list(m.stride),
            "padding": list(m.padding),
        },
    )
    register_module(
        "Sequential",
        Sequential,
        # `_build_load_node` (`forge/serialization/model.py`) requires a
        # freshly `from_config`-constructed module to already have a child
        # under every name the file is about to attach (it validates
        # `expected_child_names == actual_child_names` *before* the
        # attach loop) -- a fixed-shape invariant that holds for free for
        # every other registered type (their child/parameter set is
        # entirely determined by config, e.g. `Linear`'s `in_features`/
        # `out_features`). `Sequential`'s child *count* is itself data, not
        # config, so `get_config`/`from_config` report/consume exactly that
        # one extra fact -- `n_children` -- to build the right number of
        # placeholder `Module()` children up front; the attach loop then
        # overwrites every placeholder with its real reconstructed child
        # (Conv2d, ReLU, ...) via the ordinary `setattr()` mechanism, the
        # same as any other module's children. This is the smallest
        # extension the existing `from_config` mechanism already
        # anticipates (see `register_module`'s docstring) -- no change to
        # the generic save/load tree walk itself, and still no
        # Sequential-specific persistence *format*.
        get_config=lambda m: {"n_children": len(m._modules)},
        from_config=lambda config: Sequential(*(Module() for _ in range(int(config.get("n_children", 0))))),
    )
    register_module(
        "Flatten",
        Flatten,
        get_config=lambda m: {"start_dim": m.start_dim, "end_dim": m.end_dim},
    )
    register_module(
        "Dropout",
        Dropout,
        # Only `p` is persisted -- never RNG state or the current mask (see
        # `docs/architecture/modules.md`'s **Dropout** section). `.training`
        # round-trips generically already: `_build_save_node`/
        # `_build_load_node` (`forge/serialization/model.py`) save/restore
        # every module's `.training` flag regardless of type.
        get_config=lambda m: {"p": m.p},
    )


_register_builtins()


__all__ = ["ModuleSpec", "register_module", "spec_for_class", "spec_for_name"]
