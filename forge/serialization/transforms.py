"""Preprocessing-transform configuration serialization (Milestone 71).

`forge.data.transforms` are ordinary Python objects, not `Module`s -- they
never went through `forge.serialization.registry`'s save/load path, so a
saved model previously carried no record of the preprocessing its inputs
were expected to have already gone through (see `docs/development/
m70-image-preprocessing.md`'s "Future Pressure Identified" section, and
`docs/architecture/persistence.md`). This module closes that gap the same
way `forge.serialization.registry` closes it for `Module` types: an
explicit, in-process registry mapping a stable `type_name` string to
`get_config`/`from_config` functions, so a transform's *configuration* --
never a pickled object or arbitrary code -- is what travels through a file.

```python
forge.serialization.transforms.serialize_transform(Resize((64, 64)))
# {"type": "Resize", "config": {"size": [64, 64]}}
```

**Only configuration-representable transforms are registered**: `Resize`,
`Compose`, `Normalize`, `ToTensor`, `Reshape`, `Flatten` -- every transform
in `forge.data.transforms` whose entire behavior is already just a handful
of JSON-safe constructor arguments. `Lambda` wraps an arbitrary Python
callable (frequently a closure) with no safe general representation short
of pickling executable code -- exactly the "serialize arbitrary Python
callables" outcome Milestone 71's brief explicitly rejected -- so it is
**deliberately never registered**. Attempting to serialize a `Lambda` (or
any other unregistered transform) raises `PersistenceError` immediately,
the same clear "not registered" failure `forge.serialization.registry`
already produces for an unregistered `Module` subclass -- never a silent
best-effort encoding. A pipeline that needs a `Lambda` step for training
can still use one; it just cannot be part of what `save_model(...,
preprocessing=...)` persists (see that function's docstring for the
`Normalize`-based alternative `examples/image_folder_classification/`
uses for pixel scaling).

`Compose` is the one transform whose configuration is itself recursive
(a list of child transform configurations) -- its `get_config`/
`from_config` call `serialize_transform`/`deserialize_transform`
recursively, exactly mirroring how `forge.serialization.registry`'s
`Sequential` entry is the one `Module` type whose reconstruction needs
help beyond a flat `cls(**config)` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..data.transforms import Compose, Flatten, Normalize, Reshape, Resize, ToTensor, Transform
from ..exceptions import PersistenceError


@dataclass(frozen=True)
class TransformSpec:
    """How to persist and reconstruct one registered `Transform` type."""

    type_name: str
    cls: type
    get_config: "Callable[[Any], dict]"
    from_config: "Callable[[dict], Any]"


_BY_NAME: "dict[str, TransformSpec]" = {}
_BY_CLASS: "dict[type, TransformSpec]" = {}


def register_transform(
    type_name: str,
    cls: type,
    get_config: "Callable[[Any], dict]",
    from_config: "Callable[[dict], Any] | None" = None,
) -> TransformSpec:
    """Register `cls` as a persistable preprocessing-transform type under `type_name`.

    Mirrors `forge.serialization.register_module()` exactly: `get_config(instance)
    -> dict` must extract a JSON-safe configuration mapping; `from_config(config)
    -> instance` reconstructs a fresh instance from it, defaulting to
    `cls(**config)` when omitted. Re-registering the same `(type_name, cls)`
    pair is a no-op-safe re-registration; registering either half against a
    different counterpart raises `PersistenceError`.
    """
    if not isinstance(type_name, str) or not type_name.strip():
        raise PersistenceError(
            f"register_transform() requires a non-empty string type_name, got {type_name!r}."
        )
    if not isinstance(cls, type):
        raise PersistenceError(f"register_transform() requires a class, got {cls!r}.")

    existing_by_name = _BY_NAME.get(type_name)
    if existing_by_name is not None and existing_by_name.cls is not cls:
        raise PersistenceError(
            f"Cannot register {cls!r} as '{type_name}': that name is already registered "
            f"for a different transform class ({existing_by_name.cls!r})."
        )
    existing_by_class = _BY_CLASS.get(cls)
    if existing_by_class is not None and existing_by_class.type_name != type_name:
        raise PersistenceError(
            f"Cannot register {cls!r} under '{type_name}': it is already registered "
            f"under a different type name ('{existing_by_class.type_name}')."
        )

    resolved_from_config = from_config if from_config is not None else (lambda config, _cls=cls: _cls(**config))
    spec = TransformSpec(type_name=type_name, cls=cls, get_config=get_config, from_config=resolved_from_config)
    _BY_NAME[type_name] = spec
    _BY_CLASS[cls] = spec
    return spec


def spec_for_class(cls: type) -> TransformSpec:
    """Look up the registered spec for a transform class, or raise `PersistenceError`.

    Lookup is by exact class, matching `forge.serialization.registry.spec_for_class`
    -- a subclass of a registered transform is not automatically persistable.
    """
    spec = _BY_CLASS.get(cls)
    if spec is None:
        raise PersistenceError(
            f"Cannot serialize a preprocessing transform of type "
            f"'{cls.__module__}.{cls.__qualname__}': it is not registered for persistence. "
            "Register it first with forge.serialization.transforms.register_transform("
            "type_name, cls, get_config), or -- if it wraps an arbitrary Python callable "
            "such as Lambda -- express the same preprocessing with a supported transform "
            "instead (Resize, Compose, Normalize, ToTensor, Reshape, Flatten): arbitrary "
            "callables have no safe serialized representation and are deliberately never "
            "registered."
        )
    return spec


def spec_for_name(type_name: str) -> TransformSpec:
    """Look up the registered spec for a saved type name, or raise `PersistenceError`.

    The only place a file's transform `"type"` string is used: as a key into
    this process's registry, never `eval`'d or dynamically imported.
    """
    spec = _BY_NAME.get(type_name)
    if spec is None:
        raise PersistenceError(
            f"Cannot reconstruct preprocessing transform type '{type_name}': it is not "
            "registered for persistence in this process. Register the corresponding class "
            "first with forge.serialization.transforms.register_transform(type_name, cls, get_config)."
        )
    return spec


def serialize_transform(transform: Any) -> dict:
    """Encode `transform` as a JSON-safe `{"type": ..., "config": ...}` node.

    `transform` must be an instance of a registered type (see
    `register_transform`) -- raises `PersistenceError` otherwise, naming the
    offending class.
    """
    spec = spec_for_class(type(transform))
    config = spec.get_config(transform)
    if not isinstance(config, dict):
        raise PersistenceError(
            f"Cannot serialize preprocessing transform type '{spec.type_name}': get_config() "
            f"must return a dict, got {type(config).__name__}."
        )
    return {"type": spec.type_name, "config": config}


def deserialize_transform(node: Any) -> Any:
    """Reconstruct a transform instance from a `serialize_transform()` node.

    Raises `PersistenceError` for a malformed node (not an object, missing
    `"type"`/`"config"`, non-object `"config"`) or an unregistered/unknown
    `"type"` -- the exact same failure modes `forge.serialization.model`
    already uses for a malformed/unrecognized module node.
    """
    if not isinstance(node, dict) or "type" not in node or "config" not in node:
        raise PersistenceError(
            f"Cannot reconstruct preprocessing transform: malformed metadata node {node!r} "
            "(expected an object with 'type' and 'config')."
        )
    type_name = node["type"]
    config = node["config"]
    if not isinstance(config, dict):
        raise PersistenceError(
            f"Cannot reconstruct preprocessing transform '{type_name}': 'config' is not an object."
        )
    spec = spec_for_name(type_name)
    try:
        instance = spec.from_config(dict(config))
    except PersistenceError:
        raise
    except Exception as exc:
        raise PersistenceError(
            f"Cannot reconstruct preprocessing transform '{type_name}': invalid configuration "
            f"{config!r} ({exc})."
        ) from exc
    if not isinstance(instance, Transform):
        raise PersistenceError(
            f"Cannot reconstruct preprocessing transform '{type_name}': the registered "
            "constructor did not produce a forge.data.transforms.Transform."
        )
    return instance


def _register_builtins() -> None:
    register_transform(
        "Resize",
        Resize,
        get_config=lambda m: {"size": list(m.size)},
    )
    register_transform(
        "Normalize",
        Normalize,
        # `mean`/`std` may be a plain scalar or a per-channel sequence --
        # `np.asarray(...).tolist()` gives a JSON-safe form for either shape
        # (a bare float, or a nested list) without inventing a second
        # config format for the scalar case.
        get_config=lambda m: {
            "mean": np.asarray(m.mean, dtype=np.float64).tolist(),
            "std": np.asarray(m.std, dtype=np.float64).tolist(),
        },
    )
    register_transform(
        "ToTensor",
        ToTensor,
        get_config=lambda m: {
            "dtype": str(m.dtype) if m.dtype is not None else None,
            "device": m.device,
        },
    )
    register_transform(
        "Reshape",
        Reshape,
        # `Reshape.__init__(*shape)` takes its target shape as positional
        # varargs, not a `shape=` keyword -- `from_config` must unpack the
        # saved list rather than relying on the generic `cls(**config)`
        # default (which would pass `shape=[...]` and fail).
        get_config=lambda m: {"shape": list(m.shape)},
        from_config=lambda config: Reshape(*config["shape"]),
    )
    register_transform(
        "Flatten",
        Flatten,
        get_config=lambda m: {},
    )
    register_transform(
        "Compose",
        Compose,
        # Recursive, mirroring `forge.serialization.registry`'s `Sequential`
        # entry: the one transform whose configuration is itself a list of
        # child transform configurations rather than flat JSON-safe scalars.
        get_config=lambda m: {"transforms": [serialize_transform(t) for t in m.transforms]},
        from_config=lambda config: Compose([deserialize_transform(t) for t in config["transforms"]]),
    )


_register_builtins()


__all__ = [
    "TransformSpec",
    "register_transform",
    "spec_for_class",
    "spec_for_name",
    "serialize_transform",
    "deserialize_transform",
]
