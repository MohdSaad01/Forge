"""`save_model()` / `load_model()`: the public model-persistence API.

```text
model metadata (format version, device)
    v
module type/configuration        (forge.serialization.registry)
    v
child modules (recursive)
    v
parameter state (name, shape, dtype, requires_grad, values)
```

A saved model is *state and configuration*, never a live computation graph
or executable code -- see `docs/architecture/persistence.md` for the full
format, versioning, and trust-model writeup. Optimizer state is never
persisted here; this is model (inference-time) persistence only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import PersistenceError
from ..nn.module import Module
from ..nn.parameter import Parameter
from .archive import read_archive, write_archive
from .registry import spec_for_class, spec_for_name

FORMAT_VERSION = 1
SUPPORTED_DEVICE = "cpu"


def save_model(model: Module, path: str) -> None:
    """Save `model`'s architecture, configuration, and parameter state to `path`.

    `model` must be built entirely from module types registered with
    `forge.serialization.register_module()` (Forge's built-in `Linear`/
    `ReLU` are pre-registered) -- an unregistered type anywhere in the tree
    raises `PersistenceError` before anything is written, so a save either
    fully succeeds or leaves no file behind. Only parameter *values* are
    saved, never `.grad` or any autograd graph state, and never optimizer
    state -- see `docs/architecture/persistence.md`.
    """
    if not isinstance(model, Module):
        raise PersistenceError(f"save_model() requires a forge.nn.Module, got {type(model).__name__}.")

    arrays: "dict[str, np.ndarray]" = {}
    root_node = _build_save_node(model, prefix="", arrays=arrays)

    metadata = {
        "forge_format_version": FORMAT_VERSION,
        "device": SUPPORTED_DEVICE,
        "root": root_node,
    }
    write_archive(path, metadata, arrays)


def load_model(path: str) -> Module:
    """Reconstruct and return the `Module` saved at `path`.

    Only ever constructs instances of classes registered with
    `forge.serialization.register_module()` in *this* process, using
    JSON-decoded configuration data as keyword arguments -- never `eval`,
    `exec`, dynamic import, or `pickle` on file content. An unknown module
    type, unsupported format version, unsupported device, or any shape/
    dtype/structural inconsistency between the file and what the registered
    constructors actually produce raises `PersistenceError` with a specific
    reason rather than a raw parsing exception.

    Returned parameters are fresh leaf `Tensor`/`Parameter` objects with no
    autograd graph attached -- a subsequent forward pass with gradients
    enabled builds an entirely new graph, exactly as for a freshly
    constructed model.
    """
    metadata, arrays = read_archive(path)

    if not isinstance(metadata, dict):
        raise PersistenceError(f"Cannot load model from '{path}': metadata is not a JSON object.")

    version = metadata.get("forge_format_version")
    if version != FORMAT_VERSION:
        raise PersistenceError(
            f"Cannot load model from '{path}': unsupported format version {version!r} "
            f"(this build of Forge supports version {FORMAT_VERSION})."
        )

    device = metadata.get("device")
    if device != SUPPORTED_DEVICE:
        raise PersistenceError(
            f"Cannot load model from '{path}': saved for device {device!r}, but this build "
            f"of Forge's persistence system only supports loading onto '{SUPPORTED_DEVICE}'."
        )

    root = metadata.get("root")
    if not isinstance(root, dict):
        raise PersistenceError(f"Cannot load model from '{path}': malformed metadata (missing 'root').")

    try:
        return _build_load_node(root, prefix="", arrays=arrays, path=path)
    except PersistenceError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise PersistenceError(f"Cannot load model from '{path}': malformed metadata ({exc}).") from exc


# -- save: recursive tree walk -------------------------------------------


def _build_save_node(module: Module, prefix: str, arrays: "dict[str, np.ndarray]") -> dict:
    spec = spec_for_class(type(module))
    config = spec.get_config(module)
    if not isinstance(config, dict):
        raise PersistenceError(
            f"Cannot save module type '{spec.type_name}': get_config() must return a dict, "
            f"got {type(config).__name__}."
        )

    parameters: "dict[str, dict]" = {}
    for name, param in module._parameters.items():
        dotted = name if not prefix else f"{prefix}.{name}"
        arrays[dotted] = np.array(param.numpy(), copy=True)
        parameters[name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "requires_grad": param.requires_grad,
        }

    children: "dict[str, dict]" = {}
    for name, child in module._modules.items():
        child_prefix = name if not prefix else f"{prefix}.{name}"
        children[name] = _build_save_node(child, child_prefix, arrays)

    return {
        "type": spec.type_name,
        "config": config,
        "training": module.training,
        "parameters": parameters,
        "children": children,
    }


# -- load: recursive tree walk -------------------------------------------


def _build_load_node(node: dict, prefix: str, arrays: "dict[str, np.ndarray]", path: str) -> Module:
    label = prefix or "<root>"
    for key in ("type", "config", "training", "parameters", "children"):
        if key not in node:
            raise PersistenceError(
                f"Cannot load model from '{path}': malformed metadata, module '{label}' is missing '{key}'."
            )

    type_name = node["type"]
    config = node["config"]
    if not isinstance(config, dict):
        raise PersistenceError(
            f"Cannot load model from '{path}': malformed metadata, 'config' for module "
            f"'{label}' is not an object."
        )

    spec = spec_for_name(type_name)
    try:
        module = spec.from_config(dict(config))
    except PersistenceError:
        raise
    except Exception as exc:
        raise PersistenceError(
            f"Cannot load model from '{path}': invalid configuration for module '{label}' "
            f"of type '{type_name}': {exc}."
        ) from exc
    if not isinstance(module, Module):
        raise PersistenceError(
            f"Cannot load model from '{path}': the registered constructor for type "
            f"'{type_name}' did not produce a forge.nn.Module."
        )

    param_meta = node["parameters"]
    if not isinstance(param_meta, dict):
        raise PersistenceError(
            f"Cannot load model from '{path}': malformed metadata, 'parameters' for module "
            f"'{label}' is not an object."
        )
    expected_param_names = set(param_meta.keys())
    actual_param_names = set(module._parameters.keys())
    if expected_param_names != actual_param_names:
        raise PersistenceError(
            f"Cannot load model from '{path}': inconsistent model state for module '{label}' "
            f"(type '{type_name}'): file declares parameters {sorted(expected_param_names)}, "
            f"but the reconstructed module has {sorted(actual_param_names)}."
        )

    for name, meta in param_meta.items():
        dotted = name if not prefix else f"{prefix}.{name}"
        array = arrays.get(dotted)
        if array is None:
            raise PersistenceError(
                f"Cannot load model from '{path}': missing parameter data for '{dotted}'."
            )
        expected_shape = tuple(meta.get("shape", []))
        if tuple(array.shape) != expected_shape:
            raise PersistenceError(
                f"Cannot load model from '{path}': parameter '{dotted}' has shape "
                f"{tuple(array.shape)} in the file but metadata declares {expected_shape}."
            )
        expected_dtype = meta.get("dtype")
        if str(array.dtype) != expected_dtype:
            raise PersistenceError(
                f"Cannot load model from '{path}': parameter '{dotted}' has dtype "
                f"'{array.dtype}' in the file but metadata declares '{expected_dtype}'."
            )
        requires_grad = bool(meta.get("requires_grad", True))
        setattr(
            module,
            name,
            Parameter(array, dtype=expected_dtype, device=SUPPORTED_DEVICE, requires_grad=requires_grad),
        )

    child_meta = node["children"]
    if not isinstance(child_meta, dict):
        raise PersistenceError(
            f"Cannot load model from '{path}': malformed metadata, 'children' for module "
            f"'{label}' is not an object."
        )
    expected_child_names = set(child_meta.keys())
    actual_child_names = set(module._modules.keys())
    if expected_child_names != actual_child_names:
        raise PersistenceError(
            f"Cannot load model from '{path}': inconsistent model state for module '{label}' "
            f"(type '{type_name}'): file declares children {sorted(expected_child_names)}, "
            f"but the reconstructed module has {sorted(actual_child_names)}."
        )

    for name, child_node in child_meta.items():
        child_prefix = name if not prefix else f"{prefix}.{name}"
        setattr(module, name, _build_load_node(child_node, child_prefix, arrays, path))

    object.__setattr__(module, "_training", bool(node["training"]))
    return module


__all__ = ["save_model", "load_model", "FORMAT_VERSION", "SUPPORTED_DEVICE"]
