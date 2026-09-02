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

As of Milestone 13, a model may be saved from (and loaded onto) either
`"cpu"` or `"cuda"`. A CUDA `Parameter`'s values are always copied to host
memory before being written -- a persistence *transfer*, never a
computation -- and the archive itself remains the same portable,
CPU-readable ZIP(json + .npy) format `"cpu"`-only files have always used;
only the recorded `"device"` value and the set of devices `load_model()`
will restore onto changed. See **Device semantics** in
`docs/architecture/persistence.md` for the full loading policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend import get_backend
from ..backend.device import SUPPORTED_DEVICE_TYPES
from ..exceptions import PersistenceError
from ..nn.module import Module
from ..nn.parameter import Parameter
from .archive import read_archive, write_archive
from .registry import spec_for_class, spec_for_name

FORMAT_VERSION = 1


def save_model(model: Module, path: str) -> None:
    """Save `model`'s architecture, configuration, and parameter state to `path`.

    `model` must be built entirely from module types registered with
    `forge.serialization.register_module()` (Forge's built-in `Linear`/
    `ReLU` are pre-registered) -- an unregistered type anywhere in the tree
    raises `PersistenceError` before anything is written, so a save either
    fully succeeds or leaves no file behind. Only parameter *values* are
    saved, never `.grad` or any autograd graph state, and never optimizer
    state -- see `docs/architecture/persistence.md`.

    `model` may live on `"cpu"` or `"cuda"` (see `Module.device`) -- a
    mixed-device tree raises `ModuleError` before anything is written, the
    same error `Module.device` itself already raises. Every `Parameter`'s
    values are copied to host memory for the archive regardless of device
    (a CUDA parameter via a real device-to-host transfer, through the same
    `Backend.to_numpy()` `Tensor.to()` already uses); no model computation
    ever runs as part of saving.
    """
    if not isinstance(model, Module):
        raise PersistenceError(f"save_model() requires a forge.nn.Module, got {type(model).__name__}.")

    model_device = model.device
    device_str = model_device.type if model_device is not None else "cpu"

    arrays: "dict[str, np.ndarray]" = {}
    root_node = _build_save_node(model, prefix="", arrays=arrays)

    metadata = {
        "forge_format_version": FORMAT_VERSION,
        "device": device_str,
        "root": root_node,
    }
    write_archive(path, metadata, arrays)


def load_model(path: str, device: "str | None" = None) -> Module:
    """Reconstruct and return the `Module` saved at `path`.

    Only ever constructs instances of classes registered with
    `forge.serialization.register_module()` in *this* process, using
    JSON-decoded configuration data as keyword arguments -- never `eval`,
    `exec`, dynamic import, or `pickle` on file content. An unknown module
    type, unsupported format version, unsupported/invalid device, or any
    shape/dtype/structural inconsistency between the file and what the
    registered constructors actually produce raises `PersistenceError` with
    a specific reason rather than a raw parsing exception.

    **Device placement policy.** By default (`device=None`), the model is
    restored onto the device recorded in the archive -- but only when that
    device is actually available: a CUDA-saved file loaded with no CUDA
    backend present raises `PersistenceError` explaining that CUDA is
    required, rather than silently falling back to CPU. Passing an explicit
    `device="cpu"` or `device="cuda"` overrides the recorded device (a
    deliberate persistence-time conversion, e.g. loading a CUDA checkpoint
    onto a CPU-only machine); an unavailable `device="cuda"` override still
    fails clearly rather than falling back. Any other `device` value raises
    `PersistenceError`.

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

    saved_device = metadata.get("device")
    if saved_device not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(SUPPORTED_DEVICE_TYPES)
        raise PersistenceError(
            f"Cannot load model from '{path}': saved for device {saved_device!r}, but this build "
            f"of Forge's persistence system only recognizes: {supported}."
        )

    if device is None:
        target_device = saved_device
    elif device in SUPPORTED_DEVICE_TYPES:
        target_device = device
    else:
        supported = ", ".join(SUPPORTED_DEVICE_TYPES)
        raise PersistenceError(
            f"load_model() received an invalid device override {device!r}; "
            f"expected None or one of: {supported}."
        )

    if target_device == "cuda":
        from ..backend.cuda import is_cuda_available

        if not is_cuda_available():
            if device == "cuda":
                raise PersistenceError(
                    "load_model() was explicitly asked to load onto device='cuda', but CUDA is "
                    "not available on this machine."
                )
            raise PersistenceError(
                f"Cannot load model from '{path}': it was saved for device 'cuda', but CUDA is "
                "not available on this machine. Pass device='cpu' to load_model() to explicitly "
                "convert it to a CPU model instead."
            )

    root = metadata.get("root")
    if not isinstance(root, dict):
        raise PersistenceError(f"Cannot load model from '{path}': malformed metadata (missing 'root').")

    try:
        return _build_load_node(root, prefix="", arrays=arrays, path=path, target_device=target_device)
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
        # A persistence transfer, not computation: `Backend.to_numpy()` is the
        # same device-to-host copy `Tensor.to()` already uses, and is a no-op
        # copy for a CPU parameter. No model computation runs here.
        host_array = get_backend(param.device).to_numpy(param._data)
        arrays[dotted] = np.array(host_array, copy=True)
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


def _build_load_node(
    node: dict, prefix: str, arrays: "dict[str, np.ndarray]", path: str, target_device: str
) -> Module:
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
        # `Parameter(..., device=target_device)` routes through the target
        # device's own `Backend.from_array()` -- for `"cuda"` this is a real
        # cudaMalloc + host-to-device transfer (`CUDABackend.from_array`),
        # never a NumPy array relabeled as CUDA storage.
        setattr(
            module,
            name,
            Parameter(array, dtype=expected_dtype, device=target_device, requires_grad=requires_grad),
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
        setattr(
            module,
            name,
            _build_load_node(child_node, child_prefix, arrays, path, target_device=target_device),
        )

    object.__setattr__(module, "_training", bool(node["training"]))
    return module


__all__ = ["save_model", "load_model", "FORMAT_VERSION", "SUPPORTED_DEVICE_TYPES"]
