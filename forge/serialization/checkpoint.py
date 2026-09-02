"""`save_checkpoint()` / `load_checkpoint()`: the training-checkpoint API (Milestone 18).

```text
save_model()      -> model state only        (forge/serialization/model.py)
save_checkpoint() -> training state           (this module)
                       = model state
                       + optimizer type/hyperparameters/per-parameter state
                       + training progress (epoch, global_step)
                       + Forge's default RNG state
                       + caller-supplied JSON-safe `extra`
```

A checkpoint is a strict superset of a model save, in a separate,
independently-versioned file format (`CHECKPOINT_FORMAT_VERSION`,
distinct from `forge.serialization.model.FORMAT_VERSION`) -- see
`docs/architecture/persistence.md`'s **Checkpointing** section. `save_model`/
`load_model` are unchanged and remain optimizer-state-free; this module adds
a second, separate API layered on the same archive primitives
(`forge.serialization.archive`), not a variant of `save_model`.

## Parameter identity across the archive boundary

`Adam`'s runtime state (`forge/optim/adam.py`) keys `optimizer.state` by
`Parameter` **object identity**, which cannot survive serialization. The
checkpoint format therefore records, for every `Parameter` the optimizer
owns, its dotted name (`Module.named_parameters()`'s own naming scheme,
e.g. `"fc1.weight"`) -- both the ordered list defining which parameters the
optimizer was constructed from, and a name -> state mapping for whichever of
those parameters actually have state. On load, the model is reconstructed
first (exactly as `load_model()` does), giving a fresh name -> `Parameter`
mapping; the optimizer is then reconstructed from that same ordered name
list, and each parameter's state is re-associated by name and installed into
the new optimizer's `state` dict by object identity again -- Adam's identity
based runtime lookup is never changed, only bridged across the save/load
boundary. See `forge.serialization.optimizer_registry`.

## Optimizer registry

Exactly like `forge.serialization.registry` for `Module` types:
`load_checkpoint()` never instantiates an optimizer class based solely on a
string read from the archive -- `"optimizer"."type"` is only ever a lookup
key into `forge.serialization.optimizer_registry`'s in-process registry.
`SGD` and `Adam` are the two types registered by default.

## RNG / determinism policy (Policy A)

A checkpoint captures `forge.random`'s process-global default-generator
state (`forge.random.get_state()`) and restores it in `load_checkpoint()`
(`forge.random.set_state()`). This is the one generator `Dropout` draws from
by default -- on CPU directly, and on CUDA as the host-side per-call seed
for `CUDABackend.dropout_mask` (`docs/architecture/cuda-backend.md`'s
**CUDA Dropout** section) -- so restoring it reproduces the exact future
sequence of Dropout draws (CPU mask values, or CUDA per-call seeds) that
would have followed at save time. Training resumed from a checkpoint is
therefore bitwise-deterministic for any model whose only randomness is
default-generator `Dropout`, **provided** the resumed run is driven by the
same sequence of operations (batches, forward/backward calls) as the
original would have been -- a checkpoint does not by itself make an
unrelated resumed run deterministic.

**Not covered:** a `Dropout(..., generator=...)` constructed with an
explicit `numpy.random.Generator` (not Forge's default), or any other
caller-owned generator (e.g. a `DataLoader`'s own `shuffle` generator).
Those live outside any `Module`/`Optimizer` state a checkpoint inspects, and
are the caller's own responsibility to seed/restore for full reproducibility
-- Forge does not build a general-purpose RNG-tracking framework for this.

## What is *not* checkpointed

`TrainingHistory`/`EpochResult` (`forge/training/trainer.py`) are not part
of the checkpoint contract -- they are a plain in-memory record of past
`Trainer.fit()` calls, not state required to *continue* training correctly.
A resumed `Trainer.fit()` call returns a fresh `TrainingHistory` starting
from the checkpoint's `epoch`; a caller wanting continuous history across a
resume should concatenate the returned histories itself. Gradients
(`.grad`) and the autograd graph are never saved, matching `save_model()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import random as forge_random
from ..backend.device import SUPPORTED_DEVICE_TYPES, Device
from ..exceptions import PersistenceError
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from .archive import read_archive, write_archive
from .model import _build_load_node, _build_save_node
from .optimizer_registry import spec_for_class as _optimizer_spec_for_class
from .optimizer_registry import spec_for_name as _optimizer_spec_for_name

CHECKPOINT_FORMAT_VERSION = 1

_MODEL_DIR = "model/parameters"
_OPTIMIZER_STATE_DIR = "optimizer/state"


@dataclass
class Checkpoint:
    """The result of `load_checkpoint()`: a ready-to-use model, optimizer, and progress.

    `model`/`optimizer` are freshly constructed instances (exactly as
    `load_model()`'s return value is a fresh `Module`) -- `optimizer` is
    already bound to `model`'s own reconstructed `Parameter`s, with its
    state restored, so `optimizer.step()` works immediately without any
    further wiring. `epoch`/`global_step` are the training-progress counters
    recorded at save time (see `Trainer.epoch`/`Trainer.global_step`).
    `extra` is whatever JSON-safe dict `save_checkpoint(..., extra=...)` was
    given (`{}` if none).
    """

    model: Module
    optimizer: Optimizer
    epoch: int
    global_step: int
    extra: "dict[str, Any]"


def save_checkpoint(
    path: str,
    model: Module,
    optimizer: Optimizer,
    *,
    epoch: int = 0,
    global_step: int = 0,
    extra: "dict[str, Any] | None" = None,
) -> None:
    """Save `model` + `optimizer` state + training progress to `path`.

    `model` must be built entirely from registered module types (same
    requirement as `save_model()`); `optimizer` must be an instance of a
    type registered with `forge.serialization.optimizer_registry` (`SGD`,
    `Adam` are built in). Every `Parameter` `optimizer` was constructed from
    must be reachable from `model.named_parameters()` -- an optimizer built
    from parameters outside `model`'s own tree raises `PersistenceError`,
    since checkpoint state is associated by dotted name within `model`.

    `epoch`/`global_step` are plain training-progress counters the caller
    supplies (`Trainer.save_checkpoint()` fills these in automatically from
    `Trainer.epoch`/`Trainer.global_step`); `extra` is an optional JSON-safe
    dict for any small additional caller-defined state (e.g. best validation
    loss so far) -- never arbitrary Python objects.

    As with `save_model()`, saving never runs model computation: every
    array is copied to host memory via `Backend.to_numpy()` before being
    written, and the archive is written atomically (temp file + `os.replace`).
    """
    if not isinstance(model, Module):
        raise PersistenceError(f"save_checkpoint() requires a forge.nn.Module model, got {type(model).__name__}.")
    if not isinstance(optimizer, Optimizer):
        raise PersistenceError(
            f"save_checkpoint() requires a forge.optim.Optimizer optimizer, got {type(optimizer).__name__}."
        )
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise PersistenceError(f"save_checkpoint() requires epoch >= 0 (int), got {epoch!r}.")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise PersistenceError(f"save_checkpoint() requires global_step >= 0 (int), got {global_step!r}.")
    if extra is not None and not isinstance(extra, dict):
        raise PersistenceError(f"save_checkpoint() requires extra to be a dict or None, got {type(extra).__name__}.")

    model_device = model.device
    device_str = model_device.type if model_device is not None else "cpu"

    arrays: "dict[str, np.ndarray]" = {}

    model_arrays: "dict[str, np.ndarray]" = {}
    model_root = _build_save_node(model, prefix="", arrays=model_arrays)
    for name, array in model_arrays.items():
        arrays[f"{_MODEL_DIR}/{name}"] = array

    opt_spec = _optimizer_spec_for_class(type(optimizer))
    opt_config = opt_spec.get_config(optimizer)
    if not isinstance(opt_config, dict):
        raise PersistenceError(
            f"Cannot save optimizer type '{opt_spec.type_name}': get_config() must return a "
            f"dict, got {type(opt_config).__name__}."
        )

    name_by_id = {id(param): name for name, param in model.named_parameters()}
    optimizer_param_names: "list[str]" = []
    param_state_meta: "dict[str, dict]" = {}
    for param in optimizer.parameters:
        name = name_by_id.get(id(param))
        if name is None:
            raise PersistenceError(
                "save_checkpoint(): the optimizer owns a Parameter that is not reachable from "
                "model.named_parameters() -- optimizer state can only be checkpointed for "
                "parameters that belong to the given model."
            )
        optimizer_param_names.append(name)

        state = opt_spec.get_param_state(optimizer, param)
        if state is None:
            continue
        if not isinstance(state, dict):
            raise PersistenceError(
                f"Cannot save optimizer type '{opt_spec.type_name}': get_param_state() must "
                f"return a dict or None, got {type(state).__name__}."
            )
        scalars: "dict[str, Any]" = {}
        array_fields: "list[str]" = []
        for field, value in state.items():
            if isinstance(value, np.ndarray):
                arrays[f"{_OPTIMIZER_STATE_DIR}/{name}/{field}"] = np.array(value, copy=True)
                array_fields.append(field)
            else:
                scalars[field] = value
        param_state_meta[name] = {"scalars": scalars, "arrays": array_fields}

    metadata = {
        "forge_checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "device": device_str,
        "model": model_root,
        "optimizer": {
            "type": opt_spec.type_name,
            "config": opt_config,
            "parameters": optimizer_param_names,
            "param_state": param_state_meta,
        },
        "training_progress": {"epoch": epoch, "global_step": global_step},
        "rng": {"default_generator_state": forge_random.get_state()},
        "extra": extra if extra is not None else {},
    }

    try:
        write_archive(path, metadata, arrays)
    except TypeError as exc:
        raise PersistenceError(f"Cannot save checkpoint to '{path}': non-JSON-safe data ({exc}).") from exc


def load_checkpoint(path: str, device: "str | None" = None) -> Checkpoint:
    """Reconstruct and return the `Checkpoint` saved at `path`.

    `device` follows the exact same policy as `load_model(path, device=)`:
    `None` restores the recorded device (failing clearly if that device
    isn't available), `"cpu"`/`"cuda"` explicitly override it (still
    requiring CUDA to actually be available for a `"cuda"` target). Both the
    reconstructed model's `Parameter`s and the reconstructed optimizer's
    per-parameter state (Adam's `m`/`v`) end up on the same resolved target
    device -- restoring onto `"cuda"` allocates genuine `CUDAStorage` via
    `Backend.from_array()`, never a NumPy array relabeled as CUDA.

    Restoring also overwrites `forge.random`'s process-global default
    generator state with the one captured at save time -- see this module's
    docstring, **RNG / determinism policy**, for exactly what that does and
    does not make deterministic.

    Only ever constructs `Module`/`Optimizer` instances registered in this
    process (`forge.serialization.registry` / `.optimizer_registry`); never
    `eval`, `exec`, dynamic import, or `pickle` on archive content. Raises
    `PersistenceError` for any structural inconsistency, unsupported format
    version, or unrecognized/unavailable device.
    """
    metadata, arrays = read_archive(path, kind="checkpoint")

    if not isinstance(metadata, dict):
        raise PersistenceError(f"Cannot load checkpoint from '{path}': metadata is not a JSON object.")

    version = metadata.get("forge_checkpoint_format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise PersistenceError(
            f"Cannot load checkpoint from '{path}': unsupported checkpoint format version "
            f"{version!r} (this build of Forge supports version {CHECKPOINT_FORMAT_VERSION})."
        )

    saved_device = metadata.get("device")
    if saved_device not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(SUPPORTED_DEVICE_TYPES)
        raise PersistenceError(
            f"Cannot load checkpoint from '{path}': saved for device {saved_device!r}, but this "
            f"build of Forge's persistence system only recognizes: {supported}."
        )

    if device is None:
        target_device_str = saved_device
    elif device in SUPPORTED_DEVICE_TYPES:
        target_device_str = device
    else:
        supported = ", ".join(SUPPORTED_DEVICE_TYPES)
        raise PersistenceError(
            f"load_checkpoint() received an invalid device override {device!r}; "
            f"expected None or one of: {supported}."
        )

    if target_device_str == "cuda":
        from ..backend.cuda import is_cuda_available

        if not is_cuda_available():
            if device == "cuda":
                raise PersistenceError(
                    "load_checkpoint() was explicitly asked to load onto device='cuda', but "
                    "CUDA is not available on this machine."
                )
            raise PersistenceError(
                f"Cannot load checkpoint from '{path}': it was saved for device 'cuda', but "
                "CUDA is not available on this machine. Pass device='cpu' to load_checkpoint() "
                "to explicitly convert it to a CPU checkpoint instead."
            )

    model_root = metadata.get("model")
    if not isinstance(model_root, dict):
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed metadata (missing 'model').")

    model_prefix = f"{_MODEL_DIR}/"
    model_arrays = {
        name[len(model_prefix):]: array for name, array in arrays.items() if name.startswith(model_prefix)
    }

    try:
        model = _build_load_node(model_root, prefix="", arrays=model_arrays, path=path, target_device=target_device_str)
    except PersistenceError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed model metadata ({exc}).") from exc

    optimizer_meta = metadata.get("optimizer")
    if not isinstance(optimizer_meta, dict):
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed metadata (missing 'optimizer').")
    for key in ("type", "config", "parameters", "param_state"):
        if key not in optimizer_meta:
            raise PersistenceError(
                f"Cannot load checkpoint from '{path}': malformed metadata, 'optimizer' is missing '{key}'."
            )

    opt_spec = _optimizer_spec_for_name(optimizer_meta["type"])
    opt_config = optimizer_meta["config"]
    if not isinstance(opt_config, dict):
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed metadata, 'optimizer.config' is not an object.")

    name_to_param = dict(model.named_parameters())
    param_names = optimizer_meta["parameters"]
    if not isinstance(param_names, list):
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed metadata, 'optimizer.parameters' is not a list.")
    ordered_params = []
    for name in param_names:
        param = name_to_param.get(name)
        if param is None:
            raise PersistenceError(
                f"Cannot load checkpoint from '{path}': optimizer references parameter '{name}', "
                "which does not exist in the reconstructed model."
            )
        ordered_params.append(param)

    try:
        optimizer = opt_spec.from_config(ordered_params, opt_config)
    except PersistenceError:
        raise
    except Exception as exc:
        raise PersistenceError(
            f"Cannot load checkpoint from '{path}': invalid configuration for optimizer type "
            f"'{opt_spec.type_name}': {exc}."
        ) from exc
    if not isinstance(optimizer, Optimizer):
        raise PersistenceError(
            f"Cannot load checkpoint from '{path}': the registered constructor for optimizer "
            f"type '{opt_spec.type_name}' did not produce a forge.optim.Optimizer."
        )

    param_state_meta = optimizer_meta["param_state"]
    if not isinstance(param_state_meta, dict):
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed metadata, 'optimizer.param_state' is not an object.")

    state_prefix = f"{_OPTIMIZER_STATE_DIR}/"
    for name, entry in param_state_meta.items():
        param = name_to_param.get(name)
        if param is None:
            raise PersistenceError(
                f"Cannot load checkpoint from '{path}': optimizer state references parameter "
                f"'{name}', which does not exist in the reconstructed model."
            )
        if not isinstance(entry, dict) or "scalars" not in entry or "arrays" not in entry:
            raise PersistenceError(
                f"Cannot load checkpoint from '{path}': malformed optimizer state entry for '{name}'."
            )
        scalars = entry["scalars"]
        field_arrays: "dict[str, np.ndarray]" = {}
        for field in entry["arrays"]:
            array_key = f"{state_prefix}{name}/{field}"
            array = arrays.get(array_key)
            if array is None:
                raise PersistenceError(
                    f"Cannot load checkpoint from '{path}': missing optimizer state array "
                    f"'{field}' for parameter '{name}'."
                )
            field_arrays[field] = array
        try:
            opt_spec.set_param_state(optimizer, param, field_arrays, scalars, param._device)
        except PersistenceError:
            raise
        except Exception as exc:
            raise PersistenceError(
                f"Cannot load checkpoint from '{path}': failed to restore optimizer state for "
                f"parameter '{name}': {exc}."
            ) from exc

    progress = metadata.get("training_progress")
    if not isinstance(progress, dict) or "epoch" not in progress or "global_step" not in progress:
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed metadata (missing 'training_progress').")

    rng_meta = metadata.get("rng")
    if isinstance(rng_meta, dict) and "default_generator_state" in rng_meta:
        try:
            forge_random.set_state(rng_meta["default_generator_state"])
        except (TypeError, ValueError, KeyError) as exc:
            raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed RNG state ({exc}).") from exc

    extra = metadata.get("extra", {})
    if not isinstance(extra, dict):
        raise PersistenceError(f"Cannot load checkpoint from '{path}': malformed metadata, 'extra' is not an object.")

    return Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=int(progress["epoch"]),
        global_step=int(progress["global_step"]),
        extra=extra,
    )


__all__ = ["save_checkpoint", "load_checkpoint", "Checkpoint", "CHECKPOINT_FORMAT_VERSION"]
