"""Read-only access to Forge's model/checkpoint archive metadata, for CLI inspection only.

`forge model inspect` / `forge checkpoint inspect` deliberately read only
`metadata.json` via `forge.serialization.archive.read_archive` -- the same
low-level primitive `forge.serialization.model`/`.checkpoint` themselves call
internally -- rather than going through `forge.load_model()`/
`forge.load_checkpoint()`. This is a deliberate departure from those two
higher-level functions, for three reasons specific to *inspection*:

1. **No CUDA requirement, regardless of the recorded device.** `load_model()`/
   `load_checkpoint()` need a real CUDA backend to restore a CUDA-recorded
   archive onto its own device; reading metadata only never touches a
   backend at all, so inspecting a CUDA-saved file works identically on a
   CPU-only machine.
2. **No RNG mutation.** `load_checkpoint()` documents restoring
   `forge.random`'s process-global generator state as part of loading -- a
   real, intentional side effect for *resuming training*, but one an
   `inspect` command must never trigger (Milestone 19 requires inspect
   commands to be strictly read-only: no mutated model/optimizer/RNG/
   training-progress state).
3. **No registry requirement.** Metadata inspection does not need the saved
   module/optimizer types to be registered in the current process --
   `load_model()`/`load_checkpoint()` would refuse to reconstruct an
   unregistered type even though its name, config, and shapes are all
   already sitting in the archive's plain JSON metadata.

`forge model convert` / `forge checkpoint convert` are unaffected by any of
this -- they still call `forge.load_model()`/`forge.save_model()`/
`forge.load_checkpoint()`/`forge.save_checkpoint()` directly, since an actual
device conversion requires the real reconstruction those functions provide.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..backend.device import SUPPORTED_DEVICE_TYPES
from ..exceptions import PersistenceError
from ..serialization.archive import read_archive
from ..serialization.checkpoint import CHECKPOINT_FORMAT_VERSION
from ..serialization.model import FORMAT_VERSION
from .errors import CLIError


def _require_dict_key(node: Any, key: str, what: str) -> Any:
    if not isinstance(node, dict) or key not in node:
        raise CLIError(f"Cannot inspect {what}: malformed archive (missing '{key}').")
    return node[key]


def read_model_metadata(path: str) -> dict:
    """Read and lightly validate a model archive's `metadata.json`, without reconstructing it."""
    try:
        metadata, _ = read_archive(path, kind="model")
    except PersistenceError as exc:
        raise CLIError(str(exc)) from exc
    if not isinstance(metadata, dict):
        raise CLIError(f"Cannot inspect model '{path}': malformed archive (metadata is not an object).")

    version = metadata.get("forge_format_version")
    if version != FORMAT_VERSION:
        raise CLIError(
            f"Cannot inspect model '{path}': unsupported format version {version!r} "
            f"(this build of Forge supports version {FORMAT_VERSION})."
        )
    device = metadata.get("device")
    if device not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(SUPPORTED_DEVICE_TYPES)
        raise CLIError(
            f"Cannot inspect model '{path}': unrecognized recorded device {device!r} "
            f"(expected one of: {supported})."
        )
    _require_dict_key(metadata, "root", f"model '{path}'")
    return metadata


def read_checkpoint_metadata(path: str) -> dict:
    """Read and lightly validate a checkpoint archive's `metadata.json`, without reconstructing it."""
    try:
        metadata, _ = read_archive(path, kind="checkpoint")
    except PersistenceError as exc:
        raise CLIError(str(exc)) from exc
    if not isinstance(metadata, dict):
        raise CLIError(f"Cannot inspect checkpoint '{path}': malformed archive (metadata is not an object).")

    version = metadata.get("forge_checkpoint_format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CLIError(
            f"Cannot inspect checkpoint '{path}': unsupported checkpoint format version {version!r} "
            f"(this build of Forge supports version {CHECKPOINT_FORMAT_VERSION})."
        )
    device = metadata.get("device")
    if device not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(SUPPORTED_DEVICE_TYPES)
        raise CLIError(
            f"Cannot inspect checkpoint '{path}': unrecognized recorded device {device!r} "
            f"(expected one of: {supported})."
        )
    for key in ("model", "optimizer", "training_progress"):
        _require_dict_key(metadata, key, f"checkpoint '{path}'")
    return metadata


def walk_modules(node: dict, prefix: str = "") -> "Iterator[tuple[str, str]]":
    """Yield `(dotted_name, type_name)` for `node` and every descendant, self first.

    `node` follows the same `{"type", "config", "training", "parameters",
    "children"}` shape `forge.serialization.model._build_save_node` writes
    (the archive's own stable, versioned format) -- the root module is
    yielded as `"(root)"`.
    """
    if not isinstance(node, dict) or "type" not in node or "children" not in node:
        raise CLIError("Malformed module metadata: missing 'type' or 'children'.")
    yield prefix or "(root)", node["type"]
    children = node["children"]
    if not isinstance(children, dict):
        raise CLIError("Malformed module metadata: 'children' is not an object.")
    for name, child in children.items():
        child_prefix = name if not prefix else f"{prefix}.{name}"
        yield from walk_modules(child, child_prefix)


def walk_parameters(node: dict, prefix: str = "") -> "Iterator[tuple[str, dict]]":
    """Yield `(dotted_name, {'shape', 'dtype', 'requires_grad'})` for `node` and every descendant."""
    if not isinstance(node, dict) or "parameters" not in node or "children" not in node:
        raise CLIError("Malformed module metadata: missing 'parameters' or 'children'.")
    parameters = node["parameters"]
    if not isinstance(parameters, dict):
        raise CLIError("Malformed module metadata: 'parameters' is not an object.")
    for name, meta in parameters.items():
        dotted = name if not prefix else f"{prefix}.{name}"
        yield dotted, meta
    children = node["children"]
    for name, child in children.items():
        child_prefix = name if not prefix else f"{prefix}.{name}"
        yield from walk_parameters(child, child_prefix)


def module_training_state(node: dict) -> bool:
    if not isinstance(node, dict) or "training" not in node:
        raise CLIError("Malformed module metadata: missing 'training'.")
    return bool(node["training"])


def count_elements(shape: "list[int] | tuple[int, ...]") -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


__all__ = [
    "read_model_metadata",
    "read_checkpoint_metadata",
    "walk_modules",
    "walk_parameters",
    "module_training_state",
    "count_elements",
]
