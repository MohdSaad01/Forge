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

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .. import random as forge_random
from ..backend import get_backend
from ..backend.device import SUPPORTED_DEVICE_TYPES
from ..data.feature_names import validate_feature_names
from ..data.target_transform import TARGET_TRANSFORM_TYPES, StandardizeTarget
from ..exceptions import DataError, PersistenceError
from ..nn.module import Module
from ..nn.parameter import Parameter
from ..tensor.tensor import Tensor
from .archive import PARAMETERS_DIR, read_archive, write_archive
from .registry import spec_for_class, spec_for_name
from .transforms import deserialize_transform, serialize_transform

FORMAT_VERSION = 2

# Milestone 116: an artifact that carries a `"target_transform"` (a regression model
# trained on transformed targets, whose raw output is NOT in the user's units) is
# written as version 3; every other artifact is still written as version 2, byte for
# byte as before. The version is what keeps an older Forge from reading a version-3
# file, ignoring the unknown `"target_transform"` key, and returning standardised
# numbers as if they were native: it refuses the file ("unsupported format version 3")
# instead. This build reads both. Version 3 *requires* the key and version 2 *forbids*
# it, so a stripped or smuggled key is an error rather than a silent reinterpretation.
TARGET_TRANSFORM_FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (FORMAT_VERSION, TARGET_TRANSFORM_FORMAT_VERSION)


def _supported_versions_text() -> str:
    return " and ".join(str(v) for v in SUPPORTED_FORMAT_VERSIONS)

# Milestone 87 (classification/regression/segmentation) + Milestone 90
# (sequence) + Milestone 91 (tabular_classification): the fixed, small
# vocabulary of prediction workflows a saved artifact can explicitly declare
# -- see save_model()'s `task=` parameter and `docs/architecture/
# persistence.md`'s **Task metadata** section. Each name corresponds
# directly to one of Forge's five existing artifact inference workflows
# (`forge.predict_artifact()`/`predict_tensor_artifact()`/
# `predict_image_artifact()`/`predict_sequence_artifact()`/
# `predict_tabular_classification_artifact()`, Milestones 82-84/90/91) -- not
# a generic/open-ended task registry. `"sequence"` is a genuinely distinct
# *prediction problem* from the other three original tasks (autoregressive
# next-token generation from a seed, over a stepwise `model.step()`/
# `model.init_hidden()` recurrence -- see `predict_sequence_artifact()`), not
# merely "uses an RNN/LSTM layer" -- an RNN-based classifier would still be
# saved with `task="classification"`. `"tabular_classification"` is likewise
# a genuinely distinct *input modality* from `"classification"`, not a
# variant of it: `task="classification"` always means "input is an image
# file path, decoded via `ImageFolder._load_image()`" (`predict_artifact()`);
# `task="tabular_classification"` means "input is an already-batched numeric
# feature vector" (`predict_tabular_classification_artifact()`), exactly the
# same input shape `task="regression"` already uses. A tabular classifier
# (numeric features in, a class label out) genuinely needs the second shape,
# not the first -- see `predict_tabular_classification_artifact()`'s own
# docstring and `docs/development/m91-tabular-classification-artifact-inference.md`
# for the real, demonstrated gap this closes.
TASK_TYPES = ("classification", "regression", "segmentation", "sequence", "tabular_classification")


def _validate_task(task: "str | None", classes: "list[str] | None") -> None:
    if task is None:
        return
    if task not in TASK_TYPES:
        raise PersistenceError(
            f"save_model() task= must be one of {TASK_TYPES!r}, got {task!r}."
        )
    if task in ("regression", "segmentation") and classes is not None:
        raise PersistenceError(
            f"save_model() task={task!r} is incompatible with classes= -- {task} artifacts have "
            f"no class vocabulary (only task='classification'/'sequence' may be saved with classes=)."
        )
    if task == "sequence" and classes is None:
        raise PersistenceError(
            "save_model() task='sequence' requires classes= -- a sequence artifact's vocabulary "
            "(index i -> classes[i] token), the same list `predict_sequence_artifact()` needs to "
            "encode/decode tokens. There is no automatic way to derive a token vocabulary from "
            "model architecture alone."
        )


def _validate_classes(classes: "list[str] | None", task: "str | None" = None) -> None:
    if classes is None:
        return
    if not isinstance(classes, list) or not classes:
        raise PersistenceError(
            f"save_model() classes= must be a non-empty list of strings, got {classes!r}."
        )
    for label in classes:
        # A classification label that is empty or all whitespace is almost
        # certainly a mistake (`label.strip()` catches both) -- but for
        # task="sequence", `classes` is a *token vocabulary*, not a set of
        # human-readable names, and a whitespace character (a space, most
        # commonly) is one of the most ordinary tokens a text vocabulary can
        # contain (discovered via this milestone's own char-RNN artifact:
        # `Vocab.chars` for any real corpus with word boundaries includes
        # `" "`). Only the genuinely empty string (no token content at all)
        # is rejected for a sequence vocabulary.
        if not isinstance(label, str) or (label == "" if task == "sequence" else not label.strip()):
            what = "tokens" if task == "sequence" else "strings"
            raise PersistenceError(
                f"save_model() classes= must contain only non-empty {what}, got {label!r} "
                f"in {classes!r}."
            )
    if len(set(classes)) != len(classes):
        raise PersistenceError(
            f"save_model() classes= must not contain duplicate labels, got {classes!r}."
        )


def _validate_target_transform(target_transform: Any, task: "str | None") -> None:
    if target_transform is None:
        return
    if not isinstance(target_transform, StandardizeTarget):
        raise PersistenceError(
            f"save_model() target_transform= must be a forge.data.StandardizeTarget, got "
            f"{type(target_transform).__name__}. Only the closed set of persistable target "
            f"transforms {TARGET_TRANSFORM_TYPES!r} can be saved (arbitrary callables cannot)."
        )
    if task != "regression":
        raise PersistenceError(
            f"save_model() target_transform= requires task='regression', got task={task!r} -- only a "
            "regression artifact has a target to map back to native units."
        )


def _validate_feature_names_for_save(
    feature_names: Any, task: "str | None", root_node: dict,
) -> "tuple[str, ...] | None":
    """The validated `feature_names=` for `save_model()`, or `None` (Milestone 119).

    Beyond `validate_feature_names()`'s own rules, a name list is only meaningful on an
    artifact that has an input contract to attach it to, and it must describe exactly
    that contract: one name per input column. Both are checked against the tree that is
    about to be written -- never against anything guessed -- so a saved file can never
    carry names its own model cannot have.
    """
    if feature_names is None:
        return None
    try:
        names = validate_feature_names(feature_names)
    except DataError as exc:
        raise PersistenceError(f"save_model() {exc}") from exc
    if task not in _INPUT_SCHEMA_TASKS:
        raise PersistenceError(
            f"save_model() feature_names= requires task in {_INPUT_SCHEMA_TASKS!r} (the tabular workflows, "
            f"whose input is a fixed-width numeric row), got task={task!r}."
        )
    width = _leading_linear_in_features(root_node)
    if width is None:
        raise PersistenceError(
            "save_model() feature_names= needs a model whose input width Forge can read from its saved "
            "architecture (a Sequential whose first layer is Linear), and this model's is not one -- the "
            "names could not be checked against the input they describe."
        )
    if len(names) != width:
        raise PersistenceError(
            f"save_model() feature_names= has {len(names)} name(s), but the model's first Linear layer takes "
            f"{width} input feature(s). There must be exactly one name per input column."
        )
    return names


def save_model(
    model: Module,
    path: str,
    preprocessing: "Any | None" = None,
    classes: "list[str] | None" = None,
    task: "str | None" = None,
    target_transform: "StandardizeTarget | None" = None,
    feature_names: "Sequence[str] | None" = None,
) -> None:
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

    `preprocessing` (Milestone 71) optionally records the
    `forge.data.transforms.Transform` (e.g. a `Resize`/`Compose` pipeline)
    a caller's inputs must already have been passed through before reaching
    `model` -- the exact preprocessing configuration, not a copy of the
    model's own state, saved as a JSON-safe sibling metadata entry (see
    `forge.serialization.transforms.serialize_transform`) rather than an
    attribute of `model` itself: a model's parameters and its input
    preprocessing are conceptually distinct, and `Module` gains no new
    state for this. Only transform types registered with
    `forge.serialization.transforms.register_transform()` (`Resize`,
    `Compose`, `Normalize`, `ToTensor`, `Reshape`, `Flatten`) can be saved
    this way -- notably **not** `Lambda`, which wraps an arbitrary Python
    callable with no safe serialized representation; passing one raises
    `PersistenceError` immediately. Omitting `preprocessing` (the default)
    writes no `"preprocessing"` key at all, so files saved before Milestone
    71 and files saved with no preprocessing configured are byte-for-byte
    equivalent in this respect and remain loadable by `load_model()`
    unchanged -- see `load_preprocessing()`.

    `classes` (Milestone 72; reused for `task="sequence"` in Milestone 90)
    optionally records the ordered list of human-readable strings a model's
    output indices refer to -- `output[..., i]` means `classes[i]`, matching
    `forge.data.ImageFolder.classes`'s own index convention exactly, so a
    caller can pass `some_image_folder.classes` directly. For a
    `task="sequence"` artifact, this same index-to-string list *is* the
    model's token vocabulary (e.g. `Vocab.chars` in `examples/char_rnn`) --
    the concept ("index i names token/class i") is identical, so Milestone 90
    reused `classes` rather than inventing a parallel `vocab=` parameter. Must
    be a non-empty list of unique strings; anything else raises
    `PersistenceError` before anything is written. For `task="classification"`
    (or no task), each string must also be non-whitespace (a label of `" "`
    is almost always a mistake); `task="sequence"` relaxes this to "non-empty"
    only, since a whitespace character is an ordinary, common vocabulary
    token (e.g. `" "` in `examples/char_rnn`'s vocabulary) rather than a
    human-readable label. Stored as a plain
    JSON-safe list, a sibling metadata entry alongside `"preprocessing"` --
    never inferred from `model`'s architecture (Forge does not introspect a
    module tree to guess an output-class count), and never merged into
    `preprocessing` (a class vocabulary is not "how to prepare an input",
    it is how to interpret an output). See `load_classes()`,
    `forge.training.interpret_classification()`, and
    `forge.training.predict_sequence_artifact()`.

    `task` (Milestone 87; extended in Milestones 90/91) optionally declares
    which of Forge's five portable-artifact inference workflows this file
    represents -- `"classification"`, `"regression"`, `"segmentation"`,
    `"sequence"`, or `"tabular_classification"` (`forge.serialization.model.
    TASK_TYPES`) -- so `forge.predict_model()` can dispatch to the right one
    of `predict_artifact()`/`predict_tensor_artifact()`/
    `predict_image_artifact()`/`predict_sequence_artifact()`/
    `predict_tabular_classification_artifact()` reliably, from the artifact's
    own metadata, rather than guessing from its module architecture (see
    `docs/architecture/persistence.md`'s **Task metadata** section for why
    the pre-Milestone-87 architecture-based guess was unreliable, and why
    Milestone 90's stepwise-recurrence models cannot use that guess at all --
    they have no `forward()`). `"tabular_classification"` (Milestone 91) is
    the same distinction for a different reason: it is architecturally
    identical to `"classification"` (both may be `Linear`-terminated with a
    `classes=` vocabulary) but expects an already-batched numeric feature
    vector, not an image file path -- no architecture-based guess could ever
    tell the two apart, so an explicit declaration is the only reliable
    signal, exactly like `"sequence"`'s own reasoning. Any other string
    raises `PersistenceError` before anything is written -- this is a small,
    fixed vocabulary matching Forge's five existing inference workflows
    exactly, not an open-ended task registry.

    `task="regression"` or `task="segmentation"` combined with a non-`None`
    `classes=` raises `PersistenceError` immediately: neither workflow has a
    class-vocabulary concept (see `predict_tensor_artifact()`/
    `predict_image_artifact()`'s own docstrings), so saving both together
    would describe an artifact whose two pieces of metadata disagree about
    what it is. `task="sequence"` is the opposite: it *requires* `classes=`
    (the token vocabulary `predict_sequence_artifact()` needs) and raises
    `PersistenceError` if omitted. `task="classification"`/
    `task="tabular_classification"` place no such restriction on `classes` --
    `classes=None` remains a real, valid state for either (see
    `predict_artifact()`'s/`predict_tabular_classification_artifact()`'s own
    docstrings), and `task=` is what now lets `predict_model()` recognize
    that state correctly instead of misidentifying it as regression (the M86
    ambiguity `docs/development/m86-unified-artifact-prediction.md`
    documented and Milestone 87 closes).

    Omitting `task` (the default) writes no `"task"` key at all -- exactly
    like `preprocessing=`/`classes=`'s own optional-key convention -- so
    files saved before Milestone 87 and files saved with no task metadata
    are byte-for-byte equivalent in this respect and remain fully loadable.
    `forge.predict_model()` falls back to an isolated, documented legacy
    heuristic for such files (`forge/training/inference.py::
    _legacy_infer_workflow()`) rather than treating an absent task as any
    particular value.

    `target_transform` (Milestone 116) optionally records the
    `forge.data.StandardizeTarget` a regression model's *targets* went through in
    training -- the output-side counterpart of `preprocessing`. The model then
    predicts in that transformed space, and `predict()`/`evaluate()` on the artifact
    (`load_predictor()`, `predict_tensor_artifact()`, `forge model predict`) apply its
    inverse so callers get native units and never need to know it exists. It requires
    `task="regression"` (`PersistenceError` otherwise) and is written as a JSON-safe
    `"target_transform"` sibling entry. Only this closed, configuration-only type is
    accepted: no callable and nothing pickled. An artifact saved with a target
    transform has format version 3; omitting it (the default) writes the
    version-2 file exactly as before -- see `TARGET_TRANSFORM_FORMAT_VERSION`,
    `load_target_transform()`. `load_model()` alone still returns just the model,
    whose raw output is in the transformed space.

    `feature_names` (Milestone 119) optionally records *which* feature each input column is,
    in the order the model consumes them -- the identity half of the input contract that
    `InputSchema.feature_count` (a width only) cannot express. A later named input
    (`predict(..., feature_names=...)`, a CSV header) is then matched to it by name: the
    same names in a different order are reordered, anything else is rejected. Requires
    `task="regression"` or `"tabular_classification"` and exactly one name per input
    column of the model's first `Linear` layer (`PersistenceError` otherwise); a list of
    non-empty, mutually different strings, stored exactly as given -- never stripped or
    case-folded. Names describe the *raw* input columns, i.e. what the caller passes,
    before any persisted `preprocessing` runs. Omitting it (the default) writes no
    `"feature_names"` key: the file is byte-for-byte what it was before this milestone and
    still format version 2 (or 3 with a target transform). A build that predates the key
    ignores it and behaves as it always did -- it checks the input's width, not its
    columns -- which is why it needs no format-version change (a target transform, by
    contrast, changes what the model's *output means* and did). See `InputSchema`,
    `inspect_model()`.
    """
    if not isinstance(model, Module):
        raise PersistenceError(f"save_model() requires a forge.nn.Module, got {type(model).__name__}.")
    _validate_classes(classes, task)
    _validate_task(task, classes)
    _validate_target_transform(target_transform, task)

    model_device = model.device
    device_str = model_device.type if model_device is not None else "cpu"

    arrays: "dict[str, np.ndarray]" = {}
    root_node = _build_save_node(model, prefix="", arrays=arrays)
    names = _validate_feature_names_for_save(feature_names, task, root_node)

    preprocessing_node = serialize_transform(preprocessing) if preprocessing is not None else None

    metadata = {
        "forge_format_version": FORMAT_VERSION if target_transform is None else TARGET_TRANSFORM_FORMAT_VERSION,
        "device": device_str,
        "root": root_node,
        "preprocessing": preprocessing_node,
        "classes": list(classes) if classes is not None else None,
        "task": task,
    }
    if target_transform is not None:
        metadata["target_transform"] = target_transform.to_config()
    if names is not None:
        metadata["feature_names"] = list(names)
    prefixed_arrays = {f"{PARAMETERS_DIR}/{name}": array for name, array in arrays.items()}
    write_archive(path, metadata, prefixed_arrays)


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
    metadata, prefixed_arrays = read_archive(path, kind="model")
    param_prefix = f"{PARAMETERS_DIR}/"
    arrays = {
        name[len(param_prefix):]: array
        for name, array in prefixed_arrays.items()
        if name.startswith(param_prefix)
    }

    if not isinstance(metadata, dict):
        raise PersistenceError(f"Cannot load model from '{path}': metadata is not a JSON object.")

    version = metadata.get("forge_format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise PersistenceError(
            f"Cannot load model from '{path}': unsupported format version {version!r} "
            f"(this build of Forge supports version {_supported_versions_text()})."
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

    # Reconstructing a module tree calls each registered type's ordinary
    # constructor (e.g. `Conv2d.__init__`), which draws an initial-weights
    # sample from `forge.random.default_generator()` -- immediately
    # overwritten below by this file's own saved parameter values, so the
    # draw itself is wasted, not meaningful. Snapshotting/restoring
    # `forge.random`'s state around reconstruction (the same get_state()/
    # set_state() mechanism `forge.serialization.checkpoint` already uses)
    # makes `load_model()` a pure read with no observable side effect on
    # unrelated future draws (e.g. a caller's next `Dropout` step) -- a
    # real, previously-invisible leak Milestone 81 found: calling
    # `load_model()` between two otherwise-identical training runs (e.g.
    # via `save_and_verify()`) silently shifted their subsequent Dropout
    # draws out of sync.
    random_state = forge_random.get_state()
    try:
        return _build_load_node(root, prefix="", arrays=arrays, path=path, target_device=target_device)
    except PersistenceError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise PersistenceError(f"Cannot load model from '{path}': malformed metadata ({exc}).") from exc
    finally:
        forge_random.set_state(random_state)


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

    # Milestone 53: buffers (e.g. `nn.BatchNorm2d`'s `running_mean`/
    # `running_var`) -- saved the same way as parameters (a values array plus
    # shape/dtype metadata), just from `_buffers` instead of `_parameters`,
    # and with no `requires_grad` field (always False -- see
    # `Module.register_buffer`). A buffer registered as `None` (unset) is
    # recorded as `None` in the metadata with no array, so `_build_load_node`
    # can tell "no buffer data to restore" apart from "buffer data present".
    buffers: "dict[str, dict | None]" = {}
    for name, buf in module._buffers.items():
        dotted = name if not prefix else f"{prefix}.{name}"
        if buf is None:
            buffers[name] = None
            continue
        host_array = get_backend(buf.device).to_numpy(buf._data)
        arrays[dotted] = np.array(host_array, copy=True)
        buffers[name] = {"shape": list(buf.shape), "dtype": str(buf.dtype)}

    children: "dict[str, dict]" = {}
    for name, child in module._modules.items():
        child_prefix = name if not prefix else f"{prefix}.{name}"
        children[name] = _build_save_node(child, child_prefix, arrays)

    return {
        "type": spec.type_name,
        "config": config,
        "training": module.training,
        "parameters": parameters,
        "buffers": buffers,
        "children": children,
    }


# -- load: recursive tree walk -------------------------------------------


def _build_load_node(
    node: dict, prefix: str, arrays: "dict[str, np.ndarray]", path: str, target_device: str
) -> Module:
    label = prefix or "<root>"
    for key in ("type", "config", "training", "parameters", "buffers", "children"):
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

    # Milestone 53: buffers, mirroring the parameter-restoration loop above
    # exactly except there is no `requires_grad` field (a buffer never
    # requires grad -- `Module.register_buffer` enforces that at
    # registration time) and a `None` entry means "no buffer data" rather
    # than "missing/malformed" (an optional buffer no consumer ever set).
    buffer_meta = node["buffers"]
    if not isinstance(buffer_meta, dict):
        raise PersistenceError(
            f"Cannot load model from '{path}': malformed metadata, 'buffers' for module "
            f"'{label}' is not an object."
        )
    expected_buffer_names = set(buffer_meta.keys())
    actual_buffer_names = set(module._buffers.keys())
    if expected_buffer_names != actual_buffer_names:
        raise PersistenceError(
            f"Cannot load model from '{path}': inconsistent model state for module '{label}' "
            f"(type '{type_name}'): file declares buffers {sorted(expected_buffer_names)}, "
            f"but the reconstructed module has {sorted(actual_buffer_names)}."
        )

    for name, meta in buffer_meta.items():
        if meta is None:
            setattr(module, name, None)
            continue
        if not isinstance(meta, dict):
            raise PersistenceError(
                f"Cannot load model from '{path}': malformed metadata, buffer '{name}' for "
                f"module '{label}' is not an object or null."
            )
        dotted = name if not prefix else f"{prefix}.{name}"
        array = arrays.get(dotted)
        if array is None:
            raise PersistenceError(
                f"Cannot load model from '{path}': missing buffer data for '{dotted}'."
            )
        expected_shape = tuple(meta.get("shape", []))
        if tuple(array.shape) != expected_shape:
            raise PersistenceError(
                f"Cannot load model from '{path}': buffer '{dotted}' has shape "
                f"{tuple(array.shape)} in the file but metadata declares {expected_shape}."
            )
        expected_dtype = meta.get("dtype")
        if str(array.dtype) != expected_dtype:
            raise PersistenceError(
                f"Cannot load model from '{path}': buffer '{dotted}' has dtype "
                f"'{array.dtype}' in the file but metadata declares '{expected_dtype}'."
            )
        setattr(module, name, Tensor(array, dtype=expected_dtype, device=target_device, requires_grad=False))

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


def load_preprocessing(path: str) -> "Any | None":
    """Reconstruct the `Transform` saved alongside a model at `path`, or `None`.

    Reads only the `"preprocessing"` metadata entry a `save_model(...,
    preprocessing=...)` call wrote -- independent of `load_model()`, so a
    caller can fetch just the preprocessing configuration (e.g. before
    deciding which device to load the model itself onto), or call both
    against the same file. Returns `None` when `path` was saved with no
    `preprocessing` (including every file saved before Milestone 71, or a
    Milestone-71-or-later file saved with `preprocessing=None`) -- this is
    an ordinary, expected outcome, not an error.

    Raises `PersistenceError` for a corrupt/unreadable file, or a
    `"preprocessing"` entry that fails to reconstruct (unregistered
    transform type, malformed configuration) -- see
    `forge.serialization.transforms.deserialize_transform`.
    """
    metadata, _ = read_archive(path, kind="model")
    if not isinstance(metadata, dict):
        raise PersistenceError(f"Cannot load preprocessing from '{path}': metadata is not a JSON object.")
    node = metadata.get("preprocessing")
    if node is None:
        return None
    return deserialize_transform(node)


def load_classes(path: str) -> "list[str] | None":
    """Reconstruct the class-name vocabulary saved alongside a model at `path`, or `None`.

    Mirrors `load_preprocessing()` exactly: reads only the `"classes"`
    metadata entry a `save_model(..., classes=...)` call wrote, independent
    of `load_model()`. Returns `None` when `path` was saved with no
    `classes` (including every file saved before Milestone 72, or a
    Milestone-72-or-later file saved with `classes=None`) -- an ordinary,
    expected outcome, not an error.

    Raises `PersistenceError` for a corrupt/unreadable file or a
    `"classes"` entry that is present but not a JSON list of strings
    (a malformed/tampered file, since `save_model()` itself never writes
    anything else there).
    """
    metadata, _ = read_archive(path, kind="model")
    if not isinstance(metadata, dict):
        raise PersistenceError(f"Cannot load classes from '{path}': metadata is not a JSON object.")
    classes = metadata.get("classes")
    if classes is None:
        return None
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise PersistenceError(
            f"Cannot load classes from '{path}': malformed 'classes' metadata (expected a list "
            f"of strings, got {classes!r})."
        )
    return classes


def _target_transform_from_metadata(metadata: dict, path: str, what: str) -> "StandardizeTarget | None":
    """The `StandardizeTarget` an artifact's metadata declares, or `None` -- validated, never guessed (Milestone 116).

    No `"target_transform"` key and format version 2 is the ordinary artifact (every
    file saved before Milestone 116 is this): the identity, `None`. Anything that
    would make the model's raw output be misread as native units is an error instead:
    version 3 without the key (stripped), version 2 with the key (a version-2 reader
    would not know to invert it), a key that is not a well-formed `standardize`
    node, or one on a non-regression artifact.
    """
    version = metadata.get("forge_format_version")
    present = "target_transform" in metadata
    if version == TARGET_TRANSFORM_FORMAT_VERSION and not present:
        raise PersistenceError(
            f"Cannot {what} '{path}': format version {TARGET_TRANSFORM_FORMAT_VERSION} declares a target "
            "transform, but the 'target_transform' metadata entry is missing. The model's output is in "
            "transformed units and cannot be converted back without it."
        )
    if not present:
        return None
    if version != TARGET_TRANSFORM_FORMAT_VERSION:
        raise PersistenceError(
            f"Cannot {what} '{path}': it has a 'target_transform' entry but format version {version!r} "
            f"(a target transform requires version {TARGET_TRANSFORM_FORMAT_VERSION})."
        )
    node = metadata["target_transform"]
    if (
        not isinstance(node, dict)
        or node.get("type") not in TARGET_TRANSFORM_TYPES
        or "mean" not in node
        or "std" not in node
    ):
        raise PersistenceError(
            f"Cannot {what} '{path}': malformed 'target_transform' metadata (expected an object with "
            f"'type' one of {TARGET_TRANSFORM_TYPES!r}, 'mean' and 'std', got {node!r})."
        )
    try:
        transform = StandardizeTarget.from_config(node)
    except (DataError, TypeError, ValueError) as exc:
        raise PersistenceError(f"Cannot {what} '{path}': invalid 'target_transform' metadata ({exc}).") from exc
    if metadata.get("task") != "regression":
        raise PersistenceError(
            f"Cannot {what} '{path}': a 'target_transform' is only valid on a task='regression' artifact, "
            f"this one declares task={metadata.get('task')!r}."
        )
    return transform


def load_target_transform(path: str) -> "StandardizeTarget | None":
    """Reconstruct the `StandardizeTarget` saved alongside a regression model at `path`, or `None` (Milestone 116).

    Mirrors `load_preprocessing()`/`load_classes()`: reads only the
    `"target_transform"` metadata entry a `save_model(..., target_transform=...)` call
    wrote, independent of `load_model()`. `None` is the ordinary answer for every
    artifact saved without one (all files saved before Milestone 116 included) -- the
    model's output is already in native units.

    Raises `PersistenceError` for a corrupt file, an unsupported format version, and
    for metadata that would make the model's output be misread: a version-3 file whose
    entry is missing, a version-2 file that has one, a malformed or non-finite entry,
    or one on a non-regression artifact.
    """
    metadata, _ = read_archive(path, kind="model")
    if not isinstance(metadata, dict):
        raise PersistenceError(f"Cannot load target transform from '{path}': metadata is not a JSON object.")
    version = metadata.get("forge_format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise PersistenceError(
            f"Cannot load target transform from '{path}': unsupported format version {version!r} "
            f"(this build of Forge supports version {_supported_versions_text()})."
        )
    return _target_transform_from_metadata(metadata, path, "load target transform from")


@dataclass(frozen=True)
class ModelSummary:
    """Identifies a saved module tree without exposing raw serialization internals (Milestone 85).

    `type` is the root module's registered type name -- the same string
    `save_model()` writes to `root["type"]` (`spec_for_class(...).type_name`),
    e.g. `"Sequential"`, `"Linear"`, or a custom `register_module()` name.
    `module_types` lists every module type in the tree, root first, in the
    same depth-first order `_build_save_node` walks it -- e.g. `("Sequential",
    "Conv2d", "ReLU", "MaxPool2d", "Linear")` -- letting a caller recognize an
    architecture's shape without a full per-parameter dump (see `forge model
    inspect` for that level of detail). `parameter_count` is the total number
    of learnable parameter elements across the whole tree.
    """

    type: str
    module_types: "tuple[str, ...]"
    parameter_count: int


@dataclass(frozen=True)
class PreprocessingInfo:
    """The preprocessing pipeline persisted alongside a model, reconstructed read-only (Milestone 85).

    `description` is a human-readable rendering (e.g. `"Resize(size=(64,
    64)) -> Normalize(mean=0.0, std=255.0)"`) built from each transform's own
    `__repr__` -- a `Compose`'s steps are joined with `" -> "` rather than
    shown as `Compose([...])`, matching how a caller actually thinks about a
    pipeline. `transform` is the same reconstructed `forge.data.transforms.
    Transform` instance `load_preprocessing()` returns, for a caller who
    wants to apply it directly rather than just read about it.
    """

    description: str
    transform: Any


@dataclass(frozen=True)
class InputSchema:
    """The portable structural input contract for a fixed-width numeric artifact (Milestone 101).

    Answers exactly one question: *how many values does one input row need?*
    `feature_count` is the width `predict_tensor_artifact()`/
    `predict_tabular_classification_artifact()` require along an input's last
    axis, for either a single unbatched sample (`(feature_count,)`) or a
    batch (`(N, feature_count)`) -- the same two shapes `predict()`/`Linear`
    already accept (see **Batch dimension** in `docs/architecture/
    persistence.md`'s own Milestone 101 section).

    **`feature_names` (Milestone 119): the identity half of the contract.**
    `feature_names` is `None` -- the case for every artifact saved before Milestone 119,
    and for one trained from plain arrays -- or a tuple of exactly `feature_count` distinct
    names, `feature_names[i]` being what input column `i` *is*, in the order the model
    consumes them and before any persisted `preprocessing`. It is never derived or
    invented: it exists only when `save_model(..., feature_names=...)` recorded it
    (`train_tabular_*(..., feature_names=...)`, or a CSV header read with
    `forge.data.load_csv(..., return_feature_names=True)`). Named input
    (`predict(..., feature_names=...)`) is then matched to it by name; unnamed input
    (a bare NumPy array) carries no names and is checked for width only, exactly as ever.

    **What `feature_count` alone is not.** A *structural* contract only -- it says
    nothing about which value belongs in which position. A same-length input
    whose columns have been reordered (e.g. swapping `Glucose` and
    `Pregnancies` in `examples/tabular_diabetes`) passes it and still reaches
    the model. For an artifact with no `feature_names` that remains true, and
    for an unnamed array it always will: an array has no column names to
    check. See `docs/architecture/persistence.md`'s **Portable input-contract
    validation** section for the full semantic-honesty discussion.
    """

    feature_count: int
    feature_names: "tuple[str, ...] | None" = None


def _leading_linear_in_features(node: "dict | None") -> "int | None":
    """Find the `in_features` of the `Linear` layer that actually consumes a
    saved model's raw input, reading only already-persisted architecture
    metadata (Milestone 101) -- no model reconstruction, no format change.

    Descends through `"Sequential"` wrapper nodes only (following their
    first child, `"0"`, in construction order -- the same key
    `nn.Sequential`'s own registration already uses) to reach the module that
    actually receives the model's input; every real fixed-width numeric
    workload in this repo (`examples/regression`, `examples/
    tabular_diabetes`, `examples/tabular_classification`) is exactly this
    shape: a `Sequential` of `Linear`/`ReLU` layers, input-first. Deliberately
    does **not** descend into or search any other container/module type, and
    does not search past the first child -- an architecture whose raw input
    is not immediately consumed by a `Linear` (e.g. a CNN, where the first
    `Linear` is a classifier head deep after `Conv2d`/`Flatten` layers) must
    not be misidentified as a fixed-width-vector-input model, so this
    returns `None` for any shape it cannot resolve unambiguously -- no
    guessing, matching this milestone's semantic-honesty requirement.
    """
    current = node
    while isinstance(current, dict):
        node_type = current.get("type")
        if node_type == "Linear":
            in_features = current.get("config", {}).get("in_features")
            return int(in_features) if isinstance(in_features, int) and not isinstance(in_features, bool) else None
        if node_type != "Sequential":
            return None
        children = current.get("children")
        if not isinstance(children, dict) or "0" not in children:
            return None
        current = children["0"]
    return None


# Milestone 101: the only two tasks whose input is genuinely "an already-
# batched fixed-width numeric feature vector" -- see `predict_tensor_
# artifact()`/`predict_tabular_classification_artifact()`'s own docstrings.
# `"classification"`/`"segmentation"` take an image *file path* (a
# Conv2d-first architecture would make `_leading_linear_in_features()` return
# `None` anyway, but gating on task explicitly avoids ever computing a
# feature-count contract for a workflow whose input isn't a plain numeric
# vector at all); `"sequence"` has no fixed-width input (see
# `docs/architecture/persistence.md`'s **Portable input-contract validation**
# section, Sequence-artifacts sub-section); a legacy artifact with no `task`
# at all gets no contract either -- never invented for an artifact that
# never explicitly declared what kind of workflow it is (mirroring
# `_legacy_infer_workflow()`'s own "no `task=` means no free guess" stance
# for a *different*, but analogous, ambiguity).
_INPUT_SCHEMA_TASKS = ("regression", "tabular_classification")


@dataclass(frozen=True)
class ModelInfo:
    """The structured, stable result of `inspect_model()` (Milestone 85).

    What a developer holding a `.forge` file needs to know before deciding
    how to use it -- without loading model parameters, requiring CUDA, or
    knowing the archive's internal metadata shape. Deliberately does not
    include per-parameter shapes/dtypes or dotted module names: that level of
    serialization detail remains `forge model inspect`'s own CLI-only report
    (`forge/cli/_archive_info.py`), not part of this smaller product-level
    contract -- see `ModelSummary`.
    """

    model: ModelSummary
    preprocessing: "PreprocessingInfo | None"
    classes: "list[str] | None"
    task: "str | None"
    format_version: int
    device: str
    input_schema: "InputSchema | None" = None
    target_transform: "StandardizeTarget | None" = None

    def __str__(self) -> str:
        preprocessing_line = self.preprocessing.description if self.preprocessing is not None else "none"
        classes_line = ", ".join(self.classes) if self.classes else "none"
        task_line = self.task if self.task is not None else "unknown (legacy artifact, saved before Milestone 87)"
        input_line = f"{self.input_schema.feature_count} feature(s)" if self.input_schema is not None else "n/a"
        # Only shown when present, so an artifact without names prints exactly as before Milestone 119.
        names = self.input_schema.feature_names if self.input_schema is not None else None
        names_line = f"Feature names: {', '.join(names)}\n" if names is not None else ""
        # Only shown when present, so an artifact without one prints exactly as before Milestone 116.
        target_line = (
            f"Target transform: standardize {self.target_transform!r} (predictions are returned in native units)\n"
            if self.target_transform is not None else ""
        )
        return (
            f"Model: {self.model.type} ({self.model.parameter_count:,} parameters)\n"
            f"Task: {task_line}\n"
            f"Input: {input_line}\n"
            f"{names_line}"
            f"Input preprocessing: {preprocessing_line}\n"
            f"{target_line}"
            f"Classes: {classes_line}\n"
            f"Artifact format: version {self.format_version} (device={self.device})"
        )


def _feature_names_from_metadata(
    metadata: dict, path: str, what: str, task: "str | None", feature_count: "int | None",
) -> "tuple[str, ...] | None":
    """The `feature_names` an artifact's metadata declares, or `None` -- validated, never guessed (Milestone 119).

    No `"feature_names"` entry is the ordinary artifact (every file saved before
    Milestone 119, and every one trained from unnamed arrays): `None`. An entry that is
    present is only trusted if it could have come from `save_model()`: a well-formed
    name list, on a tabular task, whose length is the width the saved architecture
    itself declares. Anything else is a `PersistenceError` rather than a schema that
    would reorder input against the wrong names.
    """
    raw = metadata.get("feature_names")
    if raw is None:
        return None
    if task not in _INPUT_SCHEMA_TASKS:
        raise PersistenceError(
            f"Cannot {what} '{path}': it has a 'feature_names' entry but task={task!r} (feature names apply "
            f"only to {_INPUT_SCHEMA_TASKS!r} artifacts)."
        )
    if feature_count is None:
        raise PersistenceError(
            f"Cannot {what} '{path}': it has a 'feature_names' entry, but the saved architecture has no "
            "input width to check it against."
        )
    try:
        names = validate_feature_names(raw)
    except DataError as exc:
        raise PersistenceError(f"Cannot {what} '{path}': malformed 'feature_names' metadata ({exc})") from exc
    if len(names) != feature_count:
        raise PersistenceError(
            f"Cannot {what} '{path}': 'feature_names' lists {len(names)} name(s) but the saved model takes "
            f"{feature_count} input feature(s)."
        )
    return names


def _module_types(node: dict) -> "list[str]":
    types = [node.get("type", "?")]
    children = node.get("children")
    if isinstance(children, dict):
        for child in children.values():
            if isinstance(child, dict):
                types.extend(_module_types(child))
    return types


def _count_parameters(node: dict) -> int:
    total = 0
    parameters = node.get("parameters")
    if isinstance(parameters, dict):
        for meta in parameters.values():
            shape = meta.get("shape", []) if isinstance(meta, dict) else []
            count = 1
            for dim in shape:
                count *= int(dim)
            total += count
    children = node.get("children")
    if isinstance(children, dict):
        for child in children.values():
            if isinstance(child, dict):
                total += _count_parameters(child)
    return total


def _describe_preprocessing(transform: Any) -> str:
    from ..data.transforms import Compose

    if isinstance(transform, Compose):
        return " -> ".join(repr(step) for step in transform.transforms)
    return repr(transform)


def inspect_model(path: str) -> ModelInfo:
    """Return a structured, read-only summary of the model artifact at `path` (Milestone 85).

    ```python
    info = forge.inspect_model("model.forge")
    print(info)
    info.model.type              # "Sequential"
    info.preprocessing.description  # "Resize(size=(64, 64)) -> Normalize(mean=0.0, std=255.0)"
    info.classes                 # ["cat", "dog"], or None
    info.task                    # one of forge.serialization.model.TASK_TYPES, or None
    info.input_schema            # InputSchema(feature_count=8, feature_names=(...) or None), or None (M101/M119)
    ```

    Answers "what is this artifact?" -- a question a developer holding just a
    `.forge` file needs to answer *before* choosing which of `forge.
    predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`
    applies, without already knowing Forge's archive format or writing
    `load_model()`/`load_preprocessing()`/`load_classes()` calls by hand. As
    of **Milestone 87**, `info.task` is the authoritative signal for this --
    see `forge.predict_model()`'s own docstring for how it uses it.

    Reads only `metadata.json` via `read_archive()` -- the same primitive
    `load_model()`/`load_preprocessing()`/`load_classes()` themselves use
    internally -- and never reconstructs a live `Module` (no registered
    module types are required, unlike `load_model()`), never requires CUDA
    regardless of the device the artifact was saved for, and never mutates
    `forge.random`'s state. This makes it cheap relative to `load_model()`
    and safe to call before deciding whether/how to load the model at all.
    Reconstructing the preprocessing pipeline (when present) does still go
    through `deserialize_transform()`, so a preprocessing transform type must
    be registered in this process -- the same requirement `load_preprocessing()`
    already has, and a much smaller registry than `load_model()`'s full
    module-type registry.

    Works identically on artifacts saved with or without `preprocessing=`/
    `classes=`/`task=` (Milestones 71/72/87), including files saved before any
    of them existed: `preprocessing`/`classes`/`task` are simply `None` in
    that case, exactly matching `load_preprocessing()`/`load_classes()`'s own
    backward-compatible behavior -- an ordinary, expected outcome, never an
    error. `info.task is None` means exactly "this artifact never declared an
    explicit task" -- it is never guessed from `model.module_types` here (that
    heuristic, when needed at all, lives only in `forge.predict_model()`'s own
    isolated legacy fallback -- see that function's docstring -- never in this
    read-only inspection contract).

    Raises `PersistenceError` for a missing/corrupt file, an unsupported
    format version, or malformed metadata -- the same conditions `load_model()`
    itself raises for these cases -- including a `"task"` value that is
    present but not one of `forge.serialization.model.TASK_TYPES` (a
    malformed/tampered file, since `save_model()` itself never writes
    anything else there).
    """
    metadata, _ = read_archive(path, kind="model")
    if not isinstance(metadata, dict):
        raise PersistenceError(f"Cannot inspect model '{path}': metadata is not a JSON object.")

    version = metadata.get("forge_format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise PersistenceError(
            f"Cannot inspect model '{path}': unsupported format version {version!r} "
            f"(this build of Forge supports version {_supported_versions_text()})."
        )

    device = metadata.get("device")
    if device not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(SUPPORTED_DEVICE_TYPES)
        raise PersistenceError(
            f"Cannot inspect model '{path}': unrecognized recorded device {device!r} "
            f"(expected one of: {supported})."
        )

    root = metadata.get("root")
    if not isinstance(root, dict):
        raise PersistenceError(f"Cannot inspect model '{path}': malformed metadata (missing 'root').")

    model_summary = ModelSummary(
        type=root.get("type", "?"),
        module_types=tuple(_module_types(root)),
        parameter_count=_count_parameters(root),
    )

    preprocessing_node = metadata.get("preprocessing")
    preprocessing_info = None
    if preprocessing_node is not None:
        transform = deserialize_transform(preprocessing_node)
        preprocessing_info = PreprocessingInfo(description=_describe_preprocessing(transform), transform=transform)

    classes = metadata.get("classes")
    if classes is not None and not (isinstance(classes, list) and all(isinstance(c, str) for c in classes)):
        raise PersistenceError(
            f"Cannot inspect model '{path}': malformed 'classes' metadata (expected a list of "
            f"strings, got {classes!r})."
        )

    task = metadata.get("task")
    if task is not None and task not in TASK_TYPES:
        raise PersistenceError(
            f"Cannot inspect model '{path}': malformed 'task' metadata (expected one of "
            f"{TASK_TYPES!r} or null, got {task!r})."
        )

    input_schema = None
    feature_count = _leading_linear_in_features(root) if task in _INPUT_SCHEMA_TASKS else None
    feature_names = _feature_names_from_metadata(metadata, path, "inspect model", task, feature_count)
    if feature_count is not None:
        input_schema = InputSchema(feature_count=feature_count, feature_names=feature_names)

    target_transform = _target_transform_from_metadata(metadata, path, "inspect model")

    return ModelInfo(
        model=model_summary,
        preprocessing=preprocessing_info,
        classes=list(classes) if classes is not None else None,
        task=task,
        format_version=version,
        device=device,
        input_schema=input_schema,
        target_transform=target_transform,
    )


__all__ = [
    "save_model", "load_model", "load_preprocessing", "load_classes", "load_target_transform", "inspect_model",
    "ModelInfo", "ModelSummary", "PreprocessingInfo", "InputSchema", "FORMAT_VERSION",
    "TARGET_TRANSFORM_FORMAT_VERSION", "SUPPORTED_FORMAT_VERSIONS", "SUPPORTED_DEVICE_TYPES", "TASK_TYPES",
]
