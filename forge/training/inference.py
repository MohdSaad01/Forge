"""`predict()`: the standard post-training inference path (Milestone 68).

`Trainer` owns the training/evaluation lifecycle, but both `fit()` and
`evaluate()` require a `Loss` and an `Optimizer` at construction -- neither
exists, or is needed, once a model is trained and you just want to run it on
new data (`forge.load_model()` returns a bare `Module`, nothing else).
Every example that reaches this point (`mnist`, `regression`, `resnet`,
`segmentation`, `autoencoder`, `waveform_classification`) had independently
hand-rolled the identical four-line sequence to do it:

```python
with no_grad():
    prediction = model(x.to(device)).to("cpu").numpy()
```

`predict()` is that sequence, written once: it puts `model` in eval mode
(restoring whatever mode it was in afterward), moves the input to the
model's device (or an explicit `device=`), runs the forward pass inside
`forge.no_grad()`, and returns the result as a CPU `Tensor` -- ready for
`.numpy()`, comparison, or display, with no device/autograd bookkeeping left
to the caller. See `docs/architecture/training-engine.md`'s **Inference**
section.

Two input shapes are supported:

- A single `Tensor` (one batch, or one sample with a leading batch
  dimension) -- returns a single `Tensor`.
- An iterable of batches (a `DataLoader`, or any iterable yielding either a
  bare `Tensor` or a `(features, ...)` tuple, matching Forge's existing
  dataset/batch conventions) -- runs the model over every batch and returns
  the concatenated per-batch outputs as one `Tensor`. Concatenation happens
  on already-materialized host arrays (plain NumPy), never inside the
  differentiable `Tensor`/autograd core -- there is no `Tensor.cat`
  primitive in Forge, and this is a non-differentiable inference-time
  convenience, not a computation any real consumer has ever needed
  differentiable (the same reasoning `docs/architecture/tensor-api.md`
  already documents for `Metric`'s own host-side reductions).

This is deliberately a free function, not a `Trainer` method or a
`Module.predict()` -- `Trainer` still requires a `Loss`/`Optimizer` it has no
reason to own for pure inference, and attaching `.predict()` directly to
`Module` would blur the line `docs/architecture/modules.md` draws between
"what a Module computes" and "how it's orchestrated," the same distinction
`Trainer` itself already exists to preserve.

`predict_artifact()` (Milestone 82) is the next layer up: a portable `.forge`
artifact already carries everything `predict()` + `interpret_classification()`
need (model, and optionally preprocessing/classes -- Milestones 71/72), but a
developer holding just the file still had to know to call `load_model()`,
`load_preprocessing()`, `load_classes()`, decode the image the same way
`ImageFolder` does, apply the preprocessing, batch it, call `predict()`, and
call `interpret_classification()`, in that order -- exactly the internal
framework knowledge `docs/product/vision.md`'s portable-artifact workflow is
supposed to hide. `predict_artifact(path, image)` is that entire sequence,
written once, for the one artifact shape Forge can currently fully describe
end-to-end: an image-classification model saved with `preprocessing=`.

`predict_tensor_artifact()` (Milestone 83) is the equivalent one call for a
materially different artifact shape: a regression (or any other Tensor-in,
Tensor-out) model, whose input is already a plain numeric array rather than
an image file, and whose output is a number, not a class label. It is
deliberately a separate function rather than a second branch inside
`predict_artifact()` -- see that function's own docstring's **Scope**
paragraph for why forcing classification-specific semantics (mandatory
preprocessing, file decoding, class interpretation) onto a numeric input
would misrepresent what a regression artifact actually needs.

`predict_image_artifact()` (Milestone 84) is the third artifact shape: an
image-to-image, dense-prediction model (`examples/segmentation`), whose
input is an image file (like `predict_artifact()`) but whose output is
itself another image-shaped Tensor (a per-pixel mask), not a class label or
a scalar. See that function's own docstring for the one small, already-
established output conversion it applies and why it is not a generalized
"any image-to-image model" function.

`predict_model()` (Milestone 86) is the single entry point over all three:
a developer holding a `.forge` file no longer has to already know which of
the three functions above applies -- `predict_model(path, input_data)`
reads the artifact's own persisted metadata (via `forge.inspect_model()`,
Milestone 85) to pick the one supported workflow it describes, then
delegates unchanged to the matching function above. See its own docstring
for exactly which persisted signal decides this, and why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .. import random as forge_random
from ..autograd import no_grad
from ..backend.device import Device
from ..exceptions import DataError, PersistenceError, TrainerError
from ..nn.module import Module
from ..tensor.tensor import Tensor


def _resolve_device(model: Module, device: "str | Device | None") -> Device:
    if device is not None:
        return Device.parse(device)
    model_device = model.device
    if model_device is not None:
        return model_device
    # A model with no Parameters (Module.device is None, e.g. a bare
    # activation-only module) has no device to infer from -- default to
    # "cpu", matching Trainer's own exemption for this same edge case
    # (`Trainer._check_model_device`).
    return Device.parse("cpu")


def _extract_input(batch: Any) -> Tensor:
    x = batch[0] if isinstance(batch, tuple) else batch
    if not isinstance(x, Tensor):
        raise DataError(
            f"predict() requires each input to be a Tensor (or a tuple whose first "
            f"element is a Tensor), got {type(x).__name__}."
        )
    return x


def predict(
    model: Module,
    inputs: "Tensor | Iterable[Any]",
    device: "str | Device | None" = None,
) -> Tensor:
    """Run `model` over `inputs` for inference and return the result as a CPU Tensor.

    ```python
    model = forge.load_model("model.forge", device="cuda")
    prediction = forge.predict(model, x)              # x: a single Tensor
    predictions = forge.predict(model, test_loader)    # a DataLoader
    ```

    `device` defaults to `model.device` (the device the model already lives
    on -- inferred, never assumed); pass it explicitly to also move a bare,
    Parameter-less model's input somewhere specific. Raises `TrainerError`
    if `model` is not a `forge.nn.Module`, and `DataError` if a batch (or an
    iterable of them) does not contain a `Tensor` where one is required, or
    if the iterable yields no batches at all.

    Restores whatever training/eval mode `model` was in before the call,
    the same way `Trainer.evaluate()` does.
    """
    if not isinstance(model, Module):
        raise TrainerError(f"predict() requires a forge.nn.Module model, got {type(model).__name__}.")

    target_device = _resolve_device(model, device)
    was_training = model.training
    model.eval()
    try:
        with no_grad():
            if isinstance(inputs, Tensor):
                output = model(inputs.to(target_device))
                return output.to("cpu")

            chunks: "list[np.ndarray]" = []
            out_dtype = None
            for batch in inputs:
                x = _extract_input(batch).to(target_device)
                output = model(x)
                out_dtype = output.dtype
                chunks.append(output.to("cpu").numpy())
    finally:
        model.train(was_training)

    if not chunks:
        raise DataError("predict() received an empty iterable of inputs (no batches to run).")
    combined = np.concatenate(chunks, axis=0)
    return Tensor(combined, dtype=out_dtype, device="cpu")


def save_and_verify(
    model: Module,
    path: str,
    sample: Tensor,
    device: "str | Device | None" = None,
    preprocessing: "Any | None" = None,
    classes: "list[str] | None" = None,
    task: "str | None" = None,
    atol: float = 1e-5,
) -> Module:
    """Save `model` to `path`, then immediately prove it is a genuinely portable
    artifact by reloading it fresh and confirming its prediction on `sample`
    matches the model that was just saved (Milestone 78).

    ```python
    reloaded = save_and_verify(
        trainer.model, str(model_path), query_x.reshape(1, *query_x.shape),
        preprocessing=build_transform(), classes=full_dataset.classes,
    )
    result = interpret_classification(predict(reloaded, new_image_batch), reloaded_classes)
    ```

    Every Trainer-based Forge example (`mnist`, `regression`, `resnet`,
    `autoencoder`, `segmentation`, `waveform_classification`,
    `image_folder_classification`) independently hand-wrote the identical
    "`save_model()` -> `load_model()` -> `predict()` twice -> `numpy.allclose(...,
    atol=1e-5)` -> `assert`" sequence to prove exactly this property -- the
    literal "portable artifact" half of `docs/product/vision.md`'s workflow,
    duplicated across all ten examples' `train.py` scripts (the three
    stepwise-recurrence examples hand-write an equivalent check against
    `model.step()` instead -- see the **Scope** paragraph below). This
    function is that sequence, written once: `save_model(model, path,
    preprocessing=preprocessing, classes=classes)` (Milestones 71/72's
    unmodified persistence call), then `load_model(path, device=...)` +
    `predict()` on both the pre-save model and the freshly reloaded one,
    compared with `numpy.allclose(..., atol=atol)`.

    `device` defaults to `model.device` (the device `model`/`sample` already
    live on, matching `predict()`'s own default) -- the reloaded model is
    restored onto the same device `save_model()` recorded, mirroring
    `load_model()`'s own default policy exactly. `sample` must already be
    batched (a leading batch dimension) the same way any `predict()` input
    is -- this function does no reshaping of its own.

    Raises `forge.DataError` if `sample` is not a `Tensor`, and
    `forge.PersistenceError` if the reloaded model's prediction diverges from
    the pre-save prediction by more than `atol` -- the exact condition every
    example's own hand-written `assert` used to catch, now a real, catchable
    error rather than an `AssertionError` a caller could silently lose under
    `python -O`.

    Returns the freshly **reloaded** `Module` (not `model` itself) -- ready
    for immediate further use (e.g. an interpretation or reconstruction
    demo), so a caller's next step genuinely exercises the file on disk, not
    lingering in-memory state from training.

    `task` (Milestone 87) is passed straight through to `save_model()` --
    optionally declaring which of `"classification"`/`"regression"`/
    `"segmentation"` this artifact represents, so `forge.predict_model()` can
    dispatch to it reliably. See `save_model()`'s own docstring for the exact
    vocabulary and its interaction with `classes`.

    **Scope.** Deliberately narrow: this only composes `save_model()` +
    `load_model()` + `predict()`, so it covers exactly `predict()`'s own
    calling convention (`model(x)` on a single batched `Tensor`) -- it does
    not touch `Trainer.save_checkpoint()`/`TrainingSession.save_checkpoint()`
    (checkpointing is a separate, resumable-training concern, orthogonal to
    "is this inference artifact portable"; callers compose the two by calling
    both, exactly as every retrofitted example does), and it does not cover
    the stepwise-recurrence sequence models (`char_rnn`/`word_rnn`/
    `long_range_recall`), which verify via `model.step(x, state)`, a
    different calling convention `predict()` was never built for (see
    `predict()`'s own docstring) -- those three examples keep their existing,
    independent hand-written check rather than being forced into a shape
    that doesn't fit them.
    """
    if not isinstance(model, Module):
        raise TrainerError(f"save_and_verify() requires a forge.nn.Module model, got {type(model).__name__}.")
    if not isinstance(sample, Tensor):
        raise DataError(f"save_and_verify() requires sample to be a Tensor, got {type(sample).__name__}.")

    from ..serialization.model import load_model as _load_model, save_model as _save_model

    resolved_device = Device.parse(device) if device is not None else None
    pre_save = predict(model, sample, device=resolved_device).numpy()

    _save_model(model, path, preprocessing=preprocessing, classes=classes, task=task)
    reloaded = _load_model(path, device=resolved_device.type if resolved_device is not None else None)
    post_load = predict(reloaded, sample).numpy()

    if not np.allclose(pre_save, post_load, atol=atol):
        max_diff = float(np.max(np.abs(pre_save - post_load)))
        raise PersistenceError(
            f"save_and_verify(): the model reloaded from '{path}' produced a prediction "
            f"that differs from the model that was just saved (max abs diff {max_diff:.6g} "
            f"exceeds atol={atol})."
        )
    return reloaded


def predict_artifact(
    path: str,
    image: "str | os.PathLike",
    *,
    device: "str | Device | None" = None,
) -> "ClassificationPrediction | int":
    """Classify one image file with a portable `.forge` artifact, in one call (Milestone 82).

    ```python
    result = forge.predict_artifact("model.forge", "new_photo.jpg")
    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence:.1%}")
    ```

    `examples/image_folder_classification/infer.py` and `forge model predict`
    (`forge/cli/model.py`) each independently hand-wrote the identical
    "`load_preprocessing()` -> `load_model()` -> decode the image via
    `ImageFolder._load_image()` -> apply the preprocessing -> add a batch
    dimension -> `predict()` -> `load_classes()` -> `interpret_classification()`"
    sequence -- exactly the internal framework knowledge a developer holding a
    `.forge` file should never need to reconstruct by hand
    (`docs/product/vision.md`). `predict_artifact()` is that sequence, written
    once; both call sites above now delegate to it instead of duplicating it.

    `image` must be a path (`str` or `os.PathLike`) to one image file on disk
    -- the one input shape a saved artifact's persisted preprocessing can
    already fully describe end-to-end (**Milestone 82's** scope is
    image-classification artifacts specifically, not a generic input/artifact
    runtime; see this function's module docstring). Anything else raises
    `forge.DataError` rather than failing deep inside image decoding.

    **Preprocessing is mandatory.** `path` must have been saved with
    `forge.save_model(..., preprocessing=...)` -- there is no way to prepare
    an arbitrary new image for the model otherwise, and silently skipping
    preprocessing would be exactly the hidden-assumption failure mode
    Milestone 71 closed. A `path` saved with no preprocessing raises
    `forge.PersistenceError` naming the missing configuration.

    **Classes are optional, exactly like `forge model predict`.** When `path`
    was also saved with `forge.save_model(..., classes=...)`, the raw
    prediction is turned into a `ClassificationPrediction` via
    `interpret_classification()` -- `.label`/`.index`/`.confidence`. When no
    class vocabulary was saved, this returns the raw predicted class index as
    a plain `int` instead: a classification model with no name for its
    outputs (`load_classes()` returning `None`) is a real, valid artifact
    state (see `load_classes()`'s own docstring) -- fabricating a placeholder
    label for it would be exactly the "pretend it's a classification result"
    failure mode this milestone's brief warns against, so this deliberately
    returns *less* structured information rather than a dishonest one.

    `device` defaults to the device recorded in the archive (`load_model()`'s
    own default) -- pass `device="cpu"`/`device="cuda"` to override, exactly
    as `load_model()` itself accepts.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact (the same conditions `load_model()`/`load_preprocessing()`/
    `load_classes()` already raise), and `forge.DataError` if `image` does
    not point to a readable image file -- both by composing the existing
    lower-level functions' own error handling, not by re-implementing it.

    **Scope.** Composes exactly `load_model()` + `load_preprocessing()` +
    `load_classes()` + `ImageFolder._load_image()` + `predict()` +
    `interpret_classification()`, each called unchanged. No new artifact
    format, input abstraction, or model-serving machinery is introduced --
    see this module's own docstring for what Milestone 82 deliberately does
    not build.
    """
    if not isinstance(image, (str, os.PathLike)):
        raise DataError(
            f"predict_artifact() requires image to be a file path (str or os.PathLike), "
            f"got {type(image).__name__}."
        )

    from ..data.image_folder import ImageFolder
    from ..serialization.model import load_classes as _load_classes
    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    preprocessing = _load_preprocessing(path)
    if preprocessing is None:
        raise PersistenceError(
            f"'{path}' was saved with no preprocessing configuration (see "
            "forge.save_model(..., preprocessing=...)) -- predict_artifact() has no automatic "
            "way to prepare the input image for this model."
        )

    model = _load_model(path, device=device.type if isinstance(device, Device) else device)

    raw = ImageFolder._load_image(Path(image))
    prepared = preprocessing(raw)
    batch = prepared.reshape(1, *prepared.shape)
    output = predict(model, batch)

    classes = _load_classes(path)
    if classes is not None:
        return interpret_classification(output, classes)[0]
    return int(np.argmax(output.numpy(), axis=1)[0])


def predict_tensor_artifact(
    path: str,
    input_data: "Tensor | np.ndarray | Sequence[Any]",
    *,
    device: "str | Device | None" = None,
) -> Tensor:
    """Run one already-batched numeric input through a portable `.forge` artifact, in one call (Milestone 83).

    ```python
    prediction = forge.predict_tensor_artifact("price_model.forge", input_batch)
    ```

    The non-classification counterpart to `predict_artifact()` (Milestone
    82): `examples/regression/train.py` needs the same "a developer holding
    just the `.forge` file shouldn't have to know `load_model()`/
    `load_preprocessing()`/`predict()` exist" guarantee, but its input is a
    plain numeric feature vector, not an image file -- there is no file to
    decode, no class vocabulary to interpret the output against, and
    (unlike a decoded image, which is always exactly one sample) no single
    canonical "add a batch dimension for me" convention to apply on the
    caller's behalf. See `predict_artifact()`'s own docstring for why that
    function is not generalized to also cover this shape instead of adding
    this one.

    `input_data` must already be batched exactly the way a direct
    `forge.predict(model, input_data)` call requires -- a `Tensor`, or a
    NumPy array / nested list or tuple of numbers convertible to one via
    `Tensor(input_data)`, with a leading batch dimension matching what the
    saved model's `forward()` expects. Anything else raises
    `forge.DataError` before any file I/O.

    **Preprocessing is optional here** (unlike `predict_artifact()`, where a
    missing one is an error): a saved artifact may or may not have been
    given `preprocessing=` at save time. When present, it is applied to
    `input_data` before the forward pass -- e.g. `examples/regression/
    train.py` saves the fitted `Normalize(mean=..., std=...)`
    feature-standardization transform this way, so a brand-new raw feature
    vector is standardized identically to how training data was; when
    absent, `input_data` is passed to the model exactly as given.

    No class vocabulary is ever consulted here -- there is no `classes=`
    equivalent for a numeric result, and fabricating one would be exactly
    the "pretend every model is a classification model" failure mode this
    function exists to avoid. Returns `predict()`'s own raw output `Tensor`
    (CPU-resident, same shape as `input_data`'s batch dimension) with no
    further interpretation -- the honest final result for a model with no
    class-label concept.

    `device` defaults to the device recorded in the saved artifact
    (`load_model()`'s own default) -- pass `device="cpu"`/`device="cuda"` to
    override, exactly as `load_model()`/`predict_artifact()` themselves
    accept.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact or an unreconstructable `"preprocessing"` entry (the same
    conditions `load_model()`/`load_preprocessing()` already raise).

    **Scope.** Composes exactly `load_model()` + `load_preprocessing()` +
    `predict()`, each called unchanged -- no new artifact format, generic
    input abstraction, or task-detection machinery (see this module's own
    docstring, and `docs/architecture/training-engine.md`'s own
    Milestone-83 section).
    """
    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    if isinstance(input_data, Tensor):
        prepared = input_data
    elif isinstance(input_data, (np.ndarray, list, tuple)):
        prepared = Tensor(input_data)
    else:
        raise DataError(
            f"predict_tensor_artifact() requires input_data to be a Tensor, NumPy array, or "
            f"list/tuple of numbers, got {type(input_data).__name__}."
        )

    preprocessing = _load_preprocessing(path)
    if preprocessing is not None:
        prepared = preprocessing(prepared)

    model = _load_model(path, device=device.type if isinstance(device, Device) else device)
    return predict(model, prepared)


def predict_image_artifact(
    path: str,
    image: "str | os.PathLike",
    *,
    device: "str | Device | None" = None,
    threshold: float = 0.5,
) -> Tensor:
    """Run one new image file through a portable image-to-image `.forge`
    artifact and get back a savable prediction Tensor, in one call (Milestone 84).

    ```python
    prediction = forge.predict_image_artifact("segmentation_model.forge", "new_image.png")
    forge.data.save_image(prediction, "prediction.png")
    ```

    `examples/segmentation/train.py` produces a dense, per-pixel prediction
    (a `(1, H, W)` mask) rather than a class label or a number -- neither
    `predict_artifact()` (Milestone 82, which always ends in a class-vocabulary
    interpretation) nor `predict_tensor_artifact()` (Milestone 83, whose input
    is already a numeric array, not an image file) fits this shape: the input
    is a file that must be decoded like `predict_artifact()`'s, but the output
    is itself another image, not a label. `predict_image_artifact()` is the
    dense-prediction counterpart: it composes `load_model()`,
    `load_preprocessing()`, `ImageFolder._load_image()`, and `predict()`
    exactly like `predict_artifact()` does, then applies the one conversion
    `examples/segmentation/train.py`/`metrics.py` already define for turning
    this architecture's raw, unbounded per-pixel output into an actual
    predicted mask: threshold at `threshold` (default `0.5`, the same
    `_THRESHOLD` both of those modules use) to obtain a `{0, 1}`-valued
    result. This is not a generic "any image-to-image model" function --
    it reuses one specific, already-established output convention; a future
    image-output workload with different result semantics (e.g. an
    autoencoder's unbounded reconstruction, which needs no thresholding at
    all) would need its own function, following the same task-specific-
    boundary discipline `predict_artifact()`/`predict_tensor_artifact()`
    already established, not a generalization of this one.

    `image` must be a path (`str` or `os.PathLike`) to one image file on disk,
    decoded via `ImageFolder._load_image()` -- the same `(3, H, W)`, raw
    `[0, 255]`-range decode `predict_artifact()` uses. Anything else raises
    `forge.DataError`.

    **Preprocessing is mandatory**, exactly like `predict_artifact()`: `path`
    must have been saved with `forge.save_model(..., preprocessing=...)` --
    there is no way to rescale a freshly decoded `[0, 255]` image into the
    model's trained input range otherwise. A `path` saved with no
    preprocessing raises `forge.PersistenceError` naming the missing
    configuration.

    `device` defaults to the device recorded in the archive (`load_model()`'s
    own default) -- pass `device="cpu"`/`device="cuda"` to override, exactly
    as `load_model()`/`predict_artifact()` themselves accept.

    Returns a CPU `Tensor` of shape `(1, H, W)` (the model's own per-pixel
    output shape, with the batch dimension dropped) whose values are exactly
    `0.0` or `1.0` -- ready to pass directly to `forge.data.save_image()`.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact or a missing preprocessing configuration, and `forge.DataError`
    if `image` is not a path or does not point to a readable image file.

    **Scope.** Composes exactly `load_model()` + `load_preprocessing()` +
    `ImageFolder._load_image()` + `predict()`, plus the one small,
    already-established threshold conversion described above. No new
    artifact format, input abstraction, or model-serving machinery is
    introduced.
    """
    if not isinstance(image, (str, os.PathLike)):
        raise DataError(
            f"predict_image_artifact() requires image to be a file path (str or os.PathLike), "
            f"got {type(image).__name__}."
        )

    from ..data.image_folder import ImageFolder
    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    preprocessing = _load_preprocessing(path)
    if preprocessing is None:
        raise PersistenceError(
            f"'{path}' was saved with no preprocessing configuration (see "
            "forge.save_model(..., preprocessing=...)) -- predict_image_artifact() has no automatic "
            "way to prepare the input image for this model."
        )

    model = _load_model(path, device=device.type if isinstance(device, Device) else device)

    raw = ImageFolder._load_image(Path(image))
    prepared = preprocessing(raw)
    batch = prepared.reshape(1, *prepared.shape)
    output = predict(model, batch)

    mask = (output.numpy() >= threshold).astype(np.float32)
    return Tensor(mask[0], device="cpu")


def _legacy_infer_workflow(info: "Any") -> str:
    """The pre-Milestone-87 architecture-based guess, kept **only** as an
    isolated fallback for artifacts with no explicit `"task"` metadata.

    This is the exact heuristic Milestone 86 introduced, before Milestone 87
    added explicit task metadata (`save_model(..., task=...)`): never model
    weights, a trial forward pass, or tensor dimensions -- only metadata
    `inspect_model()` (Milestone 85) already exposes.

    - `classes` **present** -> `"classification"`. This is unambiguous:
      `save_model()` (`forge/serialization/model.py`) never populates
      `classes` for anything but a classification model, and every
      classification example in this repo (`mnist`, `image_folder_
      classification`) always saves one.
    - `classes` **absent** -> distinguished by whether the saved architecture
      contains a `"Linear"` layer (`ModelSummary.module_types`, also
      Milestone 85): `examples/regression`'s model is `Linear`-only (no
      `Conv2d`); `examples/segmentation`'s is `Conv2d`-only, fully
      convolutional, with no `Linear` layer at all (a dense per-pixel head
      needs no fixed-size fully-connected reduction) -- see each example's
      own `model.py`. `"Linear"` present -> `"regression"`; `"Linear"` absent
      but `"Conv2d"` present -> `"segmentation"`.
    - Neither `"Linear"` nor `"Conv2d"` present (and no `classes`) -> genuinely
      undetermined; raises `forge.PersistenceError` rather than guessing.

    **Known, permanent limitation of this legacy fallback specifically**
    (documented since Milestone 86, `docs/development/
    m86-unified-artifact-prediction.md`): a classification model saved with
    `classes=None` (a real, valid state -- see `predict_artifact()`'s own
    docstring) is architecturally indistinguishable from a regression model
    when neither carries a `"task"` -- both are `Linear`-terminated with no
    saved `classes`. This function still guesses `"regression"` for that
    shape, exactly as Milestone 86 did, because a genuinely legacy file (no
    `task` key at all, pre-Milestone-87) gives `predict_model()` no other
    signal to work with, and real `examples/regression` artifacts saved
    before Milestone 87 -- architecturally identical, and needing to keep
    working -- are indistinguishable from that ambiguous case by
    architecture alone. **This is why Milestone 87 exists**: any artifact
    saved with an explicit `task=` (see `_determine_workflow()` below) never
    reaches this function at all, so a modern classification artifact with no
    `classes` is identified correctly regardless of this limitation. Only a
    genuinely legacy file, or one saved with `task=` deliberately omitted,
    can still hit this documented gap.
    """
    if info.classes is not None:
        return "classification"

    module_types = info.model.module_types
    if "Linear" in module_types:
        return "regression"
    if "Conv2d" in module_types:
        return "segmentation"

    raise PersistenceError(
        "predict_model() could not determine a supported prediction workflow for this "
        f"artifact: it has no explicit task metadata (see forge.save_model(..., task=...)), "
        f"no class vocabulary was saved (ruling out the legacy classification signal), and its "
        f"architecture ({', '.join(module_types)}) contains neither a Linear layer "
        "(the legacy regression signal) nor a Conv2d layer (the legacy segmentation signal). "
        "Supported workflows: classification (forge.predict_artifact(), requires classes= at "
        "save time), regression (forge.predict_tensor_artifact()), segmentation "
        "(forge.predict_image_artifact()). Call one of these directly if you already know "
        "which applies, or inspect the artifact first with forge.inspect_model()."
    )


def _determine_workflow(info: "Any") -> str:
    """Pick one of `"classification"`/`"regression"`/`"segmentation"` for a `ModelInfo`, or fail clearly (Milestones 86/87).

    As of **Milestone 87**, `info.task` -- the explicit metadata a caller
    declared via `forge.save_model(..., task=...)` (or `train_and_save()`/
    `save_and_verify()`'s own `task=`) -- is the primary, authoritative
    signal: when present, it is already one of `forge.serialization.model.
    TASK_TYPES` (`inspect_model()` validates this), so it is returned
    directly, with no architecture inspection at all. This is what finally
    closes Milestone 86's documented ambiguity: a classification artifact
    saved with `classes=None` but `task="classification"` is identified
    correctly, because the explicit declaration is used before `classes`/
    module-architecture are ever consulted.

    Only when `info.task is None` -- a genuinely legacy artifact, saved
    before Milestone 87, or one where `task=` was deliberately omitted --
    does this fall back to `_legacy_infer_workflow()`, the exact Milestone 86
    heuristic, unchanged and isolated in its own function (see that
    function's own docstring for what it can and cannot safely tell apart,
    and why that documented limitation cannot be resolved without an
    explicit task declaration).
    """
    if info.task is not None:
        return info.task
    return _legacy_infer_workflow(info)


def predict_model(
    path: str,
    input_data: Any,
    *,
    device: "str | Device | None" = None,
) -> "ClassificationPrediction | int | Tensor":
    """Predict from any supported portable `.forge` artifact, in one call, with no manual workflow choice (Milestones 86/87).

    ```python
    result = forge.predict_model("model.forge", input_data)
    ```

    Before this function, a developer holding a `.forge` file first had to
    call `forge.inspect_model()` (Milestone 85) -- or already know, out of
    band -- whether it was a classification, regression, or segmentation
    artifact, in order to pick the right one of `predict_artifact()`/
    `predict_tensor_artifact()`/`predict_image_artifact()` (Milestones
    82-84). `predict_model()` closes that gap: it inspects `path` itself via
    `inspect_model()`, determines which of the three workflows the artifact's
    own persisted metadata supports (see `_determine_workflow()` above for
    the exact signal and why it is reliable), and delegates to that function
    unchanged -- returning exactly what it would have returned. This function
    adds no new inference logic, artifact format, or input-conversion
    machinery of its own.

    `input_data` is whatever the *selected* workflow's own function expects
    -- a `str`/`os.PathLike` image path for classification or segmentation,
    or a `Tensor`/NumPy array/nested list for regression -- and is validated
    exactly as strictly as calling that function directly would be: an
    incompatible input still raises `forge.DataError` with that function's
    own message (e.g. a `Tensor` given to a classification artifact raises
    the same error `predict_artifact()` itself raises for a non-path
    `image`). `predict_model()` deliberately does not re-validate `input_data`
    itself -- the artifact's workflow, not the input's type, decides
    dispatch (see the module's own Milestone 86 paragraph): input type is
    only ever used, by the delegated function, to check compatibility with
    the workflow the artifact's metadata already selected.

    `device` is passed through unchanged to the selected function --
    defaults to the device recorded in the archive, exactly like
    `predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`
    themselves.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact (the same conditions `inspect_model()` itself raises), and also
    `forge.PersistenceError` when the artifact's metadata does not reliably
    identify one of the three supported workflows (see
    `_determine_workflow()`) -- this is a deliberate refusal to guess, not a
    bug: an unsupported artifact should fail clearly rather than silently
    produce a result under the wrong interpretation.

    **Scope.** Supports exactly the three workflows `predict_artifact()`/
    `predict_tensor_artifact()`/`predict_image_artifact()` already implement
    -- no generic task registry, no model-architecture discovery beyond
    `inspect_model()`'s existing `ModelSummary.module_types`, no automatic
    input conversion between shapes. A future artifact shape none of the
    three functions covers stays unsupported here too, until Forge has one.
    """
    from ..serialization.model import inspect_model as _inspect_model

    info = _inspect_model(path)
    workflow = _determine_workflow(info)

    if workflow == "classification":
        return predict_artifact(path, input_data, device=device)
    if workflow == "regression":
        return predict_tensor_artifact(path, input_data, device=device)
    return predict_image_artifact(path, input_data, device=device)


def generate_sequence(
    model: Module,
    seed: "Sequence[Any]",
    encode: "Callable[[Any], Tensor]",
    decode: "Callable[[int], Any]",
    length: int,
    device: "str | Device | None" = None,
    rng: "np.random.Generator | None" = None,
) -> list:
    """Autoregressively sample `length` tokens from a stepwise sequence model (Milestone 75).

    ```python
    text = generate_sequence(
        model, seed=list("a tensor"),
        encode=lambda ch: Tensor(one_hot(vocab.encode(ch), vocab.size)),
        decode=lambda idx: vocab.decode([idx]),
        length=200,
    )
    ```

    `examples/char_rnn/train.py` and `examples/word_rnn/train.py` each
    independently hand-wrote the identical "prime a hidden state over a seed
    sequence, then repeatedly sample a next token from the model's own
    output distribution and feed it back in" loop -- this is that loop,
    extracted once both examples' `generate()` functions turned out to be
    structurally identical (see `docs/development/m75-sequence-generation.md`).

    `model` must implement the informal stepwise-recurrence protocol every
    Forge sequence example already follows (`RNNCell`/`LSTMCell`-based
    models via `examples/char_rnn/model.py`/`examples/word_rnn/model.py`):

    - `model.init_hidden(batch_size, device=...) -> state` -- a fresh
      initial hidden state.
    - `model.step(x, state) -> (logits, state)` -- one recurrence step for a
      single-timestep, batch-size-1 input `x`, returning per-class logits
      `(1, num_classes)` and the next state.

    This is deliberately duck-typed, not a new base class or `Protocol` --
    the same "no abstraction based on one weak consumer" discipline this
    codebase already applies elsewhere, except here two independent, already
    -existing consumers share the exact same shape, which is what justifies
    extracting it at all.

    `seed` is a non-empty sequence of already-tokenized items (characters,
    word strings, token ids -- whatever `encode`/`decode` agree on). The
    returned list always starts with `list(seed)` followed by `length` newly
    sampled tokens, mirroring both examples' existing "seed text is part of
    the output" behavior. `encode(token) -> Tensor` builds one timestep's
    model input from a single token (any leading batch dimension `encode`
    produces is used as-is, matching `model.step`'s own batch-size-1
    convention); `decode(index) -> token` turns one sampled class index back
    into a token of the same kind `seed` holds.

    Sampling draws from the model's own output distribution (a host-side
    softmax over `model.step`'s logits, via `rng.choice`) rather than always
    taking the argmax -- greedy decoding was never what either example did,
    and averaging over a whole distribution is what makes repeated calls
    with different `rng` states produce varied continuations. `rng` defaults
    to `forge.random.default_generator()` (Forge's existing process-global
    generator, e.g. `Metric`/`random_split`'s own default-argument
    convention) when omitted.

    Runs under eval mode and `forge.no_grad()`, restoring whatever
    training/eval mode `model` was in before the call, exactly like
    `predict()`. Raises `TrainerError` if `model` is not a `forge.nn.Module`,
    and `DataError` if `seed` is empty or `length` is negative.
    """
    if not isinstance(model, Module):
        raise TrainerError(f"generate_sequence() requires a forge.nn.Module model, got {type(model).__name__}.")
    seed_tokens = list(seed)
    if not seed_tokens:
        raise DataError("generate_sequence() requires a non-empty seed sequence.")
    if length < 0:
        raise DataError(f"generate_sequence() requires length >= 0, got {length}.")

    target_device = _resolve_device(model, device)
    sample_rng = rng if rng is not None else forge_random.default_generator()

    was_training = model.training
    model.eval()
    try:
        with no_grad():
            state = model.init_hidden(1, device=target_device)
            for token in seed_tokens[:-1]:
                x_t = encode(token).to(target_device)
                _, state = model.step(x_t, state)

            generated = list(seed_tokens)
            current = seed_tokens[-1]
            for _ in range(length):
                x_t = encode(current).to(target_device)
                logits, state = model.step(x_t, state)
                probs = np.exp(logits.to("cpu").numpy()[0])
                probs = probs / probs.sum()
                next_index = int(sample_rng.choice(probs.shape[0], p=probs))
                current = decode(next_index)
                generated.append(current)
    finally:
        model.train(was_training)

    return generated


@dataclass(frozen=True)
class ClassificationPrediction:
    """One row of `interpret_classification()`'s output: a human-readable result.

    `label` is `classes[index]`; `confidence` is that class's softmax
    probability under the row's raw output values (see
    `interpret_classification()`'s docstring for why this is a legitimate,
    not merely convenient, use of softmax).
    """

    label: str
    index: int
    confidence: float


def interpret_classification(output: Tensor, classes: "Sequence[str]") -> "list[ClassificationPrediction]":
    """Turn `predict()`'s raw per-class output into human-readable `ClassificationPrediction`s.

    ```python
    output = forge.predict(model, batch)                 # Tensor(batch, num_classes) -- raw logits
    results = interpret_classification(output, classes)  # one ClassificationPrediction per row
    print(f"Predicted class: {results[0].label}")
    print(f"Confidence: {results[0].confidence:.1%}")
    ```

    This is Milestone 72's "tensor output" -> "useful prediction" step:
    `predict()` deliberately stops at a raw `Tensor` (it has no way to know
    what the output's indices *mean*), and `classes[i]` -- `forge.
    save_model(..., classes=...)`'s own index convention (**Milestone 72**
    in `docs/architecture/persistence.md`) -- is what supplies that meaning.

    `output` must be 2-D, `(batch_size, num_classes)`, with `num_classes ==
    len(classes)` -- exactly the shape `nn.CrossEntropyLoss` itself requires
    of its `logits` argument (`forge/nn/loss.py`), since this function
    interprets `output` the same way: as unnormalized per-class scores
    (logits), not already-normalized probabilities. `confidence` is
    therefore the row's softmax probability of its predicted class --
    numerically stable (max-subtracted before `exp`, mirroring `Cross
    EntropyLoss`'s own log-sum-exp trick), computed here in plain host-side
    NumPy rather than through the `Tensor`/autograd graph (this runs after
    `no_grad()` inference, on data that is about to be printed, not
    differentiated -- the same non-differentiable-host-reduction precedent
    `Metric`/`predict()`'s own batch-concatenation already use). This is a
    legitimate probability reading of `output`, not an unjustified
    confidence claim: every classification model in Forge is trained with
    `CrossEntropyLoss`, which is defined in terms of `log_softmax(logits)`
    -- softmax is the same transform its own training objective already
    assumes.

    Raises `TrainerError` if `output` is not 2-D, if `classes` is empty, or
    if `output.shape[1] != len(classes)` (**"output dimension inconsistent
    with class count"**) -- this is the one place that check can honestly be
    made: at save time, `save_model()` never introspects a model's
    architecture to learn its output width (see that function's own
    docstring), but here `output` is already the model's *actual* produced
    shape, so a mismatch is unambiguous.
    """
    classes = list(classes)
    if not classes:
        raise TrainerError("interpret_classification() requires a non-empty classes list.")
    if output.ndim != 2:
        raise TrainerError(
            f"interpret_classification() expects a 2-D (batch_size, num_classes) output, "
            f"got shape {output.shape}."
        )
    num_classes = output.shape[1]
    if num_classes != len(classes):
        raise TrainerError(
            f"interpret_classification() output dimension inconsistent with class count: "
            f"output has {num_classes} class score(s) but {len(classes)} class label(s) "
            f"were given ({classes!r})."
        )

    array = output.to("cpu").numpy()
    shifted = array - array.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predicted_indices = np.argmax(array, axis=1)

    return [
        ClassificationPrediction(
            label=classes[int(idx)],
            index=int(idx),
            confidence=float(probabilities[row, idx]),
        )
        for row, idx in enumerate(predicted_indices)
    ]


__all__ = [
    "predict", "save_and_verify", "predict_artifact", "predict_tensor_artifact", "predict_image_artifact",
    "predict_model", "generate_sequence", "interpret_classification", "ClassificationPrediction",
]
