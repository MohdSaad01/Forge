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

`predict_sequence_artifact()` (Milestone 90) is the fourth artifact shape: a
stepwise-recurrence sequence model (`examples/char_rnn`, whose forward pass
is `model.step(x, state)`, not `model.forward(x)`), whose input is a seed
sequence of vocabulary tokens and whose output is an autoregressively
generated continuation, not a single label/number/mask. Composes
`load_model()` + `load_classes()` (the artifact's persisted token vocabulary)
+ `generate_sequence()` (Milestone 75) rather than `predict()` -- `predict()`
calls `model(x)` directly, which raises for a model with no `forward()`; see
that function's own docstring for why this needed a dedicated artifact-level
function rather than reusing `predict_tensor_artifact()`.

`predict_tabular_classification_artifact()` (Milestone 91) is the fifth
artifact shape: a classification model whose input is an already-batched
numeric feature vector, not an image file (`examples/tabular_classification`)
-- discovered as a genuine, reproduced blocker: before this function existed,
`predict_model()` routed every `task="classification"` artifact to
`predict_artifact()` unconditionally, which raises `forge.DataError`
immediately for anything that is not an image file path. A tabular
classification model (numeric features in, a class label out) could already
be fully *trained* with `Trainer`/`DataLoader`/`CrossEntropyLoss`/`classes=`,
but had no supported *inference* path at all. `task="tabular_classification"`
(`forge.serialization.model.TASK_TYPES`) is the new, explicit signal that
routes here instead -- see that function's own docstring, and
`docs/development/m91-tabular-classification-artifact-inference.md`, for the
full investigation.

`predict_model()` (Milestone 86, extended in Milestones 90/91) is the single
entry point over all five: a developer holding a `.forge` file no longer has
to already know which of the five functions above applies --
`predict_model(path, input_data)` reads the artifact's own persisted metadata
(via `forge.inspect_model()`, Milestone 85) to pick the one supported
workflow it describes, then delegates unchanged to the matching function
above. See its own docstring for exactly which persisted signal decides
this, and why.

**Input-contract validation (Milestone 101).** `predict_tensor_artifact()`/
`predict_tabular_classification_artifact()` now reject a structurally
wrong-length numeric input (`forge.DataError`) before preprocessing or the
model ever run, using `forge.inspect_model()`'s new `ModelInfo.input_schema`
(`forge.serialization.InputSchema`) -- derived purely from already-persisted
architecture metadata, no format change. This closes a real, demonstrated
gap: previously a wrong-length input either reached `Normalize`/`Linear` and
failed there with a confusing, internals-revealing shape error, or -- for a
same-length but *reordered* input -- produced no error at all, a
structurally valid but semantically wrong prediction. See each function's
own docstring, and `docs/architecture/persistence.md`'s **Portable
input-contract validation** section, for the explicit, tested limitation
this does *not* solve: Forge's training APIs never capture which column is
which, so same-length feature reordering is undetectable and this milestone
does not pretend otherwise.

**Reusable inference (Milestone 102).** Every function above -- `predict_
artifact()`, `predict_tensor_artifact()`, `predict_image_artifact()`,
`predict_sequence_artifact()`, `predict_tabular_classification_artifact()`,
and `predict_model()` on top of them -- reopens and reconstructs the entire
`.forge` artifact (`inspect_model()`, `load_model()`, `load_preprocessing()`,
`load_classes()`) on every call. That is the right tradeoff for a single,
one-off prediction, but wasteful for an application making many predictions
from the same artifact. `forge.load_predictor(path)` performs that same
loading exactly once and returns an `ArtifactPredictor` -- a small object
retaining the loaded model/preprocessing/classes/`InputSchema` -- whose
`predict()` method reruns only the genuinely per-call work (input
validation, preprocessing, the forward pass), by delegating to the same
artifact-independent core functions (`_classify_image_core()`,
`_predict_tensor_core()`, `_segment_image_core()`, `_generate_sequence_
core()`) the five functions above already call -- see `ArtifactPredictor`'s
own docstring for the full contract. `predict_model()` and the five
task-specific functions are unchanged and remain the right choice for a
single prediction; `load_predictor()` is the natural next step for "load
once, predict many times."

**Evaluation (Milestone 113).** `ArtifactPredictor.evaluate(X, y)` scores the
loaded artifact on held-out labeled data through that same path -- raw input,
the artifact's persisted preprocessing, the model, then metrics
(`forge/training/evaluation.py`) -- so a saved artifact never needs its
training pipeline rebuilt to be measured. It adds no prediction machinery: each
batch goes through `_predict_tensor_core()` (or, for image classifiers, the same
decode-and-preprocess step `_classify_image_core()` uses).
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
from ..exceptions import DataError, PersistenceError, ShapeMismatchError, TrainerError
from ..nn.module import Module
from ..tensor.tensor import Tensor
from .evaluation import (
    ClassificationEvaluationResult,
    RegressionEvaluationResult,
    build_classification_result,
    build_regression_result,
    encode_class_labels,
    encode_regression_targets,
    require_evaluation_classes,
    require_finite_output,
)


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


def _expected_image_channels(model: Module) -> "int | None":
    """The number of input channels an image-consuming `model` expects, or `None` (Milestone 94).

    Walks `model.modules()` (self-first depth-first, the same order
    `save_model()`/`inspect_model()` already walk the tree in) for the first
    `Conv2d` and returns its `in_channels` -- the real, already-fixed
    contract every image-classification/segmentation model in Forge
    declares today (`examples/mnist`: `Conv2d(1, ...)`; `examples/
    image_folder_classification`/`examples/segmentation`: `Conv2d(3, ...)`).
    This is input *adaptation*, not an architecture change: no `Conv2d`
    behavior is touched, only its existing public `in_channels` attribute is
    read, exactly the way `_legacy_infer_workflow()` already reads
    `ModelSummary.module_types` without altering the modules it inspects.

    Returns `None` if `model` contains no `Conv2d` at all -- there is no
    image-channel contract to discover, so callers fall back to the
    pre-Milestone-94 default (`channels=3`, i.e. always decode as RGB)
    rather than guessing.
    """
    from ..nn.conv import Conv2d

    for module in model.modules():
        if isinstance(module, Conv2d):
            return module.in_channels
    return None


def _decode_image_for_model(model: Module, image: "str | os.PathLike") -> Tensor:
    """Decode `image` via `ImageFolder._load_image()`, in the channel representation `model` expects (Milestone 94).

    The shared image/model-boundary adaptation both `predict_artifact()` and
    `predict_image_artifact()` need: `_expected_image_channels()` reads the
    model's own declared contract (its first `Conv2d`'s `in_channels`), and
    the image is decoded directly into that representation -- `1` channel
    (`Image.convert("L")`) or `3` channels (`Image.convert("RGB")`, the
    original, unconditional pre-Milestone-94 behavior) -- *before* the
    artifact's own persisted `preprocessing` pipeline (`Resize`/`Normalize`/
    ...) ever runs, so a channel conversion never happens after a transform
    that already assumes a particular channel count. A model whose first
    `Conv2d` expects a channel count other than `1`/`3` (or a model with no
    `Conv2d` at all -- `_expected_image_channels()` returns `None`) is not a
    contract this function can safely convert for: the former raises
    `DataError` (`ImageFolder._load_image()`'s own message); the latter
    defaults to `channels=3`, preserving the original behavior for any
    image-consuming model shape Milestone 94 did not change.
    """
    from ..data.image_folder import ImageFolder

    channels = _expected_image_channels(model)
    return ImageFolder._load_image(Path(image), channels=channels if channels is not None else 3)


def _validate_feature_count(data: Tensor, expected: "int | None", fn_name: str) -> None:
    """Reject an input whose last-axis width disagrees with an artifact's
    known feature-count contract, before preprocessing or model execution
    (Milestone 101).

    `expected` is `ModelInfo.input_schema.feature_count` -- `None` for any
    artifact this milestone does not build a contract for (no explicit
    `task="regression"`/`task="tabular_classification"`, or an architecture
    `_leading_linear_in_features()` cannot resolve), in which case this is a
    silent no-op: the pre-Milestone-101 behavior (the model's own `Linear`,
    or a `Normalize`/`ReplaceValue` preprocessing step, still raises its own
    shape error, just later and less clearly -- see `docs/architecture/
    persistence.md`'s **Portable input-contract validation** section).

    Only checks a 1-D `(feature_count,)` (single unbatched sample) or 2-D
    `(N, feature_count)` (batch) input, matching `Linear`'s/`predict()`'s own
    two supported input shapes exactly (Milestone 101's **Batch dimension**
    policy) -- any other rank is left to fail with whatever error the model
    or preprocessing itself already raises for it, since this function's
    only job is the specific, common "right rank, wrong width" mistake.

    **What this does not check.** Only the *count* of values, never their
    order or meaning -- a same-width input whose feature columns have been
    swapped passes this check and always will, since Forge's training APIs
    never captured which column is which (see `InputSchema`'s own
    docstring). This function never claims otherwise.
    """
    if expected is None or data.ndim not in (1, 2):
        return
    actual = data.shape[-1]
    if actual != expected:
        raise DataError(f"{fn_name}() expected {expected} input feature(s), received {actual}.")


def _require_preprocessing(path: str, preprocessing: "Any | None", fn_name: str) -> None:
    """Shared "preprocessing is mandatory for this artifact shape" check
    (Milestone 102), extracted from `predict_artifact()`/`predict_image_
    artifact()`'s own identical body so `load_predictor()` can run the same
    check once, at load time, instead of at every `predict()` call.
    """
    if preprocessing is None:
        raise PersistenceError(
            f"'{path}' was saved with no preprocessing configuration (see "
            f"forge.save_model(..., preprocessing=...)) -- {fn_name}() has no automatic "
            "way to prepare the input image for this model."
        )


def _coerce_numeric_input(input_data: Any, fn_name: str) -> Tensor:
    """Shared "turn a Tensor/ndarray/list/tuple into a Tensor, or reject it"
    step every numeric-input artifact function requires (Milestone 102).
    """
    if isinstance(input_data, Tensor):
        return input_data
    if isinstance(input_data, (np.ndarray, list, tuple)):
        return Tensor(input_data)
    raise DataError(
        f"{fn_name}() requires input_data to be a Tensor, NumPy array, or list/tuple of "
        f"numbers, got {type(input_data).__name__}."
    )


def _require_image_path(image: Any, fn_name: str) -> None:
    """Shared "image must be a file path" guard (Milestone 102), extracted
    from `predict_artifact()`/`predict_image_artifact()`'s own identical
    check so `_classify_image_core()`/`_segment_image_core()` -- and
    therefore `ArtifactPredictor.predict()` too -- reject a non-path `image`
    with the same clear `DataError` instead of a raw `TypeError` surfacing
    from deep inside `pathlib`/`ImageFolder._load_image()`.
    """
    if not isinstance(image, (str, os.PathLike)):
        raise DataError(
            f"{fn_name}() requires image to be a file path (str or os.PathLike), "
            f"got {type(image).__name__}."
        )


def _classify_image_core(
    model: Module,
    preprocessing: Any,
    classes: "list[str] | None",
    image: "str | os.PathLike",
    fn_name: str = "predict_artifact",
) -> "ClassificationPrediction | int":
    """The artifact-independent body of `predict_artifact()` (Milestone 102):
    decode/preprocess/predict/interpret against an already-loaded `model`/
    `preprocessing`/`classes`, with no file I/O of its own beyond decoding
    `image`. `predict_artifact()` and `ArtifactPredictor.predict()` both call
    this unchanged, so the two never drift -- see this module's Milestone 102
    paragraph.
    """
    _require_image_path(image, fn_name)
    raw = _decode_image_for_model(model, image)
    prepared = preprocessing(raw)
    batch = prepared.reshape(1, *prepared.shape)
    output = predict(model, batch)
    if classes is not None:
        return interpret_classification(output, classes)[0]
    return int(np.argmax(output.numpy(), axis=1)[0])


def _segment_image_core(
    model: Module,
    preprocessing: Any,
    image: "str | os.PathLike",
    threshold: float,
    fn_name: str = "predict_image_artifact",
) -> Tensor:
    """The artifact-independent body of `predict_image_artifact()` (Milestone
    102) -- see `_classify_image_core()`'s own docstring for why this split
    exists.
    """
    _require_image_path(image, fn_name)
    raw = _decode_image_for_model(model, image)
    prepared = preprocessing(raw)
    batch = prepared.reshape(1, *prepared.shape)
    output = predict(model, batch)
    mask = (output.numpy() >= threshold).astype(np.float32)
    return Tensor(mask[0], device="cpu")


def _predict_tensor_core(
    model: Module,
    preprocessing: "Any | None",
    input_schema: "Any | None",
    input_data: "Tensor | np.ndarray | Sequence[Any]",
    fn_name: str,
    *,
    require_batch: bool = False,
) -> Tensor:
    """The artifact-independent body shared by `predict_tensor_artifact()`
    and `predict_tabular_classification_artifact()` (Milestone 102): coerce
    `input_data`, validate it against `input_schema` (Milestone 101,
    unchanged), apply `preprocessing` if present, then `predict()`.

    `require_batch=True` (Milestone 105) rejects an unbatched 1-D input
    before it ever reaches the model -- needed only by callers that go on to
    call `interpret_classification()`/`np.argmax(..., axis=1)` on the
    result, both of which require a genuine `(batch_size, ...)` output and
    otherwise fail late with an internals-revealing error naming a function
    the caller never called directly (found against a real artifact: a
    `predict_tabular_classification_artifact()`-style single flat row raised
    `interpret_classification() expects a 2-D ... output, got shape (2,)`).
    `predict_tensor_artifact()`'s plain regression path deliberately keeps
    the default `False` -- an unbatched 1-D input there has always been a
    legitimate "single unbatched sample" call, per `_validate_feature_count()`'s
    own docstring, and returns an unbatched 1-D result correctly.
    """
    prepared = _coerce_numeric_input(input_data, fn_name)
    if require_batch and prepared.ndim == 1:
        raise DataError(
            f"{fn_name}() requires a batched input, shape (batch_size, feature_count) -- "
            f"received an unbatched 1-D input with {prepared.shape[0]} feature(s). "
            f"Wrap a single row in an extra list, e.g. [[...]] instead of [...]."
        )
    expected_features = input_schema.feature_count if input_schema is not None else None
    _validate_feature_count(prepared, expected_features, fn_name)
    if preprocessing is not None:
        prepared = preprocessing(prepared)
    _require_finite_input(prepared, fn_name)
    return predict(model, prepared)


def _require_finite_input(prepared: Tensor, fn_name: str) -> None:
    """Reject NaN/Inf numeric input before it reaches the model (Issue I1).

    Checked *after* preprocessing, on exactly what the model would receive,
    so a persisted transform that removes a value is never second-guessed.
    Without this, a NaN row came back as a real-looking
    `ClassificationPrediction(label=..., confidence=nan)` -- an arbitrary
    `argmax` of NaN -- or a NaN regression output, with no error. Training
    already refuses NaN/Inf (`Trainer`), so inference now agrees.
    """
    values = prepared.to("cpu").numpy()
    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        count = int((~np.isfinite(values)).sum())
        raise DataError(
            f"{fn_name}() received {count} non-finite value(s) (NaN/Inf) in its input, after "
            "preprocessing. The model cannot produce a meaningful prediction from them -- "
            "replace them with real values before calling."
        )


def _require_sequence_vocab(path: str, vocab: "list[str] | None", fn_name: str) -> None:
    """Shared "a sequence artifact must have a saved vocabulary" load-time
    check (Milestone 102), extracted so `load_predictor()` runs it once
    instead of at every `predict()` call.
    """
    if vocab is None:
        raise PersistenceError(
            f"'{path}' was saved with no vocabulary (see forge.save_model(..., classes=..., "
            f"task='sequence')) -- {fn_name}() has no way to encode/decode tokens for this model."
        )


def _require_sequence_protocol(path: str, model: Module, fn_name: str) -> None:
    """Shared "a sequence artifact's model must implement `init_hidden`/`step`"
    load-time check (Milestone 102), extracted for the same reason as
    `_require_sequence_vocab()`.
    """
    if not hasattr(model, "init_hidden") or not hasattr(model, "step"):
        raise PersistenceError(
            f"'{path}' does not implement the stepwise-recurrence protocol "
            "(model.init_hidden(batch_size, device=...) and model.step(x, state)) that "
            f"{fn_name}() requires -- see forge.training.generate_sequence()'s own docstring "
            "for the full protocol."
        )


def _generate_sequence_core(
    model: Module,
    vocab: "list[str]",
    seed: "Sequence[str]",
    length: int,
    rng: "np.random.Generator | None",
    fn_name: str,
) -> list:
    """The artifact-independent body of `predict_sequence_artifact()`
    (Milestone 102): validate `seed` against `vocab`, build the one-hot
    `encode`/`decode` closures, and call `generate_sequence()`. Takes an
    already-loaded `model`/`vocab` -- the missing-vocabulary and
    protocol (`init_hidden`/`step`) checks stay in the caller, since those
    are artifact-loading concerns, not per-prediction ones (see
    `load_predictor()`, which runs them once).
    """
    if not isinstance(seed, (list, tuple)) or not seed:
        raise DataError(f"{fn_name}() requires seed to be a non-empty sequence of tokens.")

    token_to_index = {token: i for i, token in enumerate(vocab)}
    unknown = [token for token in seed if token not in token_to_index]
    if unknown:
        raise DataError(f"{fn_name}() seed contains token(s) not in the saved vocabulary: {unknown!r}.")

    vocab_size = len(vocab)

    def _encode(token: str) -> Tensor:
        one_hot = np.zeros((1, vocab_size), dtype=np.float32)
        one_hot[0, token_to_index[token]] = 1.0
        return Tensor(one_hot)

    def _decode(index: int) -> str:
        return vocab[index]

    return generate_sequence(model, seed=list(seed), encode=_encode, decode=_decode, length=length, rng=rng)


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

    **Image channel handling (Milestone 94).** `image` is decoded to match
    `model`'s own declared input contract -- its first `Conv2d`'s
    `in_channels` (`1` or `3`; see `_expected_image_channels()`) -- rather
    than always decoding to RGB. A grayscale (`examples/mnist`, `Conv2d(1,
    ...)`) model converts the input to grayscale (`Image.convert("L")`,
    identity for an already-grayscale source, a standard luminance
    conversion for an RGB/RGBA source); an RGB (`examples/
    image_folder_classification`, `Conv2d(3, ...)`) model converts to RGB
    exactly as every prior milestone's artifacts already did (grayscale
    replicated across channels, RGBA's alpha channel discarded). This
    conversion happens *before* `preprocessing` runs, so a `Resize`/
    `Normalize` step never sees the wrong channel count. A model whose first
    `Conv2d` expects a channel count other than `1`/`3` is not a contract
    this function can convert for and raises `forge.DataError` naming it
    (`ImageFolder._load_image()`'s own validation); a model with no `Conv2d`
    at all falls back to the original, unconditional RGB decode.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact (the same conditions `load_model()`/`load_preprocessing()`/
    `load_classes()` already raise), and `forge.DataError` if `image` does
    not point to a readable image file -- both by composing the existing
    lower-level functions' own error handling, not by re-implementing it.

    **Scope.** Composes exactly `load_model()` + `load_preprocessing()` +
    `load_classes()` + `ImageFolder._load_image()` (via
    `_decode_image_for_model()`, Milestone 94's channel-matching wrapper) +
    `predict()` + `interpret_classification()`, each called unchanged. No new
    artifact format, input abstraction, or model-serving machinery is
    introduced -- see this module's own docstring for what Milestone 82
    deliberately does not build.
    """
    if not isinstance(image, (str, os.PathLike)):
        raise DataError(
            f"predict_artifact() requires image to be a file path (str or os.PathLike), "
            f"got {type(image).__name__}."
        )

    from ..serialization.model import load_classes as _load_classes
    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    preprocessing = _load_preprocessing(path)
    _require_preprocessing(path, preprocessing, "predict_artifact")

    model = _load_model(path, device=device.type if isinstance(device, Device) else device)
    classes = _load_classes(path)
    return _classify_image_core(model, preprocessing, classes, image)


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

    **Input-contract validation (Milestone 101).** When `path` was saved
    with `task="regression"` and its architecture is a `Linear`-first
    `Sequential` (`inspect_model(path).input_schema` is not `None`),
    `input_data`'s last-axis width is checked against
    `input_schema.feature_count` *before* preprocessing or the model ever
    run, raising `forge.DataError` with a clear, actionable message (e.g.
    `"predict_tensor_artifact() expected 8 input feature(s), received 7."`)
    instead of a confusing shape error surfacing deep inside `Normalize` or
    `Linear`. This is a purely *structural* check -- see `InputSchema`'s own
    docstring and `docs/architecture/persistence.md`'s **Portable
    input-contract validation** section for exactly what it can and cannot
    catch (in particular: it cannot detect same-width feature reordering).
    An artifact with no `task=`, or an architecture this milestone cannot
    resolve to a fixed feature count, gets no such check -- this call
    behaves exactly as it did before Milestone 101.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact or an unreconstructable `"preprocessing"` entry (the same
    conditions `load_model()`/`load_preprocessing()` already raise).

    **Scope.** Composes exactly `load_model()` + `load_preprocessing()` +
    `predict()`, each called unchanged -- no new artifact format, generic
    input abstraction, or task-detection machinery (see this module's own
    docstring, and `docs/architecture/training-engine.md`'s own
    Milestone-83 section).
    """
    from ..serialization.model import inspect_model as _inspect_model
    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    checked = _coerce_numeric_input(input_data, "predict_tensor_artifact")

    info = _inspect_model(path)
    preprocessing = _load_preprocessing(path)
    model = _load_model(path, device=device.type if isinstance(device, Device) else device)
    return _predict_tensor_core(model, preprocessing, info.input_schema, checked, "predict_tensor_artifact")


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
    decoded via `ImageFolder._load_image()` in whichever channel
    representation `model`'s own first `Conv2d` expects (Milestone 94 -- see
    `predict_artifact()`'s own docstring for the exact policy; every real
    segmentation example today expects `3` channels, so this is unchanged
    RGB decoding in practice, but the same channel-matching applies here as
    for classification). Anything else raises `forge.DataError`.

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

    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    preprocessing = _load_preprocessing(path)
    _require_preprocessing(path, preprocessing, "predict_image_artifact")

    model = _load_model(path, device=device.type if isinstance(device, Device) else device)
    return _segment_image_core(model, preprocessing, image, threshold)


def predict_sequence_artifact(
    path: str,
    seed: "Sequence[str]",
    length: int,
    *,
    device: "str | Device | None" = None,
    rng: "np.random.Generator | None" = None,
) -> list:
    """Autoregressively generate tokens from a portable stepwise-recurrence `.forge` artifact (Milestone 90).

    ```python
    generated = forge.predict_sequence_artifact("char_rnn_model.forge", seed=list("a tensor"), length=200)
    print("".join(generated))
    ```

    The fourth artifact-shape-specific function alongside `predict_artifact()`/
    `predict_tensor_artifact()`/`predict_image_artifact()` (Milestones 82-84):
    `examples/char_rnn`/`examples/word_rnn`/`examples/long_range_recall` are
    stepwise-recurrence models (`model.init_hidden(batch_size, device=...)` +
    `model.step(x, state) -> (logits, state)`, the same duck-typed protocol
    `generate_sequence()` (Milestone 75) already samples from) rather than
    ordinary `forward(x) -> output` models -- `predict()`/`predict_tensor_
    artifact()` call `model(x)` directly, which raises `ModuleError` for one of
    these (they never implement `forward()`; only `step()`), and `save_and_
    verify()`'s own docstring already documents this as out of its scope for
    exactly this reason. `predict_sequence_artifact()` is the missing artifact-
    level counterpart: it loads the model and its saved vocabulary (`classes=`,
    required at save time for `task="sequence"` -- see `save_model()`'s own
    docstring for why this reuses `classes` rather than a separate `vocab=`),
    builds one-hot `encode`/`decode` closures over that vocabulary, and calls
    `generate_sequence()` unchanged.

    `seed` is a non-empty sequence of tokens already split the way the saved
    vocabulary tokenizes (individual characters for a char-level vocabulary
    like `examples/char_rnn`'s; whatever `classes` actually lists otherwise --
    e.g. whole words for a word-level vocabulary like `examples/word_rnn`'s).
    This function does no tokenization of its own: exactly like
    `predict_tensor_artifact()`'s "input must already be batched" contract,
    the caller is responsible for splitting a raw string into the artifact's
    token unit before calling. Every token in `seed` must already be present
    in the saved vocabulary; an unknown token raises `forge.DataError` naming
    it, rather than failing deep inside a `KeyError` during encoding.

    `length` is the number of *new* tokens to sample after priming on `seed`
    -- passed straight through to `generate_sequence()`, whose own docstring
    documents the sampling behavior (draws from the model's own softmax
    output distribution via `rng`, not greedy argmax) and the returned list's
    shape (`list(seed)` followed by `length` newly generated tokens).

    `device`/`rng` are passed straight through to `load_model()`/
    `generate_sequence()` respectively, with the same defaults.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact, for an artifact saved with no vocabulary (`classes=None` --
    every valid `task="sequence"` artifact has one; see `save_model()`), and
    for a loaded model that does not implement the stepwise-recurrence
    protocol (`init_hidden`/`step`) this function requires -- checked
    explicitly here, before calling `generate_sequence()`, specifically so
    that failure is this clear message rather than a raw `AttributeError`
    from deep inside the sampling loop. Raises `forge.DataError` for an empty
    `seed` or a `seed` token outside the saved vocabulary.

    **Scope.** Composes exactly `load_model()` + `load_classes()` +
    `generate_sequence()`, plus the one-hot encode/decode closures every
    stepwise sequence example in this repo already hand-wrote identically
    (`examples/char_rnn/train.py::_one_hot`, `examples/word_rnn/train.py`'s
    equivalent) -- no new artifact format, tokenization convention, or
    model-serving machinery. A vocabulary-free numeric sequence model (e.g.
    forecasting raw floats rather than a fixed token vocabulary) is not
    covered by this function; it would need its own artifact-shape function,
    following the same task-specific-boundary discipline `predict_artifact()`/
    `predict_tensor_artifact()`/`predict_image_artifact()` already established.
    """
    from ..serialization.model import load_classes as _load_classes
    from ..serialization.model import load_model as _load_model

    if not isinstance(seed, (list, tuple)) or not seed:
        raise DataError("predict_sequence_artifact() requires seed to be a non-empty sequence of tokens.")

    vocab = _load_classes(path)
    _require_sequence_vocab(path, vocab, "predict_sequence_artifact")

    model = _load_model(path, device=device.type if isinstance(device, Device) else device)
    _require_sequence_protocol(path, model, "predict_sequence_artifact")

    return _generate_sequence_core(model, vocab, seed, length, rng, "predict_sequence_artifact")


def predict_tabular_classification_artifact(
    path: str,
    input_data: "Tensor | np.ndarray | Sequence[Any]",
    *,
    device: "str | Device | None" = None,
) -> "list[ClassificationPrediction] | list[int]":
    """Classify an already-batched numeric input with a portable `.forge` artifact, in one call (Milestone 91).

    ```python
    results = forge.predict_tabular_classification_artifact("health_model.forge", raw_features)
    print(f"Prediction: {results[0].label}")
    print(f"Confidence: {results[0].confidence:.1%}")
    ```

    The tabular counterpart to `predict_artifact()` (Milestone 82, image
    classification): `examples/tabular_classification/train.py` needs the
    same "a developer holding just the `.forge` file shouldn't have to know
    `load_model()`/`load_preprocessing()`/`predict()`/`interpret_classification()`
    exist" guarantee, but its input is a plain numeric feature vector, not an
    image file -- there is no file to decode. Discovered as a genuine,
    demonstrated Milestone 91 blocker: before this function existed, a
    tabular classification model saved with `task="classification"` (the
    only classification task that existed) was routed by `predict_model()`
    straight to `predict_artifact()`, which raises `forge.DataError`
    immediately because its `image` argument requires a file path -- there
    was no supported way to predict on a tabular classification artifact at
    all, despite `Trainer`/`DataLoader`/`CrossEntropyLoss`/`classes=`
    already fully supporting *training* one. See
    `docs/development/m91-tabular-classification-artifact-inference.md` for
    the full investigation. This function, plus the new
    `task="tabular_classification"` (`forge.serialization.model.TASK_TYPES`),
    closes that gap the same way `predict_tensor_artifact()`/`task="regression"`
    already cover the equivalent numeric-in/numeric-out shape.

    `input_data` must already be batched exactly like `predict_tensor_
    artifact()`'s own `input_data` -- a `Tensor`, or a NumPy array / nested
    list or tuple of numbers convertible to one via `Tensor(input_data)`,
    with a leading batch dimension matching what the saved model's
    `forward()` expects: a single row is `[[...]]`, not `[...]`, since
    (unlike `predict_tensor_artifact()`'s plain regression path) every
    result here goes through `interpret_classification()`, which requires a
    genuine `(batch_size, num_classes)` output (Milestone 105 -- an
    unbatched 1-D input, previously accepted by mistake, now raises
    `forge.DataError` naming this exact fix instead of failing deep inside
    `interpret_classification()`). Anything else raises `forge.DataError`
    before any file I/O.

    **Returns one result per row, not one result overall** -- unlike
    `predict_artifact()` (always exactly one image in, one prediction out),
    `input_data` here follows `predict_tensor_artifact()`'s "already batched,
    any batch size" convention, so silently keeping only the first row's
    result (as `predict_artifact()` does for its inherently-one-image input)
    would silently discard real caller data for any batch size > 1. A
    single-sample call (the common case, e.g. `input_data` shaped `(1,
    n_features)`) still returns a length-1 list -- index `[0]` for that
    result, exactly as this docstring's own example does.

    **Preprocessing is optional here**, exactly like `predict_tensor_
    artifact()` (and unlike `predict_artifact()`, whose image input always
    needs at least a decode step): when the artifact was saved with
    `preprocessing=` (e.g. a fitted `Normalize` feature-standardization
    transform, so a brand-new *raw* feature vector is standardized
    identically to training data), it is applied to `input_data` before the
    forward pass; when absent, `input_data` is passed to the model exactly
    as given.

    **Classes are optional, exactly like `predict_artifact()`.** When `path`
    was saved with `forge.save_model(..., classes=...)`, each row's raw
    prediction is turned into a `ClassificationPrediction` via
    `interpret_classification()` -- `.label`/`.index`/`.confidence` --
    returning `list[ClassificationPrediction]`. When no class vocabulary was
    saved, this returns `list[int]` (one raw predicted class index per row)
    instead -- the same honest "less structured information rather than a
    fabricated label" policy `predict_artifact()` already documents.

    `device` defaults to the device recorded in the archive (`load_model()`'s
    own default) -- pass `device="cpu"`/`device="cuda"` to override, exactly
    as `load_model()`/`predict_tensor_artifact()` themselves accept.

    **Input-contract validation (Milestone 101).** Identical to
    `predict_tensor_artifact()`'s own -- see that function's docstring for
    the exact mechanism, message format, and honest scope limits (structural
    width only, never feature order/semantics). Here the required task is
    `task="tabular_classification"` rather than `"regression"`.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact or an unreconstructable `"preprocessing"` entry (the same
    conditions `load_model()`/`load_preprocessing()` already raise), and
    `forge.DataError` if `input_data` is not a `Tensor`/NumPy array/list/tuple.

    **Scope.** Composes exactly `load_model()` + `load_preprocessing()` +
    `predict()` + `load_classes()` + `interpret_classification()`, each
    called unchanged -- no new artifact format, generic input abstraction, or
    task-detection machinery, mirroring `predict_tensor_artifact()`'s own
    Scope paragraph.
    """
    from ..serialization.model import inspect_model as _inspect_model
    from ..serialization.model import load_classes as _load_classes
    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    checked = _coerce_numeric_input(input_data, "predict_tabular_classification_artifact")

    info = _inspect_model(path)
    preprocessing = _load_preprocessing(path)
    model = _load_model(path, device=device.type if isinstance(device, Device) else device)
    output = _predict_tensor_core(
        model, preprocessing, info.input_schema, checked, "predict_tabular_classification_artifact",
        require_batch=True,
    )

    classes = _load_classes(path)
    if classes is not None:
        return interpret_classification(output, classes)
    return [int(i) for i in np.argmax(output.numpy(), axis=1)]


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
    length: "int | None" = None,
) -> "ClassificationPrediction | int | Tensor | list[ClassificationPrediction] | list[int] | list[str]":
    """Predict from any supported portable `.forge` artifact, in one call, with no manual workflow choice (Milestones 86/87/90).

    ```python
    result = forge.predict_model("model.forge", input_data)
    ```

    Before this function, a developer holding a `.forge` file first had to
    call `forge.inspect_model()` (Milestone 85) -- or already know, out of
    band -- whether it was a classification, regression, segmentation,
    sequence, or tabular-classification artifact, in order to pick the right
    one of `predict_artifact()`/`predict_tensor_artifact()`/
    `predict_image_artifact()`/`predict_sequence_artifact()`/
    `predict_tabular_classification_artifact()` (Milestones 82-84/90/91).
    `predict_model()` closes that gap: it inspects `path` itself via
    `inspect_model()`, determines which of the five workflows the artifact's
    own persisted metadata supports (see `_determine_workflow()` above for
    the exact signal and why it is reliable), and delegates to that function
    unchanged -- returning exactly what it would have returned. This
    function adds no new inference logic, artifact format, or
    input-conversion machinery of its own.

    `input_data` is whatever the *selected* workflow's own function expects
    -- a `str`/`os.PathLike` image path for classification or segmentation, a
    `Tensor`/NumPy array/nested list for regression or tabular classification,
    or a non-empty sequence of vocabulary tokens for sequence generation
    (`predict_sequence_artifact()`'s own `seed`) -- and is validated exactly
    as strictly as calling that function directly would be: an incompatible
    input still raises `forge.DataError` with that function's own message
    (e.g. a `Tensor` given to a `task="classification"` artifact raises the
    same error `predict_artifact()` itself raises for a non-path `image` --
    this is exactly the error a tabular classification artifact raised
    before Milestone 91 added `task="tabular_classification"`, since every
    classification artifact used to be routed to `predict_artifact()`
    unconditionally). `predict_model()` deliberately does not re-validate
    `input_data` itself -- the artifact's workflow, not the input's type,
    decides dispatch (see the module's own Milestone 86 paragraph): input
    type is only ever used, by the delegated function, to check compatibility
    with the workflow the artifact's metadata already selected.

    `length` (Milestone 90) is required, and used, only for a `task="sequence"`
    artifact -- the number of new tokens to generate, passed straight through
    to `predict_sequence_artifact()`. Omitting it for a sequence artifact
    raises `forge.DataError` immediately (there is no sensible default number
    of tokens to generate); it is silently ignored for the other four
    workflows, exactly as `device` already is when a workflow doesn't need it.

    `device` is passed through unchanged to the selected function --
    defaults to the device recorded in the archive, exactly like
    `predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`/
    `predict_sequence_artifact()` themselves.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact (the same conditions `inspect_model()` itself raises), and also
    `forge.PersistenceError` when the artifact's metadata does not reliably
    identify one of the four supported workflows (see
    `_determine_workflow()`) -- this is a deliberate refusal to guess, not a
    bug: an unsupported artifact should fail clearly rather than silently
    produce a result under the wrong interpretation.

    **Scope.** Supports exactly the five workflows `predict_artifact()`/
    `predict_tensor_artifact()`/`predict_image_artifact()`/
    `predict_sequence_artifact()`/`predict_tabular_classification_artifact()`
    already implement -- no generic task registry, no model-architecture
    discovery beyond `inspect_model()`'s existing `ModelSummary.module_types`,
    no automatic input conversion between shapes. A future artifact shape
    none of these functions covers stays unsupported here too, until Forge
    has one.
    """
    from ..serialization.model import inspect_model as _inspect_model

    info = _inspect_model(path)
    workflow = _determine_workflow(info)

    if workflow == "classification":
        return predict_artifact(path, input_data, device=device)
    if workflow == "regression":
        return predict_tensor_artifact(path, input_data, device=device)
    if workflow == "segmentation":
        return predict_image_artifact(path, input_data, device=device)
    if workflow == "tabular_classification":
        return predict_tabular_classification_artifact(path, input_data, device=device)

    if length is None:
        raise DataError(
            "predict_model() requires length= for a sequence artifact -- the number of new "
            "tokens to generate (see forge.predict_sequence_artifact()'s own length parameter)."
        )
    return predict_sequence_artifact(path, input_data, length, device=device)


class ArtifactPredictor:
    """A `.forge` artifact loaded once, for repeated in-process inference (Milestone 102).

    ```python
    predictor = forge.load_predictor("model.forge")

    result_1 = predictor.predict(input_1)
    result_2 = predictor.predict(input_2)
    result_3 = predictor.predict(input_3)
    ```

    `forge.predict_model()` (Milestones 86/87/90/91) is a genuinely convenient
    one-shot call, but it reopens and reconstructs the entire artifact --
    `inspect_model()`, `load_model()`, `load_preprocessing()`, `load_classes()`
    -- on every single call. That is invisible for one prediction; for an
    application making many predictions from the same artifact (a batch job,
    a loop over incoming rows, a long-running process), it means every call
    repeats the same disk read, archive parsing, and model reconstruction for
    no reason -- the artifact never changes between calls. `ArtifactPredictor`
    is the reusable counterpart: `forge.load_predictor(path)` performs exactly
    the loading `predict_model()` would have performed, once, and returns an
    object that retains it -- the model, its preprocessing, its class/
    vocabulary metadata, its `InputSchema` (Milestone 101), and which of the
    five workflows (`_determine_workflow()`) the artifact represents.
    `predictor.predict(...)` then reruns only the genuinely per-call work:
    validating this input, applying preprocessing, and running the model
    forward pass.

    **This is not a second inference engine.** `predict()` delegates to the
    exact same artifact-independent core functions (`_classify_image_core()`,
    `_predict_tensor_core()`, `_segment_image_core()`, `_generate_sequence_
    core()`) that `predict_artifact()`/`predict_tensor_artifact()`/
    `predict_image_artifact()`/`predict_tabular_classification_artifact()`/
    `predict_sequence_artifact()` themselves call -- see this module's own
    Milestone 102 paragraph. A prediction through `ArtifactPredictor` and the
    equivalent one-shot call agree exactly, for the same artifact/input/
    device, because they run the identical code path after loading.

    **Construction.** Never constructed directly -- `forge.load_predictor()`
    is the only producer, mirroring `TrainingResult`'s own "never constructed
    directly" convention. This keeps "an `ArtifactPredictor` always reflects
    a successfully loaded, validated artifact" an invariant a caller can rely
    on, rather than a bare dataclass a caller could partially construct.

    **Task support.** All five workflows `predict_model()` supports --
    `classification`, `regression`, `segmentation`, `sequence`,
    `tabular_classification` -- are supported here, unchanged. `predict()`'s
    calling convention matches `predict_model()`'s own: an image file path
    for classification/segmentation, a `Tensor`/NumPy array/nested list for
    regression/tabular classification, or a non-empty token sequence (plus
    required `length=`) for sequence generation.

    **Metadata.** `.task`/`.input_schema`/`.classes` expose the same
    already-loaded `ModelInfo` fields `inspect_model()` would have returned --
    no second `inspect_model()`/archive read. `.model` exposes the loaded
    `Module` itself, read-only in spirit (Forge does not enforce Python
    attribute immutability): mutating it invalidates the "this predictor's
    cached state matches what was loaded from `path`" assumption the same way
    directly mutating any other loaded object would, and is the caller's own
    responsibility, not something `predict()` guards against on every call.

    **Lifetime.** An `ArtifactPredictor` owns its loaded model/preprocessing
    for exactly as long as the object exists -- ordinary Python object
    lifetime, no context manager, no global cache. Forge never tracks or
    reuses `ArtifactPredictor` instances on the caller's behalf; a caller
    wanting one loaded model shared across a request handler keeps its own
    reference, exactly as it would for any other Python object.

    **Concurrency.** No thread-safety infrastructure is added or implied.
    `predict()` mutates no persistent state on `self` (the same `model.eval()`
    /`no_grad()`/mode-restoration discipline `predict()` itself already
    documents), so sequential reuse from one thread is safe by construction,
    but concurrent calls from multiple threads follow whatever thread-safety
    `forge.nn.Module.__call__`/the active backend already provide -- the same
    guarantee (or lack of one) calling `predict()` directly on a shared model
    from multiple threads would have. This milestone does not change that.

    **Immutability of the artifact.** `predict()` never writes to `path` --
    loading happens once, in `load_predictor()`, and nothing afterward
    reopens the archive. `evaluate()` (Milestone 113) is equally read-only.
    """

    def __init__(
        self,
        path: str,
        info: Any,
        workflow: str,
        model: Module,
        preprocessing: "Any | None",
        classes: "list[str] | None",
    ) -> None:
        self._path = path
        self._info = info
        self._workflow = workflow
        self._model = model
        self._preprocessing = preprocessing
        self._classes = classes

    @property
    def task(self) -> "str | None":
        """The workflow this artifact was resolved to (`_determine_workflow()`'s result) --
        one of `forge.serialization.model.TASK_TYPES`. Never `None`: unlike
        `ModelInfo.task` (which is `None` for a legacy artifact with no
        `task=` metadata), `load_predictor()` already resolved the workflow
        (falling back to `_legacy_infer_workflow()` when needed) or raised
        `PersistenceError` trying -- an existing `ArtifactPredictor` always
        has a definite, resolved workflow.
        """
        return self._workflow

    @property
    def input_schema(self) -> "Any | None":
        """`InputSchema | None` (Milestone 101), read from the artifact's own
        `ModelInfo` at load time -- see `InputSchema`'s own docstring for
        what it is and is not.
        """
        return self._info.input_schema

    @property
    def classes(self) -> "list[str] | None":
        """The saved class/vocabulary list, or `None` -- the same value
        `load_classes(path)` would return, read once at load time.
        """
        return self._classes

    @property
    def model(self) -> Module:
        """The loaded `forge.nn.Module` this predictor runs -- see this
        class's own **Metadata** docstring paragraph for the mutability
        contract.
        """
        return self._model

    def predict(
        self,
        input_data: Any,
        *,
        length: "int | None" = None,
        rng: "np.random.Generator | None" = None,
        threshold: float = 0.5,
    ) -> Any:
        """Run one prediction through the already-loaded artifact -- no reload, no reconstruction.

        `input_data` must match this predictor's own `.task`, exactly as
        `predict_model()`'s own `input_data` must match the artifact's task
        (see that function's docstring): an image file path for
        `"classification"`/`"segmentation"`, a `Tensor`/NumPy array/nested
        list for `"regression"`/`"tabular_classification"`, or a non-empty
        token sequence for `"sequence"` (with `length=` required for that
        task -- omitting it raises `forge.DataError`, mirroring
        `predict_model()`'s own requirement). `rng`/`threshold` are only used
        for `"sequence"`/`"segmentation"` respectively, and silently ignored
        otherwise -- the same "irrelevant keyword is ignored, not an error"
        policy `predict_model()`'s own `device=`/`length=` already use.

        Returns exactly what the equivalent one-shot function would have
        returned for the same artifact/input (`ClassificationPrediction`/
        `int` for classification, `Tensor` for regression/segmentation,
        `list[ClassificationPrediction]`/`list[int]` for tabular
        classification, `list` of tokens for sequence).
        """
        if self._workflow == "classification":
            return _classify_image_core(self._model, self._preprocessing, self._classes, input_data)
        if self._workflow == "regression":
            return _predict_tensor_core(
                self._model, self._preprocessing, self.input_schema, input_data, "ArtifactPredictor.predict",
            )
        if self._workflow == "segmentation":
            return _segment_image_core(self._model, self._preprocessing, input_data, threshold)
        if self._workflow == "tabular_classification":
            output = _predict_tensor_core(
                self._model, self._preprocessing, self.input_schema, input_data, "ArtifactPredictor.predict",
                require_batch=True,
            )
            if self._classes is not None:
                return interpret_classification(output, self._classes)
            return [int(i) for i in np.argmax(output.numpy(), axis=1)]

        if length is None:
            raise DataError(
                "ArtifactPredictor.predict() requires length= for a sequence artifact -- the "
                "number of new tokens to generate (see forge.predict_sequence_artifact()'s own "
                "length parameter)."
            )
        return _generate_sequence_core(
            self._model, self._classes, input_data, length, rng, "ArtifactPredictor.predict",
        )

    def evaluate(
        self,
        X: Any,
        y: Any = None,
        *,
        batch_size: int = 256,
    ) -> "ClassificationEvaluationResult | RegressionEvaluationResult":
        """Score this artifact on held-out labeled data, using its own persisted preprocessing (Milestone 113).

        ```python
        predictor = forge.load_predictor("model.forge")
        result = predictor.evaluate(X_test, y_test)
        result.accuracy, result.confusion_matrix      # classification
        result.mse, result.mae                        # regression
        ```

        The evaluation path is exactly the prediction path: raw `X` ->
        `InputSchema` check -> the artifact's persisted `preprocessing` ->
        the loaded model -> metrics. The caller never re-creates
        `Normalize`/`ReplaceValue`/`Resize` outside the artifact, which is the
        whole point: `Trainer.evaluate()` runs whatever it is handed straight
        through the model and, on raw rows, silently returned 37.0% for an
        artifact that actually scores 72.1% (Milestone 97). Nothing here loads
        the artifact again, rebuilds the model, or writes anything: it is the
        already-loaded predictor, run in batches of `batch_size`.

        **Supported tasks** (anything else raises `forge.DataError`):

        - `"tabular_classification"` -- `X` is `(n, ...)` numeric features
          (`Tensor`/NumPy array/nested list, as for `predict()`); `y` is `n`
          class **names** (each must be in `predictor.classes`) or integer class
          **indices** in `[0, len(classes))`. Returns
          `ClassificationEvaluationResult`.
        - `"regression"` -- same `X`; `y` is `(n,)`, `(n, 1)` or `(n, outputs)`
          matching the model's output. Returns `RegressionEvaluationResult`.
        - `"classification"` (image) -- `X` is a directory in the
          `forge.data.ImageFolder` layout (`root/class_a/*.jpg`,
          `root/class_b/*.jpg`, ...) and `y` must be omitted: the labels are the
          directory names, matched **by name** against `predictor.classes` (never
          by discovery order); a directory naming a class the artifact does not
          have is a `DataError` naming both vocabularies. An unreadable image
          raises rather than being skipped, because skipping would silently
          change what is being measured. Returns
          `ClassificationEvaluationResult`.
        - `"segmentation"` and `"sequence"` have no evaluation semantics yet.

        Classification artifacts must carry a persisted class list
        (`PersistenceError` otherwise): the confusion matrix is indexed by the
        artifact's class order, never by an order inferred from `y`.

        `batch_size` only bounds memory; results are the same up to float32
        rounding in `loss` (batched matrix products round differently).
        Read-only: the artifact file, the model's parameters, the preprocessing
        and the model's train/eval mode are untouched, and no RNG is consumed,
        so repeated calls on the same inputs return identical results. Runs on
        whatever device the predictor was loaded onto
        (`forge.load_predictor(path, device=...)`).

        Raises `forge.DataError` for an unsupported task, a missing/mismatched/
        unknown `y`, a `y` of the wrong shape or dtype, an `X` with the wrong
        feature count or rank, non-finite values in `X` (after preprocessing) or
        `y`, and a model output that is non-finite.
        """
        fn_name = "ArtifactPredictor.evaluate"
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise DataError(f"{fn_name}() batch_size must be a positive int, got {batch_size!r}.")

        if self._workflow == "classification":
            return self._evaluate_image_folder(X, y, batch_size, fn_name)
        if self._workflow == "tabular_classification":
            classes = require_evaluation_classes(self._path, self._classes)
            features = self._evaluation_features(X, fn_name)
            labels = self._evaluation_labels(y, features.shape[0], classes, fn_name)
            output = self._evaluate_features(features, batch_size, fn_name)
            return build_classification_result(self._workflow, output, labels, classes, self._path)
        if self._workflow == "regression":
            features = self._evaluation_features(X, fn_name)
            if y is None:
                raise DataError(f"{fn_name}() requires y (the regression targets) for a regression artifact.")
            targets = encode_regression_targets(y, fn_name)
            if targets.shape[0] != features.shape[0]:
                raise DataError(
                    f"{fn_name}() X has {features.shape[0]} sample(s) but y has {targets.shape[0]}."
                )
            output = self._evaluate_features(features, batch_size, fn_name)
            return build_regression_result(self._workflow, output, targets, fn_name)

        raise DataError(
            f"{fn_name}() does not support task '{self._workflow}' -- evaluation is defined only for "
            "'classification' (image folders), 'tabular_classification' and 'regression' artifacts. "
            "Use predict() for this artifact."
        )

    @staticmethod
    def _evaluation_features(X: Any, fn_name: str) -> np.ndarray:
        """`X` as one host array with a leading sample axis, or `DataError` (`_coerce_numeric_input()`'s own rules)."""
        try:
            features = _coerce_numeric_input(X, fn_name).to("cpu").numpy()
        except ValueError as exc:  # e.g. ragged nested lists, which Tensor() rejects with a raw NumPy error
            raise DataError(f"{fn_name}() could not interpret X as a numeric array: {exc}") from exc
        if features.ndim < 2:
            raise DataError(
                f"{fn_name}() requires a batched X, shape (n_samples, ...), got shape {features.shape}. "
                "Wrap a single row in an extra list, e.g. [[...]]."
            )
        if features.shape[0] == 0:
            raise DataError(f"{fn_name}() received no samples.")
        return features

    @staticmethod
    def _evaluation_labels(y: Any, n_samples: int, classes: "list[str]", fn_name: str) -> np.ndarray:
        if y is None:
            raise DataError(
                f"{fn_name}() requires y (one class name or class index per sample) for a "
                "classification artifact."
            )
        labels = encode_class_labels(y, classes, fn_name)
        if labels.shape[0] != n_samples:
            raise DataError(f"{fn_name}() X has {n_samples} sample(s) but y has {labels.shape[0]}.")
        return labels

    def _evaluate_in_batches(
        self, n_samples: int, batch_size: int, fn_name: str, run_batch: "Callable[[int, int], Tensor]",
    ) -> Tensor:
        """Call `run_batch(start, stop)` over `n_samples` in chunks; return all outputs as one CPU `Tensor`.

        Metrics are computed once on the concatenated outputs, not averaged per
        batch, so a result never depends on `batch_size`. A raw
        `ShapeMismatchError` (the model or a transform rejecting the input's
        shape) becomes a `DataError`, as the `InputSchema` check already does
        for artifacts that have one.
        """
        chunks: "list[np.ndarray]" = []
        dtype = None
        for start in range(0, n_samples, batch_size):
            try:
                output = run_batch(start, min(start + batch_size, n_samples))
            except ShapeMismatchError as exc:
                raise DataError(
                    f"{fn_name}() X is not compatible with this artifact's model/preprocessing: {exc}"
                ) from exc
            dtype = output.dtype
            chunks.append(output.numpy())
        combined = np.concatenate(chunks, axis=0)
        require_finite_output(combined, fn_name)
        return Tensor(combined, dtype=dtype, device="cpu")

    def _evaluate_features(self, features: np.ndarray, batch_size: int, fn_name: str) -> Tensor:
        """Numeric `evaluate()`: each chunk goes through `_predict_tensor_core()` unchanged (schema check ->
        persisted preprocessing -> non-finite check -> `predict()`), so evaluation and prediction cannot
        disagree about how a row is prepared.
        """
        return self._evaluate_in_batches(
            features.shape[0], batch_size, fn_name,
            lambda start, stop: _predict_tensor_core(
                self._model, self._preprocessing, self.input_schema, features[start:stop], fn_name,
                require_batch=True,
            ),
        )

    def _evaluate_image_folder(
        self, root: Any, y: Any, batch_size: int, fn_name: str,
    ) -> ClassificationEvaluationResult:
        """`evaluate()` for a `task="classification"` artifact: `root` is an `ImageFolder`-layout directory."""
        from ..data.image_folder import ImageFolder

        classes = require_evaluation_classes(self._path, self._classes)
        if y is not None:
            raise DataError(
                f"{fn_name}() does not take y for an image-classification artifact: the labels are the "
                "directory names under X (root/<class_name>/<image>)."
            )
        if not isinstance(root, (str, os.PathLike)):
            raise DataError(
                f"{fn_name}() requires X to be a directory path (str or os.PathLike) laid out as "
                f"root/<class_name>/<image files> for an image-classification artifact, got "
                f"{type(root).__name__}."
            )

        dataset = ImageFolder(root)
        index_of = {name: i for i, name in enumerate(classes)}
        unknown = sorted({dataset.classes[idx] for _, idx in dataset.samples if dataset.classes[idx] not in index_of})
        if unknown:
            raise DataError(
                f"{fn_name}() directory '{root}' has class folder(s) {unknown!r} that are not among "
                f"this artifact's classes {classes!r}. Folder names are matched to the artifact's "
                "classes by name."
            )
        paths = [path for path, _ in dataset.samples]
        labels = np.array([index_of[dataset.classes[idx]] for _, idx in dataset.samples], dtype=np.int64)

        def run_batch(start: int, stop: int) -> Tensor:
            prepared = [
                self._preprocessing(_decode_image_for_model(self._model, path)).to("cpu").numpy()
                for path in paths[start:stop]
            ]
            return predict(self._model, Tensor(np.stack(prepared)))

        output = self._evaluate_in_batches(len(paths), batch_size, fn_name, run_batch)
        return build_classification_result(self._workflow, output, labels, classes, self._path)

    def __repr__(self) -> str:
        return f"ArtifactPredictor(path={self._path!r}, task={self._workflow!r})"


def load_predictor(path: str, *, device: "str | Device | None" = None) -> ArtifactPredictor:
    """Load a portable `.forge` artifact once and return a reusable `ArtifactPredictor` (Milestone 102).

    ```python
    predictor = forge.load_predictor("model.forge")
    for row in incoming_rows:
        result = predictor.predict(row)
    ```

    Performs exactly the loading work `predict_model()` performs on every
    call -- `inspect_model()` (task/preprocessing/classes/`InputSchema`
    metadata, Milestone 85/101), `_determine_workflow()` (Milestones 86/87),
    `load_model()`, `load_preprocessing()`, `load_classes()` -- exactly once,
    and retains the results on the returned `ArtifactPredictor` (see that
    class's own docstring for the full state it keeps and why). Use this
    instead of `forge.predict_model()` when an application will make more
    than one prediction from the same artifact; use `predict_model()`
    directly for a single, one-off prediction, where the extra object this
    function returns has no benefit.

    `device` defaults to the device recorded in the archive (`load_model()`'s
    own default) -- pass `device="cpu"`/`device="cuda"` to load the model
    there instead. Every subsequent `predictor.predict(...)` call runs on
    that same, already-loaded model; there is no per-call `device=` override
    (a caller needing a different device loads a second `ArtifactPredictor`).

    **Validation performed at load time, not at every `predict()` call**:
    for a `"classification"`/`"segmentation"` artifact, a missing
    preprocessing configuration raises `forge.PersistenceError` here (the
    same condition `predict_artifact()`/`predict_image_artifact()` raise, now
    caught once instead of on every prediction); for a `"sequence"` artifact,
    a missing vocabulary or a model that does not implement the
    stepwise-recurrence protocol (`init_hidden`/`step`) each raise
    `forge.PersistenceError` here too, mirroring `predict_sequence_
    artifact()`'s own checks.

    Raises `forge.PersistenceError` for a missing/corrupt/unsupported-version
    artifact (the same conditions `inspect_model()` itself raises), and also
    when the artifact's metadata does not reliably identify one of the five
    supported workflows (`_determine_workflow()`'s own deliberate refusal to
    guess) -- identical failure conditions to `predict_model()`'s own.

    **Scope.** Composes exactly `inspect_model()` + `load_model()` +
    `load_preprocessing()` + `load_classes()`, each called unchanged, once.
    No new artifact format, model cache, or serving infrastructure -- see
    this module's own Milestone 102 paragraph.
    """
    from ..serialization.model import inspect_model as _inspect_model
    from ..serialization.model import load_classes as _load_classes
    from ..serialization.model import load_model as _load_model
    from ..serialization.model import load_preprocessing as _load_preprocessing

    info = _inspect_model(path)
    workflow = _determine_workflow(info)

    model = _load_model(path, device=device.type if isinstance(device, Device) else device)
    preprocessing = _load_preprocessing(path)
    classes = _load_classes(path)

    if workflow in ("classification", "segmentation"):
        _require_preprocessing(path, preprocessing, "load_predictor")
    if workflow == "sequence":
        _require_sequence_vocab(path, classes, "load_predictor")
        _require_sequence_protocol(path, model, "load_predictor")

    return ArtifactPredictor(path, info, workflow, model, preprocessing, classes)


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
    "predict_sequence_artifact", "predict_tabular_classification_artifact", "predict_model", "generate_sequence",
    "interpret_classification", "ClassificationPrediction", "ArtifactPredictor", "load_predictor",
]
