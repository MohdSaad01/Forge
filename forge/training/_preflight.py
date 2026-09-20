"""Checks the high-level training workflows run before epoch 1 (Milestones 114, 115).

`train_tabular_classifier()`, `train_tabular_regressor()` (M114) and
`train_image_classifier()` (M115) each build a model, train it for minutes or
hours, and only then write an artifact. Two things a caller can get wrong are
knowable up front, and are checked here rather than discovered at the end of
the run (or, for a model with the wrong number of outputs, never -- it trains,
saves and verifies, and only fails on the first `predict()`):

- `check_save_path()` / `preflight_save()` -- can `path` be written, and can the
  model, the fitted preprocessing and the class list be serialised?
- `check_model_contract()` -- does a caller-supplied `model=` accept the real
  preprocessed input and produce `(batch, outputs)` scores?

These are plain functions the three workflows call, not a framework: each
workflow decides what to check and in what order. Nothing here trains, and
nothing here writes the final artifact.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from ..exceptions import DataError, PersistenceError, TrainerError
from ..nn.module import Module
from ..tensor.tensor import Tensor
from .inference import predict


def check_save_path(path: "str | os.PathLike", fn: str) -> None:
    """Raise `DataError` for an empty/non-path `path`, `PersistenceError` if its directory does not exist or it is a directory.

    Cheap (two filesystem calls, no model), so a workflow can run it before it
    does any expensive work such as scanning a dataset. It does not prove `path`
    is writable or its filename valid -- `preflight_save()` does.
    """
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise DataError(f"{fn}() path must be a non-empty file path (str or os.PathLike), got {path!r}.")
    if os.path.isdir(path):
        raise PersistenceError(f"{fn}() path '{path}' is a directory; give a file path such as 'model.forge'.")
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise PersistenceError(
            f"{fn}() cannot save to '{path}': directory '{directory}' does not exist. Create it first."
        )


def preflight_save(
    model: Module, path: "str | os.PathLike", preprocessing: Any, classes: "list[str] | None", task: str, fn: str,
    target_transform: Any = None,
) -> None:
    """Prove `path` can be saved to, before training, without writing the artifact itself.

    Saves the *untrained* `model` (same architecture, so the same serialisability)
    with the real `save_model()` to a temporary sibling of `path`, then removes it.
    The temporary name embeds `path`'s own filename, so an invalid name fails here
    exactly as it would at the final save.
    """
    from ..serialization.model import save_model

    check_save_path(path, fn)
    directory = os.path.dirname(os.path.abspath(path))
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".forge-preflight-", suffix="-" + os.path.basename(path), dir=directory)
    except OSError as exc:
        raise PersistenceError(f"{fn}() cannot save to '{path}': {exc}") from exc
    os.close(fd)
    try:
        save_model(
            model, tmp_path, preprocessing=preprocessing, classes=classes, task=task,
            target_transform=target_transform,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def check_model_contract(
    model: Module, probe: Tensor, outputs: int, what: str, expects: str, output_meaning: str, fn: str,
) -> None:
    """Raise `TrainerError` unless `model` maps the real `probe` batch to `(batch, outputs)`.

    The model is *run* once on `probe` (a few real, already-preprocessed rows or
    images) rather than inspected: that verifies its input shape and its output
    width exactly, for any `Module`, with no architecture guessing. `what` names
    the probe in the message ("features", "images"), `expects` is the input shape
    the model must take ("(batch, 8)"), and `output_meaning` says what the
    `outputs` columns must be.
    """
    try:
        output = predict(model, probe)
    except Exception as exc:  # the caller's model, whatever it raises
        raise TrainerError(
            f"{fn}() model= could not process a {tuple(probe.shape)} batch of preprocessed "
            f"{what} ({type(exc).__name__}: {exc}). The model must take {expects} input."
        ) from exc
    if output.ndim != 2 or output.shape[1] != outputs:
        raise TrainerError(
            f"{fn}() model= produces output of shape {tuple(output.shape)} for a "
            f"{tuple(probe.shape)} input, but this task needs (batch, {outputs}): "
            f"{output_meaning}. Its last layer must have {outputs} output(s)."
        )
