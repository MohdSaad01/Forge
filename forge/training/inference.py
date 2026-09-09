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
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..autograd import no_grad
from ..backend.device import Device
from ..exceptions import DataError, TrainerError
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


__all__ = ["predict"]
