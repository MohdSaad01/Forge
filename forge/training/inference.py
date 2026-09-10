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

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .. import random as forge_random
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


__all__ = ["predict", "generate_sequence", "interpret_classification", "ClassificationPrediction"]
