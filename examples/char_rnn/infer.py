"""Milestone 90: standalone, fresh-process text generation from one saved char-RNN artifact.

```bash
python -m examples.char_rnn.infer \
    --model examples/char_rnn/artifacts/char_rnn_model.forge \
    --seed "a tensor" \
    --length 200
```

This script is deliberately independent of `train.py`: it does not import
`build_model()`, `Vocab`, or `_one_hot` -- only `forge.predict_model()`
(Milestone 86), which reads the artifact's own persisted `task="sequence"`
metadata and delegates to `predict_sequence_artifact()` (Milestone 90).
Nothing here depends on any in-memory state `train.py` happened to build --
everything a caller needs (model architecture + weights, and the character
vocabulary that turns raw indices back into text) comes from the one
`.forge` file `train.py` wrote.

The one exception -- and it is registration, not state -- is importing
`examples.char_rnn.model`: `CharRNN` is a custom composite `Module` (unlike
every other example this repo has an `infer.py` for, which are built purely
from pre-registered `Sequential`/`Linear`/`Conv2d`), so it must register
itself with `forge.serialization.register_module()` before `load_model()`
can reconstruct it, exactly as `model.py`'s own docstring documents (ADR-003
in `docs/architecture/persistence.md`). This import triggers that
module-level registration side effect only -- it never calls `build_model()`
or constructs a `CharRNN` instance directly here.

This is the concrete Milestone 90 workflow: a developer trains and saves a
char-RNN language model in one process, then -- potentially days later, in a
completely separate process -- loads it here and generates new text with no
manual one-hot-encode/step/decode reconstruction required, and no need to
already know this particular artifact is a sequence one rather than a
classification, regression, or segmentation one (see
`forge/training/inference.py::predict_model()`).

**Tokenization convention.** The seed string is split into individual
characters (`list(seed)`) before being handed to `forge.predict_model()`,
matching the char-level vocabulary `train.py`/`dataset.py::Vocab` build. A
seed containing a character never seen in the training corpus raises
`forge.DataError` naming it -- there is no vocabulary entry to encode it
with.
"""

from __future__ import annotations

import argparse

import forge

import examples.char_rnn.model  # noqa: F401 -- side effect only: registers CharRNN for persistence


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., classes=..., task='sequence').")
    parser.add_argument("--seed", required=True, help="Seed text to prime generation with (tokenized as individual characters).")
    parser.add_argument("--length", type=int, default=200, help="Number of new characters to generate.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, seed: str, length: int, device: "str | None" = None) -> str:
    """Generate `length` new characters from `seed` with the artifact at `model_path`.

    A thin wrapper over `forge.predict_model()` (Milestone 86/90) -- this
    script no longer needs to already know `model_path` is a sequence
    artifact specifically; `predict_model()` determines that from the
    artifact's own persisted metadata and delegates to
    `predict_sequence_artifact()` (Milestone 90) unchanged. Returns the full
    generated string, seed included (matching `generate_sequence()`'s own
    "seed is part of the output" convention).
    """
    generated = forge.predict_model(model_path, list(seed), device=device, length=length)
    return "".join(generated)


def main(argv=None) -> None:
    args = parse_args(argv)
    text = run(args.model, args.seed, args.length, device=args.device)
    print(text)


if __name__ == "__main__":
    main()
