"""Original, deterministically-generated training corpus (Milestone 50).

`TEXT` is not downloaded, not copied from any external source, and quotes no
real work -- it is generated from a small fixed vocabulary of Forge-related
words and sentence templates via a seeded `numpy.random.Generator`, the same
synthetic-data convention `examples/trainer_demo.py` already uses for its
regression/classification demos (there: Gaussian blobs; here: templated
sentences). This sidesteps any copyright/fabrication concern entirely and
keeps the example fully offline and reproducible.

Repetition is deliberate, not incidental: a vanilla RNN this small (default
`--hidden-size 64`) needs recurring character-level structure (word
boundaries, common substrings, sentence-final periods) to show a visibly
decreasing next-character loss within the few CPU/940MX-practical epochs
this example targets -- see `docs/development/m50-char-rnn.md`.
"""

from __future__ import annotations

import numpy as np

_SUBJECTS = [
    "a tensor", "the model", "a parameter", "the optimizer", "a dataset",
    "the loader", "a module", "the kernel", "a gradient", "the backend",
]
_VERBS = [
    "holds", "updates", "computes", "stores", "tracks",
    "produces", "consumes", "returns", "reshapes", "accumulates",
]
_OBJECTS = [
    "a gradient", "a batch", "the loss", "a shape", "a value",
    "the weights", "a sample", "the output", "a device", "the state",
]
_CONNECTORS = ["and then", "while", "before it", "so that it", "until it"]


def _generate(seed: int = 0, n_sentences: int = 180) -> str:
    """Deterministically build `n_sentences` short templated sentences, space-joined."""
    rng = np.random.default_rng(seed)
    sentences = []
    for _ in range(n_sentences):
        subject = rng.choice(_SUBJECTS)
        verb = rng.choice(_VERBS)
        obj = rng.choice(_OBJECTS)
        if rng.random() < 0.5:
            sentence = f"{subject} {verb} {obj}."
        else:
            connector = rng.choice(_CONNECTORS)
            subject2 = rng.choice(_SUBJECTS)
            verb2 = rng.choice(_VERBS)
            obj2 = rng.choice(_OBJECTS)
            sentence = f"{subject} {verb} {obj} {connector} {subject2} {verb2} {obj2}."
        sentences.append(sentence)
    return " ".join(sentences)


TEXT = _generate()

__all__ = ["TEXT"]
