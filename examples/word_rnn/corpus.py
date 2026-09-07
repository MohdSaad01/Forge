"""Original, deterministically-generated word-level training corpus (Milestone 54).

Not downloaded, not copied from any external source, and quotes no real
work -- generated entirely from a synthetic, procedurally-enumerated
vocabulary via a seeded `numpy.random.Generator`, the same
"synthetic/reproducible/offline" convention `examples/char_rnn/corpus.py`
(Milestone 50) already established for its character-level corpus. The one
deliberate difference from that module: this vocabulary is sized (~1,800
distinct word tokens) specifically to sit in the "thousands+" range M50
predicted would make Forge's one-hot-plus-Linear workaround materially
worse than a real embedding lookup -- see `docs/development/
m54-product-direction.md` for the direct measurement that motivated
`nn.Embedding`.

`VOCAB_WORDS` is a fixed, deterministic enumeration of pronounceable
four-letter CVCV tokens (`"bamo"`, `"dilu"`, ...) -- no RNG involved in
building the vocabulary itself, only in choosing which slice of it plays
which grammatical role and which words fill each generated sentence. This
sidesteps any copyright/fabrication concern entirely (these are not real
English words) while still giving the RNN genuine, learnable
next-word structure: sentences follow a fixed
`subject verb object (connector subject verb object)? .` template (mirroring
`examples/char_rnn/corpus.py`'s own subject/verb/object/connector grammar),
so the model can learn "the word after a subject-class word is usually a
verb-class word" the same way it could at the character level -- just with
~1,800 possible words instead of ~30 possible characters.
"""

from __future__ import annotations

import numpy as np

_CONSONANTS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"


def _build_vocabulary(size: int) -> "list[str]":
    """Deterministically enumerate `size` distinct CVCV tokens (fixed nested order, no RNG)."""
    words: "list[str]" = []
    for c1 in _CONSONANTS:
        for v1 in _VOWELS:
            for c2 in _CONSONANTS:
                for v2 in _VOWELS:
                    if len(words) >= size:
                        return words
                    words.append(c1 + v1 + c2 + v2)
    raise ValueError(f"Alphabet too small to enumerate {size} distinct CVCV tokens.")


# ~1,800 tokens, split into three disjoint, equally-sized grammatical roles --
# a word is always drawn from exactly one role, so the roles never overlap
# and `len(VOCAB_WORDS) == len(_SUBJECT_WORDS) + len(_VERB_WORDS) + len(_OBJECT_WORDS)`.
_ALL_WORDS = _build_vocabulary(1800)
_ROLE_SIZE = len(_ALL_WORDS) // 3
_SUBJECT_WORDS = _ALL_WORDS[:_ROLE_SIZE]
_VERB_WORDS = _ALL_WORDS[_ROLE_SIZE : 2 * _ROLE_SIZE]
_OBJECT_WORDS = _ALL_WORDS[2 * _ROLE_SIZE :]
_CONNECTORS = ["ke", "ta", "no", "fen", "ru"]

VOCAB_WORDS = _ALL_WORDS + _CONNECTORS + ["."]


def _generate(seed: int = 0, n_sentences: int = 4000) -> str:
    """Deterministically build `n_sentences` short templated sentences, space-joined."""
    rng = np.random.default_rng(seed)
    sentences = []
    for _ in range(n_sentences):
        subject = rng.choice(_SUBJECT_WORDS)
        verb = rng.choice(_VERB_WORDS)
        obj = rng.choice(_OBJECT_WORDS)
        if rng.random() < 0.5:
            sentence = f"{subject} {verb} {obj} ."
        else:
            connector = rng.choice(_CONNECTORS)
            subject2 = rng.choice(_SUBJECT_WORDS)
            verb2 = rng.choice(_VERB_WORDS)
            obj2 = rng.choice(_OBJECT_WORDS)
            sentence = f"{subject} {verb} {obj} {connector} {subject2} {verb2} {obj2} ."
        sentences.append(sentence)
    return " ".join(sentences)


TEXT = _generate()

__all__ = ["TEXT", "VOCAB_WORDS"]
