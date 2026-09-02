"""A minimal, process-global source of deterministic randomness.

Not a general random-number framework -- just enough to make parameter
initialization reproducible. Layers draw from ``default_generator()`` unless
given an explicit ``numpy.random.Generator``; call ``seed()`` once (e.g. at
the start of a script) to make a run's initialization deterministic.
"""

from __future__ import annotations

import numpy as np

_default_rng = np.random.default_rng()


def seed(value: int) -> None:
    """Reseed Forge's default generator, e.g. for reproducible parameter init."""
    global _default_rng
    _default_rng = np.random.default_rng(value)


def default_generator() -> np.random.Generator:
    """The process-global generator layers use when no generator is given explicitly."""
    return _default_rng


def get_state() -> dict:
    """A JSON-safe snapshot of the default generator's exact internal state.

    Unlike `seed()` (which reseeds to the *start* of a new integer-seeded
    stream), `get_state()`/`set_state()` capture/restore the generator's
    current position in its stream, so future draws after `set_state()`
    reproduce exactly the draws that would have followed at capture time --
    the mechanism `forge.serialization.checkpoint` uses for deterministic
    training resume (see `docs/architecture/persistence.md`).
    """
    return _default_rng.bit_generator.state


def set_state(state: dict) -> None:
    """Restore the default generator to a state previously captured by `get_state()`."""
    _default_rng.bit_generator.state = state


__all__ = ["seed", "default_generator", "get_state", "set_state"]
