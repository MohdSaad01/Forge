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


__all__ = ["seed", "default_generator"]
