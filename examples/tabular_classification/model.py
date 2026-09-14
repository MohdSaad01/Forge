"""The small MLP architecture used by the tabular classification example (Milestone 91).

Built entirely from existing `forge.nn` layers, exactly like
`examples/regression/model.py` -- no new module type, no `register_module()`
call needed.

```text
(N, 10) -> Linear(10, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 4) -> (N, 4)
```

Output is raw per-class logits (`CrossEntropyLoss`'s expected input), not a
single scalar -- the one architectural difference from `examples/regression`'s
otherwise identical shape.
"""

from __future__ import annotations

from forge.nn import Linear, ReLU, Sequential

from .dataset import N_CLASSES, N_FEATURES

_HIDDEN_1 = 64
_HIDDEN_2 = 32


def build_model() -> Sequential:
    """Construct a fresh, untrained classification MLP -- see the module docstring for the shape trace."""
    return Sequential(
        Linear(N_FEATURES, _HIDDEN_1),
        ReLU(),
        Linear(_HIDDEN_1, _HIDDEN_2),
        ReLU(),
        Linear(_HIDDEN_2, N_CLASSES),
    )


__all__ = ["build_model"]
