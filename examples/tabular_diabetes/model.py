"""The small MLP architecture used by the real-dataset tabular classification example (Milestone 92).

Built entirely from existing `forge.nn` layers, exactly like
`examples/tabular_classification/model.py` -- no new module type needed.

```text
(N, 8) -> Linear(8, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 2) -> (N, 2)
```

Smaller than `tabular_classification`'s MLP (64/32 hidden units): this
dataset has 768 real rows total (vs. 1,700+ synthetic rows there), and a
smaller network overfits less on a small, noisy, real dataset.
"""

from __future__ import annotations

from forge.nn import Linear, ReLU, Sequential

from .dataset import N_CLASSES, N_FEATURES

_HIDDEN_1 = 32
_HIDDEN_2 = 16


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
