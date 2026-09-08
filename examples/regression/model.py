"""The small MLP architecture used by the tabular regression example (Milestone 60).

Built entirely from existing `forge.nn` layers -- no new module type. `Linear`
and `ReLU`/`Sequential` are already registered for persistence
(`forge/serialization/registry.py`), so this architecture needs no
`register_module()` call, exactly like `examples/mnist/model.py`'s
`Sequential`-based CNN.

```text
(N, 8) -> Linear(8, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> (N, 1)
```

~2.7k trainable parameters total.
"""

from __future__ import annotations

from forge.nn import Linear, ReLU, Sequential

from .dataset import N_FEATURES

_HIDDEN_1 = 64
_HIDDEN_2 = 32


def build_model() -> Sequential:
    """Construct a fresh, untrained regression MLP -- see the module docstring for the shape trace."""
    return Sequential(
        Linear(N_FEATURES, _HIDDEN_1),
        ReLU(),
        Linear(_HIDDEN_1, _HIDDEN_2),
        ReLU(),
        Linear(_HIDDEN_2, 1),
    )


__all__ = ["build_model"]
