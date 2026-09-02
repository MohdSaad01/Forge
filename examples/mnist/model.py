"""The small CNN architecture used by the MNIST example (Milestone 20).

Built entirely from existing `forge.nn` Modules -- no new layer types. Kept
small deliberately: this is meant to train practically on the 940MX
(`docs/development/development-environment.md`), not to be architecturally
interesting.

```text
(N, 1, 28, 28)
    -> Conv2d(1, 8, k=3)   -> (N, 8, 26, 26)
    -> ReLU
    -> MaxPool2d(2)        -> (N, 8, 13, 13)
    -> Conv2d(8, 16, k=3)  -> (N, 16, 11, 11)
    -> ReLU
    -> MaxPool2d(2)        -> (N, 16, 5, 5)
    -> Flatten             -> (N, 400)
    -> Linear(400, 64)
    -> ReLU
    -> Linear(64, 10)      -> (N, 10) logits
```

~27.6k trainable parameters total.
"""

from __future__ import annotations

from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

_FLATTENED_FEATURES = 16 * 5 * 5


def build_model() -> Sequential:
    """Construct a fresh, untrained MNIST CNN -- see the module docstring for the shape trace."""
    return Sequential(
        Conv2d(1, 8, kernel_size=3),
        ReLU(),
        MaxPool2d(2),
        Conv2d(8, 16, kernel_size=3),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(_FLATTENED_FEATURES, 64),
        ReLU(),
        Linear(64, 10),
    )


__all__ = ["build_model"]
