"""The small 1D-convolutional architecture used by the waveform classification example (Milestone 62).

Built entirely from existing `forge.nn` layers plus this milestone's two
additions (`Conv1d`, `MaxPool1d`) -- no custom `Module` subclass, so no
`register_module()` call is needed for persistence (every layer here is
already a registered built-in, `forge/serialization/registry.py`), exactly
like `examples/mnist/model.py`'s CNN.

```text
(N, 1, 64) -> Conv1d(1, 16, k=7, pad=3) -> ReLU -> MaxPool1d(2)   -> (N, 16, 32)
           -> Conv1d(16, 32, k=5, pad=2) -> ReLU -> MaxPool1d(2)  -> (N, 32, 16)
           -> Flatten -> Linear(512, 64) -> ReLU -> Linear(64, 4) -> (N, 4)
```

~35k trainable parameters total. Padding keeps each `Conv1d`'s output length
equal to its input length, so `MaxPool1d(2)` alone halves the sequence
length at each stage (`64 -> 32 -> 16`) -- the same "conv preserves length,
pooling downsamples" shape discipline `examples/mnist/model.py`'s CNN uses
in 2D.
"""

from __future__ import annotations

from forge.nn import Conv1d, Flatten, Linear, MaxPool1d, ReLU, Sequential

from .dataset import LENGTH, NUM_CLASSES

_CHANNELS_1 = 16
_CHANNELS_2 = 32
_HIDDEN = 64
_POOLED_LENGTH = LENGTH // 4  # two MaxPool1d(2) stages
_FLAT_SIZE = _CHANNELS_2 * _POOLED_LENGTH


def build_model() -> Sequential:
    """Construct a fresh, untrained waveform-classification CNN -- see the module docstring for the shape trace."""
    return Sequential(
        Conv1d(1, _CHANNELS_1, kernel_size=7, padding=3),
        ReLU(),
        MaxPool1d(2),
        Conv1d(_CHANNELS_1, _CHANNELS_2, kernel_size=5, padding=2),
        ReLU(),
        MaxPool1d(2),
        Flatten(),
        Linear(_FLAT_SIZE, _HIDDEN),
        ReLU(),
        Linear(_HIDDEN, NUM_CLASSES),
    )


__all__ = ["build_model"]
