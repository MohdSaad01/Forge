"""The small encoder/decoder architecture used by the segmentation example (Milestone 64).

Built entirely from existing `forge.nn` layers -- no new module type, exactly
`examples/regression/model.py`'s and `examples/mnist/model.py`'s established
"`build_model()` returns a plain `Sequential`" pattern. `Conv2d`, `ReLU`,
`MaxPool2d`, `UpsampleNearest2d`, and `Sequential` are all already registered
for persistence (`forge/serialization/registry.py`), so this architecture
needs no `register_module()` call either.

```text
(N, 3, 32, 32)
    -> Conv2d(3, 16, k=3, pad=1)  -> ReLU -> MaxPool2d(2)   -> (N, 16, 16, 16)
    -> Conv2d(16, 32, k=3, pad=1) -> ReLU                    -> (N, 32, 16, 16)
    -> UpsampleNearest2d(2)                                  -> (N, 32, 32, 32)
    -> Conv2d(32, 1, k=3, pad=1)                              -> (N, 1, 32, 32)
```

This is exactly the architecture Milestone 64's brief suggests as a starting
point (`Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> UpsampleNearest2d ->
Conv2d -> output`) -- direct execution against the real Forge API (forward,
backward, an Adam step, and a save/load round trip, all on both CPU and
CUDA) confirmed it needs no new framework capability, so nothing more
elaborate was built. No skip connections (no channel-concatenation
primitive exists in Forge, and none was demonstrated to be necessary by this
workload -- see `docs/development/m64-unet-segmentation.md`), so this is a
"U-Net-style" shrink-then-grow encoder/decoder, not a literal U-Net.

The final `Conv2d` has no activation: the mask target is `{0, 1}`-valued and
compared via `MSELoss`, an ordinary regression target -- the same
no-final-activation convention `examples/autoencoder/model.py::decode()`
already established (Forge has no `Sigmoid` primitive, and none is needed
here; `train.py`'s evaluation thresholds the raw output at `0.5` to compute
pixel accuracy/IoU).

~5.4k trainable parameters total.
"""

from __future__ import annotations

from forge.nn import Conv2d, MaxPool2d, ReLU, Sequential, UpsampleNearest2d

from .dataset import NUM_CHANNELS

_ENC_CHANNELS = 16
_BOTTLENECK_CHANNELS = 32


def build_model() -> Sequential:
    """Construct a fresh, untrained segmentation model -- see the module docstring for the shape trace."""
    return Sequential(
        Conv2d(NUM_CHANNELS, _ENC_CHANNELS, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(_ENC_CHANNELS, _BOTTLENECK_CHANNELS, kernel_size=3, padding=1),
        ReLU(),
        UpsampleNearest2d(2),
        Conv2d(_BOTTLENECK_CHANNELS, 1, kernel_size=3, padding=1),
    )


__all__ = ["build_model"]
