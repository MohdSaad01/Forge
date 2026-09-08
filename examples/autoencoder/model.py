"""`ConvAutoencoder`: the convolutional autoencoder used by this example (Milestone 63).

```text
Encoder:
(N, 1, 28, 28)
    -> Conv2d(1, 16, k=3, pad=1)  -> ReLU -> MaxPool2d(2)   -> (N, 16, 14, 14)
    -> Conv2d(16, 32, k=3, pad=1) -> ReLU -> MaxPool2d(2)   -> (N, 32, 7, 7)
    -> Flatten -> Linear(1568, latent_dim) -> ReLU          -> (N, latent_dim)

Decoder (mirrors the encoder):
(N, latent_dim)
    -> Linear(latent_dim, 1568) -> ReLU -> reshape          -> (N, 32, 7, 7)
    -> UpsampleNearest2d(2) -> Conv2d(32, 16, k=3, pad=1) -> ReLU  -> (N, 16, 14, 14)
    -> UpsampleNearest2d(2) -> Conv2d(16, 1, k=3, pad=1)          -> (N, 1, 28, 28)
```

`UpsampleNearest2d` (Milestone 63) is the one new primitive this example
required: composing it with `Conv2d` (upsample, then convolve) grows a
spatial feature map back up without a transposed convolution -- a standard,
deliberate architectural choice (it avoids the checkerboard-artifact
failure mode transposed convolution is well known for), not a workaround.
Every other layer (`Conv2d`, `MaxPool2d`, `Flatten`, `Linear`, `ReLU`)
already existed and needed no change.

`ConvAutoencoder` is a custom composite `Module` (not one of the library's
built-in registered types, since it needs `encode()`/`decode()` exposed
separately from `forward()` for `train.py`'s latent-space evaluation), so it
registers itself for persistence via `forge.serialization.register_module()`
at import time -- the same pattern `examples/char_rnn/model.py`'s `CharRNN`
and `examples/word_rnn/model.py`'s `WordRNN` already establish.

~111.5k trainable parameters at the default `latent_dim=32`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, UpsampleNearest2d
from forge.serialization import register_module
from forge.tensor.tensor import Tensor

_ENC_CHANNELS_1 = 16
_ENC_CHANNELS_2 = 32
_FEATURE_MAP_SIZE = 7  # 28 -> 14 -> 7 after two stride-2 MaxPool2d(2) layers
_FLATTENED_FEATURES = _ENC_CHANNELS_2 * _FEATURE_MAP_SIZE * _FEATURE_MAP_SIZE


class ConvAutoencoder(Module):
    def __init__(
        self,
        latent_dim: int = 32,
        dtype: Any = None,
        device: str = "cpu",
        generator: "np.random.Generator | None" = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        kwargs = dict(dtype=dtype, device=device, generator=generator)

        # Encoder
        self.enc_conv1 = Conv2d(1, _ENC_CHANNELS_1, kernel_size=3, padding=1, **kwargs)
        self.enc_pool1 = MaxPool2d(2)
        self.enc_conv2 = Conv2d(_ENC_CHANNELS_1, _ENC_CHANNELS_2, kernel_size=3, padding=1, **kwargs)
        self.enc_pool2 = MaxPool2d(2)
        self.flatten = Flatten()
        self.enc_fc = Linear(_FLATTENED_FEATURES, latent_dim, **kwargs)

        # Decoder (mirrors the encoder)
        self.dec_fc = Linear(latent_dim, _FLATTENED_FEATURES, **kwargs)
        self.dec_up1 = UpsampleNearest2d(2)
        self.dec_conv1 = Conv2d(_ENC_CHANNELS_2, _ENC_CHANNELS_1, kernel_size=3, padding=1, **kwargs)
        self.dec_up2 = UpsampleNearest2d(2)
        self.dec_conv2 = Conv2d(_ENC_CHANNELS_1, 1, kernel_size=3, padding=1, **kwargs)

        self.relu = ReLU()

    def encode(self, x: Tensor) -> Tensor:
        """`(N, 1, 28, 28) -> (N, latent_dim)`."""
        h = self.relu(self.enc_pool1(self.enc_conv1(x)))
        h = self.relu(self.enc_pool2(self.enc_conv2(h)))
        h = self.flatten(h)
        return self.relu(self.enc_fc(h))

    def decode(self, z: Tensor) -> Tensor:
        """`(N, latent_dim) -> (N, 1, 28, 28)` (linear output -- no final activation).

        The final layer has no activation, matching `examples/regression/
        model.py`'s bare-`Linear` output convention: the reconstruction
        target (pixels scaled to `[0, 1]` by `train.py`'s transform, never
        renormalized to zero mean) is compared via `MSELoss`, an ordinary
        regression target, not a probability -- Forge has no `Sigmoid`
        primitive yet and none is needed here (see `docs/development/
        m63-conv-autoencoder.md`).
        """
        n = z.shape[0]
        h = self.relu(self.dec_fc(z))
        h = h.reshape(n, _ENC_CHANNELS_2, _FEATURE_MAP_SIZE, _FEATURE_MAP_SIZE)
        h = self.relu(self.dec_conv1(self.dec_up1(h)))
        return self.dec_conv2(self.dec_up2(h))

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def __repr__(self) -> str:
        return f"ConvAutoencoder(latent_dim={self.latent_dim})"


register_module(
    "ConvAutoencoder",
    ConvAutoencoder,
    get_config=lambda m: {"latent_dim": m.latent_dim},
)


def build_model(latent_dim: int = 32, device: str = "cpu", generator=None) -> ConvAutoencoder:
    """Construct a fresh, untrained `ConvAutoencoder` -- see the module docstring for the shape trace."""
    return ConvAutoencoder(latent_dim=latent_dim, device=device, generator=generator)


__all__ = ["ConvAutoencoder", "build_model"]
