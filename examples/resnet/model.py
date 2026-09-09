"""A small ResNet-style residual CNN for MNIST (Milestone 66).

Built entirely from existing `forge.nn` layers (`Conv2d`, `BatchNorm2d`,
`ReLU`, `MaxPool2d`, `Flatten`, `Linear`) plus ordinary Python/`Module`
composition -- no new framework primitive. Milestone 66's own direct-API
investigation (`docs/development/m66-residual-cnn.md`) proved, by running
real forward/backward/finite-difference checks against this exact
`ResidualBlock`, that Forge's existing `Tensor.__add__`/autograd engine
already accumulates gradients correctly across the two branches of a
residual connection -- so `ResidualBlock` needs no dedicated backward rule,
exactly like `examples/segmentation/model.py`'s encoder/decoder needed none.

```text
(N, 1, 28, 28)
    -> Conv2d(1, 8, k=3, pad=1) -> BatchNorm2d(8) -> ReLU     -> (N, 8, 28, 28)   [stem]
    -> ResidualBlock(8, 8, stride=1)   [identity shortcut]     -> (N, 8, 28, 28)
    -> ResidualBlock(8, 16, stride=2)  [projection shortcut]   -> (N, 16, 14, 14)
    -> ResidualBlock(16, 16, stride=1) [identity shortcut]     -> (N, 16, 14, 14)
    -> MaxPool2d(2)                                            -> (N, 16, 7, 7)
    -> Flatten -> Linear(784, 10)                              -> (N, 10) logits
```

`ResidualBlock` follows the brief's own suggested body exactly:

```text
identity = x (or shortcut_bn(shortcut_conv(x)) when shape/channels change)
x = conv1(x); x = bn1(x); x = relu(x)
x = conv2(x); x = bn2(x)
x = x + identity
x = relu(x)
```

A projection shortcut (1x1 `Conv2d` + `BatchNorm2d`) is only constructed when
`in_channels != out_channels or stride != 1` -- when the input already has
compatible shape, the shortcut is a true identity (no extra parameters), the
standard ResNet "basic block" convention. This is not a speculative
generalized-residual-container primitive: it is one concrete, fixed-shape
`Module` subclass local to this example.

~17.6k trainable parameters total.
"""

from __future__ import annotations

from forge.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU
from forge.serialization import register_module


class ResidualBlock(Module):
    """One residual block: two 3x3 convolutions plus a shortcut, added and re-activated.

    `shortcut_conv`/`shortcut_bn` are only created (and only appear in the
    persisted module tree) when a projection is actually needed; otherwise
    `self.shortcut_conv`/`self.shortcut_bn` are left as plain `None`
    attributes (unregistered, exactly like `Conv2d`'s optional `bias`), and
    `forward()` adds the raw input back in as a true identity.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, device: str = "cpu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self._needs_projection = in_channels != out_channels or stride != 1

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, device=device)
        self.bn1 = BatchNorm2d(out_channels, device=device)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.bn2 = BatchNorm2d(out_channels, device=device)

        if self._needs_projection:
            self.shortcut_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, device=device)
            self.shortcut_bn = BatchNorm2d(out_channels, device=device)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def forward(self, x):
        if self._needs_projection:
            identity = self.shortcut_bn(self.shortcut_conv(x))
        else:
            identity = x

        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h + identity
        return self.relu(h)

    def __repr__(self) -> str:
        return (
            f"ResidualBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"stride={self.stride}, projection_shortcut={self._needs_projection})"
        )


register_module(
    "ResidualBlock",
    ResidualBlock,
    # `stride`/channel counts fully determine whether `__init__` builds a
    # projection shortcut, so `_needs_projection` itself is never persisted
    # -- reconstructing from this config always reproduces the identical
    # child-module set `_build_load_node` expects.
    get_config=lambda m: {
        "in_channels": m.in_channels,
        "out_channels": m.out_channels,
        "stride": m.stride,
    },
)


_STEM_CHANNELS = 8
_STAGE2_CHANNELS = 16
_POOLED_SIZE = 7  # 28 -> (stride-2 block) 14 -> (MaxPool2d(2)) 7
_FLATTENED_FEATURES = _STAGE2_CHANNELS * _POOLED_SIZE * _POOLED_SIZE
_NUM_CLASSES = 10


class ResNetMNIST(Module):
    """The small residual classifier used by this example -- see the module docstring for the shape trace."""

    def __init__(self, num_classes: int = _NUM_CLASSES, device: str = "cpu"):
        super().__init__()
        self.num_classes = num_classes

        self.stem_conv = Conv2d(1, _STEM_CHANNELS, kernel_size=3, padding=1, device=device)
        self.stem_bn = BatchNorm2d(_STEM_CHANNELS, device=device)
        self.relu = ReLU()

        self.block1 = ResidualBlock(_STEM_CHANNELS, _STEM_CHANNELS, stride=1, device=device)
        self.block2 = ResidualBlock(_STEM_CHANNELS, _STAGE2_CHANNELS, stride=2, device=device)
        self.block3 = ResidualBlock(_STAGE2_CHANNELS, _STAGE2_CHANNELS, stride=1, device=device)

        self.pool = MaxPool2d(2)
        self.flatten = Flatten()
        self.fc = Linear(_FLATTENED_FEATURES, num_classes, device=device)

    def forward(self, x):
        h = self.relu(self.stem_bn(self.stem_conv(x)))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.pool(h)
        h = self.flatten(h)
        return self.fc(h)

    def __repr__(self) -> str:
        return f"ResNetMNIST(num_classes={self.num_classes})"


register_module(
    "ResNetMNIST",
    ResNetMNIST,
    get_config=lambda m: {"num_classes": m.num_classes},
)


def build_model(num_classes: int = _NUM_CLASSES) -> ResNetMNIST:
    """Construct a fresh, untrained `ResNetMNIST` -- see the module docstring for the shape trace."""
    return ResNetMNIST(num_classes=num_classes)


__all__ = ["ResidualBlock", "ResNetMNIST", "build_model"]
