"""A small CNN for the `image_folder_classification` example (Milestone 69,
input resolution updated in Milestone 70).

Built entirely from existing `forge.nn` Modules -- no new layer types.
Three differences from `examples/mnist/model.py`'s plain architecture: the
input channel count (3, RGB, vs. MNIST's 1) and class count (3 shapes vs.
10 digits); a `BatchNorm2d` after each `Conv2d` (Milestone 53's existing
layer, reused unmodified); and a `Dropout(0.3)` before the final `Linear`
(Milestone 4's existing layer). This dataset's per-image background/
foreground colors are drawn independently per sample (see
`generate_dataset.py`), so raw per-channel activation statistics vary far
more across the batch than MNIST's near-uniform background -- `BatchNorm2d`
measurably closed that gap (val accuracy stuck near the 33% chance baseline
without it). With `BatchNorm2d` alone the model still overfit sharply
(near-100% train accuracy, a much lower validation accuracy) at this
dataset's size; `Dropout` closed most of that remaining gap. Both findings
are from Milestone 69's own training runs -- see
`docs/development/m69-image-folder.md`'s Training Results section. Kept
small deliberately to train practically on the reference hardware
(`docs/development/development-environment.md`).

Milestone 70 made the source images mixed-resolution and introduced
`forge.data.transforms.Resize` to normalize every sample to a fixed
`(64, 64)` before it reaches this model -- see
`docs/development/m70-image-preprocessing.md`. Only the flattened feature
count below changed to match; the architecture itself is unchanged.

```text
(N, 3, 64, 64)
    -> Conv2d(3, 16, k=3)   -> (N, 16, 62, 62)
    -> BatchNorm2d(16)
    -> ReLU
    -> MaxPool2d(2)         -> (N, 16, 31, 31)
    -> Conv2d(16, 32, k=3)  -> (N, 32, 29, 29)
    -> BatchNorm2d(32)
    -> ReLU
    -> MaxPool2d(2)         -> (N, 32, 14, 14)
    -> Flatten              -> (N, 6272)
    -> Linear(6272, 64)
    -> ReLU
    -> Dropout(0.3)
    -> Linear(64, num_classes)
```
"""

from __future__ import annotations

from forge.nn import BatchNorm2d, Conv2d, Dropout, Flatten, Linear, MaxPool2d, ReLU, Sequential

_FLATTENED_FEATURES = 32 * 14 * 14


def build_model(num_classes: int = 3) -> Sequential:
    """Construct a fresh, untrained shape-classification CNN for `(N, 3, 32, 32)` input."""
    return Sequential(
        Conv2d(3, 16, kernel_size=3),
        BatchNorm2d(16),
        ReLU(),
        MaxPool2d(2),
        Conv2d(16, 32, kernel_size=3),
        BatchNorm2d(32),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(_FLATTENED_FEATURES, 64),
        ReLU(),
        Dropout(0.3),
        Linear(64, num_classes),
    )


__all__ = ["build_model"]
