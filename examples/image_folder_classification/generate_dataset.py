"""Generates a small synthetic shape-classification dataset (Milestone 69,
extended to mixed resolutions in Milestone 70).

`forge.data.ImageFolder`'s real consumer needs "ordinary files arranged as
a dataset," not another MNIST re-hash -- so this writes three classes of
generated shape images (`circle/`, `square/`, `triangle/`) straight to
disk as PNG files under an `ImageFolder`-compatible directory tree:

```text
root/
    circle/
        circle_0000.png
        circle_0001.png
        ...
    square/
        ...
    triangle/
        ...
```

Each image varies **position**, **scale**, **orientation** (square/triangle
rotation; circle is rotationally near-symmetric so instead varies its x/y
aspect), **background color**, **shape color**, **per-pixel noise**, and
(Milestone 70) **height and width**, each drawn independently per image from
`[MIN_IMAGE_SIZE, MAX_IMAGE_SIZE]` -- enough that the task requires real
shape recognition, not a hard-coded pixel-location shortcut, and enough that
`ImageFolder` cannot be batched by `DataLoader` without a `Resize` transform
(see `docs/development/m70-image-preprocessing.md`). Fully reproducible from
one `--seed`; nothing is downloaded. Uses only Pillow (`ImageFolder`'s own
image-decoding dependency) and NumPy -- no new Forge framework code.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ("circle", "square", "triangle")
DEFAULT_MIN_SIZE = 48
DEFAULT_MAX_SIZE = 128
DEFAULT_SAMPLES_PER_CLASS = 200
_NOISE_SIGMA = 8.0
# The background is always light and the foreground shape always
# meaningfully darker (a consistent light-on-dark polarity) -- this keeps
# the shape's exact hue/shade genuinely random per image (so color alone
# can't be memorized), while remaining a learnable edge-detection task for
# a small 2-conv-layer CNN. An earlier version drew both colors fully
# independently (either could be darker) and was empirically much harder to
# learn within a practical epoch budget on the reference hardware.
_BG_BASE = np.array([235, 235, 235])
_BG_JITTER = 12
_MIN_LUMINANCE_GAP = 90.0


def _random_color(rng: np.random.Generator) -> "tuple[int, int, int]":
    return tuple(int(c) for c in rng.integers(0, 256, size=3))


def _luminance(color: "tuple[int, int, int]") -> float:
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def _distinct_colors(rng: np.random.Generator) -> "tuple[tuple[int, int, int], tuple[int, int, int]]":
    """Pick a light background and a randomly-hued but reliably-darker foreground."""
    jitter = rng.integers(-_BG_JITTER, _BG_JITTER + 1, size=3)
    bg = tuple(int(c) for c in np.clip(_BG_BASE + jitter, 0, 255))
    bg_luminance = _luminance(bg)
    while True:
        fg = _random_color(rng)
        if bg_luminance - _luminance(fg) >= _MIN_LUMINANCE_GAP:
            return bg, fg


def _regular_polygon(
    center: "tuple[float, float]", radius: float, n_sides: int, rotation_deg: float
) -> "list[tuple[float, float]]":
    cx, cy = center
    points = []
    for i in range(n_sides):
        angle = math.radians(rotation_deg) + 2 * math.pi * i / n_sides
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _render_image(rng: np.random.Generator, shape: str, width: int, height: int) -> Image.Image:
    bg_color, fg_color = _distinct_colors(rng)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    min_dim = min(width, height)
    margin = min_dim * 0.15
    cx = rng.uniform(margin, width - margin)
    cy = rng.uniform(margin, height - margin)
    max_radius = min(cx, cy, width - cx, height - cy, min_dim * 0.42)
    radius = rng.uniform(max_radius * 0.8, max_radius)
    rotation = rng.uniform(0.0, 360.0)

    if shape == "circle":
        rx = radius * rng.uniform(0.85, 1.15)
        ry = radius * rng.uniform(0.85, 1.15)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fg_color)
    elif shape == "square":
        draw.polygon(_regular_polygon((cx, cy), radius, 4, rotation), fill=fg_color)
    elif shape == "triangle":
        draw.polygon(_regular_polygon((cx, cy), radius, 3, rotation), fill=fg_color)
    else:
        raise ValueError(f"Unknown shape {shape!r}, expected one of {CLASSES}.")

    array = np.array(img, dtype=np.uint8).astype(np.float32)
    noisy = array + rng.normal(0.0, _NOISE_SIGMA, size=array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def generate_dataset(
    root: "str | Path",
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    seed: int = 0,
) -> None:
    """Write `samples_per_class` PNGs per class under `root/<class>/`, deterministically.

    Each image's width and height are drawn independently and uniformly
    from `[min_size, max_size]` (Milestone 70) -- the dataset is
    **mixed-resolution by construction**, so `ImageFolder` cannot be
    batched by `DataLoader` without a `Resize` transform (the exact gap
    `docs/development/m70-image-preprocessing.md` documents). Re-running
    with the same arguments overwrites the same deterministic filenames
    with the same deterministic content (one shared `Generator` stream,
    classes visited in a fixed order) -- idempotent in practice.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    for shape in CLASSES:
        class_dir = root / shape
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            width = int(rng.integers(min_size, max_size + 1))
            height = int(rng.integers(min_size, max_size + 1))
            image = _render_image(rng, shape, width, height)
            image.save(class_dir / f"{shape}_{i:04d}.png")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="examples/image_folder_classification/data", help="Output directory.")
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE, help="Minimum generated image width/height.")
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE, help="Maximum generated image width/height.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    generate_dataset(args.root, args.samples_per_class, args.min_size, args.max_size, args.seed)
    total = len(CLASSES) * args.samples_per_class
    print(f"Wrote {total} images ({args.samples_per_class} per class) to '{args.root}' "
          f"(mixed resolutions in [{args.min_size}, {args.max_size}]^2, seed={args.seed}).")


if __name__ == "__main__":
    main()
