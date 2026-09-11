"""Milestone 77: standalone, fresh-process inference on one digit image file.

```bash
python -m examples.mnist.infer \
    --model examples/mnist/artifacts/mnist_model.forge \
    --image path/to/some_digit.png
```

Mirrors `examples/image_folder_classification/infer.py`'s role and shape
exactly, extended to Forge's original flagship example for the first time:
this script is deliberately independent of `train.py` -- it does not import
`build_transform()`, `build_model()`, or `MNISTDataset`, only
`forge.load_model()`, `forge.load_preprocessing()`, `forge.load_classes()`,
`forge.predict()`, and `forge.interpret_classification()`. Nothing here
depends on any in-memory state `train.py` happened to build -- everything a
caller needs (model architecture + weights, preprocessing configuration, and
the digit-index-to-label vocabulary) comes from the one `.forge` file
`train.py` wrote (see `docs/development/m77-mnist-family-portable-artifacts.md`).

The one piece this script cannot reuse from `forge.data.ImageFolder` is its
`_load_image` decode helper: that helper always converts to RGB (3
channels, matching `ImageFolder`'s own documented convention), but MNIST
models expect a single grayscale channel. `_load_digit_image()` below is
the ~10-line, MNIST-shaped equivalent -- ordinary application code, not a
framework capability, mirroring the way `examples/mnist/dataset.py` already
implements its own IDX-format decode without any framework help.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import forge
from forge.serialization import load_classes, load_model, load_preprocessing
from forge.training import ClassificationPrediction, interpret_classification, predict

_SIZE = (28, 28)


def _load_digit_image(path: Path) -> forge.Tensor:
    """Decode `path` into a `(1, 28, 28)` float32 Tensor, raw `[0, 255]` range.

    Converts to grayscale (`"L"`) and resizes to MNIST's fixed `28x28` if
    the source image is a different size, so this works on an arbitrary
    photo of a digit, not only an image already shaped exactly like an IDX
    sample.
    """
    with Image.open(path) as img:
        gray = img.convert("L")
        if gray.size != _SIZE:
            gray = gray.resize(_SIZE, Image.BILINEAR)
        array = np.array(gray, dtype=np.float32)
    return forge.Tensor(array[np.newaxis, :, :])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., preprocessing=..., classes=...).")
    parser.add_argument("--image", required=True, help="Path to one grayscale digit image file to classify.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, image_path: str, device: "str | None" = None) -> "ClassificationPrediction | int":
    """Load `model_path`'s model + preprocessing + classes, classify `image_path`.

    Returns a `ClassificationPrediction` (`.label`, `.confidence`) if
    `model_path` was saved with `classes=...`, otherwise the raw predicted
    class index as an `int`. Raises `forge.PersistenceError` if `model_path`
    was never saved with `preprocessing=` -- there is nothing to reproduce
    automatically in that case, matching
    `examples.image_folder_classification.infer.run()`'s own policy exactly.
    """
    preprocessing = load_preprocessing(model_path)
    if preprocessing is None:
        raise forge.PersistenceError(
            f"'{model_path}' was saved with no preprocessing configuration (see "
            "save_model(..., preprocessing=...)) -- infer.py has no automatic way to prepare "
            "the input image for this model."
        )
    model = load_model(model_path, device=device)

    raw = _load_digit_image(Path(image_path))
    prepared = preprocessing(raw)
    batch = prepared.reshape(1, *prepared.shape)
    output = predict(model, batch)

    classes = load_classes(model_path)
    if classes is not None:
        return interpret_classification(output, classes)[0]
    return int(np.argmax(output.numpy(), axis=1)[0])


def main(argv=None) -> None:
    args = parse_args(argv)
    result = run(args.model, args.image, device=args.device)
    if isinstance(result, ClassificationPrediction):
        print(f"Prediction: digit {result.label}")
        print(f"Confidence: {result.confidence:.1%}")
    else:
        print(f"Prediction: class index {result} "
              "(no class-name vocabulary was saved with this model)")


if __name__ == "__main__":
    main()
