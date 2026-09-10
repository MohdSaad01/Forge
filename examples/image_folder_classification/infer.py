"""Milestone 71/72: standalone, fresh-process inference on one image file.

```bash
python -m examples.image_folder_classification.infer \
    --model examples/image_folder_classification/artifacts/image_folder_model.forge \
    --image path/to/some_photo.jpg
```

This script is deliberately independent of `train.py`: it does not import
`build_transform()`, `build_model()`, or `ImageFolder` as a *dataset* --
only `forge.load_model()`, `forge.load_preprocessing()`, `forge.load_classes()`,
`forge.predict()`, and `forge.interpret_classification()`, plus
`ImageFolder._load_image` reused as its documented single-image decode
helper (see `forge/data/image_folder.py`'s own docstring: "the right place
to reuse... if a caller ever needs to preprocess one arbitrary image file
the exact same way `ImageFolder` does"). Nothing here depends on any
in-memory state `train.py` happened to build -- everything a caller needs
(model architecture + weights, preprocessing configuration, and class-name
vocabulary) comes from the one `.forge` file `train.py` wrote.

This is the concrete Milestone 71/72 workflow: a developer trains and saves
a model in one process, then -- potentially days later, in a completely
separate process, on a completely new image -- loads it here and gets a
human-readable prediction without having to remember or re-implement the
`Resize`/`Normalize` steps `train.py` used, or keep a hand-written
`classes.json` sidecar file next to the model (Milestone 72 -- before this,
`--classes path/to/classes.json` was a required separate argument; see
`docs/development/m72-classification-metadata.md`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import forge
from forge.data import ImageFolder
from forge.serialization import load_classes, load_model, load_preprocessing
from forge.training import ClassificationPrediction, interpret_classification, predict


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., preprocessing=..., classes=...).")
    parser.add_argument("--image", required=True, help="Path to one image file to classify.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, image_path: str, device: "str | None" = None) -> "ClassificationPrediction | int":
    """Load `model_path`'s model + preprocessing + classes, classify `image_path`.

    Returns a `ClassificationPrediction` (`.label`, `.confidence`) if
    `model_path` was saved with `classes=...`, otherwise the raw predicted
    class index as an `int`. Raises `forge.PersistenceError` if `model_path`
    was never saved with `preprocessing=` (i.e. `load_preprocessing()`
    returns `None`) -- there is nothing to reproduce automatically in that
    case, and silently skipping preprocessing would be exactly the
    hidden-assumption failure mode Milestone 71 closed.
    """
    preprocessing = load_preprocessing(model_path)
    if preprocessing is None:
        raise forge.PersistenceError(
            f"'{model_path}' was saved with no preprocessing configuration (see "
            "save_model(..., preprocessing=...)) -- infer.py has no automatic way to prepare "
            "the input image for this model."
        )
    model = load_model(model_path, device=device)

    raw = ImageFolder._load_image(Path(image_path))
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
        print(f"Prediction: {result.label}")
        print(f"Confidence: {result.confidence:.1%}")
    else:
        print(f"Prediction: class index {result} "
              "(no class-name vocabulary was saved with this model)")


if __name__ == "__main__":
    main()
