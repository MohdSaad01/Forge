"""Milestone 71: standalone, fresh-process inference on one image file.

```bash
python -m examples.image_folder_classification.infer \
    --model examples/image_folder_classification/artifacts/image_folder_model.forge \
    --classes examples/image_folder_classification/artifacts/classes.json \
    --image path/to/some_photo.jpg
```

This script is deliberately independent of `train.py`: it does not import
`build_transform()`, `build_model()`, or `ImageFolder` as a *dataset* --
only `forge.load_model()`, `forge.load_preprocessing()`, and `forge.predict()`,
plus `ImageFolder._load_image` reused as its documented single-image decode
helper (see `forge/data/image_folder.py`'s own docstring: "the right place
to reuse... if a caller ever needs to preprocess one arbitrary image file
the exact same way `ImageFolder` does"). Nothing here depends on any
in-memory state `train.py` happened to build -- everything a caller needs
(model architecture + weights + preprocessing configuration) comes from the
one `.forge` file `train.py` wrote.

This is the concrete Milestone 71 workflow: a developer trains and saves a
model in one process, then -- potentially days later, in a completely
separate process, on a completely new image -- loads it here and gets a
prediction without having to remember or re-implement the `Resize`/
`Normalize` steps `train.py` used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import forge
from forge.data import ImageFolder
from forge.serialization import load_model, load_preprocessing
from forge.training import predict


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., preprocessing=...).")
    parser.add_argument("--image", required=True, help="Path to one image file to classify.")
    parser.add_argument("--classes", default=None,
                         help="Optional path to a JSON list of class names (train.py writes classes.json "
                              "alongside the model); if omitted or missing, the raw class index is printed.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, image_path: str, classes_path: "str | None" = None, device: "str | None" = None) -> str:
    """Load `model_path`'s model + preprocessing, classify `image_path`, and return the predicted label.

    Returns the human-readable class name if `classes_path` resolves to a
    valid JSON list, otherwise the predicted class index as a string. Raises
    `forge.PersistenceError` if `model_path` was never saved with
    `preprocessing=` (i.e. `load_preprocessing()` returns `None`) -- there is
    nothing to reproduce automatically in that case, and silently skipping
    preprocessing would be exactly the hidden-assumption failure mode this
    milestone closes.
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
    predicted_idx = int(np.argmax(output.numpy(), axis=1)[0])

    if classes_path is not None and Path(classes_path).is_file():
        classes = json.loads(Path(classes_path).read_text())
        return str(classes[predicted_idx])
    return str(predicted_idx)


def main(argv=None) -> None:
    args = parse_args(argv)
    label = run(args.model, args.image, classes_path=args.classes, device=args.device)
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
