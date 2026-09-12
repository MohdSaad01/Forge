"""Milestone 71/72/82: standalone, fresh-process inference on one image file.

```bash
python -m examples.image_folder_classification.infer \
    --model examples/image_folder_classification/artifacts/image_folder_model.forge \
    --image path/to/some_photo.jpg
```

This script is deliberately independent of `train.py`: it does not import
`build_transform()`, `build_model()`, or `ImageFolder` as a *dataset* -- only
`forge.training.predict_artifact()` (Milestone 82), which itself composes
`load_model()`, `load_preprocessing()`, `load_classes()`,
`ImageFolder._load_image()` (reused as its documented single-image decode
helper -- see `forge/data/image_folder.py`'s own docstring), `predict()`, and
`interpret_classification()`. Nothing here depends on any in-memory state
`train.py` happened to build -- everything a caller needs (model architecture
+ weights, preprocessing configuration, and class-name vocabulary) comes from
the one `.forge` file `train.py` wrote.

This is the concrete Milestone 71/72/82 workflow: a developer trains and
saves a model in one process, then -- potentially days later, in a
completely separate process, on a completely new image -- loads it here and
gets a human-readable prediction without having to remember or re-implement
the `Resize`/`Normalize` steps `train.py` used, keep a hand-written
`classes.json` sidecar file next to the model (Milestone 72), or manually
re-assemble the load-preprocess-predict-interpret pipeline at all
(Milestone 82 -- see `forge/training/inference.py::predict_artifact()`).
"""

from __future__ import annotations

import argparse

import forge
from forge.training import ClassificationPrediction


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., preprocessing=..., classes=...).")
    parser.add_argument("--image", required=True, help="Path to one image file to classify.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, image_path: str, device: "str | None" = None) -> "ClassificationPrediction | int":
    """Classify `image_path` with the artifact at `model_path`.

    A thin wrapper over `forge.predict_artifact()` (Milestone 82) -- see that
    function's docstring for the full behavior: returns a
    `ClassificationPrediction` (`.label`, `.confidence`) if `model_path` was
    saved with `classes=...`, otherwise the raw predicted class index as an
    `int`; raises `forge.PersistenceError` if `model_path` was never saved
    with `preprocessing=`.
    """
    return forge.predict_artifact(model_path, image_path, device=device)


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
