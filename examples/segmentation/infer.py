"""Milestone 84: standalone, fresh-process inference on one new image file.

```bash
python -m examples.segmentation.infer \
    --model examples/segmentation/artifacts/segmentation_model.forge \
    --image examples/segmentation/artifacts/segmentation_new_image.png \
    --output examples/segmentation/artifacts/segmentation_new_predicted_mask.png
```

This script is deliberately independent of `train.py`: it does not import
`build_transform()`, `build_model()`, or `SegmentationDataset` -- only
`forge.predict_model()` (Milestone 86), which itself composes
`inspect_model()`, `load_model()`, `load_preprocessing()`,
`ImageFolder._load_image()`, `predict()`, and the same raw-output-to-binary
-mask threshold `examples/segmentation/metrics.py` already defines -- via
`predict_image_artifact()` (Milestone 84), the workflow this artifact's own
metadata describes. Nothing here depends on any in-memory state `train.py`
happened to build -- everything a caller needs (model architecture +
weights, and the preprocessing pipeline that rescales a freshly decoded
image into the model's trained input range) comes from the one `.forge`
file `train.py` wrote.

This is the concrete Milestone 84/86 workflow: a developer trains and saves
a segmentation model in one process, then -- potentially days later, in a
completely separate process, on a completely new image -- loads it here and
writes out a real predicted-mask image file, with no manual
load-preprocess-predict-threshold-save reconstruction required, and no need
to already know this particular artifact is a segmentation one rather than
a classification or regression one (see `forge/training/inference.py::
predict_model()`).
"""

from __future__ import annotations

import argparse

import forge
from forge.data import save_image


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., preprocessing=...).")
    parser.add_argument("--image", required=True, help="Path to one new image file to segment.")
    parser.add_argument("--output", required=True, help="Where to write the predicted mask as a PNG.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, image_path: str, output_path: str, device: "str | None" = None):
    """Predict a mask for `image_path` with the artifact at `model_path`, and save it to `output_path`.

    A thin wrapper over `forge.predict_model()` (Milestone 86) -- this script
    no longer needs to already know `model_path` is a segmentation artifact
    specifically; `predict_model()` determines that from the artifact's own
    persisted metadata and delegates to `predict_image_artifact()`
    (Milestone 84) unchanged. See that function's docstring for the full
    behavior: returns the predicted `{0, 1}`-valued mask `Tensor`, already
    saved to `output_path` via `forge.data.save_image()`. Raises
    `forge.PersistenceError` if `model_path` was never saved with
    `preprocessing=`.
    """
    prediction = forge.predict_model(model_path, image_path, device=device)
    save_image(prediction, output_path)
    return prediction


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args.model, args.image, args.output, device=args.device)
    print(f"Saved predicted mask -> {args.output}")


if __name__ == "__main__":
    main()
