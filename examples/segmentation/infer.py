"""Milestone 84: standalone, fresh-process inference on one new image file.

```bash
python -m examples.segmentation.infer \
    --model examples/segmentation/artifacts/segmentation_model.forge \
    --image examples/segmentation/artifacts/segmentation_new_image.png \
    --output examples/segmentation/artifacts/segmentation_new_predicted_mask.png
```

This script is deliberately independent of `train.py`: it does not import
`build_transform()`, `build_model()`, or `SegmentationDataset` -- only
`forge.predict_image_artifact()` (Milestone 84), which itself composes
`load_model()`, `load_preprocessing()`, `ImageFolder._load_image()`,
`predict()`, and the same raw-output-to-binary-mask threshold
`examples/segmentation/metrics.py` already defines. Nothing here depends on
any in-memory state `train.py` happened to build -- everything a caller
needs (model architecture + weights, and the preprocessing pipeline that
rescales a freshly decoded image into the model's trained input range)
comes from the one `.forge` file `train.py` wrote.

This is the concrete Milestone 84 workflow: a developer trains and saves a
segmentation model in one process, then -- potentially days later, in a
completely separate process, on a completely new image -- loads it here and
writes out a real predicted-mask image file, with no manual
load-preprocess-predict-threshold-save reconstruction required.
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

    A thin wrapper over `forge.predict_image_artifact()` (Milestone 84) --
    see that function's docstring for the full behavior: returns the
    predicted `{0, 1}`-valued mask `Tensor`, already saved to `output_path`
    via `forge.data.save_image()`. Raises `forge.PersistenceError` if
    `model_path` was never saved with `preprocessing=`.
    """
    prediction = forge.predict_image_artifact(model_path, image_path, device=device)
    save_image(prediction, output_path)
    return prediction


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args.model, args.image, args.output, device=args.device)
    print(f"Saved predicted mask -> {args.output}")


if __name__ == "__main__":
    main()
