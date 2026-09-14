"""Milestone 92: standalone, fresh-process prediction on one new raw patient row.

```bash
python -m examples.tabular_diabetes.infer \\
    --model examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge \\
    --input '[1, 85, 66, 29, 0, 26.6, 0.351, 31]'
```

This script is deliberately independent of `train.py`: it does not import
`build_model()` or `dataset.py` -- only `forge.predict_model()`, which
composes `inspect_model()`, `load_model()`, `load_preprocessing()`,
`predict()`, `load_classes()`, and `interpret_classification()` via
`predict_tabular_classification_artifact()`. Everything a caller needs
(model architecture + weights, the fitted `Compose([ReplaceValue,
Normalize])` preprocessing pipeline, and the class vocabulary) comes from
the one `.forge` file `train.py` wrote -- including the missing-value
imputation, which is why `--input` can be a literal raw row (sentinel zeros
and all) rather than something the caller must pre-clean by hand. See
`examples/tabular_diabetes/README.md`.

`--input` is a literal JSON array on the command line (a flat list for one
sample, or a nested list for an already-batched query), matching
`examples/tabular_classification/infer.py`'s convention.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import forge


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., task='tabular_classification').")
    parser.add_argument("--input", required=True,
                         help="A raw patient row as a JSON list, in "
                              "[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
                              "DiabetesPedigreeFunction, Age] order (or a nested list for multiple rows).")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, input_json: str, device: "str | None" = None) -> list:
    """Predict on `input_json` (a JSON-encoded flat or nested list of numbers) with the artifact at `model_path`.

    A thin wrapper over `forge.predict_model()` -- returns one
    `ClassificationPrediction`/`int` per input row.
    """
    raw = json.loads(input_json)
    array = np.array(raw, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return forge.predict_model(model_path, array, device=device)


def main(argv=None) -> None:
    args = parse_args(argv)
    results = run(args.model, args.input, device=args.device)
    for i, result in enumerate(results):
        prefix = f"Sample {i}: " if len(results) > 1 else ""
        if isinstance(result, int):
            print(f"{prefix}Prediction: class index {result} (no class-name vocabulary was saved with this model)")
        else:
            print(f"{prefix}Prediction: {result.label} (confidence {result.confidence:.1%})")


if __name__ == "__main__":
    main()
