"""Milestone 102: a real multi-row consumer application built around
`forge.load_predictor()`, the reusable in-process inference workflow.

```bash
python -m examples.tabular_diabetes.app \\
    --model examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge \\
    --input '[[2, 130, 70, 25, 0, 28.5, 0.5, 35], [1, 85, 66, 29, 0, 26.6, 0.351, 31]]'
```

`infer.py` (Milestone 92) is the right tool for a single, one-off patient
row -- it calls `forge.predict_model()` once, which reloads and
reconstructs the entire artifact for that one prediction. This script is
the multi-row counterpart: a real application receiving a batch of patient
rows (a CSV upload, a loop over incoming requests) should not pay that
reload cost once per row. It loads `--model` exactly once via
`forge.load_predictor()`, then calls `predictor.predict(row)` once per row
-- demonstrating the core Milestone 102 product requirement ("load once,
predict many times") against a real, previously-trained artifact, not a
toy example.

Deliberately independent of `train.py`/`dataset.py`, exactly like
`infer.py`/`evaluate.py`: only `forge` is imported, and everything needed
(model architecture + weights, the fitted `Compose([ReplaceValue,
Normalize])` preprocessing, the `InputSchema` feature-count contract, the
class vocabulary) comes from the one `.forge` file `train.py` wrote.

`--input` is a JSON list of patient rows (each a flat list of 8 numbers, in
`FEATURE_NAMES` order -- see `dataset.py`), or a single flat row for
convenience. A structurally invalid row (wrong feature count -- Milestone
101's `InputSchema` validation, reused unchanged by the loaded predictor)
is reported inline rather than aborting the whole batch, since one bad row
in a real batch should not discard every other row's prediction.
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
                         help="A JSON list of raw patient rows, each in "
                              "[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
                              "DiabetesPedigreeFunction, Age] order (or a single flat row).")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto once (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, input_json: str, device: "str | None" = None):
    """Load `model_path` exactly once, then predict on every row in `input_json`.

    Returns `(predictor, results)`: the loaded `forge.ArtifactPredictor` (so
    a caller can also read `.task`/`.classes`/`.input_schema` without a
    second load) and a list with one entry per row -- a
    `ClassificationPrediction`/`int` (mirroring `infer.py::run()`'s own
    per-row result types) for a structurally valid row, or the caught
    `forge.DataError` itself for an invalid one.
    """
    rows = json.loads(input_json)
    if not rows or not isinstance(rows[0], list):
        rows = [rows]

    predictor = forge.load_predictor(model_path, device=device)

    results = []
    for row in rows:
        array = np.array([row], dtype=np.float32)
        try:
            results.append(predictor.predict(array)[0])
        except forge.DataError as exc:
            results.append(exc)
    return predictor, results


def main(argv=None) -> None:
    args = parse_args(argv)
    predictor, results = run(args.model, args.input, device=args.device)

    print(
        f"Loaded {args.model} once ({predictor.task} artifact"
        f"{', ' + str(len(predictor.classes)) + ' classes' if predictor.classes else ''}) "
        f"-- predicting {len(results)} row(s)."
    )
    for i, result in enumerate(results):
        if isinstance(result, forge.DataError):
            print(f"Row {i}: invalid input -- {result}")
        elif isinstance(result, int):
            print(f"Row {i}: class index {result} (no class-name vocabulary was saved with this model)")
        else:
            print(f"Row {i}: {result.label} (confidence {result.confidence:.1%})")


if __name__ == "__main__":
    main()
