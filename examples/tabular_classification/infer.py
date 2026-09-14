"""Milestone 91: standalone, fresh-process prediction on one new raw tabular sample.

```bash
python -m examples.tabular_classification.infer \
    --model examples/tabular_classification/artifacts/tabular_classification_model.forge \
    --input '[0.5, -1.2, 0.3, 1.8, -0.4, 2.0, -1.5, 0.9, 0.1, -0.2]'
```

This script is deliberately independent of `train.py`: it does not import
`build_model()` or `dataset.py` -- only `forge.predict_model()` (Milestone
86, extended in Milestone 91), which itself composes `inspect_model()`,
`load_model()`, `load_preprocessing()`, `predict()`, `load_classes()`, and
`interpret_classification()` via `predict_tabular_classification_artifact()`
(Milestone 91), the workflow this artifact's own `task="tabular_classification"`
metadata describes. Nothing here depends on any in-memory state `train.py`
happened to build -- everything a caller needs (model architecture +
weights, the fitted `Normalize` feature-standardization transform, and the
class vocabulary) comes from the one `.forge` file `train.py` wrote.

`--input` is a literal JSON array on the command line (a flat list for one
sample, or a nested list for an already-batched query), not a file path --
mirroring `examples/char_rnn/infer.py`'s "seed is text, not a file" choice
for the one other task whose natural input is short enough to pass directly
as an argument, rather than `forge model predict`'s own JSON-*file*
convention (this script is a Python-API example, not the CLI).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import forge


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written by save_model(..., task='tabular_classification').")
    parser.add_argument("--input", required=True, help="A raw feature vector as a JSON list, e.g. '[0.5, -1.2, ...]' (or a nested list for multiple samples).")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def run(model_path: str, input_json: str, device: "str | None" = None) -> list:
    """Predict on `input_json` (a JSON-encoded flat or nested list of numbers) with the artifact at `model_path`.

    A thin wrapper over `forge.predict_model()` -- this script does not need
    to already know `model_path` is a tabular classification artifact
    specifically; `predict_model()` determines that from the artifact's own
    persisted `task="tabular_classification"` metadata and delegates to
    `predict_tabular_classification_artifact()` (Milestone 91) unchanged.
    Returns one `ClassificationPrediction`/`int` per input row -- see that
    function's own "one result per row" docstring paragraph.
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
