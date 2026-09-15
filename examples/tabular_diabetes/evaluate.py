"""Milestone 97: post-training evaluation of a saved .forge artifact on unseen labeled data.

```bash
python -m examples.tabular_diabetes.evaluate \\
    --model examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge \\
    --data examples/tabular_diabetes/data/diabetes_holdout_eval.csv
```

Answers the question `train.py` only ever answered as a side effect of
training (`Trainer(model, loss_fn, optimizer, ...).evaluate(test_loader)`,
called with the live model/loss/optimizer/loader `train.py` already has in
scope): **given just the saved artifact and a CSV of unseen labeled rows,
how accurate is it, and is that meaningfully better than guessing the
majority class?**

Deliberately independent of `train.py`/`dataset.py`/`model.py`, exactly like
`infer.py` -- it does not import `build_model()`, `make_datasets()`, or
`CLASS_NAMES`. Everything it needs (model architecture + weights, the fitted
`Compose([ReplaceValue, Normalize])` preprocessing, and the class vocabulary)
comes from the `.forge` file itself.

## Why this composes `load_model()` + `load_preprocessing()` + `predict()` +
`forge.training.Accuracy`, not `Trainer.evaluate()`

`Trainer.evaluate()` (`forge/training/trainer.py`) looks like the obvious
tool -- it is even named `evaluate` -- but it requires constructing a `Loss`
and an `Optimizer` that are never actually used for anything (evaluation
never calls `optimizer.step()`), and, more importantly, it runs whatever
batches its `DataLoader` yields through the model **as-is**: it has no
persisted-preprocessing hook, because it predates `preprocessing=`
(Milestone 71) by many milestones and was never meant to consume a `.forge`
file directly. Feeding this dataset's raw CSV rows (sentinel zeros and all)
to `Trainer.evaluate()` without separately fetching and applying
`load_preprocessing()` first produces a real number with no error --
37.0% accuracy, *worse* than the 62.3% majority-class baseline -- silently,
because the sentinel-zero missing-value encoding this exact dataset uses
(see `dataset.py`'s module docstring) is never imputed or standardized.
`Trainer.evaluate()`'s own docstring was updated (Milestone 97) to say this
explicitly. The composition below reuses the artifact's own persisted
preprocessing the same way every `predict_*_artifact()` function already
does, which is the one thing that makes the resulting accuracy trustworthy.

## Held-out data

`data/diabetes_holdout_eval.csv` is exactly `dataset.make_datasets(seed=0)`'s
`test_ds` split (154 rows), exported once to a plain CSV -- the same rows
`train.py --seed 0`'s own final `Trainer(...).evaluate(test_loader)` line
reports accuracy for, but here reachable from a genuinely separate process
with no producer code at all. Reproducible via:

```python
from examples.tabular_diabetes.dataset import make_datasets, load_raw
_, _, test_ds, _ = make_datasets(seed=0)
X, y = load_raw()
# test_ds.indices selects the held-out rows from X/y.
```
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import numpy as np

import forge
from forge.training import Accuracy


@dataclass(frozen=True)
class EvaluationSummary:
    """The small, user-useful result this script prints -- not a new Forge API.

    Bundles exactly what a developer needs to decide whether to trust the
    model: how many rows were evaluated, its accuracy, and how that compares
    to the simplest legitimate baseline for this task (always predict the
    majority class).
    """

    model_path: str
    samples: int
    accuracy: float
    baseline_accuracy: float

    @property
    def improvement_pp(self) -> float:
        return (self.accuracy - self.baseline_accuracy) * 100.0


def load_labeled_csv(path: str) -> "tuple[np.ndarray, np.ndarray]":
    """Parse a CSV of feature columns followed by one integer label column.

    Ordinary stdlib `csv` + NumPy -- application responsibility, matching
    `dataset.py::load_raw()`'s own precedent (Milestone 92): Forge is not
    responsible for parsing every external tabular file format.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    n_columns = len(header)
    data = np.array(rows, dtype=np.float32).reshape(len(rows), n_columns)
    X = data[:, :-1]
    y = data[:, -1].astype(np.int64)
    return X, y


def evaluate_artifact(model_path: str, data_path: str, device: "str | None" = None) -> EvaluationSummary:
    """Evaluate `model_path` on the labeled rows in `data_path`, reusing the artifact's own persisted preprocessing.

    Composes exactly `forge.load_model()` + `forge.load_preprocessing()` +
    `forge.predict()` + `forge.training.Accuracy` -- each an existing,
    unmodified public call. No Trainer, no Loss, no Optimizer: none of them
    are needed to run a forward pass and compare it against ground truth.
    """
    X, y = load_labeled_csv(data_path)

    model = forge.load_model(model_path, device=device)
    preprocessing = forge.load_preprocessing(model_path)

    x = forge.Tensor(X)  # stays on CPU -- preprocessing (ReplaceValue/Normalize) is host-side
    if preprocessing is not None:
        x = preprocessing(x)

    output = forge.predict(model, x)  # predict() moves x to model's device internally

    metric = Accuracy()
    metric.update(output, y)
    accuracy = metric.compute()

    majority_class = int(np.bincount(y).argmax())
    baseline_accuracy = float((y == majority_class).mean())

    return EvaluationSummary(
        model_path=model_path, samples=len(y), accuracy=accuracy, baseline_accuracy=baseline_accuracy,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a .forge file written with task='tabular_classification'.")
    parser.add_argument("--data", required=True, help="Path to a CSV of unseen labeled rows (feature columns, then an integer label column).")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device to load the model onto (default: whatever device it was saved from).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = evaluate_artifact(args.model, args.data, device=args.device)

    print(f"Model:             {summary.model_path}")
    print(f"Test samples:      {summary.samples}")
    print(f"Accuracy:          {summary.accuracy:.1%}")
    print(f"Baseline accuracy: {summary.baseline_accuracy:.1%}")
    print(f"Improvement:       {summary.improvement_pp:+.1f} percentage points")
    if summary.accuracy > summary.baseline_accuracy:
        print("-> model is meaningfully better than the majority-class baseline.")
    else:
        print("-> model does NOT beat the majority-class baseline.")


if __name__ == "__main__":
    main()
