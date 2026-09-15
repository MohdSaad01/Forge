"""Milestone 98: train and save one configured experiment variant.

```bash
python -m experiment.run_experiment --name baseline --hidden 32 16 --lr 1e-3 --init-seed 0
python -m experiment.run_experiment --name capacity --hidden 64 32 --lr 1e-3 --init-seed 0
python -m experiment.run_experiment --name high_lr  --hidden 32 16 --lr 1e-2 --init-seed 0
```

This script is the "change one variable, retrain" half of the M98 loop. It
is intentionally a single parameterized script rather than a copy-pasted
`train.py` per variant -- the M98 brief's Section 11 explicitly asks whether
Forge's existing training workflow lets a developer represent a change of
one parameter as *data* rather than as duplicated code. It does: everything
that varies between experiments (hidden layer sizes, learning rate, epoch
count, init seed) is a plain CLI argument, and the model architecture itself
is built here from unmodified public `forge.nn` primitives
(`Linear`/`ReLU`/`Sequential`) -- no change to `examples/tabular_diabetes/
model.py` or to `forge/` was needed to vary capacity or learning rate.

## What is held fixed across every experiment (by design, not by accident)

- **The train/val/test split** (`examples.tabular_diabetes.dataset.
  make_datasets(seed=SPLIT_SEED)`, `SPLIT_SEED = 0`, hardcoded below, not a
  CLI flag) -- every experiment must be evaluated against the identical
  held-out rows (M98 brief Section 7), so the split seed is deliberately not
  configurable here. This is also exactly the split
  `examples/tabular_diabetes/data/diabetes_holdout_eval.csv` (M97) already
  captures, so `experiment/compare.py` can evaluate straight from that file.
- **The fitted preprocessing** -- `make_datasets()` fits `ReplaceValue`
  medians and `Normalize` mean/std from the training split only, so no
  variant ever sees validation/test statistics (M98 brief Section 8).

## What varies (CLI-controlled, recorded in the output JSON)

`--hidden`, `--lr`, `--epochs`, `--batch-size`, `--init-seed` (governs
`forge.random.seed()` parameter initialization and the DataLoader shuffle
generator only -- never the split).

## What this script deliberately does NOT do

It does not evaluate the saved artifact against the held-out test set. That
step is `experiment/compare.py`'s job, run afterward as a genuinely separate
step against the saved `.forge` file alone -- keeping training and
evaluation isolated (M98 brief Section 19), exactly like
`examples/tabular_diabetes/evaluate.py` already keeps itself independent of
`train.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import forge
from forge.data import DataLoader
from forge.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from forge.optim import Adam
from forge.training import Accuracy

from examples.tabular_diabetes.dataset import CLASS_NAMES, N_CLASSES, N_FEATURES, make_datasets

SPLIT_SEED = 0  # Not a CLI flag -- every experiment must share one held-out split. See module docstring.

_HERE = Path(__file__).parent


def build_variant_model(hidden: "list[int]") -> Sequential:
    """Build an MLP with the given two hidden-layer widths, from public `forge.nn` primitives only.

    Mirrors `examples/tabular_diabetes/model.py::build_model()`'s fixed
    32/16 shape but with widths supplied as data, demonstrating that varying
    model capacity across experiments needs no change to Forge or to the
    example's own `model.py`.
    """
    h1, h2 = hidden
    return Sequential(
        Linear(N_FEATURES, h1),
        ReLU(),
        Linear(h1, h2),
        ReLU(),
        Linear(h2, N_CLASSES),
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Experiment identifier; used for the artifact and result file names.")
    parser.add_argument("--hidden", type=int, nargs=2, default=[32, 16], metavar=("H1", "H2"), help="Two hidden-layer widths.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--init-seed", type=int, default=0, help="Seed for forge.random (param init) and DataLoader shuffling. Never the split.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default=str(_HERE / "artifacts"))
    parser.add_argument("--results-dir", default=str(_HERE / "results"))
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch training output.")
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.name}.forge"

    train_ds, val_ds, test_ds, stats = make_datasets(seed=SPLIT_SEED)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    loss_fn = CrossEntropyLoss()

    forge.random.seed(args.init_seed)
    data_loader_rng = np.random.default_rng(args.init_seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_loader_rng)

    model = build_variant_model(args.hidden).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    query_x, _ = test_ds[0]
    query_x_batch = query_x.to(args.device).reshape(1, N_FEATURES)

    start = time.perf_counter()
    train_result = forge.train_and_save(
        model, train_loader,
        loss=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        validation_dataset=val_loader,
        device=args.device,
        metrics=[Accuracy()],
        path=str(model_path),
        sample=query_x_batch,
        preprocessing=stats["transform"],
        classes=CLASS_NAMES,
        task="tabular_classification",
        verbose=not args.quiet,
    )
    duration = time.perf_counter() - start
    history = train_result.history

    record = {
        "name": args.name,
        "hidden": list(args.hidden),
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "init_seed": args.init_seed,
        "split_seed": SPLIT_SEED,
        "device": args.device,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "majority_baseline": stats["majority_baseline"],
        "train_loss_first": history[0].train_loss,
        "train_loss_last": history[-1].train_loss,
        "val_loss_last": history[-1].val_loss,
        "val_accuracy_last": history[-1].val_metrics["accuracy"],
        "duration_sec": duration,
        "artifact_path": str(model_path),
    }

    result_path = results_dir / f"{args.name}.json"
    result_path.write_text(json.dumps(record, indent=2))

    print(f"[{args.name}] hidden={args.hidden} lr={args.lr} init_seed={args.init_seed} "
          f"-> val_accuracy={record['val_accuracy_last']:.1%} in {duration:.2f}s")
    print(f"  artifact: {model_path}")
    print(f"  result:   {result_path}")
    return record


def main(argv=None) -> None:
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
