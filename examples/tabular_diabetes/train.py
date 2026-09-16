"""Forge Milestone 92: real-dataset tabular classification, end-to-end.

```text
diabetes.csv -> load_raw() -> make_datasets() -> DataLoader
    -> MLP (Linear/ReLU/Linear/ReLU/Linear) -> CrossEntropyLoss -> Adam
    -> forge.train_and_save() -> forge.predict_model()
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`) -- this script adds
no framework logic of its own, only example wiring, exactly like
`examples/tabular_classification/train.py`. See
`examples/tabular_diabetes/README.md` for prerequisites and expected
behavior, and `docs/development/m92-real-dataset-ingestion.md` for why this
example exists and what it found.

No `--resume`/checkpoint path, matching `tabular_classification` -- this
workload trains in well under a second, so interrupted-training resume is
not a realistic concern.

## Determinism

`forge.random.seed(args.seed)` governs `Linear` parameter initialization;
`make_datasets()`'s own `numpy.random.default_rng(args.seed + 1)` governs
the train/val/test `random_split` permutation; `DataLoader`'s explicit
`numpy.random.Generator` governs batch shuffling.

## Usage

```bash
python -m examples.tabular_diabetes.train --epochs 60 --device cpu
python -m examples.tabular_diabetes.train --epochs 60 --device cuda
```
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import forge
from forge.data import DataLoader
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.training import Accuracy, EarlyStopping

try:
    from .dataset import CLASS_NAMES, N_FEATURES, load_raw, make_datasets
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/tabular_diabetes/train.py`)
    from dataset import CLASS_NAMES, N_FEATURES, load_raw, make_datasets
    from model import build_model


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random, the split, and DataLoader shuffling.")
    parser.add_argument("--output-dir", default="examples/tabular_diabetes/artifacts", help="Where to write the model file.")
    parser.add_argument(
        "--early-stopping", action="store_true",
        help="Monitor val_loss and stop once it stops improving, restoring the best-epoch model "
             "(Milestone 100). Off by default, matching this example's pre-Milestone-100 behavior.",
    )
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience (only used with --early-stopping).")
    parser.add_argument("--min-delta", type=float, default=0.0, help="EarlyStopping min_delta (only used with --early-stopping).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tabular_diabetes_model.forge"

    print("Loading real Pima Indians Diabetes CSV ...")
    train_ds, val_ds, test_ds, stats = make_datasets(args.val_fraction, args.test_fraction, seed=args.seed)
    print(f"train: {len(train_ds)} samples {stats['train_class_counts']}, "
          f"val: {len(val_ds)} samples, test: {len(test_ds)} samples, features: {N_FEATURES}, classes: {CLASS_NAMES}")
    print(f"Majority-class baseline (always predict '{CLASS_NAMES[0]}'): accuracy = {stats['majority_baseline']:.1%}")
    print(f"Fitted missing-value fill (training-split medians): {dict(zip(['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'], stats['fill']))}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    loss_fn = CrossEntropyLoss()

    query_x, query_y = test_ds[0]
    query_x_batch = query_x.to(args.device).reshape(1, N_FEATURES)

    forge.random.seed(args.seed)
    data_loader_rng = np.random.default_rng(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_loader_rng)
    model = build_model().to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, restore_best=True)
        print(f"Early stopping enabled: monitor=val_loss, patience={args.patience}, min_delta={args.min_delta}")

    start = time.perf_counter()
    train_result = forge.train_and_save(
        model, train_loader,
        loss=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        validation_dataset=val_loader,
        device=args.device,
        metrics=[Accuracy()],
        early_stopping=early_stopping,
        path=str(model_path),
        sample=query_x_batch,
        # The fitted Compose([ReplaceValue, Normalize]) pipeline -- both the
        # missing-value imputation and the feature standardization -- saved
        # as preprocessing= so a fresh process applies both, in the same
        # order, to any brand-new raw row automatically. This Compose shape
        # is the Milestone 92 fix: ReplaceValue (forge/data/transforms.py)
        # is what makes the imputation step persistable at all -- see
        # docs/development/m92-real-dataset-ingestion.md.
        preprocessing=stats["transform"],
        classes=CLASS_NAMES,
        task="tabular_classification",
    )
    duration = time.perf_counter() - start
    history = train_result.history

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {train_result.history.epochs_completed} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"train loss: {history[0].train_loss:.4f} -> {train_result.train_loss:.4f}")
    print(f"val loss:   {train_result.val_loss:.4f}")
    print(f"val accuracy: {train_result.val_metrics['accuracy']:.1%}")
    if args.early_stopping:
        print(f"\nEarly stopping: requested {args.epochs} epoch(s), completed {history.epochs_completed}, "
              f"stopped_early={train_result.stopped_early}")
        print(f"  best_epoch={train_result.best_epoch}, best_val_loss={train_result.best_monitored_value:.4f}, "
              f"final_val_loss={train_result.val_loss:.4f}")

    from forge.training import Trainer
    final_eval = Trainer(
        model, loss_fn, optimizer, device=args.device, metrics=[Accuracy()], verbose=False,
    ).evaluate(test_loader)
    print(f"\nFinal test evaluation: loss={final_eval.loss:.4f}, accuracy={final_eval.metrics['accuracy']:.1%}")
    print(f"Majority-class baseline accuracy was {stats['majority_baseline']:.1%}.")

    # train_result.artifact_path (Milestone 99) is the path save_and_verify()
    # actually wrote and verified -- the same string passed in as path=
    # above, echoed back on the result so this script does not need its own
    # model_path variable to remember where training put the artifact.
    print(f"\nSaved + verified model + preprocessing -> {train_result.artifact_path}")

    print("\nInspect the generated artifact with the M19 CLI:")
    print(f"  python -m forge model inspect {train_result.artifact_path}")

    # A raw new patient row, taken directly from the untouched CSV (via its
    # original row index, not test_ds[0]'s already-preprocessed Tensor) --
    # deliberately the literal source data, sentinel zeros and all, exactly
    # the case docs/development/m92-real-dataset-ingestion.md found silently
    # broken before ReplaceValue existed. No manual imputation needed here:
    # the saved Compose pipeline does it.
    X_raw, _ = load_raw()
    raw_query = X_raw[test_ds.indices[0]]
    print("\nPredicting a brand-new raw tabular row via forge.predict_model():")
    result = forge.predict_model(train_result.artifact_path, raw_query.reshape(1, N_FEATURES).astype(np.float32))[0]
    print(f"  Prediction: {result.label} (confidence {result.confidence:.1%}) "
          f"-- true label was '{CLASS_NAMES[int(query_y.numpy())]}'")
    print(f"\nPredict from the saved artifact alone, in a fresh process:")
    print(f"  python -m examples.tabular_diabetes.infer --model {train_result.artifact_path} "
          f"--input '{raw_query.tolist()}'")


if __name__ == "__main__":
    main()
