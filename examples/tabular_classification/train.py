"""Forge Milestone 91: an end-to-end tabular *classification* example.

```text
make_datasets() -> DataLoader -> MLP (Linear/ReLU/Linear/ReLU/Linear)
    -> CrossEntropyLoss -> Adam -> forge.train_and_save() -> forge.predict_model()
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`) -- this script adds
no framework logic of its own, only example wiring, exactly like
`examples/regression/train.py`. See `examples/tabular_classification/README.md`
for prerequisites and expected behavior.

No `--resume`/checkpoint path -- this workload's dataset generation and
training are both fast enough that interrupted-training resume is not a
realistic concern (Milestone 91 brief, Section 15); unlike
`examples/regression`, this script is a single fresh-training path only.

## Determinism

`forge.random.seed(args.seed)` governs `Linear` parameter initialization;
`examples.tabular_classification.dataset.generate_raw()`'s own
`numpy.random.default_rng(args.seed)` governs the synthetic features/noise;
`make_datasets()`'s own `numpy.random.default_rng(args.seed + 1)` governs the
train/val/test `random_split` permutation; `DataLoader`'s explicit
`numpy.random.Generator` governs batch shuffling -- four independent
deterministic streams, matching `examples/regression/train.py`'s documented
policy.

## Usage

```bash
python -m examples.tabular_classification.train --epochs 30 --device cpu
python -m examples.tabular_classification.train --epochs 30 --device cuda
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
from forge.training import Accuracy

try:
    from .dataset import CLASS_NAMES, N_FEATURES, make_datasets
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/tabular_classification/train.py`)
    from dataset import CLASS_NAMES, N_FEATURES, make_datasets
    from model import build_model


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-per-class-train", type=int, default=300)
    parser.add_argument("--samples-per-class-val", type=int, default=60)
    parser.add_argument("--samples-per-class-test", type=int, default=60)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random, the synthetic dataset, the split, and DataLoader shuffling.")
    parser.add_argument("--output-dir", default="examples/tabular_classification/artifacts", help="Where to write the model file.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tabular_classification_model.forge"

    print(f"Generating synthetic tabular classification data (seed={args.seed}) ...")
    train_ds, val_ds, test_ds, stats = make_datasets(
        args.samples_per_class_train, args.samples_per_class_val, args.samples_per_class_test, seed=args.seed
    )
    print(f"train: {len(train_ds)} samples {stats['train_class_counts']}, "
          f"val: {len(val_ds)} samples, test: {len(test_ds)} samples, features: {N_FEATURES}, classes: {CLASS_NAMES}")
    trivial_baseline_acc = 1.0 / len(CLASS_NAMES)
    print(f"Trivial baseline (uniform random guess): accuracy = {trivial_baseline_acc:.1%}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    loss_fn = CrossEntropyLoss()

    # Milestone 81/83's save_and_verify()/train_and_save() need a
    # sample to prove the eventual artifact is portable -- already
    # normalized (test_ds applies the fitted Normalize transform on
    # __getitem__), matching predict()'s "already-prepared input" calling
    # convention.
    query_x, query_y = test_ds[0]
    query_x_batch = query_x.to(args.device).reshape(1, N_FEATURES)

    forge.random.seed(args.seed)
    data_loader_rng = np.random.default_rng(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_loader_rng)
    model = build_model().to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

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
        # The fitted train-split Normalize transform, saved as preprocessing=
        # so a fresh process can standardize a brand-new raw feature vector
        # the same way -- mirrors examples/regression/train.py exactly.
        # classes= is this workload's class vocabulary (the one addition
        # over examples/regression, which has no classes= at all).
        preprocessing=stats["transform"],
        classes=CLASS_NAMES,
        # Milestone 91: task="tabular_classification", not "classification" --
        # this artifact's input is an already-batched numeric feature vector,
        # not an image file path. Saving it as task="classification" is
        # exactly the bug this milestone found and fixed: forge.predict_model()
        # would route it straight to predict_artifact(), which requires an
        # image file path and raises DataError for anything else. See
        # forge.predict_tabular_classification_artifact()'s own docstring and
        # docs/development/m91-tabular-classification-artifact-inference.md.
        task="tabular_classification",
    )
    duration = time.perf_counter() - start
    history = train_result.history

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"train loss: {history[0].train_loss:.4f} -> {history[-1].train_loss:.4f}")
    print(f"val loss:   {history[-1].val_loss:.4f}")
    print(f"val accuracy: {history[-1].val_metrics['accuracy']:.1%}")

    from forge.training import Trainer
    final_eval = Trainer(
        model, loss_fn, optimizer, device=args.device, metrics=[Accuracy()], verbose=False,
    ).evaluate(test_loader)
    print(f"\nFinal test evaluation: loss={final_eval.loss:.4f}, accuracy={final_eval.metrics['accuracy']:.1%}")
    print(f"Trivial baseline accuracy was {trivial_baseline_acc:.1%}.")

    print(f"\nSaved + verified model + preprocessing -> {model_path}")

    print("\nInspect the generated artifact with the M19 CLI:")
    print(f"  python -m forge model inspect {model_path}")

    # A developer holding only the saved artifact should be able to predict
    # on a brand-new *raw* (un-normalized) tabular sample with one call --
    # exactly the workflow examples/regression/train.py's own end-of-run demo
    # exercises via forge.predict_tensor_artifact(). No such raw-tabular
    # input path exists for a *classification* artifact before Milestone 91
    # -- see the README's "Portable-artifact inference" section for what
    # this call does and why it needed a Milestone-91 fix.
    raw_query = query_x.numpy().reshape(1, N_FEATURES) * stats["std"] + stats["mean"]
    print("\nPredicting a brand-new raw (un-normalized) sample via forge.predict_model():")
    # predict_tabular_classification_artifact() (Milestone 91) returns one
    # result per input row (input_data is already-batched numeric data, not
    # a single image file) -- index [0] for this one-row query.
    result = forge.predict_model(str(model_path), raw_query.astype(np.float32))[0]
    print(f"  Prediction: {result.label} (confidence {result.confidence:.1%}) "
          f"-- true label was '{CLASS_NAMES[int(query_y.numpy())]}'")
    print(f"\nPredict from the saved artifact alone, in a fresh process:")
    print(f"  python -m examples.tabular_classification.infer --model {model_path} "
          f"--input '{raw_query.tolist()[0]}'")


if __name__ == "__main__":
    main()
