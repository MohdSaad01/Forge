"""Forge Milestone 60: an end-to-end tabular regression example.

```text
make_datasets() -> DataLoader -> Trainer -> MLP (Linear/ReLU/Linear/ReLU/Linear)
    -> MSELoss -> Adam
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`) -- this script adds
no framework logic of its own, only example wiring, exactly like
`examples/mnist/train.py`. See `examples/regression/README.md` for
prerequisites, expected behavior, and how to reproduce checkpoint/resume and
model persistence from this one entry point.

## Determinism

`forge.random.seed(args.seed)` governs every draw from Forge's own default
generator: `Linear` parameter initialization at model construction.
`examples.regression.dataset.generate_raw()` uses its own explicit
`numpy.random.default_rng(args.seed)` for the synthetic data, and
`DataLoader` shuffling uses a third, independent `numpy.random.Generator`
(`--seed`-derived) -- three separate deterministic streams, matching
`examples/mnist/train.py`'s documented policy (see
`docs/architecture/persistence.md`'s checkpoint RNG policy).

## Usage

```bash
# First run: train from scratch, save a checkpoint + model.
python -m examples.regression.train --epochs 40 --device cpu

# Continue training from the saved checkpoint for 10 more epochs.
python -m examples.regression.train --resume artifacts/regression_checkpoint.forge --epochs 10

# CUDA (requires a working Forge CUDA backend).
python -m examples.regression.train --epochs 40 --device cuda
```
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import forge
from forge import no_grad
from forge.data import DataLoader
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import MeanAbsoluteError, Trainer

try:
    from .dataset import make_datasets
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/regression/train.py`)
    from dataset import make_datasets
    from model import build_model


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=4000, help="Number of synthetic training samples.")
    parser.add_argument("--n-val", type=int, default=500, help="Number of synthetic validation samples.")
    parser.add_argument("--n-test", type=int, default=500, help="Number of synthetic test samples.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train this run.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random, the synthetic dataset, and DataLoader shuffling.")
    parser.add_argument("--output-dir", default="examples/regression/artifacts", help="Where to write model/checkpoint files.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint saved by a previous run to resume from.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "regression_model.forge"
    checkpoint_path = output_dir / "regression_checkpoint.forge"

    forge.random.seed(args.seed)
    data_rng = np.random.default_rng(args.seed)

    print(f"Generating synthetic tabular regression data (seed={args.seed}) ...")
    train_ds, val_ds, test_ds, stats = make_datasets(args.n_train, args.n_val, args.n_test, seed=args.seed)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples, test: {len(test_ds)} samples, features: 8")
    baseline_mse = stats["y_train_var"]
    print(f"Trivial baseline (predict train mean): MSE = {baseline_mse:.4f}, RMSE = {math.sqrt(baseline_mse):.4f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_rng)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    loss_fn = MSELoss()

    if args.resume:
        print(f"Resuming from checkpoint '{args.resume}' ...")
        checkpoint = load_checkpoint(args.resume, device=args.device)
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[MeanAbsoluteError()])
        trainer.resume(checkpoint)
        print(f"Resumed at epoch={trainer.epoch}, global_step={trainer.global_step}")
    else:
        model = build_model().to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[MeanAbsoluteError()])

    start = time.perf_counter()
    history = trainer.fit(train_loader, epochs=args.epochs, validation_loader=val_loader)
    duration = time.perf_counter() - start

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"train MSE: {history[0].train_loss:.4f} -> {history[-1].train_loss:.4f}")
    print(f"val MSE:   {history[-1].val_loss:.4f}  (RMSE {math.sqrt(history[-1].val_loss):.4f})")
    print(f"val MAE:   {history[-1].val_metrics['mae']:.4f}")

    final_eval = trainer.evaluate(test_loader)
    test_mse = final_eval.loss
    print(f"\nFinal test evaluation: MSE={test_mse:.4f}, RMSE={math.sqrt(test_mse):.4f}, "
          f"MAE={final_eval.metrics['mae']:.4f}")
    print(f"Trivial baseline MSE was {baseline_mse:.4f} -- "
          f"model achieves a {(1 - test_mse / baseline_mse):.1%} reduction over predicting the mean.")

    trainer.save_checkpoint(str(checkpoint_path))
    print(f"\nSaved checkpoint -> {checkpoint_path}")
    save_model(model, str(model_path))
    print(f"Saved model -> {model_path}")

    # Model-persistence round trip: reload fresh and confirm predictions
    # match, the same property `examples/mnist/train.py` demonstrates.
    query_x, _ = test_ds[0]
    query_x = query_x.to(args.device).reshape(1, -1)
    with no_grad():
        pre_save_pred = model(query_x).to("cpu").numpy()
    reloaded = load_model(str(model_path), device=args.device)
    with no_grad():
        post_load_pred = reloaded(query_x).to("cpu").numpy()
    assert np.allclose(pre_save_pred, post_load_pred, atol=1e-5), "reloaded model prediction diverged"
    print("Verified: reloaded model reproduces the pre-save prediction.")

    print("\nInspect the generated artifacts with the Milestone 19 CLI:")
    print(f"  python -m forge model inspect {model_path}")
    print(f"  python -m forge checkpoint inspect {checkpoint_path}")


if __name__ == "__main__":
    main()
