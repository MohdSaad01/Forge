"""Forge Milestone 66: an end-to-end residual (ResNet-style) CNN training example.

```text
MNISTDataset (examples.mnist.dataset) -> DataLoader -> Trainer
    -> ResNetMNIST (Conv2d/BatchNorm2d/ReLU residual blocks) -> CrossEntropyLoss -> Adam
```

Reuses the existing MNIST dataset/download infrastructure
(`examples/mnist/dataset.py`, `examples/mnist/train.py::build_transform`)
rather than introducing a second dataset -- this milestone's architecture is
what's new, not the data. Every step otherwise uses only public Forge APIs;
no framework logic lives in this file.

## Determinism

Identical policy to `examples/mnist/train.py`/`examples/regression/train.py`:
`forge.random.seed(args.seed)` governs Forge's default generator (every
`Conv2d`/`Linear` parameter draw at construction), and `DataLoader` shuffling
uses its own independent `numpy.random.Generator` derived from `--seed`.

**Reproducibility workflow (Milestone 65, reused here rather than
reimplemented):** the `DataLoader` shuffle generator's exact stream position
is saved into `save_checkpoint(..., extra=...)` and restored on `--resume`
(`examples.regression.experiment`'s established pattern), and every run
writes a JSON run record (config + per-epoch history + final evaluation +
runtime) via the same `examples.regression.experiment`/`compare` module --
no competing experiment-tracking mechanism is introduced for this example.

## Usage

```bash
# First run: download once, train from scratch, save a checkpoint + model.
python -m examples.resnet.train --download --epochs 5 --device cpu

# Continue training from the saved checkpoint for 2 more epochs.
python -m examples.resnet.train --resume examples/resnet/artifacts/resnet_checkpoint.forge --epochs 2

# CUDA (requires a working Forge CUDA backend).
python -m examples.resnet.train --epochs 5 --device cuda

# Compare this run against another (Milestone 65's tool, reused directly).
python -m examples.regression.compare examples/resnet/artifacts/resnet_history.json <other_run>/resnet_history.json
```
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import forge
from forge import no_grad
from forge.data import DataLoader
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Accuracy, Trainer

try:
    from examples.mnist.dataset import MNISTDataset
    from examples.mnist.train import build_transform
    from examples.regression.experiment import (
        extend_run_record,
        load_run_record,
        new_run_record,
        save_run_record,
    )

    from .model import build_model
except ImportError:  # running as a plain script (`python examples/resnet/train.py`)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.mnist.dataset import MNISTDataset
    from examples.mnist.train import build_transform
    from examples.regression.experiment import (
        extend_run_record,
        load_run_record,
        new_run_record,
        save_run_record,
    )
    from model import build_model

_NUM_CLASSES = 10


def build_datasets(data_root: str, download: bool):
    transform = build_transform()
    train_ds = MNISTDataset(data_root, train=True, transform=transform, download=download)
    test_ds = MNISTDataset(data_root, train=False, transform=transform, download=download)
    return train_ds, test_ds


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="examples/mnist/data", help="Directory holding the MNIST IDX files.")
    parser.add_argument("--download", action="store_true", help="Download MNIST into --data-root if not already present.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train this run.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random and DataLoader shuffling.")
    parser.add_argument("--output-dir", default="examples/resnet/artifacts", help="Where to write model/checkpoint/run-record files.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint saved by a previous run to resume from.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "resnet_model.forge"
    checkpoint_path = output_dir / "resnet_checkpoint.forge"
    history_path = output_dir / "resnet_history.json"

    forge.random.seed(args.seed)
    data_rng = np.random.default_rng(args.seed)

    print(f"Loading MNIST from '{args.data_root}' (download={args.download}) ...")
    train_ds, test_ds = build_datasets(args.data_root, args.download)
    print(f"train: {len(train_ds)} samples, test: {len(test_ds)} samples, shape (1, 28, 28)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_rng)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    loss_fn = CrossEntropyLoss()
    chance_accuracy = 1.0 / _NUM_CLASSES
    print(f"Baseline (10-class chance accuracy): {chance_accuracy:.2%}")

    if args.resume:
        print(f"Resuming from checkpoint '{args.resume}' ...")
        checkpoint = load_checkpoint(args.resume, device=args.device)
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[Accuracy()])
        trainer.resume(checkpoint)
        print(f"Resumed at epoch={trainer.epoch}, global_step={trainer.global_step}")
        saved_rng_state = checkpoint.extra.get("data_loader_rng_state")
        if saved_rng_state is not None:
            data_rng.bit_generator.state = saved_rng_state
        run_record = load_run_record(history_path)
        if run_record is None:
            print(f"(no existing run record at '{history_path}' -- starting a new one)")
            run_record = new_run_record(vars(args))
    else:
        model = build_model().to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[Accuracy()])
        run_record = new_run_record(vars(args))

    start = time.perf_counter()
    history = trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
    duration = time.perf_counter() - start

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"loss: {history[0].train_loss:.4f} -> {history[-1].train_loss:.4f}")
    print(f"val accuracy: {history[-1].val_metrics['accuracy']:.2%}")

    final_eval = trainer.evaluate(test_loader)
    print(f"\nFinal test evaluation: loss={final_eval.loss:.4f}, accuracy={final_eval.metrics['accuracy']:.2%}")
    print(f"Baseline (chance) accuracy was {chance_accuracy:.2%} -- "
          f"model improves by {final_eval.metrics['accuracy'] - chance_accuracy:.2%} absolute.")

    trainer.save_checkpoint(str(checkpoint_path), extra={"data_loader_rng_state": data_rng.bit_generator.state})
    print(f"\nSaved checkpoint -> {checkpoint_path}")
    save_model(model, str(model_path))
    print(f"Saved model -> {model_path}")

    extend_run_record(run_record, history=history, final_eval=final_eval, duration_seconds=duration)
    save_run_record(history_path, run_record)
    print(f"Saved run record -> {history_path}")

    # Model-persistence round trip (Section 10): load fresh and confirm
    # predictions match, the same property every other Forge example
    # verifies.
    query_x, _ = test_ds[0]
    query_x = query_x.to(args.device).reshape(1, 1, 28, 28)
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
    print("\nCompare this run against another with the Milestone 65 tool:")
    print(f"  python -m examples.regression.compare {history_path} <other_run>/resnet_history.json")


if __name__ == "__main__":
    main()
