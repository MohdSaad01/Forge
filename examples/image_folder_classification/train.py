"""Forge Milestone 69: an end-to-end directory-based image classification example.

```text
generate_dataset() -> ImageFolder -> random_split -> DataLoader -> Trainer -> CNN
    -> CrossEntropyLoss -> Adam -> save/load -> forge.predict()
```

This is the first Forge example whose data source is **ordinary image files
on disk** discovered by `forge.data.ImageFolder`, rather than a bundled
binary format (MNIST's IDX files) or in-memory arrays. Every step below uses
only public Forge APIs (`forge`, `forge.data`, `forge.nn`, `forge.optim`,
`forge.training`, `forge.save_model`/`save_checkpoint`) -- this script adds
no framework logic of its own, only example wiring. See
`examples/image_folder_classification/README.md` for prerequisites and
expected output.

## Determinism

`forge.random.seed(args.seed)` governs `Conv2d`/`Linear` parameter
initialization; `--seed` also seeds the synthetic-dataset generator and the
`random_split`/`DataLoader` shuffling generators -- independent streams,
matching every other Forge example's documented RNG policy
(`docs/architecture/persistence.md`).

## Usage

```bash
# First run: generate the synthetic shape dataset, then train.
python -m examples.image_folder_classification.train --generate --epochs 8 --device cpu

# Continue training from the saved checkpoint for 4 more epochs.
python -m examples.image_folder_classification.train --resume examples/image_folder_classification/artifacts/image_folder_checkpoint.forge --epochs 4

# CUDA (requires a working Forge CUDA backend).
python -m examples.image_folder_classification.train --generate --epochs 8 --device cuda
```
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import forge
from forge.data import Compose, DataLoader, ImageFolder, Lambda, random_split
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Accuracy, Trainer, predict

try:
    from .generate_dataset import generate_dataset
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/image_folder_classification/train.py`)
    from generate_dataset import generate_dataset
    from model import build_model

# Raw pixels are [0, 255] (see `forge/data/image_folder.py`); scale to [0, 1]
# via the existing `forge.data.transforms` primitives, same pattern
# `examples/mnist/train.py` uses -- no new preprocessing API.
_PIXEL_SCALE = 1.0 / 255.0
_TRAIN_FRACTION = 0.8


def build_transform():
    return Compose([Lambda(lambda x: x * _PIXEL_SCALE)])


def build_datasets(data_root: str, split_seed: int):
    """Load the full `ImageFolder`, then split it 80/20 into train/test.

    A single directory of generated shape images has no separate train/test
    split on disk (unlike MNIST's four dedicated files), so this uses
    `forge.data.random_split` -- an existing primitive, not new example
    logic -- to carve one out deterministically.
    """
    transform = build_transform()
    full_dataset = ImageFolder(data_root, transform=transform)
    n_train = int(len(full_dataset) * _TRAIN_FRACTION)
    n_test = len(full_dataset) - n_train
    train_ds, test_ds = random_split(
        full_dataset, [n_train, n_test], generator=np.random.default_rng(split_seed)
    )
    return full_dataset, train_ds, test_ds


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="examples/image_folder_classification/data",
                         help="Directory holding the class subdirectories (ImageFolder root).")
    parser.add_argument("--generate", action="store_true",
                         help="(Re)generate the synthetic shape dataset into --data-root before training.")
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train this run.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random, dataset generation, and splitting/shuffling.")
    parser.add_argument("--output-dir", default="examples/image_folder_classification/artifacts",
                         help="Where to write model/checkpoint files.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint saved by a previous run to resume from.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "image_folder_model.forge"
    checkpoint_path = output_dir / "image_folder_checkpoint.forge"

    if args.generate:
        print(f"Generating synthetic shape dataset into '{args.data_root}' (seed={args.seed}) ...")
        generate_dataset(args.data_root, samples_per_class=args.samples_per_class,
                          image_size=args.image_size, seed=args.seed)

    forge.random.seed(args.seed)

    print(f"Loading images from '{args.data_root}' ...")
    full_dataset, train_ds, test_ds = build_datasets(args.data_root, split_seed=args.seed)
    print(f"classes: {full_dataset.classes}  class_to_idx: {full_dataset.class_to_idx}")
    print(f"train: {len(train_ds)} samples, test: {len(test_ds)} samples, "
          f"shape (3, {args.image_size}, {args.image_size})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=np.random.default_rng(args.seed + 1))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    loss_fn = CrossEntropyLoss()

    if args.resume:
        print(f"Resuming from checkpoint '{args.resume}' ...")
        checkpoint = load_checkpoint(args.resume, device=args.device)
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[Accuracy()])
        trainer.resume(checkpoint)
        print(f"Resumed at epoch={trainer.epoch}, global_step={trainer.global_step}")
    else:
        model = build_model(num_classes=len(full_dataset.classes)).to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[Accuracy()])

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

    trainer.save_checkpoint(str(checkpoint_path))
    print(f"\nSaved checkpoint -> {checkpoint_path}")
    save_model(model, str(model_path))
    print(f"Saved model -> {model_path}")

    # Model-persistence round trip: save -> reload -> predict() must agree.
    query_x, query_y = test_ds[0]
    query_x_batch = query_x.to(args.device).reshape(1, *query_x.shape)
    pre_save_pred = predict(model, query_x_batch).numpy()
    reloaded = load_model(str(model_path), device=args.device)
    post_load_pred = predict(reloaded, query_x_batch).numpy()
    assert np.allclose(pre_save_pred, post_load_pred, atol=1e-5), "reloaded model prediction diverged"
    print("Verified: reloaded model reproduces the pre-save prediction.")

    # Section 11: standalone single-image inference through forge.predict(),
    # mapping the predicted class index back to a human-readable class name.
    predicted_idx = int(np.argmax(post_load_pred, axis=1)[0])
    true_idx = int(query_y.numpy())
    predicted_name = full_dataset.classes[predicted_idx]
    true_name = full_dataset.classes[true_idx]
    print(f"\nInference demo (test sample 0, true class: {true_name}):")
    print(f"Prediction: {predicted_name}")

    print("\nInspect the generated artifacts with the M19 CLI:")
    print(f"  python -m forge model inspect {model_path}")
    print(f"  python -m forge checkpoint inspect {checkpoint_path}")


if __name__ == "__main__":
    main()
