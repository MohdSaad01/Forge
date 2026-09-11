"""Forge Milestone 20: an end-to-end MNIST training example.

```text
MNISTDataset -> DataLoader -> Trainer -> CNN (Conv2d/ReLU/MaxPool2d/Flatten/Linear)
    -> CrossEntropyLoss -> Adam
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.save_model`/`save_checkpoint`) --
this script adds no framework logic of its own, only example wiring. See
`examples/mnist/README.md` for prerequisites, expected behavior, and how to
reproduce every part of Milestone 20 (checkpoint/resume, model persistence,
CLI inspection) from this one entry point.

## Determinism

`forge.random.seed(args.seed)` governs every draw from Forge's own default
generator: `Conv2d`/`Linear` parameter initialization at model construction,
and `Dropout` if any were used (this architecture has none). `DataLoader`
shuffling uses its own explicit `numpy.random.Generator` (`--seed` derived,
but independent of Forge's default generator) -- see `forge/data/dataloader.py`
and `docs/architecture/persistence.md`'s checkpoint RNG policy. This makes a
given `--seed` run's model initialization and batch order both reproducible,
but the two are deliberately separate generators, not one shared stream.

## Usage

```bash
# First run: download once, train from scratch, save a checkpoint + model.
python -m examples.mnist.train --download --epochs 3 --device cpu

# Continue training from the saved checkpoint for 2 more epochs.
python -m examples.mnist.train --resume artifacts/mnist_checkpoint.forge --epochs 2

# CUDA (requires a working Forge CUDA backend).
python -m examples.mnist.train --epochs 3 --device cuda
```
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import forge
from forge import Tensor
from forge.data import Compose, DataLoader, Normalize, save_image
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_classes, load_preprocessing
from forge.training import Accuracy, Trainer, interpret_classification, predict, save_and_verify

try:
    from .dataset import MNISTDataset
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/mnist/train.py`)
    from dataset import MNISTDataset
    from model import build_model

# Conventional MNIST normalization constants (mean/std of the raw [0, 1]
# pixel distribution over the training set) -- applied via the existing
# `forge.data.transforms` primitives, not a bespoke MNIST preprocessing API.
_MEAN = 0.1307
_STD = 0.3081


def build_transform():
    """`[0, 255]` uint8-valued pixels -> scaled, mean/std-normalized float32.

    Milestone 77: the `/255` scale is expressed as `Normalize(mean=0.0,
    std=255.0)` -- `(x - 0) / 255 == x / 255` -- composed with the existing
    mean/std `Normalize` step, not `Lambda(lambda x: x * _PIXEL_SCALE)` as
    earlier milestones had it. `Lambda` wraps an arbitrary Python callable
    with no safe serialized representation (see
    `forge/serialization/transforms.py`), so a `Lambda`-based pipeline can
    never be passed to `save_model(..., preprocessing=...)`. This is the
    exact substitution `examples/image_folder_classification/train.py`
    already established in Milestone 71, applied here for the first time to
    Forge's flagship example -- no new Forge primitive, no MNIST-specific
    preprocessing API added.
    """
    return Compose([Normalize(mean=0.0, std=255.0), Normalize(mean=_MEAN, std=_STD)])


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
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train this run.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random and DataLoader shuffling.")
    parser.add_argument("--output-dir", default="examples/mnist/artifacts", help="Where to write model/checkpoint files.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint saved by a previous run to resume from.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mnist_model.forge"
    checkpoint_path = output_dir / "mnist_checkpoint.forge"

    forge.random.seed(args.seed)
    data_rng = np.random.default_rng(args.seed)

    print(f"Loading MNIST from '{args.data_root}' (download={args.download}) ...")
    train_ds, test_ds = build_datasets(args.data_root, args.download)
    print(f"train: {len(train_ds)} samples, test: {len(test_ds)} samples, shape (1, 28, 28)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_rng)
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
        model = build_model().to(args.device)
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
    # Milestone 77: `preprocessing=`/`classes=` (Milestones 71/72) now save
    # alongside the model for the first time in this example -- previously
    # blocked by `build_transform()`'s `Lambda` step (see that function's
    # docstring). `classes=[str(d) for d in range(10)]` is MNIST's own
    # digit-index-to-label vocabulary (`output[..., i]` means the digit `i`),
    # the same convention `ImageFolder.classes` already uses for images.
    # Milestone 78: `save_and_verify()` (Section 12's old hand-written
    # "save -> reload -> predict() must agree" round trip, now the shared
    # abstraction every retrofitted Forge example calls the same way) saves
    # the model and immediately proves it by reloading it fresh.
    query_x, query_y = test_ds[0]
    query_x = query_x.to(args.device).reshape(1, 1, 28, 28)
    reloaded = save_and_verify(
        model, str(model_path), query_x,
        preprocessing=build_transform(), classes=[str(d) for d in range(10)],
    )
    print(f"Saved + verified model + preprocessing + classes -> {model_path}")

    # Milestone 72's interpretation step, using load_classes()'s own
    # reconstructed vocabulary rather than this process's in-memory literal,
    # to prove the file alone is enough (matches
    # examples/image_folder_classification/train.py's own precedent).
    reloaded_classes = load_classes(str(model_path))
    result = interpret_classification(predict(reloaded, query_x), reloaded_classes)[0]
    print(f"\nInference demo (test sample 0, true digit {int(query_y.numpy())}):")
    print(f"Prediction: digit {result.label}")
    print(f"Confidence: {result.confidence:.1%}")

    # Milestone 77: a brand-new image file -- decoded from raw, un-normalized
    # pixels and prepared via the *file's own* reconstructed preprocessing
    # pipeline (`load_preprocessing()`, not this process's own
    # `build_transform()` call) -- proving the saved artifact travels to a
    # fresh process the same way Milestone 71/72 already proved for
    # `examples/image_folder_classification`. `examples/mnist/infer.py`
    # (run here in-process for the demo, and independently as a genuinely
    # separate process below) is the standalone counterpart to
    # `image_folder_classification/infer.py`.
    raw_test_ds = MNISTDataset(args.data_root, train=False, transform=None, download=False)
    raw_image, raw_label = raw_test_ds[1]
    new_image_path = output_dir / "new_digit_query.png"
    save_image(Tensor(raw_image.numpy() / 255.0), new_image_path)
    reloaded_preprocessing = load_preprocessing(str(model_path))
    new_batch = reloaded_preprocessing(raw_image).to(args.device).reshape(1, 1, 28, 28)
    new_pred = predict(reloaded, new_batch)
    new_result = interpret_classification(new_pred, reloaded_classes)[0]
    print(f"\nNew image '{new_image_path.name}' (never passed through this process's own "
          f"build_transform() call, true digit {int(raw_label.numpy())}):")
    print(f"Prediction: digit {new_result.label}")
    print(f"Confidence: {new_result.confidence:.1%}")

    print("\nInspect the generated artifacts with the CLI:")
    print(f"  python -m forge model inspect {model_path}")
    print(f"  python -m forge checkpoint inspect {checkpoint_path}")
    print("\nRun standalone inference in a fresh process (Milestone 77):")
    print(f"  python -m examples.mnist.infer --model {model_path} --image {new_image_path}")


if __name__ == "__main__":
    main()
