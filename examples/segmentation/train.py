"""Forge Milestone 64: an end-to-end U-Net-style image segmentation example.

```text
SegmentationDataset -> DataLoader -> Trainer -> Sequential
    (Conv2d/ReLU/MaxPool2d encoder -> UpsampleNearest2d/Conv2d decoder)
    -> MSELoss -> Adam
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`) -- this script adds
no framework logic of its own, only example wiring, exactly like
`examples/regression/train.py` and `examples/autoencoder/train.py`. See
`examples/segmentation/README.md` for the task description and expected
results.

## Determinism

`forge.random.seed(args.seed)` governs `Conv2d` parameter initialization at
model construction. `SegmentationDataset` draws its own images/masks from an
independent `numpy.random.default_rng` stream seeded from `args.seed` (see
`dataset.py::make_datasets`), and `DataLoader` shuffling uses a third,
explicit `numpy.random.Generator` -- the same three-separate-streams
convention `examples/regression/train.py` documents.

## Usage

```bash
python -m examples.segmentation.train --epochs 15 --device cpu
python -m examples.segmentation.train --epochs 15 --device cuda
python -m examples.segmentation.train --resume examples/segmentation/artifacts/segmentation_checkpoint.forge --epochs 5
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
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Trainer

try:
    from .dataset import IMAGE_SIZE, make_datasets
    from .metrics import IoU, PixelAccuracy
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/segmentation/train.py`)
    from dataset import IMAGE_SIZE, make_datasets
    from metrics import IoU, PixelAccuracy
    from model import build_model

_THRESHOLD = 0.5


def majority_class_baseline(test_ds) -> "tuple[float, float]":
    """Pixel accuracy / IoU of always predicting "no foreground" (an all-zero mask).

    The segmentation-task analog of `examples/regression/train.py`'s
    predict-the-mean baseline and `examples/autoencoder/train.py`'s
    predict-the-mean-image baseline: the trivial constant prediction that
    needs no model at all. Background pixels far outnumber foreground pixels
    by construction (`dataset.py`'s shapes cover a small fraction of the
    image), so this baseline's pixel accuracy is already fairly high --
    which is exactly why pixel accuracy alone is not a sufficient metric,
    and IoU (which this baseline scores at `0.0` whenever any test image has
    a foreground pixel) is reported alongside it.
    """
    masks = np.stack([test_ds[i][1].numpy() for i in range(len(test_ds))])
    zero_pred = np.zeros_like(masks)
    targ = masks >= _THRESHOLD
    pred = zero_pred >= _THRESHOLD
    accuracy = float(np.mean(pred == targ))
    union = int(np.sum(pred | targ))
    iou = 1.0 if union == 0 else float(np.sum(pred & targ) / union)
    return accuracy, iou


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=3000, help="Number of synthetic training images to generate.")
    parser.add_argument("--n-test", type=int, default=500, help="Number of synthetic test images to generate.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train this run.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random, dataset generation, and DataLoader shuffling.")
    parser.add_argument("--output-dir", default="examples/segmentation/artifacts", help="Where to write model/checkpoint files.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint saved by a previous run to resume from.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "segmentation_model.forge"
    checkpoint_path = output_dir / "segmentation_checkpoint.forge"

    forge.random.seed(args.seed)
    data_rng = np.random.default_rng(args.seed)

    print(f"Generating synthetic segmentation dataset (train={args.n_train}, test={args.n_test}, "
          f"{IMAGE_SIZE}x{IMAGE_SIZE}) ...")
    train_ds, test_ds = make_datasets(args.n_train, args.n_test, seed=args.seed)

    baseline_accuracy, baseline_iou = majority_class_baseline(test_ds)
    print(f"Trivial baseline (predict all-background): pixel_accuracy={baseline_accuracy:.4f}, iou={baseline_iou:.4f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_rng)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    loss_fn = MSELoss()
    metrics = [PixelAccuracy(), IoU()]

    if args.resume:
        print(f"Resuming from checkpoint '{args.resume}' ...")
        checkpoint = load_checkpoint(args.resume, device=args.device)
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=metrics)
        trainer.resume(checkpoint)
        print(f"Resumed at epoch={trainer.epoch}, global_step={trainer.global_step}")
    else:
        model = build_model().to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=metrics)

    start = time.perf_counter()
    history = trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
    duration = time.perf_counter() - start

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"train loss (MSE): {history[0].train_loss:.5f} -> {history[-1].train_loss:.5f}")
    print(f"train pixel_accuracy: {history[0].train_metrics['pixel_accuracy']:.4f} -> "
          f"{history[-1].train_metrics['pixel_accuracy']:.4f}")
    print(f"train iou: {history[0].train_metrics['iou']:.4f} -> {history[-1].train_metrics['iou']:.4f}")

    final_eval = trainer.evaluate(test_loader)
    print(f"\nFinal test evaluation: MSE={final_eval.loss:.5f}, "
          f"pixel_accuracy={final_eval.metrics['pixel_accuracy']:.4f}, iou={final_eval.metrics['iou']:.4f}")
    print(f"Trivial baseline: pixel_accuracy={baseline_accuracy:.4f}, iou={baseline_iou:.4f}")
    print(f"Pixel accuracy improvement over baseline: "
          f"{(final_eval.metrics['pixel_accuracy'] - baseline_accuracy):+.4f}")
    print(f"IoU improvement over baseline: {(final_eval.metrics['iou'] - baseline_iou):+.4f}")

    trainer.save_checkpoint(str(checkpoint_path))
    print(f"\nSaved checkpoint -> {checkpoint_path}")
    save_model(model, str(model_path))
    print(f"Saved model -> {model_path}")

    # Model-persistence round trip: reload fresh and confirm predictions
    # match, the same property every other Forge example demonstrates.
    query_x, _ = test_ds[0]
    query_x = query_x.to(args.device).reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
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
