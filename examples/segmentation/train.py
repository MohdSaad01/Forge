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

## Portable artifact inference (Milestone 84)

The saved model now also carries a `Normalize`-based preprocessing pipeline
(`build_transform()` below), so `forge.predict_image_artifact()` can turn a
brand-new image *file* -- not an in-memory `Tensor` from this run's own
dataset -- directly into a predicted mask, in a completely separate process:
see `examples/segmentation/infer.py`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import forge
from forge.data import Compose, DataLoader, Normalize, save_image
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint
from forge.training import Trainer, predict, predict_image_artifact, save_and_verify

try:
    from .dataset import IMAGE_SIZE, generate_raw, make_datasets
    from .metrics import IoU, PixelAccuracy
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/segmentation/train.py`)
    from dataset import IMAGE_SIZE, generate_raw, make_datasets
    from metrics import IoU, PixelAccuracy
    from model import build_model

_THRESHOLD = 0.5


def build_transform():
    """`[0, 255]` uint8-valued decoded-image pixels -> `[0, 1]` float32.

    `SegmentationDataset` already generates its images directly in `[0, 1]`
    (Milestone 64), so training itself needs no transform. This is needed
    only for Milestone 84's portable-artifact workflow: a real image *file*
    on disk (written by `forge.data.save_image()`, or any other `[0, 255]`
    -encoded PNG/JPEG) decodes via `ImageFolder._load_image()` into the same
    raw `[0, 255]` range `examples/autoencoder/train.py`'s MNIST files and
    `examples/image_folder_classification`'s photos do -- `Normalize(mean=0.0,
    std=255.0)` rescales that back down to the `[0, 1]` range the model was
    actually trained on, the same substitution those two examples already
    established for exactly this reason (`Lambda` cannot be saved via
    `save_model(..., preprocessing=...)`).
    """
    return Compose([Normalize(mean=0.0, std=255.0)])


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

    # Milestone 78: save_and_verify() saves the model, then reloads it fresh
    # and confirms the reload's prediction on query_x matches the model that
    # was just saved -- the same "save -> reload -> predict() must agree"
    # round trip every other Forge example demonstrates, now the shared
    # abstraction.
    query_x, query_mask = test_ds[0]
    query_x = query_x.to(args.device).reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    reloaded = save_and_verify(model, str(model_path), query_x, preprocessing=build_transform())
    print(f"Saved + verified model + preprocessing -> {model_path}")

    # Milestone 76: a segmentation model's real output is the predicted
    # mask, but until now nothing ever rendered it -- every prior run only
    # printed scalar pixel_accuracy/iou numbers. Write the input image, the
    # thresholded predicted mask, and the ground-truth mask out as real
    # PNGs so a person can actually see what the model segmented -- using
    # the freshly reloaded model (already proven to match the pre-save model
    # to within `atol` by `save_and_verify()` above).
    input_image_path = output_dir / "segmentation_input.png"
    predicted_mask_path = output_dir / "segmentation_predicted_mask.png"
    ground_truth_mask_path = output_dir / "segmentation_ground_truth_mask.png"
    reloaded_pred = predict(reloaded, query_x).numpy()
    predicted_mask = (reloaded_pred.reshape(1, IMAGE_SIZE, IMAGE_SIZE) >= _THRESHOLD).astype(np.float32)
    save_image(query_x.reshape(3, IMAGE_SIZE, IMAGE_SIZE), str(input_image_path))
    save_image(forge.Tensor(predicted_mask), str(predicted_mask_path))
    save_image(query_mask, str(ground_truth_mask_path))
    print(f"Saved segmentation input/predicted-mask/ground-truth-mask -> "
          f"{input_image_path}, {predicted_mask_path}, {ground_truth_mask_path}")

    # Milestone 84: the complete portable, file-based artifact workflow --
    # a brand-new synthetic image (independent seed, never part of train or
    # test), written to disk as a real PNG, then read back through nothing
    # but the '.forge' file at model_path via forge.predict_image_artifact().
    # Unlike the in-memory `query_x` demo above, this never touches the
    # Trainer/dataset objects still alive in this process -- it exercises
    # exactly what a developer holding only model_path and a new image file
    # can do.
    new_image, new_mask = generate_raw(1, seed=args.seed + 2, size=IMAGE_SIZE)
    new_image_path = output_dir / "segmentation_new_image.png"
    new_ground_truth_path = output_dir / "segmentation_new_ground_truth_mask.png"
    new_predicted_mask_path = output_dir / "segmentation_new_predicted_mask.png"
    save_image(forge.Tensor(new_image[0]), str(new_image_path))
    save_image(forge.Tensor(new_mask[0]), str(new_ground_truth_path))

    artifact_prediction = predict_image_artifact(str(model_path), str(new_image_path))
    save_image(artifact_prediction, str(new_predicted_mask_path))
    print(f"\nPortable artifact inference (forge.predict_image_artifact()): "
          f"{new_image_path} -> {new_predicted_mask_path}")

    print("\nInspect the generated artifacts with the Milestone 19 CLI:")
    print(f"  python -m forge model inspect {model_path}")
    print(f"  python -m forge checkpoint inspect {checkpoint_path}")


if __name__ == "__main__":
    main()
