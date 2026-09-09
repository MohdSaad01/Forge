"""Forge Milestone 63: an end-to-end convolutional autoencoder example.

```text
AutoencoderDataset -> DataLoader -> Trainer -> ConvAutoencoder
    (Conv2d/MaxPool2d encoder -> Linear bottleneck -> Linear/UpsampleNearest2d/Conv2d decoder)
    -> MSELoss -> Adam
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`) -- this script adds
no framework logic of its own, only example wiring, exactly like
`examples/mnist/train.py`. See `examples/autoencoder/README.md` for
prerequisites, expected behavior, and the evaluation methodology.

## Determinism

`forge.random.seed(args.seed)` governs every draw from Forge's own default
generator: `Conv2d`/`Linear` parameter initialization at model construction.
`DataLoader` shuffling uses its own explicit `numpy.random.Generator`
(`--seed`-derived) -- three separate deterministic streams is unnecessary
here (unlike `regression`'s synthetic-data generator) since MNIST's images
are a fixed, already-downloaded corpus, not something this script draws
randomly.

## Usage

```bash
# First run: download once (if not already fetched by examples/mnist),
# train from scratch, save a checkpoint + model.
python -m examples.autoencoder.train --download --epochs 3 --device cpu

# Continue training from the saved checkpoint for 2 more epochs.
python -m examples.autoencoder.train --resume artifacts/autoencoder_checkpoint.forge --epochs 2

# CUDA (requires a working Forge CUDA backend).
python -m examples.autoencoder.train --epochs 3 --device cuda
```
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import forge
from forge import no_grad
from forge.data import Compose, DataLoader, Lambda
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Trainer, predict

try:
    from .dataset import DEFAULT_ROOT, AutoencoderDataset
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/autoencoder/train.py`)
    from dataset import DEFAULT_ROOT, AutoencoderDataset
    from model import build_model

_PIXEL_SCALE = 1.0 / 255.0


def build_transform():
    """`[0, 255]` uint8-valued pixels -> `[0, 1]` float32 -- no mean/std normalization.

    Reconstruction targets stay in their natural pixel scale (unlike
    `examples/mnist/train.py`'s classification preprocessing, which also
    zero-centers): `MSELoss` compares the decoder's output against this same
    scaled image, and there is no `Sigmoid` primitive to bound the decoder's
    output to `[0, 1]` (see `model.py`'s `decode()` docstring) -- a
    zero-centered, unbounded-range target would make the trivial "predict
    the mean image" baseline (see `main()`) harder to interpret for no
    benefit.
    """
    return Compose([Lambda(lambda x: x * _PIXEL_SCALE)])


def build_datasets(data_root: str, download: bool):
    transform = build_transform()
    train_ds = AutoencoderDataset(data_root, train=True, transform=transform, download=download)
    test_ds = AutoencoderDataset(data_root, train=False, transform=transform, download=download)
    return train_ds, test_ds


def trivial_baseline_mse(train_ds: AutoencoderDataset, test_ds: AutoencoderDataset) -> float:
    """MSE of predicting the training-set mean image for every test image.

    The regression-example (`examples/regression/train.py`) analog of
    "predict the mean" for an image-reconstruction target: computed directly
    with NumPy (evaluation/reporting only, not part of the trained model or
    the training loop), matching that script's own use of plain NumPy for
    baseline statistics.
    """
    train_images = np.stack([train_ds[i][0].numpy() for i in range(len(train_ds))])
    mean_image = train_images.mean(axis=0)
    test_images = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))])
    return float(((test_images - mean_image) ** 2).mean())


def latent_nearest_neighbor_label_agreement(model, dataset: AutoencoderDataset, device: str, n_samples: int, seed: int) -> float:
    """Fraction of `n_samples` test images whose nearest latent-space neighbor shares its digit label.

    A qualitative check that the learned bottleneck captures *semantic*
    structure, not just enough information to copy pixels back out: encode a
    random subset of test images (`model.encode()`, under `no_grad`, never
    touching the labels during encoding), find each one's nearest neighbor
    by Euclidean distance in latent space (excluding itself), and check
    whether that neighbor is the same digit. Random guessing over 10 classes
    would score ~10%; a representation that has learned nothing beyond
    per-pixel reconstruction would not necessarily beat that, so a
    meaningfully higher score is evidence the bottleneck is a genuinely
    useful, non-trivial representation, not just proof the model runs.
    Pure NumPy analysis after training -- no framework code, matching
    `trivial_baseline_mse`'s convention above.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    images = np.stack([dataset[int(i)][0].numpy() for i in indices])
    labels = np.array([dataset.label_at(int(i)) for i in indices])

    with no_grad():
        latents = model.encode(forge.Tensor(images, device=device)).to("cpu").numpy()

    # Pairwise squared Euclidean distances -- n_samples is small (<= a few
    # hundred), so a plain O(n^2) NumPy computation is appropriate; this is
    # evaluation code, not a Forge primitive.
    diffs = latents[:, None, :] - latents[None, :, :]
    dist_sq = (diffs ** 2).sum(axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    nearest = dist_sq.argmin(axis=1)
    return float((labels[nearest] == labels).mean())


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_ROOT, help="Directory holding the MNIST IDX files.")
    parser.add_argument("--download", action="store_true", help="Download MNIST into --data-root if not already present.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train this run.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Bottleneck dimensionality.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for forge.random and DataLoader shuffling.")
    parser.add_argument("--output-dir", default="examples/autoencoder/artifacts", help="Where to write model/checkpoint files.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint saved by a previous run to resume from.")
    parser.add_argument("--nn-eval-samples", type=int, default=200, help="Test samples used for the latent nearest-neighbor evaluation.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "autoencoder_model.forge"
    checkpoint_path = output_dir / "autoencoder_checkpoint.forge"

    forge.random.seed(args.seed)
    data_rng = np.random.default_rng(args.seed)

    print(f"Loading MNIST from '{args.data_root}' (download={args.download}) ...")
    train_ds, test_ds = build_datasets(args.data_root, args.download)
    print(f"train: {len(train_ds)} samples, test: {len(test_ds)} samples, shape (1, 28, 28)")

    print("Computing trivial baseline (predict the training-set mean image) ...")
    baseline_mse = trivial_baseline_mse(train_ds, test_ds)
    print(f"Trivial baseline: test MSE = {baseline_mse:.5f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_rng)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    loss_fn = MSELoss()

    if args.resume:
        print(f"Resuming from checkpoint '{args.resume}' ...")
        checkpoint = load_checkpoint(args.resume, device=args.device)
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device)
        trainer.resume(checkpoint)
        print(f"Resumed at epoch={trainer.epoch}, global_step={trainer.global_step}")
    else:
        model = build_model(latent_dim=args.latent_dim).to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device)

    start = time.perf_counter()
    history = trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
    duration = time.perf_counter() - start

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"train MSE: {history[0].train_loss:.5f} -> {history[-1].train_loss:.5f}")
    print(f"val MSE:   {history[-1].val_loss:.5f}")

    final_eval = trainer.evaluate(test_loader)
    test_mse = final_eval.loss
    print(f"\nFinal test evaluation: MSE={test_mse:.5f}")
    print(f"Trivial baseline MSE was {baseline_mse:.5f} -- "
          f"model achieves a {(1 - test_mse / baseline_mse):.1%} reduction over predicting the mean image.")

    print(f"\nEvaluating latent-space semantics on {args.nn_eval_samples} test samples ...")
    agreement = latent_nearest_neighbor_label_agreement(
        model, test_ds, args.device, args.nn_eval_samples, args.seed
    )
    print(f"Latent nearest-neighbor label agreement: {agreement:.1%} (random baseline: 10.0%)")

    trainer.save_checkpoint(str(checkpoint_path))
    print(f"\nSaved checkpoint -> {checkpoint_path}")
    save_model(model, str(model_path))
    print(f"Saved model -> {model_path}")

    # Model-persistence round trip: reload fresh and confirm reconstructions
    # match, the same property every other Forge example demonstrates.
    query_x, _ = test_ds[0]
    query_x = query_x.to(args.device).reshape(1, 1, 28, 28)
    pre_save_pred = predict(model, query_x).numpy()
    reloaded = load_model(str(model_path), device=args.device)
    post_load_pred = predict(reloaded, query_x).numpy()
    assert np.allclose(pre_save_pred, post_load_pred, atol=1e-5), "reloaded model prediction diverged"
    print("Verified: reloaded model reproduces the pre-save reconstruction.")

    print("\nInspect the generated artifacts with the Milestone 19 CLI:")
    print(f"  python -m forge model inspect {model_path}")
    print(f"  python -m forge checkpoint inspect {checkpoint_path}")


if __name__ == "__main__":
    main()
