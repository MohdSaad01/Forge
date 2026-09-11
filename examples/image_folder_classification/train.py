"""Forge Milestone 69/70/71/72/73: an end-to-end directory-based image
classification example, now with mixed-resolution source images, *persisted*
preprocessing, a *self-describing* class vocabulary, and a reusable
fresh-or-resumed training session.

```text
generate_dataset() [mixed H, W] -> ImageFolder -> Resize+Normalize -> random_split -> DataLoader
    -> Trainer -> CNN -> CrossEntropyLoss -> Adam
    -> save (model + preprocessing + classes) -> forge.predict() -> interpret_classification()
```

This is the first Forge example whose data source is **ordinary image files
on disk** discovered by `forge.data.ImageFolder`, rather than a bundled
binary format (MNIST's IDX files) or in-memory arrays. Milestone 70 made the
generated source images genuinely mixed-resolution (each image's height and
width drawn independently) and added `forge.data.transforms.Resize` so the
same preprocessing step normalizes every training sample *and* every new
inference image to one fixed shape before it reaches the model or
`DataLoader`'s batching. Milestone 71 closed M70's own documented gap (see
`docs/development/m70-image-preprocessing.md` Section 16/17): this script's
preprocessing pipeline is now `Compose([Resize(...), Normalize(...)])`
-- both serializable -- and is saved *alongside* the model via
`save_model(model, path, preprocessing=...)`, so a separate, later process
(`infer.py`) can reconstruct it without re-running or re-importing any of
this script's own code. See `docs/development/m71-preprocessing-
persistence.md`. Milestone 72 closed the *next* gap M71 explicitly left open
(`docs/architecture/persistence.md`'s old "Class-index-to-name vocabularies"
note): `full_dataset.classes` is now saved via `save_model(..., classes=...)`
in the same file, and `forge.training.interpret_classification()` turns a
raw prediction `Tensor` plus that vocabulary into a human-readable
`"dog"`/confidence result -- no more hand-written `classes.json` sidecar a
caller had to remember to keep next to the model file. See
`docs/development/m72-classification-metadata.md`. Milestone 73 replaced
this script's own hand-rolled `--resume`-or-fresh-start `Trainer`
construction with `forge.training.start_training_session()` -- shared with
`examples/regression/train.py` and five other checkpoint-capable examples
that independently duplicated the identical branch -- which also gave this
script `DataLoader`-shuffle resume-equivalence for the first time (see
`docs/development/m73-reusable-training-workflow.md`).

Every step below uses only public Forge APIs (`forge`, `forge.data`,
`forge.nn`, `forge.optim`, `forge.training`, `forge.save_model`/
`save_checkpoint`) -- this script adds no framework logic of its own, only
example wiring. See `examples/image_folder_classification/README.md` for
prerequisites and expected output.

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

from forge.data import Compose, DataLoader, ImageFolder, Normalize, Resize, random_split
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_classes, load_preprocessing
from forge.training import Accuracy, interpret_classification, predict, save_and_verify, start_training_session

try:
    from .generate_dataset import _render_image, generate_dataset
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/image_folder_classification/train.py`)
    from generate_dataset import _render_image, generate_dataset
    from model import build_model

_TRAIN_FRACTION = 0.8
# Milestone 70: the target shape every sample is resized to. Source images
# are mixed-resolution by construction (see `generate_dataset.py`).
_RESIZE_SIZE = (64, 64)


def build_transform():
    """`Resize` then pixel-scale to `[0, 1]` -- the exact preprocessing every
    training sample and every later inference image must go through.

    Milestone 71: pixel scaling is `Normalize(mean=0.0, std=255.0)`
    (`(x - 0) / 255 == x / 255`), not the `Lambda(lambda x: x * (1/255))`
    earlier milestones used -- `Lambda` wraps an arbitrary Python callable
    with no safe serialized representation (see
    `forge/serialization/transforms.py`), so it cannot be part of what
    `save_model(..., preprocessing=...)` persists. `Normalize` expresses the
    identical computation as an explicit, serializable configuration; this
    is the only change `build_transform()` needed to become fully
    persistable. `Resize` must still run first (it expects raw `[0, 255]`
    8-bit-range values -- see `forge/data/transforms.py`'s `Resize`
    docstring).
    """
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


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
    parser.add_argument("--min-size", type=int, default=48, help="Minimum generated image width/height.")
    parser.add_argument("--max-size", type=int, default=128, help="Maximum generated image width/height.")
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
        print(f"Generating mixed-resolution synthetic shape dataset into '{args.data_root}' "
              f"(sizes in [{args.min_size}, {args.max_size}]^2, seed={args.seed}) ...")
        generate_dataset(args.data_root, samples_per_class=args.samples_per_class,
                          min_size=args.min_size, max_size=args.max_size, seed=args.seed)

    # Milestone 73: dataset loading/splitting is independent of both
    # forge.random and DataLoader shuffling (ImageFolder/random_split use
    # their own args.seed-derived numpy.random.Generator), so it can safely
    # run before start_training_session() below seeds forge.random itself.
    print(f"Loading images from '{args.data_root}' ...")
    full_dataset, train_ds, test_ds = build_datasets(args.data_root, split_seed=args.seed)
    print(f"classes: {full_dataset.classes}  class_to_idx: {full_dataset.class_to_idx}")
    print(f"train: {len(train_ds)} samples, test: {len(test_ds)} samples, "
          f"resized to (3, {_RESIZE_SIZE[0]}, {_RESIZE_SIZE[1]}) via forge.data.transforms.Resize")

    # Milestone 73: start_training_session() replaces this script's own
    # hand-rolled "--resume ? load_checkpoint()+Trainer.resume() :
    # build_model()+Adam+Trainer" branch (see
    # forge/training/session.py) -- and, as a direct consequence, also
    # closes a latent M65-class reproducibility gap this script never had a
    # fix for: session.data_loader_rng's shuffle-stream position is now
    # saved/restored automatically across --resume (previously, resuming
    # this script's shuffle=True train_loader silently diverged from
    # continuous training, exactly the bug M65 fixed for
    # examples/regression/train.py alone).
    session = start_training_session(
        build_model=lambda: build_model(num_classes=len(full_dataset.classes)),
        build_optimizer=lambda params: Adam(params, lr=args.lr),
        loss_fn=CrossEntropyLoss(),
        seed=args.seed,
        device=args.device,
        metrics=[Accuracy()],
        resume=args.resume,
    )
    if session.resumed:
        print(f"Resumed from checkpoint '{args.resume}' at epoch={session.trainer.epoch}, "
              f"global_step={session.trainer.global_step}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=session.data_loader_rng)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    start = time.perf_counter()
    history = session.trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
    duration = time.perf_counter() - start

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"loss: {history[0].train_loss:.4f} -> {history[-1].train_loss:.4f}")
    print(f"val accuracy: {history[-1].val_metrics['accuracy']:.2%}")

    final_eval = session.trainer.evaluate(test_loader)
    print(f"\nFinal test evaluation: loss={final_eval.loss:.4f}, accuracy={final_eval.metrics['accuracy']:.2%}")

    # Milestone 73: TrainingSession.save_checkpoint() automatically folds
    # session.data_loader_rng's current stream position into the
    # checkpoint's extra dict -- see this function's earlier comment.
    session.save_checkpoint(str(checkpoint_path))
    print(f"\nSaved checkpoint -> {checkpoint_path}")
    # Milestone 71: `preprocessing=` saves the exact `Resize`/`Normalize`
    # pipeline used above as a sibling metadata entry in the same model
    # file -- see `forge/serialization/model.py`'s `save_model()` docstring
    # and `docs/architecture/persistence.md`'s **Preprocessing metadata**
    # section. Milestone 72: `classes=` saves `full_dataset.classes` (the
    # same ordered vocabulary `ImageFolder` already computed from the
    # directory layout) as a second sibling metadata entry, so the model
    # file alone -- not a hand-written `classes.json` sidecar a caller had
    # to remember to keep next to it -- is enough to turn a predicted index
    # back into a class name later. See `load_classes()` and
    # `forge.training.interpret_classification()`. Milestone 78:
    # `save_and_verify()` saves the model, then reloads it fresh and
    # confirms the reload's prediction on `query_x_batch` matches the
    # pre-save model -- the same "save -> reload -> predict() must agree"
    # proof this script used to hand-write, now the shared Milestone 78
    # abstraction (`forge/training/inference.py`) every retrofitted Forge
    # example calls the same way.
    query_x, query_y = test_ds[0]
    query_x_batch = query_x.to(args.device).reshape(1, *query_x.shape)
    reloaded = save_and_verify(
        session.trainer.model, str(model_path), query_x_batch,
        preprocessing=build_transform(), classes=full_dataset.classes,
    )
    print(f"Saved + verified model + preprocessing + classes ({full_dataset.classes}) -> {model_path}")

    # Section 11: standalone single-image inference through forge.predict(),
    # interpreted into a human-readable class name + confidence via Milestone
    # 72's interpret_classification() -- using `load_classes()`'s own
    # reconstructed vocabulary, not the in-memory `full_dataset.classes`
    # this process still happens to have, to prove the file alone is enough.
    reloaded_classes = load_classes(str(model_path))
    result = interpret_classification(predict(reloaded, query_x_batch), reloaded_classes)[0]
    true_idx = int(query_y.numpy())
    true_name = full_dataset.classes[true_idx]
    print(f"\nInference demo (test sample 0, true class: {true_name}):")
    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence:.1%}")

    # Section 12 (Milestone 70, preprocessing reloaded from disk since
    # Milestone 71): inference on a brand-new image that was never part of
    # the dataset, at a resolution the model was never trained at
    # (deliberately outside [--min-size, --max-size]) -- proving the *same*
    # preprocessing, reconstructed from the saved file via
    # `load_preprocessing()` rather than reused from this process's own
    # `build_transform()` call, makes an arbitrary new image usable by the
    # saved model. This is the in-process half of the M71/M72 proof;
    # `infer.py` (run as a genuinely separate process below) is the other
    # half, and now needs only `--model` and `--image` -- no separate
    # `--classes` sidecar file (Milestone 72).
    reloaded_preprocessing = load_preprocessing(str(model_path))

    new_image_shape = full_dataset.classes[0]
    new_image_rng = np.random.default_rng(args.seed + 1000)
    new_image = _render_image(new_image_rng, new_image_shape, width=200, height=140)
    new_image_path = output_dir / "new_mixed_resolution_query.png"
    new_image.save(new_image_path)

    raw_query = ImageFolder._load_image(new_image_path)
    preprocessed_query = reloaded_preprocessing(raw_query)
    assert preprocessed_query.shape == (3, *_RESIZE_SIZE), "Resize did not normalize the new image's shape"
    new_query_batch = preprocessed_query.to(args.device).reshape(1, *preprocessed_query.shape)
    new_pred = predict(reloaded, new_query_batch)
    new_result = interpret_classification(new_pred, reloaded_classes)[0]
    print(f"\nNew mixed-resolution image (200x140, true class: {new_image_shape}, "
          f"never seen during training), preprocessing + classes reconstructed from '{model_path}':")
    print(f"Prediction: {new_result.label}")
    print(f"Confidence: {new_result.confidence:.1%}")

    print("\nInspect the generated artifacts with the CLI:")
    print(f"  python -m forge model inspect {model_path}")
    print(f"  python -m forge checkpoint inspect {checkpoint_path}")
    print(f"  python -m forge model predict {model_path} --image {new_image_path}")
    print("\nRun standalone inference in a fresh process (Milestone 71/72):")
    print(f"  python -m examples.image_folder_classification.infer --model {model_path} "
          f"--image {new_image_path}")


if __name__ == "__main__":
    main()
