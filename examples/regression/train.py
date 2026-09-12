"""Forge Milestone 60/65/83: an end-to-end tabular regression example.

```text
make_datasets() -> DataLoader -> Trainer -> MLP (Linear/ReLU/Linear/ReLU/Linear)
    -> MSELoss -> Adam -> forge.train_and_save() -> forge.predict_tensor_artifact()
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`) -- this script adds
no framework logic of its own, only example wiring, exactly like
`examples/mnist/train.py`. See `examples/regression/README.md` for
prerequisites, expected behavior, and how to reproduce checkpoint/resume and
model persistence from this one entry point. **Milestone 83** split this
script's single `start_training_session()` call into the same
`--resume`-or-fresh two-branch shape `examples/mnist/train.py` and
`examples/image_folder_classification/train.py` already have (Milestones
80/81): the fresh path now trains and saves through one
`forge.train_and_save()` call -- including the fitted `Normalize` feature
transform as `preprocessing=`, so the saved artifact alone is enough to
standardize a brand-new raw feature vector -- while `--resume` keeps using
`start_training_session()` unchanged. See
`docs/development/m83-regression-artifact-workflow.md`.

## Determinism

`forge.random.seed(args.seed)` (called directly on the fresh path, or via
`start_training_session()` on `--resume`) governs `Linear` parameter
initialization at model construction; the `DataLoader` shuffle generator is
a second, independent stream; `examples.regression.dataset.generate_raw()`
uses its own explicit `numpy.random.default_rng(args.seed)` for the
synthetic data -- three separate deterministic streams, matching
`examples/mnist/train.py`'s documented policy (see
`docs/architecture/persistence.md`'s checkpoint RNG policy).

**Milestone 65**: the `DataLoader` shuffle generator's exact position in its
stream is saved into `save_checkpoint(..., extra=...)` and restored on
`--resume`, closing the one reproducibility gap Milestone 65's baseline
investigation found (a resumed run previously re-seeded a *fresh*
`--seed`-derived generator rather than continuing the interrupted run's
shuffle stream -- see `docs/development/m65-reproducible-training.md`). As of
**Milestone 83**, the fresh path writes this "data_loader_rng_state" extra
field itself via plain `forge.save_checkpoint()` (the exact key
`start_training_session()`'s resume path already reads), exactly like
`examples/image_folder_classification/train.py`'s own Milestone 80 fresh
path -- so a later `--resume` continues this run's exact shuffle stream with
zero regression. Every run also writes a JSON "run record" (config +
per-epoch metrics history + final evaluation + runtime) next to its
checkpoint -- see `experiment.py` -- so two runs can be compared with
`compare.py` without parsing terminal output.

## Usage

```bash
# First run: train from scratch, save a checkpoint + model + run record.
python -m examples.regression.train --epochs 40 --device cpu

# Continue training from the saved checkpoint for 10 more epochs.
python -m examples.regression.train --resume artifacts/regression_checkpoint.forge --epochs 10

# CUDA (requires a working Forge CUDA backend).
python -m examples.regression.train --epochs 40 --device cuda

# Compare two completed runs' configuration/metrics.
python -m examples.regression.compare artifacts/regression_history.json artifacts2/regression_history.json
```
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import forge
from forge.data import DataLoader
from forge.nn import MSELoss
from forge.optim import Adam
from forge.training import MeanAbsoluteError, Trainer, save_and_verify, start_training_session

try:
    from .dataset import N_FEATURES, generate_raw, make_datasets
    from .experiment import extend_run_record, load_run_record, new_run_record, save_run_record
    from .model import build_model
except ImportError:  # running as a plain script (`python examples/regression/train.py`)
    from dataset import N_FEATURES, generate_raw, make_datasets
    from experiment import extend_run_record, load_run_record, new_run_record, save_run_record
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
    history_path = output_dir / "regression_history.json"

    print(f"Generating synthetic tabular regression data (seed={args.seed}) ...")
    train_ds, val_ds, test_ds, stats = make_datasets(args.n_train, args.n_val, args.n_test, seed=args.seed)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples, test: {len(test_ds)} samples, features: 8")
    baseline_mse = stats["y_train_var"]
    print(f"Trivial baseline (predict train mean): MSE = {baseline_mse:.4f}, RMSE = {math.sqrt(baseline_mse):.4f}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    loss_fn = MSELoss()

    # Milestone 81/83: the sample save_and_verify()/train_and_save() need to
    # prove the eventual artifact is portable -- picked once, up front, since
    # both branches below end up saving+verifying against it. Already
    # normalized (test_ds applies the fitted Normalize transform on
    # __getitem__), matching predict()'s "already-prepared input" calling
    # convention exactly like image_folder_classification's own
    # already-Resize+Normalize'd query image.
    # `-1`-inferred reshape is a CPU-`Tensor.reshape` convenience only
    # (`CUDABackend.reshape` requires every dim explicit, see
    # `forge/backend/cuda/backend.py`); pass `N_FEATURES` explicitly so this
    # round trip works identically on both devices -- a pre-existing bug
    # (Milestone 62 found this crashing `--device cuda`'s final persistence
    # check via `ShapeMismatchError`, unrelated to this milestone's own
    # changes) fixed here using the same pattern
    # `examples/waveform_classification/train.py` uses for the same reason.
    query_x, _ = test_ds[0]
    query_x_batch = query_x.to(args.device).reshape(1, N_FEATURES)

    start = time.perf_counter()
    if args.resume:
        # Milestone 73: start_training_session() replaces this branch's own
        # hand-rolled load_checkpoint()+Trainer.resume() sequence, restoring
        # session.data_loader_rng's shuffle-stream position from the
        # checkpoint's "data_loader_rng_state" extra field (Milestone 83: now
        # written on the fresh path below via plain forge.save_checkpoint(),
        # but the same key, so this branch reads either kind of checkpoint
        # identically).
        session = start_training_session(
            build_model=build_model,
            build_optimizer=lambda params: Adam(params, lr=args.lr),
            loss_fn=loss_fn,
            seed=args.seed,
            device=args.device,
            metrics=[MeanAbsoluteError()],
            resume=args.resume,
        )
        print(f"Resumed from checkpoint '{args.resume}' at epoch={session.trainer.epoch}, "
              f"global_step={session.trainer.global_step}")
        run_record = load_run_record(history_path)
        if run_record is None:
            print(f"(no existing run record at '{history_path}' -- starting a new one)")
            run_record = new_run_record(vars(args))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   generator=session.data_loader_rng)
        history = session.trainer.fit(train_loader, epochs=args.epochs, validation_loader=val_loader)
        session.save_checkpoint(str(checkpoint_path))
        final_eval = session.trainer.evaluate(test_loader)
        # Milestone 78/83: save_and_verify() saves the model (+ the fitted
        # Normalize transform as preprocessing=, Milestone 83) then reloads
        # it fresh and confirms the reload's prediction on query_x_batch
        # matches the model that was just saved. The fresh (non-resume)
        # branch below gets this for free from train_and_save() instead.
        save_and_verify(
            session.trainer.model, str(model_path), query_x_batch, preprocessing=stats["transform"],
        )
    else:
        # Milestone 83: the fresh (non-resume) path now trains through
        # forge.train_and_save() (forge/training/api.py, Milestone 81)
        # instead of building its Trainer via start_training_session() --
        # the same fresh/resume split examples/mnist/train.py and
        # examples/image_folder_classification/train.py already have
        # (Milestone 80). This branch builds data_loader_rng itself (exactly
        # what start_training_session() would have built internally) and
        # saves its state into the checkpoint's own "data_loader_rng_state"
        # extra field via plain forge.save_checkpoint() -- so a later
        # --resume (the branch above) still continues this run's exact
        # shuffle stream, preserving the Milestone 65/73 resume-equivalence
        # guarantee this example has had since Milestone 73, with zero
        # regression.
        run_record = new_run_record(vars(args))

        forge.random.seed(args.seed)
        data_loader_rng = np.random.default_rng(args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=data_loader_rng)
        model = build_model().to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        train_result = forge.train_and_save(
            model, train_loader,
            loss=loss_fn,
            optimizer=optimizer,
            epochs=args.epochs,
            validation_dataset=val_loader,
            device=args.device,
            metrics=[MeanAbsoluteError()],
            path=str(model_path),
            sample=query_x_batch,
            # Milestone 83: the fitted train-split Normalize transform
            # (dataset.py's make_datasets()) is saved alongside the model so
            # a later process can standardize a brand-new raw feature vector
            # the same way -- see forge.predict_tensor_artifact() below --
            # mirroring image_folder_classification's Resize/Normalize
            # preprocessing persistence (Milestones 71/81). No classes= --
            # a regression model has no class vocabulary (see
            # forge/training/inference.py's predict_tensor_artifact()
            # docstring).
            preprocessing=stats["transform"],
        )
        history = train_result.history

        epoch, global_step = len(history), len(history) * len(train_loader)
        forge.save_checkpoint(
            str(checkpoint_path), model, optimizer, epoch=epoch, global_step=global_step,
            extra={"data_loader_rng_state": data_loader_rng.bit_generator.state},
        )
        # The held-out test split (distinct from the val split
        # train_and_save() already validated against every epoch) still
        # needs its own final evaluation -- train_and_save()/train() only
        # ever evaluate validation_dataset (mirroring Trainer.fit()'s own
        # contract). Reuses the already-trained `model`/`optimizer` (still
        # in scope; train_and_save() mutates `model` in place, exactly like
        # train()) through a throwaway Trainer purely for its existing
        # evaluate() -- no new evaluation logic.
        final_eval = Trainer(
            model, loss_fn, optimizer, device=args.device, metrics=[MeanAbsoluteError()], verbose=False,
        ).evaluate(test_loader)
    duration = time.perf_counter() - start

    samples_per_sec = (len(train_ds) * args.epochs) / duration if duration > 0 else float("inf")
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s "
          f"({samples_per_sec:.0f} train samples/sec).")
    print(f"train MSE: {history[0].train_loss:.4f} -> {history[-1].train_loss:.4f}")
    print(f"val MSE:   {history[-1].val_loss:.4f}  (RMSE {math.sqrt(history[-1].val_loss):.4f})")
    print(f"val MAE:   {history[-1].val_metrics['mae']:.4f}")

    test_mse = final_eval.loss
    print(f"\nFinal test evaluation: MSE={test_mse:.4f}, RMSE={math.sqrt(test_mse):.4f}, "
          f"MAE={final_eval.metrics['mae']:.4f}")
    print(f"Trivial baseline MSE was {baseline_mse:.4f} -- "
          f"model achieves a {(1 - test_mse / baseline_mse):.1%} reduction over predicting the mean.")

    print(f"\nSaved checkpoint -> {checkpoint_path}")

    extend_run_record(run_record, history=history, final_eval=final_eval, duration_seconds=duration)
    save_run_record(history_path, run_record)
    print(f"Saved run record -> {history_path}")

    print(f"Saved + verified model + preprocessing -> {model_path}")

    print("\nInspect the generated artifacts with the Milestone 19 CLI:")
    print(f"  python -m forge model inspect {model_path}")
    print(f"  python -m forge checkpoint inspect {checkpoint_path}")
    print("\nCompare this run against another with the Milestone 65 tool:")
    print(f"  python -m examples.regression.compare {history_path} <other_run>/regression_history.json")

    # Milestone 83: fresh-process-style, high-level artifact inference on a
    # brand-new *raw* feature vector -- the model file alone (not this
    # process's in-memory stats["transform"]) standardizes and predicts,
    # mirroring image_folder_classification's forge.predict_artifact() demo
    # (Milestone 82) for this example's own artifact shape.
    new_raw_x, _ = generate_raw(1, seed=args.seed + 1000)
    prediction = forge.predict_tensor_artifact(str(model_path), new_raw_x, device=args.device)
    print(f"\nNew raw feature vector {new_raw_x[0].tolist()}, "
          f"preprocessing reconstructed from '{model_path}':")
    print(f"Prediction: {float(prediction.numpy()[0, 0]):.4f}")


if __name__ == "__main__":
    main()
