"""`TrainingSession`/`start_training_session()`: the fresh-or-resumed `Trainer`
construction workflow every checkpoint-capable example repeats (Milestone 73).

By M72, seven of Forge's ten trainable examples (`mnist`, `regression`,
`resnet`, `segmentation`, `autoencoder`, `waveform_classification`,
`image_folder_classification`) each independently hand-rolled the identical
~10-line branch in their own `train.py`:

```python
if args.resume:
    checkpoint = load_checkpoint(args.resume, device=args.device)
    model = checkpoint.model
    optimizer = checkpoint.optimizer
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[...])
    trainer.resume(checkpoint)
else:
    model = build_model().to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device, metrics=[...])
```

`start_training_session()` is that branch, written once. It is deliberately
*not* a `Trainer` change and *not* a generic training-configuration object --
`Trainer` itself is untouched, and the only inputs this function needs beyond
what `Trainer` already requires are the two factories (`build_model`,
`build_optimizer`) a fresh run needs and a checkpoint path a resumed run needs
instead.

## The second, latent bug this closes

Milestone 65 found and fixed a real reproducibility gap in
`examples/regression/train.py`: with `shuffle=True` (every example's actual
default), resuming from a checkpoint re-seeded a *fresh* `--seed`-derived
`DataLoader` generator rather than continuing the interrupted run's shuffle
stream, so a resumed run silently diverged from what continuous training
would have produced. The fix was two lines -- save
`data_rng.bit_generator.state` into `save_checkpoint(..., extra=...)`,
restore it on resume -- but M65 only applied it to `regression` (and it was
later copied by hand into `resnet`). `mnist`, `segmentation`, `autoencoder`,
`waveform_classification`, and `image_folder_classification` never received
it: resuming any of those five with their own `shuffle=True` default has the
exact same latent divergence bug M65 already diagnosed and fixed once.

`TrainingSession` owns a `data_loader_rng` (`numpy.random.Generator`) built
from the *same* `seed` a fresh run's `forge.random.seed(seed)` uses, and
`TrainingSession.save_checkpoint()` automatically folds its current
`bit_generator.state` into the checkpoint's `extra` dict -- restoring it
automatically on the next `start_training_session(..., resume=path)` call.
An example that builds its `DataLoader`s from `session.data_loader_rng`
(instead of its own separately-seeded generator) and saves through
`session.save_checkpoint()` (instead of `trainer.save_checkpoint()`) gets
M65's fix for free, with no per-example code to remember. This reuses M65's
existing mechanism (a JSON-safe `numpy.random.Generator` state living in
`save_checkpoint(..., extra=...)`'s pre-existing caller-defined dict) rather
than building a second reproducibility system -- see
`forge/serialization/checkpoint.py`'s **RNG / determinism policy** section
for what a checkpoint's own `forge.random` state does and does not cover
(exactly the gap `data_loader_rng` closes).

## What this does *not* do

`start_training_session()` never touches `DataLoader` construction, `fit()`,
`evaluate()`, or any other `Trainer` method -- it returns a fully-constructed
`Trainer` (already `.resume()`d, when resuming) plus a `data_loader_rng`, and
the caller drives training exactly as before. It does not read a dataset, it
does not accept a config object/YAML file, and it does not introduce
callbacks -- an example still writes its own `build_model()`/`build_optimizer
()`, still constructs its own `DataLoader`s, and still calls
`trainer.fit(...)` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .. import random as forge_random
from ..backend.device import Device
from ..exceptions import TrainerError
from ..nn.loss import Loss
from ..nn.module import Module
from ..nn.parameter import Parameter
from ..optim.optimizer import Optimizer
from .metrics import Metric
from .trainer import Trainer


@dataclass
class TrainingSession:
    """A ready-to-train `Trainer` plus the `DataLoader` shuffle generator that
    belongs with it -- returned by `start_training_session()`, never
    constructed directly.

    `trainer` is already `.resume()`d (when `resumed` is `True`) -- call
    `trainer.fit(...)`/`trainer.evaluate(...)` exactly as with any other
    `Trainer`. `data_loader_rng` is the generator to construct this run's
    training `DataLoader(..., generator=session.data_loader_rng)` from; it
    already reflects a resumed run's saved shuffle-stream position when one
    was present (see this module's docstring). `checkpoint` is the raw
    `forge.serialization.Checkpoint` a resumed session loaded (`None` for a
    fresh session) -- exposed for any additional caller-defined `extra` field
    beyond `data_loader_rng_state`.
    """

    trainer: Trainer
    data_loader_rng: "np.random.Generator"
    resumed: bool
    checkpoint: "Any | None" = field(default=None)

    def save_checkpoint(self, path: str, *, extra: "dict[str, Any] | None" = None) -> None:
        """`self.trainer.save_checkpoint(path, extra=...)`, with
        `data_loader_rng_state` merged in automatically.

        `extra`, if given, must not itself define `"data_loader_rng_state"`
        (raises `TrainerError` -- this method owns that key so a later
        `start_training_session(..., resume=path)` can always find it).
        """
        if extra is not None and "data_loader_rng_state" in extra:
            raise TrainerError(
                "TrainingSession.save_checkpoint()'s extra dict must not define "
                "'data_loader_rng_state' -- that key is populated automatically from "
                "self.data_loader_rng."
            )
        merged = dict(extra) if extra is not None else {}
        merged["data_loader_rng_state"] = self.data_loader_rng.bit_generator.state
        self.trainer.save_checkpoint(path, extra=merged)


def start_training_session(
    *,
    build_model: "Callable[[], Module]",
    build_optimizer: "Callable[[Iterator[Parameter]], Optimizer]",
    loss_fn: Loss,
    seed: int,
    device: "str | Device" = "cpu",
    metrics: "Iterable[Metric] | None" = None,
    verbose: bool = True,
    prefetch: bool = False,
    prefetch_size: int = 2,
    resume: "str | None" = None,
) -> TrainingSession:
    """Build a fresh `Trainer`, or resume one from a checkpoint -- whichever
    `resume` asks for -- plus this session's `data_loader_rng`.

    ```python
    session = start_training_session(
        build_model=lambda: build_model(num_classes=len(full_dataset.classes)),
        build_optimizer=lambda params: Adam(params, lr=args.lr),
        loss_fn=CrossEntropyLoss(),
        seed=args.seed,
        device=args.device,
        metrics=[Accuracy()],
        resume=args.resume,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=session.data_loader_rng)
    history = session.trainer.fit(train_loader, epochs=args.epochs, validation_loader=test_loader)
    session.save_checkpoint(str(checkpoint_path))
    save_model(session.trainer.model, str(model_path))
    ```

    Always calls `forge.random.seed(seed)` first (Milestone 20's documented
    per-example policy) and constructs `data_loader_rng =
    numpy.random.default_rng(seed)`. When `resume` is falsy (`None` or `""`),
    that is the fresh-run path: `build_model()` (then `.to(device)`) and
    `build_optimizer(model.parameters())` are called, and a new `Trainer` is
    constructed around their result.

    When `resume` names a checkpoint path, `build_model`/`build_optimizer`
    are never called -- `forge.load_checkpoint(resume, device=device)`
    supplies the model and optimizer instead (this also overwrites `forge.
    random`'s state with the checkpoint's own saved state, superseding the
    `seed()` call above -- see `forge/serialization/checkpoint.py`'s **RNG /
    determinism policy**). The resulting `Trainer` is constructed then
    immediately `.resume()`d, and `data_loader_rng`'s `bit_generator.state`
    is overwritten from `checkpoint.extra["data_loader_rng_state"]` when that
    key is present (absent for a checkpoint saved via `Trainer.
    save_checkpoint()` directly rather than through
    `TrainingSession.save_checkpoint()` -- `data_loader_rng` is left at its
    freshly-`seed`-derived state in that case, matching every example's
    pre-Milestone-73 behavior for a checkpoint with no saved shuffle state).

    `loss_fn`/`device`/`metrics`/`verbose`/`prefetch`/`prefetch_size` are
    passed straight through to `Trainer(...)`'s own construction -- see that
    class for their validation and meaning. Raises `TrainerError` if
    `build_model`/`build_optimizer` are not callable.
    """
    if not callable(build_model):
        raise TrainerError(
            f"start_training_session() requires build_model to be callable, got "
            f"{type(build_model).__name__}."
        )
    if not callable(build_optimizer):
        raise TrainerError(
            f"start_training_session() requires build_optimizer to be callable, got "
            f"{type(build_optimizer).__name__}."
        )

    from ..serialization.checkpoint import load_checkpoint as _load_checkpoint

    forge_random.seed(seed)
    data_loader_rng = np.random.default_rng(seed)

    if resume:
        checkpoint = _load_checkpoint(resume, device=device)
        trainer = Trainer(
            model=checkpoint.model,
            loss_fn=loss_fn,
            optimizer=checkpoint.optimizer,
            device=device,
            metrics=metrics,
            verbose=verbose,
            prefetch=prefetch,
            prefetch_size=prefetch_size,
        )
        trainer.resume(checkpoint)
        saved_state = checkpoint.extra.get("data_loader_rng_state") if checkpoint.extra else None
        if saved_state is not None:
            data_loader_rng.bit_generator.state = saved_state
        return TrainingSession(
            trainer=trainer, data_loader_rng=data_loader_rng, resumed=True, checkpoint=checkpoint
        )

    model = build_model().to(device)
    optimizer = build_optimizer(model.parameters())
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        metrics=metrics,
        verbose=verbose,
        prefetch=prefetch,
        prefetch_size=prefetch_size,
    )
    return TrainingSession(trainer=trainer, data_loader_rng=data_loader_rng, resumed=False, checkpoint=None)


__all__ = ["TrainingSession", "start_training_session"]
