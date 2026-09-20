"""`forge.train_image_classifier()`: the image-folder-to-artifact convenience call (Milestone 107).

## The gap this closes

Milestone 107's real-world acceptance workload -- `sandbox/petimages`, an
external ~25,000-image cat/dog dataset arranged one directory per class --
was trained successfully on Forge's *existing* public API
(`sandbox/t1_cat_dog/train.py`), proving the framework itself already
handles a realistic image-classification workload. But that script needed a
developer to manually assemble every stage of the pipeline by hand: build a
`Resize`+`Normalize` transform (and remember to build it *twice* -- once for
`ImageFolder(transform=...)`, again for `train_and_save(preprocessing=...)`,
since nothing kept the two in sync); pick a CNN architecture and compute its
flattened `Linear` input size by hand for the chosen image resolution;
compute train/validation split sizes from a fraction; reach past
`ImageFolder`'s public surface into the underscore-prefixed `_load_image()`
and mutate `.samples` directly just to skip 2 corrupt JPEGs out of 25,000
(fixed properly in `forge.data.ImageFolder`'s own `on_error=` parameter,
Milestone 107 -- see that module); and hand-pick a batched `sample` Tensor
purely to satisfy `save_and_verify()`'s calling convention. None of that is
image-classification *domain* knowledge -- it is Forge plumbing a developer
had to learn before ever getting to "train a classifier on my folder of
images."

`train_image_classifier()` is that plumbing, written once, for exactly the
common case `sandbox/petimages` represents: a directory of `root/class_a/`,
`root/class_b/`, ... image files, one classifier out.

```python
result = forge.train_image_classifier(
    "sandbox/petimages",
    path="cat_dog_model.forge",
    epochs=5,
    device="cuda",
)
result.val_metrics["accuracy"]  # e.g. 0.833
result.artifact_path            # "cat_dog_model.forge"
```

Then, exactly as with every other Forge artifact, `forge.predict_artifact()`
/ `forge.load_predictor()` / `forge model predict` (all unmodified by this
milestone) consume the saved file.

## What this composes (nothing new underneath)

`train_image_classifier()` builds no new training loop, no new persistence
format, and no second `Dataset`/`Trainer`/artifact abstraction. It is
argument selection over existing pieces, in this order:

1. `forge.data.Compose([Resize(image_size), Normalize(mean=0.0, std=255.0)])`
   -- one transform object, built once, used as both `ImageFolder`'s
   `transform=` (applied while training) *and* `train_and_save()`'s
   `preprocessing=` (persisted for inference) -- eliminating the
   build-it-twice duplication above by construction, not by convention.
2. `forge.data.ImageFolder(data_dir, transform=..., on_error=on_error)` --
   discovers classes and samples; `on_error="skip"` (this function's
   default, unlike `ImageFolder`'s own default of `"raise"` -- see below)
   reports and drops unreadable files rather than failing the run.
3. `forge.data.random_split()` into `1 - val_fraction` / `val_fraction`,
   `forge.data.DataLoader` for both splits.
4. A default CNN (`_default_image_classifier_cnn()`, this module) if
   `model=` is not given -- see **Default architecture** below -- or the
   caller's own `forge.nn.Module` used exactly as given.
5. `forge.nn.CrossEntropyLoss()` and `forge.optim.Adam(lr=learning_rate)` --
   image-folder classification is unambiguously multi-class classification,
   so this one case is where a default loss *is* safe to choose
   automatically, unlike `forge.train()`'s deliberate refusal to guess one
   in the general case (see `forge/training/api.py`'s module docstring).
6. `forge.training.train_and_save()` (Milestone 81), unmodified -- this
   function performs no training itself.

## Default architecture

`_default_image_classifier_cnn()` is exactly the three-block CNN
(`Conv2d`/`BatchNorm2d`/`ReLU`/`MaxPool2d`, x3, widths 16/32/64, then
`Linear(flattened, 128) -> ReLU -> Dropout(0.3) -> Linear(128, num_classes)`)
`sandbox/t1_cat_dog/model.py` hand-wrote and validated end-to-end on the
real `petimages` dataset (83.3% validation accuracy after 5 epochs on the
reference 940MX) -- not a new, unvalidated architecture. The flattened
`Linear` input size is computed from `image_size` (three rounds of "conv
k=3, no padding" then "max-pool 2") rather than hand-derived per resolution,
and a resolution too small for three such blocks raises a clear `DataError`
naming the problem, pointing at `model=` as the escape hatch, rather than
failing deep inside `Linear`'s shape check.

## Preflight (Milestone 115)

Everything below that a caller can get wrong and Forge can know in advance is
checked **before epoch 1**, so a mistake costs seconds, not a training run
(estimated for a 25,000-image, 5-epoch run on the reference machine: ~10 minutes
scanning, then ~45 minutes training on the GPU or ~85 on the CPU):

- `path`: its directory must exist and it must not be a directory
  (`PersistenceError`) -- checked *before* the dataset scan, since with
  `on_error="skip"` the scan alone decodes every image once.
- `model=` must be a `forge.nn.Module` (`TrainerError`, also before the scan).
- After the split, a caller-supplied `model=` is *run once* on two real
  preprocessed images and must return `(batch, len(classes))` scores
  (`TrainerError`). Running it, rather than reading its structure, checks the
  input shape and the output width exactly for any `Module`. This closes the
  worst late failure: a model with the wrong number of outputs used to train,
  save and verify without complaint, and fail only at the first `predict()`.
- The untrained model, the persisted preprocessing and the class list are
  saved through the real `save_model()` to a temporary sibling of `path`, which
  is removed at once (`PersistenceError`): an unwritable path, an invalid
  filename or an unregistered `Module` fails here. The final artifact is still
  written only after training. The helpers are shared with the tabular
  workflows (`forge/training/_preflight.py`).

The default architecture is not probed (it is built here, sized for `image_size`).

## Advanced control

Every meaningful decision remains overridable, never hidden:
`model=` replaces the default architecture entirely (the only requirement
is that it accept `(N, 3, *image_size)` and end in `num_classes` logits);
`epochs`/`batch_size`/`learning_rate`/`val_fraction`/`image_size`/`device`/
`seed`/`on_error`/`path` are all plain keyword arguments, not buried
configuration. What is deliberately *not* exposed here: optimizer choice
(always `Adam`), loss choice (always `CrossEntropyLoss`), and the exact
default architecture's internal widths -- a caller needing any of those
builds the pipeline directly from `ImageFolder`/`train_and_save()`, exactly
as `sandbox/t1_cat_dog/train.py` still does; this function does not replace
that lower-level path, and does not become harder to bypass because it
exists.

## Explicitly out of scope

No hyperparameter search, no architecture search, no automatic image-size
selection, no automatic train/validation strategy beyond one `random_split`,
no experiment tracking. See `docs/development/progress.md`'s Milestone 107
entry for the full design rationale (why this interface over the
alternatives considered) and the real `petimages` acceptance-test results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import random as forge_random
from ..backend.device import Device
from ..data.dataloader import DataLoader
from ..data.dataset import random_split
from ..data.image_folder import ImageFolder
from ..data.transforms import Compose, Normalize, Resize
from ..exceptions import DataError, TrainerError
from ..nn.batchnorm import BatchNorm2d
from ..nn.container import Sequential
from ..nn.conv import Conv2d
from ..nn.dropout import Dropout
from ..nn.flatten import Flatten
from ..nn.linear import Linear
from ..nn.loss import CrossEntropyLoss
from ..nn.module import Module
from ..nn.pooling import MaxPool2d
from ..nn.activation import ReLU
from ..optim.adam import Adam
from ..tensor.tensor import Tensor
from ._preflight import check_model_contract, check_save_path, preflight_save
from .api import TrainingResult, train_and_save
from .metrics import Accuracy


def _default_image_classifier_cnn(num_classes: int, image_size: "tuple[int, int]") -> Sequential:
    """The three-block CNN `train_image_classifier()` uses when `model=` is not given.

    See this module's docstring, **Default architecture**, for why this
    exact shape (not a new design) and how `image_size` determines the
    final `Linear`'s input width.
    """
    h, w = image_size
    for _ in range(3):
        h, w = h - 2, w - 2
        if h <= 0 or w <= 0:
            raise DataError(
                f"image_size {image_size!r} is too small for the default 3-block CNN "
                "(each block removes 2px via a kernel=3 convolution, then halves via "
                "max-pooling). Pass a larger image_size, or pass model= with a custom "
                "architecture sized for this resolution."
            )
        h, w = h // 2, w // 2
        if h <= 0 or w <= 0:
            raise DataError(
                f"image_size {image_size!r} is too small for the default 3-block CNN "
                "after max-pooling. Pass a larger image_size, or pass model= with a "
                "custom architecture sized for this resolution."
            )
    flattened = 64 * h * w
    return Sequential(
        Conv2d(3, 16, kernel_size=3),
        BatchNorm2d(16),
        ReLU(),
        MaxPool2d(2),
        Conv2d(16, 32, kernel_size=3),
        BatchNorm2d(32),
        ReLU(),
        MaxPool2d(2),
        Conv2d(32, 64, kernel_size=3),
        BatchNorm2d(64),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(flattened, 128),
        ReLU(),
        Dropout(0.3),
        Linear(128, num_classes),
    )


@dataclass(frozen=True)
class ImageClassifierResult:
    """`train_image_classifier()`'s return value (Milestone 107).

    Everything a caller needs to answer "what happened when I trained this
    classifier?" and "where is it?" in one object -- the same transparency
    principle `TrainAndSaveResult` (`forge/training/api.py`) already
    established, extended with the image-folder-specific facts
    `train_image_classifier()` itself discovered (`classes`, dataset/split
    sizes, which files were skipped) that a plain `TrainAndSaveResult`
    has no field for.

    - `history` -- the underlying `TrainingResult` (per-epoch train/val
      loss and metrics; also carries the trained in-memory `Module`).
    - `train_loss`/`train_metrics`/`val_loss`/`val_metrics` -- the final
      epoch's results, copied here exactly as `TrainAndSaveResult` does.
    - `model` -- the freshly reloaded, verified `Module` (`save_and_verify()`'s
      own reload, not the in-memory training-time object).
    - `artifact_path` -- the `.forge` file path that was written and verified.
    - `classes` -- the class-name vocabulary discovered from `data_dir`'s
      subdirectories (also already persisted inside the artifact itself).
    - `dataset_size`/`train_size`/`val_size` -- sample counts after any
      `on_error="skip"` filtering.
    - `skipped_images` -- `(path, reason)` for every file `ImageFolder`
      could not decode when `on_error="skip"` (empty when `on_error="raise"`,
      since a real read failure would have raised instead of reaching here).
    """

    history: TrainingResult
    train_loss: float
    train_metrics: "dict[str, float]"
    val_loss: "float | None"
    val_metrics: "dict[str, float]"
    model: Module
    artifact_path: str
    classes: "list[str]"
    dataset_size: int
    train_size: int
    val_size: int
    skipped_images: "list[tuple[Path, str]]"


def train_image_classifier(
    data_dir: "str | Path",
    *,
    path: str,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 4e-4,
    val_fraction: float = 0.2,
    image_size: "tuple[int, int]" = (64, 64),
    model: "Module | None" = None,
    device: "str | Device | None" = None,
    seed: int = 0,
    on_error: str = "skip",
    verbose: bool = True,
) -> ImageClassifierResult:
    """Train an image-folder classifier and save it as a portable artifact, in one call.

    ```python
    result = forge.train_image_classifier(
        "sandbox/petimages", path="cat_dog_model.forge", epochs=5, device="cuda",
    )
    ```

    `data_dir` must be a directory-per-class layout `forge.data.ImageFolder`
    already understands (`root/class_a/*.jpg`, `root/class_b/*.jpg`, ...) --
    the exact same contract, same errors (`forge.DataError` for a missing
    root, no class subdirectories, or no supported image files). This
    function is the common-case convenience layer over `ImageFolder` +
    `forge.training.train_and_save()`; it replaces neither -- see this
    module's own docstring for the full composition and what remains
    overridable.

    **Preprocessing.** Every image is resized to `image_size` (default
    `(64, 64)`, matching `sandbox/t1_cat_dog`'s validated real-dataset run)
    then pixel-scaled to `[0, 1]` (`Normalize(mean=0.0, std=255.0)`) -- the
    same `Compose([Resize(...), Normalize(...)])` pipeline every existing
    Forge image example uses, built once and reused for both training and
    the artifact's persisted `preprocessing=`, so a later
    `forge.predict_artifact()`/`load_predictor()` call needs no separate
    reimplementation.

    **Unreadable files.** `on_error` is passed straight through to
    `ImageFolder(..., on_error=on_error)` -- `"skip"` (this function's
    default, matching the real `petimages` dataset's 2-bad-files-out-of-
    25,000 shape) reports and excludes files that fail to decode;
    `"raise"` (`ImageFolder`'s own default, for a caller who wants a bad
    file to fail the run loudly) does not pre-scan at all. Either way, see
    `ImageFolder`'s own docstring for the exact contract; this function adds
    no data-cleaning logic beyond that parameter.

    **Split.** `val_fraction` (default `0.2`) is turned into `random_split()`
    sample counts (rounded, with at least 1 sample in each split, else
    `forge.DataError`), split via a `seed`-derived generator -- the same
    determinism convention every Forge example's own `--seed` already
    documents (see this function's `seed` parameter below).

    **Model.** When `model=` is omitted, a default CNN sized for
    `image_size`/the discovered class count is constructed (see
    `_default_image_classifier_cnn()` and this module's **Default
    architecture** section) after seeding `forge.random.seed(seed)` for
    reproducible initialization. When `model=` is given, it is used exactly
    as given (not reseeded), but it is verified on two real preprocessed
    images before epoch 1: it must accept `(N, 3, *image_size)` input and
    return `(N, len(classes))` raw scores (`forge.TrainerError` otherwise).

    **Loss/optimizer.** Always `CrossEntropyLoss()` and
    `Adam(model.parameters(), lr=learning_rate)` -- image-folder
    classification has one obvious loss choice, unlike `forge.train()`'s
    general case (see this module's docstring). Neither is configurable
    here; use `train_and_save()` directly for a different loss/optimizer.

    **Task/artifact metadata.** Always saved with `task="classification"`
    and `classes=` the discovered class-name vocabulary -- exactly what
    `forge.predict_artifact()`/`forge.predict_model()`/`forge model predict`
    require to consume the artifact with zero extra configuration.

    Returns an `ImageClassifierResult` (see that class's own docstring).
    Raises `forge.DataError` for an invalid `data_dir` (`ImageFolder`'s own
    errors), an invalid `on_error`/`val_fraction`, fewer than 2 discovered
    classes, or an `image_size` too small for the default architecture (only
    when `model=` is not given); `forge.TrainerError` if `model=` is not a
    `forge.nn.Module` or fails the contract above; `forge.PersistenceError` if
    `path` cannot be written or the model cannot be serialised. All of these are
    raised before epoch 1 (see this module's **Preflight** section);
    `forge.TrainerError` is also raised if training itself diverges to NaN/Inf.
    """
    fn = "train_image_classifier"
    if not (0.0 < val_fraction < 1.0):
        raise DataError(f"val_fraction must be strictly between 0 and 1, got {val_fraction!r}.")
    if model is not None and not isinstance(model, Module):
        raise TrainerError(f"{fn}() requires model= to be a forge.nn.Module, got {type(model).__name__}.")
    check_save_path(path, fn)  # before the dataset scan: with on_error="skip" the scan alone can take minutes

    transform = Compose([Resize(image_size), Normalize(mean=0.0, std=255.0)])
    full_dataset = ImageFolder(data_dir, transform=transform, on_error=on_error)

    if len(full_dataset.classes) < 2:
        raise DataError(
            f"train_image_classifier() requires at least 2 classes, found "
            f"{len(full_dataset.classes)} under '{data_dir}'."
        )

    n_val = max(1, round(len(full_dataset) * val_fraction))
    n_train = len(full_dataset) - n_val
    if n_train < 1:
        raise DataError(
            f"train_image_classifier() has no training samples left after reserving "
            f"{n_val} for validation out of {len(full_dataset)} total (val_fraction="
            f"{val_fraction!r}). Use a smaller val_fraction or a larger dataset."
        )

    if verbose and full_dataset.skipped_samples:
        print(f"Skipped {len(full_dataset.skipped_samples)} unreadable image(s) (on_error='skip'):")
        for skipped_path, reason in full_dataset.skipped_samples:
            print(f"  {skipped_path}: {reason}")

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=np.random.default_rng(seed)
    )
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, generator=np.random.default_rng(seed)
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    supplied = model is not None
    if not supplied:
        forge_random.seed(seed)
        model = _default_image_classifier_cnn(len(full_dataset.classes), image_size)
    model.to(Device.parse(device) if device is not None else (model.device or Device.parse("cpu")))

    sample_x, _ = val_subset[0]
    sample_batch = sample_x.reshape(1, *sample_x.shape)

    # Everything below is checkable now and would otherwise surface only after (or, for a wrong
    # output width, only after saving -- at the first predict()) the whole training run.
    if supplied:
        train_x, _ = train_subset[0]
        probe = Tensor(np.stack([train_x.numpy(), sample_x.numpy()]))
        check_model_contract(
            model, probe, len(full_dataset.classes), "images", f"(batch, {', '.join(map(str, sample_x.shape))})",
            f"one raw score per class, for the {len(full_dataset.classes)} classes {full_dataset.classes!r}", fn,
        )
    preflight_save(model, path, transform, full_dataset.classes, "classification", fn)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    result = train_and_save(
        model, train_loader,
        loss=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        validation_dataset=val_loader,
        device=device,
        metrics=[Accuracy()],
        verbose=verbose,
        path=path,
        sample=sample_batch,
        preprocessing=transform,
        classes=full_dataset.classes,
        task="classification",
    )

    return ImageClassifierResult(
        history=result.history,
        train_loss=result.train_loss,
        train_metrics=result.train_metrics,
        val_loss=result.val_loss,
        val_metrics=result.val_metrics,
        model=result.model,
        artifact_path=result.artifact_path,
        classes=full_dataset.classes,
        dataset_size=len(full_dataset),
        train_size=n_train,
        val_size=n_val,
        skipped_images=list(full_dataset.skipped_samples),
    )


__all__ = ["train_image_classifier", "ImageClassifierResult"]
