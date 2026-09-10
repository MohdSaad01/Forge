# M74 — `forge.data.sequential_split`: closing a duplicated data-preparation gap

## 1. Objective

M73 made the training lifecycle reusable (`start_training_session()`) and,
in doing so, exposed a broader question: is the *data* side of Forge's
developer workflow still comparatively low-level and example-specific? The
brief forbade starting from a generic data framework or another readiness
assessment, and required finding a real, repeated data-preparation problem
in the actual repository, then implementing the smallest reusable
capability that closes it, with a real consumer and an end-to-end
verification.

## 2. Investigation

Read/ran directly, not assumed from prior reports:

- `forge/data/dataset.py`, `dataloader.py`, `image_folder.py`,
  `transforms.py`: `Dataset`, `TensorDataset`, `Subset`, `random_split`,
  `DataLoader`, `ImageFolder`, and the full `Compose`/`Resize`/`Normalize`/
  transform set **already exist** -- `Subset`/`random_split` in particular
  date to the original M5 data-pipeline commit, not something M74 needed to
  invent from the brief's own "Candidate Areas."
- `examples/image_folder_classification/train.py` (the brief's suggested
  primary consumer) already implements almost the entire target workflow:
  `ImageFolder -> Resize/Normalize (Compose) -> random_split -> DataLoader
  -> start_training_session() -> save_model(..., preprocessing=, classes=)
  -> predict() -> interpret_classification()`. This ruled out most of the
  brief's candidate areas outright: dataset splitting, transform pipelines,
  and dataset metadata (`ImageFolder.classes`/`class_to_idx`) are already
  built, tested, and in production use -- adding a second, competing
  version of any of them would violate the "don't build what already
  exists" principle this codebase's own milestones repeatedly emphasize.
- Grepped every `examples/*/dataset.py` for split/slicing logic
  (`n_train|X\[:n|\[start:|\[offset:`). Found the real gap:
  `examples/regression/dataset.py::make_datasets` and
  `examples/waveform_classification/dataset.py::make_datasets` each
  independently hand-rolled the **identical** contiguous-block slicing
  arithmetic --
  ```python
  X_train, y_train = X[:n_train], y[:n_train]
  X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
  X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
  ```
  -- then built three separate `TensorDataset`s from the three slices.
  Both modules' own docstrings independently note the *same* justification
  ("every row is an i.i.d. draw, so a contiguous block is equivalent to a
  random split") without sharing any code -- exactly the "two or three
  examples independently implement splitting" trigger condition the brief
  names for building a reusable primitive (Section 4.A).
- Checked `segmentation/dataset.py`'s `make_datasets`: genuinely different
  shape (draws train/test from two independent RNG streams rather than one
  combined array to slice), correctly *not* using this pattern -- confirms
  the gap is real and scoped, not "every dataset should be forced through
  one abstraction."
- Checked `DataLoader` construction across all 13 examples/demos
  (`grep -n "DataLoader("`): batch_size/shuffle/generator arguments are
  already terse one-liners with no repeated configuration boilerplate
  worth abstracting -- confirms Section 4.C's "DataLoader construction"
  candidate has no evidence behind it, matching M73's own explicit
  rejection of a `forge.train()`/config-object API.

## 3. The problem

Forge already has `forge.data.random_split` (permutation-then-slice) for a
dataset whose sample order carries meaning. It had no equivalent for a
dataset whose order is already meaningless (i.i.d.-generated rows) and
therefore needs no shuffling -- so the two examples with this exact
property each wrote their own raw-`numpy`-array-slicing partition logic
instead of using a `Dataset`/`Subset`-based primitive, the same way
`image_folder_classification` already uses `random_split`. This is a small
but genuine piece of duplicated, un-tested-as-shared, easy-to-get-wrong
(off-by-one boundaries) framework plumbing a developer had to write twice.

## 4. Rejected alternatives

- **A generic `DatasetManager`/`DataModule`/`DataConfig`.** No evidence any
  example needs configuration beyond two positional arguments
  (`dataset`, `lengths`); the brief explicitly forbids this shape.
- **Extending `random_split` itself with a `shuffle=` flag.** Considered,
  but `random_split`'s whole contract is built around `generator=`
  (reproducible permutation); bolting a `shuffle=False` mode onto it would
  make `generator=` meaningless in that mode and complicate one function's
  docstring/tests for what is a genuinely different, simpler operation
  (no RNG at all). A second top-level function with a name that says
  exactly what it does (`sequential_split`, mirroring `random_split`'s own
  naming) is clearer and matches this codebase's existing precedent of
  small, single-purpose functions over parameterized modes (e.g.
  `Resize`/`Normalize` are separate transforms, not one `Transform(mode=)`).
- **A `forge.data.random_split(..., shuffle=False)`-free rewrite that has
  `TensorDataset` itself accept `n_train/n_val/n_test`.** Rejected --
  would conflate a general-purpose array-backed dataset with a
  train/val/test-partitioning policy that not every `TensorDataset`
  consumer wants (e.g. `data_pipeline_demo.py`, `persistence_demo.py`,
  `long_range_recall`, `word_rnn`, `char_rnn` all use `TensorDataset`
  without any such split).
- **Forcing `segmentation/dataset.py` onto the same primitive.** Rejected
  per the brief's own instruction to preserve real differences rather than
  force them into one abstraction -- `segmentation` has no single combined
  array to split in the first place (independent train/test generator
  streams), so there is nothing for `sequential_split` to do there.

## 5. Design

`forge.data.sequential_split(dataset, lengths) -> list[Subset]`
(`forge/data/dataset.py`), exported from `forge.data`:

```python
def sequential_split(dataset: Dataset, lengths: Iterable[int]) -> list[Subset]:
    lengths = _split_lengths(dataset, lengths, "sequential_split")
    subsets = []
    offset = 0
    for n in lengths:
        subsets.append(Subset(dataset, range(offset, offset + n)))
        offset += n
    return subsets
```

Deliberately the smallest possible change: `random_split`'s own
length-validation logic was factored into a shared `_split_lengths()`
helper (used by both functions, avoiding a second copy of the "must be
non-negative and sum to `len(dataset)`" check) and `sequential_split` reuses
the exact same `Subset` return type, so a caller can use whichever split
function fits its dataset's semantics without learning two different result
shapes.

## 6. Reproducibility

`sequential_split` takes no RNG and needs none -- its result is a pure
function of `lengths` alone, which is *more* deterministic than
`random_split` (no `generator=`/seed to manage or restore across resume).
This does not interact with M65/M73's DataLoader-shuffle-resume
reproducibility guarantees at all: `sequential_split` only decides which
samples belong to which split, not the order `DataLoader` later iterates
them in.

## 7. Compatibility

Purely additive. `forge.data.Dataset`, `TensorDataset`, `Subset`,
`random_split`, `DataLoader`, `ImageFolder`, transforms, `Trainer`,
`TrainingSession`, `forge.predict()`, checkpointing, and model/preprocessing/
class persistence are all unchanged. `examples/regression/dataset.py::
make_datasets` and `examples/waveform_classification/dataset.py::
make_datasets` keep their exact existing signature and return shape
(`(train_ds, val_ds, test_ds, stats)`) -- `examples/regression/train.py`
and `examples/waveform_classification/train.py` needed **zero** changes.

## 8. Implementation

- `forge/data/dataset.py`: added `_split_lengths()` (shared validation,
  refactored out of `random_split`'s own body) and `sequential_split()`.
- `forge/data/__init__.py`: exported `sequential_split`.
- `examples/regression/dataset.py::make_datasets`: now builds one
  `TensorDataset(Tensor(X), Tensor(y), transform=Normalize(...))` over the
  full generated array and calls `sequential_split(full_ds, [n_train,
  n_val, n_test])`, instead of hand-slicing `X`/`y` into three pairs and
  constructing three separate `TensorDataset`s. The one array slice that
  remains (`X[:n_train]`, to compute train-only `mean`/`std` for
  `Normalize`) is a legitimate modeling concern -- fitting a transform on
  training data only -- that `sequential_split` itself has no reason to
  know about, not leftover partitioning plumbing.
- `examples/waveform_classification/dataset.py::make_datasets`: same
  refactor, simpler (no transform to fit), so `sequential_split` now
  replaces the manual slicing entirely.
- Docstrings in both files and in `docs/architecture/data-pipeline.md`/
  `docs/architecture/data-system.md` updated to describe the shared
  primitive instead of independent hand-written slicing.

## 9. Tests

`tests/test_dataset.py` (mirrors the existing `random_split` test block
exactly): `test_sequential_split_sizes_and_contiguous_blocks_in_order`,
`test_sequential_split_is_deterministic_with_no_generator_needed`,
`test_sequential_split_three_way_matches_manual_array_slicing` (asserts the
new primitive reproduces byte-for-byte the exact three-way slice
`regression`/`waveform_classification` used to compute by hand),
`test_sequential_split_rejects_mismatched_lengths`,
`test_sequential_split_rejects_negative_length`,
`test_sequential_split_zero_length_split_is_allowed`. 6 new tests, all CPU
(no CUDA involvement -- `sequential_split` only manipulates Python
indices/`Subset`, exactly like `random_split`, which also has no dedicated
CUDA test file).

No new tests were added purely to `examples/regression`/
`waveform_classification`'s own suites -- their **existing** integration
tests (`tests/test_regression_example_integration.py`,
`tests/test_waveform_classification_example_integration.py`, including
`test_make_datasets_splits_are_disjoint_and_correctly_sized` and
`test_make_datasets_normalizes_training_features_to_near_zero_mean_unit_std`)
passed **unmodified** against the refactor, proving behavioral equivalence
the same way M73's retrofit of `examples/regression/train.py` was
validated against its own pre-existing `test_regression_reproducible_
training.py` suite.

## 10. Real workload results

- `tests/test_dataset.py`: 36 passed (30 pre-existing + 6 new).
- `tests/test_regression_example_integration.py` +
  `tests/test_waveform_classification_example_integration.py`: 22 passed,
  unmodified, confirming `make_datasets()`'s observable behavior
  (split sizes, disjointness, normalization statistics, downstream
  training) is unchanged by the internal refactor.
- CUDA-hardware-gated variants (`test_regression_example_cuda_
  integration.py`, `test_waveform_classification_example_cuda_
  integration.py`) plus `image_folder_classification`'s own CPU/CUDA
  integration suites, re-run together on the 940MX: 21 passed (176.6s,
  includes real training runs).
- Full suite: **2,221 collected, 2,220 passed, 1 failed** (2,215 + 6 new).
  The one failure is the same pre-existing `test_dataloader_prefetch.py::
  test_repeated_epochs_do_not_grow_cuda_or_pinned_memory` allocator-
  measurement flake documented since M63 -- reproduced passing cleanly in
  isolation (`1 passed in 0.96s`), no M74 regression.

## 11. Limitations

- `sequential_split` only handles the "one combined array/dataset, slice
  into contiguous blocks" shape. It does not help `segmentation` (two
  independent generator streams) or `mnist`/`resnet`/`autoencoder`
  (separate train/test files on disk) -- those examples have no duplicated
  logic for `sequential_split` to remove, and forcing them onto it would
  contradict the brief's "preserve real differences" instruction.
- No stratified/class-balanced split variant was added -- no current
  example's split target is class-imbalanced enough to need one, and no
  consumer asked for it.
- `image_folder_classification` still uses `random_split`, not
  `sequential_split` -- correctly, since its sample order (sorted by class
  then filename) is exactly the case `sequential_split`'s own docstring
  warns against.

## 12. Future triggers

- If a third example needs an i.i.d.-array train/val/test partition, it
  should use `sequential_split` directly rather than re-deriving the
  slicing arithmetic a third time -- the primitive is already general.
- If a future dataset needs a stratified (class-balanced) split, that is a
  concrete, evidence-backed trigger for a `stratified_split`-shaped
  addition -- not before then.
- No CLI or high-level `forge.train()`-style wrapper is justified by this
  milestone; M73's own rejection of that API stands unchanged.
