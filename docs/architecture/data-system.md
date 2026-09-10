# Dataset, DataLoader, and Transforms (Milestone 5)

## Package layout
```
forge/
    data/
        dataset.py       Dataset, TensorDataset, Subset, random_split
        dataloader.py    DataLoader, batch collation
        transforms.py    Transform, Compose, ToTensor, Normalize, Reshape, Flatten, Resize, Lambda
        image_folder.py  ImageFolder, IMAGE_EXTENSIONS (Milestone 69)
```
`forge.data` is exposed as a submodule of `forge` (`forge.data.TensorDataset`,
`forge.data.DataLoader`, ...), alongside `forge.nn`/`forge.optim`/`forge.random`.

## Target flow
```text
Raw data -> Dataset -> Transforms -> DataLoader -> Batches -> Model / Trainer
```
`forge.data` has no dependency on `Module`, `Loss`, `Optimizer`, or a
training engine (there is no `Trainer` yet -- see `docs/development/roadmap.md`).
A `Dataset` only needs to know how many samples it has and how to produce
one; a `DataLoader` only needs a `Dataset` and produces `Tensor` batches. The
model-facing boundary is: whatever a `DataLoader` yields must be directly
callable against an existing `forge.nn.Module`.

## Dataset
`Dataset` (`forge/data/dataset.py`) is a minimal two-method protocol:
```python
class Dataset:
    def __len__(self) -> int: ...
    def __getitem__(self, index) -> Any: ...
```
The base class's methods raise `DataError`, the same "must implement"
pattern already used by `Module.forward`/`Loss.forward`/`Optimizer.step`. A
sample may be a single `Tensor`, a tuple such as `(features, target)`, or
another representation a custom `Dataset` documents for itself -- Forge does
not force one hard-coded schema.

### TensorDataset
An in-memory dataset over one or more aligned Tensors:
```python
dataset = TensorDataset(features, targets)
x, y = dataset[0]
```
Every tensor must share the same size along its first (sample) axis --
`DataError` at construction otherwise, along with a scalar (0-D) tensor or
zero samples. A dataset built from a single tensor returns that tensor's
sample directly from `__getitem__` rather than wrapping it in a 1-tuple; two
or more tensors return a tuple of per-sample components, one per input
tensor, in order. Each returned sample is a freshly built `Tensor` sliced
from the source tensor's underlying NumPy storage, preserving that source
tensor's `dtype` and `device`.

`TensorDataset(..., transform=..., target_transform=...)` applies `transform`
only to the first tensor's sample (the conventional "features" position) and
`target_transform` only to the second tensor's sample (the conventional
"target" position) -- see Transforms below for why these are kept separate.

### Subset / random_split / sequential_split
`Subset(dataset, indices)` is a read-only view over a subset of another
dataset's indices, in the given order; `dataset[Subset's index]` maps back
through `indices` first. `random_split(dataset, lengths, generator=None)`
permutes `range(len(dataset))` once (via `generator`, defaulting to
`forge.random.default_generator()`) and slices it into consecutive
`Subset`s matching `lengths`, which must sum to `len(dataset)`. Slicing a
single permutation (rather than sampling each subset independently)
guarantees the subsets are disjoint and jointly cover every original index
exactly once, preserving feature/target correspondence.

`sequential_split(dataset, lengths)` (Milestone 74) is `random_split` without
the permutation: it slices `range(len(dataset))` directly into consecutive
`Subset`s, so it needs no `generator` and is deterministic by construction.
Prefer it over hand-slicing a dataset's backing arrays for an
already-order-independent dataset (e.g. i.i.d.-generated rows); prefer
`random_split` when sample order carries meaning (e.g. `ImageFolder`, sorted
by class then filename) and every split needs a representative mix. See
`docs/development/m74-data-workflow.md`.

### ImageFolder (Milestone 69)
`forge.data.ImageFolder(root, transform=None, target_transform=None,
extensions=IMAGE_EXTENSIONS)` discovers `(image, label)` samples from a
directory-per-class image tree:
```python
dataset = ImageFolder("data/cats-dogs", transform=preprocess)
image, label = dataset[0]        # image: Tensor(3, H, W) float32, [0, 255]
dataset.classes                  # sorted class names, deterministic indices
dataset.class_to_idx
dataset.samples                  # [(Path, class_idx), ...], deterministic order
```
Each immediate subdirectory of `root` is one class (sorted by name for
deterministic indices); only files directly inside a class directory are
scanned (no recursion into further subdirectories) and only files whose
suffix matches `extensions` (case-insensitive) count as samples. `root`
must exist and contain at least one class with at least one matching image
(`DataError` otherwise); an individual empty class directory is valid and
just contributes zero samples. Every image is decoded via Pillow (the
`Pillow` dependency `docs/architecture/data-pipeline.md`'s "Constraints"
section anticipated) and converted to `(3, H, W)` float32, raw `[0, 255]`
range, always RGB (grayscale replicated across channels, RGBA alpha
discarded) -- decoded fresh on every `__getitem__`, no caching/preloading.
`ImageFolder` does not resize -- every file under `root` must already share
one `(H, W)`, or `transform=` must normalize that; see `Resize` below
(Milestone 70), which exists specifically to fill this gap.
Full contract: `forge/data/image_folder.py`'s module docstring.

## DataLoader
`DataLoader` (`forge/data/dataloader.py`) iterates a `Dataset` in batches:
```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_x, batch_y in loader:
    ...
```
- `batch_size` (default `1`): must be a positive `int` (`bool` explicitly
  rejected, matching `SGD`'s `lr` validation) -- `DataError` otherwise.
- `shuffle` (default `False`): if `True`, sample indices are permuted at the
  start of each `for batch in loader` iteration, in place, via `generator`
  if supplied or `forge.random.default_generator()` otherwise -- the same
  process-global generator `Linear` draws from for parameter
  initialization, so `forge.random.seed(...)` makes a script's shuffling
  reproducible without any DataLoader-specific seeding. Passing an explicit
  `numpy.random.Generator` makes one loader's ordering reproducible
  independently of global state: two loaders constructed with fresh,
  equally-seeded generators produce identical batch sequences.
- `drop_last` (default `False`): whether a final batch smaller than
  `batch_size` is yielded or dropped. For `N=10, batch_size=4`:
  `drop_last=False` yields sizes `4, 4, 2`; `drop_last=True` yields `4, 4`.
- `len(loader)` returns the resulting batch count (`ceil(N/B)` or `N//B`
  depending on `drop_last`), matching what iteration actually produces.

Construction validates that `dataset` supports `len()` (`DataError`
otherwise); it does not eagerly validate `__getitem__` or dataset contents,
since a `Dataset` may be arbitrarily custom.

### Batching / collation
A batch is assembled by indexing the dataset once per selected index and
stacking the results (`forge/data/dataloader.py`'s `_collate`/`_stack`):
- A `Tensor`-valued sample batches to a `Tensor` of shape
  `(batch_size, *sample.shape)`.
- A tuple-valued sample (e.g. `(features, target)`) batches to a tuple of
  such Tensors, one per component, at matching batch indices -- this is how
  `batch_x[i]`/`batch_y[i]` stay correlated.
- Every sample in a batch must share the tuple structure, dtype, and shape
  of the first sample; a mismatch raises `DataError` rather than a confusing
  NumPy stacking failure.

Collation goes through each `Tensor`'s `.numpy()` and `np.stack`, then
rewraps the result as a single `Tensor` with the source samples' shared
`dtype`/`device` -- no NumPy array is exposed as the public batch type, only
Forge Tensors. No multiprocessing workers or asynchronous prefetching:
iteration is plain synchronous Python, per this milestone's scope.

## Transforms
`Transform` (`forge/data/transforms.py`) is a minimal callable base class
(`__call__` raises `DataError` if unimplemented, matching `Dataset`/`Module`/
`Loss`). `Compose(transforms)` threads a sample through a sequence of
transforms in order, feeding each one's output to the next.

Transforms operate on **one sample component** (typically a `Tensor`), not
on a whole `(features, target)` tuple -- a transform written for features
cannot silently reach into a label. `TensorDataset` enforces this boundary
structurally: `transform` is wired to the first tensor's sample and
`target_transform` to the second's, configured separately rather than one
callable receiving the full tuple and being trusted to leave the target
alone.

### Built-in transforms
- `ToTensor(dtype=None, device="cpu")`: wraps array-like data as a `Tensor`.
- `Normalize(mean, std)`: `(x - mean) / std`, elementwise, broadcasting
  `mean`/`std` against the sample the same way Tensor arithmetic broadcasts
  elsewhere in Forge. Implemented as `(x - mean) * (1/std)` since `Tensor`
  has no division operator; rejects a zero `std` at construction.
- `Reshape(*shape)` / `Flatten()`: thin wrappers over `Tensor.reshape`.
- `Resize(size)` (Milestone 70): resizes a `(C, H, W)` image `Tensor`
  (`C` in `{1, 3}`) to an explicit `(height, width)` via Pillow's bilinear
  filter, preserving dtype/device. Exists to make mixed-resolution
  `ImageFolder` datasets batchable by `DataLoader` -- `_stack` requires
  every sample in a batch to share one shape, and ordinary image
  collections rarely arrive pre-normalized to one resolution. Runs on raw
  `[0, 255]`-range pixel data (apply it before a rescaling transform like
  `Lambda(lambda x: x / 255)`, not after); executes on the CPU via Pillow,
  outside the autograd graph -- it is a data-pipeline preprocessing step,
  not a differentiable Tensor op, so it introduces no CUDA kernel and no
  `Tensor.resize()` primitive. See
  `docs/development/m70-image-preprocessing.md`.
- `Lambda(fn)`: wraps an arbitrary callable as a `Transform`.

Deliberately not a computer-vision transform library -- `Resize` is the one
image-shape-normalizing exception, added because `DataLoader`'s batching
contract made it a genuine (not speculative) requirement; no augmentation
transforms (crop/flip/color-jitter) exist.

## Errors
All dataset/loader/transform failures raise `forge.exceptions.DataError`
(new in this milestone), covering: an unimplemented `Dataset`/`Transform`
base method, mismatched `TensorDataset` tensor sample counts, a scalar or
empty `TensorDataset` tensor, an out-of-range or non-int dataset index, a
`target_transform` without a target tensor, an invalid `DataLoader`
`batch_size`/`shuffle`/`drop_last`, a dataset without `len()`, inconsistent
sample structure/shape/dtype within a batch, `random_split` lengths that
don't sum to the dataset size or that are negative, a zero `Normalize` std,
and a non-callable `Compose`/`Lambda` member. As of Milestone 69,
`ImageFolder` raises `DataError` for a missing/non-directory root, a root
with no class subdirectories, a root with zero total discovered images, an
out-of-range/non-int index, and an unreadable/corrupt image file. As of
Milestone 70, `Resize` raises `DataError` for a malformed/non-2-tuple
`size`, a non-positive or non-integer dimension, a non-`Tensor` sample, a
sample that isn't a 3-D `(C, H, W)` Tensor, and an unsupported channel
count (anything other than 1 or 3).

## Device behavior
`forge.data` (`Dataset`/`DataLoader`/transforms) remains CPU-only,
unchanged since Milestone 5 -- this was a deliberate, permanent boundary,
not a placeholder later milestones were expected to lift. As of Milestone
12, `forge.training.Trainer` supports CUDA training (see
`docs/architecture/training-engine.md`), and it is the layer that turned out
to own device movement, exactly as anticipated here: `Trainer` explicitly
calls `x.to(device)`/`y.to(device)` on each batch a CPU `DataLoader` yields,
immediately before the forward pass. Nothing in `forge.data` itself became
device-aware to make that possible -- `DataLoader` still only ever produces
plain CPU Tensors, with no GPU-batching, pinned-memory, or async-prefetch
behavior (explicit Milestone 12 non-goals), regardless of what device the
`Trainer` consuming it is configured for.

## Known limitations
- No multiprocessing workers or asynchronous prefetching.
- No tabular-convenience dataset yet (see `docs/architecture/
  data-pipeline.md` for what's deferred); `ImageFolder` (M69) covers the
  directory-based image case.
- `ImageFolder` does not resize/augment -- every file under one root must
  already share the same `(H, W)`; no `Resize` transform exists yet.
- No custom `collate_fn`; batching always stacks same-shape/dtype Tensor
  components.
- `Normalize`/`Reshape`/`Flatten` operate on a single Tensor component only,
  not on nested/dict-shaped samples.
- As of Milestone 6, `forge.training.Trainer` consumes `DataLoader` output
  for training/evaluation -- see `docs/architecture/training-engine.md`. As
  of Milestone 12, that includes CUDA training: `Trainer` explicitly moves
  each CPU batch to its configured device, but `DataLoader` itself gained no
  new capability and no device awareness.
