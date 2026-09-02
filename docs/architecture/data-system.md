# Dataset, DataLoader, and Transforms (Milestone 5)

## Package layout
```
forge/
    data/
        dataset.py      Dataset, TensorDataset, Subset, random_split
        dataloader.py   DataLoader, batch collation
        transforms.py   Transform, Compose, ToTensor, Normalize, Reshape, Flatten, Lambda
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

### Subset / random_split
`Subset(dataset, indices)` is a read-only view over a subset of another
dataset's indices, in the given order; `dataset[Subset's index]` maps back
through `indices` first. `random_split(dataset, lengths, generator=None)`
permutes `range(len(dataset))` once (via `generator`, defaulting to
`forge.random.default_generator()`) and slices it into consecutive
`Subset`s matching `lengths`, which must sum to `len(dataset)`. Slicing a
single permutation (rather than sampling each subset independently)
guarantees the subsets are disjoint and jointly cover every original index
exactly once, preserving feature/target correspondence.

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
- `Lambda(fn)`: wraps an arbitrary callable as a `Transform`.

Deliberately not a computer-vision transform library -- just enough to
establish composability and cover common numeric preprocessing.

## Errors
All dataset/loader/transform failures raise `forge.exceptions.DataError`
(new in this milestone), covering: an unimplemented `Dataset`/`Transform`
base method, mismatched `TensorDataset` tensor sample counts, a scalar or
empty `TensorDataset` tensor, an out-of-range or non-int dataset index, a
`target_transform` without a target tensor, an invalid `DataLoader`
`batch_size`/`shuffle`/`drop_last`, a dataset without `len()`, inconsistent
sample structure/shape/dtype within a batch, `random_split` lengths that
don't sum to the dataset size or that are negative, a zero `Normalize` std,
and a non-callable `Compose`/`Lambda` member.

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
- No file-backed or image/tabular-convenience datasets yet -- only the
  in-memory `TensorDataset` (see `docs/architecture/data-pipeline.md` for
  what's deferred).
- No custom `collate_fn`; batching always stacks same-shape/dtype Tensor
  components.
- `Normalize`/`Reshape`/`Flatten` operate on a single Tensor component only,
  not on nested/dict-shaped samples.
- As of Milestone 6, `forge.training.Trainer` consumes `DataLoader` output
  for training/evaluation -- see `docs/architecture/training-engine.md`. As
  of Milestone 12, that includes CUDA training: `Trainer` explicitly moves
  each CPU batch to its configured device, but `DataLoader` itself gained no
  new capability and no device awareness.
