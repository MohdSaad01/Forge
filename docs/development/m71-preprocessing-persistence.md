# M71 — Persisted Preprocessing: Closing the "Hidden Training-Time Assumption" Gap

## 1. Objective

M70 closed the mixed-resolution `ImageFolder` gap with `Resize`, but its own
report (`docs/development/m70-image-preprocessing.md`, Sections 16/17)
documented the next problem directly: a saved model carries no record of
the preprocessing its inputs must already have gone through. A developer
loading a model in a fresh process has to *remember* (or read the original
training script) that inputs need `Resize((64, 64))` then a specific pixel
scale -- nothing in the `.forge` file itself says so, and nothing checks it.
M71's brief asked for the smallest concrete engineering change that closes
this gap for the currently-supported transforms, without inventing a
general-purpose artifact/serving system.

## 2. Investigation

Read/verified directly (not assumed from prior reports):

- `forge/data/transforms.py`: `Transform`/`Compose`/`ToTensor`/`Normalize`/
  `Reshape`/`Flatten`/`Resize`/`Lambda`. Every transform except `Lambda` is
  already fully described by a handful of plain constructor arguments
  (ints, floats, small lists) -- no transform holds any large/opaque state.
  `Lambda` wraps an arbitrary Python callable with no safe representation.
- `forge/data/image_folder.py`: `ImageFolder._load_image` is already
  documented as the reusable single-image decode step "if a caller ever
  needs to preprocess one arbitrary image file the exact same way
  `ImageFolder` does" -- exactly `infer.py`'s situation.
- `forge/serialization/registry.py`/`model.py`: the existing `Module`
  persistence registry (`register_module`/`spec_for_class`/`spec_for_name`,
  `get_config`/`from_config`) is precisely the "explicit config, no
  arbitrary code" pattern the brief asked for -- for modules, not
  transforms. `Sequential`'s registration was the one precedent for a type
  whose reconstruction needs more than a flat `cls(**config)` call.
- `forge/serialization/archive.py`: `write_archive`/`read_archive` already
  generalize to arbitrary caller-chosen metadata keys and array paths
  (used by `checkpoint.py` to add `"optimizer"`/`"training_progress"`/
  `"rng"` alongside `save_model()`'s own keys) -- adding one more top-level
  metadata key needs no format change to this layer at all.
- `forge/training/inference.py` (`predict()`, M68): unaffected either way --
  it operates on already-prepared Tensors/batches, and has no opinion about
  preprocessing.
- `examples/image_folder_classification/train.py` (M69/M70): the concrete
  consumer. Its `build_transform()` is `Compose([Resize((64, 64)),
  Lambda(lambda x: x * (1/255))])` -- the `Lambda` step is the only
  obstacle to full serializability, and it computes an operation (`x /
  255`) `Normalize` already expresses exactly (`Normalize(mean=0.0,
  std=255.0)` == `(x - 0) / 255`).
- `docs/architecture/persistence.md`: confirmed the exact versioning/
  compatibility policy (`FORMAT_VERSION`, bumped only for *required* new
  keys, per M53's buffer-support precedent) and the "explicit registry,
  never arbitrary code" security/trust model this had to match.

**Established directly, not assumed:**
- `ImageFolder` itself does not need to be persisted -- only the transform
  its samples were passed through does. (Confirmed by re-reading the
  brief's own explicit rejection of "coupling Dataset objects to trained
  models" against what `ImageFolder`/`Resize`/`predict()` actually need
  from each other -- nothing beyond the transform.)
- Transforms can already be represented safely as configuration -- for
  every registered type, `get_config`/`from_config` are a few lines,
  mirroring `Module`'s own registry exactly.
- Transform serialization belongs in `forge.serialization` (a new
  `forge/serialization/transforms.py`), not `forge.data` -- `forge.data`
  stays free of any persistence-format concern, matching the "preserve
  clear boundaries...between...data...and...serialization" principle in
  `CLAUDE.md`, and mirroring how `forge.serialization.registry` (not
  `forge.nn`) already owns `Module` persistence.
- A simpler example-level-only solution (e.g. `infer.py` re-implementing
  `build_transform()` by hand) was rejected: it would not close the actual
  gap -- a fresh process still has to *know* the exact transform sequence
  from source, which is exactly the hidden-assumption failure this
  milestone exists to remove.
- Enough infrastructure already existed (the registry pattern, the generic
  archive format, `ImageFolder._load_image` as a reusable single-image
  decode step) that no new Tensor/autograd/CUDA/DataLoader capability was
  needed anywhere.

## 3. Chosen Implementation

**A transform-configuration registry (`forge/serialization/transforms.py`),
mirroring `forge.serialization.registry` exactly**, plus one new optional
keyword argument on `save_model()` and one new free function:

```python
forge.save_model(model, path, preprocessing=Compose([Resize((64, 64)), Normalize(0.0, 255.0)]))
transform = forge.load_preprocessing(path)  # -> the reconstructed Transform, or None
```

- **Registered transforms**: `Resize`, `Compose` (recursively, over its own
  child transforms -- the one type needing more than a flat `cls(**config)`
  call, exactly like `Sequential`'s own precedent), `Normalize`, `ToTensor`,
  `Reshape`, `Flatten`.
- **Deliberately not registered**: `Lambda`. Attempting to serialize one
  (directly or nested inside a `Compose`) raises `PersistenceError`
  immediately -- the exact "not registered, here's why, here's what to do
  instead" failure `spec_for_class()` already gives for an unregistered
  `Module`.
- **Storage**: `"preprocessing"` is a new, optional, top-level sibling key
  in the *same* `save_model()` metadata dict as `"root"`/`"device"` --
  `null` when omitted. `preprocessing` is **not** attached to `Module` in
  any way (no new `Module` attribute, no change to the module tree walk) --
  a model's parameters and its input-preprocessing configuration remain two
  separate things stored side by side in one file, per the brief's explicit
  architectural constraint.
- **No format-version bump.** `FORMAT_VERSION` stays `2`. Forward-compatible
  (an old build's `load_model()` never reads the new key) and backward-
  compatible (`metadata.get("preprocessing")` treats a missing key exactly
  like an explicit `null`) -- verified directly with a hand-constructed
  archive that has the key deleted entirely, not just set to `null`
  (`tests/test_preprocessing_persistence.py::
  test_backward_compatible_file_has_no_preprocessing_key`).
- **One small, necessary fix to `Normalize` itself**: it previously stored
  only `1/std` (`self._inv_std`), never the original `std` -- insufficient
  to serialize its own constructor arguments. Added `self.std = std`
  (mirroring the already-stored `self.mean`); no behavior change to
  `Normalize.__call__`.

## 4. Required Consumer: `examples/image_folder_classification/`

- `train.py`'s `build_transform()` now returns `Compose([Resize((64, 64)),
  Normalize(mean=0.0, std=255.0)])` -- the `Lambda` pixel-scale step is
  gone, replaced by the mathematically identical, serializable `Normalize`
  call. This was the *only* change needed to make the example's existing
  pipeline fully persistable.
- `save_model(model, str(model_path), preprocessing=build_transform())`
  replaces the old `save_model(model, str(model_path))` call. A
  `classes.json` sidecar (a plain list of class names) is written alongside
  it -- explicitly *not* part of the preprocessing mechanism (a label
  vocabulary is not "how to prepare an input tensor"), ordinary example-
  level bookkeeping, the same category as the `new_mixed_resolution_query
  .png` file the script already wrote before this milestone.
- The script's own post-save inference demo now calls
  `forge.load_preprocessing(model_path)` and applies the *reconstructed*
  transform to the new out-of-distribution image, rather than reusing its
  own in-process `build_transform()` result -- proving the persisted
  configuration, not just the example script's internal consistency.
- **New file, `infer.py`**: a genuinely separate, standalone script.
  ```bash
  python -m examples.image_folder_classification.infer \
      --model .../image_folder_model.forge --classes .../classes.json --image path/to/new.jpg
  ```
  Imports only `forge.load_model`/`forge.load_preprocessing`/
  `forge.predict`/`ImageFolder._load_image` -- never `train.py`'s
  `build_transform()`, `build_model()`, or any in-memory object from a
  training run. Run and verified as an actual second process against an
  artifact produced by a first `train.py` run (Section 6 below). Raises a
  clear `PersistenceError` (not a silent skip) if pointed at a model saved
  without `preprocessing=`.

## 5. Architecture Impact

- **New file**: `forge/serialization/transforms.py` (registry +
  `serialize_transform`/`deserialize_transform`).
- **Changed**: `forge/serialization/model.py` (`save_model(..., 
  preprocessing=None)`, new `load_preprocessing()`), `forge/serialization/
  __init__.py` and `forge/__init__.py` (new exports), `forge/data/
  transforms.py` (`Normalize` retains raw `std`).
- **Zero changes** to `Tensor`, autograd, any CUDA kernel, `Module`,
  `DataLoader`, `Trainer`, `forge.predict()`, or the checkpoint format.
  `forge.data`/`ImageFolder` themselves are untouched -- this milestone's
  entire surface is in `forge.serialization` plus example wiring, matching
  the brief's explicit "preserve clear boundaries" instruction.

## 6. API Changes

```python
# New, optional -- every existing call site is unaffected.
forge.save_model(model, path, preprocessing=None)

# New free function.
forge.load_preprocessing(path) -> Transform | None

# New, mirrors register_module().
forge.serialization.register_transform(type_name, cls, get_config, from_config=None)
```
`load_model()`'s signature and return type are completely unchanged.

## 7. Persistence / Compatibility

- No `FORMAT_VERSION` bump (see Section 3) -- documented explicitly in
  `docs/architecture/persistence.md`'s new **Preprocessing metadata**
  section, with the forward/backward-compatibility argument spelled out and
  a direct test proving both directions.
- `save_checkpoint()`/`load_checkpoint()` do **not** gain `preprocessing=`
  -- out of scope (a resumed training run reconstructs its own pipeline
  from its own script, as before); documented as a known limitation.

## 8. Tests

28 new tests, all passing:
- `tests/test_preprocessing_persistence.py` (26, CPU): `serialize_transform`/
  `deserialize_transform` round trips for every registered type (including
  nested `Compose`), `Lambda`/unknown-type/malformed-node rejection (all
  `PersistenceError`), `register_transform` conflict/no-op-safe re-
  registration, `save_model(..., preprocessing=...)`/`load_preprocessing()`
  round trips, the `preprocessing=None`-vs-omitted equivalence, rejection
  of a `Lambda`-containing pipeline *before* any file is written, the
  backward-compatibility test described above, and two real end-to-end
  tests: `ImageFolder` (real PNG files on disk, via `PIL.Image.save`) ->
  train one epoch -> `save_model(..., preprocessing=...)` -> delete every
  in-memory reference -> reload both model and preprocessing from the file
  alone -> preprocess a brand-new, different-resolution image -> `forge.
  predict()`; and a direct proof that training-time and reloaded
  inference-time preprocessing produce bit-identical output on the same
  raw image.
- `tests/test_preprocessing_persistence_cuda_integration.py` (2, CUDA-gated,
  skips cleanly without CUDA): `load_preprocessing()` (always host-only)
  composed with a model reloaded onto `device="cuda"`, and CPU-loaded vs.
  CUDA-loaded predictions agreeing given identical reconstructed
  preprocessing.

## 9. Full-Suite Result

Before this milestone: 2,139 tests collected (per M70's own report). After:
**2,167 tests collected** (+28, exactly the new test count). A full
`python -m pytest tests/` run: **2,167 collected, 2,166 passed, 1 failed**
-- the one failure is `tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, the same
pre-existing CUDA allocator-measurement flake documented since M63
(re-run in isolation immediately afterward: passes cleanly, `1 passed`) --
not an M71 regression. Both pre-existing image-folder-classification
integration suites (`tests/test_image_folder_classification_integration.py`,
`tests/test_image_folder_classification_cuda_integration.py`) were re-run
and pass unmodified (they define their own local `_build_transform()`
using `Lambda`, independent of `train.py`'s -- left as-is since they test
`ImageFolder`/`Resize`/`DataLoader`/`Trainer` behavior, not preprocessing
persistence, and nothing about this milestone required changing them).

One real test bug found and fixed during verification, not swept under the
rug: the CUDA integration test's second test originally resized to
`(12, 12)` while its `_TinyCUDACNN` model's `Linear` layer was fixed at
`4 * 8 * 8 = 256` input features (sized for `(16, 16)` after one
`MaxPool2d(2)`) -- a genuine shape mismatch (`ShapeMismatchError:
Linear(in_features=256) cannot accept input of shape (1, 144)`), not a
framework bug. Fixed by using `(16, 16)` consistently in both CUDA tests.

## 10. End-to-End Workflow Result

Ran the real example at small scale (`--samples-per-class 15 --epochs 2`):
`train.py` trained, saved `image_folder_model.forge` + `classes.json`,
verified its own in-process reload, and printed the exact `infer.py`
command to run next. That command was run as a **genuinely separate Python
process** against the just-written artifact and produced the same
prediction (`triangle`) as the in-process demo -- confirming the file alone
carries everything needed. Separately, a model saved *without*
`preprocessing=` was pointed at `infer.py`, producing the intended clear
`PersistenceError` rather than a wrong/silent prediction.

## 11. CPU/CUDA Verification

CPU: full new test file, full existing `image_folder_classification`
integration suite, and the small-scale real `train.py`/`infer.py` run
above (all CPU). CUDA: `tests/
test_preprocessing_persistence_cuda_integration.py` (hardware-verified on
the reference 940MX) plus the pre-existing CUDA image-folder-classification
integration suite (unmodified, still passing) -- confirming a CUDA-resident
model composes correctly with the (always host-only) reconstructed
preprocessing.

## 12. Limitations

- Only the six registered transform types round-trip; a custom `Transform`
  subclass (including `Lambda`) needs its own `register_transform()` call
  before it can be part of `preprocessing=`, the same opt-in
  `register_module()` already requires for a custom `Module`.
- `save_checkpoint()`/`load_checkpoint()` do not carry `preprocessing=`
  (Section 7).
- Class-index-to-name mapping (`classes.json`) is example-level, not a
  framework feature -- deliberately, per the brief's rejection of coupling
  a `Dataset` to a trained model.
- No preprocessing *validation against the model's actual expected input
  shape* (e.g. Forge does not check that a `Resize((64,64))` in
  `preprocessing` actually matches a model's first-layer expected spatial
  size) -- a mismatch still surfaces as an ordinary shape-mismatch error
  from the model's own forward pass, not a dedicated diagnostic. No real
  workload has needed this yet.

## 13. Rejected Alternatives

- **Pickling `Transform` objects directly**: rejected outright per the
  brief -- arbitrary executable state, defeats the entire trust model
  `forge.serialization` already established for `Module`s.
- **A general-purpose data-pipeline execution engine / artifact-management
  system**: no consumer needs more than "reconstruct one `Transform`
  pipeline from a model file" -- building more would be exactly the kind of
  speculative machinery `CLAUDE.md` and this milestone's brief both warn
  against.
- **Attaching `preprocessing` to `Module` as an attribute**: rejected per
  the brief's own explicit architectural constraint -- parameters and
  preprocessing are conceptually distinct; storing it as a metadata-only
  sibling key achieves the "travels with the file" goal without blurring
  that boundary.
- **Persisting `ImageFolder` itself**: rejected -- a `Dataset` is a data
  *source*, not part of "how to prepare one input tensor"; nothing
  downstream (`predict()`, `infer.py`) needs it.
- **Supporting `Lambda` via source-inspection or restricted-`eval`**:
  rejected -- still executes caller-supplied code from a file, exactly the
  outcome the brief ruled out; `Normalize` already covers the one real
  consumer's actual need (pixel scaling).
- **A `forge predict` CLI subcommand wired directly into the framework**:
  out of scope for this milestone (M68 already deferred this); `infer.py`
  demonstrates the same workflow at the example level, which is sufficient
  to prove the underlying mechanism without committing to CLI-surface
  design decisions prematurely.

## 14. Practical Product Impact

Before this milestone: a saved Forge model was architecture + weights only.
A developer reloading it in a new process for inference on a new image had
to already know -- from memory, from reading the original training script,
or from a comment -- exactly which `Resize`/scaling steps to reproduce, in
which order, with no way for Forge itself to check or supply that
information. After this milestone: `save_model(model, path,
preprocessing=...)` records that configuration in the same file;
`load_preprocessing(path)` reconstructs it; a completely separate process
(`infer.py`, verified as an actual second `python` invocation, not a
simulated one) goes from "a model file and an image path" to a correct
prediction with no hidden knowledge required. This is the concrete
transition M71's brief asked for: from "Forge can perform this workflow"
toward "Forge reliably preserves and reproduces what the workflow needs."

## 15. Recommended Next Step

Not selected here, per the brief's own instruction not to let milestone-
count pressure manufacture unjustified work; candidates for a future
milestone, in rough order of evidence strength:
1. A `forge predict`/`forge infer` CLI subcommand that wraps exactly what
   `infer.py` does today (`load_model` + `load_preprocessing` + decode +
   predict) as a first-class Forge command, now that two independent
   examples (`infer.py` here, `persistence_demo.py` earlier) have
   demonstrated the same shape by hand.
2. Extending `preprocessing=` support to `save_checkpoint()`/
   `load_checkpoint()`, if a real resume-from-fresh-process workflow
   demonstrates a need (none has yet -- Section 7).
3. Returning to M52's still-deferred normalization/attention direction, or
   a fresh product-decision survey per M49's established discipline, if no
   further persistence-adjacent gap is found.
