# M77 — Portable Artifacts for the MNIST Family: Preprocessing/Classes Persistence + Fresh-Process Inference

## 1. Executive summary

M77 investigated the gap between Forge's current end-to-end workflows and
"using a trained model as an application," using
`examples/image_folder_classification` as the reference-quality workflow the
brief named. The investigation found that Milestone 71/72's
preprocessing-and-classes persistence mechanism (`save_model(...,
preprocessing=..., classes=...)` / `load_preprocessing()` / `load_classes()`)
-- the exact mechanism that makes a `.forge` file a genuinely
self-contained, portable artifact -- has **never been used outside the one
example that introduced it**. `mnist`, `resnet`, and `autoencoder` all still
save a bare model with no preprocessing or class metadata, because each
one's pixel-scaling transform uses `Lambda`, which Forge deliberately
refuses to serialize. This is not a hypothetical gap: attempting
`save_model(..., preprocessing=build_transform())` against any of these
three examples as they stood raised `PersistenceError` immediately.

A second, related finding, confirmed by direct experimentation rather than
assumed: of Forge's examples with custom-registered (non-`Sequential`)
`Module` trees that are actually saved to disk, none had ever been
`load_model()`-ed by a process that had not already imported their own
`model.py`. `load_model()` on a `resnet`-saved artifact reliably fails with
`PersistenceError` ("not registered for persistence in this process")
unless `examples.resnet.model` is imported first -- a real gap in what had
been *implicitly assumed* proven by `image_folder_classification`'s own
fresh-process story, which only ever exercises `Sequential` (needing no
registration at all).

M77 fixed the mechanical `Lambda` blocker in `mnist` and `autoencoder`
(replacing it with the already-established `Normalize`-based equivalent
`image_folder_classification` uses), extended `preprocessing=`/`classes=`
persistence to `mnist`, `resnet`, and `autoencoder`, and added
`examples/mnist/infer.py` -- Forge's flagship example's first standalone,
fresh-process inference script, proving the complete "train → save → move
the file → load in a fresh process → classify a brand-new image" workflow
for a second, real-dataset example, and (via `resnet`) for a
custom-registered Module tree with persisted preprocessing/classes for the
first time. No `forge/` framework file was changed.

## 2. Product-vision context

`docs/product/vision.md`'s **Core workflow** ends in "...evaluation →
persistence → inference," and its **Success** section names "reload them,
perform inference" directly. M77's brief sharpened this to: "A developer
should be able to use Forge to solve a machine-learning problem without
needing to understand Forge's internal Tensor/autograd/training machinery,"
using the classification workflow's own `predict()`/`interpret_
classification()`/preprocessing-persistence chain as the reference. This
milestone's finding is squarely about that chain: the mechanism exists and
works, but had a real, previously undiscovered mechanical blocker (`Lambda`)
in every consumer except the one that originally justified building it, and
an unverified assumption (custom-class fresh-process portability) baked
into the product story.

## 3. Developer workflow examined

Per the brief, the primary reference was the complete `image_folder_
classification` lifecycle:

```text
raw files -> ImageFolder -> preprocessing -> train -> checkpoint ->
save_model -> move artifact elsewhere -> load_model -> load_preprocessing ->
load_classes -> new image -> predict -> interpret result
```

This was traced end to end by reading `examples/image_folder_classification/
train.py` and `infer.py` directly (not just their docstrings), confirming
every step already works and is already tested
(`tests/test_classification_metadata.py::
test_end_to_end_train_save_classes_then_fresh_process_infer`).

The same lifecycle was then traced for `mnist`, `resnet`, `autoencoder`,
`segmentation`, `waveform_classification`, and the RNN examples
(`char_rnn`/`word_rnn`/`long_range_recall`), specifically asking: does this
example's `save_model()` call carry `preprocessing=`/`classes=`? Does it
have (or need) a standalone, `train.py`-independent inference script? Does
its saved artifact type require `register_module()`, and if so, has loading
it in a process that hasn't imported its own `model.py` ever actually been
tried?

## 4. Repository investigation

Directly inspected/executed:

- `forge/data/` (`image_folder.py`, `transforms.py`, `dataset.py`,
  `dataloader.py`), `forge/serialization/` (`model.py`, `registry.py`,
  `transforms.py`, `archive.py`), `forge/training/` (`inference.py`,
  `session.py`), `forge/cli/model.py`.
- Every `examples/*/train.py`, `examples/*/model.py`,
  `examples/image_folder_classification/infer.py`.
- `docs/product/vision.md`, `use-cases.md`, `docs/architecture/
  persistence.md` (in full -- **Custom-module limitations**, **Preprocessing
  metadata**, **Class-label metadata** sections directly informed this
  milestone's direction).
- `docs/development/m68/m71/m72/m76-*.md` and the persistent memory record
  of M49-M76, to confirm what was already established and must not be
  rebuilt (see §12).
- Ran `grep -rln "Lambda(" examples/*/train.py examples/*/dataset.py`,
  finding exactly three matches: `mnist/train.py`, `autoencoder/train.py`,
  and `image_folder_classification/train.py` (the last already fixed in
  M71).
- Ran `grep -rn "register_module" examples/` to enumerate which examples'
  models need explicit registration: `autoencoder`, `char_rnn`,
  `long_range_recall`, `resnet`, `word_rnn`. `regression`, `segmentation`,
  `waveform_classification` explicitly document needing **no**
  `register_module()` call (pure `Sequential`/built-in-layer trees).
- Directly executed (before writing any fix) `forge.serialization.
  load_model()` against an already-saved `examples/autoencoder/artifacts/
  autoencoder_model.forge` in a fresh Python process that had **not**
  imported `examples.autoencoder.model` -- confirmed `PersistenceError:
  Cannot load module type 'ConvAutoencoder': it is not registered for
  persistence in this process.` Re-ran with `import examples.autoencoder.
  model` first -- confirmed it then loads correctly. This is the direct
  evidence behind §7's second finding, not an inference from documentation.

## 5. Existing capabilities (must not be rebuilt)

Per the brief's explicit list, none of these were touched or reimplemented:

- `Trainer.evaluate()` / `Metric` (M76) -- evaluation abstraction, unchanged.
- `TrainingSession` (M73) -- fresh-or-resumed training, unchanged.
- `predict()` (M68) -- standalone inference, reused unmodified by the new
  `infer.py` and by every retrofitted `train.py`.
- `save_model(..., preprocessing=...)` / `load_preprocessing()` (M71) --
  reused unmodified; M77 only adds new *callers*.
- `save_model(..., classes=...)` / `load_classes()` /
  `interpret_classification()` (M72) -- reused unmodified.
- `generate_sequence()` (M75), `save_image()` (M76) -- unrelated to this
  milestone's finding, `save_image()` is reused as-is to write the new
  MNIST demo image.

## 6. Concrete remaining friction

Two concrete, evidenced items, both about the "artifact self-containment"
category the brief named:

1. **`Lambda`-blocked preprocessing.** `mnist`, `resnet` (via `mnist`'s
   `build_transform`), and `autoencoder` each scale pixels with
   `Lambda(lambda x: x * (1/255))`. `forge/serialization/transforms.py`
   deliberately never registers `Lambda` (it wraps an arbitrary Python
   callable with no safe serialized form) -- so none of these three real,
   already-existing, already-trained example workloads can call
   `save_model(..., preprocessing=...)` at all. Only `image_folder_
   classification` (fixed in M71) can.
2. **Unverified custom-class fresh-process portability.** Every example
   whose model needs `register_module()` (`autoencoder`, `resnet`,
   `char_rnn`, `word_rnn`, `long_range_recall`) only ever calls
   `load_model()` from inside the same process that already imported its
   own `model.py` (hence already ran `register_module()`). No example had
   ever proven -- by an actual standalone script or test -- that its saved
   artifact loads correctly from a process that imports nothing but `forge`
   itself plus (necessarily) that one model module. This is documented as
   deliberate, permanent Forge behavior in `docs/architecture/
   persistence.md`'s **Custom-module limitations** section, but no example
   had ever demonstrated it actually working end to end.

## 7. Evidence establishing the problem

- `grep -rln "Lambda(" examples/*/train.py examples/*/dataset.py` ->
  `autoencoder/train.py`, `mnist/train.py` (plus `image_folder_
  classification/train.py`, already fixed).
- Direct execution: `save_model(model, path,
  preprocessing=Compose([Lambda(...), Normalize(...)]))` raises
  `PersistenceError` immediately (regression-guarded by
  `test_mnist_example_integration.py::
  test_save_model_with_lambda_based_transform_would_have_failed`).
- Direct execution: `forge.serialization.load_model()` against a real,
  already-saved `autoencoder_model.forge` in a process that had not
  imported `examples.autoencoder.model` raised `PersistenceError`; importing
  the module first made the identical call succeed (§4, reproduced again in
  §18 against `resnet`).
- `grep` across `examples/*/` found **zero** existing `infer.py`-style
  standalone scripts outside `image_folder_classification`.

## 8. Existing consumers/workflows

Three real, already-existing, already-trained example workloads directly
affected: `examples/mnist/train.py` (Forge's flagship example, real
downloaded MNIST), `examples/resnet/train.py` (reuses `mnist`'s dataset and
`build_transform`), `examples/autoencoder/train.py` (also real MNIST). All
three were run end-to-end against real data (not synthetic stand-ins) both
before and after the fix to confirm unchanged training behavior (§18).

## 9. Alternatives considered

- **Add the `source_module` diagnostic to persistence metadata** (record
  `cls.__module__` per node, purely to improve `PersistenceError`'s message
  when a type is unregistered). Investigated by prototyping the error path;
  rejected for this milestone -- it would be a real but purely
  informational, unevidenced-by-any-actual-developer-confusion nicety
  layered on top of the *actual* finding (that a truly independent,
  never-executed process boundary needed proving/fixing for real examples).
  Left as a candidate for a future milestone if a real developer
  (not this investigation) is confused by the current message (see §26).
- **Add `examples/resnet/infer.py` and/or `examples/autoencoder/infer.py`**
  (a standalone script per remaining example). Rejected: `mnist/infer.py`
  already proves the identical `Sequential`-model fresh-process pattern
  `image_folder_classification/infer.py` established, and a dedicated test
  (`test_resnet_example_integration.py::
  test_model_persistence_with_preprocessing_and_classes_round_trips`) plus
  direct manual verification (§18) already prove the custom-registered-class
  case works. A third/fourth near-identical script would be "another example
  to increase example count," explicitly out of scope.
- **Fix `forge model predict` (CLI) to auto-detect channel count** so it
  could classify grayscale MNIST models too. Investigated: Forge
  deliberately never introspects a module tree to infer facts like input
  channel count (the same principle `classes=` validation already
  documents -- "not validated against model's actual output width"). Adding
  channel-count detection would break that invariant for a single CLI
  command's convenience. Rejected; documented as a known, permanent scope
  boundary instead (`examples/mnist/README.md`).
- **A generic single-channel/grayscale `ImageFolder._load_image` mode
  parameter** (framework change) instead of a small MNIST-local decode
  helper. Rejected: only one consumer (`mnist/infer.py`) needs this, and
  `ImageFolder`'s own docstring already scopes `_load_image` as an
  RGB-producing helper matching `ImageFolder`'s own dataset convention --
  generalizing it for one caller is exactly the kind of premature framework
  expansion the brief forbids. The ~10-line grayscale decode lives in
  `examples/mnist/infer.py` instead, mirroring `examples/mnist/dataset.py`'s
  own precedent of implementing its own decode logic outside the framework.
- **A `RegressionPrediction`-style interpretation helper, a generalized
  evaluation framework, a CLI training command** -- all already
  investigated and rejected in M72/M73/M76; re-confirmed still inapplicable
  here and not revisited.

## 10. Rejected alternatives (summary)

See §9 -- every rejected alternative either had no evidenced need this
milestone (source_module diagnostic, extra infer.py scripts), would have
violated an existing, deliberate Forge invariant (CLI channel
auto-detection, generalizing `_load_image`), or was already settled by an
earlier milestone.

## 11. Selected direction

Apply the already-established M71 fix pattern (`Lambda` -> `Normalize`) to
its two other real, currently-blocked consumers (`mnist`, `autoencoder`),
extend M71/M72's `preprocessing=`/`classes=` persistence to those two plus
`resnet`, and add one new standalone fresh-process inference script
(`examples/mnist/infer.py`) proving the complete workflow for Forge's
flagship example and (via `resnet`) for a custom-registered Module tree.

## 12. Why it clears the evidence bar

- **Multiple existing real consumers duplicate the exact same blocker**
  (`mnist`, `resnet`, `autoencoder` all use the identical `Lambda`-based
  pixel scale that M71 already diagnosed and fixed once, for a fourth
  consumer, three milestones ago).
- **Direct dependency of the product vision**: "reload them, perform
  inference" (`vision.md`) and the brief's own worked example
  (`load_model`/`load_preprocessing`/`load_classes`/`predict`/
  `interpret_classification`) are exactly what this closes for three more
  real examples.
- **Blocks a real workflow today**: `save_model(model, path,
  preprocessing=build_transform())` against `mnist`/`autoencoder`/`resnet`
  as they stood raised `PersistenceError` immediately -- not a hypothetical,
  a reproduced failure.
- The custom-registered-class fresh-process story was an *implicit
  assumption* the product vision depends on but had never been exercised;
  proving (and now regression-testing) it removes a real, previously
  unverified risk to "a saved artifact travels to a fresh process."

## 13. Exact implementation

**`examples/mnist/train.py`**
- `build_transform()`: `Lambda(lambda x: x * _PIXEL_SCALE)` ->
  `Normalize(mean=0.0, std=255.0)`, composed with the unchanged mean/std
  `Normalize` step. `(x - 0) / 255 == x / 255`, bit-exact with the old
  computation (verified by test, §17).
- `save_model(model, str(model_path))` ->
  `save_model(model, str(model_path), preprocessing=build_transform(),
  classes=[str(d) for d in range(10)])`.
- Added a Section-11/12-style demo (mirroring `image_folder_
  classification/train.py`): interpret the existing round-trip query sample
  via `load_classes()`'s reconstructed vocabulary; write a brand-new, raw
  (never-normalized) test digit to a real PNG via `forge.data.save_image()`;
  classify it using `load_preprocessing()`'s reconstructed pipeline (not
  this process's own `build_transform()` call); print the
  `examples.mnist.infer` CLI invocation.

**`examples/mnist/infer.py`** (new file) -- mirrors `image_folder_
classification/infer.py`'s `parse_args()`/`run()`/`main()` shape exactly:
`load_preprocessing()` (raises `forge.PersistenceError` if absent, same
policy), `load_model()`, a local `_load_digit_image()` helper (Pillow
`convert("L")` + resize-if-needed to `28x28`, since `ImageFolder._load_image`
is RGB-only by design), `predict()`, `interpret_classification()`.

**`examples/resnet/train.py`**
- `save_model(model, str(model_path))` ->
  `save_model(model, str(model_path),
  preprocessing=build_transform(), classes=[str(d) for d in range(_NUM_CLASSES)])`
  (reuses `examples.mnist.train.build_transform`, already imported,
  unmodified by this change).
- Added the same `load_classes()` + `interpret_classification()` demo as
  `mnist/train.py`'s Section 11, on the existing round-trip query sample.

**`examples/autoencoder/train.py`**
- `build_transform()`: `Lambda(lambda x: x * _PIXEL_SCALE)` ->
  `Normalize(mean=0.0, std=255.0)` (single-step `Compose`, no mean/std
  zero-centering, matching the existing docstring's own justification for
  why reconstruction targets stay unbounded/non-zero-centered).
- `save_model(model, str(model_path))` ->
  `save_model(model, str(model_path), preprocessing=build_transform())` (no
  `classes=` -- an autoencoder has no class vocabulary).
- Removed the now-dead `_PIXEL_SCALE` module constant in both `mnist/
  train.py` and `autoencoder/train.py` (only ever referenced by the removed
  `Lambda` calls).

No `forge/` file was changed.

## 14. Files changed

- `examples/mnist/train.py` -- `build_transform()` rewrite,
  `save_model(preprocessing=, classes=)`, new fresh-image inference demo.
- `examples/mnist/infer.py` -- new file.
- `examples/mnist/README.md` -- documented preprocessing/classes
  persistence, the new standalone-inference workflow, and the `forge model
  predict` CLI scope limitation.
- `examples/resnet/train.py` -- `save_model(preprocessing=, classes=)` +
  interpretation demo.
- `examples/resnet/README.md` -- documented preprocessing/classes
  persistence on a custom-registered tree, and the "must import
  `examples.resnet.model` first" fresh-process requirement.
- `examples/autoencoder/train.py` -- `build_transform()` rewrite,
  `save_model(preprocessing=)`.
- `examples/autoencoder/README.md` -- documented preprocessing persistence.
- `examples/README.md` -- updated the `mnist/` row.
- `tests/test_mnist_example_integration.py` -- 5 new tests.
- `tests/test_resnet_example_integration.py` -- 1 new test.
- `tests/test_autoencoder_example_integration.py` -- 2 new tests.
- `docs/development/progress.md` -- appended the M77 entry.
- `docs/development/m77-mnist-family-portable-artifacts.md` -- this report
  (new file).

No `forge/tensor`, `forge/autograd`, `forge/backend`, `forge/nn`,
`forge/optim`, `forge/data`, `forge/serialization`, or `forge/training` file
was touched.

## 15. Public API changes

None. Every function used (`save_model`, `load_model`, `load_preprocessing`,
`load_classes`, `predict`, `interpret_classification`, `save_image`,
`Normalize`, `Compose`) is pre-existing and unmodified. `examples/mnist/
infer.py` is a new example script, not a framework API.

## 16. Architecture impact

None. This milestone is a pure application of existing, already-designed
mechanisms (M71 preprocessing persistence, M72 class persistence, M68
`predict()`) to real consumers that had not yet adopted them, plus one new
example-local file. No Tensor, autograd, backend, module, or persistence
*format* code was touched, per the brief's Architecture Discipline
constraint.

## 17. Testing

`tests/test_mnist_example_integration.py` (5 new tests):
- `test_build_transform_is_bit_exact_with_the_old_lambda_based_pipeline` --
  numeric regression guard: the rewritten pipeline must compute exactly what
  the old `Lambda`-based one did.
- `test_model_persistence_with_preprocessing_and_classes_round_trips_to_a_new_raw_image`
  -- trains a tiny model, saves with `preprocessing=`/`classes=`, reloads
  both independently, classifies a brand-new raw-pixel image.
- `test_save_model_with_lambda_based_transform_would_have_failed` --
  regression guard proving the *reason* for the fix: a `Lambda`-containing
  pipeline is still, and must remain, unsaveable.
- `test_infer_run_classifies_a_fresh_process_style_png` -- `infer.py::run()`
  against a real PNG (written via `forge.data.save_image()`), cross-checked
  against a hand-assembled `load_preprocessing`/`load_model`/`predict`/
  `interpret_classification` call on the identical decoded input.
- `test_infer_run_without_preprocessing_raises_persistence_error` -- error
  path when a model was saved with no `preprocessing=`.

`tests/test_resnet_example_integration.py` (1 new test):
- `test_model_persistence_with_preprocessing_and_classes_round_trips` --
  proves `preprocessing=`/`classes=` persistence on a custom-registered
  (`ResNetMNIST`/`ResidualBlock`) tree, not only `Sequential`.

`tests/test_autoencoder_example_integration.py` (2 new tests):
- `test_build_transform_is_bit_exact_with_the_old_lambda_based_pipeline`.
- `test_model_persistence_with_preprocessing_round_trips`.

Not applicable/correctly omitted: no persistence-format version bump (no
new metadata key was introduced -- `preprocessing=`/`classes=` are M71/M72's
existing optional keys), no new CUDA kernel or backend code (nothing here is
device-specific beyond `Module.to()`/`Tensor.to()`, already covered by every
example's existing CUDA integration suite).

## 18. End-to-end verification

Ran real, non-synthetic, full workloads (not just unit tests):

```bash
python -m examples.mnist.train --epochs 1 --output-dir <tmp>
```
Real MNIST, 1 epoch: 95.43% validation accuracy (consistent with this
example's documented ~95-98% single-epoch range), `Saved model +
preprocessing + classes -> ...`, `Inference demo (test sample 0, true digit
7): Prediction: digit 7, Confidence: 100.0%`, and a brand-new raw digit
image correctly classified (`New image 'new_digit_query.png' ... true digit
2: Prediction: digit 2, Confidence: 99.1%`).

```bash
python -m examples.mnist.infer --model <model_path> --image <new_image_path>
```
Run as a genuinely separate `python -m` process: reproduced the identical
`Prediction: digit 2 / Confidence: 99.1%` result, using only the `.forge`
file -- no in-memory state from the training run.

```bash
python -m examples.resnet.train --epochs 1 --batch-size 256 --output-dir <tmp>
```
Real MNIST, 1 epoch: 97.79% validation accuracy (consistent with M66's
documented multi-epoch trajectory), `Saved model + preprocessing + classes
-> ...`.

Then, in a **separate `python -c` invocation, not the training process**:
```python
from forge.serialization import load_model
load_model(<resnet_model_path>)  # PersistenceError: 'ResNetMNIST' not registered
```
raised `PersistenceError` as predicted; re-run with `import
examples.resnet.model` first (no other change) succeeded, and
`load_classes()`/`load_preprocessing()` both returned the expected saved
values (`['0', ..., '9']`, a working `Compose`) -- direct, reproducible
confirmation of §6's second finding and of its documented resolution.

```bash
python -m examples.autoencoder.train --epochs 1 --output-dir <tmp>
```
Real MNIST, 1 epoch: 64.5% MSE reduction over the mean-image baseline
(consistent with M76's documented 67.2% at a similar epoch count), `Saved
model + preprocessing -> ...`. A follow-up direct check
(`save_model(..., preprocessing=build_transform())` -> `load_preprocessing()`
-> `load_model()` -> forward pass on a fresh raw-pixel Tensor) confirmed the
round trip produces a correctly shaped `(1, 1, 28, 28)` reconstruction.

Full suite: `python -m pytest tests/ -q` -> **2,254 collected, 2,253
passed, 1 failed** (2,246 + 8 new). The one failure is
`tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, the same
pre-existing CUDA-allocator-measurement flake documented since M63 (see
[[forge-testing-conventions]]/memory); re-run in isolation immediately after
and passed cleanly, confirming no M77 regression.

## 19. CPU/CUDA verification where applicable

Everything changed in M77 is preprocessing/persistence/example-script logic
that already has its own CPU/CUDA-parity coverage from earlier milestones
(`save_model`/`load_model` CUDA device semantics: M13; `Normalize`: no
device-specific code, pure Tensor arithmetic already exercised by every
example's existing CUDA integration suite). No new CUDA-specific behavior
was introduced, so no new CUDA test file was needed; the existing
`tests/test_mnist_example_cuda_integration.py`, `tests/
test_resnet_example_cuda_integration.py`, and `tests/
test_autoencoder_example_cuda_integration.py` files (unmodified by this
milestone) continue to cover CUDA residency/parity for these three
examples' training pipelines. `--device cuda` was not re-run end-to-end for
this milestone's new inference demo specifically (matching M76's own
precedent: the CUDA path is exercised by dedicated, existing tests rather
than a full hardware re-run of every new print/demo block).

## 20. Persistence/compatibility impact

None to the format itself. `preprocessing=`/`classes=` are M71/M72's
existing optional metadata keys (no `FORMAT_VERSION` bump then, none now) --
this milestone only adds new *callers* passing them. Files saved by
`mnist`/`resnet`/`autoencoder` **before** this milestone remain fully
loadable (`load_model()` never required these keys); they simply return
`None` from `load_preprocessing()`/`load_classes()`, exactly as documented.
No backward-compatibility shim was needed or added.

## 21. Before/after developer workflow

**Before M77** (`mnist`, `resnet`, `autoencoder`):
```python
save_model(model, path)  # no preprocessing, no classes
...
# in a DIFFERENT process, a developer with a new digit image had to:
# 1. already know MNIST's exact normalization constants (0.1307, 0.3081)
#    and pixel-scale convention -- nothing in the file records this,
# 2. reimplement Resize/Normalize by hand,
# 3. reimplement digit-index -> label bookkeeping (trivial here, but the
#    file itself gave no confirmation of the class count/order),
# 4. attempting save_model(..., preprocessing=build_transform()) at all
#    would have raised PersistenceError immediately (Lambda unregistered).
```

**After M77**:
```python
save_model(model, path, preprocessing=build_transform(), classes=[...])
...
# fresh process, no in-memory state from training:
preprocessing = forge.load_preprocessing(path)
model = forge.load_model(path)
classes = forge.load_classes(path)
image = _load_digit_image("my_photo.png")     # the one small, genuinely
                                                # application-specific step
result = forge.interpret_classification(
    forge.predict(model, preprocessing(image).reshape(1, 1, 28, 28)), classes
)
print(result.label, result.confidence)
```
or, equivalently, as an actual runnable command:
```bash
python -m examples.mnist.infer --model mnist_model.forge --image my_photo.png
```
A developer with a saved MNIST-family `.forge` file no longer needs to know
or re-derive MNIST's own normalization convention, and can verify the
complete workflow with one command instead of writing framework glue.

## 22. Limitations

- `examples/mnist/infer.py`'s grayscale decode assumes a single dominant
  digit roughly filling the frame (matching real MNIST's own crop
  convention) -- it does not do any digit localization/cropping for an
  arbitrary photo with background clutter. This mirrors
  `image_folder_classification/infer.py`'s own equivalent assumption (a
  clean, single-subject image) and is not a new limitation this milestone
  introduces.
- `forge model predict` (the CLI command) remains RGB-only and therefore
  unusable for MNIST-family (1-channel) models -- documented, not fixed
  (§9). `examples/mnist/infer.py` is the correct tool for this case.
- `resnet`/`char_rnn`/`word_rnn`/`long_range_recall` still require the
  loading process to import their own `model.py` before `load_model()` will
  succeed -- this is Forge's deliberate, permanent security-motivated
  design (no dynamic import of unregistered types), not a bug, but it means
  a truly "hand the `.forge` file to someone with zero context" story still
  requires them to also have (or be told to `pip install`/copy) the small
  Python file defining the architecture. This is now explicitly documented
  per affected example rather than silently assumed.
- `autoencoder`/`resnet` do not get their own dedicated `infer.py` this
  milestone (§9) -- `mnist/infer.py` plus each example's own integration
  test cover the same underlying mechanism.

## 23. Deferred work

- A `source_module` persistence-metadata hint (purely diagnostic, to make
  `PersistenceError`'s "not registered" message name the Python module a
  custom type was originally defined in) remains a plausible, small future
  improvement (§9) -- not built this milestone since no real developer
  confusion (only this investigation's own deliberate probing) has yet
  surfaced the need.
- Dedicated `examples/resnet/infer.py` / `examples/autoencoder/infer.py`
  standalone scripts remain unbuilt -- build if a future milestone finds a
  concrete reason `mnist/infer.py`'s proof isn't sufficient evidence for
  those two examples specifically.
- `char_rnn`/`word_rnn`/`long_range_recall` were not touched this
  milestone -- their inference story (`generate_sequence()`, M75) is
  already a different shape (autoregressive generation, not
  classify-one-input) and was out of scope for this investigation.

## 24. Relationship to the long-term Forge vision

`docs/product/vision.md`'s **Success** section names "reload them, perform
inference" as a first-class outcome, and M77's own brief frames the goal as
removing the need to "understand Forge's internal Tensor/autograd/training
machinery." M71/M72 already built the *mechanism* for this; M77's
contribution is proving and extending that mechanism to Forge's actual
flagship, most-referenced example (MNIST) and to a residual-CNN example with
a genuinely different (custom-registered) module-tree shape -- closing the
gap between "the capability exists" and "the capability is proven to work
for more than one example," which is what actually matters for a developer
deciding whether to trust it.

## 25. Practical developer impact

Before M77, only one of Forge's ten examples (`image_folder_classification`)
had ever produced a `.forge` file that could be handed to a fresh process
and used to classify a brand-new input with zero extra knowledge. After
M77, Forge's flagship MNIST example does too, verified as a genuinely
separate OS process invocation, and the mechanism is now proven (via
`resnet`) to also work correctly for the more architecturally complex
custom-registered-Module case the product vision depends on but had never
actually exercised.

## 26. Follow-up triggers

- If a real user reports confusion from `PersistenceError`'s current "not
  registered for persistence in this process" message (rather than this
  investigation's own deliberate probing), that is the point to add the
  `source_module` diagnostic hint (§23).
- If a concrete workload needs to classify a photo with a MNIST-family model
  where the digit isn't already cleanly cropped/centered, that is the point
  to consider a shared cropping/localization helper -- not before.
- If `forge model predict`'s RGB-only decode blocks a second real grayscale
  workload (beyond this milestone's own MNIST case, which now uses
  `examples.mnist.infer` instead), that is the point to reconsider whether
  the CLI needs a `--channels`/mode flag, weighed against Forge's existing
  no-introspection invariant.

## 27. Suggested commit message

```
feat: extend preprocessing/classes persistence to mnist/resnet/autoencoder

M71/M72's save_model(..., preprocessing=..., classes=...) mechanism was
never used outside image_folder_classification: mnist/resnet/autoencoder
all scale pixels via Lambda, which Forge deliberately refuses to serialize,
so attempting to save preprocessing there raised PersistenceError
immediately. Replaced Lambda(lambda x: x/255) with the already-established
Normalize(mean=0, std=255) equivalent in mnist/autoencoder (bit-exact,
verified by test), and wired preprocessing=/classes= through all three
examples' save_model() calls -- resnet is the first real proof this
mechanism works on a custom-registered (non-Sequential) Module tree.

Added examples/mnist/infer.py, Forge's flagship example's first standalone
fresh-process inference script, mirroring image_folder_classification's
infer.py. Also directly confirmed (and documented) that load_model() on a
custom-registered artifact requires the loading process to import that
model's defining module first -- a real, previously unverified constraint
on artifact portability, not a bug.
```
