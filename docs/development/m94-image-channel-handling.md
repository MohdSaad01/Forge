# Milestone 94 — Robust Image Classification Input Handling

## Objective

Turn a real, M93-discovered inference defect into a properly defined,
tested, general image-input contract: make Forge's image-classification
artifact inference robust to legitimate grayscale image inputs, without
introducing an MNIST-specific workaround.

## M93-discovered failure

> `forge model predict` on a grayscale MNIST PNG raises a channel-mismatch
> error.

Reproduced from both the source tree and the installed package -- not a
packaging defect.

## Exact reproduction

```bash
python -m forge model predict \
    examples/mnist/artifacts/mnist_model.forge \
    examples/mnist/artifacts/new_digit_query.png
```

```text
Error: Conv2d(in_channels=1) cannot accept input with 3 channels (input shape (1, 3, 28, 28)).
```

`examples/mnist/artifacts/new_digit_query.png` is a real asset --
`examples/mnist/train.py` writes it from an actual MNIST test-set image
(`save_image(Tensor(raw_image.numpy() / 255.0), new_image_path)`), genuinely
mode `"L"` (single-channel grayscale), not a synthetic fixture. The same
error reproduced identically through `forge.predict_artifact()` (the public
Python API `forge model predict` delegates to).

## Root cause

`forge.training.inference.predict_artifact()`/`predict_image_artifact()`
both decoded the input image via `forge.data.ImageFolder._load_image()`,
which **unconditionally** converted every image to RGB
(`Image.convert("RGB")`) regardless of what the saved model actually
expected. This was correct for `ImageFolder`'s own dataset contract (every
training sample must share one shape for `DataLoader` batching, and every
existing `ImageFolder`-based example trains a 3-channel model), but
`predict_artifact()`/`predict_image_artifact()` reused the same decode step
for arbitrary artifacts -- including `examples/mnist`'s `Conv2d(1, ...)`
model, which `ImageFolder` was never involved in training at all (MNIST
trains from the IDX format via `MNISTDataset`, not `ImageFolder`).

The mismatch was real, not cosmetic: the RGB-decoded, preprocessed image
reached `Conv2d.forward()`'s own shape check, which raised a `ShapeMismatchError`
-- a genuinely correct, already reasonably clear error, but for an input
that a developer had every reason to believe was valid (a real grayscale
digit image for a grayscale digit classifier).

## Inference path involved

```text
forge model predict (CLI)
    -> forge.predict_model()
        -> forge.training.inference._determine_workflow()  (task="classification")
        -> forge.predict_artifact()
            -> load_preprocessing() / load_model()
            -> ImageFolder._load_image(path)          <-- always channels=3 (the bug)
            -> preprocessing(raw)
            -> predict()  -> Conv2d.forward()          <-- raised here
            -> load_classes() / interpret_classification()
```

`predict_image_artifact()` (segmentation) shared the identical unconditional
decode call, though no shipped segmentation model currently expects a
non-3-channel input, so it never manifested there.

## Existing image-input contract (before this milestone)

- Internal representation: `(C, H, W)`, channels-first, `float32`, raw
  `[0, 255]` range -- Forge's one established image-tensor convention
  (`ImageFolder`, `Resize`, `save_image`, `examples/mnist/dataset.py` all
  already agree on this).
- `ImageFolder`/`predict_artifact()`/`predict_image_artifact()`: always
  decoded to 3 channels (RGB), unconditionally -- grayscale replicated
  across channels, RGBA alpha discarded.
- `examples/mnist/infer.py` (a separate, independent, hand-rolled
  application script -- explicitly documented as *not* reusing
  `ImageFolder._load_image()`, since it always produced RGB) had its own
  `_load_digit_image()` that decoded via `Image.convert("L")` -- proving
  Forge's own example code already understood grayscale decoding was
  necessary for this model shape, just not at the shared framework
  boundary.
- No artifact metadata recorded an image's expected channel count.

## Final image-input contract

`forge.training.inference._expected_image_channels(model)` reads the
model's own already-fixed contract: the `in_channels` of the first `Conv2d`
found via `model.modules()` (self-first depth-first -- the same order
`save_model()` walks the tree). This is genuinely reliable for every real
Forge image model today: `examples/mnist` is `Conv2d(1, ...)`;
`examples/image_folder_classification`/`examples/segmentation` are both
`Conv2d(3, ...)`. No new metadata was needed -- the model's architecture
already fully determines this, exactly the same "read an existing signal,
don't invent a new one" discipline `_legacy_infer_workflow()` (M86) already
established for `ModelSummary.module_types`.

`forge.training.inference._decode_image_for_model(model, image)` composes
that channel count with `ImageFolder._load_image(path, channels=...)`
(Milestone 94's new parameter, default `3` -- `ImageFolder.__getitem__`
itself always passes the default, so its own dataset contract is
byte-for-byte unchanged) to decode the image directly into the
representation the model expects, *before* `preprocessing` runs.

## Supported channel modes

- `channels=1` -> `Image.convert("L")`: identity for an already-grayscale
  source; Pillow's standard luminance-weighted conversion for an RGB/RGBA
  source.
- `channels=3` -> `Image.convert("RGB")`: the original, unconditional
  pre-M94 behavior -- grayscale replicated across channels, RGBA alpha
  discarded.
- Any other requested channel count (a model whose first `Conv2d` expects,
  e.g., 2 or 4 channels) is rejected by `ImageFolder._load_image()` with a
  clear `DataError` naming the unsupported count. No Forge model today
  has such a first layer, so this is a defined boundary, not a gap.
- A model with no `Conv2d` at all (`_expected_image_channels()` returns
  `None`) falls back to the original `channels=3` default -- no image
  model shape this milestone did not change loses its prior behavior.

## Conversion behavior

| Model expects | Image is | Result |
|---|---|---|
| 1 (grayscale) | grayscale | decoded as-is (Case A) |
| 3 (RGB) | RGB | decoded as-is (Case B, unchanged) |
| 3 (RGB) | grayscale | replicated across channels (Case C, unchanged `ImageFolder` behavior) |
| 1 (grayscale) | RGB/RGBA | converted via standard luminance formula (Case D, new) |
| 1 (grayscale) | RGBA | alpha discarded, then luminance conversion (same `convert("L")` call) |
| other (2, 4, ...) | any | `DataError` naming the unsupported count |

Conversion happens at decode time -- before `Resize`/`Normalize`/any other
persisted preprocessing step -- so a transform that assumes a particular
channel count never sees the wrong one.

## Rejection behavior

`ImageFolder._load_image()` raises `forge.DataError` immediately for
`channels` outside `{1, 3}`, naming the unsupported value, rather than
letting decoding proceed and failing later inside the model. `Conv2d.
forward()`'s own existing `ShapeMismatchError` (`"Conv2d(in_channels=N)
cannot accept input with M channels..."`) remains the safety net for any
image-consuming model shape this milestone's heuristic does not cover (e.g.
a hypothetical future non-`Conv2d`-first image model) -- unchanged, and
already sufficiently clear.

## Architecture decision

Input adaptation only -- no architecture change. `_expected_image_channels()`
reads `Conv2d.in_channels`, an existing public attribute; no `Conv2d`,
`Sequential`, or other module type was modified. No new task type (grayscale
vs. RGB is an input representation, not a new ML problem -- `task=
"classification"` is unchanged). No new artifact metadata, no
`FORMAT_VERSION` bump -- the model's own already-persisted architecture is
sufficient signal. No new image-dataset abstraction, no image-schema
registry -- `ImageFolder._load_image()` gained one optional parameter,
nothing else.

## Files changed

- `forge/data/image_folder.py`: `ImageFolder._load_image()` gained
  `channels: int = 3`, selecting `Image.convert("L")` vs. `Image.convert
  ("RGB")`; validates `channels in (1, 3)`. `ImageFolder.__getitem__` passes
  no explicit `channels=`, so its own contract is unchanged.
- `forge/training/inference.py`: added `_expected_image_channels()` and
  `_decode_image_for_model()`; `predict_artifact()`/`predict_image_artifact()`
  now call `_decode_image_for_model()` instead of
  `ImageFolder._load_image(Path(image))` directly. Docstrings updated to
  describe the new channel-matching policy.
- `docs/architecture/training-engine.md`: documented the M94 channel-handling
  policy under `predict_artifact()`'s section.
- Tests: `tests/test_image_folder.py`, `tests/test_artifact_inference.py`,
  `tests/test_artifact_inference_cuda.py`, `tests/test_cli_predict.py`,
  `tests/test_packaging_smoke.py` (see below).

## Tests added

- `tests/test_image_folder.py`: `ImageFolder._load_image(channels=)` unit
  coverage -- default unchanged, grayscale decode, RGB->grayscale luminance
  conversion, grayscale->grayscale identity (no lossy RGB round trip),
  unsupported channel count rejection, corrupt-file handling for
  `channels=1`.
- `tests/test_artifact_inference.py`: a new `_TinyGrayscaleCNN` (`Conv2d(1,
  ...)`-first) alongside the existing RGB `_TinyCNN`; Cases A-D (grayscale
  model/grayscale image, RGB model/RGB image regression, grayscale image to
  RGB model, RGB image to grayscale model); a manual-pipeline equivalence
  test proving the channel conversion happens before preprocessing; a real,
  non-synthetic-fixture test against `examples/mnist/artifacts/mnist_model.forge`
  + `examples/mnist/artifacts/new_digit_query.png` (skips cleanly if that
  example hasn't been trained in the checkout) asserting the predicted
  digit is correct (`"2"`), not merely that no exception is raised; an
  unsupported-channel-count `DataError` test; a genuine fresh-process
  (`subprocess`) test for the grayscale shape.
- `tests/test_artifact_inference_cuda.py`: grayscale-model CUDA prediction
  and CPU/CUDA parity, hardware-verified on the reference 940MX.
- `tests/test_cli_predict.py`: CLI-level Cases A/C/D, a fresh-process CLI
  test, and a CUDA-hardware-verified CLI grayscale test.
- `tests/test_packaging_smoke.py`: a real wheel-build + clean-venv-install +
  grayscale-artifact-prediction test, outside the repository (reuses M93's
  fixtures).

## Focused test results

```text
tests/test_image_folder.py ............................. 40 passed
tests/test_artifact_inference.py ..................... 21 passed
tests/test_artifact_inference_cuda.py ..... 5 passed  (real 940MX hardware)
tests/test_cli_predict.py ................................. 33 passed
tests/test_packaging_smoke.py ........ 8 passed  (real wheel build + clean venv)
tests/test_segmentation_artifact_workflow.py + test_unified_artifact_prediction.py
    + test_task_metadata.py + test_classification_metadata.py
    + test_preprocessing_persistence.py .......................... 134 passed
```

## Full-suite result

```text
2,586 collected
2,585 passed, 1 failed -- tests/test_dataloader_prefetch.py::
    test_repeated_epochs_do_not_grow_cuda_or_pinned_memory
```

The one failure is the pre-existing CUDA allocator-measurement flake
documented since ~M63 -- reproduced passing cleanly in isolation
immediately after the full-suite run (`before=320, after=288` bytes vs.
`before=320, after=320` when run alone; a measurement-ordering artifact of
running thousands of other CUDA-touching tests first, not a regression
introduced by this milestone).

## Real MNIST validation

```bash
python -m forge model predict \
    examples/mnist/artifacts/mnist_model.forge \
    examples/mnist/artifacts/new_digit_query.png
```

```text
Prediction: 2
Confidence: 99.6%
```

`new_digit_query.png` is `examples/mnist`'s real test-set sample index 1,
true digit `2` (verified directly against `MNISTDataset`) -- the prediction
is correct, not merely exception-free.

## RGB regression validation

`examples/image_folder_classification/artifacts_m89/image_folder_model.forge`
predicted on a real dataset image (`data/circle/circle_0000.png`) produces
**bit-for-bit identical output** before and after this change (verified via
`git stash`/re-run) -- the RGB decode path is untouched for any model whose
first `Conv2d` expects 3 channels.

## CPU validation

All of the above ran on CPU by default.

## CUDA validation

`forge.predict_artifact(..., device="cuda")` against the real MNIST
artifact, and `tests/test_artifact_inference_cuda.py`'s two new grayscale
tests, both hardware-verified on the reference GeForce 940MX -- identical
prediction to CPU.

## Fresh-process validation

`tests/test_artifact_inference.py::
test_predict_artifact_grayscale_works_from_a_genuinely_separate_process`
and `tests/test_cli_predict.py::test_cli_predict_grayscale_model_fresh_process`
both launch a real `subprocess` that only imports `forge`/invokes the CLI,
with no shared module-cache/import state with the test process.

## Installed-package validation

`tests/test_packaging_smoke.py::
test_installed_forge_predicts_grayscale_image_classification_artifact`
(new, reusing M93's `built_wheel`/`clean_install`/`outside_repo_dir`
fixtures): builds the real wheel via `python -m build`, installs it into a
fresh venv (never `pip install -e .`), saves a grayscale-model artifact
with the dev-tree `forge`, then runs `forge model predict` against a
genuine grayscale PNG through the **installed** distribution, via
`subprocess`, from a working directory outside the repository. Passes.

## Artifact compatibility result

No `FORMAT_VERSION` bump; no new metadata key. Every artifact saved before
this milestone remains loadable and predictable unchanged -- the channel
decision is derived entirely from the model's already-persisted
architecture (`Conv2d.in_channels`), which every prior milestone already
wrote.

## CLI behavior

`forge model predict` (`task="classification"`/`"segmentation"`) now
decodes the input image according to the artifact's own model, transparent
to the caller -- no new flag, no behavior change for existing RGB
workflows.

## Public API behavior

`forge.predict_artifact()`, `forge.predict_image_artifact()`, and
`forge.predict_model()` (which delegates to the first two) all share the
fix identically -- there is exactly one shared decode boundary
(`_decode_image_for_model()`), not a CLI-only or single-function patch.

## Documentation changes

`docs/architecture/training-engine.md`'s `predict_artifact()` section
gained an **Image channel handling (Milestone 94)** paragraph describing
the policy above; this file.

## Remaining limitations

- The unsupported-channel-count path (`{2, 4, ...}`) has no real Forge
  model to exercise it against today -- covered only by a synthetic test
  model built specifically to trigger it.

## Anything discovered but intentionally deferred

- `examples/mnist/infer.py`'s own independent `_load_digit_image()` helper
  could now be simplified to call `ImageFolder._load_image(path,
  channels=1)` directly, since the framework-level decode step supports
  grayscale as of this milestone. Left unchanged: that script is
  deliberately independent of any shared framework decode helper (see its
  own module docstring), was already working correctly, and touching it
  was not required to close the real M93 gap (the shared `predict_artifact()`/
  CLI boundary, which `examples/mnist/infer.py` never used). Retrofitting
  it would be scope creep against Section 30's "smallest general fix"
  discipline.
