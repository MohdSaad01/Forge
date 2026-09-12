# M84 — Portable Image-Output Artifact Workflow: `forge.predict_image_artifact()`

## 1. Executive summary

M82 (image classification) and M83 (numeric regression) each proved Forge's
high-level train -> save -> fresh-process -> predict workflow for one
artifact shape. M84's brief asked whether that workflow extends to a real
image-*output* workload -- `examples/segmentation` or `examples/autoencoder`
-- where the useful result is itself another image, not a label or a
number, and M76 (`forge.data.save_image()`) had already established how to
render such a result but stopped there.

Targeted inspection of both candidates found `examples/segmentation` was the
cleaner fit: its model takes a `(3, H, W)` RGB input and produces a `(1, H,
W)` mask, which lines up almost exactly with `ImageFolder._load_image()`'s
existing `(3, H, W)`, raw `[0, 255]` image-file decode (the same decode
`predict_artifact()` already reuses) and `save_image()`'s `(C, H, W)`, `C in
(1, 3)` output contract. `examples/autoencoder`, by contrast, needs a
single-channel `(1, 28, 28)` MNIST-shaped input, but `ImageFolder._load_image()`
always forces 3-channel RGB -- reusing it there would need a new grayscale
-conversion capability with no existing consumer, exactly the kind of
unjustified addition the brief warns against. Segmentation required no such
new capability.

The one real gap: `examples/segmentation` had never gone through an actual
image *file* at all -- its dataset is generated entirely in-memory as `[0,
1]`-scaled `float32` arrays (Milestone 64), so its saved model never carried
a `preprocessing=` pipeline the way `mnist`/`autoencoder`/
`image_folder_classification` do. M84 built `forge.training.
predict_image_artifact(path, image, *, device=None, threshold=0.5)` and
retrofitted `examples/segmentation/train.py` to save a `Normalize(mean=0.0,
std=255.0)` preprocessing pipeline (rescaling a real decoded `[0, 255]`
image file back into the `[0, 1]` range the model was trained on) and to
demonstrate the complete file-based workflow on a brand-new synthetic image.

## 2. What was actually missing (and what was not)

`Trainer`/`DataLoader`/checkpoint/resume/`save_and_verify()`/`save_image()`
all already worked for this architecture unchanged (M64/M76/M78) -- nothing
about training, evaluation, or rendering an image-shaped output needed to
change. What did not exist:

1. A `preprocessing=` pipeline for segmentation's saved model, so a real
   image *file* (as opposed to an in-memory `[0, 1]`-scaled array this
   process happened to generate) could be rescaled into the model's trained
   input range.
2. A one-call way to turn `(path, new_image_file)` into a savable predicted
   mask -- the "load model, load preprocessing, decode the image, apply
   preprocessing, predict, threshold" sequence `train.py` would otherwise
   have had to hand-assemble, mirroring the sequence `predict_artifact()`
   already replaced for classification.

No persistence format change was needed (`preprocessing=`/`load_preprocessing()`
already fully generic since M71), and no new `Tensor`/`Backend`/`nn`
capability was needed (confirmed by M64's own zero-new-capability finding,
still true here).

## 3. What was implemented

### 3.1 `forge.training.predict_image_artifact()`

`forge/training/inference.py`, alongside `predict_artifact()`/
`predict_tensor_artifact()`:

```python
def predict_image_artifact(
    path: str,
    image: "str | os.PathLike",
    *,
    device: "str | Device | None" = None,
    threshold: float = 0.5,
) -> Tensor:
    ...
```

Behavior:

1. `image` must be a path (`str`/`os.PathLike`); anything else raises
   `forge.DataError` immediately.
2. `load_preprocessing(path)` -- **mandatory**, like `predict_artifact()`:
   there is no way to rescale a freshly decoded `[0, 255]` image into the
   model's trained input range otherwise. Missing preprocessing raises
   `forge.PersistenceError`.
3. `load_model(path, device=device)`.
4. `ImageFolder._load_image(image)` -- the exact `(3, H, W)`, raw `[0, 255]`
   decode `predict_artifact()` already reuses -- then the reconstructed
   preprocessing is applied and a batch dimension added.
5. `predict(model, batch)` -- the model's raw, unbounded per-pixel output.
6. Threshold at `threshold` (default `0.5`, the same `_THRESHOLD`
   `examples/segmentation/metrics.py`/`train.py` already use) to obtain a
   `{0, 1}`-valued mask, drop the batch dimension, and return a CPU `Tensor`
   of shape `(1, H, W)` -- ready for `forge.data.save_image()` unchanged.

### 3.2 `examples/segmentation/train.py`

- New `build_transform()` returning `Compose([Normalize(mean=0.0,
  std=255.0)])` -- the same `Normalize`-replaces-`Lambda` substitution
  `autoencoder`/`mnist`/`image_folder_classification` already established
  (`Lambda` cannot be saved via `save_model(..., preprocessing=...)`).
- `save_and_verify(...)` now passes `preprocessing=build_transform()`.
- New end-of-run demo: a brand-new synthetic `(image, mask)` pair is
  generated at `seed=args.seed + 2` (independent of both the train seed and
  the test seed `+1`), written to disk as real PNGs via `save_image()`
  (`segmentation_new_image.png`, `segmentation_new_ground_truth_mask.png`),
  then read back through nothing but the saved `.forge` file via
  `forge.predict_image_artifact()`, with the result saved as
  `segmentation_new_predicted_mask.png`.
- The pre-existing in-memory `query_x` demo (`segmentation_input.png`/
  `segmentation_predicted_mask.png`/`segmentation_ground_truth_mask.png`,
  Milestone 76) is unchanged -- both demos now coexist: one shows the
  lower-level in-process path, the other the new portable, file-based one.

### 3.3 `examples/segmentation/infer.py` (new)

A standalone script built entirely on `forge.predict_image_artifact()` +
`forge.data.save_image()`, mirroring `image_folder_classification/infer.py`'s
shape exactly:

```bash
python -m examples.segmentation.infer \
    --model examples/segmentation/artifacts/segmentation_model.forge \
    --image examples/segmentation/artifacts/segmentation_new_image.png \
    --output examples/segmentation/artifacts/segmentation_new_predicted_mask.png
```

It imports nothing from `train.py`/`dataset.py`/`model.py` -- everything it
needs (architecture, weights, preprocessing) comes from the `.forge` file.

## 4. Why this implementation was chosen

### 4.1 Segmentation over autoencoder

Autoencoder's `(1, 28, 28)` grayscale MNIST input conflicts with
`ImageFolder._load_image()`'s hard-coded 3-channel RGB decode (used by every
other image-file consumer in Forge, including `predict_artifact()`).
Supporting it would require a new grayscale-conversion capability with no
existing consumer -- exactly the kind of unjustified, speculative addition
the brief's Section 6 scope boundaries warn against. Segmentation's 3-channel
RGB input and `(1, H, W)` mask output need no such new capability.

### 4.2 A separate function, not a generalized image-inference dispatcher

`predict_artifact()` always ends in a class-vocabulary interpretation;
`predict_tensor_artifact()`'s input is never a file. `predict_image_artifact()`
is a third, materially different shape: file input like `predict_artifact()`,
but an image-shaped output requiring its own (not classification, not
"no interpretation at all") conversion. Branching any of the three existing
functions to also cover this shape was rejected for the same reason M83
rejected merging into `predict_artifact()`: the contracts differ in more
than one dimension, and forcing them into one function would be exactly the
"awkward task detection" the brief warns against.

### 4.3 Threshold baked in, not a generic "any image-to-image model" function

`predict_image_artifact()` deliberately reuses one specific,
already-established output convention (`examples/segmentation`'s
threshold-at-`0.5` binary mask, already used by `metrics.py`/`train.py`)
rather than returning the model's raw, unbounded output unconverted. This
is the "smallest correct conversion required by the actual workload" the
brief's Section 9 calls for, reusing the example's own already-defined way
of producing a predicted mask rather than inventing a new one. `threshold`
is exposed as an overridable keyword (matching `save_and_verify()`'s own
`atol=` precedent) but always applied -- there is no `threshold=None`
escape hatch, since no second real consumer with different output semantics
exists yet to justify one. A future image-output workload with genuinely
different semantics (e.g. an unbounded reconstruction) would need its own
function, not a generalization of this one -- see 4.2.

### 4.4 No format change

`preprocessing=`/`load_preprocessing()` (M71) and `Normalize` (already fully
generic, not image-specific -- see M83's own Section 2 finding, still true
here) needed no changes. `FORMAT_VERSION` is unchanged.

## 5. Files changed

- `forge/training/inference.py` -- added `predict_image_artifact()`;
  updated the module docstring and `__all__`.
- `forge/training/__init__.py`, `forge/__init__.py` -- exported
  `predict_image_artifact` at both levels; updated docstrings and `__all__`.
- `examples/segmentation/train.py` -- new `build_transform()`;
  `save_and_verify(..., preprocessing=build_transform())`; new end-of-run
  file-based artifact-inference demo (new synthetic image -> real PNG ->
  `predict_image_artifact()` -> predicted-mask PNG); module docstring
  updated.
- `examples/segmentation/infer.py` (new) -- standalone fresh-process
  inference script.
- `examples/segmentation/README.md` -- new **Portable artifact inference
  (Milestone 84)** section; **Model persistence**/**Integration tests**
  sections updated.
- `tests/test_segmentation_artifact_workflow.py` (new, CPU) -- general
  contract tests for `predict_image_artifact()` plus the real
  `examples/segmentation` end-to-end acceptance test, including a genuine
  `subprocess` fresh-process run of `infer.py`.
- `tests/test_segmentation_artifact_workflow_cuda.py` (new, CUDA; skips
  cleanly without a working CUDA backend) -- CPU/CUDA parity coverage.
- `docs/architecture/training-engine.md` -- new **Portable-artifact
  inference for image-to-image output** section; package-layout table and
  milestone header updated.
- `README.md`, `examples/README.md` -- updated to describe the new artifact
  shape and add the previously-missing `segmentation/` row.
- `docs/development/progress.md` -- new M84 entry.

No changes to `Tensor`/autograd, CUDA kernels, `Trainer`, `TrainingSession`,
optimizer infrastructure, `ImageFolder`/`save_image()`, or the persisted
model-file format.

## 6. Tests added

`tests/test_segmentation_artifact_workflow.py` (17 tests, CPU): re-export
identity; output shape/`{0, 1}`-value contract; direct `save_image()`
compatibility (round-tripped through Pillow and checked for a valid
grayscale PNG); `pathlib.Path` input acceptance; bit-for-bit equivalence
against the manual `load_model()`/`load_preprocessing()`/
`ImageFolder._load_image()`/`predict()`/threshold pipeline it replaces;
`threshold=` overriding the result; explicit `device="cpu"` override;
missing-preprocessing/missing-model/missing-image/corrupt-image/unsupported
-input-type error handling (`PersistenceError`/`DataError`); the real
`examples/segmentation/train.py` run saving preprocessing and all three new
demo artifacts; the artifact reproducing train.py's own saved prediction
exactly; a real-training-quality sanity check (IoU > 0.3 against a
never-trained-on image after 8 real epochs); and a genuine `subprocess`
fresh-process run of both `examples/segmentation/infer.py` (as a script)
and a tiny pre-registered-only model (matching `test_artifact_inference.py`'s
own fresh-process precedent).

`tests/test_segmentation_artifact_workflow_cuda.py` (3 tests, CUDA):
an artifact saved from a CUDA model restores onto CUDA by default; an
explicit `device="cpu"` override works; and CPU- and CUDA-saved artifacts of
identical weights produce bit-for-bit identical thresholded masks.

## 7. Full-suite result

Full suite: **2,358 collected, 2,357 passed, 1 failed** (2,338 pre-M84 + 20
new: 17 in `tests/test_segmentation_artifact_workflow.py`, 3 in
`tests/test_segmentation_artifact_workflow_cuda.py`). The one failure,
`tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, is the same
pre-existing CUDA-allocator-measurement flake documented since M63 --
reproduced passing cleanly in isolation immediately afterward, confirming no
M84 regression. `tests/test_segmentation_example_integration.py` (20/20) and
`tests/test_segmentation_example_cuda_integration.py` (5/5, hardware
-verified on the 940MX) both passed unmodified after the `train.py`
retrofit.

## 8. Real segmentation training verification

```bash
python -m examples.segmentation.train --epochs 15 --device cuda
```

trained to `test pixel_accuracy=0.9981, iou=0.9909` (matching the
already-documented M64 reference numbers, confirming the retrofit is
behavior-preserving), then printed:

```
Saved + verified model + preprocessing -> .../segmentation_model.forge
Saved segmentation input/predicted-mask/ground-truth-mask -> ...

Portable artifact inference (forge.predict_image_artifact()): .../segmentation_new_image.png -> .../segmentation_new_predicted_mask.png
```

Direct inspection of the produced files: `segmentation_new_image.png` is a
real `32x32` RGB PNG; `segmentation_new_predicted_mask.png` and
`segmentation_new_ground_truth_mask.png` are real `32x32` grayscale PNGs
with only `{0, 255}` pixel values. Comparing the predicted mask against the
ground-truth mask for this brand-new (never trained-on) image gives
**IoU = 1.0** -- a pixel-perfect segmentation of a genuinely new shape.

## 9. Fresh-process inference verification

```bash
python -m examples.segmentation.infer \
    --model .../segmentation_model.forge \
    --image .../segmentation_new_image.png \
    --output .../fresh_process_prediction.png \
    --device cuda
```

run as a real, separate `python -m` process against the artifact from
Section 8, produced a PNG **bit-for-bit identical** to the
`segmentation_new_predicted_mask.png` `train.py` itself wrote in-process --
confirming the `.forge` file (model + preprocessing) is genuinely portable,
with no in-memory training state required.
`tests/test_segmentation_artifact_workflow.py::
test_examples_segmentation_infer_works_from_a_genuinely_separate_process`
automates this exact check via `subprocess.run(...)`.

## 10. CUDA verification

`tests/test_segmentation_artifact_workflow_cuda.py` (3/3, hardware-verified
on the reference 940MX) confirms `predict_image_artifact()`'s device
handling: a CUDA-saved artifact restores onto CUDA by default, an explicit
`device="cpu"` override works, and CPU- and CUDA-saved artifacts of
identical weights produce bit-for-bit identical masks (thresholding to
`{0, 1}` absorbs any residual CPU/CUDA floating-point difference before
comparison). Section 8/9's real end-to-end run also used `--device cuda`
throughout.

## 11. Limitations

- `predict_image_artifact()` always applies the M64 threshold-at-`0.5`
  binary-mask convention -- it is not a generic "any image-to-image model"
  function (see Section 4.3). A future workload needing raw, unconverted
  image output (e.g. `examples/autoencoder`'s reconstruction) would need its
  own function.
- `examples/autoencoder` was not retrofitted -- its `(1, 28, 28)`
  single-channel input conflicts with `ImageFolder._load_image()`'s
  hard-coded 3-channel RGB decode (Section 4.1); adding grayscale support
  with no other consumer was out of scope.
- No CLI command was added (`forge model predict` remains
  classification-specific); a generic image-output CLI path was not asked
  for and has no second consumer yet.
- Scoped, like `predict_artifact()`/`predict_tensor_artifact()`, to models
  whose calling convention is a single `model(x)` forward pass on a batched
  `Tensor`.

## 12. Rejected/deferred alternatives

- **Autoencoder instead of segmentation** -- rejected per Section 4.1
  (would require a new, unjustified grayscale-conversion capability).
- **A generic `predict_image_artifact()` with a `threshold=None` raw
  -passthrough mode** -- considered, rejected as speculative generality with
  no second real consumer to justify it (Section 4.3); can be added later
  if/when a genuinely different image-output workload needs it.
- **Merging into `predict_artifact()`/`predict_tensor_artifact()`** --
  rejected per Section 4.2, following M83's own precedent for the same kind
  of decision.
- A `concat`/skip-connection primitive, a real U-Net, image augmentation,
  batch/HTTP serving, and every other Section 6 scope boundary from the
  brief -- none was needed and none was built.

## 13. What can an external Forge developer do now that they could not before M84?

Before: an external segmentation developer could train through
`Trainer`/`forge.train()`-adjacent APIs and render a predicted mask from an
in-memory query tensor already held by the training process (M76), but a
saved `segmentation_model.forge` carried no preprocessing -- there was no
way to feed it a real image *file* at all, let alone from a fresh process.

After:
```python
prediction = forge.predict_image_artifact("segmentation_model.forge", "new_image.png")
forge.data.save_image(prediction, "prediction.png")
```
one call, exercised by `examples/segmentation/train.py`'s own end-of-run
demo, proven bit-for-bit portable by a genuinely separate OS process
(`examples/segmentation/infer.py`, Section 9), and shown to produce a
pixel-perfect mask (IoU = 1.0, Section 8) on a real, brand-new synthetic
image -- the same "train through the high-level API, save a verified
artifact, move it to a fresh process, get a useful result with no manual
`load_model()`/`predict()` reconstruction" guarantee M82/M83 established for
classification and regression, now also true for a genuinely different
(dense, image-output) task shape.

## 14. Suggested commit message

```
feat: add forge.predict_image_artifact(), one-call image-to-image artifact inference

Composes load_model()/load_preprocessing()/ImageFolder._load_image()/
predict() into a single call that turns a .forge file and one new image
file directly into a thresholded {0, 1} predicted-mask Tensor, ready for
forge.data.save_image() -- the image-to-image counterpart to
predict_artifact() (M82) and predict_tensor_artifact() (M83). Retrofits
examples/segmentation/train.py to save a Normalize preprocessing pipeline
alongside the model and demonstrate the complete file-based workflow on a
brand-new synthetic image, adds a standalone examples/segmentation/infer.py,
and proves the artifact is genuinely portable via a subprocess-based
fresh-process test (bit-for-bit match, IoU = 1.0 on a never-trained-on
image).
```
