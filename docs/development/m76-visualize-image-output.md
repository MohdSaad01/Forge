# M76 — `forge.data.save_image()`: rendering image-shaped model output

## 1. Executive summary

M76 investigated Forge's full developer lifecycle (dataset → preprocessing →
model → training → evaluation → persistence → inference → useful
application output) to find the next highest-value reusable workflow gap.
The brief's own candidate area 1 (a generic evaluation abstraction) turned
out to already be solved: `Trainer.evaluate()` + `forge.training.Metric`
form a genuinely reusable, already-extended (by
`examples/segmentation/metrics.py`) evaluation mechanism used by 7 of 9
examples — building another evaluation framework would have been pure
symmetry, not evidence.

The real gap was found in candidate area 5 (inference/application
workflow): `predict()` (M68) and `interpret_classification()` (M72) close
the "tensor output → useful result" gap for classification, but two real
consumers — `examples/autoencoder/train.py` (image reconstruction) and
`examples/segmentation/train.py` (predicted binary mask) — produce
image-shaped Tensor outputs that were **never rendered anywhere a person
could look at them**. Both scripts only ever printed a scalar metric (MSE,
pixel_accuracy, IoU); the actual reconstructed image or predicted mask
existed only as an in-memory NumPy array.

M76 added `forge.data.save_image(tensor, path)` — a minimal, Pillow-backed
(already a Forge dependency) write-side counterpart to `ImageFolder`'s
existing decode step — and retrofitted both real consumers to write actual
PNG files. This closes the pipeline's final "useful application output"
step for every current image-shaped (non-classification) Forge workload,
the same way M72 closed it for classification.

## 2. Investigation performed

Read and/or executed, directly:

- `forge/data/` (`dataset.py`, `dataloader.py`, `image_folder.py`,
  `transforms.py`, `prefetch.py`, `__init__.py`)
- `forge/training/` (`trainer.py`, `metrics.py`, `inference.py`,
  `session.py`, `__init__.py`)
- `forge/serialization/` (`model.py`, `checkpoint.py`, `registry.py`)
- `forge/cli/` (`main.py`, `model.py`, `checkpoint.py`, `_archive_info.py`)
- Every `examples/*/train.py`, plus `examples/image_folder_classification/
  infer.py` and `examples/*/model.py`/`dataset.py`/`metrics.py` where a
  workload's output shape needed checking
- `docs/product/vision.md`, `use-cases.md`, `requirements.md`, `scope.md`
- `docs/architecture/data-pipeline.md`, `training-engine.md`,
  `persistence.md`
- The persistent cross-session memory record of M49–M75 (which milestones
  already closed which gaps, and what each one explicitly deferred)
- Ran `pytest --collect-only` and the full suite before making changes to
  establish the pre-M76 baseline (2,234 collected, 2,233 passed, 1 known
  flake)

## 3. Existing workflow analysis

Grepping every `examples/*/train.py` for evaluation/reporting patterns
found:

- `Trainer.evaluate()` (`forge/training/trainer.py`) already provides
  eval-mode + `no_grad()` + mode-restoration + device-validated batch
  iteration + loss accumulation + `Metric` aggregation, returning a typed
  `EvaluationResult(loss, metrics, samples, duration, device)`.
- `forge.training.metrics.Metric` is an already-designed extension point:
  `Accuracy`, `MeanSquaredError`, `MeanAbsoluteError` ship built-in, and
  `examples/segmentation/metrics.py` already demonstrates a real consumer
  extending it locally (`PixelAccuracy`, `IoU`) with **zero** framework
  changes — exactly the extension path the `Metric` base class's own
  docstring describes.
- Every classification/regression/dense-prediction example already calls
  `trainer.evaluate(test_loader)` and prints `final_eval.loss`/
  `final_eval.metrics[...]` — there is no duplicated *evaluation loop* left
  to extract.
- `forge model inspect`/`forge model convert`/`forge model predict`
  (`forge/cli/model.py`) already cover model-artifact introspection, device
  conversion, and single-image classification inference end-to-end.
- `start_training_session()`/`TrainingSession` (M73) already eliminated the
  duplicated resume/fresh-Trainer branch across all 7 checkpoint-capable
  examples.
- `forge.data.sequential_split` (M74) already eliminated duplicated
  array-slicing in `regression`/`waveform_classification`.
- `forge.training.generate_sequence()` (M75) already eliminated duplicated
  autoregressive sampling in `char_rnn`/`word_rnn`.

This left the lower/middle of the pipeline (dataset → preprocessing →
model → train → evaluate) with no evidenced gap. The investigation moved to
the pipeline's final stage: does every workload actually deliver a *useful
application output*, not just a printed number?

## 4. Concrete gap identified

Grepping every example for image-writing code (`PNG`, `.png`, `imsave`,
`Image.fromarray`, `save_image`) found **zero matches under `examples/`**.
No example — including the two whose entire output *is* an image
(`autoencoder`'s reconstruction, `segmentation`'s predicted mask) — had ever
written a viewable file. `examples/mnist/train.py`'s own
`reloaded = load_model(...)` round trip, duplicated identically across 9
examples, only ever asserts `np.allclose(pre_save_pred, post_load_pred)` —
it proves persistence correctness, never surfaces the *result* to a human.

For `autoencoder`: `model.py::decode()`'s docstring documents "no final
activation" — the reconstruction is an unbounded `(1, 1, 28, 28)` Tensor
compared via `MSELoss` against `[0, 1]`-scaled pixels. `train.py`'s `main()`
reports only `test_mse` and a `latent_nearest_neighbor_label_agreement`
percentage — the actual reconstructed digit image is discarded after the
persistence-verification `predict()` call.

For `segmentation`: `model.py`'s final `Conv2d` also has no activation; the
mask is thresholded at `0.5` purely inside `metrics.py`'s `PixelAccuracy`/
`IoU` and `train.py`'s own baseline computation — never written out.
`train.py`'s `main()` reports only `pixel_accuracy`/`iou` numbers against a
trivial "predict all-background" baseline.

This directly answers the brief's own investigation area 5 question: "For
image classification, M72 already added class metadata + interpretation +
CLI predict. Determine whether other important workloads expose a similar
real gap." They do — dense/image-shaped output has no analogous
"tensor → useful result" step at all.

## 5. Evidence for the gap

- Direct grep (`PNG|\.png|imsave|Image.fromarray|save_image`) across
  `examples/` and `forge/` found image writing only inside
  `examples/image_folder_classification/generate_dataset.py` (synthetic
  *input* generation, unrelated) and `forge/data/image_folder.py`/
  `transforms.py` (both *decode*-only, via `Image.open`).
- Direct reading of `examples/autoencoder/train.py::main()` and
  `examples/segmentation/train.py::main()` end to end (lines 150–230 and
  95–170 respectively) confirmed no image-writing code exists anywhere in
  either script's control flow.
- Both examples already compute a query prediction (`pre_save_pred`) as
  part of the pre-existing persistence-round-trip check — the *data* needed
  to render an image already exists in memory at exactly the point where
  nothing currently uses it for display.
- `pyproject.toml` already lists `Pillow>=10.0` as a dependency (confirmed
  by `grep`), and `forge/data/transforms.py::Resize` already demonstrates
  the exact `(C,H,W)` Tensor ↔ Pillow `Image` conversion pattern this
  capability needs, in the opposite direction — no new dependency, no novel
  technique, a direct application of an already-approved library and an
  already-established conversion idiom.

## 6. Existing consumers

Two real, already-existing example workloads, confirmed by direct
execution (not merely by reading their code):

1. `examples/autoencoder/train.py` — MNIST reconstruction, `~111.5k`
   parameters. Run end-to-end on CPU (`--epochs 1`, real MNIST data already
   present locally): produced `reconstruction_input.png` (mode `L`,
   `28x28`, pixel range `0–255`, mean `23.5`) and `reconstruction_output.png`
   (mode `L`, `28x28`, pixel range `0–231`, mean `22.9`) — closely matching
   statistics, consistent with the run's own reported "67.2% MSE reduction
   over the mean-image baseline."
2. `examples/segmentation/train.py` — synthetic circle/square segmentation,
   `~5.4k` parameters. Run end-to-end on CPU (`--n-train 40 --n-test 20
   --epochs 1`): produced `segmentation_input.png` (mode `RGB`, `32x32`,
   real pixel content), `segmentation_predicted_mask.png` (mode `L`, all
   `0` after 1 epoch — consistent with the run's own reported `iou=0.0000`,
   i.e. an undertrained model predicting all-background, exactly matching
   the printed metric rather than contradicting it), and
   `segmentation_ground_truth_mask.png` (mode `L`, values `{0, 255}`,
   confirming the mask-rendering path itself is correct even though this
   particular 1-epoch run hadn't learned to predict foreground yet).

## 7. Alternatives considered

- **A dedicated `RegressionPrediction`/interpretation helper for
  `examples/regression`** (M72's own suggested next candidate). Rejected
  after direct inspection: `examples/regression`'s `predict()` output is
  already the useful numeric value (e.g. a predicted price) — there is no
  index-to-name mapping step analogous to classification's `argmax` → label
  name. Building this would be symmetry-only, with no evidenced gap.
- **A standalone `forge.evaluate()` free function** (mirroring `predict()`,
  removing `Trainer`'s required-but-unused `Optimizer` for pure
  post-training evaluation). Investigated: no example currently reloads a
  saved model purely to evaluate it (every evaluation happens inside the
  same `train.py` process that already has a real `Optimizer` in scope).
  The friction is real in principle but not evidenced by any existing
  duplicated code — would have been anticipatory, not gap-driven.
- **Extracting the duplicated "load fresh, `predict()`, `assert
  np.allclose(...)`" persistence-round-trip block** (found identically
  duplicated, ~4 lines, across all 9 saving examples). Considered but
  rejected as the primary M76 capability: it is a self-verification/test
  convenience internal to the example scripts, not a capability that turns
  a trained model into a *user-facing* application result — closer to the
  brief's own explicitly-named "trivial, cosmetic" category (per M75's
  precedent for rejecting device-selection boilerplate dedup). It remains a
  legitimate small follow-up (see §21).
- **A generic evaluation/metric framework** (Outcome A, generalized).
  Rejected: `Trainer.evaluate()` + `Metric` already exist, are already
  reusable, and are already extended by a real consumer
  (`examples/segmentation/metrics.py`) with zero framework changes. No
  additional framework machinery is missing.
- **Extending `forge model predict` (CLI) to non-image workloads** (a
  possible reading of "CLI inference improvements"). Rejected: `cmd_predict`
  is deliberately scoped to a single image file because Forge has no
  generic cross-workload input-file convention (images, tabular rows, and
  raw sequences all shape differently) — this is documented as a deliberate
  scope boundary in `forge/cli/model.py`'s own module docstring, not an
  oversight to close.
- **A generic "image grid"/side-by-side comparison compositor.** Rejected
  as a premature generalization: neither real consumer's workflow currently
  needs more than "save the input and the output as two separate files";
  composing them into one comparison image is a caller-level concern with
  no evidenced multi-consumer need.

## 8. Rejected alternatives (summary)

See §7 above — every alternative considered either (a) had no real
consumer/evidenced duplication, (b) was already solved by existing Forge
capability, or (c) would have generalized beyond what any current consumer
needs.

## 9. Selected direction

**Outcome B** (a concrete workflow capability, not a generalized
evaluation framework): `forge.data.save_image(tensor, path)`.

## 10. Why the selected direction clears the evidence bar

- Two real, already-existing, already-executed consumers (`autoencoder`,
  `segmentation`) produce image-shaped Tensor output today, confirmed by
  direct execution, not hypothesized.
- Neither consumer has any current way to render that output — confirmed
  by a direct grep finding zero image-writing code anywhere in `examples/`.
- The brief's own pipeline diagram explicitly names "useful application
  output" as the terminal stage, and its own investigation area 5
  explicitly asks whether workloads beyond classification have a similar
  real gap to the one M72 closed — this is a direct, on-target answer to
  that named question, not an invented parallel feature.
- The implementation requires no new dependency (Pillow is already used by
  `ImageFolder`/`Resize`) and no novel technique (the CHW↔Pillow conversion
  idiom already exists in `Resize`, applied here in the write direction).
- The capability is minimal: one function, no new class hierarchy, no
  configuration surface, no generalized "visualization framework."

## 11. Exact implementation

`forge/data/image_folder.py` — added `save_image(tensor, path) -> None`
alongside the existing `ImageFolder._load_image` decoder:

```python
def save_image(tensor: Tensor, path: "str | Path") -> None:
    if not isinstance(tensor, Tensor):
        raise DataError(...)
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
        raise DataError(...)

    target = Path(path)
    if not target.parent.is_dir():
        raise DataError(...)

    array = tensor.to("cpu").numpy().astype(np.float64)
    clipped = np.clip(array, 0.0, 1.0)
    scaled = np.round(clipped * 255.0).astype(np.uint8)
    hwc = np.ascontiguousarray(scaled.transpose(1, 2, 0))

    if tensor.shape[0] == 1:
        image = Image.fromarray(hwc[:, :, 0], mode="L")
    else:
        image = Image.fromarray(hwc, mode="RGB")
    image.save(target)
```

Exported via `forge/data/__init__.py` as `forge.data.save_image` (not a
top-level `forge.save_image` re-export — matches `ImageFolder`/`Resize`'s
own placement, since `forge.data` already owns Pillow-based image I/O and
this is not a training-orchestration function like `predict()`).

Retrofitted consumers:

- `examples/autoencoder/train.py`: after the existing persistence-round
  -trip check (`pre_save_pred`), writes `reconstruction_input.png`
  (`query_x`) and `reconstruction_output.png` (`pre_save_pred`, wrapped
  back into a `Tensor`) into `output_dir`.
- `examples/segmentation/train.py`: captures `query_mask` (previously
  discarded via `_`) alongside the existing `query_x`/`pre_save_pred`, then
  writes `segmentation_input.png` (`query_x`), `segmentation_predicted_
  mask.png` (`pre_save_pred` thresholded at the script's existing
  `_THRESHOLD = 0.5`), and `segmentation_ground_truth_mask.png`
  (`query_mask`) into `output_dir`.

No new query sample and no new forward/evaluation pass were introduced —
both retrofits reuse the exact Tensor values the pre-existing persistence
check already computed.

## 12. Files changed

- `forge/data/image_folder.py` — added `save_image()`.
- `forge/data/__init__.py` — exported `save_image`.
- `examples/autoencoder/train.py` — import + 2-line image-writing addition.
- `examples/segmentation/train.py` — import + capture `query_mask` +
  3-line image-writing addition.
- `examples/autoencoder/README.md` — new "Viewing a reconstruction" section.
- `examples/segmentation/README.md` — new "Viewing a predicted mask"
  section.
- `docs/architecture/data-pipeline.md` — documented `save_image` under
  **Built-in sources**.
- `docs/development/progress.md` — appended the M76 entry.
- `tests/test_save_image.py` — 11 new tests (new file).
- `tests/test_save_image_cuda.py` — 1 new test (new file).
- `docs/development/m76-visualize-image-output.md` — this report (new
  file).

No `forge/tensor`, `forge/autograd`, `forge/backend`, `forge/nn`,
`forge/optim`, `forge/serialization`, or `forge/training` file was touched.

## 13. Public API changes

New: `forge.data.save_image(tensor: Tensor, path: str | Path) -> None`.

No existing public API changed signature or behavior. No re-export was
added at the top-level `forge` namespace (consistent with `ImageFolder`/
`Resize`'s existing placement).

## 14. Architecture impact

None. `save_image` is a plain function using `Tensor.to("cpu")`/`Tensor.
numpy()` (both pre-existing, public) and Pillow (already a dependency,
already used the same way by `Resize`/`ImageFolder._load_image` in the
opposite direction). No Tensor, autograd, backend, module, or persistence
code was touched, per the brief's architecture constraints.

## 15. Test coverage

`tests/test_save_image.py` (11 tests, all passing):

- Normal operation: RGB round-trip (verified via a real Pillow re-read of
  exact pixel values), grayscale round-trip, `Path` and `str` path
  acceptance.
- Clipping: values above `1.0` clip to `255`; values below `0.0` clip to
  `0` (covers unbounded model output, the real `autoencoder`/`segmentation`
  case).
- Invalid inputs: non-Tensor input, wrong `ndim` (2-D), unsupported channel
  count (`C=2`), missing parent directory (with the path named in the error
  message).
- Shape edge case: a single-pixel `(3, 1, 1)` image.

`tests/test_save_image_cuda.py` (1 test, hardware-verified on the 940MX):

- A CUDA-resident Tensor is transferred to CPU and written correctly
  (device handling — the one CUDA-relevant behavior this host-only
  function has).

Not applicable to this capability (and correctly omitted): empty-input
handling (no meaningful "empty image" case beyond the 1x1 edge case
already covered), `eval()`/training-mode behavior (this is not a `Module`
forward pass), persistence (this writes a plain image file, not a `.forge`
archive), backward compatibility (new function, nothing to preserve).

## 16. End-to-end verification

Both real consumers were run to completion, not just unit-tested:

```bash
python -m examples.segmentation.train --n-train 40 --n-test 20 --epochs 1 --output-dir <tmp>
python -m examples.autoencoder.train --epochs 1 --output-dir <tmp>
```

Both completed successfully and printed the new `Saved ... -> ...` lines.
The written files were then inspected directly with Pillow/NumPy (mode,
size, min/max/mean pixel values) — not merely asserted to exist — and their
statistics were cross-checked against each run's own printed metrics (see
§6). This is real data → real model → real framework API → a real,
independently-inspected result, per the brief's End-to-End Requirement.

## 17. CPU/CUDA verification where applicable

`save_image` itself is a host-only, non-differentiable reporting operation
(no CUDA kernel is possible or needed, the same category as `forge.
training.metrics._as_numpy`). Its one CUDA-relevant behavior — transferring
a CUDA-resident Tensor before writing — is hardware-verified on the
project's real 940MX in `tests/test_save_image_cuda.py`, which passed.
Both example scripts' own `--device cuda` paths were not re-run end-to-end
this milestone (unchanged from before M76 — `save_image` is only reached
after `predict()`, which already returns a CPU Tensor); the CUDA-tensor
path is independently covered by the dedicated CUDA test instead.

## 18. Persistence implications

None. `save_image` writes a plain image file, entirely outside Forge's
`.forge` model/checkpoint archive format — no interaction with
`forge/serialization/` in either direction.

## 19. Before/after developer workflow

**Before** (`examples/autoencoder/train.py`, `examples/segmentation/
train.py`):

```python
query_x, _ = test_ds[0]
query_x = query_x.to(args.device).reshape(1, 1, 28, 28)
pre_save_pred = predict(model, query_x).numpy()
reloaded = load_model(str(model_path), device=args.device)
post_load_pred = predict(reloaded, query_x).numpy()
assert np.allclose(pre_save_pred, post_load_pred, atol=1e-5)
print("Verified: reloaded model reproduces the pre-save reconstruction.")
# ... the actual reconstructed image is now discarded; nothing to look at.
```

**After**:

```python
# (the same round-trip check runs unchanged, then:)
save_image(query_x.reshape(1, 28, 28), output_dir / "reconstruction_input.png")
save_image(forge.Tensor(pre_save_pred.reshape(1, 28, 28)), output_dir / "reconstruction_output.png")
```

A developer running either example now gets real files they can open and
look at, not only a scalar loss/metric number. This is a concrete, visible
improvement, not a cosmetic line-count reduction — it did not exist as
capability before this milestone.

## 20. Limitations

- `save_image` handles exactly one `(C, H, W)` image per call; a caller
  wanting a batch or a comparison grid composes multiple calls themselves
  (no grid/montage compositor was built — no evidenced multi-consumer
  need, see §7).
- The `[0, 1]`-scaled-input convention is a documented assumption, not an
  auto-detected one: a Tensor already in Pillow's native `[0, 255]`
  convention (e.g. straight from `ImageFolder`, pre-`Resize`) would be
  clipped incorrectly if passed to `save_image` directly — this is
  intentional (matches the two real consumers' actual convention) but is a
  real constraint a future third consumer would need to respect or work
  around.
- No alpha-channel/RGBA support (mirrors `ImageFolder`'s own RGB-only
  convention).

## 21. Deferred work

- The ~4-line "load fresh, `predict()`, `assert np.allclose(...)`"
  persistence-round-trip block remains duplicated across all 9 saving
  examples (see §7) — a legitimate small extraction candidate for a future
  milestone if a tenth consumer or a concrete correctness incident
  motivates it, but explicitly not built now (see §7's rejection reasoning).
- A standalone `forge.evaluate()` free function (Trainer-evaluation without
  a required `Optimizer`) remains a plausible but unevidenced future
  capability (see §7) — build only once a real consumer needs to evaluate a
  saved model in a process that never constructs an `Optimizer`.
- A `RegressionPrediction`/interpretation helper for `examples/regression`
  remains explicitly not justified (see §7) — M72's own suggested next
  candidate is now answered: no.

## 22. Relationship to the long-term Forge vision

`docs/product/vision.md`'s core workflow ends in "evaluation → persistence
→ inference," and this milestone's own brief extends that explicitly to
"useful application output." M68 (predict), M72 (classification
interpretation + CLI), and M75 (sequence generation) each closed this final
step for one workload family. M76 closes it for the two remaining
image-shaped-output workloads Forge currently has, without inventing a
generalized "visualization" subsystem — directly advancing "a developer can
use Forge to build and train real small models, evaluate them, persist
them, reload them, perform inference" (`vision.md`'s own **Success**
section) into something whose result a developer can actually see.

## 23. Practical developer impact

Anyone running `examples/autoencoder` or `examples/segmentation` today gets
real PNG files instead of only console-printed numbers — the difference
between "the model achieves 67.2% MSE reduction" and being able to open
`reconstruction_output.png` and see the digit the model actually
reconstructed. This is a directly visible, immediately useful change to
both examples' existing developer experience, achieved with a ~60-line
framework addition and no architectural change.

## 24. Follow-up triggers

- If a third image-shaped-output workload is added (e.g. a GAN, a
  denoiser, a super-resolution example) and needs multi-image comparison
  output, that is the point to revisit a grid/montage compositor — not
  before.
- If any example needs to reload a saved model purely to evaluate it
  (no `Optimizer` in scope), that is the point to revisit a standalone
  `forge.evaluate()` free function.
- If a real consumer needs to `save_image` a Tensor already in `[0, 255]`
  scale (e.g. round-tripping `ImageFolder` input directly), that is the
  point to revisit whether `save_image` needs an explicit value-range
  parameter rather than its current single documented convention.

## 25. Suggested commit message

```
feat: add forge.data.save_image(), rendering autoencoder/segmentation output as real PNGs

Trainer.evaluate()/Metric already provide a reusable, already-extended
evaluation abstraction (examples/segmentation/metrics.py's PixelAccuracy/
IoU) -- no evaluation-framework gap exists. The real gap: predict()/
interpret_classification() turn a classification model's output into a
usable label, but autoencoder's reconstruction and segmentation's predicted
mask are image-shaped Tensor outputs with no equivalent step -- both
examples only ever printed a scalar metric. Added save_image(tensor, path)
(Pillow-backed, mirroring Resize's existing CHW<->Pillow conversion in the
write direction) and retrofitted both real consumers to write actual PNGs
at their existing persistence-round-trip query sample.
```
