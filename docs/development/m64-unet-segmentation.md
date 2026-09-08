# Milestone 64 — U-Net-Style Image Segmentation: Forge's First Dense-Prediction Workload

## 1. Selected Workload

A small, complete image segmentation example: given a `(3, 32, 32)` RGB
image, predict a same-resolution `(1, 32, 32)` per-pixel binary mask
(foreground shape vs. background). `examples/segmentation/` -- dataset
generation, model, training, evaluation, README -- following the exact
`examples/regression`/`examples/autoencoder` file layout and conventions.

The architecture is exactly Milestone 64's brief-suggested minimal shape:

```text
Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> UpsampleNearest2d -> Conv2d -> output
```

with no elaboration beyond it (Section 6 explains why nothing more was
added).

## 2. Why Dense Prediction Is a Meaningful Next Capability

Forge's five prior examples (`mnist`, `char_rnn`, `word_rnn`, `regression`,
`waveform_classification`) each reduce their input down to a single
per-sample output: a class index, a next-token distribution, a regression
scalar. `autoencoder` (M63) was the first to produce a full-resolution
output, but it is a special case -- the target *is* the input, an
unsupervised reconstruction task with no independent ground truth.

No prior Forge example produces a per-pixel prediction *evaluated against an
independent, structured, pixel-level ground truth*. This is a distinct task
shape from both prior families: like classification, it has a real external
target the model has never seen; like the autoencoder, its output is
full-resolution and spatial. Dense prediction (segmentation, depth
estimation, super-resolution, per-pixel regression) is one of the most
common real deep-learning workload categories, and M63's own "Follow-up
Recommendations" section named a segmentation/U-Net-style architecture as
one of two concrete, ready-made consumers of `UpsampleNearest2d` -- not
proposed there as a foregone conclusion, but flagged as the natural next
question to actually investigate, which this milestone's brief then did.

## 3. Existing Forge Capabilities Reused

`Conv2d`, `MaxPool2d`, `ReLU`, `UpsampleNearest2d` (all pre-existing, the
last added in M63), `nn.Sequential` composition, `MSELoss`, `optim.Adam`,
`data.DataLoader`, `data.Dataset`/`TensorDataset`-style custom `Dataset`
(`SegmentationDataset`), `training.Trainer` (`fit`/`evaluate`, checkpoint/
resume, unmodified), `training.Metric` (subclassed, not modified --
`PixelAccuracy`/`IoU`), `serialization.save_model`/`load_model`/
`save_checkpoint`/`load_checkpoint`, the module persistence registry (no new
registration needed -- see Section 5), and the Milestone 19 CLI
(`model inspect`/`checkpoint inspect`). No change to any of these was
required.

## 4. Blockers Discovered

**None.** Per the brief's explicit instruction ("attempt the workload with
the existing... capabilities" before assuming a gap), the architecture was
built and driven directly against the real Forge API -- forward pass,
backward pass, one `Adam` step, and a save/load round trip, exercised on
both CPU and CUDA -- *before* writing any example files:

```python
model = Sequential(
    Conv2d(3, 16, kernel_size=3, padding=1), ReLU(), MaxPool2d(2),
    Conv2d(16, 32, kernel_size=3, padding=1), ReLU(),
    UpsampleNearest2d(2),
    Conv2d(32, 1, kernel_size=3, padding=1),
)
pred = model(x)                 # (4, 1, 32, 32) -- shapes line up
loss = MSELoss()(pred, y)
loss.backward(); optimizer.step()   # backward/step OK, CPU and CUDA
save_model(model, path); load_model(path)  # persistence round trip OK
```

Every step succeeded with no error, no shape mismatch, and no missing
`Backend`/`Tensor`/`Module` method -- on both CPU and CUDA (`is_cuda_available()
== True` on the reference 940MX). This is the direct, executed evidence for
**Outcome A** (Section 16): the workload needed no new production
capability.

The one candidate gap considered and explicitly rejected: true U-Net skip
connections (concatenating an encoder feature map onto a decoder feature map
along the channel axis) would need a `concat`/channel-concatenation
primitive Forge does not have (`grep` over `forge/tensor/tensor.py` and
`forge/backend/base.py` confirms no `cat`/`concat`/`stack` method exists).
This was **not** implemented: Milestone 64's own brief-suggested minimal
architecture has no skip connections, and Section 10's results show the task
is fully, genuinely learnable without them -- so adding a speculative
`concat` primitive with no demonstrated consumer would have violated the
brief's own explicit guardrail ("Do not add features simply because they
are common in modern deep-learning frameworks").

## 5. Implementation

Files added (all in `examples/segmentation/`, all new -- no `forge/`
production file was touched):

- `__init__.py`
- `dataset.py` -- `SegmentationDataset`, `generate_raw()`, `make_datasets()`:
  an in-process synthetic generator (one filled circle or square per image,
  random position/size/color, against a fixed-statistics noisy background,
  with a minimum foreground/background contrast margin enforced by
  rejection sampling).
- `model.py` -- `build_model()`, returning a plain `Sequential` (no custom
  `Module` subclass -- the first Forge image example that needs neither a
  custom `Module` nor a `register_module()` call, since `Conv2d`,
  `MaxPool2d`, `ReLU`, `UpsampleNearest2d`, and `Sequential` were all
  already registered for persistence by prior milestones).
- `metrics.py` -- `PixelAccuracy`, `IoU`: two `forge.training.Metric`
  subclasses (example-local code, not a framework change -- `Metric` is
  designed for exactly this kind of extension).
- `train.py` -- training/evaluation/checkpoint/resume/persistence script,
  plus `majority_class_baseline()`.
- `README.md`.

Files modified:

- `.gitignore` -- added `examples/segmentation/artifacts*/` (same pattern as
  every prior example).

No `forge/` file was added, modified, or needed to be.

## 6. Architecture

```text
(N, 3, 32, 32)
    -> Conv2d(3, 16, k=3, pad=1)  -> ReLU -> MaxPool2d(2)   -> (N, 16, 16, 16)
    -> Conv2d(16, 32, k=3, pad=1) -> ReLU                    -> (N, 32, 16, 16)
    -> UpsampleNearest2d(2)                                  -> (N, 32, 32, 32)
    -> Conv2d(32, 1, k=3, pad=1)                              -> (N, 1, 32, 32)
```

~5.4k trainable parameters. Deliberately the smallest architecture the
brief itself suggests, unmodified: two encoder `Conv2d` layers (the second
without pooling, so the bottleneck stays at half resolution rather than
shrinking twice), one `UpsampleNearest2d` growing back to full resolution,
and one output `Conv2d`. No skip connections (Section 4), no additional
depth or width beyond what the smoke tests (Section 10) showed was already
sufficient to solve the task convincingly.

The final `Conv2d` has no activation -- the `{0, 1}`-valued mask target is
compared via plain `MSELoss`, the exact no-final-activation-regression
convention `examples/autoencoder/model.py::decode()` established (Forge
still has no `Sigmoid` primitive, and this milestone did not need one
either). `train.py`/`metrics.py` threshold the raw output at `0.5` only at
evaluation time.

## 7. Files Changed

**Added**: `examples/segmentation/__init__.py`, `dataset.py`, `model.py`,
`metrics.py`, `train.py`, `README.md`;
`tests/test_segmentation_example_integration.py`,
`tests/test_segmentation_example_cuda_integration.py`;
`docs/development/m64-unet-segmentation.md` (this file).

**Modified**: `.gitignore` (one new ignore block);
`docs/development/progress.md` (this milestone's entry).

**Not touched**: every file under `forge/`.

## 8. Tests

30 new tests, all passing:

- `tests/test_segmentation_example_integration.py` (25, CPU) -- dataset
  generation determinism/cross-seed variation, shape/dtype/value-range
  checks, non-degenerate mask coverage (every sample's foreground is
  nonempty and non-full-image), disjoint train/test streams, model forward
  shape and exact parameter count, `PixelAccuracy`/`IoU` correctness against
  hand-crafted predictions (exact fractional accuracy, raw-output
  thresholding at `0.5`, cross-batch weighted accumulation, perfect/zero/
  partial overlap, the `union == 0` edge case scoring `1.0` not `NaN`), the
  majority-class baseline computed against a manual NumPy recomputation,
  full-pipeline training that beats that baseline on both pixel accuracy
  and IoU with real parameter movement, checkpoint save/resume + resume
  -equivalence-against-continuous-training, model save/load prediction
  consistency, and CLI inspection (`model inspect`/`checkpoint inspect`
  reporting the correct architecture and parameter count).
- `tests/test_segmentation_example_cuda_integration.py` (5, CUDA
  hardware-gated, skips cleanly without CUDA) -- full pipeline trains and
  beats baseline on CUDA, parameter/gradient/Adam-state CUDA residency,
  checkpoint save/resume on CUDA, model persistence on CUDA, CPU/CUDA
  prediction parity after identical training.

Full suite run after this milestone's changes: **1,961 passed**
(1,936 + 25 in the first full run)[^flaky], zero regressions attributable to
this milestone, via `python -m pytest tests/ -q`.

[^flaky]: One unrelated, pre-existing test
(`tests/test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`)
failed once during the full-suite run with a small (320 vs. 288 bytes)
CUDA-allocator-measurement discrepancy, then passed cleanly in isolation
immediately afterward -- the exact same pre-existing flakiness M63's own
report footnoted; no code in `forge/backend/cuda/allocator.py` or
`forge/data/prefetch.py` was touched by this milestone.

## 9. CPU Results

Reference hardware: i5-7200U, `--seed 0`, default hyperparameters
(`--n-train 3000 --n-test 500 --epochs 15 --batch-size 32 --lr 1e-3`):

| Quantity | Value |
|---|---:|
| Trivial baseline (predict all-background): pixel accuracy | 0.7975 |
| Trivial baseline: IoU | 0.0000 |
| Train loss (MSE), epoch 1 -> 15 | 0.04520 -> 0.00619 |
| Final test MSE | 0.00619 |
| Final test pixel accuracy | 0.9980 |
| Final test IoU | 0.9901 |
| Throughput | ~116 train samples/sec (~26s/epoch, ~389s total) |

## 10. CUDA Results

Reference hardware: GeForce 940MX (CC 5.0, driver 582.53, CUDA Toolkit
12.6), identical seed/hyperparameters:

| Quantity | CPU | CUDA |
|---|---:|---:|
| Train loss (MSE), epoch 1 -> 15 | 0.04520 -> 0.00619 | 0.04521 -> 0.00619 |
| Final test MSE | 0.00619 | 0.00617 |
| Final test pixel accuracy | 0.9980 | 0.9981 |
| Final test IoU | 0.9901 | 0.9909 |
| Throughput | ~116 samples/sec (~26s/epoch) | ~689 samples/sec (~4s/epoch, ~65s total) |

CPU and CUDA agree closely on every metric (final MSE within 0.00002, IoU
within 0.0008) -- genuine end-to-end parity, not just kernel-level. CUDA
trained **~5.9x faster** than CPU, consistent with `mnist`'s and
`autoencoder`'s own CUDA advantage on real multi-`Conv2d` workloads (this
model's three `Conv2d` layers give the GPU enough real work per batch to
amortize per-batch launch/sync overhead).

## 11. Baseline Comparison

The trivial baseline is "always predict no foreground" (an all-zero mask) --
the segmentation-task analog of `regression`'s predict-the-mean and
`autoencoder`'s predict-the-mean-image baselines. Because the synthetic
shapes cover a small fraction of each `32x32` image, this baseline already
scores a moderately high pixel accuracy (0.7975) purely from class
imbalance -- exactly why pixel accuracy alone would be a misleading success
metric here, and why IoU (which the trivial baseline scores at **exactly
0.0**, by definition: an empty predicted region can never intersect a
nonempty target) is reported alongside it. The trained model reaches
**0.9901 IoU** (CPU) / **0.9909 IoU** (CUDA) -- unambiguous evidence of
genuine, near-pixel-perfect dense prediction, not an artifact of the
accuracy metric's class imbalance.

## 12. Serialization/Checkpoint Validation

Both the CPU and CUDA `train.py` runs performed and printed a live
save/load prediction-parity check (`Verified: reloaded model reproduces the
pre-save prediction.`). Checkpoint save/resume, resume equivalence against
continuous training, and model persistence are additionally covered
permanently by the test suite (Section 8) on both CPU and CUDA, including
explicit CUDA-residency assertions for parameters, gradients, and Adam
optimizer state after a checkpoint reload.

## 13. Limitations

- The model has no skip connections, so it is a shrink-then-grow
  encoder/decoder in the U-Net family, not a literal U-Net -- see Section 4
  for why this was a deliberate, evidence-based choice, not an oversight.
- The synthetic dataset uses a single foreground shape per image against a
  fixed-statistics background -- it demonstrates the dense-prediction task
  shape and Forge's ability to train it, not realistic natural-image
  segmentation (multiple overlapping objects, class-label masks, real
  photographic texture). This mirrors `regression`'s and
  `waveform_classification`'s own synthetic-data precedent: the point is to
  validate the framework capability and task shape, not to produce a
  state-of-the-art segmentation benchmark.
- The output layer regresses directly to a `{0, 1}` target with no bounding
  activation (no `Sigmoid` primitive exists in Forge) -- consistent with
  `autoencoder`'s established convention, and sufficient given the reported
  results, but a probabilistic (cross-entropy-style) formulation was not
  attempted.
- Only binary (single-class) segmentation is demonstrated; multi-class
  per-pixel classification would need `CrossEntropyLoss` extended to a
  `(N, C, H, W)` logits shape (currently `(N, C)`-only) -- not attempted
  here since the binary-mask task already fully demonstrates dense
  prediction with zero framework changes, and no concrete multi-class
  consumer motivated the extension.

## 14. Rejected Additions

Per the brief's explicit guardrails, the following were considered and
**not** added, since the workload never demonstrated a need for any of
them:

- **Channel concatenation / `concat`** (true U-Net skip connections) --
  Section 4/13.
- **`ConvTranspose2d`** -- `UpsampleNearest2d` + `Conv2d` already solves the
  "grow a spatial map" problem (M63), and remains the deliberately preferred
  choice (avoids checkerboard artifacts); nothing about this workload
  motivated a second growth mechanism.
- **A 4D-logits `CrossEntropyLoss`** (multi-class per-pixel classification)
  -- the binary-mask + `MSELoss` formulation fully demonstrates dense
  prediction; no concrete multi-class consumer exists yet.
- **`Sigmoid`** -- the no-final-activation regression convention (already
  established by `autoencoder`) was sufficient; the model reaches 0.99 IoU
  without it.
- **A deeper/wider network, attention, additional optimizer features** --
  the brief-suggested minimal architecture already solves the task to
  near-perfect IoU; adding capacity would not have demonstrated anything
  the results don't already show.

## 15. Practical Impact on Forge

Forge can now express and train dense, per-pixel-prediction models --
segmentation-shaped architectures -- using the same `Trainer`/`Adam`/
checkpoint/persistence pipeline every other workload already uses, with
**zero new framework code**. This is direct, executed confirmation (not
speculation) that M63's `UpsampleNearest2d` primitive generalizes beyond the
one autoencoder that motivated it, and that Forge's serialization registry
already covers every layer a small encoder/decoder CNN needs -- this is the
first Forge image example whose model needs no `register_module()` call at
all. It also gives Forge a second, structurally different real consumer of
the "shrink then grow" encoder/decoder pattern (`autoencoder`'s bottleneck
reconstruction vs. this example's same-resolution dense prediction),
reinforcing that the M63 primitive was a genuinely reusable, non-speculative
addition rather than one-off machinery.

## 16. Next-Step Recommendation

No specific next capability is proposed as a foregone conclusion, per the
same discipline this milestone's own brief asked to be applied (and that
M62/M63 asked of their successors in turn): inspect the current
architecture and examples directly, identify whether a concrete, credible
gap with a real consumer exists, and only build if one is found. Two
candidates this milestone's own investigation surfaced but explicitly did
not pursue (Sections 4/13/14) -- channel concatenation for true U-Net skip
connections, and a `(N, C, H, W)`-shaped `CrossEntropyLoss` for multi-class
per-pixel segmentation -- are reasonable future directions *if* a concrete
workload demonstrates a real need for either, but neither is being proposed
as a foregone conclusion here.

## Final Decision

**Outcome A: workload completed with existing framework, no new production
capability required.**

Direct execution against the real Forge API (Section 4) confirmed the
entire segmentation workload -- model, training, evaluation, checkpointing,
persistence, on both CPU and CUDA -- is expressible with capabilities that
already existed before this milestone, with the single newest dependency
being M63's `UpsampleNearest2d`. No `forge/` production file was added,
modified, or needed to be.

## Suggested Commit Message

```
feat: add examples/segmentation (U-Net-style dense prediction, zero new framework capability)

Milestone 64: a small encoder/decoder segmentation example (synthetic
shape-on-background dataset, Conv2d/MaxPool2d/UpsampleNearest2d/Conv2d
model, PixelAccuracy/IoU metrics) demonstrating Forge's first genuine
dense-prediction workload. Direct execution against the existing Forge
API confirmed no new production capability was required -- every layer
this model needs (Conv2d, MaxPool2d, ReLU, UpsampleNearest2d, Sequential)
was already implemented and already registered for persistence.

Reaches 0.9901 IoU (CPU) / 0.9909 IoU (CUDA) against a 0.0 trivial
baseline over 15 epochs on a 3,000-image synthetic dataset; CUDA trains
~5.9x faster than CPU. 30 new tests (25 CPU + 5 CUDA), full suite at
1,961 passed with zero regressions.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012T7MMwNUz2iBZLptDzJvLu
```
