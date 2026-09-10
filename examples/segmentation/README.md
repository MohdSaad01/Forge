# Forge Image Segmentation Example (Milestone 64)

An end-to-end demonstration of Forge's newest task shape -- **dense, per
-pixel prediction** -- brought to the same standard as `examples/mnist`,
`examples/char_rnn`/`word_rnn`, `examples/regression`,
`examples/waveform_classification`, and `examples/autoencoder`:

```text
SegmentationDataset -> DataLoader -> Trainer -> Sequential
    (Conv2d/ReLU/MaxPool2d encoder -> UpsampleNearest2d/Conv2d decoder)
    -> MSELoss -> Adam
```

Every previous example reduces an image down to a single vector output (a
class index, a regression scalar, a latent code) or reconstructs the whole
image back (`autoencoder`). This example produces a same-resolution binary
mask -- one prediction *per pixel*, not one prediction per image -- the
task shape behind real segmentation, super-resolution, and depth-estimation
workloads.

## Files

- `dataset.py` -- `SegmentationDataset`, a synthetic shape-on-background
  generator, plus `make_datasets()`.
- `model.py` -- `build_model()`, the encoder/decoder architecture.
- `metrics.py` -- `PixelAccuracy`/`IoU`, example-local `forge.training.Metric`
  subclasses for dense-prediction evaluation.
- `train.py` -- the runnable example: training, evaluation, checkpointing,
  resume, and model persistence.

## The dataset

Unlike `mnist`/`autoencoder` (a real downloaded corpus), this example
generates its data entirely in-process from a fixed seed -- the same
`char_rnn`/`word_rnn`/`regression`/`waveform_classification` convention, so
no download and no third-party dependency beyond NumPy.

Each `32x32` RGB image has a fixed-statistics noisy background and one
randomly placed, randomly sized (radius 4-10px), randomly colored foreground
shape (a filled circle or square). The paired target is a `(1, 32, 32)`
binary mask marking exactly which pixels the shape covers. The foreground
color is rejected and redrawn until it differs from the background by a
minimum contrast margin, so every image has a genuine, locally detectable
edge -- learnable by a small local-receptive-field CNN -- while shape type,
size, position, and exact color all vary every sample, so the task cannot be
solved by memorizing a fixed mask or a single global color threshold.

```bash
python -m examples.segmentation.train --epochs 15 --device cpu
```

generates 3,000 training / 500 test images by default (`--n-train`/
`--n-test`), entirely in memory, in well under a second.

## Model

```text
(N, 3, 32, 32)
    -> Conv2d(3, 16, k=3, pad=1)  -> ReLU -> MaxPool2d(2)   -> (N, 16, 16, 16)
    -> Conv2d(16, 32, k=3, pad=1) -> ReLU                    -> (N, 32, 16, 16)
    -> UpsampleNearest2d(2)                                  -> (N, 32, 32, 32)
    -> Conv2d(32, 1, k=3, pad=1)                              -> (N, 1, 32, 32)
```

~5.4k trainable parameters -- exactly Milestone 64's brief-suggested minimal
architecture (`Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU ->
UpsampleNearest2d -> Conv2d -> output`), built as a plain `forge.nn.Sequential`
(the same `build_model()`-returns-`Sequential` convention as `examples/
regression/model.py` and `examples/mnist/model.py` -- no custom `Module`
subclass, no `register_module()` call needed). This is a shrink-then-grow
encoder/decoder in the U-Net family, but **not** a literal U-Net: it has no
skip connections, since that needs a channel-concatenation primitive Forge
does not have (see **Framework impact** below) and this milestone's own
implementation-first investigation found no consumer that required one --
the task is fully, genuinely learnable without it (see **Results**).

The final `Conv2d` has no activation: the mask target is `{0, 1}`-valued and
compared via `MSELoss`, the same no-final-activation regression convention
`examples/autoencoder/model.py::decode()` established (Forge has no
`Sigmoid` primitive, and none was needed here). `train.py`/`metrics.py`
threshold the raw output at `0.5` only at evaluation time, never inside the
model or the loss.

## Framework impact: **zero new production capability required**

Milestone 64's brief asked to attempt the workload against Forge's existing
`Tensor`/`Module`/loss/optimizer/`DataLoader`/`Trainer`/serialization/
`Conv2d`/`UpsampleNearest2d` capabilities *before* assuming a gap exists.
Direct execution against the real API -- forward pass, backward pass, an
`Adam` step, and a save/load round trip, all exercised on both CPU and CUDA
-- confirmed the entire workload is expressible today:

- `Conv2d`/`MaxPool2d`/`ReLU`/`UpsampleNearest2d` (Milestone 63) compose
  into the encoder/decoder with zero new `Tensor`/`Backend` methods.
- `MSELoss` against a `{0, 1}` per-pixel target needs no new loss (matching
  `examples/autoencoder`'s precedent of comparing an unbounded model output
  against a `[0, 1]`-ranged regression target).
- `Trainer.fit()`/`evaluate()` needed no changes to train a per-pixel,
  same-resolution target -- confirming (again, after `autoencoder`) that
  `Trainer` makes no assumption about what its loss target's shape means,
  only that batches are `(features, target)` pairs.
- `Conv2d`, `MaxPool2d`, `ReLU`, `UpsampleNearest2d`, and `Sequential` are
  *already* registered in `forge/serialization/registry.py` (Milestones
  15/53/62/63), so this example's `build_model()` needs no
  `register_module()` call at all -- the first Forge image example that
  doesn't.
- `PixelAccuracy`/`IoU` (`metrics.py`) are ordinary `forge.training.Metric`
  subclasses, the extension point that base class already documents itself
  for -- no framework change to add a dense-prediction metric.

The one capability a true U-Net would need beyond this -- concatenating an
encoder feature map onto a decoder feature map along the channel axis (a
skip connection) -- has no Forge primitive today. This milestone
deliberately did not add one: Milestone 64's brief's own suggested minimal
architecture has no skip connections, and this workload's results (below)
show the task is fully solvable without them. Adding a `concat` primitive
speculatively, with no consumer that needs it, would have been exactly the
kind of scope creep the brief warns against. See
`docs/development/m64-unet-segmentation.md` for the full investigation.

## CPU training

```bash
python -m examples.segmentation.train --epochs 15 --device cpu
```

Trains with Adam (`lr=1e-3`) over 3,000 synthetic training images (500 test
images for validation), reports per-epoch training/validation loss, pixel
accuracy, and IoU, then saves a checkpoint and a model file under
`--output-dir` (default `examples/segmentation/artifacts`).

### Results (reference: this repository's CPU, i5-7200U)

Measured directly on this run (`--seed 0`, default hyperparameters,
`--epochs 15`):

| Quantity | Value |
|---|---:|
| Trivial baseline (predict all-background): pixel accuracy | 0.7975 |
| Trivial baseline: IoU | 0.0000 |
| Train loss (MSE), epoch 1 -> 15 | 0.04520 -> 0.00619 |
| Final test MSE | 0.00619 |
| Final test pixel accuracy | 0.9980 |
| Final test IoU | 0.9901 |
| Pixel accuracy improvement over baseline | +0.2005 |
| IoU improvement over baseline | +0.9901 |
| Throughput | ~116 train samples/sec (~26s/epoch) |

The trivial baseline (predicting "no foreground" everywhere) already scores
a fairly high pixel accuracy, simply because background pixels vastly
outnumber foreground pixels by construction -- exactly why IoU is reported
alongside pixel accuracy: the baseline's IoU is `0.0` by definition (no
predicted foreground ever intersects the target), so any positive IoU is
direct evidence of genuine dense prediction, not an accuracy-metric
artifact. `0.9901` final IoU means the model's predicted foreground region
overlaps the true shape almost pixel-for-pixel.

## CUDA training

```bash
python -m examples.segmentation.train --epochs 15 --device cuda
```

Identical model/optimizer/data pipeline; only `Trainer(..., device="cuda")`
and `model.to("cuda")` differ. Hardware-verified on the reference GeForce
940MX (CC 5.0, driver 582.53, CUDA Toolkit 12.6, see
`docs/development/development-environment.md`):

| Quantity | CPU | CUDA |
|---|---:|---:|
| Train loss (MSE), epoch 1 -> 15 | 0.04520 -> 0.00619 | 0.04521 -> 0.00619 |
| Final test MSE | 0.00619 | 0.00617 |
| Final test pixel accuracy | 0.9980 | 0.9981 |
| Final test IoU | 0.9901 | 0.9909 |
| Throughput | ~116 samples/sec (~26s/epoch) | ~689 samples/sec (~4s/epoch) |

CPU and CUDA agree closely on every metric (final test MSE within 0.00002,
IoU within 0.0008) -- genuine end-to-end CPU/CUDA parity. **CUDA trains
~5.9x faster than CPU** here, consistent with `mnist`'s and `autoencoder`'s
own CUDA advantage on real `Conv2d`-heavy workloads (this model's three
`Conv2d` layers give the GPU enough real work per batch to amortize
per-batch launch/sync overhead) -- unlike `regression`'s small-batch MLP
(CUDA slower) or `waveform_classification` (roughly tied). No dedicated
CUDA optimization was pursued (per the M59 performance policy): both
devices train this small model to convergence well within a few minutes on
the reference hardware, and no evidence surfaced of a bottleneck significant
enough to justify optimization effort.

## Determinism

`forge.random.seed(args.seed)` (default `0`) governs `Conv2d` parameter
initialization at model construction. `SegmentationDataset` draws its own
images/masks from an independent `numpy.random.default_rng` stream (seeded
from `args.seed` via `dataset.py::make_datasets`, train and test using
disjoint seeds `seed`/`seed + 1`), and `DataLoader` shuffling uses a third,
explicit `numpy.random.Generator` -- the same three-separate-streams
convention `examples/regression/train.py` documents.

## Checkpointing and resume

Every `train.py` run saves a checkpoint (`segmentation_checkpoint.forge`)
capturing model + Adam state + epoch/global_step + Forge's RNG state:

```bash
python -m examples.segmentation.train --epochs 15 --output-dir artifacts
python -m examples.segmentation.train --resume artifacts/segmentation_checkpoint.forge --epochs 5 --output-dir artifacts
```

`--resume` restores the model, Adam state, and epoch/global_step counters
via `forge.load_checkpoint()` + `Trainer.resume()`, exactly like every other
Forge example, and is covered by
`tests/test_segmentation_example_integration.py::test_checkpoint_save_and_resume_restores_state_and_continues_training`
and `::test_resume_equivalence_matches_continuous_training`.

## Model persistence

`train.py` also demonstrates the plain (optimizer-free) persistence path:
after training, it records a prediction, calls `forge.save_model()`, reloads
with `forge.load_model()`, and asserts the reloaded model reproduces the
same prediction -- printed as `Verified: reloaded model reproduces the
pre-save prediction.` at the end of every run.

## Viewing a predicted mask

Every run also writes `segmentation_input.png`, `segmentation_predicted_mask.png`,
and `segmentation_ground_truth_mask.png` into `--output-dir` (Milestone 76,
via `forge.data.save_image`) -- the same query image `Model persistence`
above already runs through the model, rendered as real files instead of only
reported as scalar `pixel_accuracy`/`iou` numbers. Open all three to see what
the model actually segmented.

## CLI inspection

```bash
python -m forge model inspect examples/segmentation/artifacts/segmentation_model.forge
python -m forge checkpoint inspect examples/segmentation/artifacts/segmentation_checkpoint.forge
```

## Integration tests

`tests/test_segmentation_example_integration.py` (CPU, 25 tests) and
`tests/test_segmentation_example_cuda_integration.py` (CUDA, 5 tests; skips
cleanly without a working CUDA backend) exercise this pipeline end-to-end:
deterministic dataset generation (reproducibility, shape/dtype/value-range
checks, non-degenerate mask coverage), model forward shape and parameter
count, `PixelAccuracy`/`IoU` correctness against hand-crafted predictions
(perfect overlap, no overlap, partial overlap, empty-mask edge case,
threshold-on-raw-output behavior, batch-weighted accumulation), the
majority-class baseline computation, full-pipeline training that beats that
baseline on both pixel accuracy and IoU, checkpoint save/resume + resume
equivalence, model save/load prediction consistency, CPU/CUDA prediction
parity, CUDA parameter/gradient/Adam-state residency, and CLI inspection.
Run them with:

```bash
python -m pytest tests/test_segmentation_example_integration.py tests/test_segmentation_example_cuda_integration.py
```
