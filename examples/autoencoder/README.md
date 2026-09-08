# Forge Convolutional Autoencoder Example (Milestone 63)

An end-to-end demonstration of Forge's newest model family -- unsupervised
representation learning via a convolutional autoencoder -- brought to the
same standard as `examples/mnist`, `examples/char_rnn`/`word_rnn`,
`examples/regression`, and `examples/waveform_classification`:

```text
AutoencoderDataset -> DataLoader -> Trainer -> ConvAutoencoder
    (Conv2d/MaxPool2d encoder -> Linear bottleneck -> Linear/UpsampleNearest2d/Conv2d decoder)
    -> MSELoss -> Adam
```

Every previous example is supervised (classify a label, predict a
continuous target, generate the next token). This example trains a model to
reconstruct its own input through a compressed bottleneck -- a genuinely
different task shape (the training signal is the input itself, not a
separate label) that exercises a real, previously-uncovered architecture:
an encoder that shrinks a spatial feature map down, and a decoder that must
grow it back up again. Growing a spatial map back up is something no prior
Forge model needed, and Forge had no primitive for it -- see **The new
primitive: `nn.UpsampleNearest2d`** below.

## Files

- `dataset.py` -- `AutoencoderDataset`, a thin wrapper around
  `examples.mnist.dataset.MNISTDataset` that returns `(image, image)`
  reconstruction pairs instead of `(image, label)` pairs.
- `model.py` -- `ConvAutoencoder`, the encoder/decoder architecture, plus
  `build_model()`.
- `train.py` -- the runnable example: training, evaluation, checkpointing,
  resume, model persistence, and a latent-space semantic-quality check.

## Prerequisites

- Forge installed (`pip install -e .` from the repo root) with its `numpy`
  dependency; no other third-party packages.
- The real MNIST dataset -- see **The dataset** below for how to fetch it
  (this example reuses `examples/mnist`'s download machinery and, by
  default, its data directory).
- For CUDA training: a working Forge CUDA backend (see
  `docs/architecture/cuda-backend.md` and
  `docs/development/development-environment.md`).

## The dataset

Unlike `char_rnn`/`word_rnn`/`regression`/`waveform_classification`'s
synthetic, in-process data, this example reuses `examples/mnist`'s real,
downloaded corpus -- an autoencoder's reconstruction task is far more
interesting against real image structure than a synthetic one, and MNIST is
already the one real external dataset in the project.
`AutoencoderDataset` (`dataset.py`) wraps `examples.mnist.dataset.
MNISTDataset` and discards the label, returning `(image, image)` pairs: the
"target" `Trainer.fit()` expects is simply the (transformed) input itself.
No MNIST-specific parsing/downloading code is duplicated -- `AutoencoderDataset`
composes the already-tested `MNISTDataset` class directly.

By default, `--data-root` points at `examples/mnist/data` (not a separate
`examples/autoencoder/data`), so a user who has already run `examples/mnist`
does not need a second ~11MB download:

```bash
# If you have not already run examples/mnist, download once here:
python -m examples.autoencoder.train --download --epochs 3 --device cpu
```

Pixels are scaled to `[0, 1]` (`x / 255`) but **not** mean/std-normalized,
unlike `examples/mnist/train.py`'s classification preprocessing -- the
reconstruction target must stay in a natural, interpretable pixel range
since the decoder has no bounding activation (see **Model** below).

## The new primitive: `nn.UpsampleNearest2d`

Every prior Forge example only ever *shrinks* a spatial feature map
(`Conv2d`'s valid-region shrink, `MaxPool2d`'s stride-2 shrink). This
example's decoder needs the opposite: grow a `7x7` feature map back up to
`28x28`. Forge had no primitive for that -- `Tensor.reshape` cannot repeat
elements, and general N-D broadcasting (which could otherwise fake a
"repeat" via a zero-add) is a deliberately scoped-out CUDA capability (see
`docs/architecture/cuda-backend.md`). This was a genuine, real blocker
discovered by attempting the workload, not a speculative addition.

The smallest general fix: `nn.UpsampleNearest2d` (`forge/nn/upsample.py`),
backed by a new `Backend.upsample_nearest2d`/`upsample_nearest2d_backward`
primitive pair (`forge/backend/base.py`, real implementations in
`forge/backend/cpu.py` and a dedicated CUDA kernel pair in
`forge/backend/cuda/kernels.cu`/`backend.py`). Composing
`UpsampleNearest2d` with `Conv2d` (upsample, then convolve) is itself a
standard, deliberate architectural choice -- not a workaround for a missing
`ConvTranspose2d` -- since it avoids the checkerboard-artifact failure mode
transposed convolution is well known for. See
`docs/development/m63-conv-autoencoder.md` for the full justification, and
`tests/test_upsample_nearest2d.py`/`tests/test_cuda_upsample_nearest2d.py`
for its own dedicated test coverage (forward/backward correctness,
finite-difference gradients, CPU/CUDA parity, serialization, device
movement, `Sequential` composition/training).

## Model

```text
Encoder:
(N, 1, 28, 28)
    -> Conv2d(1, 16, k=3, pad=1)  -> ReLU -> MaxPool2d(2)   -> (N, 16, 14, 14)
    -> Conv2d(16, 32, k=3, pad=1) -> ReLU -> MaxPool2d(2)   -> (N, 32, 7, 7)
    -> Flatten -> Linear(1568, 32) -> ReLU                  -> (N, 32)  [latent]

Decoder (mirrors the encoder):
(N, 32)
    -> Linear(32, 1568) -> ReLU -> reshape                  -> (N, 32, 7, 7)
    -> UpsampleNearest2d(2) -> Conv2d(32, 16, k=3, pad=1) -> ReLU  -> (N, 16, 14, 14)
    -> UpsampleNearest2d(2) -> Conv2d(16, 1, k=3, pad=1)          -> (N, 1, 28, 28)
```

~111.5k trainable parameters at the default `--latent-dim 32`. Built as a
custom `ConvAutoencoder` `Module` (not a plain `Sequential`) so `encode()`/
`decode()` can be called separately -- `train.py`'s latent-space evaluation
needs `encode()` alone. It registers itself for persistence
(`forge.serialization.register_module()`), the same pattern
`examples/char_rnn/model.py`'s `CharRNN` and `examples/word_rnn/model.py`'s
`WordRNN` already establish for a custom composite `Module`.

## CPU training

```bash
python -m examples.autoencoder.train --epochs 3 --device cpu
```

Trains the autoencoder above with Adam (`lr=1e-3`) over the full 60,000-image
MNIST training split (10,000-image test split for validation), reports
per-epoch training/validation MSE, then saves a checkpoint and a model file
under `--output-dir` (default `examples/autoencoder/artifacts`).

### Expected approximate behavior (reference: this repository's CPU, i5-7200U)

Measured directly on this run (`--seed 0`, default hyperparameters,
`--epochs 3`):

| Quantity | Value |
|---|---:|
| Trivial baseline MSE (predict the training-mean image) | 0.06747 |
| Train MSE, epoch 1 | 0.03314 |
| Train MSE, epoch 3 | 0.01833 |
| Final test MSE | 0.01740 |
| Reduction over trivial baseline | 74.2% |
| Latent nearest-neighbor label agreement (200 test samples) | 80.0% (random baseline: 10.0%) |
| Throughput | ~190 train samples/sec (~310s/epoch) |

The latent nearest-neighbor check is a qualitative bonus, not the primary
success criterion: it asks whether two images that end up with nearby
32-dimensional latent codes tend to be the same digit, even though the
model was never given labels during training. A result well above the 10%
random baseline is evidence the bottleneck learned genuine digit-shape
structure, not merely enough information to copy pixels back out.
`tests/test_autoencoder_example_integration.py` checks the *algorithm*
(nearest-neighbor + label-agreement computation) deterministically against
hand-crafted latents, not this specific number, since the exact percentage
will vary run to run.

**Throughput note**: `examples/mnist`'s classifier trains at ~1,000-1,200
samples/sec on this same CPU. This autoencoder is slower (~190 samples/sec)
because it runs twice as many `Conv2d` layers per sample (2 encoder + 2
decoder) and, unlike `mnist`'s valid-padding convolutions (which shrink the
feature map every layer), this architecture's `padding=1` convolutions keep
the full spatial resolution at every layer -- most of the cost is the
decoder's `Conv2d(32, 16, ...)` at `14x14` and `Conv2d(16, 1, ...)` at
`28x28`, both processing far more spatial positions per layer than `mnist`'s
shrinking convolutions do. Reported as observed, per Milestone 59's
performance policy (`docs/development/m59-vision-and-next-stage.md` Section
12) -- 3 epochs over the full 60,000-image training set still completes in
under 16 minutes on the reference i5-7200U, which is practical for this
machine, and no evidence surfaced during this milestone that a deeper
optimization investigation is warranted.

## CUDA training

```bash
python -m examples.autoencoder.train --epochs 3 --device cuda
```

Identical model/optimizer/data pipeline; only `Trainer(..., device="cuda")`
and `model.to("cuda")` differ (`build_model()` itself is device-agnostic).
Hardware-verified on the reference GeForce 940MX (CC 5.0, driver 582.53,
CUDA Toolkit 12.6, see `docs/development/development-environment.md`):

| Quantity | CPU | CUDA |
|---|---:|---:|
| Train MSE, epoch 1 -> 3 | 0.03314 -> 0.01833 | 0.03310 -> 0.01836 |
| Final test MSE | 0.01740 | 0.01755 |
| Reduction over trivial baseline | 74.2% | 74.0% |
| Latent nearest-neighbor label agreement | 80.0% | 79.0% |
| Throughput | ~190 samples/sec (~310s/epoch) | ~1,170 samples/sec (~51s/epoch) |

Unlike `examples/regression`'s small-batch MLP (where CUDA was *slower*
than CPU) or `examples/waveform_classification`'s roughly-tied result, this
architecture's four `Conv2d` layers give CUDA real, substantial work per
batch -- **CUDA trains ~6.1x faster than CPU** here, closer to `mnist`'s own
CUDA speedup than to those other two examples' overhead-dominated results.
CPU and CUDA agree closely on every reported metric (final test MSE within
0.00015, baseline reduction within 0.2 percentage points), confirming
CPU/CUDA parity for this architecture end to end, not just at the kernel
level. `tests/test_autoencoder_example_cuda_integration.py` additionally
verifies, permanently:

- every model `Parameter` remains CUDA-resident (`CUDAStorage`, never a
  NumPy array) throughout training, including the new `UpsampleNearest2d`
  decoder path's own gradient flow,
- every `Parameter.grad` remains CUDA-resident,
- Adam's `m`/`v` state remains CUDA-resident,
- a CPU-trained and a CUDA-trained model (identical seed/data/one epoch)
  produce matching predictions on a held-out query batch within
  `rtol=1e-3, atol=1e-3`.

## Determinism

`forge.random.seed(args.seed)` (default `0`) governs `Conv2d`/`Linear`
parameter initialization at model construction. `DataLoader` shuffling uses
its own explicit `numpy.random.Generator` (`--seed`-derived), matching every
other Forge example's documented policy. MNIST's images themselves are a
fixed, already-downloaded corpus, not something this script draws randomly
(unlike `regression`'s synthetic-data generator, there is no third
"dataset generation" random stream here).

## Checkpointing and resume

Every `train.py` run saves a checkpoint (`autoencoder_checkpoint.forge`)
capturing model + Adam state + epoch/global_step + Forge's RNG state:

```bash
# Train from scratch for 3 epochs, saving a checkpoint.
python -m examples.autoencoder.train --epochs 3 --output-dir artifacts

# Resume from that checkpoint and continue for 2 more epochs.
python -m examples.autoencoder.train --resume artifacts/autoencoder_checkpoint.forge --epochs 2 --output-dir artifacts
```

`--resume` restores the model, Adam state, and epoch/global_step counters
via `forge.load_checkpoint()` + `Trainer.resume()`, exactly like every other
Forge example, and is covered by
`tests/test_autoencoder_example_integration.py::test_checkpoint_save_and_resume_restores_state_and_continues_training`
and `::test_resume_equivalence_matches_continuous_training`.

## Model persistence

`train.py` also demonstrates the plain (optimizer-free) persistence path:
after training, it records a reconstruction, calls `forge.save_model()`,
reloads with `forge.load_model()`, and asserts the reloaded model reproduces
the same reconstruction -- printed as `Verified: reloaded model reproduces
the pre-save reconstruction.` at the end of every run. The same property is
covered by
`tests/test_autoencoder_example_integration.py::test_model_persistence_preserves_predictions`
(CPU) and
`tests/test_autoencoder_example_cuda_integration.py::test_model_persistence_preserves_predictions_on_cuda`
(CUDA).

## CLI inspection

Every `train.py` run prints the exact commands to inspect its own output
with the Milestone 19 CLI:

```bash
python -m forge model inspect examples/autoencoder/artifacts/autoencoder_model.forge
python -m forge checkpoint inspect examples/autoencoder/artifacts/autoencoder_checkpoint.forge
```

`model inspect` reports the `ConvAutoencoder` layer tree, per-parameter
shapes/dtypes, and the total parameter count. `checkpoint inspect`
additionally reports the Adam hyperparameters, epoch/global_step, and how
many parameters have saved optimizer state.

## Integration tests

`tests/test_autoencoder_example_integration.py` (CPU) and
`tests/test_autoencoder_example_cuda_integration.py` (CUDA; skips cleanly
without a working CUDA backend) exercise this pipeline end-to-end against a
tiny synthetic `(N, 1, 28, 28)` stand-in dataset (matching
`tests/test_mnist_example_integration.py`'s established precedent, so
neither test file needs the real MNIST download): model construction/
forward shape, training loss reduction well below a trivial predict-the
-mean-image baseline, parameter updates, checkpoint save/restore (model
state, Adam state, epoch/global_step, device residency), resume
equivalence, model save/load prediction consistency, CPU/CUDA prediction
parity after identical training, CLI inspection, the latent
nearest-neighbor evaluation algorithm (checked deterministically against
hand-crafted latents), and `AutoencoderDataset`'s own MNIST-wrapping logic
(checked against tiny in-memory, real-IDX-format files -- no network
access). Run them with:

```bash
python -m pytest tests/test_autoencoder_example_integration.py tests/test_autoencoder_example_cuda_integration.py
```

`nn.UpsampleNearest2d`'s own dedicated correctness tests
(`tests/test_upsample_nearest2d.py`, `tests/test_cuda_upsample_nearest2d.py`)
are separate from this example's integration tests, matching how
`nn.Conv1d`/`nn.MaxPool1d`'s own tests are separate from
`examples/waveform_classification`'s integration tests.
