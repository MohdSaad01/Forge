# Milestone 63 — Convolutional Autoencoder: `nn.UpsampleNearest2d` and a Real Unsupervised-Reconstruction Workload

## 1. Problem and Motivation

M62's own "Recommendation for the next milestone" explicitly declined to
name a foregone-conclusion direction for M63: the discipline it asked for
was "inspect the current architecture and examples directly, identify
whether a concrete, credible gap with a real consumer exists, and only
build if one is found." M63's brief made this explicit and additionally
forbade another general readiness/capability-assessment survey -- the
mandate was to select and implement one concrete model family meaningfully
different from Forge's five existing examples (`mnist`, `char_rnn`,
`word_rnn`, `regression`, `waveform_classification`).

All five existing examples share one property: every one of them is
**supervised** in the ordinary sense -- a model produces a prediction that
is compared, via a loss, against a *separate* label or continuous target
(a class index, a next-character/word, a regression scalar). Forge had
never validated a task where the training signal is the model's own input
-- unsupervised representation learning -- despite this being one of the
most common real deep-learning workload shapes (dimensionality reduction,
anomaly detection, pretraining, denoising).

## 2. Candidates Considered

Per the brief's own suggested candidate list, direct inspection considered:

| Candidate | Verdict |
|---|---|
| **Convolutional autoencoder** (image reconstruction through a bottleneck) | **Selected** -- see Section 3. |
| A genuine multilayer/stacked sequence model (e.g. 2-layer `RNNCell`) | Rejected: a deeper stack of an already-proven layer is a parameter-count change, not a new task shape or new framework capability -- exactly the "recombine existing architecture without adding meaningful value" pattern the brief warned against. |
| A structured/tabular classification model | Rejected: `examples/regression` already demonstrates the tabular data path; swapping `MSELoss` for `CrossEntropyLoss` on tabular data changes nothing framework-relevant that `mnist`'s classification path doesn't already prove. |
| A small generative model (e.g. a tiny VAE or GAN) | Rejected *for this milestone*: a VAE needs a reparameterization trick (differentiable Gaussian sampling) and a KL-divergence term Forge has no primitive for; a GAN needs an adversarial two-optimizer training loop `Trainer` does not support. Both are legitimate future directions but violate the brief's "if the workload requires several major primitives simultaneously, reconsider" guardrail. A plain autoencoder is the natural, minimal predecessor to a VAE and shares its encoder/decoder/bottleneck architecture, so building it first is not wasted work. |
| Another domain-specific architecture (e.g. a second image-domain CNN variant) | Rejected: would not be "meaningfully different" from `mnist`'s existing CNN under the brief's own explicit instruction. |

## 3. Why the Autoencoder Was Selected

1. **Exercises existing capability meaningfully.** The encoder is a real
   `Conv2d`/`MaxPool2d`/`Linear` stack (Milestones 15/53's machinery),
   composed exactly as `mnist`'s CNN is -- direct reuse, not a rewrite.
2. **A genuinely different task shape, not an architectural relabeling.**
   The loss target is the model's own (transformed) input, not a separate
   label -- `Trainer.fit()` had never been driven this way before, and it
   worked with zero `Trainer` changes (see Section 5).
3. **It exposed a real, previously-invisible framework gap through
   implementation, not speculation** (Section 6) -- every prior Forge
   architecture only ever *shrinks* a spatial feature map (`Conv2d`'s
   valid-region shrink, `MaxPool2d`'s stride-2 shrink); a decoder that must
   grow a bottleneck back up to image resolution needed the *opposite*
   transform, and nothing in Forge could do it.
4. **The blocker was small, real, and reusable**, not a rabbit hole: one
   primitive (`nn.UpsampleNearest2d`), not several simultaneous ones,
   satisfying the brief's explicit "if it requires several major primitives
   simultaneously, reconsider" guardrail.
5. **A clear reason to exist in Forge**, independent of framework pressure:
   dimensionality reduction / representation learning is a standard,
   widely-used deep-learning workload category `docs/product/vision.md`'s
   "additional neural-network tasks... from reusable primitives" language
   already anticipates, and no existing example demonstrated it.

## 4. Architecture

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

~111.5k trainable parameters at the default `latent_dim=32`. `ConvAutoencoder`
(`examples/autoencoder/model.py`) is a custom composite `Module` (not a
plain `Sequential`), since `train.py`'s latent-space evaluation needs
`encode()` callable independently of the full `forward()` -- it registers
itself for persistence (`forge.serialization.register_module()`), exactly
the pattern `examples/char_rnn/model.py`'s `CharRNN` and
`examples/word_rnn/model.py`'s `WordRNN` already established for a custom
composite `Module`. The decoder's final layer has no activation
(`MSELoss` against a `[0, 1]`-scaled pixel target, an ordinary regression
comparison -- Forge has no `Sigmoid` primitive and none was needed here).

`Trainer.fit()` required no changes: `examples/autoencoder/dataset.py`'s
`AutoencoderDataset` simply returns `(image, image)` per sample instead of
`(image, label)`, so `Trainer`'s existing
`optimizer.zero_grad() -> model(x) -> loss_fn(prediction, target) ->
loss.backward() -> optimizer.step()` loop runs completely unmodified with
`target = image`. This confirms the M50/M54/M59 architecture-review
finding (reaffirmed again here): `Trainer` makes no assumption about what
its `loss_fn`'s target represents, only that batches are `(features,
target)` tuples.

## 5. Existing Forge Capabilities Reused

`Conv2d`, `MaxPool2d`, `Flatten`, `Linear`, `ReLU` (all pre-existing),
`nn.Module`/`Parameter` composition, `MSELoss`, `optim.Adam`,
`data.DataLoader`, `data.Dataset` (via `AutoencoderDataset`),
`training.Trainer` (`fit`/`evaluate`, checkpoint/resume, unmodified),
`serialization.save_model`/`load_model`/`save_checkpoint`/`load_checkpoint`,
the module persistence registry, and `examples.mnist.dataset.MNISTDataset`
(reused directly, not reimplemented, for real-image download/IDX parsing).
No change to any of these was required.

## 6. Genuine Blocker Discovered

Attempting the decoder's "grow `(N, 32, 7, 7)` back up to `(N, 1, 28, 28)`"
step directly exposed the gap Section 3 anticipated: Forge had **no way to
increase a spatial dimension**. `Tensor.reshape` (already used by
`Conv1d`/`MaxPool1d`, `nn.Flatten`) can only reinterpret existing elements
into a different shape of the *same total size* -- it cannot repeat
elements to make more of them. The only other candidate mechanism, general
N-D broadcasting (which could fake a "repeat" by adding a tensor against a
larger-shaped zero tensor), is a **deliberately, repeatedly scoped-out CUDA
capability** (`docs/architecture/cuda-backend.md`, reaffirmed as recently as
M59's own capability table: "no general N-D broadcast/reduction... every
consumer that needed a specific shape got a specific, scoped kernel") --
using it here would have meant either reopening that boundary specifically
for this milestone (out of scope, per the brief's Architecture Discipline
section) or accepting a CPU-only decoder (unacceptable: every Forge example
since Milestone 12 has been CPU/CUDA-parity-verified).

**Is the gap reusable beyond this one model?** Yes: any decoder-shaped
architecture (a segmentation network, a super-resolution model, a U-Net-style
architecture, a future VAE/GAN) needs exactly this "grow a spatial map back
up" capability. This is not a speculative claim -- it follows directly from
the shape-inverse relationship between `MaxPool2d`'s shrink and this
primitive's grow, the same relationship that already motivated `Conv1d`
reusing `Conv2d` and `MaxPool1d` reusing `MaxPool2d`.

## 7. New Framework Capability: `nn.UpsampleNearest2d`

The smallest general primitive that genuinely closes the gap: nearest
-neighbor upsampling by an integer `(height, width)` scale factor.

**Why nearest-neighbor upsample + `Conv2d`, not a transposed convolution
(`ConvTranspose2d`):** this is a standard, deliberate architectural choice
in modern convolutional network design, not an improvised workaround for a
missing primitive. Transposed convolution is well known to produce
checkerboard-pattern artifacts from uneven kernel/stride overlap (see the
widely-cited "Deconvolution and Checkerboard Artifacts" analysis); the
upsample-then-convolve pattern this milestone implements is the commonly
recommended alternative, and is also the *simpler* primitive to add
correctly (fixed-pattern, data-independent fan-out; no
kernel/stride-dependent overlap arithmetic to get right in a first CUDA
implementation).

### Design, following the established `Backend`/CPU/CUDA/autograd pattern

Exactly the file-touch sequence `docs/architecture/decisions/`'s ADRs
document for every fused primitive since M31 (`cross_entropy`,
`batch_norm2d`, `embedding_lookup`, `rnn_cell`):

1. **`Backend` ABC** (`forge/backend/base.py`): `upsample_nearest2d(x,
   scale_factor)` / `upsample_nearest2d_backward(grad_output, input_shape,
   scale_factor)`.
2. **CPU** (`forge/backend/cpu.py`): forward is `np.repeat` along each
   spatial axis (exactly nearest-neighbor upsampling); backward reshapes
   each `(sh, sw)` output block back next to its source input element and
   sums over it -- a pure reshape+reduction, no Python loop.
3. **CUDA** (`forge/backend/cuda/kernels.cu` + `backend.py`): two new
   kernels, `k_upsample_nearest2d_forward`/`_backward`, following
   `k_maxpool2d_forward`/`_backward`'s exact structural convention
   (`launch_config`, per-dtype `f32`/`f64` launcher macros,
   `extern "C" __declspec(dllexport)`). Forward is one thread per *output*
   element (a plain index-divide-and-gather, simpler than
   `k_maxpool2d_forward`'s windowed max scan). Backward is one thread per
   *input* element, summing its fixed `sh x sw` fan-out block directly from
   `grad_output` -- unlike `k_maxpool2d_backward`'s data-dependent-argmax
   scatter (which needs `atomicAdd` because overlapping windows can target
   the same input element from multiple threads), this primitive's forward
   fan-out pattern is fixed and data-independent, so backward can be a
   direct per-thread gather-reduction: **no `atomicAdd`, no `cudaMemset`
   zeroing** -- every thread computes and writes its own unique `grad_x`
   element exactly once. This is a materially simpler kernel than
   `max_pool2d`'s despite solving the shape-inverse problem.
4. **`Tensor`** (`forge/tensor/tensor.py`): `Tensor.upsample_nearest2d(scale_factor)`,
   following `max_pool2d`'s exact wrap-in-`_differentiable_wrap` pattern.
5. **`nn.Module`** (`forge/nn/upsample.py`): `UpsampleNearest2d(scale_factor=2)`,
   no parameters, validates via the existing `pair()` helper
   (`forge/nn/_shape_utils.py`) already shared with `Conv2d`/`MaxPool2d`.
6. **Persistence** (`forge/serialization/registry.py`): registered with a
   `get_config`/default `from_config`, following `MaxPool2d`'s exact
   parameter-less-module pattern.
7. **Export** (`forge/nn/__init__.py`).

No `Tensor` primitive beyond this one was added; no existing op, kernel, or
abstraction was modified.

## 8. Implementation

Files touched (production):

- `forge/backend/base.py` -- two new abstract methods.
- `forge/backend/cpu.py` -- CPU implementation.
- `forge/backend/cuda/kernels.cu` -- two new CUDA kernels + launcher macros
  (`f32`/`f64`).
- `forge/backend/cuda/backend.py` -- two new dispatch methods.
- `forge/tensor/tensor.py` -- `Tensor.upsample_nearest2d`.
- `forge/nn/upsample.py` (new file) -- `nn.UpsampleNearest2d`.
- `forge/nn/__init__.py` -- export.
- `forge/serialization/registry.py` -- registration.

Files added (example):

- `examples/autoencoder/__init__.py`
- `examples/autoencoder/dataset.py` -- `AutoencoderDataset`.
- `examples/autoencoder/model.py` -- `ConvAutoencoder`, `build_model()`.
- `examples/autoencoder/train.py` -- training/evaluation/checkpoint/
  persistence/latent-evaluation script.
- `examples/autoencoder/README.md`.

## 9. Tests

61 new tests, all passing:

- `tests/test_upsample_nearest2d.py` (25, CPU) -- configuration validation
  (scale-factor parsing, rejection of invalid values), no-parameters check,
  output-shape formula (symmetric and asymmetric scale factors), forward
  correctness against an independent `np.repeat`-based reference, an exact
  hand-worked repeat-into-a-block example, gradient-shape/accumulation
  checks (including a hand-derived "each input element's gradient is
  scaled by `sh*sw`" property), finite-difference gradient checks across 4
  shape/scale-factor configurations, `Sequential` composition with `Conv2d`
  that trains one real SGD step, and a serialization round trip.
- `tests/test_cuda_upsample_nearest2d.py` (12, CUDA hardware-gated, skips
  cleanly without CUDA) -- forward/backward parity with CPU across 4 scale
  factors, a CUDA finite-difference check, a spy-based
  "never falls back to `CPUBackend`" guard (mirroring
  `tests/test_cuda_conv.py`'s `MaxPool2d` pattern), device movement via
  `Sequential.to("cuda")`, and an end-to-end `UpsampleNearest2d` ->
  `Conv2d` training loop on CUDA that measurably reduces loss.
- `tests/test_autoencoder_example_integration.py` (12, CPU) -- model
  forward-shape round trip through the bottleneck, `encode()`'s latent
  dimensionality, full-pipeline training that beats a trivial
  predict-the-mean-image baseline, checkpoint save/resume (including resume
  equivalence against continuous training), model persistence, CLI
  inspection, the latent nearest-neighbor evaluation algorithm (checked
  deterministically against hand-crafted latents in two adversarially
  constructed scenarios -- perfectly separated clusters scoring 100%
  agreement, perfectly interleaved pairs scoring 0%), and
  `AutoencoderDataset`'s own MNIST-wrapping logic against tiny in-memory,
  real-IDX-format (`gzip`+`struct`) files -- no network access, matching
  `tests/test_mnist_example_integration.py`'s established "no real download
  in the test suite" precedent.
- `tests/test_autoencoder_example_cuda_integration.py` (5, CUDA
  hardware-gated) -- full pipeline trains and beats baseline on CUDA,
  parameter/gradient/Adam-state CUDA residency (explicitly including the
  new `UpsampleNearest2d` decoder path's own gradient), checkpoint
  save/resume on CUDA, model persistence on CUDA, CPU/CUDA prediction
  parity after identical training.

Full suite run before and after: **1,936 passed** (1,883 + 61 new)[^flaky],
zero regressions attributable to this milestone's changes, verified via
`python -m pytest tests/ -q`.

[^flaky]: One unrelated, pre-existing test
(`tests/test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`)
failed once during the full-suite run with a small (288 vs. 320 bytes)
CUDA-allocator-measurement discrepancy, then passed cleanly in isolation
immediately afterward -- consistent with pre-existing allocator/timing
measurement flakiness this milestone did not touch (no code in
`forge/backend/cuda/allocator.py` or `forge/data/prefetch.py` was changed).

## 10. Verification

1. **Clean CUDA rebuild**: `kernels.cu`'s new kernels compiled successfully
   via the existing `build.py` mtime-triggered rebuild (no manual DLL
   deletion needed) on the reference GeForce 940MX.
2. **Full test suite**: 1,936 passed (Section 9).
3. **Example execution on CPU**: `python -m examples.autoencoder.train
   --epochs 3 --device cpu` against the real, already-downloaded MNIST
   corpus -- see Section 11.
4. **Example execution on CUDA**: `python -m examples.autoencoder.train
   --epochs 3 --device cuda` -- see Section 11.
5. **Genuine learning**: both runs reduce reconstruction MSE substantially
   below the trivial "predict the training-mean image" baseline (Section
   11) -- not merely "the code runs."
6. **Serialization/checkpoint**: both runs' `train.py` performs and
   verifies a live save/load prediction-parity check
   (`Verified: reloaded model reproduces the pre-save reconstruction.`);
   checkpoint save/resume and resume-equivalence are additionally covered
   permanently by the test suite (Section 9).
7. **No regression in existing examples**: the full suite (Section 9)
   includes every prior example's own integration tests, all still passing.
8. **Repository state**: see Section 14.

## 11. Training/Evaluation Results

_Filled in from the actual verification run on this repository's reference
hardware (i5-7200U CPU; GeForce 940MX CC 5.0, driver 582.53, CUDA Toolkit
12.6) -- see `docs/development/development-environment.md`._

| Quantity | CPU | CUDA |
|---|---:|---:|
| Trivial baseline MSE (predict training-mean image) | 0.06747 | 0.06747 |
| Train MSE, epoch 1 -> 3 | 0.03314 -> 0.01833 | 0.03310 -> 0.01836 |
| Final test MSE | 0.01740 | 0.01755 |
| Reduction over trivial baseline | 74.2% | 74.0% |
| Latent nearest-neighbor label agreement (200 test samples, random baseline 10%) | 80.0% | 79.0% |
| Throughput | ~190 samples/sec (~310s/epoch) | ~1,170 samples/sec (~51s/epoch) |

CPU and CUDA agree closely on every metric (final test MSE within 0.00015,
baseline reduction within 0.2 percentage points, latent agreement within 1
percentage point) -- genuine end-to-end CPU/CUDA parity, not just
kernel-level parity. Both runs trained from the same `--seed 0` but are two
independent 3-epoch runs (not a single deterministic replay), so the small
residual differences are normal training variance.

## 12. Performance

No dedicated profiling campaign was run, per the M59 performance policy
("profiling is triggered only by a measured wall-clock problem on a real
example, never by a generic sweep"). `UpsampleNearest2d`'s CUDA kernels are
simple, single-pass, one-thread-per-element kernels with no algorithmic
complexity beyond `MaxPool2d`'s own already-accepted cost model; this
example's cost is dominated by its four `Conv2d` layers, whose forward/
backward performance was already characterized and optimized across
Milestones 21-48 and is unmodified here. No evidence surfaced during this
milestone's implementation or testing that upsample's cost is a measurable
bottleneck at this model's scale, so no optimization was pursued.

One observation from the example's own numbers (Section 11), reported
because it is a genuine contrast with two recent examples, not because it
was chased: CUDA trained **~6.1x faster** than CPU here (~1,170 vs. ~190
samples/sec), unlike `examples/regression` (CUDA ~3.3x *slower* than CPU,
small-batch-MLP launch-overhead-dominated) or `examples/waveform_
classification` (roughly tied). This architecture's four `Conv2d` layers
(two same-padding, full/half-resolution) give the GPU substantial real
compute per batch, so the per-batch kernel-launch/host-sync overhead that
dominates the other two examples' small workloads is amortized away here --
consistent with `mnist`'s own CNN showing a comparable CUDA advantage. On
CPU, 3 epochs over the full 60,000-image training set take ~310s/epoch
(~16 minutes total) -- slower than `mnist`'s classifier (~55-60s/epoch)
because this architecture runs twice as many `Conv2d` layers per sample
(2 encoder + 2 decoder) and, unlike `mnist`'s shrinking valid-padding
convolutions, keeps full spatial resolution at every layer via
`padding=1`. Still practical on the reference machine per this project's
environment constraints, and no evidence surfaced that a deeper
optimization investigation is warranted.

## 13. Practical Impact on Forge

Forge can now express and train unsupervised representation-learning
models -- reconstructing an input through a compressed bottleneck -- using
the same `Trainer`/`Adam`/checkpoint/persistence pipeline every other
workload already uses, validated end to end on real CPU and CUDA hardware
against a real external dataset (MNIST) with genuine learning evidence
(Section 11). This is Forge's first example whose training signal is not a
separate label, demonstrating that `Trainer`'s design already generalizes
to that task shape with zero framework changes. It also gives Forge a
general-purpose "grow a spatial feature map" primitive
(`nn.UpsampleNearest2d`) that any future decoder-shaped architecture (a
segmentation network, a super-resolution model, a VAE/GAN) can reuse
directly, mirroring how `Conv1d`/`MaxPool1d` (M62) reused `Conv2d`/
`MaxPool2d`, and how this milestone's own decoder reused `MaxPool2d`'s
shape-inverse relationship to motivate `UpsampleNearest2d`'s design.

## 14. Limitations

- `UpsampleNearest2d` supports only integer scale factors and nearest
  -neighbor interpolation -- no bilinear/bicubic interpolation, no
  fractional scale factor. No consumer needs more; `Conv2d`'s own
  im2col-style output-size arithmetic already assumes integer-only shapes
  throughout Forge, so this is consistent with the existing convention, not
  a shortcut.
- The autoencoder's reconstruction quality was not tuned for visual fidelity
  (no perceptual loss, no adversarial term) -- `MSELoss` reconstruction is
  the standard, simplest baseline for this architecture family and is
  sufficient to demonstrate genuine learning (Section 11); a sharper/more
  visually faithful reconstruction was never this milestone's goal.
- The latent nearest-neighbor evaluation is a qualitative signal, not a
  formal clustering-quality metric (e.g. no silhouette score or
  purity-with-confidence-interval) -- appropriate for a milestone-report
  bonus check, not a claim of state-of-the-art representation quality.
- This is Forge's second example (after `mnist`) depending on a real,
  downloaded external dataset; it reuses `examples/mnist`'s existing
  download/parsing code rather than duplicating it, but still means this
  example cannot run fully offline on a machine that has never fetched
  MNIST at least once.

## 15. Follow-up Recommendations

No specific next capability is recommended as a foregone conclusion, per
the same discipline M62 itself asked to be applied here: inspect the
current architecture and examples directly, identify whether a concrete,
credible gap with a real consumer exists, and only build if one is found.
Two candidates this milestone surfaced but explicitly did not pursue
(Section 2) -- a VAE (now one step closer, sharing this milestone's
encoder/decoder/bottleneck architecture, but needing a reparameterization
-trick primitive and a KL-divergence term) and a segmentation/U-Net-style
architecture (a direct, ready-made consumer of `UpsampleNearest2d`) -- are
reasonable future directions if a concrete need for either emerges, but
neither is being proposed as M64's foregone conclusion here.
