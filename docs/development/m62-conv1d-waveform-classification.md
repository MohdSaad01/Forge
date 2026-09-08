# Milestone 62 — `nn.Conv1d`/`nn.MaxPool1d`: 1D Temporal Convolution + Waveform Classification

## 1. Objective

M59 ended the recurring "assessment only, no capability added" pattern that
ran through M49-M58. M60 and M61 both delivered concrete work (a complete
tabular regression workload; README/discoverability fixes). M62's brief was
explicit: no more capability surveys, no more "what should we build next?"
milestones -- implement exactly one concrete, reusable framework capability
with a credible consumer, validate it end to end on CPU and CUDA, and stop.

## 2. Why this capability was selected

Inspection of the current architecture (not another six-area survey) started
from what Forge's four existing examples actually cover:

| Example | Model family | Data shape |
|---|---|---|
| `mnist` | 2D CNN (`Conv2d`/`MaxPool2d`) | `(N, C, H, W)` images |
| `char_rnn`/`word_rnn` | Recurrence (`RNNCell`, step-by-step) | token sequences |
| `regression` | Plain MLP (`Linear`/`ReLU`) | flat feature vectors |

Nothing in Forge could express **1D/temporal convolution** -- the standard
architecture family for fixed-length time-series/sensor/audio
classification (sliding a learned kernel along a single time axis, as
opposed to `Conv2d`'s two spatial axes or `RNNCell`'s step-by-step
recurrence). This is a real, structural gap in `UC1` ("train a classifier",
`docs/product/use-cases.md`) coverage: every one of Forge's classifiers so
far is either 2D-image-shaped or sequence-generation-shaped, and no example
demonstrates the classify-a-fixed-length-1D-signal workload at all.

Critically, this is **not** one of the milestone brief's explicitly rejected
directions (LSTM/GRU, attention/transformers, deeper RNN, generic softmax,
LayerNorm) resurrected under a new name -- it is a different op family
(convolution, not recurrence or attention) addressing a use case none of
those would have addressed either.

**Why implement it now, and why this way specifically**: `nn.Flatten`
already established the exact pattern needed -- "compose a new, useful
`nn.Module` entirely from `Tensor.reshape` plus an existing differentiable
op, with zero new `Backend` code" -- for collapsing dimensions ahead of
`Linear`. The same trick applies directly to `Conv1d`: reshape `(N, C, L)`
to a dummy `(N, C, 1, L)`, run the existing (already CPU/CUDA-hardware-
verified, already performance-optimized across Milestones 21-48) `Conv2d`
machinery with a `1`-sized spatial dimension, reshape the result back down.
This makes `Conv1d`/`MaxPool1d` a **near-zero-risk** addition: no new CUDA
kernel, no new `Backend` method, no new autograd rule, and therefore no new
CPU/CUDA-parity surface to get wrong -- it inherits `Conv2d`'s correctness
and performance characteristics by construction. That combination (real,
previously-uncovered use case; implementable with essentially no new
low-level risk; fits the existing Tensor -> Backend -> Module -> autograd
architecture with no rewrite) is exactly the profile the brief asked for.

**Why the obvious alternatives were not selected**:

- **LSTM/GRU/attention/LayerNorm/generic softmax/deeper RNN** -- explicitly
  pre-rejected by the brief; no new evidence surfaced this milestone that
  changes that.
- **Sequence classification via the existing `Embedding`+`RNNCell` stack**
  (e.g. sentiment analysis) -- rejected under Selection Criterion 7 ("if the
  best candidate requires no new framework capability, reject it"): this is
  already fully expressible with Forge's current API (as the model-family
  table above shows, `char_rnn`/`word_rnn` already exercise exactly that
  stack for generation; classification would only change the loss/output
  wiring, not add framework capability).
- **A learning-rate scheduler / gradient clipping** -- legitimate training
  utilities, but training-robustness conveniences, not a new *model family*
  or expansion of what Forge can express; the brief's stated preference is
  for the latter, and no concrete evidence (e.g. an existing example
  observably failing to converge without one) motivated adding one now.
- **`AvgPool2d`/`AdaptiveAvgPool2d`/other isolated layer conveniences** --
  cosmetic API surface expansion, explicitly the kind of "technically
  interesting but not the priority" addition the brief said to avoid.

## 3. Concrete consumer / workload

`examples/waveform_classification/` -- a new, complete example at the same
quality bar as `mnist`/`char_rnn`/`word_rnn`/`regression`: `dataset.py`,
`model.py`, `train.py`, `README.md`.

**Dataset** (`dataset.py`): length-64, single-channel synthetic time series,
one of 4 classes (sine/square/sawtooth/triangle), each drawn with a random
frequency (`Uniform(2, 6)` cycles per window), phase (`Uniform(0, 2*pi)`),
and amplitude (`Uniform(0.7, 1.3)`), then corrupted with additive Gaussian
noise (`std=0.25`). Generated entirely in-process from a fixed seed --
`char_rnn`/`word_rnn`/`regression`'s established "no download" precedent,
not a new pattern. The random per-sample phase/frequency is a deliberate
design choice: it makes the task **not** trivially solvable by memorizing
one fixed waveform per class, so a model must learn genuine, shift-invariant
local shape features -- precisely what a convolutional kernel (as opposed to
a plain `Linear` layer operating on raw time-indexed features) is suited to
extract. Trivial baseline: uniform random guessing over 4 classes = 25%
accuracy.

**Model** (`model.py`):

```text
(N, 1, 64) -> Conv1d(1, 16, k=7, pad=3) -> ReLU -> MaxPool1d(2)   -> (N, 16, 32)
           -> Conv1d(16, 32, k=5, pad=2) -> ReLU -> MaxPool1d(2)  -> (N, 32, 16)
           -> Flatten -> Linear(512, 64) -> ReLU -> Linear(64, 4) -> (N, 4)
```

~35k trainable parameters, entirely `forge.nn.Sequential` composition -- no
custom `Module` subclass, no `register_module()` call needed (every layer is
an already-registered built-in).

**Training** (`train.py`): identical pipeline shape to `mnist`/`regression`
-- `DataLoader` -> `Trainer` -> `CrossEntropyLoss` -> `Adam`, with
checkpoint/resume and model-persistence verification, run from one CLI
entry point.

## 4. Existing Forge capability used

`Tensor.reshape` (already differentiable, already real on both CPU and
CUDA), `Tensor.conv2d`/`Tensor.max_pool2d` (already CPU/CUDA-tested,
CUDA-optimized), `nn.Module`/`Parameter` composition, `nn.Sequential`,
`nn.Flatten`, `nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss`, `optim.Adam`,
`data.TensorDataset`/`DataLoader`, `training.Trainer`/`Accuracy`,
`serialization.save_model`/`load_model`/`save_checkpoint`/`load_checkpoint`,
the module persistence registry. No change to any of these was required.

## 5. Implementation

`forge/nn/conv.py` gained `Conv1d` (alongside the existing `Conv2d`);
`forge/nn/pooling.py` gained `MaxPool1d` (alongside `MaxPool2d`). Both:

- Own real, independent `Parameter`s in their natural 1D-friendly shapes
  (`weight: (C_out, C_in, K)`, `bias: (C_out,)` for `Conv1d`; no parameters
  for `MaxPool1d`) -- only the *view* passed into `conv2d`/`max_pool2d` is
  reshaped per `forward()` call, not the stored Parameter.
- `forward()`: reshape input `(N, C, L)` -> `(N, C, 1, L)`, reshape
  `weight` -> `(C_out, C_in, 1, K)` (Conv1d only), call the existing
  `Tensor.conv2d`/`max_pool2d` with a `1`-sized first spatial dimension
  (`stride=(1, s)`, `padding=(0, p)`), reshape the `(N, C_out, 1, L_out)`
  result back down to `(N, C_out, L_out)`.
- Same `Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))` initialization scheme as
  `Conv2d` (`fan_in = in_channels * kernel_size`).
- Same validation conventions as `Conv2d`/`MaxPool2d` (`ShapeMismatchError`
  for invalid channels/kernel_size/stride/padding, wrong input rank, channel
  mismatch, kernel-larger-than-padded-input).

Both registered for persistence (`forge/serialization/registry.py`,
`get_config`/default `from_config` following `Conv2d`/`MaxPool2d`'s exact
pattern) and exported from `forge.nn` (`forge/nn/__init__.py`).

## 6. Architecture impact

None beyond the two new `nn.Module` classes. No `Backend` method was added,
no CUDA kernel was written or modified, no autograd rule was added, no
`Tensor` primitive was added. `Conv1d`/`MaxPool1d` sit entirely at the
`Module` composition layer, calling only existing, already-real `Tensor`
operations -- exactly the boundary discipline the project's core principles
require ("keep tensors, autograd, model abstractions ... inside Forge" was
already satisfied by `Conv2d`; this milestone reuses that, it does not
re-derive it).

## 7. CPU behavior

Verified directly:

- Forward correctness against an independent triple-loop reference
  implementation (not `Conv2d`-derived, so a shared bug would still be
  caught) across stride/padding/bias combinations.
- Gradient accumulation (weight reused across calls, input reused across
  layers) matches summed independent-call gradients.
- Finite-difference gradient checks for input/weight/bias pass across
  4 shape/stride/padding configurations (`tests/test_conv1d.py`).
- `MaxPool1d`: forward matches an independent reference, deterministic
  first-occurrence tie-breaking, overlapping-window (`stride < kernel_size`)
  gradient accumulation, finite-difference check.
- Serialization round trip (`save_model`/`load_model`) preserves both
  predictions and construction-time config for both layers.
- End-to-end: the waveform classification example trains from 25% baseline
  to 90.80% test accuracy over 15 epochs on the reference i5-7200U CPU.

## 8. CUDA behavior

**Hardware-verified on the reference GeForce 940MX** (CC 5.0, driver 582.53,
CUDA Toolkit 12.6) -- not simulated:

- `Conv1d`/`MaxPool1d` forward output is real `CUDAStorage` throughout (no
  silent CPU fallback at any point in the reshape-then-`conv2d` chain).
- Forward and backward (input/weight/bias gradients) numerically match CPU
  within `rtol=1e-4, atol=1e-4` across stride/padding/bias configurations.
- Device movement (`.to("cuda")`/`.to("cpu")`) moves `Conv1d`'s
  weight/bias correctly.
- An end-to-end tiny temporal-CNN (`Conv1d`->`ReLU`->`MaxPool1d`->`Linear`)
  trains one real SGD step on CUDA with a measurable parameter change.
- `Conv1d`/`MaxPool1d` compose correctly inside `nn.Sequential` on CUDA,
  with gradients flowing back to a CUDA-resident `Conv1d.weight`.
- The full waveform classification example trains end to end on CUDA:
  25% baseline -> 91.80% test accuracy over 15 epochs, parameter/gradient/
  Adam-state CUDA residency verified throughout, checkpoint save/resume on
  CUDA verified, and CPU-trained vs. CUDA-trained models (identical seed/
  data/one epoch) produce matching predictions within `rtol=1e-3,
  atol=1e-3`.

No new CUDA kernel exists to have a parity bug in the first place -- every
CUDA number above is `Conv2d`/`MaxPool2d`'s own already-verified kernel
running on a reshaped view, which is precisely why this implementation
strategy was chosen (Section 2).

## 9. Autograd behavior

`Tensor.reshape` is already a fully differentiable, backend-dispatched
operation with a correct backward rule (`reshape_backward`, which reshapes
the incoming gradient back to the original shape). `Conv1d.forward` chains
two reshapes around one `conv2d` call; `MaxPool1d.forward` chains two
reshapes around one `max_pool2d` call. Since every one of these ops
(`reshape`, `conv2d`, `max_pool2d`) already builds correct autograd graph
nodes independently, their composition is correct by the ordinary chain
rule -- no new backward-rule code was written, and none was needed. This
was verified empirically, not just argued: finite-difference gradient
checks pass for `Conv1d`'s input/weight/bias and `MaxPool1d`'s input on
both CPU and CUDA.

One subtlety confirmed directly: `self.weight.reshape(...)` (a `Parameter`,
a `Tensor` subclass) produces a plain differentiable `Tensor` whose one
graph input is the `Parameter` itself -- gradients flow back through the
reshape node and accumulate into `Parameter.grad` exactly as they do for
any other leaf-tensor-used-via-an-intermediate-op case already exercised
elsewhere in Forge (e.g. `RNNCell` reusing the same `Linear` weight across
timesteps).

## 10. Testing

91 new tests, all passing:

- `tests/test_conv1d.py` (36, CPU) -- configuration validation, parameter
  shapes/initialization, forward correctness vs. an independent reference,
  output-shape formula, runtime shape validation, gradient accumulation
  (weight reused / input reused), SGD integration, finite-difference
  gradient checks (4 configurations), serialization round trip.
- `tests/test_maxpool1d.py` (27, CPU) -- same structure adapted to 1D,
  including deterministic tie-breaking and overlapping-window gradient
  accumulation, plus serialization round trip.
- `tests/test_cuda_conv1d.py` (12, CUDA hardware-gated, skips cleanly
  without CUDA) -- forward/backward parity with CPU (with/without bias),
  `MaxPool1d` forward/backward parity, device movement, an end-to-end tiny
  temporal-CNN training step, `Sequential` integration -- every assertion
  checks for real `CUDAStorage`, not just "did not crash."
- `tests/test_waveform_classification_example_integration.py` (11, CPU) --
  deterministic dataset generation/reproducibility, split sizing, class
  balance sanity, model forward shape, full-pipeline training beats
  baseline, checkpoint save/resume, resume equivalence, model persistence,
  CLI inspection.
- `tests/test_waveform_classification_example_cuda_integration.py` (5, CUDA
  hardware-gated) -- full pipeline trains and beats baseline on CUDA,
  parameter/gradient/Adam-state CUDA residency, checkpoint save/resume on
  CUDA, model persistence on CUDA, CPU/CUDA prediction parity after
  identical training.

Full suite run before and after: **1,792 -> 1,883 passed** (91 new), zero
regressions, verified via `python -m pytest tests/ -q` (both runs) and
`python -m pytest tests/ --collect-only -q` for the exact count.

## 11. Real workload validation

Not merely "the code runs" -- the workload demonstrates genuine learning
against a well-defined, non-trivial baseline:

| Metric | CPU (i5-7200U) | CUDA (940MX) |
|---|---:|---:|
| Trivial baseline accuracy | 25.00% | 25.00% |
| Train loss, epoch 1 -> 15 | 0.8386 -> 0.1612 | 0.8386 -> 0.1545 |
| Final test accuracy | 90.80% | 91.80% |
| Throughput | ~2,900 samples/sec | ~3,100 samples/sec |

CPU and CUDA results agree within normal training-run-to-run variance (both
trained from the identical seed/data but are two independent 15-epoch runs,
not a single deterministic replay); the dedicated CUDA integration test
additionally verifies exact numerical parity for a controlled single-epoch,
identical-data comparison (`rtol=1e-3, atol=1e-3`).

## 12. Serialization/device behavior

`Conv1d`/`MaxPool1d` are registered built-in persistable types
(`forge/serialization/registry.py`); `save_model`/`load_model` round trips
preserve both predictions and construction-time configuration (verified in
`tests/test_conv1d.py`/`tests/test_maxpool1d.py`). `Module.to(device)`
already moves any `Parameter`/buffer generically -- `Conv1d.weight`/`.bias`
require no special-casing and were verified to move correctly in both
directions (`tests/test_cuda_conv1d.py`). The waveform classification
example additionally verifies checkpoint save/resume (model + Adam state +
epoch/step) on both CPU and CUDA, matching `regression`'s established
contract exactly.

## 13. Performance observations

No dedicated profiling campaign was run, per the brief's "benchmark only
where it matters" guidance -- `Conv1d`/`MaxPool1d` introduce zero new
kernels to profile; their cost is entirely `Conv2d`/`MaxPool2d`'s
already-characterized cost (Milestones 21-48) plus two cheap reshape calls
(a device-to-device memcpy on CUDA, a view on CPU) per invocation.

One observation from the example's own numbers: CPU and CUDA trained at
comparable throughput on this small architecture/batch size (~2,900 vs.
~3,100 samples/sec) -- neither a CUDA win nor a regression. This matches
`regression`'s documented finding (`examples/regression/README.md`) that
small-batch, small-parameter-count workloads on the reference 940MX are
dominated by per-batch kernel-launch/host-sync overhead rather than raw
compute; it is reported as observed, not chased with further optimization,
consistent with Milestone 59's performance policy and this milestone's own
explicit "do not repeat the M31-M48 optimization cycle without fresh
evidence of a genuine bottleneck" instruction.

## 14. Repository hygiene

- Removed the tracked `.idea/` directory from git (`git rm -r --cached
  .idea/`; files remain on disk, just untracked).
- Added `.idea/` to `.gitignore`.
- Added `examples/waveform_classification/artifacts*/` to `.gitignore`,
  matching the existing per-example generated-artifact-directory pattern.
- Verified via `git status`/`git check-ignore -v` that `.idea/` and both
  generated artifact directories (`artifacts/`, `artifacts_cuda/`) are now
  correctly ignored, and that `git status` shows only the intended,
  reviewable set of changes.

## 15. Limitations

- `Conv1d`/`MaxPool1d` share `Conv2d`/`MaxPool2d`'s existing restrictions:
  integer stride, integer symmetric zero padding, no dilation, no groups,
  no transposed/depthwise convolution. No evidence this milestone motivates
  relaxing any of those.
- The reshape-based implementation means `Conv1d`'s CUDA cost includes one
  extra device-to-device memcpy per `forward()` call (for the input
  reshape) beyond what a dedicated 1D kernel would need; at the shapes this
  milestone's example uses this is not measurable against total step time,
  and no evidence suggests it matters at any shape Forge currently
  exercises. If a future workload's profiling ever showed this reshape
  overhead dominating, a dedicated `Backend.conv1d` could be added later --
  not warranted now.
- The waveform classification dataset is synthetic (matching `char_rnn`/
  `word_rnn`/`regression`'s established precedent for no-download
  examples), not a real external corpus -- appropriate for demonstrating
  the capability, as `regression`'s synthetic tabular dataset was judged
  appropriate for the same reason in M60.

## 16. Rejected alternatives

See Section 2 for the full reasoning. Summary: LSTM/GRU/attention/LayerNorm/
generic softmax remain rejected (no new evidence this milestone); an
`Embedding`+`RNNCell`-based sequence *classification* example was rejected
under the "no new framework capability" criterion (already fully
expressible); an LR scheduler and isolated pooling/layer convenience
additions were rejected as training-utility/cosmetic additions rather than
new model-family expansions, which the brief explicitly prioritized higher.

## 17. Practical impact on Forge

Forge can now express and train 1D-convolutional models -- the standard
architecture family for fixed-length time-series, sensor, and audio
classification -- with the same `Trainer`/`Adam`/checkpoint/persistence
pipeline every other workload already uses, validated end to end on real
CPU and CUDA hardware with genuine learning evidence (25% -> ~91% test
accuracy). This closes a real gap in Forge's classifier coverage (`UC1`)
that no existing example addressed, at effectively zero new low-level
implementation risk (no new kernels, no new backend code, no new autograd
rules). Incidentally, CUDA-verifying this example's persistence-check step
surfaced and fixed a real, previously-undetected bug in
`examples/regression/train.py`'s own CUDA path (a `-1`-shape-inference
crash), directly improving this milestone's "existing examples remain
functional on CUDA" regression-safety bar.

## 18. Recommendation for the next milestone

This milestone closed a real, previously-uncovered classifier model family
with near-zero implementation risk because a clean composition path already
existed (`reshape` + `Conv2d`). That kind of opportunity is not guaranteed
to recur, and M62's own instruction not to "manufacture work" applies to
M63 too. No specific next capability is recommended here as a foregone
conclusion -- the honest next step is the same discipline this milestone
followed: inspect the current architecture and examples directly, identify
whether a concrete, credible gap with a real consumer exists, and only
build if one is found. If direct inspection finds no such gap, the correct
outcome is a short, honest "no concrete gap found this pass," not a
survey-shaped milestone and not invented work.
