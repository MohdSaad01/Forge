# Milestone 66 — Residual CNN Workload: ResNet-Style Image Classification

## 1. Workload Selection

A small ResNet-style image classifier for MNIST: `examples/resnet/` --
model, training, evaluation, checkpointing, resume, persistence, README --
following the exact `examples/mnist`/`examples/segmentation` file layout and
conventions, and reusing `examples/mnist/dataset.py` (the dataset) and
`examples/regression/experiment.py`/`compare.py` (the Milestone 65
reproducible-training workflow) rather than introducing either a new
dataset or a second experiment-tracking mechanism.

The architecture follows the brief's own suggested shape exactly:

```text
input -> Conv2d -> BatchNorm2d -> ReLU -> residual block (conv branch +
identity/shortcut, added, re-activated) -> ... -> classifier
```

## 2. Why Residual Networks Are a Meaningful Next Workload

Every prior Forge CNN example (`mnist`, `segmentation`, `autoencoder`) is a
strictly feed-forward stack: each layer's output is the next layer's only
input. A residual connection introduces a structurally different data-flow
shape -- one tensor consumed by **two** independent downstream paths that
later recombine by addition -- which no prior Forge workload exercised.
This is exactly the shape M50's char-RNN already proved Forge's autograd
handles for a *parameter* (one `Linear` reused across every timestep of an
unrolled graph); M66 asks the analogous question for a *tensor value*
consumed by two branches of one forward pass. Residual connections are also
the specific architectural idea that makes very deep CNNs trainable in
practice (mitigating vanishing gradients), making this a natural, concrete
next capability question for Forge's CNN support -- not a speculative
"because other frameworks have it" addition.

## 3. Baseline Forge Capabilities

Everything this workload was expected to need already existed:

- `Tensor.__add__`/`Tensor.sum()`/autograd's reverse-topological-order
  gradient accumulation (M1-era, exercised at scale by every training
  workload since).
- `Conv2d` with arbitrary stride/padding, including a 1x1 "projection"
  convolution (`kernel_size=1`) -- no special case needed for this.
- `BatchNorm2d` (M53), including its buffer (`running_mean`/`running_var`)
  machinery, and correct behavior nested inside a custom `Module` tree.
- `Module`/`Parameter`/custom `Module` subclassing, `Module.to()`,
  `train()`/`eval()` recursive propagation.
- `Adam`, `DataLoader`, `Trainer.fit()`/`evaluate()`/checkpoint/resume.
- `forge.serialization.save_model`/`load_model`/`save_checkpoint`/
  `load_checkpoint`, and the `register_module()` custom-type registry
  (already proven for a composite `Module` by `ConvAutoencoder`/`CharRNN`/
  `WordRNN`).
- `examples/mnist/dataset.py`'s `MNISTDataset` (real MNIST, already
  downloaded locally from a prior milestone).
- `examples/regression/experiment.py`/`compare.py` (M65's reproducible
  run-record workflow, already generic over any `Trainer`-shaped example).

## 4. Direct API Experiment

Per the brief's explicit instruction ("do not assume a framework gap
exists"), a direct probe script was run against the real Forge API
*before* writing any example file, covering:

1. A `TinyResidualBlock` (`Conv2d -> BatchNorm2d -> ReLU -> Conv2d ->
   BatchNorm2d -> + identity -> ReLU`) forward + `.backward()` on CPU and
   CUDA, confirming `x.grad` is populated and the output shape matches the
   input shape (identity shortcut).
2. A `ProjectionResidualBlock` (channel count and stride change, so the
   shortcut is `Conv2d(1x1, stride) -> BatchNorm2d`, not a bare identity)
   forward + backward on CPU and CUDA, confirming the projected shortcut's
   parameters receive gradient and the output shape is correct
   (`(2, 4, 8, 8) -> (2, 8, 4, 4)`).
3. `BatchNorm2d` train-vs-eval output divergence inside a residual block
   (fresh running statistics vs. batch statistics), on both devices.
4. A **finite-difference gradient check** (float64, `eps=1e-5`) against the
   reduced `TinyResidualBlock`'s input gradient: **max abs difference
   `3.23e-10`** -- the analytic and numeric gradients agree to essentially
   float64 precision.

Every step succeeded on the first attempt, with no shape error, no missing
`Backend`/`Tensor`/`Module` method, and no numerical discrepancy -- on both
CPU and CUDA (`is_cuda_available() == True` on the reference 940MX). This is
the direct, executed evidence for **Outcome A** (Section 20).

## 5. Blockers Discovered

**None.** No `forge/` change was required at any point in this milestone.

One design question was resolved by engineering judgment rather than
treated as a blocker: whether a residual block needs a projection
shortcut. Following the standard ResNet "basic block" convention,
`ResidualBlock` constructs a 1x1 `Conv2d` + `BatchNorm2d` shortcut only
when `in_channels != out_channels or stride != 1`; otherwise the shortcut
is a true identity (no extra parameters). This is ordinary architecture
design, not a framework capability gap -- `Conv2d` already supports
arbitrary `kernel_size`/`stride`/`padding`, so a 1x1 strided convolution
needed no new code.

## 6. Implementation

Files added (all in `examples/resnet/` and `tests/`, all new -- no
`forge/` production file was touched):

- `examples/resnet/__init__.py`
- `examples/resnet/model.py` -- `ResidualBlock`, `ResNetMNIST`,
  `build_model()`; both classes call
  `forge.serialization.register_module()` at import time.
- `examples/resnet/train.py` -- training/evaluation/checkpoint/resume/
  persistence script, importing `examples.mnist.dataset.MNISTDataset`/
  `examples.mnist.train.build_transform` for data and
  `examples.regression.experiment`'s run-record functions for the M65
  reproducibility workflow.
- `examples/resnet/README.md`.
- `tests/test_resnet_example_integration.py` (CPU).
- `tests/test_resnet_example_cuda_integration.py` (CUDA).
- `tests/test_resnet_autograd_validation.py` (dedicated finite-difference
  validation, Section 9).

Files modified:

- `.gitignore` -- added `examples/resnet/artifacts*/` (same pattern as
  every prior example).
- `docs/development/progress.md` -- this milestone's entry.

No `forge/` file was added, modified, or needed to be.

## 7. Architecture

```text
(N, 1, 28, 28)
    -> Conv2d(1, 8, k=3, pad=1) -> BatchNorm2d(8) -> ReLU     -> (N, 8, 28, 28)   [stem]
    -> ResidualBlock(8, 8, stride=1)   [identity shortcut]     -> (N, 8, 28, 28)
    -> ResidualBlock(8, 16, stride=2)  [projection shortcut]   -> (N, 16, 14, 14)
    -> ResidualBlock(16, 16, stride=1) [identity shortcut]     -> (N, 16, 14, 14)
    -> MaxPool2d(2)                                            -> (N, 16, 7, 7)
    -> Flatten -> Linear(784, 10)                              -> (N, 10) logits
```

~17.6k trainable parameters (17,578 exactly). Each `ResidualBlock`:

```text
identity = x  (or shortcut_bn(shortcut_conv(x)) when shape/channels change)
x = relu(bn1(conv1(x)))
x = bn2(conv2(x))
x = x + identity
x = relu(x)
```

`ResidualBlock` is one ordinary `forge.nn.Module` subclass local to this
example -- not a new core Forge primitive, per the brief's own preference
for plain module composition over a generalized residual container.

## 8. Files Changed

**Added**: `examples/resnet/__init__.py`, `model.py`, `train.py`,
`README.md`; `tests/test_resnet_example_integration.py`,
`tests/test_resnet_example_cuda_integration.py`,
`tests/test_resnet_autograd_validation.py`;
`docs/development/m66-residual-cnn.md` (this file).

**Modified**: `.gitignore` (one new ignore block);
`docs/development/progress.md` (this milestone's entry).

**Not touched**: every file under `forge/`.

## 9. Autograd Validation

Beyond Section 4's initial probe, `tests/test_resnet_autograd_validation.py`
adds permanent, dedicated coverage:

- **Finite-difference input-gradient check** for both the identity-shortcut
  and projection-shortcut `ResidualBlock` variants (float64, `eps=1e-5`,
  central difference): max abs analytic-vs-numeric difference `< 1e-4` in
  both cases.
- **Finite-difference parameter-gradient check** for a representative
  parameter from *each* branch (`conv1.weight` on the main path,
  `shortcut_conv.weight` on the projection shortcut when present) --
  confirming gradient reaches every branch's parameters correctly, not
  only the shared input tensor.
- **An isolated additive-gradient check** on the residual addition itself
  (`out = branch + identity`, `branch = x @ W`, both functions of the same
  leaf `x`): `d(out.sum())/dx` computed by the real engine is compared,
  exactly, against the *sum* of each path's independently-computed
  gradient -- the defining correctness property a residual connection's
  backward pass depends on, verified to `atol=1e-10` with no forgiving
  tolerance.

`tests/test_resnet_example_integration.py` additionally checks that
gradients are nonzero (not merely "not None") through both branches of a
full `ResidualBlock` forward/backward, for both shortcut variants.

## 10. BatchNorm Validation

`tests/test_resnet_example_integration.py` validates `BatchNorm2d` in its
first nested-inside-a-custom-Module-tree context (`model.block2.
shortcut_bn`, three levels deep):

- Train and eval mode produce different output (fresh batch statistics vs.
  frozen running statistics) -- `test_batchnorm_train_and_eval_modes_
  produce_different_output`.
- Running-mean updates during training and freezes exactly (bit-identical)
  in eval mode -- `test_batchnorm_running_stats_update_during_training_
  and_freeze_in_eval`.
- `Module.train()`/`.eval()` propagates recursively into every nested
  `ResidualBlock`'s `BatchNorm2d` instances, including a projection
  shortcut's `shortcut_bn` -- `test_train_eval_mode_propagates_
  recursively_into_nested_residual_blocks`.
- Buffers (`running_mean`/`running_var`) round-trip correctly through
  checkpoint save/resume nested inside the custom tree -- see Section 12.

No change to `forge/nn/batchnorm.py` was needed or made; M53's existing
implementation already handles this correctly.

## 11. Serialization

`ResidualBlock` and `ResNetMNIST` are both custom `Module` subclasses, so
each registers itself via `register_module()` at import time -- the same
pattern `ConvAutoencoder`/`CharRNN`/`WordRNN` already established. No
change to `forge/serialization/model.py`'s generic recursive tree walk was
needed: `_build_save_node`/`_build_load_node` already handle an arbitrary
depth of nested custom `Module`s, since they recurse over `module._modules`
regardless of type. Verified directly (before writing any test):
`save_model` -> `load_model` on a freshly-constructed `ResNetMNIST`
reproduces identical predictions (`atol=1e-6`), and the reconstructed tree
has the correct nested type structure
(`isinstance(reloaded.block2, ResidualBlock)`,
`reloaded.block2.shortcut_conv is not None`).

## 12. Checkpoint/Resume

`trainer.save_checkpoint()`/`load_checkpoint()`/`Trainer.resume()` needed
no change. Verified: model parameters, `BatchNorm2d` buffers (including a
projection shortcut's), and Adam optimizer state (`step`/`m`/`v`) all
restore correctly across a save/load round trip
(`test_checkpoint_save_and_resume_restores_state_and_continues_training`),
resumed training continues epoch/global_step counting correctly, and
`N+M`-epoch continuous training exactly matches `N`-epochs -> checkpoint ->
reload -> `M`-more-epochs to `atol=1e-5`
(`test_resume_equivalence_matches_continuous_training`, `shuffle=False`,
mirroring `examples/mnist`'s own precedent for why).

## 13. CPU Results

Reference hardware: i5-7200U, `--seed 0`, real MNIST (60,000 train /
10,000 test images), default hyperparameters (`--epochs 3 --batch-size 128
--lr 1e-3`):

| Epoch | Train loss | Train acc | Val loss | Val acc | Epoch time |
|------:|-----------:|----------:|---------:|--------:|-----------:|
| 1     | 0.2000     | 93.89%    | 0.0508   | 98.52%  | 493.95s    |
| 2     | 0.0527     | 98.31%    | 0.0384   | 98.72%  | 418.17s    |
| 3     | 0.0387     | 98.81%    | 0.0442   | 98.60%  | 399.98s    |

Total: 1312.1s for 3 epochs (137 train samples/sec average). Final test
evaluation: loss=0.0442, accuracy=98.60% -- an 88.60 percentage-point
absolute improvement over the 10.00% chance baseline. Model save/load
prediction parity verified live by `train.py`'s own end-of-run check.
Epoch 1's time reflects unrelated concurrent CPU activity on the reference
machine during this particular run (an isolated single-epoch timing
measured ~414s beforehand); epochs 2-3 (~400-420s/epoch) are the cleaner
reference, consistent with that isolated measurement.

## 14. CUDA Results

Reference hardware: GeForce 940MX (CC 5.0, driver 582.53, CUDA Toolkit
12.6), identical seed/hyperparameters:

| Epoch | Train loss | Train acc | Val loss | Val acc | Epoch time |
|------:|-----------:|----------:|---------:|--------:|-----------:|
| 1     | 0.2001     | 93.86%    | 0.0498   | 98.45%  | 79.7s      |
| 2     | 0.0525     | 98.33%    | 0.0410   | 98.71%  | 84.7s      |
| 3     | 0.0389     | 98.78%    | 0.0417   | 98.64%  | 82.8s      |

Total: 247.2s for 3 epochs (728 train samples/sec). Final test evaluation:
loss=0.0417, accuracy=98.64%. Model save/load prediction parity verified
live by `train.py`'s own end-of-run check.

| Quantity | CPU | CUDA |
|---|---:|---:|
| Train loss, epoch 1 -> 3 | 0.2000 -> 0.0387 | 0.2001 -> 0.0389 |
| Final test loss | 0.0442 | 0.0417 |
| Final test accuracy | 98.60% | 98.64% |
| Throughput | ~137 samples/sec (~437s/epoch avg) | ~728 samples/sec (~82s/epoch avg) |

CPU and CUDA agree closely (epoch-1 train loss within 0.0001, final test
accuracy within 0.04 percentage points) -- genuine end-to-end CPU/CUDA
parity, not just kernel-level. **CUDA trains ~5.3x faster than CPU** here,
consistent with `segmentation`'s (~5.9x) and `mnist`'s own CUDA advantage on
`Conv2d`-heavy workloads (this model's five real `Conv2d` layers give the
GPU enough work per batch to amortize per-batch launch/sync overhead). No
dedicated CUDA optimization was pursued, per the M59 performance policy and
this milestone's own explicit "not an optimization milestone" scope.

## 15. Reproducibility

- `forge.random.seed(args.seed)` governs `Conv2d`/`Linear`/`BatchNorm2d`
  parameter initialization at model construction; `DataLoader` shuffling
  uses its own independent `numpy.random.Generator` derived from `--seed`
  -- the same two-separate-streams policy `examples/mnist/train.py`
  documents.
- `examples/resnet/train.py` reuses (imports directly, does not
  reimplement) `examples/regression/experiment.py`'s `new_run_record()`/
  `extend_run_record()`/`save_run_record()`/`load_run_record()`, and the
  `DataLoader` shuffle-generator-state-in-`checkpoint.extra` pattern M65
  established -- no competing reproducibility mechanism was created.
- CPU/CUDA numerical agreement: epoch-1 train loss (0.2000 CPU vs. 0.2001
  CUDA) and accuracy (93.89% vs. 93.86%) match closely under independent
  runs with the same `--seed`, and `tests/test_resnet_example_cuda_
  integration.py::test_cpu_and_cuda_forward_agree_on_a_fresh_residual_
  block` verifies bit-level-close (`atol=1e-4, rtol=1e-4`) agreement for a
  single `ResidualBlock` moved from CPU to CUDA via `Module.to()`.
- Checkpoint/resume and model save/load reproducibility are both covered
  permanently by the test suite (Sections 9-12) on CPU; CUDA-side
  checkpoint/resume and persistence are covered by
  `tests/test_resnet_example_cuda_integration.py`.

## 16. Test Coverage

27 new tests, all passing:

- `tests/test_resnet_example_integration.py` (17, CPU) -- model forward
  shape, identity/projection `ResidualBlock` shape correctness, total
  parameter count, gradient flow through both residual branches (identity
  and projection variants, including an isolated-identity-path check),
  `BatchNorm2d` train/eval-mode divergence, running-statistics update/
  freeze behavior, recursive `train()`/`eval()` propagation into nested
  blocks, full-pipeline training and learning, repeated-forward
  determinism in eval mode, checkpoint save/resume + resume equivalence
  (including buffer restoration), model save/load prediction consistency
  and reconstructed-tree-type checks, device-movement idempotence, and CLI
  inspection.
- `tests/test_resnet_example_cuda_integration.py` (5, CUDA hardware-gated,
  skips cleanly without CUDA) -- full pipeline trains and learns on CUDA,
  parameter/gradient/Adam-state/buffer CUDA residency, checkpoint save/
  resume on CUDA, model persistence on CUDA, CPU-to-CUDA `ResidualBlock`
  forward parity.
- `tests/test_resnet_autograd_validation.py` (5, CPU) -- the dedicated
  finite-difference validation described in Section 9.

Full suite run after this milestone's changes: **2,008 passed** (1,981 +
27), plus one unrelated, pre-existing flaky test
(`tests/test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`)
that failed once in this full-suite run (`CUDA active bytes grew: 320 ->
288`) and passed cleanly in isolation immediately afterward -- the exact
same CUDA-allocator-measurement flake M63/M64/M65 already footnoted; no
code in `forge/backend/cuda/allocator.py` or `forge/data/prefetch.py` was
touched by this milestone. Zero regressions attributable to this
milestone, via `python -m pytest tests/ -q`.

## 17. Limitations

- The MNIST classification task is already close to saturated by simpler
  architectures (the existing plain CNN in `examples/mnist/model.py`
  reaches ~97.8% val accuracy in 2 epochs), so this workload demonstrates
  that Forge *can train* a residual architecture correctly, not that
  residual connections meaningfully improve MNIST accuracy specifically --
  MNIST is not a task where depth/residual connections are expected to
  matter much. This is consistent with the brief's own framing (Section
  7's "do not optimize the model specifically to produce an impressive
  benchmark").
- Only a 3-block, single-stage-transition residual network was built
  ("deliberately small," per the brief) -- deeper multi-stage ResNets
  (e.g. ResNet-18/34-scale) were not attempted, since nothing in this
  workload demonstrated a need for more depth to prove the architectural
  question the milestone asked.
- No pre-activation ("ResNet v2") ordering, no bottleneck blocks, no
  stochastic depth/DropPath -- all deliberately out of scope per the
  brief's explicit guardrails, and none was demonstrated necessary.

## 18. Rejected Extensions

Per the brief's explicit guardrails, the following were considered and
**not** added, since the workload never demonstrated a need for any of
them:

- **DropPath / stochastic depth** -- no evidence of overfitting or a
  depth-related training difficulty this small a network would exhibit.
- **A generalized residual-container primitive** -- `ResidualBlock` is one
  concrete, fixed-shape `Module` subclass; nothing here needed a reusable
  "wrap any two branches and add them" core abstraction.
- **Attention** -- unrelated to the residual-connection question this
  milestone investigated.
- **Bottleneck blocks (1x1 -> 3x3 -> 1x1)** -- the "basic block" (two 3x3
  convolutions) already fully exercises the residual-addition/`BatchNorm2d`
  properties this milestone needed to validate; a bottleneck variant would
  add parameters/complexity without validating anything new.
- **Grouped or dilated convolution** -- `Conv2d` already supports
  everything this architecture needs (arbitrary stride/padding); neither
  grouping nor dilation is used by any Forge workload.
- **Generalized broadcasting** -- the residual addition here is always
  between two identically-shaped tensors (the projection shortcut is
  constructed specifically to match shape); no broadcast-shaped addition
  was needed.
- **New optimizers, LR scheduling, mixed precision, distributed training**
  -- unrelated to this milestone's architectural question; `Adam` with a
  fixed learning rate trains this model to convergence already.

## 19. Practical Impact on Forge

Forge can now express and train a genuinely residual architecture --
multi-path gradient flow through an addition node, nested custom `Module`
composition, `BatchNorm2d` several levels deep in a module tree, and full
serialization/checkpoint/resume through that nested structure -- using the
exact same `Trainer`/`Adam`/persistence pipeline every other workload
already uses, with **zero new framework code**. This is direct, executed
confirmation that Forge's autograd engine's reverse-topological-order
gradient accumulation (already proven for a *parameter* shared across many
graph positions by M50's char-RNN) generalizes correctly to a *tensor
value* consumed by two branches of one forward pass -- the property every
residual/skip-connection architecture (ResNet, U-Net-with-skip-connections,
DenseNet-style concatenation aside) fundamentally depends on. It also gives
Forge's `register_module()` persistence registry its first *multi-level*
nested custom-type tree (a custom `Module` containing custom `Module`
instances, each independently registered), reinforcing that the generic
recursive save/load walk (`forge/serialization/model.py`) needed no
depth-related special-casing.

## 20. M67 Recommendation

No specific next capability is proposed as a foregone conclusion, per the
same discipline this milestone's own brief applied (and that M64/M65 asked
of their successors in turn): inspect the current architecture and
examples directly, identify whether a concrete, credible gap with a real
consumer exists, and only build if one is found. Two candidates this
milestone's own investigation explicitly did not pursue (Section 18) --
bottleneck blocks and a generalized residual container -- remain reasonable
future directions *if* a concrete workload demonstrates a real need for
either, but neither is being proposed here as a foregone conclusion.

## Final Decision

**Outcome A: residual model completed using existing Forge capabilities.**

Direct execution against the real Forge API (Section 4), a dedicated
finite-difference autograd validation (Section 9), and a full training/
checkpoint/persistence run on both CPU and CUDA (Sections 11-15) all
confirm the entire residual-CNN workload -- model, training, evaluation,
checkpointing, persistence -- is expressible with capabilities that already
existed before this milestone. No `forge/` production file was added,
modified, or needed to be.

## Suggested Commit Message

```
feat: add examples/resnet (residual CNN, zero new framework capability)

Milestone 66: a small ResNet-style image classifier (Conv2d/BatchNorm2d
stem, three ResidualBlock instances with identity-or-projection
shortcuts, MaxPool2d/Linear classifier) for MNIST, reusing the existing
MNIST dataset and the M65 reproducible-training workflow. Direct
execution against the existing Forge API -- including a finite-difference
gradient check on the residual addition -- confirmed no new production
capability was required: Tensor.__add__'s existing autograd rule already
distributes gradient correctly to both a residual block's branch and
its identity/shortcut path, and the existing serialization registry
already handles multi-level nested custom Module trees.

Reaches 98.60% test accuracy (CPU) / 98.64% (CUDA) against a 10.00%
chance baseline over 3 epochs on real MNIST; CUDA trains ~5.3x faster
than CPU. 27 new tests (17 CPU integration + 5 CUDA + 5 dedicated
autograd finite-difference checks), full suite at 2,008 passed with
zero regressions.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Srh1rtMoMj2GLD8GituQo9
```
