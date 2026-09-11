# Forge Residual CNN (ResNet-Style) Example (Milestone 66)

A small ResNet-style image classifier with genuine residual connections,
brought to the same standard as `examples/mnist`, `examples/segmentation`,
and `examples/autoencoder`:

```text
MNISTDataset (examples.mnist.dataset) -> DataLoader -> Trainer
    -> ResNetMNIST (Conv2d/BatchNorm2d/ReLU residual blocks) -> CrossEntropyLoss -> Adam
```

This milestone asked a narrow question: **can Forge express and train a
residual network -- branch + shortcut addition, repeated module composition,
`BatchNorm2d`, serialization, checkpoint/resume -- using only the existing
public API?** Direct execution against the real `Tensor`/autograd/`Module`/
`Conv2d`/`BatchNorm2d`/serialization stack (finite-difference gradient
checks included) found **no framework gap at all** -- see
`docs/development/m66-residual-cnn.md` for the full investigation.
**Outcome A**: this example is built entirely from existing Forge
capabilities plus one ordinary, example-local `Module` subclass
(`ResidualBlock`), with zero changes to `forge/`.

## Files

- `model.py` -- `ResidualBlock` (identity-or-projection-shortcut residual
  block) and `ResNetMNIST` (the full classifier), plus `build_model()`.
  Both classes call `forge.serialization.register_module()` at import time,
  the same pattern `examples/autoencoder/model.py`'s `ConvAutoencoder` and
  `examples/char_rnn/model.py`'s `CharRNN` already establish.
- `train.py` -- the runnable example: training, evaluation, checkpointing,
  resume, and model persistence. Reuses `examples/mnist/dataset.py` (the
  dataset) and `examples/regression/experiment.py`/`compare.py` (the
  Milestone 65 reproducible-training workflow) rather than duplicating
  either.

No new `dataset.py`, `experiment.py`, or `compare.py` lives in this
directory -- this milestone's contribution is the architecture, not a new
dataset or a competing experiment-tracking mechanism.

## Prerequisites

Identical to `examples/mnist`: Forge installed (`pip install -e .`), the
MNIST IDX files under `examples/mnist/data` (see `examples/mnist/README.md`
for how to obtain them; `--download` fetches them automatically), and,
optionally, a working Forge CUDA backend for GPU training.

## Model

```text
(N, 1, 28, 28)
    -> Conv2d(1, 8, k=3, pad=1) -> BatchNorm2d(8) -> ReLU     -> (N, 8, 28, 28)   [stem]
    -> ResidualBlock(8, 8, stride=1)   [identity shortcut]     -> (N, 8, 28, 28)
    -> ResidualBlock(8, 16, stride=2)  [projection shortcut]   -> (N, 16, 14, 14)
    -> ResidualBlock(16, 16, stride=1) [identity shortcut]     -> (N, 16, 14, 14)
    -> MaxPool2d(2)                                            -> (N, 16, 7, 7)
    -> Flatten -> Linear(784, 10)                              -> (N, 10) logits
```

~17.6k trainable parameters. Each `ResidualBlock` is:

```text
identity = x  (or shortcut_bn(shortcut_conv(x)) when in/out channels or stride differ)
x = relu(bn1(conv1(x)))
x = bn2(conv2(x))
x = x + identity
x = relu(x)
```

-- exactly the brief's own suggested body, written as one ordinary
`forge.nn.Module` subclass local to this example. `block2`'s stride-2,
channel-doubling transition needs a projection shortcut (a 1x1 `Conv2d` +
`BatchNorm2d`); `block1`/`block3` keep the input's shape exactly and use a
true identity shortcut (no extra parameters).

## Framework impact: **zero new production capability required**

Every piece this architecture needs -- `Tensor.__add__`'s autograd rule,
`Conv2d` with arbitrary stride/padding (including 1x1 projections),
`BatchNorm2d` nested inside a custom `Module` tree, `Module.to()`/buffer
persistence, `save_model`/`load_model`/checkpoint round trips through a
multi-level custom `Module` tree -- already existed and needed no change.
In particular:

- **The residual addition's gradient correctness was the one property this
  milestone could not simply assume.** A direct probe (forward + backward +
  a finite-difference check against a reduced `ResidualBlock`) confirmed
  `Tensor.__add__`'s existing reverse-topological-order gradient
  accumulation (the same mechanism M50 already proved handles a `Parameter`
  reused across many graph positions) distributes gradient correctly to
  *both* the branch and the identity/shortcut path -- max abs analytic-vs-
  numeric difference `3.2e-10` (float64, `eps=1e-5`). See
  `tests/test_resnet_autograd_validation.py` for the permanent regression
  coverage and `docs/development/m66-residual-cnn.md` Section 9 for the
  full investigation.
- Custom nested `Module` subclasses (`ResidualBlock` inside `ResNetMNIST`)
  serialize correctly with no change to `forge/serialization/model.py`'s
  generic recursive tree walk -- each level just needs its own
  `register_module()` call, exactly like `ConvAutoencoder`'s precedent.
- `BatchNorm2d`'s buffers (`running_mean`/`running_var`) move, train/eval-
  switch, and persist correctly three levels deep in the module tree
  (`model.block2.shortcut_bn.running_mean`), with no special-casing beyond
  what M53 already built.

No speculative extension (DropPath, stochastic depth, a generalized
residual container, attention, bottleneck blocks, grouped/dilated
convolution, generalized broadcasting, a new optimizer, LR scheduling,
mixed precision, distributed training) was added -- none was demonstrated
necessary by this workload.

## CPU training

```bash
python -m examples.resnet.train --download --epochs 3 --device cpu
```

Trains on the real 60,000-image MNIST training set (10,000-image test set)
with Adam (`lr=1e-3`), reporting per-epoch training loss/accuracy and
validation loss/accuracy, then saves a checkpoint, model file, and JSON run
record under `--output-dir` (default `examples/resnet/artifacts`).

### Results (reference: this repository's CPU, i5-7200U, `--seed 0`, real MNIST)

Measured directly on this run (default hyperparameters, `--epochs 3`):

| Epoch | Train loss | Train acc | Val loss | Val acc | Epoch time |
|------:|-----------:|----------:|---------:|--------:|-----------:|
| 1     | 0.2000     | 93.89%    | 0.0508   | 98.52%  | ~494s      |
| 2     | 0.0527     | 98.31%    | 0.0384   | 98.72%  | ~418s      |
| 3     | 0.0387     | 98.81%    | 0.0442   | 98.60%  | ~400s      |

Total: 1312.1s for 3 epochs (~137 train samples/sec). Final test evaluation:
loss=0.0442, accuracy=98.60% -- an 88.60 percentage-point absolute
improvement over the 10.00% chance baseline. Epoch 1's time is inflated by
unrelated concurrent CPU activity on the reference machine during this
particular run (an isolated single-epoch timing measured ~414s); epochs 2-3
are the cleaner reference (~400-420s/epoch, ~145-150 samples/sec).

## CUDA training

```bash
python -m examples.resnet.train --epochs 3 --device cuda
```

Identical model/optimizer/data pipeline; only `Trainer(..., device="cuda")`
and `model.to("cuda")` differ. Hardware-verified on the reference GeForce
940MX (CC 5.0, driver 582.53, CUDA Toolkit 12.6):

| Epoch | Train loss | Train acc | Val loss | Val acc | Epoch time |
|------:|-----------:|----------:|---------:|--------:|-----------:|
| 1     | 0.2001     | 93.86%    | 0.0498   | 98.45%  | 79.7s      |
| 2     | 0.0525     | 98.33%    | 0.0410   | 98.71%  | 84.7s      |
| 3     | 0.0389     | 98.78%    | 0.0417   | 98.64%  | 82.8s      |

Total: 247.2s for 3 epochs (~728 train samples/sec). Final test evaluation:
loss=0.0417, accuracy=98.64%.

| Quantity | CPU | CUDA |
|---|---:|---:|
| Train loss, epoch 1 -> 3 | 0.2000 -> 0.0387 | 0.2001 -> 0.0389 |
| Final test loss | 0.0442 | 0.0417 |
| Final test accuracy | 98.60% | 98.64% |
| Throughput | ~137 samples/sec (~437s/epoch avg) | ~728 samples/sec (~82s/epoch avg) |

CPU and CUDA agree closely (epoch-1 loss within 0.0001, final accuracy
within 0.04 percentage points) -- genuine CPU/CUDA parity for this
architecture, consistent with `examples/mnist`'s and `examples/
segmentation`'s own precedent. **CUDA trains ~5.3x faster than CPU** here
-- a real, unoptimized speedup (no dedicated CUDA optimization was pursued
for this milestone, per the M59 performance policy); this model is small
enough that per-batch kernel-launch/synchronization overhead still takes a
real bite out of the GPU's advantage relative to a larger model, but the
five real `Conv2d` layers here give the GPU enough work per batch to win
clearly, similar to `segmentation`'s three-`Conv2d` result.

## Determinism

Identical policy to `examples/mnist/train.py`: `forge.random.seed(args.seed)`
governs every `Conv2d`/`Linear` parameter draw at model construction;
`DataLoader` shuffling uses its own independent `numpy.random.Generator`
derived from `--seed`. This example uses no `Dropout`, so `BatchNorm2d`'s
batch-statistics computation (deterministic given the batch contents) is the
only other training-time source of numerical behavior, and it introduces no
additional randomness.

## Checkpointing and resume

Every `train.py` run saves a checkpoint (`resnet_checkpoint.forge`)
capturing model + Adam state + `BatchNorm2d` running-mean/var buffers +
epoch/global_step + Forge's RNG state + (reusing the Milestone 65 pattern)
the `DataLoader` shuffle generator's exact stream position:

```bash
python -m examples.resnet.train --epochs 3 --output-dir artifacts
python -m examples.resnet.train --resume artifacts/resnet_checkpoint.forge --epochs 2 --output-dir artifacts
```

`--resume` restores the model (including every nested `ResidualBlock`'s
parameters and `BatchNorm2d` buffers), Adam state, and epoch/global_step
counters via `forge.load_checkpoint()` + `Trainer.resume()`, then continues
training exactly as `docs/architecture/persistence.md` documents. Resume
equivalence (`N+M` continuous epochs == `N` epochs -> checkpoint -> reload
-> `M` more epochs, matching within `1e-5`) is verified on a synthetic
dataset by
`tests/test_resnet_example_integration.py::test_resume_equivalence_matches_continuous_training`.

## Reproducible training workflow (Milestone 65, reused directly)

`train.py` imports `examples.regression.experiment`'s `new_run_record()`/
`extend_run_record()`/`save_run_record()`/`load_run_record()` and writes the
same JSON run-record format every other post-M65 example uses -- no
competing experiment-tracking code lives in this directory. Compare two
runs with the same tool every other example uses:

```bash
python -m examples.regression.compare examples/resnet/artifacts/resnet_history.json <other_run>/resnet_history.json
```

## Model persistence

`train.py` demonstrates the plain (optimizer-free) persistence path: after
training, it calls `forge.training.save_and_verify()` (Milestone 78), which
saves the model, then reloads it fresh and confirms the reload's prediction
matches the pre-save model -- printed as `Saved + verified model +
preprocessing + classes -> ...` at the end of every run, raising
`forge.PersistenceError` instead if the reload ever disagrees. Because
`ResNetMNIST`/`ResidualBlock`
are custom `Module` subclasses, this exercises the full recursive
save/load path through `register_module()`-registered custom types, not
just the built-in layer types every earlier example used.

**Preprocessing + classes (Milestone 77).** `train.py` now also saves
`examples.mnist.train.build_transform()`'s preprocessing pipeline and the
digit-index-to-label vocabulary alongside the model
(`save_model(..., preprocessing=..., classes=[...])`) -- the first time this
mechanism (Milestones 71/72) has been exercised on a custom-registered,
non-`Sequential` Module tree. Reconstructing this model in a fresh process
(`forge.load_model()`) requires that process to have already imported
`examples.resnet.model` at least once, so that its own `register_module()`
call has run -- see `docs/architecture/persistence.md`'s "Custom-module
limitations" section; there is no `examples/resnet/infer.py` (the
`examples/mnist` standalone-inference workflow already demonstrates the
same fresh-process pattern with fewer moving parts).

## CLI inspection

```bash
python -m forge model inspect examples/resnet/artifacts/resnet_model.forge
python -m forge checkpoint inspect examples/resnet/artifacts/resnet_checkpoint.forge
```

`model inspect` reports the full nested module tree (`ResNetMNIST` ->
`ResidualBlock` x3 -> `Conv2d`/`BatchNorm2d`/`ReLU`), per-parameter shapes/
dtypes, and the total parameter count (17,578).

## Autograd and BatchNorm validation

`tests/test_resnet_autograd_validation.py` is a dedicated finite-difference
gradient check for `ResidualBlock` (both identity-shortcut and
projection-shortcut variants), covering the input gradient, one
representative parameter from each branch, and an isolated additive-
gradient check on the residual addition itself. `tests/test_resnet_
example_integration.py` additionally covers `BatchNorm2d` train/eval-mode
output differences, running-statistics updates during training and freezing
in eval mode, and recursive `train()`/`eval()` propagation into nested
residual blocks (including the `shortcut_bn` of a projection block).

## Integration tests

`tests/test_resnet_example_integration.py` (CPU, 17 tests) and
`tests/test_resnet_example_cuda_integration.py` (CUDA, 5 tests; skips
cleanly without a working CUDA backend) exercise this pipeline end-to-end
against a fast, deterministic *synthetic* `(N, 1, 28, 28)` dataset (the same
convention `examples/mnist`'s own test suite uses) -- covering: residual
block forward/shape correctness (identity and projection variants),
gradient flow through both residual branches, `BatchNorm2d` train/eval
behavior and buffer persistence nested inside the custom `Module` tree,
full-pipeline training and learning, checkpoint save/resume + resume
equivalence, model save/load prediction consistency, device-movement
idempotence, CPU/CUDA prediction parity, CUDA parameter/gradient/Adam-state/
buffer residency, and CLI inspection. `tests/test_resnet_autograd_
validation.py` (CPU, 5 tests) adds the dedicated finite-difference
validation described above. Run them with:

```bash
python -m pytest tests/test_resnet_example_integration.py tests/test_resnet_example_cuda_integration.py tests/test_resnet_autograd_validation.py
```

The full real-MNIST run above is a hardware/example validation, not part of
the mandatory test suite (same rationale as `examples/mnist`'s own
README: downloading/training on real data on every test run would make the
suite slow for no correctness benefit beyond what the synthetic-dataset
tests already prove).
