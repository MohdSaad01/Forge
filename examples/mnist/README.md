# Forge MNIST Example (Milestone 20)

An end-to-end, realistic validation of Forge's framework core:

```text
MNISTDataset -> DataLoader -> forge.train_and_save() -> CNN -> CrossEntropyLoss -> Adam
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.save_model`/`save_checkpoint`/
`load_model`/`load_checkpoint`). Nothing here is new framework logic --
`examples/mnist/` only adapts the standard MNIST file format into a
`forge.data.Dataset` and assembles a small CNN from existing `forge.nn`
layers.

**Milestone 79/81.** A fresh (non-`--resume`) run trains and saves through
`forge.train_and_save()`, the single call that trains via `forge.train()`
(the high-level entry point that builds on `DataLoader`/`Trainer`) and then
saves + verifies the result via `save_and_verify()` -- see `docs/
architecture/training-engine.md`'s **Single-call high-level training** and
**Train, evaluate, persist, verify in one call** sections. `--resume` still
uses a plain `Trainer` + `trainer.resume(checkpoint)`, followed by its own
`save_and_verify()` call (`train_and_save()` has no checkpoint/resume
concept, matching `train()`), so this script is also the clearest real
example of when to reach for each API.

## Files

- `dataset.py` -- `MNISTDataset` (the IDX-format parser + `Dataset`
  implementation) and `download_mnist()`.
- `model.py` -- `build_model()`, the small CNN architecture.
- `train.py` -- the runnable example: training, evaluation, checkpointing,
  resume, and model + preprocessing + classes persistence.
- `infer.py` (Milestone 77) -- a **separate, standalone script** for
  classifying one new digit image file in a fresh process, independent of
  `train.py`. See **Standalone inference** below.

## Prerequisites

- Forge installed (`pip install -e .` from the repo root) with its `numpy`
  dependency; no other third-party packages.
- For CUDA training: a working Forge CUDA backend (see
  `docs/architecture/cuda-backend.md` and
  `docs/development/development-environment.md`).

## Obtaining the dataset

MNIST is **not** bundled with Forge and is never downloaded implicitly.
`train.py --download` (or `dataset.download_mnist(root)` directly) fetches
the four standard IDX files via plain `urllib` from a well-known public
mirror of the original files (the same mirror `torchvision` uses):

```text
https://ossci-datasets.s3.amazonaws.com/mnist/
    train-images-idx3-ubyte.gz   60,000 training images
    train-labels-idx1-ubyte.gz   60,000 training labels
    t10k-images-idx3-ubyte.gz    10,000 test images
    t10k-labels-idx1-ubyte.gz    10,000 test labels
```

Files are expected under `--data-root` (default `examples/mnist/data`),
compressed or already decompressed -- see `dataset.py`'s module docstring
for the exact format. If you already have these files (e.g. from another
project), just copy them into `--data-root` and omit `--download`.

## CPU training

```bash
python -m examples.mnist.train --download --epochs 3 --device cpu
```

Trains a small CNN (`Conv2d(1,8,3) -> ReLU -> MaxPool2d(2) -> Conv2d(8,16,3)
-> ReLU -> MaxPool2d(2) -> Flatten -> Linear(400,64) -> ReLU -> Linear(64,10)`,
~27.6k parameters) with Adam (`lr=1e-3`, default betas/eps), reports
per-epoch training loss/accuracy and validation loss/accuracy, then saves a
checkpoint and a model file under `--output-dir` (default
`examples/mnist/artifacts`).

### Expected approximate behavior (reference: this repository's CPU, i5-7200U)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|------:|-----------:|----------:|---------:|--------:|
| 1     | 0.346      | 90.1%     | 0.107    | 97.0%   |
| 2     | 0.095      | 97.1%     | 0.066    | 97.8%   |

~55-60s/epoch on the reference CPU (~1000-1200 train samples/sec). Exact
numbers will vary by hardware and are not a stability guarantee -- only
"loss decreases, accuracy well above the 10%-chance baseline" is a
guaranteed property (and is what `tests/test_mnist_example_integration.py`
checks, on a fast synthetic stand-in dataset rather than this real run).

## CUDA training

```bash
python -m examples.mnist.train --epochs 3 --device cuda
```

Identical model/optimizer/data pipeline; only `forge.train_and_save(...,
device="cuda")` differs, which moves `model` to CUDA in place before
training (`build_model()` itself is device-agnostic).
Hardware-verified on the reference GeForce 940MX (CC 5.0, driver 582.53,
CUDA Toolkit 12.6, see `docs/development/development-environment.md`):
first-epoch loss/accuracy matched the CPU run's within floating-point
tolerance (same `--seed`), confirming CPU/CUDA parity for this architecture,
and:

- every model `Parameter` remained CUDA-resident (`CUDAStorage`, never a
  NumPy array) throughout training,
- every `Parameter.grad` remained CUDA-resident,
- Adam's `m`/`v` state remained CUDA-resident,
- no `CPUBackend` numerical computation occurred (the entire
  `forward -> loss -> backward -> optimizer.step()` sequence dispatches
  through `CUDABackend`).

These four properties are enforced permanently by
`tests/test_mnist_example_cuda_integration.py::test_cuda_residency_of_parameters_gradients_and_adam_state`
(same tiny synthetic dataset as the CPU integration suite, so it runs in
under a second rather than requiring a full MNIST epoch).

### Performance observation (reference: 940MX)

On this small architecture/batch size (128), CUDA trained at roughly
1.5-1.7x the CPU sample throughput -- a real but modest speedup, not the
order-of-magnitude difference larger models/batches would show, since
kernel-launch and per-batch host/device transfer overhead is a proportionally
larger share of the work here. This is reported as observed, not tuned
further -- see Milestone 20's "Performance Observation" scope.

## Determinism

`--seed` (default `0`) calls `forge.random.seed()`, governing every draw
from Forge's default generator -- concretely, `Conv2d`/`Linear` parameter
initialization at model construction. `DataLoader` shuffling uses its own
explicit `numpy.random.Generator`, derived from `--seed` but **independent**
of Forge's default generator (see `forge/data/dataloader.py`): a given
`--seed` reproduces both model initialization and batch order, but they are
two separate streams, not one shared one. This example does not use
`Dropout`, so there is no other source of training-time randomness.

## Checkpointing and resume

Every `train.py` run saves a checkpoint (`mnist_checkpoint.forge`) capturing
model + Adam state + epoch/global_step + Forge's RNG state:

```bash
# Train from scratch for 3 epochs, saving a checkpoint.
python -m examples.mnist.train --download --epochs 3 --output-dir artifacts

# Resume from that checkpoint and continue for 2 more epochs.
python -m examples.mnist.train --resume artifacts/mnist_checkpoint.forge --epochs 2 --output-dir artifacts
```

`--resume` restores the model, Adam state, and epoch/global_step counters
via `forge.load_checkpoint()` + `Trainer.resume()` (the plain `Trainer` path
-- see the **Milestone 79/81** note above for why resuming does not go
through `forge.train_and_save()`), then continues training exactly as
`docs/architecture/persistence.md` documents -- verified by
`tests/test_mnist_example_integration.py::test_checkpoint_save_and_resume_restores_state_and_continues_training`.
A fresh run's checkpoint is written via the plain `forge.save_checkpoint()`
function instead of `Trainer.save_checkpoint()` (`epoch`/`global_step` are
derived from the `TrainingHistory` in `train_and_save()`'s returned
`TrainAndSaveResult.history`, since `forge.train()` itself tracks neither)
-- same checkpoint format either way.

**Resume equivalence.** For a small deterministic configuration with
`shuffle=False` (no caller-owned `DataLoader` generator state to restore --
see `forge.serialization.checkpoint`'s documented RNG policy), continuous
`N+M`-epoch training and `N` epochs -> checkpoint -> reload -> `M` more
epochs produce parameters matching within `1e-5` --
`tests/test_mnist_example_integration.py::test_resume_equivalence_matches_continuous_training`.

## Model persistence

`train.py` also demonstrates the plain (optimizer-free) persistence path:
`forge.training.save_and_verify()` (Milestone 78) -- called directly on
`--resume`, or as the second half of `train_and_save()` (Milestone 81) on
the fresh path -- saves the model, then reloads it fresh and confirms the
reload's prediction matches the pre-save model -- printed as `Saved +
verified model + preprocessing + classes -> ...` at the end of every run,
raising `forge.PersistenceError` instead if the reload ever disagrees. The
same property is covered on a synthetic dataset by
`tests/test_mnist_example_integration.py::test_model_persistence_preserves_predictions`
(CPU) and `tests/test_mnist_example_cuda_integration.py::test_model_persistence_preserves_predictions_on_cuda`
(CUDA).

**Preprocessing + classes (Milestone 77).** `train.py` now saves the exact
`Normalize`-based preprocessing pipeline (`forge.save_model(..., preprocessing=
build_transform())`) and the digit-index-to-label vocabulary
(`classes=["0", ..., "9"]`) alongside the model -- the same mechanism
`examples/image_folder_classification` established in Milestones 71/72,
applied here for the first time. Earlier, `build_transform()`'s pixel
scaling used `Lambda(lambda x: x * (1/255))`, which cannot be saved (`Lambda`
wraps an arbitrary Python callable); it is now `Normalize(mean=0.0,
std=255.0)`, computing exactly the same value, so this file's saved model
can be reconstructed and used to classify a brand-new digit image with no
external knowledge of MNIST's normalization convention.

## Standalone inference (fresh process, Milestone 77)

```bash
python -m examples.mnist.infer --model examples/mnist/artifacts/mnist_model.forge --image path/to/some_digit.png
```

`infer.py` is a genuinely separate script from `train.py`: it only calls
`forge.load_model()`, `forge.load_preprocessing()`, `forge.load_classes()`,
`forge.predict()`, and `forge.interpret_classification()` -- everything it
needs (architecture, weights, preprocessing, and the digit vocabulary) comes
from the one `.forge` file `train.py` wrote. The one piece of
MNIST-specific glue is decoding an arbitrary image file to a single grayscale
28x28 channel (`infer.py::_load_digit_image`) -- `forge.data.ImageFolder`'s
own single-image decode helper always produces 3-channel RGB, which does not
match a grayscale MNIST model's expected input. Covered by
`tests/test_mnist_example_integration.py::test_infer_run_classifies_a_fresh_process_style_png`
(imports `infer.run()` into the test process) and, since Milestone 80,
`tests/test_mnist_example_integration.py::
test_infer_cli_runs_in_a_genuinely_separate_process` (launches `python -m
examples.mnist.infer` as a real `subprocess` -- a genuinely separate OS
process, closing the gap between what earlier milestones' reports described
as "verified in a separate process" and what the automated test suite
actually enforced).

Note: unlike `examples/image_folder_classification`, `forge model predict`
(the CLI command) is **not** usable with an MNIST model -- that command
always decodes its `--image` argument as 3-channel RGB, but MNIST's CNN
expects 1 channel. Use `examples.mnist.infer` instead.

## CLI inspection

Every `train.py` run prints the exact commands to inspect its own output
with the Milestone 19 CLI:

```bash
python -m forge model inspect examples/mnist/artifacts/mnist_model.forge
python -m forge checkpoint inspect examples/mnist/artifacts/mnist_checkpoint.forge
```

`model inspect` reports the full `Sequential` layer tree (`Conv2d`, `ReLU`,
`MaxPool2d`, `Flatten`, `Linear`), per-parameter shapes/dtypes, and the total
parameter count (27,562 for this architecture). `checkpoint inspect`
additionally reports the Adam hyperparameters, epoch/global_step, and how
many parameters have saved optimizer state.

## Integration tests (no MNIST download required)

`tests/test_mnist_example_integration.py` (CPU) and
`tests/test_mnist_example_cuda_integration.py` (CUDA; skips cleanly without
a working CUDA backend) exercise this exact pipeline end-to-end against a
fast, deterministic *synthetic* `(N, 1, 28, 28)` dataset -- shape-compatible
with real MNIST but requiring no download -- covering: dataset/model shape,
training loss reduction and above-chance accuracy, parameter updates,
checkpoint save/restore (model state, Adam state, epoch/global_step, device
residency), resume equivalence, model save/load prediction consistency, and
CLI inspection of generated artifacts. Run them with:

```bash
python -m pytest tests/test_mnist_example_integration.py tests/test_mnist_example_cuda_integration.py
```

The full real-MNIST run above is a hardware/example validation, not part of
the mandatory test suite (downloading ~11MB of data on every test run would
make the suite slow and network-dependent for no correctness benefit beyond
what the synthetic-dataset tests already prove).
