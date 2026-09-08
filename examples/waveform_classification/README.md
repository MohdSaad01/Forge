# Forge Waveform Classification Example (Milestone 62)

An end-to-end demonstration of Forge's newest layer family -- 1D temporal
convolution (`nn.Conv1d`/`nn.MaxPool1d`, added this milestone) -- brought to
the same standard as `examples/mnist`, `examples/char_rnn`/`word_rnn`, and
`examples/regression`:

```text
make_datasets() -> DataLoader -> Trainer -> CNN (Conv1d/ReLU/MaxPool1d/Flatten/Linear)
    -> CrossEntropyLoss -> Adam
```

This is Forge's `UC1` ("train a classifier", `docs/product/use-cases.md`)
use case expressed with a model family none of the other four examples
cover: `mnist` classifies with a 2D `Conv2d` CNN, `char_rnn`/`word_rnn`
generate sequences with `RNNCell`, and `regression` predicts a continuous
target with a plain MLP. This example classifies fixed-length 1D time series
with a 1D convolutional network -- the standard architecture family for
real workloads like sensor/audio/ECG classification.

## Files

- `dataset.py` -- deterministic synthetic waveform data generation
  (`generate_raw()`, `make_datasets()`), built on `forge.data.TensorDataset`.
- `model.py` -- `build_model()`, the small 1D-CNN architecture.
- `train.py` -- the runnable example: training, evaluation, checkpointing,
  resume, and model persistence.

## Prerequisites

- Forge installed (`pip install -e .` from the repo root) with its `numpy`
  dependency; no other third-party packages, and no network access.
- For CUDA training: a working Forge CUDA backend (see
  `docs/architecture/cuda-backend.md` and
  `docs/development/development-environment.md`).

## The dataset

Follows `char_rnn`/`word_rnn`/`regression`'s precedent of generating data
entirely in-process from a fixed seed -- no download, ever. Each sample is a
length-64, single-channel time series belonging to one of 4 waveform shapes
(sine, square, sawtooth, triangle), drawn with a random frequency, phase, and
amplitude, then corrupted with additive Gaussian noise:

```text
x(t) = amplitude * shape(2*pi*freq*t + phase) + noise,  noise ~ Normal(0, 0.25)
```

The random per-sample phase/frequency means a model cannot memorize one
fixed waveform per class -- it must learn genuine, shift-invariant local
shape features, exactly what `Conv1d`'s sliding kernel is suited to extract.
See `dataset.py`'s module docstring for the full generation contract.

Trivial baseline: uniform random guessing over 4 classes = **25% accuracy**.

## Model

```text
(N, 1, 64) -> Conv1d(1, 16, k=7, pad=3) -> ReLU -> MaxPool1d(2)   -> (N, 16, 32)
           -> Conv1d(16, 32, k=5, pad=2) -> ReLU -> MaxPool1d(2)  -> (N, 32, 16)
           -> Flatten -> Linear(512, 64) -> ReLU -> Linear(64, 4) -> (N, 4)
```

~35k trainable parameters, composed entirely from `forge.nn.Sequential`/
`Conv1d`/`ReLU`/`MaxPool1d`/`Flatten`/`Linear` -- no custom `Module`
subclass, so no `register_module()` call is needed for persistence (every
layer here, including this milestone's `Conv1d`/`MaxPool1d`, is already a
registered built-in), exactly like `examples/mnist/model.py`'s CNN.

## How `Conv1d`/`MaxPool1d` are implemented

Both are thin compositions -- reshape the `(N, C, L)` input to a dummy
`(N, C, 1, L)` 4D tensor, dispatch to the existing `Conv2d`/`MaxPool2d`
machinery, reshape the result back down -- rather than new `Backend` methods
or CUDA kernels. `Tensor.reshape` is already real and differentiable on both
CPU and CUDA, so this needed no new low-level code and inherits `Conv2d`'s
already CPU/CUDA-tested (including hardware-optimized CUDA kernel)
correctness automatically. See `forge/nn/conv.py`'s `Conv1d` docstring for
the full rationale.

## CPU training

```bash
python -m examples.waveform_classification.train --epochs 15 --device cpu
```

Trains the CNN above with Adam (`lr=1e-3`) over 4,000 synthetic training
samples (500 validation, 500 test), reports per-epoch training/validation
loss and accuracy, then saves a checkpoint and a model file under
`--output-dir` (default `examples/waveform_classification/artifacts`).

### Expected approximate behavior (reference: this repository's CPU, i5-7200U)

Measured directly on this run (`--seed 0`, default hyperparameters):

| Quantity | Value |
|---|---:|
| Trivial baseline accuracy (uniform random over 4 classes) | 25.00% |
| Train loss, epoch 1 | 0.8386 |
| Train loss, epoch 15 | 0.1612 |
| Val accuracy, epoch 15 | 91.00% |
| Final test loss / accuracy | 0.1960 / 90.80% |

~1.35s/epoch on the reference CPU (~2,900-3,000 train samples/sec). Exact
numbers will vary by hardware/dataset draw and are not a stability
guarantee -- only "accuracy well above the trivial baseline" is a guaranteed
property, and is what
`tests/test_waveform_classification_example_integration.py` checks on a
smaller dataset/epoch count for test speed.

## CUDA training

```bash
python -m examples.waveform_classification.train --epochs 15 --device cuda
```

Identical model/optimizer/data pipeline; only `Trainer(..., device="cuda")`
and `model.to("cuda")` differ (`build_model()` itself is device-agnostic).
Hardware-verified on the reference GeForce 940MX (CC 5.0, driver 582.53,
CUDA Toolkit 12.6, see `docs/development/development-environment.md`):
training with the same `--seed 0` produced train loss `0.8386 -> 0.1545` and
final test accuracy `91.80%` (vs. CPU's `90.80%`) -- both well above the 25%
baseline and consistent with CPU within normal run-to-run training
variance. `tests/test_waveform_classification_example_cuda_integration.py`
additionally verifies, permanently:

- every model `Parameter` remains CUDA-resident (`CUDAStorage`, never a
  NumPy array) throughout training,
- every `Parameter.grad` remains CUDA-resident,
- Adam's `m`/`v` state remains CUDA-resident,
- a CPU-trained and a CUDA-trained model (identical seed/data/one epoch)
  produce matching predictions on a held-out query batch within
  `rtol=1e-3, atol=1e-3`.

Training throughput was comparable between CPU and CUDA on this small
architecture and batch size (~2,900-3,100 samples/sec either way) -- neither
a CUDA win nor a regression worth chasing further, consistent with
`regression`'s documented finding that small-batch, small-model workloads on
the reference 940MX are dominated by per-batch kernel-launch/sync overhead
rather than raw compute (`examples/regression/README.md`'s **Performance
observation** section). No further optimization was pursued, per Milestone
59's performance policy.

## Determinism

`--seed` (default `0`) governs three independent, deliberately separate
generators, matching `examples/regression/train.py`'s documented policy:

1. `forge.random.seed(args.seed)` -- `Conv1d`/`Linear` parameter
   initialization at model construction.
2. `examples.waveform_classification.dataset.generate_raw()`'s own
   `numpy.random.default_rng(args.seed)` -- the synthetic waveforms.
3. `DataLoader`'s explicit `numpy.random.Generator` -- batch shuffling.

## Checkpointing, resume, and model persistence

Identical mechanics to `examples/regression/train.py` (see that README's
**Checkpointing and resume** / **Model persistence** sections for the full
contract) applied to this pipeline:

```bash
python -m examples.waveform_classification.train --epochs 15 --output-dir artifacts
python -m examples.waveform_classification.train --resume artifacts/waveform_checkpoint.forge --epochs 5 --output-dir artifacts
```

Covered permanently by
`tests/test_waveform_classification_example_integration.py`'s checkpoint
save/resume, resume-equivalence, and model-persistence tests.

## CLI inspection

```bash
python -m forge model inspect examples/waveform_classification/artifacts/waveform_model.forge
python -m forge checkpoint inspect examples/waveform_classification/artifacts/waveform_checkpoint.forge
```

## Integration tests

`tests/test_waveform_classification_example_integration.py` (CPU) and
`tests/test_waveform_classification_example_cuda_integration.py` (CUDA;
skips cleanly without a working CUDA backend) exercise this exact pipeline
end-to-end. Run them with:

```bash
python -m pytest tests/test_waveform_classification_example_integration.py tests/test_waveform_classification_example_cuda_integration.py
```

`Conv1d`/`MaxPool1d` themselves are covered separately by
`tests/test_conv1d.py`, `tests/test_maxpool1d.py` (CPU) and
`tests/test_cuda_conv1d.py` (CUDA).
