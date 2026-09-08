# Forge Tabular Regression Example (Milestone 60)

An end-to-end, production-quality validation of Forge's third vision-named
workload family -- tabular regression (`docs/product/vision.md`,
`docs/product/use-cases.md`'s UC2) -- brought to the same standard as
`examples/mnist`, `examples/char_rnn`, and `examples/word_rnn`:

```text
make_datasets() -> DataLoader -> Trainer -> MLP (Linear/ReLU/Linear/ReLU/Linear)
    -> MSELoss -> Adam
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.training`, `forge.serialization`). Nothing here is new
framework logic -- `examples/regression/` only generates a deterministic
synthetic dataset and assembles a small MLP from existing `forge.nn` layers.
No new `Tensor` primitive, `nn.Module`, optimizer, or `Trainer` feature was
added for this example; the milestone confirmed `Linear`/`ReLU`/`MSELoss`/
`Adam`/`DataLoader`/`Trainer`/`forge.serialization` already fully cover the
workload (see `docs/development/m60-tabular-regression.md`).

## Files

- `dataset.py` -- deterministic synthetic tabular data generation
  (`generate_raw()`, `make_datasets()`), built on `forge.data.TensorDataset`
  and `forge.data.Normalize`.
- `model.py` -- `build_model()`, the small MLP architecture.
- `train.py` -- the runnable example: training, evaluation, checkpointing,
  resume, and model persistence.

## Prerequisites

- Forge installed (`pip install -e .` from the repo root) with its `numpy`
  dependency; no other third-party packages, and no network access.
- For CUDA training: a working Forge CUDA backend (see
  `docs/architecture/cuda-backend.md` and
  `docs/development/development-environment.md`).

## The dataset

Unlike MNIST (a real external corpus), this example follows `char_rnn`/
`word_rnn`'s precedent of generating its data entirely in-process from a
fixed seed -- no download, ever. Eight continuous features are drawn i.i.d.
`Uniform(-2, 2)`; the target mixes linear, interaction, and quadratic terms
so a plain linear model cannot fit it exactly:

```text
y = 3.0*x0 - 2.0*x1 + 1.5*x2 + 0.5*x3      (linear terms)
    + 1.2 * x4 * x5                          (interaction term)
    + 0.8 * x6**2                            (quadratic term)
    + noise,  noise ~ Normal(0, 0.5)
```

`x7` is a pure distractor feature -- drawn from the same distribution as
every other feature but never used by the target function, so a model that
has genuinely learned the relationship must implicitly down-weight it. See
`dataset.py`'s module docstring for the full generation/splitting/
normalization contract, including why `make_datasets()`'s train/val/test
split needs no separate shuffle and why feature standardization is fit on
the training split only.

## Model

```text
(N, 8) -> Linear(8, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> (N, 1)
```

~2.7k trainable parameters (2,689 exactly), composed entirely from
`forge.nn.Sequential`/`Linear`/`ReLU` -- no custom `Module` subclass, so no
`register_module()` call is needed for persistence (`Linear`/`ReLU`/
`Sequential` are already registered built-ins), exactly like
`examples/mnist/model.py`'s CNN.

## CPU training

```bash
python -m examples.regression.train --epochs 40 --device cpu
```

Trains the MLP above with Adam (`lr=1e-3`, default betas/eps) over 4,000
synthetic training samples (500 validation, 500 test), reports per-epoch
training/validation MSE and MAE, then saves a checkpoint and a model file
under `--output-dir` (default `examples/regression/artifacts`).

### Expected approximate behavior (reference: this repository's CPU, i5-7200U)

Measured directly on this run (`--seed 0`, default hyperparameters):

| Quantity | Value |
|---|---:|
| Trivial baseline MSE (predict train mean) | 24.29 |
| Train MSE, epoch 1 | 15.11 |
| Train MSE, epoch 40 | 0.294 |
| Val MSE, epoch 40 | 0.380 (RMSE 0.616) |
| Final test MSE / RMSE / MAE | 0.335 / 0.579 / 0.451 |
| Reduction over trivial baseline | 98.6% |

The dataset's irreducible noise variance is `0.5^2 = 0.25` (the Gaussian
noise added to every target), so a test MSE of 0.335 means the model has
recovered the underlying `true_function` almost to the noise floor -- this
is the concrete evidence of genuine learning, not merely "the code runs."
~0.3s/epoch on the reference CPU (~13,000-14,000 train samples/sec). Exact
numbers will vary by hardware/dataset draw and are not a stability
guarantee -- only "loss drops well below the trivial baseline" is a
guaranteed property, and is what
`tests/test_regression_example_integration.py` checks on a smaller
dataset/epoch count for test speed.

## CUDA training

```bash
python -m examples.regression.train --epochs 40 --device cuda
```

Identical model/optimizer/data pipeline; only `Trainer(..., device="cuda")`
and `model.to("cuda")` differ (`build_model()` itself is device-agnostic).
Hardware-verified on the reference GeForce 940MX (CC 5.0, driver 582.53,
CUDA Toolkit 12.6, see `docs/development/development-environment.md`):
training with the same `--seed 0` produced train MSE `15.1054 -> 0.2934`
(vs. CPU's `15.1054 -> 0.2938`) and an identical 98.6% baseline reduction --
matching within floating-point tolerance, confirming CPU/CUDA parity for
this architecture. `tests/test_regression_example_cuda_integration.py`
additionally verifies, permanently:

- every model `Parameter` remains CUDA-resident (`CUDAStorage`, never a
  NumPy array) throughout training,
- every `Parameter.grad` remains CUDA-resident,
- Adam's `m`/`v` state remains CUDA-resident,
- a CPU-trained and a CUDA-trained model (identical seed/data/one epoch)
  produce matching predictions on a held-out query batch within
  `rtol=1e-3, atol=1e-3`.

### Performance observation (reference: 940MX)

On this small architecture (2,689 parameters) and batch size (32), CUDA
trained at roughly **3.3x lower** sample throughput than CPU (~4,200 vs.
~13,600 samples/sec) -- the reverse of MNIST's CUDA speedup. This is an
expected consequence of the workload's shape, not a regression: each batch
is a handful of tiny matmuls (`8x64`, `64x32`, `32x1`), so per-batch
kernel-launch and host/device synchronization overhead dominates total time
far more than it does for MNIST's `Conv2d`-heavy, larger-batch workload.
Per Milestone 59's performance policy (`docs/development/
m59-vision-and-next-stage.md` Section 12) and the M60 brief's explicit
instruction, this is reported as observed and **not** optimized further --
the example's purpose is demonstrating a correct, complete CPU/CUDA
regression workflow, not maximizing this specific tiny model's CUDA
throughput, and the slowdown does not undermine that purpose (training still
completes in under a minute on the reference machine).

## Determinism

`--seed` (default `0`) governs three independent, deliberately separate
generators, matching `examples/mnist/train.py`'s documented policy:

1. `forge.random.seed(args.seed)` -- `Linear` parameter initialization at
   model construction.
2. `examples.regression.dataset.generate_raw()`'s own
   `numpy.random.default_rng(args.seed)` -- the synthetic features/noise.
3. `DataLoader`'s explicit `numpy.random.Generator` -- batch shuffling.

A given `--seed` reproduces model initialization, the dataset, and batch
order identically, but the three are separate streams, not one shared one.
This example uses no `Dropout`, so there is no other source of
training-time randomness.

## Checkpointing and resume

Every `train.py` run saves a checkpoint (`regression_checkpoint.forge`)
capturing model + Adam state + epoch/global_step + Forge's RNG state:

```bash
# Train from scratch for 40 epochs, saving a checkpoint.
python -m examples.regression.train --epochs 40 --output-dir artifacts

# Resume from that checkpoint and continue for 10 more epochs.
python -m examples.regression.train --resume artifacts/regression_checkpoint.forge --epochs 10 --output-dir artifacts
```

`--resume` restores the model, Adam state, and epoch/global_step counters
via `forge.load_checkpoint()` + `Trainer.resume()`, then continues training
exactly as `docs/architecture/persistence.md` documents -- verified on this
repository (resuming a 40-epoch checkpoint for 5 more epochs continued train
MSE from `0.2938 -> 0.2877`, global step `5000 -> 5625`) and by
`tests/test_regression_example_integration.py::test_checkpoint_save_and_resume_restores_state_and_continues_training`.

**Resume equivalence.** For a small deterministic configuration with
`shuffle=False`, continuous `N+M`-epoch training and `N` epochs ->
checkpoint -> reload -> `M` more epochs produce parameters matching within
`1e-5` --
`tests/test_regression_example_integration.py::test_resume_equivalence_matches_continuous_training`.

## Model persistence

`train.py` also demonstrates the plain (optimizer-free) persistence path:
after training, it records a prediction, calls `forge.save_model()`, reloads
with `forge.load_model()`, and asserts the reloaded model reproduces the
same prediction -- printed as `Verified: reloaded model reproduces the
pre-save prediction.` at the end of every run. The same property is covered
by `tests/test_regression_example_integration.py::test_model_persistence_preserves_predictions`
(CPU) and `tests/test_regression_example_cuda_integration.py::test_model_persistence_preserves_predictions_on_cuda`
(CUDA).

## CLI inspection

Every `train.py` run prints the exact commands to inspect its own output
with the Milestone 19 CLI:

```bash
python -m forge model inspect examples/regression/artifacts/regression_model.forge
python -m forge checkpoint inspect examples/regression/artifacts/regression_checkpoint.forge
```

`model inspect` reports the full `Sequential` layer tree (`Linear`, `ReLU`),
per-parameter shapes/dtypes, and the total parameter count (2,689 for this
architecture). `checkpoint inspect` additionally reports the Adam
hyperparameters, epoch/global_step, and how many parameters have saved
optimizer state.

## Integration tests

`tests/test_regression_example_integration.py` (CPU) and
`tests/test_regression_example_cuda_integration.py` (CUDA; skips cleanly
without a working CUDA backend) exercise this exact pipeline end-to-end,
covering: deterministic dataset generation/reproducibility, train/val/test
split disjointness and feature normalization, model construction/forward
shape, training loss reduction well below a trivial predict-the-mean
baseline, parameter updates, checkpoint save/restore (model state, Adam
state, epoch/global_step, device residency), resume equivalence, model
save/load prediction consistency, CPU/CUDA prediction parity after
identical training, and CLI inspection of generated artifacts. Run them
with:

```bash
python -m pytest tests/test_regression_example_integration.py tests/test_regression_example_cuda_integration.py
```

This example's dataset is synthetic and fast to generate (no download), so
-- unlike MNIST -- the mandatory test suite and the README's reported
numbers use the same generation code (`make_datasets()`), just at different
sample counts/epoch budgets for test speed.
