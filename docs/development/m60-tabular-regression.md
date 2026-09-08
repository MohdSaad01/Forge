# M60 — Tabular Regression Example (`examples/regression`, UC2)

## 1. Objective

Build Forge's third vision-named workload family -- tabular regression
(`docs/product/vision.md`'s "classification, regression, image workloads",
`docs/product/use-cases.md`'s UC2) -- to the same production standard
already established by `examples/mnist` (image classification) and
`examples/char_rnn`/`examples/word_rnn` (sequence/language modeling): a
deterministic, reproducible dataset; a real `nn.Module` model; training
through the existing `Trainer`; CPU and CUDA execution; model save/load;
checkpoint save/resume; a README matching the established convention; and
CPU/CUDA integration tests.

## 2. Why Tabular Regression Was Selected

Not selected by this milestone -- selected by M59's project-level
reassessment (`docs/development/m59-vision-and-next-stage.md`, Sections
6/10/13/18), the first milestone in the M49-M59 sequence to look at the
*project surface* rather than searching for another Tensor/kernel candidate.
M59's direct finding: Forge demonstrates only two of the vision's three
named workload classes at production quality; `examples/trainer_demo.py`
exercises a regression model but is, by the project's own established
convention (no README, no CUDA narrative, no checkpoint/resume
demonstration), a milestone-verification script, not an example. This is
the one gap in M59's entire survey whose justification is the original
product vision itself, not a measured framework-capability pressure -- and
M59 estimated it as low-risk/low-complexity, since the workload composes
entirely from already-proven primitives (`Linear`/`ReLU`/`MSELoss`/`Adam`/
`Trainer`/`DataLoader`) with no new `Tensor`/`Backend`/`nn` surface
required. This milestone's own work (Sections 3-13 below) confirms that
estimate was accurate.

## 3. Existing Forge Capabilities Used

No new framework code. The entire example is built from:

- `forge.nn.Linear`, `forge.nn.ReLU`, `forge.nn.Sequential`
- `forge.nn.MSELoss`
- `forge.optim.Adam`
- `forge.data.TensorDataset`, `forge.data.DataLoader`, `forge.data.Normalize`
- `forge.training.Trainer`, `forge.training.MeanAbsoluteError`
- `forge.serialization.save_model`/`load_model`/`save_checkpoint`
  (via `Trainer.save_checkpoint`)/`load_checkpoint`
- `forge.random.seed()`
- `forge.cli` (`model inspect`, `checkpoint inspect`)

`Linear`, `ReLU`, and `Sequential` were already registered in
`forge/serialization/registry.py`'s persistence registry as built-in types,
so this milestone required zero `register_module()` calls -- confirmed by
direct inspection before writing any example code (Section 4 of M59's
report; re-verified here directly against the current
`forge/serialization/registry.py`).

## 4. Dataset Design

`examples/regression/dataset.py` generates a fully deterministic, in-process
synthetic dataset -- no download, following `char_rnn`/`word_rnn`'s
precedent rather than MNIST's (a real external corpus was not required by
the milestone brief, and vision.md's own "Success" criterion does not name
a specific dataset).

Eight continuous features, each drawn i.i.d. `Uniform(-2, 2)`. The target:

```text
y = 3.0*x0 - 2.0*x1 + 1.5*x2 + 0.5*x3      (4 linear terms)
    + 1.2 * x4 * x5                          (interaction term)
    + 0.8 * x6**2                            (quadratic term)
    + noise,  noise ~ Normal(0, 0.5)
```

`x7` is a pure distractor -- drawn from the same distribution as every other
feature but never used by `true_function`. This was a deliberate design
choice so the dataset cannot be solved by a trivial one-to-one mapping or a
bare linear model: the interaction and quadratic terms require the `ReLU`
nonlinearity to fit, and the distractor requires the model to actually learn
which features matter rather than using all eight uniformly.

`make_datasets(n_train, n_val, n_test, seed)` draws all samples in one
`generate_raw()` call and slices contiguous train/val/test blocks (valid
because every row is an independent draw -- no separate shuffle is needed,
unlike `random_split`'s use elsewhere in Forge for order-sensitive data).
Feature standardization (`forge.data.Normalize`) is fit on the training
split's mean/std only, then applied identically to all three splits, so
validation/test statistics never leak into the transform. The target `y` is
left in its natural scale so reported MSE/RMSE stay directly interpretable.

## 5. Model Architecture

```text
(N, 8) -> Linear(8, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> (N, 1)
```

2,689 trainable parameters (confirmed by `forge model inspect` against a
real saved artifact -- Section 9). A plain `Sequential` of built-in layers,
per the brief's explicit instruction not to introduce a new regression
-specific module.

## 6. Training Design

`Trainer.fit()` is used directly and unmodified -- this workload is exactly
one forward pass per batch with no multi-timestep recurrence, so it fits
`Trainer`'s existing shape cleanly (matching M59's prediction; no
hand-written loop was needed, unlike `char_rnn`/`word_rnn`). `MSELoss` is
the training objective; `MeanAbsoluteError` is reported alongside it as a
`Trainer` metric; RMSE is derived in the example script from the reported
MSE loss (`math.sqrt`) rather than adding a new `Metric` class, since
`forge.training.metrics` already provides `MeanSquaredError`/
`MeanAbsoluteError` and the loss itself already *is* the MSE value `Trainer`
reports per epoch -- introducing a redundant `MeanSquaredError` metric
alongside an `MSELoss` loss would duplicate the same computation for no
new information.

`train.py` additionally computes and reports a trivial "predict the training
mean" baseline MSE (the training target's own variance, computed directly
with NumPy in the example script, not a new framework capability) so the
README and every run's own console output can state a concrete "how much
better than doing nothing" number, directly addressing the brief's
"demonstrate meaningful learning" requirement with a quantified comparison
rather than only a before/after loss delta.

Determinism follows `examples/mnist/train.py`'s established three-generator
policy: `forge.random.seed()` for parameter initialization,
`dataset.generate_raw()`'s own `numpy.random.default_rng()` for the
synthetic data, and `DataLoader`'s own generator for batch shuffling --
three separate, deliberately independent streams.

## 7. Serialization/Checkpoint Design

Identical mechanism to `examples/mnist/train.py`, with no persistence code
changes required:

- `trainer.save_checkpoint(path)` / `forge.load_checkpoint(path, device=...)`
  + `Trainer.resume(checkpoint)` for full training-state resume (model,
  Adam `m`/`v`/step, epoch, global_step, RNG state).
- `forge.save_model(model, path)` / `forge.load_model(path, device=...)` for
  the plain (optimizer-free) persistence path, verified by an in-script
  prediction round-trip assertion on every run.

No `register_module()` call was needed (Section 3) -- the entire model tree
(`Sequential`, `Linear`, `ReLU`) is already covered by Forge's built-in
persistence registry.

## 8. CPU Results

Reference machine: i5-7200U, 8 GB RAM (`docs/development/
development-environment.md`). Command:
`python -m examples.regression.train --epochs 40 --device cpu --seed 0`
(defaults: 4,000 train / 500 val / 500 test samples, batch size 32, Adam
`lr=1e-3`).

| Quantity | Value |
|---|---:|
| Trivial baseline MSE (predict train mean) | 24.2888 |
| Train MSE, epoch 1 -> 40 | 15.1054 -> 0.2938 |
| Val MSE, epoch 40 | 0.3795 (RMSE 0.6160) |
| Final test MSE / RMSE / MAE | 0.3354 / 0.5792 / 0.4510 |
| Reduction over trivial baseline | 98.6% |
| Wall-clock (40 epochs) | 11.7s (~13,643 train samples/sec) |

Checkpoint resume, measured directly: resuming the above 40-epoch checkpoint
for 5 more epochs continued train MSE `0.2889 -> 0.2877` (global step
`5000 -> 5625`, epoch `40 -> 45`), and the resumed run's own model-save/load
round trip still verified exact prediction match.

`forge model inspect`/`forge checkpoint inspect` were run directly against
the generated artifacts (not simulated): `model inspect` reported the exact
`Sequential(Linear, ReLU, Linear, ReLU, Linear)` tree and `Total parameters:
2689`; `checkpoint inspect` reported `Optimizer: Adam`, `Epoch: 40`,
`Global step: 5000`, and `Optimizer parameters: 6 total, 6 with saved
state`.

## 9. CUDA Results

Hardware-verified on the reference GeForce 940MX (CC 5.0, driver 582.53,
CUDA Toolkit 12.6). Command:
`python -m examples.regression.train --epochs 40 --device cuda --seed 0`
(identical hyperparameters/dataset).

| Quantity | CPU | CUDA |
|---|---:|---:|
| Train MSE, epoch 1 -> 40 | 15.1054 -> 0.2938 | 15.1054 -> 0.2934 |
| Val MSE, epoch 40 | 0.3795 | 0.3792 |
| Final test MSE | 0.3354 | 0.3354 |
| Baseline reduction | 98.6% | 98.6% |
| Wall-clock (40 epochs) | 11.7s | 38.1s |
| Throughput | ~13,643 samples/sec | ~4,196 samples/sec |

CPU/CUDA numeric parity is close (train MSE differs by 0.0004 at epoch 40;
final test MSE identical to 4 significant figures), consistent with
floating-point non-associativity between the two backends' reduction
orders, not a correctness defect. `tests/
test_regression_example_cuda_integration.py::test_cuda_residency_of_parameters_gradients_and_adam_state`
additionally confirms every `Parameter`, every `Parameter.grad`, and Adam's
`m`/`v` state remain genuinely CUDA-resident (`CUDAStorage`, never a NumPy
array) throughout training -- not just that the numbers happen to match.

**Performance observation, reported not chased (per M59's Section 12
policy):** CUDA trained at ~3.3x *lower* throughput than CPU on this
workload -- the reverse of MNIST's ~1.5-1.7x CUDA speedup. This is an
expected consequence of the workload's shape: each batch is three tiny
matmuls (`8x64`, `64x32`, `32x1`) at batch size 32, so per-batch kernel
-launch and host/device synchronization overhead dominates far more than it
does for MNIST's larger `Conv2d` kernels. This was measured, not assumed,
and is reported honestly rather than optimized: the milestone brief and
M59's performance policy both require investigation only when a measured
slowdown materially undermines the example's usefulness, and a sub-minute
40-epoch CUDA training run does not.

## 10. Convergence Evidence

Three independent pieces of evidence the model genuinely learned the
underlying function, not merely that the code executes:

1. **Large, monotone-trending loss reduction**: train MSE dropped from
   15.11 to 0.29 (a 98% relative reduction) over 40 epochs.
2. **Beats a quantified trivial baseline by a wide margin**: a model that
   always predicts the training-set mean achieves MSE 24.29 (by
   definition, the target's own variance); the trained model's test MSE is
   0.335 -- a 98.6% reduction, not merely "better than random."
3. **Approaches the theoretical noise floor**: the dataset's Gaussian
   observation noise has variance `0.5^2 = 0.25`, an irreducible lower
   bound no model can beat without overfitting the noise itself. The
   trained model's test MSE (0.335) is close to that floor, meaning it has
   recovered `true_function` almost exactly, including its interaction and
   quadratic terms, using only a `Linear`/`ReLU` MLP -- not just "some
   improvement," but recovery of the actual generating process to within
   measurement noise.

`tests/test_regression_example_integration.py::test_full_pipeline_trains_and_beats_baseline_on_cpu`
and its CUDA counterpart encode point 2 as a permanent regression check
(final MSE `< 0.3 * baseline_variance`), and both also assert every model
parameter changed value during training (ruling out a no-op optimizer step
as a false-positive "convergence").

## 11. Test Coverage

16 new tests, none inflated beyond what M60 actually introduced:

`tests/test_regression_example_integration.py` (CPU, 11 tests):
deterministic dataset reproducibility (same seed -> identical arrays,
different seed -> different arrays), documented shape/dtype, train/val/test
split disjointness and sizing, training-split feature normalization
(near-zero mean/unit std), model forward shape, full-pipeline training
-beats-baseline with parameter-change verification, checkpoint save/restore
(model state, Adam state, epoch/global_step), resume-equivalence against
continuous training (parameters match within `1e-5`), model save/load
prediction parity, and CLI inspection of real generated artifacts.

`tests/test_regression_example_cuda_integration.py` (CUDA, 5 tests, skips
cleanly via `pytest.mark.skipif(not is_cuda_available())`): full-pipeline
training-beats-baseline on CUDA, CUDA residency of parameters/gradients/Adam
state, checkpoint save/resume on CUDA, model save/load prediction parity on
CUDA, and CPU-vs-CUDA prediction parity after identical one-epoch training
(`rtol=1e-3, atol=1e-3`, following the precedent in
`tests/test_word_rnn_example_cuda_integration.py::test_cpu_and_cuda_first_epoch_loss_match`).

## 12. Verification

Performed directly, not assumed:

1. Ran `python -m examples.regression.train --epochs 40 --device cpu
   --seed 0` end-to-end from the command line (Section 8).
2. Ran the identical command with `--device cuda` (Section 9).
3. Verified genuine convergence via the three-part evidence in Section 10.
4. Verified model save/load prediction parity on both devices (the
   in-script assertion passed on every run; also covered by dedicated
   tests).
5. Verified checkpoint save/resume by actually resuming a real 40-epoch
   checkpoint for 5 more epochs and observing continued loss reduction
   (Section 8), plus the dedicated resume-equivalence test.
6. Ran the 16 new tests directly: all pass (`pytest tests/
   test_regression_example_integration.py tests/
   test_regression_example_cuda_integration.py` -> 16 passed).
7. Ran the full existing suite: **1,787 passed** (1,771 pre-M60 + 16 new),
   zero failures, zero regressions.
8. Checked `git status`/`git diff`: only `.gitignore` (one new entry),
   `docs/development/progress.md` (this milestone's entry appended), this
   report, and the new `examples/regression/`/`tests/
   test_regression_example*.py` files are new/changed by this milestone.
9. Confirmed no generated artifacts are tracked: `examples/regression/
   artifacts/`, `examples/regression/artifacts_cuda/`, and
   `examples/regression/__pycache__/` all show as git-ignored (`!!` in
   `git status --ignored`), matching the `.gitignore` entry added in
   Section 13.
10. Confirmed no `forge/` file was touched by this milestone (`git status`
    shows no changes under `forge/`).

## 13. Files Changed

New:
- `examples/regression/__init__.py`
- `examples/regression/dataset.py`
- `examples/regression/model.py`
- `examples/regression/train.py`
- `examples/regression/README.md`
- `tests/test_regression_example_integration.py`
- `tests/test_regression_example_cuda_integration.py`
- `docs/development/m60-tabular-regression.md` (this report)

Changed:
- `.gitignore` -- one new entry for `examples/regression/artifacts*/`,
  following the exact precedent of the `mnist`/`char_rnn`/`word_rnn`
  entries already present (and pre-emptively added this milestone, closing
  the gap pattern M57 had to retroactively fix for `word_rnn`).
- `docs/development/progress.md` -- this milestone's entry appended.

Not changed: any file under `forge/`, `benchmarks/`, or `docs/architecture/`.

## 14. Architecture/API Impact

None. No new `Tensor` primitive, `Backend` method, `nn.Module`, loss,
optimizer, `Trainer` feature, CUDA kernel, or persistence-registry entry was
added. This is Outcome A (existing framework sufficient) as defined by the
milestone brief -- confirmed, not assumed, by actually building the full
workload and finding no point at which the existing public API was
insufficient.

## 15. Limitations

- The dataset is synthetic, like `char_rnn`/`word_rnn`'s corpora -- MNIST
  remains the only real external dataset among Forge's four examples. This
  was an explicit, brief-sanctioned choice ("a deterministic synthetic
  tabular dataset unless the existing project requirements explicitly
  justify introducing a dependency or downloaded dataset"), not an
  oversight; no such justification was found.
- CUDA is measurably slower than CPU for this specific tiny
  architecture/batch size (Section 9). This is reported, not treated as a
  defect requiring a fix -- per M59's performance policy, an optimization
  is only warranted when a measured problem materially undermines a real
  workload's usefulness, and training still completes in well under a
  minute here.
- The distractor feature (`x7`) is not directly probed by any test (e.g. no
  test asserts the learned first-layer weight column for `x7` is small) --
  the convergence tests verify the model fits the target function well
  overall, which is sufficient evidence of learning, but a reader curious
  specifically about feature-relevance recovery would need to inspect
  `model.parameters()` manually. Not pursued further since the brief does
  not require it and it would not change any pass/fail test outcome.

## 16. Practical Impact on Forge

Closes the one gap in the entire 59-milestone history whose justification
was the original product vision document itself (M59 Section 10, item 1).
A developer can now pick any of Forge's three vision-named workload
classes -- image classification (`mnist`), sequence/language modeling
(`char_rnn`/`word_rnn`), or tabular regression (`regression`) -- and follow
that example's own README end-to-end (train, evaluate, checkpoint/resume,
persist/reload, CPU or CUDA) without reading a line of `forge/` source,
exactly matching M59's North-Star Goal (Section 16 of that report).

## 17. What This Milestone Proves About UC2

`use-cases.md`'s UC2 ("A developer supplies numeric features and targets,
defines a network, trains it, evaluates a regression metric, and predicts
numeric outputs") is now demonstrated concretely and repeatedly, at the same
standard as UC1/UC3/UC5/UC6 already were: numeric features and targets
(`dataset.py`), a network definition (`model.py`), training (`train.py` via
`Trainer.fit()`), a regression metric evaluated and reported (MSE/RMSE/MAE),
and numeric prediction (the persistence round-trip's own prediction check).
It also incidentally reinforces UC4 (custom dataset) at the `TensorDataset`
level and UC5 (CPU/CUDA execution) with a genuine, measured throughput
comparison in the opposite direction from MNIST's -- useful evidence that
Forge's CUDA backend behavior is workload-shape-dependent, not uniformly
"faster," which is itself a more honest picture of the framework's real
characteristics than either single example alone would give.

## 18. What Remains Missing, If Anything

At the example-family level: nothing UC2-specific. At the project level,
M59's other two identified gaps remain open and were explicitly out of this
milestone's scope: the top-level onboarding surface (`README.md`,
`forge/__init__.py`'s module docstring) is still stale relative to current
capability (M59 Section 7), and no versioning/CI discipline exists (M59
Section 10, items 2/4 -- both explicitly low-urgency for a solo-developer
pre-release project).

## 19. Recommendation for M61

Per the brief's own instruction, this is not automatically prescribed --
M59's North-Star Goal (three vision-named workload families, each at
production quality) is now concretely achieved, which removes the one
project-level objective that has been driving milestone selection since
M59. Two candidates exist in the evidence already gathered, neither urgent:

1. **Onboarding documentation refresh** (M59 Section 13, candidate B):
   `README.md` and `forge/__init__.py`'s module docstring are stale (last
   updated narrative-wise around Milestone 26) and now additionally do not
   mention `examples/regression`. Real value, low cost, zero architectural
   risk -- the most concrete, already-evidenced candidate on the table.
2. **No default assessment-loop milestone** (M59 Section 17, guardrail 9):
   absent a named external trigger (a new use case from the project owner,
   a real deployment/usage attempt, a workload that fails, or a material
   hardware/environment change), a fresh unscoped capability survey is
   still not the default next step. M61 should be either the documentation
   refresh above or whatever concrete need the project owner names
   directly -- not another six-candidate survey.

## Verification

Full existing suite plus this milestone's 16 new tests: **1,787 passed, 0
failed** (`python -m pytest tests/` at the end of this milestone). CPU and
CUDA runs of `examples/regression/train.py` were both executed directly on
the reference machine (not estimated); every number in Sections 8-10 is
copied from that actual console output. `git status`/`git diff` confirm no
`forge/` file was touched and no generated artifact directory is tracked.

## Suggested Commit Message

```
feat: add tabular regression example

Add a production-quality deterministic tabular regression workload
using Forge's existing Linear, ReLU, MSELoss, Adam, DataLoader,
Trainer, and serialization infrastructure, with CPU/CUDA integration
coverage and checkpoint/resume validation.
```
