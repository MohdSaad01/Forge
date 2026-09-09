# M68 — `predict()`: A Standalone Inference Path, and the Dependency Map Behind It

## 1. Executive Summary

M68's brief asked for a project-level engineering decision, not another
capability survey: read the vision/requirements/use-case documents, the
current public API, every example, and M59-M67's own history; build a
dependency map from "Forge can train models" to "a developer can use Forge
to solve a task"; and implement exactly one foundational, reusable piece of
that map, proven against a real consumer.

The investigation (Sections 3-7) found that most of the dependency chain
`dataset -> model -> train -> evaluate -> save -> load -> predict` is
already solid, reusable, public API, exercised by real examples:
`Dataset`/`TensorDataset`/`Subset`/`random_split`/`DataLoader`/transforms
(data), `Trainer.fit`/`evaluate` (train/evaluate), and
`save_model`/`load_model`/`save_checkpoint`/`load_checkpoint` (persistence)
all already satisfy their layer. The one layer that does not: **the last
step, predict**. Every single-forward-per-step example that reaches "load a
saved model and run it on new data" -- `mnist`, `regression`, `resnet`,
`segmentation`, `autoencoder`, `waveform_classification`, six of Forge's
nine examples -- had independently hand-rolled the identical four-line
sequence to do it, because `Trainer` (the only orchestration object Forge
has) requires a `Loss` and an `Optimizer` neither exists nor is needed once
a model is trained.

M68 implements `predict()` (`forge/training/inference.py`, re-exported as
`forge.predict`/`forge.training.predict`): a small, free function that puts
a model in eval mode, moves input to the right device, runs the forward
pass under `no_grad()`, restores training mode, and returns a CPU `Tensor`
-- accepting either a single `Tensor` or an iterable of batches (a
`DataLoader`). It replaces the duplicated block in all six affected
examples, is covered by 18 new tests (15 CPU, 3 CUDA, hardware-verified on
the reference 940MX), and closes the "receive a useful prediction" step of
`docs/product/vision.md`'s own core workflow without requiring a `Trainer`
to exist at prediction time -- matching the described end state
(`load_model() -> give it new data -> receive a prediction`) exactly.

Zero new `Tensor`/`Backend`/`nn` primitives. Full suite: 2,057 -> 2,075
tests collected, 2,074 passed, 1 pre-existing unrelated flake (reproduced
passing in isolation -- see Section 16).

## 2. Long-Term Product Goal

`docs/product/vision.md`: "A developer can use Forge to build and train
real small models, evaluate them, persist them, reload them, perform
inference, and execute supported workloads on CPU and CUDA where
available." The M68 brief operationalizes this as a nine-step workflow:
select/define a model, provide a dataset, configure training, train,
evaluate, save, load, give it new data, receive a prediction. M68 does not
build the eventual image-folder/cat-dog dataset experience -- it identifies
where that experience's dependency chain is actually weak today and closes
the highest-value link.

## 3. Current Forge State (verified this milestone, not carried forward)

Read fresh this milestone: `docs/product/vision.md`/`requirements.md`/
`use-cases.md`; `forge/__init__.py`; `forge/data/` (`dataset.py`,
`dataloader.py`, `transforms.py`, `prefetch.py`); `forge/training/`
(`trainer.py`, `metrics.py`); `forge/serialization/model.py`; `forge/cli/`
(`main.py`, `model.py`); every `examples/*/train.py`; and M59/M61's own
reports (the two most recent project-level milestones, both explicitly
about the framework's usability rather than a single primitive).

- **Data.** `Dataset`/`TensorDataset`/`Subset`, `random_split` (seeded,
  deterministic), `DataLoader` (batching/shuffling), `transforms.Compose`/
  `Normalize`/`Reshape`/`Lambda`, `CUDAPrefetchLoader`. Fully public,
  fully reusable, exercised by all nine examples.
- **Training/evaluation.** `Trainer(model, loss_fn, optimizer, device,
  metrics)` with `fit()`/`evaluate()`, checkpoint save/resume
  (`Trainer.save_checkpoint`/`Trainer.resume`), a `TrainingHistory`. Solid
  and reusable for every "one forward call per training step" model (six of
  Forge's nine examples fit this shape); two hand-written-loop examples
  (`char_rnn`, `word_rnn`, plus `long_range_recall`) intentionally bypass
  `Trainer` for their per-timestep recurrence, a previously-documented,
  deliberate boundary (M50/M54/M55/M59).
- **Persistence.** `save_model`/`load_model` (architecture + parameter
  values, via an explicit module-type registry, versioned format) and
  `save_checkpoint`/`load_checkpoint` (adds optimizer state + RNG state for
  exact resume). Both are real, public, and exercised by every example's
  own persistence round-trip check.
- **CLI.** `forge model inspect/convert`, `forge checkpoint inspect/
  convert`, `forge benchmark` -- a thin, read-only/explicit-conversion
  adapter over the same public API, deliberately not a training interface
  (`train`/`evaluate`/`predict` subcommands were explicitly declined in
  M19/M52/M59 pending a config/YAML system Forge has chosen not to build).
- **Inference.** No dedicated API existed before this milestone. See
  Section 5.

## 4. End-State Workflow

```text
dataset -> preprocessing -> DataLoader -> model -> training configuration
    -> train (Trainer.fit) -> evaluate (Trainer.evaluate)
    -> save (save_model/save_checkpoint) -> load (load_model/load_checkpoint)
    -> predict (new: forge.predict) -> use the prediction
```

## 5. Dependency Map

| Layer | Forge provides | Public/reusable? | Real example? | Missing piece | Foundational or convenience | Depends on | Build now? |
|---|---|---|---|---|---|---|---|
| Data acquisition/loading | `Dataset`, `TensorDataset` | Yes | Yes (9/9) | ImageFolder-style directory loader | Convenience today (no real consumer needs directory scanning yet) | Dataset abstraction (already stable) | No -- no consumer |
| Preprocessing/transforms | `Compose`/`Normalize`/`Reshape`/`Lambda` | Yes | Yes | Image decode/augmentation | Convenience | Data loading | No -- no consumer |
| Train/val/test split | `random_split`, `Subset` | Yes | Yes (`regression`, `trainer_demo`) | None found | -- | Dataset | Already done |
| Model definition | `nn.Module` + 12 layer types | Yes | Yes (9/9) | None found | -- | Tensor/autograd | Already done |
| Training configuration | argparse per example, `Trainer` constructor kwargs | Yes | Yes | A config/YAML system | Convenience (explicitly declined, M19/M52/M59) | Trainer | No -- declined non-goal |
| Training loop | `Trainer.fit()` | Yes | Yes (6/9; 3 use a documented hand-written loop) | None found | -- | Data, model, loss, optimizer | Already done |
| Metrics/evaluation | `Trainer.evaluate()`, `Metric` subclasses | Yes | Yes | None found | -- | Training loop | Already done |
| Checkpointing | `Trainer.save_checkpoint`/`resume` | Yes | Yes (`regression`'s M65 workflow) | None found | -- | Training loop, persistence | Already done |
| Model serialization | `save_model`/`load_model` | Yes | Yes (9/9) | None found | -- | Module registry | Already done |
| **Inference/prediction** | **nothing before this milestone** | **No** | **No -- hand-rolled 6x** | **A standalone `predict()`** | **Foundational** | **Module, `no_grad`, device transfer** | **Yes -- this milestone** |
| User-facing workflow (CLI train/evaluate/predict, high-level `.fit()`/`.predict()` on `Module`) | Read-only/convert-only CLI | Partial | Partial | `forge predict` CLI, `Module.predict()` | Convenience, and arguably wrong shape (see Section 12) | Inference API (this milestone) | No -- deferred, see Section 12 |

## 6. Existing Capabilities (verified by direct execution this milestone)

`python -m pytest tests/test_mnist_example_integration.py ...` (all six
affected examples' existing integration tests) was run before any change,
confirming the pre-M68 baseline actually passed (see Section 16's before/
after counts). `forge/data/dataset.py`, `forge/training/trainer.py`, and
`forge/serialization/model.py` were read in full (not skimmed) to confirm
`random_split`, `Trainer.evaluate`, and `load_model`'s device-placement
policy all already work exactly as `docs/architecture/*` describes -- no
surprises, no stale documentation found in any of the three.

## 7. Missing Capabilities and the Real Gap

A `grep` for `predict\(|inference` across the repository found the word
"predict"/"inference" only in docstrings and comments -- no function named
`predict` existed anywhere in `forge/`. Reading every example's `train.py`
directly (not just grepping) found the same block, byte-for-byte
near-identical, in six files:

```python
with no_grad():
    pre_save_pred = model(query_x).to("cpu").numpy()
reloaded = load_model(str(model_path), device=args.device)
with no_grad():
    post_load_pred = reloaded(query_x).to("cpu").numpy()
```

(`examples/mnist/train.py:157-161`, `examples/regression/train.py:182-186`,
`examples/resnet/train.py:183-187`, `examples/segmentation/train.py:
164-168`, `examples/autoencoder/train.py:220-224`,
`examples/waveform_classification/train.py:139-143`, all before this
milestone's changes.) This is the exact friction the M68 brief's Candidate
C predicted: "If [inference] is fragmented across examples or requires
knowledge of internal Tensor/device mechanics, it may be a high-value
product gap" -- confirmed by direct repository evidence, not assumed.

`char_rnn`/`word_rnn`/`long_range_recall` do not share this block (their
`generate()` functions call `model.step()` once per output timestep, not
`model(x)` once), consistent with M50/M59's own established "two training
patterns, one clear boundary" finding -- `predict()` is scoped to the
`Trainer`-shaped six, not forced onto the hand-written-loop three.

## 8. Foundational vs. Convenience Gaps

**Foundational** (built this milestone): a standalone inference path
decoupled from `Trainer`. Every future single-forward-per-step example
(and the eventual image-folder workload, once it exists) needs this same
"load a model, run it on new data, get a usable result" step; building it
now, generically, means no future example re-derives it.

**Convenience, correctly deferred** (Section 12): a `forge predict` CLI
subcommand, and `Module.predict()`/`model.fit()`-style methods directly on
`Module`. Both are real ideas with real future value, but neither has a
concrete consumer yet and the brief explicitly forbids combining "CLI
tooling" and "inference APIs" work in the same milestone (see Section 12).

**Non-gaps, re-confirmed, not rebuilt**: `ImageFolder`-style dataset
loading, image decoding, augmentation (Candidate A) -- no example needs
directory-based ingestion yet, and `TensorDataset`/custom `Dataset`
subclasses (MNIST's own dataset.py, `waveform_classification`'s synthetic
generator) already cover every current workload's actual data shape.
Dataset splitting (Candidate D) is already solved (`random_split`).
Training/evaluation lifecycle (Candidate B) is already sufficient --
`Trainer.fit`/`evaluate` were not changed, and no `model.fit()`/
`model.evaluate()` was added directly to `Module` (the brief explicitly
warns against inventing that shape without evidence, and this milestone
found none: `Trainer`'s API is not what was blocking anyone).

## 9. M68 Objective Selection

**Objective**: implement `predict(model, inputs, device=None) -> Tensor`,
a standalone inference function in `forge/training/inference.py`,
re-exported as `forge.predict` and `forge.training.predict`; retrofit
every example exhibiting the duplicated pattern (Section 7) to use it;
document it in `docs/architecture/training-engine.md`.

Checked against the brief's five selection criteria:

- **A. Advances the end-state.** Directly implements the "load it -> give
  it new data -> receive a useful prediction" steps of
  `docs/product/vision.md`'s own core workflow -- the one step of that
  workflow with no dedicated API before this milestone.
- **B. Concrete consumer.** Six real, already-existing, already-tested
  examples (Section 7), not a synthetic demonstration.
- **C. Reusable.** One function serves classification (`mnist`, `resnet`,
  `waveform_classification`), regression (`regression`), dense prediction
  (`segmentation`), and reconstruction (`autoencoder`) -- five materially
  different task shapes, proving it is not an example-specific hack.
- **D. Clear dependency position.** Sits directly on top of `Module`/
  `no_grad`/`Device` (all stable, unchanged) and directly below every
  future example that will need "run this trained model" -- exactly the
  "smallest reusable solution that closes the gap" the brief asks for.
- **E. Measurable acceptance criteria.** Stated up front, before writing
  `predict()`'s implementation (Section 10) -- see Section 11.

## 10. Evidence of the Current Gap (measured before implementation)

Before writing `forge/training/inference.py`, the six duplicated blocks
listed in Section 7 were located and diffed against each other directly
(`grep -n "with no_grad" -A3` across all `examples/*/train.py`, and reading
each file's imports) to confirm they were not superficially similar but
functionally identical -- they are: same four-line shape, same
`.to(device)`/`.to("cpu")`/`.numpy()` sequence, same purpose (verify a
reload reproduces a pre-save prediction), differing only in tensor shape
and variable names. This is the throwaway-probe evidence the brief asks
for, done via the examples themselves rather than a separate script, since
the examples already embodied the friction directly.

## 11. Acceptance Criteria (written before implementation)

1. `predict(model, x)` for a single `Tensor` returns a value identical
   (within floating-point tolerance) to the hand-rolled
   `with no_grad(): model(x.to(device)).to("cpu")` sequence it replaces.
2. `predict(model, loader)` for an iterable of batches returns the
   per-batch outputs concatenated in iteration order, identical to running
   the same sequence per batch and concatenating manually.
3. `model.training` is restored to its pre-call value afterward, even on
   an exception mid-inference.
4. The model is genuinely in eval mode during the call (proven by a
   `Dropout`-bearing model producing identical output across two
   consecutive `predict()` calls -- eval-mode dropout is a deterministic
   identity).
5. The returned `Tensor` always lands on `"cpu"`, has `requires_grad ==
   False`, and needs no further device/autograd handling from the caller.
6. Device is inferred from `model.device` by default; an explicit
   `device=` override is accepted; a `Parameter`-less model defaults to
   `"cpu"`.
7. A non-`Module` `model`, a non-`Tensor` batch element, or an empty
   iterable each raise a specific, typed `ForgeError` subclass.
8. Every one of the six examples in Section 7 passes its existing CPU
   *and* CUDA integration tests after retrofitting, with zero behavioral
   change to what each test actually verifies.
9. Zero new `Tensor`/`Backend` primitives; zero regressions in the full
   suite.

All nine were verified directly (Sections 13-16) before this report was
written, not assumed from the design.

## 12. Design

`predict()` is a **free function**, not a `Trainer` method and not a
`Module.predict()`:

- **Not a `Trainer` method.** `Trainer.__init__` requires a `Loss` and an
  `Optimizer` -- both meaningless for pure inference. Requiring a caller
  to construct a full `Trainer` (with a throwaway loss/optimizer) just to
  call `.predict()` once would be exactly the kind of awkward,
  misleading-capability API `docs/product/requirements.md`'s non-functional
  requirements warn against ("avoid hidden magic and misleading capability
  claims").
- **Not `Module.predict()`.** Attaching orchestration behavior (device
  placement, mode switching, autograd suspension) directly to `Module`
  would blur the boundary `docs/architecture/modules.md` and `Trainer`
  itself already draw between "what a `Module` computes" and "how it's
  orchestrated" -- the same reasoning the M68 brief's own Candidate B
  section warns against ("Do not invent `model.fit()/evaluate()/predict()`
  ... just because it resembles other frameworks").
- **Lives in `forge/training/`**, alongside `Trainer`/`metrics.py`, because
  that subpackage already owns "run a model over data with device/mode
  bookkeeping" as a concept -- `predict()` is a sibling to `Trainer`, not a
  new top-level subsystem.

**Signature**: `predict(model: Module, inputs: Tensor | Iterable[Any],
device: str | Device | None = None) -> Tensor`.

**Two input shapes**, chosen because both are real (Section 7's blocks are
all single-`Tensor`; a full-test-set inference pass over a `DataLoader` is
the obvious next need any future consumer would hit):
- A single `Tensor` -> one forward pass, one `Tensor` result.
- An iterable of batches (anything yielding a bare `Tensor` or a
  `(features, ...)` tuple -- matching `DataLoader`'s and `Trainer`'s own
  batch convention) -> per-batch outputs, concatenated.

**Concatenation happens on host NumPy arrays**, not inside the
differentiable `Tensor` core: each batch's output is materialized via
`.to("cpu").numpy()`, collected into a list, and joined with
`np.concatenate` only at the end, then wrapped back into one `Tensor`. This
was a deliberate choice against adding a `Tensor.cat` primitive: no
existing Forge op is more than a real consumer's exact need (M49's
`docs/development/m49-capability-assessment.md` already rejected an
unrelated generic op for the identical reason), and concatenating an
already-computed, non-differentiable inference result is not a
computation that has ever needed autograd support -- the same precedent
`Metric._as_numpy`'s own host-side reductions already establish for a
different consumer.

**Device resolution** mirrors `Trainer._check_model_device`'s existing
exemption for a `Parameter`-less model (`Module.device is None`): default
to `model.device` when available, else to `"cpu"`, never raising for that
edge case, since there is no way to be "wrong" about a device that does
not exist yet.

**Error types**: reused Forge's existing exception hierarchy rather than
adding a new one. A bad `model` type raises `TrainerError` (the existing
type for `Trainer`-shaped-API misuse, docstring extended in this
milestone -- Section 14); a bad batch element or empty iterable raises
`DataError` (the existing type for `Dataset`/`DataLoader`-shaped misuse).
No new `ForgeError` subclass was introduced.

## 13. Implementation

`forge/training/inference.py` (new, ~90 lines): `_resolve_device()`,
`_extract_input()`, and `predict()` itself -- eval-mode switch (with
`finally`-guaranteed restore), `no_grad()` scope, the single-Tensor/
iterable branch, and NumPy-level concatenation for the iterable case.

`forge/training/__init__.py`: exports `predict` alongside `Trainer`/
`Metric`/etc.

`forge/__init__.py`: re-exports `predict` at the top level (`forge.predict`,
matching `no_grad`/`save_model`/`load_model`'s existing top-level
convention), plus a docstring update naming it under `forge.training` and
in the top-level re-export list.

`forge/exceptions.py`: extended `TrainerError`'s and `DataError`'s
docstrings with one example each, naming `predict()`'s specific triggers --
no new exception type, no behavior change to either class.

Six examples retrofitted (`mnist`, `regression`, `resnet`, `segmentation`,
`autoencoder`, `waveform_classification`): each `train.py`'s
model-persistence round-trip block now calls `predict(model, query_x)`/
`predict(reloaded, query_x)` instead of the hand-rolled
`with no_grad(): ...` sequence; each file's now-unused `from forge import
no_grad` import was removed (`autoencoder/train.py` kept its `no_grad`
import -- it has a second, unrelated `no_grad` use for `model.encode()`
during latent-space visualization, untouched by this milestone).

`docs/architecture/training-engine.md`: new **Inference** section
(between **Checkpointing and resume** and **Known limitations**)
documenting the design in full, plus a package-layout/title update.

`README.md`: the "First model" quickstart snippet's own prediction step
now uses `forge.predict(model, ...)` instead of the same hand-rolled
pattern it previously demonstrated (verified to still run correctly --
Section 15); the `forge.training` capability bullet mentions `predict()`.

## 14. Files Changed

**Production** (`forge/`): `forge/training/inference.py` (new),
`forge/training/__init__.py`, `forge/__init__.py`, `forge/exceptions.py`.

**Examples**: `examples/mnist/train.py`, `examples/regression/train.py`,
`examples/resnet/train.py`, `examples/segmentation/train.py`,
`examples/autoencoder/train.py`, `examples/waveform_classification/
train.py`.

**Tests** (new): `tests/test_inference.py` (15 CPU tests),
`tests/test_inference_cuda.py` (3 CUDA-hardware-gated tests).

**Docs**: `docs/architecture/training-engine.md`, `README.md`,
`docs/development/m68-standalone-inference.md` (this file),
`docs/development/progress.md` (appended).

No `forge/tensor/`, `forge/autograd/`, `forge/nn/`, `forge/optim/`,
`forge/data/`, `forge/serialization/`, `forge/backend/`, or `forge/cli/`
file was touched -- confirmed by `git status`/`git diff --stat` before
writing this report (Section 16).

## 15. Architecture Impact

One new file in an existing subpackage (`forge/training/inference.py`),
one new top-level re-export (`forge.predict`), two exception docstrings
extended (no new exception type, no behavior change). No `Tensor`,
`Backend`, `Module`, `Loss`, `Optimizer`, `DataLoader`, or CUDA-kernel
surface changed. `Trainer` itself is completely unmodified -- `predict()`
composes `Module`/`no_grad`/`Device`, the same public primitives any
external caller already has, so it introduces no privileged internal
access `Trainer` didn't already have.

## 16. Tests

Baseline (immediately before this milestone's changes, re-run fresh, not
carried forward from M67's own report): **2,057 tests collected, 2,056
passed, 1 failed** (`test_dataloader_prefetch.py`'s pre-existing allocator
-measurement flake -- see `docs/development/m67-lstm-long-range-recall.md`
and this project's `forge-hardware-quirks`/`forge-allocator-deadlock-bug`
history).

New tests (18): `tests/test_inference.py` (15, CPU) -- re-export identity,
single-`Tensor` correctness against a manual reference, iterable/`DataLoader`
concatenation correctness (bare-Tensor and `(x, y)`-tuple batches),
training-mode restoration (both directions), eval-mode-Dropout determinism,
no-grad-graph/CPU-device output guarantees, all three validation-error
paths, device-default/override/`Parameter`-less-model behavior, and a
direct `Trainer`-vs-`predict()` forward-pass agreement check.
`tests/test_inference_cuda.py` (3, CUDA-hardware-gated) -- a CPU-resident
input automatically moved to a CUDA model with no explicit `.to("cuda")`
from the caller, `DataLoader`-driven concatenation on CUDA matching a
manual CUDA reference, and the "output always returns to CPU regardless of
the model's device" guarantee.

Final: **2,075 tests collected, 2,074 passed, 1 failed** (the same
pre-existing flake, re-run in isolation immediately after the full-suite
run and confirmed passing: `python -m pytest
tests/test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory
-q` -> `1 passed`). 2,057 + 18 = 2,075, exactly accounting for every new
test; zero regressions.

## 17. Verification

- `python -m pytest tests/test_inference.py -q` -> 15 passed.
- `python -m pytest tests/test_inference_cuda.py -q` -> 3 passed, on the
  reference 940MX (CUDA 12.6).
- `python -m pytest tests/test_mnist_example_integration.py
  tests/test_regression_example_integration.py
  tests/test_resnet_example_integration.py
  tests/test_segmentation_example_integration.py
  tests/test_autoencoder_example_integration.py
  tests/test_waveform_classification_example_integration.py -q` -> 78
  passed (every retrofitted example's own CPU integration suite,
  unmodified expectations, exercising the new `predict()` calls for real).
- `python -m pytest tests/test_mnist_example_cuda_integration.py
  tests/test_regression_example_cuda_integration.py
  tests/test_resnet_example_cuda_integration.py
  tests/test_segmentation_example_cuda_integration.py
  tests/test_autoencoder_example_cuda_integration.py
  tests/test_waveform_classification_example_cuda_integration.py -q` -> 30
  passed, on the reference 940MX -- every retrofitted example's CUDA path
  re-verified on real hardware, not simulated.
- Full suite: `python -m pytest tests/ -q` -> 2,074 passed, 1 pre-existing
  flake (reproduced passing in isolation, Section 16).
- README "First model" snippet extracted and run directly (Section 13)
  after the edit -- prints a real prediction, confirming the doc change is
  truthful and copy-paste-runnable, not illustrative pseudocode.
- `git status --short` reviewed before and after all changes: only the
  files listed in Section 14 appear; no generated artifact, checkpoint, or
  temporary probe file was left tracked or untracked.

## 18. Real Consumer

Six existing, already-tested, real-workload examples now call `predict()`
in place of hand-rolled inference code:

```python
# examples/mnist/train.py (classification, CNN)
pre_save_pred = predict(model, query_x).numpy()
reloaded = load_model(str(model_path), device=args.device)
post_load_pred = predict(reloaded, query_x).numpy()
assert np.allclose(pre_save_pred, post_load_pred, atol=1e-5)
```

identically shaped in `regression/train.py` (tabular MLP),
`resnet/train.py` (residual CNN), `segmentation/train.py` (dense
prediction, U-Net-style), `autoencoder/train.py` (reconstruction), and
`waveform_classification/train.py` (1D CNN) -- five materially different
task shapes proving reusability, not a single example-specific fit. Every
one of these scripts' own existing CPU and CUDA integration tests (108
total across the six) passed unmodified after the retrofit, meaning the
new code path was exercised against real trained models on real data,
saved, reloaded in a fresh `load_model()` call, and used for a real
prediction comparison -- not a toy unit test.

## 19. Results

- 18 new tests, all passing, 3 hardware-verified on the reference 940MX.
- 6 duplicated four-line inference blocks (Section 7) replaced by 2-line
  calls to a single, shared, documented function.
- 5 unused `from forge import no_grad` imports removed (one file kept it
  for an unrelated, still-live use).
- Zero regressions: 2,057 -> 2,075 tests, only the one pre-existing,
  independently-verified-unrelated flake present either before or after.
- README's own canonical "first model" snippet now demonstrates the
  correct, current inference idiom instead of the pattern this milestone
  found duplicated six times.

## 20. Limitations

- `predict()` only supports the `Trainer`-shaped "one forward call per
  step" model family (matching `Trainer.fit`/`evaluate`'s own existing
  scope). `char_rnn`/`word_rnn`/`long_range_recall`'s per-timestep
  recurrent generation is intentionally not covered -- no generalized
  "recurrent predict" abstraction was built without a second real
  consumer to validate its shape against (the same discipline M59's
  guardrails require).
- No CLI `forge predict` subcommand. The M68 brief explicitly lists "CLI
  tooling" and "inference APIs" together in its "do not build all of this
  simultaneously" list; this milestone built the inference API and
  deliberately deferred the CLI surface (Section 12/22) rather than
  combining both in one milestone.
- No batched multi-GPU or streaming-dataset inference path; `predict()`
  inherits the same single-GPU-index-0-only limitation every other CUDA
  path in Forge already has.
- Concatenation of batched results is a host-side (NumPy) operation, so
  very large `predict(model, loader)` calls materialize every batch's
  output in host memory at once before returning -- acceptable for every
  current Forge workload's data scale (consistent with the project's
  documented 8GB-RAM development-hardware constraint), not addressed
  further since no real consumer has hit this limit.

## 21. Rejected Alternatives

- **`Trainer.predict()` method.** Rejected (Section 12): would force every
  caller to construct a `Loss`/`Optimizer` they do not have and do not
  need, for a call that never uses either.
- **`Module.predict()` method.** Rejected (Section 12): blurs the
  "computation vs. orchestration" boundary `Trainer`'s own existence
  already establishes; the brief's own Candidate B explicitly warns
  against this shape without a demonstrated need.
- **A new `Tensor.cat`/`concatenate` primitive** to build the batched
  result. Rejected (Section 12): no existing Forge op exists without a
  proven differentiable-training consumer, and this concatenation is
  purely a non-differentiable, inference-time convenience -- the same
  precedent M49 already established for a different op.
- **`forge predict` CLI subcommand** (`load_model` + a `.npy` input +
  `predict()` + save/print the result). A genuinely good, small, real next
  step (Section 22) -- deferred, not because it lacks value, but because
  the brief explicitly separates "inference API" work from "CLI tooling"
  work in its do-not-combine list, and this milestone's evidence and
  budget were both spent on the API itself, proven against six real
  examples.
- **`ImageFolder`-style dataset loading** (Candidate A). Rejected for this
  milestone: no current example needs directory-based ingestion, and
  building it now would be exactly the "speculative abstraction" the
  brief's Phase 3 explicitly warns against building before its real
  consumer (the eventual cat/dog workflow) exists. Revisit once a real
  image-folder-shaped dataset is actually being built.
- **A general training-configuration/YAML system** (Candidate F). Not
  re-litigated this milestone -- M19/M52/M59 already rejected it
  repeatedly for the same reason (`requirements.md`'s own non-functional
  requirement against "hidden magic"), and nothing found here changes
  that calculus.

## 22. Future Dependency / Next Layer

`predict()` is now a stable foundation two concrete next layers could
build on, neither committed to as an M69 requirement:

1. **A `forge predict` CLI subcommand** -- `forge predict <model.forge>
   --input <input.npy> [--device] [--output <output.npy>]` -- would be a
   small, mechanical addition now that `load_model()` + `predict()`
   already compose cleanly into exactly that flow, and would be the
   purest possible demonstration of the vision's own "load it, give it
   new data, receive a prediction" sequence, fully decoupled from any
   training script.
2. **A directory-based dataset loader** (the eventual `image_folder/
   cats/dogs/` shape named in the M68 brief) now has a complete
   `train -> evaluate -> save -> load -> predict` pipeline to land into
   once it exists -- Section 8's non-gap finding stands until a real
   image-folder-shaped workload is actually proposed.

## 23. Practical Impact on Forge

A developer who trains a model with Forge today no longer needs to
remember `no_grad()`, explicit `.to(device)` calls on both input and
output, and `.numpy()` extraction, or reverse-engineer the correct
sequence from an example's own persistence-check code, to get a usable
prediction out of a loaded model. `forge.predict(model, x)` is now the
one, documented, tested way to do it -- discoverable from `forge.predict`
directly (matching `forge.save_model`/`forge.load_model`'s existing
top-level convention), and proven correct against six real, materially
different trained workloads rather than a synthetic demonstration.

## 24. Follow-Up Triggers

- If a real cat/dog- or image-folder-shaped dataset is actually proposed,
  revisit Candidate A (dataset ingestion) with `predict()`'s now-complete
  downstream pipeline as the target it should plug into.
- If a developer (or a future milestone) actually needs a `forge predict`
  CLI invocation (e.g. batch-scoring a directory of saved inputs without
  writing a Python script), build the CLI subcommand named in Section 22
  -- the API it would wrap already exists and is already tested.
- If a real multi-step/recurrent inference consumer emerges beyond
  `char_rnn`/`word_rnn`/`long_range_recall`'s existing hand-written
  generation loops, revisit whether a second, recurrence-shaped
  `predict`-like helper is justified -- not before, per M59's evidence
  -requirement guardrail.

## Suggested Commit Message

```
feat: add forge.predict() -- standalone post-training inference, closing
Forge's last undemonstrated vision-workflow step

Add a decoupled inference path (forge/training/inference.py) that does not
require constructing a Trainer, replacing an identical hand-rolled
no_grad()/device-transfer/numpy-extraction block duplicated across six
examples (mnist, regression, resnet, segmentation, autoencoder,
waveform_classification). 18 new tests (15 CPU, 3 CUDA, hardware-verified
on the reference 940MX); zero forge/tensor, forge/nn, forge/data, or
forge/serialization changes; zero regressions (2,057 -> 2,075 tests).
```
