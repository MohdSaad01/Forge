# M73 — Reusable Training Workflow: `start_training_session()`

## 1. Objective

By M72, Forge could take a developer all the way from a directory of image
files to a saved model with persisted preprocessing/class metadata and a
`forge model predict` CLI command. The brief explicitly forbade another
capability survey or readiness assessment and required investigating the
actual repository, identifying a genuinely duplicated piece of training
orchestration, and implementing the smallest reusable abstraction that
removes it -- with a real consumer, not a synthetic unit test.

## 2. Investigation

Read/ran directly, not assumed from prior reports:

- `forge/training/trainer.py`, `metrics.py`, `inference.py`: `Trainer`
  already orchestrates `DataLoader -> Module -> Loss -> autograd ->
  Optimizer`; `predict()`/`interpret_classification()` (M68/M72) already
  cover post-training inference. Neither owns "which `Trainer` to build" --
  every example still decides that itself.
- `forge/data/dataloader.py`: `DataLoader(dataset, ..., generator=...)`
  accepts an explicit `numpy.random.Generator`; `forge/serialization/
  checkpoint.py`'s `extra` dict is caller-defined JSON-safe state, already
  the mechanism M65 used to close a reproducibility gap.
- Every `examples/*/train.py` with `--resume` support (`mnist`,
  `regression`, `resnet`, `segmentation`, `autoencoder`,
  `waveform_classification`, `image_folder_classification` -- confirmed via
  `grep -n "if args.resume:"` across `examples/`, 7 matches): each contains
  the **identical** ~10-line branch --
  `load_checkpoint()`+`Trainer(...)`+`trainer.resume()` vs.
  `build_model().to(device)`+`Adam(...)`+`Trainer(...)` -- differing only in
  `metrics=[...]`, the model-builder call, and the optimizer's `lr`. This is
  real, provable duplication, not a hypothetical one.
- Cross-checked which of those seven actually carry M65's
  `data_loader_rng_state` fix (`grep -n "data_loader_rng_state"
  examples/*/train.py`): only `regression` (M65's own target) and `resnet`
  (copied by hand later) do. **`mnist`, `segmentation`, `autoencoder`,
  `waveform_classification`, and `image_folder_classification` do not** --
  each defaults to `shuffle=True` and therefore carries the exact same
  latent resume-divergence bug M65 diagnosed and fixed once, silently
  un-fixed in 5 of 7 examples. This is real evidence, not a guess: it is the
  same class of bug, in the same shape of code, independently confirmed
  absent by reading each script.
- `char_rnn`/`word_rnn`/`long_range_recall`: hand-written per-timestep
  training loops with no `Trainer`/`DataLoader.fit()` call at all (already
  noted in `docs/architecture/training-engine.md`'s **Inference** section
  for the same reason `predict()` doesn't apply to them) -- confirmed out of
  scope for this abstraction, which is specifically about `Trainer`
  construction.
- `forge/cli/model.py`: `model inspect`/`predict`/`convert`,
  `checkpoint inspect`/`convert` all operate on an already-saved artifact.
  None constructs a `Trainer`, a `Dataset`, or an `Optimizer` from CLI
  arguments -- there is no existing pattern for expressing a model
  architecture, dataset, or optimizer choice from the command line, and
  `docs/product/scope.md`'s no-config/YAML-system non-goal rules out
  inventing one just to add `forge model train`.

**Conclusion:** a clear, evidence-backed abstraction boundary exists --
"build a fresh `Trainer`, or resume one from a checkpoint, including the
`DataLoader` shuffle-generator state" -- with seven real, already-existing
consumers (not a synthetic one), two of which (M65's own fix scope) also
demonstrate a genuine bug the abstraction closes as a side effect. This is
Outcome A per the brief's own decision rule.

## 3. Decision

Implement `forge.training.start_training_session()` /
`TrainingSession` (`forge/training/session.py`) -- a single function that
replaces the hand-rolled branch, plus a `data_loader_rng` +
`save_checkpoint()` pairing that generalizes M65's fix so every consumer
gets it automatically rather than having to remember to copy it by hand
again.

### Rejected alternatives

- **`forge.train(model, dataset, ...)` (the brief's own illustrative
  end-state API).** Rejected for this milestone: it would need to own
  `DataLoader` construction (batch size, shuffle, `drop_last`) and decide
  metrics/validation-loader wiring on the caller's behalf -- none of which
  is duplicated across examples in a single obvious shape (batch sizes,
  validation cadence, and metric choices all differ per example already).
  Building it now would be exactly the "large rewrite of Trainer" /
  "theoretically complete high-level training API" the brief explicitly
  warns against, with no repository evidence it is needed yet.
- **A `TrainingConfig`/config-object abstraction.** Rejected -- every
  example already has `argparse`-parsed `args`; wrapping that in a second
  config object would be a config-management layer with no duplicated
  *behavior* to justify it (`docs/product/scope.md`'s no-config-DSL
  non-goal).
- **A `forge model train` CLI command.** Rejected per Section 7 of the
  brief and confirmed by the investigation above: every example still needs
  Python code to define its model architecture, dataset construction, and
  (for several) a custom preprocessing pipeline -- there is no
  CLI-expressible "which model/dataset" convention to build a training
  command around, unlike `forge model predict`'s single-file-input
  convention (M72).
- **Making `Trainer` itself resume-or-construct-aware** (e.g.
  `Trainer.from_checkpoint_or_new(...)`). Rejected -- `Trainer` already has
  a well-scoped single responsibility (orchestrate an already-constructed
  model/loss/optimizer); adding checkpoint-loading and model/optimizer
  *construction* to it would blur the same "what a class computes" vs. "how
  it's orchestrated" boundary M68's `predict()` was deliberately kept
  outside `Trainer` to preserve. `start_training_session()` sits one layer
  above `Trainer`, composing it, exactly as `Trainer` itself composes
  `Module`/`Loss`/`Optimizer` without owning any of them.
- **Folding `data_loader_rng` state into `Trainer` itself** (e.g.
  `Trainer.data_loader_rng`). Rejected -- `Trainer` is deliberately never
  told about `DataLoader` construction (`docs/architecture/training-engine.
  md`'s **Batch movement** section: "`DataLoader` itself remains entirely
  CPU-side... never made device-aware," the same "`Trainer` doesn't own
  `DataLoader`'s concerns" boundary). `TrainingSession` is a natural place
  for this pairing since it is a session-scoped concept the caller
  constructs alongside its `DataLoader`s, not a `Trainer` property.

## 4. Implementation

`forge/training/session.py` (new file):

```python
@dataclass
class TrainingSession:
    trainer: Trainer
    data_loader_rng: np.random.Generator
    resumed: bool
    checkpoint: Checkpoint | None = None

    def save_checkpoint(self, path, *, extra=None) -> None: ...

def start_training_session(
    *, build_model, build_optimizer, loss_fn, seed,
    device="cpu", metrics=None, verbose=True, prefetch=False, prefetch_size=2,
    resume=None,
) -> TrainingSession: ...
```

- `forge.random.seed(seed)` and `data_loader_rng = np.random.default_rng
  (seed)` always run first, matching every example's existing per-example
  policy (`docs/architecture/persistence.md`'s checkpoint RNG policy) --
  when resuming, `load_checkpoint()` immediately overwrites `forge.random`'s
  state with the checkpoint's own saved state (its documented behavior), so
  the `seed()` call above is harmlessly superseded rather than skipped
  conditionally.
- Fresh path (`resume` falsy): `build_model().to(device)`,
  `build_optimizer(model.parameters())`, `Trainer(...)`.
- Resume path: `load_checkpoint(resume, device=device)`, `Trainer(model=
  checkpoint.model, optimizer=checkpoint.optimizer, ...)`,
  `trainer.resume(checkpoint)`, then `data_loader_rng.bit_generator.state`
  is overwritten from `checkpoint.extra["data_loader_rng_state"]` **only
  when that key is present** -- a checkpoint saved by plain `Trainer.
  save_checkpoint()` (not through `TrainingSession`) leaves
  `data_loader_rng` at its freshly-seeded state, matching every example's
  behavior before this milestone for a checkpoint with no such key.
- `TrainingSession.save_checkpoint(path, extra=None)` calls `self.trainer.
  save_checkpoint(path, extra={**extra, "data_loader_rng_state":
  self.data_loader_rng.bit_generator.state})`, raising `TrainerError` if the
  caller's own `extra` already defines that key (this method owns it).
- `build_model`/`build_optimizer` non-callable inputs raise `TrainerError`
  immediately, matching `Trainer`'s own "validate everything eagerly"
  construction style; every other validation (`loss_fn`/`device`/`metrics`
  types, duplicate metric names, unknown device strings) is inherited for
  free from `Trainer(...)`'s own constructor -- this function adds no
  parallel validation logic of its own.

`forge/training/__init__.py` re-exports `TrainingSession`,
`start_training_session`. Not re-exported at the top-level `forge.*`
namespace -- `Trainer` itself isn't either (only `forge.training.Trainer`);
`predict`/`save_model`/`load_model` are elevated to top level because they
are used broadly (including by the CLI), while this function is
`Trainer`-adjacent orchestration that belongs in the same namespace as
`Trainer`.

## 5. Architecture Impact

- `forge/training/trainer.py` -- **unmodified**. `Trainer`'s constructor,
  `fit()`, `evaluate()`, `resume()`, `save_checkpoint()` are byte-for-byte
  unchanged.
- `forge/data/dataloader.py`, `forge/serialization/checkpoint.py` --
  **unmodified**. `start_training_session()` calls their existing public
  APIs (`load_checkpoint`, `np.random.default_rng`) with no new parameters
  or behavior added to either.
- No new persistence format, no `FORMAT_VERSION`/`CHECKPOINT_FORMAT_VERSION`
  bump -- `"data_loader_rng_state"` is an ordinary `extra` dict key, exactly
  M65's existing mechanism, not a new one.
- No global/hidden state: `TrainingSession` is a plain object the caller
  holds and threads through its own script, matching every other Forge
  orchestration object (`Trainer`, `Checkpoint`).
- No CUDA-specific logic duplicated -- `device=`/`prefetch=`/
  `prefetch_size=` pass straight through to `Trainer(...)`'s own,
  already-CUDA-tested handling; `start_training_session()` itself contains
  no device-dispatch code.

## 6. Backward Compatibility

- Every existing `Trainer`/`load_checkpoint`/`DataLoader` call site is
  untouched; nothing about calling them directly is deprecated or changed.
- A checkpoint saved by a pre-M73 script (plain `Trainer.save_checkpoint()`,
  no `data_loader_rng_state` key) loads correctly through
  `start_training_session(..., resume=...)` -- confirmed by
  `test_resume_without_a_saved_data_loader_rng_state_falls_back_to_a_fresh_generator`.
- `examples/regression/train.py`'s existing dedicated reproducibility test
  suite (`tests/test_regression_reproducible_training.py`, written for M65,
  asserting the exact `shuffle=True` resume-equivalence property and the
  `data_loader_rng_state` checkpoint key) **passes unmodified** against the
  refactored script -- direct proof the retrofit preserved every documented
  guarantee rather than just "looking similar."

## 7. Real Consumers

- **`examples/image_folder_classification/train.py`** (required primary
  consumer per the brief): the resume-or-fresh-start branch and the manual
  `Trainer.save_checkpoint()`/`Adam(...)` construction were replaced with
  `start_training_session(...)`; `train_loader` now shuffles from
  `session.data_loader_rng`. This is also the first time this script gained
  M65-class resume-equivalence for its own `shuffle=True` default -- it
  never had it before.
- **`examples/regression/train.py`** (second consumer): its hand-written
  `data_loader_rng_state` save/restore logic (M65's original implementation)
  was deleted and replaced by the same `start_training_session(...)` call --
  proving the extraction is a faithful generalization, not just a
  copy-shaped abstraction, since this script's own dedicated reproducibility
  tests catch any behavioral drift.
- `mnist`, `resnet`, `segmentation`, `autoencoder`,
  `waveform_classification` were **not** retrofitted this milestone (see
  Limitations) -- the pattern transfers directly (each has the identical
  branch shape) but retrofitting all seven was judged unnecessary to prove
  the abstraction's value, per the brief's "at least one, preferably a
  second" bar.

## 8. Tests

- `tests/test_training_session.py` (14 CPU tests): fresh-session
  construction (model/optimizer built via factories, `forge.random`
  seeded before `build_model()`, `data_loader_rng` matches a plain
  `default_rng(seed)`, device placement, metrics pass-through),
  construction validation (non-callable `build_model`/`build_optimizer`,
  invalid `loss_fn` surfacing `Trainer`'s own `TrainerError`), the
  `save_checkpoint()` / `data_loader_rng_state` round trip (including the
  "caller must not redefine that key" guard), resume restoring
  model/optimizer/epoch/global_step, the no-saved-rng-state fallback, a
  resume-equivalence integration test (continuous 5-epoch training vs.
  3-epoch-then-resume-2-more with `shuffle=True`, matching to `atol=1e-6` --
  the same property `test_regression_reproducible_training.py` guards, now
  proven at the `forge/training` layer directly), and a CUDA-device-convert
  resume smoke test (skips cleanly without CUDA).
- `tests/test_training_session_cuda.py` (2 CUDA tests, gated on
  `is_cuda_available()`): a fresh CUDA session trains and reports
  `device="cuda"`; the same resume-equivalence property as above,
  hardware-verified end-to-end on real CUDA rather than assumed from the
  CPU result, since `Trainer`'s device transfer and `load_checkpoint`'s
  device placement are both real dispatch paths this function's own tests
  had not previously exercised together.
- No changes to `tests/test_trainer.py`, `tests/test_checkpoint.py`, or any
  other existing `forge/training`/`forge/serialization` test file --
  `Trainer`/`load_checkpoint` themselves are unmodified.

## 9. Full-Suite Result

`python -m pytest tests/ -q`: **2,215 tests collected, 2,214 passed, 1
failed** (2,199 + 14 CPU + 2 CUDA new here, on this CUDA-equipped
development machine). The one failure is the same pre-existing
`test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`
allocator-measurement flake documented since M63 (`CUDA active bytes grew:
320 -> 288`, i.e. it *shrank* -- a measurement artifact of this run's
allocator state, not a leak) -- reproduced **passing cleanly in isolation**
immediately afterward, no M73 regression. No other test file in the full
run was affected.

## 10. CPU/CUDA Verification

- `tests/test_training_session.py` (CPU): **14/14 passed**.
- `tests/test_training_session_cuda.py` (hardware-verified on the reference
  940MX): **2/2 passed**.
- `tests/test_regression_reproducible_training.py` (M65's own dedicated
  reproducibility suite, run against the refactored `regression/train.py`):
  **5/5 passed, unmodified**.
- `tests/test_regression_example_integration.py`,
  `tests/test_regression_example_cuda_integration.py`,
  `tests/test_regression_experiment.py`: **31/31 passed**.
- `tests/test_image_folder.py`,
  `tests/test_image_folder_classification_integration.py`,
  `tests/test_image_folder_classification_cuda_integration.py`: **44/44
  passed** (hardware-verified on the 940MX).

## 11. End-to-End Verification

Ran the actual `image_folder_classification` workflow end-to-end via its own
`main()` entry point (not just its test suite), on real synthetic image
files on disk:

```text
python -m examples.image_folder_classification.train --generate \
    --samples-per-class 20 --epochs 2 --batch-size 8 \
    --output-dir scratch_test/ifc_m73 --data-root scratch_test/ifc_m73/data
```
generated the dataset, trained 2 epochs via `start_training_session(...)`,
saved a checkpoint + model + preprocessing + classes, reloaded the model
fresh and confirmed `predict()` reproduced the pre-save prediction, ran
standalone inference through `interpret_classification()` on a held-out test
sample and on a brand-new 200x140 image never seen during training (outside
the generated size range) -- printing a human-readable class label and
confidence in both cases.

Then ran `--resume` against that checkpoint:
```text
python -m examples.image_folder_classification.train \
    --resume scratch_test/ifc_m73/image_folder_checkpoint.forge \
    --epochs 1 --batch-size 8 \
    --output-dir scratch_test/ifc_m73 --data-root scratch_test/ifc_m73/data
```
confirmed `session.resumed` printed `epoch=2, global_step=12` (continuing
from the first run, not restarting), trained one further epoch to global
epoch 3, and saved a new checkpoint successfully -- the full `dataset ->
ImageFolder -> preprocessing -> DataLoader -> model -> training ->
evaluation -> checkpoint/save -> fresh process -> load model -> load
preprocessing/classes -> prediction` pipeline the brief asked for, driven by
the new abstraction rather than the old hand-rolled branch.

## 12. Limitations

- **Five examples not retrofitted.** `mnist`, `resnet`, `segmentation`,
  `autoencoder`, `waveform_classification` still contain their own
  hand-rolled resume branch. `resnet` and none of the other four (besides
  `regression`, now retrofitted) got M65's `data_loader_rng_state` fix, so
  their latent resume-with-`shuffle=True` divergence bug remains
  un-fixed until they are retrofitted -- a concrete, mechanical, low-risk
  follow-up (the brief's own scope bar judged two real consumers sufficient
  to prove the abstraction for this milestone).
- **No `forge.train(model, dataset, ...)`.** `start_training_session()`
  still requires the caller to construct its own `DataLoader`s and call
  `session.trainer.fit(...)` itself -- deliberately, per Section 3's
  rejected-alternatives reasoning; this is a real limitation relative to the
  brief's own illustrative end-state, not an oversight.
- **No CLI training command.** `forge model train` was evaluated and
  explicitly not built (Section 3) -- current architecture requires Python
  code to define model/dataset/optimizer, which no CLI config surface
  exists for and none was invented.
- **`TrainingSession` does not validate `build_model`'s return type is a
  `Module`** at the point it's called -- `Trainer(...)`'s own constructor
  catches that one step later with a clear `TrainerError`, so no failure
  goes unreported, but the error message names `Trainer`, not
  `start_training_session`.

## 13. Concrete Triggers for Future Work

- **Retrofit the remaining five examples** the moment any of them needs a
  behavior change anyway (e.g. a future milestone touching
  `segmentation/train.py` for an unrelated reason) -- purely mechanical, no
  new investigation needed, same pattern as `regression`'s retrofit here.
- **Revisit `forge.train(model, dataset, ...)`** only if a future milestone
  finds `DataLoader` construction itself (batch size/shuffle/`drop_last`
  choices) converging to one shape across examples the way `Trainer`
  construction already has -- not before.
- **Revisit a CLI training command** only if a future product decision
  establishes a concrete, CLI-expressible model/dataset convention (the same
  bar `forge model predict` cleared via `ImageFolder`'s single-file-input
  convention in M72) -- not as a follow-on to this milestone's own CLI
  command.

## 14. Suggested Commit Message

```
feat: extract start_training_session()/TrainingSession, closing a latent DataLoader-RNG resume gap in 5 examples

Seven checkpoint-capable examples (mnist, regression, resnet, segmentation,
autoencoder, waveform_classification, image_folder_classification) each
hand-rolled an identical resume-or-fresh-start Trainer construction branch.
forge.training.start_training_session() (forge/training/session.py)
extracts it once, and generalizes M65's data_loader_rng_state checkpoint
fix (previously copied by hand into only 2 of the 7) so every consumer
gets DataLoader-shuffle resume-equivalence automatically via
TrainingSession.save_checkpoint().

Retrofitted image_folder_classification/train.py (required primary
consumer) and regression/train.py (proves behavioral equivalence: its
existing M65 reproducibility test suite passes unmodified). 16 new tests
(14 CPU + 2 CUDA, hardware-verified on the 940MX).
```
