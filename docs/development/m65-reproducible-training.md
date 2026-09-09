# M65 — Reproducible Training & Experiment Workflow

## 1. Objective

Turn one existing Forge example into a repeatable experiment workflow that
lets a real user reliably answer: *"I changed my model/training
configuration. Can I reproduce the run, compare the result, and resume
it?"* — without building a generic experiment-management platform.

## 2. Selected consumer

`examples/regression/` (Milestone 60's tabular-regression example), per the
brief's own preference for "the cleanest deterministic baseline." Its
dataset is synthetic, in-process, and fast (no download), and its model is
small enough that dozens of full training runs fit comfortably in this
milestone's investigation budget on the reference i5-7200U/940MX.

## 3. Baseline workflow (before this milestone)

Investigated by reading `examples/regression/train.py`, `forge/training/
trainer.py`, `forge/serialization/checkpoint.py`, `forge/data/dataloader.py`,
`forge/random.py`, and by **directly executing** the example and small probe
scripts (not just reading source) — see Section 4 for what execution found.

Already present and already working, confirmed by reading + running:

- A CLI (`argparse`): `--n-train/--n-val/--n-test`, `--device`, `--epochs`,
  `--batch-size`, `--lr`, `--seed`, `--output-dir`, `--resume`. This already
  satisfies the brief's Section 5 "configure without editing Python source"
  requirement — no new configuration mechanism was needed.
- Three separate, already-documented deterministic RNG streams: `forge.
  random.seed(args.seed)` (parameter init), `numpy.random.default_rng(args.
  seed)` inside `dataset.generate_raw()` (synthetic data), and a third
  explicit `numpy.random.Generator` passed to `DataLoader(..., generator=)`
  (shuffle order).
- `Trainer.fit()`/`.evaluate()` returning a structured `TrainingHistory` of
  `EpochResult`/`EvaluationResult` — both already plain `@dataclass(frozen=
  True)` records with only JSON-safe fields (`float`/`int`/`str`/`dict[str,
  float]`), confirmed by inspecting `forge/training/trainer.py`.
- `save_checkpoint()`/`load_checkpoint()` (Milestone 18) already save model
  state, full Adam per-parameter state, `epoch`/`global_step`, and `forge.
  random`'s default-generator state — restored automatically on load.
- `save_checkpoint(..., extra=...)` already accepts an arbitrary
  caller-defined JSON-safe dict, round-tripped through the archive
  unmodified — confirmed by reading `forge/serialization/checkpoint.py` and
  `forge/serialization/archive.py` (a ZIP of `metadata.json` + `.npy`
  arrays; `metadata` is written via plain `json.dumps`).
- `Trainer.resume()` already restores model/optimizer/epoch/global_step from
  a loaded `Checkpoint` and continues epoch numbering correctly (existing
  test: `test_checkpoint_save_and_resume_restores_state_and_continues_training`).
- An existing resume-equivalence test
  (`test_resume_equivalence_matches_continuous_training`) already proved `N`
  epochs → checkpoint → resume → `M` epochs matches continuous `N+M`-epoch
  training within `1e-5` — **but only for `shuffle=False`**, which the test's
  own docstring states explicitly ("matching within 1e-5]... Uses
  `shuffle=False` so both runs see an identical batch sequence with no
  caller-owned `DataLoader` generator to restore").

Genuinely absent (confirmed by grep/read, not assumed):

- No JSON/structured on-disk record of a run's per-epoch metrics — history
  was documented in `checkpoint.py`'s own module docstring as explicitly
  **not** part of the checkpoint contract ("`TrainingHistory`/`EpochResult`
  ... are not part of the checkpoint contract... A resumed `Trainer.fit()`
  call returns a fresh `TrainingHistory`").
- No persisted record of the CLI configuration used to produce a given
  checkpoint/model.
- No run-comparison tool of any kind.
- `train.py`'s actual default is `shuffle=True` (`DataLoader(train_ds, ...,
  shuffle=True, generator=data_rng)`) — the *documented* resume-equivalence
  guarantee (`shuffle=False`) does not cover `train.py`'s real, default
  behavior.

## 4. Real gap found by direct execution

Per the brief's requirement to prove gaps by running code, not just reading
it: a throwaway probe script (`baseline_probe.py`, not committed) reproduced
`train.py`'s exact `shuffle=True` default resume path — `N=2` epochs, save
checkpoint, then (mimicking `train.py`'s pre-M65 code exactly) construct a
**fresh** `data_rng = np.random.default_rng(args.seed)` for the "resumed"
process and continue `M=2` more epochs — and compared final parameters
against one continuous `N+M`-epoch run with the same seed.

**Result: `max abs param diff = 0.0252`, far outside any reasonable
tolerance.** This is a real, reproducible gap: resuming a `shuffle=True` run
did not continue the interrupted run's shuffle stream, it restarted a
fresh one from `--seed` — the resumed run saw a *different* sequence of
batches than the interrupted run's continuation would have, so training
diverged even with an otherwise-perfect model/optimizer/RNG restore.

A second probe (`fix_probe.py`) confirmed the fix: saving `data_rng.
bit_generator.state` (a plain JSON-safe dict, confirmed directly — round
-tripped through `json.dumps`/`json.loads` and reproduced identical future
draws) into `save_checkpoint(..., extra={"data_loader_rng_state": ...})`,
then restoring it into a fresh `numpy.random.Generator` before resuming,
closed the gap completely: **`max abs param diff = 0.0` exactly**, using
only `save_checkpoint`'s *already-existing* `extra` parameter — no `forge/`
production code was touched to achieve this.

This is the one genuine reproducibility gap this milestone found and fixed;
everything else the brief's Section 4 asks to investigate (Python/NumPy/
Forge RNG, model init, checkpoint/resume) was already correctly handled.

## 5. Implementation

**No changes to `forge/`.** Every piece was buildable from Forge's existing
public API:

### `examples/regression/train.py` (modified)
- `data_rng`'s state is saved into `save_checkpoint(..., extra={
  "data_loader_rng_state": data_rng.bit_generator.state})` on every save,
  and restored (`data_rng.bit_generator.state = saved_state`) immediately
  after `load_checkpoint()` on `--resume`, before `trainer.fit()` runs.
- Every run loads (if resuming) or creates a JSON run record, extends it
  with this invocation's history/final evaluation, and saves it next to the
  checkpoint as `regression_history.json`.

### `examples/regression/experiment.py` (new)
Small, JSON-only bookkeeping:
- `history_to_dicts()`/`evaluation_to_dict()` — `dataclasses.asdict()` on
  `TrainingHistory`'s/`EvaluationResult`'s existing dataclasses; needed no
  new method on either, since both were already JSON-safe.
- `new_run_record()`/`load_run_record()`/`save_run_record()` — a small
  versioned (`forge_run_record_format_version`) JSON file: `config` (fixed
  at the *first* invocation, never overwritten by a later `--resume`
  invocation's own possibly-different `--epochs`), `history` (cumulative
  across resumes), `final_eval`, `total_duration_seconds`.
- `extend_run_record()` — appends one `fit()` call's epochs onto an existing
  record in place (used by both a from-scratch run and a resumed one).
- `compare_runs()` — a plain-dict diff of two run records: differing config
  keys, epoch count, final train/eval loss and metrics, total runtime.

### `examples/regression/compare.py` (new)
`python -m examples.regression.compare <run_a> <run_b>` — loads two run
records and prints `compare_runs()`'s diff as short text. No UI, no chart.

## 6. Architecture / API impact

None. `forge/training/`, `forge/serialization/`, `forge/data/`, `forge/
random.py` are byte-for-byte unchanged. `docs/architecture/persistence.md`
received one clarifying paragraph (not a policy change) pointing at this
milestone's worked example of fulfilling the already-documented "caller's
own responsibility" for a caller-owned generator.

## 7. Configuration design

CLI-only (`argparse`), reusing `examples/regression/train.py`'s existing
flags unchanged — no YAML/JSON config-file loader was added, since the
brief explicitly discourages "a heavyweight configuration framework" and
the existing CLI already satisfies "important training parameters can be
changed without editing Python source." The *new* piece is only that the
resolved `vars(args)` dict is now persisted (in the run record's `config`
field) rather than living only in that one process's memory/stdout.

## 8. Reproducibility design

Three already-existing, already-independent seeded streams (`forge.random`,
the dataset generator, the `DataLoader` shuffle generator) plus one new
piece: the `DataLoader` generator's state is now captured/restored across a
checkpoint save/resume via the pre-existing `extra` mechanism (Section 4).
No new RNG abstraction, no change to `forge/random.py`.

## 9. Checkpoint/resume design

Unchanged checkpoint *format* — `CHECKPOINT_FORMAT_VERSION` is still `2`,
since `extra` was already an open, caller-defined dict; this milestone only
adds one new *key* (`data_loader_rng_state`) that `examples/regression/
train.py` itself writes and reads, not a `forge/serialization` schema
change. Existing checkpoints without that key still load fine (`checkpoint.
extra.get("data_loader_rng_state")` returns `None`, and `train.py` then
falls back to the pre-M65 behavior of a freshly seeded generator — a graceful
degradation, not an error).

## 10. Metrics / history

`regression_history.json`: versioned, indented, sort-keyed JSON, one file
per output directory, continuous across resumes. Not a database, not a
time-series store — a single small file, matching the brief's Section 6/14
guardrails.

## 11. Run comparison

`compare.py` — plain text, config-diff plus final-loss/metric/epoch/runtime
deltas. No visualization.

## 12. Test coverage

**Before this milestone: 1,961 passed** (per M64's own report).

**New tests (20 total):**
- `tests/test_regression_experiment.py` (14, CPU, no training — pure
  bookkeeping logic): `history_to_dicts`/`evaluation_to_dict` JSON-safety,
  run-record create/save/load round trip, missing-file handling, unsupported
  -format-version rejection (invalid configuration handling), cumulative
  history across multiple `extend_run_record()` calls with `config`
  preserved, `compare_runs()` correctness (identical configs, differing
  config keys, differing final metrics/epoch counts/runtime, a
  not-yet-trained record), and `compare.py`'s CLI (report formatting,
  missing-run-record error path).
- `tests/test_regression_reproducible_training.py` (5, CPU, actually
  invokes `examples.regression.train.main(argv)` — the real CLI entry
  point, not a hand-reconstructed pipeline, since the CLI wiring itself is
  what Milestone 65 changed): same-config-twice bitwise reproducibility,
  **the flagship regression test for the `shuffle=True` resume-gap fix**
  (full training vs. partial-then-resume, bit-exact match, run record stays
  continuous across the resume), checkpoint `extra` carries a JSON-safe
  `data_loader_rng_state`, two `--lr` configs produce a run-record `compare_
  runs()` diff that actually reflects a real behavioral difference, and
  resuming after deleting the run-record sidecar starts a fresh one rather
  than raising (a documented limitation, Section 17).
- `tests/test_regression_example_cuda_integration.py` (+1, CUDA hardware):
  the CUDA counterpart of the flagship shuffle-resume test — hardware
  -verified on the reference 940MX.

**After this milestone: 1,981 passed** (1,961 + 20), one pre-existing,
unrelated flaky test (`test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, the same CUDA
allocator-measurement flake M63/M64's own reports footnoted) failed in the
full-suite run and passed cleanly in isolation immediately afterward — zero
regressions attributable to this milestone.

## 13. CPU validation

All new tests above ran and passed on the reference i5-7200U. Directly
executed (not just tested): from-scratch runs, `--resume` runs, `compare.py`
against two real run records, and the pre-fix/post-fix probe scripts from
Section 4.

## 14. CUDA validation

Hardware-verified on the reference GeForce 940MX (CC 5.0, driver 582.53,
CUDA Toolkit 12.6): `python -m examples.regression.train --device cuda`
(from-scratch, resume, and the full/partial+resume comparison) all produced
correct results; the full/partial+resume comparison matched to `atol=1e-5`
(the same tolerance the pre-existing CUDA tests in this file already use)
with `shuffle=True`. `tests/test_regression_example_cuda_integration.py`'s
new test makes this permanent.

## 15. Reproducibility results

**(A) Same config, twice, from scratch (CPU, `--seed 123`, 4 epochs, 200
train samples):** every per-epoch `train_loss`/`train_metrics`/`val_loss`/
`val_metrics` matched exactly; final model parameters matched **exactly**
(`atol=0.0`, i.e. bit-identical `float32` arrays) — the only differing field
across the two runs' JSON records was wall-clock `duration`, as expected.

**(B) Full `N+M`-epoch training vs. `N` epochs → checkpoint → resume → `M`
epochs, with `shuffle=True` (CPU, `--seed 55`, `N=3, M=2`):** final model
parameters matched within `1e-6` (effectively exact — small residual is
`float32` accumulation order, not a real divergence); the run record's
epoch sequence was continuous (`[1, 2, 3, 4, 5]`), not restarted at 1 for
the resumed invocation. Repeated on CUDA: parameters matched within `1e-5`.

**(C) Two configurations, `--lr 1e-4` vs. `--lr 1e-1`:** `compare_runs()`
correctly reported `config_diff = {"lr": (0.0001, 0.1)}`, and the two runs'
final training losses were measurably different (confirming the recorded
metadata tracks a real behavioral difference, not just inert bookkeeping).

## 16. Limitations

- The `DataLoader` generator fix is implemented **in the example**
  (`train.py` reads/writes `checkpoint.extra`), not as a reusable `forge/`
  API — a different example wanting the same guarantee must copy the same
  four-line pattern. This was a deliberate choice (Section 17): the pattern
  is trivial and example-specific enough (which generator, what key name)
  that promoting it to `forge/` now would be speculative generalization
  from a single consumer, matching this codebase's own established
  "dedicated per-consumer addition, not a general primitive" precedent
  (e.g. M48's dInput kernel, M50's `tanh`).
- If a run's `regression_history.json` sidecar is deleted (or was never
  written by an older pre-M65 checkpoint) but the checkpoint itself is
  intact, `--resume` still works correctly (model/optimizer/RNG are
  unaffected — none of that state lives in the sidecar), but the run record
  starts a new file rather than recovering the original run's earlier
  history/config. Covered by
  `test_resuming_without_an_existing_run_record_starts_a_fresh_one`.
- `compare.py` only compares two runs of *this one example* (it assumes the
  `examples/regression`-specific run-record shape) — not a cross-example or
  cross-framework comparison tool.
- Bitwise CPU reproducibility is not promised for every possible Forge
  workload — only demonstrated for this example's actual operations
  (`Linear`/`ReLU`/`MSELoss`/`Adam`, no `Dropout`, no CUDA-specific
  nondeterministic reduction order in this small architecture). CUDA
  matched within `1e-5`/`1e-6`, not asserted bit-exact, per this milestone's
  own instruction not to over-promise CUDA determinism.

## 17. Rejected approaches

- **A YAML/JSON config-file loader.** Rejected — the existing CLI already
  satisfies the brief's configuration requirement, and Forge's own prior
  milestones (M49/M52) already explicitly ruled out a config/YAML system as
  a non-goal for this codebase's current maturity. Adding one now would
  have been scope creep with no demonstrated need.
- **Promoting the DataLoader-RNG-state pattern into `forge/data/dataloader.
  py` or `forge/training/trainer.py`** (e.g. a `DataLoader.get_state()`/
  `Trainer` auto-capturing every loader it touches). Rejected: exactly one
  consumer exists today (`examples/regression/train.py`); the existing
  `extra` mechanism already fully solves it with zero framework risk.
  Generalizing now would guess at a shape (which loaders? keyed how?
  restored automatically or explicitly?) with no second real consumer to
  validate the design against — the same discipline this codebase's
  `docs/development/m49-capability-assessment.md` already established for
  its rejected generic-elementwise-divide primitive.
- **A database or time-series store for metrics** (section 6/14's explicit
  guardrail). Rejected outright — a single small JSON file per run
  fully satisfies "compare two runs without parsing terminal output."
  A visualization/plotting layer was similarly not built (section 8/14).
- **Streaming the run record to disk once per epoch** (crash resilience
  mid-epoch). Rejected — `Trainer.fit()` itself has no per-epoch callback
  hook to attach to without changing `forge/training/trainer.py`, and the
  brief's Section 10 requires proving existing APIs cannot support a
  requirement before adding one; the checkpoint itself (saved once per
  `train.py` invocation, not per epoch) is the actual resume/crash-recovery
  boundary this workflow offers, matching every other Forge example's
  existing checkpoint granularity.

## 18. Practical impact on Forge

A developer using `examples/regression/train.py` can now reliably: change a
hyperparameter via a flag, get a fixed `--seed` to reproduce initialization
and training bit-for-bit, see per-epoch metrics recorded to a file rather
than only scrolled past in a terminal, interrupt and resume training
(including its shuffle order, not just its parameters) without silently
diverging from what continuous training would have produced, and diff two
completed runs' configuration and results with one command. This closes the
actual gap between "Forge can train a model" and "Forge supports iterating
on a model" for this workload, using only existing framework capability plus
~250 lines of small, example-local, non-speculative bookkeeping code.

## 19. Outcome

**Outcome A — Workflow completed mostly with existing framework.** The one
genuine gap found (`DataLoader` shuffle-generator resume) was closed using
`save_checkpoint`'s pre-existing `extra` parameter, not a new `forge/` API.
Every other required capability (CLI configuration, three-stream
determinism, checkpoint/resume, structured/JSON-safe history, model
persistence) was already present and already correct — this milestone's
main contribution is example-level wiring plus the one probe-verified fix
and its permanent regression test.

## 20. Recommendation for M66

No further Conv2d/CUDA optimization is implied by this milestone (M65 made
no performance change and found no regression). Two reasonable directions,
neither forced:

1. Apply the same `data_loader_rng_state`-in-`extra` pattern (now proven,
   tested, documented) to a second example with meaningful shuffle-order
   sensitivity — e.g. `examples/segmentation/` — purely as a small,
   low-risk repeat of an already-validated pattern, not a new capability.
2. Revisit `docs/development/m52-product-direction.md`'s deferred
   `nn.BatchNorm2d`/`Module` buffer direction, or survey for a new product
   decision the way M49/M52 did, if no further reproducibility-workflow
   value remains once (1) above is done. Do not re-run this same 5-area
   survey without a new product decision driving it, per M49's own
   established guidance.
