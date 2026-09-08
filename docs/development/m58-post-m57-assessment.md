# M58 — Fresh post-M57 assessment: no production change warranted (assessment only)

## 1. Objective

Per the milestone brief: determine whether any concrete piece of work would
materially advance Forge from its current state, without defaulting to
another speculative feature, capability survey, or optimization pass. Decide
between IMPLEMENT, INVESTIGATE, or NO PRODUCTION CHANGE using direct
evidence gathered this milestone, not carried forward from M49/M52/M56/M57's
own reports.

## 2. Current repository state (verified directly, not assumed)

- Full test suite, executed fresh on this machine with the real CUDA backend
  live (940MX, driver present, `is_cuda_available() == True`):
  **1,771 passed, 0 failed, 0 skipped** — identical to M56's and M57's own
  counts, confirming zero drift since M55's last production change.
- `forge/nn/__init__.py` exports unchanged since M57's own check: `Module,
  Parameter, Linear, ReLU, Tanh, Conv2d, MaxPool2d, Sequential, Flatten,
  Dropout, RNNCell, BatchNorm2d, Embedding, Loss, MSELoss,
  CrossEntropyLoss`.
- `forge/cli/` still exposes only `model`/`checkpoint`/`benchmark` — no
  `train`/`evaluate`/`predict`, unchanged since M19, matching
  `requirements.md`'s "Planned commands" note. M52 already evaluated and
  rejected building these (conflicts with Forge's own "no config/YAML
  system" non-goal, no driving workload); nothing new contradicts that.
- No `TODO`/`FIXME`/`XXX` markers anywhere in `forge/` (grepped directly).
- Examples (`examples/mnist`, `examples/char_rnn`, `examples/word_rnn`, plus
  three small demo scripts) are unchanged in architecture since M55/M57.
  `examples/word_rnn/` now has its `.gitignore` entry and `README.md`
  (M57's fixes — still uncommitted from a prior session, verified present
  in the working tree and left untouched here).
- `git status` at the start of this milestone showed M57's own uncommitted
  work (`.gitignore`, `docs/development/progress.md`, two assessment
  reports, `examples/word_rnn/README.md`) still sitting on top of the M55
  commit — this milestone's own diff is additive on top of that, per this
  project's "Claude Code must not commit" rule (`CLAUDE.md`).

## 3. Real consumer / problem identified

No new external consumer or use case has appeared since M57 — same five
workload families (MNIST CNN, char-RNN, word-RNN, plus three demo scripts),
same `docs/product/use-cases.md` UC1-UC7, all already demonstrated. This
milestone's contribution is verifying that claim by direct execution rather
than by re-reading it (Section 4).

## 4. Candidates considered and evidence gathered

### (1) A third model/workload family
Re-examined the same candidate set M52/M54/M56/M57 already evaluated
(autoencoder, binary classification, stacked RNN, seq2seq,
attention/transformer). No new evidence changes any of their conclusions:
the first four compose entirely from existing, already-proven primitives
(no framework pressure); attention/transformer is still simultaneously
blocked on `softmax`, `transpose`/`permute`, and batched 3D matmul with no
single workload driving any one of them. **Reject, unchanged.**

### (2) A framework capability gap
No third-workload candidate above survived far enough to expose a new
primitive gap. **Nothing new.**

### (3) Developer ergonomics / usability
- Top-level `README.md` is thin (no quickstart, no feature list) but
  accurate and not misleading, and this is a single-developer project with
  no external contributors currently blocked by it — the same "real but not
  valuable enough" category as CLI `train`/`evaluate`/`predict`. Not
  pursued: no concrete discoverability problem is evidenced (nobody has
  hit this; there is no external audience yet).
- `char_rnn`/`word_rnn`'s shared training-loop duplication: re-examined:
  M56's own trigger ("a third independent change beyond M55's, or a third
  sequence workload") still hasn't fired — `git log` confirms M55's
  `compute_stream` commit (`4b7187e`) remains the most recent commit
  touching either training loop. **Reject, unchanged.**
- `benchmarks/training_bench.py`'s module docstring is stale — it still
  says "`forge.training.Trainer` has no CUDA integration", which stopped
  being true at M12 (six milestones and 40+ before this docstring was
  written, since that benchmark predates none of this — it's simply never
  been revisited). Real, but purely cosmetic (the code itself is correct;
  it deliberately uses a hand loop for a *different*, still-valid reason:
  comparing CPU/CUDA on identical code). Not worth a standalone milestone;
  noted here as a one-line fix a future milestone touching that file should
  make in passing, not pursued on its own.

### (4) Testing/reliability/infrastructure
- Full suite re-verified clean (Section 2). No CI configuration exists;
  re-examined and rejected on the same grounds as M57 (hosted CI cannot
  exercise the CUDA paths that are this project's own primary verification
  requirement; no missed-regression incident on record across 57 prior
  milestones; single-developer, single-GPU-hardware model). **Reject,
  unchanged.**
- **New this milestone — the one genuinely unexamined angle**: M55 found
  and fixed a real problem where CUDA's default-stream per-kernel-launch
  `cudaDeviceSynchronize()` made the two *RNN* examples train 1.5x-7.3x
  *slower* on CUDA than CPU, because their unrolled per-timestep loops issue
  many small kernel launches. That investigation was explicitly scoped to
  `char_rnn`/`word_rnn` only. **No milestone since M30 (`prefetch=True`'s
  introduction) has measured whether Forge's other, `Trainer`-based
  example — MNIST's CNN — has the same problem**, since `examples/mnist/
  train.py` does not pass `prefetch=True` and so also runs on the default
  CUDA stream. This is a real, previously-unmeasured question about a real,
  flagship workload (MNIST, UC1/UC3's primary demonstration).

  Measured directly (`benchmarks/training_bench.py`'s existing MLP-only
  probe does not cover this — a small throwaway script was used instead,
  reusing `examples/mnist/model.py::build_model()` unmodified against a
  synthetic random MNIST-shaped dataset — offline, no download needed — run
  through the real `Trainer`, 3 epochs, batch size 64, 2048 samples):

  | Configuration | Wall time |
  |---|---|
  | CPU | 4.76s |
  | CUDA, default stream (`examples/mnist/train.py`'s actual current config) | 1.37s |
  | CUDA, `prefetch=True` | 1.15s |

  **Result: CUDA is already ~3.5x faster than CPU for MNIST even without
  `prefetch=True`** (0.29x of CPU time); `prefetch=True` adds only a modest
  further 1.19x. This is the opposite of what M55 found for the RNN
  examples, and it is the expected result once explained: MNIST's Conv2d/
  matmul kernels each do far more arithmetic per launch than an RNN's
  per-timestep `Linear`+`tanh` step, so the fixed per-launch synchronize
  cost that dominated the RNN case is comparatively negligible here. **No
  bug exists in the MNIST path; M55's fix did not need extending. Reject,
  with fresh direct evidence rather than an assumption.**

### (5) Serialization/checkpointing/training infrastructure
No new gap; re-confirmed intact and fully exercised by the full suite
(unchanged since M49/M53).

### (6) Whether any CUDA optimization is newly worth revisiting
No new workload, no new measured bottleneck. `benchmarks/` still has no
post-M48 Conv2d entries; M55's RNN stream fix remains the most recent
CUDA-performance-relevant production change. Per the brief's explicit
instruction, Conv2d/RNN/allocator are left alone absent fresh evidence they
are a bottleneck for a real workload — none was found. **Reject,
unchanged.**

## 5. Candidate ranking

| Candidate | Real consumer | Evidence this milestone | Decision |
|---|---|---|---|
| MNIST default-CUDA-stream sync overhead (M55-style problem, unexamined for the CNN/Trainer path) | `examples/mnist` | Directly measured: CUDA already 3.5x faster than CPU even without `prefetch=True`; no problem exists | Reject — investigated, no bug found |
| Model/workload families (autoencoder, seq2seq, attention, etc.) | none new | No new primitive gap in any case | Reject, unchanged from M52/M54/M56/M57 |
| Shared RNN training-loop helper | `char_rnn`, `word_rnn` | M56's own trigger (third change or third workload) still not met | Reject, unchanged |
| CLI `train`/`evaluate`/`predict` | none | Conflicts with documented non-goal; no workload need | Reject, unchanged from M52 |
| CI configuration | — | No missed-regression incident; can't cover CUDA | Reject, unchanged |
| README quickstart/content expansion | none blocked today | No concrete discoverability problem evidenced (no external audience yet) | Reject — real but not valuable enough now |
| `training_bench.py` stale docstring | — | Cosmetic, one line, not worth a standalone milestone | Deferred, noted for next touch of that file |

## 6. Selected outcome

**C — NO PRODUCTION CHANGE.** This is the third assessment-only milestone in
a row (M56, M57, now M58), but each has independently gathered fresh
evidence rather than repeating the prior report — this milestone in
particular investigated and closed out a genuinely new, previously-unasked
question (whether M55's RNN stream-sync fix needed extending to the
`Trainer`/CNN path) with real measurement, and found no bug. No candidate
surveyed clears the project's evidence bar (a concrete, measured or
directly-observed gap with a real, non-speculative consumer).

## 7. Implementation / investigation details

No implementation. The one INVESTIGATE-grade question this milestone
identified (Section 4.4) was fully answered by direct measurement within
this milestone — MNIST's CUDA training path has no default-stream
performance problem — so it closes here rather than carrying forward as an
open question.

## 8. Files changed

- `docs/development/m58-post-m57-assessment.md` (new, this report).
- `docs/development/progress.md` (append-only milestone log entry, this
  milestone's own edit layered on top of M57's still-uncommitted entry).

No `forge/`, `examples/`, `tests/`, or `benchmarks/` file was modified.

## 9. Architecture/API impact

None. No public API, module, kernel, training-loop, or example behavior
changed.

## 10. Tests

No new tests (nothing in `forge/` changed to regression-test). Full suite
was run once, fresh, at the start of this milestone (Section 2):

```
python -m pytest tests/ -q
1771 passed in 82.70s
```

Identical to M56's and M57's own reported counts.

## 11. Verification

- CUDA availability was checked directly this milestone
  (`is_cuda_available() == True`), not assumed.
- `examples/word_rnn/train.py` was actually run end-to-end (1 epoch, CPU) to
  confirm the example still works from a real user's command line, not just
  via its integration test — it trained, sampled, saved, and reloaded
  correctly, and its generated artifact was correctly excluded by M57's
  `.gitignore` fix (confirmed via `git status`).
- `python -m forge --help` was run directly to confirm the CLI's actual
  current command surface.
- `forge/` was grepped directly for `TODO`/`FIXME`/`XXX` (none found).
- The MNIST default-stream measurement (Section 4.4) was executed directly
  on this machine's real CUDA hardware, not estimated.

## 12. Performance/resource impact

None — no production code changed. The MNIST probe (Section 4.4) is
throwaway, uncommitted, and not part of the shipped benchmark suite.

## 13. Rejected alternatives

Two are highlighted per the brief's expectation that at least one rejection
be a close call:

1. **MNIST default-CUDA-stream investigation** — the strongest candidate
   entering this milestone (a real, previously-unexamined question about a
   flagship workload, directly analogous to M55's accepted fix). Rejected
   only after being measured, not by extrapolation from the RNN case: CNN
   kernels are large enough per launch that the RNN-specific overhead does
   not reproduce here.
2. **README quickstart/content improvement** — real and easy, but rejected
   because no concrete discoverability problem is evidenced for a
   single-developer project with no current external audience; revisit if
   that changes (Section 14).

## 14. Limitations

- This is a point-in-time assessment against Forge's current five workload
  families and unchanged `forge/` tree (M55 remains the last production
  commit). It does not claim no future capability will ever be worth
  adding, only that none is evidenced now.
- The MNIST stream measurement (Section 4.4) used a synthetic random
  dataset at one representative size (2048 samples, batch 64, 3 epochs),
  not the real downloaded MNIST dataset — sufficient to answer the
  yes/no synchronization-overhead question (which depends on kernel launch
  count/shape, not on the data's semantic content), consistent with M55's
  own use of a small deterministic corpus for its RNN measurements.
- The `training_bench.py` stale-docstring finding (Section 4.3) was noted
  but deliberately not fixed standalone, since editing a file for one
  cosmetic line with no other change would itself be the kind of
  disproportionate-to-value action the brief's high bar discourages; it is
  recorded here so a future milestone that touches that file for a real
  reason fixes it in passing.

## 15. Practical impact on Forge

None directly — no code changed. Indirectly: this milestone closes out a
real open question (whether M55's RNN-specific stream fix has a hidden
counterpart bug in the CNN/Trainer path) with hardware-measured evidence
rather than leaving it as an assumption, and reconfirms, via fresh
execution rather than report-reading, that Forge's current capability set,
training model, and example coverage remain internally consistent.

## 16. Concrete future triggers

Unchanged from M56/M57, plus this milestone's own closed question removed:

- **If a third sequence-model workload is ever added**: revisit the shared
  `char_rnn`/`word_rnn` training-loop helper with three real data points.
- **If `char_rnn`/`word_rnn`'s hand-written loop needs a third independent
  change** beyond M55's `compute_stream` addition: treat as evidence of
  real maintenance cost.
- **If a future workload needs `softmax`/`transpose`/batched 3D matmul for
  a genuine, single reason**: let that reason drive which primitive is
  built first.
- **If Forge gains contributors or users beyond its current
  single-developer model**: revisit both the CI-configuration rejection and
  the README-usability rejection (Section 13.2) — both were rejected
  specifically because no external audience currently exists to be
  blocked by their absence.
- Otherwise, the highest-leverage trigger for a materially different M59
  finding remains a real external change to what Forge is asked to do, or
  an implementation milestone actually landing (M55 is still the last one).

## 17. Suggested commit message

```
docs: M58 post-M57 assessment — MNIST CUDA default-stream investigated (no
bug found); no production change warranted
```
