# M56 — Fresh Post-M55 Framework Engineering & Workload Assessment

## 1. Executive summary

M56's brief explicitly forbade defaulting to another RNNCell/Embedding/
Conv2d optimization pass and required a fresh, evidence-driven survey of
Forge's real post-M55 state before deciding whether any production change
is justified at all.

This milestone re-read every architecture document (`docs/architecture/`),
every product-direction/assessment milestone report from M49 through M55
(`docs/development/m49-capability-assessment.md` through
`m55-post-m54-assessment.md`), the full `forge/nn`/`forge/training`/
`forge/tensor`/`forge/data`/`forge/serialization` surface, and all five real
example workloads (`examples/mnist`, `examples/char_rnn`, `examples/
word_rnn`, `trainer_demo.py`, `data_pipeline_demo.py`,
`persistence_demo.py`), then re-ran the full test suite as a baseline.

**Finding: no new evidence has emerged since M55 that overturns any of the
candidates M49/M52/M54/M55 already evaluated and rejected**, and no new
candidate of comparable strength to M52's buffers/BatchNorm2d finding or
M54's Embedding finding was discovered. The one place a plausible new
angle exists — extracting `char_rnn`/`word_rnn`'s duplicated hand-written
training-loop structure into a shared helper — is examined in detail
(Section 7) and rejected on the same "sample size of two, no third
consumer, premature generalization" basis M49/M52 already used to reject
an analogous `forge.data` tokenization abstraction.

**Outcome D: assessment only.** No production code was changed. Full test
suite re-verified: **1,771 passed, 0 failed** — identical to M55's own
count, confirming zero drift and zero untested work already sitting in the
tree.

## 2. Post-M55 capability snapshot

Re-verified by direct inspection of the current tree, not assumed from
memory:

- **Tensor** (`forge/tensor/tensor.py`): `+ - * / @ sum reshape relu exp
  log tanh sqrt conv2d max_pool2d dropout_mask cross_entropy batch_norm2d
  (CUDA-only) embedding_lookup rnn_cell (CUDA-only) to backward`. Confirmed
  still absent: `.mean()`, `.transpose()`/`.permute()`, `.softmax()`,
  `.sigmoid()`, `__pow__`, `__neg__` as a public op, general N-D CUDA
  broadcast/reduction beyond the scoped shapes each consumer needed.
- **`nn`** (`forge/nn/__init__.py`, confirmed by direct import list):
  `Module Parameter Linear ReLU Tanh Conv2d MaxPool2d Sequential Flatten
  Dropout RNNCell BatchNorm2d Embedding Loss MSELoss CrossEntropyLoss`. No
  `Sigmoid`/`LSTM`/`GRU`/`LayerNorm`/attention primitives — unchanged from
  M55, still no consumer for any of them.
- **`Module`**: `_parameters`/`_modules`/`_buffers` (M53), full
  discovery/device-movement/persistence/train-eval support for all three.
- **Optimizers**: `SGD` (no momentum, deliberate), `Adam` (full). No LR
  scheduler — still no workload need.
- **Data**: `Dataset`/`TensorDataset`/`Subset`/`random_split`,
  `DataLoader`, `Compose`/`ToTensor`/`Normalize`/`Reshape`/`Flatten`/
  `Lambda`, `CUDAPrefetchLoader`. CPU-only by deliberate, permanent
  boundary (`docs/architecture/data-system.md`).
- **Training**: `Trainer` (`fit`/`evaluate`, checkpoint/resume, metrics,
  opt-in `prefetch=True` compute-stream path). Still one-`forward()`-call-
  per-step by design; sequence models still use a hand-written loop
  (Section 7).
- **Serialization**: parameters + buffers persist generically,
  `FORMAT_VERSION`/`CHECKPOINT_FORMAT_VERSION` both `2`, unchanged since
  M53.
- **CUDA backend**: real kernels for every CPU-supported op except general
  N-D broadcast/reduction (unchanged scope boundary since M9/M14/M53); real
  async-stream machinery (M27/M28/M29/M30) now demonstrably used by both
  `Trainer(prefetch=True)` and, as of M55, both hand-written RNN example
  loops.
- **Examples**: unchanged in architecture from M55 — MNIST CNN (with and
  without BatchNorm2d), char-RNN, word-RNN, plus the three CPU-only
  milestone-verification demos.

Nothing in this list is new since M55's own snapshot; this milestone's
contribution is confirming it by direct re-inspection rather than assuming
it carried forward unchanged.

## 3. Current real workloads

Five real, currently-exercised workload families, all inspected directly
(`examples/*/train.py`/`model.py`):

| Workload | Modules exercised | Trainer or hand-loop | Why |
|---|---|---|---|
| MNIST CNN (`examples/mnist`) | `Conv2d`, `MaxPool2d`, `ReLU`, `Flatten`, `Linear`, optional `BatchNorm2d`, `CrossEntropyLoss`, `Adam` | **`Trainer.fit()`** | One `forward(batch)` call per step fits `Trainer`'s shape exactly; also drives checkpoint/resume and `Accuracy` metric. |
| char-RNN (`examples/char_rnn`) | one-hot input, `RNNCell`, `Linear`, `CrossEntropyLoss`, `Adam` | **Hand-written loop** (`train_one_epoch`) | Per-timestep recurrence over a shared hidden state does not fit `Trainer`'s one-call-per-step shape (module docstring explains this explicitly). Uses M55's `compute_stream` pattern. |
| word-RNN (`examples/word_rnn`) | `Embedding`, `RNNCell`, `Linear`, `CrossEntropyLoss`, `Adam` | **Hand-written loop**, structurally identical to char-RNN's | Same reason; model docstring states it is "structurally identical to `examples/char_rnn/model.py`". |
| `trainer_demo.py` (regression + classification) | `Linear`, `ReLU`, `MSELoss`/`CrossEntropyLoss`, `SGD` | **`Trainer.fit()`** | Canonical single-forward-per-step case; exercises validation split, `Accuracy`/`MeanAbsoluteError`. |
| `persistence_demo.py` | `Linear`, `MSELoss`, `SGD` | **`Trainer.fit()`** | Train → save → cold-process load → predict round trip. |

`data_pipeline_demo.py` uses no training loop at all (dataset/transform/
loader wiring only).

**Observed awkwardness**: `char_rnn/train.py::train_one_epoch` and
`word_rnn/train.py::train_one_epoch` are near-line-for-line duplicates —
`zero_grad()` → `init_hidden()` → per-timestep forward/loss-accumulate loop
→ mean-loss scale → `backward()` → `step()` → the identical
`compute_stream` context-manager wrapper M55 added to both files
independently. This is examined as a candidate in Section 7 and rejected.

No other awkwardness was found: MNIST's `Trainer` path, the demo scripts,
and both RNN examples' save/load-parity checks all run cleanly (confirmed
by the full suite re-run, Section 15) and match their own documentation.

## 4. Framework strengths

Worth stating explicitly, since M56's brief asks whether Forge has "a
coherent training model or merely a collection of working mechanisms":

- **Two training patterns, each with a clear, documented boundary**:
  `Trainer` for any model whose forward pass is one call per step (3 of 5
  workload families), and a hand-written `zero_grad → loop → backward →
  step` loop for multi-timestep recurrence (2 of 5) — not an accident, but
  a decision independently re-affirmed by M50, M54, and M55, each time
  re-examining whether extending `Trainer` was warranted and each time
  finding it was not.
- **The M55 stream-scope pattern generalizes cleanly**: it required zero
  changes to `Trainer`, `Tensor`, or `forge/autograd/engine.py` — ambient
  stream state made both `Trainer(prefetch=True)` (M30) and the hand-written
  RNN loops (M55) pick it up automatically, without either code path
  knowing about the other.
- **Persistence, device movement, and buffers all compose without
  per-module special-casing** — `BatchNorm2d`'s buffers and `Embedding`'s
  ordinary `Parameter` both round-trip through the same generic tree walk
  introduced at M7 and extended once (M53) since.
- **The project's own evidence discipline is holding**: four consecutive
  milestones (M49, M52, M54, M55) have each independently re-examined the
  same candidate list (binary classification, deeper/stacked RNN,
  LayerNorm, attention, optimizer/scheduler changes, sparse Embedding
  gradients, further RNN backward fusion) against fresh workload evidence
  and reached the same rejection every time — this is a sign of a stable,
  well-scoped framework, not a stalled one.

## 5. Problems/gaps discovered

**None material.** The one candidate this milestone examined that no prior
milestone had specifically named — deduplicating `char_rnn`/`word_rnn`'s
training-loop structure — is real but small (Section 7) and does not clear
the bar the project has consistently applied to similar "only two
consumers" cases (M49/M52's tokenization-abstraction rejection).

No reliability issue was found. The full test suite (Section 15) passes
identically to M55's own count with no skips, and no `docs/architecture/*`
"Known limitations" section names an open defect (as opposed to a
deliberate, documented scope boundary) that a real workload has actually
hit.

## 6. Candidate directions

Re-examined, not re-invented, against this milestone's own fresh reading:

- **A. Model capability**: binary classification, autoencoder,
  deeper/stacked RNN, attention/transformer, another normalization
  architecture. All previously rejected (M49/M52/M54); no new evidence.
- **B. Tensor/autograd capability**: `.mean()`, `.transpose()`/`.permute()`,
  `.softmax()`, `.sigmoid()`, general N-D CUDA broadcast/reduction. All
  previously deferred for lack of a concrete consumer; still true.
- **C. Training infrastructure**: extending `Trainer` for multi-step
  sequence training (repeatedly rejected, M50/M54/M55); a shared
  sequence-training-loop helper for `char_rnn`/`word_rnn` (new this
  milestone, Section 7).
- **D. Reliability/engineering quality**: none found (Section 5).
- **Performance**: further RNNCell backward-kernel fusion, sparse Embedding
  gradients, another Conv2d optimization pass, another general CUDA
  bottleneck sweep — all explicitly excluded by this milestone's brief and
  re-confirmed to still lack fresh evidence (Section 8).

## 7. Evidence for each candidate

### A. Model capability — no change from M54's evidence

- **Binary classification**: still fully expressible via
  `CrossEntropyLoss(num_classes=2)` (`forge/nn/loss.py`, unchanged code
  path, confirmed by direct read this milestone). No gap. Reject.
- **Autoencoder**: composes entirely from `Linear`/`Conv2d`/`ReLU`/
  `MSELoss`/`Adam`, all already proven by the existing MNIST/`trainer_demo`
  workloads. No new framework pressure — exactly M52's "cosmetic repository
  growth" pattern. Reject.
- **Deeper/stacked RNN**: still pure Python composition of `RNNCell`
  instances, no primitive gap (unchanged reasoning since M50). Reject.
- **Attention/transformer**: still simultaneously blocked on `softmax`,
  `transpose`/`permute`, and batched 3D matmul, with no concrete workload
  driving any one of them individually. Reject (unchanged from M52/M54).
- **LayerNorm**: still composable from M53's `sum`/`sqrt`/`div` on CPU, but
  still has no consumer — neither RNN example needs it to converge
  (re-confirmed: both trained to a large loss reduction with no
  normalization layer, per M55's own runs, unchanged this milestone).
  Reject.

### B. Tensor/autograd capability — no change from M54's evidence

No workload in the current tree calls or would benefit from `.mean()`,
`.transpose()`, `.softmax()`, or `.sigmoid()` — every place a mean is
needed (`BatchNorm2d`) already composes it from `sum` * `1/count`; every
place softmax-like behavior is needed (`generate()` in both RNN examples)
is post-training, non-differentiable NumPy sampling that has no reason to
be a differentiable Tensor primitive. Reject all, unchanged.

### C. Training infrastructure

**`Trainer` sequence-training support** — re-examined against this
milestone's own reading of `train_one_epoch` in both RNN examples: the
per-timestep hidden-state threading, per-step loss accumulation, and
single `backward()` call at the end of an unrolled sequence still do not
map onto `Trainer`'s one-`forward()`-call-per-step contract without either
(a) `Trainer` growing an entirely new sequence-mode API surface, or (b)
`Trainer` becoming a leaky abstraction that hands the caller back partial
control mid-`fit()`. No third sequence-model workload exists to validate
either design against. Reject, unchanged from M50/M54/M55.

**Shared sequence-training-loop helper (new candidate this milestone)** —
`char_rnn/train.py::train_one_epoch` and `word_rnn/train.py::
train_one_epoch` share: batch unpacking, `optimizer.zero_grad()`,
`model.init_hidden()`, a per-timestep forward+loss-accumulate loop, mean-
loss scaling, `backward()`/`step()`, loss/sample-count bookkeeping, and
(since M55) an identical `compute_stream` context-manager wrapper. The only
per-example differences are the per-timestep input construction (one-hot
NumPy array vs. raw int64 token ids) and the loss-normalization divisor
name (`total_chars` vs. `total_tokens`).

- **Real consumer**: both RNN examples, today.
- **Current limitation**: ~50 lines of near-identical control flow
  maintained in two files; M55 already had to make the identical
  `compute_stream` change in both places independently.
- **Existing workaround**: none — the duplication is simply accepted.
  Workaround quality: adequate (both files are short, well-tested, and
  drifted zero times in practice — M55's identical change is the only time
  either has needed to change since M50/M54).
- **Reusability**: exactly two consumers exist. A third sequence-model
  workload (deeper stacked RNN, an attention block) is exactly the kind of
  speculative addition Section 6/A above just rejected — so no near-term
  third consumer is anticipated to validate a shared-helper design against.
- **Implementation cost**: low (a small `forge.training` or example-shared
  helper function/module), but any such module either duplicates
  `Trainer`'s public shape under a different name (confusing — two ways to
  train a model that don't compose) or introduces a new, narrower
  abstraction (`unroll_and_step(model, x_seq, y_seq, ...)`) whose contract
  would be guessed from a sample size of two data points, precisely the
  failure mode M49/M52 already named and avoided when rejecting a shared
  `forge.data` tokenization abstraction for `char_rnn`/`word_rnn`'s
  `Vocab` classes.
- **Regression risk**: low technically, but a wrong abstraction boundary
  chosen now would need to be revisited (and likely broken) the moment a
  genuinely different sequence-model shape (e.g. teacher forcing, a
  stacked RNN with per-layer hidden states, or an attention block with no
  single "hidden state") arrives — exactly the scenario the project has
  consistently deferred generalization for elsewhere.

**Decision: reject, revisit if and when a third sequence-model workload
exists** — see Section 22.

### Performance (explicitly constrained by this milestone's brief)

- **Further RNNCell backward fusion**: brief explicitly excludes revisiting
  this without evidence of a new, larger workload; none exists. Reject.
- **Sparse Embedding gradients**: re-confirmed, still no measured slowdown
  at any tested vocabulary size (M54's own measurement, unchanged). Reject.
- **Conv2d**: brief explicitly excludes reopening without a new workload
  and a genuinely new optimization direction; neither exists. Reject.
- **A generic CUDA bottleneck sweep**: brief explicitly excludes this as a
  default action; no real workload in Section 3 shows a new, unexplained
  slowdown to investigate. Reject.

## 8. Candidate ranking

| Candidate | Real consumer | Severity | Evidence this milestone | Decision |
|---|---|---|---|---|
| Shared sequence-loop helper | `char_rnn`, `word_rnn` | Minor improvement | ~50 lines of duplicated control flow, one prior identical-change incident (M55's `compute_stream`) | Reject — 2 consumers, no validated abstraction boundary |
| Any new `nn`/Tensor primitive (Sec 6/7 A-B) | none | Speculative | No workload evidence, unchanged from M49-M55 | Reject |
| `Trainer` sequence-mode API | `char_rnn`, `word_rnn` | Speculative | No workload needs it beyond what the hand-loop already handles | Reject |
| Further RNNCell/Embedding/Conv2d optimization | existing RNN/CNN examples | Speculative | Explicitly excluded by this milestone's brief; no new measurement contradicts M55/M54/M48 | Reject |
| Reliability fix | — | None found | Clean full-suite run, no reproducible defect | N/A |

No candidate reached "meaningful limitation" or higher.

## 9. Rejected directions

All of Section 7's rejections apply. The two explicitly required by the
brief's "at least two plausible technical directions should be rejected if
their benefit does not justify their cost":

1. **Shared sequence-training-loop helper** — real, measurable duplication
   exists, but with only two consumers and no anticipated third, any
   extracted abstraction's contract would be guessed rather than
   evidenced, and a wrong guess would need reworking the moment a
   genuinely different sequence-model shape arrives. Rejected on cost
   (design-risk relative to two files' worth of ~50 duplicated lines) vs.
   benefit.
2. **`Trainer` sequence-mode support** — would resolve the duplication more
   thoroughly, but at far higher cost (a new API surface on Forge's most
   central training abstraction) for the same two-consumer evidence base,
   and has already been rejected on identical grounds by three prior
   milestones (M50, M54, M55) with no new evidence to overturn it this
   time. Rejected.

## 10. Selected outcome

**Outcome D — Assessment only.** No candidate surveyed clears the
project's own established bar (a concrete, measured or directly-observed
gap with a real, non-speculative consumer) for implementation this
milestone. No production code was changed.

## 11. Root cause, if applicable

Not applicable — no defect was found or pursued.

## 12. Design/implementation, if applicable

Not applicable.

## 13. Files changed

- `docs/development/m56-post-m55-assessment.md` (new, this report).
- `docs/development/progress.md` (append-only milestone log entry).

No `forge/`, `examples/`, or `tests/` file was modified.

## 14. API/architecture impact

None. No public API, module, kernel, or training-loop behavior changed.

## 15. Tests

No new tests were added (nothing changed to regression-test). Per the
brief's Testing Requirements, the existing suite was re-verified rather
than skipped:

```
python -m pytest tests/ -q
1771 passed in 69.51s
```

Identical to M55's own reported count (1,771), confirming zero regression
and zero untested change already present in the tree at the start of this
milestone.

## 16. Verification

Verification for an assessment-only outcome consists of confirming the
survey itself is accurate, not of validating new code:

- Every `nn`/Tensor/training/data/serialization capability claim in
  Section 2 was checked against the current source (`forge/nn/__init__.py`'s
  actual export list, `forge/tensor/tensor.py`'s method surface, direct
  reads of `forge/training/trainer.py`, `forge/nn/module.py`,
  `forge/data/*.py`, `forge/serialization/*.py`), not carried forward from
  memory of M55's report.
- Every example's training pattern (Section 3) was confirmed by reading
  `examples/*/train.py`/`model.py` directly, not inferred from docstrings
  alone (though the docstrings' own stated reasoning matched what the code
  does in every case).
- The full test suite was executed on this machine (Section 15), not
  assumed passing.

## 17. Real workload results

No workload was retrained or re-benchmarked — this milestone made no
change that could affect training behavior or performance. Every example's
existing, previously-verified behavior (M50/M52/M53/M54/M55's own reported
results) stands unchanged, confirmed indirectly by the identical full-suite
pass count (Section 15), since every example's integration test
(`tests/test_*_example_*integration.py`) re-ran and passed.

## 18. Performance results, if applicable

Not applicable — no performance work was undertaken, per this milestone's
explicit RNNCell/Embedding/Conv2d/generic-sweep exclusions and the absence
of any fresh evidence to justify overriding them.

## 19. Limitations

- This is a point-in-time assessment against Forge's current five workload
  families; it is not a claim that no future capability will ever be
  worth adding, only that none is evidenced *now*.
- The "shared sequence-loop helper" rejection (Section 7/9) is a judgment
  call, not a hard measurement — reasonable engineers could differ on
  whether ~50 duplicated lines across two files justifies an abstraction
  today. This report documents the reasoning (two consumers, no validated
  contract, precedent from M49/M52's identical tokenization-abstraction
  rejection) so a future milestone can revisit it explicitly if a third
  sequence-model workload appears.
- No hardware-level (940MX) measurement was taken this milestone, since no
  performance candidate reached the bar for investigation (Section 8 of
  this milestone's brief: only benchmark a candidate once a real workload,
  material cost, and credible optimization direction are all first
  established — none were).

## 20. Practical impact on Forge

None directly — no code changed. Indirectly, this milestone re-confirms
(rather than merely assumes) that Forge's current capability set, training
model, and example coverage remain internally consistent and that the
project's evidence-first discipline (four consecutive prior milestones
reaching the same rejections independently) continues to hold under a
fresh, skeptical re-read rather than becoming a rubber stamp.

## 21. Explicit value justification

The value of an Outcome D milestone is the confidence it produces, not
code delivered: this milestone independently re-derived (not copied) the
current capability snapshot, re-examined every previously-rejected
candidate against fresh reading of the actual source and every example,
found and seriously evaluated one new candidate (the sequence-loop
duplication) rather than stopping at "nothing to report," and confirmed
via a real full-suite run that the repository is exactly as healthy as
M55 left it. This matches the brief's own success criterion: "Forge should
have a better-evidenced understanding of the highest-value next
engineering step" — which here is the explicit conclusion that no step
currently clears the bar, plus a concrete trigger condition (below) for
when that could change.

## 22. Recommendation for next milestone

Not a predetermined feature. Concrete triggers worth watching for:

- **If a third sequence-model workload is ever added** (a deeper stacked
  RNN, teacher forcing, an attention block): revisit the shared
  sequence-training-loop helper (Section 7) with three real data points
  instead of two — likely enough evidence to choose a real abstraction
  boundary rather than guess one.
- **If `char_rnn`/`word_rnn`'s hand-written loop needs a third independent
  change** (beyond M55's `compute_stream` addition): treat that as
  evidence the duplication has a real maintenance cost, not just a
  theoretical one, and revisit then.
- **If a future workload needs `softmax`/`transpose`/batched 3D matmul for
  a genuine, single reason** (not "attention might be nice"): that reason,
  not attention architecture novelty, should drive which primitive gets
  built first.
- Otherwise, repeat this milestone's own method: measure or directly
  observe a real workload before assuming the next investment is another
  primitive, another layer, or another kernel optimization.

## 23. Suggested commit message

```
docs: M56 post-M55 assessment — no production change warranted
```
