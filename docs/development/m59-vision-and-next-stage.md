# M59 — Forge Vision Reassessment & Next-Stage Product/Engineering Plan

## 1. Executive Summary

M49–M58 ran a ten-milestone loop that repeatedly asked the same narrow
question — "does the current workload set justify one more Tensor
primitive/`nn` layer/kernel optimization?" — and, correctly, mostly
answered no. That loop was itself diagnosed as exhausted by M56, M57, and
M58's own reports. M59's brief asks a different, one-level-higher
question: is Forge, as a *project*, becoming the thing `docs/product/
vision.md` describes, or has it become a well-tested pile of individually
justified additions?

The direct evidence, gathered this milestone by reading the vision/
requirements/use-case documents, every prior milestone report, and the
actual source of `forge/tensor/tensor.py`, `forge/nn/module.py`,
`forge/backend/base.py`, `forge/exceptions.py`, `forge/training/
trainer.py`, `forge/optim/optimizer.py`, `forge/cli/*`, all three example
READMEs, and the full test suite (re-run fresh: **1,771 passed**, identical
to M58), is more encouraging about the *code* than the assessment-loop
framing implied, and more concerning about the *project surface around the
code* than any of M49–M58 ever measured, because none of them looked there.

**The core finding**: Forge's engineering core — `Tensor`, `Module`,
`Backend`, exceptions, `Trainer`, the CUDA backend — is unusually coherent
for ten milestones of incremental, evidence-gated growth. Every fused
primitive added since M31 (`cross_entropy`, `batch_norm2d`,
`embedding_lookup`, `rnn_cell`) follows an identical, discoverable file
-touch pattern (ABC method in `backend/base.py` → `cpu.py` → `cuda/
kernels.cu` + `cuda/backend.py` → `Tensor` method → consumer `nn.Module` →
`serialization/registry.py` if needed → tests), documented in `docs/
architecture/decisions/`. Every public error is a typed `ForgeError`
subclass with a specific, actionable message — no bare NumPy exception was
found leaking through any sampled path. This is *not* "a pile of
individually bolted-on capabilities" at the code level; the discipline the
milestone reports describe in prose is genuinely reflected in the source.

**What is not coherent, and has never been assessed because no milestone
was scoped to look**: the *project* around that code. Forge demonstrates
only two of the vision's three named workload classes at production
quality (image classification, sequence/language modeling); regression —
named explicitly in `vision.md` and `use-cases.md` (UC2) — has only a
leftover milestone-verification script (`examples/trainer_demo.py`, no
README, not a "real example" by the project's own established convention).
The top-level onboarding surface (`README.md`, `forge/__init__.py`'s module
docstring) is stale relative to the framework's actual, current
capability — a newcomer's first two reads of the project would undersell
it. And the milestone-report process itself has become heavy: five of the
last ten milestones were assessment-only, each 15–28KB, increasingly
re-deriving the same conclusions from an increasingly unchanged tree.

**Decision (Section 12/21): a bounded, targeted implementation direction
(the third named-in-vision workload family) plus explicit process
guardrails that end the assessment-loop pattern as a default milestone
shape.** No `forge/` code is changed in M59 itself — per the brief's own
Section 15/18, this is a strategic decision milestone, and the identified
implementation (Section 18/M60) is real engineering work (a full example,
README, CPU+CUDA integration tests) that deserves its own milestone budget,
exactly the precedent M52 set for BatchNorm2d.

## 2. Original Forge Vision

From `docs/product/vision.md`, `requirements.md`, `use-cases.md`,
`scope.md` (read fresh this milestone, not recalled from any prior
report):

- **Purpose**: "a general deep-learning framework for constructing,
  training, evaluating, saving, loading, and executing neural-network
  models."
- **Primary user**: "A Python developer who wants to define neural
  networks and training workflows through Forge's APIs without depending
  on PyTorch or TensorFlow for the core learning machinery."
- **Supported workload direction**: explicitly named as "classification,
  regression, image workloads, and additional neural-network tasks... from
  reusable primitives rather than separate application-specific systems."
- **Success** (vision.md's own definition, verbatim): "A developer can use
  Forge to build and train real small models, evaluate them, persist them,
  reload them, perform inference, and execute supported workloads on CPU
  and CUDA where available."
- **Use cases** (`use-cases.md`): UC1 classifier, **UC2 regression**, UC3
  image workload, UC4 custom dataset, UC5 CPU/CUDA parity, UC6
  persistence, UC7 benchmarking.
- **Scope rule**: "Future capabilities may influence interfaces, but
  implementation should prioritize representative, working functionality
  over breadth." Distributed training, large-model training, cloud
  infrastructure, and wrapping PyTorch/TensorFlow are explicitly out of
  scope — permanently, not "not yet."

This vision has not changed in any of the 58 prior milestones' own
re-readings, and this milestone's own re-reading confirms it is still the
operative document — nothing in `forge/`'s evolution contradicts it.

## 3. Original Success Criteria

Vision.md's "Success" sentence (Section 2 above) is the only explicit
success definition in the product docs, and it is a *capability* statement,
not a feature-count or milestone-count statement — consistent with
`docs/development/roadmap.md`'s own "not a fixed number of milestones"
framing and the brief's Section 17 ("not maximize feature count... but
become useful, credible, coherent"). `requirements.md`'s functional
requirements operationalize it: tensor computation, autograd, composable
`nn`, losses/optimizers, dataset/dataloader abstractions, a training
engine, persistence, a CPU baseline + first-class real CUDA backend, a CLI
delegating to public APIs, reproducible benchmarking, and clear errors.

## 4. Current Capability State

Verified directly this milestone (source, not memory of prior reports):

| Area | State |
|---|---|
| Tensor/autograd | `+ - * / @`, `sum`, `reshape`, `relu/exp/log/tanh/sqrt`, `conv2d`, `max_pool2d`, `dropout_mask`, `cross_entropy`, `embedding_lookup`, `batch_norm2d`(CUDA)/`rnn_cell`(CUDA), `.to()`, `.backward()`. Every op: typed shape/dtype/device validation, a real `ForgeError` on misuse. |
| `nn`/`Module` | `Linear, Conv2d, MaxPool2d, ReLU, Tanh, Dropout, Flatten, Sequential, RNNCell, BatchNorm2d, Embedding, MSELoss, CrossEntropyLoss`. `_parameters`/`_buffers`/`_modules`, full discovery/device-move/persistence/train-eval support for all three (M53 closed the one real Module-level gap found in this whole 58-milestone history). |
| Optimizers | `SGD` (no momentum, deliberate), `Adam` (full). No scheduler — no workload has ever needed one. |
| Data | `Dataset`/`TensorDataset`/`Subset`/`random_split`, `DataLoader`, `Compose`/`ToTensor`/`Normalize`/`Reshape`/`Flatten`/`Lambda`, `CUDAPrefetchLoader`. CPU-only by permanent design (`docs/architecture/data-system.md`). |
| Training | `Trainer` (`fit`/`evaluate`, checkpoint/resume, metrics, opt-in `prefetch=True`) for one-forward-per-step models; hand-written loops (documented, tested, now stream-aware since M55) for multi-timestep recurrence. Two training patterns, one clear boundary — not an accident (re-affirmed independently by M50/M54/M55). |
| Serialization | Parameters + buffers + optimizer state + RNG state, versioned archive format (`FORMAT_VERSION 2`), CLI-inspectable without CUDA or module registration. |
| CPU backend | Complete for every op above; the reference implementation every CUDA op is checked against. |
| CUDA backend | Real kernels for every CPU-supported op except general N-D broadcast/reduction (a deliberate, repeatedly-reaffirmed scope boundary, not a gap — every consumer that needed a specific shape got a specific, scoped kernel). Hardware-verified on the reference 940MX for every claim in every milestone report. |
| CLI | `forge model/checkpoint inspect|convert`, `forge benchmark` — real, useful, but not a training interface; `train`/`evaluate`/`predict` remain unbuilt, a documented, deliberate M19/M52 scope narrowing (needs a config/YAML system Forge has explicitly declined to build). |
| Tests | **1,771 passed** (this milestone's own fresh run, re-verified independently — see Verification), 103 test files, 57 CUDA-specific (by filename), 13 integration-test files (8 of which are example-integration tests, covering all three real examples CPU+CUDA), zero `TODO`/`FIXME` in `forge/` (M58's grep, re-confirmed unchanged). |
| Examples | Three real, trained, checkpointed, CPU/CUDA-verified workloads (`mnist`, `char_rnn`, `word_rnn`) plus three unREADME'd milestone-verification demo scripts. |
| Packaging | `pyproject.toml`, version `0.1.0` — unchanged across all 58 milestones. No `CHANGELOG`. `.idea/` (JetBrains project files) is tracked in git (6 files) — harmless but not a deliberate policy. No `.github/` CI. |

## 5. Vision-vs-Reality Gap Analysis

| Dimension | Classification |
|---|---|
| A. Tensor/autograd system | **Strong enough** |
| B. NN/module system | **Strong enough** — buffers (M53) closed the one real architectural gap this project has ever found by direct probe |
| C. Model architecture coverage | **Adequate but immature** — two of three vision-named workload classes have flagship examples; regression does not |
| D. Optimizers | **Strong enough** for every real workload; intentionally minimal |
| E. Training infrastructure | **Strong enough** for single-forward-per-step models; **adequate but immature** for multi-step models (accepted, documented duplication between two examples) |
| F. Data loading | **Strong enough** |
| G. Serialization/checkpointing | **Strong enough** |
| H. CPU backend | **Strong enough** |
| I. CUDA backend | **Strong enough** for every real workload; scope boundary (general N-D broadcast/reduction) is deliberate, not a gap |
| J. Device/stream management | **Strong enough** — M55 closed the one measured real defect (RNN default-stream overhead) |
| K. Memory allocator | **Strong enough** — M51 fixed the one correctness defect ever found |
| L. Numerical correctness | **Strong enough** — finite-difference + CPU/CUDA parity checked for every differentiable op |
| M. Testing/reliability | **Adequate but immature** — 1,771 tests is real coverage, but 57/103 files are hardware-coupled to one specific GPU, and there is no CI of any kind |
| N. Developer ergonomics/API coherence | **Strong enough** — verified directly this milestone (Section 7) |
| O. Documentation/examples | **Adequate but immature** — architecture docs are living and accurate; top-level onboarding docs (`README.md`, `forge/__init__.py`) are stale |
| P. Extensibility/maintainability | **Strong enough** — a new op/layer follows one discoverable, documented, repeated pattern |
| Q. Performance methodology | **Strong enough / mature** — an explicit measure-before-optimize discipline, followed consistently since M47 |
| R. Real-world workload credibility | **Adequate but immature** — three trained/checkpointed/CPU-CUDA-verified workloads is real, but two of three corpora are synthetic and the third (MNIST) is the only external real-world dataset in the whole project |

No dimension is a **Major blocker**. This matters: it means M59's honest
conclusion is not "Forge is broken," it is "Forge's core is solid and its
project surface has a specific, nameable, boundedly fixable set of gaps."

## 6. Workload Credibility

Three real, trained, checkpointed, save/load-verified, CPU-and-CUDA
-hardware-verified workloads exist: MNIST CNN classification (`Trainer
`-driven, real external dataset), char-level RNN language modeling
(hand-written loop, synthetic corpus), word-level RNN language modeling
(hand-written loop, synthetic corpus, `Embedding`). This satisfies UC1,
UC3, UC5, UC6 concretely and repeatedly. It does **not** satisfy UC2
(regression) at the same standard — `trainer_demo.py` exercises a
regression model but is, by every prior milestone's own classification
(M56 Section 3, M57 Section 2), a milestone-verification script, not an
example: no README, no CUDA-verification narrative, no checkpoint/resume
demonstration, not held to the `mnist`/`char_rnn`/`word_rnn` convention.
UC4 (custom dataset) is satisfied only incidentally, by MNIST's own
`Dataset` subclass — never demonstrated as its own named capability.

char-RNN and word-RNN are, honestly, one workload family (recurrent
language modeling) at two vocabulary granularities, not two — M57's own
Section 3 already says this in different words ("not two distinct model
families"). So the accurate count is **two** demonstrated workload
families against a vision that names three (classification, regression,
image — with image already folded into classification via MNIST). A third
family that is a genuine recombination of already-existing primitives
(Linear/ReLU/MSELoss/Adam/Trainer, all independently proven) would not
expose a framework gap, and per M49/M52/M54's own well-established
"cosmetic repository growth" test, that is normally a reason to reject it —
**except** that this one is not proposed for framework-pressure reasons,
it is proposed because it is a named, unmet piece of the original vision
(Section 2) that the project has simply never gotten around to finishing
at production quality. That is a different, legitimate justification the
narrow op-survey milestones were never scoped to consider.

## 7. Framework Maturity Assessment

Verified by direct source reading this milestone (`forge/tensor/tensor.py`,
`forge/nn/module.py`, `forge/backend/base.py`, `forge/exceptions.py`,
`forge/training/trainer.py`, `forge/optim/optimizer.py`,
`forge/cli/main.py`, `docs/development/cli.md`, all three example
READMEs, `docs/architecture/tensor-api.md`):

- **API coherence**: every binary op validates device match then shape
  broadcastability before touching the backend, in the same order, raising
  `UnsupportedDeviceError`/`ShapeMismatchError` with the operand shapes
  inlined in the message every time (`tensor.py:239-250`, `:290-301`,
  `:447-483`, `:515-527`, `:581-593`, `:625-633`). Every fused primitive
  added since M31 carries a docstring naming the milestone that added it,
  the mathematical derivation of its backward rule, and a cross-reference
  to the milestone report that justified it. This is unusually dense,
  consistent documentation for hand-written framework code.
- **Error handling**: `forge/exceptions.py` defines eleven `ForgeError`
  subclasses (`ShapeMismatchError`, `UnsupportedDTypeError`,
  `UnsupportedDeviceError`, `GradientStateError`, `ModuleError`,
  `LossError`, `OptimizerError`, `DataError`, `TrainerError`, `CUDAError`,
  `PersistenceError`), each with a docstring naming its concrete trigger
  conditions.
  No sampled path (`Tensor.__init__`, `_binary_op`, `matmul`, `sum`,
  `conv2d`, `Module.__setattr__`, `Trainer.__init__`) was found raising a
  bare Python/NumPy exception for a user-facing misuse; internal
  invariant violations are left to propagate with a full traceback by
  design (`docs/development/cli.md`'s own stated policy), which is a
  reasonable, explicit choice, not an oversight.
- **Extensibility**: adding a new op touches the same ~6 files in the same
  order every time (`backend/base.py` ABC method → `cpu.py` → `cuda/
  kernels.cu` + `cuda/backend.py` → `Tensor` method → consumer `nn.Module`
  → tests), and this pattern is documented in `docs/architecture/decisions/
  ADR-002-reference-backend.md`/`ADR-005-backend-aware-autograd.md`. A
  third-party backend author has a stable, if unversioned, ABC contract to
  implement against.
- **CLI**: genuinely useful for what it claims — read-only, no-CUDA-needed
  archive inspection and explicit-device conversion — and explicitly,
  documentedly not a training interface. Not a stub; a correctly scoped
  thin adapter.
- **Examples as onboarding**: `mnist`/`char_rnn`/`word_rnn`'s READMEs are
  genuine tutorials — prerequisites, exact commands, expected numeric
  output, CUDA notes, determinism notes, a link to the framework addition
  each required. A newcomer could follow any of the three end-to-end
  without reading `forge/` source. This is a real strength the narrow
  milestone assessments never had reason to state explicitly.
- **Documentation coherence — the one real inconsistency found**:
  `docs/architecture/tensor-api.md` is a living reference, edited in place
  through M55 (still correctly describes `rnn_cell`, `embedding_lookup`,
  etc.). `forge/__init__.py`'s own module docstring is not: it stops
  narrating at Milestone 26 (`memory_stats`/`synchronize`) and never
  mentions M31's `cross_entropy` fusion, M53's buffers/`BatchNorm2d`, M54's
  `Embedding`, or M55's fused `RNNCell` — a reader of `import forge; help
  (forge)` today gets a materially incomplete picture of the framework's
  own front door. `README.md` is similarly thin (unchanged finding from
  M58, re-verified: no quickstart, no feature list, accurate but
  minimal). This is the clearest, most concrete evidence this milestone
  found of documentation *not* staying in sync with code — everywhere else
  checked, it does.
- **Packaging/release**: version `0.1.0` since inception, no `CHANGELOG`,
  no CI. `.idea/` tracked in git — minor, harmless hygiene gap, not
  pursued further (same "real but not valuable enough" bar M57/M58 used
  for comparable findings).

**Single biggest structural observation**: Forge does not feel like
individually bolted-on capabilities *at the code level* — the opposite is
true, and the evidence is that every fused-primitive addition across four
different milestones (M31, M53, M54, M55) independently converged on the
identical implementation shape without any of those milestones citing the
others' code as a template consciously (each cites the *reasoning*, not
copy-pastes structure). What creates the "pile of capabilities" *feeling*,
to the extent it exists, is entirely at the project-narrative layer: ten
consecutive milestone reports, each 15–28KB, each independently
re-deriving the same six-candidate survey, is a lot of repeated prose
around a small, stable set of conclusions — and a reader who only sees
that layer (not the code) would reasonably conclude the project is
thrashing. It is not; the reports are.

## 8. Architectural Strengths

- A single, ABC-enforced backend boundary (`Backend`) that every op
  crosses the same way, with CPU as the always-correct reference and CUDA
  checked against it on every claim.
- A `Module` attribute-registration model (`__setattr__`/`__getattr__`
  dispatch to `_parameters`/`_buffers`/`_modules`) that scales to a new
  stateful layer (buffers, M53) without changing any existing consumer.
- A training model with two named, bounded, independently-reaffirmed
  patterns (`Trainer` vs. hand-written loop) rather than one leaky
  abstraction straining to cover both.
- A measure-before-optimize performance culture that is now explicit
  policy, not folklore (Section 12).
- A persistence format that has needed exactly one shape change (buffers,
  M53) across 58 milestones and versions itself explicitly.

## 9. Architectural Weaknesses

- No mechanism exists (or is needed yet) for a model whose training loop
  is neither "one forward call per step" nor "the exact `char_rnn`/
  `word_rnn` unrolled-sequence shape" — a real but currently theoretical
  limitation, since no third shape has ever been attempted.
- The CUDA test suite's hard coupling to one specific GPU (57/103 files)
  means "the test suite passes" and "Forge works" are only fully
  equivalent claims on this one reference machine.
- `Tensor.batch_norm2d`/`Tensor.rnn_cell` being CUDA-only fused primitives
  with a CPU-composed equivalent is a correct, precedented design choice,
  but it does mean two independent code paths must be kept behaviorally
  identical by hand (mitigated today by parity tests, but a latent
  maintenance cost as more fused ops accumulate).

## 10. Important Gaps

1. **No flagship regression/tabular example.** Named explicitly in
   `vision.md`/`use-cases.md` (UC2), never built to the standard the
   other two workload families were. This is the one gap in this entire
   report that is justified by the *original product vision itself*,
   independent of any framework-pressure argument.
2. **Stale top-level onboarding surface** (`README.md`,
   `forge/__init__.py`'s docstring) — Section 7's concrete finding.
3. **The assessment-milestone pattern has exhausted its marginal value**
   as a *default* milestone shape — M56/M57/M58 each said this themselves;
   M59's own project-level view confirms it from a different angle (ten
   milestones of narrow-scope survey cannot see project-level gaps like
   #1/#2 by construction, since neither is a Tensor/`nn` capability gap).
4. **No versioning/release discipline** — real, but low urgency for a
   solo-developer, pre-release project; will matter the moment Forge is
   shared with anyone else.

## 11. Non-Gaps / Things That Should NOT Be Built

Re-confirmed, not re-litigated, from M49–M58's own extensive direct
evidence, none overturned by anything found this milestone: attention/
transformer (still three simultaneous primitive blockers, no concrete
consumer), LayerNorm (no consumer), general N-D CUDA broadcast/reduction
(every real consumer already got a correctly-scoped kernel), SGD momentum/
LR scheduler (no workload need), CLI `train`/`evaluate`/`predict`
(requires a config system Forge has correctly declined to build), a shared
RNN training-loop helper (two consumers, no validated abstraction
boundary), sparse `Embedding` gradients (no measured slowdown at any
tested scale), further RNN backward-kernel fusion (dominant cost already
removed by M55), a hosted CI workflow (cannot exercise this project's own
primary CUDA verification requirement; no missed-regression incident on
record across 58 milestones).

## 12. Performance Strategy

Codifying what M47–M58 already established as practice into explicit
policy for every future milestone:

- Conv2d, RNNCell, the CUDA allocator, and the stream/synchronization
  model are **good enough until a real, currently-running workload
  measurably proves otherwise** — not "until a synthetic benchmark could
  theoretically show more headroom."
- Profiling is triggered only by a **measured** wall-clock problem on a
  real example (M55's discovery method), never by a generic sweep "just to
  check" (M56/M57/M58 each explicitly rejected this as a default action).
- An optimization is meaningful only when it is hardware-measured on the
  reference 940MX, end-to-end, on a real workload — an isolated kernel
  -level speedup that does not move a real training run's wall-clock time
  (M55 Section 6's rejected "further backward fusion") is not sufficient
  justification on its own.
- Reject an optimization when the dominant cost has already been removed
  and the remaining opportunity is a shrinking secondary contributor
  (M55's own explicit example).

## 13. Candidate Strategic Directions

| Direction | Problem addressed | Evidence | Value | Complexity | Architectural consequence | Risk | Verdict |
|---|---|---|---|---|---|---|---|
| **A. Flagship regression example** (`examples/regression` or similar, UC2) | Named vision/use-case gap never closed | `vision.md`/`use-cases.md` direct text; `trainer_demo.py`'s own "demo, not example" status re-confirmed against `mnist`/`char_rnn`/`word_rnn`'s convention | High — closes the one gap this report found that the original vision itself, not framework pressure, justifies | Low — composes entirely from existing, already-proven primitives (`Linear`/`ReLU`/`MSELoss`/`Adam`/`Trainer`) | None — no new `Tensor`/`Backend`/`nn` surface | Low | **Pursue now (M60)** |
| **B. Onboarding documentation refresh** (`README.md`, `forge/__init__.py` docstring) | Stale front door undersells current capability | Section 7's direct diff between actual capability and documented capability | Medium — costs a reader real understanding today | Low | None | Low | **Pursue, bundled with M60 or as its own tiny follow-up** |
| **C. End the assessment-loop-as-default pattern** | Five of the last ten milestones produced large reports with diminishing marginal signal | M56/M57/M58's own explicit self-diagnosis, confirmed independently here from the project-narrative angle | High — this is process debt, and process debt compounds | None (a guardrail, not code) | None | None | **Adopt immediately (Section 17)** |
| D. Attention/transformer block | Speculative architectural breadth | Still 3 simultaneous primitive blockers, no concrete consumer (unchanged since M52) | Low/speculative | High | High | Medium-high | Reject, unchanged |
| E. CI/CPU-only regression gate | Would formalize regression coverage | No missed-regression incident across 58 milestones; cannot cover the CUDA half of the suite anyway | Low today | Medium | None | Low | Defer — revisit only if the single-developer/single-machine model changes |
| F. Versioning/`CHANGELOG` discipline | No release process exists | Real, but zero urgency pre-release | Low today | Low | None | None | Defer — adopt whenever `forge/` next changes for an unrelated reason |

## 14. Strategic Ranking

1. **A — Flagship regression example.** Highest-confidence, lowest-risk,
   directly vision-justified. Ranked first because it is the only
   candidate in this entire report (across all 58 milestones' surveys)
   whose justification comes from the *original product document* rather
   than a measured framework-pressure gap — a different, and in this
   project's own stated terms (Section 17 of the brief) equally valid,
   category of evidence.
2. **C — End the assessment-loop pattern.** Free, immediate, addresses a
   real and now well-evidenced (three consecutive self-diagnoses)
   process cost. Ranked second only because it produces no artifact by
   itself — it changes what M60+ is allowed to default to.
3. **B — Onboarding documentation refresh.** Real value, low cost, no
   architectural weight — but smaller in scope than A, and best done
   alongside A (a regression example is itself new content the README
   should mention) rather than as its own milestone.
4–7. **D, E, F** — correctly not pursued now, each for a documented,
   specific reason (Section 13), each with a named concrete trigger for
   reconsideration (Section 17).

## 15. Selected Next-Stage Direction

**A bounded implementation (the third vision-named workload family) plus
a permanent process guardrail**, not a pure investigation and not a pause.
This is Outcome mix (A + C from the brief's Section 12 options): a
concrete implementation direction is identified and scoped (Section 18),
and a concrete, permanent change to how future milestones are chosen is
adopted starting immediately (Section 17), without waiting for M60 to
formalize it.

M59 itself makes no `forge/` change: per Section 18 of this report's own
brief and the precedent M52 set for BatchNorm2d, a full example (dataset,
model, training script, README, CPU+CUDA integration tests) is real
engineering work deserving its own milestone budget, not a rushed
addition under a strategic-decision milestone's time pressure.

## 16. North-Star Goal

**At the end of the next stage, a developer unfamiliar with Forge's
internals should be able to pick any one of three genuinely distinct,
vision-named workload classes — image classification, sequence/language
modeling, or tabular regression — and, using only that workload's README
and Forge's public API, train it, evaluate it, checkpoint and resume it,
persist and reload it, and run it on CPU or CUDA, without reading a single
line of `forge/` source.**

Measurable directly: three example directories, each with a README
matching the `mnist`/`char_rnn`/`word_rnn` convention (prerequisites,
exact commands, expected numeric behavior, CUDA notes, determinism notes,
framework-additions-required section), each with CPU and CUDA integration
tests in `tests/`, each independently hardware-verified on the reference
940MX. Today this is true for two of three; the north star is three of
three, with nothing else added speculatively to reach it.

## 17. Milestone Guardrails

Mandatory, effective starting with M60:

1. **Consumer requirement.** A concrete workload, user need, reliability
   requirement, or architectural problem must exist — not a hypothetical
   one.
2. **Evidence requirement.** Demonstrated by direct execution/inspection,
   never assumed. A throwaway probe against the real backend beats
   reasoning from documentation alone whenever the two could disagree.
3. **Value requirement.** Expected benefit must exceed implementation
   complexity, stated explicitly, not left implicit.
4. **Scope requirement.** Implement the smallest coherent solution — one
   fused primitive, one example, one fix, never a speculative platform.
5. **Validation requirement.** Every production change gets correctness
   tests (finite-difference/parity where applicable) and a full-suite
   re-run, hardware-verified on the reference machine for any CUDA claim.
6. **Stop condition.** If evidence shows marginal, uncertain, or
   complexity-outweighed benefit, stop — an Outcome-D/assessment-only
   milestone remains a legitimate, complete outcome.
7. **Revisit condition.** A previously rejected candidate (Section 11) may
   be reconsidered only when new evidence — a new workload, a new
   measurement, a new external requirement — actually changes the
   calculus, not on a fixed schedule.
8. **No milestone-count objective.** Creating a milestone is never itself
   a justification for work inside it.
9. **No default assessment-loop milestones (new, from Section 15/C).** A
   milestone whose only mandate is "survey Forge fresh and decide" is not
   the default next step anymore. It is reasonable **only** when a named
   external trigger has actually fired: a new use case from the project
   owner, a real deployment/usage attempt, a workload that fails, or a
   material change to the hardware/environment. Absent such a trigger, the
   default next milestone is either (a) the next item in this report's
   ranked list (Section 14) or (b) whatever concrete need the project
   owner names directly.

Area-specific rules:
- **Performance optimization**: governed by Section 12 above in full.
- **New Tensor primitives**: only for a real, currently-attempted
  consumer; fused only when a real training loop's measured launch-count
  cost justifies it (the M31/M53/M54/M55 precedent), never speculatively.
- **New model layers**: only for a workload actually being built, never
  "for coverage."
- **Infrastructure/tooling** (CI, versioning, packaging): only when the
  single-developer/single-machine assumption actually changes, or when a
  concrete pain incident occurs — not preemptively.
- **Examples**: only for a vision-named or use-case-named workload class
  not yet demonstrated at production quality, or a measured framework
  -pressure discovery (the M50/M52/M54 pattern) — never a second
  demonstration of an already-proven capability.
- **Documentation**: architecture docs (`docs/architecture/*`) are living
  references and must be updated in the same change that alters the
  behavior they describe (already the practice — Section 7 confirms this
  holds). Top-level onboarding docs (`README.md`, `forge/__init__.py`)
  should be refreshed whenever a new example family lands, so the
  Section 7 staleness finding does not recur.

## 18. M60 Recommendation

**Objective**: build Forge's third vision-named workload family —
tabular regression — to the same production standard as `mnist`/
`char_rnn`/`word_rnn`.

**Concrete consumer/problem**: `docs/product/vision.md`'s explicitly named
"regression" workload class and `use-cases.md`'s UC2 have never been
demonstrated at the standard the project itself already established for
its other two families (README with prerequisites/expected numbers/CUDA
notes/determinism notes, CPU+CUDA integration tests, checkpoint/resume,
save/load prediction parity) — only a bare, un-READMEd milestone
-verification script (`trainer_demo.py`) exists.

**Why it matters**: this is the one concrete gap in the entire 59
-milestone history whose justification is the original product vision
itself, not a measured framework-capability pressure — a different and
independently sufficient category of evidence per this report's own
Section 17 guardrails.

**Expected value**: closes UC2 concretely; gives the project three, not
two, genuinely distinct demonstrated workload families, directly serving
the North-Star Goal (Section 16); requires zero new `Tensor`/`Backend`/
`nn` surface, since `Linear`/`ReLU`/`MSELoss`/`Adam`/`Trainer`/`DataLoader`
already fully cover it — this is assembly and validation work, not
primitive-design work, and should be scoped and estimated as such.

**Bounded scope**: one new `examples/` directory (a synthetic-but
-realistic tabular dataset generated in-process, matching `char_rnn`/
`word_rnn`'s own no-download convention; a small `Linear`/`ReLU` MLP;
`MSELoss`; `Adam`; `Trainer.fit()` — this workload fits `Trainer`'s
one-forward-per-step shape exactly, unlike the RNN examples, so no
hand-written loop is needed); a README matching the established
three-part convention (usage, expected numbers, CUDA notes); CPU and CUDA
integration tests mirroring `tests/test_mnist_example_*integration.py`'s
structure; a `.gitignore` entry for its generated-artifact directory from
the start (closing M57's own retroactively-discovered gap pattern
pre-emptively this time).

**Success criteria**: trains and reduces loss/a regression metric (e.g.
mean absolute error) well below a trivial baseline; CPU/CUDA first-epoch
parity within the project's established tolerance; checkpoint save/resume
equivalence; save/load prediction-parity; README accepted against the
`mnist`/`char_rnn`/`word_rnn` convention; full suite green with new tests
added, zero regressions.

**Explicit exclusions**: no new `Tensor` primitive, no new `nn.Module`, no
feature-engineering/preprocessing pipeline beyond `forge.data.transforms`'
existing `Normalize`, no hyperparameter tuning beyond "converges
reliably," no `Trainer` API change.

**Validation strategy**: identical methodology to `mnist`'s own (Section 4
table) — synthetic fast dataset for the mandatory test suite, a
real/larger run for the README's reported numbers, hardware-verified on
the reference 940MX for every CUDA claim, never estimated.

**If a small, low-risk documentation refresh is bundled in** (Section 13/B
— `README.md`/`forge/__init__.py` docstring updated to mention the new
example and the post-M26 capability additions), that is acceptable per
this report's own Section 17 documentation guardrail, but is not required
for M60's success criteria above.

## 19. Success Criteria (for this stage overall)

The next stage is complete when: three example families (not two) each
independently satisfy the North-Star Goal's per-example bar (Section 16);
the top-level onboarding surface accurately reflects current capability;
and at least one milestone cycle has passed using Section 17's guardrails
without defaulting back to an assessment-only survey absent a named
trigger.

## 20. Risks and Limitations

- This report's own gap-finding method (direct source/doc reading plus
  targeted execution) inherits the same limitation every prior milestone's
  own "Limitations" section names: it is a point-in-time assessment.
  Nothing here claims permanent validity — Section 17's revisit condition
  exists precisely so this does not calcify into unquestioned doctrine the
  way the six-candidate capability survey nearly did.
- The regression-example recommendation (Section 18) is evidenced by the
  original vision document, not by a measured technical failure the way
  M52/M54/M55's implementations were — this is an intentionally different,
  and weaker in the narrow "proven by direct probe" sense, category of
  justification. It is still valid: `vision.md` and `use-cases.md` are
  Forge's own stated success criteria, and satisfying them is not
  optional just because no workload has yet failed without them.
- Section 7's documentation-staleness finding and Section 10's gap list
  are both, by nature, judgment calls about what "good enough onboarding"
  means for a still-solo-developer, pre-release project — reasonable
  engineers could set the bar differently. This report documents the
  specific evidence (Section 7) so a future milestone can re-evaluate the
  bar explicitly rather than accept or reject it silently.
- Ending the assessment-loop-as-default pattern (Section 17, guardrail 9)
  carries a real risk: it means Forge will spend less time systematically
  re-checking whether prior rejections still hold. This is accepted
  deliberately — the marginal detection value of five consecutive
  assessment-only milestones each re-confirming the same six rejections
  had already fallen to effectively zero (Section 1), and the guardrails'
  revisit condition (Section 17.7) still allows any of them to be reopened
  the moment real evidence appears.

## 21. Final Decision

**M59 concludes with a project-level direction, not another "no candidate"
assessment**: pursue the third vision-named workload family (tabular
regression) as M60's concrete implementation objective, and adopt this
report's Section 17 guardrails — most importantly, that a fresh, unscoped
capability-assessment survey is no longer the default shape of the next
milestone — starting immediately.

No `forge/`, `examples/`, or `tests/` file was modified this milestone.
This document and the accompanying `docs/development/progress.md` entry
are the only changes.

## Verification

No production code was changed (`git status` confirms only this
document and the `progress.md` append, consistent with a strategic
-decision milestone). The full existing suite was re-run fresh at the
start of this milestone to establish repository integrity before drawing
any conclusion from it: **1,771 passed, 0 failed** — identical to M56's,
M57's, and M58's own reported counts, confirming zero drift across the
entire assessment-loop period this report evaluates. Every capability
claim in Sections 4/7 was checked against the current source directly
this milestone (`forge/tensor/tensor.py`, `forge/nn/module.py`,
`forge/backend/base.py`, `forge/exceptions.py`, `forge/training/
trainer.py`, `forge/optim/optimizer.py`, `forge/cli/main.py`, all three
example READMEs, `docs/architecture/tensor-api.md`, `pyproject.toml`,
`.gitignore`, and a direct `git ls-files`/test-collection count), not
carried forward from any prior milestone report's own snapshot.

## Suggested Commit Message

```
docs: M59 vision reassessment — select tabular regression (UC2) as the
next workload; end assessment-loop-as-default milestone pattern
```
