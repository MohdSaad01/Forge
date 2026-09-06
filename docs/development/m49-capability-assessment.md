# M49 — Framework Capability & Value Assessment (measurement/assessment-only)

## 1. Executive summary

M48 closed the current Conv2d optimization thread: no Conv2d
forward/backward sub-component has a comparable fresh-evidence-backed
opportunity left (`docs/performance/m47-bottleneck-recharacterization.md`,
`docs/performance/m48-dinput-value-assessment.md`). M49's brief explicitly
forbids continuing CUDA optimization by default and instead requires an
evidence-based survey of five candidate areas (optimizer coverage, NN
layer/operator coverage, data-loading ergonomics, serialization/
checkpointing, example/model coverage) to determine the highest-value next
investment.

**Conclusion: no implementation is justified this milestone.** Every
candidate area was inspected against Forge's own documented requirements
(`docs/product/requirements.md`) and use cases (`docs/product/use-cases.md`),
and every one of them already satisfies what it is asked to satisfy. Every
concrete gap found is a previously-considered, deliberately deferred
capability with no current real-workload impact, not an oversight. This is
a valid, evidence-backed M49 outcome per the brief's own decision rule.

## 2. Method

For each area: inspect the implementation, inspect its tests, inspect what
it is asked to support (`requirements.md`/`use-cases.md`/`vision.md`/
`scope.md`), and — where reasoning from source alone was insufficient —
run real code against the installed package to get first-hand evidence
rather than inferring from docstrings. The full existing test suite was run
once at the end to confirm the assessment did not alter repository
behavior (no production files were touched).

## 3. Area 1 — Optimizer coverage

**Current state.** `forge.optim`: `Optimizer` (base), `SGD` (plain, no
momentum/weight decay/schedule — `forge/optim/sgd.py`), `Adam` (full
Kingma & Ba, with L2-style `weight_decay`, per-parameter state, explicit
device-migration guard rails — `forge/optim/adam.py`). No learning-rate
scheduler exists anywhere in the tree (confirmed by grep across `forge/`
and `docs/`).

**Deficiency.** SGD has no momentum; there is no LR scheduler.

**Real workload relevance.** Forge's one trained, checkpointed, benchmarked
real workload (`examples/mnist/train.py`) uses **Adam**, not SGD, with a
fixed learning rate, and already converges and trains successfully across
M20-M48. No real Forge workload currently uses SGD, and none currently
needs a schedule. `docs/architecture/optimization.md` and `forge/optim/
sgd.py`'s own docstring already document "no momentum, weight decay, or
learning-rate schedule" as an explicit, considered scope boundary, not an
accidental gap. `requirements.md` asks only for "at least one practical
optimizer" — satisfied twice over (SGD + Adam).

**Verdict: real gap, but not currently load-bearing.** Momentum/scheduling
are well-understood, low-risk additions *when a real workload needs them*.
None does yet. Implementing them now would be feature-count-driven, exactly
what the brief prohibits. **Rejected for M49**; revisit if/when a workload
that plausibly needs SGD+momentum or a decaying schedule is added.

## 4. Area 2 — NN layer / operator coverage

**Current state.** `forge.nn`: `Linear`, `Conv2d`, `MaxPool2d`, `ReLU`,
`Dropout`, `Flatten`, `Sequential`, `MSELoss`, `CrossEntropyLoss`. Tensor
primitives (`forge/tensor/tensor.py`): `+ - * @`, `.sum()`, `.reshape()`,
`.relu()`, `.exp()`, `.log()`, `.conv2d()`, `.max_pool2d()`,
`.dropout_mask()`, `.cross_entropy()`, `.to()`, `.backward()`. Verified by
direct execution (not just reading) that `-x`, `x / y`, `x ** 2`,
`.sqrt()`, `.mean()`, `.sigmoid()`, `.tanh()`, `.transpose()`, `.softmax()`
all fail today — none of these exist at the Tensor level.

**Deficiency, and why it looks intentional rather than accidental.**
`forge/backend/cuda/kernels.cu`'s own M14 comment on `log_backward`
explicitly rejected adding "a generic elementwise-divide primitive that
nothing else in Forge needs" in favor of a dedicated `1/x` kernel scoped to
`log`'s own gradient. `forge/data/transforms.py`'s `Normalize` docstring
documents working around the same absence in the public data API:
`(x - mean) * (1/std)`, "since Forge's Tensor has no division operator."
Every operation that exists was added for one specific consumer at the
milestone that needed it (`docs/architecture/tensor-api.md`'s "Not yet
implemented" section traces this exactly), and the op set today is *exactly*
what `Linear`/`Conv2d`/`MaxPool2d`/`ReLU`/`Dropout`/`CrossEntropyLoss`/
`MSELoss`/`Adam`/`SGD` require — nothing more, nothing less.

**Real workload relevance.** The `Normalize` workaround is functional, not
broken: `std` is known at transform-construction time, so `1/std` is a
precomputed constant, not a runtime tensor division. No current real Forge
workload (MNIST classification, or the regression/classification demos in
`examples/trainer_demo.py`) is blocked by the missing ops. They would matter
for a *new* class of model this milestone was not asked to build (binary
classification needing `sigmoid`, normalization layers needing `sqrt`/`/`,
attention needing `transpose`/`softmax`, RNNs needing `sigmoid`/`tanh`).

**Verdict: consistent with a deliberate, already-precedented design
decision (M14).** Expanding the operator set now, with no concrete new
model driving the requirement, would be exactly the "speculative framework
machinery" and "operator count for its own sake" the brief and
`CLAUDE.md` both warn against. **Rejected for M49.** If a specific new
model family is ever chosen (e.g. binary classification, a small
normalization layer), let *that* choice drive exactly the ops it needs, the
same way every existing op was added.

## 5. Area 3 — Data-loading ergonomics

**Current state.** `forge.data`: `Dataset`, `TensorDataset`, `Subset`,
`random_split`, `DataLoader` (batching, shuffling with a reproducible
generator, `drop_last`), `Compose`/`ToTensor`/`Normalize`/`Reshape`/
`Flatten`/`Lambda` transforms, `CUDAPrefetchLoader` (M30, real asynchronous
device-transfer overlap). All five of `requirements.md`'s Data bullets
(dataset/dataloader abstractions, raw array support via `TensorDataset`,
extensible custom datasets via subclassing `Dataset`, transforms/batching,
image+tabular support without format coupling) are already met, and are
exercised end-to-end by `examples/mnist/dataset.py` (custom `Dataset`,
image) and `examples/trainer_demo.py`/`data_pipeline_demo.py` (tabular,
`TensorDataset` + `Normalize`).

**Deficiency.** None found. No multiprocessing workers — explicitly
documented as out of scope (`forge/data/dataloader.py`'s module docstring),
consistent with `scope.md`'s "no distributed/enterprise-scale
orchestration" and a single-machine 940MX/i5-7200U/8GB target where worker
processes would not obviously help.

**Verdict: no gap justifying work. Rejected for M49.**

## 6. Area 4 — Serialization / checkpointing

**Current state.** `forge.serialization`: `save_model`/`load_model`
(registry-based reconstruction, `PersistenceError` on an unregistered
type, device-aware, CPU-portable archive format) and `save_checkpoint`/
`load_checkpoint` (M18: model state + optimizer type/hyperparameters/
per-parameter state keyed by dotted name across the serialization
boundary + epoch/global_step + Forge's default RNG state + caller `extra`).
`examples/mnist/train.py` exercises the full loop in one real script:
train → checkpoint → resume → model save → cold reload → prediction-parity
assertion. A CLI (`python -m forge model inspect` / `checkpoint inspect`)
already exists for both formats.

**Deficiency.** None found relative to what `requirements.md`'s
Persistence bullets and UC6 ask for ("save enough model state to reliably
reconstruct a trained model," "load without the original training
process"): both are already demonstrated, including the harder case
(resume training, not just inference).

**Verdict: no gap justifying work. Rejected for M49.**

## 7. Area 5 — Real example / model coverage

**Current state.** `examples/mnist/` is the one polished, README-documented,
CLI-integrated, CPU+CUDA-verified, checkpoint/resume-capable end-to-end
example. But it is **not** the only demonstration of a non-CNN workload:
`examples/trainer_demo.py` already runs both a regression task (`Linear` →
`MSELoss` → `SGD`, tabular synthetic data, `MeanAbsoluteError` metric) and
a classification task end-to-end through the same `Trainer`/`DataLoader`
stack; `examples/data_pipeline_demo.py` demonstrates `TensorDataset` +
`Normalize` + `DataLoader` on tabular regression data; `examples/
persistence_demo.py` demonstrates save/cold-reload for a `Linear`
regression model across separate processes.

**Mapping to `docs/product/use-cases.md`:** UC1 (classifier) → MNIST. UC2
(regression) → `trainer_demo.py`/`data_pipeline_demo.py`. UC3 (image
workload) → MNIST. UC4 (custom dataset) → `examples/mnist/dataset.py`
subclassing `Dataset`. UC5 (CPU/CUDA) → MNIST `--device`. UC6
(persistence) → `persistence_demo.py`/MNIST checkpointing. UC7 (benchmark)
→ `benchmarks/` (extensive). **All seven use cases already have a working,
runnable demonstration in the repository.**

**Deficiency.** The regression/classification demos are smaller "manual
verification" scripts (their own docstrings label them as milestone
verification, not showcase examples) rather than MNIST-tier polished
examples with a README and CLI integration. That is a presentation gap, not
a capability gap — and building a second MNIST-tier example today would, by
the current operator/optimizer survey above (Sections 3-4), only be able to
recombine primitives Forge already has and already demonstrates working
(Linear/ReLU/MSE/SGD/Adam/DataLoader/Trainer/persistence) — i.e. cosmetic
repository growth, which the brief explicitly says not to do.

**Verdict: rejected for M49.** A second showcase example only becomes
justified once a new model family (motivating new ops, per Section 4) is
actually chosen — at which point the example and the primitives it needs
should be scoped together, not the example first.

## 8. Conv2d

Per the M49 brief's explicit constraint, Conv2d was not re-examined as an
optimization target. M47/M48 already closed that thread (below-256 dWeight
formally deprioritized after four independent attempts across M21/M33/
M45/M46; dInput low-`Cin` accepted in M48). Nothing in this milestone's
survey surfaced new evidence that would reopen it — all five areas
investigated here are orthogonal to Conv2d kernel performance. **The
Conv2d optimization thread remains closed.**

## 9. Candidate ranking

| Area | Current deficiency | Real workload relevance | Impact if fixed | Scope | Risk | Verdict |
|---|---|---|---|---|---|---|
| Operator coverage | Real (no `/ - (unary) ** sqrt sigmoid tanh transpose softmax mean`) | None for any *existing* real workload | High, but only for models not yet chosen | Small per-op, unbounded in aggregate | Low per-op | Rejected — no driving model |
| Optimizer coverage | Real (no SGD momentum, no LR scheduler) | None (real workload uses fixed-LR Adam) | Medium, only if a workload needed it | Small (mirrors existing Adam pattern) | Low | Rejected — not load-bearing |
| Example/model coverage | Presentation-only (demos exist, aren't showcase-polished) | N/A — use cases already covered | Low (would recombine existing primitives) | Small-medium | Low | Rejected — cosmetic |
| Data-loading ergonomics | None found | Fully covered | N/A | N/A | N/A | Rejected — no gap |
| Serialization/checkpointing | None found | Fully covered | N/A | N/A | N/A | Rejected — no gap |

## 10. Rejected directions and why (summary)

- **SGD momentum / LR scheduler**: real but currently unused capability;
  adding it now has no workload to validate it against beyond synthetic
  unit tests, and the project's own `sgd.py` docstring already scopes this
  out deliberately. Revisit when a workload needs it.
- **Elementwise division / negation / sqrt / sigmoid / tanh / transpose /
  softmax / mean**: real, empirically confirmed absences, but each one
  Forge has hit before (`log_backward`'s M14 divide rejection, `Normalize`'s
  documented workaround) and deliberately declined to generalize because
  nothing consumed it. Still true today. Revisit alongside a concrete new
  model family that needs a specific one of these, exactly as every
  existing op was added for a specific consumer.
- **A second MNIST-tier example**: use cases UC1-UC7 are all already
  demonstrated; a second polished example today would only repackage
  primitives already proven to work, at the cost of new repository surface
  with no new capability behind it.
- **Data-loading / serialization enhancements**: no deficiency found against
  `requirements.md`, `use-cases.md`, or actual example usage.

## 11. Selected M49 direction

**None. M49 is assessment-only**, per the brief's explicit decision rule:
"The milestone must be willing to conclude 'No implementation is currently
justified.'" No production code was changed.

## 12. Validation

No production code changed (`git status` clean before and after this
milestone's work). Full existing suite re-run to confirm the assessment did
not alter repository behavior: **1,579 passed** (unchanged from M48's own
count), run on this machine's real CUDA backend (940MX) — CUDA tests are
not skipped here.

## 13. Limitations

- This is a point-in-time assessment. "No real workload needs X" is a
  statement about Forge's *current* two demonstrated workload families
  (image classification, tabular regression/classification); it is not a
  claim that X is never worth building.
- The ranking in Section 9 is qualitative, consistent with the other four
  areas requiring no benchmarking to assess (their gaps are presence/
  absence, not performance) — no benchmarks were written this milestone.

## 14. M50 recommendation

Do not pick a candidate from this survey speculatively. Instead, the
highest-leverage next decision is a **product** one, not an engineering
one: choose a concrete *second* model family Forge should demonstrably
support end-to-end (e.g. a small binary-classification or normalization-
using architecture), and let that choice determine — narrowly — exactly
which Tensor ops (from Section 4's list) and, if applicable, which
optimizer feature it actually needs. That keeps every future addition tied
to a real consumer, matching how every op in the codebase today got added.
Absent such a product decision, re-running this same five-area survey again
next milestone would not find new evidence — nothing here is time-sensitive
the way CUDA occupancy/thermal measurements were in M40-M48.
