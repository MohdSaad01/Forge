# M52 — Product Direction & Third-Workload Assessment (assessment-only)

## 1. Executive summary

M49 surveyed five candidate areas (optimizers, NN ops, data loading,
serialization, examples) and found every one already satisfied what it was
asked to satisfy, with no real workload driving expansion. M50 then made the
product decision M49 recommended: it picked a genuinely new model family
(char-RNN), attempted it against the framework as-is, and discovered exactly
one real blocker (`tanh`). M51 fixed an unrelated, real allocator
correctness defect.

M52's brief is to make the next concrete product decision the same
evidence-first way: survey Forge as it stands today, propose several
plausible next workloads/directions, attempt the strongest one against
Forge's actual API (not an estimate), and either build a genuine discovered
gap, declare nothing is justified, or recommend hardening.

**Conclusion: a genuine, reusable gap exists, and it is not a new op or a
new example — it is a missing architectural concept.** Forge's `Module` has
no notion of *buffers*: non-parameter, non-differentiable, persistent,
device-moved, train/eval-mode-sensitive state. This has been independently
flagged as an explicit, deliberate omission three separate times
(`docs/architecture/modules.md:139-144` and `:222` at M16; M50's own model
-selection section, `docs/development/m50-char-rnn.md` Section 2, naming
BatchNorm2d specifically as the rejected alternative). It has never been
built because no workload has needed it. Attempting a normalization-based
CNN (BatchNorm2d added to the existing MNIST family) against Forge exactly
as M51 left it (Section 8, a direct probe, not an estimate) confirms the gap
is real, small, and precisely bounded: two missing general elementwise ops
(`sqrt`, division) plus the buffer concept itself plus one CUDA reduction
kernel scoped narrowly to BatchNorm's own math — not a general N-D
reduction/broadcast engine.

**Outcome A: build the capability**, minimally scoped, in a future milestone
(recommended as M53 in Section 14). M52 itself makes no production change
(Section 8 of the brief explicitly permits this) — the gap is already
proven by direct inspection and a small throwaway probe script, and writing
`BatchNorm2d` production code and its CUDA kernel is real engineering work
that deserves its own milestone rather than being folded into a decision
milestone under time/scope pressure.

## 2. Current Forge capability snapshot

Surveyed by direct inspection of the current tree (`forge/`), not repeated
from M49's inventory:

- **Tensor** (`forge/tensor/tensor.py`): `+ - *` (broadcasting), `@`
  (matmul, 1D/2D), `.sum(axis, keepdims)`, `.reshape()`, `.relu()`, `.exp()`,
  `.log()`, `.tanh()` (M50), `.conv2d()`, `.max_pool2d()`, `.dropout_mask()`,
  `.cross_entropy()`, `.to(device)`, `.backward()`. Confirmed absent by
  direct execution (Section 8): `.sqrt()`, `__truediv__`/`__rtruediv__`,
  `__neg__`, `__pow__`, `.mean()`, `.transpose()`/`.permute()`, `.softmax()`,
  `.sigmoid()`. This exactly matches M49's finding — still true.
- **`nn`**: `Linear`, `Conv2d`, `MaxPool2d`, `ReLU`, `Tanh`, `Dropout`,
  `Flatten`, `Sequential`, `RNNCell` (M50), `MSELoss`, `CrossEntropyLoss`.
- **`Module`** (`forge/nn/module.py`): `_parameters`/`_modules` dicts,
  `named_parameters`/`named_modules`/`parameters`/`modules`/
  `named_children`/`children`, `train()`/`eval()`/`.training`, `.to(device)`
  (moves `Parameter`s only), `.device`. **No `_buffers` dict, no buffer
  registration API, no buffer traversal.** `Dropout` is the only module
  whose forward reads `.training`, and it needs no persistent state to do
  so (Section 4).
- **Optimizers**: `SGD` (no momentum/weight decay/schedule — deliberate,
  documented in `sgd.py`'s own docstring), `Adam` (full, with L2-style
  `weight_decay`). No LR scheduler.
- **Data**: `Dataset`, `TensorDataset`, `Subset`, `random_split`,
  `DataLoader`, `Compose`/`ToTensor`/`Normalize`/`Reshape`/`Flatten`/
  `Lambda`, `CUDAPrefetchLoader`. Unchanged since M49; no gap found there
  again.
- **Training**: `Trainer` (`fit`/`evaluate`, metrics, checkpointing,
  prefetch). `evaluate()` already switches to eval mode and runs inside
  `forge.no_grad()` — the inference path required by `requirements.md`'s
  "evaluation and inference paths" already exists and is exercised by every
  example (`examples/*/train.py`'s own save→reload→predict-parity
  assertions).
- **Serialization**: `save_model`/`load_model` (parameter values +
  per-module `config`/`.training`, registry-based reconstruction),
  `save_checkpoint`/`load_checkpoint` (+ optimizer state, epoch,
  global_step, RNG state). **Persists parameters only — there is no
  mechanism to persist a buffer**, because none exist yet
  (`forge/serialization/model.py:186-211`, `_build_save_node` walks
  `module._parameters` only).
- **CUDA**: real kernels for every op above except `sqrt`/division/etc
  (which don't exist anywhere yet). `sum()` on CUDA supports only
  `axis=None` (full reduction) or `axis=1`/`-1` on a strictly 2D tensor
  (`forge/backend/cuda/backend.py:958-987`, scoped exactly to
  `CrossEntropyLoss`'s M14 need — no general N-D axis reduction exists).
  Elementwise broadcasting on CUDA is scoped to exactly the two shapes
  `Linear`'s forward/backward need (`(rows,cols)+(cols,)` row-broadcast,
  `(rows,cols)+(rows,1)` column-broadcast — `docs/architecture/
  cuda-backend.md:163-194`); **no general N-D broadcast exists on CUDA**.
- **CLI**: `forge model/checkpoint inspect|convert`, `forge benchmark`.
  `requirements.md:47` still says "Planned commands include train,
  evaluate, predict, and benchmark" — only `benchmark` was ever built,
  `docs/development/cli.md`'s own **Limitations** section explains why
  (no config/YAML system, no generic model-construction-from-CLI-args
  story) — a deliberate M19 scope narrowing, not an oversight (Section 11).

## 3. Existing demonstrated workloads

Two, per M50: MNIST CNN (`Conv2d`/`MaxPool2d`/`ReLU`/`Flatten`/`Linear`/
`CrossEntropyLoss`/`Adam`/`DataLoader`/CUDA/checkpointing) and the char-RNN
(`RNNCell`/`tanh`/`Linear`/`CrossEntropyLoss`/`Adam`/`DataLoader`/CUDA/
weight-sharing-across-time/persistence). Plus the smaller, milestone
-verification-only `trainer_demo.py`/`data_pipeline_demo.py`/
`persistence_demo.py` (regression + tabular classification, already
covering UC1/UC2 per M49 Section 7).

## 4. Candidate next workloads/directions

Five were considered, evaluated in Section 6:

- **(A) Normalization-based CNN** — add `BatchNorm2d` to the existing
  MNIST-family CNN.
- **(B) Small self-attention / transformer block** — a minimal
  scaled-dot-product-attention layer over short sequences.
- **(C) Autoencoder** — `Linear`/`Conv2d` encoder-decoder trained with
  `MSELoss` on MNIST images.
- **(D) Extend the char-RNN family** — multi-layer/stacked `RNNCell`, or
  teacher-forced seq2seq.
- **(E) Framework ergonomics** — generic CLI `train`/`evaluate`/`predict`
  commands (closing the `requirements.md:47` gap), and/or SGD momentum + an
  LR scheduler (M49's still-open, still-unused finding).

## 5. Evaluation criteria

Per the brief's table: architectural novelty, practical value, framework
pressure (does it expose a genuine gap), implementation scope, validation
value, complexity risk, reusability.

## 6. Candidate comparison

| Candidate | Novelty | Practical value | Framework pressure | Scope | Validation | Complexity risk | Reusability |
|---|---|---|---|---|---|---|---|
| (A) BatchNorm2d CNN | Medium — extends an existing family, but the *mechanism* (buffers, mode-dependent stats) is new | High — near-ubiquitous in real CNN training, explicitly named as deferred three times already | **Real, proven by direct probe (Section 8)**: buffer concept + 2 elementwise ops + 1 scoped CUDA reduction kernel | Small-medium, tightly bounded (Section 9-10) | High — loss/parity/train-vs-eval-mode behavior all independently checkable | Low-medium — CUDA reduction kernel is the one real new-engineering item, but scoped narrowly like every prior CUDA addition (M14 `axis=1`, M50 `tanh`) | **High** — buffers serve any future stateful module (LayerNorm, InstanceNorm, running-stat trackers) |
| (B) Attention/transformer | High — architecturally distinct from CNN/RNN | Medium — not yet a concrete Forge consumer need | Real, but a *large* simultaneous blocker set: softmax, transpose/permute, division, sqrt, batched 3D matmul, masking, likely embedding | Large — many new primitives at once, each needing CPU+CUDA | Medium | **High** — exactly the "large feature expansion without a narrow driving need" the brief's Section 5 warns against | Speculative — no second consumer yet |
| (C) Autoencoder | Low — `Linear`+`ReLU`+`MSELoss`+`Adam` all already proven (M49 Section 7) | Low-medium | **None** — composes entirely from existing primitives; a `Sigmoid` output activation is a nice-to-have, not a blocker | Small | Low — mostly repackages existing capability, no property under test is new | Low | Low — a second demo of already-covered capability, the exact "cosmetic repository growth" M49 rejected |
| (D) Extend char-RNN | Low-medium — incremental to M50's own family | Low | None found without a driving longer-sequence/seq2seq consumer; M50 Section 6 already rejected this for the same reason | Small-medium | Medium | Low | Low-medium |
| (E) CLI train/predict + SGD momentum/scheduler | Low | Low — no workload needs SGD momentum/scheduling yet (still true, M49 Section 3); a generic CLI trainer conflicts with Forge's explicit "no config/YAML system" non-goal (`docs/development/cli.md`'s own Limitations) | None newly found | Medium (CLI) / small (optimizer) | Low | Low | Low |

## 7. Leading candidate

**(A) Normalization-based CNN — `BatchNorm2d`.** It is the only candidate
with *proven* (not estimated) framework pressure, the only one whose gap has
been independently flagged three separate times across three milestones
without ever being acted on, and the only one whose payoff (a reusable
buffer/non-parameter-state concept) serves future modules beyond the one
workload that motivates it — matching the brief's own reusability and
"do not choose based on novelty alone" criteria better than any alternative.
(B) is rejected specifically for pulling in a large, simultaneous primitive
set with no single narrow driving need (Section 5's explicit disqualifier).
(C)/(D)/(E) are rejected as recombination/no-new-pressure, mirroring M49's
own "cosmetic repository growth" and "not currently load-bearing" verdicts.

## 8. Concrete workload definition

Minimal architecture: the existing MNIST CNN
(`examples/mnist/model.py`) with one `BatchNorm2d(8)` inserted after the
first `Conv2d` (before `ReLU`) — the smallest change that exercises
BatchNorm's full train/eval-mode-dependent behavior end-to-end through the
existing `Trainer`/checkpoint/persistence stack, without inventing a new
example family.

```text
(N, 1, 28, 28)
    -> Conv2d(1, 8, k=3)   -> (N, 8, 26, 26)
    -> BatchNorm2d(8)      -> (N, 8, 26, 26)   [new]
    -> ReLU
    -> MaxPool2d(2) -> Conv2d(8, 16, k=3) -> ReLU -> MaxPool2d(2)
    -> Flatten -> Linear(400, 64) -> ReLU -> Linear(64, 10)
```

Forward (training mode): per-channel batch mean/variance over `(N, H, W)`,
normalize, then a learned per-channel scale/shift (`Parameter`s, like
`Conv2d`'s bias), plus an exponential-moving-average update of
`running_mean`/`running_var` (buffers). Eval mode: normalize using the
stored running statistics instead of the current batch's.

## 9. Existing-capability mapping (what needs no new work)

Confirmed by direct probe against the current CPU backend
(`batchnorm_probe.py`, throwaway, run in the scratchpad — not committed;
Section 10 lists exactly what it found missing):

- Per-channel reduction `x.sum(axis=(0, 2, 3), keepdims=True)` **already
  works on CPU** (`CPUBackend.sum` passes `axis` straight to `np.sum`, no
  restriction to a single axis).
- Broadcasting a `(1, C, 1, 1)` tensor against `(N, C, H, W)` in `__sub__`
  **already works on CPU** (`Tensor._binary_op` uses `np.broadcast_shapes`
  generally, not scoped to any particular rank/shape).
- `Parameter`, `Conv2d`'s own `weight`/`bias` pattern, `register_module()`,
  `Module.train()`/`.eval()`/`.training` (unused by anything but `Dropout`
  today), `Trainer.evaluate()`'s existing eval-mode switch, and
  `CrossEntropyLoss`/`Adam`/`DataLoader` all compose unchanged — no
  training-loop, loss, optimizer, or data-pipeline change is needed.

This mirrors M50's own finding almost exactly: composing the *forward math*
from existing Forge primitives mostly already works on CPU.

## 10. Genuine gaps discovered

1. **No buffer concept in `Module`.** (`forge/nn/module.py`,
   `docs/architecture/modules.md:139-144`/`:222`.) `running_mean`/
   `running_var` are non-differentiable, must move with `.to(device)`, and
   must persist across save/load — none of which `_parameters`/`_modules`
   can express today. This is the real, load-bearing new capability; every
   other gap below is small in comparison.
2. **`Tensor.sqrt()` does not exist.** Confirmed by direct execution
   (`AttributeError`). Needed for `std = sqrt(var + eps)`.
3. **`Tensor.__truediv__` does not exist.** Confirmed by direct execution
   (`TypeError`). Needed for `(x - mean) / std`; this is the exact division
   gap `Normalize`'s docstring already worked around in M49's own survey.
4. **CUDA has no per-channel `(N,C,H,W)`-reduction or `(1,C,1,1)`-broadcast
   path.** `CUDABackend.sum()` supports only full reduction or `axis=1` on a
   strictly-2D tensor (`backend.py:958-987`); CUDA elementwise broadcasting
   is scoped only to `Linear`'s two shapes
   (`docs/architecture/cuda-backend.md:163-194`). CPU already handles both
   generally (Section 9) — this is a CUDA-only gap, not a CPU one.
5. **No persistence path for buffers.** `_build_save_node`/`_build_load_node`
   (`forge/serialization/model.py`) walk `module._parameters` only; a buffer
   dict would need its own (small, analogous) save/load branch.

## 11. Rejected gaps / workarounds

- **General `Tensor.transpose()`/N-D CUDA reduction/N-D CUDA broadcast as
  standalone general primitives.** Rejected in favor of one dedicated CUDA
  kernel scoped exactly to BatchNorm's own per-channel mean/var/normalize
  math — the same precedent as `axis=1` reduction (M14, scoped to
  `CrossEntropyLoss`) and the M14 `log_backward` divide rejection ("a
  generic elementwise-divide primitive that nothing else in Forge needs").
  A fused `Tensor`-level op (mirroring `conv2d`/`max_pool2d`/
  `cross_entropy`/`dropout_mask`, all of which are single fused ops rather
  than compositions of generic primitives) is the Forge-idiomatic shape for
  the CUDA path specifically, even though `sqrt`/division themselves should
  be added as ordinary general elementwise ops (Section 12) since CPU
  already needs no such fusion and general division/`sqrt` are broadly
  reusable on their own (the exact "let a concrete model determine the op"
  precedent M49 asked for).
- **CLI `predict`/`train`/`evaluate` commands** (`requirements.md:47`):
  re-examined this milestone, not newly discovered. Rejected again — a
  generic model-construction-from-CLI-args story requires exactly the
  config/YAML system `docs/development/cli.md`'s own Limitations section
  already declares out of scope; nothing about the BatchNorm workload
  depends on it.
- **SGD momentum / LR scheduler**: re-examined, still no workload uses or
  needs either (BatchNorm training uses Adam, same as MNIST/char-RNN
  today). Rejected again, unchanged from M49.
- **Autoencoder / attention / extended char-RNN** (candidates C/B/D):
  rejected per Section 6/7 — no proven framework pressure, or too large a
  simultaneous blocker set for one milestone.

## 12. Product decision

**Outcome A — build the capability, but not in M52.** The gap (buffer
concept + two general elementwise ops + one scoped CUDA reduction kernel)
is real, proven by direct inspection and probe (Sections 9-10), narrowly
bounded, and reusable beyond this one workload. Per the brief's Section 8
("M52 is not required to modify production code... prototype only what is
necessary"), M52 itself makes **no production change** — the throwaway
probe script already proved what needed proving without writing any
`forge/` code, and `BatchNorm2d` plus its CUDA kernel is real engineering
work (the CUDA reduction kernel specifically) that deserves a dedicated
milestone with its own test/verification budget, not a rushed addition
under a decision milestone's time pressure. Minimal API (for M53):

- `Tensor.sqrt()`, `Tensor.__truediv__`/`__rtruediv__` — ordinary general
  elementwise ops, CPU + CUDA, following the exact `tanh` pattern
  (`Backend.sqrt`/`sqrt_backward`, `Backend.div`/`div_backward` added to the
  ABC; CPU trivial via NumPy; CUDA via the existing `UNARY_LAUNCHER`/
  `ELEMENTWISE_LAUNCHER` kernel-generation pattern).
- `Module._buffers: dict[str, Tensor]` alongside `_parameters`/`_modules`,
  plus a `register_buffer(name, tensor)` method (or `__setattr__`-based
  auto-registration mirroring `Parameter`/`Module`, whichever review finds
  cleaner) — non-differentiable, moved by `Module.to()`, walked by a new
  `named_buffers()`/`buffers()` pair mirroring the parameter API.
- `forge/serialization/model.py`: `_build_save_node`/`_build_load_node`
  gain a `buffers` branch analogous to `parameters`, bumping
  `FORMAT_VERSION` if the archive shape changes (per
  `docs/architecture/decisions/ADR-003-persistence-format.md`'s existing
  versioning convention).
- `nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1)`: `weight`/`bias`
  `Parameter`s (affine scale/shift), `running_mean`/`running_var` buffers,
  reads `self.training` (existing `Module` API) to pick batch-vs-running
  statistics — the second consumer of the `.training`-dependent-forward
  pattern `Dropout` established at M16.
- One dedicated CUDA kernel pair (forward: per-channel mean/var/normalize/
  affine in one fused pass over `(N,C,H,W)`; backward: the standard
  BatchNorm gradient, which needs the batch mean/var/normalized values
  saved from forward, mirroring how `conv2d_backward`/`max_pool2d_backward`
  already recompute/reuse saved forward state) — scoped exactly to this
  op, not a general reduction/broadcast primitive.

**Affected subsystems**: `forge/tensor/tensor.py`, `forge/backend/base.py`
(+`cpu.py`+`cuda/{backend.py,kernels.cu}`), `forge/nn/module.py`,
`forge/nn/` (new `BatchNorm2d`), `forge/serialization/{model.py,registry.py}`
(persistence + registration), `examples/mnist/model.py` (or a variant, to
demonstrate it end-to-end) — no change to `Trainer`, `Loss`, `Optimizer`,
or `forge.data`.

**Validation strategy**: CPU forward vs. an independent NumPy reference;
finite-difference gradient check; CPU/CUDA parity (forward and backward);
train-mode-vs-eval-mode behavioral difference (a dedicated test proving
eval mode does *not* recompute batch statistics); buffer save/load
round-trip (values, not gradients, persisted; buffer restored on the right
device); a real training run on the BatchNorm-augmented MNIST CNN
confirming it still converges (comparable or better loss/accuracy than the
non-BatchNorm baseline) on both CPU and the 940MX.

**Explicitly excluded** (Section 11): `LayerNorm`/`InstanceNorm` (the
buffer mechanism should generalize to them later, but nothing requires
building them now), general N-D CUDA reduction/broadcast as standalone
primitives, CLI training commands, SGD momentum/scheduler, 1D/3D BatchNorm
variants.

## 13. Proposed next milestone (M53)

1. **Problem**: Forge's `Module` has no way to express non-trainable,
   persistent, device-and-mode-aware state — a normalization layer (the
   single most common near-term real-world CNN addition) cannot be built
   without it, and the gap has been independently identified three times
   without being acted on.
2. **Why it matters**: it is the one concrete, reusable architectural gap
   this milestone's direct-probe validation actually proved (Sections 9-10)
   — not a speculative "might be nice" addition.
3. **Concrete consumer**: `BatchNorm2d` inserted into the existing MNIST CNN
   (Section 8) — a real, trainable, checkpointable, CPU/CUDA-verified
   workload, reusing an existing example family rather than adding a new
   one.
4. **Minimum required capability**: `Module` buffers (register/move/
   persist/traverse), `Tensor.sqrt()`/division as general ops, one scoped
   CUDA BatchNorm kernel (forward+backward).
5. **Reused infrastructure**: `Trainer` (unmodified), `Adam`, `DataLoader`,
   the existing `register_module()`/archive persistence machinery (extended,
   not replaced), the existing `Parameter`/`.to(device)`/CUDA-kernel-pattern
   conventions.
6. **Success demonstrated by**: the validation strategy in Section 12 —
   parity tests, mode-dependent-behavior test, buffer persistence round
   trip, and a real convergent training run on real hardware (940MX).
7. **Out of scope**: everything in Section 12's exclusion list.
8. **Why more valuable than the rejected alternatives**: it is the only
   candidate with *proven*, not estimated, framework pressure (Section 9's
   probe), the only one whose payoff is reusable by future modules beyond
   this one workload, and the only one that closes a gap three separate
   milestones have already flagged rather than opening a new speculative
   one.

## 14. Scope boundaries

Buffers are scoped to what `BatchNorm2d` needs: a flat `name -> Tensor`
dict per `Module`, moved/persisted, not differentiated. No generic
"non-Parameter, non-Module Tensor state" framework beyond that (e.g. no
buffer-level requires_grad toggle, no partial/lazy buffer initialization
machinery) unless a future consumer demonstrates a need, exactly M49/M50's
established principle. `sqrt`/division are added as ordinary general
elementwise ops (like `tanh`), not restricted to BatchNorm's use — but no
other new elementwise op (`mean`, `transpose`, `softmax`, `sigmoid`) is
added without its own driving consumer.

## 15. Risks and limitations

- The CUDA BatchNorm kernel (forward mean/var/normalize/affine fused, plus
  a multi-term backward) is a genuinely larger single piece of CUDA
  engineering than any single M49-M51 addition (`tanh` was a pure
  elementwise op; this is a real reduction+broadcast+elementwise fusion) —
  M53 should budget for this explicitly rather than assume `tanh`-level
  effort.
- BatchNorm's behavior at MNIST's typical batch sizes/precision on the
  940MX has not been measured this milestone (deliberately — Section 9's
  Performance Rule forbids speculative CUDA profiling); M53 should treat
  correctness as the primary bar and only measure performance if a real
  slowdown is observed once BatchNorm actually trains.
- This assessment's probe (Section 9) covered only the CPU backend's
  forward math; the CUDA gap (Section 10, item 4) is confirmed by reading
  `backend.py`/`cuda-backend.md` directly, not by attempting and failing a
  CUDA probe on this machine's 940MX this session — M53's own
  implementation work is where that gets hardware-verified.
- As with every prior milestone's own limitations section: this is a
  point-in-time assessment against Forge's *current* two workload families;
  it is not a claim that BatchNorm is the only or best possible next
  capability in some absolute sense, only the best-evidenced one available
  from this survey.

## Verification

No production code was changed this milestone (`git status` clean except
this document and the progress-log append). Full existing suite re-run to
confirm the assessment did not alter repository behavior: **1,627 passed**,
single process (`pytest tests/`), unchanged from M51's own count — M51's
allocator fix means the CPU/CUDA-split workaround M50 required is no longer
necessary; this milestone ran the full suite as one process directly. The
`BatchNorm2d`-feasibility probe (Section 9) was run directly against the
real CPU backend (not simulated) and is not part of the committed test
suite (throwaway, scratchpad-only, per the brief's "prototype only what is
necessary" instruction).
