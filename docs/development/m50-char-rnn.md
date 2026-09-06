# M50 — Char-RNN: Forge's Second Model Family

## 1. Executive summary

M49 concluded that no framework capability was justified without a concrete
second model family to drive the requirement, and recommended making that
product decision first (`docs/development/m49-capability-assessment.md`,
Section 14). M50's brief required exactly that: select one model, attempt
it against Forge as M49 left it, and implement only the capabilities a
genuine blocker demonstrated.

**Model selected: a small character-level vanilla RNN language model**
(`examples/char_rnn/`), trained by next-character prediction on an
original, deterministically-generated synthetic corpus.

**One genuine blocker was found: `Tensor.tanh()`.** It was added, with full
CPU and real-hardware-verified CUDA support, exactly mirroring the existing
`relu`/`exp`/`log` Tensor-primitive pattern. `nn.RNNCell` (the vanilla RNN
recurrence step) was added as a thin composition of two `Linear` layers and
`.tanh()` -- no new autograd, persistence, data, or training-engine
machinery was required. `nn.Tanh` (the `Module` wrapper) was added for the
same reason `nn.ReLU` exists alongside `Tensor.relu()`.

The model trains and genuinely learns (mean per-character cross-entropy
drops from 2.74 to 0.36 nats over 30 epochs on the reference CPU, well
below the untrained 24-symbol-vocabulary baseline of `ln(24) ≈ 3.18`), with
confirmed CPU/CUDA parity on the reference 940MX. 40 new tests were added;
the full suite (1,619 tests: 721 CPU-only + 898 CUDA-hardware-verified) all
pass. A significant, unrelated pre-existing bug was also discovered and is
reported in Section 9 rather than fixed, per this milestone's scope.

## 2. Step 1 — Model selection

Candidates evaluated against the brief's criteria (meaningful capability
expansion, architectural diversity, dataset availability, hardware
practicality, likelihood of exposing genuine gaps, implementation
complexity, project value):

- **True binary classification** (sigmoid + BCE loss): smallest possible
  blocker (`sigmoid`), but marginal expansion -- `examples/trainer_demo.py`
  already trains a 2-class `CrossEntropyLoss` classifier end-to-end, so this
  would mostly repackage an existing capability.
- **A normalization-using architecture** (e.g. BatchNorm2d on the MNIST
  CNN): a real, meaningful addition (introduces persistent non-parameter
  module state -- "buffers" -- that Forge has never needed, per
  `docs/architecture/modules.md`'s existing "no buffer concept" note), but
  incremental to the *existing* CNN family rather than a new one, and pulls
  in `mean`/`sqrt`/division simultaneously -- a larger, multi-op blocker set
  for one milestone.
- **A small sequence/text model** (selected): the most architecturally
  distinct option from Forge's existing feedforward/CNN examples --
  genuine sequential processing, and (critically) the *same Parameter*
  reused across every timestep of one backward graph, which no existing
  Forge model exercises. Investigating the actual blocker set (Section 3)
  found it unexpectedly small: a plain one-hot input (built directly from
  raw character indices, no embedding/gather Tensor primitive needed at
  this vocabulary size) plus `tanh` covers a full vanilla RNN. This beat
  the normalization candidate on architectural diversity at *lower*
  implementation complexity and blocker count, once actually attempted
  rather than estimated.

**Rejected candidates are not just deferred for M50 -- see Section 6** for
why extending them (embeddings, LSTM/GRU gating, `Trainer` sequence
support) was deliberately not pursued even as a nice-to-have.

## 3. Step 2 — Architecture and workload

```text
one-hot(char_t)  (batch, vocab_size)
    -> RNNCell(vocab_size, hidden_size)  -> h_t  (batch, hidden_size)
    -> Linear(hidden_size, vocab_size)   -> logits_t  (batch, vocab_size)
```

- **Dataset**: `examples/char_rnn/corpus.py` -- an original, deterministically
  generated (seeded `numpy.random.Generator`) corpus of short templated
  sentences about Forge's own concepts (tensors, gradients, batches, ...).
  Not downloaded, not copied from any external or real-world text --
  the same synthetic-data convention `examples/trainer_demo.py` already
  uses (there: Gaussian blobs; here: templated sentences), chosen
  specifically to avoid any provenance/copyright question and keep the
  example fully offline. ~9,050 characters, 24-symbol vocabulary
  (lowercase letters, space, period).
- **Sequences**: fixed-length (`seq_len=40`), non-overlapping
  `(input, target)` index pairs, `target` = `input` shifted one character
  ahead, wrapped in the existing `forge.data.TensorDataset`/`DataLoader` --
  no new data-pipeline code.
- **Training objective**: per-timestep `CrossEntropyLoss` over the
  `seq_len`-step unroll, averaged; Adam (`lr=2e-2`).
- **Success criterion, defined before training**: mean per-character loss
  must drop well below the untrained-model uniform-guess baseline
  `ln(vocab_size)`, and generated samples should show recognizable
  word-level structure (spacing, periods, partial vocabulary) rather than
  random characters.
- Kept intentionally small for the i5-7200U/940MX target: default
  `hidden_size=64`, `batch_size=32` -- trains in ~4 seconds/30 epochs on
  CPU (Section 8).

## 4. Step 3 — Attempt with existing (post-M49) Forge

Attempted directly against the framework as M49 left it, before writing
any new code:

- **Worked immediately**: `Linear`, `CrossEntropyLoss`, `Adam`,
  `TensorDataset`/`DataLoader`, `save_model`/`load_model` (once
  `register_module()` was called for the new composite `Module` types, per
  the existing ADR-003 pattern every custom model already follows) --
  nothing here needed a workaround.
- **Workaround (not a blocker)**: one-hot encoding. Forge's Tensor has no
  embedding/gather primitive (M49 already confirmed this absence). At a
  24-symbol vocabulary, a plain NumPy one-hot vector built from each batch's
  raw character-index column, then wrapped as an ordinary `Tensor`, is
  exactly as correct and just as cheap as a dedicated lookup -- `examples/
  char_rnn/train.py`'s `_one_hot`, ~4 lines.
- **Workaround (not a blocker)**: training loop shape. `forge.training.
  Trainer.fit()` assumes one `forward(batch) -> prediction -> loss` call
  per step; a multi-timestep recurrence does not fit that shape. Rather
  than extend `Trainer` (a speculative addition no other Forge model would
  exercise), `train.py` uses the same hand-written
  `zero_grad -> forward loop -> backward -> step` loop Milestones 1-5 always
  supported. This is a fully first-class, precedented way to train in
  Forge, not a degraded fallback.
- **Verified, not assumed, before writing `RNNCell`**: does reusing the
  *same* `Linear` (and therefore the same `Parameter`s) at every timestep
  of one unrolled graph accumulate gradients correctly? Read
  `forge/autograd/engine.py::run_backward` directly: it walks a reverse
  topological order and, for any tensor with more than one consumer
  (`if id(inp) in pending: ... get_backend(inp.device).add(...)`),
  combines every contribution before that tensor's own turn is processed --
  order-independent of *how many* times a leaf is reused. No special case
  for weight sharing exists because none was needed; this "just worked" for
  a genuinely new usage pattern.
- **Genuine blocker**: `tanh`. Confirmed by direct execution (matching
  M49's own evidence standard) that `Tensor` has no `.tanh()`, `.sigmoid()`,
  or any other saturating nonlinearity -- a vanilla RNN's recurrence has no
  well-defined behavior without one (an unbounded linear recurrence is not
  the architecture this milestone selected).

## 5. Step 4/5 — The one justified capability

**`Tensor.tanh()`** (`forge/tensor/tensor.py`): elementwise
`tanh(x)`, backward `grad_output * (1 - result**2)` from the saved forward
*output* (the same shape `exp`'s backward rule already uses). Belongs in
`Tensor`/`Backend`, following the exact `relu`/`exp`/`log` pattern:
`Backend.tanh`/`tanh_backward` added to the ABC (`forge/backend/base.py`),
implemented on `CPUBackend` (`np.tanh`, ~2 lines) and `CUDABackend` (a real
kernel, `k_tanh`/`k_tanh_backward` in `kernels.cu`, `UNARY_LAUNCHER`/
`ELEMENTWISE_LAUNCHER` instantiations at f32/f64 -- no new CUDA
infrastructure, see `docs/architecture/cuda-backend.md`'s **CUDA tanh**
section). `nn.Tanh` (`forge/nn/activation.py`) is the `Module` wrapper,
added for parity with `nn.ReLU` even though `RNNCell` itself calls
`.tanh()` directly.

**`nn.RNNCell`** (`forge/nn/rnn.py`): one vanilla-RNN recurrence step,
`h' = tanh(x @ W_ih + b_ih + h @ W_hh)`, composed from two `Linear` layers
(the second without bias -- redundant with the first, since both feed the
same sum before `tanh`). No dedicated backward rule; correctness and CUDA
support follow entirely from `Linear` and `.tanh()`. Registered for
persistence (`forge/serialization/registry.py`, mirroring `Linear`'s own
registration) -- `RNNCell` carries no non-parameter state (the hidden state
is caller-threaded, not module state), so no buffer/running-statistics
persistence mechanism was needed, unlike what a normalization layer
would have required.

**Complexity/risk assessment**: both additions are minimal, low-risk,
precedented extensions of an existing, well-tested pattern (four prior
milestones -- M3, M9, M14 -- already established exactly this shape for
`relu`/`exp`/`log`). No existing public API changed; no existing test was
modified beyond adding new cases alongside `relu`/`exp` in the shared
CUDA-consistency/backend test files, per this repo's own convention (there
is no separate CPU-only `exp`/`log` test file either -- `tanh` got its own
`tests/test_tanh.py` since, unlike `exp`/`log`, it has no `CrossEntropyLoss`
already exercising it indirectly).

## 6. Rejected enhancements (deliberately not built)

- **Embedding/gather Tensor primitive**: a one-hot vector already gives the
  correct forward/backward at this vocabulary size; a dedicated lookup
  primitive would only matter at a vocabulary large enough for one-hot's
  `O(vocab_size)` per-step cost to bite (thousands+), which this milestone's
  model does not have and was not asked to support.
- **LSTM/GRU gating**: `nn.RNNCell` is deliberately the vanilla (Elman) cell
  -- the smallest recurrence that demonstrates weight-sharing-across-time
  and needs exactly one new primitive. Gated variants would need `sigmoid`
  too, with no consumer driving that requirement yet (same M49 principle:
  let a real model demand each op).
- **`Trainer` sequence-training support**: would require deciding a general
  multi-step-per-batch API shape with only one consumer to validate it
  against -- exactly the "speculative framework machinery" the brief
  prohibits. The hand-written loop this milestone uses is not a stopgap;
  it is Forge's original, still fully supported training pattern.
- **A `Vocab`/`Dataset`-level embedding-lookup convenience, a multi-layer
  stacked RNN, or truncated backpropagation-through-time for longer
  sequences**: all would only recombine or extend primitives with no
  current second consumer -- deferred until a model actually needs them.

## 7. Framework/API impact

- **New public API**: `Tensor.tanh()`, `Backend.tanh`/`tanh_backward`
  (abstract, both backends), `nn.Tanh`, `nn.RNNCell`. All additive; no
  existing signature changed.
- **New registry entries**: `"RNNCell"` (`forge/serialization/registry.py`,
  a Forge built-in, like `Linear`/`Conv2d`) and `"CharRNN"` (the example's
  own composite model, registered in `examples/char_rnn/model.py` --
  exactly the ADR-003 pattern any hand-written multi-layer model follows).
- **No change** to `forge.data`, `forge.optim`, `forge.training.Trainer`,
  or the serialization *format* -- only additive registry entries.

## 8. Training and validation results

Reference machine: i5-7200U, 8GB RAM, NVIDIA 940MX (CC 5.0, driver 582.53,
CUDA 12.6) -- `docs/development/development-environment.md`. Default
config: `hidden_size=64`, `seq_len=40`, `batch_size=32`, Adam `lr=2e-2`,
`--seed 0`.

| Epoch | CPU loss | CUDA loss |
|------:|---------:|----------:|
| 1     | 2.7433   | 2.7433    |
| 5     | 0.9251   | 0.9251    |
| 10    | 0.5150   | 0.5150    |
| 20    | 0.3909   | 0.3909    |
| 30    | 0.3554   | 0.3554    |

Untrained-baseline sanity check: `ln(24) ≈ 3.18` nats (uniform guess over
the 24-symbol vocabulary) -- epoch 1 starts near this (2.74, already below
uniform since the corpus's character *frequency* alone is informative) and
epoch 30 (0.355) is **~89% below the uniform baseline**, comfortably
clearing this milestone's pre-defined success criterion.

**CPU/CUDA parity**: every epoch's loss matches to 4 decimal places (one
epoch differs at the 4th decimal, 0.3827 vs. 0.3828 -- ordinary
floating-point accumulation-order noise), confirming `Tensor.tanh()` and
`nn.RNNCell` are numerically consistent across backends for a real,
multi-timestep training workload, not just the isolated-kernel level
(Section 5's CUDA-consistency tests already covered that in isolation).

**Sample generation** (200 characters, seeded `"a tensor"`, temperature-free
multinomial sampling from the trained model's softmax):

> a tensor thtcks a shape. a gradient holds a batch. a gradient holds a
> shape. the optimizer accumulatel a sample. a gradient holds a gradient.
> the optimizer optamulates a shape. the optimizer tesumple. the loa

Correct spacing, periods, and a majority of whole words/phrases directly
from the corpus's own vocabulary and templates (`"a gradient holds a
batch."` is an exact, verbatim template sentence) -- the small number of
garbled words (`"thtcks"`, `"accumulatel"`, `"optamulates"`) is consistent
with a genuinely learning, not memorizing-verbatim, 64-hidden-unit model on
a repetitive-but-not-trivial corpus, not a sign of a broken pipeline.

**End-to-end validation performed** (per Step 7's checklist): forward pass
(shape-checked, `tests/test_char_rnn_example_integration.py`), loss
computation, backward/autograd (including the weight-sharing property,
Section 4), optimizer updates (`test_training_updates_every_parameter`,
asserts every named parameter changed), multiple training iterations and
dataset iteration (the 30-epoch run above), measurable learning (the
89%-below-baseline result above, and a dedicated regression test asserting
final loss `< 0.5 * ln(vocab_size)`), CPU/CUDA parity (this section, plus
`tests/test_rnn_cuda.py` and `tests/test_char_rnn_example_cuda_integration.
py`), and save/load (`train.py`'s own round-trip assertion, plus
`test_model_persistence_preserves_predictions`).

## 9. A pre-existing bug discovered, not fixed (out of scope)

While running the complete test suite as one process, `pytest` deadlocked
inside `tests/test_trainer_cuda.py::test_cuda_trainer_classification_end_
to_end_learns` -- a pre-existing test unrelated to this milestone's changes.
Diagnosed directly with `py-spy` (two independent stack samples showed the
identical frame, confirming a true deadlock, not a slow-but-finite chain):

```
release (allocator.py:354, "with self._lock:")      <- innermost, blocked acquiring the lock
release (allocator.py:486)
__del__ (backend.py:532)                             <- a different CUDAStorage finalized mid-genexpr
<genexpr> (allocator.py:356, "any(b.value == ptr.value for b in blocks)")
release (allocator.py:356)                           <- OUTER call, already holding self._lock
release (allocator.py:486)
__del__ (backend.py:532)
run_backward -> Tensor.backward -> Trainer._run_training_epoch -> Trainer.fit
```

`CUDACachingAllocator._lock` (`forge/backend/cuda/allocator.py:244`) is a
plain, non-reentrant `threading.Lock`. `release()` holds it while
evaluating `any(b.value == ptr.value for b in blocks)`
(`allocator.py:356`). CPython's refcounting GC can finalize *any* object
whose refcount reaches zero at *any* bytecode step, including mid-genexpr;
here, evaluating that generator dropped the last reference to an unrelated
`CUDAStorage`, whose `__del__` called `release()` again, on the same lock,
from the same thread -- a self-deadlock.

This is a genuine, real, hardware-observed concurrency defect in the M25
caching allocator, confirmed with tooling (not inferred). It is **not**
caused by this milestone's changes: nothing in `allocator.py`, `backend.py`
`__del__`, or `run_backward` was touched by M50, and the RNN/tanh-specific
tests (`test_rnn_cuda.py`, `test_char_rnn_example_cuda_integration.py`)
were confirmed to pass in this exact run before the unrelated file
deadlocked later. It is rare enough to have never surfaced across 48 prior
milestones' worth of testing -- it depends on GC timing/accumulated object
churn across a long single-process run, similar in character to the
session-accumulation-dependent quirks already on file
(`forge-hardware-quirks` memory, M29/M47).

**Per explicit user direction this session: documented only, not fixed.**
`docs/development/progress.md`'s M50 entry and this document are the
record; a future milestone should fix `CUDACachingAllocator.release()`'s
reentrancy hazard (e.g. `threading.RLock`, or restructuring the check to
not hold the lock across arbitrary Python evaluation) as a small, isolated
concurrency fix, independent of any model/feature work. All verification
in this milestone (Section 10) was therefore done by running the CPU-only
and CUDA-only test subsets as two separate `pytest` invocations, which does
not hit this GC-accumulation-dependent path and gives 100% of the same
tests' outcomes as a single combined run would.

## 10. Testing

40 new tests, all passing:

- `tests/test_tanh.py` (7): CPU `Tensor.tanh()` forward, 2D, leaf-when-no-
  grad, analytic backward, upstream-gradient scaling, finite-difference
  gradient check, saturation at large magnitude.
- `tests/test_activation.py` (+3): `nn.Tanh` forward, no-parameters,
  backward.
- `tests/test_cuda_backend.py` (+2), `tests/test_cuda_consistency.py` (+1),
  `tests/test_cuda_autograd.py` (+1): CUDA `tanh` forward, unsupported-dtype
  error, CPU/CUDA consistency, CUDA backward vs. CPU -- mirroring the
  existing `relu`/`exp` cases in each file exactly.
- `tests/test_rnn.py` (11): `RNNCell` shapes/construction, invalid-input
  errors, forward correctness against manual NumPy, zero-input edge case,
  **multi-timestep weight-sharing gradient accumulation** (the property
  Section 4 verified before writing any code) both as a smoke check and a
  finite-difference numerical check over a 3-step unroll, and persistence
  registry round-trip.
- `tests/test_rnn_cuda.py` (4): single-step and multi-step-unroll CPU/CUDA
  parity, unrolled backward weight-gradient parity, CUDA residency.
- `tests/test_char_rnn_example_integration.py` (8) /
  `tests/test_char_rnn_example_cuda_integration.py` (2): the full example
  pipeline end-to-end (dataset/vocab, model shape, training-loss reduction
  below baseline, every-parameter-updates, generation, persistence
  round-trip; CUDA file additionally checks CUDA residency and CPU/CUDA
  first-epoch parity).

**Full suite**: 1,619 passed (1,579 + 40), verified as two separate
CPU-only (721) and CUDA-only (898, on the real 940MX) `pytest` invocations
per Section 9. A clean CUDA rebuild was performed (`kernels.cu` changed --
the compiled `_forge_cuda_kernels_sm_50.dll` was regenerated by `nvcc` on
first CUDA use in this session, confirmed by its timestamp moving past
`kernels.cu`'s).

## 11. Limitations

- The corpus is small (~9KB) and highly repetitive by design (Section 3) --
  generated text reproduces the corpus's own template structure rather than
  demonstrating novel-language generalization. This is intentional: the
  milestone's objective was validating Forge's framework, not language
  modeling quality.
- No embedding/gather primitive, no gated recurrence (LSTM/GRU), no
  multi-layer/stacked RNN, no truncated-BPTT for long sequences, no
  `Trainer` integration for sequence models -- all deliberately rejected
  (Section 6) as unjustified without a driving consumer.
- CUDA is slower than CPU for this specific tiny workload (33.2s vs. 4.2s
  for 30 epochs) -- expected and consistent with MNIST's own documented
  observation (`examples/mnist/README.md`) that per-step kernel-launch/
  transfer overhead dominates at small batch/hidden sizes on this hardware;
  not a regression, not investigated further (Section 12).
- Section 9's allocator deadlock remains unfixed by explicit scope decision
  this session -- the two-suites verification workaround is safe and
  complete for this milestone but is not a substitute for fixing the
  underlying reentrancy hazard.

## 12. Performance findings

Per the brief's conditional-performance rule: no CUDA profiling was
undertaken. The model is intentionally tiny (Section 3) and CPU already
trains it in ~4 seconds for 30 epochs -- there is no practical-usability
bottleneck to address, and Section 8's observed CUDA-slower-than-CPU result
at this scale is an already-well-understood, previously-documented property
of this hardware/workload-size combination (`examples/mnist/README.md`),
not a new finding warranting investigation. Conv2d's optimization thread
(closed M47/M48) is untouched and irrelevant here.

## 13. M51 recommendation

Two independent, unrelated next steps are now available:

1. **Fix the Section 9 allocator reentrancy bug** as a small, isolated
   concurrency fix (likely `threading.RLock`, or restructuring `release()`
   to copy what it needs before releasing the lock) -- low complexity, high
   value (removes a real, if rare, full-suite-run hazard), independent of
   any model/product work.
2. **A further product decision**, following M49/M50's own precedent:
   either extend the char-RNN family only if a concrete new consumer
   demands it (e.g. a longer-sequence task that would justify truncated
   BPTT, or a classification-over-sequences task that would justify
   `sigmoid`/gated recurrence), or select a third, still-more-diverse model
   family (e.g. a normalization-using architecture, still on M49's original
   candidate list and not diminished by this milestone's choice).

Absent a new product decision, do not re-run the M49 survey again --
nothing there is time-sensitive, and this milestone's own model choice has
already narrowly resolved the one item M49 flagged as blocking further
progress.
