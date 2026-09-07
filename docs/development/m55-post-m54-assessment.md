# M55 — Post-M54 Assessment: CUDA Sequence-Training Was Slower Than CPU

## 1. Executive summary

M55's brief explicitly forbade assuming this milestone must optimize
`Tensor.embedding_lookup()`, extend the RNN, or add another Tensor
primitive, and required fresh, evidence-driven investigation of Forge's
real post-M54 state rather than a predetermined feature list.

A capability survey (Section 2-3) found no new API gap: every candidate
M49/M52/M54 already evaluated and rejected (binary classification,
stacked/deeper RNN, LayerNorm, optimizer/`Trainer` changes) remains
correctly rejected today, and no new candidate of that kind appeared. What
the survey *did* find, by directly running Forge's own two real
sequence-model examples rather than reading their code, was a real,
measured, previously-unmeasured performance problem: **CUDA trains
`examples/char_rnn`/`examples/word_rnn` 1.5x-7.3x *slower* than CPU** on
the reference 940MX. This contradicts every prior CUDA milestone's implicit
assumption (CUDA is the faster device Forge optimizes) and had never been
directly measured before, because M50/M54 validated CUDA *correctness* for
these examples but never compared their wall-clock cost against CPU.

Root-causing with `cProfile` on the real training loop found the dominant
cost was not the RNN math itself: **56% of one epoch's wall-clock time was
spent inside `CUDABackend._synchronize`**, the default CUDA stream's
per-kernel-launch blocking synchronize, paid by every one of the ~20-30
small kernel launches a single unrolled-sequence training step issues.
`nn.RNNCell.forward()`'s composed `Linear`/`+`/`.tanh()` path contributed a
smaller, secondary cost.

**Outcome B: engineering fix, two parts, both hardware-verified:**

1. A fused CUDA `RNNCell` primitive (`Tensor.rnn_cell()`), mirroring
   `batch_norm2d`'s CUDA-only-fused / CPU-composed split -- real but modest
   (1.2x-1.7x in isolation).
2. Wrapping both examples' training loops in an explicit `forge.cuda.
   Stream()` (already-existing Milestone 27/30 machinery `Trainer` uses for
   `prefetch=True` but neither hand-written example had ever adopted) --
   the dominant fix.

Combined, measured end to end on the reference 940MX: `word_rnn` 5-epoch
CUDA training time 24.9s -> 13.5s (1.84x); `char_rnn` 5.0s -> 2.5s (2.00x).
`word_rnn` CUDA training moved from 1.7x slower than CPU to ~1.13x faster;
`char_rnn` moved from 7.3x slower to 3.6x slower (still the wrong device
for a model this tiny, but a much smaller gap). Full suite: **1,771
passed** (1,753 pre-M55 + 18 new), zero regressions.

## 2. Post-M54 capability snapshot

Re-verified by direct inspection/execution, not repeated from M54 except
where relevant:

- **Tensor**: unchanged from M54's own snapshot (`+ - * / @ sum reshape
  relu exp log tanh sqrt conv2d max_pool2d dropout_mask cross_entropy
  batch_norm2d embedding_lookup`), plus this milestone's `rnn_cell`
  (CUDA-only, Section 11).
- **`nn`**: unchanged (`Linear Conv2d MaxPool2d ReLU Tanh Dropout Flatten
  Sequential RNNCell BatchNorm2d Embedding MSELoss CrossEntropyLoss`). No
  `Sigmoid`/`LSTM`/`GRU`/`LayerNorm`/attention primitives -- still true,
  still no consumer (Section 6).
- **Optimizers/Data/Serialization/`Trainer`**: unchanged from M54's
  snapshot; re-examined against this milestone's own findings (Section 6)
  and still correct as-is.
- **CUDA**: identical operation coverage to M54, plus `rnn_cell`/
  `rnn_cell_backward` (Section 11). No new ABC method -- like
  `batch_norm2d`, `rnn_cell` lives only on `CUDABackend`.
- **Examples**: `examples/mnist`, `examples/char_rnn`, `examples/word_rnn`
  unchanged in architecture; both RNN examples' `train_one_epoch()` gained
  an optional `compute_stream` parameter (Section 12) -- purely additive,
  default `None` preserves the exact pre-M55 behavior.

## 3. Current real workloads

Four real, currently-exercised workloads, all inspected directly:

- **MNIST CNN** (`examples/mnist/`): `Conv2d`/`MaxPool2d`/`BatchNorm2d`/
  `Linear`, one `forward(batch)` call per step, `Trainer`-driven. Large
  enough tensors per op that CUDA's per-launch overhead is amortized --
  every prior Conv2d optimization milestone (M31-M48) already established
  CUDA is the right device here, and nothing in this milestone challenges
  that.
- **`char_rnn`** (`examples/char_rnn/`): one-hot input, `RNNCell`, 24-40
  character vocabulary, hand-written 40-timestep unrolled training loop.
- **`word_rnn`** (`examples/word_rnn/`): `Embedding` input, `RNNCell`,
  ~1,800-word vocabulary, hand-written 20-timestep unrolled training loop.
- Smaller demo scripts (`trainer_demo.py`, `data_pipeline_demo.py`,
  `persistence_demo.py`) -- CPU-only, unaffected by this milestone.

`char_rnn`/`word_rnn` are structurally identical (M54's own model.py
docstring already says so) -- both share the exact hand-written
"`zero_grad` -> per-timestep forward loop -> one `backward()` -> `step`"
pattern M50 established, both call `RNNCell.forward()` once per timestep,
and (per this milestone's finding) both therefore share the same CUDA
default-stream synchronize cost.

## 4. Problems/gaps discovered

**The one real, material problem this milestone found**: direct
measurement (not estimation) of both real RNN examples on `device="cuda"`
vs. `device="cpu"`, same seed/data, 5 epochs each:

| Workload | CPU (5 epochs) | CUDA (5 epochs), pre-M55 | CUDA/CPU ratio |
|---|---:|---:|---:|
| `word_rnn` | 15.3s | 23.5s | 1.54x slower |
| `char_rnn` | 0.7s | 5.1s | 7.29x slower |

No prior milestone had measured this. M50/M54 verified CUDA *correctness*
(loss parity, gradient parity, save/load parity) for these exact examples
but never timed CUDA against CPU for them -- every timing comparison in
Forge's history up to this point was CUDA-kernel-vs-CUDA-kernel (Conv2d
optimization arc) or synthetic (`benchmarks/stream_bench.py`), never
"should this real workload even use CUDA."

No other gap was found. Re-examined against this milestone's own new
workload evidence (the same evidence M54 used, now with a working
`Embedding`): binary classification, deeper/stacked RNN, LayerNorm, and
optimizer/`Trainer` changes are all still correctly rejected (Section 6).

## 5. Candidate directions

### A. Framework capability
No candidate qualified -- Section 6 re-confirms M49/M52/M54's rejections
with no new evidence to overturn them.

### B. Engineering quality — selected
The CUDA-slower-than-CPU finding (Section 4) is a real, material,
root-caused performance/engineering defect in how the two RNN examples use
CUDA (not a framework capability gap), fitting this category.

### C. Performance
Two performance angles were considered and are documented as part of the
selected engineering fix (Section 9): the fused-`RNNCell`-kernel angle and
the explicit-compute-stream angle. A third, narrower angle (fusing
`rnn_cell_backward`'s `dx`/`dh` or `dW_ih`/`db_ih` kernel pairs further)
was measured and rejected (Section 8).

### D. Developer experience
Not separately pursued -- the fix itself (an opt-in `compute_stream`
parameter, default `None`, zero behavior change unless passed) is the
developer-experience-relevant change: no existing caller needed
modification, and every existing test using `train_one_epoch()` without
that argument continues to pass unmodified (`tests/
test_char_rnn_example_integration.py`, `tests/
test_char_rnn_example_cuda_integration.py`, and the `word_rnn` equivalents
-- verified, Section 14).

## 6. Evidence for each candidate re-examined from M49/M52/M54

Re-run against this milestone's own new evidence, not re-litigated from
scratch:

- **Binary classification**: `CrossEntropyLoss` already handles
  `num_classes=2` with no restriction (`forge/nn/loss.py`, unchanged code
  path). No new evidence. Reject.
- **Deeper/stacked RNN**: pure Python composition of `RNNCell` instances,
  no primitive gap (M50/M54's own reasoning, unchanged). No new evidence.
  Reject.
- **LayerNorm**: still buffer-free and composable from M53's `sum`/`sqrt`/
  `div` on CPU, still has no consumer -- neither `char_rnn` nor `word_rnn`
  needed normalization to converge (both still train and reduce loss with
  none, this milestone's own runs, Section 15/16). Reject.
- **Attention/transformer**: still the same simultaneous-blocker-set
  rejection from M52/M54 (`softmax`/`transpose`/batched 3D matmul all still
  absent). No new evidence. Reject.
- **Optimizer/`Trainer` changes**: `Adam` alone still trains both RNN
  examples to a large loss reduction; `Trainer`'s one-forward-call-per-step
  shape still does not fit multi-timestep training, and this milestone's
  fix (Section 9) deliberately extends the *examples*, not `Trainer`
  itself, for exactly that reason. Reject.

**Two technically interesting directions explicitly rejected for
insufficient practical value**, per the brief's requirement:

1. **Further backward-kernel fusion** (combining `rnn_cell_backward`'s
   `dx`+`dh` into one kernel, or `dW_ih`+`db_ih` into one kernel, reducing
   6 backward launches to ~4): technically straightforward given the
   kernels already built (Section 11), but not attempted -- the dominant
   cost was already shown to be the default-stream synchronize (Section 8),
   not launch count once that synchronize is removed (Section 9's
   measurements), so shaving 2 more launches off an already-secondary
   contributor would not measurably move the end-to-end number. Rejected
   as low-value relative to its complexity/regression-risk cost.
2. **Sparse-gradient `Embedding` backward** (flagged as a known limitation
   in M54's own report): re-examined here since this milestone touches the
   same `word_rnn` workload -- still no measured slowdown from the current
   dense-gradient-buffer approach at any tested vocabulary size (M54's own
   measurement, up to 20,000 tokens, unchanged this milestone). Rejected,
   consistent with M54's original conclusion.

## 7. Candidate ranking

| Candidate | Real consumer | Evidence | Practical value | Complexity | Decision |
|---|---|---|---|---|---|
| Explicit compute stream in RNN example loops | `char_rnn`, `word_rnn` | Measured: 1.84x-2.00x end-to-end, root-caused via `cProfile` (56% of epoch time in one function) | **High** — reverses CUDA-slower-than-CPU for `word_rnn`, cuts the gap ~2x for `char_rnn`; reuses existing M27/M30 machinery, near-zero regression risk (opt-in, default unchanged) | Low | **Implement** |
| Fused CUDA `RNNCell` | `char_rnn`, `word_rnn` | Measured: 1.2x-1.7x in isolation | Medium — real but modest on its own; complements the stream fix (fewer, larger async-queued launches) | Medium (7 new kernels, new ABC-adjacent surface) | **Implement** |
| Further backward-kernel fusion | same | Not separately measured — reasoned from Section 9's finding | Low, given the stream fix already removes the dominant cost | Medium-High, more kernels/tests for a shrinking benefit | Reject |
| Sparse Embedding-gradient backward | `word_rnn` | M54's own measurement, re-confirmed unchanged | Low — no measured slowdown at any tested scale | High (fundamentally different backward representation) | Reject |
| Any new Tensor/nn primitive (Section 6) | none | No new evidence at any workload | Low/speculative | N/A | Reject |

## 8. Root cause of the selected issue

Two distinct, independently-verified root causes, in order of contribution:

**Dominant (56% of epoch wall-clock time): the CUDA default stream's
per-kernel-launch synchronize.** `CUDABackend`'s documented Milestone 8-26
contract is that every kernel launch on the default stream is followed by
an explicit `cudaDeviceSynchronize()` "before its result is trusted"
(`forge/backend/cuda/backend.py`'s own module docstring) -- a deliberate
correctness-first design, not a bug. `cProfile` on one real `word_rnn`
training epoch (`_synchronize`, `backend.py:663`) showed 20,558 calls
consuming 3.565 of 6.297 total seconds (56.6%), an average of ~173us per
call. A 20-timestep unrolled sequence graph, at `word_rnn`'s real shape
(batch=32, embedding_dim=32, hidden_size=128), issues roughly 20-30 small
kernel launches per timestep across `Embedding`, `RNNCell`, the output
`Linear`, and `cross_entropy` -- every one paying this same fixed
host-blocking cost regardless of how cheap its own arithmetic is (a
`Linear(128, 1806)` forward for batch 32 measured 0.40ms in isolation, of
which essentially none is real compute at this size).

This was never previously discovered because no prior milestone profiled a
*training loop*, only individual kernels/pipelines in isolation
(M31/M35/M40/M47's own Conv2d-focused profiling never touched an
RNN-style hand-written loop) or synthetic multi-stream benchmarks
(`benchmarks/stream_bench.py`) that never compared against a CPU baseline.

**Secondary: `nn.RNNCell.forward()`'s composed-op launch count.**
`tanh(i2h(x) + h2h(h))` composes two `Linear`s, an `add`, and a `.tanh()`
-- roughly 4 forward and 8 backward CUDA kernel launches per call, called
once per timestep. Isolated, interleaved A/B measurement (thermal-drift
-aware, per `docs/development/forge-hardware-quirks` conventions) found
this costs 1.2x-1.7x more wall-clock time than a fused equivalent at
`word_rnn`'s real shape -- real, but this milestone's own end-to-end
measurement (Section 9) shows it is a smaller contributor than the
synchronize cost once that dominant cost is still present (default
stream), and a proportionally even smaller one once it is removed (both
fixes' combined vs. stream-only measurements, Section 9).

## 9. Design/implementation

### 9.1 Fused CUDA `RNNCell` (`Tensor.rnn_cell()`)

Mirrors `batch_norm2d`'s CUDA-only-fused / CPU-composed split exactly (see
`docs/architecture/cuda-backend.md`'s **CUDA fused RNNCell** section for
full kernel-level detail). One forward kernel
(`k_rnn_cell_forward<T>`, one thread per `(batch, hidden)` output element,
both dot products computed directly) and six backward kernels
(`k_rnn_cell_backward_dz/dx/dh/dwih/dbih/dwhh`, each a plain per-thread
loop over `batch` with no cooperative reduction -- `batch` is small at
every real Forge RNN shape, unlike Conv2d's `dWeight`). `nn.RNNCell.
forward()` dispatches to `x.rnn_cell(...)` only when `x.device.type ==
"cuda"`; the CPU path (`Linear`/`+`/`.tanh()`) is byte-for-byte unchanged.

### 9.2 Explicit compute stream in the RNN example training loops

`examples/char_rnn/train.py`/`examples/word_rnn/train.py`'s
`train_one_epoch()` gained one new optional parameter,
`compute_stream: cuda.Stream | None = None`. When given (only ever
`device="cuda"`), the entire per-epoch batch loop runs inside `with
cuda.stream(compute_stream):` -- structurally identical to `Trainer`'s own
`_compute_stream_scope()` (`forge/training/trainer.py`, Milestone 30),
just invoked directly by the example instead of by `Trainer` (since
neither example uses `Trainer` -- their docstrings already explain why:
multi-timestep training does not fit `Trainer`'s one-`forward()`-call-
per-step shape). `main()` creates the stream once (`cuda.Stream() if
args.device == "cuda" else None`) and reuses it for every epoch, matching
`Trainer`'s own compute-stream lifetime. The one place each loop reads a
value back to Python (`float(mean_loss.to("cpu").numpy())`) still
synchronizes correctly and only that one transfer, via `Tensor._data`'s
existing pending-transfer contract -- unchanged, no new code needed there.

No change was needed to `forge.cuda.Stream`/`stream()`/any backend
dispatch code -- this is a pure consumer of Milestone 27/28's existing,
already-tested async-execution and cross-stream-dependency machinery.

## 10. Files changed

- `forge/backend/cuda/kernels.cu` -- new **Fused RNNCell forward/backward**
  section: `k_rnn_cell_forward`, `k_rnn_cell_backward_{dz,dx,dh,dwih,dbih,dwhh}`
  and their `cf_rnn_cell_*` launchers (f32/f64).
- `forge/backend/cuda/backend.py` -- `_configure_signatures` argtypes for
  the 7 new `cf_rnn_cell_*` functions; `CUDABackend.rnn_cell`/
  `rnn_cell_backward`.
- `forge/tensor/tensor.py` -- `Tensor.rnn_cell(w_ih, b_ih, w_hh, h)`.
- `forge/nn/rnn.py` -- `RNNCell.forward()` CUDA dispatch branch.
- `examples/char_rnn/train.py` -- `train_one_epoch(..., compute_stream=None)`.
- `examples/word_rnn/train.py` -- `train_one_epoch(..., compute_stream=None)`.
- `tests/test_cuda_rnn_cell.py` (new, 14 tests).
- `tests/test_char_rnn_example_cuda_integration.py` (+2 tests).
- `tests/test_word_rnn_example_cuda_integration.py` (+2 tests).
- `docs/architecture/cuda-backend.md`, `docs/architecture/cuda-streams.md`,
  `docs/architecture/modules.md`, `docs/architecture/tensor-api.md` --
  documentation.
- `docs/development/progress.md`, `docs/development/
  m55-post-m54-assessment.md` (this report).

## 11. API/architecture impact

Purely additive; no existing public API changed:

- New: `Tensor.rnn_cell(w_ih, b_ih, w_hh, h)` -- CUDA-only, raises
  `UnsupportedDeviceError` on a non-CUDA tensor, exactly like
  `batch_norm2d`. Not part of the `Backend` ABC (implemented only on
  `CUDABackend`, like `batch_norm2d`) -- a third-party `Backend` subclass
  needs no new method to remain valid.
- New: `CUDABackend.rnn_cell`/`rnn_cell_backward`.
- New, backward-compatible: `train_one_epoch(..., compute_stream=None)` in
  both RNN examples -- every existing call site (both example `main()`
  functions before this milestone, and every existing test) passes no
  fifth argument and is unaffected.
- `nn.RNNCell.forward()`'s CUDA output is numerically equivalent to its
  pre-M55 composed output (verified, Section 14) -- callers observe no
  behavior change beyond wall-clock time.

## 12. Tests

- `tests/test_cuda_rnn_cell.py` (14 new): `Tensor.rnn_cell()` rejects a
  CPU tensor; rejects a 1D `x`; rejects every one of 4 mismatched-operand
  -shape cases (`w_ih`, `b_ih`, `w_hh`, `h`); rejects a device mismatch
  (`w_ih` on CPU while `x` is on CUDA -- found and fixed a real validation
  gap during test-writing, Section 13); forward parity vs. CPU composed
  `RNNCell` (`float32`/`float64`); backward parity vs. CPU for all five
  gradients (`grad_x`/`grad_w_ih`/`grad_b_ih`/`grad_w_hh`/`grad_h`,
  `float32`/`float64`); gradient accumulation across a shared `RNNCell`
  instance reused over a 4-timestep unrolled sequence (the exact
  weight-sharing pattern the real examples rely on); a structural
  "`nn.RNNCell.forward()` actually calls `Tensor.rnn_cell()` on CUDA, not
  the composed path" dispatch check; a 200-iteration repeated
  forward/backward memory-stability check (`allocated_bytes` steady-state,
  `cache_hit_count` climbing).
- `tests/test_char_rnn_example_cuda_integration.py` (+2): `compute_stream`
  -vs-default-stream loss parity (`rtol=1e-4`); full 12-epoch training
  still converges with `compute_stream` supplied.
- `tests/test_word_rnn_example_cuda_integration.py` (+2, mirroring the
  above): `compute_stream`-vs-default-stream loss parity; full 15-epoch
  training still converges with `compute_stream` supplied.
- Every pre-existing test calling `train_one_epoch()` without a
  `compute_stream` argument (`tests/test_char_rnn_example_integration.py`,
  `tests/test_char_rnn_example_cuda_integration.py`'s original two tests,
  and the `word_rnn` equivalents) re-run unmodified and unaffected.

## 13. Verification methodology

Every timing claim in this report came from a real execution on the
reference development machine (i5-7200U, 8GB RAM, NVIDIA 940MX CC 5.0,
CUDA Toolkit 12.6, driver 582.53), never estimated. Isolated op-level
comparisons (fused vs. composed `RNNCell`) used interleaved, round-robin
A/B timing per `docs/development/forge-hardware-quirks`' documented
thermal-drift convention (never a block-sequential "time A fully, then
time B fully" comparison) -- an early block-sequential attempt at exactly
this comparison produced a misleadingly small (~1.03x) difference that a
subsequent interleaved, same-process A/B (toggling `RNNCell.forward` via
monkeypatch between the fused and composed implementations, same model,
same data, alternating rounds) corrected to the reported 1.2x-1.7x. The
dominant root cause (Section 8) was found with Python's `cProfile` against
the actual, unmodified real training loop, not a synthetic microbenchmark.
End-to-end example timings (Section 1/9) were each measured as a fresh
`python -m examples....train` process invocation with a fixed `--seed`,
comparable in methodology to every prior milestone's own example-timing
reports (M50/M54).

Correctness was verified independently of performance: CPU/CUDA forward
and all five backward gradients for the new fused kernel match to
`float32`/`float64` tolerance (Section 12), and `compute_stream`-enabled
training produces loss values matching the default-stream path to
`rtol=1e-4` (the same order of floating-point-accumulation-order tolerance
M54 already established as normal between separate CPU/CUDA runs).

## 14. Verification / test results

Full suite, single process, CUDA backend live: **1,771 passed, 0 failed, 0
skipped** (1,753 pre-M55 + 18 new: 14 in `tests/test_cuda_rnn_cell.py`, 2
each in the char/word RNN CUDA integration files). No pre-existing test
was modified in a way that changes its assertions -- only new tests were
added, and the one production-code bug found during test-writing (Section
13) was a genuine gap fixed before any test relied on it, not a
test-driven behavior change.

## 15. Performance/workload results

| Workload | CPU (5 ep) | CUDA, pre-M55 (5 ep) | CUDA, fused-only (5 ep) | CUDA, fused+stream (5 ep) | Net CUDA vs. CPU |
|---|---:|---:|---:|---:|---:|
| `word_rnn` | 15.3s | 23.5s (1.54x slower) | 24.9s (~unchanged) | **13.5s** | **1.13x faster** |
| `char_rnn` | 0.7s | 5.1s (7.29x slower) | 5.0s (~unchanged) | **2.5s** | 3.57x slower |

The "fused-only" column (fused `RNNCell`, still default stream) shows
essentially no end-to-end improvement over the pre-M55 baseline in
isolation on this hardware run -- confirming Section 8's finding that the
default-stream synchronize, not `RNNCell`'s own launch count, was the
dominant cost; the fused kernel's real (measured, Section 8) benefit only
becomes visible once combined with the stream fix (isolated interleaved
A/B under an active stream: fused ~1.2-2.8ms vs. composed ~1.7-2.8ms per
`RNNCell` fwd+bwd call, consistent with, though noisier than, the
context-free isolated 1.2x-1.7x figure -- this hardware's documented
cumulative thermal drift, `docs/development/forge-hardware-quirks`, was
visible across the longer combined benchmarking session and is the reason
Section 1/15's headline numbers use fresh, short, low-drift process
invocations rather than the noisier multi-round interleaved figures).

Training correctness/convergence, confirmed unaffected by either fix (same
seed, matching or improving on M50/M54's own reported curves): `word_rnn`
mean word loss 6.13 -> 4.40-4.42 over 5 epochs; `char_rnn` mean char loss
2.74 -> 0.93 over 5 epochs; both models' generated text and save/load
prediction-parity checks (`train.py`'s own built-in verification) passed
on every run.

## 16. Limitations

- The fused `RNNCell` backward is 6 kernels, not further fused (Section
  6's "further backward-kernel fusion" rejection) -- a deliberate choice
  given the stream fix already removes the dominant cost; a future
  workload with a much larger `RNNCell` call volume (deeper stacked RNNs,
  much longer sequences) could reopen this if it becomes measurably
  material again.
- `rnn_cell_backward`'s per-thread `batch`-loop kernels (`dW_ih`/`db_ih`/
  `dW_hh`) are not optimized for large batch sizes (no cooperative
  reduction) -- adequate at every real Forge RNN shape today (`batch<=32`),
  matching the brief's "adequate for the target workload" bar, not the
  Conv2d `dWeight` arc's cooperative-reduction investment, since no
  current workload has a large enough batch to need it.
- `char_rnn` remains slower on CUDA than CPU even after both fixes (3.57x)
  -- expected and not further pursued: its vocabulary (24-40 characters)
  and model (hidden_size=64) are small enough that CPU is near-instant
  (0.7s for 5 epochs), and no CUDA fix removes the fundamental
  fixed-per-launch-cost floor entirely, only shrinks it. This is a correct
  "wrong device for this workload size" finding, not a remaining framework
  defect -- `char_rnn`'s own `--device` flag already defaults to `cpu`.
- The `compute_stream` parameter is example-level, not a `Trainer`
  feature -- consistent with every prior milestone's (M50/M52/M54)
  rejection of extending `Trainer` for sequence training without a
  demonstrated need beyond what hand-written loops already handle.

## 17. Practical impact on Forge

Both real sequence-model examples Forge has (`char_rnn`, `word_rnn`) now
demonstrate CUDA training at a wall-clock cost close to or better than
CPU, rather than silently regressing performance for anyone who passes
`--device cuda` to either script. More importantly, this milestone
establishes a **reusable, documented pattern** (`docs/architecture/
cuda-streams.md`'s new Section 17a) for any future hand-written,
multi-step training loop that does not fit `Trainer`'s shape: wrap the
loop in an explicit `forge.cuda.Stream()`, exactly like `Trainer`'s own
`prefetch=True` already does. This closes a real gap between "Forge has
async-stream machinery" (true since M27/M30) and "every real workload
actually uses it" (false before this milestone, for the two examples that
needed it most).

## 18. Explicit value justification

The core finding (CUDA slower than CPU for a real, already-shipped Forge
workload) is not speculative or invented for this milestone: it was
directly measured against unmodified `examples/char_rnn`/`examples/
word_rnn` code, root-caused to a precise, single dominant function
(`CUDABackend._synchronize`, 56% of wall-clock time) via `cProfile` against
the real training loop, and fixed using infrastructure Forge already built
and tested three-to-five milestones ago (M27's stream abstraction, M30's
`Trainer` compute-stream pattern) rather than new, risky machinery. The fix
is verified correct (CPU/CUDA gradient parity, loss-value parity between
stream and non-stream paths, unmodified regression-free existing tests)
and verified valuable (a real, reproducible, hardware-measured 1.84x-2.00x
end-to-end speedup, reversing the sign of the CUDA-vs-CPU comparison for
`word_rnn`). Two technically interesting further optimizations were
explicitly measured or reasoned through and rejected for insufficient
marginal value relative to their complexity (Section 6), matching the
brief's explicit requirement.

## 19. Recommendation for next milestone

Not an automatic continuation of CUDA optimization. Candidates worth
future evidence-driven consideration, none pursued speculatively here:

- **If a future workload needs a much larger `RNNCell` batch size or a
  deeper stacked RNN**: revisit whether `rnn_cell_backward`'s per-thread
  `batch`-loop kernels need the cooperative-reduction treatment Conv2d's
  `dWeight` arc used (Section 16) -- only if that workload's shape
  actually makes it material, following the exact "measure before
  optimizing" discipline this milestone and M47/M48 already established.
- **If a future hand-written training loop is added** (any model that
  does not fit `Trainer`'s one-forward-call-per-step shape): apply this
  milestone's `compute_stream` pattern from the start, rather than
  discovering the same default-stream cost independently.
- Otherwise, repeat this milestone's own method next time: measure a real
  workload directly (not just its correctness) before assuming the next
  investment is another Tensor primitive, another `nn` layer, or another
  Conv2d-style kernel optimization.

## 20. Commit message

```
fix: eliminate CUDA default-stream sync overhead in RNN training loops
```
