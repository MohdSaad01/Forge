# CUDA Conv2d Backward Profiling and Optimization (Milestone 32)

## Purpose
Milestone 31 profiled the real asynchronous M27-M30 training pipeline and
found `conv2d` backward responsible for 55-57% of total CUDA training-step
time -- the dominant remaining cost after M31's CrossEntropy fusion. This
milestone profiles the three `conv2d_backward` kernels (`dInput`, `dWeight`,
`dBias`) individually, across shapes beyond the two fixed M20 MNIST layers
Milestone 21 measured, identifies the actual dominant component per shape,
and implements exactly one measurement-justified kernel optimization.

All numbers below are from the verified development machine: NVIDIA GeForce
940MX (GM108, Maxwell, 3 SMs / 384 CUDA cores), Compute Capability 5.0, CUDA
12.6, driver 582.53, Windows/WDDM, 2GB VRAM.

## How to reproduce
```bash
python -m benchmarks.conv2d_backward_profile   # this document's main source (new, Milestone 32)
python -m benchmarks.mnist_profile             # per-phase/per-op breakdown (Milestone 21 tool)
python -m benchmarks.pipeline_profile          # Milestone 31 async pipeline profile, rerun
python -m benchmarks.async_dataloader_bench    # prefetch speedup, rerun
```
Archived before/after results: `benchmarks/results/conv2d_backward_profile_baseline.json`
/ `conv2d_backward_profile.json` (isolated kernel timings, before/after --
the primary evidence for this milestone's optimization decision), plus
`m32_mnist_profile.json`, `m32_pipeline_profile.json`, and the
`m32_mnist_run{2,3}.json` / `m32_conv2d_backward_optimized.json` end-to-end
repeats.

## Existing implementation (baseline, going into Milestone 32)
`forge/backend/cuda/kernels.cu`'s "Conv2d / MaxPool2d" section (Milestone 15,
weight/bias reduction added Milestone 21):
- **`k_conv2d_forward`**: one thread per output element, looping `Cin x KH x
  KW` in registers.
- **`k_conv2d_backward_input`**: one thread per *input* element, looping
  `Cout x KH x KW`, resolving which output position each `(co, kh, kw)`
  combination reads from via `t % SH` / `t / SH` (and the `W` analog) --
  unchanged since Milestone 15.
- **`k_conv2d_backward_weight`** (one thread per weight element, full serial
  `N x Hout x Wout` reduction) and **`k_conv2d_backward_weight_reduce`**
  (one 256-thread block per weight element, shared-memory tree reduction):
  Milestone 21 made `cf_conv2d_backward_weight_*` dispatch between these by
  weight-element count (`CONV2D_WEIGHT_REDUCE_THRESHOLD = 256`) -- see
  `docs/architecture/cuda-backend.md`'s **CUDA Conv2d backward: weight/bias
  optimization (Milestone 21)** section. Unchanged this milestone.
- **`k_conv2d_backward_bias_reduce`**: one 256-thread block per output
  channel, always used (`Cout` never large enough for the per-thread
  variant to compete). Unchanged this milestone.

## Baseline workloads
Five shapes (`benchmarks/conv2d_backward_profile.py`'s `SHAPES` +
`BATCH_SWEEP_BASE`), chosen to vary batch size, channel count, and spatial
size independently, all `stride=1, padding=1` (`Hout=H, Wout=W`):

| Shape | N | Cin | Cout | H=W | K | weight elements | dWeight reduction size (`N*Hout*Wout`) |
|---|---|---|---|---|---|---|---|
| `mnist_conv1` (real M20 layer 1) | 64 | 1 | 8 | 28 | 3 | 72 | 50,176 |
| `mnist_conv2` (real M20 layer 2, post-pool) | 64 | 8 | 16 | 13 | 3 | 1,152 | 10,816 |
| `large_channel` | 64 | 16 | 32 | 28 | 3 | 4,608 | 50,176 |
| `large_spatial` | 32 | 8 | 16 | 56 | 3 | 1,152 | 100,352 |
| `batch_{32,64,128}` (= `large_channel`'s channel/spatial shape, N swept) | 32/64/128 | 16 | 32 | 28 | 3 | 4,608 | 25,088/50,176/100,352 |

All well within the 940MX's 2GB VRAM budget (largest, `batch_128`, uses
~26MB of live buffers).

## Baseline timings (940MX, mean of 30 iterations, 5 warmup, TimedEvent-measured)

| Shape | fwd (ms) | dInput (ms) | dWeight (ms) | dBias (ms) | backward total (ms) | dominant |
|---|---|---|---|---|---|---|
| mnist_conv1 | 0.67 | 1.09 | 2.18 | 0.23 | 3.49 | dWeight (62%) |
| mnist_conv2 | 1.38 | 3.59 | 1.77 | 0.10 | 5.45 | dInput (66%) |
| large_channel | 25.74 | 65.36 | 43.55 | 0.72 | 109.63 | dInput (60%) |
| large_spatial | 13.25 | 32.87 | 21.36 | 0.75 | 54.98 | dInput (60%) |
| batch_32 | 12.67 | 32.93 | 21.70 | 0.37 | 55.00 | dInput (60%) |
| batch_64 | 26.14 | 65.55 | 43.56 | 0.72 | 109.84 | dInput (60%) |
| batch_128 | 52.34 | 131.63 | 86.88 | 1.36 | 219.87 | dInput (60%) |

## dInput analysis
`k_conv2d_backward_input` launches one thread per input element (tens of
thousands to low millions across every shape tested here -- never
under-parallelized in the thread-count sense M21 already ruled out). Each
thread loops `Cout x KH x KW`, and for every `(kh, kw)` pair resolves the
corresponding output position via `t % SH` / `t / SH` (`W` analog for `kw`).
**That resolution does not depend on `co`** -- `hi`, `wi`, `PH`, `PW`, `SH`,
`SW`, `KH`, `KW` are all fixed for a given thread -- yet the original kernel
recomputed it *inside* the `co` loop, i.e. `Cout` times per thread (32 times
redundant at the `large_channel`/batch-sweep shape). Integer division/modulo
by a runtime-valued stride has no fast path on CC 5.0 hardware (`SH`/`SW`
are kernel arguments, not compile-time constants, so the compiler cannot
special-case `stride=1`), making this redundant work disproportionately
expensive relative to its equivalent FLOP count.

**Memory access**: coalesced along `wi` for `stride=1` shapes (adjacent
threads read adjacent `grad_out`/output columns); `w` (18-144KB across the
tested shapes) is small enough to stay resident in L1/L2 and heavily reused
across the `N*H*W` threads sharing a given `ci`; `grad_out` is reread by
every one of a pixel's `Cin` threads (a real, unchanged read-amplification
inherent to the one-thread-per-input-pixel design -- noted under
**Limitations**, not addressed this milestone).

**Occupancy** (`nvcc -Xptxas -v`, `sm_50`, `float32`): 48 registers/thread,
512 bytes local ("stack") memory per thread (the two small fixed-size valid-
pair tables the fix below introduces -- see **Optimization implementation**).
At 65,536 registers/SM this still allows several resident blocks per SM;
occupancy was not the limiting factor either before or after this
milestone's change.

## dWeight analysis
Unchanged this milestone (Milestone 21 already measurement-tuned this
kernel's dispatch). At small weight-element counts (`mnist_conv1`, 72
elements) the block-reduction kernel dominates backward time simply because
`dInput` is comparatively cheap at that tiny shape. At larger counts
(>= `CONV2D_WEIGHT_REDUCE_THRESHOLD`), the one-thread-per-weight kernel's
serial `N x Hout x Wout` reduction becomes the second-largest cost at every
shape tested (43.6ms at `large_channel`, 60% the size of `dInput`'s
now-larger cost even before this milestone's fix) -- see **Limitations /
Future opportunities** below: a real remaining optimization target, out of
scope for this milestone's "one change" rule (Section 18).

## dBias analysis
Negligible everywhere measured: 0.10-1.36ms, 1-2% of `conv2d_backward`
total at every shape. Confirms Milestone 21's original finding still holds
-- `Cout` never grows large enough for this to matter for any Forge-scale
CNN. **Left unchanged**, per the milestone brief's "measure, don't assume"
instruction for this specific kernel.

## Arithmetic intensity / bottleneck classification
`dInput` and `dWeight` compute the same total multiply-accumulate count for
a given layer shape (`N x Cin x Cout x KH x KW x Hout x Wout`, symmetric in
input/weight-gradient formulations of convolution) -- yet, before this
milestone's fix, `dInput` measured 1.3-1.8x slower than `dWeight` at every
non-MNIST-scale shape despite equal FLOPs. The direct evidence this is
*not* a memory-bandwidth story: the fix below touches zero memory-access
patterns (same reads, same writes, same coalescing) and changes only
redundant integer ALU work -- yet produced the wall-clock improvements in
**Before/After Conv2d Results**. A kernel that speeds up purely from
removing scalar integer arithmetic, with no change to what or how much
memory it touches, was compute/instruction-latency-bound on that arithmetic,
not memory-bandwidth-bound. (`dWeight`'s per-thread kernel has no
`co`-independent division to hoist -- its inner-loop `hi`/`wi` computation
is a multiply-add, not a division -- so it was not a candidate for this
specific fix; see **Limitations**.)

## Kernel launch analysis
`conv2d_backward` issues exactly 3 kernel launches (`dInput`, `dWeight`,
`dBias`), each already followed by `_maybe_synchronize()` in default-stream
mode. At every shape measured here, each individual kernel's own runtime
(0.1-131ms) is one to five orders of magnitude larger than this GPU's
measured per-launch dispatch overhead (~54-65us, `stream_dependency_bench.py`,
M31). **Launch overhead is not a factor for Conv2d backward** -- this is a
compute-bound kernel-internals problem, not a launch-count problem (unlike
M31's CrossEntropy finding); no kernel fusion was considered or needed.

## Bottleneck ranking (baseline, before this milestone)
Aggregated across all seven measured shapes' `backward_total_ms`:
1. **dInput** -- dominant at 6 of 7 shapes (all but `mnist_conv1`), 60-66%
   of `conv2d_backward` at every non-MNIST-scale shape.
2. **dWeight** -- dominant only at `mnist_conv1` (72 weight elements, the
   smallest tested), 30-40% elsewhere.
3. **dBias** -- never more than 2% anywhere measured.

At the real M20 MNIST CNN specifically (summing both real conv layers):
`dInput` (1.09 + 3.59 = 4.68ms) already slightly exceeds `dWeight` (2.18 +
1.77 = 3.95ms) even though `dWeight` was M21's headline optimization target
-- a genuinely new, measurement-driven finding this milestone's broader
shape sweep surfaced that per-layer numbers alone would have hidden.

## Optimization selected
**Hoist the `co`-independent `(kh, ho)`/`(kw, wo)` validity resolution out of
`k_conv2d_backward_input`'s `co` loop**, computing each thread's small set of
valid kernel-window/output-position pairs once into local tables, then
looping `co` over pure array indexing and multiply-accumulate -- zero
division inside the hot loop. Candidate B from Section 16 of the milestone
brief (register/shared-memory-level loop restructuring) rather than A
(parallel-reduction restructuring, already done for `dWeight` in M21) or C
(different block geometry -- thread-count parallelism was never the
`dInput` problem).

### Root cause
`hi`, `wi`, `PH`, `PW`, `SH`, `SW`, `KH`, `KW` are fixed per thread and do
not depend on the loop variable `co` at all, yet the original kernel's
`t % SH` / `t / SH` (and the `W`-dimension analog) sat inside the `co` loop,
paying `Cout`-fold redundant integer division for identical results every
time. Integer division on CC 5.0 is a multi-instruction software sequence
with no fast path for a runtime (non-compile-time-constant) divisor, making
this a disproportionately expensive redundancy relative to its FLOP
contribution.

### Implementation
`kernels.cu`'s `k_conv2d_backward_input`: each thread first builds two fixed-
size local tables (`kh_valid`/`ho_valid`, `kw_valid`/`wo_valid`, capped at
`MAX_CONV_K = 32` -- generous for any kernel size Forge tests or plausibly
uses; K stays in {1, 2, 3, 5} across every model in the repo) by running the
*same* validity/division logic as before, but exactly once per thread
instead of once per `(co, kh)`/`(co, kh, kw)` triple. The `co` loop then
only does integer multiply/add for `g_idx`/`w_idx` and the same
multiply-accumulate the kernel always needed. This computes the identical
set of `(co, kh, ho, kw, wo)` contributions, in the same per-`co` summation
order, as the original kernel -- a pure redundant-work removal, not an
algorithm change. `cf_conv2d_backward_input_{f32,f64}`'s exported symbol
name, signature, and dispatch are all unchanged; `CUDABackend.conv2d_backward`
required no Python-side change.

## Correctness validation
- `tests/test_cuda_conv.py` (22 pre-existing tests, unmodified): forward vs.
  CPU, backward vs. CPU across `(kernel_size, stride, padding)` in `{(3,1,0),
  (3,1,1), (3,2,1), ((3,2),(2,1),(1,0))}`, weight-reuse accumulation, input
  finite-difference, no-CPU-fallback spy test, full `TinyCNN` train-and-
  converge -- all pass unchanged.
- `tests/test_cuda_conv2d_backward_optimization.py` (new, 8 tests):
  weight/bias finite-difference checks (input FD already existed; weight/bias
  did not), explicit-async-stream backward correctness, two cross-stream
  correctness tests (producer streams for `x`/`weight`/`bias` distinct from
  the compute stream; upstream gradient produced on a distinct stream from
  `backward()`'s own stream), and a repeated-use memory-safety test.

## Finite difference results
`test_cuda_conv2d_finite_difference_input` (pre-existing, unmodified) and
the two new weight/bias finite-difference tests all pass at `rtol=atol=1e-2`
(float64, matching the existing input FD test's tolerance) across three
`(stride, padding)` combinations.

## Cross-stream results
`test_cuda_conv2d_backward_correct_when_inputs_from_different_streams` and
`test_cuda_conv2d_backward_correct_when_grad_output_from_different_stream`
both pass -- `CUDABackend._require_compute_dtype`'s existing `_stream_guard`
chokepoint (unchanged since M28) already covers `conv2d_backward`'s three
kernel launches; this milestone's kernel-internals change needed zero new
dependency-tracking code.

## Async results
`test_cuda_conv2d_backward_on_explicit_stream_matches_cpu` passes: forward +
backward on an explicit `forge.cuda.Stream()`, verified against CPU.

## Memory results
`test_cuda_conv2d_backward_repeated_use_does_not_grow_active_memory`: 30
repeated forward+backward iterations, `allocated_bytes`/`reserved_bytes`/
`pending_bytes` all return to baseline (0 growth) after `gc.collect()` +
`empty_cache()`. No temporary CUDA workspace was introduced (`MAX_CONV_K`-
sized tables are thread-local, not device-allocated) -- allocator behavior
is completely unaffected by this change.

## Before / after Conv2d results (isolated kernel, `conv2d_backward_profile.py`)

| Shape | dInput before | dInput after | speedup | backward total before | backward total after | speedup |
|---|---|---|---|---|---|---|
| mnist_conv1 | 1.09ms | 0.72ms | 1.51x | 3.49ms | 3.12ms | 1.12x |
| mnist_conv2 | 3.59ms | 2.02ms | 1.78x | 5.45ms | 3.98ms | 1.37x |
| large_channel | 65.36ms | 36.08ms | 1.81x | 109.63ms | 81.08ms | 1.35x |
| large_spatial | 32.87ms | 19.19ms | 1.71x | 54.98ms | 41.50ms | 1.33x |
| batch_32 | 32.93ms | 18.84ms | 1.75x | 55.00ms | 41.58ms | 1.32x |
| batch_64 | 65.55ms | 36.56ms | 1.79x | 109.84ms | 81.23ms | 1.35x |
| batch_128 | 131.63ms | 73.04ms | 1.80x | 219.87ms | 160.30ms | 1.37x |

`dWeight`/`dBias`/`forward` measurements are unaffected within run-to-run
noise (this kernel change touches only `k_conv2d_backward_input`), confirmed
directly in the same before/after run.

## Before/after full training results
`benchmarks.mnist_profile`'s instrumented per-op walker (adds its own
synchronization between every op, per that tool's own documented caveat --
absolute numbers are not directly comparable to the async pipeline, but the
`conv2d`-op number is an apples-to-apples "everything else held fixed" data
point since nothing besides `k_conv2d_backward_input` changed since the
pre-M32 snapshot): `conv2d` backward op 11.16ms -> 9.95ms (~1.12x) at
batch=64 real M20 MNIST shapes -- consistent in direction with the isolated
`mnist_conv1`/`mnist_conv2` results above, smaller in magnitude because
MNIST's real layer shapes are the two smallest tested (where `dInput`'s
absolute contribution, and therefore this fix's absolute savings, is
smallest) and because `dWeight` -- left unchanged -- remains substantial at
`mnist_conv1`.

## MNIST results
`benchmarks.mnist_bench` (`training` category), CUDA, batch=64, 30
iterations/5 warmup, three repeated runs (this machine is shared/
non-dedicated -- run-to-run noise is real, matching M21's own documented
caveat): 18.53ms, 15.34ms, 17.39ms (mean 17.09ms, stdev 5.75/1.08/4.03ms
per run). The pre-M32 (post-M31-fusion) baseline was 17.90ms -- within this
run-to-run noise band, not a clean isolated win at this instrumentation
level. The isolated-kernel and per-op-walker results above are the more
reliable signal at MNIST's small shapes; the real, larger, more clearly
attributable win is in the async pipeline measurement below.

## Async prefetch results
`benchmarks.async_dataloader_bench`, real MNIST-shaped workload
(n=1024, batch_size=64): synchronous 294.6ms/epoch, prefetch 243.1ms/epoch,
**1.212x speedup** -- consistent with M30's 1.21x and M31's 1.18x within
normal variation. Conv2d acceleration did not change the CPU-prep/transfer-
vs-compute balance qualitatively (both phases scaled down together at this
workload size); memory-safety counters (CUDA active bytes, pinned active
bytes) both returned to 0 after the run.

## Updated pipeline profile
`benchmarks.pipeline_profile`, batch_size=64, prefetch_size=2 (this
milestone's post-optimization state): `fwd=2.05ms, loss=0.69ms,
bwd=8.48ms, opt=0.08ms`, compute-stream utilization 88.7%, epoch throughput
5,031 samples/sec. Compared against the checked-in pre-M31/pre-M32 baseline
snapshot (`bwd=10.82ms`, 3,915 samples/sec at the same batch size) --
**note this comparison includes both M31's CrossEntropy fusion and this
milestone's Conv2d fix together**, since no clean post-M31-only snapshot
from this exact hardware session was captured; `loss` (0.69ms, already small
in the composed/overlapped pipeline measurement) and `fwd` (2.05ms,
essentially unchanged from the pre-M31 baseline's 2.05ms) bound M31's own
contribution to this particular table, making the `bwd` column's ~22%
reduction (10.82ms -> 8.48ms) attributable predominantly to this milestone.

## New bottleneck ranking
Backward remains the dominant phase (`bwd` = 8.48ms of `fwd+loss+bwd+opt` =
11.30ms total compute-busy time per step, ~75%), same ordering as M31
(`backward > forward > loss > optimizer`) but with backward's absolute cost
reduced. Within `conv2d_backward` specifically, `dWeight`'s unchanged
per-thread kernel is now the largest single remaining contributor at most
non-MNIST shapes (see **Bottleneck ranking** above, re-measured after this
fix) -- the natural M33+ candidate, not addressed here per Section 18's
"one change" rule.

## CPU regression
`CPUBackend.conv2d_backward` was not touched (this milestone is CUDA-only,
per the milestone brief). All CPU `Conv2d`/training tests in the full suite
pass unchanged (see **Full test suite** below); no CPU benchmark numbers
were expected or observed to move.

## CUDA regression
The `backward` and `mnist` benchmark categories rerun together
(`python -m benchmarks --categories mnist backward`) show every non-Conv2d
CUDA operation (elementwise, matmul,
`max_pool2d`, `cross_entropy_loss`, multilayer Linear/ReLU/Linear) within
normal run-to-run noise of their own historical numbers; only `conv2d`
(and, downstream, `mnist_cnn_full`) moved, and only in the improving
direction.

## Stress testing / leak testing
`test_cuda_conv2d_backward_repeated_use_does_not_grow_active_memory` (30
iterations) plus the pre-existing `test_cuda_conv2d_weight_reuse_accumulates_matching_cpu`
(gradient-accumulation correctness across repeated backward calls) --
both pass. No new CUDA workspace allocation was introduced, so no new
leak surface exists.

## Full test suite
1,154 pre-existing tests (unmodified) + 8 new
(`tests/test_cuda_conv2d_backward_optimization.py`) = **1,162 tests, all
passing** on the 940MX (`python -m pytest tests/ -q`).

## Hardware verification
Every measurement in this document was collected on the real, verified
development GPU (NVIDIA GeForce 940MX, CC 5.0, CUDA 12.6, driver 582.53) --
no simulated or emulated CUDA behavior anywhere in this milestone, per
`CLAUDE.md`'s "CUDA support must be real and hardware-tested" constraint.

## Limitations / future opportunities
- **`dWeight`'s per-thread kernel** (used when weight-element count >=
  `CONV2D_WEIGHT_REDUCE_THRESHOLD`) still does a full serial `N x Hout x
  Wout` reduction per thread with no cooperative reduction -- now the
  largest single remaining `conv2d_backward` cost at most non-MNIST shapes.
  A real M33+ candidate (e.g. a warp- or small-group-cooperative reduction
  sitting between the existing two strategies), deliberately not pursued
  here per Section 18's "start with one optimization" rule.
- **`dInput`'s `grad_out` read amplification** (every input pixel's `Cin`
  threads independently reread the same `grad_out` rows) is unchanged --
  a shared-memory tiling scheme over `co`/spatial neighborhoods could
  reduce this, but was not the measured bottleneck this milestone (the
  division-hoist fix already fully explains the measured speedup with zero
  memory-pattern change) and would be a materially larger rewrite.
- `MAX_CONV_K = 32` bounds the new local tables; no Forge model or test uses
  `K > 5`, but a hypothetical `K > 32` kernel would silently truncate rather
  than error. Documented, not enforced at the Python boundary (matching
  this kernel file's existing convention of trusting its own internal
  constants rather than validating them per-call).

# Milestone 33: dWeight cooperative reduction (investigated, rejected)

## Purpose
Milestone 32 left `dWeight` (`k_conv2d_backward_weight`, unchanged since
Milestone 21: one thread per weight element, full serial `N x Hout x Wout`
reduction) as the largest single `conv2d_backward` cost at 6 of the 7
representative shapes -- the natural next target its own report identified.
This milestone profiles `dWeight` specifically, determines the exact
reduction size per shape, and experimentally evaluates whether a
cooperative (multi-thread-per-weight-element) reduction can beat the
existing per-thread kernel at the weight-element counts where production
currently uses it.

## How to reproduce
```bash
python -m benchmarks.conv2d_backward_weight_profile   # this section's main source (new, Milestone 33)
python -m benchmarks.conv2d_backward_profile           # re-run of the M32 profiler -- confirms no regression
python -m benchmarks.pipeline_profile
python -m benchmarks.async_dataloader_bench
python -m benchmarks --categories mnist backward forward
```
Archived results: `benchmarks/results/m33_conv2d_backward_weight_profile.json`
(the primary evidence for this milestone's reject decision),
`m33_conv2d_backward_profile_after.json` (re-run of M32's profiler, confirms
`dInput`/`dWeight`/`dBias` unchanged), `m33_pipeline_profile.json`,
`m33_regression_mnist_backward.json`, `m33_regression_forward.json`.
`async_dataloader_bench.py` has no `--output`/JSON-archive option (console
report only, unchanged since M29) -- its **Async Prefetch Results** number
below is taken directly from that console output, not an archived file.

## Existing dWeight architecture (unchanged going into this milestone)
`forge/backend/cuda/kernels.cu`'s `cf_conv2d_backward_weight_{f32,f64}`
dispatches at launch time between:
- **`k_conv2d_backward_weight`** (per-thread): one thread per
  `(co, ci, kh, kw)` weight element, looping the entire `N x Hout x Wout`
  reduction serially in registers. Used when `weight_elements =
  Cout*Cin*KH*KW >= CONV2D_WEIGHT_REDUCE_THRESHOLD` (256).
- **`k_conv2d_backward_weight_reduce`** (block-reduce): one 256-thread block
  per weight element, each thread striding through a slice of the reduction
  via a grid-stride loop, combined by a shared-memory tree reduction. Used
  when `weight_elements < 256`.

This is the Milestone 21 hybrid dispatch, measurement-tuned at exactly two
shapes (72 and 1,152 weight elements) -- this milestone re-measures it
across the full M32/M33 shape set.

## Baseline workloads and reduction sizes
The same 7 M32 shapes, unmodified (`benchmarks/conv2d_backward_profile.py`'s
`SHAPES`/`BATCH_SWEEP_BASE`/`BATCH_SIZES`, imported directly rather than
duplicated). `reduction_elements_per_weight = N * Hout * Wout` -- the exact
number of `(n, ho, wo)` triples each weight element's gradient sums over
(fewer than that when padding excludes some triples for a given `(kh, kw)`,
but this is the loop trip count every candidate kernel actually executes):

| Shape | weight_elements | reduction_elements_per_weight | Current M21 path |
|---|---|---|---|
| mnist_conv1 | 72 | 50,176 | block-reduce (< 256) |
| mnist_conv2 | 1,152 | 10,816 | per-thread |
| large_channel | 4,608 | 50,176 | per-thread |
| large_spatial | 1,152 | 100,352 | per-thread |
| batch_32 | 4,608 | 25,088 | per-thread |
| batch_64 | 4,608 | 50,176 | per-thread |
| batch_128 | 4,608 | 100,352 | per-thread |

Only `mnist_conv1` currently uses block-reduction; every other shape --
including both shapes that repeat `mnist_conv2`'s 1,152-element weight
count at far larger reduction sizes (`large_spatial`: 100,352 vs. 10,816) --
uses the per-thread kernel. This is the first time the hybrid dispatch has
been measured with `weight_elements` and `reduction_elements_per_weight`
varied independently rather than covarying (M21's own two data points both
had reduction sizes in the same order of magnitude).

## Existing M21 hybrid threshold: still correct
Forcing the block-reduce kernel (256 threads/block) at `mnist_conv1` (72
elements, 50,176-element reduction) measures 2.11ms vs. a forced per-thread
run's 8.25ms -- block-reduce remains ~3.9x faster, confirming the threshold
is not stale even though M32 changed `dInput` substantially elsewhere in
the same kernel file. See **Cooperative Strategy Evaluated** below for why
this does *not* generalize to larger weight-element counts.

## dWeight parallelization / occupancy analysis
Per-thread kernel launch: `launch_config` gives 256 threads/block,
`ceil(weight_elements/256)` blocks -- at every shape with `weight_elements
>= 1,152`, this launches 1,152-4,608 total resident threads, several times
the 940MX's 384 CUDA cores (3 SMs x 128 cores, Maxwell). Each thread uses
few registers (no local arrays, unlike `k_conv2d_backward_input`'s M32 fix)
and touches no shared memory, so occupancy is not register- or
shared-memory-limited -- the GPU can keep essentially all of these threads
resident simultaneously. Memory access per iteration: `grad_out[g_idx]`
depends only on `(n, co, ho, wo)` -- **not** on `ci`, `kh`, or `kw` -- so
every thread within a `(co)`-group (up to `Cin*KH*KW` adjacent threads,
e.g. 144 at `large_channel`) reads the *same* `grad_out` address at a given
loop iteration, a broadcast read serviceable by one cache-line fetch across
many concurrent threads. `x[x_idx]` varies smoothly with the fast-varying
`kw` (adjacent threads read adjacent `x` addresses), giving reasonable
coalescing. Net effect: thousands of independent, cheap, well-cached
threads in flight together, which is enough memory-level parallelism to
hide the reduction's latency without any explicit cooperative reduction.

## Arithmetic intensity / bottleneck classification
Same total multiply-accumulate count regardless of reduction strategy
(`weight_elements * reduction_elements_per_weight`, symmetric with
`dInput`'s own FLOP count for a given layer). Both cooperative candidates
below perform *identical* arithmetic to the per-thread kernel, just
distributed differently across threads -- the measured 3-4x slowdown is
therefore not an arithmetic-intensity story; it is about how many
independent memory streams the GPU has in flight, and how much
synchronization/reduction overhead is paid per unit of useful work (see
**Cooperative Strategy Evaluated**).

## Kernel launch analysis
Both cooperative candidates preserve `conv2d_backward`'s existing 3-launch
structure (`dInput`, `dWeight`, `dBias`) when/if dispatched into production
-- no launch-count change was ever on the table. Per-call launch overhead
(~54-65us, `stream_dependency_bench.py`, M31) remains one to three orders
of magnitude smaller than any of the `dWeight` kernel times measured here
(1.8-278ms) -- launch overhead is not a factor for this decision either way.

## Cooperative Strategy Evaluated
Two designs, at a fixed shape, independent of the M21 threshold, via new
profiling-only exports (never called by `CUDABackend` -- see `kernels.cu`'s
matching comment):

**Candidate A -- block-per-weight-element** (`cf_conv2d_backward_weight_
blockreduce_*`, forces the *existing* `k_conv2d_backward_weight_reduce`
regardless of `weight_elements`): one block owns one weight element;
`threads_per_block` in {64, 128, 256} tested.

**Candidate B -- warp-per-weight-element** (`cf_conv2d_backward_weight_
warpreduce_*`, new `k_conv2d_backward_weight_warp`): one warp (32 lanes)
owns one weight element, reduced via `__shfl_down_sync` (no shared memory,
no `__syncthreads()`); multiple weights packed per block via
`warps_per_block` in {2, 4, 8} (64-256 threads/block, 2-8 weights/block) --
tests whether shrinking the total block count (up to 8x fewer launched
blocks than Candidate A at the same block size) changes the outcome.

### Results (940MX, mean of 30 iterations, 5 warmup, TimedEvent-measured, all times ms)

| Shape | weight# | reduce# | current path | current | per-thread (forced) | blockreduce 64/128/256 | warpreduce 2/4/8 warps | best cooperative vs. per-thread |
|---|---|---|---|---|---|---|---|---|
| mnist_conv1 | 72 | 50,176 | block-reduce | 2.19 | 8.25 | 2.47 / 2.20 / 2.11 | 2.08 / 2.08 / 2.12 | **3.97x faster** |
| mnist_conv2 | 1,152 | 10,816 | per-thread | 1.78 | 1.78 | 7.39 / 7.45 / 7.56 | 7.34 / 7.29 / 7.23 | 0.25x (4.0x slower) |
| large_channel | 4,608 | 50,176 | per-thread | 43.92 | 43.89 | 137.66 / 137.72 / 138.01 | 135.69 / 135.64 / 135.68 | 0.32x (3.1x slower) |
| large_spatial | 1,152 | 100,352 | per-thread | 21.45 | 21.31 | 69.62 / 69.23 / 69.24 | 68.72 / 68.41 / 68.54 | 0.31x (3.2x slower) |
| batch_32 | 4,608 | 25,088 | per-thread | 21.85 | 21.99 | 69.76 / 69.92 / 70.28 | 68.40 / 68.28 / 68.01 | 0.32x (3.1x slower) |
| batch_64 | 4,608 | 50,176 | per-thread | 43.63 | 42.61 | 139.14 / 139.17 / 139.25 | 136.37 / 136.38 / 135.83 | 0.31x (3.2x slower) |
| batch_128 | 4,608 | 100,352 | per-thread | 85.86 | 86.15 | 277.74 / 277.64 / 277.36 | 272.74 / 272.48 / 271.50 | 0.32x (3.1x slower) |

### Interpretation
Below the M21 threshold, cooperative (block) reduction wins decisively --
expected, and unchanged from M21/M32. **Above** it, both cooperative designs
lose by a consistent 3-4x at *every* shape and *every* granularity tested,
and that ratio is essentially flat across `threads_per_block` (64/128/256)
and `warps_per_block` (2/4/8) alike -- e.g. at `large_channel`, block-reduce
ranges only 137.66-138.01ms across all three block sizes, and warp-reduce
135.64-135.69ms across all three warps-per-block settings, despite an 8x
difference in launched block count between the two designs at matched
`threads_per_block`. That flatness is the key finding: **the cost is not
sensitive to intra-group reduction overhead (tree depth, shuffle count) or
to total block count** -- ruling out both "too much `__syncthreads()`
overhead" and "too many block launches" as the explanation. The remaining,
best-supported explanation is that the per-thread kernel's advantage is
memory-level parallelism: at 1,152-4,608 independent, cheap, well-cached
threads, it already keeps far more independent memory requests in flight
than either cooperative design (which concentrates the same total work
behind many fewer independent streams, each now also paying a reduction
primitive the per-thread kernel never needed).

## Optimization Selected
**None.** Per the milestone's explicit stop condition ("if cooperative
reduction does not provide a meaningful end-to-end improvement across
representative workloads, do not force it into Forge... a valid outcome is
Profile -> Cooperative reduction tested -> No sufficient benefit -> Reject
-> Document evidence"), both candidates are rejected. `CUDABackend.
conv2d_backward`, `cf_conv2d_backward_weight_*`'s dispatch threshold, and
both production kernels are byte-for-byte unchanged from Milestone 21/32.

### Root cause (of why cooperative reduction does *not* help here)
`dWeight`'s per-thread kernel, at the weight-element counts Forge's own CNN
shapes reach (>= 1,152), is not under-parallelized in the way M21's
`mnist_conv1` case (72 threads) was -- it already launches enough
independent, low-register threads to saturate the GPU's memory pipeline.
Cooperative reduction is the right fix only when there are *too few*
independent units of work to hide memory latency; past that point, it adds
synchronization cost while reducing the number of independent memory
streams, a net loss confirmed here at every non-tiny shape.

## Optimization Implementation
Not adopted into production. The two candidate kernels
(`k_conv2d_backward_weight_reduce`, already production code below the
threshold; the new `k_conv2d_backward_weight_warp`) and their forced-
dispatch exports (`cf_conv2d_backward_weight_{perthread,blockreduce,
warpreduce}_*`) remain in `kernels.cu`, documented as profiling-only (never
called by `CUDABackend`), so this negative result is reproducible via
`python -m benchmarks.conv2d_backward_weight_profile` rather than only
asserted here.

## Block Configuration
Candidate A tested 64/128/256 threads/block (one weight/block).
Candidate B tested 2/4/8 warps/block, i.e. 64/128/256 threads/block with
2/4/8 weight elements/block. All three values of each parameter produced
results within a few percent of each other at a given shape -- block size
was never the deciding factor.

## Reduction Strategy
Candidate A: shared-memory tree reduction (identical structure to `k_sum`),
one block per weight. Candidate B: warp-shuffle reduction
(`__shfl_down_sync`, no shared memory, no explicit barrier -- warp-uniform
control flow throughout, verified safe: `widx` and the reduction loop's
trip count depend only on `blockIdx.x`/`threadIdx.x/32`, identical across
every lane of a warp, so the full-warp mask `0xFFFFFFFFu` genuinely applies
at the final shuffle).

## Shared Memory Usage
Candidate A: `threads_per_block * sizeof(T)` bytes/block (256 threads x 4
bytes = 1KB at the production block size). Candidate B: zero -- the entire
point of the warp-shuffle design.

## Register Usage
Not the limiting factor for either candidate (see **Parallelization
Analysis**/occupancy discussion above) -- the measured slowdown is
insensitive to `threads_per_block`/`warps_per_block`, which would move
register pressure per block in opposite directions if that were the binding
constraint.

## Correctness Validation
`tests/test_cuda_conv2d_backward_weight_cooperative.py` (new, 22 tests):
`cf_conv2d_backward_weight_{perthread,blockreduce,warpreduce}_*` each
compared directly against `CPUBackend`'s weight gradient (via a real
`Conv2d` layer's `.sum().backward()`) across three shapes (54, 1,152, and
4,608 weight elements; the last two also stride=2), the full
`threads_per_block`/`warps_per_block` sweeps, and a 20-iteration
repeated-use memory-safety check (`gc.disable()`-wrapped, matching
`forge_hardware_quirks`' documented convention for allocator-adjacent CUDA
tests). All pass at `rtol=atol=1e-4`.

## Finite Difference Results
Not re-added -- weight/bias finite-difference coverage for the *production*
dispatch already exists (`tests/test_cuda_conv2d_backward_optimization.py`,
M32) and is unaffected since production code did not change. The new
candidates are validated by direct analytic comparison against `CPUBackend`
instead (see **Correctness Validation**), the more direct check for code
that computes the same closed-form reduction as an existing, already
finite-difference-verified kernel.

## Cross-Stream Results
Not applicable to the new profiling-only kernels (called directly via
`ctypes` on the default stream in every test/benchmark here, never through
`CUDABackend._stream_guard`). The existing cross-stream tests for
`conv2d_backward` (M32's `test_cuda_conv2d_backward_correct_when_*`) are
unaffected and still pass, since the production dispatch path they exercise
is unchanged.

## Async Results
Same reasoning: `test_cuda_conv2d_backward_on_explicit_stream_matches_cpu`
(M32, unchanged) still passes; the new kernels are never reached through
`Tensor`/`Conv2d`'s async path.

## Memory Results
`test_cooperative_candidate_kernels_repeated_use_does_not_grow_active_memory`:
20 repeated iterations of all three new kernels, `allocated_bytes`/
`reserved_bytes`/`pending_bytes` all return to baseline. (One real bug was
caught and fixed while writing this test: an earlier draft released the
candidate kernels' output buffer via a raw `cf_free` instead of through
`CUDAStorage`/the M25 caching allocator's own release path, which silently
corrupted `allocated_bytes` bookkeeping for *later*, unrelated tests in the
same pytest session -- e.g. `test_cuda_cross_entropy_fusion.py`'s memory
test started failing until this was fixed. Fixed by wrapping the profiling
kernels' output pointer in a `CUDAStorage` immediately, exactly like every
other CUDA buffer in Forge.)

## Before / After dWeight Results
No change (rejected): see the **Results** table above under **Cooperative
Strategy Evaluated** -- "current" is both the before and after value at
every shape, confirmed by re-running `conv2d_backward_profile.py`
(`m33_conv2d_backward_profile_after.json`) and finding all `dWeight` numbers
within run-to-run noise of the M32 baseline (e.g. `large_channel`: 44.27ms
-> 43.73ms; `mnist_conv1`: 2.17ms -> 2.18ms).

## Before / After Conv2d Backward Results
No change (rejected optimization, unchanged kernels) -- `dInput`/`dBias`
also confirmed unchanged in the same re-run.

## MNIST Results
`benchmarks.mnist_bench` (via `python -m benchmarks --categories mnist`),
CUDA, batch=64: 15.21ms (M32's three-run mean was 17.09ms, stdev
1.08-5.75ms per run -- within that noise band, consistent with "no
production change").

## Async Prefetch Results
`benchmarks.async_dataloader_bench`, real MNIST-shaped workload (n=1024,
batch_size=64): synchronous 269.6ms/epoch, prefetch 228.8ms/epoch, 1.178x
speedup -- within the 1.07-1.21x band every milestone since M29 has
measured (M32: 1.212x). CUDA active bytes and pinned active bytes both
returned to 0 after the run.

## Updated Pipeline Profile
`benchmarks.pipeline_profile`, batch_size=64, prefetch_size=2: `fwd=2.05ms,
loss=0.63ms, bwd=8.49ms, opt=0.075ms`, compute-stream utilization 89.4-89.9%,
epoch throughput ~5,068-5,083 samples/sec -- matching M32's post-fix numbers
(`bwd=8.48ms`, 5,031 samples/sec) within noise, as expected since nothing
in the measured path changed.

## New Bottleneck Ranking
Unchanged from M32's post-fix ranking: `dWeight`/`dInput` remain
near-tied-to-dominant across `conv2d_backward` at every shape (`dWeight`
dominant at 6/7, `dInput` marginally ahead only at `mnist_conv2`);
`conv2d_backward` remains ~75% of total CUDA training-step compute time.
No further Conv2d-backward micro-optimization is currently justified by
measurement at this kernel-restructuring level of investigation -- see
**Limitations / Future opportunities** below for what a real next step
would require.

## CPU Regression
`CPUBackend.conv2d_backward` untouched (this milestone touched only new,
CUDA-only, profiling-only kernel code). All CPU tests in the full suite
pass unchanged; CPU benchmark numbers (`python -m benchmarks --categories
forward backward`) unchanged within normal noise.

## CUDA Regression
`forward`/`backward`/`training`/`mnist` benchmark categories re-run
together: every non-`dWeight`-adjacent CUDA operation (elementwise, matmul,
`max_pool2d`, `cross_entropy_loss`, dropout, mse_loss, multilayer
Linear/ReLU/Linear, Adam) within normal run-to-run noise of its historical
numbers; `conv2d`/`dWeight`/`mnist_cnn_full` numbers themselves also
unchanged, as expected.

## Stress Testing / Leak Testing
`test_cooperative_candidate_kernels_repeated_use_does_not_grow_active_memory`
(20 iterations x 3 kernel variants = 60 launches) plus the pre-existing
`test_cuda_conv2d_backward_repeated_use_does_not_grow_active_memory` (M32,
production path, unaffected) -- both pass.

## Full Test Suite
1,184 tests total (1,162 pre-existing, unmodified + 22 new in
`tests/test_cuda_conv2d_backward_weight_cooperative.py`), all passing on
the 940MX (`python -m pytest tests/ -q`).

## Hardware Verification
Every measurement in this section was collected on the real, verified
development GPU (NVIDIA GeForce 940MX, CC 5.0, CUDA 12.6, driver 582.53),
including a clean rebuild of the CUDA kernel library from source
immediately before the final verification pass -- no simulated or emulated
CUDA behavior anywhere in this milestone.

## Limitations / Future opportunities (Milestone 33)
- **Per-thread `dWeight` remains the largest `conv2d_backward` cost** at
  most shapes, and this milestone found no cooperative-reduction fix at the
  kernel-restructuring level investigated here. A structurally different
  approach -- e.g. im2col + GEMM (reusing the existing tiled `k_matmul`),
  or a shared-memory tiling scheme that increases `x`/`grad_out` reuse
  across *both* `dWeight` and `dInput` simultaneously -- is a materially
  larger rewrite than this milestone's scope and was not attempted.
- **No profiler-verified occupancy numbers** (e.g. `nvcc -Xptxas -v`
  register counts per candidate, or Nsight Compute occupancy/memory-
  throughput counters) were collected for the two rejected candidates --
  the wall-clock/CUDA-event evidence above was decisive enough (consistent
  3-4x gap, flat across every tested block-size/warps-per-block value) that
  deeper profiling tooling was not required to reach the reject decision,
  but a future investigation with such tooling could give a more precise
  mechanistic explanation than this section's reasoned inference.
- Two cooperative granularities were tested (32 and 64-256 threads per
  weight element); sub-warp cooperation (e.g. 2-16 threads via shared
  memory) was not, since the flat trend across the tested range made a
  reversal at a smaller granularity implausible -- not measured, so not
  claimed.
