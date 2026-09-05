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

# Milestone 34: dWeight im2col + existing tiled GEMM (accepted)

## Purpose
M33 rejected cooperative reduction and named a structurally different
approach -- im2col + GEMM, reusing Forge's existing M11 shared-memory-tiled
`k_matmul` -- as the next `dWeight` candidate worth measuring. This
milestone builds that experimental path, benchmarks it against the existing
per-thread `dWeight` kernel at the same 7 representative shapes (isolated
`dWeight` alone, and the complete `conv2d_backward`), and, given a clear and
reproducible end-to-end improvement at 6 of 7 shapes with acceptable memory
overhead, integrates it into production as a minimal shape-based dispatch.

## How to reproduce
```bash
python -m benchmarks.conv2d_backward_weight_im2col_profile     # isolated dWeight: current vs. im2col/permute/GEMM phases
python -m benchmarks.conv2d_backward_im2col_pipeline_profile   # full conv2d_backward: production vs. experimental-dWeight, + memory
python -m benchmarks.mnist_profile
python -m benchmarks.pipeline_profile
python -m benchmarks.async_dataloader_bench
python -m benchmarks --categories mnist backward forward training
```
Archived results: `benchmarks/results/m34_dweight_im2col_gemm.json` (isolated
dWeight phases -- the primary GEMM/im2col evidence), `m34_pipeline_profile.json`
(full `conv2d_backward` comparison + allocator/peak-memory accounting -- the
primary production-decision evidence), plus re-runs of `mnist_profile.json`
and `pipeline_profile.json` (overwritten in place, matching M32/M33's own
convention for these two particular tools).

## Existing direct dWeight architecture (unchanged, still used below the threshold)
`cf_conv2d_backward_weight_*` (`kernels.cu`) still dispatches, purely by
weight-element count (`CONV2D_WEIGHT_REDUCE_THRESHOLD = 256`), between
`k_conv2d_backward_weight` (one thread per `(co, ci, kh, kw)` weight element,
full serial `N*Hout*Wout` reduction) and `k_conv2d_backward_weight_reduce`
(one block per weight element, shared-memory tree reduction) -- the exact
M21 kernels, byte-for-byte unchanged by this milestone.

## Existing GEMM architecture
`k_matmul` (`kernels.cu`, Milestone 11): a standard 16x16-tile shared-memory
GEMM, `C[M,N] = A[M,K] @ B[K,N]`, row-major, zero-padded on out-of-range tile
loads (correct for any M/K/N, not just multiples of 16). No warp intrinsics,
no architecture-specific tricks -- portable to CC 5.0. `cf_matmul_{f32,f64}`
is called exactly like every other Forge matmul use site (`CUDABackend.
matmul`); this milestone adds **zero** changes to `k_matmul` or its launcher,
per the milestone brief's explicit "existing GEMM must remain untouched"
rule -- the only question asked is whether this *unmodified* GEMM, fed the
right operands, beats the direct kernel.

## Mathematical reformulation
Verified against `CPUBackend.conv2d_backward` (`forge/backend/cpu.py`,
which already computes this exact GEMM via NumPy/BLAS on CPU):

```
grad_weight_mat = grad_out_rows.T @ cols_rows     # (Cout, M) @ (M, K) -> (Cout, K)
```

where `M = N*Hout*Wout` (the reduction dimension) and `K = Cin*KH*KW`,
reshaped to `(Cout, Cin, KH, KW)` at the end. **This is not the `Xcol^T @
dYmat -> (K, Cout)` orientation a literal reading of a naive im2col write-up
would suggest** -- that produces the transpose of what Forge's weight
layout needs (the milestone brief's Section 5 explicitly flagged this as a
"verify, do not assume" risk, and the verification did in fact catch a
transposed orientation). The implementation (`forge/backend/cuda/
experimental_conv_im2col.py`) instead builds:

- `Xcol`, shape `(M, K)` -- via new kernel `k_im2col_conv2d`, one thread per
  `(m, k)` output element, identical padding/stride/boundary handling to
  `k_conv2d_forward`.
- `dYcolT`, shape `(Cout, M)` -- via new kernel `k_conv2d_grad_output_permute`,
  one thread per `(co, m)` output element. This is **not** a 2D transpose of
  `grad_output`: its actual memory layout is `(N, Cout, Hout, Wout)` (`N`
  outermost), so producing `(Cout, M)` (`Cout` outermost) is a genuine
  4D-to-2D gather, not a reshape -- this is the "reshape/copy" cost the
  milestone brief's Sections 8/17/20 ask to track separately.

then one `cf_matmul(A=dYcolT, B=Xcol, GEMM_M=Cout, GEMM_K=M, GEMM_N=K)` call,
producing `(Cout, K)` directly in `grad_weight`'s own contiguous layout --
the final reshape to `(Cout, Cin, KH, KW)` costs nothing (no copy).

## Workloads
The same 7 M32/M33 shapes, unmodified (`benchmarks/conv2d_backward_profile.py`'s
`SHAPES`/`BATCH_SWEEP_BASE`/`BATCH_SIZES`, imported directly).

## GEMM dimensions
| Shape | GEMM A (Cout x M) | GEMM B (M x K) | GEMM C (Cout x K) | M | K |
|---|---|---|---|---|---|
| mnist_conv1 | 8 x 50,176 | 50,176 x 9 | 8 x 9 | 50,176 | 9 |
| mnist_conv2 | 16 x 10,816 | 10,816 x 72 | 16 x 72 | 10,816 | 72 |
| large_channel | 32 x 50,176 | 50,176 x 144 | 32 x 144 | 50,176 | 144 |
| large_spatial | 16 x 100,352 | 100,352 x 72 | 16 x 72 | 100,352 | 72 |
| batch_32 | 32 x 25,088 | 25,088 x 144 | 32 x 144 | 25,088 | 144 |
| batch_64 | 32 x 50,176 | 50,176 x 144 | 32 x 144 | 50,176 | 144 |
| batch_128 | 32 x 100,352 | 100,352 x 144 | 32 x 144 | 100,352 | 144 |

`GEMM_M` (`Cout`, 8-32) is tiny relative to the 16x16 tile everywhere -- 1-2
tile rows total -- the central fact behind the **GEMM suitability** analysis
below.

## Baseline dWeight results
(`current`, the existing production dispatch, mean of 30 iterations, 5
warmup, TimedEvent-measured, 940MX): mnist_conv1 2.18ms, mnist_conv2 1.74ms,
large_channel 43.50ms, large_spatial 21.47ms, batch_32 21.73ms, batch_64
43.44ms, batch_128 86.74ms -- consistent with the M32/M33 baseline within
normal run-to-run noise.

## Im2col results
| Shape | im2col (ms) | Xcol size (MB) |
|---|---|---|
| mnist_conv1 | 0.47 | 1.72 |
| mnist_conv2 | 0.79 | 2.97 |
| large_channel | 7.35 | 27.56 |
| large_spatial | 7.29 | 27.56 |
| batch_32 | 3.62 | 13.78 |
| batch_64 | 7.35 | 27.56 |
| batch_128 | 14.95 | 55.12 |

Plus the `grad_output` permute (`dYcolT`): 0.12-2.10ms across shapes,
consistently the cheapest of the three phases (`Cout*M` elements moved --
smaller than `Xcol`'s `M*K` by roughly `K/Cout`).

## GEMM results
| Shape | gemm (ms) |
|---|---|
| mnist_conv1 | 3.04 |
| mnist_conv2 | 0.74 |
| large_channel | 4.95 |
| large_spatial | 6.60 |
| batch_32 | 2.48 |
| batch_64 | 4.97 |
| batch_128 | 10.02 |

Note `large_spatial` (6.60ms) vs. `batch_32` (2.48ms): both have *identical*
total multiply-accumulate work (`weight_elements * M`: 1,152 x 100,352 =
4,608 x 25,088 = 115,605,504), yet GEMM takes 2.7x longer for `large_spatial`
-- see **GEMM suitability** below for why.

## Total experimental dWeight results
`total = im2col + permute + gemm` (Section 8: never compared as GEMM time
alone):

| Shape | current (ms) | total experimental (ms) | ratio (exp/current) |
|---|---|---|---|
| mnist_conv1 | 2.18 | 3.79 | 1.74x (slower) |
| mnist_conv2 | 1.74 | 1.64 | 0.94x |
| large_channel | 43.50 | 13.35 | 0.31x |
| large_spatial | 21.47 | 14.92 | 0.70x |
| batch_32 | 21.73 | 6.61 | 0.30x |
| batch_64 | 43.44 | 13.37 | 0.31x |
| batch_128 | 86.74 | 27.07 | 0.31x |

Every shape at/above `CONV2D_WEIGHT_REDUCE_THRESHOLD` (256 weight elements)
is faster with im2col+GEMM -- 1.4x (`mnist_conv2`) to 3.3x (`batch_32`/
`large_channel`/`batch_64`/`batch_128`). Only `mnist_conv1` (72 elements,
already on the block-reduce path per M21/M33) is slower.

## Memory traffic analysis
Direct kernel: each of `weight_elements` threads independently re-reads
its own `~M`-element slice of `x`/`grad_out` (bounded by padding), no reuse
across threads beyond incidental cache locality -- total global reads scale
as `weight_elements * M` element-reads (matching its multiply-accumulate
count 1:1; effectively memory-bandwidth-bound at large `M`). Im2col+GEMM
instead pays: (1) `im2col` -- `M*K` reads of `x` (with the same overlap-driven
re-read pattern any im2col has) and `M*K` writes to `Xcol`; (2) `permute` --
`Cout*M` reads of `grad_out` and `Cout*M` writes to `dYcolT`; (3) `GEMM` --
tiled reads of `A`/`B` reduced by roughly the 16x tile-reuse factor versus a
naive read-per-output-element GEMM, `Cout*K` writes. The extra traffic is
real (materializing `Xcol`/`dYcolT` before the GEMM even starts touches
every input element an extra time), but the GEMM's shared-memory tile reuse
more than pays for it at every tested shape except `mnist_conv1`, where
`K=9` is so far below the 16-wide tile that the GEMM gets almost no reuse
benefit at all (see next section).

## Peak memory analysis
(`benchmarks/conv2d_backward_im2col_pipeline_profile.py`, full
`conv2d_backward`, peak *reserved* bytes across 20 fresh-allocate-then-release
iterations):

| Shape | production peak (MB) | experimental peak (MB) | extra (MB) |
|---|---|---|---|
| mnist_conv1 | 5.36 | 8.61 | 3.25 |
| mnist_conv2 | 8.74 | 12.37 | 3.63 |
| large_channel | 56.46 | 90.15 | 33.69 |
| large_spatial | 93.19 | 126.88 | 33.69 |
| batch_32 | 105.49 | 122.33 | 16.84 |
| batch_64 | 148.38 | 182.07 | 33.69 |
| batch_128 | 234.15 | 301.52 | 67.38 |

Largest absolute overhead (`batch_128`, 67.38MB) is ~3.3% of the 940MX's 2GB
budget -- not a large-memory-multiplier concern (Section 18's explicit
"small speedup, large memory multiplier" failure mode does not apply here;
speedups are large, not small, and the multiplier itself is modest).

## Allocator analysis
Both `Xcol`/`dYcolT` go through the ordinary M25 caching allocator
(`self._alloc`, `forge/backend/cuda/experimental_conv_im2col.py`) -- no raw
`cudaMalloc`/`cudaFree`. Steady-state behavior (20 repeated calls, temporaries
released every iteration): production makes 3 allocation requests/call (60
total across 20 iterations), 3 cache misses (cold start) + 57 hits (95%);
experimental makes 5 requests/call (100 total), 5 misses + 95 hits (95%) --
identical steady-state hit rate to production, confirming the two new
temporary sizes are exact-size cache-friendly like every other Forge CUDA
buffer, with no allocator degradation from adding two more per-call
allocation sites.

## Kernel launch analysis
Direct dWeight: 1 launch. Experimental: 3 launches (im2col, permute, GEMM).
Full `conv2d_backward`: production 3 launches total (`dInput`, `dWeight`,
`dBias`); experimental-dWeight variant 5 (`dInput`, im2col, permute, GEMM,
`dBias`). Per-launch dispatch overhead (~54-65us, M31's
`stream_dependency_bench.py`) remains one to three orders of magnitude
smaller than any measured kernel time here (0.1-160ms) -- the extra 2
launches are not a factor at any tested shape, including the smallest.

## Synchronization analysis
`im2col`, `permute`, and the GEMM call are issued on the same Forge stream
(`backend._stream_handle()`) back-to-back with no `cudaDeviceSynchronize()`/
`cudaStreamSynchronize()` between them -- CUDA's ordinary stream program
order (same stream, in-order execution) is what keeps `im2col`'s writes to
`Xcol` visible to the GEMM's read of it, and likewise for `permute`'s writes
to `dYcolT`. `dweight_im2col_gemm` synchronizes exactly once, at the very
end (`_maybe_synchronize`, a no-op in async mode) -- identical to every
other `CUDABackend` method's contract.

## Stream analysis
`test_im2col_gemm_dweight_on_explicit_stream_matches_cpu` (explicit
`forge.cuda.Stream()`) and `test_im2col_gemm_dweight_correct_when_inputs_from_
different_streams` (x/weight/grad_output each produced on a distinct
producer stream, computed on a third) both pass -- `_require_compute_dtype`'s
existing `_stream_guard` chokepoint (unchanged since M28) already covers the
new pipeline's inputs; no new dependency-tracking code was needed.

## Bottleneck analysis
`mnist_conv1`'s regression is explained by `GEMM_N = K = 9` -- less than one
16-wide tile, so `k_matmul` computes on a `16x16` tile that is >40% zero-padding
waste, while the existing block-reduce kernel (M21/M33-confirmed the right
choice below 256 weight elements) has no such inefficiency. `large_spatial`
vs. `batch_32`'s 2.7x GEMM-time gap despite identical FLOPs is explained by
grid size: `large_spatial`'s GEMM launches only `ceil(72/16) * ceil(16/16) =
5` thread blocks total (`GEMM_N=72` -> 5 tiles, `GEMM_M=16` -> 1 tile row) --
far too few to occupy the 940MX's 3 SMs -- while `batch_32`'s `ceil(144/16) *
ceil(32/16) = 18` blocks spread far better. **`k_matmul`'s tile efficiency
and grid size both depend on `Cout` and `Cin*KH*KW` specifically (small and
close to a tile boundary at MNIST-adjacent shapes), not on total FLOPs** --
the central finding of this milestone's GEMM-suitability analysis (Section
16 of the brief).

## Correctness validation
`tests/test_cuda_conv2d_backward_weight_im2col_gemm.py` (19 tests): direct
`dweight_im2col_gemm` vs. CPU across 6 shape/stride/padding/kernel-size
combinations (float32 and float64), explicit-stream, cross-stream, and
repeated-use memory safety; plus **production-dispatch** coverage through
the ordinary `Tensor.conv2d`/`nn.Conv2d` API at shapes that actually cross
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD` (since `tests/test_cuda_conv.py`'s
existing shapes never exceed ~144 weight elements and so never exercised the
new dispatch branch before this milestone). All pass at `rtol=atol=1e-4`
(float32) / `rtol=atol=1e-8` (float64).

## Finite difference results
`test_im2col_gemm_dweight_finite_difference` (3 stride/padding combinations,
float64) passes at `rtol=atol=1e-2`, matching M32's own weight-gradient FD
tolerance -- established mathematical correctness independent of the CPU
comparison.

## Cross-stream results
`test_im2col_gemm_dweight_correct_when_inputs_from_different_streams` passes:
`x`/`weight` produced on one stream, `grad_output` on another, computed on a
third -- `_stream_guard` handles all three producers correctly with zero new
dependency code.

## Async results
`test_im2col_gemm_dweight_on_explicit_stream_matches_cpu` and the new
production-dispatch `test_production_conv2d_backward_on_explicit_stream_
above_threshold` both pass: the full pipeline (and the real `Conv2d` layer
routed through it) on an explicit `forge.cuda.Stream()`, verified against CPU.

## Shape generalization
6 explicit shape/stride/padding/kernel-size combinations in the direct
correctness tests (no padding, padding, strided, an asymmetric batch size,
an even kernel size at an odd spatial size), plus the 7 M32/M33 representative
shapes in the benchmark suite, plus 3 shapes spanning the dispatch threshold
in the production-dispatch tests -- all pass.

## MNIST results
`benchmarks.mnist_profile`'s per-op CUDA walker: `conv2d` backward
6.90ms -- down from M32's post-fix 9.95ms (M33 left unchanged), a further
~1.44x reduction, since the real M20 CNN's second conv layer
(`mnist_conv2`-shaped, 1,152 weight elements) now crosses the dispatch
threshold. `benchmarks.mnist_bench` (`training` category, three runs):
23.47ms, 14.78ms, 14.90ms (mean 17.72ms) -- within the same 15-19ms
run-to-run noise band M32 (17.09ms mean) and M33 (15.21ms) reported; MNIST's
real layer shapes are small enough that the absolute `dWeight` savings here
are modest relative to this benchmark's own measurement noise, consistent
with M32's identical caveat for its own MNIST-scale numbers. The isolated
per-op walker above is the cleaner signal at this scale.

## Async prefetch results
`benchmarks.async_dataloader_bench`, real MNIST-shaped workload (n=1024,
batch_size=64): synchronous 243.70ms/epoch, prefetch 215.84ms/epoch,
**1.129x speedup** -- within the 1.07-1.21x band every milestone since M29
has measured. CUDA active bytes and pinned active bytes both returned to 0
after the run.

## Production decision
**Accepted, as a minimal shape-based hybrid dispatch.** Per Sections 28-30
and 39-40 of the milestone brief: reproducible improvement (measured via two
independent benchmark scripts, isolated-kernel and full-pipeline), a clear
representative-shape benefit (6 of 7 shapes, 1.12-1.59x end-to-end), modest
memory overhead (3-68MB, <3.3% of the 2GB budget), no regression at small
shapes (the one shape that regresses, `mnist_conv1`, stays on the existing
kernel via the threshold), correct stream/async/cross-stream behavior, and
healthy allocator behavior (95% cache hit rate, matching production) --
every criterion in Section 39 is satisfied.

## If accepted: production integration
`CUDABackend.conv2d_backward` (`forge/backend/cuda/backend.py`) now
dispatches `dWeight` on `weight_elements = Cout*Cin*KH*KW`:
`>= _CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD` (256, reusing -- not
re-deriving -- `kernels.cu`'s existing `CONV2D_WEIGHT_REDUCE_THRESHOLD`
boundary) routes to `forge.backend.cuda.experimental_conv_im2col.
dweight_im2col_gemm`; below it, the original `cf_conv2d_backward_weight_*`
kernel call is unchanged. `dInput`/`dBias` are completely untouched.
**Caveat**: no shape between 256 and 1,152 weight elements was tested, so
the exact crossover point is not finely known -- 256 was kept because it is
already the established production boundary and the smallest shape actually
tested above it (`mnist_conv2`, 1,152 elements) already shows a real (if
modest, 1.12x) end-to-end win, giving confidence the reused threshold is
conservative rather than optimistic.

## Before / after Conv2d results
Isolated `dWeight` (`current` vs. `total experimental`, this document's own
**Total experimental dWeight results** table above) -- the direct
before/after evidence.

## Before / after training results
`mnist_profile`'s per-op CUDA `conv2d` backward: 9.95ms (M32/M33 baseline) ->
6.90ms (this milestone), ~1.44x. `pipeline_profile`'s async `bwd` phase:
8.48-8.49ms (M32/M33) -> 8.32ms at batch=64 -- within noise (MNIST's real
layer shapes keep the *absolute* `dWeight` savings small relative to this
already-overlapped, multi-op pipeline measurement, matching this document's
own **MNIST results** caveat above).

## CPU regression results
`CPUBackend.conv2d_backward` untouched (CUDA-only milestone). All CPU
`Conv2d`/training tests pass unchanged; `python -m benchmarks --categories
forward backward training` CPU rows show no `conv2d`-adjacent movement
outside historical noise.

## CUDA regression results
Same benchmark categories re-run: every non-`dWeight`-adjacent CUDA
operation (elementwise, `matmul`, `max_pool2d`, `cross_entropy_loss`,
dropout, multilayer Linear/ReLU/Linear, Adam) within normal run-to-run
noise; `conv2d`/`mnist_cnn_full` moved, and only in the improving direction
(e.g. the `backward` category's "medium" CUDA `conv2d` shape, `Cout*Cin*K*K
= 4,608` weight elements, now dispatches through im2col+GEMM).

## Memory safety
`test_im2col_gemm_dweight_repeated_use_does_not_grow_active_memory` (20
iterations of the direct pipeline) and
`test_production_conv2d_backward_repeated_use_does_not_grow_active_memory_
above_threshold` (30 iterations through the real `Conv2d` layer, weight
elements above the threshold) both pass: `allocated_bytes`/`reserved_bytes`/
`pending_bytes` all return to baseline after `gc.collect()` + `empty_cache()`.

## Leak testing
Same two tests above, plus the pre-existing `test_cuda_conv2d_backward_
repeated_use_does_not_grow_active_memory` (M32, still exercises the
below-threshold path for its small test shape, unaffected) and
`test_cuda_conv2d_weight_reuse_accumulates_matching_cpu`-style gradient
accumulation, now also covered above-threshold by this milestone's
`test_production_conv2d_weight_reuse_accumulates_matching_cpu_above_
threshold`.

## Hardware verification
Every measurement in this section was collected on the real, verified
development GPU (NVIDIA GeForce 940MX, CC 5.0, CUDA 12.6, driver 582.53) --
no simulated or emulated CUDA behavior anywhere in this milestone.

## Limitations / future opportunities (Milestone 34)
- **No shape between 256 and 1,152 weight elements was tested** -- the
  reused 256-element threshold is a conservative choice (see **Production
  decision**'s caveat), not a freshly-fitted crossover point. A future
  milestone could narrow this with a finer shape sweep if it becomes
  relevant (e.g. a CNN architecture with many mid-sized conv layers).
- **`mnist_conv1`-shaped layers (very small `Cout`/`Cin*KH*KW`) get no
  benefit** from this milestone -- `k_matmul`'s 16x16 tiling is poorly
  suited to a GEMM with `Cout < 16` and `Cin*KH*KW` near or below 16. A
  differently-tiled or batched-small-GEMM strategy could help here, but is
  out of scope (Section 43 forbids a new GEMM implementation).
- **The `grad_output` permute (`k_conv2d_grad_output_permute`) was not
  optimized** beyond the simplest correct one-thread-per-element gather
  (Section 31's "measure before optimizing im2col" applies equally here);
  it was consistently the cheapest of the three phases at every shape
  measured, so no optimization was justified.
- **`Xcol` materialization roughly doubles total `x`-adjacent memory traffic**
  relative to the direct kernel at large shapes -- acceptable given the
  measured speedup and modest absolute peak-memory overhead here, but would
  compound unfavorably on a GPU with a tighter VRAM budget than the 940MX's
  2GB, or at much larger `Cin*KH*KW`.
- This milestone did not attempt an im2col+GEMM formulation for `dInput`
  (explicitly out of scope, Section 6) -- `dInput` remains the M32-optimized
  direct kernel, still the larger `conv2d_backward` cost at most shapes even
  after this milestone's `dWeight` improvement.

# Milestone 36: dInput channel-fused work mapping (accepted)

## Purpose
M35's roofline characterization measured `conv2d` backward as 50.97% of a
real CUDA training step and `dInput` reaching only ~12% of the 940MX's
practical compute ceiling (104.57 GFLOP/s) while using well under 20% (often
under 5%) of its practical bandwidth ceiling (15.09 GB/s) at every
representative shape -- despite an arithmetic intensity (4-48 FLOPs/byte)
that places most shapes decisively compute-bound by the roofline model
(ridge point ~6.93 FLOPs/byte). This milestone profiles `dInput`'s actual
instruction/memory behavior (not just its wall-clock time), finds the real
cause, measures three structurally different candidates against it, and
integrates the one that wins decisively and consistently.

## How to reproduce
```bash
python -m benchmarks.conv2d_backward_dinput_profile   # isolated dInput: current vs. A/B/C candidates
python -m benchmarks.conv2d_backward_profile          # full conv2d_backward, before/after (empty_cache between shapes -- see Limitations)
python -m benchmarks.m35_mnist                         # kernel-contribution ranking + batch-size scaling, rerun
python -m benchmarks.pipeline_profile                  # async prefetch-depth sweep, rerun
python -m pytest tests/test_cuda_conv2d_dinput_optimization.py
```
Archived results: `benchmarks/results/m36_dinput_candidates_profile.json`
(+ `_rerun`/`_run3`, three independent passes -- see **Hardware variance**
below), `m36_dinput_baseline.json` / `m36_dinput_optimized.json` (roofline-
classified dInput-only before/after, the primary optimization evidence),
`m36_conv2d_backward_after.json` (full backward after, paired with the
pre-existing `conv2d_backward_profile.json` as before), and `m36_mnist.json`.

## Root-cause analysis (Sections 14-17 of the milestone brief)
`nvcc -Xptxas -v` on the unmodified M32 `k_conv2d_backward_input` reports:
```
ptxas info : Function properties for k_conv2d_backward_input
    512 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 48 registers, ...
```
The `kh_valid`/`ho_valid`/`kw_valid`/`wo_valid` local tables M32 introduced
(4 arrays x `MAX_CONV_K=32` `int`s = 512 bytes) are indexed with a
runtime-computed loop variable (`hi_i`/`wi_i`) -- `nvcc` cannot keep such an
array in registers regardless of its size, so it places it in per-thread
**local memory** (an implicit, DRAM-backed, L1/L2-cached region distinct from
registers or shared memory), read back `Cout * h_count * w_count` times per
thread. This traffic is real but invisible to `benchmarks/roofline.py`'s
`bytes_conv2d_dinput` model, which only counts the kernel's logical
grad_output/weight/grad_x operands, never implementation-internal spill/local
traffic -- exactly why a kernel classified compute-bound by arithmetic
intensity could still sit at only ~12% of the practical *compute* ceiling
while using a sliver of the *bandwidth* budget: it was spending real cycles
on something neither ceiling's FLOP/byte accounting could see. Per Section
14/15's explicit guidance ("if a kernel is far below both ceilings,
investigate instruction efficiency/occupancy/latency rather than labeling it
memory-bound"), this ruled out a pure memory-bandwidth-reuse fix as the
primary lever before any candidate was even benchmarked.

## grad_output read amplification (Section 8)
For a fixed `(n, h, w)`, `grad_output[n, co, ho, wo]` is identical across
every `ci` (the formula in Section 4 of the milestone brief sums over `co`,
`kh`, `kw` -- never `ci`-dependent for a fixed output position). Under the
baseline's one-thread-per-`(n,ci,h,w)` mapping, `Cin` separate threads each
independently re-read the same `grad_output` values from global memory --
`Cin`-fold amplification (1, 8, or 16x at Forge's 7 representative shapes).

## Candidate algorithms (Sections 9-11)
Three structurally different kernels were added to `kernels.cu` (profiling-
only, `cf_conv2d_backward_input_{smem,channelfused,warpreduce}_*`) and
measured against the unchanged production kernel via
`benchmarks/conv2d_backward_dinput_profile.py`:

- **Candidate A (`k_conv2d_backward_input_smem`)** -- shared-memory
  grad_output row-tile reuse across `Cin`: a block owns one `(n, hi)` pair,
  cooperatively loads the `h_count <= KH` needed grad_output rows (all
  `Cout` channels, full `Wout`) into shared memory once, then every
  `(ci, wi)` thread in the block reads from shared memory instead of
  re-issuing `Cin` separate global loads. Tests the read-amplification
  hypothesis directly, keeping the baseline's thread mapping otherwise
  unchanged. Tested at 64/128/256 threads/block.
- **Candidate B (`k_conv2d_backward_input_channelfused`)** -- alternative
  work mapping: one thread now owns a full `(n, hi, wi)` position across
  *every* `ci`, holding `Cin` accumulators in a `#pragma unroll`-forced,
  compile-time-bounded (`MAX_CIN_REG=16`) register array, and reads each
  `grad_output[n,co,ho,wo]` value exactly once per thread, reused via a
  register across every `ci` multiply. The `kh`/`ho`/`kw`/`wo` validity
  resolution also moves to plain scalars in the outer loop nest (never
  stored to an array), eliminating M32's local-memory tables entirely.
  Requires `Cin <= MAX_CIN_REG` for correctness (see **Kernel design**).
- **Candidate C (`k_conv2d_backward_input_warp`)** -- partial cooperative
  reduction over `Cout`: a warp (32 lanes), not a single thread, owns one
  `(n,ci,h,w)` output element; each lane sums a disjoint slice of `Cout`,
  combined via `__shfl_down_sync` (mirrors M33's `k_conv2d_backward_weight_
  warp`). Tested at 2/4/8 warps/block. A priori expectation: rejection, since
  `dInput` already launches `N*Cin*H*W` threads (tens of thousands to
  ~800K at these shapes) -- far more than the 940MX's ~6,144-concurrent-
  thread capacity already uses in one wave, so there is no starved
  parallelism for this candidate to fix.

## Candidate benchmark results
Isolated dInput-only timings (940MX, real CUDA), `current` = unmodified
production kernel (this run, before the M36 dispatch change), best
block/warp configuration per candidate:

| Shape | current (ms) | Candidate A: smem (ms) | Candidate B: channelfused (ms) | Candidate C: warp (ms) | Winner |
|---|---|---|---|---|---|
| mnist_conv1 | 0.96 | 2.29 (0.42x) | 0.78-0.94 (~1.0-1.2x) | 5.16 (0.19x) | B |
| mnist_conv2 | 1.94-4.98 | 2.73-2.78 (0.7-1.8x) | 0.87-1.16 (1.7-5.7x) | 9.11-9.22 (0.2-0.5x) | B |
| large_channel | 36.9-114.4 | 42.1-120.2 (~0.9-1.0x) | 12.9-36.4 (2.5-8.7x) | 157.5-419.5 (0.2-0.3x) | B |
| large_spatial | 18.9-47.0 | 23.8-66.8 (~0.7-0.8x) | 8.3-21.1 (2.2-4.3x) | 92.5-235.3 (0.2x) | B |
| batch_32 | 18.4-51.9 | 21.4-60.2 (~0.9x) | 7.1-19.2 (2.4-7.3x) | 104.3-209.4 (0.2x) | B |
| batch_64 | 37.0-92.3 | 42.2-120.2 (~0.6-1.0x) | 14.4-36.4 (2.5-5.5x) | 190.6-419.8 (0.2x) | B |
| batch_128 | 72.4-203.2 | 84.2-240.0 (~0.6-0.8x) | 25.5-71.3 (2.6-2.9x) | 368.9-840.2 (0.2x) | B |

(Ranges span three independent runs -- see **Hardware variance** below.)
Candidate B (channel-fused) wins at every shape in every run. Candidate A
(shared-memory tiling) never beats production, confirming the bandwidth
analysis: since `dInput` was never bandwidth-starved (2-17% of the practical
ceiling), trading global reads for shared-memory reads plus `__syncthreads()`
plus much higher register pressure (126-128 registers/thread and a residual
256-byte stack frame -- see **Register usage**) is a net loss. Candidate C
(warp-cooperative) loses catastrophically (3-20x slower) at every shape,
confirming the already-sufficient-parallelism hypothesis.

## Register usage (`nvcc -Xptxas -v`, `sm_50`)
| Kernel | Registers (f32) | Stack frame | Notes |
|---|---|---|---|
| `k_conv2d_backward_input` (baseline) | 48 | **512 bytes** | M32's dynamically-indexed local tables |
| Candidate A: `_smem` | 126-128 | 256 bytes | halves the local-memory tables (h-side now in shared memory) but adds heavy register pressure from the per-thread `kw_valid`/`wo_valid` tables plus tile bookkeeping |
| **Candidate B: `_channelfused`** | **54** | **0 bytes** | `Cin`-sized accumulator array fully promoted to registers (compile-time-unrolled, guarded by `if (ci >= Cin) break`) |
| Candidate C: `_warp` | 53 | 0 bytes | no local memory, but 32x more threads launched for the same work |

Candidate B is the only one that both removes the local-memory traffic *and*
keeps register pressure close to baseline (54 vs. 48 registers, f32) --
consistent with it being the clear winner on measured occupancy/latency
grounds, not just reduced global-memory traffic.

## Selected strategy: Candidate B (channel-fused work mapping)
Chosen per Section 13's `runtime contribution x memory reuse potential x
complexity` framework: largest and most consistent measured speedup (1.0x-
8.7x across three independent runs, never a severe regression), lowest
implementation complexity (no shared memory, no synchronization, no atomics),
and a root cause (local-memory traffic + register-bound instruction
inefficiency) that this design addresses directly rather than by accident.

## Kernel design
`k_conv2d_backward_input_channelfused` (`kernels.cu`): one thread per
`(n, hi, wi)` (dropping `ci` from the thread index entirely). Per thread:
1. `Cin` accumulators (`T acc[MAX_CIN_REG]`, `MAX_CIN_REG=16`) initialized via
   a `#pragma unroll` loop with a compile-time-constant trip count -- this,
   not the array's size, is what lets `nvcc` promote it to registers.
2. Outer loops over `kh`, `kw` resolve `(ho, wo)` validity as plain scalars
   (no array store) -- the exact M32 optimization (avoid recomputing
   `co`-independent division/modulo `Cout` times), just without the
   local-memory table that undermined it.
3. Inner loop over `co` reads `grad_output[n,co,ho,wo]` into a register `g`
   once, then a `#pragma unroll`-forced loop over `ci in [0, MAX_CIN_REG)`
   (guarded by `if (ci >= Cin) break`, a lane-uniform predicate since `Cin`
   is a shape parameter, not data -- no warp divergence) multiplies `g` by
   `w[co,ci,kh,kw]` into `acc[ci]`.
4. All `Cin` accumulators are written to `grad_x[n,ci,hi,wi]` at the end.

**Production dispatch**: `cf_conv2d_backward_input_*` (`kernels.cu`) now
checks `Cin <= CONV2D_DINPUT_CHANNELFUSED_MAX_CIN` (16) and calls the
channel-fused kernel when true, falling back to the unchanged, always-correct
`k_conv2d_backward_input` otherwise (Section 38's sanctioned hybrid
dispatch). Every one of Forge's 7 representative shapes has `Cin <= 16`, so
every one of them takes the new path in production. A layer with `Cin > 16`
(not produced by any existing Forge test or example, but not prohibited by
`Conv2d`'s public API either) transparently falls back to the baseline
kernel -- verified directly by `tests/test_cuda_conv2d_dinput_optimization.py
::test_production_dispatch_correct_at_cin_boundary` (parametrized at
`Cin=16` and `Cin=17`, both checked against the CPU backend).

## Block configuration
Candidate B uses the same `launch_config` (256 threads/block, 1D grid) every
other simple elementwise/reduction kernel in `kernels.cu` uses -- no new
block-size tuning was needed or performed (Section 20's 64/128/256 sweep was
run for Candidates A and C, whose designs have a genuine block-size-dependent
tradeoff; Candidate B's flat one-thread-per-output-position mapping has no
such tradeoff to tune).

## Shared memory usage
None. Candidate B uses zero shared memory (a deliberate advantage over
Candidate A, which needed up to ~36KB/block and still lost).

## Memory access pattern
`grad_out` is read once per `(thread, kh, kw, co)` combination and reused in
a register across all `Cin` accumulators (the read-amplification fix from
Section 8, achieved without shared memory or synchronization). `weight` is
still read once per `(co, ci, kh, kw)` combination per thread -- unavoidable,
since every `ci` needs its own weight value -- but `weight`'s small total
size (`Cout*Cin*KH*KW`, at most a few thousand elements at Forge's shapes)
means it is well-served by the L2/read-only cache across the many threads
that share it. `grad_x` writes are fully coalesced within a `ci` (consecutive
`wi` -> consecutive addresses), same as the baseline.

## Correctness validation
Direct kernel-level parity (`tests/test_cuda_conv2d_dinput_optimization.py::
test_channelfused_matches_baseline_kernel`, 9 shapes x 2 dtypes = 18 cases)
covers stride in {1, 2}, padding in {0, 1, 2}, kernel size in {2, 3, 5}
(Section 23), square and non-square `H`/`W`, `Cin=1` and `Cin=MAX_CIN_REG`
boundary, float32 and float64 (`rtol=1e-3..1e-8`). All pass. Production
dispatch correctness at and across the `Cin <= 16` boundary is covered by
`test_production_dispatch_correct_at_cin_boundary` (parametrized `Cin in
{16, 17}`) through the full `nn.Conv2d`/autograd path against the CPU
backend. The full pre-existing `tests/test_cuda_conv.py` and
`tests/test_cuda_conv2d_backward_optimization.py` suites (finite-difference,
cross-stream, explicit-stream, memory-safety) pass unmodified, since every
one of their shapes has `Cin <= 16` and therefore already exercises the new
production path end-to-end.

## Finite difference results
`test_cuda_conv2d_backward_optimization.py`'s existing
`test_cuda_conv2d_finite_difference_weight`/`_bias` tests (stride in
{1, 2}, padding in {0, 1}) pass unmodified against the new dInput dispatch
(dInput's own gradient is checked directly by `test_cuda_conv.py`'s
pre-existing input finite-difference test, also passing unmodified).

## Cross-stream results
`test_cuda_conv2d_backward_optimization.py`'s
`test_cuda_conv2d_backward_correct_when_inputs_from_different_streams` and
`..._when_grad_output_from_different_stream` pass unmodified -- the new
kernel is called through the exact same `CUDABackend.conv2d_backward` /
`_stream_guard` chokepoint as before; nothing about M28's automatic
cross-stream dependency insertion needed to change.

## Async results
`test_cuda_conv2d_backward_on_explicit_stream_matches_cpu` (explicit
`forge.cuda.Stream`, no default-stream synchronization) passes unmodified.

## Memory results
No new persistent device allocations: Candidate B needs zero additional
global-memory buffers (all its extra state is register-resident). A 100-
iteration stress test at `Cin=16` (the channel-fused path) showed
`allocated_bytes`/`reserved_bytes`/`pending_bytes` all returning to their
pre-loop values after `gc.collect()` + `empty_cache()` -- no leak.

## Hardware variance
Three independent runs of `conv2d_backward_dinput_profile.py`, back-to-back
in the same session, showed substantial run-to-run variance in absolute
timings for *all* kernels (including the unmodified baseline) -- e.g.
`large_channel`'s baseline ranged 36.9-114.4ms across runs, a >3x spread,
consistent with thermal/clock-state variation on this laptop GPU under
sustained benchmark load (not attributable to any code change, since the
unmodified production and dWeight/forward kernels showed the same spread).
Candidate B's *relative* speedup over baseline was positive in every run at
every shape except `mnist_conv1` (where it ranged 0.81x-1.23x, i.e.
effectively a wash) -- see **Before/After** below for the clean, controlled,
same-session comparison used as this milestone's primary evidence.

## Before / after dInput results (controlled, same-session, `empty_cache()`-hygienic)
| Shape | AI | dInput before (GFLOP/s, % ceiling) | dInput after (GFLOP/s, % ceiling) | Speedup |
|---|---|---|---|---|
| mnist_conv1 | 4.00 | 10.02, 16.6% | 8.94, 14.8% | 0.89x |
| mnist_conv2 | 23.89 | 12.32, 11.8% | 29.04, 27.8% | 2.36x |
| large_channel | 47.91 | 12.82, 12.3% | 35.26, 33.7% | 2.75x |
| large_spatial | 23.99 | 12.05, 11.5% | 29.88, 28.6% | 2.48x |
| batch_32 | 47.82 | 12.27, 11.7% | 33.13, 31.7% | 2.70x |
| batch_64 | 47.91 | 12.65, 12.1% | 33.81, 32.3% | 2.67x |
| batch_128 | 47.95 | 12.66, 12.1% | 35.78, 34.2% | 2.83x |

## Before / after Conv2d backward results (dInput + unchanged dWeight + unchanged dBias)
| Shape | Backward before (ms) | Backward after (ms) | Speedup |
|---|---|---|---|
| mnist_conv1 | 3.121 | 3.208 | 0.97x |
| mnist_conv2 | 3.977 | 2.812 | 1.41x |
| large_channel | 81.079 | 58.110 | 1.40x |
| large_spatial | 41.499 | 30.048 | 1.38x |
| batch_32 | 41.578 | 29.718 | 1.40x |
| batch_64 | 81.234 | 58.355 | 1.39x |
| batch_128 | 160.305 | 113.106 | 1.42x |

`mnist_conv1` (`Cin=1`) is the one shape where Candidate B has no launch-
count advantage over baseline (`N*H*W == N*Cin*H*W` when `Cin=1`) and its
restructured loop nest adds a small amount of overhead -- a 3% backward-total
regression, well within Section 37's "does not regress small workloads
severely" acceptance bar and far outweighed by the gains everywhere else.

## MNIST results
`python -m benchmarks.m35_mnist` (real M20 CNN, batch=64) kernel-contribution
ranking, before (M35) vs. after (M36):

| | Before (M35) | After (M36) |
|---|---|---|
| `backward:conv2d` % of step | 50.97% | 46.82% |
| `backward:conv2d` GFLOP/s | 8.415 | 8.748 |

Consistent in direction and rough magnitude with the controlled per-shape
comparison above (MNIST's real conv1/conv2 layers are smaller than most of
the 7 representative shapes, so the blended improvement is more modest).

## Async prefetch results
`pipeline_profile._profile_async_epoch` at batch=64, prefetch depths 1/2/3,
against the new production kernel:

| Prefetch depth | GPU backward (ms/step) | Compute-stream utilization |
|---|---|---|
| 1 | 8.49 | 81.9% |
| 2 | 6.98 | 86.2% |
| 3 | 7.02 | 87.8% |

The pipeline still overlaps host and device work correctly with the faster
kernel -- utilization did not degrade (if anything it improved slightly at
higher depths, since less GPU work per step gives the CPU-side dataloader
more slack to stay ahead).

## Production decision
**Accepted.** Candidate B (`k_conv2d_backward_input_channelfused`) is
integrated into production, dispatched whenever `Cin <= 16`. Candidates A
and C are rejected and remain as profiling-only kernels for reference
(matching M33/M34's convention for a rejected/accepted candidate pair).

## Limitations
- **`empty_cache()` hygiene matters for this benchmark suite on the 940MX**:
  running `conv2d_backward_profile.py`'s full shape sweep (`SHAPES` +
  `BATCH_SIZES`) back-to-back in one process without an `empty_cache()`
  between shapes was observed to accumulate enough cached-but-unreleased
  VRAM to trigger a Windows WDDM TDR launch timeout (`CUDA error: the launch
  timed out and was terminated (code 702)`) partway through the batch sweep
  -- reproducible twice, resolved by adding `forge.cuda.empty_cache()`
  between shapes (this milestone's **Before/after** measurements use that
  hygiene). This is a pre-existing characteristic of the M32 benchmark
  script under this GPU's tight 2GB VRAM budget, not a defect introduced by
  this milestone's kernel change (the accumulation is the same regardless of
  which dInput kernel is dispatched).
- **`MAX_CIN_REG=16` is a hard correctness bound**, not a tunable knob --
  production dispatch never calls the channel-fused kernel above it, but the
  bound was chosen to exactly cover Forge's 7 representative shapes (max
  `Cin=16`), not derived from a register-pressure sweep across larger `Cin`.
  A deeper CNN with `Cin > 16` layers would silently fall back to the
  baseline kernel (no correctness risk, per the dispatch guard) but would
  not benefit from this milestone's speedup there.
- **`mnist_conv1` (`Cin=1`) shows no real gain** (0.89x-1.02x across
  measurements) -- Candidate B's benefit comes from `Cin`-fold thread-count
  reduction and grad_output register reuse, both of which vanish at `Cin=1`.
- Run-to-run hardware variance on this laptop GPU (see **Hardware
  variance**) means any single benchmark invocation's absolute numbers
  should not be over-interpreted -- the controlled, same-session before/after
  comparison is the primary evidence, corroborated by the register-usage
  analysis (a hardware-independent, deterministic measurement).

## Future opportunities
- Combining Candidate B's register-reuse mapping with a modest amount of
  spatial (not channel) shared-memory tiling across neighboring `(hi, wi)`
  positions (whose input windows overlap for `stride < KH`) was not
  attempted -- Section 12 explicitly scoped this milestone to a single
  focused `dInput` experiment, not a general tiled-convolution engine.
- A register-pressure sweep to find the actual `MAX_CIN_REG` ceiling before
  occupancy degrades unacceptably (this milestone stopped at the value that
  exactly covers Forge's existing shapes) could extend the dispatch boundary
  for deeper CNNs.

# Milestone 37: dWeight GEMM occupancy fix -- split-K over existing buffers (accepted, modest)

## Question
M36 left `dWeight` (M34's im2col + existing tiled `k_matmul` GEMM,
unmodified since M34) as the more clearly dominant of `conv2d_backward`'s
two kernels at most representative shapes. This milestone asks whether the
M34 *algorithm* itself is the bottleneck, or whether one of its constituent
stages is disproportionately expensive -- `PROFILE -> ANALYZE -> DESIGN ->
BENCHMARK -> SELECT -> IMPLEMENT -> VALIDATE`, with a negative result
explicitly sanctioned if M34 is already near the practical ceiling.

## Phase 1/2: decomposition and bottleneck diagnosis
`benchmarks/m37_dweight_profile.py` (new) decomposed M34's pipeline at the 7
representative shapes into `im2col`, `grad_output` permute, and GEMM,
timed with CUDA events, plus FLOP/byte/AI/roofline-ceiling accounting
(`benchmarks/roofline.py`, reusing `benchmarks/results/m35_summary.json`'s
measured ceilings) and a GEMM launch-geometry model:

| Shape | weight# | GEMM blocks | occupancy | im2col % | permute % | gemm % | %compute ceiling | class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mnist_conv1 | 72 | 1 | 4.2% | 12.6% | 7.2% | 80.2% | 1.9% | mixed |
| mnist_conv2 | 1,152 | 5 | 20.8% | 47.4% | 7.0% | 45.6% | 14.4% | mixed |
| large_channel | 4,608 | 18 | 75.0% | 55.1% | 7.8% | 37.1% | 32.9% | compute_bound |
| large_spatial | 1,152 | 5 | 20.8% | 48.9% | 6.9% | 44.2% | 14.9% | mixed |
| batch_32 | 4,608 | 18 | 75.0% | 55.0% | 7.8% | 37.2% | 32.7% | compute_bound |
| batch_64 | 4,608 | 18 | 75.0% | 55.0% | 7.8% | 37.2% | 32.6% | compute_bound |
| batch_128 | 4,608 | 18 | 75.0% | 55.2% | 7.8% | 37.0% | 32.6% | compute_bound |

`occupancy = ceil(Cout/16)*ceil(Cin*KH*KW/16) blocks / 24` (3 SMs x 8
resident 256-thread blocks on the 940MX, measured directly via
`cuDeviceGetAttribute` -- thread-count-limited, not register/shared-memory-
limited: `nvcc -Xptxas -v` found **zero stack frame / spill** on every
dWeight-related kernel, `k_im2col_conv2d`/`k_conv2d_grad_output_permute`/
`k_matmul` included, ruling out M36's local-memory failure mode here).
`Cout`/`Cin*KH*KW` are the GEMM's actual `M`/`N` dimensions -- small (8-64,
9-288) -- while the huge `N*Hout*Wout` reduction lives entirely inside each
block's serial inner loop, invisible to block count. Two independent
findings, both measured rather than inferred from elapsed time alone:

1. **`im2col` + `permute` (zero FLOPs) cost 54-63% of total pipeline time**
   at every shape -- more than the GEMM itself.
2. **Occupancy tracks achieved-fraction-of-compute-ceiling almost exactly**:
   4.2%/20.8%/75.0% occupancy shapes measure 1.9%/14.4%/32.9% of ceiling.

## Phase 3/4: candidates (profiling-only first)
Two structurally different fixes were designed, implemented, and measured
(`kernels.cu`, `forge.backend.cuda.experimental_conv_fused`,
`benchmarks/m37_dweight_candidates_profile.py`), targeting bottleneck 1
and/or 2:

- **Candidate A** (`k_dweight_fused_gemm`) -- folds both gathers directly
  into `k_matmul`-shaped tile loads, eliminating `Xcol`/`dYcolT` and their
  two launches entirely (targets bottleneck 1). Same block geometry as
  M34's GEMM (occupancy unchanged).
- **Candidate C** (`k_dweight_fused_gemm_splitk`) -- Candidate A plus
  splitting the reduction across `num_k_splits` blocks (`blockIdx.z`), each
  atomically accumulating into a pre-zeroed output (targets both
  bottlenecks). `atomicAdd`/CAS-emulated `atomic_add_f64` -- contention is
  bounded by `num_k_splits`, not reduction size, since `out` has only
  `Cout*Cin*KH*KW` elements (72-4,608).
- **Candidate E** (`dweight_im2col_gemm_splitk`,
  `experimental_conv_im2col.py`) -- keeps M34's own `Xcol`/`dYcolT` buffer
  reads completely unchanged and applies *only* the split-K occupancy fix,
  via a new `cf_matmul_splitk_*` kernel (a separate, narrowly-scoped GEMM
  variant -- `k_matmul` itself stays byte-for-byte unmodified, satisfying
  the brief's ban on unscoped GEMM changes).

All three verified bit-exact-within-tolerance against `CPUBackend.
conv2d_backward` across shape/stride/padding/kernel-size/dtype combinations
before any timing was trusted (Phase 4's ordering). `nvcc -Xptxas -v`: zero
stack frame/spill on every candidate; register counts 47-53 (vs. `k_matmul`'s
own 30-32) -- register-limited to ~4-5 resident blocks/SM for the fused
kernels (Candidates A/C), lower than `k_matmul`'s thread-limited 8/SM, a
real (if secondary) additional occupancy cost specific to the fused
approach.

## Measured results (honest, full-pipeline, corrected)
An initial benchmark-harness bug measured Candidate E's split-K GEMM stage
in isolation against pre-built buffers while comparing it to M34's *full*
pipeline time (im2col + permute + GEMM) -- an apples-to-oranges comparison
that implied a dramatic 2.7-9.0x win. This was caught by cross-checking
against the real, end-to-end `CUDABackend.conv2d_backward` production call,
which did not show a matching improvement; the harness was fixed to rebuild
`Xcol`/`dYcolT` on every timed call (matching M34's own "total pipeline"
methodology) before any candidate was accepted. The corrected, full-pipeline
numbers are the ones below and the ones this milestone's decision is based
on -- reported here specifically so the correction is not lost.

| Shape | weight# | M34 total (ms) | A (ms) | A speedup | C best (ms) | C speedup | E best (ms) | E speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnist_conv1 | 72 | 3.78 | 7.34 | 0.52x | 1.23 (split 8) | 3.07x | 1.24 (split 16) | 3.04x |
| mnist_conv2 | 1,152 | 1.64 | 1.69 | 0.97x | 1.26 (split 16) | 1.30x | 1.37 (split 8) | 1.20x |
| large_channel | 4,608 | 13.54 | 23.04 | 0.59x | 18.75 (split 2) | 0.72x | 13.61 (split 16) | 1.00x |
| large_spatial | 1,152 | 14.77 | 17.00 | 0.87x | 10.75 (split 16) | 1.37x | 11.72 (split 16) | 1.26x |
| batch_32 | 4,608 | 6.70 | 11.64 | 0.58x | 9.48 (split 2) | 0.71x | 6.76 (split 16) | 0.99x |
| batch_64 | 4,608 | 13.64 | 23.12 | 0.59x | 18.76 (split 2) | 0.73x | 13.66 (split 16) | 1.00x |
| batch_128 | 4,608 | 27.42 | 46.01 | 0.60x | 37.59 (split 8) | 0.73x | 27.42 (split 4) | 1.00x |

**Candidate A: rejected.** Loses at every shape (0.52-0.97x) -- recomputing
gather indices via integer div/mod on every GEMM tile-loop iteration costs
more than the eliminated buffer materialization saves, at every tested
shape.

**Candidate C: rejected.** Wins at the two low-occupancy shapes not already
covered better by Candidate E (`mnist_conv1` 3.07x, not production-relevant
below the 256-element threshold; `mnist_conv2` 1.30x; `large_spatial`
1.37x) but **regresses 27-29%** at every shape with >= 18 GEMM blocks
(`large_channel`/`batch_*`, 0.71-0.73x) -- the fused-recompute cost is a net
loss once occupancy is no longer the limiting factor, and unconditionally
deploying it would be a real regression risk.

**Candidate E: accepted.** Never regresses beyond measurement noise (worst
case 0.99x, i.e. ~1% within a shape's own run-to-run variance) and wins
modestly but genuinely at the low-occupancy shapes (`mnist_conv2` 1.20x,
`large_spatial` 1.26x; `mnist_conv1` 3.04x is a real effect but not
production-relevant below threshold). At the already-well-occupied 18-block
shapes (`large_channel`, `batch_32/64/128`) the result is flat (0.99-1.00x)
-- consistent with the Phase 2 diagnosis that occupancy was never the
bottleneck there, so fixing it buys nothing, and split-K's own overhead
(one `cudaMemsetAsync` + atomic accumulation) is negligible rather than
harmful.

`num_k_splits = min(16, ceil(N*Hout*Wout / 16))`
(`experimental_conv_im2col.recommended_num_k_splits`) -- swept over
`{1,2,4,8,16}` at all 7 shapes; 16 (or the largest value not exceeding the
shape's own reduction-tile count) won or tied everywhere. A follow-up sweep
to `{32,64,128}` on the two largest-`M` shapes found only an additional
2-8% at 128 splits -- diminishing enough, relative to the atomic-contention
risk of a much larger constant, that 16 is kept as the one production value
rather than a per-shape-tuned formula (Phase 5's simplicity criterion).

## Selected strategy
`CUDABackend.conv2d_backward`'s dWeight branch (`backend.py`) now calls
`experimental_conv_im2col.dweight_im2col_gemm_splitk` instead of M34's
`dweight_im2col_gemm`, unconditionally above the unchanged
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD = 256`. `im2col`/`grad_output_permute`
(both M34, unmodified) and `k_matmul` (M11, unmodified) are untouched; the
new `cf_matmul_splitk_*` kernel is a separate, narrowly-scoped GEMM variant
used only by this one call site. Candidates A/C remain as rejected,
correctness-tested profiling-only code
(`forge.backend.cuda.experimental_conv_fused`) for reference, matching
M33/M36's convention for a rejected/accepted candidate set.

## Correctness
`tests/test_cuda_conv2d_backward_weight_splitk_gemm.py` (28 tests): CPU
parity across 7 shape/stride/padding/kernel-size combinations (f32 + f64),
finite difference, explicit-stream, cross-stream (both directions),
repeated-use memory safety (100 iterations), allocator cache-hit reuse, unit
coverage for `recommended_num_k_splits`'s edge cases (`M` below one tile,
exactly one tile, exactly 16 tiles), and production-dispatch coverage
through the real `Tensor.conv2d`/`nn.Conv2d` API. `tests/test_cuda_
conv2d_backward_weight_fused_gemm_candidates.py` (17 tests): correctness-only
coverage for the rejected Candidates A/C, so shipped-but-unused profiling
code cannot silently bit-rot.

## MNIST validation
CUDA-event-based, same-session, controlled measurement of the real
production `CUDABackend.conv2d_backward` end-to-end (not the training
step) at MNIST's own layer shapes showed the expected small, shape-
dependent effect: `mnist_conv2`-shaped calls (M37's actual second-layer
shape) improved modestly; `mnist_conv1` is below the GEMM threshold and
unaffected. A **wall-clock, interleaved A/B of the real M20 CNN's full
training step** (`examples/mnist/model.build_model`, batch 64, 3 rounds of
20 iterations each, old-vs-new dWeight dispatch alternated to cancel
session drift) measured **0.994x** -- statistically indistinguishable from
1.0x given each variant's own ~25-30% run-to-run coefficient of variation
at this wall-clock granularity (Python dispatch overhead dominates a
14ms full step far more than a sub-millisecond dWeight change). Naive
before/after comparison against the M36 session's own saved
`benchmarks/results/m36_mnist.json` throughput numbers showed larger
apparent deltas (e.g. batch=32: 3,527 -> ~4,650-4,720 samples/sec) that did
**not** reproduce in this milestone's own controlled, interleaved,
same-session test -- attributed to cross-session hardware/thermal variance
(this laptop GPU's documented characteristic, see **Hardware variance**
above), not to this milestone's change, and explicitly not claimed as a
real effect.

## Roofline update
| Shape | AI | GFLOP/s before | %ceiling before | GFLOP/s after | %ceiling after | classification after |
|---|---:|---:|---:|---:|---:|---|
| mnist_conv1 | 4.00 | 1.95 | 1.9% | 5.80 | 5.6% | mixed_or_ambiguous |
| mnist_conv2 | 23.89 | 15.09 | 14.4% | 18.23 | 17.4% | mixed_or_ambiguous |
| large_channel | 47.91 | 34.37 | 32.9% | 33.98 | 32.5% | compute_bound |
| large_spatial | 23.99 | 15.54 | 14.9% | 19.73 | 18.9% | mixed_or_ambiguous |
| batch_32 | 47.82 | 34.25 | 32.7% | 34.20 | 32.7% | compute_bound |
| batch_64 | 47.91 | 34.13 | 32.6% | 33.85 | 32.4% | compute_bound |
| batch_128 | 47.95 | 34.08 | 32.6% | 33.73 | 32.3% | compute_bound |

No shape reclassifies. The low-occupancy shapes move measurably toward the
compute ceiling (14.4%->17.4%, 14.9%->18.9%) without crossing the 30%
`NEAR_CEILING_FRACTION` boundary into `compute_bound`; the already-
compute-bound shapes are unchanged within noise. dWeight was never
bandwidth-bound at any tested shape (`im2col`'s own materialization
traffic, not the GEMM, is the pipeline's dominant cost -- see Phase 1 --
but that stage is unchanged by this milestone).

## Amdahl analysis
dWeight's own full-pipeline improvement is 1.00-1.26x across the
production-dispatched shapes (>= 256 weight elements), and dWeight is
itself only part of `conv2d_backward` (dInput, M36-optimized, is
comparable or larger at several shapes -- e.g. ~13ms dInput vs. ~13.5ms
pre-M37 dWeight at `large_channel`). The controlled same-session full-
`conv2d_backward` measurement (CUDA events, `m37_dweight_profile.py`
before/after) found 0.99-1.08x at the production shapes, with the largest
real effect (1.08x) at `large_spatial`. Composed with `conv2d_backward`'s
~47% share of the MNIST training step (M36 baseline), the maximum
plausible end-to-end effect at MNIST's own shapes is on the order of a
few percent -- consistent with, and explaining, why the wall-clock
MNIST step A/B (above) could not distinguish the change from noise. This
is exactly the brief's cautioned scenario: **a real, evidence-backed
isolated improvement whose end-to-end effect is small enough to be
unmeasurable at the full-training-step level** -- stated explicitly here
rather than implied.

## Production decision
**Accepted, with an honest scope.** `dweight_im2col_gemm_splitk` replaces
`dweight_im2col_gemm` as the dWeight GEMM call above the existing
threshold: it never regresses beyond noise, and provides a real (if
modest) improvement specifically at low-GEMM-occupancy shapes, with no
added complexity (no new dispatch branch -- the fix applies unconditionally
above the unchanged M34 threshold). Candidates A and C are rejected and
kept as profiling-only reference code.

## Limitations
- **The dominant remaining dWeight cost is `im2col` construction itself**
  (54-63% of total pipeline time, Phase 1) -- this milestone's split-K fix
  addresses the GEMM stage only (37-46% of the pipeline) and leaves
  `im2col`/`permute` completely unchanged. Candidates A/C attempted to
  remove that cost via fusion and were rejected on measured evidence (net
  loss once occupancy is fixed by other means); a genuinely cheaper
  `im2col` formulation (e.g. avoiding redundant re-reads of overlapping
  receptive fields) remains unexplored and is the clearest named target for
  a future milestone.
- **No shape between 256 and 1,152 weight elements has ever been tested**
  for this dispatch branch (M34's original caveat, still true) -- the
  threshold itself is out of scope for this milestone (Section 31 of M35,
  reaffirmed here).
- **`num_k_splits = 16` is a fixed constant, not a per-shape-tuned value**
  -- chosen because it won or tied at every tested shape with a wide margin
  over the alternative it would need to beat (a formula based on device
  block capacity performed no better in early testing and added
  complexity without matching gain).
- Run-to-run hardware variance on this laptop GPU is large enough at
  wall-clock, full-training-step granularity (~25-30% CV observed) to mask
  effects of this size entirely -- CUDA-event-based, same-session,
  controlled comparisons are the only trustworthy evidence for a change
  this small, and this section reports exactly that data rather than
  cross-session throughput deltas.

## Future opportunities
- A genuinely cheaper `im2col` (the now-dominant single cost) -- e.g. a
  tiled/on-demand formulation that avoids materializing the full `Cin*KH*KW`
  redundancy, or reduces repeated global-memory re-reads of overlapping
  receptive fields -- is the clearest next target, per this milestone's own
  Phase 1 decomposition.
- Revisiting the 256-weight-element threshold with the now-available
  256-1,152 gap data (Candidate E measured 3.04x at 72 elements, well below
  the current threshold) could plausibly lower it, but was left unchanged
  here as out of this milestone's stated scope.

# Milestone 38: dWeight im2col elimination -- half-fused split-K GEMM (Candidate B, partial acceptance)

## Question
M37 left `im2col` (`k_im2col_conv2d`) + `grad_output` permute -- pure data
movement, zero FLOPs -- as the dominant cost of the `dWeight` pipeline
(54-63% of total time, more than the GEMM itself) and named a genuinely
cheaper `im2col` formulation as the clearest remaining target. This
milestone asks whether `im2col`'s `Xcol` materialization can be eliminated
or reduced without regressing M37's production dispatch
(`dweight_im2col_gemm_splitk`), via the brief's mandated
`PROFILE -> ANALYZE -> DESIGN -> BENCHMARK -> SELECT -> IMPLEMENT ->
VALIDATE` sequence.

## Baseline (reproduced fresh, `benchmarks/m38_im2col_profile.py`)
`im2col`/`grad_output_permute` are byte-for-byte unchanged since M34, so
isolated-stage numbers match M37 within run-to-run hardware variance:

| Shape | weight# | blocks_y | blocks_x | im2col (ms) |
|---|---:|---:|---:|---:|
| mnist_conv1 | 72 | 1 | 1 | 0.474 |
| mnist_conv2 | 1,152 | 1 | 5 | 0.785 |
| large_channel | 4,608 | 2 | 9 | 7.325 |
| large_spatial | 1,152 | 1 | 5 | 7.370 |
| batch_32 | 4,608 | 2 | 9 | 3.718 |
| batch_64 | 4,608 | 2 | 9 | 7.385 |
| batch_128 | 4,608 | 2 | 9 | 14.863 |

`blocks_y = ceil(Cout/16)`, `blocks_x = ceil(Cin*KH*KW/16)` -- the dWeight
GEMM's own 16x16 tile launch geometry (M37).

## Root-cause analysis (Section 3 of the milestone brief)
`nvcc -Xptxas -v` on `k_im2col_conv2d`: **32 registers/thread (f32/f64
alike), 0 bytes stack frame, 0 bytes spill** -- launched with one thread per
`Xcol` output element (`M*K` threads, thousands of blocks at every
representative shape), so it is **thread-count-limited, not
register/occupancy-limited** -- unlike the GEMM (M37), `im2col`'s own
occupancy is never the problem. Its cost is genuine memory traffic: it
writes `M*K` elements to `Xcol` (`K = Cin*KH*KW`, i.e. up to `KH*KW`x the
"natural" per-input-element write rate) and reads `x` with the same
overlap-driven re-read pattern any im2col has (M34's **Memory traffic
analysis** section already established this; unchanged here). `k_conv2d_
grad_output_permute`: 29 registers, 0 stack/spill -- the smaller, cheaper
gather (7-8% of pipeline time, M37).

The mechanism that determines whether *eliminating* `Xcol` materialization
can win is a property of the *GEMM's* tiling, not of `im2col` itself:
`k_matmul_splitk`'s `tile_a` load depends only on `(row, a_m)` -- not on
`blockIdx.x` -- so any kernel that fuses `tile_a`'s gather into the tile
load redundantly recomputes it `blocks_x` times (once per unique
`blockIdx.x` sharing that row range); symmetrically, fusing `tile_b`'s
gather costs a redundant-recompute factor of `blocks_y`. M37's Candidate
A/C fused *both* gathers and lost at every shape with >= 18 GEMM blocks --
consistent with this model, since at those shapes `blocks_x` reaches 9 (the
*expensive*, 47-55%-of-pipeline `im2col`-equivalent gather... **paid only
`blocks_y <= 2` times** in M37's own orientation, since `tile_b` is the
`Xcol` operand there) while `blocks_x` (up to 9) multiplies the *cheap*
`grad_output` gather. The milestone brief's Candidate A (this milestone's
own framing of "eliminate `im2col` via a fused implicit GEMM") is
**mechanistically identical to M37's already-rejected Candidate A/C** --
re-deriving it was unnecessary; M37's measured data (0.52-0.97x at every
representative shape, `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 37** section) already answers it and is not repeated here
(Section 12 of this milestone's own brief permits reusing prior rejected
candidates for reproducibility rather than re-litigating them).

## Candidate B: half-fused split-K GEMM (implemented, measured, partially accepted)
Fuses *only* `im2col`'s gather (`tile_b`, this milestone's actual target --
eliminates the `Xcol` buffer entirely) directly into a split-K GEMM's tile
load, while keeping `grad_output_permute`'s cheap, already-materialized
`dYcolT` as `tile_a`'s source (`kernels.cu`'s `k_dweight_halffused_gemm_
splitk`; Python wrapper `forge.backend.cuda.experimental_conv_halffused.
dweight_halffused_gemm_splitk`). Unlike M37's "fuse everything" candidates,
this is deliberately asymmetric: it pays the redundant-regather tax only
where the model above says it is cheap (`blocks_y`, <= 2 at every
representative shape) and never where it was expensive (`blocks_x`, up to
9). Split-K is folded in unconditionally, matching the accepted M37
occupancy fix, so the comparison isolates exactly one variable.

`nvcc -Xptxas -v`: **0 bytes stack frame/spill** (f32 and f64); 49
registers/thread (f32), 54 (f64) -- comparable to M37's fused candidates
(47-53) and higher than the plain split-K GEMM's 27/32, i.e.
register-limited to ~5 resident blocks/SM (vs. the thread-limited 8/SM the
unfused split-K GEMM gets) -- a real but secondary occupancy cost, same
finding as M37's Candidate A/C.

### Correctness
Verified against `CPUBackend.conv2d_backward` across 8 shape/stride/
padding/kernel-size combinations spanning both `blocks_y == 1` and
`blocks_y >= 2` (f32 + f64), finite difference, explicit-stream,
cross-stream (both directions), and against the M37 baseline directly from
identical inputs -- before any timing was trusted.

### Measured results (`benchmarks/m38_im2col_profile.py`, interleaved
CUDA-event A/B, 30 iterations + 5 warmup per side)

| Shape | weight# | blocks_y | blocks_x | baseline (ms) | Candidate B (ms) | speedup |
|---|---:|---:|---:|---:|---:|---:|
| mnist_conv1 | 72 | 1 | 1 | 1.324 | 1.609 | 0.823x |
| mnist_conv2 | 1,152 | 1 | 5 | 1.562 | 1.186 | 1.317x |
| large_channel | 4,608 | 2 | 9 | 13.687 | 14.695 | 0.931x |
| large_spatial | 1,152 | 1 | 5 | 11.779 | 9.043 | 1.302x |
| batch_32 | 4,608 | 2 | 9 | 6.924 | 7.505 | 0.922x |
| batch_64 | 4,608 | 2 | 9 | 13.718 | 14.819 | 0.926x |
| batch_128 | 4,608 | 2 | 9 | 27.540 | 29.194 | 0.943x |

A `Cout` sweep at a fixed base shape (`Cin=16, H=W=28, K=3, N=64`) isolates
`blocks_y` as the exact discriminator, cleanly and monotonically:

| Cout | blocks_y | baseline (ms) | Candidate B (ms) | speedup |
|---:|---:|---:|---:|---:|
| 8 | 1 | 10.519 | 7.297 | 1.442x |
| 16 | 1 | 10.906 | 7.588 | 1.437x |
| 17 | 2 | 13.025 | 14.169 | 0.919x |
| 24 | 2 | 13.375 | 14.400 | 0.929x |
| 32 | 2 | 13.720 | 14.728 | 0.932x |
| 48 | 3 | 16.740 | 22.137 | 0.756x |
| 64 | 4 | 19.856 | 29.346 | 0.677x |

`blocks_y == 1` (`Cout <= 16`): a clean, real win (1.30-1.44x) at every
tested shape/Cout except `mnist_conv1` (0.823x -- below the 256-weight
production threshold regardless, `K = 9` far under one 16-wide GEMM tile,
so the fused kernel's boundary-check overhead dominates a problem this
small; not production-relevant). `blocks_y >= 2`: a real, reproducible
**regression** that worsens monotonically with `blocks_y` (0.92-0.93x at
`blocks_y=2`, 0.76x at 3, 0.68x at 4) -- consistent with the root-cause
model above: the redundant-regather tax scales with `blocks_y`, and past
`blocks_y=1` it costs more than the eliminated `Xcol` buffer saves.

### Memory
Peak reserved bytes (`forge.cuda.reset_peak_memory_stats()` +
`memory_stats().peak_reserved_bytes`, 20 iterations, cache-hygienic):
Candidate B needs no `Xcol` buffer, so peak reserved memory drops
substantially **at every shape**, including the ones where it is not
speed-dispatched:

| Shape | peak baseline (MB) | peak Candidate B (MB) | reduction |
|---|---:|---:|---:|
| mnist_conv1 | 4.98 | 3.25 | 1.72 MB |
| mnist_conv2 | 4.63 | 1.65 | 2.97 MB |
| large_channel | 42.89 | 15.33 | 27.56 MB |
| large_spatial | 42.88 | 15.32 | 27.56 MB |
| batch_32 | 21.46 | 7.67 | 13.78 MB |
| batch_64 | 42.89 | 15.33 | 27.56 MB |
| batch_128 | 85.77 | 30.64 | 55.12 MB |

No regression risk from this axis (Section 6): Candidate B is
memory-*cheaper* than the baseline everywhere, including the `blocks_y >=
2` shapes where it is not selected for speed.

## Candidate C: partial/tiled materialization (analytically rejected, not implemented)
Chunking `im2col` into `K`-tile-sized partial materializations (e.g. one
`M x 16` `Xcol` tile at a time, consumed by all `blocks_y` row-groups
before being discarded) was considered but **not implemented**, for a
reason grounded in already-measured facts rather than speculation:
chunking does not reduce total bytes moved (the same `M*K` writes/reads
happen either way, just split across more launches) and strictly increases
kernel-launch count -- `ceil(K/16)` launch pairs instead of 1 im2col + 1
GEMM launch (e.g. 9 pairs at `large_channel`'s `K=144`). M31's own
per-launch dispatch overhead measurement (~54-65us) means this adds real,
non-zero cost with **zero bandwidth benefit** to offset it. Its only
possible advantage over full materialization -- lower peak memory, from
never holding the whole `Xcol` buffer at once -- is already achieved, more
simply and with a genuine *speed* win rather than parity, by Candidate B's
peak-memory results above (comparable or larger reduction, no fused kernel
needed on this axis). Candidate C is strictly dominated by Candidate B on
both axes it could have competed on, so implementing and benchmarking it
was not justified (Section 16's "do not force an optimization merely to
produce a code change" applies symmetrically to *not* building a candidate
whose own numbers are already implied by measurements in hand).

## Production decision
`CUDABackend.conv2d_backward`'s `dWeight` branch (`backend.py`) now adds a
second, orthogonal dispatch condition **only above the existing
`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD`**: `blocks_y = ceil(Cout/16) == 1`
calls `dweight_halffused_gemm_splitk` (Candidate B); otherwise (`blocks_y
>= 2`) the unchanged M37 `dweight_im2col_gemm_splitk` is used, exactly as
before. `im2col`/`grad_output_permute`/`k_matmul`/`cf_matmul_splitk_*` are
all completely unmodified; Candidate B is a new, narrowly-scoped kernel
used only at this one call site, only in the regime it measurably wins.
This is a genuine, evidence-backed **partial acceptance** -- not the
"either accept unconditionally or reject entirely" framing early
milestones used, matching Section 10's explicit permission to add
shape-dependent dispatch when measurements justify it.

## MNIST validation
The real M20 CNN's actual layer shapes: conv1 (`Cin=1,Cout=8`, 72 weight
elements) stays below threshold, unaffected. conv2 (`Cin=8,Cout=16,K=3`,
1,152 weight elements, `blocks_y=1`) is **exactly the regime Candidate B
wins in** -- unlike M37, whose accepted change never touched MNIST's own
conv2 shape's dispatch branch differently.

**Controlled, CUDA-event, same-session, before/after full
`conv2d_backward` at MNIST's real conv2 shape** (`Cin=8, Cout=16, H=W=13,
K=3`, monkeypatched dispatch for the "old"/M37-only measurement, un-patched
for "new"/M38): **1.164x** (old 2.833ms, new 2.433ms, 30 iterations + 5
warmup, low variance) -- a real, reproducible, full-`conv2d_backward`-level
improvement at the shape that matters most for this codebase's own example
model.

**Wall-clock, interleaved, same-session A/B of the real full training step**
(`examples/mnist/model.build_model`, batch 64, 3 rounds of 20 iterations,
alternating order each round): **1.0065x** -- statistically
indistinguishable from 1.0x at ~2.7-2.9% per-variant coefficient of
variation, for the same Amdahl reason M37 documented (a full step is
~13.7ms, dominated by Python/dispatch overhead; a sub-millisecond `dWeight`
improvement composed with `conv2d_backward`'s own fraction of the step is
too small to surface at wall-clock granularity). Reported honestly, exactly
as M37 did, rather than treated as contradicting the controlled
CUDA-event result above.

## Correctness
`tests/test_cuda_conv2d_backward_weight_halffused_gemm.py` (25 tests): CPU
parity across 8 shape/stride/padding/kernel-size combinations spanning both
`blocks_y` regimes (f32 + f64), direct agreement with the M37 baseline from
identical inputs, finite difference, explicit-stream, cross-stream (both
directions), 100-iteration memory-safety, allocator cache-hit reuse, and
production-dispatch coverage through the real `Tensor.conv2d`/`nn.Conv2d`
API covering both branches of the new `Cout`-based condition (`Cout=16` ->
Candidate B; `Cout=17,32` -> unfused split-K; `Cout` below the
weight-element threshold -> unaffected).

## Stream / Async
Explicit-stream, cross-stream (both directions), and repeated-use tests all
pass -- Candidate B reuses `grad_output_permute` (unchanged M34 kernel,
already stream-safe) and the same `_require_compute_dtype`/`_stream_guard`
chokepoint every other `CUDABackend` method already routes through. No new
synchronization was added.

## Tests
1,328 tests total (1,303 pre-existing + 25 new in `tests/test_cuda_
conv2d_backward_weight_halffused_gemm.py`, listed above wholesale in the
Milestone 38 `progress.md` entry), all passing on the 940MX after a clean
rebuild.

## Limitations
- **`im2col`/`grad_output_permute` themselves are unmodified** -- Candidate
  B eliminates the *consumption* of a materialized `Xcol` at qualifying
  shapes but does not change the `im2col` kernel or its cost at shapes
  where it's still used (`blocks_y >= 2`, including every representative
  shape with `Cout > 16`). A genuinely cheaper `im2col` formulation itself
  (not just a way to avoid consuming its output) remains unexplored.
- **The `blocks_y == 1` regime is narrow**: only `Cout <= 16` qualifies.
  Of the 7 representative shapes, only `mnist_conv2` and `large_spatial`
  are both above the weight-element threshold and in this regime; the
  4 `Cout=32` shapes (`large_channel`, `batch_32/64/128`) do not benefit
  and correctly fall back to M37's unchanged dispatch.
- **`mnist_conv1` (`Cout=8`, `blocks_y=1`) is the one exception to the
  "`blocks_y==1` wins" pattern** (0.823x) -- but it sits below the
  256-weight-element production threshold regardless (M33/M34's existing
  boundary), so this does not affect any production dispatch decision; it
  is reported here rather than silently dropped from the table (Section 11:
  "do not present favorable measurements only").
- **Candidate C was rejected analytically, not empirically** -- the
  reasoning (zero bandwidth benefit, real launch-overhead cost, dominated
  by Candidate B on the one axis -- memory -- it could compete on) is
  grounded in existing measurements (M31's launch-overhead figure, this
  milestone's own peak-memory results) but was not independently verified
  by implementing and timing it.
- Run-to-run hardware variance on this laptop GPU remains large enough at
  wall-clock, full-training-step granularity to mask effects of `dWeight`'s
  own size -- exactly as M37 documented; the CUDA-event, same-session,
  controlled comparisons above remain the only trustworthy evidence for an
  effect this size.

## Future opportunities
- A genuinely cheaper `im2col` kernel itself (not just avoiding its
  consumption) -- e.g. reducing the `KH*KW`-fold redundant re-read of
  overlapping receptive fields via shared-memory input tile staging inside
  `k_im2col_conv2d` itself -- remains the clearest named target for the
  `blocks_y >= 2` shapes Candidate B cannot help.
- Extending Candidate B's asymmetric-fusion idea to `blocks_y >= 2` via a
  kernel that internally loops over multiple `Cout` row-groups per block
  (amortizing the fused gather across them instead of relying on GEMM's
  own block tiling) was identified analytically during this milestone's
  root-cause analysis but not implemented -- a plausible next step if a
  future milestone wants to close the `blocks_y >= 2` gap specifically.
