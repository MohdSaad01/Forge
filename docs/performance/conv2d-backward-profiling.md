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
