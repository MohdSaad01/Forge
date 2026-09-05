# M40 — Post-M39 CUDA Bottleneck Re-Characterization & M41 Target Selection

## Executive Summary
A fresh, same-session, CUDA-event-based re-measurement of Forge's current CUDA
training pipeline shows that M31-M39's optimization work (CrossEntropy fusion,
`dInput` register-resident channel fusion, `dWeight` im2col+split-K+half-fused
GEMM) has succeeded well enough that **no single kernel dominates the way
`conv2d` backward did in M31 (55-57%) or M35 (51%)**. `conv2d` backward is
still the largest phase (48.4% of the measured forward+backward compute step),
but its two sub-kernels are now close to each other in cost and *both* are
compute-bound (43-44% of the practical compute ceiling) at Forge's larger
representative shapes -- meaningfully closer to their practical ceiling than
before, and closer to each other than M35/M39's own numbers suggested at the
time. At MNIST's own real layer shapes, `dWeight` (29.9% of the step) is now
larger than `dInput` (15.6%), a reversal from M35/M36's finding, driven mostly
by `mnist_conv1`'s below-threshold block-reduce kernel (already tuned in
M21/M33, little further headroom). The single largest *untouched* kernel in
the entire pipeline is **`k_conv2d_forward`**: 17.0% of the training step,
identical total FLOP count to `dInput`/`dWeight` at a given shape, yet
achieving only 10.7-18.0% of the practical compute ceiling -- meaningfully
further from its ceiling than either backward Conv2d kernel, and structurally
still the same one-thread-per-output-element, zero-reuse kernel Milestone 15
first wrote, never revisited by any of M31-M39's Conv2d optimization work.
The asynchronous pipeline is healthy (87-94% compute-stream utilization at
every tested batch size); CrossEntropy, the optimizer, and transfers each
remain under 5% of the full step. **M41's recommended target is
`k_conv2d_forward`**, via the same im2col + existing tiled/split-K GEMM
technique M34/M39 already validated and productionized for `dWeight` --
reusing existing, tested infrastructure (`im2col_smem`, `cf_matmul_splitk_*`)
rather than requiring new kernel design.

## Current Pipeline (post-M39)
```
DataLoader (CPU, background thread) --pinned staging--> async H2D (transfer stream)
                                                              |
                                                    cross-stream dependency
                                                              v
                                    compute stream: zero_grad -> forward -> loss -> backward -> optimizer.step
                                                              |
   forward:  Conv2d(naive, M15, UNCHANGED) -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear
   backward: CrossEntropy(fused, M31) -> Linear -> ReLU -> MaxPool2d -> conv2d_backward{
                 dInput:  channel-fused register mapping (M36, Cin<=16) / original kernel (Cin>16, unused by any Forge shape)
                 dWeight: weight_elements < 256      -> per-thread/block-reduce (M21/M33, unchanged)
                          weight_elements >= 256:
                              blocks_y == 1 (Cout<=16) -> permute + half-fused split-K GEMM (M38)
                              blocks_y >= 2 (Cout>16)  -> permute + im2col[-smem] (M39) + split-K GEMM (M37)
                 dBias:   block-reduce (M21, unchanged, negligible)
             } -> ReLU -> MaxPool2d -> Conv2d -> ReLU
   optimizer: Adam (M17, unchanged, negligible)
```

## Measurement Methodology
- **Hardware**: NVIDIA GeForce 940MX (GM108, Maxwell), Compute Capability
  5.0, CUDA 12.6, driver 582.53, Windows/WDDM, 2GB VRAM, 3 SMs / 384 CUDA
  cores. Development host: Intel i5-7200U, 8GB RAM, Python 3.13.5.
- **Timing**: `forge.backend.cuda.profiling_events.TimedEvent`
  (`cudaEventRecord`/`cudaEventElapsedTime`, timing-enabled events) for every
  GPU-side isolated-kernel and per-phase number in this document -- the same
  primitive M31 introduced and every milestone since M32 has used for
  kernel-level measurement. `time.perf_counter()` with explicit
  `forge.cuda.synchronize()` bracketing is used only for CPU-side/whole-epoch
  wall-clock numbers (matching `benchmarks/timing.py`'s established
  convention).
- **Warmup/iterations**: 5 warmup + 30 measured iterations for every isolated
  kernel/phase number (matching M32-M39's own convention); MNIST epoch
  numbers use 1,024 synthetic samples with 5 warmup batches.
- **Interleaving**: all numbers in this document were collected in one
  continuous session (no restart between phases) to control for the 940MX's
  documented thermal/clock-state run-to-run variance; ceilings (Phase 0) were
  re-measured fresh in the same session rather than reusing an older
  `m35_hardware.json` snapshot, so every fraction-of-ceiling number in this
  report is self-consistent within this one session.
- **New tooling**: `benchmarks/m40_bottleneck_recharacterization.py` (new,
  measurement-only) reuses `conv2d_backward_profile.SHAPES`/
  `BATCH_SWEEP_BASE`, `m35_hardware`'s ceiling methodology, `m35_mnist`'s
  kernel-contribution ranking, and `pipeline_profile._profile_async_epoch`
  directly rather than re-deriving any of them -- the only genuinely new
  code is the per-shape decomposition of the *current* production `dWeight`
  dispatch (Phase 1), since no prior script decomposes the post-M39 pipeline
  into its actual currently-selected sub-stages at every representative
  shape (M37/M38/M39's own scripts each compare a specific old-vs-new
  candidate pair, not "what is production doing right now, broken down").
  No production kernel, dispatch, or public API was modified.

## Dispatch Verification
Computed with the exact same formula `CUDABackend.conv2d_backward` uses
(`weight_elements = Cout*Cin*KH*KW`, `blocks_y = ceil(Cout/16)`), confirmed
against the actual measured code path for every representative shape:

| Shape | weight_elements | blocks_y | Selected dWeight path |
|---|---:|---:|---|
| mnist_conv1 | 72 | n/a (< 256) | `cf_conv2d_backward_weight` (M21 block-reduce) |
| mnist_conv2 | 1,152 | 1 | `dweight_halffused_gemm_splitk` (M38) |
| large_channel | 4,608 | 2 | `dweight_im2col_smem_gemm_splitk` (M39) |
| large_spatial | 1,152 | 1 | `dweight_halffused_gemm_splitk` (M38) |
| batch_32 | 4,608 | 2 | `dweight_im2col_smem_gemm_splitk` (M39) |
| batch_64 | 4,608 | 2 | `dweight_im2col_smem_gemm_splitk` (M39) |
| batch_128 | 4,608 | 2 | `dweight_im2col_smem_gemm_splitk` (M39) |

This matches every prior milestone's documented dispatch exactly -- no drift.

## Current Bottleneck Table (isolated kernels, 940MX, TimedEvent, 30 iter/5 warmup)

| Shape | fwd (ms) | dInput (ms) | dWeight (ms) | dBias (ms) | backward total (ms) | dominant |
|---|---:|---:|---:|---:|---:|---|
| mnist_conv1 | 0.645 | 0.783 | 2.093 | 0.217 | 3.092 | **dWeight (68%)** |
| mnist_conv2 | 1.324 | 0.855 | 1.051 | 0.093 | 1.999 | dWeight (53%) |
| large_channel | 25.950 | 13.006 | 10.040 | 0.722 | 23.768 | **dInput (55%)** |
| large_spatial | 13.277 | 7.759 | 8.791 | 0.753 | 17.303 | dWeight (51%) |
| batch_32 | 12.635 | 7.455 | 5.090 | 0.376 | 12.921 | **dInput (58%)** |
| batch_64 | 25.985 | 13.269 | 10.073 | 0.721 | 24.062 | **dInput (55%)** |
| batch_128 | 51.971 | 25.558 | 20.454 | 1.376 | 47.388 | **dInput (54%)** |

**New finding**: post-M39, dominance now alternates by `blocks_y`, not
uniformly favoring one kernel. At `blocks_y==1` shapes (`Cout<=16`:
`mnist_conv1`, `mnist_conv2`, `large_spatial`) `dWeight` is now larger; at
`blocks_y>=2` shapes (`Cout>16`) `dInput` is larger. Neither M35 nor M39's own
per-shape numbers stated this alternation explicitly (M39's own "Next
bottleneck" note only checked `large_channel`).

## Conv2d Backward Decomposition (dWeight sub-stages, current production functions)

`blocks_y == 1` (half-fused split-K GEMM, M38 -- `permute` + one fused kernel):

| Shape | permute (ms) | fused_gemm (ms) | permute % | fused_gemm % |
|---|---:|---:|---:|---:|
| mnist_conv2 | 0.117 | 0.935 | 11.1% | 88.9% |
| large_spatial | 1.031 | 7.760 | 11.7% | 88.3% |

`blocks_y >= 2` (im2col[-smem] + permute + split-K GEMM, M39/M37):

| Shape | permute (ms) | im2col (ms) | splitk_gemm (ms) | permute % | im2col % | gemm % |
|---|---:|---:|---:|---:|---:|---:|
| large_channel | 1.019 | 4.105 | 4.916 | 10.2% | 40.9% | 49.0% |
| batch_32 | 0.532 | 2.110 | 2.448 | 10.3% | 40.9% | 47.4% |
| batch_64 | 1.020 | 4.139 | 4.915 | 10.1% | 41.1% | 48.8% |
| batch_128 | 2.032 | 8.290 | 10.132 | 9.9% | 40.5% | 49.5% |

**GEMM/split-K reassessment (Sections 12/13 of the brief)**: post-M39, the
GEMM stage (fused or split-K) is the single largest dWeight sub-stage
everywhere -- 49-89% of the pipeline, a clear reversal from M37's own finding
(`im2col`+`permute` at 54-63%, more than the GEMM). `im2col`'s M39
shared-memory optimization worked well enough that it is no longer the
dominant cost anywhere; `permute` remains cheap everywhere (10-12%), as it
has been since M34. This means **the GEMM stage, not im2col/permute, is now
the correct place to look for further dWeight-internal gains** -- but see
the Roofline section below for why that lead does not translate into a clean
M41 target.

## Roofline / Resource Analysis (fresh ceilings, this session: 104.67 GFLOP/s compute, 15.09 GB/s bandwidth)

| Shape | dInput %ceiling / class | dWeight %ceiling / class | forward %ceiling / class |
|---|---|---|---|
| mnist_conv1 | 8.8% / mixed | 3.3% / mixed | 10.7% / mixed |
| mnist_conv2 | 27.8% / mixed | 22.6% / mixed | 18.0% / mixed |
| large_channel | 34.0% / **compute_bound** | 44.0% / **compute_bound** | 17.0% / mixed |
| large_spatial | 28.5% / mixed | 25.1% / mixed | 16.6% / mixed |
| batch_32 | 29.6% / mixed | 43.4% / **compute_bound** | 17.5% / mixed |
| batch_64 | 33.3% / **compute_bound** | 43.9% / **compute_bound** | 17.0% / mixed |
| batch_128 | 34.6% / **compute_bound** | 43.2% / **compute_bound** | 17.0% / mixed |

**Every forward measurement classifies `mixed_or_ambiguous` at 10.7-18.0% of
ceiling** -- at every shape tested, further from its practical ceiling than
either backward Conv2d kernel at the same shape, despite computing the
identical total FLOP count (`roofline.py`'s own documented symmetry:
`flops_conv2d_forward == flops_conv2d_dinput == flops_conv2d_dweight` for a
given shape). This is the central, load-bearing finding of this milestone:
**identical arithmetic work, meaningfully lower achieved efficiency, on a
kernel no prior milestone has touched.**

**dWeight's GEMM-dominated shapes (`blocks_y>=2`) are compute-bound at
43-44% of ceiling** -- consistent with, and a modest improvement on, M37's
own 32.6-32.9% finding at the same shapes, confirming the split-K occupancy
fix continues to hold. **`blocks_y==1` shapes remain further from ceiling**
(22.6-25.1%, `mixed_or_ambiguous`) -- consistent with M38's own register-
pressure finding (half-fused kernel is register-limited to ~5 resident
blocks/SM vs. 8 for the plain split-K kernel) and with `k_matmul`'s
documented poor tile efficiency at `Cout<=16`/small-`K` shapes (M34's
"Limitations": "a differently-tiled or batched-small-GEMM strategy could
help here, but is out of scope" -- still true, still out of scope: it would
require a second GEMM design, not a dispatch change).

## Non-Conv2d Operations (this session, real MNIST CNN, batch=64)
| Phase | % of step | Verdict |
|---|---:|---|
| CrossEntropy backward | 0.89% | negligible, unchanged since M31 |
| optimizer (Adam) | ~5.0% of full 5-phase step (not in the forward+backward-only ranking base) | negligible, unchanged since M17 |
| transfer | ~2.4% of full 5-phase step | negligible |
| `max_pool2d` (fwd+bwd) | 9.30% | memory_bandwidth_bound both directions, unchanged since M15 |
| `relu` (fwd+bwd) | 10.20% | mixed_or_ambiguous, unchanged |
| `Linear`/`@` (fwd+bwd) | 10.91% | mixed_or_ambiguous, matmul already M11-tiled |

None of these individually or combined justify a dedicated M41 investigation
-- all are small, unchanged, and (`max_pool2d`) already correctly classified
as bandwidth-bound with no evidence of inefficiency.

## Asynchronous Pipeline Analysis
Compute-stream utilization (`pipeline_profile._profile_async_epoch`, fresh
this session, prefetch_size=2):

| Batch | samples/sec | compute-stream utilization |
|---|---:|---:|
| 32 | 2,460 | 91.6% |
| 64 | 3,168 | 93.6% |
| 128 | 6,783 | 87.5% |

Prefetch-depth sweep (batch=64): depth 1 -> 92.1% util / 3,547 samples/sec,
depth 2 -> 93.4% util / 3,091 samples/sec, depth 3 -> 92.6% util / 3,328
samples/sec -- all three depths keep the compute stream continuously fed
(87-94% utilization at every configuration tested, batch size included).
**The compute stream is already well-fed; no further DataLoader/prefetch
work is justified** (Section 15's explicit stop condition: "if the compute
stream is already continuously fed, do not recommend further DataLoader
work"). The modest, noisy differences between prefetch depths are within
this GPU's documented run-to-run variance, not a signal that a specific
depth is meaningfully better.

## Amdahl Analysis
Using this session's own measured full (forward+backward) compute-step
composition (MNIST, batch=64) and this milestone's own measured per-shape
splits at MNIST's real two conv layers:

| Component | Fraction of step | 1.5x | 2.0x | 3.0x |
|---|---:|---:|---:|---:|
| conv2d backward total | 48.37% | 1.192x | 1.319x | 1.476x |
| **dWeight** | 29.87% | 1.111x | 1.176x | 1.249x |
| **conv2d forward** | **17.02%** | **1.060x** | **1.093x** | **1.128x** |
| dInput | 15.56% | 1.055x | 1.084x | 1.116x |
| matmul (Linear) backward | 5.96% | 1.020x | 1.031x | 1.041x |

`dWeight` has the largest raw Amdahl ceiling of any single sub-component
(driven by its now-larger MNIST-scale share -- see **Current Bottleneck
Table** above), but it is also the sub-component this codebase has already
spent three milestones (M37/M38/M39) optimizing, and it is now compute-bound
at 43-44% of ceiling at the larger shapes -- both facts lower confidence that
a *fourth* milestone finds another comparably-sized win. `conv2d forward`'s
Amdahl ceiling is smaller in absolute terms, but it starts from a much lower
efficiency floor (10.7-18.0% vs. 22.6-44.0% of ceiling) and has never been
attempted -- the headroom×confidence product (Section 8's explicit ranking
criterion) favors it over a fourth dWeight pass. See **Candidate Ranking**
below.

## Candidate Ranking

| Rank | Candidate | Time Share | Bottleneck Class | Estimated Headroom | Confidence | M41 Recommendation |
|---|---|---:|---|---|---|---|
| 1 | **`k_conv2d_forward`** | 17.0% | mixed_or_ambiguous, naive one-thread-per-output-element, zero reuse | **High** -- 10.7-18.0% of ceiling, same FLOPs as dInput/dWeight which reach 22-44% | High -- im2col+GEMM technique already built, tested, and productionized for dWeight (M34/M39); `im2col_smem`/`cf_matmul_splitk_*` reusable unmodified | **Investigate (selected, see below)** |
| 2 | dWeight GEMM (`blocks_y>=2`, split-K) | 29.9%* | compute_bound, 43-44% of ceiling | Low-medium -- already near the "compute_bound" heuristic threshold; further gains need a new GEMM tiling design | Low -- would require a second GEMM implementation, conflicting with Forge's one-portable-GEMM architecture stance (ADR-004) | Do not pursue in M41 |
| 3 | dWeight half-fused GEMM (`blocks_y==1`) | (subset of #2) | mixed_or_ambiguous, 22.6-25.1% of ceiling | Medium -- further from ceiling than split-K shapes | Low -- M34/M38 already identified this as a small-tile/register-limited regime with no clean fix found in two prior attempts | Do not pursue in M41 |
| 4 | dInput (channel-fused) | 15.6% | compute_bound (large shapes) / mixed (small), 8.8-34.6% of ceiling | Medium | Low -- M36 already tried 3 structurally different candidates; no new structural idea identified this milestone | Do not pursue in M41 |
| 5 | dWeight below-threshold (block-reduce, `mnist_conv1`-shape) | (subset of #2, MNIST-dominant) | 3.3% of ceiling (serial-reduction-latency-bound, not GEMM-shaped) | Low | Low -- M33 already tested warp/block cooperative alternatives; current kernel already within ~1-2% of the best alternative found | Do not pursue in M41 |
| 6 | `max_pool2d`/`relu`/optimizer/transfer/CrossEntropy | <=10.9% each | already near ceiling or negligible | Low | n/a | Do not pursue |
| 7 | Async pipeline / prefetch depth | n/a | 87-94% compute-stream utilization at every configuration | Low | n/a | Do not pursue |

*dWeight's 29.9% figure is MNIST-weighted and dominated by the below-threshold
`mnist_conv1` kernel (Rank 5); its `blocks_y>=2` GEMM path (Rank 2) is the
only part of dWeight with a plausible further-optimization story, and that
story is weak (see Rank 2's row).

## M41 Recommendation

**Target: `k_conv2d_forward` (`forge/backend/cuda/kernels.cu`), via im2col +
Forge's existing tiled/split-K GEMM.**

1. **Exact kernel/algorithm**: `k_conv2d_forward` (one thread per
   `(n,co,ho,wo)` output element, serial `Cin*KH*KW` loop reading `x`/`w`
   directly from global memory, unchanged since Milestone 15).
2. **Exact bottleneck being addressed**: identical total FLOP count to
   `dInput`/`dWeight` at a given shape, yet only 10.7-18.0% of the practical
   compute ceiling at every representative shape measured -- the lowest
   ceiling-fraction of any Conv2d-adjacent kernel in this report, and the
   only one with genuinely *no* explicit memory-reuse mechanism (no shared
   memory, no GEMM tiling, no register-blocking).
3. **Why the existing implementation is inefficient**: mathematically
   equivalent to a naive, untiled GEMM (`out_mat[Cout,M] = weight_mat[Cout,K]
   @ Xcol[K,M]`, `K=Cin*KH*KW`, `M=N*Hout*Wout`) with zero of the reuse
   Forge's own M11 tiled `k_matmul` was written specifically to capture --
   the exact class of inefficiency Forge already diagnosed and fixed once
   for plain matmul (M11) and again, structurally differently, for `dWeight`
   (M34's im2col+GEMM). `k_conv2d_forward` is the one remaining Conv2d
   kernel that never received either treatment.
4. **Why previous milestones did not already solve it**: M31-M39 scoped
   every investigation to `conv2d_backward` specifically (`dInput`/`dWeight`/
   `dBias`), following each milestone's own measured "what dominates
   backward" finding -- forward Conv2d was never the measured bottleneck at
   the time (M21: 17.3% of the full unoptimized step; this milestone: 17.0%
   of a step whose backward phase has since shrunk considerably), so it was
   never in scope for a "one thing at a time" milestone.
5. **Why the proposed direction has plausible headroom**: the technique is
   not novel -- it is the same reformulation M34 already measured 1.12-1.59x
   faster than a naive per-thread kernel for `dWeight`, and M39's
   `im2col_smem`/`recommended_num_k_splits`/`cf_matmul_splitk_*` are already
   built, tested, and require no new kernel code to reuse for forward's GEMM
   orientation (`weight_mat` as GEMM operand A, an `Xcol` as operand B --
   note forward's GEMM has a *small* reduction dimension `K=Cin*KH*KW`
   unlike dWeight's huge-`M`-as-reduction shape, so forward likely needs
   little or no split-K; `blocks_x = ceil(M/16)` is very large at every
   representative shape, which should give the forward GEMM *better*
   built-in occupancy than dWeight's GEMM ever had -- a plausible reason
   forward may respond even better than dWeight did).
6. **Representative shapes**: reuse all 7 existing shapes
   (`conv2d_backward_profile.SHAPES` + `BATCH_SWEEP_BASE`/`BATCH_SIZES`) --
   already exercises small (`mnist_conv1`, `Cout=8`), medium
   (`mnist_conv2`), and large (`large_channel`/`large_spatial`/`batch_*`)
   regimes without inventing new shapes.
7. **Candidate alternatives to benchmark first (PROFILE -> ANALYZE ->
   DESIGN -> BENCHMARK step of M41)**: (a) a fresh `nvcc -Xptxas -v` pass on
   the unmodified `k_conv2d_forward` (not yet done this milestone -- M40 is
   measurement-only at the roofline/wall-clock level, not the
   register/occupancy level, for this specific kernel) to confirm no
   simpler M32-style division-hoist or M36-style register-reuse fix is
   available before committing to a full GEMM rewrite; (b) im2col + plain
   `cf_matmul_*` (M34-style, simplest); (c) im2col-or-`im2col_smem` +
   `cf_matmul_splitk_*` (M39-style, only if (a)'s occupancy analysis shows
   `blocks_x*blocks_y` genuinely under-occupies the device at some shape,
   which is not expected given forward's much larger `blocks_x`).
8. **Acceptance threshold**: matching every prior accepted-candidate
   milestone's bar (M34/M36/M38/M39) -- a reproducible, same-session,
   CUDA-event-measured speedup at a majority of the 7 representative shapes
   with no shape regressing beyond measurement noise, verified against
   `CPUBackend.conv2d` (forward only -- backward is untouched) across the
   existing stride/padding/kernel-size combinations `tests/test_cuda_conv.py`
   already covers.
9. **What must remain untouched**: `k_conv2d_backward_input`
   (M36), `dweight_halffused_gemm_splitk` (M38), `dweight_im2col_smem_gemm_splitk`
   (M39), `cf_conv2d_backward_weight_*`/`_reduce` (M21), `k_matmul` itself
   (M11 -- only its already-existing `cf_matmul_*`/`cf_matmul_splitk_*`
   entry points may be *called*, never modified), `CrossEntropy` fusion
   (M31), the M25 caching allocator, and the M27-M30 stream/transfer/
   prefetch pipeline.

## Exclusions
M41 should explicitly **not** touch: `dWeight`'s split-K/half-fused GEMM
paths (already compute-bound, 43-44% of ceiling, would require a new GEMM
tiling design); `dWeight`'s below-256-threshold block-reduce kernel (M33
already found no better alternative); `dInput`'s channel-fused kernel (M36
already explored three structural alternatives); `k_matmul`'s own tiling
(a fourth consecutive milestone constraint, per M34's "no second GEMM
implementation" rule -- forward reuses the *existing* `cf_matmul_*`/
`cf_matmul_splitk_*` entry points exactly as dWeight does, never a new GEMM
kernel); `max_pool2d`, `relu`, `CrossEntropy`, `Adam`, transfers, or the
async prefetch pipeline (all confirmed small or already well-utilized this
milestone).

## Limitations
- **No `nvcc -Xptxas -v` register/occupancy data was collected for
  `k_conv2d_forward` this milestone** -- Section 22's stop condition
  ("one bottleneck clearly dominates and has plausible headroom") was met at
  the wall-clock/roofline level before this milestone reached that deeper
  profiling step; M41's own PROFILE phase should collect it before
  committing to a specific fix (see **M41 Recommendation**, candidate (a)).
- **The MNIST-weighted Amdahl fractions use this session's own two-conv-shape
  split** (`mnist_conv1`+`mnist_conv2`, this document's **Current Bottleneck
  Table**) applied proportionally to `mnist_profile`'s aggregate
  `backward:conv2d` number, rather than re-instrumenting `mnist_profile.py`
  itself to report `dInput`/`dWeight`/`dBias` separately -- consistent with
  every prior milestone's "aggregate by op name" convention
  (`mnist_profile._profiled_run_backward` groups by `node.name`, not kernel
  identity), and cross-checked for consistency (summed isolated numbers,
  5.09ms, closely match the aggregate walker's 5.26ms).
- **Run-to-run hardware variance on this laptop GPU remains real** (every
  prior milestone's own documented characteristic) -- all comparative
  numbers in this document come from one continuous session specifically to
  control for it, but absolute numbers should not be compared against
  numbers from a different session (e.g. M35's or M39's own archived JSON)
  without accounting for that documented variance.
- **The roofline model's `mixed_or_ambiguous` classification is a coarse,
  documented heuristic** (`NEAR_CEILING_FRACTION = 0.30`), not a precise
  measurement of what specifically limits `k_conv2d_forward` -- this
  milestone identifies *that* forward is inefficient and *why* (zero
  explicit reuse, same class of problem Forge already solved twice
  elsewhere) but does not claim a precise mechanistic diagnosis (local
  memory spill vs. cache-miss rate vs. instruction throughput) the way
  M32/M36's `nvcc -Xptxas -v` analysis did for `dInput` -- that is exactly
  M41's first step, not repeated here.
- **This milestone did not implement or benchmark any forward-Conv2d
  candidate** -- per the M40 brief's explicit "M40 ends after the
  bottleneck has been measured, ranked, and documented" instruction (Section
  23) and "Do not implement M41's optimization" (Section 19).

## Tests / Verification
- Full existing suite unaffected: no production `forge/` code was modified.
  `python -m pytest tests/ -q` passes with the same test count as before this
  milestone (no new tests were required or added -- Section 20 of the brief
  permits, but does not require, benchmark-only tests, and no production
  behavior changed for any test to newly cover).
- `benchmarks/m40_bottleneck_recharacterization.py` (new) exercises every
  currently-dispatched production dWeight function
  (`cf_conv2d_backward_weight_*`, `dweight_halffused_gemm_splitk`,
  `dweight_im2col_smem_gemm_splitk`'s constituent stages, `dInput`'s
  `cf_conv2d_backward_input_*`) directly, and its dispatch-decision helper
  (`_dispatch_decision`) was verified by hand against `CUDABackend.
  conv2d_backward`'s own dispatch code before any timing was trusted (the
  **Dispatch Verification** table above matches `backend.py`'s logic
  exactly, closing the "measuring the wrong implementation" risk Section 6
  of the brief and M37's own history both flag).
- `python -m benchmarks.m40_bottleneck_recharacterization` produces
  `benchmarks/results/m40_bottleneck_recharacterization.json` (this
  document's primary evidence source) and a human-readable console report.
- Re-ran `python -m benchmarks.m35_hardware`, `python -m benchmarks.m35_mnist`,
  and `pipeline_profile._profile_async_epoch` fresh in the same session
  (folded into the new script rather than as separate invocations) --
  all three already-existing tools, unmodified.

## Hardware
NVIDIA GeForce 940MX, Compute Capability 5.0, CUDA 12.6, Driver 582.53, 2GB
VRAM, 3 SMs, 384 CUDA cores -- every measurement in this document was
collected on this real, verified development GPU. No simulated or emulated
CUDA behavior.

## Suggested Commit Message
```
docs: M40 post-M39 CUDA bottleneck re-characterization

Fresh, same-session measurement of the current (post-M39) CUDA training
pipeline finds no single dominant kernel remains: conv2d backward's
dInput/dWeight sub-kernels are now close in cost and both compute-bound
(43-44% of ceiling) at large shapes, while dWeight now dominates at
MNIST's own real shapes (driven by the untouched below-threshold
block-reduce kernel). The clearest remaining optimization target is
k_conv2d_forward: identical FLOP count to dInput/dWeight but only
10.7-18.0% of the practical compute ceiling, and the only major Conv2d
kernel never restructured by M31-M39's work. Recommends M41 apply the
already-validated im2col + existing tiled/split-K GEMM technique to
forward Conv2d. Measurement-only: no production CUDA code changed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TxDxK1bqZfyvmuFQrJnS4Z
```
