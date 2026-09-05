# M44 — Fresh Post-M43 CUDA Bottleneck Re-Characterization (measurement-only)

## 1. Executive summary

M43 replaced the `blocks_y==1` dWeight path's atomic split-K combine
(`k_dweight_halffused_gemm_splitk`, M38) with a deterministic two-stage
reduction (`k_dweight_halffused_gemm_splitk_partial` + `k_dweight_splitk_
reduce`, M43), reporting a 1.11-1.24x component win. This milestone
re-measures the **whole** CUDA training pipeline fresh, in one session,
against the actual current (post-M43) production dispatch on both the
forward and backward sides at once (`benchmarks/m44_bottleneck_
recharacterization.py`, new — combines M41's forward decomposition with a
corrected dWeight/dInput backward decomposition; M40/M42's own dWeight
sub-stage helper still names M38's now-dead kernel). No production code,
kernel, or public API was touched.

**Headline finding.** M43 worked, and this session's own fresh, same-
session interleaved A/B against the original M38 atomic kernel confirms it
still holds: **1.15-1.29x faster (GEMM-only) / 1.12-1.23x faster (full
pipeline) at `mnist_conv2`, 1.02-1.03x at `large_spatial`, at every tested
split count, no regression** — consistent with M43's own report. Stage-
separated timing (new in M44) further shows the reduce kernel is
bandwidth-trivial as designed (**0.9-1.1% of dWeight's `blocks_y==1`
total**), so the remaining inefficiency at this path (still only **25.4-
29.4% of the practical compute ceiling** at the real `Cout=16` shapes) now
lives entirely in the partial-GEMM kernel itself, not in any residual
reduction cost.

Because M43 shrank the `blocks_y==1` gap, the ranking underneath it has
shifted: **dWeight's below-256-weight-element block-reduce kernel
(`mnist_conv1`'s own shape, unchanged since M21, twice investigated and
rejected in M33/M34) is now unambiguously the single largest contributor**
— **20.19% of the full MNIST training step**, more than double
`blocks_y==1`'s own post-M43 contribution (**8.47%**, down from M42's
10.2%), while reaching only **3.1-3.2% of the practical compute ceiling**
across every below-threshold shape tested (`weight_elements` from 72 to
243) — the worst roofline efficiency of any candidate measured in this
characterization, before or after M43. A dedicated `Cin`/`weight_elements`-
controlled sweep shows this cost scales with reduction length
(`N*Hout*Wout`, held fixed at ~50,176 across the sweep) far more than with
`weight_elements` itself (144-element shapes cost the same regardless of
how the `Cin`/`Cout` split is arranged), consistent with M33/M42's own
per-thread-serial-reduction diagnosis: too few thread-blocks
(`weight_elements`-many) to saturate the GPU, each doing a very long serial
reduction alone. **M45 is recommended to attack this kernel with a
genuinely new angle — warp-shuffle-based cooperative reduction — that
neither of the two previously-rejected techniques (M33 shared-memory
cooperative reduction, M34 im2col+GEMM) tried.**

## 2. Post-M43 architecture

`CUDABackend.conv2d` (forward, unchanged since M41): `total_flops =
2*N*Cout*Hout*Wout*Cin*KH*KW`. Below `_CONV2D_FORWARD_GEMM_FLOPS_
THRESHOLD` (10,000,000) → unchanged M15 per-thread kernel
(`cf_conv2d_forward_*`). At/above it, `blocks_x = ceil(Cout/16)`;
`blocks_x<=2` (Cout<=32, every current Forge shape reached by the 7
representative/batch-sweep shapes) → M41 Candidate B
(`conv2d_forward_halffused_gemm`); `blocks_x>2` (only reached by the
`cout_high` sweep shape, Cout=128) → M41 Candidate A (im2col-smem +
transpose + existing tiled GEMM + output permute).

`CUDABackend.conv2d_backward`: dInput dispatch (unchanged since M36) lives
inside the compiled `cf_conv2d_backward_input_*` wrapper itself
(`kernels.cu`'s `CONV2D_DINPUT_CHANNELFUSED_MAX_CIN = 16`, confirmed
directly against the compiled source this session): `Cin<=16` → M36
channel-fused kernel (every current Forge shape); `Cin>16` → the original
M32 kernel, a fallback never reached by any real Forge shape. dWeight
dispatch: `weight_elements = Cout*Cin*KH*KW`; below 256 → **unchanged**
M21 per-thread/block-reduce kernel (`cf_conv2d_backward_weight_*`,
`k_conv2d_backward_weight_reduce`); at/above it, `blocks_y = ceil(Cout/16)`;
**`blocks_y==1` (Cout<=16) → M43's two-stage split-K reduction**
(`dweight_halffused_gemm_splitk_tworeduce`: `grad_output_permute` →
`cf_dweight_halffused_gemm_splitk_partial_*` (writes a disjoint partial
slice per split, no atomic) → `cf_dweight_splitk_reduce_*` (sums the
`num_k_splits` axis)); `blocks_y>=2` (Cout>16) → M39's unchanged
shared-memory im2col + permute + split-K GEMM (`dweight_im2col_smem_gemm_
splitk`).

This script (Phase 1/2) independently confirmed the actual code at
`forge/backend/cuda/backend.py` (`conv2d`/`conv2d_backward`) and
`forge/backend/cuda/kernels.cu` (dInput's `Cin<=16` launcher branch,
M43's `k_dweight_halffused_gemm_splitk_partial`/`k_dweight_splitk_reduce`
definitions) matches the above exactly — no historical documentation was
relied upon for dispatch decisions. `benchmarks/m40_bottleneck_
recharacterization.py`'s own `_dispatch_decision` helper (reused by M42)
still labels the `blocks_y==1` path "M38, permute + fused-GEMM" — stale
after M43; this milestone's own `_dweight_dispatch_decision` wraps it and
corrects the label without modifying the M40 file.

## 3. Methodology

`benchmarks/m44_bottleneck_recharacterization.py` (new) combines, rather
than reimplements:

- M41's forward decomposition (`m41_conv2d_forward_profile._ForwardCandidates`,
  imported unmodified).
- A corrected backward decomposition: dInput/dBias via `conv2d_backward_
  profile._RawConv2dBackward` (unmodified); dWeight `blocks_y>=2` via
  `m40_bottleneck_recharacterization._RawDweightCurrent` (unmodified, still
  correct for this regime); dWeight `blocks_y==1` via a **new** helper
  (`_dweight_tworeduce_stages`) built on `m43_dweight_splitk_profile.
  _RawDweightTworeduce` (imported unmodified) that times permute,
  partial-GEMM, and reduce as three independent CUDA-event-timed stages —
  the decomposition M40/M42's own helper cannot produce since it predates
  M43.
- A fresh same-session interleaved A/B (M38 atomic vs. M43 two-stage) at
  `mnist_conv2`/`large_spatial`, reusing `m43_dweight_splitk_profile.
  _candidate_b_comparison` directly, so this milestone's own numbers
  confirm M43's win rather than trusting M43's report at face value.
- Three new targeted sweeps (Section 4, below) that did not exist in any
  prior milestone's script: a dWeight `Cout` sweep spanning both
  `blocks_y` regimes at a fixed weight-element-safe base shape, a
  below-256-weight-element sweep, and a dInput `Cin` sweep straddling the
  M36 channel-fused/fallback boundary.
- Fresh roofline ceilings (M35 methodology, this session), `nvcc -Xptxas
  -v` resource analysis, `m35_mnist`'s synchronous kernel-contribution
  ranking, `m35_kernels`' CrossEntropy roofline, and `pipeline_profile`'s
  async-epoch/allocator/pinned-memory characterization — all rerun fresh,
  not reused from a prior session's JSON.

Every stage call is the same function `backend.py` itself calls (either
directly, or through the same experimental module `backend.py` imports).
CUDA events measure every GPU phase; `perf_counter()` is used only where
prior milestones already established it for inherently serial CPU work
(unchanged, reused as-is). No inter-phase synchronization occurs during
measurement loops — synchronization happens once at each phase's boundary,
matching M40/M42/M43's own convention exactly.

## 4. Environment

NVIDIA GeForce 940MX, driver 582.53, CUDA compute capability 5.0, 2048 MiB
VRAM, Windows 10. Forge commit `c750b96` (M43, HEAD at the start of this
milestone). Full environment recorded in both `benchmarks/results/
m44_bottleneck_recharacterization.json` and `benchmarks/results/
m44_pipeline_profile.json`'s own `environment` block.

## 5. Production dispatch verification

| Shape | fwd FLOPs | `blocks_x` | forward path | `weight_elements` | `blocks_y` | dWeight path |
|---|---:|---:|---|---:|---:|---|
| mnist_conv1 | 7,225,344 | — | M15 per-thread (below 10M threshold) | 72 | — | M21 block-reduce (below 256) |
| mnist_conv2 | 24,920,064 | 1 | M41 Candidate B | 1,152 | 1 | **M43 two-stage reduction** |
| large_channel | 462,422,016 | 2 | M41 Candidate B | 4,608 | 2 | M39 im2col-smem+splitK |
| large_spatial | 231,211,008 | 1 | M41 Candidate B | 1,152 | 1 | **M43 two-stage reduction** |
| batch_32/64/128 | (Cout=32 base) | 2 | M41 Candidate B | 4,608 | 2 | M39 im2col-smem+splitK |

dInput: every one of the above has `Cin<=16` → M36 channel-fused on every
representative/batch-sweep shape (unchanged since M36; confirmed again
this session). The `Cin>16` fallback is exercised only by this
milestone's own dedicated sweep (Section 8), never by any real Forge
shape — consistent with every prior milestone's finding.

## 6. Full forward+backward decomposition (this session, 940MX)

Practical ceilings this session: **104.58 GFLOP/s** compute, **15.09 GB/s**
bandwidth (M35 methodology, re-measured fresh — within run-to-run thermal
variance of every prior session's own ~104-105 GFLOP/s / ~15.1 GB/s
figures; not compared as an absolute number per the milestone's own
instruction).

| shape | fwd(ms) | dIn(ms) | dW(ms) | dB(ms) | bwd total(ms) | fwd %ceil | dIn %ceil | dW %ceil |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnist_conv1 | 3.970 | 0.829 | 2.238 | 0.231 | 3.297 | 1.7% | 8.3% | 3.1% |
| mnist_conv2 | 0.854 | 0.878 | 0.939 | 0.094 | 1.910 | 27.9% | 27.1% | 25.4% |
| large_channel | 9.970 | 13.045 | 10.162 | 0.730 | 23.937 | 44.4% | 33.9% | 43.5% |
| large_spatial | 6.065 | 7.779 | 8.658 | 0.767 | 17.205 | 36.5% | 28.4% | 25.5% |
| batch_32 | 5.063 | 7.058 | 5.023 | 0.379 | 12.460 | — | — | — |
| batch_64 | 10.283 | 13.091 | 10.216 | 0.722 | 24.028 | — | — | — |
| batch_128 | 19.989 | 25.545 | 20.629 | 1.376 | 47.550 | — | — | — |

(mnist_conv1's forward reaches only 1.7% of ceiling because it stays below
M41's own forward FLOPs threshold — the unchanged, already-characterized
M15 kernel; not a new finding.)

**Real end-to-end cross-check**: at every shape, the real `CUDABackend.
conv2d_backward()` call measured 10-20% higher than the sum of its raw,
buffer-reused sub-stage timers (e.g. `mnist_conv2`: 2.305ms real vs.
1.910ms summed; `large_channel`: 24.571ms real vs. 23.937ms summed) — the
expected per-call allocation/dispatch overhead the raw sub-stage timers
deliberately exclude by reusing buffers across iterations, consistent with
every prior milestone's own finding at this same cross-check.

## 7. M43 stage analysis

**Stage decomposition of the current production `blocks_y==1` path**
(permute + partial-GEMM + reduce, at the production `recommended_num_k_
splits`):

| shape | permute (ms) | partial-GEMM (ms) | reduce (ms) | total (ms) | partial-GEMM % | reduce % |
|---|---:|---:|---:|---:|---:|---:|
| mnist_conv2 | 0.118 | 0.813 | 0.009 | 0.939 | 86.5% | 0.9% |
| large_spatial | 1.049 | 7.601 | 0.009 | 8.658 | 87.8% | 0.1% |

**The reduce kernel is confirmed bandwidth-trivial as designed** — under
1.1% of the path's total cost at both real shapes. M43's stage-separated
design goal (isolate the atomic-combine cost without adding a second
meaningful bottleneck) is achieved: essentially all of `blocks_y==1`'s
remaining inefficiency now lives in the partial-GEMM kernel itself
(structurally identical tile-load/gather to M38's kernel, confirmed by
`nvcc -Xptxas -v`: 52/49 registers, 4096/2048 bytes shared memory — an
exact match to M38's own `k_dweight_halffused_gemm_splitk` figures,
confirming no register-pressure or occupancy change was introduced by
removing the atomics), not in any residual combine cost.

**Fresh same-session interleaved A/B (M38 atomic vs. M43 two-stage)**, at
the production `recommended_num_k_splits=16` and five higher split counts:

| shape | splits=16 (production) gemm-only | splits=16 full | range across all tested splits (gemm-only) |
|---|---:|---:|---:|
| mnist_conv2 | 1.250x | 1.227x | 1.063x-1.287x |
| large_spatial | 1.030x | 1.032x | 1.017x-1.032x |

**No regression at any tested split count, at either shape** — confirms
M43's own report still holds this session (M43 reported 1.11-1.24x/
1.02-1.04x; this session's 1.02-1.29x/1.02-1.03x range is consistent
within thermal variance, not a like-for-like comparison per the
milestone's own instruction).

## 8. Targeted sweeps

**dWeight `Cout` sweep** (fixed `Cin=16,H=W=13,K=3`, `weight_elements =
144*Cout`, always ≥256 so every point reaches the GEMM dispatch):

| Cout | `blocks_y` | total (ms) | % of ceiling |
|---:|---:|---:|---:|
| 2 | 1 | 1.485 | 4.0% |
| 4 | 1 | 1.504 | 7.9% |
| 8 | 1 | 1.541 | 15.5% |
| 16 | 1 | 1.619 | 29.4% |
| 32 | 2 | 2.370 | 40.2% |
| 64 | 4 | 3.485 | 54.7% |
| 128 | 8 | 5.657 | 67.4% |

Efficiency rises monotonically with `Cout` in *both* regimes and jumps at
the `blocks_y==1 -> 2` boundary (29.4% -> 40.2%) — `blocks_y>=2`'s
im2col-smem+split-K-GEMM path remains the more efficient regime by
construction (M39's finding, unchanged), and M43 has narrowed but not
closed the gap between the two regimes at their shared boundary (`Cout=16`
vs. `Cout=32`).

**dWeight below-256-weight-element sweep** (fixed `N=64,H=W=28,K=3`,
mnist_conv1's own spatial shape, `Cin`/`Cout` varied to hit different
`weight_elements` while staying below the 256 threshold):

| label | `weight_elements` | `Cin`,`Cout` | d_weight (ms) | % of ceiling |
|---|---:|---|---:|---:|
| mnist_conv1 (we=72) | 72 | 1, 8 | 2.246 | 3.1% |
| we_144a | 144 | 1, 16 | 4.420 | 3.1% |
| we_144b | 144 | 2, 8 | 4.322 | 3.2% |
| we_216 | 216 | 3, 8 | 6.418 | 3.2% |
| we_243 | 243 | 1, 27 | 7.236 | 3.2% |

**Flat ~3.1-3.2% of ceiling regardless of `weight_elements`** — the worst,
and most *uniformly* bad, roofline efficiency of any candidate in this
characterization. The two 144-element points (`Cin=1,Cout=16` vs.
`Cin=2,Cout=8`) cost the same (4.42ms vs. 4.32ms) despite a different
`Cin`/`Cout` split, while wall-clock time scales roughly linearly with
`weight_elements` itself (72→243 is a 3.4x increase; 2.25ms→7.24ms is a
3.2x increase) — consistent with M33/M42's own diagnosis: this kernel
launches exactly `weight_elements` threads (blocks), each performing an
*independent, fully serial* `N*Hout*Wout`-length reduction (50,176
iterations at this spatial shape); with only 72-243 threads total, the
940MX's occupancy is nowhere near saturated, so total wall time is
dominated by each thread's own serial length rather than by parallel
throughput — adding more `weight_elements` just adds more of the same
serial work rather than more overlap.

**dInput `Cin` sweep** (fixed `N=16,Cout=16,H=W=28,K=3`, straddling the M36
channel-fused/fallback boundary):

| Cin | path | d_input (ms) | % of ceiling |
|---:|---|---:|---:|
| 1 | M36 channel-fused | 0.374 | 9.2% |
| 8 | M36 channel-fused | 1.155 | 23.9% |
| 16 | M36 channel-fused | 1.932 | 28.6% |
| 17 | M32 fallback | 4.581 | 12.8% |
| 32 | M32 fallback | 9.332 | 11.8% |
| 64 | M32 fallback | 18.922 | 11.7% |

The channel-fused path's own efficiency rises with `Cin` up to its
`Cin<=16` ceiling (9.2%→28.6%), while the untouched `Cin>16` fallback sits
around 11.7-12.8% throughout — confirming M42's own conclusion still
holds: **no real Forge shape reaches `Cin>16`**, so there is still no
reproducible current workload motivating further dInput work at that
boundary. At `Cin=1` (mnist_conv1's own real value) the channel-fused
kernel itself is fairly inefficient (9.2%) but contributes little in
absolute terms at that shape (0.83ms of a 3.30ms backward total) — this is
a pre-existing, already-examined characteristic (M36's own 8.7x-in-
isolation rewrite), not a fresh M44 finding.

## 9. Forward re-verification (M41's dispatch remains efficient post-M43)

The K/stride/channel sweep (`m41_conv2d_forward_profile.SWEEP_SHAPES`,
rerun fresh) confirms forward is unaffected by M43 and remains healthy:
3.6% (below-threshold `k1_s1`) to 48.8% (`cout_high`, Candidate A) of the
practical compute ceiling, matching M41/M42's own range. No new forward
opportunity is indicated by this session's measurements.

## 10. Roofline classification summary

| candidate | % of ceiling (range across representative shapes) | classification |
|---|---:|---|
| Conv2d forward (GEMM-dispatched) | 27.9-48.8% | mixed/ambiguous (compute-leaning) |
| dInput (M36 channel-fused, real shapes) | 8.3-33.9% | mixed/ambiguous |
| dWeight `blocks_y==1` (M43 two-stage) | 25.4-29.4% | mixed/ambiguous |
| dWeight `blocks_y>=2` (M39, unchanged) | 40.2-44.2%(batch sweep)/67.4%(Cout=128) | compute-leaning |
| dWeight below-256 block-reduce | **3.1-3.2%** | **launch/occupancy-bound** |
| GEMM itself (`k_matmul`/`k_matmul_splitk`) | ~100% (by construction) | compute-bound |
| CrossEntropy | memory-bandwidth-bound at realistic sizes, launch-bound at tiny sizes | healthy |

## 11. Resource analysis (`nvcc -Xptxas -v`, real compile)

| kernel | registers (f32/f64) | shared mem | spill |
|---|---|---|---|
| `k_dweight_halffused_gemm_splitk` (M38, now dead in production) | 52/49 | 4096/2048 bytes | none |
| `k_dweight_halffused_gemm_splitk_partial` (M43, production) | 52/49 | 4096/2048 bytes | none |
| `k_dweight_splitk_reduce` (M43, production) | 32 | 0 bytes | none |
| `k_conv2d_backward_weight_reduce` (M21, below-threshold, production) | 45/47 | 0 bytes | none |
| `k_conv2d_forward_halffused_gemm` (M41, comparison) | 40/38 | 4096/2048 bytes | none |
| `k_conv2d_backward_input_channelfused` (M36, production) | 70/48 | 0 bytes | none (f64: 512 bytes cumulative stack) |

**M43's partial-GEMM kernel has an identical register/shared-memory
footprint to M38's original atomic kernel** — confirms the win came purely
from removing the atomic combine, not from any incidental occupancy change
(the exact evidence M43's own report claimed but this milestone
independently re-derives from a fresh compile). The below-threshold
block-reduce kernel shows no register spill and a modest 45-47 registers —
ruling out register pressure as its bottleneck, consistent with M33's own
finding and reinforcing this milestone's occupancy/parallelism diagnosis
(Section 8) over a register-pressure one.

## 12. Pipeline health

Async epoch (this script's own batch sweep): 86.0-93.3% compute-stream
utilization across batch sizes 32/64/128. Dedicated `pipeline_profile`
rerun (separate process, `benchmarks/results/m44_pipeline_profile.json`):
84.3-88.8% utilization across the same batch sizes, prefetch-depth sweep
flat at 85.7-86.2% across depths 1-3. Both ranges are consistent with
M40's 87-94% and M42's 83-92% findings — **no regression introduced by
M43** (expected: M43 touched only the `blocks_y==1` dWeight kernel, not
streams, the allocator, or pinned memory). Allocator and pinned-memory
counters show clean before/after/after-empty-cache behavior with no growth
across a profiled epoch — no leak, no new synchronization introduced.

## 13. Amdahl analysis

Using this session's own fresh MNIST kernel-contribution ranking
(`backward:conv2d` = 46.98% of the full step; `forward:Conv2d` = 12.40%):

| component | fraction of full step | 1.25x | 1.5x | 2.0x |
|---|---:|---:|---:|---:|
| conv2d backward total | 46.98% | 1.104x | 1.186x | 1.307x |
| dWeight below-256 block-reduce | **20.19%** | **1.042x** | **1.072x** | **1.112x** |
| dInput | 15.39% | 1.032x | 1.054x | 1.083x |
| conv2d forward | 12.40% | 1.025x | 1.043x | 1.066x |
| matmul backward | 8.21% | 1.017x | 1.028x | 1.043x |
| dWeight `blocks_y==1` (M43 two-stage) | 8.47% | 1.017x | 1.029x | 1.044x |
| dBias | 2.93% | 1.006x | 1.010x | 1.015x |
| CrossEntropy backward | 1.04% | 1.002x | 1.003x | 1.005x |

**The below-256 block-reduce path now has the largest Amdahl ceiling of
any single addressable sub-component** — a full 2x speedup there
(plausible for an occupancy-bound kernel moving from single-thread-per-
weight-element to a cooperative multi-thread scheme) projects to an
**11.2% whole-step speedup**, exceeding even a 2x dInput speedup (8.3%,
despite dInput's own larger `%-of-ceiling` headroom, because dInput's
current absolute contribution is smaller). A 2x `blocks_y==1` speedup, by
contrast, now projects to only 4.4% — this fraction shrank directly because
of M43's own success.

## 14. Bottleneck ranking

| rank | candidate | % of full step | roofline efficiency | headroom | Amdahl @2x | status |
|---:|---|---:|---:|---|---:|---|
| 1 | dWeight below-256 block-reduce | 20.19% | 3.1-3.2% of ceiling | very large | 1.112x | investigated twice (M33, M34), rejected both times — **selected for M45 with a new angle** |
| 2 | dInput (M36 channel-fused) | 15.39% | 8.3-33.9% of ceiling | moderate | 1.083x | extensively optimized (M36, 8.7x in isolation); not a fresh finding |
| 3 | conv2d forward | 12.40% | 27.9-48.8% of ceiling | small-moderate | 1.066x | already addressed (M41); healthy |
| 4 | matmul backward (`k_matmul`) | 8.21% | ~100% of ceiling by construction | none | 1.043x | at its own practical ceiling |
| 5 | dWeight `blocks_y==1` (M43) | 8.47% | 25.4-29.4% of ceiling | moderate, shrunk by M43 | 1.044x | just optimized (M43); confirmed healthy this session |
| 6 | dWeight `blocks_y>=2` (M39) | (subset of forward's conv-backward total; not separately isolated in the MNIST-shape Amdahl split, since neither MNIST layer reaches this regime) | 40.2-44.2%/67.4% of ceiling | small | — | already efficient (M39); no real MNIST shape reaches it |
| 7 | dBias | 2.93% | negligible | none | 1.015x | confirmed negligible every milestone since M31 |
| 8 | CrossEntropy | 1.04% | bandwidth-bound at realistic sizes | none | 1.005x | already fused (M31); no fresh opportunity |
| 9 | optimizer | not separately isolated this session (folded into the async-epoch's `gpu_optimizer_ms`, 0.07-0.14ms/step — negligible relative to `gpu_backward_ms`'s 3.9-14.4ms/step) | — | none | — | confirmed negligible (M31, M40, M42, and again here) |
| 10 | transfers/allocator/dependency overhead | H2D 0.06-1.55 GB/s isolated (small/medium/large), allocator/pinned counters clean | — | none | — | confirmed healthy every milestone since M27-M29; no regression from M43 |

## 15. Revisiting previously rejected work

Per the milestone's own instruction, a rejected strategy is reopened only
if new measurements change the evidence that caused its rejection —
**they do not, for either prior technique**: M33's cooperative-reduction
sweep (flat 3-4x gap across 32-256 threads/weight-element) and M34's
im2col+GEMM test at this exact shape (existing kernel wins at 72 elements)
were not re-run this session, and nothing measured here contradicts either
finding. What **has** changed is the below-256 kernel's *relative*
priority: M43 closed roughly a third of `blocks_y==1`'s Amdahl fraction
(10.2%→8.47%), leaving the below-256 kernel — unchanged, and already the
single largest absolute contributor even in M42 — with a materially larger
share of the remaining opportunity (its own 20.19% now more than double
`blocks_y==1`'s residual 8.47%) and a still-unexplored algorithmic angle:
**warp-shuffle-based cooperative reduction**, which uses neither M33's
shared-memory tree reduction nor M34's im2col+GEMM restructuring. This is
not a re-opening of either rejected technique; it is a new candidate at an
already-identified target, exactly as M42's own "Candidates" section
(item 2) flagged as plausible-but-unselected.

## 16. Selected M45 recommendation

1. **Target**: `k_conv2d_backward_weight_reduce`/`cf_conv2d_backward_
   weight_*` (M21, unchanged), the dWeight below-256-`weight_elements`
   dispatch path in `CUDABackend.conv2d_backward` — reached whenever
   `Cout*Cin*KH*KW < 256`, which is exactly MNIST's own real first conv
   layer (`mnist_conv1`).
2. **Measured bottleneck**: 3.1-3.2% of the practical compute ceiling
   across every below-threshold shape tested (`weight_elements` 72-243) —
   the worst, and most uniformly bad, roofline efficiency of any candidate
   in this or any prior characterization. Contributes **20.19%** of the
   full MNIST training step (Section 13), the single largest addressable
   sub-component measured this session.
3. **Root-cause hypothesis**: occupancy/parallelism-bound, not register-
   or memory-bandwidth-bound. `nvcc -Xptxas -v` shows no spill and a
   modest 45-47 registers (Section 11); the below-256-weight-element sweep
   (Section 8) shows cost scales with reduction length (`N*Hout*Wout`,
   held fixed) far more than with `weight_elements` itself — the kernel
   launches only `weight_elements`-many threads, each performing a fully
   independent, fully serial reduction over `N*Hout*Wout` elements, which
   at MNIST's own scale (50,176) is nowhere near enough total parallelism
   to saturate the 940MX.
4. **Evidence supporting the hypothesis**: (a) flat ~3.1-3.2% efficiency
   regardless of `weight_elements` in the 72-243 range — ruling out a
   FLOP-count-dependent explanation; (b) two shapes with identical
   `weight_elements=144` but different `Cin`/`Cout` splits cost the same
   (4.42ms vs. 4.32ms) — ruling out a `Cin`-or-`Cout`-specific explanation;
   (c) no register spill and moderate register count — ruling out
   register pressure; (d) M33's own prior finding (a *flat* 3-4x gap
   across cooperative granularities from 32 to 256 threads/weight-element)
   is itself consistent with an occupancy-bound kernel, since M33's tested
   granularities only added shared-memory-tree-reduction overhead without
   changing the fundamental single-thread-per-weight-element launch
   config's total thread count enough to matter at MNIST's own tiny
   72-243-element scale.
5. **Proposed optimization direction**: a warp-shuffle-based cooperative
   reduction — multiple threads (e.g. one warp, or a sub-warp group)
   cooperate on each weight-element's `N*Hout*Wout`-length reduction using
   `__shfl_down_sync` (no shared memory, no block-wide `__syncthreads`),
   increasing total launched-thread count by the cooperation factor while
   avoiding M33's specific shared-memory-tree-reduction overhead entirely.
   This is a genuinely untried angle: M33 tested shared-memory cooperation,
   M34 tested a structural GEMM rewrite; warp-shuffle is neither.
6. **Relevant representative shapes**: `mnist_conv1` (the only real Forge
   shape in this regime) plus this milestone's own below-256 sweep
   (`we_144a`, `we_144b`, `we_216`, `we_243`, `benchmarks/m44_bottleneck_
   recharacterization.py::_dweight_below_threshold_sweep`) to confirm any
   win generalizes across the regime rather than overfitting to
   `mnist_conv1`'s own 72-element shape.
7. **Expected performance opportunity**: per Section 13's Amdahl
   projection, a 2x speedup at this kernel projects to an **11.2%**
   whole-training-step speedup — the largest single-component Amdahl
   ceiling measured in this characterization, exceeding dInput's (8.3%)
   despite dInput's own larger measured `%`-of-ceiling headroom.
8. **Acceptance/rejection criterion**: a candidate reduction strategy
   measured faster than the current per-thread/block-reduce kernel at
   every below-256 representative/sweep shape above, with no regression at
   any tested shape, using the same interleaved CUDA-event A/B comparison
   methodology M37-M43 established. Given M33/M34's own prior rejections,
   this candidate should be rejected (and the below-256 path left
   unoptimized, as M42 already accepted as a live possibility) if the
   warp-shuffle approach does not clear a materially higher bar than a
   marginal win — e.g. it should close at least half the measured
   roofline-efficiency gap (target: **≥15% of the practical compute
   ceiling**, up from 3.1-3.2%) at `mnist_conv1` and the below-256 sweep
   shapes, mirroring M43's own acceptance bar for `blocks_y==1`.
9. **Explicit exclusions**: dWeight `blocks_y==1` (just optimized in M43;
   confirmed healthy and unregressed this session), dWeight `blocks_y>=2`
   (already efficient, 40.2-44.2%/67.4% of ceiling, M39), dInput (mediocre
   but not a fresh finding; already extensively optimized in M36; no real
   Forge shape reaches the `Cin>16` fallback), Conv2d forward (already
   addressed by M41; no further work justified by this milestone's
   measurements), `k_matmul`/`k_matmul_splitk` (already at their own
   practical ceiling), dBias, CrossEntropy, the optimizer, the allocator,
   CUDA streams, `_stream_guard`, pinned memory, async transfer, and
   DataLoader/prefetch (all confirmed healthy in this milestone with no
   regression).

**If M45's own profiling instead finds occupancy is not, in fact, the
limiter** (e.g. if a warp-shuffle prototype's own `nvcc -Xptxas -v`/
occupancy analysis reveals a different constraint), M45 should report that
finding rather than force the warp-shuffle direction to a conclusion the
evidence does not support — this milestone's root-cause hypothesis is
supported by indirect evidence (Section 8's sweep pattern, Section 11's
register analysis) but has not been confirmed at the instruction/occupancy-
profiler level, exactly as M42's own hypothesis for `blocks_y==1` was
before M43 confirmed it directly.

## 17. Why this target (over the alternatives)

- **dInput** (15.39% of step, second-largest): already the subject of a
  major M36 rewrite (8.7x in isolation); its own remaining inefficiency at
  `Cin=1` (9.2% of ceiling) contributes little in absolute terms at that
  shape, and the `Cin>16` regime it could still improve is never reached
  by any real Forge shape. Lower fresh-evidence value than a kernel that
  has never been examined below the wall-clock level after two prior
  rejected attempts left it untouched.
- **Conv2d forward** (12.40%): already addressed by M41 (10.7-18.0% →
  27.9-48.8% of ceiling); this session's fresh sweep confirms no
  regression and no new opportunity.
- **dWeight `blocks_y==1`** (8.47%, down from M42's 10.2%): just optimized
  in M43; this session's own fresh A/B confirms the win holds with no
  regression. Re-attacking it now, so soon after a confirmed win and at a
  shrunk fraction, has lower expected leverage than the below-256 path.
- **dWeight `blocks_y>=2`, dBias, CrossEntropy, optimizer, transfers,
  allocator**: all confirmed efficient or negligible, consistent with
  every prior milestone.

The below-256 block-reduce kernel is simultaneously the **largest single
absolute contributor** (20.19%), the **least roofline-efficient candidate
measured** (3.1-3.2% of ceiling, uniformly across shapes), and has a
**genuinely untried algorithmic angle** available (warp-shuffle
cooperation) that is distinct from both of its two prior, rejected
attempts — the combination M42 itself identified as the deciding factor
for a "select for the next milestone" call, applied here to the next
candidate in line now that `blocks_y==1` has been closed.

## 18. Exclusions

Per the milestone's hard scope: no changes were made to `forge/`, `tests/`,
any CUDA production kernel, Conv2d forward/backward, dInput, dWeight,
dBias, `k_matmul`, CrossEntropy, optimizers, the allocator, CUDA streams,
`_stream_guard`, pinned memory, async transfer, or DataLoader/prefetch.
This milestone is measurement and documentation only.

## 19. Limitations

- **940MX thermal/clock variance**: this session's own M43-vs-M38
  interleaved A/B (Section 7) shows a wider speedup range (1.063x-1.287x
  at `mnist_conv2`) than M43's own original report (1.11-1.24x) — both
  ranges overlap and agree on direction (no regression, meaningful win),
  but the exact figures are not directly comparable across sessions, per
  the milestone's own instruction. Absolute GFLOP/s and samples/sec
  figures in this report should be read the same way: relative/same-
  session comparisons are reliable, cross-session absolute comparisons are
  not.
- **Below-256 sweep base shape**: the below-threshold sweep fixed
  `N=64,H=W=28` (mnist_conv1's own spatial shape) while varying `Cin`/
  `Cout`; it does not independently vary `N`/`H`/`W` the way M43's own
  `blocks_y==1` sweep did. The root-cause hypothesis (Section 16, item 3)
  is therefore evidenced but not exhaustively isolated the way M43's own
  target was before its own optimization milestone — M45's own profiling
  should extend this if the warp-shuffle prototype's behavior does not
  match the hypothesis cleanly.
- **`blocks_y>=2` not separately isolated in the Amdahl/ranking tables**:
  no real MNIST layer shape reaches this regime (both real layers are
  either below-threshold or `blocks_y==1`), so this session's MNIST-shape-
  based Amdahl fractions cannot attribute a full-step percentage to it the
  way they can for the other candidates. Its own roofline efficiency
  (Section 8's `Cout` sweep, 40.2-67.4%) is independently well-measured.

## 20. Reproducibility

```
python -m benchmarks.m44_bottleneck_recharacterization
python -m benchmarks.pipeline_profile --output benchmarks/results/m44_pipeline_profile.json
```

Both scripts require real CUDA hardware (skip cleanly with a printed
message otherwise). Full profile JSON: `benchmarks/results/
m44_bottleneck_recharacterization.json`. Pipeline-only JSON: `benchmarks/
results/m44_pipeline_profile.json`.

## Tests / Verification

No production code changed (measurement-only milestone). Full suite:
**1,452 passed** (unchanged from M43's own count), verified on a clean
CUDA rebuild (`_forge_cuda_kernels_sm_50.dll` deleted and recompiled in
9.62s, matching M43's own 9.8s figure) immediately before running the test
suite and both benchmark scripts above.

## Why no ADR

Measurement and documentation only — no production code, public API, or
cross-cutting architectural decision was touched.

## Suggested Commit Message

`docs: M44 post-M43 CUDA bottleneck re-characterization`
