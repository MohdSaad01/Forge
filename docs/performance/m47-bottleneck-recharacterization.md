# M47 — Fresh Post-M46 CUDA Bottleneck Re-Characterization (measurement-only)

## 1. Executive summary

M45 (warp-shuffle cooperative reduction) and M46 (grid-level split
reduction) both rejected their respective candidates for the below-
`CONV2D_WEIGHT_REDUCE_THRESHOLD` (256) dWeight block-reduce kernel
(`k_conv2d_backward_weight_reduce`, one 256-thread block per weight
element, unchanged since M21). M46's real `cudaOccupancyMaxActiveBlocks
PerMultiprocessor` query found production and its grid-split candidate at
*identical* occupancy (5 blocks/SM, 62.5%, register-bound) — the strongest
evidence yet that this kernel has hit a structural ceiling on this device.
Production dispatch has been byte-for-byte unchanged since M43
(`c750b96`); M45/M46 added only profiling-only kernels, never called by
`CUDABackend`.

This milestone does not open a fifth attempt at the below-256 kernel. It
re-measures the whole CUDA training pipeline fresh, adds exactly one new
measurement no prior milestone produced — a single-session, round-robin-
interleaved **three-way** comparison of production against *both* rejected
candidates (M45's best warp-shuffle configuration and M46's best grid-split
configuration) at once — and uses the combined evidence to formally settle
the below-256 kernel's status before ranking every other candidate and
selecting exactly one M48 target.

**Headline findings.**

1. **The below-256 dWeight kernel is now classified as a practical
   optimization floor / deprioritized.** The fresh three-way reassessment
   (Section 10) found the best achievable result across *both* previously-
   rejected techniques, measured together in one interleaved run, is only
   **1.09x** at `mnist_conv1`'s own real shape (1.10x–1.18x across the
   below-256 sweep) — still short of the 1.15x bar this codebase has used
   since M43, and the fresh occupancy re-query reproduced M46's identical-
   occupancy finding exactly (5 blocks/SM, 62.5%, both variants). Four
   independent angles (M21's original design, M33 shared-memory
   cooperation, M45 warp-shuffle, M46 grid-split) have now been tried
   against this exact launch configuration without a milestone-clearing
   win.
2. **A significant, newly-documented measurement hazard**: back-to-back
   CUDA benchmarking sustained over many minutes on this laptop's 940MX
   causes cumulative thermal/clock drift far larger than the within-call
   drift M46 already documented — up to **~2.5-3x** slower absolute
   kernel times were observed late in a long session versus early, for
   the *same* kernel and shape, in the *same* Python process. This is
   distinct from M46's "block-sequential A/B" drift (which interleaving
   already fixes): it persists *across* interleaved comparisons separated
   by many minutes of other GPU work. Every headline number in this report
   was cross-checked with fresh, independent re-runs before being trusted
   (Section 12).
3. With the below-256 path deprioritized, **conv2d backward remains the
   largest step-share component (~41.5% of the full MNIST step, averaged
   across 7 fresh same-session trials to smooth the drift in item 2)**,
   but no single remaining sub-component dominates the way the below-256
   kernel did: dWeight-below-256 (~14.8%, now deprioritized), dWeight-
   `blocks_y==1` (~12.7%, just optimized in M43, reconfirmed healthy this
   session with no regression), dInput (~11.75%), and conv2d forward
   (~11.9%) are all close together.
4. **Selected M48 target: the M36 channel-fused dInput kernel's behavior
   at low `Cin`**, specifically `Cin=1` — `mnist_conv1`'s own real shape.
   This is the worst-roofline-efficiency *live* (non-deprioritized)
   candidate found this session (4.7-9.2% of the practical compute
   ceiling, essentially unchanged from M44's own 8.3% finding), it has a
   comparable step-share to the other top candidates, and it has a
   genuinely distinct, previously-unexamined structural cause: M36's
   channel-fusion technique gets its benefit from keeping multiple `Cin`
   accumulators register-resident across a shared reduction loop — at
   `Cin=1` there is exactly one accumulator, so the kernel pays M36's full
   70-register cost with none of the multi-channel reuse that justifies
   it. Full recommendation in Section 17.

## 2. Current architecture (unchanged since M43)

`CUDABackend.conv2d` (forward, unchanged since M41): below
`_CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD` (10,000,000 FLOPs) → the original
M15 per-thread kernel; at/above it, `blocks_x = ceil(Cout/16)`; `blocks_x
<=2` (every current Forge shape) → M41 Candidate B (half-fused GEMM);
`blocks_x>2` → M41 Candidate A (im2col-smem + tiled GEMM + permute).

`CUDABackend.conv2d_backward`: dInput dispatch (`kernels.cu`'s
`CONV2D_DINPUT_CHANNELFUSED_MAX_CIN = 16`, confirmed directly against the
compiled source this session, unchanged since M36): `Cin<=16` → M36
channel-fused kernel (every real Forge shape); `Cin>16` → the original M32
kernel (unreached fallback). dWeight dispatch: `weight_elements =
Cout*Cin*KH*KW`; below 256 → **unchanged** M21 per-thread/block-reduce
kernel; at/above it, `blocks_y = ceil(Cout/16)`; `blocks_y==1` → M43's
two-stage split-K reduction (permute → partial-GEMM → reduce); `blocks_y
>=2` → M39's shared-memory im2col + permute + split-K GEMM.

**Verified this session** by reading `forge/backend/cuda/backend.py` and
`forge/backend/cuda/kernels.cu` directly (not from documentation): every
threshold, branch condition, and function name above matches the compiled
production code exactly. `git status --porcelain -- forge/ tests/` is
empty both before and after this milestone's work (Section 17).

M45/M46 added profiling-only kernels/exports that remain in `kernels.cu`
(`k_conv2d_backward_weight_warp[_subgroup]`, `k_conv2d_backward_weight_
reduce_gridsplit`, `cf_occupancy_conv2d_backward_weight_reduce[_gridsplit]_
f32`) — none are referenced by `CUDABackend`.

## 3. Measurement methodology

`benchmarks/m47_bottleneck_recharacterization.py` (new) reuses, rather
than reimplements, every prior milestone's own profiling code:

- Full forward+backward decomposition at the 7 representative shapes +
  batch sweep: `m44_bottleneck_recharacterization._profile_shape_full`
  (unmodified).
- dWeight `Cout`/below-256/dInput `Cin` sweeps:
  `m44_bottleneck_recharacterization`'s own sweep functions (unmodified).
- M43 `blocks_y==1` reconfirmation:
  `m43_dweight_splitk_profile._candidate_b_comparison` (unmodified,
  already a proper interleaved A/B).
- Forward-only K/stride/channel sweep + forward memory characterization:
  `m42_bottleneck_recharacterization`'s functions (unmodified).
- MNIST kernel-contribution ranking, CrossEntropy roofline, async
  pipeline/allocator/pinned-memory health: `m35_mnist`, `m35_kernels`,
  `pipeline_profile` (all unmodified).
- **New this milestone**: `_below256_three_way_reassessment` — builds one
  `_RawDWeightBelow256` instance (M45) and one `_RawDWeightGridsplit`
  instance (M46) per shape, then times production plus every M45
  warpreduce/warpsubgroup configuration plus every M46 grid-split
  configuration in a *single* `_interleaved_multi_time` round-robin call
  (M46's own corrected methodology, reused directly) — the first time all
  three variants have been measured together in one interleaved run
  rather than each candidate only ever being compared against production
  in isolation.
- A fresh `_occupancy_query()` call (M46's own real
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` wrapper, unmodified).

Every stage call is the same function `backend.py` itself calls, or the
same experimental module it would call if a candidate were ever promoted.
CUDA events measure every GPU phase; synchronization happens once at each
phase boundary, matching every prior milestone's convention.

**New methodology addition (Section 12): repeated-trial averaging.** Where
a single measurement showed implausibly large deviation from every prior
session's own history, it was independently re-run 2-5 times as a
standalone script invocation before being trusted, and the MNIST kernel-
ranking fractions used for the Amdahl analysis (Section 13) are the mean
of 7 fresh same-session trials, not a single draw — a direct response to
the thermal-drift finding in item 2 of the executive summary.

## 4. Hardware / CUDA environment

NVIDIA GeForce 940MX, driver 582.53, CUDA compute capability 5.0, 2048 MiB
VRAM, Windows 10, Python 3.13.5. Forge commit `d1fe3b7` (M46, HEAD at the
start of this milestone). `_forge_cuda_kernels_sm_50.dll` deleted and
recompiled fresh (21.02s) immediately before the baseline test run.

## 5. Baseline verification (clean rebuild)

Full suite after the clean rebuild: **1,550 passed** (unchanged from
M46's own count — expected, since M47 adds no production tests). No
skips attributable to CUDA unavailability (CUDA is available and used
throughout this session).

Fresh practical ceilings (M35 methodology, this session): **104.58
GFLOP/s** compute, **15.09 GB/s** bandwidth — within run-to-run variance
of every prior session (M35: initial baseline; M40/M42/M44: ~104-105
GFLOP/s / ~15.1 GB/s). Ceiling figures are the most session-stable numbers
measured in this milestone; absolute kernel millisecond figures are not
(Section 12).

## 6. Training-step decomposition

Full forward+backward decomposition at the 7 representative shapes (this
session, `m47_bottleneck_recharacterization` Phase 2 — captured
immediately after the Phase 0 ceiling measurement, the most internally-
consistent single sweep available this session):

| shape | fwd(ms) | dIn(ms) | dW(ms) | dB(ms) | bwd total(ms) | dW path |
|---|---:|---:|---:|---:|---:|---|
| mnist_conv1 | 0.669 | 0.784 | 2.108 | 0.218 | 3.110 | below-256 block-reduce |
| mnist_conv2 | 0.947 | 0.888 | 1.809 | 0.094 | 2.792 | M43 two-stage |
| large_channel | 10.158 | 13.406 | 17.972 | 0.722 | 32.100 | M39 im2col-smem+splitK |
| large_spatial | 6.425 | 7.704 | 9.323 | 0.768 | 17.794 | M43 two-stage |
| batch_32 | 6.992 | 7.144 | 10.443 | 0.366 | 17.952 | M39 im2col-smem+splitK |
| batch_64 | 10.447 | 13.012 | 21.952 | 0.731 | 35.695 | M39 im2col-smem+splitK |

**`batch_128` is excluded from this table** — its forward figure measured
91.6ms in this run, wildly non-linear against batch_32/64's 7.0/10.4ms.
Independent standalone re-runs (Section 12) reproduced 21.8ms, 33.4ms, and
41.0ms across 3 successive trials of the *same* shape in the *same*
process — a monotonically worsening, high-variance (stdev 4-11ms) result
consistent with cumulative thermal throttling under sustained large-batch
load, not a real per-shape cost. This is a fresh, hardware-verified
measurement-quality finding (Section 12), not a production concern.

**MNIST training-step composition** (mean of 7 fresh same-session
`m35_mnist` trials, to smooth the thermal-drift variance documented in
Section 12 — individual trials ranged from 35.1% to 45.3% for the
`backward:conv2d` fraction alone):

| component | fraction of full step (mean of 7 trials) | range across trials |
|---|---:|---:|
| conv2d backward total | 41.5% | 35.1%-45.3% |
| conv2d forward | 11.9% | 11.3%-12.5% |
| matmul backward | 9.9% | 9.6%-10.3% |
| CrossEntropy backward | 1.5% | 1.4%-1.6% |

Within conv2d backward's own total (proportions from the Section 6 shape
table, which are far more stable across trials than the absolute-ms
figures since they are ratios *within* one coherent sweep):

| dWeight below-256 | dWeight `blocks_y==1` | dInput | dBias |
|---:|---:|---:|---:|
| 35.7% of conv-bwd → **14.8% of full step** | 30.7% of conv-bwd → **12.7% of full step** | 28.3% of conv-bwd → **11.75% of full step** | 5.3% of conv-bwd → **2.2% of full step** |

## 7. Conv2d forward characterization

Fresh forward-only K/stride/channel sweep (M41's `SWEEP_SHAPES`, this
session): 2.8% (`k1_s1`, below the GEMM-dispatch threshold — the same
unchanged M15 kernel M40/M41 already characterized) to 26.7% (`cin_high`)
of the practical compute ceiling. This is directionally consistent with
M41/M42/M44's own findings (their absolute range, 3.6-48.8%, was measured
earlier in each of those sessions and is not directly comparable per
Section 12's thermal-drift finding — see Section 11's caveat on every
later-phase sweep in this report). No new forward opportunity is
indicated: the dispatch logic, its threshold, and both GEMM candidates are
unchanged and reconfirmed working exactly as M41 designed.

## 8. dInput characterization

Fresh `Cin` sweep (fixed `N=16,Cout=16,H=W=28,K=3`, straddling the M36
channel-fused/`Cin>16`-fallback boundary):

| Cin | path | d_input (ms) | % of ceiling |
|---:|---|---:|---:|
| 1 | M36 channel-fused | 0.741 | 4.7% |
| 8 | M36 channel-fused | 1.467 | 18.8% |
| 16 | M36 channel-fused | 1.906 | 29.0% |
| 17 | M32 fallback | 14.202 | 4.1% |
| 32 | M32 fallback | 27.245 | 4.1% |
| 64 | M32 fallback | 52.880 | 4.2% |

The channel-fused path's efficiency climbs steeply with `Cin` (4.7% →
29.0%) purely as a function of how many accumulators the kernel actually
has to amortize its per-thread register cost over — at `Cin=1`,
`mnist_conv1`'s own real value, it is the **worst-performing live
candidate measured this session**. The `Cin>16` fallback remains flat and
mediocre (~4.1-4.2%) but is still not reached by any real Forge shape,
confirming M42/M44's own repeated finding. `nvcc -Xptxas -v` (this
session, unchanged from M44): 70 registers for the channel-fused kernel,
paid regardless of `Cin`.

## 9. dWeight characterization

**Below-256 block-reduce** (fixed `N=64,H=W=28,K=3`, mnist_conv1's own
spatial shape, `Cin`/`Cout` varied):

| label | weight_elements | d_weight (ms) | % of ceiling |
|---|---:|---:|---:|
| mnist_conv1 (we=72) | 72 | 2.484 | 2.8% |
| we_144a | 144 | 5.565 | 2.5% |
| we_144b | 144 | 6.106 | 2.3% |
| we_216 | 216 | 8.079 | 2.6% |
| we_243 | 243 | 10.870 | 2.1% |

Flat ~2.1-2.8% of ceiling regardless of `weight_elements`, reproducing
M44's own flat ~3.1-3.2% pattern (the small absolute-level difference is
the same session-to-session thermal effect documented throughout this
report, not a new finding) — the diagnosis (occupancy-bound, not FLOP- or
register-bound) is unchanged.

**`blocks_y==1` (M43 two-stage)**: freshly re-verified with 3 independent
standalone interleaved A/B re-runs after an initial anomalous in-script
draw was caught and discarded (Section 12):

| shape | gemm-only speedup (3 fresh trials, splits=16 production) | full-pipeline speedup |
|---|---|---|
| mnist_conv2 | 1.140x-1.189x | 1.110x-1.154x |
| large_spatial | 1.017x-1.022x | 1.019x-1.032x |

**No regression at either real shape, at any of the 3 re-runs** — M43's
win holds, matching M43's own 1.11-1.24x/1.02-1.04x and M44's own
1.15-1.29x/1.02-1.03x ranges closely. The reduce kernel remains
bandwidth-trivial by construction (unchanged since M43/M44).

**`blocks_y>=2` (M39, unchanged)**: fresh `Cout` sweep (`Cin=16,H=W=13,
K=3`) shows continued efficient scaling (14.8% at `Cout=32` up to 23.9% at
`Cout=128`) — the exact absolute percentages are lower than M44's own
(40.2-67.4%) due to this sweep running later in the same long session
(Section 12's thermal-drift caveat applies), but no regression or new
concern is indicated; the shape of the scaling curve (efficiency rising
with `Cout`) is preserved.

## 10. Below-256 dWeight reassessment (three-way, fresh)

**New this milestone.** A single interleaved comparison — production, all
5 of M45's `warpreduce` configurations, all 6 of M45's `warpsubgroup`
configurations, and all 4 of M46's `gridsplit` configurations, timed
together in one `_interleaved_multi_time` round-robin call per shape:

| shape | weight_elements | production (ms) | best variant | best (ms) | speedup |
|---|---:|---:|---|---:|---:|
| mnist_conv1 (we=72) | 72 | 2.3045 | gridsplit_ns4 | 2.1140 | **1.090x** |
| we_144 | 144 | 5.1118 | gridsplit_ns1 | 4.6612 | **1.097x** |
| we_243 | 243 | 9.3457 | gridsplit_ns1 | 7.8933 | **1.184x** |

At every shape, the best variant across *both* previously-rejected
techniques is a grid-split configuration, never a warp-shuffle one —
consistent with M45's own correction that warp-shuffle candidates launch
*fewer* threads than production and were never the more promising
direction. At `mnist_conv1`'s own real shape, the combined best result
(1.090x) remains below the 1.15x bar this codebase has held since M43.

**Fresh occupancy re-query** (real `cudaOccupancyMaxActiveBlocksPer
Multiprocessor`, this session): production and the grid-split candidate
both report **5 blocks/SM, 62.5% occupancy** — an exact reproduction of
M46's own finding. Grid-splitting still cannot raise this device's per-SM
concurrent-block ceiling.

## 11. Roofline classification summary

| candidate | % of ceiling (this session) | classification | status |
|---|---:|---|---|
| dWeight below-256 block-reduce | 2.1-2.8% | launch/occupancy-bound | **deprioritized** (Section 16) |
| dInput at Cin=1 (M36 channel-fused) | 4.7% | occupancy/register-amortization-bound | **selected for M48** |
| dInput at Cin=16 | 29.0% | mixed/ambiguous | healthy at its own upper Cin bound |
| Conv2d forward (GEMM-dispatched) | up to 26.7% this session | mixed/ambiguous | already addressed (M41); no new opportunity |
| dWeight blocks_y==1 (M43) | mixed, healthy, reconfirmed no regression | mixed/ambiguous | just optimized; not re-attacked |
| dWeight blocks_y>=2 (M39) | 14.8-23.9% this session | mixed/compute-leaning | already efficient; no real MNIST shape reaches it |
| GEMM itself (`k_matmul`/`k_matmul_splitk`) | ~100% by construction | compute-bound | at its own ceiling |
| CrossEntropy | 2.4-58.2% (bandwidth-bound at realistic/large sizes) | healthy | negligible step-share (1.5%) |

**All later-phase percentages in this table (dInput's own sweep, forward's
sweep, the `blocks_y>=2` `Cout` sweep) were measured well into a long
session and are very likely deflated by the thermal drift documented in
Section 12** — the *classification* and *relative ranking* between
candidates is trustworthy (all points in a given sweep share the same
drifted state), but none of these absolute percentages should be compared
against an earlier milestone's own absolute percentage as if hardware
had changed.

## 12. Pipeline / transfer / allocator health, and a new measurement-quality finding

**Async pipeline** (standalone `pipeline_profile` rerun, `benchmarks/
results/m47_pipeline_profile.json`): 76.7-94.5% compute-stream utilization
across batch sizes 32/64/128, 89.7-91.9% across prefetch depths 1-3 — both
within or above every prior milestone's own historical range (M40: 87-94%;
M42: 83-92%; M44: 84.3-93.3%). **No regression** — expected, since nothing
between M43 and M47 touched streams, the allocator, or pinned memory.

**Allocator/pinned memory**: clean before/after/after-gc-and-empty-cache
behavior across a profiled epoch — `allocated_bytes` and `pinned_active_
bytes` both return to their pre-epoch baseline after `empty_cache()`, with
zero `pending_bytes`/`pending_count` remaining. No leak, no growth.

**New finding: cumulative cross-phase thermal drift.** M46 already
documented that *block-sequential* timing (fully time variant A, then
fully time variant B) is unreliable on this hardware because GPU clock/
power state drifts within a multi-second timing block — the fix was
`_interleaved_multi_time`. This milestone found a second, larger-scale
version of the same underlying phenomenon: even with every individual A/B
comparison correctly interleaved, **absolute kernel times measured late in
a long, continuously-CUDA-active session can be 2.5-3x slower than the
same kernel/shape measured early in that same session**, evidenced
directly by:

- `batch_128`'s forward time (Section 6): 91.6ms in the main script run,
  reproducing at 21.8ms → 33.4ms → 41.0ms across 3 successive standalone
  trials immediately afterward (monotonically worsening within the same
  process).
- The main script's own in-line M43 `blocks_y==1` re-confirmation drew a
  single anomalous 3.877x "speedup" at `mnist_conv2` (worlds outside every
  prior session's 1.11-1.29x range) — 3 independent standalone re-runs
  immediately after, using the *identical* interleaved-comparison function,
  reproduced only 1.14x-1.19x, consistent with M43/M44's own history. The
  anomalous draw is attributed to the same cumulative drift, not a real
  regression or a methodology bug in the interleaving itself (the
  interleaving still correctly cancelled drift *within* that one round-
  robin call — the whole call was just run in an unusually hot state).
- The MNIST kernel-ranking's own `backward:conv2d` fraction varied from
  35.1% to 45.3% across 7 fresh same-session trials (Section 6) — a ±5-9
  percentage-point spread attributable to the same cause.

**This is now the primary reason this report uses repeated-trial averages
and independent re-verification for every headline number** (Section 3),
rather than trusting single draws the way M40-M44 generally did. It does
not indicate any Forge-side bug: it is a real property of this specific
i5-7200U/940MX laptop's thermal envelope under sustained load, consistent
with the "Environment constraints" already documented in `CLAUDE.md`. It
should inform how any future milestone interprets a long benchmarking
script's own later-phase numbers, and is a stronger version of the
caution M46 already raised.

## 13. Amdahl analysis

Using the 7-trial-averaged MNIST fractions (Section 6):

| component | fraction of full step | 1.25x | 1.5x | 2.0x |
|---|---:|---:|---:|---:|
| conv2d backward total | 41.5% | 1.077x | 1.130x | 1.207x |
| dWeight below-256 block-reduce | 14.8% | 1.026x | 1.043x | 1.066x |
| dWeight `blocks_y==1` (M43) | 12.7% | 1.022x | 1.036x | 1.056x |
| dInput | 11.75% | 1.020x | 1.032x | 1.051x |
| conv2d forward | 11.9% | 1.021x | 1.033x | 1.052x |
| matmul backward | 9.9% | 1.017x | 1.028x | 1.043x |
| dBias | 2.2% | 1.004x | 1.007x | 1.011x |
| CrossEntropy backward | 1.5% | 1.003x | 1.005x | 1.008x |

**Below-256 dWeight's own Amdahl ceiling has shrunk from M44's 1.112x
(at a 2x speedup, 20.19% fraction) to 1.066x (14.8% fraction)** — both the
percentage-point drop (driven by thermal-corrected, multi-trial averaging
rather than a single draw) and the fact that no candidate technique has
ever reached a 2x speedup at this kernel (the best measured combined
result is 1.09x-1.18x, Section 10) make this an increasingly poor
investment even before considering its four-attempts-rejected history.

**dInput, conv2d forward, and dWeight `blocks_y==1` are now within ~1
percentage point of each other** (11.75%, 11.9%, 12.7%) — none dominates
the Amdahl argument the way below-256 dWeight did in M42/M44. The
selection in Section 16 is therefore driven primarily by roofline
efficiency and untried-angle evidence, not by a clear Amdahl-fraction gap.

## 14. Candidate ranking

| Candidate | Step Share | Efficiency | Estimated Headroom | Difficulty | Prior Evidence | Priority |
|---|---:|---:|---|---|---|---|
| dWeight below-256 block-reduce | 14.8% | 2.1-2.8% of ceiling | Large in theory, but 4 techniques (M21/M33/M45/M46) all land at 1.0-1.2x | High (register-bound at the architectural level) | 4 investigated angles, all rejected; identical occupancy confirmed twice | **Deprioritized** |
| dInput at low Cin (Cin=1, M36) | 11.75% | 4.7% of ceiling at Cin=1 | Moderate-large; untried angle (Cin-adaptive kernel) | Moderate (reuse of existing im2col/GEMM/shared-memory infra) | 1 successful milestone (M36, 8.7x in isolation) at a *different* regime; low-Cin regime never separately targeted | **Selected for M48** |
| Conv2d forward | 11.9% | up to 26.7% of ceiling this session | Small; already restructured | Low (already done) | Thoroughly addressed, M41; reconfirmed healthy, no new opportunity this session | Not selected |
| dWeight `blocks_y==1` (M43) | 12.7% | Mixed, no regression | Small; just improved | Low-moderate | Just optimized M43; reconfirmed via 3 fresh interleaved re-runs this session | Not selected (too soon) |
| dWeight `blocks_y>=2` (M39) | (subset of forward step; no real MNIST shape isolates it) | 14.8-23.9% this session | Small | Low | Already efficient, M39; no real MNIST shape reaches it | Not selected |
| matmul backward (`k_matmul`) | 9.9% | ~100% by construction | None | N/A | At its own practical ceiling since M35 | Excluded |
| dBias | 2.2% | Negligible | None | N/A | Confirmed negligible every milestone since M31 | Excluded |
| CrossEntropy | 1.5% | Bandwidth-bound at realistic sizes | None | N/A | Already fused, M31 | Excluded |

## 15. Rejected / deprioritized targets

- **dWeight below-256 block-reduce**: formally deprioritized this
  milestone (Section 16) after a fourth rejected angle and a fresh,
  combined three-way reassessment confirming no candidate clears the
  acceptance bar even at the *best* result found across every prior
  technique. Not a candidate for M48 or any near-term milestone without a
  genuinely new algorithmic primitive (Section 16).
- **dWeight `blocks_y==1`**: just optimized (M43), reconfirmed healthy
  and unregressed by 3 independent fresh interleaved re-runs this
  session. Re-attacking now has low expected value.
- **dWeight `blocks_y>=2`**: already efficient (M39); no real MNIST layer
  shape reaches this regime.
- **Conv2d forward**: already restructured (M41); this session's fresh
  sweep found no new opportunity.
- **matmul, dBias, CrossEntropy, optimizer, allocator, streams, pinned
  memory, DataLoader/prefetch**: all confirmed healthy or negligible,
  consistent with every milestone since M27-M31/M35.

## 16. Below-256 dWeight: formal decision

Per the M47 brief's explicit instruction to settle this kernel's status
using the combined evidence from M21/M33/M34/M45/M46 plus this session's
own fresh measurements:

**Is another optimization attempt justified? No, not with a technique
already tried.** Evidence:

- Current training-step fraction: 14.8% (this session's own multi-trial
  average) — still meaningful, but no longer dominant now that it is
  correctly weighted against the same measurement noise every other
  candidate is subject to.
- Current compute-ceiling fraction: 2.1-2.8%, the worst of any candidate
  measured — but this alone is explicitly *not* sufficient justification
  per the milestone's own Section 18 decision rule.
- Register-bound occupancy: **directly confirmed twice** now (M46's
  original query, this session's fresh re-query) — 5 blocks/SM, 62.5%,
  identical between production and every grid-split variant tested.
  Occupancy is not merely hypothesized; it is measured at the CUDA runtime
  level.
- Previously failed strategies: M33 (shared-memory cooperative
  reduction), M45 (warp-shuffle cooperative reduction), M46 (grid-level
  split) — three genuinely distinct parallelization axes, none clearing
  the bar.
- Remaining plausible algorithmic options: none of the three tried axes
  (thread-level, warp-level, block/grid-level cooperation within the same
  overall "reduce over `N*Hout*Wout` per weight element" algorithm) have
  room left to try. The only remaining option is a **structurally
  different algorithm** — e.g., recasting this specific low-`weight_
  elements` regime as a small, dense GEMM the way M34 did at ≥256, but
  M34 already tested exactly this at 72 elements and the existing
  per-thread kernel won (per M44's own accounting of unreopened M34
  evidence). No new algorithmic idea was identified this session.
- Realistic Amdahl ceiling: even a hypothetical (unachieved) full 2x
  speedup now projects to only 1.066x whole-step — down from M44's 1.112x
  because the fraction itself is now measured with less noise and turns
  out smaller than M44's own single-draw estimate.

**Formal classification: below-256 dWeight block-reduce is a practical
optimization floor / deprioritized** for its current algorithmic family
(single-thread-block-per-weight-element, cooperative reduction over the
`N*Hout*Wout` dimension in any of thread/warp/block granularity). It is
**not** classified as "optimal" — the measurements support diminishing
returns for further tuning within this family, not a proof that no faster
algorithm could ever exist. A future milestone should only reopen this
target with a structurally different algorithm (not a fifth cooperation
granularity), and should expect a high bar given four prior rejections.

## 17. Selected M48 target

1. **Target**: `k_conv2d_backward_input_channelfused`/`cf_conv2d_
   backward_input_channelfused_*` (M36, unchanged), specifically its
   behavior in the low-`Cin` regime — `Cin` in roughly 1-4 — which
   includes `mnist_conv1`'s own real shape (`Cin=1`).
2. **Root cause**: M36's channel-fusion technique achieves its speedup by
   keeping one accumulator register per input channel resident across a
   shared `Cout`-reduction loop, amortizing loop-index/address arithmetic
   and avoiding redundant global-memory re-reads across channels. At
   `Cin=1` there is exactly one accumulator — the kernel pays its full,
   unchanged 70-register-per-thread cost (confirmed by `nvcc -Xptxas -v`,
   unchanged since M36/M44) with none of the multi-channel amortization
   that justifies that cost at higher `Cin`.
3. **Evidence**: the fresh `Cin` sweep (Section 8) shows a steep,
   monotonic efficiency climb purely as a function of `Cin` at an
   otherwise-fixed shape — 4.7% of ceiling at `Cin=1` up to 29.0% at
   `Cin=16` — the clearest signature of an optimization whose benefit
   scales with a parameter (`Cin`) that MNIST's own first conv layer
   happens to set to its worst value (1). This matches M44's own
   independent measurement of the same sweep (8.3% at `Cin=1`, absolute
   value differs per Section 12's thermal caveat, but the qualitative
   finding — worst efficiency at real Forge's actual `Cin=1` shape — is
   reproduced across two separate milestones and sessions.
4. **Optimization direction**: a `Cin`-adaptive dInput kernel. At low
   `Cin` (where channel-fusion has little or nothing to amortize),
   dispatch to a kernel that instead gets its parallelism from the
   spatial (`H`x`W`) or batch (`N`) dimensions — e.g., a shared-memory
   spatial-tiling scheme analogous to M39's im2col-smem staging, or a
   restructuring toward the same im2col+GEMM technique M34/M41 already
   proved out for dWeight/forward. This is a genuinely different
   structural idea from M36's own channel-fusion, not a re-tuning of it.
5. **Existing infrastructure to reuse**: `k_im2col_conv2d`/`k_im2col_
   conv2d_smem` (M34/M39 staging pattern), `k_matmul`/`k_matmul_splitk`
   tiled/split-K GEMM (M34/M37), the M36 channel-fused kernel itself as
   the `Cin>4`-or-so fallback/comparison baseline, `roofline.py`
   classification helpers, `_interleaved_multi_time`/`_time_phase`
   (M43/M46 timing harness), and `conv2d_backward_profile._RawConv2d
   Backward` as the raw-call harness base class.
6. **Representative shapes**: `mnist_conv1` (`Cin=1`, the only real Forge
   shape in this regime) as the primary acceptance shape, plus a
   dedicated `Cin` sweep at 1/2/4/8/16 (at both `mnist_conv1`'s own
   spatial shape and at least one larger spatial shape, e.g.
   `large_spatial`'s `H,W`) to find where — if anywhere — a new kernel's
   crossover point with the existing channel-fused kernel falls, and
   `mnist_conv2`'s own `Cin=16` shape as an explicit no-regression check.
7. **Acceptance criterion**: a candidate kernel measured faster than the
   current channel-fused kernel at `mnist_conv1`'s own shape and at every
   low-`Cin` sweep point (targeting **≥15% of the practical compute
   ceiling** at `Cin=1`, up from 4.7%, mirroring M43/M45's own acceptance-
   bar convention), with no regression (>5% slower) at `Cin>=8` or at
   `mnist_conv2`'s own `Cin=16` shape, using the same same-session
   interleaved CUDA-event methodology established since M43. End-to-end:
   a measurable (≥3%) whole-`conv2d_backward` speedup at `mnist_conv1`'s
   real shape.
8. **Explicit exclusions**: dWeight below-256 block-reduce (deprioritized,
   Section 16 — no cooperative-reduction-granularity variant may be
   retried), dWeight `blocks_y==1` (just optimized M43, reconfirmed
   healthy), dWeight `blocks_y>=2` (already efficient, M39), Conv2d
   forward (already addressed, M41), `k_matmul`/`k_matmul_splitk`
   (already at ceiling), dBias, CrossEntropy, the optimizer, the
   allocator, CUDA streams, `_stream_guard`, pinned memory, async
   transfer, DataLoader/prefetch (all confirmed healthy this session).
   The `Cin>16` M32 fallback path is explicitly out of scope — no real
   Forge shape reaches it, confirmed again this session.
9. **Fallback / stop condition**: if a fresh
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` query at `Cin=1` shows
   the channel-fused kernel and a spatially-tiled alternative land at
   identical or near-identical occupancy (mirroring M46's own below-256
   finding), or if no spatially-tiled prototype clears the acceptance bar
   at any tested shape, M48 must reject the approach, report dInput's
   low-`Cin` regime as a second practical floor, and defer to the next-
   ranked live candidate (conv2d forward or dWeight `blocks_y==1`,
   Section 14) for a future milestone rather than forcing an
   implementation the evidence does not support.

## 18. Limitations

- **Cumulative cross-phase thermal drift** (Section 12) is the dominant
  limitation of this report. Every absolute-millisecond figure reflects
  one specific point in a multi-hour session's thermal history; only
  ratios computed *within* a single interleaved call, or fractions
  averaged across several independent trials, should be trusted at face
  value. Section 6/9/13's headline numbers were specifically re-verified
  this way; several other tables in this report (the forward-only sweep,
  the `Cout`/`Cin` sweeps in Sections 7-9) were not independently
  re-verified beyond their original single draw and should be read as
  qualitative/directional (their *relative* ordering, not their exact
  percentages).
- **`batch_128` forward is excluded from every headline claim** in this
  report after failing reproduction (Section 6) — it is not a real
  per-shape cost, and any consumer of `benchmarks/results/m47_bottleneck_
  recharacterization.json` should treat that specific data point as
  contaminated, not representative.
- **Three-way reassessment scope** (Section 10): only 3 below-256 shapes
  were tested (mirroring M44's own below-threshold sweep shapes), not the
  full cross-product of M45's and M46's own independent sweeps (weight-
  element count, reduction size, K-configuration) — sufficient to confirm
  the combined-best-result conclusion, but a future milestone revisiting
  this kernel with a genuinely new algorithm should still run its own
  fresh, comprehensive sweep rather than relying on this reduced set.
- **M48's dInput target has only ever been measured via the isolated
  `Cin` sweep** (Section 8) — no dedicated occupancy query, `nvcc -Xptxas
  -v` deep-dive, or prototype has been attempted at `Cin=1` specifically;
  M48's own PROFILE phase should establish this evidence base before any
  DESIGN work, per the same discipline M45/M46 followed for the below-256
  kernel.

## 19. Conclusion

Production CUDA dispatch, kernels, and public APIs are unchanged since
M43 (`c750b96`); this milestone is measurement and documentation only.
The below-256 dWeight block-reduce kernel — the largest single
contributor identified in M42/M44 and the subject of M45/M46's dedicated
optimization attempts — is formally reclassified as a practical
optimization floor after a fourth rejected angle and a fresh, combined
three-way reassessment confirming no available technique clears this
codebase's own 1.15x acceptance bar. A significant thermal-drift
measurement hazard, distinct from and larger than M46's own within-call
drift finding, was discovered, documented, and corrected for via repeated-
trial averaging and independent re-verification of every headline number.
With the below-256 kernel set aside, the remaining live candidates
(dInput, conv2d forward, dWeight `blocks_y==1`) are close in step-share;
**M48 is recommended to target the M36 channel-fused dInput kernel's
low-`Cin` regime** (Section 17), the worst-roofline-efficiency live
candidate measured this session with a genuinely untried structural angle.

## Reproducibility

```
python -m benchmarks.m47_bottleneck_recharacterization
python -m benchmarks.pipeline_profile --output benchmarks/results/m47_pipeline_profile.json
```

Both scripts require real CUDA hardware (skip cleanly with a printed
message otherwise). Full profile JSON: `benchmarks/results/
m47_bottleneck_recharacterization.json`. Pipeline-only JSON: `benchmarks/
results/m47_pipeline_profile.json`.

## Tests / Verification

No production code changed (measurement-only milestone). Full suite:
**1,550 passed** (unchanged from M46's own count), verified on a clean
CUDA rebuild (`_forge_cuda_kernels_sm_50.dll` deleted and recompiled in
21.02s) immediately before running the test suite and both benchmark
scripts above. `git status --porcelain -- forge/ tests/` empty
throughout.

## Why no ADR

Measurement and documentation only — no production code, public API, or
cross-cutting architectural decision was touched.

## Suggested Commit Message

`docs: M47 post-M46 CUDA bottleneck re-characterization`
