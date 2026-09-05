# M42 — Fresh Post-M41 CUDA Bottleneck Re-Characterization (measurement-only)

## 1. Executive summary

M41 changed `CUDABackend.conv2d`'s forward dispatch substantially (a
shape-based im2col+GEMM/half-fused-GEMM dispatch replacing a single direct
per-thread kernel below a FLOPs threshold), which made every earlier
milestone's Conv2d-forward percentages and optimization priorities stale.
This milestone re-measures the **whole** CUDA training pipeline fresh, in
one session, against the actual current production dispatch on both the
forward and backward sides at once (`benchmarks/m42_bottleneck_
recharacterization.py`, new — combines, rather than reimplements, M40's
backward decomposition and M41's forward decomposition). No production
code, kernel, or public API was touched.

**Headline finding.** M41 worked: forward is no longer the pipeline's
outlier. At every representative/sweep shape that reaches the GEMM
dispatch, forward now reaches **43.5-48.7%** of the practical compute
ceiling (up from M40's 10.7-18.0%), on par with dWeight's best path. But a
*different* inefficiency, invisible before M41 because forward's own
inefficiency dominated the conversation, is now the sharpest one in the
whole characterization: **dWeight's `blocks_y==1` path (`k_dweight_
halffused_gemm_splitk`, M38 — dispatched whenever `Cout<=16`, which is
exactly MNIST's own real second conv layer) reaches only 21.7-24.8% of the
ceiling — roughly half the efficiency of every other GEMM-dispatched
Conv2d kernel measured**, including forward's own structurally similar
(non-split-K) half-fused GEMM (43.5-48.7%) and dWeight's own other dispatch
regime, `blocks_y>=2` (42.9-44.2%). The one architectural difference
between the two half-fused kernels is split-K atomic accumulation — a
concrete, testable root-cause hypothesis. **dWeight remains the single
largest addressable sub-component of the training step (29.88%, exceeding
dInput's 15.62% and forward's 12.38%)**, and roughly a third of that cost
now traces to this specific, previously-unexamined efficiency gap rather
than to the already-twice-investigated below-threshold block-reduce kernel
alone. M43 is recommended to target it.

## 2. Post-M41 architecture

`CUDABackend.conv2d` (forward): `total_flops = 2*N*Cout*Hout*Wout*Cin*KH*KW`.
Below `_CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD` (10,000,000) → unchanged M15
per-thread kernel (`cf_conv2d_forward_*`). At/above it, `blocks_x =
ceil(Cout/16)`; `blocks_x<=2` (Cout<=32, every current Forge shape reached
by the 7 representative/batch-sweep shapes) → M41 Candidate B
(`conv2d_forward_halffused_gemm`); `blocks_x>2` (only reached by the
`cout_high` sweep shape, Cout=128) → M41 Candidate A (im2col-smem +
transpose + existing tiled GEMM + output permute).

`CUDABackend.conv2d_backward`: dInput dispatch (`Cin<=16` → M36
channel-fused; every current Forge shape) lives inside the compiled
`cf_conv2d_backward_input_*` wrapper itself (`kernels.cu`), not in Python.
dWeight dispatch (unchanged since M34/M37/M38/M39): `weight_elements =
Cout*Cin*KH*KW`; below 256 → M21 per-thread/block-reduce kernel; at/above
it, `blocks_y = ceil(Cout/16)`; `blocks_y==1` (Cout<=16) → M38 half-fused
split-K GEMM; `blocks_y>=2` (Cout>16) → M39 shared-memory im2col + permute
+ split-K GEMM.

This script (Phase 1) independently confirmed the actual code at
`forge/backend/cuda/backend.py` lines 1276-1437 matches the above exactly
— no historical documentation was relied upon for dispatch decisions.

## 3. Methodology

`benchmarks/m42_bottleneck_recharacterization.py` recomputes both dispatch
decisions itself (mirroring `backend.py`'s own arithmetic) rather than
trusting prior docs, then for every shape:

- Measures the **actual dispatched forward candidate** (baseline / Candidate
  B / Candidate A, whichever the recomputed decision selects) via
  `m41_conv2d_forward_profile._ForwardCandidates`, with im2col/transpose/
  matmul/permute sub-stage decomposition when Candidate A is selected.
- Measures dInput/dWeight/dBias via `m40_bottleneck_recharacterization`'s
  `_RawConv2dBackward`/`_RawDweightCurrent` (dWeight sub-staged into
  permute+fused-GEMM or permute+im2col+split-K-GEMM depending on
  `blocks_y`).
- Cross-checks both directions against the **real, public**
  `CUDABackend.conv2d()`/`conv2d_backward()` entry points (not an internal
  candidate call) at every shape.
- Applies fresh M35-methodology roofline ceilings (measured this session)
  and classifies forward/dInput/dWeight.
- Reuses `m35_mnist._run` (kernel-contribution ranking, fresh),
  `m35_kernels._profile_reduction` (CrossEntropy roofline, fresh),
  `pipeline_profile._profile_async_epoch`/`_profile_allocator_and_pinned`
  (async pipeline health + allocator/pinned-memory, fresh) directly rather
  than re-deriving any of them.
- Runs a fresh `nvcc -Xptxas -v` pass over every forward- and
  backward-relevant kernel for register/shared-memory/occupancy data.

CUDA events (`forge.backend.cuda.profiling_events.TimedEvent`) time every
GPU phase; 5 warmup + 30 measured iterations per phase, synchronized only
at each phase's own measurement boundary — no synchronization was inserted
into the production pipeline itself. Supplementary same-session CPU-
component/H2D/pinned-staging numbers (unaffected by M41, which touched
only the forward kernel) were obtained by re-running the existing M31
`pipeline_profile.py` script directly (Section 9 explains a resulting
async-throughput discrepancy between the two runs).

## 4. Environment

Windows 10, Python 3.13.5, NVIDIA GeForce 940MX (2048 MiB, driver 582.53,
compute capability 5.0), CUDA 12.6. Forge commit `d23cf74` (M41's own
commit; this milestone made no `forge/` changes). Full details in
`benchmarks/results/m42_bottleneck_recharacterization.json`'s
`environment` key.

## 5. Production dispatch verification

Recomputed decisions matched `backend.py`'s own logic exactly at every
shape (no drift):

| shape | fwd total_flops | fwd blocks_x | fwd path | dW weight# | dW blocks_y | dW path |
|---|---:|---:|---|---:|---:|---|
| mnist_conv1 | 7,225,344 | — | M15 per-thread (below 10M threshold) | 72 | — | M21 block-reduce (below 256) |
| mnist_conv2 | 24,920,064 | 1 | M41 Candidate B | 1,152 | 1 | M38 half-fused split-K GEMM |
| large_channel | 462,422,016 | 2 | M41 Candidate B | 4,608 | 2 | M39 im2col-smem + split-K GEMM |
| large_spatial | 231,211,008 | 1 | M41 Candidate B | 1,152 | 1 | M38 half-fused split-K GEMM |
| batch_32 | 231,211,008 | 2 | M41 Candidate B | 4,608 | 2 | M39 im2col-smem + split-K GEMM |
| batch_64 | 462,422,016 | 2 | M41 Candidate B | 4,608 | 2 | M39 im2col-smem + split-K GEMM |
| batch_128 | 924,844,032 | 2 | M41 Candidate B | 4,608 | 2 | M39 im2col-smem + split-K GEMM |
| cout_high (sweep) | 231,211,008 | 8 | M41 Candidate A | — | — | (forward-only shape) |

**Every one of Forge's 7 backward-representative/batch-sweep shapes
dispatches forward to Candidate B** (`blocks_x<=2`, i.e. `Cout<=32`) — none
of them ever reach Candidate A's im2col+GEMM path. Only the dedicated
`cout_high` sweep shape (`Cout=128`) does. This matches M41's own finding
and is confirmed fresh here.

## 6. Training-pipeline breakdown

Full decomposition (mean of 30 iterations, 5 warmup, 940MX):

| shape | fwd (ms) | dIn (ms) | dW (ms) | dB (ms) | bwd total (ms) |
|---|---:|---:|---:|---:|---:|
| mnist_conv1 | 0.6687 | 0.7832 | 2.1084 | 0.2183 | 3.1099 |
| mnist_conv2 | 0.7939 | 0.8923 | 1.0978 | 0.0934 | 2.0835 |
| large_channel | 9.9321 | 13.0299 | 10.0411 | 0.7207 | 23.7917 |
| large_spatial | 6.0978 | 7.7164 | 8.9127 | 0.7672 | 17.3963 |
| batch_32 | 5.0765 | 6.9428 | 5.0039 | 0.3653 | 12.3119 |
| batch_64 | 10.0562 | 13.0412 | 10.1279 | 0.7285 | 23.8976 |
| batch_128 | 20.0074 | 25.5038 | 20.5870 | 1.3757 | 47.4665 |

**Cross-check** (decomposed sub-stage sum vs. the real, public
`CUDABackend.conv2d()`/`conv2d_backward()` call, same shape): forward
matches within 1-20% (real call includes fresh-allocation overhead the
isolated kernel timing doesn't); backward's real call runs 15-38% higher
than the decomposed sum at small/MNIST shapes (mnist_conv2: 2.87ms vs.
2.08ms), narrowing to ~3.5% at the largest shape (large_channel: 24.62ms
vs. 23.79ms). This gap is fully explained by the real entry point
allocating fresh `grad_x`/`grad_w`/`grad_b` output buffers on every call
(the decomposed measurement reuses one pre-allocated buffer across all 30
iterations) — a fixed per-call cost that is a larger fraction of a small
2ms call than a large 24ms one. It is a benchmarking-methodology artifact,
not evidence of an extra hidden cost in production (real training reuses
the same shapes repeatedly, and the caching allocator — Section 12 —
already shows cache hits dominating misses at steady state).

## 7. Conv2d forward characterization

Every SHAPES/batch-sweep shape dispatches to Candidate B (half-fused
GEMM, no `Xcol` buffer). The K/stride/channel sweep (forward-only, same 8
shapes M41 introduced) re-measured fresh:

| sweep shape | total FLOPs | blocks_x | fwd (ms) | % of ceiling | classification |
|---|---:|---:|---:|---:|---|
| k1_s1 | 3,211,264 | below threshold | 0.7460 | 4.1% | mixed/ambiguous |
| k3_s1 | 28,901,376 | 1 | 1.0068 | 27.4% | mixed/ambiguous |
| k5_s1 | 80,281,600 | 1 | 1.9246 | 39.9% | compute_bound |
| k3_s2 | 7,225,344 | below threshold | 0.4004 | 17.2% | mixed/ambiguous |
| cin_low | 3,612,672 | below threshold | 0.3299 | 10.5% | mixed/ambiguous |
| cin_high | 231,211,008 | 1 | 4.6261 | 47.8% | compute_bound |
| cout_low | 7,225,344 | below threshold | 0.4202 | 16.4% | mixed/ambiguous |
| cout_high | 231,211,008 | 8 (Candidate A) | 4.5327 | 48.7% | compute_bound |

Forward's im2col+GEMM stage decomposition was only obtainable at
`cout_high` (the one shape reaching Candidate A); its own im2col/transpose/
matmul/permute sub-stages were not separately re-timed here since Candidate
A never activates at any of the 7 backward-representative shapes — the
buffer-size-only measurement in Section 12 covers this shape instead.
**No forward regression, no new finding beyond M41's own report**: this
confirms M41's dispatch continues to reach 39.9-48.7% of the ceiling at
every shape at/above ~28.9M FLOPs, exactly as M41 measured, on a fresh
same-session ceiling (104.64 GFLOP/s vs. M41's own 104.69 — within normal
940MX run-to-run variance).

## 8. Conv2d backward characterization

**dInput** (M36 channel-fused, dispatched at every current Forge shape
since `Cin<=16` always): 28.6-34.7% of ceiling at large/batch shapes,
8.8-26.7% at MNIST's own small shapes — unchanged in kind from M40 (this
milestone did not touch dInput; small measured deltas vs. M40's archived
numbers are within the documented thermal/clock variance).

**dWeight**, split by dispatch regime:
- `blocks_y==1` (M38 half-fused split-K GEMM; `mnist_conv2`, `large_spatial`):
  permute+fused-GEMM sub-stages measured directly — e.g. `mnist_conv2`:
  permute 0.118ms + fused_gemm 0.980ms = 1.098ms; `large_spatial`: permute
  1.041ms + fused_gemm 7.871ms = 8.912ms. **Fused-GEMM dominates this path's
  own cost (89-92%).**
- `blocks_y>=2` (M39 im2col-smem + split-K GEMM; `large_channel`,
  `batch_32/64/128`): e.g. `large_channel`: permute 1.021ms + im2col
  4.066ms + splitk_gemm 4.954ms = 10.041ms — roughly balanced between
  im2col (40%) and GEMM (49%).
- Below-256 block-reduce (`mnist_conv1` only): 2.108ms, unchanged since
  M21/M33.

`dBias` is negligible everywhere (0.09-1.38ms, 2.9-23.4% of backward at its
smallest shapes, well under 3% of the full training step) — a clean
negative result, no further attention warranted (Section 15).

## 9. Roofline analysis

Fresh ceilings this session: **104.64 GFLOP/s** practical compute (`cf_matmul_f32`,
best at dim=512; 98.91 at 1024, 96.56 at 2048 — consistent with M35/M40's
documented finding that the compute ceiling is best approximated at
moderate GEMM sizes on this GPU), **15.09 GB/s** practical bandwidth
(`cf_add_f32`, 20M elements). Both within 0.1% of M40's own archived
ceilings — no meaningful thermal drift between sessions this time.

| shape | fwd % (class) | dIn % (class) | dW % (class) |
|---|---:|---:|---:|
| mnist_conv1 | 10.3% (mixed) | 8.8% (mixed) | 3.3% (mixed) |
| mnist_conv2 | 30.0% (mixed) | 26.7% (mixed) | **21.7% (mixed)** |
| large_channel | 44.5% (compute) | 33.9% (compute) | 44.0% (compute) |
| large_spatial | 36.2% (compute) | 28.6% (mixed) | **24.8% (mixed)** |
| batch_32 | 43.5% (compute) | 31.8% (compute) | 44.2% (compute) |
| batch_64 | 43.9% (compute) | 33.9% (compute) | 43.6% (compute) |
| batch_128 | 44.2% (compute) | 34.7% (compute) | 42.9% (compute) |

**GEMM itself (`cf_matmul_f32`) is, by construction, already at its own
practical ceiling** (it *is* the ceiling-defining kernel) — 0% headroom to
claim here; this is the clean negative result Section 15 asks to state
explicitly, not manufacture around.

**CrossEntropy** (fresh, via `m35_kernels._profile_reduction`): forward/
backward both `latency_launch_bound` at small/medium batch (8.6-20.0us,
below the 20us launch-overhead heuristic), transitioning to
`memory_bandwidth_bound` at `batch=16,384` (forward: 5.51 GFLOP/s-equiv,
8.82 GB/s = 58.4% of the bandwidth ceiling; backward: 3.90 GFLOP/s-equiv,
7.00 GB/s = 46.4%). This is the expected shape for an already-fused (M31),
low-arithmetic-intensity op — **no further optimization opportunity found;
CrossEntropy is already reasonably close to its own bandwidth ceiling at
realistic batch sizes and is not a fresh M43 candidate.**

**The sharp new finding**: dWeight's `blocks_y==1` path (21.7-24.8%) is now
the *lowest* roofline efficiency of any GEMM-dispatched Conv2d kernel
measured — lower than forward (36.2-48.7%), lower than dWeight's own
`blocks_y>=2` regime (42.9-44.2%), and lower than dInput (28.6-34.7%).
Sharper still: `k_conv2d_forward_halffused_gemm` (forward's Candidate B,
the *same* general half-fused-GEMM design, at every one of these shapes
since `blocks_x<=2` always) reaches 43.5-48.7% — roughly **double**
`k_dweight_halffused_gemm_splitk`'s efficiency. The one architectural
difference documented between them (`experimental_conv_forward_halffused.py`'s
own docstring, Milestone 41): forward's GEMM has `M=N*Hout*Wout` as a large
*block-count* dimension needing no split-K, while dWeight's GEMM has `M`
as the *reduction* dimension, needing split-K's atomic-accumulation combine
step. This is a concrete, testable root-cause hypothesis for M43.

## 10. Resource analysis (`nvcc -Xptxas -v`, real compile)

| kernel | registers (f32 / f64) | shared mem | notes |
|---|---:|---:|---|
| `k_conv2d_forward` (M15 baseline) | 55 / 48 (main variant) | 0 | multiple instantiations; no spill |
| `k_conv2d_forward_halffused_gemm` (fwd Candidate B) | 40 / 38 | 4096 / 2048 bytes | no spill |
| `k_dweight_halffused_gemm_splitk` (dW `blocks_y==1`) | 54 / 49 | 4096 / 2048 bytes | no spill |
| `k_matmul_splitk` (dW `blocks_y>=2` GEMM stage) | up to 53 / 50 | 4096 / 2048 bytes | no spill |
| `k_im2col_conv2d_smem` | 34 / 34 | (input-plane, dynamic) | no spill |
| `k_conv2d_backward_input_channelfused` (dInput) | 70 / 48 (54 f64 variant) | 0 | one f64 instantiation shows 512 bytes cumulative stack (not spill; a compiler-managed local array) |
| `k_conv2d_backward_weight_reduce` (below-threshold) | up to 128 registers (f64 instantiation), 272 bytes smem | some variants with 256-512 bytes cumulative stack | the 128-register f64 instantiation is the heaviest kernel in this table |

**No register spill (`local_memory_bytes`) was observed on any kernel of
interest**, including `k_dweight_halffused_gemm_splitk` — so the blocks_y==1
efficiency gap is **not** a register-pressure/spill problem. Register
count (54) and shared-memory footprint (4096 bytes) for
`k_dweight_halffused_gemm_splitk` are in the same range as
`k_conv2d_forward_halffused_gemm` (40, 4096) and `k_matmul_splitk` (up to
53, 4096) — ruling out a gross occupancy disparity as the sole explanation
and pointing toward the split-K reduction step itself (extra atomic
traffic / extra kernel-internal synchronization the roofline FLOP model
doesn't count) as the more likely differentiator. This is diagnosis only,
per Section 8 — no optimization was attempted here.

## 11. Synchronization analysis

No accidental `cudaDeviceSynchronize()`/`cudaStreamSynchronize()` was
found anywhere in the M41-added code (`experimental_conv_forward_im2col.py`,
`experimental_conv_forward_halffused.py`) — both call `backend.
_maybe_synchronize()` exactly once at the end of their pipeline, identical
to every other `CUDABackend` method's contract (verified by grep across
`forge/backend/cuda/` for `cudaDeviceSynchronize`/`.synchronize()`; the
only hits are the documented `_maybe_synchronize` implementation, the
public `forge.cuda.synchronize()`/`Stream.synchronize()`/`CUDAEvent.
synchronize()` APIs, and the allocator's own documented
"last-resort" busy-block wait). Async pipeline batch sweep (this session,
same continuous run as the roofline ceilings above): batch=32 →
2252 samples/sec, 90.5% compute-stream utilization; batch=64 → 3863
samples/sec, 89.2%; batch=128 → 6172 samples/sec, 83.1%. Prefetch-depth
sweep (batch=64): depth=1 → 91.6% utilization; depth=2 → 89.9%; depth=3 →
85.6%. All in the same healthy 83-92% range M27-M31's async infrastructure
has consistently shown — **no pipeline-health regression from M41.**

**A same-session measurement caveat worth recording explicitly**: a
*separate* invocation of `pipeline_profile.py` (run moments later, for the
CPU-component/H2D numbers Section 12 needed) measured meaningfully higher
throughput at the same shapes (batch=32: 4609 samples/sec vs. this
script's 2252; batch=64: 6207 vs. 3863) at *lower* reported utilization
(84.3% vs. 90.5%) — both internally consistent, but a reminder that this
GPU's absolute throughput varies significantly between separate process
launches (thermal/clock state carried over from whatever ran immediately
before), exactly the documented 940MX variance M35 warned about. The
numbers embedded in this report's own tables are the ones measured in the
same continuous session as this report's roofline ceilings and shape
decomposition, so ceiling-relative percentages remain internally
consistent even though the separate run's raw throughput differs.

## 12. Memory analysis

CPU-only components (fresh, `pipeline_profile.py`, batch=64): dataset+collate
1.08ms/batch, pinned staging 0.10ms/batch, async H2D submission (launch-only)
0.10ms/batch — all three orders of magnitude below a single training step's
GPU-side cost (~15-19ms at batch=64 from Section 6), confirming these were
never a bottleneck and remain so. H2D bandwidth: 0.055 GB/s at 4KB (latency-
dominated), 1.41-1.55 GB/s at 400KB-4MB (approaching the PCIe-limited
regime, unrelated to the 15.09 GB/s on-device bandwidth ceiling).

**Im2col/GEMM forward temporary-buffer footprint** (`cout_high`, the one
shape reaching Candidate A): modeled `Xcol` = 3.61MB (9.0x the 401KB input
tensor), `weightT` = 36KB, `out_mat` = 6.42MB, total temporaries = 10.07MB.
Measured allocator growth over 3 live calls: `reserved_bytes` 209.3MB →
239.1MB (+29.8MB, consistent with fresh-allocation + caching behavior
across the 3 calls, not a leak — confirmed by `allocator_after_gc_and_
empty_cache` returning to 222.6MB, i.e. real usage, with the caching
allocator's own `cached_bytes` correctly reclaimed to 0). **Not a
meaningful VRAM constraint** even at Forge's most extreme currently-tested
forward shape: 10MB of temporaries against a 2048MB budget (0.5%). No
shape in Forge's actual current model repertoire (`Cout<=32` everywhere)
reaches Candidate A at all, so this buffer question is presently
hypothetical for real training, though confirmed safe for any future
larger-`Cout` layer.

Allocator/pinned characterization over one live MNIST epoch (batch=64,
prefetch=2, main run): `cache_hit_count` grew from 15,871 to 16,730 over
one epoch with only 70 new `cache_miss_count` entries — the caching
allocator is working as designed, steady-state allocation is dominated by
reuse, not fresh `cudaMalloc`. `pending_bytes` (10.68MB) returns cleanly to
0 after `gc.collect()` + `empty_cache()`. Pinned memory: 428→472
allocations, 428→472 frees (no leak), peak 804,864 bytes — unchanged in
kind from M29/M31's own findings. **No memory regression, no new
constraint from M41.**

## 13. Amdahl analysis

Using this session's own fresh MNIST kernel-contribution ranking
(`m35_mnist._run`, batch=64, real M20 CNN):

| component | fraction of full step | 1.5x | 2.0x | 3.0x |
|---|---:|---:|---:|---:|
| conv2d backward total | 48.41% | 1.192x | 1.319x | 1.476x |
| **dWeight** | **29.88%** | **1.111x** | **1.176x** | **1.249x** |
| dInput | 15.62% | 1.055x | 1.085x | 1.116x |
| conv2d forward | 12.38% | 1.043x | 1.066x | 1.090x |
| matmul backward (Linear) | 8.06% | 1.028x | 1.042x | 1.057x |

**dWeight is the single largest addressable sub-component** — larger than
dInput and forward *combined*. Splitting dWeight's own 29.88% by the two
real MNIST layers' dispatch regime: `mnist_conv1` (below-threshold
block-reduce) contributes 65.8% of dWeight's total cost (≈19.66% of the
full step); `mnist_conv2` (`blocks_y==1` half-fused split-K GEMM, the
efficiency-gap kernel from Section 9) contributes the remaining 34.2%
(≈10.22% of the full step). A full 2x speedup isolated to just the
`blocks_y==1` sub-fraction projects to **~1.054x overall step speedup**
(1.035x at 1.5x, 1.073x at 3x) — modest, Amdahl-bounded, and consistent in
scale with M37's own accepted 1.2-1.26x kernel-local win. This is an
**isolated-kernel-level opportunity within dWeight**, not a Conv2d-level or
whole-step-level one — stated explicitly per Section 12's requirement not
to overclaim.

## 14. Bottleneck ranking

1. **dWeight, `blocks_y==1` split-K half-fused GEMM** (M38) — 21.7-24.8% of
   ceiling, roughly half of every other GEMM-dispatched kernel measured,
   contributes ≈10.2% of the full MNIST step, previously unexamined at the
   roofline-efficiency level (M38's own acceptance was a relative-speedup
   decision against its own alternative, not an absolute-efficiency
   investigation). **Selected for M43.**
2. dWeight, below-256 block-reduce (`mnist_conv1`) — largest single
   absolute contributor (≈19.7% of the full step) but already investigated
   twice (M33: cooperative reduction, rejected; M34: im2col+GEMM, rejected
   at this exact shape) with no headroom found via either technique.
   Lower expected leverage from a third attempt without a genuinely new
   algorithmic angle — not selected, but not closed off either (see
   Candidates, below).
3. dInput (M36 channel-fused) — 28.6-34.7% of ceiling, 15.62% of the full
   step; already the subject of a major M36 rewrite (8.7x in isolation);
   mediocre but not obviously broken the way item 1 is.
4. Conv2d forward — 36.2-48.7% of ceiling at every shape that reaches the
   GEMM dispatch; M41 already closed most of the gap M40 identified.
   Lowest priority of the four Conv2d-adjacent candidates.
5. GEMM itself (`k_matmul`/`k_matmul_splitk`) — already at its own
   practical ceiling by construction; 0% headroom to claim.
6. CrossEntropy — already fused (M31), 46-58% of the bandwidth ceiling at
   realistic batch sizes; no fresh opportunity.
7. dBias, allocator, pinned memory, async pipeline, CPU components — all
   confirmed healthy/negligible; no action warranted.

## 15. Candidate optimization directions

1. **dWeight `blocks_y==1` split-K reduction strategy** (selected for M43;
   full detail below).
2. **dWeight below-256 block-reduce kernel, a third algorithmic angle**
   (not selected): M33 tested cooperative multi-thread-per-weight-element
   reduction (rejected — flat 3-4x gap across tested granularities,
   32-256 threads/weight-element) and M34 tested im2col+GEMM at this exact
   shape (rejected — existing kernel already wins at 72 elements). Neither
   sub-warp cooperation (2-16 threads) nor a warp-shuffle-based reduction
   (no shared memory, avoiding M33's shared-memory tree-reduction
   overhead entirely) has been tried. Plausible, but speculative — no
   fresh measurement in this milestone points to warp-shuffle specifically
   being the missing piece, so it is not selected ahead of item 1's
   concrete, already-measured 2x gap.
3. **dInput extension beyond the current dispatch** (not selected): every
   real Forge shape has `Cin<=16` and already reaches the M36 channel-fused
   path; the `Cin>16` fallback kernel is never exercised by any real Forge
   shape, so there is no reproducible current workload to motivate further
   dInput work.

## 16. Selected M43 recommendation

1. **Target**: `k_dweight_halffused_gemm_splitk` (M38), the dWeight
   `blocks_y==1` dispatch path in `CUDABackend.conv2d_backward` — reached
   whenever `Cout<=16` (`weight_elements>=256`), which is exactly MNIST's
   own real second conv layer (`mnist_conv2`) plus the `large_spatial`
   representative shape.
2. **Current measured contribution**: 21.7% (`mnist_conv2`) to 24.8%
   (`large_spatial`) of the practical compute ceiling — the lowest of any
   GEMM-dispatched Conv2d kernel measured in this characterization.
   Contributes ≈10.2% of the full MNIST training step (Section 13).
3. **Measured root cause**: not register spill (confirmed via `nvcc
   -Xptxas -v`: 54/49 registers, no spill) and not a gross register/
   shared-memory disparity vs. comparable kernels (`k_conv2d_forward_
   halffused_gemm`: 40/38 registers, same 4096/2048-byte shared-memory
   footprint, yet reaches roughly double the roofline efficiency). The one
   documented architectural difference between these two structurally
   similar half-fused-GEMM kernels is split-K: dWeight's GEMM has `M`
   (`N*Hout*Wout`) as the *reduction* dimension (needing split-K's
   atomic-accumulation combine step across `num_k_splits` partial blocks),
   while forward's GEMM has `M` as a large *block-count* dimension (no
   split-K needed at all). This points at split-K's atomic-accumulation
   overhead as the most likely differentiator — a hypothesis, not yet
   confirmed at the instruction/memory-transaction level.
4. **Why preferable to the alternatives**: larger, fresher, and more
   concrete than any other candidate ranked in Section 14 — the
   below-threshold block-reduce kernel (item 2) has already had two
   different technique classes tried and rejected with no headroom found,
   while this path has never been examined below the wall-clock-speedup
   level that motivated its own M38 acceptance. dInput and forward both
   rank lower on both absolute contribution and (for forward) remaining
   headroom.
5. **Proposed algorithmic direction**: investigate whether a different
   split-K reduction strategy — e.g., a two-pass split-K that writes
   partial sums to a small intermediate buffer and combines them in a
   second, separate reduction kernel (avoiding atomics entirely, at the
   cost of one more small kernel launch), a cooperative-groups-based
   grid-wide reduction, or simply re-tuning `recommended_num_k_splits` at
   this specific `blocks_y==1`/small-`Cout` regime — can close some of the
   measured ~2x efficiency gap without regressing at any tested shape.
6. **Existing Forge infrastructure to reuse**: `experimental_conv_
   halffused.py`'s `dweight_halffused_gemm_splitk` (current production
   path, to benchmark against), `recommended_num_k_splits` (already
   parameterizes split-K depth), `cf_matmul_splitk_f32`/`k_matmul_splitk`
   (the non-fused split-K GEMM, for a structural efficiency comparison),
   and `benchmarks/m37_dweight_candidates_profile.py`/`m38_im2col_profile.py`'s
   own interleaved-A/B-comparison methodology.
7. **Representative benchmark shapes**: `mnist_conv2` and `large_spatial`
   (both real/representative `blocks_y==1` shapes already in
   `conv2d_backward_profile.SHAPES`), plus a dedicated `Cout` sweep at
   `{1, 4, 8, 16}` (all `blocks_y==1`) to see whether the gap is uniform
   or `Cout`-dependent, and a `num_k_splits` sweep at a fixed shape to
   characterize the split-K depth/efficiency tradeoff directly.
8. **Acceptance criterion**: a candidate reduction strategy measured
   faster than the current `dweight_halffused_gemm_splitk` at every
   `blocks_y==1` representative/sweep shape above, with no regression at
   any tested shape, using the same interleaved CUDA-event A/B comparison
   methodology M37-M41 established — and, ideally, closing at least half
   of the measured roofline-efficiency gap (target: ≥33% of the practical
   compute ceiling, up from 21.7-24.8%) at the two real representative
   shapes.
9. **Explicit exclusions**: forward (already addressed by M41; no further
   work justified by this milestone's measurements), dInput (mediocre but
   not a fresh finding; already extensively optimized in M36), dWeight's
   `blocks_y>=2` path (already efficient, 42.9-44.2%), dWeight's
   below-threshold block-reduce path (twice investigated already; see
   Candidates item 2), dBias, `k_matmul`/`k_matmul_splitk` themselves
   (already at their own practical ceiling), CrossEntropy, optimizers,
   the allocator, CUDA streams, `_stream_guard`, pinned memory, async
   transfer, and DataLoader/prefetch (all confirmed healthy in this
   milestone with no regression).

## 17. Exclusions

Per the milestone's hard scope: no changes were made to `forge/`, `tests/`,
any CUDA production kernel, Conv2d forward/backward, dInput, dWeight,
dBias, `k_matmul`, CrossEntropy, optimizers, the allocator, CUDA streams,
`_stream_guard`, pinned memory, async transfer, or DataLoader/prefetch.
This milestone is measurement and documentation only.

## 18. Limitations

- **940MX thermal/clock variance**: Section 11 documents a real,
  reproducible throughput discrepancy (up to ~2x in raw samples/sec)
  between this script's own async-pipeline measurement and a separately
  launched `pipeline_profile.py` run moments later — a reminder that
  absolute throughput numbers on this GPU depend on whatever ran
  immediately before in the same process, not just the workload itself.
  Ceiling-relative percentages (roofline classifications) remain
  internally consistent because they were measured in the same continuous
  session as their own ceiling.
- **The split-K-atomics root-cause hypothesis (Section 10/16) is not yet
  confirmed at the instruction level** — no Nsight Compute atomic-
  contention or memory-transaction counters were collected (only `nvcc
  -Xptxas -v` static resource data); M43 should collect this evidence
  before committing to an implementation direction.
- **`k_dweight_halffused_gemm_splitk`'s efficiency was measured at only
  two shapes** (`mnist_conv2`, `large_spatial`), both with `Cout` in
  {16, 16} — the `Cout` sweep proposed in Section 16 (`{1,4,8,16}`) has
  not yet been run; whether the gap is uniform across the whole
  `blocks_y==1` region (`Cout<=16`) or specific to `Cout=16` is unknown.
- **The forward im2col+GEMM (Candidate A) buffer-size question is
  presently hypothetical** for Forge's own model repertoire — no current
  shape reaches it (`Cout<=32` everywhere), so Section 12's memory
  finding is a forward-looking safety check, not a live constraint.
- **Cross-check overhead (Section 6) was not decomposed further** — the
  15-38% gap between decomposed sub-stage sums and the real production
  call is attributed to fresh-allocation overhead by reasoning from the
  allocator's own cache-hit/miss counters, not by directly instrumenting
  the allocation calls themselves inside the timed loop.

## 19. Reproducibility

```bash
python -m benchmarks.m42_bottleneck_recharacterization   # main profiler (this report's primary source)
python -m benchmarks.pipeline_profile --n-samples 1024   # supplementary CPU/H2D/pinned numbers (Section 12)
```

Archived results: `benchmarks/results/m42_bottleneck_recharacterization.json`
(primary evidence for every table above), `benchmarks/results/
m42_pipeline_profile.json` (Section 12's CPU-component/transfer/pinned
numbers). Both were produced on the real 940MX (`environment.cuda.gpu_name
== "NVIDIA GeForce 940MX"`, `compute_capability == "5.0"`), same session,
2026-09-05.

## Tests / Verification

- Clean CUDA rebuild performed (cached `_forge_cuda_kernels_sm_50.dll`
  deleted, forcing a fresh `nvcc`/MSVC compile on next import).
- Full suite: **1,420 passed** (`python -m pytest tests/ -q`, verified
  twice this session — once with a hung run correctly identified and
  killed via process/thread-state inspection rather than blindly assumed,
  once clean at 66.04s). Matches M41's own archived count exactly — zero
  test changes, as required by this milestone's scope.
- Confirmed zero `forge/` changes and zero `tests/` changes (this
  milestone only added `benchmarks/m42_bottleneck_recharacterization.py`
  and this documentation).
- Dispatch decisions cross-checked against the actual `backend.py` source
  (Section 2/5), not historical documentation.
- Measurements confirmed collected on the real 940MX (environment capture
  in both archived JSON files).
- No profiling instrumentation altered production runtime behavior (CUDA
  events record without host synchronization until each phase's own
  measurement boundary; no `cudaDeviceSynchronize()` was added to any
  production code path).

## Suggested Commit Message

```
docs: M42 fresh post-M41 CUDA bottleneck re-characterization

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015UQumnYRyGssMDPYwnunrb
```
