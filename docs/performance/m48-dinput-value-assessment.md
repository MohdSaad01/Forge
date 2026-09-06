# M48 — CUDA dInput Low-Cin Value Assessment (ACCEPTED)

## 1. Executive summary

M47 selected the M36 channel-fused `dInput` kernel's low-`Cin` regime
(`Cin=1`, `mnist_conv1`'s own real shape) as the next candidate — the
worst live-candidate roofline efficiency measured in that session (4.7% of
the practical compute ceiling). M48's brief explicitly required treating
that as a hypothesis to test, not a foregone conclusion: value-assess
first, design only if justified, and stop if the evidence shows diminishing
returns.

**The evidence justified proceeding, and the resulting candidate is
ACCEPTED.**

1. **Real-workload relevance is narrow but real.** `Cin=1` matters to
   exactly one real Forge workload: `examples/mnist/model.py`'s first
   conv layer — the only trained network in the repository. Every other
   `Cin=1` shape in the codebase is test-only correctness coverage or a
   deliberately constructed sweep point, not a second real workload.
2. **The root cause was more nuanced than M47's own hypothesis.**
   M47 attributed the low efficiency to register pressure (the channel-
   fused kernel's accumulator array is sized for 16 channels regardless of
   the runtime `Cin`). A fresh `nvcc -Xptxas -v` and real
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` query at `Cin=1` found
   that shrinking the accumulator array to a `Cin=1`-specialized template
   parameter only drops f32 register usage 54→48 and occupancy 4→5
   blocks/SM (50%→62.5%) — a real but modest gain, not the dramatic
   jump a pure register-pressure story predicts. The larger effect turned
   out to be per-thread **instruction overhead**: the unspecialized
   kernel's `#pragma unroll`ed accumulator loop still emits 16
   runtime-checked iterations per `Cout*KH*KW` outer-loop step even when
   only the first is ever useful at `Cin=1`.
3. **The candidate measured a large, reproducible, bit-exact-correct win.**
   A same-session, round-robin-interleaved, order-independent A/B
   (reproduced across 2 independent script runs) measured a
   **2.18x–2.53x isolated kernel speedup** at `Cin=1` across 4 shapes.
   Roofline efficiency at `mnist_conv1`'s own shape rose from 4.7% to
   **28.0%** of the practical compute ceiling — comfortably clearing
   M47's own ≥15% acceptance bar.
4. **The end-to-end payoff is real and measured, not just projected.**
   The actual `conv2d_backward()` call at `mnist_conv1` is **1.155x–1.158x
   faster**, directly measured (not an Amdahl projection). A fresh,
   same-session fraction reconstruction put dInput's full-training-step
   share at ~15.4% before this milestone (M47's own prior-session figure
   was 11.75%) — Amdahl projects **~1.09x–1.10x whole-training-step
   improvement** at the real measured kernel speedup, comparable to or
   better than M43's own accepted full-pipeline win (1.02x–1.04x /
   1.11x–1.24x component).
5. **No regression.** `Cin=2` (still the M36 channel-fused path,
   byte-for-byte unchanged) and `mnist_conv2`'s own `Cin=16` shape were
   both re-confirmed unaffected.

**Decision: ACCEPT.** `k_conv2d_backward_input_channelfused_lowcin<T,
CIN_MAX>` (`kernels.cu`), instantiated at `CIN_MAX=1`, is now production,
dispatched at `Cin<=CONV2D_DINPUT_LOWCIN_MAX_CIN` (1) — the one real Forge
shape it was measured at — ahead of the unchanged M36 channel-fused path
(`Cin<=16`) and the M32 fallback (`Cin>16`).

## 2. Project-direction decision

Per CLAUDE.md's "do not optimize for the sake of optimization" and the
brief's own Section 21 stop condition, this milestone was structured to
be able to stop at any of several points without implementing anything:
after the real-workload-relevance check (if `Cin=1` turned out irrelevant
even to MNIST), after the fresh occupancy/register profile (if it showed
no credible technical path — mirroring M46's actual outcome for a
different kernel), or after the isolated candidate benchmark (if the
speedup were marginal). None of those stop conditions triggered: relevance
is real (if narrow), the profile showed a large speedup was structurally
plausible (instruction overhead, not just an unmovable occupancy ceiling),
and the measured result cleared every bar the brief set. The milestone
therefore proceeded through DESIGN → BENCHMARK → ACCEPT, all within the
scope the brief allows.

## 3. Current production architecture (before this milestone)

`CUDABackend.conv2d_backward`'s `dInput` dispatch (`kernels.cu`'s
`CONV2D_DINPUT_CHANNELFUSED_MAX_CIN = 16`, unchanged since M36): `Cin<=16`
→ `k_conv2d_backward_input_channelfused` (one thread per `(n,h,w)`, all
`Cin` accumulators register-resident in a `MAX_CIN_REG=16`-sized array,
regardless of the runtime `Cin`); `Cin>16` → the original M32
`k_conv2d_backward_input` (never reached by any real Forge shape).
Verified directly against `kernels.cu` and `backend.py` at the start of
this milestone — unchanged since M36/M44/M47.

## 4. Real-workload relevance

Grepped `examples/`, `tests/`, `benchmarks/` for every `Conv2d(...)`
instantiation in the repository (see `benchmarks/
m48_dinput_value_assessment.py`'s `REAL_WORKLOAD_RELEVANCE`, hardcoded
from this session's own grep since re-deriving it at runtime adds no
value over the one-time finding):

- **Real Forge networks**: exactly one — `examples/mnist/model.py`
  (`Conv2d(1, 8, kernel_size=3) → ReLU → MaxPool2d(2) → Conv2d(8, 16,
  kernel_size=3)`). Its first layer is `mnist_conv1`'s own `Cin=1`
  shape — the *only* real `Cin=1` workload in the repository.
- **Synthetic/test-only `Cin=1` shapes**: `tests/test_conv.py`,
  `tests/test_cuda_conv.py`, `tests/test_cuda_streams.py`,
  `tests/test_lifetime.py`, `tests/test_serialization.py`,
  `tests/test_cuda_persistence.py`,
  `tests/test_sequential_flatten_dropout_integration.py`,
  `tests/test_conv_trainer_integration.py` — small `Cin=1` layers on tiny
  inputs (6x6–8x8, `N<=4`), used purely for API/correctness coverage,
  never as a performance workload. `benchmarks/*.py` sweep files use
  `Cin=1` only as one point in a deliberately constructed sweep
  (`Cin` in {1,2,4,8,16,...}), not evidence of a second real workload.

**Conclusion (measured fact, not inferred)**: `Cin=1` matters to exactly
one real Forge workload — MNIST's own first conv layer. This is not a
broadly representative CNN input pattern within Forge's current scope;
optimizing it helps MNIST specifically. This narrows but does not
eliminate the case for optimizing it, since MNIST is Forge's only
executable end-to-end demonstration of the framework today.

## 5. Fresh baseline measurements

Clean CUDA rebuild (`_forge_cuda_kernels_sm_50.dll` deleted and
recompiled, ~10s) before any measurement. Full suite passed at 1,550
(unchanged from M47) before this milestone's own changes.

Fresh practical ceilings (M35 methodology, this session): **104.6–104.8
GFLOP/s** compute, **15.09 GB/s** bandwidth — consistent with every prior
session (M35/M40/M42/M44/M47: ~104–105 GFLOP/s / ~15.1 GB/s).

**`nvcc -Xptxas -v`** (this session, `_run_nvcc_ptxas_verbose` in
`benchmarks/m48_dinput_value_assessment.py`):

| kernel | dtype | registers | stack frame / spills |
|---|---|---:|---|
| `k_conv2d_backward_input_channelfused` (production, pre-M48) | f32 | 54 | 0 bytes, no spills |
| `k_conv2d_backward_input_channelfused` (production, pre-M48) | f64 | 70 | 0 bytes, no spills |
| `k_conv2d_backward_input_channelfused_lowcin<T,1>` (M48 candidate) | f32 | 48 | 0 bytes, no spills |
| `k_conv2d_backward_input_channelfused_lowcin<T,1>` (M48 candidate) | f64 | 48 | 0 bytes, no spills |

(Note: M47's report cited "70 registers" for the channel-fused kernel
without separating f32/f64 — this session's fresh measurement shows that
figure is the f64 count; f32, the dtype MNIST actually trains in, is 54.)

**Real occupancy query** (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
256 threads/block, 0 shared memory — the kernel's actual launch
configuration):

| kernel | max active blocks/SM | occupancy |
|---|---:|---:|
| `k_conv2d_backward_input_channelfused` (pre-M48) | 4 | 50.0% |
| `k_conv2d_backward_input_channelfused_lowcin<T,1>` (M48) | 5 | 62.5% |

## 6. Low-Cin characterization

Fresh `Cin` sweep (fixed `N=16,Cout=16,H=W=28,K=3`, post-M48 live
dispatch):

| Cin | path | d_input (ms) | % of ceiling |
|---:|---|---:|---:|
| 1 | **M48 lowcin1 (new)** | 0.1234 | **28.0%** |
| 8 | M36 channel-fused (unchanged) | 1.1538 | 23.9% |
| 16 | M36 channel-fused (unchanged) | 1.9683 | 28.0% |
| 17 | M32 fallback (unused by any real shape) | 4.6134 | 12.7% |
| 32 | M32 fallback (unused by any real shape) | 9.0370 | 12.2% |
| 64 | M32 fallback (unused by any real shape) | 19.5241 | 11.3% |

`Cin=1`'s roofline efficiency (28.0%) now sits *above* `Cin=8`'s (23.9%)
and matches `Cin=16`'s (28.0%) — the low-`Cin` efficiency cliff M44/M47
both measured (4.7%–8.3% at `Cin=1`) is gone.

## 7. Root-cause analysis

M47 hypothesized the efficiency cliff was primarily register-pressure/
occupancy-driven: the channel-fused kernel's accumulator array is sized
for `MAX_CIN_REG=16` regardless of the runtime `Cin`, so `Cin=1` pays the
full register cost with none of the multi-channel amortization. This
milestone's fresh measurement (Section 5) shows that story is only
partially correct:

- Register reduction going from a 16-slot to a 1-slot accumulator array
  is **modest** (f32: 54→48, an 11% cut) — most of the kernel's register
  pressure comes from address arithmetic and loop state, not the
  accumulator array itself.
- Occupancy rises correspondingly modestly (4→5 blocks/SM, 25% more
  resident warps) — real, but far short of reaching this device's
  theoretical maximum (`65536 registers / 32 registers-per-thread-at-full-
  occupancy ≈ 8 blocks/SM` would need a ~32-register kernel; 48 registers
  is still well above that).
- Yet the **measured kernel speedup (2.18x–2.53x) is far larger** than a
  25% occupancy increase alone would predict. The dominant effect is
  **per-thread instruction count**: the unspecialized kernel's `#pragma
  unroll`ed `ci` loop (`for (int ci = 0; ci < MAX_CIN_REG; ++ci) { if (ci
  >= Cin) break; ... }`) still emits all 16 unrolled iterations —
  including 15 dead `if`-checks-then-break at `Cin=1` — at *every* one of
  `Cout*KH*KW` outer-loop steps. `nvcc` cannot eliminate this at compile
  time because `Cin` is a runtime parameter, not a template parameter, in
  the production kernel. Turning `CIN_MAX` into an actual template
  parameter (instantiated at 1) lets the compiler generate exactly one
  accumulator update per outer-loop step, with zero dead branches.

This matches the brief's own diagnostic guidance (Section 6): the kernel
was not simply grid-size-limited (total launched threads, `N*H*W`, is
identical regardless of `Cin` — never small) or purely occupancy-bound
(the occupancy delta alone under-explains the speedup) — it was
**instruction overhead**, confirmed directly rather than assumed.

## 8. Amdahl / value analysis

Real, measured kernel-level speedups (Section 10) plugged into Amdahl's
law against two independent "before" fraction estimates:

| Hypothetical/measured dInput speedup | Full-step max (M47's 11.75% fraction) | Full-step max (fresh 15.37% reconstruction) |
| --------------------------: | ----------------: | ----------------: |
| 2.18x (measured, cin1_strided) | 1.0679x | 1.0907x |
| 2.26x (measured, mnist_conv1) | 1.0701x | 1.0937x |
| 2.48x (measured, cin1_batch64_cout16) | 1.0754x | 1.1010x |
| 2.53x (measured, cin1_large_spatial) | 1.0765x | 1.1024x |

The fresh reconstruction (Section 6 of the brief's own methodology,
computed in `_end_to_end_backward_delta`/`_mnist_fraction_trials`) derives
a same-session "before" fraction (~15.4%) by combining this session's own
measured whole-step `conv2d backward` fraction (mean of 5 fresh trials,
47.0%, range 46.0%–47.7%) with a fresh dInput-of-conv2d-backward ratio
computed from `mnist_conv1`+`mnist_conv2`'s own isolated stage timings,
before and after M48. It is presented alongside — not instead of — M47's
own carried-over 11.75% figure, since production dispatch cannot be
reverted in-process to re-derive that exact figure identically; both
independently support the same conclusion (a genuine, non-negligible
whole-step improvement in the 6.8%–10.2% range depending on which
"before" fraction is used).

**This clears a real bar**: M43's own accepted dWeight `blocks_y==1`
optimization — the most recent accepted CUDA kernel change before this
one — delivered a comparable or smaller full-pipeline win (1.02x–1.04x at
`large_spatial`, 1.11x–1.24x at `mnist_conv2`) for comparable
implementation complexity (a template/dispatch change reusing existing
infrastructure, no new algorithmic primitive). M48's ~1.09x–1.10x
whole-step projection, backed by a *measured* (not just projected)
1.155x–1.158x `conv2d_backward()` speedup, is at least as strong a result.

## 9. Candidate optimization

**Selected**: Candidate B from the brief's Section 8 menu — an
input-channel-specialized kernel for the low-`Cin` case, removing
unnecessary generality from the M36 implementation. Implementation:
`k_conv2d_backward_input_channelfused<T>`'s body, unchanged, but with its
accumulator array's compile-time bound turned into a real template
parameter `CIN_MAX` instead of the fixed `MAX_CIN_REG=16`, instantiated at
`CIN_MAX=1` — the exact value `mnist_conv1` needs. No shared memory was
added (ruled out per the brief's Section 9 anti-pattern list — "add shared
memory merely because memory reuse sounds useful" — the root cause here is
instruction overhead, not memory reuse), and no cooperative-thread
restructuring was attempted (grid size was never the bottleneck — see
Section 7).

## 10. Candidate benchmark

Round-robin-interleaved, order-independent A/B (`_interleaved_multi_time`,
the M46-corrected methodology), 40 iterations + 8 warmup, at 4 `Cin=1`
shapes spanning batch size, spatial size, `Cout`, and stride — reproduced
across 2 independent full script invocations:

| shape | production (lowcin1, ms) | pre-M48 baseline (ms) | speedup |
|---|---:|---:|---:|
| mnist_conv1 (N=64,Cout=8,H=W=28,K=3,P=1) | 0.3811 | 0.8632 | **2.265x** |
| cin1_large_spatial (N=16,Cout=32,H=W=64,K=3,P=1) | 1.4272 | 3.6083 | **2.528x** |
| cin1_batch64_cout16 (N=64,Cout=16,H=W=28,K=3,P=1) | 0.5967 | 1.4792 | **2.479x** |
| cin1_strided (N=32,Cout=8,H=W=32,K=3,S=2,P=1) | 0.1313 | 0.2866 | **2.183x** |

Correctness: bit-exact (`max abs err = 0.0`) against the M36 channel-fused
kernel at multiple shapes (checked directly with `ctypes`, before writing
the pytest suite), and `rtol=1e-3` parity in the pytest suite across
stride/padding/kernel-size/dtype combinations.

## 11. Production decision

**ACCEPT.** `k_conv2d_backward_input_channelfused_lowcin<T, CIN_MAX>`
(`kernels.cu`), instantiated at `CIN_MAX=1`, is promoted to production.
`CONV2D_BACKWARD_INPUT_LAUNCHER`'s dispatch now reads:

```text
Cin <= CONV2D_DINPUT_LOWCIN_MAX_CIN (1)  -> k_conv2d_backward_input_channelfused_lowcin<T,1>  (NEW, M48)
Cin <= CONV2D_DINPUT_CHANNELFUSED_MAX_CIN (16) -> k_conv2d_backward_input_channelfused          (M36, unchanged)
Cin >  16                                       -> k_conv2d_backward_input                       (M32, unchanged fallback)
```

A simple `Cin<=1` boundary, matching the brief's Section 11 preference —
no shape-dependent heuristic table. The kernel is templated on `CIN_MAX`
so a future milestone could widen the band (e.g. `CIN_MAX=2`) if evidence
ever justified it, but this milestone measured and claims only `Cin=1`
(the one real Forge shape), per Section 12's scope discipline.

**Production scope, exactly as the brief limits it**: the new specialized
kernel, its minimal launcher, and the minimal dispatch condition. dWeight,
dBias, Conv2d forward, `k_matmul`, the M34/M37/M39/M43 dWeight paths, the
allocator, streams, prefetch, and public APIs are untouched — confirmed by
re-running the full test suite (Section 12) and the `Cin=2` regression
check (Section 14).

## 12. Correctness

29 new tests in `tests/test_cuda_conv2d_dinput_lowcin_candidate.py`:

- Direct kernel-level parity (lowcin1 vs. M36 channel-fused, and lowcin1
  vs. the original pre-M36 kernel) across 8 shape/stride/padding/
  kernel-size combinations, f32 and f64 (16 parametrized cases).
- Production dispatch correctness at and across the new `Cin<=1` boundary
  (`Cin=1` and `Cin=2`) through the real `nn.Conv2d`/autograd API, CPU vs.
  CUDA gradient parity.
- Production dispatch correctness at 4 additional real-shape-like
  configurations (varying `Cout`, batch, spatial size, stride, `K=1`).
- f64 parity through the real API.
- Explicit-stream and cross-stream (producer streams != compute stream)
  correctness.
- Repeated-use memory-lifecycle safety (`allocated_bytes` returns to
  baseline after 20 repeated calls) and allocator-reuse safety
  (`cache_hit_count` increases across repeated same-size allocations).
- An occupancy-regression guard (`lowcin1` occupancy `>=` production
  occupancy), so a future kernel-body change cannot silently invalidate
  this milestone's acceptance evidence without a test failing.

Full suite: **1,579 passed** (1,550 + 29), verified on the same clean CUDA
rebuild used for this milestone's own measurements.

## 13. Memory / resource validation

- No persistent allocation growth: `test_lowcin1_repeated_use_does_not_
  grow_active_memory` confirms `allocated_bytes` returns to its pre-loop
  baseline after 20 repeated kernel calls.
- Allocator reuse confirmed: `test_lowcin1_allocator_reuse_across_
  repeated_calls` confirms `cache_hit_count` increases across repeated
  same-size allocations (not a vacuous pass on a single fresh allocation).
- Register count: 48 (f32/f64 both), 0 spills, 0 stack frame — confirmed
  via fresh `nvcc -Xptxas -v` (Section 5).
- Occupancy: 5 blocks/SM, 62.5% — confirmed via real
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (Section 5), guarded by
  a dedicated regression test (Section 12).
- Shared memory: 0 bytes (kernel uses only registers and global memory,
  unchanged from the M36 kernel's own design).

## 14. Regression analysis

- **`Cin=2`** (still M36 channel-fused, byte-for-byte unchanged):
  isolated `d_input_ms` measured at 0.4545ms (`N=32,Cout=8,H=W=28,K=3,
  P=1`) — dispatch decision confirmed unchanged (`k_conv2d_backward_
  input_channelfused (M36, unchanged)`).
- **`mnist_conv2` (`Cin=16`)**: dispatch decision confirmed unchanged
  (still M36 channel-fused); fresh `Cin` sweep (Section 6) shows `Cin=16`
  at 1.9683ms / 28.0% of ceiling, consistent with M47's own range.
- **`Cin>16` fallback**: unchanged kernel, unchanged dispatch condition,
  reconfirmed via the fresh sweep (Section 6) — still unreached by any
  real Forge shape.
- **Full test suite**: 1,579 passed, 0 failed, 0 unexpected skips —
  every pre-existing CUDA conv test (including `tests/
  test_cuda_conv2d_dinput_optimization.py`'s own `Cin=1` coverage, which
  now exercises the new lowcin1 path instead of the old channel-fused
  path it previously reached) still passes unmodified.

No regression found anywhere this milestone measured.

## 15. Training-step impact

Directly measured, not merely projected: `conv2d_backward()` at
`mnist_conv1`'s exact shape is **1.155x–1.158x faster** (2 independent
runs), reconstructed by comparing the real, live `_profile_shape_full`
call (which now uses the new dispatch) against a same-session
reconstruction of the pre-M48 total (dWeight/dBias isolated timings are
identical either way; only dInput's isolated timing is swapped for the
forced pre-M48 baseline kernel).

Whole-training-step Amdahl projection (Section 8): **~1.07x–1.10x**,
depending on which "before" dInput fraction is used (M47's carried-over
11.75%, or this session's fresh 15.37% reconstruction) — both above the
measurement-noise floor this codebase has documented (M46/M47's own
within-call interleaving already cancels the larger cross-phase thermal
drift for *ratios*; only cross-session absolute-millisecond comparisons
are unsafe, and every number in this section is a same-session ratio or a
directly measured `conv2d_backward()` delta, not a cross-session absolute
comparison).

## 16. Limitations

- **Whole-step fraction estimates carry the same thermal-drift caveat
  M47 documented.** This session's own 5-trial `conv_bwd_frac_mean`
  ranged 46.0%–47.7% — narrower than M47's own 35.1%–45.3% range, but
  still real trial-to-trial variance on this hardware. The Amdahl
  projection in Section 8 should be read as "roughly 7–10% whole-step
  improvement," not a precise single figure.
- **The dispatch band is deliberately narrow (`Cin<=1`)**, matching the
  one real Forge shape measured. A wider band (e.g. `Cin<=2` or `<=4`)
  was not tested and is not claimed to help — a future milestone would
  need its own fresh measurement before widening it, per this codebase's
  own scope discipline.
- **No real CNN beyond MNIST exists in Forge today** to validate this
  optimization against a second real `Cin=1` workload — the relevance
  case (Section 4) rests on Forge's one example network remaining
  representative of near-term real usage.

## 17. Next engineering direction

With dInput's low-`Cin` regime now addressed, M47's remaining live
candidates (conv2d forward, dWeight `blocks_y==1`) were both already
found "not selected" in M47's own ranking (forward already restructured
in M41 with no new opportunity found; `blocks_y==1` just optimized in M43
and reconfirmed healthy). No Conv2d backward/forward sub-component
currently shows the kind of clear, fresh-evidence-backed opportunity this
milestone found for dInput. Per the brief's own Section 17 guidance
("identify the next highest-value Forge engineering area... may be
framework functionality, API completeness, usability, testing
infrastructure... rather than inventing another Conv2d optimization"), the
next milestone should perform a fresh, honest survey of non-Conv2d
subsystems (optimizer coverage, additional layer types, data-loading
ergonomics, serialization format completeness, or a second real example
network beyond MNIST to broaden Forge's own real-workload evidence base
per this milestone's Section 4/16 finding) before returning to CUDA
micro-optimization. This is a recommendation for the next milestone to
verify, not a decision this milestone makes on its own.

## 18. Conclusion

M48 valued M47's selected target before committing to it, found the
value case real (if narrow — one real workload), found the technical
root cause more nuanced than initially hypothesized (instruction
overhead, not primarily register pressure), and found a low-complexity,
narrowly-scoped candidate that delivered a large, reproducible,
bit-exact-correct kernel-level win (2.18x–2.53x) and a measured, not
merely projected, end-to-end `conv2d_backward()` win (1.155x–1.158x) with
zero regression anywhere tested. The candidate is **ACCEPTED** and is now
production, dispatched at the exact boundary the evidence supports
(`Cin<=1`), with full correctness/memory/regression coverage.

## Required final recommendation (Section 20 of the brief)

1. **Does low-Cin dInput matter enough?** Yes, narrowly — it is the
   dInput cost for Forge's only real trained network's first layer,
   ~11.75%–15.4% of that network's full training step.
2. **Is there meaningful technical headroom?** Yes — confirmed by
   profiling (roofline efficiency 4.7%→28.0%), not merely hypothesized.
3. **Is there a credible low-complexity optimization?** Yes — a
   templated accumulator-array bound, reusing the existing kernel body
   and dispatch pattern, no new algorithmic primitive.
4. **What is the realistic end-to-end payoff?** ~1.09x–1.10x
   whole-training-step (Amdahl, two independent fraction estimates),
   backed by a directly measured 1.155x–1.158x `conv2d_backward()`
   speedup.
5. **Should Forge spend another milestone on it?** No further milestone
   needed on *this* target — it is now resolved (ACCEPT, implemented).
6. **What should Forge work on instead (next milestone)?** A fresh survey
   of non-Conv2d subsystems (Section 17) — no further Conv2d backward/
   forward sub-component currently shows comparable fresh-evidence-backed
   opportunity.

## Reproducibility

```
python -m benchmarks.m48_dinput_value_assessment
```

Requires real CUDA hardware (skips cleanly with a printed message
otherwise). Full profile JSON: `benchmarks/results/
m48_dinput_value_assessment.json`.

## Tests / Verification

`tests/test_cuda_conv2d_dinput_lowcin_candidate.py` (29 new tests). Full
suite: **1,579 passed** (1,550 + 29), verified on a clean CUDA rebuild
(`_forge_cuda_kernels_sm_50.dll` deleted and recompiled) immediately
before running both the test suite and the benchmark script above.
End-to-end CPU/CUDA parity additionally spot-checked directly through the
real `Tensor`/`Conv2d` API across the full dispatch range (`Cin` in
{1, 2, 16, 20}) before writing the pytest suite.

## Why no ADR

A new dispatch band within an existing, already-documented hybrid-dispatch
pattern (Section 38's convention, extended identically by M36 itself when
it first introduced the channel-fused/fallback split) — not a new
architectural decision or public API change.

## Suggested Commit Message

`perf: add CUDA dInput low-Cin (Cin=1) specialized kernel, M48`
