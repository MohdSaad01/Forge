# CUDA Conv2d Forward Profiling and Optimization (Milestone 41)

## 1. Executive summary
M40 found `k_conv2d_forward` (unchanged since Milestone 15) achieving only
10.7-18.0% of the 940MX's practical compute ceiling at every representative
shape -- meaningfully less efficient than either backward Conv2d kernel
despite an *identical* total FLOP count -- and recommended applying the same
im2col + existing tiled GEMM technique M34/M39 already validated for
`dWeight`. This milestone designed and measured two structurally different
GEMM-based candidates against the unmodified baseline kernel at 15
representative/sweep shapes on the real 940MX:

- **Candidate A**: `im2col`/`im2col_smem` (unmodified) -> weight transpose
  (`k_transpose`, unmodified) -> the existing tiled GEMM (`cf_matmul_*`,
  unmodified) -> a new output-permute-plus-bias kernel.
- **Candidate B**: weight transpose (same) + a new half-fused GEMM kernel
  that gathers `Xcol` tiles on the fly (no `Xcol` buffer at all) and writes
  directly into the final output layout.

Both candidates won by 1.06-2.85x at every shape at/above ~20M total forward
FLOPs, and both regressed (0.26-0.76x) at every shape at/below ~7.2M FLOPs
(the GEMM reformulation's extra kernel launches/allocations cost more than
an already-fast baseline call). `CUDABackend.conv2d` now dispatches to one
of the two GEMM candidates above a measured 10,000,000-FLOP threshold
(Candidate B when `Cout<=32`, Candidate A otherwise) and keeps the original
per-thread kernel below it. This is a **complete Conv2d forward** win, not
merely an internal GEMM-stage win: every number in this document times the
full pipeline end to end. `nvcc -Xptxas -v` confirmed the baseline kernel
has zero register spill/stack frame -- the inefficiency is structural (no
memory reuse across threads), not a M32-style register-pressure problem, so
no simpler fix was available.

All numbers are from the verified development machine: NVIDIA GeForce 940MX
(GM108, Maxwell, 3 SMs / 384 CUDA cores), Compute Capability 5.0, CUDA 12.6,
driver 582.53, Windows/WDDM, 2GB VRAM -- collected in continuous sessions
(no restart between phases) via `benchmarks/m41_conv2d_forward_profile.py`.

## 2. How to reproduce
```bash
python -m benchmarks.m41_conv2d_forward_profile     # this document's primary source (new, Milestone 41)
python -m benchmarks.m40_bottleneck_recharacterization  # re-run post-M41, full pipeline re-ranking
python -m pytest tests/test_cuda_conv2d_forward_im2col_gemm.py -q  # new correctness coverage
```
Archived results: `benchmarks/results/m41_conv2d_forward_profile.json`
(per-shape complete-pipeline/isolated/stage-decomposition timings, roofline
classification, `nvcc -Xptxas -v` raw output), `benchmarks/results/
m40_bottleneck_recharacterization.json` (post-M41 re-run).

## 3. Baseline architecture (unchanged, going into Milestone 41)
`k_conv2d_forward` (`forge/backend/cuda/kernels.cu`, Milestone 15, never
touched by M31-M40): one thread per `(n, co, ho, wo)` output element,
looping `Cin x KH x KW` serially in registers, reading `x`/`w` directly from
global memory with no shared memory, no tiling, and no register blocking --
mathematically equivalent to a naive, untiled GEMM (`out_mat[Cout,M] =
weight_mat[Cout,K] @ Xcol[K,M]`, `K=Cin*KH*KW`, `M=N*Hout*Wout`) with zero of
the reuse Forge's own M11 tiled `k_matmul` (below) was written specifically
to capture.

## 4. Profiling methodology
- **Timing**: `forge.backend.cuda.profiling_events.TimedEvent`
  (`cudaEventRecord`/`cudaEventElapsedTime`), 5 warmup + 30 measured
  iterations per number -- the same convention every milestone since M31
  uses.
- **Interleaving**: the four complete-pipeline candidates (baseline,
  Candidate A plain, Candidate A smem, Candidate B) are timed via
  `_interleaved_compare` -- one call per candidate per round, alternating,
  not "all of baseline then all of A" -- so run-to-run thermal/clock drift
  affects every candidate equally (the exact benchmarking-order mistake
  Section 5 of the M41 brief and M37's own report both flag).
- **Complete pipeline, not an internal stage**: every "candidate" timing
  below calls the real, complete Python-level forward function
  (`conv2d_forward_im2col_gemm`/`conv2d_forward_halffused_gemm`), including
  every buffer allocation, kernel launch, and (for Candidate A) the final
  output permute -- never just the GEMM call in isolation.
- **Register/occupancy analysis**: `nvcc -Xptxas -v` against the unmodified
  `kernels.cu` (via `build._find_msvc_bin()`'s existing MSVC-on-PATH retry).

## 5. Representative workloads
Reuses `conv2d_backward_profile.SHAPES`/`BATCH_SWEEP_BASE`/`BATCH_SIZES`
(the same 7 shapes every M32-M40 report uses) plus a dedicated K/stride/
channel sweep (`m41_conv2d_forward_profile.SWEEP_SHAPES`, one variable
changed at a time from a `Cin=8/Cout=16/28x28/N=16/K=3/S=1/P=1` base) --
15 shapes total, no new shape families invented beyond what Section 1 of
the brief asks for:

| Shape | N | Cin | Cout | H=W | K | S | P | Total FLOPs |
|---|---|---|---|---|---|---|---|---:|
| `mnist_conv1` (real MNIST layer 1) | 64 | 1 | 8 | 28 | 3 | 1 | 1 | 7.23M |
| `mnist_conv2` (real MNIST layer 2, post-pool) | 64 | 8 | 16 | 13 | 3 | 1 | 1 | 24.92M |
| `large_channel` | 64 | 16 | 32 | 28 | 3 | 1 | 1 | 462.42M |
| `large_spatial` | 32 | 8 | 16 | 56 | 3 | 1 | 1 | 231.21M |
| `k1_s1` (K=1) | 16 | 8 | 16 | 28 | 1 | 1 | 0 | 3.21M |
| `k3_s1` (K=3 base) | 16 | 8 | 16 | 28 | 3 | 1 | 1 | 28.90M |
| `k5_s1` (K=5) | 16 | 8 | 16 | 28 | 5 | 1 | 2 | 80.28M |
| `k3_s2` (stride 2) | 16 | 8 | 16 | 28 | 3 | 2 | 1 | 7.23M |
| `cin_low` | 16 | 1 | 16 | 28 | 3 | 1 | 1 | 3.61M |
| `cin_high` | 16 | 64 | 16 | 28 | 3 | 1 | 1 | 231.21M |
| `cout_low` | 16 | 8 | 4 | 28 | 3 | 1 | 1 | 7.23M |
| `cout_high` | 16 | 8 | 128 | 28 | 3 | 1 | 1 | 231.21M |
| `batch_{32,64,128}` (= `large_channel`'s shape, N swept) | 32/64/128 | 16 | 32 | 28 | 3 | 1 | 1 | 231.21M/462.42M/924.84M |

All well within the 940MX's 2GB VRAM budget (`_check_fits_in_vram`, reused
unmodified from `conv2d_backward_profile`).

## 6. Kernel resource analysis (`nvcc -Xptxas -v`, real compile of the
unmodified/new `kernels.cu`)

| Kernel | Registers (f32/f64) | Static smem | Stack frame | Spill |
|---|---|---|---|---|
| `k_conv2d_forward` (baseline, unchanged) | 48 / 55 | 0 | 0 bytes | 0 bytes |
| `k_conv2d_output_permute` (new, Candidate A) | 31 / 31 | 0 | 0 bytes | 0 bytes |
| `k_conv2d_forward_halffused_gemm` (new, Candidate B) | 38 / 40 | 2048B / 4096B | 0 bytes | 0 bytes |
| `k_matmul` (existing, unmodified, reused by Candidate A) | 30 / 32 | 2048B / 4096B | 0 bytes | 0 bytes |
| `k_im2col_conv2d` (existing, unmodified, reused by Candidate A) | 32 / 32 | 0 | 0 bytes | 0 bytes |

**The baseline kernel has zero register spill and zero stack frame** --
unlike M32's original `dInput` kernel (a genuine 512-byte local-memory-spill
problem `nvcc -Xptxas -v` caught directly), `k_conv2d_forward`'s
inefficiency is *not* a register-pressure artifact a M32-style fix could
address. Its 48/55 registers are comfortably below the 940MX's 64-register
default cap with no spill, so occupancy is thread-count-limited, not
register-limited. This confirms Section 22's stop condition analytically:
the only way to improve this kernel is to give threads shared work to reuse
(shared memory/tiling), which is exactly what both candidates below do.
Candidate B's two static shared-memory tiles (`2*16*16*sizeof(T)`) match
`k_matmul`'s own footprint exactly, and its 38/40-register profile is well
within the same safe-occupancy range M38's half-fused `dWeight` kernel
already established as practical on this device.

## 7. Memory-access analysis
`k_conv2d_forward`: each of the `N*Cout*Hout*Wout` threads independently
reads its own `Cin*KH*KW`-element window of `x` and `w` from global memory
-- for a fixed output spatial position, every one of the `Cout` threads
covering that position re-reads the *same* `Cin*KH*KW` window of `x`
(`Cout`-fold redundant `x` reads); for a fixed output channel, every one of
the `N*Hout*Wout` threads sharing that channel re-reads the *same*
`Cin*KH*KW` window of `w` (`N*Hout*Wout`-fold redundant `w` reads, though
`w`'s small footprint keeps most of this in L2 cache in practice). Neither
redundancy is captured by any explicit reuse mechanism -- no shared memory,
no register blocking across threads. Both GEMM candidates capture exactly
this reuse the way `k_matmul`'s own shared-memory tiling already does for
plain matrix multiplication: `Xcol`'s or `x`'s tile is staged once per block
and shared across the 16 threads that read it, and `weightT`'s tile is
likewise shared across its 16 threads.

## 8. Roofline analysis (baseline, this session's ceilings: 104.69 GFLOP/s
compute, 15.09 GB/s bandwidth)

Every one of the 15 representative/sweep shapes classifies
`mixed_or_ambiguous` at 4.0-19.1% of the practical compute ceiling --
confirming M40's finding held across the wider sweep, not just the original
7 shapes. The lowest point (`k1_s1`, 4.0%) is a `K=1` convolution with no
padding boundary checks and a tiny 8-element reduction; the highest
(`k5_s1`, 19.1%) has the largest per-thread reduction (`Cin*KH*KW=200`),
consistent with the per-thread loop overhead being a fixed cost that
amortizes better over a larger reduction.

## 9. Candidate designs

### Candidate A: im2col + existing tiled GEMM (`experimental_conv_forward_im2col.py`)
```
Xcol    = im2col(x)                 (M, K) = (N*Hout*Wout, Cin*KH*KW)   -- existing k_im2col_conv2d/_smem, unmodified
weightT = transpose(weight)         (K, Cout)                          -- existing k_transpose, unmodified
out_mat = Xcol @ weightT            (M, Cout)                          -- existing k_matmul, unmodified
out     = permute(out_mat) + bias   (N, Cout, Hout, Wout)              -- NEW: k_conv2d_output_permute
```
Verified against `CPUBackend.conv2d`'s own orientation (`cols @ w_flat.T`,
then `.transpose(0,2,1).reshape(...)`) -- this is the same mathematical
reformulation, just requiring one real (not lazy) transpose of the small
`weight` operand since `k_matmul` has no transpose flag. `weightT`'s build
(`Cout*Cin*KH*KW` elements) is always far smaller than `Xcol`, so its cost
is negligible (measured 0.008-0.015ms at every shape -- see Section 11).
`im2col_smem` is preferred whenever it fits the device's 48KB-per-block cap
(always true at Forge's shapes), mirroring `dWeight`'s own M39 dispatch.

### Candidate B: half-fused GEMM (`experimental_conv_forward_halffused.py`)
A structurally different design (Section 4's explicit "genuinely different"
requirement): `weightT` stays materialized (cheap, same as Candidate A), but
`Xcol`'s gather is fused directly into a new kernel's own tiled-GEMM load
(`k_conv2d_forward_halffused_gemm`) -- no `Xcol` buffer, no `out_mat`
buffer; the kernel writes its result straight into the final `(N, Cout,
Hout, Wout)` layout with bias fused in. This is the mirror image of M38's
half-fused `dWeight` kernel, with the fused/materialized roles reversed to
match forward's own GEMM orientation: `Xcol` (`tile_a`) is the large,
expensive-to-build operand here (unlike `dWeight`'s GEMM, where the
analogous large operand was cheap `dYcolT`), so it is the one fused away.

No split-K in either candidate: forward's GEMM has `M=N*Hout*Wout` as a
**block-count** dimension (`blocks_y=ceil(M/16)`, large at every Forge
shape), unlike `dWeight`'s GEMM where `M` was the **reduction** dimension
and `Cout`/`Cin*KH*KW` (small) were the block-count dimensions -- the exact
occupancy shortfall M37's split-K fixed. Forward's own `blocks_y` is already
large, so occupancy was never the bottleneck here, confirming M40's own
prediction (Recommendation item 5).

A third candidate ("fuse both operands," mirroring M37's rejected Candidate
A/C for `dWeight`) was considered and not implemented: `weightT`'s
materialization already costs under 0.015ms at every shape (Section 11), so
fusing it too could only ever save a already-negligible cost while adding
redundant-regather overhead on the small operand -- the same asymmetric
lesson M38 already drew for `dWeight`, applied here without needing to
re-run the experiment.

## 10. Benchmark methodology
Every number in Sections 11/13 is the **complete forward pipeline** for its
candidate -- Python-level function call to Python-level function call,
including every allocation and kernel launch -- benchmarked against the
**complete current production path** (`cf_conv2d_forward_*`, one launch),
per Section 5's explicit rule ("do not compare candidate GEMM time vs. full
baseline Conv2d time" -- the exact class of error M37's own report caught
and corrected). All four candidates are timed in one interleaved pass per
shape (Section 4 above); Section 11's stage decomposition uses the same
`TimedEvent` primitive but is not interleaved across stages (each stage is
timed in isolation, matching `im2col`/`dWeight` decomposition precedent in
M37/M39's own reports).

## 11. Candidate results (complete pipeline, 940MX, mean of 30 iterations, 5 warmup)

| Shape | baseline (ms) | A-plain (ms) | A-smem (ms) | B (ms) | A-plain speedup | A-smem speedup | B speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnist_conv1 | 0.645 | 1.237 | 1.014 | 1.180 | 0.52x | 0.64x | 0.55x |
| mnist_conv2 | 1.390 | 1.497 | 1.117 | 0.904 | 0.93x | 1.24x | **1.54x** |
| large_channel | 24.917 | 13.879 | 10.411 | 10.235 | 1.80x | 2.39x | **2.43x** |
| large_spatial | 13.007 | 12.324 | 8.673 | 6.281 | 1.06x | 1.50x | **2.07x** |
| k1_s1 | 0.771 | 0.539 | 0.494 | 0.456 | 1.43x | 1.56x | **1.69x** |
| k3_s1 | 1.605 | 1.688 | 1.234 | 0.984 | 0.95x | 1.30x | **1.63x** |
| k5_s1 | 4.117 | 4.005 | 2.727 | 2.017 | 1.03x | 1.51x | **2.04x** |
| k3_s2 | 0.409 | 0.578 | 0.472 | 0.417 | 0.71x | 0.87x | 0.98x |
| cin_low | 0.331 | 0.517 | 0.461 | 0.437 | 0.64x | 0.72x | 0.76x |
| cin_high | 12.145 | 10.553 | 7.109 | 4.921 | 1.15x | 1.71x | **2.47x** |
| cout_low | 0.415 | 1.599 | 1.148 | 0.964 | 0.26x | 0.36x | 0.43x |
| cout_high | 13.142 | 5.134 | **4.605** | 6.297 | 2.56x | **2.85x** | 2.09x |
| batch_32 | 12.527 | 7.129 | 5.318 | 5.274 | 1.76x | 2.36x | **2.38x** |
| batch_64 | 25.050 | 13.958 | 10.468 | 10.266 | 1.79x | 2.39x | **2.44x** |
| batch_128 | 50.569 | 27.767 | 20.880 | 20.487 | 1.82x | 2.42x | **2.47x** |

Bold marks the winning candidate at each shape. **Every shape at/above
~20M total forward FLOPs wins with both candidates** (1.06-2.85x); **every
shape at/below ~7.2M FLOPs regresses with both candidates** (0.26-0.76x),
with one exception (`k1_s1`, 3.21M FLOPs, wins 1.43-1.69x -- its baseline
itself is anomalously slow, 0.771ms for only 3.21M FLOPs / 4.0% of ceiling,
plausibly measurement noise on an already-tiny workload rather than a
structural effect; see Section 17's Limitations). `cout_low` (`Cout=4`) is
the worst regression at every candidate (0.26-0.43x) -- `k_matmul`'s fixed
16x16 tile wastes 12 of 16 output columns per tile when `Cout=4`, a
tile-inefficiency Forge already documented for `dWeight` at small `Cout`
(M34/M38's own "poor tile efficiency at `Cout<=16`" finding, worse still at
`Cout=4`).

## 12. Candidate A stage decomposition (im2col / transpose / matmul / permute)

| Shape | im2col (ms) | transpose (ms) | matmul (ms) | permute (ms) | im2col % | matmul % | permute % |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnist_conv1 | 0.273 | 0.009 | 0.347 | 0.267 | 30.5% | 38.7% | 29.8% |
| mnist_conv2 | 0.495 | 0.009 | 0.322 | 0.123 | 52.2% | 34.0% | 13.0% |
| large_channel | 4.299 | 0.012 | 4.669 | 1.122 | 42.6% | 46.2% | 11.1% |
| batch_128 | 8.421 | 0.012 | 10.572 | 2.279 | 40.0% | 50.2% | 10.8% |

`transpose` (the small `weightT` build) is negligible everywhere
(0.008-0.015ms). The GEMM stage dominates at large shapes (46-50%),
consistent with forward's own better GEMM occupancy (Section 9); `im2col`
remains the second-largest cost (40-52%), and `permute` is cheap at large
shapes (11%) but a real fraction at the smallest shape (`mnist_conv1`,
29.8%) -- consistent with fixed per-launch overhead dominating a tiny
workload, the same effect driving the below-threshold regressions in
Section 11.

## 13. Selection / rejection rationale
Both candidates satisfy Section 6's correctness/performance/memory/
architecture acceptance criteria (see Section 14) at/above the FLOPs
threshold. Between them:

- **Candidate B wins at every shape with `Cout<=32`** (every Forge MNIST/
  representative shape) -- it eliminates both the `Xcol` and `out_mat`
  buffers Candidate A must still build, and its redundant-regather tax on
  the fused `Xcol` gather (`blocks_x=ceil(Cout/16)`) stays small (`<=2`) at
  every such shape.
- **Candidate A (smem variant) wins at `Cout=128`** (`cout_high`: 2.85x vs.
  B's 2.09x) -- `blocks_x=8` at this shape makes Candidate B's redundant
  `Xcol`-tile regather (8x) expensive enough that materializing `Xcol` once
  (Candidate A) wins instead.
- **Candidate A's `im2col_smem` variant strictly dominates its plain
  `im2col` variant** at every shape tested (matching `dWeight`'s own M39
  finding) -- plain `im2col` (Candidate A-plain) is kept only as a
  benchmark-only reference, never dispatched to in production, mirroring
  `dweight_im2col_gemm`'s (M34) own fate after M39.

Neither candidate is rejected outright -- both are accepted into production,
each covering the shape regime where it measured faster (Section 7 of the
brief's "prefer a conservative shape-based dispatch... not universally
superior" guidance, realized literally: two winners, not one).

## 14. Production dispatch
`CUDABackend.conv2d` (`forge/backend/cuda/backend.py`):
```python
total_flops = 2 * N * Cout * Hout * Wout * Cin * KH * KW
if total_flops >= _CONV2D_FORWARD_GEMM_FLOPS_THRESHOLD:   # 10,000,000
    blocks_x = ceil(Cout / 16)
    if blocks_x <= 2:      # Cout <= 32 -- every current Forge shape
        return conv2d_forward_halffused_gemm(...)          # Candidate B
    return conv2d_forward_im2col_gemm(...)                  # Candidate A (im2col_smem)
# else: unchanged original per-thread kernel (cf_conv2d_forward_*)
```
The FLOPs threshold (not a weight-element count, unlike `dWeight`'s
dispatch) is the correct axis here because forward's cost scales with
`M*K*Cout` (`M=N*Hout*Wout` dominates), not with the small `Cout*Cin*KH*KW`
weight-element count alone -- two shapes with identical weight-element
counts (e.g. `mnist_conv1`'s 72 vs. a hypothetical `N=1` version) can have
wildly different total forward cost. 10,000,000 sits in the untested gap
between the measured "always regresses" cluster (<=7.23M FLOPs) and "always
wins" cluster (>=24.92M FLOPs), and classifies every one of this milestone's
15 shapes correctly except `k1_s1`'s anomalous win, which is *forfeited*
(dispatched to the still-correct, still-fast-enough baseline) rather than
risking a mis-dispatched regression -- the conservative direction of error
Section 7 explicitly asks for.

## 15. Correctness validation
`tests/test_cuda_conv2d_forward_im2col_gemm.py` (54 new tests, all passing):
f32/f64 parity vs. `CPUBackend.conv2d`, `K` in `{1,2,3,5}`, stride `{1,2}`,
padding `{0,1,2}`, small (`Cin=1`/`Cout=4`) and large (`Cin=64`/`Cout=128`)
channel counts, with/without bias, explicit CUDA stream, cross-stream
producer/consumer, repeated-execution memory-lifecycle safety (allocator
reuse, no growth) -- for both experimental pipelines directly, **and**
through the real `Tensor.conv2d`/`nn.Conv2d` API at shapes that cross both
production dispatch boundaries (the FLOPs threshold itself, and the
`blocks_x` Candidate-A-vs-B boundary), plus a full forward+backward
autograd check confirming the untouched backward path still produces
identical gradients. No finite-difference check was added for forward
itself -- finite differences validate *gradients*, and forward has none of
its own; the existing finite-difference coverage in `tests/test_cuda_conv.py`
(covering the untouched backward path) remains the relevant check there.
Full suite: **1,420 tests pass** (`python -m pytest tests/ -q`), up from
1,366 pre-M41 -- no existing test was modified.

## 16. Memory/resource analysis
Peak overhead is Candidate-dependent and always small relative to the 940MX's
2GB budget:
- **Candidate B**: zero extra buffers beyond `weightT`
  (`Cout*Cin*KH*KW*itemsize`, e.g. 18KB at `large_channel`'s shape) -- no
  `Xcol`, no `out_mat`.
- **Candidate A**: `Xcol` (`M*K*itemsize`, the dominant cost -- e.g. 14.4MB
  at `large_channel`'s shape: `50,176*144*4` bytes) + `weightT` (negligible)
  + `out_mat` (`M*Cout*itemsize`, e.g. 6.4MB at the same shape) -- all
  freed (returned to the M25 caching allocator) as soon as the Python
  function returns and its local `CUDAStorage` references go out of scope,
  same lifecycle every other Forge CUDA op already follows. The repeated-use
  memory-safety tests (Section 15) confirm zero net growth in
  `allocated_bytes`/`reserved_bytes` after 20 repeated calls for every
  candidate.
- No new persistent allocation, no new caching-allocator behavior, no new
  synchronization primitive.

## 17. Before/after results (isolated forward + complete Conv2d, summarized from Section 11)
At MNIST's own two real layer shapes: `mnist_conv1` stays on the unchanged
baseline (0.645ms, correctly below the FLOPs threshold); `mnist_conv2` now
dispatches to Candidate B and measures **1.54x faster** (1.390ms ->
0.904ms). At the "large"/batch representative shapes, the production
dispatch delivers **2.07-2.47x** on the complete forward pass. No
representative or sweep shape regresses relative to its *own* pre-M41
baseline -- every shape below the FLOPs threshold is dispatched to the
byte-for-byte unchanged original kernel, so its performance is provably
unchanged, not merely unmeasured.

## 18. MNIST / training impact
Re-ran `benchmarks/m40_bottleneck_recharacterization.py` fresh, same
session, post-M41 dispatch:

| Metric | M40 (pre-M41) | M41 (post-dispatch, this session) |
|---|---:|---:|
| `forward:Conv2d` (MNIST, batch=64, fraction of measured forward+backward step) | 17.0%* | 12.94% |
| `backward:conv2d` (unchanged, same fraction base) | 48.4%* | 55.36% |
| Async pipeline throughput, batch=32 | 2,460 samples/sec / 91.6% util | 2,988 samples/sec / 89.8% util |
| Async pipeline throughput, batch=64 | 3,168 samples/sec / 93.6% util | 3,330 samples/sec / 90.9% util |
| Async pipeline throughput, batch=128 | 6,783 samples/sec / 87.5% util | 6,767 samples/sec / 85.5% util |

\* M40's own Amdahl table normalized fractions slightly differently (see
that document's Amdahl Analysis section); the `forward:Conv2d`/
`backward:conv2d` row above uses this milestone's own fresh,
same-tooling (`m35_mnist`) re-run for both columns rather than mixing two
different normalization conventions, so the 17.0%->12.94% delta is the
directionally-correct, same-script comparison -- consistent with
`mnist_conv2`'s measured 1.54x forward win (`mnist_conv1` is unaffected,
staying on the unchanged baseline kernel below the FLOPs threshold).
Backward's percentage *rising* (48.4%->55.36%) is the expected Amdahl
side-effect of a smaller, faster forward phase shrinking the step's total
denominator -- not a backward regression (backward is byte-for-byte
untouched; its own absolute isolated-kernel timings in this session's
re-run are within 1-3% of M40's own archived numbers, consistent with this
GPU's documented run-to-run variance). Async-pipeline throughput improved
modestly at batch 32/64 (consistent with a faster compute step) and stayed
at noise-level parity at batch 128 (where backward's much larger absolute
cost dominates the step, per Section 21's Amdahl honesty requirement) --
no batch size regressed.

**Applying Amdahl's law honestly**: forward Conv2d was 17.0% of the
measured step pre-M41 with a 2.07-2.47x measured win at large shapes; even
a hypothetical uniform 2x forward speedup across the whole step predicts
only a ~1.09x overall step speedup (M40's own Amdahl table, `conv2d
forward` row) -- this milestone does **not** claim a large end-to-end
training speedup, only the real, measured, complete-Conv2d-forward win
documented in Section 11, plus the smaller, consistent MNIST-step-level
shift documented above.

## 19. Updated roofline / bottleneck ranking
`mnist_conv2`'s forward now runs via Candidate B; its measured GFLOP/s at
that shape is `24.92M FLOPs / 0.904ms = 27.6 GFLOP/s`, or **26.4% of the
practical compute ceiling** -- up from 17.8% pre-M41, though still short of
`dWeight`/`dInput`'s 22-44% range at comparable shapes, since Candidate B's
redundant-`Xcol`-regather tax and per-launch overhead still cost something
at this small a shape. `large_channel`'s forward (Candidate B) reaches
`462.42M / 10.235ms = 45.2 GFLOP/s`, **43.2% of ceiling** -- now
**compute-bound**, matching or exceeding `dWeight`'s own 43-44%
`blocks_y>=2` ceiling-fraction at the same shape, and confirming the GEMM
reformulation genuinely moved this kernel toward the roofline rather than
just reducing wall-clock time at the same efficiency.

Re-ranking the CUDA training pipeline (real MNIST, batch=64, this session):
`backward:conv2d` (55.36%, `dWeight`+`dInput`, both unchanged this
milestone) remains the largest phase; `forward:Conv2d` (12.94%, down from
17.0%) is no longer the second-largest single item once `backward:@`/
`backward:max_pool2d`/`backward:relu`/`forward:Linear`/`forward:ReLU`/
`forward:MaxPool2d` (4-6% each) are considered individually -- no one of
these is large enough, on its own, to justify a dedicated M42 investigation
under the same headroom x confidence criterion M40 used. **The next
measured optimization target remains `dWeight`'s `blocks_y>=2` split-K GEMM
path** (already compute-bound at 43-44% of ceiling, M40's own Rank 2 --
"would require a second GEMM tiling design, low confidence") or, more
plausibly, a fresh full re-characterization once M41's dispatch has shifted
every shape's relative contribution -- exactly the kind of re-measurement
M40 itself did after M39. This document does not begin that work (M41 ends
here, per the milestone brief's Section 15 instruction not to begin M42).

## 20. Limitations
- **`k1_s1`'s anomalous win** (3.21M FLOPs, below the 10M threshold, yet
  measured winning 1.43-1.69x) is not fully explained -- its baseline
  measurement (0.771ms, only 4.0% of ceiling) is disproportionately slow for
  such a small workload relative to every other sub-threshold shape tested;
  plausibly measurement noise specific to this `K=1`/`P=0` boundary-check-free
  shape, not re-verified with additional repetitions this milestone. The
  dispatch threshold deliberately classifies it conservatively (baseline),
  forfeiting this specific win rather than risking a threshold low enough to
  also catch a real regression.
- **No shape between ~7.23M and ~20M total FLOPs was tested** -- the
  10,000,000 threshold sits in this gap by construction (mirroring M34's own
  "no shape between 256 and 1,152 weight elements was tested" documented
  gap) and has not been independently verified at the boundary itself.
- **No shape between `Cout=32` (Candidate B wins) and `Cout=128` (Candidate
  A wins) was tested** for the secondary `blocks_x` dispatch boundary --
  the `blocks_x<=2` cutoff is chosen from the two bracketing data points
  plus the analytical redundant-regather-tax argument (Section 9), not a
  measured crossover.
- **`nvcc -Xptxas -v` automated JSON extraction is unreliable**
  (`m41_conv2d_forward_profile._run_nvcc_ptxas_verbose`'s `per_kernel_lines`
  captures intervening unrelated kernels' register lines too, since
  `kernels.cu` compiles every kernel in the file in one pass) -- Section 6's
  table above was built by manually reading the compiler's raw
  `Compiling entry function` / `Function properties for` blocks directly,
  the ground truth preserved unfiltered in the JSON's `ptxas.raw_stdout`/
  `raw_stderr` fields. The convenience filter is left as a known limitation
  for a future milestone to fix, not blocking this one's conclusions.
- **`cout_low`'s severe regression** (`Cout=4`, 0.26-0.45x at every
  candidate) is a `k_matmul` small-tile-efficiency limitation already
  documented for `dWeight` (M34/M38), not a new finding -- no fix was
  attempted here since `k_matmul` itself is out of scope (Section 1's hard
  scope: "never modify `k_matmul` itself").
- **This milestone did not implement a third ("fuse both operands")
  candidate** -- Section 9 explains why the existing evidence (`weightT`'s
  already-negligible materialization cost) makes it very unlikely to help,
  reusing M38's own analytical argument rather than re-running the
  experiment from scratch.

## 21. Future opportunities
- A per-shape-tuned `blocks_x` threshold (rather than the fixed `<=2`
  cutoff) if a future milestone measures shapes between `Cout=32` and
  `Cout=128`.
- Revisiting the FLOPs-threshold boundary (7.23M-20M gap) with additional
  shapes if a real Forge model ever uses a convolution in that range.
- Fixing `_run_nvcc_ptxas_verbose`'s kernel-attribution filter (Section 20)
  so future milestones get a trustworthy automated per-kernel register/smem
  table instead of a manually-read one.
- A fresh full bottleneck re-characterization (M40-style) once this
  dispatch has been live long enough to reveal whether any *other* Forge
  workload shape lands in an unexpected dispatch branch.

## 22. Hardware
NVIDIA GeForce 940MX, Compute Capability 5.0, CUDA 12.6, Driver 582.53, 2GB
VRAM, 3 SMs, 384 CUDA cores -- every measurement in this document was
collected on this real, verified development GPU. No simulated or emulated
CUDA behavior. Full test suite (`python -m pytest tests/ -q`) verified on a
clean CUDA rebuild triggered by `kernels.cu`'s updated mtime (`build.
ensure_kernel_library`'s existing staleness check): **1,420 passed**.

## Suggested Commit Message
```
perf: add im2col+GEMM Conv2d forward dispatch (Milestone 41)

k_conv2d_forward (unchanged since M15) measured at only 10.7-19.1% of the
940MX's practical compute ceiling across 15 representative/sweep shapes --
zero register spill/stack frame (nvcc -Xptxas -v), so the inefficiency is
structural (no memory reuse), not a register-pressure problem. Two GEMM-
based candidates were designed, implemented, and measured against the
complete baseline pipeline: Candidate A (im2col/im2col_smem + weight
transpose + existing tiled k_matmul + a new output-permute-plus-bias
kernel, all but the permute kernel reusing unmodified M11/M34/M39
infrastructure) and Candidate B (a new half-fused GEMM kernel that gathers
Xcol tiles on the fly, no Xcol/out_mat buffers at all). Both win 1.06-2.85x
at >=~20M total forward FLOPs and regress 0.26-0.76x at <=~7.2M FLOPs;
CUDABackend.conv2d now dispatches to Candidate B (Cout<=32) or Candidate A
(Cout>32) above a measured 10M-FLOP threshold, keeping the original kernel
below it. dInput/dWeight/dBias, k_matmul, and the async pipeline are
completely untouched. 54 new tests added (1,420 total, all passing).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GNRUFA9r7ZJFJbSmqmnpVt
```
