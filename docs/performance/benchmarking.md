# Forge Benchmarking (Milestone 11)

## Purpose
Establish a reproducible performance-measurement system for Forge, use it
to identify real bottlenecks from measurements, and apply only the
optimizations those measurements justify. Per `docs/development/roadmap.md`'s
Milestone 11 brief: **do not make performance claims without measurements**,
and do not replace a kernel merely because an optimized library exists.

## Package layout
```text
benchmarks/
    __init__.py       package docstring; deliberately never imported by `forge`
    timing.py           Timing, time_cpu, time_cuda, time_calls -- the measurement primitives
    sizes.py             tiny/small/medium size configurations, warmup/iteration defaults
    environment.py       hardware/software environment capture
    results.py            BenchmarkResult, JSON output, human-readable table
    ops_bench.py           forward Tensor/layer/loss/optimizer op benchmarks
    backward_bench.py       backward-pass benchmarks (incl. a full M20 CNN backward)
    transfer_bench.py       CPU<->CUDA transfer cost benchmarks
    training_bench.py       end-to-end toy-model training-throughput benchmark
    mnist_bench.py            end-to-end real M20 CNN training-throughput benchmark (Milestone 21)
    mnist_profile.py           MNIST workload phase/per-op profiling script (Milestone 21; not a benchmark category)
    memory.py                   cuda_memory_extra() -- CUDA memory-stats reporting for BenchmarkResult.extra (Milestone 22)
    run.py                    CLI entry point
    __main__.py                `python -m benchmarks`
    results/latest.json         most recent full run's structured output (not test data)
    results/m11_baseline.json    archived Milestone 11 baseline
    results/m21_baseline.json    archived Milestone 21 pre-optimization baseline
    results/mnist_profile_baseline.json    archived Milestone 21 pre-optimization MNIST profile
    results/mnist_profile_optimized.json    Milestone 21 post-optimization MNIST profile
```
`benchmarks/` lives outside the `forge` package and is never imported by it
-- `import forge` never touches this code, matching the milestone brief's
"benchmarks must not be required for package import or normal usage."
`tests/test_benchmarks.py` is the one place the normal `pytest` suite
touches this package, and it only exercises the harness's own mechanics
(iteration counts, aggregation math, JSON round-tripping) with trivial,
near-instant callables -- it never runs a real benchmark workload and never
asserts a timing threshold, per the brief's "do NOT put timing thresholds
into normal unit tests."

## Running the suite
```bash
python -m benchmarks                              # everything, prints a table, writes benchmarks/results/latest.json
python -m benchmarks --categories forward transfer  # a subset
python -m benchmarks --output my_run.json           # a different output path
python -m benchmarks --categories mnist              # real M20 CNN training throughput only (Milestone 21)
python -m benchmarks.mnist_profile                    # MNIST workload phase/per-op breakdown (Milestone 21; not a category)
```
This is always an explicit, separate action from running `pytest` -- the
correctness suite and the benchmark suite are different commands with
different purposes (see `docs/testing/strategy.md`).

## Timing methodology

### CPU
`time.perf_counter()` -- Python's monotonic, highest-resolution wall-clock
timer, immune to system clock adjustments (`benchmarks/timing.py::time_cpu`).

### CUDA
CUDA kernel launches are asynchronous: they return to Python as soon as the
launch is *queued*, not once the kernel has actually finished. Naively
wrapping `start = perf_counter(); cuda_op(); end = perf_counter()` measures
launch overhead only. Every CUDA measurement in this suite instead follows:

```text
synchronize()            # drain any prior in-flight work
start = perf_counter()
cuda_op()                 # launches, returns immediately
synchronize()              # block until the launched work actually finishes
end = perf_counter()
```

`synchronize()` is `CUDABackend.synchronize()` (`forge/backend/cuda/backend.py`,
added in this milestone), a thin public wrapper around the same
`cudaDeviceSynchronize()` every CUDA operation already calls internally
before trusting its own result (see `docs/architecture/cuda-backend.md`).
No new native/CUDA-event code was added -- this reuses an existing,
already-verified synchronization point rather than introducing a second
timing mechanism. `benchmarks/timing.py::time_cuda` implements exactly this
pattern for every CUDA measurement in the suite.

### Warmup
Every measured callable runs `warmup` times, uncounted, before any timing
starts. This absorbs first-call effects that would otherwise inflate early
measured iterations and misrepresent steady-state performance:
- CUDA context / lazy kernel-module initialization on first use.
- The `nvcc` kernel-library compile-cache check (`forge/backend/cuda/build.py`).
- CPU/GPU cache warming for the operand data itself.

Default: `warmup=5`, `iterations=20` (`benchmarks/sizes.py`) for every
forward/backward/transfer benchmark; the training benchmark uses
`warmup_iterations=5`, `iterations=50` (more iterations, since a training
step is comparatively cheap and a larger sample reduces noise in the
forward/backward/optimizer sub-splits).

### Backward-pass timing: `time_calls`
A Tensor's non-leaf `grad_fn` is freed the moment `backward()` consumes it
(`docs/architecture/autograd.md`'s "Graph freed on use" -- a second
`backward()` call on the same non-leaf output raises `GradientStateError`),
so a backward benchmark cannot simply call `y.backward()` in a loop the way
a forward benchmark repeats `a + b`. Instead, `benchmarks/backward_bench.py`
pre-builds `warmup + iterations` independent fresh forward passes *before*
timing starts (so forward-pass cost is never folded into the backward
measurement), and `benchmarks/timing.py::time_calls` times each iteration's
single `backward()` call individually, applying the same CUDA
synchronization bracketing as `time_cuda` when `device="cuda"`.

### Aggregation
Every result reports mean, median, min, max, and sample standard deviation
(`statistics.stdev`) over the `iterations` measured samples -- never a
single cherry-picked timing (`benchmarks/results.py::BenchmarkResult`,
`benchmarks/timing.py::Timing`). The human-readable table
(`render_table`) shows mean and stdev in milliseconds; the JSON output
(`save_json`) carries the full set of aggregates plus every category's
`extra` fields (e.g. transfer throughput, training samples/sec).

## Benchmark categories

### 1. Forward Tensor operations (`ops_bench.py`)
`add`, `sub`, `mul`, `relu`, `sum`, `reshape` at three element-count scales,
and `matmul` at three square-matrix scales -- exactly the operation set
both backends share (`docs/architecture/cuda-backend.md`'s Operation set
table). CUDA-unsupported ops (`exp`/`log`) are not benchmarked on CUDA, per
the brief's "do not benchmark unsupported CUDA operations." Measured at the
`Tensor` API call site (`a + b`, `a.relu()`), so results include whatever
`Tensor`/`Backend` dispatch overhead sits on top of the raw kernel -- the
same overhead real model code pays.

### 2. Backward operations (`backward_bench.py`)
Elementwise (`add`) backward, `relu` backward, `matmul` backward, `sum`
backward, and a representative multi-layer backward pass
(`Linear -> ReLU -> Linear`, batch of 64), CPU and CUDA where both exist.

### 3. Transfer costs (`transfer_bench.py`)
`Tensor.to("cuda")` (H2D) and `Tensor.to("cpu")` (D2H) at three byte sizes
(~4 KB / ~400 KB / ~4 MB, float32). Reports throughput in GB/s
(`bytes / mean_seconds`) alongside raw timing. Transfer time is always its
own measurement, never folded into a kernel-timing number elsewhere in this
suite, per the brief's "do not hide transfer time inside kernel execution
time." Produces no results when CUDA is unavailable.

### 4. End-to-end training (`training_bench.py`)
A `Linear(64,128) -> ReLU -> Linear(128,10)` model, batch size 64, trained
for 50 measured iterations (5 warmup) against one fixed synthetic batch --
never asserts or depends on convergence, per the brief. Reports total time,
mean time per step, batches/sec, samples/sec, and the mean per-phase time
(`zero_grad+forward`, `backward`, `optimizer.step`) via internal
synchronization checkpoints. `forge.training.Trainer` has no CUDA
integration (`docs/architecture/cuda-backend.md`'s documented limitation),
so both CPU and CUDA runs use the same hand-written
`zero_grad -> forward -> loss -> backward -> step` loop (matching
`docs/architecture/training-engine.md`'s documented lifecycle) rather than
comparing `Trainer` overhead against a hand-written loop.

## Benchmark sizes
Chosen to stay well inside the 940MX's 2 GB VRAM and the development
machine's 8 GB system RAM (`docs/development/development-environment.md`),
and to match the milestone brief's own example dimensions:

| Scale | matmul dims | elementwise elements | transfer elements (float32) |
|---|---|---|---|
| tiny | 32x32 | 1,024 | 1,024 (~4 KB) |
| small | 128x128 | 16,384 | 100,000 (~400 KB) |
| medium | 512x512 | 262,144 | 1,000,000 (~4 MB) |

The largest single allocation in this suite is a 512x512 float32 matrix
(1 MiB); the training benchmark's largest tensor is a `(64, 128)` hidden
activation. Nothing here approaches the 2 GB VRAM budget.

## Environment (this baseline)
Captured automatically by every run (`benchmarks/environment.py`) and
recorded in `benchmarks/results/latest.json`'s `"environment"` key:

- **CPU**: Intel Core i5-7200U (`docs/development/development-environment.md`)
- **RAM**: 8 GB
- **GPU**: NVIDIA GeForce 940MX, 2 GB VRAM, Compute Capability 5.0
- **Driver**: 582.53
- **CUDA Toolkit**: 12.6 (`nvcc` 12.6.85)
- **OS**: Windows 10
- **Python**: 3.13.5
- **NumPy**: 2.3.2
- **Forge commit**: `40cece1` (the commit this milestone's work built on)

No claim in this document generalizes beyond this exact environment, per
`docs/testing/acceptance-criteria.md`'s evidence rule.

## Baseline results (representative)
Full results (76 measurements) are in `benchmarks/results/latest.json`;
selected numbers below illustrate the overall shape. All times are the mean
of 20 measured iterations (50 for training) after 5 (training: 5) warmup
iterations; `Std` is one sample standard deviation.

### Forward ops, medium scale (262,144 elements / 512x512 matmul)
| Operation | CPU mean | CUDA mean |
|---|---|---|
| add | ~1.0-1.3 ms | ~1.4-2.3 ms |
| relu | ~0.8-0.9 ms | ~1.2-1.4 ms |
| sum | ~0.35-0.48 ms | ~1.5-2.6 ms |
| matmul (post-optimization) | ~4.5-5.3 ms | ~9.0-9.2 ms |

### Transfer (float32)
| Scale | Bytes | H2D mean | D2H mean | Throughput (medium) |
|---|---|---|---|---|
| tiny | ~4 KB | ~0.09-0.25 ms | ~0.08-0.32 ms | -- |
| small | ~400 KB | ~0.46-2.1 ms | ~0.51-0.76 ms | -- |
| medium | ~4 MB | ~6.6-8.7 ms | ~6.2-7.1 ms | ~0.5-0.6 GB/s |

Small/tiny transfers are dominated by fixed per-call overhead (`cudaMalloc`
on the destination side, driver call overhead), not achievable PCIe
bandwidth -- throughput only becomes a meaningful number at the medium
scale, and even there it is far below theoretical PCIe bandwidth, which is
expected for single, unpipelined transfers of this size on this hardware.

### End-to-end training (`Linear(64,128) -> ReLU -> Linear(128,10)`, batch=64)
| Device | Mean time/step |
|---|---|
| CPU | ~0.9-2.8 ms |
| CUDA | ~5.8-6.7 ms |

**CUDA is slower than CPU for this model on this hardware.** This is an
expected, measured result, not a defect: at this size, host<->device
transfer setup, kernel-launch overhead, and the 940MX's age dominate over
the (very small) amount of actual compute per step -- exactly the effect
`docs/architecture/cuda-backend.md`'s "No performance claims" section
anticipated. Forge does not claim "GPU is faster" here; the measurement
says the opposite for this workload, and that is reported as-is.

## Bottleneck identification
Comparing forward `matmul` across scales made the shape of the problem
clear:

| Scale | CPU (NumPy/BLAS) | CUDA (naive kernel, pre-optimization) | CUDA/CPU ratio |
|---|---|---|---|
| tiny (32x32) | ~0.017 ms | ~0.160 ms | ~9.4x slower |
| small (128x128) | ~0.254 ms | ~0.824 ms | ~3.2x slower |
| medium (512x512) | ~3.67 ms | ~16.00 ms | ~4.4x slower |

At tiny/small scale, CUDA being slower than CPU is exactly the expected
launch/transfer-overhead effect the milestone brief calls out -- not
evidence of a kernel-quality problem. But the ratio *growing* again at
medium scale (512x512), where the compute itself should dominate launch
overhead, does not fit that explanation: it points at the matmul kernel's
own algorithm. Milestone 8's kernel (`k_matmul` in `kernels.cu`, unchanged
through Milestones 9-10) was a naive one-thread-per-output-element
implementation: every thread re-reads a full `K`-length row of `A` and
column of `B` directly from global memory, with zero reuse across the 256
threads in a block that all read overlapping data. That access pattern gets
proportionally worse as `K` grows, exactly matching the medium-scale
slowdown observed.

Backward `matmul` (which composes the same forward `matmul` kernel via
`grad_output @ b.T` / `a.T @ grad_output`, see `docs/architecture/cuda-backend.md`)
showed the same pattern even more sharply at medium scale: ~10.3 ms (CPU)
vs. ~34.7 ms (CUDA, naive kernel) -- a ~3.4x slowdown.

This is the "naive CUDA matmul" candidate the milestone brief explicitly
lists, backed by measurement rather than assumption (the brief warns "do
not assume matmul is automatically the bottleneck" -- here it demonstrably
was, but only at the scale where compute dominates launch overhead, which
only measurement could show).

## Optimization decision: tiled shared-memory matmul
**Chosen: a shared-memory-tiled matmul kernel, not cuBLAS.**

The brief permits considering cuBLAS if matmul is a measured bottleneck,
but is explicit that this must not be automatic and must be justified. Given:
- The measured bottleneck is specifically the naive kernel's poor memory
  reuse, a problem a standard tiling optimization directly addresses.
- The brief explicitly sanctions "a tiled implementation... if the naive
  implementation is clearly measured as a bottleneck" for exactly this case.
- cuBLAS would add a new external binary dependency Forge has avoided since
  ADR-004 (`docs/architecture/decisions/ADR-004-cuda-execution-strategy.md`)
  chose hand-written `nvcc`-compiled kernels specifically to avoid
  depending on another numerical library, and the brief warns against
  "making the entire backend dependent on an unnecessarily large
  abstraction."
- The workloads this milestone's hardware constraints target (940MX, 2 GB
  VRAM, matrices up to a few hundred elements per side) do not need a
  general-purpose GEMM library's breadth -- one well-chosen tiling
  optimization is proportionate.

A tiled kernel was therefore implemented and cuBLAS was not introduced.

### Implementation
`k_matmul` in `forge/backend/cuda/kernels.cu` is now a standard 16x16-tile
shared-memory GEMM: each thread block cooperatively loads one `TILE x TILE`
tile of `A` and one of `B` into `__shared__` memory per outer-loop step, so
each element is read from global memory once per tile (reused by all 16
threads that need it) instead of once per output element. Out-of-range
tile loads (when `M`/`K`/`N` is not a multiple of 16) are zero-padded,
preserving the naive kernel's exact boundary semantics. No warp-level
intrinsics or architecture-specific features are used, so this remains
correct on Compute Capability 5.0. The exported symbol names
(`cf_matmul_f32`/`cf_matmul_f64`) and their signatures are unchanged --
`CUDABackend.matmul` (`forge/backend/cuda/backend.py`) required **no
Python-side changes**, since this is purely an internal kernel-algorithm
improvement behind the existing `Backend.matmul` boundary.

This is not a generalized GEMM library: it is one tiling strategy, sized
for the matrices this milestone's hardware/workload targets, matching the
brief's "do not implement a generalized GEMM library" and "do not introduce
complex shared-memory/tiled implementations unless measurements justify
them" constraints -- both satisfied here, the second by the bottleneck
measurement above.

### Correctness preserved
`matmul_backward` (`CUDABackend.matmul_backward`) composes the same
forward `matmul` kernel (plus `cf_transpose`, unchanged) for all four 1D/2D
cases, so the tiling change is exercised by both forward and backward
tests with no separate backward-kernel change needed. All 121 CUDA tests
(`tests/test_cuda_backend.py`, `tests/test_cuda_consistency.py`,
`tests/test_cuda_autograd.py`, `tests/test_module_cuda.py`) pass unchanged
after the kernel rewrite, including CPU/CUDA numerical-consistency checks,
finite-difference gradient checks, and the full 20-epoch CUDA training-loop
test -- see **Verification** in the milestone report for the exact command
and result.

### Before / after (measured)
Mean of 20 iterations, 5 warmup, on the verified 940MX; forward and
backward both measured via `python -m benchmarks --categories forward
backward`.

| Operation | Scale | Before (naive) | After (tiled) | Speedup |
|---|---|---|---|---|
| forward matmul | tiny (32x32) | 0.160 ms | 0.171-0.233 ms | ~no change (launch-overhead-bound) |
| forward matmul | small (128x128) | 0.824 ms | 0.247-0.270 ms | ~3.1-3.3x |
| forward matmul | medium (512x512) | 16.00 ms | 9.03-9.25 ms | ~1.7-1.8x |
| backward matmul | tiny (32x32) | 1.34 ms | 0.38-1.06 ms | mixed (noise-dominated at this size) |
| backward matmul | small (128x128) | 1.07 ms | 1.37-1.61 ms | ~no change (launch-overhead-bound) |
| backward matmul | medium (512x512) | 34.74 ms | 19.61-21.63 ms | ~1.6-1.8x |

At the medium scale -- the scale where the bottleneck was actually
identified -- the tiled kernel delivers a consistent, real ~1.7-1.8x
speedup on both forward and backward matmul. At tiny scale, the result is
noisy and shows no reliable improvement, because per-call overhead
(kernel launch, `cudaMalloc` for the output buffer) dominates total time
far more than the ~2x2-tile compute itself -- consistent with the
bottleneck analysis above, which specifically identified the *medium*
scale as compute-dominated. **CUDA matmul remains slower than CPU at
medium scale even after this optimization** (~9 ms vs. ~4.5-5.3 ms): NumPy
dispatches to a mature, highly optimized BLAS implementation, and a single
hand-tiled kernel on an eight-year-old, 2 GB laptop GPU is not expected to
close that gap. This document reports that honestly rather than claiming
otherwise -- the optimization's justified claim is "measurably faster than
Forge's own prior CUDA kernel," not "faster than CPU" or "fast."

### Why no ADR
This change does not alter the `Backend`/`CUDABackend` abstraction
boundary, the `Tensor -> Backend` dispatch pattern, or any public API --
`k_matmul`'s internals changed, `cf_matmul_{f32,f64}`'s signature and every
caller did not. Matching the precedent set by Milestones 9-10's own kernel
additions (the `relu` kernel, the row-broadcast kernels, all seven backward
kernels), which did not warrant ADRs for the same reason, this is
documented here and in `docs/architecture/cuda-backend.md`'s kernel table
rather than as a separate architecture decision record.

## Known limitations
- **Small workloads may still be slower on CUDA than CPU.** This is
  measured and expected on this hardware (see **Baseline results** and
  **Optimization decision** above), not a defect Forge attempts to hide.
- **Tiling is fixed at 16x16.** No autotuning across matrix sizes, no
  double-buffering/prefetch, no `float4`-vectorized loads -- a single,
  measured-appropriate choice for this milestone's hardware and workloads,
  not a generalized, tunable GEMM implementation.
- **Only matmul was retuned.** Other operations (`add`/`sub`/`mul`/`relu`/
  `sum`/`reshape`, transfer) were measured but not the target of this
  milestone's optimization -- their current CUDA implementations are the
  ones from Milestones 8-10, unchanged. A separate CPU-side inefficiency
  was also observed (leaf gradient accumulation on CPU does an eager
  `np.array(...)` copy that CUDA's leaf path does not need -- see
  `Tensor._accumulate_grad`, `forge/tensor/tensor.py`), but was not the
  measured bottleneck this milestone targeted and was left unchanged,
  per the brief's "apply only targeted optimizations justified by
  measurements" (one bottleneck, not every inefficiency the benchmarks
  happened to surface).
- **No CUDA event-based timing.** Synchronization-bracketed
  `perf_counter()` (see **CUDA** above) was used instead of
  `cudaEventRecord`/`cudaEventElapsedTime`, since it needed no new native
  code and is an accepted approach per the milestone brief ("a valid
  conceptual measurement is start -> op -> cudaDeviceSynchronize -> end").
  It cannot isolate a single kernel's device-side time from Python-level
  dispatch overhead the way CUDA events can; every number in this document
  is therefore "wall-clock time for the Python call, including CUDA
  execution," not "GPU-side kernel time" in isolation.
- **Single run per number reported in this document's summary tables**,
  though every individual benchmark result is itself an aggregate of 20 (or
  50, for training) iterations with reported stdev; multiple full
  `python -m benchmarks` runs were executed during development (values
  above show the observed range across those runs) to confirm the
  medium-scale matmul speedup was consistent and not a one-off, per the
  brief's "run the benchmark suite multiple times where practical."
- **No CI/regression-tracking integration.** Nothing in this milestone
  automatically compares a new run against `results/latest.json` or fails a
  build on regression -- explicitly out of scope (`docs/development/roadmap.md`
  Phase 5 is "Performance," not "Performance CI").

## Milestone 21: expanded coverage, MNIST profiling, targeted CUDA optimization

### New benchmark coverage
Milestone 11 covered `add`/`sub`/`mul`/`relu`/`sum`/`reshape`/`matmul` and
(Milestone 15) `conv2d`. Milestone 21 fills in the rest of Forge's current
operation surface, using the exact same `time_cpu`/`time_cuda`/`time_calls`
methodology (unchanged -- see **Timing methodology** above) and the exact
same `tiny`/`small`/`medium` scale philosophy (new configs added to
`benchmarks/sizes.py`: `LOSS_CONFIGS`, `ADAM_PARAM_SIZES`, plus reusing
`CONV2D_CONFIGS`'/`ELEMENTWISE_SIZES`' existing scales for `max_pool2d`/
`dropout`):
- **Forward** (`benchmarks/ops_bench.py`): `exp`, `log` (now benchmarked --
  they gained real CUDA kernels in Milestone 14 but were never added to this
  suite), `max_pool2d`, `mse_loss`, `cross_entropy_loss`, `dropout`,
  `adam_step` (an isolated optimizer-step call, not a training loop).
- **Backward** (`benchmarks/backward_bench.py`): `exp`/`log` backward,
  `max_pool2d` backward, `cross_entropy_loss` backward, and a complete
  small-CNN backward pass (`mnist_cnn_full`) -- the real M20 architecture
  (`examples.mnist.model.build_model()`), not a synthetic model.
- **End-to-end MNIST training** (`benchmarks/mnist_bench.py`, new category
  `"mnist"`): the real M20 CNN trained against a fixed synthetic
  `(64, 1, 28, 28)` batch through `CrossEntropyLoss -> Adam` (the same
  optimizer/loss combination `examples/mnist/train.py` uses, unlike the
  pre-existing `training_bench.py`'s `Linear -> ReLU -> Linear` + `MSELoss`
  + `SGD` toy model) -- reports samples/sec and an extrapolated epoch time,
  directly comparable to a real `train.py` run.

### MNIST workload profiling: `benchmarks/mnist_profile.py`
A new, separate diagnostic script (not a `BenchmarkResult`-producing
category -- `python -m benchmarks.mnist_profile`) that breaks one M20 CNN
training step into `transfer -> forward -> loss -> backward -> optimizer`,
and further into per-layer-type forward timing and per-op backward timing.
See that module's docstring for exactly how each breakdown is obtained
(manually unpacking `Sequential.forward()`'s loop for the forward
breakdown; a small instrumented re-implementation of
`forge.autograd.engine.run_backward`, timing each `node.backward_fn(...)`
call by `node.name`, for the backward breakdown). This instrumented walker
adds synchronization overhead between every single backward op on CUDA that
a normal `backward()` call does not pay -- appropriate for a profiling tool
needing per-op resolution, never used on Forge's actual training path.
`tests/test_benchmarks.py::test_profiled_backward_matches_real_backward`
verifies this instrumented walker computes identical gradients to the real
`run_backward` on a small multi-op CPU graph, since a subtly wrong profiler
would produce a misleading bottleneck analysis.

### M21 baseline (before optimization)
Captured with `python -m benchmarks --output benchmarks/results/m21_baseline.json`
(archived; the M11-era baseline is separately preserved as
`benchmarks/results/m11_baseline.json`) and
`python -m benchmarks.mnist_profile --output benchmarks/results/mnist_profile_baseline.json`,
both on the same verified 940MX/i5-7200U environment as the M11 baseline.

MNIST workload profile (batch=64, 30 iterations, 5 warmup):

| Phase | CPU mean | CPU % | CUDA mean | CUDA % |
|---|---|---|---|---|
| transfer | 0.05ms | 0.1% | 0.39ms | 1.1% |
| forward | 19.82ms | 43.8% | 5.98ms | 17.3% |
| loss | 0.22ms | 0.5% | 1.79ms | 5.2% |
| backward | 24.82ms | 54.8% | 25.29ms | 73.4% |
| optimizer | 0.36ms | 0.8% | 1.01ms | 2.9% |
| **TOTAL** | **45.27ms** | | **34.46ms** | |

CUDA forward, by layer type: `Conv2d` 2.51ms, `Linear` 1.35ms, `ReLU` 1.26ms,
`MaxPool2d` 0.58ms, `Flatten` 0.15ms. CUDA backward, by op: **`conv2d`
18.66ms** (73.8% of the entire backward phase), `max_pool2d` 1.34ms, `relu`
1.34ms, `@` (matmul) 0.97ms, everything else under 0.6ms each.

End-to-end MNIST training throughput (`mnist_bench.py`, batch=64):
CPU ~1,060-1,392 samples/sec (56.6-43.1s/epoch extrapolated; this range
reflects real run-to-run noise on this shared, non-dedicated development
machine -- see **Known limitations** below), CUDA ~1,875 samples/sec
(32.0s/epoch extrapolated) at the pre-optimization baseline.

### Bottleneck identification
The MNIST profile makes the dominant CUDA cost unambiguous: `conv2d`
backward alone is 73.8% of the backward phase and 54.2% of the entire
training step -- an order of magnitude larger than every other backward op
combined. Isolating the three `conv2d_backward` sub-kernels
(`cf_conv2d_backward_{input,weight,bias}`) at the CNN's own two layer
shapes (conv1: N=64,Cin=1,Cout=8,H=28,W=28,K=3; conv2:
N=64,Cin=8,Cout=16,H=13,W=13,K=3) narrowed this to the `weight`/`bias`
kernels specifically -- see `docs/architecture/cuda-backend.md`'s **CUDA
Conv2d backward: weight/bias optimization (Milestone 21)** section for the
full per-kernel numbers and the underlying cause (a handful of CUDA threads
each serially reducing tens of thousands of elements alone, while hundreds
of the 940MX's cores sat idle). `MaxPool2d`, elementwise ops, and `Adam`
were also measured and found *not* to justify optimization at this
workload's scale -- see below.

### Optimizations applied
**`conv2d_backward`'s weight/bias kernels** (see the cuda-backend.md section
above for the complete writeup): rewritten as a block-per-output-element
shared-memory tree reduction, with the weight kernel additionally
dispatching between that reduction kernel and the original Milestone 15
one-thread-per-element kernel based on a measured element-count threshold
(a single strategy was *not* uniformly better across both MNIST layer
shapes -- re-measuring after the first attempt is what caught this).

**Everything else measured but not optimized, with the measurement that justified leaving it alone:**
- **`MaxPool2d`.** CPU forward/backward are the single largest CPU-side
  cost in the whole profile (12.23ms forward, 10.42ms backward -- more than
  `Conv2d`'s own CPU cost) but CUDA `max_pool2d` backward is already cheap
  (1.34ms, 5.3% of CUDA backward) -- the milestone's own optimization
  target is CUDA, and CUDA MaxPool2d was not the bottleneck there.
  `CPUBackend.max_pool2d`'s relative CPU cost is a real, measured
  observation (see `docs/architecture/cuda-backend.md`'s **CUDA Conv2d /
  MaxPool2d** section for the underlying im2col/strided-view CPU
  implementation) but optimizing *CPU* MaxPool2d is outside the CUDA
  optimization this milestone's brief scopes -- recorded here as an
  observed fact, not acted on.
- **Elementwise CUDA ops** (`relu`, `exp`, `log`, `+`, `*`, `-`, `sum`,
  `reshape` in the backward-by-op breakdown) collectively account for well
  under 15% of CUDA backward time even before the conv2d fix (and a smaller
  share after it, since `conv2d` shrank and they did not). No repeated-
  kernel-launch fusion was implemented -- the brief explicitly forbids a
  general fusion/graph-compiler system, and nothing here measured large
  enough to justify even a small targeted one.
- **`Adam`.** An isolated CUDA `adam_step` call is sub-millisecond even at
  262,144 parameters (`benchmarks/ops_bench.py`'s new `adam_step` forward
  entries), and the MNIST profile's `optimizer` phase is 2.9% (baseline) of
  total CUDA training-step time. Left completely unchanged, per the
  brief's explicit "if Adam is negligible, leave it alone."
- **Memory transfers.** The MNIST profile's `transfer` phase is 1.1% of the
  CUDA training step (0.39ms out of 34.46ms) at batch=64 -- not remotely
  large enough to justify pinned memory, async prefetching, or a GPU
  DataLoader, all of which the brief explicitly rules out unless transfer
  overhead *dominates*. It does not, here.

### M21 optimized results (after optimization)
Captured with `python -m benchmarks --output benchmarks/results/m21_optimized.json`
(now also `benchmarks/results/latest.json`, per Section 16 of the milestone
brief) and `python -m benchmarks.mnist_profile --output benchmarks/results/mnist_profile_optimized.json`,
same environment, same fixed configuration, no other code changed.

MNIST workload profile (batch=64, 30 iterations, 5 warmup) -- CUDA only
changed (CPU code untouched):

| Phase | CUDA mean (before) | CUDA mean (after) |
|---|---|---|
| transfer | 0.39ms | 0.42ms |
| forward | 5.98ms | 6.37ms |
| loss | 1.79ms | 1.73ms |
| backward | 25.29ms | **16.10ms** |
| optimizer | 1.01ms | 0.91ms |
| **TOTAL** | **34.46ms** | **25.53ms** |

CUDA backward, `conv2d` specifically: 18.66ms -> **9.58ms** (~1.95x). Every
other backward op's own time is within normal run-to-run noise of its
baseline value (none were touched).

Isolated `conv2d_backward` kernel timing (direct `ctypes` calls, bypassing
Tensor/autograd dispatch overhead -- the cleanest before/after comparison
since it isolates exactly the changed code):

| Layer shape | Before | After | Speedup |
|---|---|---|---|
| conv1 (weight_elems=72) | 12.62ms | 3.64-3.73ms | ~3.4-3.5x |
| conv2 (weight_elems=1,152) | 7.20ms | 6.71-10.34ms | ~1.0-1.1x (bias-only gain at this shape; see hybrid dispatch above) |

End-to-end MNIST training throughput (`mnist_bench.py`, batch=64): CUDA
~1,875 -> **~2,569 samples/sec** (~1.37x), extrapolated epoch time
~32.0s -> ~23.4s. Two repeat `mnist` benchmark runs after the optimization
were consistent (CUDA ~23.2-24.9ms/step, CPU ~46.0-47.3ms/step), confirming
the baseline run's noisier CPU number (60.37ms, stdev 33.7ms -- likely
transient background load on this shared development machine, not a code
effect) was the outlier, not the post-optimization numbers.

### Summary table (Section 14 format)
```text
CPU:
  samples/sec:  ~1,060-1,392 (baseline) / ~1,392-1,517 (post-optimization runs)
                (CPU code unchanged throughout; range reflects measured
                 run-to-run system noise, not a code effect -- see below)
  epoch time:   ~43-57s (extrapolated from measured per-batch rate)

CUDA before:
  samples/sec:  ~1,875
  epoch time:   ~32.0s (extrapolated)

CUDA after:
  samples/sec:  ~2,569
  epoch time:   ~23.4s (extrapolated)
```
The primary success metric -- improved end-to-end CUDA throughput -- moved
by ~1.37x from a single, measurement-justified kernel change. CUDA training
remains slower than CPU is capable of at full efficiency on this small
architecture/batch size (consistent with `docs/architecture/cuda-backend.md`'s
long-standing "no performance claims" position for small workloads on this
GPU), but the CUDA-vs-CPU comparison was never this optimization's target;
the CUDA-vs-CUDA-before improvement is.

### Known limitations (Milestone 21 additions)
- **CPU throughput numbers show real run-to-run variance on this
  development machine** (a shared laptop, not a dedicated benchmark host)
  independent of any code change -- the baseline run's CPU MNIST-training
  number (60.37ms/step, stdev 33.7ms) was materially noisier than three
  later measurements of the *same, unmodified* CPU code path (45.96ms,
  46.50ms, 47.34ms/step). This is reported honestly rather than smoothed
  over: the CUDA-side numbers (which this milestone's optimization actually
  touched) were comparatively stable throughout (stdev 4-6ms on a ~25-34ms
  mean), and the isolated `ctypes`-level kernel measurements (least exposed
  to system noise, since they bypass the most Python-level dispatch) are
  the most trustworthy single before/after comparison in this section.
- **The weight-kernel dispatch threshold (256 weight elements) was chosen
  from two measured data points** (72 and 1,152), not a sweep across many
  layer shapes -- it is a reasonable, conservative midpoint for the
  architectures Forge's current scope targets (small CNNs, few-to-moderate
  channel counts), not a tuned/autotuned constant. A future Conv2d layer
  shape near that boundary might not realize the full measured speedup, or
  might pick the less-optimal kernel; this is documented as a known
  approximation, not hidden.
- **No further Conv2d optimization was attempted beyond weight/bias.** The
  `input` kernel (already one thread per input pixel at every measured
  shape) was left unchanged since it was not the measured bottleneck --
  see `docs/architecture/cuda-backend.md`'s writeup for the reasoning.
  Shared-memory input/weight tiling, im2col, and cuDNN remain explicitly
  out of scope per the milestone brief.
- **`benchmarks/mnist_profile.py`'s instrumented backward walker adds
  synchronization overhead** between every backward op that a normal
  `backward()` call does not pay (see that module's docstring) -- its
  absolute per-op numbers are therefore not directly comparable to
  `backward_bench.py`'s `time_calls`-based numbers for the same op in
  isolation; both are internally consistent for their own before/after
  comparisons, which is what this milestone's analysis relies on.

## Milestone 22: CUDA memory-stats reporting

### Purpose
Milestone 22 is not another optimization pass -- it adds CUDA memory
observability (`forge.cuda.memory_stats()`, see
`docs/architecture/cuda-backend.md`'s **CUDA Memory Statistics** section)
and extends the existing benchmark subsystem to report it *alongside* the
established timing methodology, never in place of it.

### New: `benchmarks/memory.py`
One pure function, `cuda_memory_extra(before, after)`, turning a
before/after pair of `CUDAMemoryStats` snapshots into a plain dict:
`cuda_allocated_before_bytes`, `cuda_peak_allocated_bytes`,
`cuda_allocated_after_bytes`, `cuda_allocation_count_delta`,
`cuda_free_count_delta`. Merged into `BenchmarkResult.extra` (Milestone 11's
existing free-form field) for CUDA-device results only -- a CPU
`BenchmarkResult`'s `extra` is completely unaffected, and existing JSON
consumers that only read the established `BenchmarkResult` fields see no
schema change. `tests/test_benchmarks.py::
test_cuda_memory_extra_reports_expected_keys_and_deltas` is the harness-
mechanics test, per this project's "no timing thresholds in the normal
test suite" rule -- it checks the dict-building math with synthetic
`CUDAMemoryStats` values, never a real CUDA workload.

### Wired into `training_bench.py` and `mnist_bench.py`
Both call, once, immediately before and after their existing timed loop
(never inside it, so per-iteration timing is unaffected by the two extra
calls):
```python
gc.collect()
forge.cuda.reset_peak_memory_stats()
mem_before = forge.cuda.memory_stats()
# ... existing timed loop, unchanged ...
gc.collect()
extra.update(cuda_memory_extra(mem_before, forge.cuda.memory_stats()))
```
The `gc.collect()` calls are not decorative: see
`docs/architecture/cuda-backend.md`'s **Known limitations** -- Forge's
Tensor/autograd/Module/Optimizer object graph for a full training step
contains genuine Python reference cycles, so an `allocated_bytes` snapshot
taken without an intervening `gc.collect()` can substantially overstate
true live CUDA memory (confirmed by first wiring this up *without* the
`gc.collect()` calls: the small-MLP `training_bench.py` case reported a
misleading ~2.7MB apparent growth over 50 SGD iterations, which a single
`gc.collect()` before each snapshot reduced to a few KB of residual --
itself smaller than one training batch's own tensor footprint).

### Measured example (940MX, real hardware)
`python -m benchmarks --categories training mnist`, one representative run:

| Operation | `cuda_allocated_before_bytes` | `cuda_peak_allocated_bytes` | `cuda_allocated_after_bytes` | alloc/free count delta |
|---|---|---|---|---|
| `zero_grad_forward_loss_backward_step` (small MLP, SGD) | 95,824 | 3,420,048 | 98,388 | 1450 / 1448 |
| `mnist_cnn_zero_grad_forward_loss_backward_step` (M20 CNN, Adam) | 440,992 | 69,014,688 | 440,992 | 1920 / 1920 |

Both cases return to (near-)their starting `allocated_bytes` after their
timed loop -- the small residual in the first row (2,564 bytes, ~2.6% of
the steady-state footprint) reflects the same GC-timing sensitivity
documented above (a `gc.collect()` catches the large majority but is not
guaranteed to reclaim every cyclic object in a single pass), not a growing
leak; `mnist_cnn_...`'s exact-zero delta on the same measurement machinery
shows the effect is bounded, not systematic. `cuda_peak_allocated_bytes`
correctly captures the much larger transient footprint mid-training (e.g.
69MB for the CNN, vs. 441KB before/after) that the before/after numbers
alone would miss entirely -- exactly the gap `reset_peak_memory_stats()`
plus peak tracking exists to close.

### Performance overhead
Measured two ways, per the milestone's "must not materially slow down
normal CUDA execution" constraint:
1. **Isolated accounting cost**: `timeit`-measuring 200,000
   `record_alloc()`+`record_free()` call pairs directly gives ~2.45us per
   pair (~1.2us per call) -- almost entirely Python `threading.Lock`
   acquisition overhead, not the counter arithmetic itself.
2. **Same-process instrumented-vs-no-op A/B**: monkeypatching
   `record_alloc`/`record_free` to no-ops and timing a warmed-up, 500-
   iteration tight loop of CUDA `add` (~118us/op on this hardware,
   including WDDM driver call and kernel-launch overhead) against the same
   loop with real accounting active showed **no measurable difference**
   (the instrumented case was fractionally *faster*, within run-to-run
   noise, on repeated trials) -- consistent with (1): ~1.2us against a
   ~118us baseline is below this measurement's noise floor. A naive
   cross-process comparison (run the full benchmark suite once with
   instrumentation code physically removed via `git stash`, once with it
   present) was attempted first and discarded: successive full-process CUDA
   benchmark runs on this laptop GPU (940MX, WDDM) showed 2-10x swings
   attributable to GPU clock/thermal warm-up state alone, which completely
   dominates and hides an effect this small -- reported here as a
   methodology finding in its own right, not papered over with a
   misleadingly precise cross-process percentage.

## Milestone 24: CUDA allocation profiling and caching-allocator design

### Purpose
Not another optimization pass, and not another timing benchmark: Milestone
24 asks whether Forge's direct `cudaMalloc`/`cudaFree`-per-`CUDAStorage`
model is a real bottleneck, using a new optional allocation-*event* profiler
(`forge.cuda.profiler`, distinct from Milestone 22's current/peak
`memory_stats()`) layered on top of the existing timing methodology. See
`docs/architecture/cuda-memory-allocator.md` for the full writeup this
section summarizes.

### New: `benchmarks/alloc_profile.py` / `benchmarks/alloc_analysis.py`
`alloc_profile.py` is a diagnostic script (like `mnist_profile.py`,
Milestone 21) -- not part of the stable `BenchmarkResult` JSON schema. It
profiles: the real M20 MNIST workload (warmup vs. steady-state, phases
tagged via `forge.cuda.profiler.tag()`), sixteen representative operations'
forward and backward allocation traffic, CPU<->CUDA transfer allocation
behavior, and direct `cudaMalloc`/`cudaFree` host-API timing.
`alloc_analysis.py` holds the pure-function analysis this data feeds into:
size/lifetime distributions, persistent-vs-temporary classification, a
same-size reuse-opportunity statistic, and an **offline** caching-allocator
simulation (never wired into Forge's real allocator).

### Measured example (940MX, real hardware; batch=64, 30 steady-state iterations)

| Metric | Value |
|---|---:|
| Mean full-iteration wall-clock time | 25.21 ms |
| Allocations / frees per iteration | 64.0 / 64.0 |
| True persistent CUDA memory (`memory_stats()`, flat before/after) | 440,992 bytes |
| Peak allocated bytes (any instant) | 7,789,424 bytes |
| Distinct allocation sizes across the whole trace | 14 |
| Bytes that are an exact-size repeat of an earlier allocation | 99.1% |
| Offline exact-size cache simulation: real `cudaMalloc` calls needed | 42 (vs. 1,920 today) |
| Direct `cudaMalloc` / `cudaFree` cost (isolated, size-independent) | ~175-205 us / ~245-295 us |

The last row is the headline finding: `cudaMalloc`/`cudaFree` are
host-blocking CUDA Runtime API calls (no asynchronous queue to correct for,
unlike a kernel launch), and on this machine's WDDM driver stack they cost
roughly 175-300 microseconds *per call*, essentially independent of request
size across 4 KB-1 MB. Multiplied by 64 allocations + 64 frees/iteration,
that is the same order of magnitude as the entire ~25 ms measured training
step -- explicitly reported as an order-of-magnitude estimate, not a
precise attribution (isolated timing has no concurrent kernel traffic to
overlap with, unlike the real workload), per this document's and the
milestone brief's own caution against overclaiming allocation-latency
numbers.

### Decision
`docs/architecture/cuda-memory-allocator.md`'s **Decision** section:
**caching allocator JUSTIFIED** (evidence-backed recommendation for a
future milestone; not implemented in Milestone 24). The combination of
near-total same-size repetition, a trivial offline simulation collapsing
1,920 real driver calls to 42, and this environment's unusually high
per-call `cudaMalloc`/`cudaFree` cost together make a strong case -- but see
that document's **Conditions under which caching should be implemented**
and **Risks** sections before treating this as a mandate.

### Performance overhead of the profiler itself
Identical mechanism and identical conclusion to Milestone 22's own
counters (see immediately above): `CUDAMemoryProfiler.record()`'s
disabled-path cost is one `bool` check, and `memory_stats()`'s own numbers
are unaffected by whether the profiler happens to be running
(`tests/test_cuda_alloc_profiler.py::
test_profiler_running_does_not_change_real_memory_stats`). No new
synchronization was added at either instrumentation point.
