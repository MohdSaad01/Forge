# M35 -- CUDA Performance Characterization & Roofline-Style Analysis

Reproduce with:
```bash
python -m benchmarks.m35_hardware
python -m benchmarks.m35_kernels
python -m benchmarks.m35_transfer_stream_alloc
python -m benchmarks.m35_mnist
python -m benchmarks.m35_report
```

## Hardware
NVIDIA GeForce 940MX (GM108, Maxwell), Compute Capability 5.0, CUDA 12.6, driver 582.53.

## Practical ceilings (measured, not theoretical)
- Compute: **104.57 GFLOP/s** (`cf_matmul_f32`, large square GEMM)
- Bandwidth: **15.09 GB/s** (`cf_add_f32`, large streaming add)
- Theoretical FP32 peak: 953.1 GFLOP/s (public spec + measured clocks -- **not** an achievable target)
- Theoretical bandwidth: 16.02 GB/s (same caveat)

## Kernel runtime ranking (real MNIST training step, Section 26)

| Op | % of step | GFLOP/s | GB/s | AI | Classification |
|---|---|---|---|---|---|
| backward:conv2d | 50.97% | 8.415 | 0.728 | 11.559 | mixed_or_ambiguous |
| forward:Conv2d | 14.09% | 13.037 | 1.317 | 9.899 | mixed_or_ambiguous |
| backward:@ | 7.20% | 7.118 | 0.514 | 13.840 | mixed_or_ambiguous |
| backward:relu | 5.44% | 0.000 | 0.000 | 0.000 | mixed_or_ambiguous |
| backward:max_pool2d | 5.29% | 0.000 | 6.074 | 0.000 | memory_bandwidth_bound |
| forward:Linear | 4.79% | 5.355 | 0.387 | 13.840 | mixed_or_ambiguous |
| forward:ReLU | 4.48% | 0.000 | 0.000 | 0.000 | mixed_or_ambiguous |
| forward:MaxPool2d | 3.23% | 0.000 | 5.292 | 0.000 | memory_bandwidth_bound |
| backward:+ | 1.82% | 0.000 | 0.000 | 0.000 | mixed_or_ambiguous |
| forward:Flatten | 1.02% | 0.000 | 0.000 | 0.000 | mixed_or_ambiguous |

## Optimization headroom ranking (Section 29)
`headroom_score = runtime_fraction * (1 - fraction_of_practical_ceiling)` -- an explicit proxy, not exact recoverable time.

| Op | % of step | Distance from ceiling | Headroom score |
|---|---|---|---|
| backward:conv2d | 50.97% | 8.0% | 0.4687 |
| forward:Conv2d | 14.09% | 12.5% | 0.1233 |
| backward:@ | 7.20% | 6.8% | 0.0671 |
| backward:relu | 5.44% | 0.0% | 0.0544 |
| forward:Linear | 4.79% | 5.1% | 0.0454 |
| forward:ReLU | 4.48% | 0.0% | 0.0448 |
| backward:max_pool2d | 5.29% | 40.3% | 0.0316 |
| forward:MaxPool2d | 3.23% | 35.1% | 0.0210 |

## Amdahl analysis (Section 30, hypothetical)

Top contributor: **backward:conv2d** at 51.0% of the CUDA training step.

| Hypothetical per-kernel speedup | Hypothetical overall speedup |
|---|---|
| 1.5x | 1.205x |
| 2.0x | 1.342x |
| 3.0x | 1.515x |

## M34 256-1152 weight-element threshold region (Section 31)

| Shape | weight_elements | in untested region | im2col+GEMM speedup vs. direct | memory overhead (MB) |
|---|---|---|---|---|
| mnist_conv1 | 72 | False | 1.81x | 3.25 |
| mnist_conv2 | 1152 | False | 0.88x | 3.63 |
| large_channel | 4608 | False | 0.31x | 33.69 |
| large_spatial | 1152 | False | 0.73x | 33.69 |
| batch_32 | 4608 | False | 0.31x | 16.84 |
| batch_64 | 4608 | False | 0.32x | 33.69 |
| batch_128 | 4608 | False | 0.31x | 67.38 |
| thresh_288 | 288 | True | 0.60x | 8.42 |
| thresh_576 | 576 | True | 0.81x | 15.31 |
| thresh_864 | 864 | True | 0.79x | 16.08 |
| thresh_1008 | 1008 | True | 0.76x | 16.46 |

Speedup is `total_experimental_time / current_direct_time` -- **below 1.0 means im2col+GEMM is faster**. The threshold constant (`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD = 256` in `backend.py`) is **not** changed by this milestone (Section 31).

## Batch-size scaling (Section 33)

| Batch | samples/sec | compute-stream utilization |
|---|---|---|
| 32 | 1587 | 77.8% |
| 64 | 2780 | 81.2% |
| 128 | 3983 | 74.9% |

## Profiling overhead (Section 37)

Plain step: 15.1346ms. Instrumented step: 14.5923ms. Overhead: -3.6% (within run-to-run noise -- instrumentation is not materially altering the numbers reported).

## Plots

- `benchmarks\results\m35_plots\roofline.png`
- `benchmarks\results\m35_plots\kernel_contribution.png`
- `benchmarks\results\m35_plots\gemm_scaling.png`

## Milestone 36 update: dInput roofline reclassification

M36 optimized `dInput` (see `docs/performance/conv2d-backward-profiling.md`'s
**Milestone 36** section for the full investigation). Re-running this
document's methodology after that change:

| Shape | AI | dInput before (GFLOP/s, % ceiling, class) | dInput after (GFLOP/s, % ceiling, class) |
|---|---|---|---|
| mnist_conv1 | 4.00 | 10.02, 16.6%, mixed | 8.94, 14.8%, mixed |
| mnist_conv2 | 23.89 | 12.32, 11.8%, mixed | 29.04, 27.8%, mixed |
| large_channel | 47.91 | 12.82, 12.3%, mixed | 35.26, 33.7%, **compute_bound** |
| large_spatial | 23.99 | 12.05, 11.5%, mixed | 29.88, 28.6%, mixed |
| batch_32 | 47.82 | 12.27, 11.7%, mixed | 33.13, 31.7%, **compute_bound** |
| batch_64 | 47.91 | 12.65, 12.1%, mixed | 33.81, 32.3%, **compute_bound** |
| batch_128 | 47.95 | 12.66, 12.1%, mixed | 35.78, 34.2%, **compute_bound** |

Four of seven shapes cross `benchmarks/roofline.py`'s 30%-of-ceiling
`NEAR_CEILING_FRACTION` threshold and reclassify from `mixed_or_ambiguous` to
`compute_bound` -- consistent with the root cause being instruction/register
efficiency (local-memory traffic removed, not a bandwidth-reuse fix): the
kernel moved *toward* its compute ceiling, not toward the bandwidth ceiling,
exactly as expected for a fix that eliminates local-memory reads and keeps
register pressure near baseline rather than reducing global-memory traffic.
`mnist_conv1` (`Cin=1`) shows no reclassification and a small GFLOP/s
regression -- expected, since Candidate B's benefit requires `Cin > 1`.

**Updated bottleneck ranking** (`python -m benchmarks.m35_mnist`, real MNIST
training step, batch=64):

| Op | % of step (M35) | % of step (M36) |
|---|---|---|
| `backward:conv2d` | 50.97% | 46.82% |
| `forward:Conv2d` | 14.09% | 15.68% |
| `backward:@` | 7.20% | 6.77% |

`backward:conv2d` remains the single largest contributor after M36 (dWeight
is unchanged and still dominates the combined `dInput+dWeight` kernel-launch
time at most shapes -- see the profiling doc's before/after table), but its
share of the step dropped by ~4 percentage points. **Amdahl check**: dInput
alone was roughly half of `backward:conv2d`'s time pre-M36 at the
representative shapes (`d_input_ms` vs. `d_weight_ms` in `conv2d_backward_
profile.json`), i.e. roughly 25% of the full step; a ~2.5x average dInput
speedup at that share predicts an overall step speedup of
`1 / (0.75 + 0.25/2.5) = 1.18x` -- in the same direction and rough range as
the ~1.09x implied by the measured 50.97% -> 46.82% shift (`0.5097/0.4682 x
(1/step_before) `, not a precise match since the two measurements come from
different sessions with the hardware variance documented in the profiling
doc's **Hardware variance** section, but consistent enough to confirm no
surprising interaction effect). `dWeight` (im2col+GEMM, still M34-optimized
and untouched) is now the more clearly dominant of the two `conv2d_backward`
kernels at most shapes -- the natural M37+ candidate if further Conv2d
backward optimization is pursued.
